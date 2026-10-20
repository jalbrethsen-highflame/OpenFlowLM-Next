r"""mbe: moe_batch with eb's core program -- the routed experts over a block of tokens, MB_NT
token slots per expert visit, sized for MB_NT = 32 on the bfp16 MAC array.

moe_batch (moe_batch.py) holds the whole expert hidden h in the B fifo for the down, and the
up and gate accumulators of both of the core's bands at once; that fills L1 at MB_NT = 16.
At MB_NT = 8 it is already bound by the weight stream (16 slots: 0.895 ms against 0.779 for
the stream alone), so the only way down is more tokens per streaming of an expert's 1.97 MB.
eb (designs/lfm2moe/eb.py, LFM2-MoE) solved the same L1 problem for a wider expert: up and
gate band by band ([u 64 | g 64] live), h streamed again k-tile by k-tile for each output
band. This is that core program over moe_batch's weight taps, so it reads the Qwen3.6-MoE
pool exactly as packed (two bands of a 128-row stripe interleaved at k-group granularity) and
writes x / h / y in moe_batch's layouts: the host driver (Core::moe_block) does not change,
only the manifest's moe_batch.nt.

Per core (column c, row r) and slot: for each half (band) of row r's stripe: up then gate,
NBG_UP k-groups each, x streamed once per (half, proj); silu -> that band's 64 h rows. Then
for each of the core's NRB_DN output bands (band 4 rbg + r of the down): the band's NBG_DN
k-groups against h streamed again, -> y.

The core kernels are eb's (designs/lfm2moe/eb_*.cc, on moe_batch.h's arithmetic); MB_BFP=1
takes eb_step_bfp (the bfp16 MAC array, round to nearest even).

Build: MB_SLOTS=256 MB_NT=32 MB_BFP=1 python build_design.py designs/moe_batch/mbe.py designs/moe_batch/build_mbe_s256
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
EB = HERE.parent / "lfm2moe"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N_ROWS, N_COLS = 4, 8
NT = int(os.environ.get("MB_NT", 32))
BAND = 10240                # one band-k-group: 64 rows x 256 K of q4_1
K_TILE = 64

# ---- the model's expert geometry (env, the 35B's by default) and the slot count: moe_batch.py's
HID = int(os.environ.get("MB_HID", 2048))
FF = int(os.environ.get("MB_FF", 512))
SLOTS = int(os.environ.get("MB_SLOTS", 256))
EXPERTS = int(os.environ.get("MB_EXPERTS", 256))
STRIPE = 128 * HID * 5 // 8
SPP = FF // 128
EXP_UG = 2 * SPP * STRIPE
DOWN_BAND = 128 * FF * 5 // 8
DOWN_BYTES = HID * FF * 5 // 8
POOL_DOWN = int(os.environ.get("MB_POOL_DOWN", EXPERTS * EXP_UG))
POOL_BYTES = int(os.environ.get("MB_POOL_BYTES", 512 << 20))
BFP = os.environ.get("MB_BFP") == "1"
X_BYTES = SLOTS * HID * NT * 2
H_BYTES = SLOTS * FF * NT * 2
Y_BYTES = SLOTS * HID * NT * 4
NBG_UP, NBG_DN = HID // 256, FF // 256
NRB_DN = HID // (64 * N_ROWS)
B_DEPTH = 2 if BFP else 4          # the bfp step's L1 scratch comes out of the x fifo (eb's trade)
C_DEPTH = 1
L1 = ((64 + NT) * 72 * BFP + 0x1000 + 2 * BAND + 128 * NT * 4 + C_DEPTH * 64 * NT * 4
      + B_DEPTH * K_TILE * NT * 2)

assert NT % 8 == 0 and NT >= 8, "NT is a whole number of 8-token MAC tiles"
assert SPP == N_ROWS, "one 128-row stripe of up (and of gate) per row: FF = 512"
assert HID % (64 * N_ROWS) == 0 and HID % 256 == 0 and FF % 256 == 0
assert SLOTS % N_COLS == 0 and SLOTS <= EXPERTS
assert POOL_DOWN + EXPERTS * DOWN_BYTES <= POOL_BYTES
assert L1 <= 64 << 10, f"mbe.py: NT={NT} wants {L1} B of a core's 65536"


def lin(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((total,), off, [1, n], [n, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mbe(pool: In, x: In, h: Out, y: Out, *, slots: CompileTime[int], srchash: CompileTime[int] = 0):
    band_ty = np.ndarray[(BAND,), np.dtype[np.uint8]]
    a4_ty = np.ndarray[(N_ROWS * BAND,), np.dtype[np.uint8]]
    b_ty = np.ndarray[(K_TILE * NT,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(64 * NT,), np.dtype[np.float32]]
    c4_ty = np.ndarray[(N_ROWS * 64 * NT,), np.dtype[np.float32]]
    ug_ty = np.ndarray[(128 * NT,), np.dtype[np.float32]]      # [u 64 rows | g 64 rows] of one band
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(X_BYTES,), np.dtype[np.uint8]]
    h_ty = np.ndarray[(H_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(Y_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(HERE)]
    flags = [f"-DMB_NT={NT}"] + [f"-D{v}" for v in ("MB_NULL_MM", "MB_NULL_DQ") if os.environ.get(v) == "1"]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(EB / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)

    step = ef("eb_step_bfp" if BFP else "eb_step", [band_ty, b_ty, ug_ty, np.int32, np.int32])
    zero_ug = ef("eb_zero_ug2", [ug_ty])
    zero1 = ef("eb_zero1", [ug_ty])
    silu = ef("eb_silu2", [ug_ty, c_ty, np.int32])
    outk = ef("eb_out", [ug_ty, c_ty])

    A_l3l2 = [ObjectFifo(a4_ty, name=f"A{c}", depth=2) for c in range(N_COLS)]
    A_l2l1 = [A_l3l2[c].cons().split([r * BAND for r in range(N_ROWS)], obj_types=[band_ty] * N_ROWS,
                                     names=[f"A{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS, tile=Tile(c, 1))
              for c in range(N_COLS)]
    B = [ObjectFifo(b_ty, name=f"B{c}", depth=B_DEPTH) for c in range(N_COLS)]
    C_l2l3 = [ObjectFifo(c4_ty, name=f"C{c}", depth=2) for c in range(N_COLS)]
    C_l1l2 = [C_l2l3[c].prod().join([r * 64 * NT for r in range(N_ROWS)], obj_types=[c_ty] * N_ROWS,
                                    names=[f"C{c}_{r}" for r in range(N_ROWS)], depths=[C_DEPTH] * N_ROWS,
                                    tile=Tile(c, 1))
              for c in range(N_COLS)]
    ugbuf = [[Buffer(ug_ty, name=f"ug_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]

    # Every loop a hardware loop (range_), as eb.py: unrolled copies overflow program memory.
    def core_fn(in_a, in_b, out_c, k_step, k_zero_ug, k_zero1, k_silu, k_out, ug):
        he = out_c.acquire(1)                        # the stripe's 128 h rows, bf16, natural order
        for half in range_(2):                       # band 0, then band 1 of the core's stripe
            k_zero_ug(ug)
            for proj in range_(2):                   # up, gate
                for _ in range_(NBG_UP):
                    band = in_a.acquire(1)
                    for ky in range_(4):
                        xe = in_b.acquire(1)
                        k_step(band, xe, ug, ky, proj)
                        in_b.release(1)
                    in_a.release(1)
            k_silu(ug, he, half)                     # this band's 64 h rows
        out_c.release(1)
        for _ in range_(NRB_DN):                     # this core's output bands
            k_zero1(ug)
            for _ in range_(NBG_DN):
                band = in_a.acquire(1)
                for ky in range_(4):
                    hb = in_b.acquire(1)
                    k_step(band, hb, ug, ky, 0)
                    in_b.release(1)
                in_a.release(1)
            ye = out_c.acquire(1)
            k_out(ug, ye)
            out_c.release(1)

    workers = [Worker(core_fn,
                      [A_l2l1[c][r].cons(), B[c].cons(), C_l1l2[c][r].prod(), step, zero_ug, zero1, silu, outk,
                       ugbuf[r][c]],
                      tile=Tile(c, 2 + r), stack_size=0x1000)
               for c in range(N_COLS) for r in range(N_ROWS)]

    # moe_batch.py's taps: up / gate, row r owns stripe r and `half` is the band inside it; down,
    # row r owns band 4 rbg + r. One element = the four rows' bands at one k-group.
    def up_tap(slot: int, proj: int, half: int) -> TensorAccessPattern:
        off = slot * EXP_UG + proj * STRIPE + half * BAND
        return TensorAccessPattern((POOL_BYTES,), off, [NBG_UP, N_ROWS, 4, 2560], [2 * BAND, 2 * STRIPE, 2560, 1])

    def down_tap(slot: int, rbg: int) -> TensorAccessPattern:
        off = POOL_DOWN + slot * DOWN_BYTES + rbg * 2 * DOWN_BAND
        return TensorAccessPattern((POOL_BYTES,), off, [NBG_DN, 2, 8, 2560], [2 * BAND, DOWN_BAND, 2560, 1])

    A_prods = [A_l3l2[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    B_prods = [B[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    C_conss = [C_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]
    XS, HS, YS = HID * NT * 2, FF * NT * 2, HID * NT * 4      # one slot's x, h, y

    def sequence(a_pool, a_x, a_h, a_y, A_hs, B_hs, C_hs):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        for rnd in range(slots // N_COLS):
            slot = [rnd * N_COLS + c for c in range(N_COLS)]
            # drains before fills (npu-model-porting): the h drain is issued before its fills
            for c in range(N_COLS):
                py.drain(C_hs[c], a_h, lin(H_BYTES, slot[c] * HS, HS))
            for half in range(2):
                for proj in range(2):
                    for c in range(N_COLS):
                        pw.fill(A_hs[c], a_pool, up_tap(slot[c], proj, half))
                        px.fill(B_hs[c], a_x, lin(X_BYTES, slot[c] * XS, XS))
            for c in range(N_COLS):
                py.finish(C_hs[c])                   # h is in DDR
            for c in range(N_COLS):
                py.drain(C_hs[c], a_y, lin(Y_BYTES, slot[c] * YS, YS))
            for rbg in range(NRB_DN):
                for c in range(N_COLS):
                    pw.fill(A_hs[c], a_pool, down_tap(slot[c], rbg))
                    px.fill(B_hs[c], a_h, lin(H_BYTES, slot[c] * HS, HS))
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, x_ty, h_ty, y_ty, A_prods, B_prods, C_conss])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = mbe
_src = b"".join([(HERE / "mbe.py").read_bytes()] + sorted(f.read_bytes() for f in EB.glob("eb_*.cc"))
                + [(HERE / "moe_batch.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes()]
                + [f"{k}={v}".encode() for k, v in (("NT", NT), ("HID", HID), ("FF", FF), ("EXPERTS", EXPERTS),
                                                     ("POOL_DOWN", POOL_DOWN), ("POOL_BYTES", POOL_BYTES))]
                + [f"{v}=1".encode() for v in ("MB_NULL_MM", "MB_NULL_DQ", "MB_BFP") if os.environ.get(v) == "1"])
SPECIALIZE = {"slots": SLOTS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}

if __name__ == "__main__":
    print(f"HID={HID} FF={FF} NT={NT} BFP={BFP} L1={L1} SLOTS={SLOTS} EXPERTS={EXPERTS} POOL_DOWN={POOL_DOWN} "
          f"POOL_BYTES={POOL_BYTES} x={X_BYTES} h={H_BYTES} y={Y_BYTES}")
