r"""mbi: the routed experts over a block of tokens on the int8 x int4 MAC mode, one expert per
column and the column's four rows splitting its TOKENS (MBI_NT = 16 each, 64 a slot): the weights
stream once per slot and are broadcast from the mem tile to the four rows, and the x / h traffic
around them is int8.

The pool's experts are transformed at load (Core, OFLM_OPEN_MBI): nibbles XOR 0x88 (signed q - 8),
each chunk's d and m as int16 d 2^S and (m + 8 d) 2^S, one S per expert matrix (mbi_core.h, v3).

Per column and slot (expert e):
  weights  shim -> mem tile -> the 4 rows (broadcast), one 64-row band k-group (10240 B) an element:
           up / gate band pb (8 of them): up kg 0..7, gate kg 0..7 (one task a band); then the down
           in the pool's order (one task): per 128-row stripe, kg 0..1, its two 64-row bands each.
  x        shim -> mem tile, the slot's [row 4][kg 8][MBI_XEL3] in one object; split by row and
           replayed 16 times (each up / gate band reads it again), consumed one k-group element at a
           time. Header int32 [S_ug, S_dn, real tokens of the row]; xa int8, qsum int16, scale bf16.
  core     u, g accumulators per band; h = silu(g) u (bf16 tanh form) quantized per (token, band)
           into hq (the down's A tiles, L1); the down over each k-group with the band scales folded
           into d and mc; y as bf16. Token-group pairs past the row's real tokens are skipped.
  y        the rows join at the mem tile and stream out token-major: [token 64][2048] bf16 a slot.

256 slots of 64 tokens (16384 token visits): 13.3 ms with the GPU idle (DMA floor 10.5 ms), ~21 ms
with a GPU job running (the NPU cores slow ~1.6x then, so the core steps bound it); mbe 16.0 ms for
8192 visits. Check: mbi_test.py gen DIR 256 [rand] -> run -> mbi_test.py check DIR (rel ~1.7 %).

Build: MBI_SLOTS=256 MBI_EXPERTS=256 python build_design.py designs/moe_batch/mbi.py <dir>
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
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N_ROWS, N_COLS = 4, 8
NT = int(os.environ.get("MBI_NT", 16))           # tokens a core
NTG = NT // 4
HID = int(os.environ.get("MBI_HID", 2048))
FF = int(os.environ.get("MBI_FF", 512))
SLOTS = int(os.environ.get("MBI_SLOTS", 8))
EXPERTS = int(os.environ.get("MBI_EXPERTS", 256))
BAND = 10240                                     # one 64-row band k-group: two q4_1 chunks
STRIPE = 128 * HID * 5 // 8
EXP_UG = 2 * (FF // 128) * STRIPE
DOWN_BAND = 128 * FF * 5 // 8
DOWN_BYTES = HID * FF * 5 // 8
POOL_DOWN = int(os.environ.get("MBI_POOL_DOWN", EXPERTS * EXP_UG))
POOL_BYTES = int(os.environ.get("MBI_POOL_BYTES", POOL_DOWN + EXPERTS * DOWN_BYTES))
NPB, KG_UP, NOB, KG_DN = FF // 64, HID // 256, HID // 64, FF // 256

XHDR, XA, QS, SX = 64, 16 * NTG * 64, 8 * NTG * 128, NTG * 128
XEL = XHDR + XA + QS + SX                        # one k-group of one row's tokens
XROW = KG_UP * XEL
XSLOT = N_ROWS * XROW
ACC = 4 * NTG * 64                               # floats: [4 row groups][NTG][64]
HQ = 2 * 16 * NTG * 64
HMETA = 64 + 8 * NT * 4 + 16 * NT * 2 + 2 * NT * 2
CB = ACC * 2                                     # one core's y band, bf16
YSLOT = NOB * N_ROWS * CB
X_BYTES, Y_BYTES = SLOTS * XSLOT, SLOTS * YSLOT
STACK = int(os.environ.get("MBI_STACK", "0x1000"), 0)
L1 = 2 * BAND + 2 * XEL + CB + 2 * ACC * 4 + HQ + HMETA + STACK

assert NT == 16, "mbi_core.h is written for 16 tokens a core (four token groups, two a pass)"
assert FF % 128 == 0 and HID % 256 == 0 and FF % 256 == 0 and FF <= 512, "the h buffer holds two down k-groups"
assert SLOTS % N_COLS == 0 and SLOTS <= EXPERTS
assert POOL_DOWN + EXPERTS * DOWN_BYTES <= POOL_BYTES
assert L1 <= 64 << 10, f"mbi.py: a core wants {L1} B of L1"


def lin(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((total,), off, [1, n], [n, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mbi(pool: In, x: In, y: Out, *, slots: CompileTime[int], srchash: CompileTime[int] = 0):
    band_ty = np.ndarray[(BAND,), np.dtype[np.uint8]]
    xel_ty = np.ndarray[(XEL,), np.dtype[np.uint8]]
    xrow_ty = np.ndarray[(XROW,), np.dtype[np.uint8]]
    xslot_ty = np.ndarray[(XSLOT,), np.dtype[np.uint8]]
    c_ty = np.ndarray[(ACC,), np.dtype[bfloat16]]
    c4_ty = np.ndarray[(N_ROWS * ACC,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(ACC,), np.dtype[np.float32]]
    hq_ty = np.ndarray[(HQ,), np.dtype[np.int8]]
    hm_ty = np.ndarray[(HMETA,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(X_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(Y_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(HERE)]
    flags = [f"-DMBI_NT={NT}"] + [f"-D{d}" for d in os.environ.get("MBI_DEFS", "").split(",") if d]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)

    k_zero = ef("mbi_zero", [acc_ty])
    k_ug = ef("mbi_ug", [band_ty, xel_ty, acc_ty, hm_ty])
    k_hq = ef("mbi_hq", [acc_ty, acc_ty, hq_ty, hm_ty, np.int32])
    k_dn = ef("mbi_dn", [band_ty, hq_ty, hm_ty, acc_ty, np.int32])
    k_out = ef("mbi_out", [acc_ty, c_ty, hm_ty])

    # weights: one element a band k-group, broadcast from the mem tile to the column's four rows
    W = [ObjectFifo(band_ty, name=f"W{c}", depth=2) for c in range(N_COLS)]
    W_l1 = [W[c].cons().forward(obj_type=band_ty, name=f"W{c}_l1", depth=2, tile=Tile(c, 1)) for c in range(N_COLS)]
    # x: the slot's four rows in one mem-tile object, split by row, each part replayed NPB * 2 times
    # and consumed one k-group element at a time (consumer_obj_type: split() does not take it)
    X = [ObjectFifo(xslot_ty, name=f"X{c}", depth=1) for c in range(N_COLS)]
    X_l1 = []
    for c in range(N_COLS):
        subs = X[c].cons().split([r * XROW for r in range(N_ROWS)], obj_types=[xrow_ty] * N_ROWS,
                                 names=[f"X{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS,
                                 repeat_counts=[NPB * 2] * N_ROWS, tile=Tile(c, 1))
        for s in subs:
            s._consumer_obj_type = xel_ty
        X_l1.append(subs)
    # y: the rows' bands join at the mem tile, in row order ([row][rg][tg][t][16] bf16), and stream
    # out token-major (the dims apply to each row part: [tg t][rg][16], a token's 64 columns of the
    # band), so the drain lands y as [token][2048] (a host visit reads 4 KB in one run, not 128
    # lines at 8 KB strides)
    C = [ObjectFifo(c4_ty, name=f"C{c}", depth=2, dims_to_stream=[(16, 16), (4, 4 * NTG * 16), (16, 1)])
         for c in range(N_COLS)]
    C_l1 = [C[c].prod().join([r * ACC for r in range(N_ROWS)], obj_types=[c_ty] * N_ROWS,
                             names=[f"C{c}_{r}" for r in range(N_ROWS)], depths=[1] * N_ROWS, tile=Tile(c, 1))
            for c in range(N_COLS)]

    au = [[Buffer(acc_ty, name=f"au_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    ag = [[Buffer(acc_ty, name=f"ag_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    hq = [[Buffer(hq_ty, name=f"hq_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    hm = [[Buffer(hm_ty, name=f"hm_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]

    def core_fn(w_in, x_in, c_out, kz, kug, khq, kdn, kout, bu, bg, bh, bm):
        for pb in range_(NPB):
            kz(bu)
            kz(bg)
            for _ in range_(KG_UP):
                w = w_in.acquire(1)
                xe = x_in.acquire(1)
                kug(w, xe, bu, bm)
                x_in.release(1)
                w_in.release(1)
            for _ in range_(KG_UP):
                w = w_in.acquire(1)
                xe = x_in.acquire(1)
                kug(w, xe, bg, bm)
                x_in.release(1)
                w_in.release(1)
            khq(bu, bg, bh, bm, pb)
        # the down in the pool's order: per 128-row stripe, k-group major, its two 64-row bands (au, ag)
        for _ in range_(NOB // 2):
            kz(bu)
            kz(bg)
            for kg in range_(KG_DN):
                w = w_in.acquire(1)
                kdn(w, bh, bm, bu, kg)
                w_in.release(1)
                w = w_in.acquire(1)
                kdn(w, bh, bm, bg, kg)
                w_in.release(1)
            ce = c_out.acquire(1)
            kout(bu, ce, bm)
            c_out.release(1)
            ce = c_out.acquire(1)
            kout(bg, ce, bm)
            c_out.release(1)

    workers = [Worker(core_fn,
                      [W_l1[c].cons(), X_l1[c][r].cons(), C_l1[c][r].prod(), k_zero, k_ug, k_hq, k_dn, k_out,
                       au[r][c], ag[r][c], hq[r][c], hm[r][c]],
                      tile=Tile(c, 2 + r), stack_size=STACK)
               for c in range(N_COLS) for r in range(N_ROWS)]

    def up_tap(slot: int, pb: int) -> TensorAccessPattern:      # band pb: up kg 0..7, gate kg 0..7
        off = slot * EXP_UG + (pb // 2) * 2 * STRIPE + (pb % 2) * BAND
        return TensorAccessPattern((POOL_BYTES,), off, [2, KG_UP, 4, 2560], [STRIPE, 2 * BAND, 2560, 1])

    def down_tap(slot: int) -> TensorAccessPattern:              # the whole down, as the pool holds it
        return TensorAccessPattern((POOL_BYTES,), POOL_DOWN + slot * DOWN_BYTES, [NOB * KG_DN, 4, 2560],
                                   [BAND, 2560, 1])

    W_p = [W[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    X_p = [X[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    C_c = [C[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]

    def y_tap(slot: int) -> TensorAccessPattern:              # [ob][token][64 columns] -> [token][2048]
        return TensorAccessPattern((Y_BYTES,), slot * YSLOT, [NOB, 4 * NT, 128], [128, HID * 2, 1])

    def sequence(a_pool, a_x, a_y, W_hs, X_hs, C_hs):
        # The next round's x is issued after this round's up / gate fills: the mem tile's x object
        # frees when this round's up / gate ends, and the next x then streams during the down (issued
        # after the down fills, it waited for them and stalled every slot boundary).
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        rounds = slots // N_COLS
        for c in range(N_COLS):
            px.fill(X_hs[c], a_x, lin(X_BYTES, c * XSLOT, XSLOT))
        for rnd in range(rounds):
            slot = [rnd * N_COLS + c for c in range(N_COLS)]
            for c in range(N_COLS):                  # drains before fills (npu-model-porting)
                py.drain(C_hs[c], a_y, y_tap(slot[c]))
            for pb in range(NPB):
                for c in range(N_COLS):
                    pw.fill(W_hs[c], a_pool, up_tap(slot[c], pb))
            if rnd + 1 < rounds:
                for c in range(N_COLS):
                    px.fill(X_hs[c], a_x, lin(X_BYTES, (slot[c] + N_COLS) * XSLOT, XSLOT))
            for c in range(N_COLS):
                pw.fill(W_hs[c], a_pool, down_tap(slot[c]))
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, x_ty, y_ty, W_p, X_p, C_c])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = mbi
_src = b"".join([(HERE / "mbi.py").read_bytes(), (HERE / "mbi_core.h").read_bytes()]
                + sorted(f.read_bytes() for f in HERE.glob("mbi_*.cc"))
                + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes()]
                + [f"{k}={v}".encode() for k, v in (("NT", NT), ("HID", HID), ("FF", FF), ("EXPERTS", EXPERTS),
                                                     ("POOL_DOWN", POOL_DOWN), ("POOL_BYTES", POOL_BYTES),
                                                     ("DEFS", os.environ.get("MBI_DEFS", "")), ("STACK", STACK))])
SPECIALIZE = {"slots": SLOTS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}

if __name__ == "__main__":
    print(f"HID={HID} FF={FF} NT={NT} L1={L1} SLOTS={SLOTS} EXPERTS={EXPERTS} XEL={XEL} XSLOT={XSLOT} "
          f"YSLOT={YSLOT} POOL_DOWN={POOL_DOWN} POOL_BYTES={POOL_BYTES}")
