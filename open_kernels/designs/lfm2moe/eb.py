r"""eb: LFM2-MoE's routed experts over a block of tokens, NT token slots per expert visit --
the prefill counterpart of ex.py (OPEN-MOE-BATCH, generalised to a wide expert).

Per slot the host has gathered up to NT tokens that chose the same expert into
x[slot] = [HID/8][NT][8] bf16 (moe_batch's A tiles); the kernel computes
h = silu(gate x) * (up x) and y = down h, and the host scatters y[slot] back with the
router weights (Core::lfm2_experts_block). A slot is patched to any expert (moebatch:
slot s is compiled as expert s), so SLOTS <= the expert count.

WHAT IS moe_batch's AND WHAT IS NOT. The core arithmetic is moe_batch.h's, unchanged
(mb_step_ug / mb_step_dn / mb_silu / mb_zero_*): it reads a 64-row band's 256-wide k-group
as two consecutive q4_1 chunks, which is exactly how ex.py's std_perm band layout stores an
expert. What changes is the loop around it, because moe_batch is built for an expert width
of a quarter of the hidden (512): one 128-row band pair per core for up / gate, and the
WHOLE expert hidden held in the B fifo for the down. At 1536 that is three pairs per core,
and h is 48 KB at NT = 16 -- not in a 64 KB core beside everything else. So:

    up / gate   for each of the core's HB2 band pairs (pair p = 4 pg + r of the column's row r):
                  both projections, both bands, x streamed once per (pg, proj, band);
                  silu -> 128 h rows (bf16) -> C
    down        for each of the core's 64-row output bands (band 4 rbg + r):
                  h streamed again, k-tile by k-tile, against the band's 6 k-groups -> C

So x is streamed 12 times per slot and h 8 times: ~1.2 MB per slot per column against the
~5.9 MB of the expert's weights on the other shim channel.

Mapping (moe_batch's): one slot per column, eight in flight; the column's four rows split
the expert's rows. Weights: one 4-row element per k-group into the mem tile, split by row.
x / h: broadcast to the column's rows. C: the rows join in order, so h lands in DDR in
natural row order (the down's A tiles) and y as moe_batch's C tiles.

Build: EB_SLOTS=64 OPEN_KERNELS_SPEC=<lfm2moe spec> python build_design.py designs/lfm2moe/eb.py <dir>
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
MB = HERE.parent / "moe_batch"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402

SPEC = current_spec()
QR = for_spec(SPEC)
if SPEC.family != "lfm2moe":
    sys.exit(f"eb.py: an {SPEC.family!r} spec; this design is LFM2-MoE's")
if QR.F32:
    sys.exit("eb.py: reads the bf16-scale pool (moe_batch.h's 5120 B chunks); OPEN_LFM2MOE_QUANT=q4_1_f32 has none")
R = QR.recipe(SPEC)
L, M = R.layout, R.moe
E = QR.eb_geometry(SPEC)

N_ROWS, N_COLS = 4, 8
NT, HID, FF = E["nt"], M.HID, M.FF
BAND = 10240                 # one band's 256-wide k-group: two q4_1 chunks
K_TILE = 64
SLOTS = int(os.environ.get("EB_SLOTS", 64))
HB2 = FF // 128 // N_ROWS    # band pairs per core (3 at 1536)
NRB_DN = HID // 64 // N_ROWS # down bands per core (8 at 2048)
NKG_UP, NKG_DN = HID // 256, FF // 256
X_BYTES, H_BYTES, Y_BYTES = E["x_bytes"](SLOTS), E["h_bytes"](SLOTS), E["y_bytes"](SLOTS)
B_DEPTH = 4
C_DEPTH = 1
L1 = 0x1000 + 2 * BAND + (256 * NT * 4) + C_DEPTH * (64 * NT * 4) + (B_DEPTH + 1) * (K_TILE * NT * 2)

assert FF % (128 * N_ROWS) == 0, f"eb.py: the expert width {FF} is not whole band pairs over {N_ROWS} rows"
assert HID % (64 * N_ROWS) == 0
assert SLOTS % N_COLS == 0 and SLOTS <= M.E, f"eb.py: {SLOTS} slots (multiple of 8, at most {M.E} experts)"
assert L1 <= 64 << 10, f"eb.py: a core wants {L1} B of L1"


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def eb(pool: In, x: In, h: Out, y: Out, *, slots: CompileTime[int], srchash: CompileTime[int] = 0):
    band_ty = np.ndarray[(BAND,), np.dtype[np.uint8]]
    a4_ty = np.ndarray[(N_ROWS * BAND,), np.dtype[np.uint8]]
    b_ty = np.ndarray[(K_TILE * NT,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(64 * NT,), np.dtype[np.float32]]
    c4_ty = np.ndarray[(N_ROWS * 64 * NT,), np.dtype[np.float32]]
    ug_ty = np.ndarray[(256 * NT,), np.dtype[np.float32]]      # [u 128 rows | g 128 rows]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(X_BYTES,), np.dtype[np.uint8]]
    h_ty = np.ndarray[(H_BYTES,), np.dtype[np.uint8]]
    y_ty = np.ndarray[(Y_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(MB)]
    flags = [f"-DMB_NT={NT}"]

    def ef(sym, args, d=MB):
        return ExternalFunction(sym, source_file=str(d / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)

    # mb_step_ug writes band `half` of its buffer: with the [u | g] buffer, half = 2 proj + band
    step_ug = ef("mb_step_ug", [band_ty, b_ty, ug_ty, np.int32, np.int32])
    step_dn = ef("mb_step_dn", [band_ty, b_ty, c_ty, np.int32])
    zero_ug = ef("eb_zero_ug2", [ug_ty], HERE)
    zero_y = ef("mb_zero_y", [c_ty])
    silu = ef("eb_silu2", [ug_ty, c_ty], HERE)

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

    # Every loop is a hardware loop (range_): unrolled in Python, the fifo bookkeeping around
    # each call is emitted once per copy, and the core program overflowed its 16 KB by 512 B.
    def core_fn(in_a, in_b, out_c, k_ug, k_dn, k_zero_ug, k_zero_y, k_silu, ug):
        for _ in range_(HB2):                        # this core's band pairs
            k_zero_ug(ug)
            for pb in range_(4):                     # (up, band 0) (up, 1) (gate, 0) (gate, 1)
                for _ in range_(NKG_UP):
                    band = in_a.acquire(1)
                    for ky in range_(4):
                        xe = in_b.acquire(1)
                        k_ug(band, xe, ug, ky, pb)
                        in_b.release(1)
                    in_a.release(1)
            he = out_c.acquire(1)
            k_silu(ug, he)                           # 128 h rows, bf16, natural order
            out_c.release(1)
        for _ in range_(NRB_DN):                     # this core's output bands
            ye = out_c.acquire(1)
            k_zero_y(ye)
            for _ in range_(NKG_DN):
                band = in_a.acquire(1)
                for ky in range_(4):
                    hb = in_b.acquire(1)
                    k_dn(band, hb, ye, ky)
                    in_b.release(1)
                in_a.release(1)
            out_c.release(1)

    workers = [Worker(core_fn,
                      [A_l2l1[c][r].cons(), B[c].cons(), C_l1l2[c][r].prod(), step_ug, step_dn, zero_ug, zero_y, silu,
                       ugbuf[r][c]],
                      tile=Tile(c, 2 + r), stack_size=0x1000)
               for c in range(N_COLS) for r in range(N_ROWS)]

    def up_tap(slot: int, proj: int, pg: int, half: int) -> TensorAccessPattern:
        # row r's band is 2 (4 pg + r) + half; one element = the four rows' bands at one k-group
        off = L.POOL_EXP + slot * 2 * M.UP_BYTES + proj * M.UP_BYTES + (8 * pg + half) * M.BB_H
        return TensorAccessPattern((L.POOL_BYTES,), off, [NKG_UP, N_ROWS, 4, 2560], [BAND, 2 * M.BB_H, 2560, 1])

    def down_tap(slot: int, rbg: int) -> TensorAccessPattern:
        # row r's band is 4 rbg + r: the four rows' drains are y rows 256 rbg .. in order
        off = L.POOL_DOWN_EXP + slot * M.UP_BYTES + 4 * rbg * M.BB_F
        return TensorAccessPattern((L.POOL_BYTES,), off, [NKG_DN, N_ROWS, 4, 2560], [BAND, M.BB_F, 2560, 1])

    def lin(total: int, off: int, n: int) -> TensorAccessPattern:
        return TensorAccessPattern((total,), off, [1, n], [n, 1])

    A_prods = [A_l3l2[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    B_prods = [B[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    C_conss = [C_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]
    XS, HS, YS = HID * NT * 2, FF * NT * 2, HID * NT * 4      # one slot's x, h, y

    def sequence(a_pool, a_x, a_h, a_y, A_hs, B_hs, C_hs):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        for rnd in range(slots // N_COLS):
            slot = [rnd * N_COLS + c for c in range(N_COLS)]
            # the h drains FIRST (a core emits HB2 h elements; see npu-model-porting: drains before fills)
            for c in range(N_COLS):
                py.drain(C_hs[c], a_h, lin(H_BYTES, slot[c] * HS, HS))
            for pg in range(HB2):
                for proj in range(2):
                    for half in range(2):
                        for c in range(N_COLS):
                            pw.fill(A_hs[c], a_pool, up_tap(slot[c], proj, pg, half))
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


DESIGN = eb
_src = b"".join([(HERE / "eb.py").read_bytes(), (HERE / "eb_zero_ug2.cc").read_bytes(), (HERE / "eb_silu2.cc").read_bytes()]
                + sorted(f.read_bytes() for f in MB.glob("*.cc"))
                + [(MB / "moe_batch.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   f"NT={NT}".encode(), SPEC.spec_hash().encode()])
SPECIALIZE = {"slots": SLOTS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
