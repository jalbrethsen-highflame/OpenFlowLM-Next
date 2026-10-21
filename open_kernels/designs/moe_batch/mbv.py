r"""mbv: a decode GEMV y = W x over the first 32 Q bands (64 rows each) of a q4_1 region in mbi's
transform (signed nibbles, int16 d / mc at one shift a matrix), for up to MBI_NT = 16 rows of int8
x, on the int8 x int4 MAC mode (mbi_core.h's band step). The CPU reads the region's other rows at
the same time: with a GPU job running the NPU's DMA keeps ~35 GB/s where the CPU cores get 12-55.

The region is the pool's std_perm band order (host::q41_std): band b's K / 256 k-group elements of
10240 B (two 32-row chunks) are contiguous, bands in order. Core (c, r) takes bands
(4 c + r) Q .. (4 c + r + 1) Q - 1.

Per column c:
  weights  shim -> mem tile, one object = one element for each of the 4 rows (40 KB), split to them
  x        shim -> mem tile once (K / 256 elements of MBI_XEL3, mbi's x element: header with the
           shift and the real rows, xa, qsum, scales), replayed Q times to the 4 rows (broadcast)
  y        the rows' f32 accumulators ([4 row groups][4 token groups][64]) join at the mem tile:
           y [c][band j][row r][4][4][64] f32
The weights tap starts at byte 0 of arg 0; the driver patches the region's offset in (mbv_table).

    MBV_K=2048 MBV_Q=4 python build_design.py designs/moe_batch/mbv.py <dir>
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern
from aie.iron.kernel import ExternalFunction

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N_ROWS, N_COLS = 4, 8
NT = 16
NTG = NT // 4
K = int(os.environ.get("MBV_K", 2048))
Q = int(os.environ.get("MBV_Q", 4))                # bands a core
W_BYTES = int(os.environ.get("MBV_W_BYTES", 1 << 30))   # the weight argument's size (a pool)
KG = K // 256
BAND_EL = 10240                                    # one 64-row band k-group
BAND = KG * BAND_EL
XEL = 64 + 16 * NTG * 64 + 8 * NTG * 128 + NTG * 128
ACC = 4 * NTG * 64                                 # floats
X_BYTES = KG * XEL
Y_BYTES = N_COLS * Q * N_ROWS * ACC * 4
STACK = 0x1000
assert K % 256 == 0 and K <= 4096
assert 2 * BAND_EL + 2 * XEL + 2 * ACC * 4 + STACK <= 64 << 10


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mbv(w: In, x: In, y: Out, *, q: CompileTime[int], srchash: CompileTime[int] = 0):
    el_ty = np.ndarray[(BAND_EL,), np.dtype[np.uint8]]
    el4_ty = np.ndarray[(N_ROWS * BAND_EL,), np.dtype[np.uint8]]
    xel_ty = np.ndarray[(XEL,), np.dtype[np.uint8]]
    xall_ty = np.ndarray[(X_BYTES,), np.dtype[np.uint8]]
    acc_ty = np.ndarray[(ACC,), np.dtype[np.float32]]
    acc4_ty = np.ndarray[(N_ROWS * ACC,), np.dtype[np.float32]]
    inc = include_dirs() + [str(HERE)]
    flags = [f"-DMBI_NT={NT}"]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)

    k_zero = ef("mbi_zero", [acc_ty])
    k_step = ef("mbv_step", [el_ty, xel_ty, acc_ty])

    W = [ObjectFifo(el4_ty, name=f"W{c}", depth=2) for c in range(N_COLS)]
    W_l1 = [W[c].cons().split([r * BAND_EL for r in range(N_ROWS)], obj_types=[el_ty] * N_ROWS,
                              names=[f"W{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS, tile=Tile(c, 1))
            for c in range(N_COLS)]
    # x: one object at the mem tile, replayed q times to the column's four rows, consumed an element at a time
    X = [ObjectFifo(xall_ty, name=f"X{c}", depth=1) for c in range(N_COLS)]
    X_l1 = []
    for c in range(N_COLS):
        (part,) = X[c].cons().split([0], obj_types=[xall_ty], names=[f"X{c}_b"], depths=[2], repeat_counts=[q],
                                    tile=Tile(c, 1))
        part._consumer_obj_type = xel_ty
        X_l1.append(part)
    Y = [ObjectFifo(acc4_ty, name=f"Y{c}", depth=2) for c in range(N_COLS)]
    Y_l1 = [Y[c].prod().join([r * ACC for r in range(N_ROWS)], obj_types=[acc_ty] * N_ROWS,
                             names=[f"Y{c}_{r}" for r in range(N_ROWS)], depths=[1] * N_ROWS, tile=Tile(c, 1))
            for c in range(N_COLS)]

    def core_fn(w_in, x_in, y_out, kz, kstep):
        for _ in range_(q):
            o = y_out.acquire(1)
            kz(o)
            for _ in range_(KG):
                we = w_in.acquire(1)
                xe = x_in.acquire(1)
                kstep(we, xe, o)
                x_in.release(1)
                w_in.release(1)
            y_out.release(1)

    workers = [Worker(core_fn, [W_l1[c][r].cons(), X_l1[c].cons(), Y_l1[c][r].prod(), k_zero, k_step],
                      tile=Tile(c, 2 + r), stack_size=STACK)
               for c in range(N_COLS) for r in range(N_ROWS)]

    W_p = [W[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    X_p = [X[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    Y_c = [Y[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]

    def sequence(a_w, a_x, a_y, W_hs, X_hs, Y_hs):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_COLS):
            py.drain(Y_hs[c], a_y, TensorAccessPattern((Y_BYTES // 4,), c * q * N_ROWS * ACC, [1, q * N_ROWS * ACC],
                                                       [q * N_ROWS * ACC, 1]))
        for c in range(N_COLS):
            px.fill(X_hs[c], a_x, TensorAccessPattern((X_BYTES,), 0, [KG, 4, XEL // 4], [XEL, XEL // 4, 1]))
            # element j of rows 0..3 (each row's q bands contiguous), 10240 B as 4 x 2560 (BD wraps < 1024 words)
            pw.fill(W_hs[c], a_w, TensorAccessPattern((W_BYTES,), c * N_ROWS * q * BAND, [q * KG, N_ROWS, 4, 2560],
                                                      [BAND_EL, q * BAND, 2560, 1]))
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [np.ndarray[(W_BYTES,), np.dtype[np.uint8]], np.ndarray[(X_BYTES,), np.dtype[np.uint8]],
                            np.ndarray[(Y_BYTES // 4,), np.dtype[np.float32]], W_p, X_p, Y_c])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = mbv
_src = b"".join([(HERE / f).read_bytes() for f in ("mbv.py", "mbi_core.h", "mbv_step.cc", "mbi_zero.cc")]
                + [(HERE.parent.parent / "include" / "vecmath.h").read_bytes(), f"{K},{W_BYTES}".encode()])
SPECIALIZE = {"q": Q, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}

if __name__ == "__main__":
    print(f"K={K} Q={Q} XEL={XEL} X_BYTES={X_BYTES} Y_BYTES={Y_BYTES} rows on the NPU {N_COLS * N_ROWS * Q * 64}")
