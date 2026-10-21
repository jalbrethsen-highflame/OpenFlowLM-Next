r"""dnc: the gated delta rule over a prompt chunk on the NPU (dnc_core.h), 16 tokens a step.

One call = one linear layer x STEPS * 16 tokens. Column c takes value heads 4c .. 4c + 3 one after
another; its row r takes value dims 32r .. 32r + 31 of the head (the unit), whose state slice (dk 128 x
dv 32 fp32, 16 KB) stays in L1 while the head's tokens stream through.

Per column and head:
  rec    shim -> mem tile -> the 4 rows (broadcast): STEPS records of 9216 B (K and Q tiles, the
         decays; dnc_core.h)
  vs     shim -> mem tile, 4 KB objects split 4 ways (1 KB a row): first the state's 16 pieces
         (from `s`, the layer's state in the tile layout), then STEPS v pieces (from `vo`)
  os     the rows -> mem tile (join, 4 KB) -> shim: STEPS o pieces, written over the v pieces in
         `vo` (each v piece is read before its o is made), then the state's 16 pieces back to `s`

Buffers (host layouts):
  rec  [32 heads][STEPS][9216]
  vo   [32 heads][STEPS][4 rows][16 t][32] bf16    (v in, o out)
  s    [32 heads][16 pieces][4 rows][256] f32       (each row's 16 KB slice: S^T tiles [db][kb][8][8])

Channels: mem tile S2MM 6 (rec, vs, 4 os) and MM2S 6 (rec, 4 vs, os); shim MM2S 2, S2MM 1.
L1 a core: S 16 KB, scratch 18 KB, rec 2 x 9 KB, vs / os 2 x 1 KB each, stack 2 KB: ~58 KB.

    DNC_STEPS=64 python build_design.py designs/deltanet/dnc.py <dir>
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402

N_ROWS, N_COLS = 4, 8
HEADS = 32
HPC = HEADS // N_COLS                    # value heads a column
STEPS = int(os.environ.get("DNC_STEPS", 64))
REC, PIECE, SPC, SCB = 9216, 1024, 16, 18432
OBJ = N_ROWS * PIECE                     # a mem-tile object of vs / os
DEFS = [f"-D{d}" for d in os.environ.get("DNC_DEFS", "").split(",") if d]
REC_BYTES = HEADS * STEPS * REC
VO_BYTES = HEADS * STEPS * OBJ
S_BYTES = HEADS * SPC * OBJ


def lin(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((total,), off, [1, n], [n, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dnc(rec: In, vo: Out, s: Out, *, steps: CompileTime[int], srchash: CompileTime[int] = 0):
    rec_ty = np.ndarray[(REC,), np.dtype[np.uint8]]
    pc_ty = np.ndarray[(PIECE,), np.dtype[np.uint8]]
    obj_ty = np.ndarray[(OBJ,), np.dtype[np.uint8]]
    st_ty = np.ndarray[(SPC * PIECE // 4,), np.dtype[np.float32]]
    sc_ty = np.ndarray[(SCB,), np.dtype[np.uint8]]
    inc = include_dirs() + [str(HERE)]

    def ef(sym, args):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc, compile_flags=DEFS)

    k_sin = ef("dnc_sin", [pc_ty, st_ty, np.int32])
    k_sout = ef("dnc_sout", [st_ty, pc_ty, np.int32])
    k_step = ef("dnc_step", [rec_ty, pc_ty, pc_ty, st_ty, sc_ty])

    R = [ObjectFifo(rec_ty, name=f"R{c}", depth=2) for c in range(N_COLS)]
    R_l1 = [R[c].cons().forward(obj_type=rec_ty, name=f"R{c}_l1", depth=2, tile=Tile(c, 1)) for c in range(N_COLS)]
    VS = [ObjectFifo(obj_ty, name=f"VS{c}", depth=2) for c in range(N_COLS)]
    VS_l1 = [VS[c].cons().split([r * PIECE for r in range(N_ROWS)], obj_types=[pc_ty] * N_ROWS,
                                names=[f"VS{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS, tile=Tile(c, 1))
             for c in range(N_COLS)]
    OS = [ObjectFifo(obj_ty, name=f"OS{c}", depth=2) for c in range(N_COLS)]
    OS_l1 = [OS[c].prod().join([r * PIECE for r in range(N_ROWS)], obj_types=[pc_ty] * N_ROWS,
                               names=[f"OS{c}_{r}" for r in range(N_ROWS)], depths=[2] * N_ROWS, tile=Tile(c, 1))
             for c in range(N_COLS)]
    S = [[Buffer(st_ty, name=f"S_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]
    SC = [[Buffer(sc_ty, name=f"SC_{r}_{c}") for c in range(N_COLS)] for r in range(N_ROWS)]

    def core(r_in, vs_in, os_out, ksin, ksout, kstep, Sb, scb):
        for _ in range_(HPC):
            for p in range_(SPC):
                e = vs_in.acquire(1)
                ksin(e, Sb, p)
                vs_in.release(1)
            for _ in range_(steps):
                re = r_in.acquire(1)
                ve = vs_in.acquire(1)
                oe = os_out.acquire(1)
                kstep(re, ve, oe, Sb, scb)
                os_out.release(1)
                vs_in.release(1)
                r_in.release(1)
            for p in range_(SPC):
                e = os_out.acquire(1)
                ksout(Sb, e, p)
                os_out.release(1)

    workers = [Worker(core, [R_l1[c].cons(), VS_l1[c][r].cons(), OS_l1[c][r].prod(), k_sin, k_sout, k_step,
                             S[r][c], SC[r][c]], tile=Tile(c, 2 + r), stack_size=0x800)
               for c in range(N_COLS) for r in range(N_ROWS)]

    R_p = [R[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    VS_p = [VS[c].prod(tile=Tile(c, 0)) for c in range(N_COLS)]
    OS_c = [OS[c].cons(tile=Tile(c, 0)) for c in range(N_COLS)]

    def sequence(a_rec, a_vo, a_s, R_hs, VS_hs, OS_hs):
        pr, pv, po = Pipeline(3), Pipeline(3), Pipeline(3)
        for hh in range(HPC):
            for c in range(N_COLS):
                h = c * HPC + hh
                po.drain(OS_hs[c], a_vo, lin(VO_BYTES, h * steps * OBJ, steps * OBJ))   # drains first
                po.drain(OS_hs[c], a_s, lin(S_BYTES, h * SPC * OBJ, SPC * OBJ))
            for c in range(N_COLS):
                h = c * HPC + hh
                pv.fill(VS_hs[c], a_s, lin(S_BYTES, h * SPC * OBJ, SPC * OBJ))
                pv.fill(VS_hs[c], a_vo, lin(VO_BYTES, h * steps * OBJ, steps * OBJ))
                pr.fill(R_hs[c], a_rec, lin(REC_BYTES, h * steps * REC, steps * REC))
        pr.finish()
        pv.finish()
        po.finish()

    rt = Runtime(sequence, [np.ndarray[(REC_BYTES,), np.dtype[np.uint8]], np.ndarray[(VO_BYTES,), np.dtype[np.uint8]],
                            np.ndarray[(S_BYTES,), np.dtype[np.uint8]], R_p, VS_p, OS_c])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dnc
_src = b"".join([(HERE / f).read_bytes() for f in ("dnc.py", "dnc_core.h", "dnc_sin.cc", "dnc_sout.cc", "dnc_step.cc")]
                + [f"{STEPS},{DEFS}".encode()])
SPECIALIZE = {"steps": STEPS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
