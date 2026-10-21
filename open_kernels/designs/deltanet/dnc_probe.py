r"""dnc_probe: dnc_core.h's 16-token step on one core over PROBE_STEPS steps of one unit: the state
slice in (16 pieces), the steps, the state out. Inputs rec [steps][9216], vs [16 + steps][1024] (the
state's 16 pieces, then the steps' v); outputs o [steps][1024], s_out [16384].

    PROBE_STEPS=8 python build_design.py designs/deltanet/dnc_probe.py <dir>
"""
from __future__ import annotations
import os, sys, hashlib
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
from ironutil import include_dirs  # noqa: E402

STEPS = int(os.environ.get("PROBE_STEPS", 8))
DEFS = [f"-D{d}" for d in os.environ.get("DNC_DEFS", "").split(",") if d]
REC, PIECE, SB, SCB = 9216, 1024, 16384, 18432


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def dnc_probe(rec: In, vs: In, o: Out, s_out: Out, *, steps: CompileTime[int], srchash: CompileTime[int] = 0):
    rec_ty = np.ndarray[(REC,), np.dtype[np.uint8]]
    pc_ty = np.ndarray[(PIECE,), np.dtype[np.uint8]]
    s_ty = np.ndarray[(SB // 4,), np.dtype[np.float32]]
    sc_ty = np.ndarray[(SCB,), np.dtype[np.uint8]]
    inc = include_dirs() + [str(HERE)]
    k_sin = ExternalFunction("dnc_sin", source_file=str(HERE / "dnc_sin.cc"), compile_flags=DEFS, arg_types=[pc_ty, s_ty, np.int32], include_dirs=inc)
    k_sout = ExternalFunction("dnc_sout", source_file=str(HERE / "dnc_sout.cc"), compile_flags=DEFS, arg_types=[s_ty, pc_ty, np.int32], include_dirs=inc)
    k_step = ExternalFunction("dnc_step", source_file=str(HERE / "dnc_step.cc"), compile_flags=DEFS, arg_types=[rec_ty, pc_ty, pc_ty, s_ty, sc_ty], include_dirs=inc)
    R = ObjectFifo(rec_ty, name="R", depth=2)
    V = ObjectFifo(pc_ty, name="V", depth=2)                 # the state's 16 pieces, then the steps' v
    O = ObjectFifo(pc_ty, name="O", depth=2)
    SO = ObjectFifo(pc_ty, name="SO", depth=2)
    S = Buffer(s_ty, name="S")
    SC = Buffer(sc_ty, name="SC")

    def core(r_in, v_in, o_out, so_out, ksin, ksout, kstep, Sb, scb):
        for p in range_(16):
            e = v_in.acquire(1)
            ksin(e, Sb, p)
            v_in.release(1)
        for _ in range_(steps):
            re = r_in.acquire(1)
            ve = v_in.acquire(1)
            oe = o_out.acquire(1)
            kstep(re, ve, oe, Sb, scb)
            o_out.release(1)
            v_in.release(1)
            r_in.release(1)
        for p in range_(16):
            e = so_out.acquire(1)
            ksout(Sb, e, p)
            so_out.release(1)

    w = Worker(core, [R.cons(), V.cons(), O.prod(), SO.prod(), k_sin, k_sout, k_step, S, SC],
               tile=Tile(0, 2), stack_size=0x800)
    R_p, V_p = R.prod(tile=Tile(0, 0)), V.prod(tile=Tile(0, 0))
    O_c, SO_c = O.cons(tile=Tile(0, 0)), SO.cons(tile=Tile(0, 0))
    lin = lambda n: TensorAccessPattern((n,), 0, [1, n], [n, 1])

    def sequence(a_r, a_v, a_o, a_so, r_p, v_p, o_c, so_c):
        r_p.fill(a_r, tap=lin(steps * REC), wait=False)
        v_p.fill(a_v, tap=lin(SB + steps * PIECE), wait=False)
        o_c.drain(a_o, tap=lin(steps * PIECE), wait=True)
        so_c.drain(a_so, tap=lin(SB), wait=True)

    rt = Runtime(sequence, [np.ndarray[(steps * REC,), np.dtype[np.uint8]], np.ndarray[(SB + steps * PIECE,), np.dtype[np.uint8]],
                            np.ndarray[(steps * PIECE,), np.dtype[np.uint8]], np.ndarray[(SB,), np.dtype[np.uint8]],
                            R_p, V_p, O_c, SO_c])
    return Program(iron.get_current_device(), rt, workers=[w]).resolve_program()


DESIGN = dnc_probe
_src = b"".join([(HERE / f).read_bytes() for f in ("dnc_probe.py", "dnc_core.h", "dnc_sin.cc", "dnc_sout.cc", "dnc_step.cc")] + [str(STEPS).encode(), str(DEFS).encode()])
SPECIALIZE = {"steps": STEPS, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
