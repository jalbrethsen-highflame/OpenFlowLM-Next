r"""peak: the matrix unit's raw rate on all 32 cores (PEAK_MODE bfp | i8 | i4, PEAK_ITERS).
Every core runs peak_run once per dispatch on a 192 B input it got from DDR, and writes 1 KB
of accumulators back; the dispatch time is then ~all MACs. Timing tool, not a kernel."""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import include_dirs  # noqa: E402

MODE = os.environ.get("PEAK_MODE", "bfp")
ITERS = int(os.environ.get("PEAK_ITERS", 1 << 20))
R, C = 4, 8
IN, OUT = 256, 256        # bytes in, int32 out per core

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def peak(a: In, b: Out, *, srchash: CompileTime[int] = 0):
    in_ty = np.ndarray[(IN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT,), np.dtype[np.int32]]
    in4 = np.ndarray[(R * IN,), np.dtype[np.int8]]
    out4 = np.ndarray[(R * OUT,), np.dtype[np.int32]]
    k = ExternalFunction("peak_run", source_file=str(HERE / "peak.cc"), arg_types=[in_ty, out_ty],
                         include_dirs=include_dirs() + [str(HERE.parent / "moe_batch")], compile_flags=[f"-DPEAK_{MODE.upper()}", f"-DPEAK_ITERS={ITERS}"] + [f"-D{x}" for x in os.environ.get("PEAK_DEFS", "").split(",") if x])
    A = [ObjectFifo(in4, name=f"A{c}", depth=1) for c in range(C)]
    Asp = [A[c].cons().split([r * IN for r in range(R)], obj_types=[in_ty] * R, names=[f"A{c}_{r}" for r in range(R)],
                             depths=[1] * R, tile=Tile(c, 1)) for c in range(C)]
    O = [ObjectFifo(out4, name=f"O{c}", depth=1) for c in range(C)]
    Oj = [O[c].prod().join([r * OUT for r in range(R)], obj_types=[out_ty] * R, names=[f"O{c}_{r}" for r in range(R)],
                           depths=[1] * R, tile=Tile(c, 1)) for c in range(C)]
    def core(i, o, kern):
        x = i.acquire(1); y = o.acquire(1)
        kern(x, y)
        i.release(1); o.release(1)
    ws = [Worker(core, [Asp[c][r].cons(), Oj[c][r].prod(), k], tile=Tile(c, 2 + r)) for c in range(C) for r in range(R)]
    ap = [A[c].prod(tile=Tile(c, 0)) for c in range(C)]
    oc = [O[c].cons(tile=Tile(c, 0)) for c in range(C)]
    def seq(ai, bo, aps, ocs):
        from aie.helpers.taplib import TensorAccessPattern
        for c in range(C):
            aps[c].fill(ai, tap=TensorAccessPattern((C * R * IN,), c * R * IN, [1, R * IN], [R * IN, 1]))
        for c in range(C):
            ocs[c].drain(bo, tap=TensorAccessPattern((C * R * OUT,), c * R * OUT, [1, R * OUT], [R * OUT, 1]), wait=True)
    rt = Runtime(seq, [np.ndarray[(C * R * IN,), np.dtype[np.int8]], np.ndarray[(C * R * OUT,), np.dtype[np.int32]], ap, oc])
    return Program(iron.get_current_device(), rt, workers=ws).resolve_program()

DESIGN = peak
SPECIALIZE = {"srchash": hash((MODE, ITERS)) & 0xFFFFFFF}
