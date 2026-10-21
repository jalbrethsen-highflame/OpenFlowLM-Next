r"""mbi_probe: mbi_band_kg (mbi_step.h) on one core over one band k-group, PROBE_ITERS times:
its result against numpy (make_mbi_probe.py) and its time. Probe, not a kernel."""
from __future__ import annotations
import hashlib, os, sys
from pathlib import Path
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import include_dirs  # noqa: E402
NT = int(os.environ.get("MBI_NT", 16)); NTG = NT // 4
ITERS = int(os.environ.get("PROBE_ITERS", 1))
V3 = "MBI_V3" in os.environ.get("MBI_DEFS", "")
W_B, X_B, A_B = 10240, (64 if V3 else 0) + 16 * NTG * 64 + 8 * NTG * 128 + NTG * 256, 4 * NTG * 64 * 4

@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def mbi_probe2(w: In, x: In, acc: Out, *, srchash: CompileTime[int] = 0):
    w_ty = np.ndarray[(W_B,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(X_B,), np.dtype[np.uint8]]
    a_ty = np.ndarray[(A_B // 4,), np.dtype[np.float32]]
    k = ExternalFunction("mbi_probe2", source_file=str(HERE / "mbi_probe2.cc"), arg_types=[w_ty, x_ty, a_ty],
                         include_dirs=include_dirs() + [str(HERE)], compile_flags=[f"-DMBI_NT={NT}", f"-DPROBE_ITERS={ITERS}"] + [f"-D{v}" for v in os.environ.get("MBI_DEFS", "").split(",") if v] + os.environ.get("MBI_CFLAGS", "").split())
    W, X, A = ObjectFifo(w_ty, name="W", depth=1), ObjectFifo(x_ty, name="X", depth=1), ObjectFifo(a_ty, name="A", depth=1)
    def core(wi, xi, ao, kern):
        a = wi.acquire(1); b = xi.acquire(1); c = ao.acquire(1)
        kern(a, b, c)
        wi.release(1); xi.release(1); ao.release(1)
    ws = [Worker(core, [W.cons(), X.cons(), A.prod(), k], tile=Tile(0, 2), stack_size=0x1800)]
    wp, xp, ac = W.prod(tile=Tile(0, 0)), X.prod(tile=Tile(0, 0)), A.cons(tile=Tile(0, 0))
    def lin(n):
        return TensorAccessPattern((n,), 0, [1, n], [n, 1])
    def seq(wb, xb, ab, wp_, xp_, ac_):
        wp_.fill(wb, tap=lin(W_B))
        xp_.fill(xb, tap=lin(X_B))
        ac_.drain(ab, tap=lin(A_B // 4), wait=True)
    rt = Runtime(seq, [w_ty, x_ty, a_ty, wp, xp, ac])
    return Program(iron.get_current_device(), rt, workers=ws).resolve_program()


DESIGN = mbi_probe2
_src = b"".join([(HERE / f).read_bytes() for f in ("mbi_probe2.py", "mbi_probe2.cc", "mbi_core.h")] + [f"{NT},{ITERS},{os.environ.get('MBI_DEFS', '')},{os.environ.get('MBI_CFLAGS', '')}".encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
