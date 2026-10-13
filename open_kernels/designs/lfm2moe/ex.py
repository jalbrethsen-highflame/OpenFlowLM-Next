r"""ex: LFM2-MoE's routed experts for one token, ONE dispatch, the router already run.

    acc  = sum_s w[s] * down_e(s)( bf16( silu(gate_e(s) @ xm) * (up_e(s) @ xm) ) )
    xres = res + acc

for the top-k slots s the host picked (`hostroute`, src/open_qwen36/core.cpp). xm (bf16,
ffn_norm of the residual) and res (f32, the residual) were left in `act` by the mixer
dispatch before this one (cx.py / dx.py with *_MIX=1); the routing record (idx u32[8] at
ROUT_IDX_OFF, the weights f32 after them) was written there by the host.

WHY NOT layer_x's MoE body (OPEN-MOE-WIDE-FF). That one splits ONE 128-row expert stripe
over two cores, which holds while the expert width is a quarter of the hidden. LFM2-MoE's
is three quarters (1536 against 2048): 12 stripes over 8 cores. So the experts here are
stored as plain std_perm 64-row bands -- [up | gate] per expert from POOL_EXP, the down
slices from POOL_DOWN_EXP, each expert exactly a dense FFN of width FF -- and each core
owns HB whole up / gate bands (3) and DB whole down bands (4). The routed slots are
compiled as experts 0..k-1, and moeroute2 (stream_patch.hpp, with the geometry's `base`)
re-points every fill at the host's choice.

Per token, per core (Tile(c, 2)), all f32 accumulation, bf16 activations into the GEMV:

    xm -> table (K = HID) ; header: routing record -> rw, residual rows -> xr
    for each slot:  HB x [ up band -> u, gate band -> g, silu(g) * u -> y ]   (h part, to DDR)
                    the whole h back (H_ELEMS broadcast x elements) -> table (K = FF)
                    DB x [ down band -> yd + 64 j ]
                    acc = (slot 0 ? 0 : acc) + w[slot] * yd
    acc = res + acc  -> DB y elements -> xres

Streams: w (CALL_BYTES weight elements, one shim fill per band), x (4 KB broadcast: xm,
then h per slot), y (64-float results). The h parts meet in DDR (act at AD_HP) and come
back on x; `py.finish` on the h drains orders that.

Args: pool, xres, act -- pool FIRST, because moeroute2 re-points the patches on arg 0.
Build: OPEN_KERNELS_SPEC=<lfm2moe spec> python build_design.py designs/lfm2moe/ex.py designs/lfm2moe/build
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402

SPEC = current_spec()
QR = for_spec(SPEC)
if SPEC.family != "lfm2moe":
    sys.exit(f"ex.py: an {SPEC.family!r} spec; this design is LFM2-MoE's expert block")
R = QR.recipe(SPEC)
L, M = R.layout, R.moe
N_CORES, HID, NE = M.N_CORES, M.HID, M.NE
XN_ELEMS = HID * 2 // ELEM
assert HID * 2 % ELEM == 0, "ex.py: xm is a whole number of 4 KB elements"
OS = ["-Os"]
GEMV_DEF = ["-DGEMV_Q4_SCALES_F32=1", "-DGEMV_Q4_PREFIX=gemv_q4s32"]     # GGUF-direct pools only


def bt(total: int, off: int, n: int) -> TensorAccessPattern:
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def ex(pool: In, xres: InOut, act: InOut, *, srchash: CompileTime[int] = 0):
    elem = np.ndarray[(M.CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(M.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(M.MS_FLOATS,), np.dtype[np.float32]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    i32 = np.int32

    inc = include_dirs() + [str(GEMV), str(HERE)]

    def ef(sym, args, flags=None):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=(OS if flags is None else flags) + GEMV_DEF)

    f_gu = ef("ex_gu", [elem, tab_ty, ms_ty, i32, i32])
    f_gd = ef("ex_gd", [elem, tab_ty, ms_ty, i32, i32])
    f_px = ef("ex_px", [x_ty, tab_ty, i32])
    f_ph = ef("ex_ph", [x_ty, tab_ty, i32])
    f_hdr = ef("ex_hdr", [elem, ms_ty, i32])
    f_act = ef("ex_act", [ms_ty, y_ty])
    f_acc = ef("ex_acc", [ms_ty, i32])
    f_out = ef("ex_out", [ms_ty, y_ty, i32])

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)

    def main_body(win, xin, yout, tab, ms, f_gu, f_gd, f_px, f_ph, f_hdr, f_act, f_acc, f_out):
        for i in range(XN_ELEMS):                    # xm -> its table
            xe = xin.acquire(1)
            f_px(xe, tab, i)
            xin.release(1)
        for mode in range(2):                        # routing record, residual rows
            we = win.acquire(1)
            f_hdr(we, ms, mode)
            win.release(1)
        for s in range_(NE):
            for _ in range_(M.HB):                   # this core's up / gate bands -> h part
                for g in range_(M.NG_H):
                    we = win.acquire(1)
                    f_gu(we, tab, ms, g, M.MS_U)
                    win.release(1)
                for g in range_(M.NG_H):
                    we = win.acquire(1)
                    f_gu(we, tab, ms, g, M.MS_G)
                    win.release(1)
                ye = yout.acquire(1)
                f_act(ms, ye)
                yout.release(1)
            for i in range(M.H_ELEMS):               # the whole h back -> its table
                he = xin.acquire(1)
                f_ph(he, tab, i)
                xin.release(1)
            for j in range_(M.DB):                   # this core's down rows
                for g in range_(M.NG_F):
                    we = win.acquire(1)
                    f_gd(we, tab, ms, g, j)
                    win.release(1)
            f_acc(ms, s)
        f_acc(ms, -1)                                # acc = res + acc
        for j in range_(M.DB):
            ye = yout.acquire(1)
            f_out(ms, ye, j)
            yout.release(1)

    workers = [Worker(main_body,
                      fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                               Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                               f_gu, f_gd, f_px, f_ph, f_hdr, f_act, f_acc, f_out],
                      tile=Tile(c, 2), stack_size=0x1400)
               for c in range(N_CORES)]

    YB = BAND_ROWS * 4

    def sequence(a_pool, c_xres, a_act, w_prods, x_prod, y_conss):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, XN_ELEMS * ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_act, bt(L.AD_BYTES, L.AD_ROUT, M.CALL_BYTES))
            pw.fill(w_prods[c], a_act, bt(L.AD_BYTES, L.AD_RES + c * M.ROWS_PC * 4, M.CALL_BYTES))
        for s in range(NE):
            up = L.POOL_EXP + s * 2 * M.UP_BYTES     # slot s compiled as expert s (moeroute2)
            # The h drains FIRST: a core blocks on its depth-2 y fifo after two bands, and the
            # Pipeline only issues a fill once the one three back has completed -- so with the
            # drain behind a core's own fills, more than three bands per core is a deadlock
            # (the 4-core 8B, and the 24B at 4 cores, hung exactly there). dx.py's order.
            for c in range(N_CORES):
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_HP + c * M.HID_PC * 4, M.HID_PC * 4))
            for b in range(M.HB):                    # band by band across the cores
                for c in range(N_CORES):
                    band = c * M.HB + b
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, up + band * M.BB_H, M.BB_H))
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, up + M.UP_BYTES + band * M.BB_H, M.BB_H))
            py.finish(*y_conss)                      # the h parts are in DDR
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_HP, M.H_ELEMS * ELEM))
            down = L.POOL_DOWN_EXP + s * M.UP_BYTES
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, down + c * M.DB * M.BB_F, M.DB * M.BB_F))
        for c in range(N_CORES):
            py.drain(y_conss[c], c_xres, bt(HID, c * M.ROWS_PC, M.ROWS_PC))   # the layer output
        # (the down fill is one transfer per core per slot, so nothing above waits on a core
        # that is itself waiting on this drain)
        pw.finish()
        px.finish()
        py.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, act_ty,
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = ex
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + [(GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
