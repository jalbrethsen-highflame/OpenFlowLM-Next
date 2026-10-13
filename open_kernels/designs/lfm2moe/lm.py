r"""lm: LFM2-MoE's MoE layer -- both mixers AND the routed experts -- on ONE image, so the
decode layer walk stays in one hardware context (OPEN-DECODE-ONE-CONTEXT, which ux.py does
for Qwen3.6). Three instruction streams over the same cores:

    part 0 = cm   the conv block alone      (cx.py with CX_MIX=1)
    part 1 = am   the attention block alone (dx.py with DX_MIX=1)
    part 2 = ex   the routed experts        (ex.py)

The engine runs (cm | am), then the host router, then ex, per layer. Separate images cost a
hardware-context switch on every one of those dispatches -- ~0.7 ms each, 76 a token on the
24B, about half of the step (the 2026-10-13 profile).

WHY IT FITS
-----------
The main cores (Tile(c, 2)) run ONE program per layer: [mixer stage 1 | mixer stage 2 |
experts]. Both mixers are the same two GEMV stages with one number changed -- stage 1 is
nA bands of K = HID against xn (conv: B | C | u, 3 x SC_PC = 12; attention: q | k | v,
Q_PC + 2 KV_PC = 6), stage 2 is 4 bands of K = HID against the conv's y or the attention's
og (both HID wide). nA arrives as an RTP word written at the head of parts 0 and 1, read
only after the core's first x acquire (which the stream's xn fill orders behind the write
-- ux.py's argument). Then the expert body (ex.py's, unchanged) runs on part 2's elements.
One pass of the body is one whole layer of either type, so between layers every main core
is parked on the same acquire.

The helpers need no parameter: each is fed by one layer type only and blocks through the
other's dispatches.

    (0, 3)      the norm helper: [x0 x1 lnw] -> xn, then res = x + out, xm = norm(res)
                -- identical in both mixers
    (2..5, 3)   the attention cores (dx.py's, blocked walk)
    (2, 4)      the conv core (cx.py puts it at (2, 3), which is attention core 0 here;
                rows 4-5 are empty)

SHIM BUDGET (2 MM2S + 2 S2MM per shim tile, 16 + 16 over the array), placed by hand:
    MM2S 13:  T0 lni+w0 | T1 w1+x | T2 w2+scin | T3 w3+scw | T4 w4+ain | T5 w5 | T6 w6 | T7 w7
    S2MM 16:  T0 y0+lno | T1 y1+scy | T2 y2+scst | T3 y3+aout | T4..T7 y4..y7 + og0..og3
Every S2MM channel is used. A mixer that needs one more drain does not fit this image.

Args are the attention layer's six (pool, xres, consts, state, act, ptab) for every part
-- one image, one kernel signature. `state` is the conv state in a conv layer and the KV
cache in an attention layer; ex ignores consts, state and ptab.

The weight elements are 12 KB (two f32-scale chunks) in all three parts: one fifo type.
The standalone cx / dx builds of this family stream 6 KB, because the dense layers' 11776
wide FFN table leaves no room for two; the mixers here have no FFN, so the table is K =
HID (and the experts' K = FF table beside it).

Build: LM_PART=<0|1|2> OPEN_KERNELS_SPEC=<lfm2moe spec> python build_design.py designs/lfm2moe/lm.py <dir>
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
DESIGNS = HERE.parent
GEMV = DESIGNS / "gemv_q4"
ATTN = DESIGNS / "attn"
LN = DESIGNS / "ln"
LINL = DESIGNS / "lin_layer"
SC = DESIGNS / "short_conv"
sys.path.insert(0, str(DESIGNS.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes.families import for_spec  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402

SPEC = current_spec()
QR = for_spec(SPEC)
if SPEC.family != "lfm2moe":
    sys.exit(f"lm.py: an {SPEC.family!r} spec; this image is LFM2-MoE's")
R = QR.recipe(SPEC)
L, G, M = R.layout, R.geo, R.moe

PART = int(os.environ.get("LM_PART", 0))
if PART not in (0, 1, 2):
    sys.exit(f"lm.py: LM_PART={PART} (0 = cm, 1 = am, 2 = ex)")

N_CORES, HID = M.N_CORES, M.HID
if G.N_CORES != N_CORES:
    sys.exit(f"lm.py: the mixers split over {G.N_CORES} cores and the experts over {N_CORES}; "
             "one main-core program needs one count")
if G.QW != HID or G.XN_ELEMS != 1 or G.OG_ELEMS != 1 or HID * 2 != ELEM:
    sys.exit("lm.py: every mixer projection must be K = hidden = 2048 (one 4 KB x element): "
             f"q width {G.QW}, hidden {HID}")
if G.QKVB or G.SANDWICH or G.PTAB_ELEMS != 1 or G.RB <= 1:
    sys.exit("lm.py: carries dx.py's no-bias, no-sandwich, one-record, blocked-walk attention only")
TAPS = QR.TAPS
SCW = 512
SC_ELEMS = HID // SCW
SC_PC = HID // (BAND_ROWS * N_CORES)          # bands per core of a [hidden, hidden] projection
NA_CONV = 3 * SC_PC                           # RTP word 0, part 0
NA_ATTN = G.Q_PC + 2 * G.KV_PC                # RTP word 0, part 1
NB = SC_PC
assert NB == G.O_PC, "lm.py: the conv out and the attention o projections must be the same band count"
QW, KVW = G.QW, G.KVW
ELN, E_A = L.ELN, L.E_A
ACORES, NHL, RB = G.ACORES, G.NHL, G.RB
OGH = min(NHL, G.HPO)
N_OG = NHL // OGH
if ACORES != 4:
    sys.exit(f"lm.py: the shim budget is laid out for 4 attention cores, the spec has {ACORES}")

OS = ["-Os"]
GEMV_DEF = ["-DGEMV_Q4_SCALES_F32=1", "-DGEMV_Q4_PREFIX=gemv_q4s32"]
LN_FLAGS = [f"-DLN_N={HID}", f"-DLN_EPS={G.EPS:g}f"]
SC_FLAGS = [f"-DSC_TAPS={TAPS}", f"-DSC_W={SCW}", "-DSC_PASSTHROUGH=0"]
ATTN_FLAGS = [f"-DATTN_NH={G.NH}", f"-DATTN_KVH={G.KVH}", f"-DATTN_HD={G.HD}", f"-DATTN_ROT={G.ROT}", "-DATTN_GATE=0",
              f"-DATTN_QKNORM={1 if G.QKNORM else 0}", f"-DATTN_QKNORM_POST={1 if G.QKNORM_POST else 0}",
              f"-DATTN_EPS={G.EPS:g}f", f"-DATTN_VEXP={G.VEXP}", f"-DATTN_NHL={G.NHL}", f"-DATTN_RB={G.RB}"]
for _k, _v in QR.probe_env().items():
    if _k not in ("ATTN_RB", "ATTN_FAST"):
        ATTN_FLAGS.append(f"-D{_k}={_v}")

# (the helper cores' kernels take exactly the flags dx.py / cx.py give them, no -Os: the same objects)
# one set of buffer-argument shapes for all three streams (a tap's total only bounds it)
STATE_T = max(L.STATE_BYTES, L.KV_BYTES)
CALL = M.CALL_BYTES
YB = BAND_ROWS * 4
BB_H = M.BB_H


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def lm(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
       srchash: CompileTime[int] = 0):
    elem = np.ndarray[(CALL,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    tab_ty = np.ndarray[(M.TAB_BYTES,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(M.MS_FLOATS,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_T,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]
    # the conv core's elements
    fsc = np.ndarray[(SCW,), np.dtype[np.float32]]
    bsc = np.ndarray[(SCW,), np.dtype[bfloat16]]
    # the attention cores' (dx.py)
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]
    bhd = np.ndarray[(G.HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * G.HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(G.HD,), np.dtype[np.float32]]
    fq = (np.ndarray[(2 * QW,), np.dtype[bfloat16]] if G.VEXP else np.ndarray[(QW,), np.dtype[np.float32]])
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * G.HD,), np.dtype[np.float32]]
    i32 = np.int32

    inc = include_dirs() + [str(GEMV), str(HERE), str(ATTN), str(LN), str(LINL), str(SC)]

    def ef(sym, src, args, flags):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    def ex(sym, args):
        return ef(sym, HERE / f"{sym}.cc", args, OS + GEMV_DEF)

    # main cores
    f_gy = ex("ex_gy", [elem, tab_ty, y_ty, i32])
    f_gu = ex("ex_gu", [elem, tab_ty, ms_ty, i32, i32])
    f_gd = ex("ex_gd", [elem, tab_ty, ms_ty, i32, i32])
    f_px = ex("ex_px", [x_ty, tab_ty, i32])
    f_ph = ex("ex_ph", [x_ty, tab_ty, i32])
    f_hdr = ex("ex_hdr", [elem, ms_ty, i32])
    f_act = ex("ex_act", [ms_ty, y_ty])
    f_acc = ex("ex_acc", [ms_ty, i32])
    f_out = ex("ex_out", [ms_ty, y_ty, i32])
    # norm helper
    f_nr = ef("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = ef("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = ef("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    # conv core
    f_sc = ef("short_conv_step", SC / "sc_elem.cc", [fsc, fsc, fsc, bsc, bsc, bsc, fsc, fsc, bsc, fsc, fsc],
              SC_FLAGS)
    # attention cores
    f_meta = ef("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, pb_ty], ATTN_FLAGS)
    f_q = ef("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = ef("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = ef("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = ef("attn_init", ATTN / "attn_init.cc", [foacc, fml], ATTN_FLAGS)
    f_step = ef("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
    f_stepn = ef("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml, i32], ATTN_FLAGS)
    f_stepb = ef("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * RB) + [fq, foacc, fml, pb_ty, i32],
                 ATTN_FLAGS)
    f_fin = ef("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS)

    # ---- fifos
    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=2)
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_scin = ObjectFifo(fsc, name="scin", depth=6)
    of_scw = ObjectFifo(bsc, name="scw", depth=4)
    of_scy = ObjectFifo(bsc, name="scy", depth=4)
    of_scst = ObjectFifo(fsc, name="scst", depth=4)
    of_ain = ObjectFifo(u8_a, name="ain", depth=max(4, 2 * RB + 2))
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    of_og = [ObjectFifo(og_ty, name=f"og{c}", depth=2) for c in range(ACORES)]

    rtp = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(2, dtype=np.int32), use_write_rtp=True)
           for c in range(N_CORES)]

    # ---- the main cores: one layer of either type per pass
    def main_body(win, xin, yout, my_rtp, tab, ms, f_gy, f_gu, f_gd, f_px, f_ph, f_hdr, f_act, f_acc, f_out):
        def gy_bands(nbands):
            for _ in range_(nbands):
                ye = yout.acquire(1)
                for g in range_(M.NG_H):
                    we = win.acquire(1)
                    f_gy(we, tab, ye, g)
                    win.release(1)
                yout.release(1)
        # mixer stage 1: the acquire FIRST, then the RTP word (the stream writes it before this fill)
        xe = xin.acquire(1)
        na = my_rtp[0]                              # 12 conv | 6 attention
        f_px(xe, tab, 0)
        gy_bands(na)
        xin.release(1)
        # mixer stage 2: out (conv) / o (attention), K = HID
        oe = xin.acquire(1)
        f_px(oe, tab, 0)
        gy_bands(NB)
        xin.release(1)
        # the routed experts (ex.py's body)
        xe = xin.acquire(1)
        f_px(xe, tab, 0)
        xin.release(1)
        for mode in range(2):
            we = win.acquire(1)
            f_hdr(we, ms, mode)
            win.release(1)
        for s in range_(M.NE):
            for _ in range_(M.HB):
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
            for i in range(M.H_ELEMS):
                he = xin.acquire(1)
                f_ph(he, tab, i)
                xin.release(1)
            for j in range_(M.DB):
                for g in range_(M.NG_F):
                    we = win.acquire(1)
                    f_gd(we, tab, ms, g, j)
                    win.release(1)
            f_acc(ms, s)
        f_acc(ms, -1)
        for j in range_(M.DB):
            ye = yout.acquire(1)
            f_out(ms, ye, j)
            yout.release(1)

    # ---- the norm helper (cx.py's / dx.py's at mix: the entry norm, then res and xm)
    def ln_body(ain, aout, f_nr, f_lny, f_lnx):
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)
        e = ain.acquire(5)
        for i in range(2):
            o = aout.acquire(1)
            f_lny(e[0], e[1], e[3], e[4], o, i)
            aout.release(1)
        o = aout.acquire(1)
        f_lnx(e[0], e[1], e[3], e[4], e[2], o)
        aout.release(1)
        ain.release(5)

    # ---- the conv core (cx.py's)
    def sc_body(scin, scw, scy, scst, f_sc):
        for _ in range_(SC_ELEMS):
            w = scw.acquire(TAPS)
            e = scin.acquire(5)
            oy = scy.acquire(1)
            os_ = scst.acquire(2)
            f_sc(e[0], e[1], e[2], w[0], w[1], w[2], e[3], e[4], oy, os_[0], os_[1])
            scst.release(2)
            scy.release(1)
            scin.release(5)
            scw.release(TAPS)

    # ---- the attention cores (dx.py's _attn: no bias, one position record, blocked walk)
    def _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0):
        e = ain.acquire(2)
        f_meta(e[0], e[1], qn, kn, cs, pb)
        ain.release(2)
        for h in range_(G.Q_AIN_ELEMS):
            e = ain.acquire(1)
            f_q(e, qn, cs, qs, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_k(e, kn, cs, tmp, kout, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_v(e, vout, h)
            ain.release(1)
        if aout is not None:
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = kout[j]
            aout.release(1)
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = vout[j]
            aout.release(1)
        f_init(oacc, ml)
        for _ in range_(pb[4]):
            e = ain.acquire(2 * RB)
            f_stepb(*[e[i] for i in range(2 * RB)], qs, oacc, ml, pb, h0)
            ain.release(2 * RB)
        for _ in range_(pb[5]):
            e = ain.acquire(2)
            f_step(e[0], e[1], qs, oacc, ml, pb, h0)
            ain.release(2)
        f_stepn(kout, vout, qs, oacc, ml, h0)
        for hp in range_(N_OG):
            o = ogout.acquire(1)
            f_fin(oacc, ml, o, hp)
            ogout.release(1)

    def attn_body(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
        _attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
              f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, 0)

    def make_attn_body(c):
        h0 = c * NHL

        def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                 f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
            _attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pb,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, h0)
        return body

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}")]

    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb]
    mfns = [f_gy, f_gu, f_gd, f_px, f_ph, f_hdr, f_act, f_acc, f_out]
    workers = [Worker(ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx],
                      tile=Tile(0, 3), stack_size=0x1800)]
    workers += [Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(), rtp[c],
                                           Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}")] + mfns,
                       tile=Tile(c, 2), stack_size=0x1400)
                for c in range(N_CORES)]
    workers.append(Worker(sc_body, fn_args=[of_scin.cons(), of_scw.cons(), of_scy.prod(), of_scst.prod(), f_sc],
                          tile=Tile(2, 4), stack_size=0x1800))
    workers.append(Worker(attn_body, fn_args=[of_ain.cons(), of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=Tile(2, 3), stack_size=0x1800))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn_body(c), fn_args=[of_ain.cons(), of_og[c].prod()] + abufs(c) + afns,
                              tile=Tile(2 + c, 3), stack_size=0x1800))

    # ---- part 0: the conv block (cx.py's _sequence through step 5)
    def conv_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                      scin_p, scw_p, scy_c, scst_c):
        for c in range(N_CORES):
            rtp[c][0] = NA_CONV
            rtp[c][1] = 0
        SCB, SCYB, TAPB = SCW * 4, SCW * 2, SCW * 2
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELN), wait=True, group=tg_ln)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            for base in (L.POOL_SC_B, L.POOL_SC_C, L.POOL_SC_U):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, base + c * SC_PC * BB_H, SC_PC * BB_H))
            for dst in (L.AD_B, L.AD_C, L.AD_U):
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, dst + c * SC_PC * YB, SC_PC * YB))
        tg_ln.finish()
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELEM), wait=True, group=tg_x)
        ps_in, ps_out = Pipeline(3), Pipeline(3)
        for i in range(SC_ELEMS):
            for k in range(TAPS):
                ps_in.fill(scw_p, a_consts, bt(L.CD_BYTES, L.CD_CONV + (k * HID + i * SCW) * 2, TAPB))
        py.finish(*y_conss)                                   # B, C, u are in DDR
        for i in range(SC_ELEMS):
            ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_B + i * SCB, SCB))
            ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_C + i * SCB, SCB))
            ps_in.fill(scin_p, a_act, bt(L.AD_BYTES, L.AD_U + i * SCB, SCB))
            ps_in.fill(scin_p, a_state, bt(STATE_T, i * SCB, SCB))
            ps_in.fill(scin_p, a_state, bt(STATE_T, HID * 4 + i * SCB, SCB))
            ps_out.drain(scy_c, a_act, bt(L.AD_BYTES, L.AD_Y + i * SCYB, SCYB))
            ps_out.drain(scst_c, a_state, bt(STATE_T, i * SCB, SCB))
            ps_out.drain(scst_c, a_state, bt(STATE_T, HID * 4 + i * SCB, SCB))
        ps_out.finish()
        ps_in.finish()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_Y, ELEM), wait=True, group=tg_x)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_SC_OUT + c * SC_PC * BB_H, SC_PC * BB_H))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT + c * SC_PC * YB, SC_PC * YB))
        norm_tail(c_xres, a_consts, a_act, lni, lno, py)
        pw.finish()
        tg_x.finish()

    def norm_tail(c_xres, a_consts, a_act, lni, lno, py):
        """res = x + out; xm = ffn_norm(res) -> act (both mixers' step 5)."""
        tg = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM, ELN), wait=True, group=tg)
        py.finish()                                           # out is in DDR
        lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT, HID * 4), wait=True, group=tg)
        tg.finish()

    # ---- part 1: the attention block (dx.py's _sequence through step 5)
    def attn_sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                      ain_p, aout_c, og_cs):
        for c in range(N_CORES):
            rtp[c][0] = NA_ATTN
            rtp[c][1] = 0
        tg_ln = TaskGroup()
        lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln)
        lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_LNW, ELN), wait=True, group=tg_ln)
        lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELN), wait=True, group=tg_ln)
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_Q + c * G.Q_PC * BB_H, G.Q_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_K + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_V + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_Q + c * G.Q_PC * YB, G.Q_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + c * G.KV_PC * YB, G.KV_PC * YB))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4 + c * G.KV_PC * YB, G.KV_PC * YB))
        tg_ln.finish()
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, ELEM), wait=True, group=tg_x)
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        pa_out.drain(aout_c, a_kv, bt(STATE_T, L.KV_ROW, L.KV_ROW))               # [k' | v'] -> row pos (attnpos)
        for c in range(ACORES):
            pa_out.drain(og_cs[c], a_act, bt(L.AD_BYTES, L.AD_OG + c * NHL * G.HD * 2, NHL * G.HD * 2))
        pa_in.fill(ain_p, a_consts, bt(L.CD_BYTES, L.CD_META, E_A))               # [qn | kn]
        pa_in.fill(ain_p, a_ptab, bt(L.PTAB_BYTES, L.PTAB_ROW, L.PTAB_ROW))       # the position record (attnpos)
        py.finish(*y_conss)                                   # q, k, v are in DDR
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_Q, QW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN, KVW * 4))
        pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + KVW * 4, KVW * 4))
        pa_in.fill(ain_p, a_kv, bt(STATE_T, 0, L.KV_ROW))                         # the window (attnpos)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_O + c * G.O_PC * BB_H, G.O_PC * BB_H))
            py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT + c * G.O_PC * YB, G.O_PC * YB))
        pa_out.finish()                                       # og (and the new cache row) are in DDR
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OG, ELEM), wait=True, group=tg_x)
        norm_tail(c_xres, a_consts, a_act, lni, lno, py)
        pw.finish()
        pa_in.finish()
        tg_x.finish()

    # ---- part 2: the routed experts (ex.py's sequence)
    def expert_sequence(a_pool, c_xres, a_act, w_prods, x_prod, y_conss):
        pw, px, py = Pipeline(3), Pipeline(3), Pipeline(3)
        px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_act, bt(L.AD_BYTES, L.AD_ROUT, CALL))
            pw.fill(w_prods[c], a_act, bt(L.AD_BYTES, L.AD_RES + c * M.ROWS_PC * 4, CALL))
        for s in range(M.NE):
            up = L.POOL_EXP + s * 2 * M.UP_BYTES     # slot s compiled as expert s (moeroute2)
            for c in range(N_CORES):                 # the h drains FIRST (ex.py)
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_HP + c * M.HID_PC * 4, M.HID_PC * 4))
            for b in range(M.HB):
                for c in range(N_CORES):
                    band = c * M.HB + b
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, up + band * BB_H, BB_H))
                    pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, up + M.UP_BYTES + band * BB_H, BB_H))
            py.finish(*y_conss)
            px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_HP, M.H_ELEMS * ELEM))
            down = L.POOL_DOWN_EXP + s * M.UP_BYTES
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, down + c * M.DB * M.BB_F, M.DB * M.BB_F))
        for c in range(N_CORES):
            py.drain(y_conss[c], c_xres, bt(HID, c * M.ROWS_PC, M.ROWS_PC))
        pw.finish()
        px.finish()
        py.finish()

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                 scin_p, scw_p, scy_c, scst_c, ain_p, aout_c, og_cs):
        if part == 0:
            conv_sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                          scin_p, scw_p, scy_c, scst_c)
        elif part == 1:
            attn_sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                          ain_p, aout_c, og_cs)
        else:
            expert_sequence(a_pool, c_xres, a_act, w_prods, x_prod, y_conss)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [of_w[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_scin.prod(tile=Tile(2, 0)), of_scw.prod(tile=Tile(3, 0)),
                            of_scy.cons(tile=Tile(1, 0)), of_scst.cons(tile=Tile(2, 0)),
                            of_ain.prod(tile=Tile(4, 0)), of_aout.cons(tile=Tile(3, 0)),
                            [of_og[c].cons(tile=Tile(4 + c, 0)) for c in range(ACORES)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = lm
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in SC.glob("*.cc")) + sorted(f.read_bytes() for f in SC.glob("*.h"))
                + [(LN / "ln.h").read_bytes(), (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (DESIGNS.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
# NOT `part`: the three builds must emit the same image (the cores do not depend on it), and
# the srchash is baked into it. `part` is a CompileTime arg because the instruction stream is.
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
