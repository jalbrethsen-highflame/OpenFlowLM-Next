"""LFM2-MoE (LFM2-8B-A1B, LFM2.5-24B-A2B): LFM2's short-conv / attention hybrid with the FFN
replaced by routed experts after the first `num_dense_layers` layers.

Three layer types, one pool size:

    short_conv_dense   the leading dense layers: LFM2's `cx` unchanged (conv + dense FFN)
    short_conv         `cxm` (cx.py, CX_MIX=1: the conv block, the residual and the FFN norm,
                       nothing after) -> host router -> `ex` (the routed experts)
    full_attention     `dxm` (dx.py, DX_MIX=1: the same for an attention block) -> host
                       router -> `ex`

The mixer dispatches stop where the dense layer would start its FFN: `res` (f32, the
residual after the mixer) and `xm` (bf16, ffn_norm(res)) are in `act`. The router runs on
the host between the two dispatches -- it is a 2048 x 64 f32 GEMV, a sigmoid, a biased
top-k and a renormalisation, microseconds on a CPU core -- and writes the routing record
the expert dispatch reads (`hostroute`, src/open_qwen36/core.cpp). The record's layout is
qwen36moe's: idx u32[8] at ROUT_IDX_OFF, the weights f32[topk] right after it.

`ex` (designs/lfm2moe/ex.py) is the routed-expert block for an expert width that is a
whole number of 64-row bands per core (OPEN-MOE-WIDE-FF): each expert is stored as plain
std_perm bands, [up | gate] per expert from POOL_EXP and the down slices from
POOL_DOWN_EXP, so the stream_patch moeroute2 geometry (with its `base`) re-points it.

GGUF-direct only: the engine packs the GGUF's Q4_0 blocks straight into the pool. The pool
uses the q4nx chunk layout (bf16 scales, 5120 B) by default: the GEMV narrows f32 scale
planes to bf16 on load anyway, so the f32 layout (6144 B, OPEN_LFM2MOE_QUANT=q4_1_f32 at
export) computes the same bits from 20 % more bytes -- and the prefill GEMMs read only the
bf16 one. Every layout here is computed at QUANT whatever the spec says, and the manifest's
top level and its `gguf` twin are the same kernel set.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from . import dense as D
from . import lfm2 as LF
from .attnknobs import probe_env  # noqa: F401  (cache.py / dx.py read it off the family module)
from .catalogue import OpRangeError, check_buffer_args
from .qwen36moe import BAND_ROWS, ELEM, MB, ROUT_IDX_OFF, q4_bytes, q4_chunks, roundup, tab_bytes
from .qwen36moe import Q8_CHUNK, Q8_CHUNK_F32
from .spec import FULL, SHORT_CONV, ModelSpec

import os as _os
QUANT = _os.environ.get("OPEN_LFM2MOE_QUANT", "q4_1")      # the pool's scale layout (see above)
if QUANT not in ("q4_1", "q4_1_f32"):
    raise OpRangeError(f"OPEN_LFM2MOE_QUANT={QUANT!r} (q4_1 | q4_1_f32)")
F32 = QUANT == "q4_1_f32"
Q8_CH = Q8_CHUNK_F32 if F32 else Q8_CHUNK
SFX = "_f32" if F32 else ""
band_bytes = D.band_bytes    # noqa: F401  (dx.py / cx.py: band_bytes(K, quant))
chunk_bytes = D.chunk_bytes  # noqa: F401  (dx.py / cx.py: chunk_bytes(quant))
TAPS = LF.TAPS
DENSE_CONV = "short_conv_dense"     # the manifest's name for the leading dense layers
MOE_PER_CALL = 2                    # chunks per expert weight element (12 KB at f32 scales)
L1_BUDGET = 60 * 1024
F32_STACK = 0x1400


# The head is the q8 GEMV (designs/lm_head_q8, f32 scales): LFM2's GGUFs ship the tied
# embedding at Q6_K, and the q4_1 head the dense recipe would use costs a logits correlation
# of 0.995 against it on its own (model/compare_lfm2moe.py) -- enough to flip the argmax.
# `lmhead_q8_gguf` re-quantizes each 32-block to 8 bits at pack time.
LM_BAND_ROWS = 128


def lm_q8(spec: ModelSpec) -> dict:
    rows = D.lm_rows(lfm2_view(spec))
    if rows % LM_BAND_ROWS:
        raise OpRangeError(f"lfm2moe: {rows} head rows are not whole 128-row bands")
    band = 4 * (spec.hidden // 256) * Q8_CH
    bands = rows // LM_BAND_ROWS
    return {"rows": rows, "band": band, "bands": bands,
            "pool": roundup(roundup(bands, 8) * band, MB)}


def lfm2_view(spec: ModelSpec) -> ModelSpec:
    """The spec as the LFM2 recipe sees it: the dense layers' shapes, f32-scale chunks."""
    return replace(spec, family="lfm2", num_experts=0, experts_per_tok=0, moe_intermediate=0,
                   num_dense_layers=0, quant=QUANT)


def moe_cores(spec: ModelSpec) -> int:
    """Cores the expert block splits over: whole 64-row bands of the expert width AND of the
    hidden per core. 8 for the 24B (1536 = 24 bands, 2048 = 32); 4 for the 8B (1792 = 28).
    OPEN_LFM2MOE_CORES caps it (an experiment knob: a different core count is a different
    kernel set, so it is part of the build dir name)."""
    import os
    cap = int(os.environ.get("OPEN_LFM2MOE_CORES", 8))
    for n in range(min(cap, 8), 0, -1):
        if (spec.moe_intermediate // BAND_ROWS) % n == 0 and (spec.hidden // BAND_ROWS) % n == 0:
            return n
    return 1


@dataclass(frozen=True)
class MoeGeometry:
    N_CORES: int; HID: int; FF: int; NE: int; E: int
    PER_CALL: int; CH: int; CALL_BYTES: int
    BB_H: int; NG_H: int; PB_H: int          # an up / gate band (K = HID): bytes, elements, chunks
    BB_F: int; NG_F: int; PB_F: int          # a down band (K = FF)
    HB: int; DB: int                         # up / gate bands, down bands per core
    HID_PC: int; ROWS_PC: int                # expert-hidden rows, output rows per core
    H_ELEMS: int                             # 4 KB x elements the f32 expert hidden spans
    UP_BYTES: int                            # one expert's up (= gate = down)
    TAB_X: int; TAB_BYTES: int; H_TAB_OFF: int
    MS_RW: int; MS_XR: int; MS_ACC: int; MS_U: int; MS_G: int; MS_YD: int; MS_FLOATS: int
    L1: int


def moe_geometry(spec: ModelSpec) -> MoeGeometry:
    n = moe_cores(spec)
    hid, ff = spec.hidden, spec.moe_intermediate
    ch = D.chunk_bytes(QUANT)
    call = MOE_PER_CALL * ch
    bb_h, bb_f = D.band_bytes(hid, QUANT), D.band_bytes(ff, QUANT)
    hb, db = ff // BAND_ROWS // n, hid // BAND_ROWS // n
    rows_pc = db * BAND_ROWS
    tab_x = tab_bytes(hid)
    tabb = tab_x + tab_bytes(ff)
    ms = {}
    off = 0
    for name, size in (("rw", 32), ("xr", rows_pc), ("acc", rows_pc), ("u", BAND_ROWS), ("g", BAND_ROWS),
                       ("yd", rows_pc)):
        ms[name] = off
        off += size
    l1 = tabb + off * 4 + 2 * call + 2 * ELEM + 2 * BAND_ROWS * 4 + F32_STACK
    return MoeGeometry(
        N_CORES=n, HID=hid, FF=ff, NE=spec.experts_per_tok, E=spec.num_experts,
        PER_CALL=MOE_PER_CALL, CH=ch, CALL_BYTES=call,
        BB_H=bb_h, NG_H=bb_h // call, PB_H=bb_h // ch, BB_F=bb_f, NG_F=bb_f // call, PB_F=bb_f // ch,
        HB=hb, DB=db, HID_PC=hb * BAND_ROWS, ROWS_PC=rows_pc,
        H_ELEMS=roundup(ff * 4, ELEM) // ELEM, UP_BYTES=q4_bytes(ff, hid, QUANT),
        TAB_X=tab_x, TAB_BYTES=tabb, H_TAB_OFF=tab_x,
        MS_RW=ms["rw"], MS_XR=ms["xr"], MS_ACC=ms["acc"], MS_U=ms["u"], MS_G=ms["g"], MS_YD=ms["yd"],
        MS_FLOATS=off, L1=l1,
    )


# ---- the prefill route's token-batched experts (designs/lfm2moe/eb.py)
EB_NT = int(_os.environ.get("OPEN_LFM2MOE_EB_NT", 32))   # token slots per expert visit (moe_batch's NT).
# eb streams a whole expert's weights per visit and is bound by that stream (the arithmetic
# removed, 8.33 of 8.47 ms), so tokens per visit is the lever: 16 -> 32 halves it per token
EB_STREAMS = (64, 16)        # slot counts, one instruction stream each over one image


def eb_ok(spec: ModelSpec) -> bool:
    """eb.py needs the bf16-scale pool and an expert width of whole 128-row band pairs over
    a column's four rows (1536 is three pairs a row; the 8B's 1792 is not whole)."""
    return (not F32 and spec.moe_intermediate % 512 == 0 and spec.hidden % 256 == 0
            and spec.num_experts >= max(EB_STREAMS))


PROMPT_T = 256              # tokens per prompt GEMM (gemm_q4_prefill's T; one image serves every shape)


def prompt_gemm(spec: ModelSpec) -> dict:
    """The prompt route's batched mixers (Core::prompt_mixers): every projection of a mixer --
    and of the dense layers' FFN -- as a gemm_q4_prefill dispatch over PROMPT_T tokens, reading
    its weights straight out of the layer pool (an xrt sub-buffer at the region's offset: the
    pool's band order IS the GEMM's weight order). The norms, the 3-tap conv, q/k norm + RoPE +
    attention, silu and the residuals run on the host between them (block_host.cpp)."""
    Lm, G = layout(spec), geometry(spec)
    hid, ff, qw, kvw = spec.hidden, spec.intermediate, G.QW, G.KVW
    regions = {"conv_in": (Lm.POOL_SC_B, 3 * hid, hid), "conv_out": (Lm.POOL_SC_OUT, hid, hid),
               "qkv": (Lm.POOL_Q, qw + 2 * kvw, hid), "o": (Lm.POOL_O, hid, qw),
               "ffn_gate": (Lm.POOL_SC_GATE, ff, hid), "ffn_up": (Lm.POOL_SC_UP, ff, hid),
               "ffn_down": (Lm.POOL_SC_DOWN, hid, ff)}
    # the regions must be the contiguous band-order matrices the GEMM reads
    assert Lm.POOL_SC_C == Lm.POOL_SC_B + q4_bytes(hid, hid, QUANT) and Lm.POOL_SC_U == Lm.POOL_SC_C + q4_bytes(hid, hid, QUANT)
    assert Lm.POOL_K == Lm.POOL_Q + q4_bytes(qw, hid, QUANT) and Lm.POOL_V == Lm.POOL_K + q4_bytes(kvw, hid, QUANT)
    for name, (off, n, k) in regions.items():
        if n % 256 or k % 256:
            raise OpRangeError(f"lfm2moe prompt GEMM {name}: [{n}, {k}] is not a multiple of 256")
        if off % 4096:
            raise OpRangeError(f"lfm2moe prompt GEMM {name}: pool offset {off} is not page aligned (sub-buffer)")
    shapes = sorted({(n, k) for _, n, k in regions.values()})
    return {
        "t": PROMPT_T,
        "regions": {k: list(v) for k, v in regions.items()},
        "kernels": {f"{n}x{k}": f"gemm_n{n}_k{k}" for n, k in shapes},
        "x": {str(k): f"gemm_x_k{k}" for k in sorted({k for _, k in shapes})},
        "y": {str(n): f"gemm_y_n{n}" for n in sorted({n for n, _ in shapes})},
        "kinds": {DENSE_CONV: "conv_dense", SHORT_CONV: "conv", FULL: "attn"},
        "tensors": {"attn_norm": "blk.{l}.attn_norm.weight", "ffn_norm": "blk.{l}.ffn_norm.weight",
                    "conv": "blk.{l}.shortconv.conv.weight", "q_norm": "blk.{l}.attn_q_norm.weight",
                    "k_norm": "blk.{l}.attn_k_norm.weight"},
        "hid": hid, "ff": ff, "nh": spec.num_heads, "kvh": spec.num_kv_heads, "hd": spec.head_dim,
        "rot": spec.rotary_dim, "taps": TAPS, "eps": spec.norm_eps, "kv_row_elems": Lm.KV_ROW // 2,
        "_shapes": shapes,
    }


def eb_geometry(spec: ModelSpec) -> dict:
    hid, ff = spec.hidden, spec.moe_intermediate
    return {"nt": EB_NT,
            "x_bytes": lambda s: s * hid * EB_NT * 2,
            "h_bytes": lambda s: s * ff * EB_NT * 2,
            "y_bytes": lambda s: s * hid * EB_NT * 4}


def _check(spec: ModelSpec) -> None:
    if spec.family != "lfm2moe":
        raise OpRangeError(f"lfm2moe recipe on a {spec.family!r} spec")
    if not spec.num_experts or not spec.experts_per_tok or not spec.moe_intermediate:
        raise OpRangeError("lfm2moe: no routed experts in the spec")
    if spec.shared_expert_intermediate:
        raise OpRangeError("lfm2moe: a shared expert; the expert block has no slot for one")
    if spec.experts_per_tok > 8:
        raise OpRangeError(f"lfm2moe: top-{spec.experts_per_tok}; the routing record holds 8")
    if any(spec.layer_types[l] != SHORT_CONV for l in range(spec.num_dense_layers)):
        raise OpRangeError("lfm2moe: a leading dense layer that is not a short-conv layer; only "
                           "`cx` carries the dense FFN here")
    LF._check(lfm2_view(spec))
    G = moe_geometry(spec)
    for what, v in (("expert width", G.FF), ("hidden", G.HID)):
        if v % 256:
            raise OpRangeError(f"lfm2moe: {what} {v} is not a whole number of 256-wide k-tiles")
    if G.BB_H % G.CALL_BYTES or G.BB_F % G.CALL_BYTES:
        raise OpRangeError("lfm2moe: a band is not a whole number of expert weight elements")
    if G.L1 > L1_BUDGET:
        raise OpRangeError(f"lfm2moe: the expert block's main core needs {G.L1} B of L1 ({L1_BUDGET} budget)")


@dataclass(frozen=True)
class Lfm2MoeLayout:
    """LFM2's layout (dense + short-conv) plus the expert block's regions. Anything not
    defined here comes from the LFM2 layout underneath, so dx.py and cx.py read the same
    offsets they read for LFM2 -- only POOL_BYTES / AD_BYTES grow."""
    lfm2: LF.Lfm2Layout
    POOL_EXP: int; POOL_DOWN_EXP: int; POOL_END: int; POOL_BYTES: int
    AD_ROUT: int; AD_HP: int; AD_BYTES: int

    def __getattr__(self, name: str):
        try:
            return getattr(object.__getattribute__(self, "lfm2"), name)
        except AttributeError:
            raise AttributeError(f"{type(self).__name__} has no {name!r}") from None

    def constants(self) -> dict[str, int]:
        out = self.lfm2.constants()
        out.update({k: v for k, v in self.__dict__.items() if k != "lfm2"})
        return out


def layout(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2MoeLayout:
    _check(spec)
    Lx = LF.layout(lfm2_view(spec), max_ctx)
    G = moe_geometry(spec)
    # the experts sit past the larger mixer: the conv block's four projections, or the
    # attention block's q k v o. The dense layers' FFN overlaps them -- a dense layer has no
    # experts, so one pool size serves all three types.
    mixer_end = max(Lx.POOL_SC_UP, Lx.dense.POOL_UP)
    pexp = roundup(mixer_end, MB)
    pdown = pexp + G.E * 2 * G.UP_BYTES
    pend = pdown + G.E * G.UP_BYTES
    pool_bytes = max(Lx.POOL_BYTES, roundup(pend, MB))
    if pool_bytes >= 1 << 32:
        raise OpRangeError("lfm2moe: the layer pool passes 4 GB; the stream patch's offsets are 32-bit")
    # act: the router record and the expert hidden past LFM2's regions; and the residual's
    # header fills (one CALL_BYTES element per core from AD_RES + c * ROWS_PC * 4) must stay
    # inside the buffer
    off = Lx.AD_BYTES
    rout, off = off, off + G.CALL_BYTES
    hp, off = off, off + G.H_ELEMS * ELEM
    ad = roundup(max(off, Lx.AD_RES + (G.N_CORES - 1) * G.ROWS_PC * 4 + G.CALL_BYTES), ELEM)
    return Lfm2MoeLayout(lfm2=Lx, POOL_EXP=pexp, POOL_DOWN_EXP=pdown, POOL_END=pend, POOL_BYTES=pool_bytes,
                         AD_ROUT=rout, AD_HP=hp, AD_BYTES=ad)


def geometry(spec: ModelSpec) -> D.DenseGeometry:
    """The mixer designs' geometry: LFM2's (which is the dense one) at f32-scale chunks."""
    return D.geometry(lfm2_view(spec))


@dataclass(frozen=True)
class Lfm2MoeRecipe:
    spec: ModelSpec
    layout: Lfm2MoeLayout
    geo: D.DenseGeometry
    moe: MoeGeometry
    max_ctx: int = 4096


def recipe(spec: ModelSpec, max_ctx: int = 4096) -> Lfm2MoeRecipe:
    _check(spec)
    return Lfm2MoeRecipe(spec=spec, layout=layout(spec, max_ctx), geo=geometry(spec),
                         moe=moe_geometry(spec), max_ctx=max_ctx)


def _gguf_names(ops: list[dict]) -> list[dict]:
    return [dict(op) for op in ops]


def pack_plan(spec: ModelSpec) -> dict:
    """Three layer types. The tensor names are the HF-style ones `GgufFile::gguf_name` maps
    (model.layers.N.shortconv.* passes through as blk.N.shortconv.*), except where LFM2's
    GGUF names have no HF twin: the experts, the final norm (token_embd_norm) and the
    embedding."""
    V = lfm2_view(spec)
    Lm = layout(spec)
    G = moe_geometry(spec)
    base = LF.pack_plan(V)
    sc_dense = base["layer_types"][SHORT_CONV]
    attn = base["layer_types"][FULL]
    ffn_tensors = ("mlp.up_proj.weight", "mlp.gate_proj.weight", "mlp.down_proj.weight")

    def gguf_ops(entry):
        return {"pool": [dict(op, op="std_perm_gguf") if op["op"] == "std_perm" else op for op in entry["pool"]],
                "consts": entry["consts"]}

    sc_dense, attn = gguf_ops(sc_dense), gguf_ops(attn)

    def no_ffn(entry):
        return {"pool": [op for op in entry["pool"] if not op["tensor"].endswith(ffn_tensors)],
                "consts": entry["consts"]}

    pre = "blk.{l}."
    nch_ug, nch_d = q4_chunks(G.FF, G.HID, QUANT), q4_chunks(G.HID, G.FF, QUANT)
    experts = []
    for e in range(G.E):
        up = Lm.POOL_EXP + e * 2 * G.UP_BYTES
        experts += [
            {"op": "std_perm_gguf", "tensor": pre + "ffn_up_exps.weight", "dst": up,
             "chunk0": e * nch_ug, "nch": nch_ug, "in_dim": G.HID},
            {"op": "std_perm_gguf", "tensor": pre + "ffn_gate_exps.weight", "dst": up + G.UP_BYTES,
             "chunk0": e * nch_ug, "nch": nch_ug, "in_dim": G.HID},
            {"op": "std_perm_gguf", "tensor": pre + "ffn_down_exps.weight", "dst": Lm.POOL_DOWN_EXP + e * G.UP_BYTES,
             "chunk0": e * nch_d, "nch": nch_d, "in_dim": G.FF},
        ]
    sc_moe, attn_moe = no_ffn(sc_dense), no_ffn(attn)
    sc_moe["pool"] = sc_moe["pool"] + experts
    attn_moe["pool"] = attn_moe["pool"] + experts
    types = {DENSE_CONV: sc_dense, SHORT_CONV: sc_moe, FULL: attn_moe}
    return {
        "pool_bytes": Lm.POOL_BYTES, "chunk_bytes": D.chunk_bytes(QUANT),
        "layer_types": {lt: types[lt] for lt in sorted(set(manifest_layers(spec)))},
        "lm_head": {"pool_bytes": lm_q8(spec)["pool"],
                    "ops": [{"op": "lmhead_q8_gguf", "tensor": "lm_head.weight", "dst": 0,
                             "chunk_bytes": Q8_CH, "in_dim": spec.hidden, "rows": lm_q8(spec)["rows"]}]},
        "embed": {"tensor": "model.embed_tokens.weight", "dim": spec.hidden},
        "norm": {"tensor": "token_embd_norm.weight", "bytes": spec.hidden * 2},
    }


def manifest_layers(spec: ModelSpec) -> list[str]:
    """The per-layer type names the manifest lists: the leading dense layers get their own."""
    return [DENSE_CONV if l < spec.num_dense_layers else t for l, t in enumerate(spec.layer_types)]


def one_context(spec: ModelSpec) -> bool:
    """The merged layer image (designs/lfm2moe/lm.py): both mixers and the experts as three
    instruction streams over ONE image, so the layer walk never switches hardware context.
    OPEN_LFM2MOE_ONE_CTX=0 at export builds the three separate images (cx.py / dx.py with
    *_MIX, ex.py) instead. A spec whose experts split over a different core count than its
    mixers (the 8B: 4 against 8) has no single main-core program, so it keeps three images."""
    import os
    if os.environ.get("OPEN_LFM2MOE_ONE_CTX", "1") == "0":
        return False
    return moe_cores(spec) == geometry(spec).N_CORES


def programs(spec: ModelSpec, max_ctx: int = 4096) -> dict:
    V = lfm2_view(spec)
    Lm = layout(spec, max_ctx)
    base = D.programs(V, max_ctx)            # f32: the tail, the globals, ln / lm, the ptab
    out = {"contexts": {"ln": base["contexts"]["ln"], "lm": "lm_head_q8/final.xclbin",
                        "cx": "cx_dense/final.xclbin", "cxm": "short_conv_mix/final.xclbin",
                        "dxm": "dx_mix/final.xclbin", "ex": "ex/final.xclbin"},
           "kernels": {"ln": base["kernels"]["ln"],
                       "lm": {"context": "lm", "insts": "lm_head_q8/insts.bin", "build": "lm_head_q8"}},
           "layer_types": {}, "tail": base["tail"], "globals": dict(base["globals"])}
    out["globals"]["lmpool"] = lm_q8(spec)["pool"]
    sc_args = ["pool", "xres", "consts", "state", "act"]
    ax_args = ["pool", "xres", "consts", "state", "act", "ptab"]
    ex_args = ["pool", "xres", "act"]
    for kn, args in (("cx", sc_args), ("cxm", sc_args), ("dxm", ax_args), ("ex", ex_args)):
        check_buffer_args(kn, args)
    out["kernels"]["cx"] = {"context": "cx", "insts": "cx_dense/insts.bin", "build": "cx_dense"}
    out["kernels"]["cxm"] = {"context": "cxm", "insts": "short_conv_mix/insts.bin", "build": "short_conv_mix"}
    out["kernels"]["dxm"] = {"context": "dxm", "insts": "dx_mix/insts.bin", "patch": "attnpos",
                             "build": "dx_mix", "window": 0}
    out["kernels"]["ex"] = {"context": "ex", "insts": "ex/insts.bin", "patch": "moeroute2", "build": "ex"}
    bufs = {"consts": Lm.CD_BYTES, "act": Lm.AD_BYTES}
    route = {"op": "hostroute", "kernel": "ex", "act_off": Lm.AD_ROUT}
    conv_kn, conv_args = "cxm", sc_args
    if one_context(spec):
        # one image, three streams, one six-argument signature (lm.py)
        for k in ("cxm", "dxm", "ex"):
            del out["contexts"][k]
        out["contexts"]["layer"] = "lm_cm/final.xclbin"
        out["kernels"].pop("cxm")
        out["kernels"]["cm"] = {"context": "layer", "insts": "lm_cm/insts.bin", "build": "lm_cm"}
        out["kernels"]["dxm"] = {"context": "layer", "insts": "lm_am/insts.bin", "patch": "attnpos",
                                 "build": "lm_am", "window": 0}
        out["kernels"]["ex"] = {"context": "layer", "insts": "lm_ex/insts.bin", "patch": "moeroute2", "build": "lm_ex"}
        conv_kn, conv_args, ex_args = "cm", ax_args, ax_args
    lts = set(manifest_layers(spec))
    if DENSE_CONV in lts:
        out["layer_types"][DENSE_CONV] = {
            "buffers": {**bufs, "state": {"kind": "linear", "bytes": Lm.STATE_BYTES}},
            "program": [{"op": "run", "kernel": "cx", "args": sc_args}]}
    if SHORT_CONV in lts:
        out["layer_types"][SHORT_CONV] = {
            "buffers": {**bufs, "state": {"kind": "linear", "bytes": Lm.STATE_BYTES}},
            "program": [{"op": "run", "kernel": conv_kn, "args": conv_args}, route,
                        {"op": "run", "kernel": "ex", "args": ex_args}]}
    if FULL in lts:
        out["layer_types"][FULL] = {
            "buffers": {**bufs, "state": {"kind": "kv", "row": Lm.KV_ROW}},
            "program": [{"op": "run", "kernel": "dxm", "args": ax_args}, route,
                        {"op": "run", "kernel": "ex", "args": ex_args}]}
    if eb_ok(spec) and one_context(spec):
        # the merged image runs mixer + experts as one core pass, so the prompt route's
        # per-token mixers are the standalone images (cx.py / dx.py *_MIX)
        out["contexts"]["cxm"] = "short_conv_mix/final.xclbin"
        out["contexts"]["dxm_p"] = "dx_mix/final.xclbin"
        out["kernels"]["cxm"] = {"context": "cxm", "insts": "short_conv_mix/insts.bin", "build": "short_conv_mix"}
        out["kernels"]["dxm_p"] = {"context": "dxm_p", "insts": "dx_mix/insts.bin", "patch": "attnpos",
                                   "build": "dx_mix", "window": 0}
        if SHORT_CONV in out["layer_types"]:
            out["layer_types"][SHORT_CONV]["prompt"] = [{"op": "run", "kernel": "cxm", "args": sc_args}]
        if FULL in out["layer_types"]:
            out["layer_types"][FULL]["prompt"] = [{"op": "run", "kernel": "dxm_p", "args": ax_args}]
    if eb_ok(spec):
        # the prompt route's batched mixers: one image for every GEMM shape
        P = prompt_gemm(spec)
        for n, k in P["_shapes"]:
            name = f"gemm_n{n}_k{k}"
            out["contexts"].setdefault("gemm", f"{name}/final.xclbin")
            out["kernels"][name] = {"context": "gemm", "insts": f"{name}/insts.bin", "build": name}
            out["globals"][f"gemm_x_k{k}"] = k * PROMPT_T * 2
            out["globals"][f"gemm_y_n{n}"] = n * PROMPT_T * 4
    if eb_ok(spec):
        # the prompt route's token-batched experts: every stream on one image (the core
        # program does not depend on the slot count), x / h / y sized for the widest
        E = eb_geometry(spec)
        smax = max(EB_STREAMS)
        out["contexts"]["eb"] = f"eb{smax}/final.xclbin"
        for sl in EB_STREAMS:
            out["kernels"][f"eb{sl}"] = {"context": "eb", "insts": f"eb{sl}/insts.bin", "patch": "moebatch",
                                         "build": f"eb{sl}"}
        out["globals"].update({"ebx": E["x_bytes"](smax), "ebh": E["h_bytes"](smax), "eby": E["y_bytes"](smax)})
    if spec.quant != "q4_1_f32":
        # the engine reads `gguf` for a model.gguf: the same kernel set (this family is
        # GGUF-direct only), so the twin is this section again
        out["gguf"] = {k: out[k] for k in ("contexts", "kernels", "layer_types", "tail", "globals")}
    return out


def builds(spec: ModelSpec) -> dict[str, dict]:
    V = lfm2_view(spec)
    db = D.builds(V)
    h, ff = spec.hidden, spec.intermediate
    f32 = {"GEMV_SCALES_F32": "1"} if F32 else {}
    sc_env = {**f32, "SC_HID": str(h), "SC_TAPS": str(TAPS)}
    head = {"design": "lm_head_q8/lm_head_q8.py",
            "build_dir": f"lm_head_q8/build_{lm_q8(spec)['rows']}_k{h}{SFX}",
            "env": {"LMHEAD_N": str(lm_q8(spec)["rows"]), "LMHEAD_K": str(h), "LMHEAD_CORES": "8",
                    **({"LMHEAD_SCALES_F32": "1"} if F32 else {})}}
    if one_context(spec):
        tag = f"h{h}_ff{spec.moe_intermediate}_e{spec.num_experts}_k{spec.experts_per_tok}_c{moe_cores(spec)}"
        b = {"cx_dense": {"design": "short_conv/cx.py", "build_dir": f"short_conv/build_lfm2moe_h{h}_ff{ff}{SFX}",
                                "env": sc_env}}
        for name, part in (("lm_cm", 0), ("lm_am", 1), ("lm_ex", 2)):
            b[name] = {"design": "lfm2moe/lm.py", "build_dir": f"lfm2moe/build_lm{part}_{tag}{SFX}",
                       "env": {"LM_PART": str(part), "OPEN_LFM2MOE_CORES": str(moe_cores(spec))}}
        b["ln"] = db["ln"]
        b["lm_head_q8"] = head
        if eb_ok(spec):      # the prompt route's standalone mixers (programs())
            b["short_conv_mix"] = {"design": "short_conv/cx.py", "build_dir": f"short_conv/build_lfm2moe_h{h}_mix{SFX}",
                                   "env": {**sc_env, "CX_MIX": "1"}}
            b["dx_mix"] = {"design": "dense/dx.py", "build_dir": f"dense/build_lfm2moe_h{h}_mix{SFX}",
                           "env": {**f32, "DX_MIX": "1"}}
        return _with_eb(spec, b)
    return _with_eb(spec, {
        "cx_dense": {"design": "short_conv/cx.py", "build_dir": f"short_conv/build_lfm2moe_h{h}_ff{ff}{SFX}",
                           "env": sc_env},
        "short_conv_mix": {"design": "short_conv/cx.py", "build_dir": f"short_conv/build_lfm2moe_h{h}_mix{SFX}",
                           "env": {**sc_env, "CX_MIX": "1"}},
        "dx_mix": {"design": "dense/dx.py", "build_dir": f"dense/build_lfm2moe_h{h}_mix{SFX}",
                   "env": {**f32, "DX_MIX": "1"}},
        "ex": {"design": "lfm2moe/ex.py",
               "build_dir": f"lfm2moe/build_h{h}_ff{spec.moe_intermediate}_e{spec.num_experts}_k{spec.experts_per_tok}"
                            f"_c{moe_cores(spec)}{SFX}",
               "env": {"OPEN_LFM2MOE_CORES": str(moe_cores(spec))}},
        "ln": db["ln"],
        "lm_head_q8": head,
    })


def _with_eb(spec: ModelSpec, b: dict) -> dict:
    if eb_ok(spec):
        for n, k in prompt_gemm(spec)["_shapes"]:
            # GQP_BFP: mm.cc on the bfp16 MAC array (the bf16 mmul runs on the vector unit)
            b[f"gemm_n{n}_k{k}"] = {"design": "gemm_q4_prefill/gemm_q4_prefill.py",
                                    "build_dir": f"gemm_q4_prefill/build_n{n}_k{k}_t{PROMPT_T}_bfp_pipe",
                                    "env": {"GQP_N": str(n), "GQP_K": str(k), "GQP_T": str(PROMPT_T),
                                            "GQP_BFP": "1", "GQP_PIPE": "1"}}
        for s in EB_STREAMS:
            # EB_BFP: the expert products on the bfp16 MAC array (eb_step_bfp.cc), 2x the bf16 path
            b[f"eb{s}"] = {"design": "lfm2moe/eb.py",
                           "build_dir": f"lfm2moe/build_eb{s}_h{spec.hidden}_ff{spec.moe_intermediate}_nt{EB_NT}_bfp",
                           "env": {"EB_SLOTS": str(s), "EB_BFP": "1"}}
    return b


def manifest_layout(spec: ModelSpec, max_ctx: int) -> dict:
    V = lfm2_view(spec)
    out = D.manifest_layout(V, max_ctx)
    Lm, G = layout(spec, max_ctx), moe_geometry(spec)
    out["pool_bytes"] = Lm.POOL_BYTES
    out["chunk_bytes"] = D.chunk_bytes(QUANT)
    out["lmhead_pool_bytes"] = lm_q8(spec)["pool"]
    out["lmhead_chunk_bytes"] = Q8_CH
    # moeroute2's geometry (stream_patch::MoeGeometry): routed [up | gate] per expert from
    # `base`, the down slices from pool_down; nothing shared, so share_* mark the pool end
    out["moe"] = {"experts": G.E, "topk": G.NE, "stripe": G.BB_H, "up_bytes": G.UP_BYTES,
                  "down_core": G.DB * G.BB_F, "base": Lm.POOL_EXP, "pool_down": Lm.POOL_DOWN_EXP,
                  "share_up": Lm.POOL_END, "share_gate": Lm.POOL_END, "share_down": Lm.POOL_END}
    # the host router (`hostroute`): what it reads, what it computes, where it writes
    out["router"] = {"weight": "blk.{l}.ffn_gate_inp.weight", "bias": "blk.{l}.exp_probs_b.bias",
                     "gating": "sigmoid", "norm_topk": True, "norm_eps": 1e-6, "scale": 1.0,
                     "xm_off": Lm.AD_XM, "res_off": Lm.AD_RES, "idx_off": ROUT_IDX_OFF,
                     "weights_off": ROUT_IDX_OFF + 32}
    if eb_ok(spec):
        out["prompt_gemm"] = {k: v for k, v in prompt_gemm(spec).items() if not k.startswith("_")}
        out["expert_batch"] = {"nt": EB_NT, "kernels": {str(sl): f"eb{sl}" for sl in EB_STREAMS},
                               "args": ["pool", "ebx", "ebh", "eby"]}
    return out


def hf_config_check(spec: ModelSpec) -> dict:
    d = D.hf_config_check(lfm2_view(spec))
    d.update({"num_experts": spec.num_experts, "num_experts_per_tok": spec.experts_per_tok,
              "moe_intermediate_size": spec.moe_intermediate, "num_dense_layers": spec.num_dense_layers,
              "use_expert_bias": True, "norm_topk_prob": True, "routed_scaling_factor": 1.0})
    return d


GEN_KERNELS = "designs/lfm2moe/gen_kernels.py"
KERNEL_SOURCES = list(LF.KERNEL_SOURCES) + ["designs/gemm_q4_prefill/*.py", "designs/gemm_q4_prefill/*.cc",
                                           "designs/gemm_q4_prefill/*.h","designs/lfm2moe/*.py", "designs/lm_head_q8/*.py",
                                           "designs/lm_head_q8/*.cc", "designs/lm_head_q8/*.h"]
