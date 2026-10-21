r"""Flint (RecursiveSI/flint, FlintForCausalLM) kernel set for the open engine: the prompt pipeline's
NPU kernels and manifest.json. Decode runs on the CPU (the q4_1 pools in place), so there are no
decode layer images.

Flint: hidden 2048, 49 layers -- 2 dense (SwiGLU ff 6144) then 47 MoE (256 experts of ff 512,
top-6 sigmoid routing, an ungated shared expert of ff 1024); every layer GQA attention 32 / 4 heads
at head dim 128 with per-head q / k RMSNorm and learned sinks, 35 of them a 513-key sliding window
with RoPE (theta 1e4), 14 full and without positions; vocab 150144, untied q4_1 head.

Kernels (designs reused, new shapes built):
  * dit2 int4 GEMMs: q|k|v [5120, 2048], o [2048, 4096]; the FFNs as dit2's SwiGLU pair (up|gate
    with the packed-h epilogue, then down on the a_units): the shared expert at ff 1024, the dense
    layers' at ff 6144. Each at the pipe's M 1024, the small tier's 512 and the tiny tier's 256.
  * the mbi expert streams (designs/moe_batch/mbi.py): Qwen3.6-35B-A3B's experts have this geometry
    (hidden 2048, ff 512, 256 of them, the same pool layout), so its builds are copied.
  * flash attention: designs/dit_fa256b (head dim 256): Flint's 128 zero-padded, the sinks as a
    64-key chunk in front (FLINT.md).

The pool (one per layer): the experts exactly where Qwen3.6-35B-A3B keeps them (the mbi streams
are built for those offsets), then the shared expert and the attention projections; a dense layer
type has its own, smaller pool.

    python open_kernels/export_flint_kernels.py OUT_DIR [--reuse K11_DIR] [--jobs N] [--only NAME,...]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGNS = HERE / "designs"

HID, NL, VOCAB = 2048, 49, 150144
NH, KVH, HD = 32, 4, 128
QW, KVW = NH * HD, KVH * HD
E, TOPK, FF_E, FF_SH, FF_D = 256, 6, 512, 1024, 6144
WINDOW = 513
NOPE = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48}
DENSE = {0, 1}
CH, MB = 5120, 1 << 20
POOL = 512 * MB
EPS = 1e-6


def q4(rows: int, cols: int) -> int:
    """Chunks of a [rows, cols] q4_1 tensor."""
    return rows * cols // 8192


# ---- the pool layouts
STRIPE = 163840                        # 128 rows x 2048 K of q4_1 (32 chunks)
UP_BYTES = q4(FF_E, HID) * CH          # one expert's down (2048 x 512): 655360
POOL_DOWN = E * 2 * (FF_E // 128) * STRIPE     # 335544320: the downs after every expert's up|gate
SH_UP = POOL_DOWN + E * UP_BYTES               # 503316480


def attn_ops(at: int) -> tuple[list[dict], int]:
    ops = []
    for t, rows, cols in (("attn_q", QW, HID), ("attn_k", KVW, HID), ("attn_v", KVW, HID), ("attn_output", HID, QW)):
        n = q4(rows, cols)
        ops.append({"op": "std_perm", "tensor": f"blk.{{l}}.{t}.weight", "dst": at, "nch": n, "in_dim": cols})
        at += n * CH
    return ops, at


def moe_pool() -> tuple[list[dict], dict]:
    ops = [
        {"op": "expert_stripes", "up": "blk.{l}.ffn_up_exps.weight", "gate": "blk.{l}.ffn_gate_exps.weight", "dst": 0,
         "experts": E, "stripes": FF_E // 128, "stripe_bytes": STRIPE, "in_dim": HID},
        {"op": "expert_down", "tensor": "blk.{l}.ffn_down_exps.weight", "dst": POOL_DOWN, "experts": E,
         "expert_bytes": UP_BYTES},
    ]
    at = SH_UP
    off = {}
    for t, rows, cols, key in (("ffn_up_shexp", FF_SH, HID, "share_up"), ("ffn_gate_shexp", FF_SH, HID, "share_gate"),
                               ("ffn_down_shexp", HID, FF_SH, "share_down")):
        n = q4(rows, cols)
        off[key] = at
        ops.append({"op": "std_perm", "tensor": f"blk.{{l}}.{t}.weight", "dst": at, "nch": n, "in_dim": cols})
        at += n * CH
    a, end = attn_ops(at)
    assert end <= POOL, end
    return ops + a, off


def dense_pool() -> tuple[list[dict], int]:
    ops, at = [], 0
    for t, rows, cols in (("ffn_up", FF_D, HID), ("ffn_gate", FF_D, HID), ("ffn_down", HID, FF_D)):
        n = q4(rows, cols)
        ops.append({"op": "std_perm", "tensor": f"blk.{{l}}.{t}.weight", "dst": at, "nch": n, "in_dim": cols})
        at += n * CH
    a, end = attn_ops(at)
    return ops + a, (end + MB - 1) // MB * MB


def consts(moe: bool) -> tuple[list[dict], int]:
    ops, at = [], 0
    names = [("input_layernorm", HID * 4), ("post_attention_layernorm", HID * 4), ("attn_q_norm", HD * 4),
             ("attn_k_norm", HD * 4), ("attn_sinks", NH * 4)]
    if moe:
        names.append(("moe_router", HID * E * 4))
    for t, cap in names:
        ops.append({"op": "put", "tensor": f"blk.{{l}}.{t}.weight", "dst": at, "cap": cap})
        at += (cap + 4095) // 4096 * 4096
    return ops, at


# ---- the builds: name -> (design, build dir, env)
def dit2(n: int, k: int, m: int, layout: dict | None = None, tag: str = "") -> tuple[str, dict]:
    name = f"dit2_n{n}_k{k}{tag}" + ("" if m == 1024 else f"_m{m}")
    env = {"DG_M": str(m), "DG_K": str(k), "DG_N": str(n), "DG_LAYOUT": json.dumps(layout or {}), "DG_KFLAGS": "-DDIT_HS"}
    if m == 256:
        env["DG_MT"] = "64"
    bdir = f"dit2/build_m{m}_k{k}_n{n}{tag}" + ("_mt64" if m == 256 else "") + "_hs"
    return name, {"design": "dit2/dit2.py", "build_dir": bdir, "env": env}


FA_QOFF = 202372583                   # dit_fa256b's q_off sentinel (patched per call)
FA_CB = [1024 * i for i in range(1, 130)]           # key counts to 132096: 128K + the sink chunk, rounded up
FA_CBS = [512 * i for i in range(1, 6)]             # the small tier's (512 queries), to 2560
FA_MAX_LK = int(os.environ.get("FLINT_FA_MAX_LK", 132096))


def fa(L: int, lk: int, window: int = 0) -> tuple[str, dict]:
    """A causal dit_fa256b stream; window > 0: the sliding layers' (keys after the 64-key sink chunk)."""
    kind = "bw" if window else "bc"
    name = f"fa256{kind}{'s' if L == 512 else ''}_lk{lk}"
    env = {"DF_L": str(L), "DF_LK": str(lk), "DF_CAUSAL": "1", "DF_QOFF": str(FA_QOFF), "DF_KV_INTER": "1",
           "DF_HEADS": str(NH), "DF_KV_HEADS": str(KVH), "DF_WINDOW": str(window)}
    # (v3: the window streams' live-chunk check, fa_qkpv_w -- their own image, context faw)
    bdir = f"dit_fa256b/build_l{L}_lk{lk}_causal_kvi_qs_h{NH}k{KVH}" + (f"_w{window}_v3" if window else "_v2")
    return name, {"design": "dit_fa256b/dit_fa256b.py", "build_dir": bdir, "env": env}


# the sliding layers' streams: a chunk's queries over the sink chunk and keys from 512 before it (64 + at
# most 576 + M keys)
FA_WIN = {1024: 1664, 512: 1152}


EPI_PH = {"epi": {"first_cb": 0}, "pack_h": True}
AU = {"a_units": True}
TIERS = (1024, 512, 256)


def builds() -> dict:
    b = {}
    for m in TIERS:
        for n, k, lay, tag in ((QW + 2 * KVW, HID, None, ""), (HID, QW, None, ""),
                               (2 * FF_SH, HID, EPI_PH, "_epi_ph"), (HID, FF_SH, AU, "_au"),
                               (2 * FF_D, HID, EPI_PH, "_epi_ph"), (HID, FF_D, AU, "_au")):
            name, d = dit2(n, k, m, lay, tag)
            b[name] = d
    for lk in FA_CB:
        if lk <= FA_MAX_LK:
            name, d = fa(1024, lk)
            b[name] = d
    for lk in FA_CBS:
        name, d = fa(512, lk)
        b[name] = d
    for L, lk in FA_WIN.items():
        name, d = fa(L, lk, WINDOW)
        b[name] = d
    for s in (256, 128, 64, 32, 16, 8):
        b[f"mb_s{s}"] = {"design": "moe_batch/mbi.py", "build_dir": f"moe_batch/build_mbi_s{s}",
                         "env": {"MBI_SLOTS": str(s), "MBI_EXPERTS": str(E), "MBI_POOL_DOWN": str(POOL_DOWN),
                                 "MBI_POOL_BYTES": str(POOL), "MBI_QSC": "1"}}
    return b


def context_of(name: str) -> str:
    if name.startswith("mb_"):
        return "mb"
    if name.startswith("fa256bw"):
        return "faw"
    if name.startswith("fa256b"):
        return "fab"
    if name.startswith("dit2") and name.endswith("_m256"):
        return "dit2t"
    return "dit2"


def ffn_section(ff: int, m: int) -> dict:
    up = dit2(2 * ff, HID, m, EPI_PH, "_epi_ph")[0]
    dn = dit2(HID, ff, m, AU, "_au")[0]
    # up2 / down2 (the host-repacked FFN) are not built: the device-packed pair serves every tier
    return {"ff": ff, "up2": up, "down2": dn, "up2p": up, "down2u": dn}


def manifest(bset: dict) -> dict:
    moe_ops, sh = moe_pool()
    dense_ops, dense_bytes = dense_pool()
    cmoe, cmoe_b = consts(True)
    cden, cden_b = consts(False)
    n_moe = len(moe_ops)

    def qkv_weights(first: int) -> dict:
        return {"gqkv_w": {"from": "pool", "ops": [first, first + 1, first + 2]}, "go_w": {"from": "pool", "ops": [first + 3]}}

    def gemm_program() -> list:
        return [{"op": "run", "kernel": f"dit2_n{QW + 2 * KVW}_k{HID}", "args": ["gqkv_w", "pp_x", "pp_y"]},
                {"op": "run", "kernel": f"dit2_n{HID}_k{QW}", "args": ["go_w", "pp_x", "pp_y"]}]

    moe_batch = {"kernels": {str(s): f"mb_s{s}" for s in (256, 128, 64, 32, 16, 8)}, "args": ["pool", "mb_x", "mb_y"],
                 "nt": 64, "kind": "mbi", "xq": "compact"}

    def layer_type(moe: bool, swa: bool) -> dict:
        gb = {"kind": "full", "t": 256, "eps": EPS, "program": gemm_program(), "qw": QW, "kvw": KVW, "nh": NH, "kvh": KVH,
              "hd": HD, "rot": 0 if not swa else HD, "gate": False, "sinks": True, "window": WINDOW if swa else 0}
        if moe:
            gb.update({"weights": qkv_weights(n_moe - 4), "shared_ff": FF_SH, "shared_gated": False,
                       "shared_program": [{"op": "run", "kernel": f"dit2_n{2 * FF_SH}_k{HID}_epi_ph", "args": ["gshare_w", "pp_x", "pp_h"]},
                                          {"op": "run", "kernel": f"dit2_n{HID}_k{FF_SH}_au", "args": ["gsdown_w", "pp_h", "pp_y"]}],
                       "shared_weights": {"gshare_w": {"from": "pool", "ops": [2, 3]}, "gsdown_w": {"from": "pool", "ops": [4]}},
                       "moe_batch": moe_batch})
        else:
            gb.update({"weights": qkv_weights(3), "ff": FF_D, "ffn_key": "ffn_dense",
                       "ffn_program": [{"op": "run", "kernel": f"dit2_n{2 * FF_D}_k{HID}_epi_ph", "args": ["gffn_ug_w", "pp_x", "pp_h"]},
                                       {"op": "run", "kernel": f"dit2_n{HID}_k{FF_D}_au", "args": ["gffn_dn_w", "pp_h", "pp_y"]}],
                       "ffn_weights": {"gffn_ug_w": {"from": "pool", "ops": [0, 1]}, "gffn_dn_w": {"from": "pool", "ops": [2]}}})
        lt = {"buffers": {"consts": cmoe_b if moe else cden_b, "act": 4096, "state": {"kind": "kv", "row": 2 * KVW * 2}},
              "program": [], "gemm_block": gb, "pack": {"pool": moe_ops if moe else dense_ops, "consts": cmoe if moe else cden}}
        if swa:
            lt["buffers"]["state"]["ring"] = 4096          # the window's rows plus a chunk, reused
        if not moe:
            lt["pool_bytes"] = dense_bytes
        return lt

    layers = []
    for l in range(NL):
        layers.append("dense" if l in DENSE else "global" if l in NOPE else "swa")
    assert all(l in NOPE for l in DENSE)
    kernels = {n: {"context": context_of(n), "insts": f"{n}/insts.bin", "build": n} for n in bset}
    for n in kernels:
        if n.startswith("mb_"):
            kernels[n]["patch"] = "moebatch"
    contexts = {"mb": "mb_s256/final.xclbin", "fab": "fa256bc_lk1024/final.xclbin",
                "faw": f"{fa(1024, FA_WIN[1024], WINDOW)[0]}/final.xclbin", "dit2": f"dit2_n{HID}_k{QW}/final.xclbin",
                "dit2t": f"dit2_n{HID}_k{QW}_m256/final.xclbin"}

    def tier(m: int) -> dict:
        return {"m": m, "kernels2": {f"{QW + 2 * KVW}x{HID}": dit2(QW + 2 * KVW, HID, m)[0], f"{HID}x{QW}": dit2(HID, QW, m)[0]},
                "ffn": ffn_section(FF_SH, m), "ffn_dense": ffn_section(FF_D, m)}

    gd = tier(1024)
    # the full layers' attention on dit_fa256b: head dim 256, Flint's 128 zero-padded, the sinks a chunk in front
    gd["fa"] = {"hd": 256, "q_cols": NH * 256, "cb_qoff": FA_QOFF, "cbs_m": 512,
                "cb": {str(lk): fa(1024, lk)[0] for lk in FA_CB if fa(1024, lk)[0] in bset},
                "cbs": {str(lk): fa(512, lk)[0] for lk in FA_CBS if fa(512, lk)[0] in bset},
                "win": {str(L): {"kernel": fa(L, lk, WINDOW)[0], "lk": lk} for L, lk in FA_WIN.items()}}
    # experts over groups of 8192 tokens (bit-identical; 32K GPU idle: 35.2 -> 33.5 s, slots fuller at top-6)
    gd["moe_group"] = 8192
    gd.update({"hs": True, "kernels": {}, "npad": {str(n): n for n in (QW + 2 * KVW, HID, 2 * FF_SH, 2 * FF_D)},
               "small": tier(512), "tiny": {**tier(256), "mt": 64}})
    inv = [1.0 / (10000.0 ** (2 * i / HD)) for i in range(HD // 2)]
    hf_types = ["full_attention" if l in NOPE else "sliding_attention" for l in range(NL)]
    return {
        "manifest_version": 3,
        "family": "flint",
        "spec": {"family": "flint", "hidden": HID, "num_layers": NL, "vocab": VOCAB, "num_heads": NH, "num_kv_heads": KVH,
                 "head_dim": HD, "sliding_window": WINDOW, "num_experts": E, "experts_per_tok": TOPK,
                 "moe_intermediate": FF_E, "shared_expert_intermediate": FF_SH, "intermediate": FF_D, "quant": "q4_1"},
        "max_ctx_default": 4096,
        "hf_config_check": {"model_type": "flint", "hidden_size": HID, "num_hidden_layers": NL, "vocab_size": VOCAB,
                            "num_experts": E, "num_experts_per_tok": TOPK, "moe_intermediate_size": FF_E,
                            "shared_expert_intermediate_size": FF_SH, "intermediate_size": FF_D, "head_dim": HD,
                            "num_attention_heads": NH, "num_key_value_heads": KVH, "sliding_window": WINDOW,
                            "layer_types": hf_types, "no_rope_layer_indices": sorted(NOPE), "mlp_only_layers": sorted(DENSE),
                            "router_score_function": "sigmoid", "norm_topk_prob": True, "softmax_type": "learnable"},
        "hf_config_defaults": {},
        "layout": {"hidden": HID, "vocab": VOCAB, "real_vocab": VOCAB, "chunk_bytes": CH, "pool_bytes": POOL,
                   "lmhead_pool_bytes": q4(VOCAB, HID) * CH, "lmhead_chunk_bytes": CH, "kv_row": 2 * KVW * 2, "ptab_row": 1024,
                   "rotary_dim": HD, "rope_theta": 10000.0, "rope_inv_freq": inv,
                   "moe": {"experts": E, "topk": TOPK, "stripe": STRIPE, "up_bytes": UP_BYTES, "down_core": UP_BYTES // 8,
                           "pool_down": POOL_DOWN, **sh},
                   "router": {"weight": "blk.{l}.moe_router.weight", "gating": "sigmoid", "norm_topk": True, "norm_eps": 1e-20,
                              "xm_off": 0},
                   "gemm_dit": gd},
        "layers": layers,
        "contexts": contexts,
        "kernels": kernels,
        "layer_types": {"dense": layer_type(False, False), "global": layer_type(True, False), "swa": layer_type(True, True)},
        "tail": [],
        # (pp_x / pp_y / pp_h: the route steps' nominal x / y; the pipeline sizes its own per chunk)
        "globals": {"lmpool": q4(VOCAB, HID) * CH, "mb_x": 71827456, "mb_y": 67108864, "pp_x": 4096, "pp_y": 4096,
                    "pp_h": 4096},
        "builds": bset,
        "pack": {"pool_bytes": POOL, "chunk_bytes": CH,
                 "lm_head": {"pool_bytes": q4(VOCAB, HID) * CH,
                             "ops": [{"op": "std_perm", "tensor": "output.weight", "dst": 0, "nch": q4(VOCAB, HID), "in_dim": HID}]},
                 "embed": {"tensor": "token_embd.weight", "dim": HID}, "norm": {"tensor": "output_norm.weight", "bytes": HID * 4}},
    }


def build_one(name: str, d: dict) -> tuple[str, Path]:
    out = DESIGNS / d["build_dir"]
    env = dict(os.environ)
    env.update(d["env"])
    t0 = time.time()
    log = out.with_suffix(".log")
    out.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, str(HERE / "build_design.py"), str(DESIGNS / d["design"]), str(out)], env=env,
                           cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"[{name}] build FAILED, see {log}")
    print(f"[{name}] built in {time.time() - t0:.0f}s", flush=True)
    return name, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--reuse", help="a kernel set whose builds of the same name and env are copied (Qwen3.6-35B-A3B's)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--manifest-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bset = builds()
    m = manifest(bset)
    if not a.manifest_only:
        reuse = {}
        if a.reuse:
            rm = json.loads((Path(a.reuse) / "manifest.json").read_text())
            for n, d in bset.items():
                if n in rm["builds"] and rm["builds"][n]["env"] == d["env"] and (Path(a.reuse) / n / "insts.bin").exists():
                    reuse[n] = Path(a.reuse) / n
        only = set(x for x in a.only.split(",") if x)
        todo = [n for n in bset if n not in reuse and (not only or n in only)]
        for n, src in reuse.items():
            if src.resolve() != (out / n).resolve():
                shutil.copytree(src, out / n, dirs_exist_ok=True)
                print(f"[{n}] copied from {src}")
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            for n, bdir in ex.map(lambda n: build_one(n, bset[n]), todo):
                (out / n).mkdir(exist_ok=True)
                for f in ("final.xclbin", "insts.bin"):
                    shutil.copy2(bdir / f, out / n / f)
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
