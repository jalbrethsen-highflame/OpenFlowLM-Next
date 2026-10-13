"""fp32 CPU reference for LFM2-MoE (LiquidAI LFM2 / LFM2.5 *-A*B), read straight from a GGUF.

The oracle the open lfm2moe kernels are checked against. One decode step per token,
from a zeroed conv state and an empty KV cache, so a run over a prompt is exact prefill.

The architecture (transformers' `Lfm2MoeForCausalLM`, llama.cpp's `lfm2moe`):

    every layer:   h = x + mixer(attn_norm(x))
                   x' = h + ffn(ffn_norm(h))
    mixer:         short conv (most layers) or GQA attention with q/k RMSNorm over the
                   head and full RoPE (the layers `head_count_kv` marks non-zero)
    ffn:           layers < leading_dense_block_count: one silu-gated FFN
                   the rest: 64 routed experts, top-4, no shared expert
    router:        s = sigmoid(Wr @ xm); pick top-k of s + expert_bias; the weights are
                   the UNBIASED s of those experts, / (sum + 1e-6), * routed_scaling_factor
    final norm:    token_embd_norm (llama.cpp's name for LFM2's final norm)
    lm head:       tied to token_embd

The short conv per token (`Lfm2ShortConv`): [B | C | u] = W_in @ x; Bx = B * u; the cache
holds the last L-1 Bx rows; conv[c] = sum_k window[k][c] * w[c, k] with this token's Bx as
the LAST tap; out = W_out @ (C * conv).

Needs the `gguf` package (pip install gguf). Weights stay memory-mapped; the non-expert
matrices of the layers that run are dequantized once, the experts per token (only the
top-k), so a few layers fit in a couple of GB.

    python replica_lfm2moe.py --gguf model.gguf --ids 1,2,3 --layers 4 --out out_dir
    -> out_dir/ref_x_t{i}.bin (f32 hidden after the last layer that ran)
       out_dir/ref_logits_t{i}.bin (f32, final norm + head on that hidden)
       out_dir/ref_route.json (per token, per MoE layer: experts and weights)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import gguf


def rms(x, w, eps):
    x = x.astype(np.float64)
    return (x / np.sqrt(np.mean(x * x) + eps) * w).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Model:
    def __init__(self, path: str):
        self.r = gguf.GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        f = self.r.fields

        def kv(key, default=None):
            fld = f.get(key)
            if fld is None:
                return default
            v = [fld.parts[i] for i in fld.data]
            v = [x.tolist() if hasattr(x, "tolist") else x for x in v]
            v = [x[0] if isinstance(x, list) and len(x) == 1 else x for x in v]
            return v if len(v) > 1 else v[0]

        a = "lfm2moe"
        self.n_layer = int(kv(f"{a}.block_count"))
        self.hidden = int(kv(f"{a}.embedding_length"))
        self.n_head = int(kv(f"{a}.attention.head_count"))
        kvh = kv(f"{a}.attention.head_count_kv")
        self.kv_heads = [int(x) for x in kvh] if isinstance(kvh, list) else [int(kvh)] * self.n_layer
        self.head_dim = self.hidden // self.n_head
        self.eps = float(kv(f"{a}.attention.layer_norm_rms_epsilon"))
        self.theta = float(kv(f"{a}.rope.freq_base"))
        self.n_dense = int(kv(f"{a}.leading_dense_block_count", 0))
        self.topk = int(kv(f"{a}.expert_used_count"))
        self.n_exp = int(kv(f"{a}.expert_count"))
        self.scale = float(kv(f"{a}.expert_weights_scale", 1.0))
        self.norm_topk = bool(kv(f"{a}.expert_weights_norm", True))
        self.taps = int(kv(f"{a}.shortconv.l_cache"))
        self._cache: dict[str, np.ndarray] = {}

    # ---- weights
    def deq(self, name: str) -> np.ndarray:
        """A whole tensor as f32, row-major [rows, cols] (cached)."""
        if name not in self._cache:
            t = self.t[name]
            self._cache[name] = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        return self._cache[name]

    def expert(self, name: str, e: int) -> np.ndarray:
        t = self.t[name]
        return gguf.quants.dequantize(t.data[e], t.tensor_type).astype(np.float32)

    def embed(self, tok: int) -> np.ndarray:
        t = self.t["token_embd.weight"]
        return gguf.quants.dequantize(t.data[tok:tok + 1], t.tensor_type).astype(np.float32)[0]

    def logits(self, x: np.ndarray) -> np.ndarray:
        xn = rms(x, self.deq("token_embd_norm.weight"), self.eps).astype(np.float64)
        t = self.t["token_embd.weight"]
        out = np.empty(t.data.shape[0], np.float32)
        step = 8192
        for r0 in range(0, t.data.shape[0], step):
            w = gguf.quants.dequantize(t.data[r0:r0 + step], t.tensor_type).astype(np.float64)
            out[r0:r0 + step] = (w @ xn).astype(np.float32)
        return out

    def is_attn(self, layer: int) -> bool:
        return self.kv_heads[layer] > 0


def rope(v: np.ndarray, pos: int, theta: float) -> np.ndarray:
    """NeoX-style (rotate_half) RoPE over the whole head; v is [heads, hd]."""
    hd = v.shape[-1]
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = pos * inv
    cos, sin = np.cos(ang), np.sin(ang)
    v = v.astype(np.float64)
    a, b = v[..., : hd // 2], v[..., hd // 2:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(np.float32)


class State:
    def __init__(self, m: Model):
        self.conv = {l: np.zeros((m.taps - 1, m.hidden), np.float32)
                     for l in range(m.n_layer) if not m.is_attn(l)}
        self.k = {l: [] for l in range(m.n_layer) if m.is_attn(l)}
        self.v = {l: [] for l in range(m.n_layer) if m.is_attn(l)}


def mixer(m: Model, st: State, l: int, xn: np.ndarray, pos: int) -> np.ndarray:
    p = f"blk.{l}."
    if not m.is_attn(l):
        h = m.deq(p + "shortconv.in_proj.weight") @ xn
        H = m.hidden
        B, C, u = h[:H], h[H:2 * H], h[2 * H:]
        bx = B * u
        w = m.deq(p + "shortconv.conv.weight")                   # [hidden, taps]
        win = np.vstack([st.conv[l], bx[None, :]])               # [taps, hidden], oldest first
        conv = np.einsum("kc,ck->c", win.astype(np.float64), w.astype(np.float64)).astype(np.float32)
        st.conv[l] = win[1:]
        return m.deq(p + "shortconv.out_proj.weight") @ (C * conv)
    hd, nh, kvh = m.head_dim, m.n_head, m.kv_heads[l]
    q = (m.deq(p + "attn_q.weight") @ xn).reshape(nh, hd)
    k = (m.deq(p + "attn_k.weight") @ xn).reshape(kvh, hd)
    v = (m.deq(p + "attn_v.weight") @ xn).reshape(kvh, hd)
    qn, kn = m.deq(p + "attn_q_norm.weight"), m.deq(p + "attn_k_norm.weight")
    q = np.stack([rms(q[i], qn, m.eps) for i in range(nh)])
    k = np.stack([rms(k[i], kn, m.eps) for i in range(kvh)])
    q, k = rope(q, pos, m.theta), rope(k, pos, m.theta)
    st.k[l].append(k)
    st.v[l].append(v)
    K = np.stack(st.k[l]).astype(np.float64)                     # [T, kvh, hd]
    V = np.stack(st.v[l]).astype(np.float64)
    g = nh // kvh
    o = np.empty((nh, hd), np.float32)
    for h_ in range(nh):
        s = K[:, h_ // g, :] @ q[h_].astype(np.float64) / np.sqrt(hd)
        s = np.exp(s - s.max())
        o[h_] = ((s / s.sum()) @ V[:, h_ // g, :]).astype(np.float32)
    return m.deq(p + "attn_output.weight") @ o.reshape(-1)


def route(m: Model, l: int, xm: np.ndarray):
    p = f"blk.{l}."
    logits = m.deq(p + "ffn_gate_inp.weight").astype(np.float64) @ xm.astype(np.float64)
    s = sigmoid(logits)
    sel = s + m.deq(p + "exp_probs_b.bias").astype(np.float64)
    top = np.argsort(-sel, kind="stable")[: m.topk]
    w = s[top]
    if m.norm_topk:
        w = w / (w.sum() + 1e-6)
    return top, (w * m.scale).astype(np.float32)


def ffn(m: Model, l: int, xm: np.ndarray, routes: list):
    p = f"blk.{l}."
    if l < m.n_dense:
        g = m.deq(p + "ffn_gate.weight") @ xm
        u = m.deq(p + "ffn_up.weight") @ xm
        return m.deq(p + "ffn_down.weight") @ (silu(g) * u)
    top, w = route(m, l, xm)
    routes.append({"layer": l, "experts": [int(e) for e in top], "weights": [float(x) for x in w]})
    acc = np.zeros(m.hidden, np.float64)
    for e, we in zip(top, w):
        g = m.expert(p + "ffn_gate_exps.weight", int(e)) @ xm
        u = m.expert(p + "ffn_up_exps.weight", int(e)) @ xm
        y = m.expert(p + "ffn_down_exps.weight", int(e)) @ (silu(g) * u)
        acc += float(we) * y.astype(np.float64)
    return acc.astype(np.float32)


def step(m: Model, st: State, tok: int, pos: int, layers: int, routes: list) -> np.ndarray:
    x = m.embed(tok)
    for l in range(layers):
        p = f"blk.{l}."
        h = x + mixer(m, st, l, rms(x, m.deq(p + "attn_norm.weight"), m.eps), pos)
        x = h + ffn(m, l, rms(h, m.deq(p + "ffn_norm.weight"), m.eps), routes)
    return x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--layers", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-logits", action="store_true")
    a = ap.parse_args(argv)
    m = Model(a.gguf)
    layers = m.n_layer if a.layers < 0 else a.layers
    ids = [int(x) for x in a.ids.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = State(m)
    all_routes = []
    for i, tok in enumerate(ids):
        routes: list = []
        x = step(m, st, tok, i, layers, routes)
        all_routes.append(routes)
        x.astype(np.float32).tofile(out / f"ref_x_t{i}.bin")
        if not a.no_logits:
            lg = m.logits(x)
            lg.tofile(out / f"ref_logits_t{i}.bin")
            top = np.argsort(-lg)[:5]
            print(f"t{i} tok {tok}: argmax {int(top[0])} top5 {top.tolist()} |x| {np.abs(x).max():.4f}", flush=True)
        else:
            print(f"t{i} tok {tok}: |x| {np.abs(x).max():.4f}", flush=True)
    (out / "ref_route.json").write_text(json.dumps(all_routes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
