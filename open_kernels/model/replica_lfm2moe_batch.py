r"""replica_lfm2moe_batch: replica_lfm2moe.py's arithmetic, layer-major over the whole prompt.

The token-by-token replica dequantizes four experts and the whole head again for every
token, which puts a few hundred tokens out of reach. This one runs each layer over every
token at once: the mixers as one f32 matmul, the conv recurrence and causal attention (f64
softmax) over the sequence, the router per token in f64, and each expert dequantized ONCE
per layer and applied to the tokens that chose it. The head is dequantized once.

Same weights and the same math as replica_lfm2moe.py (it imports that module's Model,
rms, rope, sigmoid, silu); check it against that replica on a short prompt before trusting
a long one (tests: identical routes, logits corr >= 0.99999).

    python -I replica_lfm2moe_batch.py --gguf <model.gguf> --ids-file ids.txt --out ref512 \
        [--layers N] [--logits-every 1]

--bf16 rounds to bf16 where the engine holds bf16 (every matmul input, K and V in the cache,
the expert inputs, h): the precision floor of a correct NPU path, to tell rounding from a bug.

Writes out/ref_logits_p{i}.bin (f32, every --logits-every positions) and ref_route.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replica_lfm2moe import Model, rms, sigmoid, silu  # noqa: E402

import gguf  # noqa: E402


BF16 = False      # --bf16: round where the NPU holds bf16 (matmul inputs, K / V, expert inputs)


def bf(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 -> f32, round to nearest even (as the engine's f32_to_bf16), when --bf16."""
    if not BF16:
        return x
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def rms_rows(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return np.stack([rms(r, w, eps) for r in x])


def rope_rows(v: np.ndarray, pos: np.ndarray, theta: float) -> np.ndarray:
    """v [T, heads, hd] f32 -> NeoX RoPE at positions pos[T] (replica_lfm2moe.rope, batched)."""
    hd = v.shape[-1]
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    v = v.astype(np.float64)
    a, b = v[..., : hd // 2], v[..., hd // 2:]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(np.float32)


def mixer(m: Model, l: int, xn: np.ndarray) -> np.ndarray:
    p = f"blk.{l}."
    T = xn.shape[0]
    if not m.is_attn(l):
        h = bf(xn) @ m.deq(p + "shortconv.in_proj.weight").T          # [T, 3H]
        H = m.hidden
        B, C, u = h[:, :H], h[:, H:2 * H], h[:, 2 * H:]
        bx = (B * u).astype(np.float64)
        w = m.deq(p + "shortconv.conv.weight").astype(np.float64)       # [H, taps]
        pad = np.vstack([np.zeros((m.taps - 1, H)), bx])                # oldest first
        conv = sum(pad[k:k + T] * w[:, k] for k in range(m.taps)).astype(np.float32)
        return bf(C * conv) @ m.deq(p + "shortconv.out_proj.weight").T
    hd, nh, kvh = m.head_dim, m.n_head, m.kv_heads[l]
    xb = bf(xn)
    q = (xb @ m.deq(p + "attn_q.weight").T).reshape(T, nh, hd)
    k = (xb @ m.deq(p + "attn_k.weight").T).reshape(T, kvh, hd)
    v = bf((xb @ m.deq(p + "attn_v.weight").T).reshape(T, kvh, hd))
    qn, kn = m.deq(p + "attn_q_norm.weight"), m.deq(p + "attn_k_norm.weight")
    q = np.stack([[rms(q[t, i], qn, m.eps) for i in range(nh)] for t in range(T)])
    k = np.stack([[rms(k[t, i], kn, m.eps) for i in range(kvh)] for t in range(T)])
    pos = np.arange(T)
    q, k = rope_rows(q, pos, m.theta), bf(rope_rows(k, pos, m.theta))
    K, V = k.astype(np.float64), v.astype(np.float64)
    g = nh // kvh
    o = np.empty((T, nh, hd), np.float32)
    mask = np.triu(np.full((T, T), -np.inf), 1)
    for h_ in range(nh):
        s = q[:, h_].astype(np.float64) @ K[:, h_ // g].T / np.sqrt(hd) + mask
        s = np.exp(s - s.max(axis=1, keepdims=True))
        o[:, h_] = ((s / s.sum(axis=1, keepdims=True)) @ V[:, h_ // g]).astype(np.float32)
    return bf(o.reshape(T, -1)) @ m.deq(p + "attn_output.weight").T


def ffn(m: Model, l: int, xm: np.ndarray, routes: list) -> np.ndarray:
    p = f"blk.{l}."
    T = xm.shape[0]
    xm = bf(xm)
    if l < m.n_dense:
        gt = xm @ m.deq(p + "ffn_gate.weight").T
        up = xm @ m.deq(p + "ffn_up.weight").T
        return bf(silu(gt) * up) @ m.deq(p + "ffn_down.weight").T
    logits = xm.astype(np.float64) @ m.deq(p + "ffn_gate_inp.weight").astype(np.float64).T
    s = sigmoid(logits)
    sel = s + m.deq(p + "exp_probs_b.bias").astype(np.float64)
    owed: dict[int, list] = {}
    for t in range(T):
        top = np.argsort(-sel[t], kind="stable")[: m.topk]
        w = s[t, top]
        if m.norm_topk:
            w = w / (w.sum() + 1e-6)
        w = (w * m.scale).astype(np.float32)
        routes[t].append({"layer": l, "experts": [int(e) for e in top], "weights": [float(x) for x in w]})
        for e, we in zip(top, w):
            owed.setdefault(int(e), []).append((t, float(we)))
    acc = np.zeros((T, m.hidden), np.float64)
    for e, lst in owed.items():
        rows = np.array([t for t, _ in lst])
        wts = np.array([w for _, w in lst], np.float64)
        x = xm[rows]
        gt = x @ m.expert(p + "ffn_gate_exps.weight", e).T
        up = x @ m.expert(p + "ffn_up_exps.weight", e).T
        y = bf(silu(gt) * up) @ m.expert(p + "ffn_down_exps.weight", e).T
        np.add.at(acc, rows, wts[:, None] * y.astype(np.float64))
    return acc.astype(np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--ids-file")
    ap.add_argument("--layers", type=int, default=-1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--logits-every", type=int, default=1)
    ap.add_argument("--bf16", action="store_true", help="round to bf16 where the NPU holds bf16")
    a = ap.parse_args(argv)
    global BF16
    BF16 = a.bf16
    text = Path(a.ids_file).read_text() if a.ids_file else a.ids
    ids = [int(x) for x in text.replace(",", " ").split()]
    m = Model(a.gguf)
    layers = m.n_layer if a.layers < 0 else a.layers
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    x = np.stack([m.embed(t) for t in ids]).astype(np.float32)
    routes: list = [[] for _ in ids]
    for l in range(layers):
        p = f"blk.{l}."
        h = x + mixer(m, l, rms_rows(x, m.deq(p + "attn_norm.weight"), m.eps))
        x = h + ffn(m, l, rms_rows(h, m.deq(p + "ffn_norm.weight"), m.eps), routes)
        m._cache.clear()
        print(f"layer {l} |x| {np.abs(x).max():.4f}", flush=True)
    xn = rms_rows(x, m.deq("token_embd_norm.weight"), m.eps).astype(np.float64)
    t = m.t["token_embd.weight"]
    pos = list(range(0, len(ids), a.logits_every))
    for r0 in range(0, t.data.shape[0], 8192):
        w = gguf.quants.dequantize(t.data[r0:r0 + 8192], t.tensor_type).astype(np.float64)
        part = (xn[pos] @ w.T).astype(np.float32)
        if r0 == 0:
            lg = np.empty((len(pos), t.data.shape[0]), np.float32)
        lg[:, r0:r0 + 8192] = part
    for j, i in enumerate(pos):
        lg[j].tofile(out / f"ref_logits_p{i}.bin")
        x[i].tofile(out / f"ref_x_p{i}.bin")
    (out / "ref_route.json").write_text(json.dumps(routes))
    print(f"wrote {len(pos)} positions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
