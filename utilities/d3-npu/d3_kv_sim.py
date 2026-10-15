r"""d3_kv_sim: d3 decisions on the CPU with the full-attention layers' Q, K and V quantized as an
NPU attention kernel would hold them, to choose the K / V format before building it.

Every call of torch's scaled_dot_product_attention (only the full-attention layers make one; the
Gated DeltaNet layers do not) gets Q, K and V through a block quantizer first: integer mantissas
and one power-of-two exponent per block. The blocks run along what the 8x8x8 MAC shares an
exponent over: Q and K along the head dim (per query / per key), V along the keys (per head-dim
column). --kfmt / --vfmt: none | b8 (int8 mantissas, block 8: bfp16ebs8) | i4b<n> (int4
mantissas, block n, the lower-error of two exponents as pack_b4). Q is b8 unless --q none.

    python d3_kv_sim.py --src ~/models/d3-src --cases long.json --out kv.json --kfmt b8 --vfmt i4b8
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_npu import Prompter, options  # noqa: E402
from d3_ref_cpu import last_hidden, load_language_model  # noqa: E402


def block_quant(x: torch.Tensor, dim: int, fmt: str) -> torch.Tensor:
    """x quantized in blocks along dim (padded with zeros to whole blocks)."""
    if fmt == "none":
        return x
    bits, n, choose = (8, 8, False) if fmt == "b8" else (4, int(fmt[3:]), True)
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    xt = x.float().movedim(dim, -1)
    L = xt.shape[-1]
    pad = (-L) % n
    if pad:
        xt = torch.nn.functional.pad(xt, (0, pad))
    g = xt.reshape(*xt.shape[:-1], -1, n)
    m = g.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(m / hi)))
    q0 = torch.round(g / s).clamp(lo, hi) * s
    if choose:
        q1 = torch.round(g / (s / 2)).clamp(lo, hi) * (s / 2)
        take = ((q1 - g) ** 2).sum(-1, keepdim=True) < ((q0 - g) ** 2).sum(-1, keepdim=True)
        q0 = torch.where(take, q1, q0)
    out = q0.reshape(*xt.shape)[..., :L].movedim(-1, dim)
    return out.to(x.dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kfmt", default="none")
    ap.add_argument("--vfmt", default="none")
    ap.add_argument("--q", default="b8")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def quant_sdpa(q, k, v, *args, **kw):   # [B, heads, L, D]
        return sdpa(block_quant(q, -1, a.q), block_quant(k, -1, a.kfmt), block_quant(v, -2, a.vfmt), *args, **kw)

    if (a.kfmt, a.vfmt, a.q) != ("none", "none", "none"):
        torch.nn.functional.scaled_dot_product_attention = quant_sdpa
    from safetensors.numpy import load_file
    W = torch.from_numpy(load_file(str(a.src / "readout.safetensors"))["weight"].astype(np.float32))
    pr = Prompter(a.src)
    lm, cfg = load_language_model(a.src)
    out = []
    for c in json.loads(a.cases.read_text()):
        ids = pr.ids(c["state"], c["question"])
        t0 = time.time()
        h = last_hidden(lm, cfg, ids)
        lg = (h.float() @ W.T).double().numpy()
        keys = options(c["question"])[0]
        z = lg[:len(keys)] / pr.temperature
        p = np.exp(z - z.max()); p /= p.sum()
        out.append({"id": c["id"], "tokens": len(ids), "keys": keys, "probs": p.tolist(), "logits": lg[:len(keys)].tolist()})
        a.out.write_text(json.dumps(out, indent=1))
        print(f"{c['id'][:40]:40s} {len(ids):5d} tok {time.time() - t0:5.0f} s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
