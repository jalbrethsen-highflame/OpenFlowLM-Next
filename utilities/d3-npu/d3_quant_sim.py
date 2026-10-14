r"""d3_quant_sim: d3 decisions on the CPU with W4A8 fake quantization, to see what int4 weights
and int8 activations would cost before any NPU kernel exists.

Per selected linear layer: y = Qa(x R) Qw(W R)^T, with R a block Hadamard rotation along K
(orthonormal, so exact before quantization; --rot 0 for none), Qw symmetric int4 with a scale
per output row and K group (--group), Qa symmetric int8 with a scale per token (whole row).
--w-only / --a-only quantize one side. Groups (--groups, comma list):

    ug    mlp.up_proj, mlp.gate_proj          down  mlp.down_proj
    qkv   self_attn.{q,k,v}_proj, linear_attn.in_proj_{qkv,z}
    out   self_attn.o_proj, linear_attn.out_proj

--layers a:b limits it to decoder layers [a, b). Same forward and readout as d3_ref_cpu.py.

    python d3_quant_sim.py --src ~/models/d3-src --cases cases.json --out sim.json --groups ug --rot 128
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

GROUPS = {"ug": ("mlp.up_proj", "mlp.gate_proj"), "down": ("mlp.down_proj",),
          "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "linear_attn.in_proj_qkv",
                  "linear_attn.in_proj_z"),
          "out": ("self_attn.o_proj", "linear_attn.out_proj")}


def hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == n, n
    return h / n ** 0.5


def rotate(x: torch.Tensor, H: torch.Tensor | None) -> torch.Tensor:
    if H is None:
        return x
    b = H.shape[0]
    s = x.shape
    return (x.reshape(*s[:-1], s[-1] // b, b) @ H).reshape(s)


def quant_w(w: torch.Tensor, group: int, bits: int = 4) -> torch.Tensor:
    q = 2 ** (bits - 1) - 1
    o, k = w.shape
    g = w.reshape(o, k // group, group)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / q
    return (torch.round(g / s).clamp(-q, q) * s).reshape(o, k)


def quant_w_po2(w: torch.Tensor, group: int, bits: int = 4) -> torch.Tensor:
    """bfp-style weights: int mantissas in [-2^(b-1), 2^(b-1) - 1] times a power-of-two scale per
    output row and K group (the exponent the NPU's bfp16 MAC would share), the smallest scale
    that keeps the group's largest magnitude unclipped (rounding may still clip it to the top code)."""
    lo, hi = -2 ** (bits - 1), 2 ** (bits - 1) - 1
    o, k = w.shape
    g = w.reshape(o, k // group, group)
    m = g.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(m / hi)))
    return (torch.round(g / s).clamp(lo, hi) * s).reshape(o, k)


def quant_a(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    q = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / q
    return torch.round(x / s).clamp(-q, q) * s


class QLinear(torch.nn.Module):
    def __init__(self, lin: torch.nn.Linear, H, group: int, w_q: bool, a_q: bool, wfmt: str = "int4"):
        super().__init__()
        w = rotate(lin.weight.float(), H)
        if w_q:
            w = quant_w(w, group) if wfmt == "int4" else quant_w_po2(w, group)
        self.weight = torch.nn.Parameter(w.to(torch.bfloat16), requires_grad=False)
        self.bias = lin.bias
        self.H, self.a_q = H, a_q

    def forward(self, x):
        xr = rotate(x.float(), self.H)
        if self.a_q:
            xr = quant_a(xr)
        y = torch.nn.functional.linear(xr.to(torch.bfloat16), self.weight)
        return y if self.bias is None else y + self.bias


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--groups", default="ug,down,qkv,out")
    ap.add_argument("--layers", default="0:64")
    ap.add_argument("--rot", type=int, default=0, help="block Hadamard size along K (0 = no rotation)")
    ap.add_argument("--group", type=int, default=128, help="int4 weight scale group along K")
    ap.add_argument("--w-only", action="store_true")
    ap.add_argument("--wfmt", default="int4", choices=("int4", "po2"),
                    help="int4: float scale per group; po2: power-of-two scale per group (bfp-style, NPU-native)")
    ap.add_argument("--a-only", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from safetensors.numpy import load_file
    W = torch.from_numpy(load_file(str(a.src / "readout.safetensors"))["weight"].astype(np.float32))
    pr = Prompter(a.src)
    lm, cfg = load_language_model(a.src)
    l0, l1 = (int(x) for x in a.layers.split(":"))
    names = [n for g in a.groups.split(",") if g for n in GROUPS[g]]
    t0, n_q = time.time(), 0
    Hs = {}
    with torch.inference_mode():
        for li in range(l0, l1):
            layer = lm.layers[li]
            for name in names:
                parent, _, attr = name.rpartition(".")
                mod = getattr(layer, parent, None) if parent else layer   # linear-attention layers have no self_attn
                lin = getattr(mod, attr, None) if mod is not None else None
                if not isinstance(lin, torch.nn.Linear):
                    continue
                k = lin.in_features
                H = None
                if a.rot:
                    H = Hs.setdefault(a.rot, hadamard(a.rot))
                    assert k % a.rot == 0, (name, k)
                setattr(mod, attr, QLinear(lin, H, a.group, not a.a_only, not a.w_only, a.wfmt))
                n_q += 1
    print(f"quantized {n_q} linears in {time.time() - t0:.0f} s ({a.groups}, layers {a.layers}, rot {a.rot}, "
          f"group {a.group}, {a.wfmt}, {'W only' if a.w_only else 'A only' if a.a_only else 'W4A8'})", file=sys.stderr)
    out = []
    for c in json.loads(a.cases.read_text()):
        ids = pr.ids(c["state"], c["question"])
        h = last_hidden(lm, cfg, ids)
        lg = (h.float() @ W.T).double().numpy()
        keys = options(c["question"])[0]
        z = lg[:len(keys)] / pr.temperature
        p = np.exp(z - z.max()); p /= p.sum()
        out.append({"id": c["id"], "tokens": len(ids), "keys": keys, "probs": p.tolist(), "logits": lg[:len(keys)].tolist()})
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
