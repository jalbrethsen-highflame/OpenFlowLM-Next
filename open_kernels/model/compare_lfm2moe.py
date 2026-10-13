"""Score the open engine's LFM2-MoE logits against replica_lfm2moe.py.

    python compare_lfm2moe.py --gguf model.gguf --ref REF_DIR --npu NPU_PREFIX [--positions N]

REF_DIR is replica_lfm2moe.py's --out (ref_x_t{i}.bin, ref_logits_t{i}.bin); NPU_PREFIX
is open_qwen36_cli's --dump-logits prefix with --prefill-logits (NPU_PREFIX_p{i}.bin).

`exact` is the NPU logits against the reference logits (the GGUF's own Q6_K head). The
recipe's head is q8 (lmhead_q8_gguf), which scores 0.9999+ there. With --q4-head a second
number scores the NPU against the reference HIDDEN through a q4_1 head and a bf16 hn --
the check for a kernel set built with the old q4_1 head (std_perm_gguf requantizes the
Q6_K embedding to fp16 d/m per 32-block, which on its own costs ~0.995).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gguf


def bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32)


def q4_head(E) -> np.ndarray:
    """The tied embedding as std_perm_gguf requantizes it: per 32-block min / (max-min)/15,
    both rounded to fp16, codes rounded to nearest and clamped."""
    out = np.empty(E.data.shape[:1] + (E.shape[0],), np.float32)
    for r0 in range(0, E.data.shape[0], 8192):
        w = gguf.quants.dequantize(E.data[r0:r0 + 8192], E.tensor_type).astype(np.float32)
        b = w.reshape(w.shape[0], -1, 32)
        lo, hi = b.min(-1, keepdims=True), b.max(-1, keepdims=True)
        d = ((hi - lo) / 15.0).astype(np.float16).astype(np.float32)
        m = lo.astype(np.float16).astype(np.float32)
        inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
        q = np.clip(np.rint((b - m) * inv), 0, 15)
        out[r0:r0 + 8192] = (q * d + m).reshape(w.shape)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--npu", required=True)
    ap.add_argument("--positions", type=int, default=0, help="0 = every ref_x_t*.bin present")
    ap.add_argument("--q4-head", action="store_true", help="also score against a q4_1 head (old kernel sets)")
    a = ap.parse_args(argv)
    r = gguf.GGUFReader(a.gguf)
    t = {x.name: x for x in r.tensors}
    eps = float(r.fields["lfm2moe.attention.layer_norm_rms_epsilon"].parts[-1][0])
    nw = t["token_embd_norm.weight"].data.astype(np.float64)
    ref = Path(a.ref)
    n = a.positions or len(list(ref.glob("ref_x_t*.bin")))
    head = q4_head(t["token_embd.weight"]) if a.q4_head else None
    rows = t["token_embd.weight"].data.shape[0]
    real = 125017 if rows == 128000 else rows
    worst = 1.0
    for i in range(n):
        le = np.fromfile(ref / f"ref_logits_t{i}.bin", np.float32)
        y = np.fromfile(f"{a.npu}_p{i}.bin", np.float32)
        ce = np.corrcoef(le[:real], y[:real])[0, 1]
        worst = min(worst, ce)
        cq = ""
        if head is not None:
            x = np.fromfile(ref / f"ref_x_t{i}.bin", np.float32).astype(np.float64)
            hn = bf16(x / np.sqrt(np.mean(x * x) + eps) * nw).astype(np.float32)
            cq = f"  q4head {np.corrcoef((head @ hn)[:real], y[:real])[0, 1]:.6f}"
        same = "same" if int(np.argmax(le[:real])) == int(np.argmax(y[:real])) else "DIFF"
        top_r, top_y = np.argsort(-le[:real])[:5], np.argsort(-y[:real])[:5]
        print(f"p{i}: exact {ce:.6f}{cq}  argmax {same} ({int(top_r[0])} / {int(top_y[0])})  "
              f"top5 overlap {len(set(top_r) & set(top_y))}/5")
    print(f"worst exact {worst:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
