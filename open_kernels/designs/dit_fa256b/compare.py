r"""dit_fa256b output vs fp64 noncausal attention over keys [0, valid): against the bf16 inputs
(the format's total error) and against the bfp16-rounded inputs (the kernel's own error).

    python compare.py <test dir> [o.bin]
"""
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from pack_fa import QSCALE, round_cols, round_rows

D, HEADS, KVH = 256, 24, 4
out = Path(sys.argv[1]); r = np.load(out / "ref.npz")
q, k, v, valid = r["q"], r["k"], r["v"], int(r["valid"])
causal = int(r["causal"]) if "causal" in r.files else 0
q_off = int(r["q_off"]) if "q_off" in r.files else 0
L = q.shape[0]
o = np.fromfile(sys.argv[2] if len(sys.argv) > 2 else out / "o.bin", dtype=bfloat16).astype(np.float64).reshape(L, HEADS * D)


def attention(qf, kf, vf):
    ref = np.empty((L, HEADS * D))
    for h in range(HEADS):
        g = h // (HEADS // KVH)
        s = qf[:, h].astype(np.float64) @ kf[:valid, g].astype(np.float64).T / np.sqrt(D)
        if causal:   # row i is position q_off + i: keys past it masked
            s = np.where(np.arange(valid)[None, :] > (q_off + np.arange(L))[:, None], -np.inf, s)
        s = np.exp(s - s.max(1, keepdims=True)); s /= s.sum(1, keepdims=True)
        ref[:, h * D:(h + 1) * D] = s @ vf[:valid, g]
    return ref


def report(name, ref):
    rel = np.linalg.norm(o - ref) / np.linalg.norm(ref)
    oh, rh = o.reshape(L * HEADS, D), ref.reshape(L * HEADS, D)
    cos = (oh * rh).sum(1) / np.linalg.norm(oh, axis=1) / np.linalg.norm(rh, axis=1)
    print(f"{name}: rel_fro {rel:.3e}  min row cos {cos.min():.5f}")
    return rel


qr = np.stack([round_rows(q[:, h] * np.float32(QSCALE)) / QSCALE for h in range(HEADS)], 1)
kr = np.stack([round_rows(k[:, g]) for g in range(KVH)], 1)
vr = np.stack([round_cols(v[:, g]) for g in range(KVH)], 1)
r1 = report("vs bf16 inputs ", attention(q, k, v))
r2 = report("vs bfp16 inputs", attention(qr, kr, vr))
print("PASS" if r2 < 3e-2 and np.isfinite(o).all() else "FAIL")
