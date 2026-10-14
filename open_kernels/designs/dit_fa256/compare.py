"""dit_fa256 output vs an fp64 reference (causal with offset, GQA): rel_fro, min row cosine."""
import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
D, HEADS, KVH = 256, 24, 4
out = Path(sys.argv[1]); r = np.load(out / "ref.npz")
L, lk, qoff = int(r["L"]), int(r["lk"]), int(r["qoff"])
q = r["q"].view(bfloat16).astype(np.float64).reshape(L, HEADS * D)
kv = r["kv"].view(bfloat16).astype(np.float64).reshape(lk, 2 * KVH * D)
o = np.fromfile(out / "o.bin", dtype=bfloat16).astype(np.float64).reshape(L, HEADS * D)
ref = np.empty_like(o)
mask = np.arange(lk)[None, :] > (qoff + np.arange(L))[:, None]
for h in range(HEADS):
    g = h // (HEADS // KVH)
    K = kv[:, g * D:(g + 1) * D]; V = kv[:, KVH * D + g * D:KVH * D + (g + 1) * D]
    s = q[:, h * D:(h + 1) * D] @ K.T / np.sqrt(D)
    s[mask] = -np.inf
    s = np.exp(s - s.max(1, keepdims=True)); s /= s.sum(1, keepdims=True)
    ref[:, h * D:(h + 1) * D] = s @ V
rel = np.linalg.norm(o - ref) / np.linalg.norm(ref)
oh = o.reshape(L * HEADS, D); rh = ref.reshape(L * HEADS, D)
cos = (oh * rh).sum(1) / np.linalg.norm(oh, axis=1) / np.linalg.norm(rh, axis=1)
print(("PASS" if rel < 4e-2 and np.isfinite(o).all() else "FAIL"), f"rel_fro {rel:.3e} min row cos {cos.min():.5f}")
