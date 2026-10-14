import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
out = Path(sys.argv[1]); ref = np.load(out / "ref.npy").astype(np.float64)
c = np.fromfile(out / "c.bin", dtype=bfloat16).astype(np.float64).reshape(ref.shape)
if (out / "mask.npy").exists():
    mk = np.load(out / "mask.npy"); ref = ref[:, mk]; c = c[:, mk]
rel = np.linalg.norm(c - ref) / np.linalg.norm(ref)
cos = (c * ref).sum(1) / np.linalg.norm(c, axis=1) / np.linalg.norm(ref, axis=1)
print(("PASS" if rel < 3e-2 and np.isfinite(c).all() else "FAIL"), f"rel_fro {rel:.3e} row-cos min {cos.min():.6f}")
