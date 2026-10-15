import sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
out = Path(sys.argv[1]); ref = np.load(out / "ref.npy").astype(np.float64)
if (out / "pack_h").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pack_a import unpack_a, unpack_a2
    M_, ff = ref.shape   # slots [rg][ks][crow] -> pack order [rg][crow][ks]
    mt = int((out / "mt").read_text()) if (out / "mt").exists() else 128
    raw = np.fromfile(out / "c.bin", dtype=np.uint8).reshape(M_ // (4 * mt), ff // 64, 4, 36864)[..., :72 * mt]
    raw = raw.transpose(0, 2, 1, 3).reshape(-1)
    c = (unpack_a2(raw, *ref.shape, mt) if (out / "pack_h").read_text() == "hs" else unpack_a(raw, *ref.shape)).astype(np.float64)
else:
    c = np.fromfile(out / "c.bin", dtype=bfloat16).astype(np.float64).reshape(ref.shape)
if (out / "mask.npy").exists() and not (out / "pack_h").exists():
    mk = np.load(out / "mask.npy"); ref = ref[:, mk]; c = c[:, mk]
rel = np.linalg.norm(c - ref) / np.linalg.norm(ref)
cos = (c * ref).sum(1) / np.linalg.norm(c, axis=1) / np.linalg.norm(ref, axis=1)
print(("PASS" if rel < 3e-2 and np.isfinite(c).all() else "FAIL"), f"rel_fro {rel:.3e} row-cos min {cos.min():.6f}")
