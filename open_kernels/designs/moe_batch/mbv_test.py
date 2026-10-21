"""inputs + reference for mbv.py: mbv_test.py gen OUTDIR Q K R | check OUTDIR"""
import sys, numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
sys.path.insert(0, str(Path(__file__).parent))
mode, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
NT, NTG, CH = 16, 4, 5120
XEL = 64 + 16 * NTG * 64 + 8 * NTG * 128 + NTG * 128


def q41_matrix(rng, rows, K):
    d = (np.abs(rng.normal(0, 0.004, (rows, K // 32))) + 5e-4).astype(bfloat16)
    m = (-8 * d.astype(np.float32) + rng.normal(0, 0.002, (rows, K // 32))).astype(bfloat16)
    return d, m, rng.integers(0, 16, (rows, K))


def chunk_bytes(d, m, q, r0, kg, S):
    dd = d[r0:r0 + 32, kg * 8:(kg + 1) * 8].astype(np.float64).T
    mm = m[r0:r0 + 32, kg * 8:(kg + 1) * 8].astype(np.float64).T
    di = np.rint(dd * 2.0 ** S).astype(np.int16)
    mi = np.rint((mm + 8 * dd) * 2.0 ** S).astype(np.int16)
    qq = q[r0:r0 + 32, kg * 256:(kg + 1) * 256]
    nib = np.zeros(8192, np.uint8)
    for rb in range(2):
        nib.reshape(2, 256, 16)[rb] = qq[rb * 16:(rb + 1) * 16].T
    nb = (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8) ^ 0x88
    return di.tobytes() + mi.tobytes() + nb.tobytes()


if mode == "gen":
    Q, K, R = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    KG, N = K // 256, 32 * Q * 64
    rng = np.random.default_rng(7)
    d, m, q = q41_matrix(rng, N, K)
    dmax = np.abs(d.astype(np.float64)).max(); mcmax = np.abs(m.astype(np.float64) + 8 * d.astype(np.float64)).max()
    S = int(min(np.floor(np.log2(8000 / dmax)), np.floor(np.log2(32000 / mcmax))))
    w = bytearray(N // 64 * KG * 10240)
    for b in range(N // 64):
        for kg in range(KG):
            off = (b * KG + kg) * 10240
            w[off:off + 10240] = chunk_bytes(d, m, q, 64 * b, kg, S) + chunk_bytes(d, m, q, 64 * b + 32, kg, S)
    xf = np.zeros((NT, K)); xf[:R] = rng.normal(0, 1, (R, K))
    x = np.zeros_like(xf); xb = bytearray(KG * XEL)
    for kg in range(KG):
        xs = xf[:, kg * 256:(kg + 1) * 256]
        sx = (np.maximum(np.abs(xs).max(1), 1e-30) / 126).astype(bfloat16).astype(np.float64)
        xq = np.clip(np.rint(xs / sx[:, None]), -127, 127).astype(np.int8)
        x[:, kg * 256:(kg + 1) * 256] = xq * sx[:, None]
        hdr = np.zeros(16, np.int32); hdr[0], hdr[2] = S, R
        xa = xq.reshape(NTG, 4, 16, 16).transpose(2, 0, 1, 3)
        s = xq.astype(np.int32).reshape(NT, 8, 32).sum(2)
        qs = np.repeat(s.T.reshape(8, NTG, 4, 1), 16, 3).astype(np.int16)
        sxt = np.repeat(sx.reshape(NTG, 4, 1), 16, 2).astype(bfloat16)
        el = hdr.tobytes() + xa.tobytes() + qs.tobytes() + sxt.tobytes()
        assert len(el) == XEL
        xb[kg * XEL:(kg + 1) * XEL] = el
    W = q * np.repeat(d.astype(np.float64), 32, 1) + np.repeat(m.astype(np.float64), 32, 1)
    (out / "w.bin").write_bytes(bytes(w)); (out / "x.bin").write_bytes(bytes(xb))
    np.save(out / "ref.npy", x[:R] @ W.T); np.save(out / "geo.npy", np.array([Q, K, R]))
    print("w", len(w), "x", len(xb), "y", 32 * Q * 4 * NTG * 64 * 4, "S", S)
else:
    Q, K, R = np.load(out / "geo.npy"); ref = np.load(out / "ref.npy")
    y = np.fromfile(out / "y.bin", np.float32).reshape(8, Q, 4, 4, NTG, 4, 16)   # [c][j][r][rg][tg][t][n]
    got = y.transpose(4, 5, 0, 2, 1, 3, 6).reshape(NT, -1)     # token, then rows (c, r, j, rg, n): band (4c + r) Q + j
    got = got[:R]
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"rel {rel:.3e}  rows {got.shape[1]}  worst row-token {np.abs(got - ref).max() / np.abs(ref).max():.3e}")
