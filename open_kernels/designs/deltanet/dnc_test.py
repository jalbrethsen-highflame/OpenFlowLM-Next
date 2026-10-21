"""dnc.py data + check: dnc_test.py gen DIR [steps] | check DIR (rel ~1.3 % O, ~1 % S against the recurrence in f64)"""
import sys, numpy as np
from pathlib import Path
from ml_dtypes import bfloat16
mode, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
NS = int(sys.argv[3]) if len(sys.argv) > 3 else (64 if "64" in sys.argv[2] else 8)
H, KH, D, T, C = 32, 16, 128, NS * 16, 16
bfv = lambda x: np.asarray(x, np.float64).astype(bfloat16)
bf = lambda x: bfv(x).astype(np.float64)
def hilo(x):
    h = bfv(x); return h, bfv(np.asarray(x, np.float64) - h.astype(np.float64))
def norm(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
rng = np.random.default_rng(11)
if mode == "gen":
    K = bf(norm(rng.normal(size=(T, KH, D)))); Qm = bf(norm(rng.normal(size=(T, KH, D)))); V = bf(rng.normal(size=(T, H, D)))
    la = -np.abs(rng.normal(0.05, 0.08, (T, H))); b = 1 / (1 + np.exp(-rng.normal(0, 1, (T, H))))
    S0 = rng.normal(0, 0.3, (H, D, D)).astype(np.float32).astype(np.float64)
    rec = np.zeros((H, NS, 9216), np.uint8); vo = np.zeros((H, NS, 4, 16, 32), bfloat16); s = np.zeros((H, 16, 4, 256), np.float32)
    tiles = lambda x: bfv(x).reshape(2, 8, 16, 8).transpose(0, 2, 1, 3)
    for h in range(H):
        kh = h // 2
        for q in range(4):   # row q's slice: S^T tiles [db][kb][8][8] of dv 32q .. 32q + 31, as 16 pieces of 256
            st = S0[h][:, 32 * q:32 * q + 32].T.reshape(4, 8, 16, 8).transpose(0, 2, 1, 3).astype(np.float32).reshape(16, 256)
            s[h, :, q] = st
        for n in range(NS):
            sl = slice(n * C, (n + 1) * C)
            G = np.cumsum(la[sl, h]); Gm = G[7]
            e, f, Gam, w = np.exp(G - Gm), np.exp(Gm - G), np.exp(G), np.exp(G[15] - G)
            aux = np.zeros(512, bfloat16)
            for off, x in ((0, e), (32, f), (64, b[sl, h])):
                hh, ll = hilo(x); aux[off:off + 16] = hh; aux[off + 16:off + 32] = ll
            hh, ll = hilo(np.repeat(Gam.reshape(2, 8), 8, axis=1)); aux[96:224] = hh.reshape(-1); aux[224:352] = ll.reshape(-1)
            aux[352:480] = bfv(np.tile(w.reshape(2, 8), (1, 8))).reshape(-1)
            hh, ll = hilo(Gam[15]); aux[480] = hh; aux[481] = ll
            rec[h, n] = np.frombuffer(tiles(K[sl, kh]).tobytes() + tiles(Qm[sl, kh]).tobytes() + aux.tobytes(), np.uint8)
            for q in range(4): vo[h, n, q] = bfv(V[sl, h, 32 * q:32 * q + 32])
    (out / "rec.bin").write_bytes(rec.tobytes()); (out / "vo.bin").write_bytes(vo.tobytes()); (out / "s.bin").write_bytes(s.tobytes())
    Or = np.zeros((T, H, D)); Sr = np.zeros((H, D, D))
    a = np.exp(la)
    for h in range(H):
        S = S0[h].copy()
        for t in range(T):
            S = a[t, h] * S
            d = b[t, h] * (V[t, h] - S.T @ K[t, h // 2])
            S = S + np.outer(K[t, h // 2], d)
            Or[t, h] = S.T @ Qm[t, h // 2]
        Sr[h] = S
    np.save(out / "refO.npy", Or); np.save(out / "refS.npy", Sr)
    print("rec", rec.nbytes, "vo", vo.nbytes, "s", s.nbytes)
else:
    vo = np.fromfile(out / "vo_out.bin", bfloat16).astype(np.float64).reshape(H, NS, 4, 16, 32)
    O = vo.transpose(1, 3, 0, 2, 4).reshape(T, H, D)
    s = np.fromfile(out / "s_out.bin", np.float32).astype(np.float64).reshape(H, 16, 4, 256)
    S = np.zeros((H, D, D))
    for h in range(H):
        for q in range(4):
            S[h][:, 32 * q:32 * q + 32] = s[h, :, q].reshape(4, 16, 8, 8).transpose(0, 2, 1, 3).reshape(32, 128).T
    rO, rS = np.load(out / "refO.npy"), np.load(out / "refS.npy")
    rel = lambda x, y: np.linalg.norm(x - y) / np.linalg.norm(y)
    print(f"O rel {rel(O, rO):.3e}  S rel {rel(S, rS):.3e}")
    print("per head O rel", np.round([rel(O[:, h], rO[:, h]) for h in range(H)], 3))
