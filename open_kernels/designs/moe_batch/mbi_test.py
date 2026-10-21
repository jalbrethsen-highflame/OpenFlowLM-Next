"""inputs + reference for mbi.py (MBI_SLOTS = MBI_EXPERTS = E, slot s = expert s / MBI_P: the
kernel's visit v is compiled as expert v): MBI_P=p mbi_test.py gen OUTDIR [E] [full | rand] | check OUTDIR"""
import os, sys, numpy as np
from pathlib import Path
from ml_dtypes import bfloat16

mode, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(exist_ok=True, parents=True)
E = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NV_MODE = sys.argv[4] if len(sys.argv) > 4 else "full"     # full | rand: a slot's real tokens
HID, FF, NT, NTG, ROWS = 2048, 512, 16, 4, 4
TOK = NT * ROWS
CH = 5120
STRIPE = 128 * HID * 5 // 8
EXP_UG = 2 * (FF // 128) * STRIPE
DOWN_BYTES = HID * FF * 5 // 8
P = int(os.environ.get("MBI_P", 1))
QSC = os.environ.get("MBI_QSC", "0") == "1"      # compact block sums: int16 [NTG][2 block halves][4 tokens][4 blocks]
XEL = 64 + 16 * NTG * 64 + (NTG * 4 * 8 * 2 if QSC else 8 * NTG * 128) + NTG * 128


def q41_matrix(rng, rows, K):
    """a q4_1 matrix: d, m [rows][K/32] bf16, q [rows][K]"""
    d = (np.abs(rng.normal(0, 0.004, (rows, K // 32))) + 5e-4).astype(bfloat16)
    m = (-8 * d.astype(np.float32) + rng.normal(0, 0.002, (rows, K // 32))).astype(bfloat16)
    q = rng.integers(0, 16, (rows, K))
    return d, m, q


def shift_for(mats):
    dmax = max(np.abs(d.astype(np.float64)).max() for d, m, q in mats)
    mcmax = max(np.abs(m.astype(np.float64) + 8 * d.astype(np.float64)).max() for d, m, q in mats)
    return int(min(np.floor(np.log2(8000 / dmax)), np.floor(np.log2(32000 / mcmax))))


def chunk_bytes(d, m, q, r0, kg, S):
    """the transformed chunk for rows r0..r0+31, k-group kg: d_i, mc_i int16 [8 blk][32 rows], nibbles ^ 0x88"""
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


def deq(d, m, q):
    return q * np.repeat(d.astype(np.float64), 32, 1) + np.repeat(m.astype(np.float64), 32, 1)


def silu(v):
    return v / (1 + np.exp(-v))


if mode == "gen":
    rng = np.random.default_rng(29)
    pool = bytearray(E * (EXP_UG + DOWN_BYTES))
    xb = bytearray(E * ROWS * 8 * XEL)
    ref = np.zeros((E, TOK, HID))
    nvs = np.full(E, TOK) if NV_MODE == "full" else rng.integers(1, TOK + 1, E)
    for slot in range(E):
      e = slot // P
      if slot % P == 0:
        up, gate, dn = q41_matrix(rng, FF, HID), q41_matrix(rng, FF, HID), q41_matrix(rng, HID, FF)
        S_ug, S_dn = shift_for([up, gate]), shift_for([dn])
        for sp in range(FF // 128):
            for proj, mat in enumerate((up, gate)):
                for kg in range(HID // 256):
                    for ci in range(4):
                        off = e * EXP_UG + (sp * 2 + proj) * STRIPE + (kg * 4 + ci) * CH
                        pool[off:off + CH] = chunk_bytes(*mat, 128 * sp + 32 * ci, kg, S_ug)
        for st in range(HID // 128):
            for kg in range(FF // 256):
                for ci in range(4):
                    off = E * EXP_UG + e * DOWN_BYTES + st * 128 * FF * 5 // 8 + (kg * 4 + ci) * CH
                    pool[off:off + CH] = chunk_bytes(*dn, 128 * st + 32 * ci, kg, S_dn)
      if True:
        xf = rng.normal(0, 1, (TOK, HID))
        x = np.zeros_like(xf)
        for r in range(ROWS):
            for kg in range(8):
                xs = xf[r * NT:(r + 1) * NT, kg * 256:(kg + 1) * 256]
                sx = (np.abs(xs).max(1) / 126).astype(bfloat16).astype(np.float64)   # the scale as bf16
                xq = np.clip(np.rint(xs / sx[:, None]), -127, 127).astype(np.int8)
                x[r * NT:(r + 1) * NT, kg * 256:(kg + 1) * 256] = xq * sx[:, None]
                hdr = np.zeros(16, np.int32); hdr[0], hdr[1] = S_ug, S_dn
                hdr[2] = min(max(int(nvs[slot]) - NT * r, 0), NT)
                xa = xq.reshape(NTG, 4, 16, 16).transpose(2, 0, 1, 3)          # [kt][tg][t][k]
                s = xq.astype(np.int32).reshape(NT, 8, 32).sum(2)               # [tok][blk]
                qs = (s.reshape(NTG, 4, 2, 4).transpose(0, 2, 1, 3) if QSC else np.repeat(s.T.reshape(8, NTG, 4, 1), 16, 3)).astype(np.int16)
                sxt = np.repeat(sx.reshape(NTG, 4, 1), 16, 2).astype(bfloat16)
                el = hdr.tobytes() + xa.tobytes() + qs.tobytes() + sxt.tobytes()
                assert len(el) == XEL
                off = ((slot * ROWS + r) * 8 + kg) * XEL
                xb[off:off + XEL] = el
        h = silu(x @ deq(*gate).T) * (x @ deq(*up).T)
        ref[slot] = h @ deq(*dn).T
        print(f"slot {slot} expert {e}: S_ug {S_ug} S_dn {S_dn} |y| {np.abs(ref[e]).mean():.3e}")
    (out / "pool.bin").write_bytes(bytes(pool)); (out / "x.bin").write_bytes(bytes(xb))
    np.save(out / "ref.npy", ref); np.save(out / "nv.npy", nvs)
    print("pool", len(pool), "x", len(xb), "y", E * TOK * HID * 2)
else:
    ref = np.load(out / "ref.npy"); E = ref.shape[0]
    got = np.fromfile(out / "y.bin", bfloat16).astype(np.float64).reshape(E, TOK, HID)   # [slot][token][2048]
    nvs = np.load(out / "nv.npy") if (out / "nv.npy").exists() else np.full(E, TOK)
    for e in range(E):
        n = int(nvs[e]); g, r = got[e, :n], ref[e, :n]
        rel = np.linalg.norm(g - r) / np.linalg.norm(r)
        rows = np.linalg.norm(g - r, axis=1) / np.linalg.norm(r, axis=1)
        # a padded token in a computed pair is computed from x = 0 (y 0); later pairs stay 0
        skip = [t for t in range(TOK) if t % NT >= ((min(max(n - NT * (t // NT), 0), NT) + 7) // 8) * 8]
        pad = np.abs(got[e, skip]).max() if skip else 0.0
        print(f"expert {e}: rel {rel:.3e} worst token {rows.max():.3e}  n {n}  pad max {pad:.2e}")
