"""dit2_dec test: random A [32, K] and int4 B [K, NT]; run.cfg for the harness; --check compares c.bin.

    python dec_test.py --build <dir> --K 5120 --NT 16384 --N 4096 --OFF 4096 --out <dir> [--runs 30]
    python dec_test.py --check --K .. --N .. --OFF .. --out <dir>
"""
import argparse, sys
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "dit_gemm"))
from pack_a import pack_a2, unpack_a2
from pack import pack_b4, unpack_b4
ap = argparse.ArgumentParser()
ap.add_argument("--build"); ap.add_argument("--K", type=int); ap.add_argument("--NT", type=int); ap.add_argument("--N", type=int)
ap.add_argument("--OFF", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--check", action="store_true")
ap.add_argument("--KROW", type=int, default=0, help="the weight's full K (a K-split kernel reads the first K)")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
K, N, OFF = a.K, a.N, a.OFF
KR = a.KROW or K
if a.check:
    ref = np.load(out / "ref.npy")
    raw = np.fromfile(out / "c.bin", dtype=bfloat16).astype(np.float64).reshape(N // 128, 4, 16, 8, 8)   # [strip][rb][cb][8][8]
    c = raw.transpose(1, 3, 0, 2, 4).reshape(32, N)          # [rb][r8][strip][cb][c8] -> rows x cols
    rel = np.linalg.norm(c - ref) / np.linalg.norm(ref)
    r0 = np.linalg.norm(c[0] - ref[0]) / np.linalg.norm(ref[0])
    print(("PASS" if rel < 3e-2 and np.isfinite(c).all() else "FAIL"), f"rel_fro {rel:.3e} row0 {r0:.3e}")
    sys.exit(0)
rng = np.random.default_rng(0)
A = rng.standard_normal((32, KR)).astype(np.float32)
Bm = (rng.standard_normal((KR, a.NT)) * 0.02).astype(np.float32)
A128 = np.zeros((128, KR), np.float32); A128[:32] = A
pa = pack_a2(A128, mt=32)[: KR // 64 * 2304]               # core row 0's units: rows 0..31
pb = pack_b4(Bm)
ref = (unpack_a2(pack_a2(A128, mt=32), 128, KR, mt=32)[:32] @ unpack_b4(pb, KR, a.NT)[:, OFF:OFF + N]) if KR == K else \
      (unpack_a2(pack_a2(A128, mt=32), 128, KR, mt=32)[:32, :K] @ unpack_b4(pb, KR, a.NT)[:K, OFF:OFF + N])
np.save(out / "ref.npy", ref.astype(np.float32))
pa.tofile(out / "a.bin"); pb.tofile(out / "b.bin")
b = Path(a.build).resolve(); cb = N // 128 * 4096 * 2
(out / "run.cfg").write_text(f"device\nxclbin G {b}/final.xclbin\nkernelx k G {b}/insts.bin\nbuf a {pa.size} {out.resolve()}/a.bin\n"
                             f"buf b {pb.size} {out.resolve()}/b.bin\nbuf c {cb}\n" + "run k a b c\n" * a.runs +
                             f"dump c {out.resolve()}/c.bin {cb}\n")
print(f"B {pb.size / 1e6:.1f} MB, read {N * K * 0.515625 / 1e6:.2f} MB a run")
