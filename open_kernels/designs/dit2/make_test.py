"""dit2 correctness: random A, B -> a.bin (pack_a), b.bin (pack_b), ref.npy; run.cfg for the harness."""
import sys, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "dit_gemm"))
from pack_a import pack_a, unpack_a
from pack import pack_b, unpack_b
ap = argparse.ArgumentParser(); ap.add_argument("--build"); ap.add_argument("--M", type=int); ap.add_argument("--K", type=int)
ap.add_argument("--N", type=int); ap.add_argument("--out"); ap.add_argument("--epi", action="store_true")
a = ap.parse_args(); M, K, N = a.M, a.K, a.N; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
A = rng.standard_normal((M, K)).astype(np.float32) * np.exp(rng.standard_normal((M, 1))).astype(np.float32)
B = (rng.standard_normal((K, N)) * 0.02).astype(np.float32)
if a.epi:
    from pack import interleave_swiglu
    B = interleave_swiglu(B)
pa, pb = pack_a(A), pack_b(B)
pa.tofile(out / "a.bin"); pb.tofile(out / "b.bin")
ref = unpack_a(pa, M, K) @ unpack_b(pb, K, N)
if a.epi:   # columns 0..63 of every 128 become silu(gate) * up; compare those only
    r3 = ref.reshape(M, N // 128, 128).copy()
    g, u = r3[:, :, :64], r3[:, :, 64:]
    r3[:, :, :64] = g / (1 + np.exp(-g)) * u
    ref = r3.reshape(M, N)
    np.save(out / "mask.npy", np.tile(np.r_[np.ones(64), np.zeros(64)], N // 128).astype(bool))
np.save(out / "ref.npy", ref.astype(np.float32))
b = Path(a.build)
(out / "run.cfg").write_text(f"device\nxclbin G {b}/final.xclbin\nkernelx k G {b}/insts.bin\nbuf a {pa.size} a.bin\n"
                             f"buf b {pb.size} b.bin\nbuf c {M*N*2}\nrun k a b c\nrun k a b c\ndump c c.bin {M*N*2}\n")
