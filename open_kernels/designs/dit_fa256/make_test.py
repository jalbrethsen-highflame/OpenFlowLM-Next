r"""Test vectors for dit_fa256 in the engine's layout: Q [L, heads*256], a KV cache [lk, 2048]
(rows [k kv_heads*256 | v kv_heads*256]), O [L, heads*256]; causal with query offset q_off
(row t sees keys 0 .. q_off + t). Writes q.bin, kv.bin, ref.npz and run.cfg (K and V are the
same buffer, passed twice).

    python make_test.py --xclbin <x> --insts <i> --out <dir> --L 1024 --lk 8192 --qoff 7168
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16

D, HEADS, KVH = 256, 24, 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True); ap.add_argument("--insts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=1024); ap.add_argument("--lk", type=int, required=True)
    ap.add_argument("--qoff", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    q = rng.standard_normal((a.L, HEADS * D), dtype=np.float32).astype(bfloat16)
    kv = rng.standard_normal((a.lk, 2 * KVH * D), dtype=np.float32).astype(bfloat16)
    q.tofile(out / "q.bin"); kv.tofile(out / "kv.bin")
    np.savez(out / "ref.npz", q=q.view(np.uint16), kv=kv.view(np.uint16), L=a.L, lk=a.lk, qoff=a.qoff)
    ob = a.L * HEADS * D * 2
    cfg = ["device", f"xclbin G {Path(a.xclbin).resolve()}", f"kernelx k G {Path(a.insts).resolve()}",
           f"buf q {q.nbytes} q.bin", f"buf kv {kv.nbytes} kv.bin", f"buf o {ob}"]
    cfg += ["run k q kv kv o"] * a.runs + [f"dump o o.bin {ob}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    pairs = sum(min(a.lk, a.qoff + t + 1) for t in range(a.L))
    print(f"L={a.L} lk={a.lk} qoff={a.qoff}: causal {4 * HEADS * pairs * D / 1e9:.1f} GFLOP -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
