r"""Test vectors for dit_fa256b: random Q [L, 24 x 256], K and V [lk, 4 x 256], packed per head;
rows of K / V at or past --valid are random junk the mask must hide. Writes q.bin, k.bin, v.bin,
ref.npz and run.cfg.

    python make_test.py --xclbin <x> --insts <i> --out <dir> --L 1024 --lk 8192 --valid 8000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from pack_fa import pack_k, pack_q, pack_v

D, HEADS, KVH = 256, 24, 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True); ap.add_argument("--insts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=1024); ap.add_argument("--lk", type=int, required=True)
    ap.add_argument("--valid", type=int, default=0)
    ap.add_argument("--qk-scale", type=float, default=1.0, help="score spread (Q multiplier)")
    ap.add_argument("--causal", action="store_true", help="queries are the last L positions of the lk keys")
    ap.add_argument("--qoff", type=int, default=-1, help="causal: the queries' first position (default lk - L)")
    ap.add_argument("--kv-inter", action="store_true", help="K / V as [chunk][kv head] (DF_KV_INTER=1)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    valid = a.valid or a.lk
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    q = (rng.standard_normal((a.L, HEADS, D), dtype=np.float32) * a.qk_scale).astype(bfloat16).astype(np.float32)
    k = rng.standard_normal((a.lk, KVH, D), dtype=np.float32).astype(bfloat16).astype(np.float32)
    v = rng.standard_normal((a.lk, KVH, D), dtype=np.float32).astype(bfloat16).astype(np.float32)
    k[valid:] *= 50.0                                 # junk past valid_len: large, so a leak shows
    v[valid:] *= 50.0
    qb = np.concatenate([pack_q(q[:, h]) for h in range(HEADS)])
    kb = np.concatenate([pack_k(k[:, g]) for g in range(KVH)])
    vb = np.concatenate([pack_v(v[:, g]) for g in range(KVH)])
    if a.kv_inter:
        kb = kb.reshape(KVH, a.lk // 64, -1).transpose(1, 0, 2).copy().reshape(-1)
        vb = vb.reshape(KVH, a.lk // 64, -1).transpose(1, 0, 2).copy().reshape(-1)
    qb.tofile(out / "q.bin"); kb.tofile(out / "k.bin"); vb.tofile(out / "v.bin")
    np.savez(out / "ref.npz", q=q, k=k, v=v, valid=valid, causal=int(a.causal),
             q_off=(a.qoff if a.qoff >= 0 else a.lk - a.L) if a.causal else 0)
    ob = a.L * HEADS * D * 2
    cfg = ["device", f"xclbin G {Path(a.xclbin).resolve()}", f"kernelx k G {Path(a.insts).resolve()}",
           f"buf q {qb.nbytes} {out / 'q.bin'}", f"buf k {kb.nbytes} {out / 'k.bin'}",
           f"buf v {vb.nbytes} {out / 'v.bin'}", f"buf o {ob}"]
    cfg += ["run k q k v o"] * a.runs + [f"dump o {out / 'o.bin'} {ob}", ""]
    (out / "run.cfg").write_text("\n".join(cfg))
    print(f"L={a.L} lk={a.lk} valid={valid}: {4 * HEADS * a.L * valid * D / 1e9:.1f} GFLOP -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
