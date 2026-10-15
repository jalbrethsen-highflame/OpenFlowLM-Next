r"""Host packing for dit_fa256b: Q, K and V in bfp16ebs8, in the order fa256b.cc's MAC pops them.

    Q (one head, [L, 256])  -> per 32-row block: [row pair 2][d step 32][tile 2][8 rows][8 d]
    K (one kv head, [lk, 256]) -> per 64-key chunk, per 16-key piece: [d step 32][tile 2][8 keys][8 d]
    V (one kv head, [lk, 256]) -> per 64-key chunk, per 64-column element, per 16-column pair:
                                  [key step 8][tile 2][8 d columns][8 keys]

A bfp block (1 exponent byte + 8 int8 mantissas) is the last axis: 8 d of a row or key for Q
and K, 8 keys of a d column for V. Byte counts: 9216 per Q row block, 18432 per K or V chunk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dit_gemm"))
from pack import bfp16ebs8_decode, bfp16ebs8_encode  # noqa: E402

D = 256
QSCALE = np.log2(np.e) / np.sqrt(D)   # Q packed pre-scaled: the kernel's scores are in log2 units


def q_order(q: np.ndarray) -> np.ndarray:
    L = q.shape[0]
    t = q.reshape(L // 32, 2, 2, 8, D // 8, 8)       # [rb, zp, r2, r8, i, d8]
    return t.transpose(0, 1, 4, 2, 3, 5).reshape(-1)


def k_order(k: np.ndarray) -> np.ndarray:
    lk = k.shape[0]
    t = k.reshape(lk // 64, 4, 2, 8, D // 8, 8)      # [c, j, n2, k8, i, d8]
    return t.transpose(0, 1, 4, 2, 3, 5).reshape(-1)


def v_order(v: np.ndarray) -> np.ndarray:
    lk = v.shape[0]
    t = v.reshape(lk // 64, 8, 8, 4, 4, 2, 8)        # [c, i, k8, e, jl, n2, dc8]
    return t.transpose(0, 3, 4, 1, 5, 6, 2).reshape(-1)


def pack_q(q: np.ndarray) -> np.ndarray:
    """q unscaled; packed times QSCALE."""
    return bfp16ebs8_encode(q_order(np.asarray(q, dtype=np.float32) * np.float32(QSCALE)))


def pack_k(k: np.ndarray) -> np.ndarray:
    return bfp16ebs8_encode(k_order(np.asarray(k, dtype=np.float32)))


def pack_v(v: np.ndarray) -> np.ndarray:
    return bfp16ebs8_encode(v_order(np.asarray(v, dtype=np.float32)))


def round_rows(x: np.ndarray) -> np.ndarray:
    """x [n, 256] as bfp16 with blocks of 8 along the row (what Q and K become)."""
    return bfp16ebs8_decode(bfp16ebs8_encode(np.asarray(x, dtype=np.float32).reshape(-1))).reshape(x.shape)


def round_cols(x: np.ndarray) -> np.ndarray:
    """x [n, 256] as bfp16 with blocks of 8 down each column (what V becomes)."""
    return round_rows(np.ascontiguousarray(np.asarray(x).T)).T
