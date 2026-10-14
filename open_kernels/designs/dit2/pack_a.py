"""Host packing for dit2's A operand: bfp16ebs8 blocks in the order the cores consume them.

A [M, K] -> [row group M/512][core row 4][k-step K/64][quarter 4][k-block 8][row-block 4]
blocks, each 8 rows x 8 k row-major, every row's 8 k sharing one exponent (pack.py's
bfp16ebs8_encode, round to nearest). The memtile forwards it linearly, so this order is the
whole contract with mm_dit2.cc's matmul_4x1_bfp16."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dit_gemm"))
from pack import bfp16ebs8_decode, bfp16ebs8_encode  # noqa: E402


def order(a: np.ndarray) -> np.ndarray:
    M, K = a.shape
    assert M % 512 == 0 and K % 64 == 0, (M, K)
    t = a.reshape(M // 512, 4, 4, 4, 8, K // 64, 8, 8)
    #            [rg, crow, quarter, rowblock, r8, kstep, kblock, k8]
    return t.transpose(0, 1, 5, 2, 6, 3, 4, 7).reshape(-1)


def unorder(flat: np.ndarray, M: int, K: int) -> np.ndarray:
    t = np.asarray(flat).reshape(M // 512, 4, K // 64, 4, 8, 4, 8, 8)
    #                          [rg, crow, kstep, quarter, kblock, rowblock, r8, k8]
    return t.transpose(0, 1, 3, 5, 6, 2, 4, 7).reshape(M, K)


def pack_a(a: np.ndarray) -> np.ndarray:
    return bfp16ebs8_encode(order(np.asarray(a, dtype=np.float32)))


def unpack_a(packed: np.ndarray, M: int, K: int) -> np.ndarray:
    return unorder(bfp16ebs8_decode(packed), M, K)
