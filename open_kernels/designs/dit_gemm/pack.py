r"""Host packing for dit_gemm's B operand: bfp16ebs8 in the order the kernel streams it.

One function (pack_b) is shared by the tests and the model converter, so what is tested
is what ships.

bfp16ebs8: blocks of 8 values share one exponent byte (the largest element's IEEE
exponent), followed by 8 two's-complement int8 mantissas; value = mant * 2^(exp-127) / 64.
9 bytes per 8 values. The encoder rounds to nearest, ties to even (it was ties up, a bias on bf16 inputs); the device
only decodes, so the rounding is the host's choice -- nearest measured 1.25e-2 end to end
against truncation's 1.52e-2 (utilities/dit-gemm-bench/README.md).

Layout, per 64 (K) x 128 (N) tile: [n-block 16][k-block 8][8 n rows][8 k values], a bfp
block being the 8 k values of one output column. Tiles are column-major: all K/64 tiles of
output columns 0..127, then 128..255, and so on -- the order dit_gemm.py's B fill walks.
"""

from __future__ import annotations

import numpy as np

K_T, N_T = 64, 128


def bfp16ebs8_encode(x: np.ndarray) -> np.ndarray:
    """float32 [n*8] -> uint8 [n*9], blocks of 8 consecutive values, round to nearest."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).reshape(-1, 8)
    exp = ((bits >> 23) & 0xFF).astype(np.int64)
    max_exp = exp.max(axis=1, keepdims=True)
    mant = (bits & 0x7FFFFF).astype(np.int64) | np.where(exp != 0, 1 << 23, 0)
    signed = np.where(bits >> 31 == 1, -mant, mant)
    shift = 17 + np.minimum(max_exp - exp, 31)
    half = np.left_shift(np.int64(1), shift - 1)
    q = np.right_shift(signed, shift)                     # floor
    r = signed - np.left_shift(q, shift)
    q = q + ((r > half) | ((r == half) & (q & 1 == 1)))   # nearest, ties to even (bf16 inputs tie often)
    vals = np.clip(q, -128, 127)
    out = np.empty((bits.shape[0], 9), dtype=np.uint8)
    out[:, 0] = max_exp[:, 0].astype(np.uint8)
    out[:, 1:] = vals.astype(np.int8).view(np.uint8)
    return out.reshape(-1)


def bfp16ebs8_decode(b: np.ndarray) -> np.ndarray:
    blk = np.asarray(b, dtype=np.uint8).reshape(-1, 9)
    scale = np.ldexp(1.0, blk[:, :1].astype(np.int64) - 127) / 64.0
    return (blk[:, 1:].view(np.int8).astype(np.float64) * scale).reshape(-1)


def layout(b: np.ndarray) -> np.ndarray:
    """[K, N] -> the kernel's streaming order (flat)."""
    K, N = b.shape
    assert K % K_T == 0 and N % N_T == 0, (K, N)
    t = b.reshape(K // K_T, K_T // 8, 8, N // N_T, N_T // 8, 8)
    #          [K tile,   k-block,  kin, N tile,  n-block,  nin]
    return t.transpose(3, 0, 4, 1, 5, 2).reshape(-1)


def unlayout(flat: np.ndarray, K: int, N: int) -> np.ndarray:
    t = np.asarray(flat).reshape(N // N_T, K // K_T, N_T // 8, K_T // 8, 8, 8)
    return t.transpose(1, 3, 5, 0, 2, 4).reshape(K, N)


def layout2(b: np.ndarray) -> np.ndarray:
    """[K, N] -> dit2's hand-scheduled 2x2 order (DIT_HS): per [N tile][K tile], column-block
    pairs, each [k-block 8][the pair's 2 column blocks], blocks of 8 columns x 8 k as layout()."""
    K, N = b.shape
    assert K % K_T == 0 and N % N_T == 0, (K, N)
    t = b.reshape(K // K_T, K_T // 8, 8, N // N_T, N_T // 16, 2, 8)
    #          [K tile,   k-block,  kin, N tile,  n-pair,    n2, nin]
    return t.transpose(3, 0, 4, 1, 5, 6, 2).reshape(-1)


def unlayout2(flat: np.ndarray, K: int, N: int) -> np.ndarray:
    t = np.asarray(flat).reshape(N // N_T, K // K_T, N_T // 16, K_T // 8, 2, 8, 8)
    return t.transpose(1, 3, 6, 0, 2, 4, 5).reshape(K, N)


def pack_b2(b: np.ndarray) -> np.ndarray:
    return bfp16ebs8_encode(layout2(np.asarray(b, dtype=np.float32)))


def unpack_b2(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    return unlayout2(bfp16ebs8_decode(packed), K, N)


B4_TILE = 4096 + 128   # bytes per [128 columns x 64 k] tile: nibbles, then exponents


def pack_b4(b: np.ndarray) -> np.ndarray:
    """B [K, N] -> dit2's int4-mantissa weights (DIT_B4, hsgen HSGEN_B4): per [N tile][K tile]
    4096 bytes of nibbles, blocks [column pair 8][k-block 8][2] of 8 columns x 8 k (column-major
    inside: value i = column * 8 + k, low nibble first), then 128 exponent bytes, one per column for
    the tile's 64 of K. A weight is mantissa (-8..7) x 2^(exponent - 133), bfp16ebs8's scale, the
    exponent the smallest that keeps the column's 64 values within +-7 before rounding, or one less
    when clipping to -8..7 there gives a lower squared error (dit_encode_b4's rule)."""
    b = np.asarray(b, dtype=np.float64)
    K, N = b.shape
    assert K % K_T == 0 and N % N_T == 0, (K, N)
    t = b.reshape(K // K_T, K_T, N // N_T, N_T).transpose(2, 0, 3, 1)       # [nt, kt, col 128, k 64]
    amax = np.abs(t).max(-1)                                                  # [nt, kt, 128]
    e2 = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-300) / 7.0)), -133).astype(np.int64)
    e2 = np.clip(e2, -133, 122)

    def q(e):
        return np.clip(np.rint(t / np.ldexp(1.0, e)[..., None]), -8, 7)

    err = [((q(e) * np.ldexp(1.0, e)[..., None] - t) ** 2).sum(-1) for e in (e2, e2 - 1)]
    e2 = np.where((amax > 0) & (e2 > -133) & (err[1] < err[0]), e2 - 1, e2)
    m = q(e2).astype(np.int64)
    # blocks [pair 8][kb 8][n2 2][col-in-block 8][k-in-block 8]
    m = m.reshape(N // N_T, K // K_T, 8, 2, 8, 8, 8).transpose(0, 1, 2, 5, 3, 4, 6)
    nib = (m & 15).reshape(N // N_T, K // K_T, -1)
    packed = (nib[..., 0::2] | (nib[..., 1::2] << 4)).astype(np.uint8)       # [nt, kt, 4096]
    ex = (e2 + 133).astype(np.uint8)                                          # [nt, kt, 128]
    return np.concatenate([packed, ex], axis=-1).reshape(-1)


def unpack_b4(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    """The B dit2 multiplies by under DIT_B4, as float64 [K, N]."""
    t = np.asarray(packed, dtype=np.uint8).reshape(N // N_T, K // K_T, B4_TILE)
    nb, ex = t[..., :4096], t[..., 4096:].astype(np.int64)
    v = np.stack([nb & 15, nb >> 4], -1).reshape(N // N_T, K // K_T, 8192).astype(np.int64)
    v = np.where(v >= 8, v - 16, v)
    m = v.reshape(N // N_T, K // K_T, 8, 8, 2, 8, 8).transpose(0, 1, 2, 4, 5, 3, 6).reshape(N // N_T, K // K_T, 128, 64)
    w = m * np.ldexp(1.0, ex - 133)[..., None]                                # [nt, kt, col, k]
    return w.transpose(1, 3, 0, 2).reshape(K, N)


def pack_b(b: np.ndarray) -> np.ndarray:
    """B [K, N] (any float dtype) -> the uint8 buffer dit_gemm reads, K*N*9/8 bytes."""
    return bfp16ebs8_encode(layout(np.asarray(b, dtype=np.float32)))


def unpack_b(packed: np.ndarray, K: int, N: int) -> np.ndarray:
    """The B the kernel actually multiplies by, as float64 [K, N]."""
    return unlayout(bfp16ebs8_decode(packed), K, N)


def interleave_swiglu(b: np.ndarray, first_col: int = 0) -> np.ndarray:
    """[K, N] weight whose columns [first_col, N) are a SwiGLU input -- gate (first half)
    then up (second half), as FLUX.2's Flux2SwiGLU and Qwen3's gate/up split them -- to
    dit_gemm's epilogue order: every 128-column tile of that range holds 64 gate columns
    and the matching 64 up columns. Columns before first_col are untouched."""
    K, N = b.shape
    half = (N - first_col) // 2
    assert (N - first_col) % 2 == 0 and half % 64 == 0, (N, first_col)
    gate = b[:, first_col:first_col + half].reshape(K, half // 64, 64)
    up = b[:, first_col + half:].reshape(K, half // 64, 64)
    mixed = np.concatenate([gate, up], axis=2).reshape(K, 2 * half)
    return np.concatenate([b[:, :first_col], mixed], axis=1)
