#pragma once
// bfp16ebs8 encoding of 8 floats (dit2's A blocks: an exponent byte, 8 int8 mantissas) and where a
// row's 8-wide block sits in dit2's packed A (core.cpp dit2_pack_a's order), shared by the GEMM
// input packs and the DeltaNet rule, which writes the OUT GEMM's A rows directly.
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

namespace open_qwen36 {

// The same block with AVX2 (bit-identical: every shift past 24 yields 0 for a 24-bit mantissa,
// so the int32 lanes never overflow, and the arithmetic shift is the same floor).
__attribute__((target("avx2"))) inline void bfp_encode8_avx2(const float* v, uint8_t* blk) {
    const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(v));
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xFF));
    __m256i mx = _mm256_max_epi32(e, _mm256_permute2x128_si256(e, e, 1));
    mx = _mm256_max_epi32(mx, _mm256_shuffle_epi32(mx, 0x4E));
    mx = _mm256_max_epi32(mx, _mm256_shuffle_epi32(mx, 0xB1));
    const __m256i implicit = _mm256_andnot_si256(_mm256_cmpeq_epi32(e, _mm256_setzero_si256()), _mm256_set1_epi32(1 << 23));
    __m256i mant = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi32(0x7FFFFF)), implicit);
    const __m256i neg = _mm256_srai_epi32(bits, 31);                      // -1 where negative
    mant = _mm256_sub_epi32(_mm256_xor_si256(mant, neg), neg);           // two's complement negate
    const __m256i shift = _mm256_add_epi32(_mm256_sub_epi32(mx, e), _mm256_set1_epi32(17));
    const __m256i big = _mm256_cmpgt_epi32(shift, _mm256_set1_epi32(24));
    const __m256i sh = _mm256_min_epi32(shift, _mm256_set1_epi32(24));
    const __m256i half = _mm256_sllv_epi32(_mm256_set1_epi32(1), _mm256_sub_epi32(sh, _mm256_set1_epi32(1)));
    // nearest, ties to even (ties up biased bf16 inputs: they tie often, +0.0022 mean on V)
    const __m256i q0 = _mm256_srav_epi32(mant, sh);                                  // floor
    const __m256i r = _mm256_sub_epi32(mant, _mm256_sllv_epi32(q0, sh));
    const __m256i up = _mm256_or_si256(_mm256_cmpgt_epi32(r, half),
                                       _mm256_and_si256(_mm256_cmpeq_epi32(r, half),
                                                        _mm256_cmpeq_epi32(_mm256_and_si256(q0, _mm256_set1_epi32(1)),
                                                                           _mm256_set1_epi32(1))));
    __m256i q = _mm256_sub_epi32(q0, up);                                             // up is -1 where set
    q = _mm256_andnot_si256(big, q);
    q = _mm256_max_epi32(_mm256_min_epi32(q, _mm256_set1_epi32(127)), _mm256_set1_epi32(-128));
    const __m128i lo = _mm256_castsi256_si128(q), hi = _mm256_extracti128_si256(q, 1);
    const __m128i w16 = _mm_packs_epi32(lo, hi);
    const __m128i b8 = _mm_packs_epi16(w16, w16);
    blk[0] = static_cast<uint8_t>(_mm256_cvtsi256_si32(mx));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(blk + 1), b8);
}


/// The byte offset of row r's block at K column k0 (a multiple of 8) in dit2_pack_a's layout for
/// K columns, mt-row tiles, hs order or not.
inline size_t dit2_a_block_off(size_t r, size_t k0, size_t K, size_t mt, bool hs) {
    const size_t rc = r / mt, rr = r % mt, q = rr / 32, rb = (rr % 32) / 8, i = rr % 8, ks = k0 / 64, kb = (k0 % 64) / 8;
    const size_t j = hs ? (rb / 2) * 16 + 2 * kb + rb % 2 : kb * 4 + rb;
    return ((rc * (K / 64) + ks) * (mt / 32) * 256 + (q * 32 + j) * 8 + i) * 9;
}

}  // namespace open_qwen36
