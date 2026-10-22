/// \file block_host.cpp
/// \brief The block prefill's host stages (see block_host.hpp).
///
/// Written for the compiler's vectoriser: float, contiguous inner loops over a
/// head's dims, the per-head work of a block spread over OpenMP threads
/// (heads are independent across every token of the block, so each thread
/// owns its heads and walks the tokens). Reductions that decide a norm or a
/// softmax accumulate in double.
#include "open_qwen36/block_host.hpp"

#include <algorithm>
#include <functional>
#include <immintrin.h>
#include <omp.h>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "open_qwen36/q4nx_file.hpp"   // bf16_to_f32 / f32_to_bf16
#include "open_qwen36/bfp_pack.hpp"

namespace open_qwen36 {
namespace host {

namespace {

float bf16r(float x) { return bf16_to_f32(f32_to_bf16(x)); }
float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 0 ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x)); }

// bf16r over eight floats: f32_to_bf16's integer rounding, so the same bits
inline __m256 bf16r8(const float* p) {
    const __m256i u = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i odd = _mm256_and_si256(_mm256_srli_epi32(u, 16), _mm256_set1_epi32(1));
    const __m256i r = _mm256_add_epi32(u, _mm256_add_epi32(odd, _mm256_set1_epi32(0x7FFF)));
    return _mm256_castsi256_ps(_mm256_and_si256(r, _mm256_set1_epi32(static_cast<int>(0xFFFF0000u))));
}

// x[d] / sqrt(mean(x^2) + eps) * w[d]
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void rms_vec(const float* x, size_t d, const float* w, double eps, float* out) {
    double ss = 0;
    for (size_t j = 0; j < d; ++j) ss += static_cast<double>(x[j]) * x[j];
    const float r = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(d) + eps));
    for (size_t j = 0; j < d; ++j) out[j] = x[j] * r * w[j];
}

// the partial rotation over the first 2 * half dims, half-split (Qwen's layout)
void rope(float* x, size_t half, const double* inv_freq, double pos) {
    for (size_t i = 0; i < half; ++i) {
        const double a = pos * inv_freq[i];
        const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
        const float x1 = x[i], x2 = x[half + i];
        x[i] = x1 * c - x2 * s;
        x[half + i] = x2 * c + x1 * s;
    }
}

}  // namespace

__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void rmsnorm_rows(const float* x, size_t T, size_t d, const float* w, double eps, float* out) {
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) rms_vec(x + t * d, d, w, eps, out + t * d);
}

namespace {

/// One 8x8 float tile, src rows `sstride` apart, dst rows `dstride` apart.
///
/// The scalar loop this replaces walks one side of every 32x32 block four bytes at a time --
/// 32 separate cache lines touched per source row -- and reached about 10 GB/s over sixteen
/// threads on a 2582-token prefill's 16 GB of transposes. The AVX2 network reads eight whole
/// 32-byte rows, shuffles them in registers and writes eight whole 32-byte rows, so both
/// sides move in vector-width runs and the only scattered access left is one cache line per
/// row of a tile. It is pure data movement, so the result is identical to the last bit.
inline void t8x8(const float* src, size_t sstride, float* dst, size_t dstride) {
    __m256 r0 = _mm256_loadu_ps(src + 0 * sstride), r1 = _mm256_loadu_ps(src + 1 * sstride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * sstride), r3 = _mm256_loadu_ps(src + 3 * sstride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * sstride), r5 = _mm256_loadu_ps(src + 5 * sstride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * sstride), r7 = _mm256_loadu_ps(src + 7 * sstride);
    const __m256 u0 = _mm256_unpacklo_ps(r0, r1), u1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 u2 = _mm256_unpacklo_ps(r2, r3), u3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 u4 = _mm256_unpacklo_ps(r4, r5), u5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 u6 = _mm256_unpacklo_ps(r6, r7), u7 = _mm256_unpackhi_ps(r6, r7);
    const __m256 s0 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s1 = _mm256_shuffle_ps(u0, u2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s2 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s3 = _mm256_shuffle_ps(u1, u3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s4 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s5 = _mm256_shuffle_ps(u4, u6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 s6 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 s7 = _mm256_shuffle_ps(u5, u7, _MM_SHUFFLE(3, 2, 3, 2));
    _mm256_storeu_ps(dst + 0 * dstride, _mm256_permute2f128_ps(s0, s4, 0x20));
    _mm256_storeu_ps(dst + 1 * dstride, _mm256_permute2f128_ps(s1, s5, 0x20));
    _mm256_storeu_ps(dst + 2 * dstride, _mm256_permute2f128_ps(s2, s6, 0x20));
    _mm256_storeu_ps(dst + 3 * dstride, _mm256_permute2f128_ps(s3, s7, 0x20));
    _mm256_storeu_ps(dst + 4 * dstride, _mm256_permute2f128_ps(s0, s4, 0x31));
    _mm256_storeu_ps(dst + 5 * dstride, _mm256_permute2f128_ps(s1, s5, 0x31));
    _mm256_storeu_ps(dst + 6 * dstride, _mm256_permute2f128_ps(s2, s6, 0x31));
    _mm256_storeu_ps(dst + 7 * dstride, _mm256_permute2f128_ps(s3, s7, 0x31));
}

/// dst[t * width + n] = src[n * T + t] for n in [0, width), t in [0, T): one thread's slice of
/// the N axis, 8x8 tiles where both sides are whole tiles and scalar at the edges.
void transpose_range(const float* src, size_t T, size_t width, size_t n0, size_t n1, float* dst) {
    constexpr size_t B = 64;                       // cache block, a whole number of 8x8 tiles
    for (size_t nb = n0; nb < n1; nb += B)
        for (size_t tb = 0; tb < T; tb += B) {
            const size_t ne = std::min(nb + B, n1), te = std::min(tb + B, T);
            size_t n = nb;
            for (; n + 8 <= ne; n += 8) {
                size_t t = tb;
                for (; t + 8 <= te; t += 8) t8x8(src + n * T + t, T, dst + t * width + n, width);
                for (; t < te; ++t)
                    for (size_t i = 0; i < 8; ++i) dst[t * width + n + i] = src[(n + i) * T + t];
            }
            for (; n < ne; ++n)
                for (size_t t = tb; t < te; ++t) dst[t * width + n] = src[n * T + t];
        }
}

}  // namespace

void transpose(const float* y, size_t N, size_t T, float* out) {
    const long long nthr = omp_get_max_threads();
    const size_t chunk = (N / 8 + nthr - 1) / nthr * 8;   // whole tiles per thread
#pragma omp parallel for
    for (long long i = 0; i < nthr; ++i) {
        const size_t a = std::min(static_cast<size_t>(i) * chunk, N);
        transpose_range(y, T, N, a, std::min(a + chunk, N), out);
    }
}

void transpose_parts(const float* y, size_t T, const TransposePart* parts, size_t n_parts) {
    // One parallel region for every part, not one each: a full-attention layer asks for four
    // ranges and four regions is four thread wake-ups, which under OMP_WAIT_POLICY=PASSIVE is
    // four times the wake-up and four chances for the NPU to see the cores come up.
    // (MSVC's OpenMP ignores `collapse`, so the two axes are flattened by hand.)
    const long long nthr = omp_get_max_threads();
#pragma omp parallel for
    for (long long j = 0; j < static_cast<long long>(n_parts) * nthr; ++j) {
        const TransposePart& q = parts[j / nthr];
        const long long i = j % nthr;
        const size_t chunk = (q.width / 8 + nthr - 1) / nthr * 8;
        const size_t a = std::min(static_cast<size_t>(i) * chunk, q.width);
        transpose_range(y + q.off * T, T, q.width, a, std::min(a + chunk, q.width), q.dst);
    }
}

void split_cols(const float* y, size_t T, size_t N, const TransposePart* parts, size_t n_parts) {
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t)
        for (size_t p = 0; p < n_parts; ++p)
            std::memcpy(parts[p].dst + t * parts[p].width, y + t * N + parts[p].off, parts[p].width * 4);
}

void tile_x(const float* x, size_t T, size_t K, uint16_t* out) {
    // [T,K] fp32 -> bf16, pre-tiled [K,T] in "k,n" order: K_TILE 64 x tile_n 32 tiles, each
    // tile in (8 x 8) MAC sub-tiles -- the layout gemm_q4_prefill.py streams its activation in
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (K % TK || T % TN) throw std::runtime_error("open_qwen36: tile_x: K or T does not tile by (64, 32)");
    const size_t NB = T / TN;
#pragma omp parallel for
    for (long long kb = 0; kb < static_cast<long long>(K / TK); ++kb)
        for (size_t nb = 0; nb < NB; ++nb) {
            uint16_t* w = out + (kb * NB + nb) * TK * TN;
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t)
                            *w++ = f32_to_bf16(x[(nb * TN + ti * MAC + t) * K + kb * TK + si * MAC + s]);
        }
}

// 16-lane vector helpers for the DeltaNet (GCC vector types: one body for the AVX-512 and AVX2
// clones). dn_exp is the Cephes expf polynomial over [-87, 88].
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
typedef float dnv __attribute__((vector_size(64)));
typedef int32_t dni __attribute__((vector_size(64)));
typedef float dnu __attribute__((vector_size(64), aligned(4)));
__attribute__((always_inline)) inline dnv dn_ld(const float* p) { return *reinterpret_cast<const dnu*>(p); }
__attribute__((always_inline)) inline void dn_st(float* p, dnv v) { *reinterpret_cast<dnu*>(p) = v; }
__attribute__((always_inline)) inline dnv dn_bc(float c) { return dnv{} + c; }
__attribute__((always_inline)) inline dnv dn_exp(dnv x) {
    x = x < dn_bc(-87.f) ? dn_bc(-87.f) : x;
    x = x > dn_bc(88.f) ? dn_bc(88.f) : x;
    const dnv t = x * 1.44269504088896341f + 12582912.f;
    const dnv n = t - 12582912.f;
    const dni ni = reinterpret_cast<dni>(t) - 0x4B400000;
    dnv r = x - n * 0.693359375f;
    r = r - n * -2.12194440e-4f;
    dnv p = dn_bc(1.9875691500e-4f);
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    const dnv y = p * r * r + r + 1.f;
    return reinterpret_cast<dnv>(reinterpret_cast<dni>(y) + (ni << 23));
}
__attribute__((always_inline)) inline dnv dn_silu(dnv x) { return x / (dn_bc(1.f) + dn_exp(-x)); }

// row t, head h of z (128 values): the f32 argument, or g.z16's bf16 row converted into buf
inline const float* dn_zrow(const DeltaGeom& g, const float* z, size_t vw, size_t t, size_t h, float* buf) {
    if (!g.z16) return z + t * vw + h * 128;
    const uint16_t* zs = g.z16 + t * g.ldz + h * 128;
    for (size_t j = 0; j < 128; ++j) buf[j] = bf16_to_f32(zs[j]);
    return buf;
}

// The gated delta rule in chunks of (up to) 4 tokens, head_dim 128 -- the same recurrence
// (S_t = a_t S_{t-1} + k_t d_t^T, d_t = b_t (v_t - a_t S_{t-1}^T k_t), o_t = S_t^T q_t) in its
// exact chunked form: with G_t = a_0..a_t and r_ts = a_{s+1}..a_t,
//   d_t = b_t (v_t - G_t S0^T k_t - sum_{s<t} r_ts (k_t.k_s) d_s)
//   o_t = G_t S0^T q_t + sum_{s<=t} r_ts (q_t.k_s) d_s
//   S   = G_last S0 + sum_s r_{last,s} k_s d_s^T
// One pass over S per chunk: the update of chunk n and the S^T k / S^T q of chunk n + 1 are
// computed row by row together, so S is read and written once per 4 tokens instead of read
// twice and written once per token (the recurrent form was bound by that traffic, ~7K cycles a
// token-head for ~32K FMAs). Columns go in quarters of 32 so the 16 + 8 accumulators fit the
// 32 vector registers. OFLM_OPEN_DN_RECURRENT=1 runs the recurrent form instead.
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void delta_rule_chunk4(const DeltaGeom& g, size_t R, size_t grp, float inv_sqrt, const float* Qp, const float* Kp,
                       const float* Vp, const float* decayp, const float* betap, const float* z, const float* nw, float* S,
                       float* og) {
    constexpr size_t D = 128, CN = 4;
    const size_t key_w = g.key_heads * D, vw = g.value_heads * D, nv = g.value_heads;
    const size_t nchunks = (R + CN - 1) / CN;
#pragma omp parallel for schedule(static)
    for (long long hh = 0; hh < static_cast<long long>(nv); ++hh) {
        const size_t h = static_cast<size_t>(hh), kh = h / grp;
        float* Sh = S + h * g.s_rows * D;
        // g.s_in (one chunk only): the state read from there, the update written to S
        const float* Shi = (g.s_in ? g.s_in : S) + h * g.s_rows * D;
        alignas(64) float P[CN][D], Qs[CN][D], dl[CN][D], o[D], on[D];
        // S0^T k and S0^T q for chunk 0 (a plain pass)
        auto first_pass = [&](size_t t0, size_t nt) {
            for (size_t u = 0; u < CN; ++u)
                for (size_t j = 0; j < D; ++j) P[u][j] = Qs[u][j] = 0.f;
            for (size_t i = 0; i < D; ++i) {
                const float* Si = Shi + i * D;
                for (size_t u = 0; u < nt; ++u) {
                    const float ki = Kp[(t0 + u) * key_w + kh * D + i], qi = Qp[(t0 + u) * key_w + kh * D + i];
                    for (size_t j = 0; j < D; j += 16) {
                        dn_st(&P[u][j], dn_ld(&P[u][j]) + dn_bc(ki) * dn_ld(Si + j));
                        dn_st(&Qs[u][j], dn_ld(&Qs[u][j]) + dn_bc(qi) * dn_ld(Si + j));
                    }
                }
            }
        };
        first_pass(0, std::min(CN, R));
        for (size_t c = 0; c < nchunks; ++c) {
            const size_t t0 = c * CN, nt = std::min(CN, R - t0);
            const float* kt[CN];
            const float* qt[CN];
            float a[CN], b[CN], G[CN];
            for (size_t u = 0; u < nt; ++u) {
                kt[u] = Kp + (t0 + u) * key_w + kh * D;
                qt[u] = Qp + (t0 + u) * key_w + kh * D;
                a[u] = decayp[(t0 + u) * nv + h];
                b[u] = betap[(t0 + u) * nv + h];
                G[u] = (u ? G[u - 1] : 1.f) * a[u];
            }
            // r[t][s] = a_{s+1} .. a_t (s <= t)
            float r[CN][CN], KK[CN][CN], QK[CN][CN];
            for (size_t t = 0; t < nt; ++t) {
                r[t][t] = 1.f;
                for (size_t s2 = t; s2-- > 0;) r[t][s2] = r[t][s2 + 1] * a[s2 + 1];
                for (size_t s2 = 0; s2 <= t; ++s2) {
                    dnv kk{}, qk{};
                    for (size_t i = 0; i < D; i += 16) {
                        const dnv ks = dn_ld(kt[s2] + i);
                        kk += dn_ld(kt[t] + i) * ks;
                        qk += dn_ld(qt[t] + i) * ks;
                    }
                    float a1 = 0.f, a2 = 0.f;
                    for (int e = 0; e < 16; ++e) { a1 += kk[e]; a2 += qk[e]; }
                    KK[t][s2] = a1;
                    QK[t][s2] = a2;
                }
            }
            // d_t, then o_t
            for (size_t t = 0; t < nt; ++t) {
                const float* v = Vp + (t0 + t) * vw + h * D;
                for (size_t j = 0; j < D; j += 16) {
                    dnv x = dn_ld(v + j) - dn_bc(G[t]) * dn_ld(&P[t][j]);
                    for (size_t s2 = 0; s2 < t; ++s2) x = x - dn_bc(r[t][s2] * KK[t][s2]) * dn_ld(&dl[s2][j]);
                    dn_st(&dl[t][j], dn_bc(b[t]) * x);
                }
                if (g.sv_d) std::memcpy(g.sv_d + (t0 + t) * vw + h * D, dl[t], D * 4);
            }
            for (size_t t = 0; t < nt; ++t) {
                for (size_t j = 0; j < D; j += 16) {
                    dnv x = dn_bc(G[t]) * dn_ld(&Qs[t][j]);
                    for (size_t s2 = 0; s2 <= t; ++s2) x = x + dn_bc(r[t][s2] * QK[t][s2]) * dn_ld(&dl[s2][j]);
                    dn_st(o + j, x * dn_bc(inv_sqrt));
                }
                rms_vec(o, D, nw, g.eps, on);
                alignas(64) float ga[D];
                float* out = g.a2 ? ga : og + (t0 + t) * vw + h * D;
                alignas(64) float zb[D];
                const float* zz = dn_zrow(g, z, vw, t0 + t, h, zb);
                for (size_t j = 0; j < D; j += 16) dn_st(out + j, dn_ld(on + j) * dn_silu(dn_ld(zz + j)));
                if (g.a2)
                    for (size_t kb = 0; kb < D / 8; ++kb)
                        bfp_encode8_avx2(ga + kb * 8, g.a2 + dit2_a_block_off(g.a2_row0 + t0 + t, h * D + kb * 8, vw, g.a2_mt, g.a2_hs));
            }
            // S <- G_last S + sum_s r[last][s] k_s d_s^T, fused with the next chunk's S^T k, S^T q
            const size_t last = nt - 1;
            float cs[CN];
            for (size_t s2 = 0; s2 < nt; ++s2) cs[s2] = r[last][s2];
            const dnv gl = dn_bc(G[last]);
            const size_t n0 = t0 + CN, nn = n0 < R ? std::min(CN, R - n0) : 0;
            const float* kn[CN];
            const float* qn[CN];
            for (size_t u = 0; u < nn; ++u) {
                kn[u] = Kp + (n0 + u) * key_w + kh * D;
                qn[u] = Qp + (n0 + u) * key_w + kh * D;
            }
            if (nt == CN && nn == CN) {
                // the bulk: every count a constant, so the 16 accumulators and 8 deltas stay in registers
                const float *k0 = kt[0], *k1 = kt[1], *k2 = kt[2], *k3 = kt[3];
                const float *nk0 = kn[0], *nk1 = kn[1], *nk2 = kn[2], *nk3 = kn[3];
                const float *nq0 = qn[0], *nq1 = qn[1], *nq2 = qn[2], *nq3 = qn[3];
                for (size_t jq = 0; jq < D; jq += 32) {
                    const dnv e00 = dn_ld(&dl[0][jq]), e01 = dn_ld(&dl[0][jq + 16]), e10 = dn_ld(&dl[1][jq]),
                              e11 = dn_ld(&dl[1][jq + 16]), e20 = dn_ld(&dl[2][jq]), e21 = dn_ld(&dl[2][jq + 16]),
                              e30 = dn_ld(&dl[3][jq]), e31 = dn_ld(&dl[3][jq + 16]);
                    dnv pa0{}, pa1{}, pb0{}, pb1{}, pc0{}, pc1{}, pd0{}, pd1{};
                    dnv qa0{}, qa1{}, qb0{}, qb1{}, qc0{}, qc1{}, qd0{}, qd1{};
                    for (size_t i = 0; i < D; ++i) {
                        float* Si = Sh + i * D + jq;
                        const dnv w0 = dn_bc(cs[0] * k0[i]), w1 = dn_bc(cs[1] * k1[i]), w2 = dn_bc(cs[2] * k2[i]),
                                  w3 = dn_bc(cs[3] * k3[i]);
                        const dnv s0 = gl * dn_ld(Si) + w0 * e00 + w1 * e10 + w2 * e20 + w3 * e30;
                        const dnv s1 = gl * dn_ld(Si + 16) + w0 * e01 + w1 * e11 + w2 * e21 + w3 * e31;
                        dn_st(Si, s0);
                        dn_st(Si + 16, s1);
                        const dnv u0 = dn_bc(nk0[i]), u1 = dn_bc(nk1[i]), u2 = dn_bc(nk2[i]), u3 = dn_bc(nk3[i]);
                        pa0 += u0 * s0; pa1 += u0 * s1; pb0 += u1 * s0; pb1 += u1 * s1;
                        pc0 += u2 * s0; pc1 += u2 * s1; pd0 += u3 * s0; pd1 += u3 * s1;
                        const dnv v0 = dn_bc(nq0[i]), v1 = dn_bc(nq1[i]), v2 = dn_bc(nq2[i]), v3 = dn_bc(nq3[i]);
                        qa0 += v0 * s0; qa1 += v0 * s1; qb0 += v1 * s0; qb1 += v1 * s1;
                        qc0 += v2 * s0; qc1 += v2 * s1; qd0 += v3 * s0; qd1 += v3 * s1;
                    }
                    dn_st(&P[0][jq], pa0); dn_st(&P[0][jq + 16], pa1); dn_st(&P[1][jq], pb0); dn_st(&P[1][jq + 16], pb1);
                    dn_st(&P[2][jq], pc0); dn_st(&P[2][jq + 16], pc1); dn_st(&P[3][jq], pd0); dn_st(&P[3][jq + 16], pd1);
                    dn_st(&Qs[0][jq], qa0); dn_st(&Qs[0][jq + 16], qa1); dn_st(&Qs[1][jq], qb0); dn_st(&Qs[1][jq + 16], qb1);
                    dn_st(&Qs[2][jq], qc0); dn_st(&Qs[2][jq + 16], qc1); dn_st(&Qs[3][jq], qd0); dn_st(&Qs[3][jq + 16], qd1);
                }
                continue;
            }
            for (size_t jq = 0; jq < D; jq += 32) {
                dnv p0[CN], p1[CN], q0[CN], q1[CN];
                for (size_t u = 0; u < CN; ++u) p0[u] = p1[u] = q0[u] = q1[u] = dnv{};
                dnv e0[CN], e1[CN];
                for (size_t s2 = 0; s2 < CN; ++s2) {
                    e0[s2] = s2 < nt ? dn_ld(&dl[s2][jq]) : dnv{};
                    e1[s2] = s2 < nt ? dn_ld(&dl[s2][jq + 16]) : dnv{};
                }
                for (size_t i = 0; i < D; ++i) {
                    float* Si = Sh + i * D + jq;
                    const float* Sr = Shi + i * D + jq;
                    dnv s0 = gl * dn_ld(Sr), s1 = gl * dn_ld(Sr + 16);
                    for (size_t s2 = 0; s2 < nt; ++s2) {
                        const dnv w = dn_bc(cs[s2] * kt[s2][i]);
                        s0 = s0 + w * e0[s2];
                        s1 = s1 + w * e1[s2];
                    }
                    dn_st(Si, s0);
                    dn_st(Si + 16, s1);
                    for (size_t u = 0; u < nn; ++u) {
                        const dnv ku = dn_bc(kn[u][i]), qu = dn_bc(qn[u][i]);
                        p0[u] = p0[u] + ku * s0; p1[u] = p1[u] + ku * s1;
                        q0[u] = q0[u] + qu * s0; q1[u] = q1[u] + qu * s1;
                    }
                }
                for (size_t u = 0; u < nn; ++u) {
                    dn_st(&P[u][jq], p0[u]); dn_st(&P[u][jq + 16], p1[u]);
                    dn_st(&Qs[u][jq], q0[u]); dn_st(&Qs[u][jq + 16], q1[u]);
                }
            }
        }
    }
}

__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void delta_rollback(const DeltaGeom& g, size_t m, const float* S0, const float* kp, const float* dp, const float* ap, float* S) {
    constexpr size_t D = 128, CN = 4;
    if (g.head_dim != D || m < 1 || m >= CN) throw std::runtime_error("open_qwen36: delta_rollback: head_dim 128, 1..3 rows");
    const size_t key_w = g.key_heads * D, vw = g.value_heads * D, nv = g.value_heads, grp = nv / g.key_heads;
#pragma omp parallel for schedule(static)
    for (long long hh = 0; hh < static_cast<long long>(nv); ++hh) {
        const size_t h = static_cast<size_t>(hh), kh = h / grp;
        float a[CN], G[CN], r[CN][CN];
        for (size_t u = 0; u < m; ++u) {
            a[u] = ap[u * nv + h];
            G[u] = (u ? G[u - 1] : 1.f) * a[u];
        }
        for (size_t t = 0; t < m; ++t) {
            r[t][t] = 1.f;
            for (size_t s2 = t; s2-- > 0;) r[t][s2] = r[t][s2 + 1] * a[s2 + 1];
        }
        const size_t last = m - 1;
        float cs[CN];
        for (size_t s2 = 0; s2 < m; ++s2) cs[s2] = r[last][s2];
        const dnv gl = dn_bc(G[last]);
        const float* kt[CN];
        for (size_t s2 = 0; s2 < m; ++s2) kt[s2] = kp + s2 * key_w + kh * D;
        const float* Shi = S0 + h * g.s_rows * D;
        float* Sh = S + h * g.s_rows * D;
        for (size_t jq = 0; jq < D; jq += 32) {
            dnv e0[CN], e1[CN];
            for (size_t s2 = 0; s2 < CN; ++s2) {
                e0[s2] = s2 < m ? dn_ld(dp + s2 * vw + h * D + jq) : dnv{};
                e1[s2] = s2 < m ? dn_ld(dp + s2 * vw + h * D + jq + 16) : dnv{};
            }
            for (size_t i = 0; i < D; ++i) {
                float* Si = Sh + i * D + jq;
                const float* Sr = Shi + i * D + jq;
                dnv s0 = gl * dn_ld(Sr), s1 = gl * dn_ld(Sr + 16);
                for (size_t s2 = 0; s2 < m; ++s2) {
                    const dnv w = dn_bc(cs[s2] * kt[s2][i]);
                    s0 = s0 + w * e0[s2];
                    s1 = s1 + w * e1[s2];
                }
                dn_st(Si, s0);
                dn_st(Si + 16, s1);
            }
        }
    }
}

// The gated delta rule over every token of the block, one head per work item (phase 2 of
// deltanet_block). For head_dim 128 the per-token work is register-blocked: tv = S^T k and
// o = S'^T q live in 8 sixteen-lane registers each for the whole pass over S's 128 rows
// (they were arrays the compiler reloaded and stored per row: ~20K cycles a token-head for
// ~32K FMAs), and silu is the vector polynomial. Same per-element operation order as the
// scalar form (sums over i ascending, mul then add), so only silu's last bits differ.
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void delta_rule_heads(const DeltaGeom& g, size_t R, size_t grp, float inv_sqrt, const float* Qp, const float* Kp,
                      const float* Vp, const float* decayp, const float* betap, const float* z, const float* nw, float* S,
                      float* og) {
    const size_t dim = g.head_dim, key_w = g.key_heads * dim, vw = g.value_heads * dim;
    if (g.s_in && dim != 128) throw std::runtime_error("open_qwen36: DeltaGeom.s_in needs head_dim 128");
    if (g.a2 && (dim != 128 || std::getenv("OFLM_OPEN_DN_RECURRENT"))) throw std::runtime_error("open_qwen36: DeltaGeom.a2 needs the chunk4 rule");
    if (dim != 128) {
#pragma omp parallel for
        for (long long h = 0; h < static_cast<long long>(g.value_heads); ++h) {
            std::vector<float> tv(dim), delta(dim), o(dim), on(dim);
            float* Sh = S + h * g.s_rows * dim;
            for (size_t t = 0; t < R; ++t) {
                const float* kk = Kp + t * key_w + (h / grp) * dim;
                const float* qq = Qp + t * key_w + (h / grp) * dim;
                const float* v = Vp + t * vw + h * dim;
                const float dc = decayp[t * g.value_heads + h], bt = betap[t * g.value_heads + h];
                std::fill(tv.begin(), tv.end(), 0.f);
                for (size_t i = 0; i < dim; ++i) {
                    const float* Si = Sh + i * dim;
                    const float ki = kk[i];
                    for (size_t j = 0; j < dim; ++j) tv[j] += ki * (Si[j] * dc);
                }
                for (size_t j = 0; j < dim; ++j) delta[j] = bt * (v[j] - tv[j]);
                std::fill(o.begin(), o.end(), 0.f);
                for (size_t i = 0; i < dim; ++i) {
                    float* Si = Sh + i * dim;
                    const float ki = kk[i], qi = qq[i];
                    for (size_t j = 0; j < dim; ++j) {
                        const float s = Si[j] * dc + ki * delta[j];
                        Si[j] = s;
                        o[j] += s * qi;
                    }
                }
                for (size_t j = 0; j < dim; ++j) o[j] *= inv_sqrt;
                rms_vec(o.data(), dim, nw, g.eps, on.data());
                float* out = og + t * vw + h * dim;
                alignas(64) float zb[128];
                const float* zz = dn_zrow(g, z, vw, t, h, zb);
                for (size_t j = 0; j < dim; ++j) out[j] = on[j] * silu(zz[j]);
            }
        }
        return;
    }
    if (g.s_in && (R > 4 || std::getenv("OFLM_OPEN_DN_RECURRENT"))) throw std::runtime_error("open_qwen36: DeltaGeom.s_in needs one chunk (<= 4 tokens) on the chunk4 path");
    if (std::getenv("OFLM_OPEN_DN_RECURRENT") == nullptr) {
        delta_rule_chunk4(g, R, grp, inv_sqrt, Qp, Kp, Vp, decayp, betap, z, nw, S, og);
        return;
    }
#pragma omp parallel for
    for (long long h = 0; h < static_cast<long long>(g.value_heads); ++h) {
        float* Sh = S + h * g.s_rows * dim;
        alignas(64) float o[128], on[128];
        for (size_t t = 0; t < R; ++t) {
            const float* kk = Kp + t * key_w + (h / grp) * dim;
            const float* qq = Qp + t * key_w + (h / grp) * dim;
            const float* v = Vp + t * vw + h * dim;
            const float dc = decayp[t * g.value_heads + h], bt = betap[t * g.value_heads + h];
            const dnv vdc = dn_bc(dc);
            dnv t0{}, t1{}, t2{}, t3{}, t4{}, t5{}, t6{}, t7{};
            for (size_t i = 0; i < 128; ++i) {
                const float* Si = Sh + i * 128;
                const dnv ki = dn_bc(kk[i]);
                t0 += ki * (dn_ld(Si) * vdc);       t1 += ki * (dn_ld(Si + 16) * vdc);
                t2 += ki * (dn_ld(Si + 32) * vdc);  t3 += ki * (dn_ld(Si + 48) * vdc);
                t4 += ki * (dn_ld(Si + 64) * vdc);  t5 += ki * (dn_ld(Si + 80) * vdc);
                t6 += ki * (dn_ld(Si + 96) * vdc);  t7 += ki * (dn_ld(Si + 112) * vdc);
            }
            const dnv vbt = dn_bc(bt);
            const dnv d0 = vbt * (dn_ld(v) - t0), d1 = vbt * (dn_ld(v + 16) - t1), d2 = vbt * (dn_ld(v + 32) - t2),
                      d3 = vbt * (dn_ld(v + 48) - t3), d4 = vbt * (dn_ld(v + 64) - t4), d5 = vbt * (dn_ld(v + 80) - t5),
                      d6 = vbt * (dn_ld(v + 96) - t6), d7 = vbt * (dn_ld(v + 112) - t7);
            dnv o0{}, o1{}, o2{}, o3{}, o4{}, o5{}, o6{}, o7{};
            for (size_t i = 0; i < 128; ++i) {
                float* Si = Sh + i * 128;
                const dnv ki = dn_bc(kk[i]), qi = dn_bc(qq[i]);
#define DN_ROW(J, D, O) { const dnv s_ = dn_ld(Si + J) * vdc + ki * D; dn_st(Si + J, s_); O += s_ * qi; }
                DN_ROW(0, d0, o0) DN_ROW(16, d1, o1) DN_ROW(32, d2, o2) DN_ROW(48, d3, o3)
                DN_ROW(64, d4, o4) DN_ROW(80, d5, o5) DN_ROW(96, d6, o6) DN_ROW(112, d7, o7)
#undef DN_ROW
            }
            const dnv is = dn_bc(inv_sqrt);
            dn_st(o, o0 * is); dn_st(o + 16, o1 * is); dn_st(o + 32, o2 * is); dn_st(o + 48, o3 * is);
            dn_st(o + 64, o4 * is); dn_st(o + 80, o5 * is); dn_st(o + 96, o6 * is); dn_st(o + 112, o7 * is);
            rms_vec(o, dim, nw, g.eps, on);
            float* out = og + t * vw + h * dim;
            alignas(64) float zb[128];
            const float* zz = dn_zrow(g, z, vw, t, h, zb);
            for (size_t j = 0; j < 128; j += 16) dn_st(out + j, dn_ld(on + j) * dn_silu(dn_ld(zz + j)));
        }
    }
}

// deltanet_block's phases (us, summed): setup, phase 1 (conv, silu, q / k norm), phase 1b
// (alpha / beta), window; read by the prompt pipe's OFLM_OPEN_MID_PROF
int g_attn_flash = -1;
double g_dn_prof[6] = {0, 0, 0, 0, 0, 0};
double g_dn_p1[4] = {0, 0, 0, 0};   // phase 1 on thread 0 (us): taps, silu, q / k norm, V copy
#define DN_MARK(i) do { auto _n = std::chrono::steady_clock::now(); g_dn_prof[i] += std::chrono::duration<double, std::micro>(_n - dn_t).count(); dn_t = _n; } while (0)
// alpha / beta for a prompt block: decay = exp(A softplus(xn Wa + dtb)), beta = sigmoid(xn Wb), 16
// tokens x 16 lanes of accumulators in zmm (FMA) a pass. OFLM_OPEN_AB_AVX2=1: the AVX2 path (mul
// then add, the earlier rounding).
__attribute__((target("avx512f"))) void dn_alpha_beta_512(size_t hid, size_t lanes, size_t nv, const float* xn,
                                                          const float* Wa, const float* Wb, const float* A,
                                                          const float* dtb, size_t R, float* decay, float* beta) {
    constexpr size_t TB = 16;
    const size_t nlc = (lanes + 15) / 16;
#pragma omp parallel for schedule(dynamic, 1) collapse(2)
    for (long long tb = 0; tb < static_cast<long long>((R + TB - 1) / TB); ++tb)
        for (long long lc = 0; lc < static_cast<long long>(nlc); ++lc) {
            const size_t t0 = static_cast<size_t>(tb) * TB, nt = std::min(TB, R - t0), h0 = static_cast<size_t>(lc) * 16;
            const __mmask16 m = h0 + 16 <= lanes ? __mmask16(0xFFFF) : __mmask16((1u << (lanes - h0)) - 1);
            alignas(64) float al[TB][16], be[TB][16];
            for (int pass = 0; pass < 2; ++pass) {
                const float* W = pass ? Wb : Wa;
                __m512 acc[TB];
                for (size_t u = 0; u < TB; ++u) acc[u] = _mm512_setzero_ps();
                for (size_t i = 0; i < hid; ++i) {
                    const __m512 w = _mm512_maskz_loadu_ps(m, W + i * lanes + h0);
                    for (size_t u = 0; u < TB; ++u)
                        acc[u] = _mm512_fmadd_ps(_mm512_set1_ps(u < nt ? xn[(t0 + u) * hid + i] : 0.f), w, acc[u]);
                }
                for (size_t u = 0; u < TB; ++u) _mm512_store_ps(pass ? be[u] : al[u], acc[u]);
            }
            for (size_t u = 0; u < nt; ++u)
                for (size_t h = h0; h < std::min(h0 + 16, nv); ++h) {
                    decay[(t0 + u) * nv + h] = std::exp(A[h] * softplus(al[u][h - h0] + dtb[h]));
                    beta[(t0 + u) * nv + h] = sigmoid(be[u][h - h0]);
                }
        }
}
namespace {
inline uint16_t dnc_bf(float x) { return f32_to_bf16(x); }
inline void dnc_hilo(double x, uint16_t* h, uint16_t* l) {
    *h = f32_to_bf16(static_cast<float>(x));
    *l = f32_to_bf16(static_cast<float>(x - static_cast<double>(bf16_to_f32(*h))));
}
}  // namespace

// 16 floats -> 16 bf16, rounded to nearest even as f32_to_bf16
__attribute__((target("avx512f"))) static void dnc_cvt16_512(const float* x, uint16_t* d) {
    const __m512i u = _mm512_castps_si512(_mm512_loadu_ps(x));
    const __m512i r = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF),
                                                           _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1))));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(d), _mm512_cvtepi32_epi16(_mm512_srli_epi32(r, 16)));
}
static inline void dnc_cvt16(const float* x, uint16_t* d) {
    static const bool avx512 = __builtin_cpu_supports("avx512f");
    if (avx512) dnc_cvt16_512(x, d);
    else for (size_t j = 0; j < 16; ++j) d[j] = dnc_bf(x[j]);
}

// One 10240 B block a key head and 16-token step: [aux of value head 2 kh | K tiles | Q tiles |
// aux of 2 kh + 1] (dnc.py reads the even head's record from byte 0, the odd one's from 1024).
void dnc_pack(const DeltaGeom& g, const DeltaP1& p, size_t steps, uint8_t* rec, uint16_t* vo) {
    const size_t D = 128, nv = g.value_heads, nk = g.key_heads, grp = nv / nk, key_w = nk * D, vw = nv * D, R = g.t_real;
    const bool fused = g.dnc_rec != nullptr;   // phase 1 wrote the real tokens' tiles and v
    if (g.head_dim != D || R > steps * 16 || grp != 2)
        throw std::runtime_error("open_qwen36: dnc_pack: head dim 128, two value heads a key head, t_real <= steps * 16");
#pragma omp parallel for schedule(dynamic, 4)
    for (long long kn = 0; kn < static_cast<long long>(nk * steps); ++kn) {
        const size_t kh = static_cast<size_t>(kn) / steps, n = static_cast<size_t>(kn) % steps, t0 = n * 16;
        uint8_t* blk = rec + (kh * steps + n) * 10240;
        // K, Q tiles [tb][kb][8 t][8 dk]: a token's 128 dk as 16 blocks of 8, two a conversion
        for (size_t which = 0; which < 2; ++which) {
            const float* src = which ? p.Q : p.K;
            uint16_t* dst = reinterpret_cast<uint16_t*>(blk + 1024) + which * 2048;
            for (size_t tb = 0; tb < 2; ++tb)
                for (size_t i = 0; i < 8; ++i) {
                    const size_t t = t0 + tb * 8 + i;
                    if (fused && t < R) continue;
                    for (size_t kb = 0; kb < 16; kb += 2) {
                        uint16_t* d0 = dst + ((tb * 16 + kb) * 8 + i) * 8;
                        uint16_t* d1 = d0 + 64;
                        alignas(32) uint16_t c[16] = {};
                        if (t < R) dnc_cvt16(src + t * key_w + kh * D + kb * 8, c);
                        std::memcpy(d0, c, 16);
                        std::memcpy(d1, c + 8, 16);
                    }
                }
        }
        for (size_t j = 0; j < grp; ++j) {
            const size_t h = kh * grp + j;
            // the decays: G_t = sum log a (in the step), e = G_t - G_7, f = G_7 - G_s, gamma = G_t, w = G_15 - G_s
            double G[16], la = 0, bt[16];
            for (size_t i = 0; i < 16; ++i) {
                const size_t t = t0 + i;
                // log a >= -10 (a >= 4.5e-5): e and f span up to 8 tokens of decay either side of the
                // middle, and must stay below e^88 (bf16 / f32); some heads decay by e^-100 a token, where
                // the state keeps 4.5e-5 instead of ~0
                la += t < R ? std::max(-10.0, std::log(static_cast<double>(p.a[t * nv + h]))) : 0.0;
                G[i] = la;
                bt[i] = t < R ? p.b[t * nv + h] : 0.0;
            }
            uint16_t* aux = reinterpret_cast<uint16_t*>(blk + (j ? 9216 : 0));
            std::memset(aux, 0, 1024);
            for (size_t i = 0; i < 16; ++i) {
                dnc_hilo(std::exp(G[i] - G[7]), aux + i, aux + 16 + i);
                dnc_hilo(std::exp(G[7] - G[i]), aux + 32 + i, aux + 48 + i);
                dnc_hilo(bt[i], aux + 64 + i, aux + 80 + i);
            }
            for (size_t tb = 0; tb < 2; ++tb)
                for (size_t i = 0; i < 8; ++i) {
                    uint16_t gh, gl;
                    dnc_hilo(std::exp(G[tb * 8 + i]), &gh, &gl);
                    const uint16_t w = dnc_bf(static_cast<float>(std::exp(G[15] - G[tb * 8 + i])));
                    for (size_t c = 0; c < 8; ++c) {
                        aux[96 + tb * 64 + i * 8 + c] = gh;          // [8 t][8]: row i
                        aux[224 + tb * 64 + i * 8 + c] = gl;
                        aux[352 + tb * 64 + c * 8 + i] = w;          // [8 dk][8 t]: column i
                    }
                }
            dnc_hilo(std::exp(G[15]), aux + 480, aux + 481);
            // v pieces [4 rows][16 t][32]
            uint16_t* v = vo + (h * steps + n) * 2048;
            for (size_t q = 0; q < 4; ++q)
                for (size_t i = 0; i < 16; ++i) {
                    const size_t t = t0 + i;
                    uint16_t* d = v + (q * 16 + i) * 32;
                    if (fused && t < R) continue;
                    if (t < R) {
                        const float* x = p.V + t * vw + h * D + q * 32;
                        dnc_cvt16(x, d);
                        dnc_cvt16(x + 16, d + 16);
                    } else {
                        std::memset(d, 0, 64);
                    }
                }
        }
    }
}

// The gated norm on dnc's o: vector code throughout (the norm's sum of squares in 16 float lanes, not
// rms_vec's serial double sum: 2.8 -> ~0.5 ms a 1024-token chunk)
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void dnc_finish(const DeltaGeom& g, const uint16_t* vo, size_t steps, const float* nw, float* og) {
    const size_t D = 128, nv = g.value_heads, vw = nv * D, R = g.t_real;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(D));
    std::fill(og + R * vw, og + g.T * vw, 0.f);
#pragma omp parallel for schedule(static)
    for (long long th = 0; th < static_cast<long long>(R * nv); ++th) {
        const size_t t = static_cast<size_t>(th) / nv, h = static_cast<size_t>(th) % nv, n = t / 16, i = t % 16;
        alignas(64) float o[128], zb[128];
        for (size_t q = 0; q < 4; ++q) {
            const uint16_t* s = vo + ((h * steps + n) * 4 + q) * 512 + i * 32;
            for (size_t j = 0; j < 32; ++j) {
                const uint32_t u = static_cast<uint32_t>(s[j]) << 16;
                float f;
                std::memcpy(&f, &u, 4);
                o[q * 32 + j] = f * inv_sqrt;
            }
        }
        dnv ss{};
        for (size_t j = 0; j < D; j += 16) { const dnv x = dn_ld(o + j); ss += x * x; }
        float sum = 0.f;
        for (int e = 0; e < 16; ++e) sum += ss[e];
        const dnv r = dn_bc(static_cast<float>(1.0 / std::sqrt(static_cast<double>(sum) / D + g.eps)));
        const float* zz = dn_zrow(g, nullptr, vw, t, h, zb);
        float* out = og + t * vw + h * D;
        for (size_t j = 0; j < D; j += 16) dn_st(out + j, dn_ld(o + j) * r * dn_ld(nw + j) * dn_silu(dn_ld(zz + j)));
    }
}

namespace {
// element (dk, dv) of head h in dnc's tile layout (floats)
inline size_t dnc_tidx(size_t h, size_t dk, size_t dv) {
    const size_t q = dv / 32, dvl = dv % 32, idx = (((dvl / 8) * 16 + dk / 8) * 8 + dvl % 8) * 8 + dk % 8;
    return ((h * 16 + idx / 256) * 4 + q) * 256 + idx % 256;
}
}  // namespace

void dnc_state_to_tiles(const DeltaGeom& g, const float* S, float* tiles) {
#pragma omp parallel for
    for (long long hh = 0; hh < static_cast<long long>(g.value_heads); ++hh) {
        const size_t h = static_cast<size_t>(hh);
        for (size_t dk = 0; dk < 128; ++dk)
            for (size_t dv = 0; dv < 128; ++dv) tiles[dnc_tidx(h, dk, dv)] = S[(h * g.s_rows + dk) * 128 + dv];
    }
}
void dnc_state_from_tiles(const DeltaGeom& g, const float* tiles, float* S) {
#pragma omp parallel for
    for (long long hh = 0; hh < static_cast<long long>(g.value_heads); ++hh) {
        const size_t h = static_cast<size_t>(hh);
        for (size_t dk = 0; dk < 128; ++dk)
            for (size_t dv = 0; dv < 128; ++dv) S[(h * g.s_rows + dk) * 128 + dv] = tiles[dnc_tidx(h, dk, dv)];
    }
}

// cloned like the rule: phase 1's silu and norms are 16-lane vector code (two AVX2 halves without it)
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void deltanet_block(const DeltaGeom& g, const float* qkv, const float* z, const float* xn, const float* convw,
                    const float* Wa, const float* Wb, const float* A, const float* dtb, const float* nw,
                    uint16_t* conv_state, float* S, float* og, double* phase_ms) {
    const auto tp0 = std::chrono::steady_clock::now();
    const size_t dim = g.head_dim, key_w = g.key_heads * dim, vw = g.value_heads * dim, nch = 2 * key_w + vw;
    auto dn_t = tp0;
    if (g.value_heads % g.key_heads || g.t_real > g.T || g.s_rows < dim || g.lanes < g.value_heads)
        throw std::runtime_error("open_qwen36: deltanet_block: inconsistent geometry");
    const size_t grp = g.value_heads / g.key_heads, R = g.t_real;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(dim));
    std::fill(og + g.t_real * vw, og + g.T * vw, 0.f);   // the real rows: every head written by phase 2

    // ---- phase 1, per token: the conv (state rows carried), q / k normalised, alpha / beta
    // The conv is a fixed taps-wide window over the bf16-rounded rows, not a recurrence, so
    // every token's work is independent: row r of token t's window is qkv row t - pre + r,
    // and the carried state stands in where that runs before the block.
    const size_t pre = g.taps - 1;
    // packed prompts: row t's sequence starts at seg0[t]; window rows before it read as zero
    // (LLONG_MIN: the sequence began before the block, the carried rows are its own)
    std::vector<long long> seg0(R, LLONG_MIN);
    if (g.reset)
        for (size_t t = 0; t < R; ++t)
            seg0[t] = g.reset[t] ? static_cast<long long>(t) : t ? seg0[t - 1] : LLONG_MIN;
    std::vector<float> carry(pre * nch);
    for (size_t r = 0; r < pre; ++r)
        for (size_t j = 0; j < nch; ++j) carry[r * nch + j] = bf16_to_f32(conv_state[r * nch + j]);
    // kept across calls (a fresh ~10 MB per 256-token block was page faults on every call)
    // (references: the OpenMP workers below must see THIS thread's arrays, not their own)
    static thread_local std::vector<float> Q_tl, K_tl, V_tl, decay_tl, beta_tl;
    std::vector<float> &Q = Q_tl, &Kk = K_tl, &V = V_tl, &decay = decay_tl, &beta = beta_tl;
    const bool dncw = g.p1 && g.dnc_rec;   // K / Q / v straight to dnc's buffers as bf16
    if (dncw && (grp != 2 || dim != 128 || R > g.dnc_steps * 16))
        throw std::runtime_error("open_qwen36: deltanet_block: dnc's layout wants two value heads a key head, head dim 128, t_real <= steps * 16");
    if (!dncw) { Q.resize(R * key_w); Kk.resize(R * key_w); V.resize(R * vw); }
    decay.resize(R * g.value_heads); beta.resize(R * g.value_heads);
    DN_MARK(0);
    // one or two tokens: on this thread (a team costs more than the work, ~125 us against 37)
#pragma omp parallel if (R > 2)
    {
        std::vector<float> c(nch), al(g.lanes), be(g.lanes);
#pragma omp for
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
            const size_t t = static_cast<size_t>(tt);
            const bool t0p = omp_get_thread_num() == 0;
            auto p1 = std::chrono::steady_clock::now();
            auto p1m = [&](int i) { if (t0p) { auto n = std::chrono::steady_clock::now(); g_dn_p1[i] += std::chrono::duration<double, std::micro>(n - p1).count(); p1 = n; } };
            // The taps as `taps` contiguous passes over the channels rather than a strided
            // inner loop of depth `taps` inside the channel loop. Each pass reads one whole
            // row of qkv and one of convw in order, and the "is this row carried state or
            // qkv" test sits outside the channel loop instead of inside it -- which is what
            // lets the vectoriser take the bf16 round trip and the multiply-add at all. The
            // accumulation order over r is unchanged and c starts at zero exactly as `acc`
            // did, so every bit of the result is the same.
            std::fill(c.begin(), c.end(), 0.f);
            for (size_t r = 0; r < g.taps; ++r) {
                const long long s = tt - static_cast<long long>(pre) + static_cast<long long>(r);
                const float* __restrict cw = convw + r * nch;
                if (s < seg0[t]) continue;
                if (s < 0) {
                    const float* __restrict v = carry.data() + (static_cast<size_t>(s) + pre) * nch;
                    for (size_t j = 0; j < nch; ++j) c[j] += cw[j] * v[j];
                } else {
                    // MSVC leaves the scalar bf16r loop unvectorized; mul then add keeps today's rounding
                    size_t j = 0;
                    if (g.qkv16) {   // the bf16 rows themselves: their f32 values, exactly what bf16r8 gives
                        const uint16_t* __restrict v = g.qkv16 + static_cast<size_t>(s) * g.ld16;
                        for (; j + 8 <= nch; j += 8) {
                            const __m256 x = _mm256_castsi256_ps(_mm256_slli_epi32(
                                _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(v + j))), 16));
                            _mm256_storeu_ps(c.data() + j, _mm256_add_ps(_mm256_loadu_ps(c.data() + j), _mm256_mul_ps(_mm256_loadu_ps(cw + j), x)));
                        }
                        for (; j < nch; ++j) c[j] += cw[j] * bf16_to_f32(v[j]);
                    } else {
                        const float* __restrict v = qkv + static_cast<size_t>(s) * nch;
                        for (; j + 8 <= nch; j += 8)
                            _mm256_storeu_ps(c.data() + j, _mm256_add_ps(_mm256_loadu_ps(c.data() + j),
                                                                         _mm256_mul_ps(_mm256_loadu_ps(cw + j), bf16r8(v + j))));
                        for (; j < nch; ++j) c[j] += cw[j] * bf16r(v[j]);
                    }
                }
            }
            p1m(0);
            {   // silu, 16 lanes at a time (nch is a multiple of 16 at every Qwen3.5 width)
                size_t j = 0;
                for (; j + 16 <= nch; j += 16) dn_st(c.data() + j, dn_silu(dn_ld(c.data() + j)));
                for (; j < nch; ++j) c[j] = silu(c[j]);
            }
            p1m(1);
            const size_t n16 = t / 16, tb = (t % 16) / 8, ti = t % 8;
            for (size_t hh = 0; hh < g.key_heads; ++hh)
                for (int which = 0; which < 2; ++which) {
                    const float* src = c.data() + which * key_w + hh * dim;
                    double ss = 0;
                    for (size_t j = 0; j < dim; ++j) ss += static_cast<double>(src[j]) * src[j];
                    const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));   // dn_glue's L2 norm
                    if (dncw) {
                        // dnc's K / Q tiles [tb][kb][8 t][8 dk] (K first) in the key head's block at 1024
                        uint16_t* dt = reinterpret_cast<uint16_t*>(g.dnc_rec + (hh * g.dnc_steps + n16) * 10240 + 1024) +
                                       (which ? 0 : 2048);
                        alignas(64) float x[16];
                        alignas(32) uint16_t cv[16];
                        for (size_t kb = 0; kb < 16; kb += 2) {
                            for (size_t j = 0; j < 16; ++j) x[j] = src[kb * 8 + j] * r;
                            dnc_cvt16(x, cv);
                            std::memcpy(dt + ((tb * 16 + kb) * 8 + ti) * 8, cv, 16);
                            std::memcpy(dt + ((tb * 16 + kb + 1) * 8 + ti) * 8, cv + 8, 16);
                        }
                        continue;
                    }
                    float* dst = (which ? Kk : Q).data() + t * key_w + hh * dim;
                    for (size_t j = 0; j < dim; ++j) dst[j] = src[j] * r;
                }
            p1m(2);
            if (dncw) {
                // v pieces [4 rows][16 t][32] of each value head's step
                for (size_t h = 0; h < g.value_heads; ++h)
                    for (size_t q = 0; q < 4; ++q) {
                        uint16_t* d = g.dnc_vo + (h * g.dnc_steps + n16) * 2048 + (q * 16 + t % 16) * 32;
                        const float* x = c.data() + 2 * key_w + h * dim + q * 32;
                        dnc_cvt16(x, d);
                        dnc_cvt16(x + 16, d + 16);
                    }
            } else {
                std::copy(c.begin() + 2 * key_w, c.end(), V.begin() + t * vw);
            }
            p1m(3);
            (void)al; (void)be;   // alpha / beta: phase 1b below, 8 tokens a pass over Wa / Wb
        }
    }
    // ---- phase 1b: alpha / beta = xn Wa, xn Wb over the block, 8 tokens a pass so each Wa / Wb
    // row is read once for 8 tokens instead of once per token (per token it was 2.6 MB of weight
    // traffic for 1.3 MFLOP at the 27B's width). Each lane still sums over i in the same order.
    DN_MARK(1);
    // A decode-sized block (R <= 8, one token block): split over the hidden dim instead, each
    // thread a slice of i for every lane, then the partial sums added -- one thread alone read
    // all of Wa / Wb (2.6 MB at the 27B's width) for 0.5 ms a layer.
    static const bool dn_ab512 = __builtin_cpu_supports("avx512f") && !std::getenv("OFLM_OPEN_AB_AVX2");
    if (R <= 8 && omp_get_max_threads() > 1) {
        const size_t lanes = g.lanes, hid = g.hid, nv = g.value_heads;
        const int nt = omp_get_max_threads();
        std::vector<float> pa(static_cast<size_t>(nt) * R * lanes, 0.f), pb(pa.size(), 0.f);
#pragma omp parallel num_threads(nt)
        {
            const size_t ti = static_cast<size_t>(omp_get_thread_num()), T = static_cast<size_t>(omp_get_num_threads());
            const size_t i0 = hid * ti / T, i1 = hid * (ti + 1) / T;
            float* a = pa.data() + ti * R * lanes;
            float* b = pb.data() + ti * R * lanes;
            for (size_t u = 0; u < R; ++u) {
                float* __restrict au = a + u * lanes;
                float* __restrict bu = b + u * lanes;
                size_t hh = 0;
                for (; hh + 8 <= lanes; hh += 8) {   // 8 lanes in registers over the whole i slice
                    __m256 sa = _mm256_setzero_ps(), sb = _mm256_setzero_ps();
                    if (g.wa16 && g.wb16) {
                        for (size_t i = i0; i < i1; ++i) {
                            const __m256 xv = _mm256_set1_ps(xn[u * hid + i]);
                            const __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(g.wa16 + i * lanes + hh));
                            const __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(g.wb16 + i * lanes + hh));
                            sa = _mm256_fmadd_ps(xv, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(a16), 16)), sa);
                            sb = _mm256_fmadd_ps(xv, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(b16), 16)), sb);
                        }
                    } else
                    for (size_t i = i0; i < i1; ++i) {
                        const __m256 xv = _mm256_set1_ps(xn[u * hid + i]);
                        sa = _mm256_fmadd_ps(xv, _mm256_loadu_ps(Wa + i * lanes + hh), sa);
                        sb = _mm256_fmadd_ps(xv, _mm256_loadu_ps(Wb + i * lanes + hh), sb);
                    }
                    _mm256_storeu_ps(au + hh, sa);
                    _mm256_storeu_ps(bu + hh, sb);
                }
                for (; hh < lanes; ++hh)
                    for (size_t i = i0; i < i1; ++i) {
                        au[hh] += xn[u * hid + i] * Wa[i * lanes + hh];
                        bu[hh] += xn[u * hid + i] * Wb[i * lanes + hh];
                    }
            }
        }
        for (size_t u = 0; u < R; ++u)
            for (size_t h = 0; h < nv; ++h) {
                float sa = 0.f, sb = 0.f;
                for (int t = 0; t < nt; ++t) {
                    sa += pa[(static_cast<size_t>(t) * R + u) * lanes + h];
                    sb += pb[(static_cast<size_t>(t) * R + u) * lanes + h];
                }
                decay[u * nv + h] = std::exp(A[h] * softplus(sa + dtb[h]));
                beta[u * nv + h] = sigmoid(sb);
            }
    } else if (dn_ab512) {
        // 16 tokens x 16 lanes a pass in zmm (FMA): each Wa / Wb row read once for 16 tokens
        dn_alpha_beta_512(g.hid, g.lanes, g.value_heads, xn, Wa, Wb, A, dtb, R, decay.data(), beta.data());
    } else {
        const size_t lanes = g.lanes, hid = g.hid, nv = g.value_heads;
        constexpr size_t TB = 8;
#pragma omp parallel for schedule(static)
        for (long long tb = 0; tb < static_cast<long long>((R + TB - 1) / TB); ++tb) {
            const size_t t0 = static_cast<size_t>(tb) * TB, nt = std::min(TB, R - t0);
            std::vector<float> al(TB * lanes, 0.f), be(TB * lanes, 0.f);
            for (size_t h0 = 0; h0 < lanes; h0 += 16) {
                const size_t w16 = std::min<size_t>(16, lanes - h0);
                if (w16 == 16) {
                    __m256 a[TB][2], b[TB][2];
                    for (size_t u = 0; u < TB; ++u) a[u][0] = a[u][1] = b[u][0] = b[u][1] = _mm256_setzero_ps();
                    for (size_t i = 0; i < hid; ++i) {
                        const __m256 wa0 = _mm256_loadu_ps(Wa + i * lanes + h0), wa1 = _mm256_loadu_ps(Wa + i * lanes + h0 + 8);
                        const __m256 wb0 = _mm256_loadu_ps(Wb + i * lanes + h0), wb1 = _mm256_loadu_ps(Wb + i * lanes + h0 + 8);
                        for (size_t u = 0; u < TB; ++u) {
                            const __m256 xb = _mm256_set1_ps(u < nt ? xn[(t0 + u) * hid + i] : 0.f);
                            a[u][0] = _mm256_add_ps(a[u][0], _mm256_mul_ps(xb, wa0));
                            a[u][1] = _mm256_add_ps(a[u][1], _mm256_mul_ps(xb, wa1));
                            b[u][0] = _mm256_add_ps(b[u][0], _mm256_mul_ps(xb, wb0));
                            b[u][1] = _mm256_add_ps(b[u][1], _mm256_mul_ps(xb, wb1));
                        }
                    }
                    for (size_t u = 0; u < TB; ++u) {
                        _mm256_storeu_ps(al.data() + u * lanes + h0, a[u][0]);
                        _mm256_storeu_ps(al.data() + u * lanes + h0 + 8, a[u][1]);
                        _mm256_storeu_ps(be.data() + u * lanes + h0, b[u][0]);
                        _mm256_storeu_ps(be.data() + u * lanes + h0 + 8, b[u][1]);
                    }
                } else {
                    for (size_t u = 0; u < nt; ++u)
                        for (size_t hh = h0; hh < h0 + w16; ++hh) {
                            float sa = 0.f, sb = 0.f;
                            for (size_t i = 0; i < hid; ++i) {
                                sa += xn[(t0 + u) * hid + i] * Wa[i * lanes + hh];
                                sb += xn[(t0 + u) * hid + i] * Wb[i * lanes + hh];
                            }
                            al[u * lanes + hh] = sa;
                            be[u * lanes + hh] = sb;
                        }
                }
            }
            for (size_t u = 0; u < nt; ++u)
                for (size_t h = 0; h < nv; ++h) {
                    decay[(t0 + u) * nv + h] = std::exp(A[h] * softplus(al[u * lanes + h] + dtb[h]));
                    beta[(t0 + u) * nv + h] = sigmoid(be[u * lanes + h]);
                }
        }
    }
    DN_MARK(2);
    if (g.sv_k) std::memcpy(g.sv_k, Kk.data(), R * key_w * 4);
    if (g.sv_a) std::memcpy(g.sv_a, decay.data(), R * g.value_heads * 4);
    // the window the next block starts from: the last `pre` rows, short blocks keeping what
    // the shift would have left in front of them
    const long long last0 = R ? seg0[R - 1] : LLONG_MIN;   // the sequence the next block continues
    for (size_t r = 0; r < pre; ++r) {
        const long long s = static_cast<long long>(R) - static_cast<long long>(pre) + static_cast<long long>(r);
        if (s < last0) {
            std::fill(conv_state + r * nch, conv_state + (r + 1) * nch, uint16_t{0});
        } else if (s < 0) {
            std::copy(conv_state + (R + r) * nch, conv_state + (R + r + 1) * nch, conv_state + r * nch);
        } else {
            if (g.qkv16) {
                std::copy(g.qkv16 + static_cast<size_t>(s) * g.ld16, g.qkv16 + static_cast<size_t>(s) * g.ld16 + nch, conv_state + r * nch);
            } else {
                const float* row = qkv + static_cast<size_t>(s) * nch;
                for (size_t j = 0; j < nch; ++j) conv_state[r * nch + j] = f32_to_bf16(row[j]);
            }
        }
    }

    DN_MARK(3);
    if (g.p1) {   // the rule runs elsewhere (dnc): hand phase 1's arrays over
        g.p1->Q = Q.data(); g.p1->K = Kk.data(); g.p1->V = V.data(); g.p1->a = decay.data(); g.p1->b = beta.data();
        if (phase_ms) phase_ms[0] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
        return;
    }
    // ---- phase 2, per head over every token: the gated delta rule on S (in place), the gated norm
    const auto tp1 = std::chrono::steady_clock::now();
    // one pass per packed sequence, S zeroed where one starts
    for (size_t a = 0; a < R;) {
        size_t b = a + 1;
        while (b < R && !(g.reset && g.reset[b])) ++b;
        if (g.reset && g.reset[a]) std::fill(S, S + g.value_heads * g.s_rows * dim, 0.f);
        DeltaGeom ga = g;
        if (g.z16) ga.z16 = g.z16 + a * g.ldz;
        ga.a2_row0 = a;
        delta_rule_heads(ga, b - a, grp, inv_sqrt, Q.data() + a * key_w, Kk.data() + a * key_w, V.data() + a * vw,
                         decay.data() + a * g.value_heads, beta.data() + a * g.value_heads, g.z16 ? nullptr : z + a * vw, nw, S,
                         og + a * vw);
        a = b;
    }
    if (phase_ms) {
        using ms = std::chrono::duration<double, std::milli>;
        const auto tp2 = std::chrono::steady_clock::now();
        phase_ms[0] += ms(tp1 - tp0).count();
        phase_ms[1] += ms(tp2 - tp1).count();
    }
}

namespace {
// The tiled attention behind attention_block (OFLM: LFM2-MoE's prompt route spends most of
// its host time here). Loops are laid out so the innermost one runs over contiguous floats
// with no reduction, which vectorises without -ffast-math: S over the chunk's key rows (K
// transposed into the chunk scratch), acc over the head dims. Two clones picked at load by the
// CPU: x86-64-v4 (AVX-512; Zen 5 Strix parts) and x86-64-v3 (AVX2 + FMA; every Ryzen AI part).
constexpr size_t kQT = 16;     // query tokens per work item
constexpr size_t kKC = 64;     // key rows per chunk
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void attention_tile(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                    const uint16_t* kv, size_t kv_row_elems, float* og, size_t g, size_t t0, size_t kend) {
    const size_t grp = nh / kvh, qw = nh * hd, kvw = kvh * hd;
    const size_t nt = std::min(kQT, R - t0), M = grp * nt;
    const size_t last = kend ? kend - 1 : pos0 + t0 + nt - 1;   // the highest key row any query here sees
    std::vector<float> q(M * hd), kt(hd * kKC), vc(kKC * hd), s(M * kKC), acc(M * hd, 0.f), m(M, -1e30f), l(M, 0.f);
    std::vector<size_t> qpos(M);
    for (size_t i = 0; i < M; ++i) {                      // row i = (head gi, token ti)
        const size_t gi = i / nt, ti = i % nt, h = g * grp + gi;
        qpos[i] = kend ? kend - 1 : pos0 + t0 + ti;
        const float* src = Q + (t0 + ti) * qw + h * hd;
        for (size_t d = 0; d < hd; ++d) q[i * hd + d] = src[d] * scale;
    }
    for (size_t c0 = 0; c0 <= last; c0 += kKC) {
        const size_t kc = std::min(kKC, last + 1 - c0);
        for (size_t j = 0; j < kc; ++j) {
            const uint16_t* row = kv + (c0 + j) * kv_row_elems + g * hd;
            for (size_t d = 0; d < hd; ++d) {
                kt[d * kKC + j] = bf16_to_f32(row[d]);
                vc[j * hd + d] = bf16_to_f32(row[kvw + d]);
            }
        }
        for (size_t i = 0; i < M; ++i) {
            float* si = s.data() + i * kKC;
            for (size_t j = 0; j < kKC; ++j) si[j] = 0.f;
            const float* qi = q.data() + i * hd;
            for (size_t d = 0; d < hd; ++d) {
                const float a = qi[d];
                const float* kd = kt.data() + d * kKC;
                for (size_t j = 0; j < kKC; ++j) si[j] += a * kd[j];
            }
            // causal mask, then the online softmax update for this row
            const size_t valid = qpos[i] < c0 ? 0 : std::min(kc, qpos[i] - c0 + 1);
            if (!valid) continue;
            float mx = m[i];
            for (size_t j = 0; j < valid; ++j) mx = std::max(mx, si[j]);
            const float corr = std::exp(m[i] - mx);
            float sum = 0.f;
            for (size_t j = 0; j < valid; ++j) sum += (si[j] = std::exp(si[j] - mx));
            for (size_t j = valid; j < kKC; ++j) si[j] = 0.f;
            l[i] = l[i] * corr + sum;
            m[i] = mx;
            float* ai = acc.data() + i * hd;
            for (size_t d = 0; d < hd; ++d) ai[d] *= corr;
            for (size_t j = 0; j < valid; ++j) {
                const float p = si[j];
                const float* vj = vc.data() + j * hd;
                for (size_t d = 0; d < hd; ++d) ai[d] += p * vj[d];
            }
        }
    }
    for (size_t i = 0; i < M; ++i) {
        const size_t gi = i / nt, ti = i % nt, h = g * grp + gi;
        float* out = og + (t0 + ti) * qw + h * hd;
        const float inv = 1.f / l[i];
        for (size_t d = 0; d < hd; ++d) out[d] = acc[i * hd + d] * inv;
    }
}

// ---- the fast path (hd a multiple of 64): the window's K and V converted from bf16 ONCE per
// call, K transposed per 64-key chunk to KT[g][chunk][d][64] and V to VF[g][pos][d] (a chunk
// of either is one contiguous 16 KB block: KT[g][d][pos] put a chunk's 64 rows P floats
// apart, a power-of-two stride from 8K keys on that aliased one cache set -- 4.5x the 4K
// time at 8K, 9.8x more at 16K) (pos padded to whole 64-key
// chunks with zeros), then per work item (kv head, query tile) S = Q K^T and acc += P V as
// register-blocked micro-GEMMs over 64-key chunks (4 query rows x 64 keys / 64 dims), the
// online softmax in between with a vector exp (Cephes expf's polynomial, ~1 ulp on the range
// it sees). Masking only on the chunks that cross the diagonal. GCC vector types, so one body
// serves the AVX-512 and AVX2 clones.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpsabi"
typedef float v16f __attribute__((vector_size(64)));
typedef int32_t v16i __attribute__((vector_size(64)));
typedef float v16u __attribute__((vector_size(64), aligned(4)));   // unaligned loads / stores

__attribute__((always_inline)) inline v16f ld(const float* p) { return *reinterpret_cast<const v16u*>(p); }
__attribute__((always_inline)) inline void st(float* p, v16f v) { *reinterpret_cast<v16u*>(p) = v; }
__attribute__((always_inline)) inline v16f bc(float c) { return v16f{} + c; }

__attribute__((always_inline)) inline v16f vexp(v16f x) {           // x <= 0 here
    x = x < bc(-86.f) ? bc(-86.f) : x;
    const v16f t = x * 1.44269504088896341f + 12582912.f;            // round(x log2 e) by the 1.5*2^23 trick
    const v16f n = t - 12582912.f;
    const v16i ni = reinterpret_cast<v16i>(t) - 0x4B400000;
    v16f r = x - n * 0.693359375f;
    r = r - n * -2.12194440e-4f;
    v16f p = bc(1.9875691500e-4f);
    p = p * r + 1.3981999507e-3f;
    p = p * r + 8.3334519073e-3f;
    p = p * r + 4.1665795894e-2f;
    p = p * r + 1.6666665459e-1f;
    p = p * r + 5.0000001201e-1f;
    const v16f y = p * r * r + r + 1.f;
    return reinterpret_cast<v16f>(reinterpret_cast<v16i>(y) + (ni << 23));
}

constexpr size_t kRB = 4;      // query rows per micro-kernel block

__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void attention_tile64(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, size_t qt, float scale, const float* Q,
                      const float* KT, const float* VF, size_t P, float* og, size_t g, size_t t0, size_t kend,
                      const float* sinks = nullptr, size_t window = 0, size_t kb = 0) {
    // kb: the first key KT / VF hold (a multiple of 64; the window's start); sinks: one more logit per
    // head in each denominator; window: row q sees keys [q + 1 - window, q]
    const size_t grp = nh / kvh, qw = nh * hd;
    const size_t nt = std::min(qt, R - t0), M = grp * nt, Mp = (M + kRB - 1) / kRB * kRB;
    const size_t first = pos0 + t0, last = kend ? kend - 1 : first + nt - 1;   // kend: every query sees [0, kend)
    static thread_local std::vector<float> qb, sb, ab, mb, lb, cb;
    static thread_local std::vector<size_t> qp;
    qb.assign(Mp * hd, 0.f);
    sb.resize(Mp * 64);
    ab.assign(Mp * hd, 0.f);
    mb.assign(Mp, -1e30f);
    lb.assign(Mp, 0.f);
    cb.resize(Mp);
    qp.resize(Mp);
    float *q = qb.data(), *s = sb.data(), *acc = ab.data(), *m = mb.data(), *l = lb.data(), *cr = cb.data();
    for (size_t i = 0; i < Mp; ++i) {                     // row i = (head gi, token ti); pad rows see all keys
        if (i >= M) { qp[i] = last; continue; }
        const size_t gi = i / nt, ti = i % nt;
        qp[i] = kend ? last : first + ti;
        const float* src = Q + (t0 + ti) * qw + (g * grp + gi) * hd;
        for (size_t d = 0; d < hd; ++d) q[i * hd + d] = src[d] * scale;
    }
    const float* kt = KT + g * hd * P;
    const float* vf = VF + g * P * hd;
    for (size_t c0 = kb; c0 <= last; c0 += 64) {
        const size_t kc = std::min<size_t>(64, last + 1 - c0);
        // S = Q K^T over the chunk's 64 keys
        for (size_t i0 = 0; i0 < Mp; i0 += kRB) {
            v16f a00{}, a01{}, a02{}, a03{}, a10{}, a11{}, a12{}, a13{}, a20{}, a21{}, a22{}, a23{}, a30{}, a31{}, a32{}, a33{};
            const float *q0 = q + i0 * hd, *q1 = q0 + hd, *q2 = q1 + hd, *q3 = q2 + hd;
            for (size_t d = 0; d < hd; ++d) {
                const float* kd = kt + ((c0 - kb) / 64 * hd + d) * 64;
                const v16f k0 = ld(kd), k1 = ld(kd + 16), k2 = ld(kd + 32), k3 = ld(kd + 48);
                const v16f b0 = bc(q0[d]), b1 = bc(q1[d]), b2 = bc(q2[d]), b3 = bc(q3[d]);
                a00 += b0 * k0; a01 += b0 * k1; a02 += b0 * k2; a03 += b0 * k3;
                a10 += b1 * k0; a11 += b1 * k1; a12 += b1 * k2; a13 += b1 * k3;
                a20 += b2 * k0; a21 += b2 * k1; a22 += b2 * k2; a23 += b2 * k3;
                a30 += b3 * k0; a31 += b3 * k1; a32 += b3 * k2; a33 += b3 * k3;
            }
            float* sr = s + i0 * 64;
            st(sr, a00); st(sr + 16, a01); st(sr + 32, a02); st(sr + 48, a03);
            st(sr + 64, a10); st(sr + 80, a11); st(sr + 96, a12); st(sr + 112, a13);
            st(sr + 128, a20); st(sr + 144, a21); st(sr + 160, a22); st(sr + 176, a23);
            st(sr + 192, a30); st(sr + 208, a31); st(sr + 224, a32); st(sr + 240, a33);
        }
        // causal mask (the diagonal chunks only), then the online softmax update per row
        for (size_t i = 0; i < Mp; ++i) {
            float* si = s + i * 64;
            const size_t valid = qp[i] < c0 ? 0 : std::min(kc, qp[i] - c0 + 1);
            const size_t lo = window && qp[i] + 1 > window ? qp[i] + 1 - window : 0;   // the row's first key
            if (!valid || lo >= c0 + valid) {
                for (size_t j = 0; j < 64; ++j) si[j] = 0.f;
                cr[i] = 1.f;
                continue;
            }
            for (size_t j = valid; j < 64; ++j) si[j] = -1e30f;
            for (size_t j = 0; c0 + j < lo; ++j) si[j] = -1e30f;
            const v16f x0 = ld(si), x1 = ld(si + 16), x2 = ld(si + 32), x3 = ld(si + 48);
            v16f vm = x0 > x1 ? x0 : x1;
            const v16f vm2 = x2 > x3 ? x2 : x3;
            vm = vm > vm2 ? vm : vm2;
            float mx = m[i];
            for (int e = 0; e < 16; ++e) mx = std::max(mx, vm[e]);
            const v16f bm = bc(mx);
            const v16f p0 = vexp(x0 - bm), p1 = vexp(x1 - bm), p2 = vexp(x2 - bm), p3 = vexp(x3 - bm);
            st(si, p0); st(si + 16, p1); st(si + 32, p2); st(si + 48, p3);
            for (size_t j = valid; j < 64; ++j) si[j] = 0.f;
            const v16f ps = (ld(si) + ld(si + 16)) + (ld(si + 32) + ld(si + 48));
            float sum = 0.f;
            for (int e = 0; e < 16; ++e) sum += ps[e];
            const float corr = vexp(bc(m[i] - mx))[0];
            l[i] = l[i] * corr + sum;
            m[i] = mx;
            cr[i] = corr;
        }
        // acc = acc * corr + P V
        for (size_t dg = 0; dg < hd; dg += 64)
            for (size_t i0 = 0; i0 < Mp; i0 += kRB) {
                float *c0r = acc + i0 * hd + dg, *c1r = c0r + hd, *c2r = c1r + hd, *c3r = c2r + hd;
                const v16f r0 = bc(cr[i0]), r1 = bc(cr[i0 + 1]), r2 = bc(cr[i0 + 2]), r3 = bc(cr[i0 + 3]);
                v16f a00 = ld(c0r) * r0, a01 = ld(c0r + 16) * r0, a02 = ld(c0r + 32) * r0, a03 = ld(c0r + 48) * r0;
                v16f a10 = ld(c1r) * r1, a11 = ld(c1r + 16) * r1, a12 = ld(c1r + 32) * r1, a13 = ld(c1r + 48) * r1;
                v16f a20 = ld(c2r) * r2, a21 = ld(c2r + 16) * r2, a22 = ld(c2r + 32) * r2, a23 = ld(c2r + 48) * r2;
                v16f a30 = ld(c3r) * r3, a31 = ld(c3r + 16) * r3, a32 = ld(c3r + 32) * r3, a33 = ld(c3r + 48) * r3;
                const float *p0 = s + i0 * 64, *p1 = p0 + 64, *p2 = p1 + 64, *p3 = p2 + 64;
                for (size_t j = 0; j < kc; ++j) {
                    const float* vj = vf + (c0 + j - kb) * hd + dg;
                    const v16f v0 = ld(vj), v1 = ld(vj + 16), v2 = ld(vj + 32), v3 = ld(vj + 48);
                    const v16f b0 = bc(p0[j]), b1 = bc(p1[j]), b2 = bc(p2[j]), b3 = bc(p3[j]);
                    a00 += b0 * v0; a01 += b0 * v1; a02 += b0 * v2; a03 += b0 * v3;
                    a10 += b1 * v0; a11 += b1 * v1; a12 += b1 * v2; a13 += b1 * v3;
                    a20 += b2 * v0; a21 += b2 * v1; a22 += b2 * v2; a23 += b2 * v3;
                    a30 += b3 * v0; a31 += b3 * v1; a32 += b3 * v2; a33 += b3 * v3;
                }
                st(c0r, a00); st(c0r + 16, a01); st(c0r + 32, a02); st(c0r + 48, a03);
                st(c1r, a10); st(c1r + 16, a11); st(c1r + 32, a12); st(c1r + 48, a13);
                st(c2r, a20); st(c2r + 16, a21); st(c2r + 32, a22); st(c2r + 48, a23);
                st(c3r, a30); st(c3r + 16, a31); st(c3r + 32, a32); st(c3r + 48, a33);
            }
    }
    for (size_t i = 0; i < M; ++i) {
        const size_t gi = i / nt, ti = i % nt;
        float* out = og + (t0 + ti) * qw + (g * grp + gi) * hd;
        if (sinks) l[i] += std::exp(sinks[g * grp + gi] - m[i]);
        const float inv = 1.f / l[i];
        for (size_t d = 0; d < hd; ++d) out[d] = acc[i * hd + d] * inv;
    }
}
// (no pop: GCC reports -Wpsabi at the end of the file, for the clones)

void attention_tiles(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                     const uint16_t* kv, size_t kv_row_elems, float* og, size_t kend = 0, const float* sinks = nullptr,
                     size_t window = 0) {
    if ((sinks || window) && (hd % 64 || kend))
        throw std::runtime_error("open_qwen36: attention_tiles: sinks / a window need head dim % 64 and a causal block");
    if (hd % 64 == 0) {
        // kb: the first key any row's window reaches (64-aligned); rows below it are never read
        const size_t kb = window && pos0 + 1 > window ? (pos0 + 1 - window) / 64 * 64 : 0;
        const size_t rows = kend ? kend : pos0 + R, P = (rows - kb + 63) / 64 * 64, kvw = kvh * hd, grp = nh / kvh;
        static thread_local std::vector<float> KT, VF;
        KT.resize(kvh * hd * P);
        VF.resize(kvh * P * hd);
        float *const kt = KT.data(), *const vf = VF.data();   // thread_local: the workers need the caller's
        // the window once, in 64-row blocks per thread; the pad rows zero
#pragma omp parallel for
        for (long long pb = 0; pb < static_cast<long long>(P / 64); ++pb)
            for (size_t p = static_cast<size_t>(pb) * 64; p < static_cast<size_t>(pb) * 64 + 64; ++p) {
                const uint16_t* row = kv + (kb + p) * kv_row_elems;
                for (size_t gh = 0; gh < kvh; ++gh) {
                    float* kd = kt + gh * hd * P + (p / 64 * hd) * 64 + p % 64;
                    float* vd = vf + (gh * P + p) * hd;
                    for (size_t d = 0; d < hd; ++d) {
                        kd[d * 64] = kb + p < rows ? bf16_to_f32(row[gh * hd + d]) : 0.f;
                        vd[d] = kb + p < rows ? bf16_to_f32(row[kvw + gh * hd + d]) : 0.f;
                    }
                }
            }
        const size_t qt = std::max<size_t>(8, 128 / grp);   // ~128 query rows per work item
        const size_t chunks = (R + qt - 1) / qt;
        const long long items = static_cast<long long>(kvh * chunks);
#pragma omp parallel for schedule(dynamic, 1)
        for (long long it = items - 1; it >= 0; --it)
            attention_tile64(hd, nh, kvh, R, pos0, qt, scale, Q, kt, vf, P, og,
                             static_cast<size_t>(it) % kvh, (static_cast<size_t>(it) / kvh) * qt, kend, sinks, window, kb);
        return;
    }
    const size_t chunks = (R + kQT - 1) / kQT;
    const long long items = static_cast<long long>(kvh * chunks);
    // the later chunks see longer windows: hand them out first
#pragma omp parallel for schedule(dynamic, 1)
    for (long long it = items - 1; it >= 0; --it)
        attention_tile(hd, nh, kvh, R, pos0, scale, Q, kv, kv_row_elems, og,
                       static_cast<size_t>(it) % kvh, (static_cast<size_t>(it) / kvh) * kQT, kend);
}
}  // namespace

namespace {
// The flash kernels' team from 2048 keys (OFLM_OPEN_ATTN_THREADS, default the OMP places: the
// physical cores with OMP_PLACES=cores): decode passes run 8 threads, which suits the memory-bound
// weight products, but attention over a long cache is compute-bound (32K keys, 4 rows: 95 -> 50
// ms a pass with 16)
int flash_threads(size_t keys) {
    static const int n = [] {
        if (const char* e = std::getenv("OFLM_OPEN_ATTN_THREADS")) return std::max(1, std::atoi(e));
        return std::max(0, omp_get_num_places());
    }();
    const int cur = omp_get_max_threads();
    return keys >= 2048 && n > cur ? n : cur;
}

// exp for x <= 0 (softmax terms): 2^(x log2 e) as 2^f (degree-7 polynomial, f in [0, 1), relative
// error < 1e-6) scaled by 2^floor; lanes under -87 give 0
__attribute__((target("avx512f,avx512dq,fma")))
inline __m512 exp512_neg(__m512 x) {
    const __m512 t = _mm512_mul_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.f)), _mm512_set1_ps(1.44269504088896341f));
    const __m512 fi = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(t, fi);
    __m512 p = _mm512_set1_ps(1.5252733804059840e-5f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.5403530393381609e-4f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.3333558146428443e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(9.6181291076284772e-3f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(5.5504108664821580e-2f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(2.4022650695910071e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(6.9314718055994531e-1f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.f));
    const __m512 r = _mm512_scalef_ps(p, fi);
    return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(x, _mm512_set1_ps(-87.f), _CMP_GT_OQ), r);
}

// 16 bf16 -> 16 floats
__attribute__((target("avx512f,avx512bw,avx512dq,fma")))
inline __m512 ld_bf16x16(const uint16_t* p) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p))), 16));
}
// 16 x 16 floats transposed in registers: r[i] lane j -> r[j] lane i
__attribute__((target("avx512f,avx512dq,fma")))
inline void transpose16(__m512 r[16]) {
    __m512 t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_ps(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_ps(r[i], r[i + 1]);
    }
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_shuffle_ps(t[i], t[i + 2], _MM_SHUFFLE(1, 0, 1, 0));
        r[i + 1] = _mm512_shuffle_ps(t[i], t[i + 2], _MM_SHUFFLE(3, 2, 3, 2));
        r[i + 2] = _mm512_shuffle_ps(t[i + 1], t[i + 3], _MM_SHUFFLE(1, 0, 1, 0));
        r[i + 3] = _mm512_shuffle_ps(t[i + 1], t[i + 3], _MM_SHUFFLE(3, 2, 3, 2));
    }
    for (int i = 0; i < 16; i += 8)
        for (int j = 0; j < 4; ++j) {
            t[i + j] = _mm512_shuffle_f32x4(r[i + j], r[i + j + 4], _MM_SHUFFLE(2, 0, 2, 0));
            t[i + j + 4] = _mm512_shuffle_f32x4(r[i + j], r[i + j + 4], _MM_SHUFFLE(3, 1, 3, 1));
        }
    for (int j = 0; j < 8; ++j) {
        r[j] = _mm512_shuffle_f32x4(t[j], t[j + 8], _MM_SHUFFLE(2, 0, 2, 0));
        r[j + 8] = _mm512_shuffle_f32x4(t[j], t[j + 8], _MM_SHUFFLE(3, 1, 3, 1));
    }
}

// Decode-sized causal attention (1-8 query rows) as flash decoding over long caches: a work item
// is (kv head, 512-key chunk) with every query that reads that head -- R rows x grp heads, at most
// 64 -- so each K / V row is converted once for all of them. In the item, 32-key blocks: K
// transposed to floats, the scores 32 at a time per query, an online softmax with the
// accumulators in registers (64 dims a pass), a vector exp. One partial (max, sum, sum p v) per
// (row, head, chunk), merged in chunk order. Chunks are fixed from key 0, so a row's result
// depends only on its own position (the same bits alone or in a verify pass), as the split path.
// At 16K keys, 4 rows, the 27B: the split path's partials (one per 32 keys, ~50 MB a layer) and
// per-key accumulator traffic made a verify pass's attention ~330 ms.
__attribute__((target("avx512f,avx512bw,avx512dq,fma")))
void attention_decode_flash(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                            const uint16_t* kv, size_t kvr, float* og) {
    constexpr size_t CK = 512, KB = 32, MAXQ = 64;
    const size_t grp = nh / kvh, qw = nh * hd, Kmax = pos0 + R, NC = (Kmax + CK - 1) / CK, NQ = R * grp, PS = hd + 2;
    if (NQ > MAXQ || hd % 64 || hd > 512) throw std::runtime_error("open_qwen36: attention_decode_flash: geometry");
    static thread_local std::vector<float> part_tl;
    std::vector<float>& part = part_tl;
    part.resize(R * nh * NC * PS);
    float* pp = part.data();
#pragma omp parallel num_threads(flash_threads(Kmax))
    {
        // per thread: the scaled queries, their accumulators, K transposed and V as floats
        std::vector<float, std::allocator<float>> buf(MAXQ * hd * 2 + hd * KB + KB * hd + MAXQ * KB + 64);
        float* base = reinterpret_cast<float*>((reinterpret_cast<uintptr_t>(buf.data()) + 63) & ~uintptr_t(63));
        float* qs = base;                       // [NQ][hd]
        float* acc = qs + MAXQ * hd;            // [NQ][hd]
        float* KT = acc + MAXQ * hd;            // [hd][KB]
        float* Vf = KT + hd * KB;               // [KB][hd]
        float* pr = Vf + KB * hd;               // [NQ][KB]
        float m[MAXQ], l[MAXQ];
#pragma omp for schedule(dynamic, 1)
        for (long long it = 0; it < static_cast<long long>(kvh * NC); ++it) {
            const size_t kh = static_cast<size_t>(it) % kvh, c = static_cast<size_t>(it) / kvh;
            const size_t k0 = c * CK, k1 = std::min(k0 + CK, Kmax);
            // query qi = t grp + g: row t, head kh grp + g, sees keys < pos0 + t + 1
            for (size_t t = 0; t < R; ++t)
                for (size_t g2 = 0; g2 < grp; ++g2) {
                    const size_t qi = t * grp + g2;
                    const float* qp = Q + t * qw + (kh * grp + g2) * hd;
                    for (size_t d = 0; d < hd; d += 16)
                        _mm512_store_ps(qs + qi * hd + d, _mm512_mul_ps(_mm512_loadu_ps(qp + d), _mm512_set1_ps(scale)));
                    std::fill(acc + qi * hd, acc + (qi + 1) * hd, 0.f);
                    m[qi] = -INFINITY;
                    l[qi] = 0.f;
                }
            for (size_t b0 = k0; b0 < k1; b0 += KB) {
                const size_t nk = std::min(KB, k1 - b0);
                // K as floats transposed (KT [hd][KB], 16 x 16 tiles in registers), V as floats [KB][hd]
                for (size_t jg = 0; jg < KB; jg += 16)
                    for (size_t d = 0; d < hd; d += 16) {
                        __m512 r[16];
                        for (size_t i = 0; i < 16; ++i)
                            r[i] = jg + i < nk ? ld_bf16x16(kv + (b0 + jg + i) * kvr + kh * hd + d) : _mm512_setzero_ps();
                        transpose16(r);
                        for (size_t i = 0; i < 16; ++i) _mm512_store_ps(KT + (d + i) * KB + jg, r[i]);
                    }
                for (size_t j = 0; j < KB; ++j) {
                    float* vd = Vf + j * hd;
                    if (j < nk) {
                        const uint16_t* vp = kv + (b0 + j) * kvr + kvh * hd + kh * hd;
                        // the next block's K and V slices of this key slot: every line (the slices are
                        // 4 KB apart, too far for the hardware prefetcher to run ahead)
                        if (b0 + KB + j < k1) {
                            const char* nkp = reinterpret_cast<const char*>(kv + (b0 + KB + j) * kvr + kh * hd);
                            const char* nvp = reinterpret_cast<const char*>(kv + (b0 + KB + j) * kvr + kvh * hd + kh * hd);
                            for (size_t l = 0; l < hd * 2; l += 64) {
                                _mm_prefetch(nkp + l, _MM_HINT_T0);
                                _mm_prefetch(nvp + l, _MM_HINT_T0);
                            }
                        }
                        for (size_t d = 0; d < hd; d += 16) _mm512_store_ps(vd + d, ld_bf16x16(vp + d));
                    } else {
                        std::fill(vd, vd + hd, 0.f);
                    }
                }
                // scores, 4 queries at a time: 8 independent accumulator chains share each K load
                alignas(64) float sc[MAXQ][KB];
                for (size_t q0 = 0; q0 < NQ; q0 += 4) {
                    const size_t nq4 = std::min<size_t>(4, NQ - q0);
                    __m512 s[4][2];
                    for (int u = 0; u < 4; ++u) s[u][0] = s[u][1] = _mm512_setzero_ps();
                    const float* qa = qs + q0 * hd;
                    for (size_t d = 0; d < hd; ++d) {
                        const __m512 k0v = _mm512_load_ps(KT + d * KB), k1v = _mm512_load_ps(KT + d * KB + 16);
                        for (int u = 0; u < 4; ++u) {
                            const __m512 qd = _mm512_set1_ps(qa[(u < static_cast<int>(nq4) ? u : 0) * hd + d]);
                            s[u][0] = _mm512_fmadd_ps(qd, k0v, s[u][0]);
                            s[u][1] = _mm512_fmadd_ps(qd, k1v, s[u][1]);
                        }
                    }
                    for (size_t u = 0; u < nq4; ++u) {
                        _mm512_store_ps(sc[q0 + u], s[u][0]);
                        _mm512_store_ps(sc[q0 + u] + 16, s[u][1]);
                    }
                }
                // the online softmax per query: p for this block, the old accumulator's factor
                float corr[MAXQ];
                size_t nvq[MAXQ];
                for (size_t qi = 0; qi < NQ; ++qi) {
                    const size_t t = qi / grp, lim = pos0 + t + 1;         // keys < lim
                    const size_t nv = b0 >= lim ? 0 : std::min(nk, lim - b0);
                    nvq[qi] = nv;
                    float* pq = pr + qi * KB;
                    if (!nv) { corr[qi] = 1.f; std::fill(pq, pq + KB, 0.f); continue; }
                    const __mmask16 v0 = nv >= 16 ? __mmask16(0xFFFF) : __mmask16((1u << nv) - 1);
                    const __mmask16 v1 = nv >= 32 ? __mmask16(0xFFFF) : nv > 16 ? __mmask16((1u << (nv - 16)) - 1) : __mmask16(0);
                    const __m512 s0 = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), v0, _mm512_load_ps(sc[qi]));
                    const __m512 s1 = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), v1, _mm512_load_ps(sc[qi] + 16));
                    const float bm = std::max(_mm512_reduce_max_ps(s0), _mm512_reduce_max_ps(s1));
                    const float mn = std::max(m[qi], bm);
                    corr[qi] = m[qi] == -INFINITY ? 0.f : std::exp(m[qi] - mn);
                    const __m512 p0 = _mm512_maskz_mov_ps(v0, exp512_neg(_mm512_sub_ps(s0, _mm512_set1_ps(mn))));
                    const __m512 p1 = _mm512_maskz_mov_ps(v1, exp512_neg(_mm512_sub_ps(s1, _mm512_set1_ps(mn))));
                    l[qi] = l[qi] * corr[qi] + (_mm512_reduce_add_ps(p0) + _mm512_reduce_add_ps(p1));
                    m[qi] = mn;
                    _mm512_store_ps(pq, p0);
                    _mm512_store_ps(pq + 16, p1);
                }
                // sum p v, 2 queries x 64 dims at a time (8 chains sharing each V load); p is 0 past a
                // query's valid keys, so the pair runs to the longer of the two
                for (size_t q0 = 0; q0 < NQ; q0 += 2) {
                    const size_t q1 = std::min(q0 + 1, NQ - 1), nv = std::max(nvq[q0], nvq[q1]);
                    const bool two = q1 != q0;
                    if (!nv) continue;
                    float* a0p = acc + q0 * hd;
                    float* a1p = acc + q1 * hd;
                    const float* p0q = pr + q0 * KB;
                    const float* p1q = pr + q1 * KB;
                    const __m512 c0 = _mm512_set1_ps(corr[q0]), c1 = _mm512_set1_ps(corr[q1]);
                    for (size_t d = 0; d < hd; d += 64) {
                        __m512 x0 = _mm512_mul_ps(_mm512_load_ps(a0p + d), c0), x1 = _mm512_mul_ps(_mm512_load_ps(a0p + d + 16), c0);
                        __m512 x2 = _mm512_mul_ps(_mm512_load_ps(a0p + d + 32), c0), x3 = _mm512_mul_ps(_mm512_load_ps(a0p + d + 48), c0);
                        __m512 y0 = _mm512_mul_ps(_mm512_load_ps(a1p + d), c1), y1 = _mm512_mul_ps(_mm512_load_ps(a1p + d + 16), c1);
                        __m512 y2 = _mm512_mul_ps(_mm512_load_ps(a1p + d + 32), c1), y3 = _mm512_mul_ps(_mm512_load_ps(a1p + d + 48), c1);
                        for (size_t j = 0; j < nv; ++j) {
                            const float* vr = Vf + j * hd + d;
                            const __m512 v0 = _mm512_load_ps(vr), v1 = _mm512_load_ps(vr + 16), v2 = _mm512_load_ps(vr + 32), v3 = _mm512_load_ps(vr + 48);
                            const __m512 pa = _mm512_set1_ps(p0q[j]), pb = _mm512_set1_ps(p1q[j]);
                            x0 = _mm512_fmadd_ps(pa, v0, x0); x1 = _mm512_fmadd_ps(pa, v1, x1);
                            x2 = _mm512_fmadd_ps(pa, v2, x2); x3 = _mm512_fmadd_ps(pa, v3, x3);
                            y0 = _mm512_fmadd_ps(pb, v0, y0); y1 = _mm512_fmadd_ps(pb, v1, y1);
                            y2 = _mm512_fmadd_ps(pb, v2, y2); y3 = _mm512_fmadd_ps(pb, v3, y3);
                        }
                        _mm512_store_ps(a0p + d, x0); _mm512_store_ps(a0p + d + 16, x1);
                        _mm512_store_ps(a0p + d + 32, x2); _mm512_store_ps(a0p + d + 48, x3);
                        if (two) {
                            _mm512_store_ps(a1p + d, y0); _mm512_store_ps(a1p + d + 16, y1);
                            _mm512_store_ps(a1p + d + 32, y2); _mm512_store_ps(a1p + d + 48, y3);
                        }
                    }
                }
            }
            for (size_t qi = 0; qi < NQ; ++qi) {
                const size_t t = qi / grp, h = kh * grp + qi % grp;
                float* P = pp + ((t * nh + h) * NC + c) * PS;
                P[0] = m[qi]; P[1] = l[qi];
                std::copy(acc + qi * hd, acc + (qi + 1) * hd, P + 2);
            }
        }
        // merge the chunks in order
#pragma omp for schedule(static) collapse(2)
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt)
            for (long long hh = 0; hh < static_cast<long long>(nh); ++hh) {
                const size_t t = static_cast<size_t>(tt), h = static_cast<size_t>(hh);
                const float* P0 = pp + (t * nh + h) * NC * PS;
                const size_t nc = (pos0 + t + 1 + CK - 1) / CK;           // the row's chunks
                float M = -INFINITY;
                for (size_t c = 0; c < nc; ++c) M = std::max(M, P0[c * PS]);
                float L = 0.f;
                float* o = og + t * qw + h * hd;
                std::fill(o, o + hd, 0.f);
                for (size_t c = 0; c < nc; ++c) {
                    const float* P = P0 + c * PS;
                    if (P[0] == -INFINITY) continue;
                    const float w = std::exp(P[0] - M);
                    L += P[1] * w;
                    for (size_t i = 0; i < hd; ++i) o[i] += P[2 + i] * w;
                }
                const float inv = 1.f / L;
                for (size_t i = 0; i < hd; ++i) o[i] *= inv;
            }
    }
}
}  // namespace

namespace {
// The same flash decoding with AVX-512 BF16 dot products (vdpbf16ps: 32 bf16 products into 16
// fp32 sums): the scores from the bf16 cache directly (Q rounded to bf16, K pairs transposed in
// registers, no float conversion), sum p v with p rounded to bf16 and V rows paired by key. Half
// the instructions of the fp32 kernel; the rounding is the NPU attention's (bf16 Q / K / V / P).
__attribute__((target("avx512f,avx512bw,avx512dq,avx512bf16,fma")))
void attention_decode_flash_bf16(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                                 const uint16_t* kv, size_t kvr, float* og, const float* sinks = nullptr, size_t window = 0) {
    constexpr size_t CK = 512, KB = 32, MAXQ = 64;
    // window (Flint's sliding layers): row t sees keys [pos0 + t + 1 - window, pos0 + t]; the chunks start
    // at the first one's 32-key block (kv may be a sliding buffer's logical pointer: rows below never read)
    const size_t KB0 = window && pos0 + 1 > window ? (pos0 + 1 - window) / KB * KB : 0;
    const size_t grp = nh / kvh, qw = nh * hd, Kmax = pos0 + R, NC = (Kmax - KB0 + CK - 1) / CK, NQ = R * grp, PS = hd + 2;
    const size_t HP = hd / 2;                   // bf16 pairs a row
    if (NQ > MAXQ || hd % 64 || hd > 512) throw std::runtime_error("open_qwen36: attention_decode_flash_bf16: geometry");
    static thread_local std::vector<float> part_tl;
    std::vector<float>& part = part_tl;
    part.resize(R * nh * NC * PS);
    float* pp = part.data();
#pragma omp parallel num_threads(flash_threads(Kmax))
    {
        std::vector<uint32_t> buf(MAXQ * HP + HP * KB + (KB / 2) * hd + MAXQ * hd + 64);
        uint32_t* base = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(buf.data()) + 63) & ~uintptr_t(63));
        uint32_t* qb = base;                                   // [NQ][HP] bf16 pairs of the scaled queries
        uint32_t* KT = qb + MAXQ * HP;                         // [HP][KB] key pairs transposed
        uint32_t* VI = KT + HP * KB;                           // [KB / 2][hd] (v_2j, v_2j+1) per dim
        float* acc = reinterpret_cast<float*>(VI + (KB / 2) * hd);   // [NQ][hd]
        float m[MAXQ], l[MAXQ], corr[MAXQ];
        size_t nvq[MAXQ];
        alignas(64) float sc[MAXQ][KB];
        alignas(64) uint16_t pb[MAXQ][KB];
#pragma omp for schedule(dynamic, 1)
        for (long long it = 0; it < static_cast<long long>(kvh * NC); ++it) {
            const size_t kh = static_cast<size_t>(it) % kvh, c = static_cast<size_t>(it) / kvh;
            const size_t k0 = KB0 + c * CK, k1 = std::min(k0 + CK, Kmax);
            for (size_t t = 0; t < R; ++t)
                for (size_t g2 = 0; g2 < grp; ++g2) {
                    const size_t qi = t * grp + g2;
                    const float* qp = Q + t * qw + (kh * grp + g2) * hd;
                    uint16_t* qh = reinterpret_cast<uint16_t*>(qb + qi * HP);
                    for (size_t d = 0; d < hd; d += 16)
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qh + d), reinterpret_cast<__m256i>(
                            _mm512_cvtneps_pbh(_mm512_mul_ps(_mm512_loadu_ps(qp + d), _mm512_set1_ps(scale)))));
                    std::fill(acc + qi * hd, acc + (qi + 1) * hd, 0.f);
                    m[qi] = -INFINITY;
                    l[qi] = 0.f;
                }
            for (size_t b0 = k0; b0 < k1; b0 += KB) {
                const size_t nk = std::min(KB, k1 - b0);
                // K pairs transposed: KT[p][j] = (k_j[2p], k_j[2p + 1]), 16 keys x 16 pairs a tile
                for (size_t jg = 0; jg < KB; jg += 16)
                    for (size_t p0 = 0; p0 < HP; p0 += 16) {
                        __m512 r[16];
                        for (size_t i = 0; i < 16; ++i)
                            r[i] = jg + i < nk ? _mm512_loadu_ps(reinterpret_cast<const float*>(kv + (b0 + jg + i) * kvr + kh * hd + 2 * p0))
                                              : _mm512_setzero_ps();
                        transpose16(r);
                        for (size_t i = 0; i < 16; ++i) _mm512_store_ps(reinterpret_cast<float*>(KT + (p0 + i) * KB + jg), r[i]);
                    }
                // V paired by key: VI[jp][d] = v_2jp[d] (low) | v_2jp+1[d] (high)
                for (size_t jp = 0; jp < KB / 2; ++jp) {
                    const size_t ja = 2 * jp, jb = ja + 1;
                    const uint16_t* va = kv + (b0 + ja) * kvr + kvh * hd + kh * hd;
                    const uint16_t* vb = kv + (b0 + jb) * kvr + kvh * hd + kh * hd;
                    if (b0 + KB + ja < k1) {
                        const char* nkp = reinterpret_cast<const char*>(kv + (b0 + KB + ja) * kvr + kh * hd);
                        const char* nvp = reinterpret_cast<const char*>(kv + (b0 + KB + ja) * kvr + kvh * hd + kh * hd);
                        for (size_t l2 = 0; l2 < hd * 2; l2 += 64) {
                            _mm_prefetch(nkp + l2, _MM_HINT_T0); _mm_prefetch(nvp + l2, _MM_HINT_T0);
                            _mm_prefetch(nkp + kvr * 2 + l2, _MM_HINT_T0); _mm_prefetch(nvp + kvr * 2 + l2, _MM_HINT_T0);
                        }
                    }
                    for (size_t d = 0; d < hd; d += 16) {
                        const __m512i lo = ja < nk ? _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(va + d))) : _mm512_setzero_si512();
                        const __m512i hi = jb < nk ? _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(vb + d))) : _mm512_setzero_si512();
                        _mm512_store_si512(VI + jp * hd + d, _mm512_or_si512(lo, _mm512_slli_epi32(hi, 16)));
                    }
                }
                // scores, 6 queries at a time (12 sums in flight: 8 left the dot products' latency
                // exposed)
                for (size_t q0 = 0; q0 < NQ; q0 += 6) {
                    const size_t nq6 = std::min<size_t>(6, NQ - q0);
                    const uint32_t* qr[6];
                    for (size_t u = 0; u < 6; ++u) qr[u] = qb + (q0 + (u < nq6 ? u : 0)) * HP;
                    __m512 s6[6][2];
                    for (int u = 0; u < 6; ++u) s6[u][0] = s6[u][1] = _mm512_setzero_ps();
                    for (size_t p = 0; p < HP; ++p) {
                        const __m512bh ka = reinterpret_cast<__m512bh>(_mm512_load_si512(KT + p * KB));
                        const __m512bh kb2 = reinterpret_cast<__m512bh>(_mm512_load_si512(KT + p * KB + 16));
                        for (int u = 0; u < 6; ++u) {
                            const __m512bh qv = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(qr[u][p])));
                            s6[u][0] = _mm512_dpbf16_ps(s6[u][0], qv, ka);
                            s6[u][1] = _mm512_dpbf16_ps(s6[u][1], qv, kb2);
                        }
                    }
                    for (size_t u = 0; u < nq6; ++u) {
                        _mm512_store_ps(sc[q0 + u], s6[u][0]);
                        _mm512_store_ps(sc[q0 + u] + 16, s6[u][1]);
                    }
                }
                // the online softmax; p rounded to bf16 (the sums use the rounded values too)
                for (size_t qi = 0; qi < NQ; ++qi) {
                    const size_t t = qi / grp, lim = pos0 + t + 1, lo = window && lim > window ? lim - window : 0;
                    size_t nv = b0 >= lim ? 0 : std::min(nk, lim - b0);
                    const size_t nlo = lo > b0 ? std::min(nk, lo - b0) : 0;   // the block's keys below the row's window
                    if (nlo >= nv) nv = 0;
                    nvq[qi] = nv;
                    if (!nv) { corr[qi] = 1.f; std::fill(pb[qi], pb[qi] + KB, uint16_t(0)); continue; }
                    const uint32_t vm = static_cast<uint32_t>((nv >= 32 ? 0xFFFFFFFFull : (1ull << nv) - 1) & ~((1ull << nlo) - 1));
                    const __mmask16 v0 = static_cast<__mmask16>(vm & 0xFFFF);
                    const __mmask16 v1 = static_cast<__mmask16>(vm >> 16);
                    const __m512 s0 = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), v0, _mm512_load_ps(sc[qi]));
                    const __m512 s1 = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), v1, _mm512_load_ps(sc[qi] + 16));
                    const float bm = std::max(_mm512_reduce_max_ps(s0), _mm512_reduce_max_ps(s1));
                    const float mn = std::max(m[qi], bm);
                    corr[qi] = m[qi] == -INFINITY ? 0.f : std::exp(m[qi] - mn);
                    const __m256bh b0h = _mm512_cvtneps_pbh(_mm512_maskz_mov_ps(v0, exp512_neg(_mm512_sub_ps(s0, _mm512_set1_ps(mn)))));
                    const __m256bh b1h = _mm512_cvtneps_pbh(_mm512_maskz_mov_ps(v1, exp512_neg(_mm512_sub_ps(s1, _mm512_set1_ps(mn)))));
                    _mm256_store_si256(reinterpret_cast<__m256i*>(pb[qi]), reinterpret_cast<__m256i>(b0h));
                    _mm256_store_si256(reinterpret_cast<__m256i*>(pb[qi] + 16), reinterpret_cast<__m256i>(b1h));
                    const __m512 r0 = _mm512_cvtpbh_ps(b0h), r1 = _mm512_cvtpbh_ps(b1h);
                    l[qi] = l[qi] * corr[qi] + (_mm512_reduce_add_ps(r0) + _mm512_reduce_add_ps(r1));
                    m[qi] = mn;
                }
                // sum p v, 3 queries x 64 dims a pass (12 sums in flight), 2 keys an instruction
                for (size_t q0 = 0; q0 < NQ; q0 += 3) {
                    const size_t q1 = std::min(q0 + 1, NQ - 1), q2 = std::min(q0 + 2, NQ - 1);
                    const size_t nv = std::max({nvq[q0], nvq[q1], nvq[q2]});
                    if (!nv) continue;
                    const size_t npair = (nv + 1) / 2;
                    float* a0p = acc + q0 * hd;
                    float* a1p = acc + q1 * hd;
                    float* a2p = acc + q2 * hd;
                    const uint32_t* p0w = reinterpret_cast<const uint32_t*>(pb[q0]);
                    const uint32_t* p1w = reinterpret_cast<const uint32_t*>(pb[q1]);
                    const uint32_t* p2w = reinterpret_cast<const uint32_t*>(pb[q2]);
                    const __m512 c0 = _mm512_set1_ps(corr[q0]), c1 = _mm512_set1_ps(corr[q1]), c2 = _mm512_set1_ps(corr[q2]);
                    for (size_t d = 0; d < hd; d += 64) {
                        __m512 x[3][4];
                        for (int i = 0; i < 4; ++i) {
                            x[0][i] = _mm512_mul_ps(_mm512_load_ps(a0p + d + 16 * i), c0);
                            x[1][i] = _mm512_mul_ps(_mm512_load_ps(a1p + d + 16 * i), c1);
                            x[2][i] = _mm512_mul_ps(_mm512_load_ps(a2p + d + 16 * i), c2);
                        }
                        for (size_t jp = 0; jp < npair; ++jp) {
                            const uint32_t* vr = VI + jp * hd + d;
                            const __m512bh pa = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(p0w[jp])));
                            const __m512bh pbv = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(p1w[jp])));
                            const __m512bh pc = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(p2w[jp])));
                            for (int i = 0; i < 4; ++i) {
                                const __m512bh v = reinterpret_cast<__m512bh>(_mm512_load_si512(vr + 16 * i));
                                x[0][i] = _mm512_dpbf16_ps(x[0][i], pa, v);
                                x[1][i] = _mm512_dpbf16_ps(x[1][i], pbv, v);
                                x[2][i] = _mm512_dpbf16_ps(x[2][i], pc, v);
                            }
                        }
                        // (a clamped duplicate in the last group holds the same sums as its query)
                        for (int i = 0; i < 4; ++i) {
                            _mm512_store_ps(a2p + d + 16 * i, x[2][i]);
                            _mm512_store_ps(a1p + d + 16 * i, x[1][i]);
                            _mm512_store_ps(a0p + d + 16 * i, x[0][i]);
                        }
                    }
                }
            }
            for (size_t qi = 0; qi < NQ; ++qi) {
                const size_t t = qi / grp, h = kh * grp + qi % grp;
                float* P = pp + ((t * nh + h) * NC + c) * PS;
                P[0] = m[qi]; P[1] = l[qi];
                std::copy(acc + qi * hd, acc + (qi + 1) * hd, P + 2);
            }
        }
#pragma omp for schedule(static) collapse(2)
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt)
            for (long long hh = 0; hh < static_cast<long long>(nh); ++hh) {
                const size_t t = static_cast<size_t>(tt), h = static_cast<size_t>(hh);
                const float* P0 = pp + (t * nh + h) * NC * PS;
                const size_t nc = (pos0 + t + 1 - KB0 + CK - 1) / CK;
                float M = sinks ? sinks[h] : -INFINITY;
                for (size_t c = 0; c < nc; ++c) M = std::max(M, P0[c * PS]);
                float L = sinks ? std::exp(sinks[h] - M) : 0.f;   // a sink: one more logit in the denominator, no value
                float* o = og + t * qw + h * hd;
                std::fill(o, o + hd, 0.f);
                for (size_t c = 0; c < nc; ++c) {
                    const float* P = P0 + c * PS;
                    if (P[0] == -INFINITY) continue;
                    const float w = std::exp(P[0] - M);
                    L += P[1] * w;
                    for (size_t i = 0; i < hd; ++i) o[i] += P[2 + i] * w;
                }
                const float inv = 1.f / L;
                for (size_t i = 0; i < hd; ++i) o[i] *= inv;
            }
    }
}
// The int8 copy's layout: blocks of kKv8Blk keys, [block][kv head] kv8_blk_bytes(hd) each: K (hd/2
// lines: pair (2p, 2p + 1) of keys 0-15, then of keys 16-31), V (16 key pairs x hd/32 lines: dims
// d..d + 15, then d + 16..d + 31, keys (2jp, 2jp + 1) each), the 32 K scales, the 32 V scales (f32).
// A line is 64 bytes, value i at kv8_byte(i): the order kv8_cvt's unpacks put back. Values are
// sign-magnitude (sign bit 7, |q| <= 127), so their bf16 bytes come from two 128-entry tables.
constexpr size_t kKv8Blk = 32;
constexpr size_t kv8_blk_bytes(size_t hd) { return kKv8Blk * (2 * hd + 8); }
constexpr size_t kv8_byte(size_t i) { return ((i & 31) >> 3) * 16 + (i >> 5) * 8 + (i & 7); }
struct Kv8Tab { __m512i hi0, hi1, lo0, lo1; };
__attribute__((target("avx512f,avx512bw")))
inline Kv8Tab kv8_tab() {
    alignas(64) uint8_t t[4][64];
    for (int m = 0; m < 128; ++m) {
        const float f = static_cast<float>(m);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        t[m >> 6][m & 63] = static_cast<uint8_t>(u >> 24);           // bf16 high byte
        t[2 + (m >> 6)][m & 63] = static_cast<uint8_t>(u >> 16);     // bf16 low byte (exact: m has <= 7 bits)
    }
    return {_mm512_load_si512(t[0]), _mm512_load_si512(t[1]), _mm512_load_si512(t[2]), _mm512_load_si512(t[3])};
}
// one line (64 sign-magnitude int8) -> two vectors of 32 bf16, exact: values 0-31 in a, 32-63 in b
__attribute__((target("avx512f,avx512bw,avx512vbmi")))
inline void kv8_cvt(const uint8_t* line, const Kv8Tab& t, __m512i& a, __m512i& b) {
    const __m512i x = _mm512_loadu_si512(line);   // (the engine's buffer: no 64-byte alignment)
    const __m512i hi = _mm512_ternarylogic_epi32(_mm512_permutex2var_epi8(t.hi0, x, t.hi1), x, _mm512_set1_epi8(static_cast<char>(0x80)), 0xF8);
    const __m512i lo = _mm512_permutex2var_epi8(t.lo0, x, t.lo1);
    a = _mm512_unpacklo_epi8(lo, hi);
    b = _mm512_unpackhi_epi8(lo, hi);
}
// The bf16 flash decoding over an int8 copy of the cache (kv8_quant_rows: one scale per position
// and kv head, symmetric): K and V rows half the bytes; int8 values are exact in bf16, so the dot
// products are the bf16 kernel's, the K scales multiply each block's 32 scores and the V scales
// fold into the probabilities the value products read (the softmax sums keep the unscaled ones).
// The copy is in blocks of 32 keys (kKv8Blk, kv8_byte): a 64-byte line converts to two vectors of
// bf16 pairs with two table lookups and two unpacks (kv8_cvt), with no transpose; all of a kv
// head's queries (up to 8 a pass) read each converted line.
__attribute__((target("avx512f,avx512bw,avx512dq,avx512bf16,avx512vbmi,fma")))
void attention_decode_flash_kv8(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                                const uint8_t* kv8, size_t row8, float* og, const float* sinks = nullptr) {
    constexpr size_t CK = 512, KB = kKv8Blk, MAXQ = 64, KB0 = 0, window = 0;   // (no window: the full layers' copy)
    const size_t grp = nh / kvh, qw = nh * hd, Kmax = pos0 + R, NC = (Kmax - KB0 + CK - 1) / CK, NQ = R * grp, PS = hd + 2;
    const size_t HP = hd / 2, DL = hd / 32;      // bf16 pairs a row; V lines a key pair
    const size_t BLK = kv8_blk_bytes(hd);        // a block's bytes for one kv head: K, V, K scales, V scales
    if (NQ > MAXQ || hd % 64 || hd > 512 || row8 != 2 * kvh * hd + 8 * kvh)
        throw std::runtime_error("open_qwen36: attention_decode_flash_kv8: geometry");
    static thread_local std::vector<float> part_tl;
    std::vector<float>& part = part_tl;
    part.resize(R * nh * NC * PS);
    float* pp = part.data();
#pragma omp parallel num_threads(flash_threads(Kmax))
    {
        std::vector<uint32_t> buf(MAXQ * HP + MAXQ * hd + 64);
        uint32_t* base = reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(buf.data()) + 63) & ~uintptr_t(63));
        uint32_t* qb = base;                                   // [NQ][HP] bf16 pairs of the scaled queries
        float* acc = reinterpret_cast<float*>(qb + MAXQ * HP); // [NQ][hd]
        float m[MAXQ], l[MAXQ], corr[MAXQ];
        size_t nvq[MAXQ];
        alignas(64) float sc[MAXQ][KB];
        alignas(64) uint16_t pb[MAXQ][KB];
        const Kv8Tab tab = kv8_tab();
#pragma omp for schedule(dynamic, 1)
        for (long long it = 0; it < static_cast<long long>(kvh * NC); ++it) {
            const size_t kh = static_cast<size_t>(it) % kvh, c = static_cast<size_t>(it) / kvh;
            const size_t k0 = KB0 + c * CK, k1 = std::min(k0 + CK, Kmax);
            for (size_t t = 0; t < R; ++t)
                for (size_t g2 = 0; g2 < grp; ++g2) {
                    const size_t qi = t * grp + g2;
                    const float* qp = Q + t * qw + (kh * grp + g2) * hd;
                    uint16_t* qh = reinterpret_cast<uint16_t*>(qb + qi * HP);
                    for (size_t d = 0; d < hd; d += 16)
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qh + d), reinterpret_cast<__m256i>(
                            _mm512_cvtneps_pbh(_mm512_mul_ps(_mm512_loadu_ps(qp + d), _mm512_set1_ps(scale)))));
                    std::fill(acc + qi * hd, acc + (qi + 1) * hd, 0.f);
                    m[qi] = -INFINITY;
                    l[qi] = 0.f;
                }
            for (size_t b0 = k0; b0 < k1; b0 += KB) {
                const size_t nk = std::min(KB, k1 - b0);
                const uint8_t* blk = kv8 + ((b0 / KB) * kvh + kh) * BLK;
                const uint8_t* Kb = blk;                       // [HP lines]: keys 0-15, 16-31, pair (2p, 2p + 1)
                const uint8_t* Vb = blk + 32 * hd;             // [16 key pairs][DL lines]: dims d, d + 16; keys 2jp, 2jp + 1
                const float* sp = reinterpret_cast<const float*>(blk + 64 * hd);
                if (b0 + KB < k1) {   // the head's next block (a longer distance measured the same)
                    const char* nb = reinterpret_cast<const char*>(blk + kvh * BLK);
                    for (size_t o = 0; o < BLK; o += 64) _mm_prefetch(nb + o, _MM_HINT_T0);
                }
                // this block's per-key scales (K for the scores, V for the probabilities), keys past nk 0
                const __mmask16 n0 = static_cast<__mmask16>(nk >= 16 ? 0xFFFF : (1u << nk) - 1);
                const __mmask16 n1 = static_cast<__mmask16>(nk >= 32 ? 0xFFFF : nk > 16 ? (1u << (nk - 16)) - 1 : 0);
                const __m512 ks0 = _mm512_maskz_loadu_ps(n0, sp), ks1 = _mm512_maskz_loadu_ps(n1, sp + 16);
                const __m512 vs0 = _mm512_maskz_loadu_ps(n0, sp + 32), vs1 = _mm512_maskz_loadu_ps(n1, sp + 48);
                // scores, 8 queries at a time over each converted K line (keys past nk: masked below)
                for (size_t q0 = 0; q0 < NQ; q0 += 8) {
                    const size_t nq8 = std::min<size_t>(8, NQ - q0);
                    const uint32_t* qr[8];
                    for (size_t u = 0; u < 8; ++u) qr[u] = qb + (q0 + (u < nq8 ? u : 0)) * HP;
                    __m512 s8[8][2];
                    for (int u = 0; u < 8; ++u) s8[u][0] = s8[u][1] = _mm512_setzero_ps();
                    for (size_t p = 0; p < HP; ++p) {
                        __m512i ka, kb2;
                        kv8_cvt(Kb + p * 64, tab, ka, kb2);
                        for (int u = 0; u < 8; ++u) {
                            const __m512bh qv = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(qr[u][p])));
                            s8[u][0] = _mm512_dpbf16_ps(s8[u][0], qv, reinterpret_cast<__m512bh>(ka));
                            s8[u][1] = _mm512_dpbf16_ps(s8[u][1], qv, reinterpret_cast<__m512bh>(kb2));
                        }
                    }
                    for (size_t u = 0; u < nq8; ++u) {
                        _mm512_store_ps(sc[q0 + u], _mm512_mul_ps(s8[u][0], ks0));
                        _mm512_store_ps(sc[q0 + u] + 16, _mm512_mul_ps(s8[u][1], ks1));
                    }
                }
                // the online softmax; p rounded to bf16 (the sums use the rounded values too). 8 queries a
                // pass, in phases (masks and maxima, then the exponentials, then the sums), so their
                // reductions overlap; exp(0) = 1 exactly, so an unchanged maximum skips the call
                for (size_t q0 = 0; q0 < NQ; q0 += 8) {
                    const size_t nq8 = std::min<size_t>(8, NQ - q0);
                    __m512 S0[8], S1[8], P0[8], P1[8];
                    __mmask16 V0[8], V1[8];
                    float bm[8], mn[8];
                    for (size_t u = 0; u < nq8; ++u) {
                        const size_t qi = q0 + u;
                        const size_t t = qi / grp, lim = pos0 + t + 1, lo = window && lim > window ? lim - window : 0;
                        size_t nv = b0 >= lim ? 0 : std::min(nk, lim - b0);
                        const size_t nlo = lo > b0 ? std::min(nk, lo - b0) : 0;   // the block's keys below the row's window
                        if (nlo >= nv) nv = 0;
                        nvq[qi] = nv;
                        const uint32_t vm = nv ? static_cast<uint32_t>((nv >= 32 ? 0xFFFFFFFFull : (1ull << nv) - 1) & ~((1ull << nlo) - 1)) : 0;
                        V0[u] = static_cast<__mmask16>(vm & 0xFFFF);
                        V1[u] = static_cast<__mmask16>(vm >> 16);
                        S0[u] = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), V0[u], _mm512_load_ps(sc[qi]));
                        S1[u] = _mm512_mask_mov_ps(_mm512_set1_ps(-INFINITY), V1[u], _mm512_load_ps(sc[qi] + 16));
                    }
                    for (size_t u = 0; u < nq8; ++u) bm[u] = std::max(_mm512_reduce_max_ps(S0[u]), _mm512_reduce_max_ps(S1[u]));
                    for (size_t u = 0; u < nq8; ++u) {
                        const size_t qi = q0 + u;
                        if (!nvq[qi]) { corr[qi] = 1.f; std::fill(pb[qi], pb[qi] + KB, uint16_t(0)); continue; }
                        mn[u] = std::max(m[qi], bm[u]);
                        corr[qi] = m[qi] == -INFINITY ? 0.f : m[qi] == mn[u] ? 1.f : std::exp(m[qi] - mn[u]);
                        const __m256bh b0h = _mm512_cvtneps_pbh(_mm512_maskz_mov_ps(V0[u], exp512_neg(_mm512_sub_ps(S0[u], _mm512_set1_ps(mn[u])))));
                        const __m256bh b1h = _mm512_cvtneps_pbh(_mm512_maskz_mov_ps(V1[u], exp512_neg(_mm512_sub_ps(S1[u], _mm512_set1_ps(mn[u])))));
                        P0[u] = _mm512_cvtpbh_ps(b0h);
                        P1[u] = _mm512_cvtpbh_ps(b1h);
                        // p v = (p vs) v_int8: the V scales into the probabilities the products read
                        _mm256_store_si256(reinterpret_cast<__m256i*>(pb[qi]), reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(_mm512_mul_ps(P0[u], vs0))));
                        _mm256_store_si256(reinterpret_cast<__m256i*>(pb[qi] + 16), reinterpret_cast<__m256i>(_mm512_cvtneps_pbh(_mm512_mul_ps(P1[u], vs1))));
                    }
                    for (size_t u = 0; u < nq8; ++u) {
                        const size_t qi = q0 + u;
                        if (!nvq[qi]) continue;
                        l[qi] = l[qi] * corr[qi] + (_mm512_reduce_add_ps(P0[u]) + _mm512_reduce_add_ps(P1[u]));
                        m[qi] = mn[u];
                    }
                }
                // sum p v, 8 queries x 32 dims a pass over each converted V line, 2 keys an instruction
                for (size_t q0 = 0; q0 < NQ; q0 += 8) {
                    const size_t nq8 = std::min<size_t>(8, NQ - q0);
                    size_t qs[8], nv = 0;
                    for (size_t u = 0; u < 8; ++u) { qs[u] = q0 + (u < nq8 ? u : 0); nv = std::max(nv, nvq[qs[u]]); }
                    if (!nv) continue;
                    const size_t npair = (nv + 1) / 2;
                    const uint32_t* pw[8];
                    __m512 cr[8];
                    for (size_t u = 0; u < 8; ++u) { pw[u] = reinterpret_cast<const uint32_t*>(pb[qs[u]]); cr[u] = _mm512_set1_ps(corr[qs[u]]); }
                    for (size_t dl = 0; dl < DL; ++dl) {
                        __m512 x[8][2];
                        for (int u = 0; u < 8; ++u) {
                            x[u][0] = _mm512_mul_ps(_mm512_load_ps(acc + qs[u] * hd + dl * 32), cr[u]);
                            x[u][1] = _mm512_mul_ps(_mm512_load_ps(acc + qs[u] * hd + dl * 32 + 16), cr[u]);
                        }
                        for (size_t jp = 0; jp < npair; ++jp) {
                            __m512i va, vb;
                            kv8_cvt(Vb + (jp * DL + dl) * 64, tab, va, vb);
                            for (int u = 0; u < 8; ++u) {
                                const __m512bh pv = reinterpret_cast<__m512bh>(_mm512_set1_epi32(static_cast<int>(pw[u][jp])));
                                x[u][0] = _mm512_dpbf16_ps(x[u][0], pv, reinterpret_cast<__m512bh>(va));
                                x[u][1] = _mm512_dpbf16_ps(x[u][1], pv, reinterpret_cast<__m512bh>(vb));
                            }
                        }
                        // (a clamped duplicate in the last group holds the same sums as its query)
                        for (int u = 7; u >= 0; --u) {
                            _mm512_store_ps(acc + qs[u] * hd + dl * 32, x[u][0]);
                            _mm512_store_ps(acc + qs[u] * hd + dl * 32 + 16, x[u][1]);
                        }
                    }
                }
            }
            for (size_t qi = 0; qi < NQ; ++qi) {
                const size_t t = qi / grp, h = kh * grp + qi % grp;
                float* P = pp + ((t * nh + h) * NC + c) * PS;
                P[0] = m[qi]; P[1] = l[qi];
                std::copy(acc + qi * hd, acc + (qi + 1) * hd, P + 2);
            }
        }
#pragma omp for schedule(static) collapse(2)
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt)
            for (long long hh = 0; hh < static_cast<long long>(nh); ++hh) {
                const size_t t = static_cast<size_t>(tt), h = static_cast<size_t>(hh);
                const float* P0 = pp + (t * nh + h) * NC * PS;
                const size_t nc = (pos0 + t + 1 - KB0 + CK - 1) / CK;
                float M = sinks ? sinks[h] : -INFINITY;
                for (size_t c = 0; c < nc; ++c) M = std::max(M, P0[c * PS]);
                float L = sinks ? std::exp(sinks[h] - M) : 0.f;   // a sink: one more logit in the denominator, no value
                float* o = og + t * qw + h * hd;
                std::fill(o, o + hd, 0.f);
                for (size_t c = 0; c < nc; ++c) {
                    const float* P = P0 + c * PS;
                    if (P[0] == -INFINITY) continue;
                    const float w = std::exp(P[0] - M);
                    L += P[1] * w;
                    for (size_t i = 0; i < hd; ++i) o[i] += P[2 + i] * w;
                }
                const float inv = 1.f / L;
                for (size_t i = 0; i < hd; ++i) o[i] *= inv;
            }
    }
}
}  // namespace

namespace {
// Decode-sized causal attention (1-8 query rows: a decode step or an MTP verify pass), split over
// (kv head, 64-key block) work items so a step's few queries still fill the threads ("flash
// decoding"): each item keeps a partial softmax (max, sum, sum p v) per (row, head), merged in
// block order. Blocks are fixed from key 0, so a row's result depends only on its own position --
// the same bits whether it comes alone (decode) or with others (verify). At 4K keys, 8 threads:
// the tiled path took ~2 ms a layer (one work item per kv head).
void attention_decode_split(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                            const uint16_t* kv, size_t kvr, float* og) {
    // a work item = one 32-key block for every kv head: whole contiguous cache rows (K then V)
    constexpr size_t KB = 32;
    const size_t grp = nh / kvh, qw = nh * hd, kvw = kvh * hd, Kmax = pos0 + R, NB = (Kmax + KB - 1) / KB, PS = hd + 2;
    if (R > 8 || grp > 8 || hd % 16 || hd > 512 || kvw > 2048) throw std::runtime_error("open_qwen36: attention_decode_split: geometry");
    static thread_local std::vector<float> part_tl;
    std::vector<float>& part = part_tl;
    part.resize(R * nh * NB * PS);
    float* pp = part.data();
#pragma omp parallel
    {
        std::vector<float> rowf(kvw), sc(R * nh * KB);
#pragma omp for schedule(dynamic, 1)
        for (long long bb = 0; bb < static_cast<long long>(NB); ++bb) {
            const size_t b = static_cast<size_t>(bb), k0 = b * KB, k1 = std::min(k0 + KB, Kmax), nk = k1 - k0;
            // scores: each key's K row (every kv head) converted once
            for (size_t key = k0; key < k1; ++key) {
                const uint16_t* kp = kv + key * kvr;
                _mm_prefetch(reinterpret_cast<const char*>(kv + (key + 4) * kvr), _MM_HINT_T0);
                for (size_t i = 0; i < kvw; i += 8)
                    _mm256_storeu_ps(rowf.data() + i, _mm256_castsi256_ps(_mm256_slli_epi32(
                                                          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kp + i))), 16)));
                for (size_t t = 0; t < R; ++t)
                    for (size_t h = 0; h < nh; ++h) {
                        float& s0 = sc[(t * nh + h) * KB + (key - k0)];
                        if (key >= pos0 + t + 1) { s0 = -INFINITY; continue; }
                        const float* qp = Q + t * qw + h * hd;
                        const float* kf = rowf.data() + (h / grp) * hd;
                        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                        for (size_t i = 0; i < hd; i += 16) {
                            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(qp + i), _mm256_loadu_ps(kf + i), a0);
                            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(qp + i + 8), _mm256_loadu_ps(kf + i + 8), a1);
                        }
                        alignas(32) float l8[8];
                        _mm256_store_ps(l8, _mm256_add_ps(a0, a1));
                        s0 = (((l8[0] + l8[1]) + (l8[2] + l8[3])) + ((l8[4] + l8[5]) + (l8[6] + l8[7]))) * scale;
                    }
            }
            for (size_t t = 0; t < R; ++t)
                for (size_t h = 0; h < nh; ++h) {
                    float* s1 = sc.data() + (t * nh + h) * KB;
                    float m = -INFINITY;
                    for (size_t j = 0; j < nk; ++j) m = std::max(m, s1[j]);
                    float l = 0.f;
                    for (size_t j = 0; j < nk; ++j) {
                        s1[j] = m == -INFINITY ? 0.f : std::exp(s1[j] - m);
                        l += s1[j];
                    }
                    float* P = pp + ((t * nh + h) * NB + b) * PS;
                    P[0] = m; P[1] = l;
                    std::fill(P + 2, P + PS, 0.f);
                }
            // sum p v: each key's V row converted once
            for (size_t key = k0; key < k1; ++key) {
                const uint16_t* vp = kv + key * kvr + kvw;
                for (size_t i = 0; i < kvw; i += 8)
                    _mm256_storeu_ps(rowf.data() + i, _mm256_castsi256_ps(_mm256_slli_epi32(
                                                          _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vp + i))), 16)));
                for (size_t t = 0; t < R; ++t) {
                    if (key >= pos0 + t + 1) continue;
                    for (size_t h = 0; h < nh; ++h) {
                        const __m256 p = _mm256_set1_ps(sc[(t * nh + h) * KB + (key - k0)]);
                        const float* vf = rowf.data() + (h / grp) * hd;
                        float* acc = pp + ((t * nh + h) * NB + b) * PS + 2;
                        for (size_t i = 0; i < hd; i += 8)
                            _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(p, _mm256_loadu_ps(vf + i), _mm256_loadu_ps(acc + i)));
                    }
                }
            }
        }
        // merge the blocks in order
#pragma omp for schedule(static) collapse(2)
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt)
            for (long long h = 0; h < static_cast<long long>(nh); ++h) {
                const size_t t = static_cast<size_t>(tt);
                const float* P0 = pp + ((t * nh) + static_cast<size_t>(h)) * NB * PS;
                const size_t nb = (pos0 + t + 1 + KB - 1) / KB;           // the row's blocks
                float M = -INFINITY;
                for (size_t b = 0; b < nb; ++b) M = std::max(M, P0[b * PS]);
                float L = 0.f;
                float* o = og + t * qw + static_cast<size_t>(h) * hd;
                std::fill(o, o + hd, 0.f);
                for (size_t b = 0; b < nb; ++b) {
                    const float* P = P0 + b * PS;
                    if (P[0] == -INFINITY) continue;
                    const float w = std::exp(P[0] - M);
                    L += P[1] * w;
                    for (size_t i = 0; i < hd; ++i) o[i] += P[2 + i] * w;
                }
                const float inv = 1.f / L;
                for (size_t i = 0; i < hd; ++i) o[i] *= inv;
            }
    }
}
}  // namespace

// (the blocked layout of kv8_byte; the values as the scalar lrint(x * 127 / amax) gave, rounded to
// nearest even; one block of 32 rows a work item)
__attribute__((target("avx512f,avx512bw,avx512dq,fma")))
void kv8_quant_rows(const uint16_t* kv, size_t kv_row_elems, size_t kvh, size_t hd, size_t r0, size_t r1, uint8_t* kv8, size_t row8) {
    const size_t kvw = kvh * hd, BLK = kv8_blk_bytes(hd), DL = hd / 32;
    if (hd % 64 || hd > 512 || row8 != 2 * kvw + 8 * kvh) throw std::runtime_error("open_qwen36: kv8_quant_rows: geometry");
    if (r1 <= r0) return;
    const long long b0 = static_cast<long long>(r0 / kKv8Blk), b1 = static_cast<long long>((r1 - 1) / kKv8Blk);
#pragma omp parallel for schedule(static) if (r1 - r0 > 16)
    for (long long bi = b0; bi <= b1; ++bi) {
        alignas(64) uint8_t q[512];
        const size_t ra = std::max(r0, static_cast<size_t>(bi) * kKv8Blk), rb = std::min(r1, static_cast<size_t>(bi + 1) * kKv8Blk);
        for (size_t rr = ra; rr < rb; ++rr) {
            const uint16_t* src = kv + rr * kv_row_elems;
            const size_t j = rr % kKv8Blk;
            for (size_t part = 0; part < 2; ++part)            // K, then V
                for (size_t h = 0; h < kvh; ++h) {
                    const uint16_t* x = src + part * kvw + h * hd;
                    uint8_t* blk = kv8 + (static_cast<size_t>(bi) * kvh + h) * BLK;
                    __m512 mx = _mm512_setzero_ps();
                    for (size_t i = 0; i < hd; i += 16) mx = _mm512_max_ps(mx, _mm512_abs_ps(ld_bf16x16(x + i)));
                    const float amax = _mm512_reduce_max_ps(mx);
                    const float s = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
                    for (size_t i = 0; i < hd; i += 16) {
                        const __m512i v = _mm512_cvtps_epi32(_mm512_mul_ps(ld_bf16x16(x + i), _mm512_set1_ps(inv)));
                        const __m512i sm = _mm512_or_si512(_mm512_abs_epi32(v), _mm512_and_si512(_mm512_srli_epi32(v, 24), _mm512_set1_epi32(0x80)));
                        _mm_store_si128(reinterpret_cast<__m128i*>(q + i), _mm512_cvtepi32_epi8(sm));
                    }
                    if (part == 0) {
                        const size_t o = kv8_byte((j / 16) * 32 + 2 * (j % 16));   // the pair's two bytes are adjacent
                        for (size_t p = 0; p < hd / 2; ++p) std::memcpy(blk + p * 64 + o, q + 2 * p, 2);
                    } else {
                        uint8_t* vb = blk + 32 * hd + (j / 2) * DL * 64;
                        for (size_t d = 0; d < hd; ++d) vb[(d / 32) * 64 + kv8_byte(((d % 32) / 16) * 32 + 2 * (d % 16) + j % 2)] = q[d];
                    }
                    std::memcpy(blk + 64 * hd + part * 128 + j * 4, &s, 4);
                }
        }
    }
}

void attention_block(const AttnGeom& g, const float* q, const float* k, const float* v, const float* gate,
                     const float* qn, const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems,
                     float* og) {
    const size_t qw = g.nh * g.hd, kvw = g.kvh * g.hd, half = g.rot / 2, rows = g.pos0 + g.t_real;
    if (g.nh % g.kvh || g.t_real > g.T || g.rot > g.hd || kv_row_elems < 2 * kvw)
        throw std::runtime_error("open_qwen36: attention_block: inconsistent geometry");
    const size_t grp = g.nh / g.kvh, R = g.t_real;
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.hd));
    std::fill(og, og + g.T * qw, 0.f);

    // ---- phase 1: the block's queries normed and roped; its keys normed, roped and written
    // to the cache in bf16 with the values. The window is read straight from the bf16 cache in
    // phase 2, chunk by chunk -- converting the whole cache to floats per block was most of the
    // cost at a 4K window.
    // Parallel over tokens; each token's cos / sin computed once for all its heads (the same
    // values rope() computes per head, so the result is bit-identical -- per head in double it
    // was most of this phase at a 4K block).
    std::vector<float> Q(R * qw);
#pragma omp parallel
    {
        std::vector<float> kh(kvw), cs(2 * half);
#pragma omp for
        for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
            const size_t t = static_cast<size_t>(tt), p = g.pos0 + t;
            for (size_t i = 0; i < half; ++i) {
                const double a = static_cast<double>(p) * inv_freq[i];
                cs[i] = static_cast<float>(std::cos(a));
                cs[half + i] = static_cast<float>(std::sin(a));
            }
            auto rot = [&](float* x) {
                for (size_t i = 0; i < half; ++i) {
                    const float c = cs[i], s = cs[half + i], x1 = x[i], x2 = x[half + i];
                    x[i] = x1 * c - x2 * s;
                    x[half + i] = x2 * c + x1 * s;
                }
            };
            for (size_t h = 0; h < g.nh; ++h) {
                float* dst = Q.data() + t * qw + h * g.hd;
                rms_vec(q + t * qw + h * g.hd, g.hd, qn, g.eps, dst);
                rot(dst);
            }
            for (size_t h = 0; h < g.kvh; ++h) {
                rms_vec(k + t * kvw + h * g.hd, g.hd, kn, g.eps, kh.data() + h * g.hd);
                rot(kh.data() + h * g.hd);
            }
            for (size_t j = 0; j < kvw; ++j) {
                kv[p * kv_row_elems + j] = f32_to_bf16(kh[j]);
                kv[p * kv_row_elems + kvw + j] = f32_to_bf16(v[t * kvw + j]);
            }
        }
    }
    (void)rows;
    // ---- phase 2: tiled causal attention, one work item per (kv head, 16-query chunk): its
    // grp query heads x 16 tokens attend together, so every K / V row is read once for all of
    // them (GQA), with an online softmax over 64-row key chunks.
    // OFLM_OPEN_ATTN_SPLIT=0 keeps the tiled path for decode-sized blocks too
    static const bool split = [] { const char* e = std::getenv("OFLM_OPEN_ATTN_SPLIT"); return !(e && *e == '0'); }();
    // OFLM_OPEN_ATTN_FLASH (default on with AVX-512; 0 off): the flash-decoding kernel for these
    // blocks; g_attn_flash (tests) overrides
    static const bool flash_cpu = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512dq");
    static const bool flash_env = [] { const char* e = std::getenv("OFLM_OPEN_ATTN_FLASH"); return !(e && *e == '0'); }();
    const bool flash = flash_cpu && (g_attn_flash < 0 ? flash_env : g_attn_flash > 0) && R * grp <= 64 && g.hd % 64 == 0 && g.hd <= 512;
    // OFLM_OPEN_ATTN_BF16 (default on with AVX-512 BF16; 0 the fp32 flash kernel): bf16 dot
    // products; g_attn_flash 2 forces it, 1 the fp32 one (tests)
    static const bool bf16_cpu = __builtin_cpu_supports("avx512bf16");
    static const bool bf16_env = [] { const char* e = std::getenv("OFLM_OPEN_ATTN_BF16"); return !(e && *e == '0'); }();
    const bool bf16 = flash && bf16_cpu && (g_attn_flash < 0 ? bf16_env : g_attn_flash == 2);
    const bool sw = g.sinks || g.window;   // Flint: the bf16 / int8 flash kernels or the tiles carry them
    if (g.kv8) {
        // the int8 copy: this block's rows quantized, then the flash kernel over it
        if (!(bf16 && split && R <= 8) || g.window) throw std::runtime_error("open_qwen36: attention_block: the int8 cache needs the bf16 flash path (and no window)");
        kv8_quant_rows(kv, kv_row_elems, g.kvh, g.hd, g.pos0, g.pos0 + g.t_real, g.kv8, g.kv8_row);
        attention_decode_flash_kv8(g.hd, g.nh, g.kvh, R, g.pos0, scale, Q.data(), g.kv8, g.kv8_row, og, g.sinks);
    } else if (bf16 && split && R <= 8)
        attention_decode_flash_bf16(g.hd, g.nh, g.kvh, R, g.pos0, scale, Q.data(), kv, kv_row_elems, og, g.sinks, g.window);
    else if (sw && split && R <= 8)
        throw std::runtime_error("open_qwen36: attention_block: sinks / a window in decode need the AVX-512 BF16 flash kernel");
    else if (sw)
        attention_tiles(g.hd, g.nh, g.kvh, R, g.pos0, scale, Q.data(), kv, kv_row_elems, og, 0, g.sinks, g.window);
    else if (flash && split && R <= 8)
        attention_decode_flash(g.hd, g.nh, g.kvh, R, g.pos0, scale, Q.data(), kv, kv_row_elems, og);
    else if (split && R <= 8 && g.hd % 8 == 0 && grp <= 8)
        attention_decode_split(g.hd, g.nh, g.kvh, R, g.pos0, scale, Q.data(), kv, kv_row_elems, og);
    else
        attention_tiles(g.hd, g.nh, g.kvh, R, g.pos0, scale, Q.data(), kv, kv_row_elems, og);
    if (gate)                                       // Qwen3.6's output gate; LFM2 has none
        for (size_t i = 0; i < R * qw; ++i) og[i] *= sigmoid(gate[i]);
    (void)grp;
}

void attention_full(const AttnGeom& g, const float* Q, const uint16_t* kv, size_t kv_row_elems, size_t keys, float* og) {
    if (g.nh % g.kvh || g.t_real > g.T || keys == 0)
        throw std::runtime_error("open_qwen36: attention_full: inconsistent geometry");
    std::fill(og, og + g.T * g.nh * g.hd, 0.f);
    attention_tiles(g.hd, g.nh, g.kvh, g.t_real, g.pos0, 1.0f, Q, kv, kv_row_elems, og, keys);
}

__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void attention_prep(const AttnGeom& g, const float* q, const float* k, const float* v, const float* qn,
                    const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems, float* Q) {
    const size_t qw = g.nh * g.hd, kvw = g.kvh * g.hd, half = g.rot / 2, R = g.t_real;
    if (g.nh % g.kvh || g.t_real > g.T || g.rot > g.hd || kv_row_elems < 2 * kvw)
        throw std::runtime_error("open_qwen36: attention_prep: inconsistent geometry");
    const float scale = 1.0f / std::sqrt(static_cast<float>(g.hd));
    std::fill(Q, Q + g.T * qw, 0.f);
#pragma omp parallel
    {
    std::vector<float> kh(kvw);                 // once a thread, not once a token
#pragma omp for
    for (long long tt = 0; tt < static_cast<long long>(R); ++tt) {
        const size_t t = static_cast<size_t>(tt), p = g.pos0 + t, rp = g.rpos ? g.rpos[t] : p;
        for (size_t h = 0; h < g.nh; ++h) {
            float* dst = Q + t * qw + h * g.hd;
            rms_vec(q + t * qw + h * g.hd, g.hd, qn, g.eps, dst);
            rope(dst, half, inv_freq, static_cast<double>(rp));
            for (size_t j = 0; j < g.hd; ++j) dst[j] *= scale;
        }
        for (size_t h = 0; h < g.kvh; ++h) {
            rms_vec(k + t * kvw + h * g.hd, g.hd, kn, g.eps, kh.data() + h * g.hd);
            rope(kh.data() + h * g.hd, half, inv_freq, static_cast<double>(rp));
        }
        for (size_t j = 0; j < kvw; ++j) {
            kv[p * kv_row_elems + j] = f32_to_bf16(kh[j]);
            kv[p * kv_row_elems + kvw + j] = f32_to_bf16(v[t * kvw + j]);
        }
    }
    }
}

void tile_rows_as_bt(const uint16_t* rows, size_t stride, size_t n_real, size_t n, size_t k, uint16_t* out) {
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (k % TK || n % TN) throw std::runtime_error("open_qwen36: tile_rows_as_bt: k or n does not tile by (64, 32)");
    const size_t NB = n / TN;
#pragma omp parallel for
    for (long long kb = 0; kb < static_cast<long long>(k / TK); ++kb)
        for (size_t nb = 0; nb < NB; ++nb) {
            uint16_t* w = out + (kb * NB + nb) * TK * TN;
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t) {
                            const size_t r = nb * TN + ti * MAC + t;
                            *w++ = r < n_real ? rows[r * stride + kb * TK + si * MAC + s] : 0;
                        }
        }
}

void tile_rows_as_b(const uint16_t* rows, size_t stride, size_t k_real, size_t k, size_t n, uint16_t* out) {
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (k % TK || n % TN) throw std::runtime_error("open_qwen36: tile_rows_as_b: k or n does not tile by (64, 32)");
    const size_t NB = n / TN;
#pragma omp parallel for
    for (long long kb = 0; kb < static_cast<long long>(k / TK); ++kb)
        for (size_t nb = 0; nb < NB; ++nb) {
            uint16_t* w = out + (kb * NB + nb) * TK * TN;
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s) {
                        const size_t r = kb * TK + si * MAC + s;
                        for (size_t t = 0; t < MAC; ++t)
                            *w++ = r < k_real ? rows[r * stride + nb * TN + ti * MAC + t] : 0;
                    }
        }
}

void softmax_chunk(size_t M, size_t L, size_t hd, size_t c0, const float* s, const size_t* pos, float* m, float* l,
                   float* acc, uint16_t* p) {
#pragma omp parallel for
    for (long long rr = 0; rr < static_cast<long long>(M); ++rr) {
        const size_t r = static_cast<size_t>(rr);
        const float* sr = s + r * L;
        uint16_t* pr = p + r * L;
        const size_t valid = pos[r] >= c0 ? std::min(L, pos[r] - c0 + 1) : 0;
        if (valid == 0) {                       // the whole chunk is past this row's position
            std::fill(pr, pr + L, uint16_t{0});
            continue;
        }
        float mc = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < valid; ++j) mc = std::max(mc, sr[j]);
        const float m_new = std::max(m[r], mc);
        if (m[r] != m_new && l[r] != 0.f) {    // an earlier chunk's max is beaten: rescale what it accumulated
            const float a = std::exp(m[r] - m_new);
            l[r] *= a;
            float* ar = acc + r * hd;
            for (size_t j = 0; j < hd; ++j) ar[j] *= a;
        }
        double sum = 0;
        for (size_t j = 0; j < valid; ++j) {
            const uint16_t b = f32_to_bf16(std::exp(sr[j] - m_new));
            pr[j] = b;
            sum += bf16_to_f32(b);
        }
        std::fill(pr + valid, pr + L, uint16_t{0});
        l[r] += static_cast<float>(sum);
        m[r] = m_new;
    }
}

namespace {
// router logits for 8 tokens x E experts (E % 64 == 0) with the accumulators in registers: tiles of
// 8 tokens x 64 experts, one fused multiply-add per (token, expert) and hid step in the same order
// as the per-token loop's (the same bits: that loop's `lg[e] += xi * wr[e]` contracts to an FMA)
__attribute__((target("avx512f,fma")))
void router_cols8(size_t nt, size_t hid, size_t E, size_t e0, const float* x, const float* Wr, float* lg) {
    __m512 acc[8][4];
    for (size_t u = 0; u < 8; ++u)
        for (int v = 0; v < 4; ++v) acc[u][v] = _mm512_setzero_ps();
    for (size_t i = 0; i < hid; ++i) {
        const float* wr = Wr + i * E + e0;
        const __m512 w0 = _mm512_loadu_ps(wr), w1 = _mm512_loadu_ps(wr + 16), w2 = _mm512_loadu_ps(wr + 32), w3 = _mm512_loadu_ps(wr + 48);
        for (size_t u = 0; u < nt; ++u) {
            const __m512 xi = _mm512_set1_ps(x[u * hid + i]);
            acc[u][0] = _mm512_fmadd_ps(xi, w0, acc[u][0]);
            acc[u][1] = _mm512_fmadd_ps(xi, w1, acc[u][1]);
            acc[u][2] = _mm512_fmadd_ps(xi, w2, acc[u][2]);
            acc[u][3] = _mm512_fmadd_ps(xi, w3, acc[u][3]);
        }
    }
    for (size_t u = 0; u < nt; ++u)
        for (int v = 0; v < 4; ++v) _mm512_storeu_ps(lg + u * E + e0 + 16 * v, acc[u][v]);
}

// prompt rows: 6 tokens x 64 experts of accumulators in registers over hid steps [i0, i1), the
// partial sums kept in lg between steps (the same FMA chain per (token, expert), so the same bits).
// xT holds the 6 tokens transposed, [hid][6]; 256 hid steps of a 64-expert slice (64 KB) stay in L2.
__attribute__((target("avx512f,fma")))
void router_tile6(size_t i0, size_t i1, size_t E, size_t e0, const float* xT, const float* Wr, float* lg) {
    __m512 acc[6][4];
#pragma GCC unroll 6
    for (int u = 0; u < 6; ++u)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) acc[u][v] = i0 ? _mm512_loadu_ps(lg + u * E + e0 + 16 * v) : _mm512_setzero_ps();
    for (size_t i = i0; i < i1; ++i) {
        const float* wr = Wr + i * E + e0;
        const __m512 w0 = _mm512_loadu_ps(wr), w1 = _mm512_loadu_ps(wr + 16), w2 = _mm512_loadu_ps(wr + 32), w3 = _mm512_loadu_ps(wr + 48);
        const float* xi = xT + i * 6;
#pragma GCC unroll 6
        for (int u = 0; u < 6; ++u) {
            const __m512 b = _mm512_set1_ps(xi[u]);
            acc[u][0] = _mm512_fmadd_ps(b, w0, acc[u][0]);
            acc[u][1] = _mm512_fmadd_ps(b, w1, acc[u][1]);
            acc[u][2] = _mm512_fmadd_ps(b, w2, acc[u][2]);
            acc[u][3] = _mm512_fmadd_ps(b, w3, acc[u][3]);
        }
    }
#pragma GCC unroll 6
    for (int u = 0; u < 6; ++u)
#pragma GCC unroll 4
        for (int v = 0; v < 4; ++v) _mm512_storeu_ps(lg + u * E + e0 + 16 * v, acc[u][v]);
}

// router logits for nt prompt rows (E % 64 == 0) into lg [nt rounded up to 6][E]
void router_logits_rows(size_t nt, size_t hid, size_t E, const float* x, const float* Wr, float* lg) {
    const size_t ns = (nt + 5) / 6;
    std::vector<float> xT(ns * hid * 6, 0.f);
    for (size_t s = 0; s < ns; ++s)
        for (size_t u = 0; u < 6 && s * 6 + u < nt; ++u)
            for (size_t i = 0; i < hid; ++i) xT[(s * hid + i) * 6 + u] = x[(s * 6 + u) * hid + i];
    for (size_t e0 = 0; e0 < E; e0 += 64)
        for (size_t i0 = 0; i0 < hid; i0 += 256)
            for (size_t s = 0; s < ns; ++s)
                router_tile6(i0, std::min(hid, i0 + 256), E, e0, xT.data() + s * hid * 6, Wr, lg + s * 6 * E);
}
}  // namespace

void router_block(size_t T, size_t hid, size_t E, size_t topk, const float* xm, const float* Wr, float* probs,
                  int32_t* idx, float* w, bool sigmoid) {
    if (topk > E) throw std::runtime_error("open_qwen36: router_block: topk past the expert count");
    static const bool fast = __builtin_cpu_supports("avx512f") && !(std::getenv("OFLM_OPEN_ROUTER_FAST") &&
                                                                     *std::getenv("OFLM_OPEN_ROUTER_FAST") == '0');
    if (fast && E % 64 == 0 && T < 8 && omp_get_max_threads() > 1) {
        // a decode step's few rows: the experts split over the threads (64 a work item), each
        // reading its columns of Wr once for all rows -- one thread reading all 2 MB was 0.25 ms a layer
        std::vector<float> lgb(8 * E);
        const long long ne = static_cast<long long>(E / 64);
#pragma omp parallel for schedule(static)
        for (long long ec = 0; ec < ne; ++ec) router_cols8(T, hid, E, static_cast<size_t>(ec) * 64, xm, Wr, lgb.data());
        std::vector<char> taken(E);
        for (size_t u = 0; u < T; ++u) router_select(E, topk, lgb.data() + u * E, taken.data(), probs + u * E, idx + u * topk, w + u * topk, sigmoid);
        return;
    }
    if (fast && E % 64 == 0) {
        // a block of rows a thread (a multiple of 6, ~T / threads): the tiles' 64-expert x 256-step
        // slices of Wr stay in L2 for the block's rows (1.7x the 8-token passes over all of Wr)
        const size_t nth = static_cast<size_t>(omp_get_max_threads());
        const size_t tb = std::min<size_t>(192, std::max<size_t>(6, ((T + nth - 1) / nth + 5) / 6 * 6));
        const long long nb = static_cast<long long>((T + tb - 1) / tb);
#pragma omp parallel for schedule(dynamic, 1)
        for (long long b = 0; b < nb; ++b) {
            const size_t t0 = static_cast<size_t>(b) * tb, nt = std::min(tb, T - t0);
            std::vector<float> lgb((nt + 5) / 6 * 6 * E);
            router_logits_rows(nt, hid, E, xm + t0 * hid, Wr, lgb.data());
            std::vector<char> taken(E);
            for (size_t u = 0; u < nt; ++u)
                router_select(E, topk, lgb.data() + u * E, taken.data(), probs + (t0 + u) * E, idx + (t0 + u) * topk, w + (t0 + u) * topk, sigmoid);
        }
        return;
    }
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        std::vector<float> lg(E, 0.f);
        std::vector<char> taken(E, 0);
        const float* x = xm + t * hid;
        for (size_t i = 0; i < hid; ++i) {
            const float xi = x[i];
            const float* wr = Wr + i * E;
            for (size_t e = 0; e < E; ++e) lg[e] += xi * wr[e];
        }
        router_select(E, topk, lg.data(), taken.data(), probs + t * E, idx + t * topk, w + t * topk, sigmoid);
    }
}

// one token's softmax, top-k and normalised weights from its logits `lg` (overwritten with the
// exponentials). The top-k in one pass, kept sorted by (probability down, index up): the same picks
// as k scans for the first largest untaken one, 4.5x faster. `taken` is unused (kept for callers).
void router_select(size_t E, size_t topk, float* lg, char* taken, float* probs, int32_t* idx, float* w, bool sigmoid) {
    (void)taken;
    if (topk > 64) throw std::runtime_error("open_qwen36: router_select: topk past 64");
    if (sigmoid) {
        // Flint: the top-k of sigmoid(logit) (the same order as the logits; ties to the lower index),
        // weights renormalised to sum 1 (SGLang's sigmoid_routing: / (sum + 1e-20))
        float tv[64];
        int32_t ti[64];
        size_t n = 0;
        for (size_t e = 0; e < E; ++e) {
            const float v = 1.f / (1.f + std::exp(-lg[e]));
            probs[e] = v;
            if (n == topk && !(v > tv[topk - 1])) continue;
            size_t j = n < topk ? n++ : topk - 1;
            for (; j > 0 && v > tv[j - 1]; --j) {
                tv[j] = tv[j - 1];
                ti[j] = ti[j - 1];
            }
            tv[j] = v;
            ti[j] = static_cast<int32_t>(e);
        }
        float wsum = 0.f;
        for (size_t s2 = 0; s2 < topk; ++s2) wsum += tv[s2];
        for (size_t s2 = 0; s2 < topk; ++s2) {
            idx[s2] = ti[s2];
            w[s2] = tv[s2] / (wsum + 1e-20f);
        }
        return;
    }
    float mx = -1e30f;
    for (size_t e = 0; e < E; ++e) mx = std::max(mx, lg[e]);
    double denom = 0;
    for (size_t e = 0; e < E; ++e) {
        lg[e] = std::exp(lg[e] - mx);
        denom += lg[e];
    }
    const float inv = static_cast<float>(1.0 / denom);
    float tv[64];
    int32_t ti[64];
    size_t n = 0;
    for (size_t e = 0; e < E; ++e) {
        const float v = lg[e] * inv;
        probs[e] = v;
        if (n == topk && !(v > tv[topk - 1])) continue;
        size_t j = n < topk ? n++ : topk - 1;
        for (; j > 0 && v > tv[j - 1]; --j) {
            tv[j] = tv[j - 1];
            ti[j] = ti[j - 1];
        }
        tv[j] = v;
        ti[j] = static_cast<int32_t>(e);
    }
    double wsum = 0;
    for (size_t s = 0; s < topk; ++s) {
        idx[s] = ti[s];
        w[s] = tv[s];
        wsum += tv[s];
    }
    for (size_t s = 0; s < topk; ++s) w[s] = static_cast<float>(w[s] / wsum);
}

// ---- gemv_b4 -------------------------------------------------------------------------------
namespace {
constexpr size_t kB4Tile = 4224;
inline uint16_t bf16_rne(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return static_cast<uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
// x [K] -> [K / 8][16 dwords]: lane j holds the bf16 pair (x[8 kb + 2 (j % 4)], the next one)
void b4_prep_x(const float* x, size_t K, uint32_t* xp) {
    for (size_t kb = 0; kb < K / 8; ++kb)
        for (int j = 0; j < 16; ++j) {
            const size_t k = kb * 8 + 2 * (j % 4);
            xp[kb * 16 + j] = uint32_t(bf16_rne(x[k])) | (uint32_t(bf16_rne(x[k + 1])) << 16);
        }
}
// strip s, K tiles [t0, t1): acc [128] += the strip's columns over those tiles. Nibbles become
// bf16 through a 16-word table (vpermw); vdpbf16ps takes 32 products a step, 4 lanes a column.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512bf16")))
void b4_strip_avx512(const uint8_t* st, const uint32_t* xp, size_t t0, size_t t1, float* acc) {
    alignas(64) uint16_t t16[32];
    for (int i = 0; i < 32; ++i) {
        const int v = (i & 15) < 8 ? (i & 15) : (i & 15) - 16;
        t16[i] = bf16_rne(static_cast<float>(v));
    }
    const __m512i tab = _mm512_load_si512(t16), m4 = _mm512_set1_epi32(0xF);
    const __m512i rep = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    const __m512i rep2 = _mm512_add_epi32(rep, _mm512_set1_epi32(4)), six = _mm512_set1_epi32(6);
    alignas(64) float run[16][2][16];
    std::memset(run, 0, sizeof run);
    for (size_t kt = t0; kt < t1; ++kt) {
        const uint8_t* t = st + kt * kB4Tile;
        _mm_prefetch(reinterpret_cast<const char*>(t + 8 * kB4Tile), _MM_HINT_NTA);
        const uint32_t* xk = xp + kt * 8 * 16;
        for (int g = 0; g < 16; ++g) {
            const int p = g / 2, n2 = g % 2;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            for (int kb = 0; kb < 8; ++kb) {
                const uint8_t* blk = t + ((p * 8 + kb) * 2 + n2) * 32;
                const __m512bh xv = (__m512bh)_mm512_loadu_si512(xk + kb * 16);
                __m512i d0 = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk)));
                __m512i d1 = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(blk + 16)));
                d0 = _mm512_or_si512(_mm512_and_si512(d0, m4), _mm512_slli_epi32(_mm512_srli_epi32(d0, 4), 16));
                d1 = _mm512_or_si512(_mm512_and_si512(d1, m4), _mm512_slli_epi32(_mm512_srli_epi32(d1, 4), 16));
                a0 = _mm512_dpbf16_ps(a0, (__m512bh)_mm512_permutexvar_epi16(d0, tab), xv);
                a1 = _mm512_dpbf16_ps(a1, (__m512bh)_mm512_permutexvar_epi16(d1, tab), xv);
            }
            // 2^(e - 133) as float bits; 0 for e <= 6 (an all-zero column)
            const __m512i e = _mm512_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(t + 4096 + g * 8)));
            const __m512i eb = _mm512_slli_epi32(_mm512_max_epi32(_mm512_sub_epi32(e, six), _mm512_setzero_si512()), 23);
            _mm512_store_ps(run[g][0], _mm512_fmadd_ps(a0, _mm512_castsi512_ps(_mm512_permutexvar_epi32(rep, eb)),
                                                       _mm512_load_ps(run[g][0])));
            _mm512_store_ps(run[g][1], _mm512_fmadd_ps(a1, _mm512_castsi512_ps(_mm512_permutexvar_epi32(rep2, eb)),
                                                       _mm512_load_ps(run[g][1])));
        }
    }
    for (int g = 0; g < 16; ++g)
        for (int hh = 0; hh < 2; ++hh)
            for (int c = 0; c < 4; ++c) {
                const float* r = run[g][hh] + 4 * c;
                acc[g * 8 + hh * 4 + c] += (r[0] + r[1]) + (r[2] + r[3]);
            }
}
// the same, scalar (no AVX-512 BF16): bf16 x times the decoded mantissa, the same order of sums
void b4_strip_scalar(const uint8_t* st, const uint32_t* xp, size_t t0, size_t t1, float* acc) {
    for (size_t kt = t0; kt < t1; ++kt) {
        const uint8_t* t = st + kt * kB4Tile;
        for (int c = 0; c < 128; ++c) {
            const int g = c / 8, p = g / 2, n2 = g % 2, cb = c % 8;
            float s = 0.f;
            for (int kb = 0; kb < 8; ++kb)
                for (int ki = 0; ki < 8; ++ki) {
                    const uint8_t byte = t[((p * 8 + kb) * 2 + n2) * 32 + cb * 4 + ki / 2];
                    int m = (ki & 1) ? byte >> 4 : byte & 15;
                    if (m > 7) m -= 16;
                    const uint32_t pr = xp[(kt * 8 + kb) * 16 + ki / 2];
                    s += bf16_to_f32(static_cast<uint16_t>((ki & 1) ? pr >> 16 : pr & 0xFFFF)) * static_cast<float>(m);
                }
            const int e = t[4096 + c];
            acc[c] += e > 6 ? s * std::ldexp(1.f, e - 133) : 0.f;
        }
    }
}
// The int8 path (OFLM_OPEN_DECODE_GEMV=int8, the default): the weight nibble as u8 (m + 8 =
// nib ^ 8), x as s8 with one scale per 64-K tile (max|x| / 127), vpdpbusd: 64 products a step,
// about 3x fewer instructions a weight than the bf16 path -- 8 cores read ~100 GB/s against 64
// (34816 x 5120). sum (m + 8) x = sum m x + 8 sum x: the second term is one integer a tile. A
// 64-byte load is a pair's two blocks = 16 columns x 8 k; low nibbles hold the even k, high the odd.
// Error from x's rounding: ~0.6 % relative per output on random data (the NPU prompt path's
// bfp16 A rounds alike).
struct B4XQ {
    std::vector<int32_t> ev, od, cs;
    std::vector<float> sx;
};
void b4_xq_size(size_t K, B4XQ& q) {
    q.ev.resize(K / 8); q.od.resize(K / 8); q.sx.resize(K / 64); q.cs.resize(K / 64);
}
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void b4_prep_xq_tiles(const float* x, size_t kt0, size_t kt1, B4XQ& q) {
    for (size_t kt = kt0; kt < kt1; ++kt) {
        float m = 0.f;
        for (int i = 0; i < 64; ++i) m = std::max(m, std::fabs(x[kt * 64 + i]));
        const float sc = m > 0.f ? m / 127.f : 1.f, inv = 1.f / sc;
        int32_t sum = 0;
        int8_t v[64];
        for (int i = 0; i < 64; ++i) {
            v[i] = static_cast<int8_t>(std::lrint(x[kt * 64 + i] * inv));
            sum += v[i];
        }
        q.sx[kt] = sc;
        q.cs[kt] = 8 * sum;
        for (int kb = 0; kb < 8; ++kb) {
            const uint8_t e[4] = {uint8_t(v[kb * 8]), uint8_t(v[kb * 8 + 2]), uint8_t(v[kb * 8 + 4]), uint8_t(v[kb * 8 + 6])};
            const uint8_t o[4] = {uint8_t(v[kb * 8 + 1]), uint8_t(v[kb * 8 + 3]), uint8_t(v[kb * 8 + 5]), uint8_t(v[kb * 8 + 7])};
            std::memcpy(&q.ev[kt * 8 + kb], e, 4);
            std::memcpy(&q.od[kt * 8 + kb], o, 4);
        }
    }
}
// The int4 / q8 weight kernels prefetch every line of the next block themselves (NTA, 8 lines
// a step): one hint a 4 KB tile left the rest to the hardware prefetcher, which falls behind
// once there is compute between the loads (8 cores, 1 GB of int4 tiles: 4 rows 11.8 -> 9.1 ms,
// 6 rows 16.8 -> 10.0).
inline void pf8(const uint8_t* p) {
    for (int l = 0; l < 8; ++l) _mm_prefetch(reinterpret_cast<const char*>(p + l * 64), _MM_HINT_NTA);
}
__attribute__((target("avx512f,avx512bw,avx512vnni")))
void b4_strip_vnni(const uint8_t* st, const B4XQ& q, size_t t0, size_t t1, float* acc) {
    const __m512i x88 = _mm512_set1_epi8(static_cast<char>(0x88)), m0f = _mm512_set1_epi8(0x0F), six = _mm512_set1_epi32(6);
    __m512 run[8];
    for (int p = 0; p < 8; ++p) run[p] = _mm512_setzero_ps();
    for (size_t kt = t0; kt < t1; ++kt) {
        const uint8_t* t = st + kt * kB4Tile;
        const __m512 corr = _mm512_set1_ps(static_cast<float>(q.cs[kt])), sx = _mm512_set1_ps(q.sx[kt]);
        for (int p = 0; p < 8; ++p) {
            pf8(t + kB4Tile + p * 512);
            if (p == 0) pf8(t + kB4Tile + 4096 - 6 * 64);   // the exponents' 2 lines (and 6 already asked)
            __m512i a = _mm512_setzero_si512();
            for (int kb = 0; kb < 8; ++kb) {
                const __m512i v = _mm512_xor_si512(_mm512_loadu_si512(t + (p * 8 + kb) * 64), x88);
                a = _mm512_dpbusd_epi32(a, _mm512_and_si512(v, m0f), _mm512_set1_epi32(q.ev[kt * 8 + kb]));
                a = _mm512_dpbusd_epi32(a, _mm512_and_si512(_mm512_srli_epi16(v, 4), m0f), _mm512_set1_epi32(q.od[kt * 8 + kb]));
            }
            // lane l of the pair's load is the tile's column 16 p + l; its scale 2^(e - 133) x sx
            const __m512i e = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(t + 4096 + p * 16)));
            const __m512 sc = _mm512_mul_ps(
                sx, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_max_epi32(_mm512_sub_epi32(e, six), _mm512_setzero_si512()), 23)));
            run[p] = _mm512_fmadd_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(a), corr), sc, run[p]);
        }
    }
    for (int p = 0; p < 8; ++p) _mm512_storeu_ps(acc + p * 16, _mm512_add_ps(_mm512_loadu_ps(acc + p * 16), run[p]));
}
}  // namespace

namespace {
// R rows of the int8 path: the pair's 16 columns x 8 k unpacked once (xor, and, shift) per k block,
// then 2 vpdpbusd a row; run[r][p] in L1
template <int R>
__attribute__((target("avx512f,avx512bw,avx512vnni")))
void b4_strip_vnni_rows(const uint8_t* st, const B4XQ* q, size_t t0, size_t t1, float* const* acc) {
    const __m512i x88 = _mm512_set1_epi8(static_cast<char>(0x88)), m0f = _mm512_set1_epi8(0x0F), six = _mm512_set1_epi32(6);
    alignas(64) float run[R][8][16];
    std::memset(run, 0, sizeof run);
    for (size_t kt = t0; kt < t1; ++kt) {
        const uint8_t* t = st + kt * kB4Tile;
        for (int p = 0; p < 8; ++p) {
            pf8(t + kB4Tile + p * 512);
            if (p == 0) pf8(t + kB4Tile + 4096 - 6 * 64);
            __m512i a[R];
            for (int r = 0; r < R; ++r) a[r] = _mm512_setzero_si512();
            for (int kb = 0; kb < 8; ++kb) {
                const __m512i v = _mm512_xor_si512(_mm512_loadu_si512(t + (p * 8 + kb) * 64), x88);
                const __m512i lo = _mm512_and_si512(v, m0f), hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), m0f);
                for (int r = 0; r < R; ++r) {
                    a[r] = _mm512_dpbusd_epi32(a[r], lo, _mm512_set1_epi32(q[r].ev[kt * 8 + kb]));
                    a[r] = _mm512_dpbusd_epi32(a[r], hi, _mm512_set1_epi32(q[r].od[kt * 8 + kb]));
                }
            }
            const __m512i e = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(t + 4096 + p * 16)));
            const __m512 pe = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_max_epi32(_mm512_sub_epi32(e, six), _mm512_setzero_si512()), 23));
            for (int r = 0; r < R; ++r) {
                const __m512 sc = _mm512_mul_ps(_mm512_set1_ps(q[r].sx[kt]), pe);
                _mm512_store_ps(run[r][p], _mm512_fmadd_ps(_mm512_sub_ps(_mm512_cvtepi32_ps(a[r]), _mm512_set1_ps(static_cast<float>(q[r].cs[kt]))),
                                                           sc, _mm512_load_ps(run[r][p])));
            }
        }
    }
    for (int r = 0; r < R; ++r)
        for (int p = 0; p < 8; ++p) _mm512_storeu_ps(acc[r] + p * 16, _mm512_add_ps(_mm512_loadu_ps(acc[r] + p * 16), _mm512_load_ps(run[r][p])));
}
using RowsFn = void (*)(const uint8_t*, const B4XQ*, size_t, size_t, float* const*);
constexpr RowsFn kRowsFn[8] = {b4_strip_vnni_rows<1>, b4_strip_vnni_rows<2>, b4_strip_vnni_rows<3>, b4_strip_vnni_rows<4>,
                               b4_strip_vnni_rows<5>, b4_strip_vnni_rows<6>, b4_strip_vnni_rows<7>, b4_strip_vnni_rows<8>};

// the strips (128 columns each, tile bytes `tb` a 64-K tile) of R rows over threads: x rounded to
// int8 per 64-K tile, the work split along K too when there are fewer strips than ~4 a thread
template <class Fn>
void rows_driver(const uint8_t* B, size_t tb, size_t npad, size_t K, const float* x, size_t R, float* y, size_t ldy, size_t kt0,
                 size_t kt1, Fn&& fn) {
    static thread_local std::vector<B4XQ> xq_tl;
    std::vector<B4XQ>& xq = xq_tl;
    xq.resize(8);
    for (size_t r = 0; r < R; ++r) b4_xq_size(K, xq[r]);
    const size_t S = npad / 128, KT = K / 64, T = static_cast<size_t>(omp_get_max_threads());
    if (!kt1) kt1 = KT;
    const size_t KR = kt1 - kt0;
    const size_t ks = std::min(KR, std::max<size_t>(1, (4 * T + S - 1) / S));
    static thread_local std::vector<float> part_tl;
    std::vector<float>& part = part_tl;
    if (ks > 1) part.resize(ks * R * npad);
#pragma omp parallel
    {
#pragma omp for schedule(static) collapse(2)
        for (long long r = 0; r < static_cast<long long>(R); ++r)
            for (long long kt = static_cast<long long>(kt0); kt < static_cast<long long>(kt1); ++kt)
                b4_prep_xq_tiles(x + r * K, static_cast<size_t>(kt), static_cast<size_t>(kt) + 1, xq[static_cast<size_t>(r)]);
#pragma omp for schedule(dynamic, 1)
        for (long long w = 0; w < static_cast<long long>(S * ks); ++w) {
            const size_t s = static_cast<size_t>(w) / ks, rr = static_cast<size_t>(w) % ks;
            const size_t t0 = kt0 + KR * rr / ks, t1 = kt0 + KR * (rr + 1) / ks;
            float* acc[8];
            for (size_t r = 0; r < R; ++r) {
                acc[r] = ks == 1 ? y + r * ldy + s * 128 : part.data() + (rr * R + r) * npad + s * 128;
                std::fill(acc[r], acc[r] + 128, 0.f);
            }
            fn(B + s * KT * tb, xq.data(), t0, t1, acc);
        }
        if (ks > 1) {
#pragma omp for schedule(static) collapse(2)
            for (long long r = 0; r < static_cast<long long>(R); ++r)
                for (long long n = 0; n < static_cast<long long>(npad); ++n) {
                    float v = 0.f;
                    for (size_t rr = 0; rr < ks; ++rr) v += part[(rr * R + static_cast<size_t>(r)) * npad + static_cast<size_t>(n)];
                    y[r * ldy + n] = v;
                }
        }
    }
}
}  // namespace

void gemv_b4_rows(const uint8_t* B, size_t npad, size_t K, const float* x, size_t R, float* y, size_t ldy, size_t kt0, size_t kt1) {
    if (R == 0) return;
    if (R > 8) throw std::runtime_error("open_qwen36: gemv_b4_rows: at most 8 rows");
    static const bool int8 = [] {
        const char* e = std::getenv("OFLM_OPEN_DECODE_GEMV");
        return !(e && std::string(e) == "bf16") && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw");
    }();
    if (!int8 || R == 1) {
        for (size_t r = 0; r < R; ++r) gemv_b4(B, npad, K, x + r * K, y + r * ldy, nullptr, kt0, kt1);
        return;
    }
    if (npad % 128 || K % 64) throw std::runtime_error("open_qwen36: gemv_b4_rows: N % 128 or K % 64");
    rows_driver(B, kB4Tile, npad, K, x, R, y, ldy, kt0, kt1, kRowsFn[R - 1]);
}

void gemv_b4(const uint8_t* B, size_t npad, size_t K, const float* x, float* y, const std::function<void()>& side, size_t kt0,
             size_t kt1) {
    if (npad % 128 || K % 64) throw std::runtime_error("open_qwen36: gemv_b4: N % 128 or K % 64");
    static const bool avx = __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("avx512bw");
    static const bool int8 = [] {
        const char* e = std::getenv("OFLM_OPEN_DECODE_GEMV");
        return !(e && std::string(e) == "bf16") && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512bw");
    }();
    static thread_local B4XQ xq_tl;
    static thread_local std::vector<uint32_t> xp_tl;
    static thread_local std::vector<float> part_tl;
    std::vector<uint32_t>& xp = xp_tl;
    B4XQ& xq = xq_tl;
    if (int8) b4_xq_size(K, xq);
    else xp.resize(K / 8 * 16);
    const size_t S = npad / 128, KT = K / 64;
    if (!kt1) kt1 = KT;
    const size_t KR = kt1 - kt0;
    const size_t T = static_cast<size_t>(omp_get_max_threads());
    // enough work items to balance the threads: K split in ranges when there are few strips
    const size_t ks = std::min(KR, std::max<size_t>(1, (4 * T + S - 1) / S));
    std::vector<float>& part = part_tl;
    if (ks > 1) part.resize(ks * npad);
    const uint32_t* xpp = xp.data();
#pragma omp parallel
    {
    // x rounded to int8 over the threads, one 64-K tile each (bf16 path: one thread)
    if (int8) {
#pragma omp for schedule(static)
        for (long long kt = static_cast<long long>(kt0); kt < static_cast<long long>(kt1); ++kt)
            b4_prep_xq_tiles(x, static_cast<size_t>(kt), static_cast<size_t>(kt) + 1, xq);
    } else {
#pragma omp single
        b4_prep_x(x, K, xp.data());
    }
    // the side task (the NPU share's pack / flush / start) on one thread while the others start
    // on the strips; it joins them when done
    if (side) {
#pragma omp single nowait
        side();
    }
#pragma omp for schedule(dynamic, 1)
    for (long long w = 0; w < static_cast<long long>(S * ks); ++w) {
        const size_t s = static_cast<size_t>(w) / ks, r = static_cast<size_t>(w) % ks;
        const size_t t0 = kt0 + KR * r / ks, t1 = kt0 + KR * (r + 1) / ks;
        // one K range: straight into y; several: a partial row each, added below
        float* acc = ks == 1 ? y + s * 128 : part.data() + r * npad + s * 128;
        std::fill(acc, acc + 128, 0.f);
        if (int8) b4_strip_vnni(B + s * KT * kB4Tile, xq, t0, t1, acc);
        else if (avx) b4_strip_avx512(B + s * KT * kB4Tile, xpp, t0, t1, acc);
        else b4_strip_scalar(B + s * KT * kB4Tile, xpp, t0, t1, acc);
    }
    if (ks > 1) {
#pragma omp for schedule(static)
        for (long long n = 0; n < static_cast<long long>(npad); ++n) {
            float v = 0.f;
            for (size_t r = 0; r < ks; ++r) v += part[r * npad + static_cast<size_t>(n)];
            y[n] = v;
        }
    }
    }
}

size_t lmhead_q8_cpu_bytes(size_t N, size_t K) { return N / 16 * (K / 32) * 544; }

void lmhead_q8_repack(const uint8_t* pool, size_t N, size_t K, uint8_t* out) {
    if (N % 128 || K % 256) throw std::runtime_error("open_qwen36: lmhead_q8_repack: N % 128 or K % 256");
    const size_t nk = K / 256, per_band = 4 * nk, GS = K / 32 * 544;
#pragma omp parallel for schedule(static)
    for (long long cc = 0; cc < static_cast<long long>(N / 128 * per_band); ++cc) {
        const size_t c = static_cast<size_t>(cc), b = c / per_band, q = c % per_band % 4, kt = c % per_band / 4;
        const uint8_t* ch = pool + c * 8704;
        for (size_t r = 0; r < 32; ++r) {
            const size_t g = b * 8 + q * 2 + r / 16, lane = r % 16;
            uint8_t* gp = out + g * GS;
            for (size_t kb = 0; kb < 8; ++kb) {
                uint8_t* blk = gp + (kt * 8 + kb) * 544;
                std::memcpy(blk + lane * 2, ch + (kb * 32 + r) * 2, 2);
                for (size_t k = kb * 32; k < kb * 32 + 32; ++k)
                    blk[32 + ((k % 32) / 4) * 64 + lane * 4 + k % 4] = ch[512 + (r / 16) * 4096 + k * 16 + r % 16];
            }
        }
    }
}

namespace {
__attribute__((target("avx512f,avx512bw,avx512vnni")))
void lmhead_groups(const uint8_t* W, size_t K, const uint32_t* ax, const uint64_t* neg, const float* sx, size_t g0, size_t g1,
                   float* y) {
    const size_t KB = K / 32, GS = KB * 544;
    const __m512i zero = _mm512_setzero_si512();
    for (size_t g = g0; g < g1; ++g) {
        const uint8_t* gp = W + g * GS;
        __m512 acc = _mm512_setzero_ps();
        for (size_t B = 0; B < KB; ++B) {
            const uint8_t* blk = gp + B * 544;
            pf8(blk + 4 * 544);                              // 4 blocks (2 KB) ahead, into the next group
            if (B % 8 == 0) pf8(blk + 4 * 544 + 512);        // 544-byte blocks: 8 lines a block plus 1 a 8
            __m512i a = _mm512_setzero_si512();
            for (int kq = 0; kq < 8; ++kq) {
                __m512i w = _mm512_loadu_si512(blk + 32 + kq * 64);
                w = _mm512_mask_sub_epi8(w, neg[B * 8 + kq], zero, w);
                a = _mm512_dpbusd_epi32(a, _mm512_set1_epi32(static_cast<int>(ax[B * 8 + kq])), w);
            }
            const __m512 sc = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk))), 16));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a), _mm512_mul_ps(sc, _mm512_set1_ps(sx[B])), acc);
        }
        _mm512_storeu_ps(y + g * 16, acc);
    }
}
}  // namespace

namespace {
struct HeadX { std::vector<uint32_t> ax; std::vector<uint64_t> neg; std::vector<float> sx; };
__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void head_prep(const float* x, size_t K, HeadX& h) {
    const size_t KB = K / 32;
    h.ax.resize(KB * 8); h.neg.resize(KB * 8); h.sx.resize(KB);
    for (size_t B = 0; B < KB; ++B) {
        float m = 0.f;
        for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(x[B * 32 + i]));
        const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
        h.sx[B] = s;
        for (int kq = 0; kq < 8; ++kq) {
            uint32_t a = 0, pat = 0;
            for (int j = 0; j < 4; ++j) {
                const long v = std::lrint(x[B * 32 + kq * 4 + j] * inv);
                a |= static_cast<uint32_t>(std::labs(v)) << (8 * j);
                if (v < 0) pat |= 1u << j;
            }
            uint64_t mk = 0;
            for (int lane = 0; lane < 16; ++lane) mk |= static_cast<uint64_t>(pat) << (4 * lane);
            h.ax[B * 8 + kq] = a;
            h.neg[B * 8 + kq] = mk;
        }
    }
}
template <int R>
__attribute__((target("avx512f,avx512bw,avx512vnni")))
void lmhead_groups_rows(const uint8_t* W, size_t K, const HeadX* hx, size_t g0, size_t g1, float* y, size_t N) {
    const size_t KB = K / 32, GS = KB * 544;
    const __m512i zero = _mm512_setzero_si512();
    for (size_t g = g0; g < g1; ++g) {
        const uint8_t* gp = W + g * GS;
        __m512 acc[R];
        for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
        for (size_t B = 0; B < KB; ++B) {
            const uint8_t* blk = gp + B * 544;
            pf8(blk + 4 * 544);
            if (B % 8 == 0) pf8(blk + 4 * 544 + 512);
            __m512i a[R];
            for (int r = 0; r < R; ++r) a[r] = _mm512_setzero_si512();
            for (int kq = 0; kq < 8; ++kq) {
                const __m512i w = _mm512_loadu_si512(blk + 32 + kq * 64);
                for (int r = 0; r < R; ++r)
                    a[r] = _mm512_dpbusd_epi32(a[r], _mm512_set1_epi32(static_cast<int>(hx[r].ax[B * 8 + kq])),
                                               _mm512_mask_sub_epi8(w, hx[r].neg[B * 8 + kq], zero, w));
            }
            const __m512 sc = _mm512_castsi512_ps(
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(blk))), 16));
            for (int r = 0; r < R; ++r)
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[r]), _mm512_mul_ps(sc, _mm512_set1_ps(hx[r].sx[B])), acc[r]);
        }
        for (int r = 0; r < R; ++r) _mm512_storeu_ps(y + r * N + g * 16, acc[r]);
    }
}
using HeadFn = void (*)(const uint8_t*, size_t, const HeadX*, size_t, size_t, float*, size_t);
constexpr HeadFn kHeadFn[8] = {lmhead_groups_rows<1>, lmhead_groups_rows<2>, lmhead_groups_rows<3>, lmhead_groups_rows<4>,
                               lmhead_groups_rows<5>, lmhead_groups_rows<6>, lmhead_groups_rows<7>, lmhead_groups_rows<8>};
}  // namespace

void lmhead_q8_cpu_rows(const uint8_t* W, size_t N, size_t K, const float* x, size_t R, float* y) {
    if (R < 1 || R > 8) throw std::runtime_error("open_qwen36: lmhead_q8_cpu_rows: 1 to 8 rows");
    static thread_local std::vector<HeadX> hx;
    hx.resize(8);
    for (size_t r = 0; r < R; ++r) head_prep(x + r * K, K, hx[r]);
    const size_t G = N / 16;
    const HeadFn fn = kHeadFn[R - 1];
    const HeadX* hp = hx.data();
#pragma omp parallel
    {
        const size_t T = static_cast<size_t>(omp_get_num_threads()), i = static_cast<size_t>(omp_get_thread_num());
        fn(W, K, hp, G * i / T, G * (i + 1) / T, y, N);
    }
}

HeadTwoStats g_head2{};

void lmhead_two_stage_rows(const uint8_t* W8, const uint8_t* W4, size_t N, size_t K, const float* x, size_t R, float* y,
                           size_t NR, size_t topk, bool check) {
    if (R < 1 || R > 8) throw std::runtime_error("open_qwen36: lmhead_two_stage_rows: 1 to 8 rows");
    // 1. int4 logits for every row (one read of the int4 head)
    gemv_b4_rows(W4, N, K, x, R, y, N);
    std::vector<float> ref;
    std::vector<std::vector<float>> i4;
    if (check) {
        ref.resize(R * N);
        lmhead_q8_cpu_rows(W8, N, K, x, R, ref.data());
        for (size_t r = 0; r < R; ++r) i4.emplace_back(y + r * N, y + r * N + NR);
    }
    static thread_local std::vector<HeadX> hx;
    hx.resize(8);
    for (size_t r = 0; r < R; ++r) head_prep(x + r * K, K, hx[r]);
    const HeadFn fn = kHeadFn[R - 1];
    // 2. each row's top-k (int4) -> their 16-row q8 groups, the union over the rows
    std::vector<uint8_t> done(N / 16, 0);
    std::vector<size_t> groups;
    std::vector<float> best(R, -INFINITY);
    for (size_t r = 0; r < R; ++r) {
        const float* yr = y + r * N;
        std::vector<std::pair<float, uint32_t>> top;
        top.reserve(topk + 1);
        float floor_v = -INFINITY;
        for (size_t v = 0; v < NR; ++v) {
            if (yr[v] <= floor_v && top.size() >= topk) continue;
            top.emplace_back(yr[v], static_cast<uint32_t>(v));
            std::push_heap(top.begin(), top.end(), std::greater<>());
            if (top.size() > topk) { std::pop_heap(top.begin(), top.end(), std::greater<>()); top.pop_back(); }
            if (top.size() >= topk) floor_v = top.front().first;
        }
        for (const auto& [val, v] : top)
            if (!done[v / 16]) { done[v / 16] = 1; groups.push_back(v / 16); }
    }
    // 3. the q8 values of those groups for every row (the full head's own kernel: bit-identical
    // values), then 4. any token whose int4 logit still reaches a row's best q8 one rescored too
    static thread_local std::vector<float> tmp;
    tmp.resize(R * N);
    // plain pointers: inside the parallel loop the thread_local names would be each worker's own
    const HeadX* hxp = hx.data();
    float* tp = tmp.data();
    auto rescore = [&](const std::vector<size_t>& gs) {
        const size_t* gp = gs.data();
#pragma omp parallel for schedule(dynamic, 4)
        for (long long i = 0; i < static_cast<long long>(gs.size()); ++i) fn(W8, K, hxp, gp[i], gp[i] + 1, tp, N);
        for (size_t g : gs)
            for (size_t r = 0; r < R; ++r)
                for (size_t j = 0; j < 16; ++j) {
                    y[r * N + g * 16 + j] = tmp[r * N + g * 16 + j];
                    if (g * 16 + j < NR) best[r] = std::max(best[r], tmp[r * N + g * 16 + j]);
                }
    };
    rescore(groups);
    g_head2.groups += groups.size();
    for (int round = 0; round < 8; ++round) {
        std::vector<size_t> more;
        for (size_t r = 0; r < R; ++r)
            for (size_t v = 0; v < NR; ++v)
                if (!done[v / 16] && y[r * N + v] >= best[r]) { done[v / 16] = 1; more.push_back(v / 16); }
        if (more.empty()) break;
        g_head2.extra += more.size();
        rescore(more);
    }
    ++g_head2.calls;
    g_head2.rows += R;
    if (check)
        for (size_t r = 0; r < R; ++r) {
            const float* a = y + r * N;
            const float* b = ref.data() + r * N;
            const size_t ia = static_cast<size_t>(std::max_element(a, a + NR) - a);
            const size_t ib = static_cast<size_t>(std::max_element(b, b + NR) - b);
            if (ia != ib) ++g_head2.mismatch;
            size_t rank = 0;   // the q8 argmax's rank in the int4 order
            for (size_t v = 0; v < NR; ++v) rank += i4[r][v] > i4[r][ib];
            g_head2.max_rank = std::max(g_head2.max_rank, rank);
            ++g_head2.checked;
        }
}

void par_copy(void* dst, const void* src, size_t bytes) {
    const size_t chunk = 1 << 20, n = (bytes + chunk - 1) / chunk;
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        const size_t o = static_cast<size_t>(i) * chunk;
        std::memcpy(static_cast<uint8_t*>(dst) + o, static_cast<const uint8_t*>(src) + o, std::min(chunk, bytes - o));
    }
}

void lmhead_q8_cpu(const uint8_t* W, size_t N, size_t K, const float* x, float* y) {
    const size_t KB = K / 32;
    static thread_local std::vector<uint32_t> ax;
    static thread_local std::vector<uint64_t> neg;
    static thread_local std::vector<float> sx;
    ax.resize(KB * 8); neg.resize(KB * 8); sx.resize(KB);
    for (size_t B = 0; B < KB; ++B) {
        float m = 0.f;
        for (int i = 0; i < 32; ++i) m = std::max(m, std::fabs(x[B * 32 + i]));
        const float s = m > 0.f ? m / 127.f : 1.f, inv = 1.f / s;
        sx[B] = s;
        for (int kq = 0; kq < 8; ++kq) {
            uint32_t a = 0, pat = 0;
            for (int j = 0; j < 4; ++j) {
                const long v = std::lrint(x[B * 32 + kq * 4 + j] * inv);
                a |= static_cast<uint32_t>(std::labs(v)) << (8 * j);
                if (v < 0) pat |= 1u << j;
            }
            uint64_t mk = 0;
            for (int lane = 0; lane < 16; ++lane) mk |= static_cast<uint64_t>(pat) << (4 * lane);
            ax[B * 8 + kq] = a;
            neg[B * 8 + kq] = mk;
        }
    }
    const size_t G = N / 16;
    const uint32_t* axp = ax.data();
    const uint64_t* negp = neg.data();
    const float* sxp = sx.data();
#pragma omp parallel
    {
        const size_t T = static_cast<size_t>(omp_get_num_threads()), i = static_cast<size_t>(omp_get_thread_num());
        lmhead_groups(W, K, axp, negp, sxp, G * i / T, G * (i + 1) / T, y);
    }
}

__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
void swiglu_tiles(const float* ug, size_t ff, float* h) {
#pragma omp parallel for schedule(static)
    for (long long jj = 0; jj < static_cast<long long>(ff / 64); ++jj) {
        const size_t j = static_cast<size_t>(jj);
        for (size_t i = 0; i < 64; i += 16)
            dn_st(h + 64 * j + i, dn_silu(dn_ld(ug + 128 * j + i)) * dn_ld(ug + 128 * j + 64 + i));
    }
}

__attribute__((target("avx512f,avx512bw,avx512dq,fma")))
static double bf16x_dot_avx512(const float* x, const float* w, size_t n) {
    __m512d a0 = _mm512_setzero_pd(), a1 = _mm512_setzero_pd();
    const __m512i one = _mm512_set1_epi32(1), bias = _mm512_set1_epi32(0x7FFF), hi = _mm512_set1_epi32(static_cast<int>(0xFFFF0000u));
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512i u = _mm512_castps_si512(_mm512_loadu_ps(x + i));
        u = _mm512_and_si512(_mm512_add_epi32(u, _mm512_add_epi32(bias, _mm512_and_si512(_mm512_srli_epi32(u, 16), one))), hi);
        const __m512 xb = _mm512_castsi512_ps(u), wv = _mm512_loadu_ps(w + i);
        a0 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm512_castps512_ps256(xb)), _mm512_cvtps_pd(_mm512_castps512_ps256(wv)), a0);
        a1 = _mm512_fmadd_pd(_mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(xb), 1))),
                             _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(wv), 1))), a1);
    }
    double d = _mm512_reduce_add_pd(_mm512_add_pd(a0, a1));
    for (; i < n; ++i) d += static_cast<double>(bf16_to_f32(f32_to_bf16(x[i]))) * w[i];
    return d;
}
double bf16x_dot(const float* x, const float* w, size_t n) {
    static const bool v4 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
    if (v4) return bf16x_dot_avx512(x, w, n);
    double d = 0;
    for (size_t i = 0; i < n; ++i) d += static_cast<double>(bf16_to_f32(f32_to_bf16(x[i]))) * w[i];
    return d;
}

Q41Lay q41_std(size_t K) {
    const size_t band = 2 * K / 256 * 5120;
    return {2 * band, band, 2 * 5120};
}
Q41Lay q41_stripe(size_t stride) { return {stride, 2 * 5120, 4 * 5120}; }

void q41_prep_x(const float* x, size_t K, Q41X& out) {
    const size_t nb = K / 32;
    out.q.resize(K);
    out.s.resize(nb);
    out.sum.resize(nb);
    out.qsum.resize(nb);
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0.f, sum = 0.f;
        for (size_t i = 0; i < 32; ++i) { amax = std::max(amax, std::fabs(x[b * 32 + i])); sum += x[b * 32 + i]; }
        const float sc = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
        int32_t qs = 0;
        for (size_t i = 0; i < 32; ++i) { out.q[b * 32 + i] = static_cast<int8_t>(std::lrint(x[b * 32 + i] * inv)); qs += out.q[b * 32 + i]; }
        out.s[b] = sc;
        out.sum[b] = sum;
        out.qsum[b] = qs;
    }
}

namespace {
// the 64 nibble bytes of 8 k x 16 rows (byte 8 kk + j: row 2j low, 2j + 1 high) into VNNI order: dword
// lane 2j + q = row 2j's (or 2j + 1's) k 4q .. 4q + 3
const uint8_t* q41_perm() {
    struct Tab {
        alignas(64) uint8_t b[64];
        Tab() {
            for (int P = 0; P < 64; ++P) {
                const int L = P / 4, t = P % 4, j = L / 2, q = L % 2;
                b[P] = static_cast<uint8_t>((4 * q + t) * 8 + j);
            }
        }
    };
    static const Tab tab;
    return tab.b;
}

// MBI: the chunks in mbi's transform (Q41Lay::shift): the nibble bytes XOR 0x88 back to q, d and
// m + 8 d from int16 at 2^-shift
template <bool MBI>
__attribute__((target("avx512f,avx512bw,avx512vnni,avx512vbmi,fma")))
void q41_group16(const uint8_t* const* chunks, size_t ng, size_t rb, const Q41X* const* x, size_t R, float* const* y,
                 size_t row0, int shift) {
    const __m512i perm = _mm512_load_si512(q41_perm()), m4 = _mm512_set1_epi8(0x0F), x88 = _mm512_set1_epi8(static_cast<char>(0x88));
    const __m512 unscale = _mm512_set1_ps(std::ldexp(1.f, -std::max(shift, 0)));
    __m512 acc[8];
    for (size_t r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
    static const size_t pf = [] { const char* e = std::getenv("OFLM_Q41_PF"); return e ? std::strtoul(e, nullptr, 10) : 2ul; }();
    for (size_t g = 0; g < ng; ++g) {
        const uint8_t* ch = chunks[g];
        const uint8_t* nib = ch + 1024 + rb * 2048;
        // the chunks sit 10-20 KB apart: fetch the one pf (2) ahead (its 2 KB of nibbles, its d / m lines)
        if (pf && g + pf < ng) {
            const uint8_t* nx = chunks[g + pf];
            for (size_t o = 0; o < 2048; o += 64) _mm_prefetch(reinterpret_cast<const char*>(nx + 1024 + rb * 2048 + o), _MM_HINT_T0);
            for (size_t o = 0; o < 512; o += 64) {
                _mm_prefetch(reinterpret_cast<const char*>(nx + o), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(nx + 512 + o), _MM_HINT_T0);
            }
        }
        const uint16_t* dp = reinterpret_cast<const uint16_t*>(ch) + rb * 16;
        const uint16_t* mp = reinterpret_cast<const uint16_t*>(ch + 512) + rb * 16;
        for (size_t kb = 0; kb < 8; ++kb) {
            __m512i ae[8], ao[8];
            for (size_t r = 0; r < R; ++r) ae[r] = ao[r] = _mm512_setzero_si512();
            for (size_t it = 0; it < 4; ++it) {
                __m512i v = _mm512_loadu_si512(nib + (kb * 32 + it * 8) * 8);
                if constexpr (MBI) v = _mm512_xor_si512(v, x88);
                const __m512i pe = _mm512_permutexvar_epi8(perm, _mm512_and_si512(v, m4));
                const __m512i po = _mm512_permutexvar_epi8(perm, _mm512_and_si512(_mm512_srli_epi16(v, 4), m4));
                const size_t k0 = g * 256 + kb * 32 + it * 8;
                for (size_t r = 0; r < R; ++r) {
                    int64_t xv;
                    std::memcpy(&xv, x[r]->q.data() + k0, 8);
                    const __m512i xb = _mm512_set1_epi64(xv);
                    ae[r] = _mm512_dpbusd_epi32(ae[r], pe, xb);
                    ao[r] = _mm512_dpbusd_epi32(ao[r], po, xb);
                }
            }
            __m512 d16, mc16;
            if constexpr (MBI) {
                d16 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dp + kb * 32)))), unscale);
                mc16 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(mp + kb * 32)))), unscale);
            } else {
                d16 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(dp + kb * 32))), 16));
                const __m512 m16 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(mp + kb * 32))), 16));
                mc16 = _mm512_fmadd_ps(d16, _mm512_set1_ps(8.f), m16);   // m + 8 d
            }
            const size_t blk = g * 8 + kb;
            for (size_t r = 0; r < R; ++r) {
                const __m512i e2 = _mm512_add_epi32(ae[r], _mm512_shuffle_epi32(ae[r], _MM_PERM_CDAB));
                const __m512i o2 = _mm512_add_epi32(ao[r], _mm512_shuffle_epi32(ao[r], _MM_PERM_CDAB));
                const __m512i isum = _mm512_sub_epi32(_mm512_mask_blend_epi32(0xAAAA, e2, o2), _mm512_set1_epi32(8 * x[r]->qsum[blk]));
                const __m512 fs = _mm512_cvtepi32_ps(isum);
                acc[r] = _mm512_fmadd_ps(mc16, _mm512_set1_ps(x[r]->sum[blk]), acc[r]);
                acc[r] = _mm512_fmadd_ps(fs, _mm512_mul_ps(d16, _mm512_set1_ps(x[r]->s[blk])), acc[r]);
            }
        }
    }
    for (size_t r = 0; r < R; ++r) _mm512_storeu_ps(y[r] + row0, acc[r]);
}
}  // namespace

void gemv_q41_bands(const uint8_t* base, const Q41Lay& lay, size_t K, const Q41X* const* x, size_t R, float* const* y,
                    size_t b0, size_t b1) {
    if (R < 1 || R > 8 || K % 256) throw std::runtime_error("open_qwen36: gemv_q41: 1..8 rows, K a multiple of 256");
    const size_t ng = K / 256;
    const uint8_t* chunks[64];
    if (ng > 64) throw std::runtime_error("open_qwen36: gemv_q41: K past 16384");
    for (size_t b = b0; b < b1; ++b)
        for (size_t h = 0; h < 2; ++h) {
            for (size_t g = 0; g < ng; ++g) chunks[g] = base + (b / 2) * lay.s2 + (b % 2) * lay.s1 + g * lay.sk + h * 5120;
            for (size_t rb = 0; rb < 2; ++rb)
                if (lay.shift >= 0) q41_group16<true>(chunks, ng, rb, x, R, y, b * 64 + h * 32 + rb * 16, lay.shift);
                else q41_group16<false>(chunks, ng, rb, x, R, y, b * 64 + h * 32 + rb * 16, -1);
        }
}

int mbi_shift(const uint8_t* chunks, size_t nch, size_t stride) {
    float dmax = 0.f, mcmax = 0.f;
    for (size_t c = 0; c < nch; ++c) {
        const uint16_t* d = reinterpret_cast<const uint16_t*>(chunks + c * stride);
        const uint16_t* m = d + 256;
        for (size_t i = 0; i < 256; ++i) {
            const float dv = bf16_to_f32(d[i]), mc = bf16_to_f32(m[i]) + 8.f * dv;
            dmax = std::max(dmax, std::fabs(dv));
            mcmax = std::max(mcmax, std::fabs(mc));
        }
    }
    const double lim = std::min(dmax > 0.f ? 8000.0 / dmax : 1e30, mcmax > 0.f ? 32000.0 / mcmax : 1e30);
    return std::max(0, std::min(24, static_cast<int>(std::floor(std::log2(lim)))));
}

void mbi_transform(uint8_t* chunks, size_t nch, size_t stride, int shift) {
    for (size_t c = 0; c < nch; ++c) {
        uint8_t* ch = chunks + c * stride;
        uint16_t* d = reinterpret_cast<uint16_t*>(ch);
        uint16_t* m = d + 256;
        for (size_t i = 0; i < 256; ++i) {
            const double dv = bf16_to_f32(d[i]), mc = bf16_to_f32(m[i]) + 8.0 * dv;
            const int16_t di = static_cast<int16_t>(std::lrint(std::ldexp(dv, shift)));
            const int16_t mi = static_cast<int16_t>(std::lrint(std::ldexp(mc, shift)));
            std::memcpy(d + i, &di, 2);
            std::memcpy(m + i, &mi, 2);
        }
        for (size_t i = 1024; i < 5120; ++i) ch[i] ^= 0x88;
    }
}

void q41_pack_std(const float* W, size_t n, size_t K, uint8_t* out) {
    if (n % 64 || K % 256) throw std::runtime_error("open_qwen36: q41_pack_std: n % 64 and K % 256");
    const Q41Lay lay = q41_std(K);
    const long long nchunks = static_cast<long long>(n / 32 * (K / 256));
#pragma omp parallel for
    for (long long cc = 0; cc < nchunks; ++cc) {
        const size_t r32 = static_cast<size_t>(cc) / (K / 256), g = static_cast<size_t>(cc) % (K / 256);
        const size_t b = r32 / 2, h = r32 % 2;
        uint8_t* c = out + (b / 2) * lay.s2 + (b % 2) * lay.s1 + g * lay.sk + h * 5120;
        std::memset(c + 1024, 0, 4096);
        for (size_t r = 0; r < 32; ++r) {
            const float* w = W + (b * 64 + h * 32 + r) * K + g * 256;
            for (size_t kb = 0; kb < 8; ++kb) {
                float lo = w[kb * 32], hi = lo;
                for (size_t i = 0; i < 32; ++i) { lo = std::min(lo, w[kb * 32 + i]); hi = std::max(hi, w[kb * 32 + i]); }
                const uint16_t mb = f32_to_bf16(lo), db = f32_to_bf16((hi - lo) / 15.f);
                const float m = bf16_to_f32(mb), d = bf16_to_f32(db), inv = d > 0.f ? 1.f / d : 0.f;
                std::memcpy(c + 2 * (kb * 32 + r), &db, 2);
                std::memcpy(c + 512 + 2 * (kb * 32 + r), &mb, 2);
                for (size_t i = 0; i < 32; ++i) {
                    const int q = std::min(15, std::max(0, static_cast<int>(std::lrint((w[kb * 32 + i] - m) * inv))));
                    const size_t kk = kb * 32 + i, rb = r / 16, r16 = r % 16, idx = (rb * 256 + kk) * 16 + r16;
                    c[1024 + idx / 2] |= static_cast<uint8_t>(q << ((idx & 1) * 4));
                }
            }
        }
    }
}

void gemv_q41_rows(const uint8_t* base, const Q41Lay& lay, size_t n, size_t K, const float* x, size_t R, float* y,
                   size_t ldy) {
    std::vector<Q41X> xq(R);
    const Q41X* xp[8];
    float* yp[8];
    for (size_t r = 0; r < R; ++r) { q41_prep_x(x + r * K, K, xq[r]); xp[r] = &xq[r]; yp[r] = y + r * ldy; }
    const long long nb = static_cast<long long>(n / 64);
#pragma omp parallel for schedule(static)
    for (long long b = 0; b < nb; ++b) gemv_q41_bands(base, lay, K, xp, R, yp, static_cast<size_t>(b), static_cast<size_t>(b) + 1);
}

}  // namespace host
}  // namespace open_qwen36
