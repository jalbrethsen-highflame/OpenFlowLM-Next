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
            }
            for (size_t t = 0; t < nt; ++t) {
                for (size_t j = 0; j < D; j += 16) {
                    dnv x = dn_bc(G[t]) * dn_ld(&Qs[t][j]);
                    for (size_t s2 = 0; s2 <= t; ++s2) x = x + dn_bc(r[t][s2] * QK[t][s2]) * dn_ld(&dl[s2][j]);
                    dn_st(o + j, x * dn_bc(inv_sqrt));
                }
                rms_vec(o, D, nw, g.eps, on);
                float* out = og + (t0 + t) * vw + h * D;
                const float* zz = z + (t0 + t) * vw + h * D;
                for (size_t j = 0; j < D; j += 16) dn_st(out + j, dn_ld(on + j) * dn_silu(dn_ld(zz + j)));
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
                const float* zz = z + t * vw + h * dim;
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
            const float* zz = z + t * vw + h * dim;
            for (size_t j = 0; j < 128; j += 16) dn_st(out + j, dn_ld(on + j) * dn_silu(dn_ld(zz + j)));
        }
    }
}

// deltanet_block's phases (us, summed): setup, phase 1 (conv, silu, q / k norm), phase 1b
// (alpha / beta), window; read by the prompt pipe's OFLM_OPEN_MID_PROF
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
    Q.resize(R * key_w); Kk.resize(R * key_w); V.resize(R * vw); decay.resize(R * g.value_heads); beta.resize(R * g.value_heads);
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
            for (size_t hh = 0; hh < g.key_heads; ++hh)
                for (int which = 0; which < 2; ++which) {
                    const float* src = c.data() + which * key_w + hh * dim;
                    float* dst = (which ? Kk : Q).data() + t * key_w + hh * dim;
                    double ss = 0;
                    for (size_t j = 0; j < dim; ++j) ss += static_cast<double>(src[j]) * src[j];
                    const float r = static_cast<float>(1.0 / std::sqrt(ss + 1e-6));   // dn_glue's L2 norm
                    for (size_t j = 0; j < dim; ++j) dst[j] = src[j] * r;
                }
            p1m(2);
            std::copy(c.begin() + 2 * key_w, c.end(), V.begin() + t * vw);
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
    // ---- phase 2, per head over every token: the gated delta rule on S (in place), the gated norm
    const auto tp1 = std::chrono::steady_clock::now();
    // one pass per packed sequence, S zeroed where one starts
    for (size_t a = 0; a < R;) {
        size_t b = a + 1;
        while (b < R && !(g.reset && g.reset[b])) ++b;
        if (g.reset && g.reset[a]) std::fill(S, S + g.value_heads * g.s_rows * dim, 0.f);
        delta_rule_heads(g, b - a, grp, inv_sqrt, Q.data() + a * key_w, Kk.data() + a * key_w, V.data() + a * vw,
                         decay.data() + a * g.value_heads, beta.data() + a * g.value_heads, z + a * vw, nw, S, og + a * vw);
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
                      const float* KT, const float* VF, size_t P, float* og, size_t g, size_t t0, size_t kend) {
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
    for (size_t c0 = 0; c0 <= last; c0 += 64) {
        const size_t kc = std::min<size_t>(64, last + 1 - c0);
        // S = Q K^T over the chunk's 64 keys
        for (size_t i0 = 0; i0 < Mp; i0 += kRB) {
            v16f a00{}, a01{}, a02{}, a03{}, a10{}, a11{}, a12{}, a13{}, a20{}, a21{}, a22{}, a23{}, a30{}, a31{}, a32{}, a33{};
            const float *q0 = q + i0 * hd, *q1 = q0 + hd, *q2 = q1 + hd, *q3 = q2 + hd;
            for (size_t d = 0; d < hd; ++d) {
                const float* kd = kt + (c0 / 64 * hd + d) * 64;
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
            if (!valid) {
                for (size_t j = 0; j < 64; ++j) si[j] = 0.f;
                cr[i] = 1.f;
                continue;
            }
            for (size_t j = valid; j < 64; ++j) si[j] = -1e30f;
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
                    const float* vj = vf + (c0 + j) * hd + dg;
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
        const float inv = 1.f / l[i];
        for (size_t d = 0; d < hd; ++d) out[d] = acc[i * hd + d] * inv;
    }
}
// (no pop: GCC reports -Wpsabi at the end of the file, for the clones)

void attention_tiles(size_t hd, size_t nh, size_t kvh, size_t R, size_t pos0, float scale, const float* Q,
                     const uint16_t* kv, size_t kv_row_elems, float* og, size_t kend = 0) {
    if (hd % 64 == 0) {
        const size_t rows = kend ? kend : pos0 + R, P = (rows + 63) / 64 * 64, kvw = kvh * hd, grp = nh / kvh;
        static thread_local std::vector<float> KT, VF;
        KT.resize(kvh * hd * P);
        VF.resize(kvh * P * hd);
        float *const kt = KT.data(), *const vf = VF.data();   // thread_local: the workers need the caller's
        // the window once, in 64-row blocks per thread; the pad rows zero
#pragma omp parallel for
        for (long long pb = 0; pb < static_cast<long long>(P / 64); ++pb)
            for (size_t p = static_cast<size_t>(pb) * 64; p < static_cast<size_t>(pb) * 64 + 64; ++p) {
                const uint16_t* row = kv + p * kv_row_elems;
                for (size_t gh = 0; gh < kvh; ++gh) {
                    float* kd = kt + gh * hd * P + (p / 64 * hd) * 64 + p % 64;
                    float* vd = vf + (gh * P + p) * hd;
                    for (size_t d = 0; d < hd; ++d) {
                        kd[d * 64] = p < rows ? bf16_to_f32(row[gh * hd + d]) : 0.f;
                        vd[d] = p < rows ? bf16_to_f32(row[kvw + gh * hd + d]) : 0.f;
                    }
                }
            }
        const size_t qt = std::max<size_t>(8, 128 / grp);   // ~128 query rows per work item
        const size_t chunks = (R + qt - 1) / qt;
        const long long items = static_cast<long long>(kvh * chunks);
#pragma omp parallel for schedule(dynamic, 1)
        for (long long it = items - 1; it >= 0; --it)
            attention_tile64(hd, nh, kvh, R, pos0, qt, scale, Q, kt, vf, P, og,
                             static_cast<size_t>(it) % kvh, (static_cast<size_t>(it) / kvh) * qt, kend);
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
    if (split && R <= 8 && g.hd % 8 == 0 && grp <= 8)
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

void router_block(size_t T, size_t hid, size_t E, size_t topk, const float* xm, const float* Wr, float* probs,
                  int32_t* idx, float* w) {
    if (topk > E) throw std::runtime_error("open_qwen36: router_block: topk past the expert count");
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
        float mx = -1e30f;
        for (size_t e = 0; e < E; ++e) mx = std::max(mx, lg[e]);
        double denom = 0;
        for (size_t e = 0; e < E; ++e) {
            lg[e] = std::exp(lg[e] - mx);
            denom += lg[e];
        }
        const float inv = static_cast<float>(1.0 / denom);
        for (size_t e = 0; e < E; ++e) {
            lg[e] *= inv;
            probs[t * E + e] = lg[e];
        }
        double wsum = 0;
        for (size_t s = 0; s < topk; ++s) {
            size_t best = E;
            for (size_t e = 0; e < E; ++e)
                if (!taken[e] && (best == E || lg[e] > lg[best])) best = e;
            taken[best] = 1;
            idx[t * topk + s] = static_cast<int32_t>(best);
            w[t * topk + s] = lg[best];
            wsum += lg[best];
        }
        for (size_t s = 0; s < topk; ++s) w[t * topk + s] = static_cast<float>(w[t * topk + s] / wsum);
    }
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
    const RowsFn fn = kRowsFn[R - 1];
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
            fn(B + s * KT * kB4Tile, xq.data(), t0, t1, acc);
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

}  // namespace host
}  // namespace open_qwen36
