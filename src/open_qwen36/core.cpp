/// \file core.cpp
/// \brief The resident open-kernel decode engine: a manifest interpreter (see core.hpp).
#include "open_qwen36/core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <exception>
#include <condition_variable>
#include <mutex>
#include <set>
#include <stdexcept>

#include <omp.h>

#include <omp.h>

#ifdef _WIN32
#include <windows.h>
#endif
#include <immintrin.h>

#include "xrt/experimental/xrt_ext.h"
#include "xrt/experimental/xrt_xclbin.h"

#include "open_qwen36/block_host.hpp"

namespace open_qwen36 {

namespace fs = std::filesystem;

namespace {
// dit_gemm's B operand from a q4_1 pool region (q4_1_pack.py's chunk order, rs = 2: a 64-row
// band's 256-wide k-group is two 5120 B chunks of 32 rows): B[k][n] = W[n][k], N padded with
// zeros to npad, as designs/dit_gemm/pack.py's pack_b lays it out -- per 64 (K) x 128 (N)
// tile [16 n-blocks][8 k-blocks][8 n][8 k], tiles column-major, each 8 k of one column a
// bfp16ebs8 block (exponent byte = the largest IEEE exponent, 8 int8 mantissas, rounded to
// nearest exactly as pack.py's bfp16ebs8_encode).
// One bfp16ebs8 block from 8 floats: exponent byte = the largest IEEE exponent, 8 int8
// mantissas rounded to nearest exactly as designs/dit_gemm/pack.py's bfp16ebs8_encode.
// OFLM_OPEN_DIT_FFN=0: the dense FFN as two plain dit GEMMs with silu on the host (an A/B
// against the device SwiGLU epilogue, whose tanh is approximate)
bool dit_ffn_on() {
    static const bool on = [] { const char* e = std::getenv("OFLM_OPEN_DIT_FFN"); return !(e && *e == '0'); }();
    return on;
}

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

inline void bfp_encode8(const float* v, uint8_t* blk) {
    uint32_t bits[8];
    std::memcpy(bits, v, 32);
    int max_exp = 0;
    for (uint32_t bt : bits) max_exp = std::max(max_exp, static_cast<int>((bt >> 23) & 0xFF));
    blk[0] = static_cast<uint8_t>(max_exp);
    for (size_t j = 0; j < 8; ++j) {
        const int e = static_cast<int>((bits[j] >> 23) & 0xFF);
        int64_t mant = static_cast<int64_t>(bits[j] & 0x7FFFFF) | (e != 0 ? (int64_t{1} << 23) : 0);
        if (bits[j] >> 31) mant = -mant;
        const int shift = 17 + std::min(max_exp - e, 31);
        int64_t q = mant >> shift;                                   // floor, then nearest, ties to even
        const int64_t r = mant - (q << shift), half = int64_t{1} << (shift - 1);
        q += (r > half) || (r == half && (q & 1));
        q = std::clamp<int64_t>(q, -128, 127);
        blk[1 + j] = static_cast<uint8_t>(static_cast<int8_t>(q));
    }
}

// designs/dit2's A operand (pack_a.py's order): x [M, K] f32 -> [row group M/512][core row 4]
// [k-step K/64][quarter 4][k-block 8][row-block 4] blocks of 8 rows x 8 k, each row's 8 k one
// bfp16ebs8 block. Rows past `rows` are zeros.
// hs (manifest gemm_dit.hs, the hand-scheduled 2x2 microkernel): inside each quarter, row-block
// pairs, each [k-block 8][the pair's 2 row blocks] (pack_a.py order2) instead of [k-block][row-block].
// mt: the per-core tile height (dit2 DG_MT, 128 or 64 under hs), mt / 32 quarters a unit
void dit2_pack_a(const float* x, size_t rows, size_t M, size_t K, uint8_t* out, bool hs = false, size_t mt = 128) {
    const size_t KS = K / 64, qn = mt / 32;
    const long long units = static_cast<long long>(M / mt * KS);
#pragma omp parallel for
    for (long long uu = 0; uu < units; ++uu) {
        const size_t u = static_cast<size_t>(uu), ks = u % KS, rc = u / KS;   // rc = rg * 4 + crow
        uint8_t* o = out + u * (qn * 8 * 4 * 8 * 9);
        float zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t q = 0; q < qn; ++q)
            for (size_t j = 0; j < 32; ++j)
                for (size_t i = 0; i < 8; ++i) {
                    // j walks [k-block 8][row-block 4], or under hs [pair 2][k-block 8][2]
                    const size_t kb = hs ? (j / 2) % 8 : j / 4, rb = hs ? (j / 16) * 2 + j % 2 : j % 4;
                    const size_t r = rc * mt + q * 32 + rb * 8 + i, k0 = ks * 64 + kb * 8;
                    bfp_encode8_avx2(r < rows ? x + r * K + k0 : zero, o);
                    o += 9;
                }
    }
}

// A dit weight as [n, k] f32 from its band-order q4_1 pool chunks (5120 B: 32 rows x 256 k).
// dit_fa256b's operands (designs/dit_fa256b/pack_fa.py, the same bytes): bfp16 in the order its
// MAC pops them. Rows at or past t_real are zero. Q: one head, rows of x (ld apart, from column
// col) times s, per 32-row block [row pair 2][d step 32][tile 2][8 rows][8 d] = 9216 bytes.
constexpr size_t kFabQBlock = 9216, kFabChunk = 18432;
void fab_pack_q(const float* x, size_t ld, size_t col, size_t T, size_t t_real, float s, uint8_t* out) {
    float v[8];
    for (size_t rb = 0; rb < T / 32; ++rb)
        for (size_t zp = 0; zp < 2; ++zp)
            for (size_t i = 0; i < 32; ++i)
                for (size_t r2 = 0; r2 < 2; ++r2)
                    for (size_t r8 = 0; r8 < 8; ++r8) {
                        const size_t row = rb * 32 + zp * 16 + r2 * 8 + r8;
                        for (size_t k = 0; k < 8; ++k) v[k] = row < t_real ? x[row * ld + col + i * 8 + k] * s : 0.f;
                        bfp_encode8_avx2(v, out);
                        out += 9;
                    }
}
// K: one kv head of bf16 cache rows (stride elements apart, from column col), per 64-key chunk
// [piece 4][d step 32][tile 2][8 keys][8 d] = 18432 bytes.
void fab_pack_k(const uint16_t* kv, size_t stride, size_t col, size_t T, size_t t_real, uint8_t* out) {
    float v[8];
    for (size_t c = 0; c < T / 64; ++c)
        for (size_t j = 0; j < 4; ++j)
            for (size_t i = 0; i < 32; ++i)
                for (size_t n2 = 0; n2 < 2; ++n2)
                    for (size_t k8 = 0; k8 < 8; ++k8) {
                        const size_t key = c * 64 + j * 16 + n2 * 8 + k8;
                        for (size_t k = 0; k < 8; ++k)
                            v[k] = key < t_real ? bf16_to_f32(kv[key * stride + col + i * 8 + k]) : 0.f;
                        bfp_encode8_avx2(v, out);
                        out += 9;
                    }
}
// V^T: per 64-key chunk [element 4][column pair 4][key step 8][tile 2][8 d columns][8 keys].
void fab_pack_v(const uint16_t* kv, size_t stride, size_t col, size_t T, size_t t_real, uint8_t* out) {
    float v[8];
    for (size_t c = 0; c < T / 64; ++c)
        for (size_t e = 0; e < 4; ++e)
            for (size_t jl = 0; jl < 4; ++jl)
                for (size_t i = 0; i < 8; ++i)
                    for (size_t n2 = 0; n2 < 2; ++n2)
                        for (size_t dc = 0; dc < 8; ++dc) {
                            const size_t d = (e * 4 + jl) * 16 + n2 * 8 + dc;
                            for (size_t k = 0; k < 8; ++k) {
                                const size_t key = c * 64 + i * 8 + k;
                                v[k] = key < t_real ? bf16_to_f32(kv[key * stride + col + d]) : 0.f;
                            }
                            bfp_encode8_avx2(v, out);
                            out += 9;
                        }
}

void dit_decode_q4(const uint8_t* pool, size_t n, size_t k, float* W) {
    const size_t per_band = 2 * k / 256, nch = n * k / (32 * 256);
#pragma omp parallel for
    for (long long cc = 0; cc < static_cast<long long>(nch); ++cc) {
        const size_t c = static_cast<size_t>(cc), band = c / per_band, ci = c % per_band;
        const size_t r0 = 64 * band + 32 * (ci % 2), c0 = 256 * (ci / 2);
        const uint8_t* b = pool + c * 5120;
        const uint16_t* d = reinterpret_cast<const uint16_t*>(b);
        const uint16_t* m = reinterpret_cast<const uint16_t*>(b + 512);
        const uint8_t* qb = b + 1024;
        for (size_t rb = 0; rb < 2; ++rb)
            for (size_t kk = 0; kk < 256; ++kk)
                for (size_t r16 = 0; r16 < 16; ++r16) {
                    const size_t i = (rb * 256 + kk) * 16 + r16, r = rb * 16 + r16;
                    const uint8_t nib = (i & 1) ? (qb[i >> 1] >> 4) : (qb[i >> 1] & 0xF);
                    const size_t si = (kk / 32) * 32 + r;
                    W[(r0 + r) * k + c0 + kk] = static_cast<float>(nib) * bf16_to_f32(d[si]) + bf16_to_f32(m[si]);
                }
    }
}

// [n, k] f32 -> designs/dit2's B operand: bfp16 [Npad/128][K/64][16][8][8] blocks (9 bytes per 8).
// colmap (optional, npad entries): packed column c takes source row colmap[c] (-1 = zeros), for
// the SwiGLU epilogue's [gate 64 | up 64] tiles.
// hs: per [N tile][K tile], column-block pairs, each [k-block 8][the pair's 2] (pack.py layout2).
void dit_encode(const float* W, size_t n, size_t k, size_t npad, uint8_t* out, const int64_t* colmap = nullptr,
                bool hs = false) {
    const size_t KT = k / 64, NT = npad / 128;
#pragma omp parallel for
    for (long long tt = 0; tt < static_cast<long long>(NT * KT); ++tt) {
        const size_t nt = static_cast<size_t>(tt) / KT, kt = static_cast<size_t>(tt) % KT;
        for (size_t nb = 0; nb < 16; ++nb)
            for (size_t kb = 0; kb < 8; ++kb)
                for (size_t nin = 0; nin < 8; ++nin) {
                    const size_t c = nt * 128 + nb * 8 + nin, k0 = kt * 64 + kb * 8;
                    const int64_t src = colmap ? colmap[c] : static_cast<int64_t>(c);
                    const size_t col = src < 0 ? n : static_cast<size_t>(src);
                    const size_t bi = hs ? ((((nt * KT + kt) * 8 + nb / 2) * 8 + kb) * 2 + nb % 2) * 8 + nin
                                         : ((((nt * KT + kt) * 16 + nb) * 8 + kb) * 8 + nin);
                    uint8_t* blk = out + bi * 9;
                    float v8[8];
                    for (size_t j = 0; j < 8; ++j) v8[j] = col < n ? W[col * k + k0 + j] : 0.f;
                    bfp_encode8_avx2(v8, blk);
                }
    }
}

// [n, k] f32 -> dit2's int4-mantissa B (manifest gemm_dit.b4; pack.py pack_b4, the same values):
// per [128-column tile][64-k tile] 4096 bytes of nibbles, blocks [column pair 8][k-block 8][2] of
// 8 columns x 8 k (value i = column * 8 + k, low nibble first), then one exponent byte per column:
// a weight is mantissa (-8..7) x 2^(exponent - 133), the exponent the smallest that keeps the
// column's 64 values within +-7, or one less when clipping to -8..7 there gives a lower squared
// error. colmap as dit_encode.
constexpr size_t kB4Tile = 4096 + 128;
void dit_encode_b4(const float* W, size_t n, size_t k, size_t npad, uint8_t* out, const int64_t* colmap = nullptr) {
    const size_t KT = k / 64, NT = npad / 128;
#pragma omp parallel for
    for (long long tt = 0; tt < static_cast<long long>(NT * KT); ++tt) {
        const size_t nt = static_cast<size_t>(tt) / KT, kt = static_cast<size_t>(tt) % KT;
        uint8_t* tile = out + (nt * KT + kt) * kB4Tile;
        std::memset(tile, 0, 4096);
        for (size_t cl = 0; cl < 128; ++cl) {
            const size_t c = nt * 128 + cl;
            const int64_t src = colmap ? colmap[c] : static_cast<int64_t>(c);
            const float* w = (src >= 0 && static_cast<size_t>(src) < n) ? W + static_cast<size_t>(src) * k + kt * 64 : nullptr;
            float amax = 0.f;
            if (w) for (size_t j = 0; j < 64; ++j) amax = std::max(amax, std::fabs(w[j]));
            int e2 = amax > 0.f ? static_cast<int>(std::ceil(std::log2(static_cast<double>(amax) / 7.0))) : -133;
            e2 = std::min(std::max(e2, -133), 122);
            auto q = [](float x, double inv) { return std::min(7.0, std::max(-8.0, std::nearbyint(x * inv))); };
            if (w && e2 > -133) {
                double err[2] = {0.0, 0.0};
                for (int d = 0; d < 2; ++d) {
                    const double inv = std::ldexp(1.0, -(e2 - d)), s = std::ldexp(1.0, e2 - d);
                    for (size_t j = 0; j < 64; ++j) {
                        const double r = q(w[j], inv) * s - static_cast<double>(w[j]);
                        err[d] += r * r;
                    }
                }
                if (err[1] < err[0]) --e2;
            }
            tile[4096 + cl] = static_cast<uint8_t>(e2 + 133);
            const double inv = std::ldexp(1.0, -e2);
            const size_t pair = cl / 16, n2 = (cl / 8) % 2, g = cl % 8;
            for (size_t j = 0; j < 64; ++j) {
                const int m = w ? static_cast<int>(q(w[j], inv)) : 0;
                const size_t kb = j / 8, t = j % 8;
                const size_t i = (((pair * 8 + kb) * 2 + n2) * 64) + g * 8 + t;
                tile[i / 2] |= static_cast<uint8_t>((m & 15) << ((i & 1) * 4));
            }
        }
    }
}

void dit_pack_region(const uint8_t* pool, size_t n, size_t k, size_t npad, uint8_t* out,
                     const int64_t* colmap = nullptr, bool hs = false) {
    std::vector<float> W(n * k);
    dit_decode_q4(pool, n, k, W.data());
    dit_encode(W.data(), n, k, npad, out, colmap, hs);
}


// 0167/#32: the GEMM route's host math reuses this file's own
// open_qwen36::bf16_to_f32 / open_qwen36::f32_to_bf16 (q4nx_file.hpp) --
// both already round-to-nearest-even, matching open_npue/npue_pack.cpp's
// bf16_rne and ml_dtypes.bfloat16's cast exactly (checked: same bit
// arithmetic, `u + 0x7FFF + ((u>>16)&1)`). NOT open_npue's own tile_b,
// which has internal (anonymous-namespace) linkage and is not declared in
// its header, so it is not callable from here; its algorithm is reproduced
// in tile_gemm_x() below instead, using these two conversions.

constexpr int kOpcode = 3;
constexpr size_t kBoAlign = 1u << 20;  // XDNA wants 1 MB-aligned buffer sizes
size_t padup(size_t n) { return (n + kBoAlign - 1) / kBoAlign * kBoAlign; }

std::vector<uint8_t> read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("open_qwen36: cannot read " + p.string());
    std::streamsize n = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> v(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(v.data()), n)) throw std::runtime_error("open_qwen36: short read " + p.string());
    return v;
}

// OPEN-REQUEST-ISOLATION. Every device-to-host read in this file goes through read_back().
// What xrt::bo::sync(FROM_DEVICE) is on the Windows driver (the shim, xrt_core.dll,
// disassembled 2026-09-23 -- the XRT headers say nothing about it): below the size of the
// L3 cache it is a user-mode CLFLUSH loop over every 64-byte line the range touches (the
// start rounded down, the partial last line included), and above it a call into the kernel
// driver. The loop has no fence after it. CLFLUSH is not ordered against later loads -- AMD's
// manual: "the only way to avoid this situation is to use the MFENCE instruction after the
// CLFLUSH instruction" -- so the caller's first load of a line it just flushed may still be
// served from the stale cached copy. The fence closes that. It does NOT make a record the
// device has not finished writing visible; route() keeps its sentinel for that.
// OFLM_OPEN_READ_FENCE=0 drops the fence, for the A/B only.
bool g_read_fence = true;

void read_back(xrt::bo& bo, size_t bytes, size_t off) {
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, off);
    if (g_read_fence) _mm_mfence();
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::string fmt_idx(const uint32_t* idx) {
    std::string s;
    for (int i = 0; i < 8; ++i) s += (i ? "," : "") + std::to_string(idx[i]);
    return s;
}

// Busy CPU cores throttle the NPU. Measured on this box (Ryzen AI, .claude/plans/
// prefill-parity.md workstream A): after every `#pragma omp parallel for` in the host
// stages the OpenMP workers spin-wait for the runtime's blocktime, and the NEXT
// dispatch runs while they spin. The CPU, NPU and iGPU share a package power budget,
// so a spinning core pulls the array's clock down -- twelve external spin processes,
// touching no memory at all, slow the q8 lm head from 11.5 to 35.4 ms. Inside the
// prefill block every dispatch ran at 1.6-2.1x its alone-time; threads ASLEEP and
// threads ABSENT gave the same dispatch times, so it is the spinning and not the host
// work, and even four spinning workers throttle it fully (no thread-count sweet spot).
// PASSIVE costs the host stages one wake-up per parallel region (+7 %) and gives back
// 19 % of the block: 2582-token prefill 38.55 -> 29.93 s. Bit-exact either way.
//
// It has to be an environment variable: MSVC's VCOMP140.DLL reads OMP_WAIT_POLICY when it
// initialises, and there is no API for it. A policy already in the environment wins, and
// OFLM_OPEN_OMP_PASSIVE=0 turns this off.
//
// **The link line is part of this.** vcomp initialises when it LOADS, not at the first
// parallel region, and an implicitly linked DLL loads before any of the exe's static
// initialisers run -- so setting the variable here reached a runtime that had already
// read it, and for a year the binary carried a PASSIVE it never applied. What proved it:
// the same trick for OMP_NUM_THREADS set the variable to 10 and `omp_get_max_threads()`
// still said 24. Delay-loading vcomp (`/DELAYLOAD:VCOMP140.DLL delayimp.lib`, see
// build.cmd and CMakeLists.txt) moves its load to the first call into it, which is after
// this initialiser. Measured at 2582 tokens: 20.5 s with the variable unset against
// 17.1 s with it set from outside, the same binary. Any build of this file that drops
// the delay-load silently loses that.
bool g_omp_policy_too_late = false;   ///< vcomp was already loaded when we set the variable

struct OmpWaitPolicy {
    OmpWaitPolicy() {
        if (std::getenv("OMP_WAIT_POLICY")) return;
        const char* off = std::getenv("OFLM_OPEN_OMP_PASSIVE");
        if (off && std::string(off) == "0") return;
#ifdef _WIN32
        // The check that makes the delay-load's absence loud. If vcomp is already in the
        // process at this point it has read its configuration and the _putenv below is a
        // no-op -- which is exactly the state this binary shipped in and nobody noticed.
        g_omp_policy_too_late = GetModuleHandleW(L"VCOMP140.DLL") != nullptr;
        _putenv_s("OMP_WAIT_POLICY", "PASSIVE");
#else
        setenv("OMP_WAIT_POLICY", "PASSIVE", 0);
#endif
    }
};
const OmpWaitPolicy g_omp_wait_policy;

/// Host threads for the block route's stages, or 0 to leave the runtime alone.
///
/// How MANY threads is a separate question from whether they spin, and once the policy
/// above is really in force it is a settled one. While the workers still spun through
/// every dispatch, fewer of them was faster: at the runtime's own count every dispatch ran
/// at 1.4x its `--bench` rate (`mb_s256` 17.5 ms against 12.4, and the same dispatch
/// repeated with no host work in front of it still 17.5), at ten threads every one was at
/// its own rate, and 2582 tokens went 20.4 -> 18.9 s. With PASSIVE applied the ordering
/// reverses -- 17.6 s at the full count against 19.5 at ten -- because sleeping workers
/// cost the package nothing and the host stages want every core. So the default is the
/// runtime's own count (on Linux, one per physical core -- see physical_cores);
/// OFLM_OPEN_OMP_THREADS stays as a knob for a box where it isn't.
///
/// This one cannot go through the environment even with the delay-load: vcomp latches its
/// thread count the first time anything asks, which is inside the Core already.
///
/// Linux: one thread per PHYSICAL core, not per SMT thread. Measured on the Strix Halo (16
/// cores / 32 threads, LFM2-MoE 24B, 4009-token prompt, libgomp, PASSIVE): 32 threads took
/// 8.0-9.9 s and 16 took 6.7 s, run after run; the short parallel regions around each
/// dispatch (tile_x, transpose, the expert gather / scatter) went 5-10x faster, the long ones
/// (attention) were unchanged.
unsigned physical_cores() {
#ifdef __linux__
    std::set<std::string> cores;
    for (unsigned c = 0; c < 4096; ++c) {
        std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list");
        if (!f) {
            if (c >= std::thread::hardware_concurrency()) break;
            continue;                                  // offline cpu
        }
        std::string sib;
        std::getline(f, sib);
        cores.insert(sib);
    }
    return static_cast<unsigned>(cores.size());
#else
    return 0;
#endif
}

unsigned omp_thread_budget() {
    if (std::getenv("OMP_NUM_THREADS")) return 0;      // the environment has already decided
    if (const char* e = std::getenv("OFLM_OPEN_OMP_THREADS"))
        return static_cast<unsigned>(std::strtoul(e, nullptr, 10));
    return physical_cores();
}

}  // namespace

// ---- exact bf16 prompt weights. q4nx-build writes bf16_map.json beside model.q4nx: for each
// quantized tensor, the HF safetensors shard, tensor name and row order its rows came from.
// The GEMM route then builds its bfp16 weights from those bf16 rows instead of the q4_1 / q8_0
// codes (OFLM_OPEN_BF16_SRC=0 off). A std_perm op's pool chunks are exactly the tensor's row
// range [32 chunk0 / ncol, + 64 nch / (in_dim / 128)) in the band law dit_decode_q4 undoes, so a
// weight is its ops' row ranges stacked in op order.
static std::string layer_tensor(const std::string& name, int layer) {
    std::string s = name;
    for (size_t p = s.find("{l}"); p != std::string::npos; p = s.find("{l}", p)) s.replace(p, 3, std::to_string(layer));
    return s;
}

struct HfBf16 {
    struct T { std::string file; uint64_t off = 0; size_t rows = 0, cols = 0; bool qproj = false; };
    std::map<std::string, T> t;

    explicit HfBf16(const fs::path& map_path) {
        std::ifstream mf(map_path);
        const nlohmann::json j = nlohmann::json::parse(mf);
        const fs::path dir = j.at("hf_dir").get<std::string>();
        std::map<std::string, std::pair<nlohmann::json, uint64_t>> hdr;   // shard -> (header, data base)
        for (const auto& [name, e] : j.at("map").items()) {
            const std::string shard = e.at("shard").get<std::string>();
            auto it = hdr.find(shard);
            if (it == hdr.end()) {
                std::ifstream f(dir / shard, std::ios::binary);
                uint64_t n = 0;
                f.read(reinterpret_cast<char*>(&n), 8);
                std::string h(n, '\0');
                f.read(h.data(), static_cast<std::streamsize>(n));
                if (!f) throw std::runtime_error("open_qwen36: bf16 source: cannot read the header of " + (dir / shard).string());
                it = hdr.emplace(shard, std::make_pair(nlohmann::json::parse(h), 8 + n)).first;
            }
            const std::string hf = e.at("hf").get<std::string>();
            const auto& te = it->second.first.at(hf);
            T x;
            x.file = (dir / shard).string();
            x.rows = e.at("rows").get<size_t>();
            x.cols = e.at("cols").get<size_t>();
            x.qproj = e.at("row_order").get<std::string>() == "qproj_heads";
            if (te.at("dtype").get<std::string>() != "BF16" || te.at("shape").size() != 2 ||
                te.at("shape")[0].get<size_t>() != x.rows || te.at("shape")[1].get<size_t>() != x.cols)
                throw std::runtime_error("open_qwen36: bf16 source: " + hf + " is not a BF16 [" + std::to_string(x.rows) +
                                         ", " + std::to_string(x.cols) + "] tensor");
            x.off = it->second.second + te.at("data_offsets")[0].get<uint64_t>();
            t[name] = x;
        }
    }
    bool has(const std::string& name) const { return t.count(name) != 0; }

    // q4nx rows [r0, r0 + n) of `name` as f32 into out [n, cols]. q_proj's q4nx rows are all q
    // heads then all gate heads ('(g p h) -> (p g h)', p = 2); HF interleaves them per head.
    void rows(const std::string& name, size_t r0, size_t n, size_t hd, float* out) const {
        const T& e = t.at(name);
        if (r0 + n > e.rows) throw std::runtime_error("open_qwen36: bf16 source: " + name + " rows out of range");
        std::ifstream f(e.file, std::ios::binary);
        const size_t G = e.qproj ? e.rows / (2 * hd) : 0;
        std::vector<uint16_t> buf;
        for (size_t r = r0; r < r0 + n;) {
            size_t hf_r = r, run = r0 + n - r;
            if (e.qproj) {
                const size_t p = r / (G * hd), g = (r % (G * hd)) / hd, h = r % hd;
                hf_r = g * 2 * hd + p * hd + h;
                run = std::min(run, hd - h);
            }
            buf.resize(run * e.cols);
            f.seekg(static_cast<std::streamoff>(e.off + hf_r * e.cols * 2));
            f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * 2));
            if (!f) throw std::runtime_error("open_qwen36: bf16 source: short read of " + name);
            float* o = out + (r - r0) * e.cols;
#pragma omp parallel for
            for (long long i = 0; i < static_cast<long long>(buf.size()); ++i) o[i] = bf16_to_f32(buf[static_cast<size_t>(i)]);
            r += run;
        }
    }
};

void Core::apply_thread_budget() const {
    if (omp_threads_ > 0 && omp_get_max_threads() != omp_threads_) omp_set_num_threads(omp_threads_);
}

nlohmann::json derive_config(const GgufFile& g) {
    const std::string arch = g.kv_str("general.architecture");
    auto u = [&](const std::string& k) { return g.kv_u64(arch + "." + k); };
    nlohmann::json c;
    c["model_type"] = arch;
    c["hidden_size"] = u("embedding_length");
    c["num_hidden_layers"] = u("block_count");
    if (g.kv_has(arch + ".vocab_size"))
        c["vocab_size"] = u("vocab_size");
    else
        c["vocab_size"] = g.tensor("token_embd.weight").dims.at(1);
    c["num_attention_heads"] = u("attention.head_count");
    c["num_key_value_heads"] = u("attention.head_count_kv");
    c["intermediate_size"] = u("feed_forward_length");
    if (g.kv_has(arch + ".attention.key_length")) c["head_dim"] = g.kv_u64(arch + ".attention.key_length");
    else c["head_dim"] = u("embedding_length") / u("attention.head_count");
    if (g.kv_has(arch + ".attention.sliding_window"))
        c["sliding_window"] = g.kv_u64(arch + ".attention.sliding_window");
    return c;
}

void Core::log(const std::string& s) const {
    if (cfg_.verbose) std::fprintf(stderr, "open_qwen36: %s\n", s.c_str());
}

Core::Core(const CoreConfig& cfg, xrt::device* dev) : cfg_(cfg) {
    omp_threads_ = static_cast<int>(omp_thread_budget());
    apply_thread_budget();
    if (g_omp_policy_too_late)
        std::fprintf(stderr, "open_qwen36: WARNING: VCOMP140.DLL was already loaded before "
                             "OMP_WAIT_POLICY could be set, so the host workers will SPIN through "
                             "every dispatch and the prefill will be ~20%% slower. Link with "
                             "/DELAYLOAD:VCOMP140.DLL, or set OMP_WAIT_POLICY=PASSIVE in the "
                             "environment before starting.\n");
    log("host threads: " + std::to_string(omp_get_max_threads()) + " (OMP_WAIT_POLICY=" +
        std::string(std::getenv("OMP_WAIT_POLICY") ? std::getenv("OMP_WAIT_POLICY") : "unset") + ")");
    // ---- the kernel set's manifest, and the model it must agree with
    man_ = Manifest::load((fs::path(cfg_.kernel_dir) / "manifest.json").string());
    // OPEN-PREFILL-MODE: swapped in before the kernel list and the packing, so the other route costs nothing
    if (cfg_.prefill_mode != "fast" && cfg_.prefill_mode != "lean")
        throw std::runtime_error("open_qwen36: prefill mode '" + cfg_.prefill_mode + "' is not fast or lean");
    bool has_lean = false;
    for (const auto& [name, t] : man_.layer_types) has_lean = has_lean || t.gemm_block_variants.count("lean");
    man_.select_prefill_route(cfg_.prefill_mode == "lean" ? "lean" : "");
    if (has_lean) prefill_route_ = cfg_.prefill_mode;
    else if (cfg_.prefill_mode == "lean") log("this kernel set has one prefill route; lean changes nothing");
    fs::path md(cfg_.model_dir);
    // ---- the weight file: the shipped container (model.q4nx) or a GGUF
    // (model.gguf, the direct path). When both exist the shipped container
    // wins -- it is the curated conversion.
    const bool have_q4nx = fs::exists(md / "model.q4nx");
    const bool have_gguf = fs::exists(md / "model.gguf");
    if (!have_q4nx && !have_gguf)
        throw std::runtime_error("open_qwen36: neither model.q4nx nor model.gguf in " + cfg_.model_dir);
    gguf_ = have_gguf && !have_q4nx;
    w_ = gguf_ ? man_.gguf.get() : &man_;
    if (gguf_ && !w_)
        throw std::runtime_error("open_qwen36: model.gguf needs the kernel set's GGUF-direct build "
                                 "(manifest.json has no gguf section; re-export the kernels)");
    // ---- the model's config: from the model dir when it ships one, else
    // derived from the GGUF metadata (the shapes the manifest checks are all
    // there); anything GGUF does not carry is refused with its name.
    if (gguf_)
        file_ = std::make_unique<GgufFile>((md / "model.gguf").string());
    nlohmann::json j;
    std::ifstream cf(md / "config.json");
    // #24: config.json is optional for a GGUF -- the metadata carries it.
    if (cf) {
        j = nlohmann::json::parse(cf, nullptr, false);
        if (!j.is_object()) throw std::runtime_error("open_qwen36: bad config.json in " + cfg_.model_dir);
        if (gguf_) {
            const auto derived = derive_config(*static_cast<GgufFile*>(file_.get()));
            for (const auto& [key, value] : derived.items())
                if (!j.contains(key)) j[key] = value;
        }
    } else if (gguf_) {
        j = derive_config(*static_cast<GgufFile*>(file_.get()));
        log("config.json absent; derived from the GGUF metadata");
    } else {
        throw std::runtime_error("open_qwen36: no config.json in " + cfg_.model_dir);
    }
    w_->check_model(j, md.filename().string());
    // The VLM bits: the image token the app expands per merged patch, and how the
    // rotary pairs split over (t, h, w). Absent on text-only models.
    image_token_id_ = j.value("image_token_id", -1);
    if (image_token_id_ < 0) {
        // Qwen3-VL-4B-Instruct-NPU2's config.json omits it where Qwen2.5-VL's carries it.
        // The tokenizer has it though, as an added token, so read it there rather than
        // hardcode 151655 the way the closed adapter does - a number that is right for
        // one model and silently wrong for the next.
        std::ifstream tf(md / "tokenizer.json");
        if (tf) {
            auto t = nlohmann::json::parse(tf, nullptr, false);
            if (t.is_object() && t.contains("added_tokens") && t["added_tokens"].is_array()) {
                for (const auto& a : t["added_tokens"]) {
                    if (a.is_object() && a.value("content", std::string()) == "<|image_pad|>") {
                        image_token_id_ = a.value("id", -1);
                        if (image_token_id_ >= 0 && cfg_.verbose)
                            std::fprintf(stderr,
                                         "open_qwen36: config.json has no image_token_id; <|image_pad|> is %d in "
                                         "this model's tokenizer\n", image_token_id_);
                        break;
                    }
                }
            }
        }
    }
    // `rope_parameters` is what transformers calls this now; a container converted before
    // the rename carries `rope_scaling` instead, and Qwen2.5-VL's (transformers 4.41) is
    // one of those. Reading only the new name left the engine reporting that a config with
    // a perfectly good mrope_section had none.
    const char* rope_key = j.contains("rope_parameters") && j["rope_parameters"].is_object() ? "rope_parameters"
                         : (j.contains("rope_scaling") && j["rope_scaling"].is_object() ? "rope_scaling" : nullptr);
    // Qwen3-VL's container carries neither, and config.json cannot be edited to add them:
    // the downloader compares every registry-listed file against a remote manifest's byte
    // count and re-pulls anything that differs. vision.json is not in that list, so that
    // is where oflm-add writes what the container omits.
    nlohmann::json side;
    if (!rope_key) {
        std::ifstream sf(md / "vision.json");
        if (sf) {
            auto parsed = nlohmann::json::parse(sf, nullptr, false);
            if (parsed.is_object() && parsed.contains("rope_scaling") && parsed["rope_scaling"].is_object()) {
                side = parsed;
                rope_key = "rope_scaling";
            }
        }
    }
    if (rope_key) {
        const auto& rp = side.is_object() && side.contains(rope_key) ? side[rope_key] : j[rope_key];
        if (rp.contains("mrope_section") && rp["mrope_section"].is_array() && rp["mrope_section"].size() == 3) {
            size_t sum = 0;
            for (const auto& v : rp["mrope_section"]) { mrope_section_.push_back(v.get<int>()); sum += v.get<int>(); }
            mrope_interleaved_ = rp.value("mrope_interleaved", false);
            if (sum != w_->rotary_dim / 2)
                throw std::runtime_error("open_qwen36: mrope_section sums to " + std::to_string(sum) + ", not the " +
                                         std::to_string(w_->rotary_dim / 2) + " rotary pairs");
        }
    }
    if (!gguf_ && !file_) {
        file_ = std::make_unique<Q4nxFile>((md / "model.q4nx").string());
    }
    if (const char* env = std::getenv("OFLM_OPEN_BF16_SRC"); fs::exists(md / "bf16_map.json") && !(env && std::string(env) == "0")) {
        hf_bf16_ = std::make_shared<HfBf16>(md / "bf16_map.json");
        log("bf16 source: " + std::to_string(hf_bf16_->t.size()) + " tensors (bf16_map.json); the prompt GEMMs use exact bf16");
    }
    int total = static_cast<int>(w_->layers.size());
    nl_ = cfg_.num_layers > 0 && cfg_.num_layers < total ? cfg_.num_layers : total;
    types_.resize(nl_);
    for (int l = 0; l < nl_; ++l) types_[l] = &w_->layer_type(l);
    int nattn = 0;
    for (int l = 0; l < nl_; ++l) nattn += is_attention_layer(l);
    log("model " + md.filename().string() + " (" + man_.family + ", " + man_.spec_hash.substr(0, 19) + "): " +
        std::to_string(nl_) + " of " + std::to_string(total) + " layers, " + std::to_string(nattn) +
        " attention, context capacity " + std::to_string(cfg_.max_ctx));

    // ---- device, contexts, kernels (only the kernels the running layers' programs and the tail name)
    if (dev) {
        dev_ = dev;
    } else {
        owned_dev_ = std::make_unique<xrt::device>(0u);
        dev_ = owned_dev_.get();
    }
    std::map<std::string, bool> wanted;
    for (int l = 0; l < nl_; ++l) {
        for (const auto& s : types_[l]->program) wanted[s.kernel] = true;
        // 0167/#32: the GEMM-route block's 5 GEMM kernels, plus its attn_kernel
        // (the attention half, driven directly by Core rather than via a
        // Step -- see manifest.hpp's GemmBlockProgram; a layer type with its
        // own sliding window names its own, e.g. Gemma 3's dxB_local) which
        // the manifest parser already required to exist whenever gemm_block
        // is present.
        for (const auto& s : types_[l]->gemm_block.program) wanted[s.kernel] = true;
        for (const auto& s : types_[l]->gemm_block.shared_program) wanted[s.kernel] = true;
        for (const auto& s : types_[l]->gemm_block.ffn_program) wanted[s.kernel] = true;
        if (types_[l]->gemm_block.t && types_[l]->gemm_block.kind == "dense") wanted[types_[l]->gemm_block.attn_kernel] = true;
        if (!types_[l]->gemm_block.moe_kernel.empty()) wanted[types_[l]->gemm_block.moe_kernel] = true;
        for (const auto& [slots, k] : types_[l]->gemm_block.moe_batch.kernels) wanted[k] = true;
        for (const auto& [rows, k] : types_[l]->gemm_block.attn_block.kernels_s) wanted[k] = true;
        for (const auto& [rows, k] : types_[l]->gemm_block.attn_block.kernels_pv) wanted[k] = true;
    }
    for (const auto& s : w_->tail) wanted[s.kernel] = true;
    if (!man_.gemm_dit.is_null())   // dit_on_ is decided later; loading an unused kernel is harmless
        for (const auto& [shape, k] : man_.gemm_dit["kernels"].items()) wanted[k.get<std::string>()] = true;
    if (man_.gemm_dit.contains("ffn"))
        for (const auto& [f, v] : man_.gemm_dit["ffn"].items())
            if (v.is_string()) wanted[v.get<std::string>()] = true;
    if (man_.gemm_dit.contains("kernels2"))
        for (const auto& [shape, k] : man_.gemm_dit["kernels2"].items()) wanted[k.get<std::string>()] = true;
    if (man_.gemm_dit.contains("decode"))
        for (const auto& [shape, v] : man_.gemm_dit["decode"]["split"].items()) wanted[v["kernel"].get<std::string>()] = true;
    for (const char* tier : {"small", "tiny"})
        if (man_.gemm_dit.contains(tier))
            for (const char* key : {"kernels2", "ffn"})
                for (const auto& [shape, k] : man_.gemm_dit[tier][key].items()) wanted[k.get<std::string>()] = true;
    if (man_.gemm_dit.contains("fa"))
        for (const char* key : {"lk", "nc", "ncb"})
            if (man_.gemm_dit["fa"].contains(key))
                for (const auto& [lk, k] : man_.gemm_dit["fa"][key].items()) wanted[k.get<std::string>()] = true;
    for (const auto& [name, d] : w_->kernels)
        if (wanted.count(name)) load_kernel(name, d);
    logits_host_.assign(w_->vocab, 0.f);

    // 0167/#32: the GEMM-route block size every loaded layer type agrees on.
    // Disagreement (a mixed dense_local/dense manifest where only one carries
    // a gemm_block program) or no gemm_block program anywhere both read as
    // "unsupported" (0), never a guess at which layer type's T applies --
    // step_gemm_block() refuses. gemm_block_t_ == 0 is not an error (most
    // kernel sets have no gemm_block program), but it silently means every
    // prefill runs one token at a time even on a model that DOES have one, if
    // the kernel_dir actually loaded is a stale copy without it
    // (Engine::find_kernels() prefers <model>/open_kernels over
    // OFLM_OPEN_KERNELS_DIR/OFLM_XCLBIN_PATH -- this project lost real time to
    // exactly that before this log line existed).
    gemm_block_t_ = nl_ > 0 ? types_[0]->gemm_block.t : 0;
    for (int l = 1; l < nl_; ++l)
        if (types_[l]->gemm_block.t != gemm_block_t_) { gemm_block_t_ = 0; break; }
    log("block prefill route: T = " + std::to_string(gemm_block_t_) +
        (gemm_block_t_ ? (prefill_route_.empty() ? "" : " (" + prefill_route_ + ")")
                       : " (no gemm_block program in this kernel set, or its layer types disagree)"));
    // the token-batched expert kernel: every stream's slot count must be what the manifest says
    if (const char* env = std::getenv("OFLM_OPEN_MOE_BATCH")) moe_batch_on_ = std::string(env) != "0";
    if (const char* env = std::getenv("OFLM_OPEN_ATTN_BLOCK")) attn_block_on_ = std::string(env) != "0";
    if (const char* env = std::getenv("OFLM_OPEN_LAYER_MAJOR")) layer_major_on_ = std::string(env) != "0";
    if (const char* env = std::getenv("OFLM_OPEN_STAGE_MAJOR")) stage_major_on_ = std::string(env) != "0";
    dit_on_ = !man_.gemm_dit.is_null();
    dit_hs_ = dit_on_ && man_.gemm_dit.value("hs", false);
    dit_b4_ = dit_hs_ && man_.gemm_dit.value("b4", false);
    if (const char* env = std::getenv("OFLM_OPEN_PIPE")) pipe_on_ = std::string(env) != "0";
    if (const char* env = std::getenv("OFLM_OPEN_GEMM_DIT"); env && std::string(env) == "0") dit_on_ = false;
    if (dit_on_ && !stage_major_on_)
        throw std::runtime_error("open_qwen36: the dit GEMMs run only stage-major (OFLM_OPEN_GEMM_DIT=0 for the block-major schedule)");
    dispatch_log_ = std::getenv("OFLM_OPEN_DISPATCH_LOG") != nullptr;
    // Stage 2.5 Gate A: per-dxB-dispatch phase CSV (layer,pos,patch,prep,submit,wait,read).
    // Measurement only; off (and free) unless the env names a file.
    if (const char* env = std::getenv("OFLM_OPEN_DXB_LOG")) {
        dxb_log_ = std::fopen(env, "w");
        if (dxb_log_) {
            std::fprintf(dxb_log_, "kind,layer,pos,patch_ms,prep_ms,submit_ms,wait_ms,read_ms\n");
            log("dxB phase log: " + std::string(env));
        }
    }
    // Stage 2.5 Gate B: opt-in host attention for the dense block route.
    if (const char* env = std::getenv("OFLM_OPEN_HOST_ATTN")) host_attn_on_ = std::string(env) != "0";
    if (host_attn_on_)
        log("gemm-block attention: HOST (OFLM_OPEN_HOST_ATTN=1; K2 dense shape only, "
            "no qk-norm/gate, geometry checked against the manifest)");
    // Stage 2.6 Gate B: opt-in host DECODE route -- the whole decode step through
    // the 2.5 gemm-block chain at T=1 (host attention forced for those calls
    // only) instead of the fused per-layer dx dispatch. Decode-only: the
    // gemm-block prefill route's own selector stays OFLM_OPEN_HOST_ATTN.
    if (const char* env = std::getenv("OFLM_OPEN_HOST_ATTN_DECODE")) host_attn_decode_on_ = std::string(env) != "0";
    if (const char* env = std::getenv("OFLM_OPEN_CPU_DECODE")) cpu_decode_on_ = std::string(env) == "1";
    if (host_attn_decode_on_)
        log("decode route: HOST (OFLM_OPEN_HOST_ATTN_DECODE=1; the 2.5 gemm-block chain at T=1 "
            "with host attention; the NPU GEMMs run padded at the kernel set's compiled T)");
    // Stage 2.7 Gate A: the runtime decode route selector. The env above only
    // picks the INITIAL route (compat/debug); set_decode_route() overrides it
    // at any moment, which is what the one-process mid-stream switch needs.
    // Stage 2.10 P6: with k2 promoted into FAST_ATTENTION (the stage 2.9
    // measurement: decode slope 1.488 -> 0.0251 ms/position-token, 59.2x), the
    // fast NPU route wins at every measured window -- 65 ms at position 512
    // against the ~1073 ms host floor, 463.7 ms at 16382 against the host
    // curve's ~3.2 s and the slow path's ~24.5 s -- so Auto is NPU-first
    // throughout the validated 0-16383 window and the 2.6 crossover thresholds
    // (896 default / 640 at <= 4 OMP threads) are dead. Host stays the explicit
    // oracle/fallback route; windows beyond 16K need a re-derived crossover
    // after 32K/64K measurements. OFLM_DECODE_AUTO_THRESHOLD still forces the
    // old position split for experiments.
    decode_route_ = host_attn_decode_on_ ? DecodeRoute::Host : DecodeRoute::Npu;
    decode_auto_threshold_ = std::numeric_limits<size_t>::max();
    if (const char* env = std::getenv("OFLM_DECODE_AUTO_THRESHOLD"))
        decode_auto_threshold_ = static_cast<size_t>(std::strtoull(env, nullptr, 10));
    log("decode route selector: runtime (npu|host|auto; initial " + decode_route_at(0) +
        ", auto npu-first [fast attention promoted, stage 2.10;"
        " OFLM_DECODE_AUTO_THRESHOLD forces a position split])");
    moe_redispatch_ = std::getenv("OFLM_OPEN_MOE_REDISPATCH") != nullptr;
    route_check_ = std::getenv("OFLM_ROUTE_CHECK") != nullptr;
    if (const char* env = std::getenv("OFLM_OPEN_ROUTE_SENTINEL")) route_sentinel_ = std::atoi(env) != 0;
    if (const char* env = std::getenv("OFLM_OPEN_READ_FENCE")) g_read_fence = std::atoi(env) != 0;
    if (const char* env = std::getenv("OFLM_OPEN_STEP_TRACE")) step_trace_ = std::atoi(env);
    if (const char* env = std::getenv("OFLM_OPEN_SPIN_US")) spin_us_ = std::atoi(env);
    // OPEN-DECODE-PIPELINE. Level 1 keeps the next layer's first dispatch in flight behind
    // this layer's last one only when the two share a hardware context; level 2 crosses
    // contexts as well (see step_impl for the ordering invariant and what proves it).
    submit_ahead_ = 1;
    if (const char* env = std::getenv("OFLM_OPEN_SUBMIT_AHEAD")) submit_ahead_ = std::atoi(env);
    if (submit_ahead_ >= 2)
        std::fprintf(stderr, "open_qwen36: WARNING: OFLM_OPEN_SUBMIT_AHEAD=2 queues a dispatch from a second "
                             "hardware context while the first still has one in flight, and this HANGS the array "
                             "(ERT state 8 on the next ax dispatch, twice in two runs). It is kept only as the "
                             "probe that established that; use 1.\n");
    log(std::string("decode route: ") +
        (submit_ahead_ <= 0 ? "serial dispatches (OFLM_OPEN_SUBMIT_AHEAD=0)"
                            : submit_ahead_ == 1 ? "submit-ahead within a hardware context"
                                                 : "submit-ahead across hardware contexts (UNSAFE)") +
        (step_trace_ ? ", step trace on" : "") +
        (spin_us_ > 0 ? ", polling " + std::to_string(spin_us_) + " us before blocking on a dispatch" : ""));
    bool any_batch = false;
    for (int l = 0; l < nl_; ++l)
        for (const auto& [slots, k] : types_[l]->gemm_block.moe_batch.kernels) {
            any_batch = true;
            if (kerns_.at(k).slots != slots)
                throw std::runtime_error("open_qwen36: " + k + " carries " + std::to_string(kerns_.at(k).slots) +
                                         " expert slots, the manifest says " + std::to_string(slots));
        }
    bool any_moe = false;
    for (const auto& t : types_) any_moe = any_moe || !t->gemm_block.moe_kernel.empty();
    if (gemm_block_t_ && any_moe)
        log(std::string("token-batched expert kernel: ") +
            (any_batch ? (moe_batch_on_ ? "on" : "off (OFLM_OPEN_MOE_BATCH=0)") : "not in this kernel set (mx per token)"));
    bool any_attn = false;
    for (const auto& t : types_) any_attn = any_attn || t->gemm_block.attn_block.present();
    if (gemm_block_t_)
        log(std::string("block attention on the NPU: ") +
            (any_attn ? (attn_block_on_ ? "on" : "off (OFLM_OPEN_ATTN_BLOCK=0)") : "not in this kernel set (attention on the host)"));
    if (gemm_block_t_)
        log(std::string("prefill schedule: ") +
            (layer_major_ok() ? std::string(any_moe ? "layer-major (the whole prompt through each layer, one MoE pass a layer)"
                                                    : "layer-major (the whole prompt through each layer)") +
                                    (stage_major_on_ ? ", stage-major inside it (the next block's GEMM overlaps the host)"
                                                     : ", block at a time inside it (OFLM_OPEN_STAGE_MAJOR=0)")
                              : std::string(layer_major_on_ ? "block-major (a MoE kind's layer types only)"
                                                            : "block-major (OFLM_OPEN_LAYER_MAJOR=0)")));
}

bool Core::layer_major_ok() const {
    if (!layer_major_on_ || !gemm_block_t_) return false;
    for (int l = 0; l < nl_; ++l) {
        const std::string& kind = types_[l]->gemm_block.kind;
        if (kind != "linear" && kind != "full") return false;
    }
    return nl_ > 0;
}

Core::~Core() {
    if (dxb_log_) {                      // Stage 2.5 Gate A phase log
        std::fflush(dxb_log_);
        std::fclose(dxb_log_);
        dxb_log_ = nullptr;
    }
}

xrt::hw_context& Core::context(const std::string& name) {
    auto it = ctxs_.find(name);
    if (it != ctxs_.end()) return *it->second;
    fs::path p = fs::path(cfg_.kernel_dir) / w_->contexts.at(name);
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing kernel " + p.string());
    xrt::xclbin xcl(p.string());
    auto uuid = dev_->register_xclbin(xcl);
    auto ctx = std::make_unique<xrt::hw_context>(*dev_, uuid);
    return *(ctxs_[name] = std::move(ctx));
}

void Core::load_kernel(const std::string& name, const KernelDesc& d) {
    fs::path p = fs::path(cfg_.kernel_dir) / d.insts;
    if (!fs::exists(p)) throw std::runtime_error("open_qwen36: missing instruction stream " + p.string());
    Kern& k = kerns_[name];
    k.name = name;
    k.patch = d.patch;
    k.ctx = d.context;
    k.k = std::make_unique<xrt::kernel>(context(d.context), "MLIR_AIE");
    auto insts = read_file(p);
    if (insts.empty() || insts.size() % 4) throw std::runtime_error("open_qwen36: " + p.string() + " is not word-sized");
    k.words.resize(insts.size() / 4);
    std::memcpy(k.words.data(), insts.data(), insts.size());
    k.instr = std::make_unique<xrt::bo>(*dev_, insts.size(), xrt::bo::flags::cacheable, k.k->group_id(1));
    std::memcpy(k.instr->map<void*>(), insts.data(), insts.size());
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (d.patch == "moeroute2") k.moe2 = stream_patch::moe2_table(k.words, name, w_->moe);
    else if (d.patch == "moebatch") {
        // every expert fill is a placeholder (slot s compiled as expert s), so the table is
        // moeroute2's with the whole expert range as slots; the stream's length is its highest
        stream_patch::MoeGeometry g = w_->moe;
        g.topk = g.experts;
        k.moe2 = stream_patch::moe2_table(k.words, name, g);
        for (const auto& p : k.moe2) k.slots = std::max(k.slots, static_cast<size_t>((p.slot & 0xff) + 1));
    } else if (d.patch == "attnpos") {
        k.attn = stream_patch::attn_table(k.words, name, w_->attn);
        k.geom = w_->attn;
        k.geom.window = d.window;
        k.geom.rb = d.rb;
    }
}

xrt::bo Core::alloc(size_t bytes, const uint8_t* init, size_t init_bytes) {
    xrt::bo bo = xrt::ext::bo(*dev_, padup(bytes));
    auto* m = bo.map<uint8_t*>();
    std::memset(m, 0, padup(bytes));
    if (init) std::memcpy(m, init, init_bytes);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return bo;
}

void Core::load_weights(const std::function<void(int, int)>& progress) {
    auto t0 = std::chrono::steady_clock::now();
    pools_.clear(); consts_.clear(); act_.clear(); state_.clear(); globals_.clear();
    gemm_w_.clear(); hc_.clear();
    ln_w_bf16_.clear(); post_ln_w_bf16_.clear(); pre_ffn_w_.clear(); post_ffn_w_.clear();
    pools_.reserve(nl_); consts_.reserve(nl_); act_.reserve(nl_); state_.reserve(nl_);
    if (gemm_block_t_) {
        ln_w_bf16_.resize(nl_); post_ln_w_bf16_.resize(nl_);
        pre_ffn_w_.resize(nl_); post_ffn_w_.resize(nl_);
        hc_.resize(nl_);
    }
    // the block route's per-layer weight buffers: each a contiguous run of pack ops of
    // the freshly packed host bytes (pool or consts), copied before that buffer's upload
    auto build_weights = [&](const LayerType& lt, int l, const std::string& from, const uint8_t* host) {
        if (!(gemm_block_t_ && lt.gemm_block.t)) return;
        std::map<std::string, GemmWeight> all = lt.gemm_block.weights;
        all.insert(lt.gemm_block.shared_weights.begin(), lt.gemm_block.shared_weights.end());
        all.insert(lt.gemm_block.ffn_weights.begin(), lt.gemm_block.ffn_weights.end());
        for (const auto& [name, gw] : all) {
            if (gw.from != from) continue;
            auto& v = gemm_w_[name];
            if (v.size() != static_cast<size_t>(nl_)) v.resize(nl_);
            if (from == "pack") {
                size_t bytes = 0;
for (const PackOp& o : gw.pack) bytes = std::max<size_t>(bytes, o.dst + pools::op_bytes(o, w_->chunk_bytes));
                xrt::bo w = xrt::ext::bo(*dev_, padup(bytes));
                std::memset(w.map<uint8_t*>(), 0, padup(bytes));
                for (const PackOp& o : gw.pack) pools::apply(o, *file_, l, w.map<uint8_t*>(), bytes, w_->chunk_bytes);
                w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                v[l] = std::move(w);
                continue;
            }
            size_t off0 = 0, total = 0;
            for (size_t i = 0; i < gw.ops.size(); ++i) {
                auto [off, bytes] = op_region(lt, from, gw.ops[i]);
                if (i == 0) off0 = off;
                else if (off != off0 + total)
                    throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": the " + from + " ops of " + name +
                                             " are not contiguous -- the route needs one memcpy per weight buffer");
                total += bytes;
            }
            if (dit_on_) {
                // dit_gemm: the band-order q4_1 bytes repacked as bfp16 [K, Npad] (9 bytes per 8)
                size_t n = 0, k = 0;
                for (const auto* prog : {&lt.gemm_block.program, &lt.gemm_block.ffn_program, &lt.gemm_block.shared_program})
                    for (const Step& st : *prog)
                        if (!st.args.empty() && st.args[0] == name) {
                            if (st.split) throw std::runtime_error("open_qwen36: dit GEMM: " + name + " is a split (q8) step");
                            const std::string& kn = st.kernel;          // gemm_n<N>_k<K>
                            n = std::stoul(kn.substr(6, kn.find("_k") - 6));
                            k = std::stoul(kn.substr(kn.find("_k") + 2));
                        }
                if (!n || n * k / 8192 * w_->chunk_bytes != total || w_->chunk_bytes != 5120)
                    throw std::runtime_error("open_qwen36: dit GEMM: no [N, K] for weight " + name + " or not a 5120 B chunk pool");
                const size_t npad = man_.gemm_dit["npad"][std::to_string(n)].get<size_t>();
                xrt::bo w = xrt::ext::bo(*dev_, padup(dit_b4_ ? npad / 128 * (k / 64) * kB4Tile : k * npad / 8 * 9));
                // the FFN up|gate weight ([up ff | gate ff] rows) for the SwiGLU epilogue: every
                // 128-column tile j is [gate 64j.. | up 64j..]
                std::vector<int64_t> cm;
                if (man_.gemm_dit.contains("ffn") && dit_ffn_on() && !lt.gemm_block.ffn_program.empty() &&
                    lt.gemm_block.ffn_program[0].args[0] == name) {
                    const size_t ff = man_.gemm_dit["ffn"]["ff"].get<size_t>();
                    if (n != 2 * ff || npad != n) throw std::runtime_error("open_qwen36: dit FFN: up|gate is not [2 ff]");
                    cm.resize(npad);
                    for (size_t c = 0; c < npad; ++c) {
                        const size_t j = c / 128, w = c % 128;
                        cm[c] = static_cast<int64_t>(w < 64 ? ff + 64 * j + w : 64 * j + (w - 64));
                    }
                }
                const auto& lops = from == "pool" ? lt.pool : lt.consts;
                bool bf = hf_bf16_ != nullptr;
                for (size_t i : gw.ops) bf = bf && hf_bf16_->has(layer_tensor(lops[i].tensor, l));
                if (bf) {
                    std::vector<float> W(n * k);
                    const size_t hd = w_->hf_config_check.at("head_dim").get<size_t>();
                    size_t row = 0;
                    for (size_t i : gw.ops) {
                        const PackOp& o = lops[i];
                        const size_t ncol = o.in_dim / 256, rows_op = o.nch / (o.in_dim / 128) * 64;
                        if (o.op != "std_perm" || o.in_dim != k || o.chunk0 % ncol || row + rows_op > n || !o.split.empty())
                            throw std::runtime_error("open_qwen36: bf16 source: " + name + ": op " + std::to_string(i) +
                                                     " is not a whole-row std_perm band of a [" + std::to_string(n) + ", " +
                                                     std::to_string(k) + "] weight");
                        hf_bf16_->rows(layer_tensor(o.tensor, l), 32 * (o.chunk0 / ncol), rows_op, hd, W.data() + row * k);
                        row += rows_op;
                    }
                    if (row != n) throw std::runtime_error("open_qwen36: bf16 source: " + name + " covers " + std::to_string(row) +
                                                           " of " + std::to_string(n) + " rows");
                    if (const char* chk = std::getenv("OFLM_OPEN_BF16_CHECK"); host && chk && std::atoi(chk) > l) {
                        // the q4 route's weight for comparison: relative Frobenius difference
                        std::vector<float> Q(n * k);
                        dit_decode_q4(host + off0, n, k, Q.data());
                        double num = 0, den = 0;
                        for (size_t i = 0; i < n * k; ++i) { const double d = Q[i] - W[i]; num += d * d; den += double(W[i]) * W[i]; }
                        log("bf16 check: layer " + std::to_string(l) + " " + name + " [" + std::to_string(n) + ", " +
                            std::to_string(k) + "]: |q4 - bf16| / |bf16| = " + std::to_string(std::sqrt(num / den)));
                    }
                    if (dit_b4_) dit_encode_b4(W.data(), n, k, npad, w.map<uint8_t*>(), cm.empty() ? nullptr : cm.data());
                    else dit_encode(W.data(), n, k, npad, w.map<uint8_t*>(), cm.empty() ? nullptr : cm.data(), dit_hs_);
                } else {
                    if (!host) throw std::runtime_error("open_qwen36: prompt only: " + name + " is not in the bf16 source");
                    if (dit_b4_) {
                        std::vector<float> Wq(n * k);
                        dit_decode_q4(host + off0, n, k, Wq.data());
                        dit_encode_b4(Wq.data(), n, k, npad, w.map<uint8_t*>(), cm.empty() ? nullptr : cm.data());
                    } else {
                        dit_pack_region(host + off0, n, k, npad, w.map<uint8_t*>(), cm.empty() ? nullptr : cm.data(), dit_hs_);
                    }
                }
                w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                v[l] = std::move(w);
                continue;
            }
            if (!host) throw std::runtime_error("open_qwen36: prompt only: " + name + " needs the q4 pool");
            xrt::bo w = xrt::ext::bo(*dev_, padup(total));
            std::memcpy(w.map<uint8_t*>(), host + off0, total);
            w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            v[l] = std::move(w);
        }
    };
    const bool skip_pools = cfg_.prompt_only && hf_bf16_ && dit_on_;
    if (skip_pools) log("prompt only: no q4 pools, no lm_head (the GEMM weights come from the bf16 source)");
    // OFLM_OPEN_CPU_DECODE=1 on an int4 dit set: decode reads the int4 GEMM weights, so each
    // layer's q4 pool is packed in host memory only to build them, then freed (the 27B: 31.8 ->
    // ~17 GB). The fused NPU decode route needs the pools: OFLM_OPEN_KEEP_POOLS=1 keeps them.
    drop_pools_ = !skip_pools && cpu_decode_on_ && dit_on_ && dit_b4_ && gemm_block_t_ &&
                  !(std::getenv("OFLM_OPEN_KEEP_POOLS") && *std::getenv("OFLM_OPEN_KEEP_POOLS") == '1');
    if (drop_pools_) log("CPU decode: the q4 pools only feed the int4 GEMM weights, then go");
    std::vector<uint8_t> pool_tmp;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (skip_pools) {
            build_weights(lt, l, "pool", nullptr);
            pools_.emplace_back();
        } else if (drop_pools_) {
            pool_tmp.assign(w_->pool_bytes, 0);
            pools::pack_pool(*w_, lt, *file_, l, pool_tmp.data());
            build_weights(lt, l, "pool", pool_tmp.data());
            pools_.emplace_back();
        } else {
            xrt::bo pool = xrt::ext::bo(*dev_, w_->pool_bytes);
            uint8_t* pool_host = pool.map<uint8_t*>();
            pools::pack_pool(*w_, lt, *file_, l, pool_host);
            build_weights(lt, l, "pool", pool_host);
            pool.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            pools_.push_back(std::move(pool));
        }
        xrt::bo c = xrt::ext::bo(*dev_, padup(lt.consts_bytes));
        std::memset(c.map<uint8_t*>(), 0, padup(lt.consts_bytes));
        uint8_t* c_host = c.map<uint8_t*>();
        pools::pack_consts(*w_, lt, *file_, l, c_host);
        build_weights(lt, l, "consts", c_host);
        build_weights(lt, l, "pack", nullptr);
        if (gemm_block_t_ && lt.gemm_block.t) {
            if (lt.gemm_block.kind == "dense") {
                // the dense route's host RMSNorm reads the two norm weights as bf16;
                // dense.py's consts plan puts input_layernorm at byte 0 and
                // post_attention_layernorm right after it (ELN = hidden * 2 each)
                ln_w_bf16_[l].resize(w_->hidden);
                post_ln_w_bf16_[l].resize(w_->hidden);
                std::memcpy(ln_w_bf16_[l].data(), c_host, w_->hidden * 2);
                std::memcpy(post_ln_w_bf16_[l].data(), c_host + w_->hidden * 2, w_->hidden * 2);
                if (lt.gemm_block.sandwich) {
                    // Read straight from the file by tensor name suffix (the MoE kinds'
                    // pattern, `const_tensor()`), not sliced from packed consts bytes: these
                    // two are additional to the plain chain's two, and pack_plan only ever
                    // packs them into consts.CD_PREFFN/CD_POSTFFN when sandwich_norms is set,
                    // so their offsets are family-specific in a way the other two are not.
                    pre_ffn_w_[l] = file_->bf16(const_tensor(lt, "pre_feedforward_layernorm.weight", l));
                    post_ffn_w_[l] = file_->bf16(const_tensor(lt, "post_feedforward_layernorm.weight", l));
                    if (pre_ffn_w_[l].size() != w_->hidden || post_ffn_w_[l].size() != w_->hidden)
                        throw std::runtime_error("open_qwen36: layer " + std::to_string(l) +
                                                 ": sandwich norm weight is not [hidden]");
                }
                // The attention products' host half (OPEN-PREFILL-ATTN) reads the q/k norms as
                // f32, from the file by name as the full layers' does. The consts plan `put`s the
                // same bf16 bytes for the dxB kernel, so both routes norm with one weight.
                const AttnBlock& ab = lt.gemm_block.attn_block;
                if (ab.prep == "qknorm_rope") {
                    HostConsts& h = hc_[l];
                    h.qn = file_->bf16(const_tensor(lt, "q_norm.weight", l));
                    h.kn = file_->bf16(const_tensor(lt, "k_norm.weight", l));
                    if (h.qn.size() != ab.hd || h.kn.size() != ab.hd)
                        throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": q_norm / k_norm is not [" +
                                                 std::to_string(ab.hd) + "], the head dim the attention products were built for");
                }
            } else {
                // the MoE kinds' host stages read their small tensors straight from the
                // file, by the names the consts plan carries (no consts layout knowledge here)
                const GemmBlockProgram& gb = lt.gemm_block;
                HostConsts& h = hc_[l];
                auto* qf = dynamic_cast<Q4nxFile*>(file_.get());
                if (!qf)
                    throw std::runtime_error("open_qwen36: GGUF-direct block prefill is not implemented for MoE models");
                auto bf = [&](const char* suffix) { return qf->bf16(const_tensor(lt, suffix, l)); };
                auto want = [&](const std::vector<float>& v, size_t n, const char* what) {
                    if (v.size() != n)
                        throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": " + what + " has " +
                                                 std::to_string(v.size()) + " values, the route wants " + std::to_string(n));
                };
                h.ln = bf("input_layernorm.weight");
                h.postln = bf("post_attention_layernorm.weight");
                want(h.ln, w_->hidden, "input_layernorm");
                want(h.postln, w_->hidden, "post_attention_layernorm");
                if (!gb.has_dense_ffn()) {
                    h.router = bf("moe_router.weight");
                    h.sgw = bf("shared_expert_gate.weight");
                    want(h.router, w_->hidden * w_->moe.experts, "moe_router");
                    want(h.sgw, w_->hidden, "shared_expert_gate");
                }
                if (gb.kind == "linear") {
                    // The host reads alpha / beta as [hidden, lanes]. The 35B's container stores
                    // exactly that; Qwen3.5's stores a bf16 [heads, hidden] copy beside the q8
                    // projection, which its consts plan transposes -- so transpose it here too.
                    auto ab = [&](const std::string& stem, std::vector<float>& out) -> size_t {
                        const std::string copy = stem + ".bf16.weight";
                        const bool t = std::any_of(lt.consts.begin(), lt.consts.end(), [&](const PackOp& op) {
                            return op.op == "transpose" && op.tensor.size() >= copy.size() &&
                                   op.tensor.compare(op.tensor.size() - copy.size(), copy.size(), copy) == 0;
                        });
                        const std::string name = const_tensor(lt, t ? copy : stem + ".weight", l);
                        const auto& shape = file_->meta(name).shape;
                        const size_t hid_dim = t ? 1 : 0;
                        if (shape.size() != 2 || shape[hid_dim] != w_->hidden)
                            throw std::runtime_error("open_qwen36: " + name + (t ? " is not [heads, hidden]" : " is not [hidden, lanes]"));
                        const size_t lanes = shape[1 - hid_dim];
                        std::vector<float> w = file_->bf16(name);
                        if (t) {
                            out.assign(w_->hidden * lanes, 0.f);
                            host::transpose(w.data(), lanes, w_->hidden, out.data());   // [lanes, hid] -> [hid, lanes]
                        } else {
                            out = std::move(w);
                        }
                        return lanes;
                    };
                    h.lanes = ab("ssm_alpha_proj", h.Wa);
                    if (ab("ssm_beta_proj", h.Wb) != h.lanes)
                        throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": alpha and beta disagree on lanes");
                    h.A = file_->f32(const_tensor(lt, "ssm_a", l));
                    h.dtb = file_->f32(const_tensor(lt, "ssm_dt.bias", l));
                    h.convw = bf("ssm_conv1d.weight");
                    h.nw = bf("ssm_norm.weight");
                    want(h.Wb, w_->hidden * h.lanes, "ssm_beta_proj");
                    want(h.A, gb.value_heads, "ssm_a");
                    want(h.dtb, gb.value_heads, "ssm_dt.bias");
                    want(h.convw, gb.conv_kernel * gb.qkv_dim, "ssm_conv1d");
                    want(h.nw, gb.head_dim, "ssm_norm");
                } else {
                    h.qn = bf("q_norm.weight");
                    h.kn = bf("k_norm.weight");
                    want(h.qn, gb.hd, "q_norm");
                    want(h.kn, gb.hd, "k_norm");
                }
            }
        }
        c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        consts_.push_back(std::move(c));
        act_.push_back(alloc(lt.act_bytes));
        // A blocked attention kernel (manifest `rb` > 1) streams its padded window up to row
        // pos + rb - 2 (stream_patch::attn_apply), which at the last position is past the
        // context capacity by rb - 2 rows. The slack is allocated -- and zeroed, so the rows
        // the kernel masks multiply as exact zeros -- rather than clamped: a shorter stream
        // than the kernel's block count deadlocks the fifo.
        uint64_t kv_slack = 0;
        for (const auto& [kn, kd] : w_->kernels)
            if (kd.patch == "attnpos" && kd.rb > 2) kv_slack = std::max<uint64_t>(kv_slack, kd.rb - 2);
        state_.push_back(alloc(lt.state_kind == "kv" ? (cfg_.max_ctx + kv_slack) * lt.state_row : lt.state_bytes));
        if (progress) progress(l + 1, nl_ + 1);
        // This layer's source pages are packed and uploaded: let them go now rather than after
        // the last layer, so the peak is one layer of mapped file, not the whole file.
        file_->drop_pages();
        if ((l + 1) % 10 == 0 || l + 1 == nl_)
            log(std::to_string(l + 1) + "/" + std::to_string(nl_) + " layers resident (" +
                std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s)");
    }
    // ---- the globals: the lm_head pool and the final norm's weight from the file, the ptab
    // computed, everything else zero (xres, zero, xresf, hn, logits)
    for (const auto& [name, bytes] : w_->globals) {
        if (name == "lmpool" && skip_pools) {
            continue;
        } else if (name == "lmpool" && drop_pools_ && !(std::getenv("OFLM_OPEN_CPU_HEAD") && *std::getenv("OFLM_OPEN_CPU_HEAD") == '0')) {
            // CPU decode: the head on the CPU, repacked once; the NPU copy only to check it
            cpu_head_check_ = std::getenv("OFLM_OPEN_CPU_HEAD_CHECK") && *std::getenv("OFLM_OPEN_CPU_HEAD_CHECK") == '1';
            std::vector<uint8_t> pool(bytes);
            pools::pack_lmhead(*w_, *file_, pool.data());
            lm_cpu_.resize(host::lmhead_q8_cpu_bytes(w_->vocab, w_->hidden));
            host::lmhead_q8_repack(pool.data(), w_->vocab, w_->hidden, lm_cpu_.data());
            cpu_head_ = true;
            if (cpu_head_check_) globals_[name] = alloc(bytes, pool.data(), pool.size());
            log("CPU decode: the lm_head on the CPU (" + std::to_string(lm_cpu_.size() >> 20) + " MB, q8)");
        } else if (name == "lmpool") {
            xrt::bo lm = xrt::ext::bo(*dev_, bytes);
            pools::pack_lmhead(*w_, *file_, lm.map<uint8_t*>());
            lm.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            globals_[name] = std::move(lm);
        } else if (name == "normw") {
            if (bytes != w_->norm_bytes) throw std::runtime_error("open_qwen36: global normw is " + std::to_string(bytes) + " B, pack.norm says " + std::to_string(w_->norm_bytes));
            std::vector<uint8_t> nw(bytes);
            pools::pack_norm(*file_, w_->norm_tensor, bytes, nw.data());
            globals_[name] = alloc(bytes, nw.data(), nw.size());
        } else {
            globals_[name] = alloc(bytes);
        }
    }
    // a "#1" twin per staged GEMM step's x / y, so the host fills or reads one while the NPU uses the other
    if (stage_major_on_ && layer_major_ok())
        for (int l = 0; l < nl_; ++l) {
            const GemmBlockProgram& gb = types_[l]->gemm_block;
            for (const std::vector<Step>* prog : {&gb.program, &gb.shared_program, &gb.ffn_program})
                for (const Step& s : *prog)
                    for (const std::string& name : {s.args[1], s.args[2]})
                        if (!globals_.count(name + "#1")) globals_[name + "#1"] = alloc(w_->globals.at(name));
        }
    for (const auto& [name, rg] : w_->per_row_globals) {
        std::vector<uint8_t> pt(cfg_.max_ctx * rg.per_row);
        pools::build_ptab(*w_, rg, cfg_.max_ctx, pt.data());
        globals_[name] = alloc(pt.size(), pt.data(), pt.size());
    }
    file_->drop_pages();  // the packers are done with the container; keep only what the steps touch
    if (progress) progress(nl_ + 1, nl_ + 1);
    weights_loaded_ = true;
    pos_ = 0;
    log("weights resident: " + std::to_string(nl_) + " pools + lm_head, " +
        std::to_string(static_cast<int>(ms_since(t0) / 1000)) + " s");
}

void Core::reset() {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: reset before load_weights");
    // The linear layers' state must start at zero. The KV rows need not: the window
    // read is [0, max(pos, 1)) and row 0 at position 0 is a dummy the kernel masks.
    // A blocked attention kernel (manifest `rb`) reads a few rows at or past `pos` as
    // padding and multiplies them by an exact zero, so their CONTENT does not matter --
    // but it must not be a NaN, which Core::alloc's memset and any earlier token's real
    // k'/v' both guarantee.
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (lt.state_kind != "linear") continue;
        std::memset(state_[l].map<uint8_t*>(), 0, lt.state_bytes);
        state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
    }
    // A request with an image rewrote the position records of the rows it used
    // (write_record); the next request expects row p to say position p again.
    if (ptab_dirty_) {
        for (const auto& [name, rg] : w_->per_row_globals) {
            xrt::bo& bo = globals_.at(name);
            pools::build_ptab(*w_, rg, ptab_dirty_, bo.map<uint8_t*>());
            bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, ptab_dirty_ * rg.per_row, 0);
        }
        ptab_dirty_ = 0;
    }
    mrope_on_ = false;
    mrope_pos_ = 0;
    pos_ = 0;
}

xrt::bo& Core::buffer(const std::string& name, int layer) {
    if (name == "pool") return pools_[layer];
    if (name == "consts") return consts_[layer];
    if (name == "act") return act_[layer];
    if (name == "state") return state_[layer];
    // the block route's per-layer weight buffers (load_weights)
    if (auto g = gemm_w_.find(name); g != gemm_w_.end()) {
        if (layer < 0 || static_cast<size_t>(layer) >= g->second.size() || !g->second[layer])
            throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no '" + name + "' (gemm_block) buffer");
        return g->second[layer];
    }
    auto it = globals_.find(name);
    if (it == globals_.end()) throw std::runtime_error("open_qwen36: the program names an unknown buffer '" + name + "'");
    return it->second;
}

Core::Inflight Core::start_run(Kern& k, const std::vector<std::string>& args, int layer) {
    Inflight f;
    f.k = &k;
    f.layer = layer;
    f.t0 = std::chrono::steady_clock::now();
    // How long the HOST sat between the previous dispatch returning and this one
    // starting. If the timeout only ever follows a long gap, the trigger is idleness
    // rather than anything about the dispatch itself.
    f.gap_ms = last_done_.time_since_epoch().count()
                   ? std::chrono::duration<double, std::milli>(f.t0 - last_done_).count()
                   : -1.0;
    f.ctx_change = last_dispatched_ && last_dispatched_->ctx != k.ctx;
    last_dispatched_ = &k;
    ++dispatches_;
    f.r = xrt::run(*k.k);
    f.r.set_arg(0, kOpcode);
    f.r.set_arg(1, *k.instr);
    f.r.set_arg(2, static_cast<int>(k.words.size()));
    int i = 3;
    for (const auto& a : args) f.r.set_arg(i++, buffer(a, layer));
    f.submit_ms = ms_since(f.t0);
    f.t1 = std::chrono::steady_clock::now();
    f.r.start();
    return f;
}

std::pair<double, double> Core::wait_run(Inflight& f) {
    Kern& k = *f.k;
    const int layer = f.layer;
    const double gap_ms = f.gap_ms;
    const auto t0 = f.t0;
    const auto t1 = f.t1;
    xrt::run& r = f.r;
    const auto tw = std::chrono::steady_clock::now();
    // A blocking wait() sleeps and pays a scheduler wake-up on the way out, and that wake-up
    // lands INSIDE this dispatch's measured time and in front of the next one. Polling the
    // command's state removes it -- at the price of a busy core, which on this box pulls the
    // array's clock down through the shared package budget (the OpenMP note at the top of this
    // file is the same effect). So this is a probe of where the per-dispatch variance is, not
    // a setting: off unless OFLM_OPEN_SPIN_US says otherwise.
    if (spin_us_ > 0) {
        const double limit = spin_us_ / 1000.0;
        while (ms_since(tw) < limit) {
            const auto s = r.state();
            if (s != ERT_CMD_STATE_NEW && s != ERT_CMD_STATE_QUEUED && s != ERT_CMD_STATE_RUNNING &&
                s != ERT_CMD_STATE_SUBMITTED)
                break;
        }
    }
    auto st = cfg_.timeout_ms ? r.wait(std::chrono::milliseconds(cfg_.timeout_ms)) : r.wait();
    if (st != ERT_CMD_STATE_COMPLETED) {
        // Is the command hung, or merely late? Throwing here used to throw that question
        // away with it. Wait a little longer and say which it was.
        //
        // Five seconds, not another sixty. On every occurrence measured so far the driver
        // reported the command as never executed (no fault, nothing in flight), and a
        // command the firmware is not running does not arrive late - so a long second
        // wait buys nothing and costs a minute. Short enough to be free, long enough to
        // catch a genuinely late one and say so. OFLM_OPEN_TIMEOUT_RETRY_MS overrides.
        unsigned extra = 5000;
        if (const char* e = std::getenv("OFLM_OPEN_TIMEOUT_RETRY_MS")) extra = static_cast<unsigned>(std::strtoul(e, nullptr, 10));
        std::fprintf(stderr,
                     "open_qwen36: %s layer %d at position %d: ERT state %d after %.0f ms "
                     "(dispatch #%llu, %.0f ms host gap before it)\n",
                     k.name.c_str(), layer, pos_, static_cast<int>(st), ms_since(t0),
                     static_cast<unsigned long long>(dispatches_), gap_ms);
        // The driver logs its own view of this, and it is the difference between "our
        // dispatch was slow" and "the hardware context faulted". Every occurrence so far
        // had a matching entry; three hours of clean running had none.
        std::fprintf(stderr, "open_qwen36:   the NPU driver logs context errors as pci Event ID 3 in the Windows"
                             " System log; look for one at this moment before blaming the dispatch\n");
        std::fflush(stderr);
        // Only a TIMEOUT can still be in flight. An abort or an error is final, and
        // waiting on it just delays the rebuild by another minute.
        if (extra && st == ERT_CMD_STATE_TIMEOUT) {
            auto st2 = r.wait(std::chrono::milliseconds(extra));
            std::fprintf(stderr, "open_qwen36:   waited %u ms more: ERT state %d%s\n", extra,
                         static_cast<int>(st2),
                         st2 == ERT_CMD_STATE_COMPLETED ? " - it was LATE, not hung" : " - still not done");
            std::fflush(stderr);
            if (st2 == ERT_CMD_STATE_COMPLETED) {
                last_done_ = std::chrono::steady_clock::now();
                f.k = nullptr;
                return {ms_since(t1), ms_since(tw)};
            }
        }
        throw std::runtime_error("open_qwen36: kernel " + k.name + " layer " + std::to_string(layer) +
                                 " at position " + std::to_string(pos_) + " ended in ERT state " +
                                 std::to_string(static_cast<int>(st)) +
                                 (st == ERT_CMD_STATE_TIMEOUT ? " (timeout)" : "") + ", dispatch #" +
                                 std::to_string(dispatches_));
    }
    last_done_ = std::chrono::steady_clock::now();
    const double elapsed = ms_since(t1), blocked = ms_since(tw);
    if (step_trace_) trace_.push_back({&k, layer, f.submit_ms, elapsed, blocked, gap_ms, f.ctx_change});
    f.k = nullptr;                                 // waited: active() is false from here
    return {elapsed, blocked};
}

std::pair<double, double> Core::run_split(Kern& k, const std::vector<std::string>& args, int layer) {
    Inflight f = start_run(k, args, layer);
    const double submit = f.submit_ms;
    const auto [elapsed, blocked] = wait_run(f);
    (void)blocked;                                 // nothing runs between start and wait here
    return {submit, elapsed};
}

double Core::run(Kern& k, const std::vector<std::string>& args, int layer) {
    const auto [submit, wait] = run_split(k, args, layer);
    if (dispatch_log_) dispatch_stats_[k.name].add(submit + wait);
    return submit + wait;
}

std::map<std::string, DispatchStat> Core::take_dispatch_stats() {
    auto out = dispatch_stats_;
    dispatch_stats_.clear();
    return out;
}

namespace {
struct BenchStat {
    double min = 1e18, sum = 0, submit = 0;
    int n = 0;
    std::vector<double> all;
    void add(double submit_ms, double wait_ms) {
        const double t = submit_ms + wait_ms;
        min = t < min ? t : min;
        sum += t;
        submit += submit_ms;
        all.push_back(t);
        ++n;
    }
    double mean() const { return n ? sum / n : 0; }
    double mean_submit() const { return n ? submit / n : 0; }
    // one dispatch that loses the box to something else drags the mean several ms - which is
    // how a probe came out reading a negative switch cost. Quote this and the min instead.
    double median() const {
        if (all.empty()) return 0;
        std::vector<double> v = all;
        std::sort(v.begin(), v.end());
        return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    }
};

/// MB a dispatch streams from DDR, and the rate that implies at `ms`.
///
/// Both engines run the same weights through the same array, so at decode this
/// ratio is most of the story: a kernel at the array's rate has nothing left in
/// it and a kernel far below it is where the time is. Read off the PATCHED
/// stream, so a kernel whose transfer sizes depend on the position (ax0's KV
/// window) reports what this position streams.
double stream_mb(const uint32_t* iw, size_t words) {
    return stream_patch::ddr_bytes_total(iw, words) / 1048576.0;
}
double gbps(double mb, double ms) { return ms > 0 ? mb / 1024.0 / (ms / 1000.0) : 0; }
}  // namespace

void Core::bench_dispatch(int layer, int reps) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: bench_dispatch before load_weights");
    if (layer < 0 || layer >= nl_) throw std::runtime_error("open_qwen36: bench_dispatch: no such layer");
    const GemmBlockProgram& gb = types_[layer]->gemm_block;
    if (!gb.t) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no block route to bench");

    // every kernel the route names, with the buffer arguments the manifest gives it
    std::vector<std::pair<std::string, std::vector<std::string>>> jobs;
    for (const auto* prog : {&gb.program, &gb.shared_program, &gb.ffn_program})
        for (const Step& st : *prog) jobs.push_back({st.kernel, st.args});
    for (const auto& [slots, name] : gb.moe_batch.kernels) jobs.push_back({name, gb.moe_batch.args});
    if (!gb.moe_kernel.empty()) jobs.push_back({gb.moe_kernel, gb.moe_args});
    // The attention GEMMs sit on their own context, so a full-attention layer pays a switch into
    // them and another one back out. They read only globals, so a linear layer can still time them.
    for (int l = 0; l < nl_; ++l) {
        const AttnBlock& ab = types_[l]->gemm_block.attn_block;
        if (!ab.present()) continue;
        jobs.push_back({ab.kernels_s.rbegin()->second, ab.args});
        jobs.push_back({ab.kernels_pv.rbegin()->second, ab.args});
        break;
    }

    std::fprintf(stderr, "\nopen_qwen36: dispatch bench, layer %d, %d reps each\n", layer, reps);
    std::fprintf(stderr, "  %-22s %8s %8s %8s %9s %7s %8s\n", "kernel", "min ms", "mean ms", "submit", "MB", "GB/s",
                 "context");
    std::map<std::string, BenchStat> alone;
    for (const auto& [name, args] : jobs) {
        Kern& k = kerns_.at(name);
        run_split(k, args, layer);                       // warm: the first call of a context pays for it
        BenchStat st;
        for (int i = 0; i < reps; ++i) {
            const auto [submit, wait] = run_split(k, args, layer);
            st.add(submit, wait);
        }
        alone[name] = st;
        const double mb = stream_mb(k.iw(), k.words.size());
        std::fprintf(stderr, "  %-22s %8.3f %8.3f %8.3f %9.1f %7.1f %8s\n", name.c_str(), st.min, st.mean(),
                     st.mean_submit(), mb, gbps(mb, st.min), w_->kernels.at(name).context.c_str());
    }

    // Every probe below compares a kernel against its own baseline taken IN THE SAME LOOP, one
    // baseline rep per probe rep. Subtracting the pass above instead -- which is what these did
    // until now -- puts whatever the box did in between straight into the delta: the k35v5 and
    // k35v6 runs of 2026-09-13 disagreed by 5.6 ms on mb_s256 that way, and one of them read a
    // context switch as free. The context-switch probe at the end was fixed first (3b86aac4);
    // this is the same fix for the other five. Deltas are quoted on the minima for the same
    // reason: one dispatch that loses the box drags a mean by several ms.
    //
    // `one_rep` does whatever the probe is measuring and returns (submit, wait); the baseline rep
    // beside it is the same kernel with nothing done to it. `note` is printed after the delta.
    using Rep = std::pair<double, double>;
    auto probe = [&](const char* title,
                     const std::vector<std::pair<std::string, std::vector<std::string>>>& what,
                     const std::function<bool(const std::string&)>& skip,
                     const std::function<Rep(const std::string&, const std::vector<std::string>&, int)>& one_rep,
                     const std::function<std::string(const std::string&)>& note) {
        std::fprintf(stderr, "  %s\n", title);
        for (const auto& [name, args] : what) {
            if (skip(name)) continue;
            Kern& k = kerns_.at(name);
            BenchStat base, st;
            run_split(k, args, layer);                   // warm, outside the timing
            for (int i = 0; i < reps; ++i) {
                { const auto [s, w] = run_split(k, args, layer); base.add(s, w); }
                const auto [s, w] = one_rep(name, args, i);
                st.add(s, w);
            }
            std::fprintf(stderr, "  %-22s %8.3f %8.3f %8.3f  %+.3f vs its own baseline (%.3f)%s\n", name.c_str(),
                         st.min, st.mean(), st.mean_submit(), st.min - base.min, base.min, note(name).c_str());
        }
    };
    auto never = [](const std::string&) { return false; };
    auto no_note = [](const std::string&) { return std::string(); };

    // The same kernels cycling through every layer's own weights. The pass above re-reads
    // layer 0's, which a real block never does: it walks 40 layers once. Whatever this costs
    // over the pass above is the price of reading weights nothing has touched recently.
    // only this layer type's own layers: the two types name different weight buffers
    std::vector<int> same;
    for (int l = 0; l < nl_; ++l)
        if (types_[l]->name == types_[layer]->name) same.push_back(l);
    probe(("cycling the " + std::to_string(same.size()) + " " + types_[layer]->name +
           " layers' weights (the cold-memory probe)").c_str(),
          jobs, never,
          [&](const std::string& n, const std::vector<std::string>& args, int i) {
              return run_split(kerns_.at(n), args, same[i % same.size()]);
          },
          no_note);

    // The same kernels with the CPU busy for ~30 ms first, the gap a real layer has between
    // its dispatches. Same context throughout, so anything here is the cost of an idle NPU.
    probe("after a 30 ms BUSY host gap (the idle probe)", jobs, never,
          [&](const std::string& n, const std::vector<std::string>& args, int) {
              volatile double spin = 0;                   // busy, not asleep: the CPU works in a real block
              auto g0 = std::chrono::steady_clock::now();
              while (ms_since(g0) < 30.0) spin += 1.0;
              return run_split(kerns_.at(n), args, layer);
          },
          no_note);

    // The same gap with the CPU ASLEEP, which is the pair that says what the busy one measured.
    // A sleeping gap leaves the NPU idle exactly as long and takes no package power, so a cost
    // here is the NPU's own clock coming back up and a cost only in the busy probe is the CPU
    // stealing the power budget (which is what A.1's PASSIVE finding turned on). Both matter:
    // a real block's host stages are tens of ms of work between dispatches.
    probe("after a 30 ms SLEEPING host gap (the idle probe's pair)", jobs, never,
          [&](const std::string& n, const std::vector<std::string>& args, int) {
              std::this_thread::sleep_for(std::chrono::milliseconds(30));
              return run_split(kerns_.at(n), args, layer);
          },
          no_note);

    // The patched kernels with their instruction stream re-synced first, as the real path does
    // it. The patch itself is a few hundred words; the sync is the whole stream.
    std::vector<uint32_t> ex(w_->moe.experts);
    for (size_t i = 0; i < ex.size(); ++i) ex[i] = static_cast<uint32_t>(i);
    std::map<std::string, BenchStat> sync_only;
    probe("with the expert patch + instruction sync (the patch probe)", jobs,
          [&](const std::string& n) { return kerns_.at(n).moe2.empty(); },
          [&](const std::string& n, const std::vector<std::string>& args, int) {
              Kern& k = kerns_.at(n);
              auto p0 = std::chrono::steady_clock::now();
              stream_patch::moe2_apply(k.iw(), k.moe2, ex.data(), w_->moe);
              k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
              sync_only[n].add(ms_since(p0), 0);
              return run_split(k, args, layer);
          },
          [&](const std::string& n) {
              char buf[80];
              std::snprintf(buf, sizeof buf, " (patch+sync itself %.3f, %zu KB)", sync_only[n].mean(),
                            kerns_.at(n).words.size() * 4 / 1024);
              return std::string(buf);
          });

    // The same kernels with the host moving their buffers around each call, as the route does.
    std::map<std::string, BenchStat> traffic;
    volatile double sink = 0;
    probe("with the host reading the output (the buffer-traffic probe)", jobs,
          [&](const std::string& n) {
              for (const auto& [name, args] : jobs)
                  if (name == n) return args.size() < 3;
              return true;
          },
          [&](const std::string& n, const std::vector<std::string>& args, int) {
              xrt::bo& xb = buffer(args[args.size() - 2], layer);   // the input the host fills
              xrt::bo& yb = buffer(args.back(), layer);             // the output the host reads
              auto h0 = std::chrono::steady_clock::now();
              xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, xb.size(), 0);
              const double up = ms_since(h0);
              const auto r = run_split(kerns_.at(n), args, layer);
              auto h1 = std::chrono::steady_clock::now();
              read_back(yb, yb.size(), 0);
              const float* y = yb.map<float*>();
              for (size_t j = 0; j < yb.size() / 4; j += 1024) sink += y[j];
              traffic[n].add(up, ms_since(h1));
              return r;
          },
          [&](const std::string& n) {
              char buf[64];
              std::snprintf(buf, sizeof buf, " (host traffic %.2f ms)", traffic[n].mean());
              return std::string(buf);
          });

    // The layer's real dispatch cycle, and the same cycle with the host writing and reading
    // ~48 MB between dispatches (the size of the tiling, gather and transpose the route does
    // around each one). All three passes -- solo, cycle, cycle+churn -- run inside ONE rep loop,
    // so the two deltas below are differences between things measured seconds apart rather than
    // minutes, and each says one thing: rotation is what the cycle costs over the same kernel
    // repeated, churn is what the host traffic costs over the SAME cycle without it. Reading
    // churn against the solo baseline, as this did before, charged it for the rotation as well.
    {
        std::vector<std::pair<std::string, std::vector<std::string>>> cycle;
        for (const Step& st : gb.program) cycle.push_back({st.kernel, st.args});
        for (const Step& st : gb.shared_program) cycle.push_back({st.kernel, st.args});
        for (const Step& st : gb.ffn_program) cycle.push_back({st.kernel, st.args});
        if (!gb.moe_batch.kernels.empty()) {
            auto big = gb.moe_batch.kernels.rbegin();                 // the widest stream, then the second pass
            cycle.push_back({big->second, gb.moe_batch.args});
            if (gb.moe_batch.kernels.size() > 1)
                cycle.push_back({std::next(big)->second, gb.moe_batch.args});
        }
        std::vector<float> churn(6u << 20), churn2(6u << 20);        // 24 MB written, 24 MB read
        volatile double csink = 0;
        std::map<std::string, BenchStat> solo, rot, chn;
        for (const auto& [name, args] : cycle) run_split(kerns_.at(name), args, layer);   // warm
        for (int i = 0; i < reps; ++i) {
            for (const auto& [name, args] : cycle) {
                run_split(kerns_.at(name), args, layer);              // the rep before pays any switch
                const auto [s, w] = run_split(kerns_.at(name), args, layer);
                solo[name].add(s, w);
            }
            for (const auto& [name, args] : cycle) {
                const auto [s, w] = run_split(kerns_.at(name), args, layer);
                rot[name].add(s, w);
            }
            for (const auto& [name, args] : cycle) {
#pragma omp parallel for
                for (long long j = 0; j < static_cast<long long>(churn.size()); ++j)
                    churn[j] = static_cast<float>(j + i);
                for (size_t j = 0; j < churn2.size(); j += 16) csink += churn2[j] + churn[j];
                const auto [s, w] = run_split(kerns_.at(name), args, layer);
                chn[name].add(s, w);
            }
        }
        std::fprintf(stderr, "  the layer's own cycle of %zu dispatches (the rotation probe) and the same "
                             "cycle with ~48 MB of host churn, both against an interleaved solo baseline\n",
                     cycle.size());
        std::fprintf(stderr, "  %-22s %8s %8s %8s %9s %9s\n", "kernel", "solo min", "cycle", "+churn",
                     "rotation", "churn");
        for (const auto& [name, args] : cycle)
            std::fprintf(stderr, "  %-22s %8.3f %8.3f %8.3f %9.3f %9.3f\n", name.c_str(), solo[name].min,
                         rot[name].min, chn[name].min, rot[name].min - solo[name].min,
                         chn[name].min - rot[name].min);
    }

    // The same kernels alternating with one from another context: if a dispatch costs more
    // here than it did after one of its own, the difference is what switching hardware
    // contexts costs.
    //
    // The baseline is taken inside this loop rather than from the pass at the top of the
    // function. Every other probe here subtracts a number measured minutes earlier, so any
    // drift in what else the box is doing lands straight in the delta - the k35v5 and k35v6
    // runs of 2026-09-13 disagree by 5.6 ms on mb_s256 that way, one of them reading a
    // switch as free. Here a baseline rep and a probe rep alternate, so whatever moves moves
    // both, and the delta is quoted on the minima.
    std::fprintf(stderr, "  alternating with the widest GEMM (the context-switch probe, "
                         "baseline interleaved)\n");
    std::fprintf(stderr, "  %-22s %8s %8s %8s %8s %9s\n", "kernel", "own min", "own med", "sw min",
                 "sw med", "switch");
    std::string other;
    size_t widest = 0;
    for (const auto& [name, args] : jobs)
        if (name.rfind("gemm_n", 0) == 0) {
            const size_t n = std::stoul(name.substr(6, name.find('_', 6) - 6));
            if (n > widest) { widest = n; other = name; }
        }
    if (other.empty()) { std::fprintf(stderr, "  (no GEMM step in this route)\n"); return; }
    const std::vector<std::string>* other_args = nullptr;
    for (const auto& [name, args] : jobs)
        if (name == other) other_args = &args;
    for (const auto& [name, args] : jobs) {
        if (name == other) continue;
        Kern& k = kerns_.at(name);
        Kern& o = kerns_.at(other);
        BenchStat solo, sw;
        run_split(o, *other_args, layer);                // warm both contexts before timing
        run_split(k, args, layer);
        for (int i = 0; i < reps; ++i) {
            run_split(k, args, layer);                   // the rep before this one pays the switch
            { const auto [s, w] = run_split(k, args, layer); solo.add(s, w); }
            run_split(o, *other_args, layer);
            { const auto [s, w] = run_split(k, args, layer); sw.add(s, w); }
        }
        std::fprintf(stderr, "  %-22s %8.3f %8.3f %8.3f %8.3f  %+.3f\n", name.c_str(), solo.min,
                     solo.median(), sw.min, sw.median(), sw.min - solo.min);
    }
    std::fprintf(stderr, "\n");
}

void Core::bench_kernel(const std::string& name, int reps, int layer, int warm_token) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: bench_kernel before load_weights");
    if (layer < 0 || layer >= nl_) throw std::runtime_error("open_qwen36: bench_kernel: no such layer");
    Kern& k = kerns_.at(name);
    // Only a patched kernel needs a real step first to put live values in its stream. An
    // unpatched one does not, and the step would run the whole program -- which is exactly what
    // must not happen when the build under test is a truncated one whose other half would hang.
    if (!k.patch.empty()) step(warm_token, false);
    // the arguments the layer's own program gives it, so the buffers are the real ones
    const std::vector<std::string>* args = nullptr;
    for (const Step& s : types_[layer]->program)
        if (s.op == "run" && s.kernel == name) args = &s.args;
    if (!args)
        for (const Step& s : w_->tail)
            if (s.kernel == name) args = &s.args;
    // the block route's kernels too: the prefill's own dispatches are the ones worth holding
    // at full rate for a whole prefill's worth of seconds (see the window report below)
    const GemmBlockProgram& gb = types_[layer]->gemm_block;
    if (!args)
        for (const auto* prog : {&gb.program, &gb.shared_program, &gb.ffn_program})
            for (const Step& s : *prog)
                if (s.kernel == name) args = &s.args;
    if (!args)
        for (const auto& [slots, kn] : gb.moe_batch.kernels)
            if (kn == name) args = &gb.moe_batch.args;
    if (!args)
        for (int l = 0; l < nl_ && !args; ++l) {
            const AttnBlock& ab = types_[l]->gemm_block.attn_block;
            if (!ab.present()) continue;
            for (const auto* m : {&ab.kernels_s, &ab.kernels_pv})
                for (const auto& [len, kn] : *m)
                    if (kn == name) args = &ab.args;
        }
    if (!args) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " does not run " + name);

    run_split(k, *args, layer);                             // the first call of a context pays for it
    // Windowed, because the question this answers is whether the rate HOLDS. A prefill is tens
    // of seconds of back-to-back dispatches, and every in-block figure is ~1.4x the same
    // kernel's `--bench` minimum even under `--pmode turbo`. If that is the package heating up
    // rather than anything the engine does between dispatches, it shows here as the window
    // minimum drifting up under a load with no host work in it at all.
    BenchStat st;
    BenchStat win;
    const int W = reps >= 200 ? reps / 20 : 0;
    const auto tb0 = std::chrono::steady_clock::now();
    if (W)
        std::fprintf(stderr, "\nopen_qwen36: %s on layer %d, %d reps in windows of %d\n  %8s %9s %9s\n",
                     name.c_str(), layer, reps, W, "at s", "win min", "win mean");
    for (int i = 0; i < reps; ++i) {
        const auto [submit, wait] = run_split(k, *args, layer);
        st.add(submit, wait);
        if (!W) continue;
        win.add(submit, wait);
        if ((i + 1) % W == 0) {
            std::fprintf(stderr, "  %8.1f %9.3f %9.3f\n", ms_since(tb0) / 1000.0, win.min, win.mean());
            win = BenchStat{};
        }
    }
    std::fprintf(stderr, "\nopen_qwen36: %s on layer %d, %d reps: %.3f min, %.3f mean, %.3f submit (context %s)\n\n",
                 name.c_str(), layer, reps, st.min, st.mean(), st.mean_submit(), w_->kernels.at(name).context.c_str());
}

void Core::bench_decode(int reps) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: bench_decode before load_weights");

    // A layer's program is ONE core program issued as two streams (lx0 then lx1, ax0 then ax1):
    // the first half fills the fifos the second drains, so neither half can be repeated on its
    // own -- it hangs. Every probe below replays a whole layer, and attributes per kernel.
    std::map<std::string, std::vector<int>> layers_of;
    for (int l = 0; l < nl_; ++l) layers_of[types_[l]->name].push_back(l);

    auto replay = [&](int l, std::map<std::string, BenchStat>& into) {
        double ms = 0;
        for (const Step& s : types_[l]->program) {
            if (s.op != "run") { route(kerns_.at(s.kernel), l, s.act_off); continue; }
            const auto [submit, wait] = run_split(kerns_.at(s.kernel), s.args, l);
            into[s.kernel].add(submit, wait);
            ms += submit + wait;
        }
        return ms;
    };

    std::fprintf(stderr, "\nopen_qwen36: decode bench at position %d, %d reps each\n", pos_, reps);
    std::fprintf(stderr, "  %-16s %-12s %8s %8s %8s %9s %7s %8s\n", "probe", "kernel", "min ms", "mean ms",
                 "submit", "MB", "GB/s", "context");

    // One layer, over and over: its weights stay in whatever cache holds them and the context
    // never changes, so this is the kernel's own cost with nothing charged on top.
    std::map<std::string, BenchStat> alone;
    std::map<std::string, double> alone_layer_ms;
    for (const auto& [tname, ls] : layers_of) {
        std::map<std::string, BenchStat> warm;
        replay(ls[0], warm);                                // the first call of a context pays for it
        double ms = 0;
        for (int i = 0; i < reps; ++i) ms += replay(ls[0], alone);
        alone_layer_ms[tname] = ms / reps;
        for (const Step& s : types_[ls[0]]->program) {
            if (s.op != "run") continue;
            const BenchStat& st = alone[s.kernel];
            Kern& k = kerns_.at(s.kernel);
            const double mb = stream_mb(k.iw(), k.words.size());
            std::fprintf(stderr, "  %-16s %-12s %8.3f %8.3f %8.3f %9.1f %7.1f %8s\n", ("one " + tname).c_str(),
                         s.kernel.c_str(), st.min, st.mean(), st.mean_submit(), mb, gbps(mb, st.min),
                         w_->kernels.at(s.kernel).context.c_str());
        }
    }

    // What each dispatch streams, per buffer argument. This is the split the rate table needs:
    // ax0's projection weights and its cached-KV walk are two very different numbers behind one
    // dispatch time, and which of them is off the array's rate decides what to fix.
    std::fprintf(stderr, "\n  what each dispatch streams (patched at position %d)\n", pos_);
    for (const auto& [tname, ls] : layers_of)
        for (const Step& s : types_[ls[0]]->program) {
            if (s.op != "run") continue;
            Kern& k = kerns_.at(s.kernel);
            std::string line;
            for (const auto& [arg, bytes] : stream_patch::ddr_bytes(k.iw(), k.words.size())) {
                const char* nm = arg < s.args.size() ? s.args[arg].c_str() : "?";
                char buf[96];
                std::snprintf(buf, sizeof buf, "  %s[%u] %.2f", nm, arg, bytes / 1048576.0);
                line += buf;
            }
            std::fprintf(stderr, "  %-12s %9.2f MB total = %s MB\n", s.kernel.c_str(),
                         stream_mb(k.iw(), k.words.size()), line.c_str());
        }
    std::fprintf(stderr, "\n");

    // The same, cycling every layer of that type: a real step touches each layer's weights once
    // and never comes back, so anything here over the pass above is the price of cold weights.
    std::map<std::string, BenchStat> cold;
    for (const auto& [tname, ls] : layers_of) {
        double ms = 0;
        for (int i = 0; i < reps; ++i) ms += replay(ls[i % ls.size()], cold);
        for (const Step& s : types_[ls[0]]->program) {
            if (s.op != "run") continue;
            const BenchStat& st = cold[s.kernel];
            std::fprintf(stderr, "  %-16s %-12s %8.3f %8.3f %8.3f  %+.3f vs one layer\n", ("cold " + tname).c_str(),
                         s.kernel.c_str(), st.min, st.mean(), st.mean_submit(), st.mean() - alone[s.kernel].mean());
        }
        std::fprintf(stderr, "  %-16s %-12s %8.1f ms per layer (%.1f warm)\n", "", "= layer", ms / reps,
                     alone_layer_ms[tname]);
    }

    // The real walk: layer 0 to nl_ in order, then the tail. The two types interleave, so this
    // is the only probe that pays for changing hardware context, and its total is the floor a
    // decode step cannot go below.
    std::map<std::string, BenchStat> walk;
    double step_ms = 0;
    for (int i = 0; i < reps; ++i) {
        for (int l = 0; l < nl_; ++l) step_ms += replay(l, walk);
        for (const Step& s : w_->tail) {
            const auto [submit, wait] = run_split(kerns_.at(s.kernel), s.args, 0);
            walk[s.kernel].add(submit, wait);
            step_ms += submit + wait;
        }
    }
    double step_mb = 0;
    for (const auto& [name, st] : walk) {
        const auto it = alone.find(name);
        char delta[48] = "";
        if (it != alone.end()) std::snprintf(delta, sizeof delta, "  %+.3f vs one layer", st.mean() - it->second.mean());
        Kern& k = kerns_.at(name);
        const int calls = st.n / reps;
        const double mb = stream_mb(k.iw(), k.words.size());
        step_mb += mb * calls;
        std::fprintf(stderr, "  %-16s %-12s %8.3f %8.3f %8.3f %9.1f %7.1f  %4d calls/step%s\n", "the real walk",
                     name.c_str(), st.min, st.mean(), st.mean_submit(), mb, gbps(mb, st.min), calls, delta);
    }
    // The one number the closed engine can be held against: both stream the same weights through
    // the same array, so a step is bytes / rate and nothing else. A kernel at the array's rate has
    // nothing left in it; the gap lives in whichever ones are not.
    const double step = step_ms / reps;
    std::fprintf(stderr, "  a step's dispatches: %.1f ms, %.0f MB, %.1f GB/s aggregate\n\n", step, step_mb,
                 gbps(step_mb, step));
}

int Core::det_step(int reps, const std::vector<int>& ids, bool full) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: det_step before load_weights");
    if (ids.empty()) throw std::runtime_error("open_qwen36: det_step wants at least one id");
    const int p0 = pos_;
    // One step's observable output. `full`: every buffer a step writes -- each layer's act,
    // the KV row it wrote or a hash of its recurrent state, xres / xresf / hn -- plus the
    // logits; otherwise the logits alone, which is all a decode loop reads, and reads nothing
    // else back that could change the timing being probed.
    struct Buf { std::string name; int layer; std::vector<uint8_t> bytes; };
    auto grab = [&](int pos) {
        std::vector<Buf> out;
        for (auto& [l, rec] : route_log_) out.push_back({"route", l, std::move(rec)});
        route_log_.clear();
        auto take = [&](const std::string& name, int layer, xrt::bo& bo, size_t off, size_t n, bool hash) {
            read_back(bo, n, off);
            const uint8_t* m = bo.map<uint8_t*>() + off;
            if (!hash) {
                out.push_back({name, layer, std::vector<uint8_t>(m, m + n)});
                return;
            }
            uint64_t h = 1469598103934665603ull;       // FNV-1a over 8-byte words
            for (size_t o = 0; o + 8 <= n; o += 8) {
                uint64_t w;
                std::memcpy(&w, m + o, 8);
                h = (h ^ w) * 1099511628211ull;
            }
            std::vector<uint8_t> hb(8);
            std::memcpy(hb.data(), &h, 8);
            out.push_back({name + "(hash)", layer, std::move(hb)});
        };
        if (full)
            for (int l = 0; l < nl_; ++l) {
                const LayerType& lt = *types_[l];
                take("act", l, act_[l], 0, lt.act_bytes, false);
                if (lt.state_kind == "kv")
                    take("kv@" + std::to_string(pos), l, state_[l], static_cast<size_t>(pos) * lt.state_row,
                         lt.state_row, false);
                else
                    take("state", l, state_[l], 0, lt.state_bytes, true);
            }
        if (full)
            for (const char* g : {"xres", "xresf", "hn"})
                if (globals_.count(g))
                    take(g, -1, globals_.at(g), 0, w_->hidden * (std::string(g) == "hn" ? 2 : 4), false);
        std::vector<uint8_t> lg(w_->vocab * 4);
        std::memcpy(lg.data(), logits_host_.data(), lg.size());
        out.push_back({"logits", -1, std::move(lg)});
        return out;
    };
    using Run = std::vector<std::vector<Buf>>;
    auto once = [&]() {
        Run run;
        reset();
        seek(p0);
        for (size_t i = 0; i < ids.size(); ++i) {
            step(ids[i], true);
            run.push_back(grab(p0 + static_cast<int>(i)));
        }
        return run;
    };
    // OFLM_DET_ROUTE=1 also compares every router record -- but syncing the whole record
    // instead of the 32-byte idx changes route()'s timing enough to hide the late-read bug
    // (repeat-bug.md), so it is off by default.
    route_log_on_ = std::getenv("OFLM_DET_ROUTE") && std::string(std::getenv("OFLM_DET_ROUTE")) == "1";
    route_log_.clear();
    once();                                   // warm: the contexts, and the first step's patches
    const Run ref = once();
    std::map<std::string, int> first_at;      // "step s L<l> <buf>" -> reps that first moved there
    int bad = 0;
    std::fprintf(stderr, "open_qwen36: det_step: %d reps of %zu steps from position %d (%s)\n", reps, ids.size(), p0,
                 full ? "every buffer" : "logits only");
    for (int r = 0; r < reps; ++r) {
        const Run got = once();
        bool rep_bad = false;
        for (size_t s = 0; s < ids.size() && !rep_bad; ++s) {
            std::string where, all;
            int nbufs = 0;
            for (size_t i = 0; i < ref[s].size(); ++i) {
                const auto& a = ref[s][i].bytes;
                const auto& b = got[s][i].bytes;
                size_t nw = 0, first = SIZE_MAX, last = 0;
                for (size_t o = 0; o + 4 <= a.size(); o += 4)
                    if (std::memcmp(&a[o], &b[o], 4)) {
                        ++nw;
                        if (first == SIZE_MAX) first = o;
                        last = o;
                    }
                if (!nw) continue;
                ++nbufs;
                char line[256];
                std::snprintf(line, sizeof line, "L%d %s: %zu words differ in [%zu, %zu]", ref[s][i].layer,
                              ref[s][i].name.c_str(), nw, first, last + 4);
                if (where.empty()) {
                    where = line;
                    first_at["step " + std::to_string(s) + " L" + std::to_string(ref[s][i].layer) + " " +
                             ref[s][i].name]++;
                    int shown = 0;
                    for (size_t o = first; o + 4 <= a.size() && shown < 8; o += 4) {
                        if (!std::memcmp(&a[o], &b[o], 4)) continue;
                        float fa, fb;
                        uint32_t ua, ub;
                        std::memcpy(&fa, &a[o], 4);
                        std::memcpy(&fb, &b[o], 4);
                        std::memcpy(&ua, &a[o], 4);
                        std::memcpy(&ub, &b[o], 4);
                        char e[160];
                        std::snprintf(e, sizeof e, "\n        @%zu %08x -> %08x (f32 %g -> %g)", o, ua, ub, fa, fb);
                        where += e;
                        ++shown;
                    }
                }
                if (nbufs <= 16) all += std::string("\n      ") + line;
            }
            if (!nbufs) continue;
            rep_bad = true;
            ++bad;
            std::fprintf(stderr, "  rep %d DIFFERS at step %zu (position %d, id %d): %d buffers; first in walk order: %s%s%s\n",
                         r, s, p0 + static_cast<int>(s), ids[s], nbufs, where.c_str(), all.c_str(),
                         nbufs > 16 ? "\n      ..." : "");
            std::fflush(stderr);
        }
    }
    std::fprintf(stderr, "open_qwen36: det_step: %d of %d reps differed from the reference\n", bad, reps);
    std::fprintf(stderr, "open_qwen36: late router reads caught (OFLM_OPEN_ROUTE_SENTINEL=%d): %llu of %llu\n",
                 route_sentinel_ ? 1 : 0, static_cast<unsigned long long>(route_late_),
                 static_cast<unsigned long long>(route_reads_));
    if (route_check_)
        std::fprintf(stderr, "open_qwen36: route check: %llu of %llu router reads changed on a re-read\n",
                     static_cast<unsigned long long>(route_stale_), static_cast<unsigned long long>(route_checks_));
    for (const auto& [k, n] : first_at) std::fprintf(stderr, "    first moved at %s: %d\n", k.c_str(), n);
    route_log_on_ = false;
    route_log_.clear();
    reset();
    return bad;
}

void Core::bench_step(int reps, int token) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: bench_step before load_weights");
    // The REAL step, not a replay of its dispatches: the host stages, the route's patches and
    // whatever the pipeline does with the gaps are all in it. Every rep starts from the same
    // position, so ax0's window is the same width in all of them; the state and KV rows the
    // steps leave behind are meaningless, exactly as after bench_decode().
    // The three schedules are swept INTERLEAVED -- rep i takes level 0, then 1, then 2 -- so
    // whatever the box does over the sweep lands on all three instead of on whichever ran last.
    // A full decode's own median swings 122-139 ms between runs of the same set on a quiet box;
    // a blocked A/B of one run each says nothing.
    const int p0 = pos_;
    const int configured = submit_ahead_;
    // Serial against same-context submit-ahead. Level 2 (queueing ACROSS hardware contexts) is
    // in the sweep only when it is the configured level, because it HANGS the array: a command
    // queued from a second context while the first still has one in flight took `ax1` into
    // ERT state 8 twice in two runs, at position 1024 and at 4000, both on the first
    // cross-context queue-ahead of the step (c_benchstep_p1024.log, c_benchstep_p4000.log).
    // Stage 2.6: the host decode route has no dispatch pipeline to schedule -- the
    // submit-ahead levels are the NPU route's own knob -- so its bench is serial-only
    // rather than paying two meaningless rows per rep.
    const int nlev = (decode_route_at(static_cast<size_t>(p0 < 0 ? 0 : p0)) == "host") ? 1 : (configured >= 2 ? 3 : 2);
    const int levels[3] = {0, 1, 2};
    std::vector<double> wall[3], disp[3];
    double embed[3] = {0, 0, 0}, patch[3] = {0, 0, 0}, route[3] = {0, 0, 0}, lm[3] = {0, 0, 0};
    double p0ms[3] = {0, 0, 0}, p1ms[3] = {0, 0, 0};
    submit_ahead_ = levels[0];
    seek(p0);
    step(token, true);                                   // warm: the first step of a sweep pays for the contexts
    for (int i = 0; i < reps; ++i)
        for (int j = 0; j < nlev; ++j) {
            submit_ahead_ = levels[j];
            seek(p0);
            auto t = std::chrono::steady_clock::now();
            step(token, true);
            wall[j].push_back(ms_since(t));
            disp[j].push_back(timing_.dispatch_ms);
            embed[j] += timing_.embed_ms;
            patch[j] += timing_.patch_ms;
            route[j] += timing_.route_ms;
            lm[j] += timing_.lmhead_ms;
            p0ms[j] += timing_.part0_ms;
            p1ms[j] += timing_.part1_ms;
        }
    submit_ahead_ = configured;
    seek(p0);
    auto stat = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        const double med = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
        double sum = 0;
        for (double x : v) sum += x;
        return std::array<double, 3>{v.front(), med, sum / static_cast<double>(v.size())};
    };
    const double n = static_cast<double>(reps);
    std::fprintf(stderr,
                 "  the REAL step, %d reps at position %d, the three schedules interleaved\n"
                 "  (min/median/mean are the step's WALL time -- the number a token costs. `disp` sums each"
                 " dispatch\n   start-to-done, so on the pipelined rows it counts the time a queued-ahead command"
                 " waited\n   for the one in front of it and is NOT comparable to the serial row.)\n",
                 reps, p0);
    std::fprintf(stderr, "  %-26s %8s %8s %8s %9s %9s %8s %8s %8s\n", "schedule", "min", "median", "mean",
                 "disp min", "disp mean", "embed", "patch", "route");
    static const char* kName[3] = {"serial (start; wait)", "submit-ahead, same ctx", "submit-ahead, any ctx"};
    for (int j = 0; j < nlev; ++j) {
        const auto w = stat(wall[j]), d = stat(disp[j]);
        std::fprintf(stderr, "  %-26s %8.1f %8.1f %8.1f %9.1f %9.1f %8.2f %8.2f %8.2f%s\n", kName[j], w[0], w[1], w[2],
                     d[0], d[2], embed[j] / n, patch[j] / n, route[j] / n,
                     levels[j] == configured ? "   <- OFLM_OPEN_SUBMIT_AHEAD" : "");
    }
    {
        const auto w = stat(wall[0]);
        std::fprintf(stderr,
                     "      serial residue (wall - part0 - part1 - route - lm - embed - patch): %.2f ms\n",
                     w[2] - (p0ms[0] + p1ms[0] + route[0] + lm[0] + embed[0] + patch[0]) / n);
    }
    std::fprintf(stderr, "\n");
}

void Core::dump_step_trace(const char* what) {
    if (!step_trace_ || trace_.empty()) return;
    struct Agg {
        const Kern* k = nullptr;
        std::vector<double> el;
        double submit = 0, gap = 0, blocked = 0;
        int n_sw = 0;
        double el_sw = 0, el_same = 0;
    };
    std::map<std::string, Agg> per;
    for (const TraceRec& r : trace_) {
        Agg& a = per[r.k->name];
        a.k = r.k;
        a.el.push_back(r.elapsed_ms);
        a.submit += r.submit_ms;
        a.blocked += r.blocked_ms;
        a.gap += r.gap_ms > 0 ? r.gap_ms : 0;
        if (r.ctx_change) { ++a.n_sw; a.el_sw += r.elapsed_ms; }
        else a.el_same += r.elapsed_ms;
    }
    const double steps = trace_steps_ ? static_cast<double>(trace_steps_) : 1.0;
    std::fprintf(stderr, "\nopen_qwen36: step trace -- %s: %zu dispatches over %d steps\n", what, trace_.size(),
                 trace_steps_);
    // OFLM_OPEN_STEP_TRACE=2 also prints the LAST step dispatch by dispatch, in order, which is
    // how "the slow ones are the first after a context change" is told from "the slow ones are
    // the late layers" -- the per-kernel split below cannot separate those two. The last step,
    // not the first: the first one after a prefill is not a typical step.
    if (step_trace_ > 1) {
        const size_t n = trace_.size() / (trace_steps_ > 1 ? static_cast<size_t>(trace_steps_) : 1);
        std::fprintf(stderr, "  the last step, dispatch by dispatch (%zu of them)\n", n);
        std::fprintf(stderr, "  %4s %-10s %-6s %5s %8s %8s %8s %s\n", "#", "kernel", "ctx", "layer", "gap", "submit",
                     "dispatch", "");
        for (size_t i = trace_.size() - n; i < trace_.size(); ++i) {
            const TraceRec& r = trace_[i];
            std::fprintf(stderr, "  %4zu %-10s %-6s %5d %8.3f %8.3f %8.3f %s\n", i - (trace_.size() - n),
                         r.k->name.c_str(), r.k->ctx.c_str(), r.layer, r.gap_ms, r.submit_ms, r.elapsed_ms,
                         r.ctx_change ? "<- context change" : "");
        }
    }
    std::fprintf(stderr, "  %-10s %-6s %7s %8s %8s %8s %8s %8s %8s %8s %8s %8s\n", "kernel", "ctx", "calls/st",
                 "min", "mean", "p90", "submit", "gap", "blocked", "n(sw)", "mean(sw)", "mean(=)");
    double sum_min = 0, sum_mean = 0;
    for (auto& [name, a] : per) {
        std::sort(a.el.begin(), a.el.end());
        const double n = static_cast<double>(a.el.size());
        double s = 0;
        for (double x : a.el) s += x;
        const double p90 = a.el[static_cast<size_t>(0.9 * (n - 1) + 0.5)];
        const int n_same = static_cast<int>(a.el.size()) - a.n_sw;
        sum_min += a.el.front() * n / steps;
        sum_mean += s / steps;
        std::fprintf(stderr, "  %-10s %-6s %7.1f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8d %8.3f %8.3f\n", name.c_str(),
                     a.k->ctx.c_str(), n / steps, a.el.front(), s / n, p90,
                     a.submit / n, a.gap / n, a.blocked / n, a.n_sw, a.n_sw ? a.el_sw / a.n_sw : 0.0,
                     n_same ? a.el_same / n_same : 0.0);
    }
    std::fprintf(stderr,
                 "  per step: wall %.1f ms; dispatch sum of MINIMA %.1f, sum of MEANS %.1f (variance %.1f);"
                 " host: embed %.2f, patch %.2f, route %.2f, lm read %.2f\n",
                 trace_wall_ms_ / steps, sum_min, sum_mean, sum_mean - sum_min, trace_embed_ms_ / steps,
                 trace_patch_ms_ / steps, trace_route_ms_ / steps, trace_lm_ms_ / steps);
    std::fprintf(stderr, "\n");
    trace_.clear();
    trace_steps_ = 0;
    trace_wall_ms_ = trace_route_ms_ = trace_patch_ms_ = trace_embed_ms_ = trace_lm_ms_ = 0;
}

void Core::route(Kern& k, int layer, uint64_t act_off) {
    auto t0 = std::chrono::steady_clock::now();
    if (k.moe2.empty()) throw std::runtime_error("open_qwen36: moeroute2 on " + k.name + ", which has no routed-expert table");
    xrt::bo& act = act_[layer];
    const size_t off = act_off + w_->rout_idx_off;
    if (route_log_on_) {
        // the whole record: the router's probabilities, then idx and the weights
        const size_t n = w_->rout_idx_off + 64;
        read_back(act, n, act_off);
        const uint8_t* m = act.map<uint8_t*>() + act_off;
        route_log_.push_back({layer, std::vector<uint8_t>(m, m + n)});
    } else {
        read_back(act, 32, off);
    }
    uint32_t idx[8];
    std::memcpy(idx, act.map<uint8_t*>() + off, 32);
    // OPEN-REQUEST-ISOLATION. The wait on this layer's first dispatch returning is NOT
    // enough for the host to see the router record it wrote: roughly one read in 3000 (in
    // the periods where it happens at all) still returns the PREVIOUS step's idx, and a
    // re-sync microseconds later returns the new one (repeat-bug.md). The previous step's
    // idx is a valid expert list, so nothing downstream notices -- the step just runs the
    // wrong experts. arm_route_records() overwrote the slot with a sentinel before this
    // step's dispatches, so a read that still shows it has not landed yet: sync again.
    if (route_sentinel_) {
        auto armed = [&] {
            for (unsigned s = 0; s < w_->moe.topk; ++s)
                if (idx[s] == kRouteSentinel) return true;
            return false;
        };
        ++route_reads_;
        if (armed()) {
            ++route_late_;
            const auto tw = std::chrono::steady_clock::now();
            do {
                if (ms_since(tw) > 1000.0)
                    throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " at position " +
                                             std::to_string(pos_) + ": the router record never landed (1 s after its "
                                             "dispatch completed)");
                read_back(act, 32, off);
                std::memcpy(idx, act.map<uint8_t*>() + off, 32);
            } while (armed());
        }
    }
    if (route_check_) {
        // read it again twice: at once, and a millisecond later, both through a fresh sync
        uint32_t again[8], later[8];
        read_back(act, 32, off);
        std::memcpy(again, act.map<uint8_t*>() + off, 32);
        const auto tw = std::chrono::steady_clock::now();
        while (ms_since(tw) < 1.0) {}
        read_back(act, 32, off);
        std::memcpy(later, act.map<uint8_t*>() + off, 32);
        ++route_checks_;
        if (std::memcmp(idx, later, 32) || std::memcmp(again, later, 32)) {
            ++route_stale_;
            std::fprintf(stderr, "open_qwen36: route layer %d pos %d: idx read %s / again %s / 1 ms later %s\n", layer,
                         pos_, fmt_idx(idx).c_str(), fmt_idx(again).c_str(), fmt_idx(later).c_str());
        }
    }
    for (unsigned s = 0; s < w_->moe.topk; ++s)
        if (idx[s] >= w_->moe.experts) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(idx[s]));
    stream_patch::moe2_apply(k.iw(), k.moe2, idx, w_->moe);
    k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timing_.route_ms += ms_since(t0);
}

void Core::arm_route_records() {
    // Every router record's idx slot -> the sentinel, so route() can tell a record this step's
    // dispatch wrote from one it has not landed yet. Called with nothing outstanding; the
    // first dispatch of each layer overwrites the whole record before anything reads it.
    for (int l = 0; l < nl_; ++l)
        for (const Step& s : types_[l]->program) {
            if (s.op == "run") continue;
            const size_t off = s.act_off + w_->rout_idx_off;
            uint32_t* p = reinterpret_cast<uint32_t*>(act_[l].map<uint8_t*>() + off);
            for (int i = 0; i < 8; ++i) p[i] = kRouteSentinel;
            act_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, 32, off);
        }
}

void Core::step(int token, bool want_logits) { step_impl(token, nullptr, want_logits, nullptr); }

void Core::embedding_row(size_t token, float* out) const {
    file_->embed_row(w_->embed_tensor, token, w_->hidden, out);
    if (w_->embed_scale != 1.0)
        for (size_t i = 0; i < w_->hidden; ++i)
            out[i] = static_cast<float>(out[i] * w_->embed_scale);
}

void Core::step_embed(const float* x, bool want_logits, const int64_t mpos[3],
                      const float* deepstack, int n_deepstack) {
    if (!has_mrope()) throw std::runtime_error("open_qwen36: step_embed on a model without M-RoPE");
    if (n_deepstack > nl_)
        throw std::runtime_error("open_qwen36: " + std::to_string(n_deepstack) +
                                 " deepstack features but only " + std::to_string(nl_) + " layers are running");
    step_impl(-1, x, want_logits, mpos, deepstack, n_deepstack);
}

void Core::mrope_begin() {
    if (mrope_on_) return;
    mrope_on_ = true;
    mrope_pos_ = pos_;
}

void Core::write_record(size_t row, const double pos[3]) {
    for (const auto& [name, rg] : w_->per_row_globals) {
        xrt::bo& bo = globals_.at(name);
        uint8_t* r = bo.map<uint8_t*>() + row * rg.per_row;
        pools::build_ptab_record(*w_, rg, row, pos, mrope_section_, mrope_interleaved_, r);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, rg.per_row, row * rg.per_row);
    }
    if (row + 1 > ptab_dirty_) ptab_dirty_ = row + 1;
}

void Core::step_impl(int token, const float* x, bool want_logits, const int64_t* mpos,
                     const float* deepstack, int n_deepstack) {
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step before load_weights");
    if (static_cast<size_t>(pos_) >= cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: position " + std::to_string(pos_) + " reached the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    if (!x && (token < 0 || static_cast<size_t>(token) >= w_->vocab)) throw std::runtime_error("open_qwen36: token id out of range");
    // ---- Stage 2.6 Gate B / Stage 2.7 Gate A: the decode route. Runtime-
    // selectable (npu / host / auto against the measured threshold); the
    // whole step runs as the 2.5 gemm-block chain at T=1 with host attention,
    // or as the fused per-layer dx dispatch walk below. The NPU route itself
    // is untouched -- route npu with the env off is the byte-anchored 2.2/2.3
    // path, which is what makes every A/B one binary with one switch.
    if (decode_step_is_host()) {
        if (deepstack || mpos || mrope_on_)
            throw std::runtime_error("open_qwen36: host decode route does not "
                                     "implement this step's deepstack/mrope inputs (refusing rather than "
                                     "silently falling back to the NPU route)");
        step_host_decode(token, x, want_logits);
        return;
    }
    if (cpu_decode_on_ && !x && !deepstack && !mpos && !mrope_on_ && cpu_decode_ok()) {
        step_cpu_decode(token, want_logits);
        return;
    }
    if (drop_pools_)
        throw std::runtime_error("open_qwen36: the q4 pools were dropped for the CPU decode route (OFLM_OPEN_CPU_DECODE=1), "
                                 "and this step cannot take it; OFLM_OPEN_KEEP_POOLS=1 keeps the NPU route");
    if (state_owner_ == 1) states_to_device();
    state_owner_ = 2;
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    xrt::bo& xres = buffer("xres", 0);
    if (x)
        std::memcpy(xres.map<float*>(), x, w_->hidden * 4);
    else
        embedding_row(static_cast<size_t>(token), xres.map<float*>());
    if (cfg_.verbose && !x) {
        const float* e = xres.map<float*>();
        int nnan = 0;
        float mx = 0.f;
        for (size_t i = 0; i < w_->hidden; ++i) {
            if (!std::isfinite(e[i])) ++nnan;
            else mx = std::max(mx, std::fabs(e[i]));
        }
        log("embed t=" + std::to_string(token) + " nan=" + std::to_string(nnan) + " maxabs=" + std::to_string(mx));
    }
    xres.sync(XCL_BO_SYNC_BO_TO_DEVICE, w_->hidden * 4, 0);
    if (mpos) {
        const double p3[3] = {static_cast<double>(mpos[0]), static_cast<double>(mpos[1]), static_cast<double>(mpos[2])};
        write_record(static_cast<size_t>(pos_), p3);
    } else if (mrope_on_) {
        const double cpos = static_cast<double>(mrope_pos_);
        const double p3[3] = {cpos, cpos, cpos};
        write_record(static_cast<size_t>(pos_), p3);
    }
    timing_.embed_ms = ms_since(t0);
    auto tp = std::chrono::steady_clock::now();
    for (auto& [name, k] : kerns_) {
        if (k.patch != "attnpos") continue;
        stream_patch::attn_apply(k.iw(), k.attn, static_cast<uint64_t>(pos_), k.geom);
        k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    timing_.patch_ms = ms_since(tp);
    if (route_sentinel_) arm_route_records();

    // ---- the layer walk (OPEN-DECODE-PIPELINE)
    //
    // Serially this is `start(); wait();` for each of a step's ~82 dispatches, so the array
    // sits idle for a host turnaround at every one of them. Pipelined, layer l + 1's FIRST
    // dispatch is queued behind layer l's LAST one before the host waits on anything, and the
    // device starts it the instant the previous command retires.
    //
    // The ordering invariant, and the exact wait that earns each part of it:
    //
    //  1. A kernel's instruction BO is shared by every layer that runs it (core.hpp's Kern),
    //     so route() must never patch one while a run on it may still be executing. route()
    //     patches the layer's SECOND kernel (lx1 / ax1) -- which is the same object as the
    //     PREVIOUS layer's second kernel whenever the two layers share a type -- so the loop
    //     waits `tail_in` (layer l - 1's second dispatch) before calling route() for layer l.
    //     attn_apply patches ax0 once per step, above, with nothing outstanding at all.
    //  2. route() reads the router's top-k out of act[l], which layer l's first dispatch
    //     writes, so `head` is waited before route() too.
    //  3. Per-layer buffers (pool / consts / act / state) are distinct per layer. `xres` is
    //     the one buffer every layer writes, and layer l + 1's first dispatch reads what
    //     layer l's second one wrote. Nothing on the host orders those two: it holds only
    //     because the device executes the commands in submission order. xrt_kernel.h promises
    //     no such thing -- start() is documented as "asynchronous" and nothing in the header
    //     says two outstanding runs retire in order -- so the guarantee here is EMPIRICAL.
    //     Level 1 therefore queues ahead only within one hardware context, where the two
    //     commands share a queue, and OPEN-DECODE-PIPELINE's gate is what proves it: a reorder
    //     would read a stale xres and every logit after it would move, and none of them does
    //     (max |diff| 0.0). Level 2 crosses contexts and is NOT safe -- it does not merely risk
    //     a reorder, it hangs the array (ERT state 8 on the next `ax` dispatch, twice in two
    //     runs, both on the step's first cross-context queue-ahead). It stays only as the probe
    //     that established that.
    //
    // Deepstack (Qwen3-VL) reads xres back between layers, which needs the layer's last
    // dispatch to have landed; the pipeline is off for that path rather than special-cased.
    const bool pipe = submit_ahead_ > 0 && !deepstack;
    Inflight head, tail_in;                 // layer l's first dispatch; layer l - 1's last
    for (int l = 0; l < nl_; ++l) {
        const std::vector<Step>& prog = types_[l]->program;
        const bool shape_ok = !prog.empty() && prog.front().op == "run" && prog.back().op == "run";
        if (!pipe || !shape_ok) {
            // the serial path, unchanged: every step of this layer start-to-wait
            if (tail_in.active()) {
                const auto [elapsed, blocked] = wait_run(tail_in);
                timing_.part1_ms += blocked;
                timing_.dispatch_ms += elapsed;
            }
            int nrun = 0;
            for (const Step& s : prog) {
                Kern& k = kerns_.at(s.kernel);
                if (s.op == "run") {
                    const double ms = run(k, s.args, l);
                    (nrun++ == 0 ? timing_.part0_ms : timing_.part1_ms) += ms;
                    timing_.dispatch_ms += ms;
                } else {
                    route(k, l, s.act_off);
                }
            }
            if (deepstack && l < n_deepstack) {
                read_back(xres, w_->hidden * 4, 0);
                float* r = xres.map<float*>();
                const float* f = deepstack + static_cast<size_t>(l) * w_->hidden;
                for (size_t i = 0; i < w_->hidden; ++i) r[i] += f[i];
                xres.sync(XCL_BO_SYNC_BO_TO_DEVICE, w_->hidden * 4, 0);
            }
            continue;
        }
        // (1) the previous layer's last dispatch, so its instruction BO is free to patch and
        // the xres it writes is final. Where this layer's first dispatch was NOT queued ahead
        // (layer 0, or a hardware-context boundary at level 1), it must land BEFORE that
        // dispatch is even submitted -- starting it first is exactly the reorder level 1
        // refuses to rely on.
        if (!head.active() && tail_in.active()) {
            const auto [elapsed, blocked] = wait_run(tail_in);
            timing_.part1_ms += blocked;
            timing_.dispatch_ms += elapsed;
        }
        if (!head.active()) {
            head = start_run(kerns_.at(prog.front().kernel), prog.front().args, l);
            timing_.part0_ms += head.submit_ms;
            timing_.dispatch_ms += head.submit_ms;
        }
        if (tail_in.active()) {
            const auto [elapsed, blocked] = wait_run(tail_in);
            timing_.part1_ms += blocked;
            timing_.dispatch_ms += elapsed;
        }
        if (prog.size() == 1) {
            // A dense family's layer is ONE dispatch with nothing on the host after it, so it
            // is itself what the next layer queues behind -- there is no second half to wait
            // for and no instruction stream to patch between the two.
            tail_in = std::move(head);
            head.k = nullptr;
        } else {
            // (2) this layer's first dispatch, so act[l] holds the router record
            {
                const auto [elapsed, blocked] = wait_run(head);
                timing_.part0_ms += blocked;
                timing_.dispatch_ms += elapsed;
            }
            // whatever sits between the two dispatches (the router), then the last one
            for (size_t i = 1; i + 1 < prog.size(); ++i) {
                const Step& s = prog[i];
                if (s.op == "run") {
                    const double ms = run(kerns_.at(s.kernel), s.args, l);
                    timing_.part1_ms += ms;
                    timing_.dispatch_ms += ms;
                } else {
                    route(kerns_.at(s.kernel), l, s.act_off);
                }
            }
            tail_in = start_run(kerns_.at(prog.back().kernel), prog.back().args, l);
            timing_.part1_ms += tail_in.submit_ms;
            timing_.dispatch_ms += tail_in.submit_ms;
        }
        // (3) and the next layer's first dispatch, queued behind it
        if (l + 1 < nl_) {
            const std::vector<Step>& next = types_[l + 1]->program;
            const bool next_ok = !next.empty() && next.front().op == "run" && next.back().op == "run";
            Kern& nk = next_ok ? kerns_.at(next.front().kernel) : *tail_in.k;
            if (next_ok && (submit_ahead_ >= 2 || nk.ctx == tail_in.k->ctx)) {
                head = start_run(nk, next.front().args, l + 1);
                timing_.part0_ms += head.submit_ms;
                timing_.dispatch_ms += head.submit_ms;
            }
        }
    }
    // The tail's norm reads the same xres, from its own hardware context, so only level 2
    // queues it behind the last layer.
    Inflight tail0;
    if (want_logits && submit_ahead_ >= 2 && tail_in.active() && !w_->tail.empty() &&
        w_->tail.front().op == "run") {
        tail0 = start_run(kerns_.at(w_->tail.front().kernel), w_->tail.front().args, 0);
        timing_.dispatch_ms += tail0.submit_ms;
    }
    if (tail_in.active()) {
        const auto [elapsed, blocked] = wait_run(tail_in);
        timing_.part1_ms += blocked;
        timing_.dispatch_ms += elapsed;
    }
    if (want_logits) {
        if (cfg_.verbose) {
            xres.sync(XCL_BO_SYNC_BO_FROM_DEVICE, w_->hidden * 4, 0);
            const float* x = xres.map<float*>();
            int nnan = 0;
            float mx = 0.f;
            for (size_t i = 0; i < w_->hidden; ++i) {
                if (!std::isfinite(x[i])) ++nnan;
                else mx = std::max(mx, std::fabs(x[i]));
            }
            log("xres after layers nan=" + std::to_string(nnan) + " maxabs=" + std::to_string(mx));
        }
        auto t1 = std::chrono::steady_clock::now();
        size_t first = 0;
        if (tail0.active()) {
            timing_.dispatch_ms += wait_run(tail0).first;
            first = 1;
        }
        for (size_t i = first; i < w_->tail.size(); ++i)
            timing_.dispatch_ms += run(kerns_.at(w_->tail[i].kernel), w_->tail[i].args, 0);
        xrt::bo& lg = buffer("logits", 0);
        read_back(lg, w_->vocab * 4, 0);
        std::memcpy(logits_host_.data(), lg.map<uint8_t*>(), w_->vocab * 4);
        timing_.lmhead_ms = ms_since(t1);
    }
    ++pos_;
    if (!mpos && mrope_on_) ++mrope_pos_;      // a text token after an image: (c, c, c), then c + 1
    timing_.total_ms = ms_since(t0);
    if (step_trace_) {
        ++trace_steps_;
        trace_wall_ms_ += timing_.total_ms;
        trace_route_ms_ += timing_.route_ms;
        trace_patch_ms_ += timing_.patch_ms;
        trace_embed_ms_ += timing_.embed_ms;
        trace_lm_ms_ += timing_.lmhead_ms;
    }
}

// ============================================================================
// Stage 2.7 Gate A: the runtime decode route selector. The 2.6 env remains as
// the constructor-time default; set_decode_route() overrides it at any moment
// -- including between steps of one process, which is the mid-stream switch
// this stage exists to test. Since the stage 2.10 promotion Auto is NPU-first
// (the fast attention path wins at every measured window; see the constructor
// comment), and OFLM_DECODE_AUTO_THRESHOLD is the experimental escape hatch
// that forces the old position split.
// ============================================================================

void Core::set_decode_route(const std::string& route) {
    const std::string prev = decode_route_at(static_cast<size_t>(pos_ < 0 ? 0 : pos_));
    if (route == "npu") decode_route_ = DecodeRoute::Npu;
    else if (route == "host") decode_route_ = DecodeRoute::Host;
    else if (route == "auto") decode_route_ = DecodeRoute::Auto;
    else throw std::runtime_error("open_qwen36: set_decode_route: unknown route '" + route + "' (npu | host | auto)");
    const std::string now = decode_route_at(static_cast<size_t>(pos_ < 0 ? 0 : pos_));
    if (now != prev) log("decode route: " + now + " (runtime switch at position " + std::to_string(pos_) + ")");
}

std::string Core::decode_route_at(size_t pos) const {
    switch (decode_route_) {
        case DecodeRoute::Host: return "host";
        case DecodeRoute::Npu: return "npu";
        case DecodeRoute::Auto: return pos < decode_auto_threshold_ ? "npu" : "host";
    }
    return "npu";
}

// ============================================================================
// Stage 2.6 Gate B (OFLM_OPEN_HOST_ATTN_DECODE=1): the decode step through the
// 2.5 gemm-block chain at T=1 with host attention, instead of the fused
// per-layer dx dispatch. One binary, one env: the NPU decode route (step_impl's
// layer walk) is untouched, and this route never runs unless the env names it,
// so the A/B is the same binary with the flag flipped. The projections and the
// FFN ride the same five NPU gemm_q4_prefill dispatches the prefill route uses
// (padded to the kernel set's compiled T, real token in column 0), the
// attention is host_attn_layer (2.5's implementation, unchanged), and the tail
// is the same NPU ln+lm pair -- so the route's cost splits cleanly into
// part0_ms (NPU GEMM), route_ms (host attention) and lmhead_ms for Gate B.
// ============================================================================

void Core::step_host_decode(int token, const float* x, bool want_logits) {
    apply_thread_budget();
    // Fail-closed: this route IS the dense gemm-block chain, so every layer must
    // carry one, all at the kernel set's compiled block T. host_attn_layer()
    // then does its own geometry check against the manifest's hf_config_check
    // and refuses anything that is not the full-rotary no-qknorm no-gate dense
    // shape, exactly as in 2.5.
    if (!gemm_block_t_)
        throw std::runtime_error("open_qwen36: host decode route: this kernel set has no gemm_block program");
    for (int l = 0; l < nl_; ++l) {
        const GemmBlockProgram& gb = types_[l]->gemm_block;
        if (gb.t != gemm_block_t_ || gb.kind != "dense" || gb.program.size() != 5)
            throw std::runtime_error("open_qwen36: host decode route: layer " + std::to_string(l) + " (" +
                                     types_[l]->name + ") is not a dense gemm-block type (refusing rather than "
                                     "computing it wrong)");
    }
    const size_t hid = w_->hidden;
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    // The residual row, host fp64 like the gemm-block route's own stream. The
    // previous step left it in the xres global -- the NPU route's own running
    // stream: tail_logits writes it there after a prefill, every host step
    // below puts it back -- so a mid-stream switch of routes reads the same
    // state in either direction.
    std::vector<double> xres(hid);
    {
        auto te = std::chrono::steady_clock::now();
        if (x) {
            for (size_t c = 0; c < hid; ++c) xres[c] = static_cast<double>(x[c]);
        } else {
            std::vector<float> row(hid);
            file_->bf16_row(w_->embed_tensor, static_cast<size_t>(token), hid, row.data());
            for (size_t c = 0; c < hid; ++c) xres[c] = static_cast<double>(row[c]);
        }
        timing_.embed_ms += ms_since(te);
    }
    // The attention must come from the host on this route: in_host_decode_
    // forces the 2.5 branch in step_gemm_block_layer for these calls only,
    // leaving the gemm-block PREFILL route (OFLM_OPEN_HOST_ATTN) untouched.
    in_host_decode_ = true;
    for (int l = 0; l < nl_; ++l) step_gemm_block_layer(l, xres, 1);
    in_host_decode_ = false;

    // Publish the residual where every route keeps it, then the NPU tail
    // (ln g2 + lm) exactly as the dx route's own last dispatches do.
    std::vector<float> row(hid);
    for (size_t c = 0; c < hid; ++c) row[c] = static_cast<float>(xres[c]);
    if (want_logits) {
        tail_logits(row.data());
    } else {
        auto te = std::chrono::steady_clock::now();
        xrt::bo& xb = buffer("xres", 0);
        std::memcpy(xb.map<uint8_t*>(), row.data(), hid * 4);
        xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, hid * 4, 0);
        timing_.state_ms += ms_since(te);
    }
    ++pos_;
    timing_.total_ms = ms_since(t0);
}

// ============================================================================
// 0167/#32: the GEMM prefill route. See manifest.hpp's GemmBlockProgram
// docstring for the shape of the chain; core.hpp's field comments explain
// each buffer's lifetime and why it is per-layer vs global.
// ============================================================================

std::pair<size_t, size_t> Core::op_region(const LayerType& lt, const std::string& from, size_t idx) const {
    const auto& ops = from == "pool" ? lt.pool : lt.consts;
    if (idx >= ops.size())
        throw std::runtime_error("open_qwen36: op_region: " + from + " op " + std::to_string(idx) + " out of range (" +
                                 std::to_string(ops.size()) + " ops)");
    const PackOp& op = ops[idx];
    // the GEMM dequantises the q4_1 band law, which is what std_perm writes; a q8_perm
    // projection has no route (the recipe does not emit one) and is refused here too
    if (op.op != "std_perm" && op.op != "std_perm_gguf")
        throw std::runtime_error("open_qwen36: op_region: " + from + " op " + std::to_string(idx) + " is a " + op.op +
                                 ", not a band-law projection the GEMM reads");
    return {static_cast<size_t>(op.dst), static_cast<size_t>(op.nch) * w_->chunk_bytes};
}

std::string Core::const_tensor(const LayerType& lt, const std::string& suffix, int layer) const {
    for (const PackOp& op : lt.consts) {
        const std::string& t = op.tensor;
        if (t.size() >= suffix.size() && t.compare(t.size() - suffix.size(), suffix.size(), suffix) == 0) {
            std::string name = t;
            const size_t at = name.find("{l}");
            if (at != std::string::npos) name.replace(at, 3, std::to_string(layer));
            return name;
        }
    }
    throw std::runtime_error("open_qwen36: layer type " + lt.name + " has no consts tensor ending in " + suffix);
}

const float* Core::gemm_run(const Step& s, const float* x, size_t T, size_t K, size_t N, int layer) {
    // A split step's weight is [hi | lo] of a q8 projection's exact q4_1 split (OPEN-PREFILL-BATCH):
    // the dispatch returns 2N rows, and rows N.. are added into rows 0.. below, so every caller
    // sees the projection's [N, T] whether its weight was split or not.
    const size_t rows = s.split ? 2 * N : N;
    xrt::bo& xb = buffer(s.args[1], 0);
    xrt::bo& yb = buffer(s.args[2], 0);
    if (xb.size() < K * T * 2 || yb.size() < rows * T * 4)
        throw std::runtime_error("open_qwen36: gemm " + s.kernel + ": the x / y globals are smaller than [" +
                                 std::to_string(K) + "] x " + std::to_string(T) + " -> [" + std::to_string(rows) + "]");
    auto t0 = std::chrono::steady_clock::now();
    host::tile_x(x, T, K, xb.map<uint16_t*>());          // straight into the mapped buffer
    timing_.part1_ms += ms_since(t0);
    timing_.gemm_tile_ms += ms_since(t0);
    auto ts = std::chrono::steady_clock::now();
    xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, K * T * 2, 0);
    timing_.sync_ms += ms_since(ts);
    timing_.part1_ms += ms_since(ts);
    timing_.part0_ms += run(kerns_.at(s.kernel), s.args, layer);
    ts = std::chrono::steady_clock::now();
    read_back(yb, rows * T * 4, 0);
    timing_.sync_ms += ms_since(ts);
    timing_.part1_ms += ms_since(ts);
    const float* y = yb.map<const float*>();
    if (!s.split) return y;
    // The sum goes to host memory, never back into the mapped y: the next read_back's
    // CLFLUSH writes a dirty line back over whatever the device wrote there since, so
    // folding in place corrupts a later dispatch's output -- by how much depends on what
    // got evicted in between, and the route stops being deterministic.
    ts = std::chrono::steady_clock::now();
    std::vector<float>& sum = split_y_[s.args[2]];       // one per y global, so it lives as long as y would
    if (sum.size() < N * T) sum.resize(N * T);
    const float* lo = y + N * T;
    float* o = sum.data();
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(N * T); ++i) o[i] = y[i] + lo[i];
    timing_.part1_ms += ms_since(ts);
    timing_.gemm_tr_ms += ms_since(ts);
    return o;
}

void Core::gemm(const Step& s, const float* x, size_t T, size_t K, size_t N, int layer,
                std::vector<float>& out) {
    const float* y = gemm_run(s, x, T, K, N, layer);
    auto t1 = std::chrono::steady_clock::now();
    if (out.size() < T * N) out.resize(T * N);             // grow-only, so only the first block pays for it
    host::transpose(y, N, T, out.data());                  // [N, T] on the device -> [T, N]
    timing_.part1_ms += ms_since(t1);
    timing_.gemm_tr_ms += ms_since(t1);
}

void Core::states_to_device() {
    for (int l = 0; l < nl_; ++l) state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    state_owner_ = 0;
}

void Core::states_to_host() {
    for (int l = 0; l < nl_; ++l) read_back(state_[l], state_[l].size(), 0);
    state_owner_ = 0;
}

bool Core::cpu_decode_ok() const {
    if (!dit_on_ || !dit_b4_ || !gemm_block_t_ || static_cast<int>(hc_.size()) != nl_) return false;
    for (int l = 0; l < nl_; ++l) {
        const GemmBlockProgram& gb = types_[l]->gemm_block;
        if ((gb.kind != "linear" && gb.kind != "full") || gb.program.size() < 2 || gb.ffn_program.size() < 2) return false;
        for (const Step* st : {&gb.program[0], &gb.program[1], &gb.ffn_program[0], &gb.ffn_program[1]}) {
            auto it = gemm_w_.find(st->args[0]);
            if (it == gemm_w_.end() || it->second.size() != static_cast<size_t>(nl_)) return false;
        }
    }
    return true;
}

void Core::step_cpu_decode(int token, bool want_logits) {
    apply_thread_budget();
    // OFLM_OPEN_DECODE_THREADS (default 8, 1-16): the cores a CPU decode step uses, so the rest
    // of the machine stays usable. Reading memory needs about 4 cores per CPU die (8 reach
    // ~108 GB/s); the int4 GEMV's own math needs more today (16 for ~97 GB/s).
    static const int dthreads = [] {
        const char* e = std::getenv("OFLM_OPEN_DECODE_THREADS");
        const int n = e ? std::atoi(e) : 8;
        return std::max(1, std::min(16, n));
    }();
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(dthreads);
    struct Restore { int n; ~Restore() { omp_set_num_threads(n); } } restore{saved_threads};
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};
    if (state_owner_ == 2) states_to_host();
    const size_t hid = w_->hidden, pos = static_cast<size_t>(pos_);
    const auto& D = man_.gemm_dit;
    // a GEMM step's [N, K] (its route kernel gemm_n<N>_k<K>) and its padded N
    auto shape = [&](const Step& st, size_t& n, size_t& k, size_t& npad) {
        const std::string& kn = st.kernel;
        n = std::stoul(kn.substr(6, kn.find("_k") - 6));
        k = std::stoul(kn.substr(kn.find("_k") + 2));
        npad = D["npad"][std::to_string(n)].get<size_t>();
    };
    // gemm_dit.decode.split (OFLM_OPEN_DECODE_SPLIT, default on): a GEMV's first nn columns on the
    // NPU -- a 256-row dit2 GEMM on 64-row tiles, row 0 real -- while the CPU does the rest of the
    // same int4 buffer; the NPU's row 0 read back as bf16
    static const bool split_on = [] { const char* e = std::getenv("OFLM_OPEN_DECODE_SPLIT"); return !(e && *e == '0'); }();
    const nlohmann::json* SP = split_on && D.contains("decode") ? &D["decode"] : nullptr;
    auto gemv = [&](const Step& st, int l, const float* in, std::vector<float>& out) {
        size_t n, k, npad;
        shape(st, n, k, npad);
        out.resize(npad);
        auto tg = std::chrono::steady_clock::now();
        const uint8_t* B = gemm_w_.at(st.args[0])[l].map<const uint8_t*>();
        const std::string key = std::to_string(n) + "x" + std::to_string(k);
        if (!SP || !(*SP)["split"].contains(key)) {
            host::gemv_b4(B, npad, k, in, out.data());
            timing_.part0_ms += ms_since(tg);
            return;
        }
        const nlohmann::json& sv = (*SP)["split"][key];
        const size_t nn = sv["nn"].get<size_t>(), M = (*SP)["m"].get<size_t>(), mt = (*SP)["mt"].get<size_t>();
        const std::string an = "dec_a_k" + std::to_string(k), cn = "dec_c_n" + std::to_string(nn);
        if (!globals_.count(an)) globals_[an] = alloc(M * k * 9 / 8);     // zero: rows 1.. stay zero
        if (!globals_.count(cn)) globals_[cn] = alloc(M * nn * 2);
        // row 0 into dit2's A order (dit2_pack_a, hs): per 64-K unit of mt rows, quarter 0, the
        // blocks j = 2 kb (row block 0), i = 0
        xrt::bo& ab = globals_.at(an);
        uint8_t* a = ab.map<uint8_t*>();
        const size_t unit = mt / 32 * 32 * 8 * 9;
        for (size_t ks = 0; ks < k / 64; ++ks)
            for (size_t kb = 0; kb < 8; ++kb) bfp_encode8_avx2(in + ks * 64 + kb * 8, a + ks * unit + (2 * kb) * 8 * 9);
        ab.sync(XCL_BO_SYNC_BO_TO_DEVICE, k / 64 * unit, 0);
        Inflight f = start_run(kerns_.at(sv["kernel"].get<std::string>()), {an, st.args[0], cn}, l);
        timing_.moe_prep_ms += ms_since(tg);      // trace: pack + flush + start before the CPU's share
        host::gemv_b4(B + nn / 128 * (k / 64) * 4224, npad - nn, k, in, out.data() + nn);
        const double cpu_ms = ms_since(tg), attn0 = timing_.attn_ms;
        auto tw = std::chrono::steady_clock::now();
        wait_run(f);
        timing_.attn_ms += ms_since(tw);
        xrt::bo& cb = globals_.at(cn);
        read_back(cb, nn * 2, 0);
        const uint16_t* c16 = cb.map<const uint16_t*>();
        for (size_t i = 0; i < nn; ++i) out[i] = bf16_to_f32(c16[i]);
        timing_.moe_read_ms += ms_since(tg) - cpu_ms - (timing_.attn_ms - attn0);   // trace: read back + convert
        timing_.part0_ms += ms_since(tg);
    };
    static thread_local std::vector<float> x, xn, y, og, h, ya, yb, yc, yd;
    x.resize(hid); xn.resize(hid);
    file_->bf16_row(w_->embed_tensor, static_cast<size_t>(token), hid, x.data());
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        const GemmBlockProgram& gb = lt.gemm_block;
        const HostConsts& hc = hc_[l];
        host::rmsnorm_rows(x.data(), 1, hid, hc.ln.data(), gb.eps, xn.data());
        gemv(gb.program[0], l, xn.data(), y);
        auto th = std::chrono::steady_clock::now();
        uint8_t* sp = state_[l].map<uint8_t*>();
        if (gb.kind == "linear") {
            const size_t nch = gb.qkv_dim, vw = gb.vw;
            og.resize(vw);
            host::DeltaGeom g;
            g.T = 1; g.hid = hid;
            g.key_heads = gb.key_heads; g.value_heads = gb.value_heads; g.head_dim = gb.head_dim;
            g.taps = gb.conv_kernel; g.lanes = hc.lanes; g.s_rows = gb.s_rows; g.eps = gb.eps;
            g.t_real = 1;
            double ph[2] = {0, 0};
            host::deltanet_block(g, y.data(), y.data() + nch, xn.data(), hc.convw.data(), hc.Wa.data(), hc.Wb.data(),
                                 hc.A.data(), hc.dtb.data(), hc.nw.data(), reinterpret_cast<uint16_t*>(sp),
                                 reinterpret_cast<float*>(sp + gb.state_s_off), og.data(), ph);
        } else {
            const size_t qw = gb.qw, kvw = gb.kvw, row = lt.state_row;
            og.resize(qw);
            host::AttnGeom g;
            g.T = 1; g.t_real = 1; g.nh = gb.nh; g.kvh = gb.kvh; g.hd = gb.hd; g.rot = gb.rot;
            g.pos0 = pos; g.eps = gb.eps;
            host::attention_block(g, y.data(), y.data() + qw, y.data() + qw + kvw, y.data() + qw + 2 * kvw, hc.qn.data(),
                                  hc.kn.data(), man_.rope_inv_freq.data(), state_[l].map<uint16_t*>(), row / 2, og.data());
        }
        timing_.part1_ms += ms_since(th);
        const std::vector<float> mid = og;
        gemv(gb.program[1], l, mid.data(), y);
        for (size_t i = 0; i < hid; ++i) x[i] += y[i];
        host::rmsnorm_rows(x.data(), 1, hid, hc.postln.data(), gb.eps, xn.data());
        gemv(gb.ffn_program[0], l, xn.data(), h);
        // up|gate C: every 128-column tile is [gate 64 | up 64]; h = silu(gate) * up
        const size_t ff = h.size() / 2;
        og.resize(ff);
        host::swiglu_tiles(h.data(), ff, og.data());
        gemv(gb.ffn_program[1], l, og.data(), y);
        for (size_t i = 0; i < hid; ++i) x[i] += y[i];
    }
    state_owner_ = 1;
    ++pos_;
    if (want_logits) tail_logits(x.data());
    timing_.total_ms = ms_since(t0);
    static const bool trace = std::getenv("OFLM_OPEN_CPU_DECODE_TRACE") != nullptr;
    if (trace)
        std::fprintf(stderr, "cpu decode pos %zu: %.1f ms | gemv %.1f (npu wait %.1f) mixer %.1f lm %.1f | split pre %.1f post %.1f\n", pos,
                     timing_.total_ms, timing_.part0_ms, timing_.attn_ms, timing_.part1_ms, timing_.lmhead_ms, timing_.moe_prep_ms,
                     timing_.moe_read_ms);
}

void Core::tail_logits(const float* row) {
    auto t1 = std::chrono::steady_clock::now();
    if (cpu_head_) {
        const size_t hid = w_->hidden;
        if (normw_f_.empty()) {
            const uint16_t* nw = buffer("normw", 0).map<const uint16_t*>();
            normw_f_.resize(hid);
            for (size_t i = 0; i < hid; ++i) normw_f_[i] = bf16_to_f32(nw[i]);
        }
        static thread_local std::vector<float> xn;
        xn.resize(hid);
        host::rmsnorm_rows(row, 1, hid, normw_f_.data(), types_[0]->gemm_block.eps, xn.data());
        host::lmhead_q8_cpu(lm_cpu_.data(), w_->vocab, hid, xn.data(), logits_host_.data());
        timing_.lmhead_ms = ms_since(t1);
        if (!cpu_head_check_) return;
        // debug: the NPU's head on the same row, compared
        std::vector<float> cpu(logits_host_);
        xrt::bo& x0 = buffer("xres", 0);
        std::memcpy(x0.map<uint8_t*>(), row, hid * 4);
        x0.sync(XCL_BO_SYNC_BO_TO_DEVICE, hid * 4, 0);
        for (const Step& st : w_->tail) run(kerns_.at(st.kernel), st.args, 0);
        xrt::bo& lg = buffer("logits", 0);
        read_back(lg, w_->vocab * 4, 0);
        const float* npu = lg.map<const float*>();
        const size_t V = w_->real_vocab ? w_->real_vocab : w_->vocab;
        double sxy = 0, sxx = 0, syy = 0, mx = 0, my = 0;
        for (size_t i = 0; i < V; ++i) { mx += cpu[i]; my += npu[i]; }
        mx /= V; my /= V;
        size_t ac = 0, an = 0;
        for (size_t i = 0; i < V; ++i) {
            sxy += (cpu[i] - mx) * (npu[i] - my); sxx += (cpu[i] - mx) * (cpu[i] - mx); syy += (npu[i] - my) * (npu[i] - my);
            if (cpu[i] > cpu[ac]) ac = i;
            if (npu[i] > npu[an]) an = i;
        }
        std::fprintf(stderr, "cpu head check: corr %.6f argmax cpu %zu npu %zu\n", sxy / std::sqrt(sxx * syy), ac, an);
        return;
    }
    xrt::bo& xres1 = buffer("xres", 0);
    std::memcpy(xres1.map<uint8_t*>(), row, w_->hidden * 4);
    xres1.sync(XCL_BO_SYNC_BO_TO_DEVICE, w_->hidden * 4, 0);
    for (const Step& s : w_->tail) run(kerns_.at(s.kernel), s.args, 0);
    xrt::bo& lg = buffer("logits", 0);
    read_back(lg, w_->vocab * 4, 0);
    std::memcpy(logits_host_.data(), lg.map<uint8_t*>(), w_->vocab * 4);
    timing_.lmhead_ms = ms_since(t1);
}

void Core::shuttle_buf(xrt::bo& wide, xrt::bo& scratch1, size_t token, size_t act_bytes, bool wide_to_scratch,
                       size_t region_off, size_t region_bytes) {
    // `wide` is an explicit argument rather than a fixed per-layer buffer
    // because the GEMM route's T-wide attention scratch ("gact") is a
    // GLOBAL, not per-layer (act_bytes is uniform across every Granite dense
    // layer, so a per-layer copy would only cost memory).
    //
    // Only [region_off, region_off + region_bytes) moves. The attention dispatch reads
    // q / k / v and writes og and touches nothing else in `act` (designs/dense/dx_attn.py
    // says so in its own header), so shuttling the WHOLE buffer moved about nine bytes
    // for every one that mattered -- 590 KB per token per layer, which at T x layers
    // dispatches a block is gigabytes of sync traffic for nothing. region_bytes == 0
    // keeps the original whole-buffer behaviour for any other caller.
    const size_t n = region_bytes ? region_bytes : act_bytes;
    const size_t r = region_bytes ? region_off : 0;
    const size_t off = token * act_bytes + r;
    if (wide_to_scratch) {
        read_back(wide, n, off);
        std::memcpy(scratch1.map<uint8_t*>() + r, wide.map<uint8_t*>() + off, n);
        scratch1.sync(XCL_BO_SYNC_BO_TO_DEVICE, n, r);
    } else {
        read_back(scratch1, n, r);
        std::memcpy(wide.map<uint8_t*>() + off, scratch1.map<uint8_t*>() + r, n);
        wide.sync(XCL_BO_SYNC_BO_TO_DEVICE, n, off);
    }
}

void Core::rmsnorm_host(const std::vector<double>& x, size_t T, size_t hid, const std::vector<uint16_t>& w_bf16,
                        double eps, std::vector<float>& out, size_t groups) {
    // Reduction AND the final multiply both in fp64 (trap 11: a fp32
    // reduction over 2560+ terms is not a safe correctness metric at this
    // width). out[t,k] = x[t,k]/sqrt(mean_k(x^2)+eps)*w[k].
    // groups > 1 (K2's GroupRMSNorm, Stage 2.2): each row splits into `groups`
    // equal groups, the mean runs over the group alone, eps stays inside the
    // sqrt -- the LN_GROUPS semantics the `ln` design bakes into the NPU
    // kernel (E3's variant C). groups = 1 runs the identical single-group
    // loop the plain form had, so pre-2.2 behaviour is bit-exact.
    out.assign(T * hid, 0.f);
    // Over tokens, which are independent: each row's reduction and its own rms stay
    // exactly where they were, so this is bit-exact against the serial form (the same
    // argument, and the same gate, as the 35B's host DeltaNet conv).
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        const double* row = &x[static_cast<size_t>(t) * hid];
        float* orow = &out[static_cast<size_t>(t) * hid];
        const size_t gw = hid / groups;   // the manifest gate guarantees groups divides hid
        for (size_t gi = 0; gi < groups; ++gi) {
            const double* grp = row + gi * gw;
            double ss = 0;
            for (size_t k = 0; k < gw; ++k) ss += grp[k] * grp[k];
            const double rms = std::sqrt(ss / static_cast<double>(gw) + eps);
            for (size_t k = 0; k < gw; ++k) {
                const double w = static_cast<double>(bf16_to_f32(w_bf16[gi * gw + k]));
                orow[gi * gw + k] = static_cast<float>((grp[k] / rms) * w);
            }
        }
    }
}

void Core::rmsnorm_host(const std::vector<double>& x, size_t T, size_t hid, const std::vector<float>& w_f32,
                        double eps, std::vector<float>& out, size_t groups) {
    // Same reduction as the bf16 overload; the weight arrives already dequantised to f32
    // (read straight from the file, not sliced from packed consts bytes -- see its caller).
    // groups carries the same GroupRMSNorm split as the bf16 overload.
    out.assign(T * hid, 0.f);
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        const double* row = &x[static_cast<size_t>(t) * hid];
        float* orow = &out[static_cast<size_t>(t) * hid];
        const size_t gw = hid / groups;
        for (size_t gi = 0; gi < groups; ++gi) {
            const double* grp = row + gi * gw;
            double ss = 0;
            for (size_t k = 0; k < gw; ++k) ss += grp[k] * grp[k];
            const double rms = std::sqrt(ss / static_cast<double>(gw) + eps);
            for (size_t k = 0; k < gw; ++k)
                orow[gi * gw + k] = static_cast<float>((grp[k] / rms) * static_cast<double>(w_f32[gi * gw + k]));
        }
    }
}

void Core::tile_gemm_x(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out) {
    out.assign(K * T, 0);
    host::tile_x(x_tk.data(), T, K, out.data());
}

void Core::tile_gemm_x_reference(const std::vector<float>& x_tk, size_t T, size_t K, std::vector<uint16_t>& out) {
    // [T,K] fp32 -> bf16, pre-tiled [K,T] "k,n" order (K_TILE=64, MAC 8x8,
    // tile_n=32 -- gemm_q4_prefill.py's own GQP_TILE_N default), matching
    // open_npue/npue_pack.cpp's tile_b algorithm exactly (copied, not
    // called -- that function has internal linkage in its own translation
    // unit). `x_tk` is [T,K] row-major (T rows of K elements, this route's
    // own natural RMSNorm-output layout); the transpose to logical [K,T] is
    // done by indexing, not a separate pass.
    constexpr size_t TK = 64, MAC = 8, TN = 32;
    if (K % TK || T % TN)
        throw std::runtime_error("open_qwen36: gemm-route tile_gemm_x: K=" + std::to_string(K) + " or T=" +
                                 std::to_string(T) + " does not tile by (" + std::to_string(TK) + "," + std::to_string(TN) + ")");
    out.assign(K * T, 0);
    size_t w = 0;
    for (size_t kb = 0; kb < K / TK; ++kb)
        for (size_t nb = 0; nb < T / TN; ++nb)
            for (size_t si = 0; si < TK / MAC; ++si)
                for (size_t ti = 0; ti < TN / MAC; ++ti)
                    for (size_t s = 0; s < MAC; ++s)
                        for (size_t t = 0; t < MAC; ++t) {
                            const size_t r = kb * TK + si * MAC + s;  // K index
                            const size_t c = nb * TN + ti * MAC + t;  // T index
                            out[w++] = f32_to_bf16(x_tk[c * K + r]);
                        }
}

void Core::host_attn_layer(int l, const std::vector<float>& y_qkv3, size_t T, std::vector<float>& og) {
    // Stage 2.5 Gate B (OFLM_OPEN_HOST_ATTN=1). The dxB dispatch's semantics on the
    // host (attn.h line 22): for head h (kv head h / (nh/kvh)), s_t = q'_h . K_t /
    // sqrt(hd) over t in [0, pos] -- cache rows plus the block's own new rows --
    // softmax with an fp32 max / fp64 denominator, and o = softmax(s) V in fp32.
    // RoPE: half-split pairs (i, i + rot/2) over the head's first rot dims, the
    // manifest's own rope_inv_freq, exactly what the ptab records were built
    // from (pools.cpp build_ptab_record) and what block_host's rope() applies.
    // No qk-norm, no output gate: the K2 dense shape. The new cache rows are
    // written bf16, and the row the softmax sees IS the rounded one (the
    // kernel reads the bf16 cache, so the host path rounds identically here).
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t qw = gb.qw, kvw = gb.kvw;
    // Geometry from the manifest's own check data -- the same numbers
    // check_model compares config.json against, so they cannot drift from
    // the model the engine was built for. Anything else: refuse (fail-closed).
    const size_t nh = w_->hf_config_check.at("num_attention_heads").get<size_t>();
    const size_t kvh = w_->hf_config_check.at("num_key_value_heads").get<size_t>();
    const size_t hd = w_->hf_config_check.at("head_dim").get<size_t>();
    const size_t half = w_->rotary_dim / 2;
    if (qw != nh * hd || kvw != kvh * hd || nh % kvh || w_->rotary_dim > hd ||
        half != w_->rope_inv_freq.size() || half * 2 != hd)
        throw std::runtime_error("open_qwen36: host attention: this model's attention geometry is "
                                 "not the full-rotary no-qknorm no-gate dense shape this path "
                                 "implements (refusing rather than computing it wrong)");
    const size_t grp = nh / kvh, P0 = static_cast<size_t>(pos_), rows = P0 + T;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const std::vector<double>& inv_freq = w_->rope_inv_freq;

    xrt::bo& st = buffer(gb.attn_args[3], l);          // "state" under its manifest name
    const size_t row_elems = lt.state_row / 2;          // bf16 elements per row: [K kvw | V kvw]
    uint16_t* kv = st.map<uint16_t*>();

    auto th = std::chrono::steady_clock::now();
    if (P0 > 0) read_back(st, P0 * lt.state_row, 0);    // the cached window, once per layer

    // K/V window as f32: rows [0, P0) from the bf16 cache, [P0, rows) filled below.
    std::vector<float> K(rows * kvw), V(rows * kvw), Q(T * qw);
#pragma omp parallel for
    for (long long r = 0; r < static_cast<long long>(P0); ++r)
        for (size_t j = 0; j < kvw; ++j) {
            K[static_cast<size_t>(r) * kvw + j] = bf16_to_f32(kv[static_cast<size_t>(r) * row_elems + j]);
            V[static_cast<size_t>(r) * kvw + j] = bf16_to_f32(kv[static_cast<size_t>(r) * row_elems + kvw + j]);
        }

    // Phase A (per token, disjoint cache rows): rope q into Q, rope k and round
    // it into the bf16 cache row, round v likewise. Padding tokens are computed
    // too -- the NPU path dispatches them as well, and real columns never read
    // rows behind them, so the real outputs do not depend on the padding.
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        const size_t tk = static_cast<size_t>(t), p = P0 + tk;
        for (size_t h = 0; h < nh; ++h) {
            float* qh = Q.data() + tk * qw + h * hd;   // y_qkv3 is [n_qkv3, T]: column tk
            for (size_t j = 0; j < hd; ++j) qh[j] = y_qkv3[(h * hd + j) * T + tk];
            for (size_t i = 0; i < half; ++i) {
                const double a = static_cast<double>(p) * inv_freq[i];
                const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
                const float x1 = qh[i], x2 = qh[i + half];
                qh[i] = x1 * c - x2 * s;
                qh[i + half] = x2 * c + x1 * s;
            }
        }
        for (size_t h = 0; h < kvh; ++h)
            for (size_t i = 0; i < half; ++i) {
                const double a = static_cast<double>(p) * inv_freq[i];
                const float c = static_cast<float>(std::cos(a)), s = static_cast<float>(std::sin(a));
                const float x1 = y_qkv3[(qw + h * hd + i) * T + tk];
                const float x2 = y_qkv3[(qw + h * hd + i + half) * T + tk];
                K[p * kvw + h * hd + i] = x1 * c - x2 * s;
                K[p * kvw + h * hd + i + half] = x2 * c + x1 * s;
            }
        for (size_t j = 0; j < kvw; ++j) {
            const float vj = y_qkv3[(qw + kvw + j) * T + tk];
            const uint16_t kb = f32_to_bf16(K[p * kvw + j]), vb = f32_to_bf16(vj);
            kv[p * row_elems + j] = kb;
            kv[p * row_elems + kvw + j] = vb;
            K[p * kvw + j] = bf16_to_f32(kb);         // the softmax sees the rounded row
            V[p * kvw + j] = bf16_to_f32(vb);
        }
    }

    // Phase B (per (head, token) pair, disjoint og slices): the causal window
    // [0, p] inclusive, max-softmax in the same numerics block_host's
    // attention_block uses (fp32 dots, fp64 denominator).
    const long long pairs = static_cast<long long>(nh) * static_cast<long long>(T);
#pragma omp parallel for schedule(dynamic, 8)
    for (long long pr = 0; pr < pairs; ++pr) {
        const size_t h = static_cast<size_t>(pr) / T;
        const size_t tk = static_cast<size_t>(pr) % T, p = P0 + tk;
        const float* qv = Q.data() + tk * qw + h * hd;
        const size_t kh_off = (h / grp) * hd;
        std::vector<float> s(p + 1);
        float mx = -1e30f;
        for (size_t r = 0; r <= p; ++r) {
            const float* kr = K.data() + r * kvw + kh_off;
            float acc = 0;
            for (size_t j = 0; j < hd; ++j) acc += kr[j] * qv[j];
            s[r] = acc * scale;
            mx = std::max(mx, s[r]);
        }
        double denom = 0;
        for (size_t r = 0; r <= p; ++r) {
            s[r] = std::exp(s[r] - mx);
            denom += s[r];
        }
        const float inv = static_cast<float>(1.0 / denom);
        float* out = og.data() + tk * qw + h * hd;
        for (size_t j = 0; j < hd; ++j) out[j] = 0;
        for (size_t r = 0; r <= p; ++r) {
            const float* vr = V.data() + r * kvw + kh_off;
            const float a = s[r] * inv;
            for (size_t j = 0; j < hd; ++j) out[j] += a * vr[j];
        }
    }
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, T * lt.state_row, P0 * lt.state_row);
    timing_.route_ms += ms_since(th);
}

void Core::dense_attention_block(int l, const std::vector<float>& y_qkv3, size_t T, size_t t_real,
                                 std::vector<float>& og) {
    const auto t0 = std::chrono::steady_clock::now();
    const double mid0 = timing_.mid_ms;
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const AttnBlock& ab = gb.attn_block;
    const HostConsts& hc = hc_[l];
    const size_t qw = gb.qw, kvw = gb.kvw, hd = ab.hd;
    // The geometry the products were built for, against what this layer's projection produces;
    // attention_npu checks m against the group size times T.
    if (ab.prep != "qknorm_rope" || qw % hd || kvw % hd || (qw / hd) % (kvw / hd) || w_->rotary_dim > hd ||
        hc.qn.size() != hd || hc.kn.size() != hd)
        throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + ": the dense attention products were built for "
                                 "head dim " + std::to_string(hd) + ", which this layer's q / k / v widths do not divide into");
    // y [n_qkv3, T] straight into the [T, width] parts attention_prep reads
    auto tt = std::chrono::steady_clock::now();
    float* q = BlockScratch::fit(bs_.part[0], T * qw);
    float* k = BlockScratch::fit(bs_.part[1], T * kvw);
    float* v = BlockScratch::fit(bs_.part[2], T * kvw);
    const host::TransposePart parts[3] = {{q, 0, qw}, {k, qw, kvw}, {v, qw + kvw, kvw}};
    host::transpose_parts(y_qkv3.data(), T, parts, 3);
    timing_.gemm_tr_ms += ms_since(tt);
    // The window's cached rows on the host map. A decode step (or a dxB block) writes them on
    // the device, so read them back on every block that has any.
    xrt::bo& st = state_[l];
    const size_t row = lt.state_row, pos0 = static_cast<size_t>(pos_);
    tt = std::chrono::steady_clock::now();
    if (pos0 > 0) read_back(st, pos0 * row, 0);
    timing_.state_ms += ms_since(tt);
    host::AttnGeom g;
    g.T = T; g.t_real = t_real; g.nh = qw / hd; g.kvh = kvw / hd; g.hd = hd; g.rot = w_->rotary_dim;
    g.pos0 = pos0; g.eps = gb.eps;
    tt = std::chrono::steady_clock::now();
    float* Q = BlockScratch::fit(bs_.qrope, T * qw);
    host::attention_prep(g, q, k, v, hc.qn.data(), hc.kn.data(), w_->rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, Q);
    timing_.mid_ms += ms_since(tt);
    og.assign(T * qw, 0.f);
    // attention_npu books its dispatches as part0 and its host stages as part1, the full layers'
    // columns; on this route the GEMM column is the five projections and the attention column
    // (route_ms, below) holds all of the attention, so the stage line compares with the dxB loop's
    const double p0 = timing_.part0_ms, p1 = timing_.part1_ms;
    attention_npu(l, g, Q, nullptr, st.map<uint16_t*>(), row / 2, og.data());
    timing_.part0_ms = p0;
    timing_.part1_ms = p1;
    tt = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, t_real * row, pos0 * row);
    timing_.state_ms += ms_since(tt);
    timing_.attn_ms += timing_.mid_ms - mid0;
    timing_.route_ms += ms_since(t0);       // the dense route's attention column, as the dxB loop's
}

void Core::step_gemm_block_layer(int l, std::vector<double>& xres, size_t T, size_t t_real) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t hid = w_->hidden, qw = gb.qw, kvw = gb.kvw, ff = gb.ff;
    const size_t n_qkv3 = qw + 2 * kvw;
    if (t_real == 0) t_real = T;

    // One GEMM dispatch: tile `x` [T,K] -> upload -> run -> download `y` [N,T].
    // Timing: part0_ms sums ALL 5 GEMM dispatches
    // (qkv3, o, gate, up, down); route_ms (otherwise unused by a dense/Granite
    // layer type -- no MoE routing here) is repurposed for the T attention
    // (dxB) dispatches below, so the two are cleanly separable instead of
    // both landing in part1_ms.
    // Tile x [T, K] (padded to the compiled gb.t columns), run step idx, and pull its output
    // [N, Tk] back into the step's y buffer; returns Tk, the columns the kernel wrote.
    auto gemm_dispatch = [&](size_t idx, const std::vector<float>& x, size_t K, size_t N) -> size_t {
        const Step& s = gb.program[idx];
        auto tt = std::chrono::steady_clock::now();
        // Stage 2.6 Gate B: the GEMM's instruction stream is compiled for gb.t
        // tokens (256 in every set this repo emits -- gemm_q4_prefill.py: "T is
        // still compiled in"), so a T < gb.t call (the host decode route's T=1)
        // runs the SAME dispatch with the real tokens in columns [0, T) and
        // zeros in the padding columns -- the padding invariance the block
        // route already relies on for short tails (hardware-proven: 2.1/2.2's
        // C gates ran 22 real columns against 234 padding ones and matched the
        // sequential route). The output side reads back the kernel's whole
        // [N, gb.t] and keeps the T real columns.
        if (T > gb.t)
            throw std::runtime_error("open_qwen36: gemm route: T=" + std::to_string(T) +
                                     " exceeds the kernel set's compiled " + std::to_string(gb.t));
        xrt::bo& xb = buffer(s.args[1], 0);
        if (T == gb.t) {
            host::tile_x(x.data(), T, K, xb.map<uint16_t*>());      // straight into the mapped buffer
        } else {
            std::vector<float> xp(gb.t * K, 0.f);
            for (size_t r = 0; r < T; ++r)
                for (size_t c = 0; c < K; ++c) xp[r * K + c] = x[r * K + c];
            host::tile_x(xp.data(), gb.t, K, xb.map<uint16_t*>());
        }
        xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, gb.t * K * 2, 0);
        timing_.gemm_tile_ms += ms_since(tt);
        Kern& k = kerns_.at(s.kernel);
        timing_.part0_ms += run(k, s.args, l);
        tt = std::chrono::steady_clock::now();
        read_back(buffer(s.args[2], 0), N * gb.t * 4, 0);
        timing_.gemm_tr_ms += ms_since(tt);
        return static_cast<size_t>(gb.t);
    };
    // The output as the kernel wrote it, [N, T]: the fused input projection, whose three parts
    // the attention paths split by row range.
    auto run_gemm = [&](size_t idx, const std::vector<float>& x, size_t K, size_t N, std::vector<float>& y_out) {
        const size_t Tk = gemm_dispatch(idx, x, K, N);
        auto tt = std::chrono::steady_clock::now();
        const float* y = buffer(gb.program[idx].args[2], 0).map<float*>();
        y_out.resize(N * T);
        if (Tk == T) {
            std::memcpy(y_out.data(), y, N * T * 4);
        } else {
            for (size_t n = 0; n < N; ++n)
                for (size_t t = 0; t < T; ++t) y_out[n * T + t] = y[n * Tk + t];
        }
        timing_.gemm_tr_ms += ms_since(tt);
    };
    // The output as token rows, [T, N] (rows past T are padding): everything after the attention
    // reads a token's row at a time, which over [N, T] is a T-float stride per element.
    auto run_gemm_rows = [&](size_t idx, const std::vector<float>& x, size_t K, size_t N, std::vector<float>& out) {
        const size_t Tk = gemm_dispatch(idx, x, K, N);
        auto tt = std::chrono::steady_clock::now();
        host::transpose(buffer(gb.program[idx].args[2], 0).map<float*>(), N, Tk, BlockScratch::fit(out, Tk * N));
        timing_.gemm_tr_ms += ms_since(tt);
    };

    // ---- entry RMSNorm, GEMM A' (qkv3, real q|k|v pool weight, ONE dispatch) ----
    std::vector<float> xnorm;
    {
        // Stage 2.5 Gate A: the entry norm was the one stage without its own clock
        // (it lived in the residue). Measurement only, code untouched.
        auto tn = std::chrono::steady_clock::now();
        rmsnorm_host(xres, T, hid, ln_w_bf16_[l], gb.eps, xnorm, gb.norm_groups);
        timing_.prenorm_ms += ms_since(tn);
    }
    std::vector<float> y_qkv3;  // [n_qkv3, T] row-major f32
    run_gemm(0, xnorm, hid, n_qkv3, y_qkv3);

    // ---- Stage 2.5 Gate B: the attention half either on the host (opt-in; the
    // Stage 2.6 decode route forces the same branch via in_host_decode_) or
    // as the T single-token dxB dispatches. The NPU path below is untouched.
    std::vector<float> og(T * qw, 0.f);
    if (host_attn_on_ || in_host_decode_) {
        host_attn_layer(l, y_qkv3, T, og);
    } else if (attn_block_on_ && gb.attn_block.present() && !gb.attn_block.prep.empty()) {
        // the two NPU products per kv head (OPEN-PREFILL-ATTN), for a family whose set declares
        // the host half they need; OFLM_OPEN_ATTN_BLOCK=0 keeps the dxB dispatches below
        dense_attention_block(l, y_qkv3, T, t_real, og);
    } else {
    // ---- T single-token dxB dispatches, position-patched, through a GLOBAL
    // T-wide "gact" scratch buffer, shuttled one token at a time via
    // shuttle_buf() -- proven on hardware before this was wired in. ----
    xrt::bo& gact = buffer("gact", 0);
    xrt::bo& act1 = buffer("act", l);
    const size_t AD = lt.act_bytes;
    {
        // Fill gact's Q/K/V region for every token from y_qkv3's columns
        // (f32 bytes, matching the T=1 "act" buffer's own AD_Q/AD_KVN format
        // -- the SAME format Core::step() writes there today).
        auto tq = std::chrono::steady_clock::now();
        uint8_t* gbase = gact.map<uint8_t*>();
        std::memset(gbase, 0, T * AD);
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t) {
            const size_t tk = static_cast<size_t>(t);
            uint8_t* base = gbase + tk * AD;
            float* qd = reinterpret_cast<float*>(base + gb.ad_q);
            float* kd = reinterpret_cast<float*>(base + gb.ad_kvn);
            float* vd = reinterpret_cast<float*>(base + gb.ad_kvn + kvw * 4);
            for (size_t c = 0; c < qw; ++c) qd[c] = y_qkv3[c * T + tk];
            for (size_t c = 0; c < kvw; ++c) kd[c] = y_qkv3[(qw + c) * T + tk];
            for (size_t c = 0; c < kvw; ++c) vd[c] = y_qkv3[(qw + kvw + c) * T + tk];
        }
        gact.sync(XCL_BO_SYNC_BO_TO_DEVICE, T * AD, 0);
        timing_.qkv_scatter_ms += ms_since(tq);   // Stage 2.5: split out of gemm_tr_ms
    }
    // The attention dispatch reads q / k / v and writes og, so only those two ranges of a
    // token's slice move between the wide scratch and the layer's own act. q, k and v are
    // laid out consecutively (ad_q -> ad_kvn -> ad_og), so the read side is one range.
    const size_t qkv_off = gb.ad_q, qkv_bytes = gb.ad_og - gb.ad_q, og_bytes = qw * 2;
    {
        Kern& dxb = kerns_.at(gb.attn_kernel);
        const std::vector<std::string>& attn_args = gb.attn_args;
        for (size_t tk = 0; tk < T; ++tk) {
            const uint64_t pos = static_cast<uint64_t>(pos_) + tk;
            // Stage 2.5 Gate A: the four phases of one dxB dispatch, each with its
            // own clock, so the per-window regression comes out of a single run.
            double p_ms, prep_ms, sub_ms = 0, wait_ms = 0, r_ms;
            auto tp = std::chrono::steady_clock::now();
            stream_patch::attn_apply(dxb.iw(), dxb.attn, pos, dxb.geom);
            dxb.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            p_ms = ms_since(tp);
            timing_.moe_patch_ms += p_ms;
            auto ts = std::chrono::steady_clock::now();
            shuttle_buf(gact, act1, tk, AD, /*wide_to_scratch=*/true, qkv_off, qkv_bytes);
            prep_ms = ms_since(ts);
            timing_.moe_prep_ms += prep_ms;
            // run_split instead of run -- the SAME code path (run() is run_split's
            // halves summed), but submit and device-wait land in the phase log
            // separately. timing_.route_ms semantics unchanged (submit + wait).
            std::tie(sub_ms, wait_ms) = run_split(dxb, attn_args, l);
            timing_.route_ms += sub_ms + wait_ms;
            if (dispatch_log_) dispatch_stats_[dxb.name].add(sub_ms + wait_ms);  // run() did this; run_split does not
            ts = std::chrono::steady_clock::now();
            shuttle_buf(gact, act1, tk, AD, /*wide_to_scratch=*/false, gb.ad_og, og_bytes);
            r_ms = ms_since(ts);
            timing_.moe_read_ms += r_ms;
            if (dxb_log_)
                std::fprintf(dxb_log_, "dxb,%d,%llu,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                             l, static_cast<unsigned long long>(pos), p_ms, prep_ms, sub_ms, wait_ms, r_ms);
        }
    }
    // ---- read back AD_OG (bf16, qw elements/token) as [T,qw] f32 -----------
    {
        auto to = std::chrono::steady_clock::now();
        const uint8_t* base = gact.map<uint8_t*>();   // the og ranges are already host-side
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t) {
            const size_t tk = static_cast<size_t>(t);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(base + tk * AD + gb.ad_og);
            for (size_t c = 0; c < qw; ++c) og[tk * qw + c] = bf16_to_f32(src[c]);
        }
        timing_.mid_ms += ms_since(to);
    }
    }  // end of the NPU dxB path (the host branch filled og above)

    // ---- GEMM O (o_proj, real weight, reusing the "qkv"-shaped context) ---
    // From here on every intermediate is token rows in scratch kept across layers and blocks:
    // the values are what the [N, T] form gave, read contiguously and never re-zeroed.
    std::vector<float>& y_o = bs_.d_o;       // [T, hid]
    run_gemm_rows(1, og, qw, hid, y_o);

    // ---- host: residual add, post-attention RMSNorm ------------------------
    // Plain: res1 = xres + y_o; xm = post_attn_norm(res1). Sandwich (Gemma 3): the norm sits
    // on the attention OUTPUT before it joins the residual, and a SEPARATE weight
    // (pre_feedforward_layernorm) norms the resulting residual to produce xm --
    // post_attention_layernorm and pre_feedforward_layernorm are different tensors here,
    // where the plain chain has only the one.
    auto th = std::chrono::steady_clock::now();
    std::vector<double>& res1 = bs_.d_res;   // [T, hid]
    BlockScratch::fit(res1, T * hid);
    std::vector<float> xm;
    if (gb.sandwich) {
        std::vector<double>& y_o_row = bs_.d_row;
        BlockScratch::fit(y_o_row, T * hid);
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t)
            for (size_t c = 0; c < hid; ++c)
                y_o_row[static_cast<size_t>(t) * hid + c] = static_cast<double>(y_o[static_cast<size_t>(t) * hid + c]);
        std::vector<float> t_attn;
        rmsnorm_host(y_o_row, T, hid, post_ln_w_bf16_[l], gb.eps, t_attn, gb.norm_groups);
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t)
            for (size_t c = 0; c < hid; ++c)
                res1[static_cast<size_t>(t) * hid + c] =
                    xres[static_cast<size_t>(t) * hid + c] + static_cast<double>(t_attn[static_cast<size_t>(t) * hid + c]);
        rmsnorm_host(res1, T, hid, pre_ffn_w_[l], gb.eps, xm, gb.norm_groups);
    } else {
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t)
            for (size_t c = 0; c < hid; ++c)
                res1[static_cast<size_t>(t) * hid + c] =
                    xres[static_cast<size_t>(t) * hid + c] + static_cast<double>(y_o[static_cast<size_t>(t) * hid + c]);
        rmsnorm_host(res1, T, hid, post_ln_w_bf16_[l], gb.eps, xm, gb.norm_groups);
    }
    timing_.tail_ms += ms_since(th);

    // ---- GEMM gate_proj + up_proj (SAME context, zero switch between them) -
    std::vector<float>& y_gate = bs_.d_gate;  // both [T, ff]
    std::vector<float>& y_up = bs_.d_up;
    run_gemm_rows(2, xm, hid, ff, y_gate);
    run_gemm_rows(3, xm, hid, ff, y_up);

    // ---- host gated FFN: silu(gate) * up, or Gemma 3's gelu_tanh(gate) * up ------------
    const bool gelu_tanh = gb.act == "gelu_tanh";
    th = std::chrono::steady_clock::now();
    std::vector<float>& h = bs_.d_h;          // [T, ff], the down projection's x
    h.resize(T * ff);                          // exactly T rows: run_gemm pads a short one itself
#pragma omp parallel for
    for (long long t = 0; t < static_cast<long long>(T); ++t) {
        const size_t tk = static_cast<size_t>(t);
        for (size_t c = 0; c < ff; ++c) {
            const double g = static_cast<double>(y_gate[tk * ff + c]);
            const double u = static_cast<double>(y_up[tk * ff + c]);
            // tanh-approximate GELU (HF's "gelu_pytorch_tanh"): sqrt(2/pi) = 0.7978845608028654
            const double act = gelu_tanh ? 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
                                         : (g / (1.0 + std::exp(-g)));
            h[tk * ff + c] = static_cast<float>(act * u);
        }
    }
    timing_.tail_ms += ms_since(th);

    // ---- GEMM down_proj, then residual -> next layer's xres -----------------
    // Plain: xres = res1 + y_down. Sandwich: y_down is normed (post_feedforward_layernorm)
    // before it joins the residual, mirroring the attention side above.
    std::vector<float>& y_down = bs_.d_down;  // [T, hid]
    run_gemm_rows(4, h, ff, hid, y_down);
    th = std::chrono::steady_clock::now();
    if (gb.sandwich) {
        std::vector<double>& y_down_row = bs_.d_row;
        BlockScratch::fit(y_down_row, T * hid);
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t)
            for (size_t c = 0; c < hid; ++c)
                y_down_row[static_cast<size_t>(t) * hid + c] = static_cast<double>(y_down[static_cast<size_t>(t) * hid + c]);
        std::vector<float> t_ffn;
        rmsnorm_host(y_down_row, T, hid, post_ffn_w_[l], gb.eps, t_ffn, gb.norm_groups);
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t)
            for (size_t c = 0; c < hid; ++c)
                xres[static_cast<size_t>(t) * hid + c] =
                    res1[static_cast<size_t>(t) * hid + c] + static_cast<double>(t_ffn[static_cast<size_t>(t) * hid + c]);
    } else {
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(T); ++t)
            for (size_t c = 0; c < hid; ++c)
                xres[static_cast<size_t>(t) * hid + c] =
                    res1[static_cast<size_t>(t) * hid + c] + static_cast<double>(y_down[static_cast<size_t>(t) * hid + c]);
    }
    timing_.tail_ms += ms_since(th);
}

void Core::step_gemm_block(const std::vector<int>& ids, size_t t_real, bool want_logits) {
    apply_thread_budget();
    if (!weights_loaded_) throw std::runtime_error("open_qwen36: step_gemm_block before load_weights");
    const size_t T = ids.size();
    if (T == 0) return;
    if (gemm_block_t_ == 0 || T != gemm_block_t_)
        throw std::runtime_error("open_qwen36: step_gemm_block called with " + std::to_string(T) +
                                 " tokens, but this kernel set's gemm-route block size is " +
                                 std::to_string(gemm_block_t_) + " (0 = no gemm_block program loaded)");
    if (t_real == 0 || t_real > T) throw std::runtime_error("open_qwen36: step_gemm_block: t_real must be in (0, T]");
    // Positions [pos_, pos_+T) are all touched (padding columns included --
    // hardware-proven exact: the real columns' output does not depend on
    // what the padding columns carry), so the CAPACITY check must cover T,
    // not just the real tokens -- even though only t_real of them advance
    // pos_ afterward.
    if (static_cast<size_t>(pos_) + T > cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: gemm-block [" + std::to_string(pos_) + ", " +
                                 std::to_string(pos_ + T) + ") would exceed the context capacity " +
                                 std::to_string(cfg_.max_ctx));
    for (int tok : ids)
        if (tok < 0 || static_cast<size_t>(tok) >= w_->vocab) throw std::runtime_error("open_qwen36: token id out of range");
    if (types_[0]->gemm_block.kind != "dense") {
        step_block_moe(ids, t_real, want_logits);
        return;
    }

    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};

    // ---- embed all T tokens (padding included) into a HOST-resident fp64
    // xres[T,hidden] -- this route's running residual stream lives on the
    // HOST between GEMM dispatches (RMSNorm/residual/SwiGLU are host-side,
    // not fused on-core), so there is no device-resident T-wide buffer for
    // it, unlike the per-layer weight/activation buffers below. ------------
    std::vector<double> xres(T * w_->hidden);
    {
        std::vector<float> row(w_->hidden);
        for (size_t tk = 0; tk < T; ++tk) {
            embedding_row(static_cast<size_t>(ids[tk]), row.data());
            for (size_t c = 0; c < w_->hidden; ++c) xres[tk * w_->hidden + c] = static_cast<double>(row[c]);
        }
    }

    for (int l = 0; l < nl_; ++l) {
        if (types_[l]->gemm_block.t == 0)
            throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + " (" + types_[l]->name + ") has no gemm_block program");
        step_gemm_block_layer(l, xres, T, t_real);
    }
    pos_ += static_cast<int>(t_real);

    block_logits_.clear();
    if (block_logits_all_) {
        std::vector<float> row(w_->hidden);
        for (size_t t = 0; t < t_real; ++t) {
            for (size_t c = 0; c < w_->hidden; ++c) row[c] = static_cast<float>(xres[t * w_->hidden + c]);
            tail_logits(row.data());
            block_logits_.push_back(logits_host_);
        }
    }
    if (want_logits) {
        // only the last REAL token's logits, as step() does for a prefill
        std::vector<float> last_row(w_->hidden);
        for (size_t c = 0; c < w_->hidden; ++c) last_row[c] = static_cast<float>(xres[(t_real - 1) * w_->hidden + c]);
        tail_logits(last_row.data());
    }
    timing_.total_ms = ms_since(t0);
}

// ---- the MoE families' block: kinds linear and full. Timing: part0 = the GEMM
// dispatches, part1 = the host stages, route = the per-token MoE (routing + kernel).

void Core::moe_stage_resize(size_t rows) {
    const size_t hid = w_->hidden, E = w_->moe.experts, topk = w_->moe.topk;
    if (moe_.res.size() >= rows * hid) return;
    moe_.res.resize(rows * hid);
    moe_.xm.resize(rows * hid);
    moe_.probs.resize(rows * E);
    moe_.idx.resize(rows * topk);
    moe_.w.resize(rows * topk);
}

void Core::block_layer_moe(int l, size_t rows, size_t t_real, float* xres) {
    const GemmBlockProgram& gb = types_[l]->gemm_block;
    if (gb.has_dense_ffn()) return;                 // the layer's own FFN already wrote xres
    const size_t hid = w_->hidden, E = w_->moe.experts, topk = w_->moe.topk;
    if (moe_batch_on_ && gb.moe_batch.present()) {
        moe_block(l, moe_.xm.data(), moe_.res.data(), moe_.idx.data(), moe_.w.data(), rows, t_real, xres);
        return;
    }
    for (size_t t = 0; t < rows; ++t) {
        if (t < t_real)
            moe_token(l, moe_.xm.data() + t * hid, moe_.res.data() + t * hid, moe_.probs.data() + t * E,
                      moe_.idx.data() + t * topk, moe_.w.data() + t * topk, xres + t * hid);
        else
            std::memcpy(xres + t * hid, moe_.res.data() + t * hid, hid * 4);   // padding: carried, never read
    }
}

void Core::step_block_moe(const std::vector<int>& ids, size_t t_real, bool want_logits) {
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};
    const size_t T = ids.size(), hid = w_->hidden;
    std::vector<float> xres(T * hid);
    for (size_t t = 0; t < T; ++t) embedding_row(static_cast<size_t>(ids[t]), xres.data() + t * hid);
    moe_stage_resize(T);
    for (int l = 0; l < nl_; ++l) {
        const std::string& kind = types_[l]->gemm_block.kind;
        if (kind == "linear") block_layer_linear(l, xres.data(), T, t_real, 0, true, true);
        else if (kind == "full") block_layer_full(l, xres.data(), T, t_real, static_cast<size_t>(pos_), 0, true);
        else throw std::runtime_error("open_qwen36: layer " + std::to_string(l) + " (" + types_[l]->name + ") has no block route");
        block_layer_moe(l, T, t_real, xres.data());
    }
    pos_ += static_cast<int>(t_real);
    block_logits_.clear();
    if (block_logits_all_)
        for (size_t t = 0; t < t_real; ++t) {
            tail_logits(xres.data() + t * hid);
            block_logits_.push_back(logits_host_);
        }
    if (want_logits) tail_logits(xres.data() + (t_real - 1) * hid);
    timing_.total_ms = ms_since(t0);
}

// ---- B(2) of .claude/plans/prefill-parity.md: the same layers in the other order.
//
// The block-major loop above runs one block of T tokens through all 40 layers, so a
// layer's MoE sees only that block's ~8 tokens per expert and streams each expert's
// 1.97 MB for them alone -- 25.9 GB a block, which at the array's ceiling is most of
// what a block costs. Closed prefills the whole prompt in ONE weight pass; forced to
// our batch size it runs at 59 tok/s against our 86, so the per-pass kernel work was
// never the gap. This is the schedule that closes it: all blocks through layer l, then
// layer l's MoE once over every token of the prompt, then layer l + 1.
//
// Nothing about a layer's arithmetic changes. Each layer's device state -- the KV rows,
// the DeltaNet S and conv rows -- is still written block by block in position order,
// and layer l still reads the residual layer l - 1 produced for the same block, because
// layer l - 1 finished every block before layer l started. The MoE's own accumulation
// order per token is ascending expert index either way (the visit list is sorted by
// expert), and a token's expert output does not depend on which other tokens share its
// slot, so the result is bit-for-bit the block-major one.
void Core::step_gemm_prompt(const std::vector<int>& ids, bool want_logits) {
    apply_thread_budget();
    if (pipe_ok()) {
        step_prompt_pipe(ids, want_logits);
        return;
    }
    if (!segs_.empty()) throw std::runtime_error("open_qwen36: packed prompts need the prompt pipeline");
    if (!layer_major_ok()) throw std::runtime_error("open_qwen36: step_gemm_prompt on a kernel set without a MoE block route");
    if (ids.empty()) throw std::runtime_error("open_qwen36: step_gemm_prompt with no tokens");
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};
    const size_t T = gemm_block_t_, hid = w_->hidden;
    const size_t N = ids.size(), B = (N + T - 1) / T, TOT = B * T;
    if (static_cast<size_t>(pos_) + N > cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: prompt of " + std::to_string(N) + " at position " + std::to_string(pos_) +
                                 " is past the context capacity of " + std::to_string(cfg_.max_ctx));
    const size_t pos0 = static_cast<size_t>(pos_);
    // The padded rows carry the last real id, exactly as the block-major caller padded
    // the tail block; they never touch the state and never reach the position.
    auto tsetup = std::chrono::steady_clock::now();
    std::vector<float> xres(TOT * hid);
    for (size_t t = 0; t < TOT; ++t)
        file_->bf16_row(w_->embed_tensor, static_cast<size_t>(ids[std::min(t, N - 1)]), hid, xres.data() + t * hid);
    moe_stage_resize(TOT);
    timing_.setup_ms += ms_since(tsetup);
    timing_.part1_ms += ms_since(tsetup);
    for (int l = 0; l < nl_; ++l) {
        const std::string& kind = types_[l]->gemm_block.kind;
        if (stage_major_on_) {
            if (kind == "linear") layer_linear_staged(l, xres.data(), B, T, N);
            else layer_full_staged(l, xres.data(), B, T, N, pos0);
            block_layer_moe(l, TOT, N, xres.data());
            continue;
        }
        for (size_t b = 0; b < B; ++b) {
            const size_t row0 = b * T, t_real = std::min(T, N - row0);
            if (kind == "linear") block_layer_linear(l, xres.data() + row0 * hid, T, t_real, row0, b == 0, b + 1 == B);
            else block_layer_full(l, xres.data() + row0 * hid, T, t_real, pos0 + row0, row0, b == 0);
        }
        block_layer_moe(l, TOT, N, xres.data());
    }
    pos_ += static_cast<int>(N);
    block_logits_.clear();
    if (block_logits_all_)
        for (size_t t = 0; t < N; ++t) {
            tail_logits(xres.data() + t * hid);
            block_logits_.push_back(logits_host_);
        }
    if (want_logits) tail_logits(xres.data() + (N - 1) * hid);
    timing_.total_ms = ms_since(t0);
}

// ============================================================================
// The Qwen3.5 dense prompt as a pipeline (dit GEMMs). Cell (l, c) = layer l over prompt
// chunk c (M = gemm_dit.m tokens). Its stages, alternating host (H) and NPU (N):
//   0 H  pre-norm -> bf16 A                       (needs (l - 1, c) finished)
//   1 N  qkv (+z / +gate) GEMM
//   2 H  DeltaNet or attention for the chunk      (needs (l, c - 1) past stage 2: S / K V chain)
//        -> og bf16
//   3 N  out GEMM
//   4 H  residual + post-norm -> bf16 A
//   5 N  FFN: up|gate with the SwiGLU epilogue, then down reading it gathered
//   6 H  residual
// An NPU thread takes the lowest ready (l, c) N stage, this thread the lowest ready H stage
// (its OpenMP team does the work), so the NPU computes later chunks and the next layer while
// the host runs a chunk's DeltaNet. Each chunk has its own device buffers; H (the FFN's
// gathered middle) is the NPU thread's alone.
// ============================================================================
bool Core::pipe_ok() const {
    if (!dit_on_ || !pipe_on_ || !man_.gemm_dit.contains("ffn") || !dit_ffn_on() || !gemm_block_t_) return false;
    for (int l = 0; l < nl_; ++l) {
        const GemmBlockProgram& gb = types_[l]->gemm_block;
        if ((gb.kind != "linear" && gb.kind != "full") || !gb.has_dense_ffn() || gb.program.size() < 2 ||
            gb.ffn_program.size() < 2)
            return false;
    }
    return true;
}

void Core::step_prompt_pipe(const std::vector<int>& ids, bool want_logits) {
    auto t0 = std::chrono::steady_clock::now();
    timing_ = StepTiming{};
    // gemm_dit.small: a prompt of up to OFLM_OPEN_SMALL_MAX tokens (default 2048) runs in chunks
    // of the small M on its own GEMM streams -- fewer padded rows, and from two chunks on the host
    // and NPU overlap. Its attention is the host's (the NPU streams take full chunks), which is
    // what sets the limit: d3 int4 at 1024 / 1536 / 2048 tokens 2.43 / 3.57 / 3.81 s -> 2.12 /
    // 2.72 / 3.37, but at 2560 4.61 -> 4.83 and at 4096 5.77 -> 8.70.
    // OFLM_OPEN_SMALL_M=0 off. Its FFN is the device-packed one only.
    static const bool small_on = [] { const char* e = std::getenv("OFLM_OPEN_SMALL_M"); return !(e && *e == '0'); }();
    static const size_t small_max = [] { const char* e = std::getenv("OFLM_OPEN_SMALL_MAX"); return e ? std::strtoul(e, nullptr, 10) : 2048ul; }();
    // gemm_dit.tiny: the same at its M on 64-row core tiles (its own image) up to OFLM_OPEN_TINY_MAX
    // tokens (default 256: one chunk, the 512-row result bit for bit, d3 median 0.96 -> 0.69 s;
    // two chunks for 257-512 only -4 %); OFLM_OPEN_TINY_MAX=0 off
    static const size_t tiny_max = [] { const char* e = std::getenv("OFLM_OPEN_TINY_MAX"); return e ? std::strtoul(e, nullptr, 10) : 256ul; }();
    static const bool small_ffn = [] { const char* e = std::getenv("OFLM_OPEN_FFN_DEVICE"); return !(e && *e == '0'); }();
    const nlohmann::json* Dp = &man_.gemm_dit;
    nlohmann::json Dsmall;
    const size_t want = static_cast<size_t>(pos_) + ids.size();
    const char* tier = Dp->contains("tiny") && want <= tiny_max ? "tiny" : Dp->contains("small") && want <= small_max ? "small" : nullptr;
    if (small_on && small_ffn && dit_hs_ && tier) {
        const nlohmann::json& S = (*Dp)[tier];
        Dsmall = *Dp;
        Dsmall["m"] = S["m"];
        Dsmall["mt"] = S.value("mt", 128);
        Dsmall["kernels2"] = S["kernels2"];
        for (const auto& [k, v] : S["ffn"].items()) Dsmall["ffn"][k] = v;
        Dp = &Dsmall;
    }
    const nlohmann::json& D = *Dp;
    const size_t M = D["m"].get<size_t>(), MT = D.value("mt", size_t{128}), hid = man_.hidden, N = ids.size();
    const size_t C = (N + M - 1) / M, TOT = C * M, L = static_cast<size_t>(nl_);
    if (static_cast<size_t>(pos_) + N > cfg_.max_ctx)
        throw std::runtime_error("open_qwen36: prompt of " + std::to_string(N) + " at position " + std::to_string(pos_) +
                                 " is past the context capacity of " + std::to_string(cfg_.max_ctx));
    const size_t pos0 = static_cast<size_t>(pos_);
    if (state_owner_ == 1) states_to_device();
    // packed prompts (set_segments): row flags where one starts (DeltaNet resets there) and
    // each row's position inside its own prompt (rope); attention stays inside each prompt
    std::vector<size_t> segs = std::move(segs_);
    segs_.clear();
    seg_x_.clear();
    const bool packed = segs.size() > 1;
    std::vector<uint8_t> rflag;
    std::vector<size_t> rpos;
    if (!segs.empty()) {
        if (!noncausal_ || segs[0] != 0)
            throw std::runtime_error("open_qwen36: packed prompts need noncausal mode and a first segment at row 0");
        for (size_t i = 1; i < segs.size(); ++i)
            if (segs[i] <= segs[i - 1] || segs[i] >= N) throw std::runtime_error("open_qwen36: packed prompt starts out of order");
    }
    if (packed) {
        rflag.assign(TOT, 0);
        rpos.resize(TOT);
        for (size_t i = 0; i < segs.size(); ++i) {
            const size_t a = segs[i], b = i + 1 < segs.size() ? segs[i + 1] : TOT;
            rflag[a] = 1;
            for (size_t t = a; t < b; ++t) rpos[t] = t - a;
        }
    }
    const size_t ff = D["ffn"]["ff"].get<size_t>();
    size_t wq = 0, wmid = 0;
    for (int l = 0; l < nl_; ++l) {
        const GemmBlockProgram& gb = types_[l]->gemm_block;
        const size_t nq = gb.kind == "linear" ? gb.qkv_dim + gb.vw : 2 * gb.qw + 2 * gb.kvw;
        wq = std::max(wq, nq);
        wmid = std::max(wmid, gb.kind == "linear" ? gb.vw : gb.qw);
    }
    static const nlohmann::json kEmpty = nlohmann::json::object();
    const nlohmann::json& K2 = D.contains("kernels2") ? D["kernels2"] : kEmpty;
    const bool up2 = D["ffn"].contains("up2");
    const bool down2 = up2 && D["ffn"].contains("down2");
    // the FFN as one NPU step: up|gate writes h packed as the down GEMM's A unit slots on the
    // device (dit2 pack_h), down reads them (a_units) -- no HPACK readback / repack on the host
    static const bool ffn_dev_on = [] { const char* e = std::getenv("OFLM_OPEN_FFN_DEVICE"); return !(e && *e == '0'); }();
    const bool ffn_dev = down2 && dit_hs_ && ffn_dev_on && D["ffn"].contains("up2p") && D["ffn"].contains("down2u");
    auto has2 = [&](size_t n, size_t k) { return K2.contains(std::to_string(n) + "x" + std::to_string(k)); };
    // designs/dit_fa256: NPU attention for a full layer's chunk when a stream exists for its key
    // length (pos0 + chunk end) and that length is at least OFLM_OPEN_FA_MIN_LK (default 3072:
    // below it the kernel is no faster than the host); OFLM_OPEN_FA=0 keeps every chunk on the host
    const nlohmann::json& FA = D.contains("fa") ? D["fa"]["lk"] : kEmpty;
    static const bool fa_on = [] { const char* e = std::getenv("OFLM_OPEN_FA"); return !(e && *e == '0'); }();
    static const size_t fa_min = [] { const char* e = std::getenv("OFLM_OPEN_FA_MIN_LK"); return e ? std::strtoul(e, nullptr, 10) : 3072ul; }();
    // noncausal (d3): one NPU stream for the whole prompt, keys [0, N) of lk = N rounded up to
    // the chunk, its valid_len sentinel patched to N; host attention below OFLM_OPEN_FA_NC_MIN
    // tokens or past the streams the set carries
    std::string nc_kernel;
    // designs/dit_fa256b (manifest fa.ncb): the bfp16 stream, Q / K / V^T packed here (fab_pack_*);
    // OFLM_OPEN_FA_NCB=0 keeps dit_fa256's
    static const bool ncb_on = [] { const char* e = std::getenv("OFLM_OPEN_FA_NCB"); return !(e && *e == '0'); }();
    bool nc_b = false;
    size_t lkn = 0;
    const bool full_m = M == man_.gemm_dit["m"].get<size_t>();     // the attention streams' query rows
    if (noncausal_ && !packed && fa_on && full_m && D.contains("fa") && D["fa"].contains("nc")) {
        static const size_t nc_min = [] { const char* e = std::getenv("OFLM_OPEN_FA_NC_MIN"); return e ? std::strtoul(e, nullptr, 10) : 1024ul; }();
        const std::string lk = std::to_string((pos0 + N + M - 1) / M * M);
        lkn = std::stoul(lk);
        nc_b = ncb_on && D["fa"].contains("ncb") && D["fa"]["ncb"].contains(lk);
        if (N >= nc_min && D["fa"][nc_b ? "ncb" : "nc"].contains(lk)) {
            nc_kernel = D["fa"][nc_b ? "ncb" : "nc"][lk].get<std::string>();
            Kern& k = kerns_.at(nc_kernel);
            const uint32_t sent = D["fa"]["nc_valid"].get<uint32_t>();
            size_t hits = 0;
            for (size_t i = 0; i < k.words.size(); ++i)
                if (k.words[i] == sent) { k.iw()[i] = static_cast<uint32_t>(pos0 + N); ++hits; }
            if (!hits) throw std::runtime_error("open_qwen36: " + nc_kernel + " carries no valid_len sentinel");
            k.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (!nc_kernel.empty() && !nc_b) {
            // The stream reads key rows up to lk; rows [N, lk) still hold an earlier, longer
            // prompt's K / V, and those changed the result despite the valid_len mask (1.6% of
            // the residual at 1809 tokens after a 2017-token prompt). Zero them.
            const size_t lkn = std::stoul(lk);
            for (int l = 0; l < nl_; ++l) {
                const LayerType& lt = *types_[l];
                if (lt.state_kind == "linear") continue;
                const size_t row = lt.state_row, a = pos0 + N;
                std::memset(state_[l].map<uint8_t*>() + a * row, 0, (lkn - a) * row);
                state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, (lkn - a) * row, a * row);
            }
        }
    }
    if (nc_kernel.empty()) nc_b = false;
    const bool nc_npu = !nc_kernel.empty();
    // fa.ncb: K / V packed per full-attention layer into a pair of buffers that alternate by
    // full layer (a layer's attention is done before the next-but-one full layer's MID starts)
    std::vector<int> full_ord(nl_, 0);
    size_t fab_nh = 0, fab_kvh = 0;
    for (int l = 0, n = 0; l < nl_; ++l)
        if (types_[l]->gemm_block.kind == "full") {
            full_ord[l] = n++;
            fab_nh = types_[l]->gemm_block.nh;
            fab_kvh = types_[l]->gemm_block.kvh;
        }
    auto fa_kernel = [&](size_t c) -> std::string {
        const size_t lk = pos0 + (c + 1) * M;
        if (!fa_on || !full_m || lk < fa_min || !FA.contains(std::to_string(lk))) return {};
        return FA[std::to_string(lk)].get<std::string>();
    };
    // per chunk device buffers (named globals, so start_run binds them), kept across requests
    auto need = [&](const std::string& name, size_t bytes) {
        auto it = globals_.find(name);
        if (it == globals_.end() || it->second.size() < bytes) globals_[name] = alloc(bytes);
    };
    for (size_t c = 0; c < C; ++c) {
        const std::string cs = std::to_string(c);
        need("pp_a_" + cs, M * hid * 2);
        need("pp_m_" + cs, M * wmid * 2);
        need("pp_q_" + cs, M * wq * 2);
        need("pp_y_" + cs, M * hid * 2);
        if (!K2.empty() || up2) {
            need("pp_a2_" + cs, M * hid * 9 / 8);
            need("pp_m2_" + cs, M * wmid * 9 / 8);
        }
        if (down2) {
            need("pp_h_" + cs, M * 2 * ff * 2);
            need("pp_d2_" + cs, M * ff * 9 / 8);
        }
        if (!FA.empty()) {
            need("fa_q_" + cs, M * wmid * 2);
            need("fa_o_" + cs, M * wmid * 2);
        }
        if (nc_b) need("fab_q_" + cs, fab_nh * (M / 32) * kFabQBlock);
    }
    if (nc_b)
        for (const char* nm : {"fab_k0", "fab_k1", "fab_v0", "fab_v1"})
            need(nm, fab_kvh * (lkn / 64) * kFabChunk);
    need("pp_h", M * 2 * ff * 2);
    if (ffn_dev) need("pp_hu", M / MT * (ff / 64) * 36864);   // h unit slots, one NPU step uses and frees them
    // Each layer's cells walk a list of operations, host (even index) and NPU (odd) alternating.
    enum Op { PRE, QKV, MID, FAT, GATE, OUT, TAIL, UP, HPACK, DOWN, RES };
    std::vector<std::vector<int>> ops(L);
    for (size_t l = 0; l < L; ++l) {
        auto& o = ops[l];
        o = {PRE, QKV, MID};
        if (types_[l]->gemm_block.kind == "full" && (!FA.empty() || noncausal_)) o.insert(o.end(), {FAT, GATE});
        o.insert(o.end(), {OUT, TAIL, UP});
        if (down2 && !ffn_dev) o.insert(o.end(), {HPACK, DOWN});
        o.push_back(RES);
    }
    // host rows
    std::vector<float> xres(TOT * hid), res(TOT * hid), xn(TOT * hid);
    for (size_t t = 0; t < TOT; ++t)
        file_->bf16_row(w_->embed_tensor, static_cast<size_t>(ids[std::min(t, N - 1)]), hid, xres.data() + t * hid);
    timing_.setup_ms += ms_since(t0);

    auto kname = [&](size_t n, size_t k) {
        if (dit_hs_) throw std::runtime_error("open_qwen36: dit GEMM " + std::to_string(n) + "x" + std::to_string(k) +
                                              " has no dit2 kernel, and the weights are in the hs (2x2) order");
        return D["kernels"][std::to_string(n) + "x" + std::to_string(k)].get<std::string>();
    };
    std::vector<int> stage(L * C, 0), busy(L * C, 0);
    std::mutex mu;
    std::condition_variable cv;
    size_t finished = 0;
    std::exception_ptr err;
    auto cell = [&](size_t l, size_t c) { return l * C + c; };
    auto nst = [&](size_t l) { return static_cast<int>(ops[l].size()); };
    auto ready = [&](size_t l, size_t c, bool npu) {
        const int s = stage[cell(l, c)];
        if (busy[cell(l, c)] || s >= nst(l) || ((s & 1) != 0) != npu) return false;
        if (s == 0) return l == 0 || stage[cell(l - 1, c)] >= nst(l - 1);
        if (ops[l][s] == MID) return c == 0 || stage[cell(l, c - 1)] > s;     // S / the KV rows chain
        if (noncausal_ && ops[l][s] == FAT)                                     // every chunk's K / V rows first
            for (size_t c2 = 0; c2 < C; ++c2)
                if (stage[cell(l, c2)] < s) return false;
        return true;
    };
    // the NPU takes the lowest ready cell, preferring one in the hardware context it is in
    // (attention vs GEMM: a context switch costs a few ms each way)
    int last_fa = -1;
    auto pick = [&](bool npu, size_t& ol, size_t& oc) {
        bool found = false;
        for (size_t l = 0; l < L; ++l)
            for (size_t c = 0; c < C; ++c)
                if (ready(l, c, npu)) {
                    if (!npu || last_fa < 0) { ol = l; oc = c; return true; }
                    const bool f = ops[l][stage[cell(l, c)]] == FAT;
                    if (f == (last_fa == 1)) { ol = l; oc = c; return true; }
                    if (!found) { ol = l; oc = c; found = true; }
                }
        return found;
    };
    double npu_busy_ms = 0, npu_idle_ms = 0, host_idle_ms = 0, fa_ms = 0;
    double hst[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int n_fa = 0;

    // ---- the NPU thread
    std::thread npu([&] {
        try {
            for (;;) {
                size_t l = 0, c = 0;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    auto ti = std::chrono::steady_clock::now();
                    cv.wait(lk, [&] { return err || finished == L * C || pick(true, l, c); });
                    npu_idle_ms += ms_since(ti);
                    if (err || finished == L * C) return;
                    busy[cell(l, c)] = 1;
                }
                const int op = ops[l][stage[cell(l, c)]];
                const GemmBlockProgram& gb = types_[l]->gemm_block;
                const std::string cs = std::to_string(c);
                auto tb = std::chrono::steady_clock::now();
                auto run = [&](const std::string& k, const std::vector<std::string>& args) {
                    Inflight f = start_run(kerns_.at(k), args, static_cast<int>(l));
                    wait_run(f);
                };
                const size_t nq = gb.kind == "linear" ? gb.qkv_dim + gb.vw : 2 * gb.qw + 2 * gb.kvw;
                const size_t km = gb.kind == "linear" ? gb.vw : gb.qw;
                if (op == QKV) {
                    if (has2(nq, hid)) run(K2[std::to_string(nq) + "x" + std::to_string(hid)].get<std::string>(),
                                           {"pp_a2_" + cs, gb.program[0].args[0], "pp_q_" + cs});
                    else run(kname(nq, hid), {"pp_a_" + cs, gb.program[0].args[0], "pp_q_" + cs});
                } else if (op == FAT && noncausal_ && nc_npu) {
                    auto tf = std::chrono::steady_clock::now();
                    if (nc_b) {
                        const std::string par = std::to_string(full_ord[l] % 2);
                        run(nc_kernel, {"fab_q_" + cs, "fab_k" + par, "fab_v" + par, "fa_o_" + cs});
                    } else {
                        run(nc_kernel, {"fa_q_" + cs, "state", "state", "fa_o_" + cs});
                    }
                    fa_ms += ms_since(tf);
                    ++n_fa;
                } else if (op == FAT && noncausal_) {
                    // noncausal on the host: the barrier only; GATE runs the attention
                } else if (op == FAT) {
                    auto tf = std::chrono::steady_clock::now();
                    run(fa_kernel(c), {"fa_q_" + cs, "state", "state", "fa_o_" + cs});
                    fa_ms += ms_since(tf);
                    ++n_fa;
                } else if (op == OUT) {
                    if (has2(hid, km)) run(K2[std::to_string(hid) + "x" + std::to_string(km)].get<std::string>(),
                                           {"pp_m2_" + cs, gb.program[1].args[0], "pp_y_" + cs});
                    else run(kname(hid, km), {"pp_m_" + cs, gb.program[1].args[0], "pp_y_" + cs});
                } else if (op == UP && ffn_dev) {
                    run(D["ffn"]["up2p"].get<std::string>(), {"pp_a2_" + cs, gb.ffn_program[0].args[0], "pp_hu"});
                    run(D["ffn"]["down2u"].get<std::string>(), {"pp_hu", gb.ffn_program[1].args[0], "pp_y_" + cs});
                    static const bool ffn_check = [] { const char* e = std::getenv("OFLM_OPEN_FFN_CHECK"); return e && *e == '1'; }();
                    if (ffn_check && c == 0 && l < 2) {
                        // debug: the host-repack route on the same input, compared
                        xrt::bo& yb = buffer("pp_y_" + cs, 0);
                        read_back(yb, M * hid * 2, 0);
                        std::vector<float> ydev(M * hid);
                        for (size_t i = 0; i < M * hid; ++i) ydev[i] = bf16_to_f32(yb.map<const uint16_t*>()[i]);
                        xrt::bo& hu = buffer("pp_hu", 0);
                        read_back(hu, 36864, 0);
                        run(D["ffn"]["up2"].get<std::string>(), {"pp_a2_" + cs, gb.ffn_program[0].args[0], "pp_h_" + cs});
                        xrt::bo& hb = buffer("pp_h_" + cs, 0);
                        read_back(hb, M * 2 * ff * 2, 0);
                        std::vector<float> hf2(M * ff);
                        for (size_t r = 0; r < M; ++r)
                            for (size_t j = 0; j < ff; ++j) hf2[r * ff + j] = bf16_to_f32(hb.map<const uint16_t*>()[r * 2 * ff + (j / 64) * 128 + j % 64]);
                        xrt::bo& db = buffer("pp_d2_" + cs, 0);
                        dit2_pack_a(hf2.data(), M, M, ff, db.map<uint8_t*>(), dit_hs_);
                        db.sync(XCL_BO_SYNC_BO_TO_DEVICE, M * ff * 9 / 8, 0);
                        int same = 0, nz = 0;
                        for (int i = 0; i < 9216; ++i) {
                            same += hu.map<const uint8_t*>()[i] == db.map<const uint8_t*>()[i];
                            nz += hu.map<const uint8_t*>()[i] != 0;
                        }
                        std::fprintf(stderr, "ffn_check: slot 0 nonzero %d/9216; dev %02x %02x %02x %02x %02x %02x %02x %02x %02x | host %02x %02x %02x %02x %02x %02x %02x %02x %02x\n", nz,
                                     hu.map<const uint8_t*>()[0], hu.map<const uint8_t*>()[1], hu.map<const uint8_t*>()[2], hu.map<const uint8_t*>()[3],
                                     hu.map<const uint8_t*>()[4], hu.map<const uint8_t*>()[5], hu.map<const uint8_t*>()[6], hu.map<const uint8_t*>()[7], hu.map<const uint8_t*>()[8],
                                     db.map<const uint8_t*>()[0], db.map<const uint8_t*>()[1], db.map<const uint8_t*>()[2], db.map<const uint8_t*>()[3],
                                     db.map<const uint8_t*>()[4], db.map<const uint8_t*>()[5], db.map<const uint8_t*>()[6], db.map<const uint8_t*>()[7], db.map<const uint8_t*>()[8]);
                        run(D["ffn"]["down2"].get<std::string>(), {"pp_d2_" + cs, gb.ffn_program[1].args[0], "pp_y_" + cs});
                        read_back(yb, M * hid * 2, 0);
                        double num = 0, den = 0; size_t nans = 0;
                        for (size_t i = 0; i < M * hid; ++i) {
                            const double r = bf16_to_f32(yb.map<const uint16_t*>()[i]), d = ydev[i] - r;
                            if (!std::isfinite(ydev[i])) ++nans;
                            num += d * d; den += r * r;
                        }
                        std::fprintf(stderr, "ffn_check layer %zu: |y_dev - y_host| / |y_host| = %.3e, %zu non-finite; unit 0 bytes equal %d/9216\n",
                                     l, std::sqrt(num / den), nans, same);
                    }
                } else if (op == UP && down2) {
                    run(D["ffn"]["up2"].get<std::string>(), {"pp_a2_" + cs, gb.ffn_program[0].args[0], "pp_h_" + cs});
                } else if (op == UP) {
                    if (up2) run(D["ffn"]["up2"].get<std::string>(), {"pp_a2_" + cs, gb.ffn_program[0].args[0], "pp_h"});
                    else if (dit_hs_) throw std::runtime_error("open_qwen36: the dit FFN up has no dit2 kernel under hs");
                    else run(D["ffn"]["up"].get<std::string>(), {"pp_a_" + cs, gb.ffn_program[0].args[0], "pp_h"});
                    run(D["ffn"]["down"].get<std::string>(), {"pp_h", gb.ffn_program[1].args[0], "pp_y_" + cs});
                } else {   // DOWN
                    run(D["ffn"]["down2"].get<std::string>(), {"pp_d2_" + cs, gb.ffn_program[1].args[0], "pp_y_" + cs});
                }
                npu_busy_ms += ms_since(tb);
                {
                    std::lock_guard<std::mutex> lk(mu);
                    busy[cell(l, c)] = 0;
                    last_fa = op == FAT ? 1 : 0;
                    ++stage[cell(l, c)];
                }
                cv.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(mu);
            if (!err) err = std::current_exception();
            cv.notify_all();
        }
    });

    // ---- the host stages, here
    auto to_bf16 = [&](const float* src, size_t rows, size_t w, const std::string& buf, float scale = 1.f) {
        xrt::bo& b = buffer(buf, 0);
        uint16_t* d = b.map<uint16_t*>();
#pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(rows * w); ++i) d[i] = f32_to_bf16(src[i] * scale);
        b.sync(XCL_BO_SYNC_BO_TO_DEVICE, rows * w * 2, 0);
    };
    auto to_bfp2 = [&](const float* src, size_t w, const std::string& buf, size_t rows) {
        // rows past `rows` (the chunk's padding) go in as zeros
        xrt::bo& b = buffer(buf, 0);
        dit2_pack_a(src, rows, M, w, b.map<uint8_t*>(), dit_hs_, MT);
        b.sync(XCL_BO_SYNC_BO_TO_DEVICE, M * w * 9 / 8, 0);
    };
    std::vector<float> ya, yb, yc, yd, og, hf;
    // OFLM_OPEN_FA_CHECK=1 (debug): every NPU-attention chunk also runs host attention, and GATE
    // prints the NPU output's error against it per (layer, chunk)
    static const bool fa_check = [] { const char* e = std::getenv("OFLM_OPEN_FA_CHECK"); return e && *e == '1'; }();
    std::map<size_t, std::vector<float>> fa_ref;
    std::vector<std::vector<float>> qprep(C);      // noncausal: each chunk's prepped Q, MID -> GATE
    if (noncausal_ && pos0 != 0) throw std::runtime_error("open_qwen36: noncausal attention needs a fresh prompt (position 0)");
    try {
        for (;;) {
            size_t l = 0, c = 0;
            {
                std::unique_lock<std::mutex> lk(mu);
                auto ti = std::chrono::steady_clock::now();
                cv.wait(lk, [&] { return err || finished == L * C || pick(false, l, c); });
                host_idle_ms += ms_since(ti);
                if (err) std::rethrow_exception(err);
                if (finished == L * C) break;
                busy[cell(l, c)] = 1;
            }
            const int op = ops[l][stage[cell(l, c)]];
            int advance = 1;
            const LayerType& lt = *types_[l];
            const GemmBlockProgram& gb = lt.gemm_block;
            const HostConsts& hc = hc_[l];
            const std::string cs = std::to_string(c);
            const size_t r0 = c * M, t_real = std::min(M, N - std::min(N, r0));
            float* xr = xres.data() + r0 * hid;
            float* rs = res.data() + r0 * hid;
            float* xc = xn.data() + r0 * hid;
            auto th = std::chrono::steady_clock::now();
            if (op == PRE) {
                host::rmsnorm_rows(xr, t_real, hid, hc.ln.data(), gb.eps, xc);
                const size_t nq = gb.kind == "linear" ? gb.qkv_dim + gb.vw : 2 * gb.qw + 2 * gb.kvw;
                if (has2(nq, hid)) to_bfp2(xc, hid, "pp_a2_" + cs, t_real);
                else to_bf16(xc, M, hid, "pp_a_" + cs);
            } else if (op == MID) {
                xrt::bo& qb = buffer("pp_q_" + cs, 0);
                const bool lin = gb.kind == "linear";
                const size_t nq = lin ? gb.qkv_dim + gb.vw : 2 * gb.qw + 2 * gb.kvw;
                read_back(qb, t_real * nq * 2, 0);
                const uint16_t* q16 = qb.map<const uint16_t*>();
                xrt::bo& st = state_[l];
                uint8_t* sp = st.map<uint8_t*>();
                if (lin) {
                    const size_t nch = gb.qkv_dim, vw = gb.vw;
                    if (c == 0) read_back(st, lt.state_bytes, 0);
                    ya.resize(M * nch);
                    yb.resize(M * vw);
                    og.resize(M * vw);
#pragma omp parallel for
                    for (long long t = 0; t < static_cast<long long>(t_real); ++t) {
                        for (size_t i = 0; i < nch; ++i) ya[t * nch + i] = bf16_to_f32(q16[t * nq + i]);
                        for (size_t i = 0; i < vw; ++i) yb[t * vw + i] = bf16_to_f32(q16[t * nq + nch + i]);
                    }
                    host::DeltaGeom g;
                    g.T = M; g.hid = hid;
                    g.key_heads = gb.key_heads; g.value_heads = gb.value_heads; g.head_dim = gb.head_dim;
                    g.taps = gb.conv_kernel; g.lanes = hc.lanes; g.s_rows = gb.s_rows; g.eps = gb.eps;
                    g.t_real = t_real;
                    g.reset = packed ? rflag.data() + r0 : nullptr;
                    double ph[2] = {0, 0};
                    host::deltanet_block(g, ya.data(), yb.data(), xc, hc.convw.data(), hc.Wa.data(), hc.Wb.data(),
                                         hc.A.data(), hc.dtb.data(), hc.nw.data(), reinterpret_cast<uint16_t*>(sp),
                                         reinterpret_cast<float*>(sp + gb.state_s_off), og.data(), ph);
                    timing_.dn_conv_ms += ph[0];
                    timing_.dn_rule_ms += ph[1];
                    if (c + 1 == C) st.sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
                    if (has2(hid, vw)) to_bfp2(og.data(), vw, "pp_m2_" + cs, t_real);
                    else to_bf16(og.data(), M, vw, "pp_m_" + cs);
                } else {
                    const size_t qw = gb.qw, kvw = gb.kvw, row = lt.state_row;
                    const std::string fk = fa_kernel(c);
                    if (c == 0 && pos0 > 0) read_back(st, pos0 * row, 0);
                    ya.resize(M * qw); yb.resize(M * kvw); yc.resize(M * kvw); yd.resize(M * qw); og.resize(M * qw);
#pragma omp parallel for
                    for (long long t = 0; t < static_cast<long long>(t_real); ++t) {
                        const uint16_t* r = q16 + t * nq;
                        for (size_t i = 0; i < qw; ++i) ya[t * qw + i] = bf16_to_f32(r[i]);
                        for (size_t i = 0; i < kvw; ++i) yb[t * kvw + i] = bf16_to_f32(r[qw + i]);
                        for (size_t i = 0; i < kvw; ++i) yc[t * kvw + i] = bf16_to_f32(r[qw + kvw + i]);
                        if (fk.empty())
                            for (size_t i = 0; i < qw; ++i) yd[t * qw + i] = bf16_to_f32(r[qw + 2 * kvw + i]);
                    }
                    host::AttnGeom g;
                    g.T = M; g.t_real = t_real; g.nh = gb.nh; g.kvh = gb.kvh; g.hd = gb.hd; g.rot = gb.rot;
                    g.pos0 = pos0 + r0; g.eps = gb.eps;
                    g.rpos = packed ? rpos.data() + r0 : nullptr;
                    auto ta = std::chrono::steady_clock::now();
                    if (noncausal_ && nc_npu && nc_b) {
                        // dit_fa256b: Q packed per head times log2(e) (og carries 1/sqrt(hd): the
                        // kernel's scores are q.k log2(e) / sqrt(hd)); this chunk's K / V rows,
                        // from the cache attention_prep just wrote, packed into the layer's pair
                        host::attention_prep(g, ya.data(), yb.data(), yc.data(), hc.qn.data(), hc.kn.data(),
                                             man_.rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, og.data());
                        timing_.attn_ms += ms_since(ta);
                        const size_t nh = gb.nh, kvh = gb.kvh, hd = gb.hd, qb_head = (M / 32) * kFabQBlock;
                        xrt::bo& qbuf = buffer("fab_q_" + cs, 0);
                        uint8_t* qm = qbuf.map<uint8_t*>();
                        const std::string par = std::to_string(full_ord[l] % 2);
                        xrt::bo& kbuf = buffer("fab_k" + par, 0);
                        xrt::bo& vbuf = buffer("fab_v" + par, 0);
                        const uint16_t* kvr = st.map<const uint16_t*>() + (pos0 + r0) * (row / 2);
                        const size_t kv_head = (lkn / 64) * kFabChunk, at = (r0 / 64) * kFabChunk, span = (M / 64) * kFabChunk;
#pragma omp parallel for schedule(dynamic, 1)
                        for (long long w = 0; w < static_cast<long long>(nh + 2 * kvh); ++w) {
                            const size_t u = static_cast<size_t>(w);
                            if (u < nh)
                                fab_pack_q(og.data(), qw, u * hd, M, t_real, 1.4426950408889634f, qm + u * qb_head);
                            else if (u < nh + kvh)
                                fab_pack_k(kvr, row / 2, (u - nh) * hd, M, t_real, kbuf.map<uint8_t*>() + (u - nh) * kv_head + at);
                            else
                                fab_pack_v(kvr, row / 2, kvh * hd + (u - nh - kvh) * hd, M, t_real,
                                           vbuf.map<uint8_t*>() + (u - nh - kvh) * kv_head + at);
                        }
                        qbuf.sync(XCL_BO_SYNC_BO_TO_DEVICE, nh * qb_head, 0);
                        for (size_t g2 = 0; g2 < kvh; ++g2) {
                            kbuf.sync(XCL_BO_SYNC_BO_TO_DEVICE, span, g2 * kv_head + at);
                            vbuf.sync(XCL_BO_SYNC_BO_TO_DEVICE, span, g2 * kv_head + at);
                        }
                    } else if (noncausal_ && nc_npu) {
                        // Q for the noncausal stream (unscaled, as the causal one), K / V into the cache
                        host::attention_prep(g, ya.data(), yb.data(), yc.data(), hc.qn.data(), hc.kn.data(),
                                             man_.rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, og.data());
                        timing_.attn_ms += ms_since(ta);
                        to_bf16(og.data(), M, qw, "fa_q_" + cs, std::sqrt(static_cast<float>(gb.hd)));
                    } else if (noncausal_) {
                        // Q prepped for GATE (after every chunk's K / V rows are in the cache)
                        qprep[c].resize(M * qw);
                        host::attention_prep(g, ya.data(), yb.data(), yc.data(), hc.qn.data(), hc.kn.data(),
                                             man_.rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, qprep[c].data());
                        timing_.attn_ms += ms_since(ta);
                    } else if (fk.empty()) {
                        host::attention_block(g, ya.data(), yb.data(), yc.data(), yd.data(), hc.qn.data(), hc.kn.data(),
                                              man_.rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, og.data());
                        timing_.attn_ms += ms_since(ta);
                        if (has2(hid, qw)) to_bfp2(og.data(), qw, "pp_m2_" + cs, t_real);
                        else to_bf16(og.data(), M, qw, "pp_m_" + cs);
                        // no NPU attention for this chunk: past FAT and GATE (when the layer has them)
                        if (static_cast<size_t>(stage[cell(l, c)]) + 1 < ops[l].size() && ops[l][stage[cell(l, c)] + 1] == FAT) advance = 3;
                    } else {
                        // NPU attention: q / k normed and roped here, K / V rows into the cache (synced),
                        // Q to the kernel unscaled (it applies 1 / sqrt(hd) itself; prep scaled by it)
                        if (fa_check) {
                            std::vector<float> gt(M * qw), ref(M * qw);
#pragma omp parallel for
                            for (long long t = 0; t < static_cast<long long>(M); ++t)
                                for (size_t i = 0; i < qw; ++i) gt[t * qw + i] = bf16_to_f32(q16[t * nq + qw + 2 * kvw + i]);
                            host::attention_block(g, ya.data(), yb.data(), yc.data(), gt.data(), hc.qn.data(), hc.kn.data(),
                                                  man_.rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, ref.data());
                            fa_ref[cell(l, c)] = std::move(ref);
                        }
                        host::attention_prep(g, ya.data(), yb.data(), yc.data(), hc.qn.data(), hc.kn.data(),
                                             man_.rope_inv_freq.data(), st.map<uint16_t*>(), row / 2, og.data());
                        timing_.attn_ms += ms_since(ta);
                        to_bf16(og.data(), M, qw, "fa_q_" + cs, std::sqrt(static_cast<float>(gb.hd)));
                    }
                    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, t_real * row, (pos0 + r0) * row);
                }
            } else if (op == GATE && noncausal_ && !nc_npu) {
                // d3's noncausal full attention: this chunk's queries over the whole prompt's K / V
                const size_t qw = gb.qw, kvw = gb.kvw, nq = 2 * qw + 2 * kvw, row = lt.state_row;
                host::AttnGeom g;
                g.T = M; g.t_real = t_real; g.nh = gb.nh; g.kvh = gb.kvh; g.hd = gb.hd; g.rot = gb.rot;
                g.pos0 = pos0 + r0; g.eps = gb.eps;
                og.resize(M * qw);
                auto ta = std::chrono::steady_clock::now();
                if (!packed) {
                    host::attention_full(g, qprep[c].data(), state_[l].map<uint16_t*>(), row / 2, pos0 + N, og.data());
                } else {
                    // each prompt's queries in this chunk over that prompt's own keys
                    std::fill(og.begin(), og.end(), 0.f);
                    const uint16_t* kv = state_[l].map<const uint16_t*>();
                    for (size_t i = 0; i < segs.size(); ++i) {
                        const size_t a = segs[i], b = i + 1 < segs.size() ? segs[i + 1] : N;
                        const size_t q0 = std::max(a, r0), q1 = std::min(b, r0 + t_real);
                        if (q0 >= q1) continue;
                        host::AttnGeom gs = g;
                        gs.T = gs.t_real = q1 - q0;
                        gs.pos0 = 0;
                        host::attention_full(gs, qprep[c].data() + (q0 - r0) * qw, kv + a * (row / 2), row / 2, b - a,
                                             og.data() + (q0 - r0) * qw);
                    }
                }
                timing_.attn_ms += ms_since(ta);
                const uint16_t* q16 = buffer("pp_q_" + cs, 0).map<const uint16_t*>();
#pragma omp parallel for
                for (long long t = 0; t < static_cast<long long>(t_real); ++t)
                    for (size_t i = 0; i < qw; ++i)
                        og[t * qw + i] = static_cast<size_t>(t) < t_real
                                             ? og[t * qw + i] / (1.f + std::exp(-bf16_to_f32(q16[t * nq + qw + 2 * kvw + i]))) : 0.f;
                if (has2(hid, qw)) to_bfp2(og.data(), qw, "pp_m2_" + cs, t_real);
                else to_bf16(og.data(), M, qw, "pp_m_" + cs);
            } else if (op == GATE) {
                // og = O * sigmoid(gate), gate from the qkv GEMM's C (still this chunk's)
                const size_t qw = gb.qw, kvw = gb.kvw, nq = 2 * qw + 2 * kvw;
                xrt::bo& ob = buffer("fa_o_" + cs, 0);
                read_back(ob, t_real * qw * 2, 0);
                const uint16_t* o16 = ob.map<const uint16_t*>();
                const uint16_t* q16 = buffer("pp_q_" + cs, 0).map<const uint16_t*>();
                og.resize(M * qw);
#pragma omp parallel for
                for (long long t = 0; t < static_cast<long long>(t_real); ++t)
                    for (size_t i = 0; i < qw; ++i) {
                        const float gt = bf16_to_f32(q16[t * nq + qw + 2 * kvw + i]);
                        og[t * qw + i] = static_cast<size_t>(t) < t_real ? bf16_to_f32(o16[t * qw + i]) / (1.f + std::exp(-gt)) : 0.f;
                    }
                if (fa_check && fa_ref.count(cell(l, c))) {
                    const std::vector<float>& ref = fa_ref[cell(l, c)];
                    double num = 0, den = 0, worst = 0; size_t wr = 0;
                    for (size_t t = 0; t < t_real; ++t) {
                        double n2 = 0, d2 = 0;
                        for (size_t i = 0; i < qw; ++i) {
                            const double e = og[t * qw + i] - ref[t * qw + i];
                            n2 += e * e; d2 += static_cast<double>(ref[t * qw + i]) * ref[t * qw + i];
                        }
                        num += n2; den += d2;
                        const double r = std::sqrt(n2 / std::max(d2, 1e-30));
                        if (r > worst) { worst = r; wr = t; }
                    }
                    std::fprintf(stderr, "fa_check layer %zu chunk %zu: rel_fro %.3e worst row %zu (pos %zu) rel %.3e\n", l, c,
                                 std::sqrt(num / den), wr, pos0 + r0 + wr, worst);
                }
                if (has2(hid, qw)) to_bfp2(og.data(), qw, "pp_m2_" + cs, t_real);
                else to_bf16(og.data(), M, qw, "pp_m_" + cs);
            } else if (op == TAIL) {
                xrt::bo& yb2 = buffer("pp_y_" + cs, 0);
                read_back(yb2, t_real * hid * 2, 0);
                const uint16_t* y16 = yb2.map<const uint16_t*>();
#pragma omp parallel for
                for (long long i = 0; i < static_cast<long long>(t_real * hid); ++i) rs[i] = xr[i] + bf16_to_f32(y16[i]);
                host::rmsnorm_rows(rs, t_real, hid, hc.postln.data(), gb.eps, xc);
                if (up2) to_bfp2(xc, hid, "pp_a2_" + cs, t_real);
                else to_bf16(xc, M, hid, "pp_a_" + cs);
            } else if (op == HPACK) {
                // h = the first 64 of every 128 columns of the up GEMM's C, packed as dit2's A
                xrt::bo& hb = buffer("pp_h_" + cs, 0);
                read_back(hb, M * 2 * ff * 2, 0);
                const uint16_t* h16 = hb.map<const uint16_t*>();
                hf.resize(M * ff);
#pragma omp parallel for
                for (long long r = 0; r < static_cast<long long>(M); ++r)
                    for (size_t j = 0; j < ff; ++j) hf[r * ff + j] = bf16_to_f32(h16[r * 2 * ff + (j / 64) * 128 + j % 64]);
                to_bfp2(hf.data(), ff, "pp_d2_" + cs, M);
            } else {   // RES
                xrt::bo& yb2 = buffer("pp_y_" + cs, 0);
                read_back(yb2, t_real * hid * 2, 0);
                const uint16_t* y16 = yb2.map<const uint16_t*>();
#pragma omp parallel for
                for (long long i = 0; i < static_cast<long long>(t_real * hid); ++i) xr[i] = rs[i] += bf16_to_f32(y16[i]);
            }
            hst[op] += ms_since(th);
            {
                std::lock_guard<std::mutex> lk(mu);
                busy[cell(l, c)] = 0;
                stage[cell(l, c)] += advance;
                if (stage[cell(l, c)] >= nst(l)) ++finished;
            }
            cv.notify_all();
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!err) err = std::current_exception();
        }
        cv.notify_all();
        npu.join();
        throw;
    }
    npu.join();
    if (err) std::rethrow_exception(err);
    state_owner_ = 0;
    pos_ += static_cast<int>(N);
    last_x_.assign(xres.data() + (N - 1) * hid, xres.data() + N * hid);
    for (size_t i = 0; i < segs.size(); ++i) {
        const size_t e = i + 1 < segs.size() ? segs[i + 1] : N;
        seg_x_.emplace_back(xres.data() + (e - 1) * hid, xres.data() + e * hid);
    }
    block_logits_.clear();
    if (block_logits_all_)
        for (size_t t = 0; t < N; ++t) {
            tail_logits(xres.data() + t * hid);
            block_logits_.push_back(logits_host_);
        }
    if (want_logits) tail_logits(xres.data() + (N - 1) * hid);
    timing_.total_ms = ms_since(t0);
    timing_.part0_ms = npu_busy_ms;
    timing_.part1_ms = hst[PRE] + hst[MID] + hst[GATE] + hst[TAIL] + hst[HPACK] + hst[RES];
    timing_.mid_ms = hst[MID];
    std::fprintf(stderr, "open_qwen36: pipe %zu tokens (%zu chunks of %zu, %zu layers): %.0f ms | NPU busy %.0f (attention %d "
                         "calls %.0f) idle %.0f | host pre %.0f mid %.0f (dn conv %.0f rule %.0f attn %.0f) gate %.0f tail %.0f "
                         "hpack %.0f res %.0f idle %.0f\n",
                 N, C, M, L, timing_.total_ms, npu_busy_ms, n_fa, fa_ms, npu_idle_ms, hst[PRE], hst[MID], timing_.dn_conv_ms,
                 timing_.dn_rule_ms, timing_.attn_ms, hst[GATE], hst[TAIL], hst[HPACK], hst[RES], host_idle_ms);
}

void Core::moe_token(int l, const float* xm, const float* res, const float* probs, const int32_t* idx, const float* w,
                     float* out) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const size_t hid = w_->hidden, E = w_->moe.experts, topk = w_->moe.topk;
    xrt::bo& act = act_[l];
    uint8_t* a = act.map<uint8_t*>();
    auto tp = std::chrono::steady_clock::now();
    // what the sequential layer's first dispatch would have left in act: xm (bf16), the
    // router record [probs f32[E] | idx i32[8] @rout_idx_off | w f32[8]], the residual (f32)
    uint16_t* xmb = reinterpret_cast<uint16_t*>(a + gb.a_xm);
    for (size_t i = 0; i < hid; ++i) xmb[i] = f32_to_bf16(xm[i]);
    std::memcpy(a + gb.a_rout, probs, E * 4);
    int32_t* ri = reinterpret_cast<int32_t*>(a + gb.a_rout + w_->rout_idx_off);
    float* rw = reinterpret_cast<float*>(a + gb.a_rout + w_->rout_idx_off + 8 * 4);
    for (size_t s = 0; s < 8; ++s) {
        ri[s] = s < topk ? idx[s] : 0;
        rw[s] = s < topk ? w[s] : 0.f;
    }
    std::memcpy(a + gb.a_res, res, hid * 4);
    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, hid * 2, gb.a_xm);
    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, E * 4 + 16 * 4, gb.a_rout);
    act.sync(XCL_BO_SYNC_BO_TO_DEVICE, hid * 4, gb.a_res);
    timing_.moe_prep_ms += ms_since(tp);
    // the MoE-only dispatch (mx.py): the routed slots patched from the ids we just wrote
    // (no readback of the record), then run
    auto t0 = std::chrono::steady_clock::now();
    Kern& mk = kerns_.at(gb.moe_kernel);
    if (mk.moe2.empty()) throw std::runtime_error("open_qwen36: " + gb.moe_kernel + " has no routed-expert table");
    uint32_t slots[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (size_t s = 0; s < topk; ++s) {
        if (idx[s] < 0 || static_cast<unsigned>(idx[s]) >= E) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(idx[s]));
        slots[s] = static_cast<uint32_t>(idx[s]);
    }
    stream_patch::moe2_apply(mk.iw(), mk.moe2, slots, w_->moe);
    mk.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    timing_.moe_patch_ms += ms_since(t0);
    timing_.moe_run_ms += run(mk, gb.moe_args, l);
    auto t1 = std::chrono::steady_clock::now();
    xrt::bo& xr = buffer("xres", 0);
    read_back(xr, hid * 4, 0);
    std::memcpy(out, xr.map<uint8_t*>(), hid * 4);
    timing_.moe_read_ms += ms_since(t1);
    timing_.route_ms += ms_since(tp);
}

// The shared expert, once over the block instead of once per token: up|gate (one GEMM --
// the two are contiguous in the pool and both band-law) then down, with silu, the sigmoid
// gate and the add on the host. The kernels' own formula, from moe_silu32 / moe_hdr2 /
// moe_accfin: h = silu(g) * u, out += sigmoid(xm . sgw) * down(h). xm is rounded to bf16
// first because that is what the dispatch would have seen.
void Core::ffn_block(int l, const std::vector<Step>& prog, size_t ff, const float* xm, float* res, size_t T,
                     size_t t_real, const std::vector<float>* gate_w) {
    const size_t hid = w_->hidden;
    gemm(prog[0], xm, T, hid, 2 * ff, l, sg_ug_);
    const std::vector<float>& ug = sg_ug_;
    auto t0 = std::chrono::steady_clock::now();
    float* h = BlockScratch::fit(bs_.sh, T * ff);
#pragma omp parallel for
    for (long long tt = 0; tt < static_cast<long long>(T); ++tt) {
        const float* u = ug.data() + static_cast<size_t>(tt) * 2 * ff;
        const float* g = u + ff;
        float* ho = h + static_cast<size_t>(tt) * ff;
        for (size_t j = 0; j < ff; ++j) ho[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];
    }
    timing_.shared_ms += ms_since(t0);
    gemm(prog[1], h, T, ff, hid, l, sg_y_);
    const std::vector<float>& y = sg_y_;
    auto t1 = std::chrono::steady_clock::now();
    if (!gate_w) {
        // a dense FFN: every row, padding included, so the block's residual stays defined
#pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(T * hid); ++i) res[i] += y[static_cast<size_t>(i)];
        timing_.shared_ms += ms_since(t1);
        return;
    }
    const std::vector<float>& sgw = *gate_w;
#pragma omp parallel for
    for (long long tt = 0; tt < static_cast<long long>(t_real); ++tt) {
        const size_t t = static_cast<size_t>(tt);
        const float* x = xm + t * hid;
        double d = 0;
        for (size_t i = 0; i < hid; ++i) d += static_cast<double>(bf16_to_f32(f32_to_bf16(x[i]))) * sgw[i];
        const float gate = 1.f / (1.f + std::exp(-static_cast<float>(d)));
        const float* yr = y.data() + t * hid;
        float* r = res + t * hid;
        for (size_t i = 0; i < hid; ++i) r[i] += gate * yr[i];
    }
    timing_.shared_ms += ms_since(t1);
}

void Core::block_layer_linear(int l, float* xres, size_t T, size_t t_real, size_t row0, bool first, bool last) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = w_->hidden, nch = gb.qkv_dim, vw = gb.vw, E = w_->moe.experts, topk = w_->moe.topk;
    float* res = moe_.res.data() + row0 * hid;
    float* xm = moe_.xm.data() + row0 * hid;

    auto tn = std::chrono::steady_clock::now();
    float* xn = BlockScratch::fit(bs_.xn, T * hid);
    host::rmsnorm_rows(xres, T, hid, hc.ln.data(), gb.eps, xn);
    timing_.part1_ms += ms_since(tn);
    timing_.prenorm_ms += ms_since(tn);
    const float* yq = gemm_run(gb.program[0], xn, T, hid, nch + vw, l);
    float* qkv = BlockScratch::fit(bs_.part[0], T * nch);
    float* z = BlockScratch::fit(bs_.part[1], T * vw);
    {
        auto tt = std::chrono::steady_clock::now();
        const host::TransposePart parts[2] = {{qkv, 0, nch}, {z, nch, vw}};
        host::transpose_parts(yq, T, parts, 2);
        timing_.part1_ms += ms_since(tt);
        timing_.gemm_tr_ms += ms_since(tt);
    }
    // the conv rows and S live in the state BO; the recurrence runs on the host in place
    // The recurrence runs in the host map. On the layer-major route nothing between one
    // block and the next touches this layer's state BO, so it is pulled off the device
    // once at the layer's first block and pushed back once at its last.
    xrt::bo& st = state_[l];
    auto ts = std::chrono::steady_clock::now();
    if (first) read_back(st, lt.state_bytes, 0);
    timing_.state_ms += ms_since(ts);
    uint8_t* sp = st.map<uint8_t*>();
    host::DeltaGeom g;
    g.T = T; g.t_real = t_real; g.hid = hid;
    g.key_heads = gb.key_heads; g.value_heads = gb.value_heads; g.head_dim = gb.head_dim; g.taps = gb.conv_kernel;
    g.lanes = hc.lanes; g.s_rows = gb.s_rows; g.eps = gb.eps;
    float* og = BlockScratch::fit(bs_.og, T * vw);
    auto t0 = std::chrono::steady_clock::now();
    double phase[2] = {0, 0};
    host::deltanet_block(g, qkv, z, xn, hc.convw.data(), hc.Wa.data(), hc.Wb.data(), hc.A.data(),
                         hc.dtb.data(), hc.nw.data(), reinterpret_cast<uint16_t*>(sp),
                         reinterpret_cast<float*>(sp + gb.state_s_off), og, phase);
    timing_.dn_conv_ms += phase[0];
    timing_.dn_rule_ms += phase[1];
    timing_.part1_ms += ms_since(t0);
    timing_.mid_ms += ms_since(t0);
    ts = std::chrono::steady_clock::now();
    if (last) st.sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
    timing_.state_ms += ms_since(ts);
    auto t1 = std::chrono::steady_clock::now();
    {
        // a q8 out projection is a split step (out_split marks it) and gemm() adds its halves
        gemm(gb.program[1], og, T, vw, hid, l, gout_);
        t1 = std::chrono::steady_clock::now();
        const std::vector<float>& out = gout_;
#pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(T * hid); ++i) res[i] = xres[i] + out[i];
    }
    host::rmsnorm_rows(res, T, hid, hc.postln.data(), gb.eps, xm);
    if (gb.has_dense_ffn()) {
        timing_.part1_ms += ms_since(t1);
        timing_.tail_ms += ms_since(t1);
        ffn_block(l, gb.ffn_program, gb.ff, xm, res, T, t_real, nullptr);
        std::memcpy(xres, res, T * hid * 4);         // the layer's output; block_layer_moe has nothing to add
        return;
    }
    host::router_block(t_real, hid, E, topk, xm, hc.router.data(), moe_.probs.data() + row0 * E,
                       moe_.idx.data() + row0 * topk, moe_.w.data() + row0 * topk);
    timing_.part1_ms += ms_since(t1);
    timing_.tail_ms += ms_since(t1);
    ffn_block(l, gb.shared_program, gb.shared_ff, xm, res, T, t_real, &hc.sgw);
}

void Core::attention_npu(int l, const host::AttnGeom& g, const float* Q, const float* gate, const uint16_t* kv,
                         size_t kv_row_elems, float* og) {
    const AttnBlock& ab = types_[l]->gemm_block.attn_block;
    const size_t T = g.T, hd = g.hd, grp = g.nh / g.kvh, M = grp * T, qw = g.nh * hd, kvw = g.kvh * hd;
    if (M != ab.m || hd != ab.hd)
        throw std::runtime_error("open_qwen36: attn_block was built for " + std::to_string(ab.m) + " rows of head dim " +
                                 std::to_string(ab.hd) + ", this layer has " + std::to_string(M) + " of " + std::to_string(hd));
    const size_t rows = g.pos0 + g.t_real;                  // the window: every cached row and the block's own
    xrt::bo& ba = buffer(ab.args[0], 0);
    xrt::bo& bb = buffer(ab.args[1], 0);
    xrt::bo& bc = buffer(ab.args[2], 0);
    if (ba.size() < M * ab.l_max * 2 || bb.size() < ab.l_max * hd * 2 || bc.size() < M * ab.l_max * 4)
        throw std::runtime_error("open_qwen36: the attn_block globals are smaller than the widest window");
    // row r of a product is query head r / T of the group at token r % T
    size_t* pos = BlockScratch::fit(bs_.pos, M);
    for (size_t r = 0; r < M; ++r) pos[r] = g.pos0 + r % T;
    uint16_t* qb = BlockScratch::fit(bs_.qb, M * hd);
    float* acc = BlockScratch::fit(bs_.acc, M * hd);
    std::fill(og, og + T * qw, 0.f);
    // The kv group's queries as the scores' A operand: row hl * T + t is query head hl of the group
    auto group_queries = [&](size_t gh) {
        for (size_t hl = 0; hl < grp; ++hl)
            for (size_t t = 0; t < T; ++t) {
                const float* src = Q + t * qw + (gh * grp + hl) * hd;
                uint16_t* dst = qb + (hl * T + t) * hd;
                for (size_t j = 0; j < hd; ++j) dst[j] = f32_to_bf16(src[j]);
            }
    };
    // og for the group's real tokens: the merged sum over the denominator, gated when the layer is
    auto group_out = [&](size_t gh, const float* lsum_g) {
        for (size_t hl = 0; hl < grp; ++hl)
            for (size_t t = 0; t < g.t_real; ++t) {
                const size_t r = hl * T + t, h = gh * grp + hl;
                const float inv = 1.0f / lsum_g[r];
                float* out = og + t * qw + h * hd;
                if (!gate) {
                    for (size_t j = 0; j < hd; ++j) out[j] = acc[r * hd + j] * inv;
                    continue;
                }
                const float* gt = gate + t * qw + h * hd;
                for (size_t j = 0; j < hd; ++j) out[j] = acc[r * hd + j] * inv / (1.0f + std::exp(-gt[j]));
            }
    };
    // A window that fits the widest stream is one chunk: every group's scores, then every group's
    // values. Per group this is the chunked loop below with one chunk, so the output is the same
    // bit for bit; only the dispatch order differs. It matters where the two products sit on
    // different xclbins (a dense head dim of 128: scores 8 columns wide, values 4), which the
    // per-group order switches between twice a group.
    if (rows <= ab.l_max) {
        const size_t L = (rows + 255) / 256 * 256;
        float* m_all = BlockScratch::fit(bs_.m, g.kvh * M);
        float* l_all = BlockScratch::fit(bs_.lsum, g.kvh * M);
        uint16_t* p_all = BlockScratch::fit(bs_.p_all, g.kvh * M * L);
        for (size_t gh = 0; gh < g.kvh; ++gh) {
            auto th = std::chrono::steady_clock::now();
            group_queries(gh);
            std::fill(m_all + gh * M, m_all + (gh + 1) * M, -std::numeric_limits<float>::infinity());
            std::fill(l_all + gh * M, l_all + (gh + 1) * M, 0.f);
            std::memcpy(ba.map<uint16_t*>(), qb, M * hd * 2);
            host::tile_rows_as_bt(kv + gh * hd, kv_row_elems, rows, L, hd, bb.map<uint16_t*>());
            ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, M * hd * 2, 0);
            bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, hd * L * 2, 0);
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            timing_.part0_ms += run(kerns_.at(ab.kernels_s.at(L)), ab.args, l);
            th = std::chrono::steady_clock::now();
            read_back(bc, M * L * 4, 0);
            timing_.sync_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            th = std::chrono::steady_clock::now();
            // acc only takes a rescale from an earlier chunk, and there is none
            host::softmax_chunk(M, L, hd, 0, bc.map<float*>(), pos, m_all + gh * M, l_all + gh * M, acc,
                                p_all + gh * M * L);
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
        }
        for (size_t gh = 0; gh < g.kvh; ++gh) {
            auto th = std::chrono::steady_clock::now();
            std::memcpy(ba.map<uint16_t*>(), p_all + gh * M * L, M * L * 2);
            host::tile_rows_as_b(kv + kvw + gh * hd, kv_row_elems, rows, L, hd, bb.map<uint16_t*>());
            ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, M * L * 2, 0);
            bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, L * hd * 2, 0);
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            timing_.part0_ms += run(kerns_.at(ab.kernels_pv.at(L)), ab.args, l);
            th = std::chrono::steady_clock::now();
            read_back(bc, M * hd * 4, 0);
            timing_.sync_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            th = std::chrono::steady_clock::now();
            std::fill(acc, acc + M * hd, 0.f);
            const float* c = bc.map<float*>();
            for (size_t i = 0; i < M * hd; ++i) acc[i] += c[i];
            group_out(gh, l_all + gh * M);
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
        }
        return;
    }
    float* m = BlockScratch::fit(bs_.m, M);
    float* lsum = BlockScratch::fit(bs_.lsum, M);
    for (size_t gh = 0; gh < g.kvh; ++gh) {
        auto th = std::chrono::steady_clock::now();
        group_queries(gh);
        std::fill(m, m + M, -std::numeric_limits<float>::infinity());
        std::fill(lsum, lsum + M, 0.f);
        std::fill(acc, acc + M * hd, 0.f);
        timing_.mid_ms += ms_since(th);
        timing_.part1_ms += ms_since(th);
        // the window in chunks of the widest stream, the softmax merged across them
        for (size_t c0 = 0; c0 < rows; c0 += ab.l_max) {
            const size_t lreal = std::min(ab.l_max, rows - c0);
            const size_t L = (lreal + 255) / 256 * 256;
            th = std::chrono::steady_clock::now();
            std::memcpy(ba.map<uint16_t*>(), qb, M * hd * 2);
            host::tile_rows_as_bt(kv + c0 * kv_row_elems + gh * hd, kv_row_elems, lreal, L, hd, bb.map<uint16_t*>());
            ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, M * hd * 2, 0);
            bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, hd * L * 2, 0);
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            timing_.part0_ms += run(kerns_.at(ab.kernels_s.at(L)), ab.args, l);
            th = std::chrono::steady_clock::now();
            read_back(bc, M * L * 4, 0);   // the whole M x L score matrix
            timing_.sync_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            th = std::chrono::steady_clock::now();
            host::softmax_chunk(M, L, hd, c0, bc.map<float*>(), pos, m, lsum, acc, ba.map<uint16_t*>());
            host::tile_rows_as_b(kv + c0 * kv_row_elems + kvw + gh * hd, kv_row_elems, lreal, L, hd, bb.map<uint16_t*>());
            ba.sync(XCL_BO_SYNC_BO_TO_DEVICE, M * L * 2, 0);
            bb.sync(XCL_BO_SYNC_BO_TO_DEVICE, L * hd * 2, 0);
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            timing_.part0_ms += run(kerns_.at(ab.kernels_pv.at(L)), ab.args, l);
            th = std::chrono::steady_clock::now();
            read_back(bc, M * hd * 4, 0);
            timing_.sync_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
            th = std::chrono::steady_clock::now();
            const float* c = bc.map<float*>();
            for (size_t i = 0; i < M * hd; ++i) acc[i] += c[i];
            timing_.mid_ms += ms_since(th);
            timing_.part1_ms += ms_since(th);
        }
        th = std::chrono::steady_clock::now();
        group_out(gh, lsum);
        timing_.mid_ms += ms_since(th);
        timing_.part1_ms += ms_since(th);
    }
}

void Core::block_layer_full(int l, float* xres, size_t T, size_t t_real, size_t pos0, size_t row0, bool first) {
    const double mid0 = timing_.mid_ms;   // whatever this layer adds to mid is the attention half
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = w_->hidden, qw = gb.qw, kvw = gb.kvw, nf = 2 * qw + 2 * kvw;
    const size_t E = w_->moe.experts, topk = w_->moe.topk;
    float* res = moe_.res.data() + row0 * hid;
    float* xm = moe_.xm.data() + row0 * hid;

    auto tn = std::chrono::steady_clock::now();
    float* xn = BlockScratch::fit(bs_.xn, T * hid);
    host::rmsnorm_rows(xres, T, hid, hc.ln.data(), gb.eps, xn);
    timing_.part1_ms += ms_since(tn);
    timing_.prenorm_ms += ms_since(tn);
    const float* yf = gemm_run(gb.program[0], xn, T, hid, nf, l);
    float* q = BlockScratch::fit(bs_.part[0], T * qw);
    float* k = BlockScratch::fit(bs_.part[1], T * kvw);
    float* v = BlockScratch::fit(bs_.part[2], T * kvw);
    float* gate = BlockScratch::fit(bs_.part[3], T * qw);
    {
        auto tt = std::chrono::steady_clock::now();
        const host::TransposePart parts[4] = {{q, 0, qw},
                                              {k, qw, kvw},
                                              {v, qw + kvw, kvw},
                                              {gate, qw + 2 * kvw, qw}};
        host::transpose_parts(yf, T, parts, 4);
        timing_.part1_ms += ms_since(tt);
        timing_.gemm_tr_ms += ms_since(tt);
    }
    // the KV rows: [0, pos0) read, [pos0, pos0 + t_real) written by the host attention.
    // The read is only needed for the layer's FIRST block -- the blocks after it want the
    // rows the blocks before them just wrote into this same host map, and pulling the
    // whole window back per block is a cost that grows with the square of the prompt.
    xrt::bo& st = state_[l];
    const size_t row = lt.state_row;
    auto ts = std::chrono::steady_clock::now();
    if (first && pos0 > 0) read_back(st, pos0 * row, 0);
    timing_.state_ms += ms_since(ts);
    host::AttnGeom g;
    g.T = T; g.t_real = t_real; g.nh = gb.nh; g.kvh = gb.kvh; g.hd = gb.hd; g.rot = gb.rot;
    g.pos0 = pos0; g.eps = gb.eps;
    float* og = BlockScratch::fit(bs_.og, T * qw);
    auto t0 = std::chrono::steady_clock::now();
    if (attn_block_on_ && gb.attn_block.present()) {
        float* Q = BlockScratch::fit(bs_.qrope, T * qw);
        host::attention_prep(g, q, k, v, hc.qn.data(), hc.kn.data(), w_->rope_inv_freq.data(),
                             st.map<uint16_t*>(), row / 2, Q);
        timing_.part1_ms += ms_since(t0);
        timing_.mid_ms += ms_since(t0);
        attention_npu(l, g, Q, gate, st.map<uint16_t*>(), row / 2, og);
    } else {
        host::attention_block(g, q, k, v, gate, hc.qn.data(), hc.kn.data(), w_->rope_inv_freq.data(),
                              st.map<uint16_t*>(), row / 2, og);
        timing_.part1_ms += ms_since(t0);
        timing_.mid_ms += ms_since(t0);
    }
    ts = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, t_real * row, pos0 * row);
    timing_.state_ms += ms_since(ts);
    timing_.attn_ms += timing_.mid_ms - mid0;
    gemm(gb.program[1], og, T, qw, hid, l, gout_);
    const std::vector<float>& out = gout_;

    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(T * hid); ++i) res[i] = xres[i] + out[i];
    host::rmsnorm_rows(res, T, hid, hc.postln.data(), gb.eps, xm);
    if (gb.has_dense_ffn()) {
        timing_.part1_ms += ms_since(t1);
        timing_.tail_ms += ms_since(t1);
        ffn_block(l, gb.ffn_program, gb.ff, xm, res, T, t_real, nullptr);
        std::memcpy(xres, res, T * hid * 4);         // the layer's output; block_layer_moe has nothing to add
        return;
    }
    host::router_block(t_real, hid, E, topk, xm, hc.router.data(), moe_.probs.data() + row0 * E,
                       moe_.idx.data() + row0 * topk, moe_.w.data() + row0 * topk);
    timing_.part1_ms += ms_since(t1);
    timing_.tail_ms += ms_since(t1);
    ffn_block(l, gb.shared_program, gb.shared_ff, xm, res, T, t_real, &hc.sgw);
}

void Core::gemm_phase(int l, const Step& s, size_t B, size_t T, size_t K, size_t N,
                      const std::function<const float*(size_t)>& prep,
                      const std::function<void(size_t, const float*)>& post) {
    if (dit_on_) {
        // dit_gemm: G = M / T route blocks a dispatch; A bf16 [M, K] and C bf16 [M, Npad], both
        // row-major, so post() gets y as ROW-major [T, N] (the callers check dit_on_). While
        // dispatch d runs, the host fills d + 1's A and converts + posts d - 1's C.
        const auto& D = man_.gemm_dit;
        const size_t M = D["m"].get<size_t>();
        if (M % T) throw std::runtime_error("open_qwen36: dit GEMM: M " + std::to_string(M) + " is not whole route blocks of " + std::to_string(T));
        const size_t G = M / T, nd = (B + G - 1) / G;
        const std::string key = std::to_string(N) + "x" + std::to_string(K);
        if (!D["kernels"].contains(key)) throw std::runtime_error("open_qwen36: dit GEMM: no kernel for " + key);
        if (dit_hs_) throw std::runtime_error("open_qwen36: dit GEMM " + key + ": this route runs dit_gemm, whose weights are "
                                              "not in the hs (2x2) order; use the pipelined prompt (OFLM_OPEN_PIPE=1)");
        Kern& k = kerns_.at(D["kernels"][key].get<std::string>());
        const size_t npad = D["npad"][std::to_string(N)].get<size_t>();
        const std::string xg = D["x"][std::to_string(K)].get<std::string>(), yg = D["y"][std::to_string(npad)].get<std::string>();
        for (const std::string& g : {xg, yg})
            if (!globals_.count(g + "_b")) globals_[g + "_b"] = alloc(globals_.at(g).size());
        const std::string xs[2] = {xg, xg + "_b"}, ys[2] = {yg, yg + "_b"};
        if (dit_y_.size() < M * N) dit_y_.resize(M * N);
        auto put = [&](size_t d, int buf) {
            auto t0 = std::chrono::steady_clock::now();
            uint16_t* a = buffer(xs[buf], 0).map<uint16_t*>();
            for (size_t j = 0; j < G; ++j) {
                const size_t b = d * G + j;
                uint16_t* dst = a + j * T * K;
                if (b < B) {
                    const float* x = prep(b);
#pragma omp parallel for
                    for (long long i = 0; i < static_cast<long long>(T * K); ++i) dst[i] = f32_to_bf16(x[i]);
                } else {
                    std::memset(dst, 0, T * K * 2);
                }
            }
            buffer(xs[buf], 0).sync(XCL_BO_SYNC_BO_TO_DEVICE, M * K * 2, 0);
            timing_.part1_ms += ms_since(t0);
            timing_.gemm_tile_ms += ms_since(t0);
        };
        auto get = [&](size_t d, int buf) {
            const size_t nb = std::min(G, B - d * G);
            auto t1 = std::chrono::steady_clock::now();
            xrt::bo& yb = buffer(ys[buf], 0);
            read_back(yb, nb * T * npad * 2, 0);
            const uint16_t* c = yb.map<const uint16_t*>();
            float* yf = dit_y_.data();
#pragma omp parallel for
            for (long long r = 0; r < static_cast<long long>(nb * T); ++r)
                for (size_t i = 0; i < N; ++i) yf[r * N + i] = bf16_to_f32(c[r * npad + i]);
            timing_.part1_ms += ms_since(t1);
            timing_.gemm_tr_ms += ms_since(t1);
            for (size_t j = 0; j < nb; ++j) post(d * G + j, yf + j * T * N);
        };
        put(0, 0);
        for (size_t d = 0; d < nd; ++d) {
            const int buf = static_cast<int>(d & 1);
            Inflight f = start_run(k, {xs[buf], s.args[0], ys[buf]}, l);
            timing_.part0_ms += f.submit_ms;
            if (d + 1 < nd) put(d + 1, buf ^ 1);
            if (d > 0) get(d - 1, buf ^ 1);
            const auto [elapsed, blocked] = wait_run(f);
            timing_.part0_ms += blocked;
            if (dispatch_log_) dispatch_stats_[k.name].add(f.submit_ms + elapsed);
        }
        get(nd - 1, static_cast<int>((nd - 1) & 1));
        return;
    }
    const size_t rows = s.split ? 2 * N : N;
    Step twin = s;
    twin.args[1] += "#1";
    twin.args[2] += "#1";
    const Step* step[2] = {&s, &twin};
    for (const Step* p : step) {
        xrt::bo& xb = buffer(p->args[1], 0);
        xrt::bo& yb = buffer(p->args[2], 0);
        if (xb.size() < K * T * 2 || yb.size() < rows * T * 4)
            throw std::runtime_error("open_qwen36: gemm " + s.kernel + ": the x / y globals " + p->args[1] + " / " +
                                     p->args[2] + " are smaller than [" + std::to_string(K) + "] x " +
                                     std::to_string(T) + " -> [" + std::to_string(rows) + "]");
    }
    Kern& k = kerns_.at(s.kernel);
    auto stage_in = [&](size_t b) {
        xrt::bo& xb = buffer(step[b & 1]->args[1], 0);
        const float* x = prep(b);
        auto t0 = std::chrono::steady_clock::now();
        host::tile_x(x, T, K, xb.map<uint16_t*>());
        timing_.part1_ms += ms_since(t0);
        timing_.gemm_tile_ms += ms_since(t0);
        auto ts = std::chrono::steady_clock::now();
        xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, K * T * 2, 0);
        timing_.sync_ms += ms_since(ts);
        timing_.part1_ms += ms_since(ts);
    };
    auto launch = [&](size_t b) {
        Inflight f = start_run(k, step[b & 1]->args, l);
        timing_.part0_ms += f.submit_ms;
        return f;
    };
    // One run outstanding: a second run of a kernel queued behind the first gave different bits run to run.
    stage_in(0);
    Inflight cur = launch(0);
    for (size_t b = 0; b < B; ++b) {
        if (b + 1 < B) stage_in(b + 1);
        const double submit = cur.submit_ms;
        // blocked, not elapsed: only the time the host sat waiting adds to the prefill
        const auto [elapsed, blocked] = wait_run(cur);
        timing_.part0_ms += blocked;
        if (dispatch_log_) dispatch_stats_[k.name].add(submit + elapsed);
        xrt::bo& yb = buffer(step[b & 1]->args[2], 0);
        auto ts = std::chrono::steady_clock::now();
        read_back(yb, rows * T * 4, 0);
        timing_.sync_ms += ms_since(ts);
        timing_.part1_ms += ms_since(ts);
        if (b + 1 < B) cur = launch(b + 1);
        const float* y = yb.map<const float*>();
        if (s.split) {
            // gemm_run's fold, into host memory and never back into the mapped y
            ts = std::chrono::steady_clock::now();
            std::vector<float>& sum = split_y_[step[b & 1]->args[2]];
            if (sum.size() < N * T) sum.resize(N * T);
            const float* lo = y + N * T;
            float* o = sum.data();
#pragma omp parallel for
            for (long long i = 0; i < static_cast<long long>(N * T); ++i) o[i] = y[i] + lo[i];
            timing_.part1_ms += ms_since(ts);
            timing_.gemm_tr_ms += ms_since(ts);
            y = o;
        }
        post(b, y);
    }
}

void Core::staged_tail(int l, size_t b, size_t T, size_t N, const float* y, const float* xres) {
    const GemmBlockProgram& gb = types_[l]->gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = man_.hidden, E = man_.moe.experts, topk = man_.moe.topk;
    const size_t row0 = b * T, t_real = std::min(T, N - row0);
    float* res = moe_.res.data() + row0 * hid;
    float* xm = moe_.xm.data() + row0 * hid;
    auto tt = std::chrono::steady_clock::now();
    const float* out = y;                         // dit: already [T, hid]
    if (!dit_on_) {
        float* o = BlockScratch::fit(gout_, T * hid);
        host::transpose(y, hid, T, o);
        out = o;
    }
    timing_.part1_ms += ms_since(tt);
    timing_.gemm_tr_ms += ms_since(tt);
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(T * hid); ++i) res[i] = xres[i] + out[i];
    host::rmsnorm_rows(res, T, hid, hc.postln.data(), gb.eps, xm);
    if (!gb.has_dense_ffn())
        host::router_block(t_real, hid, E, topk, xm, hc.router.data(), moe_.probs.data() + row0 * E,
                           moe_.idx.data() + row0 * topk, moe_.w.data() + row0 * topk);
    timing_.part1_ms += ms_since(t1);
    timing_.tail_ms += ms_since(t1);
}

void Core::ffn_staged(int l, const std::vector<Step>& prog, size_t ff, float* xres, size_t B, size_t T, size_t N,
                      const std::vector<float>* gate_w) {
    const size_t hid = man_.hidden;
    if (dit_on_ && !gate_w && man_.gemm_dit.contains("ffn") && dit_ffn_on()) {
        // The dense FFN on the device end to end: up|gate with the SwiGLU epilogue writes h into
        // the first 64 of every 128 columns of H, down reads H gathered. The host only converts
        // xm in and adds y onto the residual, both overlapped with the other dispatch.
        const auto& D = man_.gemm_dit;
        const size_t M = D["m"].get<size_t>(), G = M / T, nd = (B + G - 1) / G;
        if (dit_hs_) throw std::runtime_error("open_qwen36: the staged dit FFN runs dit_gemm; its weights are in the hs order");
        Kern& ku = kerns_.at(D["ffn"]["up"].get<std::string>());
        Kern& kd = kerns_.at(D["ffn"]["down"].get<std::string>());
        const std::string xg = D["x"][std::to_string(hid)].get<std::string>();
        const std::string hg = D["y"][std::to_string(2 * ff)].get<std::string>();
        const std::string yg = D["y"][std::to_string(hid)].get<std::string>();
        for (const std::string& g : {xg, yg})
            if (!globals_.count(g + "_b")) globals_[g + "_b"] = alloc(globals_.at(g).size());
        const std::string xs[2] = {xg, xg + "_b"}, ys[2] = {yg, yg + "_b"};
        auto put = [&](size_t d, int buf) {
            auto t0 = std::chrono::steady_clock::now();
            const size_t r0 = d * M, rows = std::min(M, B * T - r0);
            uint16_t* a = buffer(xs[buf], 0).map<uint16_t*>();
            const float* x = moe_.xm.data() + r0 * hid;
#pragma omp parallel for
            for (long long i = 0; i < static_cast<long long>(M * hid); ++i)
                a[i] = static_cast<size_t>(i) < rows * hid ? f32_to_bf16(x[i]) : 0;
            buffer(xs[buf], 0).sync(XCL_BO_SYNC_BO_TO_DEVICE, M * hid * 2, 0);
            timing_.part1_ms += ms_since(t0);
            timing_.gemm_tile_ms += ms_since(t0);
        };
        auto get = [&](size_t d, int buf) {
            auto t1 = std::chrono::steady_clock::now();
            const size_t r0 = d * M, rows = std::min(M, B * T - r0);
            xrt::bo& yb = buffer(ys[buf], 0);
            read_back(yb, rows * hid * 2, 0);
            const uint16_t* c = yb.map<const uint16_t*>();
            float* res = moe_.res.data() + r0 * hid;
            float* xo = xres + r0 * hid;
#pragma omp parallel for
            for (long long i = 0; i < static_cast<long long>(rows * hid); ++i) xo[i] = res[i] += bf16_to_f32(c[i]);
            timing_.part1_ms += ms_since(t1);
            timing_.shared_ms += ms_since(t1);
        };
        put(0, 0);
        for (size_t d = 0; d < nd; ++d) {
            const int buf = static_cast<int>(d & 1);
            Inflight fu = start_run(ku, {xs[buf], prog[0].args[0], hg}, l);
            timing_.part0_ms += fu.submit_ms;
            if (d + 1 < nd) put(d + 1, buf ^ 1);
            timing_.part0_ms += wait_run(fu).second;
            Inflight fd = start_run(kd, {hg, prog[1].args[0], ys[buf]}, l);
            timing_.part0_ms += fd.submit_ms;
            if (d > 0) get(d - 1, buf ^ 1);
            timing_.part0_ms += wait_run(fd).second;
        }
        get(nd - 1, static_cast<int>((nd - 1) & 1));
        return;
    }
    float* h_all = BlockScratch::fit(ss_.h, B * T * ff);
    gemm_phase(l, prog[0], B, T, hid, 2 * ff,
               [&](size_t b) -> const float* { return moe_.xm.data() + b * T * hid; },
               [&](size_t b, const float* y) {
                   auto tt = std::chrono::steady_clock::now();
                   const float* ug = y;
                   if (!dit_on_) {
                       float* u2 = BlockScratch::fit(sg_ug_, T * 2 * ff);
                       host::transpose(y, 2 * ff, T, u2);
                       ug = u2;
                   }
                   timing_.part1_ms += ms_since(tt);
                   timing_.gemm_tr_ms += ms_since(tt);
                   auto t0 = std::chrono::steady_clock::now();
                   float* h = h_all + b * T * ff;
#pragma omp parallel for
                   for (long long tk = 0; tk < static_cast<long long>(T); ++tk) {
                       const float* u = ug + static_cast<size_t>(tk) * 2 * ff;
                       const float* g = u + ff;
                       float* ho = h + static_cast<size_t>(tk) * ff;
                       for (size_t j = 0; j < ff; ++j) ho[j] = g[j] / (1.f + std::exp(-g[j])) * u[j];
                   }
                   timing_.shared_ms += ms_since(t0);
               });
    gemm_phase(l, prog[1], B, T, ff, hid,
               [&](size_t b) -> const float* { return h_all + b * T * ff; },
               [&](size_t b, const float* yd) {
                   const size_t row0 = b * T, t_real = std::min(T, N - row0);
                   float* res = moe_.res.data() + row0 * hid;
                   const float* xm = moe_.xm.data() + row0 * hid;
                   auto tt = std::chrono::steady_clock::now();
                   const float* y = yd;
                   if (!dit_on_) {
                       float* y2 = BlockScratch::fit(sg_y_, T * hid);
                       host::transpose(yd, hid, T, y2);
                       y = y2;
                   }
                   timing_.part1_ms += ms_since(tt);
                   timing_.gemm_tr_ms += ms_since(tt);
                   auto t1 = std::chrono::steady_clock::now();
                   if (!gate_w) {
                       // a dense FFN: every row, padding included, then the layer's output
#pragma omp parallel for
                       for (long long i = 0; i < static_cast<long long>(T * hid); ++i) res[i] += y[static_cast<size_t>(i)];
                       std::memcpy(xres + row0 * hid, res, T * hid * 4);
                       timing_.shared_ms += ms_since(t1);
                       return;
                   }
                   const std::vector<float>& sgw = *gate_w;
#pragma omp parallel for
                   for (long long tk = 0; tk < static_cast<long long>(t_real); ++tk) {
                       const size_t t = static_cast<size_t>(tk);
                       const float* x = xm + t * hid;
                       double d = 0;
                       for (size_t i = 0; i < hid; ++i) d += static_cast<double>(bf16_to_f32(f32_to_bf16(x[i]))) * sgw[i];
                       const float gate = 1.f / (1.f + std::exp(-static_cast<float>(d)));
                       const float* yr = y + t * hid;
                       float* r = res + t * hid;
                       for (size_t i = 0; i < hid; ++i) r[i] += gate * yr[i];
                   }
                   timing_.shared_ms += ms_since(t1);
               });
}

void Core::layer_linear_staged(int l, float* xres, size_t B, size_t T, size_t N) {
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = man_.hidden, nch = gb.qkv_dim, vw = gb.vw;
    float* xn_all = BlockScratch::fit(ss_.xn, B * T * hid);
    float* og_all = BlockScratch::fit(ss_.og, B * T * vw);
    // the conv rows and S, off the device once for the layer and back once after its last block
    xrt::bo& st = state_[l];
    auto ts = std::chrono::steady_clock::now();
    read_back(st, lt.state_bytes, 0);
    timing_.state_ms += ms_since(ts);
    uint8_t* sp = st.map<uint8_t*>();
    host::DeltaGeom g;
    g.T = T; g.hid = hid;
    g.key_heads = gb.key_heads; g.value_heads = gb.value_heads; g.head_dim = gb.head_dim; g.taps = gb.conv_kernel;
    g.lanes = hc.lanes; g.s_rows = gb.s_rows; g.eps = gb.eps;
    // DeltaNet still goes block by block in order: S and the conv rows chain through it
    gemm_phase(l, gb.program[0], B, T, hid, nch + vw,
               [&](size_t b) -> const float* {
                   auto tn = std::chrono::steady_clock::now();
                   float* xn = xn_all + b * T * hid;
                   host::rmsnorm_rows(xres + b * T * hid, T, hid, hc.ln.data(), gb.eps, xn);
                   timing_.part1_ms += ms_since(tn);
                   timing_.prenorm_ms += ms_since(tn);
                   return xn;
               },
               [&](size_t b, const float* yq) {
                   float* qkv = BlockScratch::fit(bs_.part[0], T * nch);
                   float* z = BlockScratch::fit(bs_.part[1], T * vw);
                   auto tt = std::chrono::steady_clock::now();
                   const host::TransposePart parts[2] = {{qkv, 0, nch}, {z, nch, vw}};
                   if (dit_on_) host::split_cols(yq, T, nch + vw, parts, 2);
                   else host::transpose_parts(yq, T, parts, 2);
                   timing_.part1_ms += ms_since(tt);
                   timing_.gemm_tr_ms += ms_since(tt);
                   g.t_real = std::min(T, N - b * T);
                   auto t0 = std::chrono::steady_clock::now();
                   double phase[2] = {0, 0};
                   host::deltanet_block(g, qkv, z, xn_all + b * T * hid, hc.convw.data(), hc.Wa.data(), hc.Wb.data(),
                                        hc.A.data(), hc.dtb.data(), hc.nw.data(), reinterpret_cast<uint16_t*>(sp),
                                        reinterpret_cast<float*>(sp + gb.state_s_off), og_all + b * T * vw, phase);
                   timing_.dn_conv_ms += phase[0];
                   timing_.dn_rule_ms += phase[1];
                   timing_.part1_ms += ms_since(t0);
                   timing_.mid_ms += ms_since(t0);
               });
    ts = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
    timing_.state_ms += ms_since(ts);
    // out (a q8 out projection is a split step, and gemm_phase adds its halves)
    gemm_phase(l, gb.program[1], B, T, vw, hid,
               [&](size_t b) -> const float* { return og_all + b * T * vw; },
               [&](size_t b, const float* y) { staged_tail(l, b, T, N, y, xres + b * T * hid); });
    if (gb.has_dense_ffn()) ffn_staged(l, gb.ffn_program, gb.ff, xres, B, T, N, nullptr);
    else ffn_staged(l, gb.shared_program, gb.shared_ff, xres, B, T, N, &hc.sgw);
}

void Core::layer_full_staged(int l, float* xres, size_t B, size_t T, size_t N, size_t pos0) {
    const double mid0 = timing_.mid_ms;   // whatever this layer adds to mid is the attention half
    const LayerType& lt = *types_[l];
    const GemmBlockProgram& gb = lt.gemm_block;
    const HostConsts& hc = hc_[l];
    const size_t hid = man_.hidden, qw = gb.qw, kvw = gb.kvw, nf = 2 * qw + 2 * kvw;
    const bool npu_attn = attn_block_on_ && gb.attn_block.present();
    float* q_all = npu_attn ? BlockScratch::fit(ss_.q, B * T * qw) : nullptr;
    float* gate_all = BlockScratch::fit(ss_.gate, B * T * qw);
    float* og_all = BlockScratch::fit(ss_.og, B * T * qw);
    // the KV rows: [0, pos0) cached on the device, [pos0, pos0 + N) written here on the host
    xrt::bo& st = state_[l];
    const size_t row = lt.state_row;
    auto ts = std::chrono::steady_clock::now();
    if (pos0 > 0) read_back(st, pos0 * row, 0);
    timing_.state_ms += ms_since(ts);
    auto geom = [&](size_t b) {
        host::AttnGeom g;
        g.T = T; g.t_real = std::min(T, N - b * T); g.nh = gb.nh; g.kvh = gb.kvh; g.hd = gb.hd; g.rot = gb.rot;
        g.pos0 = pos0 + b * T; g.eps = gb.eps;
        return g;
    };
    // each block's attention window is explicit, so later blocks' K / V rows in the map change nothing
    gemm_phase(l, gb.program[0], B, T, hid, nf,
               [&](size_t b) -> const float* {
                   auto tn = std::chrono::steady_clock::now();
                   float* xn = BlockScratch::fit(bs_.xn, T * hid);   // tiled before the next prep runs
                   host::rmsnorm_rows(xres + b * T * hid, T, hid, hc.ln.data(), gb.eps, xn);
                   timing_.part1_ms += ms_since(tn);
                   timing_.prenorm_ms += ms_since(tn);
                   return xn;
               },
               [&](size_t b, const float* yf) {
                   float* q = BlockScratch::fit(bs_.part[0], T * qw);
                   float* k = BlockScratch::fit(bs_.part[1], T * kvw);
                   float* v = BlockScratch::fit(bs_.part[2], T * kvw);
                   float* gate = gate_all + b * T * qw;
                   auto tt = std::chrono::steady_clock::now();
                   const host::TransposePart parts[4] = {
                       {q, 0, qw}, {k, qw, kvw}, {v, qw + kvw, kvw}, {gate, qw + 2 * kvw, qw}};
                   if (dit_on_) host::split_cols(yf, T, nf, parts, 4);
                   else host::transpose_parts(yf, T, parts, 4);
                   timing_.part1_ms += ms_since(tt);
                   timing_.gemm_tr_ms += ms_since(tt);
                   const host::AttnGeom g = geom(b);
                   auto t0 = std::chrono::steady_clock::now();
                   if (npu_attn)
                       host::attention_prep(g, q, k, v, hc.qn.data(), hc.kn.data(), man_.rope_inv_freq.data(),
                                            st.map<uint16_t*>(), row / 2, q_all + b * T * qw);
                   else
                       host::attention_block(g, q, k, v, gate, hc.qn.data(), hc.kn.data(), man_.rope_inv_freq.data(),
                                             st.map<uint16_t*>(), row / 2, og_all + b * T * qw);
                   timing_.part1_ms += ms_since(t0);
                   timing_.mid_ms += ms_since(t0);
               });
    // every block's attention in one visit to the attention products' context
    if (npu_attn)
        for (size_t b = 0; b < B; ++b)
            attention_npu(l, geom(b), q_all + b * T * qw, gate_all + b * T * qw, st.map<uint16_t*>(), row / 2,
                          og_all + b * T * qw);
    ts = std::chrono::steady_clock::now();
    st.sync(XCL_BO_SYNC_BO_TO_DEVICE, N * row, pos0 * row);
    timing_.state_ms += ms_since(ts);
    timing_.attn_ms += timing_.mid_ms - mid0;
    gemm_phase(l, gb.program[1], B, T, qw, hid,
               [&](size_t b) -> const float* { return og_all + b * T * qw; },
               [&](size_t b, const float* y) { staged_tail(l, b, T, N, y, xres + b * T * hid); });
    if (gb.has_dense_ffn()) ffn_staged(l, gb.ffn_program, gb.ff, xres, B, T, N, nullptr);
    else ffn_staged(l, gb.shared_program, gb.shared_ff, xres, B, T, N, &hc.sgw);
}

// The routed experts over the block on the token-batched kernel. Every expert's tokens are
// cut into visits of NT (a hot expert takes several slots of the same dispatch, since a slot
// can be patched to any expert); a pass fills the shortest stream that holds the visits still
// pending, gathers their tokens into x[slot] = the kernel's A tiles, runs, and scatters
// y[slot] back with the router weights. Three passes serve a block: 256 slots, then ~105
// of the 128-stream, then a handful (Poisson(8) tokens per expert).
void Core::moe_block(int l, const float* xm, const float* res, const int32_t* idx, const float* w, size_t T, size_t t_real,
                     float* out) {
    const MoeBatch& mb = types_[l]->gemm_block.moe_batch;
    const size_t hid = w_->hidden, E = w_->moe.experts, topk = w_->moe.topk, NT = mb.nt;
    auto tp = std::chrono::steady_clock::now();
    std::memcpy(out, res, T * hid * 4);
    std::vector<std::vector<std::pair<int, float>>> owed(E);   // per expert: its (token, weight) pairs
    for (size_t t = 0; t < t_real; ++t)
        for (size_t s = 0; s < topk; ++s) {
            const int32_t e = idx[t * topk + s];
            if (e < 0 || static_cast<size_t>(e) >= E) throw std::runtime_error("open_qwen36: router produced expert index " + std::to_string(e));
            owed[e].push_back({static_cast<int>(t), w[t * topk + s]});
        }
    std::vector<std::pair<uint32_t, size_t>> visits;             // (expert, first token of its NT)
    for (size_t e = 0; e < E; ++e)
        for (size_t off = 0; off < owed[e].size(); off += NT) visits.push_back({static_cast<uint32_t>(e), off});
    xrt::bo& xb = buffer(mb.args[1], 0);
    xrt::bo& yb = buffer(mb.args[3], 0);
    std::vector<std::vector<std::pair<size_t, float>>> per_token(t_real);   // (slot * NT + column, weight)
    static const bool log_passes = std::getenv("OFLM_OPEN_MOE_BATCH_LOG") != nullptr;
    for (size_t done = 0; done < visits.size();) {
        const size_t left = visits.size() - done;
        size_t slots = 0;
        const std::string* kname = nullptr;
        for (const auto& [s, k] : mb.kernels) {   // ascending: the shortest stream that holds them, else the longest
            slots = s;
            kname = &k;
            if (s >= left) break;
        }
        const size_t n = std::min(left, slots);
        auto t0 = std::chrono::steady_clock::now();
        uint16_t* xh = xb.map<uint16_t*>();
        std::vector<uint32_t> ex(slots, 0);   // unused slots stream expert 0: a valid read, ignored
        for (auto& v : per_token) v.clear();
        for (size_t i = 0; i < n; ++i) {
            const auto& [e, off] = visits[done + i];
            ex[i] = e;
            for (size_t j = 0; j < std::min(NT, owed[e].size() - off); ++j)
                per_token[owed[e][off + j].first].push_back({i * NT + j, owed[e][off + j].second});
        }
        // x[slot] as the kernel's A tiles: [hid / 8][NT tokens][8 k] bf16, which is already the
        // kernel's [k-block][NG sub-tiles][8 tokens][8 k] -- the sub-tile axis is the high bits
        // of the token index (designs/moe_batch/layout.py)
#pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const auto& [e, off] = visits[done + i];
            uint16_t* xs = xh + i * hid * NT;
            // A visit with fewer than NT tokens owed leaves the rest of its columns holding
            // the previous pass's activations. They are deliberately not cleared: the mmul's
            // token lanes are independent, and zeroing them was measured (2026-09-20, 600
            // tokens x 1 layer) to change not one bit of the output.
            for (size_t j = 0; j < std::min(NT, owed[e].size() - off); ++j) {
                const float* row = xm + owed[e][off + j].first * hid;
                for (size_t kb = 0; kb < hid / 8; ++kb)
                    for (size_t kl = 0; kl < 8; ++kl) xs[(kb * NT + j) * 8 + kl] = f32_to_bf16(row[kb * 8 + kl]);
            }
        }
        xb.sync(XCL_BO_SYNC_BO_TO_DEVICE, n * hid * NT * 2, 0);
        timing_.moe_prep_ms += ms_since(t0);
        auto t1 = std::chrono::steady_clock::now();
        Kern& mk = kerns_.at(*kname);
        stream_patch::moe2_apply(mk.iw(), mk.moe2, ex.data(), w_->moe);
        mk.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        timing_.moe_patch_ms += ms_since(t1);
        const double run_ms = run(mk, mb.args, l);
        timing_.moe_run_ms += run_ms;
        // Why the same dispatch costs 17.4 ms here and 12.4 ms under `--bench`: run it a
        // SECOND time, same stream, same buffers, no host work in between. If the repeat is
        // the bench's figure then what the first one pays for is the host work before it; if
        // both are 17.4 the hardware really is in a different state during a prefill.
        if (moe_redispatch_) {
            const double again = run(mk, mb.args, l);
            std::fprintf(stderr, "open_qwen36: layer %d moe redispatch: %.3f then %.3f ms\n", l, run_ms, again);
        }

        if (log_passes)
            std::fprintf(stderr, "open_qwen36: layer %d moe pass: %zu of %zu visits on %s, %.2f ms\n", l, n, visits.size(),
                         kname->c_str(), run_ms);
        auto t2 = std::chrono::steady_clock::now();
        read_back(yb, n * hid * NT * 4, 0);
        const float* yh = yb.map<float*>();
        // y[slot] comes back as C tiles: per 64-row band, [4 groups][even / odd rows][NG mmul
        // sub-tiles][8 tokens][8]; row 64 band + 16 g + 2 jj + p, token 8 sub + tt. A column
        // belongs to exactly one token, so the un-interleave is the scatter -- read the tiles
        // straight into the token's row rather than staging 20 MB a layer and reading it back.
        const size_t NG = NT / 8;                      // mmul sub-tiles a slot holds
#pragma omp parallel for
        for (long long t = 0; t < static_cast<long long>(t_real); ++t) {
            float* o = out + t * hid;
            for (const auto& [col, wt] : per_token[t]) {
                const size_t tt = col % NT;
                const float* base = yh + (col / NT) * hid * NT + (tt / 8) * 64 + (tt % 8) * 8;
                for (size_t band = 0; band < hid / 64; ++band)
                    for (size_t g = 0; g < 4; ++g) {
                        const float* even = base + ((band * 4 + g) * 2 * NG) * 64;
                        const float* odd = even + NG * 64;
                        float* d = o + band * 64 + g * 16;
                        for (size_t jj = 0; jj < 8; ++jj) {
                            d[2 * jj] += wt * even[jj];
                            d[2 * jj + 1] += wt * odd[jj];
                        }
                    }
            }
        }
        timing_.moe_read_ms += ms_since(t2);
        done += n;
    }
    timing_.route_ms += ms_since(tp);
}

void Core::seek(int pos) {
    if (pos < 0 || static_cast<size_t>(pos) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: seek out of range");
    pos_ = pos;
}

Snapshot Core::checkpoint() const {
    Snapshot s;
    s.pos = pos_;
    s.mrope_pos = mrope_pos_;
    s.mrope_on = mrope_on_;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        xrt::bo& bo = const_cast<xrt::bo&>(state_[l]);
        if (lt.state_kind == "kv") {
            size_t n = static_cast<size_t>(pos_) * lt.state_row;
            std::vector<uint8_t> rows(n);
            if (n) {
                read_back(bo, n, 0);
                std::memcpy(rows.data(), bo.map<uint8_t*>(), n);
            }
            s.kv.push_back(std::move(rows));
        } else {
            std::vector<uint8_t> st(lt.state_bytes);
            read_back(bo, lt.state_bytes, 0);
            std::memcpy(st.data(), bo.map<uint8_t*>(), lt.state_bytes);
            s.states.push_back(std::move(st));
        }
    }
    return s;
}

void Core::restore(const Snapshot& s) {
    size_t il = 0, ia = 0;
    for (int l = 0; l < nl_; ++l) {
        const LayerType& lt = *types_[l];
        if (lt.state_kind == "kv") {
            const auto& rows = s.kv.at(ia++);
            if (!rows.empty()) {
                std::memcpy(state_[l].map<uint8_t*>(), rows.data(), rows.size());
                state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, rows.size(), 0);
            }
        } else {
            const auto& st = s.states.at(il++);
            if (st.size() != lt.state_bytes) throw std::runtime_error("open_qwen36: snapshot state size mismatch");
            std::memcpy(state_[l].map<uint8_t*>(), st.data(), lt.state_bytes);
            state_[l].sync(XCL_BO_SYNC_BO_TO_DEVICE, lt.state_bytes, 0);
        }
    }
    pos_ = s.pos;
    mrope_pos_ = s.mrope_pos;
    mrope_on_ = s.mrope_on;
}

void Core::kv_row(int layer, int row, bool value, uint16_t* out) {
    if (layer < 0 || layer >= nl_ || !is_attention_layer(layer)) throw std::runtime_error("open_qwen36: layer " + std::to_string(layer) + " has no KV cache");
    if (row < 0 || static_cast<size_t>(row) >= cfg_.max_ctx) throw std::runtime_error("open_qwen36: KV row out of range");
    const size_t kv_row = types_[layer]->state_row;
    size_t off = static_cast<size_t>(row) * kv_row + (value ? kv_row / 2 : 0);
    read_back(state_[layer], kv_row / 2, off);
    std::memcpy(out, state_[layer].map<uint8_t*>() + off, kv_row / 2);
}

void Core::read_act(int layer, size_t off, size_t n, uint8_t* dst) {
    if (layer < 0 || layer >= nl_) throw std::runtime_error("open_qwen36: read_act: layer " + std::to_string(layer) + " out of range");
    const size_t bytes = types_[layer]->act_bytes;
    if (off + n > bytes)
        throw std::runtime_error("open_qwen36: read_act: [" + std::to_string(off) + ", " + std::to_string(off + n) +
                                 ") is outside the layer's " + std::to_string(bytes) + "-byte act buffer");
    read_back(act_[layer], n, off);
    std::memcpy(dst, act_[layer].map<uint8_t*>() + off, n);
}

}  // namespace open_qwen36
