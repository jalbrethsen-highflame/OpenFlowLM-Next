#pragma once
// mbi's core step: one 64-row band k-group of a routed expert (two q4_1 chunks as the pool packs
// them: rows 0-31, 32-63; 256 K) times MBI_NT tokens of int8 x, on the int8 x int4 MAC mode.
//
// Per chunk: w = d q + m = d (q - 8) + mc with mc = m + 8 d. d and mc become int16 with one
// power-of-two shift for the chunk (bfloat16_to_int; the largest of |d| and |mc| / 8 at 2^12), so a
// 32-K block of a row and a token gives, in integers,
//   sum_k (q - 8) xq * d_i  +  qsum * mc_i            (qsum: the block's sum of xq)
// accumulated over the k-group's 8 blocks in acc32 (|each| < 2^30), converted to float once with
// the chunk's shift and multiplied by the token's x scale. XOR 0x88 makes q - 8 out of the nibble.
// Two token groups a pass (their chains interleaved): the epilogue is latency-bound, and four spill.
//
// The x element for the k-group (host-packed):
//   xa  int8 [16 k-tiles][NTG][4 tokens][16 k]                         16 NTG 64 B
//   qs  int16 [8 blocks][NTG][64]: lane 16 t + n = qsum of token t     8 NTG 128 B
//   sx  bf16  [NTG][64]: lane 16 t + n = token t's scale (bf16: the host quantizes x with it)  NTG 128 B
// acc f32 [4 row groups][NTG][64], lane 16 t + n = token t, row 16 g + n (accumulated).
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MBI_NT
#define MBI_NT 16
#endif
static constexpr unsigned MBI_NTG = MBI_NT / 4;
static constexpr unsigned MBI_CHUNK = 5120;
static constexpr unsigned MBI_XA = 16 * MBI_NTG * 64, MBI_QS = 8 * MBI_NTG * 128, MBI_SX = MBI_NTG * 128;
static constexpr unsigned MBI_XEL = MBI_XA + MBI_QS + MBI_SX;

// a chunk's d and mc = m + 8 d as int16 [8 blocks][32 rows] at 2^shift; returns shift
static inline int mbi_prep_chunk(const uint8_t *__restrict ch, int16 *__restrict di, int16 *__restrict mi) {
  const bfloat16 *__restrict dp = reinterpret_cast<const bfloat16 *>(ch);
  const bfloat16 *__restrict mp = reinterpret_cast<const bfloat16 *>(ch + 512);
  alignas(64) static bfloat16 mcb[256];
  aie::vector<bfloat16, 32> amax = aie::zeros<bfloat16, 32>();
  for (unsigned i = 0; i < 256; i += 32) {
    const aie::vector<bfloat16, 32> d = aie::load_v<32>(dp + i), m = aie::load_v<32>(mp + i);
    aie::accum<accfloat, 32> mc;
    mc.from_vector(m);
    mc = aie::mac(mc, d, (bfloat16)8.0f);
    const aie::vector<bfloat16, 32> mcv = mc.template to_vector<bfloat16>();
    aie::store_v(mcb + i, mcv);
    amax = aie::max(amax, aie::abs(d));
    amax = aie::max(amax, aie::mul(aie::abs(mcv), (bfloat16)0.125f).template to_vector<bfloat16>());
  }
  const float s = (float)aie::reduce_max(amax);
  int e = 0;                                   // s < 2^e
  if (s > 0.f) {
    float t = s;
    while (t >= 1.f) { t *= 0.5f; ++e; }
    while (t < 0.5f) { t *= 2.f; --e; }
  }
  const int shift = 12 - e;                    // |d|, |mc| / 8 < 2^12 at this shift
  for (unsigned i = 0; i < 256; i += 32) {
    aie::store_v(di + i, aie::to_fixed<int16>(aie::load_v<32>(dp + i), shift));
    aie::store_v(mi + i, aie::to_fixed<int16>(aie::load_v<32>(mcb + i), shift));
  }
  return shift;
}

static inline float mbi_pow2f(int e) {
  float f = 1.f;
  for (; e > 0; --e) f *= 2.f;
  for (; e < 0; ++e) f *= 0.5f;
  return f;
}

static inline aie::vector<int16, 64> mbi_rep4(aie::vector<int16, 16> v) {
  const aie::vector<int16, 32> v2 = aie::concat(v, v);
  return aie::concat(v2, v2);
}

// acc += band k-group x the x element (all MBI_NT tokens)
static inline void mbi_band_kg(const uint8_t *__restrict band, const uint8_t *__restrict xel, float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  const int8_t *__restrict xa = reinterpret_cast<const int8_t *>(xel);
  const int16 *__restrict qs = reinterpret_cast<const int16 *>(xel + MBI_XA);
  const float *__restrict sx = reinterpret_cast<const float *>(xel + MBI_XA + MBI_QS);
  alignas(64) static int16 di[2][256], mi[2][256];
  int shift[2];
#ifdef MBI_NO_PREP
  shift[0] = shift[1] = 16;
#else
  shift[0] = mbi_prep_chunk(band, di[0], mi[0]);
  shift[1] = mbi_prep_chunk(band + MBI_CHUNK, di[1], mi[1]);
#endif
  for (unsigned rg = 0; rg < 4; ++rg) {
    const unsigned c = rg >> 1, rb = rg & 1;
    const uint8_t *__restrict nib = band + c * MBI_CHUNK + 1024 + rb * 2048;
    for (unsigned tg = 0; tg < MBI_NTG; tg += 2) {
      aie::accum<acc32, 64> a0 = aie::zeros<acc32, 64>(), a1 = aie::zeros<acc32, 64>();
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<int4, 256> b0 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256)).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256 + 128)).template cast_to<int4>();
        const aie::vector<int16, 64> dt = mbi_rep4(aie::load_v<16>(di[c] + blk * 32 + rb * 16));
        const aie::vector<int16, 64> mt = mbi_rep4(aie::load_v<16>(mi[c] + blk * 32 + rb * 16));
        MM m0, m1;
        m0.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64), b0);
        m1.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg + 1) * 64), b0);
        m0.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64), b1);
        m1.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg + 1) * 64), b1);
        a0 = aie::mac(a0, m0.template to_vector<int16>(0), dt);
        a1 = aie::mac(a1, m1.template to_vector<int16>(0), dt);
        a0 = aie::mac(a0, mt, aie::load_v<64>(qs + (blk * MBI_NTG + tg) * 64));
        a1 = aie::mac(a1, mt, aie::load_v<64>(qs + (blk * MBI_NTG + tg + 1) * 64));
      }
      float *__restrict o0 = acc + (rg * MBI_NTG + tg) * 64, *__restrict o1 = o0 + 64;
#ifdef MBI_NO_CONV
      aie::store_v(reinterpret_cast<int32 *>(o0), a0.template to_vector<int32>(0));
      aie::store_v(reinterpret_cast<int32 *>(o1), a1.template to_vector<int32>(0));
      continue;
#endif
#if !defined(MBI_CONV_A) && !defined(MBI_CONV_C)
      // int32 lanes, then the vector fix-to-float with the shift (the accumulator's own
      // to_vector<float>(shift) gave NaN in half the lanes here)
      const aie::vector<float, 64> f0 = aie::to_float<float>(a0.template to_vector<int32>(0), shift[c]);
      const aie::vector<float, 64> f1 = aie::to_float<float>(a1.template to_vector<int32>(0), shift[c]);
#elif defined(MBI_CONV_C)
      // no shift in the conversion, the 2^-shift as a float multiply
      aie::accum<accfloat, 64> c0 = aie::mul(a0.template to_vector<float>(0), mbi_pow2f(-shift[c]));
      aie::accum<accfloat, 64> c1 = aie::mul(a1.template to_vector<float>(0), mbi_pow2f(-shift[c]));
      const aie::vector<float, 64> f0 = c0.template to_vector<float>(), f1 = c1.template to_vector<float>();
#else
      const aie::vector<float, 64> f0 = a0.template to_vector<float>(shift[c]), f1 = a1.template to_vector<float>(shift[c]);
#endif
#if defined(MBI_DBG_RAW)
      aie::store_v(reinterpret_cast<int32 *>(o0), a0.template to_vector<int32>(0));
      aie::store_v(reinterpret_cast<int32 *>(o1), a1.template to_vector<int32>(0));
#elif defined(MBI_DBG_NOSX)
      aie::store_v(o0, f0);
      aie::store_v(o1, f1);
#else
      aie::accum<accfloat, 64> t0, t1;
      t0.from_vector(aie::load_v<64>(o0));
      t1.from_vector(aie::load_v<64>(o1));
      t0 = aie::mac(t0, f0, aie::load_v<64>(sx + tg * 64));
      t1 = aie::mac(t1, f1, aie::load_v<64>(sx + (tg + 1) * 64));
      aie::store_v(o0, t0.template to_vector<float>());
      aie::store_v(o1, t1.template to_vector<float>());
#endif
    }
  }
}


// ---- v3: the pool's experts transformed at load (Core: OFLM_OPEN_MBI): every nibble byte XOR 0x88
// (signed q - 8), each chunk's d and m replaced by int16 d_i = d 2^S and mc_i = (m + 8 d) 2^S with
// one shift S per expert matrix (exact for the bf16 d of these experts: 95 % of their blocks hold
// d >= 0.15 of the matrix's largest), so the step has no prep and no XOR. S rides in the x element's
// 64-byte header (int32 at 0); the x element is then [header | xa | qs | sx].
// o += float(a) 2^-shift x s, without an fp32 multiply (emulated on this core: ~250 cycles a
// 64-lane pair): the fp32 sum split into bf16 high + low parts, each a native bf16 MAC with s
// (bf16, exact: the scales are chosen as bf16). The sum keeps ~16 bits of mantissa.
static inline aie::vector<bfloat16, 64> mbi_bf(const aie::vector<float, 64> &f) {
  aie::accum<accfloat, 64> a;
  a.from_vector(f);
  return a.template to_vector<bfloat16>();
}
static inline void mbi_epi(float *__restrict o, const aie::accum<acc32, 64> &a, int shift, const aie::vector<bfloat16, 64> &s) {
  const aie::vector<float, 64> f = aie::to_float<float>(a.template to_vector<int32>(0), shift);
  const aie::vector<bfloat16, 64> fh = mbi_bf(f);
  aie::accum<accfloat, 64> fa;
  fa.from_vector(fh);
  const aie::vector<float, 64> fl = aie::sub(f, fa.template to_vector<float>());
  aie::accum<accfloat, 64> t;
  t.from_vector(aie::load_v<64>(o));
  t = aie::mac(t, fh, s);
  t = aie::mac(t, mbi_bf(fl), s);
  aie::store_v(o, t.template to_vector<float>());
}

static constexpr unsigned MBI_XHDR = 64;
// MBI_QS_COMPACT: the block sums once, qsc int16 [NTG][2 block halves][4 tokens][4 blocks] (256 B,
// not 4 KB), and the (m + 8 d) qsum term as dense int16 matrix products ([4 tokens x 4 blocks] x
// [4 blocks x 8 rows], both halves, rows 0-7 and 8-15 zipped to the [token][16 rows] lanes) added
// once after the blocks: the same integer sums, so the same results
#ifdef MBI_QS_COMPACT
static constexpr unsigned MBI_QS3 = MBI_NTG * 4 * 8 * 2;
#else
static constexpr unsigned MBI_QS3 = MBI_QS;
#endif
static constexpr unsigned MBI_XEL3 = MBI_XHDR + MBI_XA + MBI_QS3 + MBI_SX;
// The x header's int32 2 is the row's count of real tokens (0..MBI_NT; a slot's last row is often
// short, its later rows empty). The steps skip token-group pairs past it; their sums stay zero.
static inline unsigned mbi_ntg(int nv) { return ((unsigned)(nv + 7) >> 3) * 2; }
static inline void mbi_band_kg3(const uint8_t *__restrict band, const uint8_t *__restrict xel, float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  const int shift = *reinterpret_cast<const int32_t *>(xel);
  const unsigned ntg = mbi_ntg(reinterpret_cast<const int32_t *>(xel)[2]);
  const int8_t *__restrict xa = reinterpret_cast<const int8_t *>(xel + MBI_XHDR);
  const int16 *__restrict qs = reinterpret_cast<const int16 *>(xel + MBI_XHDR + MBI_XA);
  const bfloat16 *__restrict sxb = reinterpret_cast<const bfloat16 *>(xel + MBI_XHDR + MBI_XA + MBI_QS3);
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict ch = band + (rg >> 1) * MBI_CHUNK;
    const unsigned rb = rg & 1;
    const uint8_t *__restrict nib = ch + 1024 + rb * 2048;
    const int16 *__restrict di = reinterpret_cast<const int16 *>(ch) + rb * 16;
    const int16 *__restrict mi = reinterpret_cast<const int16 *>(ch + 512) + rb * 16;
#ifdef MBI_QS_COMPACT
    // mc for rows 0-7 and 8-15 of the group as [8 blocks][8 rows] B tiles
    aie::vector<int16, 32> mb[2][2];                     // [rows 0-7 / 8-15][blocks 0-3 / 4-7]: [4 blocks][8 rows]
    for (unsigned blk = 0; blk < 8; ++blk) {
      mb[0][blk >> 2].insert(blk & 3, aie::load_v<8>(mi + blk * 32));
      mb[1][blk >> 2].insert(blk & 3, aie::load_v<8>(mi + blk * 32 + 8));
    }
#endif
    for (unsigned tg = 0; tg < ntg; tg += 2) {
      aie::accum<acc32, 64> a0 = aie::zeros<acc32, 64>(), a1 = aie::zeros<acc32, 64>();
#ifndef MBI_UNROLL_N                               // the blocks unrolled by 4: 2.5 -> 1.5 us a call (by 8: 1.9,
#define MBI_UNROLL_N 4                             // and a 7 KB stack frame)
#endif
#ifndef MBI_NO_UNROLL
#pragma unroll MBI_UNROLL_N
#endif
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<int4, 256> b0 = aie::load_v<128>(nib + blk * 256).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::load_v<128>(nib + blk * 256 + 128).template cast_to<int4>();
        const aie::vector<int16, 64> dt = mbi_rep4(aie::load_v<16>(di + blk * 32));
        const aie::vector<int16, 64> mt = mbi_rep4(aie::load_v<16>(mi + blk * 32));
        MM m0, m1;
        m0.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64), b0);
        m1.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg + 1) * 64), b0);
        m0.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64), b1);
        m1.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg + 1) * 64), b1);
        a0 = aie::mac(a0, m0.template to_vector<int16>(0), dt);
        a1 = aie::mac(a1, m1.template to_vector<int16>(0), dt);
#ifndef MBI_QS_COMPACT
        a0 = aie::mac(a0, mt, aie::load_v<64>(qs + (blk * MBI_NTG + tg) * 64));
        a1 = aie::mac(a1, mt, aie::load_v<64>(qs + (blk * MBI_NTG + tg + 1) * 64));
#else
        (void)mt;
#endif
      }
#ifdef MBI_QS_COMPACT
      {
        using MQ = aie::mmul<4, 4, 8, int16, int16>;
        MQ p0, q0, p1, q1;
        const aie::vector<int16, 16> s0l = aie::load_v<16>(qs + tg * 32), s0h = aie::load_v<16>(qs + tg * 32 + 16);
        const aie::vector<int16, 16> s1l = aie::load_v<16>(qs + (tg + 1) * 32), s1h = aie::load_v<16>(qs + (tg + 1) * 32 + 16);
        p0.mul(s0l, mb[0][0]); p0.mac(s0h, mb[0][1]);
        q0.mul(s0l, mb[1][0]); q0.mac(s0h, mb[1][1]);
        p1.mul(s1l, mb[0][0]); p1.mac(s1h, mb[0][1]);
        q1.mul(s1l, mb[1][0]); q1.mac(s1h, mb[1][1]);
        const auto z0 = aie::interleave_zip(p0.template to_vector<int32>(), q0.template to_vector<int32>(), 8);
        const auto z1 = aie::interleave_zip(p1.template to_vector<int32>(), q1.template to_vector<int32>(), 8);
        a0 = aie::add(a0, aie::concat(z0.first, z0.second));
        a1 = aie::add(a1, aie::concat(z1.first, z1.second));
      }
#endif
      float *__restrict o0 = acc + (rg * MBI_NTG + tg) * 64;
      mbi_epi(o0, a0, shift, aie::load_v<64>(sxb + tg * 64));
      mbi_epi(o0 + 64, a1, shift, aie::load_v<64>(sxb + (tg + 1) * 64));
    }
  }
}


// ---- the rest of the token-split design (mbi.py): h between up / gate and down, the down, y --------
//
// hmeta (per core, L1): int32 shift_dn at 0 (from the x header), then sh bf16 [8 h-bands][MBI_NT]
// (each token's h scale per 64-row band) at 64, then hq_sum int16 [16 blocks][MBI_NT] (the sum
// of a token's quantized h over a 32-row block) at 64 + 32 MBI_NT.
// hq (per core, L1): the down's A tiles, int8 [2 k-groups][16 k-tiles][NTG][4 tokens][16 k].
#ifdef MBI_TIME
static inline uint32_t mbi_now() { return (uint32_t)__builtin_aie2p_read_tm((void *)0x340F8); }   // core module Timer_Low
struct MbiTime {
  uint32_t *w, t0; unsigned slot;
  MbiTime(uint8_t *hm, unsigned s) : w(reinterpret_cast<uint32_t *>(hm)), t0(mbi_now()), slot(s) {
    if (w[10] != 0x7111u) { for (unsigned i = 4; i < 10; ++i) w[i] = 0; w[10] = 0x7111u; w[8] = t0; }
  }
  ~MbiTime() { const uint32_t t1 = mbi_now(); w[slot] += t1 - t0; w[9] = t1; }
};
#define MBI_T(hm, s) MbiTime mbi_t_(const_cast<uint8_t *>(hm), s)
#else
#define MBI_T(hm, s)
#endif
// Then rel int16 [8 h-bands][MBI_NT] at 64 + 16 MBI_NT (a band's scale over its k-group's largest,
// 2^14 = 1) and smax bf16 [2 k-groups][MBI_NT] after qsum: the down scales d and mc by rel and
// converts its int sums once a k-group, at smax.
static constexpr unsigned MBI_HM_SH = 64, MBI_HM_REL = 64 + 8 * MBI_NT * 2, MBI_HM_QS = 64 + 8 * MBI_NT * 4;
static constexpr unsigned MBI_HM_SMAX = MBI_HM_QS + 16 * MBI_NT * 2;
static constexpr unsigned MBI_HMETA = MBI_HM_SMAX + 2 * MBI_NT * 2;
static constexpr unsigned MBI_HQ = 2 * 16 * MBI_NTG * 64;

#include "vecmath.h"

static inline aie::vector<float, 64> mbi_tok4f(const float *__restrict v) {   // lane 16 t + n = v[t]
  return aie::concat(aie::concat(aie::broadcast<float, 16>(v[0]), aie::broadcast<float, 16>(v[1])),
                     aie::concat(aie::broadcast<float, 16>(v[2]), aie::broadcast<float, 16>(v[3])));
}
static inline aie::vector<bfloat16, 64> mbi_tok4b(const bfloat16 *__restrict v) {
  return aie::concat(aie::concat(aie::broadcast<bfloat16, 16>(v[0]), aie::broadcast<bfloat16, 16>(v[1])),
                     aie::concat(aie::broadcast<bfloat16, 16>(v[2]), aie::broadcast<bfloat16, 16>(v[3])));
}
static inline aie::vector<int16, 64> mbi_tok4i(const int16 *__restrict v) {
  return aie::concat(aie::concat(aie::broadcast<int16, 16>(v[0]), aie::broadcast<int16, 16>(v[1])),
                     aie::concat(aie::broadcast<int16, 16>(v[2]), aie::broadcast<int16, 16>(v[3])));
}

// h band pb (64 rows) = silu(g) u from the up / gate accumulators, quantized per token into hq
// (k-group pb / 4, k-tiles 4 (pb % 4) + rg: a (row group, token group) tile is the A tile as is).
// silu in dit_swiglu_epi's bf16 form, h (1 + tanh h) with h = g / 2 (native bf16 MACs and tanh;
// vecmath's fp32 silu was ~34 cycles an element on the emulated fp32 multiply). h waits as bf16
// in au (no longer needed): four live 64-lane vectors spilled to the stack, and the stack ran
// into the next weight element.
static inline aie::vector<bfloat16, 32> mbi_silu_mul(const float *__restrict g, const float *__restrict u) {
  using VB = aie::vector<bfloat16, 32>;
#ifdef MBI_SILU_FP32
  aie::accum<accfloat, 32> ha;
  ha.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(g)), aie::load_v<32>(u)));
  return ha.template to_vector<bfloat16>();
#endif
  aie::accum<accfloat, 32> ga, ua;
  ga.from_vector(aie::load_v<32>(g));
  ua.from_vector(aie::load_v<32>(u));
  const VB h = aie::mul(ga.template to_vector<bfloat16>(), (bfloat16)0.5f).template to_vector<bfloat16>();
  aie::accum<accfloat, 32> ha;
  ha.from_vector(h);
  const aie::vector<float, 32> hf = ha.template to_vector<float>();
  VB th;
  th.insert(0, aie::tanh<bfloat16>(hf.template extract<16>(0)));
  th.insert(1, aie::tanh<bfloat16>(hf.template extract<16>(1)));
  const VB sg = aie::mac(ha, h, th).template to_vector<bfloat16>();
  return aie::mul(sg, ua.template to_vector<bfloat16>()).template to_vector<bfloat16>();
}
static inline void mbi_hq_band(float *__restrict au, const float *__restrict ag, int8_t *__restrict hq,
                               uint8_t *__restrict hmeta, unsigned pb) {
  bfloat16 *__restrict sh = reinterpret_cast<bfloat16 *>(hmeta + MBI_HM_SH) + pb * MBI_NT;
  int16 *__restrict qsum = reinterpret_cast<int16 *>(hmeta + MBI_HM_QS);
  bfloat16 *__restrict hb = reinterpret_cast<bfloat16 *>(au);     // h [rg][tg][64] bf16 over au's first half
  const unsigned kg = pb >> 2, kt0 = (pb & 3) * 4;
  const unsigned ntg = mbi_ntg(reinterpret_cast<const int32_t *>(hmeta)[3]);
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  // h for all (row group, token group) tiles first: au's tile (rg, tg) is read before hb writes it
  // (hb's tile i sits in the first half of au's tile i / 2)
  for (unsigned i = 0; i < 4 * MBI_NTG; ++i) {
    if (i % MBI_NTG >= ntg) continue;
    const float *__restrict u = au + i * 64;
    const float *__restrict g = ag + i * 64;
    const aie::vector<bfloat16, 32> h0 = mbi_silu_mul(g, u), h1 = mbi_silu_mul(g + 32, u + 32);
    aie::store_v(hb + i * 64, h0);
    aie::store_v(hb + i * 64 + 32, h1);
  }
  for (unsigned tg = 0; tg < ntg; ++tg) {
    aie::vector<bfloat16, 64> amax = aie::zeros<bfloat16, 64>();
    for (unsigned rg = 0; rg < 4; ++rg) amax = aie::max(amax, aie::abs(aie::load_v<64>(hb + (rg * MBI_NTG + tg) * 64)));
    bfloat16 inv[4];
    for (unsigned t = 0; t < 4; ++t) {
      const float m = (float)aie::reduce_max(amax.template extract<16>(t));
      const bfloat16 sb = (bfloat16)(m * (1.f / 126.f));   // bf16 rounds up by < 1/256: 126 keeps |q| <= 127
      sh[tg * 4 + t] = sb;
      inv[t] = (bfloat16)(m > 0.f ? 1.f / (float)sb : 0.f);
    }
    const aie::vector<bfloat16, 64> iv = mbi_tok4b(inv);
    for (unsigned b2 = 0; b2 < 2; ++b2) {          // the band's two 32-row blocks: row groups 2 b2, 2 b2 + 1
      aie::vector<int16, 64> qs16;
      for (unsigned j = 0; j < 2; ++j) {
        const unsigned rg = 2 * b2 + j;
        const aie::accum<accfloat, 64> a = aie::mul(aie::load_v<64>(hb + (rg * MBI_NTG + tg) * 64), iv);
        const aie::vector<int8, 64> q = aie::to_fixed<int8>(a.template to_vector<float>(), 0);
        aie::store_v(hq + ((kg * 16 + kt0 + rg) * MBI_NTG + tg) * 64, q);
        const aie::vector<int16, 64> q16 = q.template unpack<int16>();
        qs16 = j ? aie::add(qs16, q16) : q16;
      }
      for (unsigned t = 0; t < 4; ++t)
        qsum[(kg * 8 + (pb & 3) * 2 + b2) * MBI_NT + tg * 4 + t] = (int16)aie::reduce_add(qs16.template extract<16>(t));
    }
  }
  if ((pb & 3) == 3) {                             // the k-group's four band scales known: smax, rel
    static_assert(MBI_NT == 16, "one 16-lane vector of tokens");
    const bfloat16 *__restrict sk = reinterpret_cast<const bfloat16 *>(hmeta + MBI_HM_SH) + kg * 4 * MBI_NT;
    int16 *__restrict rel = reinterpret_cast<int16 *>(hmeta + MBI_HM_REL) + kg * 4 * MBI_NT;
    aie::vector<bfloat16, 16> mx = aie::load_v<16>(sk);
    for (unsigned b = 1; b < 4; ++b) mx = aie::max(mx, aie::load_v<16>(sk + b * MBI_NT));
    mx = aie::max(mx, aie::broadcast<bfloat16, 16>((bfloat16)1e-30f));
    aie::store_v(reinterpret_cast<bfloat16 *>(hmeta + MBI_HM_SMAX) + kg * MBI_NT, mx);
    aie::accum<accfloat, 16> ma;
    ma.from_vector(mx);
    const aie::vector<float, 16> r = vrecipN<16>(ma.template to_vector<float>());
    for (unsigned b = 0; b < 4; ++b) {
      aie::accum<accfloat, 16> sa;
      sa.from_vector(aie::load_v<16>(sk + b * MBI_NT));
      aie::store_v(rel + b * MBI_NT, aie::to_fixed<int16>(fmulN<16>(sa.template to_vector<float>(), r), 14));
    }
#ifdef MBI_HKG
    // MBI_HKG (opt-in: no faster when the DMA binds, rel 2.2 % against 1.7 %): one h scale a k-group: the four bands' int8 h requantized to smax (q rel, rel <= 1 in Q14;
    // the largest band's values stay as they are) and their block sums made again, so the down
    // has no per-band scale in its loop
    for (unsigned b = 0; b < 4; ++b)
      for (unsigned tg = 0; tg < ntg; ++tg) {
        const aie::vector<int16, 64> rv = mbi_tok4i(rel + b * MBI_NT + tg * 4);
        for (unsigned b2 = 0; b2 < 2; ++b2) {
          aie::vector<int16, 64> qs16;
          for (unsigned j = 0; j < 2; ++j) {
            int8_t *__restrict p = hq + ((kg * 16 + 4 * b + 2 * b2 + j) * MBI_NTG + tg) * 64;
            const aie::vector<int8, 64> q = aie::mul(aie::load_v<64>(p).template unpack<int16>(), rv).template to_vector<int8>(14);
            aie::store_v(p, q);
            const aie::vector<int16, 64> q16 = q.template unpack<int16>();
            qs16 = j ? aie::add(qs16, q16) : q16;
          }
          for (unsigned t = 0; t < 4; ++t)
            qsum[(kg * 8 + b * 2 + b2) * MBI_NT + tg * 4 + t] = (int16)aie::reduce_add(qs16.template extract<16>(t));
        }
      }
#endif
  }
  aie::set_rounding(saved);
}

// acc += the down band k-group (kg of 2) x hq. The h scale changes every 64 k (one h band): d and
// mc are scaled by the band's rel (int16, 2^14 = 1), so the int sums run over the whole k-group
// and convert once, at the k-group's smax (four conversions a k-group cost half the step).
static inline void mbi_dn_kg(const uint8_t *__restrict band, const int8_t *__restrict hq, const uint8_t *__restrict hmeta,
                             float *__restrict acc, unsigned kg) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  const int shift = *reinterpret_cast<const int32_t *>(hmeta);
  const unsigned ntg = mbi_ntg(reinterpret_cast<const int32_t *>(hmeta)[3]);
  const int16 *__restrict rel = reinterpret_cast<const int16 *>(hmeta + MBI_HM_REL) + kg * 4 * MBI_NT;
  (void)rel;
  const bfloat16 *__restrict smax = reinterpret_cast<const bfloat16 *>(hmeta + MBI_HM_SMAX) + kg * MBI_NT;
  const int16 *__restrict qsum = reinterpret_cast<const int16 *>(hmeta + MBI_HM_QS);
  const int8_t *__restrict xa = hq + kg * 16 * MBI_NTG * 64;
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict ch = band + (rg >> 1) * MBI_CHUNK;
    const unsigned rb = rg & 1;
    const uint8_t *__restrict nib = ch + 1024 + rb * 2048;
    const int16 *__restrict di = reinterpret_cast<const int16 *>(ch) + rb * 16;
    const int16 *__restrict mi = reinterpret_cast<const int16 *>(ch + 512) + rb * 16;
    for (unsigned tg = 0; tg < ntg; tg += 2) {
      aie::accum<acc32, 64> a0 = aie::zeros<acc32, 64>(), a1 = aie::zeros<acc32, 64>();
#ifdef MBI_DN_UNROLL
#pragma unroll MBI_DN_UNROLL
#endif
      for (unsigned blk = 0; blk < 8; ++blk) {
#ifndef MBI_HKG
        const unsigned bp = blk >> 1;
        const aie::vector<int16, 64> r0 = mbi_tok4i(rel + bp * MBI_NT + tg * 4);
        const aie::vector<int16, 64> r1 = mbi_tok4i(rel + bp * MBI_NT + (tg + 1) * 4);
#endif
        const aie::vector<int4, 256> b0 = aie::load_v<128>(nib + blk * 256).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::load_v<128>(nib + blk * 256 + 128).template cast_to<int4>();
        const aie::vector<int16, 64> dt = mbi_rep4(aie::load_v<16>(di + blk * 32));
        const aie::vector<int16, 64> mt = mbi_rep4(aie::load_v<16>(mi + blk * 32));
        MM m0, m1;
        m0.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64), b0);
        m1.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg + 1) * 64), b0);
        m0.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64), b1);
        m1.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg + 1) * 64), b1);
        const int16 *__restrict qb = qsum + (kg * 8 + blk) * MBI_NT;
#ifndef MBI_HKG
        a0 = aie::mac(a0, m0.template to_vector<int16>(0), aie::mul(dt, r0).template to_vector<int16>(14));
        a1 = aie::mac(a1, m1.template to_vector<int16>(0), aie::mul(dt, r1).template to_vector<int16>(14));
        a0 = aie::mac(a0, aie::mul(mt, r0).template to_vector<int16>(14), mbi_tok4i(qb + tg * 4));
        a1 = aie::mac(a1, aie::mul(mt, r1).template to_vector<int16>(14), mbi_tok4i(qb + (tg + 1) * 4));
#else
        a0 = aie::mac(a0, m0.template to_vector<int16>(0), dt);
        a1 = aie::mac(a1, m1.template to_vector<int16>(0), dt);
        a0 = aie::mac(a0, mt, mbi_tok4i(qb + tg * 4));
        a1 = aie::mac(a1, mt, mbi_tok4i(qb + (tg + 1) * 4));
#endif
      }
      float *__restrict o0 = acc + (rg * MBI_NTG + tg) * 64;
      mbi_epi(o0, a0, shift, mbi_tok4b(smax + tg * 4));
      mbi_epi(o0 + 64, a1, shift, mbi_tok4b(smax + (tg + 1) * 4));
    }
  }
  aie::set_rounding(saved);
}
