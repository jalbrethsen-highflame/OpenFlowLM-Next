#pragma once
// mbi: the routed experts' band step on the int8 x int4 MAC mode (aie::mmul<4, 16, 16, int8, int4>,
// 1024 MACs an instruction: 23.4 TMAC/s over the array against bfp16's 16, peak.py). The weights
// are the q4_1 pool's bytes as they are: a chunk's nibbles for one 16-row group are k-major with
// 16 rows per k, which is the mmul's B tile (16 k x 16 n, row-major) for every 16 k, so a B tile
// is one 128-byte load and one XOR -- no dequantisation (the bfp16 step spent most of its time
// there). XOR 0x88 turns each unsigned nibble q into the signed q - 8, so
//
//   w x = (d q + m) x = d (q - 8) x + (m + 8 d) x
//
// and with x = sx xq (int8, one scale per token and 256-K group) a 32-K block of a row gives
//   d sx sum((q - 8) xq)  +  (m + 8 d) sum(x)
// The int sum fits int16 exactly (|sum| <= 8 127 32). Per (16 rows, 4 tokens) tile and block:
// two int4 MACs, the sum as bf16 times d's tile into a float accumulator, (m + 8 d) times the
// block's x sums (precomputed by the host as 64-lane tiles) into another; sx once per group.
//
// Layouts, per 64-row band k-group (two q4_1 chunks: rows 0-31, 32-63) and MBI_NT tokens:
//   xa   int8 [16 k-tiles][NTG token groups][4 tokens][16 k]
//   xs   bf16 [8 blocks][NTG][64]: lane t 16 + n = the block's sum of token t's x (any n)
//   sx   bf16 [NTG][64]: lane t 16 + n = token t's scale for this k-group
//   acc  f32  [4 row groups][NTG][64]: lane t 16 + n = token t, row 16 g + n (accumulated)
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef MBI_NT
#define MBI_NT 16
#endif
static constexpr unsigned MBI_NTG = MBI_NT / 4;
static constexpr unsigned MBI_CHUNK = 5120;

static inline aie::vector<bfloat16, 64> mbi_tile16(aie::vector<bfloat16, 16> v) {
  const aie::vector<bfloat16, 32> v2 = aie::concat(v, v);
  return aie::concat(v2, v2);
}

// acc += the band k-group's product with xa (scaled), over all MBI_NT tokens
static inline void mbi_band_kg(const uint8_t *__restrict band, const int8_t *__restrict xa,
                               const bfloat16 *__restrict xs, const bfloat16 *__restrict sx,
                               float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  alignas(64) static bfloat16 dt[8][64], mt[8][64];
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict ch = band + (rg >> 1) * MBI_CHUNK;
    const unsigned rb = rg & 1;
    const uint8_t *__restrict nib = ch + 1024 + rb * 2048;
    const bfloat16 *__restrict dp = reinterpret_cast<const bfloat16 *>(ch) + rb * 16;
    const bfloat16 *__restrict mp = reinterpret_cast<const bfloat16 *>(ch + 512) + rb * 16;
    // the group's d and (m + 8 d) tiles for the 8 blocks, once for every token group
#ifndef MBI_NO_PREP
    for (unsigned blk = 0; blk < 8; ++blk) {
      const aie::vector<bfloat16, 16> d16 = aie::load_v<16>(dp + blk * 32);
      const aie::vector<bfloat16, 16> m16 = aie::load_v<16>(mp + blk * 32);
      aie::accum<accfloat, 16> mc;
      mc.from_vector(m16);
      mc = aie::mac(mc, d16, (bfloat16)8.0f);
      aie::store_v(dt[blk], mbi_tile16(d16));
      aie::store_v(mt[blk], mbi_tile16(mc.template to_vector<bfloat16>()));
    }
#endif
    for (unsigned tg = 0; tg < MBI_NTG; ++tg) {
      aie::accum<accfloat, 64> ak = aie::zeros<accfloat, 64>(), am = aie::zeros<accfloat, 64>();
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<uint8, 128> r0 = aie::load_v<128>(nib + blk * 256);
        const aie::vector<uint8, 128> r1 = aie::load_v<128>(nib + blk * 256 + 128);
        const aie::vector<int4, 256> b0 = aie::bit_xor((uint8)0x88, r0).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::bit_xor((uint8)0x88, r1).template cast_to<int4>();
        const aie::vector<int8, 64> a0 = aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64);
        const aie::vector<int8, 64> a1 = aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64);
#ifndef MBI_NO_MM
        MM m;
        m.mul(a0, b0);
        m.mac(a1, b1);
#endif
#ifndef MBI_NO_EPI
#ifdef MBI_NO_MM
        const aie::vector<bfloat16, 64> fs = aie::load_v<64>(dt[blk]);
#else
        const aie::vector<bfloat16, 64> fs = aie::to_float<bfloat16>(m.template to_vector<int16>(0), 0);
#endif
        ak = aie::mac(ak, fs, aie::load_v<64>(dt[blk]));
        am = aie::mac(am, aie::load_v<64>(mt[blk]), aie::load_v<64>(xs + (blk * MBI_NTG + tg) * 64));
#elif !defined(MBI_NO_MM)
        if (blk == 7) ak = aie::mac(ak, aie::to_float<bfloat16>(m.template to_vector<int16>(0), 0), aie::load_v<64>(dt[0]));
#endif
      }
      float *__restrict out = acc + (rg * MBI_NTG + tg) * 64;
      aie::accum<accfloat, 64> t;
      t.from_vector(aie::load_v<64>(out));
      t = aie::mac(t, ak.template to_vector<bfloat16>(), aie::load_v<64>(sx + tg * 64));
      t = aie::add(t, am.template to_vector<float>());
      aie::store_v(out, t.template to_vector<float>());
    }
  }
}


// MBI_INT probe: the same products with the epilogue in integers -- the block's int16 sum times
// int16 d / (m + 8 d) tiles (host-prepared, replicated over the 4 tokens) into acc32, converted to
// float once per (row group, token group) and k-group. Timing first; the scales are not applied.
static inline void mbi_band_kg_int(const uint8_t *__restrict band, const int8_t *__restrict xa,
                                   const int16 *__restrict dti, const int16 *__restrict mti,
                                   const int16 *__restrict xsi, float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict nib = band + (rg >> 1) * MBI_CHUNK + 1024 + (rg & 1) * 2048;
    for (unsigned tg = 0; tg < MBI_NTG; ++tg) {
      aie::accum<acc32, 64> ad = aie::zeros<acc32, 64>(), am = aie::zeros<acc32, 64>();
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<uint8, 128> r0 = aie::load_v<128>(nib + blk * 256);
        const aie::vector<uint8, 128> r1 = aie::load_v<128>(nib + blk * 256 + 128);
        const aie::vector<int4, 256> b0 = aie::bit_xor((uint8)0x88, r0).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::bit_xor((uint8)0x88, r1).template cast_to<int4>();
        const aie::vector<int8, 64> a0 = aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64);
        const aie::vector<int8, 64> a1 = aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64);
        MM m;
        m.mul(a0, b0);
        m.mac(a1, b1);
        const aie::vector<int16, 64> si = m.template to_vector<int16>(0);
        ad = aie::mac(ad, si, aie::load_v<64>(dti + (rg * 8 + blk) * 64));
#ifndef MBI_NO_M
        am = aie::mac(am, aie::load_v<64>(mti + (rg * 8 + blk) * 64), aie::load_v<64>(xsi + (blk * MBI_NTG + tg) * 64));
#endif
      }
      float *__restrict out = acc + (rg * MBI_NTG + tg) * 64;
      const aie::vector<float, 64> fd = ad.template to_vector<float>(0), fm = am.template to_vector<float>(0);
      aie::store_v(out, aie::add(aie::load_v<64>(out), aie::add(fd, fm)));
    }
  }
}


// MBI_INT2: two token groups a pass (their chains interleaved), no m term in the loop
static inline void mbi_band_kg_int2(const uint8_t *__restrict band, const int8_t *__restrict xa,
                                    const int16 *__restrict dti, float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict nib = band + (rg >> 1) * MBI_CHUNK + 1024 + (rg & 1) * 2048;
    for (unsigned tg = 0; tg < MBI_NTG; tg += 2) {
      aie::accum<acc32, 64> a0d = aie::zeros<acc32, 64>(), a1d = aie::zeros<acc32, 64>();
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<int4, 256> b0 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256)).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256 + 128)).template cast_to<int4>();
        const aie::vector<int16, 64> dt = aie::load_v<64>(dti + (rg * 8 + blk) * 64);
        MM m0, m1;
        m0.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64), b0);
        m1.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg + 1) * 64), b0);
        m0.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64), b1);
        m1.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg + 1) * 64), b1);
        a0d = aie::mac(a0d, m0.template to_vector<int16>(0), dt);
        a1d = aie::mac(a1d, m1.template to_vector<int16>(0), dt);
      }
      float *__restrict o0 = acc + (rg * MBI_NTG + tg) * 64, *__restrict o1 = o0 + 64;
      aie::store_v(o0, aie::add(aie::load_v<64>(o0), a0d.template to_vector<float>(0)));
      aie::store_v(o1, aie::add(aie::load_v<64>(o1), a1d.template to_vector<float>(0)));
    }
  }
}


// MBI_INT4: all four token groups a pass (NT 16), no m term in the loop
static inline void mbi_band_kg_int4(const uint8_t *__restrict band, const int8_t *__restrict xa,
                                    const int16 *__restrict dti, float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict nib = band + (rg >> 1) * MBI_CHUNK + 1024 + (rg & 1) * 2048;
    aie::accum<acc32, 64> ad[4];
    for (unsigned u = 0; u < 4; ++u) ad[u] = aie::zeros<acc32, 64>();
    for (unsigned blk = 0; blk < 8; ++blk) {
      const aie::vector<int4, 256> b0 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256)).template cast_to<int4>();
      const aie::vector<int4, 256> b1 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256 + 128)).template cast_to<int4>();
      const aie::vector<int16, 64> dt = aie::load_v<64>(dti + (rg * 8 + blk) * 64);
      MM m[4];
      for (unsigned u = 0; u < 4; ++u) m[u].mul(aie::load_v<64>(xa + ((2 * blk) * 4 + u) * 64), b0);
      for (unsigned u = 0; u < 4; ++u) m[u].mac(aie::load_v<64>(xa + ((2 * blk + 1) * 4 + u) * 64), b1);
      for (unsigned u = 0; u < 4; ++u) ad[u] = aie::mac(ad[u], m[u].template to_vector<int16>(0), dt);
    }
    for (unsigned u = 0; u < 4; ++u) {
      float *__restrict o = acc + (rg * 4 + u) * 64;
      aie::store_v(o, aie::add(aie::load_v<64>(o), ad[u].template to_vector<float>(0)));
    }
  }
}


// MBI_INT2M: MBI_INT2 with the m term -- inline (two more acc32 chains) or, MBI_MSEP, as its own pass
static inline void mbi_band_kg_int2m(const uint8_t *__restrict band, const int8_t *__restrict xa,
                                     const int16 *__restrict dti, const int16 *__restrict mti,
                                     const int16 *__restrict xsi, float *__restrict acc) {
  using MM = aie::mmul<4, 16, 16, int8, int4>;
  for (unsigned rg = 0; rg < 4; ++rg) {
    const uint8_t *__restrict nib = band + (rg >> 1) * MBI_CHUNK + 1024 + (rg & 1) * 2048;
    for (unsigned tg = 0; tg < MBI_NTG; tg += 2) {
      aie::accum<acc32, 64> a0d = aie::zeros<acc32, 64>(), a1d = aie::zeros<acc32, 64>();
#ifndef MBI_MSEP
      aie::accum<acc32, 64> a0m = aie::zeros<acc32, 64>(), a1m = aie::zeros<acc32, 64>();
#endif
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<int4, 256> b0 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256)).template cast_to<int4>();
        const aie::vector<int4, 256> b1 = aie::bit_xor((uint8)0x88, aie::load_v<128>(nib + blk * 256 + 128)).template cast_to<int4>();
        const aie::vector<int16, 64> dt = aie::load_v<64>(dti + (rg * 8 + blk) * 64);
        MM m0, m1;
        m0.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg) * 64), b0);
        m1.mul(aie::load_v<64>(xa + ((2 * blk) * MBI_NTG + tg + 1) * 64), b0);
        m0.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg) * 64), b1);
        m1.mac(aie::load_v<64>(xa + ((2 * blk + 1) * MBI_NTG + tg + 1) * 64), b1);
        a0d = aie::mac(a0d, m0.template to_vector<int16>(0), dt);
        a1d = aie::mac(a1d, m1.template to_vector<int16>(0), dt);
#ifndef MBI_MSEP
        const aie::vector<int16, 64> mt = aie::load_v<64>(mti + (rg * 8 + blk) * 64);
        a0m = aie::mac(a0m, mt, aie::load_v<64>(xsi + (blk * MBI_NTG + tg) * 64));
        a1m = aie::mac(a1m, mt, aie::load_v<64>(xsi + (blk * MBI_NTG + tg + 1) * 64));
#endif
      }
      float *__restrict o0 = acc + (rg * MBI_NTG + tg) * 64, *__restrict o1 = o0 + 64;
#ifndef MBI_MSEP
      a0d = aie::add(a0d, a0m);   // (the probe's scales are 1)
      a1d = aie::add(a1d, a1m);
#endif
      aie::store_v(o0, aie::add(aie::load_v<64>(o0), a0d.template to_vector<float>(0)));
      aie::store_v(o1, aie::add(aie::load_v<64>(o1), a1d.template to_vector<float>(0)));
    }
#ifdef MBI_MSEP
    for (unsigned tg = 0; tg < MBI_NTG; tg += 2) {
      aie::accum<acc32, 64> a0m = aie::zeros<acc32, 64>(), a1m = aie::zeros<acc32, 64>();
      for (unsigned blk = 0; blk < 8; ++blk) {
        const aie::vector<int16, 64> mt = aie::load_v<64>(mti + (rg * 8 + blk) * 64);
        a0m = aie::mac(a0m, mt, aie::load_v<64>(xsi + (blk * MBI_NTG + tg) * 64));
        a1m = aie::mac(a1m, mt, aie::load_v<64>(xsi + (blk * MBI_NTG + tg + 1) * 64));
      }
      float *__restrict o0 = acc + (rg * MBI_NTG + tg) * 64, *__restrict o1 = o0 + 64;
      aie::store_v(o0, aie::add(aie::load_v<64>(o0), a0m.template to_vector<float>(0)));
      aie::store_v(o1, aie::add(aie::load_v<64>(o1), a1m.template to_vector<float>(0)));
    }
#endif
  }
}
