#pragma once
//===- dnc_core.h: the gated delta rule over a prompt, 16 tokens a step, on one core -*- C++ -*-===//
//
// One unit = one value head's quarter of the value dims (32 of 128). Its state slice S (dk 128 x
// dv 32, fp32) stays in L1 as S^T tiles [dv block 4][dk block 16][8 dv][8 dk] while the tokens
// stream through, 16 at a time. In the chunked form (G_t = a_0 .. a_t inside the step, r_ts =
// G_t / G_s for s <= t):
//
//   d_t = b_t (v_t - G_t S0^T k_t - sum_{s<t} r_ts (k_t.k_s) d_s)
//   o_t = G_t S0^T q_t + sum_{s<=t} r_ts (q_t.k_s) d_s
//   S   = G_15 S0 + sum_s r_{15,s} k_s d_s^T
//
// The products run on the native bfp16 8x8x8 MAC (mac_8x8_8x8T: C += A B^T, both tiles [rows][k]):
//   KK = K K^T, QK = Q K^T          A = K / Q tiles, B = K tiles
//   P = K S0, Qs = Q S0             A = K / Q tiles, B = S^T tiles
//   O = G * Qs + (R . QK) D         A = (R . QK) tiles, B = D^T tiles
//   S^T = G_15 S^T + D^T (w . K)    A = D^T tiles, B = (w . K)^T tiles (K tiles transposed here)
// fp32 x fp32 products (scaling by the decays, the solve) are bf16 hi / lo splits: three native
// bf16 MACs, no emulated fp32 multiply. Scalars come host-split (the record's aux), so the core
// runs no scalar float arithmetic.
//
// The record (one 16-token step of one value head; host-packed, 9216 B):
//   kA   bf16 [t block 2][dk block 16][8 t][8 dk]                       4096 B
//   qA   bf16 the same                                                  4096 B
//   aux  bf16 at 8192:
//     e hi/lo [2][16], f hi/lo [2][16]: r_ts = e_t f_s (e_t = G_t / G_m, f_s = G_m / G_s)
//     b hi/lo [2][16]: beta
//     g hi/lo [2][2 t block][64]: G_t replicated as the [8 t][8] tile rows need it
//     w [2 t block][64] bf16: w_s = r_{15,s} = G_15 / G_s, replicated as [8 dk][8 t] tiles need it
//     g15 hi/lo [2]
// v: bf16 [16 t][32 dv] (this core's quarter); o: bf16 [16 t][32 dv].
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace dnc {

constexpr unsigned kT = 16, kD = 128, kQ = 32;           // tokens a step, head dim, dv a core
constexpr unsigned kTB = kT / 8, kKB = kD / 8, kDB = kQ / 8;
constexpr unsigned kTile = 72;                            // one 8x8 bfp16ebs8 tile, bytes
constexpr unsigned kRec = 9216, kAux = 8192;
constexpr unsigned kAuxE = 0, kAuxF = 32, kAuxB = 64, kAuxG = 96, kAuxW = 96 + 256, kAuxG15 = 96 + 256 + 128;
constexpr unsigned kSBytes = kDB * kKB * 64 * 4;          // 16384

using BV = aie::block_vector<bfp16ebs8, 64>;
using Acc = aie::accum<accfloat, 64>;
using Acc32 = aie::accum<accfloat, 32>;
using Acc16 = aie::accum<accfloat, 16>;
using VB64 = aie::vector<bfloat16, 64>;
using VB32 = aie::vector<bfloat16, 32>;
using VB16 = aie::vector<bfloat16, 16>;

// scratch layout (bytes), the design's Buffer. Every region, and the buffer's size, a multiple of
// 128: wide vector loads ignore the low address bits, so a buffer 32 bytes off (a scratch size not
// a multiple of 64 moved every buffer after it) read every tile shifted.
constexpr unsigned al128(unsigned x) { return (x + 127u) & ~127u; }
constexpr unsigned kScKb = 0;                                        // K as bfp16 [tb][kb]          2304
constexpr unsigned kScQb = kScKb + al128(kTB * kKB * kTile);         // Q as bfp16 [tb][kb]          2304
constexpr unsigned kScKt = kScQb + al128(kTB * kKB * kTile);         // (w . K)^T as bfp16 [kb][tb]  2304
constexpr unsigned kScP = kScKt + al128(kTB * kKB * kTile);          // P f32 [tb][db][64]           2048
constexpr unsigned kScQs = kScP + kTB * kDB * 256;                   // Qs f32 [tb][db][64]          2048
constexpr unsigned kScKK = kScQs + kTB * kDB * 256;                  // KK f32 [tb][sb][64]          1024
constexpr unsigned kScQK = kScKK + 4 * 256;                          // QK f32 [tb][sb][64]          1024
constexpr unsigned kScC = kScQK + 4 * 256;                           // c_ts hi / lo bf16 [2][16][16] 1024
constexpr unsigned kScAO = kScC + 1024;                              // R . QK f32 [tb][sb][64]      1024
constexpr unsigned kScAOb = kScAO + 1024;                            // the same as bfp16 tiles      288
constexpr unsigned kScD = kScAOb + al128(4 * kTile);                 // D hi / lo bf16 [2][16 t][32] 2048
constexpr unsigned kScDT = kScD + 2048;                              // D^T bfp16 [db][sb]           576
constexpr unsigned kScBytes = kScDT + al128(kDB * kTB * kTile);      // 18432
static_assert(kScBytes % 128 == 0, "the scratch keeps the buffers after it aligned");

static inline BV to_bfp(const VB64 &v) {
  Acc a;
  a = v;
  return a.template to_vector<bfp16ebs8>();
}
static inline BV to_bfp(const aie::vector<float, 64> &v) {
  Acc a;
  a.from_vector(v);
  return a.template to_vector<bfp16ebs8>();
}
// bfp16 tile arrays are written by one output stream in order and read by tile index (seek): a
// stream per tile at a 72-byte offset wrote over its neighbours
using BOut = aie::block_vector_output_buffer_stream<bfp16ebs8, 64>;
static inline BV ld_bfp(const uint8_t *base, unsigned idx) {
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> s(reinterpret_cast<const bfp16ebs8 *>(base));
  s.seek(idx);
  return s.pop();
}
static inline BOut out_bfp(uint8_t *base) { return BOut(reinterpret_cast<bfp16ebs8 *>(base)); }
// fp32 accumulator -> bf16 hi, lo
template <unsigned N>
static inline void split(const aie::accum<accfloat, N> &a, aie::vector<bfloat16, N> &h, aie::vector<bfloat16, N> &l) {
  h = a.template to_vector<bfloat16>();
  l = aie::sub(a, h).template to_vector<bfloat16>();
}
// (xh + xl) * (yh + yl) without the lo * lo term, added to acc
template <unsigned N, typename Y>
static inline aie::accum<accfloat, N> mac3(aie::accum<accfloat, N> acc, const aie::vector<bfloat16, N> &xh,
                                           const aie::vector<bfloat16, N> &xl, const Y &yh, const Y &yl) {
  acc = aie::mac(acc, xh, yh);
  acc = aie::mac(acc, xh, yl);
  return aie::mac(acc, xl, yh);
}

// the state slice in from a 1 KB piece (16 of them a head), out the same way
static inline void s_in(const uint8_t *__restrict piece, float *__restrict S, unsigned idx) {
  const float *p = reinterpret_cast<const float *>(piece);
  for (unsigned i = 0; i < 256; i += 16) aie::store_v(S + idx * 256 + i, aie::load_v<16>(p + i));
}
static inline void s_out(const float *__restrict S, uint8_t *__restrict piece, unsigned idx) {
  float *p = reinterpret_cast<float *>(piece);
  for (unsigned i = 0; i < 256; i += 16) aie::store_v(p + i, aie::load_v<16>(S + idx * 256 + i));
}

// one 16-token step of the unit
static inline void step(const uint8_t *__restrict rec, const bfloat16 *__restrict v, bfloat16 *__restrict o,
                        float *__restrict S, uint8_t *__restrict sc) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const bfloat16 *__restrict kA = reinterpret_cast<const bfloat16 *>(rec);
  const bfloat16 *__restrict qA = kA + kTB * kKB * 64;
  const bfloat16 *__restrict aux = reinterpret_cast<const bfloat16 *>(rec + kAux);
  // K, Q as bfp16 [tb][kb]; (w . K)^T as bfp16 [kb][tb] (the update's B tiles)
  {
    BOut ok = out_bfp(sc + kScKb), oq = out_bfp(sc + kScQb);
    for (unsigned i = 0; i < kTB * kKB; ++i) {
      ok << to_bfp(aie::load_v<64>(kA + i * 64));
      oq << to_bfp(aie::load_v<64>(qA + i * 64));
    }
    BOut ot = out_bfp(sc + kScKt);
    for (unsigned kb = 0; kb < kKB; ++kb)
      for (unsigned tb = 0; tb < kTB; ++tb) {
        const VB64 ktt = aie::transpose(aie::load_v<64>(kA + (tb * kKB + kb) * 64), 8, 8);   // [8 dk][8 t]
        Acc a = aie::mul(ktt, aie::load_v<64>(aux + kAuxW + tb * 64));
        ot << a.template to_vector<bfp16ebs8>();
      }
  }
  // KK, QK: [tb][sb] tiles of [8 t][8 s]
  float *__restrict KK = reinterpret_cast<float *>(sc + kScKK);
  float *__restrict QK = reinterpret_cast<float *>(sc + kScQK);
  for (unsigned tb = 0; tb < kTB; ++tb)
    for (unsigned sb = 0; sb <= tb; ++sb) {
      Acc ck = aie::zeros<accfloat, 64>(), cq = aie::zeros<accfloat, 64>();
      for (unsigned kb = 0; kb < kKB; ++kb) {
        const BV ks = ld_bfp(sc + kScKb, sb * kKB + kb);
        ck = mac_8x8_8x8T(ld_bfp(sc + kScKb, tb * kKB + kb), ks, ck);
        cq = mac_8x8_8x8T(ld_bfp(sc + kScQb, tb * kKB + kb), ks, cq);
      }
      aie::store_v(KK + (tb * 2 + sb) * 64, ck.template to_vector<float>());
      aie::store_v(QK + (tb * 2 + sb) * 64, cq.template to_vector<float>());
    }
  // P = K S0, Qs = Q S0: [tb][db] tiles of [8 t][8 dv]
  float *__restrict P = reinterpret_cast<float *>(sc + kScP);
  float *__restrict Qs = reinterpret_cast<float *>(sc + kScQs);
  for (unsigned db = 0; db < kDB; ++db) {
    Acc p0 = aie::zeros<accfloat, 64>(), p1 = aie::zeros<accfloat, 64>();
    Acc q0 = aie::zeros<accfloat, 64>(), q1 = aie::zeros<accfloat, 64>();
    for (unsigned kb = 0; kb < kKB; ++kb) {
      const BV st = to_bfp(aie::load_v<64>(S + (db * kKB + kb) * 64));
      p0 = mac_8x8_8x8T(ld_bfp(sc + kScKb, kb), st, p0);
      p1 = mac_8x8_8x8T(ld_bfp(sc + kScKb, kKB + kb), st, p1);
      q0 = mac_8x8_8x8T(ld_bfp(sc + kScQb, kb), st, q0);
      q1 = mac_8x8_8x8T(ld_bfp(sc + kScQb, kKB + kb), st, q1);
    }
    aie::store_v(P + db * 64, p0.template to_vector<float>());
    aie::store_v(P + (kDB + db) * 64, p1.template to_vector<float>());
    aie::store_v(Qs + db * 64, q0.template to_vector<float>());
    aie::store_v(Qs + (kDB + db) * 64, q1.template to_vector<float>());
  }
  // R rows (r_ts = e_t f_s), c_ts = r_ts KK_ts (s < t, as hi / lo), R . QK (s <= t)
  const VB16 fh = aie::load_v<16>(aux + kAuxF), fl = aie::load_v<16>(aux + kAuxF + 16);
  bfloat16 *__restrict ch = reinterpret_cast<bfloat16 *>(sc + kScC), *__restrict cl = ch + 256;
  float *__restrict AO = reinterpret_cast<float *>(sc + kScAO);
  for (unsigned t = 0; t < kT; ++t) {
    Acc16 r = aie::zeros<accfloat, 16>();
    r = mac3(r, fh, fl, aux[kAuxE + t], aux[kAuxE + 16 + t]);   // e_t f_s
    VB16 rh, rl;
    split(r, rh, rl);
    const unsigned tb = t / 8, tr = t % 8;
    // the row's KK / QK values: tiles (tb, 0) and (tb, 1), row tr
    const aie::vector<float, 16> kk = aie::concat(aie::load_v<8>(KK + (tb * 2) * 64 + tr * 8), aie::load_v<8>(KK + (tb * 2 + 1) * 64 + tr * 8));
    const aie::vector<float, 16> qk = aie::concat(aie::load_v<8>(QK + (tb * 2) * 64 + tr * 8), aie::load_v<8>(QK + (tb * 2 + 1) * 64 + tr * 8));
    Acc16 ka, qa;
    ka.from_vector(kk);
    qa.from_vector(qk);
    VB16 kh, kl, qh, ql;
    split(ka, kh, kl);
    split(qa, qh, ql);
    Acc16 c = mac3(aie::zeros<accfloat, 16>(), rh, rl, kh, kl);
    Acc16 ao = mac3(aie::zeros<accfloat, 16>(), rh, rl, qh, ql);
    const auto below = aie::mask<16>::from_uint32((1u << t) - 1u);                     // s < t
    const auto upto = aie::mask<16>::from_uint32((2u << t) - 1u);                      // s <= t
    const aie::vector<float, 16> zf = aie::zeros<float, 16>();
    const aie::vector<float, 16> cv = aie::select(zf, c.template to_vector<float>(), below);
    const aie::vector<float, 16> aov = aie::select(zf, ao.template to_vector<float>(), upto);
    Acc16 cva;
    cva.from_vector(cv);
    VB16 cvh, cvl;
    split(cva, cvh, cvl);
    aie::store_v(ch + t * 16, cvh);
    aie::store_v(cl + t * 16, cvl);
    aie::store_v(AO + (tb * 2) * 64 + tr * 8, aov.template extract<8>(0));
    aie::store_v(AO + (tb * 2 + 1) * 64 + tr * 8, aov.template extract<8>(1));
  }
  {
    BOut oa = out_bfp(sc + kScAOb);
    for (unsigned i = 0; i < 4; ++i) oa << to_bfp(aie::load_v<64>(AO + i * 64));
  }
  // the solve: d_t = b_t (v_t - G_t P_t - sum_{s<t} c_ts d_s), d as hi / lo rows of 32
  bfloat16 *__restrict dh = reinterpret_cast<bfloat16 *>(sc + kScD), *__restrict dl = dh + kT * kQ;
  const bfloat16 *__restrict gh = aux + kAuxG, *__restrict gl = aux + kAuxG + 128;
  for (unsigned t = 0; t < kT; ++t) {
    const unsigned tb = t / 8, tr = t % 8;
    Acc32 x;
    x = aie::load_v<32>(v + t * kQ);
    // - G_t P_t: P's row t is 8 values in each of the 4 dv-block tiles
    const aie::vector<float, 32> pr = aie::concat(
        aie::concat(aie::load_v<8>(P + (tb * kDB + 0) * 64 + tr * 8), aie::load_v<8>(P + (tb * kDB + 1) * 64 + tr * 8)),
        aie::concat(aie::load_v<8>(P + (tb * kDB + 2) * 64 + tr * 8), aie::load_v<8>(P + (tb * kDB + 3) * 64 + tr * 8)));
    Acc32 pa;
    pa.from_vector(pr);
    VB32 ph, pl;
    split(pa, ph, pl);
    const bfloat16 g_h = gh[tb * 64 + tr * 8], g_l = gl[tb * 64 + tr * 8];
    x = aie::msc(x, ph, g_h);
    x = aie::msc(x, ph, g_l);
    x = aie::msc(x, pl, g_h);
    for (unsigned s = 0; s < t; ++s) {
      const VB32 sh = aie::load_v<32>(dh + s * kQ), sl = aie::load_v<32>(dl + s * kQ);
      const bfloat16 c_h = ch[t * 16 + s], c_l = cl[t * 16 + s];
      x = aie::msc(x, sh, c_h);
      x = aie::msc(x, sh, c_l);
      x = aie::msc(x, sl, c_h);
    }
    VB32 xh, xl;
    split(x, xh, xl);
    Acc32 d = mac3(aie::zeros<accfloat, 32>(), xh, xl, aux[kAuxB + t], aux[kAuxB + 16 + t]);
    VB32 d_h, d_l;
    split(d, d_h, d_l);
    aie::store_v(dh + t * kQ, d_h);
    aie::store_v(dl + t * kQ, d_l);
  }
  // D^T tiles [db][sb]: D's tile [8 s][8 dv] transposed (dh is D rounded to bf16 already; the
  // bfp16 tile keeps less than that)
  {
    BOut od = out_bfp(sc + kScDT);
    for (unsigned db = 0; db < kDB; ++db)
      for (unsigned sb = 0; sb < kTB; ++sb) {
        VB64 tile;
        for (unsigned r = 0; r < 8; ++r) tile.insert(r, aie::load_v<8>(dh + (sb * 8 + r) * kQ + db * 8));
        od << to_bfp(aie::transpose(tile, 8, 8));
      }
  }
  // O = G . Qs + (R . QK) D, out as bf16 rows [16 t][32 dv]
  for (unsigned tb = 0; tb < kTB; ++tb)
    for (unsigned db = 0; db < kDB; ++db) {
      Acc qa;
      qa.from_vector(aie::load_v<64>(Qs + (tb * kDB + db) * 64));
      VB64 qh, ql;
      split(qa, qh, ql);
      Acc c = mac3(aie::zeros<accfloat, 64>(), qh, ql, aie::load_v<64>(gh + tb * 64), aie::load_v<64>(gl + tb * 64));
      for (unsigned sb = 0; sb <= tb; ++sb)
        c = mac_8x8_8x8T(ld_bfp(sc + kScAOb, tb * 2 + sb), ld_bfp(sc + kScDT, db * kTB + sb), c);
      const VB64 ob = c.template to_vector<bfloat16>();
      for (unsigned r = 0; r < 8; ++r) aie::store_v(o + (tb * 8 + r) * kQ + db * 8, ob.template extract<8>(r));
    }
#ifdef DNC_DBG
  // debug: o = an intermediate as bf16 [16][32] (1 P, 2 Qs, 3 D, 4 KK | QK [16][16 | 16])
  for (unsigned t = 0; t < kT; ++t)
    for (unsigned j = 0; j < kQ; j += 8) {
      const unsigned tb = t / 8, tr = t % 8;
      aie::vector<float, 8> x;
      if (DNC_DBG == 1) x = aie::load_v<8>(P + (tb * kDB + j / 8) * 64 + tr * 8);
      else if (DNC_DBG == 2) x = aie::load_v<8>(Qs + (tb * kDB + j / 8) * 64 + tr * 8);
      else if (DNC_DBG == 4) x = aie::load_v<8>((j < 16 ? KK : QK) + (tb * 2 + (j / 8) % 2) * 64 + tr * 8);
      else {
        Acc16 a;
        a = aie::concat(aie::load_v<8>(dh + t * kQ + j), aie::load_v<8>(dh + t * kQ + j));
        x = a.template to_vector<float>().template extract<8>(0);
      }
      Acc16 a2;
      a2.from_vector(aie::concat(x, x));
      aie::store_v(o + t * kQ + j, a2.template to_vector<bfloat16>().template extract<8>(0));
    }
  return;
#endif
  // S^T = G_15 S^T + D^T (w . K)
  const bfloat16 g15h = aux[kAuxG15], g15l = aux[kAuxG15 + 1];
  for (unsigned db = 0; db < kDB; ++db)
    for (unsigned kb = 0; kb < kKB; ++kb) {
      float *__restrict st = S + (db * kKB + kb) * 64;
      Acc sa;
      sa.from_vector(aie::load_v<64>(st));
      VB64 s_h, s_l;
      split(sa, s_h, s_l);
      Acc c = mac3(aie::zeros<accfloat, 64>(), s_h, s_l, g15h, g15l);
      for (unsigned sb = 0; sb < kTB; ++sb)
        c = mac_8x8_8x8T(ld_bfp(sc + kScDT, db * kTB + sb), ld_bfp(sc + kScKt, kb * kTB + sb), c);
      aie::store_v(st, c.template to_vector<float>());
    }
}

}  // namespace dnc
