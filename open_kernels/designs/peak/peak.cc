// peak.py: the matrix unit's raw issue rate per core, operands in registers. Each call
// runs PEAK_ITERS iterations of NACC independent multiply-accumulates (independent so the
// MAC latency pipelines), then stores the accumulators so nothing is dead code.
//   PEAK_BFP : mac_8x8_8x8T on bfp16ebs8 (8x8x8 = 512 MAC per instruction)
//   PEAK_I8  : aie::mmul<8,8,8,int8,int8> (512 MAC)
//   PEAK_I4  : aie::mmul<4,16,16,int8,int4> (1024 MAC)
#include <aie_api/aie.hpp>
#if defined(PEAK_Q4U)
#define MB_NT 8
#include "moe_batch.h"
#endif
#include <stdint.h>
#ifndef PEAK_ITERS
#define PEAK_ITERS 1048576
#endif
#define NACC 4

#if defined(PEAK_B4HSDIAG)
#define PEAK_B4HSCHK 1
#endif
#if defined(PEAK_HS) || defined(PEAK_HSREF) || defined(PEAK_HSCHK) || defined(PEAK_B4) || defined(PEAK_B4CHK) || defined(PEAK_B4HS) || defined(PEAK_B4HSCHK)
// hand-scheduled dit call (hsgen.py -> hs_call.inc) and the stock 2x2 C++ kernel it must match
// bit for bit: A [row pair][k][A0, A1], B [column pair][k][B0, B1], C [row block][column block][8][8]
#if defined(PEAK_HS) || defined(PEAK_HSCHK)
#include "hs_call.inc"
#endif
#if defined(PEAK_B4HS) || defined(PEAK_B4HSCHK)
#include "hs_b4_call.inc"
extern "C" void hs_b4_call(const bfp16ebs8 *pA, const uint8_t *pBn, bfloat16 *pC, const int8_t *pE);
#endif
extern "C" void hs_call(const bfp16ebs8 *pA, const bfp16ebs8 *pB, bfloat16 *pC);
static void hs_ref(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, bfloat16 *__restrict pC) {
  constexpr unsigned rowA = 4, colA = 8, colB = 16, sizeC = 64;
  for (unsigned z = 0; z < rowA; z += 2) {
    bfloat16 *__restrict pC1 = pC + (z * colB) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * sizeC;
    for (unsigned j = 0; j < colB; j += 2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sa(pA); sa.seek(z * colA);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sb(pB); sb.seek(j * colA);
      aie::accum<accfloat, 64> c00(aie::load_v<64>(pC1)), c01(aie::load_v<64>(pC1 + sizeC));
      aie::accum<accfloat, 64> c10(aie::load_v<64>(pC2)), c11(aie::load_v<64>(pC2 + sizeC));
      for (unsigned i = 0; i < colA; ++i) {
        const auto a0 = sa.pop(), a1 = sa.pop(), b0 = sb.pop(), b1 = sb.pop();
        c00 = mac_8x8_8x8T(a0, b0, c00); c01 = mac_8x8_8x8T(a0, b1, c01);
        c10 = mac_8x8_8x8T(a1, b0, c10); c11 = mac_8x8_8x8T(a1, b1, c11);
      }
      aie::store_v(pC1, c00.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC1, c01.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC2, c10.template to_vector<bfloat16>()); pC2 += sizeC;
      aie::store_v(pC2, c11.template to_vector<bfloat16>()); pC2 += sizeC;
    }
  }
}
// buffers in separate 8 KB regions of the core's data memory (HS_SPLIT=1), or packed
alignas(64) static uint8_t hs_pool[0xE000];
#ifndef HS_SPLIT
#define HS_SPLIT 1
#endif
static uint8_t *hs_base() { return HS_SPLIT ? (uint8_t *)(((uintptr_t)hs_pool + 0x1FFF) & ~(uintptr_t)0x1FFF) : hs_pool; }
#define hsA (hs_base())
#define hsB (hs_base() + (HS_SPLIT ? 0x2000 : 32 * 72))
#define hsC ((bfloat16 *)(hs_base() + (HS_SPLIT ? 0x6000 : 160 * 72)))
#define hsC2 ((bfloat16 *)(hs_base() + (HS_SPLIT ? 0x8000 : 160 * 72 + 8192)))
static uint32_t hs_seed;
static float hs_rnd() { hs_seed = hs_seed * 1664525u + 1013904223u; return (float)((int)((hs_seed >> 8) & 0xFFFF) - 32768) * (1.0f / 32768.0f); }
// int4-mantissa weights (b4): per weight block (8 columns x 8 k) 32 bytes of nibbles in
// unpack order, and per column block one exponent byte per column for the call's 64 of K
// (constant across its 8 k-steps). The block the MAC sees is {unpack(nibbles), exponents}.
// B nibbles [column pair 8][k 8][2][32 B], exponents [column block 16][8 B].
static void b4_call(const bfp16ebs8 *__restrict pA, const uint8_t *__restrict pBn, const int8_t *__restrict pBe,
                    bfloat16 *__restrict pC) {
  constexpr unsigned colB = 16, sizeC = 64;
  for (unsigned z = 0; z < 4; z += 2) {
    bfloat16 *__restrict pC1 = pC + (z * colB) * sizeC;
    bfloat16 *__restrict pC2 = pC + ((z + 1) * colB) * sizeC;
    for (unsigned j = 0; j < colB; j += 2) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> sa(pA); sa.seek(z * 8);
      const v8int8 e0 = *reinterpret_cast<const v8int8 *>(pBe + j * 8), e1 = *reinterpret_cast<const v8int8 *>(pBe + (j + 1) * 8);
      aie::accum<accfloat, 64> c00(aie::load_v<64>(pC1)), c01(aie::load_v<64>(pC1 + sizeC));
      aie::accum<accfloat, 64> c10(aie::load_v<64>(pC2)), c11(aie::load_v<64>(pC2 + sizeC));
      for (unsigned i = 0; i < 8; ++i) {
        const auto a0 = sa.pop(), a1 = sa.pop();
        const uint8_t *nb = pBn + ((j / 2) * 8 + i) * 64;
        v64bfp16ebs8 b0, b1;
        b0.mantissa = unpack(*reinterpret_cast<const v64int4 *>(nb)); b0.exponent = e0;
        b1.mantissa = unpack(*reinterpret_cast<const v64int4 *>(nb + 32)); b1.exponent = e1;
        const aie::block_vector<bfp16ebs8, 64> B0(b0), B1(b1);
        c00 = mac_8x8_8x8T(a0, B0, c00); c01 = mac_8x8_8x8T(a0, B1, c01);
        c10 = mac_8x8_8x8T(a1, B0, c10); c11 = mac_8x8_8x8T(a1, B1, c11);
      }
      aie::store_v(pC1, c00.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC1, c01.template to_vector<bfloat16>()); pC1 += sizeC;
      aie::store_v(pC2, c10.template to_vector<bfloat16>()); pC2 += sizeC;
      aie::store_v(pC2, c11.template to_vector<bfloat16>()); pC2 += sizeC;
    }
  }
}
// b4 weights in their own 8 KB bank of hs_pool: nibbles at +0xA000, exponents at +0xB000
#define b4N (hs_base() + 0xA000)
#define b4E (reinterpret_cast<int8_t *>(hs_base() + 0xB000))
alignas(64) static uint8_t b4A[32 * 72];       // hsA as [row pair][A0 k 0..7][A1 k 0..7] (hs_b4_call's order)
static void b4_reorder_a() {
  for (unsigned z = 0; z < 2; ++z)
    for (unsigned k = 0; k < 8; ++k)
      for (unsigned r = 0; r < 2; ++r)
        for (unsigned i = 0; i < 72; ++i) b4A[((z * 2 + r) * 8 + k) * 72 + i] = hsA[((z * 8 + k) * 2 + r) * 72 + i];
}
// the same weights as ordinary bfp16 blocks in hs_ref's B order: per group [exponent][8 mantissas]
static void b4_expand(uint8_t *dst) {
  for (unsigned blk = 0; blk < 128; ++blk) {          // blk = (pair * 8 + k) * 2 + n2
    const unsigned cb = (blk / 16) * 2 + blk % 2;
    for (unsigned g = 0; g < 8; ++g) {
      uint8_t *o = dst + (blk * 8 + g) * 9;
      o[0] = static_cast<uint8_t>(b4E[cb * 8 + g]);
      for (unsigned t = 0; t < 8; ++t) {
        const unsigned i = g * 8 + t;
        const uint8_t byte = b4N[blk * 32 + i / 2];
        int v = (i & 1) ? (byte >> 4) : (byte & 15);
        o[1 + t] = static_cast<uint8_t>(static_cast<int8_t>(v >= 8 ? v - 16 : v));
      }
    }
  }
}
static void hs_fill(uint8_t *dst, unsigned nblk, bool rnd) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o(reinterpret_cast<bfp16ebs8 *>(dst));
  for (unsigned b = 0; b < nblk; ++b) {
    aie::vector<bfloat16, 64> v;
    for (unsigned i = 0; i < 64; ++i) v[i] = (bfloat16)(rnd ? hs_rnd() * (1 + (b & 3)) : 0.25f);
    aie::accum<accfloat, 64> t; t.from_vector(v);
    o << t.template to_vector<bfp16ebs8>();
  }
}
#endif
#if defined(PEAK_B4PROBE2)
#include "b4probe2.inc"
alignas(64) static uint8_t b4qA[3 * 72 + 64];
alignas(64) static uint8_t b4qN[64];
alignas(64) static int32_t b4qE[2];
alignas(64) static bfloat16 b4qC[128];
#endif
#if defined(PEAK_FAQKT) || defined(PEAK_FAPVT)
// cycles of fa_qk_call / fa_pv_call on L1-resident data: PEAK_ITERS calls (time / iters at the clock)
#define FA_HS 1
#include "../dit_fa256b/fa256b.cc"
#ifdef PEAK_FABANK8   // A and B 8 KB apart inside one 16 KB-aligned region, C 16 KB further
struct alignas(16384) FtArena { uint8_t a[8192]; uint8_t b[8192]; bfloat16 c[TQ * DFULL]; };
static FtArena ft;
static uint8_t *const ftA = ft.a, *const ftB = ft.b;
static bfloat16 *const ftC = ft.c;
#elif defined(PEAK_FABANK)   // each operand in its own 8 KB bank
alignas(8192) static uint8_t ftA[9216];
alignas(8192) static uint8_t ftB[4608];
alignas(8192) static bfloat16 ftC[TQ * DFULL];
#else
alignas(64) static uint8_t ftA[9216], ftB[4608];
alignas(64) static bfloat16 ftC[TQ * DFULL];
#endif
#endif
#if defined(PEAK_FAEXP)
// fa_exp_call (dit_fa256b/fa_hs.inc) against the C++ pass on one core: G from a formula, guard
// bytes after every output. out: [0] P bytes that differ, [1] first one, [2] P guard bytes
// changed (asm), [3] rs guard floats changed (asm), [4] rs lanes that differ by > 1e-3 rel,
// [5] first one, [8..15] asm rs[0..7] bits, [16..23] C++ rs[0..7] bits, [24..31] asm P[0..31] words
#define FA_HS 1
#include "../dit_fa256b/fa256b.cc"
alignas(64) static bfloat16 fxG[TQ * LKP];
alignas(64) static uint8_t fxPa[2304 + 128], fxPc[2304 + 128];
alignas(64) static float fxRa[256 + 32], fxRc[256 + 32];
#endif
#if defined(PEAK_B4PROBE)
#include "b4probe.inc"
alignas(64) static uint8_t b4pN[96];
alignas(64) static int32_t b4pE[6];
alignas(64) static bfloat16 b4pC[64];
#endif
extern "C" {
void peak_run(const int8_t *__restrict in, int32_t *__restrict out) {
#if defined(PEAK_BFP)
  aie::accum<accfloat, 64> acc[NACC];
  aie::accum<accfloat, 64> t0, t1;
  t0.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
  t1.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in) + 64));
  const aie::block_vector<bfp16ebs8, 64> a = t0.template to_vector<bfp16ebs8>();
  const aie::block_vector<bfp16ebs8, 64> b = t1.template to_vector<bfp16ebs8>();
  for (unsigned i = 0; i < NACC; ++i) acc[i] = aie::zeros<accfloat, 64>();
  for (unsigned it = 0; it < PEAK_ITERS; ++it) chess_prepare_for_pipelining {
    acc[0] = mac_8x8_8x8T(a, b, acc[0]);
    acc[1] = mac_8x8_8x8T(a, b, acc[1]);
    acc[2] = mac_8x8_8x8T(a, b, acc[2]);
    acc[3] = mac_8x8_8x8T(a, b, acc[3]);
  }
  for (unsigned i = 0; i < NACC; ++i) aie::store_v(reinterpret_cast<float *>(out) + i * 64, acc[i].template to_vector<float>());
#elif defined(PEAK_I8)
  const aie::vector<int8, 64> a = aie::load_v<64>(in), b = aie::load_v<64>(in + 64);
  aie::mmul<8, 8, 8, int8, int8> m[NACC];
  for (unsigned i = 0; i < NACC; ++i) m[i].mul(a, b);
  for (unsigned it = 0; it < PEAK_ITERS; ++it) chess_prepare_for_pipelining {
    m[0].mac(a, b); m[1].mac(a, b); m[2].mac(a, b); m[3].mac(a, b);
  }
  for (unsigned i = 0; i < NACC; ++i) aie::store_v(out + i * 64, m[i].template to_vector<int32>());
#elif defined(PEAK_I4)
  const aie::vector<int8, 64> a = aie::load_v<64>(in);
  const aie::vector<int4, 256> b = aie::load_v<256>(reinterpret_cast<const int4 *>(in + 64));
  aie::mmul<4, 16, 16, int8, int4> m[NACC];
  for (unsigned i = 0; i < NACC; ++i) m[i].mul(a, b);
  for (unsigned it = 0; it < PEAK_ITERS; ++it) chess_prepare_for_pipelining {
    m[0].mac(a, b); m[1].mac(a, b); m[2].mac(a, b); m[3].mac(a, b);
  }
  for (unsigned i = 0; i < NACC; ++i) aie::store_v(out + i * 64, m[i].template to_vector<int32>());
#elif defined(PEAK_BFPL1)
  // operands from L1: 2 A x 2 B bfp16 blocks per k-step, 4 accumulators held over the K loop
  static constexpr unsigned KS = 64;                    // k-steps (8 k each) resident: K = 512
  alignas(64) static uint8_t As[2 * KS * 72], Bs[2 * KS * 72];
  {
    aie::accum<accfloat, 64> t;
    t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const aie::block_vector<bfp16ebs8, 64> v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(As)), ob(reinterpret_cast<bfp16ebs8 *>(Bs));
    for (unsigned i = 0; i < 2 * KS; ++i) { oa << v; ob << v; }
  }
  aie::accum<accfloat, 64> c00 = aie::zeros<accfloat, 64>(), c01 = c00, c10 = c00, c11 = c00;
  for (unsigned it = 0; it < PEAK_ITERS / KS; ++it) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ia(reinterpret_cast<const bfp16ebs8 *>(As)), ib(reinterpret_cast<const bfp16ebs8 *>(Bs));
    for (unsigned k = 0; k < KS; ++k) chess_prepare_for_pipelining {
      const auto a0 = ia.pop(), a1 = ia.pop(), b0 = ib.pop(), b1 = ib.pop();
      c00 = mac_8x8_8x8T(a0, b0, c00); c01 = mac_8x8_8x8T(a0, b1, c01);
      c10 = mac_8x8_8x8T(a1, b0, c10); c11 = mac_8x8_8x8T(a1, b1, c11);
    }
  }
  float *o = reinterpret_cast<float *>(out);
  aie::store_v(o, c00.template to_vector<float>()); aie::store_v(o + 64, c01.template to_vector<float>());
  aie::store_v(o + 128, c10.template to_vector<float>()); aie::store_v(o + 192, c11.template to_vector<float>());
#elif defined(PEAK_I8L1)
  static constexpr unsigned KS = 64;
  alignas(64) static int8_t As[2 * KS * 64], Bs[2 * KS * 64];
  for (unsigned i = 0; i < 2 * KS * 64; i += 64) { aie::store_v(As + i, aie::load_v<64>(in)); aie::store_v(Bs + i, aie::load_v<64>(in + 64)); }
  aie::mmul<8, 8, 8, int8, int8> c00, c01, c10, c11;
  { const auto a = aie::load_v<64>(As), b = aie::load_v<64>(Bs); c00.mul(a, b); c01.mul(a, b); c10.mul(a, b); c11.mul(a, b); }
  for (unsigned it = 0; it < PEAK_ITERS / KS; ++it) {
    const int8_t *pa = As, *pb = Bs;
    for (unsigned k = 0; k < KS; ++k) chess_prepare_for_pipelining {
      const auto a0 = aie::load_v<64>(pa), a1 = aie::load_v<64>(pa + 64), b0 = aie::load_v<64>(pb), b1 = aie::load_v<64>(pb + 64);
      pa += 128; pb += 128;
      c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
    }
  }
  aie::store_v(out, c00.template to_vector<int32>()); aie::store_v(out + 64, c01.template to_vector<int32>());
  aie::store_v(out + 128, c10.template to_vector<int32>()); aie::store_v(out + 192, c11.template to_vector<int32>());
#elif defined(PEAK_BFPL42)
  // 4 A x 2 B blocks per k-step: 6 block loads for 8 MACs
  static constexpr unsigned KS = 32;
  alignas(64) static uint8_t As[4 * KS * 72], Bs[2 * KS * 72];
  {
    aie::accum<accfloat, 64> t;
    t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const aie::block_vector<bfp16ebs8, 64> v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(As)), ob(reinterpret_cast<bfp16ebs8 *>(Bs));
    for (unsigned i = 0; i < 4 * KS; ++i) oa << v;
    for (unsigned i = 0; i < 2 * KS; ++i) ob << v;
  }
  aie::accum<accfloat, 64> c[8];
  for (unsigned i = 0; i < 8; ++i) c[i] = aie::zeros<accfloat, 64>();
  for (unsigned it = 0; it < PEAK_ITERS / KS / 2; ++it) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ia(reinterpret_cast<const bfp16ebs8 *>(As)), ib(reinterpret_cast<const bfp16ebs8 *>(Bs));
    for (unsigned k = 0; k < KS; ++k) chess_prepare_for_pipelining {
      const auto a0 = ia.pop(), a1 = ia.pop(), a2 = ia.pop(), a3 = ia.pop(), b0 = ib.pop(), b1 = ib.pop();
      c[0] = mac_8x8_8x8T(a0, b0, c[0]); c[1] = mac_8x8_8x8T(a0, b1, c[1]);
      c[2] = mac_8x8_8x8T(a1, b0, c[2]); c[3] = mac_8x8_8x8T(a1, b1, c[3]);
      c[4] = mac_8x8_8x8T(a2, b0, c[4]); c[5] = mac_8x8_8x8T(a2, b1, c[5]);
      c[6] = mac_8x8_8x8T(a3, b0, c[6]); c[7] = mac_8x8_8x8T(a3, b1, c[7]);
    }
  }
  float *o = reinterpret_cast<float *>(out);
  aie::accum<accfloat, 64> s = c[0];
  for (unsigned i = 1; i < 8; ++i) s = aie::add(s, c[i].template to_vector<float>());
  aie::store_v(o, s.template to_vector<float>());
#elif defined(PEAK_I8L42)
  static constexpr unsigned KS = 32;
  alignas(64) static int8_t As[4 * KS * 64], Bs[2 * KS * 64];
  for (unsigned i = 0; i < 4 * KS * 64; i += 64) aie::store_v(As + i, aie::load_v<64>(in));
  for (unsigned i = 0; i < 2 * KS * 64; i += 64) aie::store_v(Bs + i, aie::load_v<64>(in + 64));
  aie::mmul<8, 8, 8, int8, int8> c0, c1, c2, c3, c4, c5, c6, c7;
  { const auto a = aie::load_v<64>(As), b = aie::load_v<64>(Bs);
    c0.mul(a, b); c1.mul(a, b); c2.mul(a, b); c3.mul(a, b); c4.mul(a, b); c5.mul(a, b); c6.mul(a, b); c7.mul(a, b); }
  for (unsigned it = 0; it < PEAK_ITERS / KS / 2; ++it) {
    const int8_t *pa = As, *pb = Bs;
    for (unsigned k = 0; k < KS; ++k) chess_prepare_for_pipelining {
      const auto a0 = aie::load_v<64>(pa), a1 = aie::load_v<64>(pa + 64), a2 = aie::load_v<64>(pa + 128), a3 = aie::load_v<64>(pa + 192);
      const auto b0 = aie::load_v<64>(pb), b1 = aie::load_v<64>(pb + 64);
      pa += 256; pb += 128;
      c0.mac(a0, b0); c1.mac(a0, b1); c2.mac(a1, b0); c3.mac(a1, b1);
      c4.mac(a2, b0); c5.mac(a2, b1); c6.mac(a3, b0); c7.mac(a3, b1);
    }
  }
  auto s = aie::add(c0.template to_vector<int32>(), c1.template to_vector<int32>());
  s = aie::add(s, c2.template to_vector<int32>()); s = aie::add(s, c3.template to_vector<int32>());
  s = aie::add(s, c4.template to_vector<int32>()); s = aie::add(s, c5.template to_vector<int32>());
  s = aie::add(s, c6.template to_vector<int32>()); s = aie::add(s, c7.template to_vector<int32>());
  aie::store_v(out, s);
#elif defined(PEAK_I8L22U)
  static constexpr unsigned KS = 64;
  alignas(64) static int8_t As[2 * KS * 64], Bs[2 * KS * 64];
  for (unsigned i = 0; i < 2 * KS * 64; i += 64) { aie::store_v(As + i, aie::load_v<64>(in)); aie::store_v(Bs + i, aie::load_v<64>(in + 64)); }
  aie::mmul<8, 8, 8, int8, int8> c00, c01, c10, c11;
  { const auto a = aie::load_v<64>(As), b = aie::load_v<64>(Bs); c00.mul(a, b); c01.mul(a, b); c10.mul(a, b); c11.mul(a, b); }
  for (unsigned it = 0; it < PEAK_ITERS / KS; ++it) {
    const int8_t *pa = As, *pb = Bs;
    for (unsigned k = 0; k < KS; k += 2) chess_prepare_for_pipelining {
      const auto a0 = aie::load_v<64>(pa), a1 = aie::load_v<64>(pa + 64), b0 = aie::load_v<64>(pb), b1 = aie::load_v<64>(pb + 64);
      const auto a2 = aie::load_v<64>(pa + 128), a3 = aie::load_v<64>(pa + 192), b2 = aie::load_v<64>(pb + 128), b3 = aie::load_v<64>(pb + 192);
      pa += 256; pb += 256;
      c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
      c00.mac(a2, b2); c01.mac(a2, b3); c10.mac(a3, b2); c11.mac(a3, b3);
    }
  }
  aie::store_v(out, c00.template to_vector<int32>()); aie::store_v(out + 64, c01.template to_vector<int32>());
  aie::store_v(out + 128, c10.template to_vector<int32>()); aie::store_v(out + 192, c11.template to_vector<int32>());
#elif defined(PEAK_I4L22)
  // int8 A [4 x 16] x int4 B [16 x 16] (1024 MAC, 2 cycles): 2 A x 2 B per k-step of 16, unrolled 2
  static constexpr unsigned KS = 32;
  alignas(64) static int8_t As[2 * KS * 64];
  alignas(64) static int8_t Bs[2 * KS * 128];
  for (unsigned i = 0; i < 2 * KS * 64; i += 64) aie::store_v(As + i, aie::load_v<64>(in));
  for (unsigned i = 0; i < 2 * KS * 128; i += 64) aie::store_v(Bs + i, aie::load_v<64>(in + 64));
  aie::mmul<4, 16, 16, int8, int4> c00, c01, c10, c11;
  { const auto a = aie::load_v<64>(As); const auto b = aie::load_v<256>(reinterpret_cast<const int4 *>(Bs));
    c00.mul(a, b); c01.mul(a, b); c10.mul(a, b); c11.mul(a, b); }
  for (unsigned it = 0; it < PEAK_ITERS / KS; ++it) {
    const int8_t *pa = As, *pb = Bs;
    for (unsigned k = 0; k < KS; k += 2) chess_prepare_for_pipelining {
      const auto a0 = aie::load_v<64>(pa), a1 = aie::load_v<64>(pa + 64), a2 = aie::load_v<64>(pa + 128), a3 = aie::load_v<64>(pa + 192);
      const auto b0 = aie::load_v<256>(reinterpret_cast<const int4 *>(pb)), b1 = aie::load_v<256>(reinterpret_cast<const int4 *>(pb + 128));
      const auto b2 = aie::load_v<256>(reinterpret_cast<const int4 *>(pb + 256)), b3 = aie::load_v<256>(reinterpret_cast<const int4 *>(pb + 384));
      pa += 256; pb += 512;
      c00.mac(a0, b0); c01.mac(a0, b1); c10.mac(a1, b0); c11.mac(a1, b1);
      c00.mac(a2, b2); c01.mac(a2, b3); c10.mac(a3, b2); c11.mac(a3, b3);
    }
  }
  aie::store_v(out, c00.template to_vector<int32>()); aie::store_v(out + 64, c01.template to_vector<int32>());
  aie::store_v(out + 128, c10.template to_vector<int32>()); aie::store_v(out + 192, c11.template to_vector<int32>());
#elif defined(PEAK_DIT0) || defined(PEAK_DIT1)
  // the dit_gemm call (A quarter 32 x 64 bf16, B 64 x 128 bfp16, C quarter 32 x 128 bf16) on
  // L1-resident data, PEAK_ITERS / 512 calls (512 MAC instructions per call)
  alignas(64) static bfloat16 Aq[32 * 64];
  alignas(64) static uint8_t Bt[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  alignas(64) static uint8_t Ascr[32 * 64 / 64 * 72];
  {
    const aie::vector<bfloat16, 64> v = aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in));
    for (unsigned i = 0; i < 32 * 64; i += 64) aie::store_v(Aq + i, v);
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
    aie::accum<accfloat, 64> t; t.from_vector(v);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> ob(reinterpret_cast<bfp16ebs8 *>(Bt));
    for (unsigned i = 0; i < 128; ++i) ob << t.template to_vector<bfp16ebs8>();
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned rowA = 4, colA = 8, colB = 16, sz = 64;
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
#if defined(PEAK_DIT1)
    {   // A converted once per call
      aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ascr));
      for (unsigned i = 0; i < rowA * colA; ++i) { aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(Aq + i * 64)); oa << t.template to_vector<bfp16ebs8>(); }
    }
#endif
    for (unsigned z = 0; z < rowA; z += 2) {
      bfloat16 *__restrict pC1 = Cq + (z * colB) * sz, *__restrict pC2 = Cq + ((z + 1) * colB) * sz;
      for (unsigned j = 0; j < colB; j += 2) {
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1(reinterpret_cast<const bfp16ebs8 *>(Bt)), pB2(reinterpret_cast<const bfp16ebs8 *>(Bt));
        pB1.seek(j * colA); pB2.seek((j + 1) * colA);
        aie::accum<accfloat, 64> c00(aie::load_v<64>(pC1)), c01(aie::load_v<64>(pC1 + 64)), c10(aie::load_v<64>(pC2)), c11(aie::load_v<64>(pC2 + 64));
#if defined(PEAK_DIT1)
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa1(reinterpret_cast<const bfp16ebs8 *>(Ascr)), pa2(reinterpret_cast<const bfp16ebs8 *>(Ascr));
        pa1.seek(z * colA); pa2.seek((z + 1) * colA);
        for (unsigned i = 0; i < colA; ++i) chess_prepare_for_pipelining {
          const auto a0 = pa1.pop(), a1 = pa2.pop(), b0 = pB1.pop(), b1 = pB2.pop();
          c00 = mac_8x8_8x8T(a0, b0, c00); c01 = mac_8x8_8x8T(a0, b1, c01);
          c10 = mac_8x8_8x8T(a1, b0, c10); c11 = mac_8x8_8x8T(a1, b1, c11);
        }
#else
        const bfloat16 *__restrict pA1 = Aq + (z * colA) * sz, *__restrict pA2 = Aq + ((z + 1) * colA) * sz;
        for (unsigned i = 0; i < colA; ++i) chess_prepare_for_pipelining {
          aie::accum<accfloat, 64> a0; a0 = aie::load_v<64>(pA1); pA1 += sz;
          aie::accum<accfloat, 64> a1 = mul_elem_64(aie::load_v<64>(pA2), concat(broadcast_one_to_v32bfloat16(), broadcast_one_to_v32bfloat16())); pA2 += sz;
          const auto b0 = pB1.pop(), b1 = pB2.pop();
          c00 = mac_8x8_8x8T(a0.to_vector<bfp16ebs8>(), b0, c00); c01 = mac_8x8_8x8T(a0.to_vector<bfp16ebs8>(), b1, c01);
          c10 = mac_8x8_8x8T(a1.to_vector<bfp16ebs8>(), b0, c10); c11 = mac_8x8_8x8T(a1.to_vector<bfp16ebs8>(), b1, c11);
        }
#endif
        aie::store_v(pC1, c00.template to_vector<bfloat16>()); pC1 += 64; aie::store_v(pC1, c01.template to_vector<bfloat16>()); pC1 += 64;
        aie::store_v(pC2, c10.template to_vector<bfloat16>()); pC2 += 64; aie::store_v(pC2, c11.template to_vector<bfloat16>()); pC2 += 64;
      }
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT2) || defined(PEAK_DIT3)
  // dit call with A pre-packed bfp16 [z pair][k-block][a(z), a(z+1)] and B [j pair][k-block][b(j), b(j+1)]:
  // two block streams, no conversion in the loop
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];           // 32 blocks
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];          // 128 blocks
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned colA = 8, colB = 16;
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    for (unsigned z = 0; z < 4; z += 2) {
      bfloat16 *__restrict pC1 = Cq + (z * colB) * 64, *__restrict pC2 = Cq + ((z + 1) * colB) * 64;
      for (unsigned j = 0; j < colB; j += 2) {
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap) + (z / 2) * 2 * colA);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp) + (j / 2) * 2 * colA);
        aie::accum<accfloat, 64> c00(aie::load_v<64>(pC1)), c01(aie::load_v<64>(pC1 + 64)), c10(aie::load_v<64>(pC2)), c11(aie::load_v<64>(pC2 + 64));
#if defined(PEAK_DIT3)
#pragma clang loop unroll(full)
#endif
        for (unsigned i = 0; i < colA; ++i) chess_prepare_for_pipelining {
          const auto a0 = pa.pop(), a1 = pa.pop(), b0 = pb.pop(), b1 = pb.pop();
          c00 = mac_8x8_8x8T(a0, b0, c00); c01 = mac_8x8_8x8T(a0, b1, c01);
          c10 = mac_8x8_8x8T(a1, b0, c10); c11 = mac_8x8_8x8T(a1, b1, c11);
        }
        aie::store_v(pC1, c00.template to_vector<bfloat16>()); pC1 += 64; aie::store_v(pC1, c01.template to_vector<bfloat16>()); pC1 += 64;
        aie::store_v(pC2, c10.template to_vector<bfloat16>()); pC2 += 64; aie::store_v(pC2, c11.template to_vector<bfloat16>()); pC2 += 64;
      }
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT4)
  // 2x1 blocking: unit = 2 A row-blocks x 1 B column-block (2 accumulators), k-loop unrolled
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  constexpr unsigned colA = 8, colB = 16;
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    for (unsigned z = 0; z < 4; z += 2) {
      bfloat16 *__restrict pC1 = Cq + (z * colB) * 64, *__restrict pC2 = Cq + ((z + 1) * colB) * 64;
      for (unsigned j = 0; j < colB; ++j) {
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap) + (z / 2) * 2 * colA);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp) + j * colA);
        aie::accum<accfloat, 64> c0(aie::load_v<64>(pC1)), c1(aie::load_v<64>(pC2));
#pragma clang loop unroll(full)
        for (unsigned i = 0; i < colA; ++i) {
          const auto a0 = pa.pop(), a1 = pa.pop(), b0 = pb.pop();
          c0 = mac_8x8_8x8T(a0, b0, c0); c1 = mac_8x8_8x8T(a1, b0, c1);
        }
        aie::store_v(pC1, c0.template to_vector<bfloat16>()); pC1 += 64;
        aie::store_v(pC2, c1.template to_vector<bfloat16>()); pC2 += 64;
      }
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT5)
  // A row-block (8 k-blocks) held in registers; 4 B column-blocks per unit (4 accumulators);
  // B one continuous stream per row-block
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    for (unsigned z = 0; z < 4; ++z) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap) + z * 8);
      const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop(), a4 = pa.pop(), a5 = pa.pop(), a6 = pa.pop(), a7 = pa.pop();
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));   // [j][kb]
      bfloat16 *__restrict pC = Cq + z * 16 * 64;
      for (unsigned jg = 0; jg < 16; jg += 4) {
        aie::accum<accfloat, 64> c0(aie::load_v<64>(pC)), c1(aie::load_v<64>(pC + 64)), c2(aie::load_v<64>(pC + 128)), c3(aie::load_v<64>(pC + 192));
#define KS(A) { const auto b0 = pb.pop(); c0 = mac_8x8_8x8T(A, b0, c0); }
        // B is [j][kb]: column block j's 8 k-blocks, then j+1's -- so walk j then k
        c0 = mac_8x8_8x8T(a0, pb.pop(), c0); c0 = mac_8x8_8x8T(a1, pb.pop(), c0); c0 = mac_8x8_8x8T(a2, pb.pop(), c0); c0 = mac_8x8_8x8T(a3, pb.pop(), c0);
        c0 = mac_8x8_8x8T(a4, pb.pop(), c0); c0 = mac_8x8_8x8T(a5, pb.pop(), c0); c0 = mac_8x8_8x8T(a6, pb.pop(), c0); c0 = mac_8x8_8x8T(a7, pb.pop(), c0);
        c1 = mac_8x8_8x8T(a0, pb.pop(), c1); c1 = mac_8x8_8x8T(a1, pb.pop(), c1); c1 = mac_8x8_8x8T(a2, pb.pop(), c1); c1 = mac_8x8_8x8T(a3, pb.pop(), c1);
        c1 = mac_8x8_8x8T(a4, pb.pop(), c1); c1 = mac_8x8_8x8T(a5, pb.pop(), c1); c1 = mac_8x8_8x8T(a6, pb.pop(), c1); c1 = mac_8x8_8x8T(a7, pb.pop(), c1);
        c2 = mac_8x8_8x8T(a0, pb.pop(), c2); c2 = mac_8x8_8x8T(a1, pb.pop(), c2); c2 = mac_8x8_8x8T(a2, pb.pop(), c2); c2 = mac_8x8_8x8T(a3, pb.pop(), c2);
        c2 = mac_8x8_8x8T(a4, pb.pop(), c2); c2 = mac_8x8_8x8T(a5, pb.pop(), c2); c2 = mac_8x8_8x8T(a6, pb.pop(), c2); c2 = mac_8x8_8x8T(a7, pb.pop(), c2);
        c3 = mac_8x8_8x8T(a0, pb.pop(), c3); c3 = mac_8x8_8x8T(a1, pb.pop(), c3); c3 = mac_8x8_8x8T(a2, pb.pop(), c3); c3 = mac_8x8_8x8T(a3, pb.pop(), c3);
        c3 = mac_8x8_8x8T(a4, pb.pop(), c3); c3 = mac_8x8_8x8T(a5, pb.pop(), c3); c3 = mac_8x8_8x8T(a6, pb.pop(), c3); c3 = mac_8x8_8x8T(a7, pb.pop(), c3);
        aie::store_v(pC, c0.template to_vector<bfloat16>()); aie::store_v(pC + 64, c1.template to_vector<bfloat16>());
        aie::store_v(pC + 128, c2.template to_vector<bfloat16>()); aie::store_v(pC + 192, c3.template to_vector<bfloat16>());
        pC += 256;
      }
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT6)
  // A row-block (8 k-blocks) held in registers; 4 B column-blocks per unit (4 accumulators);
  // B one continuous stream per row-block
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    for (unsigned z = 0; z < 4; ++z) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap) + z * 8);
      const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop(), a4 = pa.pop(), a5 = pa.pop(), a6 = pa.pop(), a7 = pa.pop();
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));   // [j][kb]
      bfloat16 *__restrict pC = Cq + z * 16 * 64;
      for (unsigned jg = 0; jg < 16; jg += 4) {
        aie::accum<accfloat, 64> c0(aie::load_v<64>(pC)), c1(aie::load_v<64>(pC + 64)), c2(aie::load_v<64>(pC + 128)), c3(aie::load_v<64>(pC + 192));
#define KS(A) { const auto b0 = pb.pop(); c0 = mac_8x8_8x8T(A, b0, c0); }
        // B is [j][kb]: column block j's 8 k-blocks, then j+1's -- so walk j then k
        // B packed [j group][k-block][4 columns]: k outer, the 4 accumulators interleaved
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a0, b0, c0); c1 = mac_8x8_8x8T(a0, b1, c1); c2 = mac_8x8_8x8T(a0, b2, c2); c3 = mac_8x8_8x8T(a0, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a1, b0, c0); c1 = mac_8x8_8x8T(a1, b1, c1); c2 = mac_8x8_8x8T(a1, b2, c2); c3 = mac_8x8_8x8T(a1, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a2, b0, c0); c1 = mac_8x8_8x8T(a2, b1, c1); c2 = mac_8x8_8x8T(a2, b2, c2); c3 = mac_8x8_8x8T(a2, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a3, b0, c0); c1 = mac_8x8_8x8T(a3, b1, c1); c2 = mac_8x8_8x8T(a3, b2, c2); c3 = mac_8x8_8x8T(a3, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a4, b0, c0); c1 = mac_8x8_8x8T(a4, b1, c1); c2 = mac_8x8_8x8T(a4, b2, c2); c3 = mac_8x8_8x8T(a4, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a5, b0, c0); c1 = mac_8x8_8x8T(a5, b1, c1); c2 = mac_8x8_8x8T(a5, b2, c2); c3 = mac_8x8_8x8T(a5, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a6, b0, c0); c1 = mac_8x8_8x8T(a6, b1, c1); c2 = mac_8x8_8x8T(a6, b2, c2); c3 = mac_8x8_8x8T(a6, b3, c3); }
        { const auto b0 = pb.pop(), b1 = pb.pop(), b2 = pb.pop(), b3 = pb.pop();
          c0 = mac_8x8_8x8T(a7, b0, c0); c1 = mac_8x8_8x8T(a7, b1, c1); c2 = mac_8x8_8x8T(a7, b2, c2); c3 = mac_8x8_8x8T(a7, b3, c3); }
        aie::store_v(pC, c0.template to_vector<bfloat16>()); aie::store_v(pC + 64, c1.template to_vector<bfloat16>());
        aie::store_v(pC + 128, c2.template to_vector<bfloat16>()); aie::store_v(pC + 192, c3.template to_vector<bfloat16>());
        pC += 256;
      }
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT8)
  // 4x1: unit = the quarter's 4 A row-blocks x 1 B column-block (4 accumulators, one per row
  // block); A packed [k-block][4 row-blocks] and re-streamed per column, B [j][k-block]
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));
    for (unsigned j = 0; j < 16; ++j) {
      // C blocks of column j in the 4 row blocks: C layout [row block][col block][8][8]
      bfloat16 *__restrict c_0 = Cq + (0 * 16 + j) * 64, *__restrict c_1 = Cq + (1 * 16 + j) * 64;
      bfloat16 *__restrict c_2 = Cq + (2 * 16 + j) * 64, *__restrict c_3 = Cq + (3 * 16 + j) * 64;
      aie::accum<accfloat, 64> c0(aie::load_v<64>(c_0)), c1(aie::load_v<64>(c_1)), c2(aie::load_v<64>(c_2)), c3(aie::load_v<64>(c_3));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap));
#pragma clang loop unroll(full)
      for (unsigned k = 0; k < 8; ++k) {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mac_8x8_8x8T(a0, b, c0); c1 = mac_8x8_8x8T(a1, b, c1); c2 = mac_8x8_8x8T(a2, b, c2); c3 = mac_8x8_8x8T(a3, b, c3);
      }
      aie::store_v(c_0, c0.template to_vector<bfloat16>()); aie::store_v(c_1, c1.template to_vector<bfloat16>());
      aie::store_v(c_2, c2.template to_vector<bfloat16>()); aie::store_v(c_3, c3.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT11)
  // 4x1: unit = the quarter's 4 A row-blocks x 1 B column-block (4 accumulators, one per row
  // block); A packed [k-block][4 row-blocks] and re-streamed per column, B [j][k-block]
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));
#pragma clang loop unroll(full)
    for (unsigned j = 0; j < 16; ++j) {
      // C blocks of column j in the 4 row blocks: C layout [row block][col block][8][8]
      bfloat16 *__restrict c_0 = Cq + (0 * 16 + j) * 64, *__restrict c_1 = Cq + (1 * 16 + j) * 64;
      bfloat16 *__restrict c_2 = Cq + (2 * 16 + j) * 64, *__restrict c_3 = Cq + (3 * 16 + j) * 64;
      aie::accum<accfloat, 64> c0(aie::load_v<64>(c_0)), c1(aie::load_v<64>(c_1)), c2(aie::load_v<64>(c_2)), c3(aie::load_v<64>(c_3));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap));
#pragma clang loop unroll(full)
      for (unsigned k = 0; k < 8; ++k) {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mac_8x8_8x8T(a0, b, c0); c1 = mac_8x8_8x8T(a1, b, c1); c2 = mac_8x8_8x8T(a2, b, c2); c3 = mac_8x8_8x8T(a3, b, c3);
      }
      aie::store_v(c_0, c0.template to_vector<bfloat16>()); aie::store_v(c_1, c1.template to_vector<bfloat16>());
      aie::store_v(c_2, c2.template to_vector<bfloat16>()); aie::store_v(c_3, c3.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT9)
  // 4x1: unit = the quarter's 4 A row-blocks x 1 B column-block (4 accumulators, one per row
  // block); A packed [k-block][4 row-blocks] and re-streamed per column, B [j][k-block]
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));
    for (unsigned j = 0; j < 16; ++j) {
      // C blocks of column j in the 4 row blocks: C layout [row block][col block][8][8]
      bfloat16 *__restrict c_0 = Cq + (0 * 16 + j) * 64, *__restrict c_1 = Cq + (1 * 16 + j) * 64;
      bfloat16 *__restrict c_2 = Cq + (2 * 16 + j) * 64, *__restrict c_3 = Cq + (3 * 16 + j) * 64;
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap));
      // the unit starts from its first product, not from C: nothing waits on the C loads
      aie::accum<accfloat, 64> c0, c1, c2, c3;
      {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mul_8x8_8x8T(a0, b); c1 = mul_8x8_8x8T(a1, b); c2 = mul_8x8_8x8T(a2, b); c3 = mul_8x8_8x8T(a3, b);
      }
      const aie::vector<bfloat16, 64> o0 = aie::load_v<64>(c_0), o1 = aie::load_v<64>(c_1), o2 = aie::load_v<64>(c_2), o3 = aie::load_v<64>(c_3);
#pragma clang loop unroll(full)
      for (unsigned k = 1; k < 8; ++k) {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mac_8x8_8x8T(a0, b, c0); c1 = mac_8x8_8x8T(a1, b, c1); c2 = mac_8x8_8x8T(a2, b, c2); c3 = mac_8x8_8x8T(a3, b, c3);
      }
      { aie::accum<accfloat, 64> t; t.from_vector(o0); c0 = aie::add(c0, t); t.from_vector(o1); c1 = aie::add(c1, t);
        t.from_vector(o2); c2 = aie::add(c2, t); t.from_vector(o3); c3 = aie::add(c3, t); }
      aie::store_v(c_0, c0.template to_vector<bfloat16>()); aie::store_v(c_1, c1.template to_vector<bfloat16>());
      aie::store_v(c_2, c2.template to_vector<bfloat16>()); aie::store_v(c_3, c3.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT8U)
  // 4x1: unit = the quarter's 4 A row-blocks x 1 B column-block (4 accumulators, one per row
  // block); A packed [k-block][4 row-blocks] and re-streamed per column, B [j][k-block]
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));
#pragma clang loop unroll_count(2)
    for (unsigned j = 0; j < 16; ++j) {
      // C blocks of column j in the 4 row blocks: C layout [row block][col block][8][8]
      bfloat16 *__restrict c_0 = Cq + (0 * 16 + j) * 64, *__restrict c_1 = Cq + (1 * 16 + j) * 64;
      bfloat16 *__restrict c_2 = Cq + (2 * 16 + j) * 64, *__restrict c_3 = Cq + (3 * 16 + j) * 64;
      aie::accum<accfloat, 64> c0(aie::load_v<64>(c_0)), c1(aie::load_v<64>(c_1)), c2(aie::load_v<64>(c_2)), c3(aie::load_v<64>(c_3));
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap));
#pragma clang loop unroll(full)
      for (unsigned k = 0; k < 8; ++k) {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mac_8x8_8x8T(a0, b, c0); c1 = mac_8x8_8x8T(a1, b, c1); c2 = mac_8x8_8x8T(a2, b, c2); c3 = mac_8x8_8x8T(a3, b, c3);
      }
      aie::store_v(c_0, c0.template to_vector<bfloat16>()); aie::store_v(c_1, c1.template to_vector<bfloat16>());
      aie::store_v(c_2, c2.template to_vector<bfloat16>()); aie::store_v(c_3, c3.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT9)
  // 4x1: unit = the quarter's 4 A row-blocks x 1 B column-block (4 accumulators, one per row
  // block); A packed [k-block][4 row-blocks] and re-streamed per column, B [j][k-block]
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));
    for (unsigned j = 0; j < 16; ++j) {
      // C blocks of column j in the 4 row blocks: C layout [row block][col block][8][8]
      bfloat16 *__restrict c_0 = Cq + (0 * 16 + j) * 64, *__restrict c_1 = Cq + (1 * 16 + j) * 64;
      bfloat16 *__restrict c_2 = Cq + (2 * 16 + j) * 64, *__restrict c_3 = Cq + (3 * 16 + j) * 64;
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap));
      // the unit starts from its first product, not from C: nothing waits on the C loads
      aie::accum<accfloat, 64> c0, c1, c2, c3;
      {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mul_8x8_8x8T(a0, b); c1 = mul_8x8_8x8T(a1, b); c2 = mul_8x8_8x8T(a2, b); c3 = mul_8x8_8x8T(a3, b);
      }
      const aie::vector<bfloat16, 64> o0 = aie::load_v<64>(c_0), o1 = aie::load_v<64>(c_1), o2 = aie::load_v<64>(c_2), o3 = aie::load_v<64>(c_3);
#pragma clang loop unroll(full)
      for (unsigned k = 1; k < 8; ++k) {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mac_8x8_8x8T(a0, b, c0); c1 = mac_8x8_8x8T(a1, b, c1); c2 = mac_8x8_8x8T(a2, b, c2); c3 = mac_8x8_8x8T(a3, b, c3);
      }
      { aie::accum<accfloat, 64> t; t.from_vector(o0); c0 = aie::add(c0, t); t.from_vector(o1); c1 = aie::add(c1, t);
        t.from_vector(o2); c2 = aie::add(c2, t); t.from_vector(o3); c3 = aie::add(c3, t); }
      aie::store_v(c_0, c0.template to_vector<bfloat16>()); aie::store_v(c_1, c1.template to_vector<bfloat16>());
      aie::store_v(c_2, c2.template to_vector<bfloat16>()); aie::store_v(c_3, c3.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_DIT10)
  // dit8 + the next unit's C prefetched into bf16 vector registers during this unit's MACs
  alignas(64) static uint8_t Ap[32 * 64 / 64 * 72];
  alignas(64) static uint8_t Bp[64 * 128 / 64 * 72];
  alignas(64) static bfloat16 Cq[32 * 128 + 4 * 64];
  {
    aie::accum<accfloat, 64> t; t.from_vector(aie::load_v<64>(reinterpret_cast<const bfloat16 *>(in)));
    const auto v = t.template to_vector<bfp16ebs8>();
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> oa(reinterpret_cast<bfp16ebs8 *>(Ap)), ob(reinterpret_cast<bfp16ebs8 *>(Bp));
    for (unsigned i = 0; i < 32; ++i) oa << v;
    for (unsigned i = 0; i < 128; ++i) ob << v;
    for (unsigned i = 0; i < 32 * 128 + 256; i += 64) aie::store_v(Cq + i, aie::zeros<bfloat16, 64>());
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pb(reinterpret_cast<const bfp16ebs8 *>(Bp));
    aie::vector<bfloat16, 64> n0 = aie::load_v<64>(Cq), n1 = aie::load_v<64>(Cq + 16 * 64), n2 = aie::load_v<64>(Cq + 32 * 64), n3 = aie::load_v<64>(Cq + 48 * 64);
    for (unsigned j = 0; j < 16; ++j) {
      bfloat16 *__restrict c_0 = Cq + (0 * 16 + j) * 64, *__restrict c_1 = Cq + (1 * 16 + j) * 64;
      bfloat16 *__restrict c_2 = Cq + (2 * 16 + j) * 64, *__restrict c_3 = Cq + (3 * 16 + j) * 64;
      aie::accum<accfloat, 64> c0, c1, c2, c3;
      c0.from_vector(n0); c1.from_vector(n1); c2.from_vector(n2); c3.from_vector(n3);
      // the next unit's C (j + 1; past the end it reads the 4 spare blocks)
      const unsigned jn = j + 1 < 16 ? j + 1 : 0;
      n0 = aie::load_v<64>(Cq + (0 * 16 + jn) * 64); n1 = aie::load_v<64>(Cq + (1 * 16 + jn) * 64);
      n2 = aie::load_v<64>(Cq + (2 * 16 + jn) * 64); n3 = aie::load_v<64>(Cq + (3 * 16 + jn) * 64);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pa(reinterpret_cast<const bfp16ebs8 *>(Ap));
#pragma clang loop unroll(full)
      for (unsigned k = 0; k < 8; ++k) {
        const auto b = pb.pop();
        const auto a0 = pa.pop(), a1 = pa.pop(), a2 = pa.pop(), a3 = pa.pop();
        c0 = mac_8x8_8x8T(a0, b, c0); c1 = mac_8x8_8x8T(a1, b, c1); c2 = mac_8x8_8x8T(a2, b, c2); c3 = mac_8x8_8x8T(a3, b, c3);
      }
      aie::store_v(c_0, c0.template to_vector<bfloat16>()); aie::store_v(c_1, c1.template to_vector<bfloat16>());
      aie::store_v(c_2, c2.template to_vector<bfloat16>()); aie::store_v(c_3, c3.template to_vector<bfloat16>());
    }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(Cq));
#elif defined(PEAK_Q4U)
  // unpack one dit2 B tile (128 W rows = 2 bands of 64, k-tile ky of 64 k) from the q4_1 pool
  // layout (per band: 2 chunks of 5120 B = 32 rows x 256 k; d bf16[8 kb][32 r] at 0, m at 512,
  // nibbles [2 halves][256 k][16 r] from 1024) into bfp16 blocks in pack_b order [n-block 16]
  // [k-block 8][8 n][8 k]. PEAK_ITERS / 512 tiles.
  alignas(64) static uint8_t pool[2 * 2 * 5120];
  alignas(64) static uint8_t Bs[128 * 64 / 64 * 72];
  for (unsigned i = 0; i < sizeof(pool); i += 64) aie::store_v(reinterpret_cast<int8_t *>(pool + i), aie::load_v<64>(in));
  for (unsigned i = 0; i < 2 * 2; ++i) {   // sane scales: d = 0.01, m = -0.08 (bf16)
    bfloat16 *d = reinterpret_cast<bfloat16 *>(pool + i * 5120), *mm = d + 256;
    for (unsigned j = 0; j < 256; j += 32) { aie::store_v(d + j, aie::broadcast<bfloat16, 32>((bfloat16)0.01f)); aie::store_v(mm + j, aie::broadcast<bfloat16, 32>((bfloat16)-0.08f)); }
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const unsigned ky = 0;
  for (unsigned tile = 0; tile < PEAK_ITERS / 512; ++tile) {
    for (unsigned band = 0; band < 2; ++band)
      for (unsigned g = 0; g < 4; ++g) {          // 16-row group: chunk g / 2, half g % 2
        const uint8_t *chunk = pool + (band * 2 + (g >> 1)) * 5120;
        const unsigned half = g & 1;
        const uint8_t *nibp = chunk + 1024 + half * 2048 + ky * 512;
        const bfloat16 *dp = reinterpret_cast<const bfloat16 *>(chunk) + half * 16;
        const bfloat16 *mp = reinterpret_cast<const bfloat16 *>(chunk + 512) + half * 16;
        const unsigned nblk = (band * 64 + g * 16) / 8;   // the even/odd-zipped pair: n-blocks nblk, nblk + 1
        for (unsigned kb = 0; kb < 2; ++kb) {
          const unsigned kb_abs = ky * 2 + kb;
          const aie::vector<bfloat16, 64> de = mb_scale<false, true>(dp, kb_abs), dod = mb_scale<true, true>(dp, kb_abs);
          const aie::vector<bfloat16, 64> me = mb_scale<false, false>(mp, kb_abs), mo = mb_scale<true, false>(mp, kb_abs);
          for (unsigned il = 0; il < 4; ++il) {
            const aie::vector<uint8_t, 64> q = aie::load_v<64>(nibp + (kb * 4 + il) * 64);
            accN<64> se, so;
            se.from_vector(me); so.from_vector(mo);
            se = aie::mac(se, aie::to_float<bfloat16>(aie::bit_and((uint8_t)0x0F, q), 0), de);
            so = aie::mac(so, aie::to_float<bfloat16>(aie::bit_and((uint8_t)0xF0, q), 0), dod);
            const aie::vector<bfloat16, 64> te = aie::transpose(se.template to_vector<bfloat16>(), 8, 8);
            const aie::vector<bfloat16, 64> to = aie::transpose(so.template to_vector<bfloat16>(), 8, 8);
            const auto z = aie::interleave_zip(te, to, 8);                   // rows 0-7, 8-15
            aie::accum<accfloat, 64> b0, b1;
            b0.from_vector(z.first); b1.from_vector(z.second);
            const unsigned k8 = kb * 4 + il;
            aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o0(reinterpret_cast<bfp16ebs8 *>(Bs) + (nblk * 8 + k8));
            o0 << b0.template to_vector<bfp16ebs8>();
            aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o1(reinterpret_cast<bfp16ebs8 *>(Bs) + ((nblk + 1) * 8 + k8));
            o1 << b1.template to_vector<bfp16ebs8>();
          }
        }
      }
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<int8_t *>(out), aie::load_v<64>(reinterpret_cast<const int8_t *>(Bs)));
#elif defined(PEAK_HS) || defined(PEAK_HSREF)
  hs_seed = 1; hs_fill(hsA, 32, false); hs_fill(hsB, 128, false);
  for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(hsC + i, aie::zeros<bfloat16, 64>());
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call) {
#ifdef PEAK_HS
    hs_call(reinterpret_cast<const bfp16ebs8 *>(hsA), reinterpret_cast<const bfp16ebs8 *>(hsB), hsC);
#else
    hs_ref(reinterpret_cast<const bfp16ebs8 *>(hsA), reinterpret_cast<const bfp16ebs8 *>(hsB), hsC);
#endif
  }
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(hsC));
#elif defined(PEAK_HSCHK)
  // out[0] mismatching bf16 values after 3 calls (hand vs stock), out[1] first index,
  // out[2..3] the two values there (bits), out[4] nonzero count, out[64..] the first 64 hs values
  hs_seed = 7 + in[0] + 256 * in[1];
  hs_fill(hsA, 32, true); hs_fill(hsB, 128, true);
  for (unsigned i = 0; i < 32 * 128; ++i) { hsC[i] = (bfloat16)hs_rnd(); hsC2[i] = hsC[i]; }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned c = 0; c < 3; ++c) {
    hs_ref(reinterpret_cast<const bfp16ebs8 *>(hsA), reinterpret_cast<const bfp16ebs8 *>(hsB), hsC);
    hs_call(reinterpret_cast<const bfp16ebs8 *>(hsA), reinterpret_cast<const bfp16ebs8 *>(hsB), hsC2);
  }
  aie::set_rounding(saved);
  const uint16_t *r = reinterpret_cast<const uint16_t *>(hsC), *h = reinterpret_cast<const uint16_t *>(hsC2);
  int bad = 0, first = -1, nz = 0;
  for (unsigned i = 0; i < 32 * 128; ++i) {
    if (r[i] != h[i]) { if (first < 0) first = i; ++bad; }
    if (h[i] != 0) ++nz;
  }
  out[0] = bad; out[1] = first; out[2] = first >= 0 ? r[first] : 0; out[3] = first >= 0 ? h[first] : 0; out[4] = nz;
  for (unsigned i = 0; i < 64; ++i) out[64 + i] = h[i];
  for (unsigned i = 0; i < 64; ++i) out[128 + i] = r[i];
#elif defined(PEAK_B4) || defined(PEAK_B4HS)
  hs_seed = 3; hs_fill(hsA, 32, false);
  for (unsigned i = 0; i < 128 * 32; ++i) b4N[i] = (uint8_t)(i * 37 + 11);
  for (unsigned i = 0; i < 16 * 8; ++i) b4E[i] = (int8_t)(120 + i % 5);
  for (unsigned i = 0; i < 32 * 128; i += 64) aie::store_v(hsC + i, aie::zeros<bfloat16, 64>());
  b4_reorder_a();
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned call = 0; call < PEAK_ITERS / 512; ++call)
#ifdef PEAK_B4HS
    hs_b4_call(reinterpret_cast<const bfp16ebs8 *>(hsA), b4N, hsC, b4E);
#else
    b4_call(reinterpret_cast<const bfp16ebs8 *>(hsA), b4N, b4E, hsC);
#endif
  aie::set_rounding(saved);
  aie::store_v(reinterpret_cast<bfloat16 *>(out), aie::load_v<64>(hsC));
#elif defined(PEAK_B4CHK) || defined(PEAK_B4HSCHK)
  // out[0] mismatches after 3 calls: b4_call vs hs_ref on the same weights as bfp16 blocks
  hs_seed = 11 + in[0];
  hs_fill(hsA, 32, true);
  for (unsigned i = 0; i < 128 * 32; ++i) { hs_rnd(); b4N[i] = (uint8_t)(hs_seed >> 13); }
  for (unsigned i = 0; i < 16 * 8; ++i) { hs_rnd(); b4E[i] = (int8_t)(118 + (hs_seed >> 20) % 8); }
#ifdef PEAK_B4HSDIAG   // A block (pair, k) = 2^k, mantissas 1, one exponent, C = 0: an output / its
  // reference x 255 is the bitmask of the k-steps that reached it
  {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o(reinterpret_cast<bfp16ebs8 *>(hsA));
    for (unsigned b = 0; b < 32; ++b) {
      aie::vector<bfloat16, 64> v = aie::broadcast<bfloat16, 64>((bfloat16)(float)(1u << ((b / 2) % 8)));
      aie::accum<accfloat, 64> t; t.from_vector(v);
      o << t.template to_vector<bfp16ebs8>();
    }
  }
  for (unsigned i = 0; i < 128 * 32; ++i) b4N[i] = 0x11;
  for (unsigned i = 0; i < 16 * 8; ++i) b4E[i] = 120;
#endif
  b4_expand(hsB);
  b4_reorder_a();
  for (unsigned i = 0; i < 32 * 128; ++i) {
#ifdef PEAK_B4HSDIAG
    hsC[i] = (bfloat16)0.f;
#else
    hsC[i] = (bfloat16)hs_rnd();
#endif
    hsC2[i] = hsC[i];
  }
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned c = 0; c < 3; ++c) {
    hs_ref(reinterpret_cast<const bfp16ebs8 *>(hsA), reinterpret_cast<const bfp16ebs8 *>(hsB), hsC);
#ifdef PEAK_B4HSCHK
    hs_b4_call(reinterpret_cast<const bfp16ebs8 *>(hsA), b4N, hsC2, b4E);
#else
    b4_call(reinterpret_cast<const bfp16ebs8 *>(hsA), b4N, b4E, hsC2);
#endif
  }
  aie::set_rounding(saved);
  const uint16_t *r = reinterpret_cast<const uint16_t *>(hsC), *h = reinterpret_cast<const uint16_t *>(hsC2);
  int bad = 0, first = -1, nz = 0;
  for (unsigned i = 0; i < 32 * 128; ++i) {
    if (r[i] != h[i]) { if (first < 0) first = i; ++bad; }
    if (h[i] != 0) ++nz;
  }
  out[0] = bad; out[1] = first; out[2] = first >= 0 ? r[first] : 0; out[3] = first >= 0 ? h[first] : 0; out[4] = nz;
#ifdef PEAK_B4HSDIAG
  for (unsigned i = 0; i < 64; ++i) { out[64 + i] = h[i * 64]; out[128 + i] = r[i * 64]; }   // one value per 8x8 block
#else
  for (unsigned i = 0; i < 64; ++i) { out[64 + i] = h[i]; out[128 + i] = r[i]; }
#endif
#elif defined(PEAK_FAQKT)
  for (unsigned it = 0; it < PEAK_ITERS; ++it) fa_qk_call(ftA, ftB, ftC, s_nuc);
  out[0] = reinterpret_cast<int32_t *>(ftC)[0];
#elif defined(PEAK_FAPVT)
  for (unsigned it = 0; it < PEAK_ITERS; ++it) fa_pv_call(ftA, ftB, ftC);
  out[0] = reinterpret_cast<int32_t *>(ftC)[0];
#elif defined(PEAK_FAEXP)
  for (int i = 0; i < TQ * LKP; ++i)
    fxG[i] = (bfloat16)(-float((i * 37 + in[i % 64]) % 113) / 14.0f);   // scores - m in (-8, 0]
  fxG[5] = bf16_lowest(); fxG[700] = bf16_lowest();                    // two masked
  for (int i = 0; i < 2304 + 128; ++i) fxPa[i] = fxPc[i] = 0xAB;
  for (int i = 0; i < 256 + 32; ++i) fxRa[i] = fxRc[i] = -7.0f;
  {
    Acc oa;
    oa.from_vector(aie::broadcast<float, 64>(1.0f));
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> ones(reinterpret_cast<bfp16ebs8 *>(s_ones));
    ones << oa.template to_vector<bfp16ebs8>();
  }
  aie::set_rounding(aie::rounding_mode::conv_even);
  for (int zp = 0; zp < 2; zp++)
    fa_exp_call(fxG + 2 * zp * 64, s_ones, fxPa + zp * 1152, fxRa + 2 * zp * 64);
  exp_p_rowsum(fxG, fxPc, fxRc);
  int nd = 0, fd = -1, ng = 0, nr = 0, nrd = 0, frd = -1;
  for (int i = 0; i < 2304; ++i) if (fxPa[i] != fxPc[i]) { if (fd < 0) fd = i; ++nd; }
  for (int i = 2304; i < 2304 + 128; ++i) ng += fxPa[i] != 0xAB;
  for (int i = 256; i < 256 + 32; ++i) nr += fxRa[i] != -7.0f;
  for (int i = 0; i < 256; ++i) {
    float d = fxRa[i] - fxRc[i]; if (d < 0) d = -d;
    float r = fxRc[i] < 0 ? -fxRc[i] : fxRc[i];
    if (!(d <= 1e-3f * r + 1e-6f)) { if (frd < 0) frd = i; ++nrd; }
  }
  int tm = 0;
  for (int t = 0; t < 32; ++t)
    for (int i = 0; i < 72; ++i)
      if (fxPa[t * 72 + i] != fxPc[t * 72 + i]) { tm |= 1 << t; break; }
  out[0] = nd; out[1] = fd; out[2] = ng; out[3] = nr; out[4] = nrd; out[5] = frd; out[6] = tm;
  for (int i = 0; i < 8; ++i) { out[8 + i] = reinterpret_cast<int32_t *>(fxRa)[i]; out[16 + i] = reinterpret_cast<int32_t *>(fxRc)[i]; }
  for (int i = 0; i < 8; ++i) out[24 + i] = reinterpret_cast<int32_t *>(fxPa)[i];
  for (int i = 0; i < 8; ++i) out[32 + i] = reinterpret_cast<int32_t *>(fxPc)[i];
  for (int i = 0; i < 18; ++i) out[64 + i] = reinterpret_cast<int32_t *>(fxPa + 144)[i];   // tile 2
  for (int i = 0; i < 18; ++i) out[96 + i] = reinterpret_cast<int32_t *>(fxPc + 144)[i];
#elif defined(PEAK_B4PROBE)
  // nibbles: A all 1, B old all 1, B new all 2; exponents: A 127+..., B old e0, B new e1 (each
  // block's 8 groups the same). out[i] = c[0] of case i (bf16 bits)
  for (unsigned i = 0; i < 32; ++i) { b4pN[i] = 0x11; b4pN[32 + i] = 0x11; b4pN[64 + i] = 0x22; }
  const int32_t ea = 0x7a7a7a7a, eo = 0x76767676, en = 0x7e7e7e7e;
  b4pE[0] = ea; b4pE[1] = ea; b4pE[2] = eo; b4pE[3] = eo; b4pE[4] = en; b4pE[5] = en;
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned c = 0; c < b4p_n; ++c) {
    for (unsigned i = 0; i < 64; ++i) b4pC[i] = (bfloat16)0.f;
    b4p_cases[c](b4pN, b4pE, b4pC);
    out[c] = reinterpret_cast<const uint16_t *>(b4pC)[0];
  }
  aie::set_rounding(saved);
#elif defined(PEAK_B4PROBE2)
  // A blocks 0 / 1 / 2 = 1, 2, 4 (bf16 -> bfp16); weights all mantissa 1; out[2 c] / out[2 c + 1] = c[0] / c[64]
  {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> o(reinterpret_cast<bfp16ebs8 *>(b4qA));
    for (unsigned b = 0; b < 3; ++b) {
      aie::accum<accfloat, 64> t; t.from_vector(aie::broadcast<bfloat16, 64>((bfloat16)(float)(1u << b)));
      o << t.template to_vector<bfp16ebs8>();
    }
  }
  for (unsigned i = 0; i < 64; ++i) b4qN[i] = 0x11;
  b4qE[0] = 0x78787878; b4qE[1] = 0x78787878;
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned c = 0; c < b4q_n; ++c) {
    for (unsigned i = 0; i < 128; ++i) b4qC[i] = (bfloat16)0.f;
    b4q_cases[c](reinterpret_cast<const bfp16ebs8 *>(b4qA), b4qN, b4qE, b4qC);
    out[2 * c] = reinterpret_cast<const uint16_t *>(b4qC)[0];
    out[2 * c + 1] = reinterpret_cast<const uint16_t *>(b4qC)[64];
  }
  aie::set_rounding(saved);
#endif
}
}
