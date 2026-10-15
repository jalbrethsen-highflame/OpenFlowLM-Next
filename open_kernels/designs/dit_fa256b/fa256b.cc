//===- fa256b.cc ---------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Compute kernels for dit_fa256b (dit_fa256b.py): d3's noncausal flash attention at head
// dim 256 with Q, K, V and P in bfp16ebs8 on the native 8x8x8 block-float MAC
// (mac_8x8_8x8T), instead of dit_fa256's bf16 operands converted to bfp16 inside every
// mmul call. The host packs Q, K and V in the order the MAC pops them, so the memtile only
// copies. The softmax is dit_fa256's (fa256.cc, kept below unchanged).
//
//   Q   per core: [row pair 2][d step 32][A0, A1]  -- 8 rows x 8 d tiles (block: a row's 8 d)
//   K   per 16-key piece j: [d step 32][B0, B1]    -- 8 keys x 8 d tiles (block: a key's 8 d)
//   V   per 16-column piece: [key step 8][B0, B1]  -- 8 d cols x 8 keys (block: a col's 8 keys)
//   P   per chunk: [row pair 2][key step 8][A0, A1] -- 8 rows x 8 keys (block: a row's 8 keys)
//   G   (scores, bf16) keeps fa256.cc's [col block][row block][8][8] layout; O (bf16) is
//       [row block][col block][8][8] (the GEMM's C order, so fa_pv_call walks it linearly).
//   FA_HS=1: qk / pv run hsgen_fa.py's hand-scheduled calls (fa_hs.inc), else the C++ below.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

#include <aie_api/aie.hpp>

// Compile-time shape (dit_fa.py passes FA_*; bare names like DC collide with
// aie_api identifiers when given as -D macros).
#ifndef FA_TQ
#define FA_TQ 32 // query rows per core
#endif
#ifndef FA_LKP
#define FA_LKP 64 // keys per chunk
#endif
#ifndef FA_DC
#define FA_DC 32 // head-dim chunk: the K/V/Q piece one mmul call consumes
#endif
#ifndef FA_DFULL
#define FA_DFULL 256 // head dimension
#endif
#ifndef FA_QROWS_PASS
#define FA_QROWS_PASS 512 // query rows a group covers per pass (cores x TQ)
#endif
constexpr int TQ = FA_TQ;
constexpr int LKP = FA_LKP;
constexpr int DC = FA_DC;
constexpr int DFULL = FA_DFULL;
constexpr int QROWS_PASS = FA_QROWS_PASS;

constexpr int NDCH = DFULL / DC;
static_assert(DFULL % DC == 0, "head dim must be whole chunks");
static_assert(TQ % 32 == 0, "row loops load 32 rows at a time");
static_assert(LKP % 16 == 0 && DC % 16 == 0, "mmul 2x2 blocking");
// A Q block (two cores' rows) is the size of a K block -- they share a fifo.

constexpr int G_ELEMS = TQ * LKP;
constexpr int GP_ELEMS = TQ * DFULL;
constexpr int BLOCK_STRIDE = TQ * 8; // between 8-column blocks, all tiles here

// scale: log2(e) / sqrt(DFULL), applied inside the exponentials.
constexpr double sqrt_d = (DFULL == 64)    ? 8.0
                          : (DFULL == 128) ? 11.313708498984761
                          : (DFULL == 256) ? 16.0
                          : (DFULL == 512) ? 22.627416997969522
                                           : 0.0;
static_assert(sqrt_d != 0.0, "add sqrt(DFULL) for this head dim");
// exp_fix's quadratic (see there)
#define FA_EXP_C0 1.4406570f
#define FA_EXP_C1 -0.671875f
#define FA_EXP_C2 0.2275390625f
#define log2e (1.44269504089 / sqrt_d)

static inline bfloat16 bf16_lowest() {
  uint16_t u = 0xff7f;
  return *(bfloat16 *)&u;
}
static inline bfloat16 bf16_neg_inf() {
  uint16_t u = 0xff80;
  return *(bfloat16 *)&u;
}

// attn_npu2.cc's exp2f_poly16 (from mlir-aie aie_kernels/aie2p/exp2f_vec.cc,
// Apache-2.0 WITH LLVM-exception): aie::exp2<float> does not exist on aie2p,
// and the bf16 LUT is 6-49% off on the rescale factor's domain.
static __attribute__((noinline)) aie::vector<float, 16>
exp2f_poly16(aie::vector<float, 16> x) {
  constexpr int N = 16;
  x = aie::max(x, aie::broadcast<float, N>(-111.0f));
  aie::mask<N> overflow = aie::ge(x, aie::broadcast<float, N>(128.0f));
  x = aie::min(x, aie::broadcast<float, N>(127.999f));
  aie::vector<int32_t, N> ki = aie::to_fixed<int32_t>(x);
  aie::vector<float, N> kf = aie::to_float<float>(ki);
  aie::vector<int32_t, N> one = aie::broadcast<int32_t, N>(1);
  aie::vector<int32_t, N> zero = aie::broadcast<int32_t, N>(0);
  ki = aie::sub(ki, aie::select(zero, one, aie::lt(x, kf)));
  aie::vector<float, N> f = aie::sub(x, aie::to_float<float>(ki));
  aie::vector<float, N> p = aie::broadcast<float, N>(0.0013333558f);
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.0096181291f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.0555041087f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.2402265069f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(0.6931471805f));
  p = aie::add(aie::mul(p, f).to_vector<float>(), aie::broadcast<float, N>(1.0f));
  aie::vector<int32_t, N> ebits =
      aie::upshift(aie::add(ki, aie::broadcast<int32_t, N>(127)), 23);
  aie::vector<float, N> p2k = ebits.template cast_to<float>();
  aie::vector<float, N> result = aie::mul(p, p2k).to_vector<float>();
  aie::vector<int32_t, N> pos_inf_bits = aie::broadcast<int32_t, N>(0x7f800000);
  aie::vector<float, N> pos_inf = pos_inf_bits.template cast_to<float>();
  return aie::select(result, pos_inf, overflow);
}

// Row max of G ([TQ, LKP], 8x8-block column-major) into out[TQ].
static inline void row_max(const bfloat16 *__restrict in, bfloat16 *__restrict out) {
  constexpr int col_blocks = LKP / 8;
  constexpr int row_blocks = TQ / 8;
  aie::vector<bfloat16, 64> lowest_vec = aie::broadcast<bfloat16, 64>(bf16_lowest());
  for (int rb = 0; rb < row_blocks; rb++) {
    const bfloat16 *__restrict p = in + rb * 64;
    aie::vector<bfloat16, 64> m = lowest_vec;
    for (int cb = 0; cb < col_blocks; cb++)
      chess_prepare_for_pipelining chess_loop_range(8, ) {
        m = aie::max(m, aie::load_v<64>(p + cb * BLOCK_STRIDE));
      }
    aie::vector<bfloat16, 64> t = aie::transpose(m, 8, 8);
    aie::vector<bfloat16, 32> a = aie::max(t.extract<32>(0), t.extract<32>(1));
    aie::vector<bfloat16, 16> b = aie::max(a.extract<16>(0), a.extract<16>(1));
    aie::vector<bfloat16, 8> c = aie::max(b.extract<8>(0), b.extract<8>(1));
    aie::store_v(out + rb * 8, c);
  }
}

// Replicate each of 32 per-row values 8 times, in the 4-row x 8-lane groups a
// 32-lane load of an 8x8 block covers: rep[g*32 + row_in_group*8 + lane].
static inline void replicate_rows32(const bfloat16 *__restrict v32,
                                    bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  V uv = aie::load_v<32>(v32);
  auto z1 = aie::interleave_zip(uv, uv, 1);
  auto z2a = aie::interleave_zip(z1.first, z1.first, 2);
  auto z2b = aie::interleave_zip(z1.second, z1.second, 2);
  auto z3a = aie::interleave_zip(z2a.first, z2a.first, 4);
  auto z3b = aie::interleave_zip(z2a.second, z2a.second, 4);
  auto z3c = aie::interleave_zip(z2b.first, z2b.first, 4);
  auto z3d = aie::interleave_zip(z2b.second, z2b.second, 4);
  aie::store_v(rep + 0, z3a.first);
  aie::store_v(rep + 32, z3a.second);
  aie::store_v(rep + 64, z3b.first);
  aie::store_v(rep + 96, z3b.second);
  aie::store_v(rep + 128, z3c.first);
  aie::store_v(rep + 160, z3c.second);
  aie::store_v(rep + 192, z3d.first);
  aie::store_v(rep + 224, z3d.second);
}

#ifndef FA_EXP_FIX
#define FA_EXP_FIX 0
#endif

// The hardware exp2 (aie::exp2<bfloat16>) returns 2^n * (1 + f), f the input's
// fraction truncated to 7 bits: a linear mantissa, +3.8% mean / +6.1% worst
// (utilities/aie-probes/exp2_probe.py). FA_EXP_FIX multiplies by h(1 + f) ~
// 2^(f + 1/256) / (1 + f), a quadratic in the returned mantissa: rms error 3e-3
// on [-2, 0]. Coefficients: fa_emul.py's EXP_FIX (bf16 c1/c2, fp32 c0).
static inline __attribute__((always_inline)) aie::vector<bfloat16, 32>
exp_fix(aie::vector<bfloat16, 32> y) {
  aie::vector<int16_t, 32> yi = y.template cast_to<int16_t>();
  aie::vector<int16_t, 32> mi =
      aie::bit_or(aie::bit_and(yi, aie::broadcast<int16_t, 32>(0x007F)),
                  aie::broadcast<int16_t, 32>(0x3F80));
  aie::vector<bfloat16, 32> m = mi.template cast_to<bfloat16>();
  aie::vector<bfloat16, 32> m2 = aie::mul(m, m).template to_vector<bfloat16>();
  aie::accum<accfloat, 32> h;
  h.from_vector(aie::broadcast<float, 32>(FA_EXP_C0));
  h = aie::mac(h, m, aie::broadcast<bfloat16, 32>((bfloat16)FA_EXP_C1));
  h = aie::mac(h, m2, aie::broadcast<bfloat16, 32>((bfloat16)FA_EXP_C2));
  return aie::mul(y, h.template to_vector<bfloat16>()).template to_vector<bfloat16>();
}

// Multiply 32 rows of a block-column-major tile with TQ rows and ncols columns
// by rep's per-row factors (replicate_rows32).
template <int ncols>
static inline void scale_rows(bfloat16 *__restrict x, const bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  constexpr int col_blocks = ncols / 8;
  constexpr int chunk = col_blocks < 8 ? col_blocks : 8;
  static_assert(col_blocks % chunk == 0);
  for (int gi = 0; gi < 8; gi++) {
    V rv = aie::load_v<32>(rep + gi * 32);
    bfloat16 *__restrict p = x + gi * 32;
#pragma clang loop unroll(disable)
    for (int c0 = 0; c0 < col_blocks; c0 += chunk) {
      V v[chunk];
#pragma clang loop unroll(full)
      for (int cb = 0; cb < chunk; cb++)
        v[cb] = aie::load_v<32>(p + (c0 + cb) * BLOCK_STRIDE);
#pragma clang loop unroll(full)
      for (int cb = 0; cb < chunk; cb++)
        v[cb] = aie::mul(v[cb], rv).template to_vector<bfloat16>();
#pragma clang loop unroll(full)
      for (int cb = 0; cb < chunk; cb++)
        aie::store_v(p + (c0 + cb) * BLOCK_STRIDE, v[cb]);
    }
  }
}

// O [row block][col block][8][8] times rep's per-row factors (replicate_rows32 layout).
static inline void scale_o(bfloat16 *__restrict o, const bfloat16 *__restrict rep) {
  using V = aie::vector<bfloat16, 32>;
  for (int gi = 0; gi < TQ / 4; gi++) {                 // rows 4gi .. 4gi + 3
    V rv = aie::load_v<32>(rep + gi * 32);
    bfloat16 *__restrict p = o + (gi / 2) * (DFULL / 8) * 64 + (gi % 2) * 32;
    for (int c0 = 0; c0 < DFULL / 8; c0 += 8) {
      V v[8];
#pragma clang loop unroll(full)
      for (int cb = 0; cb < 8; cb++)
        v[cb] = aie::load_v<32>(p + (c0 + cb) * 64);
#pragma clang loop unroll(full)
      for (int cb = 0; cb < 8; cb++)
        aie::store_v(p + (c0 + cb) * 64, aie::mul(v[cb], rv).template to_vector<bfloat16>());
    }
  }
}

// Per-row values are kept "replicated": row r's value in lanes r*8 .. r*8+7, the
// layout of an 8-column mmul tile and of replicate_rows32's table.
alignas(64) static bfloat16 s_rep[TQ * 8];   // scratch: a per-row bf16 value, replicated
alignas(64) static float s_nuc[TQ * 8];      // -m*c, replicated (exp argument offset)
alignas(64) static bfloat16 s_newmax[TQ];
alignas(64) static bfloat16 s_resc[TQ];
alignas(64) static float s_rs[TQ * 8];        // this chunk's row sums, replicated
alignas(64) static float s_tmp[TQ * 8];

// Lazy rescale: the reference max m moves only when a row's chunk max exceeds it
// by more than TAU (log2 units), so P = exp2((s - m)c) <= 2^TAU (bf16 O and fp32 l
// have the range; bfp16's shared exponent is relative) and the O / l rescale runs on
// a handful of chunks per pass, not all. The rescale's frequency depends on the data:
// on wide score spreads (make_test.py --qk-scale 2-3) TAU 8 ran ~20% slower and TAU 32
// doesn't. klein's real scores rarely rescale at either (<1% of tile-chunk pairs at
// TAU 8, utilities/dit-chain/attn_rescales.py), so for klein the change is small.
#ifndef FA_TAU
#define FA_TAU 32
#endif
constexpr float TAU = float(FA_TAU);

// dst[i] (fp32) = bf16 src[i] * k, n values.
static inline void widen_scale(const bfloat16 *__restrict src, float *__restrict dst,
                               float k, int n) {
  for (int i = 0; i < n; i += 16) {
    aie::accum<accfloat, 16> a;
    a.from_vector(aie::load_v<16>(src + i));
    aie::store_v(dst + i,
                 aie::mul(a.template to_vector<float>(), k).template to_vector<float>());
  }
}

// rtp: dit_fa256.py RTP_* -- [heads, passes, chunks, valid_len @3, causal @4, the core's
// first row within a pass @5, the call's query offset @6]. Absolute row of this core's
// first query row in pass `pass`:
static inline int32_t fa_row0(const int32_t *rtp, int32_t pass) {
  return rtp[6] + pass * QROWS_PASS + rtp[5];
}
// A key chunk every one of this core's TQ rows masks entirely (causal): nothing to compute.
static inline bool fa_skip(const int32_t *rtp, int32_t pass, int32_t chunk) {
  return rtp[4] && chunk * LKP > fa_row0(rtp, pass) + TQ - 1;
}

// A key chunk wholly at or past valid_len (the pad up to lk): nothing to compute.
static inline bool fa_dead(const int32_t *rtp, int32_t chunk) { return chunk * LKP >= rtp[3]; }

using BV = aie::block_vector<bfp16ebs8, 64>;
using BIn = aie::block_vector_input_buffer_stream<bfp16ebs8, 64>;
using Acc = aie::accum<accfloat, 64>;
using Acc16 = aie::accum<accfloat, 16>;
constexpr int TILE_B = 72;                           // one 8x8 bfp16ebs8 tile
constexpr int Q_BYTES = (TQ / 16) * (DFULL / 8) * 2 * TILE_B;      // 9216
constexpr int KPIECE_BYTES = (DFULL / 8) * 2 * TILE_B;             // 16 keys: 4608
constexpr int VPAIR_BYTES = (LKP / 8) * 2 * TILE_B;                // 16 d columns: 1152
constexpr int VPAIRS_PER_ELEM = KPIECE_BYTES / VPAIR_BYTES;        // 4
static_assert(TQ == 32 && LKP == 64, "pop orders written for 4 row blocks and 8 key blocks");

static inline const bfp16ebs8 *bfp(const uint8_t *p) { return reinterpret_cast<const bfp16ebs8 *>(p); }

// The softmax's common path, one pass over G's tiles in pv's pop order ([row pair][key step]
// [A0, A1]): P = exp2(G) -- G is already score - m in log2 units (Q packed pre-scaled by
// log2(e) / sqrt(d), pack_fa.QSCALE; qk's chains start from -m) -- written as bfp16, and
// rs = rowsum(P) from those bfp16 tiles on the MAC (P x ones: every column a row sum, which is
// the replicated layout). G is not written back.
static inline void exp_p_rowsum(const bfloat16 *__restrict g, uint8_t *__restrict p,
                                float *__restrict rs) {
  aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(reinterpret_cast<bfp16ebs8 *>(p));
  Acc oa;
  oa.from_vector(aie::broadcast<float, 64>(1.0f));
  const BV ones = oa.template to_vector<bfp16ebs8>();
  for (int zp = 0; zp < TQ / 16; zp++) {
    Acc s0 = aie::zeros<accfloat, 64>(), s1 = aie::zeros<accfloat, 64>();
    for (int i = 0; i < LKP / 8; i++) {
#pragma clang loop unroll(full)
      for (int zz = 0; zz < 2; zz++) {
        const bfloat16 *__restrict t = g + (i * (TQ / 8) + 2 * zp + zz) * 64;
        aie::vector<bfloat16, 64> e;
#pragma clang loop unroll(full)
        for (int q = 0; q < 4; q++) {
          Acc16 a;
          a.from_vector(aie::load_v<16>(t + 16 * q));
          e.insert(q, aie::exp2<bfloat16>(a.template to_vector<float>()));
        }
        Acc c;
        c = e;
        const BV pb = c.template to_vector<bfp16ebs8>();
        out << pb;
        if (zz == 0)
          s0 = mac_8x8_8x8T(pb, ones, s0);
        else
          s1 = mac_8x8_8x8T(pb, ones, s1);
      }
    }
    aie::store_v(rs + (2 * zp) * 64, s0.template to_vector<float>());
    aie::store_v(rs + (2 * zp + 1) * 64, s1.template to_vector<float>());
  }
}

#ifndef FA_HS
#define FA_HS 0
#endif
#if FA_HS
#include "fa_hs.inc"
#endif

extern "C" {

#define SET_ROUNDING() ::aie::set_rounding(::aie::rounding_mode::conv_even)

// Q capture: this core's 9216 bytes arrive as two 4608-byte pieces (k = 0, 1).
void capture_q(uint8_t *__restrict src, uint8_t *__restrict dst, int32_t k) {
  uint8_t *__restrict d = dst + k * KPIECE_BYTES;
  for (int i = 0; i < KPIECE_BYTES; i += 64)
    aie::store_v(d + i, aie::load_v<64>(src + i));
}

// Start of a pass: zero the output accumulator, reset the running max/sum.
// sp = l, replicated ([TQ * 8] fp32).
void pass_init(bfloat16 *__restrict gp, bfloat16 *__restrict up, float *__restrict sp) {
  SET_ROUNDING();
  aie::vector<bfloat16, 32> z = aie::zeros<bfloat16, 32>();
  for (int i = 0; i < GP_ELEMS; i += 32)
    aie::store_v(gp + i, z);
  aie::vector<bfloat16, 32> lo = aie::broadcast<bfloat16, 32>(bf16_lowest());
  for (int i = 0; i < TQ; i += 32)
    aie::store_v(up + i, lo);
  aie::vector<float, 16> zf = aie::zeros<float, 16>();
  for (int i = 0; i < TQ * 8; i += 16) {
    aie::store_v(sp + i, zf);
    aie::store_v(s_nuc + i, zf);                // m = 0 until the first chunk sets it
  }
}

// G[:, keys 16j .. 16j + 15] = Q K_piece^T over the whole head dim, written once (bf16).
void qk(uint8_t *__restrict q, uint8_t *__restrict kp, bfloat16 *__restrict g, int32_t j,
        int32_t *__restrict rtp, int32_t pass, int32_t chunk) {
  if (fa_skip(rtp, pass, chunk) || fa_dead(rtp, chunk))
    return;
  SET_ROUNDING();
#if FA_HS
  fa_qk_call(q, kp, g + j * 2 * (TQ / 8) * 64, s_nuc);
  return;
#endif
  for (int zp = 0; zp < TQ / 16; zp++) {
    BIn pa(bfp(q));
    pa.seek(zp * (DFULL / 8) * 2);
    BIn pb(bfp(kp));
    Acc c00, c01, c10, c11;                     // from the rows' -m
    c00.from_vector(aie::load_v<64>(s_nuc + (2 * zp) * 64));
    c01 = c00;
    c10.from_vector(aie::load_v<64>(s_nuc + (2 * zp + 1) * 64));
    c11 = c10;
    for (int i = 0; i < DFULL / 8; i++) {
      const BV a0 = pa.pop(), a1 = pa.pop(), b0 = pb.pop(), b1 = pb.pop();
      c00 = mac_8x8_8x8T(a0, b0, c00);
      c01 = mac_8x8_8x8T(a0, b1, c01);
      c10 = mac_8x8_8x8T(a1, b0, c10);
      c11 = mac_8x8_8x8T(a1, b1, c11);
    }
    const int z0 = 2 * zp, cb0 = 2 * j;
    aie::store_v(g + (cb0 * (TQ / 8) + z0) * 64, c00.template to_vector<bfloat16>());
    aie::store_v(g + (cb0 * (TQ / 8) + z0 + 1) * 64, c10.template to_vector<bfloat16>());
    aie::store_v(g + ((cb0 + 1) * (TQ / 8) + z0) * 64, c01.template to_vector<bfloat16>());
    aie::store_v(g + ((cb0 + 1) * (TQ / 8) + z0 + 1) * 64, c11.template to_vector<bfloat16>());
  }
}

// O[:, d columns 64e .. 64e + 63] += P V for one 4608-byte V element (4 column pairs).
void pv(uint8_t *__restrict p, uint8_t *__restrict ve, bfloat16 *__restrict o, int32_t e,
        int32_t *__restrict rtp, int32_t pass, int32_t chunk) {
  if (fa_skip(rtp, pass, chunk) || fa_dead(rtp, chunk))
    return;
  SET_ROUNDING();
#if FA_HS
  fa_pv_call(p, ve, o + e * 2 * VPAIRS_PER_ELEM * 64);
  return;
#endif
  for (int jl = 0; jl < VPAIRS_PER_ELEM; jl++) {
    const int cb0 = 2 * (e * VPAIRS_PER_ELEM + jl);
    for (int zp = 0; zp < TQ / 16; zp++) {
      const int z0 = 2 * zp;
      bfloat16 *__restrict t00 = o + (z0 * (DFULL / 8) + cb0) * 64, *__restrict t01 = t00 + 64;
      bfloat16 *__restrict t10 = t00 + (DFULL / 8) * 64, *__restrict t11 = t10 + 64;
      Acc c00(aie::load_v<64>(t00)), c01(aie::load_v<64>(t01)), c10(aie::load_v<64>(t10)), c11(aie::load_v<64>(t11));
      BIn pa(bfp(p));
      pa.seek(zp * (LKP / 8) * 2);
      BIn pb(bfp(ve + jl * VPAIR_BYTES));
      for (int i = 0; i < LKP / 8; i++) {
        const BV a0 = pa.pop(), a1 = pa.pop(), b0 = pb.pop(), b1 = pb.pop();
        c00 = mac_8x8_8x8T(a0, b0, c00);
        c01 = mac_8x8_8x8T(a0, b1, c01);
        c10 = mac_8x8_8x8T(a1, b0, c10);
        c11 = mac_8x8_8x8T(a1, b1, c11);
      }
      aie::store_v(t00, c00.template to_vector<bfloat16>());
      aie::store_v(t01, c01.template to_vector<bfloat16>());
      aie::store_v(t10, c10.template to_vector<bfloat16>());
      aie::store_v(t11, c11.template to_vector<bfloat16>());
    }
  }
}

// Key mask for one chunk: column j of chunk `chunk` is masked for query row r
// of pass `pass` when j >= valid_len, or (causal) when j > r. rtp = the core's
// runtime parameters (dit_fa.py RTP_*): [.., valid_len @3, causal @4, the
// core's first row within a pass @5].
void apply_mask(bfloat16 *__restrict g, int32_t *__restrict rtp, int32_t pass,
                int32_t chunk) {
  if (fa_skip(rtp, pass, chunk) || fa_dead(rtp, chunk))
    return;
  SET_ROUNDING();
  const int32_t valid_len = rtp[3];
  const int32_t causal = rtp[4];
  const int32_t row0 = fa_row0(rtp, pass);
  const int32_t col0 = chunk * LKP;
  // Last column any row may keep; all rows keep everything up to it?
  int32_t keep_all = valid_len; // columns < keep_all kept by every row
  if (causal && row0 + 1 < keep_all)
    keep_all = row0 + 1;
  if (col0 + LKP <= keep_all)
    return;
  int32_t keep_any = valid_len; // columns >= keep_any masked for every row
  if (causal && row0 + TQ < keep_any)
    keep_any = row0 + TQ;
  // bf16 lowest, not -inf: the exp argument is formed as G*c - m*c, and
  // -inf*c - m*c must stay finite (it underflows to exactly 0 in exp2).
  const bfloat16 ninf = bf16_lowest();
  if (col0 >= keep_any) {
    aie::vector<bfloat16, 32> v = aie::broadcast<bfloat16, 32>(ninf);
    for (int i = 0; i < G_ELEMS; i += 32)
      aie::store_v(g + i, v);
    return;
  }
  // Ragged: per element.
  for (int row = 0; row < TQ; row++) {
    int32_t bound = valid_len;
    if (causal && row0 + row + 1 < bound)
      bound = row0 + row + 1;
    bfloat16 *__restrict p = g + (row / 8) * 64 + (row % 8) * 8;
    for (int j = 0; j < LKP; j++)
      if (col0 + j >= bound)
        p[(j / 8) * BLOCK_STRIDE + (j % 8)] = ninf;
  }
}

// One key chunk of the online softmax, after G holds this chunk's scores
// (log2(e)/sqrt(d) = c is applied here, not to Q):
//   if any row's chunk max > m + TAU/c:   m' = max(m, rowmax G), r = exp2((m - m')c),
//                                         O *= r, l *= r, m = m'
//   G = exp2(G c - m c);  l += rowsum(G)
// up = m (bf16), sp = l (fp32, replicated), gp = O (bf16, [TQ, DFULL]).
// Then P (G after the exponentials) goes to bfp16 in pv's pop order, and l += rowsum of
// that bfp16 P (fa256.cc summed the bf16 P with an emulated-bfp16 mmul; this sums what pv
// multiplies).
void softmax_step(bfloat16 *__restrict g, bfloat16 *__restrict up,
                  float *__restrict sp, bfloat16 *__restrict gp, uint8_t *__restrict p,
                  int32_t *__restrict rtp, int32_t pass, int32_t chunk) {
  (void)up;
  if (fa_skip(rtp, pass, chunk) || fa_dead(rtp, chunk))
    return;
  SET_ROUNDING();
#ifndef FA_X_NOEXP
  exp_p_rowsum(g, p, s_rs);
#endif
  // Rare path: the pass's first chunk (m is 0 until then), or a chunk whose row sum reached
  // 2^TAU (only possible when some score is past m + TAU - 6): m moves up to the row max,
  // O and l are rescaled, G shifted, and the exp pass redone.
  bool redo = chunk == 0;
  const aie::vector<float, 16> lim = aie::broadcast<float, 16>(float(1ull << FA_TAU));
  for (int i = 0; i < TQ * 8 && !redo; i += 16)
    redo = !aie::ge(aie::load_v<16>(s_rs + i), lim).empty();
  if (redo) {
    row_max(g, s_newmax);                                      // max of score - m, per row
    for (int i = 0; i < TQ; i += 16) {
      aie::vector<bfloat16, 16> d = aie::load_v<16>(s_newmax + i);
      if (chunk > 0)
        d = aie::max(d, aie::zeros<bfloat16, 16>());           // m only moves up
      aie::store_v(s_newmax + i, d);
      if (chunk > 0) {                                         // O, l *= 2^-delta
        Acc16 da;
        da.from_vector(d);
        aie::vector<float, 16> e = exp2f_poly16(aie::mul(da.template to_vector<float>(), -1.0f).template to_vector<float>());
        Acc16 ea;
        ea.from_vector(e);
        aie::store_v(s_resc + i, ea.template to_vector<bfloat16>());
      }
    }
    replicate_rows32(s_newmax, s_rep);                          // delta, replicated
    widen_scale(s_rep, s_tmp, 1.0f, TQ * 8);
    for (int i = 0; i < TQ * 8; i += 16)                        // -m -= delta
      aie::store_v(s_nuc + i, aie::sub(aie::load_v<16>(s_nuc + i), aie::load_v<16>(s_tmp + i)));
    for (int cb = 0; cb < LKP / 8; cb++)                        // G -= delta
      for (int z = 0; z < TQ / 8; z++) {
        bfloat16 *__restrict t = g + (cb * (TQ / 8) + z) * 64;
        aie::store_v(t, aie::sub(aie::load_v<64>(t), aie::load_v<64>(s_rep + z * 64)));
      }
    if (chunk > 0) {
      replicate_rows32(s_resc, s_rep);
      scale_o(gp, s_rep);
      for (int j = 0; j < TQ * 8; j += 16) {
        Acc16 ra;
        ra.from_vector(aie::load_v<16>(s_rep + j));
        aie::store_v(sp + j, aie::mul(aie::load_v<16>(sp + j), ra.template to_vector<float>())
                                 .template to_vector<float>());
      }
    }
    exp_p_rowsum(g, p, s_rs);
  }
  for (int i = 0; i < TQ * 8; i += 16)                          // l += this chunk's row sums
    aie::store_v(sp + i, aie::add(aie::load_v<16>(sp + i), aie::load_v<16>(s_rs + i)));
}

// End of a pass: O /= l (l replicated, so its bf16 inverse is scale_rows' table).
void finalize(float *__restrict sp, bfloat16 *__restrict gp) {
  SET_ROUNDING();
  for (int rh = 0; rh < TQ / 32; rh++) {
    for (int j = 0; j < 256; j += 16) {
      aie::accum<accfloat, 16> a;
      a.from_vector(aie::load_v<16>(sp + rh * 256 + j));
      aie::store_v(s_rep + j, aie::inv(a.template to_vector<bfloat16>()));
    }
    scale_o(gp, s_rep);
  }
}

} // extern "C"
