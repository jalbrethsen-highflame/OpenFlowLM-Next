// eb.py, EB_BFP=1: the band step on the bfp16 MAC array instead of the bf16 vector path.
//
// XDNA2's matrix unit takes 8-bit operands (int8, or bfp16ebs8: an 8-bit mantissa per value
// sharing one exponent per 8). bf16 x bf16 mmul runs on the fp32 vector unit, ~5x slower
// (.opencode/skill/dit-gemm). So, per call (one 64-row band x one 64-k tile x MB_NT tokens):
//
//   W  the band's weights dequantised ONCE (moe_batch.h's law: n * d + m, bf16), each 8 x 8
//      block transposed to [row][k] -- so a shared exponent covers 8 k of ONE row, which has
//      one (d, m) and so a narrow range -- and converted to bfp16 into an L1 scratch;
//   A  the token tile converted to bfp16 once ([token][8 k] per block: one exponent per
//      token per 8 k);
//   M  C[token][row] += A . W^T with mac_8x8_8x8T, one accumulator live at a time (Peano
//      spills block vectors held in registers; dit_gemm's lesson), both operands streamed
//      from the scratch.
//
// Round-to-nearest-even for the call: the core's default (floor) biases both conversions
// (dit_gemm measured -7 % at K = 3072). The C layout is moe_batch's, so nothing on the host
// changes.
#include "moe_batch.h"

static constexpr unsigned kBlk = 72;          // one bfp16ebs8 block of 64: 64 mantissas + 8 exponents
alignas(64) static uint8_t eb_wscr[64 * kBlk];          // 4 row groups x 2 parities x 8 k-octets
alignas(64) static uint8_t eb_ascr[8 * MB_NG * kBlk];   // [sub-tile][k-octet]

template <bool ODD>
static inline void eb_dequant(const uint8_t *__restrict nibp, const bfloat16 *__restrict dp,
                              const bfloat16 *__restrict mp, unsigned ky,
                              aie::block_vector_output_buffer_stream<bfp16ebs8, 64> &out) {
  AIE_LOOP_RANGE(2, 2)
  for (unsigned kb = 0; kb < 2; ++kb) {
    const unsigned kb_abs = ky * 2 + kb;
    const aie::vector<bfloat16, 64> d = mb_scale<ODD, true>(dp, kb_abs);
    const aie::vector<bfloat16, 64> m = mb_scale<ODD, false>(mp, kb_abs);
    for (unsigned il = 0; il < 4; ++il) {
      const unsigned i = kb * 4 + il;
      const aie::vector<uint8_t, 64> q = aie::load_v<64>(nibp + i * 64);
      const aie::vector<bfloat16, 64> n = aie::to_float<bfloat16>(aie::bit_and((uint8_t)(ODD ? 0xF0 : 0x0F), q), 0);
      accN<64> se;
      se.from_vector(m);
      se = aie::mac(se, n, d);                                  // [8 k][8 rows]
      const aie::vector<bfloat16, 64> wt = aie::transpose(se.template to_vector<bfloat16>(), 8, 8);
      aie::accum<accfloat, 64> wa;
      wa.from_vector(wt);                                       // [8 rows][8 k]
      out << wa.template to_vector<bfp16ebs8>();
    }
  }
}

extern "C" {
void eb_step_bfp(const uint8_t *__restrict band, const bfloat16 *__restrict xa, float *__restrict acc, int ky,
                 int part) {
#ifdef MB_NULL_MM
  return;                     // timing ablation: the streams only
#endif
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  float *__restrict c = acc + part * 64 * MB_NT;
  // W: the band's 64 rows x 64 k, once
  {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> wout(reinterpret_cast<bfp16ebs8 *>(eb_wscr));
    for (unsigned g = 0; g < 4; ++g) {
      const uint8_t *__restrict chunk = band + (g >> 1) * MB_CHUNK;
      const unsigned half = g & 1;
      const uint8_t *__restrict nibp = chunk + MB_NIB + half * 2048 + ky * 512;
      const bfloat16 *__restrict dp = reinterpret_cast<const bfloat16 *>(chunk) + half * 16;
      const bfloat16 *__restrict mp = reinterpret_cast<const bfloat16 *>(chunk + 512) + half * 16;
      eb_dequant<false>(nibp, dp, mp, ky, wout);
      eb_dequant<true>(nibp, dp, mp, ky, wout);
    }
  }
  // A: the tokens, once
  {
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> aout(reinterpret_cast<bfp16ebs8 *>(eb_ascr));
    for (unsigned s = 0; s < MB_NG; ++s)
      for (unsigned i = 0; i < 8; ++i) {
        aie::accum<accfloat, 64> xacc;
        xacc.from_vector(aie::load_v<64>(xa + (i * MB_NG + s) * 64));
        aout << xacc.template to_vector<bfp16ebs8>();
      }
  }
  // M: C[s][g][p] += A[s] . W[g][p]^T over the 8 k-octets
  for (unsigned s = 0; s < MB_NG; ++s) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> win(reinterpret_cast<const bfp16ebs8 *>(eb_wscr));
    for (unsigned gp = 0; gp < 8; ++gp) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> ain(reinterpret_cast<const bfp16ebs8 *>(eb_ascr));
      ain.seek(s * 8);
      float *__restrict cb = c + ((gp >> 1) * 2 * MB_NG + (gp & 1) * MB_NG + s) * 64;
      aie::accum<accfloat, 64> C;
      C.from_vector(aie::load_v<64>(cb));
      for (unsigned i = 0; i < 8; ++i) C = mac_8x8_8x8T(ain.pop(), win.pop(), C);
      aie::store_v(cb, C.template to_vector<float>());
    }
  }
  aie::set_rounding(saved);
}
}
