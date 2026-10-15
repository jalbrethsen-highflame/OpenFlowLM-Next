//===- dec.cc: dit2_dec's core code -------------------------------------------------------------===//
//
// One 32-row x 128-column bf16 C tile per core, the hand-scheduled int4-mantissa call
// (hs_b4_call.inc, the one dit2's DIT_B4 build runs) once per 64-K step: A one 32-row quarter in
// pack_a.py's 2x2 order (2304 B), B one pack_b4 tile (4096 B of nibbles, then 128 exponent
// bytes). C keeps hs_b4_call's layout: [row block 4][column block 16][8 rows][8 columns].
//
//===--------------------------------------------------------------------------------------------===//

// Built twice (one ExternalFunction each): DEC_ZERO for dec_zero, DEC_MM for dec_mm.
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifdef DEC_ZERO
extern "C" {
void dec_zero(bfloat16 *__restrict c) {
  for (int i = 0; i < 32 * 128; i += 32) aie::store_v(c + i, aie::zeros<bfloat16, 32>());
}
}
#endif

#ifdef DEC_MM
#include "hs_b4_call.inc"
extern "C" {
void hs_b4_call(const bfp16ebs8 *pA, const uint8_t *pBn, bfloat16 *pC, const int8_t *pE);

void dec_mm(bfp16ebs8 *__restrict pA, uint8_t *__restrict pB, bfloat16 *__restrict pC) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  hs_b4_call(pA, pB, pC, reinterpret_cast<const int8_t *>(pB + 4096));
  aie::set_rounding(saved);
}
}
#endif
