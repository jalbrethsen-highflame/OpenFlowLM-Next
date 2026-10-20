// mbi.py: an output band's accumulator as bf16 into the C element ([4 row groups][NTG][64])
#include "mbi_core.h"
extern "C" {
void mbi_out(const float *__restrict acc, bfloat16 *__restrict c, uint8_t *__restrict hmeta) {
  MBI_T(hmeta, 7);
  for (unsigned i = 0; i < 4 * MBI_NTG * 64; i += 32) {
    aie::accum<accfloat, 32> a;
    a.from_vector(aie::load_v<32>(acc + i));
    aie::store_v(c + i, a.template to_vector<bfloat16>());
  }
#ifdef MBI_TIME
  for (unsigned i = 0; i < 8; ++i) reinterpret_cast<uint32_t *>(c)[i] = reinterpret_cast<const uint32_t *>(hmeta)[4 + i];
#endif
}
}
