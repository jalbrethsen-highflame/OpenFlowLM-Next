// mbi.py: zero an accumulator ([4 row groups][NTG][64] f32)
#include "mbi_core.h"
extern "C" {
void mbi_zero(float *__restrict acc) {
  for (unsigned i = 0; i < 4 * MBI_NTG * 64; i += 16) aie::store_v(acc + i, aie::zeros<float, 16>());
}
}
