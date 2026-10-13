// eb.py: a finished down band (part 0 of the accumulator, C-tile order) into its output element
#include "moe_batch.h"
extern "C" {
void eb_out(const float *__restrict acc, float *__restrict c) {
  for (unsigned i = 0; i < 64 * MB_NT; i += 32) aie::store_v(c + i, aie::load_v<32>(acc + i));
}
}
