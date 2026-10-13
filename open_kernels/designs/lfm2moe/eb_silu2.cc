// eb.py: h = silu(g) * u for the core's band pair, u and g from the one [u | g] buffer
#include "moe_batch.h"
extern "C" {
void eb_silu2(const float *__restrict ug, float *__restrict h) {
  mb_silu_tile(ug, ug + 128 * MB_NT, reinterpret_cast<bfloat16 *>(h));
}
}
