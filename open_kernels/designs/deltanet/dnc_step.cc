// dnc.py: one 16-token step of a unit (dnc_core.h)
#include "dnc_core.h"
extern "C" {
void dnc_step(const uint8_t *__restrict rec, const bfloat16 *__restrict v, bfloat16 *__restrict o, float *__restrict S,
              uint8_t *__restrict sc, int32_t hh) {
  dnc::step(rec, hh & 1, v, o, S, sc);
}
}
