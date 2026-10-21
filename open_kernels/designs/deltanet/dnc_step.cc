// dnc.py: one 16-token step of a unit (dnc_core.h)
#include "dnc_core.h"
extern "C" {
void dnc_step(const uint8_t *__restrict rec, const bfloat16 *__restrict v, bfloat16 *__restrict o, float *__restrict S,
              uint8_t *__restrict sc) {
  dnc::step(rec, v, o, S, sc);
}
}
