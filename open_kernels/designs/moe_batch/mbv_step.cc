// mbv.py: acc += one band k-group of W (mbi's transform) x the x element (mbi_core.h)
#include "mbi_core.h"
extern "C" {
void mbv_step(const uint8_t *__restrict w, const uint8_t *__restrict x, float *__restrict acc) {
  mbi_band_kg3(w, x, acc);
}
}
