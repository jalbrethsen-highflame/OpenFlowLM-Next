// mbi.py: acc += one up / gate band k-group x the x element; the down's shift into hmeta
#include "mbi_core.h"
extern "C" {
void mbi_ug(const uint8_t *__restrict w, const uint8_t *__restrict x, float *__restrict acc, uint8_t *__restrict hmeta) {
  MBI_T(hmeta, 4);
  reinterpret_cast<int32_t *>(hmeta)[0] = reinterpret_cast<const int32_t *>(x)[1];   // S_dn
  reinterpret_cast<int32_t *>(hmeta)[3] = reinterpret_cast<const int32_t *>(x)[2];   // the row's real tokens
#if !defined(MBI_NULL) && !defined(MBI_NULL_UG)
  mbi_band_kg3(w, x, acc);
#endif
}
}
