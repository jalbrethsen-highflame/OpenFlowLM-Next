// mbi.py: acc += one down band k-group x h
#include "mbi_core.h"
extern "C" {
void mbi_dn(const uint8_t *__restrict w, const int8_t *__restrict hq, const uint8_t *__restrict hmeta, float *__restrict acc, int kg) {
  MBI_T(hmeta, 5);
#ifdef MBI_DBG_H
  // debug: output band ob < 8 is h band ob as the down reads it (hq x its scale)
  unsigned &cnt = *reinterpret_cast<unsigned *>(const_cast<uint8_t *>(hmeta) + 8);   // a spare header word
  const unsigned ob = (cnt >> 1) & 31;
  cnt = (cnt + 1) & 63;
  if (kg != 0 || ob >= 8) return;
  const bfloat16 *sh = reinterpret_cast<const bfloat16 *>(hmeta + MBI_HM_SH);
  for (unsigned rg = 0; rg < 4; ++rg)
    for (unsigned tg = 0; tg < MBI_NTG; ++tg)
      for (unsigned l = 0; l < 64; ++l)
        acc[(rg * MBI_NTG + tg) * 64 + l] =
            hq[(((ob >> 2) * 16 + (ob & 3) * 4 + rg) * MBI_NTG + tg) * 64 + l] * (float)sh[ob * MBI_NT + tg * 4 + l / 16];
  return;
#endif
#if !defined(MBI_NULL) && !defined(MBI_NULL_DN)
  mbi_dn_kg(w, hq, hmeta, acc, (unsigned)kg);
#endif
}
}
