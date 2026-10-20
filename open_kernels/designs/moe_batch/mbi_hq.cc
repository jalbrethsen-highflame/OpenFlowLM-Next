// mbi.py: h band pb = silu(g) u, quantized into the down's A tiles
#include "mbi_core.h"
extern "C" {
void mbi_hq(float *__restrict au, const float *__restrict ag, int8_t *__restrict hq, uint8_t *__restrict hmeta, int pb) {
  MBI_T(hmeta, 6);
#if !defined(MBI_NULL) && !defined(MBI_NULL_HQ)
  mbi_hq_band(au, ag, hq, hmeta, (unsigned)pb);
#endif
}
}
