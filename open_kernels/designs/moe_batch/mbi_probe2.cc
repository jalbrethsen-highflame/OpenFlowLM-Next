// mbi_probe2.py's core: mbi_band_kg (mbi_core.h) PROBE_ITERS times on one band k-group
#include "mbi_core.h"
#ifndef PROBE_ITERS
#define PROBE_ITERS 1
#endif
extern "C" {
void mbi_probe2(const uint8_t *__restrict w, const uint8_t *__restrict x, float *__restrict acc) {
  for (unsigned i = 0; i < 4 * MBI_NTG * 64; i += 16) aie::store_v(acc + i, aie::zeros<float, 16>());
#if defined(MBI_DBG_PREP)
  // acc[0..255] = d_i of chunk 0, [256..511] = mc_i, [512] = shift
  alignas(64) static int16 di[256], mi[256];
  const int sh = mbi_prep_chunk(w, di, mi);
  for (unsigned i = 0; i < 256; ++i) { acc[i] = (float)di[i]; acc[256 + i] = (float)mi[i]; }
  acc[512] = (float)sh;
  acc[513] = (float)aie::reduce_max(aie::abs(aie::load_v<32>(reinterpret_cast<const bfloat16 *>(w))));
#elif defined(MBI_PDN) || defined(MBI_PHQ)
  {   // hmeta (at x + 7680): the down's shift and 16 real tokens
    int32_t *hm = reinterpret_cast<int32_t *>(const_cast<uint8_t *>(x) + 7680);
    hm[0] = 18;
    hm[3] = 16;
  }
#endif
#if defined(MBI_DBG_PREP)
#elif defined(MBI_PDN)
  // the down step: hq and hmeta in the x buffer (8192 + 1088 B), the band in w
  for (unsigned it = 0; it < PROBE_ITERS; ++it)
    mbi_dn_kg(w, reinterpret_cast<const int8_t *>(x), x + 7680, acc, it & 1);   // (hmeta overlaps hq: timing only)
#elif defined(MBI_PHQ)
  // silu + quantize of one band: au = acc, ag = w
  for (unsigned it = 0; it < PROBE_ITERS; ++it)
    mbi_hq_band(acc, reinterpret_cast<const float *>(w), reinterpret_cast<int8_t *>(const_cast<uint8_t *>(x)),
                const_cast<uint8_t *>(x) + 7680, it & 7);
#elif defined(MBI_V3)
  reinterpret_cast<int32_t *>(const_cast<uint8_t *>(x))[2] = 16;   // the x header's real tokens
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg3(w, x, acc);
#else
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg(w, x, acc);
#endif
}
}
