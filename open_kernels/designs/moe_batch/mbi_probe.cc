// mbi_probe.py's core: mbi_band_kg PROBE_ITERS times on one band k-group (accumulating into
// acc, zeroed first), for its result (PROBE_ITERS = 1) and its time.
#include "mbi_step.h"
#ifndef PROBE_ITERS
#define PROBE_ITERS 1
#endif
extern "C" {
void mbi_probe(const uint8_t *__restrict w, const uint8_t *__restrict x, float *__restrict acc) {
  const int8_t *xa = reinterpret_cast<const int8_t *>(x);
  const bfloat16 *xs = reinterpret_cast<const bfloat16 *>(x + 16 * MBI_NTG * 64);
  const bfloat16 *sx = xs + 8 * MBI_NTG * 64;
  for (unsigned i = 0; i < 4 * MBI_NTG * 64; i += 16) aie::store_v(acc + i, aie::zeros<float, 16>());
#if defined(MBI_INT2M)
  const int16 *tm = reinterpret_cast<const int16 *>(x + 16 * MBI_NTG * 64);
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg_int2m(w, xa, tm, tm, tm, acc);
#elif defined(MBI_INT4)
  const int16 *ti4 = reinterpret_cast<const int16 *>(x + 16 * MBI_NTG * 64);
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg_int4(w, xa, ti4, acc);
#elif defined(MBI_INT2)
  const int16 *ti2 = reinterpret_cast<const int16 *>(x + 16 * MBI_NTG * 64);
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg_int2(w, xa, ti2, acc);
#elif defined(MBI_INT)
  // int16 tiles from the x buffer's bytes (timing only: the values are whatever is there)
  const int16 *ti = reinterpret_cast<const int16 *>(x + 16 * MBI_NTG * 64);
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg_int(w, xa, ti, ti, ti, acc);
#else
  for (unsigned it = 0; it < PROBE_ITERS; ++it) mbi_band_kg(w, xa, xs, sx, acc);
#endif
}
}
