// eb.py: ONE copy of moe_batch's band step for up, gate AND down -- the core's 16 KB of program
// memory holds the tile routine once. k-tile ky of a band into part `part` (64 rows x MB_NT) of acc.
#include "moe_batch.h"
extern "C" {
void eb_step(const uint8_t *__restrict band, const bfloat16 *__restrict xa, float *__restrict acc, int ky, int part) {
  mb_step_tile(band, ky, xa, acc + part * 64 * MB_NT);
}
}
