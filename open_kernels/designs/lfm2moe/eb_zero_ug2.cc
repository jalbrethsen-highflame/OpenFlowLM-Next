// eb.py: zero the core's up AND gate accumulators -- one buffer, [u 64 rows | g 64 rows] x MB_NT
#include "moe_batch.h"
extern "C" {
void eb_zero_ug2(float *__restrict c) { mb_zero_n(c, 128 * MB_NT); }
}
