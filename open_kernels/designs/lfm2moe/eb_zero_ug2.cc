// eb.py: zero the core's up AND gate accumulators -- one buffer, [u 128 rows | g 128 rows] x MB_NT
#include "moe_batch.h"
extern "C" {
void eb_zero_ug2(float *__restrict c) { mb_zero_n(c, 256 * MB_NT); }
}
