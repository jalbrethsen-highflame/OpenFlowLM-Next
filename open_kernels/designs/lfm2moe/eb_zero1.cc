// eb.py: zero part 0 (64 rows x MB_NT) of the accumulator -- one down band
#include "moe_batch.h"
extern "C" {
void eb_zero1(float *__restrict acc) { mb_zero_n(acc, 64 * MB_NT); }
}
