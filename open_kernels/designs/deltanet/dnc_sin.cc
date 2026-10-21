// dnc.py: one 1 KB piece of the state slice in (dnc_core.h)
#include "dnc_core.h"
extern "C" {
void dnc_sin(const uint8_t *__restrict piece, float *__restrict S, int idx) { dnc::s_in(piece, S, (unsigned)idx); }
}
