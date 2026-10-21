// dnc.py: one 1 KB piece of the state slice out (dnc_core.h)
#include "dnc_core.h"
extern "C" {
void dnc_sout(const float *__restrict S, uint8_t *__restrict piece, int idx) { dnc::s_out(S, piece, (unsigned)idx); }
}
