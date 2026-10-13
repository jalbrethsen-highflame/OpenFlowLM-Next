// eb.py: h = silu(g) * u for one band (64 rows) of the core's band pair, u and g from the one
// [u | g] buffer, into rows 64 half .. of the pair's h element. moe_batch.h's mb_silu_tile over
// 4 row groups instead of 8: rows 0-63 of a pair's h are its first 64-k tile.
#include "moe_batch.h"
extern "C" {
void eb_silu2(const float *__restrict ug, float *__restrict hf, int half) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  const float *__restrict u = ug, *__restrict g = ug + 64 * MB_NT;
  bfloat16 *__restrict h = reinterpret_cast<bfloat16 *>(hf) + half * 64 * MB_NT;
  for (unsigned blk = 0; blk < 8; blk += 2)           // one 16-row group: its even and odd C blocks
    for (unsigned s = 0; s < MB_NG; ++s) {            // that group's eight tokens at a time
      const float *__restrict ue = u + (blk * MB_NG + s) * 64;
      const float *__restrict ge = g + (blk * MB_NG + s) * 64;
      aie::vector<bfloat16, 32> he[2], ho[2];
      for (unsigned hh = 0; hh < 2; ++hh) {           // tokens 0-3 of the sub-tile, then 4-7
        accf32 a;
        a.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(ge + hh * 32)), aie::load_v<32>(ue + hh * 32)));
        he[hh] = a.template to_vector<bfloat16>();
        a.from_vector(fmulN<32>(vsiluN<32>(aie::load_v<32>(ge + MB_NG * 64 + hh * 32)),
                                aie::load_v<32>(ue + MB_NG * 64 + hh * 32)));
        ho[hh] = a.template to_vector<bfloat16>();
      }
      const auto z = aie::interleave_zip(aie::concat(he[0], he[1]), aie::concat(ho[0], ho[1]), 1);
      const auto kbs = aie::interleave_unzip(z.first, z.second, 8);
      aie::store_v(h + (blk * MB_NG + s) * 64, kbs.first);
      aie::store_v(h + ((blk + 1) * MB_NG + s) * 64, kbs.second);
    }
}
}
