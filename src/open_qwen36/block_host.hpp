/// \file block_host.hpp
/// \brief The block prefill's host stages: what runs on the CPU between the
///        GEMM dispatches of a 256-token block (OPEN-PREFILL-BATCH).
///
/// The 35B's DeltaNet recurrence, its attention over the KV rows and its
/// router are all on the host in the block route; the kernels do the
/// projections (whole-array GEMMs) and, one token at a time, the MoE block.
/// Every function here is the numpy reference open_kernels/model/replica_block.py
/// written out in C++ over T tokens with the state carried through the first
/// t_real of them: the kernels keep the conv state and the KV rows in bf16, so
/// those are rounded exactly as the device would; everything else accumulates
/// in double. block_host_test.cpp holds these to that reference's fixture.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>

namespace open_qwen36 {
namespace host {

/// out[t] = x[t] / sqrt(mean(x[t]^2) + eps) * w, rows of d.
void rmsnorm_rows(const float* x, size_t T, size_t d, const float* w, double eps, float* out);
/// y [N, T] row-major (the GEMM's own output order) -> out [T, N].
void transpose(const float* y, size_t N, size_t T, float* out);

/// One column range of a GEMM's output and where it goes: columns [off, off + width) of
/// y [N, T] land in dst as [T, width].
struct TransposePart {
    float* dst = nullptr;
    size_t off = 0, width = 0;
};
/// y [N, T] transposed straight into the ranges the caller is going to read it as, so a
/// fused projection needs no [T, N] copy in between. The ranges may not overlap.
void transpose_parts(const float* y, size_t T, const TransposePart* parts, size_t n_parts);
/// The same ranges out of a ROW-major y [T, N] (dit_gemm's C): dst[t * width + i] = y[t * N + off + i].
void split_cols(const float* y, size_t T, size_t N, const TransposePart* parts, size_t n_parts);
/// x [T, K] fp32 -> the GEMM's tiled bf16 activation layout ([K, T] "k,n" order, 64 x 32
/// tiles of 8 x 8 MAC sub-tiles, gemm_q4_prefill.py); out holds K * T bf16 bits.
void tile_x(const float* x, size_t T, size_t K, uint16_t* out);

struct DeltaGeom {
    size_t T = 0, t_real = 0, hid = 0;
    size_t key_heads = 0, value_heads = 0, head_dim = 0, taps = 0;
    size_t lanes = 0;       ///< columns of the packed alpha / beta projection (the value heads, padded)
    size_t s_rows = 0;      ///< rows of one head's S in the state buffer (head_dim real, the rest zero)
    double eps = 1e-6;      ///< the gated norm's eps
    /// packed prompts: reset[t] != 0 starts a new sequence at row t -- the conv window sees
    /// zeros before it and S restarts from zero. nullptr: one sequence.
    const uint8_t* reset = nullptr;
    /// optional bf16 copies of Wa / Wb ([hid, lanes]): a decode-sized block (T <= 8) reads these
    /// instead of the f32 ones (half the bytes; CPU decode)
    const uint16_t* wa16 = nullptr;
    const uint16_t* wb16 = nullptr;
    /// optional: S read from here and written to the S argument (one chunk: T <= 4) -- the MTP
    /// verify pass keeps the state before it intact without copying it
    const float* s_in = nullptr;
    /// optional: qkv as the bf16 rows themselves (row stride ld16 elements) instead of the f32
    /// qkv argument -- the prompt pipe's GEMM output, read without an f32 copy (same result)
    const uint16_t* qkv16 = nullptr;
    size_t ld16 = 0;
    /// optional: z as bf16 rows (row stride ldz elements; the prompt pipe's GEMM output) instead of
    /// the f32 z argument, read where the output gate needs it
    const uint16_t* z16 = nullptr;
    size_t ldz = 0;
    /// optional (an MTP verify pass, the chunk4 path): each row's normalised k [t_real, key_heads *
    /// head_dim], delta d [t_real, value_heads * head_dim] and decay [t_real, value_heads] saved
    /// here, so delta_rollback can rebuild the state after a prefix of the rows in one pass over S
    float* sv_k = nullptr;
    float* sv_d = nullptr;
    float* sv_a = nullptr;
};

/// The state after the first m (1..3) rows of one chunk4 chunk, from the state before it (S0) and
/// the rows' saved k, d, decay (DeltaGeom.sv_*): S = G_m S0 + sum_s r_{m-1,s} k_s d_s^T, the
/// chunk kernel's update in the same order (so the same values as re-running the m rows). k, d,
/// a point at the chunk's first row.
void delta_rollback(const DeltaGeom& g, size_t m, const float* S0, const float* k, const float* d, const float* a, float* S);

/// The linear-attention layer's middle over a block. qkv [T, 2*key_w + vw] and z [T, vw]
/// are the fused projection's outputs (pre-activation), xn [T, hid] the normed layer
/// input; convw [taps, nch], Wa / Wb [hid, lanes], A / dtb [value_heads], nw [head_dim].
/// conv_state [taps-1, nch] (bf16) and S [value_heads, s_rows, head_dim] (f32) are the
/// layer's state, updated in place through the first t_real tokens. og [T, vw] out
/// (zero past t_real). phase_ms, if given, gets the two halves' wall time: the
/// per-token one then the delta rule.
void deltanet_block(const DeltaGeom& g, const float* qkv, const float* z, const float* xn, const float* convw,
                    const float* Wa, const float* Wb, const float* A, const float* dtb, const float* nw,
                    uint16_t* conv_state, float* S, float* og, double* phase_ms = nullptr);

struct AttnGeom {
    size_t T = 0, t_real = 0, nh = 0, kvh = 0, hd = 0, rot = 0;
    size_t pos0 = 0;        ///< the block's first position (and KV row)
    double eps = 1e-6;      ///< the q / k norms' eps
    /// packed prompts (attention_prep): row t's rope position, its KV row staying pos0 + t.
    /// nullptr: position pos0 + t.
    const size_t* rpos = nullptr;
    /// an int8 copy of the cache (kv8_quant_rows' rows of kv8_row bytes) for decode-sized blocks:
    /// attention_block quantizes the block's rows into it and reads it instead of kv. nullptr: none.
    uint8_t* kv8 = nullptr;
    size_t kv8_row = 0;
};
/// kv rows [r0, r1) (bf16 [K | V], kvh x hd each) as int8 rows of row8 bytes: K int8 [kvh hd], V int8
/// [kvh hd], then the K scales and the V scales (f32 [kvh] each): one symmetric scale per position
/// and kv head. row8 >= 2 kvh hd + 8 kvh.
void kv8_quant_rows(const uint16_t* kv, size_t kv_row_elems, size_t kvh, size_t hd, size_t r0, size_t r1, uint8_t* kv8, size_t row8);

/// The full-attention layer's middle over a block. q / gate [T, nh*hd], k / v [T, kvh*hd]
/// are the fused projection's outputs; qn / kn [hd]; inv_freq [rot/2]. kv is the layer's
/// cache: rows of kv_row_elems bf16, [K_t | V_t] each kvh*hd wide, rows [0, pos0) valid on
/// entry and [pos0, pos0 + t_real) written. og [T, nh*hd] out: the gated attention
/// output (zero past t_real); gate == nullptr for a family without the output gate.
void attention_block(const AttnGeom& g, const float* q, const float* k, const float* v, const float* gate,
                     const float* qn, const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems,
                     float* og);

/// The host half of the block attention when its products run on the NPU (OPEN-PREFILL-ATTN):
/// what attention_block does before its products and nothing after. Q [T, nh*hd] fp32 out,
/// normed and roped with 1/sqrt(hd) folded in (a power of two at every head dim here, so the
/// bf16 the kernel sees rounds exactly as the unscaled value would); the block's k / v normed,
/// roped and written to the cache rows [pos0, pos0 + t_real) in bf16. Rows of Q past t_real are zero.
/// Attention of the t_real queries of attention_prep's Q (normed, roped, already scaled) over
/// the cache rows [0, keys), with no causal mask: d3's noncausal full attention, where every
/// prompt token sees the whole prompt. og [T, nh * hd]; rows past t_real are zero.
void attention_full(const AttnGeom& g, const float* Q, const uint16_t* kv, size_t kv_row_elems, size_t keys, float* og);

void attention_prep(const AttnGeom& g, const float* q, const float* k, const float* v, const float* qn,
                    const float* kn, const double* inv_freq, uint16_t* kv, size_t kv_row_elems, float* Q);

/// rows [n, k] bf16, `stride` elements apart (the cache's K or V half), as the GEMM's tiled B for
/// B = rows^T [k, n] -- tile_x's layout from a bf16 source. Rows at or past n_real read as zero;
/// n a multiple of 32, k of 64.
void tile_rows_as_bt(const uint16_t* rows, size_t stride, size_t n_real, size_t n, size_t k, uint16_t* out);
/// rows [k, n] bf16, `stride` apart, as the tiled B for B = rows itself. Rows past k_real read as zero.
void tile_rows_as_b(const uint16_t* rows, size_t stride, size_t k_real, size_t k, size_t n, uint16_t* out);

/// One window chunk of the causal row softmax, with the running (max, sum) carried across chunks:
/// s [M, L] fp32 scores whose column 0 is window row c0; row r's query sits at pos[r], so columns
/// c0 + j > pos[r] are masked. Writes p [M, L] bf16 = exp(s - m_new) (zero where masked), scales
/// acc [M, hd] and l [M] by exp(m_old - m_new), and folds the bf16-rounded p into l -- what the
/// kernel will multiply is what the denominator counts. Start with m = -inf, l = 0, acc = 0.
void softmax_chunk(size_t M, size_t L, size_t hd, size_t c0, const float* s, const size_t* pos, float* m, float* l,
                   float* acc, uint16_t* p);

/// The MoE router over a block: softmax of xm [T, hid] @ Wr [hid, E] into probs [T, E], the
/// top-k by probability (lowest index on a tie, as the kernel's router_fin picks) into
/// idx [T, topk], renormalised into w [T, topk].
void router_block(size_t T, size_t hid, size_t E, size_t topk, const float* xm, const float* Wr, float* probs,
                  int32_t* idx, float* w);
/// One token's softmax over its E logits `lg` (overwritten with the probabilities, copied to probs),
/// the top-k (idx) and their weights normalised to sum 1 (w); taken: E bytes of scratch.
void router_select(size_t E, size_t topk, float* lg, char* taken, float* probs, int32_t* idx, float* w);

/// y [npad] = x [K] times a dit2 int4-mantissa weight (designs/dit_gemm/pack.py pack_b4: per
/// [128-column tile][64-K tile] 4096 bytes of nibbles in blocks [pair 8][k-block 8][2] of 8 columns
/// x 8 k, then 128 exponent bytes, scale 2^(e - 133)) -- the bytes dit2 reads, on the CPU. x is
/// rounded to bf16 (the products and sums are fp32). OpenMP over 128-column strips and K ranges.
void gemv_b4(const uint8_t* B, size_t npad, size_t K, const float* x, float* y,
             const std::function<void()>& side = nullptr, size_t kt0 = 0, size_t kt1 = 0);
/// (kt0, kt1: only those 64-K tiles of every strip, a partial sum; kt1 = 0 means all)
/// R rows at once (1..8; an MTP verify pass): x [R, K] row-major, y [R, ldy] (columns 0..npad);
/// each weight block unpacked once for all rows. The int8 path; the bf16 one loops gemv_b4.
/// the decode attention kernel: -1 OFLM_OPEN_ATTN_FLASH decides (default flash), 0 split, 1 flash (tests)
extern int g_attn_flash;
extern double g_dn_prof[6];
extern double g_dn_p1[4];   // deltanet_block's phases (us, summed; see block_host.cpp)
void gemv_b4_rows(const uint8_t* B, size_t npad, size_t K, const float* x, size_t R, float* y, size_t ldy,
                  size_t kt0 = 0, size_t kt1 = 0);
/// The dit FFN's up|gate output to h: every 128 columns of `ug` are [gate 64 | up 64]; h [ff] =
/// silu(gate) * up. Parallel and 16 lanes wide (ff % 64 == 0).
void swiglu_tiles(const float* ug, size_t ff, float* h);

/// The q8 lm_head on the CPU. lmhead_q8_repack: the NPU pool's chunks (32 rows x 256 K, 8704 B:
/// 256 bf16 scales [kb][r], then int8 codes [(r / 16) * 4096 + k * 16 + r % 16]; band order of
/// pools.cpp's lmhead_q8 op) -> [N / 16 row groups][K / 32 blocks][16 bf16 scales | 8 x (16 rows
/// x 4 k) codes], lmhead_q8_cpu_bytes(N, K) bytes. lmhead_q8_cpu: y [N] = W x, x rounded to int8
/// per 32 (max|x| / 127), vpdpbusd on |x| with x's sign moved onto the codes.
size_t lmhead_q8_cpu_bytes(size_t N, size_t K);
void lmhead_q8_repack(const uint8_t* pool, size_t N, size_t K, uint8_t* out);
void lmhead_q8_cpu(const uint8_t* W, size_t N, size_t K, const float* x, float* y);
/// R rows (1..8) at once: x [R, K], y [R, N]; the head read once.
// The head in two stages: int4 logits for all N (W4, dit2 int4 blocks), then the top `topk` of each
// row's first NR -- and any token whose int4 logit still reaches the row's best -- rescored with the
// q8 head W8 (its own kernel: those values bit-identical to lmhead_q8_cpu_rows). The argmax is the
// q8 one unless the q8 argmax sits outside the int4 top-k (check: counted against the full q8 head).
struct HeadTwoStats { size_t calls, rows, groups, extra, checked, mismatch, max_rank; };
extern HeadTwoStats g_head2;
void lmhead_two_stage_rows(const uint8_t* W8, const uint8_t* W4, size_t N, size_t K, const float* x, size_t R, float* y,
                           size_t NR, size_t topk, bool check);
void lmhead_q8_cpu_rows(const uint8_t* W, size_t N, size_t K, const float* x, size_t R, float* y);
/// memcpy over the OpenMP threads (the MTP state snapshots: ~154 MB a pass on the 27B)
void par_copy(void* dst, const void* src, size_t bytes);

/// Where a q4_1 pool region keeps its 5120 B chunks (32 rows x 256 K: d bf16[256] at 0, m bf16[256]
/// at 512, then the nibbles k-major, 16 rows per k): chunk (64-row band b, 256-K group g, 32-row half
/// h) at base + (b / 2) s2 + (b % 2) s1 + g sk + h 5120. q41_std: the std_perm band order (bands
/// whole, groups inside); q41_stripe: Qwen3.6-MoE's experts, 128-row stripes of two bands
/// interleaved by k-group, stripes `stride` apart.
/// shift >= 0: the chunks are mbi's transform of q4_1 (Core, moe_batch.kind "mbi"): every nibble
/// byte XOR 0x88, d and m replaced by int16 d 2^shift and (m + 8 d) 2^shift.
struct Q41Lay { size_t s2 = 0, s1 = 0, sk = 0; int shift = -1; };
Q41Lay q41_std(size_t K);
Q41Lay q41_stripe(size_t stride);
/// x [K] as int8 per 32 (xq), each block's scale (sx) and its float sum (xs) -- gemv_q41_rows' input
struct Q41X { std::vector<int8_t> q; std::vector<float> s, sum; std::vector<int32_t> qsum; };   // qsum: sum of q per block
void q41_prep_x(const float* x, size_t K, Q41X& out);
/// sum_i bf16(x_i) w_i in double (the shared expert's gate): the products are exact in double, so
/// only the order of the sum differs from the scalar loop (AVX-512 when the CPU has it)
double bf16x_dot(const float* x, const float* w, size_t n);
/// y[r][band rows] = W x_r for the bands [b0, b1) of a q4_1 region, r < R (1..8): each weight byte
/// read once for all rows (AVX-512 VNNI + VBMI; int8 inputs). w = d q + m = d (q - 8) + (m + 8 d), so per
/// 32-K block y += d sx (sum(q xq) - 8 sum(xq)) + (m + 8 d) sum(x): the int8 rounding meets the
/// centred nibbles, not q's mean (with q in 0..15 and m ~ -8 d it cancelled, 1.5 % error against
/// 0.5 %). No threading inside: callers split the bands.
void gemv_q41_bands(const uint8_t* base, const Q41Lay& lay, size_t K, const Q41X* const* x, size_t R, float* const* y,
                    size_t b0, size_t b1);
/// W [n, K] f32 (n % 64, K % 256) as q4_1 chunks in the std_perm band order (q41_std(K)): per
/// (row, 32-K block) m = min, d = (max - min) / 15, q = round((w - m) / d), d and m in bf16 and q
/// against the rounded ones. out: n K 5 / 8 bytes.
void q41_pack_std(const float* W, size_t n, size_t K, uint8_t* out);
/// mbi's transform of nch q4_1 chunks in place, all at one shift; mbi_shift: that shift for the
/// chunks' largest |d| and |m + 8 d| (d 2^S <= 8000, (m + 8 d) 2^S <= 32000: mbi_core.h's int32
/// sums stay in range).
int mbi_shift(const uint8_t* chunks, size_t nch, size_t stride);
void mbi_transform(uint8_t* chunks, size_t nch, size_t stride, int shift);
/// The same over all n rows, OpenMP over 32-row halves: x [R, K] row-major, y [R, ldy].
void gemv_q41_rows(const uint8_t* base, const Q41Lay& lay, size_t n, size_t K, const float* x, size_t R, float* y,
                   size_t ldy);

}  // namespace host
}  // namespace open_qwen36
