# TODO: Qwen3.8-27B causal inference on the NPU (branch feat/qwen27b-causal)

The causal (chat) Qwen3.8-27B on the open kernels. This branch starts from `feat/qwen27b-d3`:
the prompt pipeline, dit2 GEMMs, int4 weights and short-prompt tiers apply to the causal
model too. d3-only work (the decision readout, noncausal attention) stays on that branch; its
open items are in its own TODO.md. Model: `~/models/Qwen3.8-27B-NPU2`. Kernel sets:
`src/xclbins/Qwen3.8-27B-NPU2/open_kernels_b4` (int4) and `open_kernels` (bfp16), shared
with d3.

## What carries over from d3, untested on the causal model

- dit2 GEMMs with the hand-scheduled kernel, int4 weights, the device FFN, the L1 bank layout.
- 512-row and 256-row GEMM tiers for short prompts, host stages on real rows only.
- The tiled host attention, one host thread per physical core.

## Status (2026-10-15)

Decode with `OFLM_OPEN_CPU_DECODE=1` (CLI pins 8 threads, 4 per CPU die): products on the CPU from
the int4 prompt weights (int8 VNNI kernel), ~20 % of each product's columns on the NPU at the
same time, mixer on the host, LM head on the NPU. Peak memory 16.6 GB (the q4 pools are dropped).

| Route (45-token prompt, 48 tokens) | ms/token | tok/s | Peak RSS |
|---|---|---|---|
| NPU fused kernels (q4_1) | 366 | 2.7 | 30.2 GB |
| CPU, 8 threads, unpinned, no split | 252 | 4.0 | 16.6 GB |
| CPU + NPU split, 8 threads pinned | 160-170 | 5.9-6.2 | 16.6 GB |
| CPU + NPU split, 16 threads pinned | 146 | 6.8 | 16.6 GB |

Since then the final norm and q8 LM head run on the CPU (25 -> 12 ms, logits corr 0.99998
against the NPU head) and the input's int8 rounding is parallel: **149 ms/token, 6.7 tok/s** at 8
threads. Per token: products 120 ms (of which the NPU share's serial start-up 6 ms and its wait
2.6 ms), LM head 12 ms, mixer 10.6 ms, rest 4.7 ms.

Tried and left off (switches kept):
- A larger NPU share (OFLM_OPEN_DECODE_NPU_FRAC 0.25 / 0.30 / 0.35: 160 / 174 / 211 ms). The
  NPU's 256-row kernels compute 255 padding rows for each real one and pay a fixed cost on each
  of 256 dispatches a token, so 15-20 % is all they keep up with. (MTP's verify pass fills
  padding rows with real tokens.)
- The share's start-up overlapped with the CPU's (OFLM_OPEN_DECODE_SIDE=1: 152 ms): the NPU
  starts later and waits longer.
- bf16 alpha / beta weights (OFLM_OPEN_DECODE_AB16=1): mixer +1 ms.
- int4 LM head (OFLM_OPEN_CPU_HEAD=int4): head 12 -> 6.4 ms, 147 ms/token, but logits corr
  0.990 against the q8 head (top token 47 / 48): kept as an option, q8 is the default.

## Pending

### Verify
- [ ] Causal prefill with the new tiers: compare logits against the bfp16 set and the fp64
      replica (`open_kernels/model/replica_prompt.py --layers N`) at 128, 256, 512, 1K, 4K
      tokens.
- [ ] A prompt that continues an earlier turn (`pos_ > 0`) on every tier: the tier choice uses
      `pos_ + N`.
- [ ] int4 accuracy on generated text: NLL / KL against bf16 on a text set, and greedy tokens
      against the bfp16 set. int4 was measured only on d3 decisions.
- [ ] Exact bf16 prompt weights: the model dir has no `bf16_map.json`, so the prompt GEMMs
      repack from the q4 codes. Build it from the HF checkpoint (only shard 1 is on disk,
      `~/models/hf-qwen38-27b-shard1`).

### Prompt speed
- [ ] Causal dit_fa256b: the fast attention kernel (31 TFLOPS) is noncausal only; the causal
      prompt still uses dit_fa256 (~3.4x slower). Add the diagonal-chunk mask and skip the
      key chunks past it (also halves the work).
- [ ] Re-tune OFLM_OPEN_SMALL_MAX (2048 for d3): causal host attention does about half of
      d3's work, so the crossover to 1024-row chunks is probably higher.
- [ ] OMP_WAIT_POLICY=ACTIVE in the causal runner (d3: -7 % on a short prompt, -9 % at 1K).

### Decode (target 10 tok/s) -- met with MTP

`OFLM_OPEN_CPU_DECODE=1 OFLM_OPEN_MTP=<HF model-00018-of-00018.safetensors>` (2026-10-15): greedy
decode **exactly** equal to plain decode, k = 2 drafts a pass (`OFLM_OPEN_MTP_K`):

| Prompt (200 tokens) | Tokens a pass | Drafts accepted | tok/s |
|---|---|---|---|
| explain | 2.31 | 66.9 % | 10.6 |
| code | 2.65 | 82.7 % | 12.2 |
| list | 2.43 | 71.3 % | 11.2 |
| math | 2.65 | 82.7 % | 12.6 |

Mean 11.7 tok/s (k = 3: 11.1, k = 1: ~9.5); peak 18.5 GB. The MTP layer is int4 on the CPU; drafts
read an int4 copy of the head, the verify pass the q8 head once for all its rows; MTP gets the
prompt's last 256 rows as context (OFLM_OPEN_MTP_CTX, 22-55 ms once). The delta rule runs token
by token in verify passes and rollbacks: a T > 1 block sums differently (~2e-7) and flips
near-ties over 64 layers.

- [ ] Adaptive k: k = 3 wins on code / math (12.9 / 12.4 tok/s) and loses on prose; pick k
      per pass from the recent acceptance.
- [ ] Sampling (temperature / top-p) with MTP: rejection sampling instead of greedy matching.
- [ ] The MTP weights from a converted model file rather than the HF shard.
- [x] An NPU decode kernel without the 255 padding rows: designs/dit2/dec.py (gemm_dit.decode2),
      32 rows, B split over the 32 cores, ~80 us a dispatch then ~55 GB/s, verified against numpy
      (dec_test.py). One image (K = 5120 core loop): qkv's first 4096 columns, the FFN up|gate's
      first 8192, and the FFN down split along K (its first 4096 columns over K [0, 5120)).
      145 ms/token (6.9 tok/s) against 149-150 with the 256-row kernels: at one row a pass the
      ~80 us per dispatch and the sync after every product keep the NPU near 20 % of the bytes
      (it finishes before the CPU: wait 0.1 ms). Rows 1..31 are free on the NPU -- MTP's verify
      pass is where it pays.
- [ ] dit2_dec with the K steps as a runtime parameter (RTP), so one image can take any K range:
      finer NPU shares per product and the out projection (K = 6144) too. ~4 % more.
- [ ] Per-dispatch cost: A flush 184 KB and C read-back 256 KB per dispatch (pre 3.2 ms + post
      1.4 ms a token); read back only the rows used.
- [ ] The load peak (17.7 GB): the q8 head pool and its CPU repack are both alive for a moment.
- [ ] Server use: the threads spin (ACTIVE) and are pinned; a server must let them sleep
      between requests.

### Integration
- [ ] A chat / serve entry point for the causal model (`chat.py`, the oflm app).

### Operations
- [ ] Never run NPU and GPU jobs together on the Strix Halo: AGESA 1.0.0.2 has a Data Fabric
      MCE bug with both active (ROCm/TheRock#8709). Fixed in AGESA 1.0.0.2b.
