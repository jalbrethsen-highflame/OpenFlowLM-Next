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

8 threads pinned, per token: products 124 ms (NPU wait past the CPU's share 7 ms), LM head 25 ms
(NPU, with its context switch), mixer 8.5 ms, rest 2 ms. The int8 kernel gives the same 48 tokens
as the exact bf16 one.

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

### Decode (target 10 tok/s)
- [ ] MTP: draft with the model's MTP layer, verify k + 1 tokens in one pass. Needs the HF MTP
      weights (only shard 1 of 18 is on disk), a multi-token verify pass (the CPU kernel's math
      grows k + 1 times), DeltaNet state rollback.
- [ ] LM head: 25 ms a token on the NPU, with a context switch. Try it on the CPU (q8, ~13 ms at
      100 GB/s) or split it like the products.
- [ ] Tune the NPU share per thread count: at 16 threads the NPU is the slower side (8 ms wait).
- [ ] Server use: the threads spin (ACTIVE) and are pinned; a server must let them sleep
      between requests.

### Integration
- [ ] A chat / serve entry point for the causal model (`chat.py`, the oflm app).

### Operations
- [ ] Never run NPU and GPU jobs together on the Strix Halo: AGESA 1.0.0.2 has a Data Fabric
      MCE bug with both active (ROCm/TheRock#8709). Fixed in AGESA 1.0.0.2b.
