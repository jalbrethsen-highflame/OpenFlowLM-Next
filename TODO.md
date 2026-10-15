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

### Decode
- [ ] Measure decode on Linux. The skill records 419-450 ms/token (Windows, 2026-10-01).
- [ ] Decode reads every weight for every token. At the measured ~54 GB/s NPU DRAM rate, a
      4-bit 27B dense model is limited to ~3.5 tok/s. Ways past that: speculative decoding
      with the model's MTP head (the engine has no MTP support yet), or several sequences
      a step.
- [ ] Find what limits decode now (kernel time vs host vs DRAM) before choosing.

### Integration
- [ ] A chat / serve entry point for the causal model (`chat.py`, the oflm app).

### Operations
- [ ] Never run NPU and GPU jobs together on the Strix Halo: AGESA 1.0.0.2 has a Data Fabric
      MCE bug with both active (ROCm/TheRock#8709). Fixed in AGESA 1.0.0.2b.
