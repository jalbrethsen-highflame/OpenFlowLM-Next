# TODO: LFM2-MoE on the NPU (branch feat/lfm2moe)

LiquidAI LFM2.5-24B-A2B and LFM2.5-8B-A1B (`lfm2_moe`) on the open XDNA2 kernels. Read the
skills first: `.opencode/skill/npu-model-porting` and `.opencode/skill/open-lfm2moe-kernels`.

## Status (2026-10-13)

- 24B: decode 22.8 tok/s (49 ms/token in the one-context image; 63 ms/token after a 4K
  prompt), 19 GB resident. 8B: 16.5 tok/s, 7.3 GB.
- 24B prompt speed: 1K 492, 4K 662, 8K 574, 16K 329, 32K 195 tok/s.
- Accuracy: full 24B logits corr >= 0.99987 against `open_kernels/model/replica_lfm2moe.py`.

## Pending

### Branch
- [ ] This branch still holds the Qwen3.8-27B / d3 commits (from `ea75925`). They now live
      on `feat/qwen27b-d3`, rebased onto `main`. Decide whether to move this branch back to
      `e069225` (the last LFM2 commit) plus this TODO. That needs a force-push to `personal`.

### Prompt speed
- [ ] Long prompts: host attention limits speed past 4K (32K: 195 tok/s; the f32 K/V outgrows
      L3). Move the prompt attention to the NPU, as dit_fa256 / dit_fa256b do for the 27B
      (needs LFM2's head geometry and a causal mask).
- [ ] Prompt GEMMs run on dit_gemm (15-19 TFLOPS). Port them to dit2 with the hand-scheduled
      kernel and int4 weights (31-33 TFLOPS bfp16, 38-41 int4 on the 27B shapes), as on
      `feat/qwen27b-d3`.
- [ ] Experts are bound by the weight stream (a visit streams 5.9 MB of expert weights for 32
      tokens, 4.1 s at 4K). Split tokens, not rows, across a column's 4 cores, with the
      weight band broadcast from the mem tile: up to 4x less weight traffic a token.
- [ ] The 8B has no prompt route: `eb` needs ff to be a multiple of 512 (8B ff = 1792).

### Accuracy
- [ ] The NPU prompt paths differ from each other by KL ~0.10 over 4K tokens (equal NLL).
      Check them against an fp64 reference at long context.

### Kernels
- [ ] Builds with a 4-column partition (n_cols=4) gave wrong output. Find why before using
      them again.

### Integration
- [ ] Not wired into the oflm app. The LFM2 adapter selects the open engine when a kernel set
      exists, so a `model_list` entry with family `lfm2` is the likely path.

### Operations
- [ ] Never run NPU and GPU jobs together on the Strix Halo: AGESA 1.0.0.2 has a Data Fabric
      MCE bug with both active (ROCm/TheRock#8709). Fixed in AGESA 1.0.0.2b.
