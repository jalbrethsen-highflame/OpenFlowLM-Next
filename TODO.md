# TODO: vllm-sr/d3 on the NPU (branch feat/qwen27b-d3)

d3 is the Qwen3.8-27B decision model with a 255-way readout. Front end and eval scripts:
`utilities/d3-npu/`. Kernel sets: `src/xclbins/Qwen3.8-27B-NPU2/open_kernels_b4` (int4,
default) and `open_kernels` (bfp16).

## Status (2026-10-15, int4, default NPU power mode)

| Prompt tokens | Time | | Prompt tokens | Time |
|---|---|---|---|---|
| 128 | 0.57 s | | 2048 | 3.35 s |
| 256 | 0.67 s | | 8192 | 11.6 s |
| 512 | 1.16 s | | 32768 | 55 s |
| 1024 | 2.06 s | | | |

- Accuracy: int4 equals bfp16 on 78 held-out cases. The chunk paths shift a probability by
  0.009 at most, with no changed answers.
- Memory: 14.6 GB for decisions, 24.4 GB peak for a 32K prompt.

## Pending

### Short-decision latency
- [ ] Host time is about half of a short decision (0.3 s of 0.67 s at 256 tokens). Make the
      host DeltaNet (conv and delta rule) and the bf16/bfp16 conversions faster, or overlap
      them with the NPU inside a layer.
- [ ] Find why two chunks change the result (256-row chunks for 257-512 tokens: worst
      probability shift 0.032; 512-row chunks for 513-1023 tokens: 0.005). A chunk boundary
      should be exact. Check the bf16 conv state and the delta-rule state carried between
      chunks. If the boundary becomes exact, enable two 256-row chunks for 257-512 tokens
      (OFLM_OPEN_TINY_MAX=512 was 4 % faster).

### Attention kernel (dit_fa256b), on hold
At 31 TFLOPS now (8K: 6.52 ms). It runs only for prompts over 2048 tokens; at 32K it is about
25 % of the time.
- [ ] 35 TFLOPS (8K <= 5.75 ms): merge the qk and pv calls into one generated schedule
      (prologue and epilogue, ~240 cycles a chunk); move the l update and the max check into
      the exp assembly; find the exp-pass hazard at a period of 6-7 cycles (8 is bit-exact).
- [ ] int4 V with stream rebalance: V int4 with one exponent per d column per 64-key chunk;
      fifo B carries K piece 3 plus V pairs, so each stream moves 13824 B a chunk. Data floor
      5.40 -> ~4.05 ms at 8K. Simulated accuracy (d3_kv_sim.py): k8v4b64 mean |dp| 0.0028, no
      flips.
- [ ] Then re-plan for 45-50 TFLOPS.

### GEMMs
- [ ] The int4 GEMM is bound by the A stream into each core (94 % of 4 B/cycle with no
      math). More speed needs A reuse over more columns per core, which the 64 KB L1 does not
      hold with 128x128 C tiles. Expected gain at most ~15 %.
- [ ] Run the L1 bank-layout (stack size) sweep for the bfp16 set too; only int4 has it
      (0x2000; 256-row tier 0x3000).

### Operations
- [ ] Never run NPU and GPU jobs together on the Strix Halo: AGESA 1.0.0.2 has a Data Fabric
      MCE bug with both active (ROCm/TheRock#8709). Fixed in AGESA 1.0.0.2b; ask GMKtec for
      a BIOS with it.
- [ ] Back up the model dirs (`~/models/d3-NPU2`, `~/models/d3-src`) and the kernel sets
      off the machine. Git ignores the kernel sets; the export rebuilds them.
