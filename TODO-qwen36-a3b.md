# TODO: Qwen3.6-35B-A3B on the NPU (branch feat/qwen36-a3b)

Targets (2026-10-20): **prefill >= 1000 tok/s and decode >= 30 tok/s at every context, 1K to 128K.**
Method: profile, remove the biggest bottleneck, re-profile; biggest gains first.

This branch is feat/qwen27b-causal (dit2 int4 GEMMs, causal dit_fa256b to 130K keys, CPU decode
with MTP, flash decode attention) plus the 26 LFM2-MoE commits (layer-major prompt route,
eb experts on the bfp16 MAC array, host router). Model: `~/models/Qwen3.6-35B-A3B-NPU2` (q4nx,
Atomic-Germ). Kernels: `src/xclbins/Qwen3.6-35B-A3B-NPU2/open_kernels` (export_qwen36_kernels.py,
default spec).

Model: hidden 2048, 40 layers (30 DeltaNet, 10 gated attention: 16 q heads / 2 kv heads, hd 256),
256 experts of ff 512 + a shared expert, top 8, vocab 248K. ~3B active weights a token.

## Baseline (2026-10-20, a GPU training job running beside it, CPU boost off)

`oflm bench`, 32 output tokens a stage, tok/s:

| Context | Closed prefill | Open prefill | Closed decode | Open decode |
|---|---|---|---|---|
| 1K | 163 | 142 | 13.0 | 9.9 |
| 2K | 195 | 150 | 11.2 | 9.8 |
| 4K | 227 | 168 | 11.2 | 9.0 |
| 8K | 236 | 152 | 10.3 | 7.8 |
| 16K | 262 | 137 | 9.4 | 5.9 |
| 32K | 244 | 113 | 7.8 | 4.2 |

Open engine, 4009-token prompt (CLI `--gemm-block`, 25.9 s = 155 tok/s): experts (`mb_s256`,
640 calls x 18 ms, NT = 8) 11.7 s; mixer GEMMs (gemm_q4_prefill, T = 256) 5.2 s + transposes
1.2 s; host attention 2.55 s; DeltaNet rule 0.72 s, conv 0.49 s; tail 0.97 s; moe prep 0.6 s.

Decode step at 4K, 100 ms: lx0 (DeltaNet mixer) 30 x 1.8 ms, ax0 (attention) 10 x 3.2 ms, lx1 /
ax1 (experts) 40 x 0.66 ms, lm (q8 head) 14.5 ms.

## Progress (2026-10-20, GPU job running beside it)

Prefill, CLI `--gemm-block`, tok/s:

| Prompt | Baseline | mbe experts (block route) | Prompt pipeline + grouped MoE |
|---|---|---|---|
| 4K | 155 | 212 | 413 |
| 8K | | | 426 |
| 16K | | | 612 |
| 32K | 113 (bench) | | 706 |

- mbe (5be6774): eb's core program over moe_batch's taps, bfp16, 32 tokens a visit: experts 11.7 -> 4.6 s at 4K.
- The prompt pipeline (fa9a4f0): dit2 GEMMs (bfp16: b4 repack of q4_1 loses 12-13 % of each weight and
  flipped routing, KL 0.18 after 4 layers; a block Hadamard along K only took it to 11-12 %, the weights
  have few outliers), causal dit_fa256b at 16 / 2 heads (rel 1 % against host attention), a MoE stage:
  router + gather on the host, shared expert on the device FFN, mbe passes, scatter.
- Accuracy, 600 tokens, all 40 layers, against the per-token reference: pipeline mean KL 1.0e-2, top-1
  535/600; the old block route 4.6e-3, 550/600 (bfp16 activations and attention). Not yet checked
  against the bf16 model.
- MoE passes over 4096-token chunk groups (c937102): one visit list per group, not per chunk (experts
  6.8 -> 4.8 s at 4K); register-blocked router (4.6 -> 1.8 ms per 1K tokens, same bits), AVX-512 scatter.
- Now: the experts are 76 % of NPU time at 32K and bound by both DMA and core (MB_NULL_MM stream alone
  19.7 ms vs 16.4 full per 256-slot pass; the bfp16 step ~2.5 us a call). Next lever there: a faster
  expert core (hand-scheduled q4_1 dequant + bfp16 MAC), then fewer weight bytes per token.

## Plan (biggest first; re-profile after each)

Prefill:
- [ ] P1 experts: bfp16 MAC step in moe_batch (eb_step_bfp), then more tokens per expert visit
      (NT 8 -> 32: the down streams h by k-tile, as eb does). Weight traffic a token at NT 8 caps
      the experts at ~660 tok/s.
- [ ] P2 mixer GEMMs on dit2 int4 (the 27B's), M >= 1024 rows.
- [ ] P3 attention on dit_fa256b (causal, to 130K keys; 16 / 2 heads).
- [ ] P4 host stages (DeltaNet rule / conv, tail) from the 27B pipeline.

Decode:
- [ ] D1 the DeltaNet and attention mixer kernels (lx0 1.8 ms, ax0 3.2 ms against ~0.4 ms of
      weight stream each), or CPU decode for the MoE layers (int4 VNNI, the 27B's).
- [ ] D2 the head: the 27B's two-stage CPU head (14.5 ms on the NPU now).
- [ ] D3 MTP (the HF model's mtp.* weights), flash decode attention past 16K.
