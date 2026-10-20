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
