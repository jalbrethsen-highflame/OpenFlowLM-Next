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

## Constraint tracker (keep current: update a row after every change)

The constraint for each target with the GPU job running (2026-10-21):

| Target | Constraint now | Evidence |
|---|---|---|
| Decode, all contexts | DDR bandwidth left by the GPU (CPU ~12 GB/s median, NPU DMA ~34 GB/s) | bw probe; decode trace: a pass reads ~3 GB |
| Prefill 1K-4K | DDR for the expert weights (all 256 experts a layer, ~20 GB at 1K) | pipe trace 1K: experts 27 ms a layer |
| Prefill 16K-64K | host and NPU together (each ~29 s at 30K, wall 36 s) | pipe trace 30K |
| Prefill 128K | NPU: attention at the MAC peak + experts | 120K trace: NPU 142 s of ~160 |
| Tests at 32K+ beside the trainer | memory (engine 24 GB after load; gpu-oomd kills the trainer under 14 GB available) | OFLM_OPEN_MEM_LOG |

Items (gain: expected, until measured; measure inside one run where possible):

| ID | Area | Item | Kind | Measured now | How to measure | Expected gain | Status |
|---|---|---|---|---|---|---|---|
| P1 | prefill | RES waits for every pass of a group | flow | k6 G 4096 with the GPU busy: host waits UP -> RES only 0.7-1.05 s at 30K (the 10.4 s was with G 16384 / dnc) | pipe trace "host idle by" | OFLM_OPEN_RES_EARLY=1 (RES adds finished passes early; bit-identical): no gain | measured: no gain (opt-in kept) |
| P2 | prefill | x repeats each token's block sums 16x (4.4 KB a pair, 2.3 needed); ~42 GB host writes at 30K | waste | gather 5 thread-s | MID_PROF / moe_ms gather, x sync | -2 to -3 host thread-s, less NPU DMA | open |
| P3 | prefill | shared expert as its own NPU GEMMs. OFLM_OPEN_SHARED_MBI=1: a copy in the routed layout at the pool's end, run as expert 256 of the mbi passes (every row at its gate); no shared GEMMs, input pack or add | waste | 30K, GPU busy: wall 40.9-41.7 -> 38.6 s (-6 %), NPU 29.3-31.0 -> 27.5 s, switches 739-753 -> 569, host 72-73 -> 68 thread-s; KL 1.06e-2 -> 1.18e-2 (argmax 526 -> 522 / 600). With dnc too (k9): 36.5-39.0 against 40.3-41.6 s, NPU-bound 33-35 s | pipe trace A/B | -6 % at 30K | done, opt-in (accuracy trade for the user to decide) |
| P4 | prefill | (30K, GPU busy, k9 + dnc: host 55 thread-s, NPU 37 s, wall 39 s; k6: host 62-72, NPU 27-30, wall 36-40) dnc kernel 6 ms a call against ~1.7 ms of math (dnc saves 13 host thread-s but makes the NPU the limit) | slow kernel | 5.4 s NPU at 30K | pipe trace DNC | 5-10 % at 16K+ | open |
| P5 | prefill | router on the host | placement | ~7 host thread-s | moe_ms router | -5 host thread-s | open |
| P6 | prefill | mixer out f32 -> memory -> bfp16 pack. OFLM_OPEN_A2_DIRECT (default on): the linear layers' chunk4 rule writes the OUT GEMM's bfp16 A blocks itself (bfp_pack.hpp), bit-identical | waste | 30K, GPU busy: MID 30.6-32.1 -> 28.4 thread-s, host 66.6-70.0 -> 65.2-67.4, NPU wait MID -> OUT 4.2-4.6 -> 2.8-3.3 s, wall -0.5 to -0.8 s. The same fusion in GATE (attention) was slower (2.2-2.4 -> 2.6-2.8 s): not kept | pipe trace A/B | done | done (linear) |
| P7 | prefill | expert slots 77 % full; each core receives every visit's full weights | hardware layout | per-core stream floor | null-kernel timing | large, uncertain (new tiling) | open |
| P8 | prefill | empty expert slots from rounding the last pass up | waste | 14 % of slots | slot count in the trace | -2.2 s NPU at 30K | done (ec27d29) |
| P9 | prefill | hardware context switches cost ~5 ms each (next kernel 0.46 -> 5.5 ms); 742 at 30K (~3.5 s NPU), 1699 with dnc (~8.6 s, why dnc did not pay); structural: one expert visit a group | waste | 3.5 s NPU at 30K | trace: kernel time after a switch vs not | -1.6 s at G 8192; more needs half-array designs | open: OFLM_OPEN_CTX_STICKY no gain (k6: 747 -> 712-737 switches; dnc: 1717 -> 1557, NPU 36.9 -> 36.2 s, wall +1.2 s: a dnc call waits for its chunk's host MID, ~25 ms) Groups of 8192 (A/B, GPU busy): switches 745-751 -> 413-421, NPU 27.0 -> 25.1 s, but wall 35.7 -> 38.3 s (host-bound). Switch probe: ~0.17 ms between small designs, ~5.5 ms between full-array ones (not size-dependent: the firmware saves the array state); spatial sharing is off (the driver sets AIE2_TEMPORAL_ONLY for this NPU and firmware). |
| D1 | decode | dense GEMVs only on the CPU. mbv (designs/moe_batch/mbv.py, OFLM_OPEN_MBV=1, set k10): the linear QKV's first 8192 rows on the NPU, the rest on the CPU. A/B by layer parity with the GPU busy: 0.567-0.570 ms a call against 0.543 on the CPU alone (GPU idle: 0.35 against 0.22). The decode CPU reads these weights at ~29 GB/s, not the 12 of the 512 MB probe; the NPU split adds ~0.1 ms x pack / read back + 0.12 ms dispatch | placement | QKV 0.54 ms a call | OFLM_OPEN_MBV_AB parity A/B | none as built; Q 2 might give ~-6 ms a pass but costs the expert split (-10 ms) | measured: no gain (opt-in kept) |
| D2 | decode | half of a pass's experts on the NPU | placement | -15 % MoE a layer | DEC_NPU_AB parity A/B | done | done (5b10a84) |
| D3 | decode | MTP draft length | tuning | k<=4, PCUM 0.4 | acceptance counts (load-independent) | done | done (5b10a84) |
| D4 | decode | MTP layer at q4_1 (first draft 81.5 %, second 66.7 %) | accuracy | - | MTP_LOG per-position acceptance | unknown, a few % | open |
| D5 | decode | MTP sees only the last 256 prompt rows (OFLM_OPEN_MTP_CTX) | accuracy | - | acceptance on a long prompt, 256 vs all | unknown | open |
| D6 | decode | attention at 64K ~5 ms a row, ~30 % of the CPU bf16 peak | slow code | 5 ms a row | CPU_DECODE_TRACE mixer by rows | up to -3 ms a row at 64K | open |
| D7 | decode | each verify row costs 18-20 ms (the experts it adds) | inherent | 18-20 ms | verify ms by rows | only better drafts | noted |
| M1 | memory | dense weights twice (q4_1 pool + bfp16 GEMM copy, 1.6 GB) | waste | 1.58 GB | OFLM_OPEN_MEM_LOG | -1.6 GB (with D1's kernel) | open |
| M2 | memory | expert pass buffers at full size for every pass | waste | 2.4 GB at 600 tokens | OFLM_OPEN_MEM_LOG | -1.9 GB | done (31d5e42) |
| M3 | memory | per-chunk prefill buffers grow with the prompt (0.77 GB at 16K) | waste | 0.77 GB at 16K | OFLM_OPEN_MEM_LOG | ~-1.3 GB at 30K (ring) | open |
| T1 | testing | each CLI test reloads the weights (27 s) | waste | 27 s a test | wall | most of a short test | open |
| T2 | testing | heavy tests heat the CPU to 94 C; the trainer's guard pauses the GPU job | validity | pauses logged | guard log HOT count, GPU busy samples | valid numbers | done (safe_run / safe_bench record both) |

Accuracy note: the MoE path amplifies 1-ulp changes (int8 x roundings, near-tied routing): adding the
shared expert after the routed ones instead of before moved the 600-token logits by KL 7.8e-3, as
large as the whole quantization error (1.06e-2). Keep changes bit-identical where possible; judge
others against the reference, not against the previous build.

Test rules beside the GPU job: cap every run (systemd-run MemoryMax), start only with MemAvailable
>= cap + 20 GB, stop the run's own PID if MemAvailable < 20 GB, record GPU busy samples and the
trainer guard's pauses; compare inside one run (A/B by layer parity, kernel ratios, slot counts,
acceptance counts), not whole runs.

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

Decode (CPU decode on the q4_1 pools, 16 threads, GPU job running), tok/s:

| Prompt | NPU decode (before) | CPU decode | CPU + MTP (k = 2) |
|---|---|---|---|
| chat, ~45 tokens (prose / code) | 9.9-11 | 14.6 | 20.8 / 22.0 |
| 600 tokens | 10 | 14.7 | |
| 30K chat (C++ source + question) | ~4.2 (32K bench) | 12.8 | 15.6 |

- gemv_q41 (33c0da5): VNNI + VBMI on the pool bytes in place; nibbles centred ((q - 8), m + 8 d): the
  int8 input's error 1.5 % -> 0.55 %. 100 GB/s on one row in a microbench (L3-assisted), ~40 GB/s in
  decode with the GPU busy: decode is bandwidth-bound (~2.2 GB a token: mixers 0.8, experts 0.71,
  q8 head 0.5, router, state).
- The MoE tail on the CPU (163e7f0): router split over threads (9.9 -> 3.8 ms a token), every expert
  GEMV of a phase in one parallel loop.
- MTP (c477a69): the HF model's mtp.* (19 tensors, fetched by range requests,
  ~/workspace/npu-scratch/a3b/mtp/fetch_mtp.py) as q4_1; the MoE MTP layer through moe_cpu. k = 2
  beats 1 and 3 (a verify row adds distinct experts). Acceptance 50-65 %. More MTP context (4096
  rows) did not raise acceptance at 30K and cost 2.8 s.
- At 30K a step reads ~650 MB of KV (mixer 21 ms of 77): KV in int8 is the next lever there.

Through the oflm app (`oflm bench`, 32 tokens a stage, MTP on, GPU job running), tok/s:

| Context | Prefill (closed / open at start) | Prefill now | Decode (closed / open at start) | Decode now |
|---|---|---|---|---|
| 1K | 163 / 142 | 296 | 13.0 / 9.9 | 19.1 |
| 2K | 195 / 150 | 368 | 11.2 / 9.8 | 18.9 |
| 4K | 227 / 168 | 466 | 11.2 / 9.0 | 22.7 |
| 8K | 236 / 152 | 497 | 10.3 / 7.8 | 16.7 |
| 16K | 262 / 137 | 632 | 9.4 / 5.9 | 15.1 |
| 32K | 244 / 113 | 648 | 7.8 / 4.2 | 16.1 |

CLI, chat prompts of C++ source + a question, kernels to 130048 keys (k4):

| Prompt | Prefill | TTFT | Decode (MTP k = 2) | Drafts accepted |
|---|---|---|---|---|
| 30K | ~700 | ~45 s | 15.6 | 48 % |
| 64K | 698 | 92 s | 21.8 | 60 % |
| 120K | 592 | 203 s | 11.9 | 58 % |

At 120K the NPU is busy 97 % of the prompt: experts 121 s, attention 49 s. The int8 KV copy halves
decode attention at 128K (62 -> 32 ms a step in the bench; kv8, fa19d66).

Next for prefill: an expert core on the int8 x int4 MAC mode (the q4_1 nibbles are already the B tile's
[k][16 rows] order; (q - 8) centring as gemv_q41) and a token split over a column's 4 rows (weights
broadcast from the mem tile: 128 tokens a weight stream). The current kernel is bound by DMA and core
together, so neither alone gains much.

## mbi: the experts on the int8 x int4 MAC mode (2026-10-20, 065cb6c, 1a28e22)

designs/moe_batch/mbi.py (export with OPEN_QWEN36_MBI=1; kernel set ~/workspace/npu-scratch/a3b/k5):
one expert per column, the four rows split a visit's 64 tokens, weights broadcast from the mem tile;
the pool's routed experts transformed at load (signed nibbles, int16 d / mc at one shift an expert
matrix), x packed by the host as int8 k-group elements, bf16 y. Kernel alone, 256 slots of 64
tokens: 13.3 ms with the GPU idle (DMA floor 10.5 ms; mbe 16.0 ms for half the tokens).

Prefill, CLI, chat prompts of C++ source, tok/s:

| Prompt | GPU | k4 (mbe) | k5 (mbi), group 4096 | k5, group 8192 |
|---|---|---|---|---|
| 30K | idle | 916 | 1403 | 1489 |
| 64K | idle | | 1311 | |
| 120K | idle | 789 | 1133 | |
| 30K | busy (load 12) | 598 | 807 | 792 |
| 120K | busy from mid-run | | | 842 |

Accuracy, 600 tokens against per-token: mean KL 9.95e-3 -> 1.06e-2, argmax 535 -> 526 of 600 (the
bf16 tanh silu adds ~1 % to h; MBI_SILU_FP32 is the exact one at +18 us a band).

At 120K (GPU idle) the NPU is busy 102 of 106 s: experts 47.8 s, attention 36.4 s (at the bfp16 MAC
peak, ~16 TMAC/s), GEMMs ~18 s. With the GPU busy at 30K the host is the limit (NPU idle 4.6-9 s):
DeltaNet conv + rule 9.5 s, MoE tail (router 2.2, gather 3.4, x sync 1.1) ~10 s, scatter 4.8 s.

Lessons from the kernel work:
- fp32 multiplies are emulated on this core (~250 cycles a 64-lane pair): split fp32 into bf16
  high + low and use two native bf16 MACs (mbi_epi); vecmath's fp32 silu costs ~34 cycles an
  element, the bf16 tanh form ~5.
- A kernel stack that overflows writes over the next object-fifo buffer: wrong results, no error
  (h row group 0 of every later band). Read the frame size (llvm-objdump: paddxm [sp]).
- A static variable in a core kernel hung the design (TDR); so did __builtin_aie2p_read_tm.
- #pragma unroll 8 on the block loop spilled a 7 KB frame; unroll 4 is faster (1.46 us a call).
- GPU load slows the NPU cores ~1.6x (probe times), not only DDR.
- Pipeline(3) waits per task: one task per up/gate band and one for the down (pool order) cut the
  DMA floor 19.5 -> 10.5 ms; issue the next round's x fill before the down fills.

Next: the host stages under GPU load (DeltaNet conv / rule, the MoE gather and scatter), then the
experts' core rate (they bind the NPU when the GPU is busy), then MBI default on.

## Constraint pass (2026-10-21, GPU busy; b95d8b2 .. 5a7c17a)

Pipe trace (`OFLM_OPEN_PIPE_TRACE=<csv>`, `utilities/pipe_trace.py`) at 30K showed the host as the
constraint: one host thread 100 % busy, NPU 47 % busy (it waited on TAIL->UP and MID->OUT).

| Change | 30K prompt, s |
|---|---|
| before | 63-64 |
| two host threads (`OFLM_OPEN_HOST_THREADS`, default 2; an OMP team of 8 is ~as fast as 16 under load) | 44.5-46 |
| AVX-512 mbi_quant_rows, bf16x_dot gate, z read as bf16 rows | 36.6-37.6 |
| + dnc (`OFLM_OPEN_DNC=1`) | 39-40 |

After the host fixes: NPU 83 % busy (experts 20.6 s, attention 4 s, mixer GEMMs 5.5 s); host
~63 thread-s (DeltaNet MID 29.7, router + gather 19.7, scatter 7.2, norms / packs 7.4).

dnc (designs/deltanet/dnc.py): the gated delta rule on the NPU, 16 tokens a step, bfp16 MACs, the
state slice (16 KB) in L1. O 1.3 % / S 1.0 % from the f64 recurrence; 4.0 ms a 1024-token layer.
In the model KL 1.06e-2 -> 1.31e-2 (argmax 528/600). It does not pay yet: host MID 29.7 -> 18.3 s
but MIDB (gate + pack) adds 7.5 s, and the NPU gets 5.5 s more and becomes the constraint (95 %).
To make it pay: one record a key head (both value heads replay it), the pack in phase 1, fewer
syncs (27 MB a call), the kernel to ~1.7 ms.

Kernel sets: k6 (model open_kernels), k7 = k6 + dnc by manifest merge. Do not export A3B sets from
this worktree with --no-build while a qwen35-9b export runs here: the build dirs (attention ag /
fa256bc, dit2) do not name the geometry and the 9B build replaces them (k7bad: NaN).

Next: a hand-scheduled expert up/gate + down loop (the compiler gives ~27 bundles where 12 MACs are
needed), the router on the NPU, dnc made to pay, NPU attention for long-context decode.

## Round 2 (2026-10-21 morning, GPU busy; f02e85c .. 5b10a84)

Experts (mbi) under GPU load: the full kernel ran at the DMA-only floor (256 slots 18.5 ms), so the
expert time is bytes and rounds, not MACs. Three limits meet near 11-13 ms a 256-slot call (GPU
idle): DDR, the stream into each core (each core receives every visit's full weights), core compute.
- MBI_P (OPEN_QWEN36_MBI_P, opt-in): P columns share a weight stream. DDR bytes ~halved, rel 1.7 %
  as P 1, but the per-core stream then binds (P 4 null 12.4-13.7 ms) and P adds empty slots for
  experts with few tokens. In the model <= 5 % a call.
- MBI_HKG (opt-in): one h scale a k-group; no faster when DMA binds, rel 2.2 % against 1.7 %.
- Engine: a group's remainder pass takes the largest stream that fits (180 -> 128 + 32 + 16 + 8):
  14 % of streamed slots were empty experts; 30K experts 17.6 -> 15.4 s, same logits.
- Groups of 16384 cut slots 27 % but the wall got worse: two groups a layer leave no overlap for
  the scatter (host waits UP -> RES 10 s).

dnc made cheaper for the host (same logits for the layout steps): one record block a key head
([aux even | K, Q | aux odd], the kernel reads the aux at 0 or 8192); phase 1 writes the bf16
tiles straight into the buffers (pack + sync 0.53 -> 0.14 s at 4K); vector gated norm (2.8 -> 0.8
ms a chunk; KL 1.31e-2 -> 1.34e-2). At 30K (no CPU job): host 57-60 -> 47 thread-s but the NPU
goes to 100 % (38 s: experts 15.4, dnc 5.4, QKV 4.9, FAT 4.3, shared expert 4.3, OUT 2.3), so the
wall is 2-4 s worse. Still opt-in.

Bandwidth under the GPU job (the decode constraint): CPU reads median 12.4 GB/s (p10 11, p90 15,
bursts to 109 when the GPU pauses), the NPU's DMA ~34 GB/s at the same time. Decode a pass reads
~3 GB (dense 0.93, experts ~1.6 for 3 rows, head), so it is DDR-bound; a verify row costs ~18 ms
(the experts it adds).
- OFLM_OPEN_DEC_NPU default 16 (half a pass's experts on the NPU): -15 % MoE time a layer, A/B by
  layer parity (OFLM_OPEN_DEC_NPU_AB); shares 40-60 % equal, 75 % worse, 100 % worst.
- MoE MTP default k <= 4 with PCUM 0.4: 16.5 / 19.8 -> 16.9 / 22.2 tok/s on chat1 / chat2. A draft
  vocabulary of 32768 ties with 65536.
- 64K decode measured 19.3 tok/s in a lighter phase of the GPU job; the mixer (flash kv8, one KV
  read a pass) costs ~5 ms a row at 64K, ~30 % of the CPU's bf16 peak.

Run-to-run spread is 2x with the GPU job's phases (30K: 27.5-59 s). Compare within one run
(A/B by layer parity, kernel ratios in the pipe trace, slot counts), not whole runs.

App bench 128K..1K (05:27, k6, this round's binary). The GPU job was at 100 % at the start and had
ended by the finish (train_ft.py gone), so these are mostly GPU-idle numbers, not the target case:
1K 636 / 23.3, 2K 777 / 29.5, 4K 945 / 28.8, 8K 1323 / 25.8, 16K 1458 / 37.2, 32K 1470 / 22.1,
64K 1370 / 22.1, 128K 1069 / 23.2 (prefill / decode tok/s).

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
