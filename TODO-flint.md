# TODO: Flint (RecursiveSI/flint) on the NPU (branch feat/flint)

Targets (the A3B push's): prefill >= 1000 tok/s and decode >= 30 tok/s at every context, 1K to 128K,
with the GPU busy.

Model: FlintForCausalLM, 39.8B total / ~2.5B active. Hidden 2048, 49 layers: layers 0-1 a dense
SwiGLU (ff 6144), 47 MoE (256 experts of ff 512, top-6 sigmoid routing renormalised, an ungated
shared expert of ff 1024). Every layer GQA attention 32 q / 4 kv heads at head dim 128, per-head q / k
RMSNorm, learned sinks; 35 layers a 513-key sliding window with RoPE (theta 1e4), 14 full layers
without positions (NoPE). Vocab 150144, untied head. No MTP head; no drafter exists (2026-10-21).

## How it runs

- Weights: `RecursiveSI/flint-int4-gguf` (Q4_1, the MLX export's trained INT4 codes) converted by
  `utilities/gguf-utils/flint_to_q4nx.py` into a q4nx container: the 4-bit codes copied, d / m
  fp16 -> bf16 (RNE; the only loss), f32 small weights kept f32 (the router transposed to [hid, E]).
  Model dir: `~/workspace/npu-scratch/flint/model` (config.json, tokenizer files, chat_template.jinja;
  tokenizer_config.json gains eos_token_id [150001, 150003, 150005, 150018] and bos_token_id).
- Kernels: `open_kernels/export_flint_kernels.py OUT --reuse <A3B set>`: dit2 GEMMs (q|k|v 5120x2048,
  o 2048x4096, the shared expert's SwiGLU pair at ff 1024, the dense layers' at ff 6144; tiers 1024 /
  512 / 256), A3B's mbi expert streams (same geometry and pool offsets: copied), dit_fa256b causal
  streams at 32 / 4 heads (lk 1024..132096) and two sliding-window streams (`fa.win`). Set: kf2.
- Engine (src/open_qwen36): manifest fields `gate`, `sinks`, `window`, `shared_gated`, `ffn_key`,
  `state.ring`, per-type `pool_bytes`; layout.router gating sigmoid. Prompt pipeline and CPU decode
  carry sinks, the window, NoPE (rot 0), no output gate, sigmoid routing, the ungated shared expert at
  its own width, dense layers on `gemm_dit.ffn_dense`. Sliding layers keep a 4096-row sliding KV
  buffer (`Core::kv_room` / `kv_ptr`, checkpoint / restore / seek aware). Head: q4_1 on the CPU
  (`gemv_q41_rows`).
- NPU attention: Flint's head dim 128 runs zero-padded in dit_fa256b's 256; the sinks ride as a 64-key
  chunk in front of every kv head's keys (`fab_sink_k`: key 0 carries head g's sink in padded dim
  128 + 8 g, the Q holds log2(e) there; keys 1..63 -1e4), so every key index and q_off shift by 64.
  Sliding layers: one stream per tier (1024 queries over 2048 keys), the window mask in the kernel
  (rtp[7]; chunk 0, the sinks, never masked), from OFLM_OPEN_FA_W_MIN prompt tokens (4096).
- App: family `flint` (`modeling_flint.cpp`): the template's turns, `<think>` reasoning, tool calls
  `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, the system-prompt cache
  (the system part ends at the first user / assistant / observation / developer token), sampling
  temperature 1.0, top-p 0.95, top-k 20. Model list entry `flint:40b-a2.5b`.

## Accuracy (against llama.cpp PR #30928 on the same GGUF, CPU, f16 KV)

- 15-token chat: 48 greedy decode tokens identical; prompt logits mean KL 0.023, argmax 13 / 15 (the
  two others near ties).
- 3109-token code prompt: the first 6 decode tokens identical. Prompt logits: median KL 0.02, but
  ~5 % of positions KL > 1. llama.cpp itself with a bf16 KV cache against its f16 run gives the same
  statistics (mean KL 0.29, median 0.007, KL > 1 at 56 of 1198, argmax 1060) as the engine's CPU path
  (0.31, 0.007, 59, ~1040): Flint is sensitive to KV precision, and the engine's bf16 KV is the cause.
  The pipeline (bfp16 GEMMs, int8 experts) adds a little (mean 0.45, argmax 85 %), the NPU attention
  (bfp16 Q / K / V / P, scores kept in bf16) a little more (argmax 75 % on the 3109-token prompt).
  Lever if needed: an f16 (or f32) K cache on the host paths; G in f32 in dit_fa256b.

## Fixed on the way

- dit_fa256b's lazy rescale missed a NaN row sum: a score past the running max by more than 128
  (log2 units; Flint's last layer) overflowed exp2, and `sum >= 2^TAU` is false for NaN. Now
  `fa_sums_bad` also rescales on an inf / NaN sum.
- One context: an attention (fab) kernel right after an expert (mb) run hung (ERT state 8; Flint at
  8K, every time), and an mb run after fab once. A configuration replay in between did not help; a
  real dit2 run did. `start_run` now runs the smallest dit2 stream on zero buffers between the two
  designs (OFLM_OPEN_ONE_CTX_FLUSH; 0 off). Root cause still open (stream / DMA state of a run, not
  of the configuration; the shim DMA has no reset bit).

## Results (oflm bench, one length a process, GPU idle: the trainer was not running)

| Context | Prefill tok/s (port, kf2) | Prefill tok/s (kf3) | Prefill tok/s (now, kf4: hd 128) | Decode tok/s (port) | Decode tok/s (kf3) | Decode tok/s (now, kf4) |
|---|---|---|---|---|---|---|
| 1K | 436 | 424 | 431 | 21.9 | 30.9 | 29.8 |
| 2K | 564 | 559 | 628 | 21.9 | 31.1 | 30.8 |
| 4K | 653 | 668 | 740 | 21.8 | 33.1 | 30.0 |
| 8K | 954 | 964 | 1103 | 22.2 | 31.4 | 33.1 |
| 16K | 981 | 1006 | 1169 | 19.4 | 25.4 | 26.4 |
| 32K | 925 | 973 | 1170 | 16.7 | 22.0 | 21.7 |
| 64K | 771 | 810 | 1001 | 13.3 | 17.2 | 17.1 |
| 128K | 570 | 590 | 757 | 9.8 | 11.4 | 10.9 |

## Constraint tracker (2026-10-21, GPU idle unless noted)

| Target | Constraint now | Evidence |
|---|---|---|
| Decode, short context | the NPU half of the experts: mb_s8 streams 8 experts' weights for 3, ~0.5 ms a layer, longer than the CPU's whole MoE (0.24 ms) | CPU trace: experts 25 ms of 42; OFLM_OPEN_DEC_NPU=0: 27 ms a token |
| Decode, 32K-128K | CPU attention over the full layers' KV (14 layers, ~1 KB a key in int8: 1.9 GB a token at 128K) | mixer 12.6 ms at 32K |
| Prefill 1K-4K | NPU experts streaming every expert's weights (DDR) + host attention / tails | 4K: NPU 69 %, host 80 % |
| Prefill 16K-128K | NPU: experts (14.2 s at 32K) and attention (15.1 s: window 5.8, full ~9.3, both at head dim 256 padded) | pipe trace 32K: NPU 100 %, host threads ~50 % |

| ID | Area | Item | Kind | Measured now | How to measure | Expected gain | Status |
|---|---|---|---|---|---|---|---|
| D1 | decode | NPU expert split slower than the CPU with the GPU idle; it also runs (empty) at share 0 | placement | 42 -> 27 ms a token at 4K with OFLM_OPEN_DEC_NPU=0 | CPU decode trace; per-token A/B | -15 ms a token (GPU idle); load-dependent | done (0fb0433): OFLM_OPEN_DEC_NPU_AUTO picks the faster from measured MoE ms (probe 2 of 32 tokens); bench decode 21.9 -> 31 tok/s at 1K-8K, GPU idle. Beside a busy GPU still to measure |
| D2 | decode | full-attention KV read per token | bytes | 12.6 ms at 32K | mixer ms by position | int8 is on; int4 KV would halve (accuracy risk: Flint is KV-sensitive) | open |
| D3 | decode | int8 flash decode attention: per 32-key block an int8 -> bf16 transpose of K, 3 instructions a value | slow code | 128K: 36.9 ms a token for the 14 full layers (52.6 GB/s; the CPU reads 117) | block_host microbench (attention_block, Flint geometry), engine bench decode | -25 % attention | done: the int8 copy in blocks of 32 keys in the dot products' order, sign-magnitude, 64 values to bf16 in 5 instructions (2 vpermi2b lookups), 8 queries a pass; bit-identical (microbench 10 lengths, engine decode logits). 128K 36.9 -> 26.9 ms, the copy's build 500 -> 182 ms; bench decode 32K 21.7 -> 23.0, 128K 10.9 -> 12.9 tok/s. 32 attention threads: 5-6 tok/s (oversubscribed) |
| D4 | decode | no drafts (no MTP head, no drafter) | flow | 1 token a pass | server: a code edit and a free-text reply, 1500 tokens, temperature 0 | edits / quotes 1.5x | done: prompt lookup alone (OFLM_OPEN_PLD, default on), the draft length chosen per pass from the measured verify ms and the acceptance (pld_choose, a shadow when not drafted). Edit 35.9 -> 53.9 tok/s, free text 25.5 -> 27.7. Exact for any sampler: a pass's rows bit-identical to single steps (OFLM_OPEN_SPEC_CHECK, 79 passes) after the window kernel's chunks moved to a fixed grid |
| P1 | prefill | expert slots part-filled: top-6 over groups of 4096 tokens leaves ~96 tokens an expert, 2 slots of 64 | waste | 14.2 s at 32K | OFLM_OPEN_MOE_GROUP 8192 / 16384, slot counts | NPU -10..25 % at 16K+ | done: groups of 8192 from 12288-token prompts (gemm_dit.moe_group / moe_group_min; bit-identical): 32K experts 14.2 -> 12.1 s, wall 35.2 -> 33.5 s; 16384 cut experts more but stalled the pipe (34.4 s); one group a prompt stalled 8K (954 -> 748) |
| P2 | prefill | window attention streams 2048 keys for ~576 needed, and the fused qk/pv runs every chunk | waste | 5.8 s at 32K | pipe trace FAT fa256bw | -3..4 s at 32K | done, small: 1664-key streams + a live-chunk flag (bit-identical): 5.8 -> 5.6 s. The kernel is DMA-bound: every pass of 512 query rows streams its kv head's whole K / V again (window: ~61 MB a call; full layers at 128K ~4.9 GB a call), so the math skip barely shows |
| P4 | prefill | K / V re-streamed per pass of 512 query rows (16x per kv head: 8 q heads x 2 passes) | waste (DMA) | ~all of the attention time (hd 256) | harness: fa128c_lk65536 with DF_EXPT=nokv | at hd 128: <= 3.5 % | measured: no gain at hd 128 -- the kernel is compute-bound: 52.0 ms, 50.2 with no K / V transfer; products alone 29.3, softmax and loop alone 21.5 (they add up). The softmax (already the hand-scheduled exp2) is the next cost: <= ~13 % of attention, new AIE asm |
| P5 | prefill | 1K-4K: the layer chain host <-> NPU (1K: NPU busy 61 %, host 53 %; the NPU waits on MID 0.41 s, the host on the experts 0.93 s) | flow | 1K 434, 2K 628, 4K 740 tok/s | pipe trace (utilities/pipe_trace.py), bench A B A B | +15-25 % | measured: no gain from the knobs -- window attention on the NPU at 1K (OFLM_OPEN_FA_W_MIN=512) 428 vs 434; 256-row chunks at 1K 438-442 vs 434-442; 512-row chunks at 4K 677-687 vs 732-740 (the extra host attention per chunk). Needs a new overlap (e.g. the next layer's QKV under the experts), not chunk size |
| P6 | prefill | window streams: every core gets the call's 25 key chunks, its 32 rows need ~10 (broadcast to 16 cores) | waste (DMA into the cores) | harness, GPU busy: 4.0 ms a call; 2.0 with no K / V transfer | harness A / B on the same random inputs (q_off 512 / 540 / 576, distinct sinks), engine long1 logits | -40 % of the window calls | done: fa128r (DF_GROUPS=4, DF_WRANGE=13: 8-core groups, 256-row passes, each streaming its own 13 chunks; the softmax's first-chunk test on the call's chunk index), context fawr, for q_off >= 512 (the prompt's first chunk keeps fa128w); bit-identical (harness 3 offsets, long1 3109 logits). Harness 4.0 -> 2.3 ms; 64K (GPU busy) attention 34.9 -> 31.7 s, 758 -> 778 tok/s; 16K no wall change under load (the host the constraint) |
| P7 | prefill | 1K-4K: the host idles during the expert pass (1K: 0.93 s) | placement | 1K: experts 0.9 s on the NPU | OFLM_OPEN_PF_CPU=1, A B A B | +25 % at 1K (quiet estimate) | measured: no gain with the GPU busy and the CPU loaded (CPU 4x my quiet cost: 0.13 ms an expert unit); works (1K: ~110 experts a layer on the CPU, both halves ~2.2 s). Off by default; to measure on a quiet machine |
| P3 | prefill | head dim 128 padded to 256 in dit_fa256b | waste | ~2x attention work | NPU attention s | -4..5 s at 32K, ~-70 s at 128K | done: kf4, dit_fa256b DF_D=128 with the sinks in the runtime parameters (no sink chunk); prefill +2 % (1K) .. +28 % (128K), 32K 973 -> 1170 tok/s; long1 logits vs llama.cpp mean KL 1.31 -> 1.24, argmax 2333 -> 2344 / 3109; decode tokens identical |
| M1 | memory | 128K peak ~45 GB (per-chunk buffers ~9 GB, padded key packs 3.9 GB) | memory | fits 46 GB cap | OFLM_OPEN_MEM_LOG | ring buffers: -6 GB | open |
