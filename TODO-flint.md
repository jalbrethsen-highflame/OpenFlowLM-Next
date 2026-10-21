# TODO: Flint (RecursiveSI/flint) on the NPU (branch feat/qwen36-a3b)

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

## Results

See the sweep table below (one length a process, `oflm bench`, GPU state per row).
