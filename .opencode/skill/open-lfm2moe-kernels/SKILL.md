---
name: open-lfm2moe-kernels
description: Build, verify and run the open XDNA2 kernels for LiquidAI LFM2-MoE (LFM2.5-24B-A2B, LFM2.5-8B-A1B; model_type lfm2_moe) from a Q4_0 GGUF -- the lfm2moe recipe, the merged one-context layer image (lm.py), the separate-image fallback (cx/dx *_MIX + ex.py), the host sigmoid router and the q8 head. Use when rebuilding these sets, adding another LFM2-MoE size, or debugging a wrong token / ERT timeout on this family.
---

# LFM2-MoE on the open kernels

Read `npu-model-porting` first for the general rules. Source: `open_kernels/recipes/lfm2moe.py`
(the docstrings explain each choice), `designs/lfm2moe/{lm.py,ex.py,gen_kernels.py}`.

## The layer

- Layers < `num_dense_layers` (2): LFM2's `cx` unchanged (conv + dense FFN), own context.
- Other layers, one hardware context `layer`, three streams of `lm.py`:
  `cm` (conv block) or `dxm` (attention block, built as `lm_am`) -> `hostroute` -> `ex`.
  The mixers leave `res` (f32) and `xm = ffn_norm(res)` (bf16) in `act`.
- Router (host, `Core::host_route`): `s = sigmoid(Wr @ xm)`, top-k of `s + expert_bias`,
  weights = unbiased `s` / (sum + 1e-6) * scale. Writes idx u32[8] at +1024 and the weights
  at +1056 of the record, then moeroute2 re-points ex's fills (`layout.moe.base`).
- Experts: std_perm 64-row bands, `[up | gate]` per expert from POOL_EXP, down from
  POOL_DOWN_EXP. 24B: 8 cores x (3 up/gate bands, 4 down bands).
- Head: q8 from the Q6_K tied embedding (`lmhead_q8_gguf`); final norm is `token_embd_norm`.
- Pool: q4nx chunk layout (bf16 scales, 5120 B), packed from the GGUF's Q4_0 at load. Bit-identical
  to the f32-scale layout (`OPEN_LFM2MOE_QUANT=q4_1_f32` at export, kept for A/B) and 1/6 smaller.
- When the expert core count differs from the mixers' (8B: 4 vs 8) the recipe falls back to
  three images (`cxm`, `dxm`, `ex`). `OPEN_LFM2MOE_ONE_CTX=0` forces that for an A/B.

## Prompt route (prefill)

Engine `step_prompt_lfm2` (core.cpp), selected by `--gemm-block` on the CLI and by default in
the engine's prefill. Layer-major over the whole prompt:
- Mixers (`prompt_mixers`): projections on `gemm_n{N}_k{K}` (gemm_q4_prefill, T = 256,
  bfp16, pipelined and double-buffered); short conv and attention on the host
  (`attention_tiles`, block_host.cpp); standalone `short_conv_mix` / `dx_mix` images.
- Router: host, same math as decode.
- Experts (`experts_batch`): `eb64` / `eb16` (designs/lfm2moe/eb.py, `EB_BFP=1`), 16 token
  slots per expert visit, x/y double-buffered so host gather/scatter overlaps the NPU.
- 4009-token prompt, 12.9 s: mixers 6.5 s (gemm 3.2, attention 2.35, tile 1.2,
  read+transpose 1.1), experts 6.15 s (NPU run 5.9; prep and read overlap it).

## Model directory

```
~/models/LFM2.5-24B-A2B-GGUF/
  model.gguf -> LFM2.5-24B-A2B-QAD-Q4_0.gguf     (LiquidAI/LFM2.5-24B-A2B-GGUF)
  config.json tokenizer.json tokenizer_config.json chat_template.jinja   (LiquidAI/LFM2.5-24B-A2B)
```
`config.json` is required: the GGUF's per-layer `head_count_kv` array is not readable as a number.

## Build

```
export PATH=/opt/xilinx/xrt/bin:$PATH
OPEN_KERNELS_UNVALIDATED=1 ironvenv/bin/python open_kernels/export_qwen36_kernels.py \
    --model-dir ~/models/LFM2.5-24B-A2B-GGUF -j 3          # -> src/xclbins/LFM2.5-24B-A2B-GGUF/open_kernels
```
Sets: `cx_dense` (dense layers), `lm_cm lm_am lm_ex` (one image), `ln`, `lm_head_q8`, and
the prompt sets `short_conv_mix dx_mix eb64 eb16 gemm_n*_k*`; ~6 min. `UNVALIDATED` is for gemv K = 11776 (the dense FFN) and the expert points. The three
`lm_*` xclbins must be equivalent apart from stamps (`xclbin_equivalent`); the export's
manifest points the context at `lm_cm/final.xclbin`.

## Verify (in this order)

1. Reference on a slice: `python -I open_kernels/model/replica_lfm2moe.py --gguf <dir>/model.gguf --ids 124894,124899,5922,207,2992,355 --layers 8 --out ref8`
   (needs `pip install gguf numpy`; ~2 min for 8 layers).
2. NPU on the same slice: `src/open_qwen36/build/open_qwen36_cli --model <dir> --kernels <set> --ids <same> --max-tokens 1 --layers 8 --prefill-logits --dump-logits npu8/y`
3. `python -I open_kernels/model/compare_lfm2moe.py --gguf <dir>/model.gguf --ref ref8 --npu npu8/y`
   -> every position `exact >= 0.9999`, argmax same. (2026-10-13: 0.99996+ at 8 layers,
   0.99987+ on the full 40.)
4. Chat: `python src/open_qwen36/chat.py "What is the capital of France? Answer briefly." --exe src/open_qwen36/build/open_qwen36_cli --model <dir> --kernels <set>`
   -> reasons in `<think>`, answers Paris.

## Results (Strix Halo, 2026-10-13)

| | decode | resident |
|---|---|---|
| 24B, one context, bf16 scales | 44 ms/token (22.8 tok/s) | ~16 GB |
| 24B, one context, f32 scales | 49 ms/token (20.3 tok/s) | 19 GB |
| 24B, three images | 106 ms/token | 19 GB |
| 8B, three images, bf16 scales | 57 ms/token (17.6 tok/s) | ~6.5 GB |
| 24B prompt, 4009 tokens | 311 tok/s (12.9 s) | |
| 24B decode after a 4K prompt | 63 ms/token (15.8 tok/s) | |


Prompt history: 19.5 (token by token) -> 35 (batched experts) -> 130 (batched mixers) ->
186 (overlap, pipelining) -> 209 (persistent scratch) -> 273 (bfp16 experts) -> 311 (bfp16 GEMMs).

24B step (one context): ex 38 x 0.64 ms, mixers ~0.95 ms (includes the queued ex wait),
cx 2 x 2.1, head 5.9, router 0.8.

## Known gaps

- Not wired into the `oflm` app (`model_list.json`, an AutoModel adapter); the CLI + chat.py
  drive it. The LFM2 adapter selects the open engine when a kernel set exists, so a
  `model_list` entry with family `lfm2` is the likely path.
- The prompt route needs the bf16-scale pool (q4_1, the default).
- The 8B has no prompt route: `eb` needs ff a multiple of 512 (8B ff = 1792).
- The 8B has no merged image (expert core count 4 vs mixer 8).
