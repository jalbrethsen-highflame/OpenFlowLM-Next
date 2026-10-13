---
name: npu-model-porting
description: Playbook for porting a NEW model architecture to the open XDNA2 engine (open_kernels recipes + IRON designs + src/open_qwen36) -- environment setup on a fresh Linux box, the reference-first workflow, slice testing, how the manifest/engine fit together, IRON design rules that hang or corrupt silently (drain ordering, shim/L1/program-memory budgets), GGUF-direct gotchas, profiling, and the one-context merge. Use before adding a family, when a new design hangs with ERT state 8, when NPU logits disagree with a reference, or when decode is slower than the kernels' own timings suggest.
---

# Porting a model to the NPU (open engine)

Learned bringing up LFM2-MoE (`open-lfm2moe-kernels`, 2026-10-13). Every rule below cost
real time once. Read the whole thing before writing a design.

## 0. Environment (fresh Fedora / Linux box, no sudo)

- XRT is in `/opt/xilinx/xrt`: put `/opt/xilinx/xrt/bin` on PATH for every export
  (`xclbinutil`, `aiebu-asm`). The NPU must show as `/dev/accel/accel0`.
- Toolchain: `uv venv --python 3.14 ironvenv && uv pip install --python ironvenv/bin/python -r ironvenv-requirements.txt`
  (mlir-aie 1.4.2 + Peano; cp311-cp314 wheels). Check: `ironvenv/bin/python -c "import aie.iron"`.
- `git submodule update --init --recursive` (tokenizers-cpp) before any CMake configure.
- No `readline-devel`? The engine CMake now treats readline as optional.
- Fast engine loop: `cmake -S src/open_qwen36 -B src/open_qwen36/build -G Ninja && cmake --build src/open_qwen36/build --target open_qwen36_cli`.
  The CLI is the engine without the app: `--model DIR --kernels DIR --ids a,b,c --max-tokens N --layers N --prefill-logits --dump-logits PREFIX --quiet`.
- Prove the box first: export an existing family (`utilities/export-kernels.py --specs lfm2-1.2b --skip-bert`)
  and run it with the CLI + `chat.py`. If that fails, nothing new will work.
- Cap memory on shared-memory machines (NPU buffers are host RAM): run builds and engine
  under `systemd-run --user --scope -q -p MemoryMax=9G ...`. Peak RSS counts the mapped
  model file; `Core::load_weights` now drops pages per layer.
- New catalogue points need `OPEN_KERNELS_UNVALIDATED=1` at export (an unvalidated K etc.).
- Unit tests: `ironvenv/bin/python -m pytest specs/open-engine/tests -q`. One test needs
  `libubsan`; without it, deselect `test_small_bank_pointer_arithmetic_in_compiled_cpp`.

## 1. Reference first, then slices

1. Write an fp32/fp64 numpy reference that reads the SAME weight file the engine reads
   (for GGUF: the `gguf` pip package; `gguf.quants.dequantize(t.data[e], t.tensor_type)`
   gives one expert of a merged tensor without dequantizing the rest). Confirm it with a
   full-model greedy chat BEFORE comparing anything (`model/replica_lfm2moe.py` is a template).
2. Read the HF modeling file for exact semantics (router gating, norm placement, which tap
   is "this token"). Do not guess them.
3. Test the engine on `--layers N` slices that cover every layer type (the CLI computes
   logits after N layers). Compare per position with `--prefill-logits`; 6 tokens catches
   KV-cache and recurrent-state bugs that 1 token hides.
4. Separate layer error from head error: score the NPU logits against the reference hidden
   pushed through the engine's OWN head quantization. If that is ~0.9999 and the exact
   score is lower, the head is the problem, not the layers (`model/compare_lfm2moe.py --q4-head`).
   Target: logits corr >= 0.9999, same argmax at every position.

## 2. How the pieces fit (read once)

- `recipes/<family>.py` computes every layout constant; `recipes/manifest.py` writes
  `manifest.json`; the engine (`src/open_qwen36/core.cpp`) only interprets it. A family
  module needs: `recipe layout geometry pack_plan programs builds manifest_layout
  hf_config_check GEN_KERNELS KERNEL_SOURCES` (+ `band_bytes`, `chunk_bytes`, `probe_env`
  if dx.py / cx.py import it as their family module). Register it in `recipes/families.py`
  and the config parser in `recipes/spec.py` (`HF_FAMILIES`, `_FAMILY_OF`, `ROLE_TENSORS`).
- Compose existing recipes instead of copying: build a "view" spec of the family the
  design already knows (`lfm2moe.lfm2_view`) and wrap its layout in a dataclass whose
  `__getattr__` delegates, overriding only the totals that grow (POOL_BYTES, AD_BYTES).
- Layer-type names in the manifest are free (`manifest_layers` hook); only the state kind
  (`kv` | `linear`) means something to the engine. Every layer gets the SAME pool size.
- Program steps: `run`, `moeroute2` (router on the NPU), `hostroute` (router on the host,
  `layout.router`). Patches: `attnpos` rewrites arg 3 (KV) and arg 5 (ptab) offsets;
  `moeroute2` rewrites every arg-0 (pool) fill inside the expert ranges -- so the pool MUST
  be arg 0 of an expert kernel, and `layout.moe.base` says where the experts start.
- Host reads of device buffers can be stale right after a dispatch (OPEN-REQUEST-ISOLATION).
  Anything the host reads back (router record, xm) needs a sentinel armed before the step
  (`arm_route_records`) and a re-read loop.

## 3. IRON design rules (each one hung or corrupted silently)

- **Drains before fills.** A core blocks on its depth-2 y fifo after two output elements.
  `ironutil.Pipeline(3)` issues a fill only after the fill three back COMPLETED. If a core's
  y drain is issued after its own weight fills, any core with more than ~3 output bands
  deadlocks -> `ERT state 8 (timeout)`. Issue all y drains first, then interleave fills
  band by band across cores (dx.py's order). It worked at 3 bands/core and hung at 6-7.
- **Budgets are arithmetic, check them before building:**
  - L1 per core ~60 KB: `tab (2.25*K per K prepared) + ms + 2*w element + 2*4 KB x + 2*256 B y + stack`.
  - Program memory 16 KB per core: the build prints "fullest core N of 16384"; aiecc
    reports an overflow only as an invalid ELF.
  - Shim DMA: 2 MM2S + 2 S2MM per shim tile (16 + 16 on NPU2). Place every fifo endpoint
    by hand and write the budget in the docstring (designs/lfm2moe/lm.py uses all 16 S2MM).
- **Band law:** std_perm 64-row bands; a band of K is K/128 chunks; `per_band` must be even;
  the GEMV derives K from it. Whole bands per core, or refuse by name.
- A design that needs a MoE expert width that is not 1/4 of hidden: store experts as plain
  std_perm bands (one dense FFN per expert) and give each core whole bands
  (OPEN-MOE-WIDE-FF; `designs/lfm2moe/ex.py`). Pack with one `std_perm_gguf` op per expert
  and `chunk0 = e * nch`.
- To split an existing fused layer, add a compile-time mode that stops after a stage and
  feeds only the helpers that stage needs (cx.py `CX_MIX`, dx.py `DX_MIX`). Put the knob in
  SPECIALIZE only when set, so every other family's build stays byte-identical.
- Activations wider than one 4 KB x element stream as whole elements; prep kernels take
  the element index and derive the block range (`ex_ph`, `dense_prep_f32`).
- **Do not build narrow partitions** (`from_name("npu2", n_cols=4)`): two 4-column kernels
  that were bit-exact on the full array gave WRONG logits as 4-column partitions. Unexplained.

## 4. GGUF-direct gotchas

- `GgufFile::gguf_name` maps HF names (`model.layers.N.self_attn.q_proj.weight` ->
  `blk.N.attn_q.weight`); a name not starting `model.` passes through, so use GGUF names
  directly where no HF twin exists (`blk.{l}.ffn_up_exps.weight`, `token_embd_norm.weight`).
- `std_perm_gguf` packs Q4_0/Q4_1 exactly and re-quantizes Q8_0/Q4_K/Q6_K to q4. It honours
  `chunk0` (row offset = 32 * chunk0 / (in_dim/256)) since 2026-10-13.
- A Q6_K tied embedding as a q4 head costs logits corr ~0.995 -- enough to flip argmax.
  Use the q8 head: `lmhead_q8_gguf` pack op + `lm_head_q8.py` with `LMHEAD_SCALES_F32=1`.
  That kernel narrows scales to bf16, so the scale is chosen in bf16 (rounded up).
- Small F32 weights (conv taps) need conversion to bf16; `transpose` does it for GGUF now.
- Per-layer KV arrays (`head_count_kv` on hybrids) break numeric metadata reads; ship a
  `config.json` in the model dir (it wins over GGUF metadata).
- f32-scale pool chunks are 6144 B per 8192 values (~6 bits/weight): a GGUF model takes
  ~1.3x its file size on device.
- Model dir layout: `model.gguf` (symlink is fine) + `config.json` + `tokenizer*.json`.
  `chat.py` needs a prompt branch per chat format (LFM2: `<|startoftext|>` + ChatML + `<think>`).

## 5. Profiling decode

- `OFLM_OPEN_STEP_TRACE=1` on the CLI prints per-kernel min/mean per dispatch, calls per
  step, and host time (route, embed). `--bench-kernel NAME:REPS:LAYER` times one kernel
  back to back (no context switch).
- Compare the two. In-step mean >> standalone means **hardware-context switches**: ~0.6-0.8 ms
  each, paid whenever consecutive dispatches use different xclbins. LFM2-MoE paid 76 a token
  (half the step).
- Fix: one image, several instruction streams (OPEN-DECODE-ONE-CONTEXT; `layer_x/ux.py`,
  `lfm2moe/lm.py`). Rules: the main cores run one program per layer whose differing counts
  come from RTP words (`Buffer(..., use_write_rtp=True)`, written at the head of the stream,
  read only AFTER the core's first acquire); helpers fed by one stream simply block through
  the others; every part must emit the same image -- verify with
  `export_qwen36_kernels.xclbin_equivalent` (only build stamps may differ); point all kernels
  at one context (`contexts["layer"]`). LFM2-MoE 24B: 106 -> 49 ms/token.
- Host work shows up too: a double-precision host GEMV was 4 ms/step; 16 f32 partial sums
  vectorize without -ffast-math.

## 6. Hygiene

- Generated kernel TUs (`designs/<x>/*.cc` written by gen_kernels.py) must be gitignored AND
  listed in `specs/open-engine/tests/test_generated_tus_untracked.py`.
- Never commit xclbins. Write a family skill (`open-<family>-kernels`) with the build and
  verify ladder, and add it to AGENTS.md.
