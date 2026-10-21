# Help a model out!

You can install or request toolchain components

Each time you build a new set of xclbins and libraries, document your process and create a new skill file for the next to build on.

Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.

The new format Q4NX models for this will be located on huggingface `Atomic-Germ/*-nB-OpenNPU2`, each will need a builder in q4nx-build as they gain xclbin support. Family xclbins will be distributed with the application, and a converted model ships with the rest. Each model links to a family xclbin, regardless of fine-tuning or derivitive models. The purpose of this is to allow quick additions of new but shape-identical models without updates needed, distributed through `oflm-add` to install and link properly.

As you encounter closed-source requirements, note them and replace with "not implemented" messaging before coding a replacement.

When investigating pre-compiled xclbins and libraries, save any useful tools you may create to the `utilities/` directory rather than leaving them in a temp folder.

The `oflm-test` tool at `utilities/oflm-test` is capable of running a full test suite for `--llm`, `--vision`, `--embed`, or `--tools`.

The `q4nx-build` tool at `utilities/q4nx-build` should always be in-sync with the expected formats of the models as each gains support, there should be no manual steps left. If unavoidable, the user should always recieve instruction for it.

For each successfully packed model, create or update a skill to ensure the next agent does not need to reproduce research for the next addition.

Note: Peano (llvm-aie) has not been added to PATH to avoid conflict with
      system clang/clang++. It can be found in:
      ./ironvenv/lib/python3.12/site-packages/llvm-aie/bin

Activate the ironvenv/bin/activate; use source utilities/mlir-aie/utils/env_setup.sh also *if needed*

<available_skills>
  <skill>
    <name>npu_offload_pipeline</name>
    <description>End-to-end workflow for offloading dense GEMM operations to AMD NPU2 via mlir-aie/iron. Use when: compiling NPU xclbins, integrating NPU backends into embedding/LLM engines, validating NPU vs CPU reference, debugging XRT dispatch issues, or extending to new model architectures.</description>
    <location>.opencode/skill/npu_offload_pipeline.md</location>
  </skill>
  <skill>
    <name>open-granite-kernels</name>
    <description>Build, verify and ship the open XDNA2 kernel sets (dx ln lm_head_q4) that run IBM Granite 4.2 3B on the dense recipe. Use when rebuilding those xclbins, adding another Granite size, debugging "no open kernels found" for granite:3b, or when a Granite container's attention_multiplier is refused at load.</description>
    <location>.opencode/skill/open-granite-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-ab</name>
    <description>Build and validate the separate open banked alpha/beta dispatch at H5120/H2560, including xn streaming and the fused glue DMA constraint.</description>
    <location>.opencode/skill/open-wide-deltanet-ab/SKILL.md</location>
  </skill>
  <skill>
    <name>open-wide-deltanet-chain</name>
    <description>Build and validate the synthetic 48-head AB, conv/record and recurrent-state chain for the WideDeltaNet A7 gate, including default32-head regression and known precision limits.</description>
    <location>.opencode/skill/open-wide-deltanet-chain/SKILL.md</location>
  </skill>
  <skill>
    <name>npu-model-porting</name>
    <description>Playbook for porting a NEW model architecture to the open XDNA2 engine (open_kernels recipes + IRON designs + src/open_qwen36) -- environment setup on a fresh Linux box, the reference-first workflow, slice testing, how the manifest/engine fit together, IRON design rules that hang or corrupt silently (drain ordering, shim/L1/program-memory budgets), GGUF-direct gotchas, profiling, and the one-context merge. Use before adding a family, when a new design hangs with ERT state 8, when NPU logits disagree with a reference, or when decode is slower than the kernels' own timings suggest.</description>
    <location>.opencode/skill/npu-model-porting/SKILL.md</location>
  </skill>
  <skill>
    <name>open-lfm2moe-kernels</name>
    <description>Build, verify and run the open XDNA2 kernels for LiquidAI LFM2-MoE (LFM2.5-24B-A2B, LFM2.5-8B-A1B; model_type lfm2_moe) from a Q4_0 GGUF -- the lfm2moe recipe, the merged one-context layer image (lm.py), the separate-image fallback (cx/dx *_MIX + ex.py), the host sigmoid router and the q8 head. Use when rebuilding these sets, adding another LFM2-MoE size, or debugging a wrong token / ERT timeout on this family.</description>
    <location>.opencode/skill/open-lfm2moe-kernels/SKILL.md</location>
  </skill>
  <skill>
    <name>npu-speed-push</name>
    <description>Method for making an NPU model meet throughput targets while other jobs share the machine: constraint tracker, theory-of-constraints loop, memory safety beside a GPU job, measurement under load, the bit-identical rule, A3B wins and dead ends.</description>
    <location>.opencode/skill/npu-speed-push/SKILL.md</location>
  </skill>
  <skill>
    <name>full-elf-context</name>
    <description>Several designs in one XDNA2 hardware context: full-ELF assembly, or classic xclbin flow with CDO configuration runs (OFLM_OPEN_ONE_CTX); switch cost 5.5 ms -> ~1 ms.</description>
    <location>.opencode/skill/full-elf-context/SKILL.md</location>
  </skill>
</available_skills>
