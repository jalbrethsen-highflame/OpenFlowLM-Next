---
name: full-elf-context
description: Run several mlir-aie designs (kernel sets) in ONE XDNA2 hardware context -- either by assembling their built streams into a full ELF (aiecc --get-full-elf --expand-load-pdis, aiebu-asm), switching between them with register-write reconfiguration (0.3-0.7 ms instead of ~2.1 ms per xclbin context switch). Use when a pipeline alternates between several xclbins and the switches cost time, when building or debugging open_kernels/compose_elf.py or the diffusion engine's ELFs, when a full-ELF run hangs with ERT_CMD_STATE_TIMEOUT (especially while another process uses the NPU), when XRT kernel creation is slow, or when a baked-in runtime value (an RTP write) has to vary per run. Also the classic-xclbin variant (each design's CDO replayed as an instruction stream in one shared context, OFLM_OPEN_ONE_CTX) for engines that patch their instruction streams between calls.
---

# One hardware context for several designs

Worked example: FLUX.2 klein (six kernel sets, 822 switches per image). The switch cost
went from ~2.1 ms to 0.3-0.7 ms: 5.2 → 3.7 s at 512², 13.5 → 12.0 s at 1024², pixels
identical.
- Code: `open_kernels/compose_elf.py` (assembler), `src/open_diffusion/engine.cpp`
  (replay).
- Probes and numbers: `utilities/reconfig-probe/` (README), and
  `utilities/dit-chain/switch_probe.py`.
- Plan: `specs/open-diffusion/archive/one-context.md`.

## What a switch costs (measure before building anything)

`switch_probe.py` times each op three ways: queued, waited, and alternating with an op of
another set.
- Host round trip: 0.005-0.16 ms.
- xclbin context switch: 2.0-2.4 ms, whatever the op or the set.

That is configuration loading. The options inside one full ELF:

| mode (aiecc flag) | cost per switch | notes |
|---|---:|---|
| `load_pdi` (`--get-full-elf`) | ~2.0 ms | no gain; the firmware skips a repeat of the loaded PDI |
| register writes (`--expand-load-pdis`) | 0.30 / 0.51 / 0.70 ms (gemm / ew / fa) | scales with the set's config size; every configure pays it |
| control packets (`--load-pdi-to-ctrl-pkt`) | — | won't build if the design uses every shim DMA channel |

## The shape that works

- **Assemble, don't compile one module.** aiecc's `AIEAssignBufferAddresses` walks the
  whole device once per tile (`AIEAssignBuffers.cpp:155/572`). Its per-core split clones
  the whole module once per core (`Actions.h:159`). So one module with every stream costs
  cores × sequences: 67 streams took 671 s and ~10 GB.
  - Instead run one configuration build: `main` with a configure-only sequence per set,
    plus each device with ONE stream. That gives every PDI and cfg code under one
    numbering.
  - Then one small build per stream, in parallel. A stream's control code and its set's
    PDI are byte-identical to the monolithic build's.
  - Write `full_elf_config.json` yourself and run `aiebu-asm -t aie2_config -j <json> -o
    <elf>`; that is all aiecc's last step does. 138 streams: 7 min cold, seconds cached.
- **Kernels:** a device's own sequence is kernel `<device>:<sequence>`, and it does NOT
  configure. Run it with the set not loaded and it hangs until timeout. A configure-only
  kernel is `aie.runtime_sequence @cfg_S() { aiex.configure @S { } }` in `main`.
- **Issue a configure only on a set change.** Reconfiguring a set on top of itself
  (ew then ew) corrupted outputs about 5% of the time.
- **Host side** (C++, XRT 2.21):
  - `xrt::elf(path)` → `xrt::hw_context(device, elf, {{"priority", 0x180}}, shared)`;
  - `xrt::ext::kernel(ctx, "S:stream")`;
  - `xrt::ext::bo(device, size)`, which is zero-copy (`group_id` on an ext kernel
    crashes);
  - pyxrt has `elf`, `ext.kernel`, `ext.bo`, `runlist` and `run.wait2`, but not
    `get_ctrl_scratchpad_bo`.

## Surviving another process on the NPU (the part that bites)

With register writes, the configuration is one the firmware doesn't know. If another
context takes the NPU mid-image, ours comes back without it: `ERT_CMD_STATE_TIMEOUT`,
usually taking the other process down too. A pure `load_pdi` full-ELF context failed the
same way. The xclbin path survives, so this is specific to multi-PDI full-ELF contexts.
What made it survive (`utilities/reconfig-probe/contention_trial.ps1`: 8 of 8 hung at
normal priority across four variants; 6 of 6 passed with all three below):
1. **Each set's configure and its ops as one `xrt::runlist`** (a stretch).
2. **Every configure starts from a real reset.** A cfg's first op is a `load_pdi` of an
   empty device, and the firmware skips it when it names the PDI already loaded. So build
   two variants of each cfg that name the two empty devices (`compose_elf.cfg_variants`,
   a u16 at byte 2 of that op) and alternate them.
3. **QoS priority 0x180** (amdxdna "high"; normal is 0x200). With it, the other context
   runs only between our stretches; without it, both hang.

Not tested: a contender that is itself high or realtime priority.

## Kernel creation cost

Every `xrt::ext::kernel` walks all of the ELF's control code, ~2 ms per MB per kernel.
Threads don't help; XRT serializes it. So:
- one ELF per resolution, holding only what its schedule runs;
- split a per-run variant into a small head plus a shared tail, never whole copies
  (below);
- create the kernels on a thread while the weights load. It fully overlaps (1024²:
  load 5.8 s, the same as six xclbins).

## A value that varies per run (an RTP write)

Instruction words can't be patched inside an ELF. For te_attn's valid_len:
- diff a probe build to find the words: they are the value field (byte 16) of one
  `write32` per core;
- all of them come before the stream's first DMA op, and none carries an argument patch;
- so split the TXN: a head of just those writes, one per value (784 B each, 512 values),
  plus the rest as a shared tail, run head then tail in the same stretch.

The TXN format (`compose_elf.txn_ops`):
- a 16-byte header: 6 geometry bytes, 2 spare, u32 op count, u32 bytes;
- ops back to back: write32 24 B, maskwrite/maskpoll 28 B, load_pdi 16 B;
- blockwrite: its size at +12; custom ops (≥128): their size at +4.
- The parse must end exactly at the header's byte count.

## Classic flow: one context, CDO configuration runs (patchable streams)

A full ELF's control code cannot be patched, but the open LLM engine rewrites instruction
words between calls (expert offsets, KV positions). The A3B push (2026-10-21) kept the classic
`xrt::kernel("MLIR_AIE")` + instruction-BO flow and moved the reconfiguration into it.
Code: `Core::context`, `parse_cdo`, `cfg_prefix`, `Core::start_run` in
`src/open_qwen36/core.cpp`; env `OFLM_OPEN_ONE_CTX=1` (2 = wait for each configuration run).

**The cost it removes.** On this Strix (npu5, XRT 2.25) the firmware's PDI load costs ~0.17 ms
a core: ~0.17 ms for a 1-core design, ~5.2-5.6 ms for a full 32-core array, whatever the PDI
size (138 KB and 323 KB cost the same). It is the same in one hw context with two kernels and
in a full ELF with plain `load_pdi`. The firmware skips a load of the PDI already loaded.

**How it works.**
1. All designs run in ONE hw context (the first xclbin's) through `MLIR_AIE` kernels on that
   context. IRON kernels share one signature (opcode, instr, ninstr, bo0-bo4).
2. For each context name of the set, at load: parse the CDO blocks inside its xclbin (find
   `"CDO\0"` with header word count 4; commands 0x102 mask write, 0x103 write, 0x105 DMA write,
   0x107 / 0x108 64-bit forms, 0x111 nop) and emit them as TXN ops: maskwrite 28 B, write32
   24 B, blockwrite (<= 1024 words each). The addresses are the same in both formats.
3. Put a reset prefix in front, built from the UNION of what any design of the set writes:
   - disable every core any design uses (`0x32000`, mask 1);
   - reset (bit 1 set, then clear) every core DMA channel any design uses (`0x1DE00-0x1DE1C`),
     and every memory tile's 12 channels (`0xA0600 + 8i`, `0xA0630 + 8i`, all 8 columns);
   - write 0 to every stream-switch port register any design sets (core / shim
     `0x3F000-0x3F3FF`, memory tile `0xB0000-0xB03FF`).
4. In `start_run`, when the next kernel's context differs from the one configured, start the
   configuration run first (no wait: one context runs its commands in order), then the real
   run. Wait on the configuration run before its `xrt::run` object dies, or XRT aborts with
   "xrt::run destructed while command is still in progress".

**Why each part is needed (each one failed without it).**
- IRON cores loop forever and their memory-tile channels loop over their BD chains, so a
  design is still active after its run. Without the memory-tile reset, the first run on a
  fresh context was right and every later one was garbage (rel. error ~1).
- The CDO writes only the registers its own design uses. Without the union prefix, a stale
  route or core of the previous design hung the next one (ERT state 8 on `mb_s128`).
- aiecc's `--expand-load-pdis` output starts the load with an op 8 (`0x00010008`, 16 B). In the
  classic flow that op hangs the run (state 8 after 4 s); drop it. The CDO path never emits it.
- Do not reconfigure a design on top of itself; issue the configuration only on a change.

**Numbers.** Configuration run 0.7-1.3 ms (dnc 0.71, mbi 0.96 from MLIR; ~0.9-1.3 from CDO
with the prefix); the design's own stream then runs at its steady speed. A3B 30K prompt with
the GPU busy: 712-722 configuration runs, NPU busy -2.2 to -2.8 s, wall -1.2 to -1.5 s;
600-token logits byte-identical. Instruction memory: 11 configurations, 3.4 MB.

**Probe it first.** A private harness copy with `kernelx NAME CTX insts@KNAME`, `elfctx`
and `kernelf` commands times: steady runs, runs right after a switch, and outputs dumped after
an A B A B B A sequence compared byte for byte with the classic two-context run.
