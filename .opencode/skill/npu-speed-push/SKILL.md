---
name: npu-speed-push
description: Method for making a model that already runs on the open XDNA2 engine meet throughput targets (prefill tok/s, decode tok/s) on the Strix while other jobs (a GPU trainer, CPU jobs) share the machine -- the constraint tracker, theory-of-constraints loop, memory and thermal safety beside a GPU job, measurement that survives a 2x noisy machine, the bit-identical rule, how to find what bounds a kernel, and the wins and dead ends from the Qwen3.6-35B-A3B push. Use when starting or continuing a speed push, when an optimization "did nothing", when timings disagree run to run, or before running long benches beside someone else's job.
---

# Speed push on the NPU (method)

From the Qwen3.6-35B-A3B push (2026-10-20/21, branch `feat/qwen36-a3b`, tracker
`TODO-qwen36-a3b.md`). Targets were prefill >= 1000 tok/s and decode >= 30 tok/s at 1K-128K
with the GPU busy. Port a model first with `npu-model-porting`; this skill is what comes after.

## 1. Safety before speed (a shared machine)

The NPU, CPU and GPU share one DDR and one power budget. Another person's job runs beside you.
- **Memory.** `gpu-oomd` kills the largest GPU job when MemAvailable < 8 GB, or < 14 GB when swap
  is full (it is always full). A 128K bench killed the user's trainer twice. Run every test
  under a wrapper that:
  - starts only when MemAvailable >= cap + 20 GB;
  - runs the command under `systemd-run --user --scope -p MemoryMax=<cap>G -p MemorySwapMax=0`;
  - stops ITS OWN PID if MemAvailable < 20 GB;
  - logs GPU-busy samples (`/sys/class/drm/card*/device/gpu_busy_percent`) and the other job's
    guard pauses.
  Know your peak first (`OFLM_OPEN_MEM_LOG=1`): the A3B engine is 24 GB after load, 28.8 GB at
  4K, ~40 GB at 30K.
- **Processes.** Kill only exact PIDs you started; never pattern kills. Do not touch the
  other job, its guard or its tmpfs files. A shared build directory that another process
  rebuilds is a hazard: build kernel sets by copying a known set and merging the manifest.
- **NPU hangs.** A TDR or ERT state 8 can wedge the array; reset by running a known-good tiny
  design on the harness (`run_kernel <small>/run.cfg`) before the next test.

## 2. The tracker

Write the targets and a constraint table into the TODO before any code:
`ID | area | item | kind | measured now | how to measure | expected gain | status`.
- Kinds: waste, slow code, placement, flow, memory, accuracy, testing.
- Every row gets a measurement method that works under load (section 4).
- Status ends as done (commit), measured: no gain (keep it opt-in, say why), or open.
  Dead ends stay in the table so nobody repeats them.
- Answer the user's design questions before you write code; record their decisions.

## 3. The loop (theory of constraints)

1. Find the ONE resource that limits the wall time. Prefill: the pipe trace
   (`OFLM_OPEN_PIPE_TRACE=<csv>`, `utilities/pipe_trace.py`) shows NPU busy / idle, each host
   thread's busy time by op, and what the host waited on. If the NPU idles while a host thread
   is 100 % busy, the host is the constraint, and NPU work does not change the wall.
2. Improve only that resource. Count its work in units that add up (host thread-seconds, NPU
   seconds, bytes, dispatches, switches).
3. Re-measure. The constraint moves: on A3B at 30K it went host -> NPU (after two host threads)
   -> host (beside the GPU job) -> both near the limit. With the GPU idle the NPU was the limit;
   with it busy, the host was. Check both states.
4. Subordinate the rest: a change that helps a non-constraint is worth it only when it is
   free (bit-identical and simple), or when it lets the constraint's work move (dnc moves 13
   host thread-s to the NPU, but only pays once NPU switches are cheap).

## 4. Measuring on a noisy machine

Whole runs vary 2x with the GPU job's phases (compute vs. eval/save). So:
- Compare inside one run where you can: A/B by layer parity (`OFLM_OPEN_DEC_NPU_AB`,
  `OFLM_OPEN_MBV_AB`), kernel time ratios in the pipe trace.
- Prefer counts that do not depend on load: expert slots, dispatches, context switches or
  configuration runs, MTP acceptance per position.
- For whole-run A/B: alternate A B A B, start each run after the GPU is >= 90 % busy for 5 s,
  and print the GPU-busy fraction beside every number. A run with an idle GPU is a different
  experiment; label it, do not average it in.
- Microbenchmark host kernels alone (same flags as the engine, `-O3`): GFLOP/s a thread
  against the core's peak. 16-45 % of peak meant spills or no unroll (GCC kept a 6x4 zmm tile
  in memory until `#pragma GCC unroll`).
- The engine's in-situ time is larger than the bench when the cores are shared: the router
  went 2.97 -> 1.72 ms in the bench but 5.0 -> 4.2 s at 30K. Report both.
- Beside the GPU job the CPU reads DDR at ~12 GB/s median (bursts to 110 between steps); the
  NPU DMA keeps ~35 GB/s. CPU work that streams memory slows 2x; fewer BYTES a token is the
  lever there, not fewer instructions.

## 5. Finding what bounds a kernel

- Build variants with parts removed (null math, DMA only: `MBI_NULL_*`, `EB_NULL_MM`) and time
  them on the harness. The mbi expert kernel ran at its DMA-only floor under load, so the
  experts were bytes and rounds, not MACs.
- Time a design alone on the harness, then inside the engine. A3B's dnc took 2.7 ms on the
  harness and 6 ms in the engine: the difference was two context switches a call.
- Probe the hardware cost model directly before redesigning: context switch cost vs. design
  size (`full-elf-context`, section "Classic flow"), DDR bandwidth with and without load,
  single-design vs. alternating runs.

## 6. Accuracy: the bit-identical rule

The MoE path amplifies 1-ulp changes: a different add order for the shared expert moved
600-token logits by KL 7.8e-3, as much as the whole quantization error (1.06e-2).
- Make speed changes bit-identical where possible: keep each output's FMA chain in the same
  order (tile over tokens and outputs, not over the reduction), keep tie rules (top-k: value
  down, index up), convert at the same points.
- Check: dump logits (`--prefill-logits --dump-logits`) and `cmp` the bytes against the
  previous build (expect KL 0 and identical bytes); greedy decode tokens on a chat prompt.
- A change that is not bit-identical is judged against the reference (`acc/tok/y`), not the
  previous build. An accuracy trade (SHARED_MBI: -6 % time, KL 1.06e-2 -> 1.18e-2) is the
  user's decision: ship it opt-in and ask.

## 7. What paid on A3B (30K prompt, GPU busy unless noted)

| change | result |
|---|---|
| two host threads with OMP teams of 8 (`OFLM_OPEN_HOST_THREADS`) | 63 -> 45 s |
| AVX-512 quant / gate, bf16 rows read directly | 45 -> 37 s |
| expert passes sized to the largest stream that fits | -14 % slots, -2.2 s NPU |
| pass buffers sized by need | -1.9 GB |
| DeltaNet rule writes the OUT GEMM's bfp16 A blocks (`A2_DIRECT`) | MID -2 to -4 thread-s |
| x with block sums once, dense int16 mmul for the qsum term (`MBI_QSC`) | x -44 %, host -2.6 thread-s |
| router as 6x64 register tiles over L2 slices, one-pass top-k | router 5.0 -> 4.2 s |
| one hardware context, CDO configuration runs (`OFLM_OPEN_ONE_CTX`) | NPU -2.2 to -2.8 s, wall -1.2 to -1.5 s; switch 5.5 -> ~1 ms |
| decode: half a pass's experts on the NPU; MTP k <= 4, PCUM 0.4 | -15 % MoE a layer; 3.9 tokens a pass |

## 8. What did not pay (and why) -- do not repeat without a new reason

- Context-sticky ordering, groups of 8192: fewer switches, but the host was the limit.
- RES early (scatter finished passes first): the host waited on RES < 1 s.
- Decode GEMV on the NPU (mbv): the CPU reads those weights at ~29 GB/s in decode; the split
  added pack and dispatch time.
- `MBI_P` (columns share a weight stream): the per-core stream became the bound.
- 3 or 4 host threads: slower than 2 (memory and cores shared with the GPU job).
- NPU router: needs a context, and bfp16 rounding moves near-tied routing.
- The same A2-direct fusion in the attention GATE step: slower.
- Spatial sharing of the array (two designs on column halves): the driver sets
  `AIE2_TEMPORAL_ONLY` for this NPU.

## 9. Reporting

Lead with the result against the target. Give each number with its load state (GPU busy
samples), the comparison it came from, and whether outputs are bit-identical. Commit each
bit-identical step on its own with the measured gain in the message; push only to the remote
the user named.
