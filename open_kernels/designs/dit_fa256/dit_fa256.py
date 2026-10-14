r"""dit_fa256: causal GQA flash attention at head dim 256 (Qwen3.5 / 3.8 full-attention layers),
a port of designs/dit_fa (head dim 128).

    O[t, h*256:(h+1)*256] = softmax(Q_h K_kv(h)^T / 16 + causal mask) V_kv(h)

Query row t of a call sits at absolute position q_off + t and sees keys 0 .. q_off + t.
The engine calls it per 1024-token prompt chunk with q_off = the chunk's start and
lk = q_off + 1024 keys, so the causal triangle is exploited at chunk granularity on the
DMA side; inside a call, key chunks past every row of a core are skipped in the core (the
kernels return at once), and only the diagonal chunks are masked element by element.

What changed from dit_fa (everything else -- topology, Q capture on the K/V broadcasts,
lazy-rescale online softmax, hardware exp2 -- is dit_fa's):

  * DFULL = 256 split into DC = 32 chunks (NDCH = 8) instead of 128 into 2 x 64. TQ stays
    32 rows per core (QROWS_PASS 512), so every kernel's row loops are unchanged; the
    [64, 256] L2 blocks leave the memtile as eight [64, 32] mmul-ordered pieces.
  * RTP_QOFF (runtime): the query offset for the causal mask and the chunk skip.
  * Q / K / V / O strides and columns per stream: Q [L, 6144], K and V read in place out
    of the engine's KV cache rows [k 1024 | v 1024] (k_ld = v_ld = 2048, v_col = 1024),
    O [L, 6144].

L1 per core (bytes): A and B fifos 2 x 2 x 4096 (one [64, 32] bf16 piece each), the
output slab 32 x 256 bf16 = 16384, q_saved 8 x 32 x 32 bf16 = 16384, G 32 x 64 bf16 = 4096,
running max 64 + sum 1024, .bss ~1.7 KB, stack 4 KB: ~59 KB of 64.

    DF_L=1024 DF_LK=8192 DF_QOFF=7168 python build_design.py designs/dit_fa256/dit_fa256.py <out>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    Buffer, CompileTime, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime,
    TaskGroup, Worker, WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = Path(__file__).resolve().parent
KERNEL_SRC = str(HERE / "fa256.cc")

TQ, LKP, DC, DFULL = 32, 64, 32, 256
NDCH = DFULL // DC
N_GROUPS, GROUP_COLS, N_ROWS = 2, 4, 4
CORES = GROUP_COLS * N_ROWS            # per group
QROWS_PASS = CORES * TQ                # 512
Q_ON_A = CORES // 2                    # Q tiles on fifo A; the rest ride B
MAX_INFLIGHT = 2                       # passes queued ahead
STACK = 0x1000                         # softmax_step with FA_EXP_FIX spills a 3 KB frame

RTP_HEADS, RTP_PASSES, RTP_CHUNKS, RTP_VALID, RTP_CAUSAL, RTP_ROW0, RTP_QOFF = range(7)
RTP_LEN = 8
EXP_FIX = int(os.environ.get("DF_EXP_FIX", 0))   # fa_dit.cc FA_EXP_FIX (a build-wide choice)
TAU = int(os.environ.get("DF_TAU", 32))          # fa_dit.cc FA_TAU, the lazy-rescale threshold
_EXPT = os.environ.get("DF_EXPT", "").split(",")  # ablation builds only


def check_shape(L: int, heads: int, kv_heads: int, lk: int | None = None) -> str | None:
    lk = L if lk is None else lk
    if L % QROWS_PASS or lk % LKP or heads % N_GROUPS or heads % kv_heads:
        return (f"dit_fa needs tokens % {QROWS_PASS}, keys % {LKP}, heads % {N_GROUPS}, "
                f"heads % kv_heads (got L={L} lk={lk} heads={heads} kv_heads={kv_heads})")
    return None


def _block_dims(rows, cols, blk=8):
    """Row-major [rows, cols] -> the 8x8-block column-major order the mmul reads
    ([col block][row block][8][8]). whisper_fa's _block_dims (attempt 21's order)."""
    return [(cols // blk, blk), (rows // blk, blk * cols), (blk, cols), (blk, 1)]


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit_fa256(
    Q: In, K: In, V: In, O: Out, *,
    L: CompileTime[int], heads: CompileTime[int], kv_heads: CompileTime[int] = 0,
    lk: CompileTime[int] = 0,
    q_ld: CompileTime[int] = 0, k_ld: CompileTime[int] = 0, v_ld: CompileTime[int] = 0,
    o_ld: CompileTime[int] = 0,
    q_col: CompileTime[int] = 0, k_col: CompileTime[int] = 0, v_col: CompileTime[int] = 0,
    o_col: CompileTime[int] = 0, o_interleave: CompileTime[int] = 0,
    causal: CompileTime[int] = 1, valid_len: CompileTime[int] = 0, q_off: CompileTime[int] = 0,
):
    kv_heads = kv_heads or heads
    lk = lk or L
    q_ld = q_ld or heads * DFULL
    o_ld = o_ld or heads * DFULL
    k_ld = k_ld or kv_heads * DFULL
    v_ld = v_ld or kv_heads * DFULL
    valid_len = valid_len or lk
    why = check_shape(L, heads, kv_heads, lk)
    assert why is None, why

    blk_ty = np.ndarray[(LKP, DFULL), np.dtype[bfloat16]]     # L2: 64 keys (or 2 cores' Q rows)
    elem_ty = np.ndarray[(LKP, DC), np.dtype[bfloat16]]       # L1: one head-dim chunk of it
    q_ty = np.ndarray[(NDCH * TQ * DC,), np.dtype[bfloat16]]
    g_ty = np.ndarray[(TQ * LKP,), np.dtype[bfloat16]]
    o_l1_ty = np.ndarray[(TQ * DFULL,), np.dtype[bfloat16]]
    o_l2_ty = np.ndarray[(N_ROWS * TQ * DFULL,), np.dtype[bfloat16]]
    up_ty = np.ndarray[(TQ,), np.dtype[bfloat16]]
    sp_ty = np.ndarray[(TQ * 8,), np.dtype[np.float32]]      # l, replicated per lane
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DFA_TQ={TQ}", f"-DFA_LKP={LKP}", f"-DFA_DC={DC}", f"-DFA_DFULL={DFULL}",
             f"-DFA_QROWS_PASS={QROWS_PASS}", f"-DFA_EXP_FIX={EXP_FIX}", f"-DFA_TAU={TAU}",
             "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
             "-Wno-deprecated-declarations"]
    # One ExternalFunction compiles the file; every call goes through a plain Kernel
    # sibling (whisper_fa's note: ExternalFunction's arg check rejects acquired bf16
    # elements).
    _ef = ExternalFunction("zero_g", source_file=KERNEL_SRC, arg_types=[g_ty],
                           compile_flags=flags)
    obj = _ef.object_file_name
    k_capture = Kernel("capture_q", obj, [elem_ty, q_ty, np.int32, np.int32])
    k_init = Kernel("pass_init", obj, [o_l1_ty, up_ty, sp_ty])
    # every per-chunk kernel takes (rtp, pass, chunk) and returns at once on a chunk that is
    # past the causal diagonal for all of this core's rows
    k_zero_g = Kernel("zero_g", obj, [g_ty, rtp_ty, np.int32, np.int32])
    k_qk = Kernel("qk", obj, [q_ty, elem_ty, g_ty, np.int32, rtp_ty, np.int32, np.int32])
    k_pv = Kernel("pv", obj, [g_ty, elem_ty, o_l1_ty, np.int32, rtp_ty, np.int32, np.int32])
    k_mask = Kernel("apply_mask", obj, [g_ty, rtp_ty, np.int32, np.int32])
    k_step = Kernel("softmax_step", obj, [g_ty, up_ty, sp_ty, o_l1_ty, rtp_ty, np.int32, np.int32])
    k_fin = Kernel("finalize", obj, [sp_ty, o_l1_ty])
    kernels = [k_capture, k_init, k_zero_g, k_qk, k_pv, k_mask, k_step, k_fin]

    def core_tile(g, i):
        return Tile(g * GROUP_COLS + i // N_ROWS, 2 + i % N_ROWS)

    # A (Q tiles 0-7, then K) and B (Q tiles 8-15, then V): L3 -> memtile -> the
    # group's 16 cores, block-transposed on the way out of the memtile.
    A_l3l2, B_l3l2, A_l1, B_l1 = [], [], [], []
    for g in range(N_GROUPS):
        fa = ObjectFifo(blk_ty, name=f"A_L3L2_{g}", depth=2)
        fb = ObjectFifo(blk_ty, name=f"B_L3L2_{g}", depth=2)
        A_l3l2.append(fa)
        B_l3l2.append(fb)
        # _block_dims over the whole [64, 128] block is [d chunk][d block][key block]
        # [8][8]: the two chunks leave back to back, each already in mmul order.
        a2 = fa.cons().forward(tile=Tile(g * GROUP_COLS, 1), obj_type=elem_ty, depth=2,
                               name=f"A_L2L1_{g}", dims_to_stream=_block_dims(LKP, DFULL))
        b2 = fb.cons().forward(tile=Tile(g * GROUP_COLS + 1, 1), obj_type=elem_ty, depth=2,
                               name=f"B_L2L1_{g}", dims_to_stream=_block_dims(LKP, DFULL))
        A_l1.append([a2.cons(depth=2) for _ in range(CORES)])
        B_l1.append([b2.cons(depth=2) for _ in range(CORES)])

    # O: a column's 4 cores (32 rows each, block layout) join at its memtile, which
    # streams each part out row-major (a join's output dims apply per part), so the
    # column's 128 rows leave in order.
    out_dims = [(TQ, 8), (DFULL // 8, TQ * 8), (8, 1)]
    O_l2l3, O_l1 = [], {}
    for c in range(N_GROUPS * GROUP_COLS):
        f = ObjectFifo(o_l2_ty, name=f"O_L2L3_{c}", depth=2, dims_to_stream=out_dims)
        O_l2l3.append(f)
        parts = f.prod().join([r * TQ * DFULL for r in range(N_ROWS)], tile=Tile(c, 1),
                              obj_types=[o_l1_ty] * N_ROWS,
                              names=[f"O_L1L2_{c}_{r}" for r in range(N_ROWS)],
                              depths=[1] * N_ROWS)
        for r in range(N_ROWS):
            O_l1[(c, r)] = parts[r]

    rtp = {}
    barriers = {}

    def make_core_fn(i):
        def core_fn(a_in, b_in, o_out, kern, q_saved, g_buf, up, sp, my_rtp, barrier):
            (capture, init, zero_g, qk, pv, mask, step, fin) = kern
            barrier.wait_for_value(1)
            n_heads = my_rtp[RTP_HEADS]
            n_pass = my_rtp[RTP_PASSES]
            n_chunks = my_rtp[RTP_CHUNKS]
            for _ in range_(n_heads):
                for p in range_(n_pass):
                    o = o_out.acquire(1)
                    # Q: a block is 2 cores' rows; its chunks arrive d-chunk by d-chunk.
                    for pair in range(CORES // 2):
                        fifo = a_in if pair < Q_ON_A // 2 else b_in
                        for d in range(NDCH):
                            e = fifo.acquire(1)
                            if pair == i // 2:
                                capture(e, q_saved, d, i % 2)
                            fifo.release(1)
                    init(o, up, sp)
                    for c in range_(n_chunks):
                        zero_g(g_buf, my_rtp, p, c)
                        for d in range_(NDCH):
                            k = a_in.acquire(1)
                            if "noqk" not in _EXPT:
                                qk(q_saved, k, g_buf, d, my_rtp, p, c)
                            a_in.release(1)
                        mask(g_buf, my_rtp, p, c)
                        if "nosm" not in _EXPT:
                            step(g_buf, up, sp, o, my_rtp, p, c)
                        for d in range_(NDCH):
                            v = b_in.acquire(1)
                            if "nopv" not in _EXPT:
                                pv(g_buf, v, o, d, my_rtp, p, c)
                            b_in.release(1)
                    fin(sp, o)
                    o_out.release(1)
            barrier.release_with_value(1)
        return core_fn

    workers = []
    for g in range(N_GROUPS):
        for i in range(CORES):
            t = core_tile(g, i)
            c, r = t.col, t.row - 2
            rtp[(g, i)] = Buffer(rtp_ty, name=f"rtp_{g}_{i}",
                                 initial_value=np.zeros(RTP_LEN, dtype=np.int32),
                                 use_write_rtp=True)
            barriers[(g, i)] = WorkerRuntimeBarrier()
            args = [A_l1[g][i], B_l1[g][i], O_l1[(c, r)].prod(), kernels,
                    Buffer(q_ty, name=f"q_saved_{g}_{i}"), Buffer(g_ty, name=f"g_{g}_{i}"),
                    Buffer(up_ty, name=f"up_{g}_{i}"), Buffer(sp_ty, name=f"sp_{g}_{i}"),
                    rtp[(g, i)], barriers[(g, i)]]
            workers.append(Worker(make_core_fn(i), args, tile=t, stack_size=STACK))

    heads_local = heads // N_GROUPS
    n_pass = L // QROWS_PASS
    n_chunks = lk // LKP
    gqa = heads // kv_heads

    Q_ty = np.ndarray[(L * q_ld,), np.dtype[bfloat16]]
    K_ty = np.ndarray[(lk * k_ld,), np.dtype[bfloat16]]
    V_ty = np.ndarray[(lk * v_ld,), np.dtype[bfloat16]]
    O_ty = np.ndarray[(L * o_ld,), np.dtype[bfloat16]]

    def row_split(ld):
        """Rows per block of a K/V fill: a divisor of LKP whose stride fits a BD (2^21 bf16)."""
        rs = LKP
        while rs * ld > (1 << 21):
            rs //= 2
        return rs

    rs_k, rs_v = row_split(k_ld), row_split(v_ld)

    def sequence(q, k, v, o, a_prods, b_prods, o_conses):
        for (g, i), buf in rtp.items():
            buf[RTP_HEADS] = heads_local
            buf[RTP_PASSES] = n_pass
            buf[RTP_CHUNKS] = n_chunks
            buf[RTP_VALID] = valid_len
            buf[RTP_CAUSAL] = causal
            buf[RTP_ROW0] = i * TQ
            buf[RTP_QOFF] = q_off
        for b in barriers.values():
            b.set(1)

        inflight: list[TaskGroup] = []
        for hl in range(heads_local):
            for p in range(n_pass):
                tg = TaskGroup()
                for g in range(N_GROUPS):
                    h = g * heads_local + hl
                    kvh = h // gqa
                    qbase = q_col + h * DFULL + p * QROWS_PASS * q_ld
                    for prods, first in ((a_prods, 0), (b_prods, Q_ON_A)):
                        prods[g].fill(q, offset=qbase + first * TQ * q_ld,
                                      sizes=[Q_ON_A * TQ, DFULL], strides=[q_ld, 1], group=tg)
                    # the keys as [lk/rs, rs, 128] row blocks: a BD stride is at most
                    # 2^20 words, so rs rows of a wide fused buffer must stay under it
                    # (the fifo only sees rows in order; the grouping is free)
                    a_prods[g].fill(k, offset=k_col + kvh * DFULL,
                                    sizes=[lk // rs_k, rs_k, DFULL],
                                    strides=[rs_k * k_ld, k_ld, 1], group=tg)
                    b_prods[g].fill(v, offset=v_col + kvh * DFULL,
                                    sizes=[lk // rs_v, rs_v, DFULL],
                                    strides=[rs_v * v_ld, v_ld, 1], group=tg)
                    for cc in range(GROUP_COLS):
                        col = g * GROUP_COLS + cc
                        row0 = p * QROWS_PASS + cc * N_ROWS * TQ
                        if o_interleave:   # head h's dims d at o_col + 256h + 128(d//64) + d%64
                            o_conses[col].drain(o, offset=o_col + 2 * h * DFULL + row0 * o_ld,
                                                sizes=[N_ROWS * TQ, 2, DFULL // 2],
                                                strides=[o_ld, DFULL, 1], wait=True, group=tg)
                            continue
                        o_conses[col].drain(o, offset=o_col + h * DFULL + row0 * o_ld,
                                            sizes=[N_ROWS * TQ, DFULL], strides=[o_ld, 1],
                                            wait=True, group=tg)
                inflight.append(tg)
                if len(inflight) == MAX_INFLIGHT:
                    inflight.pop(0).finish()
        for tg in inflight:
            tg.finish()

    a_prods = [A_l3l2[g].prod(tile=Tile(g * GROUP_COLS, 0)) for g in range(N_GROUPS)]
    b_prods = [B_l3l2[g].prod(tile=Tile(g * GROUP_COLS + 1, 0)) for g in range(N_GROUPS)]
    o_conses = [O_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_GROUPS * GROUP_COLS)]
    rt = Runtime(sequence, [Q_ty, K_ty, V_ty, O_ty, a_prods, b_prods, o_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


# build_design.py convention
DESIGN = dit_fa256
SPECIALIZE = dict(
    L=int(os.environ.get("DF_L", 1024)),
    heads=int(os.environ.get("DF_HEADS", 24)),
    kv_heads=int(os.environ.get("DF_KV_HEADS", 4)),
    lk=int(os.environ.get("DF_LK", 1024)),
    q_off=int(os.environ.get("DF_QOFF", 0)),
    # DF_CAUSAL=0: every query sees keys [0, valid_len) (d3's noncausal full attention); the
    # engine patches DF_VALID (a sentinel word) to the prompt's length in the stream
    causal=int(os.environ.get("DF_CAUSAL", 1)),
    valid_len=int(os.environ.get("DF_VALID", 0)),
    # the engine's layout: Q / O [L, heads * 256]; K and V in place in the KV cache, whose
    # rows are [k (kv_heads * 256) | v (kv_heads * 256)]
    q_ld=int(os.environ.get("DF_Q_LD", 0)),
    k_ld=int(os.environ.get("DF_KV_LD", 2048)),
    v_ld=int(os.environ.get("DF_KV_LD", 2048)),
    k_col=0,
    v_col=int(os.environ.get("DF_V_COL", 1024)),
    o_ld=int(os.environ.get("DF_O_LD", 0)),
)
