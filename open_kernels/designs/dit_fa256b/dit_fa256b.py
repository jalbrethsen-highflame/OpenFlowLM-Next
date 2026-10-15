r"""dit_fa256b: d3's noncausal GQA flash attention at head dim 256 with bfp16ebs8 Q, K, V and P
on the native block-float MAC (fa256b.cc). dit_fa256's topology (2 groups x 16 cores, 32 query
rows a core, K on fifo A and V on fifo B broadcast to a group, the lazy-rescale online softmax);
what changed:

  * Q, K and V arrive host-packed in the MAC's pop order (pack_fa.py), as 4608-byte pieces the
    memtile only copies: per core 9216 bytes of Q, per 64-key chunk 4 K pieces (16 keys x 256 d)
    and 4 V elements (64 keys x 64 d columns, transposed). 1.125 bytes a value, not 2.
  * Every key's whole head dim in one qk call, so the scores are written once (bf16, from fp32).
  * P goes to bfp16 once per chunk; l sums the same bfp16 P.
  * Key chunks wholly past valid_len (the pad up to lk) are skipped in the cores.

L1 per core (bytes): fifos 2 x 2 x 4608, O 16384, Q 9216, G 4096, P 2304, softmax state ~2.8 K,
stack 4 K: ~57 KB of 64.

    DF_L=1024 DF_LK=8192 DF_VALID=8000 python build_design.py designs/dit_fa256b/dit_fa256b.py <out>
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
KERNEL_SRC = str(HERE / "fa256b.cc")

TQ, LKP, DC, DFULL = 32, 64, 32, 256
NDCH = DFULL // DC
N_GROUPS, GROUP_COLS, N_ROWS = 2, 4, 4
CORES = GROUP_COLS * N_ROWS            # per group
QROWS_PASS = CORES * TQ                # 512
PIECE = 4608                           # bytes: one K piece (16 keys) / V element (64 d columns)
CHUNK = 4 * PIECE                      # a 64-key chunk of K (or of V)
Q_CORE = 2 * PIECE                     # one core's 32 rows of Q
MAX_INFLIGHT = 2                       # passes queued ahead
STACK = 0x1800                         # softmax_step: a 4.2 KB frame (fused exp + bfp16 P + rowsum)

RTP_HEADS, RTP_PASSES, RTP_CHUNKS, RTP_VALID, RTP_CAUSAL, RTP_ROW0, RTP_QOFF = range(7)
RTP_LEN = 8
EXP_FIX = int(os.environ.get("DF_EXP_FIX", 0))   # fa_dit.cc FA_EXP_FIX (a build-wide choice)
TAU = int(os.environ.get("DF_TAU", 32))          # fa_dit.cc FA_TAU, the lazy-rescale threshold
_EXPT = os.environ.get("DF_EXPT", "").split(",")  # ablation builds only
HS = int(os.environ.get("DF_HS", 1))              # hand-scheduled qk / pv (hsgen_fa.py)


def check_shape(L: int, heads: int, kv_heads: int, lk: int | None = None) -> str | None:
    lk = L if lk is None else lk
    if L % QROWS_PASS or lk % LKP or heads % N_GROUPS or heads % kv_heads:
        return (f"dit_fa256b needs tokens % {QROWS_PASS}, keys % {LKP}, heads % {N_GROUPS}, "
                f"heads % kv_heads (got L={L} lk={lk} heads={heads} kv_heads={kv_heads})")
    return None


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit_fa256b(
    Q: In, K: In, V: In, O: Out, *,
    L: CompileTime[int], heads: CompileTime[int], kv_heads: CompileTime[int] = 0,
    lk: CompileTime[int] = 0, o_ld: CompileTime[int] = 0, o_col: CompileTime[int] = 0,
    valid_len: CompileTime[int] = 0,
):
    kv_heads = kv_heads or heads
    lk = lk or L
    o_ld = o_ld or heads * DFULL
    valid_len = valid_len or lk
    why = check_shape(L, heads, kv_heads, lk)
    assert why is None, why

    blk_ty = np.ndarray[(CHUNK,), np.dtype[np.uint8]]          # L2: 4 pieces
    elem_ty = np.ndarray[(PIECE,), np.dtype[np.uint8]]         # L1: one piece
    q_ty = np.ndarray[(Q_CORE,), np.dtype[np.uint8]]
    p_ty = np.ndarray[(TQ * LKP * 9 // 8,), np.dtype[np.uint8]]
    g_ty = np.ndarray[(TQ * LKP,), np.dtype[bfloat16]]
    o_l1_ty = np.ndarray[(TQ * DFULL,), np.dtype[bfloat16]]
    o_l2_ty = np.ndarray[(N_ROWS * TQ * DFULL,), np.dtype[bfloat16]]
    up_ty = np.ndarray[(TQ,), np.dtype[bfloat16]]
    sp_ty = np.ndarray[(TQ * 8,), np.dtype[np.float32]]      # l, replicated per lane
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DFA_TQ={TQ}", f"-DFA_LKP={LKP}", f"-DFA_DC={DC}", f"-DFA_DFULL={DFULL}",
             f"-DFA_QROWS_PASS={QROWS_PASS}", f"-DFA_HS={HS}", *[f"-DFA_X_{x.upper()}" for x in _EXPT if x], f"-DFA_EXP_FIX={EXP_FIX}", f"-DFA_TAU={TAU}",
             "-Wno-deprecated-declarations"]
    # One ExternalFunction compiles the file; every call goes through a plain Kernel
    # sibling (whisper_fa's note: ExternalFunction's arg check rejects acquired bf16
    # elements).
    _ef = ExternalFunction("finalize", source_file=KERNEL_SRC, arg_types=[sp_ty, o_l1_ty],
                           compile_flags=flags)
    obj = _ef.object_file_name
    k_capture = Kernel("capture_q", obj, [elem_ty, q_ty, np.int32])
    k_init = Kernel("pass_init", obj, [o_l1_ty, up_ty, sp_ty])
    # every per-chunk kernel takes (rtp, pass, chunk) and returns at once on a chunk wholly
    # past valid_len
    k_qk = Kernel("qk", obj, [q_ty, elem_ty, g_ty, np.int32, rtp_ty, np.int32, np.int32])
    k_pv = Kernel("pv", obj, [p_ty, elem_ty, o_l1_ty, np.int32, rtp_ty, np.int32, np.int32])
    k_mask = Kernel("apply_mask", obj, [g_ty, rtp_ty, np.int32, np.int32])
    k_step = Kernel("softmax_step", obj, [g_ty, up_ty, sp_ty, o_l1_ty, p_ty, rtp_ty, np.int32, np.int32])
    k_fin = Kernel("finalize", obj, [sp_ty, o_l1_ty])
    kernels = [k_capture, k_init, k_qk, k_pv, k_mask, k_step, k_fin]

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
        # host-packed in pop order: the memtile only copies, a block leaving as 4 pieces
        a2 = fa.cons().forward(tile=Tile(g * GROUP_COLS, 1), obj_type=elem_ty, depth=2,
                               name=f"A_L2L1_{g}")
        b2 = fb.cons().forward(tile=Tile(g * GROUP_COLS + 1, 1), obj_type=elem_ty, depth=2,
                               name=f"B_L2L1_{g}")
        A_l1.append([a2.cons(depth=2) for _ in range(CORES)])
        B_l1.append([b2.cons(depth=2) for _ in range(CORES)])

    # O: a column's 4 cores (32 rows each, block layout) join at its memtile, which
    # streams each part out row-major (a join's output dims apply per part), so the
    # column's 128 rows leave in order.
    out_dims = [(TQ // 8, DFULL * 8), (8, 8), (DFULL // 8, 64), (8, 1)]   # O [row block][col block][8][8]
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
        def core_fn(a_in, b_in, o_out, kern, q_saved, g_buf, p_buf, up, sp, my_rtp, barrier, *dummy):
            (capture, init, qk, pv, mask, step, fin) = kern
            barrier.wait_for_value(1)
            n_heads = my_rtp[RTP_HEADS]
            n_pass = my_rtp[RTP_PASSES]
            n_chunks = my_rtp[RTP_CHUNKS]
            for _ in range_(n_heads):
                for p in range_(n_pass):
                    o = o_out.acquire(1)
                    # Q: 32 pieces, 2 a core in core order, cores 0-7 on A, 8-15 on B
                    for k in range(2 * CORES):
                        fifo = a_in if k < CORES else b_in
                        e = fifo.acquire(1)
                        if k // 2 == i:
                            capture(e, q_saved, k % 2)
                        fifo.release(1)
                    init(o, up, sp)

                    def k_piece(c, j):
                        if "nokv" in _EXPT:          # ablation: no K / V transfer, math on a dummy
                            if "noqk" not in _EXPT:
                                qk(q_saved, dummy[0], g_buf, j, my_rtp, p, c)
                            return
                        kp = a_in.acquire(1)
                        if "noqk" not in _EXPT:
                            qk(q_saved, kp, g_buf, j, my_rtp, p, c)
                        a_in.release(1)

                    def v_elem(c, e2):
                        if "nokv" in _EXPT:
                            if "nopv" not in _EXPT:
                                pv(p_buf, dummy[0], o, e2, my_rtp, p, c)
                            return
                        ve = b_in.acquire(1)
                        if "nopv" not in _EXPT:
                            pv(p_buf, ve, o, e2, my_rtp, p, c)
                        b_in.release(1)

                    def scores_done(c):
                        mask(g_buf, my_rtp, p, c)
                        if "nosm" not in _EXPT:
                            step(g_buf, up, sp, o, p_buf, my_rtp, p, c)

                    # software-pipelined: chunk c's K pieces interleaved with chunk c-1's V
                    # elements, so fifos A and B drain evenly (one after the other, each
                    # stalled while the core worked on the other: compute and transfer
                    # added up instead of overlapping)
                    for j in range(4):
                        k_piece(0, j)
                    scores_done(0)
                    for c in range_(1, n_chunks):
                        for j in range(4):
                            k_piece(c, j)
                            v_elem(c - 1, j)
                        scores_done(c)
                    for e2 in range(4):
                        v_elem(n_chunks - 1, e2)
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
                    Buffer(p_ty, name=f"p_{g}_{i}"),
                    Buffer(up_ty, name=f"up_{g}_{i}"), Buffer(sp_ty, name=f"sp_{g}_{i}"),
                    rtp[(g, i)], barriers[(g, i)]]
            if "nokv" in _EXPT:
                args.append(Buffer(elem_ty, name=f"dummy_{g}_{i}"))
            workers.append(Worker(make_core_fn(i), args, tile=t, stack_size=STACK))

    heads_local = heads // N_GROUPS
    n_pass = L // QROWS_PASS
    n_chunks = lk // LKP
    gqa = heads // kv_heads

    Q_ty = np.ndarray[(heads * (L // TQ) * Q_CORE,), np.dtype[np.uint8]]
    K_ty = np.ndarray[(kv_heads * (lk // LKP) * CHUNK,), np.dtype[np.uint8]]
    V_ty = np.ndarray[(kv_heads * (lk // LKP) * CHUNK,), np.dtype[np.uint8]]
    O_ty = np.ndarray[(L * o_ld,), np.dtype[bfloat16]]
    W = 2304                           # bytes per BD row (576 words, under the 1024 limit)

    def sequence(q, k, v, o, a_prods, b_prods, o_conses):
        for (g, i), buf in rtp.items():
            buf[RTP_HEADS] = heads_local
            buf[RTP_PASSES] = n_pass
            buf[RTP_CHUNKS] = n_chunks
            buf[RTP_VALID] = valid_len
            buf[RTP_CAUSAL] = 0
            buf[RTP_ROW0] = i * TQ
            buf[RTP_QOFF] = 0
        for b in barriers.values():
            b.set(1)

        inflight: list[TaskGroup] = []
        for hl in range(heads_local):
            for p in range(n_pass):
                tg = TaskGroup()
                for g in range(N_GROUPS):
                    h = g * heads_local + hl
                    kvh = h // gqa
                    qbase = (h * (L // TQ) + p * CORES) * Q_CORE
                    half = CORES // 2 * Q_CORE
                    for prods, off in ((a_prods, 0), (b_prods, half)):
                        prods[g].fill(q, offset=qbase + off, sizes=[half // W, W], strides=[W, 1], group=tg)
                    kvbase = kvh * n_chunks * CHUNK
                    if "nokv" in _EXPT:
                        pass
                    else:
                      a_prods[g].fill(k, offset=kvbase, sizes=[n_chunks, CHUNK // W, W],
                                    strides=[CHUNK, W, 1], group=tg)
                      b_prods[g].fill(v, offset=kvbase, sizes=[n_chunks, CHUNK // W, W],
                                    strides=[CHUNK, W, 1], group=tg)
                    for cc in range(GROUP_COLS):
                        col = g * GROUP_COLS + cc
                        row0 = p * QROWS_PASS + cc * N_ROWS * TQ
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
DESIGN = dit_fa256b
SPECIALIZE = dict(
    L=int(os.environ.get("DF_L", 1024)),
    heads=int(os.environ.get("DF_HEADS", 24)),
    kv_heads=int(os.environ.get("DF_KV_HEADS", 4)),
    lk=int(os.environ.get("DF_LK", 1024)),
    # the engine patches DF_VALID (a sentinel word) to the prompt's length in the stream
    valid_len=int(os.environ.get("DF_VALID", 0)),
    o_ld=int(os.environ.get("DF_O_LD", 0)),
)
