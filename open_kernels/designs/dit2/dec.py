r"""dit2_dec: a few-row int4 GEMM for decode (and MTP's verify pass): C [32, N] = A [32, K] x B,
B a dit2 int4-mantissa weight (pack.py pack_b4), columns [OFF, OFF + N) of it.

Unlike dit2, B is SPLIT over the cores, not broadcast: one 128-column strip per core (N / 128 =
8, 16, 24 or 32 strips: 8 columns of 1-4 cores), so every weight byte is read once; A -- 32 rows,
the most one hs_b4_call takes -- is read from DRAM once and broadcast to every core one 64-K
quarter at a time (pack_a.py order2 with mt = 32: [K / 64][2304 B]). The cores walk K and keep
their C tile in L1. Decode fills row 0, a verify pass rows 0..k; C goes out row-block major
([row block 4][strip][column block 16][8][8]): the host reads back rows 0-7 of every strip as the
first NS x 2 KB.

The K-step count is a run-time parameter (one RTP word a core, set by each instruction stream), so
one image serves every K, K range and column offset: a new product or share is a new stream only.

The shim's address generator wraps a dimension at 1023 words and iterates at most 64 times, so a
column's B goes as one fill per 64 K-steps, each [K-steps, cores, 3, 352 words] (a 4224 B tile =
3 x 1408 B).

    DD_K=5120 DD_N=4096 DD_OFF=0 python open_kernels/build_design.py open_kernels/designs/dit2/dec.py <out>
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
                      WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.utils import config as _aie_config

HERE = Path(__file__).resolve().parent
_SRC = HERE / "dec.cc"
_AIE_KERNELS_INC = Path(_aie_config.cxx_header_path()) / "aie_kernels"
B4_TILE = 4224                    # one 128-column x 64-K int4 tile
A_Q = 2304                        # one 32-row x 64-K bfp16 quarter
C_T = 32 * 128                    # one core's C tile (bf16 elements)
COLS = 8


def check_shape(K: int, N: int, OFF: int) -> str | None:
    if K % 64 or N % 1024 or not 1 <= N // 1024 <= 4 or OFF % 128:
        return f"dit2_dec needs K % 64, N in 1024..4096 by 1024, OFF % 128 (got K {K} N {N} OFF {OFF})"
    return None


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def dit2_dec(A: In, B: In, C: Out, *, K: CompileTime[int], N: CompileTime[int], OFF: CompileTime[int] = 0,
             B_SIZE: CompileTime[int] = 0, B_DEPTH: CompileTime[int] = 2, KROW: CompileTime[int] = 0,
             stack_size: CompileTime[int] = 0x800):
    """KROW: the weight's full K when this kernel takes only its first K (a K split: the same core
    code as the K x N kernel, B's strips KROW / 64 tiles apart; C is then a partial sum)."""
    why = check_shape(K, N, OFF)
    assert why is None, why
    KT, rows = K // 64, N // 1024
    KTR = (KROW or K) // 64        # the weight's tiles a strip
    a_ty = np.ndarray[(A_Q // 9,), np.dtype[v8bfp16ebs8]]
    b2_ty = np.ndarray[(rows * B4_TILE,), np.dtype[np.uint8]]
    b1_ty = np.ndarray[(B4_TILE,), np.dtype[np.uint8]]
    c1_ty = np.ndarray[(C_T,), np.dtype[bfloat16]]
    c2_ty = np.ndarray[(rows * C_T,), np.dtype[bfloat16]]
    flags = [f"-I{_AIE_KERNELS_INC}", f"-I{HERE.parent.parent / 'include'}"]
    zero = ExternalFunction("dec_zero", source_file=str(_SRC), arg_types=[c1_ty], compile_flags=flags + ["-DDEC_ZERO"])
    mm = ExternalFunction("dec_mm", source_file=str(_SRC), arg_types=[a_ty, b1_ty, c1_ty], compile_flags=flags + ["-DDEC_MM"])

    # A: once from DRAM, broadcast to every core
    a_l3 = ObjectFifo(a_ty, name="A_L3L2", depth=2)
    a_l1 = a_l3.cons().forward(obj_type=a_ty, name="A_L2L1", depth=2)
    # B: per column one stream, the memtile splits each K-step's tiles over the column's cores
    b_l3, b_l1 = [], []
    for col in range(COLS):
        f = ObjectFifo(b2_ty, name=f"B_L3L2_{col}", depth=2)
        b_l3.append(f)
        b_l1.append(f.cons().split([B4_TILE * r for r in range(rows)], obj_types=[b1_ty] * rows,
                                   names=[f"B_L2L1_{col}_{r}" for r in range(rows)], depths=[B_DEPTH] * rows))
    # C: per column, the cores' tiles joined at the memtile
    c_l3, c_l1 = [], []
    for col in range(COLS):
        f = ObjectFifo(c2_ty, name=f"C_L2L3_{col}", depth=1)
        c_l3.append(f)
        c_l1.append(f.prod().join([C_T * r for r in range(rows)], obj_types=[c1_ty] * rows,
                                  names=[f"C_L1L2_{col}_{r}" for r in range(rows)], depths=[1] * rows))

    def core_fn(a, b, c, zero_k, mm_k, rtp, bar):
        bar.wait_for_value(1)
        kt = rtp[0]
        cc = c.acquire(1)
        zero_k(cc)
        for _ in range_(kt):
            aa = a.acquire(1)
            bb = b.acquire(1)
            mm_k(aa, bb, cc)
            b.release(1)
            a.release(1)
        c.release(1)
        bar.release_with_value(1)

    rtp_ty = np.ndarray[(1,), np.dtype[np.int32]]
    rtps, bars, workers = [], [], []
    for col in range(COLS):
        for r in range(rows):
            rtps.append(Buffer(rtp_ty, name=f"rtp_{col}_{r}", initial_value=np.zeros(1, dtype=np.int32), use_write_rtp=True))
            bars.append(WorkerRuntimeBarrier())
            workers.append(Worker(core_fn, [a_l1.cons(), b_l1[col][r].cons(), c_l1[col][r].prod(), zero, mm, rtps[-1], bars[-1]],
                                  stack_size=stack_size))

    b_size = B_SIZE or (OFF // 128 + COLS * rows) * KTR * B4_TILE
    A_ty = np.ndarray[(KT * A_Q // 9,), np.dtype[v8bfp16ebs8]]
    B_ty = np.ndarray[(b_size,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(COLS * rows * C_T,), np.dtype[bfloat16]]

    def sequence(a, b, c, a_prod, b_prods, c_conses):
        for buf in rtps:
            buf[0] = KT
        for bar in bars:
            bar.set(1)
        tg = TaskGroup()
        a_prod.fill(a, group=tg, wait=False)
        for col in range(COLS):
            strip0 = OFF // 128 + col * rows
            for k0 in range(0, KT, 64):
                kn = min(64, KT - k0)
                tap = TensorAccessPattern((b_size,), (strip0 * KTR + k0) * B4_TILE, [kn, rows, 3, B4_TILE // 3],
                                          [B4_TILE, KTR * B4_TILE, B4_TILE // 3, 1])
                b_prods[col].fill(b, tap=tap, group=tg, wait=False)
        # C row-block major: [row block 4][strip NS][16 column blocks][8][8], so a pass of <= 8 rows
        # reads back the first NS x 2 KB only (a core's tile is [row block 4][1024 elements])
        NS = COLS * rows
        for col in range(COLS):
            c_conses[col].drain(c, tap=TensorAccessPattern((COLS * rows * C_T,), col * rows * (C_T // 4), [1, rows, 4, C_T // 4],
                                                          [0, C_T // 4, NS * (C_T // 4), 1]), group=tg, wait=True)
        tg.finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, a_l3.prod(), [f.prod() for f in b_l3], [f.cons() for f in c_l3]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = dit2_dec
SPECIALIZE = dict(K=int(os.environ.get("DD_K", 5120)), N=int(os.environ.get("DD_N", 4096)),
                  OFF=int(os.environ.get("DD_OFF", 0)), B_SIZE=int(os.environ.get("DD_B_SIZE", 0)),
                  B_DEPTH=int(os.environ.get("DD_B_DEPTH", 2)), KROW=int(os.environ.get("DD_KROW", 0)),
                  stack_size=int(os.environ.get("DD_STACK", "0x800"), 0))
