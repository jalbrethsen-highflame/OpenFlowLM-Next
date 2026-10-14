r"""bw: DDR read bandwidth by shim channel count (BW_CH 1..16, BW_MB per channel, BW_L2=1 via
the memtile like dit2's operands). Channel c reads from shim column c % 8 (two channels per
shim from 9 up) into core (c % 8, 2 + c // 8), which only acquires and releases: the
dispatch time is the stream. Timing tool, not a kernel.

    BW_CH=12 BW_MB=16 python build_design.py designs/peak/bw.py <out>
"""
from __future__ import annotations
import os
import numpy as np
import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile

NCH = int(os.environ.get("BW_CH", 8))
MB = int(os.environ.get("BW_MB", 16))
L2 = os.environ.get("BW_L2", "0") == "1"
OBJ = 8192                       # bytes per object
NOBJ = MB * (1 << 20) // OBJ


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def bw(a: In, b: Out, *, srchash: CompileTime[int] = 0):
    obj_ty = np.ndarray[(OBJ,), np.dtype[np.int8]]
    fifos, cons = [], []
    for c in range(NCH):
        f = ObjectFifo(obj_ty, name=f"S{c}", depth=2)
        fifos.append(f)
        cons.append(f.cons().forward(obj_type=obj_ty, name=f"S{c}_l1", depth=2) if L2 else f)

    def core(i):
        for _ in range_(NOBJ):
            i.acquire(1)
            i.release(1)

    ws = [Worker(core, [cons[c].cons()], tile=Tile(c % 8, 2 + c // 8)) for c in range(NCH)]
    prods = [fifos[c].prod(tile=Tile(c % 8, 0)) for c in range(NCH)]
    out_ty = np.ndarray[(16,), np.dtype[np.int32]]
    in_ty = np.ndarray[(NCH * NOBJ * OBJ,), np.dtype[np.int8]]

    def seq(ai, bo, ps):
        from aie.helpers.taplib import TensorAccessPattern
        per = NOBJ * OBJ
        for c in range(NCH):
            ps[c].fill(ai, tap=TensorAccessPattern((NCH * per,), c * per, [1, per], [per, 1]), wait=c == NCH - 1)

    rt = Runtime(seq, [in_ty, out_ty, prods])
    return Program(iron.get_current_device(), rt, workers=ws).resolve_program()


DESIGN = bw
SPECIALIZE = {"srchash": hash((NCH, MB, L2)) & 0xFFFFFFF}
