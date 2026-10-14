r"""hsgen: hand-scheduled aie2p assembly for one dit call (C quarter 32 x 128, K = 64).

Writes hs_call(pA, pB, pC): the stock 2x2 microkernel's arithmetic (mm_dit2.cc
matmul_vectorized_2x2_bfp16_bf16 at m=32 k=64 n=128) with every instruction placed by
cycle. A is [row pair 2][k 8][A0, A1] bfp16 blocks, B [column pair 8][k 8][B0, B1],
C [row block 4][column block 16][8][8] bf16. 16 units (2 row pairs x 8 column pairs),
each 4 accumulators x 8 k-steps = 32 MACs; one MAC issues every cycle across units.

Peano rejects inline asm, so the result is a module-level asm block (hs_call.inc,
included by peak.cc). The AIE core has no interlocks: the latencies below are the
operand cycles of llvm-aie's AIE2PGenSchedule.td (def cycle - use cycle), and the
5 accumulators (dm0-dm4) and 12 block registers (ex0-ex11) are assigned by live range.
A MAC may write a different accumulator than it reads, so each value between MACs
gets its own register.

    python3 hsgen.py [GAP [LAG]] > hs_call.inc     (GAP: idle cycles between units, debug)
"""
from __future__ import annotations

import collections
import re
import sys

NU, P = 16, 32                   # units per call, cycles per unit (32 MACs)
# Cycle a result is readable, from issue: the schedule's def cycle + 1 (LLVM operand latency is
# def - use + 1; measured: one cycle less fails the bit-exact check for all three).
POP_W, CONV_W, MAC_W = 9, 8, 7   # vld*.pop.576, vlda.conv.fp32.bf16, vmac.f
MAC_RS, MAC_RACC, ST_R = 1, 4, 1  # read cycles: vmac s1/s2, vmac acc1, vst.conv src
NDM, NEX = 5, 12
GAP = 0                          # idle cycles between units (debug)
LAG = (0, 0, 1, 1)               # k-steps chain i trails chain 0: spreads the unit switches
                                 # so 5 accumulators suffice ((0,1,2,3) runs out of ex registers)


class Sched:
    def __init__(self):
        self.slots = collections.defaultdict(set)
        self.ops = []                          # [time, slot, text]
        self.iv = {}                           # value key -> [class, start, end]

    def free(self, t, slot):
        return t >= 0 and slot not in self.slots[t]

    def put(self, t, slot, text):
        assert self.free(t, slot), (t, slot, text)
        self.slots[t].add(slot)
        self.ops.append([t, slot, text])

    def latest(self, t, slots, also=()):
        while True:
            for s in slots:
                if self.free(t, s) and all(self.free(t, a) for a in also):
                    return t, s
            t -= 1
            assert t >= 0, "no free slot"

    def earliest(self, t, slots, also=()):
        while True:
            for s in slots:
                if self.free(t, s) and all(self.free(t, a) for a in also):
                    return t, s
            t += 1

    def write(self, key, cls, t):
        v = self.iv.setdefault(key, [cls, t, t])
        v[0], v[1] = cls, min(v[1], t)

    def read(self, key, t):
        v = self.iv.setdefault(key, [None, 10 ** 9, t])
        v[2] = max(v[2], t)


def build(T0: int) -> Sched:
    s = Sched()
    mac = lambda u, k, i: T0 + (P + GAP) * u + 4 * (k + LAG[i]) + i

    # setup, one op per cycle (r3: vmac config; r4/r5 A bases per row pair, r6 B base;
    # r18/r19 C rows 2/3 for the second row pair; p3/p5 C row 1 for loads/stores). Only
    # caller-saved registers: r8-r15, p6, p7 and lr are callee-saved (AIE2PCallingConv.td).
    setup = [("alu", "movx r24, #0"), ("alu", "movx r25, #0"), ("lda", "mova r3, #780"),
             ("lng", "movxm r17, #2048"), ("lng", "movxm r7, #1152"), ("mv", "mov r4, p0"),
             ("mv", "mov r6, p1"), ("mv", "mov r16, p2"), ("alu", "add r5, r4, r7"),
             ("alu", "add r20, r16, r17"), ("alu", "add r18, r20, r17"), ("alu", "add r19, r18, r17"),
             ("mv", "mov p3, r20"), ("mv", "mov p4, r16"), ("mv", "mov p5, r20")]
    t = 0
    for slot, text in setup:
        s.put(t, slot, text)
        t += 1
    setup_end = t

    # MACs: unit u, k-step k, chain i in 0..3 = C00 C01 C10 C11 = (A0,B0) (A0,B1) (A1,B0) (A1,B1)
    for u in range(NU):
        for k in range(8):
            for i in range(4):
                t = mac(u, k, i)
                a, b = ("A", u, k, i // 2), ("B", u, k, i % 2)
                src, dst = ("C", u, i, k), ("C", u, i, k + 1)
                s.put(t, "vec", f"vmac.f {{D:{dst}}}, {{D:{src}}}, {{E:{a}}}, {{E:{b}}}, r3")
                s.read(src, t + MAC_RACC)
                s.write(dst, "dm", t + MAC_W)
                s.read(a, t + MAC_RS)
                s.read(b, t + MAC_RS)

    # stores (ASAP after the last MAC; p4 rows z, p5 rows z+1, post-increment)
    last = 0
    zswitch_st = None
    for u in range(NU):
        if u == 8:
            t0, _ = s.earliest(last + 1, ["mv"]); s.put(t0, "mv", "mov p4, r18")
            t1, _ = s.earliest(t0 + 1, ["mv"]); s.put(t1, "mv", "mov p5, r19")
            last = t1
        for i in range(4):
            p = "p4" if i < 2 else "p5"
            key = ("C", u, i, 8)
            for half in "LH":
                t, _ = s.earliest(max(mac(u, 7, i) + MAC_W - ST_R, last + 1), ["st"])
                s.put(t, "st", f"vst.conv.bf16.fp32 {{{half}:{key}}}, [{p}], #64")
                s.slots[t].add("cbank")   # C's memory bank: one access per cycle
                s.read(key, t + ST_R)
                last = t

    # C loads (lda only, ALAP, highest priority on lda; p2 rows z, p3 rows z+1)
    bound = 10 ** 9
    first_conv = {}
    for u in reversed(range(NU)):
        for i in reversed(range(4)):
            p = "p2" if i < 2 else "p3"
            key = ("C", u, i, 0)
            for half in "HL":
                t, _ = s.latest(min(mac(u, 0, i) + MAC_RACC - CONV_W, bound), ["lda"], ("cbank",))
                s.put(t, "lda", f"vlda.conv.fp32.bf16 {{{half}:{key}}}, [{p}], #64")
                s.write(key, "dm", t + CONV_W)
                if half == "H":
                    s.read(key, mac(u, 0, i) + MAC_RACC)
                bound = t - 1
        first_conv[u] = bound + 1
        if u == 8:  # the second row pair's C rows: between unit 7's loads and unit 8's
            t0, _ = s.latest(bound, ["mv"]); s.put(t0, "mv", "mov p3, r19")
            t1, _ = s.latest(t0 - 1, ["mv"]); s.put(t1, "mv", "mov p2, r18")
            bound = t1 - 1
    assert min(first_conv.values()) >= setup_end, "conv before setup"

    # FIFO streams, ALAP: A on p0/lf0/r24 (restart every unit), B on p1/lf1/r25 (restart at
    # the second row pair). Pattern per 16-block stream: fill, 8 pops, fill, 8 pops.
    def stream(name, ptr, lf, pos, deadline, restart):
        bound = 10 ** 9
        for u in reversed(range(NU)):
            seq = []
            for half in range(2):
                seq.append(("fill", None))
                for k in range(4 * half, 4 * half + 4):
                    for r in range(2):
                        seq.append(("pop", (name, u, k, r)))
            for kind, key in reversed(seq):
                lim = bound if kind == "fill" else min(bound, deadline(u, key))
                t, sl = s.latest(lim, ["ldb", "lda"])
                ins = "vlda" if sl == "lda" else "vldb"
                if kind == "fill":
                    s.put(t, sl, f"{ins}.fill.512 [{ptr}, {lf}, {pos}]")
                else:
                    s.put(t, sl, f"{ins}.pop.576 {{E:{key}}}, [{ptr}, {lf}, {pos}]")
                    s.write(key, "ex", t + POP_W)
                bound = t - 1
            src = restart(u)
            if src:
                t0, _ = s.latest(bound, ["mv"]); s.put(t0, "mv", f"mov {ptr}, {src}")
                t1, _ = s.latest(t0, ["alu"]); s.put(t1, "alu", f"movx {pos}, #0")
                bound = min(t0, t1) - 1
            if u == 0:
                assert bound + 1 >= setup_end, f"{name} stream starts before setup"

    stream("A", "p0", "lf0", "r24", lambda u, key: min(mac(u, key[2], 2 * key[3]), mac(u, key[2], 2 * key[3] + 1)) + MAC_RS - POP_W,
           lambda u: (None if u == 0 else ("r4" if u < 8 else "r5")))
    stream("B", "p1", "lf1", "r25", lambda u, key: min(mac(u, key[2], key[3]), mac(u, key[2], key[3] + 2)) + MAC_RS - POP_W,
           lambda u: ("r6" if u == 8 else None))
    return s


def color(s: Sched) -> dict:
    regs = {}
    for cls, n, name in (("dm", NDM, "dm"), ("ex", NEX, "ex")):
        ivs = sorted((v[1], v[2], k) for k, v in s.iv.items() if v[0] == cls)
        busy = [-1] * n
        for st, en, k in ivs:
            assert en >= st, (k, st, en)
            cands = [r for r in range(n) if busy[r] < st]
            if not cands:
                live = [(kk, v[1], v[2]) for kk, v in s.iv.items() if v[0] == cls and v[1] <= st <= v[2]]
                raise SystemExit(f"{cls}: no register at cycle {st} for {k}; live: {live}")
            r = min(cands, key=lambda r: busy[r])   # the register free the longest
            busy[r] = en
            regs[k] = r
    return regs


def emit(s: Sched, regs: dict) -> str:
    def sub(m):
        kind, key = m.group(1), eval(m.group(2))
        r = regs[key]
        return {"D": f"dm{r}", "L": f"cml{r}", "H": f"cmh{r}", "E": f"ex{r}"}[kind]
    by_t = collections.defaultdict(list)
    for t, slot, text in s.ops:
        by_t[t].append(re.sub(r"\{([DLHE]):(\([^)]*\))\}", sub, text))
    tmax = max(by_t)
    lines = ["\t.text", "\t.globl\ths_call", "\t.p2align\t4", "\t.type\ths_call,@function", "hs_call:"]
    for t in range(tmax + 1):
        lines.append("\t" + (";\t\t".join(by_t[t]) if by_t[t] else "nop"))
    lines += ["\tret lr"] + ["\tnop"] * 5
    lines += [".Lhs_end:", "\t.size\ths_call, .Lhs_end-hs_call"]
    return "\n".join(lines) + "\n", tmax + 1


def main():
    global GAP, LAG
    GAP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if len(sys.argv) > 2:
        LAG = tuple(int(x) for x in sys.argv[2].split(","))
    T0 = 0
    while True:
        try:
            s = build(T0)
            break
        except AssertionError as e:
            if "before setup" not in str(e) and "no free slot" not in str(e):
                raise
            T0 += 1
    regs = color(s)
    text, n = emit(s, regs)
    sys.stderr.write(f"hsgen: T0 {T0}, {n} bundles + ret (512 MACs)\n")
    print('asm(R"hs(\n' + text + ')hs");')


if __name__ == "__main__":
    main()
