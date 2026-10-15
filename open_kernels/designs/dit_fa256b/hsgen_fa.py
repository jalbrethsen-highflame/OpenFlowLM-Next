r"""hsgen_fa: hand-scheduled aie2p assembly for dit_fa256b's two MAC loops (fa_hs.inc, included by
fa256b.cc under FA_HS). designs/peak/hsgen.py's scheduler (every instruction placed by cycle,
latencies from llvm-aie's AIE2PGenSchedule.td, the 5 accumulators and 12 block registers
assigned by live range), with two shapes:

  fa_pv_call(pP, pV, pO)  O[32 rows x 64 d] += P[32 x 64 keys] V: 8 units (row pair 2 x column
                          pair 4) of 4 accumulators x 8 k-steps. P is [row pair][k 8][A0, A1],
                          V [column pair 4][k 8][B0, B1], O bf16 [row block 4][column block 32]
                          [8][8] (pO at the call's first column block; rows 4096 bytes apart).
  fa_qk_call(pQ, pK, pG, pN)  G[32 rows x 16 keys] = Q K^T - m over 256 d: 2 units (row pairs)
                          of 4 accumulators x 32 k-steps, each chain started from its rows' -m
                          (pN: fp32 [row block 4][8 rows][8 lanes], replicated per row; one dm
                          per row block, 4 vlda each, read by both chains of that row block).
                          Q is [row pair][k 32][A0, A1], K [k 32][B0, B1], G bf16 [column
                          block][row block 4][8][8] (pG at the piece's first column block; p4
                          walks its first column block, p5 the second).

    python3 hsgen_fa.py > fa_hs.inc
"""
from __future__ import annotations

import collections
import re
import sys

POP_W, CONV_W, MAC_W = 9, 8, 7   # readable from issue: vld*.pop.576, vlda.conv.fp32.bf16, vmac.f
VLD_W = 8                        # plain vlda into a dm quarter (taken as the conv load's, 8)
MAC_RS, MAC_RACC, ST_R = 1, 4, 1  # read cycles: vmac s1/s2, vmac acc1, vst.conv src
NDM, NEX = 5, 12
LAG = (0, 0, 1, 1)               # k-steps chain i trails chain 0 (hsgen.py)


class Sched:
    def __init__(self):
        self.slots = collections.defaultdict(set)
        self.ops = []
        self.iv = {}

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


SHAPES = {
    # nu units, ks k-steps a unit, sw: first unit of the second row pair, load: C in from memory,
    # a_step: bytes between the row pairs' A streams, row: bytes between C row blocks (pv)
    "fa_pv_call": dict(nu=8, ks=8, sw=4, load=True, a_step=1152, row=4096),
    "fa_qk_call": dict(nu=2, ks=32, sw=1, load=False, a_step=4608, row=0, neg_m=True),
}


def build(T0: int, sh: dict) -> Sched:
    NU, KS, SW, LOAD = sh["nu"], sh["ks"], sh["sw"], sh["load"]
    P = 4 * KS
    s = Sched()
    mac = lambda u, k, i: T0 + P * u + 4 * (k + LAG[i]) + i

    # setup (caller-saved registers only): r3 vmac config, r4/r5 A bases per row pair, r6 B base
    setup = [("alu", "movx r24, #0"), ("alu", "movx r25, #0"), ("lda", "mova r3, #780"),
             ("lng", f"movxm r7, #{sh['a_step']}"), ("mv", "mov r4, p0"), ("mv", "mov r6, p1"),
             ("mv", "mov r16, p2"), ("alu", "add r5, r4, r7")]
    if LOAD:   # pv: r16 / r20 / r18 / r19 = C rows 0-3; p2/p3 load rows z / z+1, p4/p5 store them
        setup += [("lng", f"movxm r17, #{sh['row']}"), ("alu", "add r20, r16, r17"),
                  ("alu", "add r18, r20, r17"), ("alu", "add r19, r18, r17"),
                  ("mv", "mov p3, r20"), ("mv", "mov p4, r16"), ("mv", "mov p5, r20")]
    else:      # qk: p4 the piece's first column block, p5 its second (512 bytes on); p3 = -m
        setup += [("lng", "movxm r17, #512"), ("alu", "add r20, r16, r17"),
                  ("mv", "mov p4, r16"), ("mv", "mov p5, r20"), ("mv", "mov p2, p3")]
    t = 0
    for slot, text in setup:
        s.put(t, slot, text)
        t += 1
    setup_end = t

    # MACs: chain i = C00 C01 C10 C11 = (A0,B0) (A0,B1) (A1,B0) (A1,B1)
    for u in range(NU):
        for k in range(KS):
            for i in range(4):
                t = mac(u, k, i)
                a, b = ("A", u, k, i // 2), ("B", u, k, i % 2)
                dst = ("C", u, i, k + 1)
                if k == 0 and not LOAD:   # from the row block's -m, shared by its two chains
                    src = ("N", u, i // 2)
                    s.put(t, "vec", f"vmac.f {{D:{dst}}}, {{D:{src}}}, {{E:{a}}}, {{E:{b}}}, r3")
                    s.read(src, t + MAC_RACC)
                else:
                    src = ("C", u, i, k)
                    s.put(t, "vec", f"vmac.f {{D:{dst}}}, {{D:{src}}}, {{E:{a}}}, {{E:{b}}}, r3")
                    s.read(src, t + MAC_RACC)
                s.write(dst, "dm", t + MAC_W)
                s.read(a, t + MAC_RS)
                s.read(b, t + MAC_RS)

    # stores, ASAP after each chain's last MAC (post-increment)
    last = 0
    for u in range(NU):
        if LOAD and u == SW:
            t0, _ = s.earliest(last + 1, ["mv"]); s.put(t0, "mv", "mov p4, r18")
            t1, _ = s.earliest(t0 + 1, ["mv"]); s.put(t1, "mv", "mov p5, r19")
            last = t1
        for i in (range(4) if LOAD else (0, 2, 1, 3)):
            # pv: p4 row z (C00 C01), p5 row z+1; qk: p4 column block 0 (C00 C10), p5 block 1
            p = ("p4" if i < 2 else "p5") if LOAD else ("p4" if i % 2 == 0 else "p5")
            key = ("C", u, i, KS)
            for half in "LH":
                t, _ = s.earliest(max(mac(u, KS - 1, i) + MAC_W - ST_R, last + 1), ["st"])
                s.put(t, "st", f"vst.conv.bf16.fp32 {{{half}:{key}}}, [{p}], #64")
                s.slots[t].add("cbank")
                s.read(key, t + ST_R)
                last = t

    if sh.get("neg_m"):   # -m per row block: 4 x 64 bytes from p2 (post-increment), ALAP
        bound = 10 ** 9
        for u in reversed(range(NU)):
            for r2 in reversed(range(2)):
                key = ("N", u, r2)
                first = min(mac(u, 0, 2 * r2), mac(u, 0, 2 * r2 + 1)) + MAC_RACC
                for q in reversed(("bmll", "bmlh", "bmhl", "bmhh")):
                    t, sl = s.latest(min(first - VLD_W, bound), ["lda", "ldb"])
                    s.put(t, sl, f"{'vlda' if sl == 'lda' else 'vldb'} {{{q.upper()}:{key}}}, [p2], #64")
                    s.write(key, "dm", t + VLD_W)
                    bound = t - 1
                s.read(key, first)
        assert bound + 1 >= setup_end, "-m before setup"

    if LOAD:   # C in (lda only, ALAP; p2 row z, p3 row z+1)
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
            if u == SW:
                t0, _ = s.latest(bound, ["mv"]); s.put(t0, "mv", "mov p3, r19")
                t1, _ = s.latest(t0 - 1, ["mv"]); s.put(t1, "mv", "mov p2, r18")
                bound = t1 - 1
        assert min(first_conv.values()) >= setup_end, "conv before setup"

    # FIFO streams, ALAP: A on p0/lf0/r24 (restart every unit), B on p1/lf1/r25 (restart at the
    # second row pair). A fill before every 8 pops (hsgen.py's measured pattern).
    def stream(name, ptr, lf, pos, deadline, restart):
        bound = 10 ** 9
        for u in reversed(range(NU)):
            seq = []
            for g in range(KS // 4):
                seq.append(("fill", None))
                for k in range(4 * g, 4 * g + 4):
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

    stream("A", "p0", "lf0", "r24",
           lambda u, key: min(mac(u, key[2], 2 * key[3]), mac(u, key[2], 2 * key[3] + 1)) + MAC_RS - POP_W,
           lambda u: (None if u == 0 else ("r4" if u < SW else "r5")))
    stream("B", "p1", "lf1", "r25",
           lambda u, key: min(mac(u, key[2], key[3]), mac(u, key[2], key[3] + 2)) + MAC_RS - POP_W,
           lambda u: ("r6" if u == SW else None))
    return s


def color(s: Sched) -> dict:
    regs = {}
    for cls, phys in {"dm": list(range(NDM)), "ex": list(range(NEX))}.items():
        ivs = sorted((v[1], v[2], k) for k, v in s.iv.items() if v[0] == cls)
        busy = {r: -1 for r in phys}
        for st, en, k in ivs:
            assert en >= st, (k, st, en)
            cands = [r for r in phys if busy[r] < st]
            if not cands:
                raise SystemExit(f"{cls}: no register at cycle {st} for {k}")
            r = min(cands, key=lambda r: busy[r])
            busy[r] = en
            regs[k] = r
    return regs


def emit(name: str, s: Sched, regs: dict):
    def sub(m):
        kind, key = m.group(1), eval(m.group(2))
        r = regs[key]
        return {"D": f"dm{r}", "L": f"cml{r}", "H": f"cmh{r}", "E": f"ex{r}", "BMLL": f"bmll{r}",
                "BMLH": f"bmlh{r}", "BMHL": f"bmhl{r}", "BMHH": f"bmhh{r}"}[kind]
    by_t = collections.defaultdict(list)
    for t, slot, text in s.ops:
        by_t[t].append(re.sub(r"\{(BMLL|BMLH|BMHL|BMHH|[DLHE]):(\([^)]*\))\}", sub, text))
    tmax = max(by_t)
    lines = ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
    for t in range(tmax + 1):
        lines.append("\t" + (";\t\t".join(by_t[t]) if by_t[t] else "nop"))
    lines += ["\tret lr"] + ["\tnop"] * 5
    lines += [f".L{name}_end:", f"\t.size\t{name}, .L{name}_end-{name}"]
    return "\n".join(lines) + "\n", tmax + 1


def main():
    out = []
    for name, sh in SHAPES.items():
        T0 = 0
        while True:
            try:
                s = build(T0, sh)
                break
            except AssertionError as e:
                if not any(m in str(e) for m in ("before setup", "no free slot")):
                    raise
                T0 += 1
        text, n = emit(name, s, color(s))
        macs = sh["nu"] * sh["ks"] * 4
        sys.stderr.write(f"hsgen_fa: {name}: T0 {T0}, {n} bundles + ret ({macs} MACs, {macs / n:.0%})\n")
        out.append(text)
    print('asm(R"hs(\n' + "".join(out) + ')hs");')
    print('extern "C" void fa_pv_call(const uint8_t *pP, const uint8_t *pV, bfloat16 *pO);')
    print('extern "C" void fa_qk_call(const uint8_t *pQ, const uint8_t *pK, bfloat16 *pG, const float *pN);')


if __name__ == "__main__":
    main()
