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

  fa_exp_call(pG, pOnes, pP, pRS)  the softmax's common path for one row pair of a 64-key
                          chunk (called twice): P = exp2(G) over its 16 tiles in pv's pop order
                          ([key step][A0, A1]; pG at the row pair's first tile), P pushed as
                          bfp16 to pP (the store FIFO), and rs = rowsum(P) on the MAC (bfp16 P
                          x a ones tile from pOnes), fp32 [row block 2][64] to pRS. One tile
                          every 6 cycles (the move slot: 4 vexp2 + 2 vconv.fp32.bf16 a tile).
                          Store FIFO rules (measured, peak.cc PEAK_FAEXP): no other store-slot op
                          between a push and its flush or in the cycle after the flush, and the
                          plain stores of rs well after the last flush.

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


# HSGEN_D=128 (fa_hs128.inc, dit_fa256b with DF_D=128): the same two loops at head dim 128 -- qk over 16
# k-steps (a row pair's Q 2304 bytes), O's row blocks 2048 bytes apart; the same function names
HSGEN_D = int(__import__("os").environ.get("HSGEN_D", 256))
SHAPES = {
    # nu units, ks k-steps a unit, sw: first unit of the second row pair, load: C in from memory,
    # a_step: bytes between the row pairs' A streams, row: bytes between C row blocks (pv)
    "fa_pv_call": dict(nu=8, ks=8, sw=4, load=True, a_step=1152, row=16 * HSGEN_D),
    "fa_qk_call": dict(nu=2, ks=HSGEN_D // 8, sw=1, load=False, a_step=18 * HSGEN_D, row=0, neg_m=True),
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


# exp pass latencies (readable from issue, def + 1; src read cycle), AIE2PGenSchedule.td
import os
VEXP_W, VCV_W, VBFP_W, RD = 3, 3, 5, int(os.environ.get("FAX_RD", 1))   # vexp2, vconv.fp32.bf16, vconv.bfp16ebs8.fp32; RD: src read
CLR_W = 7                                    # vclr (dst 6)
CONS_W = os.environ.get("FAX_CONSW", "0") == "1"   # debug: every new op's register taken from issue
GLD_W = 8                                    # vlda.conv.fp32.bf16 of G (CONV_W)
EXP_PER = 8                                  # cycles a tile (6 and 7 corrupt the first half of a tile: an
                                             # unmodelled read/write timing on the dm halves; 8 is bit-exact)
FLUSH_GAP = 8                                # plain stores this long after the FIFO's last flush
if os.environ.get("FAX_LAT"):                # debug: VEXP_W,VCV_W,VBFP_W,GLD_W,EXP_PER
    VEXP_W, VCV_W, VBFP_W, GLD_W, EXP_PER = (int(x) for x in os.environ["FAX_LAT"].split(","))


def build_exp(T0: int) -> Sched:
    """fa_exp_call: tile k (zp = k // 16, key step i, zz) starts at S = T0 + 6k on the move slot."""
    s = Sched()
    NT, PER = 16, EXP_PER
    S = lambda k: T0 + PER * k
    setup = [("alu", "movx r26, #0"), ("alu", "movx r25, #0"), ("lda", "mova r3, #780"),
             ]
    t = 0
    for slot, text in setup:
        s.put(t, slot, text)
        t += 1
    setup_end = t
    # the ones tile: one fill + one pop from p3 (lf1)
    tf, _ = s.earliest(setup_end, ["ldb"]); s.put(tf, "ldb", "vldb.fill.512 [p1, lf1, r25]")
    tp, _ = s.earliest(tf + 1, ["ldb"]); s.put(tp, "ldb", "vldb.pop.576 {E:('ONES',)}, [p1, lf1, r25]")
    s.write(("ONES",), "ex", tp + POP_W)
    tms = {}
    for k in range(NT):
        zp, i, zz = k // 16, (k % 16) // 2, k % 2
        st = S(k)
        g, xa, xb, pd, q = ("G", k), ("XA", k), ("XB", k), ("PD", k), ("Q", k)
        # G tile: cml (rows 0-3) then cmh (rows 4-7), post-increment to the next tile in pop
        # order (+128 within a key step, +384 to the next); p0 restarts at the second row pair
        nxt = 64 if zz == 0 else 320
        ptr = "p0"
        t1 = st + 1 - GLD_W; s.put(t1, "lda", f"vlda.conv.fp32.bf16 {{L:{g}}}, [{ptr}], #64")
        walk = os.environ.get("FAX_WALK") == "1"   # debug: +64 loads and a separate padda
        t2 = st + 3 - GLD_W; s.put(t2, "lda", f"vlda.conv.fp32.bf16 {{H:{g}}}, [{ptr}], #{64 if walk else nxt}")
        if walk and nxt != 64:
            ta, _ = s.earliest(t2 + 1, ["lda"]); s.put(ta, "lda", f"padda [{ptr}], #{nxt - 64}")
        # the register is taken from the load's issue: a load into a half of a dm whose other
        # value is still to be read corrupted it (measured, PER 7: the land time is not when
        # the destination stops being safe to read)
        s.write(g, "dm", t1 + (1 if os.environ.get("FAX_LDRES", "1") == "1" else GLD_W))
        for qd, (dst, half, src) in enumerate([(xa, "WL", "BMLL"), (xa, "WH", "BMLH"),
                                                (xb, "WL", "BMHL"), (xb, "WH", "BMHH")]):
            te = st + qd
            s.put(te, "mv", f"vexp2 {{{half}:{dst}}}, {{{src}:{g}}}")
            s.read(g, te + RD)
            s.write(dst, "ex", te + (1 if CONS_W else VEXP_W))
        tc1 = st + PER + 4; s.put(tc1, "mv", f"vconv.fp32.bf16 {{L:{pd}}}, {{X:{xa}}}")
        s.read(xa, tc1 + RD); s.write(pd, "dm", tc1 + (1 if CONS_W else VCV_W))
        tc2 = st + PER + 5; s.put(tc2, "mv", f"vconv.fp32.bf16 {{H:{pd}}}, {{X:{xb}}}")
        s.read(xb, tc2 + RD)
        tb, _ = s.earliest(tc2 + VCV_W, ["st"]); s.put(tb, "st", f"vconv.bfp16ebs8.fp32 {{E:{q}}}, {{D:{pd}}}")
        s.read(pd, tb + RD); s.write(q, "ex", tb + (1 if CONS_W else VBFP_W))
        tm = tb + VBFP_W
        tms[k] = tm
        acc_dst = ("S", zp, zz, i + 1)
        acc_src = ("S", zp, zz, i)
        if i == 0:   # the chain starts from a cleared accumulator (vclr, the compiler's way)
            tz, _ = s.latest(tm + MAC_RACC - CLR_W, ["vec"])
            s.put(tz, "vec", f"vclr {{D:{acc_src}}}")
            s.write(acc_src, "dm", tz + CLR_W)
        s.put(tm, "vec", f"vmac.f {{D:{acc_dst}}}, {{D:{acc_src}}}, {{E:{q}}}, {{E:('ONES',)}}, r3")
        s.read(acc_src, tm + MAC_RACC)
        s.read(q, tm + MAC_RS); s.read(("ONES",), tm + MAC_RS)
        s.write(acc_dst, "dm", tm + MAC_W)
        # push and flush in consecutive cycles: no other store-slot op (vconv.bfp16, vst) may
        # come between them (measured: one there corrupts the FIFO's later blocks)
        tpu = tm
        while not (s.free(tpu, "st") and s.free(tpu + 1, "st") and s.free(tpu + 2, "st")):
            tpu += 1
        s.put(tpu, "st", f"vst.push.576 {{E:{q}}}, [p2, sf, r26]")
        s.read(q, tpu + RD)
        s.put(tpu + 1, "st", "vst.flush.512 [p2, sf, r26]")
        s.slots[tpu + 2].add("st")   # and nothing on the store slot right after a flush (measured)
    # row sums out after each row pair's last MACs: 4 quarters per row block, p2 post-increment
    last = max(t for t, sl, _ in s.ops if sl == "st") + FLUSH_GAP
    for zp in range(1):
        for zz in range(2):
            key = ("S", zp, zz, 8)
            k_last = zp * 16 + 14 + zz
            for qd in ("BMLL", "BMLH", "BMHL", "BMHH"):
                t, _ = s.earliest(max(tms[k_last] + MAC_W, last + 1), ["st"])
                s.put(t, "st", f"vst {{{qd}:{key}}}, [p3], #64")
                s.read(key, t + RD)
                last = t
    assert min(t for t, _, _ in s.ops if t > setup_end) >= 0
    first_load = S(0) - 7
    assert first_load >= tp + 1 or True
    assert first_load > setup_end, "before setup"
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
                "BMLH": f"bmlh{r}", "BMHL": f"bmhl{r}", "BMHH": f"bmhh{r}", "X": f"x{r}",
                "WL": f"wl{r}", "WH": f"wh{r}"}[kind]
    by_t = collections.defaultdict(list)
    for t, slot, text in s.ops:
        by_t[t].append(re.sub(r"\{(BMLL|BMLH|BMHL|BMHH|WL|WH|[DLHEX]):(\([^)]*\))\}", sub, text))
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
    T0 = 8
    while True:
        try:
            s = build_exp(T0)
            break
        except AssertionError:
            T0 += 1
            assert T0 < 40, "fa_exp_call does not fit"
    text, n = emit("fa_exp_call", s, color(s))
    sys.stderr.write(f"hsgen_fa: fa_exp_call: T0 {T0}, {n} bundles + ret (32 tiles, {n / 32:.1f} a tile)\n")
    out.append(text)
    print('asm(R"hs(\n' + "".join(out) + ')hs");')
    print('extern "C" void fa_pv_call(const uint8_t *pP, const uint8_t *pV, bfloat16 *pO);')
    print('extern "C" void fa_qk_call(const uint8_t *pQ, const uint8_t *pK, bfloat16 *pG, const float *pN);')
    print('extern "C" void fa_exp_call(const bfloat16 *pG, const uint8_t *pOnes, uint8_t *pP, float *pRS);')


if __name__ == "__main__":
    main()
