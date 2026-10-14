r"""b4probe: straight-line aie2p asm tests of the int4-mantissa block mechanics (b4probe.inc,
peak.cc PEAK_B4PROBE). One function per case, all `void f(const uint8_t *nib, const int32_t *exp,
bfloat16 *c)`: nib = [A 32 B][B old 32 B][B new 32 B], exp = [A el, eh][B old el, eh][B new el, eh],
c = 64 zeros in, one 8 x 8 product out (fp32 accumulator stored as bf16).

    unp<d>   B old in ex6 (x and e), then vldb.unpack of B new into x6, the MAC d cycles later
    eclob    B old exponents set after B old's unpack, then B new unpacked into x6, MAC 20 later:
             A x B new with B old's exponents if the unpack leaves e alone
    mov<d>   B new in ex6 with B old's exponents, then mov el6 / eh6 to B new's, the MAC d after
             the second move

    python3 b4probe.py > b4probe.inc
"""
CASES = [f"unp{d}" for d in range(4, 13)] + ["eclob"] + [f"mov{d}" for d in range(0, 6)]


def body(case):
    o = ["mova r3, #780", "mov crunpacksize, #0",
         "lda r16, [p1, #0]", "lda r17, [p1, #4]", "lda r18, [p1, #8]", "lda r19, [p1, #12]",
         "lda r20, [p1, #16]", "lda r21, [p1, #20]"] + ["nop"] * 8
    o += ["vldb.unpack x0, unpacksign1, [p0, #0]"] + ["nop"] * 10 + ["mov el0, r16", "mov eh0, r17"]
    o += ["vlda.conv.fp32.bf16 cml0, [p2, #0]", "vlda.conv.fp32.bf16 cmh0, [p2, #64]"] + ["nop"] * 10
    if case.startswith("unp"):
        d = int(case[3:])
        o += ["vldb.unpack x6, unpacksign1, [p0, #32]"] + ["nop"] * 10 + ["mov el6, r18", "mov eh6, r19"] + ["nop"] * 10
        o += ["vldb.unpack x6, unpacksign1, [p0, #64]"] + ["nop"] * (d - 1) + ["vmac.f dm0, dm0, ex0, ex6, r3"]
    elif case == "eclob":
        o += ["vldb.unpack x6, unpacksign1, [p0, #32]"] + ["nop"] * 10 + ["mov el6, r18", "mov eh6, r19"] + ["nop"] * 10
        o += ["vldb.unpack x6, unpacksign1, [p0, #64]"] + ["nop"] * 19 + ["vmac.f dm0, dm0, ex0, ex6, r3"]
    else:
        d = int(case[3:])
        o += ["vldb.unpack x6, unpacksign1, [p0, #64]"] + ["nop"] * 10 + ["mov el6, r18", "mov eh6, r19"] + ["nop"] * 10
        o += ["mov el6, r20", "mov eh6, r21"] + ["nop"] * (d - 1 if d > 0 else 0)
        if d == 0:   # the MAC in the same bundle as the second move
            o[-1] = "mov eh6, r21;\t\tvmac.f dm0, dm0, ex0, ex6, r3"
        else:
            o += ["vmac.f dm0, dm0, ex0, ex6, r3"]
    o += ["nop"] * 12 + ["vst.conv.bf16.fp32 cml0, [p2, #0]", "vst.conv.bf16.fp32 cmh0, [p2, #64]"] + ["nop"] * 4
    return o


def main():
    out = ['asm(R"hs(']
    for c in CASES:
        name = f"b4p_{c}"
        out += ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
        out += ["\t" + x for x in body(c)] + ["\tret lr"] + ["\tnop"] * 5
        out += [f".L{name}_end:", f"\t.size\t{name}, .L{name}_end-{name}"]
    out.append(')hs");')
    for c in CASES:
        out.append(f'extern "C" void b4p_{c}(const uint8_t *nib, const int32_t *exp, bfloat16 *c);')
    out.append("static void (*const b4p_cases[])(const uint8_t *, const int32_t *, bfloat16 *) = {"
               + ", ".join(f"b4p_{c}" for c in CASES) + "};")
    out.append(f"static const unsigned b4p_n = {len(CASES)};")
    print("\n".join(out))


if __name__ == "__main__":
    main()
