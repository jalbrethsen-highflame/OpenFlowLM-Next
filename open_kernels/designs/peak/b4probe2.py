r"""b4probe2: does vldb.unpack disturb a FIFO stream popped from the same (ldb) slot? (b4probe2.inc,
peak.cc PEAK_B4PROBE2). Each case pops A block 0 and A block 1 (distinct values, bfp16 in memory
at a) from lf1 with vldb, puts vldb.unpack loads (from nib) around them at the case's distances,
then multiplies each popped block by a fixed weight block (unpacked long before) and stores both
products to c[0..63] and c[64..127]. A case is clean when c[0] / c[64] equal the block-0 / block-1
products of the reference case (no unpacks).

    void f(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c)
"""
CASES = {"none": [], **{f"p{g}u{d}": [("g", g), ("u", g + d)] for g in (1, 2, 3, 4) for d in (1, 2, 3)}}


def body(case):
    o = ["mova r3, #780", "mov crunpacksize, #0", "lda r16, [p2, #0]", "lda r17, [p2, #4]"] + ["nop"] * 8
    # the fixed weight block in ex6, long before
    o += ["vldb.unpack x6, unpacksign1, [p1, #0]"] + ["nop"] * 8 + ["mov el6, r16", "mov eh6, r16"] + ["nop"] * 4
    o += ["vlda.conv.fp32.bf16 cml0, [p3, #0]", "vlda.conv.fp32.bf16 cmh0, [p3, #64]",
          "vlda.conv.fp32.bf16 cml1, [p3, #128]", "vlda.conv.fp32.bf16 cmh1, [p3, #192]"] + ["nop"] * 8
    o += ["movx r24, #0", "nop", "vldb.fill.512 [p0, lf0, r24]"]
    ev = CASES[case]
    seq = []   # (offset relative to pop0 at 0 / pop1 at 2, text)
    seq.append((0, "vldb.pop.576 ex0, [p0, lf0, r24]"))
    ev = dict(ev)
    seq.append((ev.get("g", 20), "vldb.pop.576 ex1, [p0, lf0, r24]"))
    if "u" in ev:
        seq.append((ev["u"], "vldb.unpack x8, unpacksign1, [p1, #32]"))
    t0 = min(t for t, _ in seq)
    slots = {}
    for t, txt in seq:
        assert t - t0 not in slots, (case, t)
        slots[t - t0] = txt
    for i in range(max(slots) + 1):
        o.append(slots.get(i, "nop"))
    o += ["nop"] * 10 + ["vmac.f dm0, dm0, ex0, ex6, r3", "vmac.f dm1, dm1, ex1, ex6, r3"] + ["nop"] * 10
    o += ["vst.conv.bf16.fp32 cml0, [p3, #0]", "vst.conv.bf16.fp32 cmh0, [p3, #64]",
          "vst.conv.bf16.fp32 cml1, [p3, #128]", "vst.conv.bf16.fp32 cmh1, [p3, #192]"] + ["nop"] * 4
    return o


def main():
    out = ['asm(R"hs(']
    for c in CASES:
        name = f"b4q_{c}"
        out += ["\t.text", f"\t.globl\t{name}", "\t.p2align\t4", f"\t.type\t{name},@function", f"{name}:"]
        out += ["\t" + x for x in body(c)] + ["\tret lr"] + ["\tnop"] * 5
        out += [f".L{name}_end:", f"\t.size\t{name}, .L{name}_end-{name}"]
    out.append(')hs");')
    for c in CASES:
        out.append(f'extern "C" void b4q_{c}(const bfp16ebs8 *a, const uint8_t *nib, const int32_t *exp, bfloat16 *c);')
    out.append("static void (*const b4q_cases[])(const bfp16ebs8 *, const uint8_t *, const int32_t *, bfloat16 *) = {"
               + ", ".join(f"b4q_{c}" for c in CASES) + "};")
    out.append(f"static const unsigned b4q_n = {len(CASES)};")
    out.append("static const char *const b4q_names[] = {" + ", ".join(f'"{c}"' for c in CASES) + "};")
    print("\n".join(out))


if __name__ == "__main__":
    main()
