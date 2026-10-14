r"""compare: d3 answers of two runs (d3_npu.py / d3_ref_cpu.py outputs), per case and overall.

    python compare.py ref.json test.json
"""
import json
import math
import sys

ref = {r["id"]: r for r in json.load(open(sys.argv[1]))}
test = {r["id"]: r for r in json.load(open(sys.argv[2]))}
agree, dps, kls = 0, [], []
print(f"{'case':>15} {'tok':>5}  {'ref':>14}  {'test':>14}  {'max|dp|':>8}  {'KL':>9}  {'max|dlogit|':>11}")
for k, r in ref.items():
    if k not in test:
        continue
    t = test[k]
    pr, pt = r["probs"], t["probs"]
    ar, at = max(range(len(pr)), key=pr.__getitem__), max(range(len(pt)), key=pt.__getitem__)
    agree += ar == at
    dp = max(abs(a - b) for a, b in zip(pr, pt))
    kl = sum(a * math.log(max(a, 1e-30) / max(b, 1e-30)) for a, b in zip(pr, pt))
    dl = max(abs(a - b) for a, b in zip(r["logits"], t["logits"]))
    dps.append(dp)
    kls.append(kl)
    print(f"{k:>15} {r['tokens']:5d}  {r['keys'][ar]:>8} {pr[ar]:.3f}  {t['keys'][at]:>8} {pt[at]:.3f}  {dp:8.4f}  {kl:9.2e}  {dl:11.3f}")
n = len(dps)
print(f"\n{n} cases: argmax agrees {agree}/{n}; max|dp| mean {sum(dps) / n:.4f} worst {max(dps):.4f}; "
      f"KL mean {sum(kls) / n:.2e} worst {max(kls):.2e}")
