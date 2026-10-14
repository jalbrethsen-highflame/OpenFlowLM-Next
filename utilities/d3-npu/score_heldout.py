r"""score_heldout: accuracy against the gold labels (make_cases_heldout.py) for one or more d3 runs,
and agreement of each run with the first one.

acc = the run's argmax is the gold argmax; nll = -log p(gold argmax); soft = sum of gold-weighted log p
(cross-entropy against the gold distribution); flips / mean|dp| = against the first run.

    python score_heldout.py cases.json base.json test.json [...]
"""
import json
import math
import sys
from collections import defaultdict

cases = {c["id"]: c for c in json.load(open(sys.argv[1]))}
runs = [(p, {r["id"]: r for r in json.load(open(p))}) for p in sys.argv[2:]]
ids = [i for i in cases if all(i in r for _, r in runs)]
base = runs[0][1]


def amax(p):
    return max(range(len(p)), key=p.__getitem__)


print(f"{len(ids)} cases scored")
by_group = defaultdict(lambda: defaultdict(int))
for name, run in runs:
    acc = nll = ce = flips = 0
    dps = []
    for i in ids:
        c, r = cases[i], run[i]
        p = dict(zip(r["keys"], r["probs"]))
        g = c["gold"]
        gk = max(g, key=g.get)
        ok = r["keys"][amax(r["probs"])] == gk
        acc += ok
        by_group[c["group"]][name] += ok
        nll -= math.log(max(p[gk], 1e-12))
        ce -= sum(v * math.log(max(p[k], 1e-12)) for k, v in g.items())
        b = base[i]
        flips += amax(b["probs"]) != amax(r["probs"])
        dps.append(max(abs(x - y) for x, y in zip(b["probs"], r["probs"])))
    n = len(ids)
    ms = sorted(x["ms"] for x in run.values() if "ms" in x)
    print(f"{name}: acc {acc}/{n} = {acc / n:.3f}  nll {nll / n:.3f}  soft-CE {ce / n:.3f}  "
          f"vs base: flips {flips}, mean|dp| {sum(dps) / n:.4f}, worst {max(dps):.3f}  "
          f"median {ms[len(ms) // 2] if ms else 0} ms")
print("\nper group (correct of n):")
ng = defaultdict(int)
for i in ids:
    ng[cases[i]["group"]] += 1
for g in sorted(by_group):
    print(f"  {g:44s} n {ng[g]}  " + "  ".join(f"{by_group[g][name]}" for name, _ in runs))
