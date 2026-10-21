"""pipe_trace.py TRACE.csv: where an OFLM_OPEN_PIPE_TRACE run's time went, and what the NPU waited for"""
import csv, sys, collections
ev = list(csv.DictReader(open(sys.argv[1])))
for e in ev:
    e['a'] = float(e['a']); e['b'] = float(e['b']); e['l'] = int(e['l']); e['c'] = int(e['c'])
H = sorted([e for e in ev if e['m'] == 'H'], key=lambda e: e['a'])
N = sorted([e for e in ev if e['m'] == 'N'], key=lambda e: e['a'])
K = [e for e in ev if e['m'] == 'K']
T = max(e['b'] for e in ev)
def tot(es, key):
    d = collections.defaultdict(float)
    for e in es: d[key(e)] += e['b'] - e['a']
    return d
print(f"wall {T/1e3:.1f} s")
for name, es in (("host", H), ("npu", N)):
    busy = sum(e['b'] - e['a'] for e in es)
    print(f"\n{name}: busy {busy/1e3:.1f} s ({100*busy/T:.0f} %), idle {(T-busy)/1e3:.1f} s")
    for k, v in sorted(tot(es, lambda e: e['op']).items(), key=lambda kv: -kv[1]):
        print(f"  {k:6s} {v/1e3:7.1f} s")
print("\nnpu kernels:")
kk = tot(K, lambda e: (e['op'], e['k'].split('_s')[0] if e['k'].startswith('mb_') else e['k']))
for k, v in sorted(kk.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {k[0]:5s} {k[1]:40s} {v/1e3:7.1f} s")
# NPU idle gaps: blame the host op of the same cell that ended last before the NPU op started
byc = collections.defaultdict(list)
for e in H: byc[(e['l'], e['c'])].append(e)
def blame(gaps_after, mine, other_byc, label):
    blamed = collections.defaultdict(float); cnt = collections.defaultdict(int)
    prev = 0.0
    for e in mine:
        gap = e['a'] - prev
        if gap > 0.05:
            cands = [h for h in other_byc[(e['l'], e['c'])] if h['b'] <= e['a'] + 0.5]
            key = max(cands, key=lambda h: h['b'])['op'] if cands else 'start'
            blamed[(key, e['op'])] += gap; cnt[(key, e['op'])] += 1
        prev = max(prev, e['b'])
    tail = T - prev
    print(f"\n{label} idle, by what it waited for (other machine's op -> this op):")
    for k, v in sorted(blamed.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {k[0]:6s} -> {k[1]:6s} {v/1e3:7.2f} s in {cnt[k]} gaps")
    print(f"  (end of run after its last op: {tail/1e3:.2f} s)")
blame(None, N, byc, "npu")
bycn = collections.defaultdict(list)
for e in N: bycn[(e['l'], e['c'])].append(e)
blame(None, H, bycn, "host")
# utilisation over time
print("\nnpu busy by tenth of the run:", " ".join(
    f"{100*sum(max(0, min(e['b'], (i+1)*T/10) - max(e['a'], i*T/10)) for e in N)/(T/10):.0f}%" for i in range(10)))
print("host busy by tenth of the run:", " ".join(
    f"{100*sum(max(0, min(e['b'], (i+1)*T/10) - max(e['a'], i*T/10)) for e in H)/(T/10):.0f}%" for i in range(10)))
