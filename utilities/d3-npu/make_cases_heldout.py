r"""make_cases_heldout: a stratified d3 case set from System One's held-out tiers (test, calib_eval,
external), with gold labels, for accuracy and agreement runs of d3_ref_cpu.py / d3_npu.py.

One group per (dataset, question); multirc's per-answer questions form one group. Each group gives
--per rows chosen by a stable hash of the row id, one question per row. score questions become choice
questions over the same levels (d3 has no score type). Each case keeps "gold" (the probabilities),
"gold_kind", "group" and the source row id.

    python make_cases_heldout.py ~/workspace/system-one/corpus/heldout --per 20 > cases_heldout.json
"""
import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def h(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:12], 16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("heldout", type=Path)
    ap.add_argument("--per", type=int, default=20)
    a = ap.parse_args()
    files = sorted(a.heldout.glob("test/*.jsonl")) + sorted(a.heldout.glob("calib/*.calib_eval.jsonl")) + \
        sorted(a.heldout.glob("external/*.jsonl"))
    out = []
    for f in files:
        if ".derived." in f.name:
            continue
        ds = f.name.split(".")[0]
        groups = defaultdict(list)    # group -> [(hash, row id, question name)]
        rows = {}
        for line in f.open():
            r = json.loads(line)
            rows[r["id"]] = r
            names = [q for q in r["questions"] if q in r.get("gold", {})]
            by_group = defaultdict(list)
            for q in names:
                by_group[re.sub(r"^q\d+a\d+$", "answer", q)].append(q)
            for g, qs in by_group.items():
                q = qs[h(r["id"] + g) % len(qs)]
                groups[g].append((h(r["id"] + "|" + g), r["id"], q))
        for g, items in sorted(groups.items()):
            for _, rid, qn in sorted(items)[:a.per]:
                r = rows[rid]
                q = dict(r["questions"][qn])
                gold = r["gold"][qn]["probabilities"]
                if q["type"] == "score":      # levels as a list; gold keys are their indexes 0..n-1
                    q["type"] = "choice"
                    if isinstance(q["criteria"], list):
                        q["criteria"] = {str(i): d for i, d in enumerate(q["criteria"])}
                if q["type"] == "choice" and set(gold) - set(q["criteria"]):
                    print(f"skip {rid} {qn}: gold keys outside the options", file=sys.stderr)
                    continue
                out.append({"id": f"{ds}/{g}/{rid}", "group": f"{ds}/{g}", "state": r["state"], "question": q,
                            "gold": gold, "gold_kind": r.get("gold_kind")})
    print(json.dumps(out, ensure_ascii=False))
    n = defaultdict(int)
    for c in out:
        n[c["group"]] += 1
    print(f"{len(out)} cases in {len(n)} groups", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
