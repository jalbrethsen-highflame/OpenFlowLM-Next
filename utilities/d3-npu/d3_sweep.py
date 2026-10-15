r"""d3_sweep: d3 prompt time by length on the NPU engine (one resident engine, each length run
--reps times, the best kept). The prompt is real text (a d3 decision prompt's ids, repeated) cut
to each length; only the time matters here.

    python d3_sweep.py --model ~/models/d3-NPU2 --kernels <dir> --lens 1024,2048,4096 --out sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d3_npu import Engine, Prompter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path.home() / "models/d3-src")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--kernels", type=Path, required=True)
    ap.add_argument("--cli", type=Path, default=Path(__file__).resolve().parents[2] / "src/open_qwen36/build/open_qwen36_cli")
    ap.add_argument("--cases", type=Path, default=Path.home() / "workspace/npu-scratch/d3/heldout/long.json")
    ap.add_argument("--lens", default="1024,2048,4096,8192,16384,32768")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log", type=Path)
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",")]
    pr = Prompter(a.src)
    base = []
    for c in json.loads(a.cases.read_text()):
        base += pr.ids(c["state"], c["question"])
        if len(base) >= max(lens):
            break
    while len(base) < max(lens):
        base += base
    eng = Engine(a.cli, a.model, a.kernels, (max(lens) + 1023) // 1024 * 1024 + 512, log=a.log)
    out = []
    for n in lens:
        ts = []
        for _ in range(a.reps):
            t0 = time.time()
            eng.last_x(base[:n])
            ts.append(time.time() - t0)
        best = min(ts)
        out.append({"tokens": n, "s": round(best, 3), "tok_s": round(n / best, 1), "all": [round(t, 3) for t in ts]})
        print(f"{n:6d} tokens: {best:7.2f} s  {n / best:6.0f} tok/s", file=sys.stderr, flush=True)
        a.out.write_text(json.dumps(out, indent=1))
    eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
