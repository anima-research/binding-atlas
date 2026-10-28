"""Fluid-ICL examples for the Neuron Atlas: items a knock-out flips, with the model's picks before and after.

Reads the binding suite's fluid prompt rows (PROMPTS/fluid-v1-*.jsonl) and their scored rows for two models
(SCORES/<battery>/<model>.jsonl, each with per-candidate log-probabilities), and writes per family:
  flips     items wrong before and right after (fixed), right before and wrong after (broken), out of n
  ladder    accuracy per rung for each model given (base, after, and any --also models)
  examples  up to --per-family fixed items, chosen deterministically: the lowest rung with a fix, then the median one
Output is small (prompts of a few lines) and goes to data/inputs/binding/suite/fluid_examples.json.

  python build/fluid_examples.py --prompts DIR --scores DIR --base tsumugi-base --after tsumugi-all_neg \
      [--also tsumugi-random-all_neg-0] [--out data/inputs/binding/suite/fluid_examples.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BATTERIES = {"F1": "fluid-v1-morph", "F2": "fluid-v1-algebra", "F3": "fluid-v1-rebind", "F4": "fluid-v1-seqind"}
KEEP = ("rung", "cell", "gold", "candidates", "roles", "register", "N", "R", "k_ops", "depth", "stale")


def rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pick(r: dict) -> str:
    lps = r["cand_lps"]
    return max(lps, key=lps.get)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--also", default="", help="comma list of more models for the ladder (e.g. the random control)")
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "inputs" / "binding" / "suite" / "fluid_examples.json")
    a = ap.parse_args()
    models = [a.base, a.after] + [m for m in a.also.split(",") if m]
    out: dict = {"base": a.base, "after": a.after, "families": {}}
    for fam, bat in BATTERIES.items():
        prompts = {r["id"]: r for r in rows(a.prompts / f"{bat}.jsonl")}
        scored = {m: {r["id"]: r for r in rows(a.scores / bat / f"{m}.jsonl")} for m in models}
        ladder: dict = {m: defaultdict(lambda: [0, 0]) for m in models}
        for m in models:
            for r in scored[m].values():
                c = ladder[m][r["rung"]]
                c[0] += int(pick(r) == r["gold"])
                c[1] += 1
        b, af = scored[a.base], scored[a.after]
        ids = sorted(set(b) & set(af), key=lambda i: (b[i]["rung"], i))
        fixed = [i for i in ids if pick(b[i]) != b[i]["gold"] and pick(af[i]) == af[i]["gold"]]
        broken = [i for i in ids if pick(b[i]) == b[i]["gold"] and pick(af[i]) != af[i]["gold"]]
        chosen = []
        if fixed:
            chosen = [fixed[0]] + ([fixed[len(fixed) // 2]] if len(fixed) > 1 else [])
        examples = []
        for i in chosen[:a.per_family]:
            p = prompts[i]
            examples.append({"id": i, "prompt": p["prompt"], **{k: p[k] for k in KEEP if k in p},
                             "before": pick(b[i]), "after": pick(af[i]),
                             "p_gold_before": round(2.718281828 ** b[i]["cand_lps"][p["gold"]], 4),
                             "p_gold_after": round(2.718281828 ** af[i]["cand_lps"][p["gold"]], 4)})
        out["families"][fam] = {
            "battery": bat, "n": len(ids), "fixed": len(fixed), "broken": len(broken),
            "ladder": {m: {str(k): round(v[0] / v[1], 4) for k, v in sorted(ladder[m].items())} for m in models},
            "examples": examples}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {a.out}: " + ", ".join(f"{f} fixed {v['fixed']} / broken {v['broken']} of {v['n']}"
                                         for f, v in out["families"].items()))


if __name__ == "__main__":
    main()
