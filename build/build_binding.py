"""Neuron Atlas binding data: attribution readouts and knock-out results → data/built/neurons/binding.json.

Inputs (data/inputs/binding/, tracked):
  binding_readouts.json        per family: battery accuracy, where the attribution sits by effective position, token roles
  ablation_sets2.json          the causal sets: 100 supporters / 100 interferers per family (from training items), core28
  ablate2_<family>.jsonl       held-out items, each set mean-ablated, and random same-position sets (3 seeds)
  peritem_<family>.jsonl       held-out items, one row per (condition, item): pick, gold, margin, cell
  pos8_<kind>_<family>.jsonl   optional: whole-position and position-8 split knock-outs, with corpus NLL
  suite/                       optional: the binding suite's readers on knock-out conditions (report.json from
                               `python -m kotodama.eval.binding report --json`, fluid_auc.json from `… fluid --stat auc
                               --json`, nll/<model>.json corpus NLL); model names are tsumugi-<condition>;
                               fluid_examples.json from build/fluid_examples.py

  python build/build_binding.py [--inputs data/inputs/binding] [--out data/built/neurons/binding.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = ["attr", "dialogue", "deixis"]
N_POS = 72


def rows(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def matrix(path: Path) -> dict:
    """{base, set: {d: Δ accuracy, dm: Δ margin, r: mean Δ accuracy of the random sets, rs: their spread, nll?}}"""
    rs = rows(path)
    base = next(r for r in rs if r["condition"] == "base")
    out: dict = {"base": round(base["accuracy"], 4), "n": base["n"], "base_margin": round(base["margin"], 4)}
    if base.get("nll") is not None:
        out["base_nll"] = round(base["nll"], 5)
    rand = defaultdict(list)
    for r in rs:
        if r["condition"].startswith("random:"):
            rand[r["condition"].split(":")[1]].append(r["accuracy"] - base["accuracy"])
    for r in rs:
        c = r["condition"]
        if c == "base" or c.startswith("random:"):
            continue
        e = {"d": round(r["accuracy"] - base["accuracy"], 4), "dm": round(r["margin"] - base["margin"], 4),
             "n_neurons": r["n_neurons"]}
        if rand.get(c):
            e["r"] = round(statistics.fmean(rand[c]), 4)
            e["rs"] = round(max(abs(v) for v in rand[c]), 4)
        if r.get("nll") is not None:
            e["dnll"] = round(r["nll"] - base["nll"], 5)
        out[c] = e
    return out


def peritem(path: Path) -> dict:
    """Per condition: accuracy and share of picks that are the last-mentioned candidate; per cell, the same."""
    by = defaultdict(list)
    for r in rows(path):
        by[r["condition"]].append(r)
    out = {}
    for cond, rs in by.items():
        last = [r["pick"] == r["mention_order"][-1] for r in rs if r["mention_order"]]
        cells = defaultdict(list)
        for r in rs:
            cells[r["cell_key"]].append(r)
        out[cond] = {"n": len(rs), "acc": round(sum(r["correct"] for r in rs) / len(rs), 4),
                     "last": round(sum(last) / max(1, len(last)), 4),
                     "cells": {k: {"n": len(v), "acc": round(sum(r["correct"] for r in v) / len(v), 4)}
                               for k, v in sorted(cells.items())}}
    return out


def suite(d: Path) -> dict:
    """Per condition: the static and dynamic speaker-swap shifts (the scorecard's columns), the fluid families' AUC, the
    F3 stale-value rate and the corpus NLL."""
    rep = json.loads((d / "report.json").read_text())
    fl = json.loads((d / "fluid_auc.json").read_text())
    out = {}
    for model, res in rep.items():
        cond = model.removeprefix("tsumugi-")
        p2 = res.get("joinlite", {}).get("P2", {})
        sd = res.get("swapdyn", {})
        e: dict = {}
        for key, cell in (("static_I", p2.get("swap_self")), ("static_you", p2.get("swap_addressee")),
                          ("dyn_you_0shot", sd.get("0shot/addressee")), ("dyn_you_3shot", sd.get("3shot/addressee"))):
            if cell:
                e[key] = {"shift": round(cell["shift"], 4), "ci": [round(v, 4) for v in cell["ci"]], "reading": cell["reading"]}
        summ = fl["summaries"].get(model, {})
        for fam in ("F1", "F2", "F3", "F4", "C0"):
            if fam in summ:
                e[fam] = round(summ[fam]["auc"], 4)
        if "F3" in summ and "stale" in summ["F3"]:
            e["stale"] = round(summ["F3"]["stale"]["rate"], 4)
        nl = d / "nll" / f"{model}.json"
        if nl.exists():
            n = json.loads(nl.read_text())
            e["nll"], e["dose"], e["n_neurons"] = round(n["nll"], 5), n.get("dose", 1.0), n["n_neurons"]
        out[cond] = e
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "inputs" / "binding")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "built" / "neurons" / "binding.json")
    a = ap.parse_args()
    rd = json.loads((a.inputs / "binding_readouts.json").read_text())
    sets = json.loads((a.inputs / "ablation_sets2.json").read_text())
    fams = sorted(rd["families"])
    out: dict = {
        "n_items": rd["n_items"],
        "families": {f: {"n": rd["families"][f]["n"], "acc": rd["families"][f]["accuracy"],
                         "where": rd["families"][f]["where_mean_attr_share"], "bands": rd["families"][f]["bands"]}
                     for f in fams},
        "when": {f: {k: v["share"] for k, v in rd["when"][f].items()} for f in fams},
        "sets": {}, "matrix": {}, "peritem": {}, "pos8": {},
    }
    for name, lst in sets.items():
        h = Counter(L for L, _ in lst)
        out["sets"][name] = {"n": len(lst), "by_position": [h.get(L, 0) for L in range(N_POS)]}
    sup = {f: {tuple(x) for x in sets[f"{f}_pos100"]} for f in TARGETS}
    inter = {f: {tuple(x) for x in sets[f"{f}_neg100"]} for f in TARGETS}
    out["overlap"] = {f"{a_}_{b_}_{k}": len(s[a_] & s[b_]) for k, s in (("supporters", sup), ("interferers", inter))
                      for i, a_ in enumerate(TARGETS) for b_ in TARGETS[i + 1:]}
    out["core28_rule"] = rd["shared_core"]["rule"]
    for f in TARGETS:
        out["matrix"][f] = matrix(a.inputs / f"ablate2_{f}.jsonl")
        out["peritem"][f] = peritem(a.inputs / f"peritem_{f}.jsonl")
        for kind in ("layers", "split"):
            p = a.inputs / f"pos8_{kind}_{f}.jsonl"
            if p.exists():
                out["pos8"].setdefault(f, {}).update(matrix(p))
    if (a.inputs / "suite" / "report.json").exists():
        out["suite"] = suite(a.inputs / "suite")
    tr = a.inputs / "transfer"                                  # the causal sets: own task, leave-one-task-out, random
    if tr.exists():
        out["transfer"] = {f: matrix(tr / f"transfer_{f}.jsonl") for f in ("attr", "dialogue", "deixis")
                           if (tr / f"transfer_{f}.jsonl").exists()}
        fl = tr / "transfer_fluid_oddseeds.json"
        if fl.exists():
            for fam, res in json.loads(fl.read_text()).items():
                base = res["base"]["acc"]
                out["transfer"][fam] = {"base": base, "n": res["base"]["n"],
                                        **{k: {"d": round(v["acc"] - base, 4)} for k, v in res.items() if k != "base"}}
    if (a.inputs / "agreement.json").exists():                 # reliability and task x task similarity of the sets
        out["agreement"] = json.loads((a.inputs / "agreement.json").read_text())
    if (a.inputs / "suite" / "generative.json").exists():      # blind reads of generated dialogue under the act lever
        out["generative"] = json.loads((a.inputs / "suite" / "generative.json").read_text())
    if (a.inputs / "suite" / "fluid_examples.json").exists():   # build/fluid_examples.py
        out["fluid_examples"] = json.loads((a.inputs / "suite" / "fluid_examples.json").read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {a.out}: {len(out['sets'])} sets, matrices {sorted(out['matrix'])}, pos8 {sorted(out['pos8'])}")


if __name__ == "__main__":
    main()
