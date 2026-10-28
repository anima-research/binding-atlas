"""Cut data/inputs/synthesis.jsonl from a full synthesis table: only the rows build_stage.py reads, under public ids.

Usage: python build/subset_synthesis.py FULL_SYNTHESIS.jsonl NAMES.tsv [OUT.jsonl]   (default data/inputs/synthesis.jsonl)

NAMES.tsv (local, not tracked) maps checkpoint ids of the full table to the public ids the builders use, one per
line: `model<TAB>checkpoint<TAB>public checkpoint`. Every row is renamed first; then build_stage.py runs against
the renamed table and records each (model, checkpoint, metric) it looks up. Kept rows:
  - every row matching a looked-up key (hit or miss, so an ambiguous key stays ambiguous and a missing one missing);
  - every row the trajectory figure scans (RUN_MODELS x RUN_METRICS in build_stage.py, plus avg9 for FLOPs).
Rows keep their order. Only the measurement fields are kept (model, checkpoint, harness, metric, value, n, se, ci,
tokens_B, train_flops, family, params_B); the full table's source paths and notes stay with it. The result must
pass the public-content guard (build/forbidden.py).
"""
from __future__ import annotations

import json
import os
import re
import runpy
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from forbidden import hits  # noqa: E402

KEEP = ("model", "checkpoint", "harness", "metric", "value", "n", "se", "ci", "tokens_B", "train_flops", "family",
        "params_B")


def read_names(path: Path) -> dict[tuple[str, str], str]:
    names: dict[tuple[str, str], str] = {}
    for i, ln in enumerate(path.read_text().splitlines(), 1):
        if not ln.strip() or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) != 3:
            raise SystemExit(f"{path}:{i}: expected model<TAB>checkpoint<TAB>public checkpoint")
        names[(parts[0], parts[1])] = parts[2]
    return names


def touched_keys(table: Path) -> set[tuple[str, str, str]]:
    os.environ["ATLAS_SYNTHESIS"] = str(table)
    import facts                                     # noqa: E402  (loads the renamed table once)

    keys: set[tuple[str, str, str]] = set()
    orig = facts.Facts.row

    def row(self, model, ckpt, metric, *a, **k):     # type: ignore[no-untyped-def]
        keys.add((model, ckpt, metric))
        return orig(self, model, ckpt, metric, *a, **k)

    facts.Facts.row = row                            # type: ignore[method-assign]
    with tempfile.TemporaryDirectory() as td:
        argv = sys.argv
        try:
            sys.argv = ["build_stage.py", "--base", str(REPO / "data" / "built" / "stage.json"),
                        "--out", f"{td}/stage.json", "--ledger-dir", td]
            runpy.run_path(str(HERE / "build_stage.py"), run_name="__main__")
        finally:
            sys.argv = argv
    return keys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    full, names = Path(sys.argv[1]).resolve(), read_names(Path(sys.argv[2]))
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else REPO / "data" / "inputs" / "synthesis.jsonl"
    rows = []
    for line in full.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            r["checkpoint"] = names.get((r["model"], r["checkpoint"]), r["checkpoint"])
            rows.append({k: r[k] for k in KEEP if k in r})
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        renamed = Path(fh.name)
    try:
        keys = touched_keys(renamed)
    finally:
        renamed.unlink()
    stage_mod = (HERE / "build_stage.py").read_text()
    # the scan tables, read from build_stage.py itself so they cannot drift
    run_models = {m for m, _, _ in eval(re.search(r"^RUN_MODELS = (\[.*?\])$", stage_mod, re.S | re.M).group(1),
                                         {"T": "kotodama-tsumugi"})}
    run_metrics = set(eval(re.search(r"^RUN_METRICS = (\{.*?\})$", stage_mod, re.M).group(1)).values()) | {"avg9"}
    kept = [json.dumps(r, ensure_ascii=False) for r in rows
            if (r["model"], r["checkpoint"], r["metric"]) in keys or (r["model"] in run_models and r["metric"] in run_metrics)]
    bad = sorted({h for ln in kept for h in hits(ln)})
    if bad:
        raise SystemExit(f"REFUSING: forbidden names in the cut ({bad}); extend {sys.argv[2]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(kept) + "\n")
    print(f"wrote {out}: {len(kept)} of {len(rows)} rows ({len(keys)} looked-up keys)")


if __name__ == "__main__":
    main()
