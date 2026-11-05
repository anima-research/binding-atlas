"""An Empty Document: data/inputs/empty-document → data/built/empty-document.json (byte-for-byte reproducible).

Input: runs.jsonl.gz (one line per model after a summary line: each run's class letter, its screened text or null when
withheld, the reason, the coder's evidence span, and whether it ended on its own) and meta.json (labels, years, the
anchors, the coder's validation). The runs are screened before they reach this repository: emails, phone numbers,
links, @handles and street addresses are redacted; runs with a slur, runs where a sexual term meets a term about
minors, and runs the coder declined are withheld. Output: the page's one data file, with per-model counts and the group order the page shows.

Usage: python build/build_empty.py [--src data/inputs/empty-document] [--out data/built/empty-document.json]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ASSISTANT_ERA = ("s", "v", "x")      # chat transcript, assistant voice, synthetic instruction data
CLASSES = ("s", "v", "x", "h", "p")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=REPO / "data" / "inputs" / "empty-document")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "built" / "empty-document.json")
    a = ap.parse_args()
    meta = json.loads((a.src / "meta.json").read_text())
    # split on "\n" only: run texts hold Unicode line separators (U+2028, U+0085) that splitlines() would break on
    lines = [ln for ln in gzip.decompress((a.src / "runs.jsonl.gz").read_bytes()).decode().split("\n") if ln]
    summary = json.loads(lines[0])["summary"]
    models = []
    for line in lines[1:]:
        m = json.loads(line)
        if m["name"] not in meta["models"]:
            raise SystemExit(f"{m['name']}: not in meta.json")
        runs = m["runs"]
        if any(r["c"] not in CLASSES for r in runs):
            raise SystemExit(f"{m['name']}: unknown class letter")
        counts = {c: sum(r["c"] == c for r in runs) for c in CLASSES}
        info = meta["models"][m["name"]]
        models.append({"name": m["name"], "label": info["label"], "year": info["year"],
                       "hero": m["name"] == meta["hero"], "counts": counts,
                       "assistant_era": sum(counts[c] for c in ASSISTANT_ERA), "n": len(runs),
                       "withheld": sum(r["t"] is None for r in runs), "runs": runs})
    idx = {m["name"]: i for i, m in enumerate(models)}
    modern = sorted((m for m in models if m["name"] not in meta["anchors"] and not m["hero"]),
                    key=lambda m: (m["assistant_era"], m["name"]))
    groups = [{"id": "hero", "models": [idx[meta["hero"]]]},
              {"id": "anchors", "models": [idx[n] for n in meta["anchors"]]},
              {"id": "modern", "models": [idx[m["name"]] for m in modern]}]
    out = {"schema": "empty-document/v1", "meta": {k: v for k, v in meta.items() if k != "models"},
           "screen": summary, "groups": groups, "models": models}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    clean = sum(m["assistant_era"] == 0 for m in models)
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB): {len(models)} models, {clean} with no assistant-era run; "
          f"withheld {sum(v for k, v in summary.items() if k.startswith('withheld_')) + summary['uncoded']} of {summary['runs']}")


if __name__ == "__main__":
    main()
