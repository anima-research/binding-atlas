"""Single read path from the synthesis table (data/inputs/synthesis.jsonl) into the atlas builders.

Every number a builder prints comes through `get()` (or `row()`), which demands exactly ONE matching synthesis row
and records it in a ledger. `dump(path)` writes the ledger: the exact rows behind every number a builder wrote.
Derived numbers (ratios, differences) are computed from fetched rows; `derive()` records them with their inputs.

Usage (from a builder):  from facts import F;  v = F.get("kotodama-tsumugi", "anneal-110B-release", "avg9")
The table is read from $ATLAS_SYNTHESIS when set, else data/inputs/synthesis.jsonl (build/subset_synthesis.py
explains how that file is cut from the full synthesis).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
MASTER = Path(os.environ.get("ATLAS_SYNTHESIS") or HERE.parent / "data" / "inputs" / "synthesis.jsonl")
RELEASE = ("kotodama-tsumugi", "anneal-110B-release")     # the released checkpoint, 110B tokens
ANNEAL1 = ("kotodama-tsumugi", "anneal-110B-second")      # the second anneal from the same 100B fork


class FactError(LookupError):
    pass


@dataclass
class Facts:
    rows: list[dict[str, Any]]
    ledger: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = MASTER) -> "Facts":
        if not path.exists():
            sys.exit(f"synthesis table not found at {path} (set ATLAS_SYNTHESIS)")
        rows: list[dict[str, Any]] = []
        with path.open() as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise FactError(f"{path.name} line {i}: {e}") from e
        return cls(rows)

    def row(self, model: str, ckpt: str, metric: str, harness: str | None = None, src: str | None = None,
            tag: str = "") -> dict[str, Any]:
        hits = [r for r in self.rows if r["model"] == model and r["checkpoint"] == ckpt and r["metric"] == metric
                and (harness is None or r["harness"] == harness) and (src is None or src in r.get("source_path", ""))]
        if len(hits) != 1:
            # identical values from two batteries (full vs core decomposition) are one fact, not an ambiguity
            vals = {json.dumps(h["value"]) for h in hits}
            if len(hits) > 1 and len(vals) == 1 and src is None:
                hits = hits[:1]
            else:
                raise FactError(f"{len(hits)} rows for {model!r} / {ckpt!r} / {metric!r} (harness={harness}, src={src})")
        r = hits[0]
        self.ledger.append({"tag": tag, "model": model, "checkpoint": ckpt, "harness": r["harness"], "metric": metric,
                            "value": r["value"], "se": r.get("se"), "ci": r.get("ci"), "n": r.get("n"),
                            "tokens_B": r.get("tokens_B"), "train_flops": r.get("train_flops"),
                            **({"source_path": r["source_path"]} if "source_path" in r else {})})
        return r

    def get(self, model: str, ckpt: str, metric: str, harness: str | None = None, src: str | None = None,
            tag: str = "") -> float:
        v = self.row(model, ckpt, metric, harness, src, tag)["value"]
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError as e:
                raise FactError(f"non-numeric value {v!r} for {model}/{ckpt}/{metric}") from e
        return float(v)

    def flops(self, model: str, ckpt: str, tag: str = "") -> float:
        """Training FLOPs of a checkpoint (the synthesis convention), read off its avg9 or U-BPB row."""
        for metric in ("avg9", "U-BPB", "static you shift"):
            try:
                f = self.row(model, ckpt, metric, tag=tag or f"flops:{metric}")["train_flops"]
            except FactError:
                continue
            if f:
                return float(f)
        raise FactError(f"no FLOPs for {model}/{ckpt}")

    def derive(self, tag: str, value: float, how: str) -> float:
        self.ledger.append({"tag": tag, "derived": True, "value": value, "how": how})
        return value

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(self.ledger, indent=1, ensure_ascii=False))
        print(f"ledger: {len(self.ledger)} entries -> {path}")


F = Facts.load()


def fmt3(v: float, signed: bool = False) -> str:
    """.768 / +.235 / −.021 (unicode minus, no leading zero) — the cards' number style."""
    s = f"{abs(v):.3f}".lstrip("0") if abs(v) < 1 else f"{abs(v):.3f}"
    if v < 0:
        return "−" + s
    return ("+" + s) if signed else s
