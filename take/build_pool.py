"""Build the item pool that /take serves: fresh draws from the same generators as the scored batteries.

Every item is drawn from the recipe of a scored battery with a seed that battery never used, and rendered exactly as
a model sees it (the bench's own payload builder), so a person and a model read the same kind of text.
  * deixis-v2: the bindprobes deixis generator at the commit that drew `bp-deixis-v2`, replayed with a new seed
    (binding-bench's recipe replay); every variant (deictic, third, twin), every query (self, addressee,
    owner), two and three speakers, under 0shot and `fair` (balanced worked examples from the same draw).
  * joinlite: the static stage gauge, `joinlite_rows(seed=…)`: third, self and addressee queries, original and
    speaker-swapped.
  * frame ladder: the two-speaker deictic items of the same deixis draw, each rendered on every rung of
    `frame-ladder-v1` (R0 as built, R1 scene line, R2 addressee tags, R2-null tag format only, R3 transfers narrated,
    R4 no I / you), 0-shot. A rung item shares its story id with the deixis item it comes from, so no reader sees two
    renderings of one story.

Needs `binding_bench` importable and a kotodama clone (for the historical generator):
    python take/build_pool.py --kotodama PATH --bench PATH --seeds 900001,900002 --out take/data/pool.jsonl

Each output line: {item_id, base_id, battery, slice, condition, seed, prompt, candidates, gold, meta}.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

PINNED_SEEDS = {"bp-deixis-v2": 2026}       # the seed each scored battery used; a pool never reuses it
DEIXIS_COMMIT, DEIXIS_PER_CELL = "7a5a672", 60
CONDITIONS = ("0shot", "fair")


def deixis_items(kotodama: Path, seed: int) -> Iterator[dict[str, Any]]:
    from binding_bench.bindprobes.grid import build_payload_item
    from binding_bench.bindprobes.items import FairPool, load_items
    from binding_bench.registry import EmitRecipe
    from tools.replay_recipe import replay

    with tempfile.TemporaryDirectory(prefix="take-pool-") as tmp:
        path = replay(EmitRecipe(DEIXIS_COMMIT, seed, {"deixis": DEIXIS_PER_CELL}), kotodama, Path(tmp))
        items = load_items(path)
        yield from frame_items(path, items, seed)
    pool = FairPool(items)
    for it in items:
        P, E, q = it.cell
        variant = it.meta.get("variant", "deictic")
        for cond in CONDITIONS:
            pay = build_payload_item(it, cond, pool)
            yield {"item_id": f"dx{seed}-{it.item_id}-{cond}", "base_id": f"dx{seed}-{it.meta['pair_id']}",
                   "battery": "deixis-v2", "slice": f"deixis|{variant}|P{P}|{q}", "condition": cond, "seed": seed,
                   "prompt": pay["prompt"], "candidates": [c.strip() for c in pay["candidates"]], "gold": it.gold,
                   "meta": {"P": P, "E": E, "qkind": q, "variant": variant}}


def frame_items(path: Path, items: list[Any], seed: int) -> Iterator[dict[str, Any]]:
    from binding_bench.binding.stagegauges import frame_rows, swapdyn_rows

    pair = {it.item_id: it.meta["pair_id"] for it in items}
    for r in frame_rows(swapdyn_rows(path)):
        yield {"item_id": f"fr{seed}-{r['id']}", "base_id": f"dx{seed}-{pair[r['base_id']]}", "battery": "frame-ladder",
               "slice": f"frame|{r['rung']}|{r['qkind']}", "condition": "0shot", "seed": seed, "prompt": r["prompt"],
               "candidates": [c.strip() for c in r["candidates"]], "gold": r["gold"],
               "meta": {"rung": r["rung"], "qkind": r["qkind"]}}


def joinlite_items(seed: int) -> Iterator[dict[str, Any]]:
    from binding_bench.binding.stagegauges import JOINLITE_SEED, joinlite_rows

    if seed == JOINLITE_SEED:
        raise ValueError(f"seed {seed} is joinlite-v1's own")
    for r in joinlite_rows(seed=seed):
        scene = r["id"].rsplit("|", 2)[0]
        yield {"item_id": f"jl{seed}-{r['id']}", "base_id": f"jl{seed}-{scene}", "battery": "joinlite",
               "slice": f"joinlite|{r['qkind']}|{r['arm']}", "condition": "0shot", "seed": seed,
               "prompt": r["prompt"], "candidates": [c.strip() for c in r["candidates"]], "gold": r["gold"],
               "meta": {"qkind": r["qkind"], "arm": r["arm"]}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kotodama", type=Path, required=True, help="a kotodama git clone (historical generators)")
    ap.add_argument("--bench", type=Path, required=True, help="a binding-bench checkout (its src/ and tools/)")
    ap.add_argument("--seeds", required=True, help="comma list of fresh seeds")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path[:0] = [str(a.bench / "src"), str(a.bench)]
    seeds = [int(s) for s in a.seeds.split(",")]
    if set(seeds) & set(PINNED_SEEDS.values()):
        raise SystemExit("a pool seed equals a scored battery's seed")
    rows: list[dict[str, Any]] = []
    for s in seeds:
        rows += deixis_items(a.kotodama, s)
        rows += joinlite_items(s)
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate item ids")
    for r in rows:
        if r["gold"] not in r["candidates"]:
            raise SystemExit(f"{r['item_id']}: gold not among candidates")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} items, {len({r['slice'] for r in rows})} slices, seeds {seeds} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
