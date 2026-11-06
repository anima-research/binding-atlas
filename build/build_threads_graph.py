"""Threads, the wiring and training chapters: fan-out, registered wiring readings and the closeness array
→ data/built/neurons/graph.json and data/built/neurons/closeness.bin.

Inputs (data/inputs/threads-graph/):
  k2-report.json           the corpus graph's fan-out: hub ranks and their multiple of their position's median, the
                           top 100 broadcasters by position, each position's median fan-out (tracked)
  layer-fanout.f16.bin     per neuron, log10(1 + fan-out / its position's median), row = position*4096 + neuron (tracked);
                           with the medians it gives each neuron's fan-out, and so the top 100 broadcasters
  registered.json          the readings published only as tables in the registered wiring and training results (tracked)
  closeness-f16.npy        [checkpoint × 12 positions × 4096] closeness of each neuron's firing to the reference
  closeness.json           its checkpoints, positions, definition and caveats
                           (these two are not tracked; closeness.sha256, tracked, pins both)

closeness.bin holds a subset of the checkpoints (CHECKPOINTS_TB, nearest available) as uint8: q = round((c + 1) / 2 * 254),
255 where the neuron is constant at either checkpoint (NaN). graph.json carries every checkpoint's per-position median,
so the page's curve uses all of them. Every number the page quotes is cross-checked here where two inputs carry it.

  python build/build_threads_graph.py [--inputs data/inputs/threads-graph] [--out data/built/neurons]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
N_POS, N_NEURON = 72, 4096
TOP = 100
CHECKPOINTS_TB = (0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 13, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)
ANNEAL_TB = (102, 104, 106, 108, 110)
READ_AT_TB = (10, 30, 80, 100)


class InputError(ValueError):
    """An input is missing, altered, or disagrees with another."""


def need(d: dict, *path: str):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise InputError(f"missing key {'.'.join(path)}")
        cur = cur[k]
    return cur


def nid(i: int) -> str:
    return f"{i // N_NEURON}/{i % N_NEURON}"


def r4(x: float) -> float:
    return round(float(x), 4)


def check_pins(inp: Path) -> None:
    pins = (inp / "closeness.sha256").read_text().split("\n")
    for line in filter(None, pins):
        digest, name = line.split()
        f = inp / name
        if not f.exists():
            raise InputError(f"missing {name} (not tracked; copy it in beside closeness.sha256)")
        if hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            raise InputError(f"{name} does not match its sha256 in closeness.sha256")


def build(inp: Path) -> tuple[dict, bytes]:
    k2 = json.loads((inp / "k2-report.json").read_text())
    reg = json.loads((inp / "registered.json").read_text())
    check_pins(inp)
    meta = json.loads((inp / "closeness.json").read_text())

    # ---- broadcasters: rebuild absolute fan-out from the layer and the medians, and check it against the report
    layer = np.fromfile(inp / "layer-fanout.f16.bin", dtype=np.float16).astype(np.float64)
    med = np.asarray(need(k2, "position_median_fanout"), dtype=np.float64)
    if layer.size != N_POS * N_NEURON or med.size != N_POS:
        raise InputError("layer-fanout.f16.bin or the medians have the wrong size")
    fan = (10 ** layer - 1) * np.repeat(med, N_NEURON)
    order = np.argsort(-fan, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, order.size + 1)
    top = [int(i) for i in order[:TOP]]
    by_pos = Counter(i // N_NEURON for i in top)
    if {str(k): v for k, v in by_pos.items()} != need(k2, "top100_by_position"):
        raise InputError("the rebuilt top 100 disagrees with k2-report.json top100_by_position")
    hubs = []
    for hid in ("19/3306", "8/797", "39/3306"):
        h = need(k2, "hubs", hid)
        p, n = map(int, hid.split("/"))
        if int(rank[p * N_NEURON + n]) != h["abs_rank"]:
            raise InputError(f"{hid}: rebuilt rank {int(rank[p * N_NEURON + n])} disagrees with the report's {h['abs_rank']}")
        hubs.append({"id": hid, "rank": h["abs_rank"], "x_median": round(h["x_position_median"], 2)})
    if int(np.argmax(med)) != 0 or by_pos.get(0, 0) != 0:
        raise InputError("position 0 must have the highest median fan-out and supply none of the top 100")

    # ---- position 8's halves, the twins, genre, the edges: registered tables
    p8 = need(reg, "pos8_split")
    wiring = {
        "hubs": hubs, "top100": [nid(i) for i in top], "top100_pos8": by_pos[8],
        "pos8": {**{k: p8[k] for k in ("plus_n", "minus_n", "plus_dmargin", "minus_dmargin", "all_dmargin", "base_acc", "minus_acc")},
                 "minus_gain": r4(p8["minus_acc"] - p8["base_acc"]),
                 "minus_nll_share": r4((p8["minus_nll"] - p8["base_nll"]) / p8["base_nll"])},
        "twins": {k: v for k, v in need(reg, "twins").items() if k != "registration"},
        "genre": {**{k: v for k, v in need(reg, "genre").items() if k != "registration"},
                  "ratio": round(need(reg, "genre", "dialogue_gap") / need(reg, "genre", "code_gap"), 2)},
        "edges": {k: v for k, v in need(reg, "edges").items() if k != "registration"},
    }

    # ---- training: closeness to the reference, per checkpoint
    arr = np.load(inp / "closeness-f16.npy")
    cks = need(meta, "checkpoints")
    if list(arr.shape) != need(meta, "shape") or arr.shape[0] != len(cks) or arr.shape[2] != N_NEURON:
        raise InputError("closeness-f16.npy disagrees with closeness.json's shape")
    pos = need(meta, "positions")
    a32 = arr.astype(np.float32)
    medians = np.nanmedian(a32, axis=2)                           # [checkpoint, position]
    if not np.allclose(medians[-1], 1.0, atol=2e-3):
        raise InputError("the last checkpoint is not the reference (its closeness should be 1)")

    def pick(tb: float, kind: str) -> int:
        cand = [c for c in cks if c["kind"] == kind]
        best = min(cand, key=lambda c: abs(c["tokens_B"] - tb))
        return int(best["index"])

    chosen = sorted({pick(t, "trunk") for t in CHECKPOINTS_TB} | {pick(t, "final anneal") for t in ANNEAL_TB})
    q = np.where(np.isnan(a32[chosen]), 255, np.clip(np.rint((a32[chosen] + 1) / 2 * 254), 0, 254)).astype(np.uint8)
    reads = {}
    for t in READ_AT_TB:
        i = pick(t, "trunk")
        if cks[i]["tokens_B"] != t:
            raise InputError(f"no trunk checkpoint at {t}B")
        m = medians[i]
        reads[str(t)] = {"lo": r4(np.min(m)), "hi": r4(np.max(m)), "top_position": int(pos[int(np.argmax(m))])}
    furthest = all(pos[int(np.argmax(medians[i]))] == 8 for i in range(len(cks) - 1) if cks[i]["tokens_B"] >= 1)
    training = {
        "positions": pos,
        "checkpoints": [{"tokens_B": c["tokens_B"], "kind": c["kind"]} for c in cks],
        "medians": [[r4(v) for v in row] for row in medians],
        "frames": [{"index": i, "tokens_B": cks[i]["tokens_B"], "kind": cks[i]["kind"]} for i in chosen],
        "reads": reads, "pos8_furthest_from_1B": bool(furthest),
        "registered": {k: v for k, v in need(reg, "training").items() if k != "registration"},
        "caveats": [c for c in need(meta, "caveats")],
        "definition": need(meta, "definition"),
    }
    out = {"_source": "tsumugi's corpus graph (fan-out per neuron), registered wiring readings (position-8 sign split, loop "
                      "twins, genre graphs, edge validation), and each graph-position neuron's closeness to its firing at "
                      "the reference checkpoint over training. closeness.bin: frames × 12 positions × 4096, uint8.",
           "wiring": wiring, "training": training}
    return out, q.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "inputs" / "threads-graph")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "built" / "neurons")
    a = ap.parse_args()
    try:
        out, blob = build(a.inputs)
    except (InputError, OSError) as e:
        raise SystemExit(f"REFUSING: {e}") from e
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "graph.json").write_text(json.dumps(out, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    (a.out / "closeness.bin").write_bytes(blob)
    tr = out["training"]
    print(f"wrote graph.json and closeness.bin ({len(blob) / 1e6:.2f} MB, {len(tr['frames'])} frames); reads "
          + ", ".join(f"{k}B {v['lo']:.3f}–{v['hi']:.3f}" for k, v in tr["reads"].items()))


if __name__ == "__main__":
    main()
