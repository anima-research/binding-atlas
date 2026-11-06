"""Threads, the rule-vs-recall sections: knock-out readings and circuit lists → data/built/neurons/icl.json.

Inputs (data/inputs/threads-icl/, tracked):
  k1c-select.json            rule vs recall attribution: map Spearman, top-100 overlap, depth bands, top-10 lists
  k1c-ko-read.txt            rule vs recall knock-outs: base, each set at dose 1 and .5, same-size random sets, and the
                             damage-matched random sets (m = 32: 3,200 neurons) at dose .5
  k1b-select.json            dynamic you: loop share, reuse of static-you supporters against its null, rising heads
  scene-icl.json             dynamic you: the rising set with its depth bands, the top rising heads, the knock-out block
  k1-select.json             static you: the top interferers and supporters with their fan-out ranks, format overlaps
  k1-ko-report.json          static you: kappa (the Atlas cell) and gauge per knock-out condition
  dm-read.txt                static and dynamic you: damage-matched random sets at dose .5 (m = 4, 8, 16, 32)
  loop-heads-report.json     attention heads removed by pass: medians per arm and k, and the damage-controlled fit
  readability-exploratory.json   per set: steering reliability and firing concentration against same-depth draws
  sets.json                  the five circuits as [position, neuron] lists (rule100, recall100, dyn_rise_pos100,
                             you_neg100, you_pos100), and the first damage-matched random set for the rule circuit
                             (dm32_rule100_s0: 32 times rule100's count at each position) and for the dynamic-you
                             circuit (dm32_dyn_rise_pos100_s0), which the map story shows
  transcribed.json           the one reading only published as a table: the 95% interval of dynamic you at 3 shots

Every reading the page shows is cross-checked where two inputs carry it (bands against lists, the dynamic-you set
against the scene file, matched sets against the knock-out block, the base scores against the loop-heads gate). Any
missing row or disagreement refuses the build.

  python build/build_threads_icl.py [--inputs data/inputs/threads-icl] [--out data/built/neurons/icl.json]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
N_POS, N_NEURON = 72, 4096
BANDS = (("prelude", 0, 19), ("pass1", 19, 39), ("pass2", 39, 59), ("coda", 59, 72))
SCENE_BANDS = {"prelude": "prelude", "loop pass 1": "pass1", "loop pass 2": "pass2", "coda": "coda"}
SET_KEYS = {"rule": "rule100", "recall": "recall100", "dyn_you": "dyn_rise_pos100",
            "you_interf": "you_neg100", "you_supp": "you_pos100"}
READ_KEYS = {"rule": "rule_F2", "recall": "recall_C0", "dyn_you": "dyn_switched_on", "you_interf": "static_you_interf",
             "you_supp": "static_you_supp", "broadcasters": "top1000_broadcasters"}
SHOWN_RANDOM = "dm32_rule100_s0"     # the random set the map story lights; its knock-out row is matched["rule100"][0]
SHOWN_RANDOM_DYN = "dm32_dyn_rise_pos100_s0"   # likewise for dynamic you; its 3-shot reading is dm-read.txt's first seed
LOOP_ARMS = ("pass1", "pass2", "outside")
LOOP_KS = (24, 60, 120)
LOOP_K_SHOWN = 60
MATCHED_M = 32
NUM = r"([+-]?\d+\.\d+)"


class InputError(ValueError):
    """An input is missing a row the page needs, or two inputs disagree."""


@dataclass(frozen=True)
class Scores:
    """Changes (or levels, for the base) of the four fluid / crystallized scores, 3-shot dynamic you, and the NLL."""

    F1: float
    F2: float
    F4: float
    C0: float
    dyn3: float
    nll: float


def r4(x: float) -> float:
    return round(float(x), 4)


def need(d: dict, *path: str) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise InputError(f"missing key {'.'.join(path)}")
        cur = cur[k]
    return cur


def load_json(path: Path) -> dict:
    if not path.exists():
        raise InputError(f"missing input {path.name}")
    return json.loads(path.read_text())


def band_of(pos: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= pos < hi:
            return name
    raise InputError(f"position {pos} is outside 0–{N_POS - 1}")


# ---------------------------------------------------------------- k1c-ko-read.txt
ROW_RE = re.compile(rf"^\s*(?P<name>[\w.-]+)\s+F1 {NUM} F2 {NUM} F4 {NUM} C0 {NUM} dyn3 {NUM} dNLL {NUM}\s*$")
BASE_RE = re.compile(rf"^base (\{{.*\}}) dyn3 {NUM} nll {NUM}\s*$")
BLOCK_RE = re.compile(r"^== (?P<set>\w+) dose \.5: target .*\| matched m=(?P<m>\d+)\s*$")


def parse_ko_read(text: str) -> tuple[Scores, dict[str, Scores], dict[str, list[Scores]], dict[str, int]]:
    """→ (base levels, named condition deltas, damage-matched rows per target set, matched m per target set)."""
    base: Scores | None = None
    rows: dict[str, Scores] = {}
    matched: dict[str, list[Scores]] = {}
    mm: dict[str, int] = {}
    block: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if (m := BASE_RE.match(line)) is not None:
            d = ast.literal_eval(m.group(1))
            base = Scores(F1=d["F1"], F2=d["F2"], F4=d["F4"], C0=d["C0"], dyn3=float(m.group(2)), nll=float(m.group(3)))
            continue
        if (m := BLOCK_RE.match(line)) is not None:
            block = m.group("set")
            mm[block] = int(m.group("m"))
            matched[block] = []
            continue
        if (m := ROW_RE.match(line)) is not None:
            block = None
            rows[m.group("name")] = Scores(*[float(m.group(i)) for i in range(2, 8)])
            continue
        if line.lstrip().startswith("dm"):
            m2 = re.match(rf"^\s*(dm\d+ s\d+):\s+F1 {NUM} F2 {NUM} F4 {NUM} C0 {NUM} dyn3 {NUM} dNLL {NUM}\s*$", line)
            if m2 is None or block is None:
                raise InputError(f"k1c-ko-read.txt: unreadable line {line!r}")
            if not m2.group(1).startswith(f"dm{mm[block]} "):
                raise InputError(f"k1c-ko-read.txt: row {m2.group(1)} in a block matched at m={mm[block]}")
            matched[block].append(Scores(*[float(m2.group(i)) for i in range(2, 8)]))
            continue
        raise InputError(f"k1c-ko-read.txt: unreadable line {line!r}")
    if base is None:
        raise InputError("k1c-ko-read.txt: no base row")
    for need_row in ("rule100", "rule100-dose0.5", "recall100", "recall100-dose0.5"):
        if need_row not in rows:
            raise InputError(f"k1c-ko-read.txt: no row {need_row}")
    for s in ("rule100", "recall100"):
        if len(matched.get(s, [])) != 3:
            raise InputError(f"k1c-ko-read.txt: {s} needs 3 damage-matched rows, has {len(matched.get(s, []))}")
        if mm[s] != MATCHED_M:
            raise InputError(f"k1c-ko-read.txt: {s} matched at m={mm[s]}, the page says {MATCHED_M}")
        rnd = [k for k in rows if k.startswith(f"random-{s}-")]
        if len(rnd) != 3:
            raise InputError(f"k1c-ko-read.txt: {s} needs 3 same-size random rows, has {len(rnd)}")
    return base, rows, matched, mm


# ---------------------------------------------------------------- dm-read.txt
DM_HEAD_RE = re.compile(rf"^== (?P<set>\w+): target dNLL {NUM}\s+(?P<what>dkappa|3shot) {NUM}(?: \(base {NUM}\))?\s*$")
DM_ROW_RE = re.compile(rf"^\s+m=\s*(?P<m>\d+) dNLL mean {NUM}\s+(?:dkappa|3shot) \[(?P<vals>[^\]]*)\]\s*$")
DM_MATCH_RE = re.compile(r"^\s+MATCHED m = (?P<m>\d+|None)\s*$")


@dataclass(frozen=True)
class DMBlock:
    target_dnll: float
    target: float
    rows: dict[int, tuple[float, list[float]]]   # m → (mean ΔNLL, readout per seed)
    matched: int | None


def parse_dm_read(text: str) -> dict[str, DMBlock]:
    out: dict[str, DMBlock] = {}
    cur: str | None = None
    acc: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        if (m := DM_HEAD_RE.match(line)) is not None:
            cur = m.group("set")
            acc[cur] = {"target_dnll": float(m.group(2)), "target": float(m.group(4)), "rows": {}, "matched": "unset"}
            continue
        if cur is None:
            raise InputError(f"dm-read.txt: line before any block: {line!r}")
        if (m := DM_ROW_RE.match(line)) is not None:
            vals = [float(v) for v in m.group("vals").split(",")]
            if len(vals) != 3:
                raise InputError(f"dm-read.txt: {cur} m={m.group('m')} needs 3 seeds")
            acc[cur]["rows"][int(m.group("m"))] = (float(m.group(2)), vals)
            continue
        if (m := DM_MATCH_RE.match(line)) is not None:
            acc[cur]["matched"] = None if m.group("m") == "None" else int(m.group("m"))
            continue
        raise InputError(f"dm-read.txt: unreadable line {line!r}")
    for k, v in acc.items():
        if v["matched"] == "unset":
            raise InputError(f"dm-read.txt: {k} has no MATCHED line")
        if MATCHED_M not in v["rows"]:
            raise InputError(f"dm-read.txt: {k} has no m={MATCHED_M} row")
        out[k] = DMBlock(**v)
    for k in ("you_neg100", "you_pos100", "dyn"):
        if k not in out:
            raise InputError(f"dm-read.txt: no block {k}")
    return out


# ---------------------------------------------------------------- the builder
def ids(pairs: list[list[int]]) -> list[str]:
    out = []
    for p in pairs:
        if len(p) != 2 or not (0 <= p[0] < N_POS) or not (0 <= p[1] < N_NEURON):
            raise InputError(f"bad neuron {p}")
        out.append(f"{p[0]}/{p[1]}")
    if len(set(out)) != len(out):
        raise InputError("a circuit lists a neuron twice")
    return sorted(out, key=lambda s: tuple(map(int, s.split("/"))))


def bands(pairs: list[list[int]]) -> dict[str, int]:
    c = Counter(band_of(p) for p, _ in pairs)
    return {name: c.get(name, 0) for name, _, _ in BANDS}


def lo_hi(vals: list[float]) -> dict[str, float]:
    return {"lo": r4(min(vals)), "hi": r4(max(vals))}


def build(inp: Path) -> dict:
    k1c = load_json(inp / "k1c-select.json")
    k1b = load_json(inp / "k1b-select.json")
    scene = load_json(inp / "scene-icl.json")
    k1 = load_json(inp / "k1-select.json")
    k1ko = load_json(inp / "k1-ko-report.json")
    loop = load_json(inp / "loop-heads-report.json")
    read = load_json(inp / "readability-exploratory.json")
    sets = load_json(inp / "sets.json")
    tr = load_json(inp / "transcribed.json")
    base, rows, matched, mm = parse_ko_read((inp / "k1c-ko-read.txt").read_text())
    dm = parse_dm_read((inp / "dm-read.txt").read_text())

    # ---- circuits, and the checks that tie each list to the reading it came from
    lists = {k: need(sets, v) for k, v in SET_KEYS.items()}
    for k, v in lists.items():
        if len(v) != 100:
            raise InputError(f"sets.json: {SET_KEYS[k]} has {len(v)} neurons, not 100")
    as_set = {k: {tuple(p) for p in v} for k, v in lists.items()}
    for k in ("rule", "recall"):
        if bands(lists[k]) != {n: need(k1c, "E6", SET_KEYS[k])[n] for n, _, _ in BANDS}:
            raise InputError(f"{SET_KEYS[k]}: depth bands disagree with k1c-select.json E6")
        for nid in need(k1c, f"{k}_top10"):
            if tuple(map(int, nid.split("/"))) not in as_set[k]:
                raise InputError(f"{nid} is in k1c-select.json {k}_top10 but not in {SET_KEYS[k]}")
    shown, shown_dyn = need(sets, SHOWN_RANDOM), need(sets, SHOWN_RANDOM_DYN)
    for name, rnd, target in ((SHOWN_RANDOM, shown, "rule"), (SHOWN_RANDOM_DYN, shown_dyn, "dyn_you")):
        per_pos = Counter(p for p, _ in lists[target])
        if Counter(p for p, _ in rnd) != {p: min(N_NEURON, MATCHED_M * c) for p, c in per_pos.items()}:
            raise InputError(f"sets.json: {name} is not {MATCHED_M} times {SET_KEYS[target]}'s count at each position")
    shared = len(as_set["rule"] & as_set["recall"])
    if shared != need(k1c, "rule_recall_overlap"):
        raise InputError("rule100 ∩ recall100 disagrees with k1c-select.json rule_recall_overlap")
    if as_set["dyn_you"] != {(d["pos"], d["neuron"]) for d in need(scene, "switched_on")}:
        raise InputError("dyn_rise_pos100 disagrees with scene-icl.json switched_on")
    dyn_bands = bands(lists["dyn_you"])
    if dyn_bands != {SCENE_BANDS[k]: v for k, v in need(scene, "switched_on_by_band").items()}:
        raise InputError("dyn_rise_pos100 depth bands disagree with scene-icl.json")
    loop_share = sum(1 for p, _ in lists["dyn_you"] if 19 <= p < 59) / 100
    if abs(loop_share - need(k1b, "D5b_loop_share", "rise_support")) > 1e-9:
        raise InputError("dyn_rise_pos100 loop share disagrees with k1b-select.json D5b")
    if (19, 3306) not in as_set["you_interf"] or (8, 797) not in as_set["you_interf"]:
        raise InputError("you_neg100 lacks 19/3306 or 8/797")

    # ---- hero: rule vs recall
    gate = need(loop, "gate", "base")
    for k in ("F1", "F2", "F4", "C0"):
        if abs(getattr(base, k) - gate[k]) > 1e-9:
            raise InputError(f"k1c-ko-read.txt base {k} disagrees with loop-heads-report.json gate")

    def sc(s: Scores) -> dict[str, float]:
        d = asdict(s)
        d["dNLL"] = d.pop("nll")
        return {k: r4(v) for k, v in d.items()}

    m_rule, m_recall = matched["rule100"], matched["recall100"]
    for s, ms in (("rule100", m_rule), ("recall100", m_recall)):
        if min(x.nll for x in ms) < rows[f"{s}-dose0.5"].nll:
            raise InputError(f"{s}: a damage-matched set does less damage than the target at dose .5")
    ctrl = max(abs(getattr(x, k)) for x in m_rule + m_recall for k in ("F2", "C0"))
    hero = {
        "base": {k: r4(getattr(base, k)) for k in ("F1", "F2", "F4", "C0")},
        "rule_half": sc(rows["rule100-dose0.5"]), "recall_half": sc(rows["recall100-dose0.5"]),
        "rule_full": sc(rows["rule100"]), "recall_full": sc(rows["recall100"]),
        "matched": {k: {"m": mm[s], "n_neurons": mm[s] * 100, "rows": [sc(x) for x in ms],
                        **{f: lo_hi([getattr(x, f) for x in ms]) for f in ("F1", "F2", "F4", "C0", "nll")}}
                    for k, s, ms in (("rule", "rule100", m_rule), ("recall", "recall100", m_recall))},
        "same_size": {k: [sc(rows[f"random-{s}-{i}"]) for i in range(3)]
                      for k, s in (("rule", "rule100"), ("recall", "recall100"))},
        "control_max_abs": r4(ctrl),
        "rule_half_F2_after": r4(base.F2 + rows["rule100-dose0.5"].F2),
        "recall_half_C0_after": r4(base.C0 + rows["recall100-dose0.5"].C0),
        "random_shown": {"n": len(shown), "row": sc(m_rule[0]), "F2_after": r4(base.F2 + m_rule[0].F2),
                         "C0_after": r4(base.C0 + m_rule[0].C0)},
        "spearman": r4(need(k1c, "maps_spearman")),
        "shared": shared,
        "bands": {"rule": bands(lists["rule"]), "recall": bands(lists["recall"])},
        "rule_top": need(k1c, "rule_top10")[0],
        "recall_top": need(k1c, "recall_top10")[0],
    }
    tp, tw = map(int, hero["rule_top"].split("/"))
    if not 39 <= tp < 59:
        raise InputError("the rule circuit's top neuron is not on pass 2")
    hero["rule_top_twin"] = f"{tp - 20}/{tw}"

    # ---- the loop: heads removed by pass
    med = need(loop, "medians_by_arm_k")
    by_k = {}
    for k in LOOP_KS:
        by_k[str(k)] = {}
        for arm in LOOP_ARMS:
            row = need(med, f"{arm}-k{k}")
            by_k[str(k)][arm] = {"fluid": r4(row["dfluid"]), "C0": r4(row["dC0"]), "dNLL": r4(row["dNLL"])}
    l2 = need(loop, "L2")
    if not l2.get("pass"):
        raise InputError("loop-heads-report.json: L2 did not pass")
    draws = Counter((d["arm"], d["k"]) for d in need(loop, "draws"))
    if len(set(draws.values())) != 1 or set(draws) != {(a, k) for a in LOOP_ARMS for k in LOOP_KS}:
        raise InputError(f"loop-heads-report.json: uneven draws per arm and k: {dict(draws)}")
    loop_out = {"k": LOOP_K_SHOWN, "by_k": by_k, "draws": next(iter(draws.values())),
                "L2": {"pass2": r4(l2["coef_pass2"]), "lo": r4(l2["ci95"][0]), "hi": r4(l2["ci95"][1]),
                       "outside": r4(l2["coef_outside"])}}

    # ---- dynamic you
    ko = need(scene, "ko")
    dmd = dm["dyn"]
    dm32 = dmd.rows[MATCHED_M][1]
    if [round(v, 4) for v in dm32] != [round(v, 4) for v in need(ko, "random_damage_matched_3200", "3shot")]:
        raise InputError("dynamic you: dm-read.txt m=32 disagrees with scene-icl.json")
    if dmd.matched != MATCHED_M:
        raise InputError("dynamic you: not matched at m=32")
    if abs(dmd.target - need(ko, "examples_circuit_out_dose0.5", "3shot")) > 1e-9:
        raise InputError("dynamic you: dm-read.txt target disagrees with scene-icl.json")
    ci = need(tr, "dyn_you_3shot_ci95")
    reuse = need(k1b, "D3")
    heads = need(scene, "heads_rising_top5")
    dyn = {
        "shot0": r4(need(ko, "base", "0shot")), "shot3": r4(need(ko, "base", "3shot")),
        "shot3_ci": {"lo": r4(ci[0]), "hi": r4(ci[1])},
        "half": {k: r4(need(ko, "examples_circuit_out_dose0.5", k)) for k in ("0shot", "3shot", "dNLL")},
        "full": {k: r4(need(ko, "examples_circuit_out_dose1", k)) for k in ("0shot", "3shot", "dNLL")},
        "matched": {"m": MATCHED_M, "n_neurons": MATCHED_M * 100, "shot3": [r4(v) for v in dm32], **lo_hi(dm32),
                    "dNLL_mean": r4(dmd.rows[MATCHED_M][0])},
        "same_size": {"shot3": [r4(v) for v in need(ko, "random_same_size", "3shot")]},
        "random_shown": {"n": len(shown_dyn), "shot3": r4(dm32[0])},
        "bands": dyn_bands,
        "loop_share": r4(loop_share), "loop_share_all": r4(need(k1b, "D5b_loop_share", "all_neurons")),
        "heads": [{"head": h["head"], "d": r4(h["d"])} for h in heads],
        "reuse": {"observed": reuse["observed"], "null_median": reuse["null_median"], "null_p99": reuse["null_p99"]},
    }
    if dyn["heads"][0]["head"] != "39:0":
        raise InputError("dynamic you: the top rising head is not 39:0")

    # ---- static you
    cond = need(k1ko, "conditions")
    c = lambda name, k: need(cond, f"tsumugi-{name}", k)  # noqa: E731
    dmn, dmp = dm["you_neg100"], dm["you_pos100"]
    for name, blk in (("you_neg100", dmn), ("you_pos100", dmp)):
        if round(c(f"{name}-dose0.5", "d_kappa"), 3) != round(blk.target, 3):
            raise InputError(f"static you: {name} Δκ disagrees between k1-ko-report.json and dm-read.txt")
    # the half-dose effects are stored at the precision the page quotes them (the supporters' −.3125 is a binary tie)
    static = {
        "kappa": {"base": r4(c("base", "kappa")), "interf_out": r4(c("you_neg100", "kappa")),
                  "supp_out": r4(c("you_pos100", "kappa"))},
        "gauge": {"base": r4(c("base", "gauge")), "supp_out": r4(c("you_pos100", "gauge")),
                  "interf_out": r4(c("you_neg100", "gauge"))},
        "half": {"interf": {"dkappa": round(c("you_neg100-dose0.5", "d_kappa"), 3),
                            "ci": [r4(v) for v in c("you_neg100-dose0.5", "d_kappa_ci")]},
                 "supp": {"dkappa": round(c("you_pos100-dose0.5", "d_kappa"), 3),
                          "ci": [r4(v) for v in c("you_pos100-dose0.5", "d_kappa_ci")]}},
        "matched": {k: {"m": MATCHED_M, "n_neurons": MATCHED_M * 100, "dkappa": [r4(v) for v in b.rows[MATCHED_M][1]],
                        **lo_hi(b.rows[MATCHED_M][1]), "dNLL_mean": r4(b.rows[MATCHED_M][0]),
                        "under_matched": b.matched is None}
                    for k, b in (("interf", dmn), ("supp", dmp))},
        "top_interferers": [{"id": t["neuron"], "fanout_rank": t["fanout_rank"]}
                            for t in need(k1, "F3", "top10", "interferers")[:2]],
        "format_overlap": {"supp": need(k1, "overlap_vs_R0", "4shot", "pos"),
                           "interf": need(k1, "overlap_vs_R0", "4shot", "neg")},
    }
    if [t["fanout_rank"] for t in static["top_interferers"]] != [1, 2]:
        raise InputError("static you: the top two interferers are not the first two broadcasters")

    # ---- exploratory: readability, as percentiles (0–100) at the one decimal the page quotes
    readable = {"sets": {k: {"n": need(read, "sets", v, "n"),
                             "steer_pct": round(100 * need(read, "sets", v, "steer", "pct"), 1),
                             "conc_pct": round(100 * need(read, "sets", v, "conc", "pct"), 1)}
                         for k, v in READ_KEYS.items()}}

    return {
        "_source": "tsumugi's rule-learning and recall circuits, the loop-pass head knock-outs, and the dynamic- and "
                   "static-you circuits. Circuits: top-100 neurons by attribution, as position/neuron. Knock-outs set "
                   "neurons (or heads) to their corpus mean; dose .5 moves them half way. Scores: fluid and recall "
                   "AUCs; dynamic and static you are the Atlas speaker-swap gauge and kappa cell. Damage-matched "
                   "random sets are 32 times larger, with equal or greater next-token damage.",
        "sets": {**{k: ids(v) for k, v in lists.items()}, "random_rule": ids(shown), "random_dyn": ids(shown_dyn)},
        "hero": hero, "loop": loop_out, "dyn_you": dyn, "static_you": static, "readability": readable,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", type=Path, default=REPO / "data" / "inputs" / "threads-icl")
    ap.add_argument("--out", type=Path, default=REPO / "data" / "built" / "neurons" / "icl.json")
    a = ap.parse_args()
    try:
        out = build(a.inputs)
    except InputError as e:
        raise SystemExit(f"REFUSING: {e}") from e
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e3:.1f} kB): {', '.join(f'{k} {len(v)}' for k, v in out['sets'].items())}")


if __name__ == "__main__":
    main()
