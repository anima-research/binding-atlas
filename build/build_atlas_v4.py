"""Build data/built/atlas.json from the zoo sweep: every base model on the clean bench, one scorer, one set of batteries.

Every cell is a chance-normalized score κ over items, with exact ties between candidates split evenly (an exact tie is
no evidence for either candidate), its bootstrap 95% interval, and a state: above, at or below chance. Columns are
named by what the model must do and grouped by what the items look like; a group is not a claim that its columns
measure one ability. The few-shot reading is `fair` (balanced worked examples); `0shot` is the floor.

Also writes data/DROPPED-2026-10.tsv: every cell of the previous grid, marked with what happened to it, so the old
rows and columns can be re-added once the clean bench has scored them.

    python build/build_atlas_v4.py --zoo ZOO_DIR --prompts PROMPTS_DIR --bench BINDING_BENCH --out data/built/atlas.json
    (the one-time diff against the previous grid: add --old OLD_ATLAS --dropped data/DROPPED-2026-10.tsv)

ZOO_DIR holds <zoo_dir>/suite/<battery>/<model>.jsonl, <zoo_dir>/riders/rows_<model>.jsonl, <zoo_dir>/zoo_tierA.tsv
and <zoo_dir>/report/report.json; the vLLM-scored bases sit under <zoo_dir>/tierB/ in the same layout, listed in
<zoo_dir>/zoo_tierB.tsv (its two check models keep their standard rows).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

Row = dict[str, Any]
N_BOOT = 1000
COND_READING, COND_FLOOR = "fair", "0shot"

# ---- models --------------------------------------------------------------------------------------------------------
LABELS: dict[str, tuple[str, str]] = {      # zoo name → (label, family)
    "pythia-160m": ("Pythia 160M", "Pythia"), "pythia-1b": ("Pythia 1B", "Pythia"), "pythia-1.4b": ("Pythia 1.4B", "Pythia"),
    "pythia-2.8b": ("Pythia 2.8B", "Pythia"), "pythia-2.8b-deduped": ("Pythia 2.8B (deduped)", "Pythia"),
    "pythia-6.9b": ("Pythia 6.9B", "Pythia"), "smollm2-360m": ("SmolLM2 360M", "SmolLM2"),
    "smollm2-1.7b": ("SmolLM2 1.7B", "SmolLM2"), "gpt2-xl": ("GPT-2 XL", "GPT-2"),
    "qwen2.5-0.5b": ("Qwen2.5 0.5B", "Qwen"), "qwen2.5-1.5b": ("Qwen2.5 1.5B", "Qwen"), "qwen2.5-3b": ("Qwen2.5 3B", "Qwen"),
    "qwen2.5-7b": ("Qwen2.5 7B", "Qwen"), "qwen2.5-72b": ("Qwen2.5 72B", "Qwen"), "qwen3-0.6b": ("Qwen3 0.6B", "Qwen"),
    "qwen3-1.7b": ("Qwen3 1.7B", "Qwen"), "qwen3-4b": ("Qwen3 4B", "Qwen"), "qwen3-30b-a3b": ("Qwen3 30B-A3B (MoE)", "Qwen"),
    "olmo1-1b": ("OLMo 1B", "OLMo"), "olmo2-1b": ("OLMo-2 1B", "OLMo"), "olmo2-7b": ("OLMo-2 7B", "OLMo"),
    "olmo2-13b": ("OLMo-2 13B", "OLMo"), "olmo3-32b": ("OLMo-3 32B", "OLMo"),
    "olmo3-32b-stage1": ("OLMo-3 32B, before mid-training", "OLMo"), "llama3.2-1b": ("Llama-3.2 1B", "Llama"),
    "llama3-8b": ("Llama-3 8B", "Llama"), "llama3.1-8b": ("Llama-3.1 8B", "Llama"), "llama3.1-70b": ("Llama-3.1 70B", "Llama"),
    "llama3.1-405b": ("Llama-3.1 405B", "Llama"), "gemma-2b": ("Gemma 2B", "Gemma"), "gemma-7b": ("Gemma 7B", "Gemma"),
    "gemma2-2b": ("Gemma-2 2B", "Gemma"), "gemma2-9b": ("Gemma-2 9B", "Gemma"), "glm4-32b-base": ("GLM-4 32B", "GLM"),
    "falcon-h1-0.5b": ("Falcon-H1 0.5B", "Falcon"), "falcon-h1-1.5b": ("Falcon-H1 1.5B", "Falcon"),
    "falcon-h1-1.5b-deep": ("Falcon-H1 1.5B-Deep", "Falcon"), "falcon-h1-3b": ("Falcon-H1 3B", "Falcon"),
    "falcon-h1-7b": ("Falcon-H1 7B", "Falcon"), "falcon-h1-34b": ("Falcon-H1 34B", "Falcon"),
    "falcon3-3b": ("Falcon3 3B", "Falcon"), "falcon3-7b": ("Falcon3 7B", "Falcon"),
    "falcon-mamba-7b": ("Falcon-Mamba 7B", "Falcon"), "recurrentgemma-9b": ("RecurrentGemma 9B", "Gemma"),
    "bamba-9b": ("Bamba 9B", "Bamba"), "tsumugi": ("kotodama-tsumugi 1.4B", "kotodama"),
    "tsumugi-s17-a5B": ("kotodama-tsumugi · 5B tokens", "kotodama"), "tsumugi-s17-a11B": ("kotodama-tsumugi · 11B tokens", "kotodama"),
    "tsumugi-s17-a33B": ("kotodama-tsumugi · 33B tokens", "kotodama"), "tsumugi-s17-a55B": ("kotodama-tsumugi · 55B tokens", "kotodama"),
    "tsumugi-s17-a88B": ("kotodama-tsumugi · 88B tokens", "kotodama"),
    "tsumugi-s7-a5B": ("kotodama-tsumugi, seed 7 · 5B tokens", "kotodama"),
    "tsumugi-s7-a11B": ("kotodama-tsumugi, seed 7 · 11B tokens", "kotodama"),
    "dsv3-base": ("DeepSeek-V3 (MoE)", "DeepSeek"), "dsv4-flash-base": ("DeepSeek-V4-Flash (MoE)", "DeepSeek"),
    "kimi-k2-base": ("Kimi-K2 (MoE)", "Kimi"), "trinity-base": ("Trinity-Large (MoE)", "Trinity"),
    "trinity-truebase": ("Trinity-Large (MoE), before the anneals", "Trinity"),
    "nemotron3-ultra-550b-base": ("Nemotron-3-Ultra (MoE)", "Nemotron"), "mimo-v25-base": ("MiMo-V2.5 (MoE)", "MiMo"),
    "mimo-v25-pro-base": ("MiMo-V2.5-Pro (MoE)", "MiMo"), "gemma4-31b-base": ("Gemma-4 31B", "Gemma"),
}
# Creator-stated pretraining tokens (model cards and papers) with the uncertainty mark the page shows.
TOKENS: dict[str, tuple[Optional[float], str]] = {
    **{k: (300e9, "") for k in ("pythia-160m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-2.8b-deduped", "pythia-6.9b")},
    "smollm2-360m": (4e12, ""), "smollm2-1.7b": (11e12, ""),
    **{k: (18e12, "≤") for k in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-72b")},
    **{k: (36e12, "") for k in ("qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-30b-a3b")},
    "olmo1-1b": (3.05e12, ""), "olmo2-1b": (4.05e12, ""), "olmo2-7b": (4.15e12, ""), "olmo2-13b": (5.6e12, ""),
    "olmo3-32b": (5.7e12, ""), "olmo3-32b-stage1": (5.5e12, "stage 1 only"), "gpt2-xl": (None, "—"), "llama3.2-1b": (9e12, "≤ distilled"), "llama3-8b": (15e12, "≥"), "llama3.1-8b": (15e12, "≈"),
    "llama3.1-70b": (15e12, "≈"), "llama3.1-405b": (15.6e12, ""), "gemma-2b": (3e12, ""), "gemma-7b": (6e12, ""),
    "gemma2-2b": (2e12, "distilled"), "gemma2-9b": (8e12, "distilled"), "glm4-32b-base": (15e12, ""),
    "falcon-h1-0.5b": (2.5e12, ""), "falcon-h1-1.5b": (3e12, ""), "falcon-h1-1.5b-deep": (3e12, ""),
    "falcon-h1-3b": (2.5e12, ""), "falcon-h1-7b": (12e12, "≈"), "falcon-h1-34b": (18e12, "≈"),
    "falcon3-3b": (100e9, "distilled"), "falcon3-7b": (14e12, ""), "falcon-mamba-7b": (5.8e12, ""),
    "recurrentgemma-9b": (2e12, ""), "bamba-9b": (3.1e12, "≈"),
    "tsumugi": (110e9, ""), "tsumugi-s17-a5B": (5e9, ""), "tsumugi-s17-a11B": (11e9, ""), "tsumugi-s17-a33B": (33e9, ""),
    "tsumugi-s17-a55B": (55e9, ""), "tsumugi-s17-a88B": (88e9, ""), "tsumugi-s7-a5B": (5e9, ""), "tsumugi-s7-a11B": (11e9, ""),
    "dsv3-base": (14.8e12, ""), "dsv4-flash-base": (32e12, ""), "kimi-k2-base": (15.5e12, ""),
    "trinity-base": (17e12, ""), "trinity-truebase": (10e12, ""), "nemotron3-ultra-550b-base": (20e12, ""),
    "mimo-v25-base": (48e12, "≈ text + multimodal"), "mimo-v25-pro-base": (27e12, ""), "gemma4-31b-base": (None, "—"),
}
# Creator-stated total parameters for the vLLM-scored bases (model cards and reports); their weight files also hold
# extra prediction heads and quantization scales.
CREATOR_SIZE: dict[str, float] = {
    "dsv3-base": 671e9, "dsv4-flash-base": 284e9, "kimi-k2-base": 1e12, "trinity-base": 398e9,
    "trinity-truebase": 398e9, "nemotron3-ultra-550b-base": 550e9, "mimo-v25-base": 310e9,
    "mimo-v25-pro-base": 1.02e12, "gemma4-31b-base": 30.7e9,
}
VLLM_NOTE = ("Scored with vLLM across several GPUs: the same prompts, candidates and split-tie rule. On two check "
             "models, its scores match the standard scorer's within .03.")
BANDS = ((1e9, "under 1B"), (3e9, "1–3B"), (10e9, "3–10B"), (float("inf"), "10B and up"))
TRAJECTORY = "kotodama-tsumugi, by training tokens"


def band(active: float) -> str:
    return next(lab for lim, lab in BANDS if active < lim)


# ---- per-item credit with split ties ---------------------------------------------------------------------------------
def tie_credit(lps: dict[str, float], gold: str) -> float:
    top = max(lps.values())
    tied = [c for c, v in lps.items() if v == top]
    return 1 / len(tied) if gold in tied else 0.0


def kappa_item(credit: float, chance: float) -> float:
    return (credit - chance) / (1 - chance)


def k_choice(r: Row, gold_key: str = "gold") -> tuple[float, float]:
    """(credit, chance) for a single-answer row, split ties."""
    return tie_credit(r["cand_lps"], r[gold_key]), 1 / len(r["cand_lps"])


def k_set(r: Row) -> tuple[float, float]:
    """(credit, chance) for a set-binding row: the share of the tied top candidates that are in the gold set."""
    lps, gs = r["cand_lps"], set(r["gold_set"])
    top = max(lps.values())
    tied = [c for c, v in lps.items() if v == top]
    return sum(c in gs for c in tied) / len(tied), len(gs) / len(lps)


# ---- columns ---------------------------------------------------------------------------------------------------------
@dataclass
class Col:
    id: str
    group: str
    label: str
    what: str
    battery: str                         # suite battery, "riders", or "fluid"
    keep: Callable[[Row], bool]
    credit: Callable[[Row], tuple[float, float]] = k_choice
    conds: tuple[str, ...] = (COND_READING, COND_FLOOR)
    headline: bool = False
    flag: Optional[str] = None
    old: tuple[str, ...] = ()            # previous-grid column ids this one replaces
    kind: str = "kappa"


def bp(fam: str, f: Callable[[Row], bool] = lambda r: True) -> Callable[[Row], bool]:
    return lambda r: r.get("family") == fam and f(r)


def dx(q: str, p: int) -> Callable[[Row], bool]:
    return lambda r: (r.get("family") == "deixis" and r.get("qkind") == q and r.get("variant") == "deictic"
                      and r["cell"][0] == p)


G_STILL, G_STORY, G_CODE, G_BOX, G_YOU, G_RULE = (
    "Who has what? Nothing moves", "Who has what after things move: told as a story",
    "Who has what after things move: written as code", "One box puzzle, four formats",
    "Who are I and you?", "Learning a rule from the examples")
GROUP_SHORT = {G_STILL: "Nothing moves", G_STORY: "Moves: story", G_CODE: "Moves: code", G_BOX: "Box puzzle",
               G_YOU: "I and you", G_RULE: "New rules"}
COLS: list[Col] = [
    Col("lookup", G_STILL, "Simple lookup", "Several people each keep an object; name one person's object.",
        "bp-profile", bp("attr"), headline=True, old=("attr",)),
    Col("shortlist", G_STILL, "Short list lookup", "A list pairs one to four names with tasks; name one person's task.",
        "capacity-v1", lambda r: int(r["K"]) <= 4 and int(r["D"]) == 0, conds=(COND_FLOOR,)),
    Col("pairs", G_STILL, "Lookup among pairs", "Some people hold two objects and some one; name what a one-object "
        "person holds.", "bp-v3.2", bp("setbind", lambda r: r["cell"][0] == 1 and r["cell"][2] == 0), credit=k_set,
        old=("sb-k1",)),
    Col("noise", G_STILL, "Lookup past noise", "Boxes are filled, unrelated events follow; name what one box holds.",
        "formupdate-v3", lambda r: r["condition"] == "A", conds=(COND_FLOOR,)),
    Col("now", G_STILL, "Who holds it now", "A scene says who holds what; nothing moves. Then: 'Now X is holding the'.",
        "joinlite-v1", lambda r: r["qkind"] == "third" and r["arm"] == "orig", conds=(COND_FLOOR,),
        flag="The query opens with 'Now', which implies a change: the largest models answer as if something moved."),
    Col("handoffs", G_STORY, "Hand-offs in a story", "Objects pass between people in a short story; name who has an "
        "object at the end.", "bp-profile", bp("dialogue"), headline=True, old=("dialogue",)),
    Col("sets", G_STORY, "Sets after hand-offs", "Each person starts with a set of objects, then two gives; name the "
        "set one person ends with.", "bp-v3.2", bp("setbind", lambda r: r["cell"][2] == 2), credit=k_set,
        headline=True, old=("sb-dyn",)),
    Col("refilled", G_STORY, "Refilled box", "A box is emptied and refilled; name its new contents.",
        "formupdate-v3", lambda r: r["condition"] == "C", conds=(COND_FLOOR,)),
    Col("refills", G_STORY, "Many refills", "Boxes are emptied and refilled again and again; name one box's latest "
        "contents.", "fluid-v2-rebind", lambda r: int(r["R"]) >= 1, conds=(COND_FLOOR,), headline=True),
    Col("codevar", G_CODE, "Variable chains", "Python variables are assigned, copied and reassigned; name what one "
        "prints.", "bp-profile", bp("codevar"), headline=True, old=("codevar",)),
    Col("sharedlists", G_CODE, "Shared lists", "Two names point at one Python list that is changed, rebound or "
        "copied; name what a print shows.", "bp-v3.2", bp("codevarmut"), headline=True,
        old=("cm-mutate", "cm-rebind", "cm-copy")),
    *[Col(f"box-{reg}", G_BOX, f"Box puzzle: {lab}", f"Boxes are swapped, moved and refilled, written as {how}; name "
          "one box's contents.", "riders", (lambda reg: lambda r: r["renderer"] == reg)(reg))
      for reg, lab, how in (("narrative", "prose", "plain sentences"), ("screenplay", "script", "a two-speaker script"),
                            ("code", "code", "a Python dictionary"), ("chat", "chat", "chat turns"))],
    Col("you-static", G_YOU, "'You', nothing moves", "A narrator says who holds what; then a speaker says 'right now "
        "you're holding the'. Name the listener's object.", "joinlite-v1",
        lambda r: r["qkind"] == "addressee" and r["arm"] == "orig", conds=(COND_FLOOR,), headline=True),
    Col("i-moves", G_YOU, "'I', after hand-offs", "Two people hand objects over only through I / my / you lines; "
        "a speaker says 'right now I'm holding the'.", "bp-deixis-v2", dx("self", 2), old=("dx-self-2-deictic",)),
    Col("you-moves", G_YOU, "'You', after hand-offs", "Two people hand objects over only through I / my / you lines; "
        "a speaker says 'right now you're holding the'. Name the listener's object.", "bp-deixis-v2",
        dx("addressee", 2), headline=True, old=("dx-addressee-2-deictic",)),
    Col("you-named", G_YOU, "'You' among three", "Three people talk and hand objects over; the line naming the "
        "listener asks what they are holding.", "bp-deixis-v2", dx("addressee", 3), old=("dx-addressee-3-deictic",)),
    Col("answer-self", G_YOU, "Answering as yourself", "A speaker must answer with 'me', 'you' or a name, from where "
        "they stand in the conversation.", "bp-v3.2", bp("answer"),
        old=("an-other-any", "an-me-any", "an-you-voc", "an-you-novoc")),
    Col("grammar", G_RULE, "Made-up grammar", "Examples show an invented word ending; apply it to a new invented word.",
        "fluid-v1-morph", lambda r: True, conds=("ladder",), kind="auc"),
    Col("operators", G_RULE, "Made-up operators", "Example equations define invented symbols; solve a new equation.",
        "fluid-v2-algebra", lambda r: True, conds=("ladder",), kind="auc"),
    Col("patterns", G_RULE, "Pattern completion", "Example lines repeat or mirror a pattern of nonsense words; "
        "continue a new line. A copy rule alone scores .26 here.", "fluid-v1-seqind", lambda r: True,
        conds=("ladder",), kind="auc", headline=True),
    Col("facts", G_RULE, "Known facts (control)", "Real facts in the same examples-then-question layout; this column "
        "separates the layout from the learning.", "cryst-v1", lambda r: int(r.get("shots", r.get("rung", 1))) > 0,
        conds=("ladder",), kind="auc"),
]


# ---- cell statistics ------------------------------------------------------------------------------------------------
def cell_stats(items: list[tuple[float, float]]) -> dict[str, Any]:
    cr = np.array([c for c, _ in items])
    ch = np.array([h for _, h in items])
    k = (cr - ch) / (1 - ch)
    rng = np.random.default_rng(0)
    boots = np.sort(rng.choice(k, size=(N_BOOT, len(k)), replace=True).mean(1))
    lo, hi = float(boots[int(.025 * N_BOOT)]), float(boots[int(.975 * N_BOOT) - 1])
    state = "above" if lo > 0 else "below" if hi < 0 else "at"
    return {"acc": round(float(cr.mean()), 4), "n": int(len(k)), "chance": round(float(ch.mean()), 4),
            "score": round(float(k.mean()), 4), "se": round(float(k.std(ddof=1) / math.sqrt(len(k))), 4),
            "lo": round(lo, 4), "hi": round(hi, 4), "state": state}


def ladder_auc(rows: list[Row], rung_key: str = "rung") -> float:
    by: defaultdict[Any, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        by[r[rung_key]].append(k_choice(r))
    rs = []
    for xs in by.values():
        acc = np.mean([c for c, _ in xs])
        ch = np.mean([h for _, h in xs])
        rs.append(min(1.0, max(0.0, (acc - ch) / (1 - ch))))
    return float(np.mean(rs))


def auc_stats(rows: list[Row]) -> dict[str, Any]:
    rung_key = "rung" if "rung" in rows[0] else "shots"
    by: defaultdict[Any, list[Row]] = defaultdict(list)
    for r in rows:
        by[r[rung_key]].append(r)
    point = ladder_auc(rows, rung_key)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        sample = [rr for xs in by.values() for rr in (xs[i] for i in rng.integers(0, len(xs), len(xs)))]
        boots.append(ladder_auc(sample, rung_key))
    boots.sort()
    lo, hi = boots[5], boots[194]
    return {"score": round(point, 4), "n": len(rows), "lo": round(lo, 4), "hi": round(hi, 4),
            "state": "above" if lo > 0 else "at", "acc": None, "chance": None, "se": None}


def read(p: Path) -> list[Row]:
    return [json.loads(line) for line in p.open()] if p.exists() else []


def model_cells(zoo: Path, name: str) -> dict[str, dict[str, Any]]:
    cells: dict[str, dict[str, Any]] = {}
    cache: dict[str, list[Row]] = {}
    for c in COLS:
        src = zoo / "riders" / f"rows_{name}.jsonl" if c.battery == "riders" else zoo / "suite" / c.battery / f"{name}.jsonl"
        if str(src) not in cache:
            cache[str(src)] = read(src)
        rows = [r for r in cache[str(src)] if c.keep(r)]
        if not rows:
            continue
        if c.kind == "auc":
            cells[f"{name}|{c.id}|ladder|all"] = auc_stats(rows)
            continue
        for cond in c.conds:
            xs = [r for r in rows if r.get("condition", COND_FLOOR) == cond] if c.battery.startswith(("bp-", "riders")) \
                else rows
            if xs:
                gold_key = "gold_orig" if "gold_orig" in xs[0] else "gold"
                credit = (lambda r: c.credit(r)) if c.credit is k_set else (lambda r: k_choice(r, gold_key))
                cells[f"{name}|{c.id}|{cond}|all"] = cell_stats([credit(r) for r in xs])
    return cells


# ---- examples ------------------------------------------------------------------------------------------------------
def examples(prompts: Path) -> dict[str, dict[str, Any]]:
    from binding_bench.bindprobes.grid import build_payload_item
    from binding_bench.bindprobes.items import load_items

    P = prompts
    bp_files = {"bp-profile": P / "bindprobes-items" / "items.jsonl", "bp-v3.2": P / "bindprobes-v3.2" / "items.jsonl",
                "bp-deixis-v2": P / "bindprobes-deixis-v2-full" / "items.jsonl"}
    out: dict[str, dict[str, Any]] = {}
    item_cache: dict[str, list[Any]] = {}
    for c in COLS:
        if c.battery in bp_files:
            if c.battery not in item_cache:
                item_cache[c.battery] = load_items(bp_files[c.battery])
            for it in item_cache[c.battery]:
                r = {"family": it.family, "cell": list(it.cell), "qkind": it.meta.get("qkind", it.cell[-1]),
                     "variant": it.meta.get("variant", "deictic")}
                if c.keep(r):
                    pay = build_payload_item(it, "0shot")
                    out[c.id] = {"prompt": pay["prompt"], "gold": it.gold, "candidates": [x.strip() for x in pay["candidates"]],
                                 "gold_set": list(getattr(it, "gold_set", []) or []) or None, "item_id": it.item_id}
                    break
        elif c.battery == "riders":
            reg = c.id.split("-", 1)[1]
            job = next(j for j in rider_jobs() if j.renderer == reg)
            out[c.id] = {"prompt": job.prompt, "gold": rider_gold(job), "candidates": [x.strip() for x in job.candidates],
                         "gold_set": None, "item_id": job.id}
        else:
            f = P / f"{c.battery}.jsonl"
            for line in f.open():
                r = json.loads(line)
                if c.keep(r):
                    out[c.id] = {"prompt": r["prompt"], "gold": r.get("gold_orig", r.get("gold")),
                                 "candidates": [x.strip() for x in r["candidates"]], "gold_set": None,
                                 "item_id": r["id"]}
                    break
    return out


_RIDERS: list[Any] = []


def rider_jobs() -> list[Any]:
    """riders-lite's 0-shot jobs, as the bench builds them (the items every model was scored on)."""
    if not _RIDERS:
        from binding_bench.riders import items as ri
        from binding_bench.riders import lite

        all_items, shots = ri.generate(list(ri.BANKED_VERIFIED_NAMES), 200, 200, ri.DEFAULT_SEED)
        sel = lite.select_items(all_items, 200, 0)
        _RIDERS.extend(lite.build_jobs(sel, shots[:ri.N_SHOTS], None, lite.REGISTERS, ("0shot",)))
        _RIDERS.append(sel)
    return _RIDERS[:-1]


def rider_gold(job: Any) -> str:
    return _RIDERS[-1][job.item_index].gold


# ---- the dropped diff ------------------------------------------------------------------------------------------------
def dropped(old: dict[str, Any], kept_rows: set[str], out: Path) -> int:
    new_of = {o: c.id for c in COLS for o in c.old}
    lab = {r["id"]: r["label"] for r in old["rows"]}
    clab = {c["id"]: c["label"] for c in old["cols"]}
    lines = ["row\trow_label\tcol\tcol_label\tcondition\tmode\tscore\tacc\tn\tstatus"]
    for key, cell in sorted(old["cells"].items()):
        rid, cid, cond, mode = key.split("|")
        if rid == "human":
            continue
        status = ("row not yet on the clean bench" if rid not in kept_rows else
                  f"column replaced by {new_of[cid]}" if cid in new_of else "column not in the new grid")
        lines.append("\t".join(str(x) for x in (rid, lab.get(rid, ""), cid, clab.get(cid, ""), cond, mode,
                                                 cell.get("score"), cell.get("acc"), cell.get("n"), status)))
    out.write_text("\n".join(lines) + "\n")
    return len(lines) - 1


# ---- main ------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", type=Path, required=True)
    ap.add_argument("--bench", type=Path, required=True, help="a binding-bench checkout (its src/ is imported)")
    ap.add_argument("--prompts", type=Path, required=True, help="the battery prompt files (the bench's prompts dir)")
    ap.add_argument("--old", type=Path, help="the previous grid's atlas.json: its human row, and (with --dropped) its cells")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dropped", type=Path, help="write the one-time diff of the previous grid's cells here")
    a = ap.parse_args()
    sys.path.insert(0, str(a.bench / "src"))
    old = json.loads((a.old or a.out).read_text())
    report = json.loads((a.zoo / "report" / "report.json").read_text())["models"]
    table = [ln.split("\t")[0] for ln in (a.zoo / "zoo_tierA.tsv").read_text().splitlines() if ln and not ln.startswith("#")]
    names = [n for n in table if (a.zoo / "suite" / "bp-deixis-v2" / f"{n}.jsonl").exists()]
    tb = a.zoo / "zoo_tierB.tsv"
    vllm = [ln.split("\t")[0] for ln in (tb.read_text().splitlines() if tb.exists() else []) if ln and not ln.startswith("#")]
    vllm = [n for n in vllm if n not in names and (a.zoo / "tierB" / "suite" / "bp-deixis-v2" / f"{n}.jsonl").exists()]
    names += vllm
    missing = [n for n in names if n not in LABELS or n not in TOKENS]
    if missing:
        raise SystemExit(f"no label or token entry for {missing}")

    rows: list[dict[str, Any]] = []
    human = next(r for r in old["rows"] if r["id"] == "human")
    rows.append({k: v for k, v in human.items() if k != "sources"} | {"sources": []})
    cells: dict[str, dict[str, Any]] = {}
    new_of = {o: c.id for c in COLS for o in c.old}
    for key, cell in old["cells"].items():
        rid, cid, cond, mode = key.split("|")
        if rid == "human" and (cid in new_of or cid in {c.id for c in COLS}):   # an old id, or already a new one
            cells[f"human|{new_of.get(cid, cid)}|{cond}|{mode}"] = cell
    for n in names:
        lab, fam = LABELS[n]
        p = CREATOR_SIZE.get(n) or report[n]["params"]
        params = float(p) if p else 1.425e9
        dense = report[n].get("active") in (None, report[n]["params"])
        active = params if dense else float(report[n]["active"])
        tok, mark = TOKENS[n]
        traj = n.startswith("tsumugi-")
        rows.append({"id": n, "label": lab, "group": TRAJECTORY if traj else band(active), "family": fam,
                     "size": round(params / 1e9, 3), "active": round(active / 1e9, 3), "tokens": tok,
                     "tokens_mark": mark, "sources": [f"zoo-sweep/{n}"]} | ({"note": VLLM_NOTE} if n in vllm else {}))
        cells.update(model_cells(a.zoo / "tierB" if n in vllm else a.zoo, n))
        print(f"{n}: {sum(1 for k in cells if k.startswith(n + '|'))} cells", flush=True)

    order = {lab: i for i, (_, lab) in enumerate(BANDS)}
    rows = rows[:1] + sorted(rows[1:], key=lambda r: (r["group"] == TRAJECTORY, order.get(r["group"], 0),
                                                       r["tokens"] if r["group"] == TRAJECTORY else r["active"], r["label"]))
    cols = [{"id": c.id, "group": c.group, "group_short": GROUP_SHORT[c.group], "label": c.label, "what": c.what, "kind": c.kind, "conds": list(c.conds),
             "headline": c.headline, "flag": c.flag, "free_ok": False} for c in COLS]
    atlas = {"rows": rows, "cols": cols, "cells": cells, "examples": examples(a.prompts)}
    a.out.write_text(json.dumps(atlas, ensure_ascii=False, separators=(",", ":")))
    msg = f"atlas: {len(rows)} rows, {len(cols)} columns, {len(cells)} cells -> {a.out}"
    if a.dropped:
        msg += f"; dropped {dropped(old, set(names), a.dropped)} old cells -> {a.dropped}"
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
