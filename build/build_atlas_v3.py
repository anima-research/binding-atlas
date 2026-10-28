"""Binding atlas: every model x every binding-eval cell, chance-normalized, with a verbatim example per column.

Usage: python build/build_atlas_v3.py OUT.json --grids DIR      (or ATLAS_GRIDS=DIR)
Reads the raw score grids under DIR, laid out as data/INPUTS-MANIFEST.tsv lists them (build/link_grids.py assembles
that layout from local copies): lineage deixis-v2 grids, hybrid profile grids, battery v3.1 grids, the vLLM fleet
grids, the bindprobes suite grids, the Claude-family frame run and the API gauntlet; plus the human gauntlet items
tracked in data/inputs/. Writes {rows, cols, cells, examples} for the atlas page.
Cell = (model, col, condition, mode) -> {acc, n, chance, score}; score = (acc - chance) / (1 - chance).
mode "all" = every item; "free" = shortcut-free items (gold is neither the last-mentioned candidate nor the
last speaker) — deixis columns only. Every multi-checkpoint grid set is fingerprinted (distinct models).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from kotodama.eval.bindprobes.items import build_prompt, load_items
from kotodama.eval.bindprobes.setbind import set_metrics

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from dx_replay import classify as dx_classify  # noqa: E402

_ap = argparse.ArgumentParser(description="Build data/built/atlas.json from the raw score grids.")
_ap.add_argument("out", type=Path, help="output atlas JSON")
_ap.add_argument("--grids", type=Path, default=os.environ.get("ATLAS_GRIDS"),
                 help="root of the raw grids (logical layout of data/INPUTS-MANIFEST.tsv); default $ATLAS_GRIDS")
_args = _ap.parse_args()
if _args.grids is None or not Path(_args.grids).is_dir():
    raise SystemExit(f"grids root not found: {_args.grids!r} (pass --grids DIR or set ATLAS_GRIDS)")
G = Path(_args.grids)                                   # path strings stay unresolved: sources = parent/name
INPUTS = REPO / "data" / "inputs"                       # small inputs tracked in this repo (the human gauntlet)
SRC = G / "binding-atlas-2026-10-15/src"
V3 = G / "battery-v3-2026-10-14/grids-v3.1"
V3_ITEMS = G / "battery-v3-2026-10-14/battery-v3.1/items.jsonl"
CLAUDE = G / "claude-deixis-2026-10-14"
OR = G / "openrouter-gauntlet-2026-10-15"
CONDS = ("0shot", "3shot", "3shot_collide")

# ── columns ──────────────────────────────────────────────────────────────────
Row = dict[str, Any]
COLS: list[dict[str, Any]] = []


def col(cid: str, group: str, label: str, what: str, keep: Callable[[Row], bool], kind: str = "acc",
        free_ok: bool = False, example: Callable[[Any], bool] | None = None, items: str = "") -> None:
    COLS.append({"id": cid, "group": group, "label": label, "what": what, "keep": keep, "kind": kind,
                 "free_ok": free_ok, "example": example, "items": items})


def dx(q: str, p: int | None, v: str) -> Callable[[Row], bool]:
    def k(r: Row) -> bool:
        if r.get("family") != "deixis" or r.get("variant") != v:
            return False
        P, E, Q = r["cell_key"].split("/")
        if Q != q or (p is not None and int(P) != p):
            return False
        return not (q == "owner" and int(E) == 3)          # owner = E<=2 everywhere (= the lite subset)
    return k


def dx_ex(q: str, p: int | None, v: str) -> Callable[[Any], bool]:
    return lambda it: (it.family == "deixis" and it.meta.get("variant") == v and it.cell[2] == q
                       and (p is None or it.cell[0] == p) and it.cell[1] == 2)


for fam, label, what in (
        ("attr", "Static lookup", "N people each keep an object; query one. Pure retrieval under distractors."),
        ("codevar", "Code variables", "Python assignments, copies, swaps; read a variable after reassignment."),
        ("dialogue", "3rd-person updates", "Prose scene; objects change hands, partly via he/she/it; who has what.")):
    col(fam, "Core binding", label, what, lambda r, f=fam: r.get("family") == f,
        example=lambda it, f=fam: it.family == f and (it.cell[0] == 4 if f == "attr" else
                                                       it.cell[0] == 2 if f == "codevar" else it.cell[:2] == [3, 2]),
        items="core")

for q, p, lab in (("owner", 3, "owner"), ("self", 3, "self · P3"), ("self", 2, "self · P2"),
                  ("addressee", 3, "you · P3 (vocative)"), ("addressee", 2, "you · P2 (structural)")):
    for v, vl in (("deictic", "I/you/my"), ("third", "3rd-person twin")):
        what = {
            "owner": "Ownership stated only via I/my/you + vocatives; query 'Now the X is with' -> a name.",
            "self": "Query 'Ana: Right now I'm holding the' -> the speaker's own object.",
            "addressee": "Query 'Ben: Ana, right now you're holding the' (P3 names 'you'; P2 = the only other person).",
        }[q] if v == "deictic" else "The same events narrated with names, no pronouns (the control).";
        col(f"dx-{q}-{p}-{v}", f"Deixis · {lab}", vl, what, dx(q, p, v), free_ok=True,
            example=dx_ex(q, p, v), items="dx")

col("sb-k1", "Set binding", "1 object", "Static, one object per queried person (baseline).",
    lambda r: r.get("family") == "setbind" and r["k"] == 1 and r["n_transfers"] == 0, kind="hit1",
    example=lambda it: it.family == "setbind" and it.cell == [1, "joint", 0], items="v3")
col("sb-k23", "Set binding", "2-3 objects, exact set", "Person keeps 2-3 objects (one sentence). Score: top-k == the set.",
    lambda r: r.get("family") == "setbind" and r["k"] > 1 and r["n_transfers"] == 0 and r["layout"] == "joint",
    kind="exactk", example=lambda it: it.family == "setbind" and it.cell == [2, "joint", 0], items="v3")
col("sb-split", "Set binding", "split across sentences", "Same, but each object declared in a separate sentence.",
    lambda r: r.get("family") == "setbind" and r["k"] > 1 and r["n_transfers"] == 0 and r["layout"] == "split",
    kind="exactk", example=lambda it: it.family == "setbind" and it.cell == [3, "split", 0], items="v3")
col("sb-dyn", "Set binding", "after transfers", "Sets change through two gives; read the final set.",
    lambda r: r.get("family") == "setbind" and r["k"] > 1 and r["n_transfers"] == 2,
    kind="exactk", example=lambda it: it.family == "setbind" and it.cell == [2, "joint", 2], items="v3")

col("an-narr", "Answer deixis", "narrator query", "Same chat, narrator asks 'Now the X is with' -> a name (comprehension).",
    lambda r: r.get("family") == "answer" and r["variant"] == "narr",
    example=lambda it: it.family == "answer" and it.meta.get("variant") == "narr" and it.meta["role"] == "me", items="v3")
for role, ch, lab, what in (("other", None, "answer: a name", "'Dee: Who has the cup now?' / 'Ben: I think it's with' -> another person's name."),
                            ("me", None, "answer: me", "The answerer holds it -> ' me' (speak from your own seat)."),
                            ("you", "voc", "answer: you (named)", "The asker holds it, answer opens with the asker's name -> ' you'."),
                            ("you", "novoc", "answer: you (no name)", "The asker holds it, no name in the answer -> ' you' (reply structure only).")):
    col(f"an-{role}-{ch or 'any'}", "Answer deixis", lab, what + " Calibrated read (vs body-hidden null).",
        lambda r, ro=role, c=ch: r.get("family") == "answer" and r["variant"] == "qa" and r["role"] == ro
        and (c is None or r["channel"] == c), kind="cal",
        example=lambda it, ro=role, c=ch: it.family == "answer" and it.meta.get("variant") == "qa"
        and it.meta["role"] == ro and (c is None or it.meta["channel"] == c), items="v3")

for v, lab, what in (("adj", "'yours' by reply", "'Ben: Can I have the cup?' / 'Ana: Sure, the cup is yours.' -> Ben."),
                     ("voc", "'yours' with the name", "Same, grant names him: 'Sure, Ben, the cup is yours.'"),
                     ("third", "3rd-person twin", "'Ben asks Ana for the cup. Ana gives the cup to Ben.'")):
    col(f"adj-{v}", "Adjacency", lab, what + " Critical items (object last moved by a grant).",
        lambda r, vv=v: r.get("family") == "adjacency" and r["variant"] == vv and r["critical"],
        example=lambda it, vv=v: it.family == "adjacency" and it.meta.get("variant") == vv and it.meta["critical"]
        and it.cell == [1, 1], items="v3")

for t, lab, what in (("mutate", "mutate via alias", "b = a; a.append('x'); print(b[-1]) -> the appended value."),
                     ("rebind", "rebind after alias", "b = a; a = ['new']; print(b[-1]) -> the OLD list's value."),
                     ("copy", "copy then mutate", "b = list(a); a.append('x'); print(b[-1]) -> unaffected.")):
    col(f"cm-{t}", "Aliased lists", lab, what, lambda r, tt=t: r.get("family") == "codevarmut" and r["trap"] == tt,
        example=lambda it, tt=t: it.family == "codevarmut" and it.cell == [1, tt], items="v3")


# ── rows (models) ────────────────────────────────────────────────────────────
def lineage(name: str) -> list[str]:
    v3 = V3 / f"{name}.jsonl"
    if not v3.exists():
        v3 = SRC / "v3-gap" / f"{name}.jsonl"                 # 2026-10-15 gap fill B (post-trained + bases)
    return [str(p) for p in [SRC / "lineage" / f"{name}.jsonl", v3, SRC / "core-lineage" / f"{name}.jsonl"]
            if p.exists()]


ROWS: list[dict[str, Any]] = [
    {"id": "human", "label": "Human reader (gauntlet)", "group": "Human", "size": None, "files": [],
     "note": "reflex, read cold: round 1 (31 items, 30 right) + round 2 (24 items, 22 right; both misses self-reported reading slips)"},
]
for m, lab, fr in (("claude-haiku-4-5-20251001", "Haiku 4.5", "A"), ("claude-sonnet-5-5", "Sonnet 5.5", "A"),
                   ("claude-haiku-4-5-20251001", "Haiku 4.5", "B"), ("claude-sonnet-5-5", "Sonnet 5.5", "B"),
                   ("claude-opus-5-5", "Opus 5.5", "B"), ("claude-fable-5", "Fable 5", "B")):
    ROWS.append({"id": f"claude-{lab}-{fr}", "label": f"{lab} · {'reflex' if fr == 'A' else 'deliberate'}",
                 "group": "Claude (API)", "size": None, "files": [], "claude": (m, fr),
                 "note": "deixis sample only (seed 2027, 324 items, 0-shot)"})
for name, lab, size, grp in (
        ("olmo2-1b", "OLMo-2 1B", 1.5, "Pretrained bases"), ("pythia-6.9b", "Pythia 6.9B", 6.9, "Pretrained bases"),
        ("olmo2-7b-base", "OLMo-2 7B", 7, "Pretrained bases"), ("llama31-8b-base", "Llama-3.1 8B", 8, "Pretrained bases"),
        ("llama31-8b", "Llama-3.1 8B", 8, "Pretrained bases"),
        ("gemma2-9b", "Gemma-2 9B", 9, "Pretrained bases"), ("olmo2-13b", "OLMo-2 13B", 13, "Pretrained bases"),
        ("qwen3-30b-a3b-base", "Qwen3 30B-A3B", 30, "Pretrained bases"), ("qwen3-30b-a3b", "Qwen3 30B-A3B", 30, "Pretrained bases"),
        ("olmo3-32b", "OLMo-3 32B", 32, "Pretrained bases"),
        ("llama31-70b", "Llama-3.1 70B", 70, "Pretrained bases"), ("llama31-405b", "Llama-3.1 405B", 405, "Pretrained bases"),
        ("olmo2-7b-sft", "OLMo-2 7B SFT", 7, "Post-trained"), ("olmo2-7b-dpo", "OLMo-2 7B DPO", 7, "Post-trained"),
        ("olmo2-7b-instruct", "OLMo-2 7B Instruct", 7, "Post-trained"), ("llama31-8b-instruct", "Llama-3.1 8B Instruct", 8, "Post-trained"),
        ("falcon3-7b", "Falcon3 7B", 7, "Pretrained bases"), ("falcon-h1-0.5b", "Falcon-H1 0.5B", 0.5, "Hybrid & SSM bases"),
        ("falcon-h1-1.5b", "Falcon-H1 1.5B (24L)", 1.5, "Hybrid & SSM bases"), ("falcon-h1-1.5b-deep", "Falcon-H1 1.5B-Deep (66L)", 1.5, "Hybrid & SSM bases"),
        ("falcon-h1-3b", "Falcon-H1 3B", 3, "Hybrid & SSM bases"), ("falcon-h1-7b", "Falcon-H1 7B", 7, "Hybrid & SSM bases"),
        ("falcon-h1-34b-base", "Falcon-H1 34B", 34, "Hybrid & SSM bases"), ("falcon-h1-34b", "Falcon-H1 34B", 34, "Hybrid & SSM bases"),
        ("bamba-9b", "Bamba 9B (interleaved)", 9, "Hybrid & SSM bases"), ("falcon-mamba-7b", "Falcon-Mamba 7B (pure SSM)", 7, "Hybrid & SSM bases"),
        ("recurrentgemma-9b", "RecurrentGemma 9B", 9, "Hybrid & SSM bases"),
        ("dsv3-base", "DeepSeek-V3 Base (671B MoE)", 671, "Pretrained bases"),
        ("falcon3-3b", "Falcon3 3B", 3, "Pretrained bases"),
        ("olmo2-7b-pt2928B", "OLMo-2 7B @ 2.9T (stage 1)", 7, "OLMo-2 7B trajectory"),
        ("olmo2-7b-pt3896B", "OLMo-2 7B @ 3.9T (end stage 1)", 7, "OLMo-2 7B trajectory"),
        ("olmo2-7b-mid5B", "OLMo-2 7B mid-train +5B", 7, "OLMo-2 7B trajectory"),
        ("olmo2-7b-mid17B", "OLMo-2 7B mid-train +17B", 7, "OLMo-2 7B trajectory"),
        ("olmo2-7b-mid50B", "OLMo-2 7B mid-train +50B", 7, "OLMo-2 7B trajectory")):
    files = lineage(name) + ([str(SRC / "hybprof" / f"{name}.jsonl")] if (SRC / "hybprof" / f"{name}.jsonl").exists() else [])
    ROWS.append({"id": name, "label": lab, "group": grp, "size": size, "files": files})
# merge the two naming variants (deixis grid name vs v3 grid name) into one row
_merge = {"llama31-8b": "llama31-8b-base", "qwen3-30b-a3b": "qwen3-30b-a3b-base", "falcon-h1-34b": "falcon-h1-34b-base"}
for a, b in _merge.items():
    ra = next(r for r in ROWS if r["id"] == a); rb = next(r for r in ROWS if r["id"] == b)
    rb["files"] = sorted(set(rb["files"] + ra["files"])); ROWS.remove(ra)

# vLLM fleet (records/bindbench-2026-10-16): one grid per model with the full battery (deixis v2 + core + v3.1),
# parity-gated vs HF (SPEC-bindbench-fleet: no bias, |Δacc| ≤ .01). A model's vLLM grid REPLACES its HF files —
# never merged into one row. Sizes marked ≈ are estimated from checkpoint bytes (total params, MoE incl.).
BB = SRC / "bindbench"
FLEET = (
    ("gemma2-9b", "Gemma-2 9B", 9, "Pretrained bases", ""),
    ("qwen3-30b-a3b-base", "Qwen3 30B-A3B", 30, "Pretrained bases", ""),
    ("gemma4-31b-base", "Gemma-4 31B", 31, "Pretrained bases", ""),
    ("glm4-32b-base", "GLM-4 32B", 32, "Pretrained bases", ""),
    ("olmo3-32b", "OLMo-3 32B", 32, "Pretrained bases", ""),
    ("olmo3-32b-stage1", "OLMo-3 32B @ stage 1 (pre-midtrain)", 32, "Pretrained bases", "pre-midtraining checkpoint"),
    ("llama31-70b", "Llama-3.1 70B", 70, "Pretrained bases", ""),
    ("qwen25-72b", "Qwen2.5 72B", 72, "Pretrained bases", ""),
    ("dsv4-flash-base", "DeepSeek-V4 Flash Base", 290, "Pretrained bases", "≈290B total params (MoE)"),
    ("mimo-v25-base", "MiMo-V2.5 Base", 310, "Pretrained bases", "≈310B total params (MoE)"),
    ("trinity-truebase", "Trinity-Large TrueBase", 400, "Pretrained bases", "≈400B MoE"),
    ("trinity-base", "Trinity-Large Base", 400, "Pretrained bases", "≈400B MoE"),
    ("llama31-405b", "Llama-3.1 405B", 405, "Pretrained bases", ""),
    ("dsv3-base", "DeepSeek-V3 Base (671B MoE)", 671, "Pretrained bases", ""),
    ("dsv4-pro-base", "DeepSeek-V4 Pro Base", 860, "Pretrained bases", "≈860B total params (MoE); scored from an fp4 copy"),
    ("kimi-k2-base", "Kimi-K2 Base (1T MoE)", 1000, "Pretrained bases", ""),
    ("mimo-v25-pro-base", "MiMo-V2.5-Pro Base", 1000, "Pretrained bases", "≈1T total params (MoE)"),
    ("nemotron3-ultra-550b-base", "Nemotron-3 Ultra 550B Base (hybrid)", 550, "Hybrid & SSM bases", "Mamba–attention hybrid"),
)


def fleet_files(name: str) -> tuple[list[str], str]:
    """(files, engine) for a fleet model: the vLLM grid if present, else the HF fallback grids (e.g. OLMo-3 stage 1)."""
    v = BB / f"{name}.jsonl"
    if v.exists():
        return [str(v)], "vLLM"
    hf = [BB / f"hf-{name}-{k}.jsonl" for k in ("dx", "core", "v3")]   # exact names: 'olmo3-32b' must not match stage1
    return [str(p) for p in hf if p.exists()], "HF"


for name, lab, size, grp, note in FLEET:
    files, eng = fleet_files(name)
    if not files:
        continue
    eng_note = ("scored with vLLM (parity-checked against HF)" if eng == "vLLM"
                else "scored with HF transformers (vLLM could not load this checkpoint)")
    full_note = "; ".join(x for x in (note, eng_note) if x)
    old = next((r for r in ROWS if r["id"] == name), None)
    if old is not None:                       # replace the HF row's grids, keep its place
        old.update(files=files, label=lab, size=size, group=grp, note=full_note)
    else:
        ROWS.append({"id": name, "label": lab, "group": grp, "size": size, "files": files, "note": full_note})

# The 1.42B ablation rows are kept out of the public grid: the page reports the levers
# in Table 1 and the kotodama-tsumugi trajectory in Part III. Their grids are not part of this build's inputs.
# Small and mid public bases + the kotodama-tsumugi release, scored with the public suite
# (`python -m kotodama.eval.binding run --batteries bindprobes`; parity-gated against the grids above). deixis-lite is a
# subset of deixis v2, so a row reads deixis v2 + profile + v3.1 only. Replaces the Pythia 1.4B / 2.8B pilot rows.
BP = G / "bindprobes"
GRID = (("tsumugi-release", "kotodama-tsumugi 1.4B (release)", 1.4),
        ("pythia-160m", "Pythia 160M", 0.16), ("pythia-410m", "Pythia 410M", 0.41), ("pythia-1b", "Pythia 1B", 1),
        ("pythia-1.4b", "Pythia 1.4B", 1.4), ("pythia-2.8b", "Pythia 2.8B", 2.8),
        ("smollm2-135m", "SmolLM2 135M", 0.135), ("smollm2-360m", "SmolLM2 360M", 0.36), ("smollm2-1.7b", "SmolLM2 1.7B", 1.7),
        ("qwen2.5-0.5b", "Qwen2.5 0.5B", 0.5), ("qwen2.5-1.5b", "Qwen2.5 1.5B", 1.5), ("qwen2.5-3b", "Qwen2.5 3B", 3),
        ("qwen2.5-7b", "Qwen2.5 7B", 7), ("qwen3-0.6b-base", "Qwen3 0.6B", 0.6), ("qwen3-1.7b-base", "Qwen3 1.7B", 1.7),
        ("qwen3-4b-base", "Qwen3 4B", 4), ("llama3.2-1b", "Llama-3.2 1B", 1), ("olmo1-1b-0724", "OLMo-1 1B (0724)", 1))
for name, lab, size in GRID:
    files = [BP / d / f"{name}.jsonl" for d in ("bp-deixis-v2", "bp-profile", "bp-v3.1")]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"grid row {name}: missing {missing} (see data/INPUTS-MANIFEST.tsv)")
    ROWS.append({"id": name, "label": lab, "group": "Pretrained bases", "size": size, "files": [str(f) for f in files]})

API_META = {
    "meta-llama/llama-3.2-1b-instruct": {"label": "Llama-3.2 1B Instruct", "size": 1, "open": True},
    "meta-llama/llama-3.2-3b-instruct": {"label": "Llama-3.2 3B Instruct", "size": 3, "open": True},
    "meta-llama/llama-3.1-8b-instruct": {"label": "Llama-3.1 8B Instruct", "size": 8, "open": True},
    "meta-llama/llama-3.3-70b-instruct": {"label": "Llama-3.3 70B Instruct", "size": 70, "open": True},
    "nousresearch/hermes-3-llama-3.1-405b": {"label": "Hermes-3 405B", "size": 405, "open": True},
    "meta-llama/llama-4-maverick": {"label": "Llama-4 Maverick (400B MoE)", "size": 400, "open": True},
    "google/gemma-3-4b-it": {"label": "Gemma-3 4B IT", "size": 4, "open": True},
    "google/gemma-3-12b-it": {"label": "Gemma-3 12B IT", "size": 12, "open": True},
    "google/gemma-3-27b-it": {"label": "Gemma-3 27B IT", "size": 27, "open": True},
    "qwen/qwen-2.5-7b-instruct": {"label": "Qwen2.5 7B Instruct", "size": 7, "open": True},
    "qwen/qwen-2.5-72b-instruct": {"label": "Qwen2.5 72B Instruct", "size": 72, "open": True},
    "qwen/qwen3-14b": {"label": "Qwen3 14B", "size": 14, "open": True},
    "qwen/qwen3-32b": {"label": "Qwen3 32B", "size": 32, "open": True},
    "qwen/qwen3-235b-a22b-2507": {"label": "Qwen3 235B-A22B", "size": 235, "open": True},
    "qwen/qwen3.5-397b-a17b": {"label": "Qwen3.5 397B-A17B", "size": 397, "open": True},
    "mistralai/ministral-3b-2512": {"label": "Ministral 3B", "size": 3, "open": True},
    "mistralai/ministral-8b-2512": {"label": "Ministral 8B", "size": 8, "open": True},
    "mistralai/mistral-small-3.2-24b-instruct": {"label": "Mistral Small 3.2 24B", "size": 24, "open": True},
    "mistralai/mistral-large-2512": {"label": "Mistral Large (2512)", "size": None, "open": True},
    "microsoft/phi-4": {"label": "Phi-4 14B", "size": 14, "open": True},
    "openai/gpt-oss-20b": {"label": "gpt-oss 20B", "size": 20, "open": True},
    "openai/gpt-oss-120b": {"label": "gpt-oss 120B", "size": 120, "open": True},
    "deepseek/deepseek-v3.2": {"label": "DeepSeek V3.2 (671B MoE)", "size": 671, "open": True},
    "moonshotai/kimi-k2-0905": {"label": "Kimi K2 (1T MoE)", "size": 1000, "open": True},
    "z-ai/glm-4.6": {"label": "GLM-4.6 (355B MoE)", "size": 355, "open": True},
    "amazon/nova-pro-v1": {"label": "Amazon Nova Pro", "size": None, "open": False},
    "openai/gpt-4o-mini": {"label": "GPT-4o mini", "size": None, "open": False},
    "openai/gpt-4.1-nano": {"label": "GPT-4.1 nano", "size": None, "open": False},
    "openai/gpt-4.1-mini": {"label": "GPT-4.1 mini", "size": None, "open": False},
    "openai/gpt-4.1": {"label": "GPT-4.1", "size": None, "open": False},
    "openai/gpt-5-nano": {"label": "GPT-5 nano", "size": None, "open": False},
    "openai/gpt-5-mini": {"label": "GPT-5 mini", "size": None, "open": False},
    "openai/gpt-5.5": {"label": "GPT-5.5", "size": None, "open": False},
    "google/gemini-2.5-flash-lite": {"label": "Gemini 2.5 Flash-Lite", "size": None, "open": False},
    "google/gemini-2.5-flash": {"label": "Gemini 2.5 Flash", "size": None, "open": False},
    "google/gemini-3.1-pro-preview": {"label": "Gemini 3.1 Pro (preview)", "size": None, "open": False},
    "x-ai/grok-4.3": {"label": "Grok 4.3", "size": None, "open": False},
    "anthropic/claude-haiku-4.5": {"label": "Claude Haiku 4.5", "size": None, "open": False},
    "anthropic/claude-haiku-5.5": {"label": "Claude Haiku 5.5", "size": None, "open": False},
    "anthropic/claude-sonnet-4.5": {"label": "Claude Sonnet 4.5", "size": None, "open": False},
    "anthropic/claude-sonnet-5.5": {"label": "Claude Sonnet 5.5", "size": None, "open": False},
    "anthropic/claude-opus-4.5": {"label": "Claude Opus 4.5", "size": None, "open": False},
    "anthropic/claude-opus-5.5": {"label": "Claude Opus 5.5", "size": None, "open": False},
    "anthropic/claude-fable-5.1": {"label": "Claude Fable 5.1", "size": None, "open": False},
}


# ── scoring ──────────────────────────────────────────────────────────────────
def load(path: str) -> list[Row]:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    bad = [r for r in rows if r.get("family") == "deixis" and r.get("deixis_version") != 2]
    if bad:   # v1 deixis failed its own validity gates (SPEC-bindprobes-deixis addendum) — never chart it
        raise SystemExit(f"{path}: {len(bad)} deixis rows are not deixis v2")
    return [r for r in rows if not r.get("multi_token_candidates")]


def calibrate(rows: list[Row]) -> None:
    nulls = {(r["pair_id"], r["condition"]): r["cand_lps"] for r in rows
             if r.get("family") == "answer" and r.get("variant") == "null"}
    for r in rows:
        if r.get("family") == "answer" and r.get("variant") in ("qa", "third"):
            z = nulls.get((r["pair_id"], r["condition"]))
            if z:
                cal = {c: lp - z[c] for c, lp in r["cand_lps"].items()}
                r["_cal_ok"] = max(cal, key=cal.get) == r["gold"]


def outcome(r: Row, kind: str) -> tuple[float, float]:
    """(correct 0/1, chance) for one row under the column's scoring."""
    if kind in ("hit1", "exactk"):
        m = set_metrics(r["cand_lps"], r["gold_set"])
        return m[kind], m["chance_hit1" if kind == "hit1" else "chance_exactk"]
    if kind == "cal":
        return float(r.get("_cal_ok", False)), 1.0 / len(r["cand_lps"])
    return float(r["gold_rank_cand"] == 1), 1.0 / len(r["cand_lps"])


def is_free(r: Row) -> bool:
    return not r.get("gold_is_last_mention") and not r.get("gold_is_last_speaker")


def fingerprint(rows: list[Row]) -> str:
    return hashlib.md5(str([round(sum(r["cand_lps"].values()), 4) for r in rows[:300]]).encode()).hexdigest()[:10]


def summarize(xs: list[tuple[float, float]]) -> dict[str, float]:
    n = len(xs)
    acc = sum(x for x, _ in xs) / n
    ch = sum(c for _, c in xs) / n
    return {"acc": round(acc, 4), "n": n, "chance": round(ch, 4),
            "score": round((acc - ch) / (1 - ch), 4) if ch < 1 else 0.0,
            "se": round(math.sqrt(max(acc * (1 - acc), 1e-9) / n), 4)}


def main() -> None:
    out_path = _args.out
    cells: dict[str, dict[str, Any]] = {}
    prints: dict[str, list[str]] = defaultdict(list)
    for row in ROWS:
        if not row["files"]:
            continue
        rows: list[Row] = []
        for f in row["files"]:
            fr = load(f)
            if fr and len({r.get("family") for r in fr}) >= 1:
                prints[fingerprint(fr) + "|" + ",".join(sorted({r.get("family", "?") for r in fr}))].append(f)
            rows += fr
        calibrate(rows)
        for c in COLS:
            for cond in CONDS:
                for mode in ("all", "free"):
                    if mode == "free" and not c["free_ok"]:
                        continue
                    sub = [r for r in rows if r.get("condition") == cond and c["keep"](r) and (mode == "all" or is_free(r))]
                    if len(sub) >= 8:
                        key = f"{row['id']}|{c['id']}|{cond}|{mode}"
                        cells[key] = summarize([outcome(r, c["kind"]) for r in sub])
                        if c["id"].startswith("dx-") and c["id"].endswith("-deictic"):
                            eh = Counter((r.get("error_type") or "correct") if r.get("gold_rank_cand") != 1 else "correct" for r in sub)
                            cells[key]["errs"] = {k: round(v / len(sub), 3) for k, v in eh.items() if k != "correct"}
    dup = [v for v in prints.values() if len(v) > 1]
    if dup:
        raise SystemExit(f"REFUSING: identical-fingerprint grids (same model twice?): {dup}")
    # Claude frame run (0-shot deixis sample; no shortcut metadata)
    citems = [json.loads(l) for l in open(CLAUDE / "items.jsonl")]
    dxmeta = {it.item_id: it.meta for it in load_items(SRC / "items/deixis-v2-full.jsonl")}
    by: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for l in open(CLAUDE / "results.jsonl"):
        r = json.loads(l)
        if "error" in r:
            continue
        it = citems[r["idx"]]
        P, E, Q = it["cell"].split("/")
        if Q == "owner" and E == "3":
            continue
        x = (float(r["ok"]), 1.0 / len(it["candidates"]))
        by[(r["model"], r["frame"], Q, int(P), r["variant"], "all")].append(x)
        mm = dxmeta.get(f"{it['pair_id']}_{it['variant']}")
        if mm is not None and not mm.get("gold_is_last_mention") and not mm.get("gold_is_last_speaker"):
            by[(r["model"], r["frame"], Q, int(P), r["variant"], "free")].append(x)
    or_claude = any("anthropic/" in ln for f in OR.glob("results*.jsonl") for ln in open(f))
    for row in ROWS:
        if "claude" not in row or or_claude:      # superseded by the OpenRouter Claude rows (same items/frames as everyone)
            continue
        m, fr = row["claude"]
        for (mm, ff, q, p, v, mode), xs in by.items():
            if (mm, ff) != (m, fr):
                continue
            cid = f"dx-{q}-{p}-{v}"
            if any(c["id"] == cid for c in COLS) and len(xs) >= 8:
                cells[f"{row['id']}|{cid}|0shot|{mode}"] = summarize(xs)
    # The human gauntlet. Round 1 (31 items; miss = id 7, codevar depth/2/2). Round 2 (24 items, every item tagged with
    # its atlas column; misses = ids 10 and 17; id 9 = the answer family's narrated variant -> no column).
    g1 = json.load(open(INPUTS / "gauntlet1_items.json"))
    human: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for it in g1:
        ok = float(it["id"] != 7)
        ch = 1.0 / len(it["choices"])
        cell = it["cell"]
        if it["cat"] in ("attr", "codevar", "dialogue"):
            human[(it["cat"], "0shot", "all")].append((ok, ch))
        elif it["cat"] == "deixis":
            if "owner" in cell or "TWIN" in cell:
                cid = "dx-owner-3-deictic"
            elif "THIRD" in cell:
                cid = "dx-owner-3-third" if "owner" in cell else "dx-addressee-3-third"
            else:
                P = 3 if "P3" in cell else 2
                cid = f"dx-{'self' if 'self' in cell else 'addressee'}-{P}-deictic"
            human[(cid, "0shot", "all")].append((ok, ch))
        elif it["cat"] == "bonus":
            human[("dialogue", "3shot" if "COLLIDE" not in cell else "3shot_collide", "all")].append((ok, ch))
    g2 = json.load(open(INPUTS / "gauntlet2_items.json"))
    for it in g2:
        if it["id"] == 9:
            continue
        ok = float(it["id"] not in (10, 17))
        k = len(it["gold"]); C = len(it["choices"])
        ch = 1.0 / math.comb(C, k) if it["col"] in ("sb-k23", "sb-split", "sb-dyn") else 1.0 / C
        human[(it["col"], "0shot", "all")].append((ok, ch))
        if it["cat"] == "deixis":                                    # round-2 deixis items are all shortcut-free
            human[(it["col"], "0shot", "free")].append((ok, ch))
    for (cid, cond, mode), xs in human.items():
        cells[f"human|{cid}|{cond}|{mode}"] = summarize(xs)
    # API models in generation mode (openrouter-gauntlet-2026-10-15): reflex (A) / deliberate (B) rows
    orfs = sorted(OR.glob("results*.jsonl"))
    if orfs:
        api: dict[tuple[str, str, str, str], list[tuple[float, float]]] = defaultdict(list)
        rflags: dict[tuple[str, str], list[int]] = defaultdict(list)
        or_items = {json.loads(l)["item_id"]: json.loads(l) for l in open(OR / "items.jsonl")}
        form: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        aerr: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)     # seat-form use among referent-correct answers
        seen_api: set[tuple[str, str, str]] = set()
        for l in (ln for f in orfs for ln in open(f)):
            r = json.loads(l)
            if "error" in r or (r["model"], r["frame"], r["item_id"]) in seen_api:
                continue
            seen_api.add((r["model"], r["frame"], r["item_id"]))
            if r["frame"] == "A" and (r.get("reasoning_tokens") or 0) > 0:
                rflags[(r["model"], r["frame"])].append(1)      # a 'reflex' call that reasoned anyway: excluded
                continue
            ch = 1.0 / r["n_cands"]
            ok = bool(r["ok"])
            if r["col"] in ("an-me-any", "an-you-voc", "an-you-novoc"):
                # REFERENT scoring: the person's name is as valid as me/you — score who, record the form
                pr = or_items[r["item_id"]]["prompt"].rstrip().split("\n")
                who = (pr[-1].split(":")[0] if r["col"] == "an-me-any"
                       else next((l.split(":")[0] for l in reversed(pr) if re.match(r"^\w+: Who has", l)), "")).lower()
                named = r["pick"] == who
                if ok or named:
                    form[(r["model"], r["frame"], r["col"])].append(int(ok))
                ok = ok or named
            api[(r["model"], r["frame"], r["col"], "all")].append((float(ok), ch))
            if r["col"].startswith("dx-") and r["col"].endswith("-deictic"):
                ec = dx_classify(or_items[r["item_id"]]["prompt"], r["pick"], bool(r["ok"]), bool(r["in_cands"]))
                aerr[(r["model"], r["frame"], r["col"], "all")][ec] += 1
                if r.get("free"):
                    aerr[(r["model"], r["frame"], r["col"], "free")][ec] += 1
            if r.get("free"):
                api[(r["model"], r["frame"], r["col"], "free")].append((float(ok), ch))
            rflags[(r["model"], r["frame"])].append(0)
        known = {row["id"] for row in ROWS}
        for (m, fr), rts in sorted(rflags.items()):
            rid = f"or-{m}-{fr}"
            if rid not in known:
                meta = API_META.get(m, {})
                used = sum(1 for x in rts if x > 0) / len(rts)
                note = ("reflex frame: single next word, reasoning disabled/minimal" if fr == "A" else
                        "deliberate frame: think it through, then the word") + \
                       (f"; {used:.0%} of its reflex calls reasoned anyway and are EXCLUDED (cells use only zero-reasoning calls)" if fr == "A" and used > 0.01 else "") + \
                       "; generation scoring (first/last word == gold)"
                ROWS.append({"id": rid, "label": f"{meta.get('label', m.split('/')[-1])} · {'reflex' if fr == 'A' else 'deliberate'}",
                             "group": "API · open weights" if meta.get("open") else "API · closed",
                             "size": meta.get("size"), "files": [], "note": note, "api": m})
                known.add(rid)
        for (m, fr, cid, mode), xs in api.items():
            if len(xs) >= 8:
                cells[f"or-{m}-{fr}|{cid}|0shot|{mode}"] = summarize(xs)
                e = aerr.get((m, fr, cid, mode))
                if e:
                    n_e = sum(e.values())
                    cells[f"or-{m}-{fr}|{cid}|0shot|{mode}"]["errs"] = {k: round(v / n_e, 3) for k, v in e.items() if k != "correct"}
                f = form.get((m, fr, cid))
                if f:
                    cells[f"or-{m}-{fr}|{cid}|0shot|{mode}"]["seat_form"] = round(sum(f) / len(f), 3)
    # examples: one verbatim 0-shot prompt per column (+ a shortcut-free one for deixis)
    sets = {"core": load_items(SRC / "items/core.jsonl"), "dx": load_items(SRC / "items/deixis-v2-full.jsonl"),
            "v3": load_items(V3_ITEMS)}
    examples: dict[str, Any] = {}
    for c in COLS:
        pool = [it for it in sets[c["items"]] if c["example"](it)]
        if not pool:
            continue
        it = pool[0]
        ex = {"prompt": build_prompt(it, "0shot"), "gold": it.gold, "candidates": it.candidates,
              "gold_set": it.meta.get("gold_set"), "item_id": it.item_id}
        if c["free_ok"]:
            fp = [x for x in pool if not x.meta.get("gold_is_last_mention") and not x.meta.get("gold_is_last_speaker")]
            if fp:
                ex["free"] = {"prompt": build_prompt(fp[0], "0shot"), "gold": fp[0].gold, "candidates": fp[0].candidates}
        ex["shot3"] = build_prompt(it, "3shot")
        # "take it yourself" pool: up to 8 more items (shortcut-free first for deixis), prompt + gold only
        rng = random.Random(zlib.crc32(c["id"].encode()))
        cand = [x for x in pool[1:] if not (c["free_ok"] and (x.meta.get("gold_is_last_mention") or x.meta.get("gold_is_last_speaker")))]
        cand = cand or pool[1:]
        ex["pool"] = [{"prompt": build_prompt(x, "0shot"), "gold": x.gold, "gold_set": x.meta.get("gold_set"),
                       "candidates": list(x.candidates)} for x in rng.sample(cand, min(8, len(cand)))]
        examples[c["id"]] = ex
    have = defaultdict(int)
    for k in cells:
        have[k.split("|")[0]] += 1
    GORDER = ["Human", "API · closed", "API · open weights", "Pretrained bases", "OLMo-2 7B trajectory",
              "Post-trained", "Hybrid & SSM bases", "Kotodama · 1.42B ladder"]
    gi = lambda g: GORDER.index(g) if g in GORDER else len(GORDER)
    by_size = ("Pretrained bases", "Hybrid & SSM bases")
    ROWS.sort(key=lambda r: (gi(r["group"]), r["label"].lower() if r["group"].startswith("API") else
                             ((r["size"] or 0), r["label"].lower()) if r["group"] in by_size else 0))
    meta_rows = [{k: v for k, v in r.items() if k not in ("files", "claude", "api")} | {"sources": [f"{Path(f).parent.name}/{Path(f).name}" for f in r.get("files", [])]}
                 for r in ROWS if have[r["id"]]]
    meta_cols = [{k: v for k, v in c.items() if k in ("id", "group", "label", "what", "kind", "free_ok")} for c in COLS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": meta_rows, "cols": meta_cols, "cells": cells, "examples": examples}))
    have = defaultdict(int)
    for k in cells:
        have[k.split("|")[0]] += 1
    print(f"{len(cells)} cells; rows with data: {sum(1 for r in ROWS if have[r['id']])}/{len(ROWS)}; "
          f"examples {len(examples)}/{len(COLS)}; fingerprint check ok ({len(prints)} grid files)")
    for r in ROWS:
        print(f"  {r['label']:32s} cells {have[r['id']]}")


if __name__ == "__main__":
    main()
