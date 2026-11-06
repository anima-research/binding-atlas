"""Refresh every synthesis-derived value in the Binding Atlas front-section data (data/built/stage.json).

Usage: python build/build_stage.py [--base data/built/stage.json] [--out data/built/stage.json] [--ledger-dir out/ledgers]
The base holds the curated parts (figure layout, stage texts, model lists); this script re-reads every number from
the synthesis table through facts.py (the trajectory, Figure 2 and 3 values, the stage texts that quote numbers, the
lead, and the decomposition, release and run tables), the headline-test table from the grid (data/built/atlas.json)
and the object-tracking table from data/inputs/riders_split.json, and writes the result. Rebuilding
from an up-to-date base reproduces it exactly (`make stage-check`).
The ledger directory gets a ledger of every synthesis row behind a number written, and a log of the base values
that moved by more than .0015. Values a checkpoint does not have are written as null and render as "pending".
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from facts import F, FactError, fmt3  # noqa: E402

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--base", type=Path, default=None, help="default: the --out file")
_ap.add_argument("--out", type=Path, default=REPO / "data" / "built" / "stage.json")
_ap.add_argument("--ledger-dir", type=Path, default=REPO / "out" / "ledgers")
_args = _ap.parse_args()
BASE, OUT, LEDGERS = _args.base or _args.out, _args.out, _args.ledger_dir
T, REL, AN1 = "kotodama-tsumugi", "anneal-110B-release", "anneal-110B-second"
G4 = {"I": "static I shift", "you": "static you shift", "dyn0": "dyn-you 0shot shift", "dyn3": "dyn-you 3shot shift"}


def r3(v: float | None) -> float | None:
    return None if v is None else round(float(v), 3)


def gauges(model: str, ckpt: str, tag: str) -> dict[str, float | None]:
    return {k: r3(F.get(model, ckpt, m, tag=f"{tag} {k}")) for k, m in G4.items()}


def try_get(model: str, ckpt: str, metric: str, tag: str, src: str | None = None) -> float | None:
    try:
        return r3(F.get(model, ckpt, metric, src=src, tag=tag))
    except FactError:
        F.derive(tag, float("nan"), f"PENDING: no synthesis row for {model}/{ckpt}/{metric}")
        return None


st: dict[str, Any] = json.loads(BASE.read_text())
new: dict[str, Any] = copy.deepcopy(st)
diffs: list[str] = []

# ------------------------------------------------------------------ trajectory (Figure 4)
TRUNK = [(f"trunk-{t}B", float(t)) for t in ("1.0", "2.0", "3.0", "3.75", "10", "20", "30", "40", "50", "60", "80", "100")]
ANNEALED = [("anneal-5B", 5.0, ""), ("anneal-11B", 11.0, ""), ("anneal-33B", 33.0, ""), ("anneal-55B", 55.0, ""),
            ("anneal-88B", 88.0, ""),
            (REL, 110.0, "release"), (AN1, 110.0, "the other cooldown from the same 100B checkpoint")]
s17 = [{"tok": tok, "kind": "constant", **gauges(T, ck, f"traj s17 {ck}")} for ck, tok in TRUNK]
s17 += [{"tok": tok, "kind": "annealed", **({"note": note} if note else {}), **gauges(T, ck, f"traj s17 {ck}")}
        for ck, tok, note in ANNEALED]
s7 = [{"tok": tok, "kind": "annealed", **gauges(T, f"s7-{ck}", f"traj s7 {ck}")}
      for ck, tok in (("anneal-5B", 5.0), ("anneal-11B", 11.0))]
s7 += [{"tok": tok, "kind": "constant", **gauges(T, f"s7-{ck}", f"traj s7 {ck}")} for ck, tok in TRUNK[:5]]
old = st["tsumugi"]["traj"]
for seed, pts in (("s17", s17), ("s7", s7)):
    for q in old[seed]:
        m = [p for p in pts if p["tok"] == q["tok"] and p["kind"] == q["kind"] and not p.get("note")]
        for k in G4:
            if m and abs(m[0][k] - q[k]) > 0.0015:
                diffs.append(f"traj {seed} {q['kind']} {q['tok']}B {k}: base {q[k]} -> synthesis {m[0][k]}")

olmo_onset = [("OLMo-2 13B", "OLMo-2-1124-13B", "42B tokens"), ("OLMo-2 7B", "OLMo-2-1124-7B", "210B tokens"),
              ("OLMo-2 1B", "OLMo-2-0425-1B", "420B tokens")]
olmo_stage3 = []
for label, m, ck in olmo_onset:
    r = F.row(m, ck, "static you shift", tag=f"fig4 OLMo stage-3 onset {label}")
    olmo_stage3.append({"m": label, "tok": int(r["tokens_B"])})

GRID = json.loads((REPO / "data" / "built" / "atlas.json").read_text())
BAT_CKS = [("main · 5B", "tsumugi-s17-a5B"), ("main · 11B", "tsumugi-s17-a11B"), ("main · 33B", "tsumugi-s17-a33B"),
           ("main · 55B", "tsumugi-s17-a55B"), ("main · 88B", "tsumugi-s17-a88B"), ("110B · release", "tsumugi"),
           ("second run · 5B", "tsumugi-s7-a5B"), ("second run · 11B", "tsumugi-s7-a11B")]


def battery_from_grid() -> dict[str, Any]:
    """Table 3: the grid's headline columns plus the known-facts control at each kotodama-tsumugi checkpoint in the
    grid, each read as the grid reads it: with examples where the column has them, else cold, and the rule-learning
    columns as ladder AUC. Values are the grid's own cell scores."""
    cols = [c for c in GRID["cols"] if c["headline"] or c["id"] == "facts"]
    groups: list[dict[str, Any]] = []
    for c in cols:
        cond = "fair" if "fair" in c["conds"] else c["conds"][0]        # the column's own reading, as in the grid
        vals = []
        for _, rid in BAT_CKS:
            cell = GRID["cells"].get(f"{rid}|{c['id']}|{cond}|all")
            if cell is None:
                raise SystemExit(f"battery: no grid cell {rid}|{c['id']}|{cond}")
            vals.append(round(float(cell["score"]), 3))
        name = c["label"] + {"fair": "", "0shot": " (cold)", "ladder": " (AUC)"}[cond]
        if groups and groups[-1]["name"] == c["group_short"]:
            groups[-1]["rows"].append([name, vals])
        else:
            groups.append({"name": c["group_short"], "rows": [[name, vals]]})
    return {"cols": [lab for lab, _ in BAT_CKS], "groups": groups, "signed": True}


ts = {
    "_note": ("kotodama-tsumugi s17 trunk (constant LR, to 100B) and cooldown branches (annealed), stage gauges = P2 "
              "speaker swap (static I, static you) and dynamic swap (addressee, 0-shot / 3-shot). Seed 7 = the "
              "seed replicate to 11B. Values from the synthesis table."),
    "traj": {"s17": s17, "s7": s7},
    "olmo_stage3": olmo_stage3,
    "battery": battery_from_grid(),
    "desc": ("A 1.42B-parameter model with a looped core, run as 72 effective layers. Its trunk trains at a constant "
             "learning rate to 100B tokens. Annealed reads are cooldown branches forked from its checkpoints "
             "(cosine decay to zero over about 10% of the tokens so far). The release is the 110B branch from the 100B "
             "fork. The architecture and training run are described on the model card (huggingface.co/LuxiaSL/kotodama-tsumugi-1b), "
             "and the release table below gives the spread between two anneals."),
}
new["tsumugi"] = ts

# ------------------------------------------------------------------ Figure 2 (threshold)
TH = {"OLMo-2 1B (4T, pre-anneal)": ("OLMo-2-0425-1B", "4001B tokens"), "SmolLM2 1.7B": ("SmolLM2-1.7B", "11000B tokens"),
      "SmolLM3 3B": ("SmolLM3-3B", "11139B tokens"), "Qwen3 0.6B": ("Qwen3-0.6B-Base", "36000B tokens"),
      "Qwen3 1.7B": ("Qwen3-1.7B-Base", "36000B tokens"), "Qwen3 4B": ("Qwen3-4B-Base", "36000B tokens"),
      "Qwen3 8B": ("Qwen3-8B-Base", "36000B tokens"), "Qwen3 14B": ("Qwen3-14B-Base", "36000B tokens"),
      "Qwen2.5 0.5B": ("Qwen2.5-0.5B", "18000B tokens"), "Qwen2.5 1.5B": ("Qwen2.5-1.5B", "18000B tokens"),
      "Qwen2.5 3B": ("Qwen2.5-3B", "18000B tokens"), "Qwen2.5 7B": ("Qwen2.5-7B", "18000B tokens"),
      "Qwen2.5 14B": ("Qwen2.5-14B", "18000B tokens"), "OLMo-2 7B": ("OLMo-2-1124-7B", "3896B tokens"),
      "OLMo-2 13B": ("OLMo-2-1124-13B", "5000B tokens"), "Gemma-2 9B": ("Gemma-2-9B", "main/final"),
      "Qwen3 30B-A3B": ("Qwen3-30B-A3B-Base", "main/final"), "GLM-4 32B": ("GLM-4-32B-Base", "main/final"),
      "OLMo-3 32B": ("OLMo-3-32B", "main/final"), "Llama-3.1 70B": ("Llama-3.1-70B", "main/final"),
      "Qwen2.5 72B": ("Qwen2.5-72B", "main/final"), "Llama-3.1 405B": ("Llama-3.1-405B", "main/final"),
      "Nemotron-3 Ultra": ("nemotron3-ultra-550b-base", "final"), "DeepSeek-V3": ("dsv3-base", "final"),
      "Kimi-K2": ("kimi-k2-base", "final"), "DeepSeek-V4 Flash": ("dsv4-flash-base", "final"),
      "DeepSeek-V4 Pro": ("dsv4-pro-base", "final"), "Trinity-Large (base)": ("trinity-base", "final"),
      "MiMo-V2.5 Pro (base)": ("mimo-v25-pro-base", "final")}
for grp in ("models", "unsized"):
    for m in new["threshold"][grp]:
        model, ck = TH[m["m"]]
        z = r3(F.get(model, ck, "dyn-you 0shot shift", tag=f"fig2 {m['m']} 0s"))
        t = r3(F.get(model, ck, "dyn-you 3shot shift", tag=f"fig2 {m['m']} 3s"))
        for k, v in (("z", z), ("t", t)):
            if abs(m[k] - v) > 0.0015:
                diffs.append(f"fig2 {m['m']} {k}: base {m[k]} -> synthesis {v}")
            m[k] = v
rel_g = gauges(T, REL, "fig2 release")
new["threshold"]["ours"] = {"m": "kotodama-tsumugi 1.4B @ 110B tokens (release)", "p": 1.42,
                            "z": rel_g["dyn0"], "t": rel_g["dyn3"]}
new["threshold"]["_note"] = ("endpoint dynamic-you swap shift vs total parameters (B). MoE = total params. Models "
                             "without a public param count are listed in 'unsized'.")

# ------------------------------------------------------------------ Figure 3 (cost)
def bracket(model: str, lo_ck: str, hi_ck: str, metric: str, tag: str) -> tuple[float, float]:
    lo = F.row(model, lo_ck, metric, tag=f"{tag} (last ckpt below band)")
    hi = F.row(model, hi_ck, metric, tag=f"{tag} (first ckpt that binds)")
    if not ((lo.get("ci") or [1])[0] <= 0.02 < (hi.get("ci") or [0])[0]):
        raise SystemExit(f"bracket does not bracket: {model} {lo_ck}..{hi_ck} {metric}")
    return float(lo["tokens_B"]), float(hi["tokens_B"])


for row in new["cost"]["rows"]:
    for q in row["pts"]:
        if q["m"] == "kotodama-tsumugi" and row["stage"].startswith("1"):
            q["lo"], q["hi"] = bracket(T, "trunk-3.0B", "trunk-3.75B", "static I shift", "fig3 tsumugi stage 1")
        if q["m"] == "kotodama-tsumugi" and row["stage"].startswith("3"):
            q["lo"], q["hi"] = bracket(T, "trunk-30B", "anneal-33B", "static you shift", "fig3 tsumugi stage 3")
        if q["m"] == "kotodama-tsumugi" and row["stage"].startswith("5"):
            q["lo"], q["hi"], q["note"] = 110, None, "0-shot not reached by 110B; with three examples it binds at 110B"
        if q["m"] == "OLMo-2 7B" and row["stage"].startswith("5"):
            q["lo"], q["hi"] = bracket("OLMo-2-1124-7B", "210B tokens", "3896B tokens", "dyn-you 0shot shift", "fig3 OLMo-2-7B stage 5")
            q.pop("note", None)
        if q["m"] == "OLMo-2 13B" and row["stage"].startswith("5"):
            q["lo"], q["hi"] = bracket("OLMo-2-1124-13B", "42B tokens", "5000B tokens", "dyn-you 0shot shift", "fig3 OLMo-2-13B stage 5")
            q.pop("note", None)
        if q["m"] == "OLMo-2 1B" and row["stage"].startswith("3"):
            q["lo"], q["hi"] = bracket("OLMo-2-0425-1B", "420B tokens", "525B tokens", "static you shift", "fig3 OLMo-2-1B stage 3")
        if q["m"] == "OLMo-2 1B" and row["stage"].startswith("1"):
            q["lo"], q["hi"] = bracket("OLMo-2-0425-1B", "21B tokens", "63B tokens", "static I shift", "fig3 OLMo-2-1B stage 1")
        if q["m"] == "OLMo-2 7B" and row["stage"].startswith("1"):
            q["lo"], q["hi"] = bracket("OLMo-2-1124-7B", "21B tokens", "42B tokens", "static I shift", "fig3 OLMo-2-7B stage 1")
        if q["m"] == "OLMo-2 7B" and row["stage"].startswith("3"):
            q["lo"], q["hi"] = bracket("OLMo-2-1124-7B", "105B tokens", "210B tokens", "static you shift", "fig3 OLMo-2-7B stage 3")
        if q["m"] == "OLMo-2 13B" and row["stage"].startswith("1"):
            q["lo"], q["hi"] = bracket("OLMo-2-1124-13B", "17B tokens", "42B tokens", "static I shift", "fig3 OLMo-2-13B stage 1")
        if q["m"] == "OLMo-2 13B" and row["stage"].startswith("3"):
            q["lo"], q["hi"] = bracket("OLMo-2-1124-13B", "17B tokens", "42B tokens", "static you shift", "fig3 OLMo-2-13B stage 3")
COST_LABELS = {"1": "1 · I = speaker", "3": "3 · you = person spoken to", "5": "5 · you through hand-offs"}
for row in new["cost"]["rows"]:
    row["stage"] = COST_LABELS[row["stage"][0]]
new["cost"]["_note"] = ("tokens at which each stage first binds (gauge CI rule), OLMo-2 family (one data recipe) + "
                        "kotodama-tsumugi (1.4B, 72 effective layers). [lo, hi] = bracketing checkpoints.")

# ------------------------------------------------------------------ Figure 1 (stages) + lead
S = {s["n"]: s for s in new["stages"]}
dyn11 = F.get(T, "anneal-11B", "dyn-you 0shot shift", tag="stage4 tsumugi dyn0 anneal-11B")
you_rel = F.get(T, REL, "static you shift", tag="stage3 tsumugi static you release")
pyth28 = F.row("Pythia-2.8B", "main/final (via step300B branch)", "static you shift", tag="stage3 Pythia-2.8B static you (does not bind)")
NAMES = {1: ("I = the speaker", "\"I\" means whoever is speaking."),
         2: ("you = the speaker (a mistake)",
             "\"you\" is read as the speaker too: the pronoun-reversal phase young children go through."),
         3: ("you = the person spoken to", "\"you\" means the addressee, when who holds what is simply stated."),
         4: ("you = the speaker again", "once objects change hands in the chat, \"you\" slides back onto the speaker."),
         5: ("you followed through hand-offs",
             "\"you\" is tracked as objects change hands: first with three worked examples, later with none.")}
for n, (name, gloss) in NAMES.items():
    S[n]["name"], S[n]["gloss"] = name, gloss
S[1]["ok"] = "every base of 0.5B parameters or more"
S[2]["ok"] = "every training run measured passes through it"
S[4]["ok"] = "every base up to 4B parameters, and they stop here"
S[1]["ours"] = "~3–3.75B tokens"
S[2]["ours"] = "passes through it at ~4–20B tokens"
S[3]["ok"] = "every base of 0.6B or more trained on trillions of tokens, except Bamba 9B"
S[3]["olmo1b"] = "420–525B tokens"
S[3]["ours"] = f"by 33B tokens with a cooldown (40B without one); {fmt3(you_rel, True)} at 110B"
S[4]["ours"] = (f"from ~4B tokens; by 110B no longer reliably reversed ({fmt3(dyn11)} at 11B → "
                f"{fmt3(rel_g['dyn0'])})")
S[5]["ok"] = ("with no examples: from about 7B (OLMo-2 7B, Qwen2.5 7B), and every base from 13B up; "
              "with three worked examples: from Qwen3 4B")
S[5]["ours"] = (f"with no examples: not reached by 110B tokens; with three worked examples it passes at 110B "
                f"({fmt3(rel_g['dyn3'], True)}), the smallest base measured that does")
new["lead"] = (
    "Base language models learn to look things up long before they learn to track change. In the box puzzle, items go "
    "into boxes and are then swapped, moved or refilled, and the model is asked what one box holds. The median model "
    "under 1B already finds what a box was said to hold, and scores zero on the boxes that a later step changed. With "
    "worked examples, updating climbs steadily with size, to .77 for the 70B models and above .8 for most of the "
    "largest. In a chat, *you* means the person being spoken to. Models get this right in five stages, in the same order "
    "in every training run measured. To read a stage, the speaker's name on the chat's last line is swapped: a model "
    "that follows *you* moves its answer to the other person's object. The swap shift is how much probability leaves "
    "the first answer; a model passes when the whole 95% interval of that shift is above +.02, a bar fixed before any "
    "model was scored. Every base of 0.6B "
    "parameters or more trained on trillions of tokens reaches the third stage (*you* is the person spoken to), "
    "except Bamba 9B. The fifth stage is following *you* as objects change hands. With no examples in the prompt, it "
    "appears from about 7B (OLMo-2 7B, Qwen2.5 7B), and every base measured from 13B up reaches it.")
# kotodama-tsumugi's own paragraph: the lead of its page beside the atlas
new["tsumugi"]["lead"] = (
    "**kotodama-tsumugi** is a 1.4B model whose middle 20 layers run twice, so each token passes through 72 layers. It "
    "reaches the third stage by **33B tokens**, in the range where OLMo-2 13B gets there; OLMo-2 1B needs 420–525B. "
    "Given three worked examples, it is the smallest base measured that passes the fifth stage: "
    f"{fmt3(rel_g['dyn3'], True)}, with its whole 95% interval above +.02. The next is Qwen3 4B.")

# ------------------------------------------------------------------ NEW: stage-5 decomposition (endpoints, full battery)
DEC_COLS = [("third", "third/0shot acc (self+addressee)"), ("deictic", "deictic/0shot acc (self+addressee)"),
            ("named_transfer", "named_transfer/0shot acc (self+addressee)"), ("vocative", "vocative/0shot acc (self+addressee)")]
DEC_ROWS = [("Pythia-1.4B", "Pythia-1.4B", "134B tokens", 1.41, "134B", False),
            ("kotodama-tsumugi · 33B", T, "anneal-33B", 1.42, "33B", True),
            ("kotodama-tsumugi · 55B", T, "anneal-55B", 1.42, "55B", True),
            ("kotodama-tsumugi · 88B", T, "anneal-88B", 1.42, "88B", True),
            ("kotodama-tsumugi · 110B, other cooldown", T, AN1, 1.42, "110B", True),
            ("kotodama-tsumugi · 110B, release", T, REL, 1.42, "110B", True),
            ("OLMo-2 1B", "OLMo-2-0425-1B", "4001B tokens", 1.48, "4T", False),
            ("SmolLM2 1.7B", "SmolLM2-1.7B", "11000B tokens", 1.71, "11T", False),
            ("Qwen3 1.7B", "Qwen3-1.7B-Base", "36000B tokens", 1.72, "36T", False),
            ("Qwen3 4B", "Qwen3-4B-Base", "36000B tokens", 4.02, "36T", False),
            ("Qwen2.5 7B", "Qwen2.5-7B", "18000B tokens", 7.62, "18T", False),
            ("Llama-3.1 8B", "Llama-3.1-8B", "main/final", 8.03, "15T", False),
            ("Qwen3 8B", "Qwen3-8B-Base", "36000B tokens", 8.19, "36T", False),
            ("Qwen2.5 14B", "Qwen2.5-14B", "18000B tokens", 14.77, "18T", False),
            ("Qwen3 14B", "Qwen3-14B-Base", "36000B tokens", 14.77, "36T", False),
            ("Qwen3 30B-A3B", "Qwen3-30B-A3B-Base", "main/final", 30.53, "36T", False),
            ("Llama-3.1 70B", "Llama-3.1-70B", "main/final", 70.6, "15T", False)]
dec_rows = []
for label, model, ck, p, tok, ours in DEC_ROWS:
    vals = {k: try_get(model, ck, met, f"decomp {label} {k}") for k, met in DEC_COLS}
    dec_rows.append({"m": label, "p": p, "tok": tok, "ours": ours, **vals})
new["decomp"] = {
    "_note": ("stage5-decomp-v1: each dynamic-you gauge item rewritten along one axis. Accuracy pooled over self + "
              "addressee queries, 0-shot, 360 items per cell (SE ≈ .024); chance ≈ .29. Endpoints only."),
    "cols": [{"k": "third", "name": "story, no I or you", "ex": "Erin gives the cup to Paul. … Right now Paul is holding the"},
             {"k": "deictic", "name": "chat (the test itself)", "ex": "Erin: Here, take my cup. … Paul: Right now I'm holding the"},
             {"k": "named_transfer", "name": "chat, hand-off as a stage direction", "ex": "[Erin gives the cup to Paul.] … deictic query"},
             {"k": "vocative", "name": "chat, addressee named", "ex": "Erin: Paul, right now you're holding the"}],
    "chance": 0.287,
    "rows": dec_rows,
}

# ------------------------------------------------------------------ riders: lookup vs update, by size (Table 5)
RS = json.loads((REPO / "data" / "inputs" / "riders_split.json").read_text())["models"]
RS_REGS = ("code", "narrative", "screenplay", "chat")
RS_COLS = [("l0", "0shot", "lookup", "lookup, cold"), ("u0", "0shot", "update", "update, cold"),
           ("lf", "fair", "lookup", "lookup, with examples"), ("uf", "fair", "update", "update, with examples")]
RS_BANDS = [(0, 1, "under 1B"), (1, 2, "1–2B"), (2, 5, "2–5B"), (5, 12, "7–10B"), (12, 40, "13–34B"),
            (40, 100, "70–72B"), (100, 1e9, "284B–1.6T")]
RS_SKIP = {"olmo3-32b-stage1", "trinity-truebase"}      # earlier checkpoints of a model already in the table
SIZE = {r["id"]: r["size"] for r in GRID["rows"] if r.get("size")}


def rs_pooled(model: str, cond: str, kind: str) -> float:
    vals = [RS[model][f"{reg}|{cond}|{kind}"][0] for reg in RS_REGS]
    if any(v is None for v in vals):
        raise SystemExit(f"riders split: {model} lacks {cond}/{kind} in a register")
    return sum(vals) / len(vals)


def median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) / 2


rs_rows = []
public = [m for m in RS if not m.startswith("tsumugi") and m not in RS_SKIP]
unsized = [m for m in public if m not in SIZE]
if unsized:
    raise SystemExit(f"riders split: no grid size for {unsized}")
for lo, hi, name in RS_BANDS:
    ms = [m for m in public if lo <= SIZE[m] < hi]
    if not ms:
        raise SystemExit(f"riders split: empty band {name}")
    rs_rows.append({"m": f"{name} ({len(ms)} models)", "ours": False, "models": sorted(ms),
                    **{k: r3(median([rs_pooled(m, c, kind) for m in ms])) for k, c, kind, _ in RS_COLS}})
for label, rid in (("kotodama-tsumugi 1.4B · release", "tsumugi"),
                   ("kotodama-tsumugi 1.4B · other cooldown", "tsumugi-anneal-110B-second")):
    rs_rows.append({"m": label, "ours": True, **{k: r3(rs_pooled(rid, c, kind)) for k, c, kind, _ in RS_COLS}})
new["riders"] = {
    "_note": ("riders ladder, the box puzzle in four registers (code, narrative, screenplay, chat), items with 1-4 "
              "operations. Lookup = the queried box ends with its declared contents; update = an operation changed "
              "it. kappa, mean over the four registers; public rows = the median model of each size band (total "
              "parameters). One model's cell: 68 lookup or 82 update items per register, 272 or 328 in all, about ±.08 at 95%."),
    "cols": [{"k": k, "name": nm} for k, _, _, nm in RS_COLS], "rows": rs_rows, "signed": True,
}

# ------------------------------------------------------------------ NEW: the release against its trajectory and class
REL_COLS = [("33B", T, "anneal-33B"), ("55B", T, "anneal-55B"), ("88B", T, "anneal-88B"), ("110B · other cooldown", T, AN1),
            ("110B · release", T, REL), ("OLMo-2 1B · 4T", "OLMo-2-0425-1B", "main/final"),
            ("SmolLM2 1.7B · 11T", "SmolLM2-1.7B", "main/final"), ("Qwen3 1.7B · 36T", "Qwen3-1.7B-Base", "main/final")]
GAUGE_CK = {"OLMo-2-0425-1B": "main/final", "SmolLM2-1.7B": "11000B tokens", "Qwen3-1.7B-Base": "36000B tokens"}
REL_ROWS = [("I = speaker (swap shift)", "static I shift", "gauge"),
            ("you = person spoken to (swap shift)", "static you shift", "gauge"),
            ("you through hand-offs, no examples (swap shift)", "dyn-you 0shot shift", "gauge"),
            ("you through hand-offs, three examples (swap shift)", "dyn-you 3shot shift", "gauge"),
            ("name–task pairs held: K50", "K50 (D=0)", "capacity"),
            ("one name–task pair past 256 filler tokens (accuracy, chance .04)", "ceiling C (K=1, D=256)", "capacity"),
            ("who has what, said with I and you (accuracy, chance ≈ .33)", "dx2 deixis/deictic/0shot acc", "bindprobes"),
            ("who has what, third person (accuracy, chance ≈ .33)", "dx2 deixis/third/0shot acc", "bindprobes")]
rel_tab = []
for label, metric, harness in REL_ROWS:
    vals = []
    for col, model, ck in REL_COLS:
        ck2 = GAUGE_CK.get(model, ck) if harness == "gauge" else ck
        vals.append(try_get(model, ck2, metric, f"release-table {label} {col}"))
    rel_tab.append([label, vals, harness == "gauge"])
new["release"] = {"cols": [c for c, _, _ in REL_COLS], "ours": [m == T for _, m, _ in REL_COLS], "rows": rel_tab}

# ---- Figure 5: the run against its peers — matched checkpoints, every same-harness point per model and metric
RUN_MODELS = [(T, "kotodama-tsumugi", "ours"), ("Pythia-1.4B", "Pythia-1.4B", "dash"),
              ("OLMo-2-0425-1B", "OLMo-2 1B", "solid"), ("OLMo-1B-0724", "OLMo-1 1B", "dot"),
              ("SmolLM2-1.7B", "SmolLM2-1.7B", "dashdot")]
RUN_METRICS = {"avg9": "avg9", "ubpb": "U-BPB", "I": "static I shift", "you": "static you shift"}
run = []
for model, label, style in RUN_MODELS:
    fpt = None
    for r in F.rows:                       # FLOPs per token from any avg9 row (the synthesis convention)
        if r["model"] == model and r["metric"] == "avg9" and r.get("train_flops") and r.get("tokens_B"):
            fpt = r["train_flops"] / (r["tokens_B"] * 1e9)
            break
    by_ck: dict[str, dict] = {}
    for key, metric in RUN_METRICS.items():
        cks = sorted({r["checkpoint"] for r in F.rows if r["model"] == model and r["metric"] == metric and r.get("tokens_B")})
        for ck in cks:
            if model == T and (ck == AN1 or ck.startswith("s7-") or ck.endswith("-rescored")):
                continue   # seed 17 only (seed 7 is Figure 4's); the second anneal and scorer reruns are not separate points
            r = F.row(model, ck, metric, tag=f"run {label} {ck} {metric}")
            if isinstance(r["value"], str) or r.get("tokens_B") is None:
                continue
            pt = by_ck.setdefault(ck, {"ck": ck, "tok": r["tokens_B"],
                                       "kind": ("annealed" if ck.startswith("anneal") else "trunk") if model == T else "base"})
            pt[key] = round(float(r["value"]), 4)
    pts = sorted(by_ck.values(), key=lambda p: (p["tok"], p["ck"]))
    if fpt is None:
        raise SystemExit(f"no FLOPs per token for {model}")
    for p in pts:
        p["flops"] = p["tok"] * 1e9 * fpt
        if model == T and p["ck"] == REL:
            p["note"] = "release"
    run.append({"m": label, "style": style, "fpt": fpt, "pts": pts})
new["run"] = {"_note": "same-harness points per checkpoint (synthesis table): lm-eval avg9, Uncheatable bits/byte, stage-gauge "
              "swap shifts. tsumugi trunk = constant LR (trunk-*), annealed = cooldown branches (anneal-*), release = "
              "anneal-110B-release. "
              "FLOPs = tokens x FLOPs/token (tsumugi 1.13e10 with looped depth; public 6N).", "models": run}

OUT.parent.mkdir(parents=True, exist_ok=True)
LEDGERS.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(new, ensure_ascii=False, indent=1))
F.dump(LEDGERS / "atlas-ledger.json")
(LEDGERS / "stage-diffs.log").write_text("\n".join(diffs) + "\n")
print(f"wrote {OUT}; {len(diffs)} base-vs-synthesis diffs > .0015 ({LEDGERS / 'stage-diffs.log'})")
