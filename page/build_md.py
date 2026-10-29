"""Render the agent-friendly Markdown twin of the Binding Atlas: same data as binding-atlas.html, as plain text.

Usage: python page/build_md.py --atlas data/built/atlas.json --stage data/built/stage.json --swap data/built/swap.json
                               --out site/binding-atlas.md [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
                               [--swap-prompts swap-prompts.jsonl]   (optional: prints one example speaker-swap pair)
Sections: how to read · findings · caveats · columns (+ verbatim example prompts per column: 0-shot, shortcut-free,
3-shot, a few more items) · rows · 0-shot wide tables per battery family · speaker-swap table · every cell as CSV.
Refuses to write if the output contains an internal hostname / path (same law as build_page.py).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_page import FORBIDDEN, build_stamp, common_args, load_config  # noqa: E402  (one table of forbidden names)

N_POOL = 1                                   # extra example items per column (the page has up to 8)
SMALL_N = 20                                 # cells with fewer items get a dagger in the wide tables
API_NOTE = ("reflex frame: single next word, reasoning disabled/minimal", "deliberate frame: think it through, then the word",
            "; generation scoring (first/last word == gold)")
# page-only instructions -> their equivalent in this file
MD_REWRITES = {
    "Toggle <i>Shortcut-free</i> to watch it happen.": "Compare `mode=all` with `mode=free` in the CSV to see it.",
    "Open any deixis cell for its error anatomy.": "The error anatomy is the `errs` field of each deixis cell in the CSV.",
}
SHORT = {"dx-owner-3-deictic": "owner I", "dx-owner-3-third": "owner 3rd", "dx-self-3-deictic": "self3 I",
         "dx-self-3-third": "self3 3rd", "dx-self-2-deictic": "self2 I", "dx-self-2-third": "self2 3rd",
         "dx-addressee-3-deictic": "you3 I", "dx-addressee-3-third": "you3 3rd", "dx-addressee-2-deictic": "you2 I",
         "dx-addressee-2-third": "you2 3rd"}
GLOSSARY = [
    ("P2 / P3", "number of people in the chat: P2 = two people (so 'you' is simply the other person); P3 = three "
                "(every 'you' line opens with a vocative naming the addressee)."),
    ("owner / self / addressee", "deixis query kinds: owner = 'Now the cup is with' -> a name; self = 'S: Right now I'm "
                                 "holding the' -> the speaker's object; addressee = 'S: Right now you're holding the' -> "
                                 "the addressee's object."),
    ("I/you/my vs 3rd-person twin", "the same events told in chat with pronouns vs in prose with names; same gold and "
                                    "candidates. Column ids end in -deictic / -third."),
    ("structural you", "a 'you' whose referent is fixed only by the conversation (P2, no vocative)."),
    ("shortcut-free (mode=free)", "items where the gold is neither the last-mentioned candidate nor the last speaker, "
                                  "so recency cannot answer them."),
    ("kind: acc", "gold is the top-ranked candidate (logprob rows) or the produced word (API/human rows)."),
    ("kind: hit1 / exactk", "set binding: hit1 = top candidate is in the gold set; exactk = the top-k candidates are "
                            "exactly the gold set (k = set size); chance differs accordingly."),
    ("kind: cal", "answer deixis, open models: argmax of logp(item) − logp(same tail with the body hidden); API rows "
                  "use referent scoring instead (the person's name counts as right as well as me/you)."),
    ("seat_form", "API answer columns: share of referent-correct answers that used me/you rather than the name."),
    ("errs", "deixis I/you cells: shares of items by error type (a wrong pick only). speaker_confusion = the wrong side "
             "of I/you (for 'you' queries: an object the SPEAKER holds); previous_owner = someone who held it earlier; "
             "other_participant = a third person (owner queries) / their object; other_object = another object; "
             "tie = gold tied for top (counted wrong); format = API answer not among the candidates; other = rest."),
    ("reflex / deliberate", "API frames: reflex = answer with the next word, reasoning off (calls that reasoned anyway "
                            "are excluded); deliberate = think first, then give the word."),
    ("row ids", "or-<provider>/<model>-A = API reflex row, -B = API deliberate row; other ids are open checkpoints "
                "(e.g. llama31-405b). 'human' = the human row."),
    ("swap shift", "speaker-swap test: P(original gold | original asker) − P(original gold | swapped asker)."),
]
FAMILY_NOTES = {
    "Set binding": "3-shot exemplars show completed sentences (often one object); the test item is scored on the "
                   "exact gold set.",
    "Answer deixis": "Roles are cycled so every constant answer is at chance: holder = the answerer (me) ¼, the asker "
                     "(you) ¼, one of the two others (a name) ½. For open models the candidates are me / you / the two "
                     "OTHER people's names; exemplar answers follow the same cycle, so 'you'/'me' exemplars before a "
                     "name item are expected.",
    "Adjacency": "Critical items (the scored ones) end with the object last moved by a grant; exemplars may be "
                 "non-critical.",
    "Deixis": "Only the 0-shot and shortcut-free examples are paired with their twin; the 'more item' examples of the "
              "I/you and 3rd-person columns are independent draws.",
}


def md_text(s: str) -> str:
    """Page HTML snippets → Markdown (bold/italic kept, other tags dropped, entities decoded)."""
    s = re.sub(r"</?b>", "**", s)
    s = re.sub(r"</?i>", "*", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s)


def fence(text: str, lang: str = "text") -> str:
    t = text.replace("```", "ˋˋˋ")
    return f"```{lang}\n{t}\n```"


def cell(cells: dict, rid: str, cid: str, cond: str, mode: str):
    return cells.get(f"{rid}|{cid}|{cond}|{mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common_args(ap)
    ap.add_argument("--swap", type=Path, default=None, help="speaker-swap table (data/built/swap.json)")
    ap.add_argument("--swap-prompts", type=Path, default=None, help="swap prompts JSONL, for one example pair")
    args = ap.parse_args()
    cfg = load_config(args.config)
    PAGE_URL = cfg.page_link
    SWAP_PROMPTS = args.swap_prompts
    atlas = json.loads(args.atlas.read_text())
    out = args.out
    swap = json.loads(args.swap.read_text()) if args.swap is not None and args.swap.exists() else None
    meta = json.loads(args.meta.read_text())
    rows, cols, cells, ex = atlas["rows"], atlas["cols"], atlas["cells"], atlas["examples"]
    have = {k.split("|")[0] for k in cells}
    rows = [r for r in rows if r["id"] in have]
    L: list[str] = []
    P = L.append
    build = build_stamp(args.stamp)
    n_items = sum(v["n"] for k, v in cells.items() if k.endswith("|all"))

    P("# Binding Atlas — agent view")
    P("")
    P(f"Plain-text twin of the [Binding Atlas page]({PAGE_URL}). Same data, same build. Built {build}: "
      f"{len(rows)} rows (models + one human), {len(cols)} columns, {len(cells):,} cells, {n_items:,} scored items.")
    P("")
    stage = json.loads(args.stage.read_text())
    f3 = lambda v: ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.3f}".lstrip("0")
    P("## Summary")
    P("")
    P(stage["lead"])
    P("")
    P("Swap shift = mean[p(gold | original) − p(gold | speaker swapped)] over the candidates: above +.02 a pronoun binds "
      "(its answer moves with the speaker), below −.02 it inverts (it resolves to the speaker).")
    P("")
    P("## Part I · Where base models fail")
    P("")
    P("### The staircase")
    P("")
    P("| stage | name | what it means | example | where public models stand | OLMo-2 1B | kotodama-tsumugi 1.4B |")
    P("|---|---|---|---|---|---|---|")
    for st in stage["stages"]:
        P(f"| {st['n']} | {st['name']} | {st['gloss']} | `{st['ex']}` | {st['ok']} | {st['olmo1b']} | {st['ours']} |")
    P("")
    P("### Dynamic *you* at each model's final checkpoint (OLMo-2 1B: its 4T pre-anneal checkpoint; not reached zero-shot at "
      "or below 8B; weak above 14B)")
    P("")
    P("| model | parameters (B) | training tokens | 0-shot | 3-shot |")
    P("|---|---|---|---|---|")
    for m in sorted(stage["threshold"]["models"], key=lambda m: m["p"]):
        P(f"| {m['m']} | {m['p']} | {m['tok']} | {f3(m['z'])} | {f3(m['t'])} |")
    for m in stage["threshold"]["unsized"]:
        P(f"| {m['m']} | not public | — | {f3(m['z'])} | {f3(m['t'])} |")
    o = stage["threshold"]["ours"]
    P(f"| {o['m']} | {o['p']} | 110B | {f3(o['z'])} | {f3(o['t'])} |")
    P("")
    P("No model at or below 8B binds zero-shot dynamic *you* at any token count measured (to 36T). The same holds with "
      "each chat rewritten as a scene (8 of 8 models tested at or below 8B; the highest is +.010, Qwen2.5-7B), so the "
      "result does not come from the chat format. This is not evidence that the stage needs more parameters. No model "
      "at or below 8B has been trained far enough to rule out more tokens. Whether more tokens would bring a small "
      "model to stage 5 is unknown. From 14B to 1T parameters, stage 5 is weak and nearly flat. Ten of the 11 bases "
      "with a public size read +.01 to +.07 zero-shot. Static *you* reads +.27 to +.82 in the seven of them where it "
      "was measured.")
    P("")
    P("### Training tokens until each stage appears (the range spans the checkpoints that bracket the stage)")
    P("")
    P("| stage | model | tokens |")
    P("|---|---|---|")
    for r in stage["cost"]["rows"]:
        for q in r["pts"]:
            P(f"| {r['stage']} | {q['m']} | {q['lo']}–{q['hi']}B |" if q.get("hi") is not None else
              f"| {r['stage']} | {q['m']} | not reached by {q['lo']}B{'; ' + q['note'] if q.get('note') and not q['note'].startswith(('not reached', 'never')) else ''} |")
    P("")
    dec = stage["decomp"]
    P("### Stage 5, taken apart (accuracy, 0-shot, self + addressee queries pooled; chance ≈ .29)")
    P("")
    P("Each dynamic-*you* gauge item is rewritten along one axis: " + "; ".join(f"**{c['name']}** (`{c['ex']}`)" for c in dec["cols"]) + ".")
    P("")
    P("| model | tokens | " + " | ".join(c["name"] for c in dec["cols"]) + " |")
    P("|---|---|" + "---|" * len(dec["cols"]))
    pend = lambda v: "pending" if v is None else f"{v:.3f}".lstrip("0")
    for r in dec["rows"]:
        P(f"| {r['m']} | {r['tok']} | " + " | ".join(pend(r[c['k']]) for c in dec["cols"]) + " |")
    P("")
    P("Three layers: (1) possession transfer itself: every ~1B model is at chance even narrated (kotodama-tsumugi "
      "through 110B, Pythia-1.4B, OLMo-2 1B at 4T); SmolLM2 1.7B at 11T and Qwen3 1.7B at 36T lift off chance; from 4B "
      "it is easy. (2) The transfer as a speech act is the mid-size wall: 4B–70B models that pass the narrated version "
      "stay at .29–.41 spoken; narrating only the hand-off recovers part of the gap. (3) Naming the addressee does not "
      "help: the vocative variant is at or below the spoken one for every public model.")
    P("")
    P("### Findings")
    P("")
    for i, f in enumerate(meta["findings"], 1):
        for a, b in MD_REWRITES.items():
            f = f.replace(a, b)
        P(f"{i}. {md_text(f)}")
    P("")
    P("## Part II · The levers")
    P("")
    P("Each lever measured on 1.4B-parameter arms trained for 4B tokens on one data order; one seed per arm except the "
      "fitted boundaries (three seeds). kotodama-tsumugi (Part III) combines the three positive levers in one model.")
    P("")
    P("| lever | effect | result | cost |")
    P("|---|---|---|---|")
    for l in stage["levers"]:
        P(f"| {l['lever']} | {l['eff']} | {l['res']} | {l['cost']} |")
    P("")
    P("## Part III · kotodama-tsumugi")
    P("")
    P(stage["tsumugi"]["desc"])
    P("")
    P("### Stage gauges against training tokens")
    P("")
    P("`kind` = constant (the trunk, constant learning rate) or annealed (a cooldown branch ending at that token count). "
      "Columns are swap shifts: static *I*, static *you*, dynamic *you* 0-shot and 3-shot.")
    P("")
    P("| seed | kind | tokens (B) | static I | static you | dynamic you 0-shot | dynamic you 3-shot |")
    P("|---|---|---|---|---|---|---|")
    for seed, pts in stage["tsumugi"]["traj"].items():
        for q in sorted(pts, key=lambda q: (q["kind"] != "constant", q["tok"])):
            kind = q["kind"] + (f" ({q['note']})" if q.get("note") else "")
            P(f"| {seed} | {kind} | {q['tok']:g} | {f3(q['I'])} | {f3(q['you'])} | {f3(q['dyn0'])} | {f3(q['dyn3'])} |")
    P("")
    P("For reference, OLMo-2 reaches stage 3 at about " + ", ".join(f"{o['tok']}B tokens ({o['m']})" for o in stage["tsumugi"]["olmo_stage3"]) + ".")
    P("")
    P("### Binding battery across the annealed reads (accuracy)")
    P("")
    bat = stage["tsumugi"]["battery"]
    P("| probe | " + " | ".join(bat["cols"]) + " |")
    P("|---|" + "---|" * len(bat["cols"]))
    for g in bat["groups"]:
        P(f"| *{g['name']}* |" + " |" * len(bat["cols"]))
        for name, vals in g["rows"]:
            P(f"| {name} | " + " | ".join(f"{v:.3f}".lstrip("0") for v in vals) + " |")
    P("")
    P("Scored with one harness (the kotodama binding suite). Each item gets four worked examples, balanced so that none "
      "of them shows the item's own answer pattern; deixis-lite has no such condition and is read zero-shot. The "
      "bookkeeping probes rise with tokens: attribution from .71 to .99, set binding and dialogue tracking by .13 to .17. "
      "Aliased Python lists stays at chance through 33B and lifts only after 55B. The perspective probes stay within .15 "
      "of their 5B level. The two seeds differ at 5B (seed 7 forms set binding earlier) and agree within .05 on seven "
      "of the ten rows at 11B.")
    P("")
    rel = stage["release"]
    P("### The release against its trajectory and its class")
    P("")
    P("| probe | " + " | ".join(rel["cols"]) + " |")
    P("|---|" + "---|" * len(rel["cols"]))
    for name, vals, signed in rel["rows"]:
        P(f"| {name} | " + " | ".join("—" if v is None else f3(v) if signed else (f"{v:.1f}" if v >= 1 else f"{v:.3f}".lstrip("0")) for v in vals) + " |")
    P("")
    P("The two 110B columns are two anneals from the same fork with the same training loss. Their spread (static *you* "
      ".142 vs .235, K50 3.8 vs 4.8, and .025 bits/byte on held-out text) is the measured anneal-to-anneal variance, "
      "and every single-anneal number here carries it. A second initialization seed (seed 7, same data order) was "
      "trained to 11B. Capacity (K50, after arXiv 2609.30634) tracks recipe and tokens more than "
      "parameters on this harness, so no parameter-equivalent size is read from it.")
    P("")
    rid = stage["riders"]
    P("### Object tracking with fixed vs fair exemplars (riders v2, prose; accuracy, chance .25)")
    P("")
    P("| model · tokens | " + " | ".join(rid["cols"]) + " |")
    P("|---|" + "---|" * len(rid["cols"]))
    for r in rid["rows"]:
        P(f"| {r['m']} | " + " | ".join("pending" if r[c] is None else f"{r[c]:.3f}".lstrip("0") for c in rid["cols"]) + " |")
    P("")
    P("fixed3 = the same three exemplars for every item; fairK = K exemplars drawn per item. Fixed exemplars push "
      "accuracy well below zero-shot (a setup artifact). The penalty is not specific to these models. On a lighter "
      "riders slice, fixed exemplars lower prose accuracy for most of an eight-model panel (chat 8 of 8, screenplay 6 "
      "of 8, narrative 5 of 8) and raise code accuracy for all eight. Fair exemplars help every public model here and "
      "kotodama-tsumugi through 55B, but not the two 110B anneals (second anneal −.03, release −.045). The zero-shot "
      "column pools narrative and screenplay prose. Zero-shot riders accuracy depends on the register: on the lighter "
      "slice, it spreads by 20 points or more across the four registers for five of the eight panel models. The public "
      "pool is three 1–2B models; no larger model has been run on this harness.")
    P("")
    P("## How to read the full atlas")
    P("")
    P("- **A cell** = one model (row) on one probe type (column) under one condition, as `acc` = share of items where "
      "the gold answer is the model's top choice among the item's candidates, with `n` items and the `chance` rate "
      "(mean 1/#candidates). `score` = (acc − chance)/(1 − chance); `se` = binomial standard error.")
    P("- **Conditions:** `0shot` = the item alone; `3shot` = three solved items of the test item's own type first (a diagnostic: on aliased lists it teaches the "
      "answer's role; see Caveats); `3shot_collide` "
      "= the three exemplars reuse the test item's names/objects in other roles (interference test).")
    P("- **Modes:** `all` = every item; `free` = shortcut-free subset (deixis columns only): the answer is neither the "
      "last-mentioned candidate nor the last speaker.")
    P("- **Open models** (bases, hybrids, post-trained) are scored by log-probability over the candidate set "
      "(vLLM or HF transformers, never mixed within a row; parity-checked to ±.01). **API models** are scored by "
      "generation: `reflex` rows = single next word with reasoning off (calls that reasoned anyway are excluded); "
      "`deliberate` rows = think, then answer. **Human** row = one person reading items cold, typed answers.")
    P("- **Answer-deixis columns** (`an-*`) use a calibrated read for open models (argmax of logp(item) − logp(body "
      "hidden)); API rows use referent scoring (the person's name counts as correct as well as me/you). Per-role "
      "cells mix skill with each model's standing preference — compare a model's answer columns together.")
    P("- **Deixis twins:** every `I/you/my` item has a `3rd-person twin` with the same events, gold and candidates, "
      "told in third person with names. The gap between the two is the cost of perspective.")
    P("- **Parsing:** sections are `##` headings outside code fences (prompts inside ``` fences can contain lines "
      "that start with `#`). Two CSV blocks at the end: `rows` (id → label, group, size) and `cells` (every number).")
    P("")
    P("## Glossary")
    P("")
    P("| term | meaning |")
    P("|---|---|")
    for t, d in GLOSSARY:
        P(f"| {t} | {d} |")
    P("")
    P("## Caveats")
    P("")
    for g in meta["gaps"]:
        P(f"- {md_text(g)}")
    P("")

    # columns
    P("## Columns")
    P("")
    P("| id | family | label | what it asks | scoring | shortcut-free |")
    P("|---|---|---|---|---|---|")
    for c in cols:
        P(f"| `{c['id']}` | {c['group']} | {c['label']} | {md_text(c['what']).replace('|', '/')} | {c['kind']} | "
          f"{'yes' if c['free_ok'] else '—'} |")
    P("")
    P("## Example items per column (verbatim prompts)")
    P("")
    P("Each prompt ends right before the scored word. `candidates` = the closed set a logprob model ranks; "
      "`gold` = the right answer. Generation-scored (API, human) rows see the same prompt with no options.")
    P("")
    seen_fam: set[str] = set()
    for c in cols:
        e = ex.get(c["id"])
        if not e:
            continue
        fam = c["group"].split(" · ")[0]
        first_of_family = fam not in seen_fam
        seen_fam.add(fam)
        if first_of_family and fam in FAMILY_NOTES:
            P(f"> **{fam}:** {FAMILY_NOTES[fam]}")
            P("")
        P(f"### `{c['id']}` — {c['group']} · {c['label']}")
        P("")
        P(md_text(c["what"]))
        P("")
        gold = ", ".join(e["gold_set"]) if e.get("gold_set") else e["gold"]
        P(f"**0-shot example** (`{e['item_id']}`) — candidates: {', '.join(x.strip() for x in e['candidates'])} · "
          f"gold: **{gold.strip()}**")
        P("")
        P(fence(e["prompt"]))
        P("")
        if e.get("free"):
            fr = e["free"]
            P(f"**Shortcut-free example** — candidates: {', '.join(x.strip() for x in fr['candidates'])} · gold: "
              f"**{fr['gold'].strip()}**")
            P("")
            P(fence(fr["prompt"]))
            P("")
        if e.get("shot3") and first_of_family:
            P(f"**3-shot form** (same test item, three solved exemplars of its own type first; shown once per family — "
              f"every {fam} column uses this layout)")
            P("")
            P(fence(e["shot3"]))
            P("")
        for k, x in enumerate(e.get("pool", [])[:N_POOL], 1):
            g = ", ".join(x["gold_set"]) if x.get("gold_set") else x["gold"]
            P(f"**More item {k}** — candidates: {', '.join(y.strip() for y in x['candidates'])} · gold: **{g.strip()}**")
            P("")
            P(fence(x["prompt"]))
            P("")

    # rows
    P("## Rows")
    P("")
    P("API rows (`or-…`) are all generation-scored; `-A` = reflex frame (single next word, reasoning off), `-B` = "
      "deliberate frame (think, then the word). Notes below list only what differs from that.")
    P("")
    P("| id | label | group | size (B params) | note |")
    P("|---|---|---|---|---|")
    for r in rows:
        note = md_text(r.get("note") or "")
        for frag in API_NOTE:
            note = note.replace(frag, "")
        note = note.strip("; ").replace("|", "/")
        P(f"| `{r['id']}` | {r['label']} | {r['group']} | {r['size'] if r['size'] is not None else '—'} | {note} |")
    P("")

    # wide tables, 0-shot, free where available
    fams: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in cols:
        fam = c["group"].split(" · ")[0]
        fams.setdefault(fam, []).append(c)
    P("## 0-shot tables by family")
    P("")
    P(f"Accuracy × 100, 0-shot; deixis columns are the **shortcut-free** subset. `—` = not measured; `†` = fewer than "
      f"{SMALL_N} items in that cell (all human cells; many API cells). Rows are grouped (bold separator rows); API "
      "rows are generation-scored, open rows logprob-scored. Full numbers (n, chance, se, every condition and mode) "
      "are in the CSV below.")
    P("")
    for fam, cs in fams.items():
        P(f"### {fam}")
        P("")
        if fam == "Deixis":
            P("Key: owner = 'Now the X is with' (P3); self = 'I'm holding the'; you = 'you're holding the'; 2/3 = "
              "people in the chat (you3 has a vocative, you2 is structural); I = the I/you/my version, 3rd = its "
              "3rd-person twin.")
            P("")
        heads = [SHORT.get(c["id"]) or (f"{c['group'].split(' · ', 1)[-1]} {c['label']}" if c["group"] != fam
                                        else c["label"]) for c in cs]
        P("| model | " + " | ".join(heads) + " |")
        P("|---|" + "---|" * len(cs))
        last_group = None
        for r in rows:
            vals = []
            for c in cs:
                x = cell(cells, r["id"], c["id"], "0shot", "free" if c["free_ok"] else "all")
                vals.append("—" if x is None else f"{x['acc'] * 100:.0f}{'†' if x['n'] < SMALL_N else ''}")
            if all(v == "—" for v in vals):
                continue
            if r["group"] != last_group:
                P(f"| **{r['group']}** |" + " |" * len(cs))
                last_group = r["group"]
            P(f"| {r['label']} | " + " | ".join(vals) + " |")
        ch = []
        for c in cs:
            xs = [v["chance"] for k, v in cells.items() if k.split("|")[1] == c["id"] and k.split("|")[2] == "0shot"]
            ch.append(f"{100 * sum(xs) / len(xs):.0f}" if xs else "—")
        P("| *chance* | " + " | ".join(ch) + " |")
        P("")

    # speaker swap
    if swap:
        P("## Speaker-swap test (open bases)")
        P("")
        P("Two-person chats; the prompt is repeated with ONLY the name on the final query line changed, which flips the "
          "right answer. `shift` = mean P(original gold | original asker) − P(original gold | swapped asker), "
          "candidates renormalised; 95% CI by paired bootstrap over 180 items. `> 0` = the answer follows the speaker "
          "(binds); `≈ 0` = ignores who is speaking; `< 0` = moves the wrong way (inverts).")
        P("")
        P("**Reading rule:** `binds` if the CI's lower bound is above +.02; `inverts` if its upper bound is below −.02; "
          "otherwise `ignores speaker` (no reliable movement — a CI can exclude 0 and still be read this way).")
        P("")
        if SWAP_PROMPTS is not None and SWAP_PROMPTS.exists():
            pr = [json.loads(l) for l in SWAP_PROMPTS.read_text().splitlines()[:400]]
            pair = [r for r in pr if r["qkind"] == "addressee" and r["condition"] == "0shot"][:2]
            if len(pair) == 2:
                P(f"Example pair (original gold **{pair[0]['gold_orig']}**; under the swap it is held by the new asker):")
                P("")
                P(fence(pair[0]["prompt"]))
                P("")
                P(fence(pair[1]["prompt"]))
                P("")
        label = {r["id"]: r["label"] for r in rows}
        P("| model | I 0-shot | I 3-shot | you 0-shot | you 0-shot 95% CI | you 0-shot reading | you 3-shot |")
        P("|---|---|---|---|---|---|---|")
        for m in sorted(swap):
            v = swap[m]
            a = v["0shot/addressee"]
            P(f"| {label.get(m, m)} | {v['0shot/self']['shift']:+.3f} | {v['3shot/self']['shift']:+.3f} | {a['shift']:+.3f} | "
              f"[{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['reading']} | {v['3shot/addressee']['shift']:+.3f} |")
        P("")

    # long CSV
    P("## Rows (CSV)")
    P("")
    rl = ["row_id,label,group,size_b,scoring"]
    for r in rows:
        sc = "generation" if r["id"].startswith("or-") or r["id"] == "human" else "logprob"
        q = lambda x: '"' + str(x).replace('"', "'") + '"'  # noqa: E731
        rl.append(f"{r['id']},{q(r['label'])},{q(r['group'])},{r['size'] if r['size'] is not None else ''},{sc}")
    P(fence("\n".join(rl), "csv"))
    P("")
    P("## All cells (CSV)")
    P("")
    P("`row_id,col_id,condition,mode,acc,n,chance,se,seat_form,errs` — `score` (= (acc − chance)/(1 − chance)) is "
      "left out as derivable; `errs` = error-type shares among items (deixis I/you columns; `;`-separated "
      "`type=share`, types in the glossary); `seat_form` = share of referent-correct API answers that used me/you.")
    P("")
    lines = ["row_id,col_id,condition,mode,acc,n,chance,se,seat_form,errs"]
    order = {r["id"]: i for i, r in enumerate(rows)}
    corder = {c["id"]: i for i, c in enumerate(cols)}
    for k in sorted(cells, key=lambda k: (order.get(k.split("|")[0], 1e9), corder.get(k.split("|")[1], 1e9), k)):
        rid, cid, cond, mode = k.split("|")
        if rid not in order:
            continue
        v = cells[k]
        errs = ";".join(f"{a}={b}" for a, b in sorted((v.get("errs") or {}).items()))
        lines.append(f"{rid},{cid},{cond},{mode},{v['acc']:.3f},{v['n']},{v['chance']:.3f},{v['se']:.3f},"
                     f"{v.get('seat_form', '')},{errs}")
    P(fence("\n".join(lines), "csv"))
    P("")
    text = "\n".join(L)
    bad = sorted({m.group(0) for m in FORBIDDEN.finditer(text)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public Markdown: {bad}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}: {len(text) / 1e6:.2f} MB; {len(rows)} rows, {len(cols)} cols, {len(lines) - 1:,} CSV cells"
          + (f", swap {len(swap)} models" if swap else ""))


if __name__ == "__main__":
    main()
