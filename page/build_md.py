"""Render the agent-friendly Markdown twin of the Binding Atlas: same data as binding-atlas.html, as plain text.

Usage: python page/build_md.py --atlas data/built/atlas.json --stage data/built/stage.json --swap data/built/swap.json
                               --out site/binding-atlas.md [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
                               [--swap-prompts swap-prompts.jsonl]   (optional: prints one example speaker-swap pair)
Sections: summary · Parts I–III · how to read · glossary · columns (+ verbatim example prompts per column) · rows ·
tables by group · rows and every cell as CSV.
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
    P(f"This atlas scores {sum(1 for r in rows if r['group'] != 'Human')} models and checkpoints, plus one human reader, "
      f"on {len(cols)} tests of tracking who *I* and *you* refer to. \"Example items per column\" quotes the exact prompts "
      "that the models were given.")
    P("")
    P(stage["lead"])
    P("")
    P("Swap shift = mean[p(gold | original) − p(gold | speaker swapped)] over the candidates. Above +.02, a pronoun "
      "binds: its answer moves with the speaker. Below −.02, it inverts: it resolves to the speaker.")
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
    P("### Dynamic *you* at each model's final checkpoint (OLMo-2 1B: its 4T pre-anneal checkpoint; zero-shot from about "
      "7B, every base from 13B up)")
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
    P("Zero-shot, no model up to 4B binds, and four of the small Qwen models invert. At 7B, OLMo-2 7B (+.055) and "
      "Qwen2.5 7B (+.047) bind; Qwen3 8B and Gemma-2 9B do not yet. From 13B up, every base binds: all 19 measured, "
      "here and in the full atlas, from OLMo-2 13B to DeepSeek-V4 Pro at 1.6T. The largest readings are DeepSeek-V3 "
      "(+.216), DeepSeek-V4 Flash (+.177) and MiMo-V2.5 Pro (+.162). Static *you* reads +.27 to +.89 across these "
      "19. With three worked examples, binding starts at Qwen3 4B (+.055) and reaches +.32 for Qwen3 14B. The "
      "kotodama-tsumugi release reads −.014 zero-shot and +.038 with three examples, which binds; its second anneal "
      "reads +.022 with examples, just short of the line.")
    P("")
    P("**The binding test.** A swap shift binds when its whole 95% interval lies above +.02. The bar is a fixed "
      "convention, set before any of these models was read. Under it, kotodama-tsumugi binds with three worked "
      "examples (+.038, interval +.023 to +.055). It is the only base at 3B or below that does; the next is Qwen3 4B "
      "(+.055). The finding depends on the bar. With the bar at +.01 it still holds. With the bar at 0, Pythia-160M and Pythia-2.8B "
      "also pass, and at +.03 kotodama-tsumugi no longer does. The bench's roadmap derives the bar from each "
      "battery's own noise, registered before any model is read.")
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
    P("The rewrites show three layers. (1) The first is possession transfer itself. The 1.4–1.5B models sit near "
      "chance (.28–.38) even when the scene is narrated. These are Pythia-1.4B, OLMo-2 1B at 4T, and "
      "kotodama-tsumugi at every checkpoint through 110B tokens. "
      "SmolLM2 1.7B at 11T (.40) and Qwen3 1.7B at 36T (.48) lift off chance; from 4B the narrated version is easy "
      "(.53–.79). (2) The transfer as a speech act is the mid-size wall. Models from 4B to 30B stay at .37–.47 when it "
      "is spoken, and Llama-3.1 70B reaches .56. Narrating only the hand-off recovers part of that gap for every model "
      "from 4B up, and all of it for Llama-3.1 70B. (3) Naming the addressee does not help: the vocative variant is "
      "never more than .01 above the spoken one, and below it for every model from 4B up.")
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
    P("### The grid's headline tests across kotodama-tsumugi's checkpoints")
    P("")
    bat = stage["tsumugi"]["battery"]
    P("| probe | " + " | ".join(bat["cols"]) + " |")
    P("|---|" + "---|" * len(bat["cols"]))
    for g in bat["groups"]:
        P(f"| *{g['name']}* |" + " |" * len(bat["cols"]))
        for name, vals in g["rows"]:
            P(f"| {name} | " + " | ".join(f3(v) for v in vals) + " |")
    P("")
    P("The values are the grid's own cells. Each test is read as the grid reads it. Columns with worked examples use "
      "them, the others are read cold, and the rule-learning rows are AUC over the example ladder. Lookup rises from .69 at 5B tokens to "
      ".99 by 55B, level with the 7–9B bases (.97 to 1.00). The update tests (many refills, variable chains, shared "
      "lists) stay at chance through 33B and lift from 55B on (.04 to .22). Static *you* climbs from .08 at 5B to .49 "
      "at the release. Pattern completion reaches .55, second of 12 models at 1–2B parameters. The two seeds differ at "
      "5B (seed 7 forms lookup and set binding earlier) and agree within .05 on all ten rows at 11B.")
    P("")
    rel = stage["release"]
    P("### The release against its trajectory and its class")
    P("")
    P("| probe | " + " | ".join(rel["cols"]) + " |")
    P("|---|" + "---|" * len(rel["cols"]))
    for name, vals, signed in rel["rows"]:
        P(f"| {name} | " + " | ".join("—" if v is None else f3(v) if signed else (f"{v:.1f}" if v >= 1 else f"{v:.3f}".lstrip("0")) for v in vals) + " |")
    P("")
    P("The two 110B columns are two anneals from the same fork with the same training loss. Their spread is "
      "the measured anneal-to-anneal variance. It is .127 vs .192 on static *you* and .022 vs .038 on dynamic *you* "
      "with examples. K50 reads 4.6 vs 5.1, and held-out text differs by .025 bits/byte. "
      "Every single-anneal number here carries it. A second initialization seed (seed 7, same data order) was "
      "trained to 11B. Capacity (K50, after arXiv 2609.30634) is how many name–task pairs fit in context before recall "
      "falls halfway to chance. On this harness it tracks recipe and tokens more than parameters, so no "
      "parameter-equivalent size is read from it.")
    P("")
    rid = stage["riders"]
    P("### Object tracking: lookup early, updating later (the box puzzle; κ, mean over code, prose, script and chat)")
    P("")
    P("| models | " + " | ".join(c["name"] for c in rid["cols"]) + " |")
    P("|---|" + "---|" * len(rid["cols"]))
    for r in rid["rows"]:
        P(f"| {r['m']} | " + " | ".join("—" if r[c['k']] is None else f3(r[c['k']]) for c in rid["cols"]) + " |")
    P("")
    P("On *lookup* items, the queried box ends with its declared contents; on *update* items, an "
      "operation changed it. Each public row is the median model of its size band. Lookup is above chance at every "
      "size. The median model updates at chance below 2B. With worked examples, updating climbs with size, from .20 "
      "at 2–5B and .40 at 7–10B to .77 at 70B and .87 from 284B up. A few small models update early, each with "
      "examples: Qwen3 4B (.43), Falcon-H1 3B (.37) and Falcon-H1 1.5B-Deep (.28). kotodama-tsumugi's lookup with "
      "examples (.61) is second of 12 models at 1–2B, behind Llama-3.2 1B (.83); like most models its size, it does "
      "not yet update (−.04). One model's cell averages 272 lookup or 328 update items (68 or 82 per format), about "
      "±.08 at 95%.")
    P("")
    P("## How to read the full atlas")
    P("")
    P("- **A cell** = one model (row) on one test (column) under one condition. `score` averages, over items, "
      "(credit − chance)/(1 − chance), with `chance` = 1/#candidates. `credit` is 1 when the gold is the model's "
      "top-ranked candidate, and an exact tie between candidates splits the credit evenly. 0 = guessing, 1 = always "
      "right, below 0 = reliably wrong. `lo`, `hi` = the bootstrap 95% interval (1,000 resamples of items).")
    P("- **State:** `above` when the interval is above 0, and `at` when it includes 0. `below` means the interval is "
      "below 0: the model picks a wrong answer more often than guessing would.")
    P("- **Conditions:** `fair` = four solved items first, balanced so that no answer can be copied from them (the "
      "reading); `0shot` = the item alone (the floor). Columns without worked examples (lookups in lists, box refills, the "
      "static gauges) have a `0shot` reading only. The rule-learning columns are read across their difficulty ladder "
      "(`ladder`: the mean chance-normalized accuracy over the ladder's rungs, clipped at 0 and 1).")
    P("- **Wording controls:** the word *now* in a query implies that something moved. The largest models read that "
      "implication. After a scene where nothing moves, 'Now X is holding the' leads them to answer as if something had. "
      "Every query here uses one lead per test, whether or not anything moved: 'Right now X is holding the', 'So Box G "
      "contains the'. The wording therefore never signals a change.")
    P("- **Groups:** columns are grouped by what the questions look like. Each column is scored and read on its own.")
    P("- **Models:** every base model is scored by log-probability over the candidate set, with one scorer and one set "
      "of batteries. The **Human** row is one person reading items cold, with typed answers.")
    P("- **Parsing:** sections are `##` headings outside code fences (prompts inside code fences can contain lines "
      "that start with `#`). Two CSV blocks at the end: `rows` (id → label, group, size, tokens) and `cells` (every number).")
    P("")
    P("## Glossary")
    P("")
    P("| term | meaning |")
    P("|---|---|")
    for t, d in GLOSSARY:
        P(f"| {t} | {d} |")
    P("")
    # columns
    P("## Columns")
    P("")
    P("| id | group | label | what it asks | readings | headline |")
    P("|---|---|---|---|---|---|")
    for c in cols:
        P(f"| `{c['id']}` | {c['group']} | {c['label']} | {md_text(c['what']).replace('|', '/')}"
          f"{' (' + md_text(c['flag']).replace('|', '/') + ')' if c.get('flag') else ''} | {', '.join(c.get('conds', []))} | "
          f"{'yes' if c.get('headline') else ''} |")
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
        P(f"**0-shot example** (`{e['item_id']}`) — candidates: `{', '.join(x.strip() for x in e['candidates'])}` · "
          f"gold: **`{gold.strip()}`**")
        P("")
        P(fence(e["prompt"]))
        P("")
        if e.get("free"):
            fr = e["free"]
            P(f"**Shortcut-free example** — candidates: `{', '.join(x.strip() for x in fr['candidates'])}` · gold: "
              f"**`{fr['gold'].strip()}`**")
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
            P(f"**More item {k}** — candidates: `{', '.join(y.strip() for y in x['candidates'])}` · gold: **`{g.strip()}`**")
            P("")
            P(fence(x["prompt"]))
            P("")

    # rows
    P("## Rows")
    P("")
    P("| id | label | group | params (B) | active (B) | pretraining tokens | note |")
    P("|---|---|---|---|---|---|---|")
    for r in rows:
        tok = (f"{r.get('tokens_mark', '')} {r['tokens'] / 1e12:g}T".strip() if (r.get("tokens") or 0) >= 1e12
               else f"{r['tokens'] / 1e9:g}B" if r.get("tokens") else r.get("tokens_mark") or "—")
        note = md_text(r.get("note") or "").replace("|", "/")
        P(f"| `{r['id']}` | {r['label']} | {r['group']} | {r.get('size') if r.get('size') is not None else '—'} | "
          f"{r.get('active') if r.get('active') is not None else '—'} | {tok} | {note} |")
    P("")

    # wide tables: score per group, the reading (fair) where a column has one, else its own reading
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for c in cols:
        groups.setdefault(c["group"], []).append(c)
    P("## Tables by group")
    P("")
    P("Score (0 = guessing, 1 = always right), with worked examples where the column has them; `°` = the column's "
      "only reading (cold, or across its ladder); `↓` = below chance (the interval is below 0); `—` = not measured. "
      "Full numbers (n, chance, the interval, every condition) are in the CSV below.")
    P("")
    for gname, cs in groups.items():
        P(f"### {gname}")
        P("")
        P("| model | " + " | ".join(c["label"] for c in cs) + " |")
        P("|---|" + "---|" * len(cs))
        last_group = None
        for r in rows:
            vals = []
            for c in cs:
                own = c.get("conds") or ["0shot"]
                k = "fair" if "fair" in own else own[0]
                x = cell(cells, r["id"], c["id"], k, "all") or cell(cells, r["id"], c["id"], "0shot", "all")
                if x is None:
                    vals.append("—")
                    continue
                vals.append(f"{x['score']:+.2f}{'°' if k != 'fair' else ''}{'↓' if x.get('state') == 'below' else ''}")
            if all(v == "—" for v in vals):
                continue
            if r["group"] != last_group:
                P(f"| **{r['group']}** |" + " |" * len(cs))
                last_group = r["group"]
            P(f"| {r['label']} | " + " | ".join(vals) + " |")
        P("")

    # long CSV
    P("## Rows (CSV)")
    P("")
    rl = ["row_id,label,group,params_b,active_b,tokens,scoring"]
    for r in rows:
        sc = "generation" if r["id"] == "human" else "logprob"
        q = lambda x: '"' + str(x).replace('"', "'") + '"'  # noqa: E731
        rl.append(f"{r['id']},{q(r['label'])},{q(r['group'])},{r['size'] if r.get('size') is not None else ''},"
                  f"{r.get('active') if r.get('active') is not None else ''},{r.get('tokens') or ''},{sc}")
    P(fence("\n".join(rl), "csv"))
    P("")
    P("## All cells (CSV)")
    P("")
    P("`row_id,col_id,condition,acc,n,chance,score,lo,hi,state` — `acc` and `chance` are empty for ladder readings.")
    P("")
    lines = ["row_id,col_id,condition,acc,n,chance,score,lo,hi,state"]
    order = {r["id"]: i for i, r in enumerate(rows)}
    corder = {c["id"]: i for i, c in enumerate(cols)}
    for k in sorted(cells, key=lambda k: (order.get(k.split("|")[0], 1e9), corder.get(k.split("|")[1], 1e9), k)):
        rid, cid, cond, mode = k.split("|")
        if rid not in order:
            continue
        v = cells[k]
        f3 = lambda x: "" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"{rid},{cid},{cond},{f3(v.get('acc'))},{v['n']},{f3(v.get('chance'))},{f3(v.get('score'))},"
                     f"{f3(v.get('lo'))},{f3(v.get('hi'))},{v.get('state') or ''}")
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
