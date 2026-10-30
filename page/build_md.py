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
      "each chat rewritten as a scene, for all 8 models tested at or below 8B (the highest is Qwen2.5-7B, +.010). So the "
      "result does not come from the chat format. In these models depth grows with size, so the table leaves two "
      "routes open for small models: more tokens, and more depth at a fixed size. No model at or below 8B has been "
      "trained far enough to rule out the first. From 14B to 1T parameters, stage 5 is weak and nearly flat. Ten of the 11 bases "
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
    P("The rewrites show three layers. (1) The first is possession transfer itself. Every 1.4–1.5B model is at or near "
      "chance even when the transfer is narrated (kotodama-tsumugi through 110B, Pythia-1.4B, OLMo-2 1B at 4T). "
      "SmolLM2 1.7B at 11T and Qwen3 1.7B at 36T lift off chance, and from 4B the narrated transfer is easy. (2) The "
      "transfer as a speech act is the mid-size wall. Models from 4B to 70B that pass the narrated version stay at "
      ".29–.41 when the transfer is spoken. Narrating only the hand-off recovers part of the gap. (3) Naming the "
      "addressee does not help: the vocative variant is at or below the spoken one for every public model.")
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
    P("By the release, attribution reads .986, level with the 7–8B bases under the same balanced examples (.981 to "
      "1.00). Scored with one harness (the kotodama binding suite). Each item gets four worked examples, balanced so that none "
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
      "trained to 11B. Capacity (K50, after arXiv 2609.30634) is how many name–task pairs fit in context before recall "
      "falls halfway to chance. On this harness it tracks recipe and tokens more than parameters, so no "
      "parameter-equivalent size is read from it.")
    P("")
    rid = stage["riders"]
    P("### Object tracking with fixed vs fair exemplars (riders v2, prose; accuracy, chance .25)")
    P("")
    P("| model · tokens | " + " | ".join(rid["cols"]) + " |")
    P("|---|" + "---|" * len(rid["cols"]))
    for r in rid["rows"]:
        P(f"| {r['m']} | " + " | ".join("pending" if r[c] is None else f"{r[c]:.3f}".lstrip("0") for c in rid["cols"]) + " |")
    P("")
    P("Zero-shot, the annealed kotodama-tsumugi checkpoints from 33B on (.455–.490) sit above every public checkpoint "
      "measured on this harness (best: SmolLM2 1.7B at 8.4T, .420). The zero-shot column pools narrative and screenplay "
      "prose, and the public pool is three 1–2B models. fixed3 = the same three exemplars for every item; fairK = K "
      "exemplars drawn per item. Fixed exemplars push accuracy well below zero-shot (a setup artifact). The penalty is "
      "not specific to these models. On a lighter riders slice, fixed exemplars lower prose accuracy for most of an "
      "eight-model panel (chat 8 of 8, screenplay 6 of 8, narrative 5 of 8) and raise code accuracy for all eight. Fair "
      "exemplars help every public model here and kotodama-tsumugi through 55B. The two 110B anneals score highest "
      "zero-shot (fair6: second anneal −.03, release −.045).")
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
