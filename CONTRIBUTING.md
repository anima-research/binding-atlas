# Contributing

Two kinds of contribution arrive here, and they are handled differently.

**Code and page changes** are a pull request against this repository. It merges when the gates below pass.

**Claims** are not code. A contribution that asserts a finding arrives as an issue with the runs behind it, for
example that a model binds better or that a number holds. The page states a claim only after the run behind it has
been checked. The numbers come from binding-bench. Its rules for batteries, readings and pre-registration govern
every number that this page shows.

## How this repository is organised

Each fact has one home, and every other place reads it from there:

- the grid (rows, columns, cells, example prompts): `data/built/atlas.json`, built by `build/build_atlas_v4.py`;
- the numbers above the grid (figures, tables, the lead): `data/built/stage.json`, refreshed by `build/build_stage.py`
  from `data/inputs/synthesis.jsonl`;
- the page's layout and interaction: `page/atlas_page.html`; the Markdown twin: `page/build_md.py`;
- every absolute URL: the hosting configs in `config/`;
- the names that must never reach a public file: `build/forbidden.py`;
- the human reader at `/take`: `take/`.

A number written as a literal in a second place drifts from its source. Prose that quotes a grid count (the byline,
the card, the first screen) reads it from the build, or is checked against it before a deploy.

## The page's own rules

- **Lead with what was found.** A page reads confident and every number is true and traceable to its run. A caveat
  appears only when leaving it out would make a claim false. It then appears once, lightly, as a property of a size
  class or of the field. The full chain of caveats lives in the research records.
- **Third person, timeless present.** A page never narrates its own edits and carries no dates in its prose.
- **Columns are grouped by what the questions look like.** A group is never presented as one underlying ability: the
  claim that these tests separate into distinct constructs was tested, and it did not replicate. Each column is read on
  its own.
- **The few-shot reading is `fair`, and 0-shot is the floor.** `fair` uses four worked examples, balanced so that no
  answer can be copied from them.
- **Scores split exact ties.** An exact tie between candidates is no evidence for either of them, so its credit is
  shared evenly.
- **Published wording changes are approved first.** A change to text that a reader sees goes to the page's
  owner before it is deployed. The credit line stays under every title: "By Luxia (@kataletheia), with Claude (Anthropic) · compute
  and hosting by Anima Labs".

## The documentation rules

Two rules hold for comments, for docstrings, and for the messages that a `raise` or a logging call prints. Both are
checked mechanically.

1. **State what is true now.** A comment describes the code as it stands. It does not narrate how the code got there,
   and it carries no dates. There is no bare `TODO`, `FIXME` or `HACK`.
2. **Every referent is reachable from this repository.** A comment may name a module here, a file in this tree, a
   symbol that this repository defines, or a public URL. A private path, a planning document, a conversation, a machine, a
   host or a sibling repository is not a referent. A placeholder is written as one (`<out>.html`), so nobody mistakes it
   for a file.

## Pages that a stranger reads

The README, this file and the Markdown twin are read by a stranger, and so is the page itself. Each page is held to
stay at least as legible as it is. `tools/check_legibility.py` counts the features that make prose hard to follow on a
first read. They are long sentences, long distances between linked words, object relatives without "that", stacked
nouns, things as the subjects of mind verbs, and nouns used as verbs. The features come from Clark, Martin & Mahowald,
[*Why is AI-generated language so hard to understand?*](https://arxiv.org/abs/2610.29162), and each rate is compared
with English Wikipedia. `tools/legibility_baseline.json` records each page's numbers. A page fails when a feature rises
more than ten percent above its baseline, plus one occurrence. The check runs in an environment of its own:

```bash
uv venv .venv-legibility && uv pip install --python .venv-legibility -r tools/legibility_requirements.txt
.venv-legibility/bin/python -m tools.check_legibility                            # every public page
.venv-legibility/bin/python -m tools.check_legibility --page README.md --flags   # one page, every flagged sentence
```

A page that improves passes, and `--update-baseline` writes the new level in. A regression enters the baseline only
with a reason: `--update-baseline --reason "..."`.

## The gates

A change merges when these pass. Each is a command that runs from the repository root.

| gate | what it checks | command |
|---|---|---|
| build | the deployed page, twin and card rebuild byte for byte | `make verify` |
| stage | `stage.json` rebuilds from the tracked synthesis rows | `make stage-check` |
| guard | no tracked file holds a hostname, machine path, session id or internal run name | `make guard` |
| documentation | the two rules above, and the legibility ratchet | `make gates` |
| data | `atlas.json` rebuilds from the zoo sweep's rows | `make data-check ZOO=… PROMPTS=… BENCH=…` |
| take | the human reader's service | `take/.venv/bin/python -m pytest take/test_server.py` |

Both allowlists in `tools/` carry a reason above every entry. A gate is waived only in writing, with the reason beside
the receipt.

## Deploying

A deploy is made from a worktree. The page is built with the deployment config, and its pins in
`config/animalabs.sha256` and the `VERIFY_STAMP` in the `Makefile` are updated together. The gates pass. The page is
checked in a browser at 412 px and 1300 px, in light and dark, and it is never wider than the viewport. The live files
are backed up before they are replaced, and after the copy the live files' sha256 must match the build.
