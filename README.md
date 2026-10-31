# Binding Atlas

Base language models learn who *you* is in five stages. No public model up to 8B reaches the last one zero-shot, at
any token count measured, and from 14B to 1T parameters it stays weak. The Binding Atlas scores every base model with
one scorer on one set of tests, chance-normalized. Each column shows a verbatim example prompt, and a plain-text
Markdown twin carries the same data as CSV.

The grid covers 45 public base models and one human reader. The figures cover 30 public base models and 5 public
training trajectories. The atlas also follows **kotodama-tsumugi**, a 1.4B model run as 72 effective layers (looped depth,
boundary-fitted attention residuals). kotodama-tsumugi reaches stage 3 by 33B tokens, about what OLMo-2 13B needs,
where OLMo-2 1B needs about 420B. At 110B tokens its zero-shot dynamic *you* is not inverted (−.021), where most
1–2B bases invert, and it follows a narrated hand-off better than every Pythia model up to 6.9B.

## Threads

A second page, `threads.html` (*Threads*; 紬 tsumugi, spun thread), maps every FFN neuron of kotodama-tsumugi: 72 effective positions × 4,096 = 294,912
neurons. Each one is placed by the meaning of its strongest firing contexts in 24M training tokens, by depth, or by its
weights. Language-model agents wrote the groups and labels. They graded each label by how well three readings agree: the
contexts that a neuron fires on, the tokens that its output favours (logit lens and J-lens), and what steering it does.
The labels are drafts. The page also shows the referent-binding circuit. It gives each neuron's attribution of the
battery margin on bare items, and held-out knock-outs of the supporter and interferer sets. It also shows which parts of
the circuit hold when the items come with balanced worked examples: 246 neurons stay in a top-100 set under both
formats. The page loads its data at run
time from `threads-data/` beside it. The directory is named differently from the page on purpose: some hosts try a path
before adding `.html`, and a directory with the page's name would hide the page. `neuron-atlas.html`, the page's earlier address, is built beside it as a redirect:

| file | what |
|---|---|
| `neurons.json` | manifest: families, categories, groups (draft names), draft neuron labels, the hand-validated neurons, layout bounds, column offsets |
| `neurons.bin` | per-neuron columns (family, group, binding attribution per task as signed int8, causal-set bits) |
| `layout-<name>.bin` | x then y as uint16 for one layout: `meaning`, `masked` (firing token hidden), `depth`, `weights-read`, `weights-write`, `weights-write-pca` |
| `binding.json` | attribution by position and token role, the causal sets, the knock-out matrices and per-cell results |

```bash
make neurons                               # page + data dir into site/ (CONFIG= and SITE= as for make page)
make neurons-binding                       # data/inputs/binding/*.jsonl -> data/built/neurons/binding.json
```

`build/build_neurons.py` writes everything in `data/built/neurons/` except `binding.json`, from the compendium build (its
inputs are named in its docstring).

## Take it yourself

A third page, `/take`, lets people answer the same items that the models are scored on. With enough readers, the
atlas's single human row becomes an estimate per column. Each item is a fresh draw from the generator behind a scored
battery, with a seed that the battery never used. It is rendered exactly as a model sees it. Readers arrive by
invitation and stay anonymous. The service keeps their answers, their response times and a random id from the
browser, and no IP address or account.

Each reader answers in one of two modes, drawn at random: they pick among the model's candidates, or they type a word,
which is matched to the candidates. Readers see items without worked examples first, and items with examples only
after that. Nobody sees two variants of one story. Each new item comes from the slice with the fewest answers so far,
so coverage fills evenly. If a browser reports automation, or a reader ticks "an automated agent is answering", their
answers are flagged and kept out of the human estimate.

| file | what |
|---|---|
| `take/build_pool.py` | builds the item pool offline from `binding_bench` generators and the kotodama commit that drew each battery |
| `take/server.py` | the service (FastAPI, SQLite): invites, serving order, answers, summaries, agent reports |
| `take/static/take.html` | the page |
| `take/make_invite.py`, `take/backup.py` | an invite link (only its hash is stored); an online copy of the database |
| `take/deploy/` | the systemd units, the nightly backup timer and the reverse-proxy snippet |

```bash
uv venv take/.venv && uv pip install --python take/.venv "fastapi>=0.110" "uvicorn>=0.29" "httpx>=0.27" "pytest>=8"
take/.venv/bin/python -m pytest take/test_server.py
python take/build_pool.py --kotodama KOTODAMA_CLONE --bench BINDING_BENCH --seeds 900001 --out take/data/pool.jsonl
TAKE_POOL=take/data/pool.jsonl TAKE_DB=take/data/take.sqlite TAKE_INVITES=take/data/invites.txt \
  take/.venv/bin/uvicorn take.server:app --port 7480
```

The pool stays out of the repository (`take/data/` is ignored), so the human items are not published.
On a server, build the venv on the system Python (`uv venv --python /usr/bin/python3 take/.venv`). A venv on a
uv-managed Python links into the home directory of whoever built it, and the service user cannot run it.

## Layout

| path | what |
|---|---|
| `site/` | the built site, ready to serve: `binding-atlas.html`, `binding-atlas.md`, `binding-atlas-card.png`; `threads.html` + `threads-data/` + `threads-card.png` (+ the `neuron-atlas.html` redirect) |
| `page/` | page builders: `build_page.py` (HTML), `build_md.py` (Markdown twin), `build_card.py` + `build_card.sh` (link card), `atlas_page.html` (template), `siteconfig.py` (hosting config) |
| `build/` | data builders: `build_atlas_v4.py` (the zoo sweep's score rows → `atlas.json`), `build_stage.py` (synthesis table → `stage.json`), `facts.py`; `subset_synthesis.py` (cuts `data/inputs/synthesis.jsonl`), `forbidden.py` (the public-content guard) |
| `take/` | the human reader at `/take` (see above) |
| `tools/` | the documentation gates (see `CONTRIBUTING.md`) |
| `data/built/` | what the page shows: `atlas.json`, `stage.json` |
| `data/inputs/` | small tracked inputs: `synthesis.jsonl` (the rows of the synthesis table `build_stage.py` reads) and the human gauntlet items |
| `data/DROPPED-2026-10.tsv` | every cell of the previous edition's grid, with what happened to it |
| `config/` | hosting configs (`default.toml` neutral, `animalabs.toml` for kotodama.animalabs.ai, `aetherawi.toml` for the earlier host), `animalabs.sha256` |

## Building

### The page (`data/built` → `site/`)

```bash
make page                                  # neutral config: relative URLs, no canonical link
make page CONFIG=config/animalabs.toml     # a deployment config (see below)
make page SITE=/some/dir STAMP="2026-01-31 12:00 UTC"
```

The page and the Markdown twin need only Python ≥ 3.11 (standard library). The link card also needs Pillow and
Google Chrome (`PY=` a python with Pillow, `CHROME=` the browser binary). `STAMP` is the "atlas build" line in the
footer (default: now, UTC); its date is the "updated" line.

### The data (`data/built`)

```bash
make stage                                 # re-read every number in data/built/stage.json from data/inputs/synthesis.jsonl
make data ZOO=<sweep rows> PROMPTS=<battery files> BENCH=<binding-bench checkout> PY_DATA=<python with numpy>
```

The grid comes from the zoo sweep's score rows: `suite/<battery>/<model>.jsonl` and `riders/rows_<model>.jsonl`, with
the sweep's model table and report beside them. `build_atlas_v4.py` reads every row and scores each cell, with split credit
for exact ties. It adds each cell's bootstrap interval and state, and it renders one example prompt per column with
binding-bench's own payload builder. The batteries' prompt files come from binding-bench's prompts directory.

The previous edition's grid read other score files. They stay in the public dataset
[`LuxiaSL/binding-atlas-grids`](https://huggingface.co/datasets/LuxiaSL/binding-atlas-grids) at revision `b45667b7`,
and `data/DROPPED-2026-10.tsv` lists every cell that came from them. Each cell carries one of three marks: a column of
the new grid replaced it, the new grid has no such column, or the clean bench has not scored that row yet.

### Equivalence checks

| command | checks |
|---|---|
| `make verify` | builds with `config/animalabs.toml` and the committed build stamp into `out/verify/`; the HTML, Markdown twin and card must match `config/animalabs.sha256` (the files in `site/`) byte for byte |
| `make stage-check` | `stage.json` rebuilt from the tracked synthesis rows equals `data/built/stage.json` |
| `make guard` | no tracked file contains an internal hostname, machine path, session id or internal run / checkpoint name (`build/forbidden.py`; the page builders apply the same guard to their output) |
| `make data-check ZOO=… PROMPTS=… BENCH=…` | `atlas.json` and `stage.json` rebuilt from their inputs equal `data/built/` |
| `make check` | `verify` + `stage-check` + `guard` (no score rows needed) |
| `make gates` | the documentation gates of `CONTRIBUTING.md`: timelessness, referents and legibility |

## Hosting

The site is static. Serve these files from one directory, under any path prefix:

```
binding-atlas.html        the page (all data inline)
binding-atlas.md          the Markdown twin, linked from the page
binding-atlas-card.png    the link card (og:image / twitter:image)
```

Every URL inside the page is relative, so the directory can sit at any prefix. The page is set in Courier Prime,
loaded from Google Fonts, with the system monospace as fallback. Absolute URLs (canonical link, `og:url`, the card's `og:image`) come only from the hosting config:

```toml
[site]
name = "example.org"                       # og:site_name
base_url = "https://example.org/atlas/"    # absolute URL of the directory holding the files
clean_urls = true                          # canonical = base_url + "binding-atlas" (else "binding-atlas.html")
card_label = ""                            # URL text on the link card (default: canonical without the scheme)

[nav]
back_url = "https://example.org/"          # optional back-link in the page byline
back_label = "example.org"

[font]                                     # optional web font, supplied by the host (no font files ship here)
family = "Some TW"                         # CSS family, first in the page's font stacks
url = "fonts/"                             # where the host serves {stem}-{Regular,Bold,Italic,BoldItalic}.woff2
stem = "SomeTW"
meta = "Some Typewriter; fallback Courier Prime (OFL)"   # <meta name="typeface">
credit = "Set in Some Typewriter with Courier Prime."     # footer line
```

`page/siteconfig.py` documents every key. `config/animalabs.toml` sets a host-supplied web font; to render the link
card in that font locally, put its files in `site/fonts/` (gitignored) or point `FONT_DIR=` at them.

The page file is `binding-atlas.html`. With `clean_urls = true` the canonical URL is `…/binding-atlas`, which
assumes the server maps `/binding-atlas` to `binding-atlas.html` (as many static hosts do); without that mapping,
set `clean_urls = false`. The committed `site/` is built with `config/animalabs.toml`; rebuild it with a different
config before serving it under another domain. The card carries the canonical URL as text, so it is rebuilt by
`make page` as well.

## Data provenance

- `data/inputs/synthesis.jsonl` holds the rows of the synthesis table that `build_stage.py` reads: one row per model,
  checkpoint and metric, with value, n, standard error, interval, tokens and FLOPs. Checkpoints carry public ids
  (kotodama-tsumugi: `trunk-<tokens>` on the constant-LR trunk, `anneal-<tokens>` for cooldown branches,
  `anneal-110B-release` for the released checkpoint, `s7-…` for the seed-7 replicate). `build/subset_synthesis.py`
  cuts it from the full table, given a local map from the table's checkpoint ids to the public ones.
- `data/built/stage.json` holds the curated front-section layout (figure model lists, stage texts, the battery path);
  `make stage` re-reads every number in it from the synthesis rows.

## Credits

Led by [Luxia](https://x.com/kataletheia), with [Anima Labs](https://animalabs.ai/) · built with [Claude](https://www.anthropic.com/claude).

## Licence

- **Code** (`build/`, `page/`, `config/`, the Makefile, the page code in `site/`): Apache-2.0, see `LICENSE`.
- **Data** (`data/`, the data embedded in the built pages, and the raw score grids): CC-BY-4.0, see `data/LICENSE`.
- **Fonts:** none ship here. Courier Prime (SIL OFL) is loaded from Google Fonts. A host-supplied web font (such as
  the one `config/animalabs.toml` names) is the host's, outside both grants; see `site/fonts/README`.
