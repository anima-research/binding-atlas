# Binding Atlas

Base language models learn who *you* is in five stages. No public model up to 8B reaches the last one zero-shot, at
any token count measured, and from 14B to 1T parameters it stays weak. The Binding Atlas puts every model against
every referent-binding probe, chance-normalized, with the raw context behind each cell: verbatim example prompts
for each column, error anatomy for each deixis cell, and a plain-text Markdown twin carrying the same data as CSV.

The atlas covers 30 public base models, 5 public training trajectories, post-trained and API models, and one human
reader. It also follows **kotodama-tsumugi**, a 1.4B model run as 72 effective layers (looped depth,
boundary-fitted attention residuals). kotodama-tsumugi reaches stage 3 by 33B tokens, the token cost of OLMo-2 13B,
against about 420B for OLMo-2 1B. Trained to 110B tokens, it has not reached stage 5.

## Layout

| path | what |
|---|---|
| `site/` | the built site, ready to serve: `binding-atlas.html`, `binding-atlas.md`, `binding-atlas-card.png` |
| `page/` | page builders: `build_page.py` (HTML), `build_md.py` (Markdown twin), `build_card.py` + `build_card.sh` (link card), `atlas_page.html` (template), `meta.json` (findings, gaps), `siteconfig.py` (hosting config) |
| `build/` | data builders: `build_atlas_v3.py` (raw grids → `atlas.json`), `build_stage.py` (synthesis table → `stage.json`), `facts.py`, `dx_replay.py`; `link_grids.py` (local copies → grids root), `subset_synthesis.py` (cuts `data/inputs/synthesis.jsonl`), `forbidden.py` (the public-content guard) |
| `data/built/` | what the page shows: `atlas.json`, `stage.json`, `swap.json` |
| `data/inputs/` | small tracked inputs: `synthesis.jsonl` (the rows of the synthesis table `build_stage.py` reads) and the human gauntlet items |
| `data/INPUTS-MANIFEST.tsv` | every raw file `build_atlas_v3.py` reads, by logical path, with size and sha256 |
| `config/` | hosting configs (`default.toml` neutral, `aetherawi.toml`), `aetherawi.sha256`, `local-grids.example.tsv` |

## Building

### The page (`data/built` → `site/`)

```bash
make page                                  # neutral config: relative URLs, no canonical link
make page CONFIG=config/aetherawi.toml     # a deployment config (see below)
make page SITE=/some/dir STAMP="2026-01-31 12:00 UTC"
```

The page and the Markdown twin need only Python ≥ 3.11 (standard library). The link card also needs Pillow and
Google Chrome (`PY=` a python with Pillow, `CHROME=` the browser binary). `STAMP` is the "atlas build" line in the
footer (default: now, UTC); its date is the "updated" line.

### The data (raw grids → `data/built`)

```bash
make stage                                 # re-read every number in data/built/stage.json from data/inputs/synthesis.jsonl
make data GRIDS=/path/to/grids PY_DATA=.venv/bin/python
```

`make data` runs `make stage` and then `build_atlas_v3.py`, which reads the raw score grids under one root
(`GRIDS=` / `--grids DIR` / `ATLAS_GRIDS`) laid out as `data/INPUTS-MANIFEST.tsv` lists them, and imports
`kotodama.eval.bindprobes` to render the example prompts. kotodama is pinned in `pyproject.toml`
(`kotodama @ git+https://github.com/anima-research/kotodama@21f9989ae1cef6b8025b689b3b7ebd09430ca89f`):

```bash
uv venv && uv sync --extra data            # kotodama and its dependencies (including torch) into .venv
make data GRIDS=/path/to/grids PY_DATA=.venv/bin/python
# or, with a kotodama checkout at that commit:  make data GRIDS=... KOTODAMA_SRC=/path/to/kotodama/src
```

The published Atlas items render byte-identically at that commit, because kotodama renders pinned items with the
templates of their own item version.

When the raw files sit in several local directories, `config/local-grids.tsv` (gitignored; template
`config/local-grids.example.tsv`) maps logical path prefixes to them, and `make grids` assembles a checked tree of
symlinks in `.grids/` (every size and sha256 against the manifest):

```bash
cp config/local-grids.example.tsv config/local-grids.tsv   # then edit
make grids && make data GRIDS=.grids
```

### Equivalence checks

| command | checks |
|---|---|
| `make verify` | builds with `config/aetherawi.toml` and the committed build stamp into `out/verify/`; the HTML, Markdown twin and card must match `config/aetherawi.sha256` (the files in `site/`) byte for byte |
| `make stage-check` | `stage.json` rebuilt from the tracked synthesis rows equals `data/built/stage.json` |
| `make guard` | no tracked file contains an internal hostname, machine path, session id or internal run / checkpoint name (`build/forbidden.py`; the page builders apply the same guard to their output) |
| `make data-check GRIDS=…` | `atlas.json` and `stage.json` rebuilt from the raw grids equal `data/built/` |
| `make check` | `verify` + `stage-check` + `guard` (no grids needed) |

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

`page/siteconfig.py` documents every key. `config/aetherawi.toml` sets a host-supplied web font; to render the link
card in that font locally, put its files in `site/fonts/` (gitignored) or point `FONT_DIR=` at them.

The page file is `binding-atlas.html`. With `clean_urls = true` the canonical URL is `…/binding-atlas`, which
assumes the server maps `/binding-atlas` to `binding-atlas.html` (as many static hosts do); without that mapping,
set `clean_urls = false`. The committed `site/` is built with `config/aetherawi.toml`; rebuild it with a different
config before serving it under another domain. The card carries the canonical URL as text, so it is rebuilt by
`make page` as well.

## Data provenance

- `data/INPUTS-MANIFEST.tsv` lists the 143 files `build_atlas_v3.py` reads (an audit-hook trace of a full build):
  141 raw score grids and item files (1.86 GB) under the grids root, by logical path, and the two human-gauntlet item
  files tracked here. Each entry carries its size and sha256. The raw grids are to be published as a public
  dataset (location to be announced); the manifest is the contract that dataset meets.
- `data/inputs/synthesis.jsonl` is the subset of the synthesis table (one row per model × checkpoint × metric, with
  value, n, standard error, interval, tokens and FLOPs) that `build_stage.py` reads, under public checkpoint ids
  (kotodama-tsumugi: `trunk-<tokens>` on the constant-LR trunk, `anneal-<tokens>` for cooldown branches,
  `anneal-110B-release` for the released checkpoint, `s7-…` for the seed-7 replicate). `build/subset_synthesis.py`
  cuts it from the full table, given a local map from the table's checkpoint ids to the public ones.
- `data/built/stage.json` holds the curated front-section layout (figure model lists, stage texts, the battery path);
  `make stage` re-reads every number in it from the synthesis rows.
- `data/built/swap.json` (the speaker-swap table in the Markdown twin) is carried over from the 2026-10-15 edition and
  is not rebuilt by `make data`.

## Credits

Research, code and development: Luxia (@LuxiaSL) and Claude (Anthropic). Trained on compute provided by Anima Labs.

## Licence

- **Code** (`build/`, `page/`, `config/`, the Makefile, the page code in `site/`): Apache-2.0, see `LICENSE`.
- **Data** (`data/`, the data embedded in the built pages, and the raw score grids): CC-BY-4.0, see `data/LICENSE`.
- **Fonts:** none ship here. Courier Prime (SIL OFL) is loaded from Google Fonts. A host-supplied web font (such as
  the one `config/aetherawi.toml` names) is the host's, outside both grants; see `site/fonts/README`.
