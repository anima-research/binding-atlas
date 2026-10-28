# Binding Atlas builds. `make help` lists the targets.
#   page : data/built/*.json -> site/ (HTML, Markdown twin, link card)        needs python3 (+ Pillow, Chrome for the card)
#   data : raw grids + synthesis table -> data/built/{stage,atlas}.json       needs kotodama (pyproject [data] extra)
SHELL   := /bin/bash
PY      ?= python3
# python with kotodama importable, for `make data` (e.g. .venv/bin/python after `uv pip install -e '.[data]'`)
PY_DATA ?= $(PY)
# optional: a kotodama source checkout to put on PYTHONPATH instead of installing it
KOTODAMA_SRC ?=
CONFIG  ?= config/default.toml
SITE    ?= site
GRIDS   ?= $(ATLAS_GRIDS)
GRIDMAP ?= config/local-grids.tsv
# local copies of the configured web font, used only to render the link card (absent: fallback faces)
FONT_DIR ?= site/fonts
ifndef STAMP
STAMP := $(shell date -u '+%Y-%m-%d %H:%M UTC')
endif
# the build the aetherawi.red deployment serves (config/aetherawi.sha256)
VERIFY_STAMP := 2026-10-28 20:45 UTC

DATA_ENV := $(if $(KOTODAMA_SRC),PYTHONPATH=$(KOTODAMA_SRC),)
PAGE_ARGS = --config $(CONFIG) --stamp "$(STAMP)"

.PHONY: help page html md card neurons neurons-binding data stage atlas grids verify guard stage-check data-check check legibility clean

help:
	@echo "make page   [CONFIG=config/x.toml] [SITE=dir] [STAMP='YYYY-MM-DD HH:MM UTC']  data/built -> site/"
	@echo "make data   GRIDS=dir [PY_DATA=python] [KOTODAMA_SRC=dir]  raw grids -> data/built/{stage,atlas}.json"
	@echo "make stage  refresh every synthesis-derived value in data/built/stage.json (no grids needed)"
	@echo "make neurons [CONFIG=...] [SITE=dir]  Threads (the neuron page) + its data dir (data/built/neurons -> site/)"
	@echo "make neurons-binding   data/inputs/binding -> data/built/neurons/binding.json"
	@echo "make grids  [GRIDMAP=config/local-grids.tsv]  link local copies into .grids/ (sha256-checked)"
	@echo "make verify       rebuild the aetherawi.red deployment into out/verify and check its sha256"
	@echo "make guard        scan every tracked file for internal names (build/forbidden.py)"
	@echo "make stage-check  rebuild stage.json into out/ and compare with data/built/"
	@echo "make data-check   GRIDS=dir  rebuild atlas.json + stage.json into out/ and compare with data/built/"
	@echo "make check        verify + stage-check + guard"
	@echo "make legibility   KOTODAMA_SRC=dir  legibility report on README.md [LEGIBILITY_FILES=…] (report only)"

# ---- page: data/built -> site/
page: html md card

html:
	$(PY) page/build_page.py $(PAGE_ARGS) --out $(SITE)/binding-atlas.html

md:
	$(PY) page/build_md.py $(PAGE_ARGS) --swap data/built/swap.json --out $(SITE)/binding-atlas.md

card:
	PY=$(PY) FONT_DIR=$(FONT_DIR) bash page/build_card.sh out/card $(SITE)/binding-atlas-card.png --config $(CONFIG)

# ---- Threads, the neuron page (its data loads at run time from threads-data/ next to it)
neurons:
	$(PY) page/build_neuron_page.py --config $(CONFIG) --stamp "$(STAMP)" --out $(SITE)/threads.html
	PY=$(PY) bash page/build_neuron_card.sh $(SITE) $(SITE)/threads-card.png

neurons-binding:
	$(PY) build/build_binding.py

# ---- data: raw inputs -> data/built
data: stage atlas

stage:
	$(PY) build/build_stage.py --out data/built/stage.json

atlas:
	@test -n "$(GRIDS)" || { echo "set GRIDS=dir (or ATLAS_GRIDS); see data/INPUTS-MANIFEST.tsv and 'make grids'"; exit 2; }
	$(DATA_ENV) $(PY_DATA) build/build_atlas_v3.py data/built/atlas.json --grids $(GRIDS)

grids:
	$(PY) build/link_grids.py $(GRIDMAP) .grids --verify

# ---- equivalence checks
verify:
	$(MAKE) --no-print-directory page CONFIG=config/aetherawi.toml SITE=out/verify STAMP="$(VERIFY_STAMP)"
	@if compgen -G "$(FONT_DIR)/*.woff2" > /dev/null; then cd out/verify && sha256sum -c ../../config/aetherawi.sha256; \
	else echo "(no web font files in FONT_DIR=$(FONT_DIR): the card renders in fallback faces, so only the page and twin are checked)"; \
	  cd out/verify && grep -v binding-atlas-card.png ../../config/aetherawi.sha256 | sha256sum -c; fi

guard:
	$(PY) build/forbidden.py

stage-check:
	$(PY) build/build_stage.py --base data/built/stage.json --out out/data/stage.json
	cmp out/data/stage.json data/built/stage.json && echo "stage-check ok: out/data/stage.json = data/built/stage.json"

data-check: stage-check
	@test -n "$(GRIDS)" || { echo "set GRIDS=dir (or ATLAS_GRIDS)"; exit 2; }
	$(DATA_ENV) $(PY_DATA) build/build_atlas_v3.py out/data/atlas.json --grids $(GRIDS)
	cmp out/data/atlas.json data/built/atlas.json && echo "data-check ok: out/data/atlas.json = data/built/atlas.json"

check: verify stage-check guard

# a report, never a gate: kotodama's tools/check_legibility.py (needs spacy + en_core_web_trf + wordfreq in PY_DATA).
# The Markdown twin is mostly data and slow to parse on a CPU: add it with LEGIBILITY_FILES="README.md site/binding-atlas.md".
LEGIBILITY_FILES ?= README.md
legibility:
	@test -n "$(KOTODAMA_SRC)" || { echo "set KOTODAMA_SRC=/path/to/kotodama/src (the checker is its tools/check_legibility.py)"; exit 2; }
	$(PY_DATA) $(KOTODAMA_SRC)/../tools/check_legibility.py $(LEGIBILITY_FILES)

clean:
	rm -rf out
