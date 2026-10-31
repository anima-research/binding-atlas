# Binding Atlas builds. `make help` lists the targets.
#   page : data/built/*.json -> site/ (HTML, Markdown twin, link card)        needs python3 (+ Pillow, Chrome for the card)
#   data : the zoo sweep's score rows + synthesis table -> data/built/{stage,atlas}.json   needs binding-bench
SHELL   := /bin/bash
PY      ?= python3
# python with binding-bench's dependencies (numpy), for `make data`
PY_DATA ?= $(PY)
# python with the legibility checker's pinned spaCy (tools/legibility_requirements.txt), for `make gates`
PY_LEG ?= .venv-legibility/bin/python
CONFIG  ?= config/default.toml
SITE    ?= site
# local copies of the configured web font, used only to render the link card (absent: fallback faces)
FONT_DIR ?= site/fonts
ifndef STAMP
STAMP := $(shell date -u '+%Y-%m-%d %H:%M UTC')
endif
# the build the kotodama.animalabs.ai deployment serves (config/animalabs.sha256)
VERIFY_CONFIG := animalabs
VERIFY_STAMP := 2026-10-31 04:49 UTC

PAGE_ARGS = --config $(CONFIG) --stamp "$(STAMP)"

.PHONY: help atlas4 atlas4-check gates page html md card neurons neurons-binding data stage verify guard stage-check data-check check clean

help:
	@echo "make page   [CONFIG=config/x.toml] [SITE=dir] [STAMP='YYYY-MM-DD HH:MM UTC']  data/built -> site/"
	@echo "make data   ZOO=dir PROMPTS=dir BENCH=dir [PY_DATA=python]  the zoo sweep + synthesis -> data/built/{stage,atlas}.json"
	@echo "make atlas4 ZOO=dir PROMPTS=dir BENCH=dir [PY_DATA=python]  the zoo-sweep rows -> data/built/atlas.json (the grid)"
	@echo "make atlas4-check ZOO=... PROMPTS=... BENCH=...  rebuild the grid into out/ and compare with data/built/"
	@echo "make stage  refresh every synthesis-derived value in data/built/stage.json (no grids needed)"
	@echo "make neurons [CONFIG=...] [SITE=dir]  Threads (the neuron page) + its data dir (data/built/neurons -> site/)"
	@echo "make neurons-binding   data/inputs/binding -> data/built/neurons/binding.json"
	@echo "make verify       rebuild the kotodama.animalabs.ai deployment into out/verify and check its sha256"
	@echo "make guard        scan every tracked file for internal names (build/forbidden.py)"
	@echo "make stage-check  rebuild stage.json into out/ and compare with data/built/"
	@echo "make data-check   ZOO=... PROMPTS=... BENCH=...  rebuild atlas.json + stage.json into out/ and compare with data/built/"
	@echo "make check        verify + stage-check + guard"
	@echo "make gates        the documentation gates: timelessness, referents, legibility (PY_LEG=python with spaCy)"

# ---- the grid from the zoo sweep (v4): ZOO = the sweep's score rows, PROMPTS = the battery files, BENCH = binding-bench
ZOO     ?=
PROMPTS ?=
BENCH   ?=
atlas4:
	$(PY_DATA) build/build_atlas_v4.py --zoo $(ZOO) --prompts $(PROMPTS) --bench $(BENCH) --old data/built/atlas.json --out data/built/atlas.json

atlas4-check:
	mkdir -p out/data && $(PY_DATA) build/build_atlas_v4.py --zoo $(ZOO) --prompts $(PROMPTS) --bench $(BENCH) --old data/built/atlas.json --out out/data/atlas.json && cmp out/data/atlas.json data/built/atlas.json && echo "atlas4-check ok: out/data/atlas.json = data/built/atlas.json"

# ---- page: data/built -> site/
page: html md card

html:
	$(PY) page/build_page.py $(PAGE_ARGS) --out $(SITE)/binding-atlas.html

md:
	$(PY) page/build_md.py $(PAGE_ARGS) --out $(SITE)/binding-atlas.md

card:
	PY=$(PY) FONT_DIR=$(FONT_DIR) bash page/build_card.sh out/card $(SITE)/binding-atlas-card.png --config $(CONFIG)

# ---- Threads, the neuron page (its data loads at run time from threads-data/ next to it)
neurons:
	$(PY) page/build_neuron_page.py --config $(CONFIG) --stamp "$(STAMP)" --out $(SITE)/threads.html
	PY=$(PY) bash page/build_neuron_card.sh $(SITE) $(SITE)/threads-card.png

neurons-binding:
	$(PY) build/build_binding.py

# ---- data: raw inputs -> data/built
data: stage atlas4

stage:
	$(PY) build/build_stage.py --out data/built/stage.json

# ---- equivalence checks
verify:
	$(MAKE) --no-print-directory page CONFIG=config/$(VERIFY_CONFIG).toml SITE=out/verify STAMP="$(VERIFY_STAMP)"
	@if compgen -G "$(FONT_DIR)/*.woff2" > /dev/null; then cd out/verify && sha256sum -c ../../config/$(VERIFY_CONFIG).sha256; \
	else echo "(no web font files in FONT_DIR=$(FONT_DIR): the card renders in fallback faces, so only the page and twin are checked)"; \
	  cd out/verify && grep -v binding-atlas-card.png ../../config/$(VERIFY_CONFIG).sha256 | sha256sum -c; fi

guard:
	$(PY) build/forbidden.py

stage-check:
	$(PY) build/build_stage.py --base data/built/stage.json --out out/data/stage.json
	cmp out/data/stage.json data/built/stage.json && echo "stage-check ok: out/data/stage.json = data/built/stage.json"

data-check: stage-check atlas4-check

check: verify stage-check guard

# the documentation gates (CONTRIBUTING.md): what comments say, what they point at, and how legible the pages are
gates:
	$(PY) -m tools.check_timelessness --root build page take tools
	$(PY) -m tools.check_referents --root build page take tools
	$(PY_LEG) -m tools.check_legibility

clean:
	rm -rf out
