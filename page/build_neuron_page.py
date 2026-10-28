"""Render Threads (the neuron page): template + site config → threads.html, with its data in threads-data/ beside it.

Usage: python page/build_neuron_page.py --out site/threads.html [--data data/built/neurons]
                                        [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
The page loads its data at run time from the directory threads-data/ next to it (neurons.json, neurons.bin, the
layout-*.bin files and binding.json), which this copies from --data. Refuses to write if the page or its JSON data
contain an internal hostname / path (public-page law). The page's earlier address, neuron-atlas.html, is written beside it
as a redirect.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "build"))
from build_page import build_stamp, load_config  # noqa: E402
from forbidden import FORBIDDEN  # noqa: E402
from siteconfig import CLEAN, PAGE, render_hosting  # noqa: E402

NEURON_PAGE, NEURON_CLEAN, NEURON_CARD, NEURON_DATA = "threads.html", "threads", "threads-card.png", "threads-data"
OLD_PAGE = "neuron-atlas.html"     # the page's earlier address: a redirect to the clean URL
REDIRECT = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Threads</title>
<meta http-equiv="refresh" content="0; url={target}"><link rel="canonical" href="{canonical}">
<meta name="robots" content="noindex"></head>
<body><p>This page is now <a href="{target}">Threads</a>.</p></body></html>
"""
DATA_FILES = ("neurons.json", "neurons.bin", "binding.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=REPO / "data" / "built" / "neurons")
    ap.add_argument("--template", type=Path, default=HERE / "neuron_page.html")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bin-ext", default=".bin", help="extension the binary data files are published under")
    ap.add_argument("--fragment", action="store_true",
                    help="drop the document skeleton (doctype, html, head and body tags) for hosts that wrap the page")
    a = ap.parse_args()
    cfg = load_config(a.config)
    stamp = build_stamp(a.stamp)
    canonical = cfg.base_url + (NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE) if cfg.base_url else ""
    card = (cfg.base_url + NEURON_CARD) if cfg.base_url else NEURON_CARD
    page = render_hosting(a.template.read_text(), cfg, {"__CANONICAL_URL__": html.escape(canonical),
                                                         "__CARD_URL__": html.escape(card)})
    page = page.replace("__ATLAS_LINK__", CLEAN if cfg.clean_urls else PAGE).replace("__STAMP__", stamp)
    page = page.replace("__BIN_EXT__", a.bin_ext)
    files = [a.data / f for f in DATA_FILES] + sorted(a.data.glob("layout-*.bin"))
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing data: {missing}")
    bad = sorted({m.group(0) for text in [page] + [f.read_text() for f in files if f.suffix == ".json"]
                  for m in FORBIDDEN.finditer(text)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public page or its data: {bad}")
    if a.fragment:
        page = re.sub(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", page, flags=re.I)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    dest = a.out.parent / NEURON_DATA
    dest.mkdir(exist_ok=True)
    for f in files:
        shutil.copyfile(f, dest / (f.stem + a.bin_ext if f.suffix == ".bin" else f.name))
    target = NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE
    (a.out.parent / OLD_PAGE).write_text(REDIRECT.format(target=target, canonical=html.escape(canonical or target)))
    size = sum(f.stat().st_size for f in files)
    print(f"wrote {a.out} ({len(page) / 1e3:.0f} kB) and {len(files)} data files in {dest} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
