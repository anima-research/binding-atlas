"""Render An Empty Document: template + site config → empty-document.html, with its data in empty-document-data/.

Usage: python page/build_empty_page.py --out site/empty-document.html [--data data/built/empty-document.json]
                                       [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
The page loads its runs at run time from empty-document-data/runs.json next to it (a copy of --data). Refuses to
write if the page or its data contain an internal hostname / path (public-page law).
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "build"))
from build_page import build_stamp, load_config  # noqa: E402
from build_neuron_page import NEURON_CLEAN, NEURON_PAGE  # noqa: E402
from forbidden import FORBIDDEN  # noqa: E402
from siteconfig import CLEAN, PAGE, render_hosting  # noqa: E402

EMPTY_PAGE, EMPTY_CLEAN, EMPTY_CARD, EMPTY_DATA = "empty-document.html", "empty-document", "empty-document-card.png", \
    "empty-document-data"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=REPO / "data" / "built" / "empty-document.json")
    ap.add_argument("--template", type=Path, default=HERE / "empty_page.html")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    stamp = build_stamp(a.stamp)
    canonical = cfg.base_url + (EMPTY_CLEAN if cfg.clean_urls else EMPTY_PAGE) if cfg.base_url else ""
    card = (cfg.base_url + EMPTY_CARD) if cfg.base_url else EMPTY_CARD
    page = render_hosting(a.template.read_text(), cfg, {"__CANONICAL_URL__": html.escape(canonical),
                                                         "__CARD_URL__": html.escape(card)})
    page = (page.replace("__ATLAS_LINK__", CLEAN if cfg.clean_urls else PAGE)
                .replace("__THREADS_LINK__", NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE)
                .replace("__STAMP__", stamp))
    if not a.data.exists():
        raise SystemExit(f"missing data: {a.data} (run build/build_empty.py)")
    data = a.data.read_text()
    bad = sorted({m.group(0) for text in (page, data) for m in FORBIDDEN.finditer(text)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public page or its data: {bad}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    dest = a.out.parent / EMPTY_DATA
    dest.mkdir(exist_ok=True)
    shutil.copyfile(a.data, dest / "runs.json")
    print(f"wrote {a.out} ({len(page) / 1e3:.0f} kB) and {dest / 'runs.json'} ({a.data.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
