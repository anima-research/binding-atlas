"""Render the announcement (the site's front page): template + site config -> index.html, with its figure and link card.

Usage: python page/build_announce_page.py --out site/index.html [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
The figure (announce-data/figure.png) and the link card (kotodama-tsumugi-card.png) are copied from
data/inputs/announce/. Refuses to write if the page contains an internal hostname or path (public-page law).
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
from build_empty_page import EMPTY_CLEAN, EMPTY_PAGE  # noqa: E402
from build_neuron_page import NEURON_CLEAN, NEURON_PAGE  # noqa: E402
from forbidden import FORBIDDEN  # noqa: E402
from siteconfig import CLEAN, PAGE, render_hosting  # noqa: E402

CARD_FILE, DATA_DIR, FIGURE = "kotodama-tsumugi-card.png", "announce-data", "figure.png"
MODEL_CARD = "https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b"
SRC = REPO / "data" / "inputs" / "announce"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", type=Path, default=HERE / "announce_page.html")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    for f in (SRC / FIGURE, SRC / "card.png"):
        if not f.exists():
            raise SystemExit(f"missing input: {f}")
    canonical = cfg.base_url or ""
    card = (cfg.base_url + CARD_FILE) if cfg.base_url else CARD_FILE
    page = render_hosting(a.template.read_text(), cfg, {"__CANONICAL_URL__": html.escape(canonical),
                                                         "__CARD_URL__": html.escape(card)})
    page = (page.replace("__ATLAS_LINK__", CLEAN if cfg.clean_urls else PAGE)
                .replace("__THREADS_LINK__", NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE)
                .replace("__EMPTY_LINK__", EMPTY_CLEAN if cfg.clean_urls else EMPTY_PAGE)
                .replace("__CARD_LINK__", MODEL_CARD)
                .replace("__FIGURE__", f"{DATA_DIR}/{FIGURE}")
                .replace("__STAMP__", build_stamp(a.stamp)))
    left = sorted({w for w in page.split("__")[1::2] if w.isupper() and w.replace("_", "").isalpha()})
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    bad = sorted({m.group(0) for m in FORBIDDEN.finditer(page)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public page: {bad}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    (a.out.parent / DATA_DIR).mkdir(exist_ok=True)
    shutil.copyfile(SRC / FIGURE, a.out.parent / DATA_DIR / FIGURE)
    shutil.copyfile(SRC / "card.png", a.out.parent / CARD_FILE)
    print(f"wrote {a.out} ({len(page) / 1e3:.0f} kB), {DATA_DIR}/{FIGURE} and {CARD_FILE}")


if __name__ == "__main__":
    main()
