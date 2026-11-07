"""Render the announcement (the site's front page): template + site config -> index.html, with its figure and link card.

Usage: python page/build_announce_page.py --out site/index.html [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"] [--edit]
The words come from copy/announce.md (page/copy_text.py fills the template's slots). --edit builds the preview that
page/edit_server.py serves: each piece of page text is marked with its key, and the editor's script is loaded.
The figure (announce-data/figure.png) and the link card (kotodama-tsumugi-card.png) are copied from
data/inputs/announce/, and the scroll-driven story (story.html there) is inlined at __STORY__. Refuses to write if the page contains an internal hostname or path (public-page law).
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
from copy_text import COPY_DIR, CopyError, fill  # noqa: E402
from build_empty_page import EMPTY_CLEAN, EMPTY_PAGE  # noqa: E402
from build_neuron_page import NEURON_CLEAN, NEURON_PAGE  # noqa: E402
from forbidden import FORBIDDEN  # noqa: E402
from siteconfig import CLEAN, PAGE, TSUMUGI_CLEAN, TSUMUGI_PAGE, render_hosting  # noqa: E402

CARD_FILE, DATA_DIR, FIGURE = "kotodama-tsumugi-card.png", "announce-data", "figure.png"
MODEL_CARD = "https://huggingface.co/LuxiaSL/kotodama-tsumugi-1b"
SRC = REPO / "data" / "inputs" / "announce"
COPY = COPY_DIR / "announce.md"
EDITOR = '<link rel="stylesheet" href="/__editor/editor.css"><script src="/__editor/editor.js" defer></script>\n'
STORY = "story.html"   # the scroll-driven four-result story (generated with the animated card; same beats, same numbers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", type=Path, default=HERE / "announce_page.html")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--edit", action="store_true", help="build the editor's preview (never deployed)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for f in (SRC / FIGURE, SRC / "card.png", SRC / STORY, COPY):
        if not f.exists():
            raise SystemExit(f"missing input: {f}")
    canonical = cfg.base_url or ""
    card = (cfg.base_url + CARD_FILE) if cfg.base_url else CARD_FILE
    try:
        words, _ = fill(a.template.read_text(), COPY, edit=a.edit)
    except CopyError as e:
        raise SystemExit(f"copy: {e}") from None
    page = render_hosting(words, cfg, {"__CANONICAL_URL__": html.escape(canonical),
                                                         "__CARD_URL__": html.escape(card)})
    page = (page.replace("__ATLAS_LINK__", CLEAN if cfg.clean_urls else PAGE)
                .replace("__THREADS_LINK__", NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE)
                .replace("__TSUMUGI_LINK__", TSUMUGI_CLEAN if cfg.clean_urls else TSUMUGI_PAGE)
                .replace("__EMPTY_LINK__", EMPTY_CLEAN if cfg.clean_urls else EMPTY_PAGE)
                .replace("__CARD_LINK__", MODEL_CARD)
                .replace("__FIGURE__", f"{DATA_DIR}/{FIGURE}")
                .replace("__STAMP__", build_stamp(a.stamp))
                .replace("__STORY__", (SRC / STORY).read_text()))
    left = sorted({w for w in page.split("__")[1::2] if w.isupper() and w.replace("_", "").isalpha()})
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    bad = sorted({m.group(0) for m in FORBIDDEN.finditer(page)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public page: {bad}")
    if a.edit:
        page = page.replace("</body>", EDITOR + "</body>", 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    (a.out.parent / DATA_DIR).mkdir(exist_ok=True)
    shutil.copyfile(SRC / FIGURE, a.out.parent / DATA_DIR / FIGURE)
    shutil.copyfile(SRC / "card.png", a.out.parent / CARD_FILE)
    print(f"wrote {a.out} ({len(page) / 1e3:.0f} kB), {DATA_DIR}/{FIGURE} and {CARD_FILE}")


if __name__ == "__main__":
    main()
