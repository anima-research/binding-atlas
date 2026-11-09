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
import re
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


CITE_MK = re.compile(r'(<button type="button" class="cite-mk"[^>]*>)\d+(</button><span class="cite-leaf" role="note">)(.*?)(</span></span>)', re.S)


def renumber_citations(page: str) -> str:
    """Number citation markers across the whole page (the page's own and the story's), in order of first appearance;
    markers whose leaflets hold the same reference share a number."""
    numbers: dict[str, int] = {}

    def one(m: re.Match[str]) -> str:
        n = numbers.setdefault(re.sub(r"\s+", " ", m.group(3)).strip(), len(numbers) + 1)
        button = re.sub(r'aria-label="Reference \d+"', f'aria-label="Reference {n}"', m.group(1))
        return f"{button}{n}{m.group(2)}{m.group(3)}{m.group(4)}"
    return CITE_MK.sub(one, page)


LEDE = re.compile(r'(<p class="lede">)(<span data-copy="[^"]*"[^>]*>)?(.*?)(</span>)?(</p>)', re.S)
BREAKS = re.compile(r"\s*((?:<br\s*/?>\s*)+)")


def lede_beats(words: str) -> str:
    """The lede's lines as blocks: each run of <br> in its entry starts a new beat, set off by a short gap (one <br>)
    or a longer one (two or more). The entry keeps plain <br>; an edit build keeps its data-copy wrapper around them."""
    m = LEDE.search(words)
    if m is None:
        raise SystemExit('the template has no <p class="lede">')
    p_open, wrap_open, inner, wrap_close, p_close = m.groups()
    if bool(wrap_open) != bool(wrap_close):
        raise SystemExit("the lede's data-copy wrapper is unbalanced")
    parts = BREAKS.split(inner.strip())          # [text, breaks, text, breaks, …]
    beats = [f'<span class="beat">{parts[0]}</span>']
    for breaks, text in zip(parts[1::2], parts[2::2]):
        if text.strip():
            gap = " far" if len(re.findall(r"<br", breaks)) > 1 else ""
            beats.append(f'<span class="beat{gap}">{text}</span>')
    body = "".join(beats)
    if wrap_open:
        body = f"{wrap_open}{body}{wrap_close}"
    return words[:m.start()] + p_open + body + p_close + words[m.end():]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--template", type=Path, default=HERE / "announce_page.html")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--edit", action="store_true", help="build the editor's preview (never deployed)")
    ap.add_argument("--story", type=Path, default=SRC / STORY, help="the story fragment to inline (the editor passes its edit build)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    for f in (SRC / FIGURE, SRC / "card.png", a.story, COPY):
        if not f.exists():
            raise SystemExit(f"missing input: {f}")
    canonical = cfg.base_url or ""
    card = (cfg.base_url + CARD_FILE) if cfg.base_url else CARD_FILE
    try:
        words, _ = fill(a.template.read_text(), COPY, edit=a.edit)
        words = re.sub(r"\n?[ \t]*<p>\s*</p>", "", words)   # a paragraph whose entry is empty
        words = lede_beats(words)
    except CopyError as e:
        raise SystemExit(f"copy: {e}") from None
    page = render_hosting(words, cfg, {"__CANONICAL_URL__": html.escape(canonical),
                                                         "__CARD_URL__": html.escape(card)})
    page = (page.replace("__STORY__", a.story.read_text())   # first: the story carries link placeholders too
                .replace("__ATLAS_LINK__", CLEAN if cfg.clean_urls else PAGE)
                .replace("__THREADS_LINK__", NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE)
                .replace("__TSUMUGI_LINK__", TSUMUGI_CLEAN if cfg.clean_urls else TSUMUGI_PAGE)
                .replace("__EMPTY_LINK__", EMPTY_CLEAN if cfg.clean_urls else EMPTY_PAGE)
                .replace("__CARD_LINK__", MODEL_CARD)
                .replace("__FIGURE__", f"{DATA_DIR}/{FIGURE}")
                .replace("__STAMP__", build_stamp(a.stamp)))
    page = renumber_citations(page)
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
