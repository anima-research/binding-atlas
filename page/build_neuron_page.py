"""Render Threads (the neuron page): template + site config → threads.html, with its data in threads-data/ beside it.

Usage: python page/build_neuron_page.py --out site/threads.html [--data data/built/neurons]
                                        [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
The page loads its data at run time from the directory threads-data/ next to it (neurons.json, neurons.bin, the
layout-*.bin files, binding.json, wiring.json, icl.json, graph.json and closeness.bin), which this copies from --data. Refuses to write if the page
or its JSON data contain an internal hostname / path (public-page law), or if a number in the rule-vs-recall prose
differs from icl.json (see `check_icl_prose`). The page's earlier address, neuron-atlas.html, is written beside it
as a redirect.
"""
from __future__ import annotations

import argparse
import html
import re
import hashlib
import json
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
DATA_FILES = ("neurons.json", "neurons.bin", "binding.json", "wiring.json", "icl.json", "graph.json", "closeness.bin")
MINUS = "\u2212"
#: The prose between each pair of these markers quotes icl.json (a path may instead start "graph:" for graph.json). Each number there sits in <span data-v="path|format">text</span>,
#: and the text must equal the JSON value at that path in that format; a bare decimal or neuron id there refuses.
ICL_BEGIN, ICL_END = "<!-- icl:begin -->", "<!-- icl:end -->"
SPAN_RE = re.compile(r'<span data-v="([^"|]+)\|([a-z0-9]+)">([^<]*)</span>')
BARE_RE = re.compile(r"(?<![\w#&;])[+\-\u2212]?\d*\.\d+(?![\d]|B\b)|\b\d+[/:]\d+\b")


def _unsigned(v: float, d: int) -> str:
    return f"{abs(v):.{d}f}".removeprefix("0")


def _signed(v: float, d: int) -> str:
    a = _unsigned(v, d)
    return a if float(a) == 0 else ("+" if v > 0 else MINUS) + a


ICL_FORMATS = {
    "abs2": lambda v: _unsigned(v, 2), "abs3": lambda v: _unsigned(v, 3),
    "sgn2": lambda v: _signed(v, 2), "sgn3": lambda v: _signed(v, 3),
    "int": lambda v: str(int(v)) if float(v) == int(v) else f"not an integer: {v}",
    "intc": lambda v: f"{int(v):,}" if float(v) == int(v) else f"not an integer: {v}",
    "pct0": lambda v: f"{round(v * 100)}%",
    "pctile1": lambda v: f"{v:.1f}",
    "id": str,
    "pts": lambda v: str(abs(round(v * 100))),
    "sig2": lambda v: f"{float(f'{v:.2g}'):g}",
}


def check_icl_prose(page: str, icl: dict, graph: dict | None = None) -> int:
    """Every number the marked prose quotes equals icl.json (or graph.json); returns how many were checked."""
    starts = [m.start() for m in re.finditer(re.escape(ICL_BEGIN), page)]
    ends = [m.start() for m in re.finditer(re.escape(ICL_END), page)]
    if not starts or len(starts) != len(ends) or any(not a < b for a, b in zip(starts, ends)) \
            or any(not b < a for b, a in zip(ends, starts[1:])):
        raise SystemExit(f"REFUSING: the page needs one or more {ICL_BEGIN} … {ICL_END} blocks, each closed before the next")
    block = "\n".join(page[a:b] for a, b in zip(starts, ends))
    problems: list[str] = []
    n = 0
    for path, fmt, text in SPAN_RE.findall(block):
        n += 1
        cur, keys = (graph, path[len("graph:"):]) if path.startswith("graph:") else (icl, path)
        try:
            if cur is None:
                raise KeyError(path)
            for key in keys.split("."):
                cur = cur[int(key)] if isinstance(cur, list) else cur[key]
        except (KeyError, IndexError, ValueError, TypeError):
            problems.append(f"{path}: not in icl.json")
            continue
        if fmt not in ICL_FORMATS:
            problems.append(f"{path}: unknown format {fmt}")
        elif (want := ICL_FORMATS[fmt](cur)) != text:
            problems.append(f"{path}: the prose says {text!r}, icl.json gives {want!r}")
    bare = re.sub(r"<script.*?</script>", "", SPAN_RE.sub("", block), flags=re.S)
    bare = re.sub(r"<[^>]*>", " ", bare)
    problems += [f"a number outside a data-v span: {m.group(0)!r}" for m in BARE_RE.finditer(bare)]
    if problems:
        raise SystemExit("REFUSING: the rule-vs-recall prose disagrees with icl.json:\n  " + "\n  ".join(problems))
    return n


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
    checked = check_icl_prose(page, json.loads((a.data / "icl.json").read_text()),
                              json.loads((a.data / "graph.json").read_text()))
    print(f"rule-vs-recall prose: {checked} numbers match icl.json")
    if a.fragment:
        page = re.sub(r"<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*", "", page, flags=re.I)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page)
    dest = a.out.parent / NEURON_DATA
    dest.mkdir(exist_ok=True)
    for f in files:
        shutil.copyfile(f, dest / (f.stem + a.bin_ext if f.suffix == ".bin" else f.name))
    conn = a.data / "conn"   # the connections panel's shards (not in git; conn.json, tracked, holds each one's sha256)
    if (conn / "conn.json").exists():
        meta = json.loads((conn / "conn.json").read_text())
        (dest / "conn").mkdir(exist_ok=True)
        for name, digest in meta["files"].items():
            blob = (conn / name).read_bytes()
            if hashlib.sha256(blob).hexdigest() != digest:
                raise SystemExit(f"conn/{name} does not match its sha256 in conn.json")
            (dest / "conn" / (Path(name).stem + a.bin_ext)).write_bytes(blob)
        shutil.copyfile(conn / "conn.json", dest / "conn" / "conn.json")
        print(f"copied {len(meta['files'])} connection shards (sha256-checked)")
    target = NEURON_CLEAN if cfg.clean_urls else NEURON_PAGE
    (a.out.parent / OLD_PAGE).write_text(REDIRECT.format(target=target, canonical=html.escape(canonical or target)))
    size = sum(f.stat().st_size for f in files)
    print(f"wrote {a.out} ({len(page) / 1e3:.0f} kB) and {len(files)} data files in {dest} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
