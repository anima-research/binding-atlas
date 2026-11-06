"""Render the standalone Binding Atlas page, or kotodama-tsumugi's page beside it, from one template.

Usage: python page/build_page.py --atlas data/built/atlas.json --stage data/built/stage.json --out site/binding-atlas.html
                                 [--page atlas|tsumugi] [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
--page atlas (default) writes the benchmark page; --page tsumugi writes site/binding-atlas/tsumugi.html, the model's
training trajectory and release tables, one folder below the atlas. The template marks page-specific blocks with
<!--only:atlas--> … <!--/only:atlas--> and <!--only:tsumugi--> … <!--/only:tsumugi-->.
The stamp (default: now, UTC) is the "atlas build" line in the footer; its date is the "updated" line. Pass the same
stamp to build_md.py so the two files name one build.
Refuses to write if the rendered page contains an internal hostname / path (public-page law).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "build"))
from forbidden import FORBIDDEN  # noqa: E402  (the one table of forbidden names)
from siteconfig import (CLEAN, PAGE, TSUMUGI_CLEAN, TSUMUGI_PAGE, ConfigError, SiteConfig,  # noqa: E402
                        render_hosting, without_hidden_rows)

STAMP_FMT = "%Y-%m-%d %H:%M UTC"
PAGES = ("atlas", "tsumugi")
_ONLY = re.compile(r"<!--only:(\w+)-->\n?(.*?)<!--/only:\1-->\n?", re.S)


def page_variant(template: str, page: str) -> str:
    """The template with the other pages' <!--only:…--> blocks removed and this page's markers dropped."""
    names = {m.group(1) for m in _ONLY.finditer(template)}
    if names - set(PAGES):
        raise SystemExit(f"template marks unknown pages: {sorted(names - set(PAGES))}")
    out = _ONLY.sub(lambda m: m.group(2) if m.group(1) == page else "", template)
    if "<!--only:" in out or "<!--/only:" in out:
        raise SystemExit("template has an unmatched <!--only:…--> marker")
    return out


def stage_counts(stage: dict) -> dict[str, int]:
    """Counts the atlas byline states about its figures: public bases in Figure 2, public training runs in Figure 3."""
    return {"bases": len(stage["threshold"]["models"]),
            "runs": len({p["m"] for r in stage["cost"]["rows"] for p in r["pts"] if not p.get("ours")})}


def build_stamp(arg: str | None) -> str:
    """The build stamp: the given one (validated) or the current UTC minute."""
    if not arg:
        return time.strftime(STAMP_FMT, time.gmtime())
    try:
        time.strptime(arg, STAMP_FMT)
    except ValueError as e:
        raise SystemExit(f"--stamp must match {STAMP_FMT!r} (date, time, UTC), got {arg!r}") from e
    return arg


def load_config(path: Path | None) -> SiteConfig:
    try:
        return SiteConfig.load(path)
    except ConfigError as e:
        raise SystemExit(f"site config: {e}") from e


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--atlas", type=Path, default=REPO / "data" / "built" / "atlas.json")
    ap.add_argument("--stage", type=Path, default=REPO / "data" / "built" / "stage.json")
    ap.add_argument("--config", type=Path, default=None, help="site config TOML (default: neutral, relative URLs)")
    ap.add_argument("--stamp", default=None, help="build stamp, 'YYYY-MM-DD HH:MM UTC' (default: now)")
    ap.add_argument("--out", type=Path, required=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common_args(ap)
    ap.add_argument("--template", type=Path, default=HERE / "atlas_page.html")
    ap.add_argument("--page", choices=PAGES, default="atlas")
    a = ap.parse_args()
    cfg = load_config(a.config)
    stamp = build_stamp(a.stamp)
    atlas = without_hidden_rows(json.loads(a.atlas.read_text()))
    meta = {"updated": stamp[:10], "build": stamp,
            "items": sum(v["n"] for k, v in atlas["cells"].items() if k.endswith("|all"))}
    for r in atlas["rows"]:
        r["sources"] = [s for s in r.get("sources", []) if not FORBIDDEN.search(s)]
    clean = cfg.clean_urls
    if a.page == "tsumugi":   # one folder below the atlas: every sibling link and relative asset goes up one level
        canon = cfg.base_url + (TSUMUGI_CLEAN if clean else TSUMUGI_PAGE) if cfg.base_url else ""
        tpl = render_hosting(page_variant(a.template.read_text(), a.page), cfg,
                             overrides={"__CANONICAL_URL__": canon}, up="../")
        links = {"__THREADS_LINK__": "../threads" if clean else "../threads.html",
                 "__ATLAS_LINK__": "../" + (CLEAN if clean else PAGE), "__TAKE_LINK__": "../take"}
    else:
        tpl = render_hosting(page_variant(a.template.read_text(), a.page), cfg)
        links = {"__THREADS_LINK__": "threads" if clean else "threads.html",
                 "__TSUMUGI_LINK__": TSUMUGI_CLEAN if clean else TSUMUGI_PAGE, "__TAKE_LINK__": "take"}
    for k, v in links.items():
        tpl = tpl.replace(k, v)
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")  # noqa: E731
    stage = json.loads(a.stage.read_text())   # the front-section figures (stage map; curated from records)
    # the first-screen counts: public bases (every row but the human and kotodama-tsumugi's), tests, figure coverage
    n_bases = sum(1 for r in atlas["rows"] if r["group"] not in ("Human",) and r.get("family") not in (None, "kotodama"))
    fig = stage_counts(stage)
    tpl = (tpl.replace("__N_TESTS__", str(len(atlas["cols"]))).replace("__N_BASES__", str(n_bases))
           .replace("__N_FIG_BASES__", str(fig["bases"])).replace("__N_FIG_RUNS__", str(fig["runs"])))
    html = (tpl.replace("__ATLAS_DATA__", dump(atlas)).replace("__ATLAS_META__", dump(meta))
            .replace("__STAGE_DATA__", dump(stage)))
    bad = sorted({m.group(0) for m in FORBIDDEN.finditer(html)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public page: {bad}")
    unfilled = sorted(set(re.findall(r"__[A-Z][A-Z0-9_]*[A-Z0-9]__", html)))
    if unfilled:
        raise SystemExit(f"REFUSING: unfilled placeholders in the {a.page} page: {unfilled}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    if a.page == "tsumugi":
        print(f"wrote {a.out}: {len(html)/1e6:.2f} MB; kotodama-tsumugi on the benchmark; build {stamp}")
    else:
        print(f"wrote {a.out}: {len(html)/1e6:.2f} MB; {len(atlas['rows'])} rows, {len(atlas['cols'])} cols, "
              f"{len(atlas['cells'])} cells, {meta['items']:,} scored items; build {stamp}")


if __name__ == "__main__":
    main()
