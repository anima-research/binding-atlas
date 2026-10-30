"""Render the standalone Binding Atlas page: atlas JSON + stage JSON + meta.json + site config → binding-atlas.html.

Usage: python page/build_page.py --atlas data/built/atlas.json --stage data/built/stage.json --out site/binding-atlas.html
                                 [--config config/default.toml] [--stamp "YYYY-MM-DD HH:MM UTC"]
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
from siteconfig import ConfigError, SiteConfig, render_hosting  # noqa: E402

STAMP_FMT = "%Y-%m-%d %H:%M UTC"


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
    ap.add_argument("--meta", type=Path, default=HERE / "meta.json")
    ap.add_argument("--config", type=Path, default=None, help="site config TOML (default: neutral, relative URLs)")
    ap.add_argument("--stamp", default=None, help="build stamp, 'YYYY-MM-DD HH:MM UTC' (default: now)")
    ap.add_argument("--out", type=Path, required=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    common_args(ap)
    ap.add_argument("--template", type=Path, default=HERE / "atlas_page.html")
    a = ap.parse_args()
    cfg = load_config(a.config)
    stamp = build_stamp(a.stamp)
    atlas = json.loads(a.atlas.read_text())
    meta = json.loads(a.meta.read_text())
    meta.pop("changelog", None)   # kept in meta.json for the record; the public page does not narrate its own edits
    meta["updated"] = stamp[:10]
    meta["build"] = stamp
    meta["items"] = sum(v["n"] for k, v in atlas["cells"].items() if k.endswith("|all"))
    for r in atlas["rows"]:
        r["sources"] = [s for s in r.get("sources", []) if not FORBIDDEN.search(s)]
    tpl = render_hosting(a.template.read_text(), cfg)
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")  # noqa: E731
    stage = json.loads(a.stage.read_text())   # the front-section figures (stage map; curated from records)
    tpl = tpl.replace("__THREADS_LINK__", "threads" if cfg.clean_urls else "threads.html")   # the neuron page, a sibling file
    n_models = sum(1 for r in atlas["rows"] if r["group"] != "Human")   # the first-screen counts: models + checkpoints, tests
    n_bases = sum(1 for r in atlas["rows"] if r["group"] not in ("Human",) and r.get("family") not in (None, "kotodama"))
    tpl = (tpl.replace("__N_MODELS__", str(n_models)).replace("__N_TESTS__", str(len(atlas["cols"])))
           .replace("__N_BASES__", str(n_bases)))
    html = (tpl.replace("__ATLAS_DATA__", dump(atlas)).replace("__ATLAS_META__", dump(meta))
            .replace("__STAGE_DATA__", dump(stage)))
    bad = sorted({m.group(0) for m in FORBIDDEN.finditer(html)})
    if bad:
        raise SystemExit(f"REFUSING: internal names in the public page: {bad}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html)
    print(f"wrote {a.out}: {len(html)/1e6:.2f} MB; {len(atlas['rows'])} rows, {len(atlas['cols'])} cols, "
          f"{len(atlas['cells'])} cells, {meta['items']:,} scored items; build {stamp}")


if __name__ == "__main__":
    main()
