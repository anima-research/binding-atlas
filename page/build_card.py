"""Render the Binding Atlas link card (og:image, 1200×630): title + a reduced Figure 2 (dynamic 'you' vs parameters)
from stage.json, as HTML; build_card.sh screenshots it to PNG with headless Chrome.
Usage: python page/build_card.py <out>.html [--stage data/built/stage.json] [--config config/default.toml]
(the configured font's files, if any, in fonts/ next to the output). The URL line on the card is the site config's card text (empty: no URL line)."""
from __future__ import annotations

import argparse
import html as _html
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_page import load_config  # noqa: E402

_ap = argparse.ArgumentParser(description="Render the Binding Atlas link card as HTML.")
_ap.add_argument("out", type=Path)
_ap.add_argument("--stage", type=Path, default=HERE.parent / "data" / "built" / "stage.json")
_ap.add_argument("--config", type=Path, default=None)
_args = _ap.parse_args()
_cfg = load_config(_args.config)
CARD_TEXT = _html.escape(_cfg.card_text)
# the configured web font, if any (files copied next to the card by build_card.sh); else Courier Prime / monospace
FACES = "".join(f'@font-face {{ font-family: "{_cfg.font_family}"; src: url("fonts/{_cfg.font_stem}-{st}.woff2"); '
                f'font-weight: {w}; }}\n' for st, w in (("Regular", 400), ("Bold", 700))) if _cfg.font_family else ""
FAM = f'"{_cfg.font_family}", ' if _cfg.font_family else ""
G = json.loads(_args.stage.read_text())["threshold"]
W, H, L, R, T, B = 560, 440, 20, 16, 12, 46
x0, x1, y0, y1 = math.log10(0.4), math.log10(1500), -0.1, 0.26
X = lambda p: L + (math.log10(p) - x0) / (x1 - x0) * (W - L - R)
Y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
INK, BG, MUTED, LINE, BAND = "#0b0c0e", "#ffffff", "#5a626d", "#d5d9de", "#f1f3f5"


def sq(x: float, y: float, fill: bool, r: float = 5) -> str:
    return (f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" fill="{INK if fill else BG}" '
            f'stroke="{INK}" stroke-width="1.6"/>')


s = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">']
s.append(f'<rect x="{X(8.2):.1f}" y="{T}" width="{X(14.8) - X(8.2):.1f}" height="{H - T - B}" fill="{BAND}"/>')
for t, lab in ((1, "1B"), (10, "10B"), (100, "100B"), (1000, "1T")):
    s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{H - B}" stroke="{LINE}"/>'
             f'<text x="{X(t):.1f}" y="{H - B + 22}" text-anchor="middle" class="t">{lab}</text>')
s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{INK}" stroke-width="1.4"/>')
s.append(f'<text x="{L + 6}" y="{Y(0) - 10:.1f}" class="t">↑ binds</text>'
         f'<text x="{X(100) + 8:.1f}" y="{Y(0) + 22:.1f}" class="t">↓ inverts</text>')
for m in G["models"]:
    x = X(m["p"])
    s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(m["z"]):.1f}" y2="{Y(m["t"]):.1f}" stroke="{LINE}" stroke-width="1.2"/>')
    s.append(sq(x, Y(m["t"]), False) + sq(x, Y(m["z"]), True))
s.append(f'<text x="{(X(8.2) + X(14.8)) / 2:.1f}" y="{H - B + 40}" text-anchor="middle" class="t">8–14B</text>')
s.append(f'<text x="{W - R}" y="{H - B + 40}" text-anchor="end" class="t">parameters (log)</text></svg>')
chart = "".join(s)

html = f'''<!doctype html><html><head><meta charset="utf-8"><style>
{FACES}html, body {{ margin: 0; }}
body {{ width: 1200px; height: 630px; background: {BG}; color: {INK}; font-family: {FAM}"Courier Prime", "Courier New", monospace;
  display: grid; grid-template-columns: 500px 1fr; box-sizing: border-box; padding: 52px 52px 40px 60px; gap: 40px; overflow: hidden; }}
.l {{ display: flex; flex-direction: column; gap: 20px; border-top: 3px solid {INK}; padding-top: 20px; height: 538px; box-sizing: border-box; }}
.eb {{ font-size: 17px; letter-spacing: .14em; text-transform: uppercase; color: {MUTED}; }}
h1 {{ font-size: 56px; line-height: 1.02; margin: 0; font-weight: 700; }}
p {{ font-size: 22px; line-height: 1.38; margin: 0; }}
.cr {{ font-size: 16px; line-height: 1.4; color: {MUTED}; }}
.u {{ margin-top: auto; font-size: 17px; color: {MUTED}; }}
.r {{ display: flex; flex-direction: column; gap: 10px; padding-top: 6px; }}
.k {{ font-size: 15px; color: {MUTED}; display: flex; gap: 22px; padding-left: 20px; }}
.k b {{ display: inline-block; width: 11px; height: 11px; border: 1.6px solid {INK}; margin-right: 7px; vertical-align: -1px; }}
.t {{ font-family: {FAM}"Courier Prime", monospace; font-size: 14px; fill: {MUTED}; }}
</style></head><body>
<div class="l"><div class="eb">kotodama · referent binding</div><h1>Binding Atlas</h1>
<p>Base models learn who <i>you</i> is in five stages. The last one, tracking <i>you</i> as objects change hands, is
never reached zero-shot up to 8B, at any token count measured, and stays weak from 14B to 1T.</p>
<div class="cr">By Luxia, with Claude<br>Compute and hosting by Anima Labs</div>
<div class="u">{CARD_TEXT}</div></div>
<div class="r">{chart}<div class="k"><span><b style="background:{INK}"></b>zero-shot</span><span><b></b>3-shot</span>
<span>30 public bases</span></div></div>
</body></html>'''
_args.out.parent.mkdir(parents=True, exist_ok=True)
_args.out.write_text(html)
print("wrote", _args.out)
