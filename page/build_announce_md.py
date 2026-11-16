"""The announcement's plain-text twin (index.md): the same words as index.html, for language models and screen readers.

    from build_announce_md import announce_md; md = announce_md(page_html, base_url, stamp)

It reads the built page, not the copy files, so the twin can't drift from what the page says:
- every dog-ear note and every reference is expanded where its marker stands ("phrase [note: …]", "[ref: …]");
- every figure in the story is described in text from the same data object (`D`), labels (`COPY`) and design-spec
  cards (`ARCH`) the page draws it from;
- relative links become absolute; links to places on the page itself become plain text.
Decoration (the background kanji, the contents rail, the WIP band, scripts) is left out.
"""
from __future__ import annotations

import html
import json
import math
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

SKIP_CLASSES = {"bg-kanji", "story-stage", "wip-band"}
SKIP_IDS = {"toc"}
VOID = {"br", "img", "meta", "link", "input", "hr", "source", "wbr"}
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


class TwinError(ValueError):
    """The page lacks something the twin needs (its data object, a story beat, a figure's labels)."""


def _object(page: str, name: str):
    m = re.search(rf"\b{name}\s*=\s*", page)
    if not m:
        raise TwinError(f"the page has no {name} object")
    value, _ = json.JSONDecoder().raw_decode(page[m.end():])
    return value


def inline(fragment: str, base_url: str) -> str:
    """A small HTML fragment (a label, a card body) as one line of Markdown."""
    conv = _Converter(base_url)
    conv.feed(fragment)
    conv.close()
    return re.sub(r"\s+", " ", conv.text()).strip()


class _Converter(HTMLParser):
    """Page HTML -> Markdown blocks. Story beats get their figure description appended by `figure` (if given)."""

    def __init__(self, base_url: str, figure=None):
        super().__init__(convert_charrefs=True)
        self.base, self.figure = base_url, figure
        self.out: list[str] = []          # finished blocks
        self.buf: list[str] = []          # the current block's inline text
        self.stack: list[tuple[str, dict]] = []
        self.skip = 0
        self.cite: list[dict] = []        # open .cite spans: {"note": bool, "mk": [...], "leaf": [...], "part": ...}
        self.href: list[str | None] = []
        self.beat: str | None = None
        self.beats_seen: list[str] = []

    # ---- output helpers
    def _w(self, s: str) -> None:
        if self.cite:
            c = self.cite[-1]
            c[c["part"]].append(s) if c["part"] else None
        else:
            self.buf.append(s)

    def _flush(self, prefix: str = "") -> None:
        t = re.sub(r"[ \t]+", " ", "".join(self.buf)).strip()
        t = re.sub(r" *\n *", "\n", t)
        if t:
            self.out.append(prefix + t)
        self.buf = []

    def text(self) -> str:
        self._flush()
        return "\n\n".join(self.out)

    # ---- parsing
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = set((a.get("class") or "").split())
        if tag not in VOID:
            self.stack.append((tag, a))
        if self.skip or tag in ("script", "style", "svg", "noscript") or cls & SKIP_CLASSES or a.get("id") in SKIP_IDS:
            if tag not in VOID:
                self.skip += 1
            return
        if tag == "span" and "cite" in cls:
            self.cite.append({"note": "note" in cls, "mk": [], "leaf": [], "part": None})
            return
        if self.cite:
            c = self.cite[-1]
            if tag == "button" and "cite-mk" in cls:
                c["part"] = "mk"
                return
            if tag == "span" and "cite-leaf" in cls:
                c["part"] = "leaf"
                return
        if tag == "button":           # interface controls carry no page text
            self.skip += 1
            return
        if tag in ("p", "li", "h1", "h2", "h3", "h4") or (tag == "div" and cls & {"h", "byline", "gloss", "map", "lede"}):
            self._flush()
        if tag == "div" and "story-step" in cls:
            self._flush()
            self.beat = a.get("data-beat")
        if tag == "span" and "beat" in cls:      # the lede's lines
            self._flush()
        if tag == "span" and any(t == "footer" for t, _ in self.stack[:-1]):
            self._flush()
        if tag == "span" and len(self.stack) > 1 and "gloss" in (self.stack[-2][1].get("class") or ""):
            self._flush()                         # one gloss per line
        if tag == "br":
            self._w("\n")
        elif tag in ("b", "strong"):
            self._w("**")
        elif tag in ("i", "em"):
            self._w("*")
        elif tag == "code":
            self._w("`")
        elif tag == "sub":
            self._w("_")
        elif tag == "small":
            self._w(" ")
        elif tag == "sup":
            self._w("^")
        elif tag == "a":
            href = a.get("href") or ""
            self.href.append(None if href.startswith("#") or not href else urljoin(self.base, href))
            if self.href[-1]:
                self._w("[")

    def handle_endtag(self, tag):
        if tag in VOID or not self.stack:
            return
        # pop to the matching tag (the page is well-formed; this guards stray closers)
        while self.stack:
            t, a = self.stack.pop()
            if t == tag:
                break
        cls = set((a.get("class") or "").split())
        if self.skip:
            if tag in ("script", "style", "svg", "noscript", "button") or cls & SKIP_CLASSES or a.get("id") in SKIP_IDS \
                    or self.skip:
                self.skip -= 1
            return
        if self.cite:
            c = self.cite[-1]
            if tag == "button" and "cite-mk" in cls:
                c["part"] = None
                return
            if tag == "span" and "cite-leaf" in cls:
                c["part"] = None
                return
            if tag == "span" and "cite" in cls:
                self.cite.pop()
                mk = re.sub(r"\s+", " ", "".join(c["mk"])).strip()
                leaf = re.sub(r"\s+", " ", "".join(c["leaf"])).strip()
                self._w(f"{mk} [note: {leaf}]" if c["note"] else f" [ref: {leaf}]")
                return
        if tag in ("b", "strong"):
            self._w("**")
            if any(t == "li" for t, _ in self.stack):     # a list item's bold label: "**Weights**: …"
                self._w(": ")
        elif tag in ("i", "em"):
            self._w("*")
        elif tag == "code":
            self._w("`")
        elif tag == "a":
            href = self.href.pop() if self.href else None
            if href:
                self._w(f"]({href})")
        elif tag == "h1":
            self._flush("# ")
        elif tag == "h2":
            self._flush("## ")
        elif tag == "h3":
            self._flush("### ")
        elif tag == "li":
            self._flush("- ")
        elif tag == "div" and "h" in cls and self.beat:
            self._flush("### ")
        elif tag in ("p", "div", "span") and (tag == "p" or cls & {"byline", "gloss", "map", "beat", "lede"}):
            self._flush()
        elif tag == "span" and self.stack and "gloss" in (self.stack[-1][1].get("class") or ""):
            self._flush()
        elif tag == "span" and any(t == "footer" for t, _ in self.stack):
            self._flush()
        if tag == "div" and "story-step" in cls and self.beat:
            self._flush()
            if self.figure:
                self.out.append(self.figure(self.beat))
            self.beats_seen.append(self.beat)
            self.beat = None

    def handle_data(self, data):
        if self.skip:
            return
        self._w(data)


# ------------------------------------------------------------------ the figures, from the page's own data
def _sci(x: float) -> str:
    e = math.floor(math.log10(x))
    return f"{x / 10 ** e:.1f}e{e}"


def _f3(v: float, sign: bool = False) -> str:
    s = f"{v:+.3f}" if sign else f"{v:.3f}"
    return s.replace("0.", ".", 1)


def _at(curve: list, x: float) -> float:
    """Linear interpolation on a [[x, y], …] polyline."""
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise TwinError(f"x={x} is outside the trend's range")


def figures(page: str, base_url: str):
    D, C, ARCH = _object(page, "D"), _object(page, "COPY"), _object(page, "ARCH")
    L = lambda k: inline(C[k], base_url)  # noqa: E731

    def arch() -> str:
        b = D["bounds"]
        out = ["**Figure: design spec** (an animation; described in text).",
               f"A stack of 72 layers, drawn bottom to top: {L('stack.input')} ({L('stack.input_n')}), "
               f"{L('stack.middle')} ({L('stack.middle_sub')}), {L('stack.middle_again')} ({L('stack.middle_again_sub')}), "
               f"{L('stack.output')} ({L('stack.output_n')}); a bracket marks the middle block as run ×2. Lines mark where "
               f"the attention-residual blocks start, at layers {', '.join(map(str, b[:-1]))}, {D['bound_loop']} and {b[-1]}; "
               f"they first appear evenly spaced (at {', '.join(map(str, D['bounds_even']))}) and slide to these fitted places.",
               "Beside the stack, the four design steps, added one at a time:"]
        for i, (head, body, formula) in enumerate(ARCH, 1):
            f = inline(formula.replace("<br>", " · "), base_url) if formula else ""
            out.append(f"{i}. **{head}**: {inline(body, base_url)}" + (f" Formula: {f}" if f else ""))
        out.append(L("story.arch.together"))
        return "\n\n".join(out)

    def rules() -> str:
        h = D["hero"]
        tx, ty = h["t"]
        trend = h["trend"]
        at_t = trend[0][1] + (trend[1][1] - trend[0][1]) * (tx - trend[0][0]) / (trend[1][0] - trend[0][0])
        n_dist = sum(1 for p in h["pub"] if p[2])
        items = "; ".join(f"{a} → {b}" for a, b in D["item"])
        out = ["**Figure: in-context learning** (described in text).",
               f"Left, one item of the test: worked examples {items}, then “… 4 more examples …”, then the query "
               f"{D['query']} → ?, with five options ({', '.join(D['choices'])}); the right answer is {D['gold']} "
               "(1 right answer, 5 options).",
               f"Right, a scatter. Vertical axis: {L('rules.hero_y')}. Horizontal axis: {L('rules.hero_x')}, log scale "
               f"from {_sci(10 ** h['xr'][0])} to {_sci(10 ** h['xr'][1])}. Points: {len(h['pub'])} "
               f"{L('rules.hero_legend_full')} of open-weight base models (dots; {n_dist} of them "
               f"{L('rules.hero_legend_distilled')}, drawn as asterisks), {len(h['mid'])} {L('rules.hero_legend_mid')} "
               f"(hollow circles), and {L('rules.hero_legend_t')} (a highlighted diamond, labelled {_f3(ty)}) at "
               f"{_sci(10 ** tx)} FLOPs. A dashed line, the {L('rules.hero_legend_trend')}, runs from {_f3(trend[0][1])} at "
               f"{_sci(10 ** trend[0][0])} to {_f3(trend[1][1])} at {_sci(10 ** trend[1][0])}; at tsumugi's compute it "
               f"gives {_f3(at_t)}."]
        return "\n\n".join(out)

    def you() -> str:
        bar = D["dyn_bar"]
        chat = "\n".join(f"> {line}" for line in D["chat"])
        rows = ["| model | parameters | average drop | 95% interval | whole interval above +.02 |", "|---|---|---|---|---|"]
        for name, v, ci, params, ours in D["dyn"]:
            nm = f"**{name}**" if ours else name
            rows.append(f"| {nm} | {params:.2f}B | {_f3(v, True)} | {_f3(ci[0], True)} to {_f3(ci[1], True)} | "
                        f"{'yes' if ci[0] > bar else 'no'} |")
        prompt = L("you.prompt")
        out = ["**Figure: referent binding** (an animation; described in text).",
               f"Left, {L('you.chat_label')}:", chat,
               f"> Steve: {prompt}",
               f"With Steve saying the last line, {L('you.is_kurt')}; the probability of “battery” is {_f3(D['p_orig'])}. "
               f"The figure then swaps who says that line (“Kurt: {prompt}”), so {L('you.is_steve')}, and the probability "
               f"of “battery” falls to {_f3(D['p_swap'])}. {L('you.moves')[:1].upper() + L('you.moves')[1:]}.",
               f"Right, {L('you.bars_title')} (sorted by size; bars with 95% whiskers; a dashed line marks the +.02 bar; "
               f"left of zero: {L('you.left').strip('← ')}; right: {L('you.right').strip(' →')}; "
               f"{L('you.zero')}):",
               "\n".join(rows)]
        return "\n\n".join(out)

    def cheap() -> str:
        B = D["base"]
        cols = B["cols"]
        head = "| | " + " | ".join(c["lab"] for c in cols) + " |"
        rows = [head, "|---" * (len(cols) + 1) + "|"]
        for i, task in enumerate(B["tasks"]):
            rows.append(f"| {task} | " + " | ".join(f"{c['vals'][i]:.1f}" for c in cols) + " |")
        rows.append(f"| **{L('cheap.avg')}** | " + " | ".join(f"{sum(c['vals']) / len(c['vals']):.1f}" for c in cols) + " |")
        rows.append(f"| {L('cheap.compute')} | " + " | ".join(
            (f"{c['mult']:.0f}×" if c["mult"] >= 10 else f"{c['mult']:.1f}×".replace("1.0×", "1×")) + ("*" if c["dist"] else "")
            for c in cols) + " |")
        tx, ty = B["t"]
        lx = math.log10(tx)
        at_t = _at(B["trend"], lx)
        pts = B["dense"]
        out = ["**Figure: world knowledge evals** (described in text).",
               f"Left, {L('cheap.grid_title')}, one column per model in order of training compute "
               f"(cells shaded by score within each row; {L('cheap.foot')} from a larger model):",
               "\n".join(rows),
               f"Right, {L('cheap.bpb_title')}, against {L('cheap.bpb_x')} (log scale): {len(pts)} {L('cheap.models')} "
               f"(dots, from {min(p[1] for p in pts):.3f} to {max(p[1] for p in pts):.3f} bits per byte) and their "
               f"{L('cheap.trend')} (a dashed least-squares line on log compute, fitted after dropping any model more than "
               f"2 SD from it, which drops Pythia 160M). tsumugi is a highlighted diamond labelled {_f3(ty)} at "
               f"{_sci(tx)} FLOPs, below the line, which gives {_f3(at_t)} at that compute."]
        return "\n\n".join(out)

    def data() -> str:
        DD = D["data"]
        kinds = [("s", L("data.k_s")), ("v", L("data.k_v")), ("x", L("data.k_x")), ("h", L("data.k_h")), ("p", L("data.k_p"))]
        tiles = []
        for r in DD["rows"]:
            counts = {k: r["tiles"].count(k) for k, _ in kinds}
            tiles.append(f"- {r['label']}: {r['era']} of {r['n']} {L('data.era')} ("
                         + ", ".join(f"{name} {counts[k]}" for k, name in kinds if counts[k]) + ")")
        rows = ["| year | model | " + " | ".join(name for _, name in kinds) + " | with signs of synthetic or LLM data |",
                "|---" * (len(kinds) + 3) + "|"]
        for m in DD["all"]:
            nm = f"**{m['label']}**" if m.get("hero") or "Falcon3 3B" in m["label"] else m["label"]
            rows.append(f"| {m['year']} | {nm} | " + " | ".join(str(m["counts"].get(k, 0)) for k, _ in kinds)
                        + f" | {m['era']} of {m['n']} |")
        runs = []
        for r in DD["runs"]:
            kind = dict(kinds).get(r["c"], r["c"])
            text = "".join(seg[1] for seg in r["text"]).strip()
            quoted = "\n".join("> " + ln if ln.strip() else ">" for ln in text.splitlines())
            runs.append(f"{r['label']} ({kind}):\n\n> {L('data.start')}\n{quoted}")
        out = ["**Figure: blank document completions** (an animation; described in text).",
               f"First, {L('data.tiles_title')}, coloured by kind:", "\n".join(tiles),
               f"Then {L('data.more')}:", "\n".join(rows),
               "Beside them, one run of each of the first two models, verbatim:", "\n\n".join(runs),
               f"Under the runs: [{L('data.link').rstrip(' →')}](" + urljoin(base_url, "empty-document") + ")."]
        return "\n\n".join(out)

    table = {"arch": arch, "rules": rules, "you": you, "cheap": cheap, "data": data}

    def figure(beat: str) -> str:
        if beat not in table:
            raise TwinError(f"no figure description for story beat {beat!r}")
        return table[beat]()

    return figure


def announce_md(page: str, base_url: str, stamp: str) -> str:
    figure = figures(page, base_url)
    title = re.search(r"<title>(.*?)</title>", page, re.S)
    body = page[page.index("<body"):]
    conv = _Converter(base_url, figure)
    conv.feed(body)
    conv.close()
    if conv.beats_seen != ["arch", "rules", "you", "cheap", "data"]:
        raise TwinError(f"story beats seen: {conv.beats_seen}")
    text = re.sub(r"\n{3,}", "\n\n", conv.text())
    head = (f"<!-- {html.unescape(title.group(1).strip()) if title else 'kotodama-tsumugi'}: plain-text version of "
            f"{base_url} for language models and screen readers. The same words as the page; every note and reference "
            f"is expanded where it appears, and every figure is described in text from the data it draws. "
            f"Build {stamp}. -->")
    return head + "\n\n" + text.strip() + "\n"
