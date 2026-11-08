"""Edit a page's words in place: a local preview server for the copy files (copy/*.md).

    python3 page/edit_server.py [--page announce] [--port 8770]      # or: make edit

It builds the page's editing preview into a git-ignored scratch folder and serves it at http://127.0.0.1:PORT/ (this machine only).
Clicking outlined text opens it in a side panel; saving writes that one entry to the copy file that holds it, rebuilds
and reloads. A page can draw on several copy files (the announcement: its own words and the story's), so a key may
appear in only one of them. Text that never shows on the page (alt text, link-preview titles, citation leaflets,
contents labels) is under "Other text". Editing a copy file in any editor works too: the preview rebuilds and reloads
when it changes. Files the preview does not build (fonts, other pages) are served from site/. Nothing here deploys:
publishing still goes through the gates.

The story's fragment is built by the card builder named in ``$KOTODAMA_STORY_BUILDER`` (``make edit`` passes
STORY_BUILDER from the machine-local make settings); without it, the preview inlines the committed story and its words are not
editable here.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from html.parser import HTMLParser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from copy_text import CITE, COPY_DIR, CopyError, add_entry, check_text, parse, write_entry  # noqa: E402

EDITOR_DIR = HERE / "editor"
OUT = REPO / "out" / "edit"
_SB = os.environ.get("KOTODAMA_STORY_BUILDER", "")
STORY_BUILDER = Path(_SB) if _SB else None
MARK = re.compile(r'data-copy=\\?"([^"\\]+)')
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


@dataclass(frozen=True)
class Page:
    copies: tuple[Path, ...]   # the copy files whose words appear on the page; the first is the page's own
    template: Path
    builder: Path
    out_name: str
    story: bool = False        # build the story fragment in edit mode and inline it


PAGES = {"announce": Page((COPY_DIR / "announce.md", COPY_DIR / "story.md"), HERE / "announce_page.html",
                          HERE / "build_announce_page.py", "index.html", story=True)}


class _Balance(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes nothing open" if tag not in self.stack else f"<{self.stack[-1]}> is not closed before </{tag}>")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
            return
        self.stack.pop()


def html_problem(text: str) -> str | None:
    """Why ``text`` would break the page's markup, or None."""
    p = _Balance()
    p.feed(text)
    p.close()
    if p.errors:
        return p.errors[0]
    if p.stack:
        return f"<{p.stack[-1]}> is never closed"
    return None


class Preview:
    def __init__(self, page: Page, config: Path) -> None:
        self.page, self.config = page, config
        self.lock = threading.Lock()
        self.version = 0
        self.built_mtime = -1.0
        self.error: str | None = None
        self.story = page.story and STORY_BUILDER is not None and STORY_BUILDER.exists()
        self.copies = tuple(c for c in page.copies if c.exists() and (self.story or c.name != "story.md"))
        self.warning = (None if self.story or not page.story else
                        "story words not editable here: set STORY_BUILDER in config/local.mk (the card builder's path)")
        seen: dict[str, Path] = {}
        for c in self.copies:
            for k in parse(c):
                if k in seen:
                    raise SystemExit(f"key {k!r} is in both {seen[k].name} and {c.name}; a page's keys must be unique")
                seen[k] = c

    def _mtime(self) -> float:
        return max([c.stat().st_mtime for c in self.copies] + [self.page.template.stat().st_mtime])

    def file_of(self, key: str) -> Path | None:
        return next((c for c in self.copies if key in parse(c)), None)

    def _run(self, cmd: list[str]) -> str | None:
        env = {**os.environ, "FACTS_CURRENT_BINDING": str(REPO / "data" / "inputs" / "synthesis.jsonl")}
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env)
        return None if r.returncode == 0 else ((r.stderr or r.stdout).strip().splitlines() or ["build failed"])[-1]

    def build(self) -> str | None:
        """Rebuild the preview; the first builder's complaint, or None."""
        mt = self._mtime()
        cmd = [sys.executable, str(self.page.builder), "--config", str(self.config), "--edit",
               "--out", str(OUT / self.page.out_name)]
        err = None
        if self.story:
            anim = OUT / "_story"
            anim.mkdir(parents=True, exist_ok=True)
            err = self._run([sys.executable, str(STORY_BUILDER), str(anim / "video.html"),
                             str(REPO / "data" / "inputs" / "announce" / "card.png"), str(anim / "story.html"),
                             "--copy-dir", str(COPY_DIR), "--edit"])
            cmd += ["--story", str(anim / "story.html")]
        err = err or self._run(cmd)
        self.built_mtime = mt
        self.error = err
        self.version += 1
        return err

    def refresh(self) -> None:
        with self.lock:
            if self._mtime() != self.built_mtime:
                self.build()

    def save(self, key: str, text: str, cites: dict[str, str] | None = None) -> str | None:
        """Write one entry, plus any new citations it introduces (name -> leaflet text, added to the page's own copy
        file); on any refusal every copy file is put back as it was."""
        cites = cites or {}
        with self.lock:
            try:
                text = check_text(key, text)
                cites = {n: check_text(f"cite.{n}", t) for n, t in cites.items()}
            except CopyError as e:
                return str(e)
            for k, t in [(key, text)] + [(f"cite.{n}", t) for n, t in cites.items()]:
                problem = html_problem(t)
                if problem:
                    return f"{k}: {problem}"
            f = self.file_of(key)
            if f is None:
                return f"no entry {key!r}"
            for n in cites:
                if not CITE.fullmatch("{{cite:" + n + "}}"):
                    return f"citation name {n!r}: use letters, digits, - or _ only"
                if self.file_of(f"cite.{n}") is not None:
                    return f"a citation named {n!r} already exists; use {{{{cite:{n}}}}} as it is, or pick another name"
            snapshot = {c: c.read_text() for c in self.copies}
            try:
                for n, t in cites.items():
                    add_entry(self.copies[0], f"cite.{n}", t)
                write_entry(f, key, text)
            except CopyError as e:
                for c, s in snapshot.items():
                    c.write_text(s)
                return str(e)
            err = self.build()
            if err:
                for c, s in snapshot.items():
                    c.write_text(s)
                self.build()
                return f"not saved; the build refused it: {err}"
            return None

    def state(self) -> dict:
        """Every entry, which file holds it, and the "other text": entries with no marked place on the built page."""
        entries: dict[str, str] = {}
        files: dict[str, str] = {}
        for c in self.copies:
            for k, e in parse(c).items():
                entries[k], files[k] = e.text, str(c.relative_to(REPO))
        built = OUT / self.page.out_name
        shown = set(MARK.findall(built.read_text())) if built.exists() else set()
        return {"entries": entries, "files": files, "other": [k for k in entries if k not in shown],
                "version": self.version, "error": self.error, "warning": self.warning,
                "file": " + ".join(str(c.relative_to(REPO)) for c in self.copies)}


def handler(preview: Preview):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw) -> None:
            super().__init__(*a, directory=str(OUT), **kw)

        def log_message(self, fmt: str, *args) -> None:
            if "/__version" not in str(args[0] if args else ""):
                super().log_message(fmt, *args)

        def _json(self, obj: dict, code: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def translate_path(self, path: str) -> str:
            p = Path(super().translate_path(path))
            if path.startswith("/__editor/"):
                return str(EDITOR_DIR / Path(path.split("?")[0]).name)
            if not p.exists():
                alt = REPO / "site" / p.relative_to(OUT)
                if alt.exists():
                    return str(alt)
            return str(p)

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self) -> None:
            if self.path.startswith("/__version"):
                preview.refresh()
                return self._json({"version": preview.version, "error": preview.error})
            if self.path.startswith("/__copy"):
                preview.refresh()
                return self._json(preview.state())
            if self.path in ("/", "/index.html"):
                preview.refresh()
            super().do_GET()

        def do_POST(self) -> None:
            if not self.path.startswith("/__save"):
                return self._json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n))
                key, text = str(req["key"]), str(req["text"])
                cites = {str(n): str(t) for n, t in (req.get("cites") or {}).items()}
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                return self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            err = preview.save(key, text, cites)
            self._json({"error": err, "version": preview.version}, HTTPStatus.OK if err is None else HTTPStatus.UNPROCESSABLE_ENTITY)
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", choices=sorted(PAGES), default="announce")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--config", type=Path, default=REPO / "config" / "default.toml")
    a = ap.parse_args()
    page = PAGES[a.page]
    for f in (page.copies[0], page.template, page.builder, EDITOR_DIR / "editor.js", EDITOR_DIR / "editor.css"):
        if not f.exists():
            raise SystemExit(f"missing: {f}")
    preview = Preview(page, a.config)
    err = preview.build()
    if err:
        raise SystemExit(f"the preview does not build: {err}")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(preview))
    if preview.warning:
        print(f"note: {preview.warning}", flush=True)
    names = " + ".join(str(c.relative_to(REPO)) for c in preview.copies)
    print(f"editing {names}: open http://127.0.0.1:{a.port}/   (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
