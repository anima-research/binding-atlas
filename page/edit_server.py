"""Edit a page's words in place: a local preview server for the copy files (copy/<page>.md).

    python3 page/edit_server.py [--page announce] [--port 8770]      # or: make edit

It builds the page's editing preview into out/edit/ and serves it at http://127.0.0.1:PORT/ (this machine only).
Clicking outlined text opens it in a side panel; saving writes that one entry of the copy file, rebuilds and reloads.
Text that never shows on the page (alt text, link-preview titles, citation leaflets) is under "Other text". Editing
the copy file in any editor works too: the preview rebuilds and reloads when the file changes. Files the preview does
not build (fonts, other pages) are served from site/. Nothing here deploys: publishing still goes through the gates.
"""
from __future__ import annotations

import argparse
import json
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
from copy_text import COPY_DIR, CopyError, check_text, fill, leaflets, parse, write_entry  # noqa: E402

EDITOR_DIR = HERE / "editor"
OUT = REPO / "out" / "edit"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}


@dataclass(frozen=True)
class Page:
    copy: Path
    template: Path
    builder: Path
    out_name: str


PAGES = {"announce": Page(COPY_DIR / "announce.md", HERE / "announce_page.html", HERE / "build_announce_page.py", "index.html")}


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

    def _mtime(self) -> float:
        return max(self.page.copy.stat().st_mtime, self.page.template.stat().st_mtime)

    def build(self) -> str | None:
        """Rebuild the preview; the builder's complaint, or None."""
        mt = self._mtime()
        r = subprocess.run([sys.executable, str(self.page.builder), "--config", str(self.config), "--edit",
                            "--out", str(OUT / self.page.out_name)], capture_output=True, text=True, cwd=REPO)
        self.built_mtime = mt
        self.error = None if r.returncode == 0 else (r.stderr.strip().splitlines() or ["build failed"])[-1]
        self.version += 1
        return self.error

    def refresh(self) -> None:
        with self.lock:
            if self._mtime() != self.built_mtime:
                self.build()

    def save(self, key: str, text: str) -> str | None:
        with self.lock:
            try:
                text = check_text(key, text)
            except CopyError as e:
                return str(e)
            problem = html_problem(text)
            if problem:
                return f"{key}: {problem}"
            entries = parse(self.page.copy)
            if key not in entries:
                return f"no entry {key!r}"
            before = entries[key].text
            write_entry(self.page.copy, key, text)
            err = self.build()
            if err:
                write_entry(self.page.copy, key, before)
                self.build()
                return f"not saved; the build refused it: {err}"
            return None

    def state(self) -> dict:
        entries = parse(self.page.copy)
        _, in_tags = fill(self.page.template.read_text(), self.page.copy)
        return {"entries": {k: e.text for k, e in entries.items()}, "other": in_tags + leaflets(self.page.copy),
                "version": self.version, "error": self.error, "file": str(self.page.copy.relative_to(REPO))}


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
            except (ValueError, KeyError, TypeError) as e:
                return self._json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
            err = preview.save(key, text)
            self._json({"error": err, "version": preview.version}, HTTPStatus.OK if err is None else HTTPStatus.UNPROCESSABLE_ENTITY)
    return H


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", choices=sorted(PAGES), default="announce")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--config", type=Path, default=REPO / "config" / "default.toml")
    a = ap.parse_args()
    page = PAGES[a.page]
    for f in (page.copy, page.template, page.builder, EDITOR_DIR / "editor.js", EDITOR_DIR / "editor.css"):
        if not f.exists():
            raise SystemExit(f"missing: {f}")
    preview = Preview(page, a.config)
    err = preview.build()
    if err:
        raise SystemExit(f"the preview does not build: {err}")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), handler(preview))
    print(f"editing {page.copy.relative_to(REPO)}: open http://127.0.0.1:{a.port}/   (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
