"""Page copy: the words of a page live in a copy file, one keyed entry per string, and the template holds slots.

Copy file (``copy/<page>.md``): anything before the first ``## `` line is notes for the editor and is ignored. Each
entry is a ``## key`` line followed by its text, up to the next ``## `` line; surrounding blank lines are trimmed and
line breaks inside an entry are kept (HTML treats them as spaces). Text is an HTML fragment: links, ``<i>``, ``<b>``
and the site's link placeholders (``__ATLAS_LINK__`` …) work as in the template.

Template slots:
  ``{{copy:key}}``   the entry's text. In edit mode a slot in page text is wrapped in ``<span data-copy="key">`` so the
                     editor can find it; a slot inside a tag (an attribute such as ``alt`` or a meta ``content``, or ``<title>``) is not
                     wrapped, and the editor lists it under "other text".
Inside an entry:
  ``{{cite:NAME}}``  a citation marker whose leaflet holds entry ``cite.NAME``; markers are numbered by first use.

The fill refuses a slot without an entry, an entry no slot uses (``cite.*`` entries count as used by their markers),
an entry whose text would start a new entry, and a citation without its ``cite.`` entry.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

SLOT = re.compile(r"\{\{copy:([a-z0-9_.-]+)\}\}")
CITE = re.compile(r"\{\{cite:([A-Za-z0-9_-]+)\}\}")
KEY_LINE = re.compile(r"^## ([a-z0-9_.-]+|cite\.[A-Za-z0-9_-]+)[ \t]*$")
REPO = Path(__file__).resolve().parent.parent
COPY_DIR = REPO / "copy"


class CopyError(ValueError):
    """A copy file or a template that does not fit the rules above."""


@dataclass(frozen=True)
class Entry:
    key: str
    text: str
    start: int   # line index of the "## key" line
    end: int     # line index one past the entry's last line


def parse(path: Path) -> dict[str, Entry]:
    """Entries of a copy file, in file order."""
    lines = path.read_text().split("\n")
    heads = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
    out: dict[str, Entry] = {}
    for n, i in enumerate(heads):
        m = KEY_LINE.match(lines[i])
        if not m:
            raise CopyError(f"{path.name}:{i + 1}: not a key line: {lines[i]!r}")
        key = m.group(1)
        if key in out:
            raise CopyError(f"{path.name}:{i + 1}: key {key!r} appears twice")
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        out[key] = Entry(key, "\n".join(lines[i + 1:end]).strip("\n").strip(), i, end)
    return out


def check_text(key: str, text: str) -> str:
    """The text an editor saves for ``key``, normalized; refuses text that would break the file."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if any(ln.startswith("## ") for ln in text.split("\n")):
        raise CopyError(f"{key}: a line of the text starts with '## ', which would begin a new entry")
    if not text:
        raise CopyError(f"{key}: empty text")
    return text


def write_entry(path: Path, key: str, text: str) -> None:
    """Replace one entry's text in place; every other line of the file is kept byte for byte."""
    text = check_text(key, text)
    entries = parse(path)
    if key not in entries:
        raise CopyError(f"{path.name}: no entry {key!r}")
    e = entries[key]
    lines = path.read_text().split("\n")
    tail_blank = e.end < len(lines)
    lines[e.start + 1:e.end] = text.split("\n") + ([""] if tail_blank else [])
    path.write_text("\n".join(lines))


def add_entry(path: Path, key: str, text: str) -> None:
    """Append a new entry at the end of the file; refuses a key the file already has or one the key rules reject."""
    text = check_text(key, text)
    if not KEY_LINE.match(f"## {key}"):
        raise CopyError(f"{key!r} is not a valid key")
    if key in parse(path):
        raise CopyError(f"{path.name}: entry {key!r} already exists")
    s = path.read_text().rstrip("\n")
    path.write_text(f"{s}\n\n## {key}\n{text}\n")


def _inside_tag(page: str, pos: int) -> bool:
    """A slot inside a tag's attributes, or inside <title> (whose text cannot hold markup)."""
    return (page.rfind("<", 0, pos) > page.rfind(">", 0, pos)
            or page.rfind("<title>", 0, pos) > page.rfind("</title>", 0, pos))


def _cite(name: str, n: int, leaf: str) -> str:
    return (f'<span class="cite"><button type="button" class="cite-mk" aria-expanded="false" '
            f'aria-label="Reference: {html.escape(name)}">{n}</button><span class="cite-leaf" role="note">{leaf}</span></span>')


def fill(template: str, path: Path, edit: bool = False) -> tuple[str, list[str]]:
    """The template with every slot filled from the copy file, plus the keys whose slots sit inside a tag."""
    entries = parse(path)
    texts = {k: e.text for k, e in entries.items()}
    used: set[str] = set()
    numbers: dict[str, int] = {}
    in_tags: list[str] = []
    missing = sorted({m.group(1) for m in SLOT.finditer(template)} - texts.keys())
    if missing:
        raise CopyError(f"{path.name}: the template has slots with no entry: {missing}")

    def cites(key: str, text: str) -> str:
        def one(m: re.Match[str]) -> str:
            name = m.group(1)
            if f"cite.{name}" not in texts:
                raise CopyError(f"{key}: citation {name!r} has no entry 'cite.{name}'")
            used.add(f"cite.{name}")
            numbers.setdefault(name, len(numbers) + 1)
            return _cite(name, numbers[name], texts[f"cite.{name}"])
        return CITE.sub(one, text)

    parts: list[str] = []
    last = 0
    for m in SLOT.finditer(template):
        key = m.group(1)
        used.add(key)
        parts.append(template[last:m.start()])
        text = cites(key, texts[key])
        if _inside_tag(template, m.start()):
            if "<" in text or '"' in text:
                raise CopyError(f"{key}: text used inside a tag may not contain '<' or '\"'")
            in_tags.append(key)
            parts.append(text)
        else:
            parts.append(f'<span data-copy="{key}">{text}</span>' if edit else text)
        last = m.end()
    parts.append(template[last:])
    unused = sorted(texts.keys() - used)
    if unused:
        raise CopyError(f"{path.name}: entries no slot uses: {unused}")
    return "".join(parts), in_tags


def leaflets(path: Path) -> list[str]:
    """The ``cite.*`` keys of a copy file: text that shows only in a leaflet, listed by the editor as other text."""
    return [k for k in parse(path) if k.startswith("cite.")]
