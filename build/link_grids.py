"""Assemble a grids root (the logical layout of data/INPUTS-MANIFEST.tsv) from local copies, as a tree of symlinks.

Usage: python build/link_grids.py MAP.tsv OUT_DIR [--verify]

MAP.tsv maps logical path prefixes to local directories, one per line: `<logical prefix>\\t<local directory>`
(`#` comments allowed). The longest matching prefix wins; see config/local-grids.example.tsv. Every manifest file
(root=grids) must resolve to an existing local file of the listed size; --verify also checks each sha256 (reads all
of them, ~1.9 GB). OUT_DIR is then usable as `--grids OUT_DIR` / `ATLAS_GRIDS=OUT_DIR`.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "data" / "INPUTS-MANIFEST.tsv"


@dataclass(frozen=True)
class Entry:
    root: str
    path: str
    size: int
    sha256: str


def read_manifest(path: Path = MANIFEST) -> list[Entry]:
    out: list[Entry] = []
    for i, ln in enumerate(path.read_text().splitlines(), 1):
        if not ln.strip() or ln.startswith("#") or ln.startswith("root\t"):
            continue
        parts = ln.split("\t")
        if len(parts) != 4:
            raise SystemExit(f"{path.name}:{i}: expected 4 tab-separated fields, got {len(parts)}")
        root, p, size, sha = parts
        if root not in ("grids", "repo") or p.startswith("/") or ".." in Path(p).parts:
            raise SystemExit(f"{path.name}:{i}: bad entry {ln!r}")
        out.append(Entry(root, p, int(size), sha))
    return out


def read_map(path: Path) -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for i, ln in enumerate(path.read_text().splitlines(), 1):
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) != 2:
            raise SystemExit(f"{path}:{i}: expected `<logical prefix>\\t<local directory>`")
        pairs.append((parts[0].strip(), Path(os.path.expanduser(parts[1].strip()))))
    return sorted(pairs, key=lambda kv: -len(kv[0]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("map", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--verify", action="store_true", help="also check every sha256")
    a = ap.parse_args()
    if not a.map.exists():
        raise SystemExit(f"path map not found: {a.map} (copy config/local-grids.example.tsv and fill it in)")
    pairs = read_map(a.map)
    if a.out.exists():   # start clean: a stale link would change which grids the builder finds (it probes existence)
        for p in sorted(a.out.rglob("*"), key=lambda q: -len(q.parts)):
            if p.is_symlink():
                p.unlink()
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()
    errors: list[str] = []
    n = 0
    for e in read_manifest():
        if e.root == "repo":
            local = REPO / e.path
        else:
            hit = next(((pre, d) for pre, d in pairs if e.path.startswith(pre)), None)
            if hit is None:
                errors.append(f"no mapping for {e.path}")
                continue
            local = hit[1] / e.path[len(hit[0]):]
        if not local.is_file():
            errors.append(f"missing: {e.path} -> {local}")
            continue
        if local.stat().st_size != e.size:
            errors.append(f"size {local.stat().st_size} != {e.size}: {e.path}")
            continue
        if a.verify and sha256(local) != e.sha256:
            errors.append(f"sha256 mismatch: {e.path}")
            continue
        if e.root == "grids":
            dst = a.out / e.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(local.resolve())
            n += 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(f"{len(errors)} manifest entries failed; grids root at {a.out} is incomplete")
    print(f"linked {n} grid files under {a.out} ({'sha256 verified' if a.verify else 'sizes checked; --verify for sha256'})")


if __name__ == "__main__":
    main()
