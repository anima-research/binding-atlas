"""Copy the /take database safely while the service runs (SQLite's online backup), keeping the last N copies.

    python take/backup.py --db take.sqlite --dir backups/ [--keep 14]
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--keep", type=int, default=14)
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    dest = a.dir / f"take-{time.strftime('%Y%m%d-%H%M%S')}.sqlite"
    src, dst = sqlite3.connect(a.db), sqlite3.connect(dest)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()
    for old in sorted(a.dir.glob("take-*.sqlite"))[:-a.keep]:
        old.unlink()
    print(dest)


if __name__ == "__main__":
    main()
