"""The public-content guard: names that must not appear in anything this repository publishes.

FORBIDDEN matches internal hostnames and machine paths, session ids, internal run and checkpoint names, and the
internal id of the human row. The vendor Moonshot AI (and its model ids, moonshotai/…) is allowed. This file is the
one place the forbidden names are spelled out, so the tracked-file scan skips it.

Usage: python build/forbidden.py [FILE ...]    (default: every file git tracks; exit 1 on any hit)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN = re.compile(
    r"node[12]\b|datasci|\.ath\b|/models/|/home/|/mnt/|/tmp/|athuser|luxi-files|beegfs|heimdall|scratchpad"
    r"|kotodama-lab|_local-artifacts|luxia/|\bluxia\|"
    r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    r"|moonshot(?![\s_-]*ai)|a110B|oldcode|\brdlg|\bt32\b|\bA0\*|\bs43\b",
    re.I)
SELF = Path(__file__).resolve()


def hits(text: str) -> list[str]:
    return sorted({m.group(0) for m in FORBIDDEN.finditer(text)})


def main() -> None:
    repo = SELF.parent.parent
    files = ([Path(f) for f in sys.argv[1:]] if len(sys.argv) > 1 else
             [repo / f for f in subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True,
                                               check=True).stdout.splitlines()])
    bad = 0
    for f in files:
        if f.resolve() == SELF or not f.is_file():
            continue
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue                      # binary (images, fonts)
        for ln, line in enumerate(text.splitlines(), 1):
            h = hits(line)
            if h:
                bad += 1
                print(f"{f.relative_to(repo) if f.is_relative_to(repo) else f}:{ln}: {h}")
    if bad:
        raise SystemExit(f"guard: {bad} line(s) with forbidden names")
    print(f"guard ok: {len(files)} files, no forbidden names")


if __name__ == "__main__":
    main()
