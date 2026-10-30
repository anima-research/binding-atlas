"""Make an invite for /take: prints the link, appends the token's sha256 to the invites file (tokens are not stored).

    python take/make_invite.py --invites invites.txt --base https://example.org/take [--label friends]
"""
from __future__ import annotations

import argparse
import hashlib
import secrets
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--invites", type=Path, required=True)
    ap.add_argument("--base", required=True, help="the page's URL, e.g. https://example.org/take")
    ap.add_argument("--label", default="", help="a note kept beside the hash (who the link went to, in words)")
    a = ap.parse_args()
    token = secrets.token_urlsafe(12)
    with a.invites.open("a") as f:
        if a.label:
            f.write(f"# {a.label}\n")
        f.write(hashlib.sha256(token.encode()).hexdigest() + "\n")
    print(f"{a.base}?invite={token}")


if __name__ == "__main__":
    main()
