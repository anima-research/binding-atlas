#!/bin/bash
# Build the Binding Atlas link card: build_card.py → HTML → headless Chrome (rendered taller, then cropped to exactly
# 1200×630 — headless Chrome's viewport is shorter than --window-size).
# Usage: build_card.sh WORKDIR DEST.png [build_card.py args...]
# Env: PY (python with Pillow; default python3), CHROME (default google-chrome), FONT_DIR (local copies of the
# configured web font's woff2 files, copied into WORKDIR/fonts; default site/fonts; absent = fallback faces).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=$(mkdir -p "$1" && cd "$1" && pwd); DEST=$2; shift 2
PY=${PY:-python3}; CHROME=${CHROME:-google-chrome}; FONT_DIR=${FONT_DIR:-$HERE/../site/fonts}
rm -rf "$W/fonts" && mkdir -p "$W/fonts"
compgen -G "$FONT_DIR/*.woff2" > /dev/null && cp "$FONT_DIR"/*.woff2 "$W/fonts/"
"$PY" "$HERE/build_card.py" "$W/card.html" "$@"
timeout 90 "$CHROME" --headless=new --disable-gpu --hide-scrollbars --virtual-time-budget=6000 --window-size=1200,800 \
  --screenshot="$W/card-raw.png" "file://$W/card.html" 2>/dev/null
"$PY" -c "import sys; from PIL import Image; Image.open(sys.argv[1]).crop((0,0,1200,630)).save(sys.argv[2], optimize=True)" \
  "$W/card-raw.png" "$DEST"
echo "card: $DEST"
