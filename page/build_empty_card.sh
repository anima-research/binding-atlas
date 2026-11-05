#!/bin/bash
# Build the An Empty Document link card: serve the built site over local HTTP (the page fetches its runs), render the top
# of the page in headless Chrome at 1400 wide, crop to 1400×735 and scale to 1200×630 (the title, the four figures and the headline rows: page #card mode).
# Usage: build_empty_card.sh SITE_DIR DEST.png      Env: PY (python with Pillow), CHROME (default google-chrome), PORT,
#   FONT_DIR (the web fonts, served as fonts/ when SITE_DIR has none, so the card renders in the page's faces)
set -euo pipefail
SITE=$(cd "$1" && pwd); DEST=$2; PY=${PY:-python3}; CHROME=${CHROME:-google-chrome}; PORT=${PORT:-8772}
W=$(mktemp -d)
SERVE=$SITE
if [ ! -d "$SITE/fonts" ] && [ -n "${FONT_DIR:-}" ] && compgen -G "$FONT_DIR/*.woff2" > /dev/null; then
  SERVE=$W/site; mkdir "$SERVE"
  ln -s "$SITE/empty-document.html" "$SITE/empty-document-data" "$SERVE/"
  ln -s "$(cd "$FONT_DIR" && pwd)" "$SERVE/fonts"
fi
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$SERVE" > "$W/http.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
timeout 120 "$CHROME" --headless=new --hide-scrollbars \
  --virtual-time-budget=15000 --window-size=1400,900 --screenshot="$W/raw.png" "http://127.0.0.1:$PORT/empty-document.html#card" 2>/dev/null
"$PY" -c "import sys; from PIL import Image; Image.open(sys.argv[1]).crop((0,0,1400,735)).resize((1200,630), Image.LANCZOS).save(sys.argv[2], optimize=True)" "$W/raw.png" "$DEST"
echo "card: $DEST"
