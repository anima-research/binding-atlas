#!/bin/bash
# Build the Neuron Atlas link card: serve the built site over local HTTP (the page fetches its data), render the hero in
# headless Chrome with software WebGL, crop to 1200×630.
# Usage: build_neuron_card.sh SITE_DIR DEST.png      Env: PY (python with Pillow), CHROME (default google-chrome), PORT
set -euo pipefail
SITE=$(cd "$1" && pwd); DEST=$2; PY=${PY:-python3}; CHROME=${CHROME:-google-chrome}; PORT=${PORT:-8771}
W=$(mktemp -d)
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$SITE" > "$W/http.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true; rm -rf "$W"' EXIT
sleep 1
timeout 120 "$CHROME" --headless=new --use-angle=swiftshader --enable-unsafe-swiftshader --hide-scrollbars \
  --virtual-time-budget=15000 --window-size=1200,800 --screenshot="$W/raw.png" "http://127.0.0.1:$PORT/neuron-atlas.html" 2>/dev/null
"$PY" -c "import sys; from PIL import Image; Image.open(sys.argv[1]).crop((0,0,1200,630)).save(sys.argv[2], optimize=True)" "$W/raw.png" "$DEST"
echo "card: $DEST"
