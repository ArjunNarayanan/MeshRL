#!/usr/bin/env bash
# Assemble the single-file builds of Mesh Quest from game/src + game/js/engine.js.
#
#   ./game/build.sh [artifact-output-path]
#
# Outputs:
#   game/mesh-quest.html  standalone page (open locally, email, self-host)
#   docs/index.html       the same page plus analytics, served by GitHub Pages
#   $1 (optional)         artifact variant: no document skeleton, no analytics
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/game/src"
engine="$root/game/js/engine.js"

# Analytics live only on the public Pages build: the artifact runtime blocks
# external requests, and the local standalone file should not phone home.
analytics='<script data-goatcounter="https://arjunnarayanan.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>'

emit_head() {
  echo '<title>Mesh Quest</title>'
  echo '<style>'
  cat "$src/style.css"
  echo '</style>'
}

emit_body() {
  cat "$src/body.html"
  echo '<script>'
  cat "$engine"
  cat "$src/ui.js"
  echo '</script>'
}

standalone() {  # $1 = extra head markup
  echo '<!doctype html>'
  echo '<html lang="en">'
  echo '<head>'
  echo '<meta charset="utf-8">'
  echo '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
  emit_head
  [ -n "$1" ] && echo "$1"
  echo '</head>'
  echo '<body>'
  emit_body
  echo '</body>'
  echo '</html>'
}

standalone "" > "$root/game/mesh-quest.html"
mkdir -p "$root/docs"
standalone "$analytics" > "$root/docs/index.html"

if [ $# -ge 1 ]; then
  { emit_head; emit_body; } > "$1"
  echo "built: game/mesh-quest.html, docs/index.html, $1"
else
  echo "built: game/mesh-quest.html, docs/index.html"
fi
