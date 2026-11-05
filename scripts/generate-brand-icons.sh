#!/usr/bin/env bash
# Regenerate raster browser icons from the shared header/favicon SVG.
# Requires ImageMagick 7 (magick) and its SVG delegate (rsvg-convert).
set -euo pipefail
cd "$(dirname "$0")/../web/public"
magick -background none -density 384 favicon.svg -strip -define icon:auto-resize=48,32,16 favicon.ico
magick -background none -density 384 favicon.svg -resize 180x180 -background white -alpha remove -alpha off -strip apple-touch-icon.png
