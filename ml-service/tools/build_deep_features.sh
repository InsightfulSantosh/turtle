#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$(cd "$SERVICE_ROOT/.." && pwd)"
PROJECT_ROOT="$(cd "$APP_ROOT/.." && pwd)"
BUILD_ROOT="${TMPDIR:-/tmp}/turtle-season-intelligence"
mkdir -p "$BUILD_ROOT"

clang \
  -fobjc-arc \
  -fblocks \
  -framework Foundation \
  -framework Vision \
  -framework ImageIO \
  -framework CoreGraphics \
  "$SERVICE_ROOT/tools/deep_vision.m" \
  -o "$BUILD_ROOT/deep_vision"

"$BUILD_ROOT/deep_vision" \
  "$PROJECT_ROOT/tmp/vision-images-map.json" \
  "$BUILD_ROOT/deep-distances.json"

"${PYTHON:-python3}" "$SERVICE_ROOT/train_and_export.py" \
  --vision "$BUILD_ROOT/deep-distances.json" \
  --output "$APP_ROOT/app/generated-data.json"
