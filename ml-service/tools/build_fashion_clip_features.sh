#!/usr/bin/env bash
set -euo pipefail

SERVICE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_ROOT="$(cd "$SERVICE_ROOT/.." && pwd)"
BUILD_ROOT="${TMPDIR:-/tmp}/turtle-fashion-clip"
mkdir -p "$BUILD_ROOT"

"${PYTHON:-python3}" "$SERVICE_ROOT/tools/fashion_clip_embeddings.py" \
  --source "$APP_ROOT/app/generated-data.json" \
  --output "$BUILD_ROOT/fashion-clip-distances.json" \
  "$@"

"${PYTHON:-python3}" "$SERVICE_ROOT/train_and_export.py" \
  --vision "$BUILD_ROOT/fashion-clip-distances.json" \
  --output "$APP_ROOT/app/generated-data.json"
