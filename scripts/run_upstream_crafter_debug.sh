#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${1:-$ROOT/experiments/runs/upstream_crafter_debug_$STAMP}"

mkdir -p "$LOGDIR"
cd "$ROOT/third_party/dreamerv3"

# We want to run upstream exactly, but still keep logs inside *this* repo.
PYTHONPATH=. \
python dreamerv3/main.py \
  --logdir "$LOGDIR" \
  --configs crafter debug \
  --run.steps 2000
