#!/usr/bin/env bash
# Acquire the official PrOntoQA generator, pinned to a commit hash (plan deliverable 4).
# Colab has network access at runtime; this runs from the notebook's setup cell.
#
# Usage:  bash scripts/fetch_prontoqa.sh [TARGET_DIR] [COMMIT]
# Defaults: TARGET_DIR=./prontoqa  COMMIT=<PRONTOQA_COMMIT from config.py>
#
# The clone is pinned so the generated dataset is reproducible. UPDATE the commit before
# a real run (config.PRONTOQA_COMMIT is a placeholder).
set -euo pipefail

REPO="https://github.com/asaparov/prontoqa"
TARGET_DIR="${1:-prontoqa}"
COMMIT="${2:-PLACEHOLDER_PIN_ME}"

if [ "$COMMIT" = "PLACEHOLDER_PIN_ME" ]; then
  echo "ERROR: PrOntoQA commit not pinned. Pass the hash as arg 2 or set it in config.py." >&2
  echo "       e.g.  bash scripts/fetch_prontoqa.sh prontoqa 1a2b3c4d" >&2
  exit 2
fi

if [ -d "$TARGET_DIR/.git" ]; then
  echo "[fetch_prontoqa] $TARGET_DIR already present; fetching + checking out $COMMIT"
  git -C "$TARGET_DIR" fetch --quiet origin
else
  echo "[fetch_prontoqa] cloning $REPO -> $TARGET_DIR"
  git clone --quiet "$REPO" "$TARGET_DIR"
fi

git -C "$TARGET_DIR" checkout --quiet "$COMMIT"
echo "[fetch_prontoqa] checked out $(git -C "$TARGET_DIR" rev-parse --short HEAD)"
echo "[fetch_prontoqa] contents:"
ls -1 "$TARGET_DIR" | sed 's/^/  /'
