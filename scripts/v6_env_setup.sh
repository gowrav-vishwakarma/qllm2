#!/usr/bin/env bash
# Shared environment bootstrap for V6 scripts.
# Ensures:
# - PATH includes ~/.local/bin (for uv if installed there)
# - a Python environment is available
# - required packages for v6.train are installed
#
# Exports:
#   PYTHON_BIN  -> python executable to run v6.train

set -e

export PATH="$HOME/.local/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
  PYTHON_BIN="uv run python"
else
  PYTHON_BIN="python3"
fi

echo "[v6-setup] Using PYTHON_BIN: $PYTHON_BIN"

if ! eval "$PYTHON_BIN -c 'import torch, transformers, datasets' >/dev/null 2>&1"; then
  echo "[v6-setup] Installing missing deps (torch/transformers/datasets)..."
  if [[ "$PYTHON_BIN" == "python" || "$PYTHON_BIN" == "python3" ]]; then
    if ! eval "$PYTHON_BIN -m pip --version >/dev/null 2>&1"; then
      eval "$PYTHON_BIN -m ensurepip --upgrade >/dev/null 2>&1 || true"
    fi

    if eval "$PYTHON_BIN -m pip --version >/dev/null 2>&1"; then
      eval "$PYTHON_BIN -m pip install -U transformers datasets"
    elif command -v uv >/dev/null 2>&1; then
      uv pip install transformers datasets
    else
      echo "[v6-setup] ERROR: pip unavailable and uv not found."
      exit 1
    fi
  else
    uv pip install transformers datasets
  fi
fi

export PYTHON_BIN
