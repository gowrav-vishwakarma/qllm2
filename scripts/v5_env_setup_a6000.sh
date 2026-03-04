#!/usr/bin/env bash
# Shared environment bootstrap for V5 scripts on A6000.
# Ensures:
# - PATH includes ~/.local/bin (for uv if installed there)
# - a Python environment is available
# - required packages for v5.train are installed
#
# Exports:
#   PYTHON_BIN  -> python executable to run v5.train

set -e

export PATH="$HOME/.local/bin:$PATH"

# Prefer project venv if present; otherwise fall back to uv/python3.
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="python"
elif command -v uv >/dev/null 2>&1; then
  # We'll use uv run python later as PYTHON_BIN wrapper
  PYTHON_BIN="uv run python"
else
  PYTHON_BIN="python3"
fi

echo "[v5-setup] Using PYTHON_BIN: $PYTHON_BIN"

# Ensure dependencies exist in the chosen runtime.
if ! eval "$PYTHON_BIN -c 'import torch, transformers, datasets' >/dev/null 2>&1"; then
  echo "[v5-setup] Installing missing deps (torch/transformers/datasets)..."
  if [[ "$PYTHON_BIN" == "python" || "$PYTHON_BIN" == "python3" ]]; then
    # Some venvs are created without pip; recover automatically.
    if ! eval "$PYTHON_BIN -m pip --version >/dev/null 2>&1"; then
      eval "$PYTHON_BIN -m ensurepip --upgrade >/dev/null 2>&1 || true"
    fi

    if eval "$PYTHON_BIN -m pip --version >/dev/null 2>&1"; then
      eval "$PYTHON_BIN -m pip install -U transformers datasets"
    elif command -v uv >/dev/null 2>&1; then
      uv pip install transformers datasets
    else
      echo "[v5-setup] ERROR: pip unavailable and uv not found."
      exit 1
    fi
  else
    uv pip install transformers datasets
  fi
fi

export PYTHON_BIN

