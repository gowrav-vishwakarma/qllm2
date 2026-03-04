#!/usr/bin/env bash
# Run init strategy benchmark: tests all (or selected) init strategies on limited data.
#
# Report written to: logs/init_bench_YYYYMMDD_HHMMSS.log (or --log_file)
#
# Usage:
#   ./scripts/run_bench_init.sh                          # default: 500 samples, 2 epochs, tiny
#   ./scripts/run_bench_init.sh --samples 1000 --epochs 3
#   ./scripts/run_bench_init.sh --size small --strategies golden_ratio,dft,random
#   ./scripts/run_bench_init.sh --quiet --log_file logs/my_bench.log
#
# Uses uv run when available (project deps from pyproject.toml, no re-download).
# Set USE_V5_SETUP=1 to force v5_env_setup_a6000.sh (e.g. on A6000 server).

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v5/train.py ]] || cd ..

if [[ "$USE_V5_SETUP" == "1" ]] && [[ -f ./scripts/v5_env_setup_a6000.sh ]]; then
  # shellcheck disable=SC1091
  source ./scripts/v5_env_setup_a6000.sh
  PYTHON_CMD="$PYTHON_BIN"
elif command -v uv >/dev/null 2>&1 && [[ -f pyproject.toml ]]; then
  PYTHON_CMD="uv run python"
else
  PYTHON_CMD="python"
fi

eval "$PYTHON_CMD scripts/bench_init_strategies.py" "$@"
