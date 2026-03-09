#!/usr/bin/env bash
# Compare all 3 V6 ablation checkpoints with one prompt.
#
# Usage:
#   bash scripts/v6_compare.sh
#   bash scripts/v6_compare.sh --prompt "The dragon flew" --max_tokens 200 --temperature 0.7
#
# All flags are optional and passed straight to v6.generate.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh

# ---- defaults (override via flags) ----
PROMPT="Once upon a time, there was a little"
MAX_TOKENS=150
TEMPERATURE=0.8
TOP_K=50
TOP_P=0.9
REP_PENALTY=1.2

# ---- parse named args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompt)         PROMPT="$2";       shift 2 ;;
    --max_tokens)     MAX_TOKENS="$2";   shift 2 ;;
    --temperature)    TEMPERATURE="$2";  shift 2 ;;
    --top_k)          TOP_K="$2";        shift 2 ;;
    --top_p)          TOP_P="$2";        shift 2 ;;
    --rep_penalty)    REP_PENALTY="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

COMMON_ARGS="--prompt \"$PROMPT\" --max_tokens $MAX_TOKENS --temperature $TEMPERATURE --top_k $TOP_K --top_p $TOP_P --repetition_penalty $REP_PENALTY"

RUNS=(
  "Run 1 · no memory      |checkpoints/v6/fulldata_no_memory/best_model.pt"
  "Run 2 · tiny memory    |checkpoints/v6/fulldata_tiny_memory/best_model.pt"
  "Run 3 · tiny model     |checkpoints/v6/fulldata_tiny_model/best_model.pt"
)

echo ""
echo "============================================================"
echo "  V6 Ablation Comparison"
echo "  Prompt      : $PROMPT"
echo "  Max tokens  : $MAX_TOKENS"
echo "  Temperature : $TEMPERATURE  top_k: $TOP_K  top_p: $TOP_P  rep_penalty: $REP_PENALTY"
echo "============================================================"

for entry in "${RUNS[@]}"; do
  label="${entry%%|*}"
  ckpt="${entry##*|}"

  echo ""
  echo "------------------------------------------------------------"
  echo "  $label"
  echo "  $ckpt"
  echo "------------------------------------------------------------"

  if [[ ! -f "$ckpt" ]]; then
    echo "  [SKIP] checkpoint not found: $ckpt"
    continue
  fi

  eval "$PYTHON_BIN -m v6.generate --checkpoint \"$ckpt\" $COMMON_ARGS"
done

echo ""
echo "============================================================"
echo "Done."
echo "============================================================"
