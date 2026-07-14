#!/usr/bin/env bash
# Recall-program evaluation gate for one trained V11 checkpoint.
#
# Runs the gate-selectivity probe + the contrastive behavioral recall suite,
# then prints a ship/no-ship verdict against the recall-program criteria.
# GPU required; run OUTSIDE the sandbox (needs /dev/nvidia*).
#
# Usage:
#   ./scripts/eval_recall_gate.sh CHECKPOINT [OUT_DIR] [PRESET]
#
# Env overrides:
#   PRESET (default v11_e3_k3_chat)   GATE_TOKENS (4096)
#   BEHAVIOR_CONTEXTS (128,512,1024,2048)  BEHAVIOR_POSITIONS (0,0.5,1)
#   BEHAVIOR_ASSOCIATIONS (1,4,8)     BEHAVIOR_TRIALS (20)
#   ACC_TARGET (0.9)  GATE_TARGET (0.05)
#   ARM_PPL, BASELINE_PPL (optional; enable the PPL-within-tol check)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

CHECKPOINT="${1:?Usage: eval_recall_gate.sh CHECKPOINT [OUT_DIR] [PRESET]}"
OUT_DIR="${2:-logs/memory_probes/recall_ab/$(basename "$(dirname "$CHECKPOINT")")}"
PRESET="${3:-${PRESET:-v11_e3_k3_chat}}"
LABEL="${LABEL:-$(basename "$(dirname "$CHECKPOINT")")}"
mkdir -p "$OUT_DIR"

echo "[eval-recall-gate] checkpoint=$CHECKPOINT preset=$PRESET out=$OUT_DIR"

echo "[1/3] gate diagnostics ..."
uv run python scripts/v11_probe_gates.py \
  --checkpoint "$CHECKPOINT" \
  --preset "$PRESET" \
  --tokens "${GATE_TOKENS:-4096}" \
  --output "$OUT_DIR/gates.json"

echo "[2/3] behavioral recall ..."
uv run python scripts/run_memory_behavioral.py \
  --model-type v11 \
  --checkpoint "$CHECKPOINT" \
  --preset "$PRESET" \
  --context-lengths "${BEHAVIOR_CONTEXTS:-128,512,1024,2048}" \
  --positions "${BEHAVIOR_POSITIONS:-0,0.5,1}" \
  --association-counts "${BEHAVIOR_ASSOCIATIONS:-1,4,8}" \
  --trials "${BEHAVIOR_TRIALS:-20}" \
  --output "$OUT_DIR/v11_behavior.json"

echo "[3/3] verdict ..."
PPL_ARGS=()
[[ -n "${ARM_PPL:-}" ]] && PPL_ARGS+=(--ppl "$ARM_PPL")
[[ -n "${BASELINE_PPL:-}" ]] && PPL_ARGS+=(--baseline-ppl "$BASELINE_PPL")
uv run python scripts/recall_gate_verdict.py \
  --gates "$OUT_DIR/gates.json" \
  --behavior "$OUT_DIR/v11_behavior.json" \
  --out "$OUT_DIR/verdict.json" \
  --label "$LABEL" \
  --acc-target "${ACC_TARGET:-0.9}" \
  --gate-target "${GATE_TARGET:-0.05}" \
  "${PPL_ARGS[@]}"

echo "[eval-recall-gate] done -> $OUT_DIR"
