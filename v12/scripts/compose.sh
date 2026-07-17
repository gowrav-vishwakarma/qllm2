#!/usr/bin/env bash
# Compose (pack) an inference checkpoint from registry modules.
#
# The resolver walks each module's `requires` graph, version-solves, orders the
# stack (dependencies first), verifies geometry + substrate_hash, then pack
# assembles ONE checkpoint (base shared params + renumbered group blocks, each
# group's attach_mode stamped per layer).
#
# Demonstrates:
#   1. Default target resolve+pack (reasoning -> grammar + fact_retrieval + reasoning).
#   2. Arbitrary-order (facts-first) assembly via an explicit id@spec stack.
#   3. prelayer vs finetuned: prelayer deps are stacked+hash-verified; a finetuned
#      dependency is standalone (recorded as lineage, NOT stacked).
#
# Usage:
#   v12/scripts/compose.sh                       # pack the default target (reasoning)
#   v12/scripts/compose.sh reasoning             # pack a specific target module
#   v12/scripts/compose.sh --stack "grammar@1,fact_retrieval@>=1"   # explicit order
#   TARGET=reasoning OUT=packed/model.pt v12/scripts/compose.sh
#
# Env: PY, REGISTRY, OUT, TARGET, CONSTRAINT.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

PY="${PY:-.venv/bin/python}"
REGISTRY="${REGISTRY:-v12_registry}"
OUT="${OUT:-packed_v12/model.pt}"
CONSTRAINT="${CONSTRAINT:-*}"
mkdir -p "$(dirname "$OUT")"

if [ "${1:-}" = "--stack" ]; then
  STACK="$2"
  echo "== resolve explicit stack: $STACK =="
  run_stack=("$PY" -m v12.registry --registry "$REGISTRY" stack "$STACK")
  echo "+ ${run_stack[*]}"; "${run_stack[@]}"
  echo
  echo "note: to pack an arbitrary-order stack that is not itself a registered"
  echo "module, publish a thin group whose --requires lists them in that order,"
  echo "then: $PY -m v12.pack --target <that_module> --registry $REGISTRY --out $OUT"
  exit 0
fi

TARGET="${TARGET:-${1:-reasoning}}"

echo "== resolve $TARGET =="
"$PY" -m v12.registry --registry "$REGISTRY" resolve "$TARGET" --constraint "$CONSTRAINT"
echo
echo "== pack $TARGET -> $OUT =="
"$PY" -m v12.pack --target "$TARGET" --constraint "$CONSTRAINT" \
  --registry "$REGISTRY" --out "$OUT"
echo
echo "Packed checkpoint ready: $OUT"
echo "  (prelayer deps were stacked + substrate_hash verified; any finetuned"
echo "   deps are recorded as lineage only and are NOT stacked.)"
