#!/usr/bin/env bash
# Reproducible paper runner. Use "mac" locally and "gpu" on the RTX 4090 host.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
OUT_ROOT="${OUT_ROOT:-logs/memory_probes/publication}"
FIG_DIR="${FIG_DIR:-memory_probes/paper/figures}"

run_cpu_profile() {
  local profile="$1"
  local output="$OUT_ROOT/${profile}_results.json"
  uv run python -m memory_probes.publication --profile "$profile" --output "$output"
  uv run --extra plot python scripts/plot_memory_probes_paper.py \
    --input "$output" --output-dir "$FIG_DIR"
  uv run python -m unittest memory_probes.test_publication
}

case "$MODE" in
  smoke)
    run_cpu_profile smoke
    ;;
  mac)
    run_cpu_profile mac
    ;;
  gpu)
    : "${CHECKPOINT:?Set CHECKPOINT to the trained V11 .pt checkpoint}"
    PRESET="${PRESET:-v11_e3_k3_chat}"
    TEXT_TOKENS="${TEXT_TOKENS:-50000}"
    mkdir -p "$OUT_ROOT/gpu"

    uv run python scripts/v11_probe_gates.py \
      --checkpoint "$CHECKPOINT" \
      --preset "$PRESET" \
      --tokens "${GATE_TOKENS:-4096}" \
      --output "$OUT_ROOT/gpu/gates.json"

    for layer in ${RANK_LAYERS:-0 4 8 12 15}; do
      uv run python -m memory_probes \
        --test rank-text \
        --checkpoint "$CHECKPOINT" \
        --preset "$PRESET" \
        --layer "$layer" \
        --text-tokens "$TEXT_TOKENS" \
        --sample-every "${SAMPLE_EVERY:-100}" \
        --output-dir "$OUT_ROOT/gpu"
    done

    uv run python -m memory_probes \
      --test language-filler \
      --filler-tokens "${FILLER_TOKENS:-5000}" \
      --projection-trials "${PROJECTION_TRIALS:-50}" \
      --output-dir "$OUT_ROOT/gpu"

    uv run python -m memory_probes \
      --test arch-compare \
      --text-tokens "${ARCH_COMPARE_TOKENS:-2000}" \
      --arch-dim "${ARCH_DIM:-64}" \
      --compare-layer "${MAMBA_LAYER:-12}" \
      --output-dir "$OUT_ROOT/gpu"

    BEHAVIOR_ARGS=(
      --context-lengths "${BEHAVIOR_CONTEXTS:-128,512,1024,2048}"
      --positions "${BEHAVIOR_POSITIONS:-0,0.5,1}"
      --association-counts "${BEHAVIOR_ASSOCIATIONS:-1,4,8}"
      --trials "${BEHAVIOR_TRIALS:-20}"
      --candidate-count "${BEHAVIOR_CANDIDATES:-8}"
    )
    uv run python scripts/run_memory_behavioral.py \
      --model-type v11 \
      --checkpoint "$CHECKPOINT" \
      --preset "$PRESET" \
      --output "$OUT_ROOT/gpu/v11_behavior.json" \
      "${BEHAVIOR_ARGS[@]}"

    if [[ -n "${TRANSFORMER_CHECKPOINT:-}" ]]; then
      uv run python scripts/run_memory_behavioral.py \
        --model-type transformer \
        --checkpoint "$TRANSFORMER_CHECKPOINT" \
        --output "$OUT_ROOT/gpu/transformer_behavior.json" \
        "${BEHAVIOR_ARGS[@]}"
    else
      echo "Skipping trained Transformer behavior: set TRANSFORMER_CHECKPOINT to enable."
    fi

    if [[ "${SKIP_MAMBA_BEHAVIOR:-0}" != "1" ]]; then
      uv run python scripts/run_memory_behavioral.py \
        --model-type hf \
        --model-id "${MAMBA_MODEL:-state-spaces/mamba-130m-hf}" \
        --output "$OUT_ROOT/gpu/mamba_behavior.json" \
        "${BEHAVIOR_ARGS[@]}"
    fi

    echo "GPU diagnostics complete: $OUT_ROOT/gpu"
    echo "See memory_probes/PUBLICATION.md before using cross-model results in the paper."
    ;;
  *)
    echo "Usage: $0 {smoke|mac|gpu}" >&2
    exit 2
    ;;
esac
