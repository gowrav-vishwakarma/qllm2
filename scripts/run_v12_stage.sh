#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RETIRED: superseded by the playable module-system scripts in v12/scripts/.
#   - v12/scripts/train_curriculum.sh  (grammar -> fact_retrieval -> reasoning,
#     dynamic heads, per-stage loss, compact + publish to the registry)
#   - v12/scripts/compose.sh, v12/scripts/eval.sh, v12/scripts/registry.sh
# This script is kept only for reference to the raw depth-growth CLI flags.
# ─────────────────────────────────────────────────────────────────────────────
#
# V12 progressive DEPTH-growth curriculum (M4).
#
# Idea: a small grammar BASE (few layers, little data) is trained first, then
# specialist LAYER GROUPS are grown on top in a swappable order. Each specialist
# stage:
#   - resumes the previous stage's checkpoint (all prior layers included),
#   - grows N new layers for the skill (--grow_layers "skill:N[:head_budget]"),
#   - freezes everything below the grown group (--freeze_layers base),
#   - trains only the new layers on that skill's data with a chosen --stage_loss.
#
# Frozen blocks use requires_grad=False, so they are excluded from the optimizer
# param groups entirely (no weight-decay drift; unlike head-slot freezing which
# needs weight_decay=0). The grown stack + provenance (skill/group_id/stage/
# substrate_hash/attach_mode) is saved in the checkpoint config (layer_specs) and
# rebuilt automatically by eval/generate.
#
# SWAPPABLE ORDER: edit the ORDER array below to ablate acquisition order (e.g.
# facts first, grammar last). Each stage resumes from its predecessor in ORDER.
#
# Batch/seq below target a 24GB RTX-4090 (local). On the 96GB server raise
# BATCH/SEQ (e.g. BATCH=16 SEQ=2048).
#
# Usage:
#   ./scripts/run_v12_stage.sh base                 # train grammar base (scratch)
#   ./scripts/run_v12_stage.sh facts                # grow facts on top of base
#   ./scripts/run_v12_stage.sh reasoning            # grow reasoning on top of facts
#   ./scripts/run_v12_stage.sh code_c               # grow C on top of code
#   ./scripts/run_v12_stage.sh facts --dry          # print the command only
#
# Per-stage overrides (env): GROW, HEADS, DATASET, STAGE, STAGE_LOSS, LR,
#   TOKEN_BUDGET, ATTACH_MODE, RESUME (explicit predecessor checkpoint).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

STAGE="${1:-help}"; shift || true
DRY=0; for a in "$@"; do case "$a" in --dry|-n) DRY=1 ;; esac; done

PY="${PY:-.venv/bin/python}"
BASE_PRESET="${BASE_PRESET:-v12_base_grammar}"
BATCH="${BATCH:-2}"          # RTX-4090 safe default
SEQ="${SEQ:-1024}"           # raise to 2048 on the server
CKPT_ROOT="${CKPT_ROOT:-checkpoints_v12_grow}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Skill acquisition order (edit to ablate). "base" must stay first.
ORDER=("base" "facts" "reasoning" "math" "bio" "code" "code_c" "code_java")

run() {
  echo "+ $*"
  [ "$DRY" = "1" ] && return 0
  "$@"
}

predecessor_of() {  # echo the ORDER entry before $1 (empty for base/unknown)
  local target="$1" prev=""
  for s in "${ORDER[@]}"; do
    [ "$s" = "$target" ] && { echo "$prev"; return 0; }
    prev="$s"
  done
  echo ""
}

if [ "$STAGE" = "base" ]; then
  # Small grammar base from scratch (few layers, grammar-heavy blend warmup).
  run "$PY" -m v12.train \
    --preset "$BASE_PRESET" --stage pretrain --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb --pretrain_weights 70,30 \
    --blend_warmup_tokens 300000000 \
    --stage_loss "${STAGE_LOSS:-ce}" \
    --batch_size "$BATCH" --seq_len "$SEQ" --lr "${LR:-1e-4}" --weight_decay 0.01 \
    --token_budget "${TOKEN_BUDGET:-1000000000}" \
    --checkpoint_dir "$CKPT_ROOT/base" --save_every_steps 5000
  exit 0
fi

# ── Specialist stages: grow a layer group on top of the frozen predecessor ────
PRED="$(predecessor_of "$STAGE")"
RESUME="${RESUME:-$CKPT_ROOT/$PRED/best_model.pt}"
ATTACH_MODE="${ATTACH_MODE:-sequential}"

case "$STAGE" in
  facts)
    GROW="${GROW:-facts:4}"; DATASET="${DATASET:-pretrain_mix}"
    SRC="dclm,fineweb"; WEIGHTS="40,60"; DEFAULT_LOSS="ce_recall"; DEFAULT_LR="1e-4"
    ;;
  reasoning)
    GROW="${GROW:-reasoning:4}"; DATASET="${DATASET:-pretrain_mix}"
    SRC="fineweb,smoltalk2_mid"; WEIGHTS="50,50"; DEFAULT_LOSS="ce"; DEFAULT_LR="1e-4"
    ;;
  math|bio|code|code_c|code_java)
    # No stock corpus for these — supply your specialist dataset via DATASET/SRC.
    GROW="${GROW:-$STAGE:3}"; DATASET="${DATASET:-pretrain_mix}"
    SRC="${SRC:-fineweb}"; WEIGHTS="${WEIGHTS:-100}"; DEFAULT_LOSS="ce"; DEFAULT_LR="1e-4"
    ;;
  *)
    echo "Usage: $0 {base|facts|reasoning|math|bio|code|code_c|code_java} [--dry]"
    echo "  base is trained from scratch; every other stage grows on its ORDER predecessor."
    echo "  ORDER: ${ORDER[*]}"
    exit 1
    ;;
esac

if [ ! -f "$RESUME" ] && [ "$DRY" != "1" ]; then
  echo "ERROR: predecessor checkpoint not found: $RESUME"
  echo "  (train '$PRED' first, or pass RESUME=/path/to/best_model.pt)"
  exit 1
fi

# Optional per-layer head budget (omit the flag entirely when HEADS is unset).
HEAD_ARG=()
[ -n "${HEADS:-}" ] && HEAD_ARG=(--layer_head_budget "$HEADS")

# smoltalk2 => SFT stage + chat; pretrain_mix / other => pretrain stage.
if [ "$DATASET" = "smoltalk2" ]; then
  run "$PY" -m v12.train \
    --preset "$BASE_PRESET" --stage sft --dataset smoltalk2 \
    --resume_from "$RESUME" \
    --grow_layers "$GROW" "${HEAD_ARG[@]}" \
    --freeze_layers base --attach_mode "$ATTACH_MODE" \
    --stage_loss "${STAGE_LOSS:-$DEFAULT_LOSS}" \
    --batch_size "$BATCH" --seq_len "$SEQ" --lr "${LR:-5e-5}" --weight_decay 0.01 \
    --epochs "${EPOCHS:-1}" --checkpoint_dir "$CKPT_ROOT/$STAGE"
else
  run "$PY" -m v12.train \
    --preset "$BASE_PRESET" --stage pretrain --dataset "$DATASET" \
    --pretrain_sources "$SRC" --pretrain_weights "$WEIGHTS" \
    --resume_from "$RESUME" \
    --grow_layers "$GROW" "${HEAD_ARG[@]}" \
    --freeze_layers base --attach_mode "$ATTACH_MODE" \
    --stage_loss "${STAGE_LOSS:-$DEFAULT_LOSS}" \
    --batch_size "$BATCH" --seq_len "$SEQ" --lr "${LR:-$DEFAULT_LR}" --weight_decay 0.01 \
    --token_budget "${TOKEN_BUDGET:-1000000000}" \
    --checkpoint_dir "$CKPT_ROOT/$STAGE" --save_every_steps 5000
fi
