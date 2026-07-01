#!/usr/bin/env bash
# Saturate the 100M PAM: continued knowledge pretrain from the best checkpoint
# with FRESH web-edu shards (skip already-seen docs => no token reuse), cosine
# warm-restart, in-distro holdout PPL as the primary stop signal.
#
# Phase-aware GSP gate is the default (WikiText A/B 2026-06-30). For ablation:
#   GATE=0 ./scripts/run_v11_saturate.sh   # magnitude-only gate (--no_gate_content_aware)
#
# Usage (persistent):
#   tmux new-session -d -s v11_sat './scripts/run_v11_saturate.sh'                 # 5B saturate round
#   GATE=0 tmux new-session -d -s v11_sat './scripts/run_v11_saturate.sh 500000000'  # baseline ablation
#
#   # resume an interrupted round (full state from latest.pt):
#   RESUME=checkpoints_v11_sat/latest.pt tmux new-session -d -s v11_sat './scripts/run_v11_saturate.sh'
#
# IMPORTANT (token reuse):
#   * For the A/B *smoke* (gate fix vs baseline) reuse does NOT matter: both arms
#     see identical data and we compare the mechanism, so SKIP_DOCS=0 is correct.
#   * For real *saturation* rounds you want FRESH tokens. A big blind skip is NOT
#     practical: skip_docs streams-and-discards docs at startup and DCLM-Edu streams
#     slowly (~minutes to even begin), so skipping millions of docs can stall for
#     hours. Prefer disjoint-by-construction data instead:
#       - FINEWEB_NAME=sample-100BT (different dump than the 10B base's sample-10BT)
#       - or a one-time calibration run that counts filtered docs/source to ~5B
#         tokens each, then resume with per_source_tokens (now saved in checkpoints).
#   * The 10B base predates per-source token tracking, so its exact consumed-doc
#     offset is unknown; new runs save per_source_tokens for exact future resumes.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3_chat}"
TOKEN_BUDGET="${1:-5000000000}"             # ~5B / round
shift || true

GATE="${GATE:-1}"                           # 1=phase-aware (default); 0=magnitude-only ablation
RESUME_FROM="${RESUME_FROM:-checkpoints_v11_e3_k3_chat_pretrain/best_model.pt}"
RESUME="${RESUME:-}"

DATASET="${DATASET:-pretrain_mix}"
PRETRAIN_SOURCES="${PRETRAIN_SOURCES:-dclm,fineweb}"
PRETRAIN_WEIGHTS="${PRETRAIN_WEIGHTS:-1,1}"
FINEWEB_NAME="${FINEWEB_NAME:-sample-10BT}"
EDU_SCORE_MIN="${EDU_SCORE_MIN:-3}"

# Default 0 (smoke-appropriate: fast start, valid A/B). For fresh saturation tokens
# prefer a different shard/config (see header) over a large skip.
DCLM_SKIP_DOCS="${DCLM_SKIP_DOCS:-0}"
FINEWEB_SKIP_DOCS="${FINEWEB_SKIP_DOCS:-0}"

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-18}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
EPOCHS=9999
LR="${LR:-1e-4}"                            # warm-restart peak (< from-scratch 3e-4)
WARMUP_STEPS="${WARMUP_STEPS:-500}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-5000}"
SEED="${SEED:-1234}"                        # distinct from base run for fresh interleave
COMPILE="${COMPILE:-1}"                      # 0 => skip torch.compile (fast pipeline smoke)
EXTRA_ARGS="$*"

COMPILE_ARGS="--compile --compile_mode default"
if [[ "$COMPILE" == "0" ]]; then
  COMPILE_ARGS=""
fi

GATE_ARGS=""
RUN_KIND="gate"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_sat}"
if [[ "$GATE" == "0" ]]; then
  GATE_ARGS="--no_gate_content_aware"
  RUN_KIND="mag"
  CKPT_DIR="${CKPT_DIR_MAG:-checkpoints_v11_sat_mag}"
fi
mkdir -p "$CKPT_DIR"

GEN_PROMPT="In 1923 , the University of"
TAG="${PRESET}_saturate_${RUN_KIND}_b${TOKEN_BUDGET}"
if [[ -n "$RESUME" ]]; then
  LOG_DIR="${LOG_DIR:-$(make_log_dir "v11" "${TAG}_resume")}"
else
  LOG_DIR=$(make_log_dir "v11" "$TAG")
fi

ARGS="--preset $PRESET --stage pretrain --dataset $DATASET --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
  --token_budget $TOKEN_BUDGET --edu_score_min $EDU_SCORE_MIN \
  --pretrain_sources $PRETRAIN_SOURCES --pretrain_weights $PRETRAIN_WEIGHTS \
  --fineweb_name $FINEWEB_NAME \
  --dclm_skip_docs $DCLM_SKIP_DOCS --fineweb_skip_docs $FINEWEB_SKIP_DOCS \
  --seed $SEED --lr $LR --warmup_steps $WARMUP_STEPS --weight_decay $WEIGHT_DECAY \
  --resume_from $RESUME_FROM $GATE_ARGS \
  --amp_dtype auto --num_workers 0 --gen_every 5000 --save_every_steps $SAVE_EVERY_STEPS \
  --no_grad_ckpt $COMPILE_ARGS"

if [[ -n "$RESUME" ]]; then
  ARGS="$ARGS --resume $RESUME"
fi

RUN_DESC="V11 saturate ($RUN_KIND): $PRESET continue from $RESUME_FROM, budget=${TOKEN_BUDGET} tok, skip dclm=$DCLM_SKIP_DOCS fineweb=$FINEWEB_SKIP_DOCS"
if [[ -n "$RESUME" ]]; then
  append_run_info_resume "$LOG_DIR" "Resume from $RESUME" "$ARGS --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"
else
  write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS --gen_prompt '$GEN_PROMPT' $EXTRA_ARGS"
fi

echo "============================================================"
echo "  V11 saturate ($RUN_KIND)  preset=$PRESET  budget=$TOKEN_BUDGET"
echo "  resume_from=$RESUME_FROM  gate_content_aware=$GATE"
echo "  skip dclm=$DCLM_SKIP_DOCS fineweb=$FINEWEB_SKIP_DOCS  seed=$SEED  lr=$LR"
echo "  Log: $LOG_DIR   Ckpt: $CKPT_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
  $ARGS \
  --gen_prompt "'$GEN_PROMPT'" \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CKPT_DIR" \
  $EXTRA_ARGS
