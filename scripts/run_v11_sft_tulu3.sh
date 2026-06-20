#!/usr/bin/env bash
# Phase 3 chat SFT: Tulu-3 mixture (knowledge + coding + general) on the NEW
# from-scratch chat base. ChatML, assistant-only loss, in-distribution holdout.
#
# Usage:
#   tmux new-session -d -s v11_sft \
#     './scripts/run_v11_sft_tulu3.sh checkpoints_v11_e3_k3_chat_pretrain/best_model.pt'
#   tmux attach -t v11_sft
#
# Resume a crashed SFT (continues optimizer + scheduler + step):
#   RESUME=checkpoints_v11_sft_chat/best_model.pt \
#     tmux new-session -d -s v11_sft './scripts/run_v11_sft_tulu3.sh'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3_chat}"
RESUME_FROM="${1:-checkpoints_v11_e3_k3_chat_pretrain/best_model.pt}"
shift || true

SEQ_LEN="${SEQ_LEN:-2048}"
BATCH_SIZE="${BATCH_SIZE:-18}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
SFT_FILTER="${SFT_FILTER:-hard}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_sft_chat}"
RESUME="${RESUME:-}"
EXTRA_ARGS="$*"

GEN_PROMPT="<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
LOG_DIR=$(make_log_dir "v11" "${PRESET}_sft_tulu3")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --stage sft --dataset tulu3 --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
  --lr $LR --warmup_steps $WARMUP_STEPS --sft_filter $SFT_FILTER \
  --amp_dtype auto --num_workers 4 --gen_every 0 --no_grad_ckpt \
  --compile --compile_mode default"

# Resume (full state) takes precedence over resume_from (weights-only base).
if [[ -n "$RESUME" ]]; then
  ARGS="$ARGS --resume $RESUME"
else
  ARGS="$ARGS --resume_from $RESUME_FROM"
fi

RUN_DESC="V11 Phase 3 SFT: $PRESET on Tulu-3 (knowledge+coding+general)"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS $EXTRA_ARGS"

echo "============================================================"
echo "  V11 Phase 3 Tulu-3 SFT"
echo "  preset=$PRESET  base=$RESUME_FROM  epochs=$EPOCHS  resume=${RESUME:-<none>}"
echo "  Log: $LOG_DIR   Ckpt: $CKPT_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
  $ARGS \
  --gen_prompt "'$GEN_PROMPT'" \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CKPT_DIR" \
  $EXTRA_ARGS
