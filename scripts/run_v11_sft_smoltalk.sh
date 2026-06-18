#!/usr/bin/env bash
# Phase C SFT: filtered SmolTalk2 with assistant-only loss.
#
# Usage:
#   ./scripts/run_v11_sft_smoltalk.sh [pretrain_ckpt] [extra args]
#
# Examples:
#   ./scripts/run_v11_sft_smoltalk.sh checkpoints_v11_e3_k3_dclm/best_model.pt
#   tmux new-session -d -s v11_sft './scripts/run_v11_sft_smoltalk.sh'

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET="${PRESET:-v11_e3_k3}"
RESUME_FROM="${1:-checkpoints_v11_e3_k3_dclm/best_model.pt}"
shift || true

SEQ_LEN=2048
BATCH_SIZE=18
CHUNK_SIZE=256
EPOCHS=1
LR=5e-5
SFT_FILTER=hard
CKPT_DIR="${CKPT_DIR:-checkpoints_v11_e3_k3_sft}"
EXTRA_ARGS="$*"

GEN_PROMPT="### User:\nWhat is the capital of France?\n### Assistant:\n"
LOG_DIR=$(make_log_dir "v11" "${PRESET}_sft_smoltalk2")
mkdir -p "$CKPT_DIR"

ARGS="--preset $PRESET --stage sft --dataset smoltalk2 --seq_len $SEQ_LEN \
  --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
  --lr $LR --sft_filter $SFT_FILTER --resume_from $RESUME_FROM \
  --amp_dtype auto --num_workers 4 --gen_every 0 --no_grad_ckpt \
  --compile --compile_mode default"

RUN_DESC="V11 Phase C SFT: $PRESET on SmolTalk2 (filtered)"
write_run_info "$LOG_DIR" "$RUN_DESC" "$ARGS $EXTRA_ARGS"

echo "============================================================"
echo "  V11 Phase C SFT"
echo "  preset=$PRESET  resume=$RESUME_FROM  epochs=$EPOCHS"
echo "  Log: $LOG_DIR   Ckpt: $CKPT_DIR"
echo "============================================================"

eval "$PYTHON_BIN -m v11.train" \
  $ARGS \
  --gen_prompt "'$GEN_PROMPT'" \
  --log_dir "$LOG_DIR" \
  --checkpoint_dir "$CKPT_DIR" \
  $EXTRA_ARGS
