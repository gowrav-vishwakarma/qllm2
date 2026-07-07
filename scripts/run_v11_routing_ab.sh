#!/usr/bin/env bash
# WikiText-103 competitive-retrieval A/B (3-epoch smoke, from scratch).
# Same recipe as the gate A/B that shipped gate_content_aware (2026-06-30).
#
# Arms (priority order):
#   control              — v11_e3_k3 baseline (gate on, routing/compete off)
#   routing_ca           — +routing_content_aware
#   compete              — +routing_content_aware +state_compete
#   compete_balance      — +compete +route_balance_lambda=0.01
#   phase_spread         — +compete +phase_init=spread (no balance)
#
# Usage:
#   ./scripts/run_v11_routing_ab.sh              # all arms sequentially
#   ./scripts/run_v11_routing_ab.sh control      # single arm
#   ARM=compete ./scripts/run_v11_routing_ab.sh  # env override
#
# Kill any arm not tracking below control at epoch 3; log in v11/EXPERIMENTS_V11.md.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh
# shellcheck disable=SC1091
source ./scripts/log_utils.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRESET_BASE="${PRESET_BASE:-v11_e3_k3}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-18}"
SEQ_LEN="${SEQ_LEN:-2048}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
ONLY_ARM="${1:-${ARM:-all}}"

run_arm() {
  local name="$1"
  local ckpt_dir="$2"
  shift 2
  local extra=("$@")

  if [[ "$ONLY_ARM" != "all" && "$ONLY_ARM" != "$name" ]]; then
    echo "[skip] arm=$name (ONLY_ARM=$ONLY_ARM)"
    return 0
  fi

  echo "============================================================"
  echo "  Routing A/B arm: $name"
  echo "  ckpt: $ckpt_dir"
  echo "  extra: ${extra[*]}"
  echo "============================================================"

  LOG_DIR=$(make_log_dir "v11" "${PRESET_BASE}_routingab_${name}")
  mkdir -p "$ckpt_dir"

  ARGS="--preset $PRESET_BASE --dataset wikitext103 --seq_len $SEQ_LEN \
    --batch_size $BATCH_SIZE --epochs $EPOCHS --chunk_size $CHUNK_SIZE \
    --max_samples 9999999 --amp_dtype auto --num_workers 4 --gen_every 5000 \
    --no_grad_ckpt --compile --compile_mode default \
    --gate_content_aware"

  write_run_info "$LOG_DIR" "V11 routing A/B arm=$name" "$ARGS ${extra[*]}"

  eval "$PYTHON_BIN -m v11.train" \
    $ARGS \
    --gen_prompt "'In 1923 , the University of'" \
    --log_dir "$LOG_DIR" \
    --checkpoint_dir "$ckpt_dir" \
    "${extra[@]}"
}

# Arm 0: control (gate on, competitive retrieval off)
run_arm control checkpoints_v11_routingab_control

# Arm 1: content-aware phase routing
run_arm routing_ca checkpoints_v11_routingab_routing_ca \
  --routing_content_aware

# Arm 2: magnitude competition
run_arm compete checkpoints_v11_routingab_compete \
  --routing_content_aware --state_compete

# Arm 3: compete + MoE-style load balance
run_arm compete_balance checkpoints_v11_routingab_compete_balance \
  --routing_content_aware --state_compete --route_balance_lambda 0.01

# Arm 4: compete + spread phase init (optimization-speed check)
run_arm phase_spread checkpoints_v11_routingab_phase_spread \
  --routing_content_aware --state_compete --phase_init spread

echo "[done] Routing A/B complete. Compare epoch-3 val PPL in logs/v11/*routingab*"
echo "       Record results in v11/EXPERIMENTS_V11.md; promote winner to PRESET=v11_e3_k3_chat_compete"
