#!/usr/bin/env bash
# Run V6 training on RTX 4090 (24GB), intended for tmux.
#
# Logs to:      logs/v6_train_small-matched.log  (auto, via TeeLogger)
# Checkpoints:  checkpoints_v6/
#
# Suggested workflow:
#   1) Tune batch first (optional):
#      ./scripts/tune_batch_v6_rtx4090.sh --batch_size 16
#   2) Standard run:
#      ./scripts/run_v6_rtx4090.sh
#   3) Resume if interrupted:
#      ./scripts/run_v6_rtx4090.sh --resume checkpoints_v6/best_model.pt --epochs 20
#   4) Monitor from another terminal:
#      ./scripts/monitor_v6.sh 5 logs/v6_train_small-matched.log
#
# Ablation runs:
#   ./scripts/run_v6_rtx4090.sh --no_working_memory
#   ./scripts/run_v6_rtx4090.sh --no_internal_memory
#
# On server:
#   tmux new -s v6train
#   cd ~/Development/qllm2 && ./scripts/run_v6_rtx4090.sh [--batch_size N]
#   # Detach: Ctrl+B, D. Reattach: tmux attach -t v6train

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
[[ -f v6/train.py ]] || cd ..

# Bootstrap env and deps; exports PYTHON_BIN.
# shellcheck disable=SC1091
source ./scripts/v6_env_setup.sh

CHECKPOINT_DIR="checkpoints_v6"
if echo "$@" | grep -q -- '--resume'; then
  echo "[v6-run] Resuming -- keeping existing checkpoints in $CHECKPOINT_DIR/"
else
  if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
    echo "[v6-run] Fresh start -- clearing old checkpoints in $CHECKPOINT_DIR/"
    rm -rf "$CHECKPOINT_DIR"
  fi
fi

# RTX 4090 (24GB): batch 16 is safe for small-matched. Use --batch_size 24 or 32 if you have headroom.
eval "$PYTHON_BIN -m v6.train" \
  --size small-matched \
  --max_samples 100000 \
  --seq_len 256 \
  --batch_size 16 \
  --epochs 10 \
  --init_seed 42 \
  --log_dir logs \
  --checkpoint_dir checkpoints_v6 \
  "$@"
