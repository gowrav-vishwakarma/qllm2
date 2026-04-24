#!/usr/bin/env bash
# Transformer 5M and 10M runs at canonical sweep config
# (lr=3e-5, batch=8, seq_len=512, warmup=500, epochs=10)
# to match the QPAM/RPAM scaling sweep for cross-architecture comparison.
#
# Usage:
#   bash scripts/run_transformer_5m_10m_canonical.sh 5m
#   bash scripts/run_transformer_5m_10m_canonical.sh 10m
#   bash scripts/run_transformer_5m_10m_canonical.sh both

set -euo pipefail

run_5m() {
  if [[ -d checkpoints_transformer_5m ]]; then
    mv checkpoints_transformer_5m checkpoints_transformer_5m_oldconfig
  fi
  uv run python -m v6.train_transformer_baseline --d_model 88 --n_layers 4 --n_heads 2 --d_ff 352 --seq_len 512 --batch_size 8 --lr 3e-5 --warmup_steps 500 --weight_decay 0.01 --epochs 10 --checkpoint_dir checkpoints_transformer_5m --log_dir logs/v6_transformer_5m
}

run_10m() {
  if [[ -d checkpoints_transformer_10m ]]; then
    mv checkpoints_transformer_10m checkpoints_transformer_10m_oldconfig
  fi
  uv run python -m v6.train_transformer_baseline --d_model 136 --n_layers 13 --n_heads 2 --d_ff 544 --seq_len 512 --batch_size 8 --lr 3e-5 --warmup_steps 500 --weight_decay 0.01 --epochs 10 --checkpoint_dir checkpoints_transformer_10m --log_dir logs/v6_transformer_10m
}

case "${1:-both}" in
  5m)   run_5m ;;
  10m)  run_10m ;;
  both) run_5m ; run_10m ;;
  *)    echo "usage: $0 {5m|10m|both}" >&2 ; exit 1 ;;
esac
