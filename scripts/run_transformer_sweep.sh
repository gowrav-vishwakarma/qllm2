#!/bin/bash
# Transformer baseline scaling sweep: 5M, 10M, 50M, 100M
# PyTorch/CUDA. Run on GPU machine.
#
# Usage: bash scripts/run_transformer_sweep.sh [size]
#   size: 5m, 10m, 50m, 100m
#   e.g.: bash scripts/run_transformer_sweep.sh 10m

set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${1:-all}

run_transformer() {
    local size_tag=$1 bs=$2 lr=$3
    echo "=== Transformer $size_tag ==="
    uv run python -m v6.train_transformer_baseline \
        --size "$size_tag" \
        --batch_size $bs --seq_len 512 --lr $lr --warmup_steps 500 --epochs 10 \
        --checkpoint_dir "checkpoints_transformer_${size_tag}" \
        --log_dir "logs/v6_transformer_${size_tag}"
}

if [[ "$SIZE" == "5m" || "$SIZE" == "all" ]]; then
    run_transformer 5m 8 3e-4
fi

if [[ "$SIZE" == "10m" || "$SIZE" == "all" ]]; then
    run_transformer 10m 8 3e-4
fi

if [[ "$SIZE" == "50m" || "$SIZE" == "all" ]]; then
    run_transformer 50m 4 1e-4
fi

if [[ "$SIZE" == "100m" || "$SIZE" == "all" ]]; then
    run_transformer 100m 3 1e-4
fi