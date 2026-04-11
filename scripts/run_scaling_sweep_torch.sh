#!/bin/bash
# Scaling sweep: QPAM vs RPAM at 5M, 10M, 50M, 100M, 1B
# PyTorch/CUDA version. Run on GPU machine.
#
# Usage: bash scripts/run_scaling_sweep_torch.sh [size] [model]
#   size: 5m, 10m, 50m, 100m, 1b
#   model: qpam, rpam

set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${1:-all}
MODEL=${2:-both}

# QPAM uses v6/train.py with --size and --dataset wikitext103
run_qpam() {
    local size_tag=$1
    echo "=== QPAM $size_tag ==="
    python -m v6.train \
        --size "medium-pam-v3-${size_tag}" \
        --dataset wikitext103 \
        --epochs 10 \
        --seq_len 512 \
        --save_dir "checkpoints_qpam_${size_tag}_torch"
}

# RPAM uses v6/train_real.py with --size
run_rpam() {
    local size_tag=$1
    echo "=== RPAM $size_tag ==="
    python -m v6.train_real \
        --size "rpam-${size_tag}" \
        --epochs 10 \
        --seq_len 512 \
        --save_dir "checkpoints_rpam_${size_tag}_torch"
}

if [[ "$SIZE" == "5m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 5m
    [[ "$MODEL" != "qpam" ]] && run_rpam 5m
fi

if [[ "$SIZE" == "10m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 10m
    [[ "$MODEL" != "qpam" ]] && run_rpam 10m
fi

if [[ "$SIZE" == "50m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 50m
    [[ "$MODEL" != "qpam" ]] && run_rpam 50m
fi

if [[ "$SIZE" == "100m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 100m
    [[ "$MODEL" != "qpam" ]] && run_rpam 100m
fi

if [[ "$SIZE" == "1b" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 1b
    [[ "$MODEL" != "qpam" ]] && run_rpam 1b
fi