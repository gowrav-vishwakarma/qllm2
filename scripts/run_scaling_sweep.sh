#!/bin/bash
# Scaling sweep: QPAM vs RPAM at 5M, 10M, 50M, 100M, 1B
# Each pair is param-matched. Run sequentially or pick one.
#
# Usage: bash scripts/run_scaling_sweep.sh [size] [model]
#   size: 5m, 10m, 50m, 100m, 1b
#   model: qpam, rpam
#   e.g.: bash scripts/run_scaling_sweep.sh 10m qpam

set -euo pipefail
cd /Users/caug/npcww/qnlp/qllm-private

SIZE=${1:-all}
MODEL=${2:-both}

run_qpam() {
    local dim=$1 layers=$2 heads=$3 head_dim=$4 batch=$5 lr=$6 tag=$7
    echo "=== QPAM $tag: dim=$dim L=$layers H=$heads hd=$head_dim ==="
    uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train.py \
        --dim $dim --layers $layers --heads $heads --head_dim $head_dim \
        --batch_size $batch --seq_len 512 --lr $lr --warmup 500 --epochs 10 \
        --save_dir checkpoints_qpam_${tag}_mlx
}

run_rpam() {
    local dim=$1 layers=$2 heads=$3 head_dim=$4 batch=$5 lr=$6 tag=$7
    echo "=== RPAM $tag: dim=$dim L=$layers H=$heads hd=$head_dim ==="
    uv run python /Users/caug/npcww/qnlp/ket-nlp/qpam_mlx/train_real.py \
        --dim $dim --layers $layers --heads $heads --head_dim $head_dim \
        --batch_size $batch --seq_len 512 --lr $lr --warmup 500 --epochs 10 \
        --save_dir checkpoints_rpam_${tag}_mlx
}

# ── 5M (~5.0M params each) ──
if [[ "$SIZE" == "5m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam  44  12 2 16  16 3e-04 5m
    [[ "$MODEL" != "qpam" ]] && run_rpam  84  11 2  8  16 3e-04 5m
fi

# ── 10M (~10.0M params each) ──
if [[ "$SIZE" == "10m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam  80  12 4 16  16 3e-04 10m
    [[ "$MODEL" != "qpam" ]] && run_rpam 140  11 8 16  16 3e-04 10m
fi

# ── 50M (~50.0M params each) ──
if [[ "$SIZE" == "50m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 292  12 4 16   8 1e-04 50m
    [[ "$MODEL" != "qpam" ]] && run_rpam 496  11 2  8   8 1e-04 50m
fi

# ── 100M (~100-120M params each, existing configs) ──
if [[ "$SIZE" == "100m" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 384  16 6 64   8 3e-05 100m
    [[ "$MODEL" != "qpam" ]] && run_rpam 576  16 9 64   8 3e-05 100m
fi

# ── 1B (~980M params each) ──
if [[ "$SIZE" == "1b" || "$SIZE" == "all" ]]; then
    [[ "$MODEL" != "rpam" ]] && run_qpam 1152 24 16 72  1 3e-05 1b
    [[ "$MODEL" != "qpam" ]] && run_rpam 1632 24 24 68  1 3e-05 1b
fi
