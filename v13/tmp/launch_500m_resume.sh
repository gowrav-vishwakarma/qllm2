#!/usr/bin/env bash
# Re-runnable: resume the 100M-class 500M-token real-data run on the 4090,
# now on the K-batched fused-delta fast path (fused_e3=True in the preset since
# 2026-08-22; math-exact vs the old K-loop, 2.3K -> ~21K tok/s train step).
# Also includes the O(1) gate-surprisal redesign (2026-08-22): NLL byproduct of
# the main fused CE as the exact gate target (no 2nd O(V) GEMM) + gate stash
# built OUTSIDE the gradient-checkpoint region (no PAM recompute on bwd).
# Gate-ON step == gate-OFF step: 20,935 tok/s @ B16, 14.2GB of 24GB (4090).
#
# Resume point: checkpoints_v13/100m_realdat_500m/latest.pt (step 500, 10.24M tok).
#
# Launch DETACHED in tmux (survives session end):
#   tmux new-session -d -s v13_500m 'bash v13/tmp/launch_500m_resume.sh 2>&1 | tee -a logs/v13/100m_realdat_500m/tmux_console.log'
#
# Batch 16 chosen from 2026-08-22 sweep (4090, T=2048, gate-ON):
#   B16 20,935 tok/s @ 14.2GB | B20 20,388 @ 17.6GB | B24 19,956 @ 20.9GB.
set -euo pipefail
cd /home/gowrav/Development/qllm2

# HuggingFace Hub is flaky from this box: a transient slow `repo_info`/etag
# response trips the 10s default read-timeout and kills the data loader
# (ReadTimeout on huggingface.co, seen 2026-08-22). The stream itself is fine
# (fineweb sample-10BT first row ~39s). Raise the hub timeouts so a slow
# metadata call can't abort a long run; streaming still uses the cached hub.
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=300

exec .venv/bin/python -m v13.train \
    --preset v13_e3_k3_selective \
    --stage lm \
    --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb,smoltalk2_mid,recall,reason \
    --pretrain_weights 70,20,5,5,5 \
    --token_budget 500000000 \
    --batch_size 16 \
    --amp_dtype auto \
    --fused_ce \
    --resume checkpoints_v13/100m_realdat_500m/latest.pt \
    --checkpoint_dir checkpoints_v13/100m_realdat_500m \
    --log_dir logs/v13/100m_realdat_500m \
    "$@"
