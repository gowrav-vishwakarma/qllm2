#!/usr/bin/env bash
# Re-runnable: TRAIN FROM SCRATCH the 100M-class 500M-token real-data run on the
# 4090, with the full 2026-08-22 speed redesign (NOT a resume).
#
# Why fresh (2026-08-22): the old 100m_realdat_500m checkpoint (step 500 / 10.24M
# tok) was trained under the OLD gate aux (in-checkpoint stash + pre-fp32-CE), so
# its protect_gate weights + optimizer state reflect a different training regime.
# Resuming would blend two regimes. The ~9x speedup makes the 10.24M-token
# head-start only ~8 min, so a clean start is strictly better science.
#
# Speed redesign in this run (all measured on 4090):
#   1. K-batched fused-delta (fused_e3=True in preset): 2.3K -> ~21K tok/s.
#   2. O(1) gate-surprisal: exact per-token NLL is a FREE byproduct of the main
#      fused CE (loss._nll) — no second O(V) head GEMM for the gate target.
#   3. Gate stash built OUTSIDE the gradient-checkpoint region (V13Block._gate_in_det
#      detached leaf) so gate backprop needs no PAM recompute (was +364ms/step).

# LR recipe = the HF-uploaded v11-BEST run (qllm-pam-v11-e3k3-chat, 10B tok):
#   --lr 3e-4 --warmup_steps 2000 --batch_size 18 (its exact RUN_INFO args).
# The 2026-08-22 500M fresh attempt used the v13-program default lr 1e-4 and
# lagged v11-best by ~2.5 NLL @16M tok (9.20 vs 6.71) — that gap was lr, not
# architecture. Best-model priority (user, 2026-08-22): match the best recipe,
# sacrifice a few minutes. B18 (not B16) to match v11's batch exactly.
#
# Fresh dirs (old resume checkpoints + old logs removed; only this fresh dir
# remains under logs/v13/100m_*).
#
# Launch DETACHED in tmux (survives session end):
#   tmux new-session -d -s v13_100m 'bash v13/tmp/launch_100m_fresh.sh 2>&1 | tee logs/v13/100m_realdat_500m_fresh/tmux_console.log'
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
    --batch_size 18 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --amp_dtype auto \
    --fused_ce \
    --gen_every 5000 \
    --save_every_steps 5000 \
    --gen_prompt 'In 1923, the University of' \
    --checkpoint_dir checkpoints_v13/100m_realdat_500m_fresh \
    --log_dir logs/v13/100m_realdat_500m_fresh \
    "$@"
