#!/usr/bin/env bash
# V11 PAM 500M HEAD-TO-HEAD (2026-08-22): the additive v11 reference at
# matched tokens, so the 500M V13 (delta stack) can be compared on
# WikiText-103 val PPL. EXACT round-1 recipe (the real new-code baseline):
#   --preset v11_e3_k3_chat (additive, K=3), lr 3e-4, warmup 500, B18, T2048,
#   48/48/4 dclm/fineweb/smoltalk2_mid, edu>=3, sample-10BT, blend 1e9, seed 42.
# Gradient checkpointing ON (4090 24GB) — same math as round-1's no_grad_ckpt,
# same trajectory. 500M budget. Chained after the V13 500M run via
# v13/tmp/relay_v11_after_v13.sh.
set -euo pipefail
cd /home/gowrav/Development/qllm2
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=300
exec .venv/bin/python -m v11.train \
    --preset v11_e3_k3_chat \
    --stage pretrain \
    --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb,smoltalk2_mid \
    --pretrain_weights 48,48,4 \
    --token_budget 500000000 \
    --batch_size 18 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --amp_dtype auto \
    --gen_every 5000 \
    --save_every_steps 5000 \
    --gen_prompt 'In 1923, the University of' \
    --blend_warmup_tokens 1000000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --log_interval 25 \
    --seed 42 \
    --checkpoint_dir checkpoints_v11/500m_v11_r1recipe \
    --log_dir logs/v11/500m_v11_r1recipe \
    "$@"
