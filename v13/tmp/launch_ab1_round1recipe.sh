#!/usr/bin/env bash
# A/B test #1 (2026-08-22): v13 code (additive path = bit-identical to v11,
# proven by v13/tmp/test_v11_v13_forward_ab.py) + round-1 EXACT recipe
# (lr 3e-4, WARMUP 500, B18, 48/48/4 dclm/fineweb/smoltalk2_mid,
# blend_warmup 1e9, edu>=3, sample-10BT). Delta stack OFF (additive, no
# vault/phase/lambda). This isolates "current v13 base + current data
# pipeline" vs the round-1 curve (v11 code from Jul 1).
#   Tracks round-1 (7.52@5M, 5.87@20M, 4.81@50M, 4.36@100M) => base+pipeline
#   clean; the earlier "3 NLL regression" was the stale Jun-23 ref + warmup
#   2000 artifact.
#   Stalls => shared drift (v7/data.py +1403 or v11/model.py +1023 since Jul 1).
# 4090 note: round-1 ran no_grad_ckpt on a 75GB GPU; on the 4090 we keep
# gradient checkpointing ON (same math, ~15GB).
set -euo pipefail
cd /home/gowrav/Development/qllm2
export HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=300
exec .venv/bin/python -m v13.train \
    --preset v13_e3_k3_selective \
    --stage lm \
    --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb,smoltalk2_mid \
    --pretrain_weights 48,48,4 \
    --token_budget 150000000 \
    --batch_size 18 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --amp_dtype auto \
    --fused_ce \
    --write_mode additive \
    --gate_surprisal_lambda 0.0 \
    --no_vault_state \
    --no_write_phase_address \
    --blend_warmup_tokens 1000000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --gen_every 999999 \
    --save_every_steps 0 \
    --log_interval 25 \
    --checkpoint_dir checkpoints_v13/ab1_round1recipe \
    --log_dir logs/v13/ab1_round1recipe \
    "$@"
