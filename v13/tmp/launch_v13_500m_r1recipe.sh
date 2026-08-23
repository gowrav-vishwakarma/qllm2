#!/usr/bin/env bash
# V13 MISSION RUN (2026-08-22): 100.5M-param selective-PAM, full delta stack
# (K=3 phase-addressed, delta-write, vault, GSP protect gate, gate-surprisal
# aux λ=0.1), 500M real tokens, RTX 4090.
#
# Recipe = round-1 EXACT (the real new-code baseline to beat):
#   lr 3e-4, warmup 500 (NOT 2000), B18, T2048, 48/48/4 dclm/fineweb/smoltalk2_mid,
#   edu>=3, fineweb sample-10BT, blend_warmup 1e9, seed 42.
# Source mix is 48/48/4 (matched to r1) for a clean comparison; recall/reason
# sources are added in a LATER run (per scratchpad NEXT.3).
#
# Code state: fused_ce bug FIXED (grad_weight was zero in v13 backward —
#  dropped line restored; v13/selftest + v13/tmp/test_fused_ce_grads.py PASS).
#  2026-08-23: NaN-crash fix IN — per-vector unit-norm keys in the delta rule
#  (delta_key_norm, commit 04dcebd). diag_gate_nan.py --max_steps 2000 PASSES
#  (state bounded ~12, no NON-FINITE past the old crash step 1711/63M).
#
# Round-1 reference curve (train loss @ gtok): 7.52@5M, 6.66@10M, 5.87@20M,
# 4.81@50M, 4.36@100M, 3.97@200M. KILL if >0.7 NLL above at 20M (i.e. >6.6).
#
# 2026-08-23: wipe the ckpt dir before this launch — every prior run in that
# dir trained under the _ckpt_block detach (15/16 blocks frozen). Extra flags
# (recommended: --compile_blocks --batch_size 8 --delta_chunk 128) pass
# through "$@". See v13/SCRATCHPAD.md ACTIVE TASK.
#
# Launch detached:
#   tmux new-session -d -s v13_500m 'bash v13/tmp/launch_v13_500m_r1recipe.sh --compile_blocks --batch_size 8 --delta_chunk 128 2>&1 | tee -a logs/v13/500m_v13_r1recipe/tmux_console.log'
set -euo pipefail
cd /home/gowrav/Development/qllm2
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=300
exec .venv/bin/python -m v13.train \
    --preset v13_e3_k3_selective \
    --stage lm \
    --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb,smoltalk2_mid \
    --pretrain_weights 48,48,4 \
    --token_budget 500000000 \
    --batch_size 18 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --amp_dtype auto \
    --fused_ce \
    --gen_every 5000 \
    --save_every_steps 5000 \
    --gen_prompt 'In 1923, the University of' \
    --blend_warmup_tokens 1000000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --log_interval 25 \
    --seed 42 \
    --checkpoint_dir checkpoints_v13/500m_v13_r1recipe \
    --log_dir logs/v13/500m_v13_r1recipe \
    "$@"
