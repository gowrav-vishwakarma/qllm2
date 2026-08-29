#!/usr/bin/env bash
# V13 F RUN (2026-08-28): 82M LEARNED NGRAM FUSION BLOCK — does training the
# Qwen PLE fusion block around the n-gram hash lookup (instead of E's raw
# zero-param row add) close the 8-way multi-binding recall gap WITHOUT
# repeating E's easy-case tax?
#
# NOTE 2026-08-28 (F2 relaunch): F run-1 (21:36) was KILLED at 13.7M tok —
# the norm-after-key_proj ordering made the zero-init "slow start" a step
# function (injection max 3.027 after one optimizer step; +1.1..+1.5 NLL vs
# the matched D control, flat). Fixed in commit 4fddba2 (norm BEFORE
# key_proj; step-1 injection now 2.1e-02). This script is otherwise
# unchanged; see v13/SCRATCHPAD.md "F RUN-1 KILLED" + F2 checks.
# F2 matched-loss check @ ~step 500 (8.2M): must be within ~+0.15 of
# D's 6.0684; >+0.5 above D at 8-13M -> KILL, no third launch.
#
# WHY (post-E verdict, option B per user — see v13/SCRATCHPAD.md "2026-08-28 F"):
#   E (zero-param hash fingerprint, commit 3b5b14e) was a NET-NEGATIVE SWAP:
#   n8-all +0.033 over D-82M (real, ~1.6 SE) but n1-all 1.4 SE and n4-all
#   2.7 SE below D-82M — the cost exceeded the gain, and E failed its own
#   pre-registered bar (n8-all 0.1417 < 0.1667). The diagnosis: a raw hash
#   row added to the representation is CONTENT-BLIND (fixed scale 0.5, no
#   learned transform), so it can't suppress itself where it isn't useful
#   (the easy n1/n4 cases). F is the faithful Qwen port: NgramFusion =
#   depthwise causal Conv1d (kernel=ngram_size, over the 2*dim re/im rows)
#   -> ComplexLinear key_proj (ALL FOUR params ZERO-INIT) -> ComplexNorm.
#   Zero-init => step 0 is BIT-IDENTICAL to the no-fingerprint D model
#   (selftest init-off=0.0; GPU smoke fwd delta 0.0), and the fingerprint
#   signal GROWS with training — the model only pays for it if it finds
#   signal. +299,136 params (100.62M -> 100.92M), O(1)/token (rolling
#   _ngram_row_ctx buffer; boundary zero-fill = parallel window bit-exact).
#   Commit 9109fde.
#
# DESIGN (single-variable A/B vs E, matched recipe):
#   F = D recipe + --ngram_fusion (implies ngram_read, n=3; ngram_scale
#   dropped — the learned block subsumes it), 82M tokens, seed 42.
#   Controls at the same 82M point:
#     D-82M: n8-all 0.1083 | n1-all 0.1514 | n4-all 0.2139
#     E-82M: n8-all 0.1417 | n1-all 0.1211 | n4-all 0.1475  (GATE FAIL)
#     D-200M-final (n8 ceiling ref): n8-all 0.1367
#
# RECALL GATE @5000 steps (~81.9M, latest.pt), 300-trial battery, probe
# config delta_raw_key_readout=true delta_erase_beta_cap=1.0 ngram_fusion=true
# ngram_read=true ngram_size=3 (the checkpoint's config already carries
# ngram_fusion=true — the probe overrides exist so the command is explicit):
#     .venv/bin/python scripts/run_memory_behavioral.py --model-type v13 \
#       --checkpoint checkpoints_v13/82m_v13_F_ngram_fusion/latest.pt \
#       --preset v13_e3_k3_selective \
#       --v13-config delta_raw_key_readout=true --v13-config delta_erase_beta_cap=1.0 \
#       --v13-config ngram_fusion=true --v13-config ngram_read=true --v13-config ngram_size=3 \
#       --context-lengths 128,512,1024,2048 --positions 0,0.5,1 \
#       --association-counts 1,4,8 --trials 300 --candidate-count 8 \
#       --output logs/memory_probes/v13_F_ngram_fusion_ckpt5000_behavior.json
# PRE-REGISTERED PASS (ALL required):
#   1. n8-allctx >= 0.1667   (D-final 0.1367 + 0.03 — the E bar, unchanged)
#   2. n1-all    >= 0.1314   (D-82M 0.1514 - 0.02 — no repeat of E's tax)
#   3. CE non-regressing     (kill band: >0.7 NLL above r1 curve)
#   PASS -> scale F to 200M (D recipe, same flags) for the 8-way decision.
#   FAIL(1) only -> n8 ceiling holds even learned; bank the negative.
#   FAIL(2) with (1) -> net-negative swap like E; do not scale.
# KILL: loss > 0.7 NLL above r1 curve (7.52@5M, 5.87@20M, 4.81@50M, 4.36@100M).
# CANARY: [block-grad step1] must print all-nonzero (no DEAD=). The fusion
#   conv gets ZERO step-0 grad by construction (Jacobian through the zero
#   key_proj) but the canary inspects only raw.blocks, so it CANNOT
#   false-trip; key_proj is alive at step 0 and the conv from step 1.
#
# BASE = D/E recipe (v13_e3_k3_selective, B8/T2048, lr 3e-4, warmup 500,
#   seed 42, EAGER, --fused_ce --delta_raw_key_readout
#   --delta_erase_beta_cap 1.0, dclm,fineweb,smoltalk2_mid,recall
#   48/48/4/4, blend 1e7). v3 pretrain-mix cache exists (D built it) -> no
#   rebuild.
#
# Launch detached (GPU must be free):
#   tmux new-session -d -s v13_F 'bash v13/tmp/launch_v13_F_ngram_fusion.sh 2>&1 | tee -a logs/v13/82m_v13_F_ngram_fusion/tmux_console.log'
# Watchdog chain (re-arm on every wake):
#   bash v13/tmp/watchdog.sh logs/v13/82m_v13_F_ngram_fusion/v11_v13_e3_k3_selective_lm_pretrain_mix.log 82000000 2940
set -euo pipefail
cd /home/gowrav/Development/qllm2
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=300
exec .venv/bin/python -m v13.train \
    --preset v13_e3_k3_selective \
    --stage lm \
    --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb,smoltalk2_mid,recall \
    --pretrain_weights 48,48,4,4 \
    --token_budget 82000000 \
    --batch_size 8 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --amp_dtype auto \
    --fused_ce \
    --delta_raw_key_readout \
    --delta_erase_beta_cap 1.0 \
    --ngram_fusion \
    --ngram_size 3 \
    --gen_every 5000 \
    --save_every_steps 1000 \
    --gen_prompt 'In 1923, the University of' \
    --blend_warmup_tokens 10000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --log_interval 25 \
    --seed 42 \
    --checkpoint_dir checkpoints_v13/82m_v13_F_ngram_fusion \
    --log_dir logs/v13/82m_v13_F_ngram_fusion \
    "$@"
