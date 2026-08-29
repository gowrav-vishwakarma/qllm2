#!/usr/bin/env bash
# V13 B RUN (2026-08-25): 500M retrain with a 4% SYNTHETIC RECALL SLICE +
# two-state raw-key readout ON + erase cap 1.0. The routing-fix run.
#
# WHY (oracle evidence, v13/SCRATCHPAD.md "2026-08-25 ROOT-CAUSE"):
#   The 500M r1recipe ckpt stores multi8 values as scattered superpositions
#   (oracle recovers the 8th value on 11/128 addresses; random/zero query
#   controls fail) but the learned query is ~orthogonal to every fact-key
#   address. Bottleneck = ROUTING/alignment, not write interference. r1
#   never saw any store-now/answer-later signal: it ran with
#   --blend_warmup_tokens 1e9 > 5e8 token_budget, i.e. WEB-ONLY for the whole
#   run (the recall/chat sources never entered the mix). This run fixes that.
#
# CHANGES vs r1recipe (everything else identical to the 500M run):
#   1. --pretrain_sources dclm,fineweb,smoltalk2_mid,recall
#      --pretrain_weights 48,48,4,4        (4% recall slice, ~20M tokens)
#      recall vocab is DISJOINT from the behavioral probe (v7/data.py:1308-1312)
#      -> no probe leakage. Small on purpose: v11 Stage-3 found MORE recall
#      data HURT (w3 > w10 > w20); the fix is a routing signal, not flooding.
#   2. --blend_warmup_tokens 10000000      (1e9 -> 1e7): short grammar warmup,
#      then the full 48/48/4/4 blend for 490M tokens. The 1e9 value made the
#      500M run web-only forever (see WHY).
#   3. --delta_raw_key_readout             (two-state readout ON; selftest
#      [raw_readout] recurrent-exact, erase/mass state bit-identical to OFF)
#   4. --delta_erase_beta_cap 1.0          (0.95 -> 1.0; safe under unit keys,
#      eig = 1 - beta_e >= 0)
#
# BASE = same 500M budget/recipe: B8, T2048, lr 3e-4, warmup 500,
#   edu>=3, fineweb sample-10BT, seed 42, EAGER (no --compile_blocks:
#   Inductor crash on aten.complex.default, 2026-08-23).
#
# KILL if train loss > ~0.7 NLL above r1 reference at 20M (i.e. > 6.6):
#   r1 curve: 7.52@5M, 6.66@10M, 5.87@20M, 4.81@50M, 4.36@100M, 3.97@200M.
# RECALL GATE (first ckpt @5000 steps): multi8@128 must move off 0.1333
#   (chance 0.125) and CE must not regress. Then full battery + Wiki PPL.
#   Baseline: logs/memory_probes/v13_500m_r1recipe_FINAL500M_d169584_behavior.json
#
# NOTE: fresh pretrain-mix cache (sources/weights/warmup are cache-keyed) —
#   first launch spends ~1-2h building .cache/v7_tokens/pretrain_mix_*.
#
# Launch detached:
#   tmux new-session -d -s v13_B 'bash v13/tmp/launch_v13_B_recall.sh 2>&1 | tee -a logs/v13/500m_v13_B_recall/tmux_console.log'
# Watchdog chain (re-arm on every wake):
#   bash v13/tmp/watchdog.sh logs/v13/500m_v13_B_recall/v11_v13_e3_k3_selective_lm_pretrain_mix.log 20000000 2940
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
    --token_budget 500000000 \
    --batch_size 8 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --amp_dtype auto \
    --fused_ce \
    --delta_raw_key_readout \
    --delta_erase_beta_cap 1.0 \
    --gen_every 5000 \
    --save_every_steps 5000 \
    --gen_prompt 'In 1923, the University of' \
    --blend_warmup_tokens 10000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --log_interval 25 \
    --seed 42 \
    --checkpoint_dir checkpoints_v13/500m_v13_B_recall \
    --log_dir logs/v13/500m_v13_B_recall \
    "$@"
