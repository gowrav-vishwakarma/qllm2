#!/usr/bin/env bash
# V13 D RUN (2026-08-26): 200M VALIDATION of the DENSE short-context recall
# curriculum — the probe-hard-case training signal for the oracle-identified
# READ-SIDE routing gap.
#
# WHY (post-B verdict, v13/SCRATCHPAD.md "FINAL 500M VERDICT" +
# "NEXT-RUN DECISION"):
#   B (500M, 4% SPARSE recall slice: 3-6 bindings over a 2-200-sentence gap)
#   improved Wiki PPL (128.76 vs r1 133.88, -3.8%) but did NOT break the
#   8-way recall plateau (n8 0.1453 vs r1 0.1333, z+1.31 < 2sigma). Root
#   cause: the slice's distribution never trained the probe's hard case —
#   8 DISTINCT single-token bindings packed into ~128 tok with the query
#   immediately after. The oracle (probe_oracle.py) proved the 8th value IS
#   stored but the learned query is ~orthogonal to its address: the
#   READ-SIDE routing gap. B's data shape simply never exercised it.
#
# CHANGES vs B (everything else IDENTICAL — single-variable test):
#   1. v7/data.py recall generator now emits a DENSE variant 50% of the time
#      (6-8 distinct single-token bindings, 0-2 sentence gap, query-one-back;
#      values from a 14-value GPT-2 single-token pool DISJOINT from the
#      probe KEYS/VALUES -> no leakage). The other 50% is the original
#      long-range mix (which drove B's PPL win). Cache v2 -> v3.
#   2. --token_budget 500000000 -> 200000000 (validation budget; B's recall
#      trajectory was flat by 164M, so 200M suffices to detect a break and
#      costs ~13h vs 32h).
#
# BASE = B recipe: v13_e3_k3_selective, B8/T2048, lr 3e-4, warmup 500,
#   seed 42, EAGER (no --compile_blocks: Inductor crash on aten.complex.default),
#   --fused_ce --delta_raw_key_readout --delta_erase_beta_cap 1.0,
#   --pretrain_sources dclm,fineweb,smoltalk2_mid,recall --pretrain_weights
#   48,48,4,4 --blend_warmup_tokens 1e7 --edu_score_min 3 --fineweb sample-10BT.
#
# GATES:
#   KILL if train loss > ~0.7 NLL above r1 curve (7.52@5M, 6.66@10M, 5.87@20M,
#   4.81@50M, 3.97@200M) or the dense docs destabilize CE.
#   RECALL GATE @5000 steps (81.9M, ckpt5000) — the CLEAN A/B: B at the same
#   82M token count scored multi8@128 = 0.106. If D's 82M ckpt is clearly above
#   that (> 0.13) with CE non-regressing, the dense curriculum is working.
#   FINAL GATE @200M (step ~12200): n8@128 > 0.15 AND n8-allctx > B-246M
#   (0.233) by > 0.03 -> continue to 500M / declare; FLAT at ~0.133 (B level)
#   -> substrate can't route 8-way at 500M scale, BANK C (v11 additive:
#   3.2x cheaper/token, already beats v13 on Wiki PPL at compute-matched sizes).
#   Battery: scripts/run_memory_behavioral.py, 60 trials, probe config
#   delta_raw_key_readout=true + delta_erase_beta_cap=1.0. Baselines:
#   r1 FINAL logs/memory_probes/v13_500m_r1recipe_FINAL500M_d169584_behavior.json
#   B FINAL logs/memory_probes/v13_B_recall_FINAL500M_d201737_behavior.json
#
# NOTE: fresh v3 pretrain-mix cache (generator changed) — first launch spends
#   ~1-2h building .cache/v7_tokens/pretrain_mix_v3_*.
#
# Launch detached:
#   tmux new-session -d -s v13_D 'bash v13/tmp/launch_v13_D_dense.sh 2>&1 | tee -a logs/v13/200m_v13_D_dense/tmux_console.log'
# Watchdog chain (re-arm on every wake; verdict at 100M):
#   bash v13/tmp/watchdog.sh logs/v13/200m_v13_D_dense/v11_v13_e3_k3_selective_lm_pretrain_mix.log 100000000 2940
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
    --token_budget 200000000 \
    --batch_size 8 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --amp_dtype auto \
    --fused_ce \
    --delta_raw_key_readout \
    --delta_erase_beta_cap 1.0 \
    --gen_every 5000 \
    --save_every_steps 2500 \
    --gen_prompt 'In 1923, the University of' \
    --blend_warmup_tokens 10000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --log_interval 25 \
    --seed 42 \
    --checkpoint_dir checkpoints_v13/200m_v13_D_dense \
    --log_dir logs/v13/200m_v13_D_dense \
    "$@"
