#!/usr/bin/env bash
# V13 E RUN (2026-08-27): 82M MATCHED-TOKEN A/B — does the zero-parameter
# n-gram content read (Qwen3.8-Flash-Next PLE port) close the 8-way
# multi-binding recall gap that the oracle proved is write-key SEPARABILITY?
#
# WHY (post-D-200M verdict — LAUNCH ONLY AFTER THE D FINAL GATE):
#   The oracle (probe_oracle.py) proved the 8-way failure is NOT a storage
#   gap — the 8th value IS stored — but a READ-SIDE routing gap: 8 similar
#   bindings -> 8 near-identical write keys -> the learned query is ~orthogonal
#   to the 8th address and can't disambiguate. The zero-param n-gram read
#   (v13/model.py _ngram_repr, commit 79db28e) attacks exactly this: it
#   hashes the causal 3-gram (key,verb,value) to a row of the EXISTING tied
#   embedding table and adds a scaled fingerprint to the token representation,
#   making each write key content-DISTINCT. Zero new params, O(1)/token.
#
# DESIGN (single-variable A/B, matched to D's 82M gate point):
#   E = D recipe + --ngram_read (n=3, scale=0.5), 82M tokens, seed 42.
#   Compared against the SAME 82M ckpt point:
#     D-82M  (dense curriculum, no ngram)  multi8@128 = 0.0667
#     B-82M  (sparse curriculum, no ngram) multi8@128 = 0.100-0.1167
#   If E-82M multi8@128 > ~0.13 (clearly above BOTH, with CE non-regressing),
#   the n-gram fingerprint is working and we scale it to 200M/500M. If E-82M
#   ~= D-82M (no lift), the zero-param fingerprint is insufficient and we need
#   the PAM-state row-read or a learned fusion block.
#
# NOTE: this is the PRIMARY recipe (on top of the D dense curriculum). If the
# D 200M FINAL GATE is FLAT/BANK-C, RE-COPY this script and flip the recipe to
# B or r1 (change the recall data shape / drop the dense variant) before launch.
#
# BASE = D recipe (v13_e3_k3_selective, B8/T2048, lr 3e-4, warmup 500, seed 42,
#   EAGER, --fused_ce --delta_raw_key_readout --delta_erase_beta_cap 1.0,
#   dclm,fineweb,smoltalk2_mid,recall 48/48/4/4, blend 1e7). The v3 pretrain-mix
#   cache already exists (D built it) -> no cache rebuild.
#
# RECALL GATE @5000 steps (~81.9M, latest.pt):
#   Battery (60-300 trials, probe config delta_raw_key_readout + cap 1.0 +
#   ngram_read=true/ngram_size=3/ngram_scale=0.5):
#     .venv/bin/python scripts/run_memory_behavioral.py --model-type v13 \
#       --checkpoint checkpoints_v13/82m_v13_E_ngram/latest.pt \
#       --preset v13_e3_k3_selective \
#       --v13-config delta_raw_key_readout=true --v13-config delta_erase_beta_cap=1.0 \
#       --v13-config ngram_read=true --v13-config ngram_size=3 --v13-config ngram_scale=0.5 \
#       --context-lengths 128,512,1024,2048 --positions 0,0.5,1 \
#       --association-counts 1,4,8 --trials 300 --candidate-count 8 \
#       --output logs/memory_probes/v13_E_ngram_ckpt5000_behavior.json
#   Compare multi8@128 vs D-82M 0.0667 / B-82M 0.100-0.1167.
#
# Launch detached (single-tenant — run AFTER the D 200M verdict):
#   tmux new-session -d -s v13_E 'bash v13/tmp/launch_v13_E_ngram.sh 2>&1 | tee -a logs/v13/82m_v13_E_ngram/tmux_console.log'
# Watchdog chain (re-arm on every wake):
#   bash v13/tmp/watchdog.sh logs/v13/82m_v13_E_ngram/v11_v13_e3_k3_selective_lm_pretrain_mix.log 82000000 2940
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
    --ngram_read \
    --ngram_size 3 \
    --ngram_scale 0.5 \
    --gen_every 5000 \
    --save_every_steps 1000 \
    --gen_prompt 'In 1923, the University of' \
    --blend_warmup_tokens 10000000 \
    --edu_score_min 3 \
    --fineweb_name sample-10BT \
    --num_workers 0 \
    --log_interval 25 \
    --seed 42 \
    --checkpoint_dir checkpoints_v13/82m_v13_E_ngram \
    --log_dir logs/v13/82m_v13_E_ngram \
    "$@"
