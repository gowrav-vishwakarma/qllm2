#!/usr/bin/env bash
# DIAGNOSTIC (2026-08-22): isolate the v13-fresh learning stall.
#
# FINDING (see EXPERIMENTS_V13.md, "v13 vs v11 curve" section):
#   v13_e3_k3_selective fresh run (lr 3e-4, warmup 2000, B18, mix
#   70/20/5/5/5 dclm/fineweb/smoltalk/recall/reason) sits ~3.2 NLL ABOVE the
#   v11-best reference at every matched token:
#
#     gtok      v11 10B-scratch (additive, 50/50)   v13 fresh (delta, 70/20/5/5/5)
#     7.4M      8.67                                 9.89
#     20.3M     6.64                                 8.98
#     55.3M     5.27                                 ~8.4
#     75.6M     4.96                                 ~8.1
#
#   The lr/warmup ramp is IDENTICAL between the two runs (same lr 3e-4, same
#   warmup 2000 — verified step-by-step: lr 3.01e-05 @ step 200 in BOTH), so
#   lr/warmup is NOT the cause. Remaining suspects:
#     (a) delta-write selective-PAM stack (write_mode=delta + delta_erase_gate
#         + vault_state + write_phase_address + gate_surprisal lambda 0.1)
#         — all OFF in v11
#     (b) the 2026-08-22 speed-redesign code paths (K-batched fused delta,
#         O(1) NLL-byproduct gate target, gate stash outside gradient
#         checkpoint) — equivalence-tested, but a subtle shared-path bug
#         would still show up here
#     (c) data mix 70/20/5/5/5 vs 50/50 (15% harder data — expected offset
#         ~0.5-1.0 NLL, NOT 3.2)
#
# THIS RUN: v13 model+trainer (i.e. includes all speed-redesign code paths)
# forced to the EXACT 10B-scratch v11 reference recipe: additive writes,
# lambda 0, no vault, no phase-addressing, 50/50 dclm+fineweb, lr 3e-4,
# warmup 2000, B18, T2048.
#
#   - If loss tracks the reference (~8.7@7M, ~6.6@20M, ~5.3@55M): v13 base
#     code + speed redesign are sound; the delta-write selective stack (a) is
#     the cause of the stall. Next: per-arm ablations (delta-only, vault-only,
#     lambda-only, mix-only).
#   - If loss ALSO stalls ~9: a bug in the v13 shared code path (b). Hunt the
#     speed-redesign paths (fused K-batch, gate stash, NLL byproduct) next.
#
# 120M-token budget = ~1h at ~21K tok/s (full 24GB GPU; main 500M run stopped
# and checkpointed at step 5000 — resumable, so nothing is lost).
# No checkpoints in this run (save_every_steps 0), no gen (gen_every huge).
#
# tmux (session-independent):
#   tmux new-session -d -s v13_diag 'bash v13/tmp/launch_diag_additive.sh 2>&1 | tee logs/v13/diag_additive_20260822.log'
set -euo pipefail
cd /home/gowrav/Development/qllm2
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=300

exec .venv/bin/python -m v13.train \
    --preset v13_e3_k3_selective \
    --stage lm \
    --dataset pretrain_mix \
    --pretrain_sources dclm,fineweb \
    --pretrain_weights 50,50 \
    --token_budget 120000000 \
    --batch_size 18 \
    --seq_len 2048 \
    --lr 3e-4 \
    --warmup_steps 2000 \
    --amp_dtype auto \
    --fused_ce \
    --write_mode additive \
    --gate_surprisal_lambda 0.0 \
    --no_vault_state \
    --no_write_phase_address \
    --gen_every 999999 \
    --save_every_steps 0 \
    --log_interval 25 \
    --checkpoint_dir checkpoints_v13/diag_additive \
    --log_dir logs/v13/diag_additive_20260822 \
    "$@"
