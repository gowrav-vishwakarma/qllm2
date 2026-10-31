# v15 experiment index

Read before proposing anything. One row per idea; the write-up is the named
section of `EXPERIMENTS_V15.md`. Rerun a row from its *code at* hash in a
side worktree (`git worktree add ../qllm_v15_<id> <hash>`).

## Tools

| id | measures | status | run hash | log | rerun |
|---|---|---|---|---|---|
| G0 | selftest: parity, simplex, gradients, O(1), toy lookup | **PASS** 2026-10-31 (toy lookup 0.98, chance 0.125) | `8d60a55`, `84b41d8` | stdout | `.venv/bin/python -m v15.selftest` |
| G1 | micro: grove vs echo, B1 run 4 set (3 pairs) + B2 set (3 pairs); bars L, P (§4) | **FAIL as built** (pair 1 only, 02:34 IST 2026-10-31): grove at chance on every task (label_exact 0.15, label_category 0.25) vs echo 0.50 / 0.50; cause = season-start reads (blind window of C = 32 tokens; CPU toy: season 32 → 0.06, 8 → 0.99); pairs 2–6 not run | `84b41d8` | `logs/v14_v15g1p1_84b41d8_20261031_0146.{log,json}` | `V14_EXT_ARMS="grove=v15.bench:build_grove" TAG=v15g1p1 FAMILIES=rules ARMS=grove,echo EXTRA="--seed 21 --doc_seed0 400000000 --init_seed 21" bash v14/run_bench.sh` (code at `84b41d8`) |
| G1r | G1 rerun with ripening (within-season reads); same six pairs, same bars | **FAIL on L and P** (2 of 3 rules pairs missed, 04:44 IST 2026-10-31; pairs 3–6 not run). label_exact 2/4/8-shot: p1 0.34/0.22/0.12 vs echo 0.50/0.26/0.12, p2 0.35/0.21/0.10 vs 0.48/0.25/0.12; label_category: p1 0.33/0.42/0.25 vs 0.50/0.49/0.25, p2 0.33/0.47/0.27 vs 0.48/0.48/0.26. Rules learned but behind echo: letter_shift 8-shot 0.99 / 0.97 (echo 1.00), number_step 0.86 / 0.69 (1.00), reverse3 1.00 / 1.00. Guard FAIL: train loss 0.61 / 0.76 vs echo 0.38 / 0.46. 310–319 ms/step vs echo 176 | `abe4c89` | `logs/v14_v15g1rp1_abe4c89_20261031_0237.{log,json}`, `logs/v14_v15g1rp2_abe4c89_20261031_0327.{log,json}` | `V14_EXT_ARMS="grove=v15.bench:build_grove" TAG=v15g1rp1 FAMILIES=rules ARMS=grove,echo EXTRA="--seed 21 --doc_seed0 400000000 --init_seed 21" bash v14/run_bench.sh` (code at `abe4c89`) |
| G1d | diagnostic: grove arm of pair 1 with saved weights (SAVE=1), probed on CPU with `v15/tmp/probe_shares.py` (share entropy, address overlap same key vs other keys, learned τ/λ/α) | **DONE** 05:40 IST 2026-10-31: the traits did not learn (τ 9.8–10.6 from init 10.0 in every layer/grove, λ and α unchanged); label_exact 8-shot address overlap same key 0.78 vs other keys 0.79 (no binding), 2-shot 0.71 vs 0.56; slow groves (λ 0.74, 0.9) collapsed to a constant winner (entropy 0.1–0.3, overlap 1.0). Reference, the big-vocab toy that works (4096 keys, lr 3e-4): overlap 1.00 vs 0.13, entropy 0.4–0.7, τ 13.4. Verdict: in the multi-task bench the groves stayed a blurred hash; sharp addresses must be structural, not learned | `abe4c89` | `logs/v14_v15diag1_abe4c89_*` | `TAG=v15diag1 ARMS=grove SAVE=1 ...` same EXTRA as pair 1 |
| G2 | 150M-token holdout on the r600 recipe, Grove 102.96M vs M1 3.9225 (bar +0.05, tok/s ≥ 0.8× M1) | **PREPARED** (plumbing verified on CPU 2026-10-31; waits for G1 PASS + a whole-card slot) | — | `logs/v13_sempty_v15g2_<hash>_*.log` | `TAG=v15g2 ARCH=grove CHRONO=0 OUT_GATE=0 DELTA=0 TARGET_TOKENS=600000000 GRAD_CKPT=1 bash v13_sempty/tmp_pretrain_mix.sh` |

## Candidates

| id | idea | status | section |
|---|---|---|---|
| Grove organ | selection over accumulated evidence as the recurrence; roots and fruits grown in seasons | in code (`v15/grove.py`), G0 pass; G1 pre-registered | §1, §4 |

## Kept / Failed
None yet.

## Dropped before code
| id | idea | why | section |
|---|---|---|---|
| wheels | prime-period phases for time and numbers (CRT, exact rotations) | phases again, extends PAM; user direction 2026-10-31 | §5 |
