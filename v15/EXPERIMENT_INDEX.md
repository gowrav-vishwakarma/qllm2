# v15 experiment index

Read before proposing anything. One row per idea; the write-up is the named
section of `EXPERIMENTS_V15.md`. Rerun a row from its *code at* hash in a
side worktree (`git worktree add ../qllm_v15_<id> <hash>`).

## Tools

| id | measures | status | run hash | log | rerun |
|---|---|---|---|---|---|
| G0 | selftest: parity, simplex, gradients, O(1), toy lookup | **PASS** 2026-10-31 (toy lookup 0.98, chance 0.125) | `8d60a55`, `84b41d8` | stdout | `.venv/bin/python -m v15.selftest` |
| G1 | micro: grove vs echo, B1 run 4 set (3 pairs) + B2 set (3 pairs); bars L, P (§4) | **FAIL as built** (pair 1 only, 02:34 IST 2026-10-31): grove at chance on every task (label_exact 0.15, label_category 0.25) vs echo 0.50 / 0.50; cause = season-start reads (blind window of C = 32 tokens; CPU toy: season 32 → 0.06, 8 → 0.99); pairs 2–6 not run | `84b41d8` | `logs/v14_v15g1p1_84b41d8_20261031_0146.{log,json}` | `V14_EXT_ARMS="grove=v15.bench:build_grove" TAG=v15g1p1 FAMILIES=rules ARMS=grove,echo EXTRA="--seed 21 --doc_seed0 400000000 --init_seed 21" bash v14/run_bench.sh` (code at `84b41d8`) |
| G1r | G1 rerun with ripening (within-season reads); same six pairs, same bars | **RUNNING** since 02:35 IST 2026-10-31 (tmux `v15g1r`, worktree `../qllm_v15_g1r`) | `abe4c89` | `logs/v14_v15g1rp{1,2,3}_abe4c89_*`, `v14_v15g1rb2p*` | same command at `abe4c89`, TAG v15g1rp1 |
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
