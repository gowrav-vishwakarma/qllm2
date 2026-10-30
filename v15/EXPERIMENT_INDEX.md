# v15 experiment index

Read before proposing anything. One row per idea; the write-up is the named
section of `EXPERIMENTS_V15.md`. Rerun a row from its *code at* hash in a
side worktree (`git worktree add ../qllm_v15_<id> <hash>`).

## Tools

| id | measures | status | run hash | log | rerun |
|---|---|---|---|---|---|
| G0 | selftest: parity, simplex, gradients, O(1), toy lookup | **PASS** 2026-10-31 (toy lookup 0.98, chance 0.125) | `8d60a55`, `84b41d8` | stdout | `.venv/bin/python -m v15.selftest` |
| G1 | micro: grove vs echo, B1 run 4 set (3 pairs) + B2 set (3 pairs); bars L, P (§4) | **QUEUED** 00:12 IST 2026-10-31 (tmux `v15g1`, behind n9c) | `84b41d8` | `logs/v14_v15g1p{1,2,3}_84b41d8_*`, `v14_v15g1b2p*` | `V14_EXT_ARMS="grove=v15.bench:build_grove" TAG=v15g1p1 FAMILIES=rules ARMS=grove,echo EXTRA="--seed 21 --doc_seed0 400000000 --init_seed 21" bash v14/run_bench.sh` |

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
