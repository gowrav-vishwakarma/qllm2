# v15 experiment index

Read before proposing anything. One row per idea; the write-up is the named
section of `EXPERIMENTS_V15.md`. Rerun a row from its *code at* hash in a
side worktree (`git worktree add ../qllm_v15_<id> <hash>`).

## Tools

| id | measures | status | run hash | log | rerun |
|---|---|---|---|---|---|
| G0 | selftest: parity, simplex, gradients, O(1), toy lookup | **PASS** 2026-10-31 (toy lookup 0.98, chance 0.125) | (this commit) | stdout | `.venv/bin/python -m v15.selftest` |

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
