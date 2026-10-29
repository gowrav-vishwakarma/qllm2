# v14 experiment index: what was tried, verdict, where the code lives

**Read this before proposing an experiment**, so nothing runs twice. There is
one row per idea. The full write-up (maths, bars, numbers) is the named
section of `EXPERIMENTS_V14.md`. What v4–v13 already settled is in
`EXPERIMENTS_V14.md` Part A and `v13_sempty/EXPERIMENT_INDEX.md`.

**Update habit (user rule, 2026-10-29).** Change the row when the state
changes: queued → running → result → verdict. Commit it with the log behind
the number.

**Rerun any row** from its *code at* hash, in a side checkout so master is
untouched:

```bash
git worktree add ../qllm_v14_<id> <code-at-hash>
cd ../qllm_v14_<id> && <command in the row>
```

v14 needs sempyt ≥ `b99e038` (`rotate` e403dd8, chunked per-step scan
2790c69, pair-last guard b99e038). Wire it into the venv with a `.pth` (see
AGENTS.md). All launchers refuse a dirty tree.

## Benches (tools, not ideas)

| id | measures | status | run hash | log | rerun |
|---|---|---|---|---|---|
| B0 | memory wall: facts / bind / rebind, 8 candidates, chance 0.125 | run 1 **INVALID** (no easy rung: no arm learned lookup). Run 2 (sizes from 1) queued on the 4090 behind B1 | run 1 `a0f672a` | `logs/v14_b0_a0f672a_20261029_1232.log` | `ARMS=echo,delta,n1sum,n1gate,n1phase,op TAG=b0 bash v14/run_b0.sh` |
| B1 | rule induction in context (goal bar 3), shared generator `memory_probes/rules.py` | run 1 **running** on the 4090 since 23:29 IST 2026-10-29 | `5b1eb9d` | `logs/v14_b1_5b1eb9d_20261029_2329.log` | `TAG=b1 FAMILIES=rules ARMS=echo,op,opdiag,n1phase,delta bash v14/run_b0.sh` |

## Candidates in code (not yet judged)

| id | idea | status | notebook section | flag / micro-bench arm |
|---|---|---|---|---|
| N1 | K delta notebooks whose reads interfere by rotation (v11 E3 on the lean body) | selftest PASS; arm `n1phase` in B1, all three reads in B0 run 2 | "N1" | `phase_states=2 phase_read=sum\|gate\|phase`; arms `n1sum`, `n1gate`, `n1phase` |
| N6 | operator memory: the state is an inferred per-channel rotation, applied to the next input | selftest PASS (applies an inferred rotation, cos 1.0000); first real test in B1 | "N6" | `opmem=phase\|diag` (needs echo); arms `op`, `opdiag` |

Selftests: `.venv/bin/python -m v14.selftest` (CPU, minutes) and
`.venv/bin/python -m v14.check_layout`.

## Kept

None yet.

## Failed and removed (code at the listed commit)

None yet.

## Folded or not built

| id | idea | state | notebook section |
|---|---|---|---|
| N2 | phase-modulated activation (magnitude sets phase), complex arm only | planned; flag already in v13_sempty (`activation='phase_mod'`) | "N2" |
| N3 | formula-weight neuron `W(x)x` | **folded into N2**: `U·(c(x)⊙Vᵀx)` is a GLU-gated bottleneck, nothing new to test | "N3" |
| N4 | living neuron: per-unit fast weight, written by surprise | design only | "N4" |
| N5 | resonant binding: oscillator phases as binding tags | design only | "N5" |
