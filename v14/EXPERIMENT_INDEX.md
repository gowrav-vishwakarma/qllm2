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
AGENTS.md). All launchers refuse a dirty tree. `v14/run_bench.sh` replaced `v14/run_b0.sh` after `0671d65` (same env names, plus EXTRA, SAVE, WAIT_DONE); at older hashes use `run_b0.sh`.

## Benches (tools, not ideas)

| id | measures | status | run hash | log | rerun |
|---|---|---|---|---|---|
| B0 | memory wall: facts / bind / rebind, 8 candidates, chance 0.125 | run 1 **INVALID** (no easy rung: no arm learned lookup). Run 2 (sizes from 1) **running** on the 4090 since 01:45 IST 2026-10-30, arms echo, delta, n1sum, n1gate, n1phase, echo_tied (attempts 1-3 lost: a python segfault, an eval OOM from a missing `no_grad`, a restart to drop the closed N6 arms) | run 1 `a0f672a` | `logs/v14_b0_a0f672a_20261029_1232.log` | `TAG=b0 ARMS=echo,delta,n1sum,n1gate,n1phase,echo_tied SAVE=1 bash v14/run_bench.sh` (run 1: `run_b0.sh` at `a0f672a`) |
| B1 run 1 | rule induction in context (goal bar 3), shared generator `memory_probes/rules.py` | **done** (00:17 IST 2026-10-30): letter_shift / number_step / reverse3 at the ceiling (1.00, every arm, learned in weights); label_category and two_hop separate | `5b1eb9d` | `logs/v14_b1_5b1eb9d_20261029_2329.{log,json}` | `TAG=b1 FAMILIES=rules ARMS=echo,op,opdiag,n1phase,delta bash v14/run_b0.sh` |
| B1 run 3 | echo vs echo_tied, 3 paired seeds; trained with `label_marker` (format a); bars P (label_category) and A (label_marker_new: every item unseen) | **pre-registered, queued** behind B0 run 2 | — | `logs/v14_b1r3p<i>_<hash>_<time>.{log,json}` | see SCRATCHPAD (3 launches, `TAG=b1r3p<i>`) |
| B1 run 2 | the same, new seed, plus unseen rule parameters and the 2x2 cause arms | **done** (01:44 IST): N6 replication **not met**. Every arm is below chance on unseen rule parameters (selection among learned rules, not induction). **echo_tied** label_category 0.857 vs echo 0.471 (one seed; seed noise ≈ 0.2) | `85ecf02` | `logs/v14_b1r2_85ecf02_20261030_0057.{log,json}` | `TAG=b1r2 FAMILIES=rules ARMS=echo,op,opdiag,opdiag_ownq,echo_tied SAVE=1 EXTRA='--seed 7 --doc_seed0 50000000 --init_seed 1' bash v14/run_bench.sh` |

## Candidates in code (not yet judged)

| id | idea | status | notebook section | flag / micro-bench arm |
|---|---|---|---|---|
| N1 | K delta notebooks whose reads interfere by rotation (v11 E3 on the lean body) | selftest PASS; arm `n1phase` in B1, all three reads in B0 run 2 | "N1" | `phase_states=2 phase_read=sum\|gate\|phase`; arms `n1sum`, `n1gate`, `n1phase` |

| echo_tied | echo with one projection for query and key (W_q = W_k): a query lands on keys of hidden states like its own (similarity lookup) | B1 run 2, exploratory: label_category 0.92 / 0.92 / 0.74 vs echo 0.61 / 0.54 / 0.27 (one seed; seed noise ≈ 0.2). In B0 run 2 (memory cost). Needs ≥ 3 seeds (B1 run 3) | "B1" run 2 | `echo_tied_qk=True`; arm `echo_tied` |

Selftests: `.venv/bin/python -m v14.selftest` (CPU, minutes) and
`.venv/bin/python -m v14.check_layout`.

## Kept

None yet.

## Failed and removed (code at the listed commit)

| id | idea | verdict / key number | notebook section | run hash | code at | rerun (at *code at*) |
|---|---|---|---|---|---|---|
| N6 | operator memory: the state is an inferred per-channel rotation (circular mean of b·conj(a), a = W_in h_{t-1}, b = W_role ê(x_t)), applied to the next input q = W_in h_t | **FAIL** (B1 run 2). Replication not met: label_category mean op 0.427, opdiag 0.587 vs echo 0.471, need +0.15 (run 1's 1.00 was seed luck). Unseen rule parameters: op 0.037 vs echo 0.063 (all below chance). Rotation never beat the plain diagonal | "N6" | `85ecf02` (run 2), `5b1eb9d` (run 1) | `78e4354` | `TAG=b1r2 FAMILIES=rules ARMS=echo,op,opdiag,opdiag_ownq SAVE=1 EXTRA='--seed 7 --doc_seed0 50000000 --init_seed 1' bash v14/run_bench.sh` |

## Folded or not built

| id | idea | state | notebook section |
|---|---|---|---|
| N2 | phase-modulated activation (magnitude sets phase), complex arm only | planned; flag already in v13_sempty (`activation='phase_mod'`) | "N2" |
| N3 | formula-weight neuron `W(x)x` | **folded into N2**: `U·(c(x)⊙Vᵀx)` is a GLU-gated bottleneck, nothing new to test | "N3" |
| N4 | living neuron: per-unit fast weight, written by surprise | design only | "N4" |
| N5 | resonant binding: oscillator phases as binding tags | design only | "N5" |
