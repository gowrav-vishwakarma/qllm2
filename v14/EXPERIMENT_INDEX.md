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
| B0 | memory wall: facts / bind / rebind, 8 candidates, chance 0.125 | run 1 **INVALID** (no easy rung). Run 2 (sizes from 1, 01:45–03:31 IST 2026-10-30) **INVALID**: echo facts1 1.00 but facts2 @1024 0.55 (bar 0.90); every echo arm sits at 1/n (picks a value of the document, not the bound one). Descriptive: n1sum / n1gate partial lookup (facts2 0.79 / 0.78), n1phase back at echo's level (0.51); echo_tied no memory cost (0.50 vs 0.56). Run 3 (`--aux_weight 0.1`, plus the tpp attention reference arm, with a stop rule) **pre-registered, queued** after B1 run 4 | run 1 `a0f672a`, run 2 `bcfd88a` | `logs/v14_b0_a0f672a_20261029_1232.log`, `logs/v14_b0_bcfd88a_20261030_0145.{log,json}` | run 2: `TAG=b0 ARMS=echo,delta,n1sum,n1gate,n1phase,echo_tied SAVE=1 bash v14/run_bench.sh`; run 3: `TAG=b0r3 ARMS=echo,n1sum,n1gate,n1phase,echo_tied,tpp AUX=0.1 SAVE=1 bash v14/run_bench.sh` (run 1: `run_b0.sh` at `a0f672a`) |
| B1 run 1 | rule induction in context (goal bar 3), shared generator `memory_probes/rules.py` | **done** (00:17 IST 2026-10-30): letter_shift / number_step / reverse3 at the ceiling (1.00, every arm, learned in weights); label_category and two_hop separate | `5b1eb9d` | `logs/v14_b1_5b1eb9d_20261029_2329.{log,json}` | `TAG=b1 FAMILIES=rules ARMS=echo,op,opdiag,n1phase,delta bash v14/run_b0.sh` |
| B1 run 2 | the same, new seed, plus unseen rule parameters and the 2x2 cause arms | **done** (01:44 IST): N6 replication **not met**. Every arm is below chance on unseen rule parameters (selection among learned rules, not induction). **echo_tied** label_category 0.857 vs echo 0.471 (one seed; seed noise ≈ 0.2) | `85ecf02` | `logs/v14_b1r2_85ecf02_20261030_0057.{log,json}` | `TAG=b1r2 FAMILIES=rules ARMS=echo,op,opdiag,opdiag_ownq,echo_tied SAVE=1 EXTRA='--seed 7 --doc_seed0 50000000 --init_seed 1' bash v14/run_bench.sh` |
| B1 run 3 | echo vs echo_tied, 3 paired seeds; trained with `label_marker` (format a); bars P (label_category) and A (label_marker_new: every item unseen) | **done** (04:37 IST 2026-10-30): **P INVALID** (echo 0.999 / 0.946 in pairs 1 and 3; d +0.00 / −0.07 / −0.54). **A NOT met** (d −0.12 / +0.01 / −0.16: tied worse on unseen items). The lookup circuit forms per seed or not: echo 2 of 3 seeds, tied 1 of 3 | `44df201` | `logs/v14_b1r3p{1,2,3}_44df201_20261030_{0332,0354,0415}.{log,json}` | `for i in 1 2 3; do TAG=b1r3p$i FAMILIES=rules ARMS=echo,echo_tied SAVE=1 EXTRA="--seed 1$i --doc_seed0 ${i}00000000 --init_seed 1$i --rule_tasks letter_shift,number_step,reverse3,label_category,two_hop,label_marker" bash v14/run_bench.sh; done` |
| B1 run 4 | the clean P test: run 2's training set (no label_marker), echo vs echo_tied, 3 new seed pairs (21/22/23), P's frozen rule | **done** (05:40 IST 2026-10-30): **P MET**: d +0.589 / +0.595 / +0.222 (mean +0.469 ≥ 2·SD 0.427); echo 0.41–0.44 (in-context chance), tied 1.00 / 1.00 / 0.66. Gate for the v13 session's E2 600M arm: **GO** | `d4e0f0d` | `logs/v14_b1r4p{1,2,3}_d4e0f0d_20261030_{0443,0502,0521}.{log,json}` | `for i in 1 2 3; do TAG=b1r4p$i FAMILIES=rules ARMS=echo,echo_tied SAVE=1 EXTRA="--seed 2$i --doc_seed0 $((i+3))00000000 --init_seed 2$i" bash v14/run_bench.sh; done` |

## Candidates in code (not yet judged)

| id | idea | status | notebook section | flag / micro-bench arm |
|---|---|---|---|---|
| N1 | K delta notebooks whose reads interfere by rotation (v11 E3 on the lean body) | selftest PASS. B1 run 1 arm `n1phase` (ceiling tasks only). B0 run 2 invalid, so no judgement; descriptively turning cancelled the second notebook's lookup gain (n1phase 0.51 vs n1sum 0.79 on facts2). Decided by B0 run 3 and its stop rule | "N1" | `phase_states=2 phase_read=sum\|gate\|phase`; arms `n1sum`, `n1gate`, `n1phase` |

| echo_tied | echo with one projection for query and key (W_q = W_k): a query lands on keys of hidden states like its own | **B1 P MET (run 4)**: on run 2's training set the tie makes the label_category lookup circuit form (tied 4 of 4 seeds, echo 0 of 5). With label_marker in training (run 3) it does not (tied 1 of 3, echo 2 of 3) and A is not met (tied −0.09 on unseen items): not a similarity lookup. No memory cost (B0 run 2); 9 % faster. **Next: LM check = v13 E2** (600M, M1 body, vs C1-ctl) | "B1" runs 2–4 | `echo_tied_qk=True`; arm `echo_tied` |

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
