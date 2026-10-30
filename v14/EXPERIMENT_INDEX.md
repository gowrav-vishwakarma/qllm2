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
| B0 | memory wall: facts / bind / rebind, 8 candidates, chance 0.125 | **CLOSED** (stop rule, 10:19 IST 2026-10-30): run 1 invalid (no easy rung); run 2 invalid (echo facts2 0.55); run 3 (aux 0.1 + tpp reference) invalid: echo facts2 0.52 AND tpp facts1 0.17 (attention does not learn even one fact): the bench is broken at 27.6M / 6000 steps. No run 4 without a new design | `a0f672a`, `bcfd88a`, `bf59754` | `logs/v14_b0_a0f672a_20261029_1232.log`, `logs/v14_b0_bcfd88a_20261030_0145.{log,json}`, `logs/v14_b0r3_bf59754_20261030_0851.{log,json}` | run 3: `TAG=b0r3 ARMS=echo,n1sum,n1gate,n1phase,tpp AUX=0.1 SAVE=1 bash v14/run_bench.sh` (at `bf59754`) |
| B1 run 1 | rule induction in context (goal bar 3), shared generator `memory_probes/rules.py` | **done** (00:17 IST 2026-10-30): letter_shift / number_step / reverse3 at the ceiling (1.00, every arm, learned in weights); label_category and two_hop separate | `5b1eb9d` | `logs/v14_b1_5b1eb9d_20261029_2329.{log,json}` | `TAG=b1 FAMILIES=rules ARMS=echo,op,opdiag,n1phase,delta bash v14/run_b0.sh` |
| B1 run 2 | the same, new seed, plus unseen rule parameters and the 2x2 cause arms | **done** (01:44 IST): N6 replication **not met**. Every arm is below chance on unseen rule parameters (selection among learned rules, not induction). **echo_tied** label_category 0.857 vs echo 0.471 (one seed; seed noise ≈ 0.2) | `85ecf02` | `logs/v14_b1r2_85ecf02_20261030_0057.{log,json}` | `TAG=b1r2 FAMILIES=rules ARMS=echo,op,opdiag,opdiag_ownq,echo_tied SAVE=1 EXTRA='--seed 7 --doc_seed0 50000000 --init_seed 1' bash v14/run_bench.sh` |
| B1 run 3 | echo vs echo_tied, 3 paired seeds; trained with `label_marker` (format a); bars P (label_category) and A (label_marker_new: every item unseen) | **done** (04:37 IST 2026-10-30): **P INVALID** (echo 0.999 / 0.946 in pairs 1 and 3; d +0.00 / −0.07 / −0.54). **A NOT met** (d −0.12 / +0.01 / −0.16: tied worse on unseen items). The lookup circuit forms per seed or not: echo 2 of 3 seeds, tied 1 of 3 | `44df201` | `logs/v14_b1r3p{1,2,3}_44df201_20261030_{0332,0354,0415}.{log,json}` | `for i in 1 2 3; do TAG=b1r3p$i FAMILIES=rules ARMS=echo,echo_tied SAVE=1 EXTRA="--seed 1$i --doc_seed0 ${i}00000000 --init_seed 1$i --rule_tasks letter_shift,number_step,reverse3,label_category,two_hop,label_marker" bash v14/run_bench.sh; done` |
| B1 run 4 | the clean P test: run 2's training set (no label_marker), echo vs echo_tied, 3 new seed pairs (21/22/23), P's frozen rule | **done** (05:40 IST 2026-10-30): **P MET**: d +0.589 / +0.595 / +0.222 (mean +0.469 ≥ 2·SD 0.427); echo 0.41–0.44 (in-context chance), tied 1.00 / 1.00 / 0.66. Gate for the v13 session's E2 600M arm: **GO** | `d4e0f0d` | `logs/v14_b1r4p{1,2,3}_d4e0f0d_20261030_{0443,0502,0521}.{log,json}` | `for i in 1 2 3; do TAG=b1r4p$i FAMILIES=rules ARMS=echo,echo_tied SAVE=1 EXTRA="--seed 2$i --doc_seed0 $((i+3))00000000 --init_seed 2$i" bash v14/run_bench.sh; done` |
| B2 | bar 3: inference beats selection. Many trained rule parameters, a few held out inside the range (shift_wide / step_wide / affine_map and their `_new`); the nearest-parameter decoys are always candidates | run 1 **done** (6000, 09:54 IST, one seed): L and I met (tpp trained 0.999, held out 0.666): **valid**. Held out: echo 0.739, n1sum 0.731, tpp 0.666; numbers ~0.9–0.99 for all; **shift_wide_new: echo 0.33 (rising 0.20 → 0.47 with shots), tpp 0.02 (picks the neighbour)**. Run 2 (echo vs tpp, 3 seed pairs, bar R) pre-registered | `bf59754` | `logs/v14_b2_bf59754_20261030_0357.{log,json}` | `TAG=b2 FAMILIES=rules ARMS=tpp,echo,n1sum SAVE=1 EXTRA='--seed 31 --doc_seed0 700000000 --init_seed 31 --rule_tasks shift_wide,step_wide,affine_map' bash v14/run_bench.sh` |
| B1 tpp | calibration: tpp on B1 run 4's data, 3 seeds | **done** (6000): label_category 0.42 / 0.42 / 0.43 = chance (attention does not form the label circuit at 27M); number_step_ood 0.27–0.38 (echo 0.09–0.23); letter_shift_ood 0 | `bf59754` | `logs/v14_b1r4tpp{1,2,3}_bf59754_*.{log,json}` | `for i in 1 2 3; do TAG=b1r4tpp$i FAMILIES=rules ARMS=tpp SAVE=1 EXTRA="--seed 2$i --doc_seed0 $((i+3))00000000 --init_seed 2$i" bash v14/run_bench.sh; done` |
| B2 run 2 | replication: echo vs tpp, 3 seed pairs (seeds 41/42/43, doc_seed0 (i+7)·100M); bar R on shift_wide_new (mean d ≥ +0.15 and ≥ 2·SD) | **done** (12:28 IST 2026-10-30): **R MET**: d +0.123 / +0.234 / +0.333 (mean +0.230, 2·SD 0.210); echo rises with shots in every pair, tpp ≤ 0.03 | `9e0a3fd` | `logs/v14_b2r2p{1,2,3}_9e0a3fd_20261030_*.{log,json}` | `for i in 1 2 3; do TAG=b2r2p$i FAMILIES=rules ARMS=echo,tpp SAVE=1 EXTRA="--seed 4$i --doc_seed0 $((i+7))00000000 --init_seed 4$i --rule_tasks shift_wide,step_wide,affine_map" bash v14/run_bench.sh; done` |
| B2 state sweep | memory size at equal params (echo@32x12 / echo / echo@128x3), 2 seeds; bar S on shift_wide_new | **done** (13:34 IST): S **refuted by the frozen rule** (hd128 − hd32 +0.168) but seeds order the sizes oppositely: no reliable size effect; bottleneck hypothesis not supported. **The O(1) body solves held-out shifts fully in some seeds (1.000, 0.997)**: 3 of 10 echo-family runs ≥ 0.9, tpp 0 of 4 | `dd5b174` | `logs/v14_b2state{1,2}_dd5b174_*.{log,json}` | `for j in 1 2; do TAG=b2state$j FAMILIES=rules ARMS=echo@32x12,echo,echo@128x3 SAVE=1 EXTRA="--seed 5$j --doc_seed0 $((j+10))00000000 --init_seed 5$j --rule_tasks shift_wide,step_wide,affine_map" bash v14/run_bench.sh; done` |
| F1 | formation bench: B1 run 4's training set, label_category formed (≥ 0.9) over 5 seeds; arms echo, echo_phase (N9a), echo_qinit (v13 Q1), echo_sparse (v13 S1); bar F: ≥ 4/5 where echo ≤ 2/5 | **done**: **echo_phase (N9a) 5/5 formed (1.000 each), echo 1/5, echo_qinit (Q1) 0/5, echo_sparse (S1) 0/5 (v13 slot, `38e3c29`): bar F MET for N9a only**; N9a goes to 600M (v13, trainer hook `424afd6`) | `7c8c6e2` | `logs/v14_f1p{1..5}_7c8c6e2_*.{log,json}` | `for i in 1 2 3 4 5; do TAG=f1p$i FAMILIES=rules ARMS=echo,echo_phase SAVE=1 EXTRA="--seed 7$i --doc_seed0 $((i+15))00000000 --init_seed 7$i" bash v14/run_bench.sh; done` |

## Candidates in code (not yet judged)

None live (2026-10-30): N6, echo_tied and N1 failed and were removed. Reference arm `tpp` (Transformer++) stays in the micro-bench.

| id | idea | status | notebook section | flag / micro-bench arm |
|---|---|---|---|---|


Selftests: `.venv/bin/python -m v14.selftest` (CPU, minutes) and
`.venv/bin/python -m v14.check_layout`.

## Kept

None yet.

## Failed and removed (code at the listed commit)

| id | idea | verdict / key number | notebook section | run hash | code at | rerun (at *code at*) |
|---|---|---|---|---|---|---|
| N6 | operator memory: the state is an inferred per-channel rotation (circular mean of b·conj(a), a = W_in h_{t-1}, b = W_role ê(x_t)), applied to the next input q = W_in h_t | **FAIL** (B1 run 2). Replication not met: label_category mean op 0.427, opdiag 0.587 vs echo 0.471, need +0.15 (run 1's 1.00 was seed luck). Unseen rule parameters: op 0.037 vs echo 0.063 (all below chance). Rotation never beat the plain diagonal | "N6" | `85ecf02` (run 2), `5b1eb9d` (run 1) | `78e4354` | `TAG=b1r2 FAMILIES=rules ARMS=echo,op,opdiag,opdiag_ownq SAVE=1 EXTRA='--seed 7 --doc_seed0 50000000 --init_seed 1' bash v14/run_bench.sh` |
| echo_tied | echo with one projection for query and key (W_q = W_k) | **FAIL (stands)**: v13's box control cleared the 4090 (`38e3c29`), so E2's loss is not a box effect (v13 E2, 600M, M1 body, C1-ctl recipe, 2026-10-30): label_category 8-shot 0.365 vs C1-ctl 0.500 (d −0.135, bar +0.15); holdout 3.5298 vs 3.4743 (+0.055); recall 0.602 vs 0.825. The micro-bench effect (B1 run 4 P met: circuit in 4 of 4 seeds vs 0 of 5) did not transfer to real text | "B1" runs 2–4; v13 EXPERIMENTS_SEMPY "E2" | `d4e0f0d` (B1 run 4), `85ecf02` (run 2) | `5100fd7` (v13 flag: `26ef94a`) | `TAG=b1r4p1 FAMILIES=rules ARMS=echo,echo_tied SAVE=1 EXTRA="--seed 21 --doc_seed0 400000000 --init_seed 21" bash v14/run_bench.sh` |
| N1 | K delta notebooks whose reads interfere by rotation (v11 E3 on the lean body); arms n1sum / n1gate / n1phase | **FAIL** (B0 run 3 stop rule, 2026-10-30): mean(facts2, facts4) @1024 n1phase 0.370 < n1sum 0.380; turning never helped in B0 runs 2–3; n1sum's run-2 partial lookup (facts2 0.79) did not reappear (0.49); B2 run 1 n1sum 0.731 vs echo 0.739 | "N1", "B0" | `bcfd88a` (B0 run 2), `bf59754` (B0 run 3) | `9e0a3fd` | `TAG=b0r3 ARMS=echo,n1sum,n1gate,n1phase,tpp AUX=0.1 SAVE=1 bash v14/run_bench.sh` |

## Folded or not built

| id | idea | state | notebook section |
|---|---|---|---|
| N2 | phase-modulated activation (magnitude sets phase), complex arm only | planned; flag already in v13_sempty (`activation='phase_mod'`) | "N2" |
| N3 | formula-weight neuron `W(x)x` | **folded into N2**: `U·(c(x)⊙Vᵀx)` is a GLU-gated bottleneck, nothing new to test | "N3" |
| N7 | in-context n-gram memory: bounded hash table of (last n tokens -> next token), written while reading, read as an extra input; exact lookup that does not fade (user idea) | **in code** (`83f49dc`, selftest PASS); micro run 1 (echo vs echo_n7 vs tpp, 3 pairs, bar X on label_exact_new, guard G on B2) pre-registered | "N7" | `ngram_orders=(2,3)`; arm `echo_n7` |
| N9 | nested 2×O(1) memory (user): step 1 content-phase keys (N9a), step 2 settling; N9b (LSH over embeddings) dropped before code (no fuzzy separation in raw embeddings) | **N9a in code** (selftest PASS: template keys |cos| 1.0 -> 0.09); **F1: forms the label lookup circuit in 5/5 seeds vs echo 1/5 (bar F MET)**; next: 600M check (v13) | "N9" | `echo_phase_lags=3`; arm `echo_phase` |
| N10 | rules as phase continuation: one learned coordinate per head on several circles of spread periods, a rule = one step read from the examples (N6's mechanism + the user's circles) | **in code** (branch n10, selftest PASS; N6 restored as arm `op`); formation run (echo / op / opcoord × 5 seeds on B2, bar C) pre-registered, queued after F1 | "N10" | `opmem='coord'`; arms `opcoord`, `op` |
| N11 | reason first, then look: a planner forms the lookup query, then reads a parametric knowledge store (A4-like) and the conversation memory (N7/N9) | design only (2026-10-30) | "N11" |
| N8 | self-opened memory: write strength set by the body's own maturity (prediction confidence) instead of a fixed two-stage cut (handed over by v13) | design only (2026-10-30) | "N8" |
| N4 | living neuron: per-unit fast weight, written by surprise | design only | "N4" |
| N5 | resonant binding: oscillator phases as binding tags | design only | "N5" |
