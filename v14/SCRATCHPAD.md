# v14 handover

Read `README.md` (rules), `EXPERIMENT_INDEX.md` (one row per idea) and
`EXPERIMENTS_V14.md` (do-not-repeat digest and pre-registered bars) first.
Keep all three current as things happen (user rule, 2026-10-29).

## State (2026-10-30 07:00 IST)

* **4090: the v13 session's E2 since 05:43 IST** (tmux `e2`, run from the
  worktree `../qllm_e2` at `e5b1a2a`, ECHO_TIED code `26ef94a`; two stages,
  ETA ~10:15 IST plus probes).
  1. **B0 run 2 DONE (03:31 IST, `bcfd88a`): INVALID.** Echo facts2 @1024
     0.55 (bar 0.90): every echo arm picks a value of the document, not the
     bound one. Descriptive: n1sum / n1gate do partial lookup (facts2 0.79),
     n1phase falls back to echo's level (turning cancels the second
     notebook); echo_tied costs no memory. **B0 run 3** (`--aux_weight 0.1`,
     arms echo, n1sum, n1gate, n1phase, echo_tied + the **tpp** attention
     reference, branch `v14dev`, GPU smoke passed; validity and
     stop rule in EXPERIMENTS_V14 "B0") is pre-registered and goes after run 4, or after
     E2 if the v13 session takes the 4090.
  2. **B1 run 3 DONE (04:37 IST, `44df201`).** **P INVALID** (echo at the ceiling in pairs 1
     and 3; where the arms differ, tied is worse: d −0.07, −0.54). **A NOT
     met** (d −0.12, +0.01, −0.16: tied is worse on unseen items). The
     lookup circuit forms per seed or not: echo in 2 of 3 seeds, tied in 1
     of 3. Run 2's lead was a seed where only tied formed it.
  3. **B1 run 4 DONE (05:40 IST, `d4e0f0d`): P MET.** d +0.589 / +0.595 /
     +0.222 (mean +0.469, 2·SD 0.427); echo 0.41–0.44, tied 1.00 / 1.00 /
     0.66. On run 2's training set the tie makes the lookup circuit form
     (tied 4 of 4 seeds, echo 0 of 5). With label_marker in training it does
     not (run 3). **E2 is GO** (v13 session, 600M tied echo vs C1-ctl, M1
     body, on the 4090).
* **Branch `v14dev` merged** (`a0259ca`: the tpp reference arm, `c9bff5c`
  + `51a94cd`); v14 selftest ALL PASS and layout OK on the merged tree.
* **Shared checkout rule (2026-10-30):** both sessions work in
  `/home/gowrav/Development/qllm2`. While a bench runs, nobody commits to
  master or leaves tracked edits: each pair launch checks for a clean tree,
  `--resume` refuses a moved HEAD, and each pair re-imports v13_sempty. The
  v13 session keeps its E2 code and notes on branch `e2` (worktree
  `../qllm_e2`) and rebases it onto master after run 4.
* **Launcher:** `v14/run_bench.sh` (replaced `run_b0.sh`). It also stops waiting when the
  other run's log prints its DONE line (`WAIT_DONE=file:pattern`): the v13
  T1 tmux stayed open two hours after its run ended on 2026-10-29.
* **6000:** the v13 session's chain. tfs1 (Transformer++ one stage, started
  03:15 IST at `b788731`) -> tfs2 -> c2m2 (ETA ~08:00 IST) -> tfc1
  (Transformer++ on c1m2's aptitude recipe, tmux `s3chain`, ETA ~09:30 IST).
  Their E2 (tied echo, 600M, M1 body, C1-ctl's exact recipe, paired against
  C1-ctl; pre-registered in EXPERIMENTS_SEMPY "E2" on branch `e2`) is gated on
  **B1 run 4's P** (run 3's P was invalid). If P passes, E2 runs on the 4090
  from ~05:45 IST.
* **v13 tfs1 (Transformer++ one stage, 600M, r600, `b788731`; recorded on
  branch `e2` 174c2b8):** holdout 3.4885 (M2 3.4697 beats it, M1 ties
  3.4906), WikiText 66.4 (M2 68.8), recall 0.645 (0.84 inside 2048, 0.15 at
  8192), learning probe 0.217. **label_category 2/4/8-shot: T++
  0.69/0.84/0.73** vs M1 0.65/0.67/0.29, M2 0.59/0.61/0.31, two-stage M1
  0.62/0.71/0.55. Attention does the in-context label lookup that the O(1)
  bodies do not: **this is the gap for v14 to close** (B1's `tpp` arm
  measures the same gap at 27M).
* **v13 C1 on M2 (`fed73cf`): KEEP, best 600M LM so far.** Holdout 3.4473
  (M2 3.4697, C1-ctl 3.4743), WikiText 65.74, recall 0.870, learning probe
  0.187 (flat). **label_category 8-shot does NOT lift on M2:** c1m2 0.38,
  M2 one-stage 0.31, C1-ctl (M1, two stages) 0.55. So the two-stage lift
  depends on the body (shared reads hide it). Compare label_category with
  M1-body numbers, or within one body (an M2 arm against c1m2's 0.38).
* **v13 results this evening (600M, r600 protocol):** M2 shared reads KEEP
  (holdout 3.4697 vs M1 3.4906; recall 0.522 vs 0.615; 1.18x slower). T1
  thought stream FAIL (3.4977 vs echo 3.4885). C1 aptitude curriculum
  primary FAIL (learning probe 0.199 vs 0.175), but recall jumped (63-cell
  0.907 vs 0.615) with fewer recall docs. Their guess: the multiple-choice
  "Answer: B) text" format acts as recall practice. Learning probe: every
  model is near chance (Mamba-2 3B 0.202). **Bar 3 is open everywhere.**
* **Control body for LM checks: M1** (EXPERIMENTS_V14 Part B: faster, reads
  not shared). Compare a passing unit with M2's 600M numbers before a 6000
  slot.
* **Data for the N6 LM check (if B1 passes):** the v13 session's real
  aptitude blend, `v7/aptitude.py` (`138cddb`). It uses the train splits of
  14 reasoning datasets (AQuA-RAT, GSM8K, LogiQA2, ReClor, RACE, ARC, CSQA,
  OBQA, SciQ, QASC, BoolQ, StrategyQA, WinoGrande, HellaSwag), ~48.8M tok,
  source `aptitude` in load_pretrain_mix (C1 stage-A weights
  dclm,fineweb,smoltalk2_mid,aptitude = 0.25,0.20,0.15,0.40,
  blend_warmup_tokens 30M). The test/validation splits are untouched, so
  they also work as a real-data reasoning eval.
* **v13 C1-ctl verdict (`237173a`):** most of C1's recall jump, and its
  label_category gain, come from the **two-stage schedule** (language first,
  recall docs later, LR re-warmed), not the aptitude data or its
  multiple-choice format. The ctl beats one-stage M1 on holdout (3.4743 vs
  3.4906), recall (0.825 vs 0.615) and label_category 8-shot (0.55 vs
  0.29). So the multiple-choice B0 family is dropped. For stage 3, both arms
  use the same schedule shape. Next on the 6000 (tmux `s2chain`, after C1
  on M2, ~03:30-08:00 IST): Transformer++ in one stage vs two, then two
  stages on M2.
* **Session coordination:** v14 (this session, "Qllm v14") and v13_sempty
  ("Qllm v13") ping each other on every GPU start/stop and on every verdict.
* **sempyt:** v14 needs ≥ `b99e038`. Pull it on the 6000 before running
  v14 there.
* **Git:** origin = `qllm-private-personal` on both boxes. Commit → push
  origin → pull on the 6000.

## Next action

1. **GPU plan (IST), agreed with v13 at ~07:00.** E2 (4090) ends ~09:15
   with probes (59K tok/s, faster than planned).
   - **6000, after the v13 s3 ping (~09:30 -> ~10:40):** `cd
     ~/Development/qllm-private && git pull` (needs ≥ `1e9a0b8`; sempyt ≥
     `b99e038`), a 30-step smoke of tpp + one B2 task, then in one tmux:
     (a) **tpp B1 calibration** on run 4's data (`for i in 1 2 3; do
     TAG=b1r4tpp$i FAMILIES=rules ARMS=tpp SAVE=1 EXTRA="--seed 2$i
     --doc_seed0 $((i+3))00000000 --init_seed 2$i" bash v14/run_bench.sh;
     done`): does attention form the label_category circuit in every seed?
     (b) **B2 run 1** (`TAG=b2 FAMILIES=rules ARMS=tpp,echo,echo_tied,n1sum
     SAVE=1 EXTRA='--seed 31 --doc_seed0 700000000 --init_seed 31
     --rule_tasks shift_wide,step_wide,affine_map' bash v14/run_bench.sh`;
     readouts L and I in EXPERIMENTS_V14 "B2"). The 6000 then goes back to
     v13 (E2 port to M2 if E2 passes).
   - **4090, when v13 pings that E2 freed it (~09:15):** (c) **B0 run 3**
     (`TAG=b0r3 ARMS=echo,n1sum,n1gate,n1phase,echo_tied,tpp AUX=0.1 SAVE=1
     bash v14/run_bench.sh`, ~2 h).
   - v13 will run the B2 tasks as an inference-only probe on their 600M
     checkpoints (base and `_new` reported separately: for models with no
     B2 training they should score alike, which is the difficulty control).
2. Open question from runs 3-4: why does label_marker in training stop the
   tied arm from forming the circuit (1 of 3) while helping echo (2 of 3)?
   Probe the saved checkpoints (`checkpoints_v14/b1r3p*`, `b1r4p*`, on the
   4090 box) on CPU: what do q and k match at the label positions?
3. Bar 3 in its strong form is untouched by every arm (plan under
   EXPERIMENTS_V14 "N4", bar-3 analysis).
