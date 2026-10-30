# v14 handover

Read `README.md` (rules), `EXPERIMENT_INDEX.md` (one row per idea) and
`EXPERIMENTS_V14.md` (do-not-repeat digest and pre-registered bars) first.
Keep all three current as things happen (user rule, 2026-10-29).

## State (2026-10-30 13:45 IST)

* **4090 queue (15:40 IST):** v13 g1c (whole card) until ~18:00; v14's slot:
  **F1** (tmux `v14f1`, worktree `../qllm_v14f1` at `7c8c6e2`: echo +
  echo_phase × 5 seeds on B1) then **N10 formation run** (tmux `v14n10`,
  `../qllm_v14n10run` at `9bd36e6`, checkpoints symlinked to d_drive: echo /
  op / opcoord × 5 seeds on B2). After g1c, v13 runs echo_qinit (Q1) and
  echo_sparse (S1) on the F1 seeds. **Checkpoints lost** (worktree remove
  --force, 15:40): B2 run 2, state sweep, N7 pair 1 micro weights (numbers
  are in logs/); rerun from the commit if a formed checkpoint is needed for
  probing. F1's checkpoints are inside `../qllm_v14f1`: move them to d_drive
  before removing that worktree.
* **(old, 13:45) 4090 queue:** v13 g1c (whole card, agreed) since 13:35 ->
  ~16:45. Beside it, v14's one small job: **N7 micro run 1** (tmux `v14n7`,
  worktree `../qllm_v14n7run` at `73265e9`, driver `n7r1_driver.sh`; echo /
  echo_n7 / tpp, 3 pairs, bar X) since 13:35, then **N9a micro run 1**
  (tmux `v14n9`, `../qllm_v14n9run` at `b0e7fc6`; echo / echo_phase, 3
  pairs, bar Xa). After g1c, v13 runs **Q1** (tied START of echo q/k,
  formation rate over 5 seeds on B1 + B2; arm `echo_qinit`, `3cf2aa8`) on this
  bench as its small job, results ~21:30.
* **B2 so far:** run 2 R MET (echo infers held-out shifts, T++ only
  selects); state sweep: no reliable size effect, but the O(1) body solves
  held-out shifts FULLY in some seeds (3 of 10 echo-family runs ≥ 0.9; T++
  0 of 4). Formation rate is the open lever (Q1 and N8 target it).
 (6000, `bf59754`, one seed): L and I met (B2 is
  valid).** Held-out rule parameters: echo 0.739, n1sum 0.731, tpp 0.666.
  Numbers (held-out step / affine offset): every arm ~0.9–0.99 (exact
  inference). **Letters (held-out shift): echo 0.33, rising 0.20 → 0.47 with
  shots; tpp 0.02 (picks the neighbour = memorised tables).** First bar-3
  signal for the O(1) body. **B2 run 2** (echo vs tpp, 3 seed pairs
  41/42/43, bar R on shift_wide_new) is pre-registered and goes on the 4090.
  If R is met: a 600M check with B2 docs in training for both bodies
  (design with v13: they offered rules_ood docs in their mix).
* **Closed today:** N6 (B1 run 2), echo_tied (v13 E2 at 600M: label_category
  0.365 vs 0.500, holdout +0.055; the micro-bench win did not transfer), N1
  (B0 run 3 stop rule: n1phase 0.370 < n1sum 0.380), and B0 itself (bench
  broken: echo facts2 0.52, tpp facts1 0.17).
* **tpp calibration on B1:** attention does not form the label_category
  circuit at 27M either (0.42–0.43, 3 seeds), so micro-bench label_category
  does not predict the 600M gap (T++ 0.73).
* **Lesson (E2):** a micro-bench win is a lead, not evidence. Seed pairs
  first, then the LM check.
* **GPUs:** 6000 = v13 S4 chain (~11 h from ~10:00 IST; Mamba-2 600M ×2,
  then the reference at 3B); v14 runs there only from the worktree
  `~/Development/qllm_v14`, never by pulling `qllm-private`. 4090 = v14.
* **Shared checkout rule:** both sessions use
  `/home/gowrav/Development/qllm2`. v14 benches now run from a local
  worktree (`../qllm_v14run`, detached, `.venv` symlinked) so commits to
  master never block a crash-resume.
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
* **v13 reference body = c1m2** (M2 + aptitude two-stage; tie-break set
  before c2m2, `900dde2`). c2m2 (M2, two stages, no aptitude) KEEP: holdout
  3.4497, WikiText 65.57, recall 0.848, label_category 8-shot 0.41. vs T++ on
  the same recipe (tfs2): holdout −0.041, WikiText 65.6 vs 66.2, recall
  +0.145. **First PAM ahead of T++ on holdout AND WikiText at equal data.** T++
  still leads label_category (0.73 vs 0.41) and training speed (1.45× at
  T=2048).
* **v13 S2 fairness (`048f8c2`): MET.** T++ gets nothing from the two-stage
  schedule (tfs2 holdout 3.4907 vs 3.4885 one stage; recall 0.703;
  label_category 8-shot 0.73 unchanged); PAM does. Two-stage M1 vs tfs2:
  holdout −0.016, recall +0.122, but recall ties inside 2048 tokens (0.917
  vs 0.926); the win is past 2048, where T++ collapses (0.15 vs 0.60).
  WikiText and label_category favour T++.
* **B2 with no B2 training (v13 probe on 600M / 3B, `a214002`):** every
  model is below chance (0.04-0.09; chance 0.125), attention included:
  they pick the answer next to the right one. `_new` is 0.015-0.047 harder
  in itself (difficulty control). Numbers in EXPERIMENTS_V14 "B2".
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

## Step back (2026-10-30 ~10:45 IST, after B0 closed, N1/E2 failed, B2 run 1)

Habit (memory `meditate-before-next-step`): after each finding, look from far
and from inside before choosing the next run.

1. **Two kinds of in-context skill pull apart.** Attention wins at *lookup*
   (find this item and copy what followed: label_category at 600M, T++ 0.73
   vs PAM 0.41). The recurrent body wins at *abstraction* (carry one rule
   across examples: B2 letters, echo 0.33 rising with shots, T++ 0.02 =
   per-k tables). Today's failures all tried to buy lookup: the tie made
   q/k symmetric (similarity), which lost recall (asymmetric: "what came
   after X"), 0.825 -> 0.602 at 600M. Stop patching lookup into echo; lean
   into abstraction, where an O(1) state has an edge.
2. **(Tested 13:34: NOT supported; seeds order sizes oppositely. The O(1)
   body solves held-out shifts fully in some seeds: circuit formation is the
   open lever.) Hypothesis: the bottleneck causes the abstraction.** Attention can keep
   every example, so memorising a table per parameter is cheap. A fixed-size
   state cannot hold the examples, so it is pushed to hold the rule. If
   true, a smaller state should infer held-out rules *better* (and recall
   worse), and that is a new, testable claim. Cheap test: a head-dim sweep
   on B2 (the micro-bench already takes `echo@32`, `@64`, `@128`), with tpp
   as the reference.
3. **The accident to chase:** echo's held-out numbers get *worse* with more
   shots (step_wide_new 0.98 / 1.00 / 0.72 at 2 / 4 / 8 shots); tpp does
   not (0.97 at 8). Larger numbers (s + 8d) or state saturation over a long
   numeric run? It is a capacity / precision clue for the same story.
4. **Mechanism direction (N4 rewritten):** a *relation memory*: write the
   relation between paired items (δ = φ(y) − φ(x)) instead of the items, then
   read it once and apply it to the query. N6 tried the rotation version and
   was judged on B1, a bench that could not show inference (extrapolation
   from 3 values). B2 is the fair test for this family.
5. **Bench lesson:** B0 broke even for attention (answer-only supervision,
   redrawn words). Micro-benches need dense LM-style supervision, or they
   measure the bench. And a micro-bench win is a lead (E2): seed pairs, then
   600M, both inside and beyond the trained context.

## Meditation (2026-10-30 ~14:00 IST, asked by the user)

Everything today points at one picture: a brain-like O(1) mind, three parts.
1. **Innate coordinates (grid cells).** Whether a seed learns the rule is a
   lottery (3 of 10). When models do learn arithmetic-like rules they find
   clock / circle codes (grokking, number helices). Grid cells give brains
   multi-period phase codes from the start, and path integration (adding a
   velocity to a phase) IS rule continuation. So give tokens multi-period
   phase coordinates from the start (the user's circles idea, N10) and let
   the rule memory integrate the per-example phase step. Test: does
   formation go from 3/10 to ~10/10 on B2?
2. **Hippocampal memory for lookup: separate, then complete.** Our lookup dies
   because look-alike keys collide (template "is"). The dentate gyrus fixes
   exactly this with *pattern separation*: expand the key into a sparse,
   high-dimensional code, so similar inputs become nearly orthogonal. Sparse
   keys also raise an outer-product memory's capacity by orders of magnitude
   (Willshaw nets) and let the delta erase touch only its own entry, so
   "last write wins" stops. CA3 then does *pattern completion*, which is the
   settling step. So N9 = DG (sparse separating keys, phase as one way) +
   CA3 (settling); N7 is the extreme case (exact = maximally sparse).
3. **A prefrontal planner forms the query** (N11, pands): reason what is
   needed, then look in the knowledge store and the conversation memory.
Body = cortex; coordinates = grid cells; memory = hippocampus; planner =
prefrontal. Each part has a micro test on B2 / the N7 bench; N7 and N9a,
running now, are the first two data points.

## Step back (2026-10-30 ~22:20 IST, after v13's 3B result)

v13 r3b (`ef136f5`): the reference PAM recipe at 3B tokens ends at holdout
3.309 vs T++ 3.206 / Mamba-2 3.164 (WikiText 58.0 vs 46.6); PAM ties early
(1.2B) and plateaus, and Mamba-2 reaches PAM's 3B loss at ~1.6B tokens (~2×
data efficiency). Recall still beats T++ (0.620 vs 0.539). **So bar 2 is not
a lookup problem: the body's LM quality plateaus.** v14's wins so far
(N9a lookup formation 5/5; echo infers held-out rules where T++ selects) are
real but sit on a body that falls behind at scale. Lesson for every v14 unit:
LM effects need ≥ 1.5B tokens before belief; 600M checks only measure the
targeted skill (lookup, formation, inference) with holdout as a guard.
Open question for the design: what in the body plateaus (state precision /
decay, write rule, mixing), and does a unit change the slope, not just a
skill? A slope test needs ≥ 1.5B tokens (the 6000).

## Meditation 2 (2026-10-30 ~22:45 IST, asked by the user)

Looking at the whole day from far away:
1. **We have been patching skills onto a body that falls behind at scale.**
   Every v14 unit today (tied echo, N1, N6, N7, N9a, N10) adds a skill to
   PAM, but r3b shows the body's LM curve flattens (3B: 3.309 vs T++ 3.206,
   Mamba-2 3.164). Skills cannot fix a slope.
2. **The prime suspect for the slope is state size and how many layers mix.**
   Mamba-2 carries ~13.4 MB of state per sequence and mixes in every layer;
   PAM ~1.2 MB, with notebooks in only 4 of 16 blocks (M1/M2 were chosen at
   600M for speed and early loss). PAM's own older 16-notebook 3B run (more
   state, more mixing) ended better (3.273). Known O(1) models (Mamba-2,
   Gated DeltaNet, RWKV-7) do match T++ at this scale, so O(1) is not the
   wall; our layout may be. This is v13's lane, sent as a EUREKA.
3. **v14's real lever for the slope may be cheap capacity, not more skills.**
   N7's exact n-gram table is a large, sparse state at O(1) cost per token,
   and cache LMs (Grave 2017, kNN-LM) gave real perplexity gains on web text
   because web text repeats names and phrases. So N7 should be tested on LM
   loss (≥ 1.5B tokens), not on a micro copy task. Same family: sparse
   high-dimensional keys (Willshaw capacity) for the notebooks.
4. **Micro-benches answered "why" well, but transfer poorly.** Keep them for
   diagnosis (the collision story, formation), and send anything aimed at bar
   2 to a slope test early.
5. **The novel parts that hold up are ours to keep:** echo's rule inference
   (beats same-size T++), N9a's phase keys (lookup forms on every seed), long
   recall past 2048. The plan is a competitive base body plus these, not
   these instead of a base body.
**Checked by v13 the same night (EXPERIMENTS_SEMPY "Slope check"):** late
slope 2 -> 3B holdout: PAM 4 notebooks LR 2e-4 −0.045; 16 notebooks LR 2e-4
−0.060; **16 notebooks LR 1.2e-3 −0.086**; T++ −0.081; Mamba-2 −0.079. The
plateau is the layout AND the LR; with 16 notebooks at a high LR the body
keeps pace (it starts worse: 3.537 at 1.2B, ends 3.284). v13 runs body × LR
at 1.5B next. **So v14's LM tests use that body (16 notebooks + echo, high
LR), not M1/M2.**
Next for v14: after n9c and N10 land, pre-register N7 as an LM test (600M
first as a signal, then ≥ 1.5B on the 6000), and design the sparse-key
notebook (Willshaw) as the capacity unit.

## Next action

1. B2 run 2 on the 4090 (from `../qllm_v14run`): `for i in 1 2 3; do
   TAG=b2r2p$i FAMILIES=rules ARMS=echo,tpp SAVE=1 EXTRA="--seed 4$i
   --doc_seed0 $((i+7))00000000 --init_seed 4$i --rule_tasks
   shift_wide,step_wide,affine_map" bash v14/run_bench.sh; done` (~1 h).
   Score R (paired d on shift_wide_new), record, ping v13, tell the user.
2. If R is met: design the 600M check with v13 (B2 docs in stage B for an
   M1/c1m2 arm and a T++ arm; read shift_wide_new inside and beyond the
   trained context). If R is not met: shift_wide_new is the target cell for
   a new unit (N4 design under EXPERIMENTS_V14 "N4").
3. After R: pre-register the B2 head-dim sweep (step back, point 2) and the
   relation-memory unit (point 4) before either runs.
4. Open: why echo infers letter shifts partly and not fully (0.47 at 8
   shots); probe the B2 run 1 checkpoints (6000,
   `~/Development/qllm_v14/checkpoints_v14/b2_bf59754_*`).
