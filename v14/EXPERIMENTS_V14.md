# v14 lab notebook — new neurons

Read part A before proposing anything. It is the condensed result of every
experiment file from v4 to v13_sempty (read 2026-10-29), so v14 never reruns
a closed question.

## Index

The one-row-per-idea index (status, verdict, run hash, rerun recipe) is
`EXPERIMENT_INDEX.md`. Update it whenever a row's state changes.

---

## Part A — what v4–v13 already settled (do not repeat)

### A.1 Results that constrain any new design

| lesson | evidence | what it means for v14 |
|---|---|---|
| Binding by complex multiply into a **vector** state fails: facts interfere | V6 HSB 43.54 vs GSP 41.67 (`EXPERIMENTS_V6_PART2.md` §1–2) | phase binding must live in a **matrix** state, or select between matrix states. Never superpose facts in one vector |
| Phase interference **between matrix states** works | v11 E3: K=2 26.01, K=3 25.77 vs 26.88. The only v7→v11 architecture win (`v11/EXPERIMENTS_V11.md`) | the one proven "uniquely ours" lever. Dropped in v13_sempty with all other machinery, never retested on the lean body |
| Real arithmetic matches complex in the lean model, and is faster | v13_sempty row 3 + fair run 23.81 (real) vs v11 25.77 (complex, older recipe) | complex must earn its cost with a mechanism that real numbers cannot do (interference, cancellation) |
| Retention levers do not move the horizon | R1 dt-spread, R2 no_decay, R3 split prior, R4 store, C1 coupling, base_dt_bias, L-2 data pressure: all FAIL (`v13_sempty/EXPERIMENT_INDEX.md`) | do not tune decay again. `dt_bias` stays at init (−4) under every recipe |
| Multi-way recall fails because of the **linear read** | `q·S = Σ_i (q·k_i) v_i`: one query cannot hit its key and null 7 others. Not a width problem: a8 flat over head_dim/K (`EXPERIMENTS_SEMPY.md` A2r note) | a better read (nonlinear, selective, interference) is the open lever. Echo fixed a4/a8 on the micro-bench; the wall beyond that is unmeasured (B0) |
| Echo is a large win | micro-bench a4/a8 chance → 1.00; LM −0.068 / −0.060 nats at 150M / 300M, WikiText 88.0 vs 100.4 | echo is part of the control body |
| Hierarchy / grouped layers / cross-level drift regress | v7 7f-0..7f-3 | slow-stream designs are owned by T1 (v13_sempty, other session). v14 reuses its verdict, does not rerun it |
| Event / salience episodic memory was inert | V6 memory reframe: +35 % cost, PPL identical (56.55 vs 56.46) | a surprise-gated write must be load-bearing by construction (pands lesson: an extra path the LM loss can ignore becomes decoration, `pands/v2/EXPERIMENTS_V2.md`) |
| Iterative reasoning loops collapse to passthrough | V8 QLC: α/γ pinned at noise floor; later "muted" | extra paths that the LM loss can ignore will be ignored |
| Readout gates / norms / short conv | V9 gates (not reproduced), v13 A1 short conv FAIL, G2 read norm FAIL | no more read-path normalizers |
| Multi-scale aux loss, reverse association, QK-norm on PAM | v7 7f-1 (+0.82), 7f-2 (+2.46), V6 Bug 8 (repetition collapse) | closed |
| Per-channel decay | v11 E1 tie, hurts with E3 | closed |
| Content-routed delta memories | A2r: a8 at chance, 4–8× cost | closed |
| Module packing / narrow specialist data | V12: packing destroys capability; narrow synthetic data → catastrophic forgetting; delta ≈ additive at that budget | train one model end to end on the mix |
| Supervision density matters more than tokens | V12 finding 18 | answer-weighted recall docs stay in the mix |
| Screens must be long enough | 100M screens picked the wrong LR (1.2e-3 vs 2e-4 at 3B); judge PAM on ≥ 600M for a final verdict, holdout not train loss | micro-bench → 150M holdout check → 600M screen, in that order |
| Speed is part of the verdict | PAM 66K tok/s vs Mamba-2 189K vs Transformer++ 138K (6000) | report ms/step for every arm; a unit that costs > 1.3× needs a large quality gain |

### A.2 Ideas written down earlier but never measured

| idea | where | state | v14 |
|---|---|---|---|
| Phase-modulated activation (magnitude → phase rotation, zero-init so it starts as ModSwish) | v7 Exp 7 Design B; run `exp7b_phase_mod` stopped mid-epoch 1, no val | coded in `v13_sempty/complex_ops.py` (`activation='phase_mod'`), complex arm only | **N2** |
| Formula weights: `W = Σ_k e^{iφ_k} U_k V_kᵀ` (phase-superposed linear) | v7 Exp 4 | design only, never coded | **N3** (with input-dependent φ) |
| E/I heads (half the heads build state, half clear it) | v7 Exp 6 | design only | skipped: delta erase covers "clear", R3 (heads with different decay priors) failed |
| Spatial neuron field | an August design note, before v14 began; removed from `v14/` 2026-10-29, read it with `git show 73ea737:v14/NOTES.md` | notes only | not in the v14 plan |
| Multi-state superposition on the lean body | v13_sempty A2 `--n_states` | in code, not adopted (cost, no decode); never measured on LM with a phase-combined read | **N1** |

---

## Part B — the v14 experiments (pre-registered)

**Novelty rule (user, 2026-10-29).** Not a transformer or Mamba variant.
Clarified the same day: this does not rule out ideas that have been written
about. "Anything that can be called innovation or novelty is what we need."
Known parts are fine; the bar is that what we build is a genuine innovation
(a new mechanism, or a new scheme), not a re-run of a known model. An
earlier, stricter reading ("drop anything in a known family") was wrong and
is withdrawn.

**What is new in each item:**

| id | what is new | closest prior art |
|---|---|---|
| N1 | rotation interference between delta notebooks (v11's own idea, never on the lean body) | multi-state fast weights, v11 E3 |
| N2 | a unit whose output phase depends on its own strength | complex-valued activations |
| N4 | memory inside every unit, written in proportion to echo's surprise | fast weights, TTT, Titans |
| N5 | binding by synchrony inside an O(1) language model | AKOrN, complex autoencoders (vision) |
| N6 | the state is an inferred operator applied to new inputs, not stored items | RotatE (static KG), HRR/TPR |

Priority = how new × how likely to move the three bars: N6 (targets bar 3
directly), then N1 (cheap, proven once in v11), then N4 / N5. Before a 6000
slot is spent on a "new architecture" claim, the other session reviews it.

**The goal has three bars, all needed** (user, 2026-10-29; shared memory
`goal-pam-beats-baselines`): (1) O(1) state per token; (2) beat
Transformer++ and Mamba-2 at equal data; (3) learning beyond recall:
apply a rule seen in context to inputs that never appear in the context
(rule induction, composition, abstraction). B0 and the recall probe only
measure memory. **B1** measures bar 3 on the micro-bench, and every v14
unit reports both B0 and B1.

**Bar 2 in practice (user, 2026-10-30).** A few more parameters or a bit
more training than the transformer is acceptable: an O(1) model serves more
users and its inference is cheaper (no KV cache growing with context). So
compare quality at comparable cost and state params and tokens with every
result. **Learning past the trained context length is a strength to
measure:** report results inside and beyond the training length (v13 S2:
T++ recall collapses past 2048, 0.15, while two-stage M1 holds 0.60).

**Control body** for every LM arm: **M1** (KEEP at 600M, 2026-10-29,
session "Qllm v13"): `baseline_real_pm` + chrono + out_gate + delta + echo,
notebooks only in blocks 3, 7, 11, 15, MLP hidden 2352 (102.94M). 600M:
holdout 3.4906 (echo 3.4885), WikiText 69.4 (71.1), recall 0.615 (0.401),
1.8x faster (116K vs 64K tok/s on the 6000). Checkpoint (6000)
`checkpoints_v13_sempty/m1_m1echo_ccbf028/best_model.pt`. N1 on M1 swaps the
4 notebook blocks only (+28k params with `phase_read='phase'`). The B0
micro-bench keeps its own small all-notebook body (4 layers): it tests the
memory mechanism, not the layout.

**M2 is also KEEP** (2026-10-29, v13 session: shared reads, holdout 3.4697
vs M1 3.4906, WikiText 68.84). But its recall fell (0.522 vs 0.615) and it is
1.18x slower than M1. Decision: stage 3 keeps **M1** as the control. It is
faster, and its memory reads are not shared, so a v14 memory unit is not
mixed up with M2's shared reads. A unit that passes stage 3 is also compared
with M2's 600M numbers before it gets a 6000 slot.

**Stages for every N-arm:**
1. selftest: parallel == recurrent (or decode) to round-off; layout guard clean.
2. micro-bench (4090, minutes): `v13_sempty/tmp/recall_microbench.py`
   extended with the B0 cases.
3. LM same-batch check: **150M holdout** vs control (M1 at 150M: 3.9225;
   echo 3.9516). Holdout outranks train loss, which the ×100 recall-answer
   weight dominates for memory-changing arms.
4. 600M screen (6000) only if stage 3 passes. Same pass format as the
   v13_sempty screens (holdout, recall 63-cell, PG-19 far gain, tok/s).

### B0 — find the wall (no new code in the model)

**Question.** Echo took a4/a8 from chance to 1.00 on the micro-bench. Where
does the control body fail now? Without a failing case, a new neuron has
nothing to prove on memory.

**Design.** Extend the micro-bench (same 27.6M, 4 layers, hd 64):
- more facts in context: a16, a32, a64;
- **bound attributes**: several entities, each with 2–3 attributes, query
  one attribute of one entity ("red apple, green car, blue cup … car → ?"
  and "car colour / car owner"). This is the binding case HSB failed on;
- **distractor overwrite**: same key rebound later; query must return the
  newest value.

**Tool (`v14/microbench.py`, 2026-10-29).** Words are random single GPT-2
tokens, redrawn per document (disjoint key / value / entity pools). 4
queries per document, answers teacher-forced and the only supervised
positions. 8 candidates, hard distractors first (other values of the same
document; for `bind` the entity's other attributes and the other entities'
same-relation values; for `rebind` the key's old values). Sizes: facts
8/16/32/64, bind 4/8/16 entities, rebind 4/8/16 keys. Train lengths
512–4096; eval 1024 / 4096 / 8192 at positions 0, 0.5, 1 (16 docs each:
192 answers per cell, SE ≈ 0.036). 6000 steps × B16, LR 3e-4, same data
for every arm. Launcher: `v14/run_bench.sh` (`v14/run_b0.sh` up to `0671d65`).

**Output.** The smallest setting where the control drops below 0.5.

**Run 1 (6000, 2026-10-29 18:02–18:29 IST, code `a0f672a`, sizes from 8
up, answer-only, 6000 steps): INVALID as a bench.** Stopped after 2 of 5
arms to give the slot back to M2: neither arm learned lookup, so the N1 arms
could not have been judged.

| arm | facts 8–64 | bind 4–16 | rebind 4–16 | final train loss |
|---|---|---|---|---|
| delta | 0.09–0.12 | 0.11–0.14 | 0.11–0.15 | 6.36 (= ln 600: "some value word") |
| echo | 0.10–0.17 | 0.11–0.13 | 0.20–0.32 | 4.26 |

(range over sizes and ctx 1024/4096/8192; chance 0.125.) Echo's rebind
edge is a recency bias, not lookup. **Diagnosis:** no easy rung. The v13
micro-bench grew the lookup circuit on 1 association first (a1 0.46 → 1.00)
and then scaled; run 1 started at 8 facts with words redrawn per document.
Log + JSON: `logs/v14_b0_a0f672a_20261029_1232.{log,json}`.

**Run 2 (design, written before the run; it ran on the 4090 2026-10-30 01:45–03:31 IST).**
One change: sizes start at 1 (facts 1/2/4/8/16/32/64, bind 1/2/4/8/16,
rebind 2/4/8/16), train and eval (`316b0f9`). Still answer-only
(`--aux_weight 0`; the option exists if run 2 also fails).
**Validity check, written before the run:** the bench is valid only if echo
scores ≥ 0.90 on facts1 and facts2 at ctx 1024. If not, run 3 adds
`--aux_weight 0.1` and nothing is judged from run 2. That
setting becomes the memory bar for N1, N4, N5. If nothing fails up to a64
and bound attributes, memory is not the limiter and v14 focuses on LM
quality (N2, N3).

**Run 2 RESULT (4090, 2026-10-30 01:45–03:31 IST, code `bcfd88a`, sizes
from 1, answer-only): INVALID by its validity rule.** Echo scores 1.00 on
facts1 but only 0.55 on facts2 @1024 (bar 0.90). Nothing is judged.
Log + JSON: `logs/v14_b0_bcfd88a_20261030_0145.{log,json}`. The log is
cleaned of torch recompile warnings.

| arm | f1 | f2 | f4 | f8 | f16–64 | bind1 | bind2 | bind4 | bind8–16 | rebind2 | rebind4 | rebind8–16 | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| echo | 1.00 | 0.56 | 0.28 | 0.10 | 0.14–0.15 | 1.00 | 0.46 | 0.24 | 0.11–0.16 | 0.40 | 0.24 | 0.19–0.25 | 159 |
| delta | 0.15 | 0.16 | 0.12 | 0.13 | 0.13–0.14 | 0.15 | 0.17 | 0.12 | 0.14 | 0.19 | 0.12 | 0.11–0.13 | 146 |
| n1sum | 1.00 | **0.79** | **0.50** | 0.18 | 0.11–0.18 | 1.00 | 0.51 | 0.17 | 0.12–0.14 | 0.53 | 0.35 | 0.19–0.31 | 194 |
| n1gate | 1.00 | **0.78** | **0.52** | 0.14 | 0.11–0.15 | 1.00 | 0.49 | 0.24 | 0.12–0.17 | 0.48 | 0.27 | 0.23–0.27 | 188 |
| n1phase | 1.00 | 0.51 | 0.29 | 0.13 | 0.11–0.15 | 1.00 | 0.52 | 0.20 | 0.13–0.16 | 0.41 | 0.22 | 0.23–0.24 | 193 |
| echo_tied | 1.00 | 0.50 | 0.26 | 0.12 | 0.08–0.14 | 1.00 | 0.55 | 0.26 | 0.16 | 0.39 | 0.22 | 0.23–0.25 | 157 |

(Each cell is the mean over ctx 1024/4096/8192, 576 answers per cell, SE ≈ 0.02.
Candidate chance is 0.125; "pick any value of this document" gives 1/n on
facts n.)

**Reading (descriptive, not a verdict):**
- Echo and echo_tied sit at 1/n on facts n: they answer with a value from
  the document, but not the one bound to the queried key. That is why the
  bench is invalid: the control never learned lookup.
- **n1sum and n1gate are clearly above 1/n** (facts2 0.79 / 0.78 vs 0.50,
  facts4 0.50 / 0.52 vs 0.25, rebind4 0.35): two delta notebooks with
  different fades do real partial lookup. delta alone stays at chance
  (train loss 6.41 = ln 600), so the gain needs echo plus the second
  notebook.
- **n1phase falls back to echo's level** (0.51 / 0.29). Turning the reads
  cancels the second notebook's gain. That is the opposite of the N1
  hypothesis (phase > gate > sum).
- **echo_tied costs no memory** vs echo (facts2 0.50 vs 0.56, ~2 SE; equal
  elsewhere). It runs at the same speed (157 vs 159 ms/step).
- N1 arms cost 1.2× echo per step (188–194 ms), within the 1.35× budget.

**Run 3 (pre-registered 2026-10-30 ~04:20 IST, before any run-3 number;
runs on the 4090 after B1 run 4, or after E2 if the v13 session takes it).**
The rule written for run 2 applies: one change, `--aux_weight 0.1` (LM
loss on the non-answer positions, total weight 0.1 per answer). The other
settings are the same as run 2 (sizes from 1, 6000 steps × B16, LR 3e-4,
seed 42, same data for every arm).
- **Arms:** echo, n1sum, n1gate, n1phase, and **tpp**, a
  REFERENCE (Transformer++, attention, O(T), not a candidate; same width,
  depth and heads, MLP 1280 so params match echo: 27.56M vs 27.61M; arm
  added in `c9bff5c`). delta is dropped: it was at chance in every cell of
  runs 1 and 2, and no bar uses it.
- **Validity (amended before run 3):** the bench is learnable if **echo or
  tpp** scores ≥ 0.90 on facts1 and facts2 @1024. Run 2 asked echo alone
  because there was no reference, so a broken bench and a weak body looked
  the same. tpp is that reference.
  - echo passes: as planned. The N1 bars are judged on the headroom cells H
    (echo < 0.90).
  - tpp passes and echo fails: **the wall is found**. The O(1) control fails
    lookup at a size where attention succeeds on the same data. The N1 bars
    are judged on H as written, and the echo-vs-tpp gap per cell is the
    memory target for N1, N4 and N5.
  - both fail: the bench is broken at this size. Stop rule below.
- (echo_tied was dropped before the run: it closed with v13 E2.)
- **Stop rule:** if neither echo nor tpp passes, B0 closes as a memory
  bench at this size (lookup of words redrawn per document does not grow at
  27.6M / 6000 steps, with or without aux). There is no run 4 without a new,
  pre-registered design. N1 then has no micro-bench judgement. It gets the
  150M LM check only if n1phase ≥ n1sum on mean(facts2, facts4) @1024 in
  run 3. Otherwise N1 closes as FAIL (in both runs, turning removed the
  second notebook's gain).

**Run 3 RESULT (4090, 2026-10-30 08:51–10:19 IST, code `bf59754`, aux 0.1):
INVALID again, so the stop rule applies. B0 CLOSED as a memory bench at
this size.** Log + JSON: `logs/v14_b0r3_bf59754_20261030_0851.{log,json}`.

| arm | f1 | f2 | f4 | f8–64 | bind1 | bind2 | bind4 | rebind2 | rebind4 | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|
| echo | 1.00 | 0.52 | 0.28 | 0.11–0.16 | 1.00 | 0.49 | 0.27 | 0.41 | 0.22 | 163 |
| n1sum | 1.00 | 0.49 | 0.27 | 0.12–0.14 | 1.00 | 0.50 | 0.28 | 0.40 | 0.22 | 194 |
| n1gate | 1.00 | 0.47 | 0.28 | 0.10–0.14 | 1.00 | 0.53 | 0.26 | 0.40 | 0.22 | 193 |
| n1phase | 1.00 | 0.47 | 0.29 | 0.10–0.15 | 1.00 | 0.52 | 0.24 | 0.39 | 0.24 | 198 |
| tpp | 0.17 | 0.13 | 0.13 | 0.12–0.15 | 0.15 | 0.18 | 0.10 | 0.21 | 0.10 | 116 |

- **Validity:** echo facts2 @1024 0.52 and tpp facts1 0.17 (chance): neither
  passes, so the bench is broken at this size. Attention does not even learn
  one-fact lookup here, and the O(1) bodies sit at 1/n (they pick a value
  of the document, not the bound one).
- **N1 by the stop rule:** mean(facts2, facts4) @1024: n1phase 0.370 vs
  n1sum 0.380, so n1phase < n1sum and **N1 CLOSES AS FAIL** (no LM check).
  Run 2's n1sum / n1gate partial lookup (facts2 0.79 / 0.76) did not
  reappear with aux 0.1 (0.49 / 0.49): it was not robust either.
- No run 4 without a new, pre-registered design.

### B1 — rule induction (shared generator, pre-registered 2026-10-29)

**Question.** Can a unit learn a rule from a few examples in the context
and apply it to a new input whose answer is nowhere in the context?

**Tasks** (`memory_probes/rules.py`, the v13 session's learning probe,
`ed326a1`; 8 candidates incl. tempting wrong answers, chance 0.125):
letter_shift (hidden shift 1–3), number_step (step 2–9), reverse3 (answer
= first query word, placed by the rule), label_category (invented label per
semantic category, query item never labelled), two_hop (owner → pet →
name, facts shuffled).

**Micro-bench form (`v14/microbench.py --families rules`).** Episodes
(prompt + answer + blank line) packed into 512 / 1024-token sequences;
every answer supervised (answer-only loss). Train shots 1–8, random task
per episode; eval 2 / 4 / 8 shots, one task per document, held-out seeds,
16 docs per cell (18–73 answers per doc). Same small body and loop as B0.

**Run 1 (done 2026-10-30 00:17 IST, 4090, `5b1eb9d`, log + JSON
`logs/v14_b1_5b1eb9d_20261029_2329.{log,json}`):** arms echo, op, opdiag,
n1phase, delta; 6000 steps × B16; `TAG=b1 FAMILIES=rules`. N6 bars are
frozen in the N6 section.

| task (in-context chance at 2 / 4 / 8 shots) | echo | op | opdiag | n1phase | delta |
|---|---|---|---|---|---|
| letter_shift (0.125) | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 |
| number_step (0.125) | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 |
| reverse3 (0.33) | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 | 1.00 1.00 1.00 |
| label_category (0.50 0.50 0.25) | 0.76 0.77 0.52 | **1.00 1.00 1.00** | **1.00 1.00 1.00** | 0.54 0.51 0.24 | 0.52 0.49 0.26 |
| two_hop (0.50 0.25 0.125) | 0.52 0.25 0.12 | 0.51 0.27 0.09 | 0.51 0.27 0.11 | 0.51 0.27 0.12 | 0.49 0.24 0.11 |
| ms / step (train) | 99 | 94 | 95 | 107 | 83 |

Answers per cell: label_category 1088 / 704 / 416, two_hop 512 / 289 /
160 (16 held-out docs each).

**Reading.**
- **Validity: met by the letter of the rule, but three tasks hit the
  ceiling.** Every arm, delta included, scores 1.00 on letter_shift,
  number_step and reverse3. When a small body trains on the same task
  family, it learns these rules in its weights. Context only has to pick
  the hidden parameter (k from 3 values, d from 8). These cells cannot
  separate arms. **Bench lesson:** a rule with a small, trained parameter
  set is not an in-context induction test. Evaluate on rule parameters
  never seen in training (B1 run 2).
- **label_category separates arms.** The query item is never labelled in
  the context, so the label must be bound to the *category*. Delta and
  n1phase sit at in-context chance (they know the answer is one of the
  labels shown, not which). Echo is above chance. op and opdiag are
  perfect at all shot counts.
- **two_hop: every arm is at in-context chance.** No small body composes
  two facts. At scale it is the same (v13 learning probe).
- At scale, label_category 8-shot (chance 0.25): Transformer++ 3B 0.56,
  delta 3B 0.53, Mamba-2 3B 0.38, M1 600M 0.29 (`logs/learning_probes/`).
  It is the only rule task where any LM is above chance. **Update
  2026-10-30 (v13 C1-ctl, `237173a`):** M1's body on a *two-stage schedule*
  (120M web + chat first, then 480M of the M1 mix, LR re-warmed) reaches
  0.55 at the end (0.51 at the matched step). That is the schedule, not
  aptitude data. It also beats one-stage M1 on holdout (3.4743 vs 3.4906)
  and recall (0.825 vs 0.615). So 0.55 is what a v14 unit must beat on
  label_category at 600M, not 0.29.
**Purpose:** control levels for small bodies, and the first N6 test. N1 and
delta are reported only (memory mechanisms, not learners). **Validity check:** the bench is
usable if some arm reaches ≥ 0.50 on reverse3 or two_hop at 8 shots; else
run 2 adds `--aux_weight 0.1`.

**Run 2 (PRE-REGISTERED 2026-10-30 00:40 IST, before any run-2 number;
review by the v13 session folded in).** Code: `v14/rules_ood.py`, arms
`opdiag_ownq` and `echo_tied`, launcher `v14/run_bench.sh`.
- **Training:** the same task mix as run 1 (shared tasks, shots 1–8, 6000 ×
  B16, lr 3e-4). New data (`--seed 7 --doc_seed0 50000000`: no training
  document is shared with run 1) and new init (`--init_seed 1`). Weights
  kept (`SAVE=1`, `checkpoints_v14/b1r2_<hash>_<time>/`), so later evals
  need no retraining.
- **Arms: a 2×2 cause test plus the rotation arm.** All arms have echo. The
  "diagonal" row adds N6's diagonal memory on top of it.

  | | query and key share one projection | separate projections |
  |---|---|---|
  | + diagonal memory | `opdiag` | `opdiag_ownq` |
  | echo's matrix only | `echo_tied` (W_q = W_k) | `echo` (control) |

  Plus `op` (rotation-only diagonal, shared projection).
- **Eval:** run 1's cells, plus `letter_shift_ood` (k 4–5) and
  `number_step_ood` (d 10–13) at 2 / 4 / 8 shots. These parameters are
  never drawn in training. The tempting candidates lead with the
  training-prior answers (the trained shifts 1–3; last + 1 and last + 9),
  so falling back on the prior scores below chance.
- **Bars (frozen):**
  - **R, replication** (mean of label_category over 2 / 4 / 8 shots):
    opdiag − echo ≥ +0.15 AND op − echo ≥ +0.15. Valid only if echo's mean
    is < 0.90 (else no room to show a gain). Not met → run 1's gain is
    recorded as not replicated, and N6 closes on B1 evidence.
  - **C, cause** (read only if R is met; G = opdiag − echo, same mean):
    echo_tied − echo ≥ 0.5·G → tying the query to the key is enough on its
    own (a similarity lookup: same-category hidden states sit close).
    opdiag_ownq − echo ≥ 0.5·G → the diagonal memory is enough on its own.
    Both or neither is reported as such.
  - **O, the original N6 claim on unseen parameters** (mean over
    letter_shift_ood and number_step_ood at 4 and 8 shots): op ≥ echo +
    0.15 AND op ≥ opdiag + 0.10. **Prediction: not met.** Every arm ≤ 0.25,
    with the prior answers winning. Nothing in a small body puts letters or
    numbers on a circle.
- **Not done from the review: holding out label_category items or
  categories.** In this task, an item's category can only come from the
  weights (the context shows labels, never categories). A never-trained
  item therefore carries no category signal for any arm, and every arm
  would sit at chance by construction. What the context supplies is the
  binding of a per-episode label to a category, and that binding is never
  shown for the query item. The unseen-parameter tasks are the held-out
  test of in-context rule inference.
- **If R is met, next is the 150M LM check (stage 3), pre-registered
  before it runs.** M1 + the winning memory vs M1, both trained on their own
  150M schedule with one data mix and no aptitude data, and **the same
  schedule shape** (both one-stage, or both two-stage). The v13 C1-ctl run
  showed that the schedule alone moves label_category 8-shot from 0.29 to
  0.55. Primary: holdout loss, and the learning-probe
  label_category at 2 / 4 / 8 shots with `--trials 400` (SE ≈ 0.025).
  Also recall 63-cell.

**Run 2 result (done 2026-10-30 01:44 IST, 4090, `85ecf02`; log + JSON
`logs/v14_b1r2_85ecf02_20261030_0057.{log,json}`).** The first start
(00:39, `84f426b`) died of a python crash at 00:55 and was rerun from the
start.

| task (2 / 4 / 8 shots) | echo | op | opdiag | opdiag_ownq | echo_tied |
|---|---|---|---|---|---|
| letter_shift, number_step, reverse3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| label_category (chance 0.50 0.50 0.25) | 0.61 0.54 0.27 | 0.52 0.53 0.23 | 0.67 0.68 0.41 | 0.52 0.52 0.23 | **0.92 0.92 0.74** |
| two_hop (chance 0.50 0.25 0.125) | 0.54 0.30 0.09 | 0.53 0.31 0.08 | 0.50 0.28 0.09 | 0.52 0.27 0.12 | 0.51 0.28 0.08 |
| letter_shift_ood (chance 0.125) | 0.00 0.00 0.00 | 0.00 0.00 0.00 | 0.00 0.00 0.00 | 0.00 0.00 0.00 | 0.02 0.03 0.04 |
| number_step_ood (chance 0.125) | 0.09 0.20 0.05 | 0.04 0.09 0.06 | 0.04 0.15 0.05 | 0.08 0.14 0.04 | 0.06 0.14 0.04 |
| ms / step | 96 | 93 | 94 | 94 | 86 |

Answers per cell: label_category 1088 / 704 / 416; letter_shift_ood 1024 /
672 / 400; number_step_ood 1632 / 1168 / 736.

**Verdict against the frozen bars.**
- **R (replication): NOT MET.** label_category mean over 2 / 4 / 8 shots:
  echo 0.471, op 0.427, opdiag 0.587, opdiag_ownq 0.423, echo_tied 0.857.
  opdiag − echo = +0.116 and op − echo = −0.044, against +0.15 for both.
  Valid (echo < 0.90). Run 1's perfect op / opdiag did not replicate on a
  new seed, and op fell to chance. **By the frozen rule, N6 closes on B1
  evidence** (see "N6").
- **C (cause): not read**, since R is not met. Recorded anyway, as
  exploratory: **echo_tied − echo = +0.386**, the largest effect in either
  run. opdiag_ownq − echo = −0.048.
- **O (unseen rule parameters): NOT MET, as predicted.** The mean over
  letter_shift_ood and number_step_ood at 4 / 8 shots is 0.04–0.06 for
  every arm, below chance. letter_shift_ood is 0.00 for every arm except
  echo_tied: the models always answer with a trained shift, the tempting
  prior candidate. So no small body infers a new rule parameter from the
  examples. What run 1's ceiling hid is **selection among rules learned in
  the weights, not in-context induction.**
- two_hop: chance for every arm again.

**Seed variance is large on label_category.** Echo moved from a mean of
0.68 (run 1) to 0.47 (run 2), and op from 1.00 to 0.43. A single-seed gap
of ≈ 0.2 is within this bench's training-seed noise. So any claim from
here needs ≥ 3 training seeds per arm. echo_tied's 0.857 is above every
other arm in both runs, but it is one seed.

**What the 2×2 does say, read loosely.** The two arms whose query and key
share one projection, opdiag and echo_tied, are the two above echo. The
two with separate projections, echo and opdiag_ownq, are tied at the
bottom. op (shared, rotation-only) breaks the pattern. That fits the
similarity-lookup reading: with one projection, items of the same category
land near each other, so the query hits their keys. It is a lead for run
3, not a result.

**Run 3 (PRE-REGISTERED 2026-10-30 02:10 IST, before any run-3 number;
the v13 review's pairing and format (a) folded in).** Question: does tying
echo's query and key projections (echo_tied) really help, and does it help
because the tied query lands on the keys of *similar* earlier states (an
in-context similarity lookup)?
- **Arms:** echo vs echo_tied, in **3 seed pairs**. Within a pair both arms
  get the same data and the same init seed: pair i = `--seed 1i
  --doc_seed0 i·100M --init_seed 1i` (i = 1, 2, 3). One launch per pair,
  tag `b1r3p<i>`.
- **Training:** the shared tasks **plus `label_marker`** (format (a):
  two-word items, category marker + stem, marker first or second per
  episode; train word pools). Shots 1–8, 6000 × B16, lr 3e-4.
- **Eval:** every shared task, the unseen-parameter tasks, `label_marker`
  (train pools), and **`label_marker_new`**: marker and stem words from
  pools never used in training, so every item is unseen
  (`v14/rules_ood.py`).
- **Bars (frozen).** For each pair i, d_i = arm mean over 2 / 4 / 8 shots,
  echo_tied − echo.
  - **P, paired replication** (label_category): mean(d) ≥ +0.15 AND
    mean(d) ≥ 2·SD(d) (sample SD over the 3 pairs). Valid if echo's mean is
    < 0.90 in every pair.
  - **A, similarity lookup** (label_marker_new): the same paired rule.
    Valid if some arm's label_marker mean is > 0.517 (in-context chance mean
    0.417, plus 0.10). If no arm learns the format, A is uninformative.
- **What each outcome means:**
  - P and A both met: tying gives an in-context similarity lookup that
    works on unseen items. echo_tied goes to the LM check; the v13 session
    pre-registers a 600M tied-echo arm against two-stage M1 (holdout
    3.4743, recall 0.825, label_category 8-shot 0.55).
  - P met, A not met: the gain needs categories known in the weights, so
    the similarity-lookup reading is wrong. echo_tied still gets an LM
    check, framed as an abstraction / recall tweak.
  - P not met: run 2 was seed luck. echo_tied closes.
- **Predictions:** P met (mean d ≈ +0.3); A met, with a smaller gain.
  Unseen rule parameters stay ≤ chance for both arms.
- **Reported only:** the ceiling tasks, two_hop, the unseen-parameter
  tasks, ms / step.

**Run 3 RESULT (4090, 2026-10-30 03:32–04:37 IST, code `44df201`): P
INVALID, A NOT MET (the sign is reversed).** Logs + JSON:
`logs/v14_b1r3p{1,2,3}_44df201_20261030_{0332,0354,0415}.{log,json}`.
Each value is the mean over 2 / 4 / 8 shots. letter_shift, number_step and
reverse3 are 1.00 for every arm and pair.

| pair (seed) | arm | label_category | label_marker | label_marker_new | two_hop | letter_shift_ood | number_step_ood | ms/step |
|---|---|---|---|---|---|---|---|---|
| p1 (11) | echo | 0.999 | 0.983 | 0.653 | 0.306 | 0.062 | 0.108 | 110 |
| | echo_tied | 1.000 | 1.000 | 0.530 | 0.310 | 0.005 | 0.081 | 105 |
| p2 (12) | echo | 0.487 | 0.568 | 0.412 | 0.312 | 0.021 | 0.114 | 111 |
| | echo_tied | 0.420 | 0.409 | 0.423 | 0.287 | 0.005 | 0.227 | 104 |
| p3 (13) | echo | 0.946 | 0.938 | 0.593 | 0.301 | 0.001 | 0.114 | 110 |
| | echo_tied | 0.402 | 0.425 | 0.438 | 0.297 | 0.002 | 0.069 | 104 |

- **P:** d = +0.001, −0.067, −0.544 (mean −0.203, SD 0.297). **INVALID**:
  echo ≥ 0.90 in pairs 1 and 3. Where the arms differ, tied is worse.
- **A:** d = −0.123, +0.011, −0.155 (mean −0.089, SD 0.088). Valid
  (label_marker up to 1.00). **NOT met:** tied is worse on unseen items.

**Reading:**
- **The lookup circuit forms per seed, or it does not.** label_category and
  label_marker move together: ≈ 1.0 when it forms (echo p1 and p3, tied p1)
  and ≈ 0.4–0.5 when it does not. Echo formed it in 2 of 3 seeds, tied in 1
  of 3. Run 2's lead (tied 0.857 vs echo 0.471) was one seed where only tied
  formed it. *(Corrected by run 4: on run 2's training set, tied formed it
  in 3 of 3 seeds and echo in 0 of 3. "Seed luck" holds only with
  label_marker in the mix.)*
- **With label_marker in training, tying does not make the circuit form
  more often, and it is worse on unseen items (A).** The similarity-lookup
  reading of the tie is not supported.
- When the circuit forms, it carries over to words never seen in training
  (label_marker_new 0.59–0.65 vs in-context chance 0.417), only partly.
- **Data, not the body:** with `label_marker` in training, plain echo
  learned label_category to the ceiling in 2 of 3 seeds (runs 1 and 2
  without it: 0.68, 0.47). A related task in the mix helps the circuit form.
- Unseen rule parameters are still at or below chance (letter_shift_ood ≤ 0.06;
  number_step_ood ≤ 0.11 except one tied seed at 0.23).
- Tied runs 5 % faster per step (one projection fewer).
- The pre-registered outcome map had no row for "P invalid". Run 4 (below,
  pre-registered before any run-4 number) is the clean P test, agreed with
  the v13 session as E2's gate.

**Run 4 (PRE-REGISTERED 2026-10-30, drafted 04:00 IST before any run-4
number): the clean P test.** Run 3's P was invalid. With `label_marker` in
training, plain echo reached 0.999 on label_category in pair 1 (bar: < 0.90
in every pair), so the tying gain had no headroom to show. Run 4 restores
run 2's training set and keeps P's frozen rule.
- **Arms:** echo vs echo_tied, 3 new seed pairs. Pair i = `--seed 2i
  --doc_seed0 (i+3)·100M --init_seed 2i` (i = 1, 2, 3); tag `b1r4p<i>`.
- **Training:** the shared tasks only (`--rule_tasks` default: letter_shift,
  number_step, reverse3, label_category, two_hop). This is run 2's set.
  Shots 1–8, 6000 × B16, lr 3e-4.
- **Eval:** the same task list as run 3. label_marker / label_marker_new
  are reported only: the format is not trained.
- **Bar P (unchanged from run 3):** d_i = echo_tied − echo on label_category,
  mean over 2 / 4 / 8 shots. Met if mean(d) ≥ +0.15 AND mean(d) ≥ 2·SD(d).
  Valid if echo's mean is < 0.90 in every pair.
- **Outcomes:**
  - P met: the v13 session runs E2, tied echo at 600M against C1-ctl on the
    M1 body (their pre-registration, EXPERIMENTS_SEMPY "E2").
  - P not met (valid): run 2's +0.39 was seed luck. echo_tied closes and
    its flag is removed.
  - P invalid again (echo ≥ 0.90 in some pair): label_category does not
    separate the arms at this size. echo_tied closes for B1 and goes to the
    LM check only if A (run 3) was met.
- **Reported only, after run 4:** the `tpp` attention reference on the same
  3 seeds' data (separate launch `b1r4tpp<i>`, after branch `v14dev` merges).
  It is the 27M version of the v13 600M gap (label_category 8-shot: T++ 0.73
  vs M1 0.29), and it shows whether attention infers unseen rule parameters
  here (bar-3 analysis under "N4").
- **Run 3's lesson, recorded before run 4:** with one related in-context
  task in the mix, plain echo learned label_category to the ceiling. At
  this size, the label_category gap in run 2 was a matter of what the data
  teaches, at least as much as of the body.

**Run 4 RESULT (4090, 2026-10-30 04:44–05:40 IST, code `d4e0f0d`): P MET.**
Logs + JSON: `logs/v14_b1r4p{1,2,3}_d4e0f0d_20261030_{0443,0502,0521}.{log,json}`.
Each value is the mean over 2 / 4 / 8 shots. letter_shift, number_step and
reverse3 are 1.00 for every arm and pair.

| pair (seed) | arm | label_category (2/4/8) | two_hop | letter_shift_ood | number_step_ood | ms/step |
|---|---|---|---|---|---|---|
| p1 (21) | echo | 0.411 (0.50 / 0.49 / 0.25) | 0.293 | 0.016 | 0.227 | 96 |
| | echo_tied | **1.000** (1.00 / 1.00 / 1.00) | 0.288 | 0.022 | 0.194 | 88 |
| p2 (22) | echo | 0.405 (0.48 / 0.48 / 0.26) | 0.277 | 0.004 | 0.094 | 96 |
| | echo_tied | **1.000** (1.00 / 1.00 / 1.00) | 0.280 | 0.001 | 0.075 | 87 |
| p3 (23) | echo | 0.440 (0.52 / 0.54 / 0.26) | 0.294 | 0.031 | 0.093 | 96 |
| | echo_tied | **0.662** (0.77 / 0.76 / 0.45) | 0.297 | 0.045 | 0.088 | 87 |

- **P:** d = +0.589, +0.595, +0.222 (mean **+0.469**, SD 0.214, 2·SD
  0.427). **MET**, and valid: echo is < 0.90 in every pair (0.41–0.44).
- label_marker / label_marker_new (format not trained): every arm is at
  in-context chance (≤ 0.47). Reported only.

**Reading:**
- **On run 2's training set, tying makes the lookup circuit form.** Tied
  formed it in 4 of 4 seeds (run 2: 0.857; run 4: 1.00, 1.00, 0.66). Echo
  formed it in 0 of 5 (run 1: 0.68 partial; run 2: 0.47; run 4: 0.41, 0.41,
  0.44, at in-context chance 1/n_cat). This is not seed luck.
- **The training mix flips it.** With label_marker added (run 3), echo
  formed it in 2 of 3 seeds and tied in 1 of 3, and tied was worse on unseen
  items. Why the marker task hurts the tied arm is open (3 seeds per cell;
  the two-word items put the category word 1–2 tokens before "is").
- Unseen rule parameters are still at chance or below for both arms (bar 3 in
  its strong form is untouched).
- Tied is 9 % faster per step (one D × D projection fewer).
- **Pre-registered consequence:** P met, A (run 3) not met. Per run 3's
  map, the tie goes to the LM check framed as an abstraction / recall tweak,
  not as similarity lookup. The v13 session runs **E2** (tied echo 600M vs
  C1-ctl, M1 body, EXPERIMENTS_SEMPY "E2"). At 600M a single run per arm
  meets the same per-seed risk: v13's label_category gaps (M1 0.29, C1-ctl
  0.55) may themselves be one seed each.

**Run 3 candidates (from the v13 review, 2026-10-30; NOT pre-registered).**
(a) was adopted into run 3 above; (b) is still open.
They put category membership in the context without turning the task into
two_hop. Choose after run 2.
- **(a) Surface-feature categories.** Each item is a novel multi-token
  string, and its category is a shared marker inside it, e.g. `zq-kel ->
  alpha, zq-mor -> alpha, vb-tun -> beta; zq-pli ->`. Every item is unseen,
  and membership is visible in the context. Solving it takes one
  partial-match lookup: exactly the "similarity lookup" reading of bar C.
  The marker's position varies across episodes, so copying the first token
  does not work.
- **(b) Held-out labels and pairings.** Test labels come from a vocabulary
  never used as labels in training, and the category pairs in a test
  episode never co-occurred in training. This shows whether the binding is
  a general copy mechanism or is tied to particular label embeddings and
  combinations. Only the split changes.

**Reference numbers at scale** come from the v13 session's probe on the
600M / 3B checkpoints (PAM, M1, Transformer++, Mamba-2), so "beats the
transformer at learning" has a number.

**Why it matters for v14.** N6 (operator memory) and N4 (fast weights in
the unit) are learners by construction: B1, not B0, is their primary test.

**echo_tied VERDICT: CLOSED, FAIL at 600M (v13 E2, 2026-10-30, run by the
v13 session on the 4090; EXPERIMENTS_SEMPY "E2").** Tied echo on the M1 body
with C1-ctl's exact two-stage recipe, against C1-ctl, 400 trials both:
label_category 8-shot 0.365 vs 0.500 (d −0.135; bar +0.15), 2/4-shot
0.615 / 0.67 vs 0.593 / 0.68. Guards failed too: holdout 3.5298 vs 3.4743
(+0.055, behind at every validation), recall 0.602 vs 0.825, learning probe
0.177 vs 0.194. Stage A matched (3.9255 vs 3.9231), so the loss is in stage B,
not the box. **Lesson:** a clean micro-bench effect (B1 run 4: circuit in 4 of
4 seeds vs 0 of 5) did not transfer to real text at 600M, where tying hurt LM
and recall. Micro-bench wins are leads, not evidence; the LM check decides.
Code removed from v14 (last at `5100fd7`), v13 flag removed (`26ef94a`).

### B2 — inference beats selection (bar 3, pre-registered 2026-10-30 ~06:45 IST)

**Why.** B1's unseen parameters (shift 4–5 after training on 1–3) ask for
extrapolation from 3 trained values. Every arm, N6 included, falls back to a
trained shift. That says little about any body, because selecting one of 3
memorised tables is the cheap solution (bar-3 analysis under "N4"). B2 makes
inference the cheap solution. It trains on many parameter values and holds a
few out from inside the range, so a model that selects a memorised neighbour
is wrong and only a model that infers the parameter from the examples is right.

**Tasks (`v14/rules_ood.py`, selftest `test_b2_splits`).** Same prompt formats
as B1, 8 candidates, chance 0.125.

| task | rule | trained values | held out (`<task>_new`) |
|---|---|---|---|
| shift_wide | letter shift by k | k 1–12 without 5, 9 | k = 5, 9 |
| step_wide | arithmetic sequence, step d | d 2–20 without 7, 13, 17 | d = 7, 13, 17 |
| affine_map | x -> a·x + b (x 1–30) | a 1–3, b 0–20 without 7, 13, 17 | b = 7, 13, 17 (any a) |

The candidates always include the answers that the nearest trained
parameter would give (k ± 1, d ± 1, b ± 1, a ± 1; present in 600 of 600
checked episodes). A model that falls back to a neighbour scores ≈ 0, below
chance.

**Run 1 (calibration, one seed; runs when a GPU frees).** `--rule_tasks
shift_wide,step_wide,affine_map` (B2 only), shots 1–8, 6000 × B16, lr 3e-4,
`--seed 31 --doc_seed0 700000000 --init_seed 31`. Arms: **tpp** (attention
reference, O(T)), echo, n1sum (n1sum is the best memory arm in B0 run 2's
descriptive read; echo_tied dropped before the run, closed with v13 E2). Every B1 task is also evaluated (untrained
here, reported only).
- **L, learnable:** tpp ≥ 0.90 on the trained-value cells (mean of
  shift_wide, step_wide, affine_map over 2 / 4 / 8 shots). If not, attention
  cannot learn the tasks in 6000 steps either. Rerun with 3× steps before
  reading anything.
- **I, inference measurable:** tpp ≥ 0.40 on the held-out cells (mean of the
  three `_new` tasks). If L holds and I fails, attention at 27M also selects
  instead of inferring. B2 then cannot judge bar 3 at this size: say so,
  and move the question to scale (the v13 600M learning probe) or to other
  task families.
- **If L and I hold, B2 is the bar-3 bench.** For each O(1) arm, report the
  held-out mean and the gap closed, G = (arm − echo) / (tpp − echo). Run 1
  has no pass bar (one seed, a calibration, like B1 run 1). The candidate bar
  (paired seeds, as in B1 run 3) is written after run 1, from its numbers,
  before any candidate runs on B2.
- **Predictions:** tpp passes L and is partly above I (shift_wide_new
  highest, affine_map_new lowest). The O(1) arms pass the trained cells and
  sit below chance on `_new` (they pick a neighbour).

**Reference with no B2 training (v13 session, 600M / 3B checkpoints,
inference only, 400 trials × 2/4/8 shots, seeds 5000–5399, `a214002`; JSONs
`logs/learning_probes/v13_sempty_*_b2_t400.json`).** Mean / base / `_new`:
T++ 600M 0.048 / 0.056 / 0.040; T++ two-stage 0.053 / 0.059 / 0.048; M1
0.037 / 0.045 / 0.029; M2 0.046 / 0.053 / 0.038; C1-ctl 0.038 / 0.050 /
0.026; c1m2 0.071 / 0.084 / 0.057; T++ 3B 0.088 / 0.111 / 0.064; Mamba-2 3B
0.094 / 0.118 / 0.071; PAM 3B 0.066 / 0.078 / 0.054 (chance 0.125).
- Every model is below chance: it picks an answer next to the right one.
  More shots barely help.
- **Difficulty control:** without B2 training, `_new` sits 0.015–0.047 below
  base in every model (SE ≈ 0.006 per half). The held-out values are a
  little harder in themselves. Subtract this before reading a trained-vs-
  held-out gap at 27M (it is small next to the 0–1 range of trained models).
- **Caveat on reading "below chance":** the nearest-parameter decoys are
  also the answer ± 1 step (± 1 letter, ± 1 number). So a neighbour pick
  can be selection of a memorised parameter or imprecise inference of the
  right one. B2 tests *exact* inference and does not separate those two.
  L and I do not depend on the distinction. The affine a ± 1 decoys (answer
  ± x_q) are the only ones that separate a wrong slope from an off-by-one.

**Run 1 RESULT (6000, 2026-10-30 09:27–09:54 IST, code `bf59754`, one
seed; worktree `~/Development/qllm_v14` on the 6000): L MET, I MET, so B2 is
the bar-3 bench. The O(1) arms match attention on numbers and beat it on
letters.** Log + JSON: `logs/v14_b2_bf59754_20261030_0357.{log,json}` (6000
box clock, UTC). Mean over 2 / 4 / 8 shots, per shot in brackets.

| arm | trained values (3 tasks) | held out (3 tasks) | shift_wide_new | step_wide_new | affine_map_new | ms/step (6000) |
|---|---|---|---|---|---|---|
| tpp | 0.999 | 0.666 | **0.02** (0.01 / 0.01 / 0.03) | 0.99 | 0.99 | 61 |
| echo | 1.000 | **0.739** | **0.33** (0.20 / 0.31 / 0.47) | 0.90 (0.98 / 1.00 / 0.72) | 0.99 | 100 |
| n1sum | 1.000 | 0.731 | 0.28 (0.17 / 0.28 / 0.41) | 0.92 | 0.99 | 100 |

- **L:** tpp 0.999 ≥ 0.90. **I:** tpp 0.666 ≥ 0.40. Both met.
- **Numbers (step_wide_new, affine_map_new):** every arm infers the
  held-out step or offset almost exactly. The nearest-parameter decoys
  (± 1) are in every candidate set, so this is exact inference, not
  selection. The difficulty offset (v13 probe, ≤ 0.05) is negligible here.
- **Letters (shift_wide_new):** tpp picks the neighbouring trained shift
  (0.02, below chance): it memorised one table per trained k. Echo and n1sum
  partly infer the held-out shift, and they improve with more examples
  (echo 0.20 → 0.31 → 0.47), the signature of inference from the context.
- **My prediction was wrong** (O(1) arms below chance on `_new`). This is
  one seed: the replication (run 2) decides before any claim.
- Gap closed G is not defined (echo > tpp). shift_wide_new is the open cell:
  0.33 for the best body, 1.0 possible.

**Calibration: tpp on B1 run 4's data (6000, `bf59754`, 3 seeds, logs
`logs/v14_b1r4tpp{1,2,3}_bf59754_*.{log,json}`).** label_category 0.42 /
0.42 / 0.43 (chance; per shot 0.50 / 0.51 / 0.25): attention does NOT form
the label circuit at 27M, like plain echo (0.41–0.44); tied echo did (then
failed at 600M). So micro-bench label_category does not predict the 600M
label_category gap (T++ 0.73), which is one more reason E2 did not transfer.
Unseen rule parameters: number_step_ood tpp 0.27–0.38 vs echo 0.09–0.23
(above chance 0.125); letter_shift_ood 0.00 for everyone. tpp 52 ms/step.

**Run 2 (PRE-REGISTERED 2026-10-30 ~10:10 IST, before any run-2 number):
replication in seed pairs.** Question: does echo infer held-out letter
shifts where same-size attention only selects, across seeds?
- **Arms:** echo vs tpp, 3 pairs, same data and init seed within a pair:
  pair i = `--seed 4i --doc_seed0 (i+7)·100M --init_seed 4i` (i = 1, 2, 3;
  run 1 used 31, so these share no seed with it),
  tag `b2r2p<i>`. B2 training only, 6000 × B16, lr 3e-4.
- **Bar R (shift_wide_new):** d_i = echo − tpp, mean over 2 / 4 / 8 shots.
  Met if mean(d) ≥ +0.15 AND mean(d) ≥ 2·SD(d).
- **Inference signature (reported, not a bar):** echo's shift_wide_new
  8-shot minus 2-shot > 0 in every pair.
- **Reported:** the other `_new` cells, trained-value cells, ms/step.
- **Outcomes:** R met: at 27M, the O(1) echo body infers unseen letter
  shifts that a same-size transformer cannot. The next step is a 600M check
  with B2 data mixed into training for both bodies (design then, with the
  v13 session). R not met: run 1 was seed luck, and B2's letters cell is
  the target for new units (N4).

**B2 state sweep (PRE-REGISTERED 2026-10-30 ~10:55 IST, before any number):
does a smaller memory make the body infer rules better?** From the step back
(SCRATCHPAD): attention keeps every example and memorises a table per
parameter (tpp shift_wide_new 0.02). A fixed-size state cannot hold the
examples, so it may be pushed to hold the rule. If the bottleneck causes the
abstraction, a smaller state should infer held-out rules better.
- **Arms:** echo@32x12 (12 heads × 32: state 12,288 per layer), echo (6 ×
  64: 24,576), echo@128x3 (3 × 128: 49,152). Width 384 and params (27.59–
  27.67M) are the same, so only the memory size changes (`x<heads>` suffix,
  this commit).
- **Seeds:** 2 launches, `--seed 5j --doc_seed0 (j+10)·100M --init_seed 5j`
  (j = 1, 2), tag `b2statej`. B2 training only, 6000 × B16, lr 3e-4.
- **Bar S (shift_wide_new, mean over 2/4/8 shots and both seeds):** the
  hypothesis holds if hd32 ≥ hd64 ≥ hd128 AND hd32 − hd128 ≥ +0.15. It is
  refuted if hd128 − hd32 ≥ +0.15 (more memory infers better). Anything
  else: no effect at this range.
- **Reported:** trained-value cells, the number `_new` cells (and their
  8-shot dip), ms/step. **These run in parallel with B2 run 2 on the 4090,
  so ms/step is not comparable** with solo runs.

### N1 — phase interference revived (v11 E3 on the lean body)

**Why.** The only architecture change that beat the v7 baseline (−1.11 PPL),
and the one mechanism that is uniquely PAM. Never retested after the lean rewrite.

**Mechanism (`v14/units.py` `PhaseStatesPAM`, flags `phase_states=2`,
`phase_read`).** K = 2 delta notebooks per head. They share q, k, v and the
write/erase gates; only the fade differs (decay offsets −2 / +2 around the
head's dt, as v11 E3 and v13 A2). Each is read with the same query, then:

| arm | read | new params |
|---|---|---|
| `n1sum` | `y = Σ_k a_{h,k} y_k`, a = 1/K at start | H·K |
| `n1gate` | `y = (1/K) Σ_k (1 + c_k(x)) y_k`, `c = W x` | D·H·K + H·K |
| `n1phase` | `y = (1/K) Σ_k R(φ_k(x)) y_k`, `φ = W x`, R turns every channel pair by φ (`sempyt.rotate`) | D·H·K + H·K |

W is zero-init, so all three start as the plain average (selftest: identical
logits at start). `gate` and `phase` have identical parameters: one scales
each read by a real number, the other turns it. A `cos φ` gate was rejected:
zero gradient at φ = 0. State is 2× the control in all three. Decode is
implemented (O(1), K notebooks carried); prefill == decode to 7e-7.

**Selftest (`v14/selftest.py`, CPU, 2026-10-29): ALL PASS.** Switches off ==
v13 (bit-identical); three reads identical at start; prefill == decode for
each read across the 64-token sub-chunk edge; every new parameter gets a
gradient; nonzero angles make phase ≠ gate.

**Bars (micro-bench, written before any run).** Headroom cells H = the B0
cells where the echo control scores < 0.90 (mean over positions).
- PASS: mean_H(n1phase) − mean_H(n1sum) ≥ +0.10 AND
  mean_H(n1phase) − mean_H(n1gate) ≥ +0.05.
- If H is empty, memory is not the limiter at this size; N1 is then judged
  on the LM check only.
- Also reported: ms/step of each arm vs echo.

**Bars (LM, 150M holdout, same batches as control).** n1phase ≤ control −
0.02 nats AND ≤ n1sum − 0.01 AND ≤ n1gate − 0.01. Step time ≤ 1.35× control.
**Kill:** n1phase within 0.01 of n1gate (turning adds nothing over a real weight).

**B0 run 2 (2026-10-30): no judgement** (the bench was invalid: echo never
learned lookup). Descriptive only: n1sum / n1gate did partial lookup
(facts2 0.79 / 0.78, facts4 0.50 / 0.52, against 1/n = 0.50 / 0.25), and
n1phase fell back to echo's level (0.51 / 0.29). Turning cancelled the
second notebook's gain, the opposite of the N1 hypothesis. B0 run 3
(`--aux_weight 0.1`) carries a stop rule that decides N1 if the bench is
invalid again (section "B0").

**N1 VERDICT: CLOSED, FAIL (B0 run 3 stop rule, 2026-10-30).** B0 was
invalid twice (neither echo nor the tpp reference learned lookup), and by
the pre-registered rule n1phase had to reach n1sum on mean(facts2, facts4)
to earn an LM check: 0.370 vs 0.380. In both runs turning never helped. In
B2 run 1, n1sum (0.731 held out) did not beat echo (0.739). Code removed;
it is at the commit listed in the index.

### N2 — phase-modulated activation

**Why.** Coded since v7, never measured. A neuron whose output phase depends
on its own strength: "strong" and "weak" signals point in different
directions, so later layers can tell them apart by interference.

**Mechanism.** `act(z) = |z|·σ(β|z|+b) · e^{i(arg z + α|z| + c)}`, α, c zero-init (starts as ModSwish).

**Arms.** complex lean arm with ModSwish vs the same with `phase_mod`. Complex
arm only, so this is a mechanism test, not a speed contender, and is compared
against its own complex control.

**Bars.** 150M holdout ≤ complex control − 0.02 nats. If it passes, it is the
first evidence that complex units carry something real units cannot, and N3
gets the complex arm too. **Kill:** within ±0.01 (α stays near 0).

### N3 — formula-weight neuron: folded into N2 (analysis, 2026-10-29)

The idea (user, 2026-10-29: "a neuron is more than weight and bias"): each
weight is K interfering parts and the input picks how they combine,
`W(x) = Σ_k c_k(x) U_k V_kᵀ`.

**It reduces to a known unit.** Stack the bases, `U = [U_1 … U_K]`,
`V = [V_1 … V_K]`, and repeat each `c_k` over its rank block:

    W(x) x = Σ_k c_k(x) U_k (V_kᵀ x) = U · ( c(x) ⊙ Vᵀx )

That is a low-rank bottleneck with a multiplicative gate on it: the GLU
family, which the CGU/MLP already is (silu gate × linear). Param-matched
(D 588, hidden 1764, K 4 → rank 110 each) it costs the same FLOPs as the
dense layer and adds only another gate. In the real arm it is not a new
neuron. In the complex arm, `c_k = e^{iφ_k}` turns the bottleneck instead of
scaling it, which is exactly the question N2 already asks (does a phase gate
carry something a real gate cannot?). So N3 is not run separately.

**What this says about "new neurons".** Any unit whose output depends only
on the current input, however its weights are composed, is a function
`f(x)`, and a gated MLP already approximates it. A neuron that is really
different needs something that persists across time: **its own state**
(N4: memory inside the unit) or **its own dynamics** (N5: an oscillator with
a phase that evolves). That is where v14 looks for the new species.

### N4 — living neuron (design only; after B0, N1–N3)

Each MLP unit keeps a small fast weight written while reading (delta rule),
with write strength set by echo's prediction error (surprise). Memory moves
into every unit instead of a separate notebook.
**Prior art is close:** fast weights (Ba 2016), TTT-MLP (Sun 2024), Titans
(2025). The new parts would be the echo-surprise gain and the complex/phase
form. **Primary test: B1** (rule induction, goal bar 3). A unit that
updates itself while reading is a learner, so its claim is learning, not
recall; B0 is secondary.
**Lesson to respect:** V6 salience memory was inert, so the fast weight must
carry information the slow path cannot get (pands lesson 1).

**Bar-3 analysis before any N4 run (2026-10-30, after B1 runs 2–3; design
note, not pre-registered).** Every arm, N6 included, is below chance on
unseen rule parameters: the models pick a trained shift. Two causes are
mixed up in that number:
1. *The bench.* Training draws shift k from only 3 values (1–3) and the test
   asks for 4–5, i.e. extrapolation from 3 points. Selecting among 3
   memorised tables is cheaper than inferring the rule, so no body has a
   reason to learn inference.
2. *The body.* The delta notebook is already a least-squares fast weight
   (TTT-linear-like): S fits values from keys in context. If keys and values
   carried a linear ordinal code (letters on a line), S k(x_q) would
   extrapolate a shift from 2 examples. Nothing in training forces that code.
**Next steps, in order:** (a) the `tpp` reference on B1's unseen parameters
(run 2's data, one ~10 min arm). If attention also fails, cause 1 dominates
and B1-OOD cannot judge N4. (b) Then a bench where inference beats
selection: many training parameters with some held out for interpolation
(e.g. shift 1–12 without {5, 9}; affine number maps y = a·x + b with held-out
(a, b) pairs), tpp as the reference (**built: B2**, section "B2"). (c) Only
then an N4 unit, judged there.

### N5 — resonant binding (design only)

Units are oscillators (amplitude, phase, natural frequency, coupling). A
cheap sync step pulls co-occurring features to a shared phase (binding by
synchrony), and the phase selects where the binding is written among the
matrix states (N1's read in reverse). **HSB constraint:** never superpose
bindings in a vector. **Prior art:** AKOrN (Kuramoto neurons, vision, 2025),
Complex AutoEncoder / Rotating Features (object binding by phase, vision).
Not known in an O(1) recurrent LM. Micro-bench only, on B0's bound-attribute
case, after N1 has a verdict.

### N7 — in-context n-gram memory (design only, user idea 2026-10-30)

**Why.** Today's results split two skills. The O(1) body is good at
*abstraction*, carrying one rule across examples (B2 run 1: echo infers
held-out letter shifts, tpp picks memorised tables). Attention is good at
*exact lookup*, finding an item and copying what followed (600M
label_category, T++ 0.73 vs PAM 0.41). Patching lookup into echo failed (tied
echo, E2). User's idea: let the body think, and borrow lookup from an n-gram
system.

**Mechanism (sketch).** A fixed-size hash table per sequence, S slots. At
each token: write `hash(last n tokens) -> token that followed`, overwriting
on collision; read `hash(current last n tokens)` and give the stored
token's embedding (plus a found / not-found bit) to the body as an extra
input, gated by the body. Constant time and fixed memory per token, so still
O(1). Exact entries do not fade with distance, unlike the recurrent state,
so it should hold past the trained context length (goal note, 2026-10-30).
Several n (2, 3, 4) as separate tables.

**Limits, said up front.** It only serves exact repeats (names, numbers,
"key: value" facts). label_category needs similarity (dog ~ cat), which
exact n-grams do not give. Prior art: cache LMs (Grave et al. 2017), kNN-LM,
Infini-gram (corpus-level), hashed n-gram embeddings. The new part to
claim is a bounded, write-while-reading n-gram table inside an O(1)
recurrent body, measured past the trained context.

**Design constraints from v13's insights (EUREKA.md, 2026-10-30).** (a)
Give the exact store its OWN reader in each block that uses it, not M2-style
shared reads: shared reads seemed to dilute the lookup circuit
(label_category lifts on M1, 0.29 -> 0.55, not on M2). (b) Judge learning
after matched data exposure: reasoning-data gains fade in a web-heavy stage
B. (c) At the 600M check read WikiText, not just the web holdout: T++ gains
WikiText from stored facts (aptitude 66.2 -> 63.8), PAM does not, so the
exact store may be where that gap closes. (d) Keep q != k for anything that
serves recall (E2).

**Tests (to pre-register before any code).** (1) A dense-supervision
copy/recall micro-bench (B0 broke on answer-only supervision), inside and
beyond the trained length, tpp as the reference. (2) B2 must not drop
(abstraction kept). (3) 600M LM check with the v13 session: holdout, the
recall probe past 2048, and the long-context learning probe they are
building.

### N8 — self-opened memory (learned critical period; handed over by the v13 session, 2026-10-30)

**Observation (v13).** The two-stage schedule (language first, recall docs
later, LR re-warmed) helps only PAM, not T++. It looks like a critical
period: the notebooks need mature representations before they can learn what
to store.
**Mechanism (sketch).** Replace the fixed stage cut with a gate the model
opens itself: each memory write is scaled by the body's own maturity
signal, e.g. its prediction confidence on the current token (or a slow
running average of it), so writes stay small while language is immature
and open as it matures. In the two-memory design (N7 + rule memory) this is
the write gate of both stores; per-token surprise then decides *what* to
write, maturity decides *how much*.
**Test (to pre-register).** One-stage training with the gate vs two-stage
without it, at equal data: does the gate recover the two-stage recall gain
with no schedule? Micro first (dense copy bench), then a v13 600M check.

### N6 — operator memory (candidate, 2026-10-29; reviewed by the v13 session)

**Idea.** Prediction applies a transformation inferred from the context to
a new input, so the answer never has to be in the context: rule
application (goal bar 3), not lookup.

**The claim, stated precisely (from the cross-review).** A delta /
outer-product state is already an inferred linear operator (online least
squares), and `S k_new` also "applies a rule" to unseen inputs. So "items
vs operators" alone is not the difference. The difference is the
**inductive bias**: a phase-only diagonal operator has d numbers per head
instead of d², so it can be inferred from 1–2 examples; relations compose
by adding phases; and it generalises by construction when a rule *is* a
rotation.

**Sketch.** Tokens as complex vectors. A relation between an input and an
output is a rotation per channel, `Δφ = arg(v) − arg(k)`. The state keeps a
confidence-weighted **circular mean** of these rotations (O(d) per head,
O(1) per token). A new input is answered as `k_new · e^{iΔφ}`. Open
questions: which positions play input / output (learned role gates),
several operators at once (selected by the input), magnitude changes.

**Prior art to cite:** FHRR (Plate, phase-only holographic binding),
Associative LSTM (Danihelka et al. 2016: complex HRR inside an LSTM, very
close in spirit), RotatE (static knowledge-graph rotations), AKOrN (phase
dynamics). Not known anywhere: a circular-mean estimator of the relation,
inferred in context, used as the read-out operator of a recurrent LM.

**Implementation (`v14/units.py` `OperatorMemory`, flag `opmem`, 2026-10-29).**
Sits on top of the echo body and reads into the head next to echo. Channel
pairs are complex numbers (sempyt split-complex, named end to end):

    a_t = W_in h_{t-1},  b_t = W_role ê(x_t),  r_t = b_t · conj(a_t)
    S_t = γ_t S_{t-1} + w_t r_t          (sempyt.scan, per-step γ)
    q_t = W_in h_t,  p_t = q_t · S_t / (|S_t| + 1)

`opmem='phase'` stores `r / |r|` (rotation only: S is a weighted circular
mean). `opmem='diag'` stores `r` from per-head RMS-normalised a, b (rotation
and magnitude: a plain diagonal memory, same parameters): the control for
the "rotation only" bias. The read is RMS-normalised per head, then a
zero-init projection adds it to the pre-logit vector. +4·D² + 2·D·H
parameters (≈ 0.6M at D 384). Decode is O(1): carries (h_{T}, S_T).

**Selftest (2026-10-29): ALL PASS.** At start == the echo body (0.0 diff);
prefill == decode over 150 tokens (≤ 8e-6, both modes); every parameter
gets a gradient; **mechanism test:** with W_in = W_role = I and examples
whose outputs are their inputs turned by a hidden θ per channel (uneven
magnitudes), the read of a NEW input equals that input turned by θ:
cos 1.0000 (phase), 0.9879 (diag). sempyt fixes made along the way:
`scan` with a per-step decay was wrong past ~260 steps (fixed, `2790c69`).

**Per-task predictions and bars for B1 (FROZEN 2026-10-29, before any N6
run).** Arms trained on identical data: echo (control), op (N6), opdiag
(N6 without the rotation-only bias).
- **PASS (the N6 claim):** mean over letter_shift and number_step at 4 and
  8 shots: op ≥ echo + 0.15 AND op ≥ opdiag + 0.10.
- label_category: many-to-one, not a rotation: predicted |op − echo| ≤ 0.05.
- reverse3: positional, not a rotation of content: predicted |op − echo| ≤ 0.05.
- two_hop: composition, reported only.
- Also reported: ms/step vs echo; B0 memory cells (op must stay within
  0.05 of echo on facts 1–8).
- Context: at scale (v13 session's probe, 600M / 3B, PAM, M1, echo,
  Transformer++) no model applies an in-context rule: letter_shift ≤ 0.13,
  number_step ≤ 0.053 (mostly "last + 1"), two_hop at in-context chance.

**Original per-task reasoning (cross-review):**
- letter_shift, number_step: ordinal shifts, which are rotations if the
  model puts ordinals on a circle (Chrono-like). N6 should beat the
  echo / M1 small bodies clearly here.
- label_category: many-to-one, not a rotation. Expected to FAIL (a limit
  of the idea, worth showing).
- two_hop: composition, the informative middle case.
- reverse3: positional, not a rotation of content. No gain expected.

**B1 run 1 result and verdict (2026-10-30 00:30 IST; table in "B1").**
- **PASS bar: NOT MET.** On letter_shift and number_step, op = echo =
  opdiag = 1.00 (the ceiling: every arm, delta included, solves them in
  weights). The cell cannot show the claim either way. This is a bench
  defect, not a mechanism failure. The original claim (an inferred
  rotation applies to a new input) moves to B1 run 2, on rule parameters
  never seen in training.
- **reverse3 prediction held** (op − echo = 0.00).
- **label_category prediction WRONG, in N6's favour:** op 1.00 / 1.00 /
  1.00 vs echo 0.76 / 0.77 / 0.52 (+0.24 / +0.23 / +0.48). But opdiag is
  also 1.00, so the **rotation-only bias is not what helps.** What op and
  opdiag share, and echo lacks: a *diagonal* (per-channel) memory from the
  previous hidden state to the next token, with **one shared projection
  for key and query** (`a = W_in h_{t−1}`, `q = W_in h_t`) and a
  magnitude-normalised read. Echo pairs the same two things, but in a full
  delta matrix with separate W_q and W_k. Two candidate reasons, to be
  separated: (i) the shared projection makes "same category" match by
  construction; (ii) a diagonal state binds a few category-level features
  cleanly, where a matrix state also binds item detail.
- **two_hop:** op at in-context chance, like every arm.
- **Cost:** none. op 94 ms/step vs echo 99 (+0.5M params).
- **Status: exploratory, not pre-registered.** The label_category gain must
  replicate on a new training seed (B1 run 2) before it counts.

**N6 VERDICT: CLOSED, FAIL (B1 run 2, 2026-10-30 01:50 IST).** Every
pre-registered test failed or could not be read:
- the run 1 PASS bar sat at the ceiling;
- replication R was not met (op 0.427 vs echo 0.471 on label_category,
  opdiag +0.116 < +0.15);
- O was not met: on unseen rule parameters op is 0.037, echo 0.063.

The rotation-only bias never beat the plain diagonal, in either run. The
maths stays above. The code is removed from v14; it is reproducible from
the commit in `EXPERIMENT_INDEX.md`. What survives is the lead it
uncovered: **a tied query/key projection** (echo_tied).
---

## Order and machines

Updated 2026-10-29 23:50 IST.

1. **B1 run 1** (rules; N6 first test), 4090, running since 23:29 IST.
   Arms echo, op, opdiag, n1phase, delta, ~11 min each. It was meant to start
   at ~21:45; the v13 session's T1 tmux stayed open two hours after its run
   ended (a trailing `exec bash`, now removed on their side).
2. **B0 run 2** (memory, sizes from 1; N1 judged), 4090, starts when B1's
   tmux ends. Arms echo, delta, n1sum, n1gate, n1phase, op.
3. **LM check** (stage 3) for whichever of N1 / N6 passes, on the 4090.
   It needs a v14 hook in the v13_sempty trainer, not built yet.
4. **N2** LM check (complex arm) on the 4090.
5. Anything that passes stage 3 queues on the 6000. The v13 session's chain
   holds it until ~03:30 IST 2026-10-30 (C1 control, then C1 on M2).
