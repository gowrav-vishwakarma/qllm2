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

**Box confound CLEARED (v13 box control, `38e3c29`): the FAIL below stands.**
(Was held open 2026-10-30 ~16:30 IST:) v13 found a possible 4090 confound. E2
and g1c (unrelated changes, both trained on the 4090 with grad checkpointing)
lose the same ~0.05 holdout to the 6000 control, tracking it through stage A
and the first ~3,000 stage-B steps, then drifting up ~0.07. v13's box control
(C1-ctl stage B rerun on the 4090) decides. If the box drifts, E2 is re-run on
the 6000 (code at `5100fd7`). Micro-benches are unaffected (every pair runs on
one box, no grad checkpointing).

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

**Run 2 RESULT (4090, 2026-10-30 10:20–12:28 IST, code `9e0a3fd`, shared
with other jobs so ms/step is not comparable): R MET. Echo infers held-out
letter shifts where same-size attention only selects, in 3 of 3 seed pairs.**
Logs + JSON: `logs/v14_b2r2p{1,2,3}_9e0a3fd_20261030_*.{log,json}`.

| pair (seed) | echo shift_wide_new (2/4/8) | tpp shift_wide_new | d | step_wide_new echo / tpp | affine_map_new echo / tpp |
|---|---|---|---|---|---|
| p1 (41) | 0.141 (0.08 / 0.13 / 0.21) | 0.017 | +0.123 | 0.93 / 0.99 | 1.00 / 0.96 |
| p2 (42) | 0.259 (0.14 / 0.25 / 0.39) | 0.025 | +0.234 | 0.95 / 1.00 | 1.00 / 0.96 |
| p3 (43) | 0.339 (0.26 / 0.37 / 0.39) | 0.007 | +0.333 | 0.95 / 0.98 | 0.91 / 0.97 |

- **R:** mean d **+0.230** ≥ +0.15, SD 0.105, 2·SD 0.210 → **met**.
- **Inference signature:** echo's 8-shot > 2-shot in every pair (0.21 > 0.08,
  0.39 > 0.14, 0.39 > 0.26). tpp stays at the neighbour-pick floor (≤ 0.03).
- Trained values ≈ 1.00 for both arms; held-out numbers ≈ 0.91–1.00 for both.
- **Replicated accident:** echo's held-out steps dip at 8 shots (0.80 /
  0.85 / 0.90, vs ≥ 0.97 at 2 shots); tpp does not.
- **Claim at 27M:** the O(1) echo body infers an unseen rule parameter from
  the examples, and a same-size transformer does not (it memorises one table
  per trained value). Level is partial (0.14–0.34). Next: the state sweep
  (does the bottleneck cause it?) and a 600M check with B2 data in training.

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

**State sweep RESULT (4090, 2026-10-30 10:47–13:34 IST, code `dd5b174`, 2
seeds): by the frozen rule S is REFUTED (hd128 − hd32 = +0.168 ≥ +0.15), but
the honest reading is NO reliable state-size effect: the two seeds order the
sizes in opposite directions.** Logs: `logs/v14_b2state{1,2}_dd5b174_*.{log,json}`.

| shift_wide_new (2/4/8) | hd32 ×12 (12k state) | hd64 ×6 (25k) | hd128 ×3 (49k) |
|---|---|---|---|
| seed 51 | **1.000** (1.0 / 1.0 / 1.0) | 0.924 | 0.500 (0.32 / 0.49 / 0.70) |
| seed 52 | 0.162 | 0.086 | **0.997** (0.99 / 1.0 / 1.0) |
| mean | 0.581 | 0.505 | 0.749 |

Trained values 1.000 everywhere; held-out numbers 0.92–1.00; the 8-shot step
dip appears at every size (0.81–0.92).
- **Bottleneck hypothesis (step back, point 2): not supported.** Neither
  "smaller abstracts better" nor "bigger is better" holds across seeds.
- **New finding: the O(1) body can solve held-out shift inference COMPLETELY**
  (1.000 and 0.997 here). Whether it does is per seed, all or nothing, like
  the label circuit in B1. Count over every echo-family B2 run so far: 3 of
  10 runs ≥ 0.9 on shift_wide_new; tpp 0 of 4 (≤ 0.03). The metric that
  matters is the rate at which the inference circuit forms, and what makes
  it form reliably (seed, data order, a critical period?).

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

**Primary target: format-free exact copy (v13 label_exact, 2026-10-30).**
With the query repeated exactly from the context ("... truck is silver ...
truck is"), PAM copies poorly: M2 c2m2 0.58 / 0.73 / 0.54 and M1 C1-ctl
0.61 / 0.63 / 0.59 at 2 / 4 / 8 shots, vs T++ 0.86 / 0.92 / 0.86. Yet PAM
scores 0.94 on the recall probe in its trained recall-doc format: its copy
skill is format-bound. An n-gram store does this copy by construction, in
any format. **N7's bar at 600M (to freeze before the run): label_exact ≥
T++ inside 2048 tokens, and past 2048 (examples spread to 4096 / 8192, the
v13 LCL format) ≥ in-context chance + 0.15, i.e. the item -> label mapping
survives; recall and holdout not worse.** Why the second form: v13's LCL
(`4bc181d`) shows that past 2048 PAM keeps the label SET but loses the
mapping (label_category 0.25 / 0.20 at 4096 / 8192 = in-context chance 0.25),
while T++ loses the context entirely (0.12 / 0.13 = candidate chance). So
"≥ T++" past 2048 is trivially met and says nothing; only beating in-context
chance shows the mapping held. Exact keys that never fade are N7's case. v13's data
lever (generic repeated-text / ICL data) is complementary.

**Why PAM's lookup fails: key collisions (v13 diagnosis, `77efe61`).** Not
capacity (16 notebooks + echo, 3.3× state: exact copy 0.533 < M1 0.585; PAM
3B 0.618) and not distance (1,000 filler tokens: 0.44 -> 0.37). Accuracy
falls with the number of similar pairs in between (0.72 -> 0.29 for 0-5
back; T++ flat ~0.8); 45 % of wrong answers are the most recent label. Echo
keys at the label positions of different "X is Y" pairs have cosine
0.67-0.72 (random positions 0.31-0.38): every label is filed under the
template "<noun> is", and the delta erase makes the last write win.
**Design rules:** keys must carry the content token, not the template. N7's
hash of the last n tokens is collision-free by construction; any learned
store needs keys that see the previous tokens (Mamba-2's short conv likely
explains its 0.72) and a collision check in its selftest (key cosine at
answer positions vs random).

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

**N7 code (2026-10-30, `83f49dc`, selftest ALL PASS).** `v14/ngram.py`,
switches `ngram_orders` / `ngram_slots` / `ngram_layers`, micro arm
`echo_n7` = echo + orders (2, 3), read after the first and last block, 2^16
slots per order. Zero-init readers (bit-identical to echo at start).

**N7 micro run 1 (PRE-REGISTERED 2026-10-30 ~12:55 IST, before any N7
number).** Question: does the body use an exact in-context copy it never
trained on in that format?
- **Training:** `--rule_tasks letter_shift,number_step,reverse3,
  label_category,two_hop,shift_wide,step_wide,affine_map,repeat_copy`
  (repeat_copy = generic copying in one format). 6000 × B16, lr 3e-4.
- **Eval:** every task, plus `label_exact` (train words) and
  **`label_exact_new`** ("X is L ... X is", a format never trained, words
  never seen in training; in-context chance 1/n_items).
- **Arms:** echo, echo_n7, tpp (reference), 3 seed pairs: pair i = `--seed
  6i --doc_seed0 (i+12)·100M --init_seed 6i`, tag `n7r1p<i>`.
- **Bar X (primary, label_exact_new):** d_i = echo_n7 − echo, mean over 2 /
  4 / 8 shots; met if mean(d) ≥ +0.15 AND ≥ 2·SD.
- **Guard G (abstraction kept):** mean over pairs of the B2 held-out mean
  (3 `_new` tasks): echo_n7 ≥ echo − 0.05.
- **Reported:** label_exact, label_category, repeat_copy, tpp's level on
  label_exact_new (attention reference), ms/step (shared GPU: not comparable).
- **Outcomes:** X met and G holds: N7 goes to the v13 session for the 600M
  check (bar in "N7": label_exact ≥ T++ inside and past 2048, recall and
  holdout not worse), after wiring gradient checkpointing. X not met: the
  body does not use the exact read at this size; check the read (gate,
  injection block) before closing N7.

**N7 micro run 1: STOPPED after pair 1, inconclusive by design (not a FAIL;
2026-10-30 15:35 IST, code `73265e9`).** Pair 1 (seed 61): label_exact_new
echo 0.28 vs echo_n7 0.32 (d +0.04); in-context chance is 0.29 (1/2, 1/4,
1/8 items), so both sit AT chance in the untrained format. repeat_copy was
1.00 for both: echo already solves the training copy task alone, so the N7
reader never got a training signal to rely on. The other pairs would have
cost ~5 h (the 4090 is shared with a 600M run: ~55 min per arm) to confirm a
bench-made null. Log + JSON: `logs/v14_n7r1p1_73265e9_*` (echo, echo_n7; the
tpp arm was stopped). **N7's real test:** 600M on web text (exact repeats
past echo's reach), or a micro bench whose training needs copying echo
cannot do (many look-alike keys, long gaps).

**F1 formation bench (PRE-REGISTERED 2026-10-30 ~15:40 IST, before any F1
number; joint with v13's Q1 stage 1).** Which start or key type makes the
lookup circuit form for every seed? Bench = B1 run 4's training set (default
rule tasks), where echo formed the label_category circuit in 0 of 5 seeds and
tied echo in 4 of 4. Seeds = v13's Q1 seeds: i = 1..5, `--seed 7i --doc_seed0
(i+15)·100M --init_seed 7i` (same init and data for every arm of a seed).
- **Arms:** echo (control, run once, in the v14 slot), echo_phase (N9a, v14
  slot), echo_qinit (Q1, v13 slot), echo_sparse (S1, v13 slot, when built).
- **Formed** = label_category mean over 2/4/8 shots ≥ 0.9.
- **Bar F (per arm):** formed in ≥ 4 of 5 seeds while echo forms in ≤ 2 of 5.
  The winner (or winners) decides which key/start goes to 600M first.
- **Reported:** B1 tasks, ms/step (shared GPU, not comparable).
- This replaces N9a micro run 1 (N7's bench cannot show copy, see above).

**F1 v14 share RESULT (4090, 2026-10-30 15:33–20:10 IST, code `7c8c6e2`):
N9a (echo_phase) forms the lookup circuit in 5 of 5 seeds; echo in 1 of 5.
Bar F MET for N9a.** Logs: `logs/v14_f1p{1..5}_7c8c6e2_*.{log,json}`;
checkpoints on d_drive (`qllm_ckpt/v14/f1p*`).

| seed | 71 | 72 | 73 | 74 | 75 | formed |
|---|---|---|---|---|---|---|
| echo label_category | 0.702 | **0.910** | 0.400 | 0.723 | 0.661 | 1 / 5 |
| echo_phase (N9a) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **5 / 5** |

Other B1 tasks unchanged (letter_shift / number_step / reverse3 1.00 both;
two_hop 0.30 both; untrained formats at chance for both). So the content
phase does exactly what the collision diagnosis predicted: when "X is" keys
stop colliding, the label lookup circuit forms on every seed. v13's Q1
(echo_qinit) and S1 (echo_sparse) arms ran on the same seeds in the v13 slot
(`38e3c29`): **Q1 0/5, S1 0/5.** N9a is the only arm that meets bar F. It goes
to 600M in v13's lane through the trainer hook (`424afd6`, `--echo_phase_lags
3` / `ECHO_PHASE_LAGS=3`).

**N9a at 600M (v13 n9c, 2026-10-31, `df8aed6`; 103.5M vs C1-ctl 102.95M, 600M
tokens each, M1 body, C1-ctl recipe): LOOKUP FAIL by the pre-registered bars;
RECALL much better (exploratory, not a bar).**
- label_exact 0.472 vs 0.610 (bar ≥ 0.710); label_category 8-shot 0.318 vs
  0.500 (bar ≥ 0.650; 2 / 4 shots 0.507 / 0.502 vs 0.593 / 0.680).
- Guards held: holdout 3.4725 vs 3.4743.
- Recall 0.975 vs 0.825; **past 2048: 0.924 vs 0.597** (T++ 0.20, Mamba-2
  0.46 there): the best of any 600M model.
- Keys de-collide as designed: cosine at label positions 0.538 vs 0.715
  (random positions 0.227 vs 0.379).
- label_exact by pairs back 0..5: 0.58 / 0.54 / 0.37 / 0.43 / 0.16 / 0.12
  vs ctl 0.57 / 0.62 / 0.48 / 0.67 / 0.34 / 0.47; wrong picks are all
  in-context labels.
**Reading (with v13's B2 600M result):** PAM learns lookup and rule skills
from the formats it trains on. The micro F1 5/5 trained on label_category
itself; at 600M "<item> is <label>," is never trained, so separating keys
alone did not create the skill (the collision was a symptom). Open design
question: why exact lookup decays with pairs back once keys are distinct
(write / erase gates, or the query window reaching into the previous label).
**The recall jump is unplanned**, so it needs a pre-registered replication
with recall as the primary bar before any claim.

**n9c LM attribution (v14, 2026-10-31 08:48 IST, eval only).** Scored with
v13's `v13_sempty/tmp/attr3b.py` on v13's frozen text (60 PG-19 windows +
150 web docs, 2048 tokens) on the 4090. Log
`logs/diag/v14_attr3b_n9c_c1ctl_7d5bac4_20261031.{txt,json}`. Sign: ctl −
n9c, so positive = N9a better.

| | total | copy | seen | new |
|---|---|---|---|---|
| PG-19 | −0.006 | −0.015 | −0.023 | +0.018 |
| web | +0.005 | +0.016 | −0.008 | +0.011 |

No LM effect beyond noise (cross-run ±0.01–0.02). The recall jump does not
show up as better copy on real text: copy is −0.015 on books and +0.016 on
web. So N9a cannot be counted toward the T++ copy gap at 600M; any LM claim
would need the ≥ 1.5B-token arm.

**N9a recall replication `n9r` (PRE-REGISTERED 2026-10-31 ~02:00 IST, before
any n9r number; agreed with v13).** n9c's recall jump was unplanned, and recall
past 2048 has looked seed-lucky before, so both arms run at a NEW seed, paired.
- **Arms (4090, whole card, v13's launcher + the v14 hook, worktree on
  master):** ctl43 = C1-ctl's exact recipe (M1 body: PAM_LAYERS=3,7,11,15,
  MLP_HIDDEN=2352, echo; stage A 120M tokens dclm/fineweb/smoltalk2_mid
  0.4167/0.3333/0.25, blend warmup 30M; stage B 480M on the M1 mix with
  --init_from; LR 1.2e-3; GRAD_CKPT=1) and n9a43 = the same + ECHO_PHASE_LAGS=3.
  SEED=43 for both (init + data mix).
- **Primary bars (the screen's 40-trial recall probe, scored as in v13's n9c
  record):** recall past 2048 ≥ ctl43 + 0.15 AND overall recall ≥ ctl43 + 0.08.
- **Guard:** web holdout ≤ ctl43 + 0.01.
- **Reported:** label_diag (t200), PG-19 far gain (eval_pos_loss; n9c: +0.037
  vs +0.041), so a recall gain with no long-text LM gain is visible.
- **Outcome:** both bars met: N9a's long recall is real at 600M, and it goes
  to a ≥ 1.5B-token arm on the r3c recipe (paired with r3c). Not met: n9c's
  recall jump was seed luck, recorded as such.

**n9r RESULT (4090, 2026-10-31, `37fa333`; 103.5M vs 102.9M, 600M tokens
each, SEED=43 both): BOTH recall bars NOT met. N9a's recall gain is not
reproduced; n9c's jump was seed luck.** Logs: `logs/v13_sempty_n9r*b_main_37fa333_*.log`,
`logs/memory_probes/v13_sempty_n9r*b_main_37fa333_t40_behavioral.json`,
`logs/learning_probes/*labeldiag_t200.json`, `logs/pos_loss/*pg19.json`,
`logs/v14_n9r_chain_37fa333_20261031.log`. Scored with the v13 definition
(accuracy·n over `aggregates`; 63 rows).

| | overall | ≤ 2048 | > 2048 | label_diag | PG-19 far gain (4096-8192) | val (holdout) |
|---|---|---|---|---|---|---|
| ctl43 | 0.901 | 0.968 | 0.735 | 0.655 | 0.043 | 3.598 |
| n9a43 | 0.917 | 0.985 | 0.749 | 0.634 | 0.039 | 3.459 |
| d | **+0.016** (bar ≥ +0.08) | +0.017 | **+0.014** (bar ≥ +0.15) | −0.021 | −0.004 | −0.139 |

- **Reading:** against its own paired control N9a is the same on recall. The
  big numbers in n9c were a weak control: C1-ctl scored 0.825 / 0.597 and
  ctl43 (same recipe, seed 43) scores 0.901 / 0.735. **The seed moves recall
  by +0.08 overall and +0.14 past 2048, 5× to 10× N9a's effect.** Both seed-43
  models sit above C1-ctl, so this recipe's recall is not stable at one seed.
- **A pre-registered bar answers the question; the guard is moot.** ctl43 is
  an outlier on LM loss: holdout 3.598 vs C1-ctl 3.474 and n9a43 3.459, and
  it is 0.045 worse already at the end of stage A. So "N9a is 0.14 better on
  holdout" is NOT a finding (the control is bad, not N9a good); PG-19 shows
  the same 0.14 shift at every position. Cause unknown (seed, or an unlucky
  run; the box control was clean for C1-ctl).
- **Verdict: N9a (recall) FAIL as a registered claim; no ≥ 1.5B arm.** It
  also had no LM effect by token class (attr3b, n9c vs C1-ctl) and lost
  label lookup at 600M. N9a stays in code only because N9s builds on it.
- **Lesson:** a single 600M seed pair cannot support a recall claim; the
  control's own seed spread is ±0.1. Recall claims need ≥ 2 control seeds.

### N9s — salient phase (user idea, 2026-10-31; PRE-REGISTERED ~00:40 IST)

**Idea (user).** N9a's phase comes from a FIXED window of the last 3 tokens.
In real text the word that identifies a fact is often further back ("the big
red **truck** that we bought last year **is** silver"), so the window sees
only filler and the keys collide again. Instead let important words set the
phase: per head an importance s_t = sigmoid(w · rms(h_t) + b), and a running
angle psi_t = (1 − s_t) psi_{t-1} + s_t W_s rms(E[x_t]) added to N9a's
window angles. An important word mostly OVERWRITES the angle (so the same
"truck is" still aligns across paragraphs), filler leaves it alone, so the
window stretches with meaning. O(1): one angle per channel pair. W_s
zero-init: N9s starts exactly as N9a (selftest; scan == loop 4.5e-5; decode
== prefill 5.0e-5). Arm `echo_phase_sal`.

**Bench `label_gap` / `label_gap_new`** (`v14/rules_ood.py`): "truck, seen
near the old river, is silver; ...; truck, last year, is" -> silver; every
item has its own label; definitions put 0-5 filler words between the item
and "is", the question always 1-5, so a fixed 3-token window never sees the
question's item (0 of 300 checked). `_new` uses words never seen in training.

**N9s micro run (pre-registered).** Training: the default B1 tasks +
`label_gap` (train words). Arms echo, echo_phase (N9a), echo_phase_sal
(N9s); 3 seeds, i = 1..3: `--seed 11i --doc_seed0 (i+30)·100M --init_seed
11i`, tag `n9sp<i>`, same init and data per seed.
- **Bar G (primary, label_gap formation, ≥ 0.9 over 2/4/8 shots):** N9s forms
  in ≥ 2 of 3 seeds AND N9a in ≤ 1 of 3.
- **Secondary:** label_gap_new mean d (N9s − N9a) ≥ +0.15.
- **Guard:** label_category mean N9s ≥ N9a − 0.05.
- **If N9a also forms label_gap in ≥ 2 of 3:** the fixed window is not the
  limit on this bench (the body carries the item some other way): no claim.
- **Prediction:** N9a and echo fail label_gap (item outside the window); N9s
  forms it.

**N9s micro run RESULT (4090, 2026-10-31 00:11–04:04 IST, code `5d416bb`):
bar G NOT met, no claim (the pre-registered "N9a forms too" case). One
surprise: N9s formed two_hop in one seed, which no arm has ever done.**
Logs + JSON: `logs/v14_n9sp{1,2,3}_5d416bb_*.{log,json}`. Seed 113's N9s
arm was stopped at step 1500/6000 (04:04 IST): N9a had already formed
label_gap in 3 of 3, so no N9s result could change bar G, and the stop let
the n9r chain start 40 min earlier. Mean over 2 / 4 / 8 shots, per shot in
brackets.

| seed | arm | label_gap | label_gap_new | label_category | label_exact_new | two_hop | ms/step |
|---|---|---|---|---|---|---|---|
| 111 | echo | 0.29 | 0.29 | 0.43 | 0.28 | 0.30 | 619 |
| | N9a | **1.00** | 0.99 (0.99/1.00/0.98) | 1.00 | 0.98 | 0.30 (0.52/0.23/0.16) | 309 |
| | N9s | **1.00** | 0.84 (1.00/0.90/0.62) | 1.00 | 0.94 (0.99/0.96/0.87) | **0.92 (0.97/0.93/0.85)** | 218 |
| 112 | echo | **1.00** | 1.00 | 1.00 | 0.98 (1.00/1.00/0.95) | 0.29 | 216 |
| | N9a | **1.00** | 0.98 | 1.00 | 0.87 (0.91/0.88/0.83) | 0.29 | 180 |
| | N9s | **1.00** | 1.00 | 1.00 | **0.50** (label_exact 0.76) | 0.28 | 214 |
| 113 | echo | 0.29 | 0.27 | 0.42 | 0.30 | 0.30 | 204 |
| | N9a | **1.00** | 0.83 (1.00/0.90/0.60) | 1.00 | 0.79 (0.91/0.89/0.56) | 0.30 | 218 |
| | N9s | stopped | | | | | |

(ms/step is wall time on a shared card, not a speed comparison.)

- **G:** N9a formed label_gap in 3 of 3 (the bar needed ≤ 1). Echo formed
  it in 1 of 3, together with its whole label circuit (label_category,
  label_exact). **My prediction was wrong.** The key is built from the body
  state h_{t-1}, which carries the item across 1–5 filler words; the window
  phase only separates the keys. On this bench the fixed window is not the
  limit. The phase's value is formation again: N9a 3 of 3 here, 5 of 5 in F1,
  echo 1 of 3 and 1 of 5.
- **Secondary:** label_gap_new d (N9s − N9a) = −0.15 (111), +0.02 (112).
  Not met. **Guard:** label_category 1.00 = 1.00, met; but in seed 112 N9s
  lost label_exact (0.76) and label_exact_new (0.50, chance).
- **Phase arms lose on new words at 8 shots:** label_gap_new @8 N9a 0.98 /
  0.97 / 0.60, label_exact_new @8 0.98 / 0.83 / 0.56; echo, in the seed
  where it formed, 1.00 / 0.95. The angles come from learned embeddings, and
  words never seen in training have untrained embeddings, so they separate
  less when many entries share the context.
- **two_hop ("Ana owns the dog. The dog is called Rex. … Ana's pet is
  called" → Rex): N9s seed 111 = 0.97 / 0.93 / 0.85.** Every other arm and
  seed here is at in-context chance (0.09–0.16 at 8 shots). So is every arm
  of every earlier run: B1 runs 1–4, F1 (N9a 5 seeds), and same-size
  attention (tpp, B1 run 4, 3 seeds: 0.50 / 0.25 / 0.12). In the query, the
  name that decides the answer ("Ana") is 4–5 tokens before the answer, out
  of the 3-token window. That is the user's case for salience, better than
  label_gap. One seed of two; replication below.

**n9th partial (2026-10-31 12:55 IST): seed 131 (6000, `161a3c9`) replicates
the two_hop formation.** two_hop mean (2 / 4 / 8 shots): N9s **0.89**
(0.95 / 0.89 / 0.82), N9a 0.28, tpp 0.30 (chance 0.29). label_gap 1.00 for
N9a and N9s, tpp 0.30. With seed 111 (0.92), N9s has formed two_hop in 2 of
2 seeds, N9a in 0 of 2, and the same-size transformer in 0 of 4. Seed 132
(4090) finishes next; the run then STOPS (pivot rule: notebook-select work
ends once n9th reports). Bar H is met as soon as N9s forms in 2 seeds while
N9a forms in none, so seed 132 decides it. Logs
`logs/v14_n9thp1_161a3c9_20261030_2241.{log,json}` (box clock UTC).

**n9th RESULT (2026-10-31 13:15 IST): bar H MET at 2 seeds; the run stopped
there (pivot rule). N9s forms the two-hop lookup in every seed run (3 of 3
with seed 111), N9a and same-size attention in none.** Seeds 131 (6000) and
132 (4090), code `161a3c9`, 6000 steps × B16, 27.9M / 27.6M params. two_hop
mean over 2 / 4 / 8 shots (chance 0.29):

| seed | N9s | N9a | tpp (Transformer++) |
|---|---|---|---|
| 111 (earlier run) | 0.92 | 0.30 | – |
| 131 | **0.89** (0.95 / 0.89 / 0.82) | 0.28 | 0.30 |
| 132 | **0.90** (0.99 / 0.94 / 0.76) | 0.29 | 0.30 |

- **Bar H:** N9s forms in ≥ 2 of 5 seeds AND ≥ 2 more than N9a: 2 formed vs
  0, met by arithmetic after two seeds, so seeds 133-135 were not run.
  **Guard:** label_gap 1.00 for N9a and N9s in both seeds. Logs
  `logs/v14_n9thp{1,2}_161a3c9_*.{log,json}` (box clock UTC).
- **What it is:** the only notebook-side effect in two days that survived
  replication and beat attention at the same size. In "Ana's pet is called",
  the running angle keeps "Ana" as the current topic across the words that
  follow. Two hops need an anchor, not more memory.
- **What it is not (yet):** a 600M result. At 600M, in-context skills follow
  the trained format (B2, lb), so it needs two_hop-format data in the mix
  before any claim, and N9a's 600M lookup was negative. It is a MIND
  candidate (reasoning over fetched facts), so the decision to test it at
  600M belongs to the mind lane (v13) and the board (organiser). It also
  seeds the planner (N11): the salient angle as the retrieval query.

**N9s two-hop replication `n9th` (PRE-REGISTERED 2026-10-31 ~04:10 IST,
before any n9th number).** Question: does the salient phase let a 27M O(1)
body compose two facts in context (two_hop), where the fixed window and
same-size attention stay at chance?
- **Arms:** echo_phase (N9a, control), echo_phase_sal (N9s), tpp
  (Transformer++ reference, params matched). Same training as n9s (`--rule_tasks
  letter_shift,number_step,reverse3,label_category,two_hop,label_gap`),
  6000 steps × B16, lr 3e-4, `--seed 13i --doc_seed0 (i+50)·100M
  --init_seed 13i`, i = 1..5, tag `n9thp<i>`. Same init and data per seed.
  Code `5d416bb` (N9s unchanged). Hardware by agreement between sessions.
- **Formed:** two_hop mean over 2 / 4 / 8 shots ≥ 0.75 (in-context chance
  0.29; seed 111 had 0.92).
- **Bar H (primary):** N9s forms two_hop in ≥ 2 of 5 seeds AND in at least
  2 more seeds than N9a.
- **Guard:** N9s forms label_gap (≥ 0.9) in ≥ (N9a's count − 1) seeds.
- **Reported:** tpp's two_hop count (the "where attention fails" part),
  two_hop per shot, label_exact_new, every other cell.
- **Outcomes:** H met → salience lets the O(1) body chain two lookups in
  context at 27M. Next: find how (what the running angle holds at the query,
  per layer), then a 600M plan with v13 (two_hop-format data in training,
  as B2 did). H not met → seed 111 was luck; N9s closes as FAIL, is
  recorded in the index with its last-code commit, and is removed from code.

### N9 — phase-aligned dual memory (the user's 2×O(1) memory, 2026-10-30; design only)

**Idea (user, mapped by the v13 session against the key-collision
diagnosis):** "first strong alignment, then pick the best one."
1. **Align by phase.** Rotate each memory key and query by phases computed
   from the last few tokens (learned angle per token, a few frequency bands;
   function words can learn ≈ 0). "truck is" vs "red is" stop colliding, an
   exact repeat aligns exactly, non-matches cancel by interference instead
   of blurring, and learned angles can put car ≈ truck (label_category).
   Content-relative, not positional, so nothing runs out of range past the
   trained length. Our own phase lineage (v11 E3, Chrono).
2. **Pick by agreement.** Two such memories with independent phase banks;
   the reads are combined as a product (AND). Collisions are random per bank
   and the true value is shared by both (count-min-sketch logic). Cost 2×O(1).
**Relation to N7:** N7's exact hash is the hard limit of step 1 (no
similarity); N9 is the learned version that can also match by meaning.

**Pattern separation, measured (v13 offline, c2m2's real echo keys at label
positions, 60 prompts, fixed random expansion per head, no training).** Mean
cosine between keys of different pairs, label-label / random positions:
dense 0.671 / 0.303; mean-centred 0.462 / −0.005; sparse (D 2048, top 32)
0.297 / 0.114; **centred then sparse 0.166 / 0.053** (~4× less overlap, so the
delta erase wipes ~4× less of earlier pairs; a learned expansion should do
better). v13 tests it on the existing echo as S1 (their lane). **For N9
(hippocampus, step back / meditation): DG step = centre + k-WTA expansion,
then CA3 step = settling; phase keys become optional.** Build after N7 /
N9a land, without duplicating S1 (N9 adds the completion step).

**N9b dropped before any code (2026-10-30).** v13's offline check on c2m2's
embedding table: raw embeddings barely separate categories (cos same vs
different category 0.57 vs 0.45 for E[item]+E[is]); 2-bank sign-bit LSH puts
same-category pairs in one bucket no more often than different ones (8 bits
OR 0.09 vs 0.04; 16 bits ≈ 0 both). So N9b would only repeat N7's exact copy.
Fuzzy matching needs a learned similarity before the pick.

**N9a code (2026-10-30, branch `n9a`, selftest ALL PASS).** `echo_phase_lags`
(arm `echo_phase` = 3 lags): echo's q and k turned per channel pair by
alpha_t = Σ_j W_j rms(E[x_{t-j}]) (query: window ending at t; key: window
ending at t-1), W zero-init (+24.6k params at the micro size). Selftest: init
== echo bit for bit; decode == prefill 4.4e-5; collision check: one template
key turned by 40 "X is" windows has mean |cos| 0.09 between different X
(1.0 unturned) and 1.000 for the same X; phases get gradients.

**N9a micro run 1 (PRE-REGISTERED 2026-10-30 ~13:25 IST, before any number).**
Same bench, training tasks and seeds as N7 micro run 1 (pairs i = `--seed 6i
--doc_seed0 (i+12)·100M --init_seed 6i`), tag `n9ar1p<i>`. Arms echo vs
echo_phase (echo rerun in each pair as the paired control).
- **Bar Xa (primary, label_exact_new):** d = echo_phase − echo, mean over 2 /
  4 / 8 shots; met if mean(d) ≥ +0.15 AND ≥ 2·SD over the 3 pairs.
- **Guard G:** B2 held-out mean, echo_phase ≥ echo − 0.05.
- **Reported:** label_category (trained here; the fuzzy-match signal that
  decides whether settling, step 2, is needed), label_exact, repeat_copy;
  N7's echo_n7 from N7 run 1 at the same seeds as the exact-copy reference.
- **Next:** Xa met and label_category up: step 1 works, and settling is
  only built if a gap to tpp remains on label_category. Xa not met: the phases
  do not reach the read at this size; check the learned angles before closing.

**Refined (user via v13, 2026-10-30 ~13:00): nested, not parallel.** "Get the
phase, then a memory learned in that phase picks the best." Two facts: (1)
more phase or more state only adds address bits, and our collisions come
from look-alike keys, not too few bits; (2) two linear memories read in
sequence are still linear in the query (forward Σ(e·v)(k·q) = backward
Σ(q·k)(v·e)), so a second store cannot *pick*: picking needs a nonlinear,
competitive step. v13's pick: N8 gate (what to store) + content-phase keys
(each item its own valley) + **settling** (2–3 fixed Hopfield-style
relaxation rounds pulling the query to the nearest stored key, with a
sharpening nonlinearity; then read the value). Constant cost. Limit: a
settling d×d memory holds only a few dozen clean valleys per head.

**Build call (v14, 2026-10-30).** Step 1 in two variants on the N7 bench,
head to head, after N7 micro run 1:
- **N9a, content-phase keys** on the learned store (v13's step 1); judged by
  key cosine at answer positions and label_exact.
- **N9b, semantic N7:** N7's table with a locality-sensitive slot hash
  (sign bits of a fixed random projection) over the *embeddings* of the last
  n tokens, 2 banks, product vote. Exact repeats always share a slot; similar
  n-grams ("car is" / "truck is") share one with a probability that grows
  with similarity. The discrete lookup is the nonlinear pick: no settling,
  no few-dozen ceiling, parallel (N7's sort), O(1), and keyed on content
  tokens, so no template collision.
- Settling (step 2) is built on N9a only if N9a beats N9b on label_category.
  N8's gate goes on whichever store wins.
**Bars (v13's proposal, freeze before the run):** label_exact ≥ T++ (0.88),
label_category ≥ 0.73, accuracy 5 pairs back ≥ 0.70, LCL past 2048 ≥
in-context chance + 0.15 (the mapping, not the label set; see N7); recall
and holdout not worse. Micro first, on the N7 bench (N7 as the exact-copy
control).

### N10 — rules as phase continuation (user vision via v13, 2026-10-30; design only)

**Idea.** Give ordered things phase coordinates (letters on a circle; numbers
on several circles of different periods; multiplication via log-phase). A rule
is then one rotation, inferring it is reading an angle from the examples, and
an unseen value is exact continuation. Echo already infers held-out shifts,
but per seed (B2 state sweep: 3 of 10 runs ≥ 0.9); built-in phase structure
should make it reliable.
**Relation to N6.** N6 (operator memory) was this mechanism: the state is the
circular mean of b·conj(a) over example pairs, applied to the next input. It
was closed as FAIL on B1 run 2, a bench where selecting among 3 trained shifts
beats inference, so it never had a fair test. **N10 = N6 ported from
`78e4354`, judged on B2**, with the user's twist: several circles of different
periods (a token's phase on circle c is learned, with periods spread so that
numbers can sit on several clocks at once). Bar: formation rate on
shift_wide_new over 5 seeds vs echo (≥ 0.9 counts as formed), plus B2
held-out numbers not worse.

**N10 code (2026-10-30, branch `n10`).** N6's `OperatorMemory` restored from
`78e4354` (modes `phase`, `diag`) plus mode **`coord`** (N10): one learned
coordinate per head for the input side (W_in h_{t-1}), the output side (W_role
ê(x_t)) and the query, each read on P = head_dim/2 circles with periods spread
geometrically (omega_p = 100^(-p/P)); a rule = one step along the coordinate,
the same step on every circle; S = faded circular mean of the steps, read
p = q·S/(|S|+1), zero-init output. Init pairing fixed: the memory is built
and initialised under fork_rng and the body reloaded, so at the same seed the
model is bit-identical to plain echo (N6's old hook re-initialised the whole
body). Selftest: init == echo for both modes; decode == prefill 1.2e-5 /
5.1e-6; every parameter learns. Params: N6 +49.7k, N10 +17.4k (micro size).

**N10 formation run (PRE-REGISTERED 2026-10-30 ~16:10 IST, before any number).**
Does a built-in multi-circle coordinate make held-out rule inference form for
every seed?
- **Bench:** B2 training (`--rule_tasks shift_wide,step_wide,affine_map`),
  6000 × B16. **Formed** = shift_wide_new mean over 2/4/8 shots ≥ 0.9.
- **Arms:** echo, op (N6, its first fair test: B1 could not show
  inference), opcoord (N10). 5 seeds: i = 1..5, `--seed 9i --doc_seed0
  (i+25)·100M --init_seed 9i`, tag `n10p<i>`, same init and data per seed.
- **Bar C (per arm):** formed in ≥ 4 of 5 seeds while echo forms in ≤ 2 of
  5. **Guard:** held-out numbers (step_wide_new, affine_map_new) mean not
  below echo's by more than 0.05.
- **Reading:** if echo itself forms ≥ 3 of 5, the bench cannot show a
  reliability gain on this seed set: report rates, no verdict.

**N10 formation run: STOPPED after 2 seeds, no verdict (2026-10-30 22:30 IST,
code `9bd36e6`).** shift_wide_new: seed 91 echo 0.960 / op 1.000 / opcoord
0.999; seed 92 echo 0.998 (the op / opcoord arms of seed 92 were stopped).
Echo formed in both, so by the pre-registered reading ("if echo itself
forms ≥ 3 of 5 ... no verdict") this seed set cannot show a reliability gain,
and the remaining ~5 h would only slow the 600M N9a run beside it. B2
formation rates swing between seed sets (3/10 earlier, 2/2 here), so a
reliability claim would need many seeds, which is not worth GPU time now (user,
2026-10-30: run only what can be superior). N10 stays in code as an untested
candidate; N6 (`op`) formed in its one seed. Log + JSON:
`logs/v14_n10p1_9bd36e6_*`.

### N11 — reason first, then look (pands-style planner + three memories; user vision, design only)

**Idea.** The user's pands v1 reads a passage into an essence direction z
before speaking and steers every token toward it (its planner is the main
lever: honest vs oracle gap ~7 PPL). For lookup: reason first, form the
query ("I need truck's label"), then look in TWO stores, a knowledge store
learned from all training data (parametric; v13's A4 `cond_mem` Engram table
is in code, never run) and the conversation memory (N7 / N9), then speak
toward the intent. Today PAM looks up with the current token's state at
"is", which is where the label failure starts. Design after N7 / N9a land:
which state forms the query (a slow planner state vs the body), and how the
two stores' reads are combined.

### L — the library (v14's lane after the 2026-10-31 pivot; PRE-REGISTERED ~13:00 IST, before any L number)

**What the pivot changes.** The O(1) body is the mind; verbatim knowledge
lives in a session library and a shared growing library. v14 builds the
library's write side (which events become rows), its address (the key) and
the fetch (when to look, how to re-read). The exact n-gram organ already
recalls 1.0 at every length, so the learned part must earn its place where
the organ cannot: (a) paraphrased or fuzzy queries, (b) an instruction whose
query shares no words with it ("x ->" vs "Rule: add 3"), (c) one row per
EVENT instead of one per token, which is what lets the index reach 1M.

**Row format (proposal for MLB-1 §4).** One row per event:
- key: 128 dims × bf16 = 256 B (learned projection of the mind's pre-logit
  vector at the salient token; key-version tag below);
- span pointer: start u32 + length u16 = 6 B;
- session id u32 (4 B), write ordinal u32 (4 B, for latest-wins), model/key
  version u16 (2 B). Shared libraries add a provenance hash (8 B) and a
  source id (4 B) later.
→ **272 B/row** (284 B shared). The raw text is kept as u16 token ids
(2 B/token), so any span can be re-read.
- At 1M tokens with ~1 row per 16 tokens: 62.5k rows × 272 B = 17 MB + 2 MB
  text + 3.9 MB mind state ≈ 23 MB, against ~38 GB of T++ KV cache. At 128k:
  ≈ 6.4 MB vs 4.9 GB.
- The key version means: the key head is frozen with the model, so community
  libraries built for version v plug into any copy of that model.

**Thesis clause (user, 2026-10-31: "having intelligence separate from knowledge
is itself a big novelty, specifically when it holds beyond the training
context"; Lead's rule: every unit names the clause it serves, otherwise it is
a cost table, not a claim).**
- **separate from knowledge:** W0 (write gate), L1 (learned fetch key), the
  text-first library format, K1 (install-time keys).
- **holds beyond the context:** SR-0 / SR (the skill register), I1
  (interference of replayed rules).
- **both:** S1 (skill packs: fetched knowledge turns into a carried skill).

**Unit W0 (eval only, free): does the surprise gate catch events?**
Frozen C1-ctl (`c1ctlb_main_7ae6f1f`), pre-memory vector (head projection
before the echo add) through the tied head, u_t = −log p(x_t). Documents =
the organiser's needle docs, 8k tokens, 40 docs per cell, tau ∈ {2, 3, 4, 5}
nats.
- Cells: synthetic definitions, and natural facts (real sentences from other
  held-out documents as needles).
- **Bar W0:** at a row rate ≤ 1 per 16 tokens, ≥ 95 % of needle events have
  a gate-firing token inside the needle sentence, in both cells. Reported:
  rows per 1k tokens by tau; where the gate fires inside the needle
  (key / value / other).
- If the natural cell fails: the write gate needs training (a learned gate on
  the same signal); the frozen surprise alone is not enough.
- **Amendment A1 (~13:00 IST, before any W0 number; arithmetic, not data).**
  (1) The tau grid is {4, 5, 6, 7, 8, 9, 10} nats: mean surprise on web text is
  ~3.5 nats, so tau ≤ 5 fires on far more than 1 token in 16. A "row" is one
  firing token (u > tau); row rate = firing tokens / tokens; the bar applies
  at the lowest tau whose measured row rate on plain filler is ≤ 1/16.
  (2) "Natural" = the organiser's natural keys (real two-word phrases in the
  definition template). (3) Plain real sentences as needles are REPORTED only,
  against the chance catch rate 1 − (1 − r)^len (63 % for a 16-token sentence
  at r = 1/16): a gate that fires at random would already catch 63 %, so a
  95 % bar on them is impossible by arithmetic. The lift over chance is the
  number.
- **Amendment A2 (13:02 IST, after a 2-document smoke run, before the full
  run).** The smoke showed ~1 row per 16 tokens even at 10 nats, so the grid
  is extended to {4, ..., 14} nats; a bar at row rate ≤ 1/16 is read at the
  lowest tau in the grid that reaches it. Full run: 40 documents per cell,
  8192 tokens, 8 needles per document, seed 9100.

**W0 RESULT (4090, 2026-10-31 13:03-13:04 IST, frozen C1-ctl, code
`232748e`, 40 docs × 8k tokens × 8 needles per cell): bar NOT met on the
natural cell, met on the synthetic cell only at the edge, and the catch is
mostly a template effect.** Logs `logs/v14_w0_c1ctl_232748e_20261031_1303.{json,log}`.
The lowest tau with ≤ 1 row per 16 tokens is 11 nats:

| cell (tau 11) | rows / 16 tok | needle has a firing token (bar ≥ 0.95) | key fires | value fires | chance catch |
|---|---|---|---|---|---|
| synthetic | 0.61 | **0.950** | 0.61 | 0.02 | 0.32 |
| natural keys | 0.57 | **0.928** (not met) | 0.39 | 0.29 | 0.27 |
| plain real sentences | 0.59 | 0.887 (lift +0.31) | – | – | 0.57 |

- **Why the catch is high:** a token-level look at 15 documents per cell
  shows the firing token is mostly the FIRST word of the inserted sentence
  (" Definition": 13.3 nats, above 11 in 84-86 % of needles), not the key (9
  nats mean, above 11 in about 25 %) or the value. The inserted sentence is a
  topic break in the prose. So the frozen surprise gate detects WHERE A NEW
  EVENT BEGINS (a surprise burst), not what is in it.
- **Plain real sentences** are caught 0.887 vs 0.574 by chance: sentences with
  rare content stand out, so the gate does select content-bearing text.
- **What it means for the row:** a row anchored on the firing token alone does
  not hold the key word. The key word sits a few tokens after the firing one.

**W0, second reading (same JSON, `full` column; NOT the registered
measurement).** The harness gates on the model's actual output vector, which
includes the echo read; the pre-memory vector I registered does not. With the
echo included, filler is less surprising and the topic break stays high:

| cell (echo-included surprise) | tau 10: rows / 16 tok, caught, key fires | tau 11 |
|---|---|---|
| synthetic | 0.79, **0.994**, 0.80 | 0.47, 0.984, 0.63 |
| natural keys | 0.75, **1.000**, 0.72 | 0.44, 0.981, 0.53 |
| plain sentences | 0.78, 0.975, – | 0.46, 0.947, – |

By this reading the W0 bar would be met at tau 10 (≤ 1 row per 16 tokens,
≥ 0.95 in both cells). The registered verdict above stands (not met); this
only decides which gate L1 uses: **the harness's own gate, echo-included
surprisal at tau 10** (amendment A5, before any L1 number). The topic-break
reading is unchanged (the key still fires in only 72-80 % of needles).

**Amendment A3 to L1 (13:06 IST, after W0, before any L1 number): the row is
a BURST SPAN.** A firing token f (u > tau, refractory 16 tokens) opens a row
over tokens [f, f + 24); its key is a learned attentive pooling of the span's
vectors, k = Σ_i softmax_i(a · v_i) · W_k v_i (128 dims). The query stays the
mind's vector at the question's salient token through W_q. This keeps the
index at ≤ 1 row per 16 tokens and lets the key head find the key word inside
the span. The all-position frozen address stays the comparison, and the
"row at the firing token only" index is reported as an ablation. Bars A / P /
R are unchanged. Training documents come from the training pool (the
organiser's `--filler` switch), never from the held-out prose.

**Unit L1 (trained key head on the frozen mind).** Key head W_k and query head
W_q, 128 dims each, trained contrastively (InfoNCE, temperature learned) on
the frozen C1-ctl vectors. The mind is never updated.
- **Data:** (a) synthetic definitions with ≥ 8 templates and the organiser's
  word pools; (b) natural: a rare token in a held-out-document pair (second
  mention → the sentence of the first mention). Training documents ≤ 32k
  tokens; negatives = all other rows of the document, with hard negatives
  from other sentences carrying the same token.
- **Eval (organiser's harness, ≥ 40 docs per cell, words and templates not in
  training, rows from the W0 gate, paired with the frozen address on the same
  docs):**
  - **Bar A (address):** top-1 of the right row ≥ 0.90 at 128k, ≥ 0.75 at
    400k, ≥ 0.60 at 1M (frozen all-position address: 0.63-0.88 at 128k,
    1 of 4 at 400k). Never below the frozen address in any cell.
  - **Bar P (what the organ cannot do):** on the paraphrased-key cell and the
    instruction cell (query shares no words with the stored sentence), top-1
    ≥ organ + 0.20.
  - **Bar R (rows):** ≤ 1 row per 16 tokens on prose; memory per row = 272 B.
- Two key-head seeds; the same head is trained for the Transformer++ and
  Mamba-2 fairness arms (their own vectors), so a win is credited to the
  mind's vector only if it beats theirs.
- **Outcomes:** A + P + R met → the key goes to v13's 600M package run on
  MLB-1. A fails at 400k-1M but not at 128k → needs a learned write gate or
  a second-level index (rows of rows). P fails → the learned key is no better
  than the organ and the library is the organ plus a gate.

**Amendment A4 to L1 (after the step back, before any L1 number).**
Event segmentation reading: surprise spikes mark event BOUNDARIES, and the
episode that just ended is what gets bound (hindsight). W0's "failure" is
that behaviour. L1 therefore has two row forms, same head, same training,
same bars:
- **L1-fwd** (A3): a fixed span [f, f + 24) after each firing token.
- **L1-evt:** the event between two bursts, written when the next burst
  arrives: span [f_prev, f), key = attentive pooling over its tokens. The
  pooling is a running softmax-weighted sum (numerator and denominator per
  dimension), O(1) state until the event closes. Cap: 128 tokens.
- **Negatives:** a memory bank of ≥ 60k rows drawn from many documents, so
  the head sees 1M-token candidate counts in training (train docs ≤ 32k
  tokens would otherwise show it only ~2k rows).
- **Branch pre-registered, not built:** if bar A holds at 128k but fails at
  ≥ 400k, L2 = a two-level index: a coarse key per chunk of 256 events
  (learned pooling of its event keys), query → top-8 chunks → top rows
  inside them. Consolidation (merging superseded rows, latest-wins) is the
  same level.
- **Gate:** the absolute tau stays the W0 baseline. Candidates to judge
  later, only if rows/16 > 1 or the event recall is low: an adaptive gate
  (u_t above a running mean + k·std, refractory 16), and a gate trained on
  hindsight utility from natural text (positive = an event whose rare
  entity recurs later in the document).
- **Selection after a fetch (planner v0, no training):** fetch the top-m
  spans, let the mind re-read EACH span in front of the query, and take
  the span whose answer distribution has the highest top-1 probability
  margin. Judged on the harness's "re-read with 4 distractors" cell against
  re-reading all m spans concatenated.

**Library format: text first (proposal to v13 / organiser).** The canonical
library is plain text plus provenance (source id, hash, time, trust tag).
Keys are DERIVED by the mind: installing a library runs one forward pass of
the user's model over it and builds the index locally. So a community library
works with any model version, nothing is locked to a key format, and the
272 B/row index is a local cache, not the shared format. Install cost is one
pass of the mind (a 100M model: minutes per 10M tokens).

**Novelty rule: closest prior work and what differs (added before the first
L1 number; user rule 2026-10-31, organiser PLAN_MIND_LIBRARY.md).**
- **W0, surprise bursts as event boundaries + storing the span:** EM-LLM
  (Fountas et al., 2024) does surprise-based event segmentation into an
  episodic memory. **Different: nothing in the mechanism.** The setting is
  new (the keys are the vectors of an O(1) recurrent mind, no KV cache),
  which is a property of the body, not of this unit.
- **N12 / surprise-gated writes:** Titans (Behrouz et al., 2024) gates memory
  writes by surprise. **Different: nothing.**
- **L1, a learned attentive-pooled key over spans, InfoNCE:** standard
  contrastive dense retrieval (DPR; ColBERT-style pooling). **Different:
  nothing.**
- **Text-first library, keys rebuilt by the reader at install time:** the
  usual RAG practice of re-embedding a corpus with the local encoder. What is
  specific here: the encoder IS the mind, so keys are in the mind's own state
  space and are rebuilt with one forward pass.
- **So W0 and L1 are PLUMBING: they are needed for the benchmark, and no
  claim is made for the mechanism.** They get no GPU beyond what the
  benchmark needs.

**Where the library can be new (organiser's three; to pre-register as
separate units, design only now).**
- **S1, skill packs.** A pack = demonstrations of a rule (B2 builders:
  held-out shift / affine values). Fetched ONCE, the mind induces the rule
  and carries it in O(1) state for the rest of the session; fact recall
  stays in the exact organ. The fetch key is the mind's state at the moment
  it needs the skill (a request that names the skill in other words than
  the pack, so no lexical match exists). Judged on: rule accuracy at
  L = 2k...32k after one fetch, vs (a) the same pack re-fetched at every use
  and (b) T++ and Mamba-2 with the pack in context. The thing to beat is the
  re-fetch arm: the claim is that the state carries the skill after the pack
  has left the window.
- **I1, interference.** Replay a fetched span through the state (no window)
  while the state already holds rules from earlier fetches; measure the
  accuracy of the earlier rules before / after, and of the new one. Pass =
  the earlier rules lose ≤ 0.05 per added pack up to 8 packs.
- **K1, install-time keys.** Rebuild all keys of a text pack by the reader
  and compare retrieval to keys built by another model version; pass = a
  pack built for version v works after a mind update with ≤ 0.05 loss.
- **Closest prior work for these:** in-context demonstration retrieval
  (e.g. retrieval of in-context examples, Rubin et al. 2022) and
  skill libraries for agents (Voyager, Wang et al. 2023) store and fetch
  demonstrations / code by embedding similarity. **Different (to be shown,
  not assumed):** the rule is carried in the recurrent state after the
  pack leaves the window (S1), and replay must coexist with what the
  state already holds (I1). If the re-fetch arm matches the carried arm,
  S1 is ordinary retrieval and is recorded as such.

**SR — structural skill register (design + pre-registration of SR-0, 2026-10-31
~13:30 IST; requested by the organiser from v13's prediction that bodies lose
an induced skill between 8k and 32k of filler: echo half-lives 660-3100
tokens, spread affine 0.44-0.61 at 8k).**
- **Idea:** an O(1) slot per head, written once at the moment a rule is
  induced, never decayed, protected from overwrite (the N8 critical-period
  gate applied to skills), read by the body at every token through a
  zero-init projection. The write trigger is an "aha" latch: the surprise
  of the example outputs collapses against its own running mean after a
  few repeats of the pattern. The content is the rule's carrier, which in a
  PAM is a read-out of the notebooks at the demonstration's arrow token.
- **Closest prior work:** Function Vectors (Todd et al. 2023, arXiv
  2310.15213) and Task Vectors (Hendel et al. 2023): the residual stream
  at a demonstration's arrow token encodes the task and can be injected to
  run it zero-shot; In-Context Vectors (Liu et al. 2023); State Soup (Pioro
  et al. 2024, arXiv 2406.08423): stored / retrieved / mixed recurrent
  states as task vectors; gated fast weights; Titans' persistent memory.
  **Different, to be shown (not assumed):** (1) induced in context by a
  self-detected latch, not extracted by a researcher with a labelled
  dataset; (2) held without decay past the trained length by a slot that
  nothing overwrites; (3) portable as a text pack: replaying the pack
  rebuilds the register. If (1)-(3) are not demonstrated this is function
  vectors in a recurrent body and is recorded as plumbing.
- **SR-0 (eval only, no training; run FIRST, because a register can only
  work if a transplantable rule vector exists in a body that already infers
  rules).** Model: v13's `b2p2_main_38e3c29` (PAM 102.9M, B2-in-training; it
  is on the 6000 and needs a copy to d_drive for local runs). Tasks
  shift_wide_new and affine_map_new, 8 shots, probe seeds ≥ 7000, 80 trials
  per cell.
  - Arms: (A) the normal prompt (demos, then G tokens of filler, then the
    query): accuracy at G = 2k and 32k. (B) no demos: G tokens of filler,
    then the query, with the function vector f_l added to the residual
    stream at the query's arrow token after layer l (f_l = the residual at
    the last demonstration's arrow token in a SEPARATE short prompt with the
    same rule; scale chosen on a dev seed, not the test seeds). (C) wrong
    rule: f_l from a different rule. (D) random vector of the same norm.
  - **Bar SR-0:** for some single layer l, (B) at G = 32k ≥ 0.9 × (A) at G =
    2k, AND (C) and (D) ≤ 0.25 × (B) (it is the rule, not any push).
    Reported (organiser's request, before any SR-0 number): the transplant curve at G = 2k / 8k / 32k / 128k, layer by layer, so the curve shows where the vector dies (v13 predicts between 8k and 32k), and (A) at each G (the loss to beat). Checkpoint: `/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/b2p2_main_38e3c29/best_model.pt`.
  - **Outcomes:** met → a transplantable rule vector exists; build SR
    (the latch + a no-decay slot) and pre-register its bars. Not met → a
    residual-stream register does not work in this body; the alternative is
    long-gap training (v13's recipe), recorded as the answer.
- **SR (to build only if SR-0 is met), bars on the skill-pack cell:** skill
  at +32k / +128k / +1M ≥ 0.9 of its +2k value; recall (t40) and holdout
  not worse (≤ +0.01); one register per head or a small bank. Training needs
  a long-gap curriculum for the READ to be learned (format-trained), so the
  micro-bench also needs a `--spread` rule eval first.
- **Order (my preference):** L1 (running now) → SR-0 (eval only, small GPU,
  can run beside L1 once the b2p2 checkpoint is on d_drive) → SR only if
  SR-0 passes. S1 / I1 / K1 come after SR-0, because S1 is the same question
  asked at the library level.

**Fetch metrics to report (every package run):** fetch rate (share of
queries that fetch), abstain rate on queries with no stored answer (negative
cell: the model must NOT fetch), span precision (right row top-1 / top-5),
re-read tokens per query, row rate per 1k tokens, write-gate recall.

### N12 — surprise-selected item store (design only, 2026-10-31 ~06:45 IST)

**Why (v13 EUREKA, `adbf1f3`).** At 600M on the same B2 data, Mamba-2
nearly matches PAM in context but collapses across distance (spread
shift/affine `_new` at 2048 + 4096: PAM 0.67, Mamba-2 0.08, T++ 0.13).
Mamba-2 leads on exact single items (label_diag 0.787 vs PAM 0.57–0.63; the
last far number through filler 0.27–0.31 vs PAM ≤ 0.14).

**Reading from the two update rules.**
- PAM notebook: S_t = a_t S + v⊗k̄. The retention a_t depends on the input,
  but every token WRITES at full strength.
- Mamba-2: h_t = exp(−Δ_t A) h + Δ_t B_t x_t. One Δ_t ties "write this" to
  "forget the past".

The tie explains both results:
- Mamba-2 can skip filler (Δ ≈ 0: no write, no forget), so single items
  survive.
- But writing each new example also forgets the earlier ones, so examples
  far apart do not add up.
- PAM can write without forgetting (rules add up across distance), but it
  cannot refuse a write, so filler buries single items.

**What is missing:** a write that is selective but not tied to forgetting.

**Correction (v13, 2026-10-31 ~06:45 IST, CPU diagnostic on trained
checkpoints).** I read the non-delta formula. With DELTA = 1 (A3, on in
every M2 run) the notebooks DO write through a soft per-token, per-head gate
b_w = sigmoid(write_proj(x)) that is not tied to decay. The coupled
Mamba-2 form was v13's C1, which FAILED at 300M (far a1 0.10 vs 0.26), and
that fits the tie reading. The trained gates are:
- **Selective by rank:** AUROC(item / label > filler) is 0.84–0.95 in echo
  and blocks 7 / 11 (block 3: 0.60).
- **Far too weak by size:** echo writes ~0.12–0.14 and erases ~0.11 at each
  filler token, vs 0.2–0.5 at items. Filler outnumbers items ~100:1, so an
  item keeps about exp(−0.11·N/98) through N filler tokens (~0.57 per
  500-token gap).

So "PAM cannot refuse a write" should read "PAM refuses filler only
softly, about 10× too weakly". The case for a structural selector is
stronger, not weaker.

v13 also found that echo retention drops at item / label tokens (it learned
the tie softly), and that B2 training sharpens this into a reset on rule
prompts. The distance edge may therefore be "overwrite a rule register at
each example, carry the latest across weakly written filler" rather than
coherent integration (v13 retracted its integration guess). v13 is testing
a test-time gate floor (gates < tau set to 0) on label_diag at 0 / 2048 /
4096: if far label accuracy rises without training, filler writes are what
bury items.

**v13 gate floor (`b8f7b08`, ~07:00 IST): no rescue, and the label gap is
not filler burial.** Setting echo's gates below 0.3 to 0 at test time
moves label_diag exact @2048 / 4096 from 0.26 / 0.21 to 0.34 / 0.29 (about
1.5 SE) but costs 0.04–0.09 at L = 0; a floor on every memory is worse.
The deciding fact is label_exact with NO filler at 2 / 4 / 8 shots: PAM
0.60–0.63 / 0.68–0.74 / 0.49–0.62, Mamba-2 0.865 / 0.905 / 0.815, T++
0.725 / 0.825 / 0.705. A 2-shot prompt is ~15 tokens, so nothing is buried:
PAM's label gap is a short-range lookup / binding gap, and burial only adds
to it at distance. N12 targets the distance part (far labels, step through
filler). It does not target the short-range binding gap, which v13's `lb`
pair tests as a format effect (label-format docs in training). A short
conv on the keys is not the answer either: v13's G1 / A1 failed
(label_exact 0.517 vs 0.610).

**Why not a learned soft gate.** v13's echo memory already has separate
soft decay / write / erase gates, and label_diag is still 0.57–0.63. v15's
G1d saw the same thing in another unit: learned soft addressing stayed soft
("sharpness must be structural"). So the selection should be structural.

**Idea: the fast store keeps what the slow learner failed to predict.** A
separate item store beside the notebooks (which keep integrating
everything). A token writes to it only when the BODY was surprised by it:
u_t = −log p_body(x_t | x_<t), computed from the body's pre-memory logits.

- It is detached: no gradient flows through the choice.
- It is causal.
- It needs no training to become sparse, because filler is predictable;
  names, labels and new numbers are not.

This is the complementary-learning-systems rule (the hippocampus stores the
neocortex's prediction errors) as the write policy. Mamba-2's Δ and Gated
DeltaNet's β, by contrast, are learned soft gates.

- **Write (per head, delta rule, no decay):** w_t = 1 if u_t is in the top m
  of the last C positions' u (causal rank, so at most ~m/C of tokens write),
  else 0. S_t = S − w_t (S k_t − v_t) k_tᵀ with k_t = unit(W_k h_{t−1}) (the
  N9a phase is optional) and v_t = rms(E[x_t]) (the exact token).
- **Read:** r_t = S unit(W_q h_t), through a zero-init output into the LM
  vector, as echo does. O(1): one d×d matrix per head.
- **Salience for N9s:** the same u_t can set N9s's running angle (a topic
  register driven by surprise, not a learned gate). This links to n9th if
  H is met.

**Open questions before code.**
1. Body logits cost one more LM-head pass (no grad, chunked over time).
2. Web-text filler has its own surprising words. At a ~1/8 write rate,
   8k tokens of spread give ~1000 writes into a d = 64 store, so either the
   threshold must be absolute (write only u above a high bar) or the store
   must be larger.
3. The micro bench has no web filler; a micro `--spread` needs in-domain
   filler (label_gap's FILLER words) and says less than v13's 600M
   spread.

**Judgement (to pre-register with the code).**
- Micro: label_exact(_new), label_gap(_new), two_hop vs echo and N9a on the
  same seeds.
- 600M, through v13: label_diag AND spread step_wide_new must rise, while
  spread shift/affine `_new` must not fall.

**In code (`787db29`, hook `8ce7edd`).** Arm `echo_surp` (tau = 3 nats:
the body gave the token < 5 %).
- **What it is:** echo's own memory and init, with no new parameters. It
  never fades. A token writes (b_w = 1, b_e = 0.95, every head) only when
  u_t > tau.
- **Surprise:** u_t = −log softmax(lm_body_{t−1} · Eᵀ)[x_t], where lm_body
  is the head's pre-memory vector. It is chunked and computed without
  gradient.
- **Change from the design:** an absolute threshold instead of top-m per
  window. A fixed rank rule writes the same share of every text, so dense
  item lists ("apple is beta, dog is gold") would lose half their items.
  The threshold writes everything while the body predicts nothing, and gets
  selective as the body learns, so selection develops with training.
- **No fade means reach is counted in writes, not tokens:** for random keys,
  the delta rule keeps about the last ~d (64) writes per head.
- **Checks:** selftest `test_n12` (init == echo; surprise ==
  log-softmax 9.5e-07; an empty store adds 0; decode == prefill 1.56e-04 at
  write rate 0.58; echo's gates unused) and `test_hook` (grad ckpt on == off
  with N12). The micro-bench records `write_rate` per arm.
- **600M:** `ECHO_SURPRISE_TAU=3` (or `--echo_surprise_tau 3`) with
  `ECHO=1`.

**N12 micro screen `n12m` (PRE-REGISTERED 2026-10-31 ~06:40 IST, before any
N12 number).** Question: does the surprise-selected store select on its
own, and is it at least as good an item memory as echo's learned gates
without breaking the rules?
- **Arms:** echo (control), echo_surp (N12), same init and data per seed.
  Training as n9s / n9th (`--rule_tasks
  letter_shift,number_step,reverse3,label_category,two_hop,label_gap`),
  6000 steps × B16, lr 3e-4, `--seed 14i --doc_seed0 (i+60)·100M
  --init_seed 14i`, i = 1..3, tag `n12mp<i>`. 4090, after n9th.
- **Selects:** N12's write rate on the last training batch ≤ 0.5 in every
  seed.
- **Guard:** mean (letter_shift, number_step, reverse3) N12 ≥ echo − 0.05.
- **Screen bar S:** N12 forms the label circuit (label_gap ≥ 0.9) in at
  least as many seeds as echo, AND the mean over seeds of (label_exact_new
  + label_gap_new) / 2 at 8 shots gives d = N12 − echo ≥ 0.
- **Reported:** d itself (≥ +0.15 = a strong signal), two_hop, repeat_copy,
  every other cell, ms/step. Added ~07:25 IST before any N12 number (v13's
  request): label_exact and label_exact_new at 2 shots, where PAM's 600M
  label gap starts.
- **Outcomes:**
  - Selects + guard + S met → hand to v13 for the 600M check on the B2
    recipe, paired with b2p2 (PAM seed 43). The bars there are label_diag
    and spread step_wide_new up, and spread shift/affine `_new` not down;
    v13 pre-registers the exact numbers.
  - Write rate > 0.5 is the only failure → one retry at tau = 5, written
    down before it runs.
  - Otherwise → N12 at this form closes (recorded, removed).

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
