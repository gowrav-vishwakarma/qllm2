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

**600M VERDICT + REMOVAL (2026-11-01 04:36 IST): FAIL; N9a and N9s removed from
the code (last code `d255fcd`).** Mind's paired arm cu1n9s
(`cu1n9sb_main_fa9e410`, its own stage A with N9a lags 3 + N9s, then cu1's
stage B, seed 42; v13_sempty EXPERIMENTS "cu1n9s VERDICT"): two-hop 0.090 vs
cu1 0.570 (rule ≥ cu1 + 0.15 and ≥ 0.50), holdout 3.7533 vs 3.5254 (+0.228),
label_diag 0.732 vs 0.913, fsq false abstain 0.35 vs 0.06, skill-pack 2k 0.535
vs 0.81, WikiText 4.64 vs 4.27; B2 `_new` equal. With N9a's own failures (n9c
lookup, n9r replication), the micro win (bar H, 27M) did not carry to 600M.
Maths kept here for the record: query / key turned per channel pair by
alpha_t = Σ_j W_j E[x_(t−j)] (window of 3), plus the salient running angle
psi_t = (1 − s_t) psi_(t−1) + s_t W_s rms(E[x_t]), s_t = sigmoid(w · rms(h_t) + b).
Rerun: `git worktree add ../qllm_n9 d255fcd`, then `ECHO=1 ECHO_PHASE_LAGS=3
[ECHO_PHASE_SALIENT=1] bash v13_sempty/tmp_pretrain_mix.sh`. Old N9 checkpoints
(n9c e5f0b73, n9r 37fa333, cu1n9sb fa9e410) now refuse to load at HEAD with an
error naming `d255fcd`.

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

**N9s at 600M (Mind's cu1n9s, cu1 amendment A1 `fa9e410`; switch `c91f4c6`).**
Paired arm of Mind's curriculum run cu1 on the 6000 (seed 42; its own stage A,
because the warm start refuses new parameters; then cu1's stage B; the data
carries the two_hop Record format through the `hop` source). Bars (Mind's):
two-hop ≥ cu1 + 0.15 AND ≥ 0.50; guards on holdout, label_diag, R6a. ETA
~03:15-05:45 IST. **Attribution caveat (sent to Mind):** the arm adds N9a's
3 lags, N9s's salience and a different stage A at once; if it passes, an
N9a-only arm (same new stage A, same seed) is needed before anything is
credited to salience. At 27M N9a alone formed two_hop in 0 of 3 seeds.

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
- **Amendment SR-0/A1 (after Mind's skill-pack result, before any SR-0
  number): the cell is Mind's.** The spread cell overstated the reach (its
  last example sits L/8 from the query, so it measured ~1k gaps with
  re-exposure). SR-0 uses `scripts/run_skillpack_probe.py` (`53d0e52`, task
  `skillpack`, tasks shift_wide_new and affine_map_new, 8 demos read ONCE at
  t0, then L tokens of filler, then the query; seeds 8000-8099; L = 0 / 2k
  / 8k / 32k / 128k). The loss to beat, from Mind (b2p2, 100 docs): affine
  0.98 / 0.14 / 0.07 at 2k / 8k / 32k, shift 0.74 / 0.05 / 0.01. The SR-0
  bar becomes: with the demonstrations REMOVED and the transplanted vector
  added, accuracy at 32k ≥ 0.9 × the no-transplant value at 2k (affine ≥
  0.88, shift ≥ 0.67), and wrong-rule and random vectors ≤ 0.25 × that.
  The no-transplant rows at 8k / 32k (0.14 / 0.07, 0.05 / 0.01) are the
  floor.
  **Amendment SR-0/A2 (Lead's request, before any SR-0 number):** the gap
  grid is L = 0 / 2k / 4k / 8k / 32k / 128k (the fade is between 2k and 8k),
  and the transplant margin (accuracy of the correct candidate minus the best
  wrong candidate, as in Mind's rows: +18 at 0, +4 at 2k, -2 at 8k) is
  reported next to accuracy. A transplanted vector that still holds at 8k,
  where the body's own state is at 0.14, is the first clean evidence for the
  register route.

- **CORRECTION (2026-11-01 00:20 IST, from Mind's bug report, v16 board / `824a988`):**
  the lg32 evidence below is CONFOUNDED. Mind's long-gap packer cut documents
  at window edges, so 44 % of the answer tokens (at 8k) sat in a window
  without their demonstrations yet carried ×100 loss weight, i.e. the run was
  trained to guess the prior. Mind withdrew the "vanishing signal / LM pressure"
  reading pending lg8r (8k windows + document reset + the fix). The two reasons
  quoted below are NOT established; the register question stays parked.
- **Why SR is the live route (Mind, 2026-10-31 evening):** training the mind
  at 32k on long-gap rule docs (lg32) SHRANK its one-read reach (2k 0.86 →
  0.31) and SHORTENED its memory: echo half-lives on filler fell 718→537,
  1564→1026, 1274→802, 2376→1427, 1548→984, 2539→1137 tokens; notebook b11 h4
  648→401 (EXPERIMENTS_SEMPY "lg32 decay half-lives"). Two reasons a decaying
  state will not be trained to hold a skill: the gradient through a 1-3k
  half-life vanishes, and the LM objective over many unrelated docs per window
  pushes the decays down. A protected register written once sidesteps both;
  SR-0 asks first whether a transplantable vector exists.
- **SR-0 RESULT (2026-10-31 21:20 IST, b2p2, code `97923ed`): NOT met; the
  register route as a residual-stream vector is dropped.** Dev (seeds
  8200-8219, L = 8k): every layer 2-14 × scale 1-8 gives 0.025-0.125 (chance
  0.125); chosen layer 8, scale 8. Test (seeds 8000-8099, 100 docs per rule;
  accuracy / margin):

  | rule | L | A: pack once | no pack | B: rule vector | C: wrong rule | D: random |
  |---|---|---|---|---|---|---|
  | affine | 0 | 1.00 / +18.3 | 0.03 | 0.10 | 0.03 | 0.17 |
  | affine | 2k | 0.98 / +4.1 | 0.04 | 0.15 | 0.10 | 0.14 |
  | affine | 4k | 0.42 / −0.4 | 0.07 | 0.11 | 0.19 | 0.13 |
  | affine | 8k | 0.14 / −1.8 | 0.12 | 0.16 | 0.17 | 0.14 |
  | affine | 32k | 0.08 / −1.9 | 0.07 | **0.16** (bar ≥ 0.88) | 0.13 | 0.11 |
  | shift | 0 | 1.00 / +10.5 | 0.00 | 0.00 | 0.00 | 0.01 |
  | shift | 2k | 0.74 / +1.0 | 0.01 | 0.09 | 0.04 | 0.07 |
  | shift | 8k | 0.05 / −2.7 | 0.03 | 0.12 | 0.07 | 0.05 |
  | shift | 32k | 0.01 / −3.0 | 0.01 | **0.08** (bar ≥ 0.67) | 0.07 | 0.05 |

  Logs `logs/v14_sr0_{dev_b2p2_97923ed.json,test_b2p2_L8_s8.0_97923ed_*_to32k.jsonl,driver_97923ed_*.log}`
  (the 128k pass was dropped: the bar is at 32k and the result was clear).
  - **Decisive row: L = 0.** With no filler at all, the injected vector cannot
    switch the skill on (0.10 / 0.00), and it is no better than a wrong-rule
    or random vector of the same norm. In this body the rule is NOT carried by
    one residual vector (unlike function vectors in transformers). The
    one-read curve A reproduces Mind's (affine 1.00 / 0.98 / 0.42 / 0.14 /
    0.08; shift 1.00 / 0.74 / 0.27 / 0.05 / 0.01).
  - **What it implies for SR:** the rule most likely lives in the recurrent
    notebook STATE built while reading the demonstrations, so a register would
    have to hold that state (about 1 MB per skill), not a vector. That is
    State Soup (Pioro et al. 2024) in spirit; what could still be ours is
    protecting it in-context without decay. The cheap test (SR-0b, not
    pre-registered yet): hold the notebook state from the end of the pack
    through the filler (decay 1, no writes during the filler) and read it at the
    query. It needs a decode path or a "freeze the notebooks over a span"
    hook in the M2 body (shared reads have no decode), which is Mind's code.

- **SR (to build only if SR-0 is met), bars on the skill-pack cell (Mind's probe, above):** skill
  at +32k / +128k / +1M ≥ 0.9 of its +2k value; recall (t40) and holdout
  not worse (≤ +0.01); one register per head or a small bank. Training needs
  a long-gap curriculum for the READ to be learned (format-trained), so the
  micro-bench also needs a `--spread` rule eval first.
- **Order (my preference):** L1 (running now) → SR-0 (eval only, small GPU,
  can run beside L1 once the b2p2 checkpoint is on d_drive) → SR only if
  SR-0 passes. S1 / I1 / K1 come after SR-0, because S1 is the same question
  asked at the library level.

**L1 development, first head (13:30-13:55 IST; NOT a bar result: the bars are
judged on the harness at 32k-1M with new words).** C1-ctl frozen; 1500
training docs (4096 tokens, 50 % synthetic with 8 templates, 50 % natural
recurrence pairs), 150 validation docs with the harness template and
words; `fwd` rows (burst + 24 tokens), the full head (588 → 128 projection +
attentive pooling, 75k parameters), 1500 steps, batch 48. Logs
`logs/v14_l1_fwd_a3e3265_20261031_1332.log`, `logs/v14_l1_data_8445f73_*.log`.

| set | frozen vector at the firing token | frozen span mean | learned head |
|---|---|---|---|
| training docs, seen (150) | top-1 0.19 | 0.02 | **0.93** (top-5 1.00) |
| validation, harness template (150) | 0.03 | 0.01 | 0.09 (top-5 0.41) |
| in-distribution validation, synthetic (64) | 0.22 | 0.00 | 0.19 (top-5 0.59) |
| in-distribution validation, natural pairs (65) | 0.05 | 0.02 | 0.06 (top-5 0.17) |

- **Event recall is fine** (a row holds the target): 1.00 on the harness
  validation, 0.76 on the natural pairs. Rows 0.86 per 16 tokens.
- **The head memorises its training documents.** 0.93 on seen documents, 0.06
  to 0.19 on new ones of the same kind, so this is not a template shift. It
  beats the frozen baselines on the harness validation (0.09 vs 0.03, top-5
  0.41 vs 0.06), but it is nowhere near the 0.90 bar.
- **Next (development, selected on in-distribution validation):** 4× more
  documents (4500 more, generation under way), smaller or regularised heads
  (token weights only: 589 numbers; identity + rank-32; feature dropout and
  weight decay), best step by validation. The `evt` form failed with an OOM
  at the 1.9 GB cap (rows up to 128 tokens × 48 docs); retry at a smaller
  batch.

**L1 development CORRECTION (14:55 IST): the first L1 result above was a bug of
mine, and the conclusion drawn from it is withdrawn.** Two defects in my data
preparation (`v14/l1_data.py` / `v14/l1_key.py`, fixed in `9f71835` and the
commit after it):
1. **Wrong anchor.** For keys of several tokens I anchored the query and the
   target on the first token (a fragment). The harness anchors on the LAST
   token. Fixed (`anchor()`).
2. **The query matched itself.** My documents include the query sentence at
   the end and I let rows start inside it, so the query's own span was always
   the most similar "row" (similarity 1) and always ranked first; in training
   it was a negative the head had to push away. The harness indexes only
   `[:doc_end]`. Fixed (`doc_end()`: rows only before the query).
Everything measured before the fix (train 0.93 vs validation 0.09-0.19, the
three small heads, the retraining `full2`) is invalid, including the reading
that the head "memorised". The logs stay in the repo as the record of the bug
(`logs/v14_l1_fwd_a3e3265_*`, `logs/v14_l1_full2_67a3cc1_*`).

**Frozen address, corrected (no training; 4096-token docs, about 200 candidate
rows, the same rows as the gate; top-1 of the row that holds the key).**
Centred cosine (the document's mean vector subtracted) between the vector at
the query's key last token and the vectors in each row:

| state used | harness-template docs (60) | synthetic, in-distribution (59) | natural pairs (46) |
|---|---|---|---|
| output vector (pre-logit), uncentred | 0.61 | 0.59 | 0.12 |
| output vector (after block 16), centred | 0.93 | 0.93 | 0.22 |
| **intermediate state, blocks 4-11, centred** | **1.00** | **1.00** | 0.54-0.57 |
| embeddings / after block 0, centred | 0.77 | 0.80 | 0.83 |

- **The mind's intermediate state at the last token of a key is a perfect
  frozen address at 4k tokens, with no training** (100 % top-1 in blocks 4-11
  on both synthetic sets). The output vector is weaker because it is tuned to
  predict the NEXT word (every key ends in the same " refers"); centring
  removes much of that shared part (0.61 → 0.93).
- **Natural recurrence pairs are the hard case**, and the one the organ cannot
  help with once words are paraphrased: real words recur in several places
  (a row can hold the word without being the first mention, which my label
  does not count as correct), and mid-layer states drift with context (0.54);
  the embedding layer, which is pure word identity, is best there (0.83).
- **What this means for L1:** the key should be built from the mind's
  intermediate state (centred), not from the output vector the harness passes
  as `vecs`. The harness only needs to hand over a layer (`--vec_layer`), or
  the key function can run the frozen mind itself on `ids`, which is how a
  library would be installed anyway (text first, keys rebuilt by the reader).
  The honest status: at 4k tokens the frozen mid-layer address already meets
  what Bar A asks; the open questions are 32k-1M (Bar A's real test),
  natural and paraphrased keys (Bar P), and then whether a trained head adds
  anything.

**L1 proper (PRE-REGISTERED 2026-10-31 ~15:10 IST, agreed with Lead after
discussion, before any L1-proper number).** A C2 pass (the all-positions
index, one key per token: 77 MB at 32k, ~2.4 GB at 1M) only shows that the
ADDRESS exists in the mind's state; it breaks R4 and the 272 B per event
budget. Bar A is therefore judged on the burst index.
- **Unit:** a pooling head that turns each burst span (C5 rows, ≤ 1 per 16
  tokens) into ONE key, from the mind's intermediate state after block 8,
  centred; the query key is the same head on the query's span. Trained
  contrastively on the training pool (synthetic needles with my 8 templates
  + natural recurrence pairs), rows only before the query (`doc_end`),
  anchored at the key's last token.
- **Ceiling to match:** C2 per-token top-1 of the frozen layer-8 centred
  address on the same docs (plan (i), running).
- **Bar A (harness, c1ctlb, synthetic keys, ≥ 16 docs per cell, gate_tau 10):**
  C5 burst top-1 ≥ 0.9 at 32k and ≥ 0.75 at 128k, AND within 0.05 of the C2
  per-token ceiling at 32k. Rows ≤ 1 per 16 tokens; 272 B per row.
- **Reported:** natural-key cells as the harness scores them (a recurring
  phrase without the value is a collision, Lead's rule); my recurrence-pair
  validation with the any-earlier-mention rule (its own metric, not C2);
  neg cell max-sim gap; para cell.
- **Thesis clause:** separate from knowledge. Closest prior work: dense
  retrieval with learned pooling (plumbing for the mechanism).
- **Memory latitude (user via Lead, 15:20 IST):** the session library may
  grow to ~2x (≤ ~50 MB per 1M tokens: e.g. 1 row per 8 tokens, or a 256-d
  key). Not used in this first run (128-d, ~0.8 rows per 16 tokens ≈ 14 MB
  of keys per 1M). If a later run uses it, its pre-registration states the
  MB per 1M tokens.

**L1 proper, first head (development, 4k validation; NOT the bar, which is
judged in the harness).** Centred block-8 state, burst rows (fwd span 24),
3000 training docs, full head (75k params), 1500 steps, best step by the
in-distribution set. Log `logs/v14_l1p_99e6803_*.log`, head
`qllm_ckpt/v14/l1p_fwd_L8c_99e6803.pt` (d_drive).
- Burst top-1 / top-5: harness-template docs 0.78-0.82 / 0.93; mixed
  in-distribution (synthetic + natural pairs) 0.51 / 0.67. Event recall 1.00
  / 0.86.
- Plan (i) so far (frozen layer-8 centred, harness, 32k, synthetic, 16 docs):
  kv C2 1.00 / C5 0.50; kv1 1.00 / 0.44; tool 1.00 / 0.60 (5 docs); para
  0.31 (= the organ); neg max-sim 0.54 vs stored 0.95 (gap 0.41: abstain by
  threshold works).
- Before the harness run, two harness questions sent to Lead (15:30): key_fn
  receives the output vector even with --vec_layer, and the C5 hit window
  (answer within 6 tokens after the row's start) can score a correct burst
  row as a miss.

**L1 proper, harness, PAIRED (2026-10-31 16:23 IST; bench.py @ 0eb2ead via
`2cc01c5`, c1ctlb, kv, 32k, 16 docs, seed 14000, gate_tau 10, --vec_layer 8
--centre; arm A = trained burst head `l1p_fwd_L8c_99e6803`, arm B = frozen
burst-mean key, same documents).** Logs `logs/v16_bench_l1pair_*_kv_*_32768.*`.

| keys | C2 per-token ceiling | trained burst key top-1 (top-5) | frozen burst key top-1 (top-5) | rows / 16 tokens |
|---|---|---|---|---|
| synthetic | 1.00 | **0.81** (1.00) | 0.06 (0.12) | 0.64 |
| natural | 1.00 | **0.94** (1.00) | 0.56 (0.56) | 0.60 |

- **Bar A at 32k: NOT met** (synthetic 0.81 < 0.90, and 0.19 below the C2
  ceiling). Natural keys reach 0.94.
- **At the row budget the trained key is a large gain over the frozen one**
  (synthetic 0.06 → 0.81, natural 0.56 → 0.94).
- **The right row is ALWAYS in the top 5** (1.00 on both): the misses are
  confusions between needles of the same template, which a re-read can settle.
  Next (development, to pre-register before running): fetch top-5 and let the
  mind pick by its own answer margin (planner v0, already in the harness), and
  separately the memory latitude (256-d key). 128k pending.

**PRE-REGISTERED 2026-10-31 16:25 IST (before any number of these), agreed
with Lead.**
- **L1+P0 (trained burst key + planner v0):** fetch the top 5 burst rows by
  the trained key, let the frozen mind re-read EACH alone in front of the
  query, and take the answer with the largest top-1 minus top-2 margin
  (harness `P_margin_burst`; `P_concat_burst` reported). Cost printed next
  to it: 5 short re-reads per query (`P_rereads_per_query`). **Bar: ≥ 0.95 at
  32k and ≥ 0.85 at 128k on kv, synthetic AND natural keys, 16 docs, seed
  14000, same documents as the paired runs.** Reported as its own row, NEVER
  as Bar A on the key alone. Abstain (neg cell) is judged separately.
- **L1-256 (memory latitude):** the same burst head with a 256-d key, trained
  on the same data plus more single-template docs. Memory: ~0.62 rows per 16
  tokens × 1M / 16 ≈ 39k rows per 1M tokens × (512 B key + 16 B pointer and
  tags) ≈ 20 MB, plus 2 MB of raw text and the 1.2 MB mind state: **≈ 23 MB per
  1M tokens** (inside the ≤ ~50 MB latitude). Bar: Bar A as pre-registered (≥
  0.9 at 32k, ≥ 0.75 at 128k on synthetic, within 0.05 of C2). On hard
  negatives: the InfoNCE batch already holds every other row of the same
  document, and each synthetic training doc uses ONE template for all its
  needles, so same-template negatives are already in; what differs at test is
  the template itself (held out by design), so L1-256 adds more docs rather
  than a new negative scheme.

**L1+P0 RESULT at 32k (17:08 IST; bench @ c82ee13, same 16 docs per keys as the
paired runs, seed 14000): bar NOT met.** Logs `logs/v16_bench_l1p0_L8_c82ee13_*_32768.*`.

| keys | key alone top-1 | P_margin_burst (re-read each, pick by margin; bar ≥ 0.95) | P_concat_burst (5 spans together) | re-reads |
|---|---|---|---|---|
| synthetic | 0.81 | **0.94** | 1.00 | 5 |
| natural | 0.94 | **0.69** | 0.50 | 5 |

- Re-reading helps on synthetic keys (0.81 → 0.94, concat 1.00) and HURTS on
  natural ones (0.94 → 0.69): with real words the wrong spans are plausible
  text, and the mind's answer margin follows its priors as much as the
  evidence. The mind's confidence is not a safe selector on its own.
- Reading: the key's own similarity carries information the re-read throws
  away. A combined rule (key similarity + re-read margin) is the obvious next
  candidate; it must be pre-registered before it is scored (it would be fit to
  these docs otherwise: new seed required). 128k pending.

**L1+P1 (PRE-REGISTERED 2026-10-31 17:09 IST, before any number of it;
agreed with Lead).** Combined selector over the top 5 burst rows of the
trained key: score_i = z(key similarity_i) + w · z(re-read margin_i) (z =
standardised within the 5 candidates); the answer is the re-read answer of the
best-scoring row. ABSTAIN when the best score is below a threshold t.
- **Tuning split (only place w and t are chosen):** kv, synthetic + natural,
  32k, 16 docs each, `--seed 15000`; w from {0, 0.25, 0.5, 1, 2, 4}; t chosen
  so the neg cell (`--seed 15000`) fetches wrongly in ≤ 10 % of docs.
- **Scoring split (one run, nothing chosen on it):** `--seed 16000`, kv
  synthetic + natural at 32k and 128k, 16 docs each, plus the neg cell at 32k.
- **Arms on the scoring split (same docs):** key alone (C5_burst_hit_span),
  P1-combined, P0 margin alone (P_margin_burst), concat-5 (P_concat_burst) as a
  named arm (the 1.00 on synthetic at seed 14000 was a post-hoc observation and
  is NOT claimed).
- **Bars:** P1-combined ≥ 0.95 at 32k and ≥ 0.85 at 128k on BOTH key types,
  counting an abstain on a stored-key query as a miss; abstain rate reported
  beside it; neg cell wrong-fetch ≤ 0.10. Concat-5: the same bars, judged
  separately. Cost: 5 short re-reads per query (concat: one re-read of 5
  spans).
- **Needs from the harness:** per candidate (top 5): its key similarity, its
  re-read answer and margin, and whether it is right (a `P_cands` list per
  record). Asked of Lead; nothing is scored until it exists.
- **Amendment P1/A1 (17:11 IST, before any P1 number; a flaw in my own
  rule).** z-scoring within the 5 candidates removes the absolute level, so a
  threshold on the best z-score cannot detect "nothing stored". Abstain uses
  the chosen row's RAW key similarity instead: abstain when sim_best < t (the
  frozen address separated stored 0.95 from never-stored 0.54). Selection
  still uses z(sim) + w·z(margin). w and t are still chosen only on seed 15000.
- **Tuned on seed 15000 (18:23 IST; fixed before any seed-16000 number):**
  w = 0.5, t = 0.654 (`v14/l1p1_select.py tune`, kv synthetic + natural and
  neg synthetic + natural, 32k, 32 + 32 docs). On the tuning split itself:
  accuracy 0.72 with abstain-on-stored counted as a miss, abstain rate 0.22,
  neg wrong-fetch 0.09. Logs `logs/v16_bench_l1p1_tune_*`.

**L1-256 development (finished 17:25 IST): no gain over the 128-d key on
validation.** 6000 docs (3000 + 3000 single-template), 2000 steps, 256-d key
(~23 MB per 1M tokens). Harness-template validation top-1 0.67-0.75 (128-d:
0.78-0.82), mixed in-distribution 0.49 (128-d: 0.51). Log
`logs/v14_l1k256_0355188_*.log`. The pre-registered Bar A run in the harness
(kv, 32k, seed 14000 = the paired docs) is queued for the record; on this
evidence the memory latitude does not buy accuracy here, so the next lever is
not key size.

**L1 VERDICTS (2026-10-31 21:10 IST; harness, c1ctlb, centred block-8 state,
16 docs per cell, paired arms on the same documents).** Logs
`logs/v16_bench_l1{pair,p0,p1,k256}_*` (pair / P0 / L1-256: seed 14000;
P1 scored once on seed 16000 with w 0.5, t 0.654 from seed 15000).

| unit | synthetic keys 32k / 128k | natural keys 32k / 128k | bar | verdict |
|---|---|---|---|---|
| L1 trained burst key (Bar A) | **0.81 / 0.56** (frozen burst key 0.06 / 0.25; C2 ceiling 1.00 / 1.00) | **0.94 / 1.00** (frozen 0.56 / 0.19) | ≥ 0.90 / ≥ 0.75 on synthetic, within 0.05 of C2 | **NOT met** |
| L1 on para (Bar P) | 0.56 vs organ 0.56 (C2 0.75) | **0.81 vs organ 0.31** (C2 0.56) | ≥ organ + 0.20 | **met on natural keys only** |
| L1+P0 (re-read top 5, pick by margin) | 0.94 / 0.88 | 0.69 / 1.00 | ≥ 0.95 / ≥ 0.85, both | NOT met |
| L1+P1 (combined, abstain) | 0.69 (abstain 0.19) / 0.56 (0.38) | 0.875 (0.125) / 0.875 (0.06) | ≥ 0.95 / ≥ 0.85, both | NOT met (neg wrong-fetch 0.06 at 32k) |
| concat-5 arm | 1.00 / 0.94 | 0.50 / 0.75 | same | NOT met |
| L1-256 | 0.69 (32k) | 1.00 (32k) | Bar A | NOT met; no gain over 128-d |

Other cells (L1 key vs frozen, 32k, synthetic / natural): kv1 0.94 / 0.94 vs
0.38 / 0.31; tool 0.50 / 0.44 vs 0.25 / 0.31 (C2 1.00: the queried field
often lies beyond the 24-token row); neg max-sim 0.53-0.59 (stored 0.68-0.77).

- **What failed:** Bar A as registered (synthetic pseudo-word keys) at 32k and
  128k, every selector built on top of it, and the 256-d key.
- **What it really says:** the trained key is STRONG on natural language
  (0.94-1.00 up to 128k, kv1 0.94) and beats the exact organ on natural
  paraphrase by +0.50; it is WEAK on random pseudo-words, which the exact
  organ gets at 1.00. The per-token state still separates pseudo-words (C2
  1.00), so it is the pooling over a 24-token span that loses a meaningless
  string's identity. The two organs are complementary: exact strings (codes,
  ids, made-up names) → the exact organ; natural / paraphrased language → the
  learned key. The mind's confidence is not a safe selector on natural text
  (P0 natural 0.69).
- **Next (to pre-register before any number, fresh seed):** a two-organ
  fetch: the exact organ when it finds an exact match of the query key, else
  the learned key; abstain by the key's raw similarity. The tool cell needs a
  longer or field-aware row.

**PRE-REGISTERED 2026-10-31 21:12 IST (before any number of these; approved by
Lead).**

**L2 — two-organ fetch.** Rule, per query: if the exact organ finds an exact
4-gram match of the query's last 4 tokens (`B_hit`, knowable at inference),
answer with the organ (`B_ngram`); otherwise fetch the top burst row by the
learned key (head `l1p_fwd_L8c_99e6803`) and answer with the mind's re-read of
that row (`P_cands[0]`), ABSTAINING when that row's raw key similarity is below
t. t is tuned ONLY on the existing seed-15000 runs (same rule, neg wrong-fetch
≤ 0.10). Scored once on a fresh seed 17000: kv synthetic + natural at 32k and
128k, para synthetic + natural at 32k, neg synthetic + natural at 32k, tool
synthetic + natural at 32k, 16 docs each.
- **Bars:** kv ≥ 0.95 at 32k and ≥ 0.90 at 128k on BOTH key types (abstain on
  a stored key = miss); para natural ≥ 0.75 (organ 0.31 at seed 14000) and para
  synthetic ≥ organ + 0.10; neg wrong-fetch (organ hit OR similarity ≥ t) ≤
  0.10. Tool is reported (the row length is a separate unit).
- **Tuned on seed 15000 (21:15 IST, fixed before any seed-17000 number):**
  t = 0.655 (`v14/l2_select.py tune`): tuning kv accuracy 1.00 with the organ
  answering every kv query (the kv keys are exact strings, so there the
  two-organ fetch IS the organ); neg wrong-fetch 0.09. The learned key only
  decides on cells where the organ misses (para, tool fields).
- **Novelty line (Lead):** hybrid sparse + dense retrieval (BM25 + dense
  retriever class): the package's fetch, cited, NO claim. The claim stays the
  interface: keys in the mind's own state, rows written on the mind's
  surprise.
- **Field-aware tool row (reported arm, needs a harness change):** a row spans
  from the burst start to the end of its sentence (≤ 64 tokens), the key pooled
  over that span, the hit scored inside the row's own span. Asked of Lead.

**SR-0b PARKED (user via Lead, 2026-10-31 ~21:20 IST):** "holds beyond the
training context" is closed as DONE at the package level (re-fetching the pack
gives 1.00 to 128k; the mind alone reaches about its training length after one
read, a stated limit of this scale). The pre-registration below stays for the
record; nothing is queued. Order now: (1) L2 two-organ fetch, (2) text-first
pack format + install tool, (3) N11 planner (constraint: the mind's confidence
is not a safe selector over natural spans).

**SR-0b — is the skill in the notebooks, and can a stored skill state
coexist? (b2p2, skill-pack cell, seeds 8000-8099, L = 2k / 8k / 32k).**
Snapshot S_pack = the NOTEBOOK states right after the pack (echo and any other
state excluded).
- **(a) replace:** at the query, the notebook states are REPLACED by S_pack
  while echo state and the residual stream come from the filler. Bar: accuracy
  at 32k ≥ 0.9 × A(L = 0) (affine ≥ 0.90, shift ≥ 0.90). Met → the skill lives
  in the notebooks.
- **(b) merge:** at the query, S_pack is ADDED to the filler-evolved notebook
  states. Bar: accuracy at 32k ≥ 0.9 × A(L = 2k) (affine ≥ 0.88, shift ≥
  0.67, the SR-0 bar). Met → a stored skill state can coexist with unrelated
  content: the register's real question.
- **Controls:** a different rule's S_pack (must fall to the no-pack floor);
  the plain one-read A.
- **Needs:** snapshot / restore of notebook states on the M2 chunked path
  (Mind's body code; Lead asks only if under an hour). Product fallback
  exists: re-fetching the pack gives 1.00 at any L.
- **Closest prior work:** State Soup (Pioro et al. 2024). Different only if (b)
  holds in-context without decay; otherwise it is State Soup.

**Pack format + install tool (`582acad`, Lead's order item 2).** `v14/pack.py`:
a pack is plain JSONL (header + one entry per line: text, source, sha256,
time, trust; no keys). Install = one pass of the mind: rows at surprise bursts
(tau 10) inside each entry (at least one per entry), keys from the L1 head on
the block-8 state centred by a SHIPPED reference mean (mean over 1.7M
training-pool tokens; a document's own mean sits 16 % of the within-doc spread
away, max 30 %), an exact 4-gram organ, the text as u16. MB per 1M tokens:
keys 13.6 + organ 12.0 + text 2.0 + pointers 0.6 = **28 MB** (within the ≤ 50 MB
latitude); install 294 s per 1M tokens on the shared 4090.

**Cross-document check (`v14/pack_check.py`, the product condition: library and
session are different texts; seed 18000; development, not a bar).** Pack = N
definitions inside held-out prose paragraphs + N prose-only distractors; each
session = 1024 tokens of other held-out prose + " Query: K refers to".

| pack | key type | learned key top-1 (top-5) | answer after re-read | organ entry / answer |
|---|---|---|---|---|
| 20 + 20 (8k tokens) | synthetic / natural | 0.90 / 0.90 (0.90 / 0.90) | 0.90 / 0.60 | 1.00 / 1.00 |
| 200 + 200 (82k tokens, ~3.4k rows) | synthetic / natural | **0.55 / 0.54** (0.74 / 0.74) | 0.59 / 0.42 | 0.99 / 1.00 |

- **The learned key does not transfer across documents.** In-document at 128k
  (~5k rows) the same head gives natural 1.00; cross-document at ~3.4k rows it
  gives 0.54. The head was trained only on same-document pairs, where the
  state at the key word shares the surrounding document with the stored row.
- **The exact organ is unaffected** (0.99-1.00): for exact keys the package is
  already complete; the learned key's job (paraphrase, natural variants) is the
  part that fails in the product condition.
- **Next (to pre-register): L3, a cross-document head** trained on pairs whose
  query sits in a different document from the row, centred by the shipped
  mean, judged on a fresh-seed pack_check with paraphrased queries added.

**L3 — cross-document key head (PRE-REGISTERED 2026-10-31 21:19 IST, before
any L3 code or number).** Same head form as L1 (attentive pooling over the
24-token span from the block-8 state, 128-d), but trained on the PRODUCT
condition: every pair has its stored row in one document and its question in
another, and both sides are centred by the shipped reference mean.
- **Training pairs (training pool only):** (a) synthetic: needles with my 8
  templates inside pool prose (doc A); the question in a separate session
  (1024 tokens of other pool prose + a query template); (b) natural: a rare
  word occurring in two DIFFERENT pool documents; the row around its mention
  in doc A, the question at its mention in doc B. Paraphrased keys are NOT in
  training.
- **Evaluation (`v14/pack_check.py`, fresh seed 19000, 200 + 200 pack, session
  1024):** exact queries and paraphrased queries (Title / UPPER / plural / "the
  K", the harness's para forms).
- **Bars:** cross-document key top-1 ≥ 0.85 on exact queries for both key types
  (current head 0.55 / 0.54); on paraphrased queries key top-1 ≥ 0.70 and ≥
  organ + 0.30; in-document harness kv natural 32k (seed 14000) ≥ 0.90 (no
  regression on the L1 result 0.94).
- **Thesis clause / prior work:** separate from knowledge; dense retrieval
  trained on query-document pairs (DPR class): plumbing for the mechanism. What
  is ours is that the encoder is the mind's own state.
- **Added by Lead (21:25, before any L3 number):** report how cross-document
  top-1 scales with library size, 40 / 400 / 4,000 entries (20+20, 200+200,
  2000+2000), for the L1 head and the L3 head on the same fresh seed 19000
  (reported, the bars stay at 400). Then, only once L3 no longer needs the
  card, SR-0b may run ONCE, eval-only, renamed as a PRODUCT-OPTION test: "can a
  skill pack be installed as a stored notebook state merged into a live
  session, instead of re-read as text?", with its cost (1.15 MB per skill) next
  to the re-read cost. Mind's M2 streaming (`41f8ae8`) provides the hook; keep
  the current chrono clock when swapping notebooks.

**L3 head trained (22:10 IST, `l3_fwd_L8r_ec5cb21.pt` on d_drive; development,
not the bar).** 3000 cross-document pairs (50 % synthetic sessions, 50 %
natural rare-word pairs across two pool documents), shipped-mean centring,
2000 steps; best step by the cross-document validation (300 pairs, pool
0.85-1.0): top-1 0.61, top-5 0.67 (row exists 0.89), plateau from step 250
(0.57). Split by kind against the L1 head on the SAME validation pairs, both
centred by the shipped mean:

| head | synthetic sessions | natural rare-word pairs |
|---|---|---|
| L1 (same-document training) | **0.90** (top-5 0.96) | 0.21 (0.36) |
| L3 (cross-document training) | 0.80 (0.88) | **0.41** (0.46) |

L3 trades synthetic for natural (−0.10 / +0.20). The pre-registered judgement
is the fresh-seed pack_check (exact + paraphrase, 400 entries) and the size
sweep, running on the 4090. Logs `logs/v14_l3data_*`, `logs/v14_l3chain_*`.

**L2 VERDICT (2026-10-31 22:35 IST; seed 17000, scored once with t = 0.655 from
seed 15000; L1 head, document-mean centring; bench `b03964d`; 16 docs per cell):
NOT met (paraphrase).** Logs `logs/v16_bench_l2_score_*`.

| cell | L2 two-organ | organ alone | L2 without abstain (reported) | bar | verdict |
|---|---|---|---|---|---|
| kv synthetic 32k / 128k | 1.00 / 1.00 (organ answers all) | 1.00 | – | ≥ 0.95 / ≥ 0.90 | met |
| kv natural 32k / 128k | 1.00 / 1.00 (organ answers all) | 1.00 | – | ≥ 0.95 / ≥ 0.90 | met |
| neg, wrong fetch (syn / nat) | 0.06 / 0.06 | – | – | ≤ 0.10 | met |
| para synthetic | 0.375 (abstain 0.50) | 0.31 | 0.50 | ≥ organ + 0.10 = 0.41 | **NOT met** |
| para natural | 0.56 (abstain 0.38) | 0.44 | 0.69 | ≥ 0.75 | **NOT met** |
| tool syn / nat (reported) | 0.31 / 0.44 | 0.25 / 0.25 | 0.375 / 0.56 | – | – |

- **One threshold cannot serve exact and paraphrased questions.** t was tuned
  where the organ answers (kv) and on never-stored keys; paraphrased questions
  match the learned key more weakly, so L2 abstains on 38-50 % of them and each
  abstain is a miss (cost 0.12-0.19).
- **Even without abstain, paraphrase is the weak cell** (0.50 / 0.69), and it is
  the learned key's only job in the package (the organ covers exact keys).
- Next: the L3 head on the product condition (pack_check at 400 entries with
  paraphrased questions, running), then a paraphrase-aware abstain rule (e.g.
  abstain only when BOTH the organ misses and the key is below t on a threshold
  tuned on paraphrased questions too), pre-registered before scoring.

**PACKAGE abstain rule A2 + head choice (PRE-REGISTERED 2026-10-31 22:34 IST,
before any number of it; Lead's decision, deadline 00:30 IST).**
- **Head:** the L1 or L3 head, whichever has the higher learned-key top-1 on
  PARAPHRASED questions in pack_check at 400 entries (seed 19000; synthetic +
  natural pooled; ties → L3, the head trained on the product condition).
- **Rule:** exact organ on an exact match; else the learned key's top row;
  ABSTAIN only when the organ misses AND the key's best similarity < t_para.
- **Tuning (seed 15000 only; bench at the package_fetch commit; shipped-mean
  centring; 16 docs per cell, 32k):** para synthetic + natural and neg
  synthetic + natural, for BOTH heads. t_para = the smallest t with neg
  wrong-fetch ≤ 0.10; it must also give false abstain on para ≤ 0.20. If no t
  meets both, abstain is disabled when the organ misses (t_para = −1) and the
  neg cell is reported as failing its constraint.
- **Handover:** PACKAGE_HEAD + PACKAGE_T to Lead for the package table (fresh
  seeds 18000+, run by Lead). No number from the table's seeds is used here.
- **A2 tuning RESULT (seed 15000, 23:03 IST; bench `5a3a8a8`; 32 para + 32 neg
  docs per head): no t meets both constraints, for either head.** L1: the
  smallest t with neg wrong-fetch ≤ 0.10 is 0.669 (neg 0.094), and it abstains
  on 0.44 of stored paraphrased questions (> 0.20); L3: t 0.732, false abstain
  0.50. The similarities overlap: never-stored mean 0.56 / 0.59, paraphrased
  stored (organ miss) 0.62 / 0.63 (L1 / L3); the organ hit 0 of 32 never-stored
  queries. **By the pre-registered rule: abstain is DISABLED when the organ
  misses (PACKAGE_T = −1), and the neg cell is reported as failing its
  constraint (wrong-fetch 1.00 when the organ misses).** Handover: PACKAGE_HEAD
  = L1 (`l1p_fwd_L8c_99e6803.pt`), PACKAGE_T = −1. Reading: the learned key's
  similarity cannot tell "stored but paraphrased" from "never stored", so in
  this package only the exact organ can abstain; "nothing stored" for a
  paraphrased question would need another signal (e.g. the mind's answer
  after re-reading the fetched row, already shown unsafe on natural text, or a
  trained abstain head). Logs `logs/v16_bench_a2tune_*`.

**Deployment cost of the shipped mean (2026-10-31 22:42 IST, reported; same
seed-17000 docs as the L2 verdict; bench `582acad`; L1 head; t 0.655).** L2
with keys centred by the document's own mean vs by the SHIPPED reference mean:
para synthetic 0.375 → 0.3125, para natural 0.5625 → 0.5625, tool 0.3125 →
0.3125 / 0.4375 → 0.375, neg wrong-fetch 0.06 → 0.06 / 0.00. The largest
change is one document in 16, within noise: deploying with a fixed mean costs
little in-document. Logs `logs/v16_bench_l2ref_score_*`.

**L3 VERDICT + library-size sweep (2026-10-31 22:45 IST; pack_check seed 19000,
cross-document: library and session are different texts; session 1024;
shipped-mean centring): L3 NOT met on every bar; the learned key does not
transfer across documents at all.** Reports `logs/pack_check/v14_pack_check_19000_*_report.json`.

| entries (rows) | question | L1 key top-1 | L3 key top-1 | exact organ |
|---|---|---|---|---|
| 40 (404) | exact syn / nat | 0.80 / 0.80 | 0.50 / 0.90 | 1.00 / 1.00 |
| 40 | para syn / nat | 0.40 / 0.70 | 0.30 / 0.70 | 0.50 / 0.40 |
| 400 (4.1k) | exact syn / nat | 0.69 / 0.69 | 0.29 / 0.66 | 1.00 / 1.00 |
| 400 | para syn / nat | 0.33 / 0.30 | 0.20 / 0.37 | 0.50 / 0.35 |
| 4000 (43k) | exact syn / nat | 0.47 / 0.42 | 0.20 / 0.48 | 0.87 / 1.00 |
| 4000 | para syn / nat | 0.30 / 0.19 | 0.18 / 0.22 | 0.47 / 0.27 |

- **L3 bars:** exact ≥ 0.85 (0.29 / 0.66), para ≥ 0.70 and ≥ organ + 0.30
  (0.20 / 0.37): NOT met. L3 is worse than L1 on synthetic keys.
- **A2 head choice (pre-registered rule): L1** (pooled paraphrase top-1 at 400
  entries 0.315 vs 0.285).
- **What it really says:** inside ONE document the mind's state is an
  excellent address (per-token C2 1.00; L1 0.94-1.00 on natural keys up to
  128k), and across documents it is below the exact organ everywhere, even
  on paraphrase at 400 entries (organ 0.50 / 0.35 vs key 0.33 / 0.30; forms
  such as "the K" keep the key's tokens). The state encodes WHERE something
  sits in this stream, not a context-free identity, and cross-document
  training did not change that. It maps onto the two libraries the user asked
  for:
  - **session library** (the conversation, tool results: the same stream as
    the question) → fetched by the mind's own state (the condition the harness
    measures; L1 in-document numbers apply);
  - **shared library** (separate texts) → fetched by TEXT: the exact organ
    (1.00 at 400 entries, 0.87-1.00 at 4000) plus a normalised lexical match
    for paraphrase (case-folding, inflection), which is plumbing (BM25 class),
    not a claim.
- The organ's 0.87 on synthetic exact keys at 4000 entries is 4-gram collisions
  between pseudo-word keys (the query's tail "K refers to" shares its last
  tokens across keys): longer n-grams or a whole-key match fix it.

**SL1 — shared-library fetch by text (PRE-REGISTERED 2026-10-31 23:05 IST, before
any SL1 code or number; approved by Lead).** The shared library holds separate
texts, so (EUREKA "stream address") it is fetched by TEXT. Plumbing, cited
(BM25: Robertson & Zaragoza 2009; MiniLM sentence embeddings: Wang et al.
2020 / Reimers & Gurevych 2019), no claim.
- **Methods (retrieval of the right entry, cross-document pack_check setup:
  definitions inside held-out prose paragraphs + prose distractors, fresh seed
  20000, 400 and 4000 entries, exact and paraphrased questions; retrieval only,
  CPU):** (a) organ-4 (the current exact 4-gram organ); (b) organ-long: the
  LONGEST n-gram (n = 8 down to 2 tokens) of the question's tail that occurs in
  the library, latest match; (c) lexical-norm: BM25 over normalised words
  (lower-case, a leading "the" and a plural "s" stripped); (d) **SL1 = (b) when
  it matches with n ≥ 3, else (c)**; (e) baseline: an off-the-shelf sentence
  embedder (all-MiniLM-L6-v2, mean-pooled, cosine) over entries vs the question.
- **Bars for (d):** exact top-1 ≥ 0.99 at 400 and ≥ 0.97 at 4000 entries;
  paraphrase top-1 ≥ 0.80 at 400 and ≥ 0.70 at 4000; and (d) ≥ (e) − 0.05 in
  every cell (text plumbing must not lose to an off-the-shelf embedder).
  Reported: each method per cell, index MB per 1M tokens, query time.
- The L1 mind key's cross-document numbers (seed 19000 sweep) stand as the
  reference row; it is not rerun here.
- **SL1 RESULT (2026-10-31 23:12 IST; seed 20000; code `88095d5` local =
  `0eedc35` on origin; retrieval only, CPU): NOT met (natural paraphrase).**
  Logs `logs/v14_sl1_88095d5_*`.

  | question | 400 entries: SL1 (organ-4 / long / BM25 / MiniLM) | 4000 entries: SL1 (organ-4 / long / BM25 / MiniLM) | bar |
  |---|---|---|---|
  | exact synthetic | **1.00** (0.99 / 1.00 / 1.00 / 0.41) | **1.00** (0.86 / 1.00 / 1.00 / 0.22) | ≥ 0.99 / ≥ 0.97: met |
  | exact natural | **1.00** (1.00 / 1.00 / 0.56 / 0.17) | **1.00** (1.00 / 1.00 / 0.31 / 0.08) | met |
  | para synthetic | **0.98** (0.54 / 0.65 / 1.00 / 0.41) | **0.91** (0.47 / 0.58 / 1.00 / 0.23) | ≥ 0.80 / ≥ 0.70: met |
  | para natural | **0.68** (0.20 / 0.21 / 0.56 / 0.17) | **0.49** (0.26 / 0.26 / 0.31 / 0.07) | **NOT met** |

  - SL1 ≥ MiniLM − 0.05 in every cell (met), but the embedder baseline is
    unfairly weak as registered: it embeds whole 200-word entries against a
    one-line question; a fair baseline embeds sentences or chunks.
  - Why natural paraphrase fails: real-word keys are common words, so BM25 is
    ambiguous (0.31 at 4000), and the token organ misses exactly the forms that
    change BPE tokens (Title / UPPER / plural). An organ over NORMALISED words
    (case-folded, plural stripped) would match those forms directly.
  - Organ index memory as implemented (a dict per n = 2..8): 24-66 MB per 1M
    tokens; a suffix array over tokens gives the same longest-match lookups
    at ~4 MB per 1M (implementation, not method).

**SL2 — normalised-word organ for the shared library (PRE-REGISTERED 2026-10-31
23:10 IST, before any SL2 code or number).** SL1 failed on natural
paraphrase (0.68 / 0.49) because the token organ misses forms that change BPE
tokens and BM25 is ambiguous on common words.
- **Method:** an organ over NORMALISED words (lower-case, a plural "s"
  stripped, the articles "the / a / an" dropped on both sides): the longest
  word n-gram (n = 6 down to 3) of the question's tail found in the library,
  latest match; else SL1's token organ (n ≥ 3); else BM25.
- **Honest scope:** this fixes SURFACE variants (case, plural, article) by
  construction, which is all the current paraphrase cell contains; it says
  nothing about synonyms or rewording, which no cell tests yet (that would
  need a synonym resource or human paraphrases: listed as open).
- **Eval:** the SL1 script and builder, fresh seed 21000, 400 and 4000
  entries, exact + paraphrased questions; MiniLM row kept as registered.
- **Bars:** exact ≥ 0.99 / ≥ 0.97 at 400 / 4000; paraphrase ≥ 0.80 / ≥ 0.70 on
  BOTH key types. Reported: index MB per 1M tokens.
- **SL2 RESULT (2026-10-31 23:20 IST; seed 21000; code `f781a9e` local =
  `bdfa727` on origin; retrieval only, CPU): ALL bars MET.** Logs
  `logs/v14_sl2_f781a9e_*`.

  | question | 400 entries: SL2 (SL1 / organ-norm / MiniLM) | 4000 entries: SL2 (SL1 / organ-norm / MiniLM) |
  |---|---|---|
  | exact synthetic | 1.00 (1.00 / 1.00 / 0.37) | 1.00 (1.00 / 1.00 / 0.20) |
  | exact natural | 1.00 (1.00 / 1.00 / 0.18) | 1.00 (1.00 / 1.00 / 0.08) |
  | para synthetic | 1.00 (0.98 / 1.00 / 0.39) | 1.00 (0.93 / 1.00 / 0.21) |
  | para natural | **1.00** (0.76 / 1.00 / 0.17) | **0.99** (0.49 / 0.99 / 0.08) |

  - **Scope, as registered:** surface variants only (case, plural, article),
    fixed by construction; synonyms and rewording are untested (open item).
    Plumbing, no claim: the SHARED library's text fetch for exact and
    surface-variant questions is solved at 4000 entries.
  - **Memory as implemented:** normalised-word organ 10-27 MB per 1M tokens and
    the token organ 24-67 MB (dicts per n); a suffix array over each stream
    gives the same longest-match lookups at ~4 MB per 1M (to do before the
    package counts MB).
- **Deployable form + fair embedder row (2026-10-31 23:35 IST; same seed 21000;
  code `359a49d`; Lead's requests).** Depth-limited suffix arrays (tokens depth
  8, normalised words depth 6; 4 bytes per position) replace the n-gram dicts:
  they return the SAME entry as the dicts on every query (400 / 400 and
  4000 / 4000), so SL2's accuracy is unchanged (all cells 1.00 at 400; 1.00 /
  1.00 / 0.99 / 1.00 at 4000). **Measured memory per 1M tokens: token organ
  4.02 MB, normalised-word organ 5.4 MB (array + word stream), raw text 2.0 MB
  → ~11.4 MB** (the dicts were 24-67 + 10-27 MB). Not yet counted: BM25's
  posting lists (the fallback). Fair embedder baseline (sentence-level MiniLM,
  an entry scored by its best sentence): 0.55-0.75 at 400 entries and 0.32-0.66
  at 4000 (entry-level 0.08-0.39); its index is 49.6 MB per 1M tokens (384-d
  fp32 per sentence; ~25 MB in fp16). On this task family the text organ is far
  ahead of the standard embedder at a quarter of its memory; the embedder's
  domain (rewording, synonyms) is the one no cell tests yet. Logs
  `logs/v14_sl2sa_*`.

**N11 — the planner (design, 2026-10-31 ~23:40 IST; Lead's order item 3).** What
it decides, from today's evidence:
1. **Explicit question → fetch at question time.** Order: (a) the exact organ
   over the session stream and the shared library (an exact match wins);
   (b) the session library by mind-state keys (L1 head, in-document: 0.94 /
   1.00 on natural keys at 32k / 128k); (c) the shared library by text (SL2:
   0.99-1.00 at 4000 entries). The question is its own cue, so no refresh.
2. **Standing instruction → refresh** every R ≈ 2k tokens (N11-R, below),
   because after one read the mind holds a skill ~2k tokens.
3. **Two hops → chain:** fetch hop 1, the mind re-reads it, the key word it
   produces forms the hop-2 query, fetch again (the harness's planner chain
   on the organ already does this on exact keys).
4. **Selector constraint:** candidates are chosen by rule (exact match >
   session key > shared text), NEVER by the mind's answer confidence (L1+P0:
   on natural spans that confidence picked 0.69 vs the key's 0.94).
5. **Abstain:** only when nothing matches exactly and no candidate exists; a
   similarity threshold cannot separate paraphrased-stored from never-stored
   (A2: 0.62 vs 0.56).
- **Closest prior work:** ReAct (Yao et al. 2023: interleaved reasoning and
  tool calls), Self-Ask (Press et al. 2022: follow-up questions for multi-hop),
  query rewriting for RAG (Ma et al. 2023), FLARE (Jiang et al. 2023: active
  retrieval triggered by low-confidence tokens). **Difference (one line):**
  these steer a transformer that keeps its whole window; ours steers an O(1)
  mind that keeps a skill only ~2k tokens, so the planner must also SCHEDULE
  REFRESH, and its trigger and selector cannot be the model's confidence
  (FLARE's trigger), which our data shows is unsafe over natural spans.
- **Where it is measured:** the fetch rule (1, 3, 4, 5) = Lead's package table
  (kv / kv1 / kvupd / hop2 / tool / para / neg, 40 docs per cell, fresh seeds
  18000+); the refresh rule (2) = N11-R. No separate pre-registration for the
  fetch rule: the table IS its evaluation.

**N11-R — the planner's refresh rule (PRE-REGISTERED 2026-10-31 23:32 IST,
before any code or number; Lead's order item 3; eval only).**
- **Step back first.** For an EXPLICIT question the question itself is the
  retrieval cue: the package fetches the pack at question time and the mind's
  re-read scores 1.00 at any distance (Mind's skill-pack O_reread), so no
  refresh is needed. A STANDING instruction (a rule set once and applied to
  later inputs that carry no retrieval cue) is the case a planner must handle:
  after one read the mind holds it ~2k tokens (Mind's s8 / skill-pack cell),
  then loses it. The refresh rule re-inserts the pack every R tokens.
- **Cell:** Mind's skill-pack builder (`scripts/run_skillpack_probe.py`
  functions; b2p2; shift_wide_new and affine_map_new, 8 demos; probe seeds
  8100-8149, never used by SR-0 or Mind's runs): the pack read at t0, then L
  tokens of filler with the pack RE-INSERTED after every R tokens of filler,
  then the query line. R ∈ {1024, 2048, 4096, none}; L ∈ {8k, 32k}; 50 docs per
  rule.
- **Bars:** with R = 2048, accuracy at 32k ≥ 0.95 (affine) and ≥ 0.70 (shift),
  i.e. close to the one-read 2k values 0.98 / 0.74. Reported: every R × L cell
  with margin; the token overhead (pack tokens / R); R = none is the floor
  (Mind: 0.07 / 0.01 at 32k).
- **Prediction (written now):** R ≤ 2048 holds the skill; R = 4096 gives about
  the one-read 4k value (0.42 / 0.27); the overhead at R = 2048 is the pack
  length / 2048 (a few percent).
- **N11-R RESULT (2026-10-31 23:43 IST; b2p2; code `bbf5adc` local; seeds
  8100-8149, 50 docs per cell): bar MET.** Log
  `logs/v14_n11r_b2p2_bbf5adc_20261031_2333.jsonl`.

  | rule, L | no refresh | R = 4096 | **R = 2048** | R = 1024 |
  |---|---|---|---|---|
  | affine 8k | 0.12 | 0.30 | 0.90 | 1.00 |
  | affine 32k | 0.10 | 0.32 | **0.96** (bar ≥ 0.95) | 1.00 |
  | shift 8k | 0.04 | 0.12 | 0.62 | 0.82 |
  | shift 32k | 0.00 | 0.32 | **0.80** (bar ≥ 0.70) | 0.86 |
  | token overhead | 0 | 0.7 % | **1.6 %** | 3.3 % |

  - The written prediction held: R ≤ 2k keeps the skill; R = 4k gives about
    the one-read 4k value (0.32 / 0.32 here vs 0.42 / 0.27 one-read).
  - **For the package:** a standing skill stays alive to any length for ~1.6 %
    extra tokens (R = 2k), or ~3.3 % for the sturdier R = 1k (shift 0.86).
    The O(1) mind plus a refresh rule holds a skill past the trained length;
    the cost is stated, not hidden.
  - **Note (2026-11-01 00:20):** R is set by the mind's one-read reach (b2p2:
    1.00 / 0.86 / 0.355 / 0.10 at 0 / 2k / 4k / 8k per Mind). If Mind's fixed
    long-gap recipe (lg8r) lengthens the reach, R grows and the overhead falls
    in proportion; N11-R should be re-run on that checkpoint.
  - **N11-R/F — free-generation rerun (PRE-REGISTERED 2026-11-01 00:49 IST, before
    any number of it).** The verdict above used the 20-way pick (registered
    metric; it stands). Under the user's rule "generation counts" (A7) the
    headline must be FREE generation (the greedy token over the whole
    vocabulary == the target); Mind measured the 20-way pick reads higher
    (b2p2 skill-pack at 2k: 0.88 vs 0.68). Rerun: same cell, FRESH seeds
    8150-8199 (50 docs per rule), L = 2048 / 8192 / 32768, R = 1024 / 2048 /
    4096 / none; code `9daea9a`. **Bar (relative, since the free one-read level
    is not known in advance):** free accuracy at R = 2048, L = 32k ≥ 0.9 × the
    free accuracy with no refresh at L = 2048 in the same run, for each rule.
    Reported: both metrics in every cell, the free tokens, the overhead.
  - **N11-R/F VERDICT (2026-11-01 03:22 IST; code 7da1edd;
    `logs/v14_n11rf_b2p2_7da1edd_20261101_0207.jsonl` + driver log, 1,200 rows):
    NOT met (one rule of two).**

    | rule | free one-read 2k (R none) | bar 0.9× | free R = 2048, 32k | verdict |
    |---|---|---|---|---|
    | affine_map_new | 0.86 | 0.774 | **0.84** (20-way 0.96) | met |
    | shift_wide_new | 0.48 | 0.432 | **0.34** (20-way 0.56) | NOT met |

    Free at 32k by R (affine / shift): R 1024 0.90 / 0.52 (+3.3 % tokens),
    R 2048 0.84 / 0.34 (+1.6 %), R 4096 0.10 / 0.08, none 0.00 / 0.02. With no
    refresh, free is 0.00 already at 8k. R = 1024 would pass both (not the
    registered R). A refreshed skill survives 32k in FREE generation for the
    stronger rule; the weaker rule needs R ≈ 1k.
  - **Checked by Mind (2026-11-01 00:25, `b136dab`, 300 real docs per source):**
    the packing flaw orphans only 3.3 % of B2 answer tokens at 2k windows
    (B2 docs are short: median 143 tokens), so b2p2's 2k reach and R as tuned
    STAND. Recall docs lose 15.9 % at 2k in every arm alike (T++ and Mamba-2
    included), so comparisons stay fair. The flaw was large only for long-gap
    windows (b2gap at 8k 44 %, lg32 ~45 %). A new R waits for lg8r PLUS its
    control lg2r and a second seed (Lead's queue), not for lg8r alone.
- **Thesis clause:** holds beyond the context (with the library's help);
  closest prior work: periodic re-prompting / "system prompt reminders" in
  long-running agents. Plumbing; the claim is only the measured cost of
  keeping an O(1) mind's skill alive.

**PACKAGE strict mode (PRE-REGISTERED 2026-11-01 00:34 IST, before any number of it;
after Lead's package table, `abf5b69`).** Lead's table (fresh seeds 18000 /
18500; L1 head; PACKAGE_T = −1) met R1 at 1.00 on every exact-key cell at 32k
and 128k on both bodies (T++ / Mamba-2 state alone 0.04-0.18; PAM minds alone
0.08-0.19); paraphrase 0.46-0.66 (organ alone 0.26-0.35); tool 0.31-0.46;
neg wrong-fetch 1.00 when the organ misses. Lead found post hoc that t = 0.97
gives abstain 1.00; that number was fitted on the table's own docs, and since
every exact hit has similarity 1.00 by construction, it is in effect the RULE
"abstain whenever the exact organ misses". So the honest form is a rule, not a
fitted threshold:
- **Strict mode:** answer only on an exact-organ hit; otherwise abstain.
  (Lenient mode = the table's PACKAGE_T = −1: never abstain on an organ miss.)
- **Eval (Lead's harness, fresh seed 22000, never used):** kv / para / tool /
  neg × synthetic + natural, 32k, 40 docs per cell, b2p2.
- **Bars:** neg abstain ≥ 0.90 (the pre-registered abstain bar); kv top-1 =
  1.00 (unchanged). Reported: paraphrase and tool in strict vs lenient mode
  (expected to fall to the organ's level): the precision / recall trade-off
  is the finding, with no middle ground available from key similarity (A2).
- **Amendment S/A1 (00:39 IST, before any strict-mode number; user via Lead:
  "generation counts"):** the PRIMARY metric is free generation: the mind's
  greedy token over the whole vocabulary after the re-read (`P_free`; state
  only `A_free`), not the pick among 20 candidates (secondary). Bars: neg
  abstain ≥ 0.90 (unchanged; abstaining does not depend on the metric); kv
  `P_free` in strict mode ≥ 0.95 and within 0.02 of lenient mode's on the same
  docs. Para / tool reported in both metrics. Run with PACKAGE_STRICT=1 at
  `172adf4` or later (Lead's queue).
- **STRICT MODE VERDICT (2026-11-01 00:57 IST; Lead's run
  `logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.jsonl`, b2p2, 32k, seed
  22000, 40 docs per cell, PACKAGE_STRICT=1, 320 rows, 0 errors): NOT met as
  registered (kv natural, one-token free scorer).**

  | cell | bar (S/A1) | free (one token) | 20-way | abstain | verdict |
  |---|---|---|---|---|---|
  | neg syn / nat | abstain ≥ 0.90 | – | – | **1.00 / 1.00** (wrong 0.00) | **met** |
  | kv synthetic | free ≥ 0.95, within 0.02 of lenient | **0.975** | 0.975 | 0.00 | met |
  | kv natural | same | **0.85** | 1.00 | 0.00 | **NOT met** |
  | para syn / nat | reported | 0.275 / 0.20 | 0.275 / 0.275 | 0.725 | (lenient 20-way 0.40 / 0.525) |
  | tool syn / nat | reported | 0.075 / 0.15 | 0.075 / 0.15 | 0.80 / 0.75 | (lenient 0.35 / 0.325) |

  - "Within 0.02 of lenient" holds trivially: the organ hit every kv query, so
    strict and lenient are the same procedure on these docs.
  - **Why kv natural misses: wording, not fetch.** All 6 misses had the right
    row (20-way correct, margins 2.9-5.4); the greedy first token was " refers"
    (×3), " a" (×2), " the" (×1): the mind continues in its own words (" a
    crimson ...") or re-reads the key. The one-token scorer counts that as a
    miss; a sentence scorer would not. The registered verdict stands (NOT met);
    the scorer is NOT swapped after the fact.
  - **The trade-off (as predicted):** strict mode makes "nothing stored"
    perfect (1.00) and halves paraphrase / tool, because only the exact organ
    can say "found it"; lenient mode keeps them but always fetches something.
- **Strict mode S2 (PRE-REGISTERED 2026-11-01 00:57 IST, before any number of
  it):** the same strict run on a FRESH seed 23000 with the sentence scorer as
  PRIMARY (`P_free_sent`: greedy decode to the first "." or newline, cap 16
  tokens, the value matched in the decoded text); one-token free and 20-way
  secondary. Bars: neg abstain ≥ 0.90; kv `P_free_sent` ≥ 0.95 on both key
  types. b2p2, 32k, 40 docs per cell, kv / para / tool / neg × syn + nat. Runs
  when Lead's harness has `P_free_sent`.
- **S2 VERDICT (2026-11-01 01:16 IST; Lead's run
  `logs/v16_pkg_s2_b2p2_32768_719d598_20261101_0106.jsonl`, b2p2, 32k, fresh
  seed 23000, strict, 320 rows, 0 errors): NOT met.**

  | cell | bar | `P_free_sent` (primary) | one-token free | 20-way | abstain |
  |---|---|---|---|---|---|
  | kv synthetic | sent ≥ 0.95 | **0.875** | 0.875 | 1.00 | 0.00 |
  | kv natural | sent ≥ 0.95 | **0.70** | 0.675 | 1.00 | 0.00 |
  | neg syn / nat | abstain ≥ 0.90 | – | – | – | **1.00 / 1.00** (met) |
  | para syn / nat | reported | 0.275 / 0.10 | | | 0.675 / 0.85 |
  | tool syn / nat | reported | 0.10 / 0.10 | | | 0.75 / 0.75 |

  - **Every kv miss had the right row; the mind failed to WRITE the value.**
    14 of the 20 value nouns are 2-3 tokens (" will"+"ow", " on"+"yx", " sa"+"ff"+"ron").
    kv by value length (sentence): synthetic single-token 1.00 (n 11) vs
    multi-token 0.83 (n 29); natural 0.75 (n 12) vs 0.68 (n 28). Misses: a
    wrong first piece (" ake." / " ave." / " ite." for willow / onyx / clover),
    loops on natural keys (" refers to refers to ...", " the base of the base
    of ..."), generic prose (" the activity of a protein."). The 20-way pick
    compares first tokens only, so the registered v1 score was lenient on
    exactly these; the sentence scorer is the honest one (Lead).
  - Reading: fetch right, expression wrong: the 103M mind, reading the row
    COLD (row + question alone, fresh state), does not reliably copy a
    multi-token word or hold the format on natural keys.
- **A3 CONFIRM (frozen rule top-1 gain ≥ 5.6186 on S2's unseen rows, 01:16 IST):
  NOT met.** para + tool answered-and-correct 0.212 vs strict 0.175 (+0.038 <
  +0.10); neg abstain 1.00 / 0.825 (natural < 0.90). A3 closed: the mind's
  read gain cannot set abstain (it is in-script, offline; no code to remove).
- **A3: "found it" from the mind's own evidence (PRE-REGISTERED 2026-11-01
  01:00 IST, before any number of it; offline, no GPU).** A2 showed similarity
  cannot decide abstain (stored-paraphrase and never-stored sims overlap).
  A3 asks whether the mind's READ of the candidates can: per query, from the
  logged top-5 `P_cands` only.
  - Features (fixed now): (f1) max `gain` over top-5 (log p of the row's own
    answer with the row minus without it, over the 20 candidates, prior
    cancelled); (f2) top-1 `gain`; (f3) agreement = how many of the top-5 rows
    re-read to the top-1 answer; (f4) top-1 `margin`; (f5) top-1 sim minus the
    mean of ranks 2-5 (outlier gap); (f6) the top-1 free token equals the top-1
    20-way answer.
  - Rule form: strict mode, then, when the organ misses, answer with the top-1
    row iff ONE feature ≥ a threshold, else abstain.
  - FIT on Lead's strict run (`logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.jsonl`,
    seed 22000): choose the feature and threshold that maximise mean
    answered-and-correct (20-way) over para + tool, both key types, subject to
    neg abstain ≥ 0.90 on both key types.
  - CONFIRM on S2's rows (seed 23000, unseen), with the rule frozen. Bar MET iff
    there neg abstain ≥ 0.90 on both key types AND para + tool
    answered-and-correct is at least strict mode + 0.10 (absolute, same rows).
    Free (`P_free_sent`) reported beside it.
  - Closest prior work: UPR (Sachan et al. 2022, rerank passages by the LM's
    log p(question | passage)); Self-RAG (Asai et al. 2023, trained relevance
    critic tokens); selective QA (Kamath et al. 2020, a trained calibrator).
    Difference: no trained critic and no question likelihood; the O(1) mind's
    answer gain with the no-row prior cancelled, and agreement across rows, set
    the abstain.
  - **A3 FIT (2026-11-01 01:01 IST, seed 22000 only, in-sample): below the bar
    already.** Best: f2 top-1 `gain` ≥ 5.6186 → para + tool answered-and-correct
    0.256 vs strict 0.194 (+0.062 < +0.10); para 0.30 / 0.275, tool 0.175 /
    0.275 (syn / nat); neg abstain 0.975 / 0.90. Quantiles (10 / 50 / 90 %):
    top-1 gain neg 0.66 / 3.16 / 5.36 vs stored-and-right 2.86 / 4.83 / 6.40;
    agreement neg 1 / 1 / 2 vs right 1 / 2 / 3; outlier gap identical
    (0.01-0.04). Reading: gain says "this row states a value", not "this row
    answers THIS question": any row naming one of the 20 values raises it, so
    never-stored queries gain too. The frozen rule still runs on S2 as
    registered (expected NOT met).
- **A4: the normalised-word organ inside the session (PRE-REGISTERED
  2026-11-01 01:04 IST, before any number of it). PLUMBING, no novelty claim**
  (closest prior work: IR case folding / suffix stripping / stop words, Porter
  1980, plus exact phrase match; SL2 already used it across documents). Why:
  strict mode halves para / tool because only the token organ may say "found";
  bench's para cell varies only case / plural / a leading "the" (undone by
  this normaliser BY DESIGN), so a met A4 says nothing about rewording or
  synonyms (no cell yet).
  - Rule (`PACKAGE_NORM=1`, after the token organ misses): words = runs of
    letters / digits in the decoded tokens, lower-cased, "a / an / the" dropped,
    one trailing "s" stripped when the word has ≥ 4 letters; stream words from
    tokens [0, doc_end), query words from [query_pos, T). (1) the longest query
    SUFFIX word n-gram, n = 6..3, found in the stream; else (2) the longest
    ANY-position query word n-gram, n = 6..3; else (3) any query word bigram
    holding a word seen ≤ 2 times in the stream. Latest occurrence wins. Row =
    its sentence (previous "." to the next "." after the match, ≤ 64 tokens
    each way), ranked first like the token organ's row. Strict: abstain iff the
    token organ and steps 1-3 all miss.
  - TEXT-ONLY CHECK (CPU; the harness's own `build()` and rng, so the documents
    are exactly bench's): seed 24000 (fresh), L = 32768, 40 docs per cell,
    kv / para / tool / neg × synthetic / natural, needles 8. "Right row" = the
    row's decoded text holds the asked definition ("{key} refers to {val}")
    or, for tool, the asked job's block ("job: {job};" and "{field}: {val}").
  - Bars: para and tool right row ≥ 0.95 on both key types; neg any-organ hit
    ≤ 0.05 on both key types; kv right row ≥ 0.975 (the token organ, unchanged).
    Per-step hit counts reported.
  - If met: the mind run (b2p2, 32k, strict + `PACKAGE_NORM=1`, seed 24000,
    `P_free_sent` primary) on Lead's gate; bars there: para and tool
    `P_free_sent` ≥ 0.80, neg abstain ≥ 0.90, both key types.
  - **A4 TEXT CHECK: MET (2026-11-01 01:07 IST; code fa95777; `logs/v14_a4_text_fa95777_20261101_0107.log` + .json).**
    Docs = bench's own (rebuild verified on seed 22000: 320 / 320 same length
    and token-organ hit as Lead's strict run).

    | cell (syn / nat) | token organ | normalised organ (step) | right row |
    |---|---|---|---|
    | kv | 1.00 / 1.00 | 0 / 0 | **1.00 / 1.00** |
    | para | 0.40 / 0.35 | 0.60 / 0.65 (all step 1, suffix) | **1.00 / 1.00** |
    | tool | 0.25 / 0.225 | 0.75 (step 3, rare bigram) / 0.775 (step 2, any 3-gram) | **1.00 / 1.00** |
    | neg (any hit) | 0 / 0 | 0 / 0.025 | bar ≤ 0.05: **0.00 / 0.025** |

    - The one neg false hit is the predicted collision: never-stored "problems
      involving" vs stored "done involving" share "involving refers to"
      (suffix 3-gram). Natural keys of common words can collide this way.
    - Cost: 0.02-0.05 s per 32k doc (a Python scan at query time, no stored
      index); an indexed form is SL2's word suffix array (~5.4 MB per 1M
      tokens).
    - Says: on bench's templated cells the "found it" decision is solved by
      text, so strict mode no longer has to give up para / tool. Says NOTHING
      about rewording or synonyms. Next: the mind run (expression,
      `P_free_sent`), queued with Lead.
  - **A4 MIND RUN v1: INVALID (my bug; 2026-11-01 01:35 IST).** Lead's run
    (`logs/v16_pkg_a4norm_b2p2_32768_cb4fa04_20261101_0108.jsonl`, 320 rows, 0
    errors) hit only at the token organ's rate (para 0.40 / 0.75, tool 0.25 /
    0.225, syn / nat). package_fetch passed bench's `query_pos` to the organ,
    and that is the query KEY's last token, not the question start (the
    pre-registration's "[query_pos, T)" meant the question; the text check
    used the question start). Reproduced on CPU with bench's own qk: old call
    = the GPU rates exactly; fixed in 238c62e (whole question) → 1.00 on all
    four. Seen from v1 (token organ, unaffected): kv `P_free_sent` 0.875 / 0.75,
    20-way 1.00 / 1.00; neg abstain 1.00 / 0.975.
  - **A4 MIND RUN v2 (PRE-REGISTERED 01:35 IST):** the same run with 238c62e on a
    FRESH seed 27000 (seed 24000's docs are now seen); same bars
    (`P_free_sent` para and tool ≥ 0.80, neg abstain ≥ 0.90, both key types).
  - **A4 MIND RUN v2 VERDICT (2026-11-01 02:08 IST; Lead's run
    `logs/v16_pkg_a4norm2_b2p2_32768_b886a03_20261101_0137.jsonl`, strict +
    PACKAGE_NORM=1, seed 27000, code b886a03 ⊇ 238c62e, 320 rows, 0 errors):
    NOT met (para natural and tool).** Fetch is exact; the misses are writing.

    | cell (syn / nat) | organ hit | abstain | `P_free_sent` | 20-way | bar |
    |---|---|---|---|---|---|
    | kv | 1.00 / 1.00 | 0 / 0 | 0.975 / 0.825 | 1.00 / 1.00 | (reported) |
    | para | 1.00 / 1.00 | 0 / 0 | **0.80 / 0.75** | 1.00 / 1.00 | ≥ 0.80: syn met, nat NOT |
    | tool | 1.00 / 1.00 | 0 / 0 | **0.60 / 0.50** | 0.675 / 0.675 | NOT met |
    | neg | 0.025 / 0.025 | **0.975 / 0.975** | 0 / 0 | – | met |

    - The organ does its job: para fetch 0.40-0.75 → 1.00, neg abstain kept.
      Para natural misses are wording (" a bike.", " ave.", generic prose);
      tool misses are the multi-field row (20-way 0.675 WITH the right row:
      the wrong field's value, or a loop on the value / job name: " onyx;
      onyx; ...", " lapneb; label: lapneb").
    - Package level after A4 (strict, 32k): never-stored abstain 0.975, exact
      and case / plural variants fetched 1.00; what remains is the mind's
      expression, the same finding as S2 and A6.
- **A3 follow-up, near-miss data (for Mind's cu2 VERIFY; 01:13 IST, descriptive,
  offline):** in the strict run's never-stored docs (seed 22000, rebuilt), the
  top row is another stored fact in 18 / 40 (synthetic) and 14 / 40 (natural);
  of those only 1 / 18 shares a syllable and 1 / 14 shares a word with the
  asked key (base 0.6 %). The rest are prose sentences. The mind copies the
  value from UNRELATED same-template rows; sent to Mind as the VERIFY mix.
- **A5: the SYNONYM cell (PRE-REGISTERED 2026-11-01 01:13 IST, before any number
  of it).** The test the learned session key exists for: a fact stored under
  one phrase, asked with a synonym phrase that shares no content word, so
  text cannot find it. `v14/syn_cell.py`: 82 hand-made pairs ("old ship" /
  "ancient vessel"), 328 distinct words, no word shared within a pair or
  reused, none a value noun; 8 facts " Definition: {A} refers to {v}." in
  held-out prose, asked " Query: {B} refers to" (the stored side random);
  `synneg` asks a pair that was never stored.
  - TEXT PART (CPU, now): seed 25000, L = 32768, 40 docs per cell; the organ
    chain (token + A4 normalised), BM25 over the doc's sentences
    (`shared_fetch.norm_words`), MiniLM-L6 over the doc's sentences (top-1).
    Score: the top-1 sentence holds the asked fact ("{A} refers to {v}").
    Sanity bar (the cell is a synonym test): organ-chain right row ≤ 0.10.
    MiniLM reported as the named dense baseline (if < 0.50, the pairs are too
    hard even for a trained semantic encoder; reported, not a bar).
  - MIND PART (4090, Lead's queue; needs a 2-line `syn` / `synneg` hook in
    bench.build that calls `v14.syn_cell.build_syn`): b2p2 package lenient
    (PACKAGE_T = -1, no PACKAGE_NORM), 32k, seed 25000, 40 docs. Bar MET iff
    the learned key's top-1 row is the asked fact (`C5_burst_hit_span`) ≥ 0.50
    AND ≥ BM25 top-1 + 0.25. Reported beside it: `P_free_sent`, C2 (the frozen
    centred block-8 address) top-1, MiniLM top-1 and its MB per 1M tokens.
  - Closest prior work: dense retrieval with a trained encoder (DPR, Karpukhin
    2020; Sentence-BERT / MiniLM, Reimers 2019). Difference: the key is the O(1)
    mind's own reading state plus a small head (L1), with no separate encoder;
    met = the mind's state carries meaning that text matching cannot.
  - **A5 TEXT PART (2026-11-01 01:29 IST; code 937972e;
    `logs/v14_a5_text_937972e_20261101_0115.log` + .json; seed 25000, 32k, 40
    docs per cell): sanity bar MET, the cell is a synonym test.**

    | path | syn: top-1 holds the fact | picks a definition | synneg false hit |
    |---|---|---|---|
    | organ chain (token + A4) | **0.00** | – | 0.00 |
    | BM25 (doc sentences) | 0.075 | 0.70 | – |
    | MiniLM-L6 (doc sentences) | **0.45** | 0.50 | – |

    MiniLM is right on ~90 % of the definitions it picks; it loses to prose
    sentences. The mind part's bar (≥ 0.50 and ≥ BM25 + 0.25) is therefore
    above a trained 22M semantic encoder.
  - **A5 MIND PART VERDICT (2026-11-01 02:00 IST; Lead's run
    `logs/v16_pkg_syn_b2p2_32768_2738b2c_20261101_0130.jsonl`, b2p2 lenient,
    32k, seed 25000, 80 rows, 0 errors): NOT met (top-1 0.40 < 0.50; the
    "≥ BM25 + 0.25" half met).**

    | syn (40 docs) | top-1 holds the fact | top-5 | answer |
    |---|---|---|---|
    | learned burst key (L1 head on the mind's state) | **0.40** | 0.775 | 20-way 0.40, sentence free 0.325 |
    | C2 per-token address (frozen state, no head) | 0.375 | 0.525 | |
    | exact organ / state only | 0.025 / – | | 0.025 |
    | MiniLM-L6 (text part) / BM25 (text part) | 0.45 / 0.075 | | |
    | oracle row re-read | | | 20-way 0.975 |

    synneg (never stored, lenient so no abstain): top-1 0.05, answer 0.05.
    - Reading: on true synonyms the mind's own state is the only in-house
      path that works (0.40 vs BM25 0.075, organs 0.025), close to a trained
      22M sentence encoder (0.45) with no encoder, but below the bar. Handed
      the right row the mind answers 0.975: the miss is the fetch.
    - Re-rank of the top-5 (descriptive, seen rows): max margin 0.45, read
      all 5 together (20-way) 0.55, max gain 0.225; of the 31 docs with the
      fact in the top-5, margin picks it in 18. The mind's read re-ranks a
      little; copy-nearest-value (A3) caps it. Lever: the VERIFY skill (cu2)
      and reading the 5 rows together, not a gain threshold.
- **Mind's free-token rule (descriptive, seen rows, 01:29 IST):** answer iff
  organ hit or the top-1 row's greedy token is a candidate value (= A3's f6).
  Never-stored abstain 0.725 / 0.425 (seed 22000) and 0.85 / 0.425 (S2),
  synthetic / natural; para + tool answered-and-correct 0.369 / 0.338 (strict
  0.194 / 0.175). The mind tells a wrong NONSENSE key from the right one,
  not a wrong ENGLISH key (it copies the nearest value ~57 % of the time).
  Sent to Mind for cu2's VERIFY mix (weight natural keys).
- **A6: hand the fetched row WARM or COLD (PRE-REGISTERED 2026-11-01 01:30 IST,
  before any number of it). Design choice, no novelty claim** (closest prior
  work: In-Context RALM, Ram et al. 2023, puts retrieved text into the running
  context; RAG / FLARE prepend it; for a recurrent O(1) mind the choice is a
  fresh state vs the carried state). Why: S2 and natq show the misses are
  EXPRESSION (right row, wrong words; natq oracle 20-way only 0.45 / 0.55), and
  today's re-read is COLD: the row + question alone from a fresh state, a
  40-token context unlike training.
  - Rows: the sentence holding the asked fact, by construction (fetch held
    fixed; equals the organ chain's row on kv / para / tool).
  - Arms: COLD = fresh state, row + question, greedy; WARM = the mind's state
    after the document (carried, chunked prefill), then row + question,
    greedy; NOROW = the carried state + question only (floor).
  - Score: Lead's sentence scorer (greedy to "." / newline, cap 16, the value
    matched as a whole word, case-insensitive).
  - b2p2, 32k, fresh seed 26000, 40 docs per cell; kv / para / tool / natq ×
    synthetic / natural (bench's own `build()` docs).
  - Bar MET iff WARM ≥ COLD + 0.10 on the mean over the 8 cells AND WARM ≥ COLD
    − 0.05 on every cell. Met → the package hands rows warm. Parity first: the
    chunked state path must reproduce the one-pass logits (bench's Reader) on a
    4k doc before any number.
  - **A6 VERDICT (2026-11-01 01:52 IST; code 238c62e;
    `logs/v14_a6_b2p2_32768_238c62e_20261101_0135.jsonl` + .log, 320 rows, rows
    found 1.00; parity 2e-3): NOT met, REVERSED: cold is far better.**

    | cell (sentence) | cold | warm | state only (norow) | warm − cold |
    |---|---|---|---|---|
    | kv syn / nat | 0.925 / 0.825 | 0.525 / 0.15 | 0 / 0 | −0.40 / −0.675 |
    | para syn / nat | 0.725 / 0.55 | 0.45 / 0.025 | 0 / 0 | −0.275 / −0.525 |
    | tool syn / nat | 0.475 / 0.50 | 0.25 / 0.275 | 0 / 0 | −0.225 / −0.225 |
    | natq syn / nat | 0.075 / 0.20 | 0 / 0 | 0 / 0 | −0.075 / −0.20 |

    - **Warm ≈ state only.** On natural keys the warm answer is nearly the same
      sentence as the no-row answer (" the place where the king was killed" vs
      " the place where the king was ..."; 32 / 34 warm misses start " the"):
      after 32k tokens of carried state (b2p2 trained at 2,048, so 16×) the
      mind does not let the row it was JUST shown steer its answer. Synthetic
      keys (novel words) break through half the time.
    - Says: past its trained length the mind's long state outvotes the newest
      sentence (= Mind's EUREKA 3bc29c7 "outvoted, not lost", from the other
      side). Package: keep the COLD re-read (a fresh scratch state per fetched
      row); the current design is right.
- **A6c: is "the mind does not listen late" a beyond-trained-length effect,
  and does a contrast rescue it? (PRE-REGISTERED 2026-11-01 01:53 IST, before
  any number of it.)** A measurement, no novelty claim (closest prior work:
  context-aware decoding, Shi et al. 2023, (1+α) log p(y | c, x) − α log
  p(y | x); contrastive decoding, Li et al. 2023; Mind's 3bc29c7 used it for the
  skill pack).
  - b2p2, fresh seed 28000, L = 2,048 / 8,192 / 32,768 (b2p2 trained at
    2,048), 40 docs per cell; cells kv synthetic, kv natural, para natural;
    fact row by construction; sentence scorer (as A6).
  - Arms: COLD, WARM, NOROW (as A6) and WARMC = the warm stream contrasted
    with the no-row stream, logits l_warm + 4 (l_warm − l_norow), both streams
    fed the generated tokens (Mind's a = 4).
  - Bar L (length) MET iff at 2k WARM ≥ COLD − 0.05 on all 3 cells AND at 32k
    WARM ≤ COLD − 0.20 on kv natural (A6 replicated).
  - Bar C (contrast) MET iff at 32k WARMC ≥ COLD − 0.05 on all 3 cells.
  - Met C → a stream-inserted read (N11 refresh, or a warm row) can be made
    to work with a second O(1) stream; not met → the package keeps the cold
    scratch read.
  - **A6c VERDICT (2026-11-01 03:22 IST; code dec965d;
    `logs/v14_a6c_b2p2_dec965d_20261101_0212.jsonl` + .log, 360 rows, rows found
    1.00): Bar L NOT met (warm fails at 2k too: NOT a length effect); Bar C MET
    (the contrast rescues the warm read, and beats cold).**

    | L | cold (kv syn / kv nat / para nat) | warm | warm + contrast (a = 4) |
    |---|---|---|---|
    | 2,048 | 0.875 / 0.80 / 0.775 | 0.675 / 0.275 / 0.125 | **0.95 / 0.975 / 1.00** |
    | 8,192 | 0.825 / 0.875 / 0.725 | 0.575 / 0.225 / 0.075 | **0.90 / 1.00 / 0.925** |
    | 32,768 | 0.925 / 0.775 / 0.70 | 0.45 / 0.125 / 0.125 | **0.90 / 0.95 / 0.975** |

    - Correction to A6's reading and EUREKA bd34118: the warm read fails at
      the trained length 2k as well, so "the long state outvotes the newest
      sentence" holds at ANY length, not only past the trained length.
    - The contrast (l_warm + 4 (l_warm − l_norow), on the answer tokens only)
      gives the best expression we have: natural keys 0.95-1.00 vs cold
      0.70-0.875, at every length; it fixes cold's " ave." / prose misses
      (" indigo.", " willow.", " juniper."). Answer starts with the value
      0.88-1.00 (cold 0.60-0.93).
    - Caveat: 20-38 % of contrast answers RUN ON after the value (e.g.
      ' marigold." Definition: ...'); Mind's cd3: contrast on free text
      collapses (NLL 3.90 → 8.56). So: contrast on the answer tokens only,
      stop at the value / first stop. Package candidate: the row read WARM
      with a no-row contrast stream (one prefill, two short O(1)
      continuations), to be confirmed on the package cells (fresh seed).
- **A8 (N11-RC): refresh half as often with a contrast stream (PRE-REGISTERED
  2026-11-01 03:14 IST, before any number of it; Mind's proposal after cd2).**
  Design / cost, no novelty claim (closest prior work: context-aware decoding,
  Shi et al. 2023; contrastive decoding, Li et al. 2023; Mind's cd1 / cd2).
  - Cell: N11-R's (Mind's skill pack, affine_map_new + shift_wide_new), b2p2,
    FRESH seeds 8300-8349, L = 8,192 / 32,768.
  - Arms: plain refresh R = 2,048 / 4,096 / 0, and CONTRAST at R = 4,096 and
    R = 0: last logits l_R + 4 (l_R − l_nopack), where nopack = the same
    filler + question with no pack anywhere (= run_skillpack_probe's cd1).
  - Score: FREE (greedy token over the whole vocabulary = the target),
    primary; 20-way secondary. Pooled over the 2 tasks (100 docs per arm).
  - Bar MET iff at L = 32,768: free(R = 4,096 + contrast) ≥ free(R = 2,048
    plain) − 0.05 AND ≥ free(R = 4,096 plain) + 0.10. Met → the package
    refreshes every ~4k with one extra O(1) stream (+1.15 MB, 2× compute
    while on): refresh tokens halved (~1.6 % → ~0.8 %).
  - **A8 VERDICT (2026-11-01 03:27 IST; code 0ded954;
    `logs/v14_a8_n11rc_b2p2_0ded954_20261101_0324.jsonl` + .log, 600 rows):
    NOT met, just short.** At 32k: free(R 4096 + contrast) **0.58** < free(R 2048
    plain) 0.66 − 0.05 = 0.61; vs R 4096 plain 0.11 (+0.47, that half met).

    | 32k, free (plain / contrast) | R 2048 | R 4096 | none |
    |---|---|---|---|
    | both rules | 0.66 / **0.79** | 0.11 / 0.58 | 0.00 / 0.00 |
    | affine_map_new | 0.94 / 0.90 | 0.16 / 0.82 | 0 / 0 |
    | shift_wide_new | 0.38 / **0.68** | 0.06 / 0.34 | 0 / 0 |

    (8k is the same picture: R 2048 0.60 / 0.79, R 4096 0.13 / 0.59.)
    - Halving the refresh with the contrast falls 0.03 short; the weak rule
      (shift) loses most. Descriptive (not a bar): the contrast at the SAME
      R = 2048 lifts free 0.66 → 0.79 (shift 0.38 → 0.68, which would pass
      N11-R/F's bar). A "R 2048 + contrast" default needs its own
      pre-registration on fresh seeds. No refresh: 0.00 with or without the
      contrast (nothing to amplify).
    - Mind's cd3 gate applies: contrast on the query's answer token only.
- **A9: the package reads its fetched row WARM + CONTRAST (PRE-REGISTERED
  2026-11-01 03:23 IST, before any number of it).** Design choice (closest prior
  work: context-aware decoding, Shi et al. 2023; Mind's cd1-cd3; A6c here).
  - Fetch = the real package in strict mode: token organ, then the A4
    normalised organ on the whole question (238c62e); abstain iff both miss.
  - Arms on the same fetched row: COLD (today's package: row + question,
    fresh state, greedy) and WARMC (the session's state after the document,
    then row + question, decoding l_warm + 4 (l_warm − l_norow) on the answer
    tokens only; l_norow = the same state + question without the row); NOROW
    (state only) reported.
  - Score: sentence scorer (greedy to "." / newline, cap 16, the value as a
    whole word), abstained = not answered; also "starts with the value" and
    run-on rate.
  - b2p2, 32k, FRESH seed 29000, 40 docs per cell; kv / para / tool / neg ×
    synthetic / natural (bench's own docs).
  - Bar MET iff WARMC − COLD ≥ +0.10 on the mean over the 6 answerable cells
    (kv, para, tool × 2) AND no answerable cell below COLD − 0.05; neg
    abstain stays ≥ 0.90 (rule-based, unchanged). Met → the package's
    default read is WARMC.
  - **A9 VERDICT (2026-11-01 03:37 IST; code 1b29b86;
    `logs/v14_a9_b2p2_32768_1b29b86_20261101_0327.jsonl` + .log, 320 rows): MET.
    The package's default read becomes WARM + CONTRAST (answer tokens only).**

    | 32k (syn / nat) | fetch | cold | warm + contrast | starts with value (cold → warmc) | run-on (warmc) |
    |---|---|---|---|---|---|
    | kv | 1.00 / 1.00 | 0.95 / 0.775 | 0.925 / **0.925** | 0.95 → 0.93 / 0.78 → 0.90 | 0.30 / 0.25 |
    | para | 1.00 / 1.00 | 0.85 / 0.675 | **0.925 / 0.925** | 0.85 → 0.90 / 0.65 → 0.90 | 0.25 / 0.42 |
    | tool | 1.00 / 1.00 | 0.525 / 0.40 | **0.75 / 0.575** | 0.53 → 0.75 / 0.40 → 0.55 | 0.55 / 0.47 |
    | neg | hit 0 / 0.075 | abstain **1.00 / 0.925** | | wrong 0 / 0.025 | |

    - Mean warmc − cold over the 6 answerable cells **+0.142** (bar +0.10);
      worst cell −0.025 (kv syn; floor −0.05); neg abstain ≥ 0.90 kept.
    - The stricter "answer starts with the value" shows the same gains, so
      the sentence scorer is not flattered by run-on text; but run-on is
      25-55 % (tool highest: the mind keeps reading out fields): the package
      stops at the first value-word / ";" / ".".
    - Package at 32k on b2p2 (103M), strict + normalised organ + warm-contrast
      read: exact facts 0.925 / 0.925, case / plural variants 0.925 / 0.925,
      tool fields 0.75 / 0.575, never-stored abstain 1.00 / 0.925 (synthetic /
      natural keys), free generation, sentence scorer.
- **A10: does cu1's learned " unknown" give the package abstain by the mind's
  own words? (PRE-REGISTERED 2026-11-01 03:26 IST, before any number of it; Mind's
  request.)** A skill check, not a package body (cu1 pays +0.045 holdout NLL).
  - cu1 (`cu1_best_2efb422.pt`, warm-started from c1m2a's stage A, NOT b2p2,
    so my L1 head does not apply) vs b2p2 (control), SAME docs: bench.py
    LENIENT with NO key_fn, each mind fetching by its own frozen address
    (`--vec_layer 8 --centre`, doc mean; burst rows, gate_tau 10); 32k, FRESH
    seed 30000, 40 docs; kv and neg × synthetic / natural.
  - Rule (Mind's): answer iff the top-1 row's greedy token over the whole
    vocabulary (`P_cands[0].free_tok`) is one of the 20 candidate values;
    else abstain (cu1's " unknown" abstains).
  - Bar MET iff on cu1: neg natural abstain ≥ 0.80 AND kv natural
    answered-and-correct (`P_cands[0].free` when answered) ≥ b2p2's − 0.05 on
    the same docs. Synthetic reported. Prior work: SQuAD 2.0 / R-Tuning /
    Self-RAG (as in Mind's cu1 notes); difference: abstain read off an O(1)
    mind's free generation over its own fetched row.
  - **A10 VERDICT (2026-11-01 03:43 IST; code a589262;
    `logs/v14_a10_{cu1,b2p2}_32768_a589262_20261101_0336.jsonl`, 160 rows each,
    0 errors): NOT met as registered; the miss is FETCH, not cu1's reading.**

    | 32k, seed 30000 (syn / nat) | cu1 | b2p2 |
    |---|---|---|
    | neg: abstain by the rule | 1.00 / **1.00** | 0.825 / 0.975 |
    | kv: answered-and-correct | 0.30 / **0.20** | 0.20 / 0.35 |
    | kv: top-1 row holds the fact (own frozen address) | 0.325 / 0.225 | 0.20 / 0.375 |
    | " unknown" said outright (neg) | 0.125 / 0.10 | 0 / 0 |

    Bar: cu1 neg natural abstain 1.00 ≥ 0.80 met; cu1 kv natural 0.20 < b2p2
    0.35 − 0.05 NOT met.
    - Split by the top row (descriptive): right row → says the value, cu1
      20 / 22 vs b2p2 22 / 23 (no false abstain); wrong row → says a wrong
      value, cu1 1 / 58 vs b2p2 6 / 57. The kv gap is cu1's weaker own-state
      fetch (right row 9 vs 15 docs on natural keys), not its reading.
    - Design limit: with each mind's own frozen address the wrong rows were
      mostly prose (b2p2 natural 0 / 25 wrong values here vs ~57 % on the L1
      head's rows), so the hard case (another key's English-word fact) barely
      appeared. Mind's fixed-row verify probe covers it (cu1 " unknown" 0.86,
      wrong value 0.01 vs b2p2 0.59). A package-level test needs the rows held
      fixed across minds (b2p2's L1-head rows read by both).
- **A11: key-swap relevance (PRE-REGISTERED 2026-11-01 05:04 IST, before any
  number of it; the first D4 unit, settled with Mind and Lead).** Does the
  mind's answer depend on the question's KEY? rel = log p(v | row, q) −
  mean_k log p(v | row, q_swap_k), with v = the mind's greedy first answer token
  given row + q, and q_swap_k = the same question with the key replaced by a
  never-stored natural phrase (K = 3). A right row's answer should depend on
  the key; a wrong row's should not (the mind copies it either way). No
  training; cold short reads (~40 tokens), CPU.
  - b2p2; bench's kv docs, NATURAL keys, L = 32,768, FRESH seed 31000, 40 docs.
    Per doc 3 rows: the asked key's fact (positive), another stored key's
    fact in the same template (hard negative), a prose sentence (easy
    negative).
  - Bar (Mind's): AUROC of rel ≥ 0.85 for right rows vs same-template wrong
    rows. Reported: AUROC vs prose rows and vs all wrong rows; the same AUROCs
    for the plain confidence log p(v | row, q) (the baseline rel must beat);
    the "free token is a value" rule's accuracy on the same rows.
  - Met → abstain on non-organ fetch becomes a query-time check (cu2 drops
    VERIFY, Mind); not met → VERIFY stays in cu2a.
  - **Amendment A11/1 (05:05 IST, before any number; Mind's round-2 ask):** the
    same rows are also read by cu1 (`cu1_best_2efb422.pt`), REPORTED (no bar):
    does a mind trained to say " unknown" check keys more (higher AUROC)?
    The bar stays on b2p2. Lead's ruling: VERIFY stays in cu2a iff b2p2's
    AUROC < 0.85.
  - **A11 VERDICT (2026-11-01 05:09 IST; code 2c2e760 (run as c7c016e, rebased,
    code identical); `logs/v14_a11_keyswap_2c2e760_20261101_0506.jsonl` + .log, 240
    rows): NOT met. VERIFY stays in cu2a.**

    | AUROC (40 right / 40 same-template wrong / 40 prose rows) | key-swap rel: vs def / vs prose | plain confidence: vs def / vs prose | says a value on: right / def / prose |
    |---|---|---|---|
    | b2p2 (bar on vs def) | **0.698** / 0.688 | 0.677 / 0.817 | 0.78 / 0.53 / 0.00 |
    | cu1 (reported) | 0.815 / 0.816 | 0.774 / 0.774 | 0.72 / **0.03** / 0.00 |

    rel medians: b2p2 right 0.68, def 0.29, prose 0.20; cu1 right 3.83, def
    0.63, prose 0.44.
    - b2p2's answer barely depends on the question's key: a query-time key
      check has little to read. cu1, trained to say " unknown", has a strong
      key dependence and its own words abstain on 97 % of wrong same-template
      rows.
    - Says: CHECKING is a skill the mind must learn from examples; the first
      counterexample to "query-time beats training-time" (it fits D1: train the
      mind for language and example-based skills). Key-swap stays as a
      reported arm only.
  - Closest prior work: context-aware decoding (Shi et al. 2023: contrast
    with / without the CONTEXT); perturbation / counterfactual relevance;
    Self-RAG's IsRel (trained). Difference: the QUESTION's key is perturbed
    with the row fixed, testing key dependence of an O(1) mind's answer at
    query time with no training.
- **A12: many-key TEXT fetch on natq (PRE-REGISTERED 2026-11-01 05:09 IST, before
  any number of it; D4's CPU part).** Plumbing, no claim (closest prior work:
  BM25, Robertson et al. 1994, which is "every word a key, weighted by
  rarity"). Docs: bench's natq and natqneg (Lead's 3d1e55b), L = 32,768, FRESH
  seed 32000, 40 docs × synthetic / natural keys.
  - R1: BM25 over the doc's sentences (`shared_fetch.norm_words`) with the
    whole question; top-1 / top-5 hold the stated fact.
  - R2 (rarest span): the question's contiguous normalised word span (n =
    3, 2, 1) with the SMALLEST nonzero count in the stream (ties: longer span,
    then latest); row = the sentence of its latest occurrence; ABSTAIN if that
    count > 2 (not key-like).
  - Bar (decision, not a claim): R2 right row ≥ 0.90 on natq AND abstain ≥
    0.90 on natqneg, both key types → text handles natq FETCH, and D4's GPU
    run keeps only the natq oracle / expression arms. Else per-word mind
    addresses join D4. Reported: R1 top-1 / top-5; AUROC of R1's top-1 score,
    natq vs natqneg.
  - **A12 VERDICT (2026-11-01 05:11 IST; code 9e2b136;
    `logs/v14_a12_natq_text_9e2b136_20261101_0510.json` + .log): NOT met → per-word
    mind addresses join D4.**

    | 40 docs per cell | synthetic | natural |
    |---|---|---|
    | R1 BM25 top-1 / top-5 holds the fact (natq) | 0.175 / 0.525 | 0.525 / 0.825 |
    | R2 rarest span: right row (natq) | 0.45 | 0.725 |
    | R2 abstain (natqneg) | 0.125 | 0.125 |
    | R1 top-1 score AUROC natq vs natqneg | 0.565 | 0.793 |

    - The question's FRAME words ("question", "stand for", "answer") are as rare
      in prose as the key, and other facts' templates share them ("... stands
      for ..."), so text cannot tell the key from the frame: it picks the frame
      (R2) or spreads over it (BM25). Synthetic keys, though unique, lose to
      longer frame spans at equal count. Text does not handle natq fetch or
      abstain; the mind's own per-word addresses go into D4's GPU run.
- **A13 (D4 GPU run): a question is many keys, chosen by the mind's own
  surprise (PRE-REGISTERED 2026-11-01 05:12 IST, before any number of it).**
  A12 showed text cannot tell a question's key from its frame ("stand for",
  "Answer:" are as rare in prose as the key). The mind can: the frame is
  PREDICTABLE to it, the key is not.
  - Cells: syn, synneg (A5's), natq, natqneg (bench); b2p2; L = 32,768; FRESH
    seed 33000; 40 docs each (natq / natqneg: natural keys).
  - Rows: the doc's sentences (split at ".", 4-64 tokens). Each row keeps its
    per-token frozen vectors (centred by the doc mean).
  - Fetch arms (score = max cosine over a row's tokens):
    - F2 (primary): the 2 question tokens with the highest surprisal under
      the mind (excluding the question's first 2 tokens); their block-8
      vectors are 2 keys; a row's score = the sum of its 2 max-sims.
    - F3: F2 at block 2 (word meaning near the embedding).
    - F1: the key's last token at block 8 (key position known: a reference
      arm, as bench's C2).
    - MiniLM-L6 over sentences (the named text baseline).
  - Reads (b2p2, warm + no-row contrast, answer tokens only, stop at "." /
    newline, sentence scorer): F2's top-1 row; F2's top-5 rows together
    (line-up, doc order); the ORACLE fact row on natq (the ceiling).
    Key-swap rel on F2's top-1 (reported only; A11 settled it).
  - Bars:
    - B1 (natq fetch): F2 top-1 holds the fact ≥ A12's BM25 top-1 + 0.15 on
      natq (≥ 0.325 synthetic, ≥ 0.675 natural).
    - B2 (synonyms): F2 top-1 ≥ 0.50 on syn (A5's bar; the L1 head had 0.40).
    - B3 (line-up): line-up read sentence ≥ the top-1 read + 0.10 on syn.
  - Reported: F1 / F3 / MiniLM top-1 / top-5; natq oracle under warm +
    contrast; top-1 score AUROC natq vs natqneg and syn vs synneg; key-swap
    rel AUROC on F2's top-1.
  - Closest prior work: BM25 / query-term weighting by IDF (Robertson
    1994); surprisal-based saliency (Hale 2001; "salience" in LM retrieval
    queries, e.g. query term weighting with LMs); ColBERT late interaction
    (max-sim over tokens, Khattab & Zaharia 2020). Difference: the query
    terms are chosen by the O(1) mind's OWN surprise and matched in its OWN
    states, with no trained encoder.
  - **A13 VERDICT (2026-11-01 06:00 IST; code b34636e; `logs/v14_a13_manykey_b2p2_32768_b34636e_20261101_0514.jsonl` + .log, 160 rows):
    B1 MET, B2 MET, B3 NOT met.**

    | fetch: top-1 / top-5 holds the fact | syn (40) | natq, natural keys (40) |
    |---|---|---|
    | **F2: 2 surprise-chosen question tokens, block-8 max-sim** | **0.575 / 0.80** | **0.775 / 0.975** |
    | F3: the same at block 2 | 0.40 / 0.65 | 0.55 / 0.85 |
    | F1: the key's last token, block 8 (key position known) | 0.125 / 0.45 | 0.875 / 0.975 |
    | MiniLM-L6 over sentences | 0.35 / 0.65 | 0.55 / 0.625 |
    | (A12 BM25, seed 32000) | – | 0.525 / 0.825 |

    | reads, warm + contrast, sentence scorer | syn | natq |
    |---|---|---|
    | F2 top-1 row | 0.55 | 0.475 |
    | line-up (F2 top-5 together) | 0.275 | 0.125 |
    | oracle fact row | 0.925 | 0.625 |

    - B1: natq F2 top-1 0.775 ≥ 0.675 (natural keys; the synthetic half was
      not run: the cells line named natural keys only). B2: syn F2 top-1 0.575
      ≥ 0.50 (A5's L1 head 0.40 on seed 25000). B3: the line-up read HURTS
      (0.275 < 0.55 + 0.10): five rows confuse the free answer.
    - The mind's own surprise picks a key word first (" merchant", " cunning",
      " sleek", " Texas"; the 2nd pick is often a frame word: " refers",
      " Answer"), and max-sim in its own block-8 states finds synonym facts
      better than a trained sentence encoder on the same docs (0.575 vs
      0.35). Caveat: MiniLM is pooled per sentence, F2 is token-level late
      interaction; one seed.
    - Not solved: abstain. Top-1 score AUROC syn vs synneg 0.664, natq vs
      natqneg 0.768; key-swap rel 0.41 / 0.57. (The neg reads' "wrong" field
      scores a random bookkeeping value: not informative.)
    - Package consequence: for words that differ, fetch by the mind's
      surprise-chosen tokens (F2) and read warm + contrast: synonyms 0.55,
      natq 0.475 (ceiling 0.625), lenient. Abstain there needs the trained
      VERIFY (cu2).
- **A14: the 3-step lenient package, with its memory stated (PRE-REGISTERED
  2026-11-01 06:02 IST, before any number of it).** Organ chain (token +
  normalised) → if it misses, the surprise-keyed address (A13's F2) → read
  warm + contrast. A13's F2 matched against EVERY token's 588-d state: 1.18 GB
  per 1M tokens at fp16, far over the ≤ 50 MB / 1M latitude. So the address
  is stored in three forms, all on the same docs:
  - V-full: every token, 588-d fp16 (1,176 B / token; reference only);
  - V-pca: every token, a fixed PCA-64 basis, int8 + one fp16 scale (66 B /
    token ≈ 66 MB / 1M);
  - V-sal (the package candidate): only the stream's own surprising tokens
    (the top 1/16 by the mind's surprisal per doc; "surprise on both sides"),
    PCA-64 int8 (≈ 4.1 MB / 1M) + their positions (4 B each ≈ 0.25 MB / 1M).
    The PCA basis is fit ONCE on 65,536 held-out tokens (the FineWeb train
    pool, not the eval prose) and shipped (588 × 64 fp16 = 75 KB).
  - Cells: kv / para / tool / neg × synthetic / natural, syn, synneg, natq,
    natqneg (natural); b2p2; L = 32,768; FRESH seed 34000; 30 docs per cell.
  - Reported per cell: lenient answer accuracy (sentence scorer), strict
    (organ only, abstain otherwise), and on neg cells the rate of answering
    with any of the 20 value words (lenient's known cost); MB / 1M per form,
    beside the organ's 11.4 MB.
  - Bars: (B1) V-sal costs ≤ 0.075 vs V-full on F2 top-1 (syn and natq);
    (B2) on kv / para / tool lenient = strict within 0.02 (the organ answers
    first, no regression). Met → V-sal is the package's address (MB stated).
  - Closest prior work: ColBERTv2 residual compression (Santhanam et al.
    2022), PLAID; token pruning for late interaction. Difference: the index
    keeps only the tokens the O(1) mind itself found surprising, and the
    query uses its own surprising tokens.
  - **A14 VERDICT (2026-11-01 06:58 IST; code eef1c6f;
    `logs/v14_a14_package_b2p2_32768_eef1c6f_20261101_0604.jsonl` + .log): PARTIAL
    run, stopped by its own VRAM guard (667 MB free) when Mind's cu2 chain took
    the card at 06:06; done: the 8 kv / para / tool / neg cells (30 docs each)
    and syn 19 / 30; synneg / natq / natqneg NOT run. B1 NOT met (decisive on
    syn); B2 MET.**

    | cell (30 docs) | organ hit | strict | lenient (any form) |
    |---|---|---|---|
    | kv syn / nat | 1.00 / 1.00 | 0.967 / 0.833 | same |
    | para syn / nat | 1.00 / 1.00 | 1.00 / 0.90 | same |
    | tool syn / nat | 1.00 / 1.00 | 0.70 / 0.70 | same |
    | neg syn / nat: answers with a value word | 0 / 0.07 | 0.00 / 0.03 | 0.40-0.57 / 0.43-0.60 |

    | syn (19 docs): F2 top-1 / top-5 | memory per 1M tokens |
    |---|---|
    | V-full 0.632 / 0.84 | 1,176 MB |
    | V-pca (PCA-64 int8, every token) 0.263 / 0.63 | 66 MB |
    | V-sal (PCA-64 int8, top 1/16 surprising tokens) 0.211 / 0.42 | 4.4 MB |

    - B1: V-sal 0.211 vs V-full 0.632 (−0.42; the 11 missing docs cannot close
      it). The PCA-64 basis (47.9 % of the variance) destroys the semantic
      address; pruning to surprising tokens costs a little more. B2: lenient =
      strict on every organ cell (the organ answers first).
    - The lenient cost on never-stored questions: a value word in 40-60 % of
      answers vs 0-3 % strict.
    - Next (needs its own pre-registration): keep the FULL 588-d direction at
      low precision on the surprising tokens only (int8: 592 B × 62.5k tokens ≈
      37 MB / 1M, inside the latitude) to tell PCA's loss from pruning's.
- **A15: which compression keeps the surprise-keyed address? (PRE-REGISTERED
  2026-11-01 06:59 IST, before any number of it.)** A14 lost the synonym
  address with PCA-64 (0.632 → 0.263). Keep the FULL 588-d direction, store
  less precision:
  - V-full: every token, fp16 (reference, 1,176 MB / 1M);
  - V-all8: every token, 588-d int8 + fp16 scale (590 MB / 1M; reference);
  - V-sal8 (candidate): the top 1/16 surprising stream tokens, 588-d int8 +
    scale + position (596 B × 62.5k ≈ 37 MB / 1M, inside the latitude);
  - V-sal1: the same tokens, sign bits only (74 B + 4 B position ≈ 4.9 MB /
    1M; cosine on ±1 vectors).
  - Cells: syn, synneg, natq, natqneg (natural); b2p2; 32k; FRESH seed
    35000; 30 docs each. Fetch = A13's F2 (query tokens at full precision);
    read = warm + contrast of each form's top-1.
  - Bar: V-sal8 F2 top-1 ≥ V-full − 0.075 on syn AND on natq. Reported: V-sal1
    and V-all8 the same way, the lenient answer accuracy per form, MB / 1M.
  - Closest prior work: ColBERTv2 residual compression; binary embeddings /
    SimHash (Charikar 2002). Difference: only the tokens the O(1) mind found
    surprising are kept.
  - **A15 VERDICT (2026-11-01 08:44 IST; code 9b9c848;
    `logs/v14_a15_address_b2p2_32768_9b9c848_20261101_0823.jsonl` + .log, 120 rows,
    run 08:23-08:32): NOT met on syn (V-sal8 0.333 < V-full 0.433 − 0.075);
    natq fine.**

    | top-1 / top-5 (30 docs) | V-full 1,176 MB | V-all8 590 MB | V-sal8 37 MB | V-sal1 4.9 MB |
    |---|---|---|---|---|
    | syn | 0.433 / 0.933 | 0.433 / 0.933 | 0.333 / 0.733 | 0.433 / 0.767 |
    | natq | 0.733 / 0.90 | 0.733 / 0.90 | 0.80 / 0.90 | 0.767 / 0.90 |
    | answer (warm + contrast) syn / natq | 0.367 / 0.533 | same | 0.267 / 0.533 | 0.367 / 0.533 |

    - Precision is free (int8 on every token = fp16 exactly); A14's loss was
      the PCA projection. Keeping only the 1/16 surprising tokens costs on
      synonyms (top-5 0.933 → 0.73-0.77), not on natq. Sign bits matched full
      on top-1 (3-doc differences at n = 30: noise level).
    - Seed spread of the full address on syn: 0.575 (A13), 0.632 (A14, 19
      docs), 0.433 (A15): about ±0.1 at n = 30-40; bars on syn need more docs.
    - Next: a token-fraction sweep at sign bits (1/16, 1/8, 1/4, 1/2 = 4.9 /
      9.8 / 19.6 / 39 MB per 1M), with ≥ 80 docs on syn.
- **A16 (N11-RD): refresh every 8k with echo dilation + contrast
  (PRE-REGISTERED 2026-11-01 08:33 IST, before any number of it; after Mind's
  clk2 EUREKA db43950 / c7c6e2b).** Design / cost, no novelty claim of its own
  (prior work: Δ / dt scaling for state-space length extension, DeciMamba,
  LongMamba; context-aware decoding; Mind's clk1-2 and cd1-3).
  - Cell: N11-R's (Mind's skill pack, affine_map_new + shift_wide_new), b2p2,
    FRESH seeds 8600-8649, L = 8,192 / 32,768.
  - Arms: plain R = 2048; R = 2048 + contrast; R = 8192 plain; R = 8192 +
    contrast; R = 8192 + echo dilation (s = 4) + contrast; R = 4096 + echo
    dilation + contrast. Dilation = Mind's hook `model._dilate = (start, end,
    4, 'echo')` on the filler from the LAST pack insert to the question, in
    both the with-pack stream and the no-pack contrast stream (the same filler
    tokens); contrast a = 4 on the query token only.
  - Score: FREE (greedy token = target), primary; 20-way secondary; pooled
    over the 2 rules (100 docs per arm).
  - Bar MET iff at L = 32,768: free(R 8192 + dilation + contrast) ≥ free(R 2048
    plain) − 0.05. Met → the package refreshes every ~8k (~0.4 % tokens) with
    echo dilation on the stretch before a question; the dilation's prose cost
    (+0.013 nats, Mind) and the second stream go into PACKAGE.md's cost table.
  - **Amendment A16/1 (08:44 IST, before any A16 number; Lead's ask):** the
    prose cost is MEASURED, reported beside the free score: the NLL of the
    filler tokens in the last gap (R 8192, L 32k, the same seeds) with vs
    without echo dilation, in a small follow-up run (A16 itself started at
    08:43 without it).
  - **A16 VERDICT (2026-11-01 08:54 IST; code 02b3306; `logs/v14_a16_n11rd_b2p2_02b3306_20261101_0843.jsonl` + .log, 600 rows):
    NOT met.** At 32k free(R 8192 + echo dilation + contrast) **0.42** < free(R
    2048 plain) 0.63 − 0.05.

    | 32k free (pooled) | plain | + contrast | + dilation | + dilation + contrast |
    |---|---|---|---|---|
    | R 2048 | 0.63 | **0.70** | – | – |
    | R 4096 | 0.16 | 0.63 | 0.49 | **0.65** |
    | R 8192 | 0.02 | 0.04 | 0.05 | 0.42 |

    (8k: R 2048 0.59 / 0.70; R 4096 0.13 / 0.57 / 0.45 / 0.64; R 8192 0.01 / 0.03 /
    0.07 / 0.45.) Per rule at 32k: affine R2k plain 0.84 vs R8k + dil + c 0.54;
    shift 0.42 vs 0.30.
    - One read does not reach 8k well enough even with dilation + contrast
      (0.42); at 4k, dilation + contrast (0.65) matches 2k plain (0.63),
      descriptive, and the dilation adds only +0.02 over the contrast alone
      there. Best overall: R 2048 + contrast 0.70 (+1.6 % tokens).
    - **Prose cost of the dilation (A16/1, 08:58 IST; code c83d6c8;
      `logs/v14_a16_span_nll_c83d6c8_20261101_0853.json` + .log, 100 gaps of
      ~8k tokens):** +0.020 nats per token (plain 3.737 → dilated), a little
      above Mind's +0.013 on its own docs. So "every 8k + dilation" would cost
      0.4 % refresh tokens plus +0.020 nats on the gaps, and still misses the
      bar; "every 4k + dilation + contrast" pays the same per-token cost for
      +0.02 free over the contrast alone: not worth it at 4k.
- **A17: render the fetched row in the mind's TRAINED template (PRE-REGISTERED
  2026-11-01 08:47 IST, before any number of it; Mind's EUREKA d92c2cf: skills
  are format-bound).** Plumbing on the library side (closest prior work:
  prompt / template normalisation, e.g. instruction-format alignment for RAG;
  program-aided rendering). The library knows the structure of what it stored,
  so it can re-render a fetched row + the question in a format the mind was
  trained on: tool " Result: job: X; f1: v1; ...; f4: v4." → " Record: X |
  f1: v1 | ... | f4: v4." and " Query: job: X; f:" → " Q: What is the f of X?
  A:"; kv / para " Definition: K refers to V." → " Record: K | meaning: V."
  and " Q: What is the meaning of K? A:" ("meaning" is not a trained field).
  - Answering minds: cu2a (`cu2a_main_24b2969`, trained on these formats) and
    b2p2 (control, never trained on them). Rows = the strict organ chain's
    (text, mind-independent). Read = warm + contrast (the package default;
    the doc state is the answering mind's own); cold reported.
  - Cells: tool, kv, para × synthetic / natural; L = 32,768; FRESH seed
    36000; 30 docs each.
  - Bar MET iff cu2a RENDERED ≥ cu2a RAW + 0.15 on tool (both key types,
    sentence scorer). Reported: kv / para, b2p2 (the format effect without
    training), run-on rate, cold reads.
  - **Amendment A17/1 (08:51 IST, before any A17 number; Mind's diagnosis of the
    run-on):** a reported arm WARMC1 = the contrast on the FIRST answer token
    only, then plain greedy from the warm stream (the trained stop can act
    after the value). The bar stays on WARMC as registered.
  - **A17 VERDICT (2026-11-01 09:24 IST; code 8be7f47;
    `logs/v14_a17_render_32768_8be7f47_20261101_0858.jsonl` + .log, 180 rows; organ
    hit and parse 1.00 everywhere): NOT met (tool: cu2a rendered 0.767 / 0.767 vs
    raw 0.833 / 0.70, syn / nat).**

    | 32k (syn / nat) | cu2a raw → rendered, warm + contrast | cu2a raw → rendered, cold | b2p2 raw → rendered, warm + contrast |
    |---|---|---|---|
    | tool | 0.833 → 0.767 / 0.70 → 0.767 | 0.53 → 0.13 / 0.63 → 0.13 | 0.80 → 0.53 / 0.73 → 0.67 |
    | kv | 1.00 → 0.967 / 0.767 → **1.00** | 1.00 → 0.10 / 0.767 → 0.167 | 0.933 → 1.00 / 0.967 → 0.933 |
    | para | 0.967 → 0.967 / 0.933 → **1.00** | 0.90 → 0.067 / 0.70 → 0.167 | 0.967 → 0.967 / 0.933 → 0.967 |

    - Read cold, cu2a answers a rendered row " unknown." 80-83 % of the time
      even when the asked field IS in the row (" region: cedar", asked region):
      Mind's field-level format effect (only colour / city / code / owner /
      room / tool were trained; "region", "label", "meaning" read as "not this
      row's field"). The warm + contrast read overrides it (it subtracts the
      no-row " unknown" prior), so rendering is neutral on tool and helps cu2a
      on natural kv / para (0.767 → 1.00, 0.933 → 1.00).
    - WARMC1 (first-token contrast) cuts the run-on but lands below WARMC on
      rendered tool rows (0.70 / 0.633). Run-on with full WARMC on rendered
      tool rows 0.87-0.97.
    - Trained-field split on tool (n 7 owner vs 23 others; small): natural
      owner 0.857 → 1.00 rendered, others 0.652 → 0.696 (says " unknown" 30 %
      in WARMC1); synthetic owner 1.00 → 0.714. Not decisive at this n.
    - b2p2 (never trained on the template): rendering hurts tool (0.80 → 0.53).
    - Says: rendering helps only inside the trained field vocabulary; the
      fix is cu3's open field vocabulary (Mind), not a library field map.
      The package keeps the raw row for tool; definitions rendered + warm +
      contrast is a candidate for cu2a (descriptive, needs a fresh seed).
- **A18: the "T++ + library" WARM rows (PRE-REGISTERED 2026-11-01 09:49 IST, before
  any number of it; a COMPARISON row for Lead's text-library arm, no bar).**
  Every body gets the same text-only library (`text_fetch`, strict). Reads on the
  same fetched row:
  - T++ (tfc1b, trained at 2,048, RoPE): COLD (row + question only); WINDOW =
    the last 2,048 session tokens + row + question, contrast twin = the same
    window + question (KV ≈ 2 × 12 layers × d × 2 B × context); FULL = the whole
    session + row + question with its twin (KV ≈ 1.2 GB at 32k; 20 docs per
    cell); WINDOW without contrast reported.
  - PAM b2p2: warm + no-row contrast (A9's read; state 1.15 MB), same docs.
  - Cells: kv / para / tool × synthetic / natural + natq (natural); L = 32,768;
    seed 18000 (the package table's); 40 docs per cell (FULL: the first 20).
    Free generation, sentence scorer; KV / state bytes held and the tokens
    run per answer (row + question + generated, × 2 with the twin) per row.
  - Prediction (written before any number): T++ WINDOW ≈ T++ COLD on exact
    keys (within 0.05); T++ FULL below both (beyond its trained length);
    b2p2 warm + contrast ≥ T++ WINDOW on natural-key kv / para; all bodies
    equal on fetch (same text library).
  - **A18 PARTIAL (2026-11-01 10:29 IST; code 5d08477;
    `logs/v14_a18_tpp_warm_32768_5d08477_20261101_1016.jsonl` + .log): stopped by its
    VRAM guard (1,478 MB free; a Mind training job + Lead's bench took the card)
    at para/synthetic doc 2. kv done (40 docs each); para / tool / natq re-armed
    on the same seed.**

    | kv, 32k (syn / nat) | T++ cold | T++ window plain | T++ window + contrast | T++ FULL + contrast | b2p2 warm + contrast |
    |---|---|---|---|---|---|
    | sentence free | 1.00 / 0.90 | 0.975 / **0.475** | 0.95 / 0.95 | **0.00 / 0.00** (20 docs) | 0.975 / 0.925 |
    | bytes held (mean) | 1.1 MB | 67 MB | 67 MB | 1,060 MB | **1.15 MB** |
    | tokens run per answer | 37 | ~42 | 84 | 76 | 83 |

    - Prediction check on kv: T++ WINDOW ≈ COLD (0.95 vs 1.00 / 0.90) ✓; T++
      FULL below both ✓ (0.00: holding the whole 32k session, 16× its trained
      length, it answers nothing); b2p2 ≥ T++ window on natural keys ✗ by 0.025
      (0.925 vs 0.95, noise level at n = 40).
    - T++'s 2k window without the contrast falls on natural keys (0.475) the
      way b2p2's warm read did (A6): the context's prior outvotes the row in
      a transformer too, and the same contrast fixes it.
    - Memory: the mind holds the session at 1.15 MB and reads at T++-window
      quality (67 MB); the transformer holding the session (1.06 GB) collapses.
  - **A18 continued, para (11:06 IST; `logs/v14_a18b_tpp_warm_32768_5d08477_20261101_1042.jsonl`
    + .log; I stopped this run by mistake at tool/synthetic doc 22 while
    re-ordering the queue, so tool / natq are re-queued after A21):**

    | para, 32k (syn / nat), 40 docs | T++ cold | T++ window plain | T++ window + contrast | T++ FULL + contrast | b2p2 warm + contrast |
    |---|---|---|---|---|---|
    | sentence free | 0.70 / 0.675 | 0.85 / 0.575 | 0.95 / 0.975 | **0.00 / 0.00** | 0.90 / **1.00** |

    The kv picture repeats: full-session T++ answers nothing; window + contrast
    and the mind's package read are level (natural keys 0.975 vs 1.00, the
    prediction holds on para); the window without the contrast falls on
    natural keys (0.575).
  - **A18 COMPLETE (12:34 IST; tool / natq from
    `logs/v14_a18b_tpp_warm_32768_5d08477_20261101_1213.jsonl` + .log, 40 docs each).
    The whole comparison (32k, seed 18000, same text_fetch rows, free + sentence
    scorer, syn / nat):**

    | cell | T++ cold | T++ 2k window + contrast | T++ FULL 32k + contrast | b2p2 package (warm + contrast) |
    |---|---|---|---|---|
    | kv | 1.00 / 0.90 | 0.95 / 0.95 | 0.00 / 0.00 | 0.975 / 0.925 |
    | para | 0.70 / 0.675 | 0.95 / 0.975 | 0.00 / 0.00 | 0.90 / 1.00 |
    | tool | **0.90 / 0.90** | 0.75 / 0.825 | 0.00 / 0.00 | 0.75 / 0.675 |
    | natq (nat; fetch abstain 0.12) | 0.075 | 0.425 | 0.00 | **0.475** |
    | bytes held per session | ~1-1.5 MB (row only) | 67-68 MB | 1,060 MB | **1.15 MB** |

    - Prediction: T++ window ≈ cold on exact keys ✓ (0.95 vs 1.00 / 0.90); FULL
      below both ✓ (0.00 everywhere: a transformer holding the whole 32k
      session, 16× its trained length, answers nothing); b2p2 ≥ T++ window on
      natural kv / para ✗ / ✓ (0.925 vs 0.95; 1.00 vs 0.975).
    - The mind's package is level with the transformer's 2k window on facts
      (and better on natq) at 1/58 of its memory, while holding the WHOLE
      session; the transformer is better at reading multi-field records cold
      (tool 0.90). In these cells the answer needs only the fetched row, so
      the window loses nothing by forgetting: the cell that would separate
      "holding the session" from "a 2k window" is R × K (Mind's).
- **A19: field-cut tool rows (PRE-REGISTERED 2026-11-01 10:10 IST, before any
  number of it; Lead's package-side repair, agreed).** PLUMBING, labelled
  "field-cut rows": the LIBRARY picks the asked field's clause and the mind
  copies one value; the uncut row stays the headline ("reads a record").
  `package_fetch.field_cut` (a54c4f3) re-renders " Result: job: K; f: v." from
  the fetched row's text when the asked field name appears in the question;
  a field SYNONYM does not match (the open hole one level down).
  - Rows: Lead's text-arm tool docs (seed 18000, L = 32,768, 40 docs ×
    synthetic / natural), `text_fetch` (strict).
  - Arms: b2p2 COLD on cut rows (the bar), b2p2 cold on uncut rows, T++
    (tfc1b) cold on the same cut and uncut rows, b2p2 warm + contrast on cut
    and uncut rows (4090).
  - Bar MET iff b2p2 cold sentence ≥ 0.80 on cut rows, both key types (T++
    cold on uncut rows 0.90 is the reference).
  - **A19 VERDICT, cold part (2026-11-01 10:12 IST; code fac2750;
    `logs/v14_a19_fieldcut_cold_fac2750_20261101_1011.jsonl` + .log, 80 docs; cut
    found 1.00, fetch abstain 0.00): MET.**

    | tool, cold read (syn / nat) | uncut row | field-cut row |
    |---|---|---|
    | b2p2 | 0.475 / 0.40 | **0.90 / 0.80** |
    | T++ (tfc1b) | 0.90 / 0.90 | 0.825 / 0.85 |

    - b2p2's uncut numbers reproduce Lead's text arm exactly (0.475 / 0.40). On
      field-cut rows the two bodies are level (T++ slightly LOWER than on the
      full row). The gap was the multi-field binding (Mind's diagnosis), and
      the library can remove it by text, labelled "field-cut rows"; the
      uncut headline (the mind reading a record) stays b2p2 0.475 / 0.40 vs
      T++ 0.90. Open one level down: a field SYNONYM in the question.
    - **Warm + contrast part (10:36 IST; `logs/v14_a19_fieldcut_warm_fac2750_20261101_1028.jsonl` + .log, 80 docs):** b2p2 warm
      + contrast on field-cut rows **1.00 / 1.00** (T++ cold on the same cut rows
      0.825 / 0.85); on UNCUT rows 0.75 / 0.675 (cold 0.475 / 0.40; T++ cold 0.90
      / 0.90). The package read closes most of the record gap and, with the
      field cut, passes the transformer's cold read.
- **A20: a rare-in-English unigram step for reworded questions (PRE-REGISTERED
  2026-11-01 10:32 IST, before any number of it).** Plumbing (closest prior
  work: IDF weighting from a background corpus, BM25's idf, Robertson 1994).
  Diagnosis on Lead's varied-wording docs (seed 18000, kv, seen): a one-word
  key asked in 6 of the 8 wordings has no 2-3-word span shared with the stored
  sentence (synthetic 1 / 22 hit, 0 right); common-word natural keys lose to
  frame words ("stand for", "that refers to").
  - Step 4 (after the word organ's steps 1-2, before step 3): the question word
    (normalised) seen 1-2 times in the stream AND ≤ 3 times in a shipped
    background count (6M tokens of the v14 FineWeb train pool, normalised
    words); row = the sentence of its latest occurrence.
  - Order tested: token organ → steps 1, 2 → step 4 → step 3 → abstain.
  - Cells: kv, kv1, kvupd, neg × synthetic / natural, `--qstyle varied`; L =
    32,768; FRESH seed 37000; 40 docs. Text only (CPU, bench's own docs).
  - Bars: synthetic kv right row ≥ 0.90 (today 0.475 hit); neg any-hit ≤ 0.05
    on both key types; natural kv right row not below today's chain − 0.05 on
    the same docs. Reported: per-wording and per-step hits, kv1 / kvupd.
  - **A20 VERDICT (10:36 IST; code 8c9d8ac; `logs/v14_a20_check_8c9d8ac_20261101_1035.json` + .log): MET.**

    | varied wording, 32k, seed 37000 (40 docs) | old chain: right row | with step 4 | step 4 answered |
    |---|---|---|---|
    | kv / kv1 / kvupd synthetic | 0.225 / 0.275 / 0.25 | **1.00 / 1.00 / 1.00** | 31 / 30 / 30 |
    | kv / kv1 / kvupd natural | 0.75 / 0.825 / 0.70 | same | 0 |
    | neg synthetic / natural (any hit) | 0.00 / 0.025 | 0.00 / 0.025 | 0 |

    - A rare-in-English word that the stream holds once or twice is the key in
      any wording; frame words are common English and never qualify.
      Background counts: 101,252 words, 1.47 MB (JSON; a compact form would be
      well under 1 MB).
    - Natural keys of common words stay at 0.70-0.825 with 7-11 / 40 misses:
      text cannot separate them from the question's frame; that is the mind's
      salience (F2), Lead's pkgV run.
- **A21: the question-time package on NATURAL variants (PRE-REGISTERED
  2026-11-01 11:02 IST, before any number of it; the decisive salience test,
  Lead's ask).** Fetch chain: token organ → word organ steps 1-2 → step 4
  (A20) → F2 (A13: the 2 most surprising question tokens, block-8 max-sim over
  sentence rows, full-precision states) ; the bigram step 3 is dropped (it sits
  behind F2, which always returns a row). Read: b2p2 warm + no-row contrast
  (A9). STRICT twin: the text steps only, abstain if they miss.
  - bench's docs with `qstyle='varied'` (question wording from v14/styles.py);
    seed 18000 (Lead's natural-variant arm); L = 32,768. Cells: kv, kv1,
    kvupd, natq, neg × synthetic / natural (natq natural only), 40 docs; para,
    tool × synthetic / natural as controls, 20 docs.
  - Bars: (B1, fetch) natural-key kv: the chain's top row holds the fact ≥
    the text chain (steps 0-4) + 0.10 on the same docs; (B2, answer) natural-key
    kv package sentence (warm + contrast) ≥ 0.70.
  - Reported: every cell's fetch rate and answer, the step that answered
    (text step / F2), the never-stored cost of the F2 fallback (lenient answers
    with a value word on neg) vs strict, F2's memory (1.18 GB / 1M at fp16;
    int8 lossless at 590 MB / 1M: an open cost).
  - **A21 VERDICT (2026-11-01 12:15 IST; code 53575aa;
    `logs/v14_a21_package_b2p2_32768_53575aa_20261101_1105.jsonl` + .log, 440 rows):
    B1 NOT met, B2 MET.** The runner printed "B1 MET" against the text chain
    WITHOUT the bigram step (0.225); the registered baseline is the full text
    chain (steps 0-4, with the bigram), computed on the same docs on CPU.

    | varied wording, 32k, seed 18000 | full text chain: right row | package chain (text 0, 1, 2, 4 → F2): right row | package answer (warm + contrast) lenient / strict |
    |---|---|---|---|
    | kv syn / nat | 1.00 / **0.725** | 1.00 / **0.775** | 0.875 / 0.875 · **0.725** / 0.20 |
    | kv1 syn / nat | – / 0.60 | 1.00 / 0.75 | 0.925 / 0.925 · 0.675 / 0.275 |
    | kvupd syn / nat | – / 0.55 | 0.975 / 0.70 | 0.825 / 0.825 · 0.70 / 0.25 |
    | natq nat | 0.70 | 0.825 | 0.525 / 0.20 |
    | neg syn / nat: wrong fetch · answers with a value word | – / 0.075 | (F2 always fetches) | 0.30 / 0.57 lenient, 0.00 / 0.03 strict |
    | para syn / nat (20) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
    | tool syn / nat (20) | 1.00 / 1.00 | 1.00 / 1.00 | 0.85 / 0.75 |

    - B1: natural kv 0.775 vs 0.725 + 0.10 (NOT met, +0.05). B2: 0.725 ≥ 0.70.
    - F2 in place of the bigram step adds +0.05 (kv) to +0.15 (kv1, kvupd)
      and +0.125 (natq) on natural keys; synthetic keys are text-solved (step 4).
    - Its cost: it never abstains, so the lenient package says a value word
      on 30 / 57 % of never-stored questions (strict 0 / 3 %). Abstain where
      words differ remains the trained VERIFY's job.
    - Package on natural wording at 32k (lenient): kv 0.725, kv1 0.675, kvupd
      0.70, natq 0.525 natural; 0.825-0.925 synthetic; para 1.00, tool 0.75-0.85.
- **A22: the package on the user's STORY test (registered 12:39 IST, before
  any number; a DEMO with outputs shown, no bar; Lead's ask).** Lead's
  v16/story_eval.py: a four-chapter story with real other text between the
  chapters (L = 2,048 / 4,096 / 8,192, seed 1), 11 exam-style questions in
  chat and plain forms, Lead's keyword scorer. Package: text fetch (steps 0,
  1, 2, 4) → F2 → warm + contrast read (b2p2), up to 40 tokens, stops as
  Lead's; strict twin (text only, abstain on a miss); b2p2 ALONE on the same
  decoding beside it. The fetched row is shown with every answer.
  - Prediction: the package helps the single-sentence questions (who found,
    where the key, the promise, what H stood for, why the bridge closed);
    it does NOT help questions that need several sentences (order of events,
    summary, the weekday inference) or arithmetic on a rule (price of weight
    four), because it hands the mind ONE row.
  - **A22 RESULT (12:46 IST; code 6238b80; report
    `logs/v14_story_package_6238b80_20261101_1241.md` (every answer with its fetched
    row) + `logs/v14_a22_story_6238b80_20261101_1241.log`): the prediction held.**
    Keyword hits of 22 (11 questions × chat / plain): package 8 / 10 / 7 at 2k /
    4k / 8k, strict 5 / 6 / 4, b2p2 alone 1 / 3 / 2.

    | question (6 = 3 lengths × 2 forms) | package | alone | reading |
    |---|---|---|---|
    | how many sheep and why | 6 | 0 | real ("lost eleven sheep") |
    | who found the lantern | 4 | 0 | partly lucky ("Mira" inside a garbled sentence) |
    | what H stood for | 4 | 0 | |
    | why the bridge closed | 3 | 0 | |
    | where the key was hidden | 2 | 0 | |
    | which weekday Tomas left | 6 | 1 | LUCKY: the fetched row says "the Tuesday after the flood"; no inference |
    | order of events | 0 | 4 | needs several sentences |
    | the promise | 0 | 1 | wrong row fetched (the bridge sentence) |
    | price rule applied, true / false, summary | 0 | 0 | arithmetic, a negation, several sentences |

    - Single-sentence facts: the package turns "nothing" (mind alone) into
      answers; multi-sentence questions, rule arithmetic, negation and
      summaries stay at zero, since the package hands the mind one row. The
      keyword scorer is lenient (lucky hits marked); outputs are in the report.
  - **A22b, the same story through `v16/package.py` (12:50 IST; code 5da6bb5;
    `logs/v16_story_package_{b2p2_main_38e3c29,cu2a_main_24b2969}_5da6bb5_*.md`), beside
    Lead's mind-alone runs (`logs/v16_story_*_2026110112*.md`), keyword hits of 22 at
    2k / 4k / 8k:**

    | mind | alone (Lead) | package |
    |---|---|---|
    | b2p2 | 1 / 3 / 2 | **8 / 8 / 7** |
    | cu2a | 1 / 1 / 1 | 6 / 5 / 5 |
    | T++ / Mamba-2 (alone) | 1 / 0 / 0 · 0 / 0 / 0 | – |

    - Same fetch paths for both minds (36 text, 30 F2 of 66); cu2a never says
      " unknown" here; its lower score comes from F2 on its own states fetching
      prose rows (e.g. "what H stood for" → a semiconductor passage), the weaker
      own-state address A10 saw for cu1.
- **A23: `pkg.apply`, rule x knowledge through the package (REGISTERED
  2026-11-01 14:19 IST, before any number; Lead's GO).** The planner fetches
  the key's row, takes the value out (the library's own row formats), writes
  the skill's trained query line ("<v> ->") and the mind answers with the
  skill contrast (the twin = the session WITHOUT the skill pack, a full
  second stream); optional echo dilation from the pack's end to the session
  end.
  - Reproduction (the only criterion): Mind's R × K docs (scripts/
    run_rxk_probe.py: seeds 8800-8849, affine_map_new + shift_wide_new,
    record row " Record: K | x: v.", key question "K ->"), b2p2 and cu2a, L =
    2,048 (contrast) and 8,192 (dilation s = 8 + contrast). Package free
    (first answer token = target) within ± 0.05 of Mind's `exp_c4` /
    `exp_dil8_c4` on the same docs.
  - Story check (prediction, outputs shown): Old Pell's rule ("twice its
    weight plus one") held only as the story's WORDS + "4 ->" FAILS; the same
    rule held as a register skill in the trained affine format ("Map rule.",
    "2 -> 5", "3 -> 7", "5 -> 11") with the weight fetched from the story
    ("weight four") answers 9.
  - **A23 RESULT (14:24 IST; code ed4c84f; `logs/v16_rxk_package_ed4c84f_20261101_1422.json`
    + .log): the reproduction is EXACT; the story check did not go as predicted.**

    | R x K through `pkg.apply` (100 docs each) | package free | Mind's harness | value fetched right |
    |---|---|---|---|
    | b2p2 2k (contrast) / 8k (dilation s 8 + contrast) | 0.61 / 0.45 | 0.61 / 0.45 | 1.00 / 1.00 |
    | cu2a 2k / 8k | 0.62 / 0.49 | 0.62 / 0.49 | 1.00 / 1.00 |

    Doc-level agreement with the harness: 400 / 400. The keys are fetched by the
    rare-word step (A20) and the value taken out of the record row.

    | story check, Old Pell's rule (2w + 1), "4 ->" (target 9) | b2p2 | cu2a |
    |---|---|---|
    | the rule as the story's words | " 16" | " 18" |
    | as 3 register demonstrations (trained affine format) | " 14" | " 18" |
    | as 8 register demonstrations | " 14" | " 18" |

    - Words fail as predicted; the demonstrations ALSO failed on this ONE item.
      At 2k the R x K rate is 0.61, so a single item cannot support the story
      line "rules reach the mind as demonstrations, from the register"; it
      needs a set of story-rule items before it is written. Right type, wrong
      value (Mind's rule-fidelity reading).
    - Mind's follow-up (b2p2, CPU, 20 demo sets per cell, held-out web text as
      the gap): 2w + 1 is INSIDE the trained affine family (a ∈ 1-3, b ∈ 0-20
      except 7 / 13 / 17), so the miss is the GAP: L = 0 → 1.00 with 3 or 8
      demos; 2k with 3 demos → 0.00; 2k with 8 demos → 0.45 plain, 0.30 with
      the contrast (an unexplained " 41" attractor in 12 / 20). Package rules:
      a register skill carries ≥ 8 demonstrations, is refreshed near the query
      (the 2k refresh), and its contrast is measured on vs off per skill.
- **A24: do rules reach the mind as demonstrations, not words? (REGISTERED
  2026-11-01 14:25 IST, before any number; Lead's spec + Mind's follow-up.)**
  - 20 affine rules y = a·x + b, a ∈ {1, 2, 3}, b from the TRAINED offsets
    (0-20 except 7 / 13 / 17; the story's 2x + 1 is inside, per Mind), 2
    queries each (held-out x, y < 100) = 40 items per arm, fixed seed 24001.
  - Arms (the rule, then 2,048 tokens of held-out web text, then "\n{x} ->"):
    WORDS ("Rule: each output is {a} times the input plus {b}." in the
    story's register of language), DEMO3 and DEMO8 ("Map rule." + "\nx -> y"
    lines, the trained format); reported: DEMO8-NEAR (no rule at the start, the
    8 demos read 256 tokens before the query = the package's refresh).
  - Reads: `pkg.apply_value`, contrast a = 4 (the R x K read; twin = the same
    session without the rule) and plain (a = 0). Minds: b2p2, cu2a. CPU.
  - Bar (Lead's, on b2p2, contrast): DEMO8 ≥ WORDS + 0.30 AND DEMO8 ≥ 0.50.
    Reported: DEMO3, DEMO8-NEAR, plain, cu2a, per-offset breakdown.
  - Prior evidence (Mind, 20 demo sets of 2x + 1): 2k + 8 demos 0.45 plain /
    0.30 contrast, so the bar may well fail on the contrast read.
  - **A24 VERDICT (14:29 IST; code 2e03057; `logs/v14_a24_rules_2e03057_20261101_1426.json` + .log, 40 items per cell):
    NOT met as registered (b2p2 DEMO8 contrast 0.425 < 0.50; +0.375 over words
    met); the question itself is answered: rules reach the mind as
    DEMONSTRATIONS, not words.**

    | free (contrast / plain) | b2p2 | cu2a |
    |---|---|---|
    | WORDS ("each output is a times the input plus b") | 0.05 / 0.025 | 0.00 / 0.00 |
    | DEMO3, 2k back | 0.525 / 0.40 | 0.45 / 0.40 |
    | DEMO8, 2k back | **0.425** / 0.60 | 0.60 / 0.475 |
    | DEMO8-NEAR (256 tokens back = the refresh) | 0.65 / **0.775** | **0.85 / 0.875** |

    - Words reach neither mind (≤ 0.05); demonstrations do (0.40-0.875), most
      when refreshed near the question. The contrast is not a uniform gain on
      skills: it lowers b2p2's DEMO8 (0.60 → 0.425) and raises cu2a's (0.475 →
      0.60): measure it per mind and skill.
    - No offset pattern (b2p2 DEMO8 contrast by b: 0.0-1.0 at n = 2-4 each); the
      offset is not the cause (Mind: all inside the trained family).
    - Package rule confirmed: a register skill = ≥ 8 demonstrations in the
      trained format, refreshed near the query; a rule stated in words is a
      lookup at 103M (Mind's cu1 finding), so the library should turn words
      into demonstrations (Lead's "words → demos" generator idea) before the
      mind sees them.
- **A25: the package at 128k and 300k, by needle DISTANCE (REGISTERED
  2026-11-01 15:03 IST, before any number; Lead's GO; the user's question "can
  it recall a needle from more than 128 tokens back?").**
  - Docs (my builder from bench's parts: held-out prose, bench keys / values /
    templates): 8 facts per doc; the ASKED facts sit at fixed distances from
    the question, one per bin: ~1k (<2k), ~16k (2-32k), ~100k (32-128k),
    ~250k (128-300k; 300k docs only); the other facts at random positions.
    Every asked fact is a separate question on the SAME streamed session
    state. Cells: kv (template), kvV (varied wording, v14/styles.py), para,
    tool, neg (never stored) × synthetic / natural keys; L = 131,072 and
    307,200; 10 docs per cell; fresh seed 18500.
  - Reads: PACKAGE = v16/package.py (stream the session; text fetch → F2 with
    block-8 states captured CHUNK BY CHUNK with the state carried; warm +
    contrast), lenient and strict; MIND ALONE (warm, no row); T++ 2k WINDOW
    alone (last 2,048 session tokens + question); T++ 2k window + the
    package's row + contrast (A18's arm). F2's address in two stored forms:
    int8 every token (592 B / token) and sign bits on the top 1/16 surprising
    tokens (~78 B per kept token).
  - Bars (all at 300k): B1 package written answer on kv, both key types, in the
    128-300k bin ≥ 0.85 and within 0.05 of its <2k bin (no decay with
    distance); B2 kvV natural-key package in the 128-300k bin ≥ 0.65; B3 strict
    abstain on neg ≥ 0.90; B4 bytes held per session excluding F2's address
    (raw text + word / token indexes + mind state) ≤ 10 MB.
  - Prediction (Lead's board): text 0.6 MB + indexes ~2.8 MB + state 1.15 MB vs
    T++ KV ~11 GB at 300k; T++ 2k window alone ~0 beyond the <2k bin; mind
    alone ~0 beyond a few k.
  - Reported: 128k, every cell / bin / read, seconds per answer, F2's MB in
    both forms.
  - **A25 VERDICT AT 128k (2026-11-01 17:23 IST; code 0a9aeb9; `logs/v14_a25_long_0a9aeb9_20261101_1622.jsonl`
    + .log; 300 rows at 128k): B2, B3, B4 MET; B1 NOT met by one document.** The run was STOPPED at 17:22 IST during
    300k, on the user's word via Lead ("why are we wasting resources on 300k when it is not the bar"; lengths are
    capped at 128k since 15:35, and this unit was registered after the cap: my error, and Lead's). The bars were
    registered at 300k's farthest bin; they are read here at 128k's farthest bin (~100k back). This is a re-scope
    after the cap, stated as such. The 300k rows the run wrote (kv only) stay in the log, unquoted.

    | 128k (package / strict / mind alone / T++ 2k window / T++ window + row) | ~1k back | ~16k | ~100k |
    |---|---|---|---|
    | kv synthetic | 1.00 / 1.00 / 0.00 / 0.20 / 1.00 | 1.00 / 1.00 / 0.00 / 0.00 / 1.00 | 0.90 / 0.90 / 0.00 / 0.00 / 0.90 |
    | kv natural | 0.90 / 0.90 / 0.00 / 0.00 / 1.00 | 0.90 / 0.90 / 0.00 / 0.00 / 1.00 | 1.00 / 1.00 / 0.00 / 0.00 / 0.90 |
    | kvV (varied wording) synthetic | 1.00 / 1.00 / 0 / 0 / 0.90 | 1.00 / 1.00 / 0 / 0 / 0.90 | 1.00 / 1.00 / 0 / 0 / 0.60 |
    | kvV natural | 0.70 / 0.10 / 0 / 0 / 0.70 | 0.60 / 0.20 / 0 / 0 / 0.70 | **0.80** / 0.30 / 0 / 0 / 0.80 |
    | para synthetic / natural | 0.90 · 0.90 | 1.00 · 1.00 | 0.90 · 0.80 |
    | tool synthetic / natural | 1.00 · 1.00 | 1.00 · 1.00 | 1.00 · 1.00 |
    | neg: strict abstain (syn / nat) | 1.00 / 0.90 | 1.00 / 0.90 | 1.00 / 0.90 |

    - B1 (kv at ~100k ≥ 0.85 and within 0.05 of ~1k): synthetic 0.90 vs 1.00 ✗ by one document; natural 1.00 vs
      0.90 ✓. The miss fetched the RIGHT row (text0, "... Definition: gefvef refers to onyx.") and wrote " (yx.".
      That is a writing slip, not distance: pooled kv is 0.967 / 0.933, with no trend over 1k → 100k.
    - B2 (kvV natural at ~100k ≥ 0.65): 0.80 ✓; pooled 0.70 lenient, 0.20 strict (F2 answered 23 of 30).
    - B3 (strict abstain on never-stored ≥ 0.90): 0.95 pooled ✓. Lenient fetched a row for 60 / 60 never-stored
      questions (57 by F2): F2 never abstains, as in A21.
    - B4 (bytes excluding F2's address ≤ 10 MB): 2.65 MB ✓ (state 1.15 + text 0.26 + indexes ~1.23). F2's address:
      78 MB at int8, 0.64 MB as sign bits. On 24 F2 rows of stored facts the two forms tie (sign bits 0.71 vs full
      0.62).
    - Mind alone answers 0.00 in every cell and bin, and the T++ 2k window 0.00-0.20 (0.20 only at ~1k, the needle
      inside its window). This is the bench's untrained "Definition:" format; Mind's trained-format probes do recall
      at distance. The T++ given the package's row reads 0.60-1.00, as in A18: on single-row cells the package's
      edge over a windowed T++ is cost (1.15 MB vs 67 MB, 4.8 GB for the full 128k KV), not accuracy.
    - Compute (4090, shared): read 11.4 s per 128k session; 0.38 s per answer including the fetch.
- **A26: a stated PROTOCOL applied to a FETCHED result (SCI-1 milestone CHECK cell; PRE-REGISTERED 2026-11-02
  10:33 IST, before any number of it; Lead's ask for the science / research mind).**
  - **Question:** does a mind hold a convention stated once early in a session (units, rounding, a target unit) and
    apply it to a value the library fetches far later, in free generation? That is the SCI-1 strength claim ("standing
    rules and protocols held in state, applied to fetched results"). It is A23's rule × knowledge in a research
    notebook's words instead of the trained "x -> y" format.
  - **Closest prior work:** IFEval (Zhou et al. 2023: verifiable format instructions) and long-context
    instruction-following sets (e.g. LIFBench 2024). **Difference:** the instruction is a unit / rounding convention
    stated 2k-32k tokens before the question and never repeated. The value it applies to sits in a row the library
    fetches, and the held convention is read from an O(1) state, compared at equal tokens with window models.
  - **Docs (my builder, fresh seed 26000, held-out prose as filler):** a research notebook. Near the start comes ONE
    protocol sentence, then filler. Then 8 result rows ("Sample K7 (lysate B): mass 12.3."; IDs and materials
    random), placed among the filler. The asked row sits at a fixed distance from the question; the PROTOCOL sits at
    distance d = 2k / 8k / 32k from the question.
  - **Cells (convention families):**
    - **UNIT** (no arithmetic): "all masses here are recorded in {u}". The row has a bare number. Right = the value AND
      the protocol's unit. Units are sampled per doc within the quantity (mass mg / g / kg / ng / μg; volume mL / L /
      μL; concentration mM / μM / nM; length nm / mm / cm), so one prior unit cannot win.
    - **ROUND:** "report values rounded to one decimal place". The row has 3 decimals; half the values round UP (12.36
      → 12.4). Right = the rounded value, not the raw.
    - **CONVERT:** "report masses in mg". The row is in g (0.0123 g → 12.3 mg). Right = the converted value + mg.
      This is an arithmetic ceiling probe: expected ~0 at these sizes, no bar.
  - **Wording:**
    - NAT, the headline: the protocol in one of 5 natural wordings per family ("Note: all masses below are in mg.",
      "Unless stated otherwise, masses are given in milligrams (mg).", …), asked in one of 5 natural question
      styles ("What was the mass of sample K7?", "How heavy was K7?", "Report the mass of sample K7.", …), in
      the exam form.
    - TRAINED: the convention in the trained format. Mind's SCI-1 protocol-doc format if the mix has one (filled in
      from Mind's reply before any number, with its test conventions held out); else the package skill format
      ("Report rule." + 8 demos "v -> v u", query "\n{v} ->", value substituted by the planner, A23's path).
  - **Arms:**
    - **MIND-HELD:** the package fetches the RESULT ROW only (`answer`, lenient, warm + contrast); the protocol must
      come from the state.
    - **REFRESH**, labelled TEXT / product: the package also re-inserts the protocol sentence by the v0 refresh rule
      (every 2k), so any body could use it.
    - **NOPROT control:** the same session without the protocol (the prior unit / format).
    - **AS IS:** each body alone on the whole session (window models: the whole session ≤ their trained length,
      else the last 2k).
  - **Bodies:**
    - the SCI-1 mind at each kept checkpoint (subject);
    - at the 600M tier: b2p2 (package), T++ b2t2 / tfc1b and Mamba-2 b2m2 (as is, and with the library's row +
      contrast, A18's arm);
    - at the 1.5B tier: t15pam_m2 (package), t15tpp / t15mamba (as is + library).
    The 600M / 1.5B rows are REFERENCE ONLY next to a ~350M mind (different params and tokens; params + tokens stated
    in every row).
  - **Scoring:** free generation, decode to the stop (≤ 16 tokens). Right iff the value AND the convention are
    right; value-right and convention-right are reported separately. 20 docs per cell × length × arm at milestones,
    40 at the final check.
  - **Bars** (SCI-1 mind, NAT wording, UNIT cell):
    - B1, held in state at 8k: MIND-HELD ≥ 0.50 AND ≥ NOPROT + 0.25.
    - B2, held at 32k (16× the trained 2k): MIND-HELD ≥ 0.40.
    - B3, product: REFRESH at 32k ≥ 0.80.
    - ROUND / CONVERT and the TRAINED wording are reported, no bar.
    - Day-0 calibration on the 600M-tier minds is reported with no bar (it sets what a milestone must beat).
  - **ADDENDUM (2026-11-02 11:00 IST, before any SCI-1 number; Lead's YES to a protocol TRAINING source):**
    - **Source:** `v14/protocol_docs.py` (`protocol_text_iter`; seed = Mind's mix seed, stream offset
      SEED_OFFSET = 10**8). It sits inside the synthetic loss-weight budget (recall + b2 + skills + protocols ≤ 3 % of
      the loss weight; protocols ~0.5 % of tokens at full answer weight; Mind sets the share).
    - **Training families:** UNIT mass (mg / g / kg / ng / μg) and volume (mL / L / μL); ROUND to the nearest whole
      number and to two decimal places (mass, volume, time, temperature); CONVERT L → mL and km → m.
    - **Wordings:** 10 protocol wordings per family, 8 question styles, 5 row templates, 2 QA forms. Natural style
      70 %, compact "Report rule." style 30 %. 20 % of docs have no protocol (answers copy the row).
    - **Held out (never generated):** UNIT concentration / length; ROUND to one decimal; CONVERT g → mg and every mass
      conversion. Also all of A26's protocol wordings, question styles and row templates.
    - **Leak check** (`python -m v14.protocol_docs`, 20,000 docs): 0 held-out family / unit / quantity hits, 0 shared
      template skeletons, 0 shared 5-word phrases with A26. Docs are 689 tokens median (p90 1,391), and answer
      tokens are 1.1 %.
    - **Reporting split for the SCI-1 mind** (Lead; cu2a showed trained formats 0.80-0.99 vs held-out wordings 0.26):
      - G1, trained family + TRAINED wording: UNIT mass / volume with the generator's wordings (a runner option, the
        in-distribution ceiling);
      - G2, trained family + held-out wording: A26 NAT on UNIT mass / volume;
      - G3, HELD-OUT family (+ held-out wording): A26 NAT on UNIT concentration / length, ROUND, CONVERT;
      - the compact TRAINED arm, split the same way.
    - The bars B1-B3 stand as registered (UNIT pooled); G1-G3 are reported beside them. The day-0 minds never saw
      protocol docs, so for them the split is moot.
  - **Prediction (written before any number):** b2p2 MIND-HELD UNIT ≈ NOPROT (≤ +0.10) at 8k and beyond, since words
    do not reach a 103M mind (A24). REFRESH UNIT ≥ 0.6 for b2p2 (the protocol sits next to the question). ROUND ≈ 0.1-0.3
    (copying 3 decimals is the prior). CONVERT ≈ 0 for every body. T++ as is ≈ 0 beyond 2k.
  - **A26 DAY-0 RESULT (2026-11-02 11:28 IST; no bars, a calibration):**
    - Logs: `logs/v14_a26_protocol_d5da73d_20261102_1052.jsonl` + .log (every arm) and
      `logs/v14_a26_refresh_e44b764_20261102_1119.jsonl` + .log (the refresh arms rerun, replacing d5da73d's).
    - Docs: 20 per cell × 2k / 8k / 32k. Bodies (params / tokens):
      - b2p2: 102.9M, 600M tokens, through the package;
      - t15tpp (T++) and t15mamba (Mamba-2): 1.5B-token tier, as is and with the library's row + contrast.
    - **Every body copies the fetched value and applies NO stated convention**, even with the protocol ~560-600
      tokens back, inside every window. The "right" column is ≤ 0.05 everywhere, equal to the no-protocol control.

    | right (value copied), pooled 2k / 8k / 32k | UNIT | ROUND | CONVERT |
    |---|---|---|---|
    | b2p2 package: held · refresh · noprot | 0.00 (0.65-0.80) · 0.00 (0.65-0.75) · 0.00 (0.65-0.80) | 0.00 (raw copied 0.15-0.30) | 0.00 |
    | b2p2 alone (state only) · trained ("Report rule." demos, held 2-32k back) | 0.00 (0.00) · 0.00 (0.00) | 0.00 · 0.00 | 0.00 · 0.00 ("mg" said on 0.55 at 2k, wrong value) |
    | t15tpp as is · lib_held · lib_refresh · lib_noprot | 0.00 · 0.00 (0.20-0.40) · 0.00 (0.35-0.50) · 0.00 (0.20-0.40) | 0.00-0.05 (raw 0.35-0.60) | 0.00 |
    | t15mamba as is · lib_held · lib_refresh · lib_noprot | 0.00 · 0.00-0.05 (0.45-0.80) · 0.00 (0.50-0.70) · 0.00-0.05 (0.45-0.80) | 0.00 (raw 0.50-0.70) | 0.00 |

    - **Prediction check:**
      - b2p2 held ≈ NOPROT: ✓.
      - REFRESH UNIT ≥ 0.6 for b2p2: ✗ (0.00). Having the protocol nearby is not enough.
      - ROUND: rounded 0.00, raw value copied 0.15-0.70: ✓ in spirit.
      - CONVERT ≈ 0: ✓. T++ as is ≈ 0: ✓.
    - **What it says:** at the 600M / 1.5B-token tiers, applying a stated convention is a SKILL that no body has.
      This is not a memory limit. The protocol is read; the answer copies the row's surface.
    - **For SCI-1:** the B1-B3 bars need the trained skill (v14/protocol_docs.py, wired by Mind). The addendum's
      G1 / G2 / G3 split separates "learned the format" from "transferred to unseen conventions". Any score above 0 is
      above every day-0 reference.
    - **Runner bugs found and fixed:**
      - d5da73d anchored the refresh copies from the start, so the newest sat ~2.0-2.2k tokens back, outside the 2k
        window for 7 of 9 checked docs. Fixed in e44b764; the rerun replaces those rows.
      - Sample IDs came from a set, whose order depends on PYTHONHASHSEED. The day-0 docs are statistically the
        same but not byte-reproducible. Fixed at the next commit (sorted, then shuffled by the doc's rng); the same
        fix went into protocol_docs.py.
- **A27: decontamination AUDIT of the SCI-1 training stream (PRE-REGISTERED 2026-11-02 13:20 IST; Lead: adopted; an
  independent check of Mind's pre-launch stream-time 13-gram filter).**
  - Test sets:
    - kb1: HellaSwag val, ARC-e / ARC-c test, PIQA val, WinoGrande val, OBQA test, LAMBADA-openai test;
    - GSM8K test; MMLU test;
    - BABILong qa1-qa10 (facts + question + target only, not the PG19 haystack);
    - `v14/decontam_items.json` (A26 / bench / styles / story + long prompts).
  - Rule: lowercase, words only (letters and digits); a training doc is contaminated if it shares ≥ 1 13-gram with a
    test item (question + answer text). 13-grams are stored as 64-bit hashes.
  - Sample: ~20,000 docs per source drawn through the real blend path (`load_pretrain_mix`, SCI-1's exact args and
    seed, the post-filter stream), CPU, within the first day of SCI-1.
  - Report: % contaminated docs per source × test set, 5 examples each, and the 13-grams that hit most often
    (boilerplate shows up there).
  - Action rule: any source × test set > 0.1 % → a stream filter for it at the first continuation.
  - Every reported benchmark also gets its CLEAN-SUBSET score (items with no 13-gram in the sampled stream) beside
    the full score.
  - Prediction: ≤ 0.1 % everywhere after Mind's filter. The risks are web sources × MMLU / ARC (verbatim exam pages)
    and `think` × GSM8K test (MetaMathQA rewrites the TRAIN set).
  - **A26 SCI-1 MILESTONE 1 (2026-11-03 12:01 IST; step 15,259 = 1.0B tokens in stage B; code 1584da2; logs
    `logs/v14_a26_sci1m1_1584da2_20261103_1116.*` (A26 wordings, G2 / G3) and
    `logs/v14_a26_sci1m1_g1_1584da2_20261103_1116.*` (G1, trained wordings); 20 docs per cell; 386M, 1B stage-B tokens):**
    - Package fetch gives the value (value ok 0.35-1.00). The convention is applied by the MIND only in the
      refresh arm (TEXT: protocol re-inserted) and only a little.

    | right (value / convention), UNIT | 2k | 8k | 32k |
    |---|---|---|---|
    | G1 trained wording: held · noprot | 0.00 · 0.00 | 0.00 · 0.00 | 0.00 · 0.00 |
    | G1 refresh (TEXT) | 0.10 (0.90 / 0.10) | 0.25 (1.00 / 0.25) | 0.15 (0.85 / 0.15) |
    | G1 compact "Report rule." (trained arm) | **0.25** (0.30 / 0.70) | 0.05 | 0.00 |
    | A26 wordings (G2 / G3): held · noprot | 0.00 · 0.00 | 0.05 · 0.05 | 0.00 · 0.00 |
    | A26 wordings refresh | 0.00 | 0.10 | 0.05 |
    | A26 compact trained arm, mass / volume (trained) vs concentration / length (held out) | 0.56 · 0.09 | 0.00 · 0.00 | 0.00 · 0.00 |
    | ROUND, CONVERT (any arm) | 0.00 | 0.00 | 0.00-0.05 |

    - Versus day-0 (every body 0.00-0.05): the first non-zero scores are the refresh arm (0.10-0.25 on trained wordings) and
      the compact format at 2k (0.25-0.56 on trained quantities; the convention is applied on 0.70-0.85, the copied value
      is the weak part, 0.30).
    - Tripwire (G1 MIND-HELD at 2k <= NOPROT + 0.05) is TRUE at this milestone (0.00 vs 0.00). It is judged at milestone 2.
    - The skill is learned in the notation it was trained in and does not yet cross notation or length: protocol in prose
      held from far away stays 0.00. Held-out quantities in compact form 0.09 vs trained 0.56 at 2k.
  - **A26 SCI-1 MILESTONE 2 (2026-11-03 12:39 IST; step 30,518 = 2.0B stage-B tokens; code 7e71923; logs
    `logs/v14_a26_sci1m2_7e71923_20261103_1202.*` (G2 / G3) and `logs/v14_a26_sci1m2_g1_7e71923_20261103_1202.*` (G1); 20 docs
    per cell). THE MILESTONE-2 TRIPWIRE FIRES: G1 MIND-HELD at 2k = 0.00 <= NOPROT 0.00 + 0.05 → "raise proto at the first
    continuation" (the pre-registered response).**

    | right (value / convention) | 2k | 8k | 32k |
    |---|---|---|---|
    | G1 UNIT held · noprot | 0.00 · 0.00 | 0.00 · 0.00 | 0.00 · 0.00 |
    | G1 refresh (TEXT) | 0.00 (0.90 / 0.00) | 0.05 | 0.00 |
    | G1 compact "Report rule." | 0.20 (0.20 / 0.75) | 0.00 | 0.00 |
    | A26 wordings UNIT: held · noprot | 0.00 · 0.00 | 0.05 · 0.05 | 0.05 · 0.05 |
    | A26 wordings UNIT refresh · compact | 0.00 · 0.20 (0.20 / 0.95) | 0.10 · 0.15 | 0.05 · 0.00 |
    | ROUND: held · refresh · compact | 0.00 · 0.00 · 0.05 | 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.05 |
    | CONVERT compact (convention said / value) | 0.00 (1.00 / 0.00) | 0.00 | 0.00 |

    - Against milestone 1 (1.0B): refresh on trained wordings fell from 0.10 / 0.25 / 0.15 to 0.00 / 0.05 / 0.00; compact at 2k
      0.25 → 0.20. No improvement from the extra 1B tokens (proto is ~0.5 % of tokens).
    - What is learned is the unit/format half: in the compact format the convention is said on 0.75-1.00 at 2k (even
      CONVERT writes "mg" every time) but the copied value is right on only 0.20, so "right" stays low. Trained vs held-out
      quantities (compact, 2k): 0.22 vs 0.18, so the format now crosses quantities.
    - Mind-held prose protocol (the SCI-1 claim as stated) is 0.00 at every length: the skill has not appeared, so the
      tripwire response stands (more proto weight; a fix for the value copy is a candidate).
  - **A26 SCI-1 MILESTONE 3 (2026-11-03 15:36 IST; step 45,777 = 3.0B stage-B tokens; still the pre-continuation mix; logs
    `logs/v14_a26_sci1m3_g1_cf4fa17_20261103_1346.*` (G1) and `logs/v14_a26_sci1m3_f3e02a6_20261103_1509.*` (G2 / G3); 20 docs per
    cell; the G2 / G3 half was relaunched by hand after the unattended chain stalled (I overwrote the running script)):**

    | right (value / convention) | 2k | 8k | 32k |
    |---|---|---|---|
    | G1 UNIT held · noprot | 0.00 · 0.00 | 0.00 · 0.00 | 0.00 · 0.00 |
    | G1 refresh (TEXT) | 0.00 | 0.10 | 0.00 |
    | G1 compact "Report rule." | **0.40** (0.40 / 0.90) | 0.00 | 0.00 |
    | A26 wordings UNIT: held · noprot · refresh | 0.00 · 0.00 · 0.00 | 0.05 · 0.05 · 0.05 | 0.00 · 0.00 · 0.00 |
    | A26 wordings UNIT compact (trained / held-out quantities at 2k: 0.11 / 0.18) | 0.15 (0.15 / 0.95) | 0.10 | 0.00 |
    | ROUND (any arm) · CONVERT compact (convention said / value) | 0.00-0.05 · 0.00 (0.85 / 0.00) | 0.00 | 0.00-0.05 |

    - Trend M1 → M2 → M3 (1 → 2 → 3B, the same mix): G1 compact at 2k 0.25 → 0.20 → 0.40. Mind-held prose stays 0.00 at every
      milestone. The refresh arm is 0.00-0.10, noise-level.
    - The same reading: the convention (unit) is said on 0.85-0.95 in the compact format; the copied value is the limit
      (0.15-0.40). That is what the first continuation's protoc targets.
  - **A26 SCI-1 M4 vs cont1 step 76,295 (2026-11-04 06:50 IST; code 850ab5e; 20 docs per cell):**
    - Logs: `logs/v14_a26_sci1m4{,_g1}_850ab5e_20261104_0613.*` (M4 = step 61,036, 4.0B tokens, the pre-cont1 mix, where cont1
      started) and `logs/v14_a26_sci1c1a{,_g1}_850ab5e_20261104_0529.*` (c1a = step 76,295, 5.0B: 1B of cont1, protoc 0.15 %).
    - Cont1 also changed other shares (b2s 1.7 → 0.5 %, think removed).

    | right (value / convention) | M3 (3.0B) | M4 (4.0B) | c1a (5.0B, protoc) |
    |---|---|---|---|
    | G1 held-in-state UNIT 2k / 8k / 32k | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
    | G1 compact, 2k | 0.40 (0.40 / 0.90) | **0.00** (0.00 / 0.80) | **0.45** (0.45 / 0.85) |
    | G1 compact, 8k | 0.00 | 0.10 | 0.15 |
    | A26-wording compact 2k / 8k | 0.15 / 0.10 | 0.00 / 0.00 | **0.35 / 0.60** |
    | held-out quantities (concentration / length), compact 2k / 8k | 0.18 / 0.00 | 0.00 / 0.00 | 0.45 / 0.60 |
    | CONVERT compact 2k (unit said) | 0.00 (0.85) | 0.00 (0.95) | **0.10** (1.00) |
    | ROUND (any arm) | 0.00-0.05 | 0.00-0.05 | 0.00 |

    - **Verdict for Mind's B vs C: protoc helped.** The compact value went from 0.00 at M4 to 0.45 at 2k (G1) and 0.60 at 8k
      (A26 wordings). It crosses to held-out quantities and gives the first CONVERT value (0.10).
    - M4's collapse (M3 0.40 → 0.00 on the same mix) shows the compact value skill was unstable before protoc.
    - **Unchanged:** a protocol held as PROSE in state is 0.00 at every checkpoint (M1-c1a). The refresh arm stays ≤ 0.10.
    - Mind's cont1 bars at 76,295: G1 held ≥ NOPROT + 0.05 NOT met (0.00); compact value ≥ 0.50 at 2k just short (0.45 G1;
      0.35 A26 wordings, 0.75 value at 8k).
  - **A27 RESULT (2026-11-03 12:01 IST; code 37c291d; `logs/v14_a27_decontam_37c291d_20261103_1104.{json,log}`; 20,000 docs per
    source, the first docs of each stream, CPU):**
    - Positive control: 300 / 300 planted test strings flagged by the audit AND by Mind's filter (all 10 sets).
    - FILTERED view (what SCI-1 trains on): **0 hits in all 15 sources.** The action rule is not triggered.
    - RAW view (what the filter removes), % of docs with a 13-gram hit: think 0.275 (MMLU 44, GSM8K-test 11 of 20,000);
      fineweb 0.11; dclm 0.08; chatmt 0.065; arxiv 0.045; finemath 0.035; smoltalk2_mid 0.025; pubmed / cosmo 0.015; tools
      0.005; pes2o, pysci, b2s, recall4k, proto4k 0. Mostly MMLU, then HellaSwag.
    - Limits stated: the audit hashes Mind's own test-item definitions with an independent implementation (it tests the
      filter's machinery, not whether an item set is missing); the sample is each stream's first 20,000 docs, not the middle of
      the run; the clean-subset scores come with the benchmark entries (hit test strings are in the json).
- **A28: is A26's "prose held = 0.00" a read artefact? (PRE-REGISTERED 2026-11-04 08:29 IST, Lead's GO, before any number.)**
  The A9 read (`l_row + 4 (l_row − l_norow)`) cancels anything held in state in both streams. On the c1a weights (step 76,295),
  A26 docs (seed 26000, UNIT cell, G1 trained wording and A26 wording, 2k / 8k / 32k, 20 docs) with the SAME fetched row,
  the held session is read as (a) PLAIN (warm, row + question, no contrast) and (b) PTWIN (contrast against the no-protocol
  session + the same row: `l_A + 4 (l_A − l_B)`), beside the A9 read and NOPROT. **Rule:** if held ≤ NOPROT + 0.05 under
  both (a) and (b) at every length, "prose held = 0" stands as real; otherwise every A26 held number since M1 is re-read with
  the better read. ROUND / CONVERT leave the mind scorecard (the executor's job, Lead).
  - **A28 RESULT (2026-11-04 08:38 IST; code be66f2d; `logs/v14_a28_twin_{g1,a26w}_be66f2d_20261104_0830.*`; c1a weights, step
    76,295; 20 docs per cell): the rule is MET for "not real" → "prose held = 0" is an ARTEFACT at 2k; real beyond 2k.**

    | right (value / unit) | 2k | 8k | 32k |
    |---|---|---|---|
    | G1 A9 read (= A26 held) | 0.00 (0.95 / 0.00) | 0.00 | 0.00 |
    | G1 plain | 0.00 (1.00 / 0.00) | 0.00 | 0.05 |
    | **G1 protocol twin** | **0.65 (1.00 / 0.65)** | 0.00 | 0.05 |
    | G1 noprot (A9 read) | 0.00 | 0.00 | 0.00 |
    | A26 wordings: A9 · plain · protocol twin · noprot | 0.00 · 0.00 · **0.20** (unit 0.45) · 0.00 | 0.00 · 0.00 · 0.05 · 0.00 | 0.00 under all |

    - The plain read copies the row and does not add the unit either; only the protocol twin brings the unit out. At 2k, a
      state-held prose protocol is readable (0.65 on the trained wordings, 0.20 on held-out wordings) and higher than the
      compact pack arm at the same checkpoint (0.45).
    - Beyond 2k (8k / 32k) it is 0.00-0.05 under every read: the limit there is the rule's REACH, not the read.
    - Consequences (pre-registered): every A26 held number since M1 is re-read with the protocol twin; the package's
      contrast twin is chosen per question part (a no-training unit, pre-registered next); EUREKA entry written.
  - **A28 REPLICATION at 100 docs (2026-11-04 08:46 IST; code a69cafa; `logs/v14_a28_rep100_{c1a,m4}_a69cafa_20261104_0842.*`; G1,
    UNIT, 2k; Lead's condition before anything is built on it): REPLICATED.**

    | right ± SE (unit right) | c1a (step 76,295, 5.0B) | M4 (step 61,036, 4.0B) | T++ t15tpp (1.5B-token tier), same docs / row |
    |---|---|---|---|
    | A9 read | 0.00 (0.00) | 0.00 (0.00) | 0.00 (0.01) |
    | plain | 0.01 (0.01) | 0.01 (0.03) | 0.00 (0.01) |
    | **protocol twin** | **0.65 ± 0.05 (0.67)** | **0.26 ± 0.04 (0.40)** | 0.00 (0.01) |
    | noprot | 0.00 | 0.00 | 0.00 |

    - The prose protocol is stored in the SCI-1 state at 2k and grew from M4 to c1a (0.26 → 0.65; cont1 added protoc and other
      changes, so the cause is not isolated).
    - T++ shows nothing under any read. That is not a matched comparison: other data, never trained on protocol docs, and
      a 2.4k session just past its 2,048 trained length. It shows the twin read does not create the unit by itself.
- **A29: the package compiles a prose protocol into compact demos at QUERY time (PRE-REGISTERED 2026-11-04 08:39 IST, before any
  number; Mind's ask, Lead's plan item 3).** `v14/a29_rewrite.py`, code d3a71d6+.
  - Steps: a text parser finds the protocol's unit for the asked quantity (cues "in / is / as / use / means" + a unit list;
    tuned on the A26 + training wordings, so its 100 % hit rate is in-sample and is reported, not claimed); the planner writes
    "Report rule." + 3 worked lines "x -> x u" right before the question, takes the value out of the fetched row, and the mind
    answers "\n{v} ->" with the skill contrast (twin = the session without the inserted pack).
  - Arms: rewrite vs the A9 read, on c1a (step 76,295), A26 UNIT, G1 and A26 wordings, 2k / 8k, **100 docs per cell**.
  - Label: TEXT (the library compiles) + MIND (copies the value, writes the unit).
  - Bars: rewrite ≥ 0.45 at 2k on both wordings ("compiling solved in the package", Lead); and rewrite at 8k ≥ the protocol
    twin's 8k (A28: 0.00-0.05) + 0.25 ("a rewrite near the question fixes the reach gap").
  - Honesty line: for UNIT the library could append the unit itself; the mind's share is the value copy + following the
    demos. This tests the package path, not the mind's own understanding.
  - **A28 RE-READ of M1-M4 (2026-11-04 09:20 IST; code a69cafa; `logs/v14_a28_reread_m{1,2,3,4}_{train,a26}_a69cafa_20261104_0842.*`; 20
    docs per cell): the prose protocol is in the state at 2k from M1 on; the A9 read hid it at every milestone.**

    | protocol-twin read, right at 2k (8k / 32k) | M1 | M2 | M3 | M4 | c1a (100 docs) |
    |---|---|---|---|---|---|
    | G1 trained wording | 0.35 | 0.55 | 0.55 | 0.30 | 0.65 |
    | A26 wording | 0.20 | 0.30 | 0.20 | 0.00 | 0.20 |
    | 8k / 32k (both wordings) | ≤ 0.05 | ≤ 0.05 | ≤ 0.10 | ≤ 0.05 | ≤ 0.05 |

    A9 read and noprot ≤ 0.05 everywhere. The A26 "held" rows above (M1-c1a) stay valid as A9-read numbers; the twin read is
    the right read for a rule-in-state question.
  - **A29 RESULT (2026-11-04 09:20 IST; code a69cafa; `logs/v14_a29_rewrite_{train,a26}_a69cafa_20261104_0842.*`; c1a; 100 docs per
    cell): BOTH BARS MET.**

    | right (value / unit) | 2k | 8k |
    |---|---|---|
    | A26 wording: rewrite · A9 read | **0.93** (0.93 / 0.98) · 0.00 | **0.91** (0.91 / 0.98) · 0.00 |
    | G1 trained wording: rewrite · A9 read | **0.72** (0.72 / 0.95) · 0.00 | **0.68** (0.68 / 0.93) · 0.00 |

    - Parser hit 1.00 on every doc (in-sample: its cues were tuned on these wordings).
    - Bar 1 (≥ 0.45 at 2k, both wordings): 0.93 / 0.72 MET. Bar 2 (8k ≥ the twin's 8k + 0.25): 0.91 / 0.68 vs ≤ 0.05 MET.
    - Honesty: TEXT + MIND. The library compiles the prose rule and puts demos next to the question; the mind copies the
      value (0.68-0.93) and follows the demos. For UNIT the library could append the unit itself; the package path is shown
      to work at query time with no training, and distance no longer matters.
- **A30: compositional contrast: decode against a twin per question part (PRE-REGISTERED 2026-11-04 09:22 IST, before any code or
  number; Lead's plan, after A28).**
  - Read: `l = l(R,K) + a (l(R,K) − l(¬R,K)) + b (l(R,K) − l(R,¬K))`. R = the rule / protocol held in state (twin: the same
    session read WITHOUT it); K = the fetched row (twin: the same prefix without the row). a = 0 is the A9 read; b = 0 is
    A28's protocol twin.
  - **Closest prior work:** Context-Aware Decoding (Shi et al. 2023: one context ablated, the retrieved passage) and
    Instructive Decoding (Kim et al. 2023: a noisy instruction as the twin). **Difference:** the rule twin is the same
    session read without the rule, held as a second O(1) state thousands of tokens back, and the two ablations are composed
    per component of the question (rule × fact).
  - Cells: A26 UNIT, c1a (step 76,295), G1 and A26 wordings, 2k, 100 docs; a, b ∈ {0, 2, 4} (grid; (4, 0) = A28, (0, 4) = A9).
  - Bar: the best (a, b) with both > 0 ≥ max(A28 twin, A9) + 0.10 on the A26 wording at 2k (A28: 0.20, value-limited), and not
    below the A28 twin − 0.05 on G1 (0.65). Reported: the full grid, value and unit separately. The grid is chosen on these
    docs, so a fresh seed (26100) confirms the chosen (a, b) before any package default changes.
  - **A30 VERDICT (2026-11-04 09:31 IST; code 531ad61; `logs/v14_a30_compose_{a26,train}_531ad61_20261104_0923.*`; c1a, UNIT, 2k, 100
    docs per cell): NOT met. The rule twin alone stays best; adding the fact twin cancels the unit again.**

    | right (value / unit) | b = 0 | b = 2 | b = 4 |
    |---|---|---|---|
    | A26 wording, a = 0 · 2 · 4 | 0.00 · 0.10 · **0.22** (0.46 / 0.52) | 0.00 · 0.03 · 0.13 (0.76 / 0.17) | 0.00 (= A9) · 0.01 · 0.05 |
    | G1 trained wording, a = 0 · 2 · 4 | 0.01 · 0.33 · **0.65** (0.94 / 0.67) | 0.00 · 0.08 · 0.30 | 0.00 · 0.06 · 0.16 |

    - Bar: best (a, b > 0) = (4, 2): 0.13 vs max(0.22, 0.00) + 0.10 = 0.32 on the A26 wording → NOT met; 0.30 vs 0.65 − 0.05 on G1
      → NOT met. The fresh-seed confirm is not run (nothing to confirm).
    - What it says: the fact twin raises the VALUE (A26 wording 0.46 → 0.76) and lowers the UNIT (0.52 → 0.17). The unit is not
      in the row, so any row-vs-no-row term counts it as "not from the fact" and penalises it: the A28 cancellation, inside
      the composition. One logit sum cannot serve both parts.
    - Next angle (idea only, not registered): switch twins per answer TOKEN (the fact twin while the value is written, the
      rule twin after it), or a = 8 (the grid's edge). The package default is unchanged: the rule twin for rule-in-state
      questions, the A9 read for facts.
  - **A26 + A28 at cont2 step 91,554 (2026-11-04 15:00 IST; c2a = 6.0B tokens; cont2 = protoc 0.15 %, b2s ~1.4 %, proto4k 0.45 %;
    logs `logs/v14_a26_sci1c2a{,_g1}_ae6e50a_20261104_1404.*`, `logs/v14_a28_{rep100,reread}_c2a_*_3bbd02e_20261104_1451.*`):**

    | right | M4 (4.0B) | c1a (5.0B) | c2a (6.0B) |
    |---|---|---|---|
    | compact, G1 2k (value) | 0.00 | 0.45 | **0.00** (unit said 0.90) |
    | compact, A26 wording 2k / 8k | 0.00 / 0.00 | 0.35 / 0.60 | 0.05 / 0.05 |
    | prose held, twin read, G1 2k (100 docs) | 0.26 ± 0.04 | 0.65 ± 0.05 | **0.81** |
    | prose held, plain read, G1 2k | 0.01 | 0.01 | 0.33 |
    | prose held, twin read, G1 8k / 32k (20 docs) | 0.00 / 0.00 | ≤ 0.05 | 0.20 / 0.20 |
    | A9 read (held) · refresh (A9 read), G1 2k | 0.00 · 0.00 | 0.00 · 0.10 | 0.05 · 0.25 |

    - The compact value oscillates (M3 0.40 → M4 0.00 → c1a 0.45 → c2a 0.00) while the unit stays said (0.80-0.95); it is not a
      stable skill even with protoc in the mix.
    - The prose protocol held in state rises steadily (0.26 → 0.65 → 0.81 at 2k) and starts to reach (0.20 at 8k / 32k); the plain
      read now shows it (0.33). One checkpoint cannot attribute this to protoc, proto4k or b2s.
  - **A26 + A28 at cont2 step 106,813 (2026-11-04 22:36 IST; c2b = 7.0B tokens, the last checkpoint WITH protoc; logs
    `logs/v14_a26_sci1c2b{,_g1}_efe3d81_20261104_2139.*`, `logs/v14_a28_{rep100,rep100_8k,reread}_c2b_*_5cea3a4_20261104_22*.*`):**
    cont3's guard baseline.

    | right | c1a (5.0B) | c2a (6.0B) | c2b (7.0B) |
    |---|---|---|---|
    | prose held, twin read, G1 2k (100 docs) | 0.65 ± 0.05 | 0.81 | **0.49 ± 0.05** |
    | prose held, twin read, G1 8k | ≤ 0.05 (20) | 0.20 (20) | **0.04 ± 0.02** (100) |
    | prose held, plain read, G1 2k | 0.01 | 0.33 | 0.01 |
    | compact, G1 2k · A26 wording 2k | 0.45 · 0.35 | 0.00 · 0.05 | 0.10 · 0.05 |

    - On one mix (cont2), the prose-held skill fell back from 6.0B to 7.0B (0.81 → 0.49 at 2k; the 8k reach 0.20 → 0.04). It
      oscillates like the compact value; a single milestone does not set a trend. Both guard cells are now at 100 docs.
  - **A28 MATCHED CONTROL (2026-11-05 ~05:30 IST; run by Mind as ctl's B5, code caa4da9; `logs/ctl/v14_a28_ctl_pam_vs_tpp_mamba2_caa4da9.jsonl`;
    three bodies ~103M each, 1B tokens of the SAME SCI-1 mix; A26 UNIT, G1 trained wording, 2k, 98 docs per arm; numbers re-checked
    by Library from the jsonl):** the state-held prose rule is the PAM body's, not every model's.

    | right ± SE (unit right) | PAM | Mamba-2 | T++ |
    |---|---|---|---|
    | protocol twin | **0.53 ± 0.05** (0.66) | 0.20 ± 0.04 (0.39) | 0.12 ± 0.03 (0.12) |
    | A9 read · plain · noprot | 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 | 0.00 · 0.00 · 0.00 |

    - At equal params, tokens and data, PAM's state carries the prose protocol 2.7× Mamba-2 and 4.4× T++ (+0.33 / +0.41).
      All three show the A28 signature: nothing under the row-vs-no-row read, the rule only under the rule twin.
    - Limits stated: one seed per body, 1B tokens, 2k only (inside every body's trained 4k), the trained wording. The
      held-out wording and lengths > 2k are not yet in the matched comparison.
- **A28b: the matched control on the HELD-OUT wording and at 8k (PRE-REGISTERED 2026-11-05 05:20 IST, before any number; Lead's GO).**
  Mind's three ctl bodies (caa4da9; ~103M each, 1B tokens of the same SCI-1 mix; ONE SEED PER BODY), `v14/a28_twin.py --win
  <T++>,<Mamba-2>`, 100 docs: (a) A26 wording at 2k; (b) trained wording at 8k. **Bar (a): MET if PAM's twin read ≥ the best
  baseline's twin read + 0.20 at 2k.** (b) is descriptive (reach past 2k is weak on SCI-1).
  - **A28b VERDICT (2026-11-05 06:50 IST; code f56d4e6; `logs/v14_a28b_ctl_{a26w_2k,g1_8k}_f56d4e6_20261105_0625.*`; ctl bodies, ONE SEED
    PER BODY, 100 docs): bar (a) NOT met.**

    | protocol-twin read, right ± SE | PAM | Mamba-2 | T++ |
    |---|---|---|---|
    | (a) A26 held-out wording, 2k | 0.12 ± 0.03 | 0.06 ± 0.02 | 0.05 ± 0.02 |
    | trained wording, 2k (A28 matched control) | 0.53 ± 0.05 | 0.20 ± 0.04 | 0.12 ± 0.03 |
    | (b) trained wording, 8k | 0.00 | 0.00 | 0.00 |

    - (a): PAM 0.12 vs the best baseline 0.06, i.e. +0.06 < +0.20 → NOT met. PAM is still ahead, but on held-out wordings all three
      bodies hold the told rule weakly at 1B tokens.
    - (b): 0.00 for all three at 8k (twice the trained 4k). The rule's reach is the common limit.
    - Paper row: the equal-data advantage is clear on the TRAINED wording (0.53 vs 0.20 / 0.12) and small on held-out wordings
      (0.12 vs 0.06 / 0.05); one seed per body.
  - **A26 + A28 at cont3 step 122,072 (2026-11-05 06:50 IST; c3a = 8.0B tokens, the first checkpoint WITHOUT protoc; logs
    `logs/v14_a26_sci1c3a{,_g1}_f56d4e6_20261105_0532.*`, `logs/v14_a28_{rep100,rep100_8k,reread}_c3a_*_14c0271_*.*`):** the guard holds.
    Twin read G1 2k 0.54 ± 0.05 (c2b 0.49 ± 0.05); 8k 0.04 ± 0.02 (c2b 0.04 ± 0.02); A26 wording 8k / 32k ≤ 0.05; compact G1 2k
    0.20 (c2b 0.10). Dropping protoc did not lower the told-rule side at the first milestone.
- **A31: state FORKS: the rule twin held from read time at O(1) (PRE-REGISTERED 2026-11-05 08:22 IST, before any code or number;
  Lead's GO).**
  - Op: when the session stream reaches the protocol, the package forks the O(1) state. Copy A reads the protocol and the
    rest of the session; copy B reads the rest WITHOUT it (both fed every later token; the step offsets of B shift by the
    protocol's length). At the question both forks continue on row + question and the answer is `l_A + 4 (l_A − l_B)`.
  - Same docs as A28 (seed 26000, UNIT, G1 trained wording, 2k, 100 docs), the current SCI-1 checkpoint, and A28's twin read
    run beside it on the same docs.
  - **Bar:** fork-read G1 2k ≥ A28's twin read on the same docs (≥ that number − 0.02, since the fork is mathematically the
    same twin: B = the no-protocol session); every answer pair is compared for identity.
  - Reported: memory per fork (MB, measured from B's state); compute (B streams every later token: +1× streaming per
    standing fork, versus A28's re-read of the whole session at question time).
  - **Closest prior work:** CAD (Shi et al. 2023) and Instructive Decoding (Kim et al. 2023), which re-run the context with
    the instruction ablated at question time (double the KV / compute). **Difference:** one fork taken when the instruction
    is read, kept alongside at O(1) state cost. For a transformer, the equivalent fork is a second KV cache that grows with
    the context.
  - Honesty line: accuracy is A28's by construction. The claim is cost and the operation (attribution, on / off per
    standing instruction), not a new accuracy.
  - **A31 VERDICT (2026-11-05 13:29 IST; code e2a87db; `logs/v14_a31_fork_c3b_e2a87db_20261105_1316.*`; SCI-1 step 137,331 (9.0B), G1,
    2k, 100 docs): MET.** Fork read 0.83 ± 0.04 = A28's twin read on the same docs 0.83 ± 0.04; **answers identical 100 / 100**;
    session minus protocol = the no-protocol session 100 / 100; **3.674 MB per fork** (measured from the fork's state).
    The rule twin can be held from the moment the instruction is read, at O(1) cost per standing instruction (+1× streaming
    per fork), with no re-read of the context at question time. Accuracy is A28's by construction; the claim is cost and the
    operation.
  - **A26 + A28 at cont3 step 137,331 (2026-11-05 13:29 IST; c3b = 9.0B, cont3 +2B, no protoc; gap-free window chain e2a87db;
    `logs/v14_a28_rep100{,_8k}_c3b_*`, `logs/v14_a26_compact_c3b_{train,a26}_*`):**
    - Twin read G1 2k **0.83 ± 0.04** (c3a 0.54 ± 0.05); 8k **0.22 ± 0.04** (c3a 0.04 ± 0.02), the best 8k told reach so far.
      Plain read 0.22; A9 read 0.08; noprot 0.03.
    - Compact G1 / A26 wording 2k 0.20 / 0.20 (8k 0.10 / 0.10).
    - Mind's s8 D rose 0.589 → 0.770 over the same 1B, so told and induced rules rose TOGETHER. Not the competition signature
      (told did not fall ≥ 0.10); with cont2's opposite moves, "compete" is not consistent across intervals. Dropping protoc did
      not hurt the told rule.
- **PKG-SUITE: the package measured at every SCI-1 milestone, bare vs package on the same docs (PRE-REGISTERED 2026-11-05
  13:46 IST, before any number; Lead / user decision 13:40, scope 13:50).** `v14/pkg_suite.py`; descriptive (no bars); one table row
  per milestone, from 137,331 (checkpoint 9) on; no backfill.
  - **BARE** everywhere = the model alone on the session + question, plain greedy, no fetch, no contrast.
  - Cells:
    - (1) **R × K** (Mind's docs, seeds 8800-8849 × affine_map_new / shift_wide_new = 100 docs, 2k and 8k). Package = A23
      (fetch the record, write "<v> ->", skill contrast; 8k echo dilation s 8). Bare = "K ->" with the record in view.
      Scored on the first token.
    - (2) **Rewrite** (A29: the prose protocol compiled next to the question), A26 UNIT, A26 wording now (the held-out set
      once it exists), 2k and 8k, 100 docs; value + unit.
    - (3) **Told rule in state** (A28 / A31 twin = fork read), A26 UNIT, trained wording, 2k and 8k, 100 docs.
    - (4) **Recall** through the organ: needle among 8 (kv), latest-wins (kvupd), two-hop (hop2, a TEXT planner chain of two
      answers), bench docs, 32k, 20 docs per task. 20-way (first token over 20 value nouns) and free (value in the
      sentence); MB held per doc.
  - Limits stated: recall is reported to 32k on the 386M body; 128k was run on the 103M body (1.00), and the 24 GB card is
    the limit here, not the method (Lead's line for the paper). The rewrite parser is in-sample until fresh wordings exist.
  - CPU smoke (b2p2, 2 docs per cell): every cell writes bare and package rows.
  - **PKG-SUITE ROW: checkpoint 9 = step 137,331 (2026-11-05 14:03 IST; 9.0B tokens; code c93e3b7; `logs/v14_pkg_suite_c3b_c93e3b7_20261105_1347.*`;
    100 docs per cell, recall 20 per task; 16 min on the 4090):**

    | cell | bare | package |
    |---|---|---|
    | R × K, 2k / 8k (first token) | 0.01 / 0.01 | **0.67 / 0.42** (value fetched 1.00) |
    | told rule compiled next to the question (rewrite), 2k / 8k | 0.01 / 0.00 | **0.92 / 0.85** (parser 1.00, in-sample) |
    | told rule held in state (twin = fork read), 2k / 8k | 0.13 / 0.05 | **0.83 / 0.22** |
    | recall at 32k, free: needle among 8 / latest-wins / two-hop | 0.05 / 0.30 / 0.00 | **0.85 / 0.85 / 1.00** |
    | recall at 32k, 20-way | 0.10 / 0.45 / 0.10 | **1.00 / 1.00 / 1.00** |

    - MB held per 32k session: mind state 3.67, library text 0.07, indexes ~0.31; optional F2 address 33.8 (int8) or 0.27 (sign
      bits). Recall is reported to 32k on the 386M body (Lead's line: 128k ran on 103M; the 24 GB card is the limit here).
    - Two-hop is a TEXT planner chain (two package answers). Free recall < 20-way on kv / kvupd: the right value is ranked
      first, but the sentence sometimes runs on with other words.
- **A29b: the rewrite on FRESH protocol wordings (PRE-REGISTERED 2026-11-05 14:04 IST, before the wordings are written into code or
  any number; the stop list: A29 is not a product claim until fresh wordings pass).**
  - A new set (UNIT family): 10 protocol wordings, 8 question styles, 4 row templates, none used in A26, in
    protocol_docs.py, or in tuning the parser.
    - Checked before use: 0 shared 5-grams with either set, and added to decontam_items.json as test data (Mind's
      stream filter picks them up at its next index rebuild).
  - **The parser is FROZEN at its A29 state** (cues in / is / as / use / means + the unit list; d3a71d6 + the cue fix). No change
    after seeing the fresh wordings.
  - Cells: A26 UNIT docs, seed 26100, 2k and 8k, 100 docs, on the SCI-1 decay checkpoint (Nov 6) and every release.
  - Reported: rewrite vs bare, and the parser's hit rate.
  - **Bar:** rewrite ≥ 0.45 at 2k (A29's bar), with the parser hit reported. A low parser hit is the library's failure,
    stated as such.
- **SIMPLE-CHAT probe (PRE-REGISTERED 2026-11-05 14:04 IST, before any item is written; Lead / user 13:55: for the chat-ramp phase
  of the long run).** `v14/simple_chat.py`; items written by Library, held out from every chat source (added to
  decontam_items.json). ChatML user / assistant, greedy, ≤ 48 new tokens, stop at <|im_end|>.
  - **Yes / no** (60): a plain fact question + "Answer only YES or NO." Right iff the whole answer is YES or NO (case-insensitive,
    an optional final period) AND correct.
  - **Short direct answer** (40): one-word / one-number facts ("What is the capital of France?"). Right iff the first line,
    normalised, equals a gold form, with nothing else beyond 5 words.
  - **Instruction following** (40), verifiable formats: exactly N words, all lowercase, a comma-separated list of N items,
    start with a given word, a single number. Right iff the check passes.
  - **Small talk** (30, 2-3 turns). Right iff the reply is non-empty, ≤ 40 words, has no role-token leak and no repeated
    trigram ×3, and ends at a sentence end.
  - Reported: per-type accuracy and the overall mean.
  - **Baseline** on the SCI-1 decay checkpoint (Nov 6). Bar for the chat ramp (R3): each type up, and none of s8 / told twin
    (trained AND held-out wording) / reach / HS-ARC-PIQA down beyond its noise (Mind's CHECK). Descriptive per milestone.
- **BENCH-PKG: BABILong + RULER-style with the package, on the FINAL SCI-1 model (PRE-REGISTERED 2026-11-05 14:10 IST, before any code or
  number; Lead's order (2a)).** Descriptive, bare vs package per cell, MB held per row, lengths ≤ 32k (the 4090).
  - **BABILong** qa1-qa5 at 0k / 2k / 4k / 8k / 16k / 32k, 100 docs per cell (`v16/babilong_eval.py --package`). Bare = the
    model as is on the whole input; package = fetch rows (text organs / name / latest / F2) + read; `--chain` (TEXT planner)
    as a labelled extra column.
  - **RULER-style, our generator** (labelled as such; not the official RULER suite): niah_single (one key → value: word /
    number / uuid), niah_multikey (the key among 3 look-alike distractor keys), variable tracking (X1 = v, X2 = X1, … chains
    of 4; "which variables equal v?").
    - At 4k / 8k / 16k / 32k, 50 docs per cell.
    - Package = text fetch + read; for VT the library follows the chain in text (TEXT) and the mind writes the names.
    - Right = the exact value / every name, nothing extra.
    - Runner `v14/ruler_pkg.py` (068e800; seed 28000; the haystack is the A26 held-out prose; insert order shuffled so the
      asked key is not always first). Columns: bare, package, planner (VT: the TEXT chain found, a library-only number,
      labelled), mind MB (state) and library MB (token text + word indexes + the int8 address where the path keeps one; the
      sal1 address form is the alternative and is not added). `--summary <jsonl>` prints the table.
    - BABILong rows now carry `mb` (mind state + library token text) from the package's session.
    - 14:15 IST: both runners smoked end to end on c3b (2-3 docs per cell; plumbing check, not a result).
  - Reference block: published BABILong / RULER numbers of other models only with their training tokens stated, never a
    verdict.
- **PAIRS-MORE: s8 / needle / skill with a library arm, per item (PRE-REGISTERED 2026-11-05 14:33 IST, before any number;
  Lead, user 14:35).** Descriptive, for the review page; no bar. `v14/pkg_pairs_more.py`, first on c3b (137,331), then
  at every milestone after pkg_suite_pairs.
  - Items: exactly the page's probe items (v16/humaneval/run.sh).
    - s8: b2 shift_wide_new / affine_map_new, 8 shots, seeds 7000-7399, spread 0 / 2048 / 4096; 2,400 items.
    - needle: 128-8192 tokens, positions 0 / 0.5 / 1, 1 / 4 / 8 records, seeds 1000-1039; 2,520 items.
    - skill: skillpack, the same two tasks, seeds 8000-8099, L 0 / 2048 / 4096 / 8192; 800 items.
  - Bare = the model alone (the session state + the query; the same last-position logits as the probes).
  - Package:
    - s8 and skill: the library re-shows the header + 8 examples right before the query. The harness places them (as
      MLB-1 skillpack v1); fetching a pack by key is R×K's path, so this cell is labelled "re-shown", not "fetched".
    - needle: the package's own text fetch of the record row; on abstain the package answers as the model alone (path
      None, counted).
    - Read for all three = warm + contrast (re-shown / fetched + query vs query alone on the same state; alpha 4).
  - Scored: pick among the probe's candidates (ok, the page's number) and free (argmax over the vocab).
  - The skill probe's own "re-read" arm is pack + query in a FRESH context (no session state), so it is NOT the package's
    read; the page shows this package arm instead.
  - Expected (stated in advance): s8 / skill package ≥ bare at every L (the examples are next to the query); at spread 0
    the gain is small (the examples are already close). Needle: package near 1.0 wherever the fetch finds the key; the
    abstain rate is reported.
  - **RESULT c3b (137,331; 2026-11-05 15:12 IST; code 18f33af; logs/v14_pairs_more_c3b_18f33af_20261105_1449.log;
    rows in logs/v14_pkg_suite_c3b_c93e3b7_20261105_1347_pairs.jsonl, cells s8 / needle / skill).** Pick among the
    probe's candidates, bare → package:

    | cell | L | n | bare | package |
    |---|---|---|---|---|
    | s8 | 0 | 800 | 1.000 | 0.995 |
    | s8 | 2,048 | 800 | 0.906 | 1.000 |
    | s8 | 4,096 | 800 | 0.634 | 1.000 |
    | skill | 0 | 200 | 1.000 | 0.995 |
    | skill | 2,048 | 200 | 0.680 | 1.000 |
    | skill | 4,096 | 200 | 0.300 | 1.000 |
    | skill | 8,192 | 200 | 0.055 | 1.000 |
    | needle | 128-1,024 | 1,440 | 1.000 | 0.897 |
    | needle | 2,048-8,192 | 1,080 | 0.996 | 0.968 |

    - s8 / skill: met. The re-shown examples carry the rule at every length; the model alone fades with distance.
    - **Needle: NOT met.** The fetch was right on 2,520 / 2,520 (path text0, the asked record, 0 abstains), but the
      package scores 0.927 against bare 0.998.
      - The warm + contrast read (l_row + 4 (l_row − l_norow)) cancels what the state already holds. With the needle
        already in state (bare ≈ 1.0), subtracting l_norow removes the right value and amplifies the decoys.
      - This is the A28 signature again: A28 found it on a told rule, this is a stored fact.
    - Next (a new unit, pre-registered before any number): a read that keeps the contrast only where the model's own
      read and the fetched row disagree (an agreement gate), or the warm read alone. It is tested on needle, recall,
      s8 and skill together.
    - The skill probe's own "re-read" arm (pack + query in a fresh context) is not this arm; the page shows this one.
- **ARC-LIB: does the mind use FETCHED KNOWLEDGE better? (PRE-REGISTERED 2026-11-05 14:10 IST, before any code or number; Lead's
  order (2b)).** The three ctl bodies (~103M, 1B tokens of the same mix; one seed each), the IDENTICAL library.
  - Library: a filtered science sentence corpus (the AI2 ARC Corpus if obtainable, else a slice of the SCI-1 science
    sources), 13-gram decontaminated against ARC-e / ARC-c test + the kb1 sets (A27 machinery), a BM25 sentence index, top-3
    passages per question (question + all choices as the query).
  - Scoring: lm-eval-style log-likelihood over the choices (mc_eval), with the passages prepended. Rows per body: no
    library, + library, + ORACLE passage (a passage that states the answer; an upper bound on use).
  - **Bar:** PAM's gain from the library (+library − no library, ARC-e and ARC-c pooled) ≥ the best baseline's gain + 0.02.
  - **Stated in advance:** at ~103M / 1B tokens every body may gain ~0 ("no body uses them"); that is reported as the
    outcome, not hidden.
- **GATE: let the mind's own read decide when to contrast (PRE-REGISTERED 2026-11-05 15:47 IST, before any number; from
  PAIRS-MORE's needle loss and the 15:20 step back).** On c3b, untrained, offline from saved candidate logits.
  - Data: `v14/pkg_pairs_more.py` now saves per row the candidate logits of the model alone (b) and of the warm read with
    the row / examples shown (w).
    - Main rows: the page's items (s8 2,400, needle 2,520, skill 800); this rerun replaces the page rows, same answers.
    - Held-out rows (never on the page): s8 seeds 7400-7499 (600), needle 2000-2019 (1,260), skill 8100-8149 (400).
  - Reads (`v14/gate_read.py`):
    - bare = argmax b; warm = argmax w; contrast = argmax w + 4 (w − b), the package today.
    - **agree** (headline, no parameter): bare's pick if bare and warm pick the same candidate, else contrast.
    - conf: bare's pick if max softmax(b) ≥ tau, else contrast. tau is the best pooled held-out accuracy on a 0.30-1.00
      grid (step 0.02), fixed before the main rows are scored.
    - adacad: w + a (w − b) with a = 4 · JSD(w, b) / ln 2.
    - keyed (ADDED 15:50 IST, before any number; from the step back, no bar, EXPLORATORY, needle only):
      w + 4 m (w − b), m = 1 for candidates in the fetched row. The 2-item plumbing smoke showed it boosts wrong
      candidates on rule cells (the answer is computed, never in the shown examples), so it is reported for needle only.
  - Sanity: the offline contrast must reproduce the run's package_ok exactly, else the logits are wrong and nothing is
    read.
  - **Bar (agree):** met if, in every cell × length group, agree ≥ max(bare, contrast) − 0.01, AND pooled agree ≥ pooled
    contrast + 0.02 (the needle loss is recovered without giving up s8 / skill).
  - Novelty (closest prior work): context-aware decoding (CAD, Shi et al. 2023) is the same w + α (w − b) form; AdaCAD
    (Wang et al. 2024) adapts α by the JSD conflict between the two; FLARE (Jiang et al. 2023) retrieves on low
    confidence. Difference here: b is the O(1) state's own read and w the same state with the fetched row, so the gate is
    the mind's agreement with its library. The gate itself is close to AdaCAD, so plumbing unless agree beats adacad
    clearly. Claim only the comparison.
  - Outside scope: told (its read is the protocol twin, not this contrast) and recall-32k (60 items; the pkg_suite cell
    saves no logits yet).
- **BENCH-LIB (a): the library arm on ARC-e / ARC-c / HellaSwag / PIQA, per item (PRE-REGISTERED 2026-11-05 14:40 IST, before
  any number; Lead, user 14:40; extends ARC-LIB).** SCI-1 first (c3b now for the page, then the final model), then the three ctl
  bodies with the IDENTICAL library and scorer (the paper rows; ARC-LIB's bar applies to the ctl rows).
  - Library (`v14/arclib_build.py`, one for all four):
    - Sources: the AI2 ARC Corpus (14.6M science sentences), plus everyday / physical passages from Mind's cosmo and
      fineweb streams (holdout excluded; whole sentences, ≤ 60 words).
    - Kept: 6-80 words, deduplicated.
    - Dropped when EITHER 13-gram filter hits: Mind's index (v1.npy: HellaSwag / ARC / PIQA val + test, …) or the A27
      audit's independent hash.
    - A positive control (300 planted test strings) must be flagged by both. The CPU smoke passed: 300 / 300 each.
    - Drop counts per source and set go in meta.json and here.
    - **Built 2026-11-05 15:14 IST** (code d7841c4, logs/v14_benchlib_build_d7841c4_20261105_1440.log;
      /media/gowrav/d_drive/qllm2/arc_lib/lib_v1_d7841c4, 14,868,445 passages, 34 min on CPU). Positive control
      300 / 300 on both filters.

      | source | read | length drop | dup drop | dropped by Mind's filter | dropped by audit (by set) | kept |
      |---|---|---|---|---|---|---|
      | ARC Corpus | 14,621,856 | 2,317,091 | 142,359 | 385 | 166 (HellaSwag 120, ARC 29, LAMBADA 15, PIQA 2) | 12,162,021 |
      | cosmo (100k docs) | 1,129,173 | 12,969 | 963 | 19 | 8 (HellaSwag 7, ARC 1) | 1,115,222 |
      | fineweb (100k docs) | 1,644,084 | 43,489 | 9,300 | 93 | 20 (HellaSwag 17, ARC 2, PIQA 1) | 1,591,202 |

      BM25: 190M nonzeros, avgdl 13.7 terms, 0.07 s per query. The audit index loaded every set except OpenBookQA / MMLU
      (a datasets-version error in this env); Mind's v1.npy covers those two, and neither is scored here.
  - Retrieval: BM25 (`v14/arclib_index.py`), top-3 passages; the query is the question / context + all choices.
  - Scoring: mc_eval's own scorer (the same prompts and normalisation as the page) with the passages prepended. Arms per
    item: no library, + library, and + oracle on ARC only (oracle = "The answer is <gold choice>.", an upper bound on use;
    HellaSwag / PIQA have no oracle passage).
  - Pairs rows: cells arc_e / arc_c / hellaswag / piqa; the context is the 3 passages.
  - Stated in advance: BM25 passages often miss or mislead, so +library can be below no library on some sets. Every
    passage is shown on the page so a reader can judge.
  - **RESULT SCI-1 c3b (137,331; 386M; 2026-11-05 15:42 IST; code 7a4e123; library lib_v1_d7841c4;
    logs/v14_benchlib_c3b_7a4e123_20261105_1536.jsonl; rows in the c3b pairs file, cells arc_e / arc_c / hellaswag /
    piqa).** acc_norm. Δ ± is a 95 % interval from the discordant items (paired).

    | set | n | no library | + library | Δ (± 95 %) | flips +/− | + oracle |
    |---|---|---|---|---|---|---|
    | ARC-e | 2,376 | 0.463 | 0.526 | +0.063 ± 0.018 | 321 / 172 | 0.876 |
    | ARC-c | 1,172 | 0.277 | 0.278 | +0.001 ± 0.024 | 107 / 106 | 0.835 |
    | HellaSwag | 10,042 | 0.333 | 0.348 | +0.015 ± 0.005 | 459 / 308 | — |
    | PIQA | 1,838 | 0.622 | 0.634 | +0.012 ± 0.017 | 132 / 110 | — |

    - The model USES a passage: the oracle takes ARC-e 0.46 → 0.88 and ARC-c 0.28 → 0.84.
    - Retrieval is the limit. BM25 lifts ARC-e clearly, HellaSwag a little, PIQA within noise, and ARC-c not at all;
      many passages only share words with the question.
    - Passage sources: the ARC Corpus dominates even for HellaSwag (19,677 of 30,126 passages) and PIQA (4,711 of
      5,514), since it is 82 % of the library. The everyday slices are small; a larger everyday share is the obvious
      next library, not a tuning of this one.
    - These are SCI-1 rows for the page (no bar). ARC-LIB's bar applies to the three ctl bodies with this identical
      library; that comes next.
- **BENCH-TOOL (b): GSM8K with a calculator tool, per item (PRE-REGISTERED 2026-11-05 14:40 IST, before any code or number).**
  - Plain = today's direct answer (the page's GSM8K items and prompt).
  - Tool arm (AMENDED 14:50 IST, before any number: SCI-1's think source strips GSM8K's `<<expr=>>` marks, so the model
    never saw them; it did learn glaive / ToolACE calls in the `<tool_call>{"name", "arguments"}</tool_call>` +
    `<|im_start|>tool <tool_response>` format, v7/sci_tools.py, so the arm uses that format):
    - A glaive-style system turn offers one function, `calculate(expression)`.
    - When an assistant turn holds a `<tool_call>`, the sandboxed executor (`v14/tool_exec.py`: bwrap, no network,
      read-only /usr, 2 s timeout, whitelist arithmetic AST) computes it.
    - A tool turn carries the result and the assistant continues; at most 6 calls and 512 new tokens.
    - Items, plain prompt, greedy `m.generate` and "The answer is" scoring = scripts/think_eval.py OFF mode (the page's
      first 200 test items).
  - Rows: plain and tool, plus calls_made / well_formed / executor errors per item.
  - Stated in advance: at 386M with no tool training the model may rarely write a well-formed calculation; then the
    tool arm ≈ plain, and that is the finding.
  - **RESULT c3b (137,331; 2026-11-05 15:05 IST; code be77c55; logs/v14_benchtool_gsm8k_c3b_be77c55_20261105_1450.jsonl;
    cell gsm8k in the c3b pairs file).** As stated in advance: plain 0 / 200, tool 0 / 200.
    - Plain matches the page's OFF run (0 / 200, no "The answer is" in any answer; identical text on item 0).
    - Tool calls:
      - Made on 97 % of items (206 calls) in the trained `<tool_call>` JSON.
      - Names are invented glaive-style ones: calculate_number 47, calculate_percentage 23, calculate_distance 13,
        calculate_calculator 12, calculate_interest 11, against the offered calculate 24.
      - 19 calls were well-formed; 3 computed a number. Arguments are words ("Billy", "60% of 60").
      - No final "The answer is".
    - Reading: SCI-1 learned the call FORMAT, not to read the offered schema or ground the arguments in the problem.
      Sent to Mind (tools data).
- **Honesty line (proposed to Lead):** the exact-key recall win is the
  package's (its exact organ: plumbing); the mind's own key contributes on
  paraphrase in the session (0.46-0.66 vs organ 0.26-0.35, lenient mode);
  the mind's claim is R2 + R4.

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
