# v14 lab notebook — new neurons

Read part A before proposing anything. It is the condensed result of every
experiment file from v4 to v13_sempty (read 2026-10-29), so v14 never reruns
a closed question.

## Index (one row per v14 idea)

| id | idea | status | verdict | code at | log |
|---|---|---|---|---|---|
| B0 | find the wall: where does today's best (delta + echo) fail? | tool ready (`v14/microbench.py`), queued on the 4090 | — | — | — |
| B1 | rule induction in context (learning beyond recall, goal bar 3) | planned: uses the shared rule generator from the v13 session's learning probe | — | — | — |
| N1 | phase interference revived: v11 E3 on the lean body | coded + selftest PASS (`v14/units.py`), micro-bench queued with B0 | — | — | — |
| N2 | phase-modulated activation (magnitude sets phase) | planned (flag exists: `activation='phase_mod'`, complex arm) | — | — | — |
| N3 | formula-weight neuron | **folded into N2**: reduces to a gated bottleneck (GLU family), see N3 | — | — | — |
| N4 | living neuron: per-unit fast weight, written by surprise | design only | — | — | — |
| N5 | resonant binding: oscillator phases as binding tags | design only | — | — | — |
| N6 | operator memory: the state is an inferred transformation, not stored items | main candidate (literature check pending) | — | — | — |
| S1 | spatial neuron (learned positions, distance-gated) | notes only (`NOTES.md`) | — | — | — |

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
| Event / salience episodic memory was inert | V6 memory reframe: +35 % cost, PPL identical (56.55 vs 56.46) | a surprise-gated write must be load-bearing by construction (pands lesson 1, `NOTES.md` §1) |
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
| Spatial neuron field | `v14/NOTES.md` (2026-08-22) | notes only | **S1**, kill criteria in NOTES §5 |
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
| S1 | learned neuron positions, distance-gated, sparse by geometry | RSGN |

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

**Control body** for every LM arm: **M1** (KEEP at 600M, 2026-10-29,
session "Qllm v13"): `baseline_real_pm` + chrono + out_gate + delta + echo,
notebooks only in blocks 3, 7, 11, 15, MLP hidden 2352 (102.94M). 600M:
holdout 3.4906 (echo 3.4885), WikiText 69.4 (71.1), recall 0.615 (0.401),
1.8x faster (116K vs 64K tok/s on the 6000). Checkpoint (6000)
`checkpoints_v13_sempty/m1_m1echo_ccbf028/best_model.pt`. N1 on M1 swaps the
4 notebook blocks only (+28k params with `phase_read='phase'`). The B0
micro-bench keeps its own small all-notebook body (4 layers): it tests the
memory mechanism, not the layout.

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
for every arm. Launcher: `v14/run_b0.sh` (waits for free GPU memory).

**Output.** The smallest setting where the control drops below 0.5. That
setting becomes the memory bar for N1, N4, N5. If nothing fails up to a64
and bound attributes, memory is not the limiter and v14 focuses on LM
quality (N2, N3).

### B1 — rule induction (design; generator shared with the v13 session)

**Question.** Can a unit learn a rule from a few examples in the context
and apply it to a new input whose answer is nowhere in the context?

**Design.** Same small model and training loop as B0, with a fourth task
family drawn from the shared generator of the v13 session's learning probe
(`scripts/run_learning_probe.py`; requested 2026-10-29 as an importable
module so the micro-bench trains on the same tasks the probe scores on
checkpoints). Answers are never copied from the context, so echo's
succession memory cannot solve it by lookup. Bars are written when the
task list is fixed, before any B1 run.

**Why it matters for v14.** N4 (fast weights inside the unit) is a learner
by construction: fast-weight / test-time-training layers are known to do
in-context regression. B1, not B0, is its primary test.

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

### N5 — resonant binding (design only)

Units are oscillators (amplitude, phase, natural frequency, coupling). A
cheap sync step pulls co-occurring features to a shared phase (binding by
synchrony), and the phase selects where the binding is written among the
matrix states (N1's read in reverse). **HSB constraint:** never superpose
bindings in a vector. **Prior art:** AKOrN (Kuramoto neurons, vision, 2025),
Complex AutoEncoder / Rotating Features (object binding by phase, vision).
Not known in an O(1) recurrent LM. Micro-bench only, on B0's bound-attribute
case, after N1 has a verdict.

### N6 — operator memory (candidate, 2026-10-29)

**Idea.** Every known O(1) memory (SSM, linear attention, DeltaNet, PAM)
stores *items*: what was seen, as key → value associations, and recall
looks items up. Operator memory stores *operators*: how things change. Its
state is a transformation inferred from the context, and prediction applies
that transformation to a new input, so the answer never has to be in the
context. That is rule application (goal bar 3), not lookup.

**Sketch in phase space.** Represent tokens as complex vectors. A relation
between an input and an output is a rotation per channel,
`Δφ = arg(v) − arg(k)`. From examples ("France → Paris, Japan → Tokyo") the
state accumulates a confidence-weighted average of these rotations: a point
on a torus, O(d) per head, O(1) per token. A new input is answered as
`k_new · e^{iΔφ}`. Two relations compose by adding phases, so composition
is built in. Open design questions: which positions play "input" and
"output" (learned role gates), several operators at once (a small set,
selected by the input), and magnitude changes as well as rotations.

**Prior art to check before any claim:** RotatE (relations as complex
rotations, but static knowledge-graph embeddings with no in-context
inference), tensor-product / holographic representations, analogy by
vector offsets (word2vec), meta-learning / in-context learning theory.
No recurrent LM whose state *is* an inferred operator is known to us.

**First test:** B1 rule tasks with answers absent from the context. Bar:
beat the echo / M1 small bodies on rule application at equal size and state.

### S1 — spatial neuron

See `NOTES.md`. Kept as a candidate. Its own kill criteria (gate decorative,
or speed > 1.5× the matched transformer at 1M) apply.

---

## Order and machines

1. **B0 + N1 micro-bench** in one run on the 4090 (`v14/run_b0.sh`, arms
   delta, echo, n1sum, n1gate, n1phase; ~20 min per arm). It waits until the
   GPU has ≥ 14 GB free (T1's local check, other session, holds it now).
2. **N1 LM check** on the 4090 if the micro-bench passes (needs a v14 hook in
   the v13_sempty trainer: not built yet).
3. **N2** LM check (complex arm) on the 4090.
4. Anything that passes stage 3 queues on the 6000 behind the v13_sempty
   chain (M2, then T1 600M).
