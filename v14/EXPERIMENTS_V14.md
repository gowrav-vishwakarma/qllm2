# v14 lab notebook — new neurons

Read part A before proposing anything. It is the condensed result of every
experiment file from v4 to v13_sempty (read 2026-10-29), so v14 never reruns
a closed question.

## Index (one row per v14 idea)

| id | idea | status | verdict | code at | log |
|---|---|---|---|---|---|
| B0 | find the wall: where does today's best (delta + echo) fail? | planned | — | — | — |
| N1 | phase interference revived: v11 E3 on the lean body | planned | — | — | — |
| N2 | phase-modulated activation (magnitude sets phase) | planned | — | — | — |
| N3 | formula-weight neuron: weight = context-chosen interference of K bases | planned | — | — | — |
| N4 | living neuron: per-unit fast weight, written by surprise | design only | — | — | — |
| N5 | resonant binding: oscillator phases as binding tags | design only | — | — | — |
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

**Control body** for every arm: the current best v13_sempty preset
(`baseline_real_pm` + chrono + out_gate + delta + echo; `e1g_echo_a0fab65`).
If M1 or M2 wins on the 6000 (verdicts due 2026-10-29 evening IST), the
control switches to the winner **before** any v14 run starts, and this line
is updated.

**Stages for every N-arm:**
1. selftest: parallel == recurrent (or decode) to round-off; layout guard clean.
2. micro-bench (4090, minutes): `v13_sempty/tmp/recall_microbench.py`
   extended with the B0 cases.
3. LM same-batch check (4090): train loss vs control over steps 1750–2250,
   then **150M holdout** vs control (echo at 150M: 3.9516). Holdout outranks train loss.
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

**Output.** The smallest setting where the control drops below 0.5. That
setting becomes the memory bar for N1, N4, N5. If nothing fails up to a64
and bound attributes, memory is not the limiter and v14 focuses on LM
quality (N2, N3).

### N1 — phase interference revived (v11 E3 on the lean body)

**Why.** The only architecture change that beat the v7 baseline (−1.11 PPL),
and the one mechanism that is uniquely PAM. Never retested after the lean rewrite.

**Mechanism.** K = 2 notebooks per head, each with its own decay bias.
Read `y = Re( Σ_k e^{iφ_k(x)} · S_k q )`, `φ_k(x) = W_φ x`, zero-init, so the
model starts as an equal sum. In the real arm, the phase acts on channel
pairs (a 2-D rotation, as RoPE does), so no complex arm is needed. Write
path unchanged (delta into each S_k).

**Arms.** control; `n_states=2` with a plain sum (isolates the state count);
`n_states=2` + phase read (N1). State is 2× in both K=2 arms.

**Bars.** Micro-bench: N1 ≥ plain-sum + 0.20 on the B0 wall case.
LM: 150M holdout ≤ control − 0.02 nats AND ≤ plain-sum − 0.01. Step time ≤ 1.35× control.
**Kill:** holdout ≥ plain-sum (the phase adds nothing).

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

### N3 — formula-weight neuron ("a weight is a formula, not a number")

**Why.** The user's idea, 2026-10-29: a neuron is more than weight and bias.
Here each weight is K interfering parts, and the input picks how they combine,
so the same neuron computes a different function in a different context.

**Mechanism.** In the CGU/MLP up-projection:
`W(x) = Σ_k c_k(x) · U_k V_kᵀ`, `c_k(x) = e^{iφ_k(x)}` (complex arm) or
`c_k(x) = cos φ_k(x)` on channel pairs (real arm). Mixing coefficients can be
negative or cancel, unlike CondConv / DynamicConv, which mix with softmax
weights. Rank chosen so total params = control. Computed as
`Σ_k c_k(x) · U_k (V_kᵀ x)`, so no per-token weight matrix is materialized.

**Prior art.** CondConv (Yang 2019), DynamicConv (Chen 2020), hypernetworks,
mixture-of-LoRA. The new part is phase (cancellation) mixing inside a
recurrent O(1) LM.

**Bars.** 150M holdout ≤ control − 0.02 nats, step time ≤ 1.3×.
**Kill:** φ collapses to a constant (all tokens use the same mix) or holdout ≥ control.

### N4 — living neuron (design only; after B0, N1–N3)

Each MLP unit keeps a small fast weight written while reading (delta rule),
with write strength set by echo's prediction error (surprise). Memory moves
into every unit instead of a separate notebook.
**Prior art is close:** fast weights (Ba 2016), TTT-MLP (Sun 2024), Titans
(2025). The new parts would be the echo-surprise gain and the complex/phase
form. Run only if B0 shows a memory wall that N1 does not move.
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

### S1 — spatial neuron

See `NOTES.md`. Kept as a candidate. Its own kill criteria (gate decorative,
or speed > 1.5× the matched transformer at 1M) apply.

---

## Order and machines

1. **B0** on the 4090 (micro-bench only, minutes) once T1's local check
   (other session, `t1loc_t1_ea002e6`) frees the GPU, or alongside it if
   memory allows and T1's speed numbers are already taken.
2. **N1, N3** selftests on CPU while waiting; then micro-bench + LM check on
   the 4090.
3. **N2** LM check (complex arm) on the 4090.
4. Anything that passes stage 3 queues on the 6000 behind the v13_sempty
   chain (M2, then T1 600M).
