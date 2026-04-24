# V8 — Constrained Latent Memory + Adaptive Compute (Quantum-Inspired, Classically Tested)

> **TL;DR** — V8 splits a normal LM into three machines: a **grammar engine**
> (QPAM/V7 backbone), a **constrained latent memory** (low-rank projector +
> top-k effect bank), and an **adaptive reasoning loop** that iterates the
> memory step against the backbone state and halts when it has enough signal.
> The design is *inspired* by operational quantum logic — Sasaki updates,
> orthocomplement readouts, effect algebras — but the contribution we are
> testing for is the *classical inductive bias* (rank-bounded subspace
> memory + ACT-style halting), not a claim that quantum logic itself is the
> source of any win. The original "operational quantum logic is what helps"
> claim is treated as an **ablation hypothesis**, not a default.
>
> See [`v8/AUDIT_V8.md`](AUDIT_V8.md) for why the project was reframed,
> [`v8/EXPERIMENTS_V8.md`](EXPERIMENTS_V8.md) for the running results, and
> [`.cursor/plans/v8_classical_rethink_44e4a93c.plan.md`](../.cursor/plans/v8_classical_rethink_44e4a93c.plan.md)
> for the full classical-rethink plan.

---

## 0. What changed in v8.1 (read this first)

After Stage A.5 came in with `gamma = 0.000` for the entire smoke run and a
12× wallclock cost for a 3% PPL improvement, three independent reviews
(GPT-5.4, Gemini 3.1 Pro, Opus 4.7) converged on the same conclusion: the
flagship "operational quantum logic" claim was **structurally untestable**
in the original code, and several of the headline metrics were either
geometrically pinned (γ ≡ 0 from a sharp-projector identity) or measured the
wrong thing (the `quantale_off` ablation was a residual blend, not an order
test). See [`AUDIT_V8.md`](AUDIT_V8.md) for the full enumeration.

The current code now treats V8 as **two stacked classical hypotheses**, each
of which is testable and either of which would already be a useful result:

1. **Constrained latent memory** — restricting the per-token working state
   to a rank-`r` orthonormal subspace, then projecting against top-`k`
   selected effects from a learnable bank, is a useful inductive bias for
   LMs at small data / small model scale.
2. **Adaptive iteration on top of memory** — running the memory step a
   variable number of times per token, with an empirical halt signal,
   improves perplexity at matched FLOPs.

The "operational quantum logic" reading (γ measures non-commutativity, the
loop encodes a quantale, etc.) is an **upper bound**: if the discriminator
suite (§G of the rethink plan) shows that order matters, that γ becomes
non-trivial under unsharp targets, and that complex weights beat real ones
at matched parameters, then the stronger reading earns its claim. If those
fail, V8 still has the two classical wins above and a clean architecture to
ship; the framing just gets renamed.

The old plan, the original ablation matrix, and the explanatory math below
are kept verbatim because they are the historical motivation. The honest
*current* design rationale is in §0 (this section), §11 (the reframe), and
the discriminator suite presets in [`config.py`](config.py).

---

## 0.2 What changed in v8.2 (alpha/gamma unfreeze, 2026-04-24)

The first end-to-end medium run on WikiText-103
([`logs/v8/e2e_medium_reasoning_wikitext103_20260424_162331_4450ff0/v8_e2e_medium_reasoning_wikitext103.log`](../logs/v8/e2e_medium_reasoning_wikitext103_20260424_162331_4450ff0/v8_e2e_medium_reasoning_wikitext103.log))
hit the §9.3 hard-kill criterion: `alpha=0.001 beta=0.998 gamma=0.001` and
`halt(yes/no/cont)=0.02/0.96/0.02` were pinned for the entire 1,200+ steps
that ran before we killed it. The numbers match the closed-form prediction
when

- `psi` is unit-norm (`renormalize_psi=True`),
- `u(psi) = OrthoHalt.target_mlp(psi)` is essentially random vs `psi` -> `|u^H psi|^2 ~= 1/d = 1/384 ~= 0.0026`,
- `target_gate=0` -> `sigmoid(0)=0.5` splits the inner-product mass 50/50 between alpha and gamma,
- `cls_head.weight = 4*I` makes `beta=0.997` decisive -> halt-no fires every step -> later iterations contribute ~0 weight -> almost no gradient back to `target_mlp` or the bank,
- `infonce_weight=0` in the e2e preset (only `stageB_infonce_on` set it) -> no auxiliary objective ever asked `u_m` or `u(psi)` to align with `psi`,
- The runtime schedule ramped `ponder_lambda` up over the run, *discouraging* the loop from iterating when (with the fixes below) it would finally have signal worth iterating on.

In short: alpha/gamma had no gradient path off the `1/d` noise floor.

The v8.2 fix bundle (see [`.cursor/plans/unfreeze_qlc_alpha_gamma_2132eeda.plan.md`](../.cursor/plans/unfreeze_qlc_alpha_gamma_2132eeda.plan.md))
landed five coordinated changes:

1. **`OrthoHalt.target_gate` init** (`v8/qlc/halt.py`): zeros -> 1.5 so `sigmoid(1.5) ~= 0.82` routes most of `|u^H psi|^2` into alpha, with the gate deficit (~18%) on gamma. Without growth in `|u^H psi|^2` itself, gamma still sits at noise -- which is why we also need (3).
2. **`OrthoHalt.cls_head` init** (`v8/qlc/halt.py`): `eye*4`, zero bias -> `eye*1`, bias `[0, 0, 2]`. Default softmax distribution becomes roughly `[0.10, 0.10, 0.80]` (continue-biased) instead of collapsing to halt-no on iter 1, so the loop actually runs all `t_max` iterations from step 0 and gradient flows through every step. The matching re-init in `V8LM._init_weights` was updated identically.
3. **New `target_alignment_weight` aux** (`QLCConfig`, `QuantumLogicCore.forward`, `V8LM.forward`): adds `target_alignment_weight * (1 - mean(|u^H psi|^2)).clamp_min(0)` to `aux_loss`. Default `0.0` for backwards compat; the e2e_*_reasoning presets set it to `0.05`. This is the only term that directly rewards `u(psi)` aligning with `psi`.
4. **`infonce_weight=0.05, infonce_every=4`** baked into both `e2e_medium_reasoning` and `e2e_tiny_reasoning` presets, activating the entity-cloze pipeline that was already wired in `v8/train.py` but never enabled in the e2e workflow.
5. **Schedule retune** (`QLCRuntimeSchedule.ponder_lambda_phases`): `(0.0, 0.005, 0.01)` -> `(0.0, 0.002, 0.005)`. Smaller cap so the late phase doesn't undo the loop engagement we now have.

Behavioral expectation under the fix bundle (smoke gate before relaunching medium):

- `align` (the new diagnostic column = mean `|u^H psi|^2`) rises off the `1/d ~= 0.003` noise floor, target `>= 0.05` within an epoch and `>= 0.2` by mid-training.
- `mean_iter > 1.5` once `t_max=4` (i.e. the loop is actually pondering).
- `halt(yes/no/cont)` no longer pinned at `0.02/0.96/0.02`; none of the three should be `< 0.05` at convergence.
- `gamma >= 0.01` early, `>= 0.05` late.

The full diagnosis + math chain is in [`AUDIT_V8.md`](AUDIT_V8.md) §6 and the
empirical write-up of the 4450ff0 negative result is in
[`EXPERIMENTS_V8.md`](EXPERIMENTS_V8.md) §10.

---

## 1. The big picture (10-year-old version)

Imagine you ask a normal language model:

> "Marie Curie was born in ___."

A regular Transformer takes the words, runs them through one big network in
one shot, and immediately spits out a probability distribution over the next
word. There's no "let me think for a moment" — every question gets the same
fixed amount of compute.

V8 tries to act more like a person. It separates three jobs:

| Job              | Who does it                | Analogy                                          |
|------------------|----------------------------|--------------------------------------------------|
| **Grammar**      | QPAM backbone (V6/V7)      | The part of you that knows how sentences work    |
| **Memory**       | Effect Algebra Bank        | The shelf of facts you've learned over the years |
| **Reasoning**    | Quantum-Logic Core (QLC)   | The little voice that says "wait... is that right?" |

When V8 sees "Marie Curie was born in ___":

1. The **grammar engine** turns the words into a hidden state — a high-dimensional
   complex vector that represents "the meaning so far".
2. That state goes into the **reasoning loop**. The loop **probes the memory**
   for relevant facts (effects that score high against the current state).
3. The selected facts are bundled into a tiny "subspace of currently relevant
   truth" (a projector). The model **updates its state by projecting it onto
   that subspace** — this is the Sasaki update borrowed from quantum logic.
4. A **halt head** measures three numbers — `α` (yes), `β` (no), `γ` (still
   confused) — and decides whether to stop or take another reasoning step.
5. After a few iterations, the final state goes to the LM head and predicts
   the next word.

That last step — *iterating in latent space until you're confident* — is the
core idea V8 is testing.

---

## 2. Why three modules? Why not one big stack?

Because **grammar, memory, and reasoning are different kinds of work** and
mixing them into one giant pile of weights is what made earlier memory
experiments fail.

### Grammar is just for *expressing* — not for *knowing*

A Transformer or QPAM block is great at things like
*"verb agreement"*, *"this is the start of a list"*, *"this pronoun refers to
that name"*. These are local syntactic and semantic patterns over a few
tokens. Real LMs solve them well by stacking attention/recurrence layers.

But this kind of layer does **not** naturally store discrete facts like
*"Marie Curie was Polish"*. Facts get smeared across millions of weights with
no clean address. You can train a giant model on the whole internet and it
will *seem* to remember facts because there are so many parameters and so much
data — but you can't point at *where* the fact lives, you can't update one fact
without retraining, and you can't reason over a chain of facts step-by-step.

### Memory should be *addressable* — like a library

V8's `EffectAlgebraBank` ([`v8/qlc/effect_bank.py`](qlc/effect_bank.py)) is a
flat shelf of `M` learnable rank-1 effects:

```
E_m = sigmoid(s_m) · (u_m u_m†)
```

Each effect has an "address" `u_m` (a complex direction in state space) and a
"value" `w_m`. When the model wants to look something up, it `probe`s the
bank with the current state and picks the top-k effects that score highest.
Those top-k addresses are turned into an orthonormal basis and shoved into
the working memory.

This is **explicit, named, replaceable storage** — the opposite of a smear
over the backbone weights.

### Reasoning is *iterative* — and needs an if/else

A single forward pass is "I'll guess in one shot". Real reasoning is
"let me check this against what I know, ask a follow-up, check again, then
answer". That's why V8 has a loop (`T_max` iterations, default 4) instead of
just one read.

For the loop to be smart, the halt head needs more than a binary "stop / keep
going" signal. It needs to know:

- **`α` = "this matches what I think the answer is"** (project state onto
  target subspace, measure mass)
- **`β` = "this clearly *doesn't* match"** (project onto orthogonal complement)
- **`γ` = "the question and the answer don't even commute — I'm confused"**

`γ` is the magic one. In normal real-valued math, `γ` is always 0. But because
V8 uses **complex** state vectors and complex projectors, `γ` can be
positive whenever the state and the target effect don't commute. That's the
"native if/else" the design is built around. See
[`v8/qlc/halt.py`](qlc/halt.py) for the math.

---

## 3. What we actually built

```
Tokens
  │
  ▼
ComplexEmbed              ─────────────────────────────┐
  │                                                    │
  ▼                                                    │
[ V7Block × N ]    ◄── grammar engine (QPAM)           │
  │                  - learns syntax / local meaning   │
  │                  - trained jointly with QLC (e2e)  │
  ▼                                                    │
ComplexNorm                                            │
  │                                                    │
  ▼                                                    │
QuantumLogicCore  ◄── the new piece                    │
   ├─ Probe       → score state against EffectBank     │
   ├─ Build Π_F   → top-k effects → orthonormal basis  │
   ├─ Sasaki      → ψ_{i+1} = Π_F · ψ_i  (project!)    │
   └─ OrthoHalt   → read (α, β, γ); halt or loop       │
  │                                                    │
  ▼                                                    │
Tied complex LM head ──────────────────────────────────┘
  │
  ▼
Logits → next-token probability
```

### Files (all in [`v8/`](.))

| File                                                        | What it owns                                                      |
|-------------------------------------------------------------|-------------------------------------------------------------------|
| [`v8/config.py`](config.py)                                 | `V8Config`, `QLCConfig`, the `e2e_*_reasoning` presets, and the historical `stageB_*` ablation presets |
| [`v8/model.py`](model.py)                                   | `V8LM`: glues the V7 backbone + QLC + tied complex LM head        |
| [`v8/qlc/projector.py`](qlc/projector.py)                   | `SasakiProjectionMemory` — the rank-`r` projector working memory  |
| [`v8/qlc/effect_bank.py`](qlc/effect_bank.py)               | `EffectAlgebraBank` — the M-slot fact library                     |
| [`v8/qlc/halt.py`](qlc/halt.py)                             | `OrthoHalt` ((α, β, γ) reader) and `MLPHalt` (ablation)           |
| [`v8/qlc/reason_loop.py`](qlc/reason_loop.py)               | `QuantumLogicCore` — the iteration orchestrator                   |
| [`v8/data.py`](data.py)                                     | reuses V7 loaders + adds an entity-cloze task for InfoNCE         |
| [`v8/train.py`](train.py)                                   | Stage-aware trainer (A / B / C, KL anchor, ponder cost, diagnostics) |
| [`v8/qlc/tests/`](qlc/tests/)                               | 40 CPU-runnable unit tests; pass before any GPU run               |

### Glossary of the new words

- **Sasaki projection / Sasaki update**: a non-commutative way to revise a
  state when we learn that a new proposition is true. In quantum logic this
  is the unique measurement-consistent revision rule.
- **Orthocomplement**: the geometric "not". For a subspace `V`, its
  orthocomplement `V⊥` is everything orthogonal to it. `α` measures mass in
  `V`; `β` measures mass in `V⊥`; `γ` measures everything left over (which is
  zero in a classical world but non-zero in a complex one).
- **Effect algebra**: a math structure where each "effect" is between 0 and 1
  (think "a fuzzy yes/no"). Our bank guarantees this with `sigmoid(s_m)`.
- **Quantale composition**: applying Sasaki updates `P` then `Q` is **not** the
  same as `Q` then `P`. That ordering is potentially what lets multi-step
  reasoning encode genuinely different things — V8-F (`quantale_off`) is the
  ablation that tests whether this ordering actually matters.
- **ACT-style pondering**: the reasoning loop emits a "halt now" probability
  per step; we add a `ponder_lambda · E[steps]` term to the loss so the model
  learns to think *just enough*, not forever.

---

## 4. Why we think it will work (and where it might not)

### What we're betting on

1. **Constrained memory > free-form memory.** Forcing the working state to be
   an orthonormal-rank-`r` projector means it has *exactly* `r` degrees of
   freedom in a way the optimizer can use. The V6 analysis script showed
   QPAM's free-form complex state was hitting an effective rank of ~10/64 —
   most of its capacity was wasted. By building rank in by construction we
   waste less.
2. **Explicit, addressable facts.** The `EffectAlgebraBank` is a place we can
   point at and say "this is the model's knowledge". You can later inspect it,
   prune it, or even hot-swap effects without retraining the backbone.
3. **An algebraic if/else.** `(α, β, γ)` is *measured*, not learned by an MLP
   guessing at confidence. If reasoning genuinely needs an "I'm not sure yet"
   signal, the algebra is there for free — and only complex weights can
   provide it.
4. **Clean experiments (historical motivation).** The original three-stage
   plan would have produced a frozen Stage A backbone for Stage B to test
   against, isolating QLC effects from backbone co-adaptation. We've since
   moved to a single end-to-end run (§5) and instead use the
   `unsharp_target=True` γ signal plus the QLC ramp schedule to attribute
   reasoning behavior, accepting some entanglement with the backbone in
   exchange for one trained model rather than three.

### Where it could quietly fail

- `γ` could stay 0 forever — meaning the if/else algebra never engages and
  the win comes purely from the rank constraint. The historical Stage A.5
  smoke showed exactly this (γ ≡ 0 with `unsharp_target=False`), which is
  why the canonical e2e preset (§5) sets `unsharp_target=True`. If γ stays
  at 0 even there, the (α, β, γ) head is decorative.
- The bank could just memorize the training set (V6 §5.5 redux). We mitigate
  with top-k routing and InfoNCE on synthetic entity cloze.
- Re-orthonormalizing the projector basis every iteration could be unstable
  in autograd. The unit tests in
  [`v8/qlc/tests/test_projector.py`](qlc/tests/test_projector.py) check this
  directly; QR refresh is configurable.
- Sasaki updates are 2× more compute per step than QPAM, and we run up to
  T_max=4 of them. Even if quality wins, throughput hurts. The §5 medium
  e2e run will tell us whether the cost/benefit is worth it.

---

## 5. End-to-end single run (canonical V8 workflow)

V8 trains in **one continuous run** from random init. The original Stage A
/ A.5 / B / C plan and its launchers were retired on 2026-04-24; the
canonical workflow is now the e2e single-run preset. The motivation,
acceptance criteria, and ramp schedule are in
[`v8/EXPERIMENTS_V8.md`](EXPERIMENTS_V8.md) §9 and the v2 plan
([`.cursor/plans/single-run-v8-training_12d5e87a.plan.md`](../.cursor/plans/single-run-v8-training_12d5e87a.plan.md)).

Launchers (both go through `v8/train.py`):

| Script                                                                       | Purpose                                                                       |
|------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`scripts/run_v8_e2e_smoke.sh`](../scripts/run_v8_e2e_smoke.sh)             | TinyStories sanity gate using `e2e_tiny_reasoning`. Confirms γ > 0 wiring.    |
| [`scripts/run_v8_e2e_medium.sh`](../scripts/run_v8_e2e_medium.sh)           | WikiText-103 main run using `e2e_medium_reasoning`, 10 epochs by default.     |

Key configuration (set in the `e2e_*_reasoning` presets in
[`v8/config.py`](config.py)):

- `freeze_backbone=False` — backbone learns jointly with the QLC from step 0.
- `qlc.unsharp_target=True` — required for non-trivial γ
  (see [`AUDIT_V8.md`](AUDIT_V8.md) §1). Necessary but **not sufficient**
  on its own: see §0.2 and `AUDIT_V8.md` §6 for the alpha/gamma noise-floor
  pinning that motivated v8.2.
- `qlc.target_alignment_weight=0.05` *(v8.2)* — pulls `OrthoHalt`'s
  `target_mlp(psi)` toward `psi` so `|u^H psi|^2` actually grows above the
  `1/d` noise floor and the alpha/gamma channels carry signal. Set to
  `0.0` for legacy / ablation runs.
- `qlc.infonce_weight=0.05`, `qlc.infonce_every=4` *(v8.2)* — entity-cloze
  contrastive auxiliary on the bank, run every 4 LM steps. Activates the
  pipeline already wired in `v8/train.py` (`_compute_infonce_loss`,
  `make_entity_cloze_loaders`).
- `qlc.out_scale_init=0.05`, `out_scale_learnable=True` — soft warmup on the
  QLC residual contribution.
- `kl_anchor_weight=0.0` — **must** be 0 with random init (no Stage A
  reference logits exist; `backbone_logits()` would otherwise anchor to
  the model's own untrained logits).

Optional runtime-safe schedule (`--qlc_schedule`) ramps `t_max` (2 → 3 → 4)
and `ponder_lambda` (0 → 0.002 → 0.005, *lowered in v8.2* from
`(0, 0.005, 0.01)`) across thirds of the run. The
schedule mutates only attributes read fresh on every forward pass; anything
that would require rebuilding modules (`halt_mode`, `out_scale_learnable`,
`use_complex`, `rank`, `bank_size`) is intentionally not schedulable and
must be set in the preset.

Acceptance criteria (`mean_gamma > 0` sustained, halt distribution not
collapsed, `mean_iter > 1` once `t_max=4`, Val PPL not catastrophic vs the
V7-equivalent passthrough at matched compute) live in
[`v8/EXPERIMENTS_V8.md`](EXPERIMENTS_V8.md) §9.2.

**Optional one-off ablations.** The historical `stageB_*` / `smoke_tiny_*`
presets remain in `v8/config.py` and can be invoked manually with
`python -m v8.train --preset <name>` if you want to reproduce a specific
ablation row. They are no longer wrapped in a launcher and do not assume
any Stage A backbone checkpoint.

---

## 6. What we know so far (smoke results)

**Historical Stage A.5 smoke (TinyStories, 2026-04-23, retired launcher):**
QLC `r=4, T_max=2` beat the V7-equivalent passthrough by **5.55 PPL**
(178.95 → 173.40) on `_backbone_tiny()`. γ was 0.000 throughout because
`unsharp_target` defaulted to off — the win came from rank-r Sasaki
retrieval acting as a regularizer, not from the (α, β, γ) head.

**E2e smoke (TinyStories, 2026-04-24):** `e2e_tiny_reasoning` with
`unsharp_target=True` and the runtime QLC schedule on. Within the first
epoch (52 batches × 16 × 2k samples) we observe γ = 0.008 → 0.010,
`halt(yes/no/cont) = 0.02 / 0.96 / 0.02` (alive but no-dominated), and the
schedule transitions firing at the planned step boundaries. PPL is
meaningless on this short run — the smoke is purely a wiring gate, which
passed.

The medium WikiText-103 run via
[`scripts/run_v8_e2e_medium.sh`](../scripts/run_v8_e2e_medium.sh) is the
definitive test against the §9.2 acceptance criteria.

---

## 7. The bar for "V8 wins"

- **Win**: `V8-E-joint ≤ 23.13 PPL` on WikiText-103, 10 epochs, ≤105M params
  (beats both QPAM 29.95 and the Transformer B=6 baseline 23.13). Bonus:
  factuality probe accuracy +5 pts over QPAM.
- **Soft win**: `V8-D-T4 ≥ V8-B by ≥ 3 PPL`, mean iterations >1, and
  `orthohalt_off` regresses → the algebra is doing real work even if absolute
  PPL doesn't yet beat the Transformer.
- **Kill**: `V8-A` fails to match QPAM, or `V8-D` regresses vs `V8-B`, or
  `orthohalt_off` matches `on` → the bank/loop is dead weight; document and
  prune. Keep SPM-as-projector as the one survivable artifact and try a
  different reasoning primitive.

---

## 8. How to run things

Smoke gate (TinyStories, ~1 minute on 4090):

```bash
./scripts/run_v8_e2e_smoke.sh
# or with a custom budget:
EPOCHS=3 MAX_SAMPLES=20000 ./scripts/run_v8_e2e_smoke.sh
```

Watch for `γ > 0` in the QLC diag lines, the `[qlc-schedule]` transitions
firing at ~1/3 and ~2/3 of the run, and `halt(yes/no/cont)` with non-zero
mass on each bucket. If those look healthy, launch the medium run.

Medium WikiText-103 run (long, do it overnight on the 4090 in tmux):

```bash
tmux new -s v8_e2e
cd /home/gowrav/Development/qllm2
./scripts/run_v8_e2e_medium.sh
# Ctrl-b d  to detach

# resume from the best checkpoint:
./scripts/run_v8_e2e_medium.sh --resume

# disable the QLC schedule (use preset values fixed):
./scripts/run_v8_e2e_medium.sh --no_schedule
```

The trainer's `TeeLogger` writes everything to
`logs/v8/e2e_medium_reasoning_wikitext103_*/v8_e2e_medium_reasoning_wikitext103.log`
automatically — no need for `| tee`. The final and best checkpoints land in
`v8/checkpoints/e2e_medium_reasoning/`.

---

## 9. Where to look in the code

If you want to actually read the implementation, here's the order I'd
recommend:

1. **[`v8/config.py`](config.py)** — start here. The `QLCConfig` dataclass and
   the `PRESETS` dict tell you every knob V8 has and what each Stage B row
   tweaks.
2. **[`v8/model.py`](model.py)** — `V8LM.__init__` and `V8LM.forward` show
   how the backbone hands `psi` to the QLC and how the LM head ties back to
   the embedding.
3. **[`v8/qlc/halt.py`](qlc/halt.py)** — the cleanest place to see the (α, β,
   γ) math. `OrthoHalt.forward` is short and very readable.
4. **[`v8/qlc/effect_bank.py`](qlc/effect_bank.py)** — `probe`, `select_top_k`,
   and `infonce_loss` show how the fact library is read and how it gets
   trained.
5. **[`v8/qlc/projector.py`](qlc/projector.py)** — `streaming_step` and
   `retrieve` are the heart of the working memory. The docstring at the top
   of the file explains the invariants.
6. **[`v8/qlc/reason_loop.py`](qlc/reason_loop.py)** — `QuantumLogicCore.forward`
   is the iteration loop. Once you've read 3–5, this should be obvious.
7. **[`v8/train.py`](train.py)** — the stage-aware trainer; the QLC
   diagnostic prints (`alpha`, `beta`, `gamma`, `halt(yes/no/cont)`,
   `align`) come from here.

The v8.2 fix bundle (alpha/gamma unfreeze, see §0.2) touches files 1, 2, 3,
6 and 7 in this list -- read them in that order if you are bisecting why
the fix worked or want to add another aux on top.

---

## 10. One-line summary

> **V6 had one engine doing everything; V8 has a grammar engine, a fact
> library, and a reasoning loop with a built-in if/else, so we can finally
> tell which part is doing the work.**

---

## 11. V8.1 Reframe: From Operational Quantum Logic To Local Test Spaces

### Better operational reading of the Coecke / Moore / Wilce paper

The original V8 plan motivated the design with a global appeal to operational
quantum logic ("the LM should compute in a Hilbert-space-like algebra"). On
re-reading [`/home/gowrav/Downloads/0008019v2.pdf`](../../Downloads/0008019v2.pdf)
in light of the audit, the more honest takeaway from the paper is much
narrower and much more useful:

- The interesting structure in operational quantum logic is the existence of
  **local test spaces** — small partitions of "questions you can ask in a
  consistent context" — and **ordered transformations between them**. The
  paper does *not* claim that the global Hilbert-space algebra is what makes
  quantum theory empirically distinctive; it claims that *test spaces with
  Sasaki-style updates over them* are.
- The classical, ablation-friendly reading is therefore: **partition the
  bank into typed local "test bundles", apply ordered local updates to the
  state, and measure whether the order changes the answer.** That is exactly
  what a corrected `quantale_order_test` row probes (config preset
  [`stageB_quantale_order_off`](config.py)).
- The unsharp effect generalization (`E = σ(g) u uᴴ`, `E² ≠ E`) is the
  honest place where "non-classical" structure shows up — and it is exactly
  what makes γ non-trivial in [`OrthoHalt`](qlc/halt.py) when
  `unsharp_target=True` (preset [`stageB_unsharp_ortho`](config.py)).

### What the paper *does not* license

- The paper does **not** show that complex coefficients per se make any
  classical task easier. Whether complex weights help V8 must be *measured*
  against a real-only baseline (preset [`stageB_real_spm`](config.py)),
  not assumed.
- The paper does **not** describe a non-commutative "quantale" composition
  rule for sequential test applications in a way that would predict an LM
  perplexity improvement; it describes the algebra at the level of
  *propositions about a single system*. Treat any LM-side ordering claim as
  empirical, not derived.

### Revised success criteria (reproduced from the rethink plan)

| Result tier        | What survives                                                                                                       | Naming                                       |
|--------------------|---------------------------------------------------------------------------------------------------------------------|----------------------------------------------|
| **Strong win**     | Discriminator suite §G shows γ non-trivial, order matters, complex beats real, and out_scale grows with training.   | "V8 — Operational Quantum-Logic Core"        |
| **Soft win**       | Constrained low-rank memory + adaptive halt beats matched-FLOP passthrough, but the algebraic claims do not hold.   | "V8 — Constrained Latent Memory + Adaptive Compute" |
| **Negative**       | At matched FLOPs, neither memory nor adaptive compute beat passthrough on the medium backbone.                       | Drop V8; document the negative; keep the SPM module as an artifact for V9 / TN branch. |

The **soft-win naming is the current default**. We do not assume the strong
win until §G results disagree.

### Discriminator suite (§G of the rethink plan)

Originally designed as cheap go/no-go runs before committing to a 14h
Stage A re-train + A100 sweep. The dedicated launcher was retired with
the rest of the staged scripts on 2026-04-24, but each row remains a
preset in [`v8/config.py`](config.py) and can be invoked manually with
`python -m v8.train --preset <name>` against the canonical `stageB_T2`
baseline (which itself runs without a Stage A backbone).

| Preset                              | Tests                                                                              |
|-------------------------------------|------------------------------------------------------------------------------------|
| `stageB_equal_flop_passthrough`     | Equal-FLOP control (run for ~T_max× more epochs to match QLC compute).             |
| `stageB_real_spm`                   | Real-vs-complex SPM. If real matches, complex is decorative.                       |
| `stageB_rank_r1` / `r2` / `r4`      | Rank sweep. If r=1 captures the gain, "subspace memory" is a 1-D modulator.        |
| `stageB_outscale0` / `01` / `1`     | Pin the residual scale. If `out_scale=0` matches, the residual is noise.           |
| `stageB_halt_delta`                 | Replace OrthoHalt with empirical step-delta halt.                                  |
| `stageB_halt_entropy`               | Replace OrthoHalt with surrogate predictive-entropy halt.                          |
| `stageB_unsharp_ortho`              | Make γ non-trivial via gated rank-1 target.                                        |
| `stageB_quantale_order_off`         | True symmetrized vs sequential composition test (the *real* `quantale_off`).       |
| `stageB_lmhead_unfrozen`            | Decouple "QLC weak" from "frozen readout cannot adapt".                            |
| `stageB_infonce_on`                 | Train the bank with the InfoNCE entity-routing objective the original plan promised. |

For TinyStories smoke parity there are matching `smoke_tiny_*` rows. The
discriminator suite is intended to take ~7 hours of 4090 time end-to-end.

### How to read diagnostics under the new framing

The trainer's per-step QLC diagnostic line now prints (v8.2 added the
`align=` column):

```
QLC: iter=X.XX alpha=… beta=… gamma=… halt(yes/no/cont)=…/…/… | align=… out_scale=… psi_delta_l2=…
```

Interpretation guide:

- **`align`** *(v8.2, leading indicator)*: mean `|u(psi)^H psi|^2 = alpha + gamma`
  accumulated over the iteration loop. At random init this sits at the
  `1/d` noise floor (`~0.003` for `dim=384`, `~0.005` for `dim=192`). The
  `target_alignment_weight` aux pulls this up; expected trajectory is
  `>= 0.05` within an epoch and `>= 0.2` by mid-training. **If `align`
  stays at noise floor, alpha and gamma cannot rise either** -- treat
  this as the earliest kill signal (preceding `gamma >= 0.05` from §9.2 of
  `EXPERIMENTS_V8.md`). With `target_alignment_weight=0.0` (legacy /
  ablation) `align` is still printed but is not actively pulled up; that
  reproduces the 4450ff0 frozen-signal failure mode by design.
- **`gamma ≈ 0` with sharp OrthoHalt**: structural, expected — see
  [`AUDIT_V8.md`](AUDIT_V8.md) §1. It is *not* evidence about
  non-commutativity. To get an honest γ, use a preset with
  `unsharp_target=True` — both `e2e_*_reasoning` presets do this by
  default; the historical equivalent was `stageB_unsharp_ortho`. **v8.2
  caveat**: `unsharp_target=True` is necessary but not sufficient -- the
  gate at `sigmoid(0)=0.5` only redistributes a near-zero `amp_sq` between
  alpha and gamma, it does not make `amp_sq` itself grow. The
  `target_alignment_weight` aux is what does that growth. See
  `AUDIT_V8.md` §6.
- **`halt(yes/no/cont)`**: should be a non-degenerate distribution with
  none of the three below ~0.05 at convergence. The 4450ff0 failure mode
  was `0.02/0.96/0.02` -- pinned at halt-no because `cls_head` was init'd
  with `weight=4*I`. v8.2 init is `weight=eye*1.0, bias=[0,0,2]` so the
  default distribution is roughly `[0.10, 0.10, 0.80]`; the loop runs all
  `t_max` iterations from step 0 and the model can move off "continue"
  only by learning that some inputs deserve halt-yes/halt-no.
- **`out_scale` trajectory**: if it stays near init, the model is satisfied
  with a small QLC contribution — consistent with the "QLC is regularizer"
  hypothesis. If it grows toward 1, the model is asking for more QLC —
  consistent with "QLC is doing real work".
- **`psi_delta_l2`**: the per-token L2 magnitude of the QLC residual. Useful
  to compare across rows: a row that beats baseline with small
  `psi_delta_l2` is a stronger result than one that needs a large residual
  to win.
- **`infonce`** (only when `qlc.infonce_weight > 0`): EMA of the bank's
  contrastive loss on the synthetic entity-cloze. Should drop monotonically
  when the bank is learning to route entity-typed queries to dedicated
  effects. v8.2 enables this by default in both `e2e_*_reasoning` presets
  at `weight=0.05, every=4`.

The `target_alignment_weight` knob is **runtime-safe** in the strict sense
that toggling it between 0 and 0.05 does not change parameter shapes -- but
because it changes the gradient signal seen by `OrthoHalt.target_mlp` and
the bank, mid-run toggles will produce a regime change. Treat it as a
preset-level decision, not a per-step knob.

---
