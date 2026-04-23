# V8 — A Language Model That Tries To *Think*, Not Just Predict

> **TL;DR** — Most language models do one thing: read words, guess the next
> word. V8 splits that job into **three different machines** working together:
> a **grammar engine**, a **fact memory**, and a **reasoning loop**. The
> reasoning loop is built out of complex numbers and quantum-logic primitives
> so the model has a native, mathematical way to say *"yes / no / I don't know
> yet, let me think more"* instead of always firing one forward pass and being
> done.
>
> See the running results in
> [`v8/EXPERIMENTS_V8.md`](EXPERIMENTS_V8.md).

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
  │                  - frozen after Stage A            │
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
| [`v8/config.py`](config.py)                                 | `V8Config`, `QLCConfig`, presets for every Stage A / B / C run    |
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
4. **Clean experiments.** Stage A produces a frozen backbone. Stage B then
   tests *only the new piece*. Any PPL change is unambiguously about the QLC,
   not about the backbone co-adapting (which is what made V6's memory bolt-ons
   un-debuggable).

### Where it could quietly fail

- `γ` could stay 0 forever — meaning the if/else algebra never engages and
  the win comes purely from the rank constraint. (We already see this in
  Stage A.5 smoke; if Stage B also shows `γ ≈ 0`, the OrthoHalt-off ablation
  becomes the kill test.)
- The bank could just memorize the training set (V6 §5.5 redux). We mitigate
  with top-k routing and InfoNCE on synthetic entity cloze.
- Re-orthonormalizing the projector basis every iteration could be unstable
  in autograd. The unit tests in
  [`v8/qlc/tests/test_projector.py`](qlc/tests/test_projector.py) check this
  directly; QR refresh is configurable.
- Sasaki updates are 2× more compute per step than QPAM, and we run up to
  T_max=4 of them. Even if quality wins, throughput hurts. Stage B numbers
  will tell us whether the cost/benefit is worth it.

---

## 5. Three stages, three different questions

V8 isn't trained in one shot. We deliberately separate the experiments so
each result tests exactly one thing.

| Stage | Script                                                                    | Asks                                                                | Output                                  |
|-------|---------------------------------------------------------------------------|---------------------------------------------------------------------|-----------------------------------------|
| **A**     | [`scripts/run_v8_stageA.sh`](../scripts/run_v8_stageA.sh)                 | "Can we reproduce the QPAM baseline cleanly?" (no QLC, just grammar) | `v8/checkpoints/qpam_stageA.pt`         |
| **A.5**   | [`scripts/run_v8_stageA5_smoke.sh`](../scripts/run_v8_stageA5_smoke.sh)   | "Does the QLC primitive even work without crashing?" (TinyStories) | smoke logs under `logs/v8/stageA5_*/`   |
| **B**     | [`scripts/run_v8_stageB.sh`](../scripts/run_v8_stageB.sh)                 | "Which knobs of QLC actually help on real data?" (frozen backbone) | `v8/checkpoints/stageB_*/`              |
| **C**     | [`scripts/run_v8_stageC_joint.sh`](../scripts/run_v8_stageC_joint.sh)     | "Does letting them co-adapt buy us another PPL or two?"            | `v8/checkpoints/stageC_*/`              |

Stage B is where the **real ablations** live: rank `r ∈ {4, 8, 16}`, bank size
`M ∈ {2k, 8k}`, reasoning depth `T ∈ {1, 2, 4}`, and two ablation switches
(`quantale_off`, `orthohalt_off`) that turn off the two key algebraic claims.

---

## 6. What we know so far (Stage A.5 smoke result)

Run on RTX 4090, TinyStories 20k stories, 3 epochs, `_backbone_tiny()`
(dim=64, 2 layers) — full details in
[`v8/EXPERIMENTS_V8.md`](EXPERIMENTS_V8.md) §5.

| Variant                  | QLC params | Best Val PPL | Tok/s   |
|--------------------------|------------|--------------|---------|
| Passthrough (V7-equiv)   | 0          | **178.95**   | 225 304 |
| QLC `r=4, T_max=2`       | 57 678     | **173.40**   | 19 180  |

**The gate passed.** The QLC variant beat the passthrough by 5.55 PPL on a
tiny model — meaning the new primitive is differentiable, doesn't NaN, and
adds something useful even at smoke scale.

**Honest caveat:** `γ` was `0.000` for the entire smoke. The win came from the
rank-`r` Sasaki retrieval acting as an inductive bias / regularizer, not from
the orthocomplement if/else. This is the most important thing to watch in
Stage B — if `γ` stays at 0 across all rows and `orthohalt_off` matches
`orthohalt_on`, then the (α, β, γ) head is decorative and the operational
quantum-logic story doesn't land at this scale (we'd keep SPM and prune the
rest).

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

Stage A (long, do it overnight on the 4090 in tmux):

```bash
tmux new -s v8_stageA
cd /home/gowrav/Development/qllm2
./scripts/run_v8_stageA.sh
# Ctrl-b d  to detach
```

The trainer's `TeeLogger` writes everything to
`logs/v8/stageA_*/v8_stageA_medium_wikitext103.log` automatically — no need
for `| tee`.

Stage B (after Stage A finishes; ideally on an A100):

```bash
./scripts/run_v8_stageB.sh           # prints all available rows
./scripts/run_v8_stageB.sh stageB_T4 # run one row
./scripts/run_v8_stageB.sh all       # sequential sweep
```

Stage C (after Stage B picks winners):

```bash
./scripts/run_v8_stageC_joint.sh stageC_T4_joint
```

All Stage B/C launches need `v8/checkpoints/qpam_stageA.pt`; the Stage A
script copies it into place automatically.

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
   diagnostic prints (`alpha`, `beta`, `gamma`, `halt(yes/no/cont)`) come
   from here.

---

## 10. One-line summary

> **V6 had one engine doing everything; V8 has a grammar engine, a fact
> library, and a reasoning loop with a built-in if/else, so we can finally
> tell which part is doing the work.**
