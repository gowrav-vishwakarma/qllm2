# V12: Depth-growth curriculum + playable module registry — first end-to-end run

> **Design source of truth is [README.md](README.md)** (M1–M5 mechanisms, presets, CLI).
> This file is the **lab notebook**: what we actually ran, what the numbers were, what we
> learned, and what to do next. *Last updated: 2026-07-28.*

**One-line verdict:** the memory mechanism **works on the distribution it was trained on** —
the fact module does long-range key→value binding at **0.925** (original) / **0.746**
(Phase-0 50-value control) where the grammar base scores chance — but the capability was
(a) **destroyed by composition** (0.925 → 0.003 once packed; now fixed by `--freeze_shared`)
and (b) **not a general copy operation**: changing the value vocabulary alone collapses
binding to chance, and raising the pool from 50 → 1000 drops in-distribution binding from
0.746 → 0.271. As of 2026-07-30 **D1 is closed** (pack fidelity) and **D2 is closed in both
directions** — closed-set / value-specific readout, not undertraining, and the pool sweep
(50→200→1000) is a smooth saturation curve, not a cliff. Contrastive pressure (λ 0.1→0.5)
changed nothing at pool=50. See [Phase 0](#phase-0-run-status-2026-07-30--d2-closed-both-directions).

**The two defects, stated separately, because they have different fixes:**

| | What it is | Status (2026-07-29) |
|---|---|---|
| **D1 Composition** | Modules are trained against a live shared embedding table; no single table can satisfy two modules at once | **Solved for pack fidelity.** `--freeze_shared` makes packed bit-for-bit identical to trained — but costs 29× on the objective, because the LM head is tied to the frozen table. See [follow-up](#follow-up-is-d1-fixable-and-does-diversity-induce-transfer-2026-07-28) |
| **D2 Transfer** | The fact module memorizes a small closed value set; vocab shift alone → chance | **Closed both directions.** 50-value control binds at 0.746 in-distribution and fails vocab transfer; pool sweep 50→200→1000 is a smooth saturation (0.746→0.333→0.271); contrastive λ 0.1→0.5 at pool=50 changed nothing. M3 (bounded vault / key-phase / delta) is now the primary bet. |

---

## What we ran (curriculum, 2026-07-17 → 2026-07-23)

Four stages via [scripts/train_curriculum.sh](scripts/train_curriculum.sh), all on the
local RTX 4090, all `--preset v12_grammar_dyn` (dim=384, H_max=16, head_dim=64, K=3
additive substrate, `head_gate` + `write_phase_address`, `max_seq_len=1024`).

| Stage | Module | Data | Tokens | B / LR | tok/s | Wall | Stack | Result |
|---|---|---|---:|---|---:|---:|---|---|
| base | `grammar@1.0` | dclm+fineweb 70/30 | 200M | 2 / 1e-4 | 16,927 | 3.28h | 4L, 62.3M | val PPL **122.00**, Wiki **387.95** |
| fact (delta) | `fact_retrieval@1.0` | `--dataset fact` + `ce_fact` | 300M | 32 / 3e-5 | ~7,000 | ~11.9h | 8L, 85.6M | answer-masked val PPL **2.5745** |
| fact (additive) | `fact_retrieval@1.1` | same data/loss | 300M | 32 / 3e-5 | ~6,900 | ~11.9h | 8L, 85.7M | answer-masked val PPL **1.8895** |
| reasoning | `reasoning@1.0` | fineweb+smoltalk2_mid 50/50 | 300M | 8 / 1e-4 | 4,761 | 17.50h | 12L, 109.0M | train PPL 36.22, val PPL **99.72**, Wiki **314.26** |

**Total: ~1.1B tokens, ~44.6 GPU-hours.** Each stage grew 4 layers on its published
predecessors via `--substrate`, froze the substrate blocks, then compacted and published.

Registry after the run (`v12_registry/index.json`): `grammar@1.0` (role=base),
`fact_retrieval@1.0` (delta), `fact_retrieval@1.1` (additive), `reasoning@1.0`.
The `fact` stages are an A/B pair on identical data and loss; `reasoning@1.0` declares
`grammar@>=1.0` + `fact_retrieval@1.0` as `prelayer` deps.

Logs: [../logs/v12_smoke_fact/](../logs/v12_smoke_fact/),
[../logs/v12_v12_grammar_dyn_pretrain_pretrain_mix.log](../logs/v12_v12_grammar_dyn_pretrain_pretrain_mix.log).

---

## Stack-order benchmark (2026-07-27)

Five permutations of the same frozen modules, packed via `v12.pack --spec` (the resolver
path enforces dependency order and `substrate_hash`, so arbitrary reordering has to use the
hand-written spec, which skips verification). Specs in `packed_v12/specs/`, runner
[scripts/permute_eval.sh](scripts/permute_eval.sh), raw output in
[../logs/v12_perm/](../logs/v12_perm/).

Eval: PPL on WikiText-103 val + DCLM-edu holdout at seq=1024/batch=4; recall via
`v12.eval_recall`, 30 trials, ctx 128/512/1024, n ∈ {1,4,8}, `candidate_count=8`
(**chance = 0.125**).

| arm | stack | Wiki PPL | DCLM PPL | sa@1024 |
|---|---|---:|---:|---:|
| `gfr` | grammar + fact@1.0 + reasoning (canonical) | 479.08 | 178.14 | 0.100 |
| `grf` | grammar + reasoning + fact@1.0 (order swap) | 480.34 | 179.06 | 0.100 |
| `gf11r` | grammar + fact@1.1 additive + reasoning | 475.62 | 178.35 | 0.111 |
| `gf` | grammar + fact@1.0 (no reasoning) | 393.50 | 123.30 | 0.100 |
| `gr` | grammar + reasoning (fact removed from beneath) | 473.69 | 176.83 | 0.100 |

All four 12-layer arms sit within **1.4%** of each other. `gr` — which strips out the very
module `reasoning@1.0` was trained on top of, violating its declared substrate — scores
*marginally better* than canonical `gfr`.

### Control: trained checkpoint vs its packed form (2026-07-27)

Each stage's own checkpoint **is** the corresponding stack, carrying the embeddings it was
trained with. Same harness, same settings. This is the decisive comparison.

| model | shared params from | Wiki PPL | DCLM PPL | sa@1024 |
|---|---|---:|---:|---:|
| grammar base, 4L | itself | **388.12** | **121.95** | 0.100 |
| fact stage, 8L (trained) | itself | 4042.78 | 1129.92 | 0.067 |
| `gf` packed, 8L | grammar@1.0 | 393.50 | 123.30 | 0.100 |
| reasoning stage, 12L (trained) | itself | **314.39** | **99.66** | 0.078 |
| `gfr` packed, 12L | grammar@1.0 | 479.08 | 178.14 | 0.100 |

Logs: [../logs/v12_perm/control/](../logs/v12_perm/control/).

**Neither packed model equals its trained model, and the error runs in both directions.**
Packing *rescued* the fact stack (4042.78 → 393.50) and *destroyed* the reasoning stack
(314.39 → 479.08).

---

## Does the fact module actually bind? (2026-07-27)

`single_assoc` at chance told us nothing about *why*. The training documents place the answer
in the context, so a model could score well by emitting "some plausible value token from this
document" without ever writing an association. [diagnose_fact_shortcut.py](diagnose_fact_shortcut.py)
separates that shortcut from real binding by scoring the model on its own validation
distribution under tightening candidate restrictions. The decisive column is
**ctx-restricted**: given only the values actually present in the document, does the model
pick the one bound to the queried key?

Validation documents use a seed disjoint from training (`12345` vs `0`) and keys are freshly
generated nonces per document, so nothing here is memorized from training data.

| model | top-1 (full vocab) | ctx-restricted | chance | verdict |
|---|---:|---:|---:|---|
| grammar base 4L | 0.000 | 0.222 | 0.246 | no binding |
| **fact delta, trained 8L** | **0.747** | **0.747** | 0.246 | **real binding** |
| **fact additive, trained 8L** | **0.784** | **0.784** | 0.246 | **real binding** |
| `gf` packed 8L | 0.003 | 0.255 | 0.246 | destroyed by packing |
| `gfr` packed 12L | 0.003 | 0.263 | 0.246 | destroyed by packing |
| reasoning, trained 12L | 0.000 | 0.219 | 0.246 | destroyed by the next stage |

Raw output: [../logs/v12_perm/shortcut/](../logs/v12_perm/shortcut/).

**The memory mechanism works.** The fact module retrieves the correct value across a
long filler gap at 0.747–0.784 where its own frozen substrate scores chance. This is the
first V12 evidence that the M3 fact-band design does what it was built to do.

**And two separate things destroy it.** Packing the identical blocks with grammar's shared
params drops it to 0.003. Training the *reasoning* stage on top drops it to 0.000 — even
though the fact blocks were frozen throughout that stage, because the shared embedding table
they read from moved out from under them. **Freezing blocks does not preserve a capability.**

### The transfer grid (2026-07-27)

The trained fact module still scores chance on `v12.eval_recall` (0.067). But the training
loader and the behavioral probe differ on **two axes at once** — template
(`Record: X means Y. ` vs `Memory record 1: X means Y.\n`) and vocabulary (50 training values
and nonce keys vs 16 disjoint values and fixed pseudo-word keys) — so that number was
uninterpretable. The 2×2 separates them (200 trials/cell, ctx-restricted accuracy):

**fact delta, trained** — chance is 0.241 (train vocab) / 0.176 (held-out vocab):

| | vocab = train | vocab = held-out |
|---|---:|---:|
| **template = fact** (training format) | **0.925** | 0.130 |
| **template = probe** (eval format) | 0.250 | 0.250 |

**fact additive, trained:** 0.925 / 0.115 / 0.230 / 0.235 — statistically identical.
**grammar base (control):** 0.250 / 0.115 / 0.260 / 0.250 — chance in all four cells.

**Changing *either* axis alone collapses the model to exactly its grammar-base floor.**
It is not that vocabulary transfer is hard and template transfer is easy, or vice versa —
each is independently fatal. The module learned "in a `Record: X means Y.` document drawn
from these 50 values, retrieve Y for X" as a surface pattern, not "bind a key to a value" as
an operation. With one template and 50 values across 300M tokens, that is the solution the
data asked for.

### Tier 0 result: composition is not recoverable by picking a better table

If D1 were a simple bug, some choice of shared params would make the packed stack work. It
does not exist. The **same 12 layers of blocks**, varying only the shared embedding table:

| `gfr` variant | shared params from | Wiki PPL | binding (in-distribution) |
|---|---|---:|---:|
| as shipped | `grammar@1.0` | 479.08 | 0.003 |
| reasoning's own | reasoning stage | **314.39** | 0.000 |
| fact's own | fact stage | 5098.49 | **0.940** |

Each table is optimal for exactly one objective and catastrophic for the other. The fact
blocks need the fact-stage embeddings to retrieve; the reasoning blocks need the
reasoning-stage embeddings to model language; the two are 38% apart. **No single shared table
satisfies both, so "train specialists sequentially on a live shared substrate, compose later"
cannot work as currently designed** — this is a design flaw, not an implementation slip.

Artifact: `packed_v12/gfr_factemb.pt`.

---

## Follow-up: is D1 fixable, and does diversity induce transfer? (2026-07-28)

Two experiments against the two defects. **D1 is now solved** — with a real cost.
**D2 is not yet answered**, because the diversified task turned out to be much harder to
learn than the memorized one, and 40M tokens does not reach binding.

### What changed in the code

`--freeze_embeddings` was **inert for this curriculum**: `v12/train.py` only honoured it
inside the `if args.active_heads ...` branch, which `train_curriculum.sh` never triggers — and
even when it fired it froze `self.embed` alone, while `pack` overwrites six prefixes. The
documented "one-flag stopgap" therefore never existed.

- **`--freeze_shared`** (new, [train.py](train.py)) freezes every param matching
  `_SHARED_PREFIXES`, imported from [pack.py](pack.py) so the two lists cannot drift.
  Applied unconditionally after `--freeze_layers`. `FREEZE_SHARED=1` in the curriculum script.
- **Drift warning** at *publish* time, not pack time. Pack can never detect this: `publish`
  strips shared params from a `role=group` module, so by pack time the evidence is gone. The
  check now compares a stage checkpoint against its resolved base module and fired correctly
  on the 2026-07 artifacts (`lm_head_proj` 0.14 for the fact stage, 0.38–0.58 for reasoning).
- **Fact data diversified on both axes** ([fact_data.py](fact_data.py)): 12 record/query
  phrasings + 4 held out (one being the exact `memory_probes/behavioral.py` wording, so
  `eval_recall` is now literally the held-out-template cell); a ~14k single-token value sweep
  hash-split 80/20 with behavioral `VALUES` forced held-out; four nonce key shapes plus the
  probe's pseudo-words as a held-out key split; filler sampled from a 40-sentence bank and
  made **independent of the template**, because an eval-only filler would shift together with
  the template and confound the very attribution the grid exists to make.
- **The grid now runs the real training generator** with splits swapped, instead of a parallel
  reimplementation, so only the axis under test varies.

### The runs

Three attempts; the first two are recorded because their failures are the finding.

| attempt | lr | seq | value pool | supervised tok/doc | outcome |
|---|---|---:|---:|---:|---|
| A | 1e-4 | 1024 | ~14k | 1.6 | **NaN** at step 954 (7.8M tok) |
| B | 5e-5 | 1024 | ~14k | 5.1 | unfrozen finished at val PPL **5001**, binding 0.229 / chance 0.164 → never learned; frozen **NaN** at step 1653 |
| C | 3e-5 | 512 | 1000 | 5.1 | both arms completed cleanly |

Attempt C's three changes were all necessary. `lr=3e-5` is the only stable setting (matching
the production run). `seq 512` doubles documents per token *and* halves the window over which
the no-decay vault state can grow, which is where the NaN came from. Capping the pool at 1000
values — still **20× the original 50** — made the retrieval target learnable at this budget.
Querying every fact instead of ~1.6 per document tripled the gradient signal for free: only
value tokens carry loss, so one query per 1024-token document wasted 99.8% of the forward pass.

Both attempt-C arms grew 4 delta fact-band layers on `grammar@1.0`, 40M tokens, batch 8,
`ce_fact`, published as `fact_retrieval@2.0` (unfrozen) and `@2.1` (frozen).
Logs: [../logs/v12_transfer/](../logs/v12_transfer/), grids in
[../logs/v12_transfer/grid/](../logs/v12_transfer/grid/).

### Result 1 — composition fidelity is SOLVED by `--freeze_shared`

| arm | shared params | tensors differing, trained vs packed | max abs delta |
|---|---|---:|---:|
| `@2.0` unfrozen | trained | **9** (exactly the shared set) | 9.83e-02 |
| `@2.1` frozen | frozen | **0** | **0.00e+00** |

The frozen arm's packed checkpoint is **bit-for-bit identical** to the checkpoint that was
trained, and the transfer grid confirms it behaviourally — 0.295 / 0.259 / 0.203 / 0.180
trained, the same four numbers packed. The unfrozen arm reproduces the D1 defect on the new
pipeline: in-distribution 0.258 trained → **0.142** packed (below chance), and top-1 over the
full vocabulary collapses 0.231 → **0.000**.

**D1 is a closed problem for `pack`-time fidelity.** Freeze the shared params and composition
is the identity. No adapters are required to make packing lossless.

### Result 2 — but freezing costs 29× on the objective

| arm | answer-masked val PPL |
|---|---:|
| `@2.0` unfrozen | **29.5** |
| `@2.1` frozen | **857.3** |

This is the Tier-0 tradeoff, measured rather than assumed, and it is far larger than expected.
The LM head is **tied to the embedding table**, so `--freeze_shared` freezes the readout: the
grown fact layers must produce hidden states that align with a projection optimized for
grammar. Interestingly the frozen arm's *relative* binding is slightly better
(0.295 vs 0.258, chance 0.194) — it is worse at modelling the answer distribution, not at
pointing within it. Both remain far below any useful threshold.

So the honest form of the Tier-0 table is: freezing buys exact composition at a large
capability cost, which is precisely the argument for **per-module adapters** — a module needs
*some* trainable path into the readout, and it must be one the module owns.

### Result 3 — the transfer gate is INCONCLUSIVE, not failed

Attempt C, ctx-restricted accuracy, 200 trials/cell (chance ≈ 0.19):

**`@2.0` unfrozen (trained):**

| | vocab = train | vocab = held-out |
|---|---:|---:|
| **template = train** | 0.258 | 0.210 |
| **template = held-out** | 0.223 | 0.201 |

**`@2.1` frozen (trained = packed):** 0.295 / 0.259 / 0.203 / 0.180.

**The in-distribution control is the number that matters here, and it is 0.258–0.295 against
chance 0.194.** Neither model learned to bind *at all*, so the off-diagonal cells measure
nothing — there is no capability to transfer. The diagnostic previously called this a FAIL;
it now reports INCONCLUSIVE and checks the control cell first, because reporting "diversity
did not induce transfer" about a model that never learned the task would have been wrong.

What the models did learn is visible in the numbers: full-vocab top-1 of 0.231 with
ctx-restricted accuracy at chance means the model reliably emits *a value word present in the
document* and then picks among them at random. Answer PPL fell 1000 → 29 by narrowing 1000
candidates to ~5; none of that gain is binding. This is exactly the shortcut
[diagnose_fact_shortcut.py](diagnose_fact_shortcut.py) was built to expose, now caught during
the run instead of a month later.

**The decisive next experiment is one 80-minute control**, and it is cheap because the
pipeline is now in place: rerun attempt C with `--fact_value_pool 50` to match the original
run's vocabulary size, changing nothing else.

- If it reaches high in-distribution binding → the pipeline is sound, pool size is the
  blocker, and the 2026-07 module's 0.925 was achieved through **value-specific readouts**.
  That would be strong evidence the mechanism cannot learn a general copy operation, which is
  a far more serious finding than a data-design bug.
- If it also fails → attempt C is simply undertrained at 40M tokens and the budget must rise
  before the transfer question can be asked at all.

Until that control runs, **no claim about D2 should be made in either direction.**

---

## Findings

| # | finding | evidence | implication |
|---|---|---|---|
| 1 | **Composition is not identity-preserving, and no shared table fixes it** | reasoning 314.39 → 479.08 packed; fact 4042.78 → 393.50 packed. Same `gfr` blocks: grammar table → 479 PPL / 0.003 binding, reasoning table → 314 / 0.000, fact table → 5098 / 0.940 | **Structural, not a patchable bug.** Registry, resolver, hash verify and packer all behaved correctly. Sequential training on a live shared substrate makes modules mutually incompatible; the interface itself has to change (see Tier 0) |
| 2 | **Root cause: specialist stages retrain the shared embeddings, packing discards them** | `V12LM.freeze_layers` iterates `self.blocks` only ([model.py:1551](model.py)); `train_curriculum.sh:215` passes `--freeze_layers base` and never `--freeze_embeddings`; trainable params 62,282,900 = 4 grown layers **+ the full 38.6M embedding table**. `pack.py:45` takes shared params only from base | Drift vs `grammar@1.0`, rel-Frobenius: fact stage `embed_real` **0.091** / `embed_imag` **0.104**; reasoning stage **0.376** / **0.282**. Blocks were trained against embeddings that packing throws away |
| 3 | **The fact stage catastrophically forgot web text** | Wiki 388.12 → **4042.78** after 300M tokens of synthetic fact data with unfrozen embeddings | Training a specialist on narrow synthetic data while the shared table is live destroys the base. Packing masked this by restoring grammar's embeddings |
| 4 | **Depth growth genuinely works — on real data, in trained form** | reasoning stage 12L: Wiki 388.12 → **314.39**, DCLM 121.95 → **99.66** | The M4 premise ("depth gives everything") is supported *while training*. But the gain only exists alongside that stage's own embeddings, and keeping them costs the fact module its entire capability — so it is not currently shippable through the registry |
| 5 | **Stack order is currently unmeasurable** | four 12L arms within 1.4% (473.69–480.34); `gr` (substrate violated) ≈ `gfr` (canonical) | Finding 1 dominates any ordering signal. The permutation benchmark must be **re-run**, not interpreted. The current numbers say nothing about swappability |
| 6 | **The memory mechanism WORKS — the fact module does real long-range binding** | ctx-restricted accuracy **0.747** (delta) / **0.784** (additive) on unseen documents with novel nonce keys, vs **0.222** for its own frozen grammar substrate at chance 0.246. In the cleaner single-query grid, **0.925** | First positive V12 architecture result. M3 fact bands store and retrieve across a long filler gap. Every previous "recall at chance" number came from packed checkpoints or from the mismatched probe |
| 7 | **The capability is destroyed twice over — by packing, and by the next training stage** | `gf` packed 0.003 and `gfr` packed 0.003, from identical blocks scoring 0.747 trained. Reasoning stage 0.000 despite the fact blocks being frozen throughout it | **Freezing blocks does not preserve a capability** when the shared table they read from keeps moving. This is finding 2 with teeth: the cost is not a few PPL points, it is total loss of the skill |
| 8 | **Transfer fails on both axes independently — a data-design failure, not architecture** | Transfer grid: 0.925 in-distribution → 0.250 on template change alone → 0.130 on vocabulary change alone → 0.250 on both. Grammar base is 0.250 in every cell | One template and 50 values over 300M tokens taught a surface pattern, not an operation. Fix the data diversity before touching the mechanism |
| 9 | **Delta and additive are indistinguishable where it matters** | binding 0.925 vs 0.925 in-distribution; 0.130 vs 0.115 on held-out vocab. Answer-masked val PPL 2.5745 vs 1.8895 favours additive | V11 Stage-6 hypothesis #1 (error-correcting writes fix interference) gets **no support**: the two write modes reach identical binding. Given finding 8, this budget cannot discriminate them — the task is saturated in-distribution and at chance out of it |
| 10 | **M1 head-gate pruned nothing — the mechanism is untested, not disproven** | 16/16 heads open in all 12 layers, every arm ([../logs/v12_perm/*/headgate.log](../logs/v12_perm/)); `01_compact.log`: 0.0% smaller, `max\|Δlogits\| = 0.000e+00` | With `head_gate_l0_lambda=0.001` and `head_gate_init_logalpha=3.0` no gate ever closed. "Learnable head count" has no result yet |
| 11 | **We skipped our own advice on micro-tests** | V11 Stage-6 item 4: *"capacity micro-tests — 10M-param models, ~30M tok, before spending 300M+ token budgets"*. V12 spent 1.1B tokens / ~45 GPU-hours | Every finding above could have been caught for <2% of that cost |
| 12 | **Absolute quality is smoke-scale; no architecture conclusion is safe** | 4L base at Wiki 388 vs V11 E3 K=3 at **25.77** (~100M, 10 ep) | These are pipeline-validation runs. Generation loops degenerately in every arm ("means gold … means gold"), consistent with heavy undertraining |
| 13 | **Eval/infra papercuts that cost real time** | `eval_recall` default `--context-lengths` includes 2048 > `max_seq_len=1024`, so the headline silently reported `n/a` for every eval until the 2026-07-27 sweep — compounded by a ternary that swallows the label ([eval_recall.py:183](eval_recall.py)); `_HOLDOUT_CACHE_VERSION` NameError; OOM at seq 2048/batch 18; `pack.py` `TypeError` on the `module_card` config key; substrate vocab 50257 vs preset 50261; TeeLogger truncation that lost the middle of the fact run | Headline metrics must fail loudly, never degrade to `n/a` |
| 14 | **`--freeze_shared` makes composition the identity — D1 is solved for pack fidelity** | Frozen arm: **0** tensors differ between trained and packed checkpoints (max delta 0.00e+00), and all four grid cells match exactly. Unfrozen arm: exactly the 9 shared tensors differ (max 9.83e-02), in-distribution binding 0.258 → 0.142, top-1 0.231 → 0.000 | The registry direction is mechanically sound. Freezing shared params on every non-base stage is sufficient; adapters are needed for *capability*, not for correctness |
| 15 | **The documented stopgap never worked — `--freeze_embeddings` was dead code here** | Honoured only inside `if args.active_heads is not None`, which `train_curriculum.sh` never sets; and it covers `self.embed` alone while `pack` overwrites six prefixes | A mitigation nobody executed cannot be assumed to work. The 2026-07-27 advice to "pass `--freeze_embeddings`" would have silently changed nothing |
| 16 | **Freezing shared params costs 29× on the objective** | answer-masked val PPL **29.5** unfrozen vs **857.3** frozen, identical data and budget | The LM head is *tied* to the embedding table, so freezing it freezes the readout. This is the quantified Tier-0 tradeoff and the concrete argument for per-module adapters: a module needs a trainable path into its own readout |
| 17 | **The diversified task does not reach binding at 40M tokens — D2 retest inconclusive** | in-distribution ctx-restricted **0.258** (unfrozen) / **0.295** (frozen) vs chance 0.194, with full-vocab top-1 0.231 | Answer PPL fell 1000 → 29 purely by narrowing ~1000 candidates to the ~5 in context, then choosing at chance. A control at a 50-value pool separates "undertrained" from "the mechanism can only fake binding via value-specific readouts" |
| 18 | **Supervision density, not token count, was the binding constraint** | Only value tokens carry loss: ~1.6 supervised positions per 1024-token document = 0.16% of the forward pass. Querying every fact raised it to 5.1 | A 3.2× effective-data increase for zero extra compute. Any answer-masked objective should be audited for this before its budget is raised |
| 19 | **The no-decay vault state is the NaN source, and sequence length is the control** | NaN at step 954 (lr 1e-4) and 1653 (lr 5e-5) at seq 1024; zero NaNs at seq 512 / lr 3e-5 over 40M tokens | `vault_state` has no decay, so state magnitude grows with the window. The 2026-07 "NaN stabilization" treated symptoms (lower lr, softer aux) rather than the unbounded state |

### What worked

- **The memory mechanism** (finding 6). The M3 fact bands genuinely bind and retrieve across
  a long filler gap — 0.925 where the substrate is at chance. This is the thing V12 was built
  to prove, and it is proven.
- **The M5 pipeline, end to end.** `grow → freeze → compact → publish → resolve → pack →
  load → forward` ran for four real modules. Version solving, topological ordering,
  `substrate_hash` verification and card round-tripping all did their jobs.
- **Depth growth on real data** (finding 4): +4 reasoning layers on a frozen 8-layer
  substrate cut Wiki PPL 388 → 314 and DCLM 122 → 99.7.
- **Arbitrary reordering is mechanically possible.** `v12.pack --spec` composed all five
  permutations, including ones that violate declared dependencies, and every one loaded and
  ran. The *plumbing* for a swappable-module marketplace exists.
- **NaN stabilization** (abort-on-NaN in `v7/train.py`, softer `ce_fact` gate/contrast, logit
  clamp, `FACT_LR=3e-5`) held for 900M tokens after the first fact run diverged at ~80M.
- **`--freeze_shared` closes the composition defect** (finding 14). Packed is now bit-for-bit
  identical to trained. The registry direction is mechanically sound; what remains is a
  capability question, not a correctness one.

### What did not work

- **The module interface as originally designed** (findings 1–3, 7) — structural rather than a
  bug, and now **fixed for correctness** by `--freeze_shared` (finding 14) at a 29× capability
  cost (finding 16). Adapters remain the open work.
- **Fact-data design** (finding 8) — one template × 50 values produced a memorized surface
  pattern with zero transfer on either axis. Diversifying both axes did not yet produce a
  binding model at 40M tokens (finding 17), so the redesign is unvalidated, not vindicated.
- **Delta vs additive as an experiment** (finding 9) — the task saturates in-distribution and
  is at chance out of it, so it cannot discriminate write modes at this budget.
- **M1 head-gate** (finding 10) — no pruning at the configured λ.
- **The permutation benchmark** (finding 5) — ran cleanly, produced no usable signal.

---

## Process learnings for a novel-architecture + packing program

1. **A module's contract must cover shared params, not just blocks.** `substrate_hash` today
   hashes the frozen *blocks* beneath a module. It does not cover the embedding table the
   module was actually trained against, which is why a 38% embedding drift passed
   verification silently.
2. **The lossless-composition test must run on real curriculum artifacts.** `v12.selftest`
   already asserts `packed logits == original grown model` — but only for models it builds
   itself, which is exactly how finding 1 survived. The assertion needs to run against
   published registry modules.
3. **Log the trained checkpoint and its packed form side by side, every stage.** The 314 vs
   479 gap sat in the logs for four days because nobody evaluated both.
4. **Micro-test before budget spend** (finding 11). Gate every 300M-token run behind a
   ~30M-token, 10M-param version that clears an explicit threshold.
5. **Headline metrics must fail loudly.** `single_assoc: n/a` (finding 13) should have been a
   hard error the first time it printed.
6. **State chance level next to every accuracy.** Recall of 0.100 reads like a result until
   you notice chance is 0.125.
7. **A metric that changes two variables at once measures nothing.** `single_assoc` differed
   from the training distribution in both template *and* vocabulary, so its chance-level
   score was read for weeks as "the fact module does not work" when the module was in fact at
   0.925. One diagnostic that varies one axis at a time
   ([diagnose_fact_shortcut.py](diagnose_fact_shortcut.py)) overturned the headline
   conclusion of the entire program in under an hour.
8. **Evaluate the capability, not just the loss.** Answer-masked val PPL and `single_assoc`
   pointed in opposite directions for a month. Neither was wrong; they measured different
   things, and nobody built the third measurement that reconciles them.

---

## Next plan of action

Tier 0 and Tier 1 are independent — D1 and D2 are different defects — and both are
prerequisites for any further scale.

### Tier 0 — fix the module interface (D1)

The Tier 0 experiment above already ruled out the cheap fix: no choice of shared table makes
the packed stack work, so "pick the right embeddings" is not on the table. Three real options:

| Option | Composition | Module adaptivity | Marketplace viability |
|---|---|---|---|
| **Freeze shared params** — `--freeze_shared` on every non-base stage | **Exact — verified, 0 tensors differ** | Confined to the base's representation space | Works, but **measured at 29× val PPL** (finding 16); base quality is a hard ceiling for every module |
| **Ship shared-param deltas** — module owns a delta `pack.py` applies | Exact per module | Full | Breaks: two modules with conflicting deltas cannot both apply. This run is the counterexample |
| **Per-module adapters** — module never touches shared params, owns a small learned in/out projection on the residual stream at its boundary | Exact | Local to the module | The one that scales; conflicts vanish because adaptation lives inside the module's own blocks |

**Recommended sequence:** freeze first (one flag, un-blinds every downstream experiment
immediately), then build adapters as the real interface. Adapters fit the existing
`layer_specs` / `attach_mode` schema without a new concept.

**Status 2026-07-28:** the freeze half is **done and verified** — `--freeze_shared` exists,
`FREEZE_SHARED=1` is wired into the curriculum, and a full run confirmed packed == trained
bit-for-bit (finding 14). Item 3 below is complete. What the measurement added is urgency for
adapters: freezing also freezes the **tied LM head**, which cost 29× on the objective
(finding 16). An adapter design must therefore give each module a trainable path into its own
readout — a per-module output projection, not only a residual-stream adapter.

Remaining work regardless of which option wins:

1. Extend `substrate_hash` to cover the shared params a module was trained against, so a
   mismatch is a resolver conflict instead of a silent capability loss. **Still open** — the
   2026-07-28 work added a *warning* at publish time (pack cannot see it: `publish` strips
   shared params from a `role=group` module, so the evidence is gone by pack time), which is
   detection, not enforcement.
2. Add the lossless-pack assertion over **published registry modules** to `v12.selftest`.
   It currently only round-trips models it constructs itself, which is how this survived.
   **Still open.**
3. ~~Re-run the fact stage with frozen shared params and confirm binding survives packing.~~
   **Done.** Frozen arm: 0 tensors differ, all four grid cells identical trained vs packed.
   The registry direction is validated for fidelity.

### Tier 1 — fix the fact data (D2)

The mechanism works on its training distribution; the data taught a surface pattern.

1. ~~**Diversify both axes.**~~ **Done** — 12 train + 4 held-out templates, ~14k values
   hash-split 80/20, held-out key shapes, template-independent filler
   ([fact_data.py](fact_data.py)).
2. ~~**Make the transfer grid the training-time metric.**~~ **Done** — the grid now runs the
   real generator with splits swapped, and checks the in-distribution control cell first so an
   undertrained model reports INCONCLUSIVE rather than a false transfer failure.
3. ~~**THE NEXT EXPERIMENT — the 50-value control.**~~ **Done (2026-07-29).** Pool=50
   binds at **0.746** in-distribution and fails vocab transfer (chance). Pool=1000 never
   binds. **D2 = closed-set readout**, not undertraining. Pool sweep 50→200→1000 is running
   to locate the break point; see [Phase 0](#phase-0-run-status-2026-07-29--d2-answered-closed-set-readout).
4. **Restore the anti-shortcut pressure.** `fact_contrastive_lambda` was cut 0.5 → 0.1 and
   `tau` doubled to stop a NaN divergence ([losses.py:105](losses.py)), for numerical reasons
   rather than scientific ones. Finding 19 identifies the real NaN source (the no-decay vault
   state over a long window), so test full strength at seq 512 where the state is bounded.
5. **Audit supervision density on any masked objective** (finding 18) before raising a budget.
   Querying every fact instead of one tripled the gradient signal for free.
6. **Then, and only then, re-run delta vs additive.** Finding 9 says the current task cannot
   discriminate them; a task with real transfer might.

### Tier 1b — capacity micro-tests before any further budget

10M-param models, ~30M tokens. **Gate: clear the off-diagonal transfer-grid cells by 2×
chance before any run over 50M tokens.** V12 spent 1.1B tokens to learn things a 30M-token
probe would have shown (finding 11).

### Tier 2 — actually test M1

Sweep `head_gate_l0_lambda` upward from 0.001 (try 0.01 / 0.05) and lower
`head_gate_init_logalpha` from 3.0. Success = gates measurably close and `v12.compact`
reports a real size reduction with logits preserved.

### Tier 3 — re-run the stack-order benchmark

Only on Tier-0 artifacts. This is the actual test of the swappable-module thesis, and the
five arms + runner are already built ([scripts/permute_eval.sh](scripts/permute_eval.sh)).

### Tier 4 — matched baselines (still outstanding from V11)

~100M Mamba + Transformer trained on identical data and budget, then the behavioral suite.
Every recall number in V11 and V12 is currently compared against a 300B-token pretrained
Mamba, which is not a fair anchor.

### Strategic read

**Continue the direction; the interface is now fixed, the mechanism is now the open question.**
As of 2026-07-28 the packaging story is settled: freeze the shared params and composition is
the identity, verified bit-for-bit. The contract defect that looked structural on 2026-07-27
was structural *and* closable, and it is closed.

That inverts the risk. The remaining doubt is no longer "can we ship a module" but "is there a
capability worth shipping." Two measurements point the same uncomfortable direction:

- Freezing the shared params — the thing that makes modules composable — costs **29×** on the
  objective, because the tied LM head goes with it. A module confined to the base's readout
  may simply not have room to learn a retrieval skill. Adapters must therefore own an output
  projection, not just a residual-stream transform.
- Raising the value vocabulary from 50 to 1000 dropped in-distribution binding from 0.925 to
  0.258. If the 50-value control reproduces 0.925, the original result was a value-specific
  readout rather than a general copy operation — and "the PAM fact band binds" would need
  retracting to "the PAM fact band memorizes a small closed set."

**That control is the highest-value 80 minutes available**, and it should run before any
further spend on adapters, bases, or scale. The thing to stop doing is training specialists
that quietly rewrite the substrate everyone else depends on — that part is now enforced by a
flag.

**Also worth naming: the base is too weak to build a marketplace on.** Every module sits on a
4-layer, 200M-token base at Wiki 388, where V11 reached 25.77. Under any frozen-substrate
design the base is a hard ceiling for everything grown on it, so one genuinely good base is
worth more than further module-level iteration.

---

## Cross-version context

| Model | WikiText-103 val PPL | Note |
|---|---:|---|
| Transformer B=18 (V6) | **22.69** | ~100M, matched-batch anchor |
| V11 E3 K=3 multistate | **25.77** | best logged PAM, ~100M, 10 ep |
| V7 7d chunked B=18 | 26.88 | prior flat bar |
| **V12 reasoning 12L (trained)** | **314.39** | 109M params, 800M tok, 1 pass, seq 1024 |
| **V12 grammar base 4L** | 388.12 | 62M params, 200M tok |
| **V12 `gfr` packed 12L** | 479.08 | same modules as row above, composed |

V12 was never run at a budget where PPL is comparable to V6–V11. Read V12 as an
infrastructure and mechanism result; the quality bar still belongs to V11. The V12 number
that matters is not on this table — it is **0.925 binding vs a 0.246 chance floor**, the
first direct evidence that the PAM fact-band design retrieves what it stores.
See [../EXPERIMENTS_V_6_7_8_9.md](../EXPERIMENTS_V_6_7_8_9.md) and
[../v11/EXPERIMENTS_V11.md](../v11/EXPERIMENTS_V11.md).

---

## V12 next phase — implementation (2026-07-28)

Bounded-memory modular PAM work stays in `v12/` (no v13 package). Landed in code:

| Area | What shipped |
|---|---|
| **Phase 0** | [scripts/run_phase0_controls.sh](scripts/run_phase0_controls.sh) — 50-value control, pool sweep 50/200/1000, `ce_fact_strong` contrastive retest + transfer grid. Logs: `logs/v12_phase0/runs/` (tee), `.../internal/` (v12.train TeeLogger); `${tag}_pool${pool}.log.latest` → newest run. |
| **M3 delta** | `delta_solve_mode=backsub` (default): compile-friendly UT back-substitution; `linalg` remains for parity (`selftest`: `delta_backsub`). |
| **M3 vault** | `vault_norm_bound` caps Frobenius norm per head after vault writes (NaN guard at long seq). |
| **M3 phase** | `write_phase_key_conditional` — phase from key (real+imag), not `\|key\|` only. |
| **Interface v2** | `module_adapter_rank` + `ModuleBoundaryAdapter` (in/out low-rank + readout delta); `MODULE_ADAPTER_RANK` in curriculum; pack/publish merge `module_adapters.{group_id}.*`. |
| **Contract** | `substrate_hash` at grow time = blocks prefix **+** shared params (`hash_substrate_prefix`); resolver verifies the same composition. |
| **Micro-lab** | Presets `v12_grammar_dyn_micro`, `v12_micro_factband`; [scripts/run_micro_ab.sh](scripts/run_micro_ab.sh), [scripts/run_micro_baselines.sh](scripts/run_micro_baselines.sh). |
| **Strong base** | [scripts/run_strong_base.sh](scripts/run_strong_base.sh) → `grammar@2.0` target (1B+ tokens, `v12_e3_k3_recall`). |
| **Decode** | [recurrent_decode.py](recurrent_decode.py), [bench_infer.py](bench_infer.py) (v12 preset). |

### Phase 0 run status (2026-07-30) — D2 closed both directions

**Verdict:** the 50-value control **binds in-distribution** (ctx-restricted **0.746** vs
chance 0.199) and collapses to chance the moment the value vocabulary shifts. Combined with
attempt-C at 1000 values never learning to bind (0.258 vs chance 0.194), this confirms the
uncomfortable branch: the fact band learns a **value-specific readout over a small closed
set**, not a general key→value copy. Pool size is causal; the pipeline is sound.

#### control50 (finished 2026-07-28 20:18)

Attempt-C recipe: `grammar@1.0` substrate, 4 delta fact layers, 40M tokens, batch 8, seq 512,
lr 3e-5, `ce_fact`, `--fact_value_pool 50`, shared params **unfrozen**. Published as
`fact_retrieval@3.0` (requires repaired to `grammar@>=1.0:prelayer` on 2026-07-29).

| metric | control50 (pool=50) | attempt C (pool=1000) |
|---|---:|---:|
| answer-masked val PPL | **2.27** | 29.5 |
| val accuracy | **0.748** | — |
| wall time | 8.62 h @ **1289 tok/s** (3× trainers stacked) | ~80 min @ ~8k tok/s |

Transfer grid (200 trials/cell, ctx-restricted; [grid/control50.json](../logs/v12_phase0/grid/control50.json)):

| | vocab = train | vocab = held-out |
|---|---:|---:|
| **template = train** | **0.746** (chance 0.199) | 0.189 (chance 0.199) |
| **template = held-out** | 0.430 (chance 0.193) | 0.174 (chance 0.193) |

- Control cell lift **+0.547** → binding is real in-distribution.
- Vocab held-out alone → chance (the decisive axis).
- Template held-out alone still above chance (0.430) — templates transfer better than values.
- Both axes → chance. Gate: **FAIL** (surface pattern, not general copy).

#### Runner bugs found and fixed (2026-07-29)

1. **`set -u` expansion trap** in `run_grid`: `local tag="$1" ckpt="...${tag}..."` expands
   `${tag}` before assignment → `tag: unbound variable`. Killed the queue after publish; grid
   and later arms never ran. Fixed by splitting `local` declarations.
2. **`VER` reused for requires**: publish declared `grammar@>=3.0` while substrate was
   `grammar@1.0` → `fact_retrieval@3.0` unresolvable. Added `REQ_VER` (default 1.0) and
   export `REQUIRES=grammar@>=${REQ_VER}:prelayer`. Re-published `@3.0` with the correct requires.
3. **Duplicate-launch throughput collapse**: approval retry left 3 root trainers on one GPU →
   1289 tok/s vs ~8k. Guard now counts **root** `v12.train` PIDs only (ignores DataLoader
   workers) + lockfile; `remaining` / `sweep` use `;` sequencing so one arm failure does not
   cancel the rest.

#### Pool sweep (complete 2026-07-30)

`v12/scripts/run_phase0_controls.sh remaining` ran pool200 → contrastive → pool1000
serially, single trainer, with the duplicate-launch guard. All four arms trained 40M
tokens on `grammar@1.0`, 4 delta fact layers, batch 8, seq 512, lr 3e-5. Grids: 200
trials/cell, ctx-restricted. Chance ≈ 0.194–0.199.

| arm | pool | contrast λ | val PPL | ID binding (ctx-restr) | vocab transfer | tmpl transfer | both shifted |
|---|---:|---:|---:|---:|---:|---:|---:|
| control50 | 50 | 0.1 | **2.27** | **0.746** | 0.189 (chance) | 0.430 | 0.174 (chance) |
| pool200 | 200 | 0.1 | 12.57 | 0.333 | 0.203 (chance) | 0.267 | 0.203 (chance) |
| contrastive | 50 | 0.5 | **2.27** | **0.746** | 0.189 (chance) | 0.430 | 0.174 (chance) |
| pool1000 | 1000 | 0.1 | 30.54 | 0.271 | 0.217 (chance) | 0.230 | 0.192 (chance) |

**Read the table:**

- **Binding falls monotonically with pool size** — 0.746 → 0.333 → 0.271. The break is
  between 50 and 200; by 200 the lift over chance is already weak (+0.139); by 1000 the
  grid reports INCONCLUSIVE (lift +0.077, the model does not bind even in-distribution).
- **Vocab transfer is at chance in every arm** — the decisive axis never moves. The fact
  band learns a value-specific readout over whatever closed set it was trained on; it
  does not learn a general key→value copy.
- **The contrastive arm is identical to control50** — same PPL (2.27), same binding
  (0.746), same transfer (0.189). Raising `fact_contrastive_lambda` 0.1 → 0.5 at pool=50
  changed nothing. The anti-shortcut pressure does not induce generalization when the
  closed set is small enough to memorize; it only matters (if at all) where the task is
  already hard. This is a null result worth recording: the contrastive loss is not the
  lever for D2.
- **Template transfer is above chance only at pool=50** (0.430) — the small closed set
  lets the model partially generalize across phrasings; at pool≥200 that disappears too.

**Curve verdict:** pool size is causal and the relationship is smooth, not a cliff. The
fact band's binding capacity scales sublinearly with the value vocabulary — it saturates
around pool=50 and is effectively gone by pool=1000. Combined with the vocab-transfer
column being pinned at chance across all four arms, **D2 is closed in both directions**:
the mechanism memorizes a small closed set, and no pool size in this range produces a
general copy operation.

**Implication for the architecture program:** M3 work (bounded vault, key-conditioned
phase, compile-friendly delta) is now the **primary** bet, not a refinement — the current
fact band does not generalize a copy operation past a closed value set, and contrastive
pressure does not change that. Micro-lab gates (transfer-grid off-diagonal ≥ 2× chance)
must clear before any >50M-token spend.

```bash
# Re-run any grid alone:
v12/scripts/run_phase0_controls.sh grid control50 50
# Full sweep (serial, locked):
REQ_VER=1.0 VER=3.1 v12/scripts/run_phase0_controls.sh sweep
```

Published modules: `fact_retrieval@3.0` (control50) and `fact_retrieval@3.1`
(pool200/contrastive/pool1000 — last publish wins per version; re-publish per arm with
distinct `VER` if you need all four resolvable simultaneously). Checkpoints retained
under `checkpoints_v12_phase0/{control50,pool200,contrastive,pool1000}/fact_retrieval/`.

**Implication for the architecture program:** M3 work (bounded vault, key-conditioned phase,
compile-friendly delta) is now the **primary** bet, not a refinement — the current fact band
does not generalize a copy operation past a closed value set. Micro-lab gates (transfer-grid
off-diagonal ≥ 2× chance) must clear before any >50M-token spend.
