# V12: Depth-growth curriculum + playable module registry — first end-to-end run

> **Design source of truth is [README.md](README.md)** (M1–M5 mechanisms, presets, CLI).
> This file is the **lab notebook**: what we actually ran, what the numbers were, what we
> learned, and what to do next. *Last updated: 2026-07-27.*

**One-line verdict:** the memory mechanism **works** — the fact module does real long-range
key→value binding at **0.925** accuracy where the grammar base scores chance — but the
capability is (a) **destroyed by composition** (0.925 → 0.003 once packed) and (b) **totally
non-transferable**, collapsing to chance the moment either the prompt template or the value
vocabulary changes. V12 proved the architecture can bind and proved the packaging and data
design around it cannot ship that binding.

**The two defects, stated separately, because they have different fixes:**

| | What it is | Status |
|---|---|---|
| **D1 Composition** | Modules are trained against a live shared embedding table; no single table can satisfy two modules at once | Structural, **not** a patchable bug — see [Tier 0 result](#tier-0-result-composition-is-not-recoverable-by-picking-a-better-table) |
| **D2 Transfer** | The fact module memorized one template × 50 values; both axes independently collapse it to chance | Data design, not architecture — see [transfer grid](#the-transfer-grid-2026-07-27) |

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

### What did not work

- **The module interface** (findings 1–3, 7) — the blocking defect, and structural rather
  than a bug.
- **Fact-data design** (finding 8) — one template × 50 values produced a memorized surface
  pattern with zero transfer on either axis.
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
| **Freeze shared params** — `--freeze_embeddings` on every non-base stage | Exact | Confined to the base's representation space | Works, but base quality becomes a hard ceiling for every module |
| **Ship shared-param deltas** — module owns a delta `pack.py` applies | Exact per module | Full | Breaks: two modules with conflicting deltas cannot both apply. This run is the counterexample |
| **Per-module adapters** — module never touches shared params, owns a small learned in/out projection on the residual stream at its boundary | Exact | Local to the module | The one that scales; conflicts vanish because adaptation lives inside the module's own blocks |

**Recommended sequence:** freeze first (one flag, un-blinds every downstream experiment
immediately), then build adapters as the real interface. Adapters fit the existing
`layer_specs` / `attach_mode` schema without a new concept.

Also required regardless of which option wins:

1. Extend `substrate_hash` to cover the shared params a module was trained against, so a
   mismatch is a resolver conflict instead of a silent capability loss.
2. Add the lossless-pack assertion over **published registry modules** to `v12.selftest`.
   It currently only round-trips models it constructs itself, which is how this survived.
3. Re-run the fact stage with frozen shared params and confirm binding survives packing
   (0.925 trained → 0.925 packed). That single number validates or kills the registry
   direction.

### Tier 1 — fix the fact data (D2)

The mechanism works; the data taught a surface pattern. Before any further architecture work:

1. **Diversify both axes.** Many templates (varying the record/query phrasing, separators,
   numbering, ordering) and a much larger value vocabulary, with a held-out split on *each*
   axis so transfer is measured during training rather than discovered afterwards.
2. **Make the transfer grid the training-time metric.** `v12.diagnose_fact_shortcut --grid`
   already reports the 2×2; a run whose off-diagonal cells sit at chance is not learning
   binding no matter how good its loss curve looks.
3. **Restore the anti-shortcut pressure.** `fact_contrastive_lambda` was cut 0.5 → 0.1 and
   `tau` doubled to stop a NaN divergence ([losses.py:105](losses.py)), for numerical reasons
   rather than scientific ones. Re-test at full strength now that abort-on-NaN and the
   lower LR are in place.
4. **Then, and only then, re-run delta vs additive.** Finding 9 says the current task cannot
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

**Continue the direction, change the interface.** The two things that had to be true for a
module marketplace turned out to be true: the mechanism can learn a real skill (0.925
binding), and the packaging machinery composes arbitrary stacks correctly. What failed is the
*contract* between them — sequential training on a live shared substrate produces modules
that are mutually exclusive by construction, and no packing policy can reconcile them after
the fact.

That is a solvable problem with a known shape (adapters), not a refutation of the premise.
The thing to stop doing is training specialists that quietly rewrite the substrate everyone
else depends on.

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
