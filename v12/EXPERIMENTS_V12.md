# V12: Depth-growth curriculum + playable module registry — first end-to-end run

> **Design source of truth is [README.md](README.md)** (M1–M5 mechanisms, presets, CLI).
> This file is the **lab notebook**: what we actually ran, what the numbers were, what we
> learned, and what to do next. *Last updated: 2026-07-27.*

**One-line verdict:** the M4/M5 machinery (grow → compact → publish → resolve → pack) works
end to end and the depth-growth idea shows a real gain **while training** (Wiki 388 → 314),
but **composition is not identity-preserving** — the packed stack is never the model we
trained — and the fact module produced **no measurable recall** in any configuration. V12 is
a successful *infrastructure* run and an inconclusive *architecture* run.

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

## Findings

| # | finding | evidence | implication |
|---|---|---|---|
| 1 | **Composition is not identity-preserving** — the packed stack is never the trained model | reasoning 314.39 → 479.08 packed; fact 4042.78 → 393.50 packed | Blocks the whole marketplace thesis until fixed. Registry, resolver, hash verify and packer all behaved correctly; the bug is the **shared-param policy** |
| 2 | **Root cause: specialist stages retrain the shared embeddings, packing discards them** | `V12LM.freeze_layers` iterates `self.blocks` only ([model.py:1551](model.py)); `train_curriculum.sh:215` passes `--freeze_layers base` and never `--freeze_embeddings`; trainable params 62,282,900 = 4 grown layers **+ the full 38.6M embedding table**. `pack.py:45` takes shared params only from base | Drift vs `grammar@1.0`, rel-Frobenius: fact stage `embed_real` **0.091** / `embed_imag` **0.104**; reasoning stage **0.376** / **0.282**. Blocks were trained against embeddings that packing throws away |
| 3 | **The fact stage catastrophically forgot web text** | Wiki 388.12 → **4042.78** after 300M tokens of synthetic fact data with unfrozen embeddings | Training a specialist on narrow synthetic data while the shared table is live destroys the base. Packing masked this by restoring grammar's embeddings |
| 4 | **Depth growth genuinely works — on real data, in trained form** | reasoning stage 12L: Wiki 388.12 → **314.39**, DCLM 121.95 → **99.66** | The M4 premise ("depth gives everything") is supported. The gain is real and it is lost only at pack time. Fixing finding 2 should recover it |
| 5 | **Stack order is currently unmeasurable** | four 12L arms within 1.4% (473.69–480.34); `gr` (substrate violated) ≈ `gfr` (canonical) | Finding 1 dominates any ordering signal. The permutation benchmark must be **re-run**, not interpreted. The current numbers say nothing about swappability |
| 6 | **The fact module produced no measurable recall, in any configuration** | `sa@1024`: grammar-only 0.100, fact trained 0.067, `gf` packed 0.100, reasoning trained 0.078, all five packed arms 0.100–0.111. Chance = **0.125** | At or below chance everywhere. Packing is not hiding a working fact module — there is no working fact module. This is **independent of** findings 1–2 |
| 7 | **The fact objective was learned but did not transfer** | answer-masked val PPL 2.5745 (delta) / 1.8895 (additive) vs recall at chance | The model solved the training distribution and learned nothing that generalizes to held-out KEYS/VALUES. The `ce_fact` + `fact_data` design is the thing to question, not the budget |
| 8 | **Delta write lost to additive on its own objective** | 2.5745 vs **1.8895** answer-masked val PPL, identical data/loss/budget | V11 Stage-6 hypothesis #1 (error-correcting writes fix interference) **not supported** at this scale. Caveat: one confounded pair, and per finding 6 both are at chance downstream, so the comparison is weak evidence, not a refutation |
| 9 | **M1 head-gate pruned nothing — the mechanism is untested, not disproven** | 16/16 heads open in all 12 layers, every arm ([../logs/v12_perm/*/headgate.log](../logs/v12_perm/)); `01_compact.log`: 0.0% smaller, `max\|Δlogits\| = 0.000e+00` | With `head_gate_l0_lambda=0.001` and `head_gate_init_logalpha=3.0` no gate ever closed. "Learnable head count" has no result yet |
| 10 | **We skipped our own advice on micro-tests** | V11 Stage-6 item 4: *"capacity micro-tests — 10M-param models, ~30M tok, before spending 300M+ token budgets"*. V12 spent 1.1B tokens / ~45 GPU-hours | Every finding above could have been caught for <2% of that cost |
| 11 | **Absolute quality is smoke-scale; no architecture conclusion is safe** | 4L base at Wiki 388 vs V11 E3 K=3 at **25.77** (~100M, 10 ep) | These are pipeline-validation runs. Generation loops degenerately in every arm ("means gold … means gold"), consistent with heavy undertraining |
| 12 | **Eval/infra papercuts that cost real time** | `eval_recall` default `--context-lengths` includes 2048 > `max_seq_len=1024`, so the headline silently reported `n/a` for every eval until the 2026-07-27 sweep — compounded by a ternary that swallows the label ([eval_recall.py:183](eval_recall.py)); `_HOLDOUT_CACHE_VERSION` NameError; OOM at seq 2048/batch 18; `pack.py` `TypeError` on the `module_card` config key; substrate vocab 50257 vs preset 50261; TeeLogger truncation that lost the middle of the fact run | Headline metrics must fail loudly, never degrade to `n/a` |

### What worked

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

- **Composition fidelity** (findings 1–3) — the single blocking defect.
- **The fact module** (findings 6–8) — no recall, and delta lost to additive.
- **M1 head-gate** (finding 9) — no pruning at the configured λ.
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
4. **Micro-test before budget spend** (finding 10). Gate every 300M-token run behind a
   ~30M-token, 10M-param version that clears an explicit threshold.
5. **Headline metrics must fail loudly.** `single_assoc: n/a` (finding 12) should have been a
   hard error the first time it printed.
6. **State chance level next to every accuracy.** Recall of 0.100 reads like a result until
   you notice chance is 0.125.

---

## Next plan of action

### Tier 0 — make composition lossless (blocking; nothing else is interpretable first)

1. Decide the shared-param policy. Two options:
   - **Freeze it.** Add `--freeze_embeddings` (and norms / LM head) to every non-base stage
     in `train_curriculum.sh:215`. Simple, keeps modules genuinely composable, costs the
     specialist some adaptation capacity.
   - **Ship it.** Let a module own a shared-param **delta** that `pack.py` applies, with a
     documented conflict policy when two modules both carry one. Preserves adaptation,
     but "which delta wins" is a real design problem for a marketplace.
2. Extend `substrate_hash` to cover the shared params a module was trained against, so a
   mismatch is a resolver conflict rather than a silent quality loss.
3. Add the lossless-pack assertion over **published registry modules** to `v12.selftest`.
4. Re-publish and re-pack the existing four modules; confirm packed `gfr` returns to ~314
   Wiki. **This validates or kills the whole registry direction and costs almost nothing** —
   the checkpoints already exist.

### Tier 1 — capacity micro-tests before any further budget

10M-param models, ~30M tokens, 100% fact data. Arms: additive / delta / vault /
`write_phase_address`, each with and without the `ce_fact` contrastive term.
**Gate: beat chance (0.125) by 2× on `single_assoc@1024` before any run over 50M tokens.**
Given finding 7, also question the data design itself — held-out KEYS/VALUES may be
unreachable from a value pool of 50 seen only in `Record: X means Y` templates.

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

### Open strategic question

**Does the registry/marketplace direction continue?** The machinery works, but its first
end-to-end composition lost most of the trained quality. Tier 0 answers this cheaply: if
re-packing with correct shared params restores ~314 Wiki, the direction is sound and the
defect was a policy bug. If it does not, the "train specialists separately, compose later"
premise needs rethinking before more budget goes into it.

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
infrastructure and mechanism-plumbing result; the quality bar still belongs to V11.
See [../EXPERIMENTS_V_6_7_8_9.md](../EXPERIMENTS_V_6_7_8_9.md) and
[../v11/EXPERIMENTS_V11.md](../v11/EXPERIMENTS_V11.md).
