# V12: Learnable Phase-Band Heads + Depth Growth + Module Registry

V12 is a leaner clone of V11’s proven core, plus a stack of novel mechanisms that
share one primitive: a **max head budget `n_heads` (H_max)** where each head slot
is individually gateable, freezable, and specializable — then grown as **depth**
(specialist layer groups) and shipped as **registry modules** that compose into
one inference checkpoint.

This README is the design source of truth (it supersedes the three V12 plan files:
phase-band heads, depth-growth framework, playable module system).

> **Results and learnings live in [EXPERIMENTS_V12.md](EXPERIMENTS_V12.md)** — the first
> end-to-end curriculum run (1.1B tokens, 4 modules) and the stack-order benchmark.
> **Composition fidelity is fixed as of 2026-07-28: always pass `--freeze_shared` on
> non-base stages** (see M5), which makes a packed stack bit-for-bit identical to the trained
> one. Checkpoints built before that flag existed are unreliable. The open question is now
> the *capability*, not the packaging: freezing costs 29× val PPL, and whether the fact band
> binds generally or only memorizes a small closed set is undecided.

---

## Why V12 (the core problem)

V11’s recall stalled around ~0.20–0.23 @2048 (vs Mamba ~1.0) because of **write
interference** in the additive `S += V⊗K*` superposition. Training-side levers
(gamma_floor, recall curriculum, routing competition, gate-surprisal) were already
exhausted. V12 therefore changes **architecture + write mechanism**, not just
training:

| Mechanism | What it does |
|-----------|--------------|
| **M1** Learnable head count | L0-pruned gates over an over-provisioned H_max |
| **M2** Frozen-head growth | Activate previously gated-off head slots per curriculum stage |
| **M3** Fact-band writes | Low-decay vault + phase-addressed + delta write for facts |
| **M4** Depth growth | Attach specialist **layer groups** (facts, reasoning, bio, code…) in any order |
| **M5** Module registry | Ship each group as a versioned module; resolve + pack for inference |

Shapes stay uniform (`[B, H_max, T, d, 2]`) so the fused E3 kernel and `[H,d,d]`
state layout still work — gated heads are just zeroed.

```
Token hidden [B,T,dim,2]
  → dense QKV to H_max * head_dim
  → per-head hard-concrete gate z_h (+ optional write phase address)
  → H_max PAM slots (uniform width; some gated off)
  → gate-weighted merge + o_proj
  → [B,T,dim,2]
```

**Note on “heads by phase”:** a *constant* per-head phase is a gauge freedom that
`o_proj` can absorb. The non-vacuous form is M3’s **content-dependent per-head
phase bands** (`write_phase_address`), not a static phase center per head.

---

## Lean core (kept from V11 / dropped)

**Kept (proven winners):**
- E3 K=3 multistate phase-interference retrieval (**fused path only**)
- Phase-aware GSP write gate (`gate_content_aware=True`)
- `fused_e3` + chunked CE, RoPE, `fused_qkv`, CGU→PAM block, tied complex LM head
- Gate-surprisal recall aux (GSL=0.3, GST=0.5)

**Dropped (dead / rejected in V11):**
- E1 per-channel decay, non-fused E3 loop (reference only inside selftest if needed)
- Competitive routing (`routing_content_aware` / `state_compete` / `route_balance`)
- Flash-PAM, PAMFormer order, `qk_norm`, learned positions
- Duplicate/experiment presets

**Promoted to first-class (for M3):** `vault_state`, `write_phase_address`, and a
compile-safe delta write (`@torch.compiler.disable` on the triangular solve).

Primary modules: `model.py`, `train.py`, `complex_ops.py`, `triton_kernels.py`,
`fused_ce.py`, `losses.py`, `compact.py`, `pack.py`, `registry.py`, `publish.py`,
`eval_checkpoints.py`, `generate.py`, `selftest.py`. Scripts live under
`v12/scripts/` (not top-level `scripts/`).

---

## M1 — Learnable head count (`head_gate`)

`n_heads` is a **MAX budget**. Each slot gets a **hard-concrete L0 gate**
(`HardConcreteGate`, Louizos et al.) applied at the output merge; the expected-L0
penalty (`head_gate_l0_lambda`) rides the trainer aux hook so unused slots prune
to exactly 0. Effective head count is learned. Kernel-safe: shapes stay
`[B, n_heads, T, d, 2]`; a pruned head contributes 0.

After training, **compact** drops closed slots into a slim inference checkpoint
(per-layer `n_heads`), preserving logits (`v12/compact.py`).

```bash
# discover effective head count from an over-provisioned budget of 10
.venv/bin/python -m v12.train --preset v12_headgate_hmax10 --stage pretrain \
  --dataset pretrain_mix --batch_size 2 --seq_len 1024 --head_gate_l0_lambda 0.001
# watch model.head_gate_report() -> per-layer active head counts

.venv/bin/python -m v12.compact --checkpoint headgate.pt --out slim.pt --threshold 1e-3
```

Validate alone at ~100M on WikiText-103: target ≤ V11’s ~25.77 with effective heads
emerging near 6–8. Fallback if L0 destabilizes: fixed H_max without the L0 penalty.

---

## M2 — Progressive frozen-head growth (within a fixed depth)

Grow one over-provisioned model in stages; each stage **opens** a new block of head
slots, **freezes** earlier ones (per-head gradient-mask hooks on QKV rows, `o_proj`
columns, `dt_proj` / protect / phase rows), and trains only the new slots. Reserved
slots are hard-closed until their stage.

Run frozen stages with `weight_decay=0` so decoupled AdamW does not drift frozen
slices. Optional: `--freeze_embeddings` / `--freeze_cgu` after early stages to
protect token geometry.

```bash
# stage B: train head slots 6:8, keep 0:8 contributing, freeze 0:6 + embeddings
.venv/bin/python -m v12.train --preset v12_grow10 --stage pretrain \
  --dataset pretrain_mix --resume_from stageA.pt \
  --active_heads 6:8 --open_heads 0:8 --freeze_embeddings --freeze_cgu --weight_decay 0
```

`--active_heads lo:hi` (trainable), `--open_heads lo:hi` (contributing).
M2 grows *within* a fixed layer count; for growth in **depth** across skills, see M4.

Typical curriculum sketch: grammar warmup (web-only / `blend_warmup_tokens`) →
facts (open fact head slots) → reasoning (open more slots) → new domains.

---

## M3 — Low-interference fact writes (`v12_factband`)

Attacks the additive-superposition interference ceiling:

- **Vault state** (`vault_state`): pin one memory state to no-decay (γ≈1) for
  long-horizon facts.
- **Per-head phase bands** (`write_phase_address`): each head learns its own
  content-dependent key/query→phase map, so heads write/read in distinct phase
  bands (reduces cross-talk). This is the non-vacuous form of “heads by phase”.
- **Delta write** (`write_mode='delta'`): error-correcting write erases a stale
  association before writing; the triangular solve is `@torch.compiler.disable`d
  (eager island) so it is compile-safe.

Grammar/reasoning heads can keep fast-decay additive write; fact bands get the
vault + phase-address (+ optional delta) rules.

```bash
.venv/bin/python -m v12.train --preset v12_factband --stage pretrain \
  --dataset pretrain_mix --batch_size 2 --seq_len 1024
```

### The fact module (purpose-built data + loss + delta fact-bands + eval)

Web pretrain never *demands* long-range recall, so the state never learns durable
key→value binding. The fact module is a **frozen-base grown group** designed
specifically for facts — its own memory mechanism, data, and loss:

- **Delta fact-band group** (`v12_factband_dyn`): single-state `write_mode='delta'`
  (error-correcting) + `vault_state` (no-decay) + `write_phase_address` + dynamic
  head budget (`head_gate`, H_max), grown on the K=3 additive grammar base.
  `write_mode='delta'` only runs at `n_states=1`, so fact bands are single-state
  delta layers stacked on the multistate substrate. **Vault now composes with
  delta** (the single-state γ path honors `vault_state_idx`), so a fact band is
  error-correcting *and* non-forgetting. `FACT_MODE=additive` A/Bs the same
  data/loss against an additive+vault group.
- **Purpose-built data** (`v12/fact_data.py`, `--dataset fact`): store-then-query
  documents — several `Record: <key> means <value>.` bindings + hard-negative
  distractors (extra keys, reused values) + a long filler gap, then
  `Query: <key> means <value>` probes. Only the **value token** is supervised
  (`loss_mask`), so the loss is a direct recall objective (predict from memory,
  not local context). Value vocab is **disjoint** from `memory_probes/behavioral.py`
  so the eval stays held-out. Map-style + generated per index (memory-light,
  shuffleable). Sized from `--token_budget`.
- **Dedicated loss** (`ce_fact` in `v12/losses.py`): answer-masked CE (from the
  loader’s `loss_mask`, already threaded through the trainer) + gate-surprisal
  write program + an **in-batch hard-negative contrastive** term
  (`fact_contrastive_lambda`, `V12LM.fact_contrastive_from_lm`): at each value
  token the correct value must outrank the *other* answer tokens in the batch.
  The recall aux only lives on the fused-CE path, so `v12.train` **auto-enables
  `--fused_ce`** when a recall objective is on (and warns).
- **Recall eval** (`v12/eval_recall.py`): loads a `V12LM` (grammar base, fact
  module, or packed stack) and scores `single_assoc@2048` (+ the full grid) via
  `memory_probes/behavioral.py` — directly comparable to the V11/Mamba numbers.
  Wired into `v12/scripts/eval.sh` (`RECALL=0` to skip).

```bash
# fact stage: masked recall data + ce_fact + delta fact-bands on frozen grammar
FACT_MODE=delta BATCH=2 SEQ=1024 TOKEN_BUDGET=150000000 \
  v12/scripts/train_curriculum.sh fact_retrieval
# A/B: additive+vault fact group, same data/loss
FACT_MODE=additive ... v12/scripts/train_curriculum.sh fact_retrieval
# measure recall
.venv/bin/python -m v12.eval_recall --checkpoint packed_v12/model.pt
```

---

## M4 — Depth-growth framework (attachable specialist layer groups)

**Confirmed direction:** assume **depth gives everything** (grammar, reasoning,
nuance) → default composition is **always-on sequential**. Keep block-application
and packaging **ablation-ready** so MoE routing is a later drop-in; pack-time
config decides which groups attach `sequential` vs `moe`.

Train a small grammar **base** (few layers, little data), then grow specialist
**layer groups** on top — facts / fact_retrieval, reasoning, math, bio, code, then
sub-specialists (C, Java) — in a **swappable order**. Each stage freezes everything
below and trains only the new layers on that skill’s data, with its own objective.

### `layer_specs` (the manifest)

`V12Config.layer_specs: Optional[List[dict]]` is a per-layer manifest of:

- **Structural overrides** (inherit top-level cfg when omitted): `n_heads`,
  `head_dim`, `n_states`, `write_mode`, `vault_state`, `write_phase_address`,
  `head_gate`, …
- **Provenance / curriculum:** `skill`, `group_id`, `stage`, `frozen: bool`,
  `substrate_hash` (hash of the frozen prefix it was grown on)
- **Composition:** `attach_mode: 'sequential' | 'moe'` (default `sequential`;
  `moe` reserved)

When `layer_specs is None`, the stack is the uniform `n_layers` build
(**bit-identical** to pre-M4). The manifest rides in the checkpoint `config`, so
eval/generate rebuild grown stacks automatically.

### Growth API (`V12LM`)

| Method | Behavior |
|--------|----------|
| `grow_layers(specs)` | Append blocks (preserve base indices); stamp `substrate_hash` = hash of frozen prefix |
| `freeze_layers(idx)` | `requires_grad=False` on those blocks (excluded from optimizer — no wd drift) |
| `layer_manifest()` | Structured specs + positional `layer_idx` |
| `_hash_blocks(indices)` | SHA-256 over params of named blocks (used for `substrate_hash`) |

**Composition:** `_apply_blocks` walks groups. `sequential` runs inline;
`attach_mode='moe'` calls `_apply_moe_group`, which **currently falls back to
sequential** (`TODO(moe-router)`). Schema is recorded so a future router is additive.

### Pluggable stage loss (`v12/losses.py`)

`--stage_loss {ce, ce_recall, ce_fact, ce_prune}` applies per-stage config
profiles the trainer already honors (`ce_fact` = answer-masked CE + gate-surprisal
+ hard-negative contrastive for the fact stage). Register more with
`register_stage_loss`.

### Train CLI (raw depth growth)

```bash
.venv/bin/python -m v12.train --preset v12_base_grammar --stage pretrain \
  --dataset pretrain_mix --resume_from base.pt \
  --grow_layers "facts:4" --freeze_layers base --freeze_shared \
  --stage_loss ce_recall --attach_mode sequential
```

Flags: `--grow_layers` (`skill:count[:head_budget]` or `@path.json`),
`--layer_head_budget`, `--freeze_layers` (`base` / `lo:hi` / list),
`--stage_skill`, `--stage_loss`, `--attach_mode`,
**`--freeze_shared`** (freeze the embeddings / norms / LM head that `pack` takes from the base
— required for the packed stack to equal the trained one; see the M5 note),
`--fact_value_pool N` (`--dataset fact`: cap the value vocabulary; eval must use the same cap).

### Compaction + hand-written pack

```bash
.venv/bin/python -m v12.compact --checkpoint headgate.pt --out slim.pt --threshold 1e-3

# Hand-written compose spec (base + modules → one inference ckpt)
.venv/bin/python -m v12.pack --spec compose.json --out packed.pt
```

Compose spec shape:

```json
{
  "base": "checkpoints/.../base/best_model.pt",
  "modules": [
    {"checkpoint": ".../facts/best_model.pt", "group_id": "facts"},
    {"checkpoint": ".../code/best_model.pt", "group_id": "code", "attach_mode": "moe"}
  ]
}
```

The full playable curriculum (dynamic heads + compact + **publish to registry**)
is in `v12/scripts/train_curriculum.sh` (see M5).

---

## M5 — Playable module system (registry + resolver)

A trained layer-group becomes a shippable **module**: only its own layers + a
**card** declaring identity and dependencies. The base/root module also carries
shared params (embeddings, norms, LM head). Composition stacks modules in
dependency order into one inference checkpoint — the foundation for a later
marketplace (different authors contributing grammar / facts / bio experts, etc.).

### Dependency modes

| Mode | Meaning |
|------|---------|
| **`prelayer`** | Needs specific frozen substrate module(s) stacked beneath. Resolver includes them and verifies `substrate_hash` (hash of the concatenated substrate at train time) against the resolved stack. Mismatch → conflict (`--force` to override). |
| **`finetuned`** | Absorbed its base into its own weights → **standalone** (not stacked). Dependency is lineage/provenance only. |

Example (bio expert): may require `grammar_by_x` + `reasoning_by_y` + `fact` +
`bio_base` as **prelayer** stack; or a finetune of `bio_base` that needs no prior
checkpoints at inference.

### Card schema (`ModuleCard` / `Requirement`)

**`Requirement`:** `module_id`, `version_spec` (semver-lite: `*`, exact, `>=x,<y`),
`mode` ∈ {`prelayer`,`finetuned`}, optional `pinned_hash`.

**`ModuleCard`:** `module_id`, `version`, `provenance`, `role` ∈ {`base`,`group`},
`skill`, `group_id`, `attach_mode`, `n_layers`, `layer_specs`, geometry
(`dim`,`head_dim`,`vocab_size`), `self_hash`, `substrate_hash`,
`requires: List[Requirement]`, `created`, `description`.

Cards are stored **both** as sidecar `<ckpt>.card.json` **and** embedded in the
checkpoint config under `module_card` (survives file moves). Reads prefer sidecar,
fall back to embedded.

### Registry (`v12/registry.py`)

Local package-manager-style store (no network fetch yet):

- Root: `v12_registry/` (override `--registry`)
- Layout: `v12_registry/<module_id>/<version>/model.pt(+.card.json)`
- `index.json`: `module_id → {version → {ckpt, card, role, skill, created}}`
- API: `add`, `list`, `get(id,ver)`, `find(id, constraint)` (highest satisfying version)

### Resolver (`resolve` / `resolve_stack`)

1. Walk `requires`; **version-solve** (highest satisfying; raise on incompatible
   constraints for the same `module_id`).
2. **Topological order** (deps first; shared substrates deduped).
3. `prelayer` → stack; `finetuned` → lineage only.
4. **Verify:** geometry match; exactly one `role=base` at bottom; recompute
   `substrate_hash` over resolved prelayer blocks vs each card’s recorded hash.
5. Return `ResolvedPlan` (ordered `ModuleRef`s + report). `--force` downgrades
   conflicts to warnings.

### Publish (`v12/publish.py`)

Extract a stage’s grown blocks into a slim module checkpoint, build the card
(`substrate_hash` from grown specs / `layer_manifest()`), write sidecar + embed,
`registry.add`. Base/grammar publishes `role=base` (keeps shared params + blocks).

```bash
.venv/bin/python -m v12.publish --checkpoint ckpts/grammar/slim.pt \
  --module_id grammar --version 1.0 --role base --provenance "x lab"

.venv/bin/python -m v12.publish --checkpoint ckpts/fact/slim.pt \
  --module_id fact_retrieval --version 1.0 --role group --group_id fact_retrieval \
  --requires "grammar@>=1.0:prelayer" --provenance "x lab"
```

### Registry-aware pack (`pack_from_registry`)

```bash
.venv/bin/python -m v12.pack --target reasoning --constraint '*' \
  --registry v12_registry --out packed/model.pt
```

Resolves the target, then assembles base shared params + renumbered group blocks
into one inference checkpoint (per-group `attach_mode` stamped). Hand-written
`--spec` path remains.

> **FIXED — pass `--freeze_shared` on every non-base stage (2026-07-28).**
>
> *The defect:* shared params (embeddings, norms, LM head) are taken **only from the base
> module**, but `--freeze_layers base` freezes *blocks only* — so every specialist stage
> silently retrained the shared table and pack then discarded it. The cost was total, not a
> few PPL points: the fact module scored **0.925** on in-distribution key→value binding as
> trained and **0.003** once packed. Training the reasoning stage on top destroyed it too
> (0.000) *even though the fact blocks were frozen*, because the table they read from moved.
> And there was no "right" table to pack with — same `gfr` blocks, grammar's params give Wiki
> 479 / binding 0.003, reasoning's give 314 / 0.000, the fact stage's give 5098 / 0.940.
>
> *The fix:* **`--freeze_shared`** (or `FREEZE_SHARED=1` for
> [scripts/train_curriculum.sh](scripts/train_curriculum.sh)) freezes every param `pack` takes
> from the base. A full fact-stage run then produced a packed checkpoint **bit-for-bit
> identical** to the trained one — 0 tensors differ, all four transfer-grid cells match.
> Composition is now the identity.
>
> **Do not use `--freeze_embeddings` for this.** It was inert on this path: only honoured
> inside the `--active_heads` branch (which the curriculum never sets), and it covers
> `self.embed` alone while `pack` overwrites six prefixes.
>
> *Two caveats.* (1) Freezing the shared params also freezes the **tied LM head**, which cost
> **29×** answer-masked val PPL (29.5 unfrozen vs 857.3 frozen) — per-module adapters owning
> an output projection are still the real interface. (2) `substrate_hash` still does not cover
> shared params; `v12.publish` now *warns* on drift (pack cannot — it strips shared params
> from `role=group` modules), but nothing enforces it. Treat any packed checkpoint built from
> a stage trained without `--freeze_shared` as unreliable.
> Full evidence: [EXPERIMENTS_V12.md](EXPERIMENTS_V12.md).

### Train integration (`--substrate`)

Registry-driven alternative to `--resume_from`:

```bash
.venv/bin/python -m v12.train --preset v12_grammar_dyn --stage pretrain \
  --substrate "grammar@>=1.0,fact_retrieval@>=1.0" --registry v12_registry \
  --grow_layers "reasoning:4:16" --layer_head_budget 16 \
  --head_gate --write_phase_address --freeze_layers base \
  --stage_loss ce --batch_size 2 --seq_len 1024
```

Grown groups get dynamic heads per layer (`head_gate` + `write_phase_address`);
compact after training, then publish.

---

## Presets

| Preset | Role |
|--------|------|
| `v12_e3_k3` / `_chat` / `_recall` | Lean core (~100M) + chat vocab / recall aux |
| `v12_e3_k3_headgate` | Core + L0 gates (6 slots) |
| `v12_headgate_hmax10` / `v12_headgate_hmax16` | Over-provisioned H_max for M1/compact |
| `v12_grow10` | Fixed 10 slots for M2 active/open staging (L0 off) |
| `v12_base_grammar` | Few-layer grammar base (no head gate) |
| `v12_grammar_dyn` | **Playable curriculum base:** few layers, H_max=16, `head_gate` + `write_phase_address` |
| `v12_factband` | Vault + phase-address + recall aux (M3, additive K=3) |
| `v12_factband_dyn` | **Fact group:** single-state DELTA + vault + phase-address + dynamic heads (grown on grammar) |
| `tiny` / `tiny_e3` / `tiny_delta` / `v12_micro` | Local/CPU smoke |

Specialist groups reuse growth shorthand (`--grow_layers "fact_retrieval:4:16"`)
rather than separate full-model presets.

---

## Scripts (`v12/scripts/`)

Top-level `scripts/run_v12_stage.sh` is **retired** (kept as a pointer). Use:

```bash
# Curriculum: grammar (dynamic heads) → compact → publish base;
# then grow specialists on published predecessors (swappable ORDER)
v12/scripts/train_curriculum.sh base            # train + compact + publish grammar
v12/scripts/train_curriculum.sh fact_retrieval  # grow DELTA fact-bands on grammar (--dataset fact + ce_fact)
v12/scripts/train_curriculum.sh reasoning       # grow on grammar + fact_retrieval
v12/scripts/train_curriculum.sh all             # every stage in ORDER
FACT_MODE=additive v12/scripts/train_curriculum.sh fact_retrieval  # A/B partner
v12/scripts/train_curriculum.sh fact_retrieval --dry

# Registry / compose / eval
v12/scripts/registry.sh list
v12/scripts/registry.sh resolve reasoning
v12/scripts/registry.sh stack "grammar@1,fact_retrieval@>=1"
v12/scripts/compose.sh reasoning                # resolve + pack → inference ckpt
v12/scripts/compose.sh --stack "grammar@1,fact_retrieval@>=1"
v12/scripts/eval.sh <ckpt> [more...]            # PPL + open-head report + single_assoc@2048
```

Env knobs (curriculum): `PY`, `REGISTRY`, `VER`, `AUTHOR`, `BATCH`, `SEQ`, `LR`,
`TOKEN_BUDGET`, `HMAX`, `GROW`, `STAGE_LOSS`, `CKPT_ROOT`, `DATASET`/`SRC`/`WEIGHTS`.
Edit the `ORDER` array in `train_curriculum.sh` for facts-first (or other) ablations.
Each stage’s frozen substrate = its published predecessors.

Default batch/seq target a **24GB RTX-4090** (`BATCH=2`, `SEQ=1024`). On a 96GB
server raise batch to 16–32 and seq to 2048.

---

## Correctness

```bash
.venv/bin/python -m v12.selftest
```

Verifies **parallel training form == O(1) recurrent form** for every memory mode,
plus:

- M1 gating / pruning / L0 grad
- M2/M3 per-head freezing
- M4 grown-stack equivalence, uniform fallback identity, grow/freeze/manifest
  round-trip, moe-schema sequential fallback, head compaction
- M5 card round-trip (sidecar + embedded), multi-source resolver ordering,
  version conflict, prelayer `substrate_hash` mismatch (+`--force`),
  finetuned-standalone lineage, end-to-end
  **register → resolve → pack → strict-load → forward**
  (packed logits == original grown model; packed stack parallel == recurrent)
- Fact module: delta+vault+phase+head_gate fact-band parallel==recurrent, grown
  delta fact-band on the additive base (mixed per-layer states) equivalence,
  fact-loader `loss_mask` supervises only value tokens, `ce_fact` overrides +
  contrastive term behavior, delta fact-band compaction (beta_proj) preserves
  logits, and a `v12.eval_recall` smoke

GPU training runs on the user’s 4090/server; selftests are CPU smoke tests.

---

## Eval + matched baselines (apples-to-apples)

The V11 recall baseline compared a *pretrained* Mamba-130m (300B tokens) against a
1B-token V11 and skipped the transformer — not fair. For V12, train matched ~100M
**Mamba + Transformer** on the *same* data/budget and report side by side.

Targets / tools:

- **PPL:** WikiText-103 val + DCLM-edu holdout via `v12.eval_checkpoints` /
  `v12/scripts/eval.sh`
- **Behavioral recall:** `single_assoc@2048` (held-out KEYS/VALUES) via
  `v12/eval_recall.py` — a V12LM evaluator (the V11 `run_memory_behavioral.py`
  can’t load V12). Baseline to beat: Mamba ~1.0, V11 ~0.20.
- **Dynamic-head report:** per-layer open heads (hard-concrete gate threshold)
- **Transformer reference:** `v6/transformer_baseline.py` (~100M GPT-2-style)

```bash
v12/scripts/eval.sh <ckpt> [more...]          # PPL + head report + single_assoc@2048
# or individually:
.venv/bin/python -m v12.eval_checkpoints --checkpoints <ckpt> --labels wiki,dclm
.venv/bin/python -m v12.eval_recall --checkpoint <ckpt>
```

---

## Deferred / marketplace-ready but not built yet

- **Network marketplace fetch** — registry is local only; card schema + version
  solve + hash verify are the foundation for remote packages later.
- **Real MoE router** — `attach_mode='moe'` is recorded and groups still run
  sequentially via `_apply_moe_group`; pack-time config is ready for a router drop-in.
- **Matched baseline training runs** — code/reference exists; the full ~100M
  Mamba/Transformer side-by-side report is an experiment to run, not a code gap.
- **Shared-param policy for modules** — see the KNOWN ISSUE under M5. Either freeze the
  shared table for specialists, or let a module ship a shared-param delta with a documented
  conflict rule, and extend `substrate_hash` to cover it.

---

## Quick mental model

```
grammar (role=base, dynamic heads)
   └─ compact → publish grammar@1.0
fact_retrieval (grow DELTA fact-bands on --substrate grammar@>=1;
                --dataset fact + ce_fact + --fused_ce, dynamic heads)
   └─ compact → publish fact_retrieval@1.0  requires: grammar (prelayer)
reasoning (grow on grammar+fact, ce, dynamic heads)
   └─ compact → publish reasoning@1.0       requires: grammar+fact (prelayer)

compose / pack_from_registry(reasoning)
   → one inference checkpoint [grammar | fact_retrieval | reasoning]
```

Anyone else can publish `bio_base_from_xyz` (prelayer on their preferred stack) or a
`bio_expert` that is a **finetune** (standalone at inference). Same tokenizer +
geometry; resolver enforces the rest.
