# V12: Learnable Phase-Band Heads

Leaner clone of V11 (only the proven winners) plus three unified, novel mechanisms.
All share one primitive: a **max head budget `n_heads` (H_max)** where each head slot
is individually gateable, freezable, and specializable.

## What carried forward from V11 (the lean core)
- E3 K=3 multistate phase-interference retrieval (fused path only)
- Phase-aware GSP write gate (`gate_content_aware=True`)
- `fused_e3` + chunked CE, RoPE, `fused_qkv`, CGU->PAM block, tied complex LM head
- Gate-surprisal recall aux (GSL=0.3, GST=0.5)

Dropped (dead/rejected in V11): E1 per-channel decay, competitive routing
(`routing_content_aware`/`state_compete`/`route_balance`), Flash-PAM, PAMFormer
order, `qk_norm`, learned positions, duplicate/experiment presets. Delta write is
kept as the M3 starting point.

## The three mechanisms

### M1 - Learnable head count (`head_gate`)
`n_heads` is a MAX budget. Each slot gets a **hard-concrete L0 gate** (`HardConcreteGate`)
applied at the output merge; the expected-L0 penalty (`head_gate_l0_lambda`) rides the
trainer aux hook so unused slots prune to exactly 0. Effective head count is learned.
Kernel-safe: shapes stay `[B, n_heads, T, d, 2]`; a pruned head contributes 0.

Note: a *constant* per-head phase is a gauge freedom `o_proj` can absorb, so the
"heads-by-phase" idea is realized where it is non-vacuous - as content-dependent
per-head phase bands in M3, not as a static phase per head.

```bash
# discover effective head count from an over-provisioned budget of 10
.venv/bin/python -m v12.train --preset v12_headgate_hmax10 --stage pretrain \
  --dataset pretrain_mix --batch_size 2 --seq_len 1024 --head_gate_l0_lambda 0.001
# watch model.head_gate_report() -> per-layer active head counts
```

### M2 - Progressive frozen-head growth curriculum
Grow one over-provisioned model in stages; each stage OPENS a new block of head
slots, FREEZES the earlier ones (per-head gradient-mask hooks), and trains only the
new slots. Reserved slots are hard-closed (contribute 0) until their stage.
Run frozen stages with `weight_decay=0` so decoupled AdamW does not drift frozen slices.

Direct control via CLI:

```bash
# stage B: train head slots 6:8, keep 0:8 contributing, freeze 0:6 + embeddings
.venv/bin/python -m v12.train --preset v12_grow10 --stage pretrain \
  --dataset pretrain_mix --resume_from stageA.pt \
  --active_heads 6:8 --open_heads 0:8 --freeze_embeddings --freeze_cgu --weight_decay 0
```

`--active_heads lo:hi` (trainable), `--open_heads lo:hi` (contributing),
`--freeze_embeddings`, `--freeze_cgu`. (M2 grows *within* a fixed layer count;
for growth in DEPTH across skills, see M4 below.)

### M3 - Low-interference fact writes (`v12_factband`)
Attacks the additive-superposition interference ceiling (V11 recall ~0.20@2048):
- **Vault state** (`vault_state`): pin one memory state to no-decay for long-horizon facts.
- **Per-head phase bands** (`write_phase_address`): each head learns its own content-
  dependent key/query->phase map, so heads write/read in distinct phase bands (reduces
  cross-talk). This is the non-vacuous form of "heads by phase".
- **Delta write** (`write_mode='delta'`): error-correcting write; the triangular solve
  is `@torch.compiler.disable`d (eager island) so it is compile-safe.

```bash
.venv/bin/python -m v12.train --preset v12_factband --stage pretrain \
  --dataset pretrain_mix --batch_size 2 --seq_len 1024
```

## M4 - Depth-growth framework (attachable specialist layer groups)
Train a small grammar **base** (few layers, little data), then grow specialist
**layer groups** on top - facts, reasoning, math, bio, code, then sub-specialists
(C, Java) - in a **swappable order**. Each stage freezes everything below and trains
only the new layers on that skill's data, with its own objective.

Spec-driven, non-uniform stack. `V12Config.layer_specs` is a per-layer manifest of
structural overrides (n_heads, n_states, vault_state, ...) plus provenance
(`skill`, `group_id`, `stage`, `frozen`, `substrate_hash`, `attach_mode`). When
`None`, the stack is the uniform `n_layers` build (bit-identical to pre-M4). The
manifest rides in the checkpoint `config`, so eval/generate rebuild grown stacks
automatically.

- **Growth API** (`V12LM`): `grow_layers(specs)` appends blocks (preserving base
  indices; stamps `substrate_hash` = hash of the frozen prefix), `freeze_layers(idx)`
  sets `requires_grad=False` (excluded from the optimizer, no wd drift),
  `layer_manifest()` returns the structured manifest.
- **Composition** is always-on **sequential** depth by default. Groups tagged
  `attach_mode='moe'` are recorded and routed through `_apply_moe_group`, which
  currently runs them sequentially - a forward-compatible seam for a future router
  (`TODO(moe-router)`).
- **Pluggable stage loss** (`v12/losses.py`): `--stage_loss {ce,ce_recall,ce_prune}`
  applies per-stage config profiles the trainer honors; register more objectives
  with `register_stage_loss`.

The full curriculum with dynamic heads + compact + publish lives in
`v12/scripts/train_curriculum.sh` (see the module-system section below). Raw
depth-growth CLI control:

```bash
# direct control
.venv/bin/python -m v12.train --preset v12_base_grammar --stage pretrain \
  --dataset pretrain_mix --resume_from base.pt \
  --grow_layers "facts:4" --freeze_layers base --stage_loss ce_recall --attach_mode sequential
```

**Compaction** (`v12/compact.py`): after L0 pruning, drop closed head slots into a
slim inference checkpoint (per-layer `n_heads`), preserving logits.

```bash
.venv/bin/python -m v12.compact --checkpoint headgate.pt --out slim.pt --threshold 1e-3
```

**Pack** (`v12/pack.py`): assemble ONE inference checkpoint from separate group
checkpoints via a compose spec, recording per-group `attach_mode` (sequential now,
moe reserved).

```bash
.venv/bin/python -m v12.pack --spec compose.json --out packed.pt
```

## Playable module system (registry + resolver)
A trained layer-group becomes a shippable **module**: its own blocks plus a
**card** (`ModuleCard`) declaring identity (`module_id@version`), provenance,
geometry, and an ordered `requires` list. Modules stack in dependency order into
one inference checkpoint — the foundation for a later marketplace.

Dependency modes:
- `prelayer`: the module needs specific frozen substrate module(s) stacked
  beneath it. The resolver includes them and verifies the module's recorded
  `substrate_hash` (hash of the concatenated substrate at train time) against the
  resolved stack. A mismatch is a conflict (`--force` to override).
- `finetuned`: the module absorbed its base into its own weights → standalone
  (not stacked); recorded as lineage/provenance only.

Cards are stored **both** as a sidecar `<ckpt>.card.json` and embedded in the
checkpoint config under `module_card` (self-contained; survives file moves).

- **Registry** (`v12/registry.py`, `Registry`): local store at
  `v12_registry/<module_id>/<version>/model.pt(+.card.json)` with `index.json`;
  `add`/`get`/`find`/`list` + semver-lite version solving (`>=1.0,<2.0`).
- **Resolver** (`v12/registry.py`, `resolve` / `resolve_stack`): walks `requires`,
  version-solves (conflict on incompatible constraints), topologically orders the
  stack (deps first), verifies geometry + single-base + `substrate_hash`.
- **Publish** (`v12/publish.py`): extract a stage's grown blocks into a slim module
  checkpoint, build the card (`substrate_hash` from `layer_manifest()`), write it
  (sidecar + embedded), and `registry.add`. Base/grammar publishes `role=base`
  (keeps shared params + its blocks).
- **Registry-aware pack** (`v12/pack.py`, `pack_from_registry`): resolve a target
  then assemble the ordered stack into one inference checkpoint (per-group
  `attach_mode` stamped per layer). The hand-written compose-spec path stays too.
- **Train integration** (`v12/train.py`): `--substrate "grammar@1,fact@>=1"`
  resolves + assembles + freezes the required stack as the base before
  `--grow_layers`; grown groups get `--head_gate --write_phase_address` so each
  group's head count/phase-bands are learned per layer then compacted.

Scripts (`v12/scripts/`):

```bash
# grammar (dynamic heads) -> compact -> publish base; then grow specialists
v12/scripts/train_curriculum.sh base            # train + compact + publish grammar
v12/scripts/train_curriculum.sh fact_retrieval  # grow on the grammar module (ce_recall)
v12/scripts/train_curriculum.sh reasoning       # grow on grammar + fact_retrieval
v12/scripts/train_curriculum.sh all             # every stage in ORDER (swappable)

v12/scripts/registry.sh list                    # inspect registered modules
v12/scripts/registry.sh resolve reasoning       # show the resolved stack
v12/scripts/compose.sh reasoning                # resolve + pack -> inference ckpt
v12/scripts/eval.sh <ckpt> [more...]            # PPL + per-layer open-head report
```

Edit the `ORDER` array in `train_curriculum.sh` to ablate acquisition order
(e.g. facts-first). Each stage's frozen substrate = its published predecessors.

## Correctness
`.venv/bin/python -m v12.selftest` verifies parallel training form == O(1) recurrent
form for every mode, plus M1 gating/pruning/L0-grad, M2/M3 per-head freezing, and M4
grown-stack equivalence, grow/freeze/manifest round-trip, moe-schema fallback, and
head compaction. The module system adds: card round-trip (sidecar + embedded),
multi-source resolver ordering, version conflict, prelayer `substrate_hash` mismatch
(+`--force`), finetuned-standalone lineage, and end-to-end
register → resolve → pack → strict-load → forward (packed == original,
parallel == recurrent).

## Eval + matched baselines (apples-to-apples)
The V11 recall baseline compared a *pretrained* Mamba-130m (300B tokens) against a
1B-token V11 and skipped the transformer - not fair. For V12, train matched ~100M
Mamba + Transformer on the *same* data/budget and report side by side.

```bash
# PPL (WikiText-103 + DCLM holdout) + per-layer open-head report
v12/scripts/eval.sh <ckpt> [more...]
# or directly:
.venv/bin/python -m v12.eval_checkpoints --checkpoints <ckpt> --labels wiki,dclm
# Matched transformer baseline reference: v6/transformer_baseline.py
```

## Hardware note
Default `--batch_size` (4) and the curriculum script (batch 2 / seq 1024) target a
24GB RTX-4090. On the 96GB server raise batch to 16-32 and seq to 2048.
