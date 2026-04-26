# V8 — Quantum-Logic Core (QLC) Experiments

Running log for the V8 plan
([.cursor/plans/v8_quantum_logic_core_8df73232.plan.md](../.cursor/plans/v8_quantum_logic_core_8df73232.plan.md)).

## 0. Targets to beat

| Baseline                | Val PPL | Notes |
|-------------------------|---------|-------|
| medium-pam-v3 (QPAM)    | 29.95   | WikiText-103, 10 epochs, ~100M params, [`scripts/run_v6_medium_pam_v3.sh`](../scripts/run_v6_medium_pam_v3.sh) |
| Transformer B=6         | 23.13   | Same data/epochs/params budget; from EXPERIMENTS_V6_PART2.md §0 |

V8 wins iff **V8-E-joint ≤ 23.13 PPL on WikiText-103, 10 epochs, ≤105M params**.

## 1. Architecture summary

```
Tokens
  → ComplexEmbed → [V7Block (CGU + QPAM)] × N → ComplexNorm
  → QuantumLogicCore (Probe → Sasaki → OrthoHalt, T_max iters)
  → Tied complex LM head → logits
```

Modules in [`v8/`](.):

- [`v8/qlc/projector.py`](qlc/projector.py) — `SasakiProjectionMemory` (rank-r orthonormal basis + Sasaki update).
- [`v8/qlc/effect_bank.py`](qlc/effect_bank.py) — `EffectAlgebraBank` (M rank-1 effects, top-k probe, InfoNCE-ready).
- [`v8/qlc/halt.py`](qlc/halt.py) — `OrthoHalt` ((α, β, γ) readout) and `MLPHalt` ablation.
- [`v8/qlc/reason_loop.py`](qlc/reason_loop.py) — `QuantumLogicCore` orchestrator (ACT-style pondering).
- [`v8/model.py`](model.py) — `V8LM` (V7 backbone + QLC + LM head).
- [`v8/train.py`](train.py) — Stage-aware trainer (A / B / C, KL anchor, diagnostics).

Unit tests: [`v8/qlc/tests/`](qlc/tests/) — `pytest` runnable on CPU. 40 tests as of skeleton commit.

## 2. Training workflow

The original Stage A / A.5 / B / C plan and its launcher scripts were
retired in favor of a **single end-to-end run from random init** — see §9
("Single-run end-to-end") for the configuration, schedule, and acceptance
criteria, and the launchers
[`scripts/run_v8_e2e_smoke.sh`](../scripts/run_v8_e2e_smoke.sh) /
[`scripts/run_v8_e2e_medium.sh`](../scripts/run_v8_e2e_medium.sh).

The historical Stage A.5 smoke result (TinyStories) and the staged-plan
ablation matrix are preserved below as reference for any future re-runs that
choose to revive the staged workflow.

## 3. Ablation matrix

All rows: WikiText-103, 10 epochs, ≤105M params unless noted. ✗ = not yet run.

| Row                  | Preset                           | rank | M    | top-k | T_max | quantale | OrthoHalt | Val PPL | Mean iter | α    | β    | γ    | Tok/s | Peak VRAM | Notes |
|----------------------|----------------------------------|------|------|-------|-------|----------|-----------|---------|-----------|------|------|------|-------|-----------|-------|
| V8-A passthrough     | `stageA_medium`                  | —    | —    | —     | 0     | —        | —         | ✗       | —         | —    | —    | —    | ✗     | ✗         | Sanity gate (≈ 29.95 ± 0.5) |
| V8-B-r4              | `stageB_r4`                      | 4    | 2k   | 2     | 1     | on       | on        | ✗       | 1         | ✗    | ✗    | ✗    | ✗     | ✗         | |
| V8-B-r8              | `stageB_r8`                      | 8    | 2k   | 4     | 1     | on       | on        | ✗       | 1         | ✗    | ✗    | ✗    | ✗     | ✗         | |
| V8-B-r16             | `stageB_r16`                     | 16   | 2k   | 4     | 1     | on       | on        | ✗       | 1         | ✗    | ✗    | ✗    | ✗     | ✗         | |
| V8-C-M2k             | `stageB_M2k`                     | 8    | 2k   | 4     | 1     | on       | on        | ✗       | 1         | ✗    | ✗    | ✗    | ✗     | ✗         | |
| V8-C-M8k             | `stageB_M8k`                     | 8    | 8k   | 4     | 1     | on       | on        | ✗       | 1         | ✗    | ✗    | ✗    | ✗     | ✗         | |
| V8-D-T2              | `stageB_T2`                      | 8    | 2k   | 4     | 2     | on       | on        | ✗       | ✗         | ✗    | ✗    | ✗    | ✗     | ✗         | |
| V8-D-T4              | `stageB_T4`                      | 8    | 2k   | 4     | 4     | on       | on        | ✗       | ✗         | ✗    | ✗    | ✗    | ✗     | ✗         | Best Stage B candidate |
| V8-F-quantale-off    | `stageB_F_quantale_off`          | 8    | 2k   | 4     | 4     | **off**  | on        | ✗       | ✗         | ✗    | ✗    | ✗    | ✗     | ✗         | Tests Sasaki non-commutativity |
| V8-G-orthohalt-off   | `stageB_G_orthohalt_off`         | 8    | 2k   | 4     | 4     | on       | **off**   | ✗       | ✗         | 0    | 0    | 0    | ✗     | ✗         | Tests (α, β, γ) signal |
| V8-E-joint           | `stageC_T4_joint`                | 8    | 2k   | 4     | 4     | on       | on        | ✗       | ✗         | ✗    | ✗    | ✗    | ✗     | ✗         | Top-2 Stage B + low-LR joint |

## 4. Diagnostics conventions

Per row, in addition to PPL, log:

- **Mean iter**: average ACT-style iterations used (reported by `QLCDiagnostics`).
- **α / β / γ**: averaged orthocomplement masses (sanity check: γ > 0 validates the operational-quantum-logic claim).
- **`align`** *(v8.2)*: mean `|u(psi)^H psi|^2 = α + γ` accumulated over the iteration loop. At random init this sits at the `1/d` noise floor (`~0.003` for `dim=384`, `~0.005` for `dim=192`); the `target_alignment_weight` aux pulls it up. Expected trajectory: `>= 0.05` within an epoch and `>= 0.2` by mid-training. **Leading indicator** for the §9.2 `mean_gamma` row -- if `align` stays at noise floor, gamma cannot rise either.
- **Halt distribution**: yes / no / continue rates (detects degenerate halt collapse).
- **Effective rank of `Π_F`**: should equal `rank` (orthonormality drift sanity).
- **Tok/s** + **peak VRAM** for hardware bookkeeping.
- **Git SHA** + log path under `logs/v8/`.
- **Per-iteration breakdown** *(Phase 1, 2026-04-24)*: when a diag step fires, the trainer also prints one indented line per QLC iteration. Format: `per-iter [t=N] a=... b=... g=... yes/no/c=.../.../... | psi_step_l2=... | bank_overlap_with_prev=...`. Captures per-iter `(α, β, γ)`, halt distribution, `||ψ_t − ψ_{t−1}||²` mean over `[BT, H]`, and Jaccard overlap of the bank's top-k indices vs the previous iteration (`N/A` at `t=0`). Used to decide whether iteration `t > 0` is doing real work or echoing iter 0; see §11 for the decision rules. Plumbed in [`v8/qlc/effect_bank.py`](qlc/effect_bank.py) (`select_top_k(..., return_topk_idx=True)`), [`v8/qlc/reason_loop.py`](qlc/reason_loop.py) (per-iter accumulators on `QLCDiagnostics`), and [`v8/train.py`](train.py) (print block right after the QLC summary line). Zero cost on the hot path: all capture is gated on `return_diagnostics=True`, which only fires at log steps (every ~100 batches).

## 5. Historical Stage A.5 smoke summary (TinyStories, 2026-04-23)

Retained for reference. The Stage A.5 launcher and the related Stage A/B/C
launchers have been removed; the canonical single-run replacement is §9.

| Variant                  | Preset                    | QLC params | Best Val PPL | Tok/s   |
|--------------------------|---------------------------|------------|--------------|---------|
| Passthrough (V7-equiv)   | `smoke_tiny_passthrough`  | 0          | **178.95**   | 225 304 |
| QLC r=4, T_max=2         | `smoke_tiny_qlc_r4_T2`    | 57 678     | **173.40**   | 19 180  |

Key takeaway from that run: `γ` was 0.000 throughout because `unsharp_target`
defaulted to off — see [`AUDIT_V8.md`](AUDIT_V8.md) §1. The §9 e2e preset
(`e2e_tiny_reasoning` / `e2e_medium_reasoning`) sets `unsharp_target=True`,
which produces a non-trivial γ (smoke verified 2026-04-24: γ ≈ 0.01 within
the first epoch on TinyStories, 2k samples).

## 6. Verdicts

To be filled in from the §9 single-run e2e runs. The acceptance criteria
that gate the verdict (γ > 0 sustained, halt distribution not collapsed,
mean iter > 1, Val PPL not catastrophic) are listed in §9.2.

| Outcome   | Trigger                                                                              | Decision |
|-----------|--------------------------------------------------------------------------------------|----------|
| Win       | All §9.2 acceptance criteria met AND Val PPL ≤ V7-equivalent passthrough at matched compute | TBD      |
| Soft win  | γ > 0 sustained AND mean iter > 1, but PPL parity (not improvement)                  | TBD      |
| Kill      | γ collapses to 0 with `unsharp_target=True`, OR mean iter degenerates to 1.0          | TBD      |

| Run                      | Preset                  | Outcome | Notes |
|--------------------------|-------------------------|---------|-------|
| 4450ff0 (2026-04-24)     | `e2e_medium_reasoning`  | **Kill**    | alpha/gamma pinned at `1/d` noise floor, halt collapsed to halt-no; v8.2 fix bundle supersedes (see §10) |
| 18f4de7_dirty (2026-04-24) | `e2e_medium_reasoning` | **Soft evidence / pivot** | v8.2 fixes work mechanically: Val PPL 57.31 → 43.04 → 38.21 over epochs 1-3, roughly V6 pam-v3 trajectory and far behind transformer (30.39 at epoch 3). Per-iter diagnostics show different effects but tiny iter-1 ψ motion, so this motivates V8.3 interleaving (see §12). |

## 7. Implementation status

- **2026-04-23**: V8 skeleton committed: config, model, all four QLC primitives,
  trainer, run scripts, EXPERIMENTS stub. 40 unit tests passing on CPU.
  Stage A/A.5/B/C launch scripts ready; trainer entry point smoke-tested.
- **2026-04-23**: Stage A.5 smoke run on RTX 4090. **GATE PASSED** (QLC beats
  passthrough by 5.55 PPL on TinyStories; see §5). Caveat: γ=0 means the
  algebraic orthocomplement signal is dormant at this scale.
- **2026-04-23 (rethink)**: After multi-AI cross-review, the project was
  reframed (see [`AUDIT_V8.md`](AUDIT_V8.md) and
  [`README_V8.md`](README_V8.md) §0/§11). The current default thesis is
  "constrained latent memory + adaptive compute" with the operational
  quantum-logic claim demoted to an ablation hypothesis.
- **2026-04-24**: Staged Stage A/B/C launchers retired. Single-run e2e
  workflow (§9) becomes the canonical training path: one model, one
  continuous run from random init, with `unsharp_target=True` so γ is
  interpretable. Smoke gate verified: γ > 0 within the first epoch on
  TinyStories.
- **Next:** Launch the medium e2e run on WikiText-103 via
  [`scripts/run_v8_e2e_medium.sh`](../scripts/run_v8_e2e_medium.sh) and
  evaluate against the §9.2 acceptance criteria.
- **2026-04-24 (post-launch)**: First medium e2e run (commit 4450ff0)
  killed early. QLC reasoning signal frozen at the `1/d` noise floor for
  1,200+ steps -- §9.3 hard-kill criterion triggered. See §10 for the full
  diagnosis. The **v8.2 fix bundle** landed:
  - `OrthoHalt.target_gate` init `0 -> 1.5`,
    `cls_head` init `eye*4` -> `eye*1, bias=[0,0,2]` (continue-biased).
    Matching re-init in `V8LM._init_weights`.
  - New `target_alignment_weight=0.05` aux pulls
    `OrthoHalt.target_mlp(psi)` toward `psi` so `|u^H psi|^2` leaves the
    `1/d` floor.
  - `infonce_weight=0.05, infonce_every=4` enabled by default in both
    `e2e_*_reasoning` presets (was 0 before).
  - Schedule retune: `ponder_lambda_phases=(0.0, 0.002, 0.005)` (was
    `(0.0, 0.005, 0.01)`) so the late phase doesn't undo the loop
    engagement once it finally exists.
  - New `align=` column on the QLC diagnostic line; new `mean_amp` field on
    `QLCDiagnostics`.
  Plan: [`.cursor/plans/unfreeze_qlc_alpha_gamma_2132eeda.plan.md`](../.cursor/plans/unfreeze_qlc_alpha_gamma_2132eeda.plan.md).

## 8. (Removed) Discriminator suite

The Stage-B-tier discriminator suite required a Stage A backbone checkpoint
and a coherent staged sweep, both of which were retired with the launchers
on 2026-04-24. The corresponding `stageB_*` / `smoke_tiny_*` presets remain
in [`v8/config.py`](config.py) and can still be invoked manually with
`v8.train --preset <name>` for one-off experiments. The canonical workflow
is the single-run e2e in §9.

## 9. Single-run end-to-end (single-run-v8-training plan v2)

This is the canonical V8 training workflow as of 2026-04-24. The goal is:
**one model, one continuous run from random init**, with the QLC engaged the
whole time so the reasoning signal is interpretable as the model trains.
There is no Stage A handoff, no Stage B sweep, and no KL anchor (which is
incompatible with random init — see plan v2 §1).

### 9.1 Configuration

| Knob                  | Value                | Why                                                    |
|-----------------------|----------------------|--------------------------------------------------------|
| Preset (medium)       | `e2e_medium_reasoning` | Single end-to-end preset built on `_backbone_medium_v3` |
| Preset (smoke)        | `e2e_tiny_reasoning`   | TinyStories sanity gate before medium                  |
| `freeze_backbone`     | `False`              | Backbone learns jointly with the QLC                   |
| `qlc.enabled`         | `True`               | QLC active from step 0                                 |
| `qlc.unsharp_target`  | `True`               | Required for non-trivial γ (`AUDIT_V8.md` §1)          |
| `qlc.out_scale_init`  | `0.05`               | Soft warmup for the QLC residual via low init          |
| `qlc.out_scale_learnable` | `True`           | Optimizer can scale residual up if useful              |
| `qlc.halt_mode`       | `"ortho"`            | OrthoHalt is the only head that produces (α,β,γ)       |
| `kl_anchor_weight`    | `0.0`                | No Stage A reference logits exist in random-init run   |

Runtime-safe schedule (`--qlc_schedule`, off by default), expressed as
fractions of total steps, ramps **only** the three runtime-mutable knobs:

| Phase           | Frac of run | `t_max` | `ponder_lambda` |
|-----------------|-------------|---------|-----------------|
| Warmup          | 0 – 1/3     | 2       | 0.000           |
| Mid             | 1/3 – 2/3   | 3       | 0.005           |
| Late            | 2/3 – 1.0   | 4       | 0.010           |

Boundaries and values are configurable via
`--qlc_warmup_end_frac`, `--qlc_mid_end_frac`,
`--qlc_t_max_phases`, and `--qlc_ponder_lambda_phases`.
Anything that requires rebuilding modules or resetting the optimizer
(`halt_mode`, `out_scale_learnable`, `use_complex`, `rank`, `bank_size`) is
**not** schedulable and must be set in the preset.

### 9.2 Reasoning-signal acceptance criteria

All metrics come from `QLCDiagnostics` (`v8/qlc/reason_loop.py`), already
printed by the trainer at `--diag_every` cadence.

| Metric                       | Threshold                                  | Source / why                                                         |
|------------------------------|--------------------------------------------|----------------------------------------------------------------------|
| `align` *(v8.2, leading)*    | ≥ 0.05 by end of warmup, ≥ 0.2 mid-training | Mean `|u^H psi|^2 = α + γ`. If pinned at the `1/d` noise floor, alpha and gamma cannot rise either; kill the run before the late phase rather than after (this row exists because the 4450ff0 negative result missed every other gate by sitting on this floor for 1,200 steps -- see §10) |
| `mean_gamma`                 | ≥ 0.05 sustained late in training          | Only meaningful because `unsharp_target=True`; collapses to 0 otherwise. Cannot exceed `align`; `align` is the upper bound. |
| `halt(yes / no / cont)`      | None of the three < 0.05 at convergence    | Halt distribution must not collapse to a one-hot regime. v8.2 init makes the default distribution roughly `[0.10, 0.10, 0.80]` (continue-biased) so the loop engages from step 0. |
| `mean_iter`                  | > 1.0 once `t_max` has ramped to 4         | Confirms the loop is actually pondering, not always halting at iter 1 |
| `out_scale` (final)          | Reported, not gated                        | Interpretable per `AUDIT_V8.md` §4 — small ≈ QLC near-bypassed       |
| Final Val PPL                | Not catastrophically worse than V7-equivalent passthrough at matched compute | The model must still *learn* language while the reasoning signal lives |

### 9.3 Hard-kill rules

- If `mean_gamma` stays at 0 with `unsharp_target=True` for the whole late
  phase: the algebraic readout is dead even after the geometry fix.
  Document the negative and stop pursuing the operational quantum-logic
  framing for this preset (do not silently keep training).
- If `mean_iter` collapses to 1.0 in the late phase, the ACT loop has
  degenerated to "always halt at the first iter" and the model is no longer
  doing iterative reasoning. Re-run with a higher `ponder_lambda` ceiling or
  document and stop.
- If Val PPL diverges (NaN, or > 2× the early-training plateau), stop and
  treat the run as a failed configuration; do not auto-resume.

### 9.4 Launchers

- [`scripts/run_v8_e2e_smoke.sh`](../scripts/run_v8_e2e_smoke.sh) —
  TinyStories smoke gate (`e2e_tiny_reasoning`), `--qlc_schedule` on.
- [`scripts/run_v8_e2e_medium.sh`](../scripts/run_v8_e2e_medium.sh) —
  WikiText-103 main run (`e2e_medium_reasoning`), 10 epochs by default,
  no `--backbone_ckpt`, `--qlc_schedule` on.

Decision after the smoke: gamma > 0 sustained AND PPL trending down →
launch the medium run. Otherwise, adjust the schedule (`--qlc_t_max_phases`
or `--qlc_ponder_lambda_phases`) before spending wall-clock on medium.

## 10. 2026-04-24 — e2e medium first launch (4450ff0): alpha/gamma frozen

**Run**: [`logs/v8/e2e_medium_reasoning_wikitext103_20260424_162331_4450ff0/v8_e2e_medium_reasoning_wikitext103.log`](../logs/v8/e2e_medium_reasoning_wikitext103_20260424_162331_4450ff0/v8_e2e_medium_reasoning_wikitext103.log)
| **RUN_INFO**: [`...20260424_162331_4450ff0/RUN_INFO.txt`](../logs/v8/e2e_medium_reasoning_wikitext103_20260424_162331_4450ff0/RUN_INFO.txt)

### Symptom

For the entire 1,200+ steps that ran before the kill:

| Diagnostic                | Observed                              | Note |
|---------------------------|---------------------------------------|------|
| `alpha`                   | `0.001` (constant)                    | Predicted `1/(2d) = 1/768 ≈ 0.0013` for `dim=384`, `target_gate=0` |
| `beta`                    | `0.997 - 0.998`                       | Predicted `1 - 1/d ≈ 0.9974` |
| `gamma`                   | `0.001` (constant)                    | Same as alpha; the gate split `amp_sq ≈ 0.0026` 50/50 |
| `mean_iter`               | `2.00` (= `t_max`)                    | Loop runs to ceiling because halting is forced by softmax saturation, not signal |
| `halt(yes/no/cont)`       | `0.02 / 0.96 / 0.02`                  | One-hot at halt-no -- `cls_head.weight = 4*I` saturates the softmax on `beta=0.997` |
| `mean_gamma` slope        | flat                                  | §9.3 hard-kill triggered |

### Math diagnosis

With `renormalize_psi=True` (`v8/qlc/reason_loop.py` L295-298), `||psi||² = 1`
exactly after every Sasaki update. Inside `OrthoHalt.abg`
(`v8/qlc/halt.py` L161-178):

```
amp_sq      = |u^H psi|²
beta        = (||psi||² - amp_sq).clamp_min(0) = 1 - amp_sq
alpha       = sigma(target_gate) * amp_sq    # if unsharp_target
gamma       = (1 - sigma(target_gate)) * amp_sq
```

At init, `target_mlp(psi)` is essentially a random direction in `d`-dimensional
space, so `|u^H psi|² ≈ 1/d = 1/384 ≈ 0.0026`. With `target_gate = 0`,
`sigma(0) = 0.5`, splitting that mass 50/50 into alpha and gamma:
`alpha ≈ gamma ≈ 0.0013`, `beta ≈ 0.9974`. Exact match to the log.

`cls_head` was init'd with `weight = 4 * I_3, bias = 0`, so the softmax over
`(alpha, beta, gamma)` projected into halt logits saturates: with `beta ≈ 1`
and `weight = 4`, `p(halt-no) = softmax(0, 4, 0)[1] ≈ 0.96`. Halt-no fires
on the first iteration of every step. Subsequent iterations get
`remainder ≈ 0.02` of the ponder weight, so the gradient flowing back to
`target_mlp` and the bank from those iterations is multiplied by `~0.02` --
effectively zero.

There is no `infonce_weight` in this preset (only `stageB_infonce_on`
historically set it), and there is no auxiliary objective anywhere that
rewards `|u^H psi|²` growing. So `target_mlp` has no gradient path off the
random-init direction, `amp_sq` stays at the noise floor, the halt
distribution stays pinned, and the loop is decorative for the entire run.

### v8.2 fix bundle

See [`.cursor/plans/unfreeze_qlc_alpha_gamma_2132eeda.plan.md`](../.cursor/plans/unfreeze_qlc_alpha_gamma_2132eeda.plan.md)
and §0.2 of [`README_V8.md`](README_V8.md) for the full rationale. Five
coordinated changes:

1. `OrthoHalt.target_gate` init `0 -> 1.5` (`sigmoid ~= 0.82`). Routes most of
   `amp_sq` to alpha, ~18% to gamma.
2. `OrthoHalt.cls_head` init `eye*4, bias=0 -> eye*1, bias=[0,0,2]`. Default
   distribution becomes roughly `[0.10, 0.10, 0.80]` (continue-biased). Loop
   actually runs all `t_max` iterations, gradient flows.
3. New `target_alignment_weight` aux (default `0.05` in
   `e2e_*_reasoning` presets) adds
   `target_alignment_weight * (1 - mean(|u^H psi|²)).clamp_min(0)` to
   `aux_loss`. This is the only term that directly rewards `u(psi)`
   aligning with `psi`.
4. `infonce_weight=0.05, infonce_every=4` enabled by default in
   `e2e_*_reasoning` presets so the bank's effects learn entity routing
   in parallel.
5. `ponder_lambda_phases=(0.0, 0.002, 0.005)` (was `(0.0, 0.005, 0.01)`):
   smaller late-phase cap so the loop isn't penalised right when the new
   signal arrives.

### Acceptance gate (must pass on smoke before relaunching medium)

- `align` rises off the `1/d ≈ 0.003` floor, `>= 0.02` by end of smoke,
  `>= 0.05` within an epoch.
- `mean_iter > 1.5` once `t_max=4`.
- `halt(yes/no/cont)` no longer pinned at `0.02/0.96/0.02`; none of the
  three below `~0.05` at convergence.
- `mean_gamma > 1e-3` early, `>= 0.01` by end of smoke.

If any of these fail, do *not* relaunch medium -- iterate on the schedule
or the alignment weight first.

## 11. 2026-04-24 — Phase 1 per-iteration QLC diagnostics: decision rules

**Plan**: [`.cursor/plans/v8_per_iter_diagnostics_72b30338.plan.md`](../.cursor/plans/v8_per_iter_diagnostics_72b30338.plan.md).
**Code**: [`v8/qlc/effect_bank.py`](qlc/effect_bank.py) (`return_topk_idx`),
[`v8/qlc/reason_loop.py`](qlc/reason_loop.py) (`QLCDiagnostics.per_iter_*`),
[`v8/train.py`](train.py) (print block).

### Why this exists

The summary line averages everything across iterations:

```text
QLC: iter=2.00 alpha=0.858 beta=0.000 gamma=0.142 halt(yes/no/cont)=0.39/0.17/0.44 | align=0.9999 ...
```

This hides whether iteration `t > 0` differs from iteration 0 at all. The
v8.2 fix bundle (§10) successfully unfroze `α/γ` (now `align ≈ 0.9999`),
but the side-effect is that iteration 0 already maxes out the alignment
target, leaving no headroom for iteration 1+ to refine. We need per-iter
visibility before deciding which Phase 2 fix is warranted.

### Format on the wire

Below the existing summary line, one indented line per iter actually run:

```text
   per-iter [t=0] a=0.842 b=0.000 g=0.158 yes/no/c=0.31/0.13/0.56 | psi_step_l2=2.130e+02 | bank_overlap_with_prev=N/A
   per-iter [t=1] a=0.873 b=0.000 g=0.127 yes/no/c=0.47/0.21/0.32 | psi_step_l2=4.150e+01 | bank_overlap_with_prev=0.85
```

### Decision rules (precommit; do **not** post-hoc rationalize)

### Current v8.2 medium readout (18f4de7_dirty)

The current medium run lands in the Phase 2B + Phase 2D cells below:

- `bank_overlap_with_prev ≈ 0.04-0.05` at `t=1`, so the bank is selecting
  different effects across iterations.
- `psi_step_l2[t=0] ≈ 3.6e2` but `psi_step_l2[t=1] ≈ 0.9`, so
  `psi_step_l2[t=1] / psi_step_l2[t=0] ≈ 0.0026`. Different effects are not
  materially moving the state.
- `alpha[t=0] ≈ 0.895` with `align≈0.9999`, leaving almost no target-headroom
  for later iterations. This fires Phase 2D: iteration-aware target/halt.
- `beta≈0` this early is also a collapse signal, not a success signal:
  `align=alpha+gamma≈1` leaves no orthocomplement/refetch mass for the halt
  head to use.

Concrete next actions are no longer "observe more" for this preset: run the
cheap discriminator presets, then test `renormalize_psi=False`,
`weighted_projector=True`, bounded `target_alignment_target`, and
iteration/layer-aware QLC in the V8.3 interleaved preset.

#### Axis A — bank overlap × ψ-step magnitude (relative to iter 0)

| `bank_overlap[t≥1]` | `psi_step_l2[t≥1] / psi_step_l2[t=0]` | Interpretation                                                | Phase 2 action                                                                                                                                              |
|---------------------|---------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **> 0.7**           | **< 0.1**                             | Iter `t` picks the same effects and barely moves ψ → decorative. | **Phase 2A — force-different bank picks**: mask top-k from the previous iter, OR add a per-iter diversity bonus to the probe softmax, OR rotate probe head per iter. |
| **> 0.7**           | **> 0.3**                             | Same effects but real ψ motion → projector is refining on the same facts (legit). | None yet. Let the schedule ramp `t_max` to 3, then re-check.                                                                                                |
| **< 0.3**           | **< 0.1**                             | Different effects but ψ doesn't move → projector is washing out new info. | **Phase 2B — increase capacity**: bump `rank` from 4 → 8, or smoke with `renormalize_psi=False`.                                                            |
| **< 0.3**           | **> 0.3**                             | Different effects, real motion → genuine multi-step reasoning. | None. Continue ramp; watch `align` keep climbing past `0.9999` saturation.                                                                                  |

#### Axis B — halt and α-headroom (orthogonal to Axis A)

| Pattern (across `t`)                                                  | Interpretation                                          | Phase 2 action                                                                                                       |
|-----------------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `cont` rate similar at `t=0` and `t=1` (e.g. 0.44 → 0.42)             | Halt has no iteration-aware signal — same logits regardless of `t`. | **Phase 2C — iteration-aware halt**: feed a learned positional embedding for `t` into `OrthoHalt.cls_head` input. |
| `cont` drops sharply (e.g. 0.60 → 0.30) and `yes` rises (0.20 → 0.50) | Halt is genuinely getting more confident as ψ refines.  | None.                                                                                                                |
| `α` already ≥ 0.85 at `t=0` and Δα across iters ≤ 0.02                | Target effect satisfied on iter 0 → no headroom for iter 1+. | **Phase 2D — iteration-aware target**: `target_mlp(ψ, t)` so `u(ψ, t)` shifts and creates new headroom.            |

### When to act

1. **First 1–2 log steps after diagnostics land** — sanity check the lines appear, no NaN, `bank_overlap` is in `[0, 1]`. *No action* expected.
2. **First ~500 batches** of training in QLC schedule phase 0 (`t_max=2`) — collect ≥5 prints at *consistent* values before declaring a verdict. Early steps are noisy.
3. **Decision point** — at end of epoch 1 *or* once the schedule enters phase 1 (`t_max=3`), whichever comes first. With `t_max=3` we get both `bank_overlap[t=1]` and `[t=2]`; matching cells across both rows of Axis A is much stronger evidence than a single iter.
4. **Don't act on a single log line** — require the *same* Axis-A cell to fire for ≥3 consecutive prints (≈300 batches). Otherwise it's noise, not signal.
5. **Acceptance gate for any Phase 2 patch** — must demonstrate, in a 1k-batch smoke, that it moves the diagnostic that triggered it (e.g., Phase 2A drops `bank_overlap` from > 0.85 to < 0.4). If it doesn't, **revert**; don't ship code that didn't move the needle it was designed to move.

### Promotion to AUDIT_V8.md

Phase 1 is empirical instrumentation. A finding becomes audit-grade (and
moves to a new [`AUDIT_V8.md`](AUDIT_V8.md) §12) only when it's a
*structural* claim — e.g., "bank overlap is ≥ 0.95 across all 10 epochs
regardless of input, because `EffectAlgebraBank.probe` is deterministic in
`mean(ψ)` and ψ is renormalized to the unit sphere after every Sasaki
update, so iter 1's probe input is geometrically near-identical to iter
0's." Until we have repeated, schedule-spanning data backing such a claim,
keep the discussion in this section.

### What is NOT in Phase 1

Same exclusions as the plan:

- No code change that affects the model state dict, optimizer, scheduler, or checkpoints. Safe mid-training.
- No change to `v8/model.py`, `v8/config.py`, the launch scripts, or any tests.
- No Phase 2 fixes (force-diversity, iteration-aware halt, iteration-aware target, entropy halt). Those wait on data.
- No Phase 3 (move QLC inside the backbone block per [`AUDIT_V8.md`](AUDIT_V8.md) §3) until cheap fixes are exhausted.

## 12. V8.3 — interleaved context-level reasoning

**Plan**: [`.cursor/plans/v8_context_reasoning_28cfa913.plan.md`](../.cursor/plans/v8_context_reasoning_28cfa913.plan.md).
**Code**: [`v8/config.py`](config.py) (`e2e_*_context_reasoning` presets),
[`v8/model.py`](model.py) (QLC insertion after selected `V7Block`s),
[`v8/qlc/reason_loop.py`](qlc/reason_loop.py) (iteration/layer context,
RoPE-conditioned probe, weighted projector, insertion diagnostics), and
[`v8/train.py`](train.py) (multi-insertion diagnostics logging).

### Why this exists

The v8.2 medium run fixed the mechanical α/γ failure but did not change the
performance class: epoch-3 PPL is roughly tied with V6 pam-v3 and far behind
the transformer. The remaining architectural problem is `AUDIT_V8.md` §3:
post-backbone QLC refines each token after sequence mixing has already
finished. V8.3 moves QLC into the layer stack so memory retrieval can affect
later PAM sequence mixing.

### Presets

| Preset | Dataset intent | Insertions | Final QLC | Key switches |
|--------|----------------|------------|-----------|--------------|
| `e2e_tiny_context_reasoning` | TinyStories smoke | after block 0 | off | `renormalize_psi=False`, `target_alignment_target=0.75`, `use_iteration_context=True`, `use_layer_context=True`, `rope_conditioned_probe=True`, `weighted_projector=True` |
| `e2e_medium_context_reasoning` | WikiText-103 medium | after blocks 3, 7, 11 | off | same switches, shared interleaved QLC, `interleaved_out_scale=0.05` |

### Diagnostics

The trainer now prints one QLC block per insertion point, e.g.
`QLC[layer_3]`, `QLC[layer_7]`, and `QLC[layer_11]`. Each block keeps the
existing per-iteration readout and adds:

- `downstream_delta_l2`: scaled hidden-state change introduced at that
  insertion point.
- `echo_cos`: cosine similarity between this insertion's QLC residual and the
  previous insertion's residual. High positive values mean memory writes are
  echoing through the stack; near-zero values mean each insertion is acting
  independently.

### Required smoke ladder

Run these before any full A100 medium spend:

1. Existing discriminator presets on TinyStories / short WT103 smoke:
   `stageB_real_spm`, `stageB_outscale0`,
   `stageB_equal_flop_passthrough`, `stageB_lmhead_unfrozen`,
   `stageB_halt_delta`, `stageB_halt_entropy`,
   `stageB_quantale_order_off`.
2. `e2e_tiny_context_reasoning` with `--qlc_schedule --diag_every 100`.
3. 1% WT103 run of `e2e_medium_context_reasoning`.
4. Full medium run only if the gates below pass.

### Decision gates

Proceed to full medium only if:

- Interleaved QLC beats post-backbone `e2e_*_reasoning` at matched steps.
- `t>=1` iterations move `psi` by a meaningful fraction of iteration 0,
  not ~0.3%.
- `align` should rise above the 1/d floor without saturating at 1.0; target
  band for V8.3 is roughly `0.6-0.85`.
- `beta` should retain non-trivial mass during early/mid training (`>0.05`
  on average). `beta≈0` with `align≈1` is treated as halt-head collapse, not
  reasoning success.
- `stageB_outscale0` or an interleaved `out_scale=0` control regresses PPL.
- Equal-param / equal-FLOP controls do not match the gain.
- On medium, epoch-2/3 PPL bends closer to the transformer curve than the V6
  pam-v3 curve.

### 2026-04-26 readout: medium context run is muting QLC

**Run**:
[`logs/v8/e2e_medium_context_reasoning_wikitext103_20260425_103755_4069dac/v8_e2e_medium_context_reasoning_wikitext103.log`](../logs/v8/e2e_medium_context_reasoning_wikitext103_20260425_103755_4069dac/v8_e2e_medium_context_reasoning_wikitext103.log)
| **RUN_INFO**:
[`...20260425_103755_4069dac/RUN_INFO.txt`](../logs/v8/e2e_medium_context_reasoning_wikitext103_20260425_103755_4069dac/RUN_INFO.txt)

**Checkpoint sampled**:
`v8/checkpoints/e2e_medium_context_reasoning/best_model.pt`
(`checkpoint_epoch=2`, saved after epoch 3, `global_step=57831`,
`best_val_ppl=38.6564`).

#### Setup

This is the 104.5M-param V8.3 context preset: V7/QPAM medium backbone plus one
shared interleaved QLC after blocks 3, 7, and 11. The final post-backbone QLC
is disabled. QLC settings are `rank=8`, `bank_size=2048`, `top_k=4`,
`t_max=4`, `interleaved_out_scale=0.05`, `out_scale_learnable=True`,
`renormalize_psi=False`, `target_alignment_weight=0.05`,
`target_alignment_target=0.75`, `infonce_weight=0.05`, `infonce_every=4`,
iteration/layer context on, RoPE-conditioned probe on, weighted projector on,
and runtime QLC schedule enabled.

#### Metric trajectory

| Epoch | Train PPL | Val PPL | Tok/s | Notes |
|-------|----------:|--------:|------:|-------|
| 1 | 122.33 | 57.58 | 3386 | best |
| 2 | 53.81 | 43.73 | 3425 | best |
| 3 | 44.73 | 38.66 | 3426 | best checkpoint |

Compared at epoch 3, V8.3 is ahead of lean PAM (`40.15` PPL at epoch 3) but
only by `1.49` PPL while running roughly 10x slower (`~3.4k tok/s` vs
`~34k tok/s`). It is not bending toward the transformer curve: transformer B=3
is documented at `27.08` final PPL, V7 7a at `29.73`, V6 pam-v3 at `29.95`,
and lean L1 at `32.09`.

#### Internal QLC readout

The mechanical QLC signals are not dead: `align` rises into the intended band
or above it, InfoNCE falls from roughly random (`~2.2`) to `~0.1`, and
`bank_overlap_with_prev` is low (`~0.04`), so later iterations are selecting
different effects. However, the part that matters for the language model is
collapsing:

- At startup, QLC `out_scale` is `0.050`, with `downstream_delta_l2` around
  `0.8-1.0`.
- By epoch 2 summary, `out_scale` is still `0.061`, with
  `downstream_delta_l2` around `0.9-1.2`.
- By epoch 3 summary, the saved checkpoint has
  `interleaved_qlc.out_scale = -0.000563`, and train/val
  `downstream_delta_l2` is only `~1e-5` to `1e-4`.
- In the live epoch-4 tail, `out_scale` remains near zero/negative
  (`~-0.003` to `-0.005`), and `downstream_delta_l2` is still tiny
  (`~3e-3` to `1e-2`) despite internal `psi_delta_l2` remaining in the
  hundreds.

Interpretation: QLC is learning internal routing/alignment, but the joint LM
objective is learning to suppress the QLC residual before it reaches later PAM
layers. In other words, the model is using the learnable residual scale as a
bypass valve.

#### Best-checkpoint factual sampling

Generation used the trainer settings (`temperature=0.8`, `top_k=50`,
`top_p=0.9`, `repetition_penalty=1.2`, `max_new_tokens=80`).

| Prompt | Observed completion quality |
|--------|-----------------------------|
| `In 1923 , the University of` | Grammatical but invented: University of Chicago opens first public school in 1924, then closes in 1928. |
| `The capital of France is` | Does not answer Paris; drifts into colonial conflict. |
| `Albert Einstein was born in` | Gives 1872 and fabricated family details; correct is 1879. |
| `The Pacific Ocean is` | Produces generic ocean/port prose with confused Atlantic/Pacific history. |
| `During the Second World War ,` | Repeats "version" and drifts into B-29 section text. |
| `The chemical symbol for gold is` | Does not answer Au; gives generic gold/silver prose. |
| `A prime number is a number that` | Does not define primality; drifts into song structure. |
| `The theory of relativity was developed by` | Does not answer Einstein; says New York Post / 1999. |

Verdict from sampling: no evidence of improved factual grounding. The model is
fluent in WikiText section style, but not better at simple facts.

#### Decision

This run fails the V8.3 continuation gate as currently defined. It has not shown
transformer-ward PPL movement, is far slower than the lean/V7 baselines, and
the core diagnostic (`downstream_delta_l2`) says QLC influence has been muted.

Do not spend a full 10-epoch run on this exact configuration. Preserve it as a
negative result:

> Interleaved QLC can learn alignment, InfoNCE routing, and different bank picks,
> but with `out_scale_learnable=True` the LM objective learns to bypass the QLC
> residual, so the reasoning core becomes internally active but externally
> negligible.

Next experiments should be sharper ablations, not longer training:

1. Freeze or constrain interleaved `out_scale` positive and rerun a short
   smoke. If PPL regresses, QLC residual is harmful; if PPL improves, the
   bypass valve was hiding useful signal.
2. Disable InfoNCE and/or target alignment in short runs to test whether the
   auxiliaries are optimizing internal structure that the LM objective rejects.
3. Run a Stage-B/frozen-backbone test where QLC must prove additive value
   without the backbone adapting around it.
4. If the near-term goal is WikiText PPL, prioritize V7/CGU or
   transformer-like channel-mixing improvements over more QLC complexity.
