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
