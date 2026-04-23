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

## 2. Stage plan & hardware

| Stage | What                                                        | HW          | Wall-clock | Output                        |
|-------|-------------------------------------------------------------|-------------|------------|-------------------------------|
| A     | QPAM backbone from scratch on WT103                         | RTX 4090    | ~14h       | `v8/checkpoints/qpam_stageA.pt` |
| A.5   | TinyStories smoke of QLC primitive (2M params)              | Mac / 4090  | ~1h        | gate before A100              |
| B     | Frozen-backbone QLC ablation sweep (5–6 rows in parallel)   | A100 80GB   | overnight  | `v8/checkpoints/stageB_*/`    |
| C     | Joint fine-tune top-2 Stage B configs + KL anchor           | A100 / 4090 | overnight  | `v8/checkpoints/stageC_*/`    |

Run scripts in [`scripts/`](../scripts/): `run_v8_stageA.sh`, `run_v8_stageA5_smoke.sh`, `run_v8_stageB_*.sh`, `run_v8_stageC_joint.sh`.

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
- **Halt distribution**: yes / no / continue rates (detects degenerate halt collapse).
- **Effective rank of `Π_F`**: should equal `rank` (orthonormality drift sanity).
- **Tok/s** + **peak VRAM** for hardware bookkeeping.
- **Git SHA** + log path under `logs/v8/`.

## 5. Stage A.5 smoke results (TinyStories, 2026-04-23)

Hardware: RTX 4090. Config: `_backbone_tiny()` (dim=64, 2 layers, head_dim=32) =
6.6M backbone params; QLC adds 57k.
Recipe: 20 000 stories, seq_len=512, batch=16, 3 epochs, lr=1e-4 with 1k-step
warmup, AMP=auto (no `--compile`).

| Variant                  | Preset                    | QLC params | Best Val PPL | Tok/s   | Wall   |
|--------------------------|---------------------------|------------|--------------|---------|--------|
| Passthrough (V7-equiv)   | `smoke_tiny_passthrough`  | 0          | **178.95**   | 225 304 | 0.02 h |
| QLC r=4, T_max=2         | `smoke_tiny_qlc_r4_T2`    | 57 678     | **173.40**   | 19 180  | 0.21 h |

**Gate verdict: PASS.** Plan §5 acceptance is "QLC variant within +1 PPL of
passthrough or better" — observed delta is **−5.55 PPL** (QLC *better*). The
projector primitive is differentiable, doesn't NaN, and contributes useful
inductive bias even at a tiny scale.

**Caveats / things to watch in Stage B:**

- **`γ` was 0.000 throughout.** The orthocomplement non-commutator never
  engaged at this scale / 128-effect bank / 3 epochs. The Stage A.5 win came
  from rank-r-constrained Sasaki retrieval acting as a regulariser, *not* from
  the if/else algebra hypothesis. If Stage B also shows γ ≈ 0 across rows,
  V8-G (`orthohalt_off`) becomes the primary algebraic-utility test — if G
  matches D, the (α, β, γ) head is decorative and we should prune it per the
  plan §7 kill rubric.
- **Halt distribution drifted from ~0.02/0.96/0.02 → 0.33/0.62/0.06** over
  3 epochs. ACT pondering is alive (mean iter steady at 2.0 because we cap at
  T_max=2; will be more informative at T_max=4).
- **Throughput cost is ~12×** vs passthrough at this size (no Triton, two
  full reasoning iterations per token). Acceptable for v0; revisit only after
  V8-D-T4 shows a quality win on WT103.

Logs:
- `logs/v8/stageA5_passthrough_tinystories_20260423_114721_e3e4b90/`
- `logs/v8/stageA5_qlc_r4_T2_tinystories_20260423_114850_e3e4b90/`

## 6. Verdicts

To be filled in after Stage B & C runs. See plan §7 for the win / soft-win / kill rubric.

| Outcome  | Trigger                                                                                    | Decision |
|----------|--------------------------------------------------------------------------------------------|----------|
| Win      | `V8-E-joint ≤ 23.13` and factuality probe ≥ +5pts                                          | TBD      |
| Soft win | `V8-D-T4 ≥ V8-B by ≥ 3 PPL` and OrthoHalt-off regresses                                    | TBD      |
| Kill     | `V8-A` fails to match QPAM, or `V8-D` regresses vs `V8-B`, or OrthoHalt-off matches on    | TBD      |

## 7. Implementation status

- **2026-04-23**: V8 skeleton committed: config, model, all four QLC primitives,
  trainer, run scripts, EXPERIMENTS stub. 40 unit tests passing on CPU.
  Stage A/A.5/B/C launch scripts ready; trainer entry point smoke-tested.
- **2026-04-23**: Stage A.5 smoke run on RTX 4090. **GATE PASSED** (QLC beats
  passthrough by 5.55 PPL on TinyStories; see §5). Caveat: γ=0 means the
  algebraic orthocomplement signal is dormant at this scale; Stage B will
  need to show γ>0 for the operational-quantum-logic claim to land.
- **Next:** Stage A pretrain on RTX 4090 in tmux (~14h). After completion
  (`v8/checkpoints/qpam_stageA.pt` exists), proceed to Stage B on A100.
