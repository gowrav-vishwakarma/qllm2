# V8 Audit: Where the Code Drifts From the QLC Thesis

This audit cross-checks the current `v8/` implementation against the original
`v8_quantum_logic_core_8df73232.plan.md` thesis (operational quantum logic over
a QPAM backbone). It enumerates every place the code either weakens, dilutes,
or outright invalidates the claimed quantum-inspired mechanism. It is the
input deliverable for the `audit-current-v8` to-do in
`v8_classical_rethink_44e4a93c.plan.md`.

The audit deliberately leaves the existing files untouched so that the
critique can be reviewed before any code reshuffling.

---

## 1. The OrthoHalt γ readout collapses to zero by construction

**Claim in the plan.** "γ measures the residual mass that lies neither in the
target effect nor in its orthocomplement — the operational signal that
classical (distributive) logic fails."

**What the code does.** `v8/qlc/halt.py` builds `target_effect = u uᴴ` with `u`
forced to unit norm by `_cnormalize`, and computes:

```
α = |uᴴ ψ|²
β = ‖ψ‖² - α
γ = max(0, 1 - α - β)
```

Combined with the renormalization in `v8/qlc/reason_loop.py`:

```
sq = (ψ_next.real² + ψ_next.imag²).sum(-1)
ψ_next = ψ_next / sqrt(sq)            # ‖ψ_next‖ = 1
```

we have `α + β = ‖ψ‖² = 1` exactly, so `γ ≡ 0`. Even *without* the
renormalization, the target is a sharp rank-1 projector P (with `P² = P`), and
`α + β = ‖ψ‖²` is the Pythagorean identity on a Hilbert space — it holds in
both ℝ and ℂ. **Complex weights do not break it.**

**Verdict.** The "α/β/γ" triple is geometrically forced to lie on the line
`α + β = 1, γ = 0`. The flagship orthocomplement readout reduces to a single
free scalar (α) plus a fixed offset, which a 1-D MLP could learn directly.

**Where to fix.** `v8/qlc/halt.py:OrthoHalt.target_vector` and
`OrthoHalt.abg`. Replace the sharp rank-1 target with an *unsharp* effect
`E = σ(g)·u uᴴ` (so `E ≤ I` but `E² ≠ E`) and read γ as the gate deficit
`(1 − σ(g)) · α`, OR replace γ with a non-algebraic uncertainty signal
(predictive entropy on the next-token logits at each iteration).

Logged in `logs/v8/stageA5_qlc_r4_T2_tinystories_*` as `gamma=0.000`
throughout training, confirming the structural collapse.

---

## 2. The `quantale_off` ablation is a skip connection, not an order test

**Claim in the plan.** "When `quantale_off=True`, replace the Sasaki update
with its symmetrization so the composition becomes commutative — losing the
quantale ordering that real reasoning depends on."

**What the code does.** `v8/qlc/projector.py:sasaki_apply(symmetrize=True)`
returns `0.5 * (Π ψ + ψ)`. This is a residual blend of the projector with the
identity, applied at *one* iteration. It does not involve two distinct
projectors at all and therefore cannot test
`Π₂ Π₁ ψ` vs `0.5 (Π₁ Π₂ + Π₂ Π₁) ψ`, which is the actual non-commutativity
question.

The `t_max` loop in `v8/qlc/reason_loop.py` does run for several iterations,
and the bank produces a fresh `Π_F^{(t)}` each iteration, so the order question
*could* be asked — but the current ablation does not ask it. Whatever the row
"V8-F" measures, it is not a quantale ordering claim.

**Verdict.** The plan's V8-F row, as currently coded, is uninterpretable as
evidence about non-commutativity. Any difference it shows is between
"projector applied at full strength" vs "projector applied at half strength,
plus a residual" — that is a smoothing/regularization story, not an
algebraic-structure story.

**Where to fix.** `v8/qlc/projector.py` and `v8/qlc/reason_loop.py`. Build a
true `quantale_order_off` mode that records two consecutive projectors
`Π^{(t)}` and `Π^{(t+1)}`, then compares the sequential composition to the
symmetrized average as the loop's last step. See §2 of the rethink plan.

---

## 3. The reasoning loop runs per-token in parallel — it does not reason

**Claim in the plan / README.** "T_max iterations of probe → Sasaki → halt
let the model 'think before it speaks' — multi-step reasoning between
backbone and LM head."

**What the code does.** `v8/qlc/reason_loop.py:forward` reshapes
`ψ: [B, T, d] → [B*T, d]` and runs the inner loop on each token position
*independently* (line 179 `psi_flat = psi.reshape(B * T, d, two)`; the loop's
state per token never mixes with another token's state).

The cross-token information all sits in the QPAM backbone, which has *already
finished* before QLC starts. Therefore the QLC loop is a per-token nonlinear
refinement of a fixed input, not multi-step reasoning across positions. It is
mathematically equivalent to a deeper feed-forward stack with a clever
nonlinearity, sized to be capacity-bottlenecked through a rank-r projector.

**Verdict.** The "thinks before predicting" framing is misleading at the
architectural level. To support that claim the QLC iteration would have to be
*interleaved* with the backbone (a Universal-Transformer / latent-recurrence
pattern) so step k of token t can see step k of token t-1. Today's V8 cannot
do that.

**Where to fix.** Either (a) re-scope the claim in the README to "per-token
adaptive refinement" (which is what is actually implemented and is still a
real and testable inductive bias), or (b) move QLC inside each backbone
block. (a) is the honest minimal fix; (b) is a follow-up architecture change.

---

## 4. `out_scale = 0.1` makes the smoke result ambiguous

**Claim in the plan.** "QLC residual is added to the backbone output before
the LM head."

**What the code does.** `v8/qlc/reason_loop.py:156`:
`self.out_scale = nn.Parameter(torch.tensor(0.1))`, and the residual is
combined as `psi_out = psi + 0.1 * (psi_merged - psi)`.

This conflates two opposite hypotheses for the observed 5.55 PPL improvement
on TinyStories:

- (a) **QLC adds a useful operational primitive**, and even at 10% strength
  it contributes 5 PPL.
- (b) **QLC adds tiny structured noise**, which acts as a regularizer on the
  under-trained 6.6M-param model. At scale, this win evaporates.

Hypotheses (a) and (b) have *opposite* implications for whether to spend
A100-nights on the medium model.

**Verdict.** The current code does not log `out_scale.item()` over training
and does not run the `out_scale = 0.0` row, which is the cheapest single
discriminator between (a) and (b).

**Where to fix.** Make `out_scale` an explicit `QLCConfig` field with values
`{0.0, 0.01, 0.1, 0.5, 1.0, learnable}`, log its value at every diag step,
and add a preset row that pins it at zero (a true passthrough that still
*runs* the QLC for compute parity).

---

## 5. The InfoNCE / entity-cloze auxiliary is not wired into training

**Claim in the plan.** "Stage B trains the bank with an InfoNCE auxiliary
over the synthetic entity-cloze dataset to enforce that the bank routes
entity-related queries to dedicated effects."

**What the code does.**
`v8/qlc/effect_bank.py:EffectAlgebraBank.infonce_loss` is implemented and
`v8/data.py:EntityClozeDataset` exists, but `v8/train.py` only constructs
`load_wikitext103` / `load_tinystories` and never references either symbol.
`use_infonce=True` is the default in `QLCConfig`, but it is not consulted
anywhere in the trainer.

**Verdict.** Stage B as launched today trains the bank using LM cross-entropy
*only*, which means the "fact-routing module" claim has no objective pushing
it toward fact routing. The bank is just another set of parameters reachable
by gradient descent through the LM loss.

**Where to fix.** `v8/train.py` — add a parallel cloze loader, call
`bank.infonce_loss` on the cloze batch each step (or every K steps), and
weight the term by a new `--infonce_weight` CLI arg / `qlc.infonce_weight`
config field. Also need a small adapter in `V8LM` exposing
`encode_state(input_ids, position)` so the cloze pipeline can pull the
hidden state at the answer position without re-implementing the forward.

---

## 6. Stage B freezes the LM head along with the backbone

**Claim in the plan.** "Freeze backbone for clean attribution; QLC parameters
remain trainable."

**What the code does.** `v8/model.py:freeze_backbone` also sets
`requires_grad = False` on `lm_head_proj` and `lm_head_norm`. The embedding
matrix is shared with the LM head (tied weights), so even if the head were
unfrozen, the embedding would still be frozen.

This makes Stage B underdiagnostic: any QLC win has to flow through the
*frozen* readout, which was trained against a different state geometry. A
"QLC is helpful but the readout cannot adapt" failure is indistinguishable
from a "QLC is useless" failure.

**Where to fix.** Add a `freeze_backbone(unfreeze_lm_head: bool = False)`
option in `v8/model.py` and a `stageB_lmhead_unfrozen` preset that uses it.
Note: with tied embeddings, "unfreezing the LM head" really means unfreezing
`lm_head_proj` + `lm_head_norm`, since the embedding tensor itself remains
shared.

---

## 7. The bank's `gates()` are unsharp, but unused as such

The bank parameterizes `E_m = σ(s_m) · u_m u_mᴴ` (effect) — which *is* an
unsharp effect since `σ(s_m) ∈ (0,1)` so `E_m² ≠ E_m`. However:

- The Sasaki update uses `Π = U Uᴴ` from the *unweighted* `select_top_k` →
  `_qr_basis` path. `σ(s_m)` only modulates the top-k *score*, not the
  projector `Π_F` itself. So once the top-k columns are picked, the
  projector is sharp again.
- OrthoHalt builds its own target `u uᴴ` independently of the bank gates and
  also drops the unsharp structure.

**Verdict.** Unsharpness exists in the parameterization but is washed out by
both the QR step and the halt head. None of the downstream code consumes the
gate as part of the operator that acts on ψ.

**Where to fix.** Either (a) propagate `σ(s_m)` into `Π_F` as a weighted
projector `Π_F = U diag(g) Uᴴ` (still PSD but no longer idempotent), or (b)
plumb the bank's gates into the OrthoHalt target so γ has somewhere to live.
(a) is the cleaner fix because it makes the QLC operator unsharp in a single
controlled place; (b) only fixes the readout.

---

## 8. The smoke result is on a 6.6M-param model with no equal-FLOP control

`logs/v8/stageA5_qlc_r4_T2_tinystories_20260423_114850_e3e4b90/v8_smoke_tiny_qlc_r4_T2_tinystories.log`:

- Passthrough: 178.95 PPL @ 6.6M params, 225k tok/s.
- QLC r=4, T=2: 173.40 PPL @ 6.66M params (+0.86%), 19.2k tok/s (12× slower).

The 3% PPL improvement comes with a 12× wallclock cost. A fairer baseline at
matched compute would train the passthrough for 12× more steps; given LM
loss curves on TinyStories at this scale, that almost certainly closes the
gap. The current evidence does not justify a 14h Stage A re-run plus an
A100 sweep.

**Where to fix.** Run the discriminator suite (§G of the rethink plan)
*before* committing to large-scale Stage A. The rank-sweep and out_scale
rows in particular are cheap and decisive.

---

## 9. Summary of structural issues vs surface issues

| # | Issue | Type | Fix Cost |
|---|-------|------|----------|
| 1 | γ collapses to zero by construction | **Structural** | Module rewrite |
| 2 | quantale_off is a skip connection | **Structural** | Need new ablation path |
| 3 | Per-token loop ≠ reasoning | **Structural** | Either reframe or rewire architecture |
| 4 | out_scale=0.1 conflates interpretations | Surface | Add config + log |
| 5 | InfoNCE not wired | Surface | Implement training path |
| 6 | LM head frozen in Stage B | Surface | Add option |
| 7 | Gates parameterized but not used | Structural | Make Π_F unsharp |
| 8 | No equal-FLOP control on smoke | Surface | Add baseline preset |

Issues #1, #2, #3, #7 invalidate the *operational quantum logic* claim
directly. Issues #4, #5, #6, #8 prevent the experimental setup from
resolving the question even if the claim were true.

---

## 10. What the existing code does support, honestly

After the audit, the things that *are* backed by the current implementation:

- **Constrained low-rank subspace memory** (`SasakiProjectionMemory`):
  the rank-r orthonormal basis update is correct, well-tested for
  orthonormality drift, and is a real classical inductive bias.
- **Top-k routed bank** (`EffectAlgebraBank.select_top_k`): the QR-based
  basis construction with sqrt-score weighting is a defensible, gradient-
  friendly design even if the algebraic story is dropped.
- **Adaptive Computation Time** halting (`OrthoHalt.cls_head` + ponder
  state): the ACT plumbing is correct and modular; the only weak part is
  the *signal* it consumes (γ), not the loop.

If the discriminator suite kills the quantum framing, what survives is a
clean *Constrained Latent Memory + Adaptive Compute* architecture — a real
contribution, just not the headline one.

