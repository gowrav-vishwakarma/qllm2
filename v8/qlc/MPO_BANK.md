# MPO-Tensorized Effect Bank — Design Note

This document defines a **tensor-network follow-on experiment** for V8. It is
deliberately scoped as a *post-survival* experiment: do not run it until the
core V8 architecture survives the discriminator suite documented in
[`EXPERIMENTS_V8.md`](../EXPERIMENTS_V8.md) §8.

The companion implementation stub lives in
[`mpo_bank.py`](mpo_bank.py).

---

## 1. Why a Tensor-Network branch at all?

The rethink plan ([`v8_classical_rethink_44e4a93c.plan.md`](../../.cursor/plans/v8_classical_rethink_44e4a93c.plan.md))
recommends adding *one* tensor-network-inspired branch only after the
projector-memory hypothesis earns its keep. The literature on TN methods for
NLP is much stronger on **compression and structured prior** than on
"replace the whole LM with a TN model", so we focus on a single,
well-scoped target: **compressing the [`EffectAlgebraBank`](effect_bank.py)
itself.**

Concretely, the existing bank stores `M` rank-1 effects as
`u_m, w_m ∈ C^d`, costing `O(M·d)` parameters (currently
`2048·384·4 ≈ 3.1M` floats including value and gate). For the
"bank-as-evidence-shelf" interpretation to scale to "bank-as-knowledge-base"
(rethink-plan §6 secondary success criterion), we want `M` to grow
substantially without `O(M·d)` parameters.

The MPO answer: **factor the effect index `m` as a tensor-product index
`(m₁, m₂, …, m_C)` over `C` "cores", each carrying its own learnable
local feature**, and parameterize the effect dictionary as a Matrix
Product Operator over those cores. The number of *addressable* effects then
grows multiplicatively (`M_eff = ∏ d_c`) while the parameter count grows
**polynomially** in `(C, χ, d_per_core)`, where `χ` is the bond dimension.

This pairs naturally with the V8 thesis: the bank-as-test-space reading
(rethink-plan §11.1) argues that effects should *partition* into typed
sub-bundles. An MPO factorization is exactly the algebraic statement
"effects are organized into tensor-product cores"; a typed sub-bundle is a
slice of one core.

---

## 2. The math (one page)

### 2.1 Flat bank (current)

For state `ψ ∈ C^d`, the flat bank computes scores

    s_m = σ(g_m) · |u_mᴴ ψ|²    for m = 1..M

selects top-k indices `{m₁, …, m_k}`, then orthonormalizes the corresponding
`u_m` columns to a rank-r basis fed to the SPM.

Parameters: `O((d + 1) · M)` (effect dirs, value dirs, gates).

### 2.2 MPO bank

Choose a factorization `d = d₁ · d₂ · … · d_C` of the state dimension. Index
each effect by a tuple `(m₁, …, m_C)` with `m_c ∈ {1, …, K_c}`. Total
addressable effects: `M_eff = ∏ K_c`.

Parameterize the effect direction tensor `U[m₁, …, m_C, i₁, …, i_C]` (where
`i_c ∈ {1, …, d_c}` indexes the spatial dimension) as an MPO with bond
dimension `χ`:

    U[m₁..m_C, i₁..i_C] = Σ_{α₀..α_C}  A¹[α₀, m₁, i₁, α₁]
                                       · A²[α₁, m₂, i₂, α₂]
                                       · …
                                       · A^C[α_{C-1}, m_C, i_C, α_C]

with boundary bonds `α₀ = α_C = 1`. Each core `A^c` has shape
`[χ, K_c, d_c, χ]`; total parameter count is `O(C · χ² · K_c · d_c)`,
i.e. **logarithmic** in `M_eff` for fixed `(χ, K_c, d_c)`.

### 2.3 Probe step under the MPO bank

Given query `ψ`, reshape it as `ψ[i₁, …, i_C]` of shape
`d₁ × d₂ × … × d_C` (this is the "tensor-network-friendly view of `ψ`",
analogous to viewing a 384-dim vector as an 8×6×8 tensor). The probe score

    s[m₁..m_C] = | <U[m₁..m_C], ψ> |²

becomes a tensor contraction whose cost is `O(C · χ² · max_c K_c · d_c)`,
**not** `O(M_eff · d)`. The full score tensor of shape `K₁ × … × K_C` is
*not* materialized; we use TN-style top-k contraction (subroutines from
e.g. [`tensorly`](https://github.com/tensorly/tensorly) or the
top-k-on-tensor-trains literature) to extract the highest-scoring tuples
without enumerating them all.

For the v0 stub in [`mpo_bank.py`](mpo_bank.py) we **materialize** the full
score tensor when `M_eff` is small enough (`< 16k`), which keeps the code
simple and lets us validate that the MPO bank is gradient-friendly and
matches the flat-bank interface. The "true" sublinear top-k path is left as
a TODO (it is a known but non-trivial routine).

### 2.4 Building Π_F

After top-k selection we have `k` effect indices `(m₁⁽ⁿ⁾, …, m_C⁽ⁿ⁾)` for
`n = 1..k`. Materialize the corresponding `u_n ∈ C^d` by contracting the
MPO at the selected indices (cheap: `O(k · C · χ² · d_max)`), then run the
existing complex QR in [`effect_bank.py`](effect_bank.py) to produce the
rank-r orthonormal basis. **No change** to the SPM or the rest of the QLC
loop.

This is the key API affordance: **`MPOEffectBank.select_top_k` returns the
same `(U, V)` shape as `EffectAlgebraBank.select_top_k`**, so it is a
drop-in replacement at the [`reason_loop.py`](reason_loop.py) call site.

---

## 3. The minimum viable experiment

Once the discriminator suite passes the soft-win criterion (see
[`README_V8.md`](../README_V8.md) §11), run a single matched-parameter
ablation:

| Row                     | Bank impl            | M_eff       | Bank params | Stage B Val PPL |
|-------------------------|----------------------|-------------|-------------|-----------------|
| Flat bank, M=2k         | `EffectAlgebraBank`  | 2 048       | ~1.6M       | (baseline)      |
| Flat bank, M=8k         | `EffectAlgebraBank`  | 8 192       | ~6.3M       | (compute-heavier baseline) |
| MPO bank, K=4 cores     | `MPOEffectBank`      | 4·6·8·8 = 1 536 effective | ~0.2M | (compression test) |
| MPO bank, K=4 cores, χ=16 | `MPOEffectBank`    | up to 16k effective | ~0.5M | (capacity test) |

The MPO bank wins **iff** at *equal Stage-B Val PPL*, it uses at least 4×
fewer parameters than the flat bank (compression hypothesis), or **at equal
parameters**, it offers a meaningfully larger effective `M` and a Val PPL
improvement (capacity hypothesis).

If neither holds, the TN branch returns no signal and we close the
follow-on. The parent V8 architecture is not affected.

---

## 4. What this design deliberately does **not** do

- It does **not** replace the language-model backbone with a TN architecture.
  The literature on "MPO-as-LM" is brittle at scale; we focus on the bank
  alone, where the TN framework matches the actual structure of the data
  (a discrete dictionary of compositional concepts).
- It does **not** entangle the MPO bank with the OrthoHalt or quantale
  hypotheses. Whether γ is non-trivial under unsharp targets is independent
  of how the bank stores its effects.
- It does **not** introduce its own training objective. The bank is trained
  through the existing LM cross-entropy + (optional) InfoNCE auxiliary.
  Adding a TN-specific loss (e.g. low-bond-dim regularization) is a v2
  concern.
- It does **not** ship as part of v8.0 or v8.1. The stub is in the
  repository so the design is reviewable, but it is not wired into any
  preset and not exercised by the discriminator suite.

---

## 5. Open questions for v8.2

1. **Phase tensors.** If the project keeps complex weights, each MPO core
   carries an independent phase. Does that admit destructive interference
   across cores in a way that gives the MPO bank a different inductive bias
   than a pile of `O(M_eff)` independent complex effects?
2. **Order of cores.** MPO contractions are not symmetric across the core
   ordering. Does the choice of "which feature lives in which core" matter
   for downstream PPL? (Connected to the ordering test in
   [`reason_loop.py`](reason_loop.py).)
3. **Hierarchical TT vs flat MPS.** Tree Tensor Networks may offer a more
   natural fit to compositional language structure than a 1-D MPS.
   Out-of-scope for v8.2; flagged for v9.
