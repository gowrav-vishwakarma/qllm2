# V7: Hierarchical Phase-Associative Memory Model

## Goal

Build a lean, self-contained model that is **neither a transformer nor an SSM**, based on the Phase-Associative Memory (PAM) architecture from V6. V7 distills the proven wins from V5/V6 experiments into a clean, single-directory codebase with built-in ablation toggles, targeting competitive perplexity on WikiText-103 at ~100M parameters. The architecture is modality-agnostic -- designed for sequences of any kind (text, image patches, audio frames, diffusion steps).

Two novel architectural ideas distinguish V7 from V6:

1. **Hierarchical Timescale Specialization** -- each PAM layer operates at a different temporal granularity, from global-level (~1000 steps) down to step-level (~2 steps).
2. **Cross-Level Drift Conditioning** -- higher layers explicitly bias lower layers' retrieval queries, creating cascading goal-directed generation from intent down to prediction.

---

## Architecture Overview

```
Input -> ComplexEmbed -> [V7Block] x N -> Tied Complex LM Head -> logits

Each V7Block:
  pre-norm -> CGU (channel mixing, SwiGLU-style) -> residual
  pre-norm -> PAM (sequence mixing, matrix state)  -> residual (scaled)
```

### Core Components (inherited from V5/V6, proven wins)

- **Split-real complex representation**: `[..., dim, 2]` tensors. Never `torch.complex64/128` (OOM + autograd issues).
- **Phase-Associative Memory (PAM)**: Matrix state `S_t = gamma_t * S_{t-1} + V_t (x) K_t*`, retrieval `Y_t = S_t * Q_t`. O(T^2) dual form for training, O(1) recurrent form for inference. No softmax, no KV cache.
- **ComplexGatedUnit (CGU)**: SwiGLU-style gating in complex space. Gate magnitude controls how much, gate phase controls rotation.
- **ComplexLinear**: Block-real GEMM fusing four matmuls into one. Orthogonal initialization with Xavier-like scaling `gain=sqrt(2/(in+out))`.
- **ComplexNorm**: RMSNorm for complex -- normalizes magnitude, preserves phase.
- **ModReLU**: Phase-preserving activation (threshold on magnitude, phase untouched).
- **Complex RoPE**: Positional encoding via phase rotation on Q and K.
- **GSP (Gated State Protection)**: Learned gates that can freeze important state dimensions.
- **Tied Complex LM Head**: `logits = Re(z * conj(embed)) = z_r @ e_r^T + z_i @ e_i^T`.

### What V6 experiments showed us (distilled into V7)

**Proven wins carried forward:**
- PAM matrix state >> vector-state SSM (headline V6 result)
- Interleaved CGU + PAM >> sequential stacking
- Orthogonal init: consistent ~31% advantage
- Complex RoPE on Q/K
- GSP for state protection
- Fused QKV projection for speed
- Block-real GEMM in ComplexLinear

**Confirmed dead ends (excluded from V7):**
- Native `torch.complex64/128` (OOM, autograd failures)
- Explicit memory subsystems (working memory, internal memory, episodic)
- Diversity/role loss (no benefit with single-bank CGU)
- NamedBankPair + Coupler (replaced by simpler CGU)
- Two-pass / diffusion objectives

---

## Learnings log (rolling — keep entries short)

| When | What | Headline |
|------|------|----------|
| 2026-03 | **3a-B** ModReLU, `medium_h16_flat`, B=3, `--no_grad_ckpt`, commit `fc161ce` | Val **~30.4** @10e vs V6 pam-v3 **29.95** — parity (~1.5% gap). Log: `logs/v7/medium_h16_flat_wikitext103_20260327_104348_fc161ce/`. |
| 2026-03 | **7a** ModSwish, same preset/stack, commit `81e8ea9` (dirty) | Val **29.73** @10e vs V6 **29.95** — small win; val below V6 every epoch. Transformer B=3 still **27.08**. Log: `logs/v7/exp7a_swish_wikitext103_20260328_081707_81e8ea9_dirty/`. |
| 2026-03 | **Transformer B=6** baseline (same arch, `--batch_size 6`) | Val **23.13** @10e — **14.6%** better than B=3 (27.08). ~99k tok/s, 2.5/12.7 GB GPU. Log: `logs/v6/transformer_baseline_wikitext103_20260330_130306_8d631a6/`. |
| 2026-03 | **7d** ModSwish + chunked dual form (`C=256`), `medium_h16_flat`, B=6, commit `7555c93` | Val **27.94** @10e, beating 7a by **1.79 PPL** while raising throughput to **31.8k tok/s**. Still behind transformer B=6 **23.13**. Log: `logs/v7/exp7d_chunked_b6_wikitext103_20260330_172839_7555c93/`. |
| — | Hygiene | Use **B=3** + **no grad ckpt** for V6/transformer apples-to-apples. **B=6** + ckpt (3a-A) lowers PPL but changes steps/epoch — confounds LR schedule vs V6. **B=6 transformer baseline now available** for apples-to-apples when PAM uses B=6. |

---

## Experiment 1: Hierarchical Timescale Specialization

### Hypothesis

Instead of N identical PAM layers that must discover timescales via gradient descent, explicitly assign each layer a resolution level with a distinct memory span. This provides a stronger inductive bias and should accelerate convergence.

### Design

The `dt_bias` parameter controls each layer's base decay rate via `gamma = exp(-softplus(dt_bias))`. The schedule is adapted for seq_len=2048 so every level is fully utilized:

| Layer | Level  | dt_bias | gamma  | Memory Span | Retention @512 | Retention @2048 |
|-------|--------|---------|--------|-------------|----------------|-----------------|
| 0     | global | -6.91   | 0.999  | ~1000 steps | 60%            | 13%             |
| 1     | broad  | -5.52   | 0.996  | ~250 steps  | 13%            | ~0%             |
| 2     | mid    | -4.08   | 0.983  | ~60 steps   | ~0%            | ~0%             |
| 3     | local  | -2.64   | 0.933  | ~15 steps   | ~0%            | ~0%             |
| 4     | fine   | -1.39   | 0.800  | ~5 steps    | ~0%            | ~0%             |
| 5     | step   | 0.0     | 0.500  | ~2 steps    | ~0%            | ~0%             |

Gradient sensitivity is naturally hierarchical: global-level layers are nearly frozen (|grad|=0.0003), step-level layers are adaptive (|grad|=0.24). The model doesn't rethink the global intent step-by-step, but immediate predictions are highly dynamic.

### Preset: `medium_h6`

6 layers (one per resolution level), wider to match ~100M param budget:
- dim=512, n_heads=8, head_dim=64, expand=4
- ~102.4M params (without cross-level), ~105.0M (with cross-level)

### Ablation toggle

`--no_hierarchical_dt` reverts to uniform dt_bias=-4.0 across all layers.

---

## Experiment 2: Cross-Level Drift Conditioning

### Hypothesis

Goal-directed generation is hierarchical: high-level intent (the "global plan") guides progressively finer production. Broad structure drifts toward the global goal; local patterns drift toward the broad structure; fine detail drifts toward local coherence; each step drifts toward completing the fine pattern. The current architecture carries this implicitly through the residual stream, but an explicit mechanism should strengthen the signal.

### Design

Each PAM layer (except the topmost global-level) receives the previous layer's raw PAM output as a **drift signal**. A learned `drift_proj: ComplexLinear(dim, inner_dim)` projects this into Q-space and adds it to the query vectors:

```
Layer 0 (global):  Q = proj(z)                                [top of hierarchy]
Layer 1 (broad):   Q = proj(z) + drift_proj(PAM_0_output)     [drifts toward global goal]
Layer 2 (mid):     Q = proj(z) + drift_proj(PAM_1_output)     [drifts toward broad goal]
Layer 3 (local):   Q = proj(z) + drift_proj(PAM_2_output)     [drifts toward mid goal]
Layer 4 (fine):    Q = proj(z) + drift_proj(PAM_3_output)     [drifts toward local goal]
Layer 5 (step):    Q = proj(z) + drift_proj(PAM_4_output)     [drifts toward fine goal]
```

**Why Q (not K or V):** Q controls what the layer is looking for. Biasing Q means "look for elements that fit the higher-level goal." K and V store what's already there and shouldn't be biased.

### Parameter overhead

5 x ComplexLinear(512, 512) = 2.6M extra params (2.5% of total). No new O(T^2) terms.

### Ablation toggle

`--no_cross_level` disables the drift mechanism for A/B comparison.

---

## Experiment 1+2 Results: medium_h6 (Run A: hierarchy + drift)

**Run A**: Full system with both hierarchical timescales and cross-level drift.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1     | 219.6     | 123.4   | *best* |
| 2     | 131.1     | 107.5   | *best* |
| 3     | 118.2     | 98.3    | *best* |
| 4     | —         | —       | stopped after ~100 batches |

**V6 medium-pam-v3 at same epochs** (16L, dim=384, ~100M params):

| Epoch | Train PPL | Val PPL |
|-------|-----------|---------|
| 1     | 123.9     | 57.9    |
| 2     | 53.9      | 43.8    |
| 3     | 44.9      | 38.7    |
| 10    | 30.0      | 29.95   |

**Verdict**: Val PPL improves every epoch but lags V6 enormously (98 vs 39 at epoch 3). The 6-layer-wide design is strictly worse than 16-layer-narrow at the same param budget. The hierarchical timescale and cross-level drift hypotheses **cannot be evaluated in isolation** because the depth difference dominates — 6 layers is simply insufficient representational depth for WikiText-103.

**Runs B and C were not executed** — the depth confound must be resolved first.

**Sample at epoch 3** (prompt: "In 1923 , the University of"):
> In 1923 , the University of Cambridge 's Union , Canada , and North Carolina . The U.S. Army Department ( NCSO ) was formed in 1956 and served as an honorary officer at Fortissus . It is also known for his class of soldiers .

Quality: rep3=0.000, rep4=0.000, uniq=0.771. Coherence is weak; factual accuracy poor.

**Key lesson**: Depth matters far more than width for language modeling at this scale. Each PAM+CGU block adds a level of compositional abstraction, and 6 is not enough.

---

## Experiment 3a: Depth-Matched Flat Baseline (`medium_h16_flat`)

### Hypothesis

Before testing hierarchical timescales at depth, we need to verify that the V7 codebase itself matches V6 performance when given identical depth/width. If V7 flat 16-layer doesn't approach V6's ~30 val PPL, there is a code regression to fix before adding novel features.

### Design

Exactly match V6 medium-pam-v3's shape: 16 layers, dim=384, expand=3, H=6, head_dim=64, ~100M params. All V7 novelties disabled:
- `hierarchical_dt=False` — uniform dt_bias=-4.0 across all layers
- `cross_level=False` — no drift conditioning

### Preset: `medium_h16_flat`

- dim=384, n_layers=16, n_heads=6, head_dim=64, expand=3
- hierarchical_dt=False, cross_level=False
- Target: Val PPL ~30 (matching V6 medium-pam-v3)

### What success looks like

- **Match**: Val PPL within ~5% of V6 (~30). Confirms V7 codebase is sound → proceed to hierarchy experiments.
- **Regression**: Val PPL significantly worse. Investigate what V6 has that V7's refactoring lost.

### Run 3a-A: B=6, grad checkpointing ON (commit `383e514`, dirty)

First attempt after code fixes. Used **batch_size=6** with gradient checkpointing to fit in VRAM. Cancelled mid epoch 10.

**Code fixes applied** (vs original V7 code):
1. **ComplexLinear init scale** — was `gain=1/sqrt(2)=0.707` (fixed, dimension-independent). Fixed to `gain=sqrt(2/(in+out))` matching V6's Xavier-like scaling. **14x magnitude difference** for 384→384 layers. This was the dominant regression.
2. **CGU residual scale** — added learnable `cgu_scale` (init 1.0) matching V6's `feature_scales`.
3. **CGU dropout** — added dropout on CGU output matching V6's per-layer dropout.
4. **`_init_weights` pass** — `nn.Linear` layers (dt_proj, protect_gate) now get `normal_(std=0.02)` instead of PyTorch default Kaiming uniform.
5. **Removed extra `final_norm`** — V7 had an extra normalization before the LM head that V6 didn't have.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1     | 151.5     | 58.9    | *best* |
| 2     | 51.6      | 41.0    | *best* |
| 3     | 40.2      | 35.1    | *best* |
| 4     | 35.1      | 31.9    | *best* |
| 5     | 31.9      | 29.8    | *best* |
| 6     | 29.6      | 28.5    | *best* |
| 7     | 27.9      | 27.5    | *best* |
| 8     | 26.6      | 26.9    | *best* — train < val, overfitting starts |
| 9     | 25.7      | 26.6    | *best* — gap widening |
| 10    | —         | —       | cancelled at 25% (epoch 10, step 2450/9639) |

**Why cancelled:**
1. **Overfitting detected** — train PPL crossed below val PPL at epoch 8 (26.6 vs 26.9), gap widening at epoch 9 (25.7 vs 26.6). Diminishing returns on val PPL.
2. **Not apples-to-apples** — B=6 gives 9,639 steps/epoch vs V6's 19,277 at B=3. Different LR schedule shape (cosine over half the steps), different warmup coverage (12M tokens vs 6M). Cannot isolate whether improvement over V6 is from code fixes or batch size dynamics.
3. **Need clean V6 parity first** — before adding V7 novelties (hierarchy, drift), must confirm V7 matches V6 at identical hyperparameters.

**Throughput**: ~18,780 tok/s (with grad checkpointing). VRAM: 11.6 GB peak (B=6).

**Key finding — larger batch size converges faster and reaches lower val PPL:**

Despite being not apples-to-apples with V6 (different batch size), this run **beat V6's final val PPL (29.95) by epoch 5** (29.8) and reached **26.6 by epoch 9**. At matched epochs, the gap was dramatic:

| Epoch | V6 (B=3) Val PPL | V7 3a-A (B=6) Val PPL | Improvement |
|-------|-------------------|-----------------------|-------------|
| 1     | 57.9              | 58.9                  | -1.7%       |
| 3     | 38.7              | 35.1                  | +9.3%       |
| 5     | 33.8              | 29.8                  | +11.8%      |
| 9     | 30.0              | 26.6                  | +11.3%      |

This suggests **batch_size=6 may be a better training regime** for this architecture. The larger batch provides smoother gradients and the halved steps/epoch means the cosine LR schedule decays more aggressively per token, which may act as implicit regularization. **If Run 3a-B (B=3) matches V6 but doesn't beat it, future experiments should default to B=6 with gradient checkpointing.**

**Sample at epoch 9** (prompt: "In 1923 , the University of"):
> In 1923 , the University of Michigan opened its first school in 1926 . In 1927 , the University moved into a new campus at the University of Michigan . It now serves as the main university 's second academic institution with over 4,000 students.

Quality: rep3=0.043, rep4=0.011, uniq=0.667. Better coherence than Exp 1, but some repetition.

### Run 3a-B: B=3, grad checkpointing OFF (commit `fc161ce`) — done

Clean apples-to-apples comparison. Same batch_size=3 and no gradient checkpointing as V6 and transformer baseline.

| Setting | This run | V6 pam-v3 | Transformer |
|---------|----------|-----------|-------------|
| Batch size | 3 | 3 | 3 |
| Steps/epoch | 19,277 | 19,277 | 19,277 |
| LR / warmup | 1e-4 / 1000 | 1e-4 / 1000 | 1e-4 / 1000 |
| Params | 100.4M | 100.4M | 100.3M |
| Grad ckpt | OFF | OFF | OFF |

**Val PPL @ epoch 10**: ~**30.4** (best in-run) vs V6 **29.95** vs transformer **27.08**. See **Learnings log** for path.

---

## Experiment 3b: Grouped Hierarchical (`medium_h16_grouped`)

### Hypothesis

The hierarchical timescale idea is valid but needs V6-comparable depth. Grouping multiple layers per timescale level gives **depth within each resolution** — the model can build a richer representation at each temporal granularity before moving on.

### Design

16 layers, same dim=384 shape as 3a, but with grouped timescale hierarchy and cross-level drift enabled:

| Level  | dt_bias | Layers | Count | Rationale |
|--------|---------|--------|-------|-----------|
| global | -6.91   | 0-3    | 4     | Long-range coherence is hardest for non-attention models |
| broad  | -5.52   | 4-6    | 3     | Large-scale structure |
| mid    | -4.08   | 7-9    | 3     | Medium-range context |
| local  | -2.64   | 10-12  | 3     | Local coherence |
| fine   | -1.39   | 13-14  | 2     | Fine-grained detail |
| step   | 0.0     | 15     | 1     | Immediate prediction |

Grouping is heavier at global/broad (4+3 layers) because long-range coherence demands the most capacity. Cross-level drift operates naturally: within a group, layers share the same timescale but each still receives drift from the previous layer's PAM output.

### Preset: `medium_h16_grouped`

- dim=384, n_layers=16, n_heads=6, head_dim=64, expand=3
- hierarchical_dt=True, cross_level=True
- dt_bias_schedule: (-6.91, -6.91, -6.91, -6.91, -5.52, -5.52, -5.52, -4.08, -4.08, -4.08, -2.64, -2.64, -2.64, -1.39, -1.39, 0.0)
- Target: Beat Experiment 3a

---

## Experiment 4: Compositional Weight Encoding ("Formula Weights") — FUTURE

### Motivation

Standard weights store one scalar per parameter. Can each parameter slot encode more structure? Instead of `weight = 9`, represent `weight = f(a, b, c) = sum_k(e^{i*phi_k} * U_k @ V_k^T)` — a phase-weighted sum of basis components where interference patterns create richer interactions.

### Design — Phase-Superposed Linear (PSL)

Replace `ComplexLinear` in QKV projections with:

```python
class PhaseSuperposedLinear(nn.Module):
    """W = sum_k( e^{i*phi_k} * U_k @ V_k^T ) — each weight is a formula."""
    def __init__(self, in_dim, out_dim, n_basis=4, rank=None):
        rank = rank or min(in_dim, out_dim) // 4
        # K basis matrices, each low-rank: U_k @ V_k^T
        self.U = nn.Parameter(...)   # [K, out_dim, rank, 2] complex
        self.V = nn.Parameter(...)   # [K, rank, in_dim, 2] complex
        self.phi = nn.Parameter(...)  # [K] phase offsets per basis
```

Each effective weight W_ij = sum_k(e^{i*phi_k} * sum_r(U_k[i,r] * V_k[r,j])). Constructive interference when phases align, destructive when opposed. The network learns which basis components reinforce or cancel.

### Why this is novel for PAM

- Complex numbers already give 2-for-1 (magnitude + phase). This extends it to K-for-1.
- Phase interference is the mechanism (not gating or routing) — fits the PAM philosophy.
- Param-matched comparison: choose rank so total params equal dense ComplexLinear.

**Status**: Design only. Implement after Experiments 3a/3b establish the depth baseline.

---

## Experiment 5: Superposed PAM States — FUTURE

### Motivation

PAM maintains one matrix state S per head. True quantum-inspired superposition would maintain K overlapping states that interfere during retrieval — the network learns which states should constructively or destructively combine for each query.

### Design — Multi-State PAM

```
S = [S_1, S_2, ..., S_K]  per head, each d x d complex

Update:  S_k = gamma_k * S_k + alpha_k * (V ⊗ K*)
         alpha_k = data-dependent routing weights

Retrieval: Y = sum_k( e^{i*phi_k(x)} * S_k * Q )
           phi_k(x) = data-dependent phase (learned projector on input)
```

When phases align → constructive interference → strong retrieval from that basis state. When opposed → destructive → suppressed. K=2 doubles state memory but adds minimal params.

**Status**: Design only. Implement after Experiments 3a/3b.

---

## Experiment 6: Connectome-Inspired Sparse Topology — FUTURE

### Motivation

Biological neural networks demonstrate that **topology IS computation** — simple per-neuron rules (integrate inputs, fire if threshold exceeded) applied over structured connectivity patterns produce sophisticated behavior. Excitatory neurons amplify downstream signals while inhibitory neurons suppress them; the specific pattern of which neurons connect to which encodes the computation, not just the synaptic weights. Our architecture uses fully dense sequential connectivity (every neuron in layer i connects to every neuron in layer i+1). Structured sparse connectivity with excitatory/inhibitory specialization could encode more computation per parameter.

### Design options

- **E/I Head Specialization**: Half the PAM heads are "excitatory" (positive gamma bias, build state) and half "inhibitory" (negative gamma bias, actively clear state). Different from current uniform heads.
- **Skip-level drift**: Instead of drift only from the immediately previous layer, allow specific non-adjacent layers to influence each other (e.g., global layer directly biases step layer).
- **Small-world CGU**: Structured sparse connections in CGU projections — mostly local dims plus a few long-range connections.

**Status**: Design only. Lowest priority; needs careful implementation.

---

## Experiment 7: Activation Function Upgrades — FUTURE

### Motivation

V7 uses **ModReLU** as its only nonlinearity — a hard ReLU threshold on magnitude with phase held constant. It appears in exactly one place: the `up` path inside CGU. The PAM path is entirely linear (matrix state update + retrieval), so ModReLU is the sole source of nonlinear expressiveness per block.

ModReLU was a correct fix for V4's "real activations destroy phase" problem, but it is overly conservative:
- **Hard threshold**: ReLU kills gradient below the bias. Through 16 layers, dying neurons compound. In real-valued transformer literature, smooth activations (GELU, Swish) consistently outperform ReLU by 1-3%.
- **Phase is read-only**: ModReLU never rotates phase. It treats magnitude and phase as fully decoupled. But phase IS information — a richer activation should create magnitude-phase dependencies.
- **Liouville's theorem** constrains the design: no bounded, nonlinear, holomorphic function exists. ModReLU sacrifices holomorphicity (correct choice). The question is whether it sacrifices too much by also freezing phase.

Since activation sits inside CGU which runs in every block, a better activation is a **multiplier on depth**: the improvement compounds over 16 layers.

### Design A — Complex Swish (drop-in, zero-risk)

Replace `ReLU(mag + bias)` with smooth `mag * sigmoid(beta * mag + bias)`:

```python
class ModSwish(nn.Module):
    """Smooth phase-preserving activation: Swish on magnitude, phase untouched."""
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        return phase * activated.unsqueeze(-1)
```

Properties:
- Smooth everywhere (no dead neurons)
- Non-zero gradient below threshold (better gradient flow through deep stacks)
- Learnable `beta` controls sharpness (network finds optimal nonlinearity shape)
- Phase-preserving (same safety as ModReLU)
- +dim params vs ModReLU (negligible at dim=384)

### Design B — Phase-Modulated Activation (novel, higher risk/reward)

Allow the activation to **rotate phase based on magnitude** — high-magnitude signals get different phase rotations than low-magnitude ones:

```python
class PhaseModulatedActivation(nn.Module):
    """Activation that couples magnitude and phase."""
    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.ones(dim))
        self.phase_alpha = nn.Parameter(torch.zeros(dim))  # mag->phase coupling
        self.phase_beta = nn.Parameter(torch.zeros(dim))   # phase offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = cabs(z)
        activated = mag * torch.sigmoid(self.beta * mag + self.bias)
        phase = z / (mag.unsqueeze(-1) + 1e-8)
        # Magnitude-dependent phase rotation
        theta = self.phase_alpha * mag + self.phase_beta
        rot = torch.stack([theta.cos(), theta.sin()], dim=-1)
        phase = cmul(phase, rot)
        return phase * activated.unsqueeze(-1)
```

Properties:
- Includes all of Design A's benefits (smooth, learnable sharpness)
- Additionally creates **magnitude-phase coupling**: "important" (high-mag) signals and "weak" (low-mag) signals get different phase rotations
- `phase_alpha` initialized to 0, so it **starts as ModSwish** and discovers coupling via gradient descent
- +2*dim params (negligible)
- Risk: phase rotation might interfere with RoPE or CGU's own phase gating. Needs ablation.

### Why CGU gating doesn't fully compensate

CGU already has phase rotation via `gate_phase * up`. But:
- The phase rotation comes from `gate_proj(z)`, an independent projection — it doesn't depend on the **activated magnitude** of the `up` path.
- The activation (ModReLU) runs BEFORE gating. Units zeroed by the hard threshold are permanently gone — the gate cannot resurrect them.
- With smooth activation + phase modulation, ALL units contribute (with varying strength and rotation), giving the gate richer input to work with.

### Execution plan

1. **Design A (ModSwish)**: Add as `activation='swish'` config option. Run on Exp 3a's preset after baseline. Zero-risk, expects 1-3% PPL gain.
2. **Design B (PhaseModulated)**: Add as `activation='phase_mod'` option. Run after Design A. Higher variance, expects 3-8% gain if magnitude-phase coupling helps.
3. Compare both against ModReLU at matched everything else.

**Status**: **7a done** (ModSwish beats archived V6 pam-v3 slightly on val PPL; see Learnings log). **7b stopped** mid epoch 1 (61%, no val result yet) to free GPU for 7c throughput experiments. Can resume later with `--resume`.

---

## Comparison Plan (Updated)

| Run | Preset | Layers | dim | hierarchy | cross_level | Activation | Target Val PPL |
|-----|--------|--------|-----|-----------|-------------|------------|----------------|
| 1-A | medium_h6 | 6 | 512 | True (explicit) | True | ModReLU | — (98.3, stopped) |
| 3a  | medium_h16_flat | 16 | 384 | False | False | ModReLU | **~30.4** val @10e (3a-B, V6 parity) |
| 3b  | medium_h16_grouped | 16 | 384 | True (grouped) | True | ModReLU | Beat 3a |
| 7a  | medium_h16_flat | 16 | 384 | False | False | ModSwish | **29.73** val @10e (vs V6 29.95) |
| 7b  | medium_h16_flat | 16 | 384 | False | False | PhaseMod | Beat 7a (TBD) |
| 7c  | medium_h16_flat | 16 | 384 | False | False | ModSwish | Same PPL as 7a, higher tok/s (chunked C=256) |
| 7d  | medium_h16_flat | 16 | 384 | False | False | ModSwish | **27.94** val @10e, **31.8k tok/s** (chunked C=256, B=6) |
| 7e  | medium_h16_flat | 16 | 384 | False | False | ModSwish | Beat 7c/7d (chunked + unitary reg λ=0.01) |
| 7f-0 | medium_h16_grouped | 16 | 384 | True (grouped) | True | ModSwish | **31.29** val @10e, B=3 (confounded: grouped+multiscale) |
| 7f-1 | medium_h16_flat | 16 | 384 | False | False | ModSwish | Ablation: multi-scale loss only (vs 7a baseline) |
| 7f-2 | medium_h16_flat | 16 | 384 | False | False | ModSwish | Ablation: reverse assoc only (vs 7a baseline) |
| 7f-3 | medium_h16_grouped | 16 | 384 | True (grouped) | True | ModSwish | Ablation: grouped hierarchy only (vs 7a baseline) |

Runs B/C from the original plan are superseded by 3a (which IS the flat baseline at proper depth).
Experiments 7a/7b test activation upgrades on the depth-matched baseline.
Experiments 7c/7d/7e test throughput optimizations (chunked dual form, larger batch) and regularization.
Experiments 7f-0 was confounded (grouped hierarchy + multi-scale loss changed simultaneously).
Experiments 7f-1/7f-2/7f-3 are proper single-variable ablations vs the 7a baseline.

### Metrics

- **Val PPL** (primary): token-weighted cross-entropy on WikiText-103 validation
- **Convergence speed**: val PPL at fixed step counts (e.g., 10M, 50M, 100M steps seen)
- **Output quality**: repeat-3gram, repeat-4gram, restart fragments, unique token ratio
- **Generation coherence**: qualitative assessment of multi-sequence samples

---

## File Layout

```
v7/
  __init__.py
  model.py      -- V7Config, complex primitives, CGU, PAM, V7Block, V7LM, presets
  data.py       -- TextDataset, load_wikitext103, load_tinystories, training utilities
  train.py      -- V7Trainer, CLI with ablation flags
  EXPERIMENTS_V7.md  -- this file
```

All code is self-contained. No imports from v5/ or v6/.

---

## Presets Summary

| Preset            | Layers | dim | heads | head_dim | expand | Params  | hierarchical_dt  | cross_level |
|-------------------|--------|-----|-------|----------|--------|---------|------------------|-------------|
| tiny              | 2      | 64  | 2     | 32       | 2      | 6.6M    | False            | False       |
| medium            | 16     | 384 | 6     | 64       | 3      | 100.4M  | True (linspace)  | False       |
| medium_h6         | 6      | 512 | 8     | 64       | 4      | 105.0M  | True (explicit)  | True        |
| medium_h16_flat   | 16     | 384 | 6     | 64       | 3      | ~100M   | False            | False       |
| medium_h16_grouped| 16     | 384 | 6     | 64       | 3      | ~100M   | True (grouped)   | True        |

---

## Running

```bash
# Smoke test (CPU, tiny preset)
uv run python -m v7.train --preset tiny --epochs 2 --max_samples 100 --dataset tinystories

# Full system (medium_h6, hierarchy + drift) — STOPPED epoch 3, Val PPL 98.3
uv run python -m v7.train --preset medium_h6 --epochs 10

# Experiment 3a: flat baseline (V6-matched depth, no hierarchy)
./scripts/run_v7_medium_h16_flat.sh

# Experiment 3b: grouped hierarchy (depth + timescale specialization)
./scripts/run_v7_medium_h16_flat.sh --preset medium_h16_grouped

# Experiment 7a: ModSwish CGU activation (flat baseline)
./scripts/run_v7_exp7a_swish.sh

# Experiment 7c: Chunked dual form (throughput optimization)
./scripts/run_v7_exp7c_chunked.sh

# Experiment 7d: Chunked + larger batch (exploit memory savings)
./scripts/run_v7_exp7d_chunked_b6.sh

# Experiment 7e: Chunked + soft unitary regularization
./scripts/run_v7_exp7e_unitary.sh

# 16-layer medium with auto-hierarchy (linspace schedule)
uv run python -m v7.train --preset medium --epochs 10
```

---

## Experiments 7c/7d/7e: Throughput Optimization (Chunked Dual Form)

### Motivation

PAM training uses an O(T^2) dual form per layer. For T=2048, this materializes 2048x2048 decay/attention matrices per head — ~600 MB of peak intermediates per layer. This is the dominant bottleneck for both speed and memory:
- **~20.9k tok/s** (V7 7a) vs **~96k tok/s** (transformer baseline) — 4.6x slower
- Cannot fit B=6 without gradient checkpointing

### Design: Chunked Dual Form

Split T into chunks of size C. Within each chunk, use the existing dual form on C×C. Across chunks, propagate the d×d matrix state recurrently.

```
For each chunk c:
  1. Intra-chunk:  y_intra, S_chunk = dual_form(q_c, k_c, v_c, gamma_c)  [C×C]
  2. Inter-chunk:  y_inter = cum_decay * S_prev @ q_c                     [d×d matmul]
  3. y_c = y_intra + y_inter
  4. S_new = total_decay * S_prev + S_chunk
```

**Mathematically identical** to full dual form (verified: max logit diff < 1e-7 in fp32).

### Concrete savings (C=256, T=2048)

| Metric | Full dual | Chunked | Ratio |
|--------|-----------|---------|-------|
| D matrix entries | 4.2M/head | 65K/chunk × 8 = 524K | **8× less** |
| Peak memory (D) | ~600 MB | ~9 MB | **64× less** |
| Extra cost | — | 8 state propagations (d²=4K, negligible) | — |

### Additional optimizations (applied in same change)

1. **Split-matmul ComplexLinear**: Replaced block-real GEMM (4 `torch.cat` ops + 1 large matmul) with 4 separate `F.linear` calls. Eliminates ~112 redundant tensor allocations per forward pass.
2. **Causal mask buffer**: `register_buffer` instead of `torch.tril(torch.ones(T,T))` per PAM call. Removes 16 identical allocations per forward.

### Experiment 7c: Chunked + B=3 (throughput baseline)

Same preset/hyperparameters as 7a (ModSwish, medium_h16_flat, B=3, no grad ckpt), just with chunk_size=256 and implementation optimizations. Expected: same PPL as 7a (~29.73), significantly higher tok/s.

**Status**: Not run yet.

### Experiment 7d: Chunked + B=6 (exploit memory savings)

Chunked dual form frees ~600 MB of peak intermediates per layer. This should allow B=6 without gradient checkpointing. Exp 3a-A (B=6 with grad ckpt) reached val PPL **26.6** at epoch 9. The transformer B=6 baseline reaches **23.13** at epoch 10 (see Learnings log). If 7d approaches or beats the transformer B=6 baseline, it validates both the chunking optimization and the batch size finding.

**Status**: **Done**. B=6 fits without gradient checkpointing, but quality does **not** match the earlier 3a-A checkpointed run.

| Epoch | Train PPL | Val PPL | tok/s |
|-------|-----------|---------|-------|
| 1 | 143.40 | 57.42 | 25.7k |
| 2 | 50.77 | 41.16 | 31.7k |
| 3 | 40.77 | 35.88 | 31.8k |
| 4 | 36.34 | 33.35 | 31.8k |
| 5 | 33.49 | 31.29 | 31.8k |
| 6 | 31.38 | 29.96 | 31.8k |
| 7 | 29.71 | 28.92 | 31.8k |
| 8 | 28.40 | 28.26 | 31.8k |
| 9 | 27.51 | 27.98 | 31.8k |
| 10 | 27.06 | **27.94** | 31.8k |

**Run summary**:
- **Best val PPL:** **27.94**
- **Best val loss:** **3.3300**
- **Throughput:** **31.8k tok/s** average
- **Wall time:** **10.70h** total
- **GPU memory:** **23.2 GB**
- **Generation quality:** `rep3=0.031`, `rep4=0.021`, `restarts=0`, `uniq=0.680`
- **Log:** `logs/v7/exp7d_chunked_b6_wikitext103_20260330_172839_7555c93/`

**Comparison**:
- vs **7a**: **29.73 -> 27.94** (**+6.0%** relative PPL improvement), **23.2k -> 31.8k tok/s** (**+36.7%** throughput)
- vs **Transformer B=3**: **27.94 vs 27.08** (close, but still behind)
- vs **Transformer B=6**: **27.94 vs 23.13** (still a clear gap)

### Experiment 7e: Soft unitary regularization

Add a loss term `λ * Σ ||W_r^T W_r + W_i^T W_i - I||²` over all ComplexLinear layers (λ=0.01). This encourages complex weights to stay near-unitary (norm-preserving), potentially improving gradient flow through 16 layers. Zero forward-pass overhead; small loss computation cost.

**Status**: Not run yet.

---

## Experiments 7f/7g/7h: Multi-Scale Loss + Reverse Association

### Motivation

V7 already has hierarchical `dt_bias` per layer (controlling memory span), but only a **single next-token CE loss on the final output**. The hierarchy influences forward dynamics, yet no layer gets direct supervision at its natural temporal scale. Additionally, the dual-form PAM computes full T×T `Q @ K.T` matrices but discards the upper triangle (causal mask zeros it out) — 50% of the matmul results wasted.

These experiments add two complementary features:
1. **Multi-scale per-layer loss**: Each layer predicts tokens at a temporal offset matching its memory span
2. **Reverse association**: Reuse the upper triangle of Q@K.T for a "what was the past looking for" signal

### Feature 1: Multi-Scale Per-Layer Loss (dT-Decay Loss)

**Core idea**: Each layer gets a lightweight auxiliary prediction head (`AuxPredHead`: `ComplexNorm -> to_real_concat -> Linear`) that predicts tokens at a temporal offset derived from its memory span. Global layers (slow decay, early) predict far ahead (t+32); step layers (fast decay, late) predict the immediate next token (t+1).

**Temporal offset schedule** (derived from layer position):
```
offset_i = max(1, int(max_aux_offset * (1 - i / (n_layers - 1))))
```
For a 16-layer model with `max_aux_offset=32`, `aux_layer_stride=3`:
- Layer 0:  offset=32 (paragraph-scale)
- Layer 3:  offset=26
- Layer 6:  offset=19
- Layer 9:  offset=13
- Layer 12: offset=7
- Layer 15: offset=1  (same as main next-token loss)

**Loss formulation**:
```
loss = main_CE + aux_weight * Σ_i(w_i * CE(aux_head_i(z_i), labels[t + τ_i]))
```
Where `w_i = exp(-2 * (1 - i/(n_layers-1)))` — exponential decay giving higher weight to later (easier) layers. `aux_weight` defaults to 0.1.

**Why this helps fact retention**:
- Global layers get direct gradient signal (without aux loss, they only get gradients through 15+ subsequent layers)
- Temporal offset forces abstraction: a layer predicting 32 tokens ahead must encode semantic/factual content
- No representation alignment issues: full backprop preserved (unlike forward-only approaches)

**Overhead**: ~77K params per head (dim=384, vocab=50257), 6 heads at stride=3 → ~462K extra params (~0.5% of total)

### Feature 2: Reverse Association (Upper Triangle Reuse)

**The waste**: In `_dual_form_block`, the `Q @ K.T` matmuls produce full T×T matrices, but the lower-triangular decay mask `D` zeros out the upper triangle. The upper triangle `wr[j, i]` for j < i = "how much does past query j match current key i" — a useful signal that's discarded.

**Reverse path**: Transpose the attention weights and apply the same causal mask:
```python
ar_rev = wr.transpose(-1, -2) * D   # reverse-causal weighting
y_rev = ar_rev @ v                   # "what the past wanted from me"
y = y_fwd + rev_scale * y_rev        # rev_scale: learnable per layer, init=0.0
```

**Cost**: One transpose (free), one extra element-wise multiply, one extra matmul per dual-form block. No new Q/K/V matmuls. ~25% extra FLOPs in the dual form block.

**Safety**: `rev_scale` initializes at 0.0 — the model starts as standard PAM and discovers the reverse signal via gradient descent. If it's not useful, `rev_scale` stays near zero.

**Note**: Reverse association only applies in training (dual form). Single-token recurrent inference has no upper triangle, so `rev_scale` is not used. Benefits persist through learned representations.

### Implementation Summary

**model.py changes**:
- `V7Config`: added `multi_scale_loss`, `aux_loss_weight`, `aux_layer_stride`, `max_aux_offset`, `use_reverse_assoc`
- `AuxPredHead` class: `ComplexNorm -> to_real_concat -> Linear(dim*2, vocab_size)`
- `V7LM.__init__`: creates `nn.ModuleDict` of aux heads keyed by layer index, computes per-head temporal offsets
- `V7LM.forward`: returns `(logits, states, aux_outputs)` where `aux_outputs` is `{layer_idx: (aux_logits, offset)}`
- `PhaseAssociativeLayer`: added `rev_scale` parameter; `_dual_form_block` and `_forward_chunked` pass and apply it

**train.py changes**:
- `compute_multi_scale_loss()`: computes weighted sum of shifted-label CE losses per aux head
- Training loop: integrates multi-scale loss when aux_outputs is non-empty
- CLI flags: `--multi_scale_loss`, `--aux_loss_weight`, `--aux_layer_stride`, `--max_aux_offset`, `--no_reverse_assoc`

### Experiment 7f-0: Multi-scale + grouped (confounded)

First run combined grouped hierarchy AND multi-scale loss vs the flat 7a baseline -- two variables changed at once.

| Epoch | Val PPL | Train PPL (inflated by aux loss) |
|-------|---------|----------------------------------|
| 1     | 60.46   | 915.08 |
| 5     | 35.76   | 246.52 |
| 10    | **31.29** | 191.72 |

**Result**: Val PPL 31.29 vs 7a baseline 29.73 -- regression. But we cannot tell whether grouped hierarchy or multi-scale loss (or both) caused it.

Generation quality was better than 7d: rep3=0.022 (vs 0.031), rep4=0.000 (vs 0.021), uniq=0.705 (vs 0.680).

### Ablation Plan (7f-1/7f-2/7f-3)

All ablations use the **7a baseline** as control: `medium_h16_flat`, B=3, ModSwish, chunk_size=256, no grad ckpt, 10 epochs. Each changes exactly ONE variable.

| Run  | What changes vs 7a     | Preset             | Key flags                               | Isolates                             |
|------|------------------------|--------------------|----------------------------------------|--------------------------------------|
| 7f-1 | Multi-scale loss only  | medium_h16_flat    | `--multi_scale_loss --no_reverse_assoc` | Does aux loss help on flat baseline? |
| 7f-2 | Reverse assoc only     | medium_h16_flat    | *(default: reverse_assoc=True)*         | Does upper-triangle reuse help?      |
| 7f-3 | Grouped hierarchy only | medium_h16_grouped | `--no_reverse_assoc`                    | Does grouped dt_bias help or hurt?   |

If 7f-1 and/or 7f-2 help, combine them in a follow-up (7f-4). Only add grouped hierarchy if 7f-3 shows it helps.

### Running

```bash
# Ablation 7f-1: Multi-scale loss only (flat, same as 7a + aux heads)
./scripts/run_v7_exp7f1_multiscale_flat.sh

# Ablation 7f-2: Reverse association only (flat, same as 7a + rev_assoc)
./scripts/run_v7_exp7f2_reverse_assoc.sh

# Ablation 7f-3: Grouped hierarchy only (no new features)
./scripts/run_v7_exp7f3_grouped_only.sh
```

**Status**: 7f-0 done (confounded). 7f-1/7f-2/7f-3 not run yet.
