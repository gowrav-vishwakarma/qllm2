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
- **ComplexLinear**: Block-real GEMM fusing four matmuls into one. Orthogonal initialization with gain=1/sqrt(2).
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

## Comparison Plan

Three runs to isolate each architectural contribution:

| Run | cross_level | hierarchical_dt | Description |
|-----|-------------|-----------------|-------------|
| A   | True        | True            | Full system (hierarchy + drift) |
| B   | False       | True            | Hierarchy only (no drift) |
| C   | False       | False           | Flat baseline (uniform dt_bias, no drift) |

An additional comparison against the V6 transformer baseline (~100M params, same data pipeline) provides the external benchmark.

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

| Preset     | Layers | dim | heads | head_dim | expand | Params  | hierarchical_dt | cross_level |
|------------|--------|-----|-------|----------|--------|---------|-----------------|-------------|
| tiny       | 2      | 64  | 2     | 32       | 2      | 6.6M    | False           | False       |
| medium     | 16     | 384 | 6     | 64       | 3      | 100.4M  | True (linspace) | False       |
| medium_h6  | 6      | 512 | 8     | 64       | 4      | 105.0M  | True (explicit) | True        |

---

## Running

```bash
# Smoke test (CPU, tiny preset)
uv run python -m v7.train --preset tiny --epochs 2 --max_samples 100 --dataset tinystories

# Full system (medium_h6, hierarchy + drift)
uv run python -m v7.train --preset medium_h6 --epochs 10

# Ablation: hierarchy only, no drift
uv run python -m v7.train --preset medium_h6 --no_cross_level --epochs 10

# Ablation: flat baseline
uv run python -m v7.train --preset medium_h6 --no_cross_level --no_hierarchical_dt --epochs 10

# 16-layer medium with auto-hierarchy
uv run python -m v7.train --preset medium --epochs 10
```
