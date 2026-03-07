# V6 Experiment Log

Structured log of architecture decisions, implementation notes, bug fixes, and training runs for the V6 phase-first model.

---

## 1. Architecture Decisions

### Removed from V5

| Component | Why Removed |
|-----------|-------------|
| `PhaseAttention` | Makes model O(n²). V6 targets strict O(n). |
| `attn_every_k` config | No attention anywhere. |
| `AlgebraicBank` / `AlgebraicFusion` | Replaced by named banks + phase interference coupler. |

### Revived from V4

| Component | V4 Version | V6 Version | What Changed |
|-----------|-----------|------------|-------------|
| Named banks | `SemanticPhaseBank` + `ContextPhaseBank` | `SemanticBank` + `ContextBank` | Rebuilt with `ComplexGatedUnit` (V5-safe ops), no phase-breaking activations |
| Interference coupler | `InterferenceCoupler` | `PhaseInterferenceCoupler` | Uses `cnormalize` for unit-complex rotations, `cabs` for magnitude features |
| Associative memory | `PhaseAssociativeMemory` | `InternalMemory` | Phase-spread key init, `ComplexLinear` query projection, `ComplexNorm` output |

### New in V6

| Component | Description |
|-----------|-------------|
| Multi-timescale SSM | `state_dim` partitioned into fast (40%), medium (30%), slow (30%) decay lanes |
| Working memory | Per-sequence differentiable scratchpad with learned write/read projections |
| Persistent memory | Per-user, cross-session tensors, saved/loaded from disk |
| Expert memory | Shared read-only domain knowledge tensors |
| Session memory | Optional between-turn buffer (`--session_memory` flag, disabled by default) |
| Memory fusion | Dynamic source-count routing (2/3/4 sources) with learned complex mixing |
| Memory adaptation | Persistent memory as soft training signal for internal memory alignment |

---

## 2. Implementation Notes

### Forked from V5 (unchanged)

- `core/complex.py`: All complex primitives (`cmul`, `cconj`, `cabs`, `cnormalize`, `creal_dot`, `ModReLU`, `ComplexLinear`, `ComplexNorm`, `ComplexGatedUnit`, `ComplexEmbed`)
- `core/ssm.py`: `ComplexSSMLayer` and `ComplexSSM` with parallel scan (extended with `_default_multiscale_init`)
- `init.py`: All 13 init strategies (extended with `init_ssm_eigenvalues_multiscale` and `init_internal_memory_slots`)

### Multi-timescale SSM initialization

```python
# Fast lanes (40%): decay 0.9-0.99 -- recent tokens, grammar
fast = torch.linspace(math.log(0.9), math.log(0.99), n_fast)

# Medium lanes (30%): decay 0.999-0.9999 -- sentence coherence
medium = torch.linspace(math.log(0.999), math.log(0.9999), n_medium)

# Slow lanes (30%): decay 0.99999+ -- near-persistent facts
slow = torch.linspace(math.log(0.99999), math.log(0.999999), n_slow)
```

HiPPO, S4DLin, and S4DInv strategies were adapted to map their eigenvalue distributions into the three tiers.

### Internal memory key initialization

Keys are spread across phase space (one angle per slot) with small random perturbation for per-dimension diversity. Values are small random (learned during training).

```python
angles = torch.linspace(0, 2π * (1 - 1/num_slots), num_slots)
keys_real = cos(angles).expand(num_slots, dim) * 0.1 + randn * 0.01
keys_imag = sin(angles).expand(num_slots, dim) * 0.1 + randn * 0.01
```

### Working memory write gate bias

Initialized to -2.0 (via `bias_real.fill_(-2.0)`) so the model starts by NOT writing to memory slots. Selective writing is learned during training.

### MemoryFusion dynamic source count

The model instantiates three `MemoryFusion` modules (`memory_fusion_2`, `_3`, `_4`) to handle different combinations of active memory types at runtime:
- 2 sources: working + internal (default)
- 3 sources: + persistent OR + expert
- 4 sources: + persistent AND expert

---

## 3. Bugs Found and Fixed

### Bug 1: Coupler router input shape (found during forward pass test)

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x64 and 128x64)`

**Root cause**: In `core/coupler.py`, magnitude features were computed as `cabs(s).mean(dim=-1)`, which collapsed the `dim` dimension. The router expected `[B, L, dim*num_sources]` but got `[B, L*num_sources]`.

**Fix**: Changed to `mag_features = [cabs(s) for s in sources]` (preserving the `dim` dimension).

### Bug 2: Working memory in-place operations (found during backward pass test)

**Error**: `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

**Root cause**: `WorkingMemory.forward` used in-place tensor assignment (`slot_keys[b, idx] = ...`) inside a loop, breaking the computation graph.

**Fix**: Refactored to fully differentiable soft addressing. All write projections computed for the entire sequence at once. Top-K tokens selected by gate magnitude, then soft-blended into new slot tensors using `selected_gates * selected_keys + (1 - selected_gates) * existing_keys`. Zero in-place operations.

### Bug 3: Diversity loss L1/L2 norm (carried over from V5)

**Error**: `div=0.0000` throughout training -- diversity loss was a no-op.

**Root cause**: Same bug as V5 (documented in `v5/EXPERIMENTS.md` §10). The denominator of cosine similarity used L1 norm (`cabs(a).sum(dim=-1)`) instead of L2 norm. By Cauchy-Schwarz, `(Σ|a_i|)² ≥ Σ|a_i|²`, so the denominator was systematically too large, making even identical banks appear diverse.

**Fix**: Changed to `torch.sqrt(cabs(sem_flat).square().sum(dim=-1) + 1e-8)` for both `sem_mag` and `ctx_mag`.

**Verification**: After fix, diversity loss starts at ~1.6 and converges to ~0.0006, confirming banks are actually specializing.

---

## 4. Test Results

### Comprehensive Module Test (2026-03-07)

Python script test covering all V6 components:

| Test | Result |
|------|--------|
| Import all V6 modules | ✅ Pass |
| Model instantiation (`tiny` config) | ✅ Pass |
| Forward pass (no external memory) | ✅ Pass |
| Forward pass (with persistent + expert memory) | ✅ Pass |
| Backward pass + gradient check (working memory) | ✅ Pass |
| Backward pass + gradient check (internal memory) | ✅ Pass |
| Text generation (with all memory types) | ✅ Pass |
| Persistent memory save/load roundtrip | ✅ Pass |
| Expert memory save/load roundtrip | ✅ Pass |
| Diversity loss L2 norm correctness | ✅ Pass |

### Diversity Loss Fix Verification

After the L2 norm fix, tested with a synthetic input:

```
Before fix: div=0.0000 (L1 norm -- denominator too large)
After fix:  div=1.6016 initially → 0.0006 after training (banks specialized)
```

---

## 5. Training Runs

### Run v6-smoke (2026-03-07, Mac CPU)

Quick smoke test to validate V6 can learn.

**Setup**: size=`tiny`, batch_size=4, epochs=2, max_samples=100, seq_len=128, init=orthogonal, CPU.

| Metric | Value |
|--------|-------|
| Parameters | 7,302,935 (7.3M) |
| Epoch 1 Val PPL | 253 |
| Epoch 2 Val PPL | ~200 |
| Wall time | ~2 min |

**Outcome**: Model learns (PPL drops from 51K to ~200). Forward pass, backward pass, and generation all work. Sufficient to proceed to bigger test.

### Run v6-cpu-test (2026-03-07, Mac CPU)

Bigger CPU test with more data and epochs.

**Setup**:
- size=`tiny` (dim=64, state=128, 4 layers, 2 banks)
- batch_size=4
- epochs=5
- max_samples=2000 (419K tokens, 2926 train chunks)
- seq_len=128
- init=orthogonal (seed=1934833307)
- device=CPU (MPS disabled)

**Parameters**:

| Component | Params |
|-----------|--------|
| embed (tied w/ output) | 6,432,896 |
| banks | 394,752 |
| couplers | 133,640 |
| ssm | 199,748 |
| working_memory | 24,770 |
| internal_memory | 16,448 |
| persistent_reader | 8,256 |
| expert_reader | 8,256 |
| memory_fusion | 75,785 |
| lm_head_proj | 8,384 |
| **total** | **7,302,935** |

**Results**:

| Epoch | Train PPL | Val PPL | Train-Val Gap | Time | div |
|-------|-----------|---------|---------------|------|-----|
| 1 | 141.76 | 77.28 | Val better | 269s | 1.60→0.001 |
| 2 | 52.85 | 54.56 | +1.7 | 266s | 0.001 |
| 3 | 39.27 | 46.12 | +6.9 | 265s | 0.001 |
| 4 | 32.95 | 42.67 | +9.7 | 250s | 0.001 |
| 5 | 30.13 | **42.18** | +12.1 | 260s | 0.001 |

**Total wall time**: 1347.8s (22.5 min), throughput ~11 samples/s (~1400 tok/s) on CPU.

**Observations**:

1. **Model learns without attention**: PPL dropped from 51K to 42 val in 5 epochs.
2. **Diversity loss works**: Converged from 1.6 to 0.0006 -- the L2 norm fix is effective, banks are specializing.
3. **Val PPL still improving** every epoch (77→54→46→42.7→42.2) -- never went up.
4. **Train-val gap widening**: Expected with 7.3M params on 2000 stories. Not yet overfitting (val still dropping), but flattening. More data would fix this.

**Generation samples** (prompt: "The quick brown"):

- **Epoch 1**: *"The quick brown was playing with the park, but he had been you." At make a little girl and said. They ran to do them with his mom and dad on a big man.*
- **Epoch 3**: *"The quick brown and had an idea. She started to find a lot of it, like a shiny friend, the small girl named Lily. He looked up and put them on her back home all the other fish."*
- **Epoch 5**: *"The quick brown and soon, feeling scared at her mommy. When Joe were two of other friends, and started to cry [...] She ran to help him [...] He was happy that they would never made about to go back."*

Generation quality improves across epochs: epoch 1 is barely coherent fragments, epoch 3 shows character names and narrative structure, epoch 5 has emotional arcs and multi-character interaction.

### Comparison: V6 vs V5 no-attention smoke test

Both tested on Mac CPU with comparable tiny configs:

| Run | Attention | Params | Epoch 2 Val PPL | Data | Notes |
|-----|-----------|--------|-----------------|------|-------|
| V5 mac-smoke-no-attn-cpu | None | 7.1M | 193.00 | 100 samples, seq_len=32 | V5 architecture minus attention |
| V6 v6-smoke (epoch 2) | None | 7.3M | ~200 | 100 samples, seq_len=128 | V6 full architecture |
| V6 v6-cpu-test (epoch 2) | None | 7.3M | 54.56 | 2000 samples, seq_len=128 | V6 with 20x more data |

Not directly comparable (different data amounts and seq_len), but confirms V6 learns at least as well as V5-without-attention on similarly tiny tests.

---

## 6. Experimental Validation Plan

From the architecture plan. Status updated as runs complete.

| Run | Description | Status | Notes |
|-----|-------------|--------|-------|
| A | V5 no-attention baseline (GPU) | ❌ Pending | Needs RTX 4090 run |
| B | V6 multi-timescale SSM only, no working memory | ❌ Pending | `--no_working_memory` |
| C | V6 with multi-timescale SSM + working memory | ❌ Pending | Full model |
| D | V6 with working memory + internal memory | ❌ Pending | Full model (already default) |
| E | V6 with persistent memory save/load | ❌ Pending | Module test passed, needs real training |
| F | V6 with expert memory | ❌ Pending | Module test passed, needs real training |
| G | V6 with `--session_memory` | ❌ Pending | Only if C-D show insufficient multi-turn context |
| H | V6 with adaptation pathway | ❌ Pending | Stretch goal |

**Key coherence metric**: generate stories of 500+ tokens, measure whether character names and settings from the first 50 tokens appear correctly in the last 100 tokens.

---

## 7. Next Steps

1. **GPU training** (RTX 4090 / A6000): Run `small-matched` config on 100K+ TinyStories samples for meaningful PPL comparison against V5.
2. **Working memory ablation**: Compare `--no_working_memory` vs default to isolate working memory's contribution.
3. **Long-context coherence test**: Generate 500+ token stories and measure fact retention.
4. **Persistent memory round-trip**: Train, save persistent memory, load in new session, verify improved generation.
5. **Full TinyStories run**: Scale to 2.1M samples to compare against V5's best val PPL of 5.59.

---

## How to Update

1. **Architecture Decisions**: Add rows when design choices are made.
2. **Implementation Notes**: Document non-obvious implementation details.
3. **Bugs Found and Fixed**: Log every bug with error, root cause, and fix.
4. **Training Runs**: Add detailed tables for each significant run.
5. **Experimental Validation Plan**: Update status as runs complete.
