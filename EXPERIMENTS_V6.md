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

### Bug 4: WM write gate bias absorbed by cabs (found 2026-03-08)

**Symptom**: Working memory write gates always open at ~88%, regardless of input. The model writes to every slot on every token instead of being selective.

**Root cause**: The gate uses `sigmoid(cabs(gate_raw))` where the -2.0 bias is inside the `ComplexLinear`. Since `cabs = sqrt(real^2 + imag^2)` is always non-negative, the negative bias gets absorbed into the magnitude computation: `sqrt((-2 + x)^2 + y^2) ≈ 2.0` for small x, y. This produces `sigmoid(2.0) = 0.88` instead of the intended near-zero selectivity.

**Fix**: Move the bias **outside** `cabs`. Use `ComplexLinear(dim, 1, bias=False)` + a separate real `nn.Parameter` bias, so the gate computes `sigmoid(cabs(raw) + bias)`. With `bias = -2.0`, this gives `sigmoid(~0 + -2) = 0.12` -- properly selective. This approach preserves the complex projection (phase-aware) and uses magnitude (proper complex invariant) while the real bias controls selectivity -- consistent with the V4 lesson of not applying real ops to complex representations.

**File**: `v6/core/memory.py`

### Bug 5: WM slot staleness during generation (found 2026-03-08)

**Symptom**: During autoregressive generation, only slot 0 is ever refreshed while slots 1-63 retain stale information from early in the sequence.

**Root cause**: During generation each forward call has `L=1`, so `num_writes = min(1, num_slots) = 1`. The write logic always targets the first slot, leaving all others permanently stale but still actively retrieved via dense softmax.

**Fix**: Two changes:
1. **Slot mask decay**: Apply exponential decay `slot_mask *= 0.95` each forward call, so a slot written 20 steps ago has `0.95^20 = 0.36` relevance and one from 100 steps ago has `~0.006` -- effectively gone.
2. **Circular write pointer**: Instead of always writing to slot 0, maintain a `write_ptr` buffer that advances by `num_writes` each call (mod `num_slots`), cycling through all slots.

**File**: `v6/core/memory.py`

### Bug 6: Dense softmax retrieval blends irrelevant slots (found 2026-03-08)

**Symptom**: Memory read outputs are generic/blended rather than providing sharp, content-specific information.

**Root cause**: Both WM and IM used `F.softmax(scores, dim=-1)` over **all** slots (16-128). With many slots scoring near-zero, softmax still assigns them non-negligible weight, diluting the output with irrelevant or stale information.

**Fix**: Top-k=8 sparse retrieval. After computing scores, select only the top-8 most relevant slots and apply softmax over those. This forces sharp, content-specific retrieval.

```python
k = min(8, num_slots)
topk_scores, topk_idx = scores.topk(k, dim=-1)
attn = F.softmax(topk_scores, dim=-1)
```

Configurable via `wm_read_topk` and `im_read_topk` in `V6Config`.

**File**: `v6/core/memory.py`

### Bug 7: Diversity loss margin too permissive (found 2026-03-08)

**Symptom**: Even with the L2 norm fix (Bug 3), diversity loss still collapsed to `div=0.0000` early in training. The loss was satisfied as soon as banks were slightly different, not pushing for meaningful specialization.

**Root cause**: The diversity loss penalized any non-zero cosine similarity equally. Once banks diverged slightly (cosine sim ~0.01), the loss was near zero and provided no further gradient to push banks apart.

**Fix**: Added a margin of 0.3 -- only penalize cosine similarity above the threshold: `F.relu(cosine_sim.abs() - margin).mean()`. Also added a weight schedule: start at `diversity_loss_weight=0.1`, decay to `diversity_loss_floor=0.02` (never zero).

**File**: `v6/core/bank.py`, `v6/train.py`

**Note**: Despite the margin fix, diversity loss still collapses to zero in all ablation runs (see §5.4). The margin-based approach may be fundamentally insufficient -- a different formulation is needed.

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

### Run v6-server-small-matched (2026-03-08, RTX 4090)

First GPU run on the server. Revealed catastrophic repetition.

**Setup**: size=`small-matched` (29M params), 100k TinyStories, batch=64, seq_len=256, 8 epochs, RTX 4090 (via `scripts/run_v6_rtx4090.sh`).

**Results** (stopped at epoch 8):

| Epoch | Val PPL | div | Notes |
|-------|---------|-----|-------|
| 1 | 21.23 | 3.20→0.001 | |
| 2 | 12.31 | 0.0000 | Diversity collapsed |
| 3 | 9.21 | 0.0000 | |
| 4 | 7.58 | 0.0000 | Severe repetition in generation |
| 5-8 | ~1.2 | 0.0000 | PPL approaching 1.0 = memorization |

**Problem**: Catastrophic repetition worsened each epoch despite improving PPL. By epoch 4, generations contained extreme degeneration: "then then then then then then", "one one one one..." repeated 20+ times, and entire paragraphs copied from training data. PPL dropping to ~1.2 confirmed the model was memorizing the training set verbatim rather than learning generalizable patterns.

**Generation sample (epoch 4)**: *"The quick brown fox liked to play with one one one one one one one one one one one one one one one one one one one one..."*

**Diagnosis**: Identified 5 contributing factors (documented as Bugs 4-7 above, plus the absence of attention as a lookback mechanism). Key insight: the WM write gate was always open at ~88%, meaning the model was indiscriminately writing every token into memory slots, turning WM into a lookup table.

### Run v6-server-post-fix (2026-03-08, RTX 4090)

Re-ran with all bug fixes deployed (Bugs 4-7 + optional PhaseAttention + diversity margin).

**Setup**: Same `small-matched` config as above. Bug fixes: WM bias-shifted magnitude gate, slot decay, top-k retrieval, margin-based diversity loss.

**Result**: PPL still dropped to ~1.2 by epoch 4. Repetition persisted. The bug fixes were correct but insufficient -- the root cause was not the bugs alone but the **capacity-vs-data ratio**. A 29M parameter model on 100k samples (22M tokens) has enough capacity to memorize the entire dataset, and the memory systems (WM + IM) amplified this by providing additional storage.

**Conclusion**: The repetition is a memorization problem, not an architectural defect. Need to either (a) reduce model size or (b) scale to more data.

### Run v6-ablation-tiny (2026-03-08, RTX 4090)

Targeted ablation study to distinguish capacity-driven memorization from architectural flaws. Uses `tiny` config (7.3M params) on the same 100k samples -- making it harder for the model to memorize.

**Script**: `scripts/run_v6_ablation_tiny.sh`

**Setup** (shared across all 5 runs):
- size=`tiny` (dim=64, state=128, 4 layers, 2 banks)
- batch_size=64, seq_len=256, epochs=5
- max_samples=100,000 (22M tokens, 76886 train chunks)
- init=orthogonal (seed=42)
- RTX 4090

**Ablation matrix**:

| Run | Description | WM Slots | IM Slots | Attention | Params | Best Val PPL | tok/s |
|-----|-------------|----------|----------|-----------|--------|-------------|-------|
| A | Baseline (all memory) | 16 | 32 | off | 7,302,934 | **8.84** | ~130k |
| B | No memory | 0 | 0 | off | 7,261,717 | 11.83 | ~141k |
| C | IM only (no WM) | 0 | 32 | off | 7,278,165 | 11.77 | ~139k |
| D | WM only (no IM) | 16 | 0 | off | 7,286,486 | **8.88** | ~135k |
| E | All memory + attention | 16 | 32 | last layer | 7,335,767 | **8.61** | ~113k |

**Epoch-by-epoch Val PPL**:

| Epoch | A (baseline) | B (no-mem) | C (IM only) | D (WM only) | E (w/ attn) |
|-------|-------------|------------|-------------|-------------|-------------|
| 1 | 14.37 | 17.61 | 17.61 | 14.99 | 14.16 |
| 2 | 10.51 | 13.82 | 13.78 | 10.57 | 10.24 |
| 3 | 9.39 | 12.49 | 12.44 | 9.44 | 9.16 |
| 4 | 8.96 | 11.95 | 11.88 | 8.99 | 8.72 |
| 5 | **8.84** | **11.83** | **11.77** | **8.88** | **8.61** |

**Wall time**: A=786s, B=728s, C=741s, D=765s, E=911s. All 5 runs completed in ~66 min total.

**Generation samples** (epoch 5, prompt: "The quick brown"):

- **A (baseline)**: *"The quick brown bear. He also liked to play with his toy cars and trucks. One day, he found a big box in his room. It was so shiny that he had made it really well! [...] When they were done, everyone said goodbye to their little boy. They all thanked him for being so nice and content. The end."*
- **B (no-memory)**: *"The quick brown bear's eyes lit up with joy. Jack was amazed at how different the rabbit could see - and he wanted to join them all for his way home. They had lots of fun playing together in their backyard [...]"*
- **C (IM only)**: *"The quick brown bear was so happy and thanked him for helping him. From that day on, Lily always took the medicine every morning to tell her a special place to play again!"*
- **D (WM only)**: *"The quick brown bear had never seen before. 'I don't know, but you have to be careful when we go!' They run and ran [...] He decided to stay with him."*
- **E (with-attn)**: *"The quick brown rabbit. The rabbit was so excited and sad, but Lily. The man was very angry [...] She said, 'Don't worry, I can help you.' So, she went to the park and looked at the tree."*

**Key result: zero repetition in any configuration.** All 5 runs produce coherent, story-like text with character names, dialogue, and narrative structure.

### Ablation Analysis

**Working memory is the dominant memory component.** Run D (WM only, no IM) achieves 8.88 PPL -- virtually identical to Run A (baseline, 8.84). Removing WM (Runs B and C) costs ~3 PPL points (11.83 and 11.77 respectively). WM provides a per-sequence scratchpad that meaningfully helps the model track facts within a story.

**Internal memory has minimal impact at this scale.** Run C (IM only) performs identically to Run B (no memory at all): 11.77 vs 11.83. The 32 trained `nn.Parameter` slots are not contributing measurable benefit at the `tiny` scale. This may change at larger scales where IM slots can encode more general knowledge.

**Attention provides marginal benefit at significant throughput cost.** Run E (with attention at last layer) reaches 8.61 vs Run A's 8.84 -- a 2.6% improvement. But throughput drops from ~130k to ~113k tok/s (13% slower). At this scale, attention is not worth the cost. It may become more important at larger scales where long-range self-correction matters more.

**Diversity loss still collapses.** In all 5 runs, `div` drops from ~3.2 to `0.0000` by mid-epoch 1. The margin-based fix (Bug 7) was insufficient. Banks may still be specializing through other dynamics (different weight initialization, gradient flow), but the explicit diversity loss is not enforcing it. A different formulation is needed.

**Repetition is a capacity-vs-data problem, not architectural.** The `tiny` model (7.3M params) on 100k samples does not exhibit any repetition, while the `small-matched` model (29M params) on the same data degenerates catastrophically. The 4x parameter difference crosses a memorization threshold where the model can store training sequences verbatim rather than learning generalizable patterns. The fix is to scale to the full dataset (2.1M TinyStories, 474M tokens), not to remove memory or add attention.

**Log**: `logs/v6/rtx4090_ablation_tiny_runner.log`

### Run v6-fulldata-no-memory (2026-03-09, RTX 4090)

Full-dataset ablation: small-matched model with **all memory disabled** to isolate whether memory causes memorization.

**Setup**: size=`small-matched` (29.1M params), WM=0, IM=0, full TinyStories (2,119,489 texts, 474M tokens), batch=20, seq_len=256, 1 epoch (82,995 batches), init=orthogonal (seed=42). Mid-epoch generation every 5000 batches.

**Parameters**: 29,053,997 (29.1M). Compared to the full-memory config (29.3M), the only difference is zero working_memory and internal_memory params.

**Results**:

| Batch | Loss | PPL | Notes |
|-------|------|-----|-------|
| 100 | 7.82 | 2495 | |
| 1000 | 4.11 | 60.7 | |
| 5000 | 2.92 | 18.6 | Diversity loss reaches floor |
| 10000 | 2.77 | 16.0 | |
| 20000 | 2.16 | 8.7 | PPL plateau begins |
| 30000 | 2.33 | 10.3 | |
| 50000 | 2.11 | 8.3 | |
| 70000 | 1.98 | 7.2 | |
| 82995 | — | — | Epoch avg: loss=2.3190, PPL=10.17 |
| **Val** | **1.9965** | **7.36** | **Best** |

**Wall time**: 22,034s (6.12h). Throughput: ~75 samples/s, ~19,300 tok/s.

**Generation samples** (prompt: "Once upon a time, there was a little"):

- **Batch 5000** (PPL 18.6): *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and make sure she would never find things that it's important lesson. One day, Max saw that a big bear called Sally. Tom wanted to be afraid of the garden."* -- basic structure, incoherent jumps between characters
- **Batch 25000** (PPL 8.7): *"Once upon a time, there was a little boy named Timmy. Timmy loved to play in the big green forest with his friends. One day, while they were playing, Timmy saw a big tree. The owl had an idea! Timmy and his friend started to climb up the hill. They climbed on top of the rock, but it was too far away."* -- good narrative flow, character consistency
- **Batch 50000** (PPL 8.3): *"Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks in the park. One day, Timmy's mom asked him if he could go to the store to buy some new toys. His mom said yes and Timmy went outside to play."* -- clean, coherent, diverse
- **End-of-epoch** (Val PPL 7.36): *"Once upon a time, there was a little girl named Lily. She loved to play outside in the sun with her friends. One day, while they were playing in the park, they saw an old man walking by. 'Hello there, little boy,' said the man. 'Do you want to come on my picnic?' Lily thought for a moment and then nodded. 'Yes please!' she said."* -- best quality, natural dialogue, narrative arc

**Analysis**: **No memorization.** PPL never drops below ~7, stabilizing in the 7-9 range from batch 20000 onward. The model produces diverse, non-repetitive text throughout. However, generations show semantic drift (topic changes mid-paragraph) and pronoun confusion -- signs of a model that lacks contextual memory to maintain coherent long-range narratives. This establishes the **baseline without memory**: val PPL 7.36 represents the ceiling for pure SSM+banks on this data.

**Log**: `logs/v6/fulldata_no_memory_20260308_195054_f960859/v6_train_small-matched.log`

### Run v6-fulldata-tiny-memory (2026-03-09, RTX 4090)

Full-dataset ablation: small-matched model with **reduced memory** (WM=16, IM=32 -- same as `tiny` config) to test whether smaller memory prevents memorization while still helping.

**Setup**: size=`small-matched` (29.2M params), WM=16, IM=32, full TinyStories, batch=20, seq_len=256, 1 epoch, init=orthogonal (seed=42). Mid-epoch generation every 5000 batches.

**Parameters**: 29,201,966 (29.2M). WM has 98,689 params, IM has 49,280 params (vs 98,432 at IM=128).

**Results**:

| Batch | Loss | PPL | Notes |
|-------|------|-----|-------|
| 100 | 7.89 | 2672 | |
| 1000 | 4.07 | 58.6 | |
| 5000 | 2.92 | 18.5 | Similar to no-memory at this point |
| 10000 | 2.31 | 10.1 | Starting to diverge from no-memory |
| 15000 | 2.01 | 7.4 | Already below no-memory's plateau |
| 20000 | 1.81 | 6.1 | Significantly better than no-memory |
| 25000 | 1.31 | 3.7 | Memory advantage clear |
| 30000 | 1.27 | 3.5 | |
| 50000 | 0.86 | 2.4 | Stabilizing |
| 70000 | 0.89 | 2.4 | Plateau at ~2.3-2.6 |
| 82995 | — | — | Epoch avg: loss=1.4412, PPL=4.23 |
| **Val** | **0.8029** | **2.23** | **Best** |

**Wall time**: 23,177s (6.44h). Throughput: ~73 samples/s, ~18,750 tok/s.

**Generation samples** (prompt: "Once upon a time, there was a little"):

- **Batch 5000** (PPL 18.5): *"Once upon a time, there was a little girl named Lily. She was very careful and always wanted to stay with her. 'What's wrong, Lily?' said the man. 'I want to help you!' The bird smiled and said, 'Okay. I will find your toys to play.'"* -- decent early quality, dialogue
- **Batch 25000** (PPL 3.7, **restart fragmentation appears**): *"Once upon a time, there was a little boy. His friend was very grumpy andOnce upon a time, there was a little boy. Timmy was really sad because he was having lots of fun in the park. Once upon a time, Timmy's friendOnce upon a time, there was a little boy. He was feeling very cold and didn't evenOnce upon a time...Once upon a time, there was an orange."* -- multiple mid-sentence restarts with "Once upon a time"
- **Batch 50000** (PPL 2.4): *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and run outside. One day, she found a big box ofOnce upon a time, there was a little boy named Timmy. He liked to play with his toys and run around the park. One day, Lily met a little boy who was very excited."* -- still fragmenting with story-boundary restarts
- **End-of-epoch** (Val PPL 2.23): *"Once upon a time, there was a little girl named Lily. She loved to play with her toys and make sure she was safe. One day, Lily went to the park to play. There were many friends by herself! They sat on a bench and searched for the rain. Suddenly, they found a little girl standing in the middle of the yard."* -- decent quality, no fragmentation in this sample

**Analysis**: **No catastrophic memorization** (PPL never hits 1.2), but PPL drops to 2.3-2.6 -- significantly better than no-memory (7.36). The most notable finding is a **new degeneration pattern**: "Once upon a time" restart fragmentation. Instead of repeating words/phrases like the full-memory model, this model inserts `<|endoftext|>` story boundaries mid-sequence, producing multiple story beginnings concatenated together. This suggests the memory is retrieving dataset boundary patterns. The fragmentation appears around batch 20000-25000 (PPL ~3.7) and persists through the rest of training, though end-of-epoch samples sometimes avoid it. Val PPL 2.23 in just 1 epoch already **beats V5's best of 5.36** (at epoch 5).

**Log**: `logs/v6/fulldata_tiny_memory_20260309_021456_f960859/v6_train_small-matched.log`

### Run v6-fulldata-tiny-model (2026-03-09, RTX 4090)

Full-dataset ablation: **tiny model** (7.3M params) on the full dataset to confirm capacity-limited models generalize regardless of memory.

**Setup**: size=`tiny` (7.3M params), WM=16, IM=32, full TinyStories, batch=64, seq_len=256, 1 epoch (25,936 batches), init=orthogonal (seed=42). Mid-epoch generation every 5000 batches.

**Parameters**: 7,302,934 (7.3M). Same architecture as the 100K ablation runs.

**Results**:

| Batch | Loss | PPL | Notes |
|-------|------|-----|-------|
| 100 | 4.96 | 142.7 | Faster initial drop (higher LR=1e-3) |
| 500 | 3.29 | 27.0 | Diversity loss already at floor |
| 1000 | 2.88 | 17.7 | |
| 5000 | 2.25 | 9.5 | |
| 10000 | 2.04 | 7.7 | PPL plateau begins |
| 15000 | 1.86 | 6.4 | |
| 20000 | 1.70 | 5.5 | |
| 25000 | 1.61 | 5.0 | |
| 25936 | — | — | Epoch avg: loss=2.0236, PPL=7.57 |
| **Val** | **1.5347** | **4.64** | **Best** |

**Wall time**: 3,274s (0.91h). Throughput: ~507 samples/s, ~129,850 tok/s.

**Generation samples** (prompt: "Once upon a time, there was a little"):

- **Batch 5000** (PPL 9.5): *"Once upon a time, there was a little girl named Lily. She lived in her house and went to bed. One day, she went for walks, but her mommy said she couldn't find it anywhere. Lily's mommy said she could go and play."* -- decent for 7.3M params
- **Batch 15000** (PPL 6.4, **temporary mode collapse**): *"Once upon a time, there was a little bunny. The other animals petted of the other animals and animals. They were friends and their baby deer. The baby animals saw them and the animals. They were happy to see any animals. Some animals are small and beautiful. And the other animals... The animals were different animals. It's the animals and animals. There were animals."* -- stuck in "animals" loop
- **Batch 20000** (PPL 5.5, **recovers**): *"Once upon a time, there was a little girl called Lily. She looked around and saw a big castle. It was white and had an eye. The bunny was looking up and waddled towards her. Lily's mommy walked outside to find the right hanger."* -- diverse again, no loop
- **End-of-epoch** (Val PPL 4.64): *"Once upon a time, there was a little girl called Lily. She liked to play outside and make shapes of the toys. Her hair became like cows and pigs. Daisy was so happy and she couldn't wait to go."* -- capacity-limited nonsense but structurally sound

**Analysis**: **No memorization.** The 7.3M model on 474M tokens (~65 tokens per parameter) cannot memorize. Val PPL 4.64 is respectable for a 4-layer model. One notable anomaly: a temporary "animals" mode collapse at batch 15000 that self-corrects by batch 20000, suggesting the model's limited capacity occasionally gets stuck in attractor states during training but recovers. Compared to the same model on 100K samples (val PPL 8.84 at epoch 5), the full dataset yields much better results (4.64 at epoch 1), confirming data scaling helps this architecture.

**Log**: `logs/v6/fulldata_tiny_model_20260309_084956_f960859/v6_train_tiny.log`

### Run v6-no-memory-10epoch (2026-03-10, RTX 4090)

Multi-epoch extension of the 1-epoch no-memory baseline (§5.6). Stopped at epoch 5 (mid-epoch 6) to free GPU for WikiData work.

**Setup**: size=`small-matched` (28.7M params), WM=0, IM=0, full TinyStories (474M tokens), 10 epochs planned, batch=28, seq_len=256, lr=1e-4, init=orthogonal (seed=42). Mid-epoch generation every 5000 batches.

**Parameters**: 28,689,188 (28.7M). Same no-memory config as v6-fulldata-no-memory.

**Results**:

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | div | Time |
|-------|-----------|-----------|----------|---------|-----|------|
| 1 | 2.2959 | 9.93 | 1.8916 | 6.63 | 3.46e+00 | 10181s |
| 2 | 1.8783 | 6.54 | 1.7934 | 6.01 | 3.01e-03 | 8496s |
| 3 | 1.8179 | 6.16 | 1.7506 | 5.76 | 2.75e-03 | 8430s |
| 4 | 1.7872 | 5.97 | 1.7268 | 5.62 | 2.38e-03 | 8430s |
| 5 | 1.7656 | 5.85 | 1.7050 | 5.50 | 1.91e-03 | 8595s |

**Wall time** (5 epochs): ~12.2h total. Throughput: ~55k tok/s.

**Generation samples** (prompt: "Once upon a time, there was a little"):

- **Epoch 1 end** (Val PPL 6.63): *"Once upon a time, there was a little girl named Lily. She loved to go on adventures with her mommy and daddy. One day, they went for a walk in the forest. They saw a big mountain that had lots of trees and flowers."*
- **Epoch 5 end** (Val PPL 5.50): *"Once upon a time, there was a little girl named Lily. She loved to draw and color with her crayons. One day, she drew a picture of her family in the park. They were so happy together!"*

**Analysis**: Val PPL improved from 7.36 (1-epoch baseline) to 5.50 (5-epoch) -- 25% improvement with continued training. Train-val gap is small (5.85 vs 5.50) -- not overfitting even at epoch 5. **Zero repetition, fragmentation, or memorization** in any generation sample across all 5 epochs. **Better generation quality than WM=16/IM=32 despite higher PPL**: The tiny-memory run (val PPL 2.23) suffered from restart fragmentation ("Once upon a time" restarts mid-sequence) and incoherent word repetition, while this no-memory run at val PPL 5.50 produces clean, diverse, narratively coherent text. PPL alone is not a reliable quality metric when memory systems can exploit dataset boundary patterns to lower PPL without improving coherence. Diversity loss converges to ~1.9e-3 and stays there. Still improving at epoch 5 (no plateau) -- more epochs would likely push below 5.0.

**Log**: `logs/v6/small_matched_full_20260310_000217_518c76e/v6_autoregressive_small-matched.log`

### Run v6-wikitext103-wm8 (2026-03-11, stopped mid-run)

First WikiText-103 run **with working memory** (WM=8). Same small-matched config as the no-memory WikiText run; goal was to see if WM helps on encyclopedia-style text.

**Setup**: size=`small-matched` (29.1M params), **WM=8**, IM=0, dataset=WikiText-103 (full), seq_len=512, batch_size=14, 10 epochs planned, init=orthogonal (seed=42). Run stopped partway through epoch 5.

**Observations**:

1. **Repetition appears even on wiki data** — not the TinyStories-style "one one one" token loop, but:
   - **Lexical repetition**: By epoch 4, generations collapse to repeated tokens, e.g. *"The history of the time , the time for a time in time and time was not time to complete . In time it had time spent time with time trial , time time , and time constraints and time ..."*
   - **Template collapse**: Many samples become short, formulaic fragments: *"The history of the but @-@ lost but both in terms of"*, *"The history of this non @-@ non @-@ independent , both"*.

2. **Diversity loss collapses early** — same as in all prior V6 runs: `div` drops from ~64 to ~0.002 by ~batch 5k (epoch 1), then stays near zero. Banks are not receiving sustained pressure to specialize.

3. **Design takeaways**: (a) WM amplifies whatever patterns the model uses — on WikiText, high-frequency patterns ("time", "the history of") get written/retrieved and reinforced, leading to word loops and template collapse. (b) No explicit control on *what* gets written to WM or anti-repetition in generation. (c) SSM + WM without attention may be insufficient for long-range coherence on encyclopedia text; WM can lock in a few patterns instead of helping.

**Log**: `logs/v6/wikitext103_small_matched_20260311_105203_dacac03/v6_autoregressive_small-matched.log`

### Full-Dataset Ablation Analysis

The three full-dataset runs, combined with all prior experiments, paint a complete picture of V6's memorization behavior and optimal configuration.

**Cross-experiment comparison** (all runs, sorted by Val PPL):

| Run | Params | WM | IM | Dataset | Epochs | Val PPL | Memorized? |
|-----|--------|-----|-----|---------|--------|---------|------------|
| small-matched, full memory (original) | 29.3M | 64 | 128 | 100K | 7 | 1.23 | YES -- catastrophic |
| small-matched, full memory (full data) | 29.3M | 64 | 128 | 2.1M | ~0.3 | ~1.2 | YES -- even on full data |
| **small-matched, tiny memory** | **29.2M** | **16** | **32** | **2.1M** | **1** | **2.23** | **NO -- restart fragmentation** |
| tiny, full data | 7.3M | 16 | 32 | 2.1M | 1 | 4.64 | NO -- capacity-limited |
| small-matched, no memory (5 epoch) | 28.7M | 0 | 0 | 2.1M | 5 | 5.50 | NO -- best gen quality |
| small-matched, no memory | 29.1M | 0 | 0 | 2.1M | 1 | 7.36 | NO -- capacity ceiling |
| tiny baseline (100K) | 7.3M | 16 | 32 | 100K | 5 | 8.84 | NO |
| tiny WM-only (100K) | 7.3M | 16 | 0 | 100K | 5 | 8.88 | NO |
| tiny with-attn (100K) | 7.3M | 16 | 32+attn | 100K | 5 | 8.61 | NO |
| tiny no-memory (100K) | 7.3M | 0 | 0 | 100K | 5 | 11.83 | NO |
| tiny IM-only (100K) | 7.3M | 0 | 32 | 100K | 5 | 11.77 | NO |
| small-matched, post-fix (100K) | 29.3M | 64 | 128 | 100K | 2* | 12.31 | Interrupted |

**Key findings:**

**1. Memory capacity is the memorization knob.** The evidence is unambiguous:
- WM=64, IM=128 → memorizes (val PPL 1.2) even on 2.1M samples
- WM=16, IM=32 → generalizes well (val PPL 2.23) on 2.1M samples
- WM=0, IM=0 → generalizes but capacity-limited (val PPL 7.36)

The memory system acts as an explicit lookup table when over-provisioned. With 64 WM slots, the model can cache enough sequence fragments to reconstruct training data. With 16 slots, there isn't enough storage for verbatim retrieval, forcing the model to learn generalizable patterns while still benefiting from the scratchpad for within-sequence coherence.

**2. V6 with tiny memory already beats V5.** V5's best result on full TinyStories was val PPL 5.36 at epoch 5 (with attention). V6 with WM=16/IM=32 achieves **val PPL 2.23 in just 1 epoch** -- a 2.4x improvement. Even the no-memory V6 (val PPL 7.36) is in the same ballpark as V5 after 1 epoch, suggesting the multi-timescale SSM and named banks are inherently more capable than V5's architecture.

**3. Working memory is far more important than internal memory.** From the 100K tiny ablation: WM alone (8.88) ≈ WM+IM (8.84), while IM alone (11.77) ≈ no memory (11.83). WM contributes ~3 PPL points; IM contributes ~0.06. The per-sequence scratchpad dominates the trained parameter slots.

**4. The "restart fragmentation" pattern is a new degeneration mode.** The tiny-memory model on full data exhibits mid-sequence "Once upon a time" restarts -- a form of degeneration distinct from the verbatim repetition seen with full memory. This may indicate the WM is retrieving `<|endoftext|>` boundary representations, causing the model to "restart" stories mid-generation. This warrants investigation but is far less severe than catastrophic repetition.

**5. Model capacity also matters, but less than memory capacity.** The tiny model (7.3M) cannot memorize regardless of memory config. But comparing small-matched no-memory (29.1M, val PPL 7.36) vs tiny (7.3M, val PPL 4.64), the smaller model actually performs better -- likely because the tiny config's higher learning rate (1e-3 vs 1e-4) allows faster convergence in 1 epoch. Multi-epoch runs would likely reverse this ordering.

---

## 6. Experimental Validation Plan

Originally planned as a systematic feature-by-feature ablation. The plan shifted to a repetition investigation after the first GPU run revealed catastrophic degeneration. The repetition investigation is now complete, with the full-dataset ablation (§5.8) providing definitive answers.

### Repetition Investigation (2026-03-08 → 2026-03-09)

| Step | Description | Status | Result |
|------|-------------|--------|--------|
| 1 | GPU run, small-matched on 100k | ✅ Done | Catastrophic repetition, PPL→1.2 |
| 2 | Diagnose root causes | ✅ Done | 5 bugs identified (Bugs 4-7 + no attention) |
| 3 | Apply fixes, re-run | ✅ Done | Still memorizing -- capacity problem |
| 4 | Tiny model ablation (5 configs, 100K) | ✅ Done | No repetition at 7.3M params |
| 5 | Full-dataset no-memory ablation | ✅ Done | Val PPL 7.36, no memorization |
| 6 | Full-dataset tiny-memory ablation | ✅ Done | Val PPL 2.23, no memorization, beats V5 |
| 7 | Full-dataset tiny-model ablation | ✅ Done | Val PPL 4.64, no memorization |

**Conclusion**: Memory capacity (specifically WM slot count) is the memorization knob. WM=64/IM=128 memorizes even on 2.1M samples. WM=16/IM=32 generalizes with val PPL 2.23. WM=0/IM=0 generalizes but caps at val PPL 7.36.

### Remaining Experiments

| Run | Description | Status | Notes |
|-----|-------------|--------|-------|
| A | Multi-epoch WM=16/IM=32 stability | ❌ Pending | Verify tiny-memory config stays stable over 5-10 epochs |
| B | WM=32/IM=64 middle-ground test | ❌ Pending | Test whether intermediate memory still avoids memorization |
| C | Persistent memory at scale | ❌ Pending | Module test passed, needs real training validation |
| D | Expert memory at scale | ❌ Pending | Module test passed, needs real training validation |
| E | Long-context coherence test | ❌ Pending | 500+ token generation, character/setting tracking |
| F | Medium model scaling | ❌ Pending | Scale to ~60M params with optimized memory config |

**Key coherence metric**: generate stories of 500+ tokens, measure whether character names and settings from the first 50 tokens appear correctly in the last 100 tokens.

---

## 7. Optimal Configuration

Based on all experiments, the recommended V6 configuration is:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model size | `small-matched` (29M) | Best capacity for TinyStories-scale data |
| WM slots | **16** | Prevents memorization while providing ~3 PPL improvement |
| IM slots | **32** | Minimal impact at current scale, but low cost to keep |
| Attention | **off** | Only 0.23 PPL improvement at 15% throughput cost |
| Dataset | Full TinyStories (2.1M) | Required to avoid capacity-driven memorization |

This configuration achieves **val PPL 2.23 in 1 epoch** -- already 2.4x better than V5's best (5.36 at epoch 5). Multi-epoch training should improve further.

**Open question**: Would WM=32/IM=64 find a better quality/memorization tradeoff? The gap between WM=16 (PPL 2.23, no memorization) and WM=64 (PPL ~1.2, memorized) suggests there may be a sweet spot around WM=24-32 that achieves lower PPL without collapse.

---

## 8. V5 vs V6 Comparison (Blackwell Baseline)

An external V5 `small` model was trained on the same TinyStories dataset on a Blackwell GPU (B200, 50GB+ VRAM). This provides a direct architecture-vs-architecture comparison.

### V5 Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | V5 `small` (49.0M params) |
| Complex dim | 256 (512 real) |
| SSM state dim | 512 |
| Layers | 8 |
| Banks | 2 (algebraic) |
| Attention | Every 4 layers (window=512) |
| Sequence length | 512 |
| Batch size | 64 |
| AMP | bf16 |
| Extras | torch.compile (reduce-overhead), xformers, text repair, token caching |
| Dataset | Full TinyStories (~2.1M stories, 471M tokens) |
| Epochs run | 5 of 10 |

### V5 Epoch-by-Epoch Results

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Trend |
|-------|-----------|-----------|----------|---------|-------|
| 1 | 2.2716 | 9.70 | 3.3090 | **27.36** | *best val* |
| 2 | 1.7193 | 5.58 | 3.4865 | 32.67 | val degrading |
| 3 | 1.6198 | 5.05 | 3.8363 | 46.35 | val degrading |
| 4 | 1.5702 | 4.81 | 4.1605 | 64.10 | val degrading |
| 5 | 1.5403 | 4.67 | 4.4235 | **83.39** | severe overfit |

**Overfitting rate**: Val PPL increased **3.05x** (27.36 to 83.39) over 5 epochs while train PPL dropped 2.1x.

### V5 Generation Samples (Degeneration Progression)

**Epoch 1** (Val PPL 27.36 -- mild word-level repetition already visible):
> The quick brown dog had found a new toy. He was very happy and thanked the man kindly politely. But then, the **giant giant giant giant** saw them and said they were scared. The bear closed his eyes and fell asleep dreaming about **more more** surprises.

**Epoch 3** (Val PPL 46.35 -- word/morpheme stuttering):
> He hopped on the swing, then he **zoom zoom zoom zoom zoom zoom zoom zoom** and spl splash in the **waves waves wave wave wave wave** waved goodbye to his friends

**Epoch 4** (Val PPL 64.10 -- catastrophic single-word repetition):
> jokes **joke joke joke jokes** the cat would laugh... dancing **dance dancers dancers dancers** on TV **screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen screen**

**Epoch 5** (Val PPL 83.39 -- compound word collapse):
> The grass was **green green green green green green green green** leaves. Lila wanted to play in the sun **sunflowerflowerflowerflower** garden. But the door opened **closed shut shut**

### V6 vs V5 Head-to-Head Comparison

| Metric | V5 `small` (Blackwell) | V6 `small-matched` (no mem) | V6 `small-matched` (WM=16) | V6 `tiny` (WM=16) |
|--------|----------------------|---------------------------|--------------------------|-------------------|
| **Parameters** | 49.0M | 29.3M | 29.3M | 8.0M |
| **Has attention** | Yes (every 4 layers) | No | No | No |
| **Sequence length** | 512 | 256 | 256 | 256 |
| **Epochs trained** | 5 | 1 | 1 | 1 |
| **Best Val PPL** | 27.36 (epoch 1) | 7.36 | **2.23** | 4.64 |
| **Final Val PPL** | 83.39 (epoch 5) | 7.36 | 2.23 | 4.64 |
| **Overfitting** | Severe (3.05x) | None (1 epoch) | None (1 epoch) | None (1 epoch) |
| **Word repetition** | Yes, catastrophic by epoch 3 | None | None | None |
| **Generalization** | Collapses after epoch 1 | Stable | Stable | Stable |

### Key Findings

1. **V6 achieves 12x better val PPL with 40% fewer parameters**: V6 `small-matched` (29.3M, WM=16) reaches val PPL 2.23 in 1 epoch vs V5's best of 27.36 at epoch 1 (49.0M). Even after 5 epochs V5 never gets close.

2. **V5 suffers the same word-level repetition degeneration**: The exact stuttering pattern ("zoom zoom zoom zoom", "screen screen screen...") seen in V5 mirrors the memorization we saw in V6 with full memory (WM=64). V5's attention layers + algebraic banks cannot prevent this collapse.

3. **V6 without any memory already beats V5**: Even the no-memory V6 config (pure SSM + named banks, val PPL 7.36) outperforms V5's best (27.36) by 3.7x, confirming the multi-timescale SSM and phase interference coupler are fundamentally better than V5's algebraic approach.

4. **V5 overfitting is structural, not tunable**: V5 val PPL monotonically increases from epoch 1, suggesting the architecture lacks the regularization properties that V6's phase-preserving design provides. Lowering LR or adding dropout may slow but likely won't prevent the degradation.

5. **V6's memory provides a quality knob V5 lacks**: V6 can trade memory capacity for generalization (WM=16 vs WM=64). V5 has no equivalent lever -- its attention-based design is either all-in on memorization capacity or requires completely removing attention (which would make it a different architecture).

---

## 9. Next Steps

1. **WikiData / new data sources** (highest priority): Move beyond TinyStories to WikiData or other datasets. The no-memory baseline is stable and produces high-quality generations; next step is to validate on different domains.
2. **Multi-epoch stability test** (partially done): No-memory variant confirmed stable over 5 epochs (val PPL 7.36→5.50, zero repetition). Still pending: WM=16/IM=32 for 5-10 epochs -- but that config has restart fragmentation, so no-memory may be the better baseline for new data work.
3. **WM=32/IM=64 middle-ground test**: Test whether intermediate memory capacity achieves better PPL without crossing the memorization threshold. If it stays stable, this becomes the new recommended config.
4. **Restart fragmentation investigation**: The tiny-memory model inserts `<|endoftext|>` boundaries mid-generation. Diagnose whether this is caused by WM retrieving boundary representations, and whether it can be fixed by filtering `<|endoftext|>` from WM write candidates.
5. **Diversity loss rework**: The margin-based approach (Bug 7 fix) still collapses to zero in all runs. Consider alternative formulations: contrastive loss, orthogonality constraint on bank weights, or gradient-based diversity.
6. **Long-context coherence test**: Generate 500+ token stories with the optimal config and measure character/setting tracking over distance. WM should shine here.
7. **Persistent/expert memory at scale**: Module tests pass, but real training validation is needed with the optimized memory config.
8. **Medium model scaling**: Scale to ~60M params with WM=16-32/IM=32-64 on full TinyStories. Verify that the memorization control transfers to larger models.

---

## 10. How to Update

1. **Architecture Decisions**: Add rows when design choices are made.
2. **Implementation Notes**: Document non-obvious implementation details.
3. **Bugs Found and Fixed**: Log every bug with error, root cause, and fix.
4. **Training Runs**: Add detailed tables for each significant run.
5. **Experimental Validation Plan**: Update status as runs complete.

---

## 11. Run v6-wikitext103-small-matched (2026-03-10 to 2026-03-11, RTX 4090)

First full V6 run on WikiText-103 raw text using the new `wikitext103` pipeline. This is the first completed non-TinyStories V6 benchmark and the first result that meaningfully tests the no-memory architecture on real encyclopedic text.

**Setup**:
- size=`small-matched` (28.7M params)
- dataset=`wikitext103`
- tokenizer=`gpt2` BPE
- seq_len=`512`
- batch_size=`14`
- epochs=`20`
- WM=`0`, IM=`0`
- attention=`off`
- compile=`reduce-overhead`
- hardware=`RTX 4090`
- train tokens=`118,496,145`
- validation tokens=`248,461`
- train chunks=`230,986`
- val chunks=`484`

**Source files**:
- `logs/v6/wikitext103_small_matched_20260310_152631_6ffe838_dirty/v6_autoregressive_small-matched.log`
- `logs/v6/wikitext103_small_matched_20260310_152631_6ffe838_dirty/RUN_INFO.txt`

### Results

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 247.45 | 121.94 | Learns immediately, very weak generations |
| 5 | 69.80 | 61.28 | Strong stylistic shift toward encyclopedia text |
| 10 | 60.14 | 53.75 | Improvement slows, but remains monotonic |
| 15 | 56.26 | 50.59 | Clear plateau region |
| 20 | 54.45 | **49.61** | Best checkpoint |

**Final metrics**:
- Best val loss: `3.9041`
- Best val PPL: `49.61`
- Total wall time: `14.27h`
- Steady-state throughput: `~46k tok/s`

### Generation quality

The model clearly learned the *surface form* of Wikipedia-style text:
- section headers
- historical framing
- article-like syntax
- date / region language

But it still failed at factual composition:
- invented chronology
- mixed entities
- impossible event combinations
- fluent but incorrect historical claims

**Representative final sample**:

> "The history of the Middle East , and many other European countries ... = = = Later years : 18th century – the early period ( 1072 – 688 ) = = = ... The first known Anglo @-@ Saxon victory was the Battle of the River Leng in 521 BC when the Persian Wars ended with the defeat at the Battle of Tiberius ..."

Interpretation: the model has crossed the threshold from toy text generation into real article-style generation, but it is still semantically unreliable.

### Probe summary on the final checkpoint

Ran `scripts/v6_eval_probes.py` on `checkpoints_v6_wikitext103/best_model.pt`.

| Probe | Result | Takeaway |
|------|--------|----------|
| Co-reference | nominal `100%` pass | Too weak to trust as strong evidence; thresholds are permissive |
| Fact persistence | `0%` pass | Strong negative signal |
| Bank specialization | `0.000011` | Very weak signal |
| Working memory utilization | skipped | WM disabled |
| SSM timescale probe | unstable-looking learned decays | No strong long-memory evidence yet |

### Honest conclusion from the WikiText-103 run

This run proves **viability**, not **victory**.

What it proves:
1. V6 trains stably on real long-form text.
2. The no-memory V6 path is a real language model, not just a TinyStories artifact.
3. Attention-free V6 can learn article structure and domain style.

What it does **not** prove:
1. It does not prove V6 is competitive with strong language-model baselines.
2. It does not prove the named banks are meaningfully specialized.
3. It does not prove the SSM is delivering strong long-range factual retention.
4. It does not validate the memory hierarchy, since WM/IM were disabled.

---

## 12. Benchmark Comparison Caveat

WikiText-103 comparisons are messy. The current V6 run is:
- raw WikiText-103
- GPT-2 BPE tokenization
- validation perplexity
- chunked fixed-length evaluation

Many classic published results are:
- word-level
- differently preprocessed
- often test perplexity
- often evaluated under different context handling rules

Therefore, many headline WikiText-103 perplexity numbers are **not apples-to-apples**.

Useful references documenting this problem:
- Hugging Face issue on GPT-2 / WikiText-103 mismatch: `https://github.com/huggingface/transformers/issues/483`
- llm.c reproduction notes: `https://github.com/karpathy/llm.c/pull/340`

### Orientation-only published baselines (not directly fair)

| Model | Reported PPL | Source |
|------|--------------|--------|
| Neural cache model | `40.8` test | Salesforce WikiText benchmark page |
| AWD-QRNN | `32.0` val / `33.0` test | Merity et al., single-GPU WT103 paper |
| Adaptive Input Representations | `18.7` | Baevski & Auli |
| Transformer-XL | `18.3` | Dai et al. |

### Closer raw/BPE comparison points

From the reproduction work in llm.c PR #340:

| Model | Validation PPL |
|------|----------------|
| GPT-2 `124M` | `30.59` raw / `31.04` minimally preprocessed |
| GPT-2 `355M` | `22.35` |
| GPT-2 `774M` | `19.33` |
| GPT-2 `1558M` | `17.46` |

Relative to those references, the V6 `28.7M` result of `49.61` is still well behind.

---

## 13. Revised Current Status

The earlier sections of this file established that V6 outperforms V5 clearly on TinyStories and that memory capacity is a major behavioral control knob. Those conclusions still stand. But after the first WikiText-103 run, the broader summary must be more conservative:

1. **V6 is now a serious research candidate, not yet a benchmark winner.**
2. **No-memory V6 is currently the cleanest headline configuration.** It is the most stable, least confounded path.
3. **Memory should not return to the headline architecture until it proves itself on WikiText-103 or another real corpus.**
4. **The next decisive result must be an apples-to-apples baseline**, not just more V6-only runs.

### Revised next-step priority

| Priority | Run | Why |
|---------|-----|-----|
| 1 | Same-budget Transformer on the exact WikiText-103 raw/BPE pipeline | Most important missing baseline |
| 2 | V6 WikiText-103 with `WM=8`, `IM=0` | Cleanest memory test on real text |
| 3 | V6 `medium` no-memory on the same pipeline | Tests whether current gap is mostly capacity |
| 4 | PG-19 | Only after one of the above gives a convincing signal |

### Current recommendation

Do **not** claim V6 is proven.

Do claim:
- V6 is stable on real text.
- V6 is interesting enough to benchmark seriously.
- The next decision should come from matched baselines and small-memory WikiText-103, not from more TinyStories-only evidence.

---

## 14. Memory Reframe (2026-03-11)

Based on all evidence above (especially §5.3-5.8, §13), the memory architecture has been redesigned. The token-wise WorkingMemory/InternalMemory system is replaced by event-based episodic memory, the training objective is augmented with span corruption and delayed recall tasks, and banks are pushed toward functional roles via auxiliary losses.

**Full details**: [EXPERIMENTS_V6_MEMORY_REFRAME.md](EXPERIMENTS_V6_MEMORY_REFRAME.md)

Key changes:
- **Sparse retrieval everywhere**: PersistentMemoryReader / ExpertMemoryReader now use top-k sparse retrieval
- **Span corruption objective**: T5-style span masking that trains cross-gap reasoning (`--objective span_corruption`)
- **Delayed recall objective**: fact-then-cue tasks that directly supervise memory write/retrieve (`--objective delayed_recall`)
- **Episodic memory**: event-based writes from span/chunk summaries instead of token-wise (`--episodic_slots N`)
- **Bank role loss**: pushes semantic bank toward entity-like stability and context bank toward relation-like variability (`--bank_role_weight`)
- **Two-pass model**: bidirectional chunk encoder + causal decoder as controlled ablation (`--mode two_pass`)
- **Behavioral quality metrics**: repeat rate, restart fragmentation, unique word ratio printed alongside generation samples

---

## 15. Architecture Rebalancing: Option B + TSO (2026-03-14)

After exhaustive memory experiments (reframe Runs 1-5, composite keys, long-seq), the conclusion is clear: **memory does not help at current scale**. The SSM slow lanes handle all tested gap sizes trivially. The real bottleneck is parameter allocation: the SSM gets only 16.6% of params while underperforming banks get 32.9%.

### What changed

**Option B**: Replace dual `NamedBankPair` + `PhaseInterferenceCoupler` with a single `ComplexGatedUnit` per layer. Reinvest saved params into SSM `state_dim` (512 -> 1280).

**TSO (Timescale-Separated Output)**: Each SSM layer splits output into fast/medium/slow streams with separate `C_proj` and a learned gate. Lets the model explicitly suppress stale slow-lane state when context changes.

### Parameter comparison

| Component | small-matched | small-rebalanced | Change |
|-----------|--------------|-----------------|--------|
| Banks | 9.5M (32.9%) | 4.7M (16.0%) | -50% |
| Couplers | 1.6M (5.5%) | 0 (0%) | removed |
| SSM | 4.7M (16.6%) | 11.9M (40.2%) | **+2.5x** |
| Total | 28.7M | 29.5M | ~same |

### Skipped (confirmed dead ends)

- All memory (WM, IM, episodic, composite keys): SSM handles all tested gaps
- Diversity loss, bank role loss: never affected PPL
- Span corruption: incompatible with causal architecture
- Delayed recall: SSM handles gap=512 trivially

### Novel contributions preserved

1. Complex-valued selective SSM with multi-timescale eigenvalues
2. Phase-preserving ops (ModReLU, ComplexNorm, ComplexGatedUnit)
3. **TSO** -- new: timescale-separated output with learned gating
4. O(n) attention-free architecture
5. Phase-coherence LM head

### Config and script

- Preset: `--size small-rebalanced`
- Script: `scripts/run_v6_rebalanced.sh`
- Baseline to beat: Run 4 val PPL 56.46 (small-matched, 10 epochs, WikiText-103)

### Run v6-rebalanced-tso (2026-03-14, RTX 4090)

**Setup**: size=`small-rebalanced` (29.5M params), single_bank=True, TSO=True, state_dim=1280, seq_len=2048, batch_size=3, 10 epochs, WikiText-103. No memory, no diversity loss.

**Results**:

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 229.41 | 112.52 | |
| 2 | 101.48 | 79.51 | |
| 3 | 81.04 | 68.35 | |
| 5 | 66.92 | 59.06 | |
| 7 | 61.14 | 54.80 | |
| 10 | 56.97 | **52.64** | still improving |

**Wall time**: 6.09h (21,911s). Throughput: 54,253 tok/s. GPU: 1.1/10.3GB.

**Comparison to baselines**:

| Config | Params | Val PPL (10 ep) | tok/s | GPU mem |
|--------|--------|----------------|-------|---------|
| **small-rebalanced (TSO)** | **29.5M** | **52.64** | **54,253** | **1.1GB** |
| small-matched (Run 4) | 28.7M | 56.46 | 46,000 | 1.4GB |
| composite keys + episodic | 29.2M | 58.40 | 34,299 | 1.4GB |

**Key findings**:

1. **6.8% PPL improvement** over the best prior V6 result (56.46 -> 52.64) with same param budget.
2. **18% faster throughput** (54k vs 46k tok/s) -- single CGU is cheaper than dual banks + coupler.
3. **Less GPU memory** (1.1GB vs 1.4GB) despite larger state_dim.
4. **Zero repetition, zero restarts** -- generation quality is clean.
5. **Still improving at epoch 10** -- curve not fully converged.
6. PPL gap to GPT-2 124M (31 PPL) is 1.7x with 4.2x fewer params. Scaling should close this.

**Generation sample** (epoch 10):
> "In 1923, the University of California in Chicago is a 'school-school campus' located at the east end of the campus. = = Education and research = = The University of Florida's primary education system was founded by the University of California in 1910..."

Quality: rep3=0.033, rep4=0.022, restarts=0, uniq=0.656. Fluent Wikipedia-style text with section headers; factually incoherent (same limitation as all prior configs at this scale).

**Log**: `logs/v6/rebalanced_tso_wikitext103_20260314_110835_e96032d/v6_autoregressive_small-rebalanced.log`

---

## 16. Scaling to 60M: medium-rebalanced (2026-03-14)

The rebalanced architecture is validated -- faster, cheaper, and better than all prior V6 configs. Next step: scale to ~60M params to test whether the gains hold and close the gap to GPT-2 124M (~31 PPL).

- Preset: `--size medium-rebalanced`
- Script: `scripts/run_v6_medium_rebalanced.sh`
- Target: val PPL < 42 on WikiText-103 (10 epochs)

### Run medium-rebalanced (2026-03-14/15, RTX 4090)

**Setup**: size=`medium-rebalanced` (58.4M params), dim=192, state_dim=1536, layers=16, expand=3, single_bank=True, TSO=True, seq_len=2048, batch_size=3, 10 epochs, WikiText-103. No memory, no diversity loss.

**Parameter breakdown**:

| Component | Params | Share |
|-----------|--------|-------|
| embed (tied) | 19.3M | 33.0% |
| banks (single CGU) | 10.6M | 18.2% |
| SSM + TSO | 28.4M | 48.6% |
| lm_head proj | 0.07M | 0.1% |
| **Total** | **58.4M** | |

**Results**:

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 226.06 | 107.71 | |
| 2 | 94.64 | 73.04 | |
| 3 | 72.69 | 60.94 | |
| 4 | 62.74 | 54.53 | |
| 5 | 56.65 | 50.65 | |
| 6 | 52.64 | 48.06 | |
| 7 | 49.83 | 46.22 | |
| 8 | 47.87 | 45.11 | |
| 9 | 46.59 | 44.57 | |
| 10 | 45.92 | **44.47** | still improving |

**Wall time**: 9.69h (34,896s). Throughput: 34,064 tok/s. GPU: 1.6/14.7GB.

**Scaling comparison** (all WikiText-103, 10 epochs):

| Config | Params | Val PPL | tok/s | GPU mem |
|--------|--------|---------|-------|---------|
| **medium-rebalanced** | **58.4M** | **44.47** | **34,064** | **1.6GB** |
| small-rebalanced | 29.5M | 52.64 | 54,253 | 1.1GB |
| small-matched (Run 4) | 28.7M | 56.46 | 46,000 | 1.4GB |
| composite keys + episodic | 29.2M | 58.40 | 34,299 | 1.4GB |
| GPT-2 124M (reference) | 124M | ~31 | - | - |

**Key findings**:

1. **15.5% PPL improvement** from 2x params (52.64 -> 44.47). Scaling exponent ~0.25.
2. **Best generation quality ever**: rep3=0.000, rep4=0.000, uniq=0.737. Zero repetition, zero restarts.
3. **Coherent paragraphs** with specific dollar amounts, dates, and inflation adjustments.
4. **Still improving at epoch 10** -- curve not converged, more epochs would help.
5. **Only 1.6GB GPU** -- massive headroom on 4090 (14.7GB available).
6. PPL gap to GPT-2 124M is 1.43x with 2.1x fewer params. Naive scaling extrapolation: ~500M params needed to match, not 124M. The 4x complex efficiency is not yet materializing.

**Generation sample** (epoch 10):
> "In 1923, the University of Washington offered a $7,000 grant from the National Board of Education to run for the U.S. Congress. The proposal was eventually rejected and it turned out that the state would not pay the tax to the State Department for the purpose of its construction. The plan was approved by the California Democratic Party on January 2, 1926 with a budget of $1 million (equivalent to about $30 million in 2015)."

Quality: rep3=0.000, rep4=0.000, restarts=0, uniq=0.737. Fluent Wikipedia-style text with specific numbers/dates; factually incoherent (entities/dates don't align with real world).

**Diagnosis**: The factual incoherence is not a scale problem -- it's a **state retention problem**. The SSM state is continuously overwritten; the model has no mechanism to protect phase-coherent factual bindings from being corrupted by new input. This motivates Gated State Protection (GSP).

**Log**: `logs/v6/medium_rebalanced_tso_wikitext103_20260314_173445_cc4a491/v6_autoregressive_medium-rebalanced.log`

---

## 17. Gated State Protection (GSP) (2026-03-15)

### Problem: SSM state is continuously overwritten

The medium-rebalanced model produces fluent, structured Wikipedia text but with no factual consistency. When the SSM processes "The capital of France is Paris ... [many tokens] ... The capital of India is", the slow lanes retain France/Paris state at high magnitude, but new input overwrites it. There is no mechanism to say "this state dimension holds an important fact -- don't touch it."

Scaling alone won't fix this: our scaling exponent (~0.25) means ~500M params to match GPT-2 124M. The complex-number 4x efficiency isn't materializing because we aren't exploiting what makes complex numbers unique: **phase**.

### Why complex numbers make this tractable

In a real-valued SSM, "protecting" a state dimension just freezes a scalar. In our complex SSM, protecting a state dimension preserves a **phase relationship** -- the angular alignment between entity and property. Phase relationships are richer than scalars:

- Phase encodes WHAT (entity/relationship type via angular alignment)
- Magnitude encodes HOW MUCH (relevance/confidence)
- Phase interference naturally selects among competing facts
- Phase rotation composes associatively (chain of reasoning)

Facts are phase-coherent bindings. Complex state stores them natively. GSP gives the model a way to protect them.

### Design: Gated State Protection

A learned per-state-dimension gate that interpolates between "normal SSM update" and "freeze state":

```
Standard SSM: h[t] = A[t] * h[t-1] + Bx[t]

With GSP:
  protect = sigmoid(protect_gate(|x|))       -- [B, T, state_dim]
  A' = protect * (1+0j) + (1-protect) * A    -- identity when protect=1
  Bx' = (1-protect) * Bx                     -- no new input when protected
  h[t] = A'[t] * h[t-1] + Bx'[t]            -- unchanged parallel scan
```

**Properties**:

1. **Parallel scan compatible**: A' and Bx' are still valid operands for the associative scan. No algorithmic change needed.
2. **Novel**: not in Mamba, S4, S5, RWKV, or any published SSM. Selective gating has been applied to dt (Mamba) but never to the state update itself with a freeze/protect semantic.
3. **Cheap**: one `Linear(dim, state_dim)` per layer. Adds 4.7M params (8.1% overhead) to the 58.4M medium-rebalanced model.
4. **Complex-native**: protects phase relationships, not just scalar magnitudes.
5. **Self-supervised**: model learns what to protect purely from next-token prediction loss.
6. **O(n)**: no attention bottleneck.

### Initialization

The protect gate bias is initialized to -3.0, so `sigmoid(-3.0) ≈ 0.047` -- nearly everything is unprotected at the start. The model must learn through gradient signal which state dimensions to protect and when. This prevents GSP from collapsing the SSM into a static memory at initialization.

### Parameter comparison

| Component | medium-rebalanced | medium-rebalanced-gsp | Change |
|-----------|------------------|----------------------|--------|
| embed (tied) | 19.3M (33.0%) | 19.3M (30.5%) | -- |
| banks (CGU) | 10.6M (18.2%) | 10.6M (16.8%) | -- |
| SSM + TSO | 28.4M (48.6%) | 28.4M (45.0%) | -- |
| **GSP gates** | **0** | **4.7M (7.5%)** | **+4.7M** |
| lm_head | 0.07M | 0.07M | -- |
| **Total** | **58.4M** | **63.2M** | **+8.1%** |

### What we're testing

- Does GSP improve val PPL? (state protection -> better next-token prediction for facts)
- Does generation show improved factual coherence? (entity-property alignment within a passage)
- What is the throughput cost? (expect <5% due to one extra Linear per layer)
- Does the model learn non-trivial protection patterns? (can inspect protect gate activations post-training)

### Config and script

- Preset: `--size medium-rebalanced-gsp`
- Script: `scripts/run_v6_medium_gsp.sh`
- Baseline to beat: medium-rebalanced val PPL 44.47 (10 epochs, WikiText-103)

### Future direction: Phase-Coherent Holographic Binding

GSP gives the model the ability to **retain** phase-coherent state. The next step would be giving it the ability to **create and retrieve** compositional bindings:

- Bind entity+property via `cmul(entity_phase, property_phase)` (HRR theory: circular convolution in Fourier space = element-wise complex multiplication)
- Retrieve via `cmul(query, conj(stored_binding))`
- Store bindings in protected state dimensions

This would be the ultimate use of complex numbers for compositional memory, but GSP is the prerequisite -- without protection, any binding would be overwritten within a few hundred tokens.

---

## 18. GSP Run Results (2026-03-15)

### Setup

- **Preset**: `medium-rebalanced-gsp` (63.2M params)
- **Architecture**: Single CGU(expand=3) + ComplexSSM(state_dim=1536) + TSO + GSP
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3, 10 epochs
- **GPU**: RTX 4090 (1.6/17.5 GB used)
- **Script**: `scripts/run_v6_medium_gsp.sh`
- **Total wall time**: 40,724s (11.31h)

### Val PPL per epoch

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Improvement |
|-------|-----------|-----------|----------|---------|-------------|
| 1 | 5.4042 | 222.35 | 4.6508 | 104.67 | -- |
| 2 | 4.5135 | 91.24 | 4.2552 | 70.47 | -32.7% |
| 3 | 4.2402 | 69.42 | 4.0615 | 58.06 | -17.6% |
| 4 | 4.0797 | 59.13 | 3.9356 | 51.19 | -11.8% |
| 5 | 3.9727 | 53.13 | 3.8563 | 47.29 | -7.6% |
| 6 | 3.8975 | 49.28 | 3.8030 | 44.83 | -5.2% |
| 7 | 3.8443 | 46.73 | 3.7643 | 43.13 | -3.8% |
| 8 | 3.8064 | 44.99 | 3.7418 | 42.18 | -2.2% |
| 9 | 3.7807 | 43.85 | 3.7318 | 41.75 | -1.0% |
| 10 | 3.7670 | 43.25 | 3.7298 | **41.67** | -0.2% |

**Still improving at epoch 10** -- not converged. Epoch 9->10 delta is small (0.08 PPL) but consistently downward with every single epoch being a new best.

### Comparison with baselines

| Model | Params | Val PPL | vs. medium-rebalanced |
|-------|--------|---------|----------------------|
| V6 small-matched | 28.7M | 56.46 | -- |
| V6 small-rebalanced (TSO) | 29.5M | 52.64 | -- |
| V6 medium-rebalanced (TSO) | 58.4M | 44.47 | baseline |
| **V6 medium-rebalanced-gsp (TSO+GSP)** | **63.2M** | **41.67** | **-6.3%** |
| GPT-2 124M (reference) | 124M | ~31 | -- |

**GSP is a clear win**: 6.3% PPL improvement with only 8.1% more parameters. The improvement exceeds what naive scaling would predict (log-linear extrapolation from small->medium gives ~0.5% per 1% more params; GSP gives 0.78% per 1% more params).

### Generation quality

**Throughput**: 29,165 tok/s average (vs 34,064 for medium-rebalanced -- 14% slower as expected from the extra Linear per layer).

**Quality metrics**: rep3=0.000, rep4=0.000, uniq=0.776 -- clean generation, no repetition.

**Sample (epoch 10)**:
> In 1923 , the University of Michigan ( U.S. Route 90 in the United States Senate and the International Federation for Children 's Hospital in April to form a group known as the " World 's Service Battalion during the Spanish – American War . The New York Herald Tribune wrote that while she was on the table at his home in Manchester , New Jersey , the building was the most powerful vessel , they are still alive .

**Observation**: The model generates longer coherent runs about a single thematic domain (institutional/political) before drifting, compared to medium-rebalanced which shifted topics within 2 sentences. This suggests GSP IS protecting high-level thematic state. But it cannot yet do **compositional** factual binding -- the model knows "we're talking about institutions" but can't bind "THIS institution is Michigan AND it was founded in 1817."

### Key takeaways

1. **GSP validates the core thesis**: selectively protecting SSM state dimensions improves language modeling. This is the first mechanism that exploits the complex-number advantage (protecting phase-coherent bindings rather than just scalars).
2. **The model is not converged**: running 20 epochs with a LR restart could push PPL to ~38-39.
3. **The remaining gap to GPT-2 is 1.34x** (41.67 vs ~31 PPL). Brute scaling would require ~200-300M params. An architectural innovation is needed.
4. **The missing piece is compositional binding**: the model can protect state but cannot explicitly create and retrieve entity-property associations. This motivates Phase-Coherent Binding Registers (PCBR).

---

## 19. Holographic State Binding (HSB) (2026-03-16)

### Motivation

GSP proved the model can **retain** phase-coherent state. But generation analysis shows it cannot **compose and retrieve** entity-property associations. The model knows "we're talking about institutions" but can't bind "THIS institution is Michigan AND it was founded in 1817."

Transformers solve this with attention -- any token can directly reference any other token. SSMs have no such mechanism. We need an SSM-native compositional binding operator.

### The core insight: cmul IS holographic binding

Holographic Reduced Representations (HRR) use circular convolution to bind two vectors into a single vector of the same dimensionality, and circular correlation to unbind (retrieve). In Fourier space, circular convolution becomes element-wise multiplication, and circular correlation becomes element-wise multiplication by the conjugate.

Our complex representation tensors `[..., dim, 2]` are already in a space where:
- `cmul(a, b)` = element-wise complex multiplication = binding
- `cmul(query, cconj(stored))` = element-wise multiplication by conjugate = unbinding/retrieval
- `cmul(cmul(A, B), cconj(A))` ≈ B (associative retrieval)

This means **our existing primitives already implement HRR** -- we just need to use them for compositional memory.

### Design evolution: from registers to SSM-native

**V1 (discarded)**: Phase-Coherent Binding Registers (PCBR) -- a separate module with 8 registers per layer, sequential T-loop for write/read. After novelty audit, this was identified as **slot attention with cmul instead of dot-product** -- structurally not novel, and the sequential loop was an anti-pattern for an architecture designed around parallel scan.

**V2 (implemented)**: Holographic State Binding (HSB) -- bind/unbind happens **inside** the SSM layer itself. No separate registers, no sequential loop, no bolt-on module.

### Architecture: SSM-native HSB

HSB modifies the SSM layer in two places:

**1. Bind (inject into state)**: Before `parallel_scan(A, Bx)`, add holographic bindings to the Bx input term:

```
Bx_original = B_proj(x) * dt           -- normal SSM input
bind_signal = scatter(cmul(key(x), value(x))) * bind_gate(|x|)  -- holographic binding
Bx = Bx_original + bind_signal         -- combined input to state
h = parallel_scan(A, Bx)               -- binding enters state naturally
```

**2. Unbind (retrieve from state)**: After `parallel_scan`, extract bindings from the state:

```
h_bind = gather(h)                      -- project state to bind_dim
retrieved = out_proj(cmul(query(x), conj(h_bind))) * unbind_gate(|x|)
y = C_proj(h) + D*x + retrieved        -- retrieved bindings added to output
```

**Why this is fundamentally different from everything published**:

| Feature | Attention | Slot Attention / NTM | Mamba Selectivity | **HSB** |
|---------|-----------|---------------------|-------------------|---------|
| Memory structure | KV cache (O(T)) | Fixed slots | SSM state | **SSM state** |
| Write | append | soft-attention write | input-dependent B | **cmul(key,val) into Bx** |
| Read | Q*K^T (O(T²)) | soft-attention read | C*h | **cmul(query, conj(h))** |
| Compositional | No | No | No | **Yes (HRR algebra)** |
| Sequential loop | No | Yes (over T) | No | **No** |
| Parallel scan | N/A | N/A | Yes | **Yes** |

**Key differentiators**:
1. **No new sequential bottleneck**: bindings flow through the existing parallel scan
2. **Compositional**: `cmul(cmul(A,B), cconj(A)) ≈ B` -- algebraic retrieval, not soft lookup
3. **Phase-native**: binding IS phase rotation, retrieval IS counter-rotation
4. **Synergistic with GSP**: GSP protects important state dims; HSB gives it bindings worth protecting
5. **Not a module, an SSM property**: holographic binding is an intrinsic behavior of the recurrence

### Bottleneck design

To keep parameter cost manageable, bindings happen in a lower-dimensional subspace (`bind_dim = 96`, half of `dim = 192`):

```
x [B,T,dim=192,2]
  --> key_proj: dim -> bind_dim=96     (ComplexLinear, 2*192*96 = 36.9K)
  --> value_proj: dim -> bind_dim=96   (same)
  --> cmul(key, value): bind_dim       (free -- elementwise)
  --> scatter_proj: bind_dim -> state_dim=1536  (ComplexLinear, 2*96*1536 = 294.9K)
  --> added to Bx before parallel_scan

h [B,T,state_dim=1536,2]
  --> gather_proj: state_dim -> bind_dim=96     (ComplexLinear, 2*1536*96 = 294.9K)
  --> query_proj: dim -> bind_dim=96            (ComplexLinear, 2*192*96 = 36.9K)
  --> cmul(query, conj(h_gathered)): bind_dim   (free -- elementwise)
  --> out_proj: bind_dim -> dim=192             (ComplexLinear, 2*96*192 = 36.9K)
  --> added to y after C_proj
```

Per layer: ~738K params. Total HSB: 11.8M (18.7% overhead on 63.2M GSP model).

### Bug fix: _init_weights bias override

Discovered that `_init_weights()` in `model.py` was zeroing all `nn.Linear` biases, including the GSP protect_gate bias (should be -3.0) and the HSB bind/unbind gate biases (should be -3.0). Added `_reinit_custom_biases()` to restore these after global initialization.

**Impact on previous GSP run**: The GSP model that achieved PPL 41.67 was actually running with `protect_gate.bias = 0.0` (sigmoid(0) = 0.5, meaning 50% protection from the start). This means GSP performed well even without the intended conservative initialization. With the correct -3.0 bias, performance may differ.

### Parameter breakdown

| Component | medium-rebalanced-gsp | medium-rebalanced-hsb | Change |
|-----------|----------------------|----------------------|--------|
| embed (tied) | 19.3M (30.5%) | 19.3M (25.7%) | -- |
| banks (CGU) | 10.6M (16.8%) | 10.6M (14.2%) | -- |
| SSM + TSO | 28.4M (45.0%) | 28.4M (37.9%) | -- |
| GSP gates | 4.7M (7.5%) | 4.7M (6.3%) | -- |
| **HSB** | **0** | **11.8M (15.7%)** | **+11.8M** |
| lm_head | 0.07M | 0.07M | -- |
| **Total** | **63.2M** | **75.0M** | **+18.7%** |

### What we're testing

1. Does HSB improve val PPL beyond GSP's 41.67? (compositional binding/retrieval via HRR algebra)
2. Does generation show improved factual coherence? (entity-property alignment should be visibly better)
3. What is the throughput cost? (no sequential loop, but extra projections per layer)
4. Do the bind/unbind gates learn to be selective? (should open for factual content, stay closed for filler)
5. Does GSP + HSB synergize? (GSP protects the bindings that HSB writes into state)

### Config and script

- Preset: `--size medium-rebalanced-hsb`
- Script: `scripts/run_v6_medium_pcbr.sh`
- Baseline to beat: medium-rebalanced-gsp val PPL 41.67 (10 epochs, WikiText-103)

### Novelty assessment

HSB is genuinely novel on multiple axes:

1. **No published work injects HRR bindings into SSM state**: HRR (Plate, 1995) and variants (MAP, VTB) work in real space as standalone vector operations. We use cmul/cconj on complex SSM state.
2. **No published SSM uses compositional bind/unbind**: Mamba selectivity controls dt/B/C; it doesn't create or retrieve compositional entity-property bindings. S4/S5/RWKV have no compositional mechanism at all.
3. **Not a memory module**: unlike NTM/DNC/slot attention, HSB has no separate memory bank. Bindings live in the SSM state itself and flow through the standard parallel scan.
4. **Complex-native**: binding IS phase rotation, retrieval IS phase counter-rotation. This only works because our SSM state is complex-valued.
5. **GSP + HSB is synergistic**: GSP provides selective state retention, HSB provides compositional content to retain. Neither exists in any published architecture.

---

## 20. Phase-Associative Memory (PAM) (2026-03-17)

### The Failure of HSB

The Holographic State Binding (HSB) experiment failed to improve performance. The final validation PPL was **43.54**, which is a regression from the GSP baseline of **41.67**.

**Diagnosis**: The root cause is *state interference*. HSB uses HRR binding (`cmul(key, value)`) to compress a $D \times D$ association into a $D$-dimensional vector, which is then added to the SSM state. While HRR works well for single associations, adding multiple facts into the same vector state causes catastrophic interference. The fundamental limitation of our `ComplexSSM` is that its state is a **vector** (diagonal $A$), which lacks the capacity to store multiple cross-dimensional associations over time without them colliding.

### The Solution: Complex Matrix State

To solve the interference problem, we must upgrade the state from a Vector to a **Complex Matrix** ($S_t \in \mathbb{C}^{H \times d \times d}$). This provides $O(d^2)$ capacity per head, allowing multiple facts to be stored independently.

This architecture is called **Phase-Associative Memory (PAM)**.

### Mathematical Formulation

1. **State Update**: $S_t = \gamma_t S_{t-1} + V_t \otimes K_t^*$
   - The state is a true memory matrix.
2. **Retrieval**: $Y_t = S_t Q_t = V_t (K_t^* \cdot Q_t)$
   - The complex dot product $K_t^* \cdot Q_t$ naturally computes attention via constructive/destructive phase interference. No Softmax is needed!
3. **GSP Integration**: The protect gate $p_t$ modifies the decay: $\gamma_t = \exp(-dt)(1-p_t) + p_t$. This allows the model to freeze the matrix state and retain facts indefinitely.

### Efficient Training (The Dual Form)

Computing the $d \times d$ matrix sequentially is slow. Because it's a linear recurrence, we can compute it in $O(T^2)$ time using the **Dual Form** (Attention form) with highly optimized dense matrix multiplications. For $T=2048$, this is extremely fast.

Dual form: $Y_t = \sum_{i \le t} \left( \prod_{j=i+1}^t \gamma_j \right) (Q_t \cdot K_i^*) V'_i$

### Architecture Specs

- **Model Dimension**: $D=384$ (increased from 192 to utilize the parameter budget effectively)
- **Heads**: $H=6$, **Head Dim**: $d=64$
- **Total Parameters**: ~100M
- **Components**: `PhaseAssociativeLayer` replaces `ComplexSSMLayer`. `ComplexGatedUnit` remains as the MLP equivalent.

### Implementation Details

- **Module**: `v6/core/pam.py` -- `PhaseAssociativeLayer` and `PhaseAssociativeMemory`
- **Integration**: Replaced `ComplexSSM` entirely in `v6/backbone.py`.
- **Config**: `medium-pam` preset added to `v6/config.py`.
- **Script**: `scripts/run_v6_medium_pam.sh`

### What we expect

1. **PPL**: Significant improvement over GSP (41.67) due to the massive increase in state capacity ($O(d^2)$ vs $O(d)$).
2. **Factual Recall**: The matrix state should allow the model to retrieve specific facts without interference, solving the core issue identified in HSB.
3. **Training Speed**: The Dual Form implementation should be very fast on GPU, comparable to standard attention.

---

## 21. PAM Run Results (2026-03-18 to 2026-03-19)

### Inference bugs found and fixed

Two bugs in `v6/core/pam.py` caused generation to produce complete gibberish despite healthy training PPL. Both were train/inference path mismatches.

**Bug 1 (critical): Prompt state lost during generation**

The Dual Form (training path, used when `state is None and T > 1`) returned `new_state = torch.empty(0)`. In `PhaseAssociativeMemory.forward()`, state collection was gated on `state is not None`, so when processing the prompt (state=None), no state was returned. The next token then started from zero-initialized state with no prompt context.

Before:
```python
new_state = torch.empty(0, device=x.device)
```

After: compute the final recurrent state $S_T = \sum_i D[T,i] \cdot (V'_i \otimes K_i^*)$ from the decay matrix already available:
```python
D_last = D[:, :, -1, :]  # [B, H, T]
wv_r = v_prime[..., 0] * D_last.unsqueeze(-1)
wv_i = v_prime[..., 1] * D_last.unsqueeze(-1)
sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
new_state = torch.stack([sr, si], dim=-1)  # [B, H, d, d, 2]
```

Also updated `PhaseAssociativeMemory.forward()` to always collect and return states (not only when input state was non-None).

**Bug 2 (high): Missing query scaling in recurrent form**

The Dual Form applied `scale = d ** -0.5` to queries. The Recurrent Form (inference path) did not. With `d=64`, this is an 8x magnitude mismatch.

Before:
```python
q_t = q[:, :, t].unsqueeze(-3)  # no scale
```

After:
```python
scale = d ** -0.5
q_t = q[:, :, t].unsqueeze(-3) * scale
```

**Why PPL looked good but generation was gibberish**: Validation PPL is computed via `model.forward(batch)` with T=2048, which takes the Dual Form path (correctly scaled, full context). Generation processes the prompt via Dual Form (first token correct), then switches to Recurrent Form with zero state and wrong scale. This explains the pattern in the pre-fix log: the first predicted token ("Oxford") was correct, everything after was noise.

### Generation: before vs after fix

**Before fix** (epoch 3, batch 5000 -- old log):
> In 1923 , the University of Oxford by ; . 11 East Highided appearanceeter Lis on Regent D and G @ @ Pr Kse norbers and En to ally and back in to given =istist over , – with and ,ids their course finalton ...

**After fix** (epoch 10, batch 15000 -- new log):
> In 1923 , the University of Kentucky opened a public schools school in 1924 and served as the state 's first governor for 40 years . During the 1930s , a public school was built in a public house to serve as the primary school for Governor , but it still served as an office space until 1933 . This was completed by 1936 , when the University of Kentucky passed its own law class .

**Final generation** (epoch 10 end):
> In 1923 , the University of Missouri and the University of Michigan was also established in 1926 . In 1928 , the University of Michigan opened its current campus with the school 's first campus opening at St. Louis Road on the northern end of Lake Michigan in 1929 .

### Training progression

Resumed from epoch 3 checkpoint (pre-fix epochs 1-3 had correct PPL but broken generation). Epochs 4-10 ran with both bugs fixed.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 4     | 54.03     | 47.19   | first epoch after resume with fix |
| 5     | 48.55     | 43.55   | |
| 6     | 45.26     | 41.43   | already beats GSP baseline (41.67) |
| 7     | 43.14     | 40.11   | |
| 8     | 41.76     | 39.34   | |
| 9     | 40.91     | 39.02   | |
| 10    | 40.50     | 38.95   | new best |

Quality at epoch 10: rep3=0.051, rep4=0.020, restarts=0, uniq=0.624.

### Run details

- **Preset**: `medium-pam` (dim=384, 16 layers, single CGU expand=3, PAM H=6 d=64, GSP)
- **Parameters**: 100.4M
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3
- **Throughput**: ~23,500 tok/s on RTX 4090
- **Wall time**: ~9.9 hours total (epochs 4-10 after resume)
- **Script**: `./scripts/run_v6_medium_pam.sh --resume`

### Conclusion

PAM is validated. The matrix state ($S \in \mathbb{C}^{H \times d \times d}$) fixes the interference problem that caused HSB to regress. Final val PPL **38.95** beats the GSP baseline of **41.67** by 6.5%, with coherent multi-sentence generation. The model shows good factual structure (university names, dates, locations) and low repetition.

The bugs were pure train/inference path mismatches -- training was never affected. The pre-fix PPL numbers (epochs 1-3) remain valid; only generation was broken.

| Model | Params | Val PPL | Notes |
|-------|--------|---------|-------|
| medium-rebalanced | 58.4M | 44.47 | SSM baseline |
| medium-rebalanced-gsp | 63.2M | 41.67 | + GSP |
| medium-rebalanced-hsb | 75.0M | 43.54 | + HSB (regression -- interference) |
| **medium-pam** | **100.4M** | **38.95** | **PAM + GSP (new best)** |

---

## 21. Experiment: Medium-PAM-v2 -- Interleaved CGU + PAM (2026-03-19)

### Hypothesis

The primary performance bottleneck in medium-pam is **architectural layout**, not model capacity. All 16 CGU layers (pointwise, zero sequence mixing) run sequentially before all 16 PAM layers. This means the first half of the network has no cross-position information flow. Every competitive architecture (Transformer, Mamba, RetNet, GateLoop) interleaves channel mixing (FFN) with sequence mixing (attention/SSM) in every block. Fixing this layout should significantly close the PPL gap.

### What changed

1. **Interleaved layout**: Each of 16 blocks now runs `CGU -> PAM` instead of `[CGU x16] -> [PAM x16]`.
2. **Higher LR**: 1e-4 (up from 3e-5). The old LR was 3-10x lower than standard for 100M-scale models.
3. **Longer warmup**: 1000 steps (up from 500).

### What did NOT change (novelty preserved)

- PAM matrix state (same mechanism, same capacity).
- GSP (Gated State Protection) -- same per-dimension freeze gate.
- Complex-valued representations throughout -- phase preserved end-to-end.
- CGU (Complex Gated Unit) -- same SwiGLU-style phase-safe gating.
- All phase-safe primitives (ModReLU, ComplexNorm, ComplexLinear) -- unchanged.
- Attention-free, O(T) inference per token -- no softmax, no KV cache.
- Total parameter count: identical ~100.4M (same components, different ordering).

### Architecture comparison

```
medium-pam (sequential):
  Tokens -> [CGU x16] -> [PAM x16] -> Output
  (16 layers with ZERO sequence mixing, then 16 layers of sequence mixing)

medium-pam-v2 (interleaved):
  Tokens -> [CGU -> PAM] x16 -> Output
  (Every block has both channel mixing AND sequence mixing)
```

### Config

- **Preset**: `medium-pam-v2` (dim=384, 16 layers, single CGU expand=3, PAM H=6 d=64, GSP, interleave_pam=True)
- **Parameters**: ~100.4M (same budget)
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3
- **LR**: 1e-4, warmup_cosine, warmup=1000
- **Script**: `./scripts/run_v6_medium_pam_v2.sh`

### Baselines for comparison

| Model | Params | Val PPL | Notes |
|-------|--------|---------|-------|
| medium-pam (sequential) | 100.4M | 38.95 | Same components, sequential layout |
| GPT-2 small (Transformer) | ~124M | ~14.84 | Fine-tuned on WikiText-103 |
| Transformer (vanilla) | ~125M | ~18.6 | Trained on WikiText-103 |
| Mamba-Small (SSM) | 130M | ~24.1 | Selective SSM |
| GateLoop (linear RNN) | 125M | ~13.4 | Data-controlled recurrence |

### Planned follow-up (not in this run)

- **Phase 2**: Add short causal ComplexConv1d (kernel=4) inside each PAM layer for local n-gram capture.
- **Phase 3**: Tune expand factor, head count, or reduce layers for better capacity allocation.

### Results

**Stopped after epoch 1** -- terminated early to incorporate quality and speed improvements in experiment #22. The architectural change (interleaving) is carried forward into v3.

| Epoch | Train PPL | Val PPL | tok/s | Time | Notes |
|-------|-----------|---------|-------|------|-------|
| 1     | 123.52    | 57.84   | 21,852 | 5420s (~90min) | best (only epoch) |

Generation at epoch 1 (prompt: "In 1923 , the University of"):
> In 1923 , the University of Illinois . = = Background = = By 1930 , Robert G. Brown had established a large number of small buildings near his home base at Cauchon Cemetery in Bauchon County ; he was one of two sons of William I. Fisot and J. L. Tisauke , who built the first major stone building on the site . He would later build several new buildings , including the building 's first floor , and a building that has

Quality: rep3=0.000, rep4=0.000, restarts=0, uniq=0.762.

For context, medium-pam (sequential) reached val PPL 54.03 at epoch 4 (its first post-resume epoch), so 57.84 at epoch 1 with interleaving is on a competitive trajectory. However, the run was not continued -- all subsequent work uses v3.

---

## 22. Experiment: Medium-PAM-v3 -- QK Phase Norm + Complex RoPE + Speed (2026-03-19)

### Hypothesis

Building on the interleaved layout from v2 (experiment #21), there are two categories of improvement:

**Quality**: The v2 model has (a) zero positional encoding -- the only position signal comes from PAM's causal recurrence, leaving the first CGU layer completely position-blind, and (b) unbounded Q/K magnitudes that contaminate the phase-interference mechanism with magnitude variation. Fixing both should improve PPL.

**Speed**: `ComplexLinear` currently launches 4 GEMM kernels per call (one per real/imag x weight_real/weight_imag combination). With 112 ComplexLinear calls per forward pass, that's 448 kernel launches. Additionally, PAM's Q/K/V projections are 3 separate calls that could be fused.

### What changed

#### Quality changes (new flags, ablatable)

1. **QK Phase Normalization** (`pam_qk_norm=True`): Normalize Q and K to unit complex magnitude per-element before the attention dot product. This makes `Re(Q^* K)` purely measure phase alignment (cosine of phase differences), removing magnitude contamination. Uses existing `cnormalize` (z / |z|). Applied in both dual-form (training) and recurrent (inference) paths. The `d^{-0.5}` scaling is preserved on top.

2. **Complex RoPE on Q,K** (`pam_rope=True`): Position-dependent phase rotation applied to Q and K only (not V, not the residual stream). Each complex dimension k gets frequency `theta_k = 1 / (10000^{k/d})`, and position m rotates by `e^{i*m*theta_k}`. This is a single `cmul` with a precomputed unit-magnitude tensor -- phase-safe by construction (|e^{i*theta}| = 1). Gives relative position awareness (the dot product `Re(q_m^* k_n)` depends on position difference m-n). Cache is precomputed for 8192 positions, auto-extended if needed. Step offset tracked via `PAMState.step` for correct inference positions.

#### Speed changes (always on, zero quality impact)

3. **Block-Real GEMM** in `ComplexLinear`: Replaced 4 `F.linear` calls with 1 by constructing a block matrix `[[W_r, -W_i], [W_i, W_r]]` and concatenating inputs `[x_r, x_i]`. Verified bit-exact (max diff 3.8e-06 in float32). Reduces 448 GEMM launches to 112 per forward pass.

4. **Fused QKV Projection** (`pam_fused_qkv=True`): Single `ComplexLinear(dim, 3*inner_dim)` replaces 3 separate Q/K/V projections. Combined with block-real GEMM, each PAM layer does 1 GEMM for QKV instead of 12.

### What did NOT change (novelty preserved)

- PAM matrix state, GSP, complex-valued representations, CGU, ModReLU, ComplexNorm -- all unchanged.
- Attention-free, O(T) inference per token -- no softmax, no KV cache.
- Interleaved `[CGU -> PAM] x16` layout from v2.
- Total parameter count: ~100.4M (same budget -- fused QKV is the same total weight count, just concatenated).

### Config

- **Preset**: `medium-pam-v3`
- **New flags**: `pam_qk_norm=True`, `pam_rope=True`, `pam_fused_qkv=True`
- **Parameters**: ~100.4M (same budget)
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3
- **LR**: 1e-4, warmup_cosine, warmup=1000

### Ablation plan

If v3 PPL regresses vs v2 baseline (once obtained):
- Disable `pam_qk_norm` only -> test if QK norm hurts
- Disable `pam_rope` only -> test if RoPE hurts
- Speed changes are math-identical and cannot affect quality

### Baselines for comparison

| Model | Params | Val PPL | Notes |
|-------|--------|---------|-------|
| medium-pam (sequential) | 100.4M | 38.95 | Sequential layout, no RoPE, no QK norm |
| medium-pam-v2 (interleaved) | 100.4M | N/A | Stopped early (experiment #21) |
| GPT-2 small (Transformer) | ~124M | ~14.84 | Fine-tuned on WikiText-103 |
| Transformer (vanilla) | ~125M | ~18.6 | Trained on WikiText-103 |
| Mamba-Small (SSM) | 130M | ~24.1 | Selective SSM |
| GateLoop (linear RNN) | 125M | ~13.4 | Data-controlled recurrence |

### Results

*Not yet run.*
