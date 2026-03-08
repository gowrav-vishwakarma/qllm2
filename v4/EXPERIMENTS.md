# V4 Experiment Log

Historical record of V4 (Phase-Field Language Model) training runs, architecture decisions, and independent evaluations. Reconstructed from the Reddit post (March 1, 2026) and community feedback.

---

## 1. Architecture Overview (V4)

V4 was the first version of the phase-based language model. Key components:

| Component | Description |
|-----------|-------------|
| `ComplexEmbed` | Tokens as `[real, imag]` vectors |
| `SemanticPhaseBank` | 512 learnable concept vectors, phase-coherence retrieval, O(n × concepts) |
| `ContextPhaseBank` | Causal windowed average (window=8), complex-multiplied with current token |
| `InterferenceCoupler` | Combines bank outputs via learned interference, tiny per-token router |
| `OscillatorySSM` | `h[t+1] = damping * R(theta) @ h[t] + gate * B @ x[t]`, Cayley-transform rotation |
| `PhaseAssociativeMemory` | 1024 learned slots, chunked top-k retrieval |
| `EpisodicMemory` | Sliding window via FlashAttention SDPA |

**No attention layers. No FFN blocks. O(n) backbone.**

Similarity: phase coherence `Re(a * conj(b)) / (|a| * |b|)` instead of dot product.

All rotations via Cayley transform (arithmetic only, no sin/cos/exp):
```
cos_like = (1 - a²) / (1 + a²)
sin_like = 2a / (1 + a²)
```

**Init strategy**: Default (random). Orthogonal initialization was not used in V4 -- it was introduced and benchmarked in V5.

---

## 2. Training Runs

### Run V4-10k (March 1, 2026 -- A6000)

**Setup**: 178M params, batch=64, A6000 GPU, no compile, 10k TinyStories samples (0.5% of dataset).

| Epoch | Train PPL | Val PPL | Train CE |
|-------|-----------|---------|----------|
| 1 | 200.86 | 76.47 | 5.30 |
| 2 | 32.75 | 48.92 | 3.49 |
| 3 (partial) | ~26 | -- | ~3.26 |

Starting PPL: 55,000 (random baseline). Dropped to val PPL 49 in 2 epochs (~40 min on A6000).

**Generation (Epoch 1, 10k samples)**:
> "The quick brown house. They run and start to get a smile. Mom were very excited. Now mommy and big yellow room. There said and She are friends. Tim, she started to save the garden."

---

### Run V4-100k (March 1, 2026 -- A6000, EDIT 1 of Reddit post)

**Setup**: Same 178M model, batch=64, A6000, no compile, 100k samples (5% of dataset). 1612 batches/epoch, ~3.5 hours per epoch.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 24.00 | 18.95 | Epoch 2 opened at step-1 PPL 12.77 |
| 2 | 11.96 | 14.07 | Train-val gap suggests overfitting |
| 3 (partial) | ~10.5 | -- | Plateau, training stopped |

Train-val gap at Epoch 2 (11.96 vs 14.07): model beginning to overfit on 100k samples.

**Generation (Epoch 1, 100k samples)**:
> "The quick brown were full. Steve and Brown loved each other. At the end of the hill, the friends were very happy. They had lots of fun and shared stories. Mam and Brown were the best day ever. All of their weeks were very good friends and would often enjoy their joy! The end had had a good time with them."

Proper story structure, multiple characters, emotional arc, ending. Grammar mostly correct. 10x improvement over 10k-sample run.

**Generation (Epoch 2, 100k samples)**:
> "The quick brown boy had ever seen. But one day, the sun was setting. The next night, the room got dark. Tom and the girl continued to admire the rain. The end was so happy to be back and continued to sail in the park. And every night, the end of the day, the family and the people stayed happy. They all lived happily ever after."

Proper narrative flow, temporal transitions ("one day", "the next night"), emotional resolution ("lived happily ever after"), multi-sentence coherence.

---

## 3. Independent Community Evaluation (Reddit comment, March 2026)

A community member ran a controlled 3-way comparison on a DGX Spark with the same hyperparameters across all models.

**Setup**:
- Dataset: 20k TinyStories samples
- Tokenizer, optimizer, schedule: same for all models
- Epochs: 20
- Scale: small (256 dim, 8 layers)

| Model | Core Params | Best Val PPL | Best Val Loss | Time/Epoch |
|-------|-------------|--------------|---------------|------------|
| Transformer | ~8M | 7.56 | 2.02 | 82s |
| SSM (DiagRNN) | ~9.5M | 9.18 | 2.22 | 512s |
| V4 | ~11.9M | 17.05 | 2.84 | 1,370s |

**Key observations from the evaluator**:

1. **V4 does learn**: Loss drops consistently across all 20 epochs.
2. **V4 converges to ~2.25x transformer perplexity** while taking ~17x longer per epoch.
3. **The relevant comparison is V4 vs SSM** (both O(n)). The SSM baseline is V4's backbone without Phase2D, banks, or associative memory -- a real-valued diagonal linear recurrence with the same hidden dim. SSM reaches 9.18 PPL vs V4's 17.05. That gap isolates the cost of the V4 machinery on top of a plain SSM.
4. **Single bank problem**: The default small config ships with a single bank, so routing entropy = 0.0 and bank specialization cannot be tested. The evaluator ran a 2-bank variant separately.
5. **Sequential backbone loop is the dominant cost** (confirmed independently).

**Generation quality timeline** (small scale, 20k samples):
- Transformer: coherent stories with dialogue by epoch 5
- SSM (DiagRNN): coherent stories by epoch 7
- V4: still producing fragments like "sortsang parents laughed" and encoding artifacts at epoch 20

---

## 4. Theoretical Critique (from Community Evaluation)

The evaluator raised a fundamental question about phase encoding in language:

**Where complex-valued representations DO have theoretical basis**:
- State evolution: a complex eigenvalue `λ = |λ|·e^(iθ)` gives a damped oscillator, naturally decomposing sequences into frequency components and preserving information over long distances. This is the basis of S4, LRU, etc.

**Where the theoretical justification is weaker**:
- Applying Phase2D to embeddings, bank layers, coupler, and memory (not just recurrence). Language lacks the frequency/phase structure that makes complex numbers natural in audio, signal processing, and physics.
- `InterferenceCoupler`'s complex multiplication between bank outputs is mathematically equivalent to a 2×2 real matrix multiply with shared weights. The "interference" framing borrows physics intuition that the underlying math doesn't require.

**What the evaluator found genuinely interesting**: Multi-bank routing with learned specialization. If 2-bank results show low cosine similarity between bank outputs and meaningful routing patterns, that's worth further research -- and doesn't require complex-valued representations to work.

---

## 5. Known Bugs (Identified Retrospectively)

These bugs were present in V4 and identified later during V5 and V6 development:

### Bug 1: Diversity Loss Normalization (L1 instead of L2)

`compute_diversity_loss` used `cabs(a).sum(dim=-1)` (L1 norm) instead of `torch.sqrt(cabs(a).square().sum(dim=-1) + 1e-8)` (L2 norm). By Cauchy-Schwarz, the L1 denominator is systematically too large, making even identical banks appear diverse. This is why `div=0.0000` appeared throughout training -- bank specialization loss was effectively a no-op.

**Impact**: Bank specialization was not being enforced. Banks may have been learning similar representations without the diversity penalty working.

**Fixed in**: V6 (`v6/core/bank.py`)

### Bug 2: Phase Information Loss Through Activations

V4 used standard activations (GELU, ReLU) on the real and imaginary parts separately, which does not preserve phase. When you apply `GELU(real_part)` and separately `GELU(imag_part)`, the phase relationship `angle = atan2(imag, real)` is destroyed because GELU is not phase-equivariant.

**Impact**: The model could not maintain consistent phase information through multiple layers. This partially explains the convergence gap vs the SSM baseline.

**Fixed in**: V5 introduced `ModReLU`, `ComplexLinear`, and `ComplexNorm` to preserve phase through all operations. V6 inherited these.

### Bug 3: Episodic Memory Introduces O(n) Attention

V4 included `EpisodicMemory` using FlashAttention SDPA for sliding window episodic recall. This reintroduces O(n × buffer_size) attention within the sliding window, technically breaking the O(n) claim for that component.

**Fixed in**: V6 replaces episodic memory with Working Memory (phase-coherence retrieval, O(n × num_slots), no attention).

---

## 6. What V4 Got Right

Despite the convergence gap and identified bugs:

1. **Phase-space representation as a design space** -- confirmed viable for language modeling
2. **Named banks with specialization** (SemanticPhaseBank + ContextPhaseBank) -- a genuinely novel idea, revived and fixed in V6
3. **Phase-coherence as a retrieval mechanism** -- `Re(query * conj(key)) / (|query| * |key|)` for associative memory lookup, extended in V6 to all memory types
4. **O(n) backbone without attention** -- the model learns and generalizes (val PPL drops), validating that attention is not strictly necessary
5. **Cayley-transform rotations** -- zero trig in hot path, GEMM-compatible

---

## 7. V4 → V5 → V6 Evolution Summary

| Aspect | V4 | V5 | V6 |
|--------|----|----|-----|
| Phase-safe activations | No (GELU breaks phase) | Yes (ModReLU, ComplexLinear, ComplexNorm) | Yes (same as V5) |
| Named banks | Yes (2 banks) | No (removed, replaced with AlgebraicBank) | Yes (revived, rebuilt with V5-safe ops) |
| Attention | None (backbone), SDPA (episodic memory) | Sparse (every k layers) | None anywhere |
| Memory | PhaseAssociativeMemory + EpisodicMemory | AlgebraicBank fusion | Working + Internal + Persistent + Expert |
| SSM initialization | Random/default | 13 strategies benchmarked, orthogonal best | Multiscale (fast/medium/slow lanes) + orthogonal |
| Diversity loss | L1 norm (bug) | L1 norm (bug, documented) | L2 norm (fixed) |
| Parameters (comparison run) | 178M | ~28M (small-matched) | ~29M (small-matched) |
| Val PPL (100k, Ep 2) | 14.07 | 13.14 (orthogonal) | **12.31** (best) |
| Throughput (100k run) | ~1612 batches/epoch, 3.5 hrs | ~16k tok/s | ~16.4k tok/s |

V6 is the synthesis: V4's architectural novelty (named banks, phase-coherence memory, no attention) + V5's mathematical correctness (phase-safe ops throughout) + new memory system (working/internal/persistent/expert) and strict O(n) guarantee.
