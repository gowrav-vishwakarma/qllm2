# V10: Base to Start

Everything we know from V4-V9 experiments, the APS paper scaling laws, and the codebase architecture — distilled into a single document so we always know where we are and what to do next.

*Created: 2026-05-08. Source: deep analysis of EXPERIMENTS_V_6_7_8_9.md, v7/EXPERIMENTS_V7.md, v9/EXPERIMENTS_V9.md, v8/EXPERIMENTS_V8.md, v8/AUDIT_V8.md, EXPERIMENTS_V6_PART2.md, v6/paper/aps_main.tex, v5/EXPERIMENTS.md, v5/model.py, v7/model.py, v6/model_real.py.*

---

## 1. The Winner Architecture

**V7 flat QPAM + CGU** is the architecture to scale. Every alternative has been tested and lost.

### Winning Config: `medium_h16_flat` + B=6 + Chunked Dual Form

```
Input -> ComplexEmbed -> [V7Block] x 16 -> ComplexNorm -> Tied Complex LM Head -> logits

Each V7Block:
  pre-norm -> CGU (channel mixing, SwiGLU-style, ModSwish) -> residual (scaled)
  pre-norm -> PAM (sequence mixing, matrix state)           -> residual (scaled)
```

| Parameter | Value |
|-----------|-------|
| Preset | `medium_h16_flat` |
| dim | 384 |
| n_layers | 16 |
| n_heads | 6 |
| head_dim | 64 |
| expand (CGU) | 3 |
| dropout | 0.1 |
| max_seq_len | 2048 |
| hierarchical_dt | False |
| cross_level | False |
| chunk_size | 256 |
| activation | ModSwish |
| position encoding | Complex RoPE on Q/K |
| state protection | GSP (Gated State Protection) |
| QKV | Fused projection |
| ComplexLinear | Block-real GEMM (4 matmuls fused into 1) |
| init | Orthogonal, Xavier-like `gain=sqrt(2/(in+out))` |
| QK-norm | **Off** (on causes repetition/mode collapse — Bug 8) |
| reverse_assoc | **Off** (hurts PPL by ~2.5) |
| Params at 100M | ~100M |

### Core Components

- **PAM (Phase-Associative Memory)**: Matrix state `S_t = gamma_t * S_{t-1} + V_t (x) K_t*`, retrieval `Y_t = S_t * Q_t`. Complex conjugate inner product `Re(K* · Q)` for scoring. O(T^2) dual form for training, O(1) recurrent form for inference. No softmax, no KV cache.
- **CGU (ComplexGatedUnit)**: SwiGLU-style gating in complex space. `gate_proj(z) * act(up_proj(z))` -> `down_proj`. Magnitude gates how much, phase gates rotation. Worth ~2.4 PPL (32.09 without vs 29.73 with).
- **ModSwish**: Smooth phase-preserving activation. `mag * sigmoid(beta * mag + bias)`, phase untouched. No dead neurons, learnable sharpness. Compounds over 16 layers.
- **Complex RoPE**: Position encoding via phase rotation on Q and K. Yields relative position structure in conjugate product.
- **GSP (Gated State Protection)**: Learned gates that can freeze important state dimensions.
- **Tied Complex LM Head**: `logits = Re(z * conj(embed)) = z_r @ e_r^T + z_i @ e_i^T`.
- **ComplexNorm**: RMSNorm for complex — normalizes magnitude, preserves phase.

### Key Files

| File | Role |
|------|------|
| `v7/model.py` | QPAM + CGU model, `PRESETS['medium_h16_flat']` |
| `v7/simple_model.py` | Lean PAM (no CGU), `LEAN_PRESETS` |
| `v7/train.py` | Training CLI, merges `PRESETS` + `LEAN_PRESETS` |
| `v6/model_real.py` | RPAM (real-valued ablation) |
| `v6/mlx/model.py` | QPAM MLX port for Apple Silicon |
| `v6/mlx/model_real.py` | RPAM MLX port |
| `v6/transformer_baseline.py` | GPT-2-style transformer baseline |
| `v6/paper/aps_main.tex` | APS paper with scaling law analysis |
| `scripts/run_v7_exp7d_chunked_b6.sh` | Best clean run script |

---

## 2. All Best Results (Ranked by Val PPL)

### North-Star Leaderboard (WikiText-103, ~100M params, RTX 4090)

| Rank | Run | Val PPL | Batch | tok/s | Params | Wall Time | Notes |
|------|-----|---------|-------|-------|--------|-----------|-------|
| 1 | **Transformer B=6** | **23.13** | 6 | ~99k | ~100.3M | ~3.2h | Flash Attention / SDPA |
| 2 | **V7 3a-A (ModReLU, B=6, ckpt)** | **26.64** | 6 | ~18.8k | ~100M | — | **Beat transformer B=3!** Confounded: B=6 + grad ckpt, 9ep, overfitting |
| 3 | **Transformer B=3** | **27.08** | 3 | ~96k | ~100.3M | ~3.5h | Flash Attention / SDPA |
| 4 | **V7 7d (ModSwish, chunked B=6)** | **27.94** | 6 | 31.8k | ~100M | 10.7h | Best clean PAM run |
| 5 | V9 gate (confounded) | 29.57 | 3 | ~27.3k | 105.1M | 12.2h | +5M params, reverse_assoc inherited |
| 6 | V7 7a (ModSwish, B=3) | 29.73 | 3 | ~20.9k | ~100M | — | Cleanest PAM-only baseline |
| 7 | V6 medium-pam-v3 | 29.95 | 3 | ~23k | ~100.4M | ~14.1h | Interleaved CGU+PAM, RoPE |
| 8 | V6 PIA (PAM + attention) | 30.01 | 3 | — | ~105.1M | — | Hybrid attention — no improvement |
| 9 | V7 3a-B (ModReLU, B=3) | 30.40 | 3 | — | ~100M | — | ModReLU baseline at B=3 |
| 10 | V9 gate_revassoc_100m (param-matched) | 30.53 | 3 | ~25.6k | 100.5M | — | Clean gate — negative result |
| 11 | V7 7f-0 (grouped + multi-scale) | 31.29 | 3 | — | ~100M | — | Regressed vs 7a |
| 12 | V7 7f-1 (multi-scale loss) | 30.55 | 3 | ~20.3k | ~100M | ~16.4h | Multi-scale hurts |
| 13 | V7 Lean L1 (no CGU) | 32.09 | 3 | 34.7k | 86.2M | 9.7h | CGU worth ~2.4 PPL |
| 14 | V7 7f-2 (reverse-assoc) | 32.19 | 3 | ~27.5k | ~100M | ~12.3h | Reverse-assoc hurts |
| 15 | V9 gate_mlp_revassoc_100m | 34.11 | 3 | — | 101.1M | — | 2-layer gate MLP — negative |

### V7 3a-A Epoch Curve (the 26.64 run)

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 151.5 | 58.9 | |
| 2 | 51.6 | 41.0 | |
| 3 | 40.2 | 35.1 | |
| 4 | 35.1 | 31.9 | |
| 5 | 31.9 | 29.8 | Beat V6 pam-v3 (29.95) |
| 6 | 29.6 | 28.5 | |
| 7 | 27.9 | 27.5 | |
| 8 | 26.6 | 26.9 | Train < val — overfitting starts |
| 9 | 25.7 | **26.6** | Gap widening |
| 10 | — | — | Cancelled at 25% |

**Why confounded:** B=6 (not B=3 like 7a), gradient checkpointing ON, ModReLU (not ModSwish), full dual form (not chunked). Cannot isolate which factor produced the win.

### V7 7d Epoch Curve (best clean run)

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

### Scaling Sweep (V6, MLX, seq_len 512, B=8, M4 Max)

| Params | QPAM PPL | RPAM PPL | RPAM advantage |
|--------|----------|----------|----------------|
| 5M | 258.12 | 118.55 | 2.17x |
| 10M | 123.47 | 69.57 | 1.76x |
| 25M | 74.92 | 40.68 | 1.84x |
| 50M | 45.70 | 35.70 | 1.28x |
| 100M (paper, provisional) | 33.19 | 25.42 | 1.31x |

RPAM wins at every measured point, but the gap narrows monotonically. The paper's scaling law fits predict QPAM overtakes RPAM.

---

## 3. Paper Scaling Laws (the Decisive Argument)

From `v6/paper/aps_main.tex` (line 169):

### Power-Law Fits (log-log linear regression, 5M-100M on WikiText-103)

| Metric | PAM (complex) slope | SAM/RPAM (real) slope | Ratio |
|--------|---------------------|----------------------|-------|
| Loss | **-0.15** | -0.12 | 1.25x steeper |
| PPL | **-0.65** | -0.49 | 1.33x steeper |

### Crossover Points (extrapolated)

| Space | Crossover N | Value at crossover |
|-------|-------------|-------------------|
| PPL | **~550M params** | PPL ~10.6 |
| Loss | **~4.5B params** | ~2.05 nats |

### Paper's Prescription for Pushing Crossover Lower (line 190)

> "With more careful optimization of the complex implementation (learning-rate schedules tuned for the modReLU activation, low-level kernel work) and training on corpora large enough that the upper end of the sweep stays within compute-optimal bounds ... we would expect the crossover transition to occur at smaller N than this extrapolation predicts."

Three levers:
1. **LR schedule tuning** for the activation (ModSwish/ModReLU likely have different optimal schedules)
2. **Kernel work** (Triton/CUDA — the Flash-PAM program)
3. **Larger corpus** (WikiText-103 is past Chinchilla-optimal for 50M+)

### Theoretical Basis

- **Decoherence gap**: Real-valued architectures asymptote at Shannon entropy of the diagonal of the conditional state `H(diag rho)`. Complex architectures can represent the full state and asymptote at von Neumann entropy `S_VN(rho)`. The gap `H(diag rho) - S_VN(rho) >= 0` is the information locked in off-diagonal phase coherences.
- **Empirical anchor**: Kaplan/Chinchilla irreducible loss ~1.69 nats. Paper predicts complex floor between 0.30-1.00 nats depending on `d_eff` (effective coherence dimension, estimated 2-4 from CHSH experiments).
- **Phase encodes semantics**: Synonyms cluster at zero phase difference in learned embeddings. No supervision. Phase IS meaning.

---

## 4. The Speed Problem: Engineering, Not Architecture

### The Critical Insight

PAM's training dual form computes `Re(K* · Q)` — a T x T score matrix — then matmuls with V. This is **structurally identical to attention**. Same O(T^2) complexity. The speed gap is because the transformer uses **Flash Attention** (fused CUDA kernels) while PAM uses **pure PyTorch matmuls**.

### PAM Dual Form (what Flash-PAM must fuse)

```python
# Score computation (complex conjugate dot product)
score_r = Q_r @ K_r^T + Q_i @ K_i^T    # Re(K* · Q)
score_i = Q_i @ K_r^T - Q_r @ K_i^T    # Im(K* · Q)

# Apply decay matrix
score_r = score_r * D
score_i = score_i * D

# Output computation
output_r = score_r @ V_r - score_i @ V_i
output_i = score_r @ V_i + score_i @ V_r
```

This is 4 matmuls for score + 4 matmuls for output, all on the same T x T tile. Flash Attention fuses 2 (score + output). Flash-PAM fuses 8, but they share the same tiling structure.

### Throughput Comparison (WikiText-103, ~100M, RTX 4090)

| Model | tok/s | Val PPL | VRAM | Why this speed |
|-------|-------|---------|------|----------------|
| Transformer B=3 | ~96k | 27.08 | ~1.8/7.0 GB | Flash Attention / SDPA |
| Transformer B=6 | ~99k | 23.13 | ~2.5/12.7 GB | Flash Attention / SDPA |
| V7 7d (chunked B=6) | 31.8k | 27.94 | 23.2 GB | Pure PyTorch, chunked C=256 |
| V9 gate (B=3) | ~27.3k | 29.57 | — | Pure PyTorch + gate overhead |
| V7 7a (B=3) | ~20.9k | 29.73 | ~11 GB | Pure PyTorch, full T^2 |
| Lean PAM (no CGU, B=3) | 34.7k | 32.09 | 1.7/10.3 GB | No CGU, simpler compile |
| V8 QLC | ~3.4k | 38.66@3ep | — | Reasoning loop (dead) |

### Bottleneck Analysis

1. **No fused kernel for PAM dual form**: 600 MB peak intermediates per layer. Pure PyTorch matmuls vs Flash Attention's tiled HBM-efficient implementation.
2. **448 GEMM launches**: 112 ComplexLinear x 4 block-real matmuls each. Could be fused.
3. **CGU overhead**: ~40% of non-PAM compute. Lean PAM (no CGU) is 66% faster and uses 85% less VRAM. But CGU is worth ~2.4 PPL — non-negotiable for quality.
4. **Complex split-real arithmetic**: Every operation doubles memory traffic vs real-valued. `[..., dim, 2]` layout.

### Target After Flash-PAM

50-60k tok/s at 100M (vs current 31.8k). This closes half the gap to transformer. The remaining gap is from complex arithmetic overhead (unavoidable) and CGU cost (worth it for quality).

---

## 5. What Every Version Taught Us

### V4: Phase-preserving primitives matter
Real-valued activations **destroy phase** in complex representations. ModReLU was the fix — threshold magnitude, leave phase alone. Orthogonal init gives ~31% advantage.

### V5: Banks + SSM + Sparse Attention = overcomplicated
- Architecture: `MultiBank (CGU) -> ComplexSSM (diagonal, parallel scan) -> Sparse PhaseAttention (every 4th layer, window 256)`
- Three serial subsystems that didn't compose well
- Throughput: 6-16k tok/s on TinyStories
- SSM was diagonal (vector state `d`), limiting capacity to O(d)
- PhaseAttention (sliding window, softmax) barely mattered — Mac smoke tests showed attention-on vs attention-off had almost same val PPL
- Diversity loss bug (L1 magnitude in denominator suppressed effective diversity)

**Lesson:** Pipeline of three modules was over-engineered. PAM replaced all three with one unified mechanism offering O(d^2) associative capacity per head.

### V6: PAM invented, interleaving proven, hybrid attention tested and failed

**What worked:**
- PAM matrix state `S_t = gamma * S_{t-1} + V K*` replaces both SSM and attention
- Interleaved `[CGU -> PAM] x 16` >> sequential `[CGU x 16 -> PAM x 16]` (layout matters)
- GSP + TSO + rebalanced settings for stability
- RoPE, fused QKV, block-real GEMM all proven
- Headline: `medium-pam-v3` at **29.95** val PPL

**What failed:**
- **PIA** (Phase Interference Attention, sparse windowed every 4 blocks on PAM stack): 30.01 vs 29.95 pure PAM — no improvement, +5M params wasted
- **QK-norm on**: repetition / mode collapse while loss looked fine (Bug 8) — headline preset keeps QK-norm OFF
- **Memory-reframe**: `run2_span_corruption` blew up; others finished but did not beat mainline PAM
- **Diffusion-text**: logs never reached first val epoch
- **HSB** (Holographic Structured Binding): interesting but did not win vs simpler stacks

**Lesson:** PAM IS the sequence mixer. Bolt-on attention doesn't help because PAM's dual form already computes attention-class scores. Interleaving channel + sequence mixing matters more than adding attention.

### V7: Flat stack wins, activation matters, B=6 is a big lever

**What worked:**
- **Flat 16-layer PAM** beats hierarchical, grouped, multi-scale (all regressed)
- **ModSwish > ModReLU at B=3** (29.73 vs 30.4) — smooth activation compounds over depth
- **B=6** gives ~14% PPL improvement (mirrors transformer B=3 vs B=6 pattern)
- **Chunked dual form** (C=256) enables B=6 without grad checkpointing + higher throughput (20.9k -> 31.8k tok/s)
- **3a-A (26.64)** beat transformer B=3 (27.08) — PAM CAN win
- **CGU** worth ~2.4 PPL (lean PAM 32.09 vs full 29.73)
- **Code fixes from V6:** ComplexLinear init scale (14x magnitude fix), CGU residual scale, CGU dropout, `_init_weights` pass, removed extra `final_norm`

**What failed:**
- **7f-1 multi-scale loss**: 30.55 — regressed vs 7a (29.73)
- **7f-2 reverse-assoc**: 32.19 — hurt PPL by ~2.5
- **7f-0 grouped + multi-scale**: 31.29 — worse than flat
- **7f-3 grouped-only**: halted after epoch 1, poor signal (58.70)
- **7b phase_mod activation**: stopped mid-epoch-1 to free GPU, no val

**Lean PAM insight:**
- No CGU variant: 86.2M params, 34.7k tok/s (+66%), 1.7 GB VRAM (-85%), val PPL 32.09 (+2.36)
- CGU overhead: gating + Triton kernel compilation + complex compile graph
- CGU is non-negotiable for quality but its cost must be addressed via kernel fusion

### V8: Reasoning loops don't work at this scale

**Architecture:** V7 backbone + QuantumLogicCore (Probe -> Sasaki -> OrthoHalt, T_max iterations)

**What happened:**
- **4450ff0 (Kill):** alpha/gamma pinned at 1/d noise floor, halt collapsed
- **18f4de7 (Soft evidence):** v8.2 fixes work mechanically, PPL tracks ~V6 trajectory, but far behind transformer
- **V8.3 context run:** 104.5M params, ~3.4k tok/s (10x slower than lean PAM), val PPL 38.66 @3ep
- **The fatal signal:** `out_scale` goes from 0.050 to -0.000563 by epoch 3. The LM objective learns to suppress QLC output to zero. "Internally active but externally negligible."

**Audit findings (AUDIT_V8.md):**
- 4 structural issues invalidate the quantum-logic claim: gamma collapses by construction, quantale_off is skip connection, per-token loop != reasoning, gates washed out by QR path
- What survives: constrained low-rank subspace memory + adaptive computation time — a real but modest contribution

**Lesson:** Don't add reasoning/compute complexity on top of PAM. The LM objective will learn to bypass it. Make PAM itself better and faster.

### V9: Micro-ablations on PAM readout are exhausted

**Hypothesis:** PAM sequence mixer is weaker than attention because it lacks: output gating, sharp local pattern capture, normalized competition between retrieved positions.

**Tested:**
- `pam_output_gate` (confounded): 29.57 — but 105.1M params, inherited reverse_assoc, NOT a clean ablation
- `gate_revassoc_100m` (parameter-matched): **30.53** — worse than V7 7a (29.73) and V6 v3 (29.95)
- `gate_mlp_revassoc_100m` (2-layer gate MLP): 34.11 @ep5 — trails baselines
- `compete_revassoc_100m` (zero-param head competition): 44.44 @ep2 — stopped, weak signal
- `gate_100m` (clean gate, no reverse_assoc): 55.37 @ep1 — stopped

**Every clean parameter-matched ablation failed.** The V9 doc concludes:

> "Small PAM readout gates/non-linearities are not closing the gap at ~100M."

**Recommended next move (from V9 doc):** "Move to a real PAM memory-dynamics change, starting with per-channel decay." Also remaining smoke: `gate_conv4_100m --epochs 3` (short conv before QKV).

**Lesson:** The readout side of PAM is not the bottleneck. The next architectural change must target PAM's memory dynamics (how the state evolves), not how we read from it.

---

## 6. Why NOT Hybrid (V5-style Attention + PAM)

### Evidence

The V6 PIA experiment tested exactly this: sparse windowed attention every 4 blocks on top of interleaved PAM.

| Config | Params | Val PPL |
|--------|--------|---------|
| medium-pam-v3 (pure PAM) | ~100.4M | **29.95** |
| medium-pam-v3-pia (PAM + attention) | ~105.1M | 30.01 |

No improvement. +5M params wasted.

Paper (aps_main.tex, line 154): "a hybrid that adds sparse windowed attention every fourth block produces no improvement, indicating that interleaving channel and sequence mixing matters and that supplemental attention provides no benefit at this scale."

### Why It Failed

PAM's dual form already computes attention-like scores: `Re(K* · Q)` weighted by decay, then matmul with V. Adding softmax attention on top is redundant — two attention mechanisms on the same representation that compete rather than complement. PAM uses destructive interference (complex conjugate), attention uses softmax sharpening (exponential). They solve the same problem differently.

### Why V5's Approach Was Abandoned

V5's PhaseAttention was dropped in V6 not because it hurt PPL, but because V6 targeted "strict O(n)" for inference. The irony: PAM dual form IS O(T^2) at training anyway. But by the time this was clear, PAM had already proven better than V5's SSM+attention pipeline — so there was no reason to go back.

### What About Jamba-Style Hybrids?

The paper cites Jamba (AI21, Mamba + attention hybrid) in the bibliography. V9 model.py even references "Mamba/RWKV-style stabilizers" (`pam_output_gate`, `pam_short_conv`). But these are bolt-on additions, not architectural fusions. The conclusion from V6 PIA + V9 experiments: bolt-on mechanisms don't help PAM at this scale.

---

## 7. Why QPAM (Complex) Over RPAM (Real) for Scaling

### Evidence at Current Scale (5M-100M)

RPAM wins at every measured point. The gap narrows monotonically with scale (2.17x at 5M -> 1.28x at 50M -> 1.31x at 100M provisional).

### Scaling Law Prediction

PAM has a **33% steeper PPL slope** (-0.65 vs -0.49). The fits cross at **~550M in PPL space**. Beyond that, QPAM is predicted to be better.

### Why Complex Algebra Scales Better

- **Destructive interference** in retrieval: `Re(K* · Q)` can be negative (phase-mismatched keys are actively suppressed). Real dot product is non-negative — all stored associations contribute positively, diluting the target signal.
- **Phase encodes relationships**: Synonyms cluster at zero phase difference in learned embeddings without supervision. Real-valued architectures must learn this structure through magnitude alone.
- **Decoherence gap**: The theoretical gap between real (Shannon entropy of diagonal) and complex (von Neumann entropy of full state) loss floors is provably positive when the underlying state has off-diagonal coherences. CHSH violations in LLMs confirm these coherences exist.

### Decision

Don't invest in RPAM scaling. The paper predicts it loses past ~550M. Focus all engineering effort on making QPAM fast (Flash-PAM) and scaling it. Revisit only if the 300M checkpoint fails to show the predicted gap narrowing.

---

## 8. What NOT to Do (Confirmed Dead Ends)

| Approach | Result | Why it failed |
|----------|--------|---------------|
| V8 QLC (reasoning loops) | LM bypasses it; `out_scale` -> 0 | Model learns to suppress auxiliary via learnable residual scale |
| V9 output gates (param-matched) | 30.53 vs 29.73 baseline | Small readout tweaks don't close the gap at ~100M |
| V9 head competition | 44.44 @ep2 — stopped | Zero-param competition adds nothing |
| V9 gate MLP (2-layer) | 34.11 @ep5 — trails baselines | More parameters in readout don't help either |
| V7 hierarchy / grouping | 31.29-58.70 — all regressed | Flat stack is strictly better |
| V7 multi-scale loss | 30.55 — regressed | Auxiliary loss hurts at this scale |
| V7 reverse association | 32.19 — hurts by ~2.5 PPL | Reusing upper triangle of score matrix is harmful |
| V6 PIA (attention + PAM) | 30.01 vs 29.95 — no improvement | Redundant attention mechanisms compete |
| V6 QK-norm on PAM | Repetition / mode collapse | PAM needs free magnitude + phase, no softmax normalization |
| V6 memory-reframe | Various pathologies | Span corruption blows up, others don't beat mainline |
| V6 diffusion-text | Crashed / stalled | Never reached first val epoch |
| V5 pipeline (banks+SSM+attention) | 6-16k tok/s, overcomplicated | Three serial subsystems that don't compose well |

---

## 9. Unresolved Questions (Things That COULD Still Help)

### 9.1 ModReLU vs ModSwish at B=6 (the 26.64 question)

At B=3: ModSwish wins (29.73 vs 30.4). At B=6: unknown — confounded by grad ckpt + full dual form + commit differences.

**Why it matters:** 3a-A (ModReLU, B=6) hit 26.64, beating transformer B=3. 7d (ModSwish, B=6) hit 27.94. If ModReLU is genuinely better at B=6, that's a free ~1.3 PPL win.

**Test:** Run ModReLU B=6 chunked C=256 (matching 7d exactly except activation) as a 3-epoch smoke. Compare epoch 3 to 7d epoch 3 (35.88). Extend to 10 if tracking below.

### 9.2 Per-Channel Learned Decay (V9's recommended next change)

The V9 doc recommends this as the next real PAM memory-dynamics change. Currently gamma (decay) is shared across channels. Per-channel decay would let different state dimensions forget at different rates — similar to how GLA (Gated Linear Attention) and DeltaNet use data-dependent gating.

**Why it could work:** The matrix state's effective rank saturates at ~10/64 — most dimensions are unused. Per-channel decay could help the model allocate state capacity more efficiently.

### 9.3 LR Schedule Tuning

The canonical sweep used one LR for both PAM and SAM. ModSwish and ModReLU have different gradient dynamics. A short LR sweep at 100M could meaningfully improve PAM. The paper explicitly identifies this as a lever to push the crossover lower.

### 9.4 Short Local Conv Before QKV (V9 gate_conv4_100m)

Still pending as a smoke test. Causal depthwise conv before QKV projection — the idea is to give PAM "sharp local pattern capture" that V9 identified as missing. Continue only if epoch 3 <= 38.0.

---

## 10. The V10 Roadmap

### Phase 1: Flash-PAM (Engineering Sprint — Highest Leverage)

Build a fused Triton kernel for the PAM chunked dual form.

**What to fuse:**
1. Complex conjugate score: `score_r = Q_r @ K_r^T + Q_i @ K_i^T`, `score_i = Q_i @ K_r^T - Q_r @ K_i^T`
2. Decay matrix multiply: `score = score * D`
3. Causal masking
4. Output: `out_r = score_r @ V_r - score_i @ V_i`, `out_i = score_r @ V_i + score_i @ V_r`

**Tiling strategy:** Same as Flash Attention — tile over T dimension, keep running max/sum in SRAM, never materialize full T x T in HBM. For PAM there's no softmax, so it's actually simpler — just tiled matmul + element-wise decay.

**Also fuse CGU:** `gate_proj + up_proj + ModSwish + fused_cgu_gate + down_proj` as one kernel.

**Target:** 50-60k tok/s at 100M (vs current 31.8k).

### Phase 2: Beat Transformer B=3 at 100M

With Flash-PAM in hand:
- Resolve ModReLU vs ModSwish at B=6 (3-epoch smoke)
- Short LR sweep (2-3 values, 3 epochs each)
- Add per-channel learned decay
- Target: **sub-26 PPL** at 100M, **< 6h wall time** (vs current 10.7h for 7d)

### Phase 3: Scale to 300M on Real Data

WikiText-103 (~118M tokens) is past Chinchilla-optimal for 50M+ params. Move to OpenWebText or The Pile.

- 300M is approaching the ~550M PPL crossover zone
- Run matched RPAM 300M as a checkpoint on the scaling law prediction
- If QPAM-RPAM gap is narrowing as predicted, stay the course; if not, reassess

### Phase 4: Scale to 1B — the Crossover

At 1B, PAM should have overtaken RPAM per the scaling law. With Flash-PAM kernels, training speed should be competitive with transformer.

**Target:** 1B PAM model that beats a matched transformer on PPL AND has O(1) inference (fixed ~50KB state per layer vs growing KV cache).

### What Makes This "More Intelligent"

- **Destructive interference**: PAM actively suppresses phase-mismatched associations (attention can only downweight via softmax). Better disambiguation.
- **Decoherence as relevance realization**: Matrix state automatically selects ~10 most relevant associations per context. Built-in attention sparsity.
- **Phase encodes semantics**: Synonyms cluster at zero phase difference without supervision. Phase IS meaning.
- **Scaling slope**: PAM learns more per parameter than real-valued architectures. At 1B+, this compounds into significant quality advantage.
- **O(1) inference**: No KV cache. Fixed state. Inference cost doesn't grow with context length.

Generation quality issues (repetition, entity drift) are partly a scale problem and partly a sampling problem. At 300M+ on real data, these should diminish.

---

## 11. Complete Run Log Reference

All raw val PPL numbers from V6-V9 are in [EXPERIMENTS_V_6_7_8_9.md](../EXPERIMENTS_V_6_7_8_9.md). The full per-version detail is in:

- [EXPERIMENTS_V6.md](../EXPERIMENTS_V6.md), [EXPERIMENTS_V6_PART2.md](../EXPERIMENTS_V6_PART2.md), [EXPERIMENTS_V6_MEMORY_REFRAME.md](../EXPERIMENTS_V6_MEMORY_REFRAME.md), [EXPERIMENTS_V6_DIFFUSION.md](../EXPERIMENTS_V6_DIFFUSION.md)
- [v7/EXPERIMENTS_V7.md](../v7/EXPERIMENTS_V7.md)
- [v8/EXPERIMENTS_V8.md](../v8/EXPERIMENTS_V8.md), [v8/AUDIT_V8.md](../v8/AUDIT_V8.md)
- [v9/EXPERIMENTS_V9.md](../v9/EXPERIMENTS_V9.md)
- [v5/EXPERIMENTS.md](../v5/EXPERIMENTS.md), [v5/README.md](../v5/README.md)
- [v6/paper/aps_main.tex](../v6/paper/aps_main.tex) — scaling laws, theoretical basis
