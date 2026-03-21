# QLLM — Phase-First Language Model Research

**Attention-free, complex-valued language modeling:** tokens live in complex phase space; sequence memory is **Phase-Associative Memory (PAM)** — a matrix-state associative layer with complex-conjugate retrieval, not softmax attention and not a standard real-valued SSM.

> **Disclaimer:** We use AI assistants (e.g. Cursor) to help build this — because building AI with AI is useful. The architecture and experiments are documented in code and logs.

This repository is organized by generation: **v6** (current), **v5** (phase-preserving complex LM breakthrough), **v4** (original phase-field direction). Earlier branches **v3** (brain-inspired) and **v2** (quantum-inspired) remain for reference.

---

## TL;DR: Three Core Innovations

1. **Phase-first complex tokens** — Each token is complex: magnitude tracks salience, phase tracks *kind* of meaning. A single complex multiply `(a+bi)(c+di) = (ac-bd) + (ad+bc)i` has four cross-terms and behaves as rotation + scaling; the algebra is richer per parameter than “two independent real vectors.”
2. **Matrix-state associative memory (PAM)** — State is `S ∈ ℂ^{H×d×d}`, not a vector `s ∈ ℝ^{S×d}`. Capacity scales as **O(d²)** per head via outer-product storage.
3. **Complex-conjugate matching** — Retrieval uses **K\*·Q** (phase coherence), not **K·Qᵀ** + softmax. The core path does not need softmax normalization.

Together these define a **phase-first associative memory LM**: neither a transformer nor a standard SSM.

---

## The Core Idea: Tokens in Complex Phase Space

In a transformer, a token is a real vector refined by attention and FFN layers.

In QLLM, a token is **complex**: **magnitude** (how salient) and **phase** (what kind of meaning) are separate degrees of freedom. **Context shifts** (e.g. “bank” → finance vs river) are modeled as **phase rotations**; rotations compose and are invertible.

**Phase must be preserved end-to-end.** Early versions showed that passing complex activations through real nonlinearities (GELU, plain sigmoid gates) **destroys phase** and collapses the design. QLLM uses **phase-preserving** primitives: `modReLU`, `ComplexGatedUnit` (CGU), and `ComplexNorm` throughout the main path.

---

## ComplexGatedUnit (CGU)

**Standard GLU (typical transformer path):**

```python
gate = sigmoid(W_g * x)    # Real-valued gate
output = gate * (W_v * x)  # Controls HOW MUCH flows
```

The gate only controls **intensity**.

**QLLM’s ComplexGatedUnit:**

```python
# Gate magnitude: sigmoid(|W_g * z|)  → HOW MUCH
# Gate phase: arg(W_g * z)            → WHAT ROTATION
output = modReLU(gate_magnitude) * rotate(z, gate_phase) * (W_v * z)
```

This is **dual control**: magnitude **and** phase — only natural in complex space.

---

## Phase-Associative Memory (PAM)

Vector SSM states have **limited associative capacity**; storing many facts in one vector causes interference. An earlier **Holographic State Binding (HSB)** experiment failed for that reason — motivating **PAM**, which uses a **complex matrix state** per head.

### How it works (schematic)

```python
# State update
S_t = gamma_t * S_{t-1} + V_t ⊗ K_t*   # outer product; K* is conjugate

# Retrieval
Y_t = S_t @ Q_t
```

`K_t*` stores a full **d×d** association per (key, value) pair via the outer product.

### Standard attention vs PAM retrieval

**Transformer attention:**

```python
attention_scores = Q @ K.T / sqrt(d)
output = softmax(attention_scores) @ V
```

Dot products measure alignment; **softmax** normalizes over the sequence.

**PAM retrieval (conceptual):**

```python
coherence = K* · Q          # Complex inner product → phase coherence
output = V * coherence      # Weighted by coherence (see paper/code for full gating)
```

Aligned phases **constructively** interfere; misaligned phases **destructively** interfere — **interference**, not a length-softmax over positions (training uses a dual quadratic form; see below).

### Transformer vs SSM vs QLLM PAM

| Aspect | Transformer | SSM (e.g. Mamba) | QLLM PAM |
|--------|-------------|------------------|----------|
| **State** | N/A (KV cache) | `s_t ∈ ℝ^{S×d}` vector | `S_t ∈ ℂ^{H×d×d}` matrix |
| **Storage** | Append to cache | Linear projection | Outer product `V ⊗ K*` |
| **Matching** | QKᵀ + softmax | Gated recurrence | Complex conjugate `K* · Q` |
| **Capacity** | O(n) per seq | O(S·d) | O(H·d²) per layer |
| **Training** | O(T²) | O(T) | O(T²) dual form |
| **Inference** | O(T) per token | O(1) per token | O(1) per token |

PAM state is a **different object** than an SSM vector: it stores **rank-1 associations** between V and K, not only a linear recurrence on a single vector.

### Gated State Protection (GSP)

Learned gates can **freeze** important state dimensions so they are not overwritten. Empirically, GSP improved WikiText-103 PPL (e.g. medium-rebalanced: **44.47 → 41.67** with GSP). See experiment logs for details.

### Dual form

**Training:** quadratic (attention-like) dual form — friendly to dense GPU matmuls.  
**Inference:** recurrent form — **O(1) per token** with fixed state size (no KV cache growth with sequence length).

---

## V6 Architecture Paths (Where the Code Goes)

V6 is **modular**: named banks + SSM, single-bank + SSM, or **single-bank + PAM**. Headline WikiText-103 numbers use **`medium-pam-v3`**: interleaved **CGU → PAM** every block, GSP, **complex RoPE on PAM Q/K**, fused QKV; **`pam_qk_norm=False`** (QK phase norm **on** hurt generation — repetition; see [EXPERIMENTS_V6_PART2.md](EXPERIMENTS_V6_PART2.md) Bug 8).

**Named-bank path** (e.g. `small-matched`):

```text
Tokens --> ComplexEmbed
  --> [SemanticBank + ContextBank --> PhaseInterferenceCoupler] x N
  --> MultiTimescaleSSM
  --> WorkingMemory / InternalMemory / [PersistentMemory] / [ExpertMemory]
  --> MemoryFusion --> TiedComplexLMHead
```

**Single-bank + PAM — headline preset** (`medium-pam-v3`):

```text
Tokens --> ComplexEmbed --> ComplexNorm
  --> [ ComplexGatedUnit + residual
       -> PhaseAssociativeMemory (PAM) + residual ] x N
  --> ComplexLinear --> ComplexNorm --> TiedComplexLMHead
```

**Sequential PAM** (`medium-pam`): all CGU layers then all PAM layers — [EXPERIMENTS_V6_PART2.md](EXPERIMENTS_V6_PART2.md) §4.

Optional **multi-timescale SSM**, **memory hierarchy** (working / internal / persistent / expert), **optional PhaseAttention** — see [v6/README.md](v6/README.md).

---

## Results

### Headline run: `medium-pam-v3` (~100M params, WikiText-103)

Interleaved CGU+PAM, GSP, PAM RoPE on Q/K, `pam_qk_norm=False`, `pam_fused_qkv=True`, seq len 2048, 10 epochs, single **RTX 4090**, `torch.compile`, bf16 — see scripts and logs for full flags.

| Epoch | Val PPL |
|-------|---------|
| 1 | 57.94 |
| 2 | 43.83 |
| 3 | 38.69 |
| 4 | 35.88 |
| 5 | 33.82 |
| 6 | 32.25 |
| 7 | 31.22 |
| 8 | 30.40 |
| 9 | 30.01 |
| 10 | **29.95** |

Wall time ~14.1 h (logged run). Earlier **sequential** `medium-pam` (same ~100M budget, no RoPE, CGU then PAM) reached **38.95** at epoch 10.

### Architecture progression on WikiText-103 (V6)

| Config | Params | Val PPL (10 ep) | Notes |
|--------|--------|-----------------|--------|
| small-matched (SSM) | 28.7M | 49.61 | Vector SSM baseline |
| medium-rebalanced (TSO) | 58.4M | 44.47 | + params, timescale-separated output |
| medium-rebalanced-gsp | 63.2M | 41.67 | + GSP |
| medium-rebalanced-hsb | 75.0M | 43.54 | + HSB (vector state interference) |
| medium-pam | 100.4M | 38.95 | PAM + GSP; **sequential** CGU then PAM |
| **medium-pam-v3** | **100.4M** | **29.95** | **Interleaved** CGU+PAM + RoPE + fused QKV |

### Orientation (not apples-to-apples)

| Model | Params | Val PPL | Notes |
|-------|--------|---------|--------|
| GPT-2 Small | 124M | ~31 | WebText, different pipeline |
| **QLLM V6 (PAM v3)** | **~100M** | **~30** | WikiText-103 only, single 4090 |
| AWD-LSTM | ~24M | ~69 (WT2) | Different tokenization / split |

### TinyStories (V6)

| Run | Model | Best Val PPL | Notes |
|-----|-------|--------------|--------|
| `small_matched_full` | `small-matched` | **5.50** | Clean no-memory baseline (5/10 ep) |
| — | `small-matched` + working memory | 2.23 | 1 ep; likely overfit — not headline |

Example generation:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and make new friends. One day, she went on an adventure in the forest near her house.

### Sample (WikiText-103, `medium-pam-v3`, epoch 10)

Prompt: `In 1923 , the University of`

> In 1923 , the University of Illinois at Urbana @-@ Urdu said it was " an easy choice to do something in its own right . " The university also claimed the first students from Wisconsin had to be replaced by a more " good student " due to a lack of funds .

Fluent scaffolding; facts are **not** reliable at this scale. Logged quality after sample: `rep3=0.034 rep4=0.011 uniq=0.703`.

---

## How It Evolved

- **V4** — Complex phase-space tokens and wave-style interference; **real nonlinearities broke phase**; promising but inconsistent math.
- **V5** — **Phase-preserving** stack (`modReLU`, CGU, …). A **28.7M** model beat V4’s much larger runs; TinyStories val PPL **5.59** (full data, epoch 3). Details: [v5/README.md](v5/README.md).
- **V6** — Modular toolkit: banks, SSM, PAM, memory tiers, optional attention. **PAM** addresses vector-state interference; **`medium-pam-v3`** interleaves CGU and PAM and uses RoPE on PAM Q/K for the best WikiText-103 number above.

---

## Honest Limitations

- **No same-budget transformer baseline** on the identical WikiText-103 pipeline yet — the most important missing comparison.
- **Absolute SOTA** — GPT-2–class models use more data and compute; we report **~30** val PPL at **~100M** params on WikiText-103 only.
- **Factual reliability** — generations can be fluent but **wrong**; fact persistence probes on the checkpoint are not a success story yet.
- **Bank specialization** — diversity regularization encourages distinct banks; **strong evidence** of disentangled roles is still lacking.
- **No standard downstream benchmarks** (MMLU, HellaSwag, …) yet.
- **Pure PyTorch** — no custom CUDA/Triton; performance headroom remains.
- **Scaling** — behavior at 1B+ params is an open question.
- **Single-GPU, limited dataset diversity** in the headline runs — broader validation is needed.

The claim is **not** “beats transformers everywhere.” It is: **a genuinely different architecture class** that **learns real language** under documented training — worth iterating on honestly.

---

## Why This Direction Matters

- **Architectural diversity** — If the field only explores transformers and close variants, other viable families may be missed.
- **Phase preservation** is a **design constraint**, not branding — progress tracked math fixes, not parameter scaling alone.
- **PAM** combines matrix-state storage, complex-conjugate retrieval, and optional GSP in one trainable stack — a distinct memory mechanism from both attention and classic SSMs.
- **Inference** — Recurrent PAM inference is **O(1) per token** with bounded state; long-generation cost does not grow like a KV cache.
- **Accessibility** — The project is deliberately explored on **consumer-class GPUs** (e.g. RTX 4090) to keep research reproducible outside huge clusters.

---

## What Happens Next

- Same-budget **transformer baseline** on the WikiText-103 pipeline.
- **Scale** toward ~300M–500M params; test whether PAM improves with scale.
- **Factual / compositional binding** — can the matrix state be *used* for verifiable memory?
- Longer training / more data; **standard benchmarks** when the stack is ready.

---

## Quick Start

These are the **only presets documented here as validated “last known good” paths** — `small-matched` (TinyStories) and `medium-pam-v3` (WikiText-103 headline). Other presets and scripts (`run_v6_wikitext103.sh`, `run_v6_pg19.sh`, sequential `medium-pam`, diffusion, V5, …) live in [v6/README.md](v6/README.md) and older version dirs.

### Install

```bash
uv sync
uv sync --extra cuda
```

### Train

**1. Small — TinyStories (`small-matched`, ~28.7M params)**  
Documented val PPL **5.50** (full TinyStories, no extra memory flags in that run).

```bash
python -m v6.train --size small-matched --max_samples 9999999 --seq_len 256 \
  --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4
```

**2. Medium PAM v3 — WikiText-103 (`medium-pam-v3`, ~100M params)**  
Documented val PPL **29.95** (10 epochs, RTX 4090, RoPE on PAM Q/K, `pam_qk_norm=False`). Uses the tuned script (seq len 2048, bf16, `torch.compile`, …).

```bash
./scripts/run_v6_medium_pam_v3.sh
# Resume from checkpoint
./scripts/run_v6_medium_pam_v3.sh --resume
```

Optional **CPU smoke test** (not a headline result):  
`python -m v6.train --size tiny --epochs 5 --max_samples 1000 --seq_len 128`

### Generate

Use checkpoints produced by the commands above (paths may match your `--checkpoint_dir` / script defaults).

```bash
# After medium-pam-v3 training (WikiText-style)
python -m v6.generate --checkpoint checkpoints_v6_medium_pam_v3/best_model.pt \
  --prompt "In 1923 , the University of"

# After small-matched training (TinyStories)
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt \
  --prompt "Once upon a time"
```

Persistent memory and other generation options: [v6/README.md](v6/README.md).

---

## Model Presets (V6)

**Documented in this README for train/generate:**

| Preset | Role | Complex dim | Layers | State / PAM | Notes |
|--------|------|-------------|--------|-------------|--------|
| `small-matched` | TinyStories headline | 128 | 12 | SSM 512 | Named banks + multi-timescale SSM; val PPL **5.50** (documented run) |
| `medium-pam-v3` | WikiText-103 headline | 384 | 16 | PAM 6×64 | Interleaved CGU+PAM, GSP, RoPE on PAM Q/K; val PPL **29.95** |

**All other presets** (`tiny`, `medium-rebalanced-gsp`, sequential `medium-pam`, `large`, …): see [v6/README.md](v6/README.md). Memory slots default off; use `--wm_slots`, `--im_slots`, and persistent/session flags when experimenting.

---

## Project Structure

```text
qllm2/
├── README.md
├── v6/                    # current main line
│   ├── README.md
│   ├── model.py
│   ├── backbone.py
│   ├── diffusion_model.py
│   ├── train.py
│   ├── generate.py
│   └── core/
├── v5/
│   ├── README.md
│   ├── EXPERIMENTS.md
│   └── core/
├── v4/
├── v3/
├── v2/
└── scripts/
```

---

## Documentation

- [v6/README.md](v6/README.md) — Architecture, CLI, memory, diffusion modes, setup
- [EXPERIMENTS_V6.md](EXPERIMENTS_V6.md) — V6 experiment log (through GSP / §18)
- [EXPERIMENTS_V6_PART2.md](EXPERIMENTS_V6_PART2.md) — HSB, PAM, interleaved layouts, Bug 8 (QK norm)
- [v6/paper/](v6/paper/) — PAM paper draft (LaTeX)
- [v5/README.md](v5/README.md) — Algebraic LM, V4 → V5
- [v5/EXPERIMENTS.md](v5/EXPERIMENTS.md) — V5 experiment log
- [v4/README.md](v4/README.md), [v2/README.md](v2/README.md), [v3/README.md](v3/README.md)

**Research notes:** `QLLM_CORE_IDEA.pdf`, `v5/paper/`, `QLLM_V2.pdf`

---

## Contributing

1. Pick a version directory (`v6`, `v5`, …) and read its README.
2. Run a smoke test before large changes.
3. Keep claims tied to logs and configs.

---

## License

PolyForm Noncommercial License 1.0.0 — see [LICENSE](LICENSE). Non-commercial use allowed; commercial use requires permission — [gowravvishwakarma@gmail.com](mailto:gowravvishwakarma@gmail.com).

---

**Focus:** `v6` — PAM (`medium-pam-v3`), WikiText-103 val PPL **29.95** (~100M params). **Prior milestone:** `v5`. **Novelty origin:** `v4`. **Updated:** 2026-03-21
