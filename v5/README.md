# V5: Algebraic Language Model

A language model where every parameter is a **complex number** -- not just a single value, but a value with both **strength** (magnitude) and **direction** (phase). This lets each parameter encode richer information than a standard real-valued parameter.

---

## The Big Idea (No Math Required)

### How normal neural networks work

In a regular neural network, each "weight" is a single number. Think of it as a **volume knob** -- it controls *how much* of a signal to let through. Turn it up, more signal passes. Turn it down, less passes. That's all it can do.

### How V5 works

In V5, each weight is a **complex number** -- think of it as a knob that controls *two things at once*:

1. **How much** to let through (the "magnitude" -- like a volume knob)
2. **What direction to rotate** the signal (the "phase" -- like a steering wheel)

This means one V5 parameter does the job of two regular parameters. It doesn't just scale a signal -- it simultaneously scales AND rotates it.

### Why does rotation matter for language?

Words don't just have "strength" on various features. They have **relationships**:

- "happy" and "sad" aren't just different amounts of emotion -- they're **opposite** (180 degrees apart)
- "un-" doesn't reduce "happy" -- it **flips** it (a rotation)
- "king" is to "queen" as "man" is to "woman" -- that's a **direction**, not a distance

Complex numbers naturally represent all of these because they encode direction (phase) alongside strength (magnitude).

---

## What V4 Got Wrong (And What V5 Fixes)

V4 had the right idea (use complex numbers) but broke its own rules. Here's the problem:

**V4 created complex representations, then immediately destroyed them.**

Every time V4 needed a nonlinear activation (the thing that makes neural networks powerful), it fell back to standard real-valued operations like GELU -- which only look at the real part and throw away the phase information. It's like buying a sports car and only driving it in first gear.

Specifically, V4 did things like:
```python
# V4's BROKEN approach: apply GELU to just the real part
h = F.gelu(h[..., 0]).unsqueeze(-1) * h  # phase is DESTROYED here
```

V5 fixes this with **modReLU** -- an activation that preserves phase:
```python
# V5's CORRECT approach: threshold magnitude, keep phase intact
activated_mag = ReLU(|z| + bias)     # decide how much passes (based on magnitude)
output = activated_mag * (z / |z|)   # keep the ORIGINAL phase direction
```

The difference: V4 paid the cost of complex numbers (2x memory, 2x compute) but got none of the benefit. V5 preserves phase throughout the entire network.

---

## Architecture Overview

```
Tokens --> ComplexEmbed --> [Bank + SSM + Attention] x N --> LM Head --> Next Token
              |                                                |
         Each token            N layers of processing     Re(z · conj(embed))
        becomes a               (all in complex space)    tied weights: output
       complex vector                                     reuses embed table
```

### The Components

#### 1. ComplexEmbed -- "Give each token a direction"

Each token gets embedded as a complex vector. Unlike a standard embedding (just a list of numbers), this gives each token both magnitude (how prominent each feature is) and phase (what "direction" each feature points in).

```
"happy" --> magnitude: [0.8, 0.5, 0.3, ...]  +  phase: [0°, 45°, 180°, ...]
"sad"   --> magnitude: [0.8, 0.5, 0.3, ...]  +  phase: [180°, 225°, 0°, ...]
                         (similar strength)          (opposite directions!)
```

Code: `v5/core/complex.py` -- `ComplexEmbed`

#### 2. AlgebraicBank -- "Process the signal through multiple lenses"

Each layer has multiple "banks" (default: 2) that process the input differently. Think of it like reading a sentence for meaning vs. reading it for grammar -- same input, different processing.

Each bank uses a **ComplexGatedUnit (CGU)** -- the core innovation of V5. A CGU is like a SwiGLU block (used in LLaMA, Mistral) but complex-valued:

- **SwiGLU gate**: "let 70% of this feature through" (real number in [0,1])
- **CGU gate**: "let 70% through AND rotate it 30 degrees" (complex number)

The CGU gate simultaneously **selects** (magnitude) and **transforms** (phase). One complex gate does the work of two real gates.

A content-dependent **router** learns which bank matters most for each token, and an **AlgebraicFusion** module combines bank outputs via learned phase rotations -- genuine algebraic interference where outputs at different phases constructively and destructively combine.

**Scaling banks**: More banks = more parallel perspectives and richer phase interference (2 banks give 1 interference pair; 3 banks give 3 pairs). The tradeoff: more banks slow early training because the router, diversity loss, and fusion have more to coordinate -- but once specialized, they capture finer-grained nuances that fewer banks miss. The `bank_expand` factor controls each bank's internal width (expand=2 means the CGU projects up to dim\*2 internally before projecting back). A **diversity loss** penalizes bank outputs that are too similar, preventing them from collapsing into doing the same thing.

Code: `v5/core/bank.py` -- `AlgebraicBank`, `ComplexRouter`, `AlgebraicFusion`

#### 3. ComplexSSM -- "Remember what came before"

This is the sequential backbone -- it processes tokens one by one and maintains a hidden state (memory). It's a **selective state space model** (like Mamba) but with complex-valued state transitions.

Why complex is justified here (even the critics agree): a complex eigenvalue `A = |A| * e^(iθ)` naturally creates a **damped oscillator**:

- `|A|` controls how fast information fades (memory length)
- `θ` controls oscillation frequency (timescale)

Different state dimensions oscillate at different frequencies, giving the model a natural **multi-resolution view** of the sequence:
- Fast oscillators track word-by-word syntax
- Slow oscillators track paragraph-level topics

**Parallel scan**: V4 processed tokens sequentially (a `for` loop over each timestep -- very slow). V5 uses a **Blelloch parallel scan** that does the same computation in O(log n) parallel steps instead of O(n) sequential steps. This is the throughput fix.

Code: `v5/core/ssm.py` -- `ComplexSSM`, `parallel_scan`

#### 4. PhaseAttention -- "Look up relevant context" (sparse, every K layers)

Pure recurrence (SSMs) can't do content-addressable retrieval -- "what did the character say 200 tokens ago?" requires comparing the current token against past tokens by content, which is what attention does.

V5 uses attention sparingly (every 4th layer) with a sliding window. The attention score is **phase-aware**:

```
score = Re(query * conjugate(key)) / sqrt(d)
      = (q_real * k_real + q_imag * k_imag) / sqrt(d)
```

This captures both magnitude similarity (how similar) AND phase alignment (how related), where standard attention only captures the former.

Code: `v5/core/attention.py` -- `PhaseAttention`

#### 5. modReLU -- "The activation that doesn't break complex numbers"

Standard activations (ReLU, GELU) only work on real numbers. Applied to complex data, they destroy phase information. V5 uses modReLU:

```
modReLU(z) = ReLU(|z| + bias) * z/|z|
```

- Checks the **magnitude**: is it big enough to pass? (ReLU on magnitude)
- Preserves the **phase**: direction passes through unchanged
- Has a learnable bias that controls the "dead zone" radius

This is mathematically proven to enable universal approximation (2025 theorem) while preserving the algebraic structure of complex numbers.

Code: `v5/core/complex.py` -- `ModReLU`

#### 6. ComplexNorm -- "Normalize magnitude, keep phase"

Like RMSNorm but for complex numbers. Normalizes the distribution of magnitudes across features while preserving phase exactly.

Code: `v5/core/complex.py` -- `ComplexNorm`

---

## Why No FFN Layer?

Standard transformers have: Attention + FFN (feed-forward network) per layer.

V5 has: Banks (CGU) + SSM + optional Attention. **No separate FFN.**

This follows CliffordNet's finding (Jan 2025): when you use algebraically complete operations (like complex multiplication), the interaction is so information-dense that a separate FFN becomes redundant. The CGU in the banks IS the nonlinear transformation -- there's no need for another one.

---

## Quick Start

```bash
# Smoke test (MacBook, CPU/MPS -- tiny model, small data)
python -m v5.train --size tiny --epochs 5 --max_samples 1000

# Reviewer's benchmark (needs GPU -- RTX 4090 or similar)
python -m v5.train --size small --epochs 20 --max_samples 20000 --seq_len 512

# A6000 batch-tuning run (100k samples, small-matched)
./scripts/tune_batch_v5_a6000.sh --max_samples 100000 --batch_size 32 --epochs 10 --seq_len 256

# Resume from checkpoint
python -m v5.train --size small --resume checkpoints_v5/best_model.pt

# Ablation: no banks (isolates SSM contribution)
python -m v5.train --size small --epochs 20 --max_samples 20000 --no_banks

# Ablation: no attention (isolates recurrence)
python -m v5.train --size small --epochs 20 --max_samples 20000 --no_attention
```

**Note on caching**: v5 tokenizes fresh each time (no token cache). The HuggingFace `datasets` library caches the raw TinyStories dataset in `~/.cache/huggingface/datasets/`, but since we slice after loading (`texts[:max_samples]`), changing `max_samples` works correctly. If you suspect cache issues, clear it: `rm -rf ~/.cache/huggingface/datasets/roneneldan___tinystories/`

---

## Config Presets

| Preset | Complex Dim | Real Values | Layers | Banks | State Dim | Heads | Attention | Total Params | Core Params | Use Case |
|--------|-------------|-------------|--------|-------|-----------|-------|-----------|-------------|-------------|----------|
| `tiny` | 64 | 128 | 4 | 2 | 128 | 4 | every 4 | ~7M | ~1.5M | Smoke tests |
| `small-matched` | 128 | 256 | 12 | 2 | 512 | 8 | every 4 | ~28.7M | ~15.8M | Fair baseline comparison |
| `small` | 256 | 512 | 8 | 2 | 512 | 8 | every 4 | ~77M | ~51M | Standard experiments |
| `medium` | 512 | 1024 | 12 | 3 | 1024 | 8 | every 4 | ~260M | ~210M | Serious training |
| `large` | 768 | 1536 | 16 | 3 | 1536 | 12 | every 4 | ~540M | ~460M | Full scale |

Note: "Complex dim 256" means each position is represented by 256 complex numbers = 512 real values. The LM head uses **complex weight tying**: output logits are computed as `Re(z * conj(embed))`, reusing the embedding table. This is both more parameter-efficient and more algebraically consistent than a separate output projection. "Core Params" is banks + SSM + attention only. Exact counts are reported at the start of each training run.

**`small-matched`** is designed for fair comparison with real-valued baselines. With weight tying + reinvested core params: ~28.7M total (embed 12.9M tied, core 15.8M). Uses wider CGU (expand=4) with 2 banks and 12 layers -- favoring richer per-path processing over more parallel paths, consistent with V5's "complex ops do more per parameter" philosophy. Previous versions: v1 untied (31.6M, 5.8M core), v2 tied-only (18.7M, 5.8M core).

---

## Training Output

Training logs to both stdout and a log file (`logs/v5_train_{size}.log`). Each line in the log file is prefixed with a wall-time timestamp.

**Per-batch** (every 50 batches):
```
  [1] batch 50/1202 loss=9.6257 ppl=15149.3 div=0.0000 lr=1.00e-04 | 53.7 samples/s | 13747 tok/s
```

**Per-epoch**: train/val loss, PPL, wall time, then a text generation sample:
```
Epoch 2/10 | Train Loss: 3.3150 PPL: 27.52 | Time: 1359.5s | Val Loss: 3.0460 PPL: 21.03 *best*

Prompt: The quick brown
Generated: The quick brown. They felt very excited. The water were very funny...
```

**Checkpointing**: `best_model.pt` saved on val improvement, `checkpoint_epoch_N.pt` every 5 epochs, `final_model.pt` at end. Use `--resume <path>` to continue training from a checkpoint (appends to existing log).

**Log behavior**:
- Fresh start: overwrites the log file
- `--resume`: appends to the existing log with a separator showing the resume timestamp

### CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--size` | `small` | Model preset: `tiny`, `small`, `small-matched`, `medium`, `large` |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | preset | Override batch size from preset |
| `--lr` | preset | Override learning rate from preset |
| `--max_samples` | `20000` | Max TinyStories samples to load |
| `--seq_len` | `512` | Sequence length for chunking |
| `--num_banks` | preset | Override number of banks |
| `--no_attention` | off | Disable attention layers (ablation) |
| `--no_banks` | off | Disable banks (ablation) |
| `--log_dir` | `logs` | Directory for log files |
| `--checkpoint_dir` | `checkpoints_v5` | Directory for checkpoints |
| `--resume` | none | Path to checkpoint to resume from |

---

## A6000 Server Scripts

All scripts live in `scripts/` and auto-bootstrap the Python environment via `v5_env_setup_a6000.sh`.

| Script | Purpose | Default Size | Notes |
|--------|---------|-------------|-------|
| `tune_batch_v5_a6000.sh` | Find optimal batch size | `small-matched` | Quick runs, no compile |
| `run_v5_medium_a6000.sh` | Full training run | `small-matched` | Uses same size as tune script (batch size transfers) |
| `monitor_training_v5_a6000.sh` | Watch GPU + tail log | -- | `./monitor_training_v5_a6000.sh 5 logs/v5_train_small-matched.log` |

**Important**: Both `tune_batch_v5_a6000.sh` and `run_v5_medium_a6000.sh` use `small-matched` so the batch size you find during tuning will work for the full run. If you want to use `small` or `medium` model size, tune batch size separately for that size.

Extra args pass through to `train.py`, so you can override anything:
```bash
# Tune with 100k samples
./scripts/tune_batch_v5_a6000.sh --max_samples 100000 --batch_size 32 --epochs 10 --seq_len 256

# Full run with batch size from tuning
./scripts/run_v5_medium_a6000.sh --batch_size 32

# Resume interrupted training
./scripts/run_v5_medium_a6000.sh --resume checkpoints_v5/best_model.pt --epochs 20
```

---

## File Structure

```
v5/
  core/
    complex.py      # ComplexLinear, modReLU, ComplexNorm, ComplexGatedUnit, ComplexEmbed
    ssm.py           # ComplexSelectiveSSM with parallel scan
    attention.py     # PhaseAttention (sliding window, complex Q/K/V)
    bank.py          # AlgebraicBank, ComplexRouter, AlgebraicFusion, MultiBank
  model.py           # AlgebraicLM -- wires everything together
  config.py          # V5Config with presets
  train.py           # Training loop (TinyStories, GPT-2 tokenizer, logging, checkpoints)
```

---

## What Changed From V4

| What | V4 (broken) | V5 (fixed) | Why it matters |
|------|-------------|------------|----------------|
| Activation | `GELU(real_part)` | `modReLU(magnitude)` | Phase preserved, not destroyed |
| Gating | `sigmoid(concat(r,i))` | `sigmoid(\|gate\|) * gate/\|gate\|` | Gate both selects AND rotates |
| Backbone | Sequential for-loop | Parallel Blelloch scan | O(log n) vs O(n) depth |
| FFN | Separate FFN layer | None (CGU replaces it) | Less overhead, same expressiveness |
| Memory | Separate memory module | SSM state + sparse attention | Simpler, more effective |
| Banks | Hand-labeled roles | Learned specialization | Let the algebra learn |
| LM Head | Separate `Linear(dim, vocab)` | `Re(z · conj(embed))` tied weights | 40% fewer params, algebraically consistent |
| Naming | "Quantum Phase-Field" | "Algebraic" | Honest about the math |

---

## Training Results (In Progress)

Results from `small-matched` on 100k TinyStories (seq_len=256, batch_size=64, A6000 GPU):

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Time |
|-------|-----------|-----------|----------|---------|------|
| 1 | 4.879 | 131.45 | 3.479 | 32.41 | 1367s |
| 2 | 3.315 | 27.52 | 3.046 | 21.03 | 1360s |
| 3-10 | ... | ... | ... | ... | running |

Throughput: ~14.4k tok/s (~56 samples/s) on A6000.

**Context**: Standard transformers and Mamba at ~30M params on the same data typically reach val PPL ~20-28 by epoch 2. V5 at 21.03 is in the competitive range but not yet demonstrably better. No claims until training completes and baselines are run.

---

## The Hypothesis We're Testing

**Claim**: Complex-valued parameters store more information per parameter than real-valued parameters because of the algebraic structure of complex multiplication (rotation + scaling in one operation).

**Test**: Run V5 against a real-valued baseline with the same architecture but real numbers everywhere. If V5 with fewer real parameters matches or beats the real-valued baseline, the claim is validated.

**Status**: Training in progress. Early results (epoch 2, val PPL 21.03) are in the same ballpark as standard baselines but not yet conclusive. The hypothesis is neither confirmed nor refuted -- we need full training runs and controlled baseline comparisons.

**If it fails**: We honestly acknowledge that complex numbers don't help for language and fall back to real-valued architectures. Science means testing hypotheses, not defending them.
