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
         Each token            N layers of processing       Back to
        becomes a               (all in complex space)      real numbers
       complex vector                                      for prediction
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

# Ablation: no banks (isolates SSM contribution)
python -m v5.train --size small --epochs 20 --max_samples 20000 --no_banks

# Ablation: no attention (isolates recurrence)
python -m v5.train --size small --epochs 20 --max_samples 20000 --no_attention
```

---

## Config Presets

| Preset | Complex Dim | Layers | Banks | State Dim | Attention | Use Case |
|--------|-------------|--------|-------|-----------|-----------|----------|
| `tiny` | 64 | 4 | 2 | 128 | every 4 | Smoke tests |
| `small` | 256 | 8 | 2 | 512 | every 4 | Experiments |
| `small-matched` | 128 | 8 | 2 | 256 | every 4 | Match ~8M baseline |
| `medium` | 512 | 12 | 3 | 1024 | every 4 | Serious training |

Note: "Complex dim 256" means each position is represented by 256 complex numbers = 512 real values. The parameter count is higher than a real-valued model with dim=256, but each parameter does more work.

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
  train.py           # Training loop (TinyStories, GPT-2 tokenizer)
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
| Naming | "Quantum Phase-Field" | "Algebraic" | Honest about the math |

---

## The Hypothesis We're Testing

**Claim**: Complex-valued parameters store more information per parameter than real-valued parameters because of the algebraic structure of complex multiplication (rotation + scaling in one operation).

**Test**: Run V5 against a real-valued baseline with the same architecture but real numbers everywhere. If V5 with fewer real parameters matches or beats the real-valued baseline, the claim is validated.

**If it fails**: We honestly acknowledge that complex numbers don't help for language and fall back to real-valued architectures. Science means testing hypotheses, not defending them.
