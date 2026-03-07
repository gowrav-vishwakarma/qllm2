# V6: Phase-First Language Model with External Memory Layers

An O(n) language model with **zero attention anywhere**. Every token lives in complex phase space, processed through named banks, phase interference coupling, multi-timescale SSM, and a hierarchy of memory systems -- all preserving phase end-to-end.

V6 combines V4's architectural novelty (named banks, interference coupling, associative memory) with V5's mathematically correct phase-preserving operations (ModReLU, ComplexGatedUnit, ComplexNorm). The result is a model that is both novel and sound.

---

## What's New in V6 (vs V5)

| What | V5 | V6 | Why |
|------|----|----|-----|
| Attention | Sparse PhaseAttention every K layers | **None** | Achieves O(n) complexity; SSM + working memory handles coherence |
| Banks | Generic `AlgebraicBank` | **Named** `SemanticBank` + `ContextBank` | V4-style specialization, encouraged via diversity loss |
| Bank coupling | `AlgebraicFusion` with real router | **PhaseInterferenceCoupler** with learned phase rotations | Genuine wave interference, not just weighted averaging |
| SSM eigenvalues | Uniform `linspace(log(0.95), log(0.999))` | **Multi-timescale**: fast/medium/slow decay lanes | Slow lanes retain info across thousands of tokens |
| Working memory | None | **Learned write/read scratchpad** (per-sequence, non-decaying) | Stores important facts (names, settings) without decay |
| Internal memory | None | **Trained nn.Parameter slots** (general knowledge) | Captures language patterns learned during training |
| Persistent memory | None | **Per-user, cross-session** (saveable/loadable tensors) | Personalization without fine-tuning |
| Expert memory | None | **Shared, read-only** domain knowledge | Stackable expertise (coding, medical, etc.) |
| Session memory | None | **Optional** between-turn buffer (`--session_memory`) | Multi-turn context, disabled by default |
| Diversity loss | Broken (L1 norm bug) | **Fixed** (correct L2 norm) | Banks actually specialize now |

---

## Architecture

```
Tokens --> ComplexEmbed
  --> [SemanticBank + ContextBank --> PhaseInterferenceCoupler] x N layers
  --> MultiTimescaleSSM (fast + medium + slow decay lanes)
  --> WorkingMemory (learned write/read, non-decaying per-sequence)
  --> InternalMemory (trained slots, general knowledge)
  --> [PersistentMemory] (optional, per-user)
  --> [ExpertMemory] (optional, shared)
  --> MemoryFusion (learned complex mixing)
  --> TiedComplexLMHead
```

### Named Banks + Phase Interference Coupler

Each layer has a `SemanticBank` and `ContextBank` -- both `ComplexGatedUnit`s with pre-norm. They process the same input through different learned perspectives (meaning vs. structure). A diversity loss penalizes similarity between their outputs to prevent collapse.

The `PhaseInterferenceCoupler` combines bank outputs via:
1. **Learned phase rotations**: each bank's output is rotated by a learned unit-complex vector (near-identity init)
2. **Dynamic routing**: content-dependent weights from magnitude features
3. **Constructive/destructive interference**: outputs at aligned phases reinforce; misaligned phases cancel

This is genuine wave interference -- not just a weighted average.

### Multi-Timescale SSM

The SSM hidden state is partitioned into three explicit decay tiers:

| Lane | % of state_dim | Decay range | Retention at 1000 steps | Purpose |
|------|---------------|-------------|------------------------|---------|
| **Fast** | 40% | 0.9 -- 0.99 | <1% | Recent tokens, grammar, local syntax |
| **Medium** | 30% | 0.999 -- 0.9999 | 37--90% | Sentence/paragraph coherence |
| **Slow** | 30% | 0.99999+ | 99%+ | Character names, settings, key facts |

The `dt_proj` selectivity learns to route important information (names, settings) to slow lanes and transient information (function words) to fast lanes.

### Memory Hierarchy

All memory retrieval uses the same mechanism: **phase coherence**.

```
score = Re(query * conj(key)) / (|query| * |key|)
```

No softmax over sequence length. No token-token comparison. O(n × num_slots).

| Memory Type | Lifetime | Trainable | Storage | Purpose |
|-------------|----------|-----------|---------|---------|
| **Working** | Per-sequence | Write/read projections (nn.Module) | Runtime tensors, reset each sequence | Important per-sequence facts |
| **Internal** | Permanent | Keys + values (nn.Parameter) | Part of model weights | General language knowledge |
| **Persistent** | Cross-session | No (loaded/saved) | Separate .pt files per user | User preferences, patterns |
| **Expert** | Permanent | No (read-only) | Shared .pt files | Domain expertise |
| **Session** | Per-session | No (optional) | Between-turn buffer | Multi-turn context |

#### Working Memory in Detail

The key innovation: a **differentiable, per-sequence scratchpad** that gives the model non-decaying storage within the forward pass.

- **Write gate**: learned `ComplexLinear` projection decides WHEN to store (bias initialized negative for selectivity)
- **Write key/value**: learned projections decide WHAT to store
- **Read query**: learned projection decides HOW to retrieve
- **Soft addressing**: no in-place ops; gradients flow through write decisions back to the tokens that triggered them

During training at seq_len=128: if the model fails to predict token 100 because it forgot "forest" from token 5, gradients flow backward through retrieval → slots → write decision at token 5. The model learns what to store.

During inference beyond training seq_len: the read mechanism is distance-agnostic. Phase coherence depends on content, not position. A slot written at step 5 is equally retrievable at step 50 or step 5000.

### Multi-User Serving

One base model serves many users. External memories are just tensors on the same device:

```python
model = V6Model.load("v6_base.pt")  # shared, on GPU

user_memories = {
    "alice": load_persistent_memory("alice_memory.pt"),
    "bob": load_persistent_memory("bob_memory.pt"),
}
expert_memories = {"coding": load_expert_memory("expert_coding.pt")}

output = model.forward(
    input_ids=alice_tokens,
    persistent_memory=user_memories["alice"],
    expert_memory=expert_memories["coding"],
)
```

Memory cost per user: ~1 MB (512 slots × 128 dim × complex = 512 KB per memory layer). 1000 concurrent users ≈ 1 GB overhead.

---

## Phase-Safe Math Rules

V6 enforces strict phase preservation throughout:

- All projections use `ComplexLinear` (never real `nn.Linear` on complex data)
- All activations use `ModReLU` (never GELU/ReLU on real parts)
- All normalization uses `ComplexNorm` (preserves phase exactly)
- No `F.gelu(h[..., 0]).unsqueeze(-1) * h` anywhere
- No attention (no PhaseAttention, no SDPA, no score matrices)

**Acceptable real-valued bridges**: SSM `dt_proj` selectivity, LM head to vocab logits, loss computation, router magnitude features, residual scales.

---

## Setup

From the project root:

```bash
uv sync                    # Mac / CPU
uv sync --extra cuda       # CUDA machines
```

---

## Quick Start

```bash
# Smoke test (CPU, tiny model)
python -m v6.train --size tiny --epochs 5 --max_samples 1000 --seq_len 128

# Standard training (GPU)
python -m v6.train --size small-matched --epochs 10 --max_samples 100000 --seq_len 256

# Ablation: no working memory
python -m v6.train --size small-matched --no_working_memory

# Ablation: no internal memory
python -m v6.train --size small-matched --no_internal_memory

# Resume from checkpoint
python -m v6.train --size small-matched --resume checkpoints_v6/best_model.pt

# Generate text
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --prompt "Once upon a time"

# Generate with persistent memory
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt \
    --persistent_memory user_alice.pt --prompt "Tell me a story"
```

---

## Config Presets

| Preset | Complex Dim | Layers | Banks | State Dim | WM Slots | IM Slots | Batch | Use Case |
|--------|-------------|--------|-------|-----------|----------|----------|-------|----------|
| `tiny` | 64 | 4 | 2 | 128 | 16 | 32 | 16 | Smoke tests, CPU validation |
| `small` | 256 | 8 | 2 | 512 | 64 | 128 | 8 | Standard experiments |
| `small-matched` | 128 | 12 | 2 | 512 | 64 | 128 | 8 | Fair V5 comparison |
| `medium` | 512 | 12 | 2 | 1024 | 128 | 256 | 4 | Serious training |

---

## Initialization

V6 inherits all 13 init strategies from V5 and extends them:

| V6 Module | Init Method | Notes |
|-----------|-------------|-------|
| ComplexEmbed | `init_embedding()` | Unchanged from V5 |
| SemanticBank / ContextBank | `init_complex_linear()` | Unchanged from V5 |
| Coupler phase rotations | `init_phase_rotation()` | Near-identity |
| SSM eigenvalues | `init_ssm_eigenvalues_multiscale()` | **New**: tiered fast/medium/slow |
| Working memory projections | `init_complex_linear()` | Orthogonal default |
| Working memory write gate bias | Custom: -2.0 | Start selective |
| Internal memory keys | `init_internal_memory_slots()` | **New**: phase-spread |
| Internal memory values | `init_internal_memory_slots()` | **New**: small random |
| MemoryFusion | `init_complex_linear()` | Unchanged from V5 |

Default: `--init_strategy orthogonal --init_seed 42`

---

## CLI Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--size` | `small-matched` | Model preset |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | preset | Override batch size |
| `--lr` | preset | Override learning rate |
| `--max_samples` | `20000` | Max TinyStories samples |
| `--seq_len` | `512` | Sequence length |
| `--no_working_memory` | off | Disable working memory (ablation) |
| `--no_internal_memory` | off | Disable internal memory (ablation) |
| `--init_strategy` | `orthogonal` | Init strategy |
| `--init_seed` | auto | Seed for reproducibility |
| `--checkpoint_dir` | `checkpoints_v6` | Checkpoint directory |
| `--resume` | none | Resume from checkpoint |

---

## File Structure

```
v6/
  core/
    complex.py      # ComplexLinear, ModReLU, ComplexNorm, ComplexGatedUnit, ComplexEmbed
    ssm.py           # ComplexSSM with parallel scan + multi-timescale init
    bank.py          # SemanticBank, ContextBank, NamedBankPair, diversity loss
    coupler.py       # PhaseInterferenceCoupler with learned rotations
    memory.py        # WorkingMemory, InternalMemory, PersistentMemory*, SessionMemory, ExpertMemory*, MemoryAdaptation
  init.py            # 13 strategies + multi-timescale SSM + internal memory slot init
  model.py           # PhaseFieldLM -- wires everything together
  config.py          # V6Config with presets
  train.py           # Training loop (TinyStories, logging, checkpoints, ablation flags)
  generate.py        # Text generation with optional persistent/expert memory
```

---

## Training Results

### CPU Smoke Test (2026-03-07)

Model: `tiny` (7.3M params), orthogonal init, 2000 TinyStories samples, seq_len=128, batch_size=4, 5 epochs, Mac CPU.

| Epoch | Train PPL | Val PPL | Time | Notes |
|-------|-----------|---------|------|-------|
| 1 | 141.76 | 77.28 | 269s | PPL drops from 51K to 77 |
| 2 | 52.85 | 54.56 | 266s | |
| 3 | 39.27 | 46.12 | 265s | |
| 4 | 32.95 | 42.67 | 250s | |
| 5 | 30.13 | **42.18** | 260s | Best val, 23 min total |

Diversity loss converged properly: 1.60 → 0.0006 (L2 norm fix working).

**Sample generation (epoch 5, prompt: "The quick brown")**:

> *"The quick brown and soon, feeling scared at her mommy. When Joe were two of other friends, and started to cry and his They liked the ball for the best time! She ran to help him, but soon no, and her toys were playing with his special. He was happy that they would never made about to go back."*

With only 2000 stories on a CPU, the model shows character names, emotional arcs, and dialogue structure -- all without any attention mechanism.

---

## What Changed From V4 and V5

| What | V4 | V5 | V6 |
|------|----|----|-----|
| Phase math | Broken (GELU destroys phase) | Fixed (ModReLU preserves phase) | Fixed (same as V5) |
| Banks | Named (Semantic + Context) | Generic (AlgebraicBank) | **Named again** (revived from V4, V5-safe ops) |
| Coupling | InterferenceCoupler | AlgebraicFusion | **PhaseInterferenceCoupler** (revived, V5-safe) |
| Memory | PhaseAssociativeMemory | None (SSM state only) | **Full hierarchy** (working + internal + persistent + expert) |
| Attention | None | Sparse (every K layers) | **None** (O(n) end-to-end) |
| SSM timescales | Uniform | Uniform | **Multi-timescale** (fast/medium/slow lanes) |
| Diversity loss | Not measured | Broken (L1 norm bug) | **Fixed** (correct L2 norm) |
| Complexity | O(n) | O(n) + O(n²) sparse attention | **O(n)** strictly |

---

## The Hypotheses We're Testing

1. **Long-context coherence without attention**: Can multi-timescale SSM + working memory eliminate text drift?
2. **Learned selective storage**: Does the model learn WHAT to remember (vs. brute-force KV cache)?
3. **Per-user personalization without fine-tuning**: Can persistent memory adapt the model to users?
4. **Phase-native memory**: Does phase-coherence retrieval outperform dot-product retrieval?

**Status**: CPU smoke test confirms the architecture learns (val PPL 42 from 51K in 23 minutes). GPU training with more data needed for meaningful comparisons against V5.

**What we still need**: (1) matched GPU run against V5 no-attention baseline, (2) working memory ablation, (3) long-context coherence evaluation (500+ token generation), (4) persistent memory save/load round-trip in practice.
