# V6: Phase-First Language Model with External Memory Layers

An O(n) language model with **zero attention anywhere**. Every token lives in complex phase space, processed through named banks, phase interference coupling, multi-timescale SSM, and a hierarchy of memory systems -- all preserving phase end-to-end.

V6 supports three modes: **autoregressive** text generation, **diffusion text** generation, and **diffusion image** generation -- all sharing the same complex-valued backbone.

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
| Diversity loss | Broken (L1 norm bug) | **Fixed** (correct L2 norm) | Banks actually specialize now |
| Modes | Autoregressive only | **Autoregressive + Diffusion (text & image)** | Shared backbone, multiple generation strategies |
| LR schedule | Cosine (no warmup) | **Warmup + cosine** | Stabilizes early training |
| AMP | Float16 only | **Auto bf16/fp16**, GradScaler only when needed | Better perf on modern GPUs |
| Data pipeline | Raw text, token-level split | **Mojibake repair**, official splits, batched tokenization | Cleaner data, no leakage |
| Optimizer | Single param group | **Decay / no-decay groups** | Phase params, biases, scales excluded from weight decay |

---

## Architecture

### Autoregressive Mode

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

### Diffusion Mode

```
Input --> ComplexEmbed (text tokens) or PatchEncoder/FFTEncoder (images)
  --> ComplexTimestepEmbed (sinusoidal timestep conditioning)
  --> PhaseFieldBackbone (same banks + SSM + memory stack)
  --> OutputProjection (predict x0 or epsilon)
  --> Iterative denoising (DDPM or DDIM sampling)
```

The diffusion path reuses the same `PhaseFieldBackbone` -- banks, couplers, SSM, and memory all operate identically. Only the input encoding and output head differ.

### Named Banks + Phase Interference Coupler

Each layer has a `SemanticBank` and `ContextBank` -- both `ComplexGatedUnit`s with pre-norm. They process the same input through different learned perspectives (meaning vs. structure). A diversity loss penalizes similarity between their outputs to prevent collapse.

The `PhaseInterferenceCoupler` combines bank outputs via:
1. **Learned phase rotations**: each bank's output is rotated by a learned unit-complex vector (near-identity init)
2. **Dynamic routing**: content-dependent weights from magnitude features
3. **Constructive/destructive interference**: outputs at aligned phases reinforce; misaligned phases cancel

### Multi-Timescale SSM

The SSM hidden state is partitioned into three explicit decay tiers:

| Lane | % of state_dim | Decay range | Retention at 1000 steps | Purpose |
|------|---------------|-------------|------------------------|---------|
| **Fast** | 40% | 0.9 -- 0.99 | <1% | Recent tokens, grammar, local syntax |
| **Medium** | 30% | 0.999 -- 0.9999 | 37--90% | Sentence/paragraph coherence |
| **Slow** | 30% | 0.99999+ | 99%+ | Character names, settings, key facts |

### Memory Hierarchy

All memory retrieval uses phase coherence: `score = Re(query * conj(key)) / (|query| * |key|)`. No softmax over sequence length. O(n x num_slots).

| Memory Type | Lifetime | Trainable | Purpose |
|-------------|----------|-----------|---------|
| **Working** | Per-sequence | Write/read projections | Important per-sequence facts |
| **Internal** | Permanent | Keys + values (nn.Parameter) | General language knowledge |
| **Persistent** | Cross-session | No (loaded/saved) | User preferences, patterns |
| **Expert** | Permanent | No (read-only) | Domain expertise |
| **Session** | Per-session | No (optional) | Multi-turn context |

### Diffusion Components

| Component | Description |
|-----------|-------------|
| `ComplexNoiseSchedule` | Cosine or linear schedule for complex-valued tensors |
| `ComplexTimestepEmbed` | Sinusoidal timestep to complex embedding |
| `PatchImageEncoder/Decoder` | ViT-style patches for image diffusion |
| `FFTImageEncoder/Decoder` | 2D FFT to complex coefficients |

---

## Setup

```bash
uv sync                    # Mac / CPU
uv sync --extra cuda       # CUDA machines
```

---

## Quick Start

### Autoregressive Training

```bash
# Smoke test (CPU, tiny model)
python -m v6.train --size tiny --epochs 5 --max_samples 1000 --seq_len 128

# GPU training with full TinyStories
python -m v6.train --size small-matched --max_samples 9999999 --seq_len 256 \
    --compile --compile_mode reduce-overhead --amp_dtype auto --num_workers 4

# RTX 4090 (uses the tuned script)
./scripts/run_v6_rtx4090.sh --batch_size 28

# Resume from checkpoint
python -m v6.train --resume checkpoints_v6/best_model.pt --epochs 20
```

### Diffusion Training

```bash
# Text diffusion
python -m v6.train --mode diffusion_text --size small-matched --epochs 10

# Image diffusion (Tiny ImageNet)
python -m v6.train --mode diffusion_image --image_encoder patch --image_size 64

# Image diffusion (CIFAR-10 with FFT encoder)
python -m v6.train --mode diffusion_image --image_dataset cifar10 --image_encoder fft
```

### Generation

```bash
# Autoregressive text
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt \
    --prompt "Once upon a time"

# With persistent memory
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt \
    --persistent_memory user_alice.pt --prompt "Tell me a story"

# Diffusion text sampling
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --num_samples 4

# Diffusion image sampling
python -m v6.generate --checkpoint checkpoints_v6/best_model.pt --num_samples 16
```

### Ablations

```bash
python -m v6.train --no_working_memory     # disable working memory
python -m v6.train --no_internal_memory    # disable internal memory
./scripts/run_v6_ablation.sh               # run standard ablation suite
```

---

## CLI Arguments

### Model & Data

| Arg | Default | Description |
|-----|---------|-------------|
| `--size` | `small-matched` | Preset: `tiny`, `small`, `small-matched`, `medium` |
| `--epochs` | `20` | Training epochs |
| `--batch_size` | preset | Override batch size |
| `--lr` | preset | Override learning rate |
| `--max_samples` | `20000` | Max TinyStories samples (`9999999` for all) |
| `--seq_len` | `512` | Sequence length |
| `--no_working_memory` | off | Disable working memory |
| `--no_internal_memory` | off | Disable internal memory |
| `--wm_slots` | `0` | Working memory slots |
| `--im_slots` | `0` | Internal memory slots |
| `--init_strategy` | `orthogonal` | Init strategy (13 available) |
| `--init_seed` | auto | Seed for reproducibility |

### Training

| Arg | Default | Description |
|-----|---------|-------------|
| `--lr_schedule` | `warmup_cosine` | `cosine` or `warmup_cosine` |
| `--warmup_steps` | `200` | Linear warmup steps |
| `--dropout` | `0.1` | Dropout rate |
| `--weight_decay` | `0.01` | Weight decay (excluded from phase/bias params) |
| `--no_text_repair` | off | Skip mojibake text repair |
| `--no_cache` | off | Disable token cache |
| `--max_val_samples` | auto | Max validation samples |
| `--gen_every` | `0` | Generate sample every N batches |
| `--gen_prompt` | `Once upon a time` | Generation prompt |
| `--log_interval` | `50` | Batch logging interval |

### CUDA Performance

| Arg | Default | Description |
|-----|---------|-------------|
| `--compile` | off | Enable `torch.compile` |
| `--compile_mode` | `default` | `default`, `reduce-overhead`, `max-autotune` |
| `--fullgraph` | off | `fullgraph=True` for torch.compile |
| `--amp_dtype` | `auto` | `auto` (prefers bf16), `bf16`, `fp16` |
| `--no_tf32` | off | Disable TF32 matmul |
| `--num_workers` | `2` | DataLoader workers |
| `--no_pin_memory` | off | Disable pinned memory |

### Diffusion

| Arg | Default | Description |
|-----|---------|-------------|
| `--mode` | `autoregressive` | `autoregressive`, `diffusion_text`, `diffusion_image` |
| `--diffusion_steps` | `1000` | Noise schedule steps |
| `--noise_schedule` | `cosine` | `cosine` or `linear` |
| `--prediction_target` | `x0` | `x0` or `epsilon` |
| `--sampling_method` | `ddpm` | `ddpm` or `ddim` |
| `--image_size` | `64` | Image resolution |
| `--image_encoder` | `patch` | `patch` or `fft` |
| `--patch_size` | `8` | Patch size for patch encoder |
| `--image_dataset` | `tiny_imagenet` | `tiny_imagenet` or `cifar10` |

---

## Config Presets

| Preset | Complex Dim | Layers | Banks | State Dim | Batch | Use Case |
|--------|-------------|--------|-------|-----------|-------|----------|
| `tiny` | 64 | 4 | 2 | 128 | 16 | Smoke tests, CPU validation |
| `small` | 256 | 8 | 2 | 512 | 8 | Standard experiments |
| `small-matched` | 128 | 12 | 2 | 512 | 8 | Fair V5 comparison (28.7M params) |
| `medium` | 512 | 12 | 2 | 1024 | 4 | Serious training |

Working memory and internal memory are disabled by default (0 slots). Enable with `--wm_slots N` and `--im_slots N`.

---

## File Structure

```
v6/
  core/
    complex.py      # ComplexLinear, ModReLU, ComplexNorm, ComplexGatedUnit, ComplexEmbed
    ssm.py          # ComplexSSM with parallel scan + multi-timescale init
    bank.py         # SemanticBank, ContextBank, NamedBankPair, diversity loss
    coupler.py      # PhaseInterferenceCoupler with learned rotations
    memory.py       # WorkingMemory, InternalMemory, PersistentMemory*, ExpertMemory*, SessionMemory
    attention.py    # PhaseAttention (optional, disabled by default)
    diffusion.py    # ComplexNoiseSchedule, ComplexTimestepEmbed, complex_mse_loss
    image_codec.py  # PatchImageEncoder/Decoder, FFTImageEncoder/Decoder
  init.py           # 13 strategies + multi-timescale SSM + internal memory slot init
  model.py          # PhaseFieldLM (autoregressive) + create_model factory
  backbone.py       # PhaseFieldBackbone (shared by autoregressive + diffusion)
  diffusion_model.py # PhaseFieldDiffusion (text & image diffusion)
  config.py         # V6Config with presets
  train.py          # Training: Trainer + DiffusionTrainer (TinyStories, logging, checkpoints)
  generate.py       # Generation: autoregressive, diffusion text, diffusion image
```

---

## Data Pipeline

TinyStories preprocessing includes:

- **Mojibake repair**: CP1252/Latin-1 round-trip recovery + targeted replacements for common broken quote/apostrophe/dash sequences
- **Official splits**: Uses HuggingFace `train` and `validation` splits (no token-level cut that could split stories)
- **Batched tokenization**: GPT-2 tokenizer with batch encoding
- **Token caching**: Cached to `.cache/v6_tokens/` keyed by sample limit, seq_len, repair flag, and format version. Use `--no_cache` to bypass.

---

## Optimizer

AdamW with parameter-group-aware weight decay:

- **Decayed**: `ComplexLinear` weight matrices, embedding weights, router/projection `nn.Linear` weights
- **No decay**: SSM eigenvalue params (`log_A_real`, `log_A_imag`), `dt_bias`, all biases, `phase_rotations`, `ComplexNorm.scale`, `ModReLU.bias`, scalar gates (`layer_scales`, `bank_scales`), internal memory slots

---

## Training Results

### CPU Smoke Test (2026-03-07)

Model: `tiny` (7.3M params), orthogonal init, 2000 TinyStories samples, seq_len=128, batch_size=4, 5 epochs, Mac CPU.

| Epoch | Train PPL | Val PPL | Time |
|-------|-----------|---------|------|
| 1 | 141.76 | 77.28 | 269s |
| 2 | 52.85 | 54.56 | 266s |
| 3 | 39.27 | 46.12 | 265s |
| 4 | 32.95 | 42.67 | 250s |
| 5 | 30.13 | **42.18** | 260s |

With only 2000 stories on a CPU, the model shows character names, emotional arcs, and dialogue structure -- all without any attention mechanism.

---

## The Hypotheses We're Testing

1. **Long-context coherence without attention**: Can multi-timescale SSM + working memory eliminate text drift?
2. **Learned selective storage**: Does the model learn WHAT to remember (vs. brute-force KV cache)?
3. **Per-user personalization without fine-tuning**: Can persistent memory adapt the model to users?
4. **Phase-native memory**: Does phase-coherence retrieval outperform dot-product retrieval?
5. **Unified backbone for generation strategies**: Can the same phase-field backbone support both autoregressive and diffusion generation?
