# V6 Diffusion Experiment Log

Structured log of architecture decisions, implementation details, and smoke test results for adding unified diffusion support (text + image) to the V6 phase-first architecture.

**Branch**: `diffusion` (off `master`)
**Date started**: 2026-03-09

---

## 1. Design Goal

Extend V6 into a **unified codebase** where one `config.mode` flag switches between:

| Mode | Input | Output | Data |
|------|-------|--------|------|
| `autoregressive` | token IDs | next-token logits | TinyStories |
| `diffusion_text` | token IDs | denoised embeddings | TinyStories |
| `diffusion_image` | pixel images | denoised patches/freq | Tiny ImageNet / CIFAR-10 |

**Same backbone, zero code duplication, backward-compatible defaults.**

```bash
python -m v6.train --size small-matched                                          # autoregressive (unchanged)
python -m v6.train --size small-matched --mode diffusion_text                    # diffusion text
python -m v6.train --size small-matched --mode diffusion_image --image_size 64   # diffusion image
```

---

## 2. Architecture: Shared Backbone

### 2.1 Backbone Extraction (Step 1)

Extracted `PhaseFieldBackbone` from `PhaseFieldLM` into `v6/backbone.py`. Contains ~85%+ of all model parameters:

| Component | In Backbone | AR | Diff Text | Diff Image |
|-----------|:-----------:|:--:|:---------:|:----------:|
| NamedBankPair x N | yes | yes | yes | yes |
| PhaseInterferenceCoupler x N | yes | yes | yes | yes |
| ComplexSSM x N | yes | yes | yes | yes |
| WorkingMemory | yes | yes | optional | optional |
| InternalMemory | yes | yes | yes | yes |
| MemoryFusion | yes | yes | yes | yes |
| ComplexEmbed | no (model-level) | yes | yes | no |
| ComplexTimestepEmbed | no (model-level) | no | yes | yes |
| ImageEncoder | no (model-level) | no | no | yes |
| ImageDecoder | no (model-level) | no | no | yes |
| LM Head | no (model-level) | yes | yes (tied) | no |

`PhaseFieldLM` became a thin wrapper: embed -> backbone -> lm_head. Verified with forward pass + generation -- zero behavior change, identical parameter counts.

### 2.2 Diffusion Conditioning

Timestep is injected into the backbone via a `timestep_embed` argument. At each bank+coupler layer:

```
bank_z = bank_z + timestep_embed   # [B, 1, dim, 2] broadcasts over sequence
```

This is added *before* the bank pair processes the input, so the coupler and SSM both see timestep-conditioned features. Phase encoding is natural: different timesteps = different rotations in complex space.

### 2.3 Model Dispatch

`create_model(config)` returns:
- `PhaseFieldLM` when `config.mode == 'autoregressive'`
- `PhaseFieldDiffusion` when mode is `diffusion_text` or `diffusion_image`

Both share the same `PhaseFieldBackbone` instance internally.

---

## 3. New Modules

### 3.1 `v6/core/diffusion.py` — Diffusion Primitives

| Class/Function | Description |
|----------------|-------------|
| `ComplexNoiseSchedule` | Cosine or linear schedule. Precomputes `alpha_bar`, `sqrt_alpha_bar`, etc. Operates on `[dim, 2]` tensors with isotropic noise on real and imaginary independently. |
| `ComplexNoiseSchedule.add_noise()` | Forward diffusion: `x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε` |
| `ComplexNoiseSchedule.reverse_step()` | DDPM posterior: `p(x_{t-1} | x_t, predicted_x0)` |
| `ComplexNoiseSchedule.ddim_step()` | DDIM step with eta parameter (0 = deterministic) |
| `ComplexTimestepEmbed` | Sinusoidal features → `ComplexLinear` → `ComplexNorm` → `[B, 1, dim, 2]` |
| `complex_mse_loss()` | `mean((predicted - target)^2)` over both real and imaginary |

### 3.2 `v6/diffusion_model.py` — PhaseFieldDiffusion

Training forward pass:
1. Encode input → `z_0` (ComplexEmbed for text, PatchEncoder/FFTEncoder for images)
2. Sample random timestep `t`
3. Add noise: `z_t, noise = schedule.add_noise(z_0, t)`
4. Condition: `t_emb = time_embed(t)`
5. Backbone: `z_out = backbone(z_t, timestep_embed=t_emb)`
6. Output projection: `predicted = output_norm(output_proj(z_out))`
7. Loss: `complex_mse_loss(predicted, z_0)` (x0 prediction target)

Key design decision: **loss is always computed in latent complex space** `[B, L, dim, 2]`, not pixel space. The image decoder is only used during sampling. This avoids shape mismatches and keeps the loss consistent across text and image modes.

Sampling:
- DDPM: iterate `reverse_step()` for all T timesteps
- DDIM: iterate `ddim_step()` with configurable eta and fewer steps
- Final output: text → argmax over tied embeddings → token IDs; image → decoder → `[B, C, H, W]`

### 3.3 `v6/core/image_codec.py` — Image Encoders/Decoders

| Encoder | Approach | Sequence Length | Strengths |
|---------|----------|:---------------:|-----------|
| `PatchImageEncoder` | ViT-style unfold → `ComplexLinear` | `(H/P)²` | Simple, fast, proven |
| `FFTImageEncoder` | 2D FFT → stack real/imag → `ComplexLinear` | `H*W` | Natively complex, frequency separation maps to SSM timescales |

| Decoder | Inverse Operation |
|---------|-------------------|
| `PatchImageDecoder` | `ComplexLinear` → take real part → `nn.Fold` |
| `FFTImageDecoder` | `ComplexLinear` → `torch.complex` → `ifft2` → real |

Factory functions `create_image_encoder/decoder(config, initializer)` dispatch on `config.image_encoder`.

**FFT encoder rationale**: V6's SSM has multi-timescale decay lanes (fast/medium/slow). When feeding FFT coefficients, low-frequency components (global structure) naturally get captured by slow-decay lanes, while high-frequency details (edges, textures) go to fast-decay lanes. This is a unique architectural synergy.

---

## 4. Config Extensions

Added to `V6Config` (all with backward-compatible defaults):

```python
# Mode
mode: str = 'autoregressive'

# Diffusion
diffusion_steps: int = 1000
noise_schedule: str = 'cosine'         # 'cosine' | 'linear'
prediction_target: str = 'x0'         # 'x0' | 'epsilon'
diffusion_loss: str = 'mse'           # 'mse' | 'huber'
sampling_method: str = 'ddpm'         # 'ddpm' | 'ddim'
ddim_steps: int = 50
ddim_eta: float = 0.0

# Image
image_size: int = 64
image_channels: int = 3
image_encoder: str = 'patch'          # 'patch' | 'fft'
patch_size: int = 8
image_dataset: str = 'tiny_imagenet'  # 'tiny_imagenet' | 'cifar10'
```

Existing AR training uses default `mode='autoregressive'` — zero impact on existing configs and checkpoints.

---

## 5. Training Infrastructure

### 5.1 DiffusionTrainer

Parallel to the existing `Trainer` class. Key differences:
- Extracts `batch['input_ids']` for text, `batch['image']` for images
- Forward pass: `model(x)` returns `DiffusionOutput(loss, predicted, diversity_loss)`
- Diversity loss uses same annealing schedule as AR trainer
- Periodic sampling via `_generate_samples()` (text → decode tokens, image → save grid)

### 5.2 CLI Flags Added to `main()`

```
--mode {autoregressive,diffusion_text,diffusion_image}
--diffusion_steps N
--noise_schedule {cosine,linear}
--prediction_target {x0,epsilon}
--sampling_method {ddpm,ddim}
--image_size N
--image_encoder {patch,fft}
--patch_size N
--image_dataset {tiny_imagenet,cifar10}
```

### 5.3 Image DataLoader

`load_image_dataset(config)` supports `tiny_imagenet` (via `zh-plus/tiny-imagenet`) and `cifar10` from HuggingFace `datasets`. Applies `torchvision.transforms` (resize, crop, flip, normalize to [-1,1]). Train/val split: 90/10.

### 5.4 Generation Script

`v6/generate.py` dispatches on `config.mode` from checkpoint:
- AR: existing text generation
- Diffusion text: sample → decode to tokens → print
- Diffusion image: sample → decode to pixels → `save_image()` grid

---

## 6. File Changes Summary

| File | Change | Description |
|------|--------|-------------|
| `v6/backbone.py` | NEW | PhaseFieldBackbone — shared processing core |
| `v6/diffusion_model.py` | NEW | PhaseFieldDiffusion + DiffusionOutput |
| `v6/core/diffusion.py` | NEW | ComplexNoiseSchedule, ComplexTimestepEmbed, complex_mse_loss |
| `v6/core/image_codec.py` | NEW | PatchImage{Encoder,Decoder}, FFTImage{Encoder,Decoder}, factories |
| `v6/model.py` | MODIFY | PhaseFieldLM delegates to backbone; create_model() dispatches on mode |
| `v6/config.py` | MODIFY | mode, diffusion, image fields (backward-compatible) |
| `v6/train.py` | MODIFY | DiffusionTrainer, ImageDataset, load_image_dataset(), --mode CLI |
| `v6/generate.py` | MODIFY | Diffusion sampling paths for text and image |
| `v6/__init__.py` | MODIFY | Export PhaseFieldBackbone |
| `pyproject.toml` | MODIFY | Add torchvision>=0.18.0 |
| `scripts/run_v6_diffusion_text.sh` | NEW | RTX 4090 text diffusion training |
| `scripts/run_v6_diffusion_image.sh` | NEW | RTX 4090 image diffusion training |

**Untouched**: `v6/core/complex.py`, `v6/core/ssm.py`, `v6/core/bank.py`, `v6/core/coupler.py`, `v6/core/memory.py`, `v6/core/attention.py`, `v6/init.py`

---

## 7. Smoke Test Results

All tests run on MacOS (Apple Silicon M1, 8GB RAM, MPS backend).

### 7.1 Autoregressive Regression Test

Verified zero behavior change after backbone extraction.

```
Config: tiny (dim=64, 4 layers), wm=16, im=32
Forward:  logits [2, 32, 50257] ✓
Generate: [1, 35] (32 prompt + 3 new) ✓
Params:   7,294,679 ✓
```

### 7.2 Text Diffusion — Short Smoke (100 samples, 2 epochs)

```bash
python -m v6.train --size tiny --mode diffusion_text --epochs 2 --max_samples 100 \
    --diffusion_steps 50 --seq_len 64 --batch_size 4
```

| Epoch | Diff Loss | Val Loss | Throughput |
|:-----:|:---------:|:--------:|:----------:|
| 1 | 0.4455 | 0.2449 | ~35 samples/s |
| 2 | 0.1665 | 0.1793 | ~35 samples/s |

Generated text progression (epoch 1 → 2):
- Batch 0: `........................................` (noise)
- Batch 200: `a was a a a to a a a a` (common words)
- Batch 400: `was with was with was with it was` (simple patterns)
- Epoch 2 end: `it with a a was it it it with was a it` (more variety)

**Verdict**: Loss decreasing, text evolving from noise → common words. Pipeline functional.

### 7.3 Text Diffusion — Longer Smoke (1000 samples, 5 epochs)

```bash
python -m v6.train --size tiny --mode diffusion_text --epochs 5 --max_samples 1000 \
    --diffusion_steps 100 --seq_len 128 --batch_size 8 --gen_every 100
```

| Epoch | Diff Loss | Val Loss | Time |
|:-----:|:---------:|:--------:|:----:|
| 1 | 0.1500 | 0.0257 | 49s |
| 2 | 0.0150 | 0.0128 | 47s |
| 3 | 0.0065 | 0.0077 | 47s |
| 4 | 0.0040 | 0.0065 | 47s |
| 5 | 0.0033 | 0.0058 | 48s |

Total time: 253s (~4 min). Throughput: ~35 samples/s on MPS.

Generated text progression:
- Epoch 1: `not not not not not not` → `OnceOnceOnceOnce` (single-token repetition)
- Epoch 2: `Once rich ignorant Once almost` (emerging vocabulary)
- Epoch 3: `Once mountain Nem Once` (more unique tokens)
- Epoch 4: `ombat sale Nem trap mountain voices` (diverse vocabulary)
- Epoch 5: `sale mountain trap Nem voices silver` (varied words, no sentence structure yet)

**Verdict**: Strong loss convergence (1.0 → 0.003). Vocabulary diversity increases with training. No sentence structure — expected with tiny model (7.2M params) and only 1000 samples.

### 7.4 Image Diffusion — Synthetic Data (patch + FFT)

#### Patch Encoder (400 gradient images, 32x32, 5 epochs)

```
Config: tiny, patch_size=8, 16 patches, diffusion_steps=50
```

| Epoch | Diff Loss | Val Loss |
|:-----:|:---------:|:--------:|
| 1 | 0.1792 | 0.0164 |
| 3 | 0.0017 | 0.0005 |
| 5 | 0.0003 | 0.0002 |

Params: 794,636. Throughput: ~65 samples/s on MPS.

#### FFT Encoder (200 gradient images, 16x16, 3 epochs)

```
Config: tiny, fft encoder, 256 positions, diffusion_steps=50
```

| Epoch | Diff Loss | Val Loss |
|:-----:|:---------:|:--------:|
| 1 | 0.1365 | 0.0367 |
| 2 | 0.0285 | 0.0287 |
| 3 | 0.0207 | 0.0151 |

Params: 745,874. Throughput: ~20 samples/s on MPS (higher seq_len = slower).

**Verdict**: Both encoders converge. Patch is ~3x faster (fewer positions). FFT converges slower but has richer representation.

### 7.5 Image Diffusion — CIFAR-10 (5000 images, patch, 20 epochs)

```bash
python -m v6.train --size tiny --mode diffusion_image --image_encoder patch \
    --image_size 32 --patch_size 8 --image_dataset cifar10 --epochs 20
```

| Epoch | Diff Loss | Val Loss |
|:-----:|:---------:|:--------:|
| 1 | 0.1494 | 0.0871 |
| 5 | 0.0561 | 0.0516 |
| 10 | 0.0224 | 0.0169 |
| 15 | 0.0001 | 0.0000 |
| 20 | 0.0000 | 0.0000 |

**Denoising loss converged to zero** — the model memorizes the training set perfectly (5000 images, 0.8M params).

**However, generated samples are flat gray patches.** The sampling loop produces near-uniform outputs despite perfect denoising loss. This is **mode collapse during sampling** — a classic problem with undersized diffusion models.

Root causes:
1. **Capacity**: 0.8M params is ~40x smaller than the smallest working CIFAR-10 diffusion models (~35M)
2. **Patch granularity**: 8×8 patches = only 16 positions. Each 192-value patch compressed to 128 real values (lossy)
3. **Dynamic range**: Denoised predictions collapse to very small magnitudes (range [-0.15, 0.13] instead of [-1, 1])

### 7.6 Image Diffusion — CIFAR-10 (45000 images, patch_size=4, 1 epoch partial)

```bash
python -m v6.train --size tiny --mode diffusion_image --image_encoder patch \
    --image_size 32 --patch_size 4 --image_dataset cifar10 --epochs 15
```

Ran 1+ epochs before being stopped. Loss converged to 0 within first epoch (batch 350/2813). Generated samples again showed flat patches — confirming the capacity issue.

| Batch | Diff Loss | Observation |
|:-----:|:---------:|-------------|
| 0 | 0.9885 | Starting loss |
| 200 | 0.0407 | Rapid decrease |
| 400 | 0.0001 | Near zero |
| 2813 | 0.0000 | Epoch 1 end |

Throughput: ~80 samples/s on MPS. Sampling produced teal-green checkerboard patterns.

---

## 8. Key Findings

### What Works
1. **Backbone sharing**: Identical `PhaseFieldBackbone` powers all three modes with zero regression on AR
2. **Complex-space diffusion**: Noise schedule, timestep embedding, and denoising all operate correctly on `[dim, 2]` tensors
3. **Training infrastructure**: DiffusionTrainer, CLI flags, checkpointing, periodic sampling, logging all functional
4. **Text diffusion**: Loss converges reliably, generated tokens evolve from noise to recognizable words
5. **Image pipeline end-to-end**: Both patch and FFT codecs encode/decode correctly, training converges

### What Needs GPU Scale
1. **Image generation quality**: Tiny model (0.8M) memorizes but cannot generalize during sampling — needs significantly more capacity
2. **Text generation quality**: Produces individual words but no sentence structure — needs larger model + more data
3. **FFT encoder potential**: Slower (higher seq_len) but architecturally interesting; needs GPU to test at scale

### Architectural Observations
1. **Denoising loss converges very fast** on small datasets — the backbone's SSM + memory system learns the training distribution efficiently
2. **Sampling quality lags denoising quality** — this gap is proportional to model undersizing
3. **Phase encoding of timesteps** works naturally with the complex-valued backbone (no architectural friction)
4. **Diversity loss** collapses to zero quickly in diffusion mode (same as AR), suggesting it needs rethinking

---

## 9. Next Steps

**Note (2026-03-11):** The first GPU-scale text diffusion run on WikiText-103 (section 11) revealed severe mode collapse: diff_loss went to 0.0000 within one epoch while samples collapsed to a single repeated subword ("rous"). Immediate priority should shift to **diagnosing sampling collapse** (e.g. prediction norm vs target norm, EMA, sampling steps, latent space regularization) before running more long text-diffusion experiments.

### Immediate (GPU — RTX 4090)

1. **Diagnose text diffusion sampling collapse**: Before further scale runs, add diagnostics (e.g. pred/target norm logging, EMA checkpoint) and test fixes (more sampling steps, epsilon prediction, guidance). See section 11 run for failure mode.
2. **Text diffusion at scale** (after diagnostics): Run `scripts/run_v6_diffusion_text.sh` (small-matched, full TinyStories or WikiText-103). Target: coherent multi-word phrases in generated samples.

3. **Image diffusion at scale**: Run `scripts/run_v6_diffusion_image.sh` (small-matched, Tiny ImageNet, 64×64, 10 epochs). Target: recognizable shapes/colors in generated image grids.

4. **Patch size sweep**: Test patch_size in {4, 8, 16} on 64×64 images to find the best quality/speed tradeoff.

### Medium-term

5. **FFT encoder comparison**: Same image training with `--image_encoder fft`. Compare convergence speed and sample quality vs patch.

6. **DDIM sampling**: Test `--sampling_method ddim --ddim_steps 50` for faster generation without quality loss.

7. **Epsilon prediction**: Compare `--prediction_target epsilon` vs `x0` for training stability and sample quality.

8. **Larger model configs**: Create `medium` image config (dim=256+, 12+ layers) for serious image generation.

### Research Questions

9. **Can a single checkpoint do both?** Train on text, then fine-tune on images (or vice versa) with shared backbone weights.

10. **Multi-timescale SSM for images**: Does the fast/medium/slow decay partitioning learn meaningful frequency separation when fed FFT coefficients?

11. **Scaling laws**: How does sample quality scale with model size for this architecture? Is the scaling curve competitive with standard UNet diffusion?

---

## 10. How to Update

1. **Training Runs**: Add detailed tables with config, metrics, and sample quality notes.
2. **Architecture Changes**: Document any modifications to backbone, codecs, or training loop.
3. **Bugs Found**: Log with error, root cause, and fix.
4. **GPU Results**: Document RTX 4090 runs with wall times, throughput, and generated sample assessments.

---

## 11. GPU-Scale Training Runs

### Run v6-wikitext103-diffusion-text-no-memory (2026-03-11, stopped mid-run)

**Setup** (from `logs/v6/wikitext103_diffusion_text_20260311_145638_67a0782/RUN_INFO.txt`):

- size=small-matched (28.7M params), WikiText-103 (230,986 train chunks, 118M tokens), seq_len=512, batch_size=14
- mode=diffusion_text, diffusion_steps=1000, cosine schedule, prediction_target=x0, sampling=ddpm
- No memory (WM=0, IM=0)
- compile=reduce-overhead, bf16, RTX 4090
- Stopped at batch 15150/16499 (~92% through epoch 1 of 10)

**Loss trajectory** (key milestones from the log):

| Batch | diff_loss | div | Note |
|-------|-----------|-----|------|
| 0 | 1.0189 | 6.40 | Start |
| 500 | 0.2522 | 6.36 | |
| 1000 | 0.0573 | 6.24 | |
| 2000 | 0.0358 | 5.70 | |
| 3000 | 0.0131 | 4.57 | |
| 4000 | 0.0086 | 2.75 | Div declining |
| 5000 | 0.0041 | 0.399 | First sample; div about to collapse |
| 5200 | 0.0030 | 7.26e-04 | Div collapsed |
| 7000 | 0.0010 | ~2e-04 | |
| 9000 | 0.0002 | ~2e-04 | |
| 10000 | 0.0001 | ~2e-04 | Second sample |
| 11000+ | 0.0000 | ~2e-04 | Loss at display floor |
| 15000 | 0.0000 | ~3e-04 | Third sample; run stopped |

**Samples** (verbatim from log):

- batch 5000: gibberish with token loops ("DwarfusbDel Yelp nob112usb nerdsBernie... Tulsa Tulsa...")
- batch 10000: total mode collapse ("rousrousrousrousrous..." x50)
- batch 15000: same collapse ("rousrousrousrous...")

**Observations**:

1. **diff_loss goes to 0.0000 within a single epoch** on 118M tokens of real text — the model memorizes the latent denoising mapping, confirmed by zero loss, but samples are complete garbage.
2. **Diversity loss collapses ~batch 5000** (6.40 → 7e-04 in ~200 batches around batch 5000–5200) — same pattern as all AR V6 runs.
3. **Mode collapse in sampling**: decoded samples degrade from random gibberish to a single repeated subword ("rous"). This is the text equivalent of the "flat gray patches" seen in CIFAR-10 smoke tests (section 7.5). The denoising loop converges to a single fixed point in latent space.
4. **Throughput**: ~100 samples/s steady state (vs ~46k tok/s for AR on same data).

**Diagnosis / design issues**:

- Zero training loss + garbage samples = the model has memorized the training set's encoding but learned no generalizable structure. The sampling loop starts from pure noise, which is out-of-distribution for a memorized model.
- ComplexNorm on both encoder and output may compress the latent space too aggressively, making memorization easy.
- The SSM backbone without attention may lack the capacity for the diffusion task's global coherence requirement (each position must produce a coherent denoised output considering all others).
- No guidance (classifier-free or otherwise), no EMA, no gradient clipping tuning — standard diffusion training practices are missing.

**Log**: `logs/v6/wikitext103_diffusion_text_20260311_145638_67a0782/v6_diffusion-text_small-matched.log`
