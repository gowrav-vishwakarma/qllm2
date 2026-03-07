# V5 Experiment Log

Structured log of config changes, training runs, and results. Update this file after each significant run.

---

## 1. Change Log

| Date | Change | Why | Run |
|------|--------|-----|-----|
| 2026-03-04 | **Weight tying**: Removed `output_proj`, compute logits as `Re(z * conj(embed))` | Standard practice (GPT-2, LLaMA, Mamba). Saves 12.9M params, algebraically consistent. | v2-tied |
| 2026-03-04 | **Reinvested core params**: 12 layers, expand=4, state_dim=512, 8 heads | Put freed params into core model. Wider CGU over more banks -- "let the algebra do more per path". | v3-core-heavy |
| 2026-03-05 | **Default init changed to `orthogonal`**; removed 8 broken/inferior strategies | Benchmark (Run C) showed orthogonal 2x better than random (168 vs 349 PPL). Backed by theory (norm-preserving isometry). | -- |
| 2026-03-05 | **First orthogonal full run**: orthogonal init (seed=42) on RTX 4090, batch_size=16 | A/B test (Run D) confirmed orthogonal 31% better at epoch 10. First full 10-epoch run with new default init. | v4-ortho |
| 2026-03-05 | **Full-dataset run started**: switched `max_samples` from 100k to 9,999,999 (full TinyStories, 2.1M texts, 474M tokens) | Scale up after 100k runs showed strong signal. Same architecture and init as v4-ortho. | v5-full-ds |

---

## 2. Run Summary Table

| Run | Config | Total | Core | Layers | Banks | Expand | State | Batch | Epochs | Ep1 Val PPL | Best Val PPL | tok/s | GPU | Notes |
|-----|--------|-------|------|--------|-------|--------|-------|-------|--------|-------------|-------------|-------|-----|-------|
| v1-untied | small-matched (orig) | 31.6M | 5.8M (18%) | 8 | 2 | 2 | 256 | 64 | 10 | 32.41 | 21.03 (ep2) | ~14.4k | A6000 | Baseline, no weight tying |
| v2-tied | small-matched | 18.7M | 5.8M (31%) | 8 | 2 | 2 | 256 | 64 | 10 | partial | -- | ~14.4k | A6000 | Weight tying only; slower early convergence |
| v3-core-heavy | small-matched | 28.7M | 15.8M (55%) | 12 | 2 | 4 | 512 | 32 | 1 | 66.64 | 66.64 | ~6.2k | A6000 | Tune/stability run; 1-epoch cosine undertrains and is not quality-comparable to 10-epoch runs |
| v3-full | small-matched | 28.7M | 15.8M (55%) | 12 | 2 | 4 | 512 | 32 | 10 | 38.99 | 11.77 (ep10) | ~6.2k | A6000 | Random init, 100k samples, 9.03h |
| v4-ortho | small-matched | 28.7M | 15.8M (55%) | 12 | 2 | 4 | 512 | 16 | 10 | 18.88 | 8.00 (ep10) | ~16.1k | 4090 | Orthogonal init (seed=42), 3.48h. Best 100k-sample result. |
| v5-full-ds | small-matched | 28.7M | 15.8M (55%) | 12 | 2 | 4 | 512 | 16 | 10 | 6.27 | 5.59 (ep3, in progress) | ~16k | 4090 | Full TinyStories (474M tokens), orthogonal init (seed=42). Ep3 already beats v4-ortho ep10. |

**Data (default unless overridden)**: TinyStories + GPT-2 tokenizer.

**GPU note**: v1 through v3-full ran on A6000 (48GB). v4-ortho ran on RTX 4090 (24GB). Throughput numbers are NOT comparable across GPUs. PPL comparisons remain valid (same model, data, epochs).

---

## 3. Chronological Run Registry (Source of Truth)

Use this table for exact reproducibility: what architecture was used, what data/setup it ran with, and what result it produced.

| Run | Wall Clock Start | Commit | GPU | Architecture Snapshot | Param Snapshot | Run Setup | Result Snapshot | Status |
|-----|------------------|--------|-----|-----------------------|----------------|-----------|-----------------|--------|
| v1-untied | 2026-03-04 10:13 | pre-bdf6b87 | A6000 | `dim=128, state=256, layers=8, banks=2, expand=2, heads=4, attn_every=4, tied_output=no` | total=31,556,184; core=5,791,448 (18%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=64, epochs=10, batches/epoch=1202 | epoch1 val_ppl=32.41, epoch2 val_ppl=21.03, throughput~14.4k tok/s | baseline (primary reference) |
| v2-tied | 2026-03-04 11:31 | pre-bdf6b87 | A6000 | `dim=128, state=256, layers=8, banks=2, expand=2, heads=4, attn_every=4, tied_output=yes` | total=18,690,392; core=5,824,600 (31%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=64, epochs=10, batches/epoch=1202 | early epoch1 lagged ~0.5 loss vs v1 at same batch index | partial/incomplete |
| v3-core-heavy | 2026-03-04 12:04 | pre-bdf6b87 | A6000 | `dim=128, state=512, layers=12, banks=2, expand=4, heads=8, attn_every=4, tied_output=yes` | total=28,682,372; core=15,816,580 (55%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=32, epochs=1, batches/epoch=2403 | epoch1 val_ppl=66.64, best_val_ppl=66.64, throughput~6.2k tok/s | tune/stability only (not quality-comparable) |
| v3-full | 2026-03-04 15:17 | pre-bdf6b87 | A6000 | `dim=128, state=512, layers=12, banks=2, expand=4, heads=8, attn_every=4, tied_output=yes` | total=28,682,372; core=15,816,580 (55%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=32, epochs=10, batches/epoch=2403 | epoch1 val_ppl=38.99, epoch10 val_ppl=11.77, throughput~6.2k tok/s, 9.03h | complete |
| v4-ortho | 2026-03-05 15:25 | e24acd2 | 4090 | `dim=128, state=512, layers=12, banks=2, expand=4, heads=8, attn_every=4, tied_output=yes, init_strategy=orthogonal, init_seed=42` | total=28,682,372; core=15,816,580 (55%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=16, epochs=10, batches/epoch=4806 | epoch1 val_ppl=18.88, epoch10 val_ppl=8.00, throughput~16.1k tok/s, 3.48h | complete |
| v5-full-ds | 2026-03-05 19:18 | 0467675 | 4090 | `dim=128, state=512, layers=12, banks=2, expand=4, heads=8, attn_every=4, tied_output=yes, init_strategy=orthogonal, init_seed=42` | total=28,682,372; core=15,816,580 (55%) | dataset=TinyStories (full), max_samples=9999999, texts=2,119,489, tokens=473,992,006, seq_len=256, batch=16, epochs=10, batches/epoch=103744 | epoch1 val_ppl=6.27, epoch2 val_ppl=5.81, epoch3 val_ppl=5.59, throughput~16k tok/s, ~7.2h/epoch | in progress (epoch 3/10) |

**Comparison note**: v3-core-heavy used `epochs=1`, so cosine LR decayed to ~0 within one epoch; this makes quality metrics non-comparable to 10-epoch runs.

---

## 4. Per-Sample Learning Curves

Aligned by **samples seen** (not batch index), since batch sizes differ. v1/v2 use batch=64 (A6000), v3-core-heavy uses batch=32 (A6000), v4-ortho uses batch=16 (4090).

| Samples | v1-untied (batch=64) | v2-tied (batch=64) | v3-core-heavy (batch=32) | v4-ortho (batch=16) |
|---------|----------------------|--------------------|---------------------------|----------------------|
| ~3,200 | batch 50: loss=9.63 | batch 50: loss=9.91 | batch 100: loss=8.75 | batch 200: loss=6.24 |
| ~6,400 | batch 100: loss=8.08 | batch 100: loss=8.66 | batch 200: loss=6.97 | batch 400: loss=4.83 |
| ~9,600 | batch 150: loss=6.98 | batch 150: loss=7.54 | batch 300: loss=5.90 | batch 600: loss=4.22 |
| ~12,800 | batch 200: loss=6.15 | batch 200: loss=6.69 | batch 400: loss=5.43 | batch 800: loss=4.02 |
| ~14,400 | batch 225: ~5.9 | batch 225: ~6.4 | batch 450: loss=5.29 | batch 900: loss=3.88 |

**Takeaway**: Early in epoch, v3-core-heavy learns ~0.7 loss points faster per sample than v1-untied at 12.8k samples. v4-ortho (orthogonal init) learns even faster: at 12.8k samples, loss=4.02 vs v3-core-heavy's 5.43. Final `epoch=1` result for v3-core-heavy is `Val PPL 66.64`, which is expectedly poor because cosine LR decays to ~0 within a single epoch; that run should be treated as a tune/stability check, not a fair quality benchmark against 10-epoch runs.

---

## 5. Full Training Run: v3-full (2026-03-04)

Run with `small-matched` config, random init, A6000, 100k TinyStories samples, 10 epochs.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 121.04 | 38.99 | batch 0 starts at PPL 51,620 |
| 2 | 32.10 | 23.71 | |
| 3 | 22.28 | 18.09 | |
| 4 | 17.97 | 15.28 | |
| 5 | 15.65 | 13.68 | |
| 6 | 14.29 | 12.72 | |
| 7 | 13.47 | 12.17 | |
| 8 | 13.01 | 11.89 | |
| 9 | 12.78 | 11.79 | |
| 10 | 12.71 | 11.77 | final, 9.03h total |

**Log**: `logs/v5_train_small-matched.log` (overwritten by later runs)

---

## 6. Init Strategy Benchmark

**Script**: `python scripts/bench_init_strategies.py`

### Run A & B (early exploration, superseded by Run C)

Small-scale runs (150-200 samples, 1-2 epochs) confirmed init matters but were too short to distinguish convergence speed from quality.

### Run C: Comprehensive benchmark (2026-03-05)

**Setup**: `--size small --samples 1000 --epochs 5 --num_seeds 3 --strategies all --batch_size 33`

Config: dim=256, state_dim=512, 8 layers, 2 banks, expand=2, lr=1e-4, weight_decay=0.01, seq_len=128, CUDA.

| Strategy | Mean ValPPL | Std | Best Seed | Worst Seed | Avg Time |
|----------|-------------|-----|-----------|------------|----------|
| **orthogonal** | **168.27** | 2.29 | 43 (165.6) | 42 (169.6) | 50.9s |
| **hadamard** | **173.88** | 1.21 | 43 (172.8) | 44 (175.2) | 55.4s |
| dft | 275.18 | 3.35 | 44 (271.6) | 42 (278.3) | 55.3s |
| uniform | 289.08 | 2.95 | 44 (285.7) | 43 (291.1) | 52.8s |
| pi | 309.49 | 16.58 | 44 (293.2) | 42 (326.4) | 54.6s |
| sqrt_primes | 311.00 | 0.65 | 42 (310.5) | 43 (311.7) | 53.0s |
| dct | 317.19 | 3.09 | 43 (314.4) | 44 (320.5) | 54.7s |
| sinusoidal | 320.06 | 15.73 | 44 (302.2) | 42 (331.7) | 54.1s |
| s4d_lin | 337.12 | 4.94 | 43 (331.6) | 42 (341.1) | 52.7s |
| **random** | **348.80** | 10.78 | 43 (338.1) | 42 (359.6) | 55.1s |
| hartley | 350.96 | 7.94 | 44 (342.3) | 43 (357.8) | 53.5s |
| hippo | 351.97 | 6.47 | 43 (344.9) | 42 (357.5) | 53.7s |
| s4d_inv | 354.80 | 5.35 | 43 (349.4) | 42 (360.1) | 53.4s |
| circular | 946.67 | 274.36 | 42 (720.1) | 44 (1251.7) | 54.1s |
| van_der_corput | 1737.22 | 1600.30 | 42 (724.2) | 44 (3582.1) | 54.3s |
| halton | 1808.43 | 162.25 | 42 (1647.2) | 43 (1971.7) | 54.4s |
| golden_ratio | 2898.85 | 472.28 | 42 (2530.1) | 43 (3431.2) | 55.1s |
| roots_of_unity | 4679.95 | 3854.16 | 43 (800.5) | 42 (8508.4) | 53.2s |
| fermat_spiral | 52838.85 | 677.00 | -- | -- | 53.8s |
| fibonacci_spiral | 53096.23 | 214.77 | -- | -- | 53.8s |
| log_spiral | NaN | -- | -- | -- | 52.4s |

**Logs**: `logs/init_bench_20260305_112158.log`, `logs/init_bench_20260305_112158.json`

### Analysis

**Tier 1 -- Clear winners (2x better than random)**:
- `orthogonal` (168 PPL, std=2.29) and `hadamard` (174 PPL, std=1.21)
- Both are orthogonal matrices (norm-preserving isometry). Prevents vanishing/exploding gradients in complex linear layers.
- Backed by theory: orthogonal init requires width independent of depth for convergence ([ICLR 2020](https://openreview.net/forum?id=rkgqN1SYvr)).

**Tier 2 -- Solid (20% better than random)**:
- `dft`, `uniform`, `sqrt_primes`, `dct`, `sinusoidal` -- all between 275-320 PPL.
- `sqrt_primes` has the tightest std (0.65) of any strategy.

**Tier 3 -- At or near random baseline**:
- `s4d_lin`, `hartley`, `hippo`, `s4d_inv` cluster with random (~337-355).
- SSM-specific strategies (hippo, s4d_lin, s4d_inv) only override eigenvalues, not complex linear layers, so their effect is dominated by the default random complex linear init.

**Broken/removed (consistently worse than random)**:
- Spirals (fibonacci, fermat, log): unbounded radius causes NaN training loss.
- circular, roots_of_unity, golden_ratio, van_der_corput, halton: all > 2x worse than random.
- These 8 strategies were removed from `v5/init.py` on 2026-03-05.

### Caveats

- Benchmark used `small` config (dim=256, 8 layers), not `small-matched` (dim=128, 12 layers).
- Only 1k samples / 5 epochs (~1M tokens) vs 100k samples / 10 epochs (~22M tokens) in real training.
- This measures early convergence speed, not final quality. Literature suggests init advantages can diminish with longer training and proper regularization.
### Run D: A/B test -- orthogonal vs random (2026-03-05)

**Setup**: `--size small --samples 5000 --epochs 10 --num_seeds 3 --strategies orthogonal,random --batch_size 33`

Config: dim=256, state_dim=512, 8 layers, seq_len=128, 5000 TinyStories samples (~1M tokens), CUDA.

| Strategy | Mean ValPPL | Std | Best Seed | Worst Seed | Avg Time |
|----------|-------------|-----|-----------|------------|----------|
| **orthogonal** | **32.97** | 0.18 | 42 (32.77) | 43 (33.09) | 472.1s |
| random | 47.86 | 0.19 | 42 (47.65) | 44 (47.99) | 484.5s |

**Per-epoch convergence (mean across 3 seeds)**:

| Epoch | orthogonal ValPPL | random ValPPL | Relative Gap |
|-------|-------------------|---------------|-------------|
| 1 | 89.94 | 161.58 | 1.80x |
| 2 | 55.48 | 89.96 | 1.62x |
| 3 | 45.72 | 70.55 | 1.54x |
| 5 | 37.19 | 54.56 | 1.47x |
| 7 | 33.91 | 49.22 | 1.45x |
| 10 | **32.97** | **47.86** | **1.45x** |

**Conclusion**: Orthogonal init is **31% better than random at epoch 10** (32.97 vs 47.86 PPL). The gap narrows from 1.80x at epoch 1 to 1.45x by epoch 10, but stabilizes -- it is not converging to parity. Both strategies show extremely tight cross-seed variance (std < 0.2). This confirms orthogonal init provides a persistent, not transient, quality advantage.

**Logs**: `logs/init_bench_20260305_125523.log`, `logs/init_bench_20260305_125523.json`

---

## 7. Full Training Run: v4-ortho (2026-03-05)

Run with `small-matched` config, **orthogonal init** (seed=42), RTX 4090, batch_size=16, 100k TinyStories samples, 10 epochs.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 41.40 | 18.88 | batch 0 starts at PPL 51,460 |
| 2 | 16.32 | 13.14 | |
| 3 | 12.51 | 10.81 | |
| 4 | 10.72 | 9.61 | |
| 5 | 9.71 | 8.95 | |
| 6 | 9.08 | 8.52 | |
| 7 | 8.66 | 8.24 | |
| 8 | 8.38 | 8.08 | |
| 9 | 8.21 | 8.01 | |
| 10 | 8.13 | 8.00 | final, 3.48h total |

**Comparison vs v3-full (random init, A6000)**:

| Epoch | v3-full (random, A6000) | v4-ortho (orthogonal, 4090) | Improvement |
|-------|-------------------------|-----------------------------|-------------|
| 1 | 38.99 | 18.88 | 2.07x |
| 5 | 13.68 | 8.95 | 1.53x |
| 10 | 11.77 | 8.00 | 1.47x |

**Log**: `logs/v5_train_small-matched.log` (commit e24acd2)

---

## 8. Full Training Run: v5-full-ds (2026-03-05, in progress)

Run with `small-matched` config, **orthogonal init** (seed=42), RTX 4090, batch_size=16, **full TinyStories** (2,119,489 texts, 473,992,006 tokens), 10 epochs. Commit: `0467675`.

**Scale**: 103,744 batches/epoch (~7.2h/epoch). 21x more data than the 100k runs.

| Epoch | Train PPL | Val PPL | Wall time | Notes |
|-------|-----------|---------|-----------|-------|
| 1 | 8.59 | 6.27 | 7.18h | |
| 2 | 6.28 | 5.81 | 7.14h | |
| 3 | 5.97 | 5.59 | 7.39h | |
| 4 | -- | -- | -- | (completed, no table update yet) |
| 5 | 5.68 | **5.36** *best* | ~7.2h | New best checkpoint saved. |
| 6-10 | -- | -- | -- | in progress |

**Sample generation after each completed epoch** (prompt: `The quick brown`):

- Epoch 1: *"The quick brown bear went to the car and pulled out a big box. Inside was a treasure! Everyone clapped for their brave brave knight knight..."*
- Epoch 2: *"The quick brown bird felt so happy that it could eat the little apple and have fun with its friends. They laughed and played until it was time to go home, tired but happy."*
- Epoch 3: *"The quick brown dog wanted to go fast. He grabbed the butterfly with his paws and started jogging faster than ever before. He was so so happy that he had done it!"*

**Comparison vs v4-ortho (100k samples, epoch 10 best = 8.00)**:

| Epoch | v5-full-ds Val PPL | vs v4-ortho best (8.00) |
|-------|--------------------|-------------------------|
| 1 | 6.27 | 22% better |
| 2 | 5.81 | 27% better |
| 3 | 5.59 | 30% better |

**Takeaway**: Scaling to the full dataset immediately drops val PPL well below the 100k-sample best. The training curve is still improving consistently epoch-over-epoch with no sign of plateauing at epoch 3. Train/val gap is tiny (~0.38 at epoch 3), so no overfitting signal.

**Post–epoch 5 (2026-03-07)**: Best val PPL improved again at epoch 5 (5.36). Epoch 6 in progress. Per-batch loss in epoch 6 fluctuates ~1.5–1.9 (PPL ~4.4–6.8); epoch-level val PPL is the right metric. Expectation: curve can continue to improve through epoch 10 (v4-ortho improved every epoch 1→10). Optional follow-ups after this run: (1) short low-LR continuation on same TinyStories, or (2) resume from best checkpoint on a Wikipedia subset as a separate domain-adaptation experiment — not mid-run.

**Log**: `logs/v5_train_small-matched.log` (commit 0467675, in progress)

---

## 9. Mac No-Attention Smoke Test (2026-03-07)

Purpose: validate locally on Mac that disabling attention still allows real learning before spending RTX 4090 time on a matched ablation.

### MPS caveat

Direct MPS smoke runs were able to train through epoch 1, but crashed after saving the checkpoint during post-epoch sample generation with a Metal buffer allocation failure. Training itself looked healthy; the failure appears to be a Mac/MPS runtime issue rather than a learning failure.

- MPS run (`tiny`, batch=1, epochs=2, samples=100, seq_len=32, attention on) reached `Epoch 1 Val PPL 228.56` before crashing after checkpoint save.
- Because of that, the matched comparison below was completed on **CPU** on the same Mac, with identical settings except for `--no_attention`.

### Matched local smoke test (CPU, same Mac)

**Setup**:
- size=`tiny`
- batch_size=`1`
- epochs=`2`
- max_samples=`100`
- seq_len=`32`
- init=`orthogonal`, seed=`42`
- tokenizer/data=`TinyStories`
- comparison=`attention on` vs `--no_attention`

| Run | Attention | Params | Epoch 1 Val PPL | Epoch 2 Val PPL | Wall time | Notes |
|-----|-----------|--------|-----------------|-----------------|-----------|-------|
| mac-smoke-baseline-cpu | every 4 layers | 7,168,364 | 239.20 | **190.33** | 84.7s | Completed cleanly on CPU |
| mac-smoke-no-attn-cpu | none | 7,135,532 | 231.85 | **193.00** | 79.6s | Completed cleanly on CPU |

### Takeaway

- **Yes, the no-attention model still learns.**
- On this tiny local smoke test, removing attention barely changed quality: `190.33` PPL with attention vs `193.00` without attention after 2 epochs.
- The no-attention variant was also slightly smaller and a bit faster on CPU.
- This is not enough to claim parity at real scale, but it is strong evidence that the **ComplexSSM + bank path is doing substantive learning on its own**.

### Promotion decision

Promote this to a matched RTX 4090 ablation. Recommended next run:

```bash
python -m v5.train \
  --size small-matched \
  --batch_size 16 \
  --epochs 10 \
  --max_samples 100000 \
  --seq_len 256 \
  --init_strategy orthogonal \
  --init_seed 42 \
  --no_attention
```

Compare directly against `v4-ortho` / `v5-full-ds` settings to measure:
- epoch-by-epoch val PPL
- throughput
- sample quality
- whether the tiny-run near-parity holds at useful scale

---

## 10. Known Issues / TODO

### Diversity loss normalization bug (found 2026-03-07)

**File**: `v5/core/bank.py`, `compute_diversity_loss`, lines 216-217.

**Bug**: The denominator uses L1 norm of component magnitudes instead of L2 norm:

```python
# Current (wrong) -- L1 norm: sum(|a_i|)
mag_a = cabs(a).sum(dim=-1)
mag_b = cabs(b).sum(dim=-1)

# Correct -- L2 norm: sqrt(sum(|a_i|^2))
mag_a = torch.sqrt(cabs(a).square().sum(dim=-1) + 1e-8)
mag_b = torch.sqrt(cabs(b).square().sum(dim=-1) + 1e-8)
```

By Cauchy-Schwarz, `(sum|a_i|)^2 >= sum(|a_i|^2)`, so the denominator is systematically too large. Even identical banks produce cosine similarity well below 1.0, making the loss think banks are already diverse.

**Evidence**: `div=0.0000` throughout the entire v5-full-ds training log. The diversity loss is effectively a no-op with weight 0.05.

**Current impact**: None. The model succeeds via CE loss, orthogonal init, and router dynamics alone.

**TODO when fixing**:
1. Replace `.sum(dim=-1)` with `.square().sum(dim=-1).sqrt()` (or use `cabs2`).
2. Re-evaluate `diversity_loss_weight` (currently 0.05) -- once the metric is properly scaled to [0, 1], the weight may need adjustment.
3. Particularly relevant for `medium`/`large` configs with 3 banks where collapse risk is higher.

---

## How to Update

1. **Change Log**: Add a row when you change config/code/logic.
2. **Run Summary**: Add/update one row per run. Include GPU for throughput context.
3. **Chronological Run Registry**: Always add exact architecture + run setup + result + **commit hash** + **GPU**.
4. **Per-Sample Curves**: Add a column or extend rows when you have batch-level data from a new run.
