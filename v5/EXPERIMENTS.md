# V5 Experiment Log

Structured log of config changes, training runs, and results. Update this file after each significant run.

---

## 1. Change Log

| Date | Change | Why | Run |
|------|--------|-----|-----|
| 2026-03-04 | **Weight tying**: Removed `output_proj`, compute logits as `Re(z * conj(embed))` | Standard practice (GPT-2, LLaMA, Mamba). Saves 12.9M params, algebraically consistent. | v2-tied |
| 2026-03-04 | **Reinvested core params**: 12 layers, expand=4, state_dim=512, 8 heads | Put freed params into core model. Wider CGU over more banks -- "let the algebra do more per path". | v3-core-heavy |

---

## 2. Run Summary Table

| Run | Config | Total | Core | Layers | Banks | Expand | State | Batch | Epochs | Ep1 Val PPL | Best Val PPL | tok/s | Notes |
|-----|--------|-------|------|--------|-------|--------|-------|-------|--------|-------------|-------------|-------|-------|
| v1-untied | small-matched (orig) | 31.6M | 5.8M (18%) | 8 | 2 | 2 | 256 | 64 | 10 | 32.41 | 21.03 (ep2) | ~14.4k | Baseline, no weight tying |
| v2-tied | small-matched | 18.7M | 5.8M (31%) | 8 | 2 | 2 | 256 | 64 | 10 | partial | -- | ~14.4k | Weight tying only; slower early convergence |
| v3-core-heavy | small-matched | 28.7M | 15.8M (55%) | 12 | 2 | 4 | 512 | 32 | 1 | 66.64 | 66.64 | ~6.2k | Tune/stability run; 1-epoch cosine undertrains and is not quality-comparable to 10-epoch runs |

**Data (default unless overridden)**: TinyStories + GPT-2 tokenizer, A6000 GPU.

---

## 3. Chronological Run Registry (Source of Truth)

Use this table for exact reproducibility: what architecture was used, what data/setup it ran with, and what result it produced.

| Run | Wall Clock Start | Architecture Snapshot | Param Snapshot | Run Setup | Result Snapshot | Status |
|-----|------------------|-----------------------|----------------|-----------|-----------------|--------|
| v1-untied | 2026-03-04 10:13 | `dim=128, state=256, layers=8, banks=2, expand=2, heads=4, attn_every=4, tied_output=no` | total=31,556,184; core=5,791,448 (18%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=64, epochs=10, batches/epoch=1202 | epoch1 val_ppl=32.41, epoch2 val_ppl=21.03, throughput~14.4k tok/s | baseline (primary reference) |
| v2-tied | 2026-03-04 11:31 | `dim=128, state=256, layers=8, banks=2, expand=2, heads=4, attn_every=4, tied_output=yes` | total=18,690,392; core=5,824,600 (31%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=64, epochs=10, batches/epoch=1202 | early epoch1 lagged ~0.5 loss vs v1 at same batch index | partial/incomplete |
| v3-core-heavy | 2026-03-04 12:04 | `dim=128, state=512, layers=12, banks=2, expand=4, heads=8, attn_every=4, tied_output=yes` | total=28,682,372; core=15,816,580 (55%) | dataset=TinyStories, max_samples=100000, seq_len=256, batch=32, epochs=1, batches/epoch=2403 | epoch1 val_ppl=66.64, best_val_ppl=66.64, throughput~6.2k tok/s | tune/stability only (not quality-comparable) |

**Comparison note**: v3 used `epochs=1`, so cosine LR decayed to ~0 within one epoch; this makes quality metrics non-comparable to 10-epoch runs.

---

## 4. Per-Sample Learning Curves

Aligned by **samples seen** (not batch index), since batch sizes differ.

| Samples | v1-untied (batch=64) | v2-tied (batch=64) | v3-core-heavy (batch=32) |
|---------|----------------------|--------------------|---------------------------|
| ~3,200 | batch 50: loss=9.63 | batch 50: loss=9.91 | batch 100: loss=8.75 |
| ~6,400 | batch 100: loss=8.08 | batch 100: loss=8.66 | batch 200: loss=6.97 |
| ~9,600 | batch 150: loss=6.98 | batch 150: loss=7.54 | batch 300: loss=5.90 |
| ~12,800 | batch 200: loss=6.15 | batch 200: loss=6.69 | batch 400: loss=5.43 |
| ~14,400 | batch 225: ~5.9 | batch 225: ~6.4 | batch 450: loss=5.29 |

**Takeaway**: Early in epoch, v3-core-heavy learns ~0.7 loss points faster per sample than v1-untied at 12.8k samples. Final `epoch=1` result is `Val PPL 66.64`, which is expectedly poor because cosine LR decays to ~0 within a single epoch; this run should be treated as a tune/stability check, not a fair quality benchmark against 10-epoch runs.

---

## 5. Init Strategy Benchmark (2026-03-04)

**Script**: `python scripts/bench_init_strategies.py` (see README Structured Initialization)

### Run A: 150 samples, 1 epoch, 3 strategies (random, golden_ratio, dft)

| Strategy    | Val PPL   | Val Loss | Train Loss | Time(s) | Tok/s |
|-------------|-----------|----------|------------|---------|-------|
| golden_ratio | 21091.24 | 9.96     | 10.31      | 10.9    | 2431  |
| random      | 22811.48 | 10.04    | 10.38      | 12.0    | 2215  |
| dft         | 22978.60 | 10.04    | 10.36      | 10.8    | 2456  |

**Takeaway (1 epoch)**: golden_ratio best; dft worst. Structured number-theoretic init (golden ratio) leads early.

### Run B: 200 samples, 2 epochs, 6 strategies

| Strategy    | Val PPL   | Val Loss | Train Loss | Time(s) | Tok/s |
|-------------|-----------|----------|------------|---------|-------|
| pi          | 1894.23  | 7.55     | 7.58       | 29.5    | 2425  |
| hippo       | 1988.42  | 7.60     | 7.65       | 29.2    | 2451  |
| random      | 1984.80  | 7.59     | 7.65       | 60.3    | 1189  |
| dft         | 1993.65  | 7.60     | 7.65       | 29.4    | 2441  |
| sinusoidal  | 2252.91  | 7.72     | 7.77       | 30.5    | 2353  |
| golden_ratio | 2847.10 | 7.95     | 8.02       | 32.2    | 2226  |

**Takeaway (2 epochs)**: pi best; golden_ratio worst. Ordering flips vs 1 epoch. pi (Weyl sequence) and hippo (SSM-specific) competitive with random; golden_ratio overfits or plateaus. Throughput: structured inits (pi, dft, hippo) ~2x faster than random (likely first-step compile/cache effects).

**Next**: Run more epochs (5–10) and more strategies (circular, fibonacci_spiral, sqrt_primes) to see if grokking attractors show different long-horizon behavior.

---

## How to Update

1. **Change Log**: Add a row when you change config/code/logic.
2. **Run Summary**: Add/update one row per run.
3. **Chronological Run Registry**: Always add exact architecture + run setup + result.
4. **Per-Sample Curves**: Add a column or extend rows when you have batch-level data from a new run.
