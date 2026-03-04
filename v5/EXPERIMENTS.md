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
| v3-core-heavy | small-matched | 28.7M | 15.8M (55%) | 12 | 2 | 4 | 512 | 32 | 1 | -- | -- | ~6.2k | Tune run; faster per-sample learning |

**Data**: 100k TinyStories, seq_len=256, A6000 GPU.

---

## 3. Per-Sample Learning Curves

Aligned by **samples seen** (not batch index), since batch sizes differ.

| Samples | v1-untied (batch=64) | v2-tied (batch=64) | v3-core-heavy (batch=32) |
|---------|----------------------|--------------------|---------------------------|
| ~3,200 | batch 50: loss=9.63 | batch 50: loss=9.91 | batch 100: loss=8.75 |
| ~6,400 | batch 100: loss=8.08 | batch 100: loss=8.66 | batch 200: loss=6.97 |
| ~9,600 | batch 150: loss=6.98 | batch 150: loss=7.54 | batch 300: loss=5.90 |
| ~12,800 | batch 200: loss=6.15 | batch 200: loss=6.69 | batch 400: loss=5.43 |
| ~14,400 | batch 225: ~5.9 | batch 225: ~6.4 | batch 450: loss=5.29 |

**Takeaway**: v3-core-heavy learns ~0.7 loss points faster per sample than v1-untied at 12.8k samples. Reinvested core params are paying off. Throughput is 2.3x slower (batch 32 vs 64, larger model).

---

## How to Update

1. **Change Log**: Add a row when you change config or code.
2. **Run Summary**: Add a row after each run; fill best val PPL when training completes.
3. **Per-Sample Curves**: Add a column or extend rows when you have batch-level data from a new run.
