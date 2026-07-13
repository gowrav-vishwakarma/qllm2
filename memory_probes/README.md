# Memory Probes

Evaluation framework for **recurrent matrix memory** — binding capacity, persistence, interference, state rank, needle-in-haystack, and long-context sweeps. No trained checkpoint required for the synthetic battery; language probes use GPT-2 embeddings or optional V11 checkpoints.

Designed for researchers comparing matrix-memory architectures, not tied to a single model implementation. V11 `V11PAMLayer` is used as a reference implementation for layer-bridge and real-text rank probes.

For the paper protocol, claim policy, Mac/RTX 4090 handoff, and exact
publication commands, see [`PUBLICATION.md`](PUBLICATION.md).

## Quick start

```bash
# Full battery (selftest + all probes)
./scripts/run_memory_probes.sh

# Individual tests
uv run python -m memory_probes --test binding
uv run python -m memory_probes --test binding-matched --arch-dim 16
uv run python -m memory_probes --test persistence --distances 64,128,512,2048
uv run python -m memory_probes --test niah --distances 64,128,512,1024,2048
uv run python -m memory_probes --test niah-grid --lengths 128,256,512,1024,2048
uv run python -m memory_probes --all

# Long context (beyond typical training seq_len)
uv run python -m memory_probes --test long-context
uv run python -m memory_probes --test long-context --max-distance 1048576   # 1M tokens

# Language / real text
uv run python -m memory_probes --test language-filler --filler-tokens 10000
uv run python -m memory_probes --test language-filler --projection-trials 50
uv run python -m memory_probes --test rank-text --text-tokens 50000 --sample-every 100
```

JSON results land in `logs/memory_probes/`. Compare against legacy runs in `logs/v11/pam_math/` via `scripts/compare_memory_probes.py`.

## Package layout

| Module | Probe |
|--------|-------|
| `selftest.py` | Implementation correctness (train ≡ recurrent) |
| `capacity.py` | Binding capacity vs vector HRR baseline |
| `persistence.py` | Association survival vs distance under decay |
| `interference.py` | Multi-association interference, conjugate retrieval, layer bridge |
| `rank.py` | State rank (synthetic + real WikiText) |
| `language.py` | Language filler (clustered embeddings) |
| `long_context.py` | Needle-in-haystack and extreme-distance sweeps |
| `behavioral.py` | Exact-length trained-LM invented-association protocol |
| `publication.py` | Multi-seed result schema, aggregation, and Mac sweep |
| `core.py` | Shared NumPy reference math |
| `cli.py` | Unified CLI |

## What this proves vs what it cannot

| Question | Memory probes (math) | Trained LM probes (Phase B) |
|----------|----------------------|----------------------------|
| Is recurrent update implemented correctly? | ✅ selftest, layer-bridge | — |
| Does matrix memory beat HRR at equal width / equal bytes? | ✅ `binding` / `binding-matched` | — |
| Does decay protect associations over distance? | ✅ persistence, NIAH | needs learned gates |
| Can protection freeze state for needles? | ✅ GSP sweep (mechanism ceiling) | does training use it? |
| Does model recall `glorp → banana` in text? | ❌ | Phase B |

## Reproducibility

| Probe class | Deterministic? | Notes |
|-------------|----------------|-------|
| NumPy math (binding, persistence, interference, NIAH, long-context) | ✅ `seed=42` | Bit-exact across runs |
| Synthetic rank (`--test rank`) | ✅ | Pure NumPy recurrence |
| Layer-bridge / selftest | ✅ | Fixed torch seeds in harness |
| Conjugate layer check | ⚠️ | Untrained `V11PAMLayer` input uses `torch.randn` without fixed seed |
| Rank on real text | ⚠️ | Untrained layer weights re-init each run; use `--checkpoint` for trained weights |
| Language filler | ✅ | GPT-2 embed + fixed projection seed; WikiText stream cached under `.cache/v7_tokens/` |

---

## Per-probe results (seed=42, d=64 unless noted)

Reference logs: `logs/memory_probes/memory_probes_all_20260623_160013.json`, `memory_probes_long_context_20260623_160102.json`, legacy `logs/v11/pam_math/pam_math_*.json`.

### selftest — train ≡ recurrent

Parallel training form (chunked / dual / UT) must match O(1) recurrent inference for all modes: baseline, E1 per-channel, E2 delta, E3 multistate, E1+E3 combo.

**Result:** all modes PASS (`max|Δ|` ≤ 1e-7).

### capacity — binding vs vector HRR

Write N random complex key–value pairs into one matrix state `S = Σᵢ vᵢ ⊗ kᵢ*`; retrieve with `yᵢ = S @ kᵢ`. Compare against vector HRR (circular convolution, theory ~O(1/√N)).

| N | Matrix PAM | Vector HRR | Theory 1/√N |
|---|------------|------------|-------------|
| 1 | 100% | 100% | 1.00 |
| 64 | **100%** | ~9% | 0.125 |
| 193 | ~99% | ~1% | 0.072 |

This equal-width diagnostic gives PAM `d²` complex state scalars and HRR only
`d`, so it does **not** establish storage efficiency. Use `binding-matched` for
the publication comparison at equal state bytes. The equal-width result shows
how the two representations behave when built from the same embedding width.

### persistence — decay vs distance

Write one association, then T filler writes with decay γ. Measure relative retrieval score at query.

**Key finding:** With γ=1, filler interference dominates. With γ=0.999, associations survive longer but filler still corrupts. Default γ≈0.982 (from `base_dt_bias=-4`) decays aggressively over 1024+ tokens unless GSP/protection activates.

### interference — multi-association + conjugate + layer bridge

Write N pairs, then M filler tokens, query the first pair.

| Config | pairs=8, filler=0 | pairs=32, filler=256 |
|--------|-------------------|----------------------|
| Additive | 0.95 | 0.50 |
| Delta (E2) | 0.46 | 0.07 |
| E3 K=3 | 1.79* | 0.48 |

\*E3 scores can exceed 1.0 relative to single-write baseline because three phase-weighted states sum constructively.

Delta writes help on repeated-key overwrite but hurt when keys are all distinct (error-correction assumes stale binding).

**Conjugate retrieval (K*·Q):** ~50% destructive (Re < 0) for random phases, as expected. Untrained layer projections preserve similar statistics (~50% destructive; varies slightly run-to-run).

**Layer bridge:** NumPy `_recur_step_additive` matches PyTorch to ~1e-17. Parallel vs recurrent `max|Δ|` < 1e-8 for baseline, E1, E2, E3.

### rank — state rank evolution

**Synthetic (random keys):** effective rank grows under random writes (max ~50/64 at γ=0.995, final ~49.5) — matrix capacity is used for distinct associations.

**Synthetic (overwrite mode):** rank stays at 1.0 — repeated key does not expand subspace.

**Real WikiText (untrained V11PAMLayer, 5K tokens):** rank rises quickly to ~20–23 / d=64 then plateaus. Singular value spectrum logged (σ₁…σ_d): effective rank can plateau while σ₁ keeps growing — shape shifts even when rank metric is flat. Re-run with `--checkpoint` after pretrain for the paper figure.

### long_context — NIAH + extreme distance

Recurrent state is **O(d²) per layer** — no KV cache, no hard context cap in the math. Sweeps to 256K–1M+ tokens (V11 training uses seq_len=2048; inference is unbounded).

**Bare decay (default γ=0.982):** needle at T=2048 retains ~3% relative score; γ=0.999 → ~32%.

**GSP protection:** `p_fill=1.0` (freeze, no filler writes) → needle preserved perfectly to 65K+. `p_fill=0.99` → ~69% at T=2048. This is the **mechanism ceiling** — training must learn to raise p on filler tokens.

**Position grid (T=2048):** strong **recency bias** under additive write (r=1.0 survives, r=0.0 ~9%). GSP p=0.99 flattens the map (early needle ~69% at T=2048).

**Multi-needle:** needle at end survives; needles at start decay — consistent with recency + decay.

**Key collision:** high cosine between filler keys and needle key (0.9) can *increase* relative score (~0.95) — interference is constructive when keys align.

#### Long-context table (65K sweep)

| Mode | T=65,536 |
|------|----------|
| Analytic γ^T only (γ=0.982) | ~0 (needle gone) |
| Filler + γ=0.982 | ~5% relative (interference dominates decay) |
| GSP p_fill=1.0 (freeze) | **100%** |
| GSP p=0.99, needle at end | ~99% |
| GSP p=0.99, needle at start | ~1% (recency still matters with partial protection) |

**Timing (CPU, d=64):** 64K simulated ~25s; 256K ~2–3 min; 1M ~10–15 min.

### language — WikiText filler (not random vectors)

Random unit vectors have E[kᵢ·kⱼ]≈0. Language embeddings **cluster** — a more realistic interference regime.

Protocol: write `glorp → banana` via GPT-2 embeddings (projected to complex K/V), stream WikiText-103 as haystack, query with `glorp`. Compare against random filler at the same length.

**Single run (5K wiki tokens, projection seed=42):** language rel ≈ 12.6, random rel ≈ 0.15, lang/rand ≈ 84×.

**50 projection-seed sweep (5K tokens):**

| Metric | mean | std | min | max |
|--------|------|-----|-----|-----|
| Language relative retrieval | 9.3 | 2.8 | 3.3 | 15.3 |
| Lang/random ratio | 94× | 86× | 14× | 461× |

Language absolute score is **stable** (CV < 35%) — not projection luck. Language **beats random on every seed** (min ratio 14×), but ratio variance is huge because the random denominator can be tiny (~0.02–0.3).

This does **not** support “English destroys the needle more than random”; it supports **constructive clustering** at the embedding-projection layer. Definitive test still needs trained `V11LM` + next-token prediction (Phase B).

### Interpreting relative scores > 1.0

Retrieval score is normalized against a fresh single-write baseline. Values > 1.0 mean later filler writes **constructively aligned** with the query direction — not a bug, but a reminder that random filler is not neutral in phase space.

---

## Hypotheses for Phase B

**protect_gate:** Math probes show recency bias and decay unless GSP `p→1`. If training does not learn *when not to write*, everything writes → interference + forgetting. Logging learned `p` on filler vs needle tokens is the top Phase B diagnostic.

The gap between Phase A ceilings and Phase B behavioral recall measures **how much architectural capacity training realized**.

## Phase B (implemented protocol; GPU results pending)

The shared contrastive next-token runner is
`scripts/run_memory_behavioral.py`. After scratch pretraining:

1. Run the same exact-length invented-association suite on trained PAM,
   Transformer, and Mamba checkpoints.
2. Overlay behavioral recall-vs-distance on NIAH math curves.
3. Persist learned `protect_gate` diagnostics on content versus filler tokens.
4. Treat the results as cross-architecture evidence only when training and
   parameter budgets are matched.

See [`PUBLICATION.md`](PUBLICATION.md) for the exact RTX 4090 command.

## Backward compatibility

```bash
# Still works, emits DeprecationWarning
uv run python -m v11.pam_math --test binding
./scripts/run_v11_pam_math.sh   # delegates to run_memory_probes.sh
```

## Files

| Path | Role |
|------|------|
| `memory_probes/*.py` | Probe implementations + CLI |
| `scripts/run_memory_probes.sh` | Full runner |
| `scripts/compare_memory_probes.py` | Diff new vs legacy JSON logs |
| `logs/memory_probes/*.json` | Timestamped results |
| `v11/selftest.py` | Correctness harness (also via `memory_probes.selftest`) |
