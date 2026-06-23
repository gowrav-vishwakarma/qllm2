# V11 PAM Math Probes (Phase A)

Pure mechanism tests for Phase-Associative Matrix memory — **no trained checkpoint required**.

Phase B (behavioral probes on `V11LM` checkpoints) is deferred until after the 10B scratch pretrain completes. See [`scripts/v11_pam_probes.py`](../scripts/v11_pam_probes.py) (not yet implemented).

## Quick start

```bash
# Full battery (selftest + all math probes)
./scripts/run_v11_pam_math.sh

# Individual tests
.venv/bin/python -m v11.pam_math --test binding
.venv/bin/python -m v11.pam_math --test persistence --distances 64,128,512,2048
.venv/bin/python -m v11.pam_math --test niah --distances 64,128,512,1024,2048
.venv/bin/python -m v11.pam_math --test niah-grid --lengths 128,256,512,1024,2048
.venv/bin/python -m v11.pam_math --all

# Long context (beyond training seq_len=2048) — pure PAM math, no checkpoint
.venv/bin/python -m v11.pam_math --test long-context
.venv/bin/python -m v11.pam_math --test long-context --max-distance 1048576   # 1M tokens
MAX_DISTANCE=524288 ./scripts/run_v11_pam_math.sh   # full battery + 512K long-context
```

JSON results land in `logs/v11/pam_math/`.

## What this proves vs what it cannot

| Question | Phase A (math) | Phase B (trained LM) |
|----------|----------------|----------------------|
| Is `S = γS + V⊗K*` implemented correctly? | ✅ `v11.selftest`, layer-bridge | — |
| Does matrix PAM beat vector HRR capacity? | ✅ binding test | — |
| Does γ protect associations over distance? | ✅ persistence, NIAH | needs learned gates |
| Can GSP freeze state for needles? | ✅ GSP sweep (mechanism ceiling) | does training use it? |
| Does model recall `glorp → banana` in text? | ❌ | Phase B |

## Test battery

### A1 — Correctness (`v11.selftest`)

Parallel training form (chunked/dual/UT) must match O(1) recurrent inference for all modes: baseline, E1 per-channel, E2 delta, E3 multistate, E1+E3 combo.

### A2 — Binding capacity

Write N random complex key–value pairs into one matrix state:

```
S = Σ_i v_i ⊗ k_i*     retrieve: y_i = S @ k_i
```

Compare against vector HRR baseline (circular convolution, expect ~O(1/√N)).

**Latest run (seed=42, d=64):**

| N | Matrix PAM | Vector HRR | Theory 1/√N |
|---|------------|------------|-------------|
| 1 | 100% | 100% | 1.00 |
| 64 | **100%** | ~13% | 0.125 |
| 193 | ~99% | ~1% | 0.072 |

Matrix PAM retains near-perfect retrieval to N≈160 before gradual drop — far above the vector baseline. This validates the architectural motivation from HSB failure (interference in vector state).

### A3 — Persistence vs distance

Write one association, then T filler writes with decay γ. Measure relative retrieval score at query.

**Key finding:** With γ=1 and no decay-only path, interference from filler dominates. With γ=0.999, associations survive longer but filler still corrupts. Default γ≈0.982 (from `base_dt_bias=-4`) decays aggressively over 1024+ tokens unless GSP activates.

### A4 — Multi-association interference

Write N pairs, then M filler tokens, query the first pair.

| Config | pairs=8, filler=0 | pairs=32, filler=256 |
|--------|-------------------|----------------------|
| Additive | 0.95 | 0.50 |
| Delta (E2) | 0.46 | 0.07 |
| E3 K=3 | 1.79* | 0.48 |

\*E3 scores can exceed 1.0 relative to single-write baseline because three phase-weighted states sum.

Delta writes help on repeated-key overwrite but hurt when keys are all distinct (error-correction assumes stale binding).

### A5 — State rank evolution

Effective rank of S_t grows under random writes (max ~50/64 at γ=0.995) but stays at 1.0 under repeated-key overwrite — confirming matrix capacity is used for distinct associations.

### A6 — Conjugate interference

Random K*·Q pairs: ~50% destructive (Re < 0), as expected for uniform random phases. Untrained `V11PAMLayer` projections preserve similar statistics (~52% destructive).

### A7 — Layer bridge

NumPy hand-rolled `_recur_step_additive` matches PyTorch to ~1e-17. Parallel vs recurrent max|Δ| < 1e-8 for baseline, E1, E2, E3.

### A8 — Needle-in-haystack (math)

PAM NIAH is **decay + interference**, not positional attention.

**A8.1 Bare decay:** Default γ=0.982 → early needles at T=2048 retain ~3% relative score. γ=0.999 → ~32%.

**A8.2 GSP protection:** With `p_fill=1.0` (γ→1, no filler writes), needle preserved perfectly at T=2048. With `p_fill=0.99`, ~69% at T=2048. This is the **mechanism ceiling** — training must learn to raise p on filler tokens.

**A8.3 Position grid:** Strong **recency bias** under additive write (r=1.0 survives, r=0.0 ~9% at T=2048). GSP p=0.99 flattens the map (early needle ~69% at T=2048).

**A8.4 Multi-needle:** Needle at end survives; needles at start decay — consistent with recency + decay.

**A8.5 Key collision:** High cosine similarity between filler keys and needle key (0.9) increases relative score (0.95) — interference can be *constructive* when keys align, corruptive when they partially overlap.

### A9 — Long-context math (beyond seq_len=2048)

V11 **training** uses `seq_len=2048`, but PAM **inference** is recurrent with fixed O(d²) state per layer — there is no KV cache and no hard context cap in the math. Test A9 sweeps to 256K–1M+ tokens.

```bash
.venv/bin/python -m v11.pam_math --test long-context
.venv/bin/python -m v11.pam_math --test long-context --max-distance 1048576
.venv/bin/python -m v11.pam_math --test long-context --max-distance 262144 --no-filler-sim  # analytic only
```

**Distances (default):** 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K (extend with `--max-distance`).

| Mode | T=65,536 (example) |
|------|---------------------|
| Analytic γ^T only (γ=0.982) | ~10⁻⁴⁸³ (needle gone) |
| Filler + γ=0.982 | ~5% relative (interference dominates decay) |
| GSP p_fill=1.0 (freeze) | **100%** — state preserved exactly |
| GSP p=0.99, needle at end | ~99% |
| GSP p=0.99, needle at start | ~1% (recency still matters with partial protection) |

**Key insight:** At extreme distance, **decay alone** destroys early needles (γ^65536 ≈ 0 for γ=0.982). Random filler writes add interference on top. Only **GSP near 1.0** gives transformer-like flat retention — the architecture supports it; training must learn when to protect.

64K simulated run completes in ~25s on CPU (d=64). 256K takes ~2–3 min; 1M ~10–15 min.

### A10 — Language filler (English / WikiText, not random vectors)

Random unit vectors have E[k_i·k_j]≈0. **Language embeddings cluster** — a harder, more realistic interference regime.

```bash
.venv/bin/python -m v11.pam_math_language --test language-filler --filler-tokens 10000
.venv/bin/python -m v11.pam_math_language --test language-filler --filler-tokens 100000
```

Protocol: write `glorp → banana` via GPT-2 embeddings (projected to complex K/V), stream WikiText-103 as haystack, query with `glorp`. Compares against random filler at the same length.

Uses GPT-2 pretrained embeddings (clustering) + fixed K/V projections. Optional `--checkpoint` on rank-text for trained V11 projections; full LM needle test is Phase B.

### A11 — Effective rank on real text

Stream WikiText through `V11PAMLayer` (or full block with `--checkpoint`); log effective rank of S_t vs token position. The signature figure: **does rank saturate early or keep growing?**

```bash
.venv/bin/python -m v11.pam_math_language --test rank-text --text-tokens 50000 --sample-every 100
.venv/bin/python -m v11.pam_math_language --test rank-text --checkpoint checkpoints_v11_e3_k3/best_model.pt
```

Early untrained run (5K tokens): rank rises quickly to ~20–23 / d=64 then plateaus — suggests capacity usage saturates well below d. Re-run with checkpoint after 10B for the paper figure.

**Projection seed sweep (50 trials, 5K wiki tokens):** Language relative retrieval is **stable** (mean 9.3, std 2.8, range 3.3–15.3) — not projection luck on the absolute score. Language **always** beats random filler on this metric (lang/rand min **14×**), but the ratio variance is huge because random denominator can be ~0.02–0.3. This does **not** support “English destroys the needle more than random”; it supports **constructive clustering** at the embedding-projection layer. Definitive test still needs trained `V11LM` + next-token prediction.

**Singular value spectrum:** Log full σ₁…σ_d at each sample — effective rank can plateau while spectrum shape shifts. See `--test rank-text` output `singular_values` in JSON.

**protect_gate hypothesis:** Math probes (A8/A9) show recency bias and decay unless GSP `p→1`. If training does not learn *when not to write*, everything writes → interference + forgetting. Logging learned `p` on filler vs needle tokens is the top Phase B diagnostic.

## Interpreting relative scores > 1.0

Retrieval score is normalized against the score from a fresh single write. Values > 1.0 mean later filler writes **constructively aligned** with the query direction — not a bug, but a reminder that random filler is not neutral in phase space.

## Phase B (deferred)

After 10B scratch pretrain:

1. Implement `scripts/v11_pam_probes.py` — invented-word, passkey, interference on real text
2. Run on `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt`
3. Overlay behavioral recall-vs-distance on A8 math curves
4. Log learned `protect_gate` on needle vs filler tokens

The gap between Phase A ceilings and Phase B results measures **how much architectural capacity training realized**.

## Files

| File | Role |
|------|------|
| [`v11/pam_math.py`](pam_math.py) | All Phase A experiments + CLI |
| [`v11/selftest.py`](selftest.py) | Train ≡ recurrent correctness |
| [`scripts/run_v11_pam_math.sh`](../scripts/run_v11_pam_math.sh) | Runner script |
| `logs/v11/pam_math/*.json` | Timestamped results |
