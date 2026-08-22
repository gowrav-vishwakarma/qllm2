# V13 experiments — selective PAM (delta + vault + phase addressing)

Lab notebook for V13: defaults make Stage-6 recall levers the production path.

## Smoke — 10M recall curriculum (`v13_micro_10m_recall`)

**Launch:**
```bash
bash scripts/run_v13_smoke.sh
# TOKEN_BUDGET=5000000 BATCH_SIZE=8 (defaults)
```

**Gates (before scale):**
| Probe | Target | Path |
|-------|--------|------|
| KV-recall @2048 | >90% | `checkpoints_v13/smoke_recall/behavioral.json` |
| Effective rank (layer 4) | >50% of d | `checkpoints_v13/smoke_recall/probes/` |
| Gate selectivity | \|p_content − p_filler\| >0.05 | `checkpoints_v13/smoke_recall/gate_probe.json` |
| O(1) selftest | pass | `python -m v13.selftest` |

## Matched baselines (same recall mix)

```bash
bash scripts/run_v13_transformer_micro.sh   # ~10M transformer
bash scripts/run_v13_mamba_micro.sh         # optional Mamba reference
```

## Scale — 100M vs Transformer

```bash
bash scripts/run_v13_scale.sh
# TOKEN_BUDGET=100000000 (default)
```

## Results

_(Append rows after each completed run.)_

| Run | Tokens | Recall@2048 (n=1) | Eff-rank % | Gate Δ | Wiki PPL | Notes |
|-----|--------|-------------------|------------|--------|----------|-------|
| smoke_recall (interim) | ~1.5M | 23.3% | 39.4% (L4) | ~0.00 | — | seq_len mix; gates not met |
| matched_recall (60 trials) | 4.1M / 3M / 2M | V13 10.6% @128 (NaN @512+); TX 10%; Mamba 12% | — | — | — | all at chance; V13 logits explode past 128 |
| cpu_recall_proof_v2 (800 steps, CPU) | 0.33M (synth) | recall 0 (all arms, 16 pairs) | — | E2b val 0.477 / filler 0.474 | — | E2b stable (no NaN); delta_legacy NaN at step 200; additive floor 5.554 |
| e2b_50m vs TX_50m (matched 75/25 recall+reason) | 50M | mem n1@2048: E2b 8.3% / TX 13.3% (chance 12.5%) | — | +0.033 (E2b) | 614k (synth-trained, expected) | E2b stable, zero NaN, flat across ctx 128→2048; TX also flat; both ~chance |

## Lessons

- V13 copies V11 with `write_mode=delta`, `vault_state`, `write_phase_address`, `n_states=3` as defaults.
- `fused_e3` disabled when delta+multistate (K-loop path).
- Chat vocab (50261) auto-selected when `cfg.vocab_size > 50257`.
- E2b gated-erase (separate erase gate, init 0.047): delta_legacy's erase fires every token and the protected memory erodes ~beta/d per filler token → training-level NaN by step 200 on the CPU recall proof. E2b stays finite at 800 steps.
- CPU proof is a token-budget-limited proxy (800 steps × 2048 tok = 0.33M, 16 eval pairs): E2b shows no recall and no gate selectivity yet (val 0.477 vs filler 0.474) — the 50M GPU run is the decisive test of the stability fix at scale.
- **50M scale (E2b, 75/25 recall+reason):** E2b is stable end-to-end (zero NaN, loss 10.85→0.32, 2.7h @ ~3.5K tok/s) and recall accuracy is *flat* across ctx 128→2048 (no V11-style collapse). But 11M params @ 50M tokens on 100% synthetic data lands **at/below chance** on the 8-way memory probe (E2b n1@2048 8.3%, TX 13.3%) and on reasoning tasks (copy/reverse ~10-15%, Caesar ~15% @gap0 → 0 @gap5). Digit-sum is at 50% for *both* models (easy task, near-ceiling for 11M params).
- **Verdict:** the E2b stability hypothesis is confirmed (no NaN, O(1) flat recall), but the *capability* goal (beat transformer at recall/reasoning) is NOT met at 11M/50M — both models are under-trained on the synthetic task distribution and neither shows a real memory advantage yet. The gate did learn slight selectivity (Δ +0.033) but it isn't enough to lift recall above chance.
- **Next (per user):** (1) train a real LLM on wikitext/DCLM/FineWeb with a small synthetic blend to keep the memory mechanism alive; (2) revisit the recall probe — at 11M params the model may be too small to beat a transformer at 8-way contrastive even with perfect O(1) memory, so consider a larger matched model (50-100M) for the definitive claim.
- **Throughput (RTX 4090 24GB, v13_micro_10m_recall, seq_len=2048):** 50M run's ~3.4K tok/s was the B=8 + `delta_chunk=32` default. Two **math-neutral** knobs (loss identical to 4 decimals): `delta_chunk 32→64` = 1.77× (fewer sequential UT chunk steps — 64 is the model.py default; the recall preset had lowered it to 32), `batch 8→16` = 1.9×. Best config: `--batch_size 16 --delta_chunk 64 --no_grad_ckpt --fused_ce` → **~11.6K tok/s** (3.4× the 50M run), 21.2GB; `--compile`/`--amp_dtype bf16` add ~3% each but B=24 OOMs. Re-derive the ceiling per GPU: max B that fits ~21GB at delta_chunk=64. Triton custom kernels remain a dead end for training (fight `torch.compile`; see V7/V11 — Flash-PAM was negative).
- **Grad-ckpt bug (FIXED, v13/model.py `_ckpt_block`):** block input `z` is a non-leaf (from embedding); torch's checkpoint sees non-leaf in the original forward but a *detached leaf* on recompute → autograd saves a different tensor sequence in the two passes → torch 2.8 `determinism_check` fails (`[B,H,T,1]`↔`[B,H,T,32]` swap; only reproduced with vault+delta+phase on). Fix: `z_leaf = z_in.detach().requires_grad_(True)` inside the checkpointed fn so both passes build identical graphs. Verified: 2-layer model, loss + all 69 param grads match no-ckpt to 5e-7; 100M now trains under ckpt (previously OOM'd or crashed).
- **Throughput — 100M-class (v13_e3_k3_selective, dim384×16L, 4090 24GB):** needs grad-ckpt (B8 no-ckpt OOMs by 2MiB). Steady: **B10 ≈ 2.3K tok/s** (19.8GB); B8 ≈ 2.0K; B12 OOMs on step-2 recompute peak. 11M stays faster per token: **B16 ≈ 11.6K tok/s** (21.2GB). Rule of thumb on 24GB: 11M→B16 no-ckpt; 100M→B10 grad-ckpt.
