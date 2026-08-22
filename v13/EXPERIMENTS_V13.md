# V13 experiments — selective PAM (delta + vault + phase addressing)

Lab notebook for V13: defaults make Stage-6 recall levers the production path.

## Speed redesign 2026-08-22 — 100M-class on 4090 (measured on 4090)

**Goal:** 100M-class V13 (`v13_e3_k3_selective`, 100.6M, dim384×16L, K=3 selective PAM)
training "like a transformer — minutes not hours". Before: **2.3K tok/s** (B10).
After: **~21K tok/s** (B16) — a 9× speedup. All changes are math-exact
(equivalence tests bit-exact or ≤1e-7); the gate system is **kept, not removed** —
only its training cost was fixed.

### What was broken → what we changed (for other LLMs reading this)

| # | Bug (symptom) | Root cause | Fix | Effect |
|---|---------------|------------|-----|--------|
| 1 | 100M train step 2.3K tok/s (vs ~27K for the 11M model) | K=3 states solved by a **per-state loop** (3 chunk-solves/layer) | `_forward_multistate_delta_fused`: collapse K=3 chunk-solves into **one batched complex triangular solve** per chunk (K batched on dim 0) | 2.3K → ~20K tok/s (the dominant win) |
| 2 | Gate aux OOM'd first resume (12GB extra backward) | Gate-surprisal stash built from **live** `x` → its loss backproped a second full-network gradient | Stash built from `x.detach()` → grad reaches only `protect_gate` weights | OOM gone |
| 3 | Gate target cost a **2nd full O(V) head GEMM** per step | Target = per-token NLL computed by `linear_ce_per_token` (separate 50k-vocab GEMM), duplicating the main CE | **NLL byproduct**: the main fused CE already computes exact per-token NLL while summing — emit it as `loss._nll` (fp32, `torch.no_grad` capture before mask) → gate target is **free and exact** | gate-target fwd: ~700ms → **0.2ms**; loss value identical to 1e-8 |
| 4 | NLL quantized to ~0.03 steps | Under `autocast(bf16)` the head GEMM is downcast to **bf16 at the matmul level** even with `.float()` inputs (8-bit mantissa) | GEMM+CE wrapped in `torch.amp.autocast(enabled=False)` (fp32 region) | NLL exact; main CE now fp32 (was bf16) — loss shifts ~1e-4, more accurate |
| 5 | **Gate-ON step 730ms slower than gate-OFF** (B16/T2048) despite #3 | **All 16 layers re-computed on every backprop:** the gate stash was a tensor node *inside* each `torch.utils.checkpoint` block. Backpropping the gate loss forced the checkpoint engine to **re-run each of the 16 PAM blocks** (fwd recompute) to recover the stash node's saved tensors. Gate BCE itself was 2.6ms; the 364ms@B8 / 730ms@B16 was pure recompute | **Move the stash OUT of the checkpoint:** `V13Block.forward` captures the detached gate input `_gate_in_det` (a detached leaf); `V13LM._collect_gate_probs` builds `sigmoid(protect_gate(_gate_in_det))` *after* the block loop. Backprop of the gate loss then needs only `protect_gate` (its input is a leaf) — no checkpoint recompute | **gate overhead 730ms → −0.4ms**: gate-ON step == gate-OFF step == ~21K tok/s. Grads/loss bit-identical (Δ=0.00e+00) |
| 6 | Run killed at data load: `ReadTimeout huggingface.co` | Hub `repo_info`/etag call uses a **10s** default read-timeout (`HF_HUB_ETAG_TIMEOUT`); transient slow response from this box | `HF_HUB_ETAG_TIMEOUT=120`, `HF_HUB_DOWNLOAD_TIMEOUT=300` in launch script (streaming uses the D: hub cache; verified fineweb stream OK at ~39s/first row) | data loader no longer dies on a slow metadata call |

### Verification (all re-runnable, all PASS)

- `v13/tmp/test_fused_delta_equivalence.py` — fused-delta vs K-loop: logits 1.5e-7, grads 3e-8.
- `v13/tmp/test_nll_byproduct_equivalence.py` — NLL byproduct == legacy `linear_ce_per_token`
  (max|Δ|=0.00e+00), gate loss identical, `protect_gate` grads identical (8 params, 4 layers
  × weight+bias), trunk isolation (no non-gate grads from aux), v11 keeps legacy path.
- `v13/tmp/probe_gate_ckpt_cost.py` — under real ckpt: trunk 778ms / gate 778ms / detach 773ms
  @B8 → stash-in-ckpt recompute **364ms → 4.2ms** after the fix.
- `v13/tmp/bench3_trainer_path.py` — trainer path, B16/T2048/gate-ON:
  **B16 20,935 tok/s @14.2GB | B20 20,388 @17.6GB | B24 19,956 @20.9GB** → picked **B16**.

### Current run (fresh start, 2026-08-22)

- **Fresh from step 0** (not a resume): the old `100m_realdat_500m` ckpt (step 500/10.24M tok)
  had its `protect_gate` weights + optimizer state shaped by the *old* gate aux (bug #5 math),
  so resuming would blend two regimes. Deleted (per user). With 9× speed the 10.24M head-start
  was only ~8 min — not worth the contamination.
- `bash v13/tmp/launch_100m_fresh.sh` (tmux `v13_100m`), B16, 500M-token budget,
  pretrain_mix 70/20/5/5/5 (dclm,fineweb,smoltalk2_mid,recall,reason), GPT-2 vocab.
- Loss-flow comparison (fresh-vs-fresh, apples-to-apples): the old v11 best
  (`v11_e3_k3_chat`, HF `qllm-pam-v11-e3k3-chat`, dclm+fineweb 50/50, 10B budget) reached
  loss ~4.0 by ~50M tok and ~3.6 by 1B tok. Fresh v13 at matched token counts tracked as
  it runs (note: v13 mix has 15% smoltalk/recall/reason, so early losses aren't directly
  comparable to v11's dclm+fineweb-only mix — the run-end WikiText-103 PPL is the clean metric).

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
