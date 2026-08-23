# V13 experiments — selective PAM (delta + vault + phase addressing)

Lab notebook for V13: defaults make Stage-6 recall levers the production path.

## Speed redesign 2026-08-22 — 100M-class on 4090 (measured on 4090)

**Goal:** 100M-class V13 (`v13_e3_k3_selective`, 100.6M, dim384×16L, K=3 selective PAM)
training "like a transformer — minutes not hours". Before: **2.3K tok/s** (B10).

**⚠️ 2026-08-23 RETRACTION:** the **~21K tok/s (B16)** / "9× speedup" figure was
measured with `V13LM._ckpt_block` detaching the block input (commit d0abeed).
That skipped backward through 15 of 16 layers. Honest eager baseline after
the grad-flow fix (`baaf5b3`, all 16 blocks learn): **4,101 tok/s** at
B16/C128 (13.9GB) and **5,027 tok/s** at B8/C128. Per-block
`torch.compile` inside the checkpoint measured **6,459 tok/s** at B8
(loss identical to 1e-6). See "Grad-flow fix" below. The fused-delta /
gate-stash / NLL-byproduct changes below are still real and still math-exact;
they just were never the source of a 9× wall-clock win on a correct backward.

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
  **RETRACTED 2026-08-23:** those numbers include the `_ckpt_block` detach
  (15/16 blocks skipped backward). Honest post-fix numbers are in
  "Grad-flow fix" below.

### 500M NaN crash — root cause + robust fix (2026-08-23, commits 04dcebd + 03c3ede)

**Symptom:** the 500M mission run died at step ~1551 (57.2M tok) on a device-side
assert `Loss.cu:91 'target_val >= zero && target_val <= one'` inside
`F.binary_cross_entropy` in the GATE-SURPRISAL aux — the main loss was still
finite and descending (5.39 @ step 1550), so it was a sudden per-token NLL
spike to inf/NaN that the per-step mean masks and only the gate-BCE
target-assert surfaces.

**Mechanism (confirmed by `v13/tmp/diag_gate_nan.py`, which catches non-finite
in Python before the BCE kernel):** the V13 delta rule is
`S ← γ·S + (βw·v − βe·(k@S))·k^H`. Its eigenvalue in the key `k` direction is
`γ − βe·‖k‖²`. With the **vault** state (γ≡1) and raw unnormalized keys
(qk_norm off, ‖k‖² up to ~d=64), this flips past −1 → positive feedback →
unbounded PAM state (diag: 0.35 @ s20 → 38 @ s990 → NaN @ s1711) while params
stay flat (pmax ~4) and grads clip fine. The gate aux is only where the NaN
*asserts*, not where it originates.

**Fix 1 (key-norm, 04dcebd) was necessary but INIT-DEPENDENT:** per-VECTOR
unit-norm keys (`cnormalize_vec`, applied in `_project` after RoPE+phase) make
‖k‖²=1 → eigenvalue exactly `1 − βe`, stable only while the *learned* erase
gain βe < 2. The diagnostic (no wiki-val RNG draw) stayed βe<2 → passed 2000
steps, state bounded ~17. But the REAL trainer loads `wiki_val_ds` BEFORE
building the model → a different weight init → trained βe past 2 → eigenvalue
flipped past −1 → the fixed trainer died EARLIER, at step ~151 (5.6M tok).

**Fix 2 (robust, 03c3ede): `delta_erase_beta_cap = 0.95`** (V13Config default
ON). Clamp the learned erase gain in `_gate_betas` (the single choke point
feeding the fused-delta, K-loop, and recurrent delta paths) so the eigenvalue
stays in [0.05, 1) for ANY init. The erase strength is still learned (sigmoid
0.047..0.95); only the pathological >2 overshoot is removed. **Safety net** in
`v7/train.py:_gate_surprisal_loss` (shared with v7): a rare non-finite NLL now
droops the token from the median+mask and `nan_to_num`s the target to [0,1]
(BCE runs over the full [L,B,T] before vmask is applied, so even masked
positions assert; clamp alone leaves NaN). Logs events, does not silence the
aux.

**Verification (all re-runnable):**
- `v13/selftest` — ALL MODES PASS, incl. new `[delta_erasecap]` (βe clamped to
  0.95 when raw sigmoid saturates at 1.0; write path untouched). (Also fixed a
  latent bug: `test_warmstart_chatml` returned None → false "SOME MODES FAILED".)
- `v13/tmp/test_gate_nonfinite_guard` — 6/6: a +inf/−inf/NaN NLL (the exact
  crash condition) yields a finite aux loss + a logged event, never an assert.
- **Real-trainer repro** (`launch_v13_500m_r1recipe.sh --token_budget 20M`):
  the old key-norm run died at step 150 (loss 7.9623, gtok 5566464); with the
  cap the trainer PASSED that exact point and ran clean to step 488 (20M, val
  6.26) — **ZERO asserts, ZERO safety-net events** (the cap alone fixed it).
500M relaunched fresh 2026-08-23 02:36 (tmux `v13_500m`, ckpt dir wiped).

### Current run (fresh start, 2026-08-22)

- **Fresh from step 0** (not a resume): the old `100m_realdat_500m` ckpt (step 500/10.24M tok)
  had its `protect_gate` weights + optimizer state shaped by the *old* gate aux (bug #5 math),
  so resuming would blend two regimes. Deleted (per user). With 9× speed the 10.24M head-start
  was only ~8 min — not worth the contamination.
- **[SUPERSEDED — stopped 2026-08-22, see RESOLVED below.]** `bash
  v13/tmp/launch_100m_fresh.sh` (tmux `v13_100m`), B18, **lr 3e-4, warmup 2000**
  (the exact recipe of the HF-uploaded v11 best), 500M-token budget,
  pretrain_mix 70/20/5/5/5 (dclm,fineweb,smoltalk2_mid,recall,reason), GPT-2
  vocab, gen_every=5000, save_every_steps=5000 (overwrite latest.pt).
- **[Superseded run's LR finding]** the first fresh attempt (v13-program default
  lr 1e-4) lagged the v11 curve ~2.5 NLL @16M (9.20 vs 6.71) — that gap was lr,
  not arch. (The follow-up lr-3e-4/warmup-2000 run then exposed the fused_ce
  bug; the current run uses round-1's warmup 500 — see RESOLVED below.)
- **RESOLVED (2026-08-22 20:30 IST): the ~3 NLL gap was a CODE BUG, not the
  delta stack.** `v13/fused_ce.py` `_FusedLinearCE.backward` was missing
  `grad_weight += (softmax_probs.T @ hidden_chunk)` — the Aug-22 16:17
  `return_nll`/autocast rewrite (commit 65546dc) dropped the line that
  `v11/fused_ce.py` always had. The tied embedding/LM-head got ZERO CE
  gradient; only the trunk learned, so loss stalled ~2-3.5 NLL above the
  round-1 reference with a noisy floor. Forward was bit-correct (step-0 loss
  identical, all forward A/B tests passed) — a backward-only regression.
  Fixed in commit 79c7cc2; verified by `v13/tmp/test_fused_ce_grads.py`
  (grad_weight rel-L2 5.8e-6, was 1.0) + `v13/selftest` (ALL PASS) + the
  ab1 canary run (10.00@100 steps vs r1 8.81 → now gone).
  Affected: every `--fused_ce` v13 run started after ~16:17 Aug 22
  (100m_realdat_500m_fresh, diag_additive, ab1). NOT affected: e2b_50m
  (started Aug 19, pre-bug; its head trained fine, loss→0.36) so the
  e2b_50m vs transformer_50m quality comparison remains VALID.
- **Note on references:** the Jun-23 10B log used above is OLD-code
  (pre-rewrite) and was a confounded reference. The correct baseline is
  **round-1 (Jul 1, new code)**: `--preset v11_e3_k3_chat --warmup 500
  --lr 3e-4 --batch_size 18 48/48/4 dclm/fineweb/smoltalk2_mid edu>=3
  sample-10BT blend 1e9`, curve 7.52@5M, 5.87@20M, 4.81@50M, 4.36@100M.
- **Current run (2026-08-22 20:01 IST, FIXED code):**
  `v13/tmp/launch_v13_500m_r1recipe.sh` (tmux `v13_500m`), full V13 delta
  stack (K=3, delta, vault, phase, λ0.1) + round-1 EXACT recipe (warmup 500,
  48/48/4, edu3, sample-10BT, blend 1e9, seed 42), 500M budget, ~21K tok/s.
  Early read: 10.36@2M vs r1 10.31 (+0.05), 7.96@5.6M vs r1 7.52 (+0.44) —
  on curve. Verdicts: 5.87@20M (kill if >6.6), 4.81@50M, 4.36@100M.

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
- **Grad-ckpt detach (RETRACTED, was the 2026-08-22 "fix"):**
  `z_leaf = z_in.detach().requires_grad_(True)` inside `_ckpt_block` did stop
  the determinism_check crash, but it **silently froze every block except the
  last** on the main loss (non-reentrant checkpoint needs the input edge).
  Removed in `cbd35d4`. The crash's real cause was TorchScript on
  `cnormalize_vec` (see Grad-flow fix). NEVER re-introduce a detach.

### Grad-flow fix (2026-08-23, commit baaf5b3)

**Symptom:** with the detach removed, `loss.backward()` raised
`CheckpointError` (saved `[B,H,T,1,1]` vs recomputed `[B,H,T,d,2]` at the
`cnormalize_vec` div). Production 500M cannot run without checkpointing
(`--no_grad_ckpt` OOMs).

**Cause:** `@torch.jit.script` on `cnormalize_vec`. TorchScript's profiling
executor swaps the two `div` operands between the checkpoint forward and the
recompute. Deterministic on a cold process (15/15 crashes); a warm JIT
process hides it (the "flaky 1-in-5" note was wrong). `PYTORCH_JIT=0` → 0
crashes; unscripting only this one fn → 0 crashes.

**Rejected "fix":** an `autograd.Function` that returns `g/mag` (19% relative
error vs the true Jacobian `(I − x̂x̂ᵀ)/‖x‖`). Would have corrupted key grads.

**Applied fix:** drop `@torch.jit.script`, keep the eager body. Autograd
computes the exact Jacobian; graph-build order is identical across passes.

**Verified:** `dbg_ckpt_probe3` 0/30 fresh-process crashes; production
`v13_e3_k3_selective` 16/16 blocks ~6.4e-3 grad, 0 params without grad;
`v13/selftest` ALL PASS including `[grad_ckpt equiv]` (dloss=0, worst_rel=0).

**Honest speed (4090, T=2048, all 16 blocks learning):**

| config | tok/s | peak |
|---|---|---|
| B16 C128 eager | 4,101 | 13.9 GB |
| B8 C128 eager | 5,027 | 7.8 GB |
| B8 C256 eager | 5,085 | 9.5 GB |
| B8 C128 compile-block | 6,459 | 7.4 GB |

`--compile_blocks` compiles each `V13Block.forward` *inside* the checkpoint
(vs `--compile`, which wraps `_hidden_to_lm` and cannot fuse a block).
`--delta_decay_factored` (default OFF) is a K-independent rewrite of the
chunk solve; `[delta_factored]` matches to 1e-8 / 1e-10. Enable only after
a bench shows a tok/s win. Trainer logs `[block-grad step1]` after the
first backward so a frozen-layer regression cannot hide again.

**500M relaunch (2026-08-23 13:26, LIVE):** old 02:28→09:20 run in
`checkpoints_v13/500m_v13_r1recipe` was the buggy-code run (20,684 tok/s avg —
the retracted detach figure; train-loss floor ~4.65, Wiki PPL 368.69 — VOID,
user-confirmed). Ckpt dir wiped; logs preserved as `*_compile_crash.log` in
`logs/v13/500m_v13_r1recipe/`.

Launch 1 with `--compile_blocks --batch_size 8 --delta_chunk 128` CRASHED at
first step (Inductor meta-kernel bug, not our code):
`assert_size_stride(buf113, (144, 128, 64), ...)` on
`torch.ops.aten.complex.default` inside the compiled block — the inductor
layout for a complex buffer disagrees between meta and real execution.
Parked (speed track, see scratchpad SPEED). Relaunched EAGER:
`launch_v13_500m_r1recipe.sh --batch_size 8 --delta_chunk 128` (tmux
`v13_500m`). `[block-grad step1]` L0..L15 ≈ 2.3e-3..5.8e-3 all-nonzero ✓;
steady ~4,930 tok/s @ 8.1GB (matches the 5,027 bench); step 0 loss 10.9055,
10.36 @ 0.84M (r1 10.31 @ 2M — on curve). Watchdog armed at 100M verdict.
Verdict points: kill if >0.7 NLL above r1 at 20M (>6.6); 4.81@50M, 4.36@100M.

**20M verdict PASSED (2026-08-23 ~14:40):** 5.46 @ 20.1M (threshold >6.6; r1
interp ~6.1 → V13 ~0.6 NLL *below* r1). Steady ~4,850 tok/s, zero errors.
Curve: 10.36@0.84M, 5.89@9.8M, 5.15@26.2M, 5.07@29.9M.
**50M checkpoint (2026-08-23 ~16:20):** 4.65 @ 50.8M — r1 ref 4.81@50M,
V13 ~0.16 NLL below r1, steady ~4,840 tok/s, zero errors.
**100M verdict PASSED (2026-08-23 ~19:10):** 4.36 @ 100.4M vs r1 4.36@100M
— gap 0.0 (kill was >5.06). generate() @82M ckpt: coherent, no repetition
loop, factually garbled (expected at 82M).
**200M verdict PASSED (2026-08-23 ~01:15):** window 198–202M mean **4.17**
vs r1 **4.04** (official log `round1_pretrain_20260701…pretrain_mix.log`)
— gap **+0.13**, well inside the 0.7 kill band (kill >4.74). The earlier
"V13 leads (~3.95)" was a single low-noise step (11850); window mean is the
fair figure. r1 reference extends to 2B: 3.96@300M, 3.92@400M, 3.77@500M
(window means). Steady ~4,830 tok/s, zero errors.
- **Throughput — 100M-class (v13_e3_k3_selective, dim384×16L, 4090 24GB):** needs grad-ckpt (B8 no-ckpt OOMs by 2MiB). Steady: **B10 ≈ 2.3K tok/s** (19.8GB); B8 ≈ 2.0K; B12 OOMs on step-2 recompute peak. 11M stays faster per token: **B16 ≈ 11.6K tok/s** (21.2GB). Rule of thumb on 24GB: 11M→B16 no-ckpt; 100M→B10 grad-ckpt.
