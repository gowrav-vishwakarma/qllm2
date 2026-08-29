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
**300M verdict PASSED (2026-08-24 ~06:40):** window 298–302M mean **4.08**
vs r1 **3.96** — gap **+0.12** (kill >4.66). Matches the 200M pattern
(+0.13): V13 tracks r1 ~0.1–0.15 NLL behind from ~200M on. Steady ~4,850
tok/s, zero errors. Watchdog armed at 400M (r1 ref 3.83).
- **Throughput — 100M-class (v13_e3_k3_selective, dim384×16L, 4090 24GB):** needs grad-ckpt (B8 no-ckpt OOMs by 2MiB). Steady: **B10 ≈ 2.3K tok/s** (19.8GB); B8 ≈ 2.0K; B12 OOMs on step-2 recompute peak. 11M stays faster per token: **B16 ≈ 11.6K tok/s** (21.2GB). Rule of thumb on 24GB: 11M→B16 no-ckpt; 100M→B10 grad-ckpt.

## 500M r1-recipe run — COMPLETE (2026-08-24 18:54, commit d169584)

`v13_e3_k3_selective`, B8/C128, eager, grad-ckpt ON, fused CE, 4090.
Log `logs/v13/500m_v13_r1recipe/v11_v13_e3_k3_selective_lm_pretrain_mix.log`;
ckpts `checkpoints_v13/500m_v13_r1recipe/{best,final}_model.pt`.

**Endpoint:** 500,000,768 tok in **29.48 h** (106,111 s), avg **4,713 tok/s**.
Val Loss **3.9135** / PPL **50.08** / Acc **0.340**. **Wiki PPL 133.88** (best).

| gtok | v13 window NLL | r1 window NLL | gap |
|---|---|---|---|
| 100M | 4.43 | 4.36 | +0.07 |
| 200M | 4.18 | 4.04 | +0.14 |
| 300M | 4.07 | 3.96 | +0.11 |
| 400M | 3.94 | 3.92 | +0.02 |
| ~500M | **3.87** | **3.79** | **+0.08** |

All kill gates PASSED (band was +0.7). Wiki PPL trajectory:
**325.76@82M → 211.66@164M → 166.75@247M → 149.62@330M → 136.20@413M →
134.03@491.5M → 133.88@500M** — flattening hard over the last 90M.

**VERDICT 1 — token-matched parity, compute-matched loss.** V13 delta ties r1
additive on CE (+0.08 NLL) but at ~3.2× the cost per token (6000 Pro bench:
v13 7.8K vs v11 25K tok/s). Compute-matched is the honest frame and it is
decisive: r1 reached **2B tok in 21.6 h → Wiki 84.57**; v13 reached **500M in
29.48 h → Wiki 133.88**. Both fully-annealed cosine endpoints. For the same
GPU-hours v11 additive sees ~4× the tokens and lands ~37% better Wiki PPL.
**Delta-write alone does not pay for itself.**

**VERDICT 2 — selective stack never woke (6 probes, 82M→500M).**
`dissect_ckpt.py` at step 30000: protect bias [−2.833, −2.672], mean protect
**0.056–0.065** (init 0.047); `phase_proj` wnorm 0.685–1.242 but bnorm
0.0036–0.0105 → phases still ≈0; `write_phase_proj` wnorm 0.070–0.200
(dormant); βw/βe ≈ **0.50** (erase learned ON early, as designed).
CE parity comes from delta writes + CGU, **not** from selectivity.

### Behavioral recall — FIRST measurement on this run

`scripts/run_memory_behavioral.py --model-type v13`, 8-way contrastive,
chance **0.125**, 60 trials/cell, 2160 examples, 0 skipped.
JSON: `logs/memory_probes/v13_500m_r1recipe_FINAL500M_d169584_behavior.json`
(and `..._step30000_...` — the two agree inside noise, max |Δ| 0.022).

| metric | step 30000 | **FINAL 500M** |
|---|---|---|
| overall | 0.2532 | **0.2537** |
| assoc=1 @ ctx128 | 0.7333 | **0.7444** |
| assoc=1 @ ctx512 | 0.3389 | 0.3333 |
| assoc=1 @ ctx1024 | 0.2833 | 0.3056 |
| **assoc=1 @ ctx2048** | 0.2556 | **0.2500** |
| assoc=1 (all ctx) | 0.4028 | 0.4083 |
| assoc=4 | 0.2250 | 0.2194 |
| assoc=8 | 0.1319 | 0.1333 |
| **multi8 @ ctx128** | 0.1389 | **0.1333** |

**VERDICT 3 — best PAM recall ever here, and it came free.** recall@2048
**0.250** with **zero recall data in the mix** (48/48/4 web/chat), beating
every arm of the v11 recall program: Stage-2 control 0.17, Stage-2 gate 0.22,
Stage-3 hypersweep ceiling 0.23, **Stage-6c vault winner 0.189 (trained
*with* 3% synthetic recall)**, v13 e2b_50m 0.083. Still 4× below the matched
Transformer's **0.956**. `assoc=1 @ ctx128 = 0.744` proves the substrate can
store and retrieve a single fact well.

**VERDICT 4 — the bottleneck is write interference, not selectivity and not
context length.** Association scaling collapses to chance independent of
context: **0.408 (a=1) → 0.219 (a=4) → 0.133 (a=8)**, and
**multi8 @ ctx128 = 0.1333 vs 0.125 chance** — 8 facts inside a *128-token*
window are already unrecoverable. Superposing writes in the outer-product
state destroys readout. This reframes the program: stop tuning selectivity
and data mix (measured exhausted in v11 Stage-2/3/6), attack interference.

**Next lever (chosen 2026-08-24):** claim 2 in `r_and_d.md` — in-chunk
raw-key readout. `delta_key_norm` makes retrieval a pure cosine `q·k̂`;
raw keys in the in-chunk `query_key` score restore magnitude contrast.
Well-targeted because multi8@128 with `delta_chunk=128` lives entirely in
the in-chunk path. Falsifier: `multi8@ctx128` must move off 0.133.

## Recall program 2026-08-25→29 — oracle correction → B → D → E → F/F2

The 8-way multi-binding gap (a=8 at chance ~0.133 vs Transformer 0.956) was
root-caused, then attacked with five runs: recall-data (B), dense curriculum
(D), zero-param n-gram fingerprint (E), learned n-gram fusion (F, killed),
learned n-gram fusion fixed (F2). All on `v13_e3_k3_selective` (100.6M,
B8/T2048 eager, lr 3e-4, warmup 500, seed 42, 4090) unless noted.
"all" = allctx = pooled over ctx 128/512/1024/2048 × pos 0/0.5/1.
Battery: `scripts/run_memory_behavioral.py --model-type v13` (8-way
contrastive, chance 0.125; probe vocab disjoint from the training recall
curriculum by design). Deep per-run notes: `SCRATCHPAD.md` sections
2026-08-25 ROOT-CAUSE / B / D / E / F.

**ROOT-CAUSE CORRECTION (2026-08-25, four probes on the 500M ckpt) — the gap
is READ-SIDE ROUTING, not write interference.** Supersedes VERDICT 4 above.
(1) Two-state raw-key readout flip = negative (retrain decision, not a
free toggle). (2) Key-gram probe: fact-key addresses are hyper-orthogonal
(off-diag |k̂ᵀk̂| = 0.0138 = 0.12× random) — the address space is NOT
clustered; the learned query is ≈orthogonal to every address (q·k_target
0.0154 ≈ q·k_other 0.0122) for BOTH a=1 (0.744) and a=8 (0.133) → gap is
dynamics/routing, not address geometry. (3) PAM=0 control: battery
0.254→0.150, a1@128 0.833→0.217 — the PAM path IS engaged (no CGU
shortcut). (4) Oracle readout: build the state normally (writes are
query-independent; ctx128 = single chunk, no carry), re-read with an oracle
query — a=8 seed1002 recovers the value at 11/128 addresses (info IS in the
state), seed1000 0/128, seed1001 128/128 (residual/LM-head case); the
recovering addresses are NOT the value/key/any of the 8 value positions.
**Values ARE stored, as scattered superpositions; the learned query never
learns which address to route to** (the 48/48/4 mix has zero
store-now/answer-later gradient). This re-opened the recall-data lever the
old write-interference read had deprioritized.

**B (2026-08-25→26) — 500M, 4% synthetic recall slice. TIED recall, BETTER PPL.**
r1 recipe + recall slice 48/48/4/4 + `--blend_warmup_tokens 1e7` (the 500M's
1e9 > 5e8 budget made r1 WEB-ONLY forever — zero store/answer signal) +
`--delta_raw_key_readout` + `--delta_erase_beta_cap 1.0`. 500.0M tok /
31.97 h, clean. **Wiki PPL 128.76 vs r1 133.88 (−3.8%)** — the one durable
positive of the program: recall data buys PPL at zero CE cost. 300-trial
final battery: n8-all 0.1453 vs r1 0.1333 (z +1.31), n1 0.3006 vs 0.4083
(z −1.12), n4 0.2042 vs 0.2194 (parity) — direction matches (hard up, easy
down) but NO significance; the 8-way stayed at chance. The slice SHAPE was
wrong for the probe: sparse 3-6 bindings over 2-200 sentences never trained
the probe's dense 8-distinct-bindings-in-128-tokens case.

**D (2026-08-26→27) — 200M, DENSE recall curriculum (same 4% weight). FAIL → BANK C.**
Same as B but the slice reshaped to dense (8 distinct single-token bindings,
0-2 sentence gap, query 1-of-8 back) — trains the probe's exact hard case.
Cache v2→v3. (First launch died silently at 17.2M: host-RAM SIGKILL,
environmental — run 2 crossed the point bit-identical; fixed the watchdog
liveness that had been blind to it: `/proc/PID/exe`-python + cmdline match,
not raw `pgrep -f`, which false-positives on Cursor-sandbox zsh cmdlines.)
Clean to 200.0M / 12.70 h. CE: 4.0971 (r1 3.97 → +0.13, inside kill band).
**Wiki PPL 178.34.** 300-trial gate: n8@128 0.1467 (bar 0.15, 0.2 SE),
**n8-all 0.1367 vs bar 0.205 = FAIL by 3.6 SE** (anchor correction: B-246M
n8-all is 0.1750, not the 0.233 transcribed earlier — that was B-246M's
n4@128 cell; verdict robust to the fix). The dense curriculum did NOT break
the 8-way gap: D-82M n8-all 0.1083 == B-82M 0.1083 exactly.

**N-GRAM INFERENCE-ONLY DIAGNOSTIC on D-final (flag ON, never trained) —
POSITIVE.** n8: +0.011..+0.019 in ALL FOUR contexts (n8-all 0.1367→0.1500,
n8@128 0.1467→0.1589), n1 −0.057 (≈3 SE) — same hard-up/easy-down signature
as B's slice. Suggestive at 1 SE → motivated E.

**E (2026-08-27) — 82M, zero-param n-gram fingerprint TRAINED ON. GATE FAIL, do not scale.**
Motivation: Qwen3.8-Flash-Next (2026-08-26) PLE — hash a 3-gram of token IDs
→ look up an embedding row → augment the representation (51.2B table; their
eval has NO 8-way binding probe, so "fixes multi-binding recall" is a
hypothesis for our battery, not a published result). Port at our budget:
hash into the EXISTING tied embedding table (commit 79db28e) — zero new
params, O(1)/token, `ngram_scale` 0.5 fixed. D recipe + `--ngram_read`,
82M, clean. CE non-regressing (train 4.45@82M on r1 curve; Wiki PPL 345.97 —
the fingerprint costs CE on wiki, pays back nothing on recall). 300-trial
gate bar n8-all ≥ D-final 0.1367 + 0.03 = 0.1667: **E 0.1417 = FAIL** (0.2
SE over D, noise) — and E, TRAINED on the fingerprint, lands BELOW the
inference-only floor (D-on 0.1500): training slightly degraded it. Matched
82M control (60 trials): E n8 0.1417 vs D/B 0.1083 (+0.033, real) BUT
n1 0.1211 vs D 0.1514 (−0.030), n4 0.1475 vs D 0.2139 (−0.066): **a
net-negative SWAP** — the content-blind hash row redistributes recall mass
easy→hard with the cost larger than the gain.

**F (2026-08-28) — learned n-gram fusion block. Run 1 KILLED at 13.7M (defect); F2 FAILED the gate.**
Faithful Qwen PLE port around the same hash lookup (commit 9109fde):
`NgramFusion = depthwise Conv1d(k=3, groups=2d) -> ComplexLinear key_proj
(all four params zero-init) -> ComplexNorm`, injected pre-embed_norm at both
`forward` and `_hidden_to_lm`; O(1) decode via a rolling row buffer
(boundary zero-fill = bit-exact vs parallel). +299,136 params (100.92M).
Zero-init = run starts BIT-IDENTICAL to D; signal supposed to grow with
training. **Run 1 defect (killed step 850):** `ComplexNorm` is
SCALE-INVARIANT (`out = (mag/rms)·scale`); placed AFTER the zero-init
key_proj it amplified the step-1 epsilon to full O(1) — the "slow start" was
a step function (injection max 3.027 after ONE optimizer step; F +1.1..+1.5
NLL vs matched D, flat). **Lesson (general): never put a scale-invariant
norm after a zero-init projection — the norm re-normalizes the slow start
away.** Fix (4fddba2): `key_proj(norm(conv(rows)))` — zero-init projection
stays last (step-0 bit-identity preserved), step-1 injection 2.1e-02 and
grows. **F2 (2026-08-28→29):** 82.0M / 5.18 h, clean. Canary all-nonzero,
step-0 10.9066 (D 10.9055, bit-noise). CE non-regressing the whole run
(matched-delta vs D: +0.07..+0.20 steps 500-750, −0.14..−0.32 at 1200-1550,
noise after; Val 4.4370/84.52, Wiki PPL 262.69). 300-trial gate:
n1-all 0.1475 (PASS ≥ 0.1314), **n8-all 0.1353 (FAIL ≥ 0.1667 by 5.5 SE,
CI [0.1241, 0.1465])**, CE PASS → **FAIL on (1): the n8 ceiling holds even
learned; bank the negative** (pre-registered decision: no 200M scale; a
user-override 200M scale was later queued, see bottom).

### Consolidated recall battery (allctx, 8-way)

| run | tok | t/cell | n1-all | n4-all | n8-all | n8@128 | Wiki PPL | verdict |
|---|---|---|---|---|---|---|---|---|
| r1-FINAL (v13 delta, no recall data) | 500M | 180 | 0.4083 | 0.2194 | 0.1333 | 0.1333 | 133.88 | reference |
| B (4% recall slice) | 82M | 180 | 0.1486 | 0.2069 | 0.1083 | 0.1056 | — | inconclusive @82M |
| B | 246M | 180 | 0.3042 | 0.2083 | 0.1750 | 0.1444 | — | positive trend (n8@128) |
| B-FINAL | 500M | 900 | 0.3006 | 0.2042 | 0.1453 | 0.1367 | **128.76** | TIED recall, BETTER PPL |
| D (dense slice) | 82M | 180 | 0.1514 | 0.2139 | 0.1083 | 0.1056 | — | = B @82M exactly |
| D-FINAL | 200M | 900 | 0.1956 | 0.1625 | 0.1367 | 0.1467 | 178.34 | FAIL (bar 0.205) → BANK C |
| D-FINAL + ngram (inference-only) | 200M | 900 | 0.1833 | 0.1556 | 0.1500 | 0.1589 | — | +0.013 n8 all-ctx, n1 −0.057 |
| E (ngram trained) | 82M | 900 | 0.1211 | 0.1475 | 0.1417 | 0.1400 | 345.97 | FAIL — net-negative swap |
| F2 (learned fusion) | 82M | 900 | 0.1475 | 0.1578 | 0.1353 | 0.1356 | 262.69 | FAIL — ceiling holds learned |
| Transformer (matched) | — | — | — | — | **0.956** | — | — | the target |

**THE CEILING (four independent readings):** D inference-only 0.1500, E
trained 0.1417, F2 trained 0.1353, D-200M-off 0.1367 — all within ~1 SE.
Content-aware n-gram fingerprinting (zero-param OR learned) lifts the hard
8-way ~+0.03 at matched size but caps at the dense-curriculum ceiling; it
does NOT close the routing gap. F2 is a strictly better swap than E (n1 tax
gone: 0.1475 ≈ D 0.1514, where E was 0.1211) — the learned block suppresses
the fingerprint where it isn't useful — yet still lands at the ceiling.
B's PPL win (−3.8%) is the bankable asset; the n8 gap is read-side routing
per the oracle, untouched by every write-side/fingerprinting lever tried.

### Lessons (permanent)

- **n8@128 = 0.1056 for both B-82M and D-82M (60t):** at 82M neither
  curriculum had touched the probe case; B's n8@128 rise (0.106→0.133→
  0.144) came at 164-246M and then plateaued; D's dense shape never
  produced it at 200M.
- **Slow-start vs scale-invariant norms:** see F run 1 above. Selftest
  `test_ngram_fusion` + the GPU smoke now assert the slow-start bound
  (step-1 injection < 0.5); a step-0 bit-identity test alone cannot catch
  this — it only checks step 0.
- **Watchdog liveness:** never `pgrep -f` alone (sandbox-shell false
  positives); require `/proc/PID/exe` = python + cmdline match (fixed in
  v13/tmp/watchdog.sh, 2026-08-26).
- **Post-budget trainer hang:** the F2 trainer held 9 GB of GPU ~2 h after
  `Wall clock end` (killed after ckpts confirmed). If it recurs, add a
  post-budget timeout to the launcher.
- **Anchor discipline:** the D gate bar's "B-246M (0.233)" was a
  transcription slip (real: 0.1750 n8-all; 0.233 = B-246M's n4@128 cell).
  Verify gate anchors against the on-disk JSON before launch.

### Next (post-F2, 2026-08-29)

Per the oracle, the untried lever is READ-SIDE: (1) **fact_contrastive
value-ranking loss** — ported (9e73e7b), needs per-token value masks
threaded through `_build_recall_doc` → blend → cache v4 → dataset → batch
(trainer branch already at v7/train.py:534-541, guarded by mask presence);
forces the correct value to outrank the 7 sibling answer tokens = exactly
the probe's discrimination. (2) **gamma_floor** memory horizon (v11 0.98;
v13 default 0). (3) F2 scale to 200M (user override of the pre-registered
bank-the-negative call).
