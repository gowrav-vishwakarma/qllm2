# V13 SCRATCHPAD — read this FIRST after any context summary

## ⚠️ ACTIVE TASK (2026-08-23) — HANDOFF: fix the gradient-checkpoint RECOMPUTE MISMATCH
**Status: root cause PINPOINTED, fix NOT yet applied. This is a fresh, self-contained
battle — read this section fully before touching anything.**

### Symptom (what blocks production training)
With `gradient_checkpointing=True` (non-reentrant, `use_reentrant=False`) — the ONLY
VRAM-fitting path for the 500M run (`--no_grad_ckpt` OOMs at B18/T2048 bf16, forward
alone 22.2GB) — `loss.backward()` intermittently crashes with:
```
torch.utils.checkpoint: Recomputed values for the following tensors have different
metadata than during the forward pass.
tensor at position 85:
  saved metadata:     {'shape': torch.Size([2, 2, 128, 1, 1]), 'dtype': torch.float32}
  recomputed metadata:{'shape': torch.Size([2, 2, 128, 32, 2]), 'dtype': torch.float32}
tensor at position 86: (the reverse)
```
→ `CheckpointError`. Tiny repro (1 layer, B2/T128, fp32) crashes in ~2s.

### ⚠️ KEY FACT: IT IS FLAKY — do NOT trust single runs
The crash fires on **~1 of 5** fresh forward+backward passes, and **tends to hit the
FIRST run of a fresh process** (CUDA lazy-init / allocator-state dependent). I ran:
- `dbg_ckpt_flaky 5` → original code: `1/5 crashes` (the first run of the process);
  and the 5 bisection toggles each `0/5`.
- `dbg_ckpt_fixtest2 15` → original code: `0/15` (warm process, no crash).
**Consequence: any single-run "PASS" is meaningless, and the earlier single-run
bisection conclusions (in my notes) are UNRELIABLE. Always test with many fresh
processes (e.g. loop `for i in $(seq 30); do .venv/bin/python -m v13.tmp.dbg_ckpt_probe3 1; done`
and count how many crash).** The crash reproduces reliably on the first run of a
cold process.

### Root cause (PINPOINTED — identical values, order-swap of 2 tensors)
The swapped pair is EXACTLY the two operands of the division inside `cnormalize_vec`:
```
v13/complex_ops.py:95-105   (called from v13/model.py:391)
  @torch.jit.script
  def cnormalize_vec(x):
      mag = torch.sqrt((x[...,0].square() + x[...,1].square()).sum(-1) + 1e-8)
      return x / mag.unsqueeze(-1).unsqueeze(-1)      # <-- the div
```
The `div` saves its two operands for backward:
- numerator `x` = `keys` → shape `[B,H,T,32,2]`
- denominator `mag.unsqueeze(-1).unsqueeze(-1)` → shape `[B,H,T,1,1]`
In the checkpoint **forward** the two are saved in order `(denom, numer)`; in the
**recompute** they are saved in order `(numer, denom)`. Same values, reversed slots →
positional metadata check fails. **Everything else matches** (positions 79–84 and 87–91
all identical) → it is a LOCAL 2-swap, not a global index shift, and not a value
difference. No grad-mode branch: both passes run with `requires_grad=True,
is_grad_enabled=True, is_inference_mode=False`.

Why torch compares positionally (so the swap is fatal) — from
`.venv/lib/python3.9/site-packages/torch/utils/checkpoint.py`:
- Non-reentrant forward runs under `_checkpoint_hook` (a `saved_tensors_hooks`, grad
  ENABLED): each saved tensor → appended to `weak_holders[i]` + `x_metadatas[i]`
  (order = graph-build order).
- Recompute runs under `_recomputation_hook` + `torch.autograd.enable_grad()`: each
  saved tensor → mapped to `weak_holders[recomp_counter]` in RECOMPUTE graph-build
  order, stored in `recomputed[gid]`.
- `_CheckpointFrame.check_recomputed_tensors_match` compares `x_metadatas[i]` vs the
  i-th recomputed tensor **position-by-position**. So the i-th save in the forward MUST
  be the same tensor as the i-th save in the recompute. A 2-swap in one op breaks it.

### What was tried (so you don't repeat it)
- **Bare `cnormalize_vec` div under `ckpt.checkpoint` in a standalone script
  (`dbg_ckpt_minrep.py`) did NOT crash** — but that script ran the checkpoint OUTSIDE a
  real grad context (block input had no `grad_fn`) and its recompute-capture was broken
  (`rec=[]`). So it is NOT a valid repro and its "fix candidates all pass" is
  CONCLUSIVE-NEGATIVE (nothing). Do not trust it.
- **Fix formulations tested on the REAL model, ONE run each (`dbg_ckpt_fixtest.py`):
  ALL "passed"** — meaningless because of the flakiness (1-run each). Same for
  `dbg_ckpt_fixtest2.py` (15 runs, warm process, 0 crashes across all).
- **The earlier config-bisection** (turning off write_phase_address / delta_key_norm /
  use_rope / fused_qkv each → "PASS") was SINGLE-RUN and is therefore UNRELIABLE.
  `delta_key_norm=False` DOES remove the only `cnormalize_vec` call in the delta path,
  so it is a plausible workaround (re-verify with fresh processes if you want a
  stopgap), but the real fix keeps the feature on.
- **A minimal standalone repro is not yet built correctly.** To repro you must run the
  checkpoint so the block input has a non-None `grad_fn` (i.e. the input is a
  grad-requiring node of a larger grad-enabled graph, e.g. `x = torch.randn(...); x =
  x*1.0` under `torch.enable_grad()`), AND capture the recompute side correctly (patch
  `_CheckpointFrame.check_recomputed_tensors_match` to read `self.recomputed`, not a
  forward-only `saved_tensors_hooks`).

### PROPOSED FIX (not yet applied) — make the norm checkpoint-safe
Replace the bare `div` (whose 2-operand save order is context-dependent) with an
explicit `torch.autograd.Function` that saves ONLY `mag` (a single tensor, fixed
order — a single save cannot swap with anything):
```python
# v13/complex_ops.py — replace cnormalize_vec body
class _CNormalizeVecFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + 1e-8)
        out = x / mag.unsqueeze(-1).unsqueeze(-1)
        ctx.save_for_backward(mag)          # single tensor — no 2-operand swap possible
        return out
    @staticmethod
    def backward(ctx, g):
        (mag,) = ctx.saved_tensors
        # d(out)/d(x) = 1/mag ; mag is derived (no independent grad path) -> only return gx
        return g / mag.unsqueeze(-1).unsqueeze(-1)

def cnormalize_vec(x):
    return _CNormalizeVecFn.apply(x)
```
Math is unchanged (same `out`, same `dL/dx`). Caveat: `cnormalize_vec` is currently
`@torch.jit.script`; an `autograd.Function` is NOT scriptable. The call site
(`v13/model.py:391`, inside `V13PAMLayer._project`) is NOT itself scripted, so dropping
the decorator is very likely safe — but GREP all uses of `cnormalize_vec` first and
confirm nothing `torch.jit.script`/`torch.jit.trace`s a path through it. If scripting IS
required, the alternative is to keep a scripted eager body but route the div through a
formulation whose backward saves a fixed-order tensor set (test empirically, see below).

### VERIFICATION PROTOCOL (do this, in order)
1. **Repro the crash first** (prove the bug is live before you "fix" it):
   `.venv/bin/python -m v13.tmp.dbg_ckpt_probe3 1` — run it in ~10 FRESH processes and
   confirm it crashes (expect ~1 in 5, reliably on cold process). Record crash count.
2. **Apply the fix** to `v13/complex_ops.py`.
3. **Re-run the same fresh-process loop**: crash count must go to 0 (run ≥30 fresh
   processes to be confident given 1/5 baseline).
4. **Per-layer gradient check** (the user's secondary question — "did only the last two
   layers get gradient?"). Once backward no longer crashes, run
   `.venv/bin/python -m v13.tmp.dbg_ckpt_probe2 4` (or add a quick loop over
   `m.blocks[i].parameters()` printing `p.grad.norm()` per block). If blocks 0..N-2 now
   all have non-zero grad, the "only last two layers" observation was the d0abeed
   detach regression (already removed — see `_ckpt_block` docstring model.py:1566-1573)
   or predates it. Report the per-block norms.
5. **Scale check**: one real training step of the production config
   (preset `v13_e3_k3_selective` or the 500M recall config) at real batch/seq with
   `gradient_checkpointing=True` → no `CheckpointError`.
6. **Clean up** `v13/tmp/dbg_ckpt_*.py` probes once the fix is verified (do NOT delete
   before step 5 passes). Record the fix rationale in a dated comment in
   `complex_ops.py`/`model.py` (match the existing date-stamped "why" note style).

### HARD CONSTRAINTS (do not violate)
- **Keep non-reentrant gradient checkpointing ON.** `--no_grad_ckpt` OOMs at scale.
- **NEVER detach the checkpointed block input** in `V13LM._ckpt_block` (model.py:1565-
  1576). Commit d0abeed detached it as a "determinism_check workaround" and that
  SILENTLY FROZE every block except the last on the main loss — the docstring records
  this; it is the leading explanation for the user's "only last two layers got
  gradient" observation. Do not re-introduce any detach/skip.
- Keep `delta_key_norm=True` (required for delta-rule stability; the 2026-08-23
  eigenvalue fix depends on unit-norm keys). The fix must keep this feature on.
- No new dependencies; venv python3.9 at `.venv/`; torch has `torch.utils.checkpoint`.
- After a verified fix: git commit (what+why+verification) per the repo commit rule.

### Probe files (all in `v13/tmp/`, run with `.venv/bin/python -m v13.tmp.<name>`)
- `dbg_ckpt_probe3.py [n_layers] [window]` — **THE repro + aligned saved-vs-recomputed
  window with value checksums.** Patches `ckpt._allowed_determinism_checks_to_fns['default']`
  (MUST patch the dict entry, not a name) + `ckpt._CheckpointFrame.check_recomputed_tensors_match`.
  Prints the mismatch window (shapes + cksums + rg/ge/ie). Run in a fresh process.
- `dbg_ckpt_flaky.py [n]` — runs original + bisection toggles n×, reports crash counts
  (proves the flakiness).
- `dbg_ckpt_probe2.py` — per-block grad-norm probe (for step 4 above).
- `dbg_ckpt_bisect.py`, `dbg_ckpt_trace.py`, `dbg_ckpt_minrep.py`, `dbg_ckpt_fixtest.py`,
  `dbg_ckpt_fixtest2.py`, `dbg_ckpt_stream.py`, `dbg_ckpt_mismatch.py`, `dbg_ckpt_isolate.py`
  — earlier explorations. NOTE: `dbg_ckpt_stream.py`'s `first_diff_at` numbers are
  UNRELIABLE (it skips GC'd weak-holders, shifting indices; 551 fwd vs 535 rec). Use
  probe3 for the trusted positional view.

### Tiny repro config (for any probe)
```python
V13Config(vocab_size=50257, dim=64, n_heads=2, head_dim=32, n_layers=1, expand=2,
    dropout=0.0, max_seq_len=256, chunk_size=64, gradient_checkpointing=True,
    n_states=3, state_dt_spread=2.0, write_mode='delta', delta_chunk=32,
    delta_erase_gate=True, gate_content_aware=True, vault_state=True,
    vault_state_idx=0, write_phase_address=True, fused_e3=True,
    gate_surprisal_lambda=0.1, delta_key_norm=True, delta_erase_beta_cap=0.95)
m = V13LM(cfg).cuda(); m.train()
ids, lab = torch.randint(0,50257,(2,128),device='cuda'), torch.randint(0,50257,(2,128),device='cuda')
lm,_,_ = m._hidden_to_lm(ids); loss = m.ce_from_lm(lm, lab, chunk=4096); loss.backward()
```


**Mission (user, 2026-08-22):** Make V13 (100.5M-param selective-PAM: complex
embeddings, K=3 phase-addressed SSM states, delta-write + vault, GSP protect
gate) mature, better, faster. Beat v11-best quality at matched tokens on
**500M real tokens**, speed-first, O(1) inference (no KV cache). User is away
**MISSION BAR (user clarified 2026-08-23):** NOT apples-to-apples same-dataset.
V13 is trained on RICH REAL data (DCLM 48 + FineWeb 48 + smoltalk2_mid 4, 500M
tokens) and must land WikiText-103 val PPL **< 25.77** (V11 E3 K=3, WikiText-only
base, the README current-best) — ideally much lower, toward/below the transformer
anchor 22.69. Also: better reasoning/maths, keep O(1) novel, ~21K tok/s, keep the
ablation loop (protect_gate_bias, gate-surprisal λ, key-norm/erase-cap A/Bs),
commit every verified change. The 50M/100M "verdicts" vs the r1 TRAIN-loss curve
are secondary; the primary bar is the WikiText val PPL number after 500M.
**WAKE PROTOCOL (CRITICAL — every time I wake, do ALL of these):**
1. Read this file (v13/SCRATCHPAD.md) fully.
2. Check `tmux ls`, running procs (`pgrep -af "v1[13].train"`), GPU, and the
   active log's last steps + error count.
3. RE-ARM THE WATCHDOG (the chain that wakes me):
   `bash v13/tmp/watchdog.sh <active_log> <verdict_gtok> 2940` as
   `async: true` + **timeout: 3300** (the timeout param is MANDATORY —
   default 300s kills the job at 5min and breaks the chain; this happened).
   The watchdog exits early on: process death / OOM / verdict gtok, and its
   auto-delivery is what wakes me. Keep the chain alive until V13 is done.
4. COMMIT RULE (user, 2026-08-22): after every verified code change in
   v11/v13/v7/scripts → git commit immediately (what+why+verification).
   Never leave good code uncommitted (see AGENTS.md).
5. Decide the next action from STATUS + NEXT, execute it, then yield only
   with the watchdog armed.
~2 days. **Kill & iterate**: if a run is way off the v11-best curve at ~20-50M
tok, kill it and iterate. Novelty: NOT transformer/Mamba re-skin.
- [CONFIRMED, 2026-08-22] **v13 fork = v11, BIT-IDENTICAL on additive path.**
  `v13/tmp/test_v11_v13_forward_ab.py`: same weights, same inputs → 0.000e+00
  max-abs diff on all 16 block hiddens, final logits, AND CE loss. NLL
  byproduct shape OK. ⇒ NO v13-fork code regression. The "3 NLL worse than
  v11" was (1) stale Jun-23 OLD-code reference + (2) warmup-2000 recipe.
- [CONFIRMED] round-1 (the real baseline) ran on a **75GB GPU** with
  --no_grad_ckpt. 4090 can't fit that; ab1 uses GC-on (same math, 11.8GB).
- [ROOT CAUSE FOUND + FIXED 2026-08-22] `v13/fused_ce.py` `_FusedLinearCE.backward`
  allocated `grad_weight` but NEVER filled it — the v13 fork dropped the
  `grad_weight += (softmax_probs.T @ hidden_chunk).to(grad_weight.dtype)` line
  that `v11/fused_ce.py` has. Tied embedding/LM-head got ZERO CE gradient;
  only the trunk learned ⇒ loss stalled ~2-3.5 NLL above r1 from step ~50.
  Forward was correct (step-0 loss identical), so forward A/B tests all passed.
  FIXED (line restored in v13/fused_ce.py:96). Verified:
  `v13/tmp/test_fused_ce_grads.py` grad_weight rel-L2 5.8e-6 (was 1.0);
  `v13/selftest` ALL PASS incl. fused_ce equiv.
- [KILLED 2026-08-22] A/B "ab1" (v13 additive, v11-features off, round-1 EXACT
  recipe): DIVERGED IMMEDIATELY — step 50 10.44 vs r1 10.31, step 100 10.00 vs
  8.81, step 250 9.33 vs 6.66 at IDENTICAL lr (warmup 500) + data; step-0 loss
  identical (10.8986). At 55M: 8.34 vs r1 ~4.8 (3.5 above). This was the
  canary that exposed the fused_ce bug above (it used --fused_ce).
- [CONFIRMED] fused_e3 additive path = bit-equivalent to K-loop (1e-8) — not
  the cause.
- [DEAD 2026-08-22 ~20:50 UTC] **V13 500M MISSION run DIED at step ~1551 /
  57.2M tok** on a device-side CUDA assert `Loss.cu:91 'target_val >= zero &&
  target_val <= one'` inside `F.binary_cross_entropy` in the GATE-SURPRISAL aux
  (v7/train.py:470, fused_ce path). No `latest.pt` (save_every 5000, died at
  1551; checkpoint dir EMPTY). GPU now free. NOT a plateau — the main loss was
  finite 5.39 @ step 1550 and STILL DESCENDING; the crash is a sudden per-token
  NLL spike to inf/NaN that the per-step mean masks and only the gate-BCE
  target-assert surfaces (CE has no such assert).
  Mechanism (root cause, 2026-08-22): the gate aux computes
  `target_p = sigmoid(sign*(median_ce - nll)/tau).detach()`. A NaN/inf in `nll`
  (per-token NLL byproduct) → `sigmoid(median - inf)=NaN` target → Loss.cu:91.
  `nll` goes inf/NaN when hidden→logits overflow. v13's delta-write + VAULT
  state (decay_gamma≡1, line 408) + qk_norm=False (unbounded keys) make the
  parallel triangular-solve mass matrix M[t,s]=beta_e*decay[t,s]*(k_t·k_sᴴ)
  unbounded → (I+M)⁻¹ amplifies ~T·‖k‖² → state update → hidden blowup.
  v11 (reached 2B) has NONE of {delta, vault, phase, gate-aux} → stable.
  Diagnostic `v13/tmp/diag_gate_nan.py` (exact 500M recipe, catches non-finite
  in Python BEFORE the BCE kernel) replaying to the crash point in tmux `diag`;
  log `logs/v13/500m_v13_r1recipe/diag_gate_nan.log`. Confirms WHICH quantity
  (state vs nll_max vs gp) goes non-finite + at what step/magnitude.
- [ROOT CAUSE CONFIRMED + FIX IMPLEMENTED 2026-08-23] Diag REPRODUCED the
  crash: NON-FINITE at step 1711 (63.1M tok; real run died 1551/57.2M — ~160
  step offset = dropout/GC RNG-stream difference, same mechanism). State
  trajectory: PAM state magnitude grows ~100x UNBOUNDEDLY while params stay
  flat (pmax~4.0) and gnorm clips fine: s20 0.35 → s150 12.6 → s300 7.5 →
  s990 29-40 → s1711 NaN. gp finite [0.02,0.98]; nllmax elevated 15-21.
  MECHANISM (v13 delta rule `S ← γS + (βw·v − βe·k@S)·k^H`): k-direction
  eigenvalue = γ·(1 − βe·‖k‖²). VAULT pins γ≡1 (model.py:403-408) and
  qk_norm=False (unbounded keys, ‖k‖² up to ~d=64) → 1−βe‖k‖² flips past
  −1 → positive feedback → unbounded state → occasional hidden inf → NaN
  per-token NLL → sigmoid(median−inf)=NaN BCE target → Loss.cu:91 assert.
  v11 (reached 2B) has NONE of {delta, vault, phase, gate-aux} → never hits
  this. **Fix: per-VECTOR unit-norm keys** (`cnormalize_vec` in
  v13/complex_ops.py; applied in `_project` after RoPE+phase, gated on
  `write_mode=='delta'` + new config `delta_key_norm: bool = True`) so
  ‖k‖²=1 → eigenvalue in [1−βe,1) ⊂ (0,1): strict contraction for ANY γ in
  [0,1], vault or not. Standard delta-net fix; preserves the delta+vault
  novelty. NOTE: the existing `qk_norm` knob uses per-ELEMENT cnormalize
  (‖k‖²→d=64) — NOT sufficient. Applied at the single shared `_project` so
  fused-delta (907), loop-delta (569), K-loop (674) AND recurrent
  (1261) all inherit it and stay equivalent. Verified: `v13/selftest` ALL
  PASS incl. new `delta_keynorm` test (fused≡loop≡recurrent to 1e-20,
  keys unit-norm). QUALITY RISK (v6 lens, user-flagged): key-magnitude
  normalization removes readout dynamic range — v6 additive PAM hit
  repetition w/ QK-norm; must verify loss tracks r1 + no repetition.
  FALLBACK if quality degrades: normalize keys ONLY in the erase/mass term
  (M[t,s]=βe·D·(k̂k̂ᴴ)), keep raw keys for readout.
- [ROOT CAUSE REFINED + ROBUST FIX 2026-08-23, commit 03c3ede] The key-norm fix
  (04dcebd) stabilized the DIAGNOSTIC (2000 steps, state bounded ~17) but the
  REAL TRAINER died EARLIER at step ~151 (gtok 5.6M) on the same Loss.cu:91
  assert. Key-norm makes the vault delta k-direction eigenvalue EXACTLY
  (1 − beta_e) (‖k‖²=1, vault γ≡1): stable only while the LEARNED erase gain
  beta_e < 2. The trainer consumes RNG differently (loads wiki_val_ds BEFORE
  building the model → different weight init) and trained beta_e past 2 →
  eigenvalue flips past −1 → oscillatory state blowup → NaN. The diagnostic's
  init stayed beta_e<2 → survived. So key-norm was INIT-DEPENDENT, not robust.
  FIX (root cause): `delta_erase_beta_cap: float = 0.95` (V13Config default ON)
  clamps the learned erase gain in `_gate_betas` (single choke point → fused,
  K-loop, recurrent paths) so the eigenvalue stays in [0.05, 1) for ANY init.
  Erase strength still learned (sigmoid 0.047..0.95); only the >2 overshoot
  removed. Complements delta_key_norm. SAFETY NET (v7/train.py, shared w/ v7):
  _gate_surprisal_loss now drops non-finite NLL tokens from the median+mask and
  nan_to_num's the target to [0,1] (BCE runs over the full [L,B,T] before vmask,
  so even masked positions assert; clamp alone leaves NaN). Logs events, does NOT
  silence the aux.
  VERIFIED: selftest ALL PASS (new [delta_erasecap]; also fixed latent bug —
  test_warmstart_chatml returned None → false "SOME MODES FAILED");
  test_gate_nonfinite_guard 6/6 (a +inf/NaN NLL → finite aux + logged event,
  no assert); REAL-TRAINER repro 20M (launch_v13_500m_r1recipe.sh
  --token_budget 20M) — old key-norm run died at step 150 (loss 7.9623,
  gtok 5566464); with cap the trainer PASSED that exact point and ran clean to
  step 488 (20M, val 6.26), ZERO asserts, ZERO safety-net events (cap alone
  fixed it). 500M relaunched FRESH 02:36 (tmux v13_500m, ckpt dir wiped);
  watchdog armed at 100M.
- [50M VERDICT (FIXED RUN) 2026-08-23 ~07:1x] Matched gap vs r1: 2M +0.01,
  10M +0.15, 20M +0.24, **50M +0.83** (V13 5.58 vs r1 4.81) — PAST the 0.7 line.
  BUT the descent-rate view: V13 per-10M rate 40-50M = -0.117, r1 50-100M =
  -0.090 → V13 is NOT decelerating below r1 at 50M (the 20M->50M gap expansion
  is mostly r1's fast 20-50M phase, -0.353/10M, vs V13 -0.20). Projected V13
  @100M: +0.30..+0.92 (gap may plateau/shrink OR keep expanding — 50M gap
  alone can't tell). DECISION (kill&iterate policy): DO NOT reflex-kill at
  borderline +0.83 when 100M is ~34min away and is the pre-set decision point.
  Run to 100M (r1 4.36): KILL if gap >0.9, CONTINUE to 500M if gap <=0.9
  (then quality probes). Watchdog re-armed at 100M. Health: 0 asserts, 0
  gate-aux events at 57M — the erase-cap fix is holding.
- [open quality issue — the next battle IF 100M gap >0.9] pre-fix V13 already
  learned ~2.5x slower than r1 (old 50M verdict +0.78); key-norm+cap add a
  small cost (+0.03-0.08 in diag A/A). Suspects in order: (a) key-norm readout
  dynamic-range loss (v6 Bug-8 lens) → try `delta_key_norm=False` + erase-cap
  only (cap alone may be enough for stability), or normalize keys ONLY in the
  erase/mass term + raw readout keys; (b) protect_gate_bias -3.0 over-protects
  → -2.0; (c) gate-surprisal aux λ0.1 → 0.05. A/B each in diag first (fast).
- [stopped] `100m_realdat_500m_fresh` @ step 3175/117M — trained under the
  fused_ce bug (head untrained) AND warmup 2000. Do NOT resume.
- [stopped] `diag_additive` @ ~39.7M — same bug; ignore its curve.
- [50M VERDICT 2026-08-22 22:1x] V13 5.59 vs r1 4.81 = **+0.78** (at kill line 0.7).
  TREND EXPANDING: gap 0.44@5M → 0.19@20M → 0.78@50M. V13 learns ~2.5x slower
  than r1 in the full-lr phase (20M→50M: V13 6.06→5.59 Δ0.47, r1 5.87→4.81 Δ1.06).
  Not killing yet (borderline + delta stack is the novel core); running to 100M
  (ref 4.36) for a clearer signal + CPU micro-diagnostic to isolate the cause.
  KILL at 100M if gap >0.9. Suspects: protect_gate_bias -3.0 (over-protects,
  slows overwrite) and/or gate-surprisal aux λ0.1 (extra loss pressure).
  when --fused_ce was on; check each run's launcher for that flag before
  reusing any of their conclusions. e2b_50m / transformer_50m comparison data
  is SUSPECT until re-verified.

## THE REFERENCE (ground truth curves, B18/T2048, current-code v11)
**round-1 (Jul 1, from scratch, NEW CODE) = the real baseline to beat:**
`--preset v11_e3_k3_chat --stage pretrain --dataset pretrain_mix --seq_len 2048
--batch_size 18 --token_budget 2e9 --edu_score_min 3
--pretrain_sources dclm,fineweb,smoltalk2_mid --pretrain_weights 48,48,4
--fineweb_name sample-10BT --blend_warmup_tokens 1000000000 --seed 42
--lr 3e-4 --warmup_steps 500 --amp_dtype auto --num_workers 0
--gen_every 5000 --save_every_steps 5000 --no_grad_ckpt --compile`
Log: `logs/v11/round1_pretrain_20260701_115022_cbb4dd2_dirty/v11_v11_e3_k3_chat_pretrain_pretrain_mix.log`
Loss: **10.31@2M, 7.52@5M, 6.66@10M, 5.87@20M, 4.81@50M, 4.36@100M, 3.97@200M**
- round-2 (Jul 4, weights-resume, lr 1e-4 warmup 500 B32, 2B tok): ~3.3-3.5 flat.
- HF v11-best = "round-4b-gate" = round2-pretrain + smoltalk2 SFT (1ep, lr 5e-5).
  config.json says val_ppl 6.65 but **user: "we never had val ppl 6 for PAM"**
  → do NOT treat 6.65 as a target (likely chat-val artifact). Fair targets:
  round-1 train-loss curve + WikiText-103 val PPL at matched tokens.
- ⚠️ The **Jun-23 10B-scratch log** (used as ref earlier) is OLD code — the whole
  tree was rewritten after Jun 23 (v7/data.py +1403, v11/model.py +1023,
  v11/train.py +251). DO NOT compare against it anymore.

## FINDINGS (status: CONFIRMED / SUSPECT / RETRACTED)
- [RETRACTED-ish] "v13 is 3.1 NLL worse than v11" was vs the OLD Jun-23 ref.
  v13-fresh was launched with **warmup 2000** (copied from old ref) while the
  real new-code recipe is **warmup 500**. At 30-75M tok v13-fresh was still
  lr-ramping (8e-5→3e-4) while round-1 was at full 3e-4 → most of the gap is a
  WARMUP/RECIPE artifact, not code.
- [CONFIRMED] Delta-write stack is NOT the regression: v13 additive-diag (v11
  features off) was equally bad as delta-fresh at matched tokens — but that
  diag also had warmup 2000, so it only exonerates delta RELATIVE to additive,
  not absolute. Re-check after ab1.
- [OPEN] Whether current v13 base code (fused K-batch delta, gate-stash
  out-of-ckpt, NLL-byproduct gate target) is numerically equivalent to v11
  additive path → `v13/tmp/test_v11_v13_forward_ab.py` (same weights, compare
  logits per block). If equal → code clean; everything was recipe.
- [OPEN] Data-pipeline drift: v7/data.py +1403 lines since Jun 23. round-1 used
  edu_score_min 3, fineweb sample-10BT, blend_warmup_tokens 1e9 — the v13 runs
  did NOT (defaults). ab1 uses round-1's exact flags so it's a clean A/B.

## CODE STATE
- v13 = v11 fork. Diffed methods: `_fused_chunk_step` identical;
  `_forward_multistate_fused` identical modulo comments; `_project`,
  `_gamma_and_vprime`, `_routing_input`, `_phase_and_alpha`, `_dual_form_block`,
  `ce_from_lm` (v13 adds return_nll), `_hidden_to_lm`, `_init_weights` all
  equivalent. v13-only: `_gate_betas` (erase-gate), `_forward_multistate_delta_fused`,
  gate-stash in `V13Block.forward`, `V13LM._collect_gate_probs` (detached,
  outside ckpt), vault/phase config.
- **KEEP** v13/train.py lines ~448-449 dirty hunk (synthetic-source resume
  cursors `skip_docs_map.setdefault`) — do not touch.
- Speed-redesign (all verified earlier): K-batched fused delta (~21K tok/s, 9x),
  gate trunk detach, NLL-byproduct gate target in v13/fused_ce.py, fp32 CE
  under autocast, gate stash outside gradient checkpoint. Equivalence PASS
  (logits 1.5e-7, grads 3e-8).
- Configs: v13 preset `v13_e3_k3_selective` (delta, K=3, vault, phase, λ0.1,
  gate_content_aware, 100,621,792 params). v11 preset `v11_e3_k3_chat`
  (additive, K=3, 50261 vocab, 100,546,832 params). v13 with v11-features-off
  = 100,546,832 (shape-identical → weight copy possible for A/B).
- v13/train.py CLI: `--write_mode {additive,delta}`, `--delta_chunk`,
  `--gate_surprisal_lambda`, `--vault_state/--no_vault_state`,
  `--write_phase_address/--no_write_phase_address`, `--delta_erase_gate`/off?,
  `--warmup_steps`, `--lr`, `--batch_size`, `--seq_len`, `--token_budget`,
  `--pretrain_sources`, `--pretrain_weights`, `--gen_every`,
  `--save_every_steps`, `--log_interval`, `--no_grad_ckpt`?, `--compile`?,
  `--edu_score_min`?, `--fineweb_name`?, `--blend_warmup_tokens`?, `--seed`?,
  `--fused_ce`, `--amp_dtype auto`. (Check each exists before use.)

## LAUNCHERS (v13/tmp/)
- `launch_100m_fresh.sh` — the stopped 500M run (warmup 2000, 70/20/5/5/5).
- `launch_diag_additive.sh` — killed diag (warmup 2000, 50/50, v11 features off).
- `launch_ab1_round1recipe.sh` — **current A/B**: v13 code, v11 features off,
  round-1 EXACT recipe (warmup 500, 48/48/4, edu3, sample-10BT, blend 1e9,
  no_grad_ckpt, compile), 150M budget.
- Pattern: `set -euo pipefail; cd /home/gowrav/Development/qllm2;
  export HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=300;
  exec .venv/bin/python -m v13.train ...` in tmux, `| tee -a logs/v13/<name>/tmux_console.log`.

## MONITOR COMMANDS
- Steps: `grep -oE "\[1\] [0-9]+ loss=[0-9.]+ .*gtok=[0-9]+" <log> | sed -E 's/\[1\] ([0-9]+) loss=([0-9.]+) ppl=([0-9.]+) lr=([0-9.e-]+) \| ([0-9]+) tok\/s.*gtok=([0-9]+)/step=\1 loss=\2 lr=\4 tok_s=\5 gtok=\6/' | tail -8`
- Errors: `grep -icE "traceback|out of memory|nan" <log>`
- GPU: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
- tmux: `tmux ls | grep v13`

## KILL CRITERIA (user policy)
- If a run's loss at matched tokens is > ~0.7 NLL above round-1 curve
  (e.g. > 5.5 at 50M tok) → KILL, iterate, do not burn hours on a bad curve.
- Watch ~every 30 min. SIGTERM is safe (trainer saves latest.pt on signal).

## NEXT (ordered)
1. [IN PROGRESS] Watch relaunched 500M (tmux `v13_500m`, robust fix 03c3ede).
   VERDICTS at matched tokens (r1 curve 7.52@5M, 6.66@10M, 5.87@20M, 4.81@50M,
   4.36@100M). KILL if >0.7 NLL above. Watchdog armed at 100M (re-arm on each
   wake). Check for `[gate-aux]` safety-net events in the log (should be 0 —
   the cap prevents non-finite NLLs; any event = the cap is being stressed).
2. [ ] If it tracks r1 (≤0.7 NLL): let it run to 500M, save checkpoints.
   Quality probe at 50M/100M: `.venv/bin/python -m v13.eval_checkpoints
   --checkpoints checkpoints_v13/500m_v13_r1recipe/latest.pt --labels wiki`
   (WikiText-103 val PPL) + a short generate() for repetition.
3. [ ] If it sits >0.7 NLL above r1 (quality battle, NOT a crash): suspects in
   order — (a) key-norm readout dynamic-range loss → try `delta_key_norm=False`
   + erase-cap only (cap alone may suffice for stability; test in diag), or
   normalize keys ONLY in the erase/mass term + raw readout keys; (b)
   protect_gate_bias -3.0 over-protects → try -2.0; (c) gate-surprisal aux
   λ0.1 → try 0.05. A/B each in the diag driver first (fast), then relaunch.
4. [ ] Chain v11 PAM 500M head-to-head (`v13/tmp/launch_v11_500m_r1recipe.sh`)
   after V13 finishes (same GPU, sequential).
5. [ ] Update EXPERIMENTS_V13.md: 500M curve + NaN root cause (eigenvalue) +
   fix (key-norm + erase-cap + safety net) + the batch-0/151 regression.

## DECISIONS LOG
- 2026-08-22: Stopped 500M run @3175 (warmup-2000 recipe, on degraded curve).
- 2026-08-22: Realized ref was stale (Jun-23, old code) + warmup 2000 was wrong
  → relaunched diagnostic with round-1 EXACT recipe (ab1).
- 2026-08-22: User: drop V14 focus, focus V13 only; user away 2 days;
  kill-and-iterate policy; novel-not-transformer/Mamba; 100M+ rich data only;
  PAM never had val_ppl 6 (don't chase 6.65).
- 2026-08-23: key-norm (04dcebd) was INIT-DEPENDENT — diag passed 2000 but
  real trainer died step ~151 (different weight init via wiki_val_ds RNG draw
  trained erase beta_e past 2 → eigenvalue 1−beta_e flipped past −1). Robust
  fix = delta_erase_beta_cap 0.95 + gate-BCE safety net (03c3ede). Verified on
  the real trainer path (20M repro, zero asserts/events). 500M relaunched fresh.
