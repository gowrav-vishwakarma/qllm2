# V13 SCRATCHPAD — read this FIRST after any context summary

**Mission (user, 2026-08-22):** Make V13 (100.5M-param selective-PAM: complex
embeddings, K=3 phase-addressed SSM states, delta-write + vault, GSP protect
gate) mature, better, faster. Beat v11-best quality at matched tokens on
**500M real tokens**, speed-first, O(1) inference (no KV cache). User is away
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
- [running] **V13 500M MISSION**: full delta stack (K=3, delta, vault, phase,
  λ0.1) + round-1 EXACT recipe (warmup 500, 48/48/4, edu3, sample-10BT,
  blend 1e9) + FIXED fused_ce. tmux `v13_500m`.
  Log: `logs/v13/500m_v13_r1recipe/v11_v13_e3_k3_selective_lm_pretrain_mix.log`
  Launched 2026-08-22 ~20:01 UTC. Verdicts: 5M (r1=7.52), 20M (5.87, kill if
  >6.6), 50M (4.81), 100M (4.36). Δ-stack may lag r1 slightly by design (it is
  the novel model) but MUST be on-curve (≤0.7 above).
- [stopped] `100m_realdat_500m_fresh` @ step 3175/117M — trained under the
  fused_ce bug (head untrained) AND warmup 2000. Do NOT resume.
- [stopped] `diag_additive` @ ~39.7M — same bug; ignore its curve.
- [dead] ALL v13 runs before 2026-08-22 20:00 UTC trained with broken fused_ce
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
1. [ ] Watch `v13_ab1` to 50M (verdict: tracks round-1 4.81 ⇒ base+recipe clean)
   then 100M (4.36).
2. [ ] Numerical A/B `v13/tmp/test_v11_v13_forward_ab.py` (weights copied,
   per-block logits compare) — confirms/disproves code equivalence fast.
3. [ ] If ab1 tracks r1: relaunch 500M FRESH (delta stack ON + round-1 recipe:
   **warmup 500**, 48/48/4 or 70/20/5/5/5 — decide: 48/48/4 = matched to r1;
   5 sources = v13 recall/reason program; start 48/48/4 for clean comparison,
   add recall/reason later). Verify tracks r1 at 20M/50M before committing.
4. [ ] If ab1 stalls: numerical A/B bisect → fix code → re-verify.
5. [ ] Update EXPERIMENTS_V13.md with the corrected reference + verdict.
6. [ ] (later) quality probes: eff-rank, gate probe, behavioral vs
   checkpoints_v13/transformer_50m, WikiText-103 val PPL at matched tokens.

## DECISIONS LOG
- 2026-08-22: Stopped 500M run @3175 (warmup-2000 recipe, on degraded curve).
- 2026-08-22: Realized ref was stale (Jun-23, old code) + warmup 2000 was wrong
  → relaunched diagnostic with round-1 EXACT recipe (ab1).
- 2026-08-22: User: drop V14 focus, focus V13 only; user away 2 days;
  kill-and-iterate policy; novel-not-transformer/Mamba; 100M+ rich data only;
  PAM never had val_ppl 6 (don't chase 6.65).
