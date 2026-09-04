# v13_sempty handover — fused real-PAM speed work (2026-09-03)

Read this first if you are picking up the real-arm training. Everything below
is committed (`git log --oneline 4740d65..HEAD -- v13_sempty/`); the lab
notebook entry is `EXPERIMENTS_SEMPY.md` → "Speed: fused real arm".

## State of play

* **N1 CHRONO RUNG DONE (2026-09-04, commit `7b24e44`, RTX Pro 6000): val PPL
  23.14** — beats the 23.81 real baseline by 0.67 at every val point, gap to
  transformer (22.69) now 0.45. sROI KEEP; chrono is the new real-arm
  reference. Full record: `EXPERIMENTS_SEMPY.md` → "N1 Chrono-PAM". Log
  `logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`, ckpt
  `checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt` (on the
  RTX box). **Chrono decode now implemented** (state = `(notebook, clock)`,
  `test_chrono_parallel_vs_recurrent`); `--gen_every` may stay on.
* **Generator for prompt testing:**
  `.venv/bin/python -m v13_sempty.generate --checkpoint
  checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt --interactive`
  (loads once; type prompts; `/set temperature=0.6 max_tokens=120` retunes;
  blank line quits). One-shot: `--prompt "..."`. Device auto (cuda if present).
* **FAIR RUN DONE (2026-09-04, commit `7af42eb`): real-101M hit val PPL
  23.81** at the reference geometry (T=2048, B=18, 10 ep, 1.18B tok). Beats
  v11 E3-K3 complex (25.77) by ~2 PPL; within 1.12 of transformer (22.69).
  Log `logs/v13_sempty_wikitext_real_fair_7af42eb_20260903_1628.log`,
  ckpt `checkpoints_v13_sempty/wikitext_real_fair_7af42eb/best_model.pt`.
  Full curve + comparison + "is this Mamba?" positioning are in
  `EXPERIMENTS_SEMPY.md` (last two sections). Open gap = recall, not PPL.
* Logging now V11-style: `_print_run_header` emits full config/args/geometry
  at the top of every log; step lines carry `epN/M`; epoch-boundary banners
  print train-loss/ppl/tok/best-val. Log naming convention is in
  `qllm2/AGENTS.md` ("Log naming convention").
* **Next: bigger runs on the RTX Pro 6000 (96 GB)** — more data + chat, and
  scale-up if 100M looks saturated. See "Bigger-run plan" below / the new plan.

## Novel math — N1 Chrono-PAM (content-modulated rotary retention) (2026-09-04)

**Idea, in one line:** make the memory's rotary phase *learned and
input-dependent* instead of fixed RoPE — a "content clock" per head.

**Why it's principled (the derivation).** A complex *rotating* retention
`gamma_t = r_t * e^{i*theta_t}` on the outer-product notebook
`S_t = gamma_t S_{t-1} + v_t (x) conj(k_t)` has closed form
`S_s = sum_{t<=s} (a_s/a_t) e^{i(Phi_s - Phi_t)} v_t conj(k_t)` with
`a` = magnitude product, `Phi_t = cumsum(theta)`. The read
`S_s q_s` shows the `e^{i(Phi_s-Phi_t)}` factor is *absorbed* by rotating
`q_s -> e^{i*Phi_s} q_s`, `k_t -> e^{i*Phi_t} k_t`. That cumulative rotation
is exactly what RoPE does with a *fixed* frequency. So **learned rotating
retention == input-dependent RoPE**, and it folds entirely into q/k — the
fused magnitude-retention kernel is UNTOUCHED (speed preserved). CoPE-style,
but on an associative-memory PAM (novel).

**Real-arm implementation** (`RealPAMLayer._rotate_learned`, behind
`cfg.chrono`): per-head warp `g_t = exp(clamp(W x, +/-3))` scales the per-step
angle; since `inv_freq` is constant in t, `cumsum(inv*g) = inv * cumsum(g)`,
so we warp a per-head clock `tau = cumsum(g)` ([B,H,T]) then `phi = tau (x)
inv_freq`. `W` is zero-init (`warp_proj._zero_init`), so at start `g=1`,
`phi = pos*inv_freq` == **exactly** fixed RoPE. cos/sin in fp32, cast to bf16
for the rotation (keeps retained activations small).

**Status: RUNG DONE — 23.14 vs 23.81 (see State of play). Notes below are
the pre-run record.**
- Parity: `test_chrono_rotary_parity` (selftest, CPU) — chrono@init == baseline
  RoPE bit-for-bit (`max|dlogit| = 0.0`), warp grads flow. All 15 selftests pass.
- Layout: `check_torch_layout` clean (`_rotate_learned` is a declared boundary).
- **Speed gate PASSED**: baseline_real_pm B8 T2048 bf16 on 4090 — baseline avg
  **65.9k** tok/s vs chrono **66.6k** tok/s (equal within noise). Mem +1.6 GB
  (per-head cos/sin x16 layers; recomputed under grad-ckpt, irrelevant at 96 GB).
- Decode: implemented after the run (carried state is `(notebook, clock)`,
  the clock replaces `step_offset`); parity test 6.6e-7 vs chunked.

**LAUNCHED 2026-09-04 06:20Z on the RTX Pro 6000** (tmux `sempty_chrono`,
commit `7b24e44`): `logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`
(on the remote box, `~/Development/qllm-private`). Header confirms
`'chrono': True`, B18 T2048 32130 steps = 10 ep — identical geometry to the
23.81 baseline. ~75-83k tok/s (vs 45.5k on the 4090), ETA ~4.5 h. The
`[gen @ 8000]` lines will say `failed: ... chrono decode not implemented` —
expected and harmless (try/except; val/ckpt use the chunked path). Remote
setup notes: `sempyt` is NOT a pip dep — it is cloned at
`~/Development/sempyt` and wired via `site-packages/sempyt_src.pth` (same as
local). WikiText/gpt2 need no HF token (public). Watch:
`ssh ubuntu@34.131.203.207 'grep -E "^step|val @" ~/Development/qllm-private/logs/v13_sempty_wikitext_chrono_fair_*.log | tail'`.

**How to run the rung (the "main" run on the RTX Pro 6000):**
```bash
# on the remote box, code already pulled to c-hash below:
cd ~/Development/qllm-private
CHRONO=1 TAG=wikitext_chrono_fair \
  tmux new-session -d -s sempty_chrono "bash v13_sempty/tmp_wikitext_fair.sh"
# identical geometry to the 23.81 baseline (T=2048 B18 10ep 1.18B tok);
# compare best val_ppl vs 23.81 and the wiki recall probe.
```
`tmp_wikitext_fair.sh` now takes `CHRONO=1` (appends `--chrono`). First run on
a fresh box tokenizes WikiText (sl2048 cache) once. Log name carries the commit
+ timestamp (naming rule in AGENTS.md); the in-file header prints `[ladder]
chrono=True`.

**Next rungs (N1 won; stack on `CHRONO=1`, one variable each, same B18 T2048
10 ep, compare vs 23.14).** All keep the scan / are elementwise, see
EXPERIMENTS "Positioning":
1. **N4 phase-resonant output gate** — memory is still underused
   (`pam_scale` 0.11–0.31 at the end of the chrono run). Not coded yet.
2. **A1 `--short_conv`** (coded) — cheapest ladder rung.
3. **A3 `--delta`** (coded) — the recall lever; needs a recall-mix run to be
   judged (WikiText-only PPL will not show it; see EXPERIMENTS ledger row 4).
4. **A4 `--cond_mem`**, then **A2 `--n_states`/`--vault`** (weaker sROI).
5. N3 interference-erase / N2 frequency-multiplexed keys if A3 is not enough.
On the RTX Pro 6000 run these with `GRAD_CKPT=0` (96 GB; the ckpt default is
a 4090 fit). **Measured 2026-09-04** (chrono, B18 T2048, 60-step smoke):
grad-ckpt OFF = **~102k tok/s, peak 29.1 GB** vs ON = 83k tok/s, 8.9 GB —
+23 %, a 10-epoch rung drops from 4.0 h to ~3.25 h. Keep B=18 T=2048 for
comparability. Launch template for the next rung — **put the env vars INSIDE
the tmux command string**:
```bash
tmux new-session -d -s sempty_n4 \
  "GRAD_CKPT=0 CHRONO=1 OUT_GATE=1 TAG=wikitext_chrono_n4gate_fair bash v13_sempty/tmp_wikitext_fair.sh"
```
**PITFALL (bit us 2026-09-04):** `VAR=x tmux new-session ...` only works when
it *starts* the tmux server. Once a server exists, new sessions inherit the
*server's* environment, so the env prefix is silently ignored and you launch
whatever the first session's env said (we got a duplicate A1 instead of N4).
**Do not run two rungs concurrently** on the 6000: measured 39k+39k = 78k
tok/s combined vs 85k single (−8%); queue instead
(`while pgrep -f <ckpt-dir-tag>; do sleep 60; done; ...` in the tmux command).

**Running / queued (2026-09-04 11:00Z, RTX Pro 6000):**
* `sempty_a1` — A1 `SHORT_CONV=1 CHRONO=1 GRAD_CKPT=0`, commit `817aa35`,
  log `logs/v13_sempty_wikitext_chrono_a1conv_fair_817aa35_20260904_1057.log`,
  85k tok/s, 33 GB, ETA ~4 h (finish ~15:00Z). Compare vs **23.14**. Note the
  speed tax: plain-torch depthwise conv costs ~17 % tok/s (102k → 85k) — A1
  needs a clear PPL win to pass sROI.
* `sempty_n4` — N4 `OUT_GATE=1 CHRONO=1 GRAD_CKPT=0` (commit `3c7b9b9`)
  queued; auto-starts when the A1 process exits, log will be
  `logs/v13_sempty_wikitext_chrono_n4gate_fair_<hash>_<stamp>.log`. Speed gate
  measured −2.8 %. Compare vs 23.14. Arm the watchdog on it when it starts.
Data/scale-up (DCLM/FineWeb mix via `--dataset pretrain_mix`, `c7a343b`) comes
*after* the ladder settles the architecture at 100M.

* Real-101M (`baseline_real_pm`) trains at **~64k tok/s** (was 5.8k), peak
  **7.6 GiB** at B32 T256 with grad-checkpointing OFF (was 21 GB at B8 with
  it on). One WikiText-103 epoch (118M tok) ≈ **30 min**.
* The one-epoch T256 run **finished** (commit `7343201`): best **val PPL
  54.89** (NLL 4.005), avg 64.1k tok/s, exit 0.
  `logs/v13_sempty_wikitext_real_7343201_20260903_1449.log`,
  checkpoints `checkpoints_v13_sempty/wikitext_real_fused_7343201/`
  (`best_model.pt`, `latest.pt`).
* **Now executing the quality program** (plan
  `.cursor/plans/v13_sempty_quality_program_7759901a.plan.md`): fair 10-epoch
  T=2048 WikiText baseline, complex arm at kernel speed, recall infra, and an
  architecture ladder (short conv, multi-state+vault, delta erase/write,
  Engram-style conditional memory, layout). See that plan for the full spec;
  results land in `EXPERIMENTS_SEMPY.md`.

## How to watch

```bash
tmux attach -t sempty_wiki                 # live; Ctrl-b d to detach
LOG=$(ls -t logs/v13_sempty_wikitext_real_*.log | head -1)
grep -E "\[val @" "$LOG" | tail            # val NLL/PPL every 500 steps (~4M tok)
grep -E "^step " "$LOG" | tail -3          # loss / tok/s / ETA
# watchdog (wakes on: process gone, error/nan in log, step reached, timeout):
bash v13_sempty/tmp_wiki_watchdog.sh "$LOG" 14400 2940
```
Re-arm the watchdog on every wake (rule in AGENTS.md). A healthy run shows
~67k tok/s, `GPU 1.3/7.6GB`, loss falling (6.2 at step 300, expect ~4.3 at
the end: the 09-01 run reached train NLL 4.38 / val PPL 68.75 at one epoch
with constant lr 5e-5 and B8).

## How to run (again / longer)

`v13_sempty/tmp_wikitext_real.sh` is the whole recipe — edit flags there.
```bash
tmux new-session -d -s sempty_wiki "bash v13_sempty/tmp_wikitext_real.sh"
```
Current flags: `--batch_size 32 --seq_len 256 --steps 14400 --lr 1e-4
--warmup_steps 100 --amp_dtype bf16 --fused_ce --fused_pam --ce_gemm_dtype
auto --val_every 500 --save_every_steps 2500`, no `--gradient_checkpointing`.
* **Recipe change vs 09-01 you should know about:** batch 4x (B8→B32) and
  lr 5e-5 → 1e-4 (sqrt scaling); `--steps 14400` = one epoch so warmup-cosine
  completes at the epoch end (09-01 had a 200k horizon = constant lr). If you
  want a literal repeat of the old regime use `--lr 5e-5 --steps 200000`.
* **Longer runs:** `--epochs N` (trainer flag) with `--steps` = N × 14409 so
  the cosine spans the whole run. Memory headroom is large: B64 T256 fits in
  11.9 GiB but tok/s *drops* (63k), B96 57k — stay at 8192 tok/step.
  `--seq_len 512 --batch_size 16` is the same tok/step at 68.5k tok/s.
* Kill switches: `--no_fused_pam` (plain-torch scan, same math, 2.4x slower
  kernel), `V13S_KERNEL=0` env (disables both Triton PAM scan and Triton CE),
  `--ce_gemm_dtype fp32` (exact head, +10 ms/step).
* Log names carry commit hash + timestamp; the header line prints
  `commit=<hash>[-dirty] fused_pam=… ce_gemm=… grad_ckpt=…`.

## What was done (for the record)

1. `triton_kernels.py` — real PAM read in chunked linear-attention form
   (`y_s = a_s (S_in.q_s) + Σ_{t≤s} (a_s/a_t)(q_s.k_t) v_t`), Triton forward
   (state scan + read) and backward (state-grad scan, dq/dk/dg, dv), tile 64,
   8 warps; `pam_scan_torch` fallback; `fused_real_pam_read` entry.
   `RealPAMLayer._chunked` hands off to it; decode `_stepwise` and the complex
   arm untouched. Carried state layout unchanged (rows = value, cols = key).
2. `pam_kernel_test.py` — parity harness (oracle anchored to `_stepwise`;
   the inherited oracle had `outer(k,v)` transposed — fixed). PASS fp32 5e-5 /
   bf16 3e-2. `selftest` 13/13 incl. `test_real_fused_kernel_parity`.
3. `fused_ce.py` — `gemm_dtype` (bf16 GEMMs, fp32 loss) and the Triton
   Liger-style `_FusedLinearCETriton` (grads in forward, one row pass).
   Validation pinned to the fp32 head.
4. `train.py` — `--ce_gemm_dtype`, `--fused_pam/--no_fused_pam`, commit hash
   in the header. `check_torch_layout.py` — new boundaries declared.

## Open items / ideas not taken

* Remaining 61 ms step: Linears + CE GEMMs at tensor-core peak; ~20 ms of
  small elementwise (norm/gate/residual/RoPE, 900 `mul` launches/step) —
  only `torch.compile` would fuse these and sempyt's Dim identities trip its
  recompile limit (needs a sempyt change). RoPE fusion measured at 3 ms max.
* Why tok/s drops beyond 8192 tok/step is unexplained (not memory).
* The one-epoch T256 result (54.89) is recorded in `EXPERIMENTS_SEMPY.md`
  → "Speed: fused real arm". The apples-to-apples number is the Phase-1
  fair run (10 epochs, T=2048); this scratchpad and superseded `tmp_*`
  scripts get cleaned up at the end of the quality program (Phase 5).
