# v13_sempty handover — fused real-PAM speed work (2026-09-03)

Read this first if you are picking up the real-arm training. Everything below
is committed (`git log --oneline 4740d65..HEAD -- v13_sempty/`); the lab
notebook entry is `EXPERIMENTS_SEMPY.md` → "Speed: fused real arm".

## State of play

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
