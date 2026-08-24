# V13 SCRATCHPAD — read this FIRST after any context summary

Long lab notes live in [v13/EXPERIMENTS_V13.md](EXPERIMENTS_V13.md). This file
steers the next session. Do not bury the mission under a battle log.

## MISSION
Build a **novel** language model: selective PAM (complex embeddings, K=3
phase-addressed SSM states, delta-write + vault, GSP protect gate). **Not** a
Transformer or Mamba reskin.

- **Inference (non-negotiable):** O(1) per token. Recurrent state, no KV cache.
  `v13/selftest` gates parallel-train form ≡ recurrent-infer form.
- **Quality (primary, now):** train on 500M rich real tokens (DCLM 48 + FineWeb
  48 + smoltalk2_mid 4) and land WikiText-103 val PPL at/below the **r1
  pretrain endpoint ~84.6** (r1 logged Wiki PPL 84.57 @ 2B, log line 1183 —
  the fair pretrain-to-pretrain number). **Stretch: < 25.77** (the
  *WikiText-trained* v11_e3_k3 anchor, v11/EXPERIMENTS_V11.md:584) — 25.77
  requires the selective stack to actually contribute; it is NOT the r1
  pretrain endpoint (corrected 2026-08-24, see r_and_d.md 330M section).
  Ideally toward/below the transformer anchor **22.69**. Also: better
  reasoning/maths than that V11.
- **Train-loss vs v11 round-1** is a kill-canary, not the prize.
- **Training speed (secondary):** honest 4090 number with *real* grads is
  ~4–6.5K tok/s. Slow is acceptable until quality is a real number. Do not
  chase a fake 21K by skipping backward. Speed ideas are parked below.

Preset: `v13_e3_k3_selective` (~100.6M). v11 additive twin: `v11_e3_k3_chat`.

## NON-NEGOTIABLES
- Keep non-reentrant gradient checkpointing ON. `--no_grad_ckpt` OOMs at
  B18/T2048 bf16 on the 4090.
- **NEVER detach** the checkpointed block input in `V13LM._ckpt_block`.
  Commit d0abeed did that as a "determinism_check workaround" and silently
  froze every block except the last. Removed in `cbd35d4`.
- Keep `delta_key_norm=True` and `delta_erase_beta_cap=0.95` (NaN / eigenvalue
  fixes). Do not replace `cnormalize_vec` with a `g/mag` autograd.Function
  (19% key-grad error).
- Long training ONLY in tmux + watchdog. Re-arm on every wake
  (`timeout 3300` is mandatory; default 300s kills the chain).
- Do not touch the dirty hunk in `v13/train.py` (~`skip_docs_map.setdefault`).
- After every verified change in `v11/` `v13/` `v7/` `scripts/`: git commit
  (what + why + evidence). `v13/tmp/` throwaways do not need commits.
- After step 1 of any train run the log MUST contain
  `[block-grad step1] L0=... L15=... all-nonzero`. `DEAD=` → KILL immediately.

## STATUS (2026-08-24 ~06:45)
Grads under checkpointing are **fixed** (`baaf5b3`). The 02:28→09:20 "500M
complete" run was the **buggy-code** run (20,684 tok/s avg = retracted detach
figure; floor ~4.65, Wiki 368.69) — VOID, user-confirmed, dir wiped.

**500M IS RUNNING** (relaunched 13:26, tmux `v13_500m`, fixed code, EAGER
B8/C128): steady **~4,850 tok/s @ 8.7GB**; verdicts (window means) all
PASSED: **20M 5.46** (kill >6.6); **50M 4.65 (r1 4.81)**; **100M 4.36
(r1 4.36)**; **200M 4.17 (r1 4.04, +0.13)**; **300M 4.08 (r1 3.96, +0.12)**;
at ~347M (2026-08-24 09:20). Probes (Wiki PPL): **325.76@82M → 211.66@164M →
166.75@247M → 149.62@330M** (fair reference is r1 pretrain 84.57@2B; 25.77
is the WikiText-trained stretch anchor — see r_and_d.md 330M section);
selective stack flat across all probes (protect ~0.06, phase ~0, write-phase
dormant, βw/βe 0.50) — see `v13/r_and_d.md`.
Watchdog armed at 400M (r1 ref 3.83).
ETA at ~4.8K: ~2.6h remaining @347M to 500M.

**`--compile_blocks` CRASHES at first step** (2026-08-23): Inductor
meta-kernel bug — `assert_size_stride` on `torch.ops.aten.complex.default`
inside the compiled block. Not our code; do not relaunch with it until
fixed (SPEED track). Eager B8/C128 (5,027 bench / 4,930 live) is the run config.

Honest speed (4090, T=2048, all 16 blocks learning): B16/C128 eager **4,101**
tok/s (13.9GB); B8/C128 eager 5,027; B8/C256 5,085; compile-block 6,459
(BROKEN now — see above). 500M wall-clock ~28h @ 4.9K — acceptable.

## NEXT
1. **500M is running** (this session). Watchdog chain: on every wake re-arm
   `bash v13/tmp/watchdog.sh logs/v13/500m_v13_r1recipe/v11_v13_e3_k3_selective_lm_pretrain_mix.log 400000000 2940`
   (async + timeout 3300). Launch cmd for any relaunch:
   ```
   rm -rf checkpoints_v13/500m_v13_r1recipe
   mkdir -p logs/v13/500m_v13_r1recipe
   tmux new-session -d -s v13_500m \
     'bash v13/tmp/launch_v13_500m_r1recipe.sh --batch_size 8 --delta_chunk 128 \
      2>&1 | tee -a logs/v13/500m_v13_r1recipe/tmux_console.log'
   ```
   (NO `--compile_blocks` — inductor complex-buffer crash.)
2. Run to 500M. Probe battery on each saved ckpt (done: 164M Wiki 211.66,
   247M 166.75, 330M 149.62; remaining: step 25000 ≈413M, step 31250 ≈500M):
   Wiki PPL
   `.venv/bin/python -m v13.eval_checkpoints --checkpoints checkpoints_v13/500m_v13_r1recipe/latest.pt --labels wiki --batch_size 2`
   plus `.venv/bin/python v13/tmp/dissect_ckpt.py checkpoints_v13/500m_v13_r1recipe/latest.pt`
   (protect gate, phase_proj, write_phase_proj, betas — see `v13/r_and_d.md`)
   + a short `generate()` for repetition. Final verdict: Wiki PPL vs r1
   pretrain endpoint **84.57** (match) / stretch 25.77 (selective stack).
3. If gap >0.7 (quality, not a crash): A/B in order — (a) key-norm only on
   the erase/mass term, raw readout keys; (b) `protect_gate_bias` -3.0 → -2.0;
   (c) gate-surprisal λ 0.1 → 0.05. Diag first, then relaunch.
4. After V13 finishes: v11 PAM 500M head-to-head, same GPU, sequential.
5. Speed work: only after 500M is healthy, or if the run is unusable at 4K.
   See SPEED (LATER). Do not start it instead of launching.

Kill if train loss is > ~0.7 NLL above r1 (e.g. >5.5 at 50M). SIGTERM is
safe (trainer writes `latest.pt`).

## SPEED (LATER)
Do not start this track unless 500M is healthy or the GPU is idle. Profile:
the step is **launch/elementwise bound** (`copy_`/`mul`/`fill_` dominate;
all matmuls+solves ~14% of CUDA). A faster triangular solve will not 5× us.
`[K,B,H,C,C]` mass/decay is why larger batch is *worse* per token.

Each idea gated on `v13/selftest` + ckpt-vs-no-ckpt grads:

1. `--compile_blocks`: **BROKEN 2026-08-23** — Inductor `assert_size_stride`
   on `aten.complex.default` at first step (meta/real layout mismatch).
   `--delta_decay_factored` (K-independent system; `[delta_factored]` PASS)
   is still unbenched.
2. Selective rematerialization: checkpoint CGU/norm, **save** the chunk-solve
   output. Today we recompute the whole block; backward is ~8× forward.
   Most likely honest 1.5–2× with no math change.
3. CUDA graphs / static chunk loop at fixed `(B,T,C)` — tens of thousands of
   elementwise launches are the profiler story.
4. Fused `mass+solve+project` so inductor (or one kernel) sees the whole
   chunk. Do **not** reopen Flash-PAM / Triton custom autograd first; that
   historically lost to `torch.compile` on this codebase.
5. Out-of-box, architecture-preserving: train with the **same O(1) recurrent
   step as inference**, parallelized by an associative scan over
   `S ← γS + (βw v − βe k@S)kᴴ`. No C×C, no UT solve, train ≡ infer by
   construction. Not a Mamba reskin (phase addressing + vault + GSP stay).
   New kernel + new selftest. Only after quality is a real Wiki PPL, or if
   4K tok/s makes 500M unusable.
6. Never: re-detach the block input; `g/mag` backward; claiming 21K;
   training without `[block-grad step1]`.

## ARCHIVE (settled — do not re-open)
- v13 additive path == v11 bit-identical (`test_v11_v13_forward_ab.py`,
  0.000e+00). Not a fork regression.
- `fused_ce` dropped `grad_weight += ...` (65546dc). Head got zero CE grad.
  Restored `79c7cc2`. Rel-L2 5.8e-6.
- 500M NaN: vault δ eigenvalue `γ(1 − βe‖k‖²)` flipped past −1. Fix:
  per-vector key-norm (`04dcebd`) + `delta_erase_beta_cap=0.95` (`03c3ede`)
  + gate-BCE nonfinite guard. Cap alone passed the real-trainer death at
  step 150.
- Ckpt crash: TorchScript `cnormalize_vec` operand-swap (`baaf5b3`). Not
  flaky. `g/mag` autograd.Function RETRACTED (19% key-grad error).
- 21K tok/s / "9×" RETRACTED — measured with `_ckpt_block` detach.
- Pre-`baaf5b3` 50M/100M train-loss verdicts VOID (frozen layers and/or
  dead head). Do not use them to kill or keep a run.
- Warmup-2000 runs and Jun-23 10B log are wrong references. Use r1 below.
- Do not chase HF config val_ppl 6.65 — user: "we never had val ppl 6 for PAM".

## REFERENCE
**v11 round-1** (Jul 1, new code, 75GB, `--no_grad_ckpt --compile`):
`--preset v11_e3_k3_chat --warmup 500 --lr 3e-4 --batch_size 18 --seq_len 2048
Loss: **10.31@2M, 7.52@5M, 6.66@10M, 5.87@20M, 4.81@50M, 4.36@100M, 3.97@200M,
3.96@300M, 3.83@400M, 3.82@500M** (last three verified 2026-08-23 from the
same log; the run goes to ~2B). Verdict gaps should use ±2M window means.
Log: `logs/v11/round1_pretrain_20260701_115022_cbb4dd2_dirty/v11_v11_e3_k3_chat_pretrain_pretrain_mix.log`

WikiText-103 val PPL: **r1 pretrain endpoint 84.57 @ 2B** (log line 1183 —
the fair pretrain-to-pretrain number to match); **stretch < 25.77** (the
WikiText-trained v11_e3_k3 anchor, requires selective stack to contribute);
**ideal ~22.69** (transformer). 500M probe decides match-vs-stretch.

## WAKE PROTOCOL (every wake, all of these)
1. Read this file fully.
2. `tmux ls`; `pgrep -af "v1[13].train"`; GPU; active log last steps + errors:
   `grep -oE "\[1\] [0-9]+ loss=[0-9.]+ .*gtok=[0-9]+" <log> | tail -8`
   `grep -icE "traceback|out of memory|nan" <log>`
3. Re-arm watchdog (timeout 3300 is mandatory):
   `bash v13/tmp/watchdog.sh <active_log> <verdict_gtok> 2940`
   as `async: true` + **timeout: 3300**. Exits early on process death / OOM /
   verdict gtok — that wake is the chain. Keep it alive until V13 is done.
4. Commit verified `v11/` `v13/` `v7/` `scripts/` changes immediately.
5. Act from STATUS + NEXT; yield only with the watchdog armed.

## MONITOR
- GPU: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
- tmux: `tmux ls | grep v13`
- Launch pattern: `set -euo pipefail; cd /home/gowrav/Development/qllm2;
  export HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=300;
  exec .venv/bin/python -m v13.train ...` in tmux,
  `| tee -a logs/v13/<name>/tmux_console.log`.
- New CLI (2026-08-23): `--compile_blocks`, `--delta_decay_factored`,
  `--delta_key_norm` / `--no_delta_key_norm`, `--delta_erase_beta_cap`.
