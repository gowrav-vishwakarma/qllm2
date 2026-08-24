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

## STATUS (2026-08-24 18:54) — 500M RUN COMPLETE
Full verdict table in [EXPERIMENTS_V13.md](EXPERIMENTS_V13.md) "500M r1-recipe
run — COMPLETE". Headlines:

- **Endpoint:** 500,000,768 tok / **29.48 h** / avg 4,713 tok/s.
  Val PPL **50.08**, **Wiki PPL 133.88**. All kill gates passed
  (~500M window NLL **3.87** vs r1 **3.79**, +0.08).
- **Wiki trajectory flattened:** 325.76@82M → 211.66@164M → 166.75@247M →
  149.62@330M → 136.20@413M → 134.03@491.5M → **133.88@500M**. The r1 pretrain
  endpoint 84.57@2B was NOT reached and is not reachable on a 500M budget.
- **COMPUTE-MATCHED VERDICT (the one that matters): v11 additive wins.**
  r1 = 2B tok in 21.6 h → Wiki **84.57**. v13 = 500M in 29.48 h → Wiki
  **133.88**. Both fully-annealed. Same GPU-hours buys v11 ~4× the tokens and
  ~37% better Wiki PPL. Delta-write alone does not pay for its ~3.2× cost.
- **Selective stack never woke — 6 probes, 82M→500M.** protect 0.056–0.065
  (init 0.047), phase bnorm ~0.004–0.010 (phases ≈0), write_phase dormant,
  βw/βe ≈ 0.50. CE parity is delta+CGU, not selectivity.
- **BUT recall is the best this repo has produced.** 8-way behavioral
  (chance 0.125), no recall data in the mix: **recall@2048 = 0.250**,
  **assoc=1 @ ctx128 = 0.744**, overall 0.254. Beats v11 Stage-6c vault
  winner **0.189** (which had 3% synthetic recall) and the Stage-3 tuned
  ceiling 0.23. Matched Transformer is 0.956.
- **REAL BOTTLENECK = WRITE INTERFERENCE, not selectivity, not context.**
  assoc 1→4→8 = **0.408 → 0.219 → 0.133**; **multi8 @ ctx128 = 0.133 vs
  0.125 chance**. 8 facts in a 128-token window are already unrecoverable.
  Do NOT re-sweep λ/τ/γ_floor/recall-weight or vault-vs-phase — v11
  Stage-2/3/6 measured that exhausted (~1.4B tok of evidence).

**`--compile_blocks` CRASHES at first step** (2026-08-23): Inductor
meta-kernel bug — `assert_size_stride` on `torch.ops.aten.complex.default`
inside the compiled block. Not our code; do not relaunch with it until
fixed (SPEED track). Eager B8/C128 (5,027 bench / 4,930 live) is the run config.

Honest speed (4090, T=2048, all 16 blocks learning): B16/C128 eager **4,101**
tok/s (13.9GB); B8/C128 eager 5,027; B8/C256 5,085; compile-block 6,459
(BROKEN now — see above). 500M wall-clock ~28h @ 4.9K — acceptable.

## NEXT — attack write interference (chosen 2026-08-24 with user)

1. **IN PROGRESS: in-chunk raw-key readout** (claim 2, `r_and_d.md:33-63`).
   `delta_key_norm=True` normalizes ONE keys tensor feeding four consumers:
   key-gram mass, in-chunk `query_key` readout, erase read `k@S`, and state
   construction `S += update ⊗ k`. Retrieval is therefore a pure cosine `q·k̂`
   — magnitude contrast is gone. **Scope agreed: in-chunk ONLY** — raw keys
   for the `query_key` score (`model.py:989-992`), unit keys retained for mass
   / erase / state construction so stability (eigenvalue `γ−βe‖k‖²`) is
   untouched. The cross-chunk carry `q@S` stays cosine-built; that is a known
   limitation of this scope, affecting ctx2048 but NOT multi8@128.
   **Why well-targeted:** multi8@ctx128 with `delta_chunk=128` lives entirely
   in the in-chunk path.
   **Falsifier:** `multi8 @ ctx128` must move off **0.1333** (chance 0.125).
   Secondary: assoc=4 off 0.219, and CE must not regress.
   **Gates before any training:** `v13/selftest.py` fused ≡ K-loop ≡
   recurrent WITH the flag on; bit-identical to today with the flag off;
   grad-ckpt vs no-ckpt grad equivalence. Non-negotiable.
2. Re-measure with the same suite/seeds for comparability:
   ```
   .venv/bin/python scripts/run_memory_behavioral.py --model-type v13 \
     --checkpoint <ckpt> --preset v13_e3_k3_selective \
     --context-lengths 128,512,1024,2048 --positions 0,0.5,1 \
     --association-counts 1,4,8 --trials 60 --candidate-count 8 \
     --output logs/memory_probes/<name>_behavior.json
   ```
   Baseline to beat: `logs/memory_probes/v13_500m_r1recipe_FINAL500M_d169584_behavior.json`.
3. If interference does NOT move: the outer-product substrate itself is the
   limit (capacity/superposition), not any v13 lever. That is the point to
   either redesign the memory or return to v11 additive.
4. Deferred, only if wanted for the record: v11 additive 500M head-to-head on
   the 6000 (same budget, same recall suite). The compute-matched r1-vs-v13
   comparison above already answers the practical question.
5. **Do NOT** re-sweep λ/τ/γ_floor/recall-weight, vault-vs-phase, or add more
   synthetic recall data — v11 Stage-2/3/6 spent ~1.4B tok proving those are
   exhausted (100% recall data still left held-out recall at chance; more
   recall data HURT: w3 > w10 > w20).

Probe commands (CPU-safe alongside training):
`.venv/bin/python v13/tmp/dissect_ckpt.py <ckpt>` and
`.venv/bin/python -m v13.eval_checkpoints --checkpoints <ckpt> --labels wiki --batch_size 2`.

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
