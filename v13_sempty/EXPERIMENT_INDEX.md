# v13_sempty experiment index — what was tried, verdict, where the code lives

**Read this before proposing an experiment** so nothing is run twice. One row
per idea. The full write-up (maths, numbers, reasoning) is the named section
of `EXPERIMENTS_SEMPY.md`.

**Removed code stays reproducible.** A failed idea's code is deleted from the
working tree (AGENTS rule), but it still exists at the commit in the *code
at* column, the last commit that has it. To rerun it:

```bash
git worktree add ../qllm_<id> <code-at-hash>     # side checkout, master untouched
cd ../qllm_<id> && <command in the row>           # launcher env names as of that commit
```

(The launchers refuse a dirty tree; a worktree is clean. Checkpoints and
logs of the original run are named after the *run* hash.)

## Kept (in the current code)

| id | idea | verdict / key number | notebook section | flag / env |
|---|---|---|---|---|
| real arm | real-arithmetic PAM, param-matched `baseline_real_pm` (102M) | KEEP: matches complex PAM, faster | "Experiment ledger", "Speed: fused real arm" | preset `baseline_real_pm` |
| N1 Chrono | content-modulated rotary retention (learned time-warp in RoPE phase) | KEEP: WikiText 23.81 → 23.14 | "N1 Chrono-PAM" | `--chrono` / `CHRONO=1` |
| N4 gate | per-head read-out gate silu(Wx+b) | KEEP: 23.14 → 22.96 | "N4 read-out gate" | `--out_gate` / `OUT_GATE=1` |
| A3 delta | delta erase/write, unit keys, WY chunked form | KEEP: micro-bench a1 0.46 → 1.00; Phase 3b multi-way recall | "A3 delta rule", "Phase 3b" | `--delta` / `DELTA=1` |
| answer weight | CE ×100 on recall answer tokens | KEEP: recall learned only with it | "Phase 3b" | `ANSWER_W=100` |
| LR 2e-4 | PAM LR at 3B tok | KEEP 2e-4: 1.2e-3 (picked by the 100M screen, −0.62 nats at 100M) is WORSE at 3B: holdout 3.2842 vs 3.2728, recall 0.602 vs 0.667; behind from step 6000. **100M screens cannot pick PAM's LR — screen PAM on ≥ 600M tok** | "Scoreboard result" | `LR=2e-4` (run `9982a79` was 1.2e-3) |
| A2 / A2b | n states per head; vault (pinned state + protect gate) | still in code, not adopted (cost, no decode) | "Retention ladder" | `--n_states`, `--vault` |
| A4 cond_mem | hashed n-gram conditional memory | still in code, not adopted | config comments | `--cond_mem` |

## Failed and removed (code at the listed commit)

| id | idea | verdict / key number | notebook section | run hash | code at | rerun (at *code at*) |
|---|---|---|---|---|---|---|
| A1 short conv | causal short conv on q/k/v | FAIL: 23.49 vs 23.14, −17 % tok/s | "A1 short conv on qkv" | `817aa35` | `2537782` | `--short_conv` (WikiText fair driver) |
| R1 dt-spread | per-head ladder of initial decay biases (half-lives 38 → 113k) | FAIL: horizon flat at T=8192, biases frozen at init | "L1 / R1 dt-spread" | `18ed358` | `4f30d23` | `DT_SPREAD=8` in `tmp_pretrain_mix.sh` |
| A2r routing | S content-routed delta memories (write by key, read by query) | FAIL: a8 at chance, 4–8× cost (micro-bench) | "A2r content-routed delta" | `dae4651` | `dae4651` | `v13_sempty/tmp/recall_microbench.py`, route arms |
| R2 no_decay | retention pinned to 1, delta erase only | FAIL at 100M/3B: holdout PPL 30.16 vs 26.38, recall worse everywhere | "R2 at scale" | `9f33b50` | `e7e2a0b` | `--no_decay` / `NO_DECAY=1` |
| L-2 data pressure | 8K context + long-gap recall docs | FAIL as a horizon lever; dt_bias stays at −4 | "Stage L-2" | `5c87148`-era | data-only (current code) | see section |
| R3 split prior | 2 of 6 heads born at dt −9 | FAIL: far horizon +≤0.08, short multi-way collapsed | "R3 split decay prior" | `5c87148` | `215fe61` | `--long_heads 2 --long_dt_bias -9` |
| R4 store | consolidation store beside the notebook | FAIL: gates ~0.05, unused; multi-way −9 pts | "R4 consolidation store" | `0868a1b` | `b1cbc74` | `STORE=1` (warm start, `tmp_r4_screen.sh`) |
| base_dt_bias −6/−8 | longer default retention | no gain (retention bench) | "R2 retention bench" | `54c62ed` | current (`--base_dt_bias`) | `BASE_DT_BIAS=-8` |
| C1 coupled gates | one step dt: write = erase = 1−e^{−dt}, fade e^{−A dt} (Mamba-2 coupling on A3) | FAIL @300M: far a1 0.10 vs 0.26, holdout +0.055; fresh multi-fact +0.17 | "C1 result" | `91daf0a` (600M), `9ebd1d2` (100M) | `4c688cb` | `COUPLE=1 COUPLE_A=0.1` (screen: `tmp_scratch_screen.sh`) |
| torch.compile blocks | `compile_blocks()` on the step | no gain: 171 → 177 ms/step (4090, B4), named-tensor graph breaks | — (2026-10-28 profile) | — | current | `model.compile_blocks()` |

## Equal-data scoreboard (3B tok, ~102M) — 2026-10-29

| model | holdout loss / PPL | WikiText PPL | recall 63-cell | tok/s (6000) | state |
|---|---|---|---|---|---|
| Mamba-2 `b36e3a5` | **3.1644 / 23.68** | 46.91 | 0.613 | **189K** | 13.4 MB |
| Transformer++ `e072a8f` | 3.2064 / 24.69 | **46.61** | 0.539 | 138K | 32 KB/tok |
| PAM 2e-4 `1c913ef` | 3.2728 / 26.38 | 54.73 | **0.667** | 66K | **3.7 MB** |
| PAM 1.2e-3 `9982a79` | 3.2842 / 26.69 | 55.83 | 0.602 | 66K | 3.7 MB |

Any PAM change must close the ~0.07–0.11 nat LM gap ("Scoreboard result").

## Baselines (equal data, same trainer and evals)

| id | model | status | notebook section | code |
|---|---|---|---|---|
| Mamba-2 | 102.6M `mamba_ssm` 2.2.5, LR 1.2e-3, 3B tok | DONE: holdout 23.68, WikiText 46.91 | "Mamba-2 result" | `baseline_mamba2.py`, run `b36e3a5` |
| Transformer++ | 101.9M RMSNorm/RoPE/SwiGLU, LR 1.2e-3, 3B tok | DONE: holdout 24.69, WikiText 46.61, recall mean 0.539 (fails past 2048) | "Transformer++ 3B result" | `baseline_transformer.py`, run `e072a8f` |

## Running / pending

| id | idea | status | notebook section |
|---|---|---|---|
| G2 read norm | per-head RMSNorm of the PAM read | FAIL (futility): +0.14 nats vs ctl on the same batches at step 2250, not closing | "G2 read norm — stopped" |
| G1 short conv | conv on qkv, single-pass pretraining regime | deferred (e1g conv arm skipped for M1) | "G screen" |
| E1 echo | top-of-stack succession memory (key h_{t-1}, value E[x_t]) into the head | running, `e1g` (6000, `a0fab65`): −0.068 / −0.060 nats at 150M / 300M | "E1 Echo memory" |
| M1 sparse memory | notebooks in blocks 3,7,11,15 only + echo, MLP 2352 (param-matched, 1.8× faster) | queued after e1g (`m1chain`) | "M1 sparse memory" |
| M2 shared reads | M1 + every notebook-less block reads the nearest notebook below (own query/gate) | queued after m1 (`m2chain`) | "M2 shared reads" |
