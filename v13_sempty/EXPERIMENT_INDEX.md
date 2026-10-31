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
| E1 echo | top-of-stack succession memory (key h_{t-1}, value E[x_t]) into the head | **PASS** `e1g` (6000, `a0fab65`) 600M: holdout 3.4885 vs 3.5408 (−0.052), WikiText 71.1 vs 75.9, recall 0.401 vs 0.343, PG-19 far gain 2.4× | "E1 Echo memory" | `--echo` / `ECHO=1` |
| M1 sparse memory | notebooks in blocks 3,7,11,15 only + echo, MLP 2352 (param-matched, 1.8× faster) | **KEEP** `m1_m1echo` (`ccbf028`) 600M: holdout 3.4906 (echo 3.4885), WikiText 69.4 (71.1), recall **0.615** (0.401), 1.8× faster | "M1 sparse memory" | `PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2352` |
| M2 shared reads | M1 + every notebook-less block reads the nearest notebook below (own query/gate) | **KEEP — best LM** (`ed326a1`, 600M): holdout 3.4697 vs M1 3.4906 (−0.021, lead held at every milestone), WikiText 68.84; recall 0.522 vs 0.615 (lower), 1.18× slower than M1; learning probe 0.185 | "M2 shared reads" | `SHARE_READS=1 PAM_LAYERS=3,7,11,15 MLP_HIDDEN=2128` |

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
| G2 read norm | per-head RMSNorm of the PAM read | FAIL (futility): +0.14 nats vs ctl on the same batches at step 2250, not closing | "G2 read norm — stopped" | `3cc2b72` (`gscreen_onorm`) | `d4a57d1` | `OUT_NORM=1` (`--out_norm`) in `tmp_pretrain_mix.sh` |
| T1 thought stream | 12 token blocks + 4-block PAM thinker over 16-token events (mean+last pooled after block 5), zero-init top-down into block 6; param-matched +1 %, 1.23× faster | FAIL @600M: holdout 3.4977 vs echo 3.4885 (+0.009; lead −0.031 @150M shrank every milestone); PG-19 far 0.0448, recall 0.539 (8-fact 0.357), learning probe 0.171 | "T1 thought stream" | `ea002e6` (`t1loc_t1`, 4090) | `5eff3b9` | `SCREEN=t1loc ARMS='t1:ECHO=1,N_LAYERS=12,THOUGHT=1' BASE='LR=1.2e-3 TARGET_TOKENS=599998464 VAL_EVERY=4069 GRAD_CKPT=1' bash v13_sempty/tmp_scratch_screen.sh` |
| E2 tied echo | echo with W_q = W_k (one Linear(dim, dim) for q and k): an induction-style lookup | FAIL @600M (M1 body, C1-ctl recipe): label_cat 8-shot 0.365 vs 0.500 (400 trials), holdout 3.5298 vs 3.4743 (+0.055), recall 0.602 vs 0.825; micro-bench gain (v14 B1 runs 2/4) did not transfer | "E2", "E2 result" | `e5b1a2a` (`e2tb_main`, 4090) | `5100fd7` | `ECHO=1,ECHO_TIED=1` with C1-ctl's recipe: `~/e2_chain.sh` from a worktree at the code-at hash (EXPERIMENTS "E2") |
| Q1 tied start | echo W_q initialised = W_k, then trained freely (`--echo_qk_tied_init`) | FAIL at F1: 0/5 seeds formed label_category (0.67/0.42/0.42/0.43/0.42) vs echo 1/5, N9a 5/5 | "Q1", "Box control result + F1 result" | `3208b64` (F1, 4090) | `8b3ff59` | `ECHO_QK_TIED_INIT=1`; F1: `TAG=... ARMS=echo_qinit` in v14/microbench at `6187681` |
| S1 separated keys | echo q/k centred by the running key mean + top-K per head (`--echo_sparse_k`) | FAIL at F1: 0/5 seeds (0.44/0.41/0.58/0.43/0.41); offline the key overlap dropped 0.67 → 0.25, but the circuit did not form | "S1 separated echo keys + F1" | `3208b64` (F1, 4090) | `8b3ff59` | `ECHO_SPARSE_K=8`; F1 arm `echo_sparse` at v14 `6187681` |

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
| Transformer++ 600M | same model, r600 protocol, one stage (`tfs1_tf_b788731`) | DONE: holdout 3.4885 (M2 3.4697, M1 3.4906), WikiText 66.38, recall 0.645 (≤2048 0.839; 8192 0.15), PG-19 far −1.095, learning 0.217 (label_cat 8-shot 0.73) | "S2 fairness, half 1" | `ARCH=transformer CHRONO=0 OUT_GATE=0 DELTA=0`, run `b788731` |
| Mamba-2 600M | Mamba-2 102.6M, r600 one stage (`mbs1`) and two stages (`mbs2`), `685de4e` | DONE: holdout 3.4349 / 3.4393, WikiText 66.25 / 65.04, recall 0.594 / 0.762 (> 2048: 0.36 / 0.46), learning 0.199 / 0.227 (label_cat 8-shot 0.47 / 0.55); vs c2m2: Mamba-2 −0.010 LM, PAM +0.086 recall (+0.26 past 2048), PAM state ~1.2 MB vs 13.4 MB | "S4 Mamba-2 at 600M result" | `ARCH=mamba2`, run `685de4e` |
| PAM reference at 3B (`r3b`) | c1m2 recipe, stage B 2.88B tokens, LR 2e-4 (`685de4e`) | **FAIL**: holdout 3.3094 vs T++ 3.2064 / Mamba-2 3.1644 / old PAM 3.2728; WikiText 58.0 vs 46.6; recall 0.620 (beats T++ 0.539). The 600M lead does not survive scale; PAM learns fast early, then plateaus | "S4 scale test result" | run `685de4e` |

## Running / pending

| id | idea | status | notebook section |
|---|---|---|---|
| Label-format pair (`lbp` / `lbt`) | `lab` source (`c4173be`): label-rule docs, probe format, disjoint vocabulary, 3 % of stage B; PAM vs T++ on the c1m2 recipe | **PENDING** (6000, tmux `s7` after s6, next to r3c): PASS if PAM label_diag ≥ 0.767; vs T++ ≥ lbt − 0.02 = gap closed | "Label-format pair `lb`" |
| n9c: v14 N9a phase keys at 600M | C1-ctl recipe + `ECHO_PHASE_LAGS=3` (4090, `e5f0b73`) | **FAIL (lookup)**: label_exact 0.472 vs ctl 0.610, label_cat 8-shot 0.318 vs 0.500 (both worse); keys de-collided (cos 0.715 → 0.538); **recall 0.975 vs 0.825 (> 2048: 0.924 vs 0.597)**; holdout 3.4725 vs 3.4743. No v13 scale run; the unit stays in v14 | "n9c result" |
| B2-in-training pair (bar 3) | `b2` rule docs (3 %, held-out values) in stage B; PAM b2p1/b2p2 vs T++ b2t + Mamba-2 b2m (`38e3c29`) | **vs T++ MET both seeds; vs Mamba-2 NOT MET** (`_new` mean PAM 0.991 / 0.944, Mamba-2 0.892: +0.099 / +0.052, bar +0.10) → bar 3 not met in its pre-registered form. **Distance edge is PAM's alone:** shift/affine `_new` @2048+4096 mean PAM 0.673 / 0.676 vs Mamba-2 0.084, T++ 0.125 (both collapse at ~110-token gaps). Mamba-2 leads labels (0.787) and exact far copy (step through filler 0.27-0.31); holdout PAM 3.4777 / 3.4800, Mamba-2 3.4754, T++ 3.5572. Distance confirmation pre-registered and queued: tmux `s8` (b2t2, b2m2, b2p3; probe seed 7000) | "B2 pair — seed-42 half", "seed-43 result", "B2 Mamba-2 arm b2m" |
| G1 short conv | conv on qkv, single-pass pretraining (`g1c`, 4090, `9c2b632`) | **FAIL as measured, verdict held open**: label_exact 0.517 vs ctl 0.610, holdout +0.049; E2 and g1c (both 4090) lose the same ~0.05 to the 6000 control, so a 4090 box control runs first | "g1c result" |
| Learning probe (bar 3) | in-context rule tasks, answer must be inferred (`ed326a1`) | baselines: all ≤ chance on letter/number/reverse incl. Transformer++ 3B (mean 0.228); follow priors ("last+1") | "Learning probe" |
| B2 probe (bar 3, selection-proof) | v14 rules_ood via `run_learning_probe.py --suite b2` (`fa404dc`): nearest-parameter answers among the decoys | every 600M/3B model BELOW chance 0.125 (0.037–0.094; T++ 3B 0.088, Mamba-2 3B 0.094, best 600M c1m2 0.071): nobody infers these rules exactly (decoys = answer ± 1 step, so imprecise inference and selection look alike); `_new` 0.015–0.047 below base (difficulty control) | "B2 probe" |
| LCL long-context learning | rule prompts with examples spread over 1k–8k tokens of held-out text (`--spread`, `7707c8c`) | DONE: learning holds past 2048 for NO model; PAM vs T++ at 4096 +0.09–0.13, at 8192 +0.03–0.07 (rule NOT met); past 2048 PAM = in-context chance 0.25 (keeps the label set), T++ = 0.125; rule tasks at the floor for all | "LCL", "LCL result" |
| label_diag (exact vs fuzzy lookup) | `--suite label_diag` (`bc7d57f`): label_category + label_exact | DONE: PAM exact copy 0.50–0.62 vs T++ 0.78–0.88, Mamba-2 0.62–0.72; not capacity (3.3× state no help), not data (3B same); lost to near-duplicate keys (recency gradient 0.72 → 0.29 over 5 pairs; key cosine 0.67–0.72) | "label_category diagnosis" |
| C1 reason-first curriculum | grammar 30M → real aptitude Q&A + chat 90M → normal mix 480M (M1 body, 600M total) | **FAIL primary** (`5b1eb9d`): learning probe 0.199 vs 0.175; **recall 0.907 vs M1 0.615** (unexpected, measured @ step 9766); control `c1ctl` + C1-on-M2 running (`7ae6f1f`) | "C1 result", "C1-ctl" |
| C1-ctl two-stage schedule (S2) | 120M web+chat (no recall docs, no aptitude) → 480M M1 mix, LR re-warm | **beats M1 on both**: holdout 3.4743 vs 3.4906, recall 0.825 vs 0.615 (`7ae6f1f`); aptitude adds +0.10 recall, costs +0.026 holdout | "C1-ctl" |
| C1 on M2 (`c1m2`) | C1's aptitude two-stage recipe on the M2 body | **KEEP — best LM at 600M** (`7ae6f1f`): holdout **3.4473** (M2 3.4697, C1-ctl 3.4743), WikiText 65.74, recall **0.870** (M2 0.522); learning probe 0.187, label_cat 8-shot 0.38 (no lift on M2); **the reference** (picked over c2m2 by the pre-registered tie-break, 2026-10-30) | "C1 on M2 result" |
| S2 fairness | Transformer++ one-stage vs two-stage at 600M | **DONE (`b788731`)**: T++ gains nothing on holdout (3.4885 → 3.4907) and +0.058 recall vs PAM +0.210 → schedule gain PAM-specific; head-to-head two-stage M1 vs T++: holdout −0.016, recall +0.122 (rules MET) — but ≤ 2048-tok recall ties (0.917 vs 0.926), WikiText T++ better (66.2 vs 69.9), learning probe T++ better (label_cat 0.73 vs 0.55) | "S2 fairness" |
| tfc1 | Transformer++ on c1m2's aptitude recipe (fairness for c1m2) | **DONE (`b788731`)**: holdout 3.4802, WikiText 63.83, recall 0.648 (gain +0.003 vs PAM +0.348: PAM-specific); c1m2 beats it on holdout (−0.033) and recall (+0.222), loses WikiText (65.7 vs 63.8) | "tfc1 result" |
| S2 on M2 (`c2m2`) | C1-ctl recipe (no aptitude) on the M2 body | **KEEP** (`b788731`): holdout 3.4497, WikiText **65.57**, recall 0.848 (> 2048: 0.718); vs T++ two-stage: holdout −0.041, WikiText −0.6 ppl, recall +0.145; learning 0.185. Reference pick: c1m2 by the tie-break (recall +0.022 > 0.02) | "S2 on M2 result + next reference" |
