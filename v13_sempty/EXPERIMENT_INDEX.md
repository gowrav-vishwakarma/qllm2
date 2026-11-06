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

## Run ingredients ledger — which findings each big run contains (user rule, 2026-10-31)

Long runs on the 6000 start before the 4090's newer findings exist. Before
comparing a big run with anything, read its row: a result only tests what is
IN it. Add a row when a big run starts; add a column when a finding is KEPT.

Findings (columns): **16nb** = notebooks in all 16 blocks (M2 = blocks 3,7,11,15 +
shared reads) · **E1** echo memory (10-29) · **2st** two-stage schedule, stage A
120M (apt = with aptitude data) · **LR** · **rec** recall docs share (answer weight
100) · **b2** rule docs, held-out values (10-31, rule inference + distance edge) ·
**lab** label docs (pending, lb pair) · **N9a** echo phase lags (v14; recall
0.975 @600M, replication n9r pending) [N9a + N9s REMOVED 2026-11-01 after cu1n9s FAIL; last code `d255fcd`, rerun `ECHO=1 ECHO_PHASE_LAGS=3 [ECHO_PHASE_SALIENT=1] bash v13_sempty/tmp_pretrain_mix.sh` in a worktree at `d255fcd`] · **N12** selective store (v14, micro pending).

| run (hash) | box | tokens / params | body | E1 | 2st | LR | rec | b2 | lab | N9a | N12 | answers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mamba-2 3B (`b36e3a5`) | 6000 | 3B / 102.6M | Mamba-2 | – | no | 1.2e-3 | 5 % | no | no | – | – | 3B baseline: 3.1644 / 46.91 |
| T++ 3B (`e072a8f`) | 6000 | 3B / 101.9M | T++ | – | no | 1.2e-3 | 5 % | no | no | – | – | 3B baseline: 3.2064 / 46.61 |
| PAM 3B (`1c913ef`) | 6000 | 3B / 102.1M | 16nb | no | no | 2e-4 | 5 % | no | no | no | no | old body at 3B: 3.2728 |
| r3b (`685de4e`) | 6000 | 3B / 102.9M | M2 | yes | apt | 2e-4 | 5 % | no | no | no | no | M2 + echo + 2st at 3B: 3.3094 (FAIL) |
| **r3c** (`38e3c29`, DONE) | 6000 | 3B / 103.2M | 16nb | yes | web/chat | 1.2e-3 | 5 % | **no** | **no** | **no** | **no** | 3.2191 / 49.95: NOT beat (T++ +0.013, Mamba-2 +0.055); recall 0.792 (>2k 0.661) vs 0.539 / 0.613. NOT the best package |
| b2p1 / b2p2 (`38e3c29`) | 6000 | 600M / 102.9M | M2 | yes | apt | 1.2e-3 | 4 % | 3 % | no | no | no | bar 3 rules (with b2t, b2m) |
| s8: b2t2 / b2m2 / b2p3 (`38e3c29`, queued) | 6000 | 600M | as b2 pair | | | | 4 % | 3 % | no | no | no | distance confirmation |
| lbp / lbt (`c4173be`) | 6000 | 600M | M2 / T++ | yes | apt | 1.2e-3 | 4 % | no | 3 % | no | no | label gap = format? |
| n9c (`e5f0b73`) | 4090 | 600M / 103.5M | M2 | yes | web (C1-ctl) | 1.2e-3 | 5 % | no | no | yes | no | N9a lookup (FAIL), recall +0.15 |
| n9r ctl43 / n9a43 (`37fa333`, v14) | 4090 | 600M / 102.9M | M1 | yes | web | 1.2e-3 | 5 % | no | no | no / yes | no | N9a recall replication **FAIL** (+0.016 / +0.014, bars +0.08 / +0.15); ctl43 is a loss outlier (3.598). **N9a is out of every plan.** |
| **r3p** (`c13d8fe`, STOPPED by futility at 1.44B: 3.5104 vs 3.431; text share 0.86) | 6000 | 3B / ~103.2M | 16nb | yes | web/chat (r3ca reused) | 1.2e-3 | 3–4 % | 3 % | 2–4 % | by cu1n9s rule | no | 3B best-package mind: b2 + lab + cu1 parts by rule + ANSWER_IN_WINDOW; expression bar ±0.01 vs T++ (predicted miss) |
| **lg8r** (`f2b7d8c`, DONE: FAIL, reach 4k 0.465) | 4090 | b2p2 + 50M / 102.9M | M2 | yes | apt | 3e-4 | 4 % | b2gap 3 % | no | no | no | 8k windows + doc-boundary reset + answer-in-window: does one-read reach follow the window? No cu1 curriculum, no N9s |

**Direction (user, 10-31): O(1) mind + growing library.** The state is the mind; verbatim knowledge goes to a session library + a shared library; benchmark first (see EXPERIMENTS "Direction").

**Not yet in any 3B run:** b2 rule docs, lab (PASS), fetch training, N12 (pending micro). N9a is out (n9r FAIL); N9s is out (cu1n9s FAIL); both removed from the code (last code `d255fcd`). A 3B "best package" run is the place
they meet; r3c's result says nothing about them.

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
| SL0 shared library (kNN-LM) | kNN over the frozen 3B PAM's pre-logit state on training text (5M / 20M rows, PCA-128), interpolated with the LM; closest prior work kNN-LM / RETRO (plumbing) | **FAIL** rule 1 (−0.05): 5M −0.001 / −0.004, 20M −0.006 / −0.012 (PG-19 / web); still +0.09 / +0.10 behind Mamba-2; 1.3 / 5.2 GB, install 106 / 328 s. A cost table, not a claim | "SL0" | `v13_sempty/tmp/sl0.py` |
| LAW-1 (reach vs training length) | fine-tunes from b2p2 with long-gap rule docs (`b2gap`, `e4d4fcd`), skill-pack reach at 0-32k | **CLOSED as a stated limit** (user): lg32 (32k) SHRANK reach (2k 0.86 → 0.31) and shortened every echo half-life 25-55 %; lg8 / lg2 / lg32-reset cancelled before start. One-read reach ≈ training length at 103M | "LAW-1 first arm lg32", "LAW-1 closed" | **b2gap source REMOVED in `45ea180`; code last at `abf6170`** (`git worktree add ../qllm_b2gap abf6170`, `SOURCES=...,b2gap B2GAP_MAX=<G>`). Reset mask KEPT (`RESET_AT_EOS=1`, `f2587b5`; kept: mechanism works, reach did not follow the window) |
| lg8r (window law, cheap check; user via Lead) | b2p2 fine-tune at SEQ 8192 × 2, 50M tokens, `RESET_AT_EOS=1` (`f2587b5`), `ANSWER_IN_WINDOW=1` (`824a988`) | **FAIL (mixed case: reset fixed the forgetting, reach did not follow)** `lg8r_main_f2b7d8c`: reach(4k) 0.465 vs bar 0.75 (b2p2 0.355; free 0.265 vs 0.170); half-life geomean 1304 vs 1361 (held; lg32 835); B2 `_new` 0.933 met; attr3b +0.111 / +0.076 FAIL (short warm restart; reset checked inert + batch-consistent). Reach-by-training STOPPED at this scale | "lg8r VERDICT" | `RESET_AT_EOS=1 ANSWER_IN_WINDOW=1 SEQ=8192 BATCH=2 B2GAP_MAX=7500` (RESET_AT_EOS kept: mechanism works, reach did not follow the window; ANSWER_IN_WINDOW kept: bug fix; b2gap source removed in `45ea180`, code last at `abf6170`) |
| cd1 / cd2 state-contrast decoding (eval only) | l_pack + a (l_pack − l_nopack) on the skill-pack cell, b2p2, CPU (`cb8053b`) | **cd1 FAIL as registered (2k free 0.70 < 0.81); cd2 MET on fresh seeds**: 4k 20-way 0.355 → 0.78 (a 4) / 0.855 (a 8), free 0.165 → 0.575 / 0.665; 8k ≤ 0.18, 16k chance. Reach ~2k plain → ~4k with contrast; cost 2nd O(1) state (+1.15 MB) + 2× compute | "cd1", "cd2" | `run_skillpack_probe.py --variants contrast --alphas 0,4,8` |
| cu1n9s (N9a + N9s paired arm, v14 unit) | cu1's stage B on its own N9s stage A (`fa9e410`, `ECHO_PHASE_LAGS=3 ECHO_PHASE_SALIENT=1`) | **FAIL** (`cu1n9sb_main_fa9e410`): two-hop 0.090 vs cu1 0.570 (rule ≥ +0.15); holdout 3.7533 (+0.228); label_diag 0.732; fsq false abstain 0.35. cu1n9a NOT run (A2). Code removal = Library's call | "cu1n9s VERDICT" | `ECHO_PHASE_LAGS=3 ECHO_PHASE_SALIENT=1` |
| R × K (rxk, eval only) | a rule read once into state (B2 pack) × a fetched fact row, L = 0 / 2k / 4k / 8k; explicit ("<v> ->") vs implicit ("K ->"); contrast / adaptive / echo dilation; ceiling + floor; T++ as it is (`99d2951`) | **FAIL as registered** (2k + contrast free 0.61 / 0.62 < 0.70; 8k + dilation s = 8 + contrast 0.45 / 0.49 ≥ 0.40 MET); floor ~0; T++ 0.01 / 0.07; implicit ≤ 0.07 for every model → the planner must substitute values | "rxk VERDICT" | `scripts/run_rxk_probe.py`; control `v13_sempty/tmp/rxk_nl.py` |
| W1 window reader (w1s + w1r) | sliding-window attention, window 128, in blocks 5 / 9 / 13, zero-init out-proj (+4.15M); 100M screen from c1m2a stage A (`508aaff`), seeds 43 + 44 | **reader: FAIL** (tool 0.600 → 0.6125; pooled +0.03; copy-class loss unchanged → the reader stays in the package, A19). **REACH unit: REPLICATES** (w1r): s8 D +0.171 / +0.323; beyond the windows' reach (s8 4k) 0.435 → 0.617 / 0.416 → 0.771; clk2 8k 20-way +0.22 / +0.12; holdout +0.006 / +0.002. Known hybrid (Samba / Griffin), no mechanism claim; 600M is the user's decision. **Window sweep (w1w, `f68ab64`): W512 / W1024 FAIL the lazy-state guard** (pack 4k + contrast 0.45 / 0.555, clk2 8k 0.32 / 0.43 vs W128 0.633 / 0.49; gain −0.11 / −0.06) → W = 128 stays | "w1s VERDICT", "w1r VERDICT" | `--swa_layers 5,9,13 --swa_window 128` / `SWA_LAYERS`, `SWA_WINDOW` |
| cu2 (cu2a fld + vfy data; cu2b + ACh-1 answer-mode distillation) | b2p2 recipe + fld 1 % + vfy 1 % (`a7c4887`), cu2b + `--ach_lambda 0.2` | **cu2a: trained formats MET** (one field + stop 0.80, verify 0.83-0.99, run-on 0.00; holdout 3.4975 ≤ 3.500); **R-fld-out FAIL** (held-out wordings 0.26; held-out row layout ~0; bench-format abstain not learned). **cu2b: FAIL by futility** (3.8069 vs 3.7691 at step 4883). **ACh-1 REMOVED in `d27281d`, code last at `7bdf568`** (`git worktree add ../qllm_ach 7bdf568`, `ACH_LAMBDA=0.2 ACH_ALPHA=4 ACH_FILL=4096`) | "cu2a result", "cu2b result" | fld / vfy kept (`SOURCES=…,fld,vfy`) |
| cu1 skill curriculum | b2p2 recipe + `srd` (stated rule + demos) / `fsq` (re-read spans, 25 % abstain) / `hop` (two-hop records), disjoint vocab, held-out values (`48674d6`) | **NOT a Win** (`cu1_main_2efb422`): R-fsq MET (abstain 1.00, present 0.94, false 0.06), R-hop MET (0.57; free 0.20); R6a / R6b FAIL (stated rule = lookup over trained values: 0-shot trained k 0.97, held-out 0.00); holdout guard FAIL (3.5254, +0.045). **srd REMOVED in `0b4bf51`, code last at `3c7a7fa`** (`git worktree add ../qllm_srd 3c7a7fa`, `SOURCES=...,srd`); fsq / hop kept | "cu1 VERDICT" | `SOURCES=…,fsq,hop` |
| A4 cond_mem | hashed n-gram conditional memory (Engram-style, 33.5M table) | **IN BETWEEN (rule 3)** `a4fb_main_d338469` vs C1-ctl: holdout 3.4488 vs 3.4743 (−0.026, met), new-token @0-64 web −0.062 (met), PG-19 −0.0275 (bar −0.03, missed by 0.0025); recall 0.916 vs 0.825. Not in the package; known (Engram), parametric knowledge: cost table | "A4 cond_mem filter a4f result" | `--cond_mem` / `COND_MEM=1` |
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
| Label-format pair (`lbp` / `lbt`) | `lab` source (`c4173be`): label-rule docs, probe format, disjoint vocabulary, 3 % of stage B; PAM vs T++ on the c1m2 recipe | **PASS, PAM ahead**: label_diag PAM 0.934 (exact 0.995, cat 0.873) vs T++ 0.798 (bar 0.767; ahead ≥ +0.05); holdout 3.4516 (cost +0.004), recall 0.846. The label gap was a data-format gap. `lbm` (Mamba-2 + lab) queued (tmux `s9`) before any claim | "Label-format pair `lb` result" |
| n9c: v14 N9a phase keys at 600M | C1-ctl recipe + `ECHO_PHASE_LAGS=3` (4090, `e5f0b73`) | **FAIL (lookup)**: label_exact 0.472 vs ctl 0.610, label_cat 8-shot 0.318 vs 0.500 (both worse); keys de-collided (cos 0.715 → 0.538); **recall 0.975 vs 0.825 (> 2048: 0.924 vs 0.597)**; holdout 3.4725 vs 3.4743. No v13 scale run; the unit stays in v14 | "n9c result" |
| B2-in-training pair (bar 3) | `b2` rule docs (3 %, held-out values) in stage B; PAM b2p1/2/3 vs T++ b2t/t2 + Mamba-2 b2m/m2 (`38e3c29`) | **Distance form MET (s8, fresh seeds + fresh probe seed 7000):** D PAM 0.645 / 0.691 / 0.741 vs T++ 0.134 / 0.134, Mamba-2 0.116 / 0.098; primary +0.607, all pairs +0.511 (bar +0.10). In-context form vs Mamba-2 NOT met (+0.099 / +0.052). Spread cell = ~250-500-token gaps; one-read reach ~2k is the stated limit. Holdout PAM 3.4775-3.4800, Mamba-2 3.4754-3.4768, T++ 3.555 | "B2 pair — seed-42 half", "seed-43 result", "B2 Mamba-2 arm b2m", "s8 VERDICT" |
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

## Lineage (user rule, 2026-11-01): every row here also has a line in `v13_sempty/EXPERIMENT_LINEAGE.tsv` (id, parent = what it extended, date, what, verdict, run hash, code at, log / result file, section); `python3 scripts/exp_tree.py` renders the root `EXPERIMENT_TREE.md`.
| sci1_cont4_worked | UNRESOLVED (11-06: confounded with the segment switch; s8 drops at every LR / segment change). Was: FAILED keep rule at M10 (+1B `worked`, step 152,590): s8 D 0.770 -> 0.418, far needles 0.98 -> 0.57, t40 16k 0.87 -> 0.33; `worked` dropped, decay branches from M9 | 8284510 | 8284510 (v13_sempty/tmp/sci1_cont4_env.sh; source sci1_env.sh + sci1_cont4_env.sh, sci1_cont_chain.sh) |
