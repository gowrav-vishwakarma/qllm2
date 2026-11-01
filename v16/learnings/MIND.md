# Mind learnings (v13_sempty body), self-contained

Each entry gives: the idea (maths), the recipe, the numbers, the verdict and the lesson. Units and conventions:
- "holdout" = mean token CE on a 500k-token dclm-edu holdout cache (edu ≥ 3, 5 % hash bucket, excluded from training);
- "WikiText" = WikiText-103 val PPL;
- "free" = greedy argmax over the whole vocabulary equals the answer (the headline from MLB-1 A7 on);
- "20-way" = the answer ranks first among the candidates;
- params ≈ 103M unless stated. Baselines are T++ (RMSNorm / RoPE / SwiGLU, 101.9M) and Mamba-2 (`mamba_ssm` 2.2.5, 102.6M),
  both trained on the same data, run as they are.

## 1. The body (PAM)

- **Delta notebooks** (per head, K × K state): S_t = g_t · S_{t-1} (I − b_e,t k_t k_tᵀ) + b_w,t v_t k_tᵀ, read r_t = S_t q_t / √K.
  - g_t is the retention, a content-dependent decay exp(−softplus(·)); b_w / b_e are the write / erase gates; k is unit-normed.
  - Chunked WY form for training; log-space retention with ε = 1e-6.
- **Chrono:** a content-modulated rotary phase, i.e. RoPE whose angle increments are scaled by a learned, content-dependent
  factor (identity at init). WikiText 23.81 → 23.14 at small scale.
- **Out gate:** silu(W x + b) per head on the read (23.14 → 22.96).
- **Echo memory (E1, "succession"):** a delta memory with key = unit(W_k h_{t−1}) and value = rms_unit(E[x_t]) (the embedding of
  the token that FOLLOWED a state like h). It is read with q = W_q h_t and added to the pre-logit vector through a zero-init
  projection, so it acts as a learned induction / copy head. −0.052 holdout at 600M.
- **M1:** notebooks only in blocks 3, 7, 11, 15; the other blocks are gated MLPs (MLP hidden 2128 to param-match); 1.8× faster.
  **M2:** each notebook-less block above the first notebook READS the nearest notebook below, with its own query rotated
  by the writer's Chrono clock, its own gate and a zero-init out-proj. Best 600M LM. State 1.15 MB (fp32 at decode).
- **16nb:** a notebook in every block (no shared reads). Better at 3B with LR 1.2e-3 (r3c).
- **Streaming:** chunked prefill carries the notebook state; decode is O(1) per token (parity 2e-4 / 5e-5).

## 2. Recipes and scale

- **Two stages:**
  - stage A: 120M tokens of dclm / fineweb-edu / smoltalk2_mid (+ "aptitude" synthetic docs in c1m2), 30M web-only warmup;
  - stage B: the main mix, `--init_from` stage A (the data stream continues after A's cursor).
  - The schedule (not the aptitude data) caused the recall jump (C1-ctl).
- **Optimiser:** LR 1.2e-3 cosine, warmup 100 / 1000, batch 18 × 2048, bf16, weight decay 0.01, clip 1.0.
- **Answer weighting:** synthetic answer spans weigh ×100 in sum(w · CE) / sum(w). Recall was learned only with it.
- **600M results** (stage B 480M; 102.9M):

| run | holdout | notes |
|---|---|---|
| c1m2 (M2) | 3.4473 | – |
| b2p1 / b2p2 / b2p3 (M2 + b2 3 %) | 3.4777 / 3.4800 / 3.4775 | – |
| T++ | 3.4885 / 3.5572 | the second is with b2 |
| Mamba-2 | 3.4349 / 3.4754 | the second is with b2 |

- **3B results** (3.0B tokens):

| run | holdout | WikiText | late slope 2→3B | recall (t40) all / ≤ 2k / > 2k |
|---|---|---|---|---|
| r3c (16nb + echo, LR 1.2e-3) | 3.2191 | 49.95 | −0.080 | 0.792 / 0.844 / 0.661 |
| T++ | 3.2064 | 46.61 | −0.081 | 0.539 / 0.693 / 0.153 |
| Mamba-2 | 3.1644 | 46.91 | −0.079 | 0.613 / 0.697 / 0.403 |

  - PG-19 loss at 4–8k: r3c 3.572 holds, T++ 5.535 collapses, Mamba-2 3.537.
  - **Lesson:** this body keeps pace at 3B (equal late slope) and holds context far better, but stays 0.013 / 0.055 behind on LM.
  - r3b (M2, LR 2e-4) failed at 3B (3.3094): 100M screens cannot pick PAM's LR; screen at ≥ 600M.

## 3. Rules in context (B2, s8)

- **B2 suites:** letter shift / numeric step / affine map, 8 shots, held-out VALUES (`_new`) never trained. The "b2" source puts
  the base tasks with TRAINED values in the mix at 3 %.
- **s8 (spread cell):** examples spread across 2–4k tokens of unrelated text; D = mean accuracy at 2k and 4k.
  - PAM 0.645 / 0.691 / 0.741 (3 seeds); T++ 0.134 / 0.134; Mamba-2 0.116 / 0.098.
  - **Worst pair +0.511** (bar +0.10): rules inferred from spread examples on unseen values survive past the trained length.
  - Holdout at parity with Mamba-2 (3.4775 vs 3.4768).
- **Label gap = format:** with label docs (lab 3 %), PAM label_diag 0.934 vs T++ 0.798.

## 4. One-read reach (skill pack)

- **Cell:** pack = header + 8 demos read ONCE at t0, then L tokens of held-out prose, then the query.
  - b2p2 20-way at 0 / 2k / 4k / 8k / 32k: 1.00 / 0.86 / 0.355 / 0.10 / 0.04; free at 2k / 4k: 0.615 / 0.17.
  - T++ / Mamba-2 at 2k: 0.03 / 0.06.
- **LAW-1 lg32** (fine-tune at 32k windows, 150M, long-gap rule docs): reach SHRANK (2k 0.86 → 0.31); every echo half-life fell
  25–55 %. Packed windows of unrelated docs reward forgetting.
- **lg8r** (8k windows + a state reset at each document separator + answer-in-window, 50M):
  - reach(4k) 0.465 (bar 0.75); long-head half-life geomean held (1304 vs 1361);
  - LM +0.08 / +0.11, caused by the warm restart on an annealed checkpoint.
  - **Lesson:** the training window does not set the reach; the decays do.
- **Reset mask:** retention 0 at the separator in every notebook; in the echo memory, retention 0 and the key zeroed there.
  Parity vs a fresh sequence ≤ 3e-4 (batch-consistent).

## 5. Query-time contrast (cd1–cd3)

- **Idea:** l = l_pack + a (l_pack − l_nopack), where l_nopack is the same stream without the pack; a = 4, at the answer only.
- **Reach at 4k** (fresh seeds): free 0.17 → 0.575, 20-way 0.355 → 0.78. 8k ≤ 0.18 (the trace is gone).
- **Must be gated:** on prose after an irrelevant pack it destroys text (NLL 3.90 → 8.56, " 64 64 64").
- **Package:** a warm read + no-row contrast became the default fetch read (+0.142, Library A9). It only needs a state copy.
- Prior work: CAD (Shi et al. 2023), contrastive decoding (Li et al. 2022).

## 6. Decay dilation (clk1 / clk2 / clk2r)

- **Idea:** between a package-paired read and its query, every retention r → r^(1/s).
- **All memory, s = 4:** reach is kept, but prose costs +0.18.
- **Echo memory only:** prose +0.013 (s = 4) / +0.016 (s = 8); the trace lives in the echo memory.
- **Echo, s = 8, + contrast, one read, 8k later:** free 0.49 / 0.58 / 0.43, 20-way 0.74 / 0.89 / 0.81 on b2p2 / cu2a / b2p1
  (three seed sets). At s = 4, b2p1 fails (0.34).
- Baselines at the same 8k gap, as they are: T++ 0.195, Mamba-2 0.015 (20-way).
- **In the package** (Library A16): a refresh every 8k with dilation 0.42 < every 2k plain 0.63 < every 2k + contrast 0.70.
  The package default stays 2k + contrast.

## 7. Curricula

- **cu1** (srd stated rule + demos; fsq definitions with 25 % absent keys → " unknown"; hop two-hop records):
  - abstain 1.00, two-hop 0.57 (free 0.20), LM +0.045;
  - **the stated rule is a lookup over trained values:** 0-shot trained k 0.97, held-out k 0.00; one demo gives 0.97;
  - fsq's abstain transfers to natural keys in the short re-read format (" unknown" 0.86 vs b2p2 copying 0.59).
- **cu1n9s** (content-phase echo lags + a salient phase): two-hop 0.09, holdout +0.228. FAIL.
- **cu2a** (fld one field from a multi-field row + stop; vfy verify a fetched row; +2 % synthetic):
  - trained formats: one field 0.80, verify 0.83–0.99, run-on 0.00, LM +0.0175;
  - held-out question wordings 0.26–0.46; held-out row layout ~0.00;
  - unfamiliar field names read as "absent" (A17: " unknown" 80–83 %).
  - **Lesson:** at 103M skills generalise VALUES and KEYS, not FORMATS.
- **cu2b ACh-1** (distil the contrast into the logits after an answer tag; teacher trusted only where its top token is gold):
  +0.038 LM at 180M, futility FAIL, removed. The first smoke without the gold filter collapsed tagged accuracy to 0.
- **Read gap** (Lead's T++ + library arm, same library, cold): tool rows T++ 0.90 vs PAM 0.475 / Mamba-2 0.45 (the fixed-state
  multi-item recall weakness, MQAR). The package field cut (Library A19) lifts PAM to 0.90 / 0.80. A window-reader repair (W1:
  sliding-window attention, window 128, blocks 5 / 9 / 13, +4.15M params) is in a 100M screen (w1s).

## 8. Budgets and data hygiene

- **Orphaned answers:** the packer cuts docs at window edges, so an answer whose doc began in an earlier window has no evidence.
  - Shares at 2k: b2 3.3 %, recall 15.9 %, srd 22 %; long-gap docs at 8k 44 %.
  - Fix: weight 1.0 for answers whose doc did not start in the window.
- **Loss-weight share:** with ×100 answers, answer-dense docs take loss weight from text. Text share: r3c 0.975, b2p2 0.872,
  r3p 0.85, cu2a 0.758. r3p (3B, +3.4 points synthetic) trailed r3c by 0.10–0.15. **Budget by loss-weight share, not tokens.**
- **Exact gradient accumulation:** micro-batch losses scaled by their weight share w_i / W reproduce the full batch (2e-5).

## 9. Process lessons

- Pre-register bars and a futility stop. A failed path leaves the code, and its row names the last commit.
- Read the clock before writing a time.
- tmux `has-session -t name` matches PREFIXES: use `=name`. A `.stop` file must block auto-resume.
- Run from pinned worktrees.
- Screen micro-bench winners at 100M LM before 600M (N9a failed three times).
- Fine-tuning an annealed checkpoint pays a warm-restart cost: keep a stable-phase checkpoint (WSD) to branch from.
- TODO: back-fill the pre-2026-10-31 failed paths (v13_sempty EXPERIMENT_INDEX "Failed and removed") in this format.
