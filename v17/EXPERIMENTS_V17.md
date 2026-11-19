# EXPERIMENTS_V17 — Two-Organ O(1) Mind (`POWLM`: Memory ≠ Intelligence)

## 0. Step Back in the Void (2026-11-19)

### 0.1 What surprised us (two accidents uncovered in the codebase + 2026 arXiv check)
1. **Literature reality check (why "cumulative phase + delta rule" is NOT enough):**
   - **Selective RoPE** (Movahedi et al., ICLR 2026, `arXiv:2511.17388`): input-dependent cumulative RoPE rotations (`Phi_t = sum_{j<=t} theta_j`) on GLA and Gated DeltaNet.
   - **Gated DeltaNet-2** (`arXiv:2605.22791`): decoupled write and erase gates on GDN/KDA.
   - **Semidirect Fourier Delta Attention (SFDA)** (Zhang, `arXiv:2607.11897`): complex diagonal phase-decay `Lambda_t = diag(alpha_t e^{i theta_t})` with chunk-WY delta memory.
   - Every one of those models—and our own `PAM-v2 Chrono`—treats phase purely as **1D cumulative elapsed time** (`Phi_t = sum_{j<t} dtheta_j`) inside a **1-pass passive key-value store**.

2. **Accident #1 — CPU Surgical Autopsy of `wio_pam` Seed 44 vs Seed 43 (2026-11-19 16:46 IST):**
   - On identical code, data (`1.0B` tokens), and GPU, `pam seed 44` scored `shift_wide @ 8k = 0.692` (`@0 = 0.997`) vs `seed 43`'s `0.233` (`@0 = 0.796`).
   - Surgical knockouts on `seed 44` (`shift_wide_new` unseen shifts):
     - Baseline `s44`: `@0 = 1.00 (+4.99 nats)`, `@4k = 1.00 (+2.06 nats)` (`s43`: `@0 = 0.90 (+1.72)`, `@4k = 0.625 (-0.06)`).
     - Zeroing `L3`, `L7`, `L11`, or `L15` notebooks' own 1-shot outputs (`pam_scale = 0`): **`1.00 (+4.91..+4.99 nats)` — zero loss!**
     - Zeroing **`readers 8..10`** (3 sequential blocks re-querying `L7`'s shared state `S^(7)_t` interleaved with MLPs): **`1.00 (+4.93) -> 0.10 (-2.57 nats)`!**
     - Zeroing **`echo`** (causal transition `h_{t-1} -> E[x_t]`): **`1.00 (+4.93) -> 0.00 (-2.47 nats)`!**
     - Freezing **Chrono** (`warp = 0` -> plain RoPE) on `s44`: **`@4k` drops `1.00 (+2.06) -> 0.625 (+0.41 nats)`** (`-1.65` nats lost).
     - Weight inspection: `seed 44` ended with **`25–44%` smaller static MLP scales (`cgu_scale`)** across middle blocks `4..11` than `seed 43`, preventing static MLP memorisation from starving the dynamic multi-read + causal-transition circuit!

3. **Accident #2 — Why `v14 N9s` hit `0.92` on `two_hop` in 1 seed and crashed (`+0.228` NLL) at 600M:**
   - Auditing `07f7d5d` (`v14/units.py` line 302) revealed `N9s` decayed raw angles in `R`: `psi_t = (1 - s_t) psi_{t-1} + s_t a_t`, then computed `(cos psi_t, sin psi_t)`. When `s_t > 0` is small on filler words, `psi_t` decays continuously toward `0` rad, **spinning `(cos psi_t, sin psi_t)` around the unit circle on every word**!

---

## 1. The `v17` Architecture (`POWLM` in `sempyt`)

Separates **Passive Binding Memory** from **Active Relational Intelligence**, keeping strictly `O(1)` state:

1. **Organ 1 — Dual-Phase Synchrony Memory (`DualPhaseWorkspaceLayer`):**
   - **Chrono Clock (`WHEN`):** `tau_t = sum_{j<t} exp(clamp(W_warp x_j, -4.5, 3.0))` on episodic planes `p >= P_sync` (`inv_freq = 0` on `0 .. P_sync-1` pure synchrony planes).
   - **Unit-Circle Synchrony Carrier (`WHOSE / WHICH SCOPE`):**
     `u_t = (cos alpha_t, sin alpha_t)`, `z_raw,t = (1 - s_t) z_raw,t-1 + s_t u_t` via `sempyt.scan`, projected onto `(S^1)^P` via `z_unit = l2_normalize(z_raw, over=rot_pair)`.
     When `s_t -> 0` on intra-clause filler, `z_raw,t` scales radially without changing direction, so `z_unit` has **zero phase spin** (`1.19e-7` over 200 steps in `v17.selftest`).
   - **Multiplicative Group Binding:** Under `SplitComplex(rot_pair)`, `rot_full = rot_time * z_unit` rotates `q, k` before the fused Triton delta kernel `pam_delta_batched`. Same role + same entity locks (`dphi = 0`) at any distance; same-role lures cancel (`<cos dphi> ~ 0`).

2. **Organ 2 — Active Intelligence Workspace (`Iterative Bounces` + `CausalOperatorEngine`):**
   - **Organ 2A (`n_bounces = 3` Iterative SRAM Bounces):** Re-queries the written `S_t` via `pam_shared_read` (zero WY re-solve, zero extra state bytes). Bounce `b` computes a fresh query AND re-locks its synchrony phasor `z^(b)_t` onto the hop-`(b-1)` entity retrieved in `h_work`, turning 1-hop lookup into multi-hop composition (`A -> B -> C`).
   - **Organ 2B (`CausalOperatorEngine`):** Maps `(h_t, e_t)` onto `M = 192` 2D phase rings under `SplitComplex(rot_pair)`. Operand latch `xi_t` holds the left-hand symbol across delimiters; when outcome `e_t` arrives, the complex quotient `r_t = zeta_out(e_t) * xi_{t-1}.conj()` accumulates the token-invariant shift/operator into slow state `Omega_t`, and `om_unit * zeta_q` applies it to unseen inputs or retrieved facts (`R x K`).
   - **Anti-Laziness Init:** `mlp_scale_init = 0.4` prevents static SwiGLU blocks from shortcutting trained rule values before the dynamic organs form.

---

## 2. Pre-Registration: `IMB17-R1` (3-Seed Equal-Param Head-to-Head on RTX 4090)

Pre-registered **before any `IMB-17` training run** (2026-11-19 17:30 IST).

- **Arms (all matched within `< 0.11%` of `31,078,928` trainable params, `dim=384, L=4, H=6x64`):**
  1. `pow_v17`: Full Two-Organ Mind (`DualPhase S^1 + 3-Bounce + CausalOperator + Echo`, O(1) state `208.5 KB`).
  2. `pam_v2`: Current `v13_sempty` PAM-v2 control (`Chrono` 1D time + 1-pass delta + `Echo`, O(1) state `198.1 KB`).
  3. `gdn_like`: Phase-free Gated-Delta + `short_conv(k=4)` + `Echo` (`use_rope=False`, O(1) state `211.5 KB`).
  4. `tpp`: `Transformer++` (`RoPE + RMSNorm + SwiGLU`, `O(T)` growing KV cache).
- **Training:** `seeds = 42, 43, 44`; `3,000` steps × `batch = 16` × `L in {512, 1024}` (`~37.7M` tokens/run, byte-identical batches across all arms per seed), cosine LR `8e-4 -> 1.2e-4`, dense real DCLM-edu prose auxiliary loss (`aux_lm_weight = 1.0`).
- **Pre-Registered Verdict Rules (3-seed means, evaluated at `2x` and `4x` beyond max train length `1024`):**
  - **Bar E1 (Overall Intelligence Beyond Recall — PRIMARY):** `pow_v17` 3-seed mean `intelligence_headline` (`mean` of `prose_lure@2048`, `scope_4axis_mean@2048`, `two_hop_new@2048`, `unseen_rules@2048`, `rxk_new@2048`) beats **every** baseline (`pam_v2`, `gdn_like`, `tpp`) by **`>= +0.08`**, and wins on **`3 / 3` seeds**.
  - **Bar E2 (True Separation from Passive Delta Models):** `pow_v17` beats both `pam_v2` and `gdn_like` by **`>= +0.10`** on at least **two** compositional/binding axes (`prose_lure@2048`, `two_hop_new@2048`, or `rxk_new@2048`).
  - **Bar E3 (Beyond-Window O(1) Advantage over Attention):** At `4x` max train length (`@4096`), `pow_v17` beats `tpp` by **`>= +0.15`** on `prose_lure@4096` and `extrap_recall@4096`.
  - **Guard G1 (Language Modeling Quality):** `pow_v17` held-out real DCLM `web_nll` within **`<= +0.03` nats** of `pam_v2`.
  - **Follow-up Ablation Rule:** If `pow_v17` passes E1, run the two pre-built surgical ablations (`pow_sync_only` vs `pow_work_only`, already param-matched in `v17/model.py`) to attribute exact gains to Organ 1 (`S^1` Synchrony) vs Organ 2 (`3-Bounce + Causal Operator`).

### 2.1 `IMB17-R1` Seed-42 Canary Verdict (`bedcf470`, `logs/v17_imb17_r1_bedcf470_20261119_1752.json`)
Stopped after all 4 arms completed Seed 42 because the `tpp` (full softmax attention) calibration canary revealed two benchmark weighting bugs, while the one formed cell (`unseen_rules@0`) showed a massive margin lead for `pow_v17`:
- **`unseen_rules@0` (held-out B2 rule parameters `{5, 9}`, `{7, 13, 17}` never seen in training):**
  - **`pow_v17`: `0.9000` acc, `+7.855 nats` margin** (train loss `1.3105`)
  - `pam_v2`: `0.8917` acc, `+4.748 nats` margin (train loss `1.3469`)
  - `gdn_like`: `0.7083` acc, `+3.029 nats` margin (train loss `1.3967`)
  - `tpp`: `0.5750` acc, `+0.431 nats` margin (train loss `1.7979`)
- **Four root causes diagnosed from `tpp`'s `@512` canary (`prose_lure@512 = 0.117`, `two_hop@512 = 0.117`, `web_nll = 10.29`):**
  1. Packed `B2` episodes had ~25 answer tokens (`mask = 25.0`) vs `1.0` for `two_hop` / `rxk` / `spread`, hijacking ~80% of every batch's gradient.
  2. Disjoint zero-occurrence `eval_*` token IDs had untrained `N(0, 0.02)` embeddings at 38M tokens (unlike `v14/microbench.py`, which shares vocabulary words and redraws fresh random bindings per document).
  3. Missing `size = 1, 2` bootstrap rungs on `prose_lure` / `two_hop` / `rxk` (`v14` B0 run-1 lesson).
  4. In `CausalOperatorEngine`, `op_gates` erase bias `-3.5` (`e_op = 0.03` per filler token -> `0.97^{256} = 5e-4`, 23-token half-life!) evaporated `Omega_t` across spread filler, and unbounded `silu(+0.5)` on `op_read_gate` leaked rule logits onto plain prose (`web_nll 17.37`).

---

## 3. Pre-Registration: `IMB17-R2` (Calibrated 3-Seed Head-to-Head)

Fixes applied before any `IMB17-R2` step:
1. `collate_docs`: equal per-document answer weight (`1.0 / n_ans` per query) + `0.6` real DCLM prose weight per doc.
2. Shared word pools (`entities`, `hop_words`, `keys`, `fact_vals`) with strictly fresh random bindings per eval doc (`seed >= 9e8`), while keeping `B2_NEW_TASKS` rule parameters and `SCOPE_HOLDOUT_PAIRS` domain pairs strictly unseen in training.
3. Bootstrap curriculum sizes `1..4` in training; eval tests hard `3`-entity lure, `4`-chain (`8`-hop) `two_hop`, `4`-register `rxk_new`, and `16`-fact `@4096` recall.
4. `CausalOperatorEngine`: `op_gates` biases `(-7.0, -3.5, -6.0)` so `Omega_t` retains `>600`-token default half-life across prose filler, and bounded `sigmoid(op_read_gate - 2.0)` so plain prose `web_nll` is protected.
5. Same pre-registered bars **E1, E2, E3, G1** across `seeds = 42, 43, 44`, `4,000` steps × `batch = 16`.

