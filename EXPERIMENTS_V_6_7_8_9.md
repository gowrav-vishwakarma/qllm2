# Experiments rollup: V6–V9

Single-page distillation of what we tried across complex LM iterations: **outcomes**, **comparable WikiText-103 validation PPL** at early and late training, and pointers to long-form lab notebooks. *Last updated: 2026-05-12.*

**Regenerate val PPL columns:** `uv run python scripts/extract_val_ppl_epochs.py` (TSV) or add `--json`. Tables below were built from that output plus a few hand-annotated rows where logs never reached a validation line.

---

## Current best PAM run (cross-version, as of 2026-05-12)

> **Best logged flat V7 val PPL (WikiText-103, ~100M):** **26.88** @10 ep — **7d** recipe (chunked `C=256`, ModSwish, `medium_h16_flat`, **B=18**, `--compile`), commit `fad662a` (dirty), May 2026. **Below** transformer **B=3** (**27.08**) and **B=6** (**23.13**); still **above** transformer **B=18** (**22.69**, May 2026 — same batch / steps-per-epoch as this PAM run). Log: `logs/v7/exp7d_chunked_b6_wikitext103_20260509_064854_fad662a_dirty/v7_medium_h16_flat_wikitext103.log` — directory name uses the `exp7d_chunked_b6` script slug; **actual batch size was 18**. Fewer optimizer steps per epoch than B=3 runs (3213 vs 19277) — not directly comparable to **7a** on schedule shape.

> **V9 `gate` (confounded) — val PPL 29.57**, 105.1M params, WikiText-103, B=3, 10 ep.
> `pam_output_gate=True` (extra `nn.Linear(2*dim → inner_dim) + 2*sigmoid` on the PAM readout)
> with inherited `use_reverse_assoc=True`. **Confounded** — not a clean ablation,
> +5M params over V7 baseline, generation still loops. Acceptance gate `<29.2` not met.
> Full config + caveats: [v9/EXPERIMENTS_V9.md → Current Best PAM Run](v9/EXPERIMENTS_V9.md#current-best-pam-run-as-of-2026-04-27).
> Prior “number to beat” for readout tweaks; raw val PPL is **worse** than the May 2026 flat V7 **7d B=18** run above.

## North-star baselines (WikiText-103, ~100M class)

| Model | Val PPL (best / typical @10 ep) | Notes |
|---|---:|---|
| Transformer B=3 | **27.08** | [EXPERIMENTS_V6_PART2](EXPERIMENTS_V6_PART2.md) — GPT-2–style, Flash-class SDPA |
| Transformer B=6 | **23.13** | Same arch, `batch_size=6`; 9,639 steps/epoch |
| **Transformer B=18** (May 2026) | **22.69** | Same pipeline; `387b2a5`; **3,213** steps/epoch (apples-to-apples batch vs V7 7d B=18). Log: `logs/v6/transformer_baseline_wikitext103_20260512_063754_387b2a5/transformer_baseline.log`. |
| **V7 Exp7d chunked B=18** (May 2026) | **26.88** | **Beats B=3/B=6** on val/step budget; still **+4.19** vs transformer **B=18** (**22.69**). [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md); `fad662a` dirty; log dir slug `exp7d_chunked_b6_*` — **B=18** in trainer header. |
| **V7 Exp7d chunked B=18 ModReLU** (May 2026) | **27.46** | Same recipe except **`activation=modrelu`**; commit **`ac02323`** (clean). **+0.58** vs ModSwish **26.88** above. Log: `logs/v7/exp7d_chunked_b6_wikitext103_20260512_122020_ac02323/`. |
| V7 Exp7d chunked B=6 (Mar 2026) | **27.94** | `7555c93`; higher tok/s (~31.8k) than B=18 rerun (~22k). |
| **V9 `gate` (confounded)** | **29.57** | 105.1M params; `pam_output_gate=True` + inherited `use_reverse_assoc=True`. See [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md#current-best-pam-run-as-of-2026-04-27). |
| V7 Exp7a (ModSwish, B=3) | **29.73** | Best **B=3** flat stack in V7; beats archived V6 v3 slightly. |
| V6 `medium-pam-v3` | **29.95** | Interleaved CGU+PAM, RoPE, no QK-norm in headline config |
| V5 `medium-v5-100m` (May 2026) | **41.88** | Algebraic V5 stack on WT103 — **not** competitive vs rows above; [v5/EXPERIMENTS.md](v5/EXPERIMENTS.md#11-wikitext-103-medium-v5-100m-2026-05-09). |

PAM/QLLM trains slower in tok/s than the transformer for stack maturity reasons; treat PPL as the clean quality readout, throughput as its own line item in the per-version docs.

---

## Lineage (where each version focused)

```mermaid
flowchart LR
  v6[V6_PAM_and_transformer_baseline]
  v7[V7_flat_PAM_CGU_stack]
  v8[V8_QLC_on_V7_backbone]
  v9[V9_PAM_readout_tweaks]
  v6 --> v7 --> v8 --> v9
```

---

## Crux: what worked

- **V6:** **Interleaved CGU + PAM** and **PAM v3** (RoPE, fused QKV, block-real path) land near a strong transformer on PAM at matched batch — headline **29.95** val PPL. **GSP + TSO + rebalanced** settings matter for stability. **HSB** was an interesting SSM-native binding story but did not win the main medium run vs simpler stacks (see HSB + medium logs and [EXPERIMENTS_V6_PART2](EXPERIMENTS_V6_PART2.md)). **Scaling-sweep** logs show **QPAM/RPAM** PPL improving with param budget on the same schedule (tab-separated `mid/end` val snapshots).
- **V7:** **ModSwish (Exp7a)** is the best **flat** 16-layer PAM result at **matched B=3** (**29.73** @10). **May 2026:** **7d**-style chunked dual form at **B=18** reaches **26.88** val PPL @10 — **below transformer B=3 (27.08)** and **B=6 (23.13)** on headline tables, but **above** the **transformer B=18** anchor (**22.69**, same steps/epoch); log `logs/v7/exp7d_chunked_b6_wikitext103_20260509_064854_fad662a_dirty/` (script-issued dirname; **batch 18** in the log). Transformer B=18 log: `logs/v6/transformer_baseline_wikitext103_20260512_063754_387b2a5/`. **Same 7d B=18 geometry with ModReLU** (`ac02323`, `use_reverse_assoc=True`) finished **27.46** @10 — **+0.58 vs ModSwish** on val PPL; log `logs/v7/exp7d_chunked_b6_wikitext103_20260512_122020_ac02323/`. **Chunked dual form + B=6 (7d, Mar 2026)** gave **27.94** with higher tok/s (~31.8k vs ~22k). The **“lean PAM”** (no CGU) run proves **channel gating is worth ~2 PPL** but saves VRAM and raises tok/s. **`medium_h16_flat` 383e514 (ModReLU) 9-epoch** run reached **26.64** val PPL (non-Swish preset — not apples-to-apples to 7a).
- **V8:** **QLC Stage A.5 on TinyStories** shows QLC can beat passthrough in a smoke setting; **WikiText e2e** runs moved PPL in the right direction with v8.2+ fixes (see `e2e_*` logs), with **context-length hack run** ending **38.66** @3 ep. Full **V8.3** story (interleaving, diagnostics) is in [v8/EXPERIMENTS_V8.md](v8/EXPERIMENTS_V8.md).
- **V9:** **PAM output gate** with inherited config reaches **29.57** @10 (**better than V7 7a and V6 v3** on PPL) but the logged run is **not a clean ablation** — `use_reverse_assoc=True` and **+5M params** vs 7a; see [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md). Clean follow-ups did **not** reproduce the win: `gate_100m` was weak at ep1, `compete_revassoc_100m` was weak by ep2, `gate_mlp_revassoc_100m` trailed baselines through ep5, the completed parameter-matched `gate_revassoc_100m` finished at **30.53**, and **`gate_conv4_100m`** (3ep smoke + **10ep** full) finished at **30.02** best val — still **not** competitive with **7a** or **7d B=18**. See [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md). The only remaining V9-class readout tweak **not** on disk is **`gate_qknorm_100m`** (high risk: Bug 8); pivot is **memory dynamics** (e.g. per-channel decay).

## Crux: what did not work or was stopped

- **V6:** **PAM QK-norm = on** in v3: **repetition / mode collapse** while loss still looked fine — run stopped; headline preset keeps **QK-norm off** ([EXPERIMENTS_V6_PART2](EXPERIMENTS_V6_PART2.md) “Bug 8”). **Memory-reframe** pathologies: `run2_span_corruption` blows up; others finish but do not beat mainline PAM. **Diffusion-text** and **rebalanced_110656** logs never reached a first val epoch line (crashed or stalled). **Small-matched 143537** (seq-512) log has no completed val epoch in file.
- **V5:** **WikiText-103 `medium-v5-100m` (May 2026)** finished at val **41.88** @10 — far from PAM/V7/transformer class; see [v5/EXPERIMENTS.md](v5/EXPERIMENTS.md#11-wikitext-103-medium-v5-100m-2026-05-09).
- **V7:** **7f-1 multi-scale** and **7f-2 reverse-assoc** and **7f-0/7f grouped+multi-scale** all **regress** vs 7a. **7f-3 grouped-only** was **halted after epoch 1** (poor signal vs 7a). **7b `phase_mod`** was **stopped mid-epoch-1** to free GPU — **no val** in the saved log. **7f-3** on-disk log in this repo has **no val lines**; the learnings table records **epoch-1 val 58.70** when the halt decision was made — we preserve that number below. **7pos (May 2026):** input learned pos + RoPE **neutral** (**26.92**, +0.04 vs **26.88**); pos-only without RoPE **26.72** (Tier A −0.20 vs hybrid; single-run) — **flat bar stays 26.88**; details → [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md) (Experiment 7pos).
- **V8:** **4450ff0 / early e2e** was effectively **killed** (align/gamma on noise floor; see V8 doc). **Stage A medium** run **hung** after printing epoch header (no val). **E2E logs starting at ep2** miss ep1 in the file (resume/rotated logs) — PPL@1 is blank in the table.
- **V9:** **Gate + reverse_assoc + 105M** is **not** a pure test of the output gate. The parameter-matched `gate_revassoc_100m` finished at **30.53**, worse than V9 gate **29.57**, flat V7 **7d B=18** (**26.88**), V7 **7a** (**29.73**), and V6 **29.95**. Zero-param competition and 2-layer gate-MLP also failed to beat the baseline trajectory. **`gate_conv4_100m`** completed (smoke + 10ep) with best val **30.02** — still trailing **7a**. **`gate_qknorm_100m`** has **no** `logs/v9/` run (optional; QK-norm risk). Lesson: small PAM readout/local conv tweaks are not closing the gap at ~100M. See [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md#2026-04-28-v9-gate--reverse-assoc-parameter-matched-final).

## Open

- **V7 (May 2026):** New flat bar is **26.88** @10 (**7d B=18 ModSwish**). **ModReLU @ same 7d B=18** (`ac02323`): **27.46** @10 — **ModSwish wins by ~0.58 PPL** (matched `chunk_size=256`, `use_reverse_assoc=True`). **Matched-batch transformer ceiling:** **22.69** @10 (**transformer B=18**, `387b2a5`). Sensible follow-ups: **7e** unitary reg on the same recipe; **B=18** LR/warmup tuned for **3213** steps/epoch (beyond the single 7d schedule); **ModReLU vs ModSwish @ B=6 chunked** (3a-A vs 7d B=6) still confounded; close gap to **22.69**; log **generation / rep-gram** beside val PPL when claiming parity.
- **V5 (May 2026):** WT103 **41.88** @100M is a **negative result** for the current V5 stack — either **targeted debugging** (data, loss, capacity vs V6) or **explicit deprioritization** on WT103 in favor of V7/V9.
- **V9 readout line:** **`gate_conv4_100m`** is **complete** on disk (3ep + 10ep; best val **30.02**) — **do not re-run**. Optional untested variant: **`gate_qknorm_100m`** (no log; **Bug 8** risk). Next lever: **PAM memory dynamics** (e.g. per-channel decay) per [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md).
- V8 path to PPL parity with **V7 QPAM passthrough** and whether **QLC** can show sustained **γ > 0** and **mean iter > 1** *without* killing PPL (see [v8/AUDIT_V8.md](v8/AUDIT_V8.md)).

---

## All runs: val PPL @ epoch 1, 2, and last completed epoch

`val@last` is validation PPL at the **highest completed epoch** found in the log (not necessarily epoch 10). Em dash (—) means the epoch is missing from the file (fragment, crash, or run stopped early). **Scaling-sweep** files use per-epoch `end` PPLs from the tab format, not the standard trainer line.

### V6 — all `logs/v6/` runs

| Log | val@1 | val@2 | val@last | last ep | Notes |
|---|---:|---:|---:|---:|---|
| `logs/v6/composite_keys_wikitext103_20260313_121107_98c2585_dirty/run2_composite_keys_episodic16/v6_autoregressive_small-matched.log` | 135.39 | 89.43 | 58.40 | 10 |  |
| `logs/v6/fulldata_no_memory_20260308_195054_f960859/v6_train_small-matched.log` | 7.36 | — | 7.36 | 1 |  |
| `logs/v6/fulldata_tiny_memory_20260309_021456_f960859/v6_train_small-matched.log` | 2.23 | — | 2.23 | 1 | non-WT103 / tiny or smoke |
| `logs/v6/fulldata_tiny_model_20260309_084956_f960859/v6_train_tiny.log` | 4.64 | — | 4.64 | 1 | non-WT103 / tiny or smoke |
| `logs/v6/long_seq_2048_wikitext103_20260312_202857_56d27eb/v6_autoregressive_small-matched.log` | 135.80 | 94.62 | 75.53 | 5 |  |
| `logs/v6/mac_cpu_ablation_baseline.log` | 78.15 | 57.42 | 54.52 | 3 | non-WT103 / tiny or smoke |
| `logs/v6/mac_cpu_ablation_no_im.log` | 77.47 | 56.48 | 52.43 | 3 | non-WT103 / tiny or smoke |
| `logs/v6/mac_cpu_ablation_no_wm.log` | 81.07 | 61.28 | 57.89 | 3 | non-WT103 / tiny or smoke |
| `logs/v6/mac_cpu_tiny_smoke.log` | 77.28 | 54.56 | 42.18 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/medium_gsp_tso_wikitext103_20260315_091445_53ca2c6/v6_autoregressive_medium-rebalanced-gsp.log` | 104.67 | 70.47 | 41.67 | 10 |  |
| `logs/v6/medium_hsb_tso_gsp_wikitext103_20260317_123509_42f5f73/v6_autoregressive_medium-rebalanced-hsb.log` | — | — | 43.54 | 10 | log fragment: first epoch in file is 4 (ep1–ep3 missing) |
| `logs/v6/medium_pam_gsp_wikitext103_20260318_112906_e1c1ef0/v6_autoregressive_medium-pam.log` | 109.30 | 67.04 | 53.49 | 3 |  |
| `logs/v6/medium_pam_gsp_wikitext103_20260318_162526_5da262e_dirty/v6_autoregressive_medium-pam.log` | — | — | 38.95 | 10 | log fragment: first epoch in file is 4 (ep1–ep3 missing) |
| `logs/v6/medium_pam_v2_interleaved_wikitext103_20260319_143546_b3f11c0_dirty/v6_autoregressive_medium-pam-v2.log` | 57.84 | — | 57.84 | 1 |  |
| `logs/v6/medium_pam_v3_pia_wikitext103_20260321_201508_5c76a92/v6_autoregressive_medium-pam-v3-pia.log` | 56.74 | 43.48 | 30.01 | 10 |  |
| `logs/v6/medium_pam_v3_qknorm_rope_wikitext103_20260319_161045_77c454a/v6_autoregressive_medium-pam-v3.log` | 63.42 | 48.85 | 40.50 | 4 | stopped mid-training (QK-norm on — Bug 8 track) |
| `logs/v6/medium_pam_v3_rope_wikitext103_20260319_231524_31397f0/v6_autoregressive_medium-pam-v3.log` | 57.94 | 43.83 | 29.95 | 10 | headline v3 / RoPE / no QK-norm |
| `logs/v6/medium_rebalanced_tso_wikitext103_20260314_173445_cc4a491/v6_autoregressive_medium-rebalanced.log` | 107.71 | 73.04 | 44.47 | 10 |  |
| `logs/v6/memory_reframe_wikitext103_20260311_180317_c8faf67_dirty/run1_control_baseline/v6_autoregressive_small-matched.log` | 123.29 | — | 123.29 | 1 | 1 epoch only in file |
| `logs/v6/memory_reframe_wikitext103_20260311_185409_c8faf67_dirty/run2_span_corruption/v6_autoregressive_small-matched.log` | 1180.02 | 954.21 | 954.21 | 2 | broken run (high PPL) |
| `logs/v6/memory_reframe_wikitext103_20260311_203225_2238a2b/run4_bank_role_0.0/v6_autoregressive_small-matched.log` | 121.97 | 84.37 | 56.46 | 10 |  |
| `logs/v6/memory_reframe_wikitext103_20260312_082554_868944d/run5_delayed_recall_episodic16/v6_autoregressive_small-matched.log` | 125.64 | 85.92 | 56.55 | 10 |  |
| `logs/v6/rebalanced_tso_wikitext103_20260314_110656_e96032d/v6_autoregressive_small-rebalanced.log` | — | — | — | — | no val epoch line in file (crashed / stalled) |
| `logs/v6/rebalanced_tso_wikitext103_20260314_110835_e96032d/v6_autoregressive_small-rebalanced.log` | 112.52 | 79.51 | 52.64 | 10 |  |
| `logs/v6/rtx4090_ablation_tiny_A_baseline.log` | 14.37 | 10.51 | 8.84 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/rtx4090_ablation_tiny_B_no_memory.log` | 17.61 | 13.82 | 11.83 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/rtx4090_ablation_tiny_C_no_wm.log` | 17.61 | 13.78 | 11.77 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/rtx4090_ablation_tiny_D_no_im.log` | 14.99 | 10.57 | 8.88 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/rtx4090_ablation_tiny_E_with_attn.log` | 14.16 | 10.24 | 8.61 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/rtx4090_ablation_tiny_runner.log` | 14.16 | 10.24 | 8.61 | 5 | non-WT103 / tiny or smoke |
| `logs/v6/rtx4090_small_matched_original.log` | 11.37 | 4.21 | 1.23 | 7 |  |
| `logs/v6/rtx4090_small_matched_v2_fixes.log` | 21.23 | 12.31 | 12.31 | 2 |  |
| `logs/v6/scaling_sweep/qpam_10m_val_ppl.log` | 381.44 | 266.44 | 123.47 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/qpam_25m_val_ppl.log` | 212.00 | 135.09 | 74.92 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/qpam_50m_val_ppl.log` | 145.35 | 92.09 | 45.70 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/qpam_5m_val_ppl.log` | 574.60 | 404.36 | 258.12 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/rpam_10m_val_ppl.log` | 175.62 | 115.50 | 69.57 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/rpam_25m_val_ppl.log` | 94.79 | 63.17 | 40.68 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/rpam_50m_val_ppl.log` | 72.55 | 52.24 | 35.70 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/scaling_sweep/rpam_5m_val_ppl.log` | 280.84 | 193.70 | 118.55 | 10 | per-epoch `end` PPL; tab format |
| `logs/v6/small_matched_full_20260310_000217_518c76e/v6_autoregressive_small-matched.log` | 6.63 | 6.01 | 5.50 | 5 |  |
| `logs/v6/small_matched_no_memory_10_epoch/v6_train_small-matched.log` | 6.75 | — | 6.75 | 1 |  |
| `logs/v6/tiny_wm2_im4/v6_train_tiny.log` | 8.48 | — | 8.48 | 1 | non-WT103 / tiny or smoke |
| `logs/v6/transformer_baseline_wikitext103_20260323_140351_e87b8e4/transformer_baseline.log` | 53.74 | 39.42 | 27.08 | 10 | B=3 transformer |
| `logs/v6/transformer_baseline_wikitext103_20260330_130306_8d631a6/transformer_baseline.log` | 52.83 | 35.67 | 23.13 | 10 | B=6 transformer |
| `logs/v6/transformer_baseline_wikitext103_20260512_063754_387b2a5/transformer_baseline.log` | 73.41 | 38.77 | 22.69 | 10 | B=18 transformer; ~206k tok/s; `checkpoints_transformer_baseline_b18_wt103` |
| `logs/v6/wikitext103_diffusion_text_20260311_145638_67a0782/v6_diffusion-text_small-matched.log` | — | — | — | — | no val epoch line in file |
| `logs/v6/wikitext103_small_matched_20260310_143537_72ce6e5_dirty/v6_autoregressive_small-matched.log` | — | — | — | — | no val epoch line in file (stalled mid-ep1) |
| `logs/v6/wikitext103_small_matched_20260310_152631_6ffe838_dirty/v6_autoregressive_small-matched.log` | 121.94 | 82.86 | 49.61 | 20 |  |
| `logs/v6/wikitext103_small_matched_20260311_105203_dacac03/v6_autoregressive_small-matched.log` | 117.56 | 76.92 | 50.79 | 4 |  |

### V7 — all `logs/v7/` runs

| Log | val@1 | val@2 | val@last | last ep | Notes |
|---|---:|---:|---:|---:|---|
| `logs/v7/exp7a_swish_wikitext103_20260328_081707_81e8ea9_dirty/v7_medium_h16_flat_wikitext103.log` | 56.15 | 43.20 | 29.73 | 10 | best flat V7 PPL at **B=3** |
| `logs/v7/exp7b_phase_mod_wikitext103_20260330_115105_920b365/v7_medium_h16_flat_wikitext103.log` | — | — | — | — | no val line; **7b stopped** mid-epoch-1 (see [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md)) |
| `logs/v7/exp7d_chunked_b6_wikitext103_20260330_172839_7555c93/v7_medium_h16_flat_wikitext103.log` | 57.42 | 41.16 | 27.94 | 10 | B=6, chunked; strong PPL, higher tok/s |
| `logs/v7/exp7d_chunked_b6_wikitext103_20260509_064854_fad662a_dirty/v7_medium_h16_flat_wikitext103.log` | 84.05 | 47.80 | 26.88 | 10 | **B=18** chunked 7d redo; **best flat V7 val in table**; dirname slug `b6` is script default — see [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md) |
| `logs/v7/exp7d_chunked_b6_wikitext103_20260512_122020_ac02323/v7_medium_h16_flat_wikitext103.log` | 90.18 | 50.22 | 27.46 | 10 | **B=18**, ModReLU, chunked 7d; **`ac02323`**; **+0.58** val PPL vs ModSwish B=18 (**26.88**); `use_reverse_assoc=True` |
| `logs/v7/exp7pos_hybrid_wikitext103_20260520_055039_124e34e_dirty/v7_medium_h16_flat_wikitext103.log` | 88.35 | 48.76 | 26.92 | 10 | **B=18**, learned pos + RoPE; **+0.04** vs 7d (**26.88**); bar unchanged — [v7 §7pos](v7/EXPERIMENTS_V7.md) |
| `logs/v7/exp7pos_only_wikitext103_20260520_131624_124e34e_dirty/v7_medium_h16_flat_wikitext103.log` | 89.39 | 49.18 | 26.72 | 10 | **B=18**, learned pos, `--no_rope`; Tier B **−0.16** vs 7d; **do not** promote as headline — [v7 §7pos](v7/EXPERIMENTS_V7.md) |
| `logs/v7/exp7f1_multiscale_flat_wikitext103_20260403_124050_ecb4b56/v7_medium_h16_flat_wikitext103.log` | 58.40 | 44.25 | 30.55 | 10 | multi-scale loss hurts vs 7a |
| `logs/v7/exp7f2_reverse_assoc_wikitext103_20260417_124743_7f52bcf/v7_medium_h16_flat_wikitext103.log` | 57.56 | 45.61 | 32.19 | 10 | reverse-assoc hurts vs 7a |
| `logs/v7/exp7f3_grouped_only_wikitext103_20260418_120504_5294937_dirty/v7_medium_h16_grouped_wikitext103.log` | 58.70 | — | 58.70 | 1 | **from learnings log**; on-disk file has no val; halted after ep1 per V7 doc |
| `logs/v7/exp7f_multiscale_wikitext103_20260402_173535_136f914_dirty/v7_medium_h16_grouped_wikitext103.log` | 60.46 | 45.85 | 31.29 | 10 | grouped+multi-scale confounded; worse than 7a |
| `logs/v7/lean_lean_medium_small_l2_unitary_wikitext103_20260410_114015_747a04a/lean_lean_medium_small_wikitext103.log` | 59.86 | 45.09 | 35.13 | 5 |  |
| `logs/v7/lean_lean_medium_small_wikitext103_20260409_152722_7717f65_dirty/lean_lean_medium_small_wikitext103.log` | 59.97 | 45.38 | 32.09 | 10 | lean (no CGU) |
| `logs/v7/medium_h16_flat_wikitext103_20260326_182543_383e514_dirty/v7_medium_h16_flat_wikitext103.log` | 58.93 | 41.01 | 26.64 | 9 | ModReLU/383e509-class sweep; 9 ep |
| `logs/v7/medium_h16_flat_wikitext103_20260327_104348_fc161ce/v7_medium_h16_flat_wikitext103.log` | 58.14 | 44.17 | 36.12 | 4 | interrupted; use recovered+tail for full curve |
| `logs/v7/medium_h16_flat_wikitext103_20260327_104348_fc161ce/v7_medium_h16_flat_wikitext103_recovered.log` | 58.14 | 44.17 | 30.89 | 8 | recovery train |
| `logs/v7/medium_h16_flat_wikitext103_20260327_104348_fc161ce/v7_medium_h16_flat_wikitext103_tail_append.log` | — | — | 30.40 | 10 | log fragment: first epoch in file is 9 (ep1–ep8 missing); finishes **30.40** @10 |

### V8 — all `logs/v8/` runs

| Log | val@1 | val@2 | val@last | last ep | Notes |
|---|---:|---:|---:|---:|---|
| `logs/v8/e2e_medium_context_reasoning_wikitext103_20260425_103755_4069dac/v8_e2e_medium_context_reasoning_wikitext103.log` | 57.58 | 43.73 | 38.66 | 3 | context hack e2e; 3 ep in file |
| `logs/v8/e2e_medium_reasoning_wikitext103_20260424_172759_ab342a1/v8_e2e_medium_reasoning_wikitext103.log` | 57.31 | — | 57.31 | 1 | 1-epoch capture |
| `logs/v8/e2e_medium_reasoning_wikitext103_20260424_232539_18f4de7_dirty/v8_e2e_medium_reasoning_wikitext103.log` | — | 43.04 | 38.21 | 3 | log fragment: ep1 missing; v8.2-style trajectory |
| `logs/v8/stageA5_passthrough_tinystories_20260423_114721_e3e4b90/v8_smoke_tiny_passthrough_tinystories.log` | 8544.20 | 329.18 | 178.95 | 3 | TinyStories smoke |
| `logs/v8/stageA5_qlc_r4_T2_tinystories_20260423_114850_e3e4b90/v8_smoke_tiny_qlc_r4_T2_tinystories.log` | 8816.38 | 323.90 | 173.40 | 3 | TinyStories smoke |
| `logs/v8/stageA_stageA_medium_wikitext103_20260423_120304_e3e4b90/v8_stageA_medium_wikitext103.log` | — | — | — | — | hung after compile; no val |

### V9 — all `logs/v9/` runs

| Log | val@1 | val@2 | val@last | last ep | Notes |
|---|---:|---:|---:|---:|---|
| `logs/v9/pam_gate_revassoc_100m_wikitext103_20260427_230817_2fe5c9a/v9_medium_h16_gate_revassoc_100m_wikitext103.log` | 55.29 | 42.96 | 30.53 | 10 | simple PAM gate + reverse_assoc, 100.5M; **negative** vs V9 gate 29.57 / V7 29.73 / V6 29.95 |
| `logs/v9/pam_gate_mlp_revassoc_100m_wikitext103_20260427_142855_133e208_dirty/v9_medium_h16_gate_mlp_revassoc_100m_wikitext103.log` | 56.83 | 43.65 | 34.11 | 5 | 2-layer PAM gate MLP + reverse_assoc, 101.1M; **stop recommended** (trails V9 gate/V7/V6 at ep5 despite clean generation) |
| `logs/v9/pam_compete_revassoc_100m_wikitext103_20260427_112939_2baf6d7/v9_medium_h16_compete_revassoc_100m_wikitext103.log` | 56.27 | 44.44 | 44.44 | 2 | zero-param cross-head competition + reverse_assoc; **stopped / weak signal** |
| `logs/v9/pam_gate_100m_wikitext103_20260427_082228_7610259/v9_medium_h16_gate_100m_wikitext103.log` | 55.37 | — | 55.37 | 1 | clean gate ~100.5M; **stopped** after ep1 when this file was written |
| `logs/v9/pam_gate_wikitext103_20260426_195110_80b725b_dirty/v9_medium_h16_gate_wikitext103.log` | 54.29 | 42.06 | 29.57 | 10 | **confound:** `pam_output_gate` + **`use_reverse_assoc=True`**; 105.1M params |

### V6 Diffusion (medium-pam-v3, WikiText-103) — Part 2 (started 2026-05-12; **paused** after Run B same day)

Diffusion Part 2 **paused** 2026-05-12 after Run B; priority returns to autoregressive (regression) work until diffusion is re-scoped. See [EXPERIMENTS_V6_DIFFUSION_PART2.md](EXPERIMENTS_V6_DIFFUSION_PART2.md) for hypotheses, sample notes, and per-run details. `div=n/a` is expected for `medium-pam-v3` (single-bank); use `--log_diff_diagnostics` for `pred/target` norm tracking.

| Log | diff_loss@~4.5k | div | val | last batch | Notes |
|---|---:|:-:|:-:|---:|---|
| `logs/v6/wikitext103_diffusion_text_medium_pam_v3_20260512_083801_d0697e1/v6_diffusion-text_medium-pam-v3.log` | 0.0125 | n/a | — | 4500/19277 | Run A pre-Phase-0 baseline; Pro 6000 host but `B=3` (~18 GB used / 96 GB), `compile=reduce-overhead`; rapid loss collapse same shape as Part 1 §11 |
| `logs/v6/wikitext103_diffusion_text_medium_pam_v3_20260512_090451_3f56d9b/v6_diffusion-text_medium-pam-v3.log` | ~0.0000 | n/a | **~0.0000** @ ep3 | 1100/3213 (ep4) | Run B: `B≈18`, Phase 0 diagnostics, `compile=reduce-overhead`; ep1–3 completed; **stopped mid-ep4**; **killed** — poor samples (Part 1 §11 class); **pivot back to regression (AR)** |

---

## See also (full detail, not replaced by this page)

- [EXPERIMENTS_V6.md](EXPERIMENTS_V6.md), [EXPERIMENTS_V6_PART2.md](EXPERIMENTS_V6_PART2.md), [EXPERIMENTS_V6_MEMORY_REFRAME.md](EXPERIMENTS_V6_MEMORY_REFRAME.md), [EXPERIMENTS_V6_DIFFUSION.md](EXPERIMENTS_V6_DIFFUSION.md), [EXPERIMENTS_V6_DIFFUSION_PART2.md](EXPERIMENTS_V6_DIFFUSION_PART2.md)
- [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md)
- [v8/EXPERIMENTS_V8.md](v8/EXPERIMENTS_V8.md), [v8/AUDIT_V8.md](v8/AUDIT_V8.md)
- [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md)
