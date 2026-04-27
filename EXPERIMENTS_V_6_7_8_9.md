# Experiments rollup: V6–V9

Single-page distillation of what we tried across complex LM iterations: **outcomes**, **comparable WikiText-103 validation PPL** at early and late training, and pointers to long-form lab notebooks. *Last updated: 2026-04-27.*

**Regenerate val PPL columns:** `uv run python scripts/extract_val_ppl_epochs.py` (TSV) or add `--json`. Tables below were built from that output plus a few hand-annotated rows where logs never reached a validation line.

---

## North-star baselines (WikiText-103, ~100M class)

| Model | Val PPL (best / typical @10 ep) | Notes |
|---|---:|---|
| Transformer B=3 | **27.08** | [EXPERIMENTS_V6_PART2](EXPERIMENTS_V6_PART2.md) — GPT-2–style, Flash-class SDPA |
| Transformer B=6 | **23.13** | Same arch, larger batch; large PPL gain vs B=3 |
| V6 `medium-pam-v3` | **29.95** | Interleaved CGU+PAM, RoPE, no QK-norm in headline config |
| V7 Exp7a (ModSwish) | **29.73** | Best flat PAM stack in V7; beats archived V6 v3 slightly |

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
- **V7:** **ModSwish (Exp7a)** is the best **flat** 16-layer PAM result (**29.73** @10). **Chunked dual form + B=6 (7d)** trades some PPL for a large **throughput** win vs B=3. The **“lean PAM”** (no CGU) run proves **channel gating is worth ~2 PPL** but saves VRAM and raises tok/s. **`medium_h16_flat` 383e514 (ModReLU) 9-epoch** run reached **26.64** val PPL (non-Swish preset — not apples-to-apples to 7a).
- **V8:** **QLC Stage A.5 on TinyStories** shows QLC can beat passthrough in a smoke setting; **WikiText e2e** runs moved PPL in the right direction with v8.2+ fixes (see `e2e_*` logs), with **context-length hack run** ending **38.66** @3 ep. Full **V8.3** story (interleaving, diagnostics) is in [v8/EXPERIMENTS_V8.md](v8/EXPERIMENTS_V8.md).
- **V9:** **PAM output gate** with inherited config reaches **29.57** @10 (**better than V7 7a and V6 v3** on PPL) but the logged run is **not a clean ablation** — `use_reverse_assoc=True` and **+5M params** vs 7a; see [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md). **Clean 100.5M gate** run (`gate_100m`) is **in progress** at the time of this file (epoch 1 val **55.37** in log).

## Crux: what did not work or was stopped

- **V6:** **PAM QK-norm = on** in v3: **repetition / mode collapse** while loss still looked fine — run stopped; headline preset keeps **QK-norm off** ([EXPERIMENTS_V6_PART2](EXPERIMENTS_V6_PART2.md) “Bug 8”). **Memory-reframe** pathologies: `run2_span_corruption` blows up; others finish but do not beat mainline PAM. **Diffusion-text** and **rebalanced_110656** logs never reached a first val epoch line (crashed or stalled). **Small-matched 143537** (seq-512) log has no completed val epoch in file.
- **V7:** **7f-1 multi-scale** and **7f-2 reverse-assoc** and **7f-0/7f grouped+multi-scale** all **regress** vs 7a. **7f-3 grouped-only** was **halted after epoch 1** (poor signal vs 7a). **7b `phase_mod`** was **stopped mid-epoch-1** to free GPU — **no val** in the saved log. **7f-3** on-disk log in this repo has **no val lines**; the learnings table records **epoch-1 val 58.70** when the halt decision was made — we preserve that number below.
- **V8:** **4450ff0 / early e2e** was effectively **killed** (align/gamma on noise floor; see V8 doc). **Stage A medium** run **hung** after printing epoch header (no val). **E2E logs starting at ep2** miss ep1 in the file (resume/rotated logs) — PPL@1 is blank in the table.
- **V9:** **Gate + reverse_assoc + 105M** is **not** a pure test of the output gate; follow **clean 100.5M** + matrix in V9 doc.

## Open

- V9 **C/R/N/K** matrix (`gate_revassoc_100m`, `gate_qknorm_100m`, `gate_conv4_100m`, etc.) and whether **any** variant closes more of the **27.08 (transformer B=3)** gap without capacity confounds.
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
| `logs/v6/wikitext103_diffusion_text_20260311_145638_67a0782/v6_diffusion-text_small-matched.log` | — | — | — | — | no val epoch line in file |
| `logs/v6/wikitext103_small_matched_20260310_143537_72ce6e5_dirty/v6_autoregressive_small-matched.log` | — | — | — | — | no val epoch line in file (stalled mid-ep1) |
| `logs/v6/wikitext103_small_matched_20260310_152631_6ffe838_dirty/v6_autoregressive_small-matched.log` | 121.94 | 82.86 | 49.61 | 20 |  |
| `logs/v6/wikitext103_small_matched_20260311_105203_dacac03/v6_autoregressive_small-matched.log` | 117.56 | 76.92 | 50.79 | 4 |  |

### V7 — all `logs/v7/` runs

| Log | val@1 | val@2 | val@last | last ep | Notes |
|---|---:|---:|---:|---:|---|
| `logs/v7/exp7a_swish_wikitext103_20260328_081707_81e8ea9_dirty/v7_medium_h16_flat_wikitext103.log` | 56.15 | 43.20 | 29.73 | 10 | best flat V7 PPL |
| `logs/v7/exp7b_phase_mod_wikitext103_20260330_115105_920b365/v7_medium_h16_flat_wikitext103.log` | — | — | — | — | no val line; **7b stopped** mid-epoch-1 (see [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md)) |
| `logs/v7/exp7d_chunked_b6_wikitext103_20260330_172839_7555c93/v7_medium_h16_flat_wikitext103.log` | 57.42 | 41.16 | 27.94 | 10 | B=6, chunked; strong PPL, higher tok/s |
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
| `logs/v9/pam_gate_100m_wikitext103_20260427_082228_7610259/v9_medium_h16_gate_100m_wikitext103.log` | 55.37 | — | 55.37 | 1 | clean gate ~100.5M; **stopped** after ep1 when this file was written |
| `logs/v9/pam_gate_wikitext103_20260426_195110_80b725b_dirty/v9_medium_h16_gate_wikitext103.log` | 54.29 | 42.06 | 29.57 | 10 | **confound:** `pam_output_gate` + **`use_reverse_assoc=True`**; 105.1M params |

---

## See also (full detail, not replaced by this page)

- [EXPERIMENTS_V6.md](EXPERIMENTS_V6.md), [EXPERIMENTS_V6_PART2.md](EXPERIMENTS_V6_PART2.md), [EXPERIMENTS_V6_MEMORY_REFRAME.md](EXPERIMENTS_V6_MEMORY_REFRAME.md), [EXPERIMENTS_V6_DIFFUSION.md](EXPERIMENTS_V6_DIFFUSION.md)
- [v7/EXPERIMENTS_V7.md](v7/EXPERIMENTS_V7.md)
- [v8/EXPERIMENTS_V8.md](v8/EXPERIMENTS_V8.md), [v8/AUDIT_V8.md](v8/AUDIT_V8.md)
- [v9/EXPERIMENTS_V9.md](v9/EXPERIMENTS_V9.md)
