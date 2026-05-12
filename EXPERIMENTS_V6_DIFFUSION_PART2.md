# V6 Diffusion Experiment Log — Part 2

Continuation of [EXPERIMENTS_V6_DIFFUSION.md](EXPERIMENTS_V6_DIFFUSION.md).
Part 1 covered the initial backbone extraction, Mac smokes, and the failed
March 2026 GPU run on `small-matched` (mode collapse to "rous"). Part 2 picks
up the diffusion research direction at ~100M scale on the host's new GPU.

**Branch**: `master`
**Restart date**: 2026-05-12
**Hardware**: RTX Pro 6000 (96 GB VRAM, default), RTX 4090 (24 GB, fallback)
**Roadmap**: [.cursor/plans/v6_diffusion_research_roadmap_a5cdc348.plan.md](.cursor/plans/v6_diffusion_research_roadmap_a5cdc348.plan.md)

**Status (2026-05-12):** Diffusion Part 2 is **paused** after **Run B** (below). The run was **killed** mid-training: latent train/val loss collapsed toward zero while epoch-end **samples remained unusable** (same failure class as Part 1 §11). **Primary effort shifts back to autoregressive (“regression”)** work — WikiText-103 PPL, matched transformer baselines, and V6/V7-style ablations — until diffusion objectives, sampling diagnostics, and EMA / target-parameterization experiments are scoped separately.

---

## 0. Restart context (2026-05-12)

We are resuming diffusion experimentation now that the V6 paper line on PAM
autoregressive ([v6/paper/paper.md](v6/paper/paper.md)) has stabilized
(`medium-pam-v3` ~30 PPL on WikiText-103, transformer baseline ~27 PPL).
The §11 failure mode in Part 1 (`diff_loss → 0` with sample collapse) is the
target to break, but only via V6-native primitives:

- complex algebra in [v6/core/complex.py](v6/core/complex.py)
- complex DDPM/DDIM in [v6/core/diffusion.py](v6/core/diffusion.py)
- matrix-state PAM in [v6/core/pam.py](v6/core/pam.py)
- multi-timescale SSM lanes in [v6/core/ssm.py](v6/core/ssm.py)
- FFT image codec in [v6/core/image_codec.py](v6/core/image_codec.py)
- memory hierarchy in [v6/core/memory.py](v6/core/memory.py)

No softmax attention, no UNet skip connections, no generic timestep MLP
grafted on top.

---

## 1. Phase 0 — observability (landed 2026-05-12)

Goal: stop reading log artifacts as if they were signal. Two changes in
[v6/train.py](v6/train.py):

### 1.1 `div=n/a` for single-bank presets

The dual-bank diversity loss in [v6/backbone.py](v6/backbone.py) is only
populated when `single_bank=False`. `medium-pam-v3` is single-bank, so
`output.diversity_loss` is `None`. Previously the trainer printed `div=0.00e+00`
which looked like a real (collapsed) value. After this change:

```
diff_loss=0.0682 div=n/a lr=1.00e-04 | 13.1 samples/s ETA 66m03s
```

Same change applied to the per-batch line, the `[gen_every]` Discord message,
and (implicitly) the epoch summary line.

### 1.2 `--log_diff_diagnostics` flag

New optional flag (default off, zero-overhead). When on, every
`--log_interval` batch appends:

```
| pred=<L2> tgt=<L2> ratio=<pred/tgt>
```

Implemented in `DiffusionTrainer._format_diff_diagnostics`. The `ratio` is the
single most useful number for the §11 collapse signature: when `diff_loss` is
tiny but `ratio` is also tiny, the model is collapsing to small predictions
(memorizing a low-magnitude solution) rather than denoising.

The flag is enabled by default in
[scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh](scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh).

---

## 2. Phase H — hardware profiles (landed 2026-05-12)

**Deferred pending AR focus:** Throughput lock (H-A), 4090 fallback (H-B), and capacity preset (H-C) runs are **not** scheduled until the team returns to diffusion; script defaults below remain for reference.

Default training target for Part 2 is the **RTX Pro 6000 (96 GB)**. The 4090
is kept as a fallback. The diffusion script now takes `--hw {pro6000|4090}`:

| Profile | `--batch_size` | `--compile_mode` | `--num_workers` |
|---------|---------------:|------------------|----------------:|
| `pro6000` (default) | 18 | `max-autotune` | 12 |
| `4090` | 3 | `reduce-overhead` | 4 |

PAM dual-form attention buffer in [v6/core/pam.py](v6/core/pam.py) at
`H=6, T=2048, bf16` is roughly `B * 0.8 GB` per layer × 16 layers. Activations
roughly double that. The Pro 6000 default targets ~80 GB of the 96 GB.

The chosen profile is recorded in `RUN_INFO.txt` (via `write_run_info`) and
echoed in the startup banner.

---

## 3. Runs

### 3.1 Run A — pre-Phase-0 baseline (2026-05-12, partial log)

```
log_dir : logs/v6/wikitext103_diffusion_text_medium_pam_v3_20260512_083801_d0697e1/
script  : scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh (pre Phase 0/H)
host    : rtx6000pro.asia-south2-a (Pro 6000 96 GB)
preset  : medium-pam-v3 (~100.7M params, single_bank, PAM H=6 d=64, GSP, RoPE)
mode    : diffusion_text  steps=1000  cosine  x0  DDPM
seq_len : 2048   batch_size : 3   epochs : 10
compile : reduce-overhead   amp_dtype : bf16
memory  : ~18 GB used / 96 GB available  (~80 % free, batch undersized for hardware)
```

Loss trajectory (first ~4500 batches of epoch 1):

| Batch | diff_loss | div | samples/s |
|------:|----------:|-----|----------:|
| 0     | 1.0145 | n/a | 0.0 |
| 500   | 0.3485 | n/a | 8.5 |
| 1000  | 0.2827 | n/a | 11.1 |
| 2000  | 0.0736 | n/a | 13.2 |
| 3000  | 0.0152 | n/a | 14.0 |
| 4000  | 0.0195 | n/a | 14.5 |
| 4500  | 0.0125 | n/a | 14.7 |

Same fast-collapse-of-train-loss pattern as Part 1 §11 (`diff_loss → 0` long
before any sample arrives). No `pred_norm` available — pre-dates the Phase 0
diagnostic flag. Superseded as the canonical May-12 throughput story by **Run B** (batch ~18, Phase 0 diagnostics); Run A remains a useful `B=3` anchor.

### 3.2 Run B — medium-pam-v3 diffusion_text (2026-05-12, **executed then killed**)

```
log_dir : logs/v6/wikitext103_diffusion_text_medium_pam_v3_20260512_090451_3f56d9b/
log_file: v6_diffusion-text_medium-pam-v3.log
host    : rtx6000pro.asia-south2-a.c.ankpal-1673529693693.internal (Pro 6000 96 GB)
preset  : medium-pam-v3 (~100.7M params), diffusion_text, WikiText-103
seq_len : 2048
batch   : implied 57831 train chunks / 3213 batches/epoch ≈ 18 microbatch
compile : reduce-overhead (log shows reduce-overhead, not max-autotune)
workers : 12   AMP: bf16   TF32: True
objective: DDPM 1000 steps, cosine, prediction_target=x0
script  : scripts/run_v6_diffusion_text_wikitext103_medium_pam_v3.sh (Phase 0: div=n/a, --log_diff_diagnostics)
checkpoints: checkpoints_v6_wikitext103_diffusion_medium_pam_v3/ (best_model.pt updated through epoch 3)
```

**Progress:** Epochs **1–3** completed (epoch-end samples + val); **epoch 4** in progress when the run was **stopped** (log shows ~batch 1100/3213 of epoch 4).

**Metrics (epoch summary lines):**

| Epoch | Train diff loss (avg) | Val loss | Notes |
|:-----:|----------------------:|---------:|-------|
| 1 | 0.1565 | 0.0032 | *best* checkpoint |
| 2 | 0.0012 | 0.0002 | *best* checkpoint |
| 3 | 0.0001 | 0.0000 | *best* checkpoint |

Mid-epoch lines: `diff_loss` → ~0 within thousands of steps; `pred` / `tgt` / `ratio≈1` on `--log_diff_diagnostics` — **global latent scale match, not good prose.**

**Representative epoch-end decoded samples** (verbatim from log, trimmed to ~200 chars; encoding artifacts as logged):

- After epoch 1: `[sample 1]` `manuscript manuscript manuscript Officer Duffy Officer Officer manuscriptHT manuscript manuscriptmouseGETptic manuscript Rivera Officer manuscripthou Rivera subreddits",casters BO Amar manuscript` … ; `[sample 2]` `Officerotenthou Hezbollah manuscript manuscript manuscript manuscript hp Officer hp manuscript manuscript Pokémon hp manuscript manuscript", maj Deadpool Moff manuscriptptic Coul Voyager manuscript` …
- After epoch 2: `[sample 1]` `ÍÍÍÍsessionInputadmin OfficerzscheadminMrssixkat OfficerMrs OfficerMrs DIRECT________ Officer amplifyburseadmin amplify OfficerCHARsix Utility Officer Officerfoliossix OfficersixMrsMrsMrs DIRECT DIREC` … ; `[sample 2]` `sixLayouthandersixMrssixfolios Officersix Officer Utility OfficersixMrssixsixfoliossixsixsixsixdelaydar Persona Utility OfficerargonargonÍÍMrsMrssixsixsixsixsixsixMrssixkatadminMrsMrsMrssixsixMrsMrsMr` …
- After epoch 3: `[sample 1]` `.;katÍÍ.;six________MrskatLayoutkatkatkatadminStudentskatkatkatkatMrssixLayoutItalyMrsMrssixsixadminMrssix RahulkattrisixsixfoliosadminsixMrssixfoliosargon facilit Utilityadminsixsix redundMrssixsixad` … ; `[sample 2]` `UtilityÍÍ actionGroup actionGroupsixkatkatÍÍ actionGroupsix.;katadminkatMrskat________InterestMrsMrsMrsMrsMrskatsixsixsixsixsixsixsixdelaysixsixsixsix642 Utility.;foliosfoliosfolios.; Utility Utility` …

**Decision:** Same failure class as [EXPERIMENTS_V6_DIFFUSION.md](EXPERIMENTS_V6_DIFFUSION.md) §11 — **near-zero latent MSE, unusable generations**. **Run killed**; **not** where to spend more GPU until autoregressive parity work and diffusion-specific fixes (EMA, ε-target, per-t diagnostics, sampling protocol) are designed. **No further diffusion epochs** planned under this config.

### 3.3 Run C — Phase H-B 4090 fallback verification (**deferred pending AR focus**)

Same script on the 4090 host with `--hw 4090`. Confirm no regression vs the
pre-Phase-0 4090 behavior.

### 3.4 Run D — Phase H-C capacity test (**deferred pending AR focus**)

Define a new `medium-pam-v3-300m` preset in [v6/config.py](v6/config.py)
(target ~3× params via `dim` and `num_layers`). Run on Pro 6000 at the
largest batch that fits. Same wall-clock budget as Run B; compare val loss
and sample quality.

---

## 4. Phase 1 plan — V6-native diffusion_text evolutions (**deferred pending AR focus**)

Sequential one-experiment-per-script, mirroring the V7 log discipline. Each
experiment forks the diffusion script, changes one CLI/config flag, and lands
a row below.

### 4.1 Controlled sweeps (no architecture change)

| ID    | Variable                  | Values                                  |
|-------|---------------------------|-----------------------------------------|
| T-S1  | `--prediction_target`     | `x0` vs `epsilon`                       |
| T-S2  | `--sampling_method` × steps | DDPM-1000, DDIM-{50, 100, 250}        |
| T-S3  | `--noise_schedule`        | `cosine` vs `linear`                    |
| T-S4  | `--seq_len`               | 1024 vs 2048                            |
| T-S5  | `--batch_size`            | per Phase H profile                     |

### 4.2 V6-native architectural evolutions

Each justified by an existing V6 primitive — none borrowed from
transformer-style diffusion fixes.

| ID   | Idea                                  | V6 primitive leveraged          |
|------|---------------------------------------|---------------------------------|
| T-N1 | PAM-state-conditioned timestep        | `PhaseAssociativeLayer` decay   |
| T-N2 | Phase/magnitude-decoupled noise       | `ComplexNoiseSchedule`          |
| T-N3 | SSM-lane-weighted denoising loss      | multi-timescale SSM             |
| T-N4 | PAM-head diversity for single-bank    | PAM heads (H=6)                 |
| T-N5 | Complex-EMA on PAM weights            | complex unitary structure       |
| T-N6 | Memory hierarchy as denoising context | working/internal/episodic mem   |

### 4.3 Stop rule

One epoch on full WikiText-103. If neither the `diff_loss` curve nor the
`gen_every` qualitative samples improve over the active baseline by a clear
margin, stop, document the negative result, and move to the next idea.

---

## 5. Phase 2 plan — diffusion_image (**deferred pending AR focus**; after Phase 1 conclusions)

| ID   | Idea                                       | V6 primitive |
|------|--------------------------------------------|--------------|
| I-S1 | Patch encoder, CIFAR-10, 10M and 100M      | baseline scale |
| I-N1 | FFT bands ↔ multi-timescale SSM lanes      | `FFTImageEncoder` + ssm |
| I-N2 | Pixel-space loss head (in addition to latent) | image_codec decoder |
| I-N3 | Patch positions via PAM RoPE               | `pam_rope` on patches |
| I-N4 | EMA + label-free CFG dropout               | training-loop only |

---

## 6. Update protocol

1. **One experiment per run script** in `scripts/`, named
   `run_v6_diffusion_text_<id>.sh` (or `_image_`). Forks the medium-pam-v3
   diffusion script and changes one flag.
2. **One log dir** per run via `make_log_dir` from
   [scripts/log_utils.sh](scripts/log_utils.sh); checkpoint/log sidecar enables
   `--resume` reusing the same log dir.
3. **One row in §3** above with the loss trajectory and qualitative sample
   notes. Negative results stay in the table.
4. **Roll-up entry** in [EXPERIMENTS_V_6_7_8_9.md](EXPERIMENTS_V_6_7_8_9.md)
   after the run completes.
