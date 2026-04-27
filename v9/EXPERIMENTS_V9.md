# V9 Experiments: PAM Sequence-Mixing Upgrades

V9 keeps V7 frozen as the historical PAM baseline and tests small, isolated
changes that target the PAM-vs-attention gap.

## Current Best PAM Run (as of 2026-04-27)

The single best PAM result we have on WikiText-103 so far is the
**V9 `gate` (confounded)** run at **val PPL 29.57**.

| Property | Value |
|---|---|
| Variant | `medium_h16_gate` (V9 preset, dim **384**) |
| Params | **105.1M** |
| Architecture | V7 `medium_h16_flat` backbone + V9 PAM output gate |
| `pam_output_gate` | **True** — extra `nn.Linear(2*dim → inner_dim)` + `2 * sigmoid` on the PAM readout (~4.7M params/layer × 16 layers) |
| `use_reverse_assoc` | **True** (inherited from V7 defaults — not deliberately set) |
| `pam_short_conv` | 0 (off) |
| `pam_head_compete` | False (off) |
| `qk_norm` | False |
| Activation | ModSwish |
| `chunk_size` | 256 |
| Data | WikiText-103, seq_len 2048, batch 3, 10 epochs |
| Best epoch | 10 |
| Throughput | ~27.3k tok/s |
| Wall time | 12.24h |
| Log | `logs/v9/pam_gate_wikitext103_20260426_195110_80b725b_dirty/v9_medium_h16_gate_wikitext103.log` |
| Run command | `bash ./scripts/run_v9_pam_upgrade.sh --variant gate` (against the pre-clean defaults) |

### Comparison snapshot

| Model | Batch | Params | Val PPL | Δ vs current best |
|---|---:|---:|---:|---:|
| Transformer (apples-to-apples) | 3 | ~100M | **27.08** | −2.49 |
| **V9 `gate` (confounded) — current best PAM** | **3** | **105.1M** | **29.57** | — |
| V7 Exp7a (ModSwish, flat) | 3 | ~100M | 29.73 | +0.16 |
| V6 `medium-pam-v3` | 3 | ~100.4M | 29.95 | +0.38 |
| Transformer | 6 | ~100M | 23.13 | (different batch — not directly comparable) |

### Caveats (read before citing this number)

- **Confounded**: `use_reverse_assoc=True` was inherited from V7 defaults, so
  the +0.16 win over V7 7a cannot be cleanly attributed to the gate. It is
  "gate **plus** inherited reverse association", not a clean gate ablation.
- **Capacity advantage**: 105.1M params vs. ~100M baselines — part of the
  small gain may just be the +5M parameter bump.
- **Generation quality**: still loops ("University of Michigan ..."),
  `rep3 = 0.160`, `rep4 = 0.081`, `uniq = 0.412`.
- **Acceptance gate not met**: V9 promotion threshold was `< 29.2`; this run
  did not clear it.

The clean parameter-matched follow-up (`gate_100m`, dim 372,
`use_reverse_assoc=False`) was started but pivoted to the
`compete_revassoc_100m` design before completion. Until that pivot run lands,
**29.57 PPL is the number to beat for any PAM-only configuration**.

## Baselines

| Model | Batch | Val PPL | Notes |
|---|---:|---:|---|
| Transformer baseline | 3 | 27.08 | GPT-2 style attention baseline |
| V7 Exp7a | 3 | 29.73 | Best flat CGU+PAM result |
| Lean PAM | 3 | 32.09 | Removes CGU, shows channel gating is worth ~2.4 PPL |

## V9 Hypothesis

PAM is fast and recurrent at inference, but its sequence mixer is weaker than
attention because it lacks:

- output gating on the memory readout,
- sharp local pattern capture,
- normalized competition between retrieved positions.

V9 starts with the two lowest-risk changes:

- `pam_output_gate`: a learned real-valued gate on the PAM readout, initialized
  as identity with `2 * sigmoid(0) = 1`.
- `pam_short_conv`: a causal depthwise convolution before QKV projection,
  initialized as identity so the run starts from V7 behavior.

Score normalization is intentionally deferred until gate/conv results are known.
It is more likely to disturb complex phase/magnitude behavior.

## Run Matrix

Use:

```bash
bash ./scripts/run_v9_pam_upgrade.sh --variant baseline
bash ./scripts/run_v9_pam_upgrade.sh --variant gate
bash ./scripts/run_v9_pam_upgrade.sh --variant conv
bash ./scripts/run_v9_pam_upgrade.sh --variant gate_conv
```

Default settings mirror V7 Exp7a:

- `medium_h16_flat` shape: dim 384, 16 layers, 6 heads, head dim 64.
- ModSwish activation.
- WikiText-103, seq len 2048, batch size 3, 10 epochs.
- CGU, GSP, RoPE, fused QKV, flat `dt_bias=-4.0`.

## Acceptance Gate

Promote a variant only if it reaches val PPL `< 29.2` at epoch 10, closing at
least ~0.5 PPL of the V7-to-transformer gap without a major throughput collapse.

## 2026-04-27: V9 Gate Run, Confounded By Reverse Association

Run:

```bash
bash ./scripts/run_v9_pam_upgrade.sh --variant gate
```

Log: `logs/v9/pam_gate_wikitext103_20260426_195110_80b725b_dirty/v9_medium_h16_gate_wikitext103.log`

### Config Readout

This run completed, but it is **not a pure one-variable gate ablation**. The
logged config shows:

```text
pam_output_gate=True
pam_short_conv=0
use_reverse_assoc=True
hierarchical_dt=False
cross_level=False
qk_norm=False
multi_scale_loss=False
chunk_size=256
activation=swish
batch_size=3
```

The important caveat is `use_reverse_assoc=True`. That came from inherited V7
defaults, not from the V9 hypothesis. V7 already showed reverse association hurt
badly in the 7f-2 ablation, so this result should be read as **gate plus
inherited reverse association**, not clean gate.

### Epoch Results

| Epoch | Train PPL | Val PPL | tok/s |
|---:|---:|---:|---:|
| 1 | 115.64 | 54.29 | 25.4k |
| 2 | 50.91 | 42.06 | 27.3k |
| 3 | 42.98 | 37.90 | 27.3k |
| 4 | 39.21 | 35.28 | 27.3k |
| 5 | 36.55 | 33.11 | 27.3k |
| 6 | 34.39 | 31.72 | 27.3k |
| 7 | 32.63 | 30.66 | 27.3k |
| 8 | 31.25 | 30.02 | 27.3k |
| 9 | 30.21 | 29.62 | 27.3k |
| 10 | 29.61 | **29.57** | 27.3k |

Summary:

- Params: **105.1M**
- Best val loss: **3.3868**
- Best val PPL: **29.57**
- Wall time: **12.24h**
- Throughput: ~**27.3k tok/s** after warmup

### Baseline Comparison

| Model | Batch | Params | Val PPL | Delta vs V9 gate |
|---|---:|---:|---:|---:|
| Transformer baseline | 3 | ~100.3M | **27.08** | V9 worse by +2.49 |
| V9 gate, confounded | 3 | 105.1M | **29.57** | — |
| V7 Exp7a | 3 | ~100M | 29.73 | V9 better by -0.16 |
| V6 medium-pam-v3 | 3 | ~100.4M | 29.95 | V9 better by -0.38 |

This is directionally positive versus V7 and V6, but it does **not** meet the
V9 promotion gate of `<29.2`, and it remains meaningfully behind transformer B=3.

### Generation Quality

The final sample still shows repetition risk despite the small PPL gain:

```text
In 1923 , the University of Michigan took over as the university 's president
and the university 's board . In 1923 , the university was divided into four
divisions : The University , University of Michigan ; University of Michigan ;
University of Michigan ...
```

Final quality metrics:

- `rep3=0.160`
- `rep4=0.081`
- `restarts=0`
- `uniq=0.412`

Earlier samples had better uniqueness, but the final sample regressed into a
University-of-Michigan loop. So the run improves validation PPL slightly, but it
does not yet solve factual binding or repetition.

### Decision

Record this as a **positive but confounded** result. The next run must be a clean
gate ablation with `use_reverse_assoc=False` so we can tell whether the output
gate itself is helping.

Code defaults were cleaned after this readout:

- `medium_h16_flat`, `medium_h16_gate`, `medium_h16_gate_100m`,
  `medium_h16_conv4`, and `medium_h16_gate_conv4` now explicitly set
  `use_reverse_assoc=False`.
- The V9 launcher also passes `--no_reverse_assoc` as a safety check.

The completed confounded gate run had **105.1M** parameters, so its small gain
over V7/V6 could partly be capacity. For a real apples-to-apples result, run the
parameter-matched gate variant next:

```bash
bash ./scripts/run_v9_pam_upgrade.sh --variant gate_100m
```

This uses `medium_h16_gate_100m`: dim **372**, 16 layers, 6 heads, head_dim 64,
expand 3, `pam_output_gate=True`, `use_reverse_assoc=False`, and about
**100.5M** parameters.

Interpretation rule:

- Better than **29.57**: gate is likely genuinely useful and reverse association
  plus extra capacity were not responsible for the gain.
- Around **29.7-30.0**: the 29.57 run may have been noise or an interaction.
- Worse than **29.57**: inspect learned `rev_scale` from the confounded run
  before deciding whether gate and reverse association interacted usefully.

Compare the result directly against V9 confounded gate **29.57**, V7 Exp7a
**29.73**, V6 medium-pam-v3 **29.95**, and transformer B=3 **27.08**. Do not run
`conv` or `gate_conv` until this matched clean gate result is known.

## 2026-04-27: V9 Gate Matrix For Reviving V7 Ablations

The clean `gate_100m` run showed a worse early trajectory than the confounded
105M gate run:

- `gate_100m`, dim 372, `use_reverse_assoc=False`: epoch 1 val PPL **55.37**
- prior `gate`, dim 384, `use_reverse_assoc=True`: epoch 1 val PPL **54.29**,
  final best val PPL **29.57**

The user decision is to stop the in-flight clean run and test a small matrix of
previously-harmful V7 mechanisms under the V9 output gate. The hypothesis is
that the gate acts as a learned veto over the PAM readout: noisy but informative
features can be admitted when useful instead of always perturbing the residual
stream.

### Matrix

All variants use WikiText-103, seq len 2048, batch size 3, 10 epochs, dim 372,
16 layers, 6 heads, head dim 64, expand 3, ModSwish, RoPE, GSP, fused QKV, and
`pam_output_gate=True`.

| ID | Script Variant | Key Change | Params | Purpose |
|---|---|---|---:|---|
| C | `gate_100m` | gate only, `use_reverse_assoc=False` | 100.52M | Clean control |
| R | `gate_revassoc_100m` | gate + `use_reverse_assoc=True` | 100.52M | Test if gate detoxifies reverse association |
| N | `gate_qknorm_100m` | gate + `qk_norm=True` | 100.52M | Test normalized Q/K angles plus gated magnitude |
| K | `gate_conv4_100m` | gate + `pam_short_conv=4` | 100.58M | Test local pattern capture before QKV |

### Commands

```bash
bash ./scripts/run_v9_pam_upgrade.sh --variant gate_revassoc_100m
bash ./scripts/run_v9_pam_upgrade.sh --variant gate_qknorm_100m
bash ./scripts/run_v9_pam_upgrade.sh --variant gate_conv4_100m
```

### Decision Rules

- Any variant beating the clean gate control by **>=0.3 PPL** is a real signal.
- If the winner beats **29.57**, the gain is not just the prior +5M parameter
  bump.
- If two variants beat the control, run a stacked follow-up at dim 372, such as
  `gate + reverse_assoc + qk_norm`.
- If none beats the control, keep gate as the only V9-positive change and move
  to a larger PAM change such as cumulative normalization or a two-state PAM.

## 2026-04-27: Pivot To Novel Pure PAM (Compete + Reverse Assoc)

The matrix above was cancelled before launch in favor of a tighter, novel
hypothesis: keep PAM the only sequence mixer and add a zero-parameter
non-linearity that creates cross-head competition for output mass.

### Mechanism (zero learned parameters)

Per (batch, token, channel) location, heads compete via softmax over their
output magnitudes, then a fixed-mix scalar gate amplifies the winner and
softly attenuates the rest. Phase preserved.

```text
mag  = |y|                                # [B, H, T, d], real
w    = softmax(mag, dim=H)                # heads compete per (token, channel)
gate = (1 - alpha) + alpha * H * w        # alpha = 0.5, fixed scalar
y    = y * gate.unsqueeze(-1)             # phase preserved (gate is real positive)
```

At alpha = 0.5:

- All heads equal: gate = 1.0 (identity, safe init).
- Dominant head: gate -> 1 + H/2.
- Suppressed head: gate -> 0.5.

This is the parameter-free analogue of the V9 sigmoid gate that produced our
best confounded result (29.57 PPL). The V9 gate conditioned on x with a 4.7M
projection per layer; cross-head competition conditions on y itself with no
learned parameters.

### Speed audit

| Op | Cost | Train | Infer |
|---|---|---|---|
| `cabs(y)` | elementwise | O(n) | O(1) |
| softmax over H heads | constant H | O(n) | O(1) |
| broadcast multiply | elementwise | O(n) | O(1) |

Both speed properties preserved.

### Architecture

Base = V6 medium-pam-v3 expressed in V9 plumbing:

- dim 384, 16 layers, 6 heads, head_dim 64, single CGU expand 3.
- ModSwish (kept over ModReLU since V7 ablation showed strict improvement).
- GSP on, RoPE on, fused QKV.
- `qk_norm = False` (V6 default).
- `use_reverse_assoc = True` (the only validated lever from V9 gate run).
- V9 learned gate stripped (`pam_output_gate = False`).
- `pam_short_conv = 0`.
- New: `pam_head_compete = True`, `pam_head_compete_alpha = 0.5`.

Approximate parameter count: 100M (head competition adds zero learned params,
reverse_assoc adds 16 scalars).

### Run

```bash
bash ./scripts/run_v9_pam_upgrade.sh --variant compete_revassoc_100m
```

### Decision Rules

| Outcome | Action |
|---|---|
| val PPL <= 27.08 | Pure PAM beats Transformer. Lock in. Stack with per-channel decay or state expansion. |
| 27.08 to 28.5 | Real progress over V7 7a (29.73). Stack with per-channel decay next. |
| 28.5 to 29.5 | Matches existing baselines. Sweep alpha in {0.25, 0.75, 1.0}. |
| >= 29.5 | Head collapse or destabilized training. Diagnose. |

### Baselines For Comparison

| Model | Batch | Params | Val PPL |
|---|---:|---:|---:|
| V6 medium-pam-v3 | 3 | 100.4M | 29.95 |
| V7 Exp7a (ModSwish) | 3 | ~100M | 29.73 |
| V9 gate (confounded) | 3 | 105.1M | 29.57 |
| Transformer | 3 | ~100M | **27.08** |
| Transformer | 6 | ~100M | 23.13 |

## 2026-04-27: V9 Gate-MLP + Reverse Assoc (Post-Compete Pivot)

Status: **queued**.

The `compete_revassoc_100m` run did not show a win signal after two epochs:

| Run | Params | Epoch 1 Val PPL | Epoch 2 Val PPL | Notes |
|---|---:|---:|---:|---|
| V9 `gate` (confounded) | 105.1M | 54.29 | 42.06 | best PAM so far, but +5M params and inherited reverse assoc |
| V7 Exp7a | ~100M | 56.15 | 43.20 | clean flat ModSwish PAM baseline |
| V6 medium-pam-v3 | 100.4M | 57.94 | 43.83 | V6 architectural ancestor |
| V9 `compete_revassoc_100m` | 100.4M | 56.27 | 44.44 | zero-param competition; worse early trajectory |

Interpretation: the zero-parameter cross-head competition did not replace the
learned V9 gate. The best evidence still points at **learned PAM readout gating**
as the useful lever, so the next experiment strengthens that gate directly.

Important negative learning: applying a direct PAM readout non-linearity/gate
without a learned linear projection is **not** showing the same benefit as the
linear+sigmoid gate. The useful part of V9's best run appears to be the
input-conditioned learned projection into gate logits, not merely "put a
sigmoid-like non-linearity on PAM output." Direct competition on `y` preserved
phase and added no parameters, but it underperformed both the learned-gate run
and the clean V7 trajectory at epoch 2.

### Mechanism

The existing V9 gate is a single real linear projection over the complex input:

```text
logits = Linear([real(x), imag(x)])
gate   = 2 * sigmoid(logits)
y      = y * gate
```

The new gate uses a 2-layer MLP when `pam_gate_hidden > 0`:

```text
h      = SiLU(Linear([real(x), imag(x)]))
logits = Linear(h)
gate   = 2 * sigmoid(logits)
y      = y * gate
```

The final linear is zero-initialized, so the run starts from exact identity:
`logits = 0`, `gate = 1`. This preserves the V7 starting behavior while giving
the gate a non-linear input-conditioned path once training begins.

### Configuration

| Field | Value |
|---|---|
| Preset | `medium_h16_gate_mlp_revassoc_100m` |
| Script variant | `gate_mlp_revassoc_100m` |
| Params | **101.094M** validated |
| dim | 368 |
| layers | 16 |
| PAM heads | 6 |
| PAM head dim | 64 |
| CGU expand | 3 |
| activation | ModSwish |
| `pam_output_gate` | True |
| `pam_gate_hidden` | 368 |
| `use_reverse_assoc` | True |
| `pam_head_compete` | False |
| `pam_short_conv` | 0 |
| `qk_norm` | False |
| Data | WikiText-103, seq_len 2048, batch 3, 10 epochs |

Validation before launch:

- Parameter count: **101,094,240**.
- Gate projection type: `Sequential`.
- Final gate linear `weight` and `bias`: exactly zero at initialization.
- Tiny forward pass: finite logits with shape `[1, 8, 50257]`.
- Launcher syntax: `bash -n scripts/run_v9_pam_upgrade.sh` passed.

### Run

```bash
bash ./scripts/run_v9_pam_upgrade.sh --variant gate_mlp_revassoc_100m
```

### Decision Rules

| Outcome at epoch 10 | Reading | Next step |
|---|---|---|
| <= 29.0 | Bigger gate beats smaller gate. Gate is the lever. | Stack with per-channel decay or fewer heads. |
| 29.0 to 29.6 | Parity with confounded gate but at ~100M instead of 105M. Real progress. | Stack per-channel decay. |
| 29.6 to 29.9 | Bigger gate gives no edge over single-layer gate. The 29.57 may mostly be capacity. | Pivot to fewer-heads + bigger-state experiment. |
| >= 29.9 | Gate is not the lever. | Pivot to per-channel decay or post-PAM mixing. |
