# V9 Experiments: PAM Sequence-Mixing Upgrades

V9 keeps V7 frozen as the historical PAM baseline and tests small, isolated
changes that target the PAM-vs-attention gap.

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
