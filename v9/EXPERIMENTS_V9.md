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
