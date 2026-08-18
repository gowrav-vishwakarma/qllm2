# V13 experiments — selective PAM (delta + vault + phase addressing)

Lab notebook for V13: defaults make Stage-6 recall levers the production path.

## Smoke — 10M recall curriculum (`v13_micro_10m_recall`)

**Launch:**
```bash
bash scripts/run_v13_smoke.sh
# TOKEN_BUDGET=5000000 BATCH_SIZE=8 (defaults)
```

**Gates (before scale):**
| Probe | Target | Path |
|-------|--------|------|
| KV-recall @2048 | >90% | `checkpoints_v13/smoke_recall/behavioral.json` |
| Effective rank (layer 4) | >50% of d | `checkpoints_v13/smoke_recall/probes/` |
| Gate selectivity | \|p_content − p_filler\| >0.05 | `checkpoints_v13/smoke_recall/gate_probe.json` |
| O(1) selftest | pass | `python -m v13.selftest` |

## Matched baselines (same recall mix)

```bash
bash scripts/run_v13_transformer_micro.sh   # ~10M transformer
bash scripts/run_v13_mamba_micro.sh         # optional Mamba reference
```

## Scale — 100M vs Transformer

```bash
bash scripts/run_v13_scale.sh
# TOKEN_BUDGET=100000000 (default)
```

## Results

_(Append rows after each completed run.)_

| Run | Tokens | Recall@2048 (n=1) | Eff-rank % | Gate Δ | Wiki PPL | Notes |
|-----|--------|-------------------|------------|--------|----------|-------|
| smoke_recall (interim) | ~1.5M | 23.3% | 39.4% (L4) | ~0.00 | — | seq_len mix; gates not met |
| smoke_recall_v2 | 5M (running) | — | — | — | — | seq_len=2048, fresh run |

## Lessons

- V13 copies V11 with `write_mode=delta`, `vault_state`, `write_phase_address`, `n_states=3` as defaults.
- `fused_e3` disabled when delta+multistate (K-loop path).
- Chat vocab (50261) auto-selected when `cfg.vocab_size > 50257`.
