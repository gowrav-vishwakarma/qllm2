# RTX 4090 GPU publication run — handoff record

## Run identity

- Date: 2026-07-13 (12:41–13:10 IST)
- Host: RTX 4090 (NVIDIA GeForce RTX 4090, 24564 MiB, driver 580.159.03)
- Git commit: `ddad7da`
- Env: `uv sync` + `uv sync --extra cuda` (xformers 0.0.32.post2), torch CUDA available
- Runner: `CHECKPOINT=checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt PRESET=v11_e3_k3_chat ./scripts/run_memory_probes_publication.sh gpu`
- Wall time: ~29 min. Peak GPU mem trivial (~0.5 GB during behavioral; 100M model).

## Checkpoint under test

- Path: `checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt`
- Tag: `round-8b-gate` (per `round_state.env`: LAST_ROUND=4, CUMULATIVE_TOKENS=8,000,000,000)
- Arch: V11 PAM, vocab 50261, dim 384, 16 layers, 6 heads, K=3 states, content-aware GSP gate, routing_content_aware=False, state_compete=False
- Params: 100,546,832 (~100M)

## Baselines

- Mamba: `state-spaces/mamba-130m-hf`, 129,135,360 params (~129M), 24 layers, hidden 768, state 16. (Slow HF sequential path — no mamba_ssm kernels; results still valid, just slow.)
- Transformer: SKIPPED (no local `checkpoints_transformer_*`; set TRANSFORMER_CHECKPOINT to enable).

## Artifacts (all schema-validated; `memory_probes.test_publication` = OK)

- `gates.json` — learned protect_gate / gamma / routing, content vs filler
- `memory_probes_rank_text_*.json` (layers 0,4,8,12,15) — trained PAM state rank on 50K WikiText
- `memory_probes_language_filler_20260713_130449.json` — 50-projection sweep
- `memory_probes_arch_compare_20260713_130507.json` — PAM/KV/Mamba state utilization
- `v11_behavior.json` — trained PAM contrastive invented-association recall (720 examples)
- `mamba_behavior.json` — Mamba baseline, same protocol
- `gpu_run.log` — full stdout

## Headline results

### Behavioral recall (invented key -> value, contrastive next-token, 8 candidates)

Overall mean accuracy: **V11 PAM = 0.21, Mamba-130m = 0.77**.

Single association (assoc=1) accuracy by context:
| ctx | V11 | Mamba |
|-----|-----|-------|
| 128 | 0.78 | 1.00 |
| 512 | 0.23 | 1.00 |
| 1024 | 0.18 | 1.00 |
| 2048 | 0.18 | 1.00 |

- V11 falls to near-chance (chance=1/8=0.125) once context > 128, except when the needle is at the very end (pos=1.0 assoc=1 stays ~0.35).
- Recency bias confirmed behaviorally: V11 acc by position 0.0/0.5/1.0 = 0.16/0.18/0.29. Predicted by the NIAH math probe.
- Mamba stays ~1.0 for single association at all tested contexts; degrades gracefully with more associations.

### Gate diagnostics (`gates.json`)

- mean protect p = 0.395 (init bias -3.0 => p~0.047, so training DID raise protection).
- mean (p_content - p_filler) = **-0.030 (NEGATIVE)** => model protects filler slightly MORE than content. Gate did not learn content-selective protection.
- mean gamma = 0.614 (aggressive decay; needle mostly gone by ~1-2K tokens without protection).

### Routing collapse (K=3 multistate)

- Every layer: token entropy = bar entropy = 1.099 = ln(3), winner share = 0.00.
- The 3 states are used uniformly; routing never specializes. E3 capacity is effectively unused.

### State utilization (arch-compare, WikiText, 2000 tokens)

| arch | rank ceiling | eff rank | utilization | state cost |
|------|--------------|----------|-------------|------------|
| pam | 64 | 15.3 | 23.9% | fixed O(d^2) |
| transformer | 64 | 55.3 | 86.4% | grows O(t*d) |
| mamba | 16 | 13.4 | 84.0% | fixed O(d_inner*d_state) |

PAM uses only ~24% of its d^2 state; Mamba uses 84% of a much smaller state.

### Trained rank by layer (50K WikiText, d=64)

Final effective rank: L0=6.6, L4=5.1, L8=15.6, L12=12.7, L15=7.1. Wikitext > random only in mid layers (L8: 15.6 vs 6.5; L12: 12.7 vs 4.3). Early/late layers barely separate structured text from random.

### Language filler (unchanged, mechanism-level)

lang rel mean=9.28, lang/rand mean=94x, beats random on all 50 seeds. Consistent with Mac results; this is a projection-clustering effect, not a trained-model claim.

## Interpretation (short)

The intrinsic math probes (capacity, NIAH ceilings) show PAM *can* store; the trained probes show this ~100M checkpoint *did not learn to use* that capacity: gate is not content-selective, K=3 routing collapsed, only ~24% of state rank used, and behavioral recall collapses beyond 128 tokens. Mamba (similar params, fixed state) realizes its mechanism far better. This is the Phase-A-ceiling vs Phase-B-realization gap the paper predicted.
