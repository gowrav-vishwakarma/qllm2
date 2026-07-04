# V11 PAM model releases

Single source of truth for every shipped round: data recipe, cursors, metrics,
checkpoint paths, and Hugging Face revision tags. Each `+2B` pretrain round
resumes the previous best (fresh tokens via cursors), runs a smoltalk2 SFT, and
ships to [gowravvishwakarma/qllm-pam-v11-e3k3-chat](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat)
with a revision tag.

- Preset: `v11_e3_k3_chat` (phase-aware GSP gate, vocab 50261: ChatML + `<think>`/`</think>`).
- Pretrain blend: `dclm` + `fineweb` (web/knowledge) + `smoltalk2_mid` (reasoning/chat, ChatML-rendered), grammar warmup then weighted blend.
- SFT: `smoltalk2` SFT config (think capped). Disjoint config from the Mid blend pool.
- Freshness: every source advances `per_source_docs` in the checkpoint; next round skips consumed docs/rows. Web unique; Mid ~35B tokens (~50+ rounds).

## Legend

- `parent`: checkpoint this round resumed from (`scratch` for the first v2 round).
- `per_source_docs`: cumulative consumed docs/rows at round end (from checkpoint). These become the next round's skip cursors.
- `holdout_ppl`: in-distribution DCLM holdout PPL (primary metric; not WikiText).
- `rankL4` / `gate_delta`: memory-utilization probes (`p_content - p_filler`).
- `hf_tag`: Hugging Face revision. `blend_repeat`: true once a blend source wrapped (uniqueness no longer guaranteed).

---

## Round 0 (legacy, magnitude gate) - reference only, DO NOT continue

| field | value |
|-------|-------|
| line | legacy (content-blind magnitude gate) |
| preset | `v11_e3_k3_chat` @ vocab 50259 (pre-reasoning-token) |
| pretrain | ~10B DCLM-Edu + FineWeb-Edu (web only, no blend) |
| sft | `smol-smoltalk` (old small dataset) |
| checkpoints | `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt`, `checkpoints_v11_sft_chat_smoltalk/best_model.pt` |
| hf_tag | `main` on Hugging Face (legacy ~10B weights); supported via unified `hf_release/` — pin `--revision main` |
| note | Trained under magnitude-only gate + vocab 50259. Not resumable by the v2 line (gate + vocab changed). Kept for comparison; chat works with current `run_chat.py` when downloaded from HF `main`. |

---

## v2 gate line (phase-aware gate, vocab 50261, blended pretrain)

New checkpoints: `checkpoints_v11_e3_k3_chat_pretrain_v2/` (pretrain), `checkpoints_v11_sft_chat_smoltalk_v2/` (SFT).

| round | hf_tag | parent | pretrain_total | round_tokens | mix_seed | fineweb | weights (dclm,fineweb,mid) | warmup | think_frac | per_source_docs (end pretrain) | holdout_ppl | sft_ppl | sft_acc | smoke | blend_repeat | date |
|-------|--------|--------|----------------|--------------|----------|---------|----------------------------|--------|-----------|------------------------------|-------------|---------|---------|-------|--------------|------|
| 1 | round-2b-gate | scratch | 2B | 2B | 42 | sample-10BT | 48,48,4 | 1B | 0.15 | dclm 765737, fineweb 766613, mid 28577 | 35.42 | 7.20 | 0.591 | [samples](../hf_release/SAMPLES_round-2b-gate.md) 2026-07-04 | false | **2026-07-04** |

**Round 1 token mix (pretrain, measured):** ~52% DCLM-Edu, ~40% FineWeb-Edu, ~8% smoltalk2 Mid
(Mid docs are much longer than web docs despite 4% doc weight). **SFT:** smoltalk2 SFT config,
hard filter, think_fraction 0.15. **Architecture:** content-aware GSP gate, vocab 50261.
