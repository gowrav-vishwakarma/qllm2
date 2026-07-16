# PAM Recall Program — implementation + RTX Pro 6000 run guide

Handoff doc for the recall-oriented architecture changes made on 2026-07-14.
It explains **what was implemented and why**, how it was validated locally, and
**exactly what to run on the RTX Pro 6000**. Companion plan:
`.cursor/plans/pam_v12_recall_program_aee548e4.plan.md` (do not need to re-read).

**Status (2026-07-16):** Stages 2–5 complete on RTX PRO 6000. Full results, per-arm
tables, probe breakdowns, and consolidated learnings are in
[v11/EXPERIMENTS_V11.md](v11/EXPERIMENTS_V11.md) (sections *Stage-2 A/B* through
*consolidated findings*). **Bottom line:** recall@2048 plateaued at 0.15–0.25 across
19 configs (chance = 0.125); gate selectivity is solved; write interference and decay
are the structural blockers. Stage 6 (delta-write, vault state, phase addressing) is
planned — see *consolidated findings* and `.cursor/plans/recall_program_stage_6_*.plan.md`.

---

## 1. Why (the 4090 Phase-B finding)

Full record: `logs/memory_probes/publication/gpu/RUN_INFO.md`. The trained V11 PAM
checkpoint `round-8b-gate` (`checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt`,
~100M params) has the **intrinsic capacity** to store bindings (Phase-A math probes
pass) but **did not learn to use it** (Phase-B behavioral realization fails):

| symptom | measurement | meaning |
|---|---|---|
| behavioral recall collapses | single-assoc acc: 0.78 @128 → 0.18 @2048 (Mamba-130m = 1.00 at all) | can't recall past ~128 tokens |
| gate not content-selective | mean `p_content − p_filler` = **−0.030** (≈flat) | protect gate ignores surprisal |
| routing collapse | every layer token-entropy = ln(3), winner-share 0.00 | K=3 states never specialize |
| aggressive decay | mean gamma = 0.614 | needle gone by ~1–2K tokens |
| low state utilization | eff-rank 15.3/64 = **24%** (Mamba 84%) | most of the O(d²) state is wasted |

The overall goal (per user): **make PAM keep more facts in the smallest model and
prove it beats Mamba/Transformer while staying O(1)**. This program attacks the
three most actionable causes: the flat gate, the short memory horizon, and the lack
of any training pressure that *demands* long-range recall.

---

## 2. What was implemented (Stage 1 — all landed in this branch)

Three levers, all **off by default** (defaults are bit-identical to the old model —
`python -m v11.selftest` still passes train≡recurrent and fused-CE equivalence).

### 2a. Gate-surprisal auxiliary loss
Ties the GSP write-protect probability `p` to per-token surprisal so the gate learns
*when to write vs. freeze* instead of emitting a flat ~0.4 on every token.

- **Config** (`v11/model.py` `V11Config`): `gate_surprisal_lambda` (weight, 0=off),
  `gate_surprisal_tau` (nats), `gate_surprisal_sign`.
- **Mechanism**: each layer stashes its mean protect prob `[B,T]`; `_hidden_to_lm`
  now returns `(lm, aux_loss, gate_probs)` (compile-safe, same pattern as route-aux).
  The trainer (`v7/train.py`) computes a **detached** per-token CE surprisal via the
  new `linear_ce_per_token` (`v11/fused_ce.py`) and applies masked BCE:
  `target_p = sigmoid(sign · (median_ce − surprisal) / tau)`. Grad flows only into
  each layer's `protect_gate`.

**Sign convention (important — differs from the plan's written metric).** Working
through the GSP update (write the needle at low `p`, freeze state through filler at
high `p`), the **recall-optimal** direction is *protect filler / write content*,
which drives `p_content − p_filler` **negative**. The plan's success bullet said
"low-surprisal → protect" (agrees) but its numeric target said `p_content − p_filler
> +0.05` (opposite). We implement the recall-optimal direction as the default
(`gate_surprisal_sign = +1`) and judge success on **absolute behavioral recall**
(unambiguous) plus `|p_content − p_filler|`, not the signed probe number. Set
`gate_surprisal_sign = -1` to get the probe's "protect content more" convention.

### 2b. `gamma_floor` decay reparam
Lifts the base (pre-GSP) decay onto `[gamma_floor, 1)`:
`base = gamma_floor + (1 − gamma_floor)·exp(−softplus_dt)`. With `gamma_floor≈0.98`
unprotected state survives far longer, directly attacking the ~1–2K-token cliff.
Applied in both gamma paths (`_gamma_and_vprime` head/per-channel and the fused
K-state `_gamma_all_and_vprime`). Config: `gamma_floor` (0=off).

### 2c. Synthetic recall curriculum (data)
Web/edu pretrain never *demands* long-range copy/recall. New `recall` source
(`v7/data.py`) emits plain-text passkey / key→value / multi-needle documents: a fact
appears, filler intervenes, then the fact must be reproduced (predicting the final
value requires recall). Log-uniform gaps (~2–200 sentences) cover short and long
range. **Vocabulary is deliberately disjoint** from the behavioral eval
(`memory_probes/behavioral.py` KEYS/VALUES) — verified zero overlap — so the eval
stays a held-out test, not train-on-test. Registered in `SOURCE_REGISTRY` as
`{'schema':'text','kind':'synthetic'}` and wired into both `pretrain_mix` builders.

### 2d. New CLI + preset + tooling
- `v11/train.py` flags: `--gamma_floor`, `--gate_surprisal_lambda`,
  `--gate_surprisal_tau`, `--gate_surprisal_sign`, and `--no_resume_cursor`
  (warm-start weights *without* seeding the multi-million-doc skip cursor).
- Preset `v11_e3_k3_chat_recall` (`v11/model.py`): same params/vocab as chat (loads
  round-8b-gate weights) with the levers pre-set — for reference; the A/B toggles
  levers via CLI instead so each arm is isolated.
- `scripts/eval_recall_gate.sh` — gate probe + behavioral suite + verdict for one ckpt.
- `scripts/recall_gate_verdict.py` — prints/writes ship criteria.
- `scripts/run_v11_recall_ab.sh` — matched A/B arms (control/gate/floor/recall/combo),
  warm-start or scratch.

### Files touched
`v11/model.py`, `v7/train.py`, `v7/data.py`, `v11/fused_ce.py`, `v11/train.py`,
`v11/bench_step.py`, `scripts/run_memory_behavioral.py` (all `_hidden_to_lm`
unpackers updated to the 3-tuple), plus the 3 new scripts.

---

## 3. Local validation done (CPU, no GPU needed)

- `python -m v11.selftest` → **all PASS** (levers off ⇒ identical to old model).
- End-to-end micro test: `_hidden_to_lm` returns `[L,B,T]` gate probs; gate BCE
  produces a **non-zero `protect_gate` gradient**; `gamma_floor` stored/applied.
- `recall` generator: deterministic, skip-aware, disjoint vocab, streams through
  `StreamingTokenChunkDataset` into 128-token training chunks.
- Full training path smoke on the 4090: config shows all levers active, `recall`
  blended into the live `pretrain_mix` stream, cursor-resume works. (Stopped before
  convergence — the real A/B is below.)

---

## 4. What to run on the RTX Pro 6000

Environment (same as 4090): `uv sync && uv sync --extra cuda`; confirm
`uv run python -c "import torch;print(torch.cuda.is_available())"` → `True`.
The warm checkpoint must exist:
`checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt` (round-8b-gate).

### Step 1 — Stage-2 A/B screen (control vs combo, warm-start, ~150M tokens/arm)

Chosen config: warm-start from round-8b-gate, **no cursor-skip** (fast startup,
accepts minor data overlap — fine for a relative screen), gate sign = +1.

```bash
ARMS="control combo" \
TOKEN_BUDGET=150000000 \
RESUME=checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt \
NO_CURSOR=1 \
./scripts/run_v11_recall_ab.sh
```

This trains each arm, then automatically runs the eval gate (gate probe + 720-example
behavioral suite) and writes a verdict per arm. Arms:
- **control**: stock arch, web-only blend (`dclm,fineweb`).
- **combo**: `--gamma_floor 0.98 --gate_surprisal_lambda 0.1 --gate_surprisal_sign 1.0`
  + `recall` mixed into the blend (`dclm,fineweb,recall` @ 48,48,3).

Notes:
- `pretrain_mix` streams dclm/fineweb from HF (network needed). `dclm-edu` is slow to
  open its first shard (~1 min) — this is normal, not a hang.
- The 6000 has more VRAM than the 4090; you can raise `BATCH_SIZE` (default 16) for
  throughput. Tune via env, e.g. `BATCH_SIZE=32`.
- Outputs: `checkpoints_v11_recall_ab/{control,combo}/` (weights + `eval/verdict.json`).

### Step 2 — read the verdicts

```bash
cat checkpoints_v11_recall_ab/control/eval/verdict.json
cat checkpoints_v11_recall_ab/combo/eval/verdict.json
```

Or re-run the gate on any checkpoint standalone:
```bash
LABEL=combo ./scripts/eval_recall_gate.sh checkpoints_v11_recall_ab/combo/best_model.pt
```

**Ship criteria (combo should beat control on):**
1. `recall_single_at_max_context` (single-assoc acc @2048) ↑ — primary, unambiguous.
   Target ≥ 0.9 for "solved"; any large lift over control is the real signal.
2. `|p_content − p_filler|` ≥ 0.05 — the gate became selective (expect the signed
   value to go **negative** with sign=+1).
3. held-out PPL not regressed (pass `ARM_PPL`/`BASELINE_PPL` to the verdict, or read
   `val_ppl` from each arm's train log) — don't trade language modeling for recall.

### Step 3 — decide next (depends on Step 2 outcome)

**Done (2026-07-16):** Full battery ran on RTX PRO 6000 — Stage-2 A/B (5 arms × 300M),
Stage-3 hypersweep (13 arms × 150M), Stage-4 from-scratch (1B, routing levers), Stage-5
baselines (V11 vs pretrained Mamba-130m-hf; Transformer skipped). None shipped. See
[v11/EXPERIMENTS_V11.md](v11/EXPERIMENTS_V11.md) for tables and learnings.

**Next (Stage 6):**

- **Fix measurement** — unify `single_at_max` aggregation (verdict vs baselines summary
  disagree on same checkpoint); raise probe n per cell.
- **Fair baselines** — train ~100M Mamba + Transformer from scratch on identical
  fineweb+recall mix / 1B tokens (current Mamba comparison uses 300B-pretrained weights).
- **Architecture changes** (PAM-native, O(1)):
  1. E2 delta-rule write (`write_mode='delta'`) — fix compile hang; attacks interference.
  2. Vault state — one no-decay K-state, gate-protected; selective persistence without γ_floor PPL hit.
  3. Phase addressing — key-conditioned write phase for content-specific retrieval.
- **Capacity micro-tests** before large budgets (10M-param, 100% recall curriculum).
- **If Stage 6 beats matched baselines** → scratch retrain v3 round line; paper pass per below.

Original Step 3 bullets (for reference):

- **If combo clearly helps recall** → run the full ablation to attribute the gain:
  ```bash
  ARMS="control gate floor recall combo" TOKEN_BUDGET=150000000 \
  RESUME=checkpoints_v11_e3_k3_chat_pretrain_v2/best_model.pt NO_CURSOR=1 \
  ./scripts/run_v11_recall_ab.sh
  ```
  Then pick the winning lever set.
- **Then matched baselines** (`matched-baselines` todo): train ~100M Transformer and
  Mamba on the *same* tokenizer/data/budget and run `scripts/run_memory_behavioral.py`
  for all three (V11 winner, Transformer, Mamba). This is the "beats Mamba/Transformer
  at O(1)" claim gate.
- **If it beats baselines** → scratch retrain a v3 round line with the winning preset
  (`retrain-v3` todo), budget per `PUBLICATION.md`.
- **Paper** (`paper-pass1`/`paper-pass2` todos): replace the `\todoexp` placeholders in
  `memory_probes/paper/main.tex` §9 with measured Phase-B numbers, extend
  `scripts/plot_memory_probes_paper.py` for the behavioral/gate/arch-compare data.


---

## 5. Tuning knobs (env vars for `run_v11_recall_ab.sh`)

| var | default | meaning |
|---|---|---|
| `ARMS` | `control gate floor recall combo` | which arms to run |
| `TOKEN_BUDGET` | `300000000` | tokens/arm (use `150000000` for the warm screen) |
| `RESUME` | *(empty)* | warm-start ckpt; empty = from scratch |
| `NO_CURSOR` | `1` | warm-start without the slow doc-skip |
| `GSL` / `GST` / `GSSIGN` | `0.1` / `1.0` / `1.0` | gate-surprisal λ / τ / sign |
| `GFLOOR` | `0.98` | gamma floor |
| `RECALL_WEIGHT` / `WEB_WEIGHT` | `3` / `48` | blend weights (recall vs each web source) |
| `BATCH_SIZE` | `16` | raise on the 6000 for throughput |
| `DRY` | `0` | print commands only |

---

## 6. Caveats
- Gate sign: default `+1` (recall-optimal, drives `p_content−p_filler` negative). If a
  reviewer expects the probe's "protect content" convention, that's `sign=-1`; the two
  are opposite and only recall accuracy disambiguates.
- Warm-start `NO_CURSOR=1` re-reads some already-seen web docs — acceptable for a
  relative A/B; use `NO_CURSOR=0` (or from-scratch) for a publishable clean-data run.
- Grad-checkpointing + the gate aux stash: the A/B uses `--no_grad_ckpt`, so this is a
  non-issue for these runs.
