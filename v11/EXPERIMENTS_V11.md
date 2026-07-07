# V11: Maturing PAM via new memory dynamics

> **New to the architecture?** Start with [BEGINNER_GUIDE.md](BEGINNER_GUIDE.md) — a plain-language tour of complex phase space, magnitude/phase, PAM memory, and O(1) inference with code walkthroughs.

Goal: close the gap to the transformer (7d **26.88** vs transformer B=18 **22.69** on
WikiText-103, ~100M) by changing **how the PAM state writes and forgets** — the one
frontier V9/V10 flagged as unexplored — while keeping the architecture identity
(complex, phase-first, matrix-state, conjugate retrieval) and **O(1)/token inference**.

NOT a transformer, NOT a vector-state SSM (Mamba/Samba). Every lever below is a
single-variable change vs the V7 7d baseline (`v11_baseline`).

## Architecture (unchanged core)

```
ComplexEmbed -> [ pre-norm CGU (ModSwish) + pre-norm PAM ] x16 -> tied complex LM head
PAM:  S_t = gamma_t * S_{t-1} + V_t (x) K_t* ;  Y_t = S_t * Q_t   (conjugate match, no softmax)
train = chunked dual form O(T*C) ; infer = recurrent O(1)/token, no KV cache
```

Only `V11PAMLayer` ([v11/model.py](model.py)) changes, dispatching on 3 flags
(defaults reproduce 7d exactly):

| Flag | baseline (7d) | new |
|------|---------------|-----|
| `decay_mode` | `head` (scalar/head) | **`per_channel`** (E1) |
| `write_mode` | `additive` | **`delta`** (E2) |
| `n_states` | `1` | **`K>1`** (E3) |

## Correctness

For all 4 modes the **parallel training form == O(1) recurrent form** (max abs Δ < 1e-7,
fp64). See [v11/selftest.py](selftest.py):

```
.venv/bin/python -m v11.selftest
```

## Experiments

### E1 — Per-channel data-dependent decay
`gamma` becomes per-(head, key-channel) `[B,H,T,d]` instead of per-head `[B,H,T]`, so
different associative slots forget at different rates (the matrix state's effective rank
saturates ~10/64 today — capacity is wasted). Chunked training folds a chunk-local
cumulative decay into Q (×α) and K (×1/α) → plain conjugate score (GLA-style, fp32,
clamped). Inference: `S = S·diag(γ_t) + V K*` = O(d²)/token. V9's explicitly
recommended next change.

### E2 — Delta-rule (error-correcting) write
`S_t = γ_t S_{t-1}(I − β_t k_t k_t*) + β_t v_t k_t*`: before writing key `k_t`, erase
the stale value already associated with it (targets entity drift / repetition). Training
uses a per-chunk UT transform — solve `(I+M)U=W` with `M` strictly-lower (complex,
block-real solve), then read. Inference: rank-1 update O(d²)/token. `delta_chunk=64`.

### E3 — Multi-state superposed PAM (phase-routed)
K=2 matrix states per head with distinct decay biases; retrieval interferes them by a
data-dependent phase: `Y_t = Σ_k e^{iφ_k(x_t)} S_k Q_t`. Constructive/destructive phase
interference selects which state answers — the uniquely-QLLM idea (V7 "Exp 5"). Nearly
param-matched (only `phase_proj` + K decay biases added). Inference O(K·d²)/token.

## Protocol

Control = `v11_baseline` (== 7d, B=18, chunk 256). Each lever: tiny + 3-epoch smoke,
then full 10-epoch WikiText-103 B=18. **Accept** if it beats control val PPL at matched
B/steps AND does not regress generation (`rep3`,`uniq`). **Kill** if epoch-3 val does not
track below control (same discipline as 7f-3).

Runs are serialized on the single GPU via [scripts/run_v11_queue.sh](../scripts/run_v11_queue.sh)
(waits for the GPU to free, then runs E1 → E3). Throughput note: E2's chunk solve is the
slow path; Flash-PAM ([v11/triton_kernels.py](triton_kernels.py)) targets the speedup.

## PAM utilization diagnosis on the trained 100M (2026-06-30) — UNDER-UTILIZED

Question: is the 100M saturated, or is the PAM memory under-used? Measured on the trained
`checkpoints_v11_e3_k3_chat_pretrain/best_model.pt` (first time on a real checkpoint; all prior
probe numbers were `"checkpoint": null` untrained layers). Verdict: **memory-mechanism under-use**,
not parameter saturation -> work the 100M before scaling to 300M.

**State rank on real WikiText (effective rank of per-head S, d=64 max), `memory_probes --test rank-text`:**

| layer | WikiText rank/64 | Random rank/64 | uses memory? |
|------:|-----------------:|---------------:|--------------|
| 0  | 6.8  | 9.6  | no (below random) |
| 4  | **20.2** | 7.8  | yes (well above random) |
| 8  | 3.5  | 7.1  | no (below random) |
| 12 | 13.7 | 5.9  | yes |
| 15 | 4.6  | 12.4 | no (below random) |

Peak ~20/64 (~1/3 capacity); several layers use *less* rank on real text than on random noise.
Matrix memory is largely idle.

**GSP write-gate + decay, `scripts/v11_probe_gates.py` (2048 WikiText tokens):**

- Mean write-protect `p` across layers = **0.356** (init bias -3.0 => p~0.047), so training *did*
  move the gate (gradients flow) -> not a dead-gradient/optimization-only problem.
- **Mean `p_content - p_filler` = -0.013 (negative).** The gate is **content-blind**: it protects
  high-surprisal content tokens no more than low-surprisal filler. The mechanism that should let the
  model decide *when* to write vs protect never learned the distinction -> low effective rank.
- `dt_bias` stayed near init (~-3.4 to -3.8 vs base -4.0); `gamma` ~0.37-0.82 (moderate).

**Conclusion:** classify as **memory-mechanism under-use** (Step 3 branch), not optimization-bound.
Levers to try (short in-distro smokes, keep what moves PPL): make the write/protect gate
content-aware (init/regularization), encourage rank usage, revisit `base_dt_bias`/K. Probe scripts:
`scripts/v11_probe_gates.py`, `python -m memory_probes --test rank-text --checkpoint <ckpt> --layer N`.

### Fix implemented (2026-06-30): content-aware GSP write gate

Direct lever for the content-blind gate: `protect_gate` previously read only `cabs(x)` (per-channel
magnitude, dim), which discards phase/direction — so it literally cannot tell two equal-energy tokens
apart. New opt-in config `gate_content_aware=True` feeds the full `to_real_concat(x)` (2*dim, real+imag)
to the gate, restoring the information needed to protect content over filler. Also added a tunable
`protect_gate_bias` (default -3.0) and exposed `base_dt_bias`.

- Code: `v11/model.py` (`V11Config.gate_content_aware` / `protect_gate_bias`, gate build + line ~200
  `_gamma_and_vprime`), preset **`v11_e3_k3_chat_gate`**.
- CLI: `v11/train.py` `--gate_content_aware`, `--base_dt_bias`, `--protect_gate_bias`.
- Resume-safe: `_drop_shape_mismatches` reinits only the grown gate weight `(H,dim)->(H,2*dim)` and
  loads everything else from `best_model.pt` (verified on the 100M ckpt). Gate-reinit only happens on
  `--resume_from` (weights-only, fresh optimizer), so optimizer state never mismatches.

### WikiText gate A/B — DONE (2026-06-30) — **SHIP phase-aware gate**

Fair test: both arms **from scratch**, same recipe as `run_v11_exp.sh` (B=18, chunk 256, seq 2048,
3 epochs, WikiText-103, `--compile`). Only difference: GSP gate input (`cabs(x)` vs `to_real_concat(x)`).

| Epoch | Arm A magnitude gate | Arm B phase-aware gate | B − A |
|------:|---------------------:|-----------------------:|------:|
| 1 val PPL | 82.97 | 81.48 | **−1.49** |
| 2 val PPL | 50.82 | 49.76 | **−0.96** |
| 3 val PPL | 46.58 | 45.61 | **−0.97** |

**Final:** Arm A best val PPL **46.58** (acc 0.359) → Arm B **45.61** (acc 0.361). **−0.97 PPL (~2.1%)**.
B leads at every epoch; gap largest at epoch 1, then stable ~1 PPL (does not widen or close). Does not hurt.

**Gate probes (post-train best checkpoints):**

| Arm | mean `p` | `p_content − p_filler` |
|-----|----------:|------------------------:|
| A magnitude | 0.007 | +0.000 (content-blind) |
| B phase-aware | 0.111 | **+0.003** (slightly content-aware) |

**Checkpoints / logs:**

- Arm A: `checkpoints_v11_gateab_base/best_model.pt` —
  `logs/v11/v11_e3_k3_wikitext103_20260630_073704_63f5435_dirty/`
- Arm B: `checkpoints_v11_gateab_gate/best_model.pt` —
  `logs/v11/v11_e3_k3_wikitext103_20260630_135748_63f5435_dirty/`

**Incident:** first Arm B run completed epoch 1 (val PPL 81.57) then **crashed saving `best_model.pt`**
— disk 100% full (`best_model.pt.tmp` corrupt). Freed ~12 GB (obsolete ckpts + tmp); restarted from scratch.

**Decision:** adopt **`gate_content_aware=True`** as default on `v11_e3_k3` and `v11_e3_k3_chat` presets;
`run_v11_pretrain_scratch.sh` and `run_v11_saturate.sh` use it by default. Resume from old magnitude-only
checkpoints reinits only the gate weights (`_drop_shape_mismatches`). Use `--no_gate_content_aware` for ablation.

**Next:** saturation continued-pretrain from 10B base with phase-aware gate; re-probe rank on web-edu ckpt
(the original under-utilization diagnosis target).

### Competitive retrieval (Tier 1) — DONE (2026-07-07) — **KEEP CONTROL, competitive retrieval LOSES**

**Problem it targeted:** gate fix (`gate_content_aware`) shipped, but E3 **phase routing** still reads
`cabs(x)` only — magnitude-only state selection. Hypothesis (GPT5.5 + our diagnosis): PAM failures are
*blending* (wrong state mix), so content-aware routing + magnitude competition should raise selection quality.
**Result: false at this scale — every variant regressed vs the gate-only control.**

**Changes implemented (all flag-gated, defaults OFF — `v11_e3_k3_chat` unchanged):**

| flag | effect |
|------|--------|
| `routing_content_aware` | `phase_proj`/`score_proj` input `cabs(x)` → `to_real_concat(x)` (2×dim) |
| `state_compete` | `c_k = K·softmax(score)·e^{iφ_k}`; fused into `Dtilde` collapse (no K-loop regression) |
| `phase_init` | `'zero'` \| `'spread'` (biases 0,±2π/3) \| `'ortho'` |
| `route_balance_lambda` | MoE load-balance on batch-mean α (needs `state_compete`; wired through fused-CE path) |

- Code: [`v11/model.py`](model.py), CLI [`v11/train.py`](train.py), parity [`v11/selftest.py`](selftest.py)
- Probes: [`scripts/v11_probe_gates.py`](../scripts/v11_probe_gates.py) — α entropy, `|S_k·Q|` (loop path)
- Launcher: [`scripts/run_v11_routing_ab.sh`](../scripts/run_v11_routing_ab.sh)
- Ablation-only preset `v11_e3_k3_chat_compete` retained for reproducibility — **known loser, do NOT ship**.

**What each arm actually did in code (so this doesn't depend on arm names):**
- **control** — shipped E3 exactly: `phi = phase_proj(cabs(x))`, retrieval `Y = Σ_k e^{iφ_k} S_k Q` with every
  state weighted `|coef|=1`. No `score_proj`. (`_forward_multistate_fused` / `_forward_multistate`.)
- **routing_ca** — flips `_routing_input()` so `phase_proj` (and `score_proj` if present) reads
  `to_real_concat(x)` (real+imag, 2·dim) instead of `cabs(x)` (magnitude). `phase_proj` grows to `Linear(2·dim, H·K)`.
- **compete** — routing_ca **plus** `state_compete`: adds `score_proj`, `_phase_and_alpha()` returns
  `alpha = K·softmax(score)`, so the coefficient becomes `c_k = alpha_k·e^{iφ_k}` (magnitude competition on top
  of phase). `alpha` folds into the `Dtilde` collapse in `_fused_chunk_step` — no K-loop, exact-equiv to the loop path.
- **compete_balance** — compete **plus** `_route_balance_loss()`: aux term `−λ·H(batch-mean α)` (λ=0.01)
  summed over PAM layers, threaded through both the standard and fused-CE loss paths (`aux_loss`).
- **phase_spread** — compete **plus** `phase_init='spread'`: `_init_phase_proj()` seeds `phase_proj.bias`
  at 0, ±2π/3 per state (max initial phase separation) instead of zero-init.

**WikiText A/B** (same recipe as gate A/B): from scratch, B=18, seq 2048, chunk 256, 3 epochs,
`--compile`, `--gate_content_aware`. Arms are cumulative on top of the gate-only control:

| Arm | flags on top of gate | Ep1 | Ep2 | **Ep3 val PPL** | Delta vs control | acc | uniq |
|-----|----------------------|----:|----:|----------------:|-----------------:|----:|-----:|
| **control** | none (gate only) | 80.46 | 48.90 | **44.86** | — | 0.362 | 0.725 |
| routing_ca | +routing_content_aware | 86.44 | 52.71 | 48.05 | **+3.19** | 0.355 | 0.742 |
| compete | +routing_ca +state_compete | 84.19 | 50.52 | 46.21 | +1.35 | 0.359 | 0.674 |
| compete_balance | +compete +route_balance_lambda=0.01 | 84.88 | 50.72 | 46.38 | +1.52 | 0.358 | 0.723 |
| phase_spread | +compete +phase_init=spread | 84.56 | 51.28 | 47.05 | +2.19 | 0.358 | 0.789 |

**Findings:**
1. **Control wins at every epoch.** All four variants are strictly worse on val PPL (and mostly acc). None met
   the accept bar; all killed per protocol.
2. **`routing_content_aware` alone is the *worst* (+3.19).** The real+imag router input that helped the *write
   gate* (-0.97) does **not** transfer to *state selection* — it hurts most.
3. **`state_compete` partially offsets routing_ca** (48.05 -> 46.21) but never reaches control. Magnitude
   competition on top of the existing phase competition does not help selection at this scale.
4. `route_balance_lambda=0.01` and `phase_init=spread` add nothing positive — consistent with the repo's
   negative aux-loss history (7f-1 +0.82) and zero-init phase being adequate.
5. Control here (44.86) ~= June gate Arm B (45.61): baseline reproduced, so the regressions are real, not noise.

**Decision:** **DO NOT ship competitive retrieval.** Keep all four flags OFF (already the `V11Config` /
`v11_e3_k3` / `v11_e3_k3_chat` default — the winning config *is* the current default). **Do NOT** promote
`v11_e3_k3_chat_compete` to round-6b. The blending problem is **not** fixed by content-aware routing or
magnitude competition — do not re-run these four levers; revisit only with a fundamentally different mechanism.

**Accept:** beats control val PPL at epoch 3 **and** routing probes move (tok alpha entropy down, bar alpha diversity up,
state rank up, `|S_k·Q|` separates content/filler). **Kill** arms not tracking below control by epoch 3.
*(Bar not met by any arm — see decision above; retained for protocol record.)*

**Probe tooling (kept for future mechanisms, in `scripts/v11_probe_gates.py`):** per-token α entropy vs ln(K); batch-mean ᾱ diversity; state–content correlation;
state rank on real text; gate delta `p_content − p_filler` must not regress.

## Live run status (2026-06-30)

**Two parallel tracks** — different GPUs, no shared code edits between them.

### Track A — Text knowledge pretrain (RTX PRO 6000, server)

**Best chatable model to date:** **10B web-edu base + SmolTalk2 SFT (ChatML + warm-start)**
(`checkpoints_v11_sft_chat_smoltalk/best_model.pt`) — coherent instruction-following chat at ~100M
(non-transformer, non-Mamba, O(1)/token). The Tulu-3 SFT on the same base **regressed** (see readout
below); the older DCLM-2B + Alpaca SmolTalk2 run remains a useful comparison baseline.

- **DONE** — Phase 1 correctness (ChatML, EOT-in-loss, in-distro val/acc, stop-on-EOS).
- **INCIDENT (2026-06-23)** — Phase 2 pretrain reached **~6.78B / 10B tokens** (~3 days) then
  **session shutdown** killed tmux with **no checkpoint on disk** (streaming runs stayed on Epoch 1;
  old code only saved at epoch end). Work lost; restart from scratch required.
- **FIXED** — Periodic checkpoints: `--save_every_steps 5000` (~184M tokens, ~2h) writes
  `latest.pt` for resume only (no mid-run val); SIGTERM/SIGINT saves `latest.pt`
  at next batch boundary. `best_model.pt` / `final_model.pt` only at run end (epoch-end val).
  Resume: `RESUME=checkpoints_v11_e3_k3_chat_pretrain/latest.pt`.
- **DONE (2026-06-27)** — Phase 2 pretrain finished the full 10B budget →
  `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt` (WikiText val PPL **65.46**).
- **REGRESSION (2026-06-28)** — Phase 3 Tulu-3 SFT on that base rambles/hallucinates. Root cause:
  the 10B base is **no better than the old 2B base** (PPL 65.46 vs 66.27), ChatML special tokens
  were never pretrained, and Tulu-3 is too broad. Full diagnosis + reproduction:
  [Phase 3 Tulu-3 SFT readout](#phase-3-tulu-3-sft--result--readout-2026-06-28--regression-do-not-ship).
- **INCIDENT (2026-06-28)** — First SmolTalk recovery attempt OOM'd at epoch start: launched while
  Tulu-3 SFT still held ~76 GB on the GPU (`Process 1334258`). Warm-start ran; no checkpoint saved
  (`checkpoints_v11_sft_chat_smoltalk/` empty). Log:
  `logs/v11/v11_e3_k3_chat_sft_smoltalk2_chat_20260628_031225_6ccd8fb/`.
- **DONE (2026-06-29)** — SmolTalk re-SFT on 10B base with `--warmstart_chatml` completed →
  `checkpoints_v11_sft_chat_smoltalk/best_model.pt`. Recovery **succeeded** (see
  [Phase 3 SmolTalk recovery readout](#phase-3-smoltalk-recovery--result--readout-2026-06-29--ship-this)).
- **DONE (2026-06-30)** — WikiText gate A/B: phase-aware GSP gate **wins** (−0.97 val PPL vs magnitude-only);
  **shipped as default** on `v11_e3_k3` / `v11_e3_k3_chat` presets. See
  [WikiText gate A/B](#wikitext-gate-ab--done-2026-06-30--ship-phase-aware-gate).
- **DEFERRED** — param scaling (300–500M) until token-scaling ROI confirmed at 100M.

PAM math for this track: unchanged — see [Architecture](#architecture-unchanged-core) above
(`S_t = γ_t S_{t-1} + V_t ⊗ K_t*`, E3 K=3 retrieval).

### Track B — Full-duplex audio POC (RTX 4090, local, **additive**)

Runs **while Track A trains**. Goal: prove V11 PAM can learn **listen / speak / backchannel**
from streaming audio embeddings (SALMONN-omni style) — **not** speech-to-speech.

| Stage | Status | Result |
|-------|--------|--------|
| 0 synthetic | **done** | val think_acc **100%** — `checkpoints_v11_duplex_5m_stage0/` |
| 1 LibriSpeech + Whisper | **done** | val think_acc **100%** — `checkpoints_v11_duplex_5m_stage1/` |
| 1 Kathbath hi/gu | **done** | val think_acc **100%**, 232 s — `checkpoints_v11_duplex_5m_stage1_hi_gu/` |
| 2 streaming infer + TTS | pending | 160 ms block loop, optional external synth on `<speak>` |
| post–10B distill | pending | merge chat base into `duplex_25m` after Track A completes |

**Whisper role:** frozen **encoder only** (`openai/whisper-small`) → 768-d vectors →
`ComplexLinear` → injected at `<env_mark>` slots. No Whisper decoder; no vocoder.

**Duplex block math** (per ~1 s chunk in POC):

```text
sequence:  <env_mark>  [audio_embed × N]  →  predict  listen | speak | backchannel
loss:      CE on thinking tokens (3-way; random baseline 33.3%)
backbone:  same PAM E3 K=3 as v11_e3_k3 (O(1)/token state, no KV cache)
```

Full write-up: [v11/duplex/EXPERIMENTS_DUPLEX.md](duplex/EXPERIMENTS_DUPLEX.md).
Launch: `./scripts/run_v11_duplex_stage1.sh`, demo `./scripts/run_v11_duplex_gradio.sh`.

## Phase 1 — correctness layer (2026-06-20)

Root-caused why the first chat SFT rambled and stated wrong facts, and fixed the
data/eval/inference bugs (architecture unchanged):

**Bugs fixed**
- **EOT never learned:** end token was `<|endoftext|>` == `pad_id`; the old mask stripped it,
  so the model never learned to stop. → New **ChatML** tokenizer adds `<|im_start|>`/`<|im_end|>`
  (vocab **50259**); `<|im_end|>` is the learned end-of-turn, **distinct** from pad `<|endoftext|>`
  (50256), and is now **included in the loss**.
- **Multi-turn loss leak:** the old single `assistant_start` let later **user** turns fall inside
  the loss. → `_encode_sft_example` now assembles tokens turn-by-turn and marks **only** assistant
  content + its closing `<|im_end|>` as targets (verified: user text never enters the mask).
- **No stop-on-EOS:** `V11LM.generate` ran to `max_new_tokens` regardless. → added `eos_token_id`;
  it breaks (per-sequence `finished` mask) when `<|im_end|>` is emitted.
- **Off-distribution val:** post-SFT WikiText PPL (263/324) measured the wrong distribution and is
  single-token noisy. → `V7Trainer.validate()` now reports **assistant-only masked PPL** on a
  deterministic **in-distribution holdout** (~2% hash bucket) **plus next-token accuracy**.

**New ChatML template (vocab 50259):**
```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

**Code:** [`v7/data.py`](../v7/data.py) `get_chat_tokenizer`, `format_chat_prompt`,
`format_chat_messages`, `_encode_sft_example`, `load_chat_sft` (generic) + `load_smoltalk2` wrapper;
[`v7/train.py`](../v7/train.py) `validate()` (masked PPL + accuracy);
[`v11/model.py`](model.py) `generate(eos_token_id=...)`;
[`v11/train.py`](train.py) vocab auto-match + `_resize_embeddings_for_vocab` (continue-from-old-base path);
[`scripts/chat_v11.py`](../scripts/chat_v11.py), [`scripts/smoke_chat_v11.py`](../scripts/smoke_chat_v11.py) (ChatML + `<|im_end|>` stop).
Cache key bumped via `_CHAT_CACHE_VERSION`.

## Phase 2 — knowledge pretrain (from scratch, better base)

Rationale: 100M @ 2B tokens is far under-trained (SmolLM-135M ≈ 600B). More tokens at fixed size
is the cheapest knowledge lever and gives a clean scaling curve. Base is **from scratch** with the
chat vocab baked in (preset `v11_e3_k3_chat`, vocab 50259) so special-token embeddings exist at init
and there is no resize hack in the production path.

- **Corpus:** DCLM-Edu + FineWeb-Edu mix (streamed, weighted round-robin), edu score ≥ 3.
- **Budget:** ~10B tokens to start (extensible), cosine LR (peak 3e-4) with warmup over the full budget.
- **Code:** [`v7/data.py`](../v7/data.py) `_dclm_edu_text_iter`, `_fineweb_edu_text_iter`,
  `_interleave_text_iters`, `load_pretrain_mix`, `load_fineweb_edu`; preset `v11_e3_k3_chat`.
- **Launch (tmux, resumable):**
```bash
tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_scratch.sh 10000000000'
# resume after crash / shutdown (restores optimizer+scheduler+step+tokens):
RESUME=checkpoints_v11_e3_k3_chat_pretrain/latest.pt \
  tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_scratch.sh'
```
- **Checkpoint protocol:** `latest.pt` every 5000 steps (~2h) for resume only — no validation
  or `best_model.pt` mid-run (batch loss is too noisy). At run end (token budget or last epoch),
  epoch-end WikiText val writes `best_model.pt` and `final_model.pt`. Atomic write (`.tmp` → rename).
  Streaming data **restarts from the beginning** on resume (token budget still counts from saved
  `global_tokens`; tail may re-see docs).
- **Metric:** WikiText val PPL as the tracking anchor (cross-run comparable); target a clean,
  decreasing curve across the larger budget at fixed 100M.

## Phase 3 — full Tulu-3 SFT on the new base

- **Data:** `allenai/tulu-3-sft-mixture` filtered to knowledge_recall + coding + general (heuristic
  source map; math/safety/multilingual excluded). `load_tulu3_sft` reuses the generic `load_chat_sft`.
- **Run:** `--dataset tulu3`, resume the **new pretrain base**, ~2 epochs, lr 5e-5 →
  `checkpoints_v11_sft_chat`. Script [`scripts/run_v11_sft_tulu3.sh`](../scripts/run_v11_sft_tulu3.sh).
- **Acceptance:** beats current DCLM-base SFT on in-distro masked PPL/accuracy and smoke quality
  (clean self-stops + better facts).

### Phase 3 Tulu-3 SFT — result & readout (2026-06-28) — REGRESSION, do not ship

The 10B-base + ChatML + Tulu-3 chat model **rambles and hallucinates** (fluent but wrong:
"what is 2+2" → "the answer is 3"; "my name is gowrav" → unrelated essay on objects). The
end-of-epoch sample even emitted LaTeX + a line of Cyrillic. **Do not ship** — use
[SmolTalk recovery](#phase-3-smoltalk-recovery--result--readout-2026-06-29--ship-this) instead.

**Exact run (reproduce):**
- Commit `7d54830`. Launch: `tmux new-session -d -s v11_sft './scripts/run_v11_sft_tulu3.sh checkpoints_v11_e3_k3_chat_pretrain/best_model.pt'`
- Args: `--preset v11_e3_k3_chat --stage sft --dataset tulu3 --seq_len 2048 --batch_size 18 --epochs 2 --chunk_size 256 --lr 5e-5 --warmup_steps 200 --sft_filter hard` (full string in RUN_INFO).
- Base ckpt: `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt` (10B from-scratch web-edu).
- SFT log: `logs/v11/v11_e3_k3_chat_sft_tulu3_20260627_180442_7d54830/` (+ `RUN_INFO.txt`).
- Pretrain log: `logs/v11/v11_e3_k3_chat_pretrain_scratch_b10000000000_20260623_085523_05f04a0/`.
- Output ckpt: `checkpoints_v11_sft_chat/best_model.pt`. Chat: `scripts/chat_v11.py --checkpoint checkpoints_v11_sft_chat/best_model.pt --preset v11_e3_k3_chat`.
- SFT metrics (full 2 epochs): epoch-1 train loss **2.64** (PPL 13.97), val **2.49** (PPL 12.05, acc 0.518);
  epoch-2 train **2.50** (PPL 12.14), val **2.44** (PPL 11.53, acc 0.527). End sample: LaTeX + Cyrillic.
  `best_model.pt` = epoch 2 (val improved).

**Base comparison — the new pretrain bought ~nothing:**

| base ckpt | corpus | tokens | WikiText val PPL | acc |
|-----------|--------|-------:|-----------------:|----:|
| `checkpoints_v11_e3_k3_dclm/best_model.pt` | DCLM-Edu | 2B | **66.27** | — |
| `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt` | DCLM+FineWeb-Edu | 10B | **65.46** | 0.309 |

5x tokens / ~105h compute moved WikiText PPL ~1 point; pretrain train-loss flatlined at ~3.5
(PPL ~33-38) for the whole run. **The novel PAM base saturates around PPL ~65 at 100M** —
shallow knowledge → SFT can only teach format, not facts.

**Root causes (3 changes stacked vs the good recipe):**
1. **Base didn't improve** (table above) → hallucinated facts regardless of SFT data.
2. **ChatML special tokens never pretrained.** Preset `v11_e3_k3_chat` (vocab 50259) adds
   `<|im_start|>`/`<|im_end|>`, but pretrain ran on raw text where they never appear, so those 2
   tied embedding rows stayed at random `std=0.02` init (no gradient). They define turn structure +
   the stop token. The good run used the Alpaca `### User:` template (ordinary tokens) → no such gap.
   `_resize_embeddings_for_vocab` ([v11/train.py](train.py)) only fixes a *smaller*-vocab base, not
   this already-50259-but-untrained case.
3. **Tulu-3 is too broad for a weak 100M base** — math/LaTeX (MATH/GSM) + multilingual (Aya →
   Cyrillic). SmolTalk2 (the good run) is clean English chat. SFT train loss rose to 2.64 vs ~2.1.

**Remediation (implemented 2026-06-28, completed 2026-06-29):** re-SFT the 10B base on **SmolTalk2**
with `--warmstart_chatml` (seed ChatML rows 50257/50258 from mean of trained GPT-2 rows). Code:
[`v11/train.py`](train.py) `--warmstart_chatml`; launch script:
[`scripts/run_v11_sft_smoltalk.sh`](../scripts/run_v11_sft_smoltalk.sh) (defaults hardened 2026-06-29:
`v11_e3_k3_chat` preset, 10B base, `--warmstart_chatml`, `--save_every_steps 5000`).

```bash
# RTX box (tmux, ~9h for 1 epoch; GPU must be free — do not overlap with other v11.train):
tmux new-session -d -s v11_smoltalk './scripts/run_v11_sft_smoltalk.sh'
# Output: checkpoints_v11_sft_chat_smoltalk/best_model.pt
# Verify:
uv run python scripts/smoke_chat_v11.py \
  --checkpoint checkpoints_v11_sft_chat_smoltalk/best_model.pt \
  --preset v11_e3_k3_chat --temperature 0.7
```

Deeper item (separate): why the PAM base saturates at PPL ~65 despite 5x tokens.

### Phase 3 SmolTalk recovery — result & readout (2026-06-29) — SHIP THIS

Re-SFT of the 10B web-edu base on **SmolTalk2** with ChatML + `--warmstart_chatml` **recovered
coherent chat**. Turn structure and stop-on-`<|im_end|>` work; factual depth is still
limited by the shallow base (same PPL ~65 ceiling as pretrain).

**Exact run (reproduce):**
- Launch: `tmux new-session -d -s v11_smoltalk './scripts/run_v11_sft_smoltalk.sh'`
- Args: `--preset v11_e3_k3_chat --stage sft --dataset smoltalk2 --seq_len 2048 --batch_size 18
  --epochs 1 --chunk_size 256 --lr 5e-5 --sft_filter hard --warmstart_chatml
  --save_every_steps 5000 --resume_from checkpoints_v11_e3_k3_chat_pretrain/best_model.pt`
  (full string in RUN_INFO).
- Base ckpt: `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt` (10B DCLM+FineWeb-Edu).
- SFT log: `logs/v11/v11_e3_k3_chat_sft_smoltalk2_20260629_053649_6ccd8fb_dirty/` (+ `RUN_INFO.txt`).
- Output ckpt: `checkpoints_v11_sft_chat_smoltalk/best_model.pt` (+ `final_model.pt`, periodic `latest.pt`).
- Wall time: **9.04 h** (32538 s). Cache hit: `.cache/v7_tokens/smoltalk2_hard_full_sl2048_chatv3.pt`.

**SFT metrics (1 epoch, in-distribution holdout):**

| metric | SmolTalk recovery | Tulu-3 (broken) | DCLM-2B SmolTalk (Alpaca) |
|--------|------------------:|----------------:|--------------------------:|
| Train loss | **2.13** (PPL 8.39) | 2.50 (PPL 12.14) | 2.10 (PPL 8.21) |
| Val loss | **1.98** (PPL **7.22**, acc **0.567**) | 2.44 (PPL 11.53, acc 0.527) | — (WikiText val only) |
| Template | ChatML (50259) | ChatML (50259) | Alpaca `### User:` (50257) |
| Warm-start ChatML | **yes** | no | N/A (no extra tokens) |

**End-of-epoch generation sample** ("capital of France"):
- Recovery: *"The capital of France is Paris."* + coherent follow-on (Eiffel Tower, Louvre).
- Tulu-3: LaTeX (`\sqrt{2}`) then multilingual garbage.

**Smoke probes** (`scripts/smoke_chat_v11.py`, temp 0.7, `best_model.pt`):

| prompt | result | notes |
|--------|--------|-------|
| Capital of France | **"The capital of France is Paris."** | `stopped_on_im_end=True`, 8 tokens |
| Photosynthesis | Describes chloroplasts/chlorophyll/CO2 | rambles after 1 sentence (160 tok cap) |
| Python add two numbers | `def add_two_numbers(a, b): return a + b + c` | correct shape, wrong `+ c`, bad examples |
| what is 2+2 | Does not answer 4; math drift | base knowledge limit |
| why are leaves green | On-topic but vague/wrong botany | shallow base |
| my name is gowrav | Confabulates ("Jack Harris…") | no real memory |

**What fixed it vs Tulu-3:** (1) SmolTalk2 clean English chat vs math/multilingual mix;
(2) `--warmstart_chatml` gave rows 50257/50258 a sane init vs random `std=0.02`;
(3) train loss ~2.1 (proven recipe) vs ~2.5; (4) run alone on GPU (no OOM contention).

**ChatML token note:** pretrain used vocab 50259 but **never saw** `<|im_start|>`/`<|im_end|>`
in raw text — only SFT teaches them. Warm-start is a **recovery heuristic** (mean of base rows),
not semantic pretraining. Proper fix for future pretrains: inject ChatML structure into pretrain
corpus so those rows get gradients.

**Compare broken ckpt:** `checkpoints_v11_sft_chat/best_model.pt` (Tulu-3) preserved.

### Prior best chatable model — SmolTalk2 SFT on DCLM-2B base (Alpaca template)

This is the **prior** coherent chat baseline (Alpaca template, no ChatML). Superseded for
production chat tooling by the 10B + SmolTalk recovery above, but still useful as a comparison
point (50257 vocab, no special-token gap).

- **Base:** `checkpoints_v11_e3_k3_dclm/best_model.pt` (DCLM-Edu, 2B tokens, preset `v11_e3_k3`, vocab 50257).
- **SFT:** SmolTalk2 (`--sft_filter hard`), 1 epoch, lr 5e-5 → **`checkpoints_v11_sft_dclm/best_model.pt`**.
- **Template:** Alpaca `### User:` / `### Assistant:` (plain GPT-2 tokens, no ChatML specials).
- **Commit:** `4aa8e74`. **Log:** `logs/v11/v11_e3_k3_sft_smoltalk2_20260619_153732_4aa8e74/` (+ `RUN_INFO.txt`).
- **Reproduce:**
```bash
tmux new-session -d -s v11_sft \
  './scripts/run_v11_sft_smoltalk.sh checkpoints_v11_e3_k3_dclm/best_model.pt'
```
- **Quality:** SFT train loss **2.10** (PPL 8.21); sample "capital of France" → *"The capital of France is Paris..."* (coherent, on-topic). WikiText val PPL is off-distribution here (263) and not the success metric.

## Phase C — Richer data + chat SFT (implemented)

**Strategy:** pretrain base on web corpus first, then SFT for chat (SmolLM2 recipe).

| Stage | Dataset | HF ID | Loss | Init | Pilot budget |
|-------|---------|-------|------|------|--------------|
| Pretrain | DCLM-Edu (score ≥ 3) | `HuggingFaceTB/dclm-edu` | full CE | `v11_e3_k3` WikiText ckpt | **2B tokens** (~2 GPU-days) |
| SFT | SmolTalk2 filtered | `HuggingFaceTB/smol-smoltalk` | assistant-only CE | pretrain ckpt | 1 epoch |

**Eval anchor:** WikiText-103 validation PPL throughout (apples-to-apples vs 25.77 baseline).

**Chat template (GPT-2, no new vocab):**
```
### System:
{system}
### User:
{user}
### Assistant:
{assistant}<|endoftext|>
```

**SmolTalk2 filter (`--sft_filter hard`):** drop function/tool samples, non-short
MagPie-Ultra, conversations >8 turns or >6×seq_len chars.

**Code:**
- [`v7/data.py`](../v7/data.py): `load_dclm_edu`, `load_smoltalk2`, `StreamingTokenChunkDataset`, `MaskedTextDataset`
- [`v7/train.py`](../v7/train.py): `loss_mask` + `token_budget` in `V7Trainer`
- [`v11/train.py`](train.py): `--stage pretrain|sft`, `--token_budget`, `--resume_from`
- [`scripts/run_v11_pretrain_dclm.sh`](../scripts/run_v11_pretrain_dclm.sh)
- [`scripts/run_v11_sft_smoltalk.sh`](../scripts/run_v11_sft_smoltalk.sh)
- [`scripts/chat_v11.py`](../scripts/chat_v11.py)
- [`v11/eval_checkpoints.py`](eval_checkpoints.py) + [`scripts/run_v11_eval_checkpoints.sh`](../scripts/run_v11_eval_checkpoints.sh)
- [`scripts/smoke_chat_v11.py`](../scripts/smoke_chat_v11.py) + [`scripts/run_v11_sft_ab.sh`](../scripts/run_v11_sft_ab.sh)

**Launch:**
```bash
tmux new-session -d -s v11_pretrain './scripts/run_v11_pretrain_dclm.sh 2000000000'
# after pretrain:
tmux new-session -d -s v11_sft './scripts/run_v11_sft_smoltalk.sh checkpoints_v11_e3_k3_dclm/best_model.pt'
```

**Acceptance (pilot) — revised after run:**

| Metric | Pretrain (2B) | SFT |
|--------|-----------------|-----|
| WikiText val PPL | beat or match **25.77** | may rise slightly; track separately |
| **DCLM holdout val PPL** | **lower than WikiText ckpt** | not primary |
| Generation | more coherent prose | follows user/assistant format |
| Chat smoke | N/A | `scripts/chat_v11.py --checkpoint ...` |

WikiText-only acceptance was **too optimistic** for pure DCLM continuation (see readout below).

### Phase C pretrain pilot — result & readout (2026-06-19)

**Run:** `v11_e3_k3`, resume `checkpoints_v11_e3_k3/best_model.pt`, DCLM-Edu stream
(`edu_int_score>=3`), **2B tokens**, lr=1e-4, WikiText val anchor only.

| Metric | Wiki ckpt (before) | After 2B DCLM |
|--------|-------------------:|--------------:|
| WikiText val PPL | **25.77** | **66.27** |
| Train loss (DCLM) | — | 3.75 (PPL **42.4**) |
| Wall / tokens | — | 21.1h / 2.00B |

Log: `logs/v11/v11_e3_k3_pretrain_dclm_b2000000000_20260618_055525_22d1afe_dirty/`  
Ckpt: `checkpoints_v11_e3_k3_dclm/best_model.pt`

**What happened:** model **learned DCLM** (train loss 7.1→3.75) but **forgot WikiText**
(catastrophic forgetting / domain shift). Pure switch from WikiText-tuned weights to
2B tokens of web text at full lr, no WikiText mixing — WikiText val was the wrong
**sole** success metric.

**Learnings (do not repeat without fix):**

1. **WikiText val tracks forgetting**, not DCLM quality, when training only on DCLM.
2. **Need DCLM holdout val** on the same checkpoint comparison (wiki ckpt vs dclm ckpt).
3. **Next pretrain retry** (if any): lower lr (1e-5), and/or 10–20% WikiText mix, and/or
   early-stop on WikiText — not another blind 2B pure stream.
4. **SFT still worth A/B** — chat may prefer DCLM base even if WikiText PPL tanked;
   validate before discarding dclm ckpt.
5. **Best WikiText base remains** `checkpoints_v11_e3_k3/best_model.pt` until proven otherwise.

**Validation ladder (current — run before SFT):**

```bash
./scripts/run_v11_eval_checkpoints.sh   # Wiki val + DCLM 5% hash holdout, both ckpts
```

Record results in table below, then proceed to SFT A/B only if readout is clear.

| checkpoint | WikiText val PPL | DCLM holdout PPL | notes |
|------------|----------------:|-----------------:|-------|
| `checkpoints_v11_e3_k3/best_model.pt` | **25.77** | **1222.11** | WikiText-trained; DCLM-naive |
| `checkpoints_v11_e3_k3_dclm/best_model.pt` | **66.26** | **33.86** | 2B DCLM; strong on holdout, WikiText forgotten |

**Validation readout (2026-06-19):** [`scripts/run_v11_eval_checkpoints.sh`](../scripts/run_v11_eval_checkpoints.sh)
completed. DCLM holdout confirms pretrain **worked** (33.86 vs 1222 baseline) while WikiText
regression (66.26) is real but expected. **Do not discard DCLM ckpt on WikiText alone.**
Proceed to SFT A/B: chat quality may favor DCLM base despite WikiText PPL.

**SFT A/B (running 2026-06-19):**

| Run | Base | Output ckpt | tmux |
|-----|------|-------------|------|
| A (control) | `checkpoints_v11_e3_k3/best_model.pt` | `checkpoints_v11_sft_wiki/best_model.pt` | `v11_sft_ab` |
| B (pilot) | `checkpoints_v11_e3_k3_dclm/best_model.pt` | `checkpoints_v11_sft_dclm/best_model.pt` | same session, after A |

Launch: `tmux attach -t v11_sft_ab` · log: `logs/v11/sft_ab_run_20260619.log`

**Pre-SFT smoke (2026-06-19):** [`scripts/smoke_chat_v11.py`](../scripts/smoke_chat_v11.py) on 3 fixed prompts.
Neither base follows `### User/Assistant` — wiki base = WikiText-style prose drift; dclm base = question spam / code gibberish.
SFT is required before chat is meaningful; compare **post-SFT** smokes only.

| base | capital of France | photosynthesis | python add |
|------|-------------------|----------------|------------|
| wiki | Wiki prose drift (French Resistance…) | DB/object gibberish | ICAO/research gibberish |
| dclm | question list spam | rhetorical Q spam | `#include` C spam |

Post-SFT smoke: auto at end of [`scripts/run_v11_sft_ab.sh`](../scripts/run_v11_sft_ab.sh) → `logs/v11/sft_ab_pre_smoke_20260619_post.log`

**Deferred (Phase C+):** sequence packing, FineWeb-Edu mix, TuluTalk, DPO, tokenizer migration.

### Next (Phase C+)
1. Complete validation ladder → update table above.
2. SFT A/B + chat smoke → pick base for any retry pretrain.
3. If DCLM path wins chat: retry pretrain with lower lr / wiki mix (not pure 2B overwrite).
4. Scale params only after data path is validated.

## Flash-PAM design constraint (from V7–V10 experience)

Custom-autograd **Triton kernels fight `torch.compile`** on this split-real complex
stack — recorded evidence:
- `v7/triton_kernels.py` guards every fused op with `not torch.compiler.is_compiling()`,
  i.e. under compiled training the Triton path is **bypassed** (PyTorch fallback runs).
- `v7/EXPERIMENTS_V7.md`: removing Triton (+ CGU) gave **+66% tok/s** — "no Triton
  compilation, simpler graph for `torch.compile`".
- `v10/base_to_start.md`: "Triton kernel compilation + complex compile graph" named as
  overhead; `v8/backbone/model.py`: opaque kernels cause **graph breaks** that block fusion.

**Therefore Flash-PAM is two-track:**
1. **Track 1 (primary, training):** pure-PyTorch **chunk-parallel** reformulation —
   batch independent intra-chunk GEMMs, keep only the cheap d×d state-carry sequential.
   No custom autograd → inductor fuses it into the compiled graph. Validated numerically
   (fwd+grad) vs `_forward_chunked_head`.
2. **Track 2 (secondary, eager + O(1) inference only):** tiled Triton kernel behind the
   same `is_compiling()` guard, so it never fragments the compiled training graph.

Benchmarks run in **spare VRAM** (small B/dims) alongside the live baseline to decide
whether the speedup justifies enabling it.

### Track 1 result — evaluated, NOT adopted (no speedup)

Built `v11/flash_pam.py` (`flash_pam_chunked_head`) + harness `v11/bench_flash_pam.py`.
Correctness: fp64 forward matches `_forward_chunked_head` to **2.5e-13**, grads ~1e-7
(fp32/triton diffs are pure float noise) — math is identical.

Speed (single PAM layer, fp32, T=2048, C=256, RTX PRO 6000), `torch.compile` fwd:

| B  | baseline | flash | verdict |
|----|---------:|------:|---------|
| 2  | 2.31 ms  | 3.21 ms | slower |
| 8  | 9.60 ms  | 13.02 ms | slower |
| 16 | 24.62 ms | 26.05 ms | slower (+ ~2× peak mem) |

**Why:** once inductor compiles the baseline, the n=8 chunk loop is already cheap and the
per-chunk GEMMs at `B*H≈288` batch already saturate the GPU. Folding chunks into the
batch (`B*H*n`) adds no utilisation but ~2× the D/score memory traffic → net slower.
Confirms the V7 lesson: the **compiled pure-PyTorch chunked dual form is already near
the efficient frontier**; the tok/s gap vs transformers is FLOP/bandwidth-fundamental
(PAM's matrix state), not an un-fused-kernel problem.

**Decision:** keep the existing `_forward_chunked_head`. Do NOT ship Track 1. Skip the
Track-2 Triton kernel for training (would be bypassed under `is_compiling()` anyway).
If **E2/delta** proves promising-but-slow, optimise its specific bottleneck (the per-chunk
complex triangular **solve**), not this generic reformulation.

## Results (fill from logs)

| Run | preset | val@10 | tok/s | params | verdict |
|-----|--------|-------:|------:|-------:|---------|
| baseline (anchor) | `v11_baseline` (7d) | **26.88** | ~48k | 100.4M | control |
| E1 per-channel decay | `v11_e1_perchannel` | **26.87** | ~58k | 105.1M | **tie** (no quality gain, ~20% faster) |
| E2 delta write | `v11_e2_delta` | — | — | 100.4M | **inconclusive** — hung after compile (linalg.solve × torch.compile) |
| E3 multistate K=2 | `v11_e3_multistate` | **26.01** | ~36k | 100.5M | **WIN** (−0.87 vs control) |
| E3 multistate K=3 | `v11_e3_k3` | **25.77** | ~29k | 100.5M | **BEST** (−1.11 vs control, −0.24 vs K=2; ahead every epoch except ep2 tie) |
| E1+E3 combo | `v11_e1e3_combo` | stopped @ep4 | ~47k | 105.1M | **per_channel does NOT stack** — behind E3 every epoch (see below) |

Anchors (same pipeline): transformer B=18 **22.69**, V7 7d B=18 ModSwish **26.88**.

### Read-out (2026-06-17, Phase B complete)
- **E3 K=3 is the new best PAM config**: 26.88 → **25.77** (−1.11 vs 7d control).
  K=3 beat K=2 at **every epoch except ep2 (tie)**; final gain **−0.24 PPL** for ~20%
  throughput cost (29k vs 36k tok/s). Generation ep10: `rep3=0.030`, `uniq=0.670` —
  on par with K=2 and 7d. **Lock `v11_e3_k3` for Phase C scaling.**
- **E3 K=2 remains the efficiency pick**: 26.01 @ ~36k tok/s — use if inference/training
  cost matters more than the extra 0.24 PPL from K=3.
- **E1 (per-channel decay) is a tie** (26.87 ≈ 26.88) but **faster** (~58k vs ~48k tok/s);
  quality-neutral; does **not** combine with E3 (combo behind E3 every epoch).
- **E2 (delta) hung** under `torch.compile` (the per-chunk complex triangular solve), and
  the `--no-compile` fallback is impractical: probe OOM'd at B=18 (88 GB) and ran at only
  **~5.9k tok/s** → **Killed and deferred.** Needs compile-friendly back-substitution.

Per-epoch K-sweep (val PPL):

| epoch | 7d control | E3 K=2 | E3 K=3 | K=3 vs K=2 |
|------:|-----------:|-------:|-------:|------------:|
| 1 | — | 82.72 | 81.54 | −1.18 |
| 2 | — | 45.59 | 45.59 | tie |
| 3 | — | 35.86 | 35.75 | −0.11 |
| 4 | — | 31.82 | 31.47 | −0.35 |
| 5 | — | 29.33 | 29.07 | −0.26 |
| 6 | — | 27.85 | 27.65 | −0.20 |
| 7 | — | 26.83 | 26.57 | −0.26 |
| 8 | — | 26.29 | 26.05 | −0.24 |
| 9 | — | 26.05 | 25.80 | −0.25 |
| 10 | **26.88** | **26.01** | **25.77** | **−0.24** |

K=3 scaling verdict: **modest but monotonic** — worth adopting for quality-first runs;
diminishing returns vs K=2 suggest **K=4 is low priority** unless Phase C data unlocks it.

### Phase B finding — E1+E3 combo: per_channel does NOT stack (stopped @ epoch 4)
Per-epoch val PPL (combo tracked between E1 and E3, **behind E3 at every epoch**):

| epoch | E3 K=2 | E1 pc | combo |
|------:|-------:|------:|------:|
| 1 | 82.72 | 84.97 | 85.59 |
| 2 | 45.59 | 47.74 | 47.03 |
| 3 | 35.86 | 37.19 | 36.68 |
| 4 | 31.82 | 32.83 | 32.34 |

Gap to E3 narrowed (2.87→0.52) but combo was never ahead; best realistic outcome was a
*tie with E3 at higher cost* (105.1M params + per_channel overhead). Confirms E1's verdict:
per-channel decay is quality-neutral here. **Stopped to free the GPU for the K-sweep.**
Decision: **drop per_channel**; keep the plain head-decay E3 line as the winner to push.

### Cross-version learnings (V6 → V7a → V7d → V11)

WikiText-103, ~100M, seq_len=2048, B=18 where noted. Transformer B=18 anchor: **22.69**.

| Milestone | val@10 | Δ vs prior | tok/s | What changed |
|-----------|-------:|-----------:|------:|--------------|
| V6 `medium-pam-v3` | **29.95** | — | ~96k* | Interleaved CGU+PAM, RoPE, fused QKV; proved PAM works at scale |
| V7 **7a** ModSwish (B=3) | **29.73** | −0.22 | ~21k | ModSwish CGU; V7 refactor parity with V6 |
| V7 **7d** chunked B=6 | **27.94** | −1.79 | ~32k | Chunked dual form O(T·C); big PPL + speed win |
| V7 **7d** chunked B=18 | **26.88** | −1.06 | ~22k | Larger batch; beats transformer B=3 (27.08) |
| V11 E3 K=2 | **26.01** | −0.87 | ~36k | Multi-state phase interference (K=2); first in-family memory win |
| **V11 E3 K=3** | **25.77** | −0.24 | ~29k | More superposed states; **best logged PAM** |
| Transformer B=18 | **22.69** | — | ~96k* | Ceiling; gap from best PAM: **+3.08 PPL** |

\*Transformer tok/s from V7 notes; PAM ~4× slower at matched geometry — FLOP-fundamental
(matrix-state d² work), not fixable by kernel fusion (Flash-PAM negative).

**What worked (cumulative stack):**
1. **Depth + narrow** (16×384) over wide-shallow — V7 3a proved depth dominates.
2. **ModSwish CGU** — phase-preserving activation; beats ModReLU (+0.58 at 7d B=18).
3. **Chunked dual form** — training speed + PPL; the compiled PyTorch path is near optimal.
4. **B=18 batch** — fewer steps/epoch but much lower val PPL vs B=3/B=6.
5. **E3 multi-state superposition** — the only architectural change that beat 7d on quality;
   uniquely PAM (phase-routed retrieval, not transformer/SSM).

**What did NOT work (dead ends, do not revisit without new evidence):**
- Hierarchy / grouped layers / cross-level drift (7f-3 halted ep1)
- Multi-scale auxiliary loss (7f-1: +0.82 PPL)
- Reverse association (7f-2: +2.46 PPL)
- Learned positional embeddings (7pos: ±0.2 PPL, neutral)
- V9 readout gates / conv gates (29.57 confounded → clean runs 30+ PPL)
- E1 per-channel decay (tie alone; hurts when stacked with E3)
- E2 delta-rule write (compile hang + OOM without compile)
- Flash-PAM kernel reformulation (correct, slower)
- Triton custom kernels under `torch.compile` (bypassed or graph-break)

**Architecture identity preserved throughout:** complex phase space, matrix-state PAM,
conjugate retrieval, O(1)/token recurrent inference, no softmax attention, no KV cache.

**Gap analysis:** V6→V11 closed **4.18 PPL** (29.95→25.77). Remaining gap to transformer
B=18 is **3.08 PPL** — likely needs **data + scale**, not more small ablations at 100M.

## Out of scope (already dead in V6–V9)
Hierarchy/grouped/cross-level, multi-scale loss, reverse-assoc, QK-norm-on, PIA
attention, V8 reasoning loops, V9 readout gates.

---

## Data recipe + continuous shipping (v2 gate line, 2026-07-01)

Decisions and implementation for the from-scratch v2 line (see
[MODEL_RELEASES.md](MODEL_RELEASES.md) and the shipping scripts).

### Reasoning tokens (vocab 50259 -> 50261)
Added `<think>` / `</think>` as special tokens in `get_chat_tokenizer()` (now
`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`). Same rationale as the
`<|im_end|>` fix: a single stable boundary the model can open/close/stop on, vs a
fragile multi-token BPE sequence. Trained from step 0 via the pretrain reasoning
blend. Preset `v11_e3_k3_chat` now defaults `vocab_size=50261`; the model auto-
matches `len(tokenizer)`. `_CHAT_CACHE_VERSION` bumped to 4.

### Format-aware source registry + blended pretrain
`v7/data.py` gained `SOURCE_REGISTRY` (schema `text`/`messages`, kind
`web`/`reason`/`chat`). `load_pretrain_mix` is now registry-driven and blends:
- `dclm` + `fineweb` (web/knowledge, raw text) with a **grammar warmup**
  (`blend_warmup_tokens`: web-only until the threshold, then the weighted blend),
- `smoltalk2_mid` (smoltalk2 **Mid** config, ~35B tokens of reasoning+chat)
  rendered to ChatML **text** via `format_chat_messages`.

So knowledge + reasoning + chat are present every round, with ChatML + `<think>`
tokens trained during pretrain (fixes the earlier "untrained ChatML tokens").

### Uniqueness across rounds (no token reuse)
Every source advances `per_source_docs` in the checkpoint (`_bump_counter`);
`skip_docs`/`skip_rows` skip already-consumed docs/rows next round. The trainer
auto-seeds skips from `--resume` checkpoints; `scripts/v11_data_cursor.py` also
computes them. Web is unique across rounds; the Mid reasoning pool (~35B) lasts
~50+ rounds of a ~5% sprinkle; FineWeb rotates `sample-10BT` -> `sample-100BT`
when near-exhausted. Note: the blend picks sources **per document** and Mid docs
are ~10x larger than web docs, so realized reasoning **token** share exceeds the
raw per-doc weight -- calibrate `PRETRAIN_WEIGHTS` from `per_source_tokens` after
round 1.

### SFT = real smoltalk2 (SFT config), not smol-smoltalk
`load_smoltalk2` now streams `HuggingFaceTB/smoltalk2` **SFT** config with a
`think_fraction` cap (default 0.15: mostly direct answers, capped reasoning for a
100M base). The SFT config is disjoint from the Mid blend pool (no leakage).

### Shipping pipeline
`scripts/run_v11_round.sh` orchestrates one round: `pretrain -> probes -> sft ->
smoke -> export` (GCP) and `ship` (RTX4090: pull -> verify -> push). Rounds ship
to HF **revision tags** (`round-2b-gate`, ...). Server catalog:
`releases/server_manifest.json`; incremental pull: `scripts/pull_v11_release.sh`
with per-machine `~/.qllm/v11_pull_state.json`. Probes fixed to load vocab/gate
flags from the checkpoint config (and the content-aware gate input).

### Validation (data pipeline smoke, 2026-07-01)
Tiny no-compile blended-pretrain smoke (batch 2, seq 512, 300k tokens): model
built at vocab 50261, blend streamed (dclm/fineweb/smoltalk2_mid), warmup drew
web-only then opened the blend, and the checkpoint saved `per_source_docs`
(`{dclm:3317, fineweb:385, smoltalk2_mid:8}`) + `per_source_tokens`. Tokenizer
check: all four specials encode to single ids.

---

## Training-speed rework (exact-math, 2026-07-04)

Goal: faster/leaner **same-GPU** training with **bit-exact math** (no quality
change), so `round-4b-gate` trains on the identical model. All changes are in
`v11/model.py` behind flags; parity is gated in `v11/selftest.py` and the step
benchmark is `v11/bench_step.py`.

### Findings (RTX Pro 6000, `v11_e3_k3_chat`, B=18/T=2048, bf16 + torch.compile, full fwd+loss+bwd+AdamW step)

| Variant | tok/s | ms/step | Peak VRAM | vs baseline |
|---------|------|---------|-----------|-------------|
| loop (old E3 K-loop + std CE) | 27,781 | 1327 | 79.9 GB | — |
| fused E3 | 44,959 | 820 | 70.3 GB | **1.62x**, −10 GB |
| fused E3 + chunked CE | 43,890 | 840 | **46.9 GB** | 1.58x, **−33 GB (−41%)** |

**Batch headroom** (fused+ce, T=2048): B=8 -> 23.0 GB / 37.4k tok/s ; B=18 -> 46.9 GB ;
B=32 -> 81.4 GB / **51.2k tok/s** (larger batch = better matmul efficiency) ; B=48 -> OOM.
So the freed memory converts directly to batch: **B=32 now fits** (was ~impossible at
79.9 GB for B=18), and **B<=8 fits a 24 GB RTX 4090** — 100M PAM training is now 4090-viable.

### What changed (why it's exact)
1. **Fused E3** (`fused_e3=True`, default). The old `_forward_multistate` ran the whole
   chunked PAM **K=3 times**. But the QK\* score `W` and protected value `v'` are
   **state-independent** (only the decay differs), and phase-routed retrieval is **linear**,
   so the per-state decay matrices collapse into ONE complex matrix
   `Dtilde[t,s] = Σ_k e^{iφ_k(t)} D_k[t,s]` and the intra-chunk output is a single complex
   matmul `y=(W⊙Dtilde)@v'`. Big C×C matmuls per chunk: 3× -> 1×. Carried-state read/write
   stay per-state but are the cheap O(C·d²) ops (K folded into batched matmul). Selftest:
   fwd `7.6e-19`, state `0.0`, grad `6.8e-21` vs the K-loop — bit-identical.
2. **Chunked cross-entropy** (`fused_ce_loss` / `ce_from_lm`, `v11/fused_ce.py`). The tied
   head is one real matmul `H@W.T`, `H=concat(lm_r,lm_i)`, `W=concat(E_r,E_i)`; a custom
   autograd fn computes CE in row-chunks so the `[B*T, 50261]` logits+softmax tensor
   (~4 GB + grad) is never materialized. Exact vs `F.cross_entropy`: loss `2.7e-7`,
   all-param grad `1.6e-8`. Compile the stack, keep CE eager.
3. **Selective recompute** (`recompute_pam_chunks=True`, off by default). Per-chunk
   `_fused_chunk_step` wrapped in `torch.utils.checkpoint`: the big `[B,H,C,C]` W/D/A
   tensors are recomputed in backward instead of stored (trades FLOPs for activation VRAM).
   Bit-exact (`6.5e-19`). VRAM saving to be measured on GPU; use only if pushing batch/seq.

### Notes
- Baseline peak was **79.9 GB** at B=18 — genuinely near the 96 GB card limit; the CE fix
  removed that pressure.
- Contrasts with the earlier **Flash-PAM negative** (kernel reformulation, slower): this is
  pure algebraic reduction of redundant work + fewer materialized tensors, which the
  inductor compiler rewards. No custom CUDA/Triton.
- Still TODO (speculative until benchmarked): Gauss 3-mult complex matmul, structure-of-
  arrays (r,i) layout, compile-mode/chunk_size sweep, bigger presets + long-context
  state-carry (TBPTT).

### round-4b-gate readiness
Math is identical to `round-2b-gate`, so `round-4b-gate` can resume from the round-1
**pretrain** checkpoint (+2B tokens) with `fused_e3=True` (already the default) and,
optionally, `fused_ce_loss` in the trainer for the VRAM win. `fused_ce` is **not yet wired
into `v7/train.py`** (the trainer still calls `forward()`+`_masked_ce`); wiring it is the
one remaining integration step to capture the memory headroom in the real round.
