# V11 Full-Duplex Audio POC (parallel track)

> **Not speech-to-speech.** This line trains a **turn-taking controller** on the same V11
> PAM math as text LM — predict `<listen>`, `<speak>`, or `<backchannel>` from streaming
> audio embeddings. It does **not** generate reply speech (Stage 2+ may hook external TTS).

Runs **in parallel** with the main **~10B-token text pretrain** on the RTX PRO 6000.
All duplex code is **additive** under `v11/duplex/` — shared `v11/model.py`, `v11/train.py`,
and pretrain scripts are untouched.

---

## Parallel GPU layout (2026-06-24)

| Track | GPU | Job | Status |
|-------|-----|-----|--------|
| **Text knowledge pretrain** | RTX PRO 6000 | `v11_e3_k3_chat`, DCLM-Edu + FineWeb-Edu, ~10B tokens | **RUNNING** (tmux `v11_pretrain`) |
| **Duplex audio POC** | RTX 4090 (local) | `duplex_5m` E3 K=3, frozen Whisper, Kathbath hi/gu | **Stage 1 done**; Gradio demo live |

Text pretrain: [`scripts/run_v11_pretrain_scratch.sh`](../../scripts/run_v11_pretrain_scratch.sh),
checkpoints `checkpoints_v11_e3_k3_chat_pretrain/`, resume via `latest.pt` every 5000 steps.

Duplex: [`scripts/run_v11_duplex_stage0.sh`](../../scripts/run_v11_duplex_stage0.sh),
[`scripts/run_v11_duplex_stage1.sh`](../../scripts/run_v11_duplex_stage1.sh),
Gradio [`scripts/run_v11_duplex_gradio.sh`](../../scripts/run_v11_duplex_gradio.sh).

**After 10B completes:** optional distill text knowledge into `duplex_25m` or a 100M audio
adapter — duplex POC is **not blocked** on pretrain finishing.

---

## What we are building (vs Moshi / PersonaPlex)

| Approach | This POC | PersonaPlex / Moshi |
|----------|----------|---------------------|
| Audio in vocab | **No** — codec-free embeddings | Mimi codec tokens in LM vocab |
| Backbone | V11 PAM E3 K=3 (~5M→10M) | ~7B transformer |
| Output (today) | **Thinking tokens** only | Discrete audio + text streams |
| Inference | O(1) fixed PAM state | KV cache or large model |
| Goal | **Mechanism proof**: barge-in / turn-taking | Production voice agent |

Inspired by **SALMONN-omni**: interleave environment + assistant stream **embeddings** with
explicit thinking tokens; audio never becomes thousands of codec IDs.

---

## Duplex math

### 1. Backbone — same PAM as V11 text LM

Per head, matrix state `S ∈ ℂ^{d×d}`, conjugate retrieval (unchanged from
[v11/EXPERIMENTS_V11.md](../EXPERIMENTS_V11.md)):

```text
Write:     S_t = γ_t · S_{t-1} + V_t ⊗ K_t*        # outer product; K* = complex conjugate
Retrieve:  Y_t = Σ_{k=1}^{K} e^{iφ_k(x_t)} · S_k · Q_t   # E3 K=3 multistate
Train:     chunked dual form O(T·C)
Infer:     recurrent O(1)/token, fixed state (no KV cache)
```

Duplex presets lock `n_states=3`, `decay_mode='head'`, `write_mode='additive'` — the
**proven** V11 winner; no E1/E2 dead ends.

### 2. Codec-free audio path (Stage 1)

Whisper is **encoder only**, frozen:

```text
waveform x[n]  →  resample to 16 kHz  →  Whisper encoder  →  h ∈ ℝ^{D}  (D=768 for small)
h  →  ComplexLinear(D → dim)  →  z_audio ∈ ℂ^{dim}  (stacked as [real, imag] channels)
```

Chunked ~1 s segments; up to 4 chunks per block. **No Whisper decoder** — not ASR, not TTS.

Injection into the LM (subclass `V11DuplexLM`, no edits to shared `V11LM`):

```text
z_t = Embed(token_id)           for text / control tokens
z_t = project_audio(h)          at positions after <env_mark>  (audio slots)
logits = LMHead(PAM_blocks(z))
```

Audio occupies **embedding positions**, not vocab IDs — keeps vocab at 512 for control + text
placeholders.

### 3. Interleaved time block (~160 ms target; ~1 s chunks in POC)

Per block the sequence is:

```text
<env_mark>  [audio_slot × N]  →  predict  <listen> | <speak> | <backchannel>
[optional <ast_mark> + assistant echo slots]
[optional reply text tokens if <speak>]
```

Training labels: CE only on **thinking tokens** (and reply tokens when present); audio slots
and env structure masked with `ignore_index=-100`.

```text
Loss = CrossEntropy(logits[:, t], labels[:, t+1])   on thinking (+ reply) positions
```

Random baseline for 3-way thinking: **33.3%**. Stage 0 uses **text token proxies** for
“user spoke”; Stage 1 uses **real Kathbath / LibriSpeech** waveforms.

### 4. What “100% thinking accuracy” means (and does not)

Hi/gu Stage 1: **val think_acc = 1.0** on 400 simulated duplex pairs (200 hi + 200 gu).
Pairs are **consecutive single-speaker clips** — not true dual-channel Fisher conversation.
Metric validates the **control head**, not open-domain Hindi/Gujarati dialogue quality.

---

## Scale ladder (`v11/duplex/config.py`)

| Preset | dim | layers | heads | K | ~params |
|--------|-----|--------|-------|---|--------|
| `duplex_5m` | 160 | 8 | 4 | 3 | **5.25M** |
| `duplex_10m` | 208 | 8 | 6 | 3 | **9.16M** |
| `duplex_25m` | 304 | 10 | 8 | 3 | **23.5M** |

Small vocab (512) so params sit in **PAM blocks**, not tied embeddings.

---

## Results log

| Stage | preset | data | metric | result | checkpoint / log |
|-------|--------|------|--------|--------|------------------|
| 0 synthetic | `duplex_5m` | text proxy, barge-in sim | val think_acc | **100%** | `checkpoints_v11_duplex_5m_stage0/` |
| 1 audio EN | `duplex_5m` | LibriSpeech + Whisper | val think_acc | **100%** | `checkpoints_v11_duplex_5m_stage1/` · `logs/v11/duplex_stage1_duplex_5m_20260624_114119.log` |
| 1 audio hi/gu | `duplex_5m` | Kathbath hindi+gujarati, 400 pairs, 10 ep | val think_acc | **100%** (ep2–10) | `checkpoints_v11_duplex_5m_stage1_hi_gu/best_model.pt` · `logs/v11/duplex_stage1_hi_gu_train.log` |

Hi/gu run: **232 s** wall, `save_every_steps=50` → resumable `latest.pt`.

---

## Code map (additive only)

```
v11/duplex/
  config.py       # duplex_5m / 10m / 25m presets (E3 K=3)
  model.py        # V11DuplexLM(V11LM): audio embed injection
  encoder.py      # FrozenWhisperEncoder + 16 kHz resample
  thinking.py     # listen / speak / backchannel vocab
  interleave.py   # synthetic duplex blocks
  audio_data.py   # LibriSpeech + Kathbath loaders
  train.py        # Stage 0
  train_stage1.py # Stage 1 + periodic checkpoints
  infer.py        # single-shot thinking prediction
  gradio_app.py   # checkpoint picker + mic demo
  probe.py        # synthetic barge-in probe
```

Plan doc (design notes): [`.cursor/plans/v11_duplex_audio_poc_9606141c.plan.md`](../../.cursor/plans/v11_duplex_audio_poc_9606141c.plan.md)

---

## Next steps

1. **Gradio / probe** on held-out audio — confirm 100% metric isn’t train-set leakage.
2. **`duplex_10m`** rerun (same hi/gu data) — scaling curve.
3. **Stage 2** — streaming `infer.py` (~160 ms blocks, PAM state carry) + optional TTS on `<speak>`.
4. **Better data** — Fisher dual-channel (LDC) or real overlap/barge-in; Kathbath is monolingual clips only.
5. **Post–10B merge** — distill `v11_e3_k3_chat` into `duplex_25m` for reply **content**, not just turn-taking.
