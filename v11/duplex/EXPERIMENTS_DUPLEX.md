# V11 Full-Duplex Audio POC (parallel track)

> **Not speech-to-speech.** This line trains a **turn-taking controller** on the same V11
> PAM math as text LM — predict `<listen>`, `<speak>`, or `<backchannel>` from streaming
> audio embeddings. It does **not** generate reply speech (Stage 2+ may hook external TTS).

Runs **in parallel** with the main **~10B-token text pretrain** on the RTX PRO 6000.
All duplex code is **additive** under `v11/duplex/` — shared `v11/model.py`, `v11/train.py`,
and pretrain scripts are untouched.

---

## Parallel GPU layout (2026-06-30)

| Track | GPU | Job | Status |
|-------|-----|-----|--------|
| **Text knowledge pretrain** | RTX PRO 6000 | `v11_e3_k3_chat`, DCLM-Edu + FineWeb-Edu, ~10B tokens | **DONE** → `checkpoints_v11_e3_k3_chat_pretrain/best_model.pt` (WikiText val PPL 65.46) |
| **Text chat SFT** | RTX PRO 6000 | SmolTalk2 + `--warmstart_chatml` on 10B base | **DONE** → `checkpoints_v11_sft_chat_smoltalk/best_model.pt` (see [EXPERIMENTS_V11.md](../EXPERIMENTS_V11.md) recovery readout) |
| **Duplex audio POC** | RTX 4090 (local) | `duplex_5m` E3 K=3, frozen Whisper, Kathbath hi/gu | **Stage 1 done**; Gradio demo live |

Text pretrain: [`scripts/run_v11_pretrain_scratch.sh`](../../scripts/run_v11_pretrain_scratch.sh),
checkpoints `checkpoints_v11_e3_k3_chat_pretrain/`, resume via `latest.pt` every 5000 steps.

Duplex: [`scripts/run_v11_duplex_stage0.sh`](../../scripts/run_v11_duplex_stage0.sh),
[`scripts/run_v11_duplex_stage1.sh`](../../scripts/run_v11_duplex_stage1.sh),
Gradio [`scripts/run_v11_duplex_gradio.sh`](../../scripts/run_v11_duplex_gradio.sh).

**After 10B + SmolTalk SFT:** optional distill text knowledge from
`checkpoints_v11_sft_chat_smoltalk/best_model.pt` into `duplex_25m` or a 100M audio
adapter — duplex POC is **not blocked** on text chat finishing.

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

## Next steps (POC track)

1. **Gradio / probe** on held-out audio — confirm 100% metric isn’t train-set leakage.
2. **`duplex_10m`** rerun (same hi/gu data) — scaling curve.
3. **Better data** — Fisher dual-channel (LDC) or real overlap/barge-in; Kathbath is monolingual clips only.

---

# Voice Interface Model (hi/gu/en) — Stages A–D  ·  2026-07-07

The POC above only predicts `listen/speak/backchannel`. This track evolves the **same
PAM backbone** into a real speech interface: streaming **S2T** (feeds an external LLM
"brain") + streaming **T2S** (speaks the brain's reply as Mimi codec tokens), keeping the
duplex control. Plan: [`.cursor/plans/duplex_voice_interface_model_7c3124dc.plan.md`].

```mermaid
flowchart LR
    mic[Mic] --> enc["Frozen Whisper encoder<br/>frame seq ~12.5Hz"]
    enc --> pam["ONE V11 PAM backbone ~93M<br/>single O(1) state"]
    pam -->|S2T text| llm["Local LLM brain<br/>(ollama / OpenAI-compatible)"]
    llm -->|reply text| pam
    pam -->|Mimi codec tokens| dec[Mimi decoder] --> spk[Speaker]
    pam -->|listen/speak/backchannel| ctl[Duplex control]
```

## Key facts

- **ONE model, not two.** Stage A/B/C are a *training curriculum* on the same
  `duplex_100m` backbone + same unified vocab, not separate networks. Stage B warm-starts
  from Stage A (`--init_from`), Stage C from A+B (`--init_asr`/`--init_tts`). Final artifact:
  a single `duplex_100m` checkpoint that does S2T, T2S, and control. (See FAQ below.)
- **Unified vocab ~40,208** (`tokenizer.py`): 16 control + 32,000 SentencePiece (hi/gu/en,
  `byte_fallback`) + 4×2048 Mimi codec. Text + codec share the tied embedding/LM head.
- **Backbone** `duplex_100m` = proven `v11_e3_k3` geometry (16×384, K=3, content-aware gate,
  chunk 256). Built: **92.83M** params (embedding 30.88M, blocks 61.65M).
- **Speech in** = frozen `whisper-small` **frame sequence** stride-4 → ~12.5 Hz (not the POC's
  1-vec/s mean-pool), injected as complex embeddings at `<audio_pad>` slots.
- **Speech out** = frozen `kyutai/mimi` (24 kHz, 12.5 Hz), first 4 codebooks, **delay pattern**.

## Stages (each independently gated on the 4090)

| Stage | file | task | gate (metric) |
|-------|------|------|---------------|
| A | `train_asr.py` | S2T (ASR): `<env>[audio]<transcribe><lang> text <eos>` | held-out **CER** (target <15%) |
| B | `train_tts.py` | T2S: `<lang> text <tts> [codec] <eos>`; `--task both` reuses each pair S2T+T2S | **round-trip WER** (Whisper on generated audio) |
| B* | `train_tts.py --task roundtrip` | `[audio]<transcribe> text <tts> codec-of-same-audio` | **ablation only** (copy-shortcut risk); keep only if it beats plain mix |
| C | `train_duplex_v2.py` | interleaved S2T + reply text + T2S + control + barge-in, init from A+B | bucketed **control / text / codec** next-token acc |
| D | `infer_stream.py` | streaming session: block loop, one PAM state, barge-in, brain | qualitative (Gradio voice tab) |

Sequence layouts and loss masks are documented in each file's docstring. All trainers share
crash-safe `latest.pt` resume, cosine LR + warmup, bf16 autocast, SIGTERM/SIGINT save.

## Data reuse (core recipe)

Each `(audio, text)` pair → **two samples**: `[audio]<transcribe>text` (S2T) and
`text<tts>codec` (T2S). Standard, low-risk, and it makes scarce **Gujarati** data train
understanding *and* speaking. Kathbath hi/gu (noisy/multi-speaker) covers both; clean
single-speaker corpora (IndicTTS/Rasa, LibriTTS-R) should be added via extra rows for output
voice quality. The single-sequence round-trip autoencoder is a **later ablation only**.

## Launch order

```bash
./scripts/run_v11_duplex_tokenizer.sh    # once: unified hi/gu/en + codec vocab
./scripts/run_v11_duplex_asr.sh          # Stage A  -> gate CER
./scripts/run_v11_duplex_tts.sh          # Stage B  -> gate round-trip WER  (inits from A)
./scripts/run_v11_duplex_v2.sh           # Stage C  -> joint duplex          (inits from A+B)
# Stage D (talk to it):
uv run python -m v11.duplex.infer_stream \
    --checkpoint checkpoints_v11_duplex_100m_duplex_v2/best_model.pt \
    --audio in.wav --lang hindi --brain
./scripts/run_v11_duplex_gradio.sh       # "Voice interface" tab
```

Brain = any OpenAI-compatible endpoint (`brain.py`): ollama / llama.cpp / vLLM. Start with
Qwen2.5-3B-Instruct or Sarvam-1 (strong Hindi). `BRAIN_BASE_URL`, `BRAIN_MODEL` env vars.

## Architecture FAQ (why one model + memory)

**Q: Is this two PAM models (one ASR, one TTS) or one?**
One. `train_asr.py` and `train_tts.py` are separate *scripts* for a staged curriculum, but
they train the **same architecture and vocab**; B initializes from A, C from A+B. At
inference (`infer_stream.DuplexSession`) the single model runs S2T, then the external LLM
produces reply text, then the **same model resumes** with T2S — the PAM state is threaded
across that boundary (`self.states`, `self.step`). So it is "one model with an intermediate
brain value, then resume", exactly.

**Q: Does the LLM see the whole conversation, or do we need a mechanism outside the model to
merge old chat with new?**
The mechanism already exists and lives **outside** the PAM model, in `brain.Conversation`:
every user transcript and every assistant reply is appended to a text message list, and the
**full history is sent to the LLM each turn** (system + all turns). The LLM is not starved —
long-conversation recall is guaranteed by the brain's text context, not by the 100M state.
(For very long chats, add optional summarization/compaction on top — not required for a demo.)

**Q: Two memory layers — which holds what?**

| Layer | Holds | Used for | Persistence |
|-------|-------|----------|-------------|
| **LLM brain text context** (`Conversation`) | verbatim transcripts + replies | semantic recall, reasoning | whole session (grows as text) |
| **PAM recurrent state** | acoustic / conversational **gist** | turn-taking, barge-in, backchannel timing, prosodic continuity within a turn | O(1) fixed size, streamed across blocks |

The PAM state is **deliberately not** the semantic memory — it is fixed-size and small; that
job is the LLM's.

**Q: If it's "just ASR + TTS + external LLM", why the single PAM model and why memory at all?
Why not cascade Whisper → LLM → an off-the-shelf TTS?**
For strict **turn-based** Q&A, a cascade would indeed be simpler and the PAM model's edge is
mainly latency/unification. The single streaming model earns its place only for **full-duplex
behavior**, which a stateless cascade fundamentally cannot do:

- **Listen while speaking** — the time-multiplexed block loop carries incoming mic frames and
  outgoing codec in the *same* O(1) state, so the model keeps hearing the user mid-reply.
- **Sub-second barge-in** — control can flip to `<listen>` and truncate codec emission the
  instant the user interrupts (`interrupt_check`), instead of finishing a TTS utterance.
- **Backchannels / turn detection** — `listen/speak/backchannel` decided continuously from the
  live state, not after a full ASR turn ends.
- **Prosodic/acoustic conditioning** — the reply is shaped by *how* the user spoke, carried in
  the state, not just the transcribed words.
- **Streaming latency** — O(1)/token, no re-encoding a growing context window.

So: **memory (the PAM state) is needed for the duplex/streaming behavior, not for semantic
recall**; semantic recall is the LLM's text context. If you only ever want polite turn-taking
with no overlap or barge-in, the model's advantage shrinks — the value is realized in live,
interruptible, overlapping conversation.

## Code map (voice-interface additions)

```
v11/duplex/
  tokenizer.py        # DuplexVocab (unified id-space) + SentencePiece hi/gu/en
  encoder.py          # + encode_frames(): ~12.5 Hz frame sequence
  config.py           # + duplex_100m preset (~40k vocab, v11_e3_k3 geometry)
  codec.py            # Mimi encode/decode + delay-pattern flatten/unflatten
  train_asr.py        # Stage A (S2T), CER gate, greedy state-carry decode
  train_tts.py        # Stage B (T2S / both / roundtrip-ablation), round-trip WER
  brain.py            # OpenAI-compatible client + Conversation + dataset builder
  train_duplex_v2.py  # Stage C joint interleaved (S2T+T2S+control+barge-in)
  infer_stream.py     # Stage D DuplexSession: block loop, state carry, barge-in
  gradio_app.py       # + "Voice interface" tab (mic -> transcript -> LLM -> speech)
```

Validated offline (tiny model + real Mimi + synthetic tokenizer): tokenizer roundtrip, codec
delay roundtrip, all stage sample-builders + forward/loss/backward, constrained codec
generation, barge-in truncation, full `converse`. `v11/duplex/selftest.py` = **10/10 pass**.

## Status / next steps (voice track)

- [x] Tokenizer, encoder frames, `duplex_100m` preset, codec, Stages A–D code + scripts, tests.
- [ ] Run tokenizer training on full Kathbath hi/gu + LibriSpeech corpus.
- [ ] Stage A training run → record CER here.
- [ ] Stage B training run (`--task both`) → record round-trip WER here.
- [ ] Stage C joint run → record control/text/codec acc here.
- [ ] Add clean single-speaker TTS corpora (IndicTTS/Rasa, LibriTTS-R) for output voice.
- [ ] Optional: distill `v11_e3_k3_chat` text knowledge to reduce brain dependence.
