---

license: mit
language:

- en
tags:
- phase-associative-memory
- complex-valued
- o1-inference
- attention-free
- non-transformer
- qllm
- pam
pipeline_tag: text-generation

---

# QLLM PAM V11 E3 K=3 Chat

**A milestone, not a SOTA claim:** the first public chat model built on **Phase-Associative Memory (PAM)** — **not a transformer**, **not Mamba** — with **O(1) per-token inference** and a **fixed-size matrix state** (no KV cache).

This checkpoint proves that architectures outside the transformer/SSM families **can learn coherent instruction-following chat**. It is deliberately **under-trained** relative to what the architecture can hold.

> **Architecture still under development.** The PAM stack (memory gates, tokenizer, data blend) is
> actively changing. We **restarted training from scratch** on the v2 line. After every **+2B
> pretrain tokens** we publish a new weights file here under a **round revision tag**
> (e.g. `round-2b-gate`, `round-4b-gate`). The `main` branch may still point at an older
> milestone checkpoint — **pin a revision tag** for the round you want.
>
> **Latest shipped:** revision **`round-2b-gate`** — **2026-07-04** (Round 1, v2 gate line).
> Only **2B pretrain tokens + SFT**, but a **new architecture** (content-aware phase gate, vocab
> 50261). See [Round 1 details](#round-1--round-2b-gate-shipped-2026-07-04) below.


|                     | **`round-2b-gate` (2026-07-04)** |
| ------------------- | -------------------------------- |
| **Params**          | ~100.5M |
| **Architecture**    | V11 E3 K=3 PAM + **content-aware GSP gate** (not legacy magnitude-only gate) |
| **Vocab**           | **50261** — ChatML + `<think>` / `</think>` |
| **Pretrain**        | **2.0B tokens** blended (see below) — from scratch, v2 line |
| **SFT**             | `HuggingFaceTB/smoltalk2` **SFT** config, **hard filter**, **15% think** cap |
| **Val (in-distro)** | SFT holdout PPL **7.20**, acc **0.591** (pretrain holdout PPL **35.42**) |
| **Inference**       | **O(1)/token**, fixed state, **no KV cache** |
| **Code**            | [github.com/gowrav-vishwakarma/qllm2](https://github.com/gowrav-vishwakarma/qllm2) |
| **Paper**           | [arXiv:2604.05030](https://arxiv.org/abs/2604.05030) |
| **HF repo**         | [huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat) |

*(Legacy `main` / ~10B milestone: vocab 50259, magnitude-only gate — comparison only.)*


---

## Releases (revision tags)

The architecture is **still under active development**. We **retrain from scratch** on the v2
line, then ship a new checkpoint after each **+2B pretrain tokens** (+ chat SFT) under its own
**revision tag** so you can pin an exact snapshot while the stack evolves.

```bash
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat --revision <tag> --local-dir .
# Repo: https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat
```

| Revision | Shipped | Pretrain total | Trained on | Metrics | Notes |
|----------|---------|----------------|-----------|---------|-------|
| `main` | (legacy) | ~10B | DCLM-Edu + FineWeb-Edu → old SmolTalk SFT | val PPL ~7.2 | **pre-v2** — vocab 50259, magnitude-only gate; comparison only |
| **`round-2b-gate`** | **2026-07-04** | **2B** | blended pretrain → smoltalk2 SFT | SFT PPL **7.20**, pretrain holdout **35.42** | **Round 1, v2 line** — see [details](#round-1--round-2b-gate-shipped-2026-07-04) |

<!-- Append one row per shipped round. -->

### Round 1 — `round-2b-gate` (shipped **2026-07-04**)

First checkpoint on the **v2 gate line**: we **restarted from scratch** (not a continuation of
the legacy ~10B run). Token count is modest (**2B pretrain + 1 SFT epoch**), but the **stack is
new** — treat this as an architecture + pipeline snapshot, not a knowledge-saturated model.

**Architecture (what changed vs legacy `main`):**

| Component | v2 (`round-2b-gate`) | Legacy (`main`) |
|-----------|----------------------|-----------------|
| GSP write gate | **Content-aware** — reads real+imag (`2×dim`) | Magnitude-only (`dim`) |
| Vocab | **50261** — ChatML + `<think>` / `</think>` | 50259 — ChatML only |
| Pretrain mix | Web + **smoltalk2 Mid** (ChatML-rendered reasoning/chat) | Web only |

**Pretrain — 2.0B tokens (from scratch, preset `v11_e3_k3_chat`):**

| Phase | Data | Notes |
|-------|------|-------|
| Warmup (first **1B** tokens) | DCLM-Edu + FineWeb-Edu only | Grammar / knowledge before chat blend |
| Blend (remaining **~1B**) | DCLM-Edu + FineWeb-Edu + **smoltalk2 Mid** | Per-doc weights **48 : 48 : 4**; Mid docs are larger so **~8% of tokens** were reasoning/chat-form |

Approximate token mix over the full 2B: **~52% DCLM-Edu**, **~40% FineWeb-Edu**, **~8% smoltalk2 Mid**
(edu score ≥ 3; deterministic cursors ensure fresh shards on the next round).

**SFT — after pretrain:**

| Setting | Value |
|---------|-------|
| Dataset | `HuggingFaceTB/smoltalk2` config **`SFT`** (disjoint from Mid pretrain pool) |
| Filter | **`hard`** (quality filter on conversations) |
| Reasoning | **`think_fraction=0.15`** — only ~15% of reasoning (think) rows kept; mostly direct answers |
| Template | ChatML; assistant-only loss |
| Metrics | In-distribution val PPL **7.20**, acc **0.591** |

**Download this round:**

```bash
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat \
  --revision round-2b-gate --local-dir .
```

### What you can ask (and current limits)

- **Good at:** short factual questions, simple instructions, everyday chat, ChatML multi-turn.
- **Improving:** light reasoning (the model may emit `<think>…</think>` before answering).
- **Still weak (100M base):** precise math, long multi-step reasoning, rare/long-tail facts,
  and occasional confabulation. Not a search engine or calculator.

Data recipe per round is documented in
[v11/MODEL_RELEASES.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/MODEL_RELEASES.md).

---



## Why this matters

Transformers pay **O(T²)** training cost and a **KV cache that grows without bound** at inference.

Vector SSMs (Mamba, etc.) get O(1) inference but store memory in a **single vector** — limited associative capacity, catastrophic interference past a handful of facts.

**QLLM's bet:** give each head a **complex matrix state** `S ∈ ℂ^{d×d}` and retrieve by **complex-conjugate phase matching**:

```text
State:     S_t = γ_t · S_{t-1} + V_t ⊗ K_t*
Retrieve:  Y_t = S_t · Q_t          (no softmax, no attention)
Train:     chunked dual form O(T·C)
Infer:     recurrent O(1)/token     (fixed state — no KV cache)
```

Each outer product writes a full **d×d** association → **O(d²) capacity per head**. Mechanism probes (no training) show **100% retrieval @ 64 associations** vs **~13% for vector HRR** — the architecture can *hold* far more than this 100M checkpoint has been trained to use.

**E3 multistate (K=3):** three superposed matrix states per head, combined by data-dependent **phase interference** — a uniquely PAM idea, not copyable from transformers or Mamba.

---



## Honest limits (read before expecting GPT-4)

This model is a **milestone checkpoint**, not a production assistant:


| What works                     | What doesn't (yet)                                           |
| ------------------------------ | ------------------------------------------------------------ |
| ChatML turn structure          | Reliable factual knowledge (base saturates ~WikiText PPL 65) |
| Stops on `<|redacted_im_end|>` | Math ("what is 2+2" drifts)                                  |
| "Capital of France → Paris"    | Name/memory confabulation                                    |
| Coherent one-sentence answers  | Deep reasoning, coding beyond toy examples                   |


Training budget: **~10B pretrain + SmolTalk2 SFT at 100M params**. PAM capacity math says the architecture is **far from saturated** — we stopped here to ship proof-of-learning, not to chase leaderboard scores.

Same-pipeline transformer baseline (100M, WikiText): val PPL **22.69** vs our PAM **25.77** (WikiText anchor) — we report the gap on purpose. The trade is a **different memory mechanism** and **O(1) inference without KV cache**.

---



## How to run (raw code — no `transformers` AutoModel)

This architecture **cannot** be loaded with `AutoModelForCausalLM`. Use the self-contained Python files bundled in this repo:

```bash
pip install torch transformers

# Download this repo (or clone from HF)
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat \
  qllm_v11_e3k3_chat.pt config.json modeling_qllm.py run_chat.py requirements.txt
```



### Interactive multi-turn chat

```bash
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt
```


| Input                   | Action                                                             |
| ----------------------- | ------------------------------------------------------------------ |
| Type a message + Enter  | Send; model sees full conversation history                         |
| Empty line (Enter only) | Start a **new chat** — history cleared, session counter increments |
| `exit`                  | Quit                                                               |
| Ctrl+C or EOF           | Quit                                                               |


Multi-line paste is supported: paste a block and the script waits briefly to absorb trailing lines as one message.

Optional tuning: `--max_new_tokens` (default 256), `--temperature` (default 0.7).

### Custom system prompt

```bash
# Inline
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt \
  --system "You are a concise science tutor. Answer in one sentence."

# From file (overrides --system)
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt \
  --system-file my_system.txt
```

The system prompt applies to **all chats in that run**. Starting a new chat (empty line) clears turn history but keeps the same system prompt until you restart the script.

### Single prompt (non-interactive)

For scripts, CI, or one-off queries:

```bash
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt \
  --prompt "What is the capital of France?" \
  --temperature 0.0 --max_new_tokens 32
```



### Minimal inference snippet

For day-to-day chatting, use `run_chat.py` above; the snippet below is for custom integrations.

```python
import torch
from transformers import AutoTokenizer
from modeling_qllm import load_model

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("qllm_v11_e3k3_chat.pt", device=device)

tok = AutoTokenizer.from_pretrained("gpt2")
tok.add_special_tokens({"additional_special_tokens": [IM_START, IM_END]})

prompt = (
    f"{IM_START}system\nYou are a helpful assistant.{IM_END}\n"
    f"{IM_START}user\nWhat is the capital of France?{IM_END}\n"
    f"{IM_START}assistant\n"
)
ids = tok.encode(prompt, return_tensors="pt").to(device)
im_end_id = tok.convert_tokens_to_ids(IM_END)

with torch.no_grad():
    out = model.generate(
        ids, max_new_tokens=64, temperature=0.7,
        top_k=50, top_p=0.9, repetition_penalty=1.15,
        eos_token_id=im_end_id,
    )

reply = tok.decode(out[0, ids.shape[1]:].tolist())
print(reply.split(IM_END, 1)[0].strip())
```

Generation uses the **O(1) recurrent PAM path** — a fixed matrix state per layer, updated one token at a time. No KV cache, no growing memory.

---



## Architecture comparison


|           | Transformer             | Vector SSM (Mamba)   | **QLLM PAM (this model)**   |
| --------- | ----------------------- | -------------------- | --------------------------- |
| State     | KV cache (grows with T) | vector `s ∈ ℝ^{S×d}` | matrix `S ∈ ℂ^{H×d×d}`      |
| Matching  | QKᵀ + softmax           | gated recurrence     | complex conjugate K*·Q      |
| Capacity  | O(n) per sequence       | ~O(S·d)              | **O(H·d²) per layer**       |
| Inference | O(T) + growing cache    | O(1)/token           | **O(1)/token, fixed state** |


---



## Training recipe

**Current v2 line (from scratch, +2B per round):**

1. **Pretrain from scratch** — preset `v11_e3_k3_chat` (vocab **50261**: ChatML + `<think>`/`</think>`), blended DCLM-Edu + FineWeb-Edu + smoltalk2-Mid (grammar warmup, then mix), **2B tokens per round**, phase-aware GSP gate.
2. **SFT** — real `HuggingFaceTB/smoltalk2` SFT config (`think_fraction` capped), 1 epoch, lr 5e-5, ChatML assistant-only loss.
3. **Ship** — export + verify, push to this repo under a **round tag** (e.g. `round-2b-gate`).

**Legacy `main` checkpoint (reference only):**

1. Pretrain ~10B DCLM-Edu + FineWeb-Edu (vocab 50259, pre-reasoning-token).
2. SFT on older SmolTalk pipeline with `--warmstart_chatml`.

**Template** — ChatML over GPT-2 tokenizer (+ reasoning markers on v2):

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

Full reproduction logs: [v11/EXPERIMENTS_V11.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/EXPERIMENTS_V11.md). Per-round provenance: [v11/MODEL_RELEASES.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/MODEL_RELEASES.md).

---



## Next milestone (explicit)

The next goal is **not** "train longer" or "scale to 7B first."

1. **Make training ~10× faster** — target **~10% of current wall-clock** for the same quality curve. (Moon shot call) 
2. **Intelligence-first at 100M–300M** — prove the PAM stack learns smarter before scaling params or token budget.

Then scale.

---



## Links

- **Hugging Face (model weights):** [huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat)
- **Paper:** [Phase-Associative Memory: Sequence Modeling in Complex Hilbert Space](https://arxiv.org/abs/2604.05030) (arXiv:2604.05030)
- **Main repo:** [github.com/gowrav-vishwakarma/qllm2](https://github.com/gowrav-vishwakarma/qllm2)
- **Experiments log:** [v11/EXPERIMENTS_V11.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/EXPERIMENTS_V11.md)
- **Beginner guide:** [v11/BEGINNER_GUIDE.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/BEGINNER_GUIDE.md)
- **PAM mechanism probes:** [memory_probes/README.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/memory_probes/README.md)

---



## License

**MIT License** — see [LICENSE](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat/blob/main/LICENSE).