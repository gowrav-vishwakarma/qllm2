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

**The first public chat model built on Phase-Associative Memory (PAM)** — not a transformer, not Mamba — with **O(1) per-token inference** and a **fixed-size matrix state** (no KV cache).

This checkpoint proves that architectures outside the transformer/SSM families **can learn coherent instruction-following chat**. It is deliberately **under-trained** relative to what the architecture can hold.

**Before you try:** this is a **proof-of-concept of the architecture**, not a polished chat product. Think of it like early GPT-2 — it shows the idea works and is progressing, but it is not yet what you might expect from a modern assistant.

---

## Which revision to use

**Use HF `main` (default).** It tracks the latest v2 round. We ship a new checkpoint after every **+2B pretrain tokens** (+ chat SFT); each round also gets a permanent **`round-*` tag** (e.g. `round-2b-gate`, `round-6b-gate`, `round-8b-gate`).

| | **`main` / latest round (recommended)** | **`v1-old-deprecated-10B-sft` (deprecated)** |
|--|----------------------------------------|------------------------------------------------|
| **Status** | **Active line** — Round 1 on `main`, continued each +2B | Legacy ~10B-token checkpoint; **comparison only** |
| **Params / pretrain** | ~100M, **8B tokens** (grows each round) | ~100M, ~10B tokens |
| **Gate** | Content-aware — real+imag (`gate_content_aware=true`) | Magnitude-only (legacy) |
| **Vocab** | **50261** — ChatML + `<think>` / `</think>` | **50259** — ChatML only |
| **Download** | default (no `--revision`) or `--revision round-8b-gate` | `--revision v1-old-deprecated-10B-sft` |

**Latest shipped:** **`round-8b-gate`** on **`main`** — **2026-07-10** (Round 4, +2B pretrain since `round-6b-gate`).

Pin **`round-6b-gate`** or earlier tags on HF to compare prior rounds — those READMEs and weights are frozen per revision.

### Round 4 recipe change (no-think, no math SFT)

Starting with **`round-8b-gate`**, we changed **both pretrain and SFT** to remove reasoning/math modes that were hurting this ~100M base:

| Change | What we removed | Why |
|--------|-----------------|-----|
| **Pretrain blend** | `<think>…</think>` traces from `smoltalk2_mid` | Base learned to open answers with thinking blocks and "explain the question" prose |
| **SFT** | `think_fraction` **0.15 → 0** | No `<think>` in SFT targets — direct ChatML answers only |
| **SFT splits** | Math/reasoning-heavy pools (e.g. OpenThoughts3, MATH-style, multilingual, tool-calling) | ~100M cannot use math/LaTeX/multilingual mix; it regressed short English factual QA |
| **SFT filter** | Verbose assistant dumps (cap 300 words, `hard` filter) | Cuts rambling table/reasoning templates |

Earlier tags (`round-2b-gate` … `round-6b-gate`) still used **15% think SFT** and the **unstripped** Mid blend. Compare them side-by-side to see the effect of this cleanup.

---

## What is it — architecture

**Phase-Associative Memory (PAM)** gives each head a **complex matrix state** `S ∈ ℂ^{d×d}` and retrieves by **complex-conjugate phase matching**:

```text
State:     S_t = γ_t · S_{t-1} + V_t ⊗ K_t*
Retrieve:  Y_t = S_t · Q_t          (no softmax, no attention)
Train:     chunked dual form O(T·C)
Infer:     recurrent O(1)/token     (fixed state — no KV cache)
```

Each outer product writes a full **d×d** association → **O(d²) capacity per head**. Mechanism probes (no training) show **100% retrieval @ 64 associations** vs **~13% for vector HRR** — the architecture can *hold* far more than this 100M checkpoint has been trained to use.

**E3 multistate (K=3):** three superposed matrix states per head, combined by data-dependent **phase interference** — a uniquely PAM idea, not copyable from transformers or Mamba.

| | **`round-8b-gate` (latest on `main`)** |
|--|--------------------------------------|
| **Params** | ~100.5M |
| **Architecture** | V11 E3 K=3 PAM + content-aware GSP gate |
| **Vocab** | 50261 — ChatML + thinking tokens (unused in SFT targets this round) |
| **Pretrain** | **8.0B tokens** cumulative (2B this round; see [Round history](#round-history)) |
| **SFT** | `smoltalk2` SFT, **no-think** (`think_fraction=0`), math/reasoning splits dropped |
| **Val (in-distro)** | SFT holdout PPL **7.88** (Round 3 @ 6B with think-SFT: PPL **6.55**) |
| **Inference** | O(1)/token, fixed state, no KV cache |

---

## Why it's better

Transformers pay **O(T²)** training cost and a **KV cache that grows without bound** at inference.

Vector SSMs (Mamba, etc.) get O(1) inference but store memory in a **single vector** — limited associative capacity, catastrophic interference past a handful of facts.

| | Transformer | Vector SSM (Mamba) | **QLLM PAM (this model)** |
|--|-------------|--------------------|---------------------------|
| **State** | KV cache (grows with T) | vector `s ∈ ℝ^{S×d}` | matrix `S ∈ ℂ^{H×d×d}` |
| **Matching** | QKᵀ + softmax | gated recurrence | complex conjugate K*·Q |
| **Capacity** | O(n) per sequence | ~O(S·d) | **O(H·d²) per layer** |
| **Inference** | O(T) + growing cache | O(1)/token | **O(1)/token, fixed state** |

**The trade:** same-pipeline transformer baseline (100M, WikiText) val PPL **22.69** vs our PAM **25.77** — we report the gap on purpose. You get a **different memory mechanism** and **O(1) inference without KV cache**.

**Honest limits (Round 4, 8B, no-think SFT):** still a **milestone checkpoint**, not a production assistant. SFT holdout PPL is **higher** than `round-6b-gate` (7.88 vs 6.55) because we removed easy-to-fit reasoning/math templates — the trade is cleaner short answers vs metric gaming. Chat/social works; some factual hits improved (e.g. US capital). Math remains weak by design (math splits removed). GPT-2 BPE is **case-sensitive**. Vocab still has thinking tokens, but this round's SFT does not train them.

---

## How to run on your system

This architecture **cannot** be loaded with `AutoModelForCausalLM`. Use the self-contained Python files bundled in this repo.

**Repo:** [huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat)

```bash
# Latest v2 round (recommended — main is the default branch)
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat \
  --local-dir qllm-pam-v11-e3k3-chat
cd qllm-pam-v11-e3k3-chat
pip install -r requirements.txt   # or: uv sync && uv run ...

# Recommended chat (--no-think, cap length for factual Q&A)
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt --no-think --max_new_tokens 128
```

| File | Purpose |
| ---- | ------- |
| `qllm_v11_e3k3_chat.pt` | SFT chat weights (~384 MB) |
| `qllm_v11_e3k3_pretrain.pt` | Pretrain base weights (~384 MB) — prefix completion |
| `config.json` | SFT architecture metadata |
| `config_pretrain.json` | Pretrain architecture metadata |
| `modeling_qllm.py` | Self-contained model (no qllm2 clone) |
| `run_chat.py` | Interactive / single-prompt **SFT chat** |
| `run_complete.py` | Prefix **completion** (pretrain base) |
| `eval_chat.py` | Batch SFT chat eval |
| `eval_compare.py` | Pretrain vs SFT comparison (4 answers per question) |
| `eval_prompts_round1.yaml` | SFT chat prompt suite |
| `eval_prompts_compare.yaml` | Paired prefix + chat questions |
| `SAMPLES_round-8b-gate.md` | Full SFT chat sample log |
| `SAMPLES_round-8b-gate-compare.md` | Pretrain vs SFT summary table |
| `requirements.txt` | `torch`, `transformers`, `PyYAML` |

One `modeling_qllm.py` serves **both** checkpoints. Use `run_chat.py` for SFT (ChatML) and `run_complete.py` for the pretrain base (plain prefix continuation).

```bash
# Legacy ~10B milestone (comparison / older stack only)
huggingface-cli download gowravvishwakarma/qllm-pam-v11-e3k3-chat \
  --revision v1-old-deprecated-10B-sft --local-dir ./legacy
python run_chat.py --checkpoint legacy/qllm_v11_e3k3_chat.pt --max_new_tokens 64
```

### Interactive multi-turn chat

```bash
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt --no-think --max_new_tokens 128
```

| Input | Action |
| ----- | ------ |
| Type a message + Enter | Send; model sees full conversation history |
| Empty line (Enter only) | Start a **new chat** — history cleared |
| `exit` / Ctrl+C / EOF | Quit |

Multi-line paste is supported: paste a block and the script waits briefly to absorb trailing lines as one message.

| Flag | Default | Effect |
| ---- | ------- | ------ |
| `--no-think` | off | Discourages `<think>`; **strips** thinking blocks from reply and history |
| `--max_new_tokens` | 256 | Cap generation length (use **128** or **64** for short factual answers) |
| `--temperature` | 0.7 | Sampling temperature (`0.0` = greedy) |
| `--system` / `--system-file` | built-in helper | Override the system prompt for the whole session |

### Single prompt (non-interactive)

```bash
python run_chat.py --checkpoint qllm_v11_e3k3_chat.pt \
  --prompt "What is the capital of France?" \
  --no-think --temperature 0.0 --max_new_tokens 32
```

### Pretrain prefix completion (non-chat)

The **pretrain base** is a next-token LM on web text. For factual probes, give a **prefix** (not a chat question):

```bash
python run_complete.py --checkpoint qllm_v11_e3k3_pretrain.pt \
  --prompt "The capital of France is" --temperature 0.1 --max_new_tokens 64

python run_complete.py --checkpoint qllm_v11_e3k3_pretrain.pt \
  --prompt "Karl Marx was" --temperature 0.7 --max_new_tokens 80
```

Same sampling defaults as training `gen_every`: top_k 50, top_p 0.9, repetition_penalty 1.2.

---

## Pretrain base vs SFT chat

SFT teaches ChatML instruction-following but can **wash out** raw factual completions the pretrain base still has. Example: prefix `"The capital of France is"` often yields **Paris** from pretrain, while the SFT chat model may ramble or hallucinate on the same fact as a question.

**Comparison doc:** [SAMPLES_round-8b-gate-compare.md](SAMPLES_round-8b-gate-compare.md) — one table row per question, four columns:

| | pretrain@0.1 | pretrain@0.7 | sft@0.1 | sft@0.7 |
|--|--|--|--|--|

Full untruncated answers live in the JSON log.

Reproduce:

```bash
cd hf_release
uv run python eval_compare.py \
  --pretrain qllm_v11_e3k3_pretrain.pt \
  --sft qllm_v11_e3k3_chat.pt \
  --prompts eval_prompts_compare.yaml \
  --round-tag round-8b-gate \
  --out-md SAMPLES_round-8b-gate-compare.md \
  --out-json ../logs/v11/round-8b-gate_compare.json
```

SFT-only chat eval (profiles recommended/raw) remains in [eval_chat.py](eval_chat.py) — see [SAMPLES_round-8b-gate.md](SAMPLES_round-8b-gate.md).

---

### Minimal inference snippet

For day-to-day chatting, use `run_chat.py` above; the snippet below is for custom integrations.

```python
import torch
from transformers import AutoTokenizer
from modeling_qllm import load_model

IM_START, IM_END = "<|im_start|>", "<|im_end|>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("qllm_v11_e3k3_chat.pt", device=device)

THINK_START, THINK_END = "<think>", "</think>"

tok = AutoTokenizer.from_pretrained("gpt2")
extras = [IM_START, IM_END]
if model.config.vocab_size >= 50261:
    extras.extend([THINK_START, THINK_END])
tok.add_special_tokens({"additional_special_tokens": extras})
assert len(tok) == model.config.vocab_size

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

reply = tok.decode(out[0, ids.shape[1]:].tolist()).split(IM_END, 1)[0]
while THINK_START in reply:
    start = reply.find(THINK_START)
    end = reply.find(THINK_END, start + len(THINK_START))
    reply = reply[:start] + (reply[end + len(THINK_END):] if end != -1 else "")
print(reply.strip())
```

Generation uses the **O(1) recurrent PAM path** — a fixed matrix state per layer, updated one token at a time. No KV cache, no growing memory.

---

## Sample chats

**Full transcripts:** [SAMPLES_round-8b-gate.md](SAMPLES_round-8b-gate.md) (72 generations, recommended + raw profiles)

| Prompt | Response (truncated) | Notes |
| ------ | -------------------- | ----- |
| what is the capital of France? | The capital of France is Paris… | Paris hit; cleaner than round-6b |
| What is 2+2? | To find the value of 2, we need to first identify… | Still fails (math not in SFT) |
| Answer in one word: is water wet? | Yes, yes. The sentence "Is water wet?"… | Starts yes; still over-explains |
| Say hello in a friendly way. | Hello! How can I help you today? | Chat/social OK |
| What is the capital of the United States? | Washington, D.C.… | **Improved** vs round-6b |

At **8B pretrain tokens** with **no-think / no-math SFT**, treat this as a **recipe experiment** on the same architecture. Pin **`round-6b-gate`** on HF to compare against the prior think-SFT line.

Reproduce the full eval:

```bash
uv run python eval_chat.py \
  --checkpoint qllm_v11_e3k3_chat.pt \
  --prompts eval_prompts_round1.yaml \
  --round-tag round-8b-gate \
  --out-md SAMPLES_round-8b-gate.md \
  --out-json ../logs/v11/round-8b-gate_chat_eval.json
```

---

## Training recipe

**Current v2 line (from scratch, +2B per round):**

1. **Pretrain from scratch** — preset `v11_e3_k3_chat` (vocab **50261**), blended DCLM-Edu + FineWeb-Edu + smoltalk2-Mid (**thinking traces stripped** from Mid since round-8b), **2B tokens per round**, content-aware GSP gate.
2. **SFT** — `HuggingFaceTB/smoltalk2` SFT config, **`think_fraction=0`**, math/reasoning splits dropped, hard filter, 1 epoch, ChatML assistant-only loss.
3. **Ship** — export SFT + pretrain bundles, verify, push under a **round tag** and update **`main`** (e.g. `round-8b-gate`).

**Legacy tag `v1-old-deprecated-10B-sft` (reference only):** ~10B DCLM-Edu + FineWeb-Edu pretrain (vocab 50259), then older SmolTalk SFT with `--warmstart_chatml`.

**ChatML template:**

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

Full reproduction: [v11/EXPERIMENTS_V11.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/EXPERIMENTS_V11.md). Per-round provenance: [v11/MODEL_RELEASES.md](https://github.com/gowrav-vishwakarma/qllm2/blob/master/v11/MODEL_RELEASES.md).

---

## Round history

| Revision | Shipped | Pretrain | Metrics | Notes |
|----------|---------|----------|---------|-------|
| `v1-old-deprecated-10B-sft` | (legacy) | ~10B | val PPL ~7.2 | Pre-v2 — vocab 50259, magnitude gate; comparison only |
| `round-2b-gate` | 2026-07-04 | 2B | SFT PPL **7.20**, pretrain holdout **35.42** | Round 1, v2 line (frozen tag) |
| `round-4b-gate` | 2026-07-06 | 4B | SFT PPL **6.65**, acc **0.594** | Round 2 (frozen tag) |
| `round-6b-gate` | 2026-07-09 | 6B | SFT PPL **6.55**, acc **0.594** | Round 3, think-SFT (frozen tag) |
| **`round-8b-gate` / `main`** | **2026-07-10** | **8B** | SFT PPL **7.88** | **Round 4, no-think / no-math SFT, latest** |

**Round 4 pretrain (+2B):** continues from `round-6b-gate` base; cumulative **8B** tokens. Pretrain Mid blend now **strips `<think>`** blocks (cache v2). SFT uses **`think_fraction=0`** and drops math/reasoning/multilingual splits — see [Round 4 recipe change](#round-4-recipe-change-no-think-no-math-sft) above.

**Round 1 pretrain mix (~2B tokens):** warmup first 1B on DCLM-Edu + FineWeb-Edu; remaining ~1B adds smoltalk2 Mid (weights 48:48:4 → ~52% DCLM-Edu, ~40% FineWeb-Edu, ~8% smoltalk2 Mid).

<!-- Append one row per shipped round. -->

---

## Next milestone

The next goal is **not** "train longer" or "scale to 7B first."

1. **Make training ~10× faster** — target **~10% of current wall-clock** for the same quality curve.
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
