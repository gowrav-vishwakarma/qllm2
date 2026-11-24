---
license: mit
language:
- en
tags:
- phase-associative-memory
- delta-rule
- o1-inference
- attention-free
- non-transformer
- qllm
- pam
pipeline_tag: text-generation
---

# SCI-1 (PAM-2, 386M)

**A 386M-parameter recurrent language model that learns a rule from examples in
its prompt and still applies it when those examples are spread over 2× and 4×
the length it was trained on, and recalls a stated fact 8× past it (0.78) —
with a fixed 3.5 MiB memory and no KV cache.**

SCI-1 is built on Phase-Associative Memory (PAM-2, the real-valued arm of
qllm2 `v13_sempty`). It is not a transformer and not Mamba. There is no
attention. The whole past is held in seven fixed-size matrix memories (6
delta-rule notebooks + 1 echo memory, each 8 heads × 128 × 128): **3.5 MiB of
state per sequence (fp32) at any context length**, and every new token costs
the same.

It is a base model whose pretraining mix includes chat data (ChatML), so it
completes plain text and holds a ChatML conversation. It is a research
checkpoint, not a polished assistant. The chat-tuned (SFT) version is
[qllm-pam2-sci1-386m-chat](https://huggingface.co/gowravvishwakarma/qllm-pam2-sci1-386m-chat).

| | |
|--|--|
| Parameters | 385,817,038 (tied input/output embedding) |
| Training | 749,500 steps, **49.12B tokens**, sequence length **4096** |
| Vocabulary | 50,261 = GPT-2 BPE + 4 added tokens: the ChatML start and end tokens and the think tags (ids 50257-50260; exact strings under Tokenizer below) |
| Weights | `sci1_386m_49b.pt` (fp32, 1.54 GB; `config` + `model_state_dict` + step/tokens) |
| State | 3.5 MiB per sequence, independent of context length |

Related: the earlier model in this line,
[qllm-pam-v11-e3k3-chat](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat)
(complex-valued PAM, V11) · code: [github.com/gowrav-vishwakarma/qllm2](https://github.com/gowrav-vishwakarma/qllm2)
(this model: `v13_sempty/`, release notes `v16/`) · PAM paper:
[arXiv:2604.05030](https://arxiv.org/abs/2604.05030).

## Key result: rules learned in context, beyond the training length

The model sees 8 worked examples of a rule (a symbol shift by k, or an affine
map a·x+b), then an unseen input, and must produce the answer. The 8 examples
are spread out with filler text, so the evidence has to be gathered across the
whole span. The model was trained only on 4,096-token windows.

What is and is not new to the model: these two rule *families* are in the
training mix as worked-example documents (as are web, books, code and chat).
The test uses rule *values* held out from training, on new inputs, and the
answer a memorised neighbouring rule would give is always one of the 8
candidates, so picking a remembered rule scores ~0. This measures inferring
a rule in context, not recalling one. Open-ended few-shot prompts in other
formats are much weaker at this size (see Limitations).

8 candidates, chance = 0.125; n = 400 per cell (seeds 7000–7399). Measured on
this exact checkpoint (sha256 `b403ad63…`).

| examples spread over | × training length | symbol shift | affine map |
|--|--|--|--|
| 0 (adjacent) | – | 1.00 | 1.00 |
| 2,048 tokens | 0.5× | 0.98 | 1.00 |
| 4,096 tokens | 1× | 0.97 | 1.00 |
| **8,192 tokens** | **2×** | **0.94** | **0.99** |
| **16,384 tokens** | **4×** | **0.69** | **0.89** |
| 32,768 tokens | 8× | 0.20 | 0.58 |

So the rule holds almost fully at 2×, mostly at 4×, and fades by 8× (still
above chance). For contrast, our Transformer++ baseline (~103M, equal data)
scores 0.05–0.13 on this rule task even inside 2k–4k, and 0.08–0.13 on
recall at 16k (table below).

**Exact recall past the window** (key–value pairs in a long context, then a
query; 8 candidates, averaged over positions and 1/4/8 pairs; n = 360 per
length, 40 per position × pair-count cell):

| context | 128–2k | 4k (1×) | 8k (2×) | 16k (4×) | 32k (8×) |
|--|--|--|--|--|--|
| accuracy | 1.00 | 0.99 | 0.98 | 0.94 | 0.78 |

At 32k it depends on where the fact sits: 0.58 at the start of the
context, 0.77 in the middle, 1.00 near the end.

What does **not** carry this far: a fact written in prose and asked in natural
words (see Limitations).

## Standard benchmarks (zero-shot, this checkpoint)

| | HellaSwag | ARC-easy | PIQA |
|--|--|--|--|
| **SCI-1, 386M, 49B tokens** | 0.372 | 0.559 | 0.636 |
| Pythia-410M at 10.5B tokens (reference) | 0.297 | 0.434 | 0.607 |
| Pythia-410M final, 300B tokens (reference) | 0.406 | 0.519 | 0.672 |

Zero-shot, full validation/test sets (10,042 / 2,376 / 1,838), one metric for
every row: HellaSwag and PIQA length-normalised accuracy (acc_norm), ARC-easy
plain accuracy. Plain accuracy for SCI-1: HellaSwag 0.320, PIQA 0.638. Harness:
`scripts/mc_eval.py`, matching lm-eval 0.4.9.1 to 4 decimals. Pythia rows are
measured on the same harness but trained on different data (the Pile): a
reference, not a controlled comparison. SCI-1 is above Pythia-410M at 10.5B
tokens on all three, and above its final 300B checkpoint on ARC-easy only.

## Against Transformer++ and Mamba-2 (same recipe, ~103M, equal data)

We have **no Transformer++ or Mamba-2 at 386M**. The controlled comparisons
were run at ~103M parameters: every model on identical tokens, same code,
same tokenizer, at several data budgets. They show what the PAM-2 recipe
does relative to the two baselines; they are not a comparison with this
checkpoint.

| at ~103M, equal data | PAM-2 (ours) | Transformer++ | Mamba-2 |
|--|--|--|--|
| rule from 8 examples, 2k–4k (1.0B tokens) | **0.52** | 0.07 | 0.15 |
| same, 600M tokens (3 seeds for ours) | **0.65 / 0.69 / 0.74** | 0.13 | 0.10–0.12 |
| same, 1.5B tokens | **0.58** | 0.05 | 0.03 |
| fact recall at 16k = 4× window (1.0B tokens) | **0.41** | 0.13 | 0.15 |
| fact recall at 16k, rule documents in everyone's data (3 seeds for ours) | **0.79–0.87** | 0.08 | 0.42 |
| HellaSwag / ARC-e / PIQA (1.0B tokens) | 0.28 / 0.46 / 0.59 | 0.28 / 0.46 / 0.59 | 0.28 / 0.46 / 0.59 |
| six-benchmark mean (3B tokens) | 0.403 | 0.406 | 0.415 |
| held-out language loss (1.0B tokens; lower is better) | 3.43 | 3.44 | **3.38** |

Plainly:
* On standard benchmarks the three are level. Mamba-2 has the best language loss.
* PAM-2's lead is in-context rules and long-distance recall; Transformer++
  collapses past its window.
* Gated DeltaNet, the closest published design, matches PAM-2 on the rule
  task in the same setting (0.41 vs 0.34 at 8k on the far-docs mix).
* Rule scores at this size vary a lot between seeds (0.28–0.84 on one cell),
  so we show every seed we ran.

## Selected examples

Picked by hand from the generations in `SAMPLES.md` (all made by
`make_samples.py` with `modeling_pam2.py` only, CPU fp32). They are the
better ones, not typical ones; `SAMPLES.md` has the full set with every
setting.

**Greedy continuation** — prompt `Photosynthesis is the process by which`:

```text
Photosynthesis is the process by which plants convert light energy into chemical energy in the form of glucose. It is the main source of energy for all living organisms.

The process of photosynthesis is a series of chemical reactions that take place in the chloroplasts of plant cells. The main chemical reactions that occur in the chloroplasts are:

1. Light energy is converted into chemical energy in the form of ATP (
```

**Greedy ChatML chat, 3 turns, state carried between turns:**

```text
user:      What is the capital of France?
assistant: The capital of France is Paris.
user:      What is it famous for?
assistant: France is famous for its art, architecture, and cuisine. It is also known for its wine country, which is home to some of the world's most famous wineries.
```

## Quick start

Only `torch` and `transformers` (for the GPT-2 tokenizer) are needed. The
qllm2 repo is not needed.

```bash
pip install -r requirements.txt
python run_complete.py --prompt "In 1923, the University of"            # sampled continuation
python run_complete.py --prompt "The capital of France is" --temperature 0   # greedy
python run_chat.py                                                      # interactive ChatML chat
python run_chat.py --prompt "What is the capital of France?"            # one turn, then exit
bash verify.sh                                                          # end-to-end smoke check
```

From Python:

```python
import torch
from modeling_pam2 import load_model, load_tokenizer

model = load_model("sci1_386m_49b.pt", device="cpu")    # or "cuda"; dtype=torch.bfloat16 is supported
tok = load_tokenizer()                                   # gpt2 + the 4 added special tokens (ids 50257-50260)
end = tok.convert_tokens_to_ids("<|im_end|>")
prompt = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
ids = torch.tensor([tok.encode(prompt)])
out = model.generate(ids, max_new_tokens=64, temperature=0.0, eos_token_id=end)
print(tok.decode(out[0, ids.shape[1]:]))
```

`model(ids, state)` returns `(logits, state)`. Pass the state back in to
continue the same sequence: the prompt is never re-read, so a chat session
feeds only each turn's new tokens (`run_chat.py` works this way).

## Tokenizer

The tokenizer is built exactly as in training (qllm2 `v7/data.get_chat_tokenizer`):
`AutoTokenizer.from_pretrained("gpt2")` + `add_special_tokens({"additional_special_tokens":
["<|im_start|>", "<|im_end|>", "<think>", "</think>"]})`, giving ids 50257, 50258,
50259, 50260. `<|endoftext|>` (50256) separates documents (and is the pad token).
At 50256 the model **resets** its memories (`reset_at_eos`). `load_tokenizer()`
checks these ids. Chat format (ChatML):

```text
<|im_start|>system\n{system}<|im_end|>\n        (optional)
<|im_start|>user\n{message}<|im_end|>\n
<|im_start|>assistant\n{reply}<|im_end|>\n
```

A reply ends at `<|im_end|>`.

## Architecture

```text
embed -> RMSNorm
24 pre-norm blocks:   x += cgu_scale * SwiGLU(RMSNorm(x))          (all blocks, hidden 3704)
  blocks 3,7,11,15,19,23:  x += pam_scale * DeltaNotebook(RMSNorm(x))
  other blocks above 3:     x += SharedRead(RMSNorm(x), nearest notebook below)
hidden = RMSNorm(x);  lm = RMSNorm(W hidden + b) + Echo(hidden, tokens);  logits = lm . embed^T
```

* **Delta notebook** (per head, K = 128):
  `S_t = g_t S_{t-1} (I - b_e k_t k_t^T) + b_w v_t k_t^T`, with read `y_t = silu(W_gate x) * S_t q_t / sqrt(K)`.
  The keys are unit vectors. Retention is `g = exp(-softplus(.))`, the write gate is
  `b_w = sigmoid(.)` and the erase gate is `b_e = min(sigmoid(. - 3), 0.95)`.
* **Chrono clock**: rotary position with a content-driven speed per head.
  Each head's step is `w_t = exp(clamp(W x, ±3))`, its clock is `tau_t = sum_{j<t} w_j`,
  and the RoPE angle is `tau_t / 10000^(i/64)`.
* **Shared reads**: each block without a notebook reads the nearest notebook
  below it. It uses its own query (rotated by the writer's clock), its own gate
  and its own output projection.
* **Echo memory**: a delta memory at the top of the stack. Its key is the previous
  hidden state and its value is the current token's embedding, which gives a
  learned copy / induction channel into the tied head.

`modeling_pam2.py` holds the full implementation (pure PyTorch, about 700 lines). It has
two execution forms of the same algebra. `forward` is the chunked closed form
(64-token chunks, delta rule turned into an additive scan by a WY
triangular solve) for the prompt. `step` is the plain per-token recurrence for
decoding.

## Files

| File | |
|--|--|
| `modeling_pam2.py` | model, state, `generate()` (greedy / temperature / top-k / top-p / repetition penalty), loader, tokenizer helper |
| `sci1_386m_49b.pt` | weights (`config`, `model_state_dict`, `global_step`, `global_tokens`). Loads with `torch.load(weights_only=True)` |
| `config.json` | architecture fields + tokenizer + training metadata + weights sha256 |
| `run_complete.py`, `run_chat.py` | text continuation and multi-turn ChatML chat |
| `SAMPLES.md`, `make_samples.py` | sample generations (greedy and sampled) and the script that made them |
| `PARITY.md`, `test_parity.py` | parity of this file against the original qllm2 model (needs the qllm2 repo) |
| `export_weights.py` | how `sci1_386m_49b.pt` was made from the training checkpoint |
| `verify.sh` | end-to-end smoke test |

## Parity with the training code

`test_parity.py` loads the same weights into the qllm2 repo model and into
`modeling_pam2.py` (CPU, fp32). The prompt forward matches to ≤ 2.4e-4 max
|logit difference|, with 100 % argmax agreement on 512-token inputs. The
per-token decode matches the repo's decode path bit for bit, and 40-token
greedy generations are identical. Details are in `PARITY.md`.

## Speed (reference implementation, batch 1)

Measured 2026-11-24 with plain PyTorch (no Triton, no compile).

| device | prefill | decode |
|--|--|--|
| CPU (16 threads, fp32) | ~790 tok/s (512 and 2048 tokens) | 36 ms/token (~28 tok/s) |
| RTX 4090, fp32 | ~6,900 tok/s (2048 / 4096 tokens) | 8.2 ms/token (~120 tok/s), 2.9 GB peak |
| RTX 4090, bf16 | ~9,600 tok/s | 6.8 ms/token (~150 tok/s), 1.5 GB peak |

Decode cost and memory do not grow with context. bf16 weights change the logits
(argmax agreement 97.5 % vs fp32 on a 512-token text). fp32 is the exact,
parity-tested setting.

## Limitations

* A research checkpoint trained on 49B tokens. Expect mistakes, made-up facts
  and repetition, especially with greedy decoding on open-ended prompts.
* **No binding of a fact to its owner.** With a same-type distractor (for example "Anna
  lives in Paris … Ben lives in Rome … where does Anna live?"), the
  distractor wins about as often as the right answer, even 64 tokens later
  (0.32 right vs 0.34 lure; 8 choices, n = 200). Same as at 100M.
* **Prose facts are short-range.** A fact stated in natural prose and asked in
  natural words is recalled at 0.47 at 64 tokens and is near chance (0.13–0.21)
  from 2k on, inside the training window. The long-range recall above holds
  for a literal key next to its value, not for prose.
* **Told rules over-apply.** A rule stated in words is carried (to 8k), but at
  short range it is often applied to the wrong quantity.
* Rules fade by 8× the training length (0.20 / 0.58 at 32k).
* **Free-form few-shot is weak.** On hand-written few-shot prompts outside the
  trained rule formats (reverse a word, uppercase, first letter, translate),
  greedy decoding mostly fails; in our quick try, 1 of 7 was right
  (`3 -> 7, 5 -> 11, ... 6 ->` gives `13`).
* `modeling_pam2.py` supports only this checkpoint's configuration; other qllm2
  options raise `NotImplementedError`.

## Citation

```bibtex
@misc{vishwakarma2026sci1,
  title  = {SCI-1: a 386M Phase-Associative Memory language model},
  author = {Vishwakarma, Gowrav},
  year   = {2026},
  url    = {https://huggingface.co/gowravvishwakarma/qllm-pam2-sci1-386m}
}
```

Earlier PAM work: [arXiv:2604.05030](https://arxiv.org/abs/2604.05030).
