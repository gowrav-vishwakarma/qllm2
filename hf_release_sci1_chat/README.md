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
- chat
base_model: gowravvishwakarma/qllm-pam2-sci1-386m
pipeline_tag: text-generation
---

# SCI-1 Chat (PAM-2, 386M)

**The chat-tuned version of [SCI-1](https://huggingface.co/gowravvishwakarma/qllm-pam2-sci1-386m): a 386M recurrent
language model with no attention and no KV cache. It keeps the base model's
long-context skills — a rule learned from 8 examples spread over 2× its
4,096-token training window (0.995), a fact recalled 16k tokens back (0.91) —
with a fixed 3.5 MiB memory per conversation.**

It is a short supervised fine-tune (SFT) of the SCI-1 base on English chat
data. It answers more directly and stops cleanly. It is still a small
research model: it gets arithmetic and many facts wrong (see Limitations).

| | |
|--|--|
| Parameters | 385,817,038 (tied input/output embedding) |
| Base | SCI-1 386M, step 749,500, 49.12B tokens |
| SFT | +0.25B tokens (0.20B constant LR 2e-4, then 0.05B linear decay to 0); step 753,315, 49.37B tokens total |
| Vocabulary | 50,261 = GPT-2 BPE + 4 added tokens: the ChatML start and end tokens and the think tags (ids 50257-50260; exact strings under Tokenizer below) |
| Weights | `sci1_chat_386m.pt` (fp32, 1.54 GB) |
| State | 3.5 MiB per sequence, independent of context length |

Related: base model [qllm-pam2-sci1-386m](https://huggingface.co/gowravvishwakarma/qllm-pam2-sci1-386m) ·
earlier line [qllm-pam-v11-e3k3-chat](https://huggingface.co/gowravvishwakarma/qllm-pam-v11-e3k3-chat) (V11) ·
code [github.com/gowrav-vishwakarma/qllm2](https://github.com/gowrav-vishwakarma/qllm2) (`v13_sempty/`) ·
PAM paper [arXiv:2604.05030](https://arxiv.org/abs/2604.05030).

## SFT recipe

* **Data:** [HuggingFaceTB/smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2), SFT config (unseen by the base),
  one pass, no repeats. Kept mix (measured): smol-magpie-ultra 41 %, OpenHermes 2.5 34 %, smol-summarize 12 %,
  systemchats-30k 6 %, smol-rewrite 4 %, tulu-3 personas IF 3 %. Left out: multilingual, long-context, tool /
  function-calling and agent splits, and reasoning ("think") traces; capped: openthoughts3 5 %, mixture-of-thoughts 10 %,
  table-gpt 20 %.
* **Decontamination:** conversations that share a 13-gram with any reported benchmark, or with our own probe
  prompts, were dropped.
* **Loss:** only on assistant tokens and the closing `<|im_end|>`. Conversations are separated by `<|endoftext|>`, where
  the model resets its memory.
* **Replay:** about 20–30 % of training windows were the base's own pretraining mix (including its long-range rule
  documents), so the base skills are rehearsed.
* **Schedule:** 65,536 tokens per step, warm-up 200 steps to 2e-4, constant to 0.20B tokens, then linear to 0 over 0.05B.
  Optimizer state carried over from the base.

## Did SFT keep the base skills? (pre-registered guards)

Same scoring code for both checkpoints; chance 0.125 on the 8-way probes.

| | SCI-1 Chat | SCI-1 base | guard |
|--|--|--|--|
| rule from 8 examples spread over 8k (2×) | **0.995** | 0.964 | ≥ 0.864 |
| exact recall 16k back (4×) | **0.908** | 0.942 | ≥ 0.892 |
| exact recall 8k back (2×) | 0.925 | 0.978 | reported |
| skill-pack rule at 8k | 0.905 | 0.645 | reported |
| harder rule task at 8k | 0.93 | 0.90 | reported |
| HellaSwag / ARC-easy / PIQA (acc_norm / acc / acc_norm) | **0.380 / 0.575 / 0.651** | 0.372 / 0.559 / 0.636 | reported |
| told rule at 8k, applied wrongly (P_fu / scope_fu; lower is better) | 0.01 / 0.01 | – | ≤ 0.20 / 0.30 |

Recall dropped a little (0.94 → 0.91 at 16k); everything else held or rose. One run, one seed, and the stage
ends in a learning-rate decay, which raised these probes at every earlier SCI-1 decay — so we do not credit
the rises to the chat data.

## Chat quality

* Ends its turn with `<|im_end|>` in 97.8 % of answers (60 prompts × 3 decodes, 768-token budget); base the same.
* No tool-call text in normal answers (the base produced 2); no non-English answers.
* **Blind A/B vs the base** on 60 prompts (greedy): SFT preferred **24**, base 8, tie 14, both bad 14 — a clear
  majority of the 32 decided prompts, but 24 of 60 overall (28 had no winner). The judge
  was a larger language model (Claude) reading both answers without knowing which was which — not a human
  panel. SFT won mostly on everyday questions (19 vs 2); the base was better on 4 of 6 technical Q&A.
* **All 60 prompts, base vs SFT answers side by side, copied exactly as generated:
  [BASE_VS_SFT.md](BASE_VS_SFT.md).**

## With the package: long texts

On its own the model holds a fixed 3.67 MB state, so over a long text it keeps rules and gist but loses most
exact facts. The **package** adds a small text library of the same text. At question time the library fetches the
sentence(s) the question is about, and the model reads them next to the question on its state. The state stays
3.67 MB; the library grows with the text: about 0.4 MB of text and word index at 32k, plus a 33.8 MB
int8 address that is used only when the text match misses.

Measured on these weights, model alone ("bare") → with the package. Full tables, method and caveats are in
**[PACKAGE.md](PACKAGE.md)**.

| test (documents per cell) | bare | package |
|--|--|--|
| recall at 32k, free answer: needle among 8 / latest of several updates / two-hop (20) | 0.05 / 0.25 / 0.05 | **1.00 / 1.00 / 0.90** |
| a rule shown earlier applied to a fetched value (R × K), 2k / 8k (100) | 0.00 / 0.00 | **0.97 / 0.85** |
| a told unit rule compiled next to the question, 2k / 8k (100) | 0.03 / 0.00 | **0.96 / 0.95** |
| RULER-style needle, single / multi-key, 32k (50; our generator, not the official suite) | 0.02 / 0.00 | 0.20 / 0.30 |
| BABILong qa1 (one supporting fact), 2k / 32k (100) | 0.10 / 0.00 | 0.52 / 0.12 |
| BABILong qa5 (three-argument relations), 2k / 32k (100) | 0.22 / 0.06 | 0.44 / 0.29 |

* **The fetch is good; the read is the weak part.** On the needle tests the library fetched the right sentence on
  every document. The model then garbles uuids, cuts long numbers, or stops after writing ":" and a newline. On
  BABILong the answer word was in the fetched rows on 0.91 (qa1) / 0.79 (qa5) of documents, but combining
  several rows is where the 386M model fails. qa3 / qa4 (three facts, two-argument relations) stay ≤ 0.12.
* **What this folder's runner does:** `run_package.py` has the fetch-and-read path (text matches, then the
  address). R × K, the held-rule "twin", BABILong's multi-sentence hops and the planner chain are research-repo
  paths (`v16/package.py`, `v14/`) and are **not** in this port. The port matches the repo on 20 / 20 test items,
  field by field ([PACKAGE.md](PACKAGE.md), "Parity").
* **This chat model adds details:** on a short note it may invent a pet's name or a year next to the right fact.
  Check answers against the text.

## SCI-1 Chat vs SCI-1 base, side by side

Same weights as published, same scoring code. Higher is better unless marked.

| | SCI-1 base | SCI-1 Chat |
|--|--|--|
| rule from 8 examples, spread over 8k (2×) | 0.964 | **0.995** |
| exact recall 16k back (4×) | **0.942** | 0.908 |
| HellaSwag / ARC-easy / PIQA | 0.372 / 0.559 / 0.636 | **0.380 / 0.575 / 0.651** |
| ends its answer with `<|im_end|>` (60 × 3 decodes) | 97.8 % | 97.8 % |
| blind A/B, 60 chat prompts (judge: a larger LM) | 8 wins | **24 wins** (14 ties, 14 both bad) |
| BABILong qa1-qa5, short input (0k), model alone | 0.00 - 0.20 | **0.18 - 0.62** |
| BABILong at 32k with the package: qa1 / qa2 / qa3 / qa4 / qa5 | 0.10 / 0.13 / 0.06 / 0.01 / 0.18 | 0.12 / 0.09 / 0.01 / 0.01 / **0.29** |
| R × K with the package, 2k / 8k | 0.97 / 0.78 | 0.97 / **0.85** |
| RULER-style needle with the package, 8k single / multi-key | **0.44 / 0.62** | 0.38 / 0.40 |
| RULER-style needle with the package, 32k single / multi-key | **0.32 / 0.40** | 0.20 / 0.30 |
| recall at 32k with the package, two-hop free / 20-way | **0.95 / 1.00** | 0.90 / 0.95 |
| state per conversation | 3.67 MB | 3.67 MB |

Short version: Chat answers short questions much better and keeps the long-range rule skill; the base reads raw
values (numbers, uuids, needle words) out of a fetched sentence more reliably, so for pure lookup over long
documents the base with the package is the better choice. One run each, no seeds repeated; differences of a few
hundredths on 50-100 documents are within noise.

## Selected examples

Picked from the A/B set (greedy, no repetition penalty; "..." marks cuts): they are the better answers, not
typical ones. The full set, failures included, is in [BASE_VS_SFT.md](BASE_VS_SFT.md); `SAMPLES.md` has
generations made with this folder's scripts.

```text
user:      How many legs does a spider have?
assistant: A spider has eight legs, ...

user:      Translate "good morning" into French.
assistant: "Good morning" in French is "Bonjour".

user:      What causes the seasons on Earth?
assistant: The Earth's seasons are caused by the tilt of its axis, ... tilted at an angle of about 23.5 degrees, ...

user:      What is the unit of electrical resistance?
assistant: The unit of electrical resistance is the ohm (Ω).
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
python run_package.py --demo                                            # 32k needle: model alone vs the package
python run_package.py --context notes.txt --question "..." --form chat  # ask your own long text
bash verify.sh                                                          # end-to-end smoke check
```

From Python:

```python
import torch
from modeling_pam2 import load_model, load_tokenizer

model = load_model("sci1_chat_386m.pt", device="cpu")    # or "cuda"; dtype=torch.bfloat16 is supported
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
| `sci1_chat_386m.pt` | weights (`config`, `model_state_dict`, `global_step`, `global_tokens`). Loads with `torch.load(weights_only=True)` |
| `config.json` | architecture fields + tokenizer + training metadata + weights sha256 |
| `run_complete.py`, `run_chat.py` | text continuation and multi-turn ChatML chat |
| `SAMPLES.md`, `make_samples.py` | sample generations (greedy and sampled) and the script that made them |
| `BASE_VS_SFT.md` | the 60 A/B prompts: base and SFT answers side by side |
| `package_lite.py`, `run_package.py` | the package (text library + fetch + read) and its runner: `--mode bare / package / both` |
| `PACKAGE.md` | how the package works, its measured results (suite, RULER-style, BABILong), memory and parity |
| `bg_wordcounts_fineweb014.json` | background word counts used by one fetch step (`text4`); derived from FineWeb, ODC-By 1.0 |
| `PARITY.md`, `test_parity.py` | parity of this file against the original qllm2 model (needs the qllm2 repo); passes on these SFT weights |
| `export_weights.py` | how `sci1_chat_386m.pt` was made from the training checkpoint |
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

* A small research model. Its answers are fluent but often wrong in the details.
* **Arithmetic fails:** in the A/B set it got "15 % of 80", a concentration, and cell doubling wrong.
* **Facts:** e.g. it says the capital of Australia is Melbourne. It wrote a Celsius-to-Kelvin function with the
  Fahrenheit formula.
* **Instructions held across turns are weak:** it ignored "answer only YES or NO", and when asked "what is my name?"
  it answered as if it were the user.
* **Prose facts are short-range** (from the base): a fact stated in prose and asked in natural words is recalled
  near chance beyond ~2k tokens; the long-range recall above is for a literal key next to its value.
* **With the package**, long-text lookup works when the question shares a key or phrase with the answer
  sentence; reading the value out is the weak step (uuids, long numbers), and multi-fact BABILong tasks stay low.
* No tool calling, no multilingual chat, no reasoning traces (left out of SFT on purpose).
* `modeling_pam2.py` supports only this checkpoint's configuration.

## Citation

```bibtex
@misc{vishwakarma2026sci1chat,
  title  = {SCI-1 Chat: a 386M Phase-Associative Memory chat model},
  author = {Vishwakarma, Gowrav},
  year   = {2026},
  url    = {https://huggingface.co/gowravvishwakarma/qllm-pam2-sci1-386m-chat}
}
```

Earlier PAM work: [arXiv:2604.05030](https://arxiv.org/abs/2604.05030).
