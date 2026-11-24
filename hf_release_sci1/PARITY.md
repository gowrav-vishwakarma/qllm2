# Parity: modeling_pam2.py vs the qllm2 training code

`test_parity.py` loads `sci1_386m_49b.pt` (exported from `final_step_749500.pt`) into the
qllm2 repo model (`scripts/run_memory_behavioral._load_v13_sempty` -> `v13_sempty.model.LM`,
eager torch on CPU: `V13S_KERNEL=0 V13S_DELTA_COMPILE=0`) and into the self-contained
`modeling_pam2.PAM2LM`. Both run on CPU in fp32. Run on 2026-11-24 with torch 2.8.0:

```bash
cd <qllm2 repo> && .venv/bin/python hf_release_sci1/test_parity.py --fp64
```

## Summary

| check | result | target |
|--|--|--|
| A. prefill logits, 512-token text | max abs diff 5.4e-05, argmax 1.0 | <= 1e-3, 1.0 |
| A. prefill logits, 512 tokens with `<|endoftext|>` at 300, 380 (reset) | 4.1e-05, argmax 1.0 | <= 1e-3, 1.0 |
| A. prefill logits, 512 random ids (50256 at 100, 101, 300) | 2.4e-04, argmax 1.0 | <= 1e-3, 1.0 |
| B. per-token decode (repo token loop with carried states vs release `step()`), 512-token text | **0.0** (bit-identical) | <= 1e-3 |
| D. prefill split at 200 with carried state vs one prefill | 6.1e-04, argmax 1.0 | <= 1e-3 |
| E. greedy generation, 40 tokens, 3 prompts (repo `LM.generate` vs release `generate`) | identical ids, 3/3 | identical |
| tokenizer ids (release `load_tokenizer` vs repo `get_chat_tokenizer`) | identical, vocab 50261 | identical |

Notes:

* **Chunked prefill vs per-token decode (section C).** This is not a parity check. The
  chunked form uses the same `log(retention + 1e-6)` as the training kernels, so in effect it
  runs on retention + 1e-6. The decode recurrence uses the exact retention. The repo has
  the same prefill/decode gap: its own token loop vs its own prefill differ by 1.0e-3 at
  256 tokens on the text input, and the release token loop is bit-identical to it. In fp64,
  with that epsilon set to 1e-12, the two release forms agree to 3.9e-08. So they are the
  same algebra, and the gap is the epsilon plus fp32 round-off. Argmax still agrees 100 %
  in fp32 on every input.
* **`<|endoftext|>` in the decode path (section B, positions >= 300).** In the repo's
  one-token decode (`RealPAMLayer._stepwise_delta`) the `reset_at_eos` mask is never
  applied to the six notebooks; only the echo memory is reset. The repo's prefill and
  training do reset all of them. The release `step()` resets all of them, matching
  training. Its result agrees with the prefill (section C, and 3.9e-08 in the fp64 check on
  the same input with separators). This matters only when 50256 is fed one token at a time
  while decoding.
* GPU fp32 (`cuda`) vs CPU fp32, 512-token text: max abs diff 4.2e-04, argmax 1.0.
  bf16 weights on GPU: max abs diff 2.6, argmax 0.975 (bf16 round-off; use fp32 for exact).

## Full output

```text
checkpoint: /home/gowrav/Development/qllm2/hf_release_sci1/sci1_386m_49b.pt
torch 2.8.0+cu128, CPU fp32, threads 16, V13S_KERNEL=0 V13S_DELTA_COMPILE=0
repo model loaded (v13_sempty.model.LM, 385,817,038 params) in 4.7s
release model loaded (385,817,038 params) in 2.4s

[tokenizer]
  len 50261 (repo 50261), specials {'<|endoftext|>': 50256, '<|im_start|>': 50257, '<|im_end|>': 50258, '<think>': 50259, '</think>': 50260}, ids identical: True

[A. prefill: repo LM.forward(x) vs release forward(x) (chunked closed form)]
 512-token text  (CPU time: repo 0.99s, release 0.66s)
  release forward (chunked)                                  max|dlogit| 5.436e-05  argmax agree 1.0000  (max|logit| 32.7)  PASS
 512 tokens with <|endoftext|> at [300, 380]  (CPU time: repo 0.69s, release 0.66s)
  release forward (chunked)                                  max|dlogit| 4.101e-05  argmax agree 1.0000  (max|logit| 32.7)  PASS
 512 random ids (50256 at 100, 101, 300)  (CPU time: repo 0.68s, release 0.68s)
  release forward (chunked)                                  max|dlogit| 2.365e-04  argmax agree 1.0000  (max|logit| 36.5)  PASS

[B. decode: repo LM.forward one token at a time with carried states (the path its generate() uses) vs release step() loop]
 512-token text  (CPU time: repo 25.9s, release 18.4s)
  release forward_recurrent (token loop)                     max|dlogit| 0.000e+00  argmax agree 1.0000  (max|logit| 32.7)  PASS
 512 tokens with <|endoftext|> at [300, 380]  (CPU time: repo 22.8s, release 19.9s)
  release token loop, positions < 300 (before the first separator) max|dlogit| 0.000e+00  argmax agree 1.0000  (max|logit| 32.7)  PASS
  positions >= 300: max|dlogit| 5.271e+01  (EXPECTED to differ: the repo decode path resets only the echo memory at <|endoftext|>, not the notebooks; the release step() resets both, as the repo prefill/training does. Release step vs release/repo prefill: see C)

[C. information: the two algebras inside the release (chunked prefill vs exact per-token recurrence), fp32]
  512-token text: repo prefill vs release token loop  max|dlogit| 9.051e-03  argmax agree 1.0000
  512 tokens with <|endoftext|> at [300, 380]: repo prefill vs release token loop  max|dlogit| 3.850e-02  argmax agree 1.0000
  512 random ids (50256 at 100, 101, 300): repo prefill vs release token loop  max|dlogit| 1.654e-01  argmax agree 1.0000
  (not a parity check: the chunked form uses log(retention + 1e-6) exactly like the training kernels, i.e. retention + 1e-6, while the per-token recurrence uses the exact retention; the repo has the same prefill/decode gap. --fp64 shows the two forms agree to ~1e-8 once that epsilon is removed.)
  fp64, chunked eps 1e-06 vs token loop (512 tokens with <|endoftext|> ...): max|dlogit| 3.850e-02
  fp64, chunked eps 1e-12 vs token loop (512 tokens with <|endoftext|> ...): max|dlogit| 3.850e-08

[D. state continuation: forward(x[:200]) -> forward(x[200:], state) vs forward(x)]
  release, prefill split at 200 (state carried)              max|dlogit| 6.132e-04  argmax agree 1.0000  (max|logit| 32.7)  PASS
  carried state: 3.5 MiB (fixed, independent of context length)

[E. greedy generation, 40 new tokens, no eos stop]
  prompt 'In 1923, the University of'
    identical ids: True (40/40)  time repo 1.6s, release 1.4s
    output: ' Pennsylvania was founded. The University of Pennsylvania was the first institution to offer undergraduate and graduate courses in the United States. The University of Pennsylvania was the first institution to offer undergraduate and graduate courses in the United'
  prompt '<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n'
    identical ids: True (40/40)  time repo 1.6s, release 1.3s
    output: 'The capital of France is Paris.<|im_end|>\n<|im_start|>user\nWhat is the population of Paris?<|im_end|>\n<|im_start|>assistant\nThe population of Paris is approximately 1.7 million people.<|im_end|>\n<|im_start|>'
  prompt 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'
    identical ids: True (40/40)  time repo 1.6s, release 1.4s
    output: '    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n '

RESULT: ALL PASS
```
