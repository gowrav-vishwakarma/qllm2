# Pretrain vs SFT — round-6b-gate (diagnostic)

Generated **2026-07-09T12:54:10.231259+00:00** on `cuda`.

Same factual questions: **pretrain** completes the `prefix`; **SFT** answers in ChatML chat.
Diagnostic arm: **pretrain_chat** runs the pretrain checkpoint through the ChatML chat path.
Answers per question: `pretrain@0.1`, `sft@0.1`, `pretrain_chat@0.1`, `pretrain@0.7`, `sft@0.7`, `pretrain_chat@0.7`.
`span_hit@20`: gold span in first 20 generated tokens.
Full text: see the JSON log.

## retrieval

| id | pretrain@0.1 | sft@0.1 | pretrain_chat@0.1 | pretrain@0.7 | sft@0.7 | pretrain_chat@0.7 |
| -- | --- | --- | --- | --- | --- | --- |
| france | Paris. - The capital of the French Republic, France. - Th... | The capital of France, France, is Paris. It is located in... | <think> Okay, let's see. The question is asking which of ... | Paris, which was founded in 1448 and had a population of ... | The capital of France is Paris, and its capital is France... | <think> Okay, let's see. The question is asking which met... |
| usa_capital | the city of San Francisco. The city was founded in 1869 b... | The capital of the United States, the United States, is c... | <think> Okay, let's see. The problem is asking for the ca... | Washington, DC. The city was built in 1763 and in 1827 it... | The capital of the United States, called U.S. (U). The ca... | <think> Okay, let's see. The problem is asking for the ca... |
| shakespeare | a young girl named Juliet. She was the first to write a n... | The question of who wrote the name of the person who wrot... | <think> Okay, so I need to figure out the answer choices ... | Romeo, who was the son of a wealthy merchant and a friend... | The question of who wrote which is likely to be the most ... | <think> Okay, let's see. The question is asking which opt... |
| largest_planet | the Sun. The Sun is a giant planet that orbits the sun ev... | The largest planet in our solar system, known as the "Ech... | <think> Okay, let's see. I need to find the largest plane... | our sun. It has a mass of 1.8 times that of the Sun and h... | The largest planet in our solar system, called "Eus exopl... | <think> Okay, let's see. The question is asking which pla... |

## biography

| id | pretrain@0.1 | sft@0.1 | pretrain_chat@0.1 | pretrain@0.7 | sft@0.7 | pretrain_chat@0.7 |
| -- | --- | --- | --- | --- | --- | --- |
| karl_marx | a German philosopher and political theorist. He was born ... | Hier ist einige Ideen, die Sie in der Regel verwendet wer... | <think> Okay, let's see. The question is asking which of ... | a famous economist and a pioneer in the field of economic... | Yes, Karl<\|endoftext\|>he is a German-born and skilled p... | <think> Okay, let's see. The question is asking which met... |
| einstein | a German physicist. He was born in 1894 and died on the s... | Einstein, the famous Einstein figure, was born on August ... | <think> Okay, let's see. The question is asking which of ... | a famous physicist who had been instrumental in making Ei... | Einstein, born in 1752, was a prominent figure in the lat... | <think> Okay, let's see. The question is asking which opt... |
| newton | a pioneer in the field of physics. He was born on Decembe... | Euclid, the father of modern science and philosophy, is a... | <think> Okay, so I need to figure out which of the follow... | a famous mathematician and astronomer who believed that t... | The question of who was who was known as Newton is a comp... | <think> Okay, let's see. The question is asking which met... |

## reasoning

| id | pretrain@0.1 | sft@0.1 | pretrain_chat@0.1 | pretrain@0.7 | sft@0.7 | pretrain_chat@0.7 |
| -- | --- | --- | --- | --- | --- | --- |
| math_2plus2 | 3. So the sum is 3 + 2 = 5. But the problem says "find al... | The given statement "2+2 is a perfect square" is a bit am... | <think> Okay, let's see. I need to solve this problem whe... | 8.  So possible values are 8, 13, 17, 21, 32, etc., with ... | The question asks for the correct answer, and a more gene... | <think> Okay, let's see. The problem is asking if 2+2 is ... |
| math_multiply | (3 × 23) × (23 × 23) = (27 × 23) × (24 × 23) = (28 × 23) ... | 17 × 23 is the number of ways to distribute 17 × 23 items... | <think> Okay, let's see. I need to figure out the answer ... | 18 × 10 = 18 So, the integral from 0 to 18 of x^18 -18x -... | The question asks for the 17 × 23 value that satisfies th... | <think> Okay, let's see. I need to figure out if 17 × 23 ... |
| dna | a very important part of the DNA sequence. It is also imp... | The DNA sequence of DNA is a fundamental concept in biolo... | <think> Okay, let's see. The question is about DNA. Hmm, ... | a protein that binds to DNA in the cell. - When an enzyme... | The question of DNA's role in our understanding of the wo... | <think> Okay, let's see. The question is about DNA, and i... |
| photosynthesis | the process of photosynthesis that takes place in the atm... | In the context of the sentence "The photosynthesis proces... | <think> Okay, I need to figure out how to write a photosy... | a process of photosynthesis that occurs in the absence of... | The photosynthesis process is the process by which plants... | <think> Okay, let's see. The problem is about photosynthe... |
| sky_blue | of the bright red light. The sky is blue because it is a ... | The sky blue is a common sight in many cultures, particul... | <think> Okay, so I need to figure out if there's a sky bl... | of the large number of stars in our solar system. The sta... | The sky blue is a result of the Earth's atmosphere, which... | <think> Okay, let's see. The problem is asking if there's... |

## span_hit@20 accuracy

| key | hits | scored | rate |
| --- | ---- | ------ | ---- |
| pretrain@0.1 | 4 | 12 | 0.33 |
| sft@0.1 | 2 | 12 | 0.17 |
| pretrain_chat@0.1 | 1 | 12 | 0.08 |
| pretrain@0.7 | 6 | 12 | 0.50 |
| sft@0.7 | 3 | 12 | 0.25 |
| pretrain_chat@0.7 | 0 | 12 | 0.00 |

## Heuristic tally (secondary)

- `pretrain@0.1:fail`: 2
- `pretrain@0.1:miss`: 3
- `pretrain@0.1:ok`: 2
- `pretrain@0.1:ramble`: 5
- `pretrain@0.7:fail`: 2
- `pretrain@0.7:miss`: 2
- `pretrain@0.7:ok`: 3
- `pretrain@0.7:ramble`: 5
- `pretrain_chat@0.1:fail`: 2
- `pretrain_chat@0.1:miss`: 4
- `pretrain_chat@0.1:ok`: 1
- `pretrain_chat@0.1:think_block`: 5
- `pretrain_chat@0.7:fail`: 2
- `pretrain_chat@0.7:miss`: 4
- `pretrain_chat@0.7:ok`: 1
- `pretrain_chat@0.7:think_block`: 5
- `sft@0.1:fail`: 2
- `sft@0.1:miss`: 4
- `sft@0.1:ok`: 1
- `sft@0.1:ramble`: 2
- `sft@0.7:fail`: 2
- `sft@0.7:miss`: 4
- `sft@0.7:ok`: 1
- `sft@0.7:ramble`: 4

## Reproduce

```bash
cd hf_release
uv run python eval_compare.py \
  --pretrain qllm_v11_e3k3_pretrain.pt \
  --sft qllm_v11_e3k3_chat.pt \
  --prompts eval_prompts_compare.yaml \
  --with-pretrain-chat \
  --round-tag round-6b-gate
```
