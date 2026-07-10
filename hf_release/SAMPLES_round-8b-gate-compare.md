# Pretrain vs SFT — round-8b-gate

Generated **2026-07-10T13:46:32.419396+00:00** on `cuda`.

Same factual questions: **pretrain** completes the `prefix`; **SFT** answers in ChatML chat.
Answers per question: `pretrain@0.1`, `sft@0.1`, `pretrain@0.7`, `sft@0.7`.
`span_hit@20`: gold span in first 20 generated tokens.
Full text: see the JSON log.

## retrieval

| id | pretrain@0.1 | sft@0.1 | pretrain@0.7 | sft@0.7 |
| -- | --- | --- | --- | --- |
| france | Paris, and the capital is Paris. The capital is Paris, wh... | The capital of France is Paris, which is located in the c... | Paris, and the city was destroyed in a war which would le... | The capital of France is Paris, which is located at the c... |
| usa_capital | Washington, D.C., and the capital is Washington. The capi... | The capital of the United States is Washington, D.C., and... | Washington, DC. The city was built in 1763 and in 1810 it... | The capital of the United States is called The Washington... |
| shakespeare | a young girl named R.M. O’Brien, who was born in the town... | In the movie "Odysseus," the character who wrote the stor... | Romeo, who was the son of a wealthy merchant. When Juliet... | The question asks for the who, what, and who wrote Shakes... |
| largest_planet | the Sun. The Sun is a giant star that has been observed t... | The largest planet in our solar system, called the sun, i... | the sun. It is also known as the "solar planet". The clos... | The largest planet in our solar system, called our sun, i... |

## biography

| id | pretrain@0.1 | sft@0.1 | pretrain@0.7 | sft@0.7 |
| -- | --- | --- | --- | --- |
| karl_marx | a German philosopher. He was the first to write about the... | Hobelers were known for his work on the development of th... | a famous economist and a pioneer in the field of economic... | Kreys was a professional football player from the United ... |
| einstein | a pioneer in the field of quantum mechanics. The first qu... | Eugene Eugene, an American scientist and writer who made ... | a famous physicist who had been instrumental in making qu... | Albert Einstein was born on August 14, 1957. He is an Ame... |
| newton | a pioneer in the field of physics. He was born on January... | The person who was born on September 11, 1812, is named I... | a famous student of Newton. He was born in 1706 in London... | The question asks for the person who was referred to as I... |

## reasoning

| id | pretrain@0.1 | sft@0.1 | pretrain@0.7 | sft@0.7 |
| -- | --- | --- | --- | --- |
| math_2plus2 | 3.      \[      \frac{1}{3} - \frac{4}{5} = \frac{8}{15} ... | To find the value of 2, we need to first identify the val... | 8. \] This is the maximum possible value of \( \frac{1}{8... | The question asks for the value of 2 in a geometric progr... |
| math_multiply | 18.      \[      \text{A}_1 = \frac{18}{2} \times 17 = 18... | 17 × 23 = 17 × 23  To find the value of 17, we need to di... | 18 × 10 = 18      \[      \text{(1, 18)^2} = 18      \]  ... | 17 × 23 = 18  In this example, 17 × 23 = 18 × 23, where 1... |
| dna | a very important part of the DNA sequence. It is also kno... | The DNA of the human body is made up of DNA, which is the... | a protein complex that contains more than 1,500 genes. Wh... | The DNA of the human being, which includes DNA and protei... |
| photosynthesis | the process of converting sunlight into energy. The proce... | In the context of photosynthesis, photosynthesis is the p... | a process of converting sunlight into energy, and it requ... | The photosynthesis process occurs in the chloroplast, whi... |
| sky_blue | of the light that is reflected off the surface. The sky i... | The sky is blue because it is a blue color that is not vi... | of the large number of stars in our solar system. The sta... | The sky blue is the result of the sky being blue, which c... |

## span_hit@20 accuracy

| key | hits | scored | rate |
| --- | ---- | ------ | ---- |
| pretrain@0.1 | 5 | 12 | 0.42 |
| sft@0.1 | 6 | 12 | 0.50 |
| pretrain@0.7 | 5 | 12 | 0.42 |
| sft@0.7 | 3 | 12 | 0.25 |

## Heuristic tally (secondary)

- `pretrain@0.1:fail`: 2
- `pretrain@0.1:miss`: 1
- `pretrain@0.1:ok`: 4
- `pretrain@0.1:ramble`: 5
- `pretrain@0.7:fail`: 2
- `pretrain@0.7:miss`: 2
- `pretrain@0.7:ok`: 3
- `pretrain@0.7:ramble`: 5
- `sft@0.1:fail`: 2
- `sft@0.1:miss`: 3
- `sft@0.1:ok`: 2
- `sft@0.1:ramble`: 2
- `sft@0.7:fail`: 2
- `sft@0.7:miss`: 2
- `sft@0.7:ok`: 3
- `sft@0.7:ramble`: 3

## Reproduce

```bash
cd hf_release
uv run python eval_compare.py \
  --pretrain qllm_v11_e3k3_pretrain.pt \
  --sft qllm_v11_e3k3_chat.pt \
  --prompts eval_prompts_compare.yaml \
  --round-tag round-8b-gate
```
