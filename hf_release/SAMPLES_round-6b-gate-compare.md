# Pretrain vs SFT — round-6b-gate

Generated **2026-07-09T12:33:54.122536+00:00** on `cuda`.

Same factual questions: **pretrain** completes the `prefix`; **SFT** answers in ChatML chat.
Four answers per question: `pretrain@0.1`, `pretrain@0.7`, `sft@0.1`, `sft@0.7`.
Full text: see the JSON log.

## Summary

| id | category | pretrain@0.1 | pretrain@0.7 | sft@0.1 | sft@0.7 |
| -- | -------- | ------------ | ------------ | ------- | ------- |
| france | web_factual | Paris. - The capital of the French Republic, France. - The capital of... | Paris, which was founded in 1448 and had a population of over 4 milli... | The capital of France, France, is Paris. It is located in the heart o... | The capital of France is Paris, and its capital is France. It's also ... |
| usa_capital | web_factual | the city of San Francisco. The city was founded in 1869 by a group of... | Washington, DC. The city was built in 1763 and in 1827 it was called ... | The capital of the United States, the United States, is called "U-sha... | The capital of the United States, called U.S. (U). The capital of the... |
| karl_marx | web_factual | a German philosopher and political theorist. He was born in Berlin, G... | a famous economist and a pioneer in the field of economic theory. Mar... | Hier ist einige Ideen, die Sie in der Regel verwendet werden kann: 1.... | Yes, Karl<\|endoftext\|>he is a German-born and skilled politician, b... |
| shakespeare | web_factual | a young girl named Juliet. She was the first to write a novel about h... | Romeo, who was the son of a wealthy merchant and a friend. The two ch... | The question of who wrote the name of the person who wrote the name i... | The question of who wrote which is likely to be the most likely perso... |
| largest_planet | web_factual | the Sun. The Sun is a giant planet that orbits the sun every 4,000 ye... | our sun. It has a mass of 1.8 times that of the Sun and has a mass of... | The largest planet in our solar system, known as the "Echo," is locat... | The largest planet in our solar system, called "Eus exoplanet," has b... |
| math_2plus2 | math_stress | 3. So the sum is 3 + 2 = 5. But the problem says "find all positive i... | 8.  So possible values are 8, 13, 17, 21, 32, etc., with the constrai... | The given statement "2+2 is a perfect square" is a bit ambiguous, but... | The question asks for the correct answer, and a more general definiti... |
| math_multiply | math_stress | (3 × 23) × (23 × 23) = (27 × 23) × (24 × 23) = (28 × 23) × (29 × 23) ... | 18 × 10 = 18 So, the integral from 0 to 18 of x^18 -18x -18y -18z =0.... | 17 × 23 is the number of ways to distribute 17 × 23 items into 5 × 3 ... | The question asks for the 17 × 23 value that satisfies the conditions... |
| dna | edu_science | a very important part of the DNA sequence. It is also important to un... | a protein that binds to DNA in the cell. - When an enzyme called ribo... | The DNA sequence of DNA is a fundamental concept in biology, and it p... | The question of DNA's role in our understanding of the world around u... |
| photosynthesis | edu_science | the process of photosynthesis that takes place in the atmosphere. Pho... | a process of photosynthesis that occurs in the absence of sunlight. I... | In the context of the sentence "The photosynthesis process is a cruci... | The photosynthesis process is the process by which plants and animals... |
| sky_blue | reasoning_format | of the bright red light. The sky is blue because it is a blue-green c... | of the large number of stars in our solar system. The stars are movin... | The sky blue is a common sight in many cultures, particularly in the ... | The sky blue is a result of the Earth's atmosphere, which is caused b... |

## Heuristic tally

- `pretrain@0.1:fail`: 2
- `pretrain@0.1:miss`: 3
- `pretrain@0.1:ok`: 2
- `pretrain@0.1:ramble`: 3
- `pretrain@0.7:fail`: 2
- `pretrain@0.7:miss`: 2
- `pretrain@0.7:ok`: 3
- `pretrain@0.7:ramble`: 3
- `sft@0.1:fail`: 2
- `sft@0.1:miss`: 4
- `sft@0.1:ok`: 1
- `sft@0.1:ramble`: 2
- `sft@0.7:fail`: 2
- `sft@0.7:miss`: 4
- `sft@0.7:ok`: 1
- `sft@0.7:ramble`: 2

## Reproduce

```bash
cd hf_release
uv run python eval_compare.py \
  --pretrain qllm_v11_e3k3_pretrain.pt \
  --sft qllm_v11_e3k3_chat.pt \
  --prompts eval_prompts_compare.yaml \
  --out-md SAMPLES_round-6b-gate-compare.md \
  --out-json ../logs/v11/round-6b-gate_compare.json
```
