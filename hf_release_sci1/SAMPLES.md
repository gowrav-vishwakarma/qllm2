# SCI-1 samples

Checkpoint: `sci1_386m_49b.pt` (step 749500, 49.12B tokens), 385,817,038 params, generated with `modeling_pam2.py` only (no qllm2 repo), cpu, fp32, 2026-11-24. Reproduce: `python make_samples.py`.

* greedy: temperature 0 (no repetition penalty)
* sampled: temperature 0.8, top-k 50, top-p 0.9, repetition penalty 1.2, seed 0 (torch.Generator on cpu)

## Plain continuation (max 80 new tokens, stop at `<|endoftext|>`)

### 1. greedy

**Prompt:**

```text
In 1923, the University of
```

**Output** (80 tokens, 3.4s):

```text
In 1923, the University of Pennsylvania was founded. The University of Pennsylvania was the first institution to offer undergraduate and graduate courses in the United States. The University of Pennsylvania was the first institution to offer undergraduate and graduate courses in the United States.

The University of Pennsylvania was founded in 1884 by the University of Pennsylvania professors, James Wilson and John Wilson. The University of Pennsylvania was the first institution to offer undergraduate and graduate courses
```

### 1. sampled

**Prompt:**

```text
In 1923, the University of
```

**Output** (80 tokens, 3.0s):

```text
In 1923, the University of Tokyo began a series of student-run programs to help students with the transition from high school to college. These were called the Graduate Student Support Programs, or GSPs. The program was run by the Graduate School of Education, which is now the Center for Graduate Study in the Education and Work Process (CGSEP).
The student support system was in place throughout the 1930s, but after
```

### 2. greedy

**Prompt:**

```text
Photosynthesis is the process by which
```

**Output** (80 tokens, 2.6s):

```text
Photosynthesis is the process by which plants convert light energy into chemical energy in the form of glucose. It is the main source of energy for all living organisms.

The process of photosynthesis is a series of chemical reactions that take place in the chloroplasts of plant cells. The main chemical reactions that occur in the chloroplasts are:

1. Light energy is converted into chemical energy in the form of ATP (
```

### 2. sampled

**Prompt:**

```text
Photosynthesis is the process by which
```

**Output** (80 tokens, 2.9s):

```text
Photosynthesis is the process by which plants convert light energy into chemical energy. It includes the following processes:
- Light synthesis
- Light absorption
- Light emission
- Light detection and amplification (LDA)
- Phosphorescence
- Phosphor bleaching
- Phosphorescence
This cycle provides a detailed explanation of the light reactions in living organisms and how they occur in different environments.
Light reactions are
```

## Chat: one 3-turn ChatML session (no system message, max 160 new tokens per reply, stop at `<|im_end|>`, state carried between turns)

### greedy

**Turn 1, user:** What is the capital of France?

**Assistant** (8 tokens, stopped_on_im_end=True, 0.3s):

```text
The capital of France is Paris.
```

**Turn 2, user:** What is it famous for?

**Assistant** (36 tokens, stopped_on_im_end=True, 1.3s):

```text
France is famous for its art, architecture, and cuisine. It is also known for its wine country, which is home to some of the world's most famous wineries.
```

**Turn 3, user:** Write a short poem about it.

**Assistant** (44 tokens, stopped_on_im_end=True, 1.5s):

```text
The sun sets on the French countryside, painting the sky with hues of orange and yellow. The wind whispers secrets through the trees, carrying the scent of blooming flowers and the sweet aroma of blooming gardens.
```

<details><summary>Raw ChatML the model saw</summary>

```text
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
<|im_start|>user
What is it famous for?<|im_end|>
<|im_start|>assistant
France is famous for its art, architecture, and cuisine. It is also known for its wine country, which is home to some of the world's most famous wineries.<|im_end|>
<|im_start|>user
Write a short poem about it.<|im_end|>
<|im_start|>assistant
The sun sets on the French countryside, painting the sky with hues of orange and yellow. The wind whispers secrets through the trees, carrying the scent of blooming flowers and the sweet aroma of blooming gardens.<|im_end|>

```

</details>

### sampled

**Turn 1, user:** What is the capital of France?

**Assistant** (66 tokens, stopped_on_im_end=True, 2.4s):

```text
The capital of France is Paris. It was the capital of French colonial power from 1608 until 1814, when it was absorbed by Napoleon's Empire. The capital city of France has a population of over 7 million people and includes four other major cities: Nice, Rouen, Le Tourgueur, and Lyon.
```

**Turn 2, user:** What is it famous for?

**Assistant** (66 tokens, stopped_on_im_end=True, 2.3s):

```text
France is known as "La République" or "the land of love," due to its beautiful landscapes, rich culture, and history of warfare against foreign powers. Its capital city, Paris, is also home to many world-class museums and art historical sites, including the Eiffel Tower and Louvre Museum.
```

**Turn 3, user:** Write a short poem about it.

**Assistant** (78 tokens, stopped_on_im_end=True, 2.8s):

```text
L'Eau sur le monde (That evening in Paris) 
A grand feast, but not so much after sunset;
I've seen the stars come alive before me,
and they're like little fires burning bright inside my bones.
There's nothing more lovely than seeing that starry sky once again,
Where I can find peace amidst chaos and strife.
```

<details><summary>Raw ChatML the model saw</summary>

```text
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris. It was the capital of French colonial power from 1608 until 1814, when it was absorbed by Napoleon's Empire. The capital city of France has a population of over 7 million people and includes four other major cities: Nice, Rouen, Le Tourgueur, and Lyon.<|im_end|>
<|im_start|>user
What is it famous for?<|im_end|>
<|im_start|>assistant
France is known as "La République" or "the land of love," due to its beautiful landscapes, rich culture, and history of warfare against foreign powers. Its capital city, Paris, is also home to many world-class museums and art historical sites, including the Eiffel Tower and Louvre Museum.<|im_end|>
<|im_start|>user
Write a short poem about it.<|im_end|>
<|im_start|>assistant
L'Eau sur le monde (That evening in Paris) 
A grand feast, but not so much after sunset;
I've seen the stars come alive before me,
and they're like little fires burning bright inside my bones.
There's nothing more lovely than seeing that starry sky once again,
Where I can find peace amidst chaos and strife.<|im_end|>

```

</details>
