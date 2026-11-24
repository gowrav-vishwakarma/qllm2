# SCI-1 Chat samples

Checkpoint: `sci1_chat_386m.pt` (step 753315, 49.37B tokens), 385,817,038 params, generated with `modeling_pam2.py` only (no qllm2 repo), cpu, fp32, 2026-11-24. Reproduce: `python make_samples.py`.

* greedy: temperature 0 (no repetition penalty)
* sampled: temperature 0.8, top-k 50, top-p 0.9, repetition penalty 1.2, seed 0 (torch.Generator on cpu)

## Plain continuation (max 80 new tokens, stop at `<|endoftext|>`)

### 1. greedy

**Prompt:**

```text
In 1923, the University of
```

**Output** (80 tokens, 2.6s):

```text
In 1923, the University of California at San Francisco (UCSF) began a program of research on the effects of the new drug, called the "Tricyclic Antihypertensive drug," on the blood pressure of patients with hypertension. The results of this study were published in the Journal of the American Medical Association in 1923.

The first clinical trial of the drug was conducted in 1923 at the University
```

### 1. sampled

**Prompt:**

```text
In 1923, the University of
```

**Output** (80 tokens, 2.7s):

```text
In 1923, the University of Michigan began a series of studies aimed at understanding the effects of exposure to air pollution on children's health. In 1919, a group of researchers conducted the first-ever experiment in which they exposed children and their parents to levels of ozone that were equivalent to the World Health Organization's concentration (WHO, n.d.). The results were shocking: the study showed that children who grew up in cities with higher
```

### 2. greedy

**Prompt:**

```text
Photosynthesis is the process by which
```

**Output** (80 tokens, 2.6s):

```text
Photosynthesis is the process by which plants, algae, and some bacteria convert light energy from the sun into chemical energy in the form of glucose. This process is essential for life on Earth, as it provides the energy needed to grow and thrive.

The process of photosynthesis is a complex and intricate process that involves a series of steps. It begins with the absorption of light energy by pigments such as chlorophyll, which
```

### 2. sampled

**Prompt:**

```text
Photosynthesis is the process by which
```

**Output** (80 tokens, 2.9s):

```text
Photosynthesis is the process by which plants use sunlight to convert water and carbon dioxide into glucose (sugar) and oxygen.
The main types of photosynthesis are:
1. Light-dependent reactions: This type of photosynthesis occurs in specialized organelles called chloroplasts, which contain pigments that absorb light energy from the sun. The energy from these molecules is used to power a series of chemical reactions that produce ATP
```

## Chat: one 3-turn ChatML session (no system message, max 160 new tokens per reply, stop at `<|im_end|>`, state carried between turns)

### greedy

**Turn 1, user:** What is the capital of France?

**Assistant** (8 tokens, stopped_on_im_end=True, 0.3s):

```text
The capital of France is Paris.
```

**Turn 2, user:** What is it famous for?

**Assistant** (14 tokens, stopped_on_im_end=True, 0.5s):

```text
Paris is famous for its beautiful architecture, art, and culture.
```

**Turn 3, user:** Write a short poem about it.

**Assistant** (74 tokens, stopped_on_im_end=True, 2.5s):

```text
In the heart of Paris, where dreams and shadows dance,
A world of beauty, where every day is a new delight.

The sun shines brightly, casting a golden glow,
A canvas of colors, a masterpiece of light.
The city's streets, a tapestry of life,
A testament to the power of love and light.
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
Paris is famous for its beautiful architecture, art, and culture.<|im_end|>
<|im_start|>user
Write a short poem about it.<|im_end|>
<|im_start|>assistant
In the heart of Paris, where dreams and shadows dance,
A world of beauty, where every day is a new delight.

The sun shines brightly, casting a golden glow,
A canvas of colors, a masterpiece of light.
The city's streets, a tapestry of life,
A testament to the power of love and light.<|im_end|>

```

</details>

### sampled

**Turn 1, user:** What is the capital of France?

**Assistant** (55 tokens, stopped_on_im_end=True, 1.9s):

```text
The capital of France is Paris. It's the largest city in the country, and it has a population of around 1 million people. Imagine if your favorite toy store was like that big building where everything from toys to books goes - just imagine all those little spaces inside!
```

**Turn 2, user:** What is it famous for?

**Assistant** (121 tokens, stopped_on_im_end=True, 4.0s):

```text
The most famous thing about Paris is its beautiful architecture, especially looking at Notre-Dame Cathedral, which looks so grandest with such tall columns called arches made out of sandstone or limestone. There are also charming cafes, shops, restaurants, museums, parks, gardens, art galleries, theater productions, opera houses, cinemas, hospitals, hotels, schools, shopping centers, amusement parks, airports, ports, bridges, roads, railway lines, subway stations, train stops, concerts, festivals, sports fields, clubs, music venues, barbecues & beyond! Isn't life amazing?
```

**Turn 3, user:** Write a short poem about it.

**Assistant** (160 tokens, stopped_on_im_end=False, 5.3s):

```text
In this enchanting world we've created,  
Where dreams take flight under starry skies.  
Beneath our vast green roof stands pride,  
A testament of strength unbound by time.  
Every morning whispers tales told on red moonlight;  
Each night reveals new wonders waiting to be discovered.  
From hidden glade trails leading through lush woods,  
To ancient ruins adorned within endless mists.  
In every corner lies secrets untold,  
Of gods who stand before earth as guardians.  

Here you can find tranquil spots swaying gently,  
Guiding travelers along winding paths without trouble.  
In these serene alleys lie stories too deep,
```

<details><summary>Raw ChatML the model saw</summary>

```text
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris. It's the largest city in the country, and it has a population of around 1 million people. Imagine if your favorite toy store was like that big building where everything from toys to books goes - just imagine all those little spaces inside!<|im_end|>
<|im_start|>user
What is it famous for?<|im_end|>
<|im_start|>assistant
The most famous thing about Paris is its beautiful architecture, especially looking at Notre-Dame Cathedral, which looks so grandest with such tall columns called arches made out of sandstone or limestone. There are also charming cafes, shops, restaurants, museums, parks, gardens, art galleries, theater productions, opera houses, cinemas, hospitals, hotels, schools, shopping centers, amusement parks, airports, ports, bridges, roads, railway lines, subway stations, train stops, concerts, festivals, sports fields, clubs, music venues, barbecues & beyond! Isn't life amazing?<|im_end|>
<|im_start|>user
Write a short poem about it.<|im_end|>
<|im_start|>assistant
In this enchanting world we've created,  
Where dreams take flight under starry skies.  
Beneath our vast green roof stands pride,  
A testament of strength unbound by time.  
Every morning whispers tales told on red moonlight;  
Each night reveals new wonders waiting to be discovered.  
From hidden glade trails leading through lush woods,  
To ancient ruins adorned within endless mists.  
In every corner lies secrets untold,  
Of gods who stand before earth as guardians.  

Here you can find tranquil spots swaying gently,  
Guiding travelers along winding paths without trouble.  
In these serene alleys lie stories too deep, <|im_end|>

```

</details>
