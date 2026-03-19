# Phase-Associative Memory: Attention-Free Sequence Modeling via Complex Matrix States

**Gowrav Vishwakarma**

March 2026

---

## Abstract

We present Phase-Associative Memory (PAM), a recurrent sequence modeling architecture that replaces softmax attention with complex-valued phase interference over a learned matrix state. Each PAM head maintains a state matrix $S_t \in \mathbb{C}^{d \times d}$, updated via outer products of complex-valued key-value pairs and retrieved via complex dot products with queries. The conjugate inner product $K_t^* \cdot Q_t$ produces constructive or destructive interference that functions as an implicit attention mechanism without softmax normalization. A Gated State Protection (GSP) mechanism allows the model to selectively freeze state dimensions, retaining important information indefinitely.

PAM admits two equivalent computational forms: a dual (quadratic) form for parallel training and a recurrent form for $O(1)$-per-token inference with fixed $O(d^2)$ memory per head, requiring no KV cache. The entire model operates in complex-valued space with phase-preserving primitives end to end.

We evaluate PAM as the core sequence layer in a 100M-parameter language model trained on WikiText-103, achieving validation perplexity of 38.95. This does not yet match GPT-2's approximately 31 perplexity at 124M parameters, but it represents, to our knowledge, the first complex-valued, attention-free architecture to reach this range on a standard benchmark. We present PAM as a promising research direction rather than a finished system, and discuss both the architectural properties that may prove advantageous at scale and the concrete limitations that remain.

---

## 1. Introduction

The transformer architecture has become the dominant paradigm for language modeling, with its core mechanism -- scaled dot-product attention -- providing flexible, content-dependent information routing across sequences. However, attention's $O(T^2)$ cost per layer during inference and its linearly growing KV cache have motivated substantial research into alternative sequence processing mechanisms, including state-space models (S4, Mamba), linear attention variants, and recurrent architectures with selection mechanisms.

This paper contributes to that line of work from an unusual angle: complex-valued computation with phase-preserving operations throughout the model. The core hypothesis is that representing tokens as complex vectors and performing retrieval via phase interference -- constructive alignment between query and key phases amplifies recall, destructive misalignment suppresses it -- can serve as a natural alternative to softmax-normalized attention.

PAM emerged from a multi-stage research process:

1. **V4** introduced the idea of tokens in complex phase space with wave-style interference, establishing the non-transformer direction.
2. **V5** identified and fixed a critical flaw: V4 passed complex representations through real-valued nonlinearities (GELU, sigmoid) that destroyed phase information. V5 introduced phase-preserving primitives (modReLU, ComplexGatedUnit, ComplexNorm) and showed that mathematical consistency materially improved results.
3. **V6** built on V5's phase-preserving foundation with a multi-timescale complex SSM backbone. Initial experiments with vector-state SSMs showed that associative capacity was limited by state interference: binding multiple facts into a single vector caused catastrophic collisions.
4. **PAM** solves the interference problem by upgrading the state from a vector ($\mathbb{C}^d$) to a matrix ($\mathbb{C}^{d \times d}$), providing $O(d^2)$ associative capacity per head.

This paper focuses on PAM itself and the evidence from its first validated training run. We are explicit about what PAM has demonstrated and what remains open.

### 1.1 Contributions

- **Phase-Associative Memory (PAM)**: a recurrent layer with complex matrix state, data-dependent decay, and retrieval via complex phase interference -- no softmax, no KV cache.
- **Gated State Protection (GSP)**: a learned per-dimension freeze gate that allows the model to retain important state indefinitely.
- **Dual-form training**: an $O(T^2)$ parallel computation equivalent to the $O(T)$ recurrence, enabling efficient GPU training.
- **Empirical validation**: WikiText-103 perplexity of 38.95 at 100M parameters, with coherent multi-sentence generation and low repetition.

### 1.2 Scope and honest framing

We do not claim that PAM outperforms transformers. GPT-2 (124M parameters) achieves approximately 31 perplexity on WikiText-103, and PAM's 38.95 at 100M parameters leaves a meaningful gap. What we do claim is narrower: PAM is a genuinely different architecture -- complex-valued, attention-free, recurrent with $O(1)$ inference cost per token -- that has reached a performance level where continued investigation is justified. We present this as a research contribution, not a production system.

---

## 2. Background and Related Work

### 2.1 Transformers and attention

The transformer (Vaswani et al., 2017) computes attention as $\text{softmax}(QK^\top / \sqrt{d})V$, where queries, keys, and values are linear projections of the input. This is powerful but costs $O(T^2)$ per layer and requires a KV cache that grows linearly with sequence length during inference. Numerous efficiency variants exist (sparse attention, linear attention, sliding window), but the core mechanism remains fundamentally quadratic.

### 2.2 State-space models and recurrent alternatives

Modern SSMs (S4, Mamba, Mamba-2, Griffin) process sequences through learned recurrences with diagonal or structured state transitions. Mamba introduced input-dependent selection (selective SSMs), allowing the recurrence to be content-aware. These models achieve $O(T)$ training via parallel scan and $O(1)$ per-token inference with fixed state. PAM shares the $O(1)$ inference property but differs in two key ways: the state is a complex-valued matrix rather than a real or complex vector, and retrieval uses phase interference rather than a learned output projection.

### 2.3 Linear attention and matrix-state models

Linear attention (Katharopoulos et al., 2020) reformulates attention by removing softmax, yielding a recurrent form with matrix state $S_t = S_{t-1} + V_t K_t^\top$. Recent work (RetNet, GLA, DeltaNet, Mamba-2) has explored decay-weighted variants of this form. PAM is structurally related to this family but operates entirely in complex space: the state update is $S_t = \gamma_t S_{t-1} + V_t \otimes K_t^*$ (outer product with conjugate), and retrieval computes $Y_t = S_t Q_t$ where the dot product $K_t^* \cdot Q_t$ produces phase interference. The complex conjugate in the key projection is not a superficial notational choice -- it means that retrieval quality depends on phase alignment between query and stored key, providing a geometric selectivity mechanism that real-valued matrix states lack.

### 2.4 Complex-valued neural networks

Complex-valued neural networks have been studied for decades (Hirose, 2012), with applications in signal processing, physics-informed modeling, and sequence modeling. Key ingredients include complex linear maps, phase-preserving activations like modReLU (Arjovsky et al., 2016), and unitary recurrent dynamics. Our work inherits this tradition but applies it specifically to language modeling at scale, which has received less attention in the complex-valued neural network literature.

### 2.5 Associative memory

Holographic Reduced Representations (Plate, 1995) showed that circular convolution can bind key-value pairs into a single vector and circular correlation can retrieve them. In Fourier space, binding becomes element-wise complex multiplication and retrieval becomes multiplication by the conjugate. PAM draws on this insight: `cmul(key, value)` is binding, `cmul(query, conj(stored))` is retrieval. However, PAM learned from a failed experiment (Holographic State Binding, Section 6.1) that vector-state binding causes interference when multiple associations accumulate. The matrix state solves this by providing $O(d^2)$ capacity instead of $O(d)$.

---

## 3. Method

### 3.1 Overview

The full model architecture is:

```
Tokens -> ComplexEmbed -> ComplexNorm
  -> [ComplexGatedUnit + residual] x 16 layers    (feature extraction)
  -> [PhaseAssociativeLayer + residual] x 16 layers  (PAM)
  -> ComplexLinear -> ComplexNorm -> TiedComplexLMHead
```

All operations in the main signal path are complex-valued and phase-preserving. Control signals (gates, decay parameters) may use real-valued projections over magnitude features, but the primary data path never converts complex representations to real-valued intermediate forms.

### 3.2 Phase-preserving primitives

These primitives, inherited from V5, form the algebraic foundation:

**Complex representation.** All tensors have shape `[..., dim, 2]`, where the last dimension holds (real, imaginary) components. This is a concrete implementation of $\mathbb{C}^{dim}$.

**Complex linear map.** Given weight matrices $W_r, W_i \in \mathbb{R}^{m \times n}$, the complex linear map computes:
$$y_r = W_r x_r - W_i x_i, \quad y_i = W_i x_r + W_r x_i$$

This is standard complex matrix multiplication, implemented as four real matrix multiplications.

**modReLU.** A phase-preserving activation that thresholds magnitude while leaving phase untouched:
$$\text{modReLU}(z) = \text{ReLU}(|z| + b) \cdot \frac{z}{|z|}$$

where $b$ is a learned bias. This replaces GELU/ReLU, which would destroy phase information.

**ComplexNorm.** RMS normalization applied to magnitudes, with phase preserved:
$$\text{ComplexNorm}(z) = s \cdot \frac{|z|}{\text{RMS}(|z|)} \cdot \frac{z}{|z|}$$

where $s$ is a learned scale parameter.

**ComplexGatedUnit (CGU).** A SwiGLU-style gating block in complex space. The gate magnitude $\sigma(|W_g z|)$ controls how much signal passes, while the gate phase controls what rotation is applied. An up-projection through modReLU provides the nonlinearity:
$$\text{CGU}(z) = W_\text{down}(\text{gate\_phase} \odot \text{modReLU}(W_\text{up} z) \cdot \sigma(|W_g z|))$$

### 3.3 Feature extraction layers

Before PAM processes the sequence, 16 CGU layers extract features with residual connections and learned scaling:

$$z^{(l)} = z^{(l-1)} + \alpha_l \cdot \text{CGU}_l(\text{ComplexNorm}(z^{(l-1)}))$$

where $\alpha_l$ is a learnable scalar initialized to 1.0. Each CGU uses an expansion factor of 3 (hidden dimension = $3 \times 384 = 1152$). Dropout is applied during training.

### 3.4 Phase-Associative Memory (PAM)

PAM is the core contribution. It replaces both the recurrent backbone (SSM) and the attention mechanism with a single module.

#### 3.4.1 State representation

Each PAM head $h$ maintains a complex matrix state $S_t^{(h)} \in \mathbb{C}^{d \times d}$, where $d$ is the head dimension. With $H$ heads and head dimension $d$, the total state capacity is $H \times d^2$ complex values. In our configuration ($H = 6$, $d = 64$), this is $6 \times 64^2 = 24{,}576$ complex values per layer.

#### 3.4.2 Projections

The input $x_t \in \mathbb{C}^D$ (where $D = 384$) is projected into queries, keys, and values via complex linear maps (without bias):

$$Q_t = W_Q x_t, \quad K_t = W_K x_t, \quad V_t = W_V x_t$$

Each projection maps $\mathbb{C}^D \to \mathbb{C}^{H \times d}$ (i.e., $\mathbb{C}^{384} \to \mathbb{C}^{384}$ reshaped to $\mathbb{C}^{6 \times 64}$).

#### 3.4.3 Data-dependent decay

The decay rate $\gamma_t$ controls how quickly the state forgets. It is computed from the input:

$$dt_t = \text{softplus}(W_{dt} \cdot \text{concat}(x_{t,r}, x_{t,i}) + b_{dt})$$

where $W_{dt} \in \mathbb{R}^{H \times 2D}$ projects the concatenated real and imaginary parts to one scalar per head, and $b_{dt}$ is initialized to $-4.0$ (yielding slow initial decay).

#### 3.4.4 Gated State Protection (GSP)

GSP adds a learned protect gate $p_t$ that can freeze state dimensions:

$$p_t = \sigma(W_p \cdot |x_t| + b_p), \quad b_p \text{ initialized to } {-3.0}$$
$$\gamma_t = e^{-dt_t} \cdot (1 - p_t) + p_t$$

When $p_t \to 1$, $\gamma_t \to 1$ and the state is frozen regardless of the decay signal. When $p_t \to 0$, the decay proceeds normally. The protect gate bias is initialized to $-3.0$ ($\sigma(-3) \approx 0.047$), so protection starts near zero and must be learned.

The value signal is also modulated by the protect gate:

$$V'_t = V_t \cdot (1 - p_t)$$

This prevents new values from overwriting protected state dimensions.

#### 3.4.5 State update

The state evolves as:

$$S_t = \gamma_t \cdot S_{t-1} + V'_t \otimes K_t^*$$

where $\otimes$ denotes complex outer product and $K_t^*$ is the complex conjugate of the key. This is a decay-weighted linear recurrence over matrices.

#### 3.4.6 Retrieval via phase interference

Retrieval computes:

$$Y_t = S_t Q_t$$

Expanding the state, this yields:

$$Y_t = \sum_{i \leq t} \left(\prod_{j=i+1}^{t} \gamma_j\right) (K_i^* \cdot Q_t) \cdot V'_i$$

The term $K_i^* \cdot Q_t$ is a complex dot product. Its magnitude depends on phase alignment between the stored key $K_i$ and the current query $Q_t$:

- **Constructive interference**: when $K_i$ and $Q_t$ have aligned phases, $|K_i^* \cdot Q_t|$ is large, amplifying retrieval of the corresponding $V'_i$.
- **Destructive interference**: when phases are misaligned, the term is small or cancels, suppressing retrieval.

This is the mechanism that replaces softmax attention. It is not learned separately -- it is an intrinsic property of complex dot products.

#### 3.4.7 Dual form (training)

During training, we avoid materializing the $d \times d$ matrix sequentially. Instead, we compute the output in $O(T^2)$ time using dense matrix multiplications:

1. Compute the decay matrix $D \in \mathbb{R}^{T \times T}$ in log space:
   $$\log D[t, i] = \sum_{j=i+1}^{t} \log \gamma_j$$
   using cumulative sums for numerical stability.

2. Apply a causal mask: $D[t, i] = 0$ for $i > t$.

3. Compute the complex attention-like matrix:
   $$W = (Q / \sqrt{d}) \cdot K^{*\top}$$

4. Apply decay: $A = W \odot D$

5. Compute output: $Y = A \cdot V'$

This is mathematically equivalent to the recurrent form but parallelizes across the sequence dimension. For $T = 2048$, this is efficient on modern GPUs.

At the end of training forward passes, we also compute the final recurrent state $S_T$ from the decay matrix, so that generation can continue from a prompt:

$$S_T = \sum_{i=1}^{T} D[T, i] \cdot (V'_i \otimes K_i^*)$$

#### 3.4.8 Recurrent form (inference)

During autoregressive generation, each token is processed in $O(1)$ time:

1. Compute $Q_t, K_t, V'_t, \gamma_t$ from the input.
2. Update state: $S_t = \gamma_t \cdot S_{t-1} + V'_t \otimes K_t^*$
3. Retrieve: $Y_t = S_t \cdot (Q_t / \sqrt{d})$

The state $S \in \mathbb{C}^{H \times d \times d}$ is fixed-size and does not grow with sequence length.

#### 3.4.9 Stacking

PAM layers are stacked with pre-normalization, residual connections, and learned layer scaling:

$$z^{(l)} = z^{(l-1)} + \alpha_l \cdot \text{PAM}_l(\text{ComplexNorm}(z^{(l-1)}))$$

Layer scales $\alpha_l$ are initialized to $0.1$ to stabilize early training. A final ComplexNorm is applied after the last PAM layer.

### 3.5 Output head

The output projection computes logits via a tied complex inner product with the embedding table:

$$\text{logits} = \text{Re}(z_\text{out} \cdot \text{conj}(E))$$

where $E$ is the complex embedding matrix (shared with the input embedding). Concretely:

$$\text{logits} = z_{out,r} \cdot E_r^\top + z_{out,i} \cdot E_i^\top$$

This weight tying reduces parameters and maintains algebraic consistency.

---

## 4. Experimental Setup

### 4.1 Dataset

We train and evaluate on WikiText-103 (Merity et al., 2017), a standard language modeling benchmark containing approximately 103 million tokens of Wikipedia text. We use the official train/validation splits, tokenized with the GPT-2 BPE tokenizer (vocabulary size 50,257).

### 4.2 Model configuration

| Parameter | Value |
|---|---|
| Complex dimension ($D$) | 384 |
| CGU layers | 16 |
| CGU expansion factor | 3 |
| PAM layers | 16 |
| PAM heads ($H$) | 6 |
| PAM head dimension ($d$) | 64 |
| GSP | enabled |
| Sequence length | 2048 |
| Total parameters | 100.4M |
| Working/internal memory | disabled |
| Attention | disabled |

### 4.3 Training details

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | $3 \times 10^{-5}$ |
| Weight decay | 0.01 |
| Warmup steps | 500 |
| LR schedule | warmup + cosine decay |
| Batch size | 3 |
| Epochs | 10 |
| Gradient clipping | 1.0 |
| Precision | automatic mixed precision (bf16) |
| Hardware | single NVIDIA RTX 4090 (24GB) |
| Compilation | torch.compile (default mode) |
| Initialization | orthogonal (complex linear maps) |

### 4.4 Generation settings (qualitative evaluation)

During training, generation samples are logged every 5,000 steps using:
- Temperature: 1.0
- Top-k: 50
- Top-p: 0.9
- Repetition penalty: 1.2

---

## 5. Results

### 5.1 Main result

The PAM model reaches validation perplexity **38.95** after 10 epochs of training on WikiText-103.

| Epoch | Train PPL | Val PPL |
|---|---|---|
| 1 | -- | -- |
| 2 | -- | -- |
| 3 | -- | -- |
| 4 | 54.03 | 47.19 |
| 5 | 48.55 | 43.55 |
| 6 | 45.26 | 41.43 |
| 7 | 43.14 | 40.11 |
| 8 | 41.76 | 39.34 |
| 9 | 40.91 | 39.02 |
| 10 | 40.50 | **38.95** |

Note: Epochs 1--3 had correct PPL but a generation bug (Section 5.4). The model was resumed from the epoch 3 checkpoint with the bug fixed; PPL numbers from epochs 1--3 remain valid but are omitted here as they were computed on a slightly different code path. Epochs 4--10 ran with the corrected code.

Total training wall time: approximately 9.9 hours on a single RTX 4090. Throughput: approximately 23,500 tokens/second.

### 5.2 Ablation across V6 architectures

The following table shows the progression of V6 architectures on WikiText-103, all trained for 10 epochs with sequence length 2048:

| Model | Architecture | Params | Val PPL |
|---|---|---|---|
| medium-rebalanced | CGU + vector-state SSM | 58.4M | 44.47 |
| medium-rebalanced-gsp | CGU + SSM + GSP | 63.2M | 41.67 |
| medium-rebalanced-hsb | CGU + SSM + GSP + HSB | 86.8M | 43.54 |
| **medium-pam** | **CGU + PAM + GSP** | **100.4M** | **38.95** |

Key observations:

- **GSP helps**: Adding Gated State Protection to the SSM baseline reduces perplexity from 44.47 to 41.67 (6.3% improvement).
- **HSB hurts**: Holographic State Binding, which attempts to inject HRR-style key-value bindings into the vector state, causes a regression to 43.54. This confirms the interference hypothesis: vector states lack the capacity for multiple simultaneous associations.
- **PAM works**: Replacing the vector-state SSM entirely with matrix-state PAM yields the best result at 38.95 (6.5% improvement over GSP).

The parameter counts are not identical across these models. PAM's 100.4M includes a larger model dimension (384 vs 192) to utilize the parameter budget effectively. A controlled parameter-matched comparison is listed as future work.

### 5.3 Comparison to GPT-2

| Model | Type | Params | Val PPL (WT-103) |
|---|---|---|---|
| GPT-2 | Transformer | 124M | ~31 |
| **PAM (ours)** | **Complex recurrent** | **100M** | **38.95** |

PAM does not match GPT-2. The gap is approximately 25% in perplexity terms. We present this comparison for honest calibration, not as a claim of competitiveness.

However, several structural differences are worth noting:

- **Inference cost**: GPT-2 attention is $O(T)$ per token (due to KV cache lookup); PAM is $O(1)$ per token with fixed-size state.
- **Memory footprint**: GPT-2 requires a KV cache that grows linearly with sequence length; PAM's state is fixed at $H \times d^2 \times 2$ floats per layer regardless of sequence length.
- **Architecture maturity**: GPT-2 represents years of transformer optimization, including extensive hyperparameter tuning, large-scale pretraining recipes, and hardware-optimized implementations. PAM is a first-generation implementation in pure PyTorch with no custom kernels.
- **Training budget**: PAM was trained for 10 epochs on a single RTX 4090 in under 10 hours. We have not explored scaling behavior, longer training, or larger models.

We do not claim these structural advantages compensate for the perplexity gap. We note them as properties that could become relevant if the architecture continues to improve.

### 5.4 Generation quality

The model produces coherent, multi-sentence text with factual structure (dates, institution names, locations), though factual accuracy is not guaranteed.

**Prompt**: "In 1923 , the University of"

**Generated text (epoch 10)**:
> In 1923 , the University of Missouri and the University of Michigan was also established in 1926 . In 1928 , the University of Michigan opened its current campus with the school 's first campus opening at St. Louis Road on the northern end of Lake Michigan in 1929 .

**Earlier generation (epoch 10, step 15000)**:
> In 1923 , the University of Kentucky opened a public schools school in 1924 and served as the state 's first governor for 40 years . During the 1930s , a public school was built in a public house to serve as the primary school for Governor , but it still served as an office space until 1933 . This was completed by 1936 , when the University of Kentucky passed its own law class .

Quality metrics at final epoch: 3-gram repetition rate 0.051, 4-gram repetition rate 0.020, zero restarts, unique token ratio 0.624.

The text is grammatically coherent and shows structural awareness (dates, proper nouns, institutional language). Factual accuracy is not reliable -- this is expected for a 100M-parameter model.

### 5.5 Earlier validation: TinyStories

Before WikiText-103, the broader V6 architecture family was validated on TinyStories (Eldan & Li, 2023):

| Model | Config | Params | Dataset | Val PPL |
|---|---|---|---|---|
| V6 small-matched | CGU + SSM, no memory | 28.7M | TinyStories (full) | 5.50 |
| V6 small-matched | CGU + SSM, WM=16 | 29.2M | TinyStories (full) | 2.23 |

These results established that the phase-preserving, attention-free architecture family can learn effectively. The TinyStories experiments also revealed that working memory capacity is a critical control variable: too much memory induces memorization, while zero memory produces cleaner but higher-perplexity generations.

---

## 6. Analysis

### 6.1 Why HSB failed and PAM works

The progression from SSM to HSB to PAM illustrates a specific failure mode and its solution.

**Holographic State Binding (HSB)** attempted to inject key-value associations into the SSM's vector state using complex multiplication (the HRR binding operation). Mathematically, multiple bindings are superposed additively in the same $d$-dimensional vector. When only a few associations are present, retrieval via conjugate multiplication works. When many associations accumulate over a long sequence, they interfere destructively -- the signal-to-noise ratio degrades as $O(1/\sqrt{n})$ for $n$ stored associations.

**PAM** solves this by replacing the vector state with a matrix state. The outer product $V_t \otimes K_t^*$ writes each association into a separate subspace of the $d \times d$ matrix. Multiple associations can coexist without interfering because they occupy different rank-1 components of the matrix. The capacity is $O(d^2)$ per head rather than $O(d)$.

This diagnosis -- vector state causes interference, matrix state fixes it -- is supported by the experimental evidence: HSB regresses from the SSM baseline, while PAM improves over it.

### 6.2 Phase interference as implicit attention

In standard attention, the softmax over $QK^\top / \sqrt{d}$ produces a probability distribution that weights values. In PAM, the complex dot product $K_i^* \cdot Q_t$ serves an analogous role:

$$\text{Attention: } \alpha_{ti} = \frac{\exp(Q_t \cdot K_i / \sqrt{d})}{\sum_j \exp(Q_t \cdot K_j / \sqrt{d})}$$

$$\text{PAM: } w_{ti} = \left(\prod_{j=i+1}^{t} \gamma_j\right) \cdot (K_i^* \cdot Q_t) / \sqrt{d}$$

The key differences:

1. PAM weights are **complex-valued**, not positive real. This means retrieval can involve both addition and cancellation, potentially providing richer composition.
2. PAM weights are **not normalized**. There is no softmax or equivalent. This removes a nonlinearity but also removes the guarantee that weights sum to 1.
3. PAM weights include a **decay factor** that naturally downweights distant positions, providing a soft locality bias that attention must learn.
4. PAM weights depend on **phase alignment**, a geometric property of complex vectors that standard real-valued attention lacks.

Whether these differences are advantageous at scale is an open question. The current evidence is that they produce a trainable system with reasonable perplexity, but we cannot yet determine whether phase interference provides benefits beyond what real-valued matrix states achieve.

### 6.3 Computational cost

**Training**: The dual form computes $W = QK^{*\top}$ as a $T \times T$ matrix per head, giving $O(T^2 H d)$ cost per layer -- the same order as standard attention. For $T = 2048$, this is efficient on modern GPUs. Training throughput is approximately 23,500 tokens/second on a single RTX 4090.

**Inference**: The recurrent form processes each token in $O(Hd^2)$ time (one matrix-vector multiply per head). With $H = 6$ and $d = 64$, this is 24,576 complex multiply-adds per layer. The state is $H \times d \times d \times 2$ floats = 6 * 64 * 64 * 2 = 49,152 floats per layer, regardless of sequence length.

For comparison, GPT-2's KV cache stores $2 \times H \times d \times T$ values per layer, growing linearly with $T$. At $T = 2048$, GPT-2's cache per layer is $2 \times 12 \times 64 \times 2048 = 3{,}145{,}728$ values. PAM's state per layer is fixed at 49,152 values, a factor of 64x smaller at this sequence length.

### 6.4 The role of GSP

GSP's protect gate allows the model to selectively override decay, retaining specific state dimensions indefinitely. This is conceptually similar to the forget gate in LSTMs but operates at the level of the entire $d \times d$ state matrix per head. Empirically, GSP improves the SSM baseline from 44.47 to 41.67 perplexity, and PAM uses GSP by default.

The protect gate also modulates the value signal ($V' = V \cdot (1 - p)$), preventing new values from overwriting protected dimensions. This creates a natural partitioning of state into "active" dimensions (low protection, normal read/write) and "frozen" dimensions (high protection, retained over long spans).

---

## 7. Limitations

We list limitations explicitly and without hedging.

### 7.1 No parameter-matched transformer baseline

Our comparison to GPT-2 uses publicly reported numbers. We have not trained a transformer baseline at exactly 100M parameters with the same data, tokenizer, sequence length, and training budget. A rigorous comparison would require this.

### 7.2 Single training run

The reported results come from a single training run. We have not measured variance across seeds, explored extensive hyperparameter tuning, or conducted architecture search. The current hyperparameters (learning rate $3 \times 10^{-5}$, 500 warmup steps, batch size 3) were chosen based on limited experimentation, not systematic optimization.

### 7.3 Small scale

100M parameters and WikiText-103 are modest by current standards. We do not know how PAM scales to billions of parameters or web-scale corpora. The $O(d^2)$ state per head could become a memory concern at very large head dimensions, though for the current $d = 64$ it is negligible.

### 7.4 No downstream evaluation

We report perplexity and qualitative generation only. Standard downstream benchmarks (LAMBADA, HellaSwag, etc.) have not been evaluated.

### 7.5 Pure PyTorch implementation

PAM is implemented in pure PyTorch with no custom CUDA kernels. The dual form uses standard matrix multiplications, which are efficient, but the recurrent form's sequential loop is not optimized. Custom kernels for the state update and retrieval could improve inference throughput substantially.

### 7.6 Complex arithmetic overhead

Complex linear maps require 4x the multiply-adds of real linear maps (since $(a+bi)(c+di) = (ac-bd) + (ad+bc)i$). Our ComplexLinear uses four `F.linear` calls per forward pass. Whether the representational advantages of complex computation compensate for this overhead at scale is unresolved.

### 7.7 Memory hierarchy not validated in this run

The V6 codebase includes working memory, internal memory, persistent memory, expert memory, and session memory modules. The PAM run reported here disables all of these (no working memory, no internal memory) to isolate PAM's contribution. Whether combining PAM with memory modules improves results is an open question.

---

## 8. Future Work

### 8.1 Scaling

The most immediate question is whether PAM's perplexity gap to transformers narrows or widens with scale. We plan to train larger PAM models (300M--1B parameters) and compare against matched transformer baselines on the same data and compute budget.

### 8.2 Custom kernels

The recurrent inference form's sequential loop is a clear optimization target. A fused CUDA kernel for the complex matrix state update and retrieval could bring PAM's wall-clock inference speed closer to its theoretical advantage.

### 8.3 Long-context evaluation

PAM's fixed-size state and absence of KV cache make it theoretically well-suited for long sequences. We plan to evaluate on PG-19 and other long-document benchmarks to test whether the $O(d^2)$ state capacity is sufficient for practical long-context modeling.

### 8.4 Memory hierarchy integration

The V6 codebase includes a memory hierarchy (working, internal, persistent, expert) with phase-coherence retrieval. These modules were disabled for the PAM evaluation to isolate the core contribution. Integrating PAM with working memory and evaluating on tasks that require explicit fact storage is a natural next step.

### 8.5 Hybrid architectures

PAM could be combined with sparse attention layers (e.g., every $k$-th layer) to create a hybrid architecture. The V6 codebase already supports optional PhaseAttention layers. Whether a small number of attention layers can close the gap to transformers while preserving PAM's inference advantages is worth investigating.

### 8.6 Multimodal extensions

The shared backbone design in V6 was built to support both autoregressive and diffusion modes on the same architecture. Extending PAM to image generation or multimodal modeling is architecturally straightforward but experimentally unvalidated.

---

## 9. Conclusion

Phase-Associative Memory is a recurrent sequence modeling architecture that processes information through complex-valued phase interference rather than softmax attention. Its key properties are:

1. A complex matrix state $S \in \mathbb{C}^{H \times d \times d}$ that provides $O(d^2)$ associative capacity per head.
2. Retrieval via complex dot product $K^* \cdot Q$, where phase alignment between query and stored key determines recall strength -- constructive interference amplifies, destructive interference suppresses.
3. Gated State Protection that allows selective state freezing.
4. A dual computational form: $O(T^2)$ parallel training, $O(1)$ per-token inference with no KV cache.

At 100M parameters on WikiText-103, PAM achieves validation perplexity of 38.95, producing coherent multi-sentence text. This does not match GPT-2's approximately 31 perplexity at 124M parameters, and we do not claim it does. What we do claim is that PAM represents a genuinely different approach to sequence modeling -- one built entirely on complex-valued computation and phase-geometric retrieval -- that has reached a level of performance where continued research is warranted.

The transition from vector-state SSMs (which suffered from interference) to matrix-state PAM (which resolved it) validates the core architectural hypothesis: complex-valued matrix states with phase-interference retrieval can serve as a viable sequence processing mechanism. Whether this approach can scale to match or exceed transformer performance remains an open and, we believe, worthwhile question.

---

## Appendix A. Hyperparameter Details

### A.1 medium-pam configuration

```
vocab_size:           50,257
complex_dim:          384
num_layers:           16
single_bank:          True
bank_expand:          3 (CGU hidden = 1,152)
state_dim:            0 (SSM disabled)
pam_num_heads:        6
pam_head_dim:         64
gated_state_protection: True
dropout:              0.1
init_strategy:        orthogonal
```

### A.2 Training configuration

```
dataset:              WikiText-103
tokenizer:            GPT-2 BPE (50,257 tokens)
seq_len:              2,048
batch_size:           3
epochs:               10
optimizer:            AdamW
learning_rate:        3e-5
weight_decay:         0.01
warmup_steps:         500
lr_schedule:          warmup + cosine decay
gradient_clip:        1.0
amp_dtype:            bf16 (automatic)
compile:              torch.compile (default mode)
hardware:             1x NVIDIA RTX 4090 (24GB)
```

### A.3 Parameter breakdown

| Component | Parameters | Percentage |
|---|---|---|
| Embedding (tied with output) | ~38.6M | 38.4% |
| CGU feature layers (16 layers) | ~21.3M | 21.2% |
| PAM layers (16 layers) | ~40.4M | 40.2% |
| LM head projection + norm | ~0.3M | 0.3% |
| **Total** | **~100.4M** | **100%** |

Note: the embedding accounts for a large fraction because dim=384 with vocab_size=50,257 requires $50{,}257 \times 384 \times 2 = 38.6$M parameters (real + imaginary embedding tables). These are tied with the output layer.

---

## Appendix B. Training Curves

### B.1 Validation perplexity by epoch

| Epoch | Train Loss | Train PPL | Val Loss | Val PPL |
|---|---|---|---|---|
| 4 | -- | 54.03 | -- | 47.19 |
| 5 | -- | 48.55 | -- | 43.55 |
| 6 | -- | 45.26 | -- | 41.43 |
| 7 | -- | 43.14 | -- | 40.11 |
| 8 | -- | 41.76 | -- | 39.34 |
| 9 | -- | 40.91 | -- | 39.02 |
| 10 | 3.7013 | 40.50 | 3.6623 | **38.95** |

The train-validation gap narrows over training (40.50 vs 38.95 at epoch 10), indicating that overfitting is not a major concern at this scale and training duration.

### B.2 Wall time

| Metric | Value |
|---|---|
| Total wall time | 35,657.6 seconds (~9.9 hours) |
| Per-epoch time | ~5,040 seconds (~84 minutes) |
| Throughput | ~23,500 tokens/second |
| GPU memory usage | 2.6 / 21.0 GB |

---

## Appendix C. Architecture Evolution

This appendix documents the architectural decisions that led to PAM, for reproducibility and to help others avoid the same dead ends.

### C.1 V5: the phase-preservation lesson

V4 introduced complex-valued tokens but passed them through real-valued activations (GELU, sigmoid), destroying phase information. V5 replaced these with phase-preserving primitives (modReLU, ComplexGatedUnit, ComplexNorm). This single change improved a 28.7M model from V4-class performance to TinyStories validation perplexity 5.59.

**Lesson**: complex-valued computation is only useful if phase is preserved throughout the signal path.

### C.2 Multi-timescale SSM

The initial V6 recurrent backbone was a complex SSM with explicit fast/medium/slow decay lanes (40%/30%/30% of state dimensions). This provided a structured prior over timescales but suffered from limited associative capacity in the vector state.

### C.3 Holographic State Binding (HSB)

HSB attempted to solve the capacity problem by using HRR-style binding/unbinding within the SSM state. Key-value pairs were bound via complex multiplication and added to the SSM input; retrieval used conjugate multiplication against the state.

**Result**: val PPL 43.54, a regression from the 41.67 GSP baseline. The root cause was interference: multiple bindings in the same vector state collide, degrading retrieval quality as more facts are stored.

### C.4 PAM: the matrix state solution

PAM replaces the vector state entirely with a matrix state, providing $O(d^2)$ capacity per head. The outer product write and matrix-vector retrieval avoid the interference problem because different associations can occupy different rank-1 subspaces of the matrix.

**Result**: val PPL 38.95, a clear improvement over all previous V6 configurations.

---

## Appendix D. Reproducibility

### D.1 Software

- Python 3.11+
- PyTorch 2.x with CUDA support
- Package management via `uv`

### D.2 Running the experiment

```bash
# Install dependencies
uv sync && uv sync --extra cuda

# Full training run
./scripts/run_v6_medium_pam.sh

# Resume from checkpoint
./scripts/run_v6_medium_pam.sh --resume
```

### D.3 Generation

```bash
python -m v6.generate \
  --checkpoint checkpoints_v6_medium_pam/best_model.pt \
  --prompt "In 1923 , the University of"
```

---

## References

> Note: formal citations to be added in the LaTeX version. Key references to include:

- Vaswani, A., et al. "Attention is all you need." NeurIPS 2017.
- Gu, A., et al. "Efficiently modeling long sequences with structured state spaces." ICLR 2022. (S4)
- Gu, A. & Dao, T. "Mamba: Linear-time sequence modeling with selective state spaces." 2023.
- Dao, T. & Gu, A. "Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality." 2024. (Mamba-2)
- Katharopoulos, A., et al. "Transformers are RNNs: Fast autoregressive transformers with linear attention." ICML 2020.
- Sun, Y., et al. "Retentive network: A successor to transformer for large language models." 2023. (RetNet)
- Yang, S., et al. "Gated linear attention transformers with hardware-efficient training." 2024. (GLA)
- Arjovsky, M., et al. "Unitary evolution recurrent neural networks." ICML 2016. (modReLU)
- Trabelsi, C., et al. "Deep complex networks." ICLR 2018.
- Plate, T. "Holographic reduced representations." IEEE Transactions on Neural Networks, 1995.
- Eldan, R. & Li, Y. "TinyStories: How small can language models be and still speak coherent English?" 2023.
- Merity, S., et al. "Pointer sentinel mixture models." ICLR 2017. (WikiText-103)
- Radford, A., et al. "Language models are unsupervised multitask learners." OpenAI, 2019. (GPT-2)
- Hirose, A. "Complex-valued neural networks." Springer, 2012.
