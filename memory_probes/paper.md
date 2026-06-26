# Measuring Memory Independently of Intelligence

## A Standardized Probe Suite for Neural Memory Architectures

**Reference implementation:** the probe suite described here was developed alongside a new associative-memory architecture, **PAM (Phase-Associative Memory)**, and is released as part of [`https://github.com/gowrav-vishwakarma/qllm2`](https://github.com/gowrav-vishwakarma/qllm2) (see `memory_probes/`). PAM is used throughout as the concrete worked example, but every definition and probe in this paper is **architecture-agnostic**: the suite can be applied unchanged to attention KV caches, state-space models, linear-attention / fast-weight memories, or any future architecture that maintains an internal state.

---

### Abstract

Large Language Models are almost always evaluated through downstream *intelligence* benchmarks: perplexity, reasoning, coding, mathematics, and long-context retrieval. These benchmarks measure the quality of the final behavior, but they entangle several distinct capabilities into a single score, making it hard to isolate the contribution of the underlying **memory** system.

We argue that memory and intelligence are distinct but tightly coupled computational properties. *Intelligence* determines **how** information is processed; *memory* determines **what** information remains available to be processed. A model with insufficient memory cannot reason about information it has already forgotten, while a model with perfect memory may still fail through poor reasoning. A single benchmark score cannot tell these failure modes apart.

We therefore propose evaluating memory as an independent subsystem. Instead of measuring downstream language performance, we define a battery of architecture-level probes that quantify implementation correctness, binding capacity, persistence, interference, effective rank, retrieval fidelity, and long-context behavior. We give exact mathematical definitions for each probe, a compact NumPy/PyTorch reference implementation, and an empirical validation on the PAM architecture. The probes are deterministic (`seed=42`, bit-reproducible for the synthetic battery) and require no trained checkpoint for the core suite. The framework is intended as a foundation for principled, apples-to-apples comparison of neural memory mechanisms.

---

# 1. Introduction

Neural architectures are commonly evaluated using tasks that require both memory and reasoning. Next-token prediction, question answering, code generation, mathematical reasoning, and long-context retrieval all simultaneously depend on three different functions:

1. the ability to **retain** information,
2. the ability to **retrieve** relevant information,
3. the ability to **transform** retrieved information into an answer.

These capabilities are usually discussed together under the broad heading of "intelligence." However, they are different computational functions. We write intelligence as a composition:

$$
\text{Intelligence} = f(\text{Memory},\ \text{Computation})
$$

where **Memory** ($M$) determines what information remains accessible, and **Computation** ($C$) determines how that accessible information is transformed. Current benchmarks measure only the composed value $f(M, C)$, never $M$ in isolation. This makes architectural analysis difficult: when a model fails, the score alone cannot attribute the failure to forgetting versus faulty reasoning.

This paper isolates $M$. We treat the memory subsystem as a measurable object with its own intrinsic properties, independent of the reasoning that consumes it.

---

# 2. Why Separate Memory From Intelligence?

Consider two systems.

- **System A** retains every relevant fact but reasons poorly.
- **System B** reasons perfectly but has forgotten the required information.

Both fail the same downstream benchmark, for opposite reasons. The benchmark score cannot distinguish them, so it cannot tell a researcher whether to invest in:

- reasoning / computation,
- optimization and training scale,
- memory dynamics (decay, gating, write rules),
- retrieval mechanisms,
- raw architectural capacity.

An independent memory benchmark isolates these effects. By analogy with computer-systems evaluation, where processor performance and memory performance are characterized separately before any application benchmark is run, we propose characterizing a neural architecture's memory before — and independently of — its end-task behavior.

---

# 3. Memory as a Dynamical System

Most modern sequence architectures maintain an internal state $S_t$ that evolves as tokens arrive:

$$
S_t = F(S_{t-1},\, x_t)
$$

where $x_t$ is the incoming token representation and $S_t$ is the memory at step $t$. Different architectures simply choose different $F$ and different state shapes:

| Architecture | State $S_t$ | Update $F$ | Cost in $t$ |
|---|---|---|---|
| Transformer (KV cache) | $\{(k_i, v_i)\}_{i\le t}$ | append $(k_t, v_t)$ | $O(t)$ memory, $O(t)$ read |
| State-space model (SSM) | vector $h_t \in \mathbb{R}^n$ | $h_t = A h_{t-1} + B x_t$ | $O(n)$ |
| Linear attention / fast weights | matrix $S_t \in \mathbb{R}^{d\times d}$ | $S_t = \gamma_t S_{t-1} + v_t k_t^{\top}$ | $O(d^2)$, constant in $t$ |
| **PAM (this work)** | complex matrix $S_t \in \mathbb{C}^{d\times d}$ | $S_t = \gamma_t S_{t-1} + v_t k_t^{*}$ | $O(d^2)$, constant in $t$ |

Regardless of implementation, **the behavior of $S_t$ is what we evaluate.** The probes below depend only on the ability to (a) write a key–value association into $S$ and (b) read it back with a query — interfaces that every architecture above supports.

## 3.1 PAM: the reference instantiation

PAM stores associations as a sum of complex outer products and retrieves by complex matrix–vector product. With key $k_t \in \mathbb{C}^d$, value $v_t \in \mathbb{C}^d$, query $q_t \in \mathbb{C}^d$, and scalar (or per-channel) decay $\gamma_t$:

$$
S_t = \gamma_t\, S_{t-1} + v_t\, k_t^{*},
\qquad
y_t = S_t\, q_t
$$

where $k_t^{*}$ is the complex conjugate (transpose) of the key. The conjugate is what makes retrieval an *unbinding* operation: querying with $q_t \approx k_i$ aligns the stored phase and reads out $v_i$.

The decay $\gamma_t \in (0, 1]$ is produced by a softplus head, giving a per-step "forget gate":

$$
\gamma = e^{-\Delta t},
\qquad
\Delta t = \mathrm{softplus}\!\big(W x + b\big)
$$

With the default bias $b = -4$ and zero input, $\Delta t = \mathrm{softplus}(-4) = \ln(1 + e^{-4}) \approx 0.0181$, so $\gamma \approx e^{-0.0181} \approx 0.982$. This default value is referenced repeatedly below: it decays aggressively over thousands of tokens unless a protection mechanism intervenes.

PAM also defines three optional dynamics, each of which the probe suite can target:

- **Gated State Protection (GSP).** A protect gate $p \in [0,1]$ convex-combines decay toward $1$ and scales the written value, so a token can choose *not* to overwrite memory:

$$
\gamma = e^{-\Delta t}(1 - p) + p,
\qquad
v' = v\,(1 - p)
$$

  At $p = 1$ the state is frozen ($\gamma = 1$, $v' = 0$); at $p = 0$ it reduces to the baseline.

- **Delta-rule write (error-correcting).** Instead of blind addition, erase the stale binding for $k_t$ before writing, with write strength $\beta_t \in (0,1)$:

$$
u_t = \beta_t\big(v_t - S\,k_t\big),
\qquad
S \mathrel{+}= u_t\, k_t^{*}
$$

- **Multi-state superposition.** $K$ states with distinct decays are kept in parallel and recombined with learned per-state phases $\phi_k$ at read time:

$$
y_t = \sum_{k=1}^{K} e^{i\phi_k}\, S_k\, q_t
$$

These four equations (baseline, GSP, delta, multi-state) are the exact forms implemented in `V11PAMLayer` (`v11/model.py`) and mirrored by the NumPy reference in `memory_probes/core.py`; the probes are validated against both (Section 4.1).

---

# 4. Probe Definitions

Each probe targets one intrinsic property of $S_t$. All probes share a small set of primitives, given here once (NumPy reference, `memory_probes/core.py`):

```python
def outer_v_kstar(v, k):          # write primitive:  v (x) k*
    return np.outer(v, np.conj(k))

def pam_step_additive(S, g, v, k):  # S = g*S + v (x) k*   (g scalar or per-channel)
    if np.ndim(g) == 0:
        return g * S + outer_v_kstar(v, k)
    return S * g[np.newaxis, :] + outer_v_kstar(v, k)

def retrieve_score(S, k_query, v_target):  # |<S q, v>|
    y = S @ k_query
    return float(np.abs(np.vdot(y, v_target)))
```

The **relative retrieval** score normalizes against a fresh single-write baseline, so $1.0$ means "as good as if nothing else had been written":

$$
\mathrm{rel}(S; k, v) = \frac{\big|\langle S q,\, v\rangle\big|}{\big|\langle (v k^{*}) k,\, v\rangle\big|}
$$

## 4.1 Implementation correctness (`selftest`, `layer-bridge`)

Any architecture with both a **parallel training** form and an **$O(1)$ recurrent inference** form must produce identical outputs from both. This is a prerequisite for all other measurements: if the two forms disagree, downstream scores are meaningless.

$$
\max_t \big| y_t^{\text{parallel}} - y_t^{\text{recurrent}} \big| \approx 0
$$

The probe runs the same input through (a) the chunked/dual parallel kernel and (b) the step-by-step recurrence, for every dynamics mode (baseline, per-channel decay, delta, multi-state), and asserts agreement to floating-point tolerance. It additionally bridges the NumPy reference math to the PyTorch layer step (`_recur_step_additive`) to catch implementation drift.

## 4.2 Binding capacity (`capacity.py`)

**Question.** How many independent key–value pairs can be stored in one state before retrieval degrades?

Write $N$ random associations into a single state and measure top-1 retrieval accuracy (argmax over all stored values). The matrix memory is compared against a vector **Holographic Reduced Representation (HRR)** baseline, which binds via circular convolution and whose retrieval signal-to-noise scales as $O(1/\sqrt{N})$:

```python
def hrr_bind(a, b):                 # circular convolution (FFT)
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))

def hrr_unbind(state, key):         # correlation = bind with conjugate spectrum
    return np.fft.ifft(np.fft.fft(state) * np.conj(np.fft.fft(key)))
```

The matrix state stores $S = \sum_i v_i k_i^{*}$ and retrieves $y_i = S k_i$; the vector HRR stores $s = \sum_i a_i \circledast b_i$. The theoretical vector ceiling is $1/\sqrt{N}$.

## 4.3 Persistence (`persistence.py`)

**Question.** How does retrieval quality decay as temporal distance grows? Write one association, then apply $T$ filler writes under decay $\gamma$, and measure relative retrieval:

$$
R(T) = \frac{\mathrm{retrieval}(T)}{\mathrm{retrieval}(0)}
$$

This separates two effects: pure geometric decay $\gamma^T$, and interference from the filler writes themselves. With $\gamma = 1$ there is no decay but filler interference still accumulates; with $\gamma < 1$ the needle additionally fades as $\gamma^T$.

## 4.4 Interference (`interference.py`)

**Question.** How much does writing new information corrupt existing memories? Write $N$ pairs, then $M$ filler tokens, then query the first pair. The probe sweeps $(N, M)$ and compares additive, delta, and multi-state writes.

It also measures **conjugate retrieval cross-talk**: for random keys/queries the off-diagonal score $W_{ts} = \langle q_t, k_s^{*}\rangle$ should be destructive (negative real part) about half the time, reflecting phase cancellation:

$$
\text{frac\_destructive} = \Pr\big[\,\mathrm{Re}\,W_{ts} < 0\,\big] \approx 0.5
$$

The same statistic is computed on untrained PAM-layer projections (layer bridge) to confirm the learned projection preserves the expected phase geometry.

## 4.5 Effective rank (`rank.py`)

**Question.** How much of the available memory subspace is actually used? Rather than the raw matrix rank, we use the **entropy (effective) rank** of the singular spectrum $\{\sigma_i\}$ of $S$:

$$
r_{\mathrm{eff}}(S) = \exp\!\Big(-\textstyle\sum_i p_i \log p_i\Big),
\qquad
p_i = \frac{\sigma_i}{\sum_j \sigma_j}
$$

This is smooth and bounded by $d$; it equals $1$ when one singular value dominates and approaches $d$ when the spectrum is flat.

```python
def effective_rank(S):
    sv = np.linalg.svd(S, compute_uv=False)
    p = sv / (sv.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p)))) if p.size else 0.0
```

The probe tracks $r_{\mathrm{eff}}(S_t)$ across writes for (a) distinct random keys, (b) a repeated overwrite key, and (c) a real WikiText stream pushed through a PAM layer. Distinct keys should expand the subspace; overwriting one key should not.

## 4.6 Long context / needle-in-haystack (`long_context.py`)

Because the recurrent state is $O(d^2)$ **per layer** with no growth in $t$, there is no hard context cap in the math; the probe sweeps distances up to $10^6$ tokens. It reports three regimes:

- **Analytic decay-only envelope:** exact $\gamma^T$ (no filler writes) — an upper bound for the bare decay.
- **Filler decay (NIAH):** needle written, then $T$ random filler writes under $\gamma$ — the realistic interference regime.
- **GSP ceiling:** filler tokens carry protection $p$. At $p = 1$ filler writes are skipped and the state freezes ($\gamma_{\text{fill}} = e^{-\Delta t}(1-p)+p = 1$), giving the **mechanism ceiling** — the best a perfectly-trained gate could do.

It additionally measures a position grid (needle at relative position $r \in [0,1]$), multi-needle survival, and key-collision (filler keys deliberately correlated with the needle key).

## 4.7 Language filler (`language.py`)

Random unit vectors satisfy $\mathbb{E}[k_i \cdot k_j] \approx 0$. Real language embeddings **cluster** ($\mathbb{E}[|k_i \cdot k_j|] > 0$), a more realistic and potentially harsher interference regime. The probe writes a synthetic needle (`glorp → banana`) via GPT-2 embeddings projected to complex $K/V$, streams WikiText-103 as the haystack, queries with `glorp`, and compares against random filler of equal length. A projection-seed sweep checks that the result is a property of language structure, not of one lucky projection.

---

# 5. Reference Implementation

The suite is a small, dependency-light Python package. Core synthetic probes are pure NumPy; language and real-text rank probes use GPT-2 embeddings and (optionally) a trained PAM checkpoint.

| Module | Probe |
|---|---|
| `selftest.py` | Implementation correctness (parallel ≡ recurrent) |
| `capacity.py` | Binding capacity vs vector HRR |
| `persistence.py` | Association survival vs distance under decay |
| `interference.py` | Multi-association interference, conjugate retrieval, layer bridge |
| `rank.py` | Effective rank (synthetic + real WikiText) |
| `language.py` | Language filler (clustered embeddings) |
| `long_context.py` | Needle-in-haystack and extreme-distance sweeps |
| `core.py` | Shared complex-matrix reference math |
| `adapters.py` | Architecture adapters (PAM / Transformer / Mamba) |
| `compare.py` | Cross-architecture state-utilization comparison on real text |
| `cli.py` | Unified CLI |

**Usage (from the repository root):**

```bash
# Full battery (selftest + all probes)
./scripts/run_memory_probes.sh

# Individual probes
python -m memory_probes --test binding
python -m memory_probes --test persistence --distances 64,128,512,2048
python -m memory_probes --test niah --distances 64,128,512,1024,2048
python -m memory_probes --test long-context --max-distance 1048576   # 1M tokens
python -m memory_probes --test language-filler --projection-trials 50
python -m memory_probes --test rank-text --text-tokens 50000 --sample-every 100

# Other architectures (see Section 6)
python -m memory_probes --arch transformer --test binding
python -m memory_probes --arch mamba --test rank --arch-dim 32
python -m memory_probes --test arch-compare --text-tokens 2000   # PAM vs SSM vs Transformer
```

**Determinism.** The synthetic probes (binding, persistence, interference, NIAH, long-context, synthetic rank) are bit-exact at `seed=42`. The layer-bridge and selftest use fixed Torch seeds. Real-text rank uses an untrained layer by default (weights re-init each run); pass `--checkpoint` for trained, reproducible weights. JSON results are written to `logs/memory_probes/`.

---

# 6. Applying the Suite to Other Architectures

The probes depend only on a tiny associative-memory interface, so the same battery runs on any architecture that can write key–value associations and read them back, or that exposes a state we can inspect. The suite ships three reference adapters (`memory_probes/adapters.py`) selected with `--arch {pam,transformer,mamba}`:

```python
class MemoryAdapter(ABC):
    associative: bool   # supports write(k, v) / read(q)
    stateful: bool      # supports state() -> 2D matrix (for effective rank)

    def reset(self): ...
    def write(self, k, v, gamma=1.0): ...   # store association (decay gamma first)
    def read(self, q): ...                   # retrieve value estimate (associative only)
    def state(self): ...                     # current memory as a matrix
```

Architectures expose what is native to them, so the suite distinguishes two **tiers**:

- **Associative tier** — `write(k,v)` / `read(q)`. Native to PAM (matrix outer products) and the Transformer KV cache (append + softmax attention). Drives `binding`, `persistence`, `niah`.
- **Stateful tier** — `write(...)` updates state; `state()` returns a matrix. Native to all three (PAM `S`, Transformer stacked-K, Mamba `ssm_states`). Drives `rank`.

## 6.1 Per-probe applicability

| Probe | PAM (matrix) | Transformer (KV cache) | Mamba (SSM) |
|---|---|---|---|
| Implementation correctness | Yes | n/a (exact by construction) | n/a |
| Binding capacity | Yes | Yes | No (no native associative read) |
| Persistence | Yes | Yes | No |
| Interference / NIAH | Yes | Yes | No |
| Effective rank | Yes | Yes | Yes |
| GSP / delta / multi-state | Yes | n/a (PAM-specific gates) | n/a |

Mamba's read is not a native associative lookup, so associative probes **skip gracefully** with an explanatory message rather than fabricating a number. Rank works on all three because it needs only `write` + `state`.

## 6.2 What each memory *is*

| | State size | Decay | Recall fidelity |
|---|---|---|---|
| PAM | fixed $O(d^2)$ | learned per-step $\gamma$ | high until capacity $\approx d$, then graceful interference |
| Transformer KV | grows $O(t)$ | none (lossless) | lossless until context cap / numerical limit |
| Mamba SSM | fixed $O(\text{intermediate}\times\text{state})$ | input-dependent | compresses; bounded by state rank |

The Transformer KV cache is a *lossless but unbounded* store; PAM and Mamba are *fixed-size* stores that must compress. The probes make this trade-off measurable rather than rhetorical.

## 6.3 Cross-architecture smoke results

Run on this machine via `./scripts/smoke_test_adapters.sh` (CPU, no GPU, no `mamba_ssm` / `causal_conv1d`; Mamba uses the HuggingFace `MambaModel` sequential path). Tiny configs: Transformer $d=64$; Mamba $d=32$, `state_size=16`, 1 layer.

| Probe | PAM ($d{=}64$) | Transformer ($d{=}64$) | Mamba ($d{=}32$) |
|---|---|---|---|
| Binding $N{=}64$ | 1.000 | **1.000** | SKIP |
| Binding $N{=}193$ | 0.992 | **1.000** | SKIP |
| NIAH rel @ $T{=}256$ | (decays) | 0.90, decay-independent | SKIP |
| Effective rank (final) | $\approx 49.5/64$ | $\approx 59/64$ | $\approx 9$–$11/16$ |

These match the architectural picture: the KV cache retrieves losslessly at every $N$ (it keeps all pairs, so capacity does not degrade — it pays in $O(t)$ growth instead), and its stacked-key state climbs toward full rank; PAM saturates near its $d$-dimensional capacity; Mamba's fixed SSM state fills its small ($\le 16$) singular subspace. The adapters are reference shims for *comparison*, not optimized kernels.

**Reproduce:**

```bash
python -m memory_probes --arch transformer --test binding --max-n 64
python -m memory_probes --arch transformer --test rank --steps 128
python -m memory_probes --arch transformer --test niah --distances 64,256
python -m memory_probes --arch mamba --test rank --steps 96 --arch-dim 32
./scripts/smoke_test_adapters.sh        # all of the above
```

## 6.4 Where PAM stands: PAM vs SSM vs Transformer on real text

The smoke test above uses synthetic random vectors. To compare the three memories on realistic, *clustered* input we stream the same WikiText-103 corpus through each and measure how much of its recurrent state each architecture actually uses — the **effective rank of the state** — against that state's hard ceiling and its memory cost. Here Mamba is the **pretrained** `state-spaces/mamba-130m-hf` (real learned $A,B,C$); PAM and the Transformer KV cache use untrained random projections of GPT-2 embeddings. The effective-rank *ceiling* is a structural property of the state geometry, so it is meaningful regardless of training.

Run: `python -m memory_probes --test arch-compare --text-tokens 2000 --arch-dim 64 --compare-layer 12` (Mamba on GPU; `mamba_ssm` not required).

| Architecture | State cost in context $t$ | Rank ceiling | State numbers / unit | Eff. rank on WikiText (final) | Utilization |
|---|---|---|---|---|---|
| Transformer KV | **grows** $O(t\,d)$ | $\min(t, d)=64$ | $t\cdot d = 128{,}000$ at $t{=}2000$ | 55.3 | 86% |
| **PAM** | fixed $O(d^2)$ | $d = 64$ | $d^2 = 4{,}096$ | 15.3 (untrained) | 24% |
| Mamba SSM (130M) | fixed $O(d_{\text{in}}\,d_{\text{state}})$ | $d_{\text{state}} = 16$ | $1536\times16 = 24{,}576$ | 13.4 | 84% |

Rank-vs-position (effective rank at token $t$, $d=64$):

| $t$ | Transformer | PAM | Mamba |
|---|---|---|---|
| 0 | 1.0 | 1.0 | 1.0 |
| 500 | 53.9 | 16.4 | 13.3 |
| 2000 | 55.3 | 15.3 | 13.4 |

**Reading the result.**

- **PAM vs SSM (the headline).** Mamba's SSM state **saturates against a hard 16-dimensional ceiling** — its effective rank flatlines at $\approx 13.4/16$ (84%) by $\sim 500$ tokens and never grows, because a diagonal SSM funnels the entire history through a length-$d_{\text{state}}$ vector ($d_{\text{state}}=16$ here). PAM's complex outer-product state has a **4× higher rank ceiling ($d=64$) at one-sixth the state size** (4,096 vs 24,576 numbers per unit). On *untrained* random projections PAM only fills 24% of that ceiling (rank 15.3, comparable in absolute terms to Mamba's 13.4), but its *capacity* is structurally far larger: the synthetic binding probe (Section 7.2) shows a clean $d=64$ PAM state holding $\approx 190$ retrievable associations and reaching effective rank $\approx 49.5$, whereas a 16-dimensional SSM state cannot exceed $\sim 16$ independent directions no matter how it is trained. In short: **same-or-smaller fixed cost, far higher associative ceiling.**
- **PAM vs Transformer (where it sits).** The Transformer KV cache climbs toward full rank (55/64) and never forgets, but its state **grows without bound** ($O(t\,d)$: 128,000 numbers at 2,000 tokens and rising). PAM is best understood as a **fixed-cost approximation of that lossless store**: it trades the Transformer's unbounded perfect recall for a constant $O(d^2)$ footprint, while keeping a far larger capacity than an SSM. It therefore sits between the two — SSM-like constant memory, Transformer-like associative richness up to its capacity limit.

**Caveat.** This is a mechanism-level comparison of *state utilization and cost*, not a behavioral recall benchmark. PAM and the Transformer adapter are untrained (parameter-free reads), and only Mamba carries trained weights; a full behavioral head-to-head needs trained checkpoints for all three (Phase B). The structural ceiling and cost differences, however, hold independently of training.

---

# 7. Empirical Results

All numbers below are from the PAM reference run at `seed=42`, `d=64` unless noted (`logs/memory_probes/memory_probes_all_20260623_160013.json`, `memory_probes_long_context_20260623_160102.json`, `memory_probes_language_filler_20260623_160106.json`, `memory_probes_rank_text_20260623_160120.json`).

## 7.1 Implementation correctness

Parallel training form matches $O(1)$ recurrent inference across every mode:

| Mode | $\max\lvert y_{\text{par}} - y_{\text{rec}}\rvert$ | Result |
|---|---|---|
| Baseline (head decay, additive) | $1.3\times10^{-8}$ | PASS |
| E1 per-channel decay | $2.1\times10^{-11}$ | PASS |
| E2 delta write | $6.3\times10^{-9}$ | PASS |
| E3 multi-state | $1.9\times10^{-8}$ | PASS |

The NumPy reference step matches the PyTorch layer step to $2.8\times10^{-17}$ (`baseline_step_diff`). Correctness is established before any capacity claim.

## 7.2 Binding capacity: matrix vs vector

| $N$ | Matrix PAM | Vector HRR | Theory $1/\sqrt{N}$ |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 |
| 64 | **1.000** | 0.091 | 0.125 |
| 193 | **0.992** | 0.012 | 0.072 |

Matrix PAM retains near-perfect top-1 retrieval out to $N \approx 190$ at $d = 64$, while the vector HRR baseline collapses to the $1/\sqrt{N}$ noise floor. This validates the central architectural motivation: a vector state suffers catastrophic interference exactly where a matrix state does not.

## 7.3 Interference

Relative retrieval of the first pair after writing $N$ pairs then $M$ filler tokens:

| Write rule | pairs=8, filler=0 | pairs=32, filler=256 |
|---|---|---|
| Additive | 0.951 | 0.500 |
| Delta (E2) | 0.459 | 0.066 |
| E3, $K=3$ | 1.789\* | 0.479 |

\*Values $>1.0$ arise because three phase-weighted states can sum constructively along the query direction (see Section 7.8). Delta writes help on repeated-key overwrite but hurt when all keys are distinct, because error-correction assumes a stale binding that is not present.

**Conjugate cross-talk.** For random phases, $\Pr[\mathrm{Re}\,W_{ts} < 0] = 0.503$ with $\mathrm{mean}(\mathrm{Re}\,W) \approx -3.6\times10^{-5}$ — destructive half the time, as predicted. Untrained PAM-layer projections preserve this geometry ($0.492$).

## 7.4 Effective rank

| Stream | Final $r_{\mathrm{eff}}$ | Max $r_{\mathrm{eff}}$ | $d$ |
|---|---|---|---|
| Synthetic, distinct random keys ($\gamma=0.995$) | 49.5 | 49.8 | 64 |
| Synthetic, repeated overwrite key | 1.000 | 1.000 | 64 |
| Real WikiText (untrained PAM layer, 5K tokens) | 21.3 | 24.1 | 64 |
| Uniform-random token ids (control) | 22.4 | — | 64 |

Distinct keys fill most of the available subspace; overwriting a single key never expands rank past $1$. On real text the rank rises quickly then plateaus around $20$–$24/64$, while the leading singular value keeps growing — i.e. the spectrum reshapes even when the effective-rank metric is flat. (Re-run with `--checkpoint` after pretraining for the publication figure.)

## 7.5 Persistence

| $\gamma$ | $T=64$ | $T=2048$ |
|---|---|---|
| 1.000 | 0.952 | 1.059 |
| 0.999 | 1.032 | 0.166 |

With $\gamma=1$ there is no geometric decay, and relative retrieval stays near (or even slightly above) $1$ as filler accumulates. With $\gamma=0.999$, the $\gamma^T$ envelope dominates by $T=2048$ ($0.999^{2048}\approx 0.13$), pulling retrieval down to $\sim 0.17$. The default $\gamma\approx 0.982$ decays far faster and motivates the protection mechanism.

## 7.6 Long context (needle-in-haystack)

Single needle, then $T$ filler writes, relative retrieval:

| Mode | $T=2048$ | $T=65{,}536$ |
|---|---|---|
| Filler, $\gamma=0.982$ (default) | — | 0.046 |
| Filler, $\gamma=0.999$ | 0.321 | 0.306 |
| GSP $p_{\text{fill}}=0.99$ | 0.692 | 0.004 |
| GSP $p_{\text{fill}}=1.0$ (freeze) | 1.000 | **1.000** |

Position dependence at $T=65{,}536$ (needle written at relative position $r$):

| Config | $r=0.0$ (start) | $r=1.0$ (end) |
|---|---|---|
| No protection ($\gamma=0.995$) | 0.353 | 0.788 |
| GSP $p=0.99$ | 0.012 | 0.993 |

Two findings. (1) Under bare decay there is a strong **recency bias** — a needle near the end survives far better than one near the start. (2) With full protection ($p=1$) the needle is preserved perfectly to 65K+ tokens (the summary reports $\ge 90\%$ relative retrieval out to the maximum swept distance), establishing the **mechanism ceiling**. Partial protection ($p=0.99$) still loses early needles, so the open question for trained models is whether the gate *learns* to raise $p$ on filler tokens.

## 7.7 Language filler

Single run (5K WikiText tokens, projection `seed=42`):

| Quantity | Value |
|---|---|
| Baseline relative | 1.000 |
| Language filler relative | 12.61 |
| Random filler relative | 0.151 |
| Language / random ratio | 83.7× |
| Mean key correlation $\mathbb{E}[\lvert k_i\cdot k_j\rvert]$ | 0.312 |

Across a 50-projection-seed sweep (5K tokens), language relative retrieval is **stable** (mean $\approx 9.3$, std $\approx 2.8$, range $[3.3, 15.3]$, CV $< 35\%$) and **beats random on every seed** (ratio mean $\approx 94\times$, min $\approx 14\times$). The huge ratio variance comes from the tiny random denominator, not from instability in the language score.

**Interpretation.** This does *not* show that English destroys the needle more than random noise; it shows the opposite — clustered embeddings ($\mathbb{E}[|k_i\cdot k_j|]=0.31 \gg 0$) align constructively with the query at the projection layer, *helping* retrieval. A definitive behavioral test requires a trained language model with next-token prediction (Phase B, Section 8).

## 7.8 Interpreting relative scores $> 1.0$

Retrieval is normalized against a fresh single-write baseline. A value $>1$ means later writes **constructively aligned** with the query direction (Sections 7.3, 7.5, 7.7). This is not a bug; it is a reminder that random filler is not phase-neutral, and that the relative metric measures alignment, not just survival.

---

# 8. Discussion

**What the probes prove vs. what they cannot.** The synthetic battery answers mechanism questions; it does not answer behavioral ones.

| Question | Memory probes (math) | Trained-LM probes (Phase B) |
|---|---|---|
| Is the recurrent update implemented correctly? | Yes — selftest, layer-bridge | — |
| Does matrix memory beat vector HRR capacity? | Yes — capacity | — |
| Does decay protect associations over distance? | Yes — persistence, NIAH | needs learned gates |
| Can protection freeze state for needles? | Yes — GSP ceiling | does training use it? |
| Does the model recall `glorp → banana` in real text? | No | Phase B |

**Central hypothesis.** The math shows recency bias and rapid forgetting at the default decay unless the protect gate drives $p \to 1$ on irrelevant tokens. If training never learns *when not to write*, everything writes, and interference plus decay erase the needle. The single most informative Phase B diagnostic is therefore logging the learned $p$ on needle versus filler tokens. The gap between the Phase A mechanism ceiling and Phase B behavioral recall measures **how much of the architectural capacity training actually realized**.

**Phase B (deferred).** After a from-scratch pretrain: (1) run invented-word / passkey / interference probes on real text against a trained checkpoint, (2) overlay behavioral recall-vs-distance on the NIAH math curves, and (3) log the learned protect gate on needle vs filler tokens.

---

# 9. Conclusion

Memory is not intelligence. Intelligence is constrained by memory but not defined by it. By specifying memory probes precisely — implementation correctness, binding capacity, persistence, interference, effective rank, long-context retrieval, and realistic language interference — and by giving a deterministic reference implementation, we make the memory subsystem a first-class, comparable quantity. The framework was developed for the PAM architecture but is architecture-agnostic, and is offered as a foundation for principled comparison across recurrent memories, attention, state-space models, and future neural memory systems.

The reference implementation is available at [`https://github.com/gowrav-vishwakarma/qllm2`](https://github.com/gowrav-vishwakarma/qllm2) under `memory_probes/`.

---

# Appendix A. Reproducibility

| Probe class | Deterministic? | Notes |
|---|---|---|
| NumPy math (binding, persistence, interference, NIAH, long-context) | Yes (`seed=42`) | Bit-exact across runs |
| Synthetic rank (`--test rank`) | Yes | Pure NumPy recurrence |
| Layer-bridge / selftest | Yes | Fixed Torch seeds |
| Conjugate layer check | Approx. | Untrained layer input uses unseeded `torch.randn`; statistics stable |
| Rank on real text | Approx. | Untrained weights re-init each run; use `--checkpoint` for trained weights |
| Language filler | Yes | GPT-2 embeddings + fixed projection seed; WikiText stream cached |

**Exact commands used for the reported numbers:**

```bash
python -m memory_probes --all --seed 42
python -m memory_probes --test long-context --max-distance 65536 --seed 42
python -m memory_probes --test language-filler --filler-tokens 5000 --seed 42
python -m memory_probes --test rank-text --text-tokens 5000 --sample-every 100 --seed 42
```

Result logs: `logs/memory_probes/*.json`. Compare against legacy runs with `scripts/compare_memory_probes.py`.
