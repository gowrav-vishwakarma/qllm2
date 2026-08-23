# V14 — Spatial-Neuron LM (design notes, 2026-08-22)

> **Status: notes only. No code written this session** (per user: let the v13 run
> train; implement in a later session). This file = research + math + a concrete
> build plan. Grounded in: the user's spatial-neuron idea, our `pands` project
> (read in full), and verified prior art (RSGN, MoP, GNG, cortical geometry).
> Nothing here is a claim — it's a design to test.

## 0. The one-line idea

Each neuron has a **learned position** in a space. **Distance between positions is
a second weight**: far-apart neurons stop communicating (soft cutoff). Neurons
**move during learning** — positions are parameters, so gradient descent reshapes
the *connectivity map itself*. The model is **sparse by architecture**: each token
activates a local neighborhood, not the whole layer.

Why this is the right next idea (and not just attention with a mask):
- Attention's sparsity is *input-dependent* but the graph is *dense* (all pairs
  computed, then softmaxed). Here the graph is **sparsified by geometry** — the
  compute is physically local, so cost scales with neighborhood size, not n².
- The *structure* (who talks to whom) is **learned and dynamic**, not fixed. This
  is the part transformers/SSMs can't do: they have a fixed topology.

## 1. Why pands didn't take off (the lessons to NOT repeat)

pands v2 (`/home/gowrav/Development/pands/v2`) is our own prior attempt at a
cousin of this idea: a diagonal complex-RNN "speaker" + a "Navigator/Course" that
plans a destination in a big complex hyperspace. Read its `EXPERIMENTS_V2.md`.
Three failures, all directly relevant:

1. **Decorative auxiliary path.** The Course/leg was fully wired but the model
   learned to *ignore* it (`none ≈ honest` on TinyStories). Root cause: with a
   capable local mixer + next-token CE, **there's no pressure to route info
   through the extra path**, so it collapses to decoration. *Lesson: any
   new pathway must be load-bearing by construction — either the main path is
   bottlenecked, or the new path carries information the main path cannot get.*
2. **LRU retrieval wall.** A pure diagonal linear RNN learns local n-gram
   stats fast (beat the transformer on TinyStories: 47 vs 71 PPL) but **cannot
   do content-based retrieval**, so it loses on retrieval-heavy text (WikiText:
   145 vs 113). *Lesson: fast local learning + a real content-addressable memory
   are BOTH required; neither alone wins.*
3. **Speed gate missed.** pands was 2–4× *slower* than the matched transformer
   (complex chunked scan on the 2·d_space view, extra passes). *Lesson: the new
   architecture must train at transformer speed — no 4× penalty. Our v13 speed
   work (fused-delta, O(1) gate) is the template for how to get there.*

**The v14 design below is explicitly built to avoid all three.**

## 2. Verified prior art (what exists, and the gap)

| Work | What it does | Why it's not enough / what's new here |
|------|-------------|----------------------------------------|
| **RSGN** (arXiv 2601.18064, Jan 2026) | Nodes in learned **hyperbolic** space; connection strength = distance decay; slow **Hebbian** position/affinity drift; input-dependent activation sparsity; local inhibition | Closest prior art. But: (a) validated only at **toy scale (41K params)** and **loses to a transformer** on their classification task; (b) positions move by a *separate slow Hebbian rule* (two-timescale), NOT by the main gradient — fragile and untested at LM scale; (c) the "space" is a fixed global graph, not a per-token local computation. **The gap: no one has done this at 100M-param LM scale, with positions as first-class gradient parameters, and shown it learns as fast + better than transformer/Mamba.** |
| **Mixture of Perceptrons (MoP)** (DeepMind, 2024) | Each neuron = a high-dim hyperplane; routing by which side of the hyperplane the input is on; sparse, scales well | Routing is by *hyperplane side*, not *learned metric distance between neuron positions*. MoP is a strong baseline to beat; v14's distance-gating is a different, geometry-native mechanism. |
| **GNG** (Growing Neural Networks, 2018) | Edges grow/prune by input statistics; neurons added dynamically | Dynamic topology but **no learned positions** and no distance-gated communication; it's about *adding units*, not *moving neurons in a metric space*. |
| **Cortical geometry / seRNN** (2025) | Recurrent dynamics shaped by learned spatial structure | Directional, but small-scale; confirms the "geometry-as-learnable-structure" line is active and not settled. |

**The open, defensible claim for v14:** a spatially-gated, position-learned
neuron field that (a) is sparse by architecture, (b) has **content-addressable
retrieval** (fixes the pands LRU wall), (c) **trains at transformer speed** on a
single 4090, and (d) **beats matched transformer + Mamba on recall/reasoning** —
at 100M scale, where RSGN has not been tested.

## 3. The math (concrete, implementable)

Let a layer have $N$ neurons, each with a **position** $\mathbf{p}_i \in \mathbb{R}^d$
(position dim $d \ll$ hidden dim, e.g. $d=32$) and a **weight vector**
$\mathbf{w}_i \in \mathbb{R}^{h}$. The position is the *second weight*.

### 3.1 Distance-gated local communication (the core)

For a token embedding $\mathbf{x}_t$, compute a **query position**
$\mathbf{q}_t = f_q(\mathbf{x}_t) \in \mathbb{R}^d$ (a small linear). Each neuron's
**coupling** to the token is a soft distance gate:

$$g_{t,i} = \sigma\!\left(\beta\,(r_t - \|\mathbf{q}_t - \mathbf{p}_i\|_2)\right)$$

- $r_t$ = learned **communication radius** (a scalar param, or a small MLP of
  $\mathbf{x}_t$ so different tokens reach different distances).
- $\beta$ = gate sharpness (init moderate, e.g. 8; larger = harder cutoff).
- $\|\cdot\|$ = Euclidean. (Hyperbolic is an option later, but start Euclidean:
  cheap, differentiable, and RSGN's hyperbolicity is unproven at scale.)

The layer output is a **gated neighborhood sum** — NOT a dense $N$×$N$:

$$\mathbf{y}_t = \sum_{i=1}^{N} g_{t,i}\;\phi(\mathbf{w}_i^\top \mathbf{x}_t)\,\mathbf{v}_i$$

where $\mathbf{v}_i$ is a value vector and $\phi$ = swish. Crucially this is a
**single matmul** $[\mathbf{x}_t]$ against the stacked weight matrix, scaled row-wise
by $g_{t,i}$ — i.e. **O(N·h)**, the same cost class as one linear layer, *not* O(N²).
Sparsity comes free: with a small radius, most $g_{t,i} \approx 0$, so the effective
support is a local neighborhood (the architecture is "non-dense" in compute and in
gradient flow, not just in a mask).

**This directly fixes pands failure #1 (decorative path):** the spatial gate is on
the *main* computation path, not an add-on leg. There is no bypass — the only way
to produce $\mathbf{y}_t$ is through the distance-gated sum.

### 3.2 Positions are learned by gradient (fixes RSGN's fragility)

$\{\mathbf{p}_i\}$ are **ordinary parameters** in the optimizer. The gradient of the
loss w.r.t. $\mathbf{p}_i$ flows through $g_{t,i}$:

$$\frac{\partial g_{t,i}}{\partial \mathbf{p}_i} = -\beta\, g_{t,i}(1-g_{t,i})\,
\frac{\mathbf{q}_t-\mathbf{p}_i}{\|\mathbf{q}_t-\mathbf{p}_i\|_2}$$

So the *main* loss moves neurons toward the query positions of the tokens that
need them, and away from the rest. **No separate Hebbian rule, no two-timescale,
no reward signal** — the same SGD that trains the weights also trains the geometry.
This is the cleanest possible version of "neurons move to learn," and it's the
biggest differentiator from RSGN.

**Stability:** to prevent all positions collapsing to one spot (a real risk with
distance gates), add a tiny **repulsion / uniformity regularizer** (a spherical
code / von-Mises-Fischer style spread term, weight $\lambda_{\text{spread}}\sim1e{-4}$):
push positions apart so the $d$-dim space is actually used. Track the effective
rank / coverage of $\{\mathbf{p}_i\}$ as a probe (we already have an eff-rank probe
for v13 states — reuse it).

### 3.3 Content-addressable memory (fixes the pands LRU retrieval wall)

A spatial gate alone is a *local* mixer — it won't do recall. Interleave the
spatial layer with a **content-addressable fast-weight memory** (the VSA /
qllm2-style outer-product memory that pands identified as missing):

$$S_t = \gamma_t S_{t-1} + \phi(\mathbf{k}_t)\otimes \mathbf{v}_t, \qquad
\mathbf{o}_t = \mathbf{q}_t^\top S_t$$

- $S_t \in \mathbb{R}^{d_k\times d_v}$ is **O(1) in sequence length** (no KV cache —
  the user's standing requirement).
- $\gamma_t$ is the **selective decay** (Mamba-style, input-dependent) so it's a
  *selective* SSM, not a fixed LRU.
- Bind keys with the spatial phase (reuse the LUT-RoPE from pands) for VSA-style
  binding.

**Block layout per layer:** `SpatialGate → (residual) → SelectiveMem → (residual) →
FFN`. The spatial layer does fast local composition; the memory does long-range
content retrieval. This is the combination pands' post-mortem said is required.

### 3.4 Why this is "not transformer, not mamba, not jamba"

- **Transformer:** dense all-pairs attention. v14 = local distance-gated + O(1)
  memory; no n² term, no KV cache.
- **Mamba/Samba:** selective SSM with a *fixed* recurrence topology. v14 adds a
  *learned, moving* spatial topology as a second, gradient-trained axis.
- **Jamba:** just an *interleaving* of Mamba + attention (fixed parts). v14's
  spatial field is a genuinely new mixer, not attention.
- **MoP:** hyperplane-side routing; v14 = metric-distance routing between
  *learned, movable* positions.

## 4. Speed plan (must pass the ≤1.5× transformer gate)

From v13 (fused-delta 2.3K→21K tok/s) and pands (2–4× too slow):
1. **Spatial gate = one matmul + a row-wise soft gate.** Precompute $\mathbf{P}$
   (N×d) as a buffer; per token compute $\|\mathbf{q}_t-\mathbf{P}\|$ via a
   GEMM-free trick: $\|\mathbf{q}-\mathbf{p}\|^2 = \|\mathbf{q}\|^2 - 2\mathbf{q}\mathbf{P}^\top
   + \|\mathbf{p}\|^2$, so it's a **single (T×d)@(d×N) GEMM** + a row norm + a
   precomputed column norm. All of it is a GEMM + elementwise → fast, chunkable.
2. **Selective memory = the existing fused-delta machinery.** v13's
   `_forward_multistate_delta_fused` already does chunk-parallel O(1)-state
   selective writes at 21K tok/s. Reuse it verbatim for the memory branch.
3. **No two-pass, no oracle pass** (pands' 4× came from an oracle second pass).
   Positions are trained in the single main pass.
4. Target: **≥ matched-transformer tok/s** at 100M on the 4090, else stop and
   profile (the pands gate, enforced).

## 5. Build plan (next session, smallest-first)

1. **Selftest (CPU, exactness):** spatial gate forward == naive loop;
   $\partial g/\partial p$ matches autograd; memory forward == recurrent;
   O(1) decode state; gate is load-bearing (`none` control must hurt PPL — the
   pands check, made a *hard* test).
2. **1M smoke (TinyStories, matched transformer):** confirm the gate is
   load-bearing (none ≫ honest) and speed ≤1.5×.
3. **Recall/needle bench** (the pands-missed test): synthetic KV-recall +
   needle, vs matched transformer + Mamba. v14 must *beat* them here (the LRU
   wall is the whole point of the memory branch).
4. **10M WikiText (retrieval-heavy):** v14 honest PPL vs transformer. Must beat
   pands' 145 and the transformer's 113 — this is the "not decorative" proof.
5. **100M real-data run** (same protocol as v13/v11: pretrain_mix, 500M tok,
   4090, tmux). Only after 1–4 pass.

**Kill criteria (inherited from pands, enforced):** stop if (a) the spatial gate
is decorative (none ≈ honest) after a bottleneck test, or (b) speed >1.5× the
matched transformer at 1M. Do NOT scale to 100M until both pass.

## 6. Open questions / to decide in the build session

- Euclidean vs hyperbolic positions (start Euclidean; revisit if coverage probe
  shows the space is under-used).
- Per-token radius $r_t$ (scalar param vs small MLP) — ablate.
- Spatial layer *before or after* the memory branch; 1 spatial + 1 mem per block
  vs 2+1. Ablate.
- Position dim $d$ (32 vs 64) and gate sharpness $\beta$ schedule.
- Whether the spatial gate should also *write-gate* the memory (VSA-style) or be
  a pure mixer. pands' Course-as-write-gate never became load-bearing — test
  carefully with the recall bench as the arbiter.

## 7. SLOT: Immortal Talks / Kali Hanu Vani / Vedanta

> **User pointed to "Immortal Talks — Kali Hanu Vani" (and "Indian Vedas") as a
> learning source. NOT YET READ — no verifiable source found via web search on
> 2026-08-22 (providers bot-blocked; direct site fetches failed; no clear
> book/talk by that exact name surfaced).** Fill this section once the user gives
> a link/PDF/exact title+author+language. Do NOT fabricate content and attribute
> it. The likely intent (from the phrasing): philosophical framing for *why* a
> self-organizing, locally-communicating, distance-gated neuron field is a
> principled model of cognition — useful as *motivation* for the NOTES intro,
> not as math. Park until sourced.

## 8. CRITICAL FINDING this session (v13 run) — see EXPERIMENTS_V13.md

While this file was being written, the fresh v13 (100M, lr 3e-4, 500M-token) run
revealed a **genuine, widening learning gap vs the v11-best HF model** (same
100M params, same lr/wd/betas, both GPT-2 vocab): at 59M tok v13 CE ≈ 8.43 vs
v11-best ≈ 5.2. The gap is 0.04 NLL at 1.9M tok but 3.2 NLL at 44M — **v13 learns
fast initially then stalls**. Suspects (unresolved, need matched ablations):
(a) the **gate-surprisal aux** (new in v13, λ=0.1) fighting the trunk;
(b) **delta write_mode** (v13) vs **additive** (v11); (c) the **15% synthetic
recall/reason** data. This is the thing to investigate *before* trusting any
v13 quality number, and it's a separate track from v14.
