# V13 Explained From Zero — the maths of the code, line by line

Written 2026-08-23 for someone who knows what a "feature" is but is new to
matrices, PyTorch shapes, and vectors. If you do **not** know what a
feature, a head, or a phase is yet, start with
[`BEGINNER_STORY.md`](BEGINNER_STORY.md) — it builds every word from zero
before this one does. For the op-by-op shape reference, see
[`MATRIX_COOKBOOK.md`](MATRIX_COOKBOOK.md).

Every line number below refers to `v13/model.py` unless noted. The production
preset is `v13_e3_k3_selective` (model.py:1832), the one the 500M run trains:
`dim=384, n_heads=6, head_dim=64, n_layers=16, n_states=3 (K), vocab=50261,
write_mode='delta', delta_chunk=128, vault_state=True, gate_surprisal_lambda=0.1`.

---

## 0. The one-paragraph version

V13 is a language model where each of the 16 layers has a **fixed-size memory
matrix** instead of a growing list of past tokens. Per token the layer:

1. **projects** the token's feature vector into a *query* `q`, *key* `k`,
   *value* `v` (three complex 64-dim vectors per attention head),
2. **reads** its memory: `y = S · q` (an inner product — retrieval),
3. **writes** the new fact: `S ← γ·S + (βw·v − βe·S·k) ⊗ k*`
   (a scaled old memory plus an **outer product** — storing the *error*
   between the new value and what was already stored),
4. **rotates** the read by a learned phase and adds it to the residual stream.

That memory matrix is `d×d` = 64×64 complex per (head, batch item, state) —
no matter whether you've seen 10 tokens or 10 million. That's the whole
"O(1) inference, no KV cache" story. Training computes the same thing a
different way (big matrix multiplies over chunks of tokens) and a self-test
proves the two forms agree to ~1e-7.

---

## 1. Prerequisites: tensors, shapes, and the letters

A **tensor** is just an array of numbers with a *shape*. Shape = list of
dimension sizes. In this codebase you'll meet these letters over and over:

| Letter | Meaning                        | Production value |
|--------|--------------------------------|------------------|
| `B`    | batch (independent sequences trained together) | 8  |
| `T`    | sequence length (tokens)       | 2048             |
| `dim`  | model width (features per token) | 384            |
| `H`    | number of attention heads      | 6                |
| `d`    | head_dim = features per head   | 64               |
| `K`    | number of memory states        | 3                |
| `C`    | delta chunk (tokens per parallel solve) | 128  |
| `V`    | vocabulary size                | 50261            |
| `2`    | split-real/imag pair (see §2)  | 2                |

A token at position `t` in sequence `b` is a vector of `dim=384` features.
After the QKV projection and head split, each of the `H=6` heads sees a
`d=64`-dim slice. So one PAM layer's inputs have shape:

```
queries, keys, values : [B, H, T, d, 2]
                        8  6 2048 64  2
```

Read that as: "for 8 sequences, 6 heads, 2048 time steps, 64 channels,
real+imaginary parts." **The `2` is the last axis.** (Why complex? §2.)

The memory state of the whole layer:

```
memory_state : [K, B, H, d, d, 2]
               3  8  6  64 64  2
```

**Your intuition is exactly right:** the `d×d` part is a *feature × feature*
matrix — "value-feature i × key-feature j". The `T` dimension has been
*collapsed*: everything the sequence contained got summed into that table.
That is the single most important idea in the file. We'll unpack it in §5.

---

## 2. Complex numbers: split-real layout, magnitude and phase

A complex number is a pair: `z = a + i·b`, with `i² = −1`. Think of it as a
2D arrow: **magnitude** `|z| = √(a²+b²)` (how loud) and **phase**
`arg(z)` (which way it points). Multiplying two complex numbers multiplies
magnitudes and *adds* phases — that's what makes them useful here.

This codebase never uses `torch.complex64`. Instead a complex vector of `d`
numbers is stored as **split-real** shape `[..., d, 2]`: the last axis holds
`[real, imag]`. You can see the arithmetic in `v13/complex_ops.py`:

```python
cmul : (a_r + i a_i)(b_r + i b_i) = (a_r b_r − a_i b_i) + i(a_r b_i + a_i b_r)
cconj: conj(a + i b) = a − i b            # flip the imag sign
cabs : |z| = √(a² + b² + 1e-8)
```

Why complex at all? Two roles:

1. **Phase = binding label.** A value bound to a key can be *addressed* by
   matching phases: the same phase reinforces, different phases interfere
   (like orthogonal vectors). This lets many facts share one matrix without
   all overwriting each other.
2. **Conjugate retrieval.** The "do these two match?" test in complex space
   is the *conjugate* inner product `⟨a,b⟩ = a·b* — we meet this constantly.

---

## 3. The two products — inner vs outer (your key question)

This is the semantic heart. Take two vectors `u, v ∈ C^d`.

### Inner product (dot) — *reading*, gives a **score**

```
u·v* = Σ_j u[j] · conj(v[j])        → one number
```

It **collapses** two vectors into a single complex number: "how much does `u`
look like `v`?" If `u = v`, then `u·u* = |u|²` — a real, positive, maximum
score. If they're orthogonal, 0. **Every *retrieval* in the model is an inner
product**: query against stored keys, state against a key, etc.

### Outer product — *writing*, gives a **table**

```
u ⊗ v*  is the d×d matrix with  (u ⊗ v*)[i, j] = u[i] · conj(v[j])
```

It **expands** two vectors into a full `d×d` matrix: "bind every channel of `u`
to every channel of `v`". **Every *storage* in the model is an outer product.**

### Why the two are the same memory

The key fact (and the whole point of matrix memory) is that writing with the
outer product and reading with the inner product *compose back into the
original value*:

```
write:  S ← S + v ⊗ k*          # store the binding "k → v"
read:   y = S · q
        y[i] = Σ_j S[i,j] q[j]
              = Σ_j (v[i] k*[j]) q[j]
              = v[i] · (k* · q)
        y = v · (k*·q)
```

The matrix multiplies the query `q` by exactly "the similarity of `q` to the
stored key `k`", and hands back the value `v` scaled by that similarity.
Ask with `q = k`: you get `v · |k|²` (all of it). Ask with something
unrelated: ~0. **The d×d matrix is a lookup table whose entries are
feature×feature associations; the inner product is how you query it; the outer
product is how you fill it.**

That's why `memory_state` is `[K,B,H,d,d,2]`: per (state, batch, head) we keep
one feature×feature table.

---

## 4. The three-line memory step (the "notebook" view)

`_recur_step_additive` (model.py:1333) is the simplest complete memory step
and the one to understand first. Inference runs this, one token at a time:

```python
# 1. FORGET (scale the whole table):      S ← γ · S
# 2. WRITE (outer product, key conjugated):
key_conj = (k_r, −k_i)                      # conj(k)
outer    = v ⊗ k*                           # [B,H,d,d,2]
memory_state = memory_state * decay_factor + outer
# 3. READ (matrix–vector, no conjugate on q):
y = S · q                                    # [B,H,d,2]
```

`γ` (gamma, 0 ≤ γ < 1) is the **decay**: multiply the whole notebook by it.
`γ = 0.9` means "forget 10% of everything per token". `γ = 1` means "never
forget". γ is *learned per token and per head* (model.py:435-446):
`γ = exp(−softplus(dt_proj(x) + bias))` — softplus keeps the argument
positive, exp squeezes the result into (0, 1).

Notice the conjugation asymmetry: **key conjugated on write, query NOT
conjugated on read.** That's deliberate: it makes the stored self-similarity
`k·k* = |k|²` a real positive number (§3). If we wrote `v ⊗ k` (no conj) and
read with `S·q`, asking with `q=k` would return `v·(k·k)` — a *complex* number
whose phase is twice the key's phase: the memory would be out of phase with
itself. With the conjugate, the memory is self-consistent.

### The production delta step (`_recur_step_delta`, model.py:1373)
The winner uses `write_mode='delta'` instead of plain additive. The step:

```python
memory_state = memory_state * decay_factor            # forget, same
predicted    = memory_state · k                       # READ at key k  (S·k)
update       = write_beta * v − erase_beta * predicted   # the DELTA
key_conj     = conj(k)
memory_state = memory_state + update ⊗ key_conj       # WRITE the delta
output       = memory_state · q                       # READ at query q
```

**Why "delta"?** Before writing value `v` under key `k`, the model asks its
memory "what do you already have for `k`?" (`predicted = S·k`). If the answer
is close to `v`, the update is ~0 — don't rewrite what's already there. If
the memory has the *wrong* value (a fact that changed: "the capital is X" →
now Y), the update is the *correction*. This is the **delta rule** from
classical associative memory (Widrow–Hoff, 1960s): *update ∝ (target −
prediction)*.

A concrete property that makes it click. With unit-norm keys (`|k| = 1`, see
§6) and equal gates `write_beta = erase_beta = β`, after the write, reading at
the same key gives:

```
new read at k = (1−β) · (old read at k) + β · v
```

i.e. the memory's answer for key `k` is an **exponential moving average**
toward `v`. β=1 → perfect overwrite. And reading at an *orthogonal* key `k'`
is completely unaffected (the new write `update ⊗ k*` is invisible to `k'`,
because `k*·k' = 0`). The delta rule is **selective storage**: it touches
only the subspace of the key it's writing, and only by the amount it's wrong.

## 5. GSP protect gate, and the vault

### The protect gate (GSP)

`_gamma_and_vprime` (model.py:422) decides, per token, how much to trust new
writes. `p = sigmoid(protect_gate(x))` ∈ (0,1) — the "protect probability":

```python
decay_gamma    = base_decay * (1 − p) + p      # p→1: decay frozen at 1
protected_values = values * (1 − p)            # p→1: nothing new written
```

- `p → 1`: the notebook **freezes** (γ→1) *and* the write is **muted**.
  "This token is filler — don't disturb long-term memory."
- `p → 0`: normal decay + full write. "Content — store it."

The gate is a plain linear layer on the token's features (real+imag
concatenated), initialized with bias −3 so `p ≈ 0.047` at start — training
begins in the "always write" regime and learns *selectivity*. The
gate-surprisal auxiliary loss (v7/train.py:354, λ=0.1 in the preset) nudges
it: filler (low-surprisal) tokens get a high-protect *target*, content tokens
a low one — so the gate learns "freeze on filler, write on content" rather
than a flat ~0.4 everywhere.

### The vault

One of the K=3 states (index 0) is **pinned to γ ≡ 1** (model.py:448-453):
it never decays — a persistent "vault". The other two states decay normally —
fast "scratchpad" memory. Writes are **broadcast to all K states**
(model.py:1020-1023), so every fact lands in the vault; but the *erase* term
`βe·S_k·k` is computed per state (model.py:1003-1012), so in the big vault
state `S_0·k` is large and the net write is self-limiting (only the residual
goes in), while the fresh decayed states take the full write. Result: the
vault accumulates facts without runaway interference, and the scratchpad
states track recent context.

---

## 6. Why the keys get unit-normalized (and the 500M NaN)

`_project` (model.py:399-400) does, for delta mode:

```python
keys = cnormalize_vec(keys)     # per (b,h,t): k ← k / ||k||  across d
```

**Why.** Look at what the delta write does to the component of `S` along the
key direction `k`. One step:

```
S ← γ·S + (βw·v − βe·S·k) ⊗ k*
```

The part of `S` pointing along `k` gets multiplied by the **eigenvalue**

```
eig = γ − βe · ||k||²
```

For the vault, γ=1, so `eig = 1 − βe·||k||²`. Stability needs `|eig| < 1`.

- **Unnormalized keys:** a random 64-dim vector has `||k||² ≈ d = 64`. Then
  `eig ≈ 1 − 64·βe` — with even βe = 0.05 that's −2.2: every step **flips
  sign and grows** the state by 2×+. Unbounded growth → **NaN**. This is
  exactly how the 500M run died at ~57M tokens.
- **Unit keys:** `||k||² = 1` → `eig = 1 − βe`. The sigmoid-gated βe lives in
  (0,1), so eig ∈ (0,1): a strict contraction. The config additionally caps
  βe ≤ 0.95 (`delta_erase_beta_cap`, model.py:105-115) as a safety margin,
  because in the chunked parallel path the erase couplings *compound across
  the tokens inside a chunk* (§9), so the single-step eigenvalue is not the
  whole story.

Note (model.py:103): per-*element* normalization (`qk_norm`) is **not**
enough — it makes `||k||² → d` instead of 1. The fix is per-*vector*
normalization across the head dim. And it must keep the **exact** autograd
Jacobian `(I − x̂x̂ᵀ)/||x||` — a `g/mag` shortcut was tried and measured 19%
wrong key gradients (SCRATCHPAD archive).

---

## 7. K states and phase routing (E3)

Instead of one notebook, the layer keeps `K=3` notebooks (states) with
*different decays* (per-state offset on `dt_bias`, model.py:232-234). One
state might hold "recent sentence", another "recent paragraph", another the
vault. **Retrieval superposes them** (model.py:1106-1116):

```
output = Σ_k  (routing_w_k · e^{i φ_k}) · y_k
```

where `y_k` is state k's read and `(w_k, φ_k)` come from
`_phase_and_alpha` (model.py:313): a small linear layer turns the token's
features into a per-state **phase** φ_k (radians) and a **weight** w_k.
Multiplying by `e^{iφ}` *rotates* each state's contribution before summing —
so the model can learn to "tune in" to one state's frequency and cancel
another's. With zero-init (the production `phase_init='zero'`) all φ=0 and
weights=1 at start: the K states begin as a plain sum and specialize during
training.

`write_phase_address` (model.py:403-410) is a sibling trick: on *write*,
rotate `v` by a phase ψ(k) computed from the key; on *read*, rotate `q` by
ψ(q). Matching bindings reinforce through conjugation — a per-key "address"
for where the value is stored.

---

## 8. RoPE in one line

`_project` (model.py:373-384) multiplies `q` and `k` by a precomputed
`e^{i·m·θ_j}` where m = position, θ_j = frequency band j:

```python
rope_positions = self.rope_cache[step_offset:position_end]   # [T, d, 2]
queries = cmul(queries, rope_positions)
keys    = cmul(keys,    rope_positions)
```

Complex multiplication *adds phases*, so the score `q_m · k_s*` now carries
phase `(m−s)·θ_j` — the score depends on the **relative** position, not
absolute. In a complex model RoPE is literally just a rotation; no extra
math. `step_offset` is why `generate()` can keep feeding one token at a time
with the correct absolute position (model.py:1711).

---

## 9. The training form: from a token loop to big matrix multiplies

The recurrent step (§4) is perfect for inference — fixed memory, one token at
a time. But it's a **loop over T=2048 tokens**, and GPUs hate loops. Training
computes the *same* math over **chunks of C=128 tokens** using batched
matrix multiplies. Three identities make this possible.

### 9.1 Unroll the recurrence → the "dual form"

Unroll the additive notebook over a chunk of tokens 0..t (ignoring the
carried-in state for a moment). The read at time t is:

```
y_t = Σ_{s ≤ t}  (decay from s to t) · (q_t · k_s*) · v_s
```

Define the **decay matrix** `D[t,s] = γ_t γ_{t−1} … γ_{s+1}` (product of
decays between s and t; computed by the Triton kernel `fused_decay_matrix`,
v13/triton_kernels.py) and the score matrix `W[t,s] = q_t · k_s*`. Then the
whole chunk's outputs are **one matrix expression**:

```
y = (W ⊙ D) @ V          # [C,C] ⊙ [C,C] then @ [C,d]  →  [C,d]
```

`⊙` = elementwise multiply. This is the **dual form** (model.py:478,
`_dual_form_block`; the fused version is `_fused_chunk_step`, model.py:812).
"Dual" = "same math, laid out for GPU matmuls instead of a token loop".
The state update has its own dual form: summing the outer products over the
chunk, `S += (V ⊙ D_last_row) @ K*` (model.py:494-498) — one matmul instead
of C outer-product additions.

### 9.2 Chunks + carry

A 2048-token sequence is 16 chunks of 128. Each chunk does:

1. **in-chunk read**: `y_intra = (W ⊙ D) @ V` (tokens talking to tokens
   inside the chunk),
2. **carry read**: the *previous* chunks' state `S` was already built, so
   `y_carry = (q · α) @ S` where `α[t] = γ_t…γ_1` (cumulative decay into
   this chunk, model.py:519-520) — the old notebook's contribution, decayed,
3. `y = y_intra + y_carry`, and the chunk's writes are folded into `S` with
   the chunk's total decay (model.py:538-539).

So: **intra-chunk = attention-like matmul; inter-chunk = the fixed state.**
That's the whole chunking story.

### 9.3 The delta rule breaks the naive unroll → the triangular solve (UT transform)

In delta mode the write at time t is `u_t = βw·v_t − βe·(S_{t−1}·k_t)`, and
`S_{t−1}` *depends on the earlier writes u_s* (s < t). Unrolling, each
`u_t` ends up depending on all earlier `u_s` through the key-similarity
`k_t · k_s*`:

```
u_t + Σ_{s < t}  βe_t · D[t,s] · (k_t · k_s*) · u_s  =  βw_t·v_t − βe_t·(k_t·S_prev)
```

That's a **linear system** `(I + M)·u = w` where `M[t,s] = βe_t·D[t,s]·(k_t·k_s*)`
for `s < t` and 0 otherwise — i.e. **M is strictly lower-triangular** (a token
can only be erased by *earlier* writes, causality). So `(I+M)` is
**unit lower-triangular**, and the system is solved *exactly* by forward
substitution — `torch.linalg.solve_triangular` (model.py:1423,
`_complex_triangular_solve`), which is ~26× faster than a general solve
because no LU factorization or pivoting is needed.

This is the **UT transform** (the standard trick from the DeltaNet /
parallel-linear-attention literature for parallelizing gated delta rules):
one chunk = build the C×C mass matrix `M`, one triangular solve, done.
The code (model.py:984-992, 1042-1067):

```python
key_gram  = K @ K*                    # [B,H,C,C]  "how similar are the chunk's keys"
mass      = βe[:,None] * D * strict_lower * key_gram      # [K,B,H,C,C]
update    = solve_triangular(I + mass, write)             # [K,B,H,C,d]
```

Everything else in the delta chunk is then exactly the additive dual form:
in-chunk read `y = ((W ⊙ D)_causal @ update) / √d` (model.py:1068-1072),
carry read (model.py:1077-1093), and the state write
`S += (update ⊙ decay_tail) @ K*` (model.py:1073-1105).

### 9.4 The fused K path (what production actually runs)

`_forward_multistate_delta_fused` (model.py:952) is the production path. It
observes that most of the chunk work is **independent of the state index k**,
so it's computed once, not K times:

| Computed ONCE (K-free)                          | Computed per-K (cheap)              |
|-------------------------------------------------|-------------------------------------|
| key-gram `K@K*`, query-key `Q@K*` (985-992)     | decay matrix D (per-state γ)        |
| the triangular solve, batched over K·B·H (1058) | carry read `q·α @ S_k`              |
| write RHS `βw·v − βe·k@S_k` (1000-1019)         | state write `S_k += update@K*`      |
|                                                 | phase-rotate + sum over K (1106-1116) |

The K per-chunk solves collapse into ONE batched solve over `K·B·H`
matrices (model.py:1058-1065): the K sequential chunk-solves become one
batched call (96 → 24 internal UT steps at K=3, C=32, per the note at
model.py:944-945). This is a pure *regrouping* — bit-identical math,
verified by `v13/selftest.py::test_fused_e3_equiv` (1e-7, fp32).

---

## 10. Putting a layer together: block, model, training step

### The block (`V13Block`, model.py:1440)

Each of the 16 layers is **two mixers on a residual stream** (pre-norm):

```python
x = x + cgu_scale * CGU(norm1(x))     # channel mix: features→features, within ONE token
x = x + pam_scale * PAM(norm2(x))     # sequence mix: tokens→tokens, via the memory matrix
```

- **CGU** (`ComplexGatedUnit`, complex_ops.py:225) is the MLP: gate/up/down
  complex linears + a phase-preserving Swish. It mixes *features inside one
  token* — no time involved.
- **PAM** is everything in §2–§9: it mixes *information across time* using
  the d×d notebook.

Both residuals start small (`pam_scale` init 0.1, model.py:1455) so early
training is dominated by the cheap MLP and the memory pathway warms up
gradually — a stability trick, not math.

### The model (`V13LM`, model.py:1487)

```
ids [B,T]
 → ComplexEmbed  → z [B,T,dim,2]          # two embeddings (real, imag) stacked
 → ComplexNorm
 → 16 × V13Block                           # carries the K,B,H,d,d,2 state per layer
 → output_norm → lm_head_proj (complex linear) → lm_head_norm   → lm [B,T,dim,2]
 → TIED head:  logits[b,t,w] = lm_r @ E_r.T + lm_i @ E_i.T      # [B,T,V]
```

The head is **tied**: it reuses the embedding table (`embed_real/imag`
weights, model.py:1576-1579) instead of learning a second one — halves the
vocab-sized parameters. Algebraically it's one real matmul on concatenated
rows: `concat(lm_r,lm_i) @ concat(E_r,E_i).T` (the trick `fused_ce` exploits).

### The training step (v7/train.py:482-573, shared by v13)

```python
for batch:                                    # [B,T] ids
    lm, aux, gate_probs = hidden_fn(ids)      # stack forward, no [B,T,V] logits
    loss = fused_CE(lm, labels)               # chunked, exact
    loss += aux                              # routing balance (off by default)
    loss += 0.1 * gate_BCE(gate_probs, nll)  # gate-surprisal (§5)
    loss.backward()                           # autograd: all 16 layers
    clip_grad_norm_(params, 1.0); optimizer.step()
```

The **fused CE** (v13/fused_ce.py) never materializes the `[N= B·T, V≈50k]`
logit tensor (~4 GB fp32 + softmax + grad). It walks the hidden rows in
4096-row chunks: each chunk does `chunk @ W.T` (one `[4096, V]`), a CE, and
its backward grads are accumulated. Peak memory O(chunk·V) instead of
O(N·V); the math is *exact* (verified against `F.cross_entropy` in
`selftest::test_fused_ce_equiv`). As a byproduct it emits the exact
per-token NLL, which the gate-surprisal aux reuses for free.

---

## 11. How backprop works here (the "why" of it)

PyTorch's autograd builds a graph of every op you ran in forward.
`loss.backward()` walks it *backwards*, and each op supplies a **Jacobian**:
"how did *my* output change my inputs?" The chain rule multiplies them along
the path. You never write gradients — you just have to make sure (a) every op
is differentiable and (b) the ops you care about are in the graph.

What that means concretely in this codebase:

1. **The loss lives at the vocab head.** `dL/dlm` = the softmax error
   `(p − one_hot)/N` — computed in `fused_ce.backward` (fused_ce.py:71-97)
   chunk by chunk. It flows into `lm_head_proj`, then `output_norm`, then
   every block.
2. **Through the PAM layer it splits three ways** (all in one backward pass,
   automatically):
   - to **Q/K/V projections** (the `@` matmuls' Jacobians route `dL/dy`
     back to `q, k, v`),
   - to the **decay/gate/beta linears** (via `γ`, `p`, `βw`, `βe`),
   - to the **chunk solve**: `solve_triangular` has an exact autograd
     formula (backward is another triangular solve), so the mass matrix's
     dependents — `erase_beta_proj`, `dt_proj`, and `keys` through the
     key-gram — all get gradients.
3. **The recurrent form is only used at inference**, so the parallel chunk
   math is what trains; `selftest` proves `parallel_train ≡ recurrent_infer`
   so training the fast form is safe.

Two bugs from this codebase's history show why (a) and (b) matter:

- **Detached checkpoint input** (model.py:1657-1665): gradient
  *checkpointing* recomputes a block in backward to save VRAM; if the block's
  input is `.detach()`ed, the input edge vanishes from the graph and 15 of 16
  blocks silently get **zero gradient** — training "works" at 4× speed while
  only the last layer learns. Caught by the `[block-grad step1]` canary in
  the log (v7/train.py:460-469) and `selftest::test_grad_ckpt_equiv`.
- **`g/mag` normalize shortcut**: replacing `x/||x||` with an
  `autograd.Function` that returns `g/mag` dropped the gradient through
  `mag` — 19% key-grad error (complex_ops.py:95-107). Lesson: the default
  autograd Jacobian of the ops you use is *exact*; hand-written backward
  usually loses terms.

---

## 12. Squeeze / unsqueeze / transpose / view / reshape — the "why"

These never change the numbers, only the **layout** (which axis is which, and
whether memory is contiguous). They exist because matmul only cares about
the *last two* axes and broadcasts against the *leading* ones.

The recurring patterns in this file:

1. **Move heads before time** — `_project` (model.py:363-371):
   `view(B,T,3,H,d,2)` then `.transpose(1,2)` → `[B,H,T,d,2]`. Why: one PAM
   matmul (e.g. `q @ k*`) must be done **per (batch, head)** — 8×6 = 48
   independent small matmuls. PyTorch's `@` treats leading axes as *batches*,
   so putting `B,H` first turns the loop into one batched call. The
   transpose makes a non-contiguous view, so `.contiguous()` copies it into
   dense memory (one copy, once, worth it).
2. **`unsqueeze` = add a size-1 axis so broadcasting lines up.**
   In the recurrent step, `decay_gamma_t` is `[B,H]` (one number per head)
   but the state is `[B,H,d,d,2]` (model.py:1381). To scale each head's whole
   `d×d×2` block by its γ:
   `γ.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)` → `[B,H,1,1,1]`, which
   broadcasts over d,d,2. The same trick scales values by the protect gate
   (`[B,H,T]` → `[B,H,T,1,1]` over d,2, model.py:806). Size-1 axes are the
   "repeat me across these dimensions" marker in broadcasting.
3. **`squeeze` = the inverse** — remove a size-1 axis:
   `write_phase_proj(cabs(keys)).squeeze(-1)` (model.py:406) turns the
   linear's `[B,H,T,1]` output into `[B,H,T]` (one phase angle per token).
4. **`view`/`reshape` = regroup the same numbers into different dims.**
   After the fused QKV projection, `view(B,T,3,H,d,2)` *interprets* the flat
   `3·H·d·2` last axis as (q/k/v, head, channel, real/imag) — no data move.
   `reshape(B*H, C)` (model.py:640) folds (batch, head) into one batch axis
   before the decay kernel; `.reshape` back after.
5. **`permute` = reorder axes (a general transpose)**, used to line up the
   phase/routing tensors `[B,T,H,K]` → `[K,B,H,T]` (model.py:958) so K is
   the batch axis of the fused path.

**The mental model:** every tensor is "a batch of 2D matrices". Matmul is
defined on the last two axes; everything in front is just "do this many
independently". All the squeeze/unsqueeze/transpose/view machinery is
reshaping so the axis you want *batched* is in front and the two axes you
want *multiplied* are at the back.

---

## 13. Where everything lives (cheat-sheet map)

| Concept (this doc)              | Code                                                                 |
|---------------------------------|----------------------------------------------------------------------|
| split-real complex layout       | `v13/complex_ops.py:29-56` (`REAL`, `IMAG`, `cmul`, `cconj`, `cabs`) |
| per-vector key norm (NaN fix)   | `v13/complex_ops.py:108-117` + `model.py:399-400`                    |
| QKV projection + RoPE + phase   | `model.py:354-410` (`_project`, `_apply_write_phase_address`)        |
| decay γ, protect gate p, vault  | `model.py:422-473` (`_gamma_and_vprime`), `779-810` (K-at-once)      |
| write/erase betas βw, βe        | `model.py:261-274` (`_gate_betas`)                                   |
| K phases / routing              | `model.py:313-350` (`_phase_and_alpha`)                              |
| additive token step (notebook)  | `model.py:1333-1371` (`_recur_step_additive`)                        |
| delta token step (production)   | `model.py:1373-1416` (`_recur_step_delta`)                           |
| additive chunk/dual form        | `model.py:478-540`                                                   |
| delta chunk + UT solve          | `model.py:614-715`, solve at `1423-1435`                             |
| K-state loop (reference)        | `model.py:719-763` (`_forward_multistate`)                           |
| **fused K production path**     | `model.py:952-1118` (`_forward_multistate_delta_fused`)              |
| block (CGU + PAM residual)      | `model.py:1440-1484`                                                 |
| model (embed→blocks→tied head)  | `model.py:1487-1586`                                                 |
| generation (prefill + O(1))     | `model.py:1674-1713` (`generate`)                                    |
| training step + gate aux        | `v7/train.py:482-573`, `354-431`                                     |
| fused (chunked) cross-entropy   | `v13/fused_ce.py`                                                    |
| train≡infer proofs              | `v13/selftest.py` (run: `python -m v13.selftest`)                    |

**Reading order suggestion:** §1–§3 (shapes + the two products) → §4
(the notebook step, with a `tiny` preset you can step through in a debugger)
→ §9 (why training uses matmuls + the triangular solve) → §11 (backprop) →
then the cheat-sheet map. `MATRIX_COOKBOOK.md` has every op in the hot path
with its real shape and its one-sentence job.
