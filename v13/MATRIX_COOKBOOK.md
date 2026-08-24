# V13 Matrix Cookbook — every op in the hot path, with real shapes

Companion to [`MATH_EXPLAINER.md`](MATH_EXPLAINER.md). If the shapes below
feel like alphabet soup, read [`BEGINNER_STORY.md`](BEGINNER_STORY.md)
first — it defines every letter (B, H, d, K, …) and every operation (inner
product, outer product, phase, magnitude) with small worked examples
before this cookbook uses them.

A "complex matmul" `A@B` is written as the two real GEMMs that implement it
(`(a_r b_r − a_i b_i, a_r b_i + a_i b_r)`) — the code always does exactly this,
never `torch.complex` (except inside the triangular solve, which is a
leaf op).

---

## 1. Per-token inputs (before PAM)

| Tensor | Shape | Made by |
|--------|-------|---------|
| `input_ids` | `[B, T]` | dataloader |
| `z` (token features) | `[B, T, dim, 2]` = `[8,2048,384,2]` | `ComplexEmbed` (two `nn.Embedding`s, stacked) |
| block input after norm | `[B, T, dim, 2]` | `ComplexNorm` (RMS on magnitude) |

## 2. QKV projection — `_project` (model.py:354-401)

```
qkv     = qkv_proj(z)                [B, T, 3·H·d·2] = [8,2048,2304]
view    → [B, T, 3, H, d, 2]        reinterpret, no data move
queries = qkv[:,:,0].transpose(1,2)  [B, H, T, d, 2] = [8,6,2048,64,2]
keys    = qkv[:,:,1] ...             same shape
values  = qkv[:,:,2] ...             same shape
```

- **job**: turn each token's 384 features into (query, key, value) × 64
  channels × 6 heads. The fused single GEMM (`ComplexLinear(384→1152)`) costs
  one `[...,384] @ [384,1152]`-pair instead of three.
- **transpose(1,2)**: moves heads in front of time so the next matmuls batch
  over (B,H) = 48 independent heads. `.contiguous()` because a transposed view
  is strided; the copy costs once, the batched GEMMs benefit for the whole
  layer.
- **RoPE** (`cmul` with `rope_cache[pos:pos+T]` `[T,d,2]`): elementwise
  complex multiply — rotates each channel of q,k by `e^{i·pos·θ_j}`.
  Magnitude untouched; only phase.
- **write_phase_address** (model.py:403-410): `values = rotate(v, ψ(|k|))`,
  `queries = rotate(q, ψ(|q|))` where `ψ = Linear` on magnitudes, `[B,H,T]`.
- **key norm** (model.py:399-400): `keys = keys / ||keys||_d` — one vector
  norm per (b,h,t) across the 64 channels (see explainer §6).

## 3. Gates and decay (per token, per head)

| Tensor | Shape | Made by | Meaning |
|--------|-------|---------|---------|
| `decay_gamma` | `[K, B, H, T]` | `exp(−softplus(dt_proj(x_flat)+bias+state_offset))` (model.py:779-810) | per-state forget factor γ∈(0,1); state 0 pinned to 1 (vault) |
| `protect_prob p` | `[B, H, T]` | `sigmoid(protect_gate(x_flat))` (model.py:801) | "how much to freeze memory / mute writes" |
| `protected_values` | `[B, H, T, d, 2]` | `values * (1−p)` (model.py:806) | the value that actually gets written |
| `write_beta` `erase_beta` | `[B, H, T]` each | `sigmoid(β_proj(cabs(x)))`, erase clamped ≤0.95 (model.py:261-274) | write gain βw; erase gain βe |

`x_flat` = `to_real_concat(x)` = `concat(real, imag)` → `[B,T,2·dim]` — the
real 768-dim view used by all the plain (non-complex) gate linears.

---

## 4. The production chunk — `_forward_multistate_delta_fused` (model.py:952-1118)

One chunk of `C=128` tokens. Slice everything to `[·, C, ·]` first.

### 4.1 K-free score matrices (computed ONCE)

```
key_gram   = K_c @ K_c*        [B, H, C, C] = [8,6,128,128]
query_key  = Q_c @ K_c*        [B, H, C, C]
```

- **key_gram[t,s] = k_t · k_s*** — "how similar is the chunk's token t to
  token s's key". A real-symmetric-ish complex Gram matrix (diagonal = |k|²
  = 1 with key-norm). Feeds the *erase* mass.
- **query_key[t,s] = q_t · k_s*** — the retrieval score "token t asks, token
  s's key answers". Feeds the *in-chunk read*.
- Both are the **inner product** (collapse, one number per token pair).
  Two real GEMMs each (e.g. `q_r @ k_r.T + q_i @ k_i.T`), batched over B·H.

### 4.2 Per-state decay (K appears here)

```
log_decay        = log(decay_gamma_chunk)          [K, B, H, C]
cumulative_alpha = exp(cumsum(log_decay, dim=-1))  [K, B, H, C]   α_t = γ_1…γ_t
decay_matrix     = fused_decay_matrix(γ)           [K, B, H, C, C] D[t,s] = γ_{s+1}…γ_t
```

- `cumulative_alpha` (α): "how much of the *start* of the chunk survives to
  time t" — used to decay the carried-in state and the cross-chunk erase.
- `decay_matrix` (D): the full "s→t survival" table, built by a small Triton
  kernel (v13/triton_kernels.py) instead of a cumsum-log-exp per pair.
- Per-state because each of the K states has its own γ (different dt offset;
  state 0 = vault = all ones).

### 4.3 The erase mass + triangular solve (the delta-specific part)

```
strict_lower            = tril(ones(C,C), -1)               [C, C]   (no diagonal!)
mass   = βe[:,None,None] · D · strict_lower · key_gram      [K, B, H, C, C]
state_key = K_c @ S_k*  (per-state carried read at keys)    [K, B, H, C, d]
         × cumulative_alpha                                  (decay it into the chunk)
write  = βw[:,None] · v − βe[:,None] · state_key             [K, B, H, C, d]
update = solve_triangular(I + mass, write)                   [K, B, H, C, d]  ← THE solve
```

- **mass[t,s]** = "erase coupling from earlier token s into token t":
  βe_t (how hard t erases) × D[t,s] (how much of s survives to t) ×
  k_t·k_s* (how much t's key looks like s's key). Strictly lower-triangular:
  a token can't be erased by its own or later writes.
- **write** is the RHS: the new value, minus what the *previous* state
  already predicts for this key (the cross-chunk delta term).
- **the solve**: `(I+M)u = w`, M strictly lower-tri ⇒ unit-triangular system
  ⇒ exact forward substitution, batched over K·B·H = 144 systems of 128×128
  (model.py:1058-1067, via `_complex_triangular_solve` which uses
  `torch.linalg.solve_triangular` on a real `torch.complex` tensor — the one
  place real complex tensors appear; it's a leaf op with exact autograd).
- Result **update** u_t: the *actual* write each token performs, with all
  intra-chunk erase interactions already accounted for. This is the UT
  transform: C sequential delta steps → one C×C solve.

### 4.4 In-chunk read

```
projection = (D ⊙ tril(ones)) · query_key                 [K, B, H, C, C]
output     = (projection @ update) / √d                   [K, B, H, C, d]
```

- **outer-product read**: "token t sees every *write* u_s in its chunk,
  weighted by (score q_t·k_s*) × (survival D[t,s])". `@ update` is a matrix
  **times vector-table**: [C,C] @ [C,d] — d outputs per query token, each a
  d-gram sum. Causal via the `tril` mask (a token never reads future writes).
- `/√d` = `query_scale` (model.py:956) — keeps output variance ~independent
  of head_dim (the same 1/√d trick as attention).

### 4.5 Carry read (previous chunks' state)

```
carried_k  = (q/√d · α) @ S_k                              [K, B, H, C, d]
output    += carried (rotated & summed over K below)
```

- The notebook built by *earlier* chunks, read at this chunk's queries,
  decayed by α (how much survives into the chunk). Matrix-vector: [C,d] @
  [d,d] per (K,B,H).

### 4.6 State write + carry-out

```
decay_tail      = α_C / α                                   [K, B, H, C]  (survival t→end of chunk)
update_decayed  = update · decay_tail                       [K, B, H, C, d]
state_chunk     = update_decayed* @ K_c                      [K, B, H, d, d]   ← OUTER PRODUCT
memory_state   = memory_state · α_C + state_chunk            [K, B, H, d, d, 2]
```

- **`state_chunk` is the outer product in batched form**: `[C,d].T @ [C,d]`
  = Σ_s u_s ⊗ k_s* — the sum of all the chunk's writes as one feature×feature
  matrix, with each write pre-decayed by "how long it lives in this chunk".
- Old state is multiplied by α_C (total decay across the whole chunk) and
  the new table added. **The [d,d] memory is what the next chunk inherits.**

### 4.7 Phase-routed combination over K

```
rotation  = routing_weights · e^{i·phase}                  [K, B, H, C]
out_token = Σ_k rotation_k · output_k                       [B, H, C, d]
```

- Elementwise complex multiply then sum over K (model.py:1106-1116). This is
  where "E3 superposition" lives: three notebooks, one rotated answer.

### 4.8 Output merge + head (after the chunk loop)

```
output   = cat(chunks, dim=2)   [B, H, T, d, 2]
         .transpose(1,2).view() [B, T, H·d, 2] = [B, T, 384, 2]   (heads merged back)
o_proj   (ComplexLinear 384→384) → dropout → residual add
```

The block does the same for CGU; `V13LM` runs 16 of these, then the tied head:

```
logits = lm_r @ E_r.T + lm_i @ E_i.T     [B, T, V]
       ≡ concat(lm_r,lm_i) @ concat(E_r,E_i).T      [B·T, 768] @ [768, V].T
```

`fused_ce` computes `mean CE(logits, labels)` chunk-by-chunk over the
`B·T=16384` rows so the `[16384, 50261]` tensor never exists.

---

## 5. The inference step — `_recurrent` → `_recur_step_delta` (model.py:1373-1416)

Same math as §4, per **single token** (C=1), so the "chunk" collapses to a
vector and the triangular solve disappears (a 1×1 solve is trivial). This is
what `generate()` runs per new token — cost independent of context length.

```
# per (k) state, per (b,h):
decay_factor   = γ_k                                [B, H, 1, 1, 1, 1]
memory_state  *= decay_factor                       [K,B,H,d,d,2]      ← forget
predicted     = S_k · k_t                           [B, H, d]          ← READ at k
update        = βw·v_t − βe·predicted               [B, H, d]          ← delta
key_conj      = (k_r, −k_i)                         [B, H, d]
outer         = update ⊗ key_conj                   [B, H, d, d, 2]    ← OUTER PRODUCT
memory_state  += outer                              [K,B,H,d,d,2]      ← write
output_k      = S_k · q_t                           [B, H, d]          ← READ at q
out           = Σ_k (w_k e^{iφ_k}) · output_k       [B, H, d]          ← phase-route
```

Note the three products in one step, which is the whole model in miniature:

| Op in the step | Product type | Collapses/expands | Job |
|----------------|--------------|-------------------|-----|
| `S_k · k_t` (predicted) | **inner** | [d,d]×[d] → [d] | ask the notebook "what do you have for k?" |
| `update ⊗ k*` (write)   | **outer** | [d]×[d] → [d,d] | store the correction under key k |
| `S_k · q_t` (output)    | **inner** | [d,d]×[d] → [d] | read back at query q |

The state is the *only* thing carried between tokens: `[K,B,H,d,d,2]` =
3·8·6·64·64·2 ≈ 1.2M complex numbers ≈ 4.8 MB fp32 **per layer**, per
sequence — fixed. (A transformer's KV cache at T=2048 would be
2·T·H·d·2 ≈ 6.3 MB per layer per sequence and grows with T.)

## 6. The shape ladder (one layer, one direction)

```
[B,T]  ids
  │  ComplexEmbed (table lookup)
[B,T,dim,2]  token features
  │  per block ×16:  CGU (channel mix, within token)
  │  per block ×16:  PAM
  │     ├─ _project:  [B,T,dim,2] → [B,H,T,d,2]  (q,k,v; heads before time)
  │     ├─ chunk loop (16 chunks of C=128):
  │     │     K-free:  [C,d] @ [d,C]  →  [C,C] scores (inner products)
  │     │     K-batch: [C,C] solve    →  [C,d] updates      (delta rule)
  │     │     K-batch: [C,C] @ [C,d]  →  [C,d] output       (in-chunk read)
  │     │     K-batch: [C,d].T @ [C,d]→ [d,d] state_chunk    (outer products)
  │     │     state:   [d,d] ← γ·[d,d] + [d,d]               (the memory)
  │     └─ phase-route over K → [B,H,T,d,2]
  │     o_proj: [B,H,T,d,2] → [B,T,dim,2] (heads merged)
[B,T,dim,2]  after 16 blocks
  │  tied head: concat @ concat.T
[B,T,V]  logits  →  cross-entropy (chunked, never fully materialized)
```

**Reading any line**: the rightmost two dims are the ones being multiplied;
everything left of them is "do this many independently" (batch axes).

## 7. Where each tensor came from and why it's shaped that way

| Tensor | Shape | Why this shape |
|--------|-------|----------------|
| `queries/keys/values` | `[B,H,T,d,2]` | heads must be a *batch axis* for the per-head matmuls; `T` stays time; `d` is the multiplied axis; `2` last is the split-real convention (complex_ops.py:29) |
| `key_gram`, `query_key` | `[B,H,C,C]` | one score per (query token, key token) pair inside the chunk — a square pair table, the shape of any "similarity between all pairs of C things" |
| `decay_matrix D` | `[K,B,H,C,C]` | survival from s→t for every pair; per-state because γ differs per state |
| `mass` | `[K,B,H,C,C]` | same pair table × erase gain × key similarity; strictly lower-tri because causality (s<t) |
| `write`, `update` | `[K,B,H,C,d]` | one d-vector per (state, batch, head, token) — the value being written |
| `memory_state` | `[K,B,H,d,d,2]` | feature×feature table per (state, seq, head); T is *gone*, collapsed by the sum of outer products |
| `cumulative_alpha` | `[K,B,H,C]` | one scalar per (state, batch, head, token): product of decays from chunk start |
| `retrieval_phase` | `[K,B,H,T]` | one angle per (state, batch, head, token) — the read-time "tuning" |
| `protected_values` | `[B,H,T,d,2]` | values muted by the protect gate; state-independent (one per token) |
| `logits` | `[B,T,V]` (virtual) | one score per (seq, position, vocab token); fused CE never builds it |

## 8. The two products — semantic summary

**Inner product `u·v*`** — *collapse*: d+d numbers → 1 number. "How much is v
in u?" Used for every **read** (retrieval) and every **similarity** (mass).
Self-similarity `u·u* = |u|² > 0` is the reason matching keys light up.

**Outer product `u⊗v*`** — *expand*: d+d numbers → d×d table. "Bind u to v
across all channels." Used for every **write** (storage). The table's entry
`[i,j] = u[i]·conj(v[j])` says "value-feature i is associated with
key-feature j".

They are inverse operations in the sense that `read(write(x))` recovers `x`
(scaled by the similarity): `(v⊗k*)·q = v·(k*·q)`. The d×d memory matrix is a
sum of such bindings; reading is one inner product against it; that's the
entire "associative memory in a matrix" idea everything else (decay, delta,
phases, K states) is built around.
