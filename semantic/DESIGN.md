# semantic/ — DESIGN

## 1. Core idea

A tensor is declared with **named axes**, and every op is expressed over those
names. The framework resolves names → positional layout work (permute/reshape/
unsqueeze/batched-matmul) and executes on standard PyTorch. Nothing about the
underlying dtype or element math is assumed — that is a **policy**, an overridable
internal.

```python
B, T, H, d = Dim("B"), Dim("T"), Dim("H"), Dim("d")
q = tensor(randn(B, T, H, d), policy=Real)          # declared layout: (B, T, H, d)
scores = contract(q, k, over=d)                       # → (B, T, H, T)
attn = softmax(scores, over=T)
out = contract(attn, v, over=T)                       # → (B, T, H, d)
```

Rank-agnostic: every position of the layout is a named `Dimension`; there are no
"row"/"column" concepts. Works for rank 1 (a vector over `d`) through rank 7+.

## 2. Components

### `Dimension`
- `Dimension(name, size=None)` — `size` is symbolic (`None`) until bound.
- Binding: first real tensor that uses a dim binds `size` from that tensor's axis.
- **Self-correlation**: the *same* `Dimension` object appearing in two tensors
  must agree on size; the algebra checks this before executing and raises
  `LayoutError` naming both dims + expected/actual sizes.
- Optional `role` tag (`batch`, `time`, `head`, `head_dim`, `complex_pair`, …) —
  purely documentary/for diagnostics, never affects math.

### `Layout = (Dim,)*`
Ordered tuple of dims. Every position named.

### `SemanticTensor`
- Wraps `torch.Tensor` + `Layout` + element `Policy`.
- `.data` is the real torch tensor; `.layout` the named axes.
- All ops return new `SemanticTensor`s (values stay standard torch tensors, so
  `.data` plugs straight into any torch code).

### Layout algebra — `arrange(new_layout)`
Computes the exact transform from old named axes to new ones:
- **Permute** existing dims in new order (→ `permute`).
- **Split** one dim into several: requires `size(D) == Π size(parts)` when all are
  bound; parts bind as factors of `D` otherwise (order = row-major C order).
- **Merge** several dims into one: requires all but the target to be bound, or the
  target to be unbound (binds to the product).
- Anything else (unknown name, size factorization impossible) → `LayoutError`
  naming the offending dims + sizes.

### Semantic ops
- `contract(a, b, over=D)` — the named matmul: `D` must appear in both, sizes must
  match (correlative check); other dims = batch axes, unioned by name in a
  canonical order; sums over `D`. Emits `bmm`/`matmul`.
- `outer(a, b, over=(A, B))` — named outer product over the two named axes.
- `read(S, probe, over=D, conjugate="probe"|"none")` — inner product ("read"):
  sums over `D`; `conjugate="probe"` conjugates the probe per policy
  (v13: key conjugated on write, query NOT on read — a flag, not a convention).
- `write(S, u, k, over=(A, B), conjugate=True)` — `S + outer(u, conj(k))` style
  update; the update itself is `outer`.
- `scale(x, y)` / `add(x, y)` — element ops with **named broadcast**: axes present
  in only one operand broadcast; shared axes must size-match (LayoutError).
- `squeeze(name)`, `normalize_vec(over=D)` (policy dispatches the norm form).

## 3. Policies — the overridable internals

`ElementPolicy` interface: `mul, add, sub, neg, conj, abs, norm_vec, stack,
unpack`. Element ops in the core (`cmul`-equivalent, `cconj`-equivalent,
`cabs`-equivalent) all dispatch through the policy, so "complex math" is a
swap-in:

| policy | backing | notes |
|---|---|---|
| `Real` | float32/64 | default; `conj` = identity |
| `SplitRealComplex` | float tensor with trailing size-2 `complex_pair` axis | transcribes `v13/complex_ops.py` exactly (epsilons `+1e-8`, `cnormalize_vec`'s per-VECTOR norm form) so bit-for-bit equivalence with v13 holds |
| `NativeComplex` | torch `complex64` | standard torch ops (`torch.conj`, `abs`) — the "standard pytorch in the back" proof |

A policy is attached per-tensor at construction (or per dim via the
`complex_pair` role). Mixing real and complex operands: real broadcasts as
real-part (documented policy rule).

## 4. Static/correlative error catching

Before any op executes, the algebra resolves: (1) layout consistency, (2) size
agreement across shared `Dimension` objects, (3) factorization for splits/merges.
Failures raise `LayoutError` with both dim names, expected size, actual size —
e.g. `contract(x, y, over=B)` with `B.size=8` in x and 6 in y raises
`B (8) vs B (6): sizes disagree for shared dimension 'B'`; contracting over a dim
present in only one operand names the missing side.

## 5. Evidence lines this absorbs (v13 pain → semantic op)

| v13 code (raw) | semantic |
|---|---|
| `decay_gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)` (model.py:1381) | `scale(state, gamma)` |
| `dim() == dim() - 3` branch guessing unsqueezes (model.py:1354-1359) | impossible — names fix it |
| hand-rolled 2-real GEMM + `sum(dim=-1)` (model.py:1384-1390) | `read(S, key, over=d, conjugate="probe")` |
| hand outer `unsqueeze(-1)/(-2)` (model.py:1398-1405) | `write` / `outer(u, conj(k), over=(d,d))` |
| `.view(B,T,3,H,d,2)` + `.transpose(1,2).contiguous()` (model.py:363-371) | `arrange((B,T,H,d,px))` + `swap(T,H)` |
| `retrieval_phase.permute(3,0,2,1)` (model.py:958-959) | `arrange((K,B,H,T))` |
| `reshape(K*B*H, C, C)` into batched ops (model.py:1058-1067) | merge `(K,B,H)` into named `BH` axis |
| `total_decay[...,None,None,None]` (model.py:889-891) | `add(scale(S, total_decay), state_chunk)` |

## 6. What is deliberately NOT in scope yet (P4+)
Full v13 PAM block port incl. RoPE `_project`; nn.Module integration;
torch.compile interplay; full model ports (transformer/gamma/kimi); backward docs.
The core + policies + selftest (P0–P3) is this deliverable.
