# semantic/ — PLAN

A generic, rank-agnostic, named-axis tensor layer over standard PyTorch.
Models are written in **easy terms** (words) and then **code terms** (named ops);
the framework computes the real layout work. qllm-v13 and pands are *users* of it,
not its purpose — transformers, gamma, kimi or anything else fit the same shape.

## Phases

### P0 — Dimension & layout algebra  [core.py]
- `Dimension(name, size=None)`: named axis object; size symbolic until bound.
- `Layout = (Dim,)*`: ordered axes, every position named at any rank.
- `arrange(new_layout)`: exact permute + split/merge (split needs size factorization).
- `swap(A, B)` / permute by names.
- `LayoutError`: names both offending dims + expected/actual sizes.
- Self-correlation: shared `Dimension` objects across tensors check size consistency
  before any op executes.

### P1 — Named semantic ops + pluggable element policies  [core.py, policies.py]
- `SemanticTensor` wraps a real `torch.Tensor` + a `Layout`; ops emit standard torch
  (`view/permute/bmm/matmul/cumsum/...`). No custom autograd.
- `contract(a, b, over=Dim)` — named matmul, any rank.
- `outer(a, b, over=(A,B))` — named outer product.
- `read(S, probe, over=Dim, conjugate=...)` — inner product with optional conj
  ("read/collapse"); `write` = `outer` with conj ("write/expand").
- `scale`, `add`, `squeeze(name)`, `normalize_vec(over=Dim)`.
- **Policy interface** (overridable internals): `mul/add/sub/neg/conj/abs/norm_vec/
  stack/unpack`. Three policies ship:
  - `Real` — plain float tensors, default.
  - `SplitRealComplex` — v13-compatible split-real `[..., 2]` algebra (bit-exact
    epsilons; `norm_vec` keeps v13 `cnormalize_vec`'s exact form).
  - `NativeComplex` — torch `complex64`, standard torch ops (`torch.conj`, `abs`) —
    proves "standard pytorch in the back".

### P2 — Static/correlative checks  [core.py]
- Size correlation across shared dims before executing; `contract` over mismatched
  named axes raises `LayoutError` with both dim names + sizes
  (the "B.D is not possible as their dimensions don't match" requirement).

### P3 — Proofs  [selftest.py]  (CPU, tiny)
- (a) layout algebra unit tests incl. error cases.
- (b) v13 `_recur_step_delta` in semantic language ≡ raw-torch transcription
  (bit-for-bit / ≤1e-6) — reference-model correctness proof.
- (c) v13 `_fused_chunk_step` dual form ≡ raw torch (first + non-first chunk).
- (d) **generic non-qllm proof**: tiny multi-head attention (B,T,H,d,K,V) in
  semantic language ≡ raw torch — the framework is common, not qllm-specific.

### P4+ — (documented, out of current scope)
- Port one full v13 PAM chunked-delta block end-to-end as a reference model in
  semantic language (incl. `_project` RoPE path).
- nn.Module integration (semantic tensors as module outputs), torch.compile
  interplay, full model ports (transformer/gamma/kimi), backward-semantics docs.

## Vocabulary: easy terms → code terms
| easy term | meaning | code term |
|---|---|---|
| read / collapse | inner product, one number out | `read(S, probe, over=d)` (conj on probe) |
| write / expand | outer product, d×d table out | `write` = `outer(u, conj(key), over=(d_out, d_in))` |
| decay / forget | per-token scale γ | `scale(S, gamma)` (auto-broadcast) |
| delta | write the correction | `u = βw·v − βe·read(S,k)` |
| attend | score = q·k̄, softmax, mix values | `contract(q,k,over=d)` → `softmax` → `contract(scores,v,over=T)` |
| phase | angle, adds under multiplication | policy-level `conj`/rotation |
