# semantic/ — SCRATCHPAD (for context-compacted instances)

**Mission (user, 2026-08-24, authoritative):** Build a **generic** semantic tensor
framework over PyTorch in this folder. NOT a qllm-only helper.

- Dimensions like `B T H D K d C` are first-class **named axis objects**; tensors are
  declared with them; ops are semantic (`arrange to (B, D, d, T)`, `swap(T, D)`,
  `contract over d`) and the system computes the real permute/reshape/unsqueeze/bmm.
- Must **statically/correlatively catch shape errors** — e.g. contracting `B` with `D`
  when their sizes don't match raises `LayoutError` naming both dims + sizes.
- Dimension objects **self-correlate**: sharing a `Dimension` object across tensors
  infers size relations before any op runs.
- **Rank-agnostic**: every axis position has a name, N-D, never "rows/columns".
- **Human-readable first**: express models (transformers, gamma, kimi, qllm, anything)
  in easy words → then code words. "easy terms then code terms".
- **Overridable internals**: element-wise math is a pluggable **policy**
  (`Real` / `SplitRealComplex` / `NativeComplex`), so complex math à la qllm is a
  swap-in, not baked in.
- **Standard PyTorch in the back where you can** (user explicit: "you should where
  you can"): wrap real torch tensors, emit real torch ops, native complex dtype where
  the policy picks it. No custom autograd.
- **Do NOT change any existing code** (user explicit). Everything lives in `semantic/`.

## Ground rules for me (future instances)
1. CPU-only, tiny tensors (live 500M GPU run in tmux `v13_500m` must not be touched;
   never edit `v11/ v13/ v7/ scripts/`, never stage `v13/` in a commit).
2. Python: `/home/gowrav/Development/qllm2/.venv/bin/python` (torch 2.8.0+cu128).
3. Run: `cd /home/gowrav/Development/qllm2 && .venv/bin/python -m semantic.selftest`
4. Commit `semantic/` only after selftest is green. Never stage the dirty
   `v13/train.py` hunk.
5. Update this file after every compaction (append a dated log entry at the bottom).

## Status (update this every session)
- [x] v13 semantic docs + pands + reference math read (verbatim math captured in handoff;
      key functions: `v13/model.py:1373-1416` `_recur_step_delta`,
      `v13/model.py:812-892` `_fused_chunk_step`, `v13/triton_kernels.py:522-528`
      `_pt_decay_matrix` (the CPU reference — use this, NOT the Triton dispatch :570),
      `v13/complex_ops.py:33-45,71-122`).
- [x] Folder `semantic/` created; SCRATCHPAD.md written.
- [ ] PLAN.md, DESIGN.md
- [ ] core.py, policies.py, selftest.py
- [ ] selftest green
- [ ] commit + user report (before/after contrast, generic-attention evidence)

## Design decisions (locked)
- `Dimension(name, size=None)`; size symbolic until bound by a real tensor; shared
  objects ⇒ size correlation; mismatch ⇒ `LayoutError` naming both dims + expected/actual.
- `Layout = (Dim,)*` — every position named, any rank.
- `arrange(new_layout)`: split (`D→H,d` needs size(D)=size(H)*size(d)), merge, permute.
- `contract(a, b, over=Dim)`: named-axis matmul (batched bmm under the hood);
  shared axis must size-match. `outer(a, b, over=(A,B))`: named outer product.
- `read` = contract with optional conjugate on the probe (key conjugated on write,
  query NOT conjugated on read — that's a policy-level `conj` flag, documented in words).
- Element math dispatches through a **policy** attached per-tensor:
  `Real` (default), `SplitRealComplex` (v13 `[..., d, 2]` split-real, epsilons kept
  bit-exact incl. `cnormalize_vec`'s `+1e-8`), `NativeComplex` (torch complex64 —
  proves "standard pytorch in the back").
- Selftest: (a) layout algebra + error cases; (b) `_recur_step_delta` semantic ≡ raw
  torch bit-for-bit; (c) `_fused_chunk_step` dual form ≡ raw torch (both first and
  non-first chunk); (d) tiny multi-head attention (B,T,H,d,K,V letters) semantic ≡ raw
  torch — the generic (non-qllm) proof.

## What changed since last compaction (append here)
- 2026-08-24: (this entry) references re-verified from v13 source; folder created;
  this scratchpad written. Next: PLAN.md + DESIGN.md, then core.py.
