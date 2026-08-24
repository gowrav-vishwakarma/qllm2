# v13_sempty — torch → sempyt mapping

This file is the audit list for `model.py` PAM kernels. **Goal:** no integer-axis
bookkeeping (`transpose`, `unsqueeze`, `view`, `permute`, `[..., 0]`) where sempyt
has a named equivalent.

## Core replacements

| Old torch pattern | sempyt replacement | Notes |
|-------------------|-------------------|-------|
| `x.view(B,T,H,d)` | `x.to(batch, time, heads, head_feature)` | layout by name |
| `x.permute(0,2,1,3)` | `x.to(batch, heads, time, head_feature)` | same data, named order |
| `x.transpose(-1,-2)` before `@` | `contract(a, b, over=shared_axis)` | matmul = contract |
| `a @ b.T` (real) | `contract(a, b, over=feat)` with `b` on the contracted axis | |
| `Q·K*` (complex) | `contract(q, k.conj(), over=head_feature)` | SplitComplex `conj()` |
| `(Q·K*) @ V` (complex) | `contract(scores, values, over=source_col)` | complex `*` comes from the policy |
| `real/imag` via `[...,0]` | `real(z)`, `imag(z)` | drops `complex_pair` axis |
| `torch.stack([r,i], -1)` | `as_complex(r, i, complex_pair)` | |
| `x.unsqueeze(-1) * y` (broadcast) | `x * y` on NamedTensors | missing axes broadcast as size-1 |
| `x.sum(dim=0)` over K | `sum(x, over=memory_states)` | |
| `torch.cos(phase)` | `cos(phase)` | `sempyt.structural` |
| `select(x, dim, i)` | `select(x, over=dim, index=i)` | |
| `S @ Q` (memory read) | `contract(memory, query.alias(feat, head_col), over=head_col)` | v13 reads with the raw probe, no conj |
| `V.T @ K` (outer write) | `contract(v.alias(feat,row), k.conj().alias(feat,col), over=time)` | |
| `torch.cumsum(..., dim=-1)` | `cumsum(named(...), over=time)` | |
| RoPE on Q,K | `q * rope_named` | SplitComplex multiply |
| phase rotate `z * e^{iφ}` | `z * as_complex(cos(φ), sin(φ), complex_pair)` | |
| `torch.zeros(K,B,H,d,d,2)` | `zeros(memory_states, batch, heads, head_row, head_col, complex_pair)` | sized from the axes |
| `torch.tril(torch.ones(C,C), -1)` | `tril_mask(write_row, source_col, diagonal=-1)` | |
| `x[:, :, a:b]` chunk slice | `take(x, over=time, start=a, length=b-a, new=chunk_time)` | |
| `torch.cat(chunks, dim=2)` | `cat(chunks, over="chunk_time", into=time)` | ragged last chunk is fine |
| `torch.linalg.solve_triangular` | `solve_triangular(mass, write, over=(write_row, source_col), feature=head_row)` | broadcasts the state axis |
| `torch.arange` + `==` mask | `eq(arange(memory_states), idx)` + `where(...)` | e.g. the vault state |

## Intentional raw-torch exits

Everything else is named end to end; `check_torch_layout.py` enforces it.

| Site | Why |
|------|-----|
| `complex_ops.fused_decay_matrix` | builds the `[time, time]` lag table (cumsum + tril) |
| `complex_ops.real_part` / `imag_part` | pack `concat(real, imag)` for the chunked CE |
| `V13LM.ce_from_lm` | chunked autograd; must not materialize `[N, vocab]` |
| `V13LM.generate` | sampling loop over raw logits |
| `_as_token` / `_as_memory` | re-state axis identities on data that does not move |

`torch.linalg.solve_triangular` is no longer an exit: it is wrapped by
`sempyt.solve_triangular`, which takes `over=(write_row, source_col)` plus a
`feature` axis and broadcasts every other axis, so the delta solve needs no
packing at all.

## `pam_ops.py` helpers (canonical)

| Helper | Replaces |
|--------|----------|
| `conjugate_scores` | `q_r@k_r.T + q_i@k_i.T` and friends |
| `cumulative_decay` | `exp(cumsum(log gamma, dim=-1))` |
| `decay_matrix` | reshape/flatten around `fused_decay_matrix` |
| `phase_rotation` | `stack([a*cos, a*sin], -1)` |
| `delta_chunk` | the whole UT-solve chunk: mask build, packing, solve, read |
| `recur_step_delta` | the decode step's transpose gymnastics |

## Status (2026-08-24)

- **Done:** every path v13_sempty implements — the E3 delta-fused parallel
  forward (both the materialized and factored solves) and the recurrent decode.
  `check_torch_layout.py` is clean; all 8 selftest stages match v13.
- **Removed:** the E1/E2 and non-fused ablation branches, which production never
  ran (`forward` now raises `NotImplementedError` outside the E3 delta path).
