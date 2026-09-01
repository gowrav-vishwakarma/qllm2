# v13_sempty

A simple, **pure** Phase-Associative Memory (PAM) language model on the
[sempyt](https://github.com/gowrav-vishwakarma/sempyt) named-axis frontend
(`/home/gowrav/Development/sempyt/src`).

It is *not* v13 ported: it is the lean recurrence v13 grew around, written
cleanly. The design, the decision log (what was cut and why, with log
evidence), and the experiment ledger live in
[`EXPERIMENTS_SEMPY.md`](EXPERIMENTS_SEMPY.md). This file is the contract.

## The model

Each layer carries one notebook per head — a complex d×d matrix:

```
notebook_t = decay_t · notebook_{t-1} + value_t ⊗ conj(key_t)
read_t     = d^(-1/2) · (notebook_t · query_t)
```

Every token fades the notebook by a learned per-head number `decay_t ∈ (0,1)`,
writes one conjugate association (value hung on the key's phase), then reads
with its query. No gates, no erasures, no extra states. RoPE on Q/K is the
only position mechanism (see the decision log for why learned positions were
cut).

Training/prefill runs the recurrence in windows of `chunk_size` via the
bounded retention matrix `M[s,t] = a_s / a_t = exp(C_t - C_s) ≤ 1`
(``notebook_s = Σ_{t≤s} M[s,t] u_t + a_s·S_0``, log space, bounded backward);
decode runs the same algebra one step per token on the carried notebook. The
two paths agree to round-off — that equivalence is a pre-registered
contract, pinned by `selftest.py`.

## Contract

- **Named end to end.** Every axis is a named `Dim`; layout is `.to()` /
  `.alias()`; products are `contract` / `outer`. `check_torch_layout.py`
  fails the build on `view` / `permute` / `transpose` / `unsqueeze` /
  `[..., 0]` / numeric `dim=` / unmarked `.raw` / `.data` escapes.
- **No finetuning machinery.** Plain cross-entropy only (chunked fused CE or
  `F.cross_entropy`); no auxiliary losses.
- **The sROI rule stands.** A mechanism must beat the baseline by more than
  noise to earn its compute (see `EXPERIMENTS_SEMPY.md`).

### Named axes

| Axis | Meaning |
|------|---------|
| `batch` | sequence items in the minibatch |
| `time` | token positions |
| `model_dim` | residual / embedding width |
| `heads` | PAM heads |
| `head_feature` | per-head channel width (d) |
| `complex_pair` | last axis of size 2: real then imag |
| `qkv_slot` / `qkv_fused` | fused Q/K/V packing (slot 0=q, 1=k, 2=v) |
| `head_row` / `head_col` | the two axes of the d×d notebook |
| `chunk_time` | token positions inside one window |
| `real_imag_feature` | `concat(real, imag)` along `model_dim` |

### Declared raw-torch boundaries

| Site | Why it stays raw |
|------|------------------|
| `complex_ops.build_rope_cache` | position table built once, outside the graph |
| `LM.generate` | sampling loop over raw logits |
| `LM.ce_from_lm` hand-off | chunked CE must not materialize `[N, vocab]` |
| `fused_ce.py` | custom autograd Function (whole module) |
| `selftest.py` | compares against plain `F.cross_entropy` |
| `train.py` / `generate.py` | optimiser / dataloader / sampling plumbing |

### torch → sempyt mapping (the old SEMPYT_OPS.md, folded in)

| Old torch pattern | sempyt replacement |
|-------------------|--------------------|
| `x.view(B,T,H,d)` / `x.permute(…)` | `x.to(batch, time, heads, head_feature)` |
| `a @ b` (real or complex) | `contract(a, b, over=shared_axis)` |
| `outer(v, k)` | `outer(v, k.conj(), over=(head_row, head_col))` |
| `real/imag` via `[...,0]` | `real(z)`, `imag(z)` |
| `torch.stack([r,i],-1)` | `as_complex(r, i, complex_pair)` |
| broadcast via `unsqueeze` | `x * y` (missing axes broadcast as size-1) |
| `x.sum(dim=…)` over an axis | `sum(x, over=that_axis)` |
| `x[:, :, a:b]` chunk slice | `take(x, over=time, start=a, length=C, new=chunk_time)` |
| `torch.cat(chunks, dim=…)` | `cat(chunks, over="chunk_time", into=time)` |
| `select(x, dim, i)` | `select(x, over=that_axis, index=i)` |
| `torch.cumprod(…, dim=-1)` | `cumprod(x, over=time)` |
| RoPE on Q, K | `q * rope_named`, `k * rope_named` |
| `torch.zeros(d1,d2,…,2)` | `zeros(d1, d2, …, complex_pair, policy=…)` |

One sempty quirk worth knowing: a fused axis splits by name only when its
factors come **at the end** of the target layout (`x.to(batch, time,
qkv_slot, heads, head_feature, complex_pair)` works; the same factors after
`time` do not). `_project` splits first, then moves `heads` before `time`.

## Layout

| File | Role |
|------|------|
| `config.py` | `PAMConfig`, `PRESETS`, `get_config` (data only) |
| `complex_ops.py` | Split-real complex modules on `NamedTensor` + `SplitComplex` |
| `model.py` | `PAMLayer` / `Block` / `LM` (named end to end) |
| `check_torch_layout.py` | fails if anything reaches around sempyt outside a declared boundary |
| `fused_ce.py` | chunked tied-head linear + CE (custom autograd; `grad_weight +=`) |
| `train.py` | self-contained trainer (no `V7Trainer`) |
| `selftest.py` | the model's own contract (no v13 imports) |
| `generate.py` | prefix completion from a checkpoint |

`sempyt` is imported from source: a `.pth` in the qllm2 venv points at
`/home/gowrav/Development/sempyt/src`, so framework edits are live (no pip
install).

## Run

```bash
# the contract: selftest + layout guard (CPU)
.venv/bin/python -m v13_sempty.selftest
.venv/bin/python -m v13_sempty.check_torch_layout

# synthetic smoke train (CPU)
.venv/bin/python -m v13_sempty.train --preset tiny --dataset synthetic --steps 8 --device cpu

# real-data baseline (GPU — confirm the GPU is free first)
.venv/bin/python -m v13_sempty.train --preset baseline --dataset wikitext103 --device cuda --steps 10000

# generate (needs a checkpoint)
.venv/bin/python -m v13_sempty.generate --checkpoint checkpoints_v13_sempty/latest.pt --device cpu
```

Presets: `baseline` (384/6/64/16 — the v11 7d geometry, ~100M), `micro`
(96/3/32/6), `tiny` (64/2/32/2). Checkpoints land in `checkpoints_v13_sempty/`.

## Verification status (CPU, 2026-08-29)

- selftest: **6/6 pass** (parallel ≡ recurrent to 2.98e-07 on logits,
  1.86e-08 on carried notebooks; fused CE exact vs `F.cross_entropy`)
- layout guard: **clean**
- tiny synthetic smoke: loss 5.5556 → 5.5409 over 8 steps, finite, decreasing
