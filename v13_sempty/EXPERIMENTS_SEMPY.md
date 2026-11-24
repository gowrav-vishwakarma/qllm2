# v13_sempty — the simple PAM, in words

This is the experiment file for `v13_sempty`: what the model is, what was
deliberately left out and why, and what we plan to measure. The code in
`model.py` is written to be read the way this file is written — the same
words.

## The model, in one paragraph

Each layer owns one notebook per head: a complex d×d matrix. Every token does
three things to it, in this order:

```
notebook_t = decay_t · notebook_{t-1} + value_t ⊗ conj(key_t)
read_t     = d^(-1/2) · (notebook_t · query_t)
```

1. **Fade.** Multiply the notebook by `decay_t`, a learned number in (0, 1).
   One number per head, read off the token:
   `decay_t = exp(−softplus(W·[Re h_t, Im h_t] + b))`. This is the *only*
   learned knob on the memory.
2. **Write.** Add one new association: the token's value hung on the phase of
   its key. The write is conjugate, so a later query with the same phase
   recalls the value exactly — this is why the model is called *phase*-
   associative.
3. **Read.** Probe the notebook with the query (raw complex dot, no
   conjugate on the query), scaled by `d^(−1/2)`.

The read happens *after* the write, so a token can read what it just wrote.

Between layers sits a complex gated channel mix (CGU) with pre-norm residual:
`x = x + cgu_scale · CGU(x)`, then `x = x + pam_scale · PAM(x)`, with
`cgu_scale = 1.0`, `pam_scale = 0.1` (the v11 bare-minimum geometry). Input is
a complex embedding; output is a *tied* complex head (the embedding matrix,
split real/imag, is the score table).

## The recurrence, and why there are two paths

The notebook is a running state: `S_t = γ_t S_{t-1} + u_t`, with
`γ_t = decay_t` and `u_t = value_t ⊗ conj(key_t)`. This is the standard
"multiply-old, add-new" recurrence. Two facts make it cheap.

**Closed form inside a window.** If `a_s = γ_s · … · γ_1` is the product of
decays up to position s, then by induction

```
S_s = a_s · ( S_0 + Σ_{j ≤ s} u_j / a_j )
```

Proof: multiply the step `S_s = γ_s S_{s-1} + u_s` through by `a_s` to get
`a_s S_s = a_{s-1} S_{s-1} + a_s u_s`, and the induction hypothesis supplies
`a_{s-1} S_{s-1}`. (Verified numerically: stepwise vs closed form agree to
8.9e-16 in fp32 on a 50-step random recurrence.) So a window of C tokens is
**one** cumulative product (`a_s`) and **one** cumulative sum (`Σ u_j / a_j`)
— O(C) work, never O(C²), and the notebook at the last position of the window
(`select(window_notebook, over=chunk_time, index=C-1)`) is carried into the
next. Training and prefill process the sequence in windows of `chunk_size`
(256 by default).

**One step per token at decode.** The same recurrence, applied once per new
token on the carried notebook: O(1) per token, independent of context length.

The two paths run *the same algebra*, so they must agree. They do, measured
in `selftest.py::test_parallel_vs_recurrent` on CPU fp32 with a deliberately
small chunk (17 tokens = 7 + 7 + 3 windows):

- max |logit diff| = **2.98e-07**
- max |carried-notebook diff| = **1.86e-08**

**Pre-registered equivalence contract:** the chunked path and the stepwise
path agree on logits and on carried notebooks to ≤ 1e-4 on CPU fp32 (fp32
round-off is ~1e-7; the bar is 100× looser on purpose so the test fails on a
real algebra bug, not on a reordering).

## What was deliberately left out, and why

This is a *lean* model by decision, not by accident. Every item below was in
v7/v11/v13 at some point and was cut because it either measured neutral or
bought a sub-point of PPL at real compute cost.

### Position: RoPE in, learned positions out

**Question:** does the recurrence need positional embeddings at all? The
recurrence alone only knows *recency* — a decaying number. Word order needs a
position carrier.

**Evidence from the old logs** (v7 7pos ablation, WikiText-103, ~100M,
matched B=18 recipe, `logs/v7/exp7pos_*_20260520_124e34e_dirty/`):

| Run | learned pos at input | RoPE on Q/K | Val PPL @10e | vs RoPE-only |
|-----|---------------------|-------------|--------------|--------------|
| 7d-control | no | yes | **26.88** | — |
| 7pos-hybrid | yes | yes | **26.92** | **+0.04** (worse, +0.8M params) |
| 7pos-only | yes | no | **26.72** | −0.16 (confounded: +0.8M params, no RoPE) |

Learned input positions measured **neutral-to-negative**: +0.04 PPL when
added to RoPE, and the "pos-only" win (−0.16) is confounded by the extra
parameters and by removing RoPE — not a clean win. v11 independently rejected
learned input positions.

**Decision:** RoPE on Q/K (a buffer + one complex multiply, zero parameters),
no learned input positions. If a future experiment shows the recurrence
itself is position-blind in a way that hurts generation, that is the first
thing to revisit — with a clean ablation, not a confounded one.

### The v13 machinery: all out

The v13 "selective" PAM had, on top of this recurrence: a GSP protect gate,
a vault (extra protected state), a delta-rule write with an erase gate,
E3 phase routing (state competition, `state_compete`, `phase_proj`),
write-phase addressing, per-channel decay, qk-norm, n-gram features,
`gamma_floor`, and a gate-surprisal auxiliary loss.

**Decision: all of it out.** The sROI rule below is the standing test, and
none of these mechanisms has cleared it in a clean ablation. The lean model
is the floor; any mechanism that comes back must first beat the floor by
more than noise.

### What stays (the v11 bare minimum)

Kept because it is either the model's identity or was measured to matter:

- **Complex** everything (embedding, PAM, head). "Phase-associative" *is*
  the complex conjugate-write / raw-read.
- **CGU** before PAM in each block (~2.4 PPL worth in v7 lean L1).
- **Tied complex head.**
- **RoPE** (above).
- **Pre-norm residual** with `pam_scale = 0.1`.
- **Fused QKV** (one projection, split by name).
- **Chunked closed-form training** + **stepwise decode** (above).

## The sROI rule (standing principle)

> **Reject any mechanism that buys a sub-point (0.xx) of PPL at meaningful
> compute cost.**

"Meaningful" = extra kernels, extra state, extra memory traffic, or extra
parameters that a lean run must then carry forever. Sub-point PPL at that
price is noise, not signal: the old logs show 0.04–0.58 PPL differences
flipping with commit, batch, and logging. A mechanism earns its keep by
clearing the noise floor, not by sitting inside it.

This is why "pure PAM" is not a style choice — it is the only design that
has passed the sROI test so far.

## Layout: named end to end

The whole point of `v13_sempty` is that the code *is* the math. Every axis is
a named `Dim`; layout changes are `.to()` / `.alias()`; matrix products are
`contract` / `outer`; the complex pair is a policy, not a bookkeeping axis.
`check_torch_layout.py` fails the build if anything reaches for `view`,
`permute`, `[..., 0]`, a numeric `dim=`, or escapes to `.raw` / `.data`
outside a declared boundary.

| Axis | Meaning |
|------|---------|
| `batch` | sequence items in the minibatch |
| `time` | token positions |
| `model_dim` | residual / embedding width |
| `heads` | PAM heads |
| `head_feature` | per-head channel width (d) |
| `complex_pair` | last axis of size 2: real then imag |
| `qkv_slot` / `qkv_fused` | the fused Q/K/V packing (slot 0=q, 1=k, 2=v) |
| `head_row` / `head_col` | the two axes of the d×d notebook |
| `chunk_time` | token positions inside one window |
| `real_imag_feature` | `concat(real, imag)` along `model_dim` |

**Declared raw-torch boundaries** (the only places raw torch is legal):

| Site | Why it stays raw |
|------|------------------|
| `complex_ops.build_rope_cache` | position table built once, outside the graph |
| `LM.generate` | sampling loop over raw logits |
| `LM.ce_from_lm` hand-off | chunked CE must not materialize `[N, vocab]` |
| `fused_ce.py` | custom autograd Function (whole module is a boundary) |
| `selftest.py` | compares against plain `F.cross_entropy` |
| `train.py` / `generate.py` | optimiser / dataloader / sampling plumbing |

### torch → sempyt mapping (folded from the old SEMPYT_OPS.md)

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

## Experiment ledger

The ledger is no longer pre-registered: the GPU experiments below were run
(commits `96513c0` A/B evidence, `87f7b42` wikitext run; full logs under
`logs/`). Note the `baseline` preset row was overtaken by events — the real
work went to the param-matched real arm and the complex-vs-real A/B.

| # | Question | Design | Bar to pass | Status |
|---|----------|--------|-------------|--------|
| — | baseline | — | — | **clean baseline, PPL regression vs v13 accepted** |
| 1 | *Does the simple PAM learn anything at all on real data?* | WikiText-103, 1 epoch; arms: complex-384 (100.4M), real-588 `baseline_real_pm` (101.9M, param-matched), real-768 (162M) on tinystories; real-588 on wikitext | finite loss, loss decreases, PPL logged for the record | **PASS** — tinystories 1-epoch A/B done (final-200 nll: complex 3.222 / real_pm 2.981 / real768 2.834); wikitext 1 epoch: val PPL 68.75, train NLL 4.38 @100M tok |
| 2 | *Is the recurrence actually position-blind, and does it matter?* | generation quality probe (rep3/rep4/uniq) | if degenerate, revisit position (clean ablation) | not started — superseded for now by in-loop gen samples + behavioral probes (below) |
| 3 | *At matched params, does the real-arithmetic PAM learn as well as the complex PAM?* | `baseline_real_pm` (101.89M) vs complex `baseline` (100.36M), same loop, tinystories + wikitext | train NLL within ~0.1 at matched tokens | **PASS** — tinystories: real −0.24 nll (real better); wikitext @100M tok: 4.38 vs 4.36 (parity), and real used the conservative lr 5e-5 / T256 vs 3e-4 / T2048 |
| 4 | *Does the real PAM actually use its memory path, or degenerate into a pure MLP?* | per-layer `pam_scale` / `cgu_scale` / realized-retention panels during the wikitext run; then behavioral recall probes (`scripts/run_memory_behavioral.py --model-type v13_sempty`, trials=20, matched to `v11_behavior.json` protocol) | memory scales > init where probes show recall | **SPLIT** — mechanism: PASS (`pam` engages selectively, L11 0.29 vs init 0.1; retention bounded 0.63–0.90; not memory-off). Behavioral recall: FAIL — real_pm mean acc 0.129 ≈ chance (0.125); matched complex sempty (tinystories) 0.150 ≈ chance. At 1 epoch natural text neither arm learns invented-association recall; NOT discriminating real-vs-complex. (Stage-6d's transformer 0.956 used an explicit recall curriculum + 1B tok — not comparable.) |

**Rules for this ledger:**
- One variable per row. No confounded runs.
- A row only "passes" if it clears the sROI rule *and* the pre-registered bar.
- Negative results are written down and closed, same as positive ones.

## Verification status (CPU, 2026-08-29)

Measured with `.venv/bin/python -m v13_sempty.selftest` on CPU fp32:

- `test_param_count_and_state` — **PASS** (tiny preset = 6,606,540 params;
  `rope_cache` correctly *not* in state_dict)
- `test_parallel_vs_recurrent` — **PASS** (2.98e-07 logits, 1.86e-08 states)
- `test_tied_logits` — **PASS** (4.77e-07 vs the manual named score)
- `test_fused_ce` — **PASS** (loss diff 0.0, max grad diff 4.10e-08)
- `test_smoke_loss_decreases` — **PASS** (5.5644 → 5.5405 over 12 steps)
- `test_generate_smoke` — **PASS** (greedy, 4 tokens, in-vocab)

Guard: `.venv/bin/python -m v13_sempty.check_torch_layout` — **clean**
("named end to end outside declared boundaries").

Train smoke: `.venv/bin/python -m v13_sempty.train --preset tiny --dataset
synthetic --steps 8 --device cpu` — loss 5.5556 → 5.5409, finite, decreasing.

## GPU verdict (2026-09-01, wikitext-103, 1 epoch)

The fully-real PAM (`baseline_real_pm`, 101.89M) matches the complex PAM at
matched params: train NLL 4.38 vs 4.36 at the 100M-token anchor, val PPL
68.75 (monotonic, no overfitting), exit clean. Per-layer diagnostics show
both paths learned: `cgu` (transform) ramps 0.55→1.76 with depth, `pam`
(memory) engages selectively in mid/late layers with bounded realized
retention (0.63–0.90; no layer saturates to 1.0). The model did not
degenerate into a memory-off MLP. Details:
`logs/v13_sempty_wikitext_real_20260901.md`; raw cadence:
`logs/ab_real_wikitext.log`; tinystories A/B:
`logs/v13_sempty_ab_generation_20260901.md` + commit `193b459`.

## Speed: fused real arm (2026-09-03) — 5.8k → 66.5k tok/s (11.4x)

The 2026-09-01 run trained at ~5.8k tok/s (B8 T256, grad-ckpt, 21 GB).
Profiling showed the model was not slow, the *formulation* was:
`RealPAMLayer._chunked` materialised the notebook at every position
(`[B*H, w, K, K]`, 472 MB fp32 per layer per window at K=98), and the fp32
fused-CE head was a third of what remained. Fixes, each committed with its
verification (commits `4740d65` → `7343201`):

| change | step B16 T256 | tok/s | peak |
|---|---|---|---|
| baseline (eager notebook, grad-ckpt, fp32 head) | ~700 ms (B8: 5.8k tok/s) | 5.8k | 21 GB |
| chunked linear-attention form, Triton fwd+bwd (`triton_kernels.py`) | 85.6 ms | 48k | 6.6 GB |
| + bf16 head GEMMs, fp32 loss (`fused_ce gemm_dtype`) | 76.3 ms | 53.7k | 6.6 GB |
| + Triton fused linear CE (grads in forward, one row pass) | 61.6 ms | 66.5k | 4.4 GB |
| + PAM kernels at 8 warps (no spills) | ~60 ms | ~67k | 4.4 GB |

- **Math.** `notebook_s . q_s = a_s (S_in . q_s) + sum_{t<=s} (a_s/a_t)(q_s.k_t) v_t`
  (`a_s = prod_{i<=s} r_i`): one `[w, w]` decayed score matrix against V
  plus the carried `[K, K]` state — the same closed form as the complex
  chunked arm, never the per-position notebook. Decode (`_stepwise`) and
  the complex `PAMLayer` are untouched. Log-decay `g = log(r + 1e-6)`, in-tile
  cumsum `G`; backward `dg_i = sum_{s>=i}(q_s.dq_s - k_s.dk_s) + [tile end]
  <S_out, dS_out>`, reverse-cumsum in torch. Kernel tile BT=64, K blocks of
  64 (K<=128), 8 warps; states recomputed in backward (never saved).
- **Parity** (`v13_sempty/pam_kernel_test.py`, oracle anchored to
  `RealPAMLayer._stepwise` at 1.9e-6): torch form and Triton vs oracle,
  forward + all five gradients, fp32 < 5e-5, bf16 < 3e-2 (`ddec` < 5e-2 —
  bf16 cancellation in `q.dq - k.dk`, the identity FLA uses too), T in
  {64,130,256,300,512,1024}, chunk in {7..1024}, D in {64, 98}, carry in/out.
  `selftest.test_real_fused_kernel_parity`: kernel on/off through the whole
  model — logits 3.6e-7, state 6e-7, parameter grads 4.2e-7 rel.
  The harness's original oracle had `outer(k, v)` (transposed) — fixed.
- **CE head.** `_FusedLinearCETriton` (Liger-style): per 4096-row chunk,
  logits GEMM in bf16, one Triton program per row does online logsumexp /
  NLL and overwrites the row with `(mask/denom)(softmax - onehot)`, then
  `grad_h = dL @ W`, `grad_W += dL^T @ h` (fp32 accumulate via `out_dtype`)
  — 3 GEMMs + 1 pass instead of 4 GEMMs + ~10 fp32 passes. Loss/NLL are fp32;
  bf16 vs fp32-truth grad rel-norm 3.8e-3 (the old bf16 torch path: 5.2e-3).
  Validation keeps the fp32 head (`gemm_dtype=torch.float32`), so val
  NLL/PPL stays exact and comparable. Head fwd+bwd N=4096: 32 → 8.8 ms.
- **Not done, measured not worth it:** fusing RoPE into the kernel (RoPE off
  saves only 3 ms of 76); splitting `dqk` into dq/dk kernels (slower: 0.66
  vs 0.56 ms — the extra dA pass costs more than the spills it removes);
  `torch.compile` (sempyt Dim identities trip the recompile limit; out of
  scope). Remaining step (61 ms): model Linears 16 ms and CE GEMMs 6 ms are
  both at tensor-core peak for dim 588; ~20 ms elementwise spread over
  norms/gates/residuals/RoPE; PAM scan 8 ms; AdamW 4.5 ms.
- **Geometry grid** (grad-ckpt off, real-101M): tok/s peaks at 8192
  tok/step — B32 T256 67k / 7.1 GB, B16 T512 68.5k / 7.1 GB — and *falls* for
  larger batches (B64 T256 63k, B96 T256 57k, B48 T512 58k).
- **Run launched** (tmux `sempty_wiki`, `tmp_wikitext_real.sh`): B32 T256
  lr 1e-4 (= 5e-5 * sqrt(4), the one recipe change) warmup 100, `--steps
  14400` = one epoch so the cosine completes (the 09-01 run's horizon was
  200k steps, i.e. constant lr), val every 500 steps (~4M tok, as before),
  fp32 val head. Log `logs/v13_sempty_wikitext_real_7343201_20260903_1449.log`.
  One epoch ≈ 30 min.
- **Result (2026-09-03, commit `7343201`).** Best **val PPL 54.89**
  (val NLL 4.005, step 14000; full 248k-token val set, fp32 head), down from
  the 09-01 run's 68.75 — the single recipe change (B32/lr 1e-4 cosine over a
  finite 14400-step horizon vs B8/lr 5e-5 constant) bought −13.9 PPL.
  avg **64.1k tok/s**, peak 7.6 GB, ~31 min wall, exit 0. End-of-run diag:
  `pam_scale` 0.11–0.28 (memory path engaged, strongest mid/late layers),
  `cgu_scale` 0.86–1.46, realized retention 0.66–0.91 — no layer saturates.
  Train loss 10.94 → 4.16.

  This is **1 epoch at T=256**. It is NOT comparable to the reference bars,
  which are all **10 epochs at T=2048, B=18** (3213 steps/epoch): transformer
  100.3M epoch-1 73.41 → epoch-2 38.77 → epoch-10 **22.69**; v11 E3-K3
  (complex, 100.5M) epoch-1 81.54 → epoch-2 45.59 → epoch-10 **25.77**. Our
  54.89 uses 4.5× more optimizer steps (B32 T256) over an 8× shorter context
  than those epoch-1 numbers, which normally *helps* PPL, so the honest
  apples-to-apples question (10 epochs, T=2048, ~1.18B tok — now ~5 h at 64k
  tok/s, previously ~57 h) is answered by the Phase-1 fair run, not by this
  number. WikiText-only behavioral recall on this checkpoint: pending
  (Phase 1 probe).

## The fair run — apples-to-apples vs transformer & v11 (2026-09-04)

**Result (commit `7af42eb`, log
`logs/v13_sempty_wikitext_real_fair_7af42eb_20260903_1628.log`).**
Real-arm `baseline_real_pm` (101.9M), WikiText-103, **T=2048, B=18, 10 epochs
(32130 steps, 1.184B tok)** — the *exact* geometry of the reference bars.
bf16, fused PAM + fused CE, gradient checkpointing on (5 GB, 48k tok/s, ~6.8 h).

Best **val PPL 23.81** (val NLL 3.170, step 32000; full 248k-token val set,
fp32 head). Train loss 10.93 → 3.12.

| model | type | params | val PPL @10ep |
|---|---|---|---|
| transformer | attention | 100.3M | **22.69** |
| **v13_sempty real** | **O(1) PAM (real)** | 101.9M | **23.81** |
| v11 E3-K3 | O(1) PAM (complex) | 100.5M | 25.77 |

So the real arm **beats the v11 complex PAM baseline by 1.96 PPL** and lands
**within 1.12 PPL of the transformer** — not a strict val-PPL win, but a
competitive novel O(1) model, and the *real* arm beats the *complex* one.

Matched-token trajectory (same tokens/step, so step ≈ token count) — sempty
was ahead of E3-K3 the whole way, roughly tied with the transformer's
per-epoch bars early:

| ~epoch (step) | transformer | v11 E3-K3 | v13_sempty real |
|---|---|---|---|
| 1 (3213)  | 73.41 | 81.54 | ~63 (49.95 @ 4000) |
| 2 (6426)  | 38.77 | 45.59 | 38.48 @ 6000 |
| 3 (9639)  | —     | 35.75 | 29.98 @ 10000 |
| 10 (32130)| 22.69 | 25.77 | **23.81** |

Full sempty val curve (step → ppl): 2000→87.62, 4000→49.95, 6000→38.48,
8000→33.17, 10000→29.98, 12000→28.18, 14000→26.89, 16000→25.84, 18000→25.22,
20000→24.80, 22000→24.35, 24000→24.13, 26000→23.96, 28000→23.85, 30000→23.82,
32000→**23.81** (essentially converged; curve flat after ~26k).

Open question remains **recall**, not PPL (the 1-epoch probe showed real_pm ≈
chance on invented-association recall while the transformer bar used a recall
curriculum). The architecture ladder (A2–A4 coded; A1 tried and removed, see below) and recall-mix
training target exactly this.

## N1 Chrono-PAM — content-modulated rotary retention: 23.14 (2026-09-04)

**Result (commit `7b24e44`, RTX Pro 6000, log
`logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`, ckpt
`checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt`).**
Identical recipe to the 23.81 fair run (real-101M `baseline_real_pm`,
WikiText-103, T=2048, B=18, 10 ep = 32130 steps, 1.184B tok, lr 1e-4 cosine,
seed 42, bf16, fused PAM+CE, grad-ckpt on) with **one** change: `--chrono`.
Best **val PPL 23.14** (val NLL 3.1415, step 32000). Train loss 10.94 → 3.06
(epoch-10 mean 3.047 / ppl 21.06). 82.8k tok/s avg on the 6000 (4 h wall);
peak 8.9 GB. +56,544 params (one zero-init `dim → n_heads` warp projection per
layer; 101,945,996 total, +0.06%).

| model | params | val PPL @10ep |
|---|---|---|
| transformer | 100.3M | **22.69** |
| **v13_sempty real + Chrono (N1)** | 101.9M | **23.14** |
| v13_sempty real (fair baseline) | 101.9M | 23.81 |
| v11 E3-K3 complex | 100.5M | 25.77 |

The gap to the transformer is now **0.45 PPL** (was 1.12); v11 is beaten by 2.63.

**The math.** A complex *rotating* retention `gamma_t = r_t e^{i theta_t}` on
the outer-product notebook has closed form `S_s = sum_t (a_s/a_t)
e^{i(Phi_s-Phi_t)} v_t k_t^*`, `Phi = cumsum(theta)`. The phase factor is
absorbed by rotating `q_s -> e^{i Phi_s} q_s`, `k_t -> e^{i Phi_t} k_t` — which
with a *fixed* `theta` is exactly RoPE. Chrono makes `theta_t` input-dependent:
per head, `g_t = exp(clamp(W x_t, ±3))`, clock `tau_t = sum_{j<t} g_j`, phase
`phi_t = tau_t (x) inv_freq`. So **learned rotating retention == a
content-modulated RoPE clock**, folded entirely into q/k; the fused
magnitude-retention scan is untouched (speed gate on the 4090: 65.9k vs 66.6k
tok/s, equal). `W` is zero-init ⇒ bit-exact RoPE at step 0
(`test_chrono_rotary_parity`). CoPE-style, on an associative memory.

**Matched-step val curve (same tokens/step), chrono vs baseline — ahead at
every checkpoint by 0.6–1.1 PPL, never a crossing:**

| step | 2k | 4k | 6k | 8k | 10k | 12k | 14k | 16k | 18k | 20k | 22k | 24k | 26k | 28k | 30k | 32k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 87.62 | 49.95 | 38.48 | 33.17 | 29.98 | 28.18 | 26.89 | 25.84 | 25.22 | 24.80 | 24.35 | 24.13 | 23.96 | 23.85 | 23.82 | 23.81 |
| chrono | 86.52 | 49.06 | 37.63 | 32.48 | 29.37 | 27.45 | 26.26 | 25.13 | 24.54 | 24.09 | 23.64 | 23.45 | 23.28 | 23.17 | 23.15 | 23.14 |
| Δ | −1.10 | −0.89 | −0.85 | −0.69 | −0.61 | −0.73 | −0.63 | −0.71 | −0.68 | −0.71 | −0.71 | −0.68 | −0.68 | −0.68 | −0.67 | −0.67 |

**Is the clock actually used (or is it RoPE with noise)?** Warp statistics on
8 val chunks (`g` per token/head, all 16 layers): **88–98 % of tokens have
|log g| > 0.1**; the median clock speed per layer sits at 0.4–1.7×, the 5–95 %
band spans ~0.2× to ~3–4× (L15: up to the 20× clamp). Per-head means range
from **0.17×** (slow "long-memory" clocks: L09 h2, L13 h1/h4/h5) to **15.8×**
(L15 h1, saturating). The model learned genuinely different per-head time
scales and modulates them by content — not a small perturbation of RoPE.
Other diagnostics unchanged vs baseline: `pam_scale` 0.11–0.31, `cgu_scale`
0.71–1.68, realized retention 0.75–0.91 (memory path still soft → N4 gate).

**sROI verdict: KEEP.** −0.67 PPL is sub-point, but the cost is ~nil (+0.06 %
params, zero kernel change, equal tok/s) and the gain is consistent at all 16
val points and above the 0.04–0.58 commit/batch noise band recorded above.
Chrono becomes the new real-arm reference (23.14); subsequent one-variable
rungs compare against it. Decode for chrono landed after the run (the
`(notebook, clock)` state carry; `test_chrono_parallel_vs_recurrent` 6.6e-7
logits vs chunked) — the in-run `[gen @ 8000/16000/24000/32000] failed` lines
are from before that and are harmless (val/ckpt use the chunked path).

## A1 short conv on qkv — 23.49: FAIL, removed (2026-09-04)

**Rung.** Chrono reference (23.14) + a depthwise causal `Conv1d(k=4)` over the
time axis on the fused qkv, added residually (`qkv + silu(conv(qkv))`), zero-init
so it was the identity at step 0 (Based / Gated-DeltaNet "short conv"
convention; +141,120 params → 102.09M). One variable vs 23.14; same recipe
(B18 T2048 10 ep, seed 42, grad-ckpt off on the RTX Pro 6000). Commit `817aa35`,
log `logs/v13_sempty_wikitext_chrono_a1conv_fair_817aa35_20260904_1057.log`.

**Math / why it was plausible.** The PAM write is `v_t ⊗ k_t` from the *current*
token only; a k-tap causal conv lets each key/value/query see a 4-token local
window before it is written, so the notebook can store short phrases as keys
rather than single tokens (the argument that made it standard in linear-
attention models).

**Result: val PPL 23.49 (best, step 32000) — +0.35 vs the 23.14 reference; and
−17 % tok/s (85k vs 102k, plain-torch conv over `[B,T,3·inner]` × 16 layers).**
Behind at every val point after 2k: 4k 50.30 (49.06), 8k 33.26 (32.48), 16k
25.53 (25.13), 24k 23.82 (23.45), 32k 23.49 (23.14). Train loss reached 2.98
(vs 3.06) — it *fit the train set better and generalised worse*: the extra
local mixing on top of Chrono's content clock is redundant capacity that
overfits WikiText at 10 epochs. `pam_scale` (0.15–0.34) and retention
(0.68–0.88) unchanged vs chrono.

**sROI verdict: FAIL on both axes (worse PPL, meaningful compute cost).** Per
the ablation rule the code path is removed (`cfg.short_conv`, `_short_conv`,
`--short_conv`, the layout-guard boundary); this section is the record. Old
checkpoints with a `short_conv` key in their saved config still load
(`_config_from_ckpt` filters unknown fields). The A1 checkpoint dir
`checkpoints_v13_sempty/wikitext_chrono_a1conv_fair_817aa35/` (2.4 GB) is dead
weight and can be deleted. Note: the wrapper line in the log says `exit=127`;
Python exited 0 (`Training complete`, `latest.pt` saved) — the 127 came from
editing `tmp_wikitext_fair.sh` while bash was still executing it (see
SCRATCHPAD pitfalls).

## N4 read-out gate — 22.96: KEEP, new reference (2026-09-04)

**Rung.** Chrono reference (23.14) + a per-token, per-head gate on the memory
read before `o_proj`: `read_h ← read_h · silu(W_g x_t + b_g)`, `W_g` zero-init,
`b_g = silu⁻¹(1) = 1.2785` ⇒ exactly 1 at step 0 (bit-parity with chrono,
`test_out_gate_parity_and_decode`). `dim → n_heads` per layer, +56,544 params
(102.00M). Elementwise; the fused scan and the `(notebook, clock)` decode carry
are untouched. Speed gate −2.8 % tok/s (ABAB, B8 T2048). One variable vs
23.14, same recipe (B18 T2048 10 ep, seed 42, grad-ckpt off, RTX Pro 6000,
99k tok/s, 30.5 GB, 3 h 20 m). Commit `3c7b9b9` (run `2537782`), log
`logs/v13_sempty_wikitext_chrono_n4gate_fair_2537782_20260904_1517.log`,
ckpt `checkpoints_v13_sempty/wikitext_chrono_n4gate_fair_2537782/best_model.pt`.

**Why.** The read reached the residual only through the block's *static*
`pam_scale`, which sat at 0.11–0.31 after 10 epochs — the model could not
decide per token how much to trust the notebook. N4 gives it that knob.

**Result: val PPL 22.96** (best, step 32000; NLL 3.1336). Train loss 10.95 →
2.92. Ahead of chrono at all 16 val points; the lead is largest early (faster
learning) and settles at −0.18:

| step | 2k | 4k | 8k | 12k | 16k | 20k | 24k | 28k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| chrono | 86.52 | 49.06 | 32.48 | 27.45 | 25.13 | 24.09 | 23.45 | 23.17 | 23.14 |
| +gate | 84.24 | 47.71 | 31.74 | 27.11 | 24.93 | 23.88 | 23.26 | 22.99 | **22.96** |
| Δ | −2.28 | −1.35 | −0.74 | −0.34 | −0.20 | −0.21 | −0.19 | −0.18 | −0.18 |

| model | params | val PPL @10ep |
|---|---|---|
| transformer | 100.3M | **22.69** |
| **v13_sempty real + Chrono + gate (N1+N4)** | 102.0M | **22.96** |
| v13_sempty real + Chrono (N1) | 101.9M | 23.14 |
| v13_sempty real (fair baseline) | 101.9M | 23.81 |
| v11 E3-K3 complex | 100.5M | 25.77 |

Gap to the transformer: **0.27 PPL** (was 1.12 → 0.45 → 0.27).

**Inside (16 val chunks, all layers).** The gate is content-dependent, not a
re-learned constant: within-head std over tokens 0.89 on means of 1.6–2.7;
5–95 % band ≈0.2 → 5.5; 30–70 % of tokens gate > 2. **The model wanted about 2×
more memory than the static scale allowed**: effective read scale
`pam_scale × mean gate` = 0.18–0.61 vs chrono's 0.11–0.31, while the static
`pam_scale` fell to 0.11–0.24 as the gate took over. No position effect (mean
gate identical for pos < 64 and ≥ 64). Gate correlates +0.3…+0.7 with
Chrono's `log g` in layers 4–15 — where the clock runs fast the read is
opened (a "topic shift ⇒ consult the notebook" policy). Frequent/function
tokens read memory more (L10: 2.16) than rare content tokens (1.52), i.e. the
notebook is used for context-conditioned predictions, the CGU/residual for
lexical ones. Retention unchanged (0.70–0.91); the warp statistics match the
chrono run.

**sROI verdict: KEEP.** Sub-point gain (−0.18) but at ~nil cost (+0.06 %
params, −2.8 % tok/s, no kernel change), consistent at every val point, with
a verified mechanism. `CHRONO=1 OUT_GATE=1` is the reference for all further
rungs (22.96). Sample at 32k (T=0.8): "In 1923, the University of Southern
California created a university for the students . At that time , there was
no official school in the state …" — fluent WikiText register.

**Where this leaves the PPL program.** Three rungs on the 23.81 baseline:
−0.67 (N1), +0.35 (A1, removed), −0.18 (N4). Returns are shrinking and the
remaining 0.27 to the transformer is one rung's worth; WikiText-103 at 100M
is close to what this architecture will show. The next information is not
another 0.2 PPL — it is whether the architecture holds on diverse data at
more tokens, and whether it can be trained to *recall*. Next: scale the data.

## Phase 3a: mix-3B pretrain — PPL fine, recall horizon ~200 tokens (2026-09-05)

**Run (commit `bbc12e9`, RTX Pro 6000, log
`logs/v13_sempty_mix3b_chrono_gate_bbc12e9_20260904_1915.log`, ckpts
`checkpoints_v13_sempty/mix3b_chrono_gate_bbc12e9/{best_model,latest,step_0X0000}.pt`).**
real-102M chrono+gate (the 22.96 arch), live stream dclm-edu .45 / fineweb-edu
.42 / smoltalk2-Mid ChatML .10 / synthetic recall .03, web-only first 300M tok,
3.0B tokens, B18 T2048 (81,380 steps), lr 2e-4 warmup 1000 cosine, dropout 0,
chat vocab 50261, grad-ckpt off. 85k tok/s live, 30.5 GB, 15.5 h, exit 0.

**Result.** Holdout val PPL **25.73** (best at 80k; 104.6 → 44.0 @8k → 31.0
@30k → 27.1 @56k → 25.7), still `*best*` at every 2k-step val point — data,
not capacity, is the limiter at this size. WikiText anchor 55.2 (single pass,
different distribution; not comparable with the 22.96 10-epoch number). Train
loss 10.96 → 3.22. Samples fluent web/encyclopedic register.

**Trajectories (`[diag]`, every 2k steps).** `cgu_scale` L0 0.95 → 0.19, L15
1.18 → 1.73; `pam_scale` mean 0.117 → 0.047, L0–7 at 0.006–0.03 by the end;
`dt_bias` −3.98 → −3.92 (never left init −4.0); realized retention 0.61–0.90
(mean 0.78); grad norms 0.1–0.6; weight norms 57 → 81 (L0) / 60 → 88 (L15).
**Lesson: `pam_scale` is not a usage meter.** Ablation on the checkpoint
(holdout val, NLL): memory off in all layers **+2.93** (ppl 25.7 → 482), L0–7
off +2.86, L8–11 off +0.63, L12–15 off +1.07; CGU L8–11 off +0.55 for
calibration. The tiny scalars are normalisation that moved into the PAM
weights (wnorm ↑); PAM is the block's only token-mixing path and carries the
model. Gate (N4) on this ckpt: means 1.5–2.5, std 0.9–1.8, effective read
`pam×gate` 0.01 (L0) → 0.35 (L13).

**Behavioral recall (`scripts/run_memory_behavioral.py`, 8-candidate
contrastive, 20 trials/cell,
`logs/memory_probes/v13_sempty_mix3b_chrono_gate_bbc12e9_behavioral.json`).**
Mean **0.354** vs chance 0.125 (1-epoch WikiText probe was 0.129). The shape is
the finding:

| ctx | a1 pos0 | a1 pos½ | a1 pos1 | a4 (mean) | a8 (mean) |
|---|---|---|---|---|---|
| 128 | **1.00** | **1.00** | **1.00** | 0.40 | 0.22 |
| 512 | 0.10 | 0.10 | 0.85 | 0.23 | 0.22 |
| 2048 | 0.10 | 0.10 | 0.55 | 0.27 | 0.25 |

Horizon sweep (a1, binding at pos 0/½): ctx 128 **1.00** → 192 0.75 → 256
0.35 → 320 0.33 → 384 0.17 → 512 **0.10** (chance). Half-life ≈ 200 tokens.
Across milestones (a1@ctx128 / a1@ctx512): step 10k 0.75 / 0.35 → 40k 0.90 /
0.10 → 80k 1.00 / 0.10 — **training on web text shortens the horizon**; more
tokens make it worse, not better. 8-way stays 0.2–0.35 at every length
(interference/read routing, the previously identified gap).

**Mechanism.** Retention is `exp(-softplus(W x + dt_bias))` with `dt_bias`
stuck at −3.9 ⇒ 0.982/token at logit 0 ⇒ the state is at 8 % after 128
tokens before any input-dependent forgetting (realized mean 0.78). Web PPL is
dominated by local context and rewards a fast-forgetting cache; a 3 % recall
slice cannot pull one shared timescale the other way. The memory is a
~200-token associative cache, not long-term storage. This — not PPL — is what
separates a matrix memory from a Mamba-style SSM, so it is the program now.

**Complex arm reconsidered: no.** Both arms share the same decay
(`model.py` `_decay`, L301 vs L705); phase gives key orthogonality (an
interference lever for the 8-way rows), not retention. Chrono already imported
phase-keyed addressing into the real arm at half the cost with a 2.6-PPL lead
(23.14 vs 25.77). Revisit phase-coded keys only if, after the horizon is
fixed, 8-way interference is the limiter and A3 delta does not solve it.

**Retention ladder (one variable per run on the mix recipe at 1B tokens;
primary metric = horizon curve incl. ctx 4096/8192 beyond the training window,
guard = holdout PPL):**
1. **R1 `--dt_spread 8`**: head h inits at −4 − 8·h/5 ⇒
   half-lives 38 / 190 / 940 / 4.6k / 23k / 113k tokens. Zero params, zero
   cost (98.9k tok/s = ref), decode intact. **Run at T=8192 → FAIL, removed
   (see "L1 / R1 dt-spread" below).**
2. **R2 A2b vault** (`--n_states 2 --vault`, coded): state 0 retention pinned
   to 1 with a protect gate. −21 % tok/s, 36 GB. No decode yet (`GEN_EVERY=0`).
3. **R3 A3 delta** (`--delta`, coded): overwrite-only forgetting, unit keys.
   −43 % tok/s (torch WY path), 56 GB. No decode yet. Needs a Triton path if
   it wins.
4. Novel candidate if R1 moves the horizon but not enough: **clocked
   retention** — decay in Chrono's content time, `ret = exp(-λ_h · g_t)`, so a
   stable topic ages slowly and a shift forgets; one λ per head, zero cost.

**Long-context readiness (measured 2026-09-05, synthetic, chrono+gate+spread):**
T=8192 B=4 → 118k tok/s, 26 GB; T=32768 B=1 → 98.6k tok/s, 26 GB — identical
to T=2048. The scan is linear; 8K/32K is a data + curriculum stage, not a
kernel stage (see SCRATCHPAD "Stage L").

## L1 / R1 dt-spread at T=8192 — horizon flat, FAIL, removed (2026-09-05)

**Run.** `mix1b_8k_r1_dtspread8`, commit `18ed358`, RTX Pro 6000, tmux
`sempty_l1`. `baseline_real_pm` (102M) + chrono + out_gate + `dt_spread 8`
(head h init `dt_bias = −4 − 8·h/5`, half-lives 38 → 113k tok), **T=8192
B=8** (65,536 tok/step), 1.0B tokens = 15,258 steps, lr 2e-4 cosine, warmup
500, dropout 0, `GRAD_CKPT=0`. Data: token-weighted blend `dclm .36 /
fineweb_long .20 / pg19 .22 / smoltalk2_mid .10 / recall .04 / recall_long
.08`, web-only for the first 100M. 79–81k tok/s, 52 GB, 3.9 h. Log
`logs/v13_sempty_mix1b_8k_r1_dtspread8_18ed358_20260905_0956.log`; ckpt
`checkpoints_v13_sempty/mix1b_8k_r1_dtspread8_18ed358/best_model.pt`; probe
`logs/memory_probes/v13_sempty_mix1b_8k_r1_dtspread8_18ed358_behavioral.json`.
(A first launch on `b07e384` was killed at 3.5k steps: the blend drew one
*document* per pick so the stream was ~95 % PG-19 — fixed in `18ed358`, see
`v7/data.py` `_blend_interleave_text_iters`.)

**PPL (guard).** Holdout val 40.26 at 1B (mix-3B at 1B tokens / 27k steps:
32.4; end 25.73). WikiText 91.7 (mix-3B @1B: 71). Worse, but confounded three
ways: 56 % web vs 87 %, 15k optimizer steps vs 27k, and 8K windows of books
vs 2K windows of web. Not the verdict metric.

**Horizon (verdict).** Behavioral recall accuracy, mean over positions
0/0.5/1, 20 trials each; a1/a4/a8 = 1/4/8 invented bindings in context:

| ctx | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| L1 a1 | 0.63 | 0.37 | 0.20 | 0.25 | 0.18 | 0.22 | 0.23 |
| L1 a4 | 0.38 | 0.25 | 0.20 | 0.17 | 0.20 | 0.23 | 0.20 |
| L1 a8 | 0.10 | 0.07 | 0.10 | 0.10 | 0.07 | 0.03 | 0.02 |
| mix-3B a1 | 1.00 | – | 0.35 | – | 0.25 | – | – |
| mix-3B a4 | 0.40 | – | 0.23 | – | 0.27 | – | – |
| mix-3B a8 | 0.22 | – | 0.22 | – | 0.25 | – | – |

Flat ~0.2 from 512 to 8192 for both models; L1 is *worse* at 128 (0.63 vs
1.00) and a8 is at/below chance everywhere. Training in 8K windows with 12 %
recall curriculum up to 5k-token gaps and slow heads with 113k-token
half-lives did not move the horizon at all. Generation confirms it in one
line: prompt "…my favourite colour is teal. Later … I said my favourite colour
is" → "yellow" (30 tokens back).

**Internals (`[diag]`).** `dtbias/head(layer-mean) = −3.95 −5.56 −7.15 −8.74
−10.32 −11.92` at step 14k = init to 2 decimals: the ladder was preserved but
**never learned** (a 6-number bias with lr 2e-4 barely moves; same as the
−3.9 stuck bias in mix-3B). Realized retention per layer 0.83–0.99 (mix-3B
0.61–0.90) — the state *does* persist longer. `pam_scale` 0.03–0.18,
`cgu_scale` 0.66–1.64 (mix-3B: 0.01–0.17 / 0.19–1.73) — same shape.

**Reading.** Retention was the hypothesis (mix-3B: "one shared timescale
cannot serve local PPL and long recall"). It is refuted as the *bottleneck*:
with information demonstrably held for thousands of tokens, the model still
cannot retrieve a keyed binding at 512+ and cannot separate 8 bindings at
128. The limiter is the **read/write path** — the k→v binding written into
`S += k vᵀ` is not recoverable by `q S` once other tokens have been added,
i.e. interference / non-orthogonal keys, exactly the oracle "READ-SIDE
routing gap" from 2026-08-26. This is what the delta rule (A3: erase the old
value under key k before writing the new one, unit keys) is for; the vault
(R2) only protects, it does not de-interfere.

**Decision.** R1 FAIL on its sole metric → `dt_bias_spread` removed from
`config.py`, `model.py`, `train.py`, `tmp_pretrain_mix.sh` (this commit;
`selftest` 17/17). Old ckpts still load (`dt_bias` is in the state_dict).
The recall_long / pg19 / fineweb_long sources and the token-weighted blend
stay — Stage L needs them once retrieval works. Next: R3 delta, but proven
first on a **minutes-long synthetic recall micro-bench** (train on the recall
curriculum only, score the probe at 512–8K) before any 1B-token run — the
1B runs measure PPL well and horizon badly, and we have now spent two of them
learning that.

**Scaling call (asked 2026-09-05).** Neither data nor params (305M) yet: on
PPL the 100M model is 0.27 from the transformer, so scaling would buy PPL we
already have; on recall a 100M transformer scores ~1.0 on every column of
the table above and we score 0.2. Scale after the mechanism retrieves.

## A3 delta rule — recall micro-bench: a1 0.46→1.00, multi-way still stuck (2026-09-05)

**Why a micro-bench.** mix-3B and L1 each cost hours and measured PPL well but
the horizon badly. The tool `v13_sempty/tmp/recall_microbench.py` (throwaway)
trains a SMALL real-arm PAM (dim 384, 6 heads, head_dim 64, 4 layers, 27M) on
the pure task — invented-association docs from `memory_probes.build_example`
("Memory record N: K means V" × assoc, filler, "query: K means" → V), lengths
bucketed per step (no padding), associations 1–8 — and scores the behavioral
probe at ctx 128–8192. Two arms on byte-identical data. **Answer-only loss
mask** (CE on the queried value token only): the first pass used full-sequence
CE, which is ≥99.9 % filler/record-copy, so train ppl hit 1.0 while the read
path got ~no gradient and both arms sat at chance — a lesson in itself. 1500
steps × B16, 32 trials, ~100–150 s/arm on the 6000.

**Result (accuracy; chance ≈0.12, 96 samples/cell).**

| arm | metric | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|
| chrono+gate | a1 | 0.46 | 0.45 | 0.46 | 0.46 | 0.45 | 0.45 | 0.46 |
| **+delta** | **a1** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| chrono+gate | a8 | 0.20 | 0.19 | 0.20 | 0.20 | 0.21 | 0.21 | 0.23 |
| +delta | a8 | 0.19 | 0.18 | 0.19 | 0.19 | 0.15 | 0.11 | 0.11 |

**Reading.** The delta rule fixes exactly the L1-identified failure. Baseline
additive memory `S += k vᵀ` accumulates a write from EVERY token (filler
included), burying the one real binding → single-binding recall is a flat 0.46
regardless of distance. Delta's erase-before-write (`S ← gS(I−bₑ k kᵀ)+b_w v kᵀ`,
unit keys) removes the stale mass, so `q·S` recovers one binding **perfectly and
length-invariantly, extrapolating past the 4096 training window to 8192** — an
O(1)-state property a transformer needs an O(T) KV cache for. This is the
"novel / performative / O(1) / not-transformer" behaviour we want, on the clean
case.

**What delta does NOT fix: multi-way interference.** a4/a8 stay at/near baseline
and a8 *droops* at long ctx. Hypothesis: 8 random unit keys in K=64 cross-talk
≈ 7/√64 ≈ 0.88 ≈ signal, i.e. an orthogonality limit — so sweep head_dim.

**head_dim sweep (delta, dim=6·K, same 1500×B16), REFUTES the capacity story:**

| arm | a4 (mean ctx) | a8 (mean ctx) |
|---|---|---|
| +delta @hd64 | 0.26 | 0.16 |
| +delta @hd96 | 0.22 | 0.12 |
| +delta @hd128 | 0.31 | 0.10 |

a1 = 1.00 at every head_dim (the delta win is robust); a8 does NOT climb with K
(flat at chance, hd128 slightly worse). So multi-way failure is not capacity or
key-orthogonality — it is the **linear `q·S` read**: with several bindings
co-resident, `q·S = Σ_i (q·k_i) v_i` is a weighted blend, and one learned query
cannot both hit its key and null the other 7. Increasing K gives more room but
the model still cannot LEARN keys/queries selective enough — a mechanism limit,
not a width limit. This is the next "better maths" target (candidates: a
nonlinear / iterative read; multi-state routing `n_states` so co-residency
drops; product-key selection — NOT softmax-over-T, which would be transformer
O(T)).

**Decision.** Delta is a **KEEP on the read path** (a1 0.46→1.00, length-
invariant, extrapolating past the train window — a qualitative O(1) win, not the
marginal PPL deltas the ladder chased), so it is NOT removed like R1 — but **no
1B run yet**: real-text recall is gated by the multi-way case, unsolved and
proven not to be a head_dim knob. Cost: ~1.4× slower (285k→203k tok/s, torch WY
path), no decode path (probe uses parallel prefill, fine for eval). Sequence:
solve multi-way on the micro-bench (minutes each), THEN one 1B run at 8K with
the winner (delta + multi-way fix), THEN scale. Tool + result logs:
`v13_sempty/tmp/recall_microbench.py`,
`logs/v13_sempty_recall_microbench_hdsweep_543b897_20260905_1503.log`.

## A2r content-routed delta — multi-way attempt #1: soft routing no gain (2026-09-05)

**Idea (novel, stays O(1) / non-attention).** S fixed matrix memories per head
(S=4,8). One shared per-head centroid table `R[H,K,S]` routes each WRITE by its
unit key and each READ by its unit query: `rw=softmax(k·R/τ)`, `rq=softmax(q·R/τ)`;
write gates for state s scaled by `rw[..,s]`, read from s scaled by `rq[..,s]`,
reads summed. Matched pairs (q≈k, already perfect under delta) route together,
so co-resident keys should spread across states → fewer keys per state → the
linear `q·S` read sees fewer competitors. This is a mixture of *content-addressed
associative memories* with fixed small S — not attention over T (no O(T)), not
MoE-of-FFNs. Code: `model.py::_chunked_delta_routed`, `cfg.state_route/route_temp`.

**Result (τ=1.0, same 1500×B16 micro-bench).**

| arm | a1 (all ctx) | a4 (mean) | a8 (mean) | tok/s |
|---|---|---|---|---|
| chrono+gate | 0.46 | 0.29 | 0.20 | 285k |
| +delta | 1.00 | 0.26 | 0.16 | 202k |
| +delta+route4 | 1.00 | 0.23 | 0.10 | ~90k |
| +delta+route8 | 1.00 | 0.25 | 0.10 | ~55k |

a1 stays perfect; **a8 does NOT improve (≈chance, slightly worse), a4 flat**, at
S× the cost. Diagnosis: uniform routing is mathematically a single delta
(all S states identical), and the near-chance result means the router never
specialized off its small init.

**Sharpened routing (S=8, τ∈{0.5,0.2}) — still no win.** Forcing the router off
uniform lifts a8 only back to delta's own level: route8 τ0.2 a8 = 0.16 0.16 0.14
0.15 0.19 0.20 0.20 (mean ~0.17) vs plain delta ~0.16 and chance 0.12 — i.e.
routing at best *matches* single delta, at 8× cost, and a4 stays ~0.18. So it is
not that routing failed to specialize; even specialized, partitioning keys
across states does not make the per-state read selective enough. Multi-way is
not a memory-structure problem.

**Verdict: A2r routing FAIL → REMOVED** (`state_route`/`route_temp`,
`_chunked_delta_routed` deleted; AGENTS rule). Delta (single state) stays.

**What the three failures (head_dim, soft route, sharp route) jointly say.** The
limiter is not capacity, not orthogonalisable room, not state partitioning — it
is that the model does not LEARN keys/queries mutually selective enough for many
co-resident bindings. The implemented write `S←gS(I−bₑkkᵀ)+b_w vkᵀ` erases the
*written* key's own component but adds `b_w v kᵀ` (not the prediction error
`b_w(v−Sk)kᵀ`); with non-orthogonal keys, writing key j partially corrupts key
i's binding. The canonical error-correcting DeltaNet/Widrow-Hoff write
`S += β(v − Sk)kᵀ` is exactly online least-squares and *does* separate
overlapping keys over the sequence. That — a small, principled change to the
EXISTING delta, still O(1) and non-attention — is the next multi-way candidate,
NOT more memories. (Explicitly avoided: high-β modern-Hopfield / softmax-over-
stored-items reads, which are attention over T in disguise.)

**SUPERSEDED the same evening by the positive-control run below: the DeltaNet
write was NOT run.** The bench turned out to have no ceiling — see next section.

## Positive control — transformer fails multi-way at 1500 steps; at 4× budget PAM-delta hits 1.00 on a1/a4/a8 to 8192, transformer still at chance (2026-09-05)

**Why.** After three failed multi-way attempts (head_dim, soft route, sharp
route) the user asked whether we were "catching the wrong error". Audit: every
multi-way verdict had compared PAM arms against each other and against an
*assumed* transformer ceiling of 1.0. No positive control had ever been run on
the micro-bench, and the only external control on disk
(`logs/memory_probes/publication/gpu/mamba_behavior.json`, Mamba-130M, ~300B
training tokens) scores a8 0.82 @128 → 0.38 @2048 — i.e. multi-way recall IS
achievable by a fixed-state model, but that number came from ~300× our budget.

**Setup.** `v13_sempty/tmp/recall_microbench.py` gained (a) an `xf` arm — the
v6 GPT-2-style causal transformer (SDPA flash, learned abs. positions), d 384 /
6 heads / d_ff 1536, 4 layers, 29.5M params, trained on the **byte-identical**
pre-drawn schedule and answer-only loss as the PAM arms; (b) a `:L<n>` depth
override. Run: 1500 steps × B16 (~24M tokens, 24K supervised answers), 32
trials. Log `logs/v13_sempty_recall_microbench_xfctrl_depth_36a8cac_20260905_1659.log`.

**Result (accuracy, chance ~0.12).**

```
                          ctx 128   256   512  1024  2048  4096  8192
transformer(ctrl) L4  a1     1.00  1.00  1.00  1.00  1.00  1.00  0.66
                      a4     0.28  0.26  0.34  0.25  0.31  0.30  0.28
                      a8     0.17  0.12  0.12  0.15  0.14  0.22  0.09
delta L4              a1     1.00  1.00  1.00  1.00  1.00  1.00  1.00
                      a4     0.25  0.28  0.29  0.29  0.28  0.23  0.18
                      a8     0.19  0.18  0.19  0.19  0.15  0.11  0.11
delta L8              a8     0.10  0.11  0.08  0.10  0.11  0.12  0.15
delta L12             a8     0.15  0.14  0.15  0.11  0.12  0.15  0.10
```

Training curves are also indistinguishable: answer-only loss at step
1000/1500 = transformer 1.21/1.93, delta-L4 1.15/1.62 (same L=256 batches).
Transformer arm: 26 s (1.15M tok/s); delta L4 ~2.5 min, L12 6 min.

**First reading (1500 steps).** The architecture with a perfect O(T) memory
scores at chance on a8 and ~0.3 on a4 on the same data at the same budget, so
the 1500-step bench had **no ceiling** and every multi-way verdict taken on it
(head_dim sweep, A2r soft/sharp routing) was *inconclusive*, not a FAIL of the
read path. Also: the plateau value is diagnostic — a model that knows the SET
of values in context but cannot match keys sits at CE = mean(0, ln2, ln4, ln6,
ln8) ≈ 1.19 over the training assoc mix; both arms plateaued at 1.3–1.9.

### 4× budget: PAM-delta SOLVES multi-way; the transformer control does not

`logs/v13_sempty_recall_microbench_xfctrl_6k_36a8cac_20260905_1714.log` —
6000 steps × B16 (~96M tokens), everything else identical.

```
                          ctx 128   256   512  1024  2048  4096  8192
transformer(ctrl) L4  a1     1.00  1.00  1.00  1.00  1.00  1.00  1.00
                      a4     0.33  0.23  0.33  0.25  0.32  0.33  0.27
                      a8     0.14  0.12  0.11  0.12  0.12  0.10  0.14
PAM chrono+gate+delta a1     1.00  1.00  1.00  1.00  1.00  1.00  1.00
  L4, 27.2M params    a4     1.00  1.00  1.00  1.00  1.00  1.00  1.00
                      a8     1.00  1.00  1.00  1.00  1.00  1.00  1.00
```

PAM-delta's answer loss: 1.23 @1000 → 0.79 @2000 → **0.0013 @3000 → 0.0000
from 3500 on, at every train length** (256–2048). A phase transition between
steps 2000 and 3000 (~40M tokens): the model learns mutually selective keys
and thereafter reads any one of 8 co-resident bindings exactly, **including at
8192 = 2× its longest training window, from a fixed 6 × 64×64 state per
layer.** The transformer's loss stays 1.2–1.7 to step 6000. Two further
transformer-only checks gave it the intermediate LM signal that induction
circuits normally form from (`--aux_weight` 0.1 and 1.0, 3000 steps;
`logs/v13_sempty_recall_microbench_xf_aux0p1_…1720.log`,
`…_xf_aux1p0_…1720.log`): a4 0.24–0.31, a8 0.03–0.19 — still chance.

*Caveat on the control:* the v6 transformer uses learned absolute positions
and records sit at random offsets, so it must learn purely content-based
matching — harder than with RoPE. A RoPE control would likely learn the task
at some budget; transformers obviously *can* do this. The fair statement is
the one the data supports: **on byte-identical data, optimiser and budget,
PAM-delta learns exact 8-way key→value recall by ~40M tokens, where a same-size
transformer has not by 96M, and PAM extrapolates to 2× the train window.**

**What this means.**

1. **Multi-way recall was never a PAM read-path deficit.** The linear `q·S`
   read with delta erase-before-write recovers one of eight co-resident
   bindings perfectly once the keys are learned. All of yesterday's negative
   conclusions about the read path (head_dim, routing, the proposed DeltaNet
   write) are withdrawn; the removals stand because plain delta suffices.
2. **The 1B-token pretrains (mix-3B, L1) failed a8 for the same reason the
   1500-step bench did: signal budget, not mechanism.** Full-sequence LM loss
   gives the answer token ~1/T of the gradient, and recall docs were a small
   fraction of the mix. The bench needed ~40M tokens of *concentrated* answer
   signal; the pretrains supplied a tiny fraction of that. This is a data-
   recipe fix (recall/needle docs with answer-emphasised loss), not maths.
3. **Delta (A3) is the biggest single win in the program**: base 0.46 →
   delta 1.00 on a1, and delta 1.00 on a4/a8 at 4× budget. KEEP, default on.
4. **Depth is not the lever** (L4 = L8 = L12 at 1500 steps); budget is.
5. **The PAM differentiator is real and measured**: O(1) state, exact 8-way
   recall, length extrapolation the abs-pos transformer cannot do (a1 0.66 at
   8192 in the 1500-step run, 0.77 in aux-1.0).

**Decision.** Multi-way is SOLVED on the bench; no more read-path maths. The
"no 1B run until multi-way is cracked" gate is lifted. Next: carry the recipe
that worked here into the scale run — `DELTA=1`, recall/needle documents with
answer-emphasised loss weights (the `aux_weight` mixing rule, applied per
document type), then re-probe. The brain-split idea (recurrent pattern core +
explicit fact memory, A4 `cond_mem`) remains the strategic path for
*parametric* facts, which this bench does not measure.

## Phase 3b: mix-3B + DELTA + answer-weight — multi-way recall transfers to the real 100M; horizon 200 → ~1000 tok; retention is now the limiter (2026-09-06)

**Run.** `mix3b_delta_answ100` (`1c913ef`), RTX Pro 6000, 2026-09-05 17:47 →
09-06 10:42 UTC (~17 h; delta is the pure-torch chunked scan, 51.5K tok/s vs
86K for chrono+gate). Identical to mix-3B `bbc12e9` (102M `baseline_real_pm`
+ chrono + out_gate, B18 T2048, 3.0B tokens, dclm/fineweb/smoltalk2/recall,
300M web-only warmup, lr 2e-4 cosine) except: **`--delta`** (A3), **`--answer_weight
100`** (CE weight ×100 on the answer tokens of the synthetic recall docs, see
`v7.data.ANSWER_MARK`), recall share 3 → 5 %. Log
`logs/v13_sempty_mix3b_delta_answ100_1c913ef_20260905_1747.log`; ckpt
`checkpoints_v13_sempty/mix3b_delta_answ100_1c913ef/best_model.pt` (step 80k).

**PPL (guard) — a wash.** Holdout val **26.38** vs mix-3B 25.73 (+0.65, +2.5 %);
WikiText val **54.73** vs 54.91 (−0.18). The train-loss column is the
answer-weighted mean and is NOT comparable to mix-3B. One transient: val
spiked to 63.8 at step 10k (mix-3B 41.7) right after the 300M web-only warmup
ended and the ×100-weighted recall docs entered; recovered by 14k and tracked
mix-3B ~0.5–1.0 behind thereafter.

**Behavioral recall (`scripts/run_memory_behavioral.py --model-type
v13_sempty --max-context 8192`, 8 candidates, 20 seeds × 3 positions, chance
0.125; `logs/memory_probes/v13_sempty_mix3b_delta_answ100_1c913ef_behavioral.json`).**
Trained at T=2048; 4096/8192 are extrapolation (new `--max-context` flag lets
the O(1)-state model be probed past `max_seq_len`; its RoPE cache grows on
demand).

```
                 ctx   128   256   512  1024  2048  4096  8192
delta+answ100  a1    1.00  1.00  1.00  0.77  0.50  0.38  0.33
               a4    1.00  1.00  0.97  0.65  0.52  0.48  0.47
               a8    0.77  0.72  0.63  0.53  0.38  0.32  0.32
mix-3B (ref)   a1    1.00    –   0.35    –   0.25    –     –
               a4    0.40    –   0.23    –   0.27    –     –
               a8    0.22    –   0.22    –   0.25    –     –
Mamba-130M     a8    0.82    –   0.57    –   0.38    –     –   (~300B tok, publication probe)
```

By needle position (pos1 = needle immediately before the query, pos0 = needle
at the start of the context):

```
                 ctx   128   256   512  1024  2048  4096  8192
a1 pos0              1.00  1.00  1.00  0.45  0.10  0.05  0.00
a1 pos0.5            1.00  1.00  1.00  0.85  0.40  0.10  0.00
a1 pos1              1.00  1.00  1.00  1.00  1.00  1.00  1.00
a8 pos0              0.70  0.70  0.60  0.25  0.20  0.15  0.10
a8 pos1              0.80  0.70  0.70  0.70  0.70  0.60  0.70
```

**Memory-path ablation** (`--pam-scale 0.0`, PAM read removed from the
residual; `…_PAM0_behavioral.json`): a1/a4/a8 = 0.05/0.20/0.25 at 128, 512
and 2048 — identical at every context, i.e. a context-blind prior. Every point
of recall above comes from the PAM memory.

**What transferred.**

1. **Multi-way recall is real at 100M.** a4 0.40 → **1.00** @128 (0.97 @512);
   a8 0.22 → **0.77** @128, 0.63 @512. The a8 curve 0.77 → 0.38 (128 → 2048)
   is the same shape and level as Mamba-130M's 0.82 → 0.38, obtained with ~3B
   tokens instead of ~300B. The bench prediction (delta + concentrated answer
   signal ⇒ selective keys) held on real mixed data.
2. **Horizon ×5.** a1 stays 1.00 to 512 (mix-3B fell to 0.35), 0.77 @1024,
   0.50 @2048. Read horizon ≈ 1000 tokens (was ~200).
3. **The read path is length-invariant.** pos1 is perfect at every length,
   including 8192 = 4× the training window (a1 1.00, a4 1.00, a8 0.70). Nothing
   breaks positionally; delta reads a fresh binding exactly at any T.
4. **What fails is RETENTION over distance, not retrieval.** pos0 accuracy
   falls monotonically with distance (1.00 → 0.45 @1024 → 0.10 @2048 → 0 @8192)
   while pos1 stays perfect: a binding written ~1–2k tokens earlier has been
   decayed out of the state before the query arrives. Internals agree:
   `dt_bias` −3.9…−4.1 (frozen at init in every run so far), realized
   retention 0.64–0.97 per layer, `pam_scale` 0.04–0.55 growing with depth.
   This is the exact inverse of the L1 verdict ("retention is fine, retrieval
   is the bottleneck") — L1 could not read even a fresh binding, so long
   retention had nothing to show. With the read fixed, the horizon is set by
   decay. R1's dt-spread failed *under a broken read*; a retention experiment is
   only now interpretable.

**Caveats (do not oversell).** (a) The probe's record format ("K means V …
query: K means") is the same *family* as the training dense docs; keys, values
and prefixes are unseen, but this is a mechanism probe, not natural-language
needle-in-a-haystack. mix-3B had the same exposure, so the comparison is fair.
(b) 102M, 3B tokens; no 100M transformer trained on this mix exists to compare
recall against. (c) Delta costs 40 % throughput on the torch path and has no
decode path (no generation from this ckpt yet).

**Decision.** **DELTA + answer-weight = the new reference recipe** (`DELTA=1
ANSWER_W=100`). Multi-way read is solved at this scale; the program moves to
**retention over distance** (the horizon) and to engineering debt the result
now justifies: Triton delta kernel (speed), delta decode (`generate.py`).

## Engineering: batched delta path (1.4× delta training) + delta decode (2026-09-06)

**Why.** Delta is the reference recipe but cost 2.0× the additive step at 100M
(B6 T2048 fwd+bwd 251 vs 125 ms; on the 6000 at B18: 51.5K vs 86K tok/s, 56 vs
31 GB) and had no decode path, so `--gen_every 0` and no `generate.py` on
delta checkpoints.

**Where the time actually was (profile, `torch.profiler`).** *Not* the Triton
scan. `pam_delta_torch` looped over T/256 chunks × 16 layers: eight fp32
`solve_triangular` + their backward (~50 ms), fp32 SIMT GEMMs and `[256,256]`
elementwise traffic (~70 ms), eight scan launches and eight slice-backward
nodes. The mathematics didn't need any of that.

**The restructuring (`triton_kernels.pam_delta_batched`).** The pseudo-values
are *linear* in the carried state: for sub-chunk c with start state S,
`W_c = A_c⁻¹ diag(b_w) V_c − A_c⁻¹ diag(b_e a) K_c Sᵀ = W0_c − U_c Sᵀ`, and
`W0, U` do not depend on `S`. So **one** batched unit-triangular solve over all
64-wide sub-chunks (rhs `[b_w V | b_e a K]`) gives both; the chunk-end state
then obeys an *affine* recurrence in `S` alone, `S_c = S_{c−1} M_c + N_c` with
`M_c = a_last I − (E∘U_c)ᵀK_c`, `N_c = (E∘W0_c)ᵀK_c` precomputed batched
(`E_t = a_last/a_t`); T/64 tiny `[K,K]` `baddbmm` steps; then
`W = W0 − U S_prevᵀ` for every chunk at once and **one**
`fused_real_pam_read(q, k, W, retention)` launch over the whole sequence (its
BT=64 tiles line up with the sub-chunks; its internal states equal `S_c` by
construction). Sub-chunk 64 instead of 256 makes the solve and all `[w,w]`
work 4× smaller per token. Things tried and rejected on the way: a log-depth
(Hillis–Steele) scan for the recurrence (5× the fp32 K³ FLOPs + `cat`
traffic: slower); per-step `select`/`slice` instead of `unbind`/`stack` (each
backward zero-fills the full `[BH,NC,K,K]` tensor: 37–58 ms). The coefficient
GEMMs run in the scan dtype (bf16 under autocast, fp32 accumulate) — the
precision the kernel itself already uses for `W` and `k`; the `[K,K]`
recurrence stays fp32. The prep is `torch.compile`d (shape-generic;
`V13S_DELTA_COMPILE=0` → eager).

**Parity (`v13_sempty/tmp/delta_parity.py`).** vs `pam_delta_torch` (kept as
the oracle): read, state and all grads (q,k,v,retention,b_w,b_e,carry), fp32
worst rel 2e-6 (tol 2e-4), bf16 2.3e-3 (tol 3e-2), incl. carry-in and ragged
T=300. `pam_kernel_test.py` PASS.

**Speed.** B6 T2048 fwd+bwd: 251 → 175 ms (additive 122); delta/additive
2.0× → 1.45×; peak 18.8 → 15.7 GB. 60-step train smoke (wikitext B8,
chrono+gate+delta, grad-ckpt, bf16, seed 42): loss trajectory identical to 4
decimals (9.7201/8.6156/8.4713 vs …/8.4712), **56.1K vs 39.9K tok/s** (RTX
6000; all numbers in this section are 6000-measured — the commit messages
say "4090" by mistake). What remains (~55 ms at B6) is diffuse — solve ~10 ms, small bmms,
casts; a hand-fused WY Triton kernel (fla-style) would buy maybe another 20–30
ms and is deferred until delta is the Stage-L throughput limiter.

**Delta decode (`PAMLayer._stepwise_delta`).** Per token on the fp32 carried
notebook: `S_t = g_t S_{t−1}(I − b_e k kᵀ) + b_w v kᵀ`, `y = S_t q` — fade,
erase along the unit-norm key, write, read; same order as the chunked WY
form. Parity (`v13_sempty/tmp/delta_decode_parity.py`): full-sequence chunked
logits vs prefill(T0)+step-wise, 2-layer delta+chrono+gate with gates
perturbed off init: fp32 rel 5.6e-7 (also ragged T0=64), bf16 3.2e-3, argmax
agree 1.00/0.97. `generate.py` on `mix3b_delta_answ100` (greedy): "The
capital of France is" → "Paris. In the year 1848, the French government …"
(coherent). **Observation worth keeping:** verbatim in-context copy works
(`The word "florp" means a small green` → `stone`), but free-form multi-way
lookups do not (`Alice lives in Toronto. Bob lives in Lisbon. Carol lives in
Nairobi. Bob lives in` → `New York`; `Key: mango -> 9028 … Key: mango ->` →
`4471`, the *first* value). The probe-format recall (records / filler /
"query: K means") has not generalised to natural phrasing: a data-format
diversity item for the recall docs, not a mechanism failure.

## R2 retention bench — under LM pressure passive decay kills multi-way recall; `no_decay` (erase-only memory) restores 1.00 to 8192 (2026-09-06)

**Question.** Phase 3b left retention over distance as the horizon limiter,
with `dt_bias` frozen at −4 and realised retention 0.64–0.97/token (content
pushes toward forgetting). The pure-recall micro-bench cannot show this: with
no LM loss the model simply learns to retain. So: a bench with the forgetting
pressure in it. `v13_sempty/tmp/retention_microbench.py`, 27M chrono+gate+delta
(4 layers, hd 64), 6000 steps × B16, every step one length bucket
(256/512/1024/2048/4096, weights .30/.28/.22/.12/.08); **75 % of each batch =
WikiText-103 train windows (LM loss, weight 1), 25 % = recall docs** (records /
filler / "query: K means" → V, 1–8 associations, needle at 0 / 0.5 / 1.0 of the
context, answer token weight ×100 — the reference recipe's loss). Eval:
position-resolved probe (pos0 = needle at START, pos1 = needle just before the
query) at ctx 256..8192 (8192 = 2× train window) for a1/a4, 24 trials each;
WikiText val PPL (32 × 2048 windows) as the LM guard; realised per-layer
retention on a web window; `dt_bias` drift. Arms, byte-identical data: `dt-4`
(reference), `dt-6`, `dt-8` (longer default retention since the bias never
learns), `nodecay` (retention pinned 1.0 — the memory changes only by delta
erase-on-rewrite). Log
`logs/v13_sempty_retention_bench_dt_nodecay_54c62ed_20260906_1411.log`, RTX
6000, ~8 min/arm at 160–200K tok/s.

**Result.**

| arm | val PPL | realised retention/layer | dt_bias | a1 pos0 256→8192 | a4 pos0 256→8192 | a4 pos1 256→8192 |
|---|---|---|---|---|---|---|
| delta dt−4 (ref) | 87.32 | .86 .89 .85 .93 | −3.94…−3.98 | 1.00 all | .08 .17 .12 .25 .17 .17 | .42 .38 .42 .42 .42 .42 |
| delta dt−6 | 86.00 | .89 .94 .89 .89 | −5.92 | 1.00 all | .04 .17 .04 .12 .08 .17 | .12 .17 .21 .21 .21 .21 |
| delta dt−8 | 85.51 | .91 .97 .91 .86 | −7.89 | 1.00 all | .17 .21 .17 .12 .12 .17 | .17 .17 .25 .25 .25 .17 |
| **delta nodecay** | **97.65** | 1.0 1.0 1.0 1.0 | (unused) | 1.00 all | **1.00 1.00 1.00 1.00 1.00 .88** | **1.00 all** |

(a1 pos1 = 1.00 for every arm; chance ≈ 0.12.)

**Reading it.**
1. **The bench reproduces the pretrain's forgetting signature, not its a1
   symptom.** Realised retention 0.86–0.93/token (an e-fold every ~10 tokens)
   with `dt_bias` moving < 0.1 from init in every decaying arm — exactly what
   the 100M showed. But a1 pos0 is 1.00 to 8192 for all arms: a single
   binding survives because the read is content-gated (retention is high on
   record tokens, low on filler), so the 100M's a1 pos0 decay is not
   reproduced at 27M/6k steps. The a1 horizon question stays with the real
   run.
2. **What the LM pressure does break is multi-way.** The same architecture
   that scored a4 = 1.00 on the pure-recall bench is at chance on a4 at
   *both* positions once 75 % of the tokens are web text — not a retention-
   over-distance effect (pos1 fails too) but the decaying memory being
   unable to hold four simultaneous bindings while the LM objective is
   pulling retention down. `dt_bias` −6/−8 do not help at all (the bias is
   a prior the content term overrides; realised retention barely moves),
   confirming the 09-06 diagnosis: the lever is not the bias.
3. **`no_decay` flips a4 from chance to 1.00 at every length, 0.88 at 8192
   (2× the train window) — from a fixed O(1) state.** Erase-on-rewrite is
   the only forgetting the memory needs; passive decay was destroying
   bindings the LM never asked it to destroy. This is the largest single
   behavioural jump in the program (the delta rule itself was a1 0.46 →
   1.00; this is a4 0.12 → 1.00 under realistic data).
4. **Cost: the LM guard is violated at this scale** — WikiText val PPL 97.65
   vs 87.32 (+12 %), far outside the ≤2 % rule. Plausible mechanism: with no
   decay, the state is the sum of *everything* written since the sequence
   start, so on ordinary text the read is a crowded superposition unless the
   erase gate learns to clear it; 27M/6k steps may simply not have learned
   that. Whether the cost persists at 100M/3B tokens (where the holdout PPL
   reference is 26.38) is the open question, and it is what the launched run
   decides. If it persists, the middle ground to test is a *learned floor*
   (retention = 1 − softplus-bounded small leak) rather than none.

**Decision.** `no_decay` stays a **candidate** (not KEEP) pending the 100M
holdout PPL; `base_dt_bias` −6/−8: no gain, leave at −4 (knob stays as a CLI
override, no code path to remove). Launched **`mix3b_delta_nodecay_answ100`**
(`9f33b50`): the exact Phase 3b recipe (102M, 3B tok, 0.44/0.41/0.10/0.05,
DELTA=1, ANSWER_W=100) + `--no_decay`, one variable; `GEN_EVERY=4000` now that
delta decodes. 60K tok/s (old delta path 51.5K), 45.5 GB, ETA ~24 h. Judge
by: holdout PPL vs 26.38 (guard: ≤ +2–3 %), a4/a8 at 512–8192 and a1 pos0
horizon vs Phase 3b (`run_memory_behavioral.py --max-context 8192`).

## R2 at scale: `no_decay` FAILS on the 100M — PPL +14 %, recall worse at every length; passive decay is load-bearing (2026-09-07)

**Run.** `mix3b_delta_nodecay_answ100` (`9f33b50`), RTX 6000, 2026-09-06 15:11
→ 09-07 05:57 UTC (~15 h, 60K tok/s with the batched delta path, 45.5 GB).
Identical to Phase 3b `1c913ef` (102M, 3B tok, 0.44/0.41/0.10/0.05, DELTA=1,
ANSWER_W=100) + `--no_decay` (retention pinned 1.0; memory changes only by
delta erase-on-rewrite). One variable. Log
`logs/v13_sempty_mix3b_delta_nodecay_answ100_9f33b50_20260906_1511.log`; probes
`logs/memory_probes/v13_sempty_mix3b_delta_nodecay_answ100_9f33b50_behavioral.json`
(+`_PAM0_`).

**PPL (guard) — FAIL.** Holdout **30.16 vs 26.38** (+14 %); WikiText **65.5 vs
54.7** (+20 %). The gap is there from step 10k (83.9 vs 63.8) and never
closes (40k: 34.8 vs 29.8; 80k: 30.16 vs 26.38) — the bench's +12 % was not a
small-model artefact.

**Recall — WORSE, not better** (20 trials, chance 0.125; mean over positions):

| | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| a1 nodecay | 0.87 | 0.85 | 0.62 | 0.63 | 0.38 | 0.33 | 0.15 |
| a1 ref | 1.00 | 1.00 | 1.00 | 0.77 | 0.50 | 0.38 | 0.33 |
| a4 nodecay | 0.78 | 0.72 | 0.58 | 0.52 | 0.50 | 0.38 | 0.27 |
| a4 ref | 1.00 | 1.00 | 0.97 | 0.65 | 0.52 | 0.48 | 0.47 |
| a8 nodecay | 0.67 | 0.47 | 0.38 | 0.37 | 0.20 | 0.18 | 0.15 |
| a8 ref | 0.77 | 0.72 | 0.63 | 0.53 | 0.38 | 0.32 | 0.32 |

Position-resolved a1: **pos1 (needle just before the query)** — ref 1.00 at
every length; nodecay 1.00 → 0.85 → 0.85 → 0.85 → 0.65 → 0.70 → **0.25 @8192**.
pos0 (needle at start): ref 1.00/1.00/1.00/0.45/0.10/0.05/0.00; nodecay
0.65/0.85/0.35/0.40/0.25/0.15/0.10 — no horizon gain either. `--pam-scale 0`
→ chance everywhere (0.15/0.30/0.15): what recall there is still comes from
the memory.

**Mechanism — state overflow, now measured.** With `g = 1` the state is the
sum of everything written since the window start, minus whatever the erase
gate removes along *re-used* keys. Web tokens whose keys never recur are
never erased, so `S` fills with a superposition of junk, and the read of even
a *fresh* binding gets noisier the longer the context — that is exactly the
pos1 curve (retrieval degrading with length is impossible under decay, and
the reference shows none). The model's own response confirms it: the
per-layer memory gate `pam_scale` at step 80k is **0.01–0.05 in layers 0–6**
(reference 0.04–0.19) and 0.10–0.18 in layers 7–13 (ref 0.22–0.41) — it
turned the memory path *down* across the stack because a crowded state is a
noisy read on ordinary text, and that is where the PPL went. Only layers 14–15
kept it (0.56/0.46) for the recall circuit. Erase alone did not learn to keep
`S` clean; passive decay was doing that job.

**Why the 27M bench got it wrong.** The bench trained on 25 % recall docs;
the real mix is 5 %. At 27M/6k steps the state was mostly *recall-loaded*, so
pinning retention looked free and helped multi-way; at 100M with 95 % web
tokens the same rule floods the state. The bench also never reproduced the
100M's a1-pos0 decay. **Do not use the retention micro-bench to vet decay
knobs**; it is fine for the read path (delta) where it did transfer.

**Decision.** `no_decay` **REMOVED** (config, model, train CLI, launcher
env, bench arm) per the sROI rule. `base_dt_bias` stays a plain CLI override
(no code path). Ckpt: `latest.pt`/`step_080000.pt` pruned, `best_model.pt`
kept (needs `git checkout 9f33b50` to load faithfully). **Passive decay is
load-bearing in this architecture: it is the mechanism that keeps the O(1)
state clean on ordinary text.** The horizon must come from *better use* of
decay, not its removal — see the next entry / SCRATCHPAD for the call.

## Stage L-2: 8K ctx + long-gap recall docs — data pressure does NOT move the decay prior; far horizon barely above chance, short-range multi-way collapses (2026-09-07)

**Question.** Is the ~1000-token horizon a *data* problem (the recall docs
never ask for longer gaps, so the model never learns to hold a binding), or a
*prior* problem (dt −4 ⇒ 0.982/token passive decay; nothing survives 8k
tokens unless the content gate pins retention to 1.0)?

**Run.** `mix2b_8k_delta_answ100_long` (`13ea6a8`), RTX 6000, 2026-09-07
07:17 → 18:18 UTC (11 h, 56K tok/s, 79 GB). Reference recipe (102M, DELTA=1,
ANSWER_W=100, decay on, dt −4) at **T=8192 B=8** (65,536 tok/step), 2.0B
tokens = **30,517 steps**, sources
`dclm,fineweb_long,pg19,smoltalk2_mid,recall,recall_long` at
`0.36,0.20,0.22,0.10,0.04,0.08` (`recall_long` gaps ≤ 6k tokens), 100M-token
web-only warmup. Log
`logs/v13_sempty_mix2b_8k_delta_answ100_long_13ea6a8_20260907_0717.log`, probe
`logs/memory_probes/v13_sempty_mix2b_8k_delta_answ100_long_13ea6a8_behavioral.json`.

**PPL.** Holdout (this mix) 30.55; WikiText 59.26 (Phase 3b 54.73; L1 at 1B
on a similar long mix 40.26 holdout — not comparable, different holdout set).

**Behavioral recall (8-way contrastive, 20 trials, `--max-context 8192`).**
a1 pos0 (needle at start = the horizon metric), ctx 128/256/512/1024/2048/4096/8192:

| arm | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| Phase 3b (T=2048, 3B, no long docs) | 1.00 | 1.00 | 1.00 | 0.45 | 0.10 | 0.05 | 0.00 |
| **L-2 (T=8192, 2B, 8 % long docs)** | 1.00 | 1.00 | **0.60** | 0.55 | 0.30 | 0.20 | 0.20 |
| L1 (T=8192, 1B, additive read) | 0.40 | 0.15 | 0.10 | 0.15 | 0.10 | 0.15 | 0.15 |

Chance is 0.125 (n=20: 4/20 is p≈0.24 under chance). a1 pos1 (needle just
before query) stays 1.00 at every length — the read is still length-invariant.
Multi-way, mean over positions, L-2 vs Phase 3b: a4 @128 **0.62 vs 1.00**,
@512 0.52 vs 0.97; a8 @128 **0.40 vs 0.77**, @512 0.30 vs 0.63. Even the
fresh-binding multi-way read regressed (a4 pos1 0.75 vs 1.00, a8 pos1 0.45
vs 0.70).

**Internals (step 30000).** `dt_bias` = **−3.98 … −4.08 in all 16 layers**,
per-head layer-mean −4.00/−3.98/−3.97/−4.00/−3.97/−3.99 — identical to Phase
3b and to init, after 2B tokens of which 8 % explicitly require 1k–6k-token
retention. Realised retention 0.62–1.00 (content gate does all the holding).
`pam_scale` 0.08–0.45 (healthy, memory path in use; no `no_decay`-style
collapse).

**Reading.**
1. **Data pressure does not move the decay prior.** The one number that
   sets the passive horizon did not move by 0.1 with the most retention-hungry
   data we have. The gradient into `dt_bias` from ordinary text (which prefers
   forgetting) swamps the recall-doc gradient; under `softplus`, the gradient
   at −4 is also small. The horizon therefore has to be given *structurally*.
2. **Far horizon: at best marginal.** pos0 at 2048–8192 moved 0.10/0.05/0.00 →
   0.30/0.20/0.20 — the 2048 point is real, the 4k/8k points are within noise
   of chance. Not the "pos0 moves ⇒ data problem" outcome the pre-registered
   rule needed.
3. **Short-range multi-way collapsed** (a4/a8 at 128–512 roughly halved).
   Confounded: 2.7× fewer optimizer steps (30.5k vs 81k), 2B vs 3B tokens,
   recall share 0.12 but half of it in long docs with few weighted answer
   tokens per sequence, and a pg19-heavy mix (WikiText +8 %). Most plausibly a
   step-budget effect, but this run cannot separate it. **Do not use T=8192
   pretraining as the vehicle for the horizon question again** — it costs 2.7×
   the steps per token for the same 100M and muddies every comparison.

**Decision.** Data pressure alone: **FAIL** as a horizon lever (pre-registered
rule). Nothing added to code from this run; the 8K data path stays (it is
Stage L infrastructure). **Next = R3, the pre-registered fallback:** a
*split* decay prior — `--long_heads 2 --long_dt_bias -9` (2 of 6 heads per
layer born at softplus(−9)=1.2e-4 ⇒ ~0.37 of a binding survives 8192 tokens
of passive decay; the other 4 heads keep −4), everything else the exact Phase
3b recipe (T=2048, B18, 3B tok, 0.44/0.41/0.10/0.05, DELTA=1, ANSWER_W=100).
One variable against the Phase 3b reference; judge by a1 pos0 at 1024/2048
(in-window: 0.45/0.10) and 4096/8192 (extrapolation: 0.05/0.00), with a4/a8
@128–512 and holdout PPL 26.38 as guards. This differs from the failed R1
ladder in two ways: the read is the working delta read (R1 ran under the
additive read that could not even read fresh bindings), and it is a split
(4 normal + 2 long) not a spread of every head. Code: `_init_dt_bias` in
`model.py`, `long_heads`/`long_dt_bias` in `config.py`, `LONG_HEADS` env in
the launcher (`6a0e3c9`). Ckpt: L-2 pruned to `best_model.pt`.

## R3 split decay prior (2 of 6 heads born at dt −9): FAIL, removed (2026-10-27)

**Run.** `mix3b_delta_answ100_long2h9` (`5c87148`), RTX 6000, 2026-09-07 18:41 →
09-08 09:26 UTC (~15 h, 56.5K tok/s, 3.00B tok, 81,380 steps, exit 0). Exact
Phase 3b recipe + `--long_heads 2 --long_dt_bias -9`. One variable. Log
`logs/v13_sempty_mix3b_delta_answ100_long2h9_5c87148_20260907_1841.log`.
The probe was not run on the 6000 (GPU held by an SGLang server); it was run
2026-10-27 on the local 4090, **40 trials** (n=20 is too noisy: 4/20 happens
by chance ~24 % of the time), and Phase 3b was re-probed at the same 40 seeds
for a like-for-like reference. Probe log
`logs/v13_sempty_probe_r3_vs_p3b_t40_215fe61_20261027_0551.log`; jsons
`logs/memory_probes/v13_sempty_mix3b_delta_answ100_long2h9_5c87148_t40_behavioral.json`,
`logs/memory_probes/v13_sempty_mix3b_delta_answ100_1c913ef_t40_behavioral.json`.

**PPL (guard) — pass.** Holdout **26.12** vs 26.38; WikiText 53.02 vs 54.73.

**Recall (8-way contrastive, 40 trials, chance 0.125), R3 / Phase 3b:**

| cell | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| a1 pos0 R3 | 0.75 | 0.62 | 0.47 | 0.33 | 0.20 | 0.17 | 0.15 |
| a1 pos0 ref | 1.00 | 1.00 | 1.00 | 0.38 | 0.12 | 0.10 | 0.07 |
| a1 pos1 R3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.80 | 0.90 | 0.85 |
| a1 pos1 ref | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| a4 pos1 R3 | 0.68 | 0.72 | 0.60 | 0.60 | 0.57 | 0.55 | 0.57 |
| a4 pos1 ref | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.95 | 0.97 |
| a8 pos1 R3 | 0.53 | 0.40 | 0.40 | 0.45 | 0.42 | 0.38 | 0.40 |
| a8 pos1 ref | 0.90 | 0.85 | 0.82 | 0.82 | 0.80 | 0.72 | 0.78 |

The far horizon (pos0 @2048–8192) moved by at most +0.08, at the edge of
chance. Every guard broke: start-of-context recall at 128–512 fell from 1.00
to 0.47–0.75, and even the **fresh** binding (pos1) read fell, most for
multi-way (a4 1.00 → ~0.6, a8 0.8 → ~0.4).

**Internals.** Per-head layer-mean `dt_bias` at step 80k
−8.94/−8.94/−3.93/−3.94/−3.97/−3.96: frozen at init, like every decay prior.

**Mechanism.** Same overflow signature as `no_decay` (R2 at scale), confined
to the two slow heads: every token still writes into them, so they fill with
web-text junk and their read pollutes the head mix, including fresh bindings.
The small PPL gain says slow heads carry some useful long context for LM;
the recall loss says an unselective slow memory is noise for exact lookup.

**Decision.** R3 FAIL by its pre-registered rule → `long_heads` /
`long_dt_bias` **removed** (config, model, train CLI, launcher). Per the same
rule: stop tuning the decay prior. The horizon needs a different *write* —
R4 below.

## R4 consolidation store — FAIL, removed: never used for recall, multi-way −9 pts (2026-10-27)

**Why every retention run failed (the physics).**
1. *Horizon ≈ capacity ÷ write rate.* A 98-dim unit-key head holds ~d ≈ 100
   clean bindings; each unrelated delta write perturbs a stored binding by
   ~1/√d, so a binding is buried after ~d writes. With every token writing
   into every head, decay is the only garbage collector (`no_decay`, R3).
2. *No gradient for retention.* A binding read L tokens later contributes
   ∝ Πγ ≈ γ^L; at dt −4 (γ = 0.982) and L = 2000 that is e^−36. The
   gradient that would say "keep this longer" is zero, so dt_bias never moves
   (L1, L-2, R3) whatever the data.
3. Every retention run changed *how fast things fade*; none changed *who
   writes*.

**Mechanism.** Per head, next to the fast notebook (unchanged), a second
d×d store E sharing the head's q/k/v, taken *before* rotary:

```
every C=64 tokens: the m=1 token with highest salience s = w_s·x writes
    E <- E + sigmoid(s) (v - E k) k^T        (unit k, error-correcting delta, NO decay)
read: y_t = fast_t + g_t * d^-1/2 * E_(chunk start) q_t     (g = zero-init linear)
```

- no decay → the gradient from a query 8K tokens later reaches the write;
- 1 write / 64 tokens → 128 writes in 8K ≈ one head's capacity (no_decay
  with 64× fewer writes);
- selection at chunk end sees 64 tokens, still causal (a token reads the
  store as of its chunk start; decode buffers ≤63 tokens → O(1));
- no rotary → facts addressed by content, not distance;
- zero-init gate → bit-identical to the reference at init (warm-start safe).

+2 `dim→heads` linears per layer (~113K params, 0.1 %). Code:
`RealPAMLayer._store`, `cfg.store/store_chunk/store_writes`, `--store`,
`STORE=1`; `--init_from` warm start (weights only, fresh optimizer, stream
continues after the source cursor). Contract: `selftest
test_store_parity_and_decode` (init parity 0.0; prefill+step 7.7e-7;
stepwise 6.6e-7; carried store 2.7e-7), 18/18 pass; layout guard clean.

**Plan.** (1) Warm-start screen from the Phase 3b checkpoint, both arms
identical except `--store`: control (no store) vs R4, same tokens/LR/seed.
(2) If it passes, the 3B scratch run on the 6000.

**Pre-registered judge (40 trials, vs the control arm):** KEEP if a1 pos0
@2048 ≥ 0.5 and @4096 ≥ 0.3; guards a4/a8 pos1 within 0.1 of control at
every ctx, a1/a4 pos0 @128–512 within 0.1, holdout PPL ≤ control + 0.3, tok/s
≥ 85 %. Diagnostics: per-layer |store gate| (is it used?), and which tokens
the salience picks (names/numbers/definitions = learning; punctuation/BOS =
not). If the gate opens and pos0 does not move → selection did not learn →
try surprise-based selection. If the gate stays ~0 → no long-gap pressure in
the 2K data.

### R4 baseline: far-context gain on natural text (PG-19, 2026-10-27)

LLM-side horizon metric (`scripts/eval_pos_loss.py`, `f366601`): 50 PG-19
test books, one 8192-token window each (2048 for the transformer). Every token
is scored with full context and with only 512–1024 tokens of context;
`far_gain` = short − full loss (nats) on the same tokens.

| model | 1–2K | 2–4K | 4–8K | loss @1–2K |
|---|---|---|---|---|
| Phase 3b (`1c913ef`) | +0.014 | +0.019 | +0.022 | 3.61 |
| R3 (`5c87148`) | +0.014 | +0.020 | +0.024 | 3.61 |
| v11-matched transformer (1B tok, 2048 ctx) | +0.027 | — | — | 4.35 |
| R4 ctrl (Phase 3b + 300M, `0868a1b`) | +0.014 | +0.020 | +0.022 | 3.61 |
| R4 store (Phase 3b + 300M + store) | +0.014 | +0.020 | +0.022 | 3.60 |
| Mamba-2 102.6M, Phase 3b data (`b36e3a5`, LR 1.2e-3) | +0.017 | +0.025 | +0.031 | 3.51 |

PAM does use far context, and the gain keeps growing past its 2K training
window, but at 1–2K it gets about half of what a (weaker) transformer gets
from the same tokens. Raw per-position loss *rises* after ~512 for all three
models, and the short-context loss rises identically, so that rise is book
structure, not forgetting. R4 should raise `far_gain`; this table is the
reference. Results: `logs/pos_loss/`,
`logs/v13_sempty_pos_loss_pg19_f366601_20261027_0637.log`.

### R4 result — FAIL, removed (2026-10-27)

**Run.** Warm-start screen `v13_sempty/tmp_r4_screen.sh` @`0868a1b` on the RTX
6000 (renamed `tmp_warm_screen.sh` after, now generic). Both arms from Phase 3b
`best_model.pt`, 300M tokens each (8,138 steps, B18 T2048, LR 5e-5, warmup 200,
DELTA=1, ANSWER_W=100), stream continued after the Phase 3b cursor; identical
except `--store`. Realised mix identical in both arms (dclm 0.430 / fineweb
0.384 / smoltalk2 0.142 / recall 0.044). Logs
`logs/v13_sempty_r4screen_{store_0868a1b_20261027_0708,ctrl_0868a1b_20261027_0941}.log`,
driver `logs/v13_sempty_r4screen_driver_0868a1b.log`, probes
`logs/memory_probes/v13_sempty_r4screen_{store,ctrl}_0868a1b_t40_behavioral.json`,
PG-19 `logs/pos_loss/v13_sempty_r4screen_{store,ctrl}_0868a1b_pg19.json`.

**Guards.** Holdout PPL store **26.40** vs ctrl **26.42**; WikiText 55.08 vs
55.02 — the store bought no LM. tok/s **50.9K vs 59.3K = 86 %** (same box,
same day; the "98 %" quoted mid-run compared against September's Phase 3b
speed, wrong reference).

**Recall (40 trials, chance 0.125), store / ctrl:**

| cell | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| a1 pos0 store | 1.00 | 1.00 | 1.00 | 0.38 | 0.30 | 0.15 | 0.07 |
| a1 pos0 ctrl | 1.00 | 1.00 | 1.00 | 0.47 | 0.20 | 0.15 | 0.07 |
| a4 pos0 store | 0.97 | 0.90 | 0.70 | 0.35 | 0.25 | 0.20 | 0.17 |
| a4 pos0 ctrl | 1.00 | 0.93 | 0.80 | 0.47 | 0.25 | 0.25 | 0.17 |
| a8 pos0 store | 0.78 | 0.62 | 0.38 | 0.25 | 0.20 | 0.17 | 0.15 |
| a8 pos0 ctrl | 0.82 | 0.78 | 0.65 | 0.30 | 0.25 | 0.17 | 0.20 |
| a8 pos1 store | 0.78 | 0.75 | 0.72 | 0.75 | 0.72 | 0.62 | 0.60 |
| a8 pos1 ctrl | 0.85 | 0.82 | 0.80 | 0.80 | 0.80 | 0.72 | 0.70 |

Mean store − ctrl over all 21 cells per count: a1 −0.008, a4 −0.038, **a8
−0.087**. Judge: a1 pos0 @2048 0.30 (bar 0.5), @4096 0.15 (bar 0.3) → **FAIL**;
guards also broken (a8 pos0 @512 −0.27, a8 pos1 @8192 −0.10).
PG-19 `far_gain` identical to ctrl at every range (table above).

**Is the store used? (final ckpt, `inspect_store.py`, 16 a1 pos0 @2048 docs.)**
|gate| 0.03–0.07 in 15 of 16 layers (layer 14: 0.17, |gate·read| 0.51).
Salience does find the needle in upper layers (top-1 within ±8 tokens of the
value 0.42–0.67 in layers 10–15 vs ~0.27 chance), but **zeroing every store
gate changes no recall cell by even one trial** (a1 pos0 @1024/2048 0.38/0.30
on and off; a8 pos0 @256/512 0.62/0.40 on vs 0.62/0.42 off; a8 pos1 @1024 0.75
both). So (1) the store read carries nothing the probe needs, and (2) the
multi-way loss is not the read adding noise — it is training: the store's
gradient flows into the q/k/v projections it *shares* with the fast notebook
and degraded the notebook's multi-way keys. The +0.10 at a1 pos0 @2048 is
not the store either (present with the gate zeroed): n=40 noise.

**Reading.** The pre-registered branch "gate stays ~0 → no long-gap pressure
in the 2K data" fired. Combined with L-2 (long-gap docs at 8K moved nothing
structural) this is the sixth retention lever that left the horizon at
~1000 tokens: R1 spread, L-2 data, R2 no_decay, R3 split prior, R4 store, and
+300M tokens (ctrl vs Phase 3b: mean +0.005 / +0.007 / +0.017 for a1/a4/a8).
Selection learned something (salience finds needles); *use* never did,
because nothing in 2K-window LM loss pays for exact recall past ~1K.

**Decision.** FAIL → store removed from code (`store/store_chunk/store_writes`,
`RealPAMLayer._store`, `store_scan`, its two selftests, `--store`, `STORE=1`).
Kept: `--init_from` warm start and the screen driver (now
`tmp_warm_screen.sh`, generic arms). The R4 ctrl arm
(`checkpoints_v13_sempty/r4screen_ctrl_0868a1b` on the 6000) is the reusable
control for any future warm-start screen with this recipe. To re-probe the
store ckpt, `git checkout 0868a1b` (current code has no store keys).

## Mamba-2 baseline at equal data — the "wind tunnel" (design + pre-registration, 2026-10-27)

**Question.** Six retention levers left PAM's single-fact horizon at ~1000
tokens (R4 result above). Our only Mamba number is HF Mamba-130M trained on
300B Pile tokens (100× our 3B): it recalls one fact at 1.00 out to 8192 but
loses most multi-fact cells. Is the horizon a **design** limit of PAM, or a
**data/scale** limit any O(1) model has at 100M params / 3B tokens?

**Design.** Reference Mamba-2 (`mamba_ssm` 2.2.5 `MambaLMHeadModel`, Mamba-2
mixer, its own init; `v13_sempty/baseline_mamba2.py`), d_model 768, 17
layers, d_state 128, headdim 64, expand 2 = **102.6M** (PAM 102.1M), trained
by `v13_sempty/train.py --arch mamba2`: same streamed data in the same order,
the same fused CE with answer weight 100, same AdamW/cosine/clip/wd, same
holdout + WikiText val, same probes. Driver `v13_sempty/tmp_mamba2_baseline.sh`:
(A) LR screen 2e-4 (PAM's) / 6e-4 / 1.2e-3 × 2,700 steps on the real mix,
lowest final holdout loss wins, a higher LR must win by > 0.01 nats; (B) the
Phase 3b recipe (3B tok, B18 T2048, warmup 1000, 300M web-only blend warmup,
0.44/0.41/0.10/0.05, seed 42) at the chosen LR; (C) 40-trial probe + PG-19.
Wrapper checks (6000): fused-CE loss == plain CE exactly, with and without the
answer mask; grads rel ≤ 2e-5; SSD kernel vs torch reference 1.5e-4; resume
(kill at step >20, relaunch → step 20 → 30) ok; 177K tok/s at B18 (PAM 59K).

**Pre-registered reading (vs Phase 3b, 40 trials, chance 0.125):**
- Mamba-2 a1 pos0 **@2048 ≥ 0.5 and @4096 ≥ 0.3** (the R4 bars) → **design**:
  equal data, Mamba-2 holds what PAM cannot. Its write is ∝ dt and its
  forget is exp(−A·dt) from the same dt, so a filler token neither writes nor
  erases — PAM has separate write/erase/decay gates. Next = port that
  coupling into PAM (one variable, warm-start screen).
- Mamba-2 a1 pos0 **@2048 ≤ 0.3** (PAM: 0.12–0.20) → **data/scale**: the
  horizon is not PAM-specific at 3B tokens. Stop retention levers; spend the
  GPU on scale/quality instead.
- In between → read PG-19 `far_gain` (PAM +0.014/+0.020/+0.022) as the
  tie-break.
- Reported either way: holdout + WikiText PPL at equal tokens ("can we beat
  Mamba"), a4/a8 multi-fact cells, far_gain, tok/s. Caveat noted up front:
  PAM's LR (2e-4) was never screened; Mamba's is — any PPL gap in Mamba's
  favour is an upper bound.

## Mamba-2 result — DESIGN: at equal data Mamba-2 keeps one fact 2–4× further, PAM holds many facts far better; Mamba-2 −10 % PPL, 3.6× faster (2026-10-28)

**Run.** `mix3b_mamba2_answ100` @`b36e3a5`, RTX 6000, 2026-10-27 14:33 →
19:08Z (**4.6 h**, 182K tok/s mean, 188.6K steady; Phase 3b took ~17 h at
51.5K, before the data-feed fix). 81,380 steps, 3.0B tokens, LR 1.2e-3 (from
the screen above), one attempt, no NaN. Log
`logs/v13_sempty_mix3b_mamba2_answ100_b36e3a5_20261027_1433.log`, driver
`logs/v13_sempty_mamba2_baseline_driver_b36e3a5.log` (LR screen:
`..._driver_3f69057.log` + `logs/v13_sempty_m2lr_*_3f69057_*.log`); ckpt
`checkpoints_v13_sempty/mix3b_mamba2_answ100_b36e3a5/best_model.pt` (step
80k, 6000 only).

**PPL at equal steps (same holdout, same WikiText val):**

| step | 2k | 12k | 20k | 40k | 60k | 80k |
|---|---|---|---|---|---|---|
| holdout Mamba-2 | 65.0 | 38.0 | 32.2 | 28.1 | 24.8 | **23.68** |
| holdout PAM (Phase 3b) | 105.0 | 43.0 | 35.4 | 29.8 | 27.2 | **26.38** |
| WikiText Mamba-2 | 201.2 | 86.2 | 69.8 | 58.4 | 49.7 | **46.91** |
| WikiText PAM | 362.6 | 102.6 | 77.1 | 63.5 | 56.6 | **54.73** |

Mamba-2 −10.2 % holdout (−0.108 nats), −14.3 % WikiText (−0.154 nats). Both
show the same bump at 10k when the ×100 recall docs enter (63.6 vs 63.8). The
gap is an **upper bound**: PAM's LR (2e-4) was never screened, and Mamba-2's
own screen moved −0.55 nats at 100M tokens going 2e-4 → 1.2e-3.

**PG-19** (50 docs × 8192): Mamba-2 lower at every position by 0.06–0.10
nats; far_gain (1024–2048 / 2048–4096 / 4096–8192) **+0.017 / +0.025 /
+0.031** vs PAM +0.014 / +0.019 / +0.022 — uses far context a bit more, both
small. `logs/pos_loss/v13_sempty_mix3b_mamba2_answ100_b36e3a5_pg19.json`.

**Recall (8-way contrastive, 40 trials, same seeds, chance 0.125;
`logs/memory_probes/v13_sempty_mix3b_mamba2_answ100_b36e3a5_t40_behavioral.json`
vs `..._1c913ef_t40_behavioral.json`). Trained at T=2048; 4096/8192 are
extrapolation for both.**

| cell | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|
| a1 pos0 Mamba-2 | 1.00 | 1.00 | 1.00 | **1.00** | **1.00** | 0.30 | 0.15 |
| a1 pos0 PAM | 1.00 | 1.00 | 1.00 | 0.38 | 0.12 | 0.10 | 0.07 |
| a1 pos0.5 Mamba-2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.15 |
| a1 pos0.5 PAM | 1.00 | 1.00 | 1.00 | 0.80 | 0.35 | 0.12 | 0.07 |
| a1 pos1 Mamba-2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| a1 pos1 PAM | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| a4 pos0 Mamba-2 | 0.80 | 0.82 | 0.80 | 0.75 | 0.60 | 0.42 | 0.20 |
| a4 pos0 PAM | 1.00 | 0.97 | 0.90 | 0.33 | 0.23 | 0.23 | 0.20 |
| a4 pos0.5 Mamba-2 | 0.65 | 0.65 | 0.62 | 0.60 | 0.57 | 0.68 | 0.28 |
| a4 pos0.5 PAM | 1.00 | 1.00 | 0.97 | 0.62 | 0.30 | 0.25 | 0.20 |
| a4 pos1 Mamba-2 | 0.68 | 0.60 | 0.60 | 0.53 | 0.47 | 0.50 | 0.45 |
| a4 pos1 PAM | **1.00** | **1.00** | **1.00** | **1.00** | **0.97** | **0.95** | **0.97** |
| a8 pos0 Mamba-2 | 0.50 | 0.53 | 0.53 | 0.47 | 0.33 | 0.20 | 0.10 |
| a8 pos0 PAM | 0.80 | 0.78 | 0.50 | 0.23 | 0.20 | 0.15 | 0.12 |
| a8 pos0.5 Mamba-2 | 0.57 | 0.42 | 0.42 | 0.45 | 0.42 | 0.23 | 0.15 |
| a8 pos0.5 PAM | 0.85 | 0.80 | 0.68 | 0.53 | 0.23 | 0.17 | 0.15 |
| a8 pos1 Mamba-2 | 0.45 | 0.45 | 0.40 | 0.33 | 0.33 | 0.28 | 0.25 |
| a8 pos1 PAM | **0.90** | **0.85** | **0.82** | **0.82** | **0.80** | **0.72** | **0.78** |

Mean over all positions and lengths, Mamba-2 − PAM: **a1 +0.167, a4 −0.135,
a8 −0.194.**

**Reading (pre-registered rule).** a1 pos0 @2048 = **1.00** (bar ≥ 0.5; PAM
0.12, i.e. 40/40 vs 5/40) and @4096 = **0.30** (bar ≥ 0.3, met exactly:
12/40, past the trained length) → **DESIGN**. With the same params, data,
loss and tokens, Mamba-2 keeps a single fact across the whole trained
window; PAM's horizon stays ~512–1024. The tie-break was not needed.

The same table shows the opposite design gap for many facts: PAM's
outer-product memory holds 4–8 recent bindings almost perfectly (a4 pos1
0.95–1.00, a8 pos1 0.72–0.90), Mamba-2 does not (0.45–0.68 / 0.25–0.45) and
never reaches 1.00 on a4 even at 128 tokens — although it saw the same 5 %
recall docs at ×100 weight. Mamba-2 wins a4/a8 only when the facts are far
(pos0 @1024–2048). **PAM = better lookup (capacity, precision); Mamba-2 =
better keeper (retention).**

**Mechanism candidates for PAM's retention gap (not yet tested):**
1. *Write/forget coupling* (the pre-registered next step): Mamba-2's write is
   ∝ dt and its forget is exp(−A·dt) from the same per-token dt, so a token
   the model does not want stored neither writes nor erases. PAM writes
   every token (delta rule) and decays on a separate path.
2. *LR*: PAM's decay priors stayed frozen at init over 3B tokens (R3
   internals, LR 2e-4), and Mamba-2 at this data size wanted ≥ 6× that LR.
   An unscreened LR could explain part of both the PPL and the retention gap.

**Decision.** Pending (user: "decide later"). Cheapest order: PAM LR screen
(same 3-arm × 100M-token protocol as Mamba-2's) → coupling port as a
warm-start screen at the winning LR. "Can we beat Mamba-2" today: not on PPL
or speed; yes on multi-fact recall.

## PAM LR screen + C1 coupled gates (design + pre-registration, 2026-10-28)

**Why both.** The Mamba-2 result points at two causes for PAM's short
single-fact horizon: (1) PAM's LR (2e-4) was never screened while Mamba-2's
own screen moved −0.55 nats at 100M tokens going 2e-4 → 1.2e-3, and PAM's
decay priors never left their init; (2) PAM's three separate memory gates
(fade, erase, write) vs Mamba-2's one step dt for write and forget.

**Protocol (both): from-scratch screen** `v13_sempty/tmp_scratch_screen.sh`
(`ce8bcc2`): the Mamba-2 LR-screen protocol on the Phase 3b PAM recipe —
2,700 steps × 36,864 tok = 99.5M tokens, recall docs from token 0 (no
web-only warmup), warmup 100, val every 900, seed 42, then the 40-trial probe
+ PG-19 per arm. RTX 6000, 66.5K tok/s (Phase 3b ran at 51.5K before the
data-feed fixes), ~32 min per arm.

**LR screen** `SCREEN=pamlr`, arms 2e-4 / 6e-4 / 1.2e-3 @`ce8bcc2`.
- Pick: lowest holdout val_loss at step 2700; a higher LR must beat a lower
  one by > 0.01 nats (Mamba-2's rule).
- If the pick beats 2e-4 by ≥ 0.05 nats, PAM was LR-undertrained: a 3B rerun
  at the new LR is the fair PPL test against Mamba-2's 23.68 (~12.5 h; user's
  call). Also reported: per-arm recall + dt_bias movement.

**C1 coupled gates** (`504da12`, `--couple`): per token and head one step
`dt = softplus(dt_proj(x) + dt_bias)` (the existing decay step) sets
write = erase = 1 − e^{−dt} and fade = e^{−A·dt}, A per head learned. dt → 0
leaves the notebook untouched; a strong write also fades the head by e^{−A·dt}.
Arms at the screen's LR: `c1` (A init 1 = today's fade exactly) and `c1a01`
(A init 0.1: strong writes fade the rest of the head 10× less, leaving
overwrite to the targeted erase). Control = the `pamlr` arm at the same LR
(same code path, data and seed; init RNG differs because C1 builds two fewer
projections per layer, so single-seed noise applies).

Reading (40 trials, chance 0.125):
- *Precondition*: the control recalls at short range (a1 pos0 @512 ≥ 0.5).
  If not, 100M tokens is too short to judge retention → rerun control + best
  C1 arm at 300M tokens before any verdict.
- **KEEP** (→ propose a 3B run): the better C1 arm has a1 mean over
  pos0/pos0.5 at 1024+2048 ≥ control + 0.25, AND a4/a8 pos1 mean over all
  lengths ≥ control − 0.10, AND holdout loss ≤ control + 0.03 nats.
- **FAIL**: neither arm meets KEEP and neither moves that a1 mean by ≥ +0.10
  → record, remove the C1 code.
- In between (retention up but multi-fact or PPL cost over the bar): report,
  no automatic decision.

**LR screen result (`pamlr` @`ce8bcc2`, 2026-10-28 05:32–07:03Z, 66.5K tok/s).**
Logs `logs/v13_sempty_pamlr_{driver,lr*}_ce8bcc2*.log`; probes
`logs/memory_probes/v13_sempty_pamlr_lr*_ce8bcc2_t40_behavioral.json`.
Mamba-2 rows = its own screen, same protocol and data (`m2lr`, `3f69057`).

| LR | PAM holdout loss (PPL) | PAM WikiText PPL | Mamba-2 holdout loss (PPL) | Mamba-2 WikiText PPL |
|---|---|---|---|---|
| 2e-4 | 4.6467 (104.2) | 380.5 | 4.6091 (100.4) | 439.3 |
| 6e-4 | 4.0963 (60.1) | 165.9 | 4.2329 (68.9) | 258.1 |
| **1.2e-3** | **4.0293 (56.2)** | **152.1** | 4.0560 (57.8) | 183.3 |

- **Pick 1.2e-3** (beats 6e-4 by 0.067 nats, 2e-4 by **0.617**): PAM *was*
  LR-undertrained, by about as much as Mamba-2 (−0.553). At the tuned LR and
  100M tokens PAM is **ahead** of Mamba-2 (holdout −0.027 nats, WikiText
  152 vs 183). Whether that holds at 3B is the fair PPL test (3B at 1.2e-3,
  ~12.5 h, user's call).
- `dt_bias` still does not move at 5–6× the LR (layer means −3.91…−4.05 at
  step 1800); the realised retention is set by the content path `dt_proj`
  and falls as LR rises (0.85–0.99 at 2e-4 → 0.78–0.96 at 1.2e-3). A frozen
  bias is not an LR symptom.
- **Recall: no arm learned it at 100M tokens.** a1 pos0 @512 = 0.28 / 0.15 /
  0.17 (precondition ≥ 0.5 failed); even a1 pos1 @128 (needle right before
  the query) is 0.35 / 0.55 / 0.78 and ~0.25 beyond; a4/a8 at chance. (Phase
  3b at 3B: 1.00 on all those cells.) 5 % recall docs × 100M = 5M recall
  tokens is too few; the 27M micro-bench needed ~24M.

**Amendment to the C1 protocol (written 2026-10-28 07:10Z, after the pamlr
probes, before any C1 result).** The precondition failed, so per the rule the
judgement moves to longer runs. To avoid a restart if 300M is still too
short: control + ONE C1 arm on a **600M-token cosine schedule** (16,276
steps, LR 1.2e-3, warmup 100, otherwise the screen recipe), milestone
checkpoints every 150M tokens (4,069 steps) copied to the 4090 and probed
there (40 trials) while the 6000 trains on. Verdict at the first milestone
where the control meets the precondition (a1 pos0 @512 ≥ 0.5), with the same
KEEP/FAIL bars on the C1 arm's matching milestone; holdout loss from the val
at that milestone. If the control never meets it by 600M: no verdict, report
the emergence curves. C1 variant: the `c1`/`c1a01` 100M arm with the higher
a1 far mean (pos0/pos0.5 @1024+2048) if they differ by ≥ 0.10, else the
lower 100M holdout loss if they differ by > 0.01, else `c1` (A = 1, the
faithful Mamba-2 port). The two 100M C1 arms give no verdict of their own.

**C1 100M arms (`c1` @`9ebd1d2`, 07:03–08:03Z; no verdict by the amended
protocol, they only pick the 600M variant).** Control = `pamlr_lr1.2e-3`.

| arm | holdout loss | WikiText PPL | train loss @2650 (answer-weighted) | A after 100M (layer range) |
|---|---|---|---|---|
| control (A3 gates) | **4.0293** | **152.1** | 3.41 | — |
| `c1` (A init 1) | 4.1332 (+0.104) | 206.5 | 3.26 | 0.71–0.96 |
| `c1a01` (A init 0.1) | 4.0677 (+0.038) | 169.2 | **3.19** | 0.07–0.11 |

- Pick (rule): `c1a01` — a1 far means 0.138 vs 0.106 (< 0.10 apart, both
  chance), holdout 4.0677 vs 4.1332.
- Early sign, not a verdict: `c1a01` is the only arm starting to recall at
  short range (a1 pos0 @128/256 **0.82/0.70** vs control 0.53/0.33, a1 pos1
  @256 0.68 vs 0.45) and both C1 arms have the lowest answer-weighted train
  loss, while their holdout loss is worse. A = 1 (`c1`) is worst on both: a
  strong write that also fades the whole head costs LM quality.
- Logs `logs/v13_sempty_c1_{driver,c1,c1a01}_9ebd1d2*.log`, probes
  `logs/memory_probes/v13_sempty_c1_*_9ebd1d2_t40_behavioral.json`.
- **600M runs launched 08:03Z** (`r600` @`91daf0a`): `ctl`, then `c1a01`.

## C1 result — FAIL, removed: coupling write/erase/fade to one step costs retention and LM; it helps only fresh multi-fact reads (2026-10-28)

**Runs** (`r600` @`91daf0a`, RTX 6000): control (A3 gates) and `c1a01` (C1, A
init 0.1, the variant the 100M arms picked), LR 1.2e-3, 600M-token cosine
schedule, milestones every 150M probed on the 4090 (40 trials). Control ran
to 600M; `c1a01` was stopped at step 8,900 right after its 300M verdict
(~1.1 GPU-h saved). Logs `logs/v13_sempty_r600_{driver,ctl,c1a01}_91daf0a*.log`,
probes `logs/memory_probes/v13_sempty_r600_*_91daf0a_s*_t40_behavioral.json`.

| tokens | arm | a1 pos0 @512 (precondition) | a1 far (pos0/0.5 @1k+2k) | a4/a8 pos1 | holdout loss | WikiText PPL |
|---|---|---|---|---|---|---|
| 150M | control | 0.35 | 0.200 | 0.439 | 4.0195 | 148.6 |
| 150M | c1a01 | 0.28 | 0.175 | 0.159 | 4.1662 | 175.5 |
| 300M | control | **0.57** | **0.263** | 0.202 | **3.7555** | **100.4** |
| 300M | c1a01 | 0.20 | 0.100 | **0.371** | 3.8105 | 107.7 |
| 450M | control | 0.70 | 0.519 | 0.266 | 3.5920 | 83.0 |
| 600M | control | 0.62 | 0.412 | 0.223 | 3.5408 | 75.9 |

**Verdict at 300M** (first milestone where the control meets the
precondition, as amended): far a1 **−0.163** (KEEP needs +0.25, FAIL is
< +0.10), holdout **+0.055** nats (bar ≤ +0.03), a4/a8 pos1 +0.170 → **FAIL**.

**Reading.** Tying the write to the fade means every token that writes also
fades, and the A3 erase is no longer free to be small when the write is
large; the single far fact is lost faster, not slower. The one gain — more
fresh bindings readable right before the query — matches Mamba-2's
pattern inverted: coupling helps "write now, read soon", not "keep for
long". With the R1–R4 levers this is the seventh retention idea that did
not move the far horizon; the only lever that did was the LR (the control's
a1 pos0 @512 0.62 at 600M from scratch).

Also from the control: recall emerges between 150M and 300M tokens from
scratch at LR 1.2e-3; mid-schedule a4/a8 pos1 is non-monotone (0.44 → 0.20
→ 0.27 → 0.22), so single milestones of that cell are noisy.

**Decision.** C1 removed (config, model, train CLI + diag, launcher env,
selftest). Code at `4c688cb`; row in `EXPERIMENT_INDEX.md`. The PAM 3B runs
as Phase 3b at LR 1.2e-3 without C1 (`TAG_FULL=mix3b_delta_answ100_lr1.2e-3`).

## Equal-data scoreboard: PAM vs Mamba-2 vs Transformer++ at 3B tokens (design + pre-registration, 2026-10-28)

**Goal (user).** A PAM model with novel parts that beats transformers and
Mamba at equal data while keeping an O(1) state.

**Design.** Three ~102M models, the EXACT Phase 3b data recipe (3B tok, B18
T2048, warmup 1000, 300M web-only warmup, 0.44/0.41/0.10/0.05, answer weight
100, seed 42), the same trainer, fused CE, holdout + WikiText val, 40-trial
probe and PG-19; each at the LR its own 3-arm from-scratch screen picked
(> 0.01 nats to win). Driver `v13_sempty/tmp_equal_3b.sh` (`ARCH=`).
- **Mamba-2** 102.6M, LR 1.2e-3: done (`b36e3a5`, "Mamba-2 result").
- **Transformer++** 101.9M (`baseline_transformer.py`: RMSNorm, RoPE, SwiGLU,
  tied, causal SDPA; 672 × 12 layers × 12 heads, SwiGLU 1920): LR screen +
  3B run queued after the C1 600M runs. NOT O(1): O(T) KV cache, O(T²)
  prefill; probes past 2048 are RoPE extrapolation (as for the others).
- **PAM** 102.1M = Phase 3b (chrono + out_gate + delta) at LR **1.2e-3**
  (pamlr screen); C1 FAILED (see "C1 result"), so no C1.

**What "beats" means (fixed now):**
- *LM*: holdout loss lower by > 0.01 nats AND WikiText PPL lower.
- *Recall*: mean accuracy over all 63 probe cells higher by > 0.05 (2/40
  trials); a1 / a4 / a8 and pos0 / pos1 reported separately.
- Also reported: PG-19 far_gain, train tok/s (same box), inference state
  per sequence (PAM 16 × 6 × 98² fp32 = 3.7 MB fixed; Mamba-2 17 × 24 × 64 ×
  128 fp32 = 13.4 MB fixed; transformer KV 2 × 12 × 672 bf16 = 32 KB per
  token → 66 MB at 2048, 264 MB at 8192).

**Transformer++ LR screen (`tflr` @`e072a8f`, 12:05–12:51Z, 142K tok/s).**
Same protocol as PAM's and Mamba-2's (100M tok from scratch). Logs
`logs/v13_sempty_tflr_{driver,lr*}_e072a8f*.log`; probes
`logs/memory_probes/v13_sempty_tflr_lr*_e072a8f_t40_behavioral.json`.

| LR | holdout loss | WikiText PPL |
|---|---|---|
| 2e-4 | 4.5543 | 365.4 |
| 6e-4 | 4.1379 | 161.2 |
| **1.2e-3** (pick) | **4.0733** | **153.8** |

At 100M tokens and each model's tuned LR (1.2e-3 for all three): holdout
**PAM 4.0293** < Mamba-2 4.0560 < Transformer++ 4.0733; WikiText **PAM 152.1**
< Transformer++ 153.8 < Mamba-2 183.3. Recall at 100M: the transformer
already reads invented bindings (a1 pos0 @512 0.97, a4/a8 pos1 0.63; means
a1/a4/a8 0.65/0.57/0.49) while PAM is still at chance (0.23/0.16/0.10) —
attention looks up exact tokens from the start; PAM must learn to write them
(it does between 150M and 300M, "C1 result" control). The 3B runs decide.

**Transformer++ 3B result (`mix3b_transformer_answ100` @`e072a8f`,
12:51–18:54Z, 138K tok/s avg).** Logs
`logs/v13_sempty_mix3b_transformer_answ100_{driver_,}e072a8f*.log`, probe
`logs/memory_probes/v13_sempty_mix3b_transformer_answ100_e072a8f_t40_behavioral.json`,
PG-19 `logs/pos_loss/v13_sempty_mix3b_transformer_answ100_e072a8f_pg19.json`.

| 3B tok, equal data | holdout PPL | WikiText PPL | recall mean, 63 cells | recall mean, ctx ≤ 2048 | a1 far (pos0/.5 @1k,2k) | a4/a8 pos1 mean | train tok/s (6000) | state |
|---|---|---|---|---|---|---|---|---|
| Mamba-2 (LR 1.2e-3) | **23.68** | 46.91 | 0.613 | 0.697 | **1.000** | 0.450 | **188.6K** | 13.4 MB fixed |
| Transformer++ (LR 1.2e-3) | 24.69 | **46.61** | 0.539 | 0.693 | 0.806 | 0.523 | 138.4K | 32 KB/token |
| PAM Phase 3b (old LR 2e-4, `1c913ef`) | 26.38 | 54.73 | **0.667** | **0.772** | 0.412 | **0.900** | 66.5K | **3.7 MB fixed** |
| PAM Phase 3b @1.2e-3 (`9982a79`, pre-registered arm) | 26.69 | 55.83 | 0.602 | 0.697 | 0.487 | 0.782 | 65.5K | 3.7 MB fixed |

- The transformer reads exact tokens perfectly up to 1024 and fails past
  its 2048 training length: a1 pos1 @4096 0.23, PG-19 loss 3.55 → 4.28 →
  5.54 on [1k,2k) / [2k,4k) / [4k,8k) (far_gain +0.018 / −0.70 / −1.92;
  RoPE does not extrapolate). Mamba-2 +0.017/+0.026/+0.031, PAM
  +0.014/+0.019/+0.022.
- Even inside 2048 the transformer's multi-fact recall (a4 0.50, a8 0.41
  means) is below PAM's (0.72 / 0.57). Attention can see the facts but the
  100M model does not reliably pick the right one among 4–8.
- LM: Mamba-2 best holdout, the transformer best WikiText by 0.3 PPL. The
  old PAM trails both; the 1.2e-3 rerun decides (at 100M tok PAM led both).

## Scoreboard result — PAM does NOT beat either baseline on LM; beats Transformer++ on recall; the 100M LR screen picked the wrong LR for PAM (2026-10-29)

PAM 3B at LR 1.2e-3 (`mix3b_delta_answ100_lr1.2e-3` @`9982a79`, 18:54Z →
07:40Z, 65.5K tok/s avg). Logs
`logs/v13_sempty_mix3b_delta_answ100_lr1.2e-3_{driver_,}9982a79*.log`, probe
`logs/memory_probes/v13_sempty_mix3b_delta_answ100_lr1.2e-3_9982a79_t40_behavioral.json`,
PG-19 `logs/pos_loss/v13_sempty_mix3b_delta_answ100_lr1.2e-3_9982a79_pg19.json`.
Table rows: "Transformer++ 3B result" above.

**Pre-registered verdict (PAM arm = 1.2e-3 run):**

| PAM vs | LM: holdout Δ nats (need < −0.01) and WikiText | recall: 63-cell Δ (need > +0.05) |
|---|---|---|
| Mamba-2 | +0.120 (3.2842 vs 3.1644), WikiText 55.83 vs 46.91 → **LOSE** | −0.011 (0.602 vs 0.613) → **tie** |
| Transformer++ | +0.078 (vs 3.2064), WikiText 55.83 vs 46.61 → **LOSE** | +0.063 (vs 0.539) → **BEATS** |

Not pre-registered, for reference: PAM at LR 2e-4 (`1c913ef`) is better than
the 1.2e-3 run on everything — holdout 3.2728 (−0.011), WikiText 54.73,
recall 0.667 (beats Mamba-2 by +0.054, Transformer++ by +0.128) — but still
LOSES LM to both (+0.108 / +0.066 nats).

**Why the LR pick was wrong.** Holdout loss, PAM 1.2e-3 vs 2e-4: step 2000
4.185 vs 4.654 (high LR far ahead; the 100M screen ends at step 2700), step
6000 3.989 vs 3.858 (already behind), then behind to the end (+0.011 at
80000). Mamba-2 and Transformer++ at 1.2e-3 lead PAM-2e-4 at every val.
So PAM at high LR learns fast early then plateaus worse; the baselines do
not. Rule from now on (EXPERIMENT_INDEX): **a 100M screen cannot pick PAM's
LR or judge a PAM change; screen PAM on ≥ 600M tokens** (the C1 r600
schedule already did this) and compare curves past step 6000.

**Where PAM stands at equal data (3B tok, ~102M):**
- LM: behind Mamba-2 by ~0.11 nats, behind Transformer++ by ~0.07. This gap
  is 10× the LR effect — it is architectural, not tuning.
- Recall: best multi-fact recall of the three (a4/a8 pos1 0.90 vs 0.45 /
  0.52 at LR 2e-4); worst far single-fact recall (a1 far 0.41 vs 1.00 /
  0.81). Transformer++ fails past its 2048 training length.
- State: 3.7 MB fixed vs 13.4 MB fixed (Mamba-2) vs 32 KB/token.
- Speed: 66K tok/s vs 189K / 138K (6000) — a 2.1–2.9× gap.

## G screen: close the LM gap with the Gated-DeltaNet parts PAM lacks (design + pre-registration, 2026-10-29)

**Why these.** PAM is Gated DeltaNet's recurrence (fade × delta erase/write,
unit keys) plus Chrono rotary and the N4 gate, inside a SwiGLU block. It is
missing two parts every strong delta / SSM LM has:
- **G1 short conv** — depthwise causal `Conv1d(k=4)` over time on fused qkv,
  residual + zero-init (`qkv + silu(conv(qkv))`, identity at start). Each
  key/value then encodes a 4-token phrase, not one token. A1 tested exactly
  this on **10-epoch WikiText** and FAILED by overfitting (train loss better,
  val worse); in **single-pass pretraining** overfitting cannot happen, so
  the A1 verdict does not carry over. Code reproduces A1's `2537782`.
- **G2 read norm** — per-head RMSNorm of y = S q (fp32, learned per-channel
  scale, init 1) before the N4 gate. The read's size today depends on |q| and
  on how full S is (it changes with context length and decay); the norm
  removes both (y/‖y‖ is invariant to q's scale, so it also covers q-norm).

**Arms** (one variable each, then the pair), 600M-token schedule on the
6000 at LR 1.2e-3 — the r600 protocol (`91daf0a`: TARGET 599,998,464, val
every 4069), so the finished **r600 control is reused** (C1 code since
removed, off-path identical):

| arm | change | step time (4090, B4) | params |
|---|---|---|---|
| ctl (`r600_ctl_91daf0a`) | — | 166.7 ms | 102.12M |
| `conv` | G1 | 192.9 ms (+16 %) | 102.26M |
| `onorm` | G2 | 176.7 ms (+6 %) | 102.13M |
| `both` | G1 + G2 | 197.5 ms (+18 %) | 102.27M |

Control at 600M: holdout **3.5408** (PPL 34.50), WikiText **75.89**, recall
63-cell mean **0.343** (a1/a4/a8 0.613/0.295/0.121).

**Decision rule (fixed now):**
- PASS = holdout ≤ **3.5208** (−0.02 nats; the scoreboard gap is 0.07–0.11)
  AND WikiText < 75.89 AND recall 63-cell mean ≥ 0.293 (not −0.05).
- `both` is taken over a single arm only if it beats it by > 0.01 nats.
- A PASS goes to a 3B confirm at LR **2e-4** (PAM's best 3B LR), compared with
  `1c913ef` (3.2728). The conv's cost is cut first with a fused causal-conv
  kernel if it passes.
- FAIL → removed, row in EXPERIMENT_INDEX (A1 row gets the pretraining-
  regime result too).
- Not a decode-ready model: G1 has no decode state yet (GEN_EVERY 0;
  val, probes, PG-19 use the chunked path). Added only if G1 passes.

Self-test `test_short_conv_and_read_norm` (conv init parity 0, causal,
grads reach every conv/norm, out_norm decode == chunked 7e-7): 21/21 pass.

Control PG-19 (run on the 4090 from `r600_ctl_91daf0a/best_model.pt`,
`logs/pos_loss/v13_sempty_r600_ctl_91daf0a_pg19.json`): loss [1k,2k) /
[2k,4k) / [4k,8k) 3.8897 / 3.8850 / 3.9201, far_gain +0.011 / +0.016 /
+0.018.

### G2 read norm — stopped for futility at step 2250 (2026-10-29)

`gscreen_onorm` @`3cc2b72`: train loss minus the control's on the SAME
batches (same seed and stream), mean per 250-step window: steps 1–250
−0.059, 251–500 −0.001, 501–750 +0.052, 751–1000 +0.072, 1001–1250 +0.134,
1251–1500 +0.163, 1501–1750 +0.168, 1751–2000 +0.129, 2001–2250 +0.137. A
pass needs −0.02 at 600M; a +0.14 gap that is not closing makes that
implausible, so the arm was stopped (as C1 was) to give the GPU to E1.
Likely why: the norm fixes every head's read at unit RMS, so the N4 gate
and `pam_scale` (0.1 init) — not the memory's own confidence — set how much
memory reaches the residual; an empty/irrelevant notebook now speaks as
loudly as a full one. Verdict **FAIL (futility)**; code removed after the
E1/G1 screen (`e1g`, pinned `a0fab65`) finishes.
G1 conv now runs as the 2nd arm of screen `e1g` (after echo).

## E1 Echo memory — an O(1) succession memory into the head (novel; design + pre-registration, 2026-10-29)

**Where it came from.** Every failed lever so far (R1–R4, C1, L-2) tried to
make the notebook keep things *longer*. But the LM gap to Mamba-2 /
Transformer++ (0.07–0.11 nats) is mostly *now*: repeated names, phrases,
copying — what a transformer's induction head does (find the earlier
occurrence of this moment, emit what came after it), and exactly what the
recall probe asks. PAM never stores *what came next*; it stores token →
token-features inside the stack and must rebuild succession indirectly.
Biology does store it directly: spike-timing plasticity strengthens the link
from the neuron that fired *before* to the one that fired *after*.

**Math.** One extra delta memory on the final hidden h (after output_norm),
6 heads × 98 tiling dim 588:

    k_t = unit(W_k h_{t-1})      (k_0 = 0: no write at t=0)
    v_t = E[x_t] / rms(E[x_t])   (the tied embedding of the token that came)
    q_t = W_q h_t
    S_t = g_t S_{t-1} (I − b_e k_t k_tᵀ) + b_w v_t k_tᵀ        (A3 form, pam_delta_batched)
    r_t = rmsnorm_head(S_t q_t);   lm_t = head(h_t) + W_o r_t,   W_o = 0 at init

so `E · lm_t` gains the logit of "the word that followed the last moment
like this one" — a learned, content-addressed neural cache (Grave et al.'s
continuous cache) with a FIXED d×d state instead of an O(T) cache. Gates
g, b_w, b_e read h_t (causal: h_t has seen x_t); echo retention init
e^{−softplus(−6)} ≈ 0.9975. Differs from A4 (static hashed n-gram table, no
context) and R4 (a second notebook of the same kind inside a block).

**Cost.** +1.05M params (103.17M, +1 %); 4090 B4 step 167.7 → 174.9 ms
(+4 %). Self-test `test_echo_memory`: identity at init, causal, grads flow,
and a hand-wired copy check (the second occurrence of x3 reads back x4).

**Run.** `e1` screen, arm `echo` (ECHO=1), the r600 protocol (600M tok, LR
1.2e-3) vs the same `r600_ctl_91daf0a` control, queued on the 6000 after
the G screen (`~/e1_chain.sh`, tmux `e1chain`; G's `both` arm was skipped
to start E1 ~3 h sooner — it runs later only if conv AND onorm pass).

**Decision rule (fixed now):** same as G — PASS = holdout ≤ 3.5208 AND
WikiText < 75.89 AND recall 63-cell mean ≥ 0.293; the expected signature is
a1 far recall (pos0/.5 @1k–2k, control 0.412) going up. PASS → 3B confirm
at LR 2e-4 (+ a decode path). FAIL → removed with an index row.

### E1 micro-bench: echo solves 4- and 8-way recall in 2000 steps; plain delta is at chance (2026-10-29, 4090)

`v13_sempty/tmp/recall_microbench.py --steps 2000 --batch 16 --arms delta,echo`
(27.6M, 4 layers, hd 64; same data/seed per arm; code `86baea2`). Log
`logs/v13_sempty_echo_microbench_86baea2_20261029_1340.log`.

| arm | a1 (128–8192) | a4 | a8 |
|---|---|---|---|
| chrono+gate+delta | 1.00 all | 0.18–0.26 | 0.12–0.21 (chance) |
| **+ echo** | 1.00 all | **1.00 all** | **1.00 all** |

The delta reference needed 6000 steps to reach this ("Positive control");
echo gets there in ≤ 2000 (train loss 0.0000 by step 1800) and holds to
8192 (4× its longest train doc). Mechanism confirmed: a succession memory
read straight into the head is a one-step induction head. Caveat: a pure
recall task; the LM verdict is the `e1g` 600M screen.

### E1 at scale — milestone log (`e1g_echo` @`a0fab65`, 6000, 600M schedule)

| tokens | holdout (echo / ctl) | Δ nats | WikiText PPL (echo / ctl) |
|---|---|---|---|
| 150M (step 4069) | 3.9516 / 4.0195 | **−0.068** | **119.64 / 148.56** (−19 %) |
| 300M (step 8138) | 3.6955 / 3.7555 | **−0.060** | **87.99 / 100.38** (−12 %) |
| 450M (step 12207) | 3.5373 / 3.5920 | **−0.055** | **75.66 / 83.01** (−9 %) |
| 600M (step 16276) | **3.4885 / 3.5408** | **−0.052** | **71.09 / 75.89** (−6 %) |

**Verdict (2026-10-29): E1 PASS at 600M — echo is now part of the PAM
default recipe.** All three axes improve at equal data (+1.05M params, 1 %):

- Holdout −0.052 nats, WikiText −6 % PPL.
- Recall (40-seed behavioral probe, mean accuracy): **0.401 vs 0.343**.
- PG-19 (50 books × 8192 tok) loss by position, echo / ctl:
  0–64 4.267 / 4.190 (worse: cold start, empty echo memory),
  64–128 3.925 / 3.952, 128–256 3.738 / 3.753, 256–512 3.715 / 3.738,
  512–1k 3.799 / 3.841, 1–2k 3.821 / 3.890, 2–4k 3.818 / 3.885,
  4–8k **3.849 / 3.920**. Far-context gain (full − 512-window loss) at 4–8k:
  **0.044 vs 0.018** (2.4×) — echo gets better the longer the book.
- Generations (seed 0, T 0.8, top-k 50, no rep. penalty, 80 tok, 600M ckpts):
  prompt "Photosynthesis is the process by which" — ctl loops ("Plants have
  large spines. / Plants have small spines. / …"), echo stays on topic
  ("…plants convert carbon dioxide and water into oxygen…"). Prompt "The
  history of the city began in" — echo repeats the name "San Diego" ×5
  (the copy habit shows in names, no hard loop).
- Gain shrinks with tokens (−0.068 → −0.052), the usual pattern; it must be
  re-checked at 3B before the scoreboard claim.

Logs: `logs/v13_sempty_e1g_echo_a0fab65_20261029_0817.log` (md5 0ee692e8…),
`logs/memory_probes/v13_sempty_e1g_echo_a0fab65_t40_behavioral.json`,
`logs/pos_loss/v13_sempty_e1g_echo_a0fab65_pg19.json` (md5 166384bf…).

Train loss on the same batches, echo − ctl per 250-step window to step
2250: −0.02, −0.44, −0.36, −0.41, −0.07, −0.24, −0.04, −0.21, −0.19 (the
train loss includes recall docs with answer weight 100; the holdout is pure
DCLM-Edu and WikiText has no recall docs, so the holdout/WikiText gains are
LM, not recall). Speed 64K tok/s (ctl 66K).

## M1 sparse memory — "few memories, deep thought" (design + pre-registration, 2026-10-29)

**Idea.** The 16 per-layer delta notebooks are PAM's slow part (flat profile
of small solves), and E1 showed that ONE well-placed memory carries a lot.
A brain does not put a hippocampus in every cortical layer. M1 keeps
notebooks in blocks 3, 7, 11, 15 only; the other 12 blocks are pure gated-MLP
"thinking" blocks (GEMM-bound); the saved params widen every MLP
(hidden 1764 → 2352) so the total is matched; echo stays on top.

| model | params | step (4090, B4) |
|---|---|---|
| PAM (16 notebooks) | 102.12M | 173.3 ms |
| PAM + echo | 103.17M | 177.2 ms |
| **M1: 4 notebooks + echo + MLP 2352** | **102.95M** | **98.7 ms (1.8×)** |

**Next step it enables (the user's framing: "memory is not the brain; a
brain that thinks hard and can reach memory at every level").** M2 = ONE
shared notebook written once low in the stack and READ by every block above
it with its own query/gate — the write (WY solve) is the cost, the read is
cheap. Writes must come from a low block so higher blocks can read in
parallel training (a top-written memory read by lower blocks would make
training sequential).

**Run.** Screen `m1`, arm `m1echo` (ECHO=1, PAM_LAYERS=3,7,11,15,
MLP_HIDDEN=2352), r600 protocol, queued after `e1g` (tmux `m1chain`).
`e1g`'s conv arm was skipped (marker) to give the GPU to M1.

**Decision rule (fixed now):** compared with `e1g_echo` (full PAM + echo,
same schedule): KEEP if holdout ≤ echo + 0.01 nats AND WikiText ≤ echo +
3 % AND recall 63-cell mean ≥ echo − 0.05 — i.e. same quality at 1.8× the
speed. Better than echo on holdout → it is the new reference outright.

**M1 milestone log** (`m1_m1echo_ccbf028`, 6000, 116K tok/s vs echo 64K):

| tokens | holdout (M1 / echo) | WikiText (M1 / echo) |
|---|---|---|
| 150M (step 4069) | **3.9225 / 3.9516** (−0.029) | **109.70 / 119.64** |
| 300M (step 8138) | **3.6829 / 3.6955** (−0.013) | **84.60 / 87.99** |
| 450M (step 12207) | **3.5348 / 3.5373** (−0.003) | **73.90 / 75.66** |
| 600M (step 16276) | 3.4906 / 3.4885 (+0.002) | **69.44 / 71.09** (−2.3 %) |

**Verdict (2026-10-29): M1 KEEP — the new reference body.** All three
pre-registered rules hold (holdout +0.002 ≤ +0.01; WikiText −2.3 % ≤ +3 %;
recall far above echo − 0.05), at **1.8× the speed** (116K vs 64K tok/s).

- **Recall 0.615 vs echo 0.401** (40-seed probe, 63-cell mean). The gain is
  multi-fact recall: accuracy by context × facts (M1 / echo):

  | ctx | 1 fact | 4 facts | 8 facts |
  |---|---|---|---|
  | 128 | 1.00 / 1.00 | **0.57** / 0.18 | **0.41** / 0.07 |
  | 512 | 0.99 / 0.98 | **0.70** / 0.21 | **0.27** / 0.08 |
  | 2048 | 0.94 / 0.90 | **0.67** / 0.23 | **0.36** / 0.12 |
  | 8192 | 0.62 / 0.61 | **0.48** / 0.22 | **0.28** / 0.12 |

  Fewer, deeper-placed notebooks hold several facts far better than 16
  shallow ones (hypothesis: with 16 notebooks at pam_scale 0.1 the recall
  signal is spread thin; with 4 each must become a real memory, and the
  wider MLPs between them have more room to use what is read).
- PG-19 loss by position (M1 / echo): 0–64 4.267 / 4.267, 1–2k 3.848 /
  3.821, 4–8k 3.872 / 3.849 (+0.023, slightly worse); far gain at 4–8k
  0.0446 / 0.0440 (same).
- Same-batch train loss was ~0.09 worse than echo early (recall-weighted);
  the holdout told the true story.

Logs: `logs/v13_sempty_m1_m1echo_ccbf028_20261029_1100.log` (md5 d6e5c816…),
`logs/memory_probes/v13_sempty_m1_m1echo_ccbf028_t40_behavioral.json`,
`logs/pos_loss/v13_sempty_m1_m1echo_ccbf028_pg19.json` (md5 147bd888…).

Lesson: on the SAME batches M1's train loss was ~0.09 *worse* than echo at
steps 1750–2500, yet its holdout is better. The train loss carries the
recall docs at answer weight 100, so for arms that change the memory it
mostly measures recall training, not LM. Same-batch train loss is an early
signal only; the 150M holdout val is the first real verdict.

## M2 shared reads — one write, read at every level (design + pre-registration, 2026-10-29)

**Idea (user: "memory is not the brain; a brain that thinks hard and can
access memory at each level").** On top of M1, every notebook-less block
above the first notebook READS the nearest notebook below it:

    y_t = silu(W_g x_t + b) * (S^w_t . rot_w(W_q x_t)) / sqrt(d),   out = W_o y   (W_o = 0 at init)

`S^w` is writer w's delta notebook, `rot_w` its Chrono rotation (the query
meets the keys in the writer's time frame). The cost of a notebook is the
WRITE (WY solve); the read is cheap and exact via the pseudo-values: the
delta recurrence over (k, W, g) is additive, so one `fused_real_pam_read`
gives S_t q_t for any q (`pam_shared_read`; self-test: equals the writer's
own read to 0.0). Readers' gradients flow into the writer, so a notebook
learns to store what every level above it needs. Writes stay low in the
stack: a top-written memory read by lower blocks would make training
sequential in time.

| model | params | step (4090, B4) |
|---|---|---|
| full PAM + echo | 103.17M | 174.4 ms |
| M1 (4 notebooks + echo, MLP 2352) | 102.95M | 99.6 ms |
| **M2 (4 notebooks, 9 readers, echo, MLP 2128)** | **102.88M** | **117.7 ms (1.5×)** |

**Run.** Screen `m2`, arm `m2echo`, r600 protocol, queued after `m1`
(tmux `m2chain`, `~/m2_chain.sh`).

**Decision rule (fixed now):** M2 KEEP over M1 if holdout ≤ M1 − 0.01 nats
OR recall 63-cell mean ≥ M1 + 0.05 (with holdout ≤ M1 + 0.01); either one,
vs full PAM + echo, the same "quality at ≥ 1.5× speed" bar as M1.

**M2 result (2026-10-29, 6000, `ed326a1`, 600M): KEEP.** Holdout 3.4697 vs
M1 3.4906 (−0.021 ≤ −0.01 → rule met); WikiText 68.84 vs 69.44; PG-19 4–8k
far gain 0.0414 vs 0.0446. Not in the rule but reported: recall 63-cell
0.522 vs M1 0.615 (a1/a4/a8 0.774/0.596/0.196 vs 0.895/0.624/0.326) — the
shared reads buy LM quality and cost single-fact recall; learning probe
0.185 (M1 0.175). Milestones (holdout): 3.9054 / 3.6602 / 3.5146 / 3.4697 at
150/300/450/600M (M1 3.9225 / 3.6829 / 3.5348 / 3.4906): the lead holds
(−0.017, −0.023, −0.020, −0.021), unlike echo, M1 and T1. 1.18× slower than M1.

## Speed: where PAM's 2.9× gap to Mamba-2 is (2026-10-29, 4090, B4 T2048)

- **Not width alignment.** Step time by geometry (all chrono+gate+delta):
  588/6×98 166.6 ms (102.1M) · 576/6×96 159.4 ms (98.6M) · 576/9×64 148.6 ms
  (98.7M) · 640/10×64 169.4 ms (118.3M) · 768/6×128 ×12 layers 173.4 ms
  (131.5M). Aligned widths buy ~4 %, mostly from fewer params.
- **Profile is flat and GPU-bound** (161 ms CUDA per 166 ms step): the
  biggest single kernel is 4.6 %; ~1,500 tiny GEMMs/solves (4 µs each) and
  thousands of elementwise launches (5–30 µs) from the chunked delta (WY,
  trsm 5.4 %), gates, Chrono rotary and norms. No hotspot to hand-tune — the
  fix is one fused chunked delta kernel.
- **Candidate: flash-linear-attention's `chunk_gated_delta_rule`** (Gated
  DeltaNet's fused Triton kernel, Mamba-2-class speed). PAM maps onto it
  exactly: S_t = g S(I − b_e k kᵀ) + b_w v kᵀ = g S(I − β k kᵀ) + β v' kᵀ
  with β = b_e, v' = (b_w / b_e) v, log-decay = log g, unit keys, scale
  d^{-1/2}. Not installed on either box (needs Python ≥ 3.10; the 4090 venv
  is 3.9, the 6000 is 3.11). Installing third-party kernels needs the user's
  OK — pending.

## Engineering: batch prefetch + the diag panel's hidden data skip (2026-10-27, `129b406`)

Found while the Mamba-2 baseline ran data-bound (98K tok/s of a possible
~170K; GPU idle between steps).

1. **Inline batches.** The streaming loader is single-process, so every
   batch was tokenized while the GPU waited: producer 0.165 s/batch (224K
   tok/s, 6000) against a 0.21 s Mamba-2 step and a 0.625 s PAM step (B18
   T2048). `train._Prefetch` builds batches on one background thread (FIFO,
   `--prefetch 4`): identical batches and order (selftest
   `test_prefetch_order_and_cursor`; 4090 PAM run: same loss at all 12 steps
   and the same saved cursor, prefetch 0 vs 4). Checkpoints save the cursor
   snapshot taken with the last batch trained on.
2. **The diag panel skipped ~20M tokens per call.** `_diagnostics` took its
   input from a fresh `iter(train_loader)`. The streaming dataset wraps ONE
   shared text generator, so the new iterator refilled its own 10,000-chunk
   shuffle buffer from the live stream, yielded one batch and was dropped:
   measured **107 s and 21,621 documents (~20M tokens) per call**, never
   trained on. With `--diag_every 2000` that is 40 calls per 3B run: Phase 3b,
   R3, L-2 and both R4 arms trained on a stream with ~40 contiguous ~20M-token
   gaps (~0.9B tokens of documents read and discarded per 3B run) and lost
   ~40 s per call (R4 ctrl: steps 2050/4050/6050 at 23K vs 59K tok/s).
   Consequences: no document was repeated, the source mix is unchanged
   (realised mix checks still hold), and arms of one screen saw identical
   streams — so no past comparison is invalidated; the "same tokens as Phase
   3b" claim holds only up to step 2000. Fixed: the panel uses the last
   training batch.

## Positioning — is this Mamba? (2026-09-04)

No, and not a Mamba variant. Mamba (S6) is a **diagonal SSM**: a *vector*
state `h_t = A_t ⊙ h_{t-1} + B_t x_t`, `y_t = C_t h_t`, with input-dependent
`A,B,C` and a hardware-aware **selective scan**. v13_sempty is an
**outer-product associative memory**: a *matrix* state
`S_t = γ_t S_{t-1} + v_t ⊗ (conj) k_t`, `read = q·S_t`, with a scalar per-head
decay `γ_t` and a closed-form chunked cumprod/cumsum (no selective scan). It
fails both of Mamba's defining tests.

It **does** belong to the broad linear-attention / fast-weight super-family
(shared by linear Transformers, RetNet, GLA, DeltaNet, RWKV — and Mamba):
one O(1) recurrent state, linear in sequence, O(1) at decode. Within that
family:
- the **real arm** (the 23.81 result) is closest to **RetNet / lightweight
  GLA** — real outer-product state + scalar head decay. Honest: a competitive
  member of gated-linear-attention, not a from-scratch paradigm.
- the **novel contribution** is the **complex phase-associative addressing**
  (write `conj(k)`, read raw `q` → phase-matched exact recall), which is the
  **Holographic Reduced Representations / VSA-HDC** lineage (Plate, Kanerva),
  *not* the SSM lineage; plus the matrix notebook with read-after-write and
  the O(1)-recall research framing (cf. the pands VSA memory).

Claim to make in writeups: "not Mamba, not a Mamba variant; a phase/complex
associative-memory member of the linear-recurrent family, competitive with a
transformer at 100M/10ep on WikiText-103." Do **not** claim "unlike anything"
— the O(1) recurrent form is a populated family.

## T1 thought stream — "two-speed mind" (design + pre-registration, 2026-10-29)

**Why.** Echo, M1 and M2 are still a token-by-token predictor. A mind does
not think in tokens: it reads them, but it thinks in events, on a slower
clock. T1 is the simplest testable form of that: a second, slow stream.

**Mechanism (`ea002e6`, `--thought`).**
1. The main stack reads tokens (blocks 0..5).
2. Every 16 tokens the chunk is pooled into one event,
   e_c = W_in [mean(h_chunk); h_last] (h = output of block 5).
3. A separate stack of 4 PAM blocks (the thinker: its own delta memory, its
   own Chrono clock) runs over the event sequence e_1..e_N (N = T/16).
4. Its thought after event c, W_out rmsnorm(t_c) (W_out zero-init), is added
   to the residual of every token of chunk c+1, before main block 6 — so it
   reaches all upper blocks (top-down). Causal at chunk granularity; training
   stays fully parallel (the thinker reads low, writes high).

Param-matched arm: 12 main blocks + 4 thinker blocks + echo = 104.20M vs the
echo arm 16 blocks 103.17M (+1 %); 4090 bench 143 vs 176 ms/step (1.23×
faster: the 4 thinker blocks run on 16× fewer positions).

**Prior art (honest).** The two-stream / fixed-chunk shape is close to the
Temporal Latent Bottleneck (Didolkar 2022), Block-Recurrent Transformer
(Hutchins 2022), MEGABYTE / Hourglass (global model over patches). What is
new here: both streams are O(1) delta memories (no attention anywhere), the
thinker writes top-down into the token stack, and it is meant as step one
toward surprise-timed events (echo's prediction error sets the boundary)
and event-level prediction targets. T1 tests only the base claim: **does a
slow event stream beat the same params spent on token-level blocks?**

**Pre-registered rules** (control = `e1g_echo_a0fab65`, same batches, same
600M schedule, LR 1.2e-3):
- Early (local 4090, `t1loc_t1_ea002e6`, GRAD_CKPT=1, same stream): mean
  train-loss difference T1 − echo over steps 1750–2250 ≤ +0.02 → promote to
  the full 600M screen on the 6000 (after M2). > +0.05 and not closing →
  FAIL for futility.
- Also reported (added 2026-10-29 11:30Z, before T1 reached it, after the
  M1 lesson above): T1's 150M holdout vs echo 3.9516 — this outranks the
  train-loss rule if they disagree.
- Full 600M: PASS if holdout ≤ 3.4885 (echo) — equal quality at 1.23× speed
  is already a win; recall ≥ 0.38; PG-19 4–8k far gain ≥ 0.035.

**T1 early check (2026-10-29 11:52Z, local 4090, 38.9K tok/s with grad ckpt):**
train loss T1 − echo on the same batches, 250-step windows to step 2250:
+0.19, +0.13, +0.09, +0.10, +0.00, +0.11, **−0.08, −0.02, +0.01** — starts
behind (4 fewer token blocks) and closes; pre-registered window 1750–2250
mean **−0.03 ≤ +0.02 → early PASS, promote to the 600M screen.** (The run
keeps going locally to the 150M holdout val.)

**T1 milestone log** (local 4090, same 600M schedule as echo):

| tokens | holdout (T1 / echo / M1) | WikiText PPL (T1 / echo / M1) |
|---|---|---|
| 150M (step 4069) | **3.9204** / 3.9516 / 3.9225 | 122.70 / 119.64 / **109.70** |
| 300M (step 8138) | **3.6810** / 3.6955 / 3.6829 | 85.85 / 87.99 / **84.60** |

Mixed at 150M: best holdout (−0.031 vs echo, in-distribution DCLM-Edu) but
WikiText +2.6 % (out of distribution). Not a rule trigger (T1 is judged on
holdout, recall, PG-19); watch whether the WikiText gap closes.
At 300M the WikiText gap closed (T1 now −2.4 % vs echo, was +2.6 %); the
holdout lead shrank −0.031 → −0.015 (same shrinking pattern as M1 and echo).

| tokens | holdout (T1 / echo / M1) | WikiText PPL (T1 / echo / M1) |
|---|---|---|
| 450M (step 12207) | 3.5414 / 3.5373 / **3.5348** | 73.12 / 75.66 / 73.90 |
| 600M (step 16276) | 3.4977 / **3.4885** / 3.4906 | 69.64 / 71.09 / **69.44** |

**T1 verdict (2026-10-29): FAIL.** Holdout 3.4977 > 3.4885 (+0.009; the
lead went −0.031, −0.015, +0.004, +0.009). Recall 0.539 ≥ 0.38 PASS
(a1/a4/a8 0.843/0.417/0.357, 8-fact above M1's 0.326); PG-19 4–8k far gain
0.0448 ≥ 0.035 PASS (best of the 600M arms, tied with M1 0.0446). Learning
probe 0.171. Reading: the slow event stream helps early (it is a shortcut
to longer context) but the same params as token blocks win by 600M. The
idea it was meant to lead to (surprise-timed events, event-level targets)
is untested. **Code removed 2026-10-30**; the last commit with it is
`5eff3b9` (rerun recipe in EXPERIMENT_INDEX "Failed and removed").

## Learning probe (goal bar 3: learning beyond recall) — baselines (2026-10-29)

**Why.** The user's goal: "a different kind of intelligence.. not just recall
of things but true learning from the data"; a transformer "is learning data,
not intelligence". LM loss and the recall probe cannot show this. New probe
(`memory_probes/rules.py`, CLI `scripts/run_learning_probe.py`, `ed326a1`):
few-shot examples of a hidden rule, then a NEW input — letter_shift and
number_step answers never appear in the context; reverse3 (positional rule),
label_category (invented label per category, query never labelled), two_hop
(owner → pet → name). 8 single-token candidates incl. tempting wrong answers
(chance 0.125); 100 seeds × shots 2/4/8; CPU.

| model | letter_shift | number_step | reverse3 | label_category | two_hop | mean |
|---|---|---|---|---|---|---|
| PAM ctl 600M (`r600_ctl_91daf0a`) | 0.053 | 0.023 | 0.023 | 0.573 | 0.310 | 0.197 |
| PAM + echo 600M (`e1g_echo_a0fab65`) | 0.023 | 0.007 | 0.010 | 0.553 | 0.307 | 0.180 |
| M1 600M (`m1_m1echo_ccbf028`) | 0.047 | 0.023 | 0.007 | 0.537 | 0.263 | 0.175 |
| PAM 3B (`1c913ef`) | 0.033 | 0.053 | 0.093 | 0.590 | 0.297 | 0.213 |
| Transformer++ 3B (`e072a8f`) | 0.130 | 0.053 | 0.070 | 0.627 | 0.260 | 0.228 |
| Mamba-2 3B (`b36e3a5`) | 0.120 | 0.067 | 0.027 | 0.517 | 0.280 | 0.202 |
| M2 600M (`m2_m2echo_ed326a1`) | 0.090 | 0.040 | 0.010 | 0.503 | 0.283 | 0.185 |
| T1 600M (`t1loc_t1_ea002e6`) | 0.057 | 0.020 | 0.003 | 0.507 | 0.267 | 0.171 |
| C1a stage A 120M (`c1a_pre_5b1eb9d`) | 0.063 | 0.113 | 0.003 | 0.497 | 0.297 | 0.195 |
| C1b @9766 (`c1b_main_5b1eb9d`) | 0.043 | 0.080 | 0.023 | 0.550 | 0.300 | 0.199 |

**Reading.** No model — the transformer included — applies an in-context
rule. On letter/number/reverse they sit at or BELOW chance because they
follow training priors (error analysis, 4 shots, 60 seeds): number_step →
"last + 1" in 54/60 (M1) and 43/60 (Transformer++), i.e. they count by one
and ignore the step shown; letter_shift → mostly copy the query; reverse3 →
repeat the wrong word. two_hop equals in-context chance at every shot count
(0.51 / 0.27 / 0.15 vs 0.50 / 0.25 / 0.125): no composition. Only
label_category is above in-context chance, modestly. Echo / M1 gain recall
(previous sections) but not rule learning. **Bar 3 is open territory:** a
model that truly applies a rule (e.g. number_step > 0.5) separates from
every baseline. This is now the third scoreboard column.

## C1 reason-first curriculum on REAL aptitude data (design + pre-registration, 2026-10-29)

**Idea (user).** "First understand grammar a bit, then complex questions
like aptitude, then chat that has complex relations, so the system learns
reasoning first, and then we feed data." Real datasets only — no synthetic
generators (the v12 synthetic reason/recall curricula were done before).
Prior evidence that order can matter: curriculum learning (Bengio 2009),
formal-language pre-pretraining (Hu et al. 2025), LIME (Wu et al. 2021),
textbook-quality data (Phi, 2023). Risk: later web data washes it out.

**Data.** Source `aptitude` (`v7/aptitude.py`, `138cddb`): TRAIN splits of
14 real datasets — AQuA-RAT (GMAT/GRE with rationale), GSM8K, LogiQA 2.0,
ReClor, RACE, ARC, CommonsenseQA, OpenBookQA, SciQ, QASC, BoolQ,
StrategyQA, WinoGrande, HellaSwag: 276K docs, ~48.8M tokens. Test splits
untouched.

**Schedule (M1 body, 600M tokens total = M1's budget).**
- Stage A `c1a_pre` (120M tok, LR 1.2e-3 cosine): first 30M web only
  (BLEND_WARMUP, "a bit of grammar"), then token shares dclm 0.25, fineweb
  0.20, smoltalk2_mid 0.15 (chat), aptitude 0.40 — the token-deficit
  blender serves aptitude first after the warmup, then chat joins; the
  90M reasoning phase is ≈ 53 % aptitude (≈ 1 epoch), 20 % chat, 27 % web.
- Stage B `c1b_main` (480M tok, `--init_from` stage A, fresh LR warmup +
  cosine): EXACTLY M1's mix (dclm/fineweb/smoltalk2_mid/recall 0.44/0.41/
  0.10/0.05); sources start at never-read files.

**Pre-registered rules** (control = M1 `m1_m1echo_ccbf028` at 600M):
- Primary: learning-probe mean (invented rule tasks, nothing like them in
  the aptitude data, so this measures transfer of reasoning) ≥ M1 + 0.05
  (M1 0.175 → ≥ 0.225; SE ≈ 0.01 at 1500 prompts).
- Guards: holdout ≤ M1 + 0.03 (≤ 3.52; stage B sees 120M fewer normal-mix
  tokens), recall ≥ M1 − 0.05 (≥ 0.565).
- Reported, not a rule: stage-A probes (does pre-training alone change the
  learning probe?), WikiText, PG-19.

### C1 result (2026-10-29, 6000, `5b1eb9d`) — primary FAIL; but recall jumps 0.615 → 0.907 (unexpected, needs a control)

| model | holdout | WikiText | recall 63-cell (a1/a4/a8) | learning probe | PG-19 4–8k far gain |
|---|---|---|---|---|---|
| M1 600M (control) | 3.4906 | 69.44 | 0.615 (0.895/0.624/0.326) | 0.175 | 0.0446 |
| C1a stage A (120M) | 4.1194 | 125.88 | 0.417 (0.815/0.337/0.100) | 0.195 | 0.0421 |
| C1b @ step 9766 (486M total) | 3.5397 | 76.00 | **0.907 (0.919/0.948/0.855)** | 0.199 | 0.0428 |

**Caveat.** C1b validated last at step 9766 of 13021 (VAL_EVERY 4883 does
not divide the run; fixed in `609fb1f`, the final step is now always
validated). So best_model.pt and every probe are the 486M-token model; the
last 114M tokens were never measured or kept. M1 at 450M was 3.5348.

**Verdict vs the pre-registered rules: FAIL.** Primary: learning probe
0.199 < 0.225 (+0.024 over M1, about 2 SE; number_step 0.023 → 0.080, still
below chance 0.125). Guards: holdout 3.5397 > 3.52 FAIL (measured early);
recall 0.907 ≥ 0.565 PASS.

**Unexpected: in-context recall 0.907 vs M1 0.615.** 8-fact recall 0.855 vs
0.326; at 8192 tokens 4-fact 0.78 vs 0.48 and 8-fact 0.53 vs 0.28. Same
probe protocol (checked identical). Stage A alone (no recall docs) scores
0.417; the jump appears in stage B, which saw FEWER recall docs than M1
(366M × 5 % vs 600M × 5 %). Hypothesis: the multiple-choice format
("A) … B) … Answer: B) text") is natural associative recall (find the key
in the context, copy its value), so after stage A the model learns the
recall docs far faster. Alternative: the two-stage schedule itself (LR
re-warmup, --init_from). The data stays (`v7/aptitude.py`, also used by
v14); no model code to remove.

Also visible in the learning probe (but see C1-ctl: the schedule does this,
not aptitude): label_category, the one rule task where
LMs score above in-context chance, moves at 8 shots — 2/4/8 shots: M1
0.65/0.67/0.29, C1b 0.57/0.59/**0.49**, Transformer++ 3B 0.63/0.69/0.56.

## C1-ctl — is the recall jump the aptitude data or the two-stage schedule? (pre-registration, 2026-10-29)

Same chain as C1, one change: stage A drops aptitude (dclm/fineweb/
smoltalk2_mid 0.417/0.333/0.25 = C1's stage-A shares renormalised; same
30M web-only warmup; 120M). Stage B identical (M1 mix, 480M, VAL_EVERY
4883, --init_from A). Final val on (`609fb1f`); step_009766.pt kept so the
probes also compare at C1b's step.

**Rule (at step 9766; C1b recall 0.907):** C1b − ctl ≥ 0.10 → the aptitude
data causes the jump (keep it as a curriculum ingredient). ctl ≥ 0.85 →
the two-stage schedule causes it, not the data. In between → mixed, report
both. Also reported: holdout and learning probe at 9766 and at the final
step.

**C1-ctl milestone log** (6000, `7ae6f1f`):
- 2026-10-29 18:31Z: stage A `c1ctla_pre` done, holdout 3.9231 (C1a
  4.1194 — not comparable: aptitude took 40 % of C1a's tokens, and the
  holdout is web text), PG-19 4–8k far gain 0.0446. Stage B started.
- 2026-10-29 19:46Z: stage B `c1ctlb_main` done; probes 19:48Z.

**C1-ctl result (2026-10-30, `7ae6f1f`): by the rule, the aptitude data
adds to the jump (C1b − ctl = +0.102 ≥ 0.10); but most of the jump is the
two-stage schedule itself.**

| model | holdout | WikiText | recall 63-cell (a1/a4/a8) | 8192-tok a1/a4/a8 | learning probe (label_cat 8-shot) |
|---|---|---|---|---|---|
| M1 600M, one stage | 3.4906 | 69.44 | 0.615 (0.895/0.624/0.326) | 0.62/0.48/0.28 | 0.175 (0.29) |
| ctl @ step 9766 | 3.5140 | 75.07 | 0.805 (0.871/0.839/0.705) | 0.56/0.54/0.39 | 0.191 (0.51) |
| C1b @ step 9766 (aptitude) | 3.5397 | 76.00 | **0.907** (0.919/0.948/0.855) | 0.67/0.78/0.53 | 0.199 (0.49) |
| **ctl final (600M total)** | **3.4743** | 69.86 | **0.825** (0.879/0.865/0.732) | 0.58/0.57/0.43 | 0.197 (0.55) |
| ctl stage A (120M, no recall docs) | 3.9231 | 130.30 | 0.408 (0.873/0.267/0.086) | | 0.182 (0.37) |

- Recall: M1 0.615 → two stages without aptitude 0.805 (+0.19) → with
  aptitude 0.907 (+0.10 more), all at step 9766. Probe SE of the 63-cell
  mean ≈ 0.01, so both steps are real; the aptitude step sits on the
  pre-registered 0.10 line.
- LM: aptitude COSTS holdout at the matched step (3.5397 vs 3.5140,
  +0.026: 120M fewer web tokens). The two-stage schedule alone ends
  **better than M1 on holdout (3.4743 vs 3.4906, −0.016) AND recall (0.825
  vs 0.615)**; WikiText equal (69.86 vs 69.44).
- The label_category 8-shot gain is the schedule, not aptitude (ctl 0.51 at
  9766, 0.55 final ≈ Transformer++ 3B 0.56). (Corrects the C1 note above.)
  (2026-10-30: single seed; v14 B1 run 3 found the lookup circuit forms
  bimodally by seed, so this is a hypothesis, see "E2" caveat.)
- Learning probe mean unchanged by anything (0.18–0.20).

**What the two-stage schedule changes vs M1's one stage:** (1) the first
120M tokens have NO recall docs (web 0.75 + chat 0.25: language first),
(2) the LR re-warms and a second cosine starts at 120M, (3) stage B reads
never-read files. (1) is the user's "grammar first" idea in its plain form;
(1) vs (2) is untested.

## S2 fairness — does the two-stage schedule help Transformer++ too? (pre-registration, 2026-10-30)

The recall/LM gain above only counts for the architecture if the baseline
does not get the same gain from the same recipe. Two Transformer++ runs
(ARCH=transformer, 17 × 768 = 101.9M, LR 1.2e-3 = its tuned LR, answer
weight 100, the r600 trainer and evals):
- `tfs1`: one stage, the r600 protocol (M1 mix from token 0, 600M) =
  M1's schedule.
- `tfs2`: two stages, C1-ctl's exact recipe (120M web+chat, then 480M M1
  mix with --init_from).

**Rules (fixed now):**
- Schedule gain is PAM-specific if PAM's recall gain (0.825 − 0.615 =
  +0.210) exceeds Transformer++'s (tfs2 − tfs1) by ≥ 0.10. Same for
  holdout: PAM −0.016; Transformer++'s gain is reported.
- Head-to-head at 600M, same two-stage recipe (C1-ctl vs tfs2): PAM BEATS
  Transformer++ on recall if 63-cell Δ ≥ +0.05; on LM if holdout Δ <
  −0.01 (the scoreboard thresholds).

## S2 on the M2 body (pre-registration, 2026-10-30)

C1-ctl's recipe (no aptitude) on the M2 body (SHARE_READS=1, MLP 2128):
`c2m2`. Controls: M2 one-stage (3.4697 / recall 0.522) and C1-ctl (3.4743 /
0.825). KEEP over M2 if final holdout ≤ 3.4797 AND recall ≥ 0.622 (as C1 on
M2). **Next reference body + recipe** (fixed now): among C1-ctl, c1m2 and
c2m2, the lowest final holdout among those with recall ≥ 0.70; within
0.005 the faster one (M1 body) wins.

Queue on the 6000 after the running C1-on-M2 (`~/s2_chain.sh`): tfs1 →
tfs2 → c2m2, ETA ~08:00 IST.

## C1 on the M2 body (pre-registration, 2026-10-29) — queued after C1-ctl

C1's exact chain (aptitude stage A 120M, M1-mix stage B 480M) on the M2 body
(SHARE_READS=1, MLP 2128), final val on. Control = M2 600M (holdout 3.4697,
recall 0.522, PG-19 far gain 0.0414). **KEEP as the new reference** if final
holdout ≤ M2 + 0.01 (≤ 3.4797) AND recall ≥ M2 + 0.10 (≥ 0.622). Learning
probe reported (bar 3 stays open).

### C1 on M2 result (2026-10-30, 6000, `7ae6f1f`) — KEEP: best LM model at 600M so far

Stage A `c1m2a_pre` 19:48Z → 20:10Z, stage B `c1m2b_main` 20:13Z → 21:39Z,
probes done 21:44Z (03:14 IST). Final val at step 13021 (the `609fb1f` fix).

| model | holdout | WikiText | recall 63-cell (a1/a4/a8) | 8192-tok a1/a4/a8 | learning probe (label_cat 8-shot) | PG-19 4–8k far gain |
|---|---|---|---|---|---|---|
| M2 600M, one stage (control) | 3.4697 | 68.84 | 0.522 (0.774/0.596/0.196) | 0.56/0.48/0.19 | 0.185 (0.31) | 0.0414 |
| C1-ctl final (M1, two stages, no aptitude) | 3.4743 | 69.86 | 0.825 (0.879/0.865/0.732) | 0.58/0.57/0.43 | 0.197 (0.55) | 0.0410 |
| c1m2 stage A (120M, 40 % aptitude) | 4.0921 | 121.74 | 0.369 (0.758/0.218/0.131) | 0.40/0.16/0.13 | 0.195 (0.40) | 0.0401 |
| c1m2 @ step 9766 | 3.4941 | 71.93 | 0.888 (0.919/0.902/0.844) | 0.62/0.60/0.50 | 0.185 (0.37) | — |
| **c1m2 final (600M)** | **3.4473** | **65.74** | **0.870** (0.945/0.890/0.775) | 0.75/0.67/0.49 | 0.187 (0.38) | 0.0436 |

**Verdict vs the pre-registered rule: KEEP.** Final holdout 3.4473 ≤ 3.4797
AND recall 0.870 ≥ 0.622. It is the lowest holdout of any 600M model
(M2 −0.022, C1-ctl −0.027, M1 −0.043) and the lowest WikiText (65.7 vs
M2 68.8), with recall second only to C1b @ 9766 (0.907).

- **Matched step 9766** (holdout / recall): C1b (M1 + aptitude) 3.5397 /
  0.907, ctl (M1, no aptitude) 3.5140 / 0.805, **c1m2 3.4941 / 0.888**. On
  M1, aptitude cost +0.026 holdout at this step; on M2 the aptitude recipe
  is −0.046 ahead of C1b, about twice M2's one-stage body gain (−0.021).
  M2's own no-aptitude two-stage control is `c2m2` (tonight); until it
  lands, the aptitude effect on M2 is not known — do not attribute.
- **label_category 8-shot does not lift on M2** (0.38; M2 one stage 0.31)
  while the two-stage M1 reaches 0.55. The lift looks body-dependent
  (shared reads), not only schedule-dependent. Single probe, ~0.04 SE.
- Learning probe mean flat (0.185–0.197): bar 3 still open.
- Speed 92.6K tok/s vs M1 110.2K (1.19× slower), the same as M2.
- Not yet a baseline comparison: Transformer++ at 600M (`tfs1`, `tfs2`)
  is running now; the aptitude recipe on Transformer++ (`tfc1`) is
  pre-registered below.

## tfc1 — does the aptitude recipe help Transformer++ as much? (pre-registration, 2026-10-30 03:20 IST)

c1m2's recipe (aptitude stage A 120M at 0.25/0.20/0.15/0.40 + 30M
web-only warmup, then the M1 mix 480M with --init_from) on Transformer++
(ARCH=transformer, CHRONO=0, OUT_GATE=0, DELTA=0, 101.9M, LR 1.2e-3), the
same code as tfs1/tfs2 (`b788731`). Queued on the 6000 after the S2 chain
(`~/s3_chain.sh`, tmux `s3chain`). Controls: tfs1 (one stage), tfs2 (two
stages, no aptitude).

**Rules (fixed now, before tfs1/tfs2/tfc1 report):**
- The aptitude recipe's gain is PAM-specific if c1m2's recall gain over
  its one-stage body (0.870 − 0.522 = +0.348) exceeds Transformer++'s
  (tfc1 − tfs1) by ≥ 0.10. Holdout gains reported (PAM: −0.022).
- Head-to-head on the same recipe (c1m2 vs tfc1): PAM BEATS
  Transformer++ on recall if 63-cell Δ ≥ +0.05; on LM if holdout Δ <
  −0.01 (the scoreboard thresholds).

**Tie-break for the next reference** (added before c2m2 runs, to the "S2 on
the M2 body" rule): c1m2 and c2m2 share the M2 body, so speed cannot break
a tie. Within 0.005 holdout, the higher recall wins; if recall is also within
0.02, the simpler recipe (c2m2, no aptitude) wins.

## E2 — tied echo at 600M (pre-registration, 2026-10-30 03:35 IST)

**Idea.** E1 echo reads `S_t q_t` with `q_t = W_q h_t` and keys `k_t =
unit(W_k h_{t-1})`. E2 ties them, `W_q = W_k = W` (one Linear(dim, dim)):
the read becomes a similarity lookup of the current state against the
states that preceded earlier tokens, returning those tokens, i.e. an
induction match ("when I was last in a state like this, the next word
was ..."). In v14 B1 run 2 (micro-bench, 1 seed) it scored label_category
8-shot 0.857 vs untied 0.471. Code: `--echo_tied` / `ECHO_TIED=1` (`26ef94a`;
selftest `test_echo_tied`, grad-checkpoint parity exact on the M1 body).

**Gate.** Runs only if v14 B1 run 3 bar P passes (label_category, tied −
untied paired over 3 seed pairs: mean d ≥ +0.15 AND ≥ 2 SD). If P fails, E2
closes without a 600M run (index row FAIL, flag removed).

**Gate amended (2026-10-30 04:00 IST, before any E2 training):** run 3's
P is INVALID by its own validity rule, not failed. Pair 1 trained with
label_marker in the mix, which taught plain echo label_category to
ceiling (echo 0.999, tied 1.000; P needs echo < 0.90). Pair 1's A
(label_marker_new, every item unseen) went the other way: tied 0.530 vs echo
0.653 (d −0.12), evidence against the induction-lookup reading. The gate
is now **v14 B1 run 4 bar P**: run 2's training set (no label_marker),
echo vs echo_tied, new seed pairs 21/22/23, P's rule frozen (~05:45 IST).
P passes → E2 as below, unchanged. P fails → echo_tied closes and the flag
is removed (code at `26ef94a`). Side note from v14: a related in-context
task in the training mix closes echo's label_category gap entirely, a data
lever (cf. C1-ctl: the schedule, not the body, moved label_category).

**Run 3 final (v14, `d4e0f0d`, 04:44 IST):** P INVALID (echo at ceiling in
pairs 1 and 3); where the arms differ, tied is worse (d +0.00 / −0.07 /
−0.54). A NOT met: d −0.12 / +0.01 / −0.16 (mean −0.09, SD 0.09), evidence
AGAINST the induction/similarity-lookup reading of the tie. v14's
mechanism note: the in-context lookup circuit forms per seed or not
(bimodal: label_category + label_marker together at ~1.0 or at
~0.4–0.5); echo formed it in 2 of 3 seeds, tied in 1 of 3; run 2's +0.39
was a seed where only tied formed it. The gate stays run 4's P.

**Caveat this puts on v13's label_category readings:** if circuit
formation is bimodal by seed, single-run label_category gaps at 600M (M1
0.29 vs C1-ctl 0.55 vs c1m2 0.38) may be seed luck, not schedule or body.
The C1-ctl note "the schedule, not aptitude, lifts label_category" is
single-seed and is downgraded to a hypothesis. Transformer++ 600M (0.73)
is also one seed, but v14's micro-bench reference arm (tpp) will show
whether attention forms the circuit reliably.

**Gate result: v14 B1 run 4 bar P MET (`b6550ec`, ~05:50 IST) → E2 GO.**
d = +0.589 / +0.595 / +0.222 (mean +0.469, SD 0.214, 2·SD 0.427); valid
(echo 0.41–0.44 in every pair, at in-context chance; tied 1.00 / 1.00 /
0.66). On run 2's training set the tie makes the circuit form: tied 4 of 4
seeds, echo 0 of 5. Only with label_marker in the mix does it flip (run
3). By v14's pre-registered map the frame is "abstraction/recall tweak",
not a similarity lookup (A was not met in run 3). Unseen rule parameters
stay ≤ chance for both arms; tied is 9 % faster per step. At 600M each arm
is one seed, so the 400-trial primary is read with that risk in mind.

**Design (option (a), agreed with v14).** Arm `e2t` = M1 body
(PAM_LAYERS=3,7,11,15, MLP 2352, ECHO=1) + ECHO_TIED=1, on C1-ctl's exact
two-stage recipe: stage A `e2ta_pre` 120M dclm/fineweb/smoltalk2_mid
0.4167/0.3333/0.25, 30M web-only warmup, VAL_EVERY 3255; stage B
`e2tb_main` 480M M1 mix 0.44/0.41/0.10/0.05, VAL_EVERY 4883 + final,
--init_from A best, step_009766.pt kept. Seed 42, LR 1.2e-3, B18 T2048.
102.60M params vs ctl 102.95M (−0.34 %, not re-matched: the tied arm is the
smaller one). Why the M1 body: label_category's two-stage lift shows on M1
(0.55) and not on M2 (c1m2 0.38), so a tied-echo effect is readable on M1;
M1 also fits the 4090 with grad checkpointing (M2 share_reads does not).
If E2 passes, it is ported to the reference picked after c2m2.

**Control.** C1-ctl final `c1ctlb_main_7ae6f1f`: holdout 3.4743, recall
0.825, label_category 8-shot 0.55 (100 trials), learning mean 0.197.

**Rules (fixed now):**
- Primary: label_category 8-shot from the learning probe at `--trials 400`
  (seeds 5000–5399), run on the 4090 for BOTH e2t final and C1-ctl final:
  e2t − ctl ≥ +0.15 (SE of the difference ≈ 0.035).
- Guards: holdout ≤ ctl + 0.01 (≤ 3.4843); recall 63-cell ≥ ctl − 0.05
  (≥ 0.775).
- **PASS** (adopt; port to the reference): primary AND both guards.
  **LM-only KEEP**: primary fails, holdout ≤ ctl − 0.01 (≤ 3.4643) and the
  recall guard holds; reported as an LM gain, not a learning gain.
  **FAIL**: anything else → flag removed, index row with the code-at hash.
- Reported, not rules: all 5 learning tasks + mean at 400 trials,
  label_category 2/4-shot, WikiText, PG-19 4–8k far gain, 8192-token recall,
  step-9766 recall, tok/s.

**Caveat.** e2t trains on the 4090 (GRAD_CKPT=1, same maths), ctl trained on
the 6000: different kernels act like a seed change, and 600M seed noise is
not measured (holdout thresholds 0.01 as everywhere; T1 was judged the same
way). The primary probe runs both models on the same box and prompts.

Launch: `~/e2_chain.sh` from the worktree `../qllm_e2` (shared checkout
stays free for v14), tmux `e2`, after v14's run 3 frees the 4090.

### S2 fairness, half 1: Transformer++ one stage at 600M (`tfs1_tf_b788731`, 6000, done 22:59Z = 04:29 IST)

The first Transformer++ at the r600 protocol (same data, trainer, evals as
M1/M2), so the first equal-data 600M baseline for the one-stage PAM models.

| model (600M, one stage) | holdout | WikiText | recall 63-cell (a1/a4/a8) | recall at 2048 / 4096 / 8192 tok | PG-19 4–8k far gain | learning probe (label_cat 2/4/8-shot) | tok/s (6000) |
|---|---|---|---|---|---|---|---|
| Transformer++ 101.9M | 3.4885 | **66.38** | **0.645** (0.694/0.627/0.613) | 0.58 / 0.17 / 0.15 | **−1.095** | **0.217** (0.69/0.84/0.73) | **136.6K** |
| M2 (PAM) | **3.4697** | 68.84 | 0.522 (0.774/0.596/0.196) | 0.45 / 0.40 / 0.41 | +0.0414 | 0.185 (0.59/0.61/0.31) | 92.6K |
| M1 (PAM) | 3.4906 | 69.44 | 0.615 (0.895/0.624/0.326) | 0.66 / 0.56 / 0.46 | +0.0446 | 0.175 (0.65/0.67/0.29) | 110.2K |

- **LM:** M2 beats Transformer++ on holdout by −0.019 (past the −0.01
  scoreboard line); M1 ties it (+0.002). Transformer++ is better on
  WikiText (66.4 vs 68.8).
- **Recall:** Transformer++ is better inside its 2048-token training
  window (mean over contexts ≤ 2048: 0.839 vs M1 0.657 / M2 0.569; 8-fact
  0.61 vs 0.33 / 0.20). It collapses past it (4096: 0.17, 8192: 0.15; PG-19
  4–8k loss +1.1 nats worse than short context), while PAM keeps 0.40–0.56
  and a positive far gain. The 63-cell mean (0.645) mixes the two regimes.
- **Learning probe:** Transformer++ is clearly ahead on label_category
  (8-shot 0.73 vs PAM 0.29–0.31 one-stage, 0.55 for two-stage M1): attention
  does the in-context label lookup that no PAM model has learned yet. The
  other four tasks are near chance for every model. Bar 3 (learning beyond
  recall) is not met by anyone; on the one task that moves, the baseline
  leads.
- Fairness rule pending: needs tfs2 (two stages, ~05:50 IST) for the
  schedule-gain comparison, and tfc1 for c1m2.

### S2 fairness result (2026-10-30, 6000, `b788731`): by the pre-registered rules, two-stage PAM beats two-stage Transformer++ on LM AND recall; the schedule's recall gain is PAM-specific

`tfs2a_pre` 22:59Z → 23:17Z, `tfs2b_main` → 00:18Z (05:48 IST). Same
recipe as C1-ctl (120M web+chat, then 480M M1 mix with --init_from).

| model (600M) | holdout | WikiText | recall 63-cell (a1/a4/a8) | recall ≤ 2048 / > 2048 tok | 2048/4096/8192 | PG-19 4–8k far gain | learning (label_cat 2/4/8) | tok/s |
|---|---|---|---|---|---|---|---|---|
| Transformer++ one stage (tfs1) | 3.4885 | 66.38 | 0.645 (0.694/0.627/0.613) | 0.839 / 0.158 | 0.58/0.17/0.15 | −1.095 | 0.217 (0.69/0.84/0.73) | 136.6K |
| Transformer++ two stages (tfs2) | 3.4907 | **66.20** | 0.703 (0.705/0.733/0.671) | **0.926** / 0.147 | 0.72/0.14/0.15 | −1.223 | **0.217** (0.70/0.84/0.73) | 133.5K |
| M1 one stage | 3.4906 | 69.44 | 0.615 (0.895/0.624/0.326) | 0.657 / 0.510 | 0.66/0.56/0.46 | +0.0446 | 0.175 (0.65/0.67/0.29) | 110.2K |
| **C1-ctl = M1 two stages** | **3.4743** | 69.86 | **0.825** (0.879/0.865/0.732) | 0.917 / **0.597** | 0.79/0.67/0.52 | +0.0410 | 0.197 (0.62/0.71/0.55) | 110.2K |
| tfs2 stage A (120M) | 4.0446 | 145.11 | 0.354 | 0.444 / 0.128 | | −1.126 | | |

**Rules (fixed before tfs1/tfs2 ran):**
- Schedule gain PAM-specific if PAM's recall gain (+0.210) exceeds
  Transformer++'s (tfs2 − tfs1 = +0.058) by ≥ 0.10: **+0.152 → MET.**
  Holdout: PAM −0.016, Transformer++ +0.002 (no LM gain from the schedule).
- Head-to-head, same recipe (C1-ctl vs tfs2): recall Δ +0.122 ≥ +0.05 →
  **PAM beats Transformer++ on recall**; holdout Δ −0.0164 < −0.01 →
  **PAM beats Transformer++ on LM.**

**What the rule verdict does NOT say (read these with it):**
- The recall win is the long-context half. Inside Transformer++'s 2048-token
  training window it is a tie (≤ 2048: 0.926 vs 0.917); past it Transformer++
  collapses (0.147) and PAM keeps 0.597. That is a real property of an
  O(1)-state model (no positional range to run out of), but it is length
  generalisation, not better recall at equal length.
- WikiText goes the other way: Transformer++ 66.20 vs C1-ctl 69.86 (+0.054
  nats for PAM). The holdout (web text from the training distribution)
  favours PAM by 0.016; the out-of-distribution encyclopedic text favours
  Transformer++ by more. LM is mixed, not a clean win.
- Learning probe: Transformer++ leads (0.217, label_category 8-shot 0.73 vs
  0.55). Bar 3 is not met by anyone.
- One seed per arm. Speed: Transformer++ 1.2× faster per token at T=2048 on
  this GPU (its cost grows with context; PAM's does not).
- Scoreboard so far at 600M equal data: bar 2 (beat Transformer++) is MET on
  the pre-registered holdout + recall rules for the two-stage M1 and
  pending for c1m2 (tfc1, ~09:30 IST); not met on WikiText or the learning
  probe. Mamba-2 at 600M is not run yet (3B only: it beat PAM there).

## B2 probe ("inference beats selection") on the 600M and 3B checkpoints (2026-10-30, 6000, `048f8c2`)

v14's B2 tasks (v14/rules_ood.py, via `run_learning_probe.py --suite b2`,
400 trials × 2/4/8 shots, seeds 5000–5399): shift_wide (k 1–12), step_wide
(d 2–20), affine_map (a·x+b), each also as `_new` (held-out parameter
values for v14's micro-bench). The answer that the nearest *other*
parameter would give is always among the 8 candidates, so a model that
selects a memorised neighbour instead of inferring the rule scores ~0.
Chance 0.125. None of these models saw B2 training, so base and `_new` are
equally unseen for them: their gap measures how much harder the held-out
values are (v14's difficulty control).

| model | B2 mean | base | `_new` | 2/4/8-shot |
|---|---|---|---|---|
| Transformer++ 600M one stage | 0.048 | 0.056 | 0.040 | 0.054/0.051/0.040 |
| Transformer++ 600M two stages | 0.053 | 0.059 | 0.048 | 0.050/0.053/0.056 |
| M1 600M | 0.037 | 0.045 | 0.029 | 0.030/0.041/0.040 |
| M2 600M | 0.046 | 0.053 | 0.038 | 0.041/0.045/0.051 |
| C1-ctl 600M | 0.038 | 0.050 | 0.026 | 0.029/0.038/0.047 |
| c1m2 600M (aptitude) | **0.071** | 0.084 | 0.057 | 0.057/0.083/0.073 |
| Transformer++ 3B | 0.088 | 0.111 | 0.064 | 0.063/0.100/0.099 |
| Mamba-2 3B | **0.094** | 0.118 | 0.071 | 0.094/0.100/0.089 |
| PAM 3B (2e-4) | 0.066 | 0.078 | 0.054 | 0.050/0.076/0.072 |

- **Every model is BELOW chance: nobody infers these rules EXACTLY in
  context.** They systematically land on a decoy. Correction (v14): the
  nearest-parameter decoys are also the answer ± 1 step, so a model that
  infers the rule roughly (approximate arithmetic, off-by-one letter) lands
  on the same decoys as one that selects a memorised neighbour. Our 600M/3B
  models had no B2 training, hence no memorised tables, so below chance most
  likely means IMPRECISE inference, not selection; B2 cannot separate the
  two (only the affine a±1 decoys, answer ± x_q, tell a wrong slope from an
  off-by-one). More shots barely help. Bar 3 (exact in-context inference)
  is not met by any architecture at 600M or 3B, attention included.
- **Difficulty control for v14:** `_new` scores 0.015–0.047 below base for
  every model with no B2 training (SE ≈ 0.006 per half), so the held-out
  values are harder by themselves; a 27M trained-vs-held-out gap has to be
  read against this.
- Weak ordering: 3B > 600M; at 600M the aptitude-trained c1m2 is highest
  (0.071; the aptitude mix has GSM8K/AQuA arithmetic), Transformer++ ≈ M2 >
  M1 ≈ C1-ctl. All differences are small next to the distance to chance.

## S2 on M2 result + next reference (2026-10-30, 6000, `b788731`)

`c2m2a_pre` 00:18Z → 00:47Z, `c2m2b_main` → 02:16Z (07:46 IST). C1-ctl's
recipe (no aptitude) on the M2 body.

| model (600M) | holdout | WikiText | recall 63-cell (a1/a4/a8) | ≤ 2048 / > 2048 tok | PG-19 4–8k | learning (label_cat 2/4/8) | tok/s |
|---|---|---|---|---|---|---|---|
| M2 one stage | 3.4697 | 68.84 | 0.522 (0.774/0.596/0.196) | 0.569 / 0.46 | 0.0414 | 0.185 (0.59/0.61/0.31) | 92.6K |
| C1-ctl (M1, two stages) | 3.4743 | 69.86 | 0.825 (0.879/0.865/0.732) | 0.917 / 0.597 | 0.0410 | 0.197 (0.62/0.71/0.55) | 110.2K |
| **c2m2 (M2, two stages)** | 3.4497 | **65.57** | 0.848 (0.945/0.849/0.749) | 0.899 / **0.718** | 0.0440 | 0.185 (0.60/0.71/0.41) | 92.3K |
| **c1m2 (M2, two stages + aptitude)** | **3.4473** | 65.74 | **0.870** (0.945/0.890/0.775) | **0.939** / 0.697 | 0.0436 | 0.187 (0.49/0.58/0.38) | 92.6K |
| Transformer++ two stages (tfs2) | 3.4907 | 66.20 | 0.703 (0.705/0.733/0.671) | 0.926 / 0.147 | −1.223 | 0.217 (0.70/0.84/0.73) | 133.5K |
| c2m2 stage A (120M) | 3.8952 | 124.32 | 0.393 | 0.441 / 0.274 | 0.0448 | | |

**c2m2 vs its rule (KEEP over M2 if holdout ≤ 3.4797 AND recall ≥ 0.622):
KEEP** (3.4497, 0.848).

**Next reference (rule + tie-break fixed before c2m2 ran):** candidates with
recall ≥ 0.70: C1-ctl 3.4743, c1m2 3.4473, c2m2 3.4497. Lowest holdout =
c1m2; c2m2 is within 0.005 (Δ 0.0024), so the higher recall wins unless
recall is also within 0.02: c1m2 0.870 vs c2m2 0.848, Δ 0.022 > 0.02 →
**c1m2 (M2 body + aptitude two-stage recipe) is the reference.** The margin
is thin (probe SE ≈ 0.01); c2m2 is equal on LM and the simpler recipe.

**What the aptitude data does on M2** (c1m2 − c2m2, matched recipe and
body): holdout −0.002, WikiText +0.2, recall +0.022 (inside 2048 +0.040,
past it −0.021), learning probe equal, label_cat 8-shot −0.03. Free on LM
at the final step (on M1 it cost +0.026 at step 9766, measured early), and
a small recall gain. Not a learning gain.

**M2 body vs M1 body on the same two-stage recipe** (c2m2 − C1-ctl): holdout
−0.025, WikiText −4.3 ppl, recall +0.023 (past 2048 +0.12). The M2 body
gain grows with the two-stage recipe (one stage: −0.021, WikiText −0.6).

**Head-to-head with Transformer++ on the same recipe (c2m2 vs tfs2;
scoreboard thresholds, not a separate pre-registered rule):** holdout
−0.041 (< −0.01: beats), WikiText 65.57 vs 66.20 (−0.0096 nats: ahead, under
the 0.01 line), recall +0.145 (≥ +0.05: beats; inside 2048 −0.027, past it
+0.57). The two-stage M2 is the first PAM model ahead of Transformer++ on
both holdout and WikiText at equal data. Transformer++ still leads the
learning probe (0.217 vs 0.185; label_cat 8-shot 0.73 vs 0.41) and is
1.45× faster per token at T = 2048.

## S4 — Mamba-2 at 600M and the reference at 3B tokens (pre-registration, 2026-10-30 08:20 IST)

User (08:00 IST): before scaling, check (1) Mamba-2 at 600M, the other O(1)
baseline, where the cost argument does not help us, and (2) whether the
600M lead holds with more data. "All your calls." Queue on the 6000 after
v14's slot (tpp calibration + B2 run 1, ~10:40 IST), `~/s4_chain.sh`,
tmux `s4`, code = master at launch (pulled).

**Runs** (Mamba-2: `ARCH=mamba2`, 17 × 768 = 102.6M, LR 1.2e-3, its tuned LR;
same trainer, data and evals):
- `mbs1`: Mamba-2 one stage, the r600 protocol (= tfs1).
- `mbs2`: Mamba-2 two stages, C1-ctl's recipe without aptitude (= tfs2,
  c2m2).
- `r3b`: the reference recipe (c1m2) scaled in its main stage: `--init_from`
  c1m2's own stage A (`c1m2a_pre_7ae6f1f`, 120M, aptitude), then stage B
  2,880,000,000 tokens (78,125 steps) on the M1 mix, M2 body, LR **2e-4**
  (the index's 3B rule: PAM's 3B LR is 2e-4, not the 100M-screen pick),
  warmup 1000 (the 3B protocol), VAL_EVERY 3906 + final. Total 3.0B tokens
  = the 3B baselines' count. ~8.7 h at 92K tok/s.

**Rules (fixed now):**
- *Mamba-2 fairness* (as S2): the two-stage gain is PAM-specific vs Mamba-2
  if PAM's recall gain on the reference body (c2m2 − M2 = +0.326) exceeds
  Mamba-2's (mbs2 − mbs1) by ≥ 0.10; holdout gains reported.
- *Head-to-head at 600M, same recipe (c2m2 vs mbs2)*, the 3B scoreboard's
  definition of "beats": LM if holdout lower by > 0.01 AND WikiText lower;
  recall if 63-cell mean higher by > 0.05. Recall also split ≤ 2048 /
  > 2048 tokens.
- *Scale test (r3b vs the 3B baselines, same 3.0B tokens, same holdout,
  same probes)*: Transformer++ 3B 3.2064 / WikiText 46.61 / recall 0.539;
  Mamba-2 3B 3.1644 / 46.91 / 0.613. Same "beats" definition. Read with the
  user's bar-2 tolerance: a gap within ±0.01 holdout counts as competitive
  at lower serving cost (report params, tokens, state size and tok/s). The
  600M lead to compare against: c1m2 vs tfs1 −0.041 (vs mbs1: pending).
  The protocols differ (baselines: one stage with a 300M web-only blend
  warmup; r3b: two stages); S2 showed Transformer++ gains nothing on LM
  from the two-stage schedule, and mbs2 − mbs1 shows it for Mamba-2.
- Reported for all: learning probe (rules, 100 trials) and B2 (400 trials).

**Order:** [E2 port to the reference, only if E2 passes; pre-registered
separately] → mbs1 → mbs2 → r3b. r3b's arm flags come from `~/s4_r3b.flags`
on the 6000 if it exists when r3b starts (set only by a passed E2 port);
otherwise the plain reference.

### E2 result (2026-10-30, 4090, `e5b1a2a`) — FAIL; code removed (last commit with it: `5100fd7`)

`e2ta_pre` 00:13Z → 00:51Z, `e2tb_main` → 03:13Z, probes → 03:17Z (08:47 IST).
57K tok/s on the 4090 with grad checkpointing, no crash.

| model (600M, M1 body, C1-ctl recipe) | holdout | WikiText | recall 63-cell (a1/a4/a8) | label_cat 2/4/8-shot @ 400 trials (4090) | learning mean @ 400 | PG-19 4–8k |
|---|---|---|---|---|---|---|
| C1-ctl (echo, untied; 6000) | **3.4743** | **69.86** | **0.825** (0.879/0.865/0.732) | 0.593 / 0.680 / **0.500** | **0.194** | 0.0410 |
| e2t (tied echo; 4090) | 3.5298 | 75.25 | 0.602 (0.827/0.615/0.364) | 0.615 / 0.670 / 0.365 | 0.177 | 0.0395 |
| e2t stage A (120M) | 3.9255 (ctl A 3.9231) | 127.01 | | | | |

**Verdict: FAIL** on every rule. Primary: label_category 8-shot 0.365 vs
0.500 (d −0.135; bar +0.15). Guards: holdout +0.055 (bar ≤ +0.01), behind
at every val (+0.063 @ 4883, +0.059 @ 9766); recall 0.602 < 0.775. Not the
LM-only case either.

- Stage A matched the control (3.9255 vs 3.9231), so the gap grows in stage
  B (recall docs + the M1 mix) and is far beyond cross-box kernel noise.
- Reading: on real text at 600M, sharing q = k costs the echo its
  separate "what am I looking for" vs "what preceded this" projections, so
  multi-fact recall collapses (8-fact 0.36 vs 0.73) and LM follows. The
  micro-bench gain (circuit forms 4/4 seeds on v14's B1 data) does not
  transfer to pretraining.
- Code removed (`--echo_tied` / `ECHO_TIED`, `_echo_qk`, `test_echo_tied`);
  index row with the rerun recipe. S4 runs without the E2 port.

### tfc1 result (2026-10-30, 6000, `b788731`): by the pre-registered rules, the aptitude recipe's recall gain is PAM-specific and c1m2 beats Transformer++ on holdout and recall; Transformer++ wins WikiText

| model (600M, aptitude two-stage recipe) | holdout | WikiText | recall 63-cell (a1/a4/a8) | ≤ 2048 / > 2048 | learning (label_cat 2/4/8) |
|---|---|---|---|---|---|
| Transformer++ (tfc1) | 3.4802 | **63.83** | 0.648 (0.735/0.623/0.586) | 0.826 / 0.201 | **0.209** (0.69/0.73/0.62) |
| **c1m2 (PAM M2)** | **3.4473** | 65.74 | **0.870** (0.945/0.890/0.775) | **0.939** / **0.697** | 0.187 (0.49/0.58/0.38) |
| tfc1 stage A (120M) | 4.2452 | 143.51 | 0.327 | | 0.229 |

- Rule 1: PAM's recall gain from the recipe (+0.348) vs Transformer++'s
  (tfc1 − tfs1 = +0.003): +0.345 ≥ 0.10 → **PAM-specific, MET.** Holdout
  gains: PAM −0.022, Transformer++ −0.008.
- Rule 2, head-to-head: recall +0.222 (≥ 0.05) → **beats**; holdout −0.033
  (< −0.01) → **beats** on holdout. Here PAM wins inside 2048 too (0.939 vs
  0.826).
- Against: WikiText 63.83 vs 65.74, Transformer++ better (the aptitude stage
  improved its WikiText by 2.4 ppl, ours by none). Learning probe:
  Transformer++ 0.209 vs 0.187; its stage A alone scores 0.229, the highest
  of any model (number_step 0.083, reverse3 0.063). Still near chance.

## LCL — does in-context learning hold past the trained context? (pre-registration, 2026-10-30 11:20 IST)

**Why (step back after S2/tfc1).** PAM's clearest edge is memory past the
2048-token training length (recall > 2048: PAM 0.60–0.72, Transformer++
0.15–0.20). The user's claim: our models' *learning* should hold beyond the
trained context too. Every learning probe so far is under ~1k tokens, so
this was never measured.

**Probe** (`memory_probes/spread.py`, `run_learning_probe.py --spread`): the
8-shot rule prompts with their examples pulled apart by blocks of held-out
web text (pretraining holdout cache, newline-free), header first and query
last, to total lengths L = 0 (no filler) / 1024 / 2048 / 4096 / 8192; the
nearest example is always a full filler block away from the query. Same
targets, candidates and scoring; 300 trials (seeds 5000–5299); the same
filler draw for every model. Tasks: label_category (primary: the only rule
task with signal at 600M), two_hop; B2 shift_wide / affine_map reported (at
the 600M floor).

**Models.** 600M: Transformer++ tfs1 / tfs2 / tfc1; PAM M1, M2, C1-ctl,
c2m2, c1m2. 3B: Transformer++, Mamba-2, PAM (2e-4). All trained at 2048.
Local 4090, one small job (the shared-4090 rule).

**Rules (fixed now):**
- *Learning holds past the trained context* (per model): label_category at
  L = 4096 AND 8192 ≥ its own L = 2048 value − 0.10.
- *PAM beats Transformer++ on long-context learning*: on the same recipe
  (c2m2 vs tfs2, c1m2 vs tfc1, M2 vs tfs1), label_category higher by ≥ 0.10
  at both L = 4096 and 8192 (difference SE ≈ 0.04). Inside the window
  (1024 / 2048) the difference is reported either way.
- Same reading for PAM 3B vs Mamba-2 3B and Transformer++ 3B.

### S4 Mamba-2 at 600M result (2026-10-30, 6000, `685de4e`): Mamba-2 is the LM leader (by 0.010) and leads label_category; PAM leads recall past 2048 with an ~11× smaller state

| model (600M) | holdout | WikiText | recall 63-cell (a1/a4/a8) | ≤ 2048 / > 2048 | PG-19 4–8k | learning (label_cat 8) | tok/s | state / seq |
|---|---|---|---|---|---|---|---|---|
| Mamba-2 one stage (mbs1) | **3.4349** | 66.25 | 0.594 (0.748/0.571/0.462) | 0.687 / 0.361 | 0.0256 | 0.199 (0.47) | 178.8K | 13.4 MB |
| Mamba-2 two stages (mbs2) | 3.4393 | **65.04** | 0.762 (0.777/0.765/0.742) | 0.884 / 0.456 | 0.0240 | **0.227 (0.55)** | 173.0K | 13.4 MB |
| PAM M2 two stages (c2m2) | 3.4497 | 65.57 | **0.848** (0.945/0.849/0.749) | **0.899 / 0.718** | **0.0440** | 0.185 (0.41) | 92.3K | ~1.2 MB |
| PAM M2 + aptitude (c1m2, reference) | 3.4473 | 65.74 | **0.870** | **0.939** / 0.697 | 0.0436 | 0.187 (0.38) | 92.6K | ~1.2 MB |

(PAM M1/M2 state: 4 notebooks × 6 heads × 98² fp32 = 0.92 MB + echo 6 × 98²
= 0.23 MB; Mamba-2 17 × 24 × 64 × 128 fp32 = 13.4 MB.)

**Rules:**
- Mamba-2 fairness: PAM's recall gain from the schedule (+0.326) vs
  Mamba-2's (+0.168): +0.158 ≥ 0.10 → **PAM-specific, MET** (Mamba-2 gains
  recall too, but half as much; neither gets an LM gain: Mamba-2 +0.004).
- Head-to-head, same recipe (c2m2 vs mbs2): LM: Mamba-2 lower by 0.0104
  AND WikiText lower (65.04 vs 65.57) → **Mamba-2 beats PAM on LM**, by the
  smallest margin the rule allows (within the user's bar-2 tolerance band
  ±0.01 it is a tie at 1/11 the state). Recall: +0.086 → **PAM beats
  Mamba-2 on recall** (inside 2048 a tie, 0.899 vs 0.884; past it 0.718 vs
  0.456).
- Not in the rules: Mamba-2 leads the learning probe (0.227 vs 0.185) and
  label_category 8-shot (0.55 vs 0.41), and is 1.9× faster to train.

**Reading (step back).** The label_category gap is NOT an attention-only
skill: Mamba-2, also O(1), gets 0.55 (two stages) where PAM M2 gets 0.41.
So a fixed-size state can do it; our memory read does it worse. PAM's real
edges at 600M: recall past the trained length (0.72 vs 0.46 vs T++ 0.15)
with an ~11× smaller state than Mamba-2. r3b (3B tokens) started 06:29Z.

## label_category diagnosis: PAM's in-context lookup is lost to key collisions, not to distance or capacity (2026-10-30, `bc7d57f`)

User: label_category (short context) is the one place we still lose. Step
back: label_category = find the in-context example of the query's category
and copy its label ("truck is silver ... car is" → silver). Diagnostic
`--suite label_diag`: the same prompts plus `label_exact`, where the query
is an item that IS in the context ("... truck is" → silver): pure lookup and
copy. 200 trials × 2/4/8 shots, CPU (Mamba-2 rows on the 6000).

| model | label_exact (2/4/8) | label_category (2/4/8) |
|---|---|---|
| Transformer++ 600M two stages | **0.883** (0.86/0.92/0.86) | **0.763** (0.71/0.81/0.76) |
| Transformer++ 600M one stage | 0.812 (0.78/0.84/0.81) | 0.750 (0.69/0.84/0.72) |
| Transformer++ 3B | 0.777 (0.68/0.84/0.81) | 0.630 (0.60/0.71/0.57) |
| Mamba-2 600M two stages | 0.720 (0.67/0.81/0.69) | 0.637 (0.56/0.76/0.59) |
| Mamba-2 600M one stage | 0.642 (0.65/0.74/0.54) | 0.572 (0.58/0.65/0.48) |
| Mamba-2 3B | 0.623 (0.44/0.74/0.69) | 0.537 (0.47/0.69/0.46) |
| PAM M1 two stages (C1-ctl) | 0.610 (0.61/0.63/0.59) | 0.595 (0.58/0.68/0.53) |
| PAM M2 two stages (c2m2) | 0.617 (0.58/0.73/0.54) | 0.578 (0.55/0.73/0.46) |
| PAM M1 one stage | 0.585 (0.62/0.72/0.41) | 0.535 (0.57/0.68/0.35) |
| PAM M2 one stage | 0.502 (0.54/0.65/0.32) | 0.478 (0.52/0.61/0.30) |
| PAM E1, 16 notebooks (3.3× state) | 0.533 (0.58/0.65/0.37) | 0.527 (0.55/0.67/0.36) |
| PAM 16 notebooks, no echo (r600 ctl) | 0.588 (0.59/0.74/0.43) | 0.558 (0.56/0.72/0.39) |
| PAM 3B (16 notebooks) | 0.618 (0.44/0.78/0.64) | 0.575 (0.47/0.71/0.54) |

In-context chance: 0.5 / 0.33 / 0.25 at 2 / 4 / 8 shots.

1. **PAM fails exact copy, not just the fuzzy match**: label_exact 0.50–0.62
   vs Transformer++ 0.78–0.88. Exact helps PAM by only +0.01–0.04
   (Transformer++ +0.06–0.12). The same models score 0.94 on the recall
   probe in the trained recall-doc format: the recall skill is format-bound.
2. **Not capacity, not data scale**: 3.3× the state (16 notebooks) does not
   help (0.533 vs M1's 0.585); 3B tokens give 0.618 ≈ 600M. The echo does
   not help either (16 notebooks: no echo 0.588, with echo 0.533). The
   two-stage schedule helps a little (one stage 0.50–0.59 → 0.61).
3. **The lookup is lost to interference from similar writes, not to
   distance** (8 shots, 300 trials, accuracy by how many pairs back the
   correct example sits): PAM M2 exact 0.72 / 0.60 / 0.52 / 0.47 / 0.38 /
   0.29 for 0–5 pairs back (~4 tokens per pair), Transformer++ 0.84 / 0.82 /
   0.82 / 0.90 / 0.78 / 0.71. Yet 1,000 tokens of unrelated filler cost PAM
   only 0.44 → 0.37 (LCL). When wrong, PAM picks an in-context label 99 % of
   the time and the most recent one 45 % (Transformer++ 23 %): last write
   wins.
4. **Mechanism**: the echo keys at the label positions of different "X is
   Y" pairs are near-duplicates: mean cosine 0.67 (c2m2) / 0.72 (C1-ctl) vs
   0.31 / 0.38 between random text positions. Every label is filed under
   almost the same "<noun> is" key, so the delta write's erase partly
   overwrites the earlier pairs, and the query cannot select among them.
   Mamba-2 (0.72 exact) has a k = 4 causal conv on its inputs, so its key
   at "is" carries "truck"; PAM's notebooks and echo have none.

**Next (my lane, the existing body):** G1 short conv (already in code,
`--short_conv`, never tested in pretraining) gives every notebook key the
last 4 tokens. Fallback idea if G1 fails: key centering (subtract a running
mean key before normalising, removing the shared template component).

## G1 short conv on the two-stage M1, judged on in-context copy (pre-registration, 2026-10-30 12:35 IST)

Arm `g1c`: C1-ctl's exact recipe and body (M1, ECHO=1, two stages) +
`SHORT_CONV=1` (k = 4 causal depthwise conv on each notebook's fused qkv,
residual, zero-init; ~28K params). Control = C1-ctl (`c1ctlb_main_7ae6f1f`).
Local 4090 with GRAD_CKPT=1 (needs the whole card ~3 h: agreed with v14
first), else the 6000 after r3b.

**Rules (fixed now):**
- Primary: label_exact mean (label_diag, 200 trials, 2/4/8 shots) ≥ ctl +
  0.10 (0.610 → ≥ 0.710), on the same box as a fresh ctl score.
- Secondary (reported, and needed for "closes the gap"): label_category
  mean ≥ ctl + 0.05 (≥ 0.645); accuracy 5 pairs back ≥ 0.50; echo / notebook
  key cosine at label positions lower than ctl's.
- Guards: holdout ≤ 3.4843 (ctl + 0.01), recall 63-cell ≥ 0.775 (ctl −
  0.05).
- PASS → G1 goes into the reference (M2 port at 600M, then 3B). FAIL →
  record and remove G1, try key centering.

### LCL result (2026-10-30, 4090, probe `7707c8c`): learning does NOT hold past the trained context for any model; past 2048 PAM keeps the label SET (in-context chance), Transformer++ loses even that

label_category, 8 shots (in-context chance 0.25 = picking any of the 4
labels shown; candidate chance 0.125), 300 trials, L = 0 / 1024 / 2048 /
4096 / 8192:

| model | L=0 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| PAM M2 two stages (c2m2) | 0.44 | 0.37 | 0.35 | 0.25 | 0.20 |
| Transformer++ two stages (tfs2) | 0.73 | 0.53 | 0.39 | 0.12 | 0.13 |
| PAM c1m2 (aptitude) | 0.39 | 0.36 | 0.33 | 0.27 | 0.19 |
| Transformer++ aptitude (tfc1) | 0.60 | 0.45 | 0.38 | 0.15 | 0.13 |
| PAM M2 one stage | 0.33 | 0.31 | 0.30 | 0.25 | 0.18 |
| Transformer++ one stage (tfs1) | 0.68 | 0.59 | 0.49 | 0.16 | 0.15 |
| PAM M1 one / two stages | 0.38 / 0.51 | 0.37 / 0.37 | 0.32 / 0.33 | 0.26 / 0.26 | 0.19 / 0.19 |
| Transformer++ 3B | 0.54 | 0.55 | 0.44 | 0.13 | 0.14 |
| PAM 3B | 0.51 | 0.23 | 0.20 | 0.14 | 0.12 |

(Mamba-2 3B not run: mamba_ssm is not installed on the 4090 box.)

**Rules:**
- *Learning holds past the trained context* (4096 and 8192 ≥ own 2048 −
  0.10): **no model.** PAM M2 two stages meets it at 4096 (0.25 vs 0.25
  needed) and misses at 8192 (0.20).
- *PAM beats Transformer++ by ≥ 0.10 at both 4096 and 8192* on matched
  recipes: 4096 +0.13 / +0.12 / +0.09, 8192 +0.07 / +0.06 / +0.03 (two
  stages / aptitude / one stage) → **NOT met** (8192 misses). 3B: tie at
  chance.

**Reading.** Past 2048, PAM sits at in-context chance (0.25): it still
knows WHICH labels were in the document but not which item maps to which.
Transformer++ drops to candidate chance (0.125): it loses the context
entirely. So PAM's long-context edge is memory of the context, not
learning from it; the mapping is lost to the same key collisions found in
the label_category diagnosis (filler adds distance, the examples still
collide). Even inside 2048, filler costs both models (PAM 0.44 → 0.35,
Transformer++ 0.73 → 0.39). PAM 3B (old 16-notebook body, one stage)
collapses with any filler (0.51 → 0.23 at 1024).

**Rule tasks** (letter_shift, number_step, shift_wide(_new),
step_wide(_new), affine_map, two_hop): at the floor (≤ 0.16, mostly below
0.125) for every model at every length, so they cannot show a long-context
edge at 600M without training on such tasks: the B2-in-training 600M pair
(planned) is the way to measure inference past 2048.

## Q1 — engineer the lucky start: echo W_q initialised = W_k, then trained freely (pre-registration, 2026-10-30 13:50 IST)

**Why (user, after v14's seed results).** In-context circuits form per seed,
all or nothing: label_category in v14 B1 (echo 0 of 5 seeds, tied echo 4 of
4), held-out letter shifts in B2 (echo-family 3 of 10 reach ≥ 0.9, one
seed solves it at 1.000; T++ 0 of 4). User: "it's not about other seeds,
it's what that seed did; can we initialise so it works with all seeds,
instead of random?" Clue: a q/k-aligned start (tied echo) makes the
circuit form, but a permanent tie cost recall at 600M (E2). Q1 keeps the
start and drops the tie: `--echo_qk_tied_init` / `ECHO_QK_TIED_INIT=1`
copies the key rows of `echo_qk` into its query rows at init; both then
train freely (selftest `test_echo_qk_tied_init`: q == k at init, identity at
init, untied after one step).

**Stage 1: formation rate on the micro-bench** (cheap, many seeds): v14's
B1 data (run 4's set, label_category) and B2 (shift_wide_new), echo vs echo
+ Q1, 5 seeds each. Rule: Q1 forms the circuit (≥ 0.9) in ≥ 4/5 seeds where
echo forms it in ≤ 2/5 → go to stage 2. Otherwise Q1 does not engineer the
start; record and remove.

**Stage 2: 600M** (C1-ctl recipe, M1 body + ECHO_QK_TIED_INIT=1, vs C1-ctl):
PASS if label_exact (label_diag mean) ≥ ctl + 0.10 OR label_category 8-shot
(400 trials) ≥ ctl + 0.15; guards holdout ≤ ctl + 0.01 (3.4843), recall ≥
ctl − 0.05 (0.775). Read with g1c (short conv) as a separate arm.

### Addendum: pattern separation would cut the collisions ~4× (offline, no training; 2026-10-30 15:25 IST)

v14's suggestion (hippocampal pattern separation: expand the key to a large
dimension and keep only the top-k entries, so look-alike inputs get
near-orthogonal codes). Measured on c2m2's real echo keys at the label
positions of 60 label_category prompts (mean cosine between keys of
different pairs; random text positions for reference), with a fixed random
expansion per head:

| key | label–label | random–random |
|---|---|---|
| dense unit keys (today) | 0.671 | 0.303 |
| dense, mean-centred | 0.462 | −0.005 |
| sparse D = 1024, top 32 | 0.334 | 0.142 |
| sparse D = 2048, top 32 | 0.297 | 0.114 |
| **centred, then sparse D = 2048, top 32** | **0.166** | 0.053 |

Centring + sparse expansion lowers the overlap of look-alike keys from 0.67
to 0.17 with no learning; the delta erase would then wipe ~4× less of the
earlier pairs, and capacity rises (sparse codes). Echo state with D = 1024
key dims per head: 6 × 98 × 1024 fp32 ≈ 2.4 MB (Mamba-2 13.4 MB). **Candidate
S1** (my lane, the existing echo): centred sparse-expansion keys and
queries in the echo read/write, pre-registered after g1c / Q1 report (one
variable at a time). Implementation check needed: `pam_delta_batched` with a
key dim different from the value dim.

## S1 separated echo keys + the joint formation bench F1 (pre-registration, 2026-10-30 16:10 IST)

**S1** (`--echo_sparse_k K` / `ECHO_SPARSE_K`): echo keys AND queries are
centred by the causal running mean of the keys so far (removes the shared
template component; carry = key sum + count, O(1)) and sparsified to their
top-K magnitudes per head (pattern separation). No kernel change, same state
size. Offline on c2m2's keys (top 8, global mean): label-key cosine 0.67 →
0.25. K = 8 for F1. Selftest `test_echo_sparse_keys` (identity at init,
causal, grads, copy, decode == chunked 8.9e-5, ≤ K non-zeros per head).

**F1** (joint with v14, pre-registered in EXPERIMENTS_V14 too): v14's B1
micro-bench on run 4's training set, label_category; seeds 7i, doc_seed0
(i+15)e8, init 7i, i = 1..5; "formed" = label_category ≥ 0.9 (mean over
2/4/8 shots). Arms paired by identical seed / init / data: `echo` (control)
and `echo_phase` (v14 N9a) in v14's slot; `echo_qinit` (Q1) and
`echo_sparse` (S1, K = 8) in v13's slot after g1c. **Rule: an arm passes if
it forms in ≥ 4/5 seeds where echo forms in ≤ 2/5; the best-forming
passing arm goes to 600M** (C1-ctl recipe, the label_diag / label_category
bars of Q1 stage 2). If none passes, record and remove the losers.

### g1c result (2026-10-30, 4090, `9c2b632`): FAIL as measured, but a 4090 confound is now suspected (verdict held open)

| model (600M, M1 body, C1-ctl recipe) | holdout | WikiText | recall 63-cell (a1/a4/a8) | label_exact (2/4/8) | label_category (2/4/8) | PG-19 4–8k |
|---|---|---|---|---|---|---|
| C1-ctl (6000) | **3.4743** | **69.86** | **0.825** | **0.610** (0.61/0.63/0.59) | **0.595** (0.58/0.68/0.53) | 0.0410 |
| g1c short conv (4090) | 3.5232 | 77.41 | 0.770 (0.873/0.800/0.638) | 0.517 (0.49/0.64/0.42) | 0.492 (0.48/0.62/0.36) | 0.0431 |

(label_diag for both scored on the 4090, same prompts.) By the rules:
primary label_exact 0.517 < 0.710 → FAIL; holdout +0.049 > +0.01 → FAIL.

**But a box confound now looks likely.** Both 4090 runs of this recipe, E2
(tied echo) and g1c (short conv), lose almost the same amount to the
6000 control: holdout +0.055 and +0.049. Their data cursors are identical
to the control's (same files and rows), and their training loss tracks the
control through stage A and the first 3,000 steps of stage B, then both
drift up ~0.07 and stay there (steps 3000–6000: ctl 3.187, E2 3.270, g1c
3.264; 6000–13021: 3.032 / 3.101 / 3.066). Differences on the 4090:
gradient checkpointing (exact parity on CPU, untested on GPU kernels),
torch / kernel versions (6000: torch 2.8.0+cu128, Blackwell). Two unrelated
changes failing by the same margin is suspicious. **Check (box control)**:
rerun C1-ctl's stage B on the 4090 from C1-ctl's own stage-A checkpoint
(same data, same seed, GRAD_CKPT=1) for ~4,500 steps; if its training loss
tracks the 6000 log, the 4090 is fine and E2 / g1c stand; if it drifts after
~3,000 steps, every 4090 600M verdict (E2, g1c, T1) is re-opened.
Queued on the 4090 after F1 (~19:50 IST).

## Box control result + F1 result + B2-in-training pair (2026-10-30 22:15 IST)

**Box control (4090, `boxchk`)**: C1-ctl's stage B rerun on the 4090 from
C1-ctl's own stage-A checkpoint (same data, seed, recipe; grad ckpt on) tracks
the 6000 log: train-loss windows +0.003 / +0.005 / +0.006 / +0.005 / +0.013 /
+0.012 (steps 0–4600), vs +0.07–0.08 for E2 and g1c. **The 4090 is not the
cause: E2, g1c and T1 FAILs stand.** Cross-box comparisons carry ~0.01 noise.

**F1 (joint formation bench, B1 label_category, 5 paired seeds; formed ≥
0.9)**: N9a content-phase keys (`echo_phase`, v14) **5/5** (1.000 each);
echo control 1/5 (0.70 / 0.91 / 0.40 / 0.72 / 0.66); **Q1 tied start 0/5**
(0.67 / 0.42 / 0.42 / 0.43 / 0.42); **S1 separated keys 0/5** (0.44 / 0.41 /
0.58 / 0.43 / 0.41). Q1 and S1 FAIL (their code is removed, index rows keep
the rerun hash); N9a goes to 600M on the 6000 (pre-registered when its
trainer hook lands).

**B2-in-training 600M pair (pre-registration).** Stage B of the reference
recipe (c1m2: M2 body, aptitude stage A reused: `--init_from
c1m2a_pre_7ae6f1f`) with the `b2` source added: dclm / fineweb /
smoltalk2_mid / recall / b2 = 0.43 / 0.40 / 0.10 / 0.04 / 0.03 (b2 = trained
rule values only, seeds ≥ 1e7). Arms on the 6000: `b2p1` PAM (seed 42),
`b2t` Transformer++ (same, from its own aptitude stage A `tfc1a_pre`), `b2p2`
PAM (SEED = 43: another data order and init draw). **Rule:** PAM infers held-out
rule values better than Transformer++ if the B2 `_new` mean (shift_wide_new,
step_wide_new, affine_map_new; 400 trials, 2/4/8 shots) is ≥ T++ + 0.10 in
BOTH PAM seeds. Reported: base (trained-value) cells, formation (≥ 0.9) per
task, `_new` past 2048 (LCL spread), holdout / recall vs c1m2 / tfc1 (b2 costs
3 % of the web tokens).

### S4 scale test result: the reference at 3B tokens (`r3b_main_685de4e`, 6000) — FAILS; the 600M lead does not survive scale

c1m2's stage A + a 2.88B-token stage B (M2 body, LR 2e-4, warmup 1000), 3.0B
tokens total = the 3B baselines' budget, same holdout and probes.

| model, 3B tokens | holdout | WikiText | recall 63-cell (a1/a4/a8) | ≤ 2048 / > 2048 | learning (label_cat 2/4/8) | B2 |
|---|---|---|---|---|---|---|
| **PAM reference (r3b)** | 3.3094 | 58.03 | 0.620 (0.874/0.550/0.437) | 0.676 / 0.482 | 0.189 (0.54/0.60/0.39) | 0.076 |
| PAM old 16-notebook (LR 2e-4) | 3.2728 | 54.73 | **0.667** | | 0.213 (0.50/0.74/0.53) | 0.066 |
| Transformer++ | 3.2064 | **46.61** | 0.539 | | **0.228** (0.63/0.69/0.56) | 0.088 |
| Mamba-2 | **3.1644** | 46.91 | 0.613 | | 0.202 | **0.094** |

Holdout along the way (matched tokens, cosine schedules over 3B): at 1.2B
r3b 3.462 / old PAM 3.461 / T++ 3.430 / Mamba-2 3.370; at 2.0B 3.354 /
3.333 / 3.288 / 3.243; at 3.0B 3.309 / 3.273 / 3.206 / 3.164.

**Verdict (pre-registered "beats" definition):** LM vs Transformer++
+0.103 and WikiText 58.0 vs 46.6 → **LOSE**; vs Mamba-2 +0.145 → **LOSE**;
outside the user's ±0.01 tolerance by 10×. Recall vs T++ +0.081 → beats; vs
Mamba-2 +0.007 → tie. **Not even better than PAM's own old 3B run**
(+0.036): the M2 / two-stage / echo recipe that won at 600M is worse at 3B.
Mamba-2 reaches r3b's final loss at ~1.6–1.7B tokens, T++ at ~1.8B: PAM
needs ~1.7–1.8× the data to match, which is not "a bit more training".

**Reading.** PAM learns fast early and plateaus: every 600M comparison sits
in the regime where that helps (the 100M LR screen fooled us the same way,
EXPERIMENT_INDEX "LR 2e-4"). The 600M wins (M2, two stages, beating T++ by
0.04) are short-horizon effects. Possible culprits for r3b vs old PAM: the
sparse body (4 of 16 blocks with a notebook) scales worse, or LR 2e-4 after
a 1.2e-3 stage A; not separated. **Consequence for the protocol:** a
change that wins at 600M must also show it at ≥ 1.5B tokens before it counts,
and LM gains are judged at the end of a ≥ 3B run.

## n9c — v14 N9a content-phase echo keys at 600M (pre-registration, 2026-10-30 22:35 IST)

The F1 winner (N9a `echo_phase`, 5/5 seeds vs echo 1/5 on B1). Arm `n9c`:
C1-ctl's exact recipe and body (M1, ECHO=1, two stages) + `ECHO_PHASE_LAGS=3`
in BOTH stages (v14 hook `424afd6`, builds V14LM), local 4090 with GRAD_CKPT=1
(box control: ±0.01). Control = C1-ctl. All lookup probes on the 4090 for both.

**Rules (fixed now).** This 600M run measures LOOKUP / FORMATION only (the 3B
lesson: LM effects need ≥ 1.5B tokens).
- PASS if label_exact (label_diag mean, 200 trials) ≥ ctl + 0.10 (0.610 →
  ≥ 0.710) OR label_category 8-shot (400 trials) ≥ ctl + 0.15 (0.500 →
  ≥ 0.650).
- Guard (big-loss only): holdout ≤ ctl + 0.03 (≤ 3.504), recall ≥ ctl − 0.05
  (≥ 0.775).
- Reported: accuracy by pairs back (recency), key cosine at label
  positions, LCL label_category past 2048 vs in-context chance 0.25.
- PASS → N9a's LM effect gets a ≥ 1.5B-token run before any bar-2 claim.

### Slope check from the 3B logs (2026-10-30 23:05 IST): the plateau tracks memory layout AND learning rate

| model (3B tokens, own cosine) | 1.2B | 2.0B | 3.0B | 1.2 → 3B | 2 → 3B |
|---|---|---|---|---|---|
| PAM M2, 4 notebooks (~1.2 MB), LR 2e-4 (r3b) | 3.447 | 3.354 | 3.309 | −0.138 | **−0.045** |
| PAM 16 notebooks (~3.7 MB), LR 2e-4 | 3.461 | 3.333 | 3.273 | −0.188 | −0.060 |
| PAM 16 notebooks, LR 1.2e-3 | 3.537 | 3.371 | 3.284 | −0.253 | **−0.086** |
| Transformer++ | 3.430 | 3.288 | 3.206 | −0.224 | −0.081 |
| Mamba-2 (13.4 MB, 17 mixing layers) | 3.370 | 3.243 | 3.164 | −0.206 | −0.079 |

- At the same LR (2e-4), more mixing / state = steeper late slope (4 →
  16 notebooks: −0.045 → −0.060).
- At LR 1.2e-3 the 16-notebook body keeps improving as fast as the
  baselines (−0.086 vs −0.081 / −0.079), but starts worse (high-LR early
  penalty), so it ends 3.284. LR 2e-4 bought a good start and a flat finish.
- r3b had BOTH flatteners: the 4-notebook body and LR 2e-4. **The slope test
  should cross body × LR**: 16-notebook + echo + two stages vs M2, at LR
  1.2e-3 (and a mid LR, 6e-4), read by the 600M → 1.5B drop and the late
  slope, not by the 600M number.

## r3c — run the known-good combination straight at 3B (pre-registration, 2026-10-30 23:20 IST)

User: "do not waste time on things we already know are not superior; run
directly what deserves priority." What the evidence already says works for
PAM's LM slope and quality: all 16 blocks with a notebook (late slope −0.060
vs −0.045 for 4; at LR 1.2e-3 −0.086 ≈ baselines), LR 1.2e-3, E1 echo (−0.052
at 600M on exactly this body), the two-stage schedule (PAM-only gain). Never
run together. So instead of a 2×2 body × LR test at 1.5B (~25 GPU-h), one
3B run of the combination:

`r3c`: baseline_real_pm (16 notebooks, chrono + out_gate + delta) + ECHO=1;
stage A 120M dclm / fineweb / smoltalk2_mid 0.4167 / 0.3333 / 0.25 (30M
web-only warmup) → stage B 2.88B tokens M1 mix 0.44 / 0.41 / 0.10 / 0.05,
`--init_from` A; LR 1.2e-3 both stages, warmup 100 / 1000, VAL_EVERY 3906,
step 39062 kept (~1.56B tokens). 6000, after b2t (b2p2 deferred). ~14 h.
N9a not included (its 600M lookup check lands 01:30; its LM effect at scale is
unknown: it gets its own run).

**Rules (fixed now):**
- *Futility at ~1.56B tokens*: stop if r3c's holdout is worse than the old
  16-notebook LR-1.2e-3 3B run at matched tokens (3.484 @ 1.475B; interpolate
  to r3c's token count) — then the combination adds nothing over the old body.
- *Final, vs the 3B baselines* (same definition as S4): beats if holdout lower
  by > 0.01 AND WikiText lower; within ±0.01 = competitive at lower serving
  cost (state ~3.9 MB vs Mamba-2 13.4 MB, vs T++ KV cache); T++ 3.2064 /
  46.61, Mamba-2 3.1644 / 46.91. Also reported: late slope 2 → 3B vs
  baselines (−0.081 / −0.079), recall, label_diag, B2, LCL.

## B2 pair — seed-42 half result + amendment (2026-10-31 ~01:00 IST)

Runs (6000, code `38e3c29`, 600M tokens each = 120M aptitude stage A + 480M
stage B with the `b2` source at 3 %): `b2p1_main_38e3c29` PAM (M2 + echo +
shared reads, 102.9M, stage A `c1m2a_pre`), `b2t_main_38e3c29`
Transformer++ (101.9M, stage A `tfc1a_pre`). Same data order (seed 42).

| 600M tokens | params | holdout | recall 63 (≤ 2048 / > 2048) | PG-19 far gain 4–8k | label_diag cat / exact | B2 base | B2 `_new` shift / step / affine (mean) |
|---|---|---|---|---|---|---|---|
| **PAM b2p1** | 102.9M | **3.4777** | **0.721** (0.781 / **0.572**) | **+0.043** | 0.547 / 0.588 | 1.000 | **1.000 / 0.973 / 1.000 (0.991)** |
| T++ b2t | 101.9M | 3.5572 | 0.672 (**0.844** / 0.242) | −0.778 | **0.673 / 0.752** | 0.999 | 0.257 / 1.000 / 0.944 (0.734) |
| same recipes without b2: c1m2 / tfc1 | | 3.4473 / 3.4802 | 0.870 / 0.648 | | | below chance (0.04–0.09) | |

Rules suite (100 trials): PAM label_category 0.557, two_hop 0.290; T++ 0.687 / 0.273.

LCL spread on the `_new` tasks (8 shots, 200 trials; L = whole prompt length,
examples pulled apart by held-out web text; candidate chance 0.125; trained
length 2048):

| `_new` @ L | 0 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| PAM shift_wide | 1.00 | 0.79 | 0.60 | 0.34 | 0.105 |
| T++ shift_wide | 0.315 | 0.00 | 0.00 | 0.00 | 0.00 |
| PAM affine_map | 1.00 | **0.965** | **0.965** | **0.785** | 0.435 |
| T++ affine_map | 0.975 | 0.075 | 0.05 | 0.45 | 0.285 |
| PAM step_wide | 1.00 | 0.115 | 0.14 | 0.10 | 0.105 |
| T++ step_wide | 1.00 | 0.09 | 0.085 | 0.09 | 0.055 |

**Half-verdict (seed 42):** `_new` mean PAM − T++ = **+0.257** (bar +0.10) →
met for seed 42; the pre-registered verdict needs b2p2 (seed 43).

**Reading.**
- With the rule formats in training (values held out), PAM infers the new
  values exactly (target margins 6–19 nats). T++ learns step and affine but
  not shift: 0.26 with a NEGATIVE margin, i.e. it prefers a decoy one step
  from the answer.
- Across distance the gap is large, and it starts INSIDE the trained length:
  affine at L = 1024 is PAM 0.965 vs T++ 0.075 (gaps of ~110 tokens). PAM
  still holds affine past the trained length (0.785 at 4096). This is the
  first time any model's in-context learning holds past 2048 in LCL. The
  earlier LCL floor (no model, no rule data) was partly a data gap.
- step_wide collapses at 1024 for BOTH: it is a sequence, so each term sits a
  filler block from the last. A format effect, not a model difference
  (not checked further).
- Cost of the b2 docs (answer weight 100): PAM recall −0.149 (0.870 → 0.721),
  holdout +0.030; T++ holdout +0.077, recall +0.024. Read it as a DATA-MIX
  trade, not "inference costs recall": the b2 mix cut the recall docs from
  5 % to 4 %, and b2 answers at weight 100 compete for the same gradient
  (untested; v14's note). A lighter weight or share is an open recipe
  question, not run now.
- Caveats: trained rule FORMATS with held-out VALUES, not unseen rule
  families. One seed. T++ only: Mamba-2 is also recurrent and O(1), so it may
  share the distance edge.

**Amendment (fixed before these runs start; tmux `s6` on the 6000, next to r3c):**
1. **b2p2** (PAM, SEED=43, exactly as pre-registered) runs NOW, not after
   r3c. Cost: r3c ends ~1.5 h later (~16:15 IST instead of ~14:45). Gain: the
   pair verdict ~12 h earlier. The s5 chain's own b2p2 step then skips
   (`.done`).
2. **New arm b2m: Mamba-2 (102.6M) on the same data.** Stage A `b2ma` =
   tfc1a's aptitude recipe with ARCH=mamba2 (120M; mbs2a had no aptitude, so
   it is not reused). Stage B = the b2 mix, seed 42, same probes. Why: if
   Mamba-2 shows the same edge, it is "recurrence vs attention", not PAM.
   **Rule:** PAM beats Mamba-2 on held-out rule values if the `_new` mean is
   ≥ b2m + 0.10 in BOTH PAM seeds. Across distance, PAM's affine/shift `_new`
   at L = 2048 and 4096 vs b2m is reported (same +0.10 margin).
3. Bar 3 counts as replicated only if PAM clears BOTH T++ and Mamba-2 in both
   seeds. Only then: EUREKA + ntfy.

### n9c result (2026-10-31 01:45 IST): FAIL on lookup — the micro formation win does not transfer; recall jumps instead

`n9cb_main_e5f0b73` (4090, 103.5M, 600M tokens, C1-ctl recipe + ECHO_PHASE_LAGS=3)
vs C1-ctl (`c1ctlb_main_7ae6f1f`, ~102.9M, same recipe; lookup probes on the 4090).

| | n9c (N9a) | C1-ctl | bar |
|---|---|---|---|
| label_exact (label_diag, 200) | **0.472** | 0.610 | ≥ 0.710 → **FAIL** |
| label_category 8-shot (400) | **0.318** | 0.500 | ≥ 0.650 → **FAIL** |
| label_category 2 / 4-shot (400) | 0.507 / 0.502 | 0.593 / 0.680 | |
| holdout | 3.4725 | 3.4743 | guard ≤ 3.504 ok |
| recall 63-cell (≤ 2048 / > 2048) | **0.975** (0.996 / **0.924**) | 0.825 (0.917 / 0.597) | guard ≥ 0.775 ok |
| PG-19 far gain 4–8k | +0.037 | +0.041 | |
| echo key cosine at label positions (random positions) | **0.538** (0.227) | 0.715 (0.379) | reported |
| label_exact by pairs back 0/1/2/3/4/5 (300, CPU) | 0.58/0.54/0.37/0.43/0.16/0.12 | 0.57/0.62/0.48/0.67/0.34/0.47 | reported |
| LCL label_category @ 0/1024/2048/4096/8192 (300) | 0.31/0.30/0.27/0.23/0.18 | 0.51/0.37/0.33/0.26/0.19 | in-context chance 0.25 |

**Verdict: FAIL** (pre-registered). Both lookup bars missed, and both moved
the WRONG way (−0.14 exact, −0.18 category 8-shot). Consequence: no ≥ 1.5B
LM run for N9a in v13. The unit stays in v14 (the v13 trainer keeps only the
`--echo_phase_lags` hook; v14 decides on its unit).

**Reading.**
- N9a did what it was built for: label keys are pulled apart (cosine 0.715 →
  0.538). The lookup still got worse. So key collision was a real symptom
  but not the cause of the label gap at 600M.
- Recall jumps (+0.150; past 2048 0.597 → 0.924, the best of any 600M
  model; T++ 0.20, Mamba-2 0.46). The training recall docs use one fixed
  template ("Query: <key> refers to"). N9a got very good at the format it
  was trained on, and worse at a format it was not ("<item> is <label>,"
  never appears in training). That fits the B2 finding the same night: PAM
  learns in-context skills from formats seen in training.
- Wrong answers are always an in-context label (100 %; ctl 93–96 %). The fall
  with distance is steeper (4–5 pairs back 0.16 / 0.12 vs 0.34 / 0.47): older
  pairs are lost faster even with distinct keys.
- Micro F1 (5/5 formed) trained on label_category itself; 600M never shows
  it. "Formation at micro" was formation on the trained format.
- Not run: why exact lookup drops with distance (write gates? query turned by
  a window that includes the previous label?). That is v14's design question.

## Label-format pair `lb` (pre-registration, 2026-10-31 02:50 IST): is the label gap a data-format gap?

Why now: two results tonight point one way. B2 docs in training took PAM's
held-out rule values from ~0.05 to 0.99 (T++ 0.73). The recall docs' single
template took N9a's recall to 0.975, while label lookup (a format never
trained) fell. "<item> is <label>," never appears in our training data, and
label_category / label_exact is the one probe T++ still wins (b2 pair: label_diag
mean PAM 0.567 vs T++ 0.712).

**Source `lab`** (`c4173be`): label-rule docs in the probes' exact format.
Vocabulary is disjoint from the probes: 14 categories (tool, bird, metal,
drink, …), 20 labels (echo, flux, north, …). 0 of the 136 distinct training
words are probe words. Half the queries are new members of a shown category,
half are in-context items. 6 prompts per doc, answers marked (weight 100).
The probes test the skill on words never seen in these docs.

**Arms** (6000, code `c4173be`, worktree `~/Development/qllm_lab`, tmux `s7`,
starts when s6 ends, next to r3c). The b2 pair's recipe with `lab` in place
of `b2`: stage B 480M dclm / fineweb / smoltalk2_mid / recall / lab = 0.43 /
0.40 / 0.10 / 0.04 / 0.03, seed 42.
- `lbp`: PAM M2 + echo + shared reads, `--init_from c1m2a_pre_7ae6f1f`.
- `lbt`: Transformer++, `--init_from tfc1a_pre_b788731`.

Probes: label_diag 200, rules 100 + t400 (label_category 2/4/8 shots), LCL
spread (label_category, two_hop; 8 shots, 300; L = 0 … 8192), recall 63-cell,
PG-19, holdout.

**Rules (fixed now):**
1. *Format gap (primary):* PAM label_diag mean (category + exact) ≥ 0.767
   (= b2p1's 0.567 + 0.20) → the label gap was mostly a data-format gap.
2. *vs T++ on the same data:* PAM label_diag mean ≥ lbt − 0.02 → gap closed;
   ≥ lbt + 0.05 → PAM ahead.
3. *Past the trained length (reported):* LCL label_category at 4096 / 8192
   vs T++ (in-context chance 0.25).
4. *Guards:* holdout ≤ 3.487 (c1m2 + 0.04, the b2 docs cost +0.030); recall
   reported (the b2 docs cost 0.15).
5. One seed. A pass gets a second seed before any claim. A FAIL on rule 1
   means the label gap is the body's (back to v14's lookup work), and `lab`
   is removed from the code (index row with the hash).

## B2 pair — seed-43 result (2026-10-31 04:05 IST): PAM beats Transformer++ on held-out rule values in BOTH seeds (Mamba-2 arm still to come)

`b2p2_main_38e3c29` (6000, PAM M2 + echo + shared reads, 102.9M, seed 43,
same recipe as b2p1: 120M aptitude stage A `c1m2a_pre` + 480M stage B with
3 % `b2` docs; 600M tokens). T++ row = b2t (101.9M, seed 42, 600M tokens).

| 600M tokens | params | holdout | recall 63 (≤ 2048 / > 2048) | PG-19 far gain 4–8k | label_diag cat / exact (mean) | B2 base | B2 `_new` shift / step / affine (mean) |
|---|---|---|---|---|---|---|---|
| **PAM b2p2 (s43)** | 102.9M | 3.4800 | 0.707 (0.801 / **0.474**) | **+0.042** | 0.592 / 0.660 (0.626) | 1.000 | **1.000 / 0.834 / 0.999 (0.944)** |
| PAM b2p1 (s42) | 102.9M | 3.4777 | 0.721 (0.781 / 0.572) | +0.043 | 0.547 / 0.588 (0.567) | 1.000 | 1.000 / 0.973 / 1.000 (0.991) |
| T++ b2t (s42) | 101.9M | 3.5572 | 0.672 (0.844 / 0.242) | −0.778 | **0.673 / 0.752 (0.713)** | 0.999 | 0.257 / 1.000 / 0.944 (0.734) |

Rules suite (100): b2p2 label_category 0.587, two_hop 0.293 (b2p1 0.557 / 0.290; T++ 0.687 / 0.273).

LCL spread, `_new` tasks (8 shots, 200 trials, chance 0.125, trained length 2048):

| `_new` @ L | 0 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| PAM s43 shift_wide | 1.00 | 0.735 | 0.53 | 0.295 | 0.12 |
| PAM s42 shift_wide | 1.00 | 0.79 | 0.60 | 0.34 | 0.105 |
| T++ shift_wide | 0.315 | 0.00 | 0.00 | 0.00 | 0.00 |
| PAM s43 affine_map | 1.00 | **0.995** | **0.975** | **0.905** | **0.61** |
| PAM s42 affine_map | 1.00 | 0.965 | 0.965 | 0.785 | 0.435 |
| T++ affine_map | 0.975 | 0.075 | 0.05 | 0.45 | 0.285 |
| PAM s43 step_wide | 0.925 | 0.09 | 0.065 | 0.085 | 0.04 |
| T++ step_wide | 1.00 | 0.09 | 0.085 | 0.09 | 0.055 |

**Verdict vs Transformer++ (pre-registered: `_new` mean ≥ T++ + 0.10 in BOTH
PAM seeds): MET.** Seed 42 +0.257, seed 43 +0.210. Across distance both seeds
agree: affine `_new` at 2048 / 4096 is PAM 0.965–0.975 / 0.785–0.905 vs T++
0.05 / 0.45 (margin ≥ +0.33). Bar 3 is NOT yet claimed: the amendment needs
PAM ≥ Mamba-2 (b2m) + 0.10 in both seeds too (b2ma → b2m running in tmux `s6`).

**Reading.**
- The rule-value edge replicates; seed noise is real on single cells
  (step_wide_new 0.973 vs 0.834, recall > 2048 0.572 vs 0.474). Read any one
  cell to ±0.1 from one seed.
- What survives distance: shift and affine need only the RULE (k, or a and b);
  the query sits next to the answer. Step needs an exact COPY of the last far
  term, and it collapses at 1024 for both models. PAM carries the abstract rule
  across thousands of tokens, but not an exact distant token through filler.
  Shift decays smoothly (≈ halving above chance per doubling of L), affine much
  less. Both use the same ±1 decoys, so decoy spacing is not why. Untested
  guess: the letter offset drifts back toward the trained k values. The probe
  JSONs keep only aggregates, so the picks were not checked.
- Labels: PAM 0.57 / 0.63 vs T++ 0.71 on the same data. The lb pair
  (pre-registered above) tests whether that is the missing format.
- Cost stays as in seed 42: holdout +0.033 vs c1m2, recall 0.870 → 0.707.

## B2 Mamba-2 arm b2m (2026-10-31 06:20 IST): bar 3 NOT met in its pre-registered form, but the distance edge is PAM's alone

`b2m_main_38e3c29` (6000, Mamba-2 102.6M, `ARCH=mamba2,CHRONO=0,OUT_GATE=0,DELTA=0`,
stage A `b2ma_pre_38e3c29` = 120M aptitude mix as tfc1a, stage B = the b2 mix,
seed 42, 600M tokens total; same probes as the PAM / T++ arms).

| 600M tokens | params | holdout | recall 63 (≤ 2048 / > 2048) | PG-19 far gain 4–8k | label_diag cat / exact (mean) | B2 base | B2 `_new` shift / step / affine (mean) |
|---|---|---|---|---|---|---|---|
| PAM b2p1 (s42) | 102.9M | 3.4777 | 0.721 (0.781 / **0.572**) | **+0.043** | 0.547 / 0.588 (0.567) | 1.000 | 1.000 / 0.973 / 1.000 (**0.991**) |
| PAM b2p2 (s43) | 102.9M | 3.4800 | 0.707 (0.801 / 0.474) | +0.042 | 0.592 / 0.660 (0.626) | 1.000 | 1.000 / 0.834 / 0.999 (0.944) |
| **Mamba-2 b2m (s42)** | 102.6M | **3.4754** | 0.702 (0.818 / 0.410) | +0.022 | **0.713 / 0.862 (0.787)** | 1.000 | 0.677 / 1.000 / 1.000 (0.892) |
| T++ b2t (s42) | 101.9M | 3.5572 | 0.672 (**0.844** / 0.242) | −0.778 | 0.673 / 0.752 (0.713) | 0.999 | 0.257 / 1.000 / 0.944 (0.734) |

Rules suite (100): b2m label_category 0.733, two_hop 0.32 (PAM 0.557–0.587 / 0.29; T++ 0.687 / 0.273).

LCL spread, `_new` tasks (8 shots, 200 trials, probe seed 5000, chance 0.125;
filler between EVERY example, so L = 1024 is ~110-token gaps, 4096 ~500):

| `_new` @ L | 0 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| PAM s42 / s43 shift_wide | 1.00 / 1.00 | 0.79 / 0.735 | **0.60 / 0.53** | **0.34 / 0.295** | 0.105 / 0.12 |
| Mamba-2 shift_wide | 0.765 | 0.00 | 0.00 | 0.00 | 0.00 |
| T++ shift_wide | 0.315 | 0.00 | 0.00 | 0.00 | 0.00 |
| PAM s42 / s43 affine_map | 1.00 / 1.00 | 0.965 / 0.995 | **0.965 / 0.975** | **0.785 / 0.905** | 0.435 / 0.61 |
| Mamba-2 affine_map | 1.00 | 0.425 | 0.225 | 0.11 | 0.055 |
| T++ affine_map | 0.975 | 0.075 | 0.05 | 0.45 | 0.285 |
| PAM s42 / s43 step_wide | 1.00 / 0.925 | 0.115 / 0.09 | 0.14 / 0.065 | 0.10 / 0.085 | 0.105 / 0.04 |
| Mamba-2 step_wide | 1.00 | **0.305** | **0.31** | **0.27** | **0.22** |
| T++ step_wide | 1.00 | 0.09 | 0.085 | 0.09 | 0.055 |

**Verdict (amendment rule 2: PAM `_new` mean ≥ b2m + 0.10 in BOTH PAM
seeds): NOT MET.** Seed 42 +0.099, seed 43 +0.052. Hence (rule 3) **bar 3 is
not replicated in its pre-registered form; no ntfy.** Mamba-2's only miss in
context is shift `_new` (0.68).

**Across distance (the amendment's reported cells, named before b2m ran:
shift/affine `_new` at L = 2048 and 4096, margin +0.10): PAM ahead in every
cell, both seeds.** Mean of the 4 cells D: PAM 0.673 / 0.676, Mamba-2 0.084,
T++ 0.125. So the distance edge is NOT "recurrence vs attention": Mamba-2 is
recurrent and O(1) too, and it collapses with ~110-token gaps exactly like T++.
Only PAM carries an inferred rule through thousands of tokens of unrelated
text, past its trained length (affine 0.61 at 8192 for seed 43).

**Reading.**
- Two kinds of memory. Mamba-2 keeps exact items better: labels 0.79 (best of
  all), the last far number through filler (step 0.27–0.31 vs PAM ≤ 0.14), and
  holdout slightly better (−0.002 / −0.005). PAM keeps the RULE. That fits
  their state: Mamba-2's input-dependent Δ chooses which tokens to keep.
  PAM writes complex outer products, so a rule that repeats as the same phase
  relation across 8 examples adds up coherently while filler adds incoherent
  noise. A single exact item has no such advantage (guess; test below).
- Labels: Mamba-2 is the leader (0.787), not T++. The lb pair compares lbp
  only with lbt, so its rule 2 is now too weak (amendment below).
- Bar 2 at 600M with this mix: Mamba-2 ≤ PAM on holdout by 0.002–0.005 (noise
  level), PAM ahead on far gain (+0.042 vs +0.022).

**Amendments (fixed now, before any of these run):**
1. *lb pair:* if lbp passes its rule 1 (label_diag mean ≥ 0.767), an `lbm`
   arm (Mamba-2 + lab, stage A `b2ma_pre`, seed 42) runs before any label
   claim; "gap closed / PAM ahead" are then judged against max(lbt, lbm).
   If lbp fails rule 1, no lbm.
2. *Distance confirmation (bar 3, distance form), tmux `s8`, 6000, code
   `38e3c29`, after s7.* The distance cells were named before b2m ran; only
   their promotion to the bar is new, so the test uses NEW training seeds and
   a NEW probe sample:
   - Runs (stage B as b2p1 / b2t / b2m; stage As reused): `b2t2` T++ seed 43,
     `b2m2` Mamba-2 seed 43, `b2p3` PAM seed 44.
   - Probe: `--suite b2 --tasks shift_wide_new,affine_map_new --shots 8
     --trials 400 --seed 7000 --spread 0,2048,4096` (probe seeds 7000–7399,
     never used before) on all 7 models. D = mean of the 4 cells at 2048 and
     4096.
   - **Primary:** D(b2p3) ≥ max(D(b2t2), D(b2m2)) + 0.10.
   - **All pairs:** every PAM seed (42, 43, 44) ≥ every baseline run (b2t,
     b2t2, b2m, b2m2) + 0.10 on probe seed 7000.
   - Both hold → bar 3 met in its distance form (in-context learning of
     held-out rule values that survives distance past the trained length; T++
     and Mamba-2 do not) → EUREKA + ntfy, params and tokens stated. Either
     fails → no claim. The in-context form stays failed vs Mamba-2 either way
     and is reported with it.
   - Reported, no bar: L = 0 and the standard probes for every new run,
     holdout, recall, step_wide_new spread.
   - Caveat that stays with any claim: trained rule FORMATS with held-out
     VALUES, not unseen rule families.

## Gate diagnostic + test-time gate floor (2026-10-31 07:30 IST; CPU, no training): soft write gates are selective by rank but weak; the label gap is NOT filler burial

Why: Mamba-2 keeps exact items better (labels, step through filler), and v14
read that as "PAM's write cannot be refused". Scripts: `v13_sempty/tmp/gates_diag.py`
(token classes item / label / format / filler in spread prompts; gates read via
hooks on `_delta_gates`, `_decay` and `echo_gates`) and
`v13_sempty/tmp/gate_floor_probe.py`. Outputs: `logs/diag/v13_sempty_gate*_3d7d889_20261031.txt`.
Models: `c1m2b_main_7ae6f1f` (M2 body, no b2 docs) and `b2p2_main_38e3c29`.

**The knob already exists.** A3 delta (on in every M2 run) writes with
b_w = sigmoid(write_proj(x)) per token and head, separate from the decay. The
echo memory has its own dt / write / erase gates. The coupled Mamba-2 form was
C1 (FAIL).

**What the gates do** (label prompts at L = 2048 / 4096, 20 / 12 prompts):
- Long memory is the echo: filler half-lives of the 6 heads 660–3100 tokens.
  Notebook heads are mostly 1–46 tokens, plus a few at 150–740 (b11 h4 ~700, b3 h5 ~180).
- Write gates are selective by rank: AUROC(item/label > filler) 0.84–0.95 in
  echo, b7 and b11; b3 0.60. By size they are weak: echo filler write 0.12–0.14
  and erase ~0.11 per token, vs 0.2–0.5 at items. Filler outnumbers items ~100:1.
- PAM learned a soft write-forget tie: echo retention drops at item/label
  tokens (c1m2b h2: filler half-life 1236 → 159 at items).
- With b2 docs, on affine prompts this becomes a reset: echo half-life at rule
  tokens 11–264 (c1m2b 107–611), while filler is still 700–2500. Reading: the
  model overwrites a rule register at each example and carries the LATEST one
  across weakly written filler. That replaces the earlier "coherent
  integration" guess (neither reading tested).

**Test-time floor** (b2p2, label_diag 8 shots, 80 trials, ±0.05; gates < tau → 0):

| | category @0 / 2048 / 4096 | exact @0 / 2048 / 4096 |
|---|---|---|
| no floor | 0.51 / 0.33 / 0.21 | 0.61 / 0.26 / 0.21 |
| echo, tau 0.2 | 0.49 / 0.33 / 0.20 | 0.54 / 0.29 / 0.23 |
| echo, tau 0.3 | 0.48 / 0.30 / 0.26 | 0.53 / 0.34 / 0.29 |
| all memories, tau 0.2 | 0.46 / 0.28 / 0.15 | 0.50 / 0.35 / 0.15 |

At most +0.05 to +0.08 at distance (echo, tau 0.3; ~1.5 SE), at a cost of
−0.04 / −0.09 in context. Not a rescue, and an untrained floor is a weak test.

**The deciding fact is short context** (label_diag, 200 trials, no filler):

| exact @ 2 / 4 / 8 shots | PAM b2p1 | PAM b2p2 | Mamba-2 b2m | T++ b2t |
|---|---|---|---|---|
| label_exact | 0.60 / 0.675 / 0.49 | 0.63 / 0.735 / 0.615 | **0.865 / 0.905 / 0.815** | 0.725 / 0.825 / 0.705 |

With 2 shots the prompt is ~15 tokens, so there is nothing to bury. PAM's label
gap is a short-context lookup / binding gap. Filler burial adds to it at distance
but is not its cause. The recall docs (one trained template) gave N9a 0.975, so the
format hypothesis is the live one: the lb pair (running) tests it. *For v14's
N12:* a hard selector targets distance; judge it on 2-shot label_exact too.

## 3B LM gap attribution (2026-10-31 08:25 IST; eval only, no training): vs T++ the gap is mostly in-window copy; vs Mamba-2 it is mostly general modelling

Why: before a 3B "best package" run, know which part of the bar-2 gap a memory
fix can close. Script `v13_sempty/tmp/attr3b.py` (dump → score per box → report;
Mamba-2 scored on the 6000, which has `mamba_ssm`, on the identical frozen text).
Models: the 3B scoreboard runs PAM `1c913ef` (old 16nb body, no echo, one stage,
2e-4), T++ `e072a8f`, Mamba-2 `b36e3a5`, plus r3b `685de4e` (M2 + echo + two
stages). Text: 60 PG-19 windows of 2048 tokens (skip 1024 into each book) and
150 web holdout docs (≥ 512 tokens, from the doc start). Token classes:
**copy** = the bigram (x_t, y_t) occurred earlier in the window, **seen** = the
target occurred earlier but not after this token, **new** = first occurrence.
Output: `logs/diag/v13_sempty_attr3b_4a4a5fb_20261031.{txt,json}`.

| PG-19 / web | loss PAM 3B | T++ | Mamba-2 | r3b |
|---|---|---|---|---|
| all tokens | 3.560 / 3.247 | 3.510 / 3.183 | **3.469 / 3.136** | 3.625 / 3.283 |

| gap (PAM − base), PG-19 / web | vs T++ | vs Mamba-2 |
|---|---|---|
| total | +0.050 / +0.064 | +0.091 / +0.112 |
| copy tokens (26 % / 23 % of tokens): gap, share | +0.113 / +0.148, **58 % / 53 %** | +0.083 / +0.116, 24 % / 24 % |
| seen tokens (38 % / 39 %) | −0.002 / +0.017, −1 % / 10 % | +0.052 / +0.071, 21 % / 25 % |
| new tokens (37 % / 39 %) | +0.060 / +0.061, 43 % / 37 % | +0.136 / +0.150, **55 % / 52 %** |
| positions 0–8 / 8–64 / 64–512 / 512–2048 (PG-19) | −0.090 / −0.041 / +0.045 / +0.056 | +0.048 / +0.061 / +0.085 / +0.094 |
| same on web | −0.464 / +0.015 / +0.053 / +0.079 | +0.095 / +0.113 / +0.107 / +0.115 |
| new tokens only, 0–8 → 512–2048 (PG-19) | −0.095 → +0.068 | +0.062 → +0.145 |

**Reading.**
- *vs T++:* over half the gap is exact in-window copy (+0.11 to +0.15 nats on
  the copyable quarter of tokens). The rest is new tokens late in the window:
  attention uses far context to predict unseen words better. PAM is better at
  the start (T++ is poor in the first 8 tokens of a web doc, −0.46). A binding /
  copy fix (N12, lab / recall formats) can close up to ~0.03 of this ~0.06.
- *vs Mamba-2:* copy is only a quarter. Over half is new tokens, and Mamba-2 is
  already 0.05–0.09 better on new tokens at positions 0–8, with almost no
  context. The gap then grows with context for every class. So the Mamba-2 gap
  is mostly GENERAL modelling (local + context semantics), not memory. Memory
  units can close at most ~0.02–0.03 of ~0.10; the rest needs the body.
- *r3b vs the old body:* r3b's extra loss (+0.036 / +0.065) is almost all on new
  tokens (67–99 %). Its copy is equal or better, because echo helps copy. The M2
  body (4 notebooks) lost general modelling capacity. That fits r3c (16nb +
  echo) as the right body test; score r3c with this script at its futility point.
- Caveat: one seed per model; the PAM 3B reference predates echo and the
  two-stage schedule (ledger).

**What it changes:** the best-package 3B run needs BOTH a copy fix (worth ~half
of the T++ gap) and a general-modelling gain (most of the Mamba-2 gap). The
general part is the body's: r3c's result, then capacity levers (width 588 vs
768; state 3.7 MB vs Mamba-2's 13.4 MB, "a few more params" is allowed). None of
these has been tested at ≥ 1.5B.

**LR confound check (v14's question, 08:45 IST).** Is the new-token gap just the
2e-4 run being under-trained? The same old body at LR 1.2e-3 (`9982a79`, 3B, one
stage) scored on the same text is WORSE overall (+0.043 PG-19 / +0.013 web vs
2e-4). Its gap to Mamba-2 is +0.134 / +0.124 with the same split: new tokens 46 %
/ 49 %, copy 24 %. New tokens at positions 0–8: +0.027 / +0.174. So for the old body
the general-modelling gap is not an LR artifact (both LRs). r3c (16nb + echo +
two stages, 1.2e-3) is the open case; the baselines kept no intermediate
checkpoints, so equal-token comparison by token class is impossible (equal-token
TOTAL loss is possible from the val curves). Output:
`logs/diag/v13_sempty_attr3b_lr12_061869d_20261031.txt`.

**Candidate for the general part: A4 `cond_mem`** (hashed n-gram conditional
memory: in the code, never run at scale; flag `--cond_mem`, slots 262144 ×
dim 64, layers 0 and 1). It targets exactly short-context next-token prediction,
with O(1) lookups and extra params (bar-2 tolerance allows them). Fairness caveat:
an n-gram table would help any architecture, so a bar-2 claim with it must say so.
Plan: a paired 600M filter run (kill if no gain on new tokens at positions 0–64 by
this script), then into the best package. Not pre-registered yet; that happens
after r3c's futility point.

## A4 cond_mem filter `a4f` (pre-registration, 2026-10-31 08:50 IST): does a hashed n-gram table close the new-token gap?

Why: the 3B attribution puts 52–55 % of the Mamba-2 gap on NEW tokens, already
+0.05–0.09 at positions 0–8, where no memory helps. A4 (`v13_sempty/cond_mem.py`,
Engram-style: canonical-id suffix 2/3-grams, 2 hash heads each, one shared
2^18 × 64 table per read, gated by the hidden state, read after blocks 0 and 1;
table lr × 5, no weight decay) targets exactly short-context next-token
prediction, with O(1) lookups. It has been in the code since `38e6595` and was
never run. Launcher switch `COND_MEM=1` (`d338469`; smoke in that commit).

**Arm** (4090, worktree `../qllm_a4`, code `d338469`): v14's n9r control recipe
exactly, plus `COND_MEM=1` in both stages. Body M1 (`PAM_LAYERS=3,7,11,15
MLP_HIDDEN=2352 ECHO=1 GRAD_CKPT=1`), LR 1.2e-3, seed 43. Stage A 120M
dclm / fineweb / smoltalk2_mid 0.4167 / 0.3333 / 0.25 (30M blend warmup);
stage B 480M M1 mix (0.44 / 0.41 / 0.10 / 0.05 recall), `--init_from` A.
**Control:** `n9rctl43{a,b}_37fa333` (same box, seed, data; v14's run). Code
differs only by additive switches (v14 hooks off, launcher switch). Params: arm
103.6M dense + 33.5M table (137.1M with table) vs 102.9M. The 4090 holdout runs
~0.12 above the 6000's for this recipe (box effect), so compare only within the 4090.

**Measures:** holdout; `attr3b` on the frozen text (new-token loss at positions
0–64 and overall, copy / seen / new split); recall 63-cell; label_diag 200;
rules 100; tok/s.

**Rules (fixed now):**
1. *KEEP for the 3B best package* if holdout ≤ ctl43 − 0.02 AND new-token loss
   at positions 0–64 is lower by ≥ 0.03 on BOTH PG-19 and web.
2. *FAIL* if the holdout gain is < 0.01: A4 is removed from the code (index row
   with the code-at hash).
3. In between: not in the package; reported, and decided at 3B only if
   nothing better exists.
4. *Guards:* recall ≥ ctl43 − 0.05; tok/s cost reported.
5. *Fairness, stated with any bar-2 claim that uses it:* an n-gram table helps
   any architecture. If it goes into the package, the bar-2 report must say so,
   and the claim must state how much of any lead it explains.
6. One seed. A 600M pass gets judged again at 3B, not claimed at 600M.

**a4f amendment (2026-10-31 13:01 IST, before its result lands ~16:50):** the paired
control `ctl43` is a loss outlier (v14, n9r): stage A 3.9456 vs a4f 3.8994, n9a43 3.9009
(same seed and box), C1-ctl 3.9231; stage B 3.598 vs n9a43 3.459, C1-ctl 3.4743. The
"4090 box effect" I assumed does not exist (n9c 3.4725, n9a43 3.459 on the 4090). So the
**reference for every a4f rule is C1-ctl** (`c1ctlb_main_7ae6f1f`, holdout 3.4743, seed 42;
checkpoint on d_drive, attr3b-able locally); `ctl43` is reported, not used. Thresholds
unchanged (holdout ≤ ref − 0.02; new-token loss @0–64 ≤ ref − 0.03 on PG-19 and web). The
chain's attr3b step scores ctl43; C1-ctl is scored right after with the same frozen text.
Seed differs (43 vs 42); seed noise on holdout at this recipe is ~0.003 (b2p1 / b2p2).

## Label-format pair `lb` result (2026-10-31 12:50 IST): PASS — the label gap was a data-format gap; PAM beats T++ on labels on the same data

`lbp_main_c4173be` (PAM M2 + echo + shared reads, 102.9M) vs `lbt_main_c4173be`
(T++ 101.9M); 600M tokens each, stage B with 3 % `lab` docs (probe format,
disjoint vocabulary), seed 42, 6000.

| 600M tokens | holdout | WikiText | recall 63 (≤ 2048 / > 2048) | PG-19 far gain 4–8k | label_diag cat / exact (mean) | category @ 2 / 4 / 8 shots | rules label_cat (100 / 400) | two_hop |
|---|---|---|---|---|---|---|---|---|
| **PAM lbp** | **3.4516** | 66.00 | **0.846** (0.945 / **0.600**) | **+0.044** | **0.873 / 0.995 (0.934)** | **0.84 / 0.88 / 0.90** | **0.877 / 0.891** | 0.31 |
| T++ lbt | 3.4927 | **65.80** | 0.667 (0.877 / 0.142) | −0.94 | 0.608 / 0.987 (0.798) | 0.425 / 0.605 / 0.795 | 0.587 / 0.626 | 0.287 |
| without lab: PAM b2p1 / c1m2 | 3.4777 / 3.4473 | | 0.721 / 0.870 | | 0.567 / – | 0.52 / 0.65 / 0.47 (b2p1) | 0.557 | 0.29 |

LCL label_category @ 0 / 1024 / 2048 / 4096 / 8192 (8 shots, 300; in-context
chance 0.25): PAM 0.903 / 0.413 / 0.337 / 0.277 / 0.207; T++ 0.76 / 0.48 / 0.393 /
0.14 / 0.133. two_hop at chance for both at every L.

**Verdict (pre-registered rules):** rule 1 **PASS** (0.934 ≥ 0.767); rule 2
**PAM ahead** (+0.136 ≥ +0.05); guard holdout 3.4516 ≤ 3.487 (cost vs c1m2
only +0.004, where the b2 docs cost +0.030); recall 0.846 (−0.024 vs c1m2; the
b2 docs cost −0.15). Labels across distance: reported, both at chance by 2048.

**Reading.**
- The "exact lookup" weakness at short range was a FORMAT gap. With the format
  trained (on different words), label_exact is 0.995 at 2 shots and category
  0.84–0.90. T++ also reaches exact 0.987 but category only 0.608: PAM
  generalises the semantic binding (new member of a shown category) better.
- Skills are format-specific: lbp has no b2 docs and is below chance on
  letter / number rules (0.003 / 0.05), as c1m2 was. The next recipe trains b2
  + lab + recall together (r3d).
- Lookup across distance still decays to chance by 2048 for both bodies, with
  the format trained. That lookup belongs to a library, not the state
  (direction below).
- Caveats: one seed; `lbm` (Mamba-2 + lab, tmux `s9` after s8) runs before any
  label claim (amendment 1), and a second PAM seed before any claim.

## s8 part 1 (2026-10-31 12:04 IST): fresh probe seed 7000 on the four finished B2 models

`--seed 7000 --trials 400 --spread 0,2048,4096`, shift_wide_new + affine_map_new,
8 shots. D = mean of the 4 cells at 2048 / 4096.

| | shift @ 0 / 2048 / 4096 | affine @ 0 / 2048 / 4096 | **D** |
|---|---|---|---|
| PAM b2p1 (s42) | 1.00 / 0.595 / 0.278 | 1.00 / 0.935 / 0.772 | **0.645** |
| PAM b2p2 (s43) | 1.00 / 0.583 / 0.300 | 1.00 / 0.970 / 0.912 | **0.691** |
| T++ b2t | 0.282 / 0.00 / 0.003 | 0.97 / 0.033 / 0.502 | 0.134 |
| Mamba-2 b2m | 0.762 / 0.00 / 0.00 | 1.00 / 0.302 / 0.160 | 0.116 |

On a never-used probe sample the old-model half of the "all pairs" rule holds
with margin ≥ +0.51 (bar +0.10). The verdict waits on the new training seeds
b2t2 (training), b2m2, b2p3 (~18:15 IST).

## Direction (user decision, 2026-10-31 ~12:30 IST): O(1) mind + growing library

The O(1) body stays and is the point: it is the MIND (rules, gist, reasoning,
learning that holds past the trained length). What is dropped is the demand
that the O(1) state also be the LIBRARY. Verbatim knowledge lives in libraries:
a per-session library (the context and tool results, as text streams in) and a
shared, growing library (corpora; later community packs: bio, coding, a
framework). "Skill = skill + lookup data." Intelligence separate from knowledge.

Why now (evidence): copy is 53–58 % of the 3B LM gap to T++; a model-free exact
organ recalls 1.0 up to 128k where PAM, T++ and Mamba-2 all fade to 0 by 32k
(Communicator, a1baefc); PAM's own pre-logit vector at the key token retrieves
the right sentence top-1 0.63–0.88 to 128k zero-shot, 1/4 at 400k (so the fetch
key must be trained past ~128k); the frozen PAM re-reading two fetched spans
answers two-hop 1.0; rule-at-distance is PAM's alone (above); short-range
binding is fine once the format is trained (lb pair).

**Order (user): a pre-registered benchmark first, the distributable model once
it wins that table.** Benchmark (spec to be fixed in its own commit before any
package run): needle, multi-needle, latest-definition-wins, two-hop,
rule-at-distance and instruction persistence at 2k–1M with the library,
against Transformer++ with FULL context (and Mamba-2), memory in MB in the same
table (mind state + library index vs KV cache), plus general LM loss by token
class so the mind's quality is never hidden.

What does not change: the mind must stay a competent general LM (the new-token
gap to Mamba-2 is a mind problem; the library does not fix it: r3c, body
levers). Novelty must be in the mind–library interface (a learned fetch keyed
in the mind's own space; rules carried in state), since retrieval itself exists.
Lanes: v14 designs the learned fetch (gate, trained key at the salient token,
re-read); v13 builds the benchmark, the training recipe that teaches the mind
to use fetched text, and the package runs.

## Step back after the pivot (2026-10-31 13:10 IST): the mind is made by its curriculum; the shared library is knowledge

Full note: SCRATCHPAD 13:10. The useful part:
- **Low prior, high ceiling.** Each in-context skill was a format gap (b2: rules
  0.05 → 0.99; lab: labels 0.57 → 0.934 on unseen words; recall docs). With the
  format PAM beats T++ on the same data; without it, it trails. Its in-context
  intelligence is made by the skill curriculum, so the curriculum is part of the
  architecture, and family-level generalisation (unseen rule FAMILIES, not only
  values) is the test that it learns to learn.
- **Correction: the shared library can attack the new-token gap.** The attribution
  put 52–55 % of the Mamba-2 gap on new tokens. The session library cannot help
  there; a shared corpus library can (world knowledge, kNN-LM / RETRO style),
  keyed by the mind's own state. Zero-training test first.
- **The organiser's stated-rule failure is confounded** (c1ctlb has no rule
  skill at all). The clean question: can words write the rule state that
  demonstrations write, in a model that has the skill (b2p)?
- **Product shape:** skills in the mind (curriculum), episodes in the session
  library (keyed at surprise), knowledge in the shared library (same key
  space), packs versioned with the mind; later, consolidation from session to
  shared library and slow skill fine-tuning.

## SL0: shared library as knowledge, zero training (pre-registration, 2026-10-31 13:11 IST)

**Closest prior work (novelty rule, added 13:27 IST):** kNN-LM (Khandelwal et al. 2020); RETRO for chunks. **What differs:** nothing in the mechanism. SL0 is kNN-LM applied to a PAM mind. It is reported as the knowledge organ's cost table (rows, MB, install cost, loss by token class) and never claimed as an invention.

Question: does a shared library keyed by the mind's own state close the 3B
new-token gap to Mamba-2 (+0.091 PG-19 / +0.112 web; 52–55 % on new tokens)?
kNN-LM style (Khandelwal et al. 2020, a known part): datastore = (PAM's
pre-logit vector at training position t → token t+1) over training text; at
eval, p = λ·p_kNN + (1 − λ)·p_LM, p_kNN(y) ∝ Σ over the k nearest keys with value
y of exp(−d / T).

- **Mind:** frozen PAM 3B `1c913ef` (the attribution's reference). **Baseline:**
  Mamba-2 3B `b36e3a5` as is (user's comparison rule), its per-token losses
  already on the frozen attr3b text.
- **Library:** 5M tokens of the training mix (dclm + fineweb, training split
  only: the frozen eval text is the holdout cache and PG-19 test, so no leak).
  Keys: the 588-d pre-logit vector, PCA to 128-d fp16 (PCA fitted on 50k training
  vectors); exact search (chunked brute force on the 4090, k = 64, squared L2).
- **Tuning split:** λ ∈ {0.1, 0.2, 0.3, 0.4}, T ∈ {1, 3, 10, 30} chosen on the
  first third of the frozen docs (PG-19 0–19, web 0–49). **Scored** on the other
  two thirds only (PG-19 20–59, web 50–149), for PAM, PAM + library and Mamba-2.
- **Rules (fixed now):**
  1. *Library helps the mind:* PAM + library ≤ PAM − 0.05 nats on BOTH test sets.
  2. *Closes the Mamba-2 gap:* PAM + library ≤ Mamba-2 on BOTH test sets.
  3. Reported: loss by token class (copy / seen / new) and position; library
     tokens, MB (keys + values), install cost (library tokens × 1 mind pass, GPU
     s), query cost per token.
- 5M tokens is 0.17 % of the 3B training data. kNN-LM gains grow with the
  datastore, so a pass at 5M is followed by a 20–50M datastore. A fail at 5M is
  not a verdict on size; it is reported as a 5M result.
- Caveats that travel with it: kNN-LM exists; this measures it for the O(1) mind
  with Mamba-2 as is; it does not touch in-context copy (the session library's
  job).

**SL0 amendment (2026-10-31 13:19 IST, after a 60k-row pipeline smoke and before the
scored run).** The smoke (60k rows, not a scored run) showed the pre-registered T grid
{1, 3, 10, 30} is degenerate: squared-L2 distances are ~4.5k, so the weights collapse
to the nearest neighbour (mean p_kNN equals the top-1 hit rate exactly). Pipeline sanity:
the right next token is among the 64 neighbours 47–48 % of the time, top-1 8–11 %.
Changes, fixed now: T is in units of the median nearest distance on the tuning docs,
T ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1}; λ ∈ {0.02, 0.05, 0.1, 0.2, 0.3}. Library sizes:
**5M rows (primary)** and **20M rows (second point, reported)**; build packs training
text into 2048-token windows (8 per forward), like training. Rules unchanged. Smoke at
60k with the new grid: +0.006 / +0.007 vs the mind alone (worse, as expected at this size).

### SL0 result, 5M rows (2026-10-31 13:27 IST; the 20M point is running)

Library: 5,000,000 rows from 3,876 training docs; 1,300 MB (1,280 MB keys + 20 MB
values); **install 106 s** (one pass of the frozen 3B mind over 5M tokens on the 6000,
~47k tok/s at the end); query 243 s for the 213k scored + tuning tokens (exact search).
Tuned on the first third: λ = 0.02, T = 0.3 × median nearest distance (both sets).

| scored 2/3 | all | copy | seen | new | new @ 0–64 | 0–8 |
|---|---|---|---|---|---|---|
| PG-19 PAM 3B | 3.4892 | 1.3060 | 2.7777 | 5.8498 | 5.2488 | 4.9085 |
| PG-19 PAM 3B + library | 3.4878 | 1.3110 | 2.7792 | 5.8407 | 5.2402 | 4.9033 |
| PG-19 Mamba-2 3B | **3.3970** | 1.2162 | 2.7280 | 5.7127 | 5.1322 | 4.7936 |
| web PAM 3B | 3.2274 | 1.3351 | 2.6167 | 4.9452 | 4.3982 | 4.0984 |
| web PAM 3B + library | 3.2234 | 1.3400 | 2.6149 | 4.9337 | 4.3882 | 4.0827 |
| web Mamba-2 3B | **3.1155** | 1.2197 | 2.5444 | 4.7958 | 4.2746 | 4.0167 |

**Rule 1 (≤ PAM − 0.05): FAIL at 5M** (−0.0014 PG-19, −0.0040 web). **Rule 2 (≤ Mamba-2):
FAIL** (+0.091 / +0.108). The small gain is where expected (new tokens −0.009 / −0.012;
copy slightly worse, +0.005). The tuner picked the smallest λ: at 0.17 % of the training
data, the library rarely holds the right continuation. This is a 5M result, not a verdict
on size (pre-registered); 20M reported next.

### Stated rule vs demonstrations on b2p2 (Lead's instr cell, 2026-10-31 ~13:40 IST; EUREKA.md top)

Trained format ("Letter rule." header), held-out shift values, 8 docs per cell (±0.15):
the rule stated once in words at the start scores **0.000 at 2k / 8k / 32k, and 0.000
even with the rule sentence re-read right in front of the query**; with 2 demonstration
lines before the query it scores 0.625 / 1.000 / 0.750 (oracle re-read 1.0).
**Reading:** examples write the rule state, words do not. A real skill gap, not a bench
artefact. Consequences: (1) the curriculum run needs "stated rule + demonstrations" docs
(so words learn to set the same state); Lead's suggested bar for that cell, taken into
the pre-registration: 0-shot stated rule ≥ 0.75 at 32k, inside and past the trained
length. (2) Skill packs must carry demonstrations, not only a rule sentence.

### SL0 verdict (2026-10-31 13:59 IST): the knowledge library is a cost table here, not a fix for the new-token gap

| library (rows) | MB (keys + values) | install (one mind pass, 6000) | PAM 3B + library − PAM 3B (PG-19 / web) | gap to Mamba-2 after (PG-19 / web) |
|---|---|---|---|---|
| 5M | 1,300 | 106 s | −0.0014 / −0.0040 | +0.091 / +0.108 |
| 20M | 5,200 | 328 s | −0.0059 / −0.0119 | +0.086 / +0.100 |

Query (exact, chunked brute force) 243 s at 5M, 951 s at 20M for ~320k query tokens.
Tuned λ = 0.02 for both sizes; T = 0.3 / 0.1 × median nearest distance.

**Verdict (pre-registered): rule 1 FAIL at 5M and at 20M; rule 2 FAIL.** 4× the rows
bought ~3× the gain; at that rate a −0.05 gain needs roughly 200M+ rows (~52 GB of keys
at 128-d fp16) and ANN search, and closing the Mamba-2 gap far more. The gain sits on
new tokens (web −0.026 at 20M) and at early positions (web 0–8 −0.036), as kNN-LM
predicts, but it is small. **Reading (novelty rule):** SL0 is kNN-LM on a PAM mind,
plumbing, never a claim. Its value is the knowledge organ's cost table (rows, MB,
install, query). It does not close the general-modelling gap to Mamba-2 at this size;
that gap stays the mind's (r3c, body levers). Next priority (Lead): the skill-pack
cell, not more library rows. Logs: `logs/diag/v13_sempty_sl0_*`.

## Skill-pack persistence, first rows (2026-10-31 14:00 IST): PAM carries a rule ~2k tokens after ONE read; the spread cell overstated the carrying distance

MLB-1 v1 R7 cell (`scripts/run_skillpack_probe.py` @ `0029128`, 6000, 100 docs per cell,
probe seeds 8000–8099): the B2 header + 8 demonstration lines (held-out values) read
ONCE at t0, then L tokens of held-out filler, then the query. A_state = read once;
O_reread = the pack re-placed right before the query (T++'s "re-fetch into a fresh
window, KV = pack + query"). Chance 0.125. b2m (Mamba-2) and b2p1 are still running.

| L | PAM b2p2 affine (A_margin) | PAM shift | T++ b2t affine (A_margin) | T++ shift | T++ KV, whole stream |
|---|---|---|---|---|---|
| 0 | 1.00 (+18.3) | 1.00 | 0.96 (+5.1) | 0.24 | 1.2 MB |
| 2k | **0.98 (+4.1)** | **0.74** | 0.06 (−1.0) | 0.00 | 67 MB |
| 8k | 0.14 (−1.8) | 0.05 | 0.39 (−0.0) | 0.00 | 266 MB |
| 32k | 0.07 (−1.9) | 0.01 | 0.08 (−1.4) | 0.00 | 1,058 MB |
| re-fetch (any L) | 1.00 | 1.00 | 0.96 | 0.24 | 1.23 MB per query |

PAM state 1.15 MB (measured, M2 body). The T++ last-position check passed at rel err
8.2e-3 (bf16 on GPU).

**Reading.**
- After ONE read PAM carries the rule ~2k tokens (affine 0.98 at 2k, inside the trained
  length, where T++ already fails at 0.06 with the pack in its window) and loses it by
  8k. That is about the training length (2048) and fits the echo memory's half-lives
  (660–3100 tokens).
- **Correction to the distance headline.** In the spread cell (s8, B2 pair) the
  examples are spread evenly, so the LAST example is L/8 from the query (~1k tokens at
  L = 8k). "Holds affine to 8k" was carrying across ~1k gaps, repeatedly re-exposed;
  the total context exceeded the trained length, the single gap did not. The edge over
  T++ and Mamba-2 is real (they fail at ~110-token gaps), but its single-exposure reach
  is ~2k.
- **My written prediction was wrong in the direction that matters:** I predicted the
  skill would be lost between 8k and 32k. It fades between 2k and 8k. R7 (at t0 + 32k,
  PAM ≥ max(...) + 0.10) fails: 0.07 vs T++ 0.08 (Mamba-2 pending).
- A_margin shows the fade before accuracy (affine +18.3 → +4.1 at 2k → −1.8 at 8k).
- What it says: the carrying distance looks LEARNED from the training length (the
  model never had to carry beyond 2048 in training). An O(1) mind can train at 8k–32k
  for the same cost per token. That makes the long-gap rule-doc recipe at long training
  lengths the direct test (next pre-registration), with Library's structural skill
  register as the other route. Re-fetch is cheap for both bodies (1.15 MB state vs
  1.23 MB KV per query), so the skill library's value at > 2k is "re-read the pack",
  not "carry it".

## LAW-1: does an O(1) mind's one-read skill reach track its training length? (pre-registration, 2026-10-31 14:10 IST)

**Hypothesis (primary, Lead's framing): the law.** After one read, PAM's reach for an
induced rule is set by the length it was trained at (b2p2, trained at 2048: skill-pack
reach ~2k, gone by 8k). If the law holds, "holds beyond the training context" becomes
"reach is set by the training length, and extending it is cheap for an O(1) mind" (per-
token cost does not grow with context; a transformer's does). If reach stays ~2k at
every training length, it is a ceiling of this body and Library's skill register is the
only route.

**Closest prior work (novelty rule):** long-context continued pretraining (e.g.
Fu et al. 2024, "Data engineering for scaling language models to 128K context") and
length-extrapolation training of SSMs / linear RNNs. **What differs:** the measured
quantity is one-read RULE reach (an induced skill carried in O(1) state across unrelated
text), not recall or perplexity; the claim, if it holds, is a law for training an O(1)
mind. The recipe itself (train longer, with long-range docs) is known.

**Data: new source `b2gap`** (v7/data.py, to be committed with tests before any run):
B2 base tasks only (trained values; held-out `_new` values never appear; example seeds
≥ 10^7, probe seeds 8000–8099), in the probe's layout: header + 8 demonstrations, then 1–3
queries each after a gap of training-split web text (fineweb, a separate iterator) of
G ~ U(0, G_max) tokens, answers marked (answer weight 100). G_max is set per arm.

**Arms** (fine-tunes from `b2p2_main_38e3c29`, 150M tokens each, LR 3e-4 with warmup 100,
cosine; stage-B mix with `b2` replaced by `b2gap` at 3 %; 32,768 tokens per step in every arm):
- `lg2`: SEQ 2048 × BATCH 16, G_max 1,800 (control: same data and steps at the old length);
- `lg8`: SEQ 8192 × BATCH 4, G_max 7,500;
- `lg32`: SEQ 32768 × BATCH 1, G_max 31,000.

**Measure:** the skill-pack cell as fixed (`scripts/run_skillpack_probe.py`, seeds
8000–8099, 100 docs) at L = 0 / 2k / 4k / 8k / 32k (+ 128k for lg32). Reach(L) = mean
A_state of affine_map_new and shift_wide_new. Today b2p2: 1.00 / 0.86 / – / 0.10 / 0.04.

**Rules (fixed now):**
1. *Law:* reach_lg8(8k) ≥ 0.75 AND reach_lg32(32k) ≥ 0.75.
2. *Not just more training:* reach_lg2(8k) ≤ reach_b2p2(8k) + 0.15 (the control does not
   extend the reach).
3. *Guard:* holdout ≤ b2p2 + 0.02 (3.500) for every arm; spread B2 `_new` at L = 0 ≥ 0.90.
4. Reported: reach_lg8 at 32k (4× beyond its training length), lg32 at 128k; T++ and
   Mamba-2 at the same L as they are (b2t, b2m, b2t2, b2m2); tokens/s per arm.
5. One seed. If rule 1 holds, a second seed and the 3B question follow before any claim.
6. If rule 1 fails while rule 2 holds: reach is a ceiling of the body; record it, and
   the register (Library, SR) is the route.

### Skill-pack persistence, four models (2026-10-31 14:05 IST)

One read at t0, 100 docs per cell, seeds 8000–8099, chance 0.125. Per cell: accuracy (mean A_margin).

| model (600M tokens, ~102M) | affine @ 0 / 2k / 8k / 32k | shift @ 0 / 2k / 8k / 32k | re-fetch (affine / shift) | state |
|---|---|---|---|---|
| PAM b2p1 (s42) | 1.00 / **0.91** (+2.6) / 0.05 / 0.08 | 1.00 / **0.80** (+1.6) / 0.02 / 0.02 | 1.00 / 1.00 | 1.15 MB |
| PAM b2p2 (s43) | 1.00 / **0.98** (+4.1) / 0.14 / 0.07 | 1.00 / **0.74** (+1.0) / 0.05 / 0.01 | 1.00 / 1.00 | 1.15 MB |
| T++ b2t | 0.96 / 0.06 (−1.0) / 0.39 / 0.08 | 0.24 / 0.00 / 0.00 / 0.00 | 0.96 / 0.24 | KV 67 / 266 / 1,058 MB |
| Mamba-2 b2m | 1.00 / 0.11 (−1.5) / 0.03 / 0.05 | 0.73 / 0.00 / 0.00 / 0.00 | 1.00 / 0.73 | 13.4 MB (scoreboard) |

**Reach (mean of the two) at 2k: PAM 0.86 / 0.86 (two seeds) vs T++ 0.03, Mamba-2 0.06.**
At a 2k one-read gap (inside the trained length) PAM keeps the rule while both baselines
lose it (T++ with the pack in its window). By 8k all four are at chance. **R7 (32k): FAIL**
for every model (0.07 / 0.08 vs 0.08 / 0.05). Re-fetching the pack works at every L for
all bodies (T++ and Mamba-2 at their L = 0 skill). Two readings for the record:
- The replicated mind edge is "carries an induced rule across ~2k tokens of unrelated
  text after one read", with an O(1) state of 1.15 MB (12× smaller than Mamba-2's).
- The reach wall at ~2k is the LAW-1 question (pre-registered above).

**LAW-1 amendment A1 (2026-10-31 14:15 IST, before any LAW-1 number; Lead's decisions).**
(1) Reach at 2k (the control's own length) is reported for every arm, so the table shows
a slope, not only the endpoint. (2) Order on the 6000: lg32 first (decisive), then lg8,
then lg2; starts when s8 ends (~18:00), or at once if r3c fails its futility check. (3)
Source `b2gap` committed with tests in `e4d4fcd` (400 docs: 0 format errors, 0 held-out
values, 0 wrong answers, deterministic, resume OK, length ≤ bound).

**`lbm` (Mamba-2 + lab) cancelled before it started** (Lead, 14:15 IST): the user's comparison
rule (baselines as they are, no new baseline training) now governs; it was the label
pair's amendment-1 arm. The label result therefore stands as PAM vs T++ on the same data
(0.934 vs 0.798) with Mamba-2's existing number without label docs (b2m label_diag 0.787)
quoted beside it, not as a same-data Mamba-2 comparison.

**ntfy EUREKA sent 14:16 IST** (user's rule: replicated, ahead of both baselines at lower
memory cost, params and tokens stated): the one-read 2k reach (PAM 0.86 / 0.86 vs T++
0.03, Mamba-2 0.06; ~103M, 600M tokens, 1.15 MB state; b2p1 / b2p2_main_38e3c29, probe 53d0e52).

**LAW-1 amendment A2 (2026-10-31 14:20 IST, before any LAW-1 number):** the holdout
guard (rule 3) is measured on the frozen attr3b text (PG-19 + web, 2048-token windows;
`~/attr/data.npz` on the 6000) for every arm AND for b2p2 with the same script, because the
trainer's own validation uses windows of the arm's SEQ (8k / 32k), which are not
comparable with b2p2's 2048-window holdout. Guard: arm − b2p2 ≤ +0.02 on both texts.

### Replay integration (MLB-1 reported cell, 2026-10-31 14:45 IST): a replayed span combines with the state inside the reach

Same probe and seeds (100 docs per cell; refresh = mean of affine and shift). *Replay
alone* = the header at t0, L filler, then 2 replayed demonstration lines + the query.
*State + replay* = the 8-demo pack at t0 as well.

| model | replay alone @ 0 / 2k / 8k / 32k | state + replay @ 0 / 2k / 8k / 32k | one-read state alone @ 2k | two_hop integrate / oracle / hop-2 only (chance 0.25) |
|---|---|---|---|---|
| PAM b2p1 | 0.99 / 0.66 / 0.67 / 0.69 | 1.00 / **0.98** / 0.69 / 0.70 | 0.86 | ~0.19–0.29 everywhere |
| PAM b2p2 | 1.00 / 0.79 / 0.75 / 0.76 | 1.00 / **0.98** / 0.77 / 0.76 | 0.86 | ~0.18–0.24 everywhere |
| T++ b2t | 0.54 / 0.53 / 0.31 / 0.18 | 0.57 / 0.53 / 0.34 / 0.20 | 0.03 | ~0.19–0.28 |
| Mamba-2 b2m | 0.81 / 0.56 / 0.56 / 0.56 | 0.88 / 0.60 / 0.56 / 0.56 | 0.06 | ~0.20–0.28 |

**Reading.**
- *Inside the reach the state and a replay add up:* PAM at 2k, state + replay 0.98 vs
  replay alone 0.66 / 0.79 and state alone 0.86 (both seeds). Beyond the reach (8k,
  32k) the state adds nothing (it has faded), and the replay works alone at ~0.7.
- *A short re-fetch works at any distance for PAM* (2 lines → ~0.7, flat from 2k to
  32k). T++ with the same 2 lines falls with total length (0.53 → 0.18; positions past
  its trained length). Mamba-2 stays at 0.56.
- *two_hop* is at chance for every model even with both hops in front (oracle): the
  two-hop skill was never trained, so this cell cannot say anything about integration
  across hops yet. It needs the format in training (the "low prior, high ceiling" rule)
  before it can be read.

### r3c futility check (2026-10-31 15:16 IST): PASSED, r3c continues

Pre-registered (EXPERIMENTS "r3c"): stop if holdout at ~1.56B tokens > 3.4714. **r3c @ step
39,060: 3.4010** (WikiText 60.23) → continues. Same step for r3b (M2 + echo, LR 2e-4):
3.4133, so r3c (16 notebooks + echo, LR 1.2e-3) has overtaken it (it trailed by 0.05 at
step 15,624). Late slope over the last 3,906 steps: r3c −0.024, r3b −0.0125. Projection
only (not a claim): ~1.5–2× r3b's late gains would end r3c near 3.20–3.25, close to T++
3.2064 and still behind Mamba-2 3.1644 (scoreboard). Kept checkpoint
`r3cb_main_38e3c29/step_039062.pt` for the attr3b split. Ledger reminder: r3c has none
of the 10-31 findings (no b2 / lab / b2gap docs).

## A4 cond_mem filter `a4f` result (2026-10-31 16:20 IST): IN BETWEEN (rule 3); not in the package

`a4fb_main_d338469` (4090; n9r ctl43 recipe + COND_MEM=1 in both stages; M1 body + echo,
seed 43; 600M tokens; 103.6M dense + 33.5M table = 137.1M). Reference per the 13:01
amendment: C1-ctl `c1ctlb_main_7ae6f1f` (seed 42); ctl43 (the seed-43 outlier) reported.

| 600M tokens | params | holdout | recall 63 (≤ 2048 / > 2048) | label_diag cat / exact | PG-19 far gain 4–8k |
|---|---|---|---|---|---|
| **a4f (table)** | 103.6M + 33.5M table | **3.4488** | **0.916** (0.991 / **0.728**) | 0.650 / 0.675 | +0.044 |
| C1-ctl (ref) | 102.9M | 3.4743 | 0.825 (0.917 / 0.597) | – / 0.610 | +0.041 |
| ctl43 (reported) | 102.9M | 3.598 | 0.901 (– / 0.735) | 0.637 / 0.673 | |

attr3b on the frozen text (a4f − C1-ctl): PG-19 all −0.025, new tokens −0.039, **new @ 0–64
−0.0275**; web all −0.031, new −0.040, **new @ 0–64 −0.0615**; copy −0.020 / −0.025, seen
−0.016 / −0.025. (vs ctl43: −0.164 / −0.149, inflated by ctl43's outlier.)

**Verdict (pre-registered).** Rule 1 KEEP needs holdout ≤ ref − 0.02 (met: −0.026) AND new-token
loss at 0–64 ≤ ref − 0.03 on BOTH texts: web met (−0.062), **PG-19 missed by 0.0025 (−0.0275)**.
Not KEEP. Rule 2 FAIL needs a holdout gain < 0.01: no. → **Rule 3, in between:** not in the
package; reported; revisited at 3B only if nothing better exists. Recall guard met (+0.09).

**Reading.** A hashed n-gram table gives a real but modest gain (~0.025–0.03 nats, spread
over all token classes, largest on new tokens and early positions on web text), about a
quarter of the 3B Mamba-2 gap, for +33% parameters (sparse rows). One seed, cross-seed
reference (seed 43 vs 42; seed noise ~0.003 here). **Novelty / thesis:** Engram-style
n-gram memory is known (DeepSeek Engram), and it is parametric knowledge inside the
weights, so it serves neither thesis clause: a cost table, not a claim. The recall jump
(+0.13 past 2048) is unexplained; it may be the table making key tokens more distinct.
A4 stays in the code as a reported option; no further GPU unless a mechanism needs it.

## LAW-1 first arm `lg32` (2026-10-31 18:59 IST): training at 32k with long-gap rule docs SHRANK the one-read reach

`lg32_main_2229a5b` (6000; fine-tune from b2p2, 150M tokens, SEQ 32768 × 1, LR 3e-4, `b2gap`
3 % with G ~ U(0, 31,000), fineweb read from local shards). Skill-pack cell (100 docs per
cell, seeds 8000–8099; the first probe run OOMed at its own 5.9 GB cap on the first 128k
doc, so 0–32k were rerun; 128k pending). Reach = mean of affine and shift A_state.

| one-read reach | L = 0 | 2k | 4k | 8k | 32k | affine A_margin @ 0 / 2k |
|---|---|---|---|---|---|---|
| b2p2 (start point, 2k-trained) | 1.00 | **0.86** | – | 0.10 | 0.04 | +18.3 / +4.1 |
| **lg32** (32k-trained, long gaps) | 1.00 | **0.31** | 0.08 | 0.06 | 0.04 | +10.2 / +0.0 |

Per task: lg32 affine 1.00 / 0.42 / 0.12 / 0.11 / 0.06, shift 0.99 / 0.20 / 0.03 / 0.00 / 0.01.
Guards: attr3b vs b2p2 PG-19 +0.015, web −0.002 (bar ≤ +0.02, met); B2 `_new` at L = 0
0.953 / 0.751 / 0.995 (mean 0.900, bar 0.90, met exactly). Own 32k-window val 3.4374.

**Rule 1 for lg32 (reach_lg32(32k) ≥ 0.75): FAIL** (0.035), and worse: reach at 2k fell from
0.86 to 0.31. The full LAW-1 verdict needs lg8 and the control lg2 (queued after s8).

**Reading (hypotheses, untested).**
1. *Vanishing signal, not a capacity wall.* The mind's memory half-life is ~1–3k tokens
   (echo). With gaps ~U(0, 31k) (mean 15.5k), the gradient from a query back to the pack
   through ~15k tokens of decay is ~0, so the model gets no push to keep the rule longer.
   Meanwhile answer-weighted (×100) loss on queries it cannot solve pushes it toward the
   trained-value prior, which hurts held-out values even at short gaps (margin +18 → +10
   at L = 0). This is the classic RNN long-dependency problem, met by the recipe.
2. *Prediction:* lg8 (G ≤ 7.5k, mean 3.75k, much of it within today's reach) does
   better than lg32, and a curriculum (gaps just beyond the current reach, growing as
   reach grows) would extend reach where uniform long gaps fail. If lg8 also shrinks
   the reach, the culprit is more likely the fine-tune itself (LR 3e-4 on a converged
   model) than the gap distribution, and lg2 (the control) will show it.
3. For Library's skill register: a register that does not decay (written once at
   induction) removes the vanishing path entirely. That fits this result.

**lg32 decay half-lives (Lead's ask, measured early for lg32, 19:05 IST).** `v13_sempty/tmp/gates_diag.py`,
affine_map_new prompts at L = 4096 (12 prompts), CPU; filler half-life per head (geometric mean
retention on filler tokens), content half-life in brackets. Logs `logs/diag/v13_sempty_halflives_{b2p2,lg32}_*`.

| head | b2p2 filler (content) | lg32 filler (content) | change |
|---|---|---|---|
| echo h0 | 718 (46) | 537 (88) | −25 % |
| echo h1 | 1,564 (47) | 1,026 (61) | −34 % |
| echo h2 | 1,274 (11) | 802 (23) | −37 % |
| echo h3 | 2,376 (220) | 1,427 (462) | −40 % |
| echo h4 | 1,548 (38) | 984 (68) | −36 % |
| echo h5 | 2,539 (264) | 1,137 (359) | −55 % |
| notebook b11 h4 | 648 (149) | 401 (149) | −38 % |
| other notebook heads (≥ 30) | 38–202 | 33–179 | small |

**The decays moved, toward FASTER forgetting.** So "no gradient reached the decays" is
refuted for lg32; the gradient reached them and shortened them, while write gates barely
changed and the reset at rule tokens weakened slightly (content half-lives up). Revised
reading (still a hypothesis): at SEQ 32k, each window packs many unrelated web documents,
so the LM objective rewards forgetting faster, and that outweighs the sparse (3 %) long-
gap rule signal, whose own gradient is weak across ~15k-token gaps. Consequences: reach
extension by training would need long COHERENT documents at long context (books, long
code), a larger share of long-gap docs or a curriculum, or a structural decay floor /
protected register (Library's SR). lg8 / lg2 get the same table in their write-up.

## LAW-1 closed (user decision via Lead, 2026-10-31 21:20 IST): one-read reach ≈ training length is a STATED LIMIT at this scale

User: "That's not a failure, it's our limit at this scale; mark it done and move next."
"Holds beyond the training context" is DONE at the package level: re-fetch 1.00 at every
length to 128k; spread rules 0.905 at 4k (re-exposure); the mind's one-read reach ≈ its
training length (~2k at 103M / 600M tokens) is the stated limit (PLAN "Stated limit,
closed"). **lg8, lg2 and lg32-reset are CANCELLED before they started** (they could not
change this decision). Only lg32 ran (result above).

**Lesson recorded as done (the half-life side finding):** fine-tuning the mind at 32k on
packed web text plus 3 % long-gap rule docs SHORTENED every echo memory half-life by
25–55 % and the longest notebook head by 38 %. The training objective at long context
teaches forgetting; extending a decaying state's reach by gradient needs a different
data regime (coherent long documents, or a curriculum), or a protected register. Not
pursued now. The reset-at-separator mask (to test the packing explanation) is left
UNFINISHED on branch `reset` (`qllm_reset`): its isolation test still fails (some cross-
document path remains), and it is not on master.

## cu1: skill curriculum for the mind (pre-registration, 2026-10-31 22:15 IST)

**Thesis clauses served:** *intelligence separate from knowledge* (the mind learns to use
re-read spans and to abstain when the library holds nothing), and *holds beyond the
training context* at the package level (a stated rule re-fetched at 32k, R6b).

**Closest prior work (novelty rule):** instruction tuning (FLAN, T0: stated instructions in
training); retrieval-augmented fine-tuning (RA-DIT; Self-RAG) and "say I don't know"
training (R-Tuning). **What differs (one line):** the recipe is known; what is measured is
whether words read once, and spans re-read from a library, write the SAME rule / lookup
state that demonstrations write in an O(1) recurrent mind, with held-out values and
vocabularies disjoint from the bench. That is the skill-plus-library contract the package
needs.

**Why (baselines measured now, b2p2):** stated rule 0-shot 0.000 even re-read in front of
the query (Lead's instr cell); bench-vocabulary probe (`scripts/run_cur_probe.py`, 200
docs, CPU, `logs/diag/v13_sempty_curprobe_b2p2_baseline_*`): re-read lookup present 0.990,
**abstain on an absent key 0.000** (it always names a value), **two-hop 0.10** (chance 0.05).

**Arm `cu1`** (6000, code at launch on master, which includes the sources `srd` / `fsq` / `hop`
of `48674d6`): b2p2's recipe (M2 + echo + shared reads, stage A `c1m2a_pre_7ae6f1f`, stage B
480M, LR 1.2e-3, seed 42 so it pairs with b2p1 / lbp), stage-B mix dclm / fineweb /
smoltalk2_mid / recall / b2 / lab / srd / fsq / hop = 0.40 / 0.37 / 0.10 / 0.03 / 0.03 / 0.02 /
0.02 / 0.02 / 0.01. **Starts when r3c ENDS** (gate on `r3cb_main_38e3c29/.done`, plus ≥ peak + 3 GB
free: b2p3's measured ~40 GB → ≥ 43 GB), so the 55 % rule is never broken. Saves every 200 steps.
**Paired arm `cu1n9s`** (same data and seed, N9s salient-phase echo keys): runs after cu1
ONLY if Library supplies an LM-trainer switch for N9s with a self-test before cu1 ends; N9s
exists only in v14's micro-bench today. Otherwise dropped, and recorded so.

**Rules (fixed now; 100 docs per bench cell, 200 per run_cur_probe cell; probe seeds fresh):**
- **R6a (mind):** `v16/bench.py --task instr --shots 0`, held-out shift values: A_state ≥ 0.75 at
  L = 0 (the rule right before the query) AND at L = 2048 (one read, inside the reach).
- **R6b (package):** the same cell at L = 32768, O_reread (rule re-fetched in front of the query)
  ≥ 0.75. Mind-only A_state at 32k is reported with the stated-limit caption, no bar.
- **R-fsq (abstain):** run_cur_probe absent-key abstain ≥ 0.75 AND present accuracy ≥ 0.94
  (b2p2 − 0.05) AND false abstain ≤ 0.10.
- **R-hop:** run_cur_probe two-hop ≥ 0.50 (b2p2 0.10, chance 0.05).
- **Bench re-read (Lead's cell):** `v16/bench.py --task kv` (8 needles, 32k) O_reread for cu1 vs b2p2
  on the same docs (crc32 seeding, paired): reported; cu1 must not be worse than b2p2 by > 0.02.
- **Guards:** holdout ≤ 3.500 (b2p2 + 0.02); B2 `_new` mean at L = 0 ≥ 0.894 (b2p2 − 0.05);
  label_diag mean ≥ 0.884 (lbp − 0.05); skill-pack reach at 2k ≥ 0.81 (b2p2 − 0.05).
- **Win** = R6a and R-fsq, with the guards met; R6b, R-hop and the bench re-read are claimed
  separately. Any FAIL is recorded with its numbers; failed sources are removed from the code
  per the AGENTS rule (index row with the code-at hash).

**cu1 amendment A1 (2026-10-31 22:25 IST, before any cu1 number): the N9s paired arm is ON.**
Library ported N9s to the LM trainer (`c91f4c6`: `ECHO_PHASE_LAGS=3 ECHO_PHASE_SALIENT=1`, v14
self-test ALL PASS, grad-ckpt parity, builds and reloads on M2) before cu1 started, which was
Lead's condition. **Arm `cu1n9s`:** its OWN stage A (the init_from warm start refuses N9s's new
parameters; same precedent as n9c), with c1m2a's recipe (dclm / fineweb / smoltalk2_mid /
aptitude 0.25 / 0.20 / 0.15 / 0.40, 120M tokens, blend warmup 30M, LR 1.2e-3, seed 42, M2 + echo)
plus N9s, then cu1's stage B exactly (same mix, seed 42). Runs after cu1 on the same card (a
second gate on cu1's end), code at master `2e75098` (cu1's code plus the additive switch).
**Caveats that travel with it:** N9s needs N9a's 3-token lags, and N9a alone FAILED its
replication (n9r) and hurt lookup at 600M (n9c), so the arm measures N9a + N9s together. Its
stage A differs from cu1's (trained with N9s on), a second confound.
**Rules for the arm (fixed now):** (i) *N9s helps two-hop at 600M* if cu1n9s two-hop (run_cur_probe)
≥ cu1's + 0.15 AND ≥ 0.50; (ii) guards against cu1: holdout ≤ cu1 + 0.02, label_diag ≥ cu1 − 0.05
(N9a hurt lookup before), R6a not worse than cu1 by > 0.10; (iii) every cu1 cell reported for
both arms. A pass is a 600M paired result for Library's unit, not a package claim.

**cu1 amendment A2 (2026-10-31 22:35 IST, before any cu1 number; Library's attribution point).**
If cu1n9s passes its two-hop rule, one more arm runs before ANY claim about salience: `cu1n9a`
= the same new stage A recipe and stage B with `ECHO_PHASE_LAGS=3` only (no salience), seed 42.
Then N9s is credited only with what cu1n9s adds over cu1n9a (same +0.15 margin on two-hop);
cu1n9a vs cu1 shows what the lags plus a fresh stage A do. If cu1n9s fails its rule, cu1n9a
does not run. GPU order for cu1n9a goes through Lead.

**cu1 amendment A3 (2026-11-01 01:20 IST, before any cu1 number; from Library's near-miss finding): a
REPORTED `verify` cell, plus free generation in every run_cur_probe cell (no new bar).** Library's
strict run showed that the package re-reads ONE row plus the question alone, and that the wrong
rows are mostly UNRELATED keys in the same template (17 / 18 synthetic, 13 / 14 natural share no
syllable or word with the asked key) or a plain prose sentence. `scripts/run_cur_probe.py` now adds,
per doc on its own random stream (fsq / hop rows stay identical to the pre-registered baseline:
400 / 400 checked), three short-format rows: right row → value, wrong-key row → " unknown", prose
row → " unknown". **b2p2 baseline** (200 docs, CPU,
`logs/diag/v13_sempty_curprobe_b2p2_verify_2b91507.jsonl`):

| verify row | says the row's value: 21-way / free | correct (value or " unknown"): 21-way / free |
|---|---|---|
| right row | 1.000 / 0.960 | 1.000 / 0.960 |
| wrong-key row | 0.700 / **0.140** | 0.000 / 0.000 |
| prose row | 0.065 / 0.000 | 0.000 / 0.000 |

b2p2 never says " unknown" (0.000 everywhere), but under free generation it says a wrong row's value
only 14 % of the time against 96 % for the right row; its greedy token is mostly " a". Reported for
cu1 / cu1n9s beside R-fsq; it becomes a bar only by a later dated amendment.

## s8 VERDICT (2026-10-31 22:05 IST): distance confirmation MET, replicated on fresh seeds and a fresh probe sample

Pre-registered in "B2 Mamba-2 arm b2m", amendment 2. Probe: `--suite b2 --tasks
shift_wide_new,affine_map_new --shots 8 --trials 400 --seed 7000 --spread 0,2048,4096` (probe seeds
7000–7399, never used before); D = mean of the four cells at 2048 and 4096. All runs 600M tokens on
code `38e3c29`.

| run | params | seed | holdout | shift @ 0 / 2k / 4k | affine @ 0 / 2k / 4k | **D** |
|---|---|---|---|---|---|---|
| PAM b2p1 | 102.9M | 42 | 3.4777 | 1.00 / 0.595 / 0.278 | 1.00 / 0.935 / 0.772 | **0.645** |
| PAM b2p2 | 102.9M | 43 | 3.4800 | 1.00 / 0.583 / 0.300 | 1.00 / 0.970 / 0.912 | **0.691** |
| **PAM b2p3** (new) | 102.9M | 44 | 3.4775 | 1.00 / 0.775 / 0.438 | 1.00 / 0.945 / 0.807 | **0.741** |
| T++ b2t | 101.9M | 42 | 3.5572 | 0.282 / 0.00 / 0.003 | 0.97 / 0.033 / 0.502 | 0.134 |
| **T++ b2t2** (new) | 101.9M | 43 | 3.5547 | 0.585 / 0.00 / 0.00 | 0.988 / 0.072 / 0.463 | 0.134 |
| Mamba-2 b2m | 102.6M | 42 | 3.4754 | 0.762 / 0.00 / 0.00 | 1.00 / 0.302 / 0.160 | 0.116 |
| **Mamba-2 b2m2** (new) | 102.6M | 43 | 3.4768 | 0.033 / 0.00 / 0.00 | 1.00 / 0.275 / 0.115 | 0.098 |

**Primary: D(b2p3) − max(D(b2t2), D(b2m2)) = +0.607** (bar +0.10): MET. **All pairs: min PAM 0.645 −
max baseline 0.134 = +0.511** (bar +0.10): MET. → **Bar 3 is met in its distance form**, as
pre-registered: in-context inference of held-out rule values from examples spread across 2–4k
tokens of unrelated text, holding past the trained length (2048) in total context, where T++ and
Mamba-2 do not. Three PAM seeds, two seeds of each baseline, a never-used probe sample. Holdout
is at parity with Mamba-2 (3.4775 vs 3.4768) and 0.08 better than T++.

**Wording that travels with the claim (the skill-pack correction, same day):** this is the SPREAD
cell. The examples are spread by filler, so the gaps between them are ~250–500 tokens and the last
example sits L/8 before the query. One-read reach (all examples together, then one long gap) is
~2k at this scale, a stated limit. The claim is "rules inferred from spread-out examples survive
unrelated text, past the trained length in total", not "one read carries a rule 4k".
Caveat (unchanged): trained rule FORMATS with held-out VALUES, not unseen rule families.

## lg8r: does an 8k training window lengthen the one-read reach once documents stop teaching forgetting? (pre-registration, 2026-11-01 00:13 IST)

**Request (user via Lead, 2026-10-31 23:55 IST):** one cheap check of the window law ("the
maths cannot be sure"), with little GPU: the 4090 after Lead's package table, not the 6000.

**Hypothesis.** lg32 shrank the reach because 32k windows of PACKED unrelated documents reward
forgetting (every echo half-life −25..55 %, long-head geomean 1361 → 835). With the state reset
at every document boundary, forgetting an old document costs nothing, so training at 8k
should no longer shorten the decays. If the one-read reach follows the training window, the
4k reach (b2p2: 0.355) rises toward b2p2's 2k level.

**Closest prior work (novelty rule):** long-context continued pretraining (Fu et al. 2024,
"Data engineering for scaling language models to 128K context"). Document-boundary state
resets in packed recurrent training are standard: Mamba-2's `seq_idx` / varlen kernels, and
intra-document attention masks for transformers (Llama 3; Zhao et al. 2024 on sequence
composition in pretraining). **What differs:** only the measured quantity (one-read reach of
an induced rule, plus per-head memory half-lives, against the training window, with the
packing confound removed). The reset and the answer-in-window weighting are plumbing; there
is no mechanism claim. (Lead's line said the reset is ours. It is not; it is known.)

**Code:** `f2587b5` reset (`RESET_AT_EOS=1`; parity "reset == fresh sequence after the
separator" PASS on CPU and GPU, `v13_sempty/tmp/reset_parity.py`); `824a988` answer-in-window
(`ANSWER_IN_WINDOW=1`; `v13_sempty/tmp/answer_in_window_test.py` PASS). **Amendment to Lead's
design (raised with Lead before launch):** the packer cuts documents at window edges, so at
SEQ 8192 **44 % of b2gap answer tokens sit in a window without their rule's demonstrations**.
Their ×100 weight can only teach guessing from the trained-value prior, and it is about as
large as the useful long-gap signal (lg32 had the same flaw). With the flag, those tokens get
weight 1.0.

**Run `lg8r_main_<hash>`** (4090, own worktree). Fine-tune from `b2p2_main_38e3c29/best_model.pt`:
- SEQ 8192 × BATCH 2 (16,384 tokens/step), 3,052 steps = 50.0M tokens (a third of a LAW-1 arm);
- LR 3e-4, warmup 100, cosine; seed 45; M2 (ECHO=1, SHARE_READS=1, PAM_LAYERS 3,7,11,15);
- mix as LAW-1: dclm 0.43, fineweb 0.40, smoltalk2_mid 0.10, recall 0.04, b2gap 0.03, `B2GAP_MAX` 7500;
- `RESET_AT_EOS=1 ANSWER_IN_WINDOW=1`; save every 200 steps; val every 1,526 steps.

**Readouts (fixed now):**
1. *Reach:* skill-pack cell (`scripts/run_skillpack_probe.py`, seeds 8000–8099, 100 docs),
   mean A_state of affine_map_new + shift_wide_new at L = 0 / 2k / 4k / 8k (+ 32k reported).
   b2p2: 1.00 / 0.86 / **0.355** / 0.10 / 0.04 (4k measured tonight:
   `logs/diag/v13_sempty_skillpack_b2p2_4k_skillpack_f2587b5_20261031_1836.jsonl`, affine 0.42,
   shift 0.29). The probe filler has no separators (480,534 tokens, 0 EOS), so the reset does
   not touch it.
2. *Half-lives:* `v13_sempty/tmp/gates_diag.py <ckpt> 4096 12 affine_map_new` (same prompts as
   b2p2's table). Summary = geometric mean of the filler half-lives of the 7 long-memory heads
   (echo h0–h5, notebook b11 h4), `v13_sempty/tmp/hl_summary.py`. b2p2 **1361**, lg32 835.

**Bars:**
- **"Reach follows the window" = reach(4k) ≥ 0.75 AND long-head geomean ≥ 1293 (0.95 × b2p2).**
- Otherwise: a ceiling of this body at this scale; stop spending GPU on reach-by-training (the
  route stays re-fetch / Library's register). Mixed cases are recorded as they are: reach up
  with half-lives shorter = "reach grew by another path"; half-lives held with reach < 0.75 =
  "the reset fixed the forgetting, the reach did not follow". Neither gets a follow-up without Lead.
- *Guards (a failed guard voids a PASS):* attr3b holdout vs b2p2 ≤ +0.02 nats in both strata
  (PG-19, web; documents scored one at a time, so the reset does not act); B2 `_new` at L = 0
  (learning probe, b2 suite, 400 trials) mean ≥ 0.894 (b2p2 0.944 − 0.05).
- Reported: per-task reach and margins, the per-head table, reach at 8k and 32k, tok/s, peak
  VRAM, own 8k val (not comparable to b2p2's 2k val).
- One seed and no control arm (cheap check). **If PASS, before any claim:** the control `lg2r`
  (same recipe at 2k windows, 50M tokens) separates "window" from "more fine-tuning with
  long-gap docs", then a second seed; GPU via Lead.

**Ledger:** lg8r = b2p2 recipe + b2gap + reset + answer-in-window. It has none of cu1's
curriculum (srd / fsq / hop) and no N9s.

**lg8r amendment approved (Lead, 2026-11-01 00:18 IST), and a correction to LAW-1.** Lead
approved ANSWER_IN_WINDOW=1 and the bars as written. **lg32 carried both confounds lg8r
removes:** packed windows that reward forgetting across unrelated documents, and ~45 % of its
b2gap answers trained at ×100 in windows without their rule's demonstrations (the same packer
cut, at 32k with G ≤ 31k). So lg32's "reach is a ceiling of this body" reading is **WITHDRAWN
pending lg8r**. The user's package-level decision (one-read reach ≈ training length is the
stated limit today; re-fetch carries the rule to 128k) stands until lg8r says otherwise. lg8r
launched 00:16 IST (`lg8r_main_f2b7d8c`, 4090).

**Orphaned answers per source (2026-11-01 00:20 IST; CPU, `v13_sempty/tmp/orphan_rate.py`).**
300 real docs per source, interleaved 1 : 1 with web-like docs. Share of answer-weighted tokens
in a window where their document did not start:

| source | doc tokens median / p90 | SEQ 2048 | SEQ 8192 |
|---|---|---|---|
| b2 (b2p2's 3 %) | 143 / 177 | 3.3 % | 1.9 % |
| lab | 159 / 191 | 4.1 % | 0.6 % |
| recall (every recipe, 4–5 %) | 101 / 760 | **15.9 %** | 4.8 % |
| srd (cu1) | 63 / 1,400 | **22.0 %** | 5.0 % |
| fsq (cu1) | 187 / 221 | 6.1 % | 0.6 % |
| hop (cu1) | 196 / 203 | 7.5 % | 2.3 % |
| b2gap, G ≤ 7.5k (lg8r without the flag) | ~3.8k mean | – | **44 %** |

Reading: b2p2's own rule docs were barely cut, so its 2k reach is not understated by this.
Recall at 2k loses ~1 answer in 6 in every arm alike (baselines included). cu1's `srd` (stated rule
+ gap) is the most affected at 2k; cu1 runs without the flag (started before it existed), so a
narrow R6a miss gets this caveat, and cu2 or any re-run turns the flag on.

## r3c VERDICT (2026-11-01 00:48 IST): does NOT beat the 3B baselines on LM quality (+0.013 vs T++, +0.055 vs Mamba-2); leads both clearly on recall, in and beyond the training length

`r3cb_main_38e3c29` (6000; 16 notebooks + echo, LR 1.2e-3, two stages, 3B tokens, 103.2M params).
Pre-registered rule (EXPERIMENTS "r3c"): beats = holdout lower by > 0.01 AND WikiText lower;
within ±0.01 = competitive. All three: ~102–103M params, 3B tokens, the same mix.

| 3B, same data | holdout | WikiText PPL | late slope 2→3B | recall all / ≤2k / >2k | PG-19 loss 1–2k / 2–4k / 4–8k | rules / B2 / label_diag | state |
|---|---|---|---|---|---|---|---|
| T++ (`e072a8f`) | 3.2064 | 46.61 | −0.081 | 0.539 / 0.693 / **0.153** | 3.553 / 4.278 / **5.535** | 0.228 / 0.088 / 0.703 | KV grows |
| Mamba-2 (`b36e3a5`) | **3.1644** | 46.91 | −0.079 | 0.613 / 0.697 / 0.403 | 3.510 / 3.502 / 3.537 | 0.202 / 0.094 / 0.580 | 13.4 MB |
| **r3c** | 3.2191 | 49.95 | −0.080 | **0.792 / 0.844 / 0.661** | 3.552 / 3.540 / 3.572 | 0.217 / 0.059 / 0.610 | ~3.9 MB |
| r3b (M2, LR 2e-4) | 3.3094 | 58.0 | | 0.620 / 0.676 / 0.482 | | | |
| old PAM 3B (`1c913ef`) | 3.2728 | | | 0.667 | | | |

(Late slope = holdout at 3B minus holdout at 2B, interpolated between the 1.95B / 2.10B vals.
Recall = mean of the 63 t40 cells, the scoreboard definition (r3b recomputes to its 0.620).)

**Verdict by the rule: NOT beat, and not "competitive"** (vs T++ +0.0127, just outside ±0.01; vs
Mamba-2 +0.0547; WikiText worse than both). It closes most of the old gap: T++ − PAM went from
−0.066 (old PAM 3B) and −0.103 (r3b) to −0.013. The late slope equals the baselines'
(−0.080 vs −0.081 / −0.079): r3c keeps pace in the last third but does not close the gap.

**Where r3c wins clearly (body alone, no library):** recall 0.792 vs 0.539 / 0.613. Beyond the
2,048 training length it is 0.661 vs T++ 0.153 (collapse) and Mamba-2 0.403. PG-19 loss holds
past the training length (4–8k 3.572, +0.02 over 1–2k) where T++ collapses (5.535); Mamba-2 holds
as well and is 0.035 better in absolute terms. Rule probes tie (no rule docs in this recipe: B2
0.059 ≈ chance for all three). label_diag is behind T++ (0.610 vs 0.703; the lab docs that fixed
this at 600M are not in r3c).

**Shown generations (A7; CPU; greedy unless stated; same prompts for b2p2):**
- r3c `r3cb_main_38e3c29/best_model.pt`, rule: "Map rule.\n3 -> 10\n5 -> 14\n8 -> 20\n11 ->" → " 21",
  then counts (12 -> 22, 13 -> 23, …). **Wrong** (no rule docs). b2p2: " 26", then a correct new
  line "28 -> 60".
- r3c, recall in plain language ("…the backup key is with Priya in room 18. [2 sentences] Question:
  who has the backup key? Answer:") → "The backup key is the one that is used by the backup".
  **Wrong.** b2p2: "The store has a backup key." **Wrong too.** The recall probe's 0.79 is a
  trained key-value FORMAT; a plain-language question in a new format is not answered. That is
  the A7 point: the 20-way / trained-format number is not yet expression.
- r3c, free text (T 0.8, top-k 50, top-p 0.9, rep 1.2, seed 1), "The first printing presses in
  Europe" → "were made by the Greek and Roman governments. In this case, they are called
  "Bardae". In 1482, the Romans invaded Poland. …". Fluent, invented facts (103M scale).

**Ledger:** r3c has none of the 10-31 findings (no b2 / lab / curriculum docs, no M2 shared reads).
**Next:** no further 3B run is queued from this. The 3B "best package" run (M2 or 16nb + echo +
b2 + lab + curriculum, with ANSWER_IN_WINDOW) is the place they meet, after cu1 says which
curriculum parts help. GPU via Lead.

## r3p: the 3B best-package mind (pre-registration, 2026-11-01 00:51 IST; GPU slot approved by Lead after cu1 + cu1n9s)

**What it is.** One 3B run that brings together everything that worked at 600M on the only body
proven at 3B, so the package's mind is measured at the baselines' scale, and LM quality
("expression", A7) is stated next to it.

**Closest prior work (novelty rule):** delta-rule linear RNNs with gated decay (DeltaNet /
Gated DeltaNet, Yang et al. 2024), Mamba-2 (SSD), RWKV-7, Titans (test-time memory); synthetic
skill data mixed into pretraining (procedural / "physics of LMs" style mixtures). **What
differs (one line):** the body (PAM: delta notebooks with content-modulated rotary retention,
an echo succession memory into the head, ~3.9 MB state) trained as the MIND of a mind + library
package and judged at equal data on recall and rules in and beyond the training length, with LM
quality stated. The recipe (pretraining with synthetic mixtures) is known; no recipe claim.

**Run `r3p`** (6000; ~103.2M params; 3.0B tokens = the baselines' budget):
- Body and schedule exactly as r3c: `baseline_real_pm` (16 notebooks, chrono + out_gate + delta) +
  ECHO=1, LR 1.2e-3, seed 42, SEQ 2048.
- **Stage A reused:** `r3ca_pre_38e3c29/best_model.pt` (120M tokens, identical recipe), so stage B
  starts at once; `--init_from` it.
- **Stage B 2.88B tokens**, `ANSWER_IN_WINDOW=1` (`824a988`). The mix is fixed now by cu1's verdict:
  - cu1 Win (R6a AND R-fsq, guards met): cu1's mix with each source kept only if its own rule passed
    (srd ← R6a, fsq ← R-fsq, hop ← R-hop); a failed source's share goes back to dclm / fineweb in
    their 0.40 : 0.37 ratio. cu1's full mix: dclm / fineweb / smoltalk2_mid / recall / b2 / lab /
    srd / fsq / hop = 0.40 / 0.37 / 0.10 / 0.03 / 0.03 / 0.02 / 0.02 / 0.02 / 0.01.
  - cu1 no Win: dclm / fineweb / smoltalk2_mid / recall / b2 / lab = 0.41 / 0.38 / 0.10 / 0.04 / 0.03 /
    0.04 (b2p2's and lbp's ingredients together, never trained jointly at 600M: stated).
  - cu1n9s: N9a + N9s go in only if cu1n9s passes its rule AND cu1n9a (A2) credits N9s; else off.
- Launcher gates: cu1n9s ended; free VRAM ≥ peak + 3 GB (r3c's measured peak); disk ≥ 8 GB free
  (the 6000 has 17 GB at writing). Saves every 1,000 steps; keeps step 39,062 (1.5B) for attribution.

**Readouts (fixed now):**
1. **Expression (R3, the public bar; A7):** holdout and WikiText vs T++ 3.2064 / 46.61 and Mamba-2
   3.1644 / 46.91. **Beats** = holdout lower by > 0.01 AND WikiText lower; **competitive** = within
   ±0.01. **Prediction, written now: r3p MISSES both, narrowly.** r3c was +0.013 behind T++, and
   ~8–13 % of r3p's stage B goes to synthetic docs that the baselines spent on web text. Loss by
   token class (attr3b) vs r3c, T++ and Mamba-2, on the 3B attribution windows.
2. **Recall** (t40, scoreboard definition) all / ≤ 2k / > 2k vs r3c 0.792 / 0.844 / 0.661 and the
   baselines. Guard: not worse than r3c by > 0.03 overall.
3. **Rules:** B2 `_new` and the spread (LCL) cell vs the 3B baselines as they are (no b2 docs there:
   the asymmetry is stated in the caption). Bar, the 600M s8 form: spread-cell distance D ≥
   max(T++, Mamba-2) + 0.10. label_diag reported (T++ 0.703, no lab docs there).
4. **Skill-pack** reach at 0 / 2k / 4k / 8k / 32k, 20-way and free; and the cu1 cells (R6a, R-fsq, R-hop)
   for every curriculum source that went in.
5. **Free generation (A7) is the primary column** of 2–4. `run_learning_probe` gets `*_free`
   before r3p ends (to do; the skill-pack probe has it, `a073d0d`).
6. **Shown generations:** the same three prompts as r3c / b2p2 (`v13_sempty/tmp/show3.sh`): rule
   (greedy), plain-language recall (greedy), free text (T 0.8, seed 1).
7. **Futility at 1.5B tokens (step 39,062):** stop if holdout > r3c's 3.4010 + 0.03 (= 3.431). The
   synthetic share is expected to cost ~0.01; three times that means the mix is hurting.
8. Package R1 (exact-key fetch at 32k / 128k) needs Library's fetch re-fit on r3p's states:
   Library's lane, GPU via Lead.

**Ledger:** r3p = r3c's body + b2 + lab + (cu1 parts by rule) + ANSWER_IN_WINDOW. Not in it: M2
shared reads (untested at 3B at LR 1.2e-3; r3b's M2 failed at LR 2e-4), the doc-boundary reset
(lg8r is testing it at 8k; r3p trains at 2k), fetch training.

## lg8r VERDICT (2026-11-01 01:18 IST): the reach does NOT follow the window. The reset stopped the forgetting (half-lives held) but the reach barely moved, and the short fine-tune cost LM quality

`lg8r_main_f2b7d8c` (4090; b2p2 + 50.0M tokens at SEQ 8192 × 2, LR 3e-4, `RESET_AT_EOS=1`,
`ANSWER_IN_WINDOW=1`, b2gap 3 % with G ≤ 7.5k; 102.9M params; 55.9k tok/s, 16.5 GB peak).
Pre-registered bars (EXPERIMENTS "lg8r"): reach(4k) ≥ 0.75 AND long-head half-life geomean ≥ 1293.

| one-read reach (100 docs × 2 tasks) | L = 0 | 2k | 4k | 8k | 32k |
|---|---|---|---|---|---|
| b2p2, 20-way | 1.00 | 0.86 | 0.355 | 0.10 | 0.04 |
| **lg8r, 20-way** | 1.00 | **0.905** | **0.465** | 0.105 | 0.055 |
| b2p2, free (A7) | 1.00 | 0.615 | 0.170 | | |
| lg8r, free (A7) | 1.00 | 0.680 | 0.265 | | |

Re-read (pack before the query): 1.00 at every L, both models, 20-way and free. Per task, lg8r
20-way at 4k: affine 0.52, shift 0.41.

| | b2p2 | lg8r | bar / guard |
|---|---|---|---|
| reach(4k), 20-way | 0.355 | 0.465 | ≥ 0.75: **FAIL** |
| long-head half-life geomean | 1361 | 1304 (0.958×) | ≥ 1293: met, narrowly |
| per head (echo h0-h5, b11 h4) | 718 / 1564 / 1274 / 2376 / 1548 / 2539 / 648 | 632 / **3052** / 1614 / 1619 / 1471 / 1847 / 468 | reported |
| B2 `_new` at L = 0 (t400) | 0.944 | 0.933 (0.987 / 0.812 / 0.999) | ≥ 0.894: met |
| attr3b holdout vs b2p2 (PG-19 / web) | – | **+0.111 / +0.076** | ≤ +0.02: **FAIL** |
| PG-19 position loss, every bucket 0–8k | 3.70–4.21 | +0.10…+0.13 worse everywhere | reported |
| WikiText val | 4.2292 | 4.2410 (+0.012) | reported |
| recall t40 all / ≤ 2k / > 2k | 0.707 / 0.801 / 0.474 | 0.747 / 0.858 / 0.469 | reported |

(lg32, for the record: half-life geomean 835, reach(2k) 0.31, attr3b +0.015 / −0.002 after 150M tokens.)

**Verdict.** Bar 1 FAILS (0.465 vs 0.75) and the LM guard FAILS. This is the pre-registered mixed
case "the reset fixed the forgetting, the reach did not follow". Training at 8k with the
packing confound and the orphaned answers removed kept the decays where they were (lg32 shortened
them by 39 %; lg8r 4 % on the summary, with one echo head doubling and two shortening ~30 %).
The reach rose a little (+0.045 at 2k, +0.11 at 4k; free +0.065 / +0.095), far from the bar. The
window alone does not set the reach. The half-lives do, and an 8k window of mostly < 8k documents
gives no pressure to lengthen them. **Per the pre-registration: reach-by-training is a ceiling of this
body at this scale. Stop spending GPU on it.** The route stays re-fetch (1.00 at every L), the
~2k refresh (Library's N11-R) and the register.

**The LM damage, checked:** the reset is NOT a masking bug. Batch-of-two consistency
(`v13_sempty/tmp/reset_batch.py`, separators at different positions per row): each row equals its
batch-1 run within 6.3e-5 (scale 17.8). The attr3b windows and the PG-19 docs contain no
separators, so the mask is inert at eval. The damage is in the weights: most likely the short warm
restart (LR re-warmed to 3e-4 on an annealed model, 16k-token batches, 50M tokens). lg32 had 3×
the tokens at 2× the batch and recovered to +0.015. Not separated by a run, and not worth one:
the reach answer does not depend on it, since the reach stayed far below the bar.

**What it leaves for the curriculum.** Recall inside the window rose (+0.057, ≤ 2k), plausibly the
answer-in-window weighting on recall docs (4.8 % orphaned at 8k vs 15.9 % at 2k). Not isolated.
Free reach is ~0.2 below 20-way at 2k-4k in both models: a fading skill is ranked before it is said
(the step-back's contrast-decoding test is the cheap follow-up, eval only; GPU via Lead).

## cd1: state-contrast decoding (pre-registration, 2026-11-01 01:22 IST; eval only, CPU; Lead: yes, 3 lines)

- **Cells / seeds:** b2p2, skill-pack cell (pack read once, L filler, query), L = 0 / 2k / 4k,
  affine_map_new + shift_wide_new. Logits `l = l_pack + a·(l_pack − l_nopack)`, where
  `l_nopack` is the same filler + query without the pack; a ∈ {0, 0.5, 1, 2, 4}. a is chosen on a
  tune split (seeds 8100–8149) and scored on seeds 8000–8099 (100 docs per task).
- **Bar:** free reach (argmax over the whole vocab == answer) with the chosen a ≥ the 20-way
  reach − 0.05 at both 2k (0.86) and 4k (0.355), i.e. ≥ 0.81 and ≥ 0.305 (a = 0 today: 0.615 / 0.170),
  AND free at L = 0 ≥ 0.98. Then "a fading skill is outvoted, not lost" holds, and the package
  gets a 2× compute, still-O(1) decode option.
- **Prior work:** context-aware decoding (Shi et al. 2023), contrastive decoding (Li et al. 2022).
  The difference is only the setting: a decaying O(1) recurrent state's one-read reach. No method claim.

### cd1 result (01:51 IST, 2026-11-01): FAIL as pre-registered (2k free 0.70 < 0.81), but at 4k the skill is outvoted, not lost: contrast lifts free reach 0.17 → 0.59 and 20-way 0.355 → 0.815

b2p2, CPU, `logs/diag/v13_sempty_cd1_b2p2_{tune,test}_cb8053b.jsonl` (`run_skillpack_probe --variants contrast`,
`cb8053b`; a = 0 equals the plain cell, 12 / 12 checked). Logits l = l_pack + a·(l_pack − l_nopack).

| test, seeds 8000–8099 (200 rows per L) | a = 0 | 0.5 | 1 | 2 | **4** |
|---|---|---|---|---|---|
| free, L = 0 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| free, 2k | 0.615 | 0.675 | 0.700 | 0.725 | 0.700 |
| free, 4k | 0.170 | 0.265 | 0.350 | 0.465 | **0.590** |
| 20-way, 2k | 0.860 | 0.900 | 0.920 | 0.945 | 0.940 |
| 20-way, 4k | 0.355 | 0.475 | 0.580 | 0.700 | **0.815** |

a was chosen on the tune split (seeds 8100–8149: best free 2k + 4k with L = 0 ≥ 0.98): **a = 4** (the edge
of the grid). Bars: free at L = 0 ≥ 0.98 met (1.00); 4k ≥ 0.305 met (0.590); **2k ≥ 0.81 missed (0.700)**,
so the pre-registered conjunction FAILS.

**Reading.** At 4k, after one read, the skill is still in the state but outvoted by the LM prior. A
decode-time contrast with the same stream minus the pack recovers it: 20-way reach 0.815, above the
0.75 that lg8r's 50M tokens of training failed to reach (0.465). Free generation triples. At 2k free
plateaus near 0.70 while 20-way is 0.94: there the misses are not a weak skill but another token
winning the free argmax. Not diagnosed yet. Cost: 2 forward streams and 2 O(1) states (2 × 1.15 MB).
In a package, the no-pack stream is the same session without the skill read. Prior work: CAD (Shi et
al. 2023), contrastive decoding (Li et al. 2022). No method claim; the measured quantity is a
recurrent mind's one-read reach.

**Not shown (needs its own pre-registration):** a fresh seed with a fixed (a = 4, plus a = 8 past the
grid edge); 8k / 16k; other skills (two-hop, recall); the effect of the contrast on ordinary
generation (it must apply only where a read happened).

## cd2: fresh-seed confirmation of state-contrast decoding (pre-registration, 2026-11-01 01:52 IST; eval only, CPU)

- **Cells / seeds:** b2p2, skill-pack cell, affine_map_new + shift_wide_new, **seeds 8200–8299 (never used)**,
  L = 0 / 2k / 4k / 8k / 16k, a ∈ {0, 4, 8} FIXED (no tuning).
- **Bar (a = 4):** 20-way reach(4k) ≥ 0.75 AND free(4k) ≥ 0.45 AND free(L = 0) ≥ 0.98. Then cd1's 4k effect
  is confirmed on fresh docs. Reported, no bar: a = 8; 8k / 16k; at 2k, the token that wins the free
  argmax on misses (the plateau diagnosis).
- Prior work as cd1 (CAD; contrastive decoding). No method claim.

**r3p amendment A1 (02:22 IST, 2026-11-01, before r3p starts; Lead: the LM guard is a HARD gate, per the
user's "expression counts").** It replaces the mix clause (cu1 had no "Win": its holdout guard failed,
+0.045 over b2p2). The synthetic share is now set by a written budget.
- **Evidence:** cu1 at 600M added ~6 points of synthetic share over b2p2 (13 % vs 7 %) and lost 0.045
  nats of holdout. About a third to a half is fewer web tokens (7 % fewer; late slopes give +0.015 to
  +0.022). The rest (~0.025) is capacity spent on the new formats at 103M.
- **Rule:** match or exceed cu1's ABSOLUTE tokens per skill at the smallest share. At 2.88B stage-B
  tokens, 0.5 % = 14.4M tokens (cu1 had 9.6M per curriculum source).
- **Stage-B mix (fixed now; srd OUT after cu1, its 0.005 to dclm = 0.432):** dclm 0.427, fineweb 0.398 (r3c's 0.44 : 0.41 ratio), smoltalk2_mid 0.10,
  recall 0.04, b2 0.01, lab 0.01, fsq 0.005, hop 0.005, srd 0.005. srd goes in only if cu1's R6a passes
  in the rerun; else its 0.005 goes to dclm. Synthetic share 7.5 % vs r3c's 5 %.
- **Absolute stage-B tokens:** recall 115M (r3c 144M); b2 and lab 28.8M each (cu1 14.4M / 9.6M); fsq,
  hop and srd 14.4M each (cu1 9.6M / 4.8M / 9.6M).
- **HARD gate (expression):** final holdout ≤ r3c + 0.02 = **3.2391**. If missed, r3p does not replace
  r3c as the expression body; its skills are reported, and the package's LM line stays r3c's.
  (Lead: this gate guards only the curriculum's cost against r3c. MLB-1 R3 as written, within +0.02 of
  Mamba-2 at 3B, stays unmet by r3c (+0.055), so the package table's R3 line stays the 600M b2 pair.)
- **Written prediction:** fewer web tokens (−2.9 %) +0.006, formats +0.006 (cu1's residual, ~0.004 per
  point × 2.5 points × 0.6 for the 3B late slope). So **r3p ≈ 3.231**: the gate passes narrowly, and
  the public ±0.01 bar vs T++ (3.2064) is missed by ~0.025. Recall may dip with its share (5 % → 4 %);
  ANSWER_IN_WINDOW should offset part of that.
- **Futility unchanged** (3.431 at step 39060). The A7 readouts (free primary, 3 shown generations)
  and A8's natq cell (when Lead's harness has it) go into the readout list.

## cu1 VERDICT (02:33 IST, 2026-11-01): NOT a Win. R-fsq and R-hop MET; R6a / R6b FAIL (the stated rule was learned as a lookup table over trained values); the holdout guard FAILS (+0.045)

`cu1_main_2efb422` (6000; b2p2's recipe, stage B 480M tokens with srd / fsq / hop at 2 / 2 / 1 %, lab 2 %,
102.9M params). The bench cells (instr, kv) errored in cu1's chain (a device bug in the pinned
`2efb422` harness). They were rerun from master `a1101e8` (`~/cu1rb.sh`, same args), for cu1 and b2p2.

| rule | cu1 | b2p2 | bar | verdict |
|---|---|---|---|---|
| R6a stated rule, 0 shots, held-out shift, L = 0 / 2k | 0.00 / 0.00 | 0.00 / 0.01 | ≥ 0.75 both | **FAIL** |
| R6b same, re-read in front of the query, 32k | 0.00 | 0.00 | ≥ 0.75 | **FAIL** |
| R-fsq abstain on an absent key / present / false abstain | 1.000 / 0.940 / 0.060 | 0.000 / 0.990 / 0.000 | ≥ 0.75 / ≥ 0.94 / ≤ 0.10 | **MET** |
| R-hop two-hop (20-way; free 0.20) | 0.570 | 0.100 | ≥ 0.50 | **MET** (20-way as registered) |
| bench kv 8 needles 32k, re-read | 1.00 | 1.00 | not worse by > 0.02 | met |
| guard holdout | 3.5254 | 3.4800 | ≤ 3.500 | **FAIL** |
| guard B2 `_new` / label_diag / skill-pack 2k | 1.000 / 0.913 / 0.81 | 0.944 / – / 0.86 | ≥ 0.894 / 0.884 / 0.81 | met |

**Why R6a fails (diagnostic, `v13_sempty/tmp/stated_rule_probe.py`, CPU, 120 prompts per cell, the bench's
exact format):**

| stated rule, L = 0 | cu1, trained k | cu1, held-out k (5, 9) | b2p2, trained k | b2p2, held-out k |
|---|---|---|---|---|
| 0 shots | **0.97** | **0.00** | 0.07 | 0.00 |
| 1 shot | 1.00 | 0.97 | 1.00 | 0.96 |
| 2 shots | 1.00 | 1.00 | 1.00 | 0.99 |

srd taught words → rule state, but as a LOOKUP TABLE over the 10 trained shift values. Told "shift by
5" or "9" (never trained), cu1 applies a trained shift (mostly 10, 4 or 1). One demonstration fixes it
in both models. That is recall of a stated-value table, not rule inference from words. It is the
honest boundary of goal 3 here: rules are induced from EXAMPLES on unseen values (B2 `_new` 1.000),
not from the WORDS of an unseen value.

**Also reported (A3, A7):**
- verify, cu1 vs b2p2: an unrelated key's row abstains 0.925 (21-way) and says " unknown" outright
  0.32 (free); says the wrong value 0.015 free (b2p2 0.14). A prose row abstains 0.28 / free 0.02.
  So fsq's abstain transfers to the short re-read format for synthetic keys, mostly when forced to choose.
- two-hop is 0.57 ranked but 0.20 said outright.

**Consequences (rules fixed before the numbers):**
1. **srd is out of r3p** (A1: in only if R6a passes); its 0.005 goes to dclm (0.432).
2. **srd is removed from the code** (AGENTS rule; the cu1 pre-registration said so). A stated-rule source
   for cu2 must be redesigned so a table cannot solve it: many values, held-out values in the probe,
   the rule's number used compositionally.
3. **fsq and hop stay** (their rules met).
4. The holdout cost is the r3p A1 budget's input. 10 % synthetic at 600M costs ~+0.045; r3p keeps the
   share at 7.0 % (srd out) with the hard gate 3.2391.

### cd2 result (03:13 IST, 2026-11-01): MET on fresh seeds. State-contrast decoding roughly DOUBLES the one-read reach (~2k → ~4k); by 8k the skill is gone

b2p2, seeds 8200–8299 (never used), a fixed, CPU, `logs/diag/v13_sempty_cd2_b2p2_cb1f9c7.jsonl` (200 rows per L).
Bar (a = 4): 20-way(4k) **0.780** ≥ 0.75 ✓, free(4k) **0.575** ≥ 0.45 ✓, free(0) **0.995** ≥ 0.98 ✓ → **MET**.

**One-read reach curve, one table (Lead's ask; 20-way / free, 100 docs × 2 tasks per cell):**

| model / decode | L = 0 | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| b2p2, plain (cd2 seeds) | 1.00 / 1.00 | 0.805 / 0.58 | 0.355 / 0.165 | 0.055 / 0.01 | 0.045 / 0.005 | – |
| b2p2, plain (seeds 8000–8099) | 1.00 / 1.00 | 0.86 / 0.615 | 0.355 / 0.170 | 0.10 / – | – | 0.04 / – |
| lg8r (8k-trained + reset), plain | 1.00 / 1.00 | 0.905 / 0.68 | 0.465 / 0.265 | 0.105 / – | – | 0.055 / – |
| **b2p2 + contrast a = 4** | 1.00 / 0.995 | 0.91 / 0.685 | **0.78 / 0.575** | 0.11 / 0.045 | 0.045 / 0.005 | – |
| b2p2 + contrast a = 8 | 1.00 / 0.995 | 0.925 / 0.665 | 0.855 / 0.665 | 0.18 / 0.085 | 0.045 / 0.005 | – |

Per task at 4k, a = 4: affine 0.88 / 0.69, shift 0.68 / 0.46.
**2k free misses (a = 4, 63 rows):** the winning token is a wrong letter (" u", " q", " j", …) or a wrong number
(" 67", " 43", …), i.e. the right format with a wrong rule application, not a format failure.

**Reading.** Up to ~4k after one read, the rule is still in the state and a decode-time contrast recovers
it. That beats what 50M tokens of 8k training (lg8r) did at 4k (0.465). Past ~8k the state has really
lost it: contrast cannot recover what is gone (16k at chance for every a). The one-read reach of this
mind is ~2k plain and ~4k with contrast. Beyond that the package's route stays re-fetch (1.00 at
every L) or Library's refresh (N11-R), whose interval may stretch from ~2k to ~4k with contrast (to test).

**Deployment cost (Lead's ask; the memory column stays honest):** the no-pack stream is a SECOND O(1)
state (+1.15 MB, so 2.30 MB in all) and a SECOND forward pass over every session token while contrast is
on (2× compute). It must be the same session without the skill read, so it starts at the read and runs
alongside. Still O(1) in context length; T++'s KV at 4k is ~0.2 GB.
**Not yet shown:** other skills (two-hop, recall, natq); effect on ordinary text when the pack is
irrelevant; a vs a per-token gate. No method claim (CAD / contrastive decoding).

## cd3: does state-contrast decoding hurt ordinary text when the pack is irrelevant? (pre-registration, 2026-11-01 03:14 IST; eval only, CPU; Lead's follow-up)

- **Cells:** b2p2; 60 held-out prose windows of 1,024 tokens (the probe filler, fixed offsets, seed 31), each
  preceded by an IRRELEVANT skill pack (B2 header + 8 demos, affine / shift alternating). Per prose token:
  NLL under plain-with-pack (a = 0), contrast a = 4 and a = 8 (l_pack + a(l_pack − l_nopack), l_nopack = the
  same prose without the pack), and no pack at all.
- **Bar:** mean prose NLL at a = 4 ≤ plain-with-pack + 0.02 → the contrast can be on by default. Otherwise it
  needs a gate (on only for the query after a read), and the size of the harm is reported.
- Also reported: the first 64 prose tokens separately (where the pack's influence is largest), and one shown
  free-text continuation at a = 0 / 4 (greedy, 40 tokens). Prior work: CAD (Shi et al. 2023) reports harm
  when the context is irrelevant; that is what this checks.

### cd3 result (03:22 IST, 2026-11-01): FAIL. The contrast must be GATED: on ordinary text it is destructive

b2p2, CPU, `logs/diag/v13_sempty_cd3_b2p2_5a7e47c.json` (`v13_sempty/tmp/cd3_plain_text.py`).

| prose NLL, 60 windows × 1,024 tokens | all tokens | first 64 |
|---|---|---|
| no pack | 3.816 | 4.287 |
| irrelevant pack, plain (a = 0) | 3.899 (+0.083) | 5.208 |
| irrelevant pack, contrast a = 4 | **8.559** | 27.45 |
| irrelevant pack, contrast a = 8 | 17.87 | 54.81 |

Bar (a = 4 ≤ plain-with-pack + 0.02): **FAIL** by 4.66 nats. Shown (greedy, 40 tokens; prose about blood
sugar after an affine "Map rule" pack): a = 0 → ".8 HbA1C = 4.8 HbA1C = 7.8 HbA1C = 7.8 …"; a = 4 →
" 64 64 64 64 …". The contrast amplifies whatever the pack changed, and on unrelated text that is pure
damage. Even without contrast an irrelevant pack costs +0.08 (+0.92 in the first 64 tokens): the mind
keeps "doing" the rule.

**Consequence for the package (written now):** state-contrast decoding is a **query-time tool, OFF by
default**. It is on only for the answer tokens of a query that the package itself paired with a skill
read (the package knows when it fetched or installed a skill). It is never on for free text. A learned
or adaptive gate (e.g. a set by the disagreement between the two streams, as in adaptive CAD) is a
separate experiment, not yet pre-registered. cd2's reach gain holds only under this gate.

**cu1 A3 addendum: the verify cell with NATURAL keys (03:23 IST, 2026-11-01; reported, CPU).** `run_cur_probe --keys
natural` uses two-word phrases from held-out prose, as `bench --keys natural`. Short re-read format, 200 docs:

| verify row, natural keys | b2p2: correct 21-way / free | b2p2: wrong value said, free | cu1: correct 21-way / free | cu1: wrong value said, free |
|---|---|---|---|---|
| right row (→ value) | 1.000 / 0.940 | – | 0.920 / 0.910 | – |
| unrelated key's row (→ " unknown") | 0.000 / 0.000 | **0.590** | **0.990 / 0.860** | **0.010** |
| prose row (→ " unknown") | 0.000 / 0.000 | 0.000 | 0.970 / 0.310 | 0.000 |

b2p2 copies a wrong natural-key row's value 59 % of the time (Library's package rows: ~57 %). **cu1, whose fsq
training used only synthetic 3-syllable keys, says " unknown" outright 86 % of the time on natural keys** (32 %
on synthetic ones) and almost never the wrong value. The cost is 8 % false abstain on right rows. So the
fresh-key abstain skill transfers to English-phrase keys in this format. Library's package rows decide it
(next: their strict scoring on cu1's checkpoint). Logs `logs/diag/v13_sempty_curprobe_{b2p2,cu1}_verifynat_573db4b.jsonl`.

**cu1 at the package level (Library A10, `7c4356a`; 03:44 IST): NOT met as registered, but the miss is FETCH, not
cu1's reading.** 32k, seed 30000; each mind fetched with its own frozen block-8 address.
- Never-stored abstain by the rule (syn / nat): cu1 1.00 / 1.00, b2p2 0.825 / 0.975. cu1 says " unknown" outright
  only 0.10-0.125.
- kv answered-and-correct: cu1 0.30 / 0.20, b2p2 0.20 / 0.35 (bar fails).
- Split by the top row: with the right row, cu1 says the value 20 / 22 (no false abstain); with a wrong row, cu1
  says a wrong value 1 / 58 (b2p2 6 / 57).
- The kv gap: **cu1's own state addresses facts less well than b2p2's.** The right row was fetched on natural
  keys in 9 docs vs 15. The curriculum changed the state the fetch keys on.

For r3p: Library's fetch must be re-fit on its states, and its address quality is a readout, not an assumption.
The near-miss English-key case barely occurred with own-state fetch (wrong rows were mostly prose). The
fixed-row verify cell (cu1 0.86 " unknown" vs b2p2 0.59 wrong value) stays the evidence for that case.

## cu1n9s VERDICT (04:32 IST, 2026-11-01): FAIL on its rule and on every guard. N9a + N9s cost 0.23 nats of LM and two-hop collapses; cu1n9a does NOT run (A2)

`cu1n9sb_main_fa9e410` (6000; its own stage A `cu1n9sa_pre_fa9e410` with N9a lags 3 + N9s salience, then cu1's
stage B exactly, seed 42). Rules from cu1 amendment A1:

| | cu1n9s | cu1 | rule / guard | verdict |
|---|---|---|---|---|
| (i) two-hop (run_cur_probe) | **0.090** | 0.570 | ≥ cu1 + 0.15 AND ≥ 0.50 | **FAIL** |
| (ii) holdout | 3.7533 | 3.5254 | ≤ cu1 + 0.02 | **FAIL (+0.228)** |
| (ii) label_diag | 0.732 (category 0.488) | 0.913 | ≥ cu1 − 0.05 | **FAIL** |
| fsq present / abstain / false abstain | 0.650 / 1.000 / 0.350 | 0.940 / 1.000 / 0.060 | reported | worse |
| B2 `_new` (t400) | 0.995 | 1.000 | reported | equal |
| skill-pack reach 2k | 0.535 | 0.81 | reported | worse |
| WikiText val | 4.6429 | 4.2748 | reported | worse |

The stage A with N9s (120M tokens) already ended at val 4.0893. The bench cells errored again (the same
pinned-harness device bug at `fa9e410`). They are not rerun: they cannot change a verdict this clear,
and R6a was 0.00 for cu1 anyway.

**Reading:** N9a's content-phase lags with N9s's salient running phase, on this body at 600M, hurt
everything they touch: LM quality, lookup (false abstain 0.35), labels and two-hop (the thing N9s was
for). This matches the earlier N9a failures (n9c lookup, n9r replication). The confound A1 named
(N9s's own stage A) cannot explain a 0.23-nat LM gap and a two-hop collapse. **Per A2, cu1n9a does not
run.** N9s is Library's unit: its removal from the code is Library's call, with the index row.

## clk1: reach is a clock: test-time decay dilation after a skill read (pre-registration, 2026-11-01 04:51 IST; eval only, CPU; Lead's angle 2)

- **Mechanism:** over the span between a skill read and the query, every retention r (notebook decays and
  the echo memory) becomes r^(1/s); s = 1 is the unchanged model. Applied by a model hook off by default
  (parity at s = 1 checked before any number).
- **Cells:** b2p2, skill-pack cell, affine_map_new + shift_wide_new, **fresh seeds 8400–8449** (50 docs per
  task), L = 4k / 8k, s ∈ {1, 2, 4}; free (primary) and 20-way. Cost: cd3's prose cell (60 windows ×
  1,024 tokens after an irrelevant pack) with the dilation over the prose; prose NLL vs s = 1.
- **Bar:** some s ∈ {2, 4} gives free reach(4k) ≥ 0.45 (plain 0.165, contrast a = 4 0.575) AND prose NLL ≤ the
  s = 1 value + 0.05. Then dilation is a cheap, second-stream-free reach extender, gated by the package
  (on only after a read). Reported: 8k, the 20-way column, dilation + contrast together.
- **Prior work:** Δ / dt scaling for SSM length extrapolation (DeciMamba, LongMamba; "Stuffed Mamba" on state
  collapse). The difference is the trigger: the package dilates after its own skill read, measured on
  one-read rule reach. No method claim.

### clk1 result (05:56 IST, 2026-11-01): FAIL as pre-registered (LM cost); dilation + contrast carry one read to 8k (free 0.52, 20-way 0.76)

b2p2, CPU, fresh seeds 8400–8449 (100 rows per L), `logs/diag/v13_sempty_clk1_b2p2_db43950.jsonl` + `..._clk1prose_b2p2_db43950.json`.

| one read, then L filler | free 4k | 20-way 4k | free 8k | 20-way 8k | prose NLL (dilated span) |
|---|---|---|---|---|---|
| s = 1 (plain) | 0.13 | 0.34 | 0.01 | 0.09 | 3.899 |
| s = 2 | 0.36 | 0.63 | 0.03 | 0.10 | 3.966 (+0.068) |
| s = 4 | 0.49 | 0.74 | 0.08 | 0.31 | 4.081 (+0.182) |
| **s = 4 + contrast a = 4** | **0.58** | **0.88** | **0.52** | **0.76** | (as s = 4 over the span; contrast only at the query) |

Bar (some s ∈ {2, 4}: free 4k ≥ 0.45 AND prose NLL ≤ +0.05): s = 4 reaches 0.49 but costs +0.182; s = 2 costs
+0.068 and reaches 0.36. **FAIL.**

**Reading.** Slowing forgetting after a read keeps the trace (20-way 4k 0.34 → 0.74), and the contrast makes it
audible. Together they carry one read to 8k, 4× the training length: free 0.52 where each alone gives 0.08
(dilation) or 0.045 (cd2's contrast). The price is the session text read during the dilation: +0.18 nats at
s = 4, because every head keeps old context longer. The next cut is targeting: dilate only part of the
memory (the echo heads, or the notebooks, or the heads that wrote the pack) to keep the trace at a fraction
of the LM cost (clk2). Prior work as clk1 (Δ / dt scaling for SSM length extension); the combination with a
query-time contrast, triggered by the package's own read, is the setting. No method claim.

## clk2: targeted dilation (pre-registration, 2026-11-01 05:58 IST; eval only, CPU; approved by Lead)

- **Cells:** as clk1 (b2p2, skill-pack affine + shift _new, one read, L = 4k / 8k), **fresh seeds 8500–8549**,
  dilation on ONE memory only: `echo` (the echo / succession memory) or `notebook` (the delta notebooks), s ∈ {1, 4, 8},
  each with and without the query-time contrast (a = 4). Cost: clk1's prose cell with the same target and s.
- **Bar:** some target and s with free(8k) WITH contrast ≥ 0.40 AND prose NLL ≤ s = 1 + 0.05. Then targeted dilation
  + contrast is a usable mind-level reach extender, gated by the package: on only between a package-paired read and
  its query, never on free text (the contrast's rule, cd3).
- Prior work as clk1. No method claim.

## cu2: one field + stop, and verify a fetched row; cu2a = data, cu2b = data + ACh-1 (pre-registration, 2026-11-01 06:04 IST; paired arms per Lead's round)

**Why.** After the package's default read became warm read + contrast (Library A9, +0.142), the mind-side gaps are:
tool fields running on past the value (47–55 %, natural 0.575), and abstaining on prose rows (cu1 free 0.31).
Abstaining where the words differ must be TRAINED: query-time checks failed on b2p2 (key-swap AUROC 0.70, A11;
fetch scores 0.66–0.77, A13), while cu1's trained abstain transferred (wrong value 3 %). The readout is the target
(step back 04:51): ACh-1 trains the plain logits after an answer tag toward the mind's own evidence contrast.

**Closest prior work (novelty rule):** SQuAD 2.0 (unanswerables), R-Tuning (refusal tuning), Self-RAG (relevance
tokens), slot filling; for ACh-1: CAD (Shi et al. 2023), decoding distillation (Wan et al. 2023), ContextDPO.
**What differs:** an O(1) recurrent mind, free generation only, held-out values / keys / formats, and ACh-1
distilling the mind's OWN evidence contrast behind a tag (gate by construction). No recipe novelty claimed.

**Arms** (4090, same seed 43 and data; b2p2's recipe: M2 + echo + shared reads, stage A `c1m2a_pre_7ae6f1f` (local copy,
md5 checked), stage B 480M tokens, LR 1.2e-3, warmup 100, VAL_EVERY 4883; batch 18 × 2048 as GRAD_ACCUM=3
(exact accumulation: `4397449`, grads within 2e-5 of the full batch); ANSWER_IN_WINDOW=1; selftest 24/24):
- mix dclm 0.42 / fineweb 0.39 / smoltalk2_mid 0.10 / recall 0.04 / b2 0.03 / fld 0.01 / vfy 0.01 (synthetic 9 % vs b2p2's 7 %;
  fld / vfy: `a7c4887`, test PASS);
- **cu2a** = the mix; **cu2b** = the same + ACH_LAMBDA=0.2, ACH_ALPHA=4 (16 short + 1 long tagged item per step, long
  fill ≤ 4096). The teacher is trusted only where its top token is the gold token: added after the first smoke
  collapsed tagged accuracy to 0.00 (KL 2.88 → 0.20), before any cu2 number; with it, tagged acc 0.79 / 0.67.
- Expected ends: cu2a ~2.4 h (55k tok/s), cu2b ~3.7 h (36k tok/s); one after the other.
- **Futility at step 4883 (180M tokens):** cu2a stops if val > b2p2's 3.7508 + 0.04 = 3.7908; cu2b stops if val >
  cu2a's at step 4883 + 0.02 (Lead's 100M LM screen, at the first val point).

**Readouts (free generation primary):**
- `run_fld_probe` (seed 31000, 100 docs; qset in = trained question formats, out = held-out ones);
- `run_cur_probe` verify (natural / synthetic);
- skill-pack 0 / 2k / 4k (plain, contrast a = 4, and --tag);
- B2 t400; attr3b holdout vs b2p2; cd3's prose cell (irrelevant pack); 3 shown generations;
- package A9 cells with the arm's own address, with and without the twin (Lead / Library harness, reported).

**Bars, cu2a (the data effect, vs b2p2):**
- R-fld-in: fld, natural keys, trained row formats, qset in: exact ≥ 0.70. b2p2's in-format baseline is being measured
  now; its out-format exact is 0.00–0.02.
- R-fld-out: the same with held-out question formats: exact ≥ 0.40 (transfer).
- R-stop: among fld answers that start with the value, run-on ≤ 0.10.
- R-vfy-in: natural keys, qset in: " unknown." exact ≥ 0.70 on prose rows AND on wrong-key rows, AND right rows exact ≥ 0.70.
- Guards: holdout ≤ 3.500 (b2p2 + 0.02); B2 `_new` ≥ 0.894; skill-pack 2k (20-way) ≥ 0.81.

**Bars, cu2b (ACh-1 alone, vs cu2a):**
- R-ach-mind: skill-pack free reach at 4k, --tag, plain decode (no twin) ≥ cu2a's free reach at 4k with contrast
  a = 4 − 0.05. The tag makes the plain read do what the twin did.
- R-ach-gate (Lead): package A9 cells on cu2b WITHOUT the twin ≥ cu2b WITH the twin − 0.03.
- Guards: cd3's prose NLL (irrelevant pack, no tag) ≤ cu2a + 0.02; holdout ≤ cu2a + 0.02; cu2a's fld / vfy cells not
  worse by > 0.05.

**Consequences:** cu2a fails a guard → no fld / vfy in any release recipe. cu2b passes → ACh-1 enters the next 3B
recipe (r3p's successor), GPU via Lead. Either failing → recorded and removed per the AGENTS rule.

### r3p interim (2026-11-01 08:30 IST): r3p trails r3c by 0.10 → 0.15 nats from the first val. Cause: the answer-weighted synthetic data takes LOSS WEIGHT from text, not plumbing

Val (step 3906 / 7812 / 11718 / 15624 / 19530 / 23436): r3p 4.0206 / 3.8953 / 3.8431 / 3.7731 / 3.6827 / 3.6665 vs r3c
3.9172 / 3.7952 / 3.7181 / 3.6251 / 3.5624 / 3.5117 (gap +0.103 → +0.155).
**Ruled out:**
- (1) Code drift: r3c's checkpoint gives bit-identical val CE under its own code `38e3c29` and HEAD (3.19988 both, same logits).
- (2) Data plumbing: same val cache, same stage-A cursor, the same fineweb-edu sample-10BT shards (local copy vs HF), same args.
- (3) The blend: it over-serves smoltalk2 in both runs alike (~15 % vs 10 %).

**What differs** (cursor token counts, r3p at step 24000 vs r3c's final):
- r3p stage-B shares: b2 1.96 % (2× its weight), lab 1.14 %, hop 0.66 %, fsq 0.53 %, recall 3.5 %. Synthetic 7.8 % vs r3c's 4.4 %.
- The loss is sum(w·CE)/sum(w) with ×100 on answers, so **the share of loss weight on ordinary text** falls:
  r3c 0.975, r3p 0.85, b2p2 0.872, cu1 ≤ 0.768, cu2a 0.758 (measured answer density per source: recall 0.0057, b2 0.041,
  lab 0.038, fsq 0.028, hop 0.029, fld 0.065, vfy 0.107 answer tokens per token).
- At 600M this costs ~0.34 nats per unit of share lost (b2p2 +0.033, cu1 +0.078 vs c1m2 3.4473; lab alone +0.004).
  That predicts r3p ≈ +0.04. The 3B gap is 2.5–3.5× that, so it is not fully explained (it may compound with
  tokens, or interact with Adam's second moments, which large answer gradients dominate).

**My error, recorded:** r3p A1 and cu2 budgeted synthetic data by TOKEN share. The unit that matters is LOSS-WEIGHT
share. cu2a's text share 0.758 predicts ~+0.04 vs b2p2, so its 3.500 guard will likely fail. **Rule for every future
recipe:** state the loss-weight share on ordinary text next to the token mix, and keep it ≥ ~0.95 (r3c's level), via a
lower answer weight for dense sources or a two-term loss (LM mean + β × answer mean). Needs a 600M check before any 3B use.
The r3p futility rule (3.431 at step 39060) decides at ~10:40 as pre-registered.
**Correction (08:31 IST):** cu2a finished (08:26 IST) with holdout **3.4975**, which PASSES its 3.500 guard (+0.0175 vs b2p2). The
"+0.04, will likely fail" prediction above was wrong. The 600M per-share fit overstates the cost for fld / vfy (perhaps
ANSWER_IN_WINDOW, or because short single-answer items weigh differently), and r3p's 3B gap stays only partly explained.
The loss-weight-share rule stands as a reporting rule; its numeric fit does not.

### clk2 result (08:32 IST, 2026-11-01): MET. Dilating only the ECHO memory carries one read to 8k (with the contrast) for +0.013 nats on the text

b2p2, CPU, fresh seeds 8500–8549 (100 rows per L), `logs/diag/v13_sempty_clk2_{echo,notebook}_b2p2_89ea841.jsonl` + prose json.

| target, s | free 4k | free 4k + contrast | free 8k | **free 8k + contrast** | 20-way 8k + contrast | prose NLL (vs 3.8988) |
|---|---|---|---|---|---|---|
| none (s = 1) | 0.17 | – | 0.01 | – | – | 3.8988 |
| **echo, s = 4** | 0.46 | 0.62 | 0.05 | **0.46** | 0.64 | **3.9120 (+0.013)** |
| echo, s = 8 | 0.51 | 0.60 | 0.13 | 0.49 | 0.74 | not measured (prose cell ran s ≤ 4) |
| notebook, s = 4 | 0.25 | 0.61 | 0.01 | 0.12 | 0.29 | 4.0630 (+0.164) |
| notebook, s = 8 | 0.31 | 0.67 | 0.05 | 0.19 | 0.39 | not measured |
| (clk1: all memory, s = 4, seeds 8400–8449) | 0.49 | 0.58 | 0.08 | 0.52 | 0.76 | 4.0812 (+0.182) |

**Bar (free 8k + contrast ≥ 0.40 AND prose ≤ +0.05): MET by echo, s = 4** (0.46, +0.013). The skill's trace lives in the
**echo memory**. Slowing only its forgetting keeps the trace to 8k, while the notebooks hold most of the language cost.

**Against the baselines at the same one-read gap (8k), as they are** (the four-model table, seeds 8000–8099, 20-way
mean of affine + shift): T++ 0.195 (pack still inside its window), Mamba-2 0.015, PAM plain ~0.10. **PAM + echo dilation +
contrast 0.64 (20-way) / 0.46 (free).** Different seed sets; the baselines get none of the query-time tools (comparison
rule).

**Deployment rule (written now, as for the contrast):** dilation is ON only between a package-paired read (a skill pack
the package installed) and its query, and only on the echo memory. It is never on during free text outside such a
span; the contrast is on only for the query's answer tokens. Cost while on: the session text in the span is read at
+0.013 nats; the contrast's second stream costs +1.15 MB and 2× compute at the query.

**Not yet shown:** a second checkpoint (e.g. b2p1 or cu2a); s between 4 and 8 with its prose cost; other skills
(two-hop, recall); where a refresh fits (Library's N11 with dilation). Prior work as clk1 (Δ / dt scaling for SSM
length extension); the setting is the package-triggered, memory-targeted dilation. No method claim.

**clk2r: replication on a second checkpoint (pre-registration, 08:32 IST; CPU).** cu2a (`cu2a_main_24b2969`, its own 600M training),
fresh seeds 8600–8649, echo-only dilation s = 4 with and without the contrast, L = 4k / 8k, plus the prose cell (s = 1 / 2 / 4, echo).
Bar: free 8k + contrast ≥ 0.40 AND prose ≤ s = 1 + 0.05, as clk2. If met on both checkpoints, the result is replicated across
two independent 600M minds.
**clk2r amendment (08:35 IST, before any clk2r number; Lead):** the replication runs on TWO checkpoints, cu2a AND b2p1
(`b2p1_main_38e3c29`, seed 42), both on fresh seeds 8600–8649, with echo s ∈ {1, 4, 8} ± contrast. The prose cost is measured at
s = 1 / 4 / 8 on cu2a, b2p1 and b2p2, so echo s = 8 can be stated or dropped. Bar unchanged, per checkpoint, at s = 4; s = 8 is
claimed only where its own prose cost is ≤ +0.05.

### cu2a result (08:46 IST, 2026-11-01): the data teaches the TRAINED formats well (one field + stop 0.80, verify 0.83–0.99, no run-on) at +0.0175 LM, but transfer to new wordings is weak and abstain is format-bound

`cu2a_main_24b2969` (4090; b2p2 recipe + fld 1 % + vfy 1 %, seed 43, 480M tokens, 102.9M params; val 3.4975). Free generation,
`run_fld_probe` seed 31000 (100 docs per cell), held-out values and keys throughout.

| bar | cu2a | bar | verdict |
|---|---|---|---|
| R-fld-in (natural keys, trained row + question formats; 1 / 3 rows) | 0.93 / 0.67 (mean 0.80) | ≥ 0.70 | **MET** (synthetic 1.00 / 0.76) |
| R-fld-out (held-out question formats) | 0.35 / 0.16 (mean 0.26) | ≥ 0.40 | **FAIL** |
| R-stop (run-on among answers that start with the value) | 0.00 in every cell | ≤ 0.10 | **MET** |
| R-vfy-in (natural: prose / wrong-key / right row) | 0.99 / 0.83 / 0.98 | ≥ 0.70 each | **MET** (synthetic 1.00 / 0.84 / 0.98) |
| guard holdout | 3.4975 | ≤ 3.500 | met (attr3b +0.016 web / +0.022 PG-19, mostly new tokens) |
| guard B2 `_new` / skill-pack 2k (20-way) | 0.990 (free 0.966) / 0.835 | ≥ 0.894 / ≥ 0.81 | met |

b2p2 baseline on the same probe: ≈ 0.00–0.03 on every cell (in and out).

**Reported:**
- held-out ROW format ("[key] f=v"): ≈ 0.00 in and out;
- missing field → " unknown." 0.99 (in) / 0.41–0.46 (out);
- verify with held-out questions: 0.32–0.46;
- fld_doc (one read, 2k filler): 0.00;
- the bench-format verify cell ("Query: K refers to"): cu2a says the WRONG value 0.78 free (cu1 0.01, b2p2 0.59).
  vfy's abstain does not carry to a format cu2a never saw it in. cu1's fsq did, in its own format.
- Skill pack: plain 2k 0.835 / 0.595, 4k 0.355 / 0.170; contrast 4k 0.795 / 0.545 (as b2p2).
- The untrained " [answer]" tag breaks answering (free 0.00 even at L = 0): cu2b's baseline for the tag.

**Shown generations (A7, cu2a):**
- rule: "Map rule. 3 -> 10 … 11 ->" → " 26", then "18 -> 40" (right);
- plain-language recall question: "A. The owner of the new office is A." (wrong);
- free text: fluent, invented facts.

**Reading.** At 103M a skill taught by examples is learned in its trained surface forms and transfers weakly to new
wordings (0.26–0.46) and not at all to a new row layout. This is the same boundary as the stated-rule table (cu1) and
natq: the mind generalises VALUES and KEYS, not FORMATS. For the package, the library should present fetched rows in
the trained row format and ask in a trained question form (the package controls both). Generalising formats would
need many more formats in training, not more weight.

### clk2r result (09:56 IST, 2026-11-01): s = 4 replicates on 2 of 3 checkpoints; **s = 8 holds on all three** (one read to 8k, free 0.43–0.58, 20-way 0.74–0.89, ≤ +0.021 nats)

Echo-only dilation + contrast a = 4, one read, L = 8k, free (20-way); prose = NLL on the dilated span vs s = 1.
Logs `logs/diag/v13_sempty_clk2r_echo_{cu2a,b2p1}_689b1d0.jsonl` + `..._clk2rprose_echo_{cu2a,b2p1,b2p2}_689b1d0.json`.

| checkpoint (seeds) | 8k s = 4 + c | 8k s = 8 + c | 4k s = 4 + c | prose s = 4 / s = 8 |
|---|---|---|---|---|
| b2p2 (8500–8549, clk2) | 0.46 (0.64) | 0.49 (0.74) | 0.62 (0.88) | +0.013 / +0.016 |
| cu2a (8600–8649) | **0.47** (0.78) | **0.58** (0.89) | 0.79 (0.95) | +0.016 / +0.021 |
| b2p1 (8600–8649) | 0.34 (0.72) | **0.43** (0.81) | 0.83 (0.97) | +0.012 / +0.015 |

**As pre-registered:** at s = 4 (bar free 8k + c ≥ 0.40): b2p2 MET, cu2a MET, b2p1 FAIL (0.34). s = 8 is claimed where its own
prose cost is ≤ +0.05: it is (+0.015 to +0.021) and passes on all three (0.43 / 0.49 / 0.58). **Replicated across three
independent 600M minds and three seed sets at s = 8.**
Baselines at the same 8k one-read gap, as they are (20-way, seeds 8000–8099): T++ 0.195, Mamba-2 0.015. PAM plain ~0.10.

**In the package (Library A16, b2p2, 32k):** refresh every 8k with echo dilation (s = 4) + contrast 0.42 < refresh every 2k
plain 0.63 < every 2k + contrast 0.70. So the package default stays refresh ~2k + contrast. The dilation is the mind-level
reach result (one read lasts ~4× the trained length at about half the accuracy), not a cheaper package default yet.
s = 8 in the refresh setting is untested.

## w1s: 100M LM screen of the W1 window reader (pre-registration, 2026-11-01 10:12 IST; Lead-approved as a screen; the 600M pair is the user's decision)

**Why:** Lead's T++ + library arm: at equal size T++ reads a fetched multi-field row much better (tool 0.90 vs b2p2
0.475, cold, free). This is the known fixed-state associative-recall weakness (MQAR / Zoology; Based). W1 = bounded
sliding-window attention (window 128) in blocks 5, 9, 13 (`d028aab`, tests `v13_sempty/tmp/w1_test.py`), so a short
fetched span is an exact-copy read. KV is fixed by the window, so the body stays O(1) in context length.
**Prior work:** Griffin / Hawk (RG-LRU + local attention), Based (linear attention + sliding window). Repair, no claim.

**Arms** (4090 after cu2b, same seed 43 and data; both warm-start from `c1m2a_pre_7ae6f1f`; stage B **100M tokens**,
b2p2's mix dclm / fineweb / smoltalk2_mid / recall / b2 = 0.43 / 0.40 / 0.10 / 0.04 / 0.03, LR 1.2e-3 cosine over the 100M,
GRAD_ACCUM 3, ANSWER_IN_WINDOW=1):
- **w1s0**: none.
- **w1s128**: SWA_LAYERS=5,9,13, SWA_WINDOW=128 (+4.15M params, ~4 %, stated; new params identity at start).

**Readouts:**
- holdout val;
- Lead's text-library cold read (`v16/bench.py --text_package --skip_state`, tool / kv / para × syn / nat, 32k, seed
  18000, 40 docs, free);
- skill-pack 0 / 2k / 4k plain + contrast;
- the s8 spread cell (b2 `_new`, 0 / 2k / 4k);
- clk2's 8k cell (echo dilation s = 8 + contrast).

**Bars:**
- **Target:** text-library TOOL cold free (syn + nat mean), w1s128 ≥ w1s0 + 0.15.
- **Guards (Lead):** holdout w1s128 ≤ w1s0 + 0.02; s8 spread D (mean of 2k / 4k) ≥ w1s0 − 0.05; clk2 8k free ≥ w1s0 − 0.05.
- At 220M total tokens the skills are weaker than at 600M; the guards are relative, arm against arm.

**Consequence:** the screen's numbers go to the user beside Library's package-side field-split result. The 600M pair is NOT
launched from this screen.

### cu2b result (10:14 IST, 2026-11-01): FAIL by its pre-registered futility rule. ACh-1 cost +0.038 nats at 180M tokens (bar +0.02); the run stopped at step 5000

`cu2b_main_24b2969` (cu2a's data and seed + ACh-1, λ 0.2, α 4, fill ≤ 4096). Val at step 4883: **3.8069 vs cu2a 3.7691** (+0.038; bar
cu2a + 0.02 = 3.7891). The chain wrote `.stop`, the launcher did not resume (the new stop file worked), and the chain moved on to w1s.
During the run the ACh KL stayed ~0.000–0.001, tagged-answer accuracy 0.49–0.75, and the teacher was trusted on 0.40–0.65 of
answer tokens. The direct gradient of the aux term was therefore small, and the LM cost is not explained by its size; arm-to-arm
noise at 180M may be part of it (two seeds at 600M differ by ~0.003). Not separated: no GPU slot is spent on it.
**Verdict:** ACh-1 (answer-mode distillation) does not pass its LM screen in this form. Per the pre-registration it is removed
from the code (the index row names the last commit that has it). The query-time contrast stays the package's tool (A9, cd2,
clk2). Exact gradient accumulation and the launcher's `.stop` file stay; they are general.

## rxk: a RULE held in state × a FACT fetched at the query (pre-registration, 2026-11-01 10:26 IST; eval only, CPU; the thesis in one row)

**Idea:** the package's whole claim in one cell. A skill pack (B2 header + 8 demos, held-out rule values) is read ONCE at t0; L tokens
of held-out prose follow; then the package hands over a fetched fact row " Record: K | x: <v>." and the question.
- **explicit:** the package substitutes the fetched value, so the question is the trained query line "<v> ->".
- **implicit:** the question names the key, "K ->", so the mind must take v from the row itself (an untrained composition).

**Cells:** b2p2 and cu2a; affine_map_new + shift_wide_new; **fresh seeds 8800–8849** (50 docs per task); L = 0 / 2k / 4k / 8k.
Synthetic keys (bench `_recall_key`). Decodes: plain; contrast a = 4 (pack vs no pack, same row + question); echo dilation s = 8 over
the filler + contrast (for L ≥ 4k). Free primary, 20-way reported.

**Bars:** explicit + contrast at 2k, free ≥ 0.70 AND explicit + echo dilation s = 8 + contrast at 8k, free ≥ 0.40, on BOTH checkpoints.
Then "a rule read once, applied to a fact fetched 4× the trained length later" works as a package. Implicit: reported only
(predicted near chance: the mind is format-bound, cu2a).

**Prior work:** none for this exact cell; the parts are CAD / contrastive decoding and Δ-scaling (clk1). The cell is the
measurement; no method claim.
**rxk amendment (10:27 IST, before any rxk number; Library's points):**
- **Ceiling control** (the transformer path): the pack re-inserted right before the row + question (RAG + few-shot, no state needed).
- **Floor:** no pack at all (the row + question alone).
- **Adaptive contrast arm** (AdaCAD-style): a = 4 · JSD(p_pack ‖ p_nopack) / ln 2, beside the fixed a = 4.
- The rule lives ONLY in the state (read once, L tokens back); the fact lives ONLY in the fetched row.
- The thesis claim needs explicit + contrast clearly above the floor, and is read against the ceiling. Bars unchanged; the
  controls and the adaptive arm are reported.
- Run on the 4090 beside w1s (probe memory cap 0.2), not on CPU.
**rxk amendment 2 (10:28 IST, before any rxk number; Lead adopts Library's controls).**
- **Exact formats** (for the package planner):
  - fetched row: " Record: <K> | x: <v>." (affine) or " Record: <K> | letter: <v>." (shift), then a newline;
  - explicit question: the trained query line "<v> ->" (the planner substitutes v from the row);
  - implicit question: "<K> ->";
  - K = the bench's synthetic `_recall_key`.
- **Controls:** CEILING = the pack re-inserted right before the row (RAG + few-shot: if it equals the state-only row, the mind
  added nothing); FLOOR = no pack anywhere.
- **T++** (`b2t2_main_38e3c29`, seed 43, the s8 baseline) on the same cells, as it is: plain decode only. At 2k the rule is
  inside its window; at 4k / 8k it is past its trained length.
- **Prefix note** (for the write-up): a reader conditioned on a single PREFIX vector would likely drop the rule (Library's SR-0:
  one vector does not carry a rule). In R × K the mind reads itself (warm + contrast); a reader organ, if ever built, is for
  records and formats, not for applying held rules.
- **Run on CPU:** the 4090 has ~2 GB free beside w1s and Library's jobs, so it is below the VRAM rule.

## bud: the loss-weight budget check at 600M on r3c's OWN body (pre-registration, 2026-11-01 10:51 IST; 6000; Lead-approved)

**Why:** r3p (3B, r3c's 16nb body + ~4 % answer-dense synthetic data) trailed r3c by 0.10–0.15 and stopped by futility at
1.44B (3.5104 vs 3.431). Its text share of the loss weight was ~0.86 vs r3c's ~0.977. The 600M fit (M2 runs) predicts only
~+0.04, so the body or the scale may amplify it. No r3p' launches before this check.

**Arms** (all from `r3ca_pre_38e3c29`; stage B 480M; seed 42; r3c's body: baseline_real_pm 16 notebooks + ECHO=1; LR 1.2e-3,
warmup 100, batch 18 × 2048 as GRAD_ACCUM=2 (exact, same math; two arms fit side by side); FINEWEB_LOCAL_DIR; VAL_EVERY 4883):
- **bud0**: r3c's mix (dclm / fineweb / smoltalk2_mid / recall = 0.44 / 0.41 / 0.10 / 0.05), as r3c (no ANSWER_IN_WINDOW).
- **bud1**: r3p's mix as run (… / recall 0.04 / b2 0.01 / lab 0.01 / fsq 0.005 / hop 0.005, ×100 answers, ANSWER_IN_WINDOW=1;
  text share ~0.86).
- **bud2**: bud1 + ANSWER_W_SRC=b2:20,lab:20,fsq:20,hop:20 (recall stays ×100). Computed text share 0.951 (`8096623`,
  parity test PASS).
- Order: bud0 + bud1 side by side, then bud2 (gated on both arms' DONE log markers).

**Readouts per arm:** holdout + WikiText; recall t40 + PG-19 (the screen's own); B2 `_new` t400; label_diag t200; rules;
`run_cur_probe` fsq / hop; skill-pack 2k.

**Bars (fixed now):**
- bud1 − bud0 holdout STATES the mix cost at 600M on this body (reported; compare with r3p's 3B gap of 0.11 at 1.44B).
- **bud2 passes** if holdout ≤ bud0 + 0.02 AND each skill (B2 `_new` mean, label_diag mean, fsq abstain, two-hop) ≥ bud1's − 0.05.
- If bud2 passes, the budget (synthetic sources at ×20 with text share ≥ 0.95) is the rule for r3p'. If it fails on skills,
  lower weights don't teach: the lesson is that the skills need their weight, and r3p' carries fewer synthetic tokens instead.

## r3p VERDICT (10:53 IST, 2026-11-01): stopped by its pre-registered futility rule at 1.44B stage-B tokens (3.5104 vs 3.431). The cause is the answer-weighted synthetic mix (text share of the loss weight 0.86 vs 0.977), not code or plumbing

`r3p_main_c13d8fe` (6000; r3c's body and stage A; stage B r3p A1 mix; 103.2M). Val at step 39060: **3.5104** (r3c 3.4010; futility
3.431). The gap narrowed 0.155 → 0.11 before the stop.
- **Per-source tokens at the stop** (all 1.56B incl. stage A): dclm 41.3 / fineweb 36.1 / smoltalk2 15.4 / recall 3.22 / b2 1.81 /
  lab 1.05 / hop 0.61 / fsq 0.49 %. b2 was over-served about 2× its weight (the blend serves short docs in excess).
- **Text share of the loss weight** (measured answer densities): ~0.86, vs r3c ~0.977.
- **Ruled out** (see "r3p interim"): code drift (bit-identical val CE), the val set, cursors, shards, args.
- **Cold text-library read** (Lead's harness, `--text_package --skip_state`, 32k, seed 18000, 40 docs; free; `logs/v16_textarm_r3p_32768_e21f58e_*.jsonl`):

| cell, syn / nat | r3p (3B mix, stop) | b2p2 (600M) | T++ (600M) | Mamba-2 (600M) |
|---|---|---|---|---|
| kv | 1.00 / 0.925 | 0.90 / 0.775 | 1.00 / 0.875 | 1.00 / 1.00 |
| para | 0.50 / 0.625 | 0.775 / 0.80 | 0.70 / 0.65 | 0.975 / 0.95 |
| **tool** | **0.775 / 0.675** | 0.475 / 0.40 | 0.90 / 0.90 | 0.425 / 0.475 |

  r3p reads multi-field tool rows far better than b2p2 (3B scale and / or its lab / fsq / hop docs), still below T++; para is lower.
  Different scales: no 3B T++ text row exists.
- **Consequence:** no r3p' until the 600M budget check (bud, running on the 6000 since 10:52 IST) states the mix cost on this body
  and whether ×20 synthetic answers (text share 0.951) keep the skills.

## w1s VERDICT (13:02 IST, 2026-11-01): target FAIL. W1 (sliding-window attention, window 128, blocks 5 / 9 / 13) does not improve the read of a fetched row (tool +0.0125, bar +0.15). All guards pass. W1 lowers neither holdout nor copy-class loss. Unexpected: the reach and rule cells rose, one seed

**Arms:**
- `w1s_w0_508aaff` (102.88M) and `w1s_w128_508aaff` (107.04M, +4.15M = +4 %).
- Both: 120M stage A (`c1m2a_pre_7ae6f1f`), then 100M stage B on b2p2's mix, seed 43; 220M tokens in total.
- The first chain never started w128: the arm overrides were split on commas, so SWA_LAYERS=5,9,13 broke apart (env exit 127;
  fixed in `2b5937f`). w128 was rerun alone from the same pinned worktree at 11:50 IST, with SWA passed via BASE (the config shows
  swa_layers (5, 9, 13)).

| readout | w0 | w128 | bar | |
|---|---|---|---|---|
| **tool cold free, syn / nat (mean)** | 0.575 / 0.625 (0.600) | 0.650 / 0.575 (0.6125) | ≥ +0.15 | **FAIL** |
| holdout | 3.7454 | 3.7510 | ≤ +0.02 | pass |
| s8 D = mean over 2k / 4k, scored / free | 0.559 / 0.334 | 0.730 / 0.464 | ≥ −0.05 | pass (higher) |
| clk2 8k (echo s = 8 + contrast), free / 20-way | 0.12 / 0.31 | 0.21 / 0.53 | ≥ −0.05 | pass (higher) |

**Other readouts:**
- kv, syn / nat: 1.00 / 1.00 in both arms.
- para, syn / nat: 0.925 / 1.00 (w0) and 0.90 / 0.95 (w128).
- Skill pack, 20-way plain at 2k / 4k: 0.61 / 0.105 → 0.735 / 0.225.
- Skill pack, free with contrast at 2k / 4k: 0.87 / 0.335 → 0.74 / 0.325.
- WikiText val loss: 4.5376 → 4.5459.
- PG-19 far_gain at 4–8k: +0.0432 → +0.0438.
- **Copy class** (attr3b, w128 − w0; `logs/diag/v13_sempty_attr3b_w1s_w128_vs_w0_508aaff.txt`):
  - PG-19: total +0.0035 (copy −0.009, seen +0.013, new +0.002);
  - web: +0.0063 (copy −0.000, seen +0.007, new +0.009).
- **Decays** (filler half-lives, affine L4096; `logs/diag/v13_sempty_halflives_w1s_*`):
  - notebooks: unchanged, every head within ±20 %;
  - echo memory: geomean 2201 → 1943, slightly SHORTER.
- **W1 is in use:** its zero-init o_proj grew to rms 0.020–0.023, a normal trained scale.

**Reference on the same cells and flags** (PACKAGE_STRICT=1 PACKAGE_NORM=1, seed 18000, 40 docs; tool free, syn / nat;
`logs/v16_textarm_{tflr_lr1.2e-3_e072a8f,b2p2_main_38e3c29,b2t2_main_38e3c29,c1m2b_main_7ae6f1f,tfc1b_main_b788731}_32768_508aaff_tool_*.jsonl`):

| model | tool free, syn / nat (mean) |
|---|---|
| T++ 100M from scratch (`tflr` LR 1.2e-3, holdout 4.07) | 0.50 / 0.40 (0.45) |
| b2p2 (PAM 600M, 102.9M) | 0.475 / 0.40 (0.4375) |
| T++ 600M (b2t2, 101.9M) | 0.80 / 0.675 (0.7375) |
| PAM 600M on the aptitude recipe (c1m2b, 102.9M) | 0.575 / 0.35 (0.4625) |
| T++ 600M on the aptitude recipe (tfc1b, 101.9M) | 0.875 / 0.90 (0.8875) |

- **Same-recipe pairs at 600M** (all rows use one flag set, the one above):
  - b2 mix: T++ 0.7375 vs PAM 0.4375, gap 0.30;
  - aptitude recipe: T++ 0.8875 vs PAM 0.4625, gap 0.425.
- **Correction (13:06 IST):** Lead's T++ 0.90 / 0.90 is tfc1b, NOT b2t2. The flags are the same as here; tfc1b gives 0.875 / 0.90 on
  them. The difference is the checkpoint, not the flags (I first wrote "other package flags", which was wrong).

**Reading:**
- **Skill, not view.** The fetched row sits inside the window (the read is 20–60 tokens) and the window is in use, yet field selection
  does not improve. The miss is knowing WHICH field the question asks for, not seeing the row exactly. This matches cu2a: skills come
  with formats in the data.
- **Where the reader lives** (the rule stated beforehand in SCRATCHPAD): W1 lowers neither holdout nor copy-class loss, and it does
  not help the read. So the reader stays in the PACKAGE (Library A19's field cut: 0.90 / 0.80).
- **Screen power:** T++'s own tool read climbs 0.45 → 0.74 from 100M to 600M, so the skill grows with training, and a 600M W1
  could differ. Nothing here points that way (no read, LM or copy gain). The 600M pair is not recommended for reading.
- **Unexpected: reach and rule cells rose with W1:**
  - s8 +0.17 scored / +0.13 free;
  - clk2 8k 20-way +0.22;
  - skill pack plain 20-way +0.12 at 2k and 4k;
  - the decays did not change.

  A 128 window cannot see across a 2–4k spread, so if this is real, the gain is in how the state is WRITTEN or READ. One guess:
  local format work moves to the window and leaves cleaner rule traces. Probe noise is small (SE ≈ 0.025–0.05), but the training
  spread between arms is unmeasured (b2p1 / 2 / 3 s8 spread 0.10 at 600M). One seed per arm: no claim.
- **Aside:** w0 (220M tokens) reads tool rows better than b2p2 (600M): 0.60 vs 0.44. The recipes differ (ANSWER_IN_WINDOW, exact
  accumulation, seed). Not chased.

**Consequence:**
- W1 is not a body unit for reading.
- Proposed to Lead: a seed replicate of the pair (w1r, below) to test the reach gain.
  - If it fails, W1's code is removed (index rule).
  - If it holds, W1 as a reach unit goes to the user as a 600M decision.

## w1r: is W1's reach gain real? Seed replicate (pre-registration, 2026-11-01 13:02 IST; PROPOSED, awaiting Lead's GPU call)

**Why:** in w1s, W1 raised s8 D by +0.17 and clk2 8k 20-way by +0.22 with no change in the decays (one seed). s8 is bar 3
(learning beyond recall). If the gain is real, a +4 % local-attention unit is a reach unit, not a reader.
**Prior work:** Griffin / Hawk (RG-LRU + local attention) and Based (linear attention + sliding window) pair local attention with a
recurrent state for LM quality and recall.
**What differs:** the claim tested here is that local attention improves a recurrent state's rule retention ACROSS 2–8k (far
outside its window) without changing its decays. No claim before replication.

**Arms:** w0 and w128 exactly as w1s (same stage A, mix, 100M, `508aaff` worktree), SEED=44. 4090, one arm after the other
(~35 min each at 46k tok/s, plus readouts).

**Readouts:** holdout; s8 (b2 `_new`, 0 / 2k / 4k, 400 trials, probe seed 7000); clk2 8k; skill pack 0 / 2k / 4k. The text-arm
tool read is also reported.

**Bars:**
- **Replicates if:** w128 − w0 ≥ +0.08 on s8 D (scored), AND ≥ +0.10 on clk2 8k 20-way, AND holdout w128 ≤ w0 + 0.02.
- **Pooled report:** the mean over seeds 43 + 44.
- **If it fails:** W1's code (`WindowRead`, `--swa_layers`, the SWA env) is removed and the index row names its last commit.

## rxk VERDICT (14:16 IST, 2026-11-01): FAIL as pre-registered. Bar 1 is missed on both minds (explicit + contrast at 2k, free 0.61 / 0.62 vs 0.70); bar 2 is MET on both (explicit + echo dilation s = 8 + contrast at 8k, free 0.45 / 0.49 vs 0.40). A rule read once, held only in the state, is applied to a fetched fact 4× past the trained length (floor 0.00–0.02; T++ as it is 0.01–0.07)

**Setup:**
- 100 docs per L: affine_map_new + shift_wide_new × seeds 8800–8849. CPU, `99d2951`.
- Models: b2p2 (102.9M) and cu2a (102.9M), both 600M tokens; T++ b2t2 (101.9M), same data, as it is.
- Logs: `logs/diag/v13_sempty_rxk_{b2p2,cu2a,b2t2}_99d2951.jsonl`.

| cell, free / 20-way | b2p2 | cu2a | T++ b2t2 |
|---|---|---|---|
| explicit, L = 0, plain | 0.92 / 0.99 | 0.00 / 0.05 (see below) | 0.25 / 0.40 |
| explicit, 2k, plain | 0.55 / 0.77 | 0.48 / 0.76 | **0.01** / 0.02 (registered) |
| **explicit, 2k, contrast a = 4 (bar ≥ 0.70)** | **0.61** / 0.94 | **0.62** / 0.90 | 0.02 / 0.07 |
| explicit, 2k, adaptive a | 0.56 / 0.88 | 0.58 / 0.84 | 0.00 / 0.02 |
| explicit, 4k: plain / contrast / dilation + contrast | 0.16 / 0.42 / 0.53 | 0.18 / 0.46 / 0.67 | 0.00 (plain) |
| explicit, 8k: plain / contrast | 0.01 / 0.02 | 0.00 / 0.02 | **0.07** / 0.25 (plain, registered) |
| **explicit, 8k, dilation s = 8 + contrast (bar ≥ 0.40)** | **0.45** / 0.71 | **0.49** / 0.84 | – |
| CEILING (pack re-inserted at the query), 2k / 8k | 0.92 / 0.92 | 0.87 / 0.83 | 0.30 / 0.02 |
| FLOOR (no pack), any L | 0.00–0.01 | 0.00–0.02 | 0.00 |
| implicit ("K ->"), best decode at any L, incl. the ceiling | ≤ 0.04 | ≤ 0.05 | ≤ 0.07 |

Notes on the table:
- The T++ registered numbers are plain only. Its contrast / adaptive values were computed but not registered.
- T++ has no echo memory, so dilation does not apply to it.

**Mamba-2 rows** (14:53 IST, Lead's request: both baselines; `qllm311` venv with `mamba_ssm` 2.2.5, 4090;
`logs/diag/v13_sempty_rxk_{b2m2,mbs2b}_9d7ac4c.jsonl`). The probe code is identical to `99d2951`; the only diff is v7/data.py's
training-only ANSWER_W_SRC. Same docs and cells; plain is the registered decode for baselines. Free / 20-way:

| cell | **b2m2** (Mamba-2, b2 mix: same data as b2p2 / b2t2; 102.6M) | mbs2b (Mamba-2, aptitude recipe: no b2 docs) |
|---|---|---|
| explicit, L = 0 | 0.42 / 0.44 | 0.00 / 0.04 |
| explicit, 2k, plain | **0.03** / 0.13 (contrast 0.09 / 0.27) | 0.00 / 0.04 |
| explicit, 4k, plain | 0.01 / 0.03 | 0.00 / 0.02 |
| explicit, 8k, plain | **0.00** / 0.04 (contrast 0.00) | 0.00 / 0.03 |
| ceiling, 2k / 8k | 0.44 / 0.43 | 0.00 / 0.00 |
| floor, 2k / 8k | 0.00 / 0.00 | 0.00 / 0.00 |

mbs2b never trained on B2 rule docs: it is 0.00 even with the pack adjacent, so its row says nothing about memory. b2m2 is the
matched baseline.

**Separation row, at equal params and data** (600M tokens; b2p2 / cu2a 102.9M, b2t2 101.9M, b2m2 102.6M):

| model, explicit, free | 2k | 8k |
|---|---|---|
| PAM b2p2 / cu2a, plain | 0.55 / 0.48 | – |
| PAM b2p2 / cu2a, with query-time tools (2k contrast; 8k echo dilation + contrast) | 0.61 / 0.62 | 0.45 / 0.49 |
| T++, plain (as it is) | 0.01 | 0.07 |
| Mamba-2, plain (as it is) | 0.03 | 0.00 |

With the contrast, the baselines reach T++ 0.02 / 0.04 and Mamba-2 0.09 / 0.00. Dilation is PAM-specific.

**Post-hoc control** (not a bar cell; `v13_sempty/tmp/rxk_nl.py`, `logs/diag/v13_sempty_rxk_nl_*_99d2951.jsonl`): at L = 0 and in
the ceiling, the probe glued the row onto the last demo line (no newline). In the bar cells, the "\n\n" separator precedes the row.

| L = 0, free / 20-way | b2p2 | cu2a | T++ |
|---|---|---|---|
| pack + question (no row) | 1.00 / 1.00 | 1.00 / 1.00 | 0.69 / 0.79 |
| pack + "\n" + row + question | 0.91 / 0.99 | 0.02 / 0.09 | 0.17 / 0.29 |
| ceiling at 2k with the newline | 0.92 / 0.98 | 0.87 / 1.00 | 0.26 / 0.39 |

The newline changes nothing, so the glue is not the cause.

Generations (greedy; prompt tail `…27 -> 94 Record: lagtud | x: 7.\n\n7 ->`, gold ` 34`, affine doc 8800):
- b2p2 at L = 0: ` 34`.
- cu2a at L = 0: ` 32` (and on shift doc 8800, `j -> j`, copying the row's value).
- cu2a at 2k: ` 34`.

**Reading:**
- **The composition works when the package substitutes the value.**
  - The rule lives only in the state: the floor is ~0.
  - With the clock tools, it is applied to the fetched value 4× past the trained length (8k: 0.45 / 0.49, about half the ceiling).
  - At 2k, the contrast lifts the 20-way score to 0.90–0.94 but free only to 0.61–0.62. The misses are classified below (b2p2): the
    20-way score overstates fidelity, because its candidate set leaves out the confusable values.
  - Dilation + contrast beats contrast alone already at 4k (0.53 / 0.67 vs 0.42 / 0.46). At 2k, dilation was not registered.
- **Implicit composition fails for EVERY model, even with the pack in view:** no one takes v from the row when asked by key. The
  planner must substitute the value (amendment 2's assumption, now measured). Stated limit at 103M.
- **A fetched row between the rule and the query:**
  - barely disturbs b2p2 (1.00 → 0.91);
  - wrecks T++ (0.69 → 0.17);
  - hijacks cu2a at the start of a document: its fld training reads a record row there as "copy the field". After 2k of text, cu2a
    is fine (ceiling 0.87). A curriculum's format can capture a rule.
- **Versus T++ at equal params and data, as it is:**
  - 2k, inside T++'s window: PAM 0.55 plain / 0.61 contrast vs T++ 0.01;
  - 8k: PAM 0.45–0.49 (with the query-time tools) vs T++ 0.07;
  - even with the pack re-inserted, T++ reaches only 0.30.
  - This mirrors s8 (T++ D 0.134). Mamba-2 was not run on this cell.
- **Prefix note** (amendment 2): a reader conditioned on one prefix vector would likely drop the rule (Library's SR-0). Here the mind
  reads itself (warm + contrast). A reader organ, if ever built, is for records and formats, not for applying held rules.

**Consequence:**
- Not a Win as registered, because the 2k bar is missed.
- The 8k half shows the package claim (one read, applied far past the trained length) when the planner substitutes the value.
- Package rules:
  - explicit substitution is REQUIRED;
  - keep the 2k refresh + contrast default (A16);
  - for one read to 8k, dilation s = 8 + contrast.
- **Mind side** (diagnostic, b2p2, the same 2k explicit cell; `v13_sempty/tmp/rxk_miss.py`): free misses are almost never the wrong
  TYPE.

| 2k, b2p2 (of 50 per task) | right | right type, not a candidate | copies the fetched v | wrong candidate |
|---|---|---|---|---|
| affine, plain / contrast | 38 / 32 | 11 / 18 | 0 / 0 | 1 / 0 |
| shift, plain / contrast | 17 / 29 | 19 / 18 | 12 / 2 | 2 / 1 |

  - The contrast removes the COPY failure (shift 12 → 2): v is in both streams, so it cancels. It also costs some affine docs.
  - What is left is the rule applied imprecisely (e.g. gold ` 34`, wrote ` 22` / ` 82`). That is memory fidelity at 2k, not answer
    format, so a type mask in the package would not help.
  - (Corrected 14:20 IST: I first wrote "an expression gap, cu3's business".)
- **Follow-up for Library's story check** (14:24 IST; `v13_sempty/tmp/pell_2x1.py`, b2p2, CPU). The rule 2x + 1 is a TRAINED affine
  value (a ∈ 1–3, b ∈ 0–20 minus {7, 13, 17}). Demos in the trained format, then held-out web text, then "4 ->"; 20 random demo sets;
  free:
  - L = 0: 1.00 with 3 or 8 demos;
  - 2k, 3 demos: 0.00;
  - 2k, 8 demos: 0.45 plain, 0.30 with contrast. The contrast pulls " 41" in 12 / 20, though it is a demo answer in only 5 of those
    sets: an unexplained attractor.

  So few demos carry a rule at once but not across a 2k gap, and the contrast must stay gated (A16's refresh is the safe lever).

## w1r VERDICT (14:28 IST, 2026-11-01): REPLICATES on every pre-registered bar. W1 (window 128 in blocks 5 / 9 / 13, +4.15M = +4 %) raises in-context rule inference from spread examples (s8) on both seeds at holdout parity, including in the cell beyond the windows' reach. A REACH unit, not a reader

**Arms:**
- `w1r_w0_508aaff` (102.88M) and `w1r_w128_508aaff` (107.04M), as in w1s (120M stage A + 100M stage B), SEED=44.
- 4090, 13:04–14:24 IST. Logs: `logs/v13_sempty_w1r_*`, `logs/learning_probes/v13_sempty_w1r_*`, `logs/diag/v13_sempty_{skillpack,clk2_echo}_w1r_*`.

| bar (w128 − w0) | seed 43 (w1s) | **seed 44 (w1r)** | bar |
|---|---|---|---|
| s8 D (mean over 2k / 4k, scored) | 0.559 → 0.730 (+0.171) | **0.522 → 0.844 (+0.323)** | ≥ +0.08 |
| clk2 8k (echo s = 8 + contrast), 20-way | 0.31 → 0.53 (+0.22) | **0.33 → 0.45 (+0.12)** | ≥ +0.10 |
| holdout | 3.7454 → 3.7510 (+0.006) | **3.7456 → 3.7480 (+0.002)** | ≤ +0.02 |

**Reach check (how far back W1 can see):**
- Three stacked windows of 128 reach at most 3 × 127 = 381 tokens back.
- In s8, the last example ends 254 tokens before the query at 2k, which is inside two windows. A single example fixes a letter shift,
  so **the 2k cell is confounded.** At 4k it ends 510 tokens back, beyond all three windows.
- The skill-pack and clk2 cells read the pack ≥ 2k back.

| cell beyond the windows' reach, w0 → w128 | seed 43 | seed 44 |
|---|---|---|
| s8 at 4k, scored (shift + affine mean) | 0.435 → 0.617 | 0.416 → 0.771 |
| s8 at 4k, free | 0.194 → 0.361 | 0.155 → 0.411 |
| s8 at 4k, shift only, scored | 0.17 → 0.453 | 0.185 → 0.74 |
| skill pack plain, 20-way, 2k / 4k | 0.61 / 0.105 → 0.735 / 0.225 | 0.63 / 0.14 → 0.67 / 0.22 |
| skill pack + contrast, free, 2k / 4k | 0.87 / 0.335 → 0.74 / 0.325 | 0.79 / 0.23 → 0.73 / 0.36 |
| clk2 8k, free | 0.12 → 0.21 | 0.18 → 0.16 |

Other readouts:
- Text-arm tool read (seed 44): 0.5125 → 0.5625; pooled over both seeds, +0.03. Still no reader gain.
- WikiText val loss (seed 44): 4.5392 → 4.5191.

**Reading:**
- **Real, and where it lives:** the gain holds beyond the windows' reach on both seeds. It is largest when the rule must be INFERRED
  from examples spread through text (s8 4k +0.18 / +0.36, shift most), and small for a contiguous pack read once (skill pack plain
  +0.04 to +0.13).
- **With the contrast:** W1 is lower at 2k free on both seeds (−0.13 / −0.06); mixed at 4k.
- **The decays are unchanged** (w1s half-lives).
- **A write-side guess:** each spread example "x -> y" sits inside a window. W1 can compute the pair's RELATION locally (a letter
  offset) and hand the notebooks a cleaner feature to accumulate, instead of the recurrence having to bind x to y itself.
  Testable: linear decodability of k from the state after the examples, w0 vs w128.
- **Prior work:** Samba (Ren et al. 2024: Mamba + sliding-window attention, better length extrapolation and recall), Griffin / Hawk
  (De et al. 2024), Based (Arora et al. 2024). The hybrid is KNOWN, so there is no mechanism claim.
- **What is measured here:** in-context rule inference from examples 500 tokens apart, beyond the windows' reach, gains at +4 %
  params with LM parity.
- **Inference memory:** the window KV is fixed at 3 blocks × 128 × (k, v) × 588 × 2 bytes ≈ 0.9 MB (bf16), so the state is
  1.15 → ~2.05 MB. Still O(1) in context length.

**Consequence (pre-registered):**
- W1 goes to the USER as a 600M decision, as a reach unit, never as a reader. Its code stays.
- Proposed 600M run: b2p2's exact recipe + `SWA_LAYERS=5,9,13 SWA_WINDOW=128`, against b2p1 / 2 / 3 as the seed spread
  (s8 D 0.645 / 0.691 / 0.741; holdout 3.4777 / 3.4800 / 3.4775).
- Bars:
  - s8 D ≥ 0.80 (the best b2p seed + 0.06);
  - s8 at 4k reported separately;
  - holdout ≤ 3.4900;
  - clk2 8k 20-way ≥ b2p2's 0.74.
- Cost: about 3 h on the 4090, or on the 6000 after bud.
- The 3B body (r3c, 16nb) is a separate question.

## w1w: window-size sweep, W = 512 and 1024 (pre-registration, 2026-11-01 15:06 IST; Lead GO; the user's question "why not a bit bigger?")

**Why:**
- The user asks whether a 128-token window is enough for long chats (128k–300k), and why not a bigger fixed window.
- Far needles are the LIBRARY's job: no fixed window recalls a needle 100k back. Lead runs the 300k package rows today, and Library
  runs them after this sweep.
- A bigger window is cheap: the KV is about 7 KB per window token, so W = 1024 → ~7.2 MB, still O(1).
- **Risk: a LAZY STATE.** If W nears the 2048 training length, the state has no work during training, and its rule and reach skills
  (which live in the state) may weaken.
- **A sign of it already at W = 128** (the t40 needle probe, both seeds pooled; needle at the start of the text):
  - 4k: W0 0.495 vs W128 0.44; 8k: 0.305 vs 0.22;
  - nearer needles improve (≤ 2k, all positions: up to +0.13).

**Prior work:** Griffin (local attention 1024 + RG-LRU), Samba (SWA 2048 + Mamba). Window size is engineering; no claim.

**Arms:**
- `w1w_w512` and `w1w_w1024`: the w1s recipe exactly (c1m2a stage A, then 100M on b2p2's mix), SEED=43, SWA_LAYERS=5,9,13.
- Reference arms: W0 and W128 exist on seeds 43 and 44 (w1s / w1r).
- Code: model.py and config.py are identical to `508aaff`. The window kernel is exact at W = 512 / 1024 (chunked vs full mask
  3e-8; nothing leaks past W).

**Stacked reach:** three windows reach 3(W − 1) tokens: 381 / 1533 / 3069 for W = 128 / 512 / 1024. A "beyond the window" cell
must be farther back than that.

**Readouts** (new arms: all; old arms: only the new skill-pack 8k cell):
- holdout;
- t40 needle recall (the screen's own probe; contexts 128–8192, needle positions 0 / 0.5 / 1);
- skill pack at 0 / 2k / 4k / 8k, plain + contrast (4k and 8k are beyond every W ≤ 1024);
- clk2 8k (echo dilation s = 8 + contrast);
- s8 at 0 / 2k / 4k, 400 trials;
- the text-arm tool read (syn / nat).

**Not used as a cell:** s8 spread to 2k with a 3.2k tail gap before the query (`--spread_tail`, new). It is at the floor at this
scale: 40 trials, scored 0.10 (W0) / 0.15 (W128), chance 0.125.

**Lazy-state GUARD** (Lead), per new W. Every one must hold, measured against W128's two-seed mean:
- (g1) skill pack 4k + contrast, 20-way ≥ 0.5825 (W128 mean 0.6325 − 0.05);
- (g2) clk2 8k, 20-way ≥ 0.44 (0.49 − 0.05);
- (g3) t40 far needles (position 0, contexts 4096 + 8192, mean) ≥ 0.28 (0.33 − 0.05). Also reported against W0 (0.40).
- (g4) holdout ≤ 3.7655 (W0 mean 3.7455 + 0.02).

**GAIN and the pick:**
- gain = the mean of (s8 D at 2k / 4k scored; skill pack 4k + contrast 20-way; clk2 8k 20-way; t40 near needles at contexts
  512–2048, all positions), minus W128's mean of the same.
- A W ≥ 512 is picked only if it passes every guard AND gain ≥ +0.03. If two W are within 0.03 of each other, the smaller wins.
  Otherwise W = 128 stays.
- New arms are a single seed. If the deciding margin is < 0.05, a seed-44 replicate of the top W runs before any 600M run.
- **SEQ 4096** (more work for the state beyond W) only if W = 1024 wins.

**Consequence:** the user's 600M decision becomes "which W", with these numbers.

## kb1: KNOWN small-model benchmarks, our bodies and the baselines as they are (pre-registration, 2026-11-01 15:38 IST; the user's standing rule via Lead: "participate in known benchmarks; intelligence first, then the infrastructure saving")

**What:** the standard zero-shot suite small LMs report, run through **lm-eval-harness 0.4.9.1** (its own task prompts, scoring
and stderr).
- Our bodies are wrapped as a harness `LM` (`scripts/lm_eval_bodies.py`; log-likelihood requests only).
- References run through the SAME harness with its stock `hf` model.

**Tasks** (full sets, zero-shot; both acc and acc_norm reported):

| task | headline metric |
|---|---|
| hellaswag | acc_norm |
| arc_easy | acc |
| arc_challenge | acc_norm |
| piqa | acc_norm |
| winogrande | acc |
| openbookqa | acc_norm |
| lambada_openai | acc, plus ppl |

**Models, as they are** (no tuning on any benchmark, one run each):
- **600M tier** (same data, 600M tokens): PAM b2p2 (102.9M), PAM cu2a (102.9M; +2 % fld / vfy), T++ b2t2 (101.9M),
  Mamba-2 b2m2 (102.6M).
- **3B tier** (3.0B tokens): PAM r3c (`r3cb_main_38e3c29`, 103.2M), T++ (`mix3b_transformer_answ100_e072a8f`), Mamba-2
  (`mix3b_mamba2_answ100_b36e3a5`).
- **Known references, same harness** (their tokens stated beside every number):
  - SmolLM2-135M (2T tokens; card: HellaSwag 42.1, ARC 43.9, PIQA 68.4 via lighteval, so its card numbers are NOT the same
    harness);
  - Pythia-160M (300B tokens, the Pile);
  - GPT-2 124M (WebText).

**Protocol** (fixed now):
- Requests are encoded exactly as the harness's HF path does it. Trailing context whitespace moves to the continuation; the
  continuation is taken as `whole[len(context):]`; an empty context becomes EOT; requests are left-truncated to 2048. No prefix
  is added for any model.
- bf16 autocast, batch sorted by length, right padding (all bodies are causal).
- One run per model. The harness stderr is reported. No prompt, metric or task is changed after the first number.

**Prediction** (written first):
- Every 100M body here is far below SmolLM2 (3,300× more data). Expected HellaSwag ~0.28–0.31 and ARC-e ~0.40–0.45.
- What matters is the EQUAL-DATA rows:
  - 600M: PAM within ±0.01 of T++ on the suite mean (b2p2 holdout 3.48 vs b2t2 3.56), and Mamba-2 about the same;
  - 3B: PAM slightly behind Mamba-2 (holdout gap 0.055) and about level with T++.
- LAMBADA (a long-range last word) is where a recall-strong body may show.

**Where it runs:** the 4090 after w1w (~16:45 IST), as Lead orders it beside Library's 300k run (each eval is a 100M model,
~2 GB). The r3c checkpoint is copied from the 6000.

**Reading rule:** this becomes the "intelligence" line of the progress bar. An equal-data row is the claim; the reference rows
are context, never a claim. No 128k+ length claims from this suite.

**kb1 amendment (15:45 IST, before any number; Lead):**
- Order: right after w1w, beside Library's A25. Each model is gated on free VRAM ≥ peak + 3 GB (7 GB).
- Added: the 100M window arms on the same suite, so the window's language cost shows on known tasks, not only on holdout:
  - w1s / w1r W0 and W128 (seeds 43 / 44, 220M tokens);
  - w1w W512 / W1024 (seed 43).
- Tokens beside every row: 220M (100M-stage arms) / 600M / 3B / SmolLM2 2T / Pythia 300B / GPT-2 WebText (~10B).

**kb1 layout (the user, 16:30 IST via Lead; before any number):**
- Comparisons go by TOKENS TRAINED ON, not parameter count alone. The benchmark runs are diagnostics for what to improve.
- Blocks:
  - the 600M tier (b2p2, cu2a, b2t2, b2m2);
  - the 3B tier (r3c, e072a8f, b36e3a5);
  - the 220M window arms;
  - a separate REFERENCE block (SmolLM2-135M 2T, Pythia-160M 300B, GPT-2 ~10B), labelled "reference, different data and tokens,
    no claim".
- Tokens and data beside every row. Claims and deltas only within a block. The same layout for the write-up
  (`v13_sempty/tmp/kb1_table.py`).

## w1w VERDICT (16:43 IST, 2026-11-01): both bigger windows FAIL the lazy-state guard; W = 128 stays. A bigger window makes the state LAZIER for reach and helps almost nothing it was hoped to help

**Arms:**
- `w1w_w512_f68ab64` and `w1w_w1024_f68ab64` (107.04M each; the window size adds no params), seed 43, the w1s recipe, 220M tokens.
- 4090, 15:07–16:35 IST; peak 14.8 GB. Score: `v13_sempty/tmp/w1w_score.py`. W0 and W128 are two-seed means.

| readout | W0 | **W128** | W512 | W1024 | guard |
|---|---|---|---|---|---|
| holdout | 3.7455 | 3.7495 | 3.7502 | **3.7430** | g4 ≤ 3.7655: all pass |
| s8 D (2k / 4k, scored) | 0.541 | **0.787** | 0.679 | 0.676 | (gain) |
| skill pack 4k + contrast, 20-way | 0.518 | **0.633** | 0.450 | 0.555 | g1 ≥ 0.5825: 512 FAIL, 1024 FAIL |
| skill pack 8k + contrast, 20-way | 0.050 | **0.143** | 0.070 | 0.045 | (reported) |
| clk2 8k (dilation + contrast), 20-way | 0.320 | **0.490** | 0.320 | 0.430 | g2 ≥ 0.44: 512 FAIL, 1024 FAIL |
| t40 far needles (position 0, 4k + 8k) | 0.398 | 0.329 | **0.471** | 0.275 | g3 ≥ 0.28: 512 pass, 1024 FAIL |
| t40 near needles (contexts 512–2048) | 0.787 | 0.865 | 0.873 | 0.888 | (gain) |
| text-arm tool read (free) | 0.556 | 0.588 | 0.525 | 0.562 | (reported) |
| **gain vs W128** (pre-registered mean) | – | 0 | **−0.113** | **−0.056** | pick needs ≥ +0.03 |

s8 by task (scored, 2k / 4k):
- shift: W512 0.713 / 0.343, W1024 0.652 / 0.320, vs W128 0.762–0.922 / 0.453–0.74;
- affine: W512 0.907 / 0.755, W1024 0.932 / 0.80.

At 2k the examples are 254 tokens apart, so W512 sees 2 and W1024 sees 4 of them at once. Shift is still worse than W128.

**Reading:**
- **A lazy state, as feared:** a bigger window lets the body lean on local attention.
  - The state's one-read reach (pack 4k + contrast, clk2 8k) falls back toward W0.
  - Rule inference from spread examples (s8) keeps only part of W128's gain, even where the bigger window SEES several examples
    together.
  - LM gains a little at W1024 (−0.0025 vs W0, within seed noise).
- **Near needles barely move** (0.865 → 0.873 / 0.888), although W1024 can attend to every needle ≤ 1024 back. Far needles are noisy
  (W512 0.471, W1024 0.275).
- **So W128 is not a compromise; it is the best tested.** A short window does the local work (copying the row, the x → y pair) and
  leaves the long range to the state. Prior work is consistent: Griffin / Samba use 1–2k windows for LM quality, not for the
  state's rule reach, which they do not measure.
- **One seed per new arm**, but both fail by margins larger than W128's own seed spread on g1 (0.62–0.645) and g2 (0.45–0.53), and
  the gains are −0.11 / −0.06. Per the pre-registration, no replicate.
- **For the user's question:** a bigger fixed window does not buy long-chat recall. Far needles stay the library's job. The window's
  value is rule inference, and that is best at 128.
- **Per-body slope:** no prior tier for this probe set (all 220M).

**Consequence:**
- The 600M W1 decision stays as proposed, at W = 128. That run is the user's call.
- SEQ 4096 is not needed (W1024 did not win).

## w1m: W1 (W = 128) at 600M on the M2 body, b2p2's recipe (pre-registration, 2026-11-01 16:57 IST; Lead APPROVED, the user handed Lead the call)

**Why:** W1 replicated as a REACH unit at 100M (w1r), and W = 128 is the best window tested (w1w). Does it hold at 600M, on b2p2's
recipe, against the three-seed b2p spread? The verdict decides PAM's body for the matched 1.5B tier. The bar order is the user's:
INTELLIGENCE first (rules, reach), then the infrastructure saving.

**Arm:** `w1m_w128`, an exact copy of b2p2's recipe (`38e3c29` log header) + `SWA_LAYERS=5,9,13 SWA_WINDOW=128`:
- `c1m2a_pre_7ae6f1f` stage A; stage B 480,006,144 tokens; LR 1.2e-3 cosine, warmup 100; batch 18 × 2048;
- mix dclm / fineweb / smoltalk2_mid / recall / b2 = 0.43 / 0.40 / 0.10 / 0.04 / 0.03; answers ×100; seed 43;
- ANSWER_IN_WINDOW off, as in b2p1 / 2 / 3;
- on the 4090, GRAD_ACCUM=3 (exact accumulation, verified equal to the full batch within 2e-5);
- 107.0M vs 102.9M (+4 %); 600M tokens in total.

**Control:** b2p1 / b2p2 / b2p3 as they are (102.9M, the same recipe, three seeds).

| readout | b2p1 / b2p2 / b2p3 | bar for w1m_w128 |
|---|---|---|
| s8 D (2k / 4k, scored, 400 trials, seeds 7000–7399) | 0.645 / 0.691 / 0.741 | **≥ 0.80** (the best seed + 0.06) |
| holdout | 3.4777 / 3.4800 / 3.4775 | **≤ 3.4900** |
| clk2 8k (echo s = 8 + contrast), 20-way | b2p1 0.81, b2p2 0.74 | **≥ 0.74** |
| t40 far needles (position 0, contexts 4096 + 8192) | 0.521 / 0.263 / 0.504 (mean 0.429) | **≥ 0.379** (W0 mean − 0.05) |

**Also reported:**
- s8 at 4k alone (beyond W128's 381-token reach; b2p 4k means 0.525 / 0.606 / 0.623);
- skill pack 0 / 2k / 4k / 8k, plain + contrast;
- the text-arm tool read;
- kb1 (the 600M tier);
- t40 near needles;
- WikiText.

**Pass:** all four bars. Then W128 is a candidate for PAM's 1.5B-tier body (see Lead's tier decision and the 16nb question).
**Fail:** W1 stays a 100M finding, and the 1.5B tier runs the plain body.

**Prior work:** Griffin / Samba (local attention + recurrence); no mechanism claim. **Per-body slope:** this is the 600M point;
the 220M point is w1s / w1r (s8 D W128 0.787, W0 0.541).

**Order on the 4090:** after kb1 → Library's A25 → Lead's BABILong pass (Lead's GPU order).

## bud VERDICT (17:17 IST, 2026-11-01): bud2 FAILS on skills (two-hop 0.35 < 0.45; one-read reach at 2k 0.48 → 0.175). Lower answer weights save LM but do NOT teach two-hop or reach. The mix cost at 600M on the 16nb body is +0.036

**Arms:** `bud0` / `bud1` / `bud2` `_main_e96588b`. r3c's 16nb body + echo, from `r3ca_pre_38e3c29`; 480M stage B; seed 42; 103.2M;
6000. Logs: `logs/v13_sempty_bud*_e96588b*`, `logs/learning_probes/v13_sempty_bud*`, `logs/diag/v13_sempty_{curprobe,skillpack}_bud*`.

| readout | bud0 (r3c's mix) | bud1 (r3p's mix, ×100) | bud2 (×20 synthetic) | bud2 bar |
|---|---|---|---|---|
| holdout | 3.5132 | 3.5487 (+0.0355) | 3.5266 (+0.0134) | ≤ 3.5332 ✓ |
| WikiText val loss | 4.2997 | 4.3335 | 4.3105 | – |
| B2 `_new` mean (shift / step / affine) | 0.03 | 0.973 (0.935 / 0.988 / 0.996) | 0.933 (0.965 / 0.838 / 0.995) | ≥ 0.923 ✓ |
| label_diag mean | 0.555 | 0.927 | 0.923 | ≥ 0.877 ✓ |
| fsq abstain (absent keys) | 0.00 | 1.00 | 0.99 | ≥ 0.95 ✓ |
| **two-hop (cur probe), correct / free** | 0.135 / 0.00 | 0.50 / 0.25 | **0.35** / 0.25 | ≥ 0.45 **✗** |
| skill pack 0 / 2k (reported) | 0.045 / 0.035 | 1.00 / 0.48 | 1.00 / **0.175** | – |
| verify (cur probe, reported) | 0.33 | 0.998 | 0.757 | – |

**Reading:**
- **The mix cost at 600M on 16nb:** bud1 − bud0 = **+0.036**, against r3p's 3B-mix gap of 0.11 at 1.44B. The cost grows with
  scale and tokens on this body, or r3p's 3B run amplified it.
- **×20 answers keep the easy skills** (B2 at L = 0, labels, abstain), **but lose two-hop (−0.15) and one-read reach at 2k (0.48 →
  0.175).** The skills that need the trace to be held (reach) or composed (two-hop) need their weight. That is the
  pre-registered failure lesson: lower weights don't teach.
- So r3p' and any tier carry FEWER synthetic tokens at full weight, not lighter weights.
- **Per-body slope:** PAM 16nb only (one tier). bud0 vs r3c at 3B: 3.5132 → 3.2191 holdout with the same mix (r3c's).

## t15: the MATCHED 1.5B-token tier, three bodies on the 600M tier's mix (pre-registration, 2026-11-01 17:32 IST; Lead's decision, the user handed Lead the call)

**Why** (the user, 16:40 IST): "if we train on more data will it improve, which version improves more; keep an eye on this."
- The 3B tier has no rule docs: at L = 0 all three 3B bodies are at chance on B2 (scored 0.055–0.09, chance 0.125). It gives no
  rules point.
- So this tier keeps the 600M tier's mix EXACTLY and changes ONLY tokens. Each body's slope 600M → 1.5B is then clean.
- The bar order is the user's: intelligence first (rules at distance, reach), then the infrastructure saving.

**Prior work:** matched-data SSM vs Transformer comparisons (Mamba-2, Dao & Gu 2024; Waleffe et al. 2024, "An Empirical Study of
Mamba-based Language Models"). This tier is a measurement; claims only by the bars.

**Recipe, all arms:**
- dclm / fineweb / smoltalk2_mid / recall / b2 = 0.43 / 0.40 / 0.10 / 0.04 / 0.03; answers ×100; ANSWER_IN_WINDOW off; seed 43;
- LR 1.2e-3 cosine, warmup 100; batch 18 × 2048;
- stage B 1,380,003,840 tokens (37,435 steps) + each body's own 120M stage A = **1.50B tokens**;
- val every 4,883 steps; save every 200; 6000; chain `v13_sempty/tmp/t15_chain.sh`.

**Arms:**
- **t15tpp:** T++ (101.9M), from `tfc1a_pre_b788731`, as b2t2.
- **t15mamba:** Mamba-2 (102.6M), from `b2ma_pre_38e3c29`, as b2m2.
- **t15pam:** PAM 16nb + echo (103.2M, r3c's body), from `r3ca_pre_38e3c29`, GRAD_ACCUM 2. It runs LAST, with W128 only by the
  decision rule below.

**Bars:**
- **Rules at distance:** s8 D (2k / 4k, scored, 400 trials) of t15pam ≥ max(t15tpp, t15mamba) + 0.10. s8 at 4k is also reported
  alone.
- **LM:** holdout of t15pam ≤ t15mamba + 0.02.
- **Reported:**
  - one-read reach (skill pack 0–8k plain + contrast; clk2 8k);
  - t40 + PG-19;
  - the text-arm tool read;
  - kb1 (run on the 4090);
  - BABILong-package (Lead's runner);
  - **the per-body slope 600M → 1.5B for every readout.**

**Decision rule for t15pam's body** (Lead):
- w1m (M2, 4090) AND w1n (16nb, below) both pass → 16nb + W128;
- only w1m passes → 16nb plain (W noted as an M2 finding);
- neither → 16nb plain.

Mind writes `checkpoints_v13_sempty/t15_pam_body.txt` after both verdicts; the chain waits for it.

**Consequence:** r3p' (3B) waits for this tier's verdict. Budget rule (bud): carry FEWER synthetic tokens at full weight, not lighter
weights.

## w1n: W1 (W = 128) on the 16nb body at 600M, on the tier's mix (pre-registration, 2026-11-01 17:32 IST; Lead GO)

**Why:** W128 was found and tested only on the M2 body. The tier's PAM body is 16nb (windows in blocks 5 / 9 / 13 sit beside
notebooks, not in place of reads). The plain arm is also PAM's 16nb 600M point on the tier's mix, for the clean slope.

**Arms** (6000, the t15 mix above, `r3ca_pre_38e3c29`, stage B 480,006,144 tokens = the 600M tier, seed 43, GRAD_ACCUM 2):
- **w1n_n0:** 16nb + echo;
- **w1n_n128:** + SWA_LAYERS=5,9,13, SWA_WINDOW=128 (+4.15M).

**Bars** (n128 vs n0):
- s8 D ≥ n0 + 0.08;
- clk2 8k 20-way ≥ n0;
- t40 far needles (position 0, contexts 4096 + 8192) ≥ n0 − 0.05;
- holdout ≤ n0 + 0.01.

**Reported:** s8 at 4k; skill pack 0–8k; the tool read; near needles; the per-body slope (w1s / w1r 220M → 600M).
**Prior work:** Griffin / Samba; no mechanism claim.

## kb1 RESULT (17:54 IST, 2026-11-01): on the known small-LM suite PAM is LEVEL with Transformer++ and Mamba-2 at 600M tokens, level with T++ and 0.012 behind Mamba-2 at 3B, and gains the least from 600M → 3B. The window costs nothing on known tasks

**Setup:**
- lm-eval-harness 0.4.9.1, zero-shot, full sets; `scripts/lm_eval_bodies.py`; bf16 on the 4090; one run each.
- Logs: `logs/kb1/v13_sempty_kb1_*_2577940.json`, chain `logs/v13_sempty_kb1_chain_2577940_*.log`.
- Layout (the user): tiers by tokens trained on; claims only within a tier; references are context, no claim.
- MC mean = the mean of the six headline metrics.

| tier | PAM | T++ | Mamba-2 |
|---|---|---|---|
| 600M tokens: MC mean | b2p2 **0.391**, cu2a 0.388 | b2t2 0.388 | b2m2 0.392 |
| 600M tokens: LAMBADA acc / ppl | 0.146 / 578 | 0.148 / 1299 | 0.132 / 945 |
| 3B tokens: MC mean | r3c **0.403** | e072a8f 0.406 | b36e3a5 **0.415** |
| 3B tokens: LAMBADA acc / ppl | 0.179 / 376 | 0.196 / 404 | 0.176 / 305 |
| **slope 600M → 3B, MC mean** | **+0.012** | +0.018 | +0.023 |
| slope 600M → 3B, LAMBADA acc | +0.033 | +0.048 | +0.044 |

- **220M window arms** (MC mean): W0 0.377 / 0.383 (seeds 43 / 44); W128 0.380 / 0.384; W512 0.381; W1024 0.381. No language cost on
  known tasks.
- **References** (no claim): SmolLM2-135M (2T tokens) 0.487, LAMBADA 0.428; GPT-2 124M (~10B) 0.397, LAMBADA 0.307; Pythia-160M
  (300B) 0.396, LAMBADA 0.354.
  - Pythia is reported in fp32. In bf16 it degrades (LAMBADA 0.117 / ppl 105, ARC-e 0.387, PIQA 0.576); fp32 matches its published
    numbers.
  - **Our bodies are NOT dtype-sensitive:** b2p2 in fp32 gives LAMBADA 0.1475 / 578.6 (bf16 0.146 / 578), ARC-e 0.454 (0.454),
    PIQA 0.578 (0.578). T++ and Mamba-2 likewise (`logs/kb1/v13_sempty_kb1fp32_*_ccf187c.json`).
  - GPT-2 (0.307 vs published 0.325) and SmolLM2 (PIQA 0.684 = its card) were fine in bf16.

**Reading** (diagnostics for what to improve, the user's framing):
- **On general knowledge / commonsense, the body does not matter much at this size and data.** All three are within ±0.004 at 600M
  (per-task stderr 0.005–0.02).
- **At 3B, PAM is level with T++ (−0.003) and behind Mamba-2 (−0.012),** the same order as holdout (3.219 / 3.206 / 3.164).
- **PAM gains the LEAST from data on this suite** (+0.012 vs +0.018 / +0.023).
  - Caveat: PAM's body changed between tiers (M2 at 600M, 16nb at 3B), and the 3B mix had no b2 docs. The matched t15 tier
    (running) gives the clean slope.
- **PAM's LAMBADA perplexity is the LOWEST of the three at 600M** (578 vs 945 / 1299) and middle at 3B. The far-context last word
  is where its state helps; its accuracy is level.
- **PAM's edge is NOT here; it is in rules held in state** (s8: PAM 0.65–0.74 vs T++ 0.13 / Mamba-2 0.10; R × K: 0.61 / 0.45 vs
  0.01–0.07). What to improve for the known suite: plain LM quality (the 0.04–0.06 holdout gap to Mamba-2 at 3B) and tokens.
  SmolLM2's 2T tokens are 3,300× ours.
- **The window:** no cost here, and no gain, as expected for short tasks.

## w1m VERDICT (22:14 IST, 2026-11-01): FAIL as pre-registered on s8 (0.772 < 0.80, by 0.028). Yet W128 at 600M is better than EVERY b2p seed on holdout, s8, s8 4k and clk2. By Lead's rule the 1.5B PAM runs the plain 16nb body

**Arm:** `w1m_w128_a913e01` (107.0M; b2p2's exact recipe + SWA 5 / 9 / 13 at 128; 600M tokens; 4090, 17:46–22:11 IST).

| readout | b2p1 / b2p2 / b2p3 | **w1m W128** | bar |
|---|---|---|---|
| s8 D (2k / 4k, scored) | 0.645 / 0.691 / 0.741 | **0.772** | ≥ 0.80 **✗** |
| s8 at 4k (beyond the window) | 0.525 / 0.606 / 0.623 | **0.647** | (reported) |
| holdout | 3.4777 / 3.4800 / 3.4775 | **3.4708** | ≤ 3.4900 ✓ |
| clk2 8k (dilation + contrast), 20-way | b2p1 0.81, b2p2 0.74 | **0.830** | ≥ 0.74 ✓ |
| t40 far needles | 0.521 / 0.263 / 0.504 | 0.446 | ≥ 0.379 ✓ |
| skill pack 20-way at 2k / 4k, plain | b2p2 0.86 / 0.355 | 0.905 / 0.275 | (reported) |
| skill pack 4k + contrast, 20-way | b2p2 0.78 | 0.87 | (reported) |
| text-arm tool free (syn / nat) | b2p2 0.475 / 0.40 | 0.45 / 0.30 | (reported) |
| WikiText val loss | – | 4.2142 | – |

**Reading:**
- **Gains:** above all three seeds on holdout (−0.007 vs the best), s8 (+0.03 vs the best, +0.08 vs the mean), s8 4k (+0.024) and
  clk2 (+0.02). The bar was the best seed + 0.06; it missed by 0.028. One seed.
- **Per-body slope (W128 − W0, M2 body):**
  - 220M: s8 D +0.246 (two seeds);
  - 600M: +0.081 vs the b2p mean;
  - the window's advantage SHRINKS with data, as more tokens teach the plain body the same inference.
  - On holdout it turns from a small cost (220M +0.004) to a small gain (600M −0.008 vs the mean).
- **Not a reader:** tool 0.375 vs b2p2 0.4375.

**Consequence** (Lead's decision rule): w1m fails, so t15pam = **16nb plain**. `checkpoints_v13_sempty/t15_pam_body.txt` = `plain`
on the 6000. w1n still runs: W128 on 16nb is a finding either way, and it costs nothing (it runs while T++ trains).
- W128 remains a candidate body unit with a modest, shrinking gain. A second 600M seed would settle the 0.028 miss; that is the
  user's / Lead's call, not launched.

**w1m amendment (22:15 IST; Lead GO, board e11c35c): w1m2 = the same arm at SEED=44** (4090, beside Lead's BABILong pass).
The bars are unchanged and are judged on the SEED MEAN of w1m + w1m2 (s8 D ≥ 0.80, holdout ≤ 3.4900, clk2 8k ≥ 0.74, far
needles ≥ 0.379). This decides the 600M release body (W128 vs b2p2). t15pam stays 16nb plain.

## w1m2 / w1m two-seed VERDICT (02:31 IST, 2026-11-02): FAIL as pre-registered on s8 (seed mean 0.784 < 0.80, by 0.016); every other bar passes. W128 adds a steady +0.09 on rules from spread examples and +0.07 on one-read reach at 600M, at LM parity: a real, modest gain, short of the bar. The 600M release body stays b2p2 (no window)

| readout | b2p1 / 2 / 3 (mean) | w1m (seed 43) | w1m2 (seed 44) | **two-seed mean** | bar |
|---|---|---|---|---|---|
| s8 D | 0.645 / 0.691 / 0.741 (0.692) | 0.772 | 0.796 | **0.784** | ≥ 0.80 ✗ |
| s8 at 4k | 0.525 / 0.606 / 0.623 (0.585) | 0.647 | 0.704 | 0.676 | – |
| holdout | 3.4777 / 3.4800 / 3.4775 (3.4784) | 3.4708 | 3.4810 | 3.4759 | ≤ 3.4900 ✓ |
| clk2 8k, 20-way | b2p1 0.81 / b2p2 0.74 | 0.830 | 0.860 | 0.845 | ≥ 0.74 ✓ |
| t40 far needles | 0.521 / 0.263 / 0.504 (0.429) | 0.446 | 0.396 | 0.421 | ≥ 0.379 ✓ |

**Reading:**
- Both seeds sit above every b2p seed on s8 and clk2. The mean gain is +0.092 on s8 D and +0.091 on s8 4k (beyond the window),
  +0.07 on clk2, with holdout at parity (−0.003). The bar (the best seed + 0.06) asked for more than the gain delivers at 600M.
- **Per-body slope of the window's s8 gain:** +0.246 at 220M, +0.092 at 600M. It shrinks with data. A W128 3B run is not
  justified by this.
- **Consequence:** the 600M release body stays b2p2 (pre-registered rule). t15pam = 16nb plain, unchanged. W1 remains in the code
  as a measured, modest reach unit (index row: "+0.09 rules at 600M, shrinking with data"). w1n (16nb) is still a finding.

## w1n VERDICT (03:50 IST, 2026-11-02): FAIL. On the 16nb body the window HURTS the rule / reach skills (s8 −0.10, clk2 −0.14, far needles −0.06) while helping LM (−0.025). W1's rule gain is M2-specific

**Arms:** `w1n_n0` / `w1n_n128_c06b939` (16nb + echo, the t15 mix, 600M tokens, seed 43; 103.2M / 107.4M; 6000).

| readout | n0 (16nb plain) | n128 (16nb + W128) | bar |
|---|---|---|---|
| s8 D | 0.768 | 0.666 | ≥ n0 + 0.08 ✗ |
| s8 at 4k | 0.626 | 0.556 | – |
| clk2 8k, 20-way | 0.39 | 0.25 | ≥ n0 ✗ |
| t40 far needles | 0.338 | 0.279 | ≥ n0 − 0.05 ✗ |
| holdout | 3.5662 | **3.5409** | ≤ n0 + 0.01 ✓ |
| skill pack 2k plain / + contrast; 4k + contrast | 0.76 / 0.995; 0.70 | 0.625 / 0.91; 0.74 | – |

**Reading:**
- **The window's effect depends on the body.**
  - On M2 (4 notebooks + shared reads) it adds +0.09 rules.
  - On 16nb (a notebook in every block) it takes −0.10 off. With notebooks everywhere, the window competes with the state's own
    local work: the lazy state again.
  - LM improves on both.
- **PAM's 16nb 600M point on the tier mix:**
  - s8 D 0.768, above the M2 b2p mean 0.692;
  - holdout 3.5662, 0.088 behind M2 b2p2;
  - so 16nb trades LM for rules at 600M.
- **Consequence:** t15pam = 16nb plain was the right body. W1 is not for the 16nb body.

## t15 INTERIM (same time): the 1.5B baselines are done. More data made T++ and Mamba-2 WORSE at rules at distance; their LM improved as expected. PAM 1.5B (16nb plain) runs since 03:16 IST, ETA ~11:00

| body | holdout 600M → 1.5B | s8 D 600M → 1.5B | s8 4k at 1.5B | clk2 8k at 1.5B | skill pack 2k + contrast at 1.5B |
|---|---|---|---|---|---|
| T++ (b2t2 → t15tpp) | 3.5572 → **3.3593** (−0.198) | 0.134 → **0.047** | 0.068 | 0.29 | 0.02 |
| Mamba-2 (b2m2 → t15mamba) | 3.4754 → **3.3025** (−0.173) | ~0.10 → **0.033** | 0.026 | 0.00 | 0.05 |
| PAM 16nb (w1n_n0 → t15pam) | 3.5662 → (running) | 0.768 → (running) | | | |

- **t15pam bars:** s8 D ≥ 0.147 (max baseline + 0.10); holdout ≤ 3.3225 (Mamba-2 + 0.02).
- The rule bar is all but certain. **The LM bar is the open question:** 16nb starts 0.09 behind M2 at 600M.
- Even at L = 0 the 1.5B baselines lose the letter rule (T++ shift 0.145, Mamba-2 0.49; affine 0.92 / 1.00). Their in-context rule
  inference at distance falls with data, while PAM's held at 600M.

## t15 VERDICT (09:03 IST, 2026-11-02): rules bar MET by a wide margin (PAM s8 D 0.545 vs T++ 0.047 / Mamba-2 0.033); LM bar MISSED (PAM 3.3784 vs ≤ 3.3225, by 0.056). With data, all three bodies' LM improve at about the same rate, and all three bodies' rules at distance FALL

**Arms:** `t15pam` / `t15tpp` / `t15mamba` `_main_c06b939` (103.2M / 101.9M / 102.6M). Same mix, own 120M stage A, 1.50B tokens,
seed 43; 6000. PAM = 16nb plain (by the w1m / w1n rule).

| readout | PAM 600M → **1.5B** | T++ 600M → **1.5B** | Mamba-2 600M → **1.5B** |
|---|---|---|---|
| holdout | 3.5662 → **3.3784** (−0.188) | 3.5572 → **3.3593** (−0.198) | 3.4754 → **3.3025** (−0.173) |
| WikiText val loss | 4.360 → 4.107 | – → 4.028 | – → 4.013 |
| s8 D (2k / 4k, scored) | 0.768 → **0.545** | 0.134 → **0.047** | ~0.10 → **0.033** |
| s8 D, free | 0.585 → 0.367 | – → 0.006 | – → 0.008 |
| s8 at 4k (scored) | 0.626 → 0.401 | → 0.068 | → 0.026 |
| B2 at L = 0 (the rule with the examples adjacent) | 1.00 → 1.00 | → 0.53 | → 0.745 |
| skill pack 4k + contrast, 20-way | 0.70 → **0.775** | → 0.105 | → 0.005 |
| clk2 8k (PAM: dilation + contrast; baselines as they are), 20-way | 0.39 → **0.47** | → 0.29 | → 0.00 |
| t40 far needles | 0.338 → 0.254 | → 0.154 | → 0.225 |
| text-arm tool read (free, syn / nat mean) | 0.263 → 0.30 | → **0.875** | → 0.69 |

The PAM 600M point is w1n_n0 (16nb, the same mix). The baselines' 600M points are b2t2 / b2m2 (the same mix and stage A).

**Bars:**
- rules: s8 D ≥ max(baselines) + 0.10 = 0.147: **MET (0.545)**;
- LM: holdout ≤ Mamba-2 + 0.02 = 3.3225: **MISSED (3.3784)**.

**Reading:**
- **LM:** every body improves by about the same amount (−0.17 to −0.20) for 2.5× the tokens. PAM 16nb stays 0.019 behind T++ and
  0.076 behind Mamba-2. This is plain modelling, and kb1 says the same.
- **Rules at distance fall with data for EVERY body.** PAM −0.22, T++ −0.09, Mamba-2 −0.07.
  - The baselines also lose the rules at L = 0 (T++ letter shift 0.145; they now follow priors).
  - PAM's lead shrinks from +0.63 to +0.50 but stays enormous.
  - Likely cause: the b2 share is fixed at 3 % while the web text grows; the model's priors strengthen against an in-context rule.
    (A guess. Testable with a higher or ramped rule share at 1.5B.)
- **Reach with query-time tools GROWS for PAM** (skill pack 4k + contrast 0.70 → 0.775; clk2 0.39 → 0.47). The trace is still
  there; plain decoding loses it to stronger priors.
- **Reading fetched rows stays T++'s** (0.875 vs PAM 0.30); the package's field cut is the reader.
- **Consequence:** r3p' (3B PAM) is not launched on this evidence; it is Lead's / the user's call. The LM gap is the next problem
  (the bar order: intelligence first, met; LM, missed).

## t15pam_m2: the M2 body at 1.5B, b2p2's exact recipe on the t15 mix (pre-registration, 2026-11-02 09:05 IST; Lead GO, board 913bc9c)

**Why:** t15 met the rules bar and missed LM (16nb 3.3784 vs ≤ 3.3225). On this mix at 600M, M2 (b2p2 3.4800) beat 16nb (w1n_n0
3.5662) by 0.086 on holdout while holding rules. This is the cheapest run that can close bar 2, and it decides which PAM body scales.

**Arm:** `t15m2_main`. M2 (PAM_LAYERS 3,7,11,15, MLP 2128, SHARE_READS, ECHO; 102.9M), from `c1m2a_pre_7ae6f1f` (b2p2's own
stage A, with aptitude docs, unlike 16nb's r3ca_pre: stated as part of the recipe). LR 1.2e-3 (b2p2's; M2 failed at 3B only at
2e-4, r3b). The t15 mix, 1.50B tokens, seed 43, 6000 alone.

**Bars** (as t15): holdout ≤ 3.3225 (Mamba-2 + 0.02) AND s8 D ≥ 0.147.
**Reported:** skill pack 4k + contrast, clk2 8k, t40 far needles, the tool read, and the slope from b2p2 (600M).
**Prior work / novelty:** as t15 (a measurement).

## t15pam_m2 VERDICT (13:20 IST, 2026-11-02): the M2 body at 1.5B is the better PAM body. LM bar MISSED by 0.003 (3.3255 vs ≤ 3.3225); rules bar MET (s8 D 0.579). It beats 16nb on LM by 0.053 and on reach; SCI-1's body = M2

**Arm:** `t15m2_main_48a3bda` (M2: 4 notebooks + shared reads + echo, 102.9M), from `c1m2a_pre`, b2p2's recipe, the t15 mix,
1.50B tokens, seed 43; 6000.

| readout at 1.5B | **M2 (t15m2)** | 16nb (t15pam) | T++ | Mamba-2 |
|---|---|---|---|---|
| holdout | **3.3255** | 3.3784 | 3.3593 | **3.3025** |
| WikiText val loss | 4.0408 | 4.1071 | 4.0275 | 4.0128 |
| s8 D (scored) | 0.579 | 0.545 | 0.047 | 0.033 |
| skill pack 4k + contrast, 20-way | **0.85** | 0.775 | 0.105 | 0.005 |
| clk2 8k, 20-way | **0.87** | 0.47 | 0.29 | 0.00 |
| tool read (free, syn / nat) | 0.55 / 0.40 | – | 0.875 mean | 0.69 mean |

**Slope 600M → 1.5B** (M2: b2p2 → t15m2): holdout 3.4800 → 3.3255 (−0.155; T++ −0.198, Mamba-2 −0.173); s8 D 0.691 → 0.579.

**Reading:**
- M2 at 1.5B lands between Mamba-2 (−0.023) and T++ (+0.034) on LM, 0.003 short of the bar, and keeps a huge rules / reach lead.
- Its LM slope is the smallest of the three (−0.155), but it started 0.083 ahead of T++ (b2p2 3.480 vs b2t2 3.557).
- 16nb was the wrong scaling body on this mix. **SCI-1 uses M2** (Lead's rule: M2 if it meets the bar or beats 3.3784).
- **Note:** M2's stage A (`c1m2a_pre`, with aptitude docs) differs from 16nb's (`r3ca_pre`). This is stated; the body comparison
  carries that difference.

## SCI-1: the first product-size mind, a science / research assistant (pre-registration, 2026-11-02 14:30 IST; user GO 10:12 IST via Lead; Lead's frame on the v16 board, "SCI-1")

**Why** (the user): "plan a bit big model with more data chosen for science and research; size for full utilisation of the 6000; a
3-5 day run is OK; resumable; checkpoints in between to check". Then:
- "include tool calling, thinking and coding, not to be a coder but to code and run as a scientist" (11:10);
- continuable after 5–10B tokens.

**Strength fit** (Lead):
- What the mind does well: standing rules and protocols held in state across long sessions; a held rule applied to fetched results;
  constant cost per session.
- What it does badly: raw LM and composition (kb1, t15, R × K implicit). So NOT a coder, a scientist that calls tools.

**Claims:** a PRODUCT run in its own tokens tier. No equal-data baselines exist at this size and token count on one card in the
week. The reference block (SmolLM2-360M at 4T tokens, Pythia-410M at 300B) is context only, never a verdict. The equal-data
claims stay at the 600M / 1.5B / 3B tiers.
**Prior work:** small open science-heavy LMs (Galactica 125M; SmolLM2's mixes). A product, so no method claim.

**Body:** **M2** (4 notebooks every 4th block + shared reads + echo + chrono + out_gate + delta) (t15pam_m2 verdict: t15m2 3.3255 vs 16nb 3.3784 at 1.5B; reach clk2 0.87 vs 0.47), no window (closed on both bodies: w1m / w1n). Package v0 contract checked by
Library on the exact config: PASS (Library, CPU, random init, reset_at_eos on: chunked / decode / deep copy 7.7e-7 / 1.0e-6 / 1.0e-6, also across EOS resets; address at block 12; _dilate, head and stops OK).
**Geometry** (chosen by the size smoke, measured, `v13_sempty/tmp/sci1_size_smoke.sh`): dim 1024 × 24 layers, 8 heads × 128, mlp_hidden 3704, notebooks in blocks 3 / 7 / 11 / 15 / 19 / 23 = 385.82M params; state
3.67 MB (O(1) in length; optional F2 address 1,028 B/token int8 or 8.25 B/token sign bits).

**Recipe:**
- from scratch, one stage, the full mix from step 0; seed 43; AdamW (0.9, 0.95), wd 0.01, clip 1.0, bf16;
- **SEQ 4096** (the user's ask via Lead; guard: stay at 2k if 4k is > 15 % slower; measured 37.8k vs 38.1k tok/s, −1 %), 16 × 4096 = 65,536 tokens / step (at SEQ 4096: 16 sequences, GRAD_ACCUM 2 (exact));
- LR 8e-4 (the 103M runs used 1.2e-3; ~1 / width scaling 1.2e-3 × 588 / 1024 ≈ 7e-4, rounded up for WSD's constant phase; no screen at this size, stated) with warmup 2,000 steps, then CONSTANT (`--lr_schedule wsd_stable`): the main line;
- decay branch: `--continue_from` the final stable state, LR linear to 0 over 1,000,013,824 tokens (10 % of the main line); its
  best_model.pt is the release point.
- **Main-line budget:** stage A 500,039,680 (7,630 steps, ~4 %) + stage B 9,999,998,976 (152,588 steps) tokens (~3.5 (+ CHECK and val overhead, ~3.8) days at the measured 37.8k tok/s); 30 (11.5B incl. the decay) tokens per param (≥ 20,
  the Chinchilla floor).

**Mix** (stage B = the main line; realized token-share targets, calibrated by streaming the real blend at SEQ 4096 with the filter
on, `calib4`; the trainer header logs the weights):

| block | sources (licence; quality filter) | share |
|---|---|---|
| web | dclm-edu (DCLM-baseline re-scored, edu ≥ 3), fineweb-edu (edu ≥ 3) | 39.1 % (20.0 / 19.1) |
| science papers | Common Pile arxiv / pubmed / peS2o, *_filtered (per-doc open licences); peS2o minus its arXiv-sourced papers | 25 % (7.5 / 8.5 / 9) |
| textbook | Cosmopedia v2 (smollm-corpus, ODC-BY; SmolLM2's set) | 9 % |
| quantitative | FineMath-4plus (ODC-BY) | 3 % |
| chat | smoltalk2 Mid (single-turn, think stripped) 3 % + smoltalk v1 multi-turn (magpie-ultra + everyday, Apache-2.0) 3 % | 6 % |
| tool calling | glaive-fc-v2 + ToolACE (Apache-2.0) + library-fetch docs (half raw sentences), Hermes format | 4 % |
| short thinking | MetaMathQA + GSM8K TRAIN (MIT), ≤ 1,200 chars, ON / OFF 50 / 50 | 2.5 % |
| scientific python | Common Pile Stack-Edu (its edu filter) ∩ a scientific-import filter (keeps 25 % of Python files) | 9 % |
| synthetic (answers ×100) | b2s rules 1.7 (half spread over the 4k window), proto4k 0.4 (Library; A26 families held out), recall4k 0.34 | 2.44 % of tokens = **~3.7 % of the loss weight** |

- **Weight per token vs text** (answers ×100; measured): b2s 1.55×, proto4k 1.82×, recall4k 1.37× (compact b2 is 5.24×, which is why
  the spread version needs more tokens for the same rule weight).
- **The synthetic weight is a deliberate exception** to the default ≤ 3 % frame (Lead), for the thesis skill. Rules at distance fell
  with data at 3 % b2 tokens (t15: PAM 0.768 → 0.545). bud's measured cost: a 4.9 % synthetic weight cost +0.013 holdout.
- **Weights:** stage B `SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,tools,think,pysci,b2s,recall4k,proto4k`, `WEIGHTS=0.21225,0.22533,0.05068,0.07697,0.09901,0.11376,0.02685,0.02258,0.03264,0.03433,0.02362,0.04893,0.02345,0.00426,0.00535`. Stage A (plain sources only, renormalized) `SOURCES=dclm,fineweb,arxiv,pubmed,pes2o,cosmo,finemath,smoltalk2_mid,chatmt,pysci`, `WEIGHTS=0.2335,0.24789,0.05575,0.08468,0.10892,0.12515,0.02954,0.02484,0.03591,0.05383`.
- **Data audit** (`v13_sempty/tmp/sci1_data_audit.py`, 2,000 train docs per source through the real iterators;
  `logs/sci1/v13_sempty_sci1_data_audit.json`):
  - decontamination drops: think 0.35 %, dclm 0.10 %, fineweb / arxiv / cosmo / finemath 0.05 %, every other source 0;
  - within-source duplicates (first 1k chars): tools 2.0 %, think 0.25 %, every other source 0;
  - peS2o arXiv-sourced dropped: 2.3 %.
- **Decontamination** (mandatory for the arena): `v7/decontam.py`, word 13-grams.
  - Sets: HellaSwag val + test, ARC-e / c val + test, PIQA val + test, Winogrande val + test, OBQA val + test, LAMBADA test,
    GSM8K test, MMLU test + val, bAbI test (BABILong's base: facts + question + answer, never a haystack), and Library's
    `v14/decontam_items.json` (A25 / A26 / bench / styles / v16 prompts / story texts; protocol_docs' training wordings are
    deliberately not in it).
  - 4,371,599 grams; index `.cache/decontam/v1.npy` sha256 `08766210ac06dd67066aee27dc3c961f000fa078b62ee1c2991b3fa97bd2096d`.
  - A doc with any hit is dropped, counted as `<source>#decon` in the cursor. GSM8K test is never in the think source.
  - Library owns a post-launch audit (`v14/decontam_audit.py`); any cell > 0.1 % gets a tighter filter at the first continuation.
  - Every benchmark is reported with its clean-subset score beside the full one.
- **Multi-turn, measured as streamed** (Lead's ask):
  - before chatmt, 3+-turn dialogs were ~14 % of chat + tool tokens (smoltalk2 Mid 100 % single-turn; tools 1 / 2 / 3+ turns =
    28 / 38 / 35 % of tool tokens);
  - with chatmt (100 % 3+ turns): ~44 %;
  - dialogs longer than the window are cut across windows by the packer; none are dropped.
- **Two stages** (the C1-ctl lesson: the schedule caused the recall jump): stage A = ~4 % of tokens on plain sources (science
  included); stage B = `--continue_from` stage A with the full mix and a 500-step re-warm; one lineage row each.
- **Training-time ingredients ON** (SCI-1 is its own tier, so comparability with t15 is not a constraint):
  - ANSWER_IN_WINDOW=1 (orphaned answers get weight 1);
  - RESET_AT_EOS=1 (state reset at document separators). lg8r: the reset removed the forgetting that packed unrelated docs teach;
    lg32 without it cut every echo half-life by 25–55 %. Sessions contain no EOS; Library's contract passes with resets crossed
    by chunks and decode.
- Holdout = the shared hash, 5 %, every source (probe filler never trains). Per-source val: dclm, fineweb, arxiv, pubmed, pes2o, cosmo,
  tools, think, pysci, chatmt.

**Checkpoints** (continuable, the user's amendment):
- latest.pt every 500 steps;
- a FULL milestone (weights + optimizer + schedule + per-source cursors + RNG) every 15,259 steps (~1B tokens), keeping
  the last 4: these are the continuation points;
- best_model.pt on holdout; stable_final.pt at the end of the main line; the decay branch in its own dir;
- disk guard: refuse to start below 40 GB free. Budget ~41 GB of 62 at ~4.6 GB per full checkpoint (386M × 12 B).

**CHECK at every milestone** (`v13_sempty/tmp/sci1_check.sh`, a watcher beside training):
- holdout + per-source split + WikiText (from the val lines);
- s8 D (2k / 4k, 400 trials);
- skill pack 2k / 4k + contrast;
- t40 far / near needles;
- the known-suite subset (`scripts/mc_eval.py`: HellaSwag / ARC-e / PIQA, identical to lm-eval);
- 5 shown generations (science, protocol, chat, a tool call + result, a short `<think>`) with step, tokens and hash;
- Library's A26 / A26-T (protocol × fetched row) at milestones, run by Library.

**Tripwires** (pre-registered responses at the FIRST continuation; a logged mix change + a new lineage segment):
- **rules:** s8 D at milestone 2 < 0.40 → raise the b2 rule share;
- **protocols:** A26 arm **G1** (Library: `--wording train`, UNIT on mass / volume in protocol_docs' own wordings, the package's
  MIND-HELD arm at 2k, 20 docs) at milestone 2 ≤ its NOPROT arm + 0.05 → raise the proto share, within the ~4 % synthetic ceiling
  and with its cost measured. Library reports G1 / G2 / G3 at every milestone, on copies on the 4090; the 6000 run is untouched. Day-0: no current body applies a stated convention (≤ 0.05 =
  control).
- **thinking** (the user's switch, via Lead):
  - default OFF; the think docs are 50 / 50 ON (a 'Reasoning: on' system line + a short `<think>` trace + the answer) / OFF (the
    answer only), so the marker is the only difference;
  - `scripts/think_eval.py` scores GSM8K test ON vs OFF plus the OFF trace leak;
  - ON ≤ OFF + 0.02 at milestone 2, OR OFF leak > 5 % → drop the think source at the first continuation (its 2.5 % returns to web);
    the default stays OFF.
- **LM** (mirror): holdout at milestone 2 > the projection + 0.02 → lower the synthetic weight. The projection is a fit
  L(t) = a + b·t^(−c) to the run's own holdout vals from 0.25B to 1B tokens.

**Predictions** (written first; no claim, its own tier):
- the release holdout 2.95–3.05 (103M M2 at 1.5B tokens: 3.3255; this is 3.7× the params and 7.7× the tokens);
- HellaSwag acc_norm 0.36–0.42 (Pythia-410M at 300B: ~0.40; our 386M at 11.5B) (SmolLM2-360M at 4T: 54.5 on its card);
- s8 D at release ≥ 0.40;
- the tool call well-formed in the shown generation by milestone 2.

**Release:** `export_hf_release.py` + verify + push. The mind + package v0 + write-up + lineage tree; the name states params and
tokens.

**Lineage:** one row per segment (main line `sci1_main`, decay `sci1_decay`, each continuation); "running" at launch.

## SCI-1 milestones 1 and 2 (CHECK on the 4090, 11:48 IST, 2026-11-03; run `59c3cf5`, M2 386M, SEQ 4096)

Milestone 1 = step 15,259 (1.0B tokens total), milestone 2 = step 30,518 (2.0B). Logs: `logs/v13_sempty_sci1_checks_59c3cf5_local.log`,
`logs/sci1/v13_sempty_gens_*` (the five generations per milestone).

| readout | M1 (1.0B) | M2 (2.0B) | tripwire / bar |
|---|---|---|---|
| s8 D (scored) | 0.322 | **0.538** | < 0.40 at M2 → raise b2s: **not triggered** |
| skill pack 2k / 4k / 8k + contrast, 20-way | 0.54 / 0.16 / 0.07 | 0.76 / 0.51 / 0.10 | – |
| clk2 8k (dilation + contrast) | 0.08 | 0.34 | – |
| t40 far needles | 0.80 | 0.91 | – |
| HellaSwag / ARC-e / PIQA | 0.281 / 0.459 / 0.576 | 0.293 / 0.479 / 0.594 | – |
| think ON / OFF accuracy (GSM8K test) | 0.005 / 0.03 | 0.02 / 0.03 | ON ≤ OFF + 0.02 at M2 → drop think: **TRIGGERED** |
| think trace rate ON / OFF leak | 1.0 / 0.005 | 1.0 / **0.24** | OFF leak > 5 % at M2 → drop think: **TRIGGERED** |

- The switch itself does not hold yet: at M2, 24 % of OFF answers still carry a trace. Thinking does not help at this size
  (accuracy near zero in both modes).
- Per the pre-registration, both thinking tripwires are acted on at the FIRST CONTINUATION (drop the think source, 2.5 % back to web),
  not mid-run.
- Not yet scored here: A26 G1 (Library) and the holdout-vs-projection tripwire (training log).
- Generations at M2 (shown): the tool-call prompt reads the result correctly ("The boiling point … is 78.4 C") and then runs on past
  `<|im_end|>` (repetition); the protocol prompt copies the row instead of applying "one decimal place" (as Library's day-0 predicted).

## cxre: complex vs real PAM at the ctl recipe, PLAIN body (CANCELLED before launch 08:02 IST 2026-11-05 by the user via Lead: port the paper body's parts to the complex head instead, then a fair cx_m2 run; pre-registration kept for the record; 07:58 IST, 2026-11-05; user GO via Lead 07:56 IST)

**Question:** at matched params and the identical product recipe, does the complex-arithmetic PAM learn differently from the real
one? **This isolates the arithmetic; it is NOT the paper's body**: the complex `PAMLayer` has no delta / chrono / out_gate /
share_reads / echo (Lead read model.py; the last two and PAM_LAYERS are asserted real-only), so both arms run the PLAIN body
where both heads have identical logic.
- **Arms:** `cx_plain` = complex `baseline` (100.36M: dim 384 complex, 6 heads × 64, 16 layers, a notebook in every layer);
  `re_plain` = real `baseline_real_pm` (101.89M: dim 588, 6 × 98, 16 layers). Both: DELTA = CHRONO = OUT_GATE = ECHO = 0, no
  PAM_LAYERS / SHARE_READS; RESET_AT_EOS = ANSWER_IN_WINDOW = 1, the same.
- **Recipe = ctl's exactly:** the cont2 mix (sampler weights of `sci1_cont2_env.sh`), SEQ 4096, BATCH 16 × 4096 = 65,536 tokens per
  step, LR 1.2e-3, warmup 500, 0.9B wsd_stable + 0.1B decay = 1.0B tokens each, seed 43, decontam index, code caa4da9 (worktree
  `../qllm_ctl`, venv qllm311). Runner `v13_sempty/tmp/cxre_chain.sh` (the ctl scheduler).
- **Bars (ctl's B1–B5, read as differences, |Δ| beyond noise = a difference):** B1 s8 D (Δ ≥ 0.10 either way is a difference);
  B2 far needles 8k / 16k (Δ ≥ 0.10); B3 holdout (Δ ≥ 0.03 nats); B4 HS / ARC-e / PIQA (reported); B5 told rule (A26 G1 2k twin
  read, if the package path supports the complex state; else reported as not run). Context: ctl's PAM M2 (the full body) on the
  same recipe: s8 D 0.523, needles 0.53 / 0.41, holdout 3.427.
- **Prior record:** ledger row 3 (EXPERIMENTS_SEMPY:209; Sept 2026): real 101.9M vs complex 100.4M, parity on WikiText-103 at 100M
  tokens (4.38 vs 4.36), real −0.24 NLL on TinyStories, with different recipes per arm. This run fixes the recipe.
- Prior work: complex-valued / unitary RNNs (Arjovsky 2016; Wisdom 2016), complex-valued Transformers; our own v11 complex PAM.

## far block RESULT, part 1 (s8 / needles / benchmarks; 12:22 IST 2026-11-12; told 8k pending from Library's FAR-TOLD)

Run `sci1_long2_4a97ca7` (segment 2, far block ramped in from M11 over 2B; ran unattended Nov 7-12). CHECKs at the CHECK's code; s8 =
mean of shift_wide_new / affine_map_new (400 trials each, seed 7000), mean target margin in brackets.

| point | s8 @2k | s8 @4k | **s8 @8k (untrained)** | t40 8k / 16k | HS / ARC-e / PIQA |
|---|---|---|---|---|---|
| M11 (11.0B, pre-ramp) | 0.849 (+1.9) | 0.529 (+0.0) | **0.186 (−1.1)** | 0.894 / 0.836 | 0.336 / 0.539 / 0.625 |
| 344k = best_model (22.5B, before fineweb ran dry) | 0.999 (+14.5) | 0.991 (+12.3) | **0.979 (+8.5)** | 0.931 / 0.886 | 0.352 / 0.566 / 0.635 |
| 26B (396,734) | 0.995 (+14.1) | 0.994 (+11.9) | **0.974 (+8.0)** | 0.919 / 0.833 | 0.357 / 0.557 / 0.621 |
| 27B (411,993) | 0.991 (+14.9) | 0.989 (+12.6) | **0.964 (+7.9)** | 0.919 / 0.831 | 0.355 / 0.554 / 0.630 |

**Headline (pre-registered cell): s8 @8k 0.186 -> 0.964-0.979 (3-point mean 0.972), margin −1.1 -> +8.** 8k is beyond the 4096 training
window (no state carries across windows) and its per-gap distance (~1,170 tokens at 8 shots) is beyond b2s_far's trained per-gap range
(<= ~800): training rule carry at 1.8-3.2k generalised to 8k. 2k / 4k are trained distance (labelled). Keep rule part 1 (s8 @8k
>= +0.05 over M11) MET by +0.79; told 8k (part 2) pending. Needles and benchmarks also up (more data too: 11B -> 27B).
**Rollback test (Lead's rule):** 344k vs 27B: s8 @8k +0.015 (within jitter), t40 16k +0.055 (< ~0.1), ARC-e +0.012: not more than the
jitter on the thesis cells so far -> segment 3 continues from 28B (final after told 8k). The fineweb loss (from 347k) shows mostly in
the web holdout loss, little in the thesis cells.

## wio: "what is ours" (PAM vs Gated DeltaNet, and PAM without its clock) at the ctl recipe + the far block (PRE-REGISTERED 16:32 IST 2026-11-16, before any number; Lead + user GO 2026-11-12)

**Question:** is the shared clock / phase what makes PAM's rule carry ours, and not the gated delta rule it shares with prior work?
**Closest prior work: Gated DeltaNet** (Yang, Kautz, Hatamizadeh 2024, arXiv 2412.06464): gated delta-rule state, short conv on
q / k / v, output gate, **no rotary or phase on q / k** (checked in fla 0.3.2 `fla/layers/gated_deltanet.py`). Difference in one line:
PAM rotates q / k by a content-driven per-head clock (Chrono) that the shared readers reuse; GDN carries position only through the
conv and the decay.
**Arms (each ~103M, identical data, seed, schedule):** the ctl recipe (1.0B tokens: 0.9B wsd_stable + 0.1B linear decay, SEQ 4096,
BATCH 16 x 4096, LR 1.2e-3, seed 43, decontam v1) on the cont2 mix **plus the far block** (b2s_far 1.2 % + proto_far 0.8 %, taken
from dclm / fineweb in proportion), so it doubles as the far-data equal-data control:
- `gdn`: fla GatedDeltaNetModel, 12 x 768, 6 heads x 64, SwiGLU 1280, **102.51M** (1f3f205); published reference implementation.
- `nochrono`: PAM M2 with Chrono off (plain RoPE; the shared readers read on the RoPE rotation, 6dfff6c; MLP 2128 as M2: the clock's
  warp projections are ~37k params, < 0.1 %, so no re-match).
- `pam`: PAM M2 all on (ctl's body).
- `mamba2`, `tpp`: ctl's baselines.
Runner `v13_sempty/tmp/wio_chain.sh` (ctl's scheduler: arm by arm to the end, gdn first, then nochrono; yields to CHECKs; never
trains while the user's own GPU jobs are on the card). Readouts in logs/wio/; told cells by Library (FAR-TOLD protocol) at each arm's end.
**Readouts per arm:** s8 @8k (b2 suite, 400 trials, seed 7000, spread 0 / 2k / 4k / 8k; **3-save mean** over the decay saves at
+500 / +1000 / end, with the margin); told 2k / 8k (ptwin); needles t40 8k / 16k; clk2 8k / 16k; holdout loss; HS / ARC-e / PIQA;
state MB at decode.
**CLARIFICATION of the 3-save mean (02:06 IST 2026-11-17, Mind + Lead; recorded after gdn's readouts and BEFORE any other arm's
readout exists):** KEEP_EVERY 500 counts from step 0, so the decay saves land at absolute steps 14,000 / 14,500 / 15,000 + end
(15,259), not at decay +500 / +1000. The 3-save mean = the mean of the LAST THREE (14,500 / 15,000 / end), for every arm; 14,000 is
reported but not in the mean. gdn: (0.37375 + 0.4075 + 0.44375) / 3 = 0.4083. So rule 1's bars for `pam`: s8 @8k 3-save mean
>= 0.508 (gdn + 0.10) AND told 8k >= gdn + 0.10 once Library has gdn's told; `nochrono` must lose the gap (rule 2 unchanged).
**SECONDARY READOUT: s8 per-task split by distance (02:07 IST 2026-11-17, Argon via Lead; recorded BEFORE any nochrono / pam / mamba2 /
tpp readout; no new bar):** every wio arm reports shift_wide_new and affine_map_new separately at 0 / 2k / 4k / 8k (3-save mean + end)
beside the s8 3-save mean. gdn 3-save: shift_wide_new 0.897 / 0.511 / 0.326 / 0.136 (chance 0.125 by 8k); affine_map_new 0.973 / 0.894
/ 0.820 / 0.681. HYPOTHESIS (Argon, not a claim): shift needs a phase reference that GDN lacks. TEST: nochrono vs pam on shift_wide_new
@8k; a nochrono arm that holds shift_wide (near pam's level) REFUTES it; a nochrono drop toward gdn's while pam holds supports it.
**SECONDARY READOUT 2 (11:52 IST 2026-11-17, Argon via Lead; recorded after nochrono's readouts and BEFORE any pam readout; no new
bar):** (a) **affine_map_new @0** for pam. nochrono 3-save 0.711, gdn 0.973. If pam ~= nochrono and still trails gdn, the affine gap
is the write / erase dynamics, not the clock. HYPOTHESIS (Argon, not a claim): GDN's gated erase on the delta write acts as an online
map update. (b) **shift_wide 2k / 4k / 8k** stays the Chrono stress test (nochrono 0.709 / 0.558 / 0.345, gdn 0.511 / 0.326 / 0.136).
**The precise erase comparison** (code-checked: v13_sempty/model.py PAMLayer delta path ~l.549 / 878-905, triton_kernels.pam_delta_batched;
fla 0.3.2 fla/layers/gated_deltanet.py ~l.150-270). Both use the same recurrence form S_t = g_t S_{t-1} (I - b_e k k^T) + b_w v k^T with
unit-norm keys and a per-token, per-head decay g_t. The differences:
- **Erase and write gates.** GDN has one tied beta = sigmoid(b_proj x) per head per token, so erase strength = write strength
  (allow_neg_eigval off: beta in (0,1)). PAM has two untied gates: write b_w = sigmoid(W_w x), and erase b_e = sigmoid(W_e x - 3),
  capped at 0.95. Erase starts at ~0.05 (near-additive at init) and must learn to erase; GDN replaces in full whenever it writes.
- **Decay.** GDN: alpha = exp(-exp(A_log) * softplus(a_proj x + dt_bias)) per head (Mamba-2 style, per-head A, dt_bias init).
  PAM: g = exp(-softplus(dt_proj x + bias)) per head, one shared initial bias.
- **Where the delta state lives.** GDN: every one of 12 layers has its own delta state (6 heads x 64 keys x 128 values,
  expand_v 2). PAM M2 (wio pam / nochrono): notebooks only in blocks 3 / 7 / 11 / 15 of 16; the other 12 blocks are cheap readers
  of the nearest notebook below (SHARE_READS). So PAM has 4 write / erase sites, GDN has 12.
- **q / k path.** GDN: short conv (k = 4) + SiLU on q / k / v and an output gate (gated RMSNorm). wio PAM: no short conv
  (SHORT_CONV=0), RoPE (nochrono) or the Chrono clock (pam) on q / k, and the N4 output gate.
Reading for (a): if the affine gap is the erase, the candidates are the tied-vs-untied gates (GDN's replace-on-write) and the 3x more
write sites. Neither is touched in this screen; a follow-up would be a v14-lane question (new mechanism), not a v13 tune.
**nochrono ARM RESULT (11:52 IST 2026-11-17; 102,870,812 params, 1.000B tokens, solo avg 45.4k tok/s; logs/wio/*nochrono*):**

| arm | params | tokens | s8 @8k 3-save (14,500 / 15,000 / end) | shift_wide 0 / 2k / 4k / 8k | affine_map 0 / 2k / 4k / 8k | clk2 8k / 16k | HS / ARC-e / PIQA | val / wiki_val NLL |
|---|---|---|---|---|---|---|---|---|
| gdn | 102.51M | 1.0B | **0.408** (0.374 / 0.408 / 0.444) | 0.897 / 0.511 / 0.326 / 0.136 | 0.973 / 0.894 / 0.820 / 0.681 | 0.17 / 0.06 | 0.275 / 0.440 / 0.594 | **3.385** / 4.066 |
| nochrono | 102.87M | 1.0B | **0.427** (0.384 / 0.446 / 0.453) | 0.873 / 0.709 / 0.558 / 0.345 | 0.711 / 0.660 / 0.628 / 0.510 | 0.54 / 0.06 | 0.273 / 0.464 / 0.592 | 3.427 / 4.144 |

NLL: the same val loader at step 15,259 (nochrono: the trainer's own [val] line; gdn: POST_VAL post hoc, since its in-trainer val
failed in fp32). **GDN leads on language: val 3.385 vs 3.427 (+0.042 for nochrono, above the 0.03 stated limit), worse on every source
(dclm 3.436 vs 3.396, fineweb 3.469 vs 3.431, chatmt 1.909 vs 1.881).** The PAM claim, if any, is rules at distance, not language.
Reading (no verdict until pam): without the clock, PAM still holds far shift (8k 0.345 vs gdn 0.136, 2.5x; 2k 0.709 vs 0.511) but
trails GDN on affine at every distance (0.711 vs 0.973 at 0). s8 @8k averages the two and lands near gdn (+0.019). Argon's
phase hypothesis: a nochrono shift of 0.345 is far above gdn, so the phase-reference story needs pam's shift to be compared against it
(refuted if pam ~= nochrono). Rule 2 needs pam - nochrono >= 0.10 -> pam >= 0.527 and pam >= 0.508 (rule 1).
**pam ARM RESULT + VERDICT (23:31 IST 2026-11-17; 102,884,948 params, 1.000B tokens (900,005,888 main + 100,007,936 decay), solo
~64k tok/s; logs/wio/*pam*):**

| arm | params | tokens | s8 @8k 3-save (14,500 / 15,000 / end) | shift 0 / 2k / 4k / 8k | affine 0 / 2k / 4k / 8k | t40 2k / 8k / 16k | clk2 8k / 16k | HS / ARC-e / PIQA | val / wiki NLL |
|---|---|---|---|---|---|---|---|---|---|
| gdn | 102.51M | 1.0B | **0.408** | 0.897 / 0.511 / 0.326 / 0.136 | 0.973 / 0.894 / 0.820 / 0.681 | 0.681 / 0.458 / 0.325 | 0.17 / 0.06 | 0.275 / 0.440 / 0.594 | **3.385** / 4.066 |
| nochrono | 102.87M | 1.0B | **0.427** | 0.873 / 0.709 / 0.558 / 0.345 | 0.711 / 0.660 / 0.628 / 0.510 | 0.478 / 0.408 / 0.353 | 0.54 / 0.06 | 0.273 / 0.464 / 0.592 | 3.427 / 4.144 |
| pam | 102.88M | 1.0B | **0.343** (0.359 / 0.328 / 0.343) | 0.796 / 0.610 / 0.440 / 0.233 | 0.656 / 0.571 / 0.529 / 0.453 | **0.900 / 0.817 / 0.803** | 0.42 / 0.10 | 0.274 / 0.451 / 0.597 | 3.425 / 4.136 |
| mamba2 (added 04:55 IST 11-18) | 102.61M | 1.0B | **0.259** (0.206 / 0.283 / 0.288) | 0.260 / 0.038 / 0.027 / 0.021 | 0.947 / 0.590 / 0.558 / 0.497 | 0.753 / 0.433 / 0.419 | 0.02 / 0.01 | 0.275 / 0.459 / 0.587 | **3.381** / 4.060 |
| tpp (added 10:10 IST 11-18; trained at 4096, so 8k / 16k are beyond its length) | 101.92M | 1.0B | **0.080** (0.099 / 0.068 / 0.074) | 0.118 / 0.082 / 0.072 / 0.011 | 0.502 / 0.453 / 0.415 / 0.149 | 0.967 / 0.336 / 0.081 | 0.01 / 0.00 | 0.271 / 0.462 / 0.602 | 3.432 / 4.080 |
Per-distance shift / affine cells are 3-save means for all five arms (mamba2 / tpp first entered as end-save values, corrected 11:2x IST 11-18 in Lead's paper check; end save: mamba2 shift 0.270 / 0.020 / 0.018 / 0.020, affine 0.968 / 0.625 / 0.598 / 0.555; tpp shift 0.138 / 0.072 / 0.055 / 0.007, affine 0.527 / 0.465 / 0.438 / 0.140). s8 D (2k + 4k, 3-save): pam 0.537, nochrono 0.639, gdn 0.638, mamba2 0.303, tpp 0.256. Solo speed = median step-line tok/s of each main log: pam 63k, nochrono 49k, gdn 114k, mamba2 113k, tpp 83k.

Per-source val (pam / nochrono / gdn): dclm 3.434 / 3.436 / 3.396, fineweb 3.467 / 3.469 / 3.431, chatmt 1.911 / 1.909 / 1.881.
**Checks before the verdict (Lead, 23:31):** (1) config: the [args] of pam main vs nochrono main differ ONLY in `chrono` (and the
checkpoint dir); lr 1.2e-3, seed 43, mix + far block, steps 13,733, decay 1,526 all equal; all 4 [args] blocks in the pam log identical.
(2) resumes: pam main ran fresh twice (killed within 1 min, no save) then resumed at 9,400 (after the reboot) and 13,200 (the 44B gap);
each resume restored the doc cursors and the step / token count from latest.pt and replayed from there (the steps after the save were
redone; the packer's partial window is not saved, so a few docs at the cut can shift); LR constant 1.2e-3 throughout the main
(wsd_stable), the decay continued from stable_final; total 1,000,013,824 tokens. Loss at the replayed steps (9,450: 2.72 -> 2.50;
13,250: 2.99 -> 2.56) is within the 50-step log noise (2.45-2.99 between consecutive lines). Val NLL at every 2,000 steps matches nochrono
within 0.007 (2k 4.042 / 4.035 ... 13,733 3.563 / 3.565): no damage from the resumes. (3) ctl's "PAM s8 D 0.523" is s8 D = mean of 2k
and 4k (scored), not @8k. On that metric wio pam = (0.610 + 0.440 + 0.571 + 0.529) / 4 = **0.538**, i.e. the same as ctl: the far
block did not lower it. On the same 2k / 4k metric nochrono 0.639, gdn 0.638.
**VERDICT (claim rule as pre-registered): claim NOT met.** Rule 1 fails: pam s8 @8k 0.343 < gdn + 0.10 (0.508), and pam < gdn
(0.408) and < nochrono (0.427); rule 2 is moot. **"The clock is what is ours" is not supported at 100M / 1B tokens with far docs: the
Chrono clock LOWERS rule carry at every distance on both tasks vs nochrono** (shift 8k 0.233 vs 0.345; affine 8k 0.453 vs 0.510).
Language: pam = nochrono (3.425 vs 3.427), both 0.04 behind GDN. **What the clock DOES buy: needles** (t40 16k 0.803 vs nochrono 0.353,
gdn 0.325; 2k 0.900 vs 0.478 / 0.681): verbatim recall at distance, not rules.
**Claim rule (pre-stated):** "the clock is what is ours" stands only if BOTH
1. `pam` beats `gdn`: s8 @8k 3-save mean >= gdn + 0.10 AND told 8k >= gdn + 0.10 (n 100, ~±0.07); and
2. `nochrono` loses that gap: pam − nochrono >= 0.10 on s8 @8k, and nochrono's lead over gdn is <= half of pam's.
If (1) holds but not (2), the edge is PAM's but not the clock's (say which parts remain: notebooks, readers, echo). If (1) fails,
no "ours" claim against GDN at this scale (stated as a limit). Language (holdout) and benchmarks are reported, not bars; a pam holdout
worse than gdn by > 0.03 is stated next to any rule claim. Params and tokens are in every table row.
**Order / cost:** after the chat-off CHECKs; ~3-4.4 h per arm on the 4090 when the card is ours (GDN speed measured in its smoke).

### wio launched (Lead GO, 20:09 IST 2026-11-16; code 227f3eb; tmux mindwio on the 4090, gdn arm first)

- gdn smoke (solo, 1,350 steps at the arm recipe): ~62k tok/s (main ~3.8 h + decay ~0.4 h), peak 11,522 MiB -> NEED_GDN 14,600
  (measured peak + 3 GB), loss 3.10-3.53 at steps 1,150-1,350 (ctl tpp 3.49 @1000). ctl solo speeds for reference: tpp 77k, mamba2 113k.
- Environment: gdn in ~/.venvs/qllm_gdn = .venv's exact pins (torch 2.8.0+cu128, datasets 4.0.0, pyarrow 21.0.0) on Python 3.11 +
  fla 0.3.2 + einops; all other arms in .venv. (qllm311's datasets 3.6 / pyarrow 26 deadlocked the loader.)
- Sharing: runs beside Library's probes on the VRAM gate. Yields to the user's jobs and to SCI-1 milestone CHECKs (~/.qllm_check_wants_gpu).
  A yield is a SIGTERM with no save handler: the arm resumes from its last save (SAVE_EVERY 200 = ~13M tokens, ~3.5 min), the data
  cursor is exact at that save, so the redone steps see the same batches and every arm ends at the same token count (1.0B); only
  wall time is lost. Speeds are quoted from SOLO windows only (log lines "GPU SOLO / SHARED").
- Claim rule unchanged from the pre-registration above (c492727).
- 20:49 IST: chain restarted at 20:49 with the scheduler fix 90711f4 (a started, blocked arm is waited for, not skipped: gdn had yielded to the
  41B CHECK and the loop moved on to nochrono). Pinned 227f3eb with ALLOW_CODE_DRIFT=1: the drift since 227f3eb is only wio_chain.sh,
  the lineage tsv and three new Library probe files (v14/a32c.py, told_margin.py, check_layout.py) that no training path imports.
  Found scripts/require_clean_tree.sh deleted in the shared checkout (not by Mind; restored from HEAD), so the guard was silently off
  for the gdn start from ~20:15; the trainer code was at 227f3eb throughout (git diff checked).
- 22:00 IST: ENVIRONMENT CAVEAT. The ctl chain ran from ../qllm_ctl, whose .venv links ~/.venvs/qllm311 (datasets 3.6, pyarrow 26,
  mamba_ssm 2.2.5); all ctl arms shared it. wio runs from qllm2: pam / nochrono / tpp in .venv (Py3.9), gdn / mamba2 in ~/.venvs/qllm_gdn
  (.venv's exact pins on Py3.11 + fla 0.3.2 + mamba_ssm / causal_conv1d copied from qllm311; launcher default b0b2426). Within wio all arms
  share package pins (torch 2.8.0+cu128, datasets 4.0.0, pyarrow 21.0.0); a wio-vs-ctl number crosses environments and is labelled so.
  (Goes into wio_chain.sh's header at the next safe restart; not edited while the chain runs, since bash reads it incrementally.)
- 00:35 IST 2026-11-17: CHAIN STALL + FIX (Lead caught it). The gdn main ended cleanly at ~23:54 IST 11-16 (logs/v13_sempty_wio_gdn_main_
  89aff04_20261116_2108.log: "Training complete. steps=13733 tokens=900,005,888"), but the chain never wrote .done: segment() looked for
  the arm's log as logs/v13_sempty_<tag>_227f3eb_*.log (the pin $H) while logs are named after HEAD at launch (89aff04), so the done
  check grepped an empty path, read "crash?", and relaunched; each relaunch found "already at step 13733; nothing to do", exited inside
  the 90 s wait, and the "no trainer after 90 s" branch skipped the done check: a loop from 00:00 to 00:35 IST, card idle ~40 min.
  Fix (wio_chain.sh, this commit): tag_log() finds the arm's newest log under any launch hash; finished() is used both after a trainer
  exit and in the "no trainer after 90 s" branch. .done touched by hand after the check above; the 5 no-op relaunch logs deleted;
  chain restarted (same SCREEN_HASH=227f3eb). Env caveat moved into the chain header at this restart.
  DRIFT: the restart lists v13_sempty/train.py changed since 227f3eb. That change (4699c66 / b0b2426, 21:58 IST) only adds --tpp_geom
  (default '') and uses it inside the `arch == 'transformer'` branch; the gdn / pam / nochrono / mamba2 paths are byte-identical. The gdn
  main's weights came from ONE trainer process launched 21:08 IST at 89aff04 (before that change; it resumed from the step-200 save of
  the 20:22 start), which ran to step 13,733 without restart, so its weights were produced entirely before the change.
- 01:35-01:45 IST 2026-11-17: d_drive FULL + gdn READOUT FIXES (Lead; Argon found the readout issues, Lead verified (1) and (2)).
  d_drive hit 100 % (from_6000 copies); the gdn decay crashed at its final save 3x, then finished at 01:37 after 43 GB were freed
  (Library 17 GB of lib_ weights extractions; Mind 26 GB: four sci1_cont2/3 full saves whose weights equal their c2a/c2b/c3a/c3b
  weights files tensor by tensor, step_138500_rw.pt, four ctl/cx main latest.pt byte-identical to stable_final.pt). Free-space gate
  scripts/require_free_gb.sh (3c7ccb9): the chain refuses launches below 10 GB.
  (1) readouts() ran every arm in .venv, which has no fla / mamba_ssm: gdn / mamba2 probes now run in ~/.venvs/qllm_gdn.
  (2) fla's chunked GDN kernel refuses fp32, so EVERY in-training val of the gdn arm failed ("does not support float32": 14x main,
  4x decay) and the fp32 probes would have crashed. Fix (26fd8c6): outside an active autocast, GDNLM runs its backbone under bf16
  autocast and returns the hidden in fp32 (training path unchanged). Selftest on the decay latest, holdout 4 x 4096: fp32-call loss
  3.2404 vs autocast 3.2405, argmax agreement 0.989.
  gdn HOLDOUT NLL, COMPUTED AFTER TRAINING (POST_VAL=1: the trainer's own loaders and _val_loss, decay latest.pt, step 15,259, 1.000B
  tokens; logs/v13_sempty_wio_gdn_postval_26fd8c65_20261117_0143.log): val 3.3854 (ppl 29.53) | wiki 4.0657 | dclm 3.3955 | fineweb
  3.4312 | arxiv 1.7263 | pubmed 2.0862 | pes2o 3.1112 | cosmo 2.5242 | tools 0.7363 | pysci 1.1006 | chatmt 1.8812. The page says
  "computed after training" for this row.
  DTYPE NOTE: the trainer's val, run_learning_probe (s8) and run_memory_behavioral (t40) run the other arms in fp32 end to end; for
  gdn the backbone runs in bf16 (the kernel has no fp32 path) with the head / CE in fp32. clk2 (skillpack) and mc_eval are bf16
  autocast for every arm. The gdn s8 / t40 / NLL rows carry this label.
  (3) run_skillpack_probe tagged gdn as mamba2 (it has .net); now tagged gdn. Its last logits were right; --dilate_where echo is a
  no-op for gdn / mamba2 / tpp (it acts only on PAM's echo path), so their clk2 is the plain 8-dilated read: labelled so.
  Step-save safety: each probe writes <out>.part, moved into place only on exit 0 AND a parse check; a decay step save is deleted
  only after its json parses. Chain restarted 01:44 IST (readouts).
- 02:04 IST 2026-11-17: gdn ARM RESULT (readouts complete; Gated DeltaNet, fla 0.3.2, 102,508,944 params, 1.000B tokens = 0.9B
  wsd_stable + 0.1B decay, seed 43, SEQ 4096; outputs logs/wio/v13_sempty_*wio_gdn_227f3eb*):
  s8 @8k (mean of shift_wide_new@8192 and affine_map_new@8192): end (15,259) 0.444 | 15,000 0.408 | 14,500 0.374 | 14,000 0.265.
  The kept saves sit at absolute steps 14,000 / 14,500 / 15,000 + end (KEEP_EVERY 500 counts from step 0), not at decay +500 / +1000;
  read as: **3-save mean = the last three (14,500 / 15,000 / end) = 0.408**, the same rule for every arm.
  s8 by distance at end: shift_wide_new 0.93 / 0.56 / 0.35 / 0.16 at 0 / 2k / 4k / 8k; affine_map_new 0.98 / 0.92 / ... / 0.73.
  t40 needles: 2k 0.681 | 4k 0.544 | 8k 0.458 | 16k 0.325. clk2 (plain 8-dilated read; echo no-op): 8k 0.17 | 16k 0.06.
  HS / ARC-e / PIQA acc 0.2745 / 0.4402 / 0.5941 (acc_norm 0.2812 / 0.396 / 0.5762). Holdout val 3.3854 (computed after training).
  State at decode: 12 x 6 x 64 x 128 x 4 B = 2.36 MB (fp32, short-conv tails excluded). Speed SOLO: median 114,074 tok/s over steps
  10,750-13,700 (the solo window from 16:47 UTC; smoke 62k was at a shorter window with warm-up). ctl solo refs: T++ 77k, Mamba-2 113k.
  Dtype: s8 / t40 / val in bf16 backbone + fp32 head (other arms fp32); clk2 / mc bf16 for all. Told 2k / 8k: Library on "wio chain DONE".

## chat-off natural experiment: does chat share push told 8k down? (PRE-REGISTERED 16:04 IST 2026-11-16, before any number; Lead agreed)

**Observation that prompted it:** told 8k (Library's ptwin, n 100) fell 0.45 (27B, chat ~5 %) -> 0.36 (29B) -> 0.31 (30B, chat ~20 %)
while reach recovered. Segment 3's chat sources then ran dry one by one (tuluif 32.95B, chatsr 34.43B, smoltalk2_mid 38.46B, chatmt2
39.88B), so its last saves are a free chat-off test.
**Points:** 39B (step_595101: chat ~6 %, chatmt2 only, for ~0.5B), 40B (step_610360: 0 % chat for ~0.1B), 40.5B (seg3_end.pt:
0 % chat for ~0.6B). Each gets the 4090 CHECK (s8 spread, t40, skill pack, clk2, benchmarks) and Library's FAR-TOLD (told 2k / 8k,
a26 2k), same protocol as 29B / 30B.
**Reading rule (told 8k, the headline):** chat IMPLICATED if 40.5B reaches >= 0.45 (27B's level; +0.14 over 30B, ~2.8 sigma at n 100)
with s8 @8k and clk2 16k not lower than 30B by more than 0.05; NOT implicated if 40.5B <= 0.36; in between = unresolved, segment 3c's
41B / 42B milestones decide. Single-checkpoint jitter is real (26B 0.03), so a point counts only if 40B and 40.5B agree in direction.
**Action:** implicated -> propose a lower chat share for 3c with the freed share going to far rule docs (b2s_far / proto_far), argued
with Lead before any change; otherwise chat stays at 19.9 % (user: base model WITH chat).
**Confounds stated:** more tokens (30B -> 40.5B) and fineweb100 at a higher share also change between the points; the test cannot
separate "chat removed" from "web added".

### chat-off: CHECK cells (4090, 18:51 IST 2026-11-16; told 8k from Library's FAR-TOLD still pending, no verdict yet)

| point (chat share) | step | s8 @8k | skill @8k | clk2 16k 8x+c4 | clk2 32k-run @16k | t40 16k | HS / ARC-e / PIQA acc |
|---|---|---|---|---|---|---|---|
| 30B (~20 %) | 457,770 | 0.949 | 0.590 | 0.86 | 0.64 | 0.803 | 0.317 / 0.553 / 0.626 |
| 39B (~6 %) | 595,101 | 0.941 | 0.615 | 0.89 | 0.62 | 0.817 | 0.318 / 0.543 / 0.640 |
| 40B (0 %, ~0.1B) | 610,360 | 0.986 | 0.795 | 0.91 | 0.68 | 0.728 | 0.323 / 0.561 / 0.640 |
| 40.5B (0 %, ~0.6B) | seg3_end | 0.961 | 0.635 | 0.90 | 0.44 | 0.789 | 0.323 / 0.558 / 0.651 |

386M PAM; tokens = the point's label. Cells parsed the same way as the 29B / 30B table (s8 = mean of the two b2 @8192 accuracies, n 400
each; skill = skillpack A_state @8192, n 200; clk2 = clk2_echo free 8x+c4, n 50; t40 16k = t40_long all-cell mean). Outputs in logs/
(learning_probes, memory_probes, diag, sci1) under `sci1_{step_595101,step_610360,seg3_end}_91bdc49`.
**Reading so far:** the rule's side conditions hold (s8 and clk2 at 40.5B are not below 30B by > 0.05). 40B's skill @8k 0.795 is one
point; 40.5B is back at 0.635, so the two disagree and neither counts alone (the single-checkpoint jitter seen at 26B). The verdict
waits on told 8k.

### chat-off VERDICT: chat NOT implicated (19:28 IST 2026-11-16; told 8k by Library, FAR-TOLD ptwin, n 100)

| point (chat share) | told 8k | bare num+dot (of 100) | told 2k | a26 2k |
|---|---|---|---|---|
| 30B (~20 %) | 0.31 | 55 | - | - |
| 39B (~6 %) | 0.06 | 36 | 0.38 | 0.23 |
| 40B (0 %, ~0.1B) | 0.37 | 50 | 0.88 | 0.46 |
| 40.5B (0 %, ~0.6B) | 0.12 | 63 | 0.69 | 0.25 |

By the pre-registered rule: "implicated" needed 40.5B >= 0.45, and no chat-off point came near it (best 0.37). 40.5B = 0.12 is in the
"NOT implicated" band. 40B (0.37, +0.06 over 30B, inside ~±0.07) and 40.5B (−0.19) do not agree in direction, so 40B carries no
weight. **Verdict: removing chat did not bring told 8k back; chat stays at 19.9 %** (user: base model WITH chat).
**What it says instead:** told 8k swings 0.06 <-> 0.37 between saves 0.5B apart, while s8 @8k (0.94-0.99) and clk2 16k (0.86-0.91)
hold. Most misses are a FORMAT flip (bare "56.2." with no unit; Library e6b66b1; Argon / Lead): proto4k's 20 % no-protocol docs
(v14/protocol_docs.py:177, `has_proto = rng.random() >= 0.2`) teach the bare answer at 100x answer weight, while the far docs
(proto_far, all protocol) carry ~1/3 of proto4k's answer-token share (0.311 % vs 0.830 %, Argon) and ask 1-3 questions (line 253).
(Reading corrected 20:18 IST: a bare answer = the told rule forgotten at 8k, a reach failure; see "CORRECTION" under segment 3d.) Next: segment 3d below.

## b386: Transformer++ and Mamba-2 at SCI-1's size on SCI-1's early chain (PRE-REGISTERED 21:58 IST 2026-11-16, before any number; Lead accepted the shape)

**Why:** the final-model page needs a same-size, same-data, same-token row beside SCI-1 (386M PAM). Every T++ / Mamba-2 we have is ~102M
and none saw a SCI-1 mix except ctl (1.0B on cont2). A row at the release point (~43B) would cost ~7-9 days per baseline (extrapolated),
so the fair rows are the early milestones, where SCI-1 already has full CHECKs.
**Closest prior work:** our own ctl (102M, 1.0B, SCI-1's cont2 mix: PAM s8 0.523 vs T++ 0.074 / Mamba-2 0.149, holdout 3.427 / 3.440 /
3.378). **Difference:** ~3.8x the size, SCI-1's own chain from scratch, read at four token counts (a curve, not one point).
**Geometries (params within 1 % of SCI-1 PAM 385,817,038):** T++ d_model 1024, 24 layers, 16 heads (head 64), SwiGLU d_ff 3168 =
385,751,040 (-0.02 %; TPP_GEOM=1024,24,16,3168); Mamba-2 d_model 1024, 51 layers, d_state 128, headdim 64, expand 2 = 388,125,216 (+0.60 %;
MAMBA_D_MODEL=1024 MAMBA_N_LAYER=51). Plumbing b0b2426 (defaults unchanged: 101,916,864 / 102,613,960).
**Recipe = SCI-1's chain exactly** (sci1_env.sh / sci1_chain.sh): stage A 0.5B (SOURCES_A, warmup 2,000, LR 8e-4 constant), then stage B
on the stage-B mix with a 500-step re-warm to 4.0B, constant LR, no decay; SEQ 4096, 65,536 tokens per step, seed 43, ANSWER_W 100,
decontam v1, RESET_AT_EOS, ANSWER_IN_WINDOW. Micro-batch may differ (grad accum) to fit the card; tokens per step do not.
**Readouts at M1 / M2 / M3 / M4 (steps 15,259 / 30,518 / 45,777 / 61,036 = 1 / 2 / 3 / 4B), with the same CHECK as SCI-1:** holdout
per source; HS (acc and acc_norm) / ARC-e / PIQA; s8 spread (0 / 2k / 4k / 8k); skill pack; clk2; t40 to 16k; told (Library FAR-TOLD).
PAM side = SCI-1's existing CHECK at those steps (M1 s8 D 0.322, M4 s8 D 0.782, M4 holdout 3.218), rerun where a cell is missing.
**Claim rule (pre-stated):** "PAM learns rules the baselines do not, at equal size and data" stands if PAM's s8 @8k beats BOTH baselines
by >= 0.10 at >= 3 of the 4 milestones, and at M4. Language: PAM's M4 holdout within +0.03 of the better baseline counts as "matched"
(bar 2 tolerance); worse by > 0.03 is stated next to any rule claim. Benchmarks are reported, not bars. Params and tokens in every row.
**Limits stated:** (superseded by the AMENDMENT below: an LR check per baseline is added; PAM's 8e-4 was width-scaled, not tuned.)
Rows end at 4B; beyond that the page uses a reference block with no claim.
**AMENDMENT (22:00 IST 2026-11-16, Mind + Lead, before any b386 run): LR check per baseline, and a correction.**
- Correction: SCI-1's LR 8e-4 was NOT tuned at 386M. It was width-scaled from the 103M runs' 1.2e-3 (~1 / width; "no screen at this
  size, stated", SCI-1 pre-registration above). "PAM's tuned LR" in Limits is wrong; PAM gets no re-tuning either way.
- Prior evidence (102M, 100M tokens, phase-3b mix, not this mix): T++ screen 2e-4 / 6e-4 / 1.2e-3 -> 4.5543 / 4.1379 / **4.0733**
  (tflr @e072a8f); Mamba-2 screen 2e-4 / 6e-4 / 1.2e-3 -> 1.2e-3 (m2lr @3f69057). Both optima sat at the TOP edge of the grid, so the
  true optimum may be higher. Width-scaled to d 1024: T++ 1.2e-3 x 672/1024 = 7.9e-4, Mamba-2 1.2e-3 x 768/1024 = 9.0e-4. A short screen
  also once picked the wrong LR for PAM (the 100M screen, "Scoreboard result" 2026-10-29: short screens favour higher LRs).
- **Check:** per baseline, stage A only (0.5B, SOURCES_A, warmup 2,000, constant LR) at 5.7e-4 / 8e-4 / 1.13e-3 / 1.6e-3 (sqrt-2 steps
  around the width-scaled optimum, one step past the old grid's top edge). Pick = lowest stage-A holdout at 0.5B; a HIGHER LR must win
  by > 0.01 nats (the old screens' guard against short-screen bias), else the lower one. If 1.6e-3 wins, add 2.26e-3 before choosing.
  Continue only the pick through stage B to 4B. The non-picked arms are reported (holdout at 0.5B) in the b386 table.
- Cost: 4 x 0.5B per baseline = 2B tokens each, added to the 4B run (estimate when the smoke gives speeds).
**NOTE (22:05 IST 2026-11-16, Argon via Lead; accepted points, before any b386 run):**
- (A) Mamba-2 geometry: the published Mamba-2 370M is d_model 1024 x 48 layers, d_intermediate 0, tied embeddings (verified:
  huggingface.co/state-spaces/mamba2-370m config.json; Mamba-1 370M the same). Ours, 1024 x 51 = 388.1M, is that shape + 3 layers to match
  params; cite it so. Argon's alternative 1280 x 31 (381.4M, -1.15 %, the same d / L ratio as our 102.6M 768 x 17) is smoked too. **Decision
  rule, set now:** keep 1024 x 51 unless its measured tok/s is below 0.75x that of 1280 x 31 at the same micro-batch; decided at the smoke,
  before the LR screen, and never changed after.
- (B) LR: covered by the 4-point screen above.
- (C) Mix: M1-M4 contain no far-block docs for PAM either (the far block starts at 11B), so b386 is like-for-like on data. The far-block
  comparison at equal data is carried by wio at 103M; the release row (~43B) has NO same-size baseline, stated on the page.
- (D) Decay: M1-M4 compare constant-LR saves for all three bodies. Optional, priced with the rest (Lead leans yes if <= ~3 h in total): a
  0.25B decay branch from M4 for all three (PAM from sci1_main step_061036 on d_drive), same decay schedule, read with the full CHECK, to give
  a decayed same-size row for benchmarks.
**Cost / order:** geometry smoke on the 4090 between wio arms (b386_smoke, tmux mindb386, after the gdn arm's readouts) measures params,
peak VRAM and tok/s; the run's box and time are the user's call through Lead (after the release decay, or earlier).

**SMOKE (03:21 IST 2026-11-17, 4090 SOLO, stage-A recipe: LR 8e-4, 65,536 tok/step, 12 min per arm; logs
logs/v13_sempty_b386_{tpp,m2,m2w}_smoke_2261be1a_20261117_*.log, driver logs/v13_sempty_b386_smoke_driver_2261be1a_20261117_0244.log):**
| arm | params | steady tok/s | peak MiB (nvidia-smi) |
|---|---|---|---|
| tpp 1024x24, 16 heads, d_ff 3168 | 385,751,040 | 27,800 | 20,508 |
| m2 1024x51 | 388,125,216 | 32,620 | 17,354 |
| m2w 1280x31 | 381,386,120 | 34,300 | 15,228 |
All three train (loss falling, 0 errors). GEOMETRY RULE APPLIED: 1024x51 is kept (32,620 >= 0.75 x 34,300 = 25,725). The first smoke
(02:06-02:43) died on FINEWEB_LOCAL_DIR=data/fineweb-edu (a 6000-only path), dropped on the 4090; the 6000 runs keep it.
PRICE on the 4090 at these speeds (stage-A 4-LR screen = 2B; main = the chosen LR's stage A continued to 4B = 3.5B more):
tpp 20.0 h + 35.0 h = 55 h; m2 17.0 h + 29.8 h = 47 h; together ~102 h (~4.2 days of the card). Optional 0.25B decay from M4: tpp 2.5 h,
m2 2.1 h, plus PAM's (386M PAM on the 4090 not measured; 6000 speed applies if run there): ~4.6 h + PAM, over Lead's ~3 h lean.
tpp at 20.5 GB peak needs the whole 4090 (no neighbour under the + 3 GB rule); m2 17.4 GB leaves ~3 GB. The 6000 would be faster.
**PLACEMENT (PROVISIONAL, Lead 03:21 IST 2026-11-17, pending the user's word; nothing launches before wio finishes):** the 4-LR
screen stays as amended (no cut). (1) After wio's last arm, the 4090 runs the stage-A LR screens (tpp ~20 h, m2 ~17 h), resumable,
yielding to milestone CHECKs and the user's jobs; Library's probes fit between arms. (2) The 3.5B main continuations go on the 6000
after SCI-1's release decay if the user agrees, else on the 4090. (3) The optional 0.25B decay branch is decided after the mains.
A 5-min 6000 speed smoke of the 386M tpp / m2 runs only in a gap that does not disturb 3d (the card is full during 3d), likely the
release-decay gap.

## SCI-1 segment table (21:58 IST 2026-11-16; steps / tokens from the checkpoints' own global_step / global_tokens, 65,536 tokens per step from step 0)

| # | run dir | global steps | tokens | parent | mix | line |
|---|---|---|---|---|---|---|
| 1 | sci1_stageA_59c3cf5 | 0 -> 7,630 | 0 -> 0.50B | scratch | SOURCES_A (10 plain sources), warmup 2,000, LR 8e-4 constant, no rule docs | main |
| 2 | sci1_main_59c3cf5 | 7,630 -> 61,036 | 0.5 -> 4.0B | stageA_final | stage-B mix (15 sources incl. think, b2s 2.3 %), 500-step re-warm; stopped at M4 | main |
| 3 | sci1_cont1_59d098d | 61,036 -> 76,295 | 4 -> 5B | main step_061036 | cont1: think out, protoc in | main |
| 4 | sci1_cont2_e61cd92 | 76,295 -> 106,813 | 5 -> 7B | cont1 step_076295 | cont2 (the ctl mix) | main |
| 5 | sci1_cont3_449e320 | 106,813 -> 137,331 | 7 -> 9B | cont2 step_106813 | cont3 (14 sources, protoc out) | main |
| 6 | sci1_cont4_8284510 | 137,331 -> 152,590 | 9 -> 10B | cont3 step_137331 | cont3 + worked ~2 % (M10-worked, failed) | side |
| 7 | sci1_decay_m9_8290f7d | 137,331 -> 152,590 | 9 -> 10B | cont3 step_137331 | cont3, LR -> 0 | side |
| 8 | sci1_c3_95498f3 | 137,331 -> 141,146 | 9 -> 9.25B | cont3 step_137331 | rule block x2, 0.25B decay | side |
| 9 | sci1_long_3c2bf1f | 137,331 -> 167,849 | 9 -> 11B | cont3 step_137331 | cont3, 500-step re-warm | main |
| 10 | sci1_c2_1b8607d | 152,590 -> 156,405 | 10 -> 10.25B | long step_152590 | rule block x2, 0.25B decay | side |
| 11 | sci1_long2_4a97ca7 | 167,849 -> 427,252 | 11 -> 28B | long step_167849 | cont3 + far block ramped over 2B (b2s_far 1.2 %, proto_far 0.8 %); best_model = step 344,000 (22.5B) | main |
| 12 | sci1_long3_91bdc49 | 427,252 -> 618,000 | 28 -> 40.5B | long2 step_427252 | fineweb100 + chat ramp 5.4 -> 19.9 % over 2B; chat sources ran dry 32.95-39.88B | main |
| 13 | sci1_long3c_3a8d7fb | 618,000 -> 625,619 | 40.5 -> 41B | long3 seg3_end | ramp-target mix + CYCLE_SOURCES (chat pass 2) | main |
| 14 | sci1_long3d_cf9dee1 | 625,619 -> running | 41B -> | long3c step_625619 | proto_far 1.6 % (from dclm / fineweb100) | main |

Main line: 1 -> 2 -> 3 -> 4 -> 5 -> 9 -> 11 -> 12 -> 13 -> 14. Mixes: v13_sempty/tmp/sci1_*env.sh at each segment's hash. Checkpoints on d_drive
(from_6000/checkpoints_v13_sempty/<run dir>/); the 6000 keeps only the running segment's saves and best_model.pt files.

## segment 3d: proto_far 0.8 % -> 1.6 % (PRE-REGISTERED 19:28 IST 2026-11-16, before any number; Lead agreed option 1, GO on the 6000 change pending)

**Mechanism (stated):** the told 8k misses are mostly the bare-number format, not lost reach. Far protocol docs are the only far
source that teaches "report with the protocol's unit at distance", and they are 2.7x thinner in answer tokens than proto4k (whose
20 % no-protocol docs teach the bare form). Doubling proto_far doubles the far protocol answer signal without touching chat.
**Caveat (stated):** if chat competes for capacity, less chat is the direct lever and more proto_far only the indirect one; the
chat-off test above did not show chat as the cause, and the user's direction keeps chat at 19.9 %.
**Closest prior work:** data-mixture optimisation (DoReMi, Xie et al. 2023; Data Mixing Laws, Ye et al. 2024). **Difference:** one
share change aimed at one probe-measured skill family (rule told in context, answered 8k tokens later, beyond the 4k training
docs), judged by that probe plus its format split, not by perplexity.
**Change:** weights only. proto_far 0.00800 -> 0.01600; dclm 0.14997 -> 0.14597; fineweb100 0.15615 -> 0.15215. Chat 19.9 %,
b2s_far 1.2 % (s8 saturated), everything else, CYCLE_SOURCES, decontam v2 unchanged. Fixing 3 questions per far doc (+50 % answer
density) is NOT in this step (it would change the generator under resumed cursors); a later, separate step if needed.
**When / how:** segment 3c runs to its 41B milestone (step 625,619 = the with-chat baseline point under 3c's mix); segment 3d
continues from that save (continue_from, cursors kept, WARMUP_C=0), like 3b -> 3c.
**Readouts:** the usual 4090 CHECK + Library FAR-TOLD at 41B (baseline), 42B, 43B.
**Rule (pre-stated):** WORKS if the mean told 8k of 42B and 43B >= 0.45 and >= 41B + 0.10, with bare num+dot mean <= 35, s8 @8k >= 0.94,
clk2 16k >= 0.85, and HellaSwag acc not below 41B by > 0.01. NO EFFECT if that mean <= 41B + 0.03: revert the weights and pre-register
the 3-question generator step. In between: run to 44B and decide on three points. Points that disagree in direction (40B vs 40.5B
kind) are reported, not read alone.
**Confounds:** chat cycling (pass 2) and more tokens change together with the share; the 41B baseline is under 3c's mix to limit this.
**unit4 basis (Library c7b0f84, 9 points; Lead's GO condition, confirmed by Lead 20:03 IST):** the stop rule hides no unit (after_dot = 0.000 at every
point) and unit_after equals the told 8k headline at every point, so the missing unit is a real model format choice, not a scorer
artifact. The bare " 56.2." answers are complete answers in the no-protocol format that proto4k's 20 % no-protocol docs train
(protocol_docs.py:177); 39B also gives 41 % sentence answers. **CORRECTION (20:18 IST, Library):** the 20 % no-protocol proto4k docs are the probe's CONTRAST condition ("no protocol stated ->
answer the raw value"). A bare " 56.2." at 8k is therefore the model answering as if NO rule had been told: the told rule did not
survive 8k tokens. It is a REACH failure of the told rule, not a competing format habit, and those docs must stay (they make the told
cell meaningful). The "format flip" readings in the chat-off verdict and in the mechanism line above are withdrawn in that sense:
told 8k does measure reach, specifically the reach of a TOLD rule; s8 (a rule shown by demos) holding at 0.94-0.99 while told 8k swings
0.06-0.37 says told-rule reach is the weak, unstable part. 3d still targets it directly: proto_far is the only source that trains
"rule told, used thousands of tokens later", and it is 2.7x thinner in answer tokens than proto4k. The chat-off verdict (chat NOT
implicated) stands; it rests on the numbers, not on the reading.
Two refinements (Lead): only 39B's sentence answers (41 %) are a separate FORMAT effect; and told 2k swings too (0.38 / 0.88 / 0.69
at 39B / 40B / 40.5B), so the told-protocol signal sits near the decision edge even at trained distance. In short: told-rule carry is
weak and flips by checkpoint, while example-rule reach (s8) and recall (clk2) hold. The WORKS / NO EFFECT rule is unchanged.
**TOLD-MARGIN (Library b47afc1, teacher-forced, told 8k twin):** the told unit is still picked above chance (0.27) at every point,
0.48-0.59 at the low points (M11, 39B, 40.5B; 26B void) vs 0.69-0.74 at 30B / 344k / 40B, but its margin over '.' (the no-protocol
answer) collapses there: gate_win 0.06-0.14 vs 0.39-0.70. So the low points are PARTIAL carry with a weak margin: not a lost rule, not a
pure gate flip. A gate-only fix would cap near 0.5; carry strength is the target, which more far-distance supervision (3d) addresses.
**Added secondary readout for 3d (reported, not a bar; the rule above is unchanged):** unit pick and gate_win at 41B / 42B / 43B. If 3d
WORKS by the rule, gate_win should rise with it; a told 8k gain with flat gate_win is reported as unexplained.

**CLARIFICATION of the 3d guards (21:34 IST 2026-11-16, Mind + Lead; recorded BEFORE any 3d number exists; the original rule text above is unchanged):**
the 41B baseline landed on both guard floors (clk2 16k 0.85 = floor 0.85; s8 @8k 0.941 vs floor 0.94), so single-save jitter could trip
them with no effect from 3d. Read them as: clk2 16k = the 42B / 43B MEAN >= 0.80 (41B − 0.05; n 50, SE ~0.05); s8 @8k = the 42B / 43B
MEAN >= 0.91 (41B − 0.03; n 800, SE ~0.008, last six saves 0.941-0.986). The told 8k bars, bare count and HellaSwag guard are unchanged.
**3d baseline, 41B (3c mix, step 625,619, CHECK done 21:02 IST):** s8 @8k 0.941 | skill @8k 0.610 | clk2 16k 0.85 | clk2 32k-run @16k 0.52 |
t40 16k 0.919 | HS / ARC-e / PIQA acc 0.319 / 0.554 / 0.628 (386M PAM, 41.0B tokens). Told 8k + TOLD-MARGIN: Library (m41far).
**41B told (Library m41far, 41ba344, logs/v14_fartold_m41far_*_41ba344_20261116_2156.jsonl + v14_told_margin_m41_41ba344_20261116_2156.jsonl):**
told 2k 0.84 | told 8k 0.35 (bare count 45, verbose 2) | a26 2k 0.43 | TOLD-MARGIN 8k: unit pick 0.68, gate_win 0.46 (noprot 0.34; chance
0.27). So the 3d bars resolve to: WORKS needs the 42B / 43B told 8k mean >= 0.45 (both 0.45 and 41B + 0.10 = 0.45) with bare count <= 35;
NO EFFECT if <= 0.38; otherwise decide at 44B.
**3d @ 42B (step 640,878, 42.0B tokens; CHECK 04:30-04:51 IST 2026-11-17 on the 4090, outputs logs/*/v13_sempty_*sci1_step_640878_cf9dee1*):**
| cell | 41B (3c) | 42B (3d) | 43B (3d) |
|---|---|---|---|
| s8 @8k | 0.941 | 0.926 (shift_wide 0.895, affine 0.958) | **0.984** (shift_wide 0.985, affine 0.983) |
| skill @8k | 0.610 | 0.735 | **0.845** |
| clk2 16k | 0.85 | 0.87 | 0.84 |
| clk2 32k-run @16k | 0.52 | 0.66 | 0.60 |
| t40 16k | 0.919 | 0.919 (t40 8k 0.969 vs 1.0) | **0.978** (t40 8k 1.000) |
| HS / ARC-e / PIQA acc | 0.319 / 0.554 / 0.628 | 0.318 / 0.559 / 0.635 | 0.317 / 0.566 / 0.636 |
| told 2k / 8k (Library jsonl, Lead's read; Library's report pending) | 0.84 / 0.35 (bare 45) | 0.66 / 0.09 (bare 70) | pending (m43far, next window) |
Verdict waits for 43B by the rule. To reach WORKS the 43B told 8k would need >= 0.81; to avoid NO EFFECT, > 0.67.
**43B CHECK (step 656,137, 43.0B tokens; 12:18-12:40 IST 2026-11-17 on the 4090; outputs logs/*/v13_sempty_*sci1_step_656137_cf9dee1*,
recorded 12:41 IST):** the far / skill / needle cells rise (s8 @8k 0.984, skill @8k 0.845, t40 16k 0.978), clk2 flat-to-down (0.84 / 0.60),
benchmarks flat. **Collateral scan (Lead (3)):** generations (6 prompts, greedy + sampled) keep their 42B formats; units appear only where
the prompt is about measured quantities (4 lines at both saves: "12.347 g", "78.4 C"); no unit on the chat / definition answers. Two
sampled continuations past <|endoftext|> fall into a repeat loop ("extreme extreme ...", "binbinbin...") vs 42B's template drift
(sampling past the end, not the answer). The skill pack scores number tokens only (no text), so it cannot show a unit; its A_free @8k
rose. The no-protocol false-unit question is Library's fu line (43B told-margin), not visible in these probes. Original 42B/43B told 8k
rule: pending m43far.
**44B CHECK (step 671,396, 44.0B tokens; 19:4x-20:03 IST 2026-11-17 on the 4090; outputs logs/*/v13_sempty_*sci1_step_671396_cf9dee1*,
recorded 20:04 IST):** s8 @8k **0.977** (shift 0.975, affine 0.978; 43B 0.984), skill @8k **0.735** (43B 0.845), t40 16k **0.808** (43B
0.978; t40 8k 0.972), clk2 0.79 / 32k-run @16k 0.48 (43B 0.84 / 0.60), HS / ARC-e / PIQA 0.316 / 0.553 / 0.643 (43B 0.317 / 0.566 /
0.636). The far rule holds; needles / skill / clock fall back to about the 42B level (single save; the 43B values were the high point).
Generations: units only on the measured-quantity prompts (protocol "12.347 g", tool "78.4 C", "2.0 L" restated from the prompt); chat
answers unit-free; formats as at 42B / 43B. Told 8k + fu: Library's 44B TM (this window).

**3d ADDENDUM: powered readout (Mind + Library + Lead, 05:38 IST 2026-11-17; MADE AFTER SEEING 42B: told 8k 0.09, twin M_gate -2.69,
C 0.42; recorded before any snapshot is scored).** The ORIGINAL 42B / 43B rule above is still evaluated and reported as written (it will
probably read NO EFFECT; that is not hidden). The addendum decides what we DO: 3d runs to 44B regardless of the original verdict, then
this rule is applied.
*Why:* single-save told 8k swings by about +-0.3 between neighbouring saves; a 2-point mean cannot resolve a 1-2B effect. Library's
decomposition (1c38ed70) of TOLD-MARGIN at 8k (per doc, n 100): twin gate = B + 5C, with B = noprot M_gate (the body's no-protocol
'.'-vs-unit prior) and C = mean(held - noprot) M_gate (the carry contrast). Mind's independent recomputation from the same jsonl
(identity residual 0.00 at every save): pre-3d (30 / 39 / 40 / 40.5 / 41B) C = 0.479 / 0.352 / 0.545 / 0.318 / 0.494, mean 0.438,
between-save SD 0.097 (within-save SE 0.02-0.03); B = -3.09 / -4.34 / -2.91 / -3.70 / -2.60, mean -3.33, SD 0.69. 42B: C 0.420,
B -4.79. Spearman with told 8k over the 6 saves Mind has told values for: B 0.89, C 0.83 (Library, 10 saves: 0.96 / 0.73). Low-C saves
co-occur with low-B ones (39B, 40.5B), so the swing is mostly B but not only B.
*Saves:* 3d = 9 saves at every 0.25B, 42.0-44.0B (milestones 42 / 43 / 44B + six snapshots of the 6000's latest.pt at steps >= 644,693
/ 648,508 / 652,322 / 659,952 / 663,767 / 667,581, v13_sempty/tmp/sci1_3d_snapshots.sh: hard links, free-space gates on both ends,
latest.pt never touched; a snapshot skipped for disk is noted, not replaced). Baseline = the 5 pre-3d saves (UNEVEN: 30B + four in
39-41B; bimodal in B). Each save: TOLD-MARGIN at 8k (n 100) + told 8k (n 100), Library, on the 4090 between wio arms.
*PRIMARY (merged with Library's text 05:39 IST, still before any snapshot is scored; supersedes the 05:38 draft line):* C = per-doc
mean(held - noprot) M_gate at 8k (TOLD-MARGIN, train wording, A28 docs seed 26000; v14/told_margin.py at the hash in each log name),
n 100 per snapshot, n 300 at the 42B / 43B / 44B milestones. Reference = the pre-3d saves 39B / 40B / 40.5B / 41B (contiguous with 3d;
30B dropped as 9B earlier): C 0.352 / 0.545 / 0.318 / 0.494, mean 0.427, SD 0.110. dC = mean(3d, 9 saves) - 0.427. WORKS: dC >= +0.10;
NO EFFECT: |dC| <= 0.05; WORSE: dC <= -0.10; in between = unresolved -> 4 more 0.25B saves to 45B. POWER stated, not tuned: SE of dC
~0.06 (Welch from save-to-save SDs), so a +0.10 result is ~1.7 SE; the SE is reported beside dC. Guards (s8 @8k, clk2 16k, HS, bare
count) on the 42 / 43 / 44B milestone means, as in the clarification.
*SECONDARY (reported, not bars):* twin M_gate delta (3d mean - the 39-41B mean -1.25; "WORKS-like" if >= +1.0 nats and >= 2 SE); B mean; the fraction
of saves in the high state (twin M_gate > -1.0): baseline 2/4 (39-41B), 3d k/9; gate_win, unit pick; told 8k mean (headline, single-save noisy).
*Data-cycle check (own line, Lead):* each snapshot logs its cumulative data counters (chat passes, per-source tokens + docs); the
window mix = the diff of consecutive saves. Test: does B (or C) move with the last-0.25B share of no-protocol proto4k docs or of
bare-number answers (cosmo / finemath / tools)? If B aligns with a data cycle, that is a EUREKA for the release recipe (average or pick
saves by state).
*Addendum series (Library TOLD-MARGIN at 8k; C = carry contrast, B = noprot '.' prior):*
| save | step | n | C (se) | B | twin unit / gate_win | tau0 pred told 8k | log |
|---|---|---|---|---|---|---|---|
| 42B | 640,878 | 300 | 0.453 (0.016) | -5.02 | 0.69 / 0.13 | 0.69 | logs/v14_told_margin_3d_640878_n300_4db71183_20261117_0541.jsonl (Library repo) |
| 42.25B snap | 645,000 | 100 | 1.092 (0.079) | -2.97 | 0.81 / 0.73 | 0.81 | v14_told_margin_3d_644693_n100_9d5f43de_20261117_1141.jsonl (noprot gate_win 0.11) |
| 42.5B snap | 649,000 | 100 | 0.588 (0.037) | -3.86 | 0.73 / 0.29 | 0.73 | v14_told_margin_3d_648508_n100_9d5f43de_20261117_1146.jsonl (noprot gate_win 0.00) |
| 42.75B snap | 652,500 | 100 | **3.644 (0.176)** | **+0.63** | 0.98 / 1.00 | 0.98 | v14_told_margin_3d_652322_n100_9d5f43de_20261117_1150.jsonl (**noprot gate_win 0.55**, unit_acc 0.34 at chance 0.27) |
| 43B | 656,137 | 100 | 2.452 (0.136) | -2.03 | 0.94 / 0.94 | 0.94 | Library 07:16:45Z line (fu 8k **0.18**, 2k: C 9.884 (0.526), B -2.30, fu 0.14) |
| 44B | 671,396 | 300 | 0.471 (0.016) | -4.29 | - / 0.18 | - | Library (20:52 IST): 2k C 1.810, B -5.01; fu 8k / 2k 0.01 / 0.01. C swings (0.45, 1.09, 0.59, 3.64, 2.45, 0.47), no climb |
**SERIES REDUCED (18:10 IST 2026-11-17, Lead):** the 43.5B / 43.75B snapshots (targets 663,767 / 667,581) were MISSED: the 6000 link
dropped 12:46-14:33 IST (a VPN on the 4090 box took all routes; the 6000's firewall drops the new exit IP) and the 4090 was rebooted
at 15:53 IST, killing the copier; by 18:05 the run was at 668,350. The 43.25B point is the late snapshot step 661,500 (43.35B,
moved aside 14:33, copied from 18:05). So the addendum's 9-save series has at most **7 points** (42, 42.25, 42.5, 42.75, 43, 43.35,
44B); the verdict rule is applied to those 7, stated as such. HARM and SCOPE rules unchanged.
**SERIES = 8 POINTS (Lead, recorded 19:02 IST 2026-11-17, BEFORE the point is scored):** a copier bug (it ignored the .skipped markers)
took an extra snapshot at step **668,500 = 43.80B** (18:09 IST). It is INCLUDED in the dC mean as an off-grid point, because it was
selected by the bug, not by any result. Series: 42 / 42.25 / 42.5 / 42.75 / 43 / **43.35 (off-grid)** / **43.80 (off-grid)** / 44B. On
d_drive it is snap_668500.pt (weights-only, marker .snap_668500.extra; target 663,767 stays skipped). All d_drive snapshots are
weights-only since 19:00 (295 tensors each verified torch.equal to the full file before the swap; Lead: d_drive was at 11 GB).
(Snapshot names: Library's targets 644693 / 648508 / 652322 = the first latest.pt at or past them, steps 645,000 / 649,000 / 652,500.
Kept on d_drive: keep_snap_645000_C1092.pt, keep_snap_652500_C3644_falseunit.pt (the false-unit test case); 649,000 deleted.)

**HARM READOUT (Lead, 11:58 IST 2026-11-17; ADDED AFTER SEEING 42.75B, dated as such):** at 42.75B the carry jumps (C 3.64) but with
NO protocol the body prefers a unit over '.' in 55 % of docs, at chance unit identity: 3d teaches "far rows + question -> a unit"
unconditionally (proto_far is 100 % protocol docs; the only no-protocol contrast is proto4k's 20 % at short gaps). Pre-3d reference
noprot gate_win 0.04; the only earlier spike was 27B at 0.34 (reverted).
*Readout:* noprot gate_win at 8k and 2k at every save. *Trigger:* >= 0.30 at TWO consecutive saves -> 3d is HARMFUL regardless of dC;
the mix changes at the next save. 42.75B = first (0.55); the 43B TOLD-MARGIN (2k + 8k) runs in today's 4090 window, right after the 43B CHECK.
*Remedy:* proto_far v2 (Library: 25-30 % no-protocol docs + other-quantity protocols) entering as a NEW source `proto_far2` (own
cursor and rng, whole-window packing), segment 3e env, READY only after (a) packer bit-identity without proto_far2, (b) Library's
check() (5-gram decontam, answer share, realized no-protocol share), (c) a CPU smoke showing the realized share. *Fallback* if v2
has not passed when the trigger fires: segment 3r = 3c's weights (proto_far 0.8 %), `v13_sempty/tmp/sci1_long3r_env.sh` via the
generic `sci1_seg_switch.sh` (waits for the next latest.pt, stops sci1s3d, starts from that save). Nothing launches without Lead's GO.
*Collateral check (43B CHECK):* skill-pack answers, generations and chat probes scanned for spurious units / changed answer formats
vs the 42B CHECK; HS / ARC as usual.
*3e READY state (12:04 IST 2026-11-17):* hook f2fcd675 (v7/data.py `proto_far2` -> protocol_text_iter(far=True, far_version=2),
whole-window, own cursor) + `v13_sempty/tmp/sci1_long3e_env.sh` (3d's weights, proto_far -> proto_far2 at 1.6 %). (a) packer: 400 chunks
identical to HEAD, and proto_far's stream identical HEAD vs hook at skip 0 / 1000 (test_proto_far2_hook.py); proto_far2 own cursor +
exact resume. (b) Library check(far_version=2): leak-free, held-out hits 0, shared 5-grams with A26 0, answer share 0.0012 (= v1),
modes own 64.7 / none 28.2 / other 7.2 %. (c) realized share: v2 docs 2,600 +- 275 tokens vs v1 2,584 +- 286 (400 docs each), so the
token share at 1.6 % equals 3d's proto_far share; the trainer's [mix] line at 3e's first val confirms it in the log. Library's bars if
it enters: fu 8k <= 0.10 within 0.5B and <= 0.05 by 1B; mean C 8k >= 0.427; twin unit_acc >= the pre-switch mean - 0.05.
**DECISION RULE at 43B (Lead, recorded 12:05 IST 2026-11-17, BEFORE the 43B fu number exists):** fu = noprot gate_win at 8k in
Library's line "3d 656137 told_margin8k n ... fu x".
(A) fu 8k at 43B >= 0.30 -> trigger fired; Lead's GO for 3e given in advance: Mind launches sci1_seg_switch.sh (3d -> 3e) at the next
latest.pt save without waiting, then reports. 3r is the fallback only if 3e fails to start or its first-val [mix] line is wrong.
(B) fu 8k at 43B < 0.30 -> no switch; 3d continues per the addendum (to 44B, 9 saves), fu watched at every snapshot, trigger armed
(any two consecutive saves >= 0.30). At the end of the addendum window 3e replaces 3d anyway if dC WORKS (v2 is the better-designed
source), as a separate GO.
**43B RESULT of the rule (12:48 IST 2026-11-17): (B).** fu 8k 0.18 < 0.30 (2k 0.14): no switch; 3d continues to 44B; 3e stays armed (trigger: any two consecutive saves >= 0.30; fu above the pre-3d 0.04, watched at every snapshot).
**SCOPE SWITCH RULE (Argon 12:49 IST, adopted by Lead; recorded 12:50 IST 2026-11-17, BEFORE any scope number exists):** switching off
3d is gated on Library's four-cell scope probe (41B / 42.75B / 43B, next 4090 window), not on the clock. Cell S = the held-out direction
(asked volume, mass protocol), plain twin, false-unit rate at 2k, on 43B.
- S-cell false-unit 2k (43B) >= 0.50 -> switch at the next save to the scope-aware sources: segment **3f** = 3d's mix with
  proto_far -> proto_far3 and proto4k -> proto4k2 at the same shares, own cursors (not v2).
- < 0.50 AND HARM (fu 8k) stays < 0.30 -> finish the addendum to 44B, then replace 3d by 3f in one step.
- 3e (v2) stays armed only for the HARM trigger. Nothing launches on the 6000 without Lead's GO; pam has priority on the 4090.
*3f READY (12:54 IST 2026-11-17; 9753ce39):* registry proto_far3 (far '3', whole-window) + proto4k2 (scope '2', max_gap 300), own
cursors; `v13_sempty/tmp/sci1_long3f_env.sh` = 3d's weights (token shares) with proto_far -> proto_far3 (1.6 %), proto4k -> proto4k2
(0.846 %). (a) packer 400 chunks identical; proto / proto4k / proto_far / proto_far2 streams identical HEAD vs hook; own cursors, exact
resume. (b) Library check(): leak-free, held_out_hits 0, 0 shared 5-grams (far3 answer share 0.0012, 4k2 0.0032); my count: held-out unit
direction 0 / 20,000 in both; modes far3 own .549 / none .251 / other .200, 4k2 own .447 / none .353 / other .200. (c) tokens far3 2,593
vs proto_far 2,588; 4k2 914 vs proto4k 992 (weights are token shares, v7/data.py l.1751, so the share holds; the [mix] line confirms).
Launch = sci1_seg_switch.sh with the 3f env from a 6000 worktree at >= 9753ce39, on Lead's GO.
**3f GO + LAUNCH (20:17 IST 2026-11-17, Lead GO by the scope rule fe241a0c / 71127a3c):** 43B scope tuple (Library): scope_fu 2k
**1.00** (>= 0.50), 8k 0.94; S_bound_fu 0.88 / 0.64; precedence u1 over the row's unit 95 % at 2k; P_fu (any unit, no protocol) 0.42 /
0.50. Switch launched 20:17 IST on the 6000 (worktree qllm_sci1e @ 91a87ed, tmux sci1s3f, sci1_seg_switch.sh + sci1_long3f_env.sh): waits
for 3d's next latest.pt, stops sci1s3d, keeps it as seg3d_end.pt, starts 3f with all cursors.
**3f BARS (Library pre-registration 4f28acfa + Lead):** scope_fu 2k <= 0.30 within 0.5B and <= 0.15 by 1B; C 8k >= 0.427; P_fu 8k <=
0.10; twin unit_acc not down by more than 0.05. **Guards vs the 44B CHECK** (as 3d): s8 @8k 3-save mean >= 0.977 - 0.05; HS / ARC-e
not below 44B by > 0.01; **clk2 16k 3-save mean >= 0.78** (floor SET AT 44B: clk2 slid 0.87 -> 0.84 -> 0.79 over 42 / 43 / 44B).
Snapshots weights-only at +0.25 / +0.5 / +0.75 / +1.0B from the start (slimmed on the 6000), plus the 45B milestone; Library scores scope
+ TOLD-MARGIN on each. Fallback 3r if 3f fails to start. The 3d addendum closes with its saves (42 .. 44B, 8 points); verdict when
Library has scored 43.35 / 43.80 / 44B.
**3d -> what it did (Library 41B reference tuple vs 43B; recorded 20:35 IST 2026-11-17, Lead):** carry C up (2k 2.3 -> 9.9, 8k 0.49 ->
2.45) but over-application worse on every axis: scope_fu 2k 0.73 -> 1.00, 8k 0.17 -> 0.94; P_fu (no protocol, any unit) 2k 0.06 -> 0.42,
8k 0.18 -> 0.50; precedence u1 2k 0.49 -> 0.95, 8k 0.02 -> 0.44; plain held read on scope 2k 0.09 -> 0.84.
**3d addendum verdict (to be written when 43.35 / 43.80 / 44B are scored) will read: "dC WORKS, but HARMFUL on scope / presence /
precedence; superseded by 3f".** LESSON: the dC bar measured strength only (one axis). A rule-carry bar must come with the
over-application axes (scope_fu, P_fu, precedence) as co-bars, pre-registered together.
**3f FALLBACK RULE (Lead, recorded 20:35 IST, BEFORE any 3f number):** at 3f + 0.5B, if NOT (scope_fu 2k <= 0.30) AND NOT (P_fu 8k <=
0.20), the over-application is baked in: restart from the 41B save (step_625619, pre-3d; d_drive from_6000/.../sci1_long3c_3a8d7fb/,
FULL, never slimmed) with the 3f mix, accepting the loss of ~3.5B tokens. If the bars are met, 3f continues.
**3f STARTED 20:30 IST** from seg3d_end.pt = step **673,500 (44.14B)** (3d stopped 20:30; log logs/v13_sempty_sci1_long3f_91a87ed_20261117_1500.log on the 6000).
**3f AMENDMENT (21:17 IST 2026-11-17, Lead; BEFORE the first 3f snapshot is scored). Reason:** Library's 44B scope tuple shows the 3d
excursion settled fully: scope_fu 2k 0.55 (41B 0.73), 8k 0.10; P_fu 0.00 / 0.05; C 1.78 / 0.46, so 44B >= 41B on every axis.
(1) Reference = 44B. (2) The rollback-to-41B fallback is WITHDRAWN (step_625619 stays kept; it costs nothing). (3) 3f is judged on a
SETTLED window: the mean of +0.75B, +1.0B and the 45B milestone, and again at +1.5 / +2.0B. **WORKS** = scope_fu 2k mean <= 0.30 AND
C 8k mean >= 0.40 AND twin unit_acc 8k >= 0.70. **NO EFFECT** = scope_fu 2k mean within +-0.10 of 0.55. **Excursion guard per save:**
P_fu 8k <= 0.20 and scope_fu 8k <= 0.30; two saves in a row failing = stop and talk (no automatic switch). (4) +0.25 / +0.5B snapshots
are reported with NO verdict. Snapshot targets added at +1.5B (696,390) and +2.0B (704,020).
**LESSON (general):** a mix change at constant LR is judged on a settled window (>= 1B after the switch), never on its first saves
(3d's C swung 0.45 / 1.09 / 0.59 / 3.64 / 2.45 / 0.47 and its harm settled by 44B).
**3f early saves (Library scope tuples, qllm_v14norm/logs/v14_fartold_watch.log; NO verdict by the amendment; recorded 01:05 IST 11-18):**

| save | L | C | scope_fu | prec_u1 | P_fu | twin unit_acc | S_fu_held | R_u2 |
|---|---|---|---|---|---|---|---|---|
| 44B ref | 2k / 8k | 1.78 / 0.46 | 0.55 / 0.10 | | 0.00 / 0.05 | | | |
| +0.25B (677,315) | 2k | +3.303 | 0.90 | 0.72 | 0.03 | 0.98 | 0.16 | 0.28 |
| | 8k | +0.744 | 0.17 | 0.00 | 0.09 | 0.79 | 0.03 | 0.98 |
| +0.5B (681,130) | 2k | +1.635 | 0.51 | 0.48 | 0.01 | 0.96 | 0.01 | 0.52 |
| | 8k | +0.413 | 0.02 | 0.01 | 0.02 | 0.66 | 0.01 | 0.97 |
| +0.75B (684,945; window 1 of 3) | 2k | +2.056 | 0.72 | | 0.03 | 0.97 | | |
| | 8k | +0.548 | 0.12 | | 0.06 | 0.70 | | |

| **45B (686,655; window 2 of 3)** | 2k | +3.859 | 0.72 | 0.67 | 0.26 | 0.98 | 0.32 | 0.33 |
| | 8k | +0.775 | **0.36** | 0.10 | **0.33** | 0.71 | 0.24 | 0.90 |

Excursion guard (P_fu 8k <= 0.20, scope_fu 8k <= 0.30) passes at +0.25 / +0.5 / +0.75B and **FAILS at 45B (excursion #1)**: P_fu 8k
0.33, scope_fu 8k 0.36. First settled-window read (Library 82bd02bd, Lead): **WORKS NOT MET** (scope_fu 2k 0.72 / 0.72 >> 0.30).
Settled window = +0.75B (684,945), +1.0B, 45B. **+1.0B (688,760) was NOT snapshotted:** the copier skipped it at 04:29 IST (6000 below
its 30 GB gate). Substitute (Lead 04:32): `grab_689500.pt` = latest.pt at step **689,500 (45.19B tokens)**, slimmed on the 6000
(tensor-equal), sha-verified on d_drive (4d635a2d...), pointed to by ~/.qllm_lib_688760; scored in Library's SOLO window after tpp,
labelled by its real step. "Two in a row" is read on it. **CPU guard read (Library, 8k n 40, no GPU): CLEAR** (P_fu 0.050, scope_fu 0.075, C +0.549, unit_acc 0.75): 45B stays a single excursion; the GPU n-100 read follows in the SOLO window. The 6000's 45B copy was removed after a sha match (d75c3895..., kept full on
d_drive). +1.5B / +2.0B copier re-armed 04:48 IST (tmux mind3fsnap2).
**45B CHECK (step 686,655, 45.0B tokens; 03:27-03:48 IST 11-18 on the 4090; outputs logs/*/v13_sempty_*sci1_step_686655_91a87ed*,
from ../qllm_sci1check):** s8 @8k **0.945** (shift 0.9125, affine 0.9775; 44B 0.977), skill @8k (A_state) **0.71** (44B 0.735),
t40 16k **0.939** (44B 0.808; t40 8k 0.981), clk2 8k **0.92** / 32k-run @16k 0.68 (44B 0.79 / 0.48), HS / ARC-e / PIQA 0.322 /
0.550 / 0.639 (44B 0.316 / 0.553 / 0.643). Guards vs 44B (single save; the 3-save means come at the verdict): s8 0.945 >= 0.927 OK;
HS +0.006, ARC-e -0.003 OK; clk2 0.92 >= 0.78 OK. Generations: units only where the prompt gives them (protocol "12.48 g", tool
"78.4 C"); chat answers unit-free; the protocol greedy rounds wrongly (12.48 / 8.48 for 12.347 / 8.912).
*3e re-armed on the patched v2 (12:32 IST 2026-11-17, Lead (b): the held-out scope direction "asked volume, mass protocol" removed
from proto_far2; Library 4ece7bb5):* re-checks on 4ece7bb5: (a) packer 400 chunks identical, hook test OK, and proto_far v1 + proto4k
streams identical pre- vs post-patch (300 docs each); (b) Library check(): own 65.4 / none 29.5 / other 5.2 %, leak-free, 0 shared
5-grams; my count over 20,000 v2 docs: unit-family "asked volume, mass protocol" = 0 (asked mass / volume protocol 334; convert both
directions 334 / 331, not probed); (c) v2 patched 2,580 +- 271 tokens (pre-patch 2,600, v1 2,584). 6000 worktree qllm_sci1e moved to
4ece7bb5 and smoke-opened; the 43B watcher launches from it. S (the probed direction) is untrained in 3e.
Snapshot 42.25B = snap_645000.pt (hard-linked as snap_644693.pt after a name mismatch made Library's watcher log it "skipped";
scored in the next window). Window mix logged per snapshot in logs/v13_sempty_3d_snapshots_*.log.

## Tonight's 100M / 1B twins on the 4090 (PRE-REGISTRATION DRAFTS, 14:59 IST 2026-11-18, before any number; Lead orders them; card = user's until released)

Common: wio recipe (wio_chain.sh @ >= 227f3eb pins, ~103M, 0.9B wsd_stable + 0.1B decay, LR 1.2e-3, seed 43, SEQ 4096, readouts =
s8 spread 0 / 2k / 4k / 8k (3-save mean of 14,500 / 15,000 / end), t40 2k..16k, clk2, HS / ARC-e / PIQA, val). Speeds measured on the
wio arms (median step tok/s): gdn 114k, nochrono 49k, pam 63k. Readouts ~0.5 h per arm. 46B / 47B CHECKs (~25 min each) + the 47B
scope tuple go first in the window (~1 h).

### (a) nofar: GDN and clock-off PAM WITHOUT far docs (the missing cell for "holds beyond the trained length")
**Question.** In ctl (no far docs; caa4da9) PAM carried rules past its 4k training length (s8 D 2k/4k 0.523, s8 @8k 0.278, t40 16k
0.586) where Mamba-2 / T++ did not (0.149 / 0.074). Is that the delta-rule family (then GDN does it too) or ours, and is it the clock?
**Arms.** gdn_nofar, nochrono_nofar: wio_chain arms with ctl's SOURCES / WEIGHTS (the wio mix with the far block's 2.0 % returned
to dclm / fineweb: 0.23324 / 0.24116). Reference: ctl pam (no far) as measured; it ran on caa4da9 / qllm311, and wio pam matched it
on the 2k/4k metric (0.538 vs 0.523), so it is used as is; if a verdict hangs within 0.05 of a bar, a pam_nofar rerun (4.5 h) is due.
**Bars (3-save s8 D at 2k / 4k, the ctl metric; s8 @8k reported beside).** OURS: ctl pam - gdn_nofar >= 0.10 (gdn_nofar <= 0.42).
FAMILY: gdn_nofar >= 0.47. In between: no verdict. CLOCK: nochrono_nofar vs ctl pam, same 0.10 rule. t40 16k reported, no bar.
**Cost.** gdn 1.0B / 114k = 2.4 h + 0.5 h; nochrono 5.7 h + 0.5 h -> **~9.1 h** (gdn alone ~3 h).
**Closest prior work.** Gated DeltaNet (Yang et al. 2024); our ctl. A control, no novelty claim.

### (b) split key geometry: half the notebook heads clocked, half plain RoPE
**Question.** wio: the clock buys needles (t40 16k 0.803 vs 0.353) and costs rules (s8 @8k 0.343 vs 0.427). Can one body have both
if the keys of some heads are clocked (episodes) and the rest phase-free (rules)?
**Change (code, ~1 h + parity test).** CHRONO_HEADS=k: the Chrono warp g_t applies to heads 0..k-1 only; heads k..H-1 keep g = 1
(exactly plain RoPE, the nochrono path). k = H/2. Zero-init warp keeps exact RoPE parity at init for all heads (test: k = 0 equals
nochrono and k = H equals pam, bit-exact forward on a fixed batch). Readers (SHARE_READS) use the writer's per-head rotation as now.
**Bars (3-save).** BOTH: s8 @8k >= 0.40 (nochrono 0.427 - 0.03) AND t40 16k >= 0.70 (pam 0.803 - 0.10) AND val <= 3.435.
RULES-ONLY / NEEDLES-ONLY: one of the two; NEITHER: below both (the geometries interfere). Diagnostic: s8 and t40 with the clocked
heads' read zeroed vs the RoPE heads' read zeroed (eval-time knockout, as kn).
**Cost.** ~56k tok/s expected -> 5.0 h + 0.5 h readouts + 1 h code / parity -> **~6.5 h**.
**Closest prior work.** Hybrid RoPE / NoPE attention (Yang et al. 2025, "Rope to Nope and Back Again"; Llama 4 interleaved NoPE
layers) and partial rotary (GPT-NeoX rotary_pct). Difference: the split is per head INSIDE a delta-rule notebook, and one half's
phase is a learned content clock (Chrono), so it separates where a key is stored (episode) from how it is erased (rule).

### (c) far share x LR at constant rate: is the 386M excursion an LR oscillation?
**Question.** 3d and 3f (far protocol 1.6 %) swung (P_fu / scope_fu up, then back) at constant LR 8e-4; 3c (0.8 %) was quiet on its
one save. Share, LR, or both?
**Design.** Parent = wio pam stable_final (0.9B, constant LR 1.2e-3, far protocol 0.8 %). Four continuations of 0.4B each at
constant LR: {0.8 %, 1.6 %} x {1.2e-3, 6e-4}, extra 0.8 % from dclm / fineweb. Saves every 0.1B (1,526 steps); GPU guard read at
2k (n 100, the wio scope method; 8k is unreadable at 100M: C <= 0.08).
**Validity.** Parent C 2k >= 0.5 (wio scope: 1.50); otherwise the cell is unreadable at 100M.
**Reading.** EXCURSION in an arm = at any of its 4 reads P_fu 2k or scope_fu 2k >= parent + 0.25. LR-driven: 1.6 % @1.2e-3 swings
and 1.6 % @6e-4 does not. Share-driven: both 1.6 % arms swing, neither 0.8 % arm. No arm swings: the excursion needs scale or
40B+ of training; stop the cell (no 386M action from it).
**Cost.** 4 x 0.4B / 63k = 7.1 h + 16 reads x ~10 min = 2.7 h -> **~10 h**. Pilot option: the two 1.6 % arms first (~5 h).
**Closest prior work.** WSD / constant-LR training (Hägele et al. 2024) and small-scale proxies of training instabilities (Wortsman
et al. 2023). Difference: a behaviour-level excursion (over-application of a told rule) after a mix change, not a loss spike.

**ADDENDA (Lead GO 20:4x IST 11-18, recorded before any of these numbers; the 6000 joins the window, 3r paused there):**
- **Placement.** 4090 (wio env): gdn_nofar (done) -> split (b) -> SECOND SEEDS (seed 44) of the wio far-mix arms pam, nochrono, then
  gdn. 6000 (paused main run): Set A = pam_nofar, nochrono_nofar, gdn_nofar concurrently in ONE environment; then Set B = (c).
  65,536 tokens / step and the same accumulation on both boxes (no bigger batch); several arms at once instead; speed not quoted there.
- **Seed 44 (REPLICATED rule for the clock's trade):** both effects keep their sign with at least half the seed-43 gap: t40 16k
  pam - nochrono >= +0.22 (seed 43 +0.45) AND s8 @8k 3-save nochrono - pam >= +0.04 (seed 43 +0.084). gdn seed 44 reported beside,
  no bar. Closest prior work as in wio.
- **Set A:** the OURS / FAMILY / CLOCK bars of (a) are read within the 6000 (pam_nofar there replaces ctl pam as the reference).
  The 4090 gdn_nofar is the cross-box control: the two gdn_nofar runs must agree within 0.05 on s8 D (3-save) and 0.03 on val;
  otherwise the cross-box comparison is unusable and only within-box reads count.
- **Set B:** parent = wio pam main stable_final copied with a sha; its parent tuple read on the 6000 first (validity C 2k >= 0.5).
- **Gates before the 3r pause:** fp32 parity of one fixed batch through pam and gdn on both boxes within 1e-4 (else stop and talk),
  and identical token streams (first 2,000 packed windows, ctl mix and wio far mix, sha). Pause at a save -> seg3r_pause.pt.
**GATES PASSED + 3r PAUSE (20:49 IST 11-18):** (2) parity, fixed 2 x 2048 real-token batch, trained weights sha-equal on both
boxes: pam fp32 CE 4.268455 (4090) = 4.268455 (6000), bf16 4.268277 / 4.268906; gdn fp32-path 4.099769 / 4.099792, bf16 4.100368 /
4.100105 (|d| 2.6e-4 < bar 2e-3; run on the 6000 after the pause, it OOMs beside 3r). (3) data: train.py --dump_batches 125 (3eacafbb)
on the logs' exact [args]: ctl mix sha 9ae682b1... and wio far mix 72e954b7... identical on both boxes (4090 repeat identical;
transformers 5.2.0 vs 4.57.6 changes no token). **3r PAUSED at step 721,000 (47.251B) -> sci1_long3r_91a87ed/seg3r_pause.pt**
(6000, 20:48 IST; resume with sci1_cont_chain.sh CONT_FROM=seg3r_pause.pt + sci1_long3r_env.sh). Set A launched 20:50 IST on the
6000 (worktree qllm_twins @ a3ffb14b, twins_chain.sh BOX=6000, tmux tw_pam_nofar / tw_nochrono_nofar / tw_gdn_nofar, 3 at once).
**(a) gdn_nofar RESULT, 4090 (21:1x IST 11-18; 102.51M, 1.0B tokens on ctl's mix, SOLO ~115k tok/s, seed 43; logs/wio/*nt_gdn_nofar_10a4c1db*):**
s8 D (2k / 4k) 3-save **0.474** (saves 0.424 / 0.505 / 0.492); s8 @8k 3-save 0.174; shift 0 / 2k / 4k / 8k 0.996 / 0.651 / 0.361 / 0.113,
affine 0.992 / 0.516 / 0.367 / 0.235; t40 2k / 4k / 8k / 16k 0.736 / 0.536 / 0.372 / **0.283**; clk2 0.05 / 0.01; HS / ARC-e / PIQA
0.272 / 0.456 / 0.602; val 3.387. Against the (single-save, other-env) ctl pam: s8 D 0.523 - 0.474 = +0.049 (OURS bar +0.10 not met);
FAMILY bar 0.47 met by 0.004 -> **inside the 0.05 band of both bars: NO VERDICT (strict rule)**; the within-box read on the 6000
(Set A: pam_nofar vs gdn_nofar vs nochrono_nofar) decides, and this run is the cross-box control for its gdn_nofar. Beside it, no bar:
past the trained length ctl pam holds more (s8 @8k 0.278 vs 0.174; t40 16k 0.586 vs 0.283).
**(b) split STARTED 21:27 IST 11-18** (4090 SOLO, CHRONO=1 CHRONO_HEADS=3 confirmed in [config] / [ladder], 102,884,948 params, night
chain restarted with ALLOW_CODE_DRIFT=1 on SCREEN_HASH 10a4c1db after a refused launch at 21:19; the refusal also deleted the launch
guard: the launcher trap bug, fixed b558dd89). **(c) Set B parent read (6000, 21:2x IST):** wio pam main stable_final 2k: C +1.659 (valid,
>= 0.5), scope_fu 0.69, prec_u1 0.20, P_fu 0.07, unit_acc 0.93. The 4 arms wait for free slots (< 3 trainers).
**(b) split RESULT + VERDICT (4090 SOLO, 102.88M, 1.0B tokens, seed 43; chain DONE 02:41 IST 11-19; recorded 07:1x IST; logs/wio/*nt_split_10a4c1db*):**
s8 @8k 3-save **0.473** (0.455 / 0.484 / 0.480; best of all wio-recipe arms: nochrono 0.427, gdn 0.408, pam 0.343); s8 D 0.695; shift
0 / 2k / 4k / 8k 0.868 / 0.774 / 0.627 / 0.379, affine 0.715 / 0.701 / 0.676 / 0.567; t40 2k / 8k / 16k 0.844 / 0.742 / **0.692**
(n 360 each, binomial se 0.019 / 0.023 / 0.024; pam 0.803, nochrono 0.353); clk2 0.43 / 0.05; HS / ARC-e / PIQA 0.271 / 0.466 / 0.610;
val 3.4248 (best_val_loss 3.42482952; wio pam 3.42476004 = a coincidence to 4 decimals: per-source val differs, e.g. dclm 3.4329 vs
3.4339, pysci 1.1397 vs 1.1213; 3 of 6 warp rows active in split vs 6 / 6 in pam). **VERDICT by the letter: RULES-ONLY** (s8 @8k
0.473 >= 0.40 MET; val 3.4248 <= 3.435 MET; t40 16k 0.692 < 0.70 NOT MET by 0.008 = 0.33 se). Needles at 86 % of pam's and 2.0x
nochrono's. One seed; the seed-44 arm (below) is the test. Knockout diagnostic (clock / rope half off) running on the 6000.
**(a) Set A on the 6000 + CROSS-BOX VOID (07:1x IST 11-19):** gdn_nofar 6000: s8 D 3-save **0.570**, s8 @8k 0.242, t40 16k 0.247, val
3.3864; 4090: 0.474 / 0.174 / 0.283 / 3.387. val agrees (0.001) but s8 D differs by **0.096 > the pre-registered 0.05 -> cross-box
comparison VOID**; only within-box reads count. Same body, data, seed, recipe on two boxes (Blackwell vs Ada bf16 kernels, different
interleaving) moves s8 D by ~0.1: **single-seed s8 gaps under ~0.10 are inside run-to-run noise** (split vs nochrono vs gdn on the
4090 included); t40 gaps of 0.3-0.6 are far outside it. Within the 6000: pam_nofar s8 D **0.559**, s8 @8k 0.258, t40 16k **0.853**,
val 3.4225 vs gdn_nofar 0.570 / 0.242 / 0.247 / 3.3864: **OURS NOT MET (pam - gdn = -0.011); FAMILY MET (gdn_nofar 0.570 >= 0.47,
outside the 0.05 band): carrying induced rules past the 4k training length without far docs is a delta-rule-family property.**
Needles past the length are ours (t40 16k 0.853 vs 0.247 on the same box). CLOCK: nochrono_nofar readouts pending (val 3.4202).
**Seeds order changed (Lead 03:08):** 4090 = pam_s44 -> **split_s44** -> nochrono_s44 -> gdn_s44 (chain restarted 06:59, adopted the
pam_s44 decay). **split seed-44 PRE-REGISTRATION (recorded before it starts):** REPLICATED if s8 @8k 3-save >= 0.40 AND t40 16k
>= 0.60, AND s8 @8k >= nochrono_s44 s8 - 0.02 AND t40 16k >= pam_s44 t40 16k - 0.20. A CHRONO_HEADS 2 / 4 rung is worth one night
only if split_s44 replicates (then: is the trade a dial?); nothing extra runs before.
**(b) split KNOCKOUT diagnostic (pre-registered; 6000, 07:0x-07:33 IST 11-19; split decay end save, sha d12a891c; eval-time
V13_CHRONO_KNOCKOUT zeroes that head half's notebook read in the writer and its M2 readers; logs/wio/v13_sempty_split_ko_*):**

| condition | s8 @8k | shift 0 / 2k / 4k / 8k | affine 0 / 2k / 4k / 8k | s8 D | t40 16k |
|---|---|---|---|---|---|
| none (same-box control) | 0.480 | .930 / .833 / .670 / .380 | .745 / .715 / .690 / .580 | 0.727 | 0.692 |
| clocked half OFF (RoPE heads 3-5 only) | **0.026** | .007 / .003 / .003 / .007 | .152 / .138 / .125 / .045 | 0.067 | 0.444 |
| RoPE half OFF (clocked heads 0-2 only) | 0.177 | .845 / .472 / .335 / .203 | .650 / .410 / .260 / .152 | 0.369 | **0.306** |

(The control reproduces the 4090 end-save cells on the 6000: s8 @8k 0.480, t40 16k 0.692.) **Reading: the hypothesis "rules in the
phase-free half, needles in the clocked half" is REFUTED in both directions.** Without the clocked heads the rules collapse even with
the examples adjacent (shift @0 0.93 -> 0.007): the rule circuit runs THROUGH the clocked heads. Without the RoPE heads rules survive
adjacent and fade with distance (s8 D 0.37), and needles fall MORE (0.692 -> 0.306) than without the clocked heads (-> 0.444). The two
halves are not modules; the split works as a joint code (clocked heads index / apply, phase-free heads hold content that survives
distance?). One seed, eval-time zeroing of a jointly trained model: diagnostic, no claim.
**(a) Set A COMPLETE (6000, within-box, ctl mix = no far docs, seed 43; 08:01 IST 11-19; logs/wio/*nt_*_nofar_a3ffb14b* on the 6000):**

| arm | s8 D (2k/4k) 3-save | s8 @8k | shift 0 / 2k / 4k / 8k | affine 0 / 2k / 4k / 8k | t40 2k / 8k / 16k | clk2 8k / 16k | HS / ARC-e / PIQA | val |
|---|---|---|---|---|---|---|---|---|
| pam_nofar | **0.559** | **0.258** | .950 / .796 / .585 / .276 | .748 / .497 / .359 / .239 | .961 / .917 / **.853** | .41 / .05 | .273 / .463 / .592 | 3.4225 |
| nochrono_nofar | 0.404 | 0.121 | .998 / .738 / .422 / .141 | .717 / .284 / .171 / .102 | .803 / .717 / .633 | .16 / .02 | .274 / .453 / .592 | 3.4202 |
| gdn_nofar | **0.570** | 0.242 | .999 / .840 / .557 / .271 | .986 / .513 / .371 / .212 | .647 / .311 / .247 | .05 / .02 | .273 / .466 / .591 | **3.3864** |

**VERDICTS (pre-registered bars, within the 6000):** OURS NOT MET (pam - gdn = -0.011). **FAMILY MET** (gdn_nofar 0.570 >= 0.47).
**CLOCK MET: pam - nochrono = +0.155 >= +0.10** (outside the 0.05 band; ~1.5x the one-pair noise estimate). Needles past the
length: pam 0.853 > nochrono 0.633 >> gdn 0.247.
**Reading:** without far docs the clock HELPS rule carry past the training length (+0.155 s8 D, +0.137 s8 @8k); with far docs in the
mix (4090 wio) it cost rules (pam 0.537 vs nochrono 0.639 s8 D). Interaction, one seed each and across boxes, so a hypothesis: far
docs teach the plain delta notebook to carry rules at distance, and the clock's distance code is what a body uses when the data do
not teach distance. The clock's recall edge holds in both mixes.
**(c) Set B, first two arms (6000; 2k scope tuple, n 100, each +0.1B read; parent P_fu 0.07 / scope_fu 0.69 / C 1.659; swing bar
= parent + 0.25 = P_fu >= 0.32 or scope_fu >= 0.94; recorded 16:3x IST 11-19):**

| arm | read (B after switch) | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 (end) |
|---|---|---|---|---|---|---|
| f08_lr12 (0.8 %, LR 1.2e-3) | P_fu | .07 | .05 | .01 | .00 | **.53** |
| | scope_fu | .71 | .54 | .48 | .24 | .76 |
| | C | 1.76 | 1.60 | 1.33 | 1.26 | 1.02 |
| f16_lr12 (1.6 %, LR 1.2e-3) | P_fu | .07 | .23 | .13 | .00 | .29 |
| | scope_fu | .68 | .67 | .62 | .15 | .82 |
| | C | 1.71 | 1.92 | 1.76 | 1.00 | 1.93 |

f08_lr12 = EXCURSION at 0.4B (P_fu 0.53); f16_lr12 = no excursion by the letter (P_fu max 0.29). Both dip at 0.3B (scope_fu 0.24 /
0.15) and rebound at 0.4B: a constant-LR swing is reproduced at 100M, and it is not larger with the doubled share ("share-driven" not
supported at LR 1.2e-3). The lr06 arms decide "LR-driven" (pending Lead's call on the 6000 window; 4090 rebooted 10:31 IST, local jobs
restarted 16:23).
**lr06 READING (Lead 16:3x IST 11-19, recorded BEFORE any lr06 read exists):** f08_lr06 and f16_lr06 run together after the s45 mains,
same parent, same +0.1B reads (incl. +0.4B end) on the same docs as lr12. **LR-DRIVEN** = neither lr06 arm reaches the swing bar
(P_fu >= 0.32 or scope_fu >= 0.94) at any read, while the lr12 arms did (f08_lr12 at 0.4B; f16_lr12 0.29, just under), AND the lr06
arms' max P_fu <= 0.15. **NOT LR-driven** = either lr06 arm swings. In between = UNCLEAR. Each arm is one continuation from one parent
(within-run dynamics, not a seed comparison). The lr12 pair already disagrees with "share-driven" (the 0.8 % arm swung more). An
LR-DRIVEN reading does NOT change the main run's LR by itself: it leads to a proposal (e.g. a step to 4e-4 judged on the four-cell tuple
over a settled window, with the decay finding in mind). 3r resumes from seg3r_pause.pt as soon as the lr06 reads are done.
## Three pre-registrations (Lead 11-19, recorded 16:44 IST BEFORE any of them runs; order: gfd on the 4090 after gdn_s44, then gdn_srope; plavg in a Library window)

**Margin finding for the paper (observation, with its seed count):** the pam seed difference is present already at distance 0 (shift
M0 1.41 vs 5.05, 2 seeds): it is circuit STRENGTH at 1B, not survival over filler. Across the delta-rule bodies (1 seed each, 2 boxes),
GDN's adjacent rule circuit is stronger (M0 3-9) but decays steeply with distance, while PAM's is weaker and flatter.

### gfd: GATE-FROZEN DECAY pilot (100M, 4090, ~1.2 h; FIRST after gdn_s44)
**Question.** A 1B-token-scale LR decay erases the told-rule carry itself (decay_M9 - M9 dC(2k) -0.37, 3f61ea54). Is the erasure in
the memory's dynamics gates? **Arms (one parent, identical data / weights / seed; treatment vs control, so powered):** from wio pam
main stable_final (step 13,733): (P) the plain wio decay (0.1B, LR 1.2e-3 -> 0, as the wio arms); (F) the same decay with these
parameter groups FROZEN (requires_grad False before the optimizer is built): every notebook's dt_proj (weight, bias) + dt_bias
(retention / forget rate), warp_proj (Chrono clock rate), erase_proj and write_proj (delta erase / write strength), and the global
echo_gates. **Why these:** they set how long and how strongly the state keeps a rule (retention, time code, overwrite); if the decay
sharpens them toward faster forgetting / stronger overwrite, freezing them should keep the carry while the rest (qkv / o projections,
MLPs, readers, embeddings, the N4 out gate gate_proj, reader gates) still sharpens prose. Not frozen: everything else.
**Reads (both arms, end save, paired docs):** dC at 2k vs the parent (scope tuple, n 100), the full four-cell tuple at 2k, s8 margins
(M0 / M4k / M8k per task, 3-save), t40 2k / 8k / 16k, val. **Bars:** KEEPS if dC(F) >= -0.10 while dC(P) <= -0.25, with val(F) <= val(P)
+ 0.01. FAILS if dC(F) <= -0.25 (the erasure is not in these gates). UNCLEAR otherwise. If dC(P) > -0.25 (the plain decay does not
erase at 100M), the pilot is UNREADABLE (say so). **Code needed:** train.py --freeze_params <regex> (default empty = off), parity:
an empty regex reproduces the plain decay's first 50 steps bit-exact. **Closest prior:** layer freezing during fine-tuning (e.g. BitFit /
partial fine-tuning) and WSD cooldown studies (Hägele 2024); difference: freezing only a recurrent memory's dynamics gates during an
LR cooldown to protect an in-state skill.

### plavg: PLATEAU-AVERAGE read (386M, zero training, a Library window on the 4090)
Uniform average (weights only, fp32 mean) of 3-4 constant-LR 3r saves: 47B (717,173) plus the +0.25B / +0.5B snapshots after the 3r
resume (and 48B if it lands first), vs each single save. Reads: the four-cell tuple (2k + 8k), C, s8 @8k (+ margins), t40 16k,
HS / ARC-e, val. Reported descriptively; prior: LAWA L1 at M9 lost dC -0.26, so a loss is expected. Bar for "worth a release step":
the average keeps C 8k >= min(singles) and its scope_fu / P_fu are no worse than the singles' mean, with val better than every single.
Closest prior: LAWA / checkpoint averaging (Kaddour 2022, Sanyal 2023); plumbing, no claim.

### gdn_srope: GDN + SELECTIVE RoPE (closest prior work as a trained baseline; 4090, 2 seeds, ~3.5 h each + 2-3 h setup)
**Arms.** gdn_srope_s43, gdn_srope_s44 at the wio recipe (far mix, 1.0B, 0.9B stable + 0.1B decay, SEQ 4096) with Selective RoPE AS
THEIR REPO RUNS IT ON GDN, no retuning: their layers/gated_deltanet.py (fla GDN + SelectiveRoPE) at their DEFAULT config, to be
written into the run header verbatim (phase gate on / off, gate parametrization, phi rank 16 low-rank, conv k 4 (+ SiLU or not),
temp bank type and theta 5e5, temp_grad off, bias off), GDN's short conv + gated output kept, our geometry (12 layers, 6 heads,
head 64, expand_v 2). Env: a separate overlay venv with their package + its pinned fla commit; every difference from qllm_gdn stated.
**Parity tests before launch:** (i) their layer with position_embedding = none equals our baseline_gdn (fla GatedDeltaNet) forward at
shared weights; (ii) their own tests (SELECTIVE_ROPE_TESTING=1) pass; (iii) streamed (two chunks, carried rotary state) equals one
full forward. **Judged on what is stable:** t40 2k / 8k / 16k and clk2 (s8 reported as margins only), against pam's two seeds (t40 16k
0.803 / 0.786) and gdn's. **Bars (per seed pair):** OUR FORM ADDS if pam t40 16k >= gdn_srope + 0.15 on BOTH seeds; EQUAL if within
0.08 on both; THEIRS BETTER if gdn_srope >= pam + 0.08 on both; mixed otherwise. **Stated now: if EQUAL, the clock's recall edge is
the idea's (input-dependent cumulative rotation, Selective RoPE first), not ours.** Closest prior: Selective RoPE (Movahedi et al.,
arXiv:2511.17388); this arm IS that prior work on our recipe.

**Chrono vs Selective RoPE (closest prior work; arXiv:2511.17388, ICLR 2026; read from the paper AND their code,
github.com/timurcarstensen/selective-rope src/selective_rope/modules/rotary.py + layers/gated_deltanet.py; Mind 16:3x IST 11-19):**
- *Selective RoPE:* per head and PER CHANNEL PAIR (head_k_dim/2 angles), increments phi_t = conv1d_k4(W_phi l2norm(x_t)) (low-rank
  W_phi, rank 16 by default; optional SiLU after the conv), optionally times a sigmoid phase gate (per head or per channel), optional
  + exp(bias); angle_t,i = temp_i * cumsum_t(phi)_i, with temp_i a FIXED (non-trainable by default) RoPE frequency bank (theta 5e5) or
  the tan(phi/2) "rfa" bank. Increments are free in sign and size per channel (no clamp); default init (kaiming / nn.Linear), so at
  init the angles are a cumsum of small random increments, NOT RoPE. In their GDN the layer keeps GDN's short conv on q / k / v and
  its gated RMSNorm output. Their repo pins fla at a git commit (not 0.3.2) and torch < 2.9.
- *Chrono (ours, v13_sempty/model.py PAMLayer ~l.565 / 715-740):* ONE positive scalar rate per head, g_t = exp(clamp(W x_t, -3, 3))
  in (0.05, 20), tau_t = cumsum_t(g) (exclusive), angle_t,i = tau_t * inv_freq_i with the standard RoPE bank (10,000): the rate warps
  the WHOLE frequency bank of a head uniformly (a learned per-head clock speed, monotone time); W zero-init so g = 1 and the rotation
  is EXACTLY RoPE at init (parity test); no conv, no gate, no l2norm; the carried decode state is one clock value per head; in M2 the
  12 reader blocks reuse the writer's rotation (SHARE_READS); and because the clock is one rate per head it can be DILATED at eval
  (clk2: g x 8) - a per-channel free angle has no single "speed" to dilate. CHRONO_HEADS lets a subset of heads keep g = 1.
- *Difference in one line:* Selective RoPE learns free per-channel phase increments (an input-dependent rotation code); Chrono learns
  a per-head monotone time warp of RoPE (an input-dependent clock) that starts as exact RoPE and stays dilatable. Same family idea
  (input-dependent cumulative rotation in a linear-recurrent body), different form. **Claim downgrade:** "a content-modulated clock on a
  delta-rule memory" is NOT a new idea after 2511.17388; what is ours is the form (monotone per-head rate, RoPE-exact init, shared
  readers, dilation) and the measured recall edge, to be tested against GDN + Selective RoPE.
**Baseline price "GDN + Selective RoPE" (wio recipe, not run, needs a pre-registration):** their layers/gated_deltanet.py is fla's GDN
+ SelectiveRoPE (low-rank phi, conv, gate). Path: a separate overlay venv with their package and its fla commit (fla 0.3.2 API may not
match; ~2-3 h setup + test), a v13_sempty/baseline_gdn_srope.py wrapper at our geometry (12 layers, 6 heads x 64 / expand_v 2), arm
"gdn_srope" in the twins chain. Parity tests: (i) their layer with position_embedding=none vs our baseline_gdn (fla GatedDeltaNet)
forward-equal at shared weights; (ii) their own tests (SELECTIVE_ROPE_TESTING=1); (iii) streamed vs full forward. Cost on the 4090:
GDN ran ~114k tok/s solo -> ~2.5 h + an assumed 10-25 % rope overhead -> ~3 h train + 0.5 h readouts; per seed. Read on t40 16k
(the claim that survives the noise) and clk2, with s8 only as a margin trio.
**Margin re-read (Argon (B); zero training; margins ARE stored: mean_target_margin = gold logit - max of the 7 other candidates, per
task and distance; 3-save means):** M0 / M4k / M8k (shift | affine):
pam s43 +1.41 / -0.27 / -1.11 | +0.79 / +0.34 / -0.07; pam s44 +5.05 / +2.35 / +0.82 | +1.28 / +0.88 / +0.36; nochrono +2.20 / +0.19 /
-0.70 | +1.35 / +0.67 / +0.22; split +1.91 / +0.54 / -0.46 | +1.21 / +0.92 / +0.40; gdn +3.28 / -1.20 / -2.52 | +4.93 / +1.93 / +0.91;
6000 nofar: pam +3.07 / +0.36 / -0.72 | +1.56 / -0.35 / -0.76; nochrono +4.89 / -0.32 / -1.28 | +1.23 / -0.84 / -1.11; gdn +9.09 / +0.50 /
-1.22 | +5.87 / -0.42 / -1.01. **The seed spread does NOT shrink:** the two pam seeds differ already at distance 0 (shift M0 1.41 vs
5.05): the circuit's STRENGTH at 1B is what varies (acquisition), not its survival over filler. The carry ratio M8k / M0 is unstable
where M0 is small (s43 shift -0.79 vs s44 +0.16) and is not a lower-variance read. Side observation: GDN builds a much stronger
adjacent circuit (M0 3-9) that decays steeply with distance; PAM's adjacent margin is smaller and decays less. pam_s45 adds a third
point when its readouts land.
**Release-recipe pilots (Lead (C); view, nothing run):** (ii) the GATE-FROZEN DECAY is worth a 100M pilot and is well powered: two
decays (plain vs gates frozen: dt / warp / erase / write / echo) continued from ONE parent (wio pam stable_final) see identical data and
start from identical weights, so their difference is a treatment effect, not seed noise (unlike the seed comparisons). Read: dC at 2k
(scope tuple, paired docs), s8 margins, t40, val; ~0.1B each, ~1.2 h for both on the 4090. It tests directly WHERE the decay erases
the carry (3f61ea54: decay dC -0.37). (i) uniform averaging of 3r plateau saves needs no pilot at all: it is a zero-training read on
saves we will have (average 3-4 weights-only saves on CPU, read the four-cell tuple + C on the 4090), but LAWA L1 already lost dC -0.26,
so expect a loss; worth one read, not a run.
**DATED NOTE (Lead decisions, 08:2x IST 2026-11-19; applies to the wio and twins sections):** (1) Every s8-based single-seed verdict
among the delta-rule bodies is re-labelled **UNDETERMINED (underpowered: seed noise >= 0.28 on one same-box pair)**: the wio claim vs
GDN (read "not met"; pam_s44 alone would clear its 0.508 bar; neither reading is evidence), "the clock costs rules with far docs",
split's rule lead, Set A's CLOCK and OURS verdicts. None is flipped in our favour; the original verdict lines stay. **What stands:**
T++'s collapse past its window (structural: shift ~0.01 at distance, needles 0.08 at 16k); the recall edge (t40 16k pam 0.80 / 0.79 on
two seeds vs 0.33-0.35; Set A 0.85 vs 0.63 vs 0.25); val and the benchmarks. (2) No new experiment whose verdict is a small s8 gap at
100M / 1B (an SD ~0.2 would need dozens of seeds per arm): no CHRONO_HEADS 2 / 4 rung, no seeds beyond the queued trios. (3) Queued
runs continue (t40 replication + a 3-seed spread): split_s44, nochrono_s44, gdn_s44 (4090), the s45 trio (6000); Set B continues
(one shared parent: a within-run swing read, not a seed comparison). Report per arm s8 @8k, s8 D, t40 2k / 8k / 16k, clk2, val, and
per-arm mean / min / max over seeds at the end. (4) The half-clocked body pays a real recall cost (t40 16k 0.692 vs pam 0.80 / 0.79)
and its rule gain is inside s8 noise: no case to change the main run's body; the line closes with the trios unless split_s44 / s45
reach pam's t40. (6) The 6000 returns to 3r when the s45 trio and Set B end (~Fri 01:00 IST).
**Why is s8 seed-sensitive at 1B? (Mind's read of the logs, Lead (5); no run):** (i) the variance sits almost all in shift_wide:
shift @8k 0.225 (s43) vs 0.720 (s44), affine @8k 0.460 vs 0.578. (ii) It moves during the decay: s44 shift @8k 0.375 at step 14,000
(0.917B, 0.07B into the decay) -> 0.672 at 14,500 -> 0.682 -> 0.720 (end); s43 0.147 -> 0.263 -> 0.212 -> 0.225 while its shift @2k
climbs 0.438 -> 0.680 over the same saves. So shift at distance is still being ACQUIRED at the read point; seed 44 crossed its
transition by 14,500, seed 43 is mid-way (its 2k is rising, 8k has not followed). (iii) It is decoupled from language loss: seed 44 is
WORSE on holdout val at every logged step (2k 4.086 vs 4.042 ... 13,733 3.587 vs 3.563; end 3.440 vs 3.425) yet far better on shift.
(iv) No b2s / b2s_far val is logged (val_sources exclude them), so the per-source answer-loss trajectory is not in the logs.
**Implication:** a skill mid-transition at the read point gives a bimodal, seed-dependent score; a later read (e.g. 2B) or a
time-to-threshold metric (the step at which shift @4k first crosses 0.5, from saves every ~0.1B) would have lower variance. Needs
denser saves / probe reads in the main phase; nothing run.
**pam seed 44 (4090, same box / env / recipe / data as wio pam seed 43; only SEED differs; 08:2x IST 11-19; logs/wio/*nt_pam_s44_0e76eac5*):**
s8 @8k 3-save **0.620** (saves 0.604 / 0.609 / 0.649; seed 43: **0.343**); s8 D 0.817 (seed 43 0.537); shift 0 / 2k / 4k / 8k
.997 / .978 / .915 / .692 (s43 .796 / .610 / .440 / .233); affine .720 / .713 / .662 / .549 (s43 .656 / .571 / .529 / .453);
t40 2k / 4k / 8k / 16k .867 / .839 / .839 / **.786** (s43 .900 / - / .817 / .803); clk2 .61 / .14 (s43 .42 / .10); HS / ARC-e / PIQA
.274 / .466 / .591 (s43 .274 / .451 / .597); val 3.4397 (s43 3.4248).
**SEED NOISE ON s8 IS LARGE: same box, only the seed changed, s8 @8k +0.277 (0.343 -> 0.620), s8 D +0.28; t40 16k moved -0.017
and val +0.015.** So the s8 differences of the single-seed wio table (pam 0.343 / nochrono 0.427 / gdn 0.408 / split 0.473) and the
pre-registered wio claim verdict are inside seed noise; the cross-box pair (0.096) underestimated it. Needles are stable across seeds.
The trio rule (>= 2 of 3 within-box trios) is now the only basis for any rule-carry statement; t40 statements are robust so far.
**Run-to-run noise estimate (one pair; Lead 11-19):** the same gdn_nofar (body, data shas, seed 43, recipe) on the 4090 vs the 6000:
s8 D (2k / 4k, 3-save) 0.474 vs 0.570 (|d| 0.096), s8 @8k 0.174 vs 0.242 (0.068), t40 16k 0.283 vs 0.247 (0.036), val 3.387 vs
3.3864 (0.001). Caveat: ONE pair (arch + kernel + interleaving differences, not only the seed); read as "s8 single-seed noise ~0.1
at 100M / 1B", t40 noise ~0.04, val ~0.001.
**Seed 45 PRE-REGISTRATION (Lead 11-19, recorded before any seed-45 arm starts):** pam_s45, nochrono_s45, split_s45 on the 6000
(far mix, same recipe; max 3 trainers there; after / into Set B's slots). Cross-box reads are void, so the unit of replication is the
WITHIN-BOX TRIO (pam, nochrono, split at one seed on one box): 4090 seed 43, 4090 seed 44, 6000 seed 45. For each trio count:
(a) t40 16k pam - nochrono >= +0.22; (b) split t40 16k >= 0.60 AND >= pam - 0.20; (c) split s8 @8k 3-save >= nochrono - 0.02.
A claim stands if it holds in at least 2 of 3 trios (and is reported with the count). The 6000 pause extends to ~Fri 01:00 IST
(~28 h); 3r resumes after it with the 30 GB floor.
**Proposed order** (one night ~10-12 h after the CHECKs): (a) gdn_nofar (3 h; closes the paper's "beyond the trained length" cell
for the family), then (b) split (6.5 h; the one new mechanism), then (a) nochrono_nofar (6.2 h); (c) last (10 h; readability risk,
and 3r's guard reads may already answer whether 0.8 % is quiet at 386M). Nothing launches before Lead's GO and the user's release.

## segment 3r: back to 3c's mix after 3f's second guard break (PRE-REGISTERED 09:25 IST 2026-11-18, before any 3r number; Lead's stop-and-talk call 09:13, Mind agreed)

**Why:** 3f's per-save guard sequence pass / pass / pass / BREAK (45B: P_fu 8k 0.33, scope_fu 8k 0.36) / pass (45.19B CPU) / BREAK
(+1.5B = 696,390, GPU n 100: 8k scope_fu 0.43, P_fu 0.22; 2k scope_fu 0.84; twin unit_acc 8k 0.56 < 0.70; C 8k 0.49). First
settled window = NO EFFECT on scope (0.653 vs 0.55). Both segments that doubled the far protocol share (3d, 3f: 1.6 % vs 3c's
0.8 %) swung; the scope contrast did not stop it, and the release run's mix changed three times in two days to chase one cell.
**Change:** sci1_long3r_env.sh = 3c's exact token shares (proto_far v1 0.8 %, proto4k 0.846 %, chat 19.9 %, same CYCLE_SOURCES),
from 3f's next latest.pt (kept as seg3f_end.pt), all cursors kept, LR 8e-4 constant as before. Worktree qllm_sci1e @ 91a87ed.
**What is known about 3c:** one save only (41B, 0.5B into 3c): P_fu 8k 0.18, scope_fu 2k 0.73 / 8k 0.17. "3c does not swing" is a
prior, not a measurement; 3r measures it.
**Guard (same as 3f, per save):** P_fu 8k <= 0.20 and scope_fu 8k <= 0.30; two saves in a row failing = stop and talk. Reads at
+0.25B / +0.5B (Library CPU guard, 8k n 40; the 4090 is the user's from 10:45), then the milestones as usual.
**Accepted cost:** the main run gives up on teaching scope (41B already scope_fu 2k 0.73): the release notes must say the body
knows a rule exists more than what it is about. Scope, the excursions and the LR question move to 100M / 1B twins on the 4090
(suggested cell: far share 0.8 vs 1.6 % x constant LR as is vs half).
**Reading (Lead GO 09:2x IST, recorded before any 3r number):** guard on every scored save, two consecutive breaks = stop and
talk; NO WORKS bar on scope (not expected to improve); s8 / clk2 / t40 / HS / ARC-e guards as before, now against the 45B CHECK
(s8 @8k 3-save >= 0.945 - 0.05; HS / ARC-e not below 45B by > 0.01; clk2 8k 3-save >= 0.78).
**3f VERDICT (Lead, 09:2x IST 11-18): NO EFFECT on scope in the first settled window (scope_fu 2k mean 0.653 vs 0.55); the guard
broke at 45B and at +1.5B; superseded by 3r.**
**Next (4090, after the user's day, ordered by Lead):** 100M / 1B twins: far share 0.8 vs 1.6 % x constant LR as is vs half; the
split-geometry arm; the no-far-docs GDN / nochrono cell. Write-ups due before that window.
**Closest prior work:** none needed (a rollback to an earlier mix; plumbing, no claim).
**3r STARTED 09:56 IST 2026-11-18** from seg3f_end.pt = step **700,000 (45.875B)** (3f stopped 09:56:45; 6000 worktree qllm_sci1e @
91a87ed, tmux sci1s3r, log logs/v13_sempty_sci1_long3r_91a87ed_20261118_0426.log, run dir sci1_long3r_91a87ed). [resume] shows the
3c weights and all cursors (proto_far 90,932); first step 700,050 loss 1.848 lr 8e-4. Snapshots +0.25B 703,815 / +0.5B 707,630
(tmux mind3rsnap) -> Library CPU guard.
**3r +0.25B (snap 704,000; Library CPU guard 8k n 40):** P_fu 0.175, scope_fu 0.300 -> inside the guard (<= 0.20 / <= 0.30), scope at
the edge; UNCLEAR by the n-40 bands (>= 0.35 broken / <= 0.15 clear). C +0.687, twin unit_acc 0.85.
**3r +0.5B (snap 708,000; CPU guard 8k n 40):** P_fu 0.150, scope_fu 0.125 -> **CLEAR** (C +0.549, unit_acc 0.72). Guard so far: edge, clear; no break.
**46B CHECK (step 701,914 = 3r + 1,914 steps, 46.0B tokens; 15:54-16:15 IST 11-18 on the 4090; outputs logs/*/v13_sempty_*sci1_step_701914_91a87ed*):**
s8 @8k **0.975** (shift 0.955, affine 0.995; 45B 0.945), skill @8k (A_state) **0.645** (45B 0.71), t40 16k **0.978** (45B 0.939; 8k 1.000),
clk2 8k **0.93** / 32k-run @16k 0.58 (45B 0.92 / 0.68), HS / ARC-e / PIQA 0.316 / 0.553 / 0.642 (45B 0.322 / 0.550 / 0.639). Guards vs 45B:
s8 OK, HS -0.006 / ARC-e +0.003 OK, clk2 OK. Skill keeps sliding (0.845 / 0.735 / 0.71 / 0.645 at 43-46B): watch item, no bar.
**GPU scope reads, n 100 (Library window 1, 16:16-16:52 IST; supersede the CPU guard reads):**

| save | L | C | scope_fu | prec_u1 | P_fu | twin unit_acc | guard |
|---|---|---|---|---|---|---|---|
| 46B (701,914; ~all 3f history) | 2k | +3.107 | 0.93 | 0.70 | 0.09 | 0.97 | |
| | 8k | +0.713 | **0.38** | 0.06 | 0.15 | 0.85 | **BREAK** (scope_fu > 0.30) |
| 3r +0.25B (704,000) | 2k | +2.561 | 0.86 | 0.69 | 0.06 | 0.99 | |
| | 8k | +0.602 | 0.26 | 0.02 | 0.15 | 0.76 | pass |
| 3r +0.5B (708,000) | 2k | +2.095 | 0.76 | 0.55 | 0.09 | 0.97 | |
| | 8k | +0.515 | 0.17 | 0.00 | 0.14 | 0.65 | pass |

Sequence since 45B: BREAK (45B) / pass (45.19B) / BREAK (+1.5B 3f) / BREAK (46B, 1,914 steps into 3r: the 3f state) / pass / pass:
no two breaks in a row under 3r; scope_fu 8k falls 0.38 -> 0.26 -> 0.17 and C 8k 0.71 -> 0.60 -> 0.52 together (the trade again).
**3r +1.0B (snap 715,500; CPU guard 8k n 40):** P_fu 0.150, scope_fu 0.025 -> pass (C +0.737, unit_acc 0.80).
**47B CHECK (step 717,173, 47.0B tokens; 18:58-19:25 IST 11-18 on the 4090, beside nt_gdn_nofar yielding):** s8 @8k **0.958** (shift 0.970,
affine 0.945; 46B 0.975), skill @8k **0.70** (46B 0.645), t40 16k **0.836** (8k 0.956; 46B 0.978), clk2 8k **0.89** / 32k-run @16k 0.56,
HS / ARC-e / PIQA **0.320 / 0.561 / 0.649** (ARC-e and PIQA the run's best so far). Guards vs 45B: all pass.
**47B scope (GPU n 100, Library window 2):** 2k C +2.061, scope_fu **0.50**, prec_u1 0.55, P_fu 0.00, unit_acc 0.97; 8k C +0.461,
scope_fu 0.01, P_fu 0.00, unit_acc 0.64 -> CLEAR (the cleanest save of the series). Under 3r: pass x4 after the 46B (3f-state) break.
3r paused at 721,000 (47.25B) for the 6000 test window.

## segment 3c: chat sources cycle (PRE-REGISTERED 15:59 IST 2026-11-16, before any number; user direction "base model WITH chat", Lead PRIORITY 1)

**Why:** segment 3's chat sources did not cycle, so R7's <= 4-epoch sizing never applied: tuluif ran dry at 32.95B, chatsr at 34.43B,
smoltalk2_mid at 38.46B, chatmt2 at 39.88B; from 39.9B the run trained at 0 % chat.
**Change:** code a31ed81 (CYCLE_SOURCES: a dry parquet source re-opens from its start up to N total passes; the pass number is saved
as `<name>@pass` beside the exact cursor). Segment 3c = segment 3's mix at the ramp's TARGET shares (chat 19.9 %: smoltalk2_mid 12 %,
chatmt2 6 %, chatsr 1.5 %, tuluif 0.4 %), no ramp, started from segment 3's latest save (moved to `seg3_end.pt`), CYCLE_SOURCES =
smoltalk2_mid:4, chatmt2:3 (chatmt's first epoch was its own source), chatsr:4, tuluif:4. Far block, rule docs, decontam v2 unchanged.
smoltalk2 SFT stays OUT (unseen for the later SFT stage; user 11-14).
**Verified before launch:** stub test (a31ed81); CPU smoke on the real tuluif stream: one pass 28,433 docs, cursor at the end -> re-opens
as pass 2 at row 0, counters tuluif@pass 2 / @r 3 after 3 docs.
**Checks:** the log shows `[cycle] ... pass 2` for all four sources within the first hour; window chat share back to ~20 %
(Lead's window-mix readout); no source `finished pass N` before the next milestones. Probes at the milestones as before (the far /
told / reach cells feed the recovery decision, a separate proposal).
**Launched 16:11 IST 2026-11-16** (6000, worktree qllm_sci1c @ 3a8d7fb, tmux sci1s3c, log logs/v13_sempty_sci1_long3c_3a8d7fb_20261116_1041.log,
from step 618,000 = 40.5B). All four `[cycle] ... re-opening from the start as pass 2` lines at the start (smoltalk2_mid after 4,779,894
docs, chatmt2 391,187, chatsr 142,179, tuluif 28,433); step 618,050 loss 1.63. Check 1 met.
**Closest prior work:** multi-epoch repetition of small high-quality sets (Muennighoff et al. 2023, "Scaling Data-Constrained LMs":
up to ~4 epochs costs little); plumbing, no claim.

## far block result part 2: told rule (FAR-TOLD, Library's jsonls) and the rollback decision (13:36 IST 2026-11-12)

| ckpt | told train 8k | told train 2k | a26 2k |
|---|---|---|---|
| M11 | 0.10 | 0.67 | 0.20 |
| 344k | 0.53 | 0.85 | 0.31 |
| 26B | 0.03 | 0.58 | 0.20 |
| 27B | 0.45 | 0.79 | 0.22 |

- 344k vs 27B on told 8k: +0.08 at n 100 (~1.6 sigma), within jitter; 26B's 0.03 is a real dip, not a broken file (Library verified files complete and the step correct; plain is 0.00 at 26B vs 0.37 at 27B). Unit-scored told 8k: 344k 0.59 vs 27B 0.50 (ce27cf6).
- Pre-stated rule: roll back only if told 8k ALSO favours 344k clearly. It does not, so **FROM = 28B** (step 427,252). Segment 3 switch armed on the 6000 at 13:36 IST (tmux sci1s3, log v13_sempty_sci1_seg3switch_91bdc49_20261112_0806.log).
- Honest record: every far cell leans to 344k (reach cells 3.5-4 sigma, told cells +0.06-0.09). The dry stretch most likely cost them.
- **Check at segment 3's next two milestones:** skill @8k, clk2 @16k, told 8k / 2k must recover toward 344k. If not, revisit the rollback (344k kept on d_drive).

## segment 3: fineweb-edu streaming + the chat ramp (PRE-REGISTERED 12:31 IST 2026-11-12, before any number; Lead's decisions, Library's R5 audit 26630a1)

**Start point:** segment 2's 28B milestone (step 427,252) or, by Lead's rollback rule, its best_model = step 344,000 (22.5B, before
fineweb ran dry) if told 8k ALSO favours 344k clearly (Library's FAR-TOLD). `sci1_seg3_switch.sh` (FROM=28b|344k), env
`sci1_long3_env.sh`, code `94eee16`+ (sources) / `8b2679d` (ramp, hold).
**Mix (R1-R7):**
- (a) replacements at their old shares: `fineweb100` = fineweb-edu sample-100BT streamed from HF (own cursor; 13-gram drop rate 0.005 %
  HS in the audit), `chatmt2` = chatmt epoch 2 (R7: 2 of 4).
- (b) the CHAT RAMP (R2: the one ramp in flight, 2B tokens, linear in cumulative tokens from the start checkpoint's counters): chat 5.4 %
  -> 19.9 %: smoltalk2_mid 2.2 -> 12 %, chatmt2 3.2 -> 6 %, `chatsr` (smol-rewrite + smol-summarize) 0 -> 1.5 %, `tuluif` (tulu-3
  personas instruction-following, the IF share) 0 -> 0.4 %; taken from dclm / fineweb100 in proportion (0.221 / 0.230 -> 0.150 /
  0.156). R7-sized for <= 4 epochs over 10B (tuluif ~11M tokens, chatsr ~54M, chatmt ~640M): **the IF share is small by data size**.
- smol-constraints OUT (47.6 % IFEval verbatim templates); smoltalk2_sft OUT (median 20.6k tokens, split across windows).
- far block (b2s_far 1.2 %, proto_far 0.8 %) and the rule docs unchanged.
- stream decontam index **v2** (v1 + IFEval, sha a4b66d1b; drop counts per source in the cursor / header); WARMUP 0; LR 8e-4.
- safety: `[DRY-SOURCE]` line on a dry or failed source (Lead's ntfy watcher), a failed network source is dropped, not fatal.
**Reads at each 1B milestone (CHECK + FAR-TOLD + the chat probe):** target = Library's simple-chat probe with its instruction-following
cell (yes / no, one word, multi-step; user: judged by our probes + human eval, no v11 comparison) and a fixed human-eval set of short
questions ("is water wet"-style); guards (R3): s8 @8k (3-save) >= −0.05, told 8k >= −0.10, t40 16k and clk2 8k >= −0.05, HS / ARC-e /
PIQA >= −0.01, per-source science holdout <= +0.03, web holdout (dclm / fineweb val) back toward the pre-dry 2.98. Also watched: the
single-gap reach cells (skill pack @8k, clk2 @16k) that the dry stretch cost (part 1b), expected to recover.
**Keep rule:** judged at ramp end + 1B (3B after the switch): keep the chat ramp if the chat probe's IF cell rises and no guard falls at
two consecutive milestones; else freeze the ramp at its current share. Release path: a short decay from the first milestone after
that, judged by the strict rule (3-save s8 + margin) plus the chat probe.
**Closest prior:** SmolLM2's staged mid-training / annealing mixes (smoltalk), OLMo-2 mid-training; difference: no claim (data
engineering for the product), judged by our own probes.

**far block RESULT, part 1b: single-gap reach cells (12:30 IST 2026-11-12; outside reviewer read, Mind verified from the diag jsonls):**

| point | skill pack @8k bare (n 200) | clk2 @8k 1x / 8x+c4 (n 100) | clk2 @16k 1x / 8x+c4 (n 50) | clk2 @32k 1x / 8x+c4 (n 50) |
|---|---|---|---|---|
| M11 | 0.095 | 0.10 / 0.77 | 0.08 / 0.20 | 0.00 / 0.02 |
| 344k (with fineweb) | **0.715** | 0.61 / 1.00 | 0.18 / **0.92** | 0.14 / **0.20** |
| 26B (no fineweb since 347k) | 0.515 | 0.49 / 0.97 | 0.10 / 0.68 | 0.02 / 0.00 |
| 27B | 0.540 | 0.52 / 0.98 | (re-run queued: partial file) | |

The far block lifted single-gap reach strongly (M11 -> 344k), and **the dry stretch (fineweb gone from 347k) cost part of it beyond the
jitter on two cells**: skill @8k −0.20 (~4 sigma at n 200), clk2 @16k 8x+c4 −0.24 (~3.5 sigma at n 50), while s8 @8k stayed flat. Lead's
rollback rule named s8 @8k, t40 8k / 16k and told 8k, not these cells: roll back to 344k only if told 8k ALSO favours 344k clearly;
otherwise continue from 28B and record this as a measured cost of the dry stretch, expected to recover with fineweb-edu (checked at the
next two milestones).

## far block: distance-supervised rule data in the long run (PRE-REGISTERED 01:46 IST 2026-11-07, before any number; Lead GO for option (b) + Library's proto_far)

**Why:** "FINDING: the far rule margin is the decay's first casualty": nothing trains the far carry (b2s: 0 / 600 prompts reach 2k;
proto4k: 13 % reach 2k, none 4k). **What:** segment 2 of the long run, from its M11 (`sci1_long_3c2bf1f/step_167849`, ~08:20 IST),
cont3 mix + the FAR BLOCK ramped in (R2: one ramp, 2B tokens, linear in cumulative tokens, `sci1_long2_env.sh`):
- `b2s_far` 0 -> 1.2 % of tokens (`v7/data.py _b2far_text_iter`): one B2 rule prompt per doc (base tasks; `_new` never; seeds 2e7+)
  in the probes' spread layout, 4 or 8 shots, total uniform 1,800-3,200 tokens of DCLM TRAIN filler (exclude_holdout; pool
  sha 78b23877). Measured (600 docs): median 2,531 / p90 3,044 / max 3,194 tokens; 500 / 600 >= 2,048; 0 >= 4,096.
- `proto_far` 0 -> 0.8 % (Library, b502422): one stated protocol per doc, first answer 2,061-3,012 tokens after it (600 / 600 >= 2k),
  doc <= 3,093 tokens; leak-free vs A26 wordings.
- Taken from dclm / fineweb in proportion (0.23083 / 0.24033 -> 0.22103 / 0.23013); b2s / proto4k / recall4k unchanged (R1).
- Packer hold: both sources are placed whole inside one 4096 window (training state does not cross windows; ANSWER_IN_WINDOW).
  Measured: far docs kept whole with their answer weight 37.8 % (no hold) -> 100 %; streams without far sources bit-identical.
- WARMUP 0 at the switch (optimizer kept; s8reg B: a 500-step re-warm cost s8 ~0.10); constant LR 8e-4.
**Reads (each 1B milestone, s8 as the 3-save mean where saves exist + the margin):** s8 @2k / @4k (labelled TRAINED DISTANCE from
this segment on), **s8 @8k = the headline (beyond the 4096 window, never trained)**, told twin 2k / 8k (Library), needles t40 16k,
clk2 8k, HS / ARC-e / PIQA, per-source holdout, the realized far-block token share and answer-token loss-weight share ([mix] line).
**Keep rule (R3, judged at ramp end + 1B = 3B after the switch, read every milestone):** keep the far block if s8 @8k rises by
>= +0.05 over M11's (3-save mean) AND told 8k rises; guards: needles 16k and clk2 8k >= -0.05, HS / ARC-e / PIQA >= -0.01, per-source
science holdout <= +0.03. If a guard falls at two consecutive milestones, the ramp freezes at its current share. A short decay is then
tested from the first milestone after ramp end (release path, strict rule).
**Prediction:** s8 @2k / @4k rise to >= 0.8 within 1-2B (trained distance); @8k moves less (+0.05-0.15); the decay's far-margin loss
shrinks because the far carry is now in the trained distribution.
**Closest prior:** long-context data engineering / length curricula (e.g. Fu et al. 2024 "Data Engineering for Scaling LMs to 128K",
LongAlign) and synthetic long-dependency data; difference: distance-supervised IN-CONTEXT RULE induction for an O(1)-state body, with
an untrained 8k cell as the headline.

## FINDING: the far rule margin is the decay's first casualty (01:29 IST 2026-11-07; four-way, Lead + outside reviewer + Mind)

**Evidence (all SCI-1 386M, same s8 protocol, logs/s8reg/):**
1. Plain 1B decay from M9 (`8290f7d`): s8 D 0.770 -> 0.231 while NLL 3.555 -> 3.383, ARC-e 0.519 -> 0.567 (same stream as the
   constant-LR M10: an LR-only comparison, 0.231 vs 0.423).
2. Weight interpolation M9 <-> Decay_M9: s8 falls ~linearly in l (0.77 / 0.69 / 0.57 / 0.43 / 0.23); the NLL gain arrives late.
3. Rule-block x2 decay from M10 (C2, `1b8607d`): 0.423 -> 0.359 (low start, little to lose).
4. Rule-block x2 decay from M9 (C3, `95498f3`): 0.770 -> 0.457, margin +1.86 -> −0.17 over 0.25B, about linear in the LR fall.
**Cause candidate (measured):** b2s, the only rule-induction source, joins 6 prompts per doc; its spread prompts reach at most 1,572
tokens and **0 of 600 reach the probe's 2k distance**; 24 % lose the answer weight (ANSWER_IN_WINDOW). Nothing in the stream trains the
far carry; a decay sharpens the model toward the trained distribution and the untrained far margin goes first.
**What it means:** the fix is in the data (b2s_far, pre-registered next), not in the decay recipe. No interim release (Lead): the paper
states M9 (rules: s8 0.770, margin +1.86) and Decay_M9 (benchmarks) side by side; the release waits for a b2s_far point after >= 2B + a
short decay, judged by the strict rule (3-save s8 mean + margin). Closest prior: catastrophic forgetting of rare skills under LR
annealing / annealing-phase data curation (MiniCPM, OLMo-2); difference: the skill is in-context rule carry over distance in an O(1)
state, measured by margin through the decay.

## s8 checkpoint jitter (found 21:10 IST 2026-11-06 in C2's first point; Lead's rules adopted)

**Seen:** C2's first save is M10 + 10 optimizer steps (LR ~8e-4): s8 D 0.423 -> **0.331** with the same deterministic probe (400 trials,
seed 7000). Where margins sit near 0 (2k / 4k), single-checkpoint s8 carries ~**±0.1 jitter** from trivial weight noise. The
milestone swings (0.78 / 0.44 / ... / 0.77 / 0.42) are within reach of that; the ctl gaps (PAM 0.523 vs T++ 0.149 vs Mamba-2 0.074)
are far outside it.
**Rules from now (Lead):** (1) s8 is reported as the mean over 3 adjacent saves where they exist, always with the mean target margin
beside the accuracy. (2) C2's pre-registered bar stays and is only made STRICTER: a point passes if it meets all three bars AND its
adjacent saved points' mean s8 is >= 0.70 too. (3) Before a C3 from M9: measure M9's neighbourhood. On disk near 137,331 there is only
138,500 (+1,169 steps after the re-warm: 0.667); best_model is at 128,000. Plan: M9 + 10 / 20 / 50 steps at constant LR 8e-4
(WARMUP 0, the cont3 stream from M9's cursor) on the 4090 at micro-batch 1, then s8 on each: was 0.77 the trend or a high draw?

## SCI-1 milestone 10 (official; CHECK re-run on the 4090, done 20:32 IST 2026-11-06; `sci1_long_3c2bf1f` step 152,590 = 10.0B = M9 + 1.0B, cont3 mix, constant LR)

| | M9 (9.0B) | **M10 (10.0B)** | M10-worked (failed segment) |
|---|---|---|---|
| s8 D (2k+4k) | 0.770 | **0.423** (affine 0.75 / 0.36, shift 0.46 / 0.12) | 0.418 |
| t40 needles 4k / 8k / 16k | 1.000 / 0.989 / 0.931 | **0.961 / 0.850 / 0.803** | 0.836 / 0.692 / 0.622 |
| HS acc_norm / ARC-e acc / PIQA acc_norm | 0.333 / 0.519 / 0.621 | **0.332 / 0.539 / 0.616** | 0.333 / 0.532 / 0.617 |
| GSM8K ON / OFF, leak | 0 / 0, 0 | 0 / 0, 0 | 0.05 / 0.06, 0 |

The s8 drop (0.770 -> 0.423) confirms Lead's review-page run exactly; **needles also fell on the unchanged mix** (8k 0.99 -> 0.85,
16k 0.93 -> 0.80), so reach drifts too, and `worked` made it worse (8k 0.69). The first CHECK attempt (19:44 IST) OOM'd beside cx_m2
and flagged itself done; fixed (22884db: .checked only when all outputs exist).

## s8 drifts on an UNCHANGED mix; two clean comparisons from M9 (20:05 IST 2026-11-06; Lead + outside reviewer; Mind checked)

**Same stream, checked:** every per-source token counter of `sci1_decay_m9_8290f7d/latest.pt` equals the new M10's
(`sci1_long_3c2bf1f/step_152590.pt`; b2s 145,132,567, dclm 2,054,762,271, ...): both read the same 1B after M9 (cont3, seed 43).
- **LR only (decay vs constant LR, same 1B):** s8 D 0.231 vs 0.423; ARC-e 0.567 vs 0.540; HS 0.343 vs 0.333.
- **Mix only (`worked` ~2 % vs none, same start, LR and 1B):** M10-worked vs new M10: s8 D 0.418 vs 0.423 -> **`worked` is neutral on
  s8**; t40 8k 0.69 vs 0.85 -> **`worked` hurt needles**. Final `worked` verdict: it costs reach, not rules (not used).
- **s8 on the unchanged cont3 mix at constant LR:** M9 0.770 -> +0.08B 0.667 (s8reg B) -> +1.0B 0.423 (new M10; Lead's review-page
  run, 400 trials; my CHECK re-run confirms). So s8reg B's reading "the big drops are mix changes" is withdrawn: s8 at 2k / 4k rises
  and falls on a constant mix too (M4 -> M5, M9 -> M10).
**Why (generator, `v7/data.py` `_b2_text_iter`):** b2s spreads only half its prompts, with log-uniform 1-30-sentence gaps of
recall-bank filler; the probe's per-gap distance is ~290 tokens at 2k and ~585 at 4k on held-out web filler: the 2k cell is in ~10 %
of training prompts, the 4k cell at the edge, with another filler. s8 at 2-4k is a weakly anchored extrapolation (s8 @0 = 1.00 at
every check); nothing in the stream holds it, so it drifts. **Long-run mix proposal (to Lead; pre-registered before the next
milestone, C2 picks):** (a) if C2 lifts s8 from 0.42: rule block x2 (C2's weights); (b) else a new ramped source `b2s_far`
(every prompt spread, per-gap log-uniform 50-600 tokens of web-train filler, one prompt per doc, 0.5 -> 1.5 % over 2B from web, b2s
kept); paper caveat for (b): s8 at 2k / 4k becomes a trained distance, the beyond-length claim moves to the 8k cell.
**Lead's condition on (b), adopted:** the probe's held-out task values (`_new`) and held-out filler stay out of training; the
headline far cell is s8 @8k (beyond the 4096 window, never trained by b2s_far); 2k / 4k are labelled "trained distance" from that
milestone on in the paper.
**b2s measured (outside reviewer, Lead checked the code; 200 docs / 1,200 prompts):** `_b2_text_iter` joins its 6 prompts with
'\n\n' into ONE document (no <|endoftext|>, so RESET_AT_EOS never resets between the 6 rules: prompts 2-6 train overwriting the previous
rule in-doc). Unspread prompts ~23 tokens; spread prompts median 472, p90 969, max 1,572 tokens: **0 of 600 spread prompts reach
2,048** (the probe's 2k cell). ANSWER_IN_WINDOW=1 drops the 100x answer weight to 1x on ~24 % of spread prompts. This is the measured
cause candidate for the weak far anchoring. **b2s_far design adds:** per_doc = 1 (no in-doc overwrite; ends with EOS), every prompt
spread, a single prompt capped at <= 3,200 tokens so it lands inside one 4096 window with its answer weight. **b2s itself is kept
unchanged (R1):** its in-doc rule switching plausibly trains erase-at-key (the kn knockout: erase-at-key carries the rule), and the probe
needs one rule per prompt, so it is not in conflict.

## SCI-1 milestone naming (user via Lead, 14:20 IST 2026-11-06)

**M10 = the long run's first milestone** (`sci1_long_3c2bf1f`, global step 152,590 = 10.0B tokens = M9 + 1.0B on the cont3 mix,
~19:30 IST): the official 10th point of the SCI-1 series (paper Fig 4, the milestone table, the review page), with the full CHECK +
Library's suite + Lead's review page. The cont4 checkpoint (`sci1_cont4_8284510` step 152,590, + `worked`) is **M10-worked (failed
segment)**: kept only as that record, out of the series. Mentions of "M10" in sections written before 14:17 IST 2026-11-06 mean
M10-worked. Release path: arm D now; if none passes, a short decay saved every 0.05B, preferably from the new M10.

## s8reg: is the s8 drop LR-driven, a re-warm transient or a mix effect? (PRE-REGISTERED 2026-11-06 12:51 IST, before any number; Lead GO)

**Facts first.** `--continue_from` keeps the model AND the optimizer state; only the LR schedule is fresh (500-step warmup 0 -> 8e-4).
So a "segment switch" = an LR dip to ~0 and back + the new mix / stream. The decay kept no mid-point checkpoint (KEEP_EVERY 0), so
Lead's arm (a) needs a new run. ctl_pam_main's `stable_final.pt` (before its 0.1B decay) exists, and its stable s8 was never read: the
ctl result "decayed PAM kept s8 0.523" does NOT show that its decay was harmless.

**Arms (s8 spread probe, the CHECK protocol: b2, shift_wide_new + affine_map_new, 8 shots, 400 trials, seed 7000, 0/2k/4k/8k):**
- **A** ctl_pam_main stable_final (103M, 0.9B stable) vs its 0.1B decay (0.523, read).
- **B** SCI-1 long run step 138,500 = M9 + 500 re-warm steps + 669 stable steps on the SAME cont3 mix (snapshot, sha-checked).
- **C** (needs the 6000; Lead's call) a 0.25B decay from M9 to LR 0, saving every 0.05B -> s8 against LR along the decay.
- (optimizer reset vs kept: only if A-C leave it open)

**Readings, fixed now:**
- A: stable within 0.05 of 0.523 -> a short decay is harmless at 103M; SCI-1's drop is tied to its 1B length / floor -> C decides.
  A >= 0.60 -> decays cost s8 in general (ctl too).
- B: >= 0.65 -> the re-warm alone is harmless: segment drops come from the mix change (cont1 / cont4 both changed the mix) and
  M10's s8 loss goes back to the `worked` mix change. B <= 0.45 -> the LR dip itself costs s8 (a transient); the long run's 19:30
  milestone (same mix, stable LR) shows whether it recovers.
- C: s8 falls steadily as the LR falls -> LR-driven: the release decays to a non-zero floor or is shorter (picked by C's curve).
  s8 falls early, then rises -> a transient.
**A + B RESULT (14:11 IST; logs/s8reg/, same protocol, each at its reference code):**
- **A:** ctl_pam stable_final s8 D **0.358** -> its 0.1B decay 0.523 (**+0.165**). Neither pre-registered reading (it was "within 0.05"
  or ">= 0.60"): at 103M a short decay RAISES s8. Decays do not cost s8 in general; the SCI-1 drop is specific to that decay
  (1B long, LR to 0, 386M, cont3 mix) -> C (or D) decides.
- **B:** SCI-1 step 138,500 (M9 + re-warm + 669 stable steps, same mix) s8 D **0.667** (M9 0.770; 2k 0.83, 4k 0.51). >= 0.65 ->
  by the fixed reading the re-warm alone is harmless (a -0.10 dip, against -0.54 in the decay); the large segment drops (76k, 152k)
  go back to their MIX changes. So M10's s8 loss is again read as `worked` (the confound is mostly removed), with needles.
  (WITHDRAWN later the same day: the new M10 fell to 0.423 on the UNCHANGED mix; see "s8 drifts on an UNCHANGED mix".)

**Arm D (added 13:27 IST, Lead + outside reviewer; pre-registered before any number):** weight-space points between M9 and its decay,
no training. theta = (1 - l) M9 + l decay for l = 0.3 / 0.5 / 0.7, and a graft = M9's notebook / reader / echo params (blocks.*.pam*,
blocks.*.reader, echo_*: 60.2M of 385.8M = 15.6 %) + the decay's MLP / embeddings / norms / head. Read per point: s8 D (2k + 4k),
t40 16k, HS / ARC-e / PIQA (the CHECK's code, ../qllm_sci1check), held-out NLL (24 x 2048, pretrain_holdout_v1, also for M9 and the
decay). **Release candidate = a point with s8 D >= 0.70 AND ARC-e >= 0.55 AND NLL within +0.03 of the decay's.** Closest prior:
WiSE-FT / model soups / checkpoint averaging; release engineering, no claim. Scripts `v13_sempty/tmp/s8reg_d.py` / `s8reg_d.sh`.
Outside hypothesis it tests: the decay's sharper LM prior outvotes a faint 2k-4k trace (M9 margin +0.23 nats); if so, lerp points
keep s8 near M9's until l is large, and the graft (old notebook, new prior) does NOT restore s8.

**D RESULT (15:01 IST; logs/s8reg/; NO release candidate):**

| point | s8 D | t40 16k | HS | ARC-e | PIQA | held-out NLL |
|---|---|---|---|---|---|---|
| M9 (l = 0) | 0.770 | 0.931 | 0.333 | 0.519 | 0.621 | 3.555 |
| lerp30 | 0.689 | 0.964 | 0.340 | 0.532 | 0.625 | 3.529 |
| lerp50 | 0.571 | 0.969 | 0.341 | 0.545 | 0.627 | 3.519 |
| lerp70 | 0.426 | 0.964 | 0.342 | 0.557 | 0.628 | 3.463 |
| decay (l = 1) | 0.231 | 0.908 | 0.343 | 0.567 | 0.631 | 3.383 |
| graft (M9 notebook + decay rest) | 0.338 | 0.847 | 0.340 | 0.532 | 0.618 | **3.916** (broken) |

Rule (s8 >= 0.70 AND ARC-e >= 0.55 AND NLL <= decay + 0.03 = 3.413): no point passes; lerp30 is the closest on s8, lerp70 on ARC-e,
none on NLL. **Reading:** s8 falls about linearly along the path (-0.11 per 0.2 of l) while most of the NLL gain sits at the end
(l 0.7 -> 1: -0.08 of -0.17): the decay's prose gain and its rule loss are separate directions. The outside hypothesis ("s8 holds until l is
large") is NOT supported. The graft is incompatible (NLL worse than either end), so it says nothing about where the rule lives. Needles
16k are higher at every mid point (0.96-0.97) than at both ends. Next: the short decay curve (C), Lead's call on the 6000.

**C2 (PRE-REGISTERED 15:04 IST 2026-11-06, Lead GO; release engineering, no claim; closest prior: MiniCPM / OLMo-2 anneal mixes).**
From the new M10 (`sci1_long_3c2bf1f/step_152590.pt`, 10.0B), a 0.25B decay (LR 8e-4 -> 0 linear, 3,815 steps) on the cont3 mix with
the rule block x2 (doc weights b2s 0.01696 -> 0.03392, recall4k 0.00377 -> 0.00754, proto4k 0.00846 -> 0.01692; the added 0.02919 from
dclm / fineweb in proportion: 0.21653 / 0.22544); env `v13_sempty/tmp/sci1_c2_env.sh`, orchestrator `sci1_c2_orchestrate.sh` (pauses
the long run at M10, runs C2, resumes the long run exactly at M10). Saves at global multiples of 763 (~0.05B): M10 + 10 / 773 / 1,536 /
2,299 / 3,062 steps and the end (+3,815). Each point gets s8 D (2k + 4k), t40 16k, HS / ARC-e / PIQA and the held-out NLL (s8reg D's
protocol). **Release candidate = the FIRST saved point with s8 D >= 0.70 AND ARC-e >= 0.55 AND NLL <= 3.413 (decay_M9 3.383 + 0.03);
ties broken by needles 16k.** If none passes: C1 (plain curve) next, as paper data or a fallback. The realized rule-block token share
is read from the trainer's [mix] line at the first val.
**Paper finding (Lead):** the s8reg D lerp curve (s8 falls ~linearly in l, the NLL gain arrives late) = prose sharpening and rule loss
are separate directions in weight space.

**C3 RESULT (01:28 IST 2026-11-07; logs/s8reg/*C3*; NO release candidate):**

| point (from M9) | s8 D | margin | s8 @8k | t40 16k | HS | ARC-e | PIQA | NLL | wdelta pam vs M9 |
|---|---|---|---|---|---|---|---|---|---|
| M9 | 0.770 | +1.86 | 0.29 | 0.931 | 0.333 | 0.519 | 0.621 | 3.555 | 0 |
| +9 steps (137,340) | 0.737 | +1.57 | 0.256 | 0.900 | 0.334 | 0.523 | 0.624 | 3.551 | 0.009 |
| +0.05B (138,103) | 0.732 | +1.28 | 0.272 | 0.792 | 0.332 | 0.531 | 0.627 | 3.531 | 0.088 |
| +0.10B (138,866) | 0.593 | +0.62 | 0.167 | 0.956 | 0.330 | 0.547 | 0.624 | 3.501 | 0.107 |
| +0.15B (139,629) | 0.629 | +0.74 | 0.164 | 0.839 | 0.336 | 0.555 | 0.626 | 3.478 | 0.114 |
| +0.20B (140,392) | 0.565 | +0.35 | 0.136 | 0.822 | 0.335 | 0.548 | 0.622 | 3.452 | 0.116 |
| end 0.25B (141,146) | 0.457 | −0.17 | 0.094 | 0.836 | 0.335 | 0.560 | 0.626 | 3.431 | 0.117 |
| A3 = avg(last three) | 0.561 | +0.32 | 0.130 | 0.836 | 0.335 | 0.556 | 0.624 | 3.435 | 0.115 |

No point meets the strict rule (s8 >= 0.70 only at +9 / +0.05B, where ARC-e and NLL miss; NLL never reaches 3.413). **Reading:** from a
HIGH start, the rule-x2 decay does NOT hold s8: 0.77 -> 0.46 (margin +1.86 -> −0.17) over 0.25B, about linear in the LR fall, as in
the lerp curve; C2's "held" (0.42 -> 0.36) was partly a low start with little left to lose. Three decays (M9 plain 1B, M10 rule x2, M9
rule x2) and the lerp now agree: **the far rule margin is the decay's first casualty, and doubling the short-gap rule docs does not
protect it.** Consistent with the b2s measurement (0 of 600 training prompts reach the probe's 2k distance): nothing in the stream
trains the far carry, so a decay that sharpens the trained distribution sharpens it away. **Implication:** the fix is in the data
(b2s_far, pre-registered next), not in the decay recipe; the release waits for a b2s_far-trained point (Lead's call).

**LAWA-lite RESULT (23:31 IST 2026-11-06; logs/s8reg/*LAWA*; NO release candidate):**

| point | s8 D | margin | t40 16k | HS | ARC-e | PIQA | NLL |
|---|---|---|---|---|---|---|---|
| M9 | 0.770 | +1.86 | 0.931 | 0.333 | 0.519 | 0.621 | 3.555 |
| L1 = mean(M9, 138,500) | **0.777** | +1.65 | 0.875 | 0.333 | 0.524 | 0.621 | **3.482** |
| L2 = mean(128,000, M9, 138,500) | 0.688 | +1.18 | 0.928 | 0.333 | 0.542 | 0.631 | 3.509 |
| L3 = 0.6 M9 + 0.25 x 138,500 + 0.15 decay | 0.744 | +1.46 | 0.939 | 0.336 | 0.530 | 0.627 | 3.481 |
| Decay_M9 | 0.231 | | 0.908 | 0.343 | 0.567 | 0.631 | 3.383 |

All miss ARC-e >= 0.55 and NLL <= 3.413. As expected (pre-registered): the averages keep s8 near M9's while buying part of the decay's NLL
gain. Notable: **L1, a plain average of two constant-LR saves 1,169 steps apart, keeps s8 (0.777, margin +1.65) and buys ~43 % of the
1B decay's NLL gain (3.555 -> 3.482 vs 3.383)**, no training: averaging removes the noise the decay removes, without the decay's rule
loss. Paper line candidate (no claim beyond LAWA's).

**C2 RESULT (22:21 IST 2026-11-06; logs/s8reg/*C2*; NO release candidate):**

| point (from M10) | s8 D | margin | t40 16k | HS | ARC-e | PIQA | NLL | wdelta vs M10 (embed+head / cgu / pam / reader+echo) |
|---|---|---|---|---|---|---|---|---|
| M10 (start) | 0.423 | −0.36 | 0.803 | 0.332 | 0.539 | 0.616 | 3.546 | 0 |
| +10 steps | 0.331 | −0.69 | 0.794 | 0.332 | 0.533 | 0.614 | 3.540 | 0.012 / 0.008 / 0.008 / 0.009 |
| +0.05B | 0.261 | −0.91 | 0.900 | 0.335 | 0.528 | 0.622 | 3.503 | 0.138 / 0.087 / 0.087 / 0.092 |
| +0.10B | 0.266 | −0.94 | 0.908 | 0.335 | 0.545 | 0.621 | 3.461 | 0.169 / 0.105 / 0.106 / 0.113 |
| +0.15B | 0.416 | −0.32 | 0.906 | 0.338 | 0.552 | 0.621 | 3.441 | 0.183 / 0.113 / 0.113 / 0.121 |
| +0.20B | 0.407 | −0.33 | 0.869 | 0.338 | 0.554 | 0.623 | 3.415 | 0.189 / 0.115 / 0.116 / 0.123 |
| end 0.25B | 0.359 | −0.56 | 0.886 | 0.338 | 0.557 | 0.626 | 3.396 | 0.190 / 0.115 / 0.116 / 0.123 |

No point meets s8 >= 0.70. **Reading:** the rule-x2 decay roughly HOLDS s8 at its start level (0.42 -> 0.36 at the end, margin
dipping then recovering mid-decay), where the plain 1B decay from M9 lost 0.54 (0.77 -> 0.23), while it still buys most of the NLL /
ARC-e gain (end NLL 3.396 vs Decay_M9 3.383; ARC-e 0.557 vs 0.567); needles 16k rise (0.80 -> 0.89). So rule x2 plausibly protects the
rule carry through a decay, but it can only keep what the start point has, and M10 starts at 0.42. Caveat: one run, a different start
point and a 4x shorter decay than Decay_M9. **Next release candidate: C3** = the same 0.25B rule-x2 decay from M9 (margin +1.86).
**Long-run mix (pre-registered rule above):** C2 did not LIFT s8 from 0.42 -> option (b) `b2s_far` (to be pre-registered with numbers).
The 6000: the long run's restart REFUSED at 21:54 IST (disk 23 GB < 25 after C2's saves); C2's six saves sha-verified on d_drive and
deleted on the 6000 (49 GB free); long run relaunched 22:10 IST from M10.

**C3 (PRE-REGISTERED 22:23 IST 2026-11-06, before any number; Lead GO; release engineering, no claim).** C2's recipe from M9 (cont3
step_137331, sha 218922f9, s8 0.770, margin +1.86): 0.25B decay to LR 0 with the rule block x2 (`sci1_c3_env.sh`), saves at global
multiples of 763 (CORRECTED: global 137,340 / 138,103 / 138,866 / 139,629 / 140,392 = M9 + 9 / 772 / 1,535 / 2,298 / 3,061; end +3,815 = 141,146), a val + [mix] line at each. The long run pauses after
its next latest.pt save and resumes from its own latest.pt (`sci1_c3_orchestrate.sh`); each save is copied, sha-verified and deleted
on the 6000 by `s8reg_c3.sh` (disk floor). Reads per point as C2 (s8 D + margin, t40 16k, HS / ARC-e / PIQA, NLL, wdelta vs M9).
**Release rule (strict, Lead):** the FIRST point with s8 D >= 0.70 AND ARC-e >= 0.55 AND NLL <= 3.413 AND its adjacent saves' mean
s8 D >= 0.70; ties by needles 16k; the margin is reported beside every s8. **Expectation (from C2):** s8 stays near M9's level (C2 held
its start within -0.06) while NLL / ARC-e approach the decay's; the end point is the most likely to pass ARC-e and NLL, the middle ones
to pass s8.
**C3-A3 (added 22:27 IST, BEFORE C3 started; outside reviewer via Lead):** the equal-weight average of C3's last three saves (+0.15B / +0.20B /
end) as one more candidate, same reads; strict rule with the margin clause of LAWA-lite (no adjacent saves exist for an average):
s8 D >= 0.70 AND margin >= +1.0 AND ARC-e >= 0.55 AND NLL <= 3.413.

**LAWA-lite (PRE-REGISTERED 21:23 IST 2026-11-06, before any number; outside reviewer + Lead; release engineering, no claim; closest prior:
LAWA / checkpoint averaging, model soups).** Weight averages of saves already on disk, no training (`s8reg_d.py avg`, built on CPU):
- **L1** = mean(M9 137,331, 138,500 [M9 + re-warm + 669 steps]); wdelta between them 0.10-0.16 per group.
- **L2** = mean(cont3 best_model 128,000, M9, 138,500).
- **L3** = 0.6 M9 + 0.25 x 138,500 + 0.15 Decay_M9 (= mean(L1, lerp30)).
Reads per point as arm D (s8 D + margin, t40 16k, HS / ARC-e / PIQA, held-out NLL). **Release rule unchanged and strict:** s8 D >= 0.70
AND ARC-e >= 0.55 AND NLL <= 3.413, plus Lead's jitter rule (here: the margin must stay clearly positive, >= +1.0, since no adjacent
saves exist). **Expectation, stated now:** the averages keep s8 near M9's but buy only a small part of the decay's NLL / ARC-e gain, because
the averaged saves sit on the constant-LR plateau; L3 is the best NLL of the three and the most likely to fail s8.

**Needles (Lead):** this decay KEPT t40 (8k 0.978, 16k 0.908) while M10 lost it (8k 0.69, 16k 0.62), so the regime switch does not
explain M10's needle loss: "`worked` hurt needles" stays the live reading; only the s8 part is unresolved.

## SCI-1 decay-from-M9 CHECK (2026-11-06 12:50 IST; `sci1_decay_m9_8290f7d/latest.pt`, 10.0B tokens = M9 9.0B + 1.0B decay on the cont3 mix)

| | M9 (stable, 9.0B) | decay (10.0B) |
|---|---|---|
| s8 D (2k+4k) | 0.770 | **0.231** (shift_wide 2k 0.13 = chance) |
| t40 needles 8k / 16k | 0.989 / 0.931 | 0.978 / 0.908 |
| HellaSwag acc_norm | 0.333 | **0.343** |
| ARC-e acc | 0.519 | **0.567** |
| PIQA acc_norm | 0.621 | **0.631** |
| GSM8K ON / OFF, leak | 0.00 / 0.00, 0 | 0.00 / 0.00, 0 |

**The s8 series over all SCI-1 checks:** 0.32 / 0.54 / 0.71 / **0.78** (61k) / 0.44 (76k, cont1 start) / 0.44 / 0.59 / 0.59 /
**0.77** (137k) / 0.42 (152k, cont4 start) / **0.23** (decay). Every large drop sits at an LR / optimizer regime change (a new segment
= fresh budget + 500-step re-warm, or the decay), then climbs back over 2-3B tokens. **Correction:** the M10 verdict "`worked` failed
its keep rule" is CONFOUNDED with the cont4 segment switch -> downgraded to **unresolved** (not re-run). The induced-rule carry
across filler is fragile to LR changes while recall, needles and the benchmarks are not. Release: decided by Lead (base = this decay;
the rule-induction claim cites M9, both stated). Next (cheap, 4090): s8 on M9 + a few hundred re-warm steps vs M9 at the decay's
midpoint, separating the LR from the optimizer state.

## SCI-1 decay from M9 (Lead GO 21:12 IST 2026-11-05)

`worked` dropped by the keep rule at M10 (s8 0.770 -> 0.418, needles 0.98 -> 0.57); **the decay branches from M9** (cont3
step_137331, 9.0B, sha256 218922f9), cont3 mix (the one that produced M9), the pre-registered decay recipe (1.0B, LR 8e-4 -> 0 linear),
launcher `v13_sempty/tmp/sci1_decay_from.sh`, dir `sci1_decay_m9_<hash>`. cont4 stopped after a clean save; its step_152590 and
latest.pt are kept (the long run may resume from them with another mix). M10 stays recorded as a failed segment (section below).

## SCI-1 M10-worked (FAILED SEGMENT, not in the milestone series; cont4 CHECK on the 4090, 21:10 IST, 2026-11-05; run `8284510`, step 152,590 = 10.0B tokens; 1.0B of + `worked`)

**The `worked` source fails its keep rule (indicative, but large):** s8 D **0.770 → 0.418**, far needles 4k + 8k **0.98 → 0.57**,
t40 at 16k **0.87 → 0.33**, clk2 8k 0.61 → 0.51; skill pack 8k 0.30 → 0.36; GSM8K ON / OFF 0.00 → 0.05 / 0.06; `<think>` leak
0.00; HS / ARC-e / PIQA 0.333 / 0.532 / 0.617 (flat). Keep rule: s8 +0.10 → it fell 0.35: **`worked` dropped.** (CORRECTED 2026-11-06, then s8reg B: the re-warm alone costs s8 only -0.10, so the s8 loss is again read as `worked`; the needle loss is NOT explained by it (the decay kept needles): `worked` hurt needles stays the live reading.) Reading: plain
worked solutions (2 % of tokens, chat-formatted, 392k short docs ~20 epochs at this share) taught the readout to answer from local
arithmetic text and cost the long-range carry (the lag rule says 1B is early, but the drop is 3x the noise). Consequence: the main
ends on the cont4 mix at 160,218, so a decay from its end would carry the damage; Lead decides between decaying from M9 (the best
checked point) and finishing the main on the cont3 mix.

## clk3: eval-time dilation of only the slow notebook heads (PRE-REGISTRATION, 17:08 IST, 2026-11-05; outside plan + Lead + Mind review; eval only)

**Question:** clk1 dilated every notebook head (prose cost) and clk2 whole memories; does dilating only the heads that ALREADY hold
long (selected by MEASURED realized retention, not by bias) extend rule-carrying past 4k without the prose cost?
**Model:** SCI-1 M9 (c3b, 386M, 6 notebook layers × 8 heads). Runner `v13_sempty/tmp/clk3_slowheads.py`.
**Measured first (dry run, 2 held-out seqs of 2048):** realized half-life (ln 0.5 / mean ln r over tokens) median **8.1 tokens**;
the six slowest (layer index among the notebook layers, head, realized / bias-only half-life): (4,0) 5,241 / 540; (2,1) 2,991 / 137;
(3,2) 1,343 / 152; (5,3) 758 / 74; (3,3) 737 / 134; (3,1) 621 / 126. So the outside plan's "stuck at 18–76" is wrong in both
directions: most heads forget within ~8 tokens in practice, the slow few hold far longer than their bias says.
**Arms** (retention r → r^(1/s) on the s8 probe's FILLER tokens only; echo via the model's span hook): base, slow_s4, slow_s8,
all_s4 (clk1-like), echo_s4 (clk2-like), slow_echo_s4. s8 spread probe, 200 trials, 0 / 2k / 4k / 8k; prose guard = held-out NLL
(24 × 2048) with the same dilation on positions ≥ 256.
**Added before any numbers (Lead / outside plan, 17:2x IST):** `coupled_s4`, `coupled_s8`, `coupled_echo_s4` = retention r^(1/s)
AND the Chrono warp g / s after its clamp, on the same heads and filler tokens (the clock slows with the decay; retention-only
dilation keeps the amplitude but lets the query rotate out of phase over 8k). Head selection reads ln r from each layer's own
`_decay` output after its `.to(batch, heads, time)`, not the trainer's `capture_decay` hook (which averages the wrong axis on the
real arm, Lead's finding; the trainer's retention panel is therefore suspect, a separate fix). The full run also logs the warp g
per slow head (clamp saturation at the −3 floor = a body question for the ladder).
**Rule:** clk3 works if an arm with the slow set raises D_far (mean of @4k and @8k) by ≥ **0.05** over base with prose NLL Δ ≤
**+0.01**; all_s4 is the comparison (dry run: NLL +0.11 vs slow +0.006).
**Prior work / difference:** test-time decay / timescale scaling in SSMs (e.g. Mamba's Δ, LongMamba-style dt scaling); the
difference is a head subset chosen by measured realized retention in a delta notebook, plus echo.

**clk3 RESULT (15:31 IST 2026-11-06; logs/diag/v13_sempty_clk3_m9_7a9eecd.json; 200 trials; NOT MET by its rule, narrowly):**

| arm | s8 D | D_far (4k+8k) | Δ D_far | NLL | Δ NLL |
|---|---|---|---|---|---|
| base | 0.760 | 0.444 | | 3.5549 | |
| slow_s4 / slow_s8 | 0.789 / 0.789 | 0.477 / 0.471 | +0.033 / +0.027 | 3.5631 / 3.5653 | +0.008 / +0.010 |
| all_s4 (clk1-like) | 0.662 | 0.316 | −0.128 | 3.6511 | +0.096 |
| echo_s4 (clk2-like) | 0.820 | 0.549 | +0.105 | 3.5595 | +0.005 |
| slow_echo_s4 | 0.836 | 0.564 | +0.120 | 3.5678 | +0.013 |
| coupled_s4 / s8 | 0.823 / 0.830 | 0.531 / 0.547 | +0.087 / +0.103 | 3.5713 / 3.5786 | +0.016 / +0.024 |
| coupled_echo_s4 | 0.846 | 0.589 | **+0.145** | 3.5762 | +0.021 |

Rule (a slow-set arm: Δ D_far >= +0.05 with Δ NLL <= +0.01): the slow-head retention arms stay under +0.05; every slow-set arm that clears
+0.05 costs NLL +0.013 to +0.024. **Not met.** What it says: (1) **the clock matters:** slowing the Chrono warp WITH the retention (coupled)
nearly triples the retention-only gain (+0.087 vs +0.033): with retention alone, the query rotates out of phase over 8k. (2) **3 of the 6
slowest heads sit at the warp clamp floor** (g = exp(−3) ≈ 0.05; 4 of 48 heads overall): the trained body asks for slower clocks than the
clamp allows. (3) Echo dilation alone remains the cheapest reach lever (+0.105 at +0.005). For the body (ladder / next body, not an eval
trick): widen the warp clamp floor (−3 → e.g. −6) and let the slow heads train slower clocks; closest prior: learned timescales in SSMs
(Mamba Δ, LongMamba); difference: a learned rotary clock in a delta notebook. Realized half-life median 8.7 tokens (dry run 8.1).

## nkr: notebook-key retrieval for ARC-c (PRE-REGISTRATION, 15:53 IST, 2026-11-05; the three step-backs' synthesis, Lead; eval only)

**Question:** can the trained mind ADDRESS its own library better than word overlap? BENCH-LIB on c3b (SCI-1 M9, 386M) found the
model reads a stated answer (ARC-c acc_norm 0.835) but BM25 top-3 adds nothing (0.278 vs bare 0.277): the limit is the address.
**Method (cheap rerank, no index build):** for each ARC-c test question, Library's BM25 top-**100** from the same science library
(`v14/arclib_index.py`, the BENCH-LIB library and query: question + all choices); rerank the 100 by the notebook's own match and
prepend the new top-3 exactly as BENCH-LIB's `library` arm (`"<p1>\n<p2>\n<p3>\n\n" + prompt`, mc_eval's scorer, acc_norm).
Score(passage) = Σ over the notebook layers and their heads of max over the passage's tokens of cos(q_head, k_head,t), with q = the
notebook's query projection at the question's last token and k = its unit key projections over the passage, both BEFORE the
Chrono rotation (position is meaningless across documents). One forward of the question and of each passage (100 passages × 1,172
questions; batched; within one 4090 pause).
**Arms:** bare (0.277), BM25 top-3 (0.278), **NK top-3** (the rerank), random-3 of the BM25 top-100 (control: does any reorder
help?), oracle (0.835).
**Rule:** NK top-3 ≥ BM25 top-3 + **0.05** (i.e. ≥ 0.33) AND ≥ random-3 + 0.03 → "the mind addresses its library better than
words", which earns the full key index (all passages keyed) as the next unit. Otherwise the rerank is closed and the notebook's
keys are not an address space for a corpus at this size. ARC-e reported with the same arms.
**Closest prior work / difference:** dense retrieval (DPR) and late-interaction max-sim (ColBERT), kNN-LM / Memorizing Transformers
(kNN over attention keys), HyDE. Difference: no retriever is trained; the scores are the frozen LM's own delta-notebook query / key
projections, i.e. the address its erase-at-key memory already uses.

**nkr RESULT (16:41 IST, 2026-11-05; `logs/v13_sempty_nkr_{nk,rand}_c3b_*.jsonl`, `logs/v13_sempty_nkr_run.log`; c3b, acc_norm, mc_eval's scorer).**

| arm | ARC-c | ARC-e |
|---|---|---|
| bare | 0.277 | 0.463 |
| BM25 top-3 (BENCH-LIB) | 0.278 | 0.526 |
| random 3 of BM25's top-100 | 0.266 | 0.501 |
| **notebook-key top-3 (NK)** | **0.260** | **0.484** |
| oracle (stated answer) | 0.837 | 0.875 |

**Verdict: FAILED, closed.** NK is below BM25 (−0.018 / −0.042) and below RANDOM picks from the same 100 (−0.006 / −0.017): the
notebook's raw query·key max-sim (pre-Chrono, at the prompt's last token) is not an address space for a corpus at this size; it
is anti-informative. Reading: the trained notebook keys encode what to retrieve WITHIN a context (successions, rules, recent
tokens), not topical relevance across documents; the query at "Answer:" seeks a continuation, not a source. The full key index is
not earned. The interface gap stands at the RETRIEVER: a trained / topical retriever (dense or HyDE-style) is the next lever, as
plumbing, not a claim.

## sci1_chat: the chat ramp, the opening of the long run (DRAFT pre-registration, 13:59 IST, 2026-11-05; final with measured rows after the Nov 6 decay)

Frame: `v16/PLAN_LONGRUN.md` (rules R1–R7). This is mix-schedule step 2; the run does not end on a decay.
- **Branch:** `--continue_from` the main line's pre-decay `stable_final.pt` (global 160,218, ~10.5B tokens), constant LR 8e-4
  (`wsd_stable`), 500-step re-warm; the Nov 6 decay stays the base release (paper checkpoint). Same geometry, SEQ 4096, 16 × 4096.
- **Ramp (R2):** one in-sampler linear ramp of the per-source targets W0 → W1 over the first **2B tokens**, then hold W1 (new env
  RAMP_WEIGHTS / RAMP_TOKENS: target(t) = (1 − a) W0 + a W1, a = min(1, consumed / RAMP_TOKENS); code + parity test before launch).
  W0 = the cont4 / final main mix; W1 = chat block ~6 % → **~22 %**, of which a named **instruction-following share ~6 %** (strict
  formats: yes / no, one word, a number only, multi-step instructions, format constraints), the gain taken from web (dclm /
  fineweb) first; science shares and the rule-doc block (b2s ~1.5 %, proto4k ~0.65 %, recall4k ~0.3 %) **fixed (R1)**; `worked`
  kept or dropped per the cont4 read.
- **Candidate sources (licence check + R5 13-gram audit on the first 20k docs BEFORE the first step, drop counts in the row):**
  SmolTalk subsets (everyday-conversations, smol-constraints for instruction following, smol-rewrite / smol-summarize), a
  Tulu-3-style SFT mixture (permissive subsets only), and an IFEval-like constraint set; decontaminated against IFEval and our probes
  as well as the current index.
- **Measured calibration rows** at W0, the midpoint and W1 (≥ 30M tokens each); the trainer's [mix] line (with per-source epochs,
  R7) is the run's own measured ramp. **R7:** no source beyond 4 epochs over the planned horizon; tools / worked / chat sources
  sized accordingly.
- **Target probe:** Library's simple-chat probe with an instruction-following cell (incl. "answer only YES or NO", one word, number
  only, multi-step) + the user's prompts; it must RISE.
- **Guards (R3, vs the decay-end readings):** s8 D ≥ −0.05; told twin 2k and held-out wording ≥ −0.10; far needles 16k and clk2 8k
  ≥ −0.05; HS / ARC-e / PIQA ≥ −0.01; per-source science holdout ≤ +0.03. Judged at ramp end + 1B (the lag rule); a guard falling at
  two consecutive milestones freezes the ramp at its current share.
- **Release (R4):** the chat variant is a 0.5B decay branch from ramp end + 1B; base and chat both tagged by token count.
- **Tool data (Library's GSM8K tool arm on c3b, 15:10 IST 11-05):** with `calculate(expression)` offered, the model calls a tool on
  97 % of items in the trained `<tool_call>` format but invents glaive-style names (calculate_number 47, calculate_percentage 23,
  calculate_distance 13 vs calculate 24); 19 / 206 calls well-formed, 3 compute a number; arguments are words. It has the FORMAT,
  not the schema-reading or argument grounding: the glaive part of `tools` is a small set of recurring function names (memorised
  templates; R7). For the ramp: (i) cap `tools`' epochs (R7) and down-weight glaive; (ii) add a schema-grounded tool source where
  the offered function's exact name and typed arguments must be copied from the system prompt and filled from the question (e.g.
  ToolACE's varied schemas, plus a synthetic "offered-schema" set with randomised function names and arithmetic / lookup
  arguments taken from the question); (iii) a probe cell: call accuracy = right name + parseable args + correct result.

## SCI-1 cont3 milestone 9 (CHECK on the 4090, 13:16 IST, 2026-11-05; run `449e320`, step 137,331 = 9.0B tokens; 2.0B of cont3, protoc out)

**cont3 verdict (pre-registered, revised before its data):** s8 D **0.770** (M8 0.589, M7 0.588) is below the revised protoc bar
(0.84 = the cont2 trend + 0.10), so the rise is **not attributed to dropping protoc**; it is the lagged b2s recovery continuing (b2s
~1.4–1.5 % realized in both cont2 and cont3). **Told side (Library, 13:23 IST, 2026-11-05; `logs/v14_a28_rep100{,_8k}_c3b_e2a87db_20261105_1316.*`, 100 docs, twin read):** G1 2k **0.83 ±
0.04** (M8 0.54 ± 0.05: +0.29), 8k **0.22 ± 0.04** (M8 0.04: the best told reach so far); plain read 2k 0.22, A9 0.08, noprot 0.03.
**Competition read (pre-registered table):** told did NOT fall; it ROSE with induced (+0.29 with s8 +0.18). "Both rise" was not one
of the four pre-registered outcomes; it rules out competition at this interval, and across the five intervals the seesaw is not
consistent (opposite moves in 2, joint rise in 1, one-sided in 2). **The competition hypothesis is not supported**; the simpler
reading (Library) is that both ride the same rule-doc rehearsal (b2s ~1.5 %, proto4k ~0.65 %) with lags. Dropping protoc did not
hurt the told rule.

| readout | M7 (7.0B) | M8 (8.0B) | **M9 (9.0B)** |
|---|---|---|---|
| s8 D | 0.588 | 0.589 | **0.770** |
| s8 shift 2k / 4k; affine 2k / 4k; @0 | 0.77 / 0.37; 0.78 / 0.43; 0.94 / 1.00 | 0.82 / 0.37; 0.71 / 0.46; 1.00 / 0.99 | **0.90 / 0.53; 0.92 / 0.74**; 1.00 / 1.00 |
| skill pack 2k / 4k / 8k | 0.88 / 0.77 / 0.35 | 0.90 / 0.70 / 0.31 | 0.84 / 0.69 / 0.30 |
| clk2 8k / 16k / 32k | 0.68 / 0.22 / 0.02 | 0.57 / 0.14 / 0.02 | 0.61 / 0.12 / 0.04 |
| far needles 4k + 8k; 16k | 0.88; 0.70 | 0.99; 0.88 | 0.98; 0.87 |
| HellaSwag / ARC-e / PIQA | 0.327 / 0.534 / 0.613 | 0.331 / 0.535 / 0.619 | 0.333 / 0.519 / 0.621 |
| think_eval (cont4 leak baseline) | – | – | ON / OFF acc 0.00 / 0.00, `<think>` leak 0.00 |

- s8 D is now near its M4 peak (0.782): the skill recovered fully about 2B tokens after b2s was restored, the lag again.
- The cont4 (`worked`) leak rule starts from 0.00; GSM8K accuracy is 0.00 in both modes at this size (no "The answer is" line).

## lad: the mechanism ablation ladder at the ctl recipe (PRE-REGISTRATION, 08:44 IST, 2026-11-05; Lead decision; on the 6000 after SCI-1's decay)

**Question:** which parts does TRAINING need for a rule to be carried across filler (the knockouts showed what the trained model
uses)? Each rung is ctl's PAM M2 (102.88M, the ctl recipe: 1.0B tokens of the cont2 mix, SEQ 4096, seed 43, 0.9B wsd_stable + 0.1B
decay) with ONE part removed and the MLP width re-matched to ±1 % params:
- `lad_all`: all on (the same body as ctl PAM, on the 6000: a hardware / run replicate of 0.523);
- `lad_nodelta`: DELTA = 0 (additive notebook writes; SHARE_READS needs delta, so the readers read the additive write);
- `lad_nochrono`: CHRONO = 0 (plain RoPE);
- `lad_noecho`: ECHO = 0;
- `lad_noread`: SHARE_READS = 0 (the 12 notebook-less blocks are pure MLP).
**Order on the 6000 (Lead): `lad_all` FIRST**, and it must reproduce ctl PAM (s8 D 0.523 ± 0.05, holdout 3.427 ± 0.02) on that
box before any minus-one rung is read (a box confound bit us once: g1c). If it does not, the ladder pauses and the gap is the finding.
**Readouts per rung (ctl's set + the told side):** s8 D (@0 / 2k / 4k / 8k), far needles 8k / 16k, clk2 8k / 16k, holdout, HS /
ARC-e / PIQA, and Library's told-twin cells (A26 G1 twin read at 2k and 8k, trained and held-out wording, 100 docs).
**Rules:** a part is NEEDED for carrying if its rung's s8 D is ≥ 0.15 below `lad_all` with @0 within 0.10 (induction intact); a
rung whose @0 also falls ≥ 0.15 is "needed for inducing / expressing", not carrying. Same rule on the told 2k / 8k cells. Language:
report holdout Δ vs `lad_all`. Prediction from kn: no-delta carries least; no-chrono costs ~0.1–0.15; no-echo / no-readers either
re-learn a read path (small loss) or lose the rule at @0 too.
**Note:** the SHARE_READS assertion needs delta + chrono + pam_layers; `lad_nodelta` / `lad_nochrono` therefore need the readers'
assertion relaxed or the rung run without readers; decided at the launch, stated in the rung's row.

## kn: eval-time knockouts on the trained ctl PAM: which mechanism carries the rule across filler? (PRE-REGISTRATION, 08:34 IST, 2026-11-05; Lead GO)

**Question:** in the trained ctl PAM M2 (102.88M, 1.0B tokens, s8 D 0.523), which part is USED to carry an induced rule across
2k / 4k of filler? **This measures what the trained model uses, not what training needs** (a model trained without a part may
compensate); the ablation ladder (1B per rung, on the 6000) answers the second.
**Probe:** the s8 spread probe exactly as the ctl readout (b2 suite, shift_wide_new + affine_map_new, 8 shots, 400 trials, seed
7000, spread 0 / 2048 / 4096, the holdout filler cache), checkpoint `ctl_pam_decay_caa4da9/latest.pt`. Runner
`v13_sempty/tmp/s8_knockouts.py` (the probe's own builders + a filler mask; examples, separators and the query untouched).
**Arms:** `on` (control: must reproduce s8 D 0.523, else the run is void); `erase0` (notebook erase gate b_e = 0 on filler tokens:
no erase-at-key there); `globfade` (on filler tokens the key-selective erase is replaced by a global fade of equal erased energy on
an isotropic notebook: b_e = 0, g ← g (1 − b_e / K)); `clock1` (Chrono warp g = 1 on filler tokens: the clock runs at plain RoPE
speed through the filler); `noecho` (E1 echo read zeroed everywhere); `noread` (M2 readers zeroed everywhere).
**Rule:** a part "carries the rule" if its knockout drops s8 D by **≥ 0.15** vs `on`. Read also @0 (whole-model switches can hurt
induction itself) and per length. Told-twin cell (Library's A26 G1 2k) under the same knockouts if it costs < 10 min more (not in
this runner yet). Prior (Mind's step back): erase-at-key most likely, then the clock; echo and readers unlikely.

**kn RESULT (08:43 IST, 2026-11-05; `logs/ctl/v13_sempty_kn_s8_ctl_pam_44342fd.{json,log}`, 400 trials per cell).** The control reproduces the
ctl readout exactly (s8 D 0.523), so the run is valid.

| arm | s8 D | Δ vs on | @0 shift / affine | shift 2k / 4k | affine 2k / 4k | verdict (Δ ≥ 0.15) |
|---|---|---|---|---|---|---|
| on | 0.523 | – | 0.765 / 0.733 | 0.630 / 0.393 | 0.600 / 0.470 | control reproduced |
| erase0 (no erase-at-key on filler) | 0.197 | **−0.326** | 0.765 / 0.733 | 0.193 / 0.117 | 0.343 / 0.135 | **carries** |
| globfade (global fade, equal energy) | 0.136 | **−0.387** | 0.765 / 0.733 | 0.147 / 0.065 | 0.233 / 0.098 | **carries** |
| clock1 (plain-RoPE clock on filler) | 0.381 | −0.142 | 0.765 / 0.733 | 0.435 / 0.258 | 0.485 / 0.345 | below the bar (contributes) |
| noecho (echo read zeroed) | 0.086 | −0.437 | **0.062 / 0.160** | 0.060 / 0.060 | 0.107 / 0.115 | needed to EXPRESS the rule |
| noread (M2 readers zeroed) | 0.059 | −0.464 | **0.003 / 0.375** | 0.025 / 0.033 | 0.113 / 0.068 | needed to EXPRESS the rule |

What it says (what the trained model uses, not what training needs):
- **The delta erase-at-key is the carrier.** With the examples adjacent nothing changes; through filler, removing the erase loses
  0.33, and replacing it by a global fade of the SAME erased energy loses more (0.39). Forgetting LOCALLY, along each filler token's
  own key, is what keeps the rule; a Mamba-like global fade is worse than not erasing at all.
- **The clock helps but is not the carrier** (−0.14, just under the bar): the learned warp keeps the rule's phase usable through
  filler; plain RoPE speed costs about a quarter of the carried rule.
- **Echo and the readers are on the READ path, not the storage path:** zeroing either collapses even the adjacent case (@0), so the
  trained M2 body expresses the induced rule through them (the four notebooks sit in blocks 3 / 7 / 11 / 15; the answer reaches the
  head through the readers and echo). Their carry role cannot be separated by an eval-time knockout; the ladder (trained without
  them) is the test.
- For the ladder: rungs "no delta" and "no chrono" are the informative ones for carrying; "no echo" / "no readers" test whether
  another read path can be learned.

## SCI-1 cont3 milestone 8 (CHECK on the 4090, 05:48 IST, 2026-11-05; run `449e320`, step 122,072 = 8.0B tokens; 1.0B of cont3, protoc out)

First of the two cont3 readouts (judged with 137,331; the lag rule).

| readout | M6 (cont2 +1B) | M7 (cont2 +2B) | **M8 (cont3 +1B)** |
|---|---|---|---|
| s8 D | 0.436 | 0.588 | **0.589** |
| s8 shift_wide 2k / 4k; affine 2k / 4k | 0.58 / 0.19; 0.65 / 0.33 | 0.77 / 0.37; 0.78 / 0.43 | 0.82 / 0.37; 0.71 / 0.46 |
| skill pack 2k / 4k / 8k + contrast | 0.86 / 0.68 / 0.18 | 0.88 / 0.77 / 0.35 | 0.90 / 0.70 / 0.31 |
| clk2 8k / 16k / 32k | 0.55 / 0.08 / 0.04 | 0.68 / 0.22 / 0.02 | 0.57 / 0.14 / 0.02 |
| t40 far needles 4k + 8k; at 16k (position 0) | 0.81; 0.47 | 0.88; 0.70 | **0.99; 0.88** |
| HellaSwag (norm) / ARC-e / PIQA (norm) | 0.320 / 0.514 / 0.617 | 0.327 / 0.534 / 0.613 | **0.331 / 0.535 / 0.619** |
| holdout val (blend) | 3.139 | 3.117 | **3.093** (122k, best) |

- s8 D is flat (0.589), far below the revised protoc bar (0.84): no sign yet that protoc was the suppressor. Per the lag rule
  the verdict waits for 137,331. Told side (Library's twin read) pending at this step.
- Far-needle recall past the trained SEQ is the best of the run (16k 0.88, 4k + 8k 0.99); clk2 with the read operator dipped
  (8k 0.57). Language and benchmarks keep improving (holdout 3.093, HS 0.331, PIQA 0.619).

## ctl RESULT: matched control at the SCI-1 recipe (B1–B5, 04:56 IST, 2026-11-05)

All three ~103M bodies, **1.0B tokens each** of the identical cont2 mix (SEQ 4096, seed 43, decontam, same code caa4da9, same
venv), 0.9B wsd_stable + 0.1B decay. Logs `logs/v13_sempty_ctl_{pam,tpp,mamba2}_{main,decay}_caa4da9_*.log`, readouts `logs/ctl/`.

| body | params | tokens | s8 D (B1) | s8 @0 / 2k / 4k / 8k | far needle 2k / 4k / 8k / 16k (B2: 8k, 16k) | clk2 8k / 16k: dilation 1; 8 + contrast | holdout (B3) | HS / ARC-e / PIQA (B4) |
|---|---|---|---|---|---|---|---|---|
| PAM M2 | 102.88M | 1.0B | **0.523** | 0.75 / 0.61 / 0.43 / 0.28 | 0.63 / 0.63 / **0.53 / 0.41** | 0.11 / 0.04; **0.59 / 0.15** | 3.427 | 0.282 / 0.455 / 0.588 |
| Transformer++ | 101.92M | 1.0B | 0.074 | 0.36 / 0.10 / 0.05 / 0.05 | **0.92** / 0.12 / 0.17 / 0.13 | 0.08 / 0.07; 0.09 / 0.06 | 3.440 | 0.281 / 0.457 / 0.590 |
| Mamba-2 | 102.61M | 1.0B | 0.149 | 0.68 / 0.18 / 0.12 / 0.08 | 0.66 / 0.27 / 0.18 / 0.15 | 0.03 / 0.03; 0.05 / 0.05 | **3.378** | 0.284 / 0.459 / 0.588 |

Verdicts (pre-registered):
- **B1 rules (s8 D): MET.** PAM 0.523 vs the best baseline 0.149 (Mamba-2): +0.374 (bar +0.20). At every distance PAM leads;
  with the examples adjacent (@0) the gap is small (0.75 vs 0.68 / 0.36), so the difference is CARRYING the rule across filler.
- **B2 far needles past the trained SEQ (8k / 16k): MET.** PAM 0.53 / 0.41 vs the best baseline 0.18 / 0.15: +0.35 / +0.26 (bar
  +0.10). Inside its window (2k) the transformer is better (0.92 vs 0.63), as expected.
- **B3 language: NOT MET.** Holdout PAM 3.427 vs Mamba-2 3.378 (+0.049 nats; bar ≤ +0.03); PAM beats T++ (3.440). Per source,
  Mamba-2 leads on every web / science source by 0.03–0.06; PAM leads T++ on web and trails it on arxiv / pubmed / pysci.
- **B4 known benchmarks:** level (all within 0.004).
- **B5 prose rule held in state (Library's A26 G1 at 2k, protocol-twin read; 05:18 IST, 2026-11-05): MET.** Right (value + unit), 98 docs per arm:
  PAM **0.53 ± 0.05**, Mamba-2 0.20 ± 0.04, T++ 0.12 ± 0.03 → +0.33 over the best baseline (bar +0.20). Under the package's A9 read,
  the plain read and the no-protocol control, every body is 0.00, so the told rule is visible only against the no-protocol twin,
  as A28 found on SCI-1. Log `logs/ctl/v14_a28_ctl_pam_vs_tpp_mamba2_caa4da9.jsonl`.
- clk2 with the O(1) read operator (dilation 8 + contrast): PAM 0.59 at 8k; the operator is PAM-only by construction (no claim
  against the baselines on clk2 itself; without the operator PAM is 0.11).
**Reading:** on the product recipe, at equal params, tokens and data (rule docs included), the thesis clauses hold against
both baselines (rules INDUCED and rules TOLD, both carried in the state; recall beyond the trained length), while Mamba-2 is the better pure language model at
this size and budget (consistent with the 3B tier, where Mamba-2's holdout also led). Prior work: Transformer++ / Mamba-2 (Gu &
Dao 2024) as baselines; our matched 600M / 1.5B tiers. The SCI-1 vs Pythia gaps were data; THIS table is the architecture evidence
at the SCI recipe.

## SCI-1 cont2 milestone 7 (CHECK on the 4090, 21:39 IST, 2026-11-04; run `e61cd92`, step 106,813 = 7.0B tokens; 2.0B of the cont2 mix)

**The skill came back with a ~1B-token lag; the M6 "erased" reading was premature.**

| readout | M4 (4.0B, old mix) | M5 (cont1) | M6 (cont2, +1B) | **M7 (cont2, +2B)** |
|---|---|---|---|---|
| s8 D | 0.782 | 0.435 | 0.436 | **0.588** |
| s8 shift_wide 2k / 4k | 0.935 / 0.595 | 0.540 / 0.285 | 0.580 / 0.188 | 0.765 / 0.370 |
| s8 affine_map 2k / 4k | 0.885 / 0.713 | 0.583 / 0.333 | 0.647 / 0.330 | 0.780 / 0.435 |
| skill pack 2k / 4k / 8k + contrast | 0.84 / 0.70 / 0.21 | 0.84 / 0.68 / 0.26 | 0.86 / 0.68 / 0.18 | **0.88 / 0.77 / 0.35** |
| clk2 8k / 16k / 32k | 0.62 / – / – | 0.67 / – / – | 0.55 / 0.08 / 0.04 | **0.68 / 0.22** / 0.02 |
| t40 far needles 4k + 8k; at 16k (position 0) | 0.95; – | 0.89; – | 0.81; 0.47 | 0.88; **0.70** |
| HellaSwag (norm) / ARC-e / PIQA (norm) | 0.311 / 0.508 / 0.618 | 0.321 / 0.500 / 0.612 | 0.320 / 0.514 / 0.617 | **0.327 / 0.534** / 0.613 |
| holdout val (blend) | 3.218 | 3.177 | 3.139 | **3.117** (104k, best) |

- The restored b2s (~1.4 %) did work, but only in the second billion tokens: s8 D +0.152 from M6 to M7. Relearning ran at
  about the first-acquisition rate (M1 → M3: +0.39 in 2B), so this is suppression with no savings speed-up, not erasure.
- Reach recovered with it and is the best yet: skill pack 8k 0.35, clk2 16k 0.22, t40 at 16k 0.70. ARC-e 0.534 and HS 0.327 are
  also bests. The told-rule side (Library's twin read) is read on this checkpoint next.
- **Consequence for cont3 (honest):** cont3 started from a point where s8 is RISING under the cont2 mix with protoc IN. Its
  pre-registered bar (s8 D ≥ 0.60 by 122,072 → "protoc was the suppressor") is now almost met by the trend alone and can no longer
  separate protoc from the lagged b2s effect. Revised read (written before the 122,072 data): an s8 rise at 122,072 is NOT
  attributed to protoc; protoc is called a suppressor only if s8 rises clearly ABOVE the cont2 trend (≥ 0.588 + 0.15 + 0.10 =
  **0.84**). The told side keeps its pre-registered read (told G1 falls ≥ 0.10 → protoc was helping it).
- Generations: the protocol prompt answers "1.912 g" (wrong, but a changed value); the tool prompt reads "78.4 C" right.
- **Told vs induced side by side (Library's twin read, 100 docs; 22:37 IST):**

  | milestone (mix) | induced: s8 D | told: twin-read G1 2k / 8k | compact G1 2k |
  |---|---|---|---|
  | M4 (old mix) | 0.782 | 0.26 / – | 0.00 |
  | M5 (cont1) | 0.435 | 0.65 / ≤ 0.05 | 0.45 |
  | M6 (cont2) | 0.436 | 0.81 / 0.20 (20 docs) | 0.00 |
  | M7 (cont2) | 0.588 | **0.49 / 0.04** | 0.10 |

  On the SAME mix (M6 → M7) told fell −0.32 while induced rose +0.15; M4 → M5 was the mirror (told +0.39, induced −0.35); M5 → M6
  moved told only. Two of three intervals are a seesaw. This strengthens the competition HYPOTHESIS (still 4 points, one run).
  It also means every skill here swings ±0.3 between milestones even at a fixed mix, so single-milestone reads of any of them are
  weak; cont3 is judged on 122,072 AND 137,331 (Library + Lead). Guard baseline for cont3: told 2k 0.49, 8k 0.04.
## SCI-1 cont2 milestone 6 (CHECK on the 4090, 14:11 IST, 2026-11-04; run `e61cd92`, step 91,554 = 6.0B tokens; 1.0B of the cont2 mix)

**The SAVINGS readout: NO recovery.** s8 D 0.435 (M5) → **0.436** after 1B tokens with b2s restored to ~1.4 %. By the
pre-registered rule (< 0.50) this reads "erased", but the result also weakens the explanation it was built on.

| readout | M4 (4.0B, old mix) | M5 (5.0B, cont1) | M6 (6.0B, cont2) |
|---|---|---|---|
| s8 D | 0.782 | 0.435 | **0.436** |
| s8 shift_wide @0 / 2k / 4k | 1.00 / 0.935 / 0.595 | 0.955 / 0.540 / 0.285 | 1.00 / 0.580 / 0.188 |
| s8 affine_map @0 / 2k / 4k | 0.988 / 0.885 / 0.713 | 0.975 / 0.583 / 0.333 | 0.988 / 0.647 / 0.330 |
| skill pack 2k / 4k / 8k + contrast | 0.84 / 0.70 / 0.21 | 0.84 / 0.68 / 0.26 | 0.86 / 0.68 / **0.18** |
| clk2 8k (dilation 8 + contrast) | 0.62 | 0.67 | **0.55** |
| clk2 16k / 32k (new) | – | – | 0.08 / 0.04 (20-way chance 0.05; dilation 1: 0.06 / 0.04) |
| t40 far needles 4k + 8k (position 0) | 0.95 | 0.89 | **0.81** |
| t40 at 16k, position 0 / 0.5 / 1 (new) | – | – | 0.47 / 0.78 / 1.00 |
| HellaSwag (norm) / ARC-e / PIQA (norm) | 0.311 / 0.508 / 0.618 | 0.321 / 0.500 / 0.612 | 0.320 / 0.514 / 0.617 |
| holdout val (blend) | 3.218 (60k) | 3.177 (76k) | **3.139** (92k, best) |

What it says:
- The b2s cut ALONE is not the cause: putting b2s back (0.5 → ~1.4 % realized) recovered nothing in 1B tokens. Two other
  changes made at the cont1 switch are still in place: **think dropped** (2.4 % of tokens; long chain-of-thought documents, the
  user's decision) and **protoc added** (median 94-token compact rule docs at 8.05× loss weight per token: ~1.2 % of the loss,
  all of it a rule used within ≤ 20 sentences). Mechanism to test: protoc trains the same rule pathway as b2s but at SHORT range
  and at high weight, so the pathway specialises to short range; or the think docs were the main natural trainer of carrying
  information across long spans. A third reading (true erasure, slow relearning at a late, constant-LR stage) cannot be excluded:
  the first acquisition (M1 → M3) happened early, when everything is learned fast.
- Reach is now slipping too (clk2 8k 0.67 → 0.55, skill pack 8k 0.26 → 0.18, far needles 0.89 → 0.81) while the holdout keeps
  improving: the "holds beyond the trained length" clause is degrading on this mix. Past 8k nothing holds at this size (clk2
  16k / 32k at chance; t40 at 16k 0.47 at position 0).
- The `--continue_from` restarts are not the cause: the optimizer state is kept; only the LR schedule restarts (500-step warmup).
- **Library's A26 + twin read at the same step (15:01 IST, 2026-11-04; EXPERIMENTS_V14):** the prose protocol HELD IN STATE kept rising while s8
  fell: twin-read G1 at 2k M4 0.26 → c1a 0.65 → **c2a 0.81** (100 docs), plain read 0.33 (c1a 0.01), and reach starting (8k / 32k
  0.20 / 0.20). The compact format oscillates (M3 0.40 → M4 0.00 → c1a 0.45 → c2a 0.00) with protoc in. So the state is not
  losing "rule carrying" in general: since the cont1 switch it holds TOLD rules (prose instructions) better and INDUCES rules from
  spread examples worse. Reading: the two uses compete for the same notebooks, and the mix decides which wins (cont1 / cont2 moved
  weight from induction docs, b2s, to instruction docs, proto4k + protoc). cont3 (protoc out, proto4k 0.65 %, b2s 1.49 %) tests one
  side of that; the guard is that twin-read G1 at 2k / 8k must not fall.
- Generations: the protocol prompt copies "12.347 g" again; the tool prompt reads "78.4 C" right and runs past `<|im_end|>`.

**cont3 PRE-REGISTRATION (14:13 IST, 2026-11-04; Lead decision 14:30): the one-cause test of protoc.** At cont2 milestone 7 (global 106,813,
~20:40 IST; two milestones per mix holds), continue with **protoc dropped**, b2s ~1.4 %, proto4k 0.8 %, recall4k 0.34 %, no think
(no think-like source at this switch, so the read stays one-cause; the user is asked separately about a plain-text worked-solution
source for the switch after). Launched only from a measured calibration row: **calib7** (31.5M tokens, exactly the launched weights) realized b2s **1.49 %**,
proto4k 0.65 % (target 0.8; kept, no unmeasured re-weighting), recall4k 0.32 % → synthetic loss weight **3.87 %**.
`sci1_cont3_env.sh`, `NEWSEG=cont3 sci1_switch.sh step_106813 <hash> checkpoints_v13_sempty/sci1_cont2_e61cd92`.
- **s8 D ≥ 0.60 by 122,072** (1B tokens of cont3) → **protoc was the suppressor** (short-range rule docs at high weight specialise
  the rule pathway); then a trade with Library's A26 compact gains is made explicitly.
- **s8 D < 0.60 by 122,072** → protoc is not the cause: (b) the think drop or (c) true erasure; the plain-text worked-solution
  source (if the user allows it) is the next one-cause test; the synaptic-tag screen tests (c) on the 4090 after ctl.
- Also read at 122,072: reach (clk2 8k ≥ 0.62 = back to M4, skill pack 8k, far needles), A26 compact (Library), holdout ≤ projection
  + 0.02. Clean capacity check in parallel: the ctl PAM s8 at 1B on the cont2 mix (from scratch, protoc in).
- **Competition vs protoc-specific (Lead, added before the switch; the competition reading is a HYPOTHESIS, one switch moved both):**
  at 122,072 read induced (s8 D) and told (Library's twin-read G1 at 2k and 8k, 100 docs) side by side against 106,813:
  - s8 rises (≥ +0.10) AND told G1 2k falls (≥ −0.10) → **competition** supported; the per-use notebook / synaptic-tag experiment
    becomes the lead architecture experiment after ctl.
  - s8 rises AND told holds (within ±0.10) → **protoc specifically** suppressed induction; no competition claim.
  - s8 flat AND told holds → neither; cause (b) / (c) remain (cont4 tests (b)).
  - s8 flat AND told falls → protoc helped told, did not hurt induction; keep protoc out only if told recovers.

**cont3 launched (21:14 IST 2026-11-04; recorded 21:30 IST, 2026-11-04).** The switcher (sci1sw, 449e320) checked milestone 106,813 (7,000,096,768
tokens), stopped cont2, deleted cont2's superseded latest.pt, checked out 449e320 (no untracked blockers) and started cont3: tmux
`sci1`, dir `checkpoints_v13_sempty/sci1_cont3_449e320`, log `logs/v13_sempty_sci1_cont3_449e320_*.log`; `[continue]` from
106,813 with a fresh budget of 3,499,950,080 tokens to global 160,218, wsd_stable, the calib7 weights (protoc out). Next
milestones 122,072 (~Nov 5 04:00 IST: the cont3 readout + the cont4 switch), 137,331, 152,590; then the decay.

**cont4 PRE-REGISTRATION: the test of cause (b) (14:27 IST, 2026-11-04; the user said YES via Lead, "as a test").** At the switch after cont3,
**moved to global 137,331** (Lead, after the M7 lag finding: cont3 gets its 2 milestones; one cause at a time), add source **`worked`**: the same MetaMathQA + GSM8K train
items as the dropped `think` (same 1,200-char cap), each rendered as a PLAIN worked solution + "The answer is X." (no `<think>`,
no system switch, the direct answer kept). 392,830 docs, median 215 tokens (max 1,030); 0.25 % of a 2,000-doc sample hit the
decontam index (dropped by the blend; GSM8K test is in the index). **~2 % of tokens**, launched from a measured calibration row.
- **Keep it only if** s8 D rises **≥ +0.10** over its value at 122,072 within the next 2 milestones (137,331 / 152,590) **AND** the
  thinking-style leak stays **< 5 %** (think_eval's leak check, re-added to the CHECK for these milestones). Otherwise drop it.
- Prior (honest): these docs are short (median 215 tokens), so they are a weak candidate for a long-span trainer; the old think
  source was half answer-only. A null here would point to (c), true erasure, and to the synaptic-tag screen. Caveat (Lead): a null from
  215-token docs does not rule out long-span reasoning text in general.
## ctl: matched control at the SCI-1 recipe (PRE-REGISTRATION, 08:41 IST, 2026-11-04; Lead + user GO 08:45 IST; on the 4090)
- **Moved and downgraded (Lead + Mind, after M7):** at 137,331 cont4 gets only ~1.5B tokens before the main budget ends at
  160,218, i.e. ONE milestone (152,590) inside the ~1B-token lag window. The SCI-1 worked test is therefore **indicative only**
  (underpowered); the clean test of worked text is planned on the matched ~103M bodies after ctl, if it matters.
- **Measured row (calib8, 31.5M tokens, exactly the launch weights; 07:16 IST 11-05):** worked **2.16 %**, b2s 1.45 %, proto4k
  0.61 %, recall4k 0.30 % (all others at cont3's realized shares, the 2 points taken from web) → synthetic loss weight 3.71 %.
  `sci1_cont4_env.sh`; switch `NEWSEG=cont4 sci1_switch.sh step_137331 <hash> checkpoints_v13_sempty/sci1_cont3_449e320`.
  think_eval is back in the CHECK from 137,331 (the cont4 baseline for the leak rule).

**Question:** does the architecture gap (rules in state, reach past the trained length) survive when all three bodies get the SCI-1
product recipe, rule docs included? The SCI-1 vs Pythia gaps are mostly data; the earlier matched tiers (600M × 3 seeds, 1.5B)
used the tier mix, not this one.

**Arms** (params from the smoke, `v13_sempty/tmp/ctl_smoke.sh`): PAM M2 (4 notebooks every 4th block + shared reads + echo + chrono +
out_gate + delta, MLP 2128) **102.88M**; Transformer++ (the tier's) ~101.9M; Mamba-2 17 × 768 ~102.6M. The ~103M tier geometries
rather than a new ~125M size: known-good, already param-matched, and comparable to the 600M / 1.5B tiers.

**Recipe (identical for all three):** the cont2 mix (sampler weights of `sci1_cont2_env.sh`; realized shares printed at every val
by the trainer's new `[mix]` line), GPT-2 + chat tokenizer, SEQ 4096, BATCH 16 × 4096 = 65,536 tokens per step (4 micro-batches of 4), LR 1.2e-3 (the
tier's for this size), warmup 500, `wsd_stable` 0.9B tokens (13,733 steps) then a linear decay 0.1B (1,526 steps) = **1.0B tokens
each**, seed 43 (same stream order), decontam index sha 08766210…, answer weight 100, ANSWER_IN_WINDOW, RESET_AT_EOS, per-source
holdout val. From scratch (no stage A). Data note: on this box every source streams from the Hub (HfFileSystem), fineweb included
(the 6000 reads a local copy of the same sample-10BT); calib6 read all 15 sources here. The same for all three arms.
Runner `v13_sempty/tmp/ctl_chain.sh` (sequential; yields the card to every CHECK / humaneval / Library eval and on RAM < 12 GB,
resumes from latest.pt, saves every 200 steps).

**Bars at the decay end (claims only inside this block; params + tokens on every row):**
- **B1 rules (s8 D, mean of 2k + 4k):** PAM ≥ max(T++, Mamba-2) + 0.20. Also reported @0 and @8k (8k = past
  the trained SEQ 4096).
- **B2 reach past SEQ:** t40 far needles at 8k and 16k: PAM ≥ max(baselines) + 0.10; clk2 at 8k / 16k reported with dilation 1
  (no tool) and 8 + contrast (the O(1) read operator; PAM only by construction, stated as such).
- **B3 language:** holdout (blend + per source) of PAM within +0.03 nats of the best baseline (bar-2 tolerance), reported.
- **B4 known benchmarks:** HellaSwag / ARC-e / PIQA reported, no bar (kb1: level at 600M).
- **B5 prose rule held in state (Lead, added before launch):** Library's A26 G1 cell at 2k with the protocol-twin read (A28),
  100 docs, on all three: PAM ≥ max(baselines) + 0.20. Reference: SCI-1 c1a 0.65 ± 0.05, M4 0.26 (old read 0.00); t15tpp
  (unmatched) 0.00. Script `v14/a28_twin.py --ckpt <PAM> --win <T++>,<Mamba-2> --wording train --lengths 2048 --docs 100`
  (same docs and fetched row for all three). Run at the chain's end.

**Launched (10:14 IST 2026-11-04; recorded 10:49 IST, 2026-11-04).** Worktree `../qllm_ctl` @ caa4da9 (fixed hash: master commits cannot break the
run's log names or trip the drift guard), venv qllm311 (Python 3.11, torch 2.8, mamba_ssm) for ALL three arms, tmux `mindctl`,
chain log `logs/v13_sempty_ctl_chain_caa4da9_20261104_0444.log` (in the worktree). Smoke (200 steps, 65,536 tokens / step): PAM
102,884,948 params, 16.1 GB peak (~40k tok/s on py3.9; 63.8k on py3.11 at the chain's step 100); T++ 101,916,864, 82.7k tok/s,
10.6 GB; Mamba-2 102,613,960, 113k tok/s, 7.0 GB. Stream startup 20–35 min per segment. ETA: PAM ~16:00 IST, T++ ~Nov 4 23:00,
Mamba-2 ~Nov 5 02:30 (plus yields to the 91,554 and 106,813 CHECKs), readouts + B5 by ~Nov 5 06:00 IST.
**Closest prior work:** Transformer++ and Mamba-2 (Gu & Dao 2024) as the standard baselines; our matched 600M / 1.5B tiers.
Difference: the same product recipe (science + chat + tools + rule docs), so the SCI-1 claims have an equal-data control.

## SCI-1 cont1 milestone 5 (CHECK on the 4090, 05:50 IST, 2026-11-04; run `59d098d`, step 76,295 = 5.0B tokens; 1.0B tokens of the cont1 mix)

**The s8 bar FAILS: rules from spread examples fell 0.782 → 0.435 in 1B tokens after b2s was cut 1.7 % → 0.5 %.**

| readout | M4 (4.0B, old mix) | M5 (5.0B, cont1 mix) |
|---|---|---|
| s8 D (scored) | 0.782 | **0.435** (bar ≥ 0.50) |
| s8 shift_wide @0 / 2k / 4k | 1.000 / 0.935 / 0.595 | 0.955 / **0.540 / 0.285** |
| s8 affine_map @0 / 2k / 4k | 0.988 / 0.885 / 0.713 | 0.975 / **0.583 / 0.333** |
| skill pack 2k / 4k / 8k + contrast | 0.84 / 0.70 / 0.21 | 0.84 / 0.68 / **0.26** |
| clk2 8k | 0.62 | **0.67** |
| t40 far needles | 0.95 | 0.89 |
| HellaSwag (norm) / ARC-e / PIQA (norm) | 0.311 / 0.508 / 0.618 | **0.321** / 0.500 / 0.612 |
| think ON / OFF; leak | 0.055 / 0.04; 0.465 | 0.0 / 0.0; trace 0.0 (source dropped, as intended) |
| holdout val (blend) | 3.218 (60k) | 3.177 (76k; best 3.174) |

- What fell is ONLY the rule carried across distance: adjacent examples (@0) still 0.96–0.98, while 2k / 4k lost ~0.35 each.
  Recall and one-read reach did not fall (skill pack flat, clk2 8k and skill pack 8k up). So the in-state rule-at-distance
  skill at this size is MAINTAINED by the rule-doc share, not acquired once: a 3.4x cut of b2s (realized ~1.7 % → 0.5 %; the
  draft said 1.2 → 0.7, the final calibration cut further to fit protoc's 8.05x weight, my error) lost half of it in 1B tokens.
- Confound: proto4k (×2) and protoc (new, "Report rule." compact demos) were added at the same switch; they are also rule docs in a
  nearby format and could interfere. Not separable from this run alone; A26 (Library) at the same step says whether protoc helped.
- Generations: the protocol prompt copies again ("12.347 g" greedy; M4 changed the value); the quant prompt answers with a long
  wrong derivation (M4 greedy 0.125). Single samples; the scored probes above are the evidence.
- Pre-registered: s8 D ≥ 0.50 read at 76,295 and 91,554. Action proposed to Lead (a cont2 mix from this milestone, below).

**Proposed cont2 (Lead's call; synthetic loss-weight share computed with the measured per-token weights):** b2s back to 1.2 %,
proto4k back to 0.4 %, keep protoc 0.15 % and recall4k 0.34 % → **4.17 %** (B). If A26 shows protoc did nothing, drop it and
restore b2s 1.7 % with proto4k 0.8 % → 4.48 % (C). Restoring everything (b2s 1.7 + protoc) is 4.92 % (A), over the ceiling.

**cont2 launched (06:08 IST, 2026-11-04; Lead decision B at 06:00).** From `sci1_cont1_59d098d/step_076295.pt` (5.0B tokens), code e61cd92, tmux
`sci1` on the 6000, log `logs/v13_sempty_sci1_cont2_e61cd92_20261104_0027.log`; `[continue]` header shows the new weights; 37.8k
tok/s at step 76,350. The untracked-file check was run by hand first (none). cont1's ~1,600 steps past M5 were discarded; its
`latest.pt` deleted (superseded). Calibration `calib6` (15.7M tokens, the launched weights): realized b2s **1.41 %** (target 1.2),
proto4k 0.45 %, protoc 0.14 %, recall4k 0.34 % → synthetic loss weight **4.52 %** (predicted 4.17; b2s overshoots). Kept as is:
the s8 recovery is the point, and the holdout tripwire (≤ projection + 0.02) guards the language cost. Milestones: 91,554
(~Nov 4 13:15 IST), 106,813 (~20:40 IST); bars within those two: s8 D ≥ 0.70, reach not below M5, holdout ≤ projection + 0.02.

**A26 at cont1 M5 (Library, 06:14 IST; logs `logs/v14_a26_sci1c1a{,_g1}_850ab5e_20261104_0529.*`; 20 docs per cell; reference M3 until
Library's M4 run lands):** G1 held-in-state 2k / 8k / 32k = 0.00 / 0.00 / 0.00 = NOPROT (bar G1 ≥ NOPROT + 0.05 **not met**);
G1 compact 2k 0.45 (M3 0.40; bar ≥ 0.50, just short), 8k 0.15 (M3 0.00); A26-wording compact 0.35 at 2k / **0.60 at 8k** (M3
0.15 / 0.10), held-out quantities 0.45 / 0.60; CONVERT compact 0.10 (first non-zero), ROUND 0. Reading: protoc moves the COMPACT
value copy and its reach; the prose protocol held in state is untouched. Supports B (protoc kept), pending the M4 comparison. **M4 comparison (06:50 IST, Library, EXPERIMENTS_V14 "A26 SCI-1 M4 vs cont1 step
76,295"):** compact value at 2k M3 0.40 → **M4 0.00** (unit said 0.80, value never right) → c1a 0.45; A26-wording compact 2k / 8k
M4 0.00 / 0.00 → 0.35 / 0.60; held-out quantities M4 0.00 → 0.45 / 0.60; CONVERT 0.00 → 0.10; prose held in state 0.00 at every
milestone. Verdict: protoc helped (likely cause, not isolated: cont1 also changed b2s and dropped think). Note the mirror of the s8
finding: on the old mix (proto ~0.4 %) the compact value skill went 0.40 → 0.00 between M3 and M4. Skills held in state at this
size come and go with their data share.

**Pre-registered: the SAVINGS readout at cont2 milestone 91,554 (08:30 IST, 2026-11-04; from the Mind step back).** The first acquisition of s8 D
took 2B tokens (M1 0.322 → M3 0.709). cont2 restores b2s (realized ~1.4 %) from 0.435 at 76,295. At 91,554 (1.0B tokens later):
- s8 D ≥ 0.70 → **suppressed, not erased** (relearning savings: faster than first acquisition). The lever is rehearsal schedule
  (dose / spacing); no protected path is needed.
- 0.50 ≤ s8 D < 0.70 → partial savings; read again at 106,813.
- s8 D < 0.50 → **erased**; a protected path (the use-dependent plasticity tag, SCRATCHPAD step back) earns a 125M screen.
Also read at 91,554: the reach curve past SEQ (clk2 + t40 at 16k / 32k, added to the CHECK) next to M5's 8k.
Closest prior work: savings in relearning (Ebbinghaus; McCloskey & Cohen on catastrophic interference; "relearning savings" in
continual-learning nets). This is a measurement, not a method claim.

## SCI-1 milestone 4 (CHECK on the 4090, 21:47 IST, 2026-11-03; run `59c3cf5`, step 61,036 = 4.0B tokens; the LAST old-mix point, think still in)

The baseline for cont1 (which starts from this checkpoint). Same CHECK; the full copy, own hard link (no interference).

| readout | M2 (2.0B) | M3 (3.0B) | M4 (4.0B) |
|---|---|---|---|
| s8 D (scored) | 0.538 | 0.709 | **0.782** |
| skill pack 2k / 4k / 8k + contrast, 20-way | 0.76 / 0.51 / 0.10 | 0.87 / 0.66 / 0.12 | 0.84 / **0.70 / 0.21** |
| clk2 8k (dilation + contrast) | 0.34 | 0.37 | **0.62** |
| t40 far needles | 0.91 | 0.95 | 0.95 |
| HellaSwag (norm) / ARC-e / PIQA (norm) | 0.293 / 0.479 / 0.594 | 0.308 / 0.487 / 0.609 | 0.311 / **0.508** / 0.618 |
| think ON / OFF acc; OFF leak | 0.02 / 0.03; 0.24 | 0.015 / 0.04; 0.165 | 0.055 / 0.04; **0.465** |
| holdout val (blend), nearest val step | 3.358 (30k) | 3.275 (46k) | 3.218 (60k) |

- Still climbing; the one-read reach to 8k jumped (clk2 0.37 → 0.62; skill pack 8k 0.12 → 0.21), the first real move past the
  trained SEQ. ARC-e passes 0.50.
- Think: the OFF leak rose to 0.465; dropping the source at cont1 was right (ON 0.055 ≈ OFF 0.04 still).
- Generations: the protocol prompt now changes the value instead of copying ("12.87 g, sample B = 8.89 g" greedy; sampled
  "8.91 g") but does not yet round to one decimal; the quant prompt answers **0.125** greedy (right; M3 said 1.25); the tool
  prompt reads "78.4 C … for Q7" and still runs past `<|im_end|>` into a new turn.
- Holdout vs projection is read with cont1 at 91,554 (this is the reference point: 3.218).

## SCI-1 reference block: Pythia-410M at matched token counts (REFERENCE, different data, no claim; 17:17 IST, 2026-11-03)

Lead + user GO 16:25 IST. Same harness as the SCI-1 CHECK: `scripts/mc_eval.py --hf` (fp32; parity with lm-eval 0.4.9.1 to 4
decimals on Pythia-160M, 59e9197) and the s8 spread probe (b2 suite, shift_wide_new + affine_map_new, 8 shots, 400 trials, seed
7000; 8-way, chance 0.125; the GPT-2 filler re-tokenized for the NeoX vocab). Runner `v13_sempty/tmp/sci1_pythia_ref.sh` @ 78c98e6,
log `logs/v13_sempty_sci1ref_pythia410m_78c98e6_20261103_1102.log`, results `logs/sci1ref/`. Pythia = 2,097,152 tokens per step.

| model | params | tokens | data | HellaSwag acc_norm | ARC-e acc | PIQA acc_norm | s8 @0 | @2k | @4k | @8k | s8 D (2k + 4k) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SCI-1 M1 | 386M | 1.0B | SCI-1 mix | 0.281 | 0.459 | 0.576 | 0.695 | 0.422 | 0.221 | – | 0.322 |
| SCI-1 M2 | 386M | 2.0B | SCI-1 mix | 0.293 | 0.479 | 0.594 | 0.979 | 0.671 | 0.405 | – | 0.538 |
| SCI-1 M3 | 386M | 3.0B | SCI-1 mix | **0.308** | **0.487** | **0.609** | 0.994 | 0.863 | 0.555 | 0.236 | **0.709** |
| Pythia-410M step1000 | 405M | 2.10B | the Pile | 0.258 | 0.294 | 0.535 | 0.085 | 0.101 | 0.073 | 0.080 | 0.087 |
| Pythia-410M step2000 | 405M | 4.19B | the Pile | 0.272 | 0.363 | 0.581 | 0.065 | 0.048 | 0.007 | 0.075 | 0.028 |
| Pythia-410M step5000 | 405M | 10.49B | the Pile | 0.297 | 0.434 | 0.607 | 0.141 | 0.098 | 0.045 | 0.066 | 0.071 |
| Pythia-410M step143000 (final) | 405M | 299.9B | the Pile | 0.406 | 0.519 | 0.672 | 0.191 | 0.109 | 0.090 | 0.098 | 0.099 |

(acc / acc_norm both in the JSONs; Pythia acc: HS 0.260 / 0.266 / 0.281 / 0.337, ARC-e norm 0.289 / 0.335 / 0.391 / 0.457,
PIQA acc 0.552 / 0.577 / 0.616 / 0.667; ours at M3 HS acc 0.287, ARC-e norm 0.446, PIQA acc 0.614.)

What it says, honestly:
- **Known benchmarks:** SCI-1 at 3.0B tokens is above Pythia-410M at 4.2B on all three and level with Pythia at 10.5B (HS 0.308
  vs 0.297, ARC-e 0.487 vs 0.434, PIQA 0.609 vs 0.607). This is mostly DATA: DCLM / FineWeb-edu / Cosmopedia are filtered,
  educational sources, the Pile is not (SmolLM / FineWeb papers show ~3-5x token efficiency from such data). It is not an
  architecture result; the architecture claims stay in the equal-data tiers vs our own T++ / Mamba-2.
- **s8 (rules from spread examples):** Pythia is near or below chance at every length and every checkpoint, even with no filler,
  even at 300B tokens. SCI-1 trained on the B2 rule-doc format (b2s source; these two families are held out, the format is not), so
  this also is mostly data. The architectural part is shown elsewhere: at equal data T++ falls to 0.047 at 1.5B (EUREKA 11-02).
- **Past the trained length:** SCI-1 is trained at 4k; at 8k s8 drops 0.555 → 0.236 (still ~2x chance and above every Pythia
  point). The reach past SEQ remains the open gap (skill pack 8k 0.12 at M3).
- Use in the write-up: the reference block, labelled "different data (the Pile), no claim", params + tokens on every row.

## SCI-1 milestone 3 (CHECK on the 4090, 14:09 IST, 2026-11-03; run `59c3cf5`, step 45,777 = 3.0B tokens; old mix, think still in)

Same CHECK as M1 / M2. The full local copy was deleted mid-CHECK by Library's A26 script (same file name; fixed both sides:
Library's temp name, my hard link bf23670), so everything after s8 ran on the stripped `m3_step_045777_weights.pt` (identical
model weights). Logs: `logs/v13_sempty_sci1_checks_59c3cf5_local.log`, `logs/sci1/v13_sempty_gens_sci1_step_045777_59c3cf5.txt`.

| readout | M1 (1.0B) | M2 (2.0B) | M3 (3.0B) |
|---|---|---|---|
| s8 D (scored) | 0.322 | 0.538 | **0.709** |
| skill pack 2k / 4k / 8k + contrast, 20-way | 0.54 / 0.16 / 0.07 | 0.76 / 0.51 / 0.10 | **0.87 / 0.66** / 0.12 |
| clk2 8k (dilation + contrast) | 0.08 | 0.34 | 0.37 |
| t40 far needles (4k / 8k, position 0) | 0.80 | 0.91 | 0.95 |
| HellaSwag (norm) / ARC-e / PIQA (norm) | 0.281 / 0.459 / 0.576 | 0.293 / 0.479 / 0.594 | 0.308 / 0.487 / 0.609 |
| think ON / OFF acc; OFF leak | 0.005 / 0.03; 0.005 | 0.02 / 0.03; 0.24 | 0.015 / 0.04; 0.165 |
| holdout val (blend) at the nearest val step | 3.524 (16k) | 3.358 (30k) | 3.275 (46k) |

- Everything still climbs with data; s8 D +0.17 per 1B tokens, the skill pack at 4k +0.15. 8k stays near chance (0.12 vs 0.05
  for 20-way): the reach past 4k (the trained SEQ) is the open gap, as at M2.
- Think confirms the drop: ON is still not better than OFF; the leak falls (0.24 → 0.165) but stays far above 5 %.
- Holdout falls 0.083 from 2.0B to 3.0B (0.166 from 1.0B to 2.0B), a normal slowing.
- Generations (shown in the gens file): the tool prompt reads the fetched result ("Q7 … boiling point of 78.4 C") and then runs
  past `<|im_end|>` into a new turn; the protocol prompt still copies "12.347 g" and does not round (the cont1 target); the new
  quant prompt (no reasoning line) answers "The answer is 1.25." greedy (wrong; 0.125) and stops at `<|im_end|>`.

## SCI-1 first continuation (DRAFT 12:41 IST, 2026-11-03; FINAL pre-registration below, "cont1 FINAL")

**Why** (milestones 1 and 2 CHECK; Library's A26 `EXPERIMENTS_V14` "A26 SCI-1 MILESTONE 2"):
- the protocol tripwire fired: G1 mind-held at 2k = 0.00 = NOPROT (+0.05 bar);
- what IS learned is the convention (compact format: the convention is said on 0.75–1.00 at 2k, even CONVERT writes "mg"), but the
  copied VALUE is right on only 0.20: the failure is the value, not the format;
- no gain from 1.0B → 2.0B tokens (proto is ~0.3 % of realized tokens);
- the think tripwires fired too (ON 0.02 vs OFF 0.03; OFF leak 24 %).

**Changes** (mix, logged in the header; one lineage row `sci1_cont1`):
1. **think dropped** (2.4 % realized tokens) → web (dclm + fineweb, split as before).
2. **proto4k raised** from ~0.3 % to ~1.0 % of tokens.
3. **a compact value-copy source** (`protoc`, ~0.2 % of tokens; NEEDS Library: `protocol_text_iter(compact_only=True)`, short
   "Report rule." demos with 2–8 demos near the trained notation, gap ≤ 20 filler sentences, 1–3 queries, the converted VALUE as
   the marked answer, trained families only; the held-out families stay excluded).
4. b2s lowered from ~1.2 % to ~0.7 % (s8 D is 0.538 against the 0.40 bar, so it has room).
5. recall4k unchanged (~0.26 %).

**Loss-weight check** (weight per token vs text: b2s 1.55×, proto4k 1.82×, recall4k 1.37×, protoc ≈ 5× ASSUMED like compact b2,
to be measured): synthetic ≈ (0.007·1.55 + 0.0026·1.37 + 0.010·1.82 + 0.002·5) / (…+0.94) ≈ **4.1 % of the loss weight**, at the
stated ~4 % ceiling (bud: a 4.9 % share cost +0.013 holdout). Tripwire kept: holdout > projection + 0.02 → lower the synthetic weight.

**Schedule:** continue from the last full milestone before the switch (`--continue_from step_XXXXXX.pt`), constant LR (`wsd_stable`),
fresh token budget, 500-step re-warm; the old segment's remaining plan is replaced by this one. The decay branch follows at the end
as planned.

**Bars for the next CHECK (2 milestones later):** G1 mind-held 2k ≥ NOPROT + 0.05 (Library); compact value right ≥ 0.50 at 2k;
s8 D ≥ 0.50 (no regression); holdout within the projection + 0.02; the shown generations: round-to-one-decimal answered.
If compact value-copy stays < 0.30 after the 2 milestones, the lesson is "this size does not copy-with-transform from a stated
rule": state it as a limit and stop raising the share.

**Prior work:** none for this exact cell (a measurement); data-mix response to a tripwire, not a method claim.

### cont1 FINAL pre-registration (12:57 IST, 2026-11-03; Lead GO 12:40 with the protoc gate; thinking off by the user via Lead)

**SCI-1 does not think.** The user switched thinking off completely: the `think` source is gone from the mix, there is no
"Reasoning: on/off" switch in the chat template, the generations or the release notes, and the release says plainly that the
model answers directly, with no reasoning trace. Lineage `think_off`: verdict **dropped** (ON 0.02 ≤ OFF 0.03, OFF leak 24 % at M2).
- Leftover-trace check (5,000 docs per source, counts go in the run header notes): `<think>` 0 / 0 / 0 and "Reasoning: on|off"
  0 / 0 / 0 in smoltalk2_mid / chatmt / tools; plain prose "Reasoning:" headings 42 / 7 / 0 are kept as normal text.
- `scripts/sci_gen.py`: the `think` prompt is now a plain `quant` prompt (same question, no system line).

**The protoc gate (passed):** `protoc` = `protocol_text_iter(compact_only=True)` (Library, 2bf4228; registered 0ed4db7): median
94 tokens per doc, measured **8.05× loss weight per token** vs text (not the 5× assumed in the draft). Calibration (`calib5`,
15.7M tokens, 3,839 batches of 4,096, mix seed 43): realized synthetic loss weight 3.67 % at pass 1; with the corrected weights
the prediction is **3.82 %** (≤ ~4 % ceiling).

| source | target share of tokens | sampler weight |
|---|---|---|
| dclm / fineweb | 0.2156 / 0.2065 | 0.23538 / 0.24337 |
| arxiv / pubmed / pes2o / cosmo / finemath | 0.075 / 0.085 / 0.09 / 0.09 / 0.03 | 0.04951 / 0.07549 / 0.10178 / 0.11172 / 0.02436 |
| smoltalk2_mid / chatmt / tools / pysci | 0.03 / 0.03 / 0.04 / 0.09 | 0.02232 / 0.03308 / 0.03257 / 0.04891 |
| b2s / recall4k / proto4k / protoc | 0.005 / 0.0034 / 0.008 / 0.0015 | 0.00712 / 0.00376 / 0.00957 / 0.00108 |

(think 0.025 → 0; b2s 0.017 → 0.005; proto4k 0.004 → 0.008; protoc new. The draft's "~1.0 %" proto4k became 0.8 % so that
protoc's real 8.05× weight keeps the total under 4 %.)

**Launch:** `v13_sempty/tmp/sci1_cont1_env.sh` + `sci1_cont_chain.sh`, from M4 `sci1_main_59c3cf5/step_061036.pt`,
`--continue_from` with 6,499,991,552 tokens (99,182 steps of 65,536 = the rest of planned stage B, to global 160,218), constant
LR 8e-4 (`wsd_stable`, 500-step re-warm), same geometry / SEQ 4096 / batch 16×2 / decontam index / per-source holdout val
(think removed from VAL_SOURCES), FULL milestones every 15,259 global steps (next 76,295), last 4 kept; then the 1B decay.
Switch: `v13_sempty/tmp/sci1_switch.sh step_061036 <hash>` in tmux on the 6000 waits until M4 is written and the log passes it,
checks it loads, stops the old chain, deletes the superseded old `latest.pt`, checks out the hash and starts cont1 (~64 steps of
the old segment past M4 are discarded). Expected switch ~21:00 IST tonight.

**Bars (the CHECK 2 milestones after the switch, i.e. at global 91,554; also read at 76,295):** G1 mind-held 2k ≥ NOPROT + 0.05
(Library); compact value right ≥ 0.50 at 2k; s8 D ≥ 0.50; holdout within the projection + 0.02 (else lower the synthetic weight);
generations: the round-to-one-decimal prompt answered. Compact value copy < 0.30 after 2 milestones → a stated limit of this size
("does not copy-with-transform from a stated rule"), no further share raise.

**Prior work / claim:** a data-mix response to a tripwire, no method claim.

**Launched (21:34 IST, 2026-11-03).** M4 written 21:12 IST; the switcher stopped the old chain at 21:13 (old `latest.pt` + `step_030518.pt`
deleted as planned) but its `git checkout 59d098d` failed on three untracked M1 CHECK result files that the new commit tracks
(the 6000 sat idle ~4 min). Moved them to `~/sci1_untracked_backup_20261103_1546` on the 6000 and launched cont1 by hand, exactly
as the switcher would (tmux `sci1`, 59d098d, 21:17 IST); the switcher now moves such files itself (3ccf9f1). Log
`logs/v13_sempty_sci1_cont1_59d098d_20261103_1547.log`: `[continue]` from step 61,036 / 4,000,055,296 tokens, fresh budget
6,499,991,552, wsd_stable, warmup 500, new 15-source mix (think out, protoc in), decontam sha 08766210…, exact stream cursors; step
61,150 at 37.7k tok/s (same as before). ETA: main to global 160,218 ~Nov 5 21:30 IST, then the 1B decay (~7.5 h). Next milestones
76,295 (~Nov 4 04:40 IST) and 91,554 (~Nov 4 12:00 IST).

## PAUSE RECORD (Lead, 2026-11-24 10:25 IST): every result that landed after the sessions stopped, and the state at the pause

The user paused the project on 2026-11-19 ~16:50 IST (hold) and stopped the 6000 on 2026-11-20 ~11:10 IST. Mind and Library were
not alive when the last results landed, so Lead read them from the committed result files (logs/wio, logs/v14_scope_c_*) and
writes them here. All s8 numbers are 3-save means (14,500 / 15,000 / end); t40 = weighted accuracy over positions x associations.

### Seed trios (far mix, ~103M, 1.0B tokens; within-box by construction)

| arm | box / seed | s8 @8k | s8 D (2k+4k) | shift 0/2k/4k/8k | affine 0/2k/4k/8k | t40 2k / 8k / 16k | HS / ARC-e / PIQA | val |
|---|---|---|---|---|---|---|---|---|
| pam | 4090 / 43 | 0.343 | 0.537 | .80/.61/.44/.23 | .66/.57/.53/.45 | .900/.817/.803 | .274/.451/.597 | 3.4248 |
| nochrono | 4090 / 43 | 0.427 | 0.639 | .87/.71/.56/.34 | .71/.66/.63/.51 | .478/.408/.353 | .273/.464/.592 | 3.4267 |
| split (3 of 6 heads clocked) | 4090 / 43 | 0.473 | 0.695 | .87/.77/.63/.38 | .71/.70/.68/.57 | .844/.742/.692 | .271/.466/.610 | 3.4248 |
| gdn | 4090 / 43 | 0.408 | 0.638 | .90/.51/.33/.14 | .97/.89/.82/.68 | .681/.458/.325 | .275/.440/.594 | 3.385 |
| pam | 4090 / 44 | 0.620 | 0.817 | 1.0/.98/.92/.69 | .72/.71/.66/.55 | .867/.839/.786 | .274/.466/.591 | 3.4397 |
| pam | 6000 / 45 | 0.284 | 0.474 | .56/.42/.27/.13 | .62/.63/.56/.44 | .992/.947/.872 | .272/.461/.592 | 3.4253 |
| nochrono | 6000 / 45 | 0.414 | 0.608 | .80/.65/.49/.25 | .73/.66/.63/.57 | .622/.536/.492 | .275/.459/.601 | 3.4206 |
| split | 6000 / 45 | 0.842 | 0.950 | 1.0/1.0/.99/.92 | .97/.92/.89/.76 | .489/.428/.408 | .274/.465/.589 | 3.4264 |

split_s44 is INCOMPLETE (stopped at step 8,350 / 13,733 by the 11-19 reboot and the pause; latest.pt kept, no readouts);
nochrono_s44 and gdn_s44 never started. So two complete trios exist (4090-s43, 6000-s45). Counts by the rule at 5c1b374f:
- (a) the clock's recall edge, t40 16k pam - nochrono >= +0.22: s43 +0.450, s45 +0.380 -> **2 of 2. REPLICATED.**
  pam's t40 16k over three seeds: 0.803 / 0.786 / 0.872 (spread 0.09); nochrono 0.353 / 0.492.
- (b) the half-clocked body keeps recall (t40 16k >= 0.60 and >= pam - 0.20): s43 yes (0.692), s45 NO (0.408) -> **1 of 2. NOT replicated.**
- (c) the half-clocked body keeps rules (s8 @8k >= nochrono - 0.02): s43 yes (0.473 vs 0.427), s45 yes (0.842 vs 0.414) -> 2 of 2,
  but s8 is inside the seed noise (pam alone: 0.343 / 0.620 / 0.284 over three seeds, a 0.34 range), so this carries no claim.
Reading: the one result that replicates is the clock's exact recall beyond the trained length. The half-clocked body is unstable
across seeds on BOTH probes (s43 kept needles, s45 lost them and gained rules): the split does not fix the trade, it flips it per
seed. The rule probe at 100M / 1B measures acquisition timing (Mind's margin read: seeds differ already at distance 0).

### Set B: far-protocol share x LR (4 x 0.4B continuations of wio pam stable_final, 6000; pre-registered 2331bcd0)

2k scope tuple per +0.1B save (P_fu = any unit with no protocol; scope_fu = unit for a protocol about the other quantity; C = carry, nats).
Swing bar = parent + 0.25: P_fu >= 0.32 or scope_fu >= 0.94. Parent (wio pam stable_final, step 13,733): P_fu 0.07, scope_fu 0.69,
C 1.66 (validity C >= 0.5 met); the arms' first saves (step 13,734) read P_fu 0.07, scope_fu 0.68-0.71, C 1.71-1.76.
Cross-box check (Library, 11-24): the 6000 calibration read of wio pam decay (logs/v14_scope_c_calib_wiopamdecay_a3ffb14b_*) gives
C 1.50 / scope_fu 0.50 / P_fu 0.01, equal to the 4090 WIO SCOPE read (1.496 / 0.50 / 0.01). Table cells re-derived from the jsonls
(n 100 each) by Library: all match except f08_lr06 +0.1B C (1.20, fixed).

| arm | P_fu at +0.1 / +0.2 / +0.3 / +0.4B | scope_fu | C | swung? |
|---|---|---|---|---|
| f08_lr12 (0.8 %, 1.2e-3) | .05 / .01 / .00 / **.53** | .54 / .48 / .24 / .76 | 1.60 / 1.33 / 1.26 / 1.02 | YES (+0.4B) |
| f16_lr12 (1.6 %, 1.2e-3) | .23 / .13 / .00 / .29 | .67 / .62 / .15 / .82 | 1.92 / 1.76 / 1.00 / 1.93 | no (0.29, just under) |
| f08_lr06 (0.8 %, 6e-4) | .06 / .06 / .00 / .03 | .47 / .54 / .10 / .27 | 1.20 / 1.50 / 0.89 / 0.93 | no |
| f16_lr06 (1.6 %, 6e-4) | .03 / .00 / .18 / .04 | .54 / .34 / .77 / .44 | 1.28 / 1.32 / 1.72 / 1.21 | no |

**Verdict by the letter: UNCLEAR** (LR-DRIVEN needed both lr06 arms' max P_fu <= 0.15; f16_lr06 reached 0.18, without swinging).
Reading: at half the LR neither arm swings and the scope error ends lower (0.27 / 0.44 vs 0.76 / 0.82 at +0.4B); at the full LR the
smaller far share swung more, so "share-driven" is not supported. Leaning LR-driven; one continuation per arm, no seeds. If the
project resumes: test a lower constant LR on the release body before changing the mix again.

### The main run (SCI-1 386M) at the pause

Segment 3r (= 3c's mix) ran from 45.875B to **step 749,500 = 49.119B tokens**, stopped at its own save on 2026-11-20 11:11 IST.
Guard on 3r: 46.0B break (the 3f residue, 0.13B in) -> 46.14B pass -> 46.40B pass -> 46.89B clear (715,500, CPU n 40) -> 47B clear (8k P_fu 0.00,
scope_fu 0.01; 2k scope_fu 0.50). Val (trainer set) 2.9946 @ 748,000 (wiki 3.533; best 19.48 ppl earlier). No dry source, no traceback.
Backups (4090 d_drive, from_6000/checkpoints_v13_sempty/sci1_long3r_91a87ed/): final_step_749500.pt FULL (resumable; sha b403ad63),
step_732432_weights.pt (48B; b821d45e), step_747691_weights.pt (49B; 763555ba), step_701914.pt / step_717173.pt (46B / 47B weights).
The 48B and 49B milestone CHECKs never ran (the 4090 sessions were down); they run now as the release CHECK (below).
Not backed up: the 100M arm checkpoints of qllm_twins on the 6000 (scores are in git).

### Pre-registered but NOT run (paused): gfd (gate-frozen decay), plavg (plateau average), gdn_srope (GDN + Selective RoPE, the
closest prior work to the clock, arXiv:2511.17388; required before any novelty claim for the clock), b386 (386M baselines),
BIND-LURE Step 0 / dual-phase binding (v14). Rerun commands are in their sections and in EXPERIMENT_INDEX.md.

### Mind's audit of the PAUSE RECORD (2026-11-24 10:4x IST; recomputed from the raw logs/wio and logs/v14_scope_c_* files)
- Every trio row, the counts and the Set B table reproduce from the raw files. **Fixed in place:** (1) pam s45 and nochrono s45 val
  were swapped: the decay logs read pam_s45 **3.4253**, nochrono_s45 **3.4206** (also fixed in v16/paper/data/numbers.json and the
  seed45_trio lineage row); (2) the 3r guard point "46.86B" is snapshot step 715,500 = **46.89B** (CPU guard, n 40); (3) pam s45
  affine @4k .56 (was .57, rounding of 0.565); (4) the split_geom lineage row still said "seed 44 pending" -> s45 result added.
- **3r resume (was unrecorded):** resumed 2026-11-19 20:23 IST (14:53Z) from seg3r_pause.pt = step 721,000 (47.251B) by
  sci1_resume_after_setb.sh after Set B ended; [resume] restored step, cursors, optimizer and the constant-LR schedule (no warm-up);
  log logs/v13_sempty_sci1_long3r_91a87ed_20261119_1453.log; ran to 749,500 with 0 tracebacks. Pause length 23.6 h. After the
  resume no guard snapshots were taken (the user's hold of 11-19 16:5x IST came first), so the 3r guard series ends at 47B.
- **Per-arm spread over seeds (far mix, ~103M, 1.0B; s43 / s44 / s45 where they exist):** pam: s8 @8k 0.343 / 0.620 / 0.284 (mean
  0.416, range 0.336), t40 16k 0.803 / 0.786 / 0.872 (mean 0.820, range 0.086), val 3.4248 / 3.4397 / 3.4253; nochrono: s8 @8k
  0.427 / - / 0.414, t40 16k 0.353 / - / 0.492, val 3.4267 / - / 3.4206; split: s8 @8k 0.473 / incomplete / 0.842, t40 16k 0.692 / - /
  0.408, val 3.4248 / - / 3.4264. Recall (t40 16k) varies 0.09 across pam's seeds; rules (s8) 0.34.
- **Margin finding, now with 3 pam seeds** (shift M0 / M4k / M8k, 3-save means): s43 +1.41 / -0.27 / -1.11; s44 +5.05 / +2.35 / +0.82;
  s45 +0.53 / -0.95 / -1.64. The seed spread sits at distance 0 already: circuit strength at 1B (acquisition), not survival over
  filler. nochrono s43 / s45: +2.20 / +1.75 at M0. split s43 / s45: +1.91 / **+10.53** (s45's split formed a very strong shift circuit,
  M8k +3.99, while its needles fell to 0.41: one body can land in either regime per seed).

## sft1: a chat SFT stage on SCI-1 386M (PRE-REGISTRATION DRAFT, Mind 2026-11-24 10:4x IST; NOT launched; Lead takes it to the user)

**Why.** The user offers "a bit SFT for smoltalk or chat" so that the released model chats better, consistent with the standing rule
(base with chat, then a separate SFT stage; smoltalk2's SFT config kept UNSEEN by the base: the base saw smoltalk2_mid, chatmt2,
chatsr, tuluif). **Closest prior work:** SmolLM2's SFT on smoltalk (Allal et al. 2025); our difference is only that the base is a
PAM O(1)-state body whose in-state rule and recall skills we guard through the stage (no novelty claim; a release step).
**Data.** HuggingFaceTB/smoltalk2, config SFT (cached locally), with the v7 policy already in v7/data.py: block multilingual /
longalign / aya / function_calling / toolcalling / smolagents splits; caps openthoughts3 0.05, mixture_of_thoughts 0.10, table_gpt
0.20; think-traces dropped (the base answers directly; the Reasoning on/off switch was removed at cont1). ChatML with the 4 added
specials as in the base, the default system prompt as v7. **Loss on assistant content + its closing <|im_end|> only**
(v7 _encode_sft_example: system / user / role headers masked). **Packing:** conversations concatenated into 4096 windows with
RESET_AT_EOS (state reset between conversations, as the base trained), no cross-conversation attention leak by construction of the
O(1) state reset. **Replay:** 20 % of tokens from the 3r base mix (incl. the far rule docs b2s_far / proto_far and recall docs) so the
in-state skills keep being exercised. **Decontamination:** the v2 13-gram index (a4b66d1b, HS / ARC / PIQA / GSM8K / MMLU ...) applied
to SFT conversations (drop on any hit), PLUS a probe-template filter: drop conversations that contain our probe surface forms
(protocol docs "report every mass in grams", b2s / shift_wide / affine_map prompt shells, t40 key-value shells, the chat-probe prompts
of v16/humaneval). Counts (kept / dropped per split, tokens) go into the run header.
**Size.** ~150M assistant-target tokens (~250M total window tokens incl. context and 20 % replay), one pass, no repeats.
**Recipe.** From final_step_749500.pt (49.119B, full, resumable; weights + optimizer). AdamW moments kept. LR: warm-up 200 steps to
**2e-4**, constant to 80 % of the budget, then linear to 0 over the last 20 % (a short decay). Tokens per step 65,536 (16 x 4096,
grad-accum as the base), seed 43. **Expectation stated now:** the closing decay will cost told-rule carry (decay_M9 - M9 dC(2k) -0.37
at 1B of decay; this decay is ~0.05B, so a smaller loss, but the SFT data itself has no long-range rule supervision; replay is there
to limit it). **Box / cost:** the 4090 fits 386M at SEQ 4096 with micro-batch 4 + gradient checkpointing (to be measured in the smoke;
the 6000 ran 37k tok/s at 77 GB). Estimate 4090 ~20-25k tok/s -> 250M tokens ~3-3.5 h; code + tests ~2-3 h; smoke 0.5 h; CHECK +
chat eval ~1.5 h: **~8 h** total on the 4090 (or ~5 h with the 6000 if it is back).
**Code needed (not written):** a v13_sempty train.py --dataset sft_mix path that streams v7's assistant-only SFT encoder packed into
windows with a 20 % base-mix replay and the probe-template filter; parity: with SFT share 0 it reproduces the base loader's sha
(--dump_batches), and an SFT window's loss_mask is 0 on every non-assistant token (unit test).
**Guards vs the 49B CHECK (stated before the run; break = report and do not release the SFT checkpoint):** s8 @8k >= 49B - 0.10 (seed
noise is large on this probe); t40 16k >= 49B - 0.05; clk2 8k >= 49B - 0.05; HS / ARC-e / PIQA each >= 49B - 0.01; told-rule four-cell
tuple at 2k + 8k: P_fu 8k <= 0.20 and scope_fu 8k <= 0.30 (the 3f / 3r guard), C 8k reported (expected to drop; no bar), unit_acc
reported. **Chat judgment (the point of the stage):** our chat probes (v16 prompts) + the human-eval page (v16/humaneval), greedy
and sampled, prompt + decoded output + checkpoint shown; stop-on-<|im_end|> rate on the chat probes >= 0.95 (base: measure at 49B);
answers stay in English, no think traces, no tool-schema leakage; no v11 comparison (user rule). **Verdict:** RELEASE-SFT if every
guard holds and the human eval prefers SFT over base on a majority of the prompts; else the base is the release.
**Recommendation (Mind):** release the BASE first (49B, measured, the artifact the paper describes; it already chats from 40.5B of
chat sources), and ship the SFT checkpoint as a second card once it passes its guards. Reason: the SFT is a new training stage with
a known risk (a decay erases carry), new code (packing + replay + filter) and ~8 h; tying the release to it delays a finished,
checked artifact for an unmeasured one.

## BUG (found 2026-11-24 by Lead's HF-port parity work; fixed 048a01ab): reset_at_eos was not applied in one-token decode
`RealPAMLayer._stepwise_delta` (the T = 1 decode step) never read the separator mask `_keep`, so feeding <|endoftext|> one token at
a time reset only the echo memory, not the notebooks; `_chunked_delta` (training, prefill, every full-sequence forward) did reset
them. On SCI-1 weights the logits after a fed separator differed from the full forward by up to 52.7 (hf_release_sci1/PARITY.md, B).
**Fix:** retention x the separator mask in the decode step, as in the chunked path. **Test:** v13_sempty/tmp/eos_decode_parity.py
(CPU fp32, M2 body: delta, Chrono, echo, share_reads; EOS at three positions; prefill 40 then token-by-token): max|full - decode|
after the EOS 8.3e-07 / 8.9e-07 = the before-EOS level (without the fix 2.3e-04 / 3.4e-04: the test fails); reset_parity.py PASS.
**What it touched (audit):** no scored number. s8 / t40 / clk2 / skill pack / scope / MC / val run the parallel forward; Library's
stepwise v14 probes never feed 50256; v16/package.py prefills in chunks and decodes answer tokens only. LM.generate prefills the
prompt in chunks (correct), so only text generated AFTER a model-emitted <|endoftext|> was decoded without the notebook reset: the
tails of the CHECK generation samples (scripts/sci_gen.py, no EOS stop: e.g. "...<|endoftext|><|im_start|>system ...") and of the
human-eval dumps. Those are descriptive; the text before the first emitted <|endoftext|> is exact. The HF release's step() resets
both memories (correct). Also noted (harmless): chunked vs per-token retention differ by the log(r + 1e-6) epsilon (~1e-3 / 1e-2 in
logits).

## reach32k: how far does the induced rule reach on the release model? (DESCRIPTIVE, no bar; recorded 2026-11-24 before any number; Lead, for the HF card)
Release checkpoint = sci1_long3r_91a87ed latest.pt on the 6000 (= final_step_749500, 49.119B; sha256 checked against b403ad63 before
the run). s8 exactly as the CHECK (b2 suite: shift_wide_new + affine_map_new, 8 shots, 400 trials, seed 7000, 8-way, held-out web
filler) with the examples spread over **8,192 (anchor, 2x the 4,096 training window) / 16,384 (4x) / 32,768 (8x)** in ONE run; plus
t40 (needles, 40 trials x positions 0 / 0.5 / 1 x associations 1 / 4 / 8) at 32,768 if VRAM / time allow. Reported: accuracy and mean
target margin per task and distance; no verdict. Code: the CHECK's probe path (scripts/run_learning_probe.py, memory_probes/spread.py;
the v13_sempty path is unchanged since 59c3cf5), run from the 6000 worktree qllm_sci1e @ 91a87ed.

### sft1 FINAL (Mind 2026-11-24 11:02 IST; code 3b9b2156; supersedes the draft where they differ; user approved SFT as a separate release, Lead gives the GO)
- **Box: the 6000** (user, 10:4x: "release base model and keep plan sft for later tonight (when 6000 will be free)"; free now).
- **Code:** train.py --sft_share 0.8 / --sft_think_fraction 0 + v13_sempty/sft_mix.py (packed, assistant-only, EOS-separated,
  interleaved by kept volume, decontam v2 + probe-template 13-grams, base replay). Verified: test_sft_mix.py ALL PASS; share 0 =
  the base loader bit for bit (wio far mix sha 72e954b7); real-data smoke OK; deterministic order + exact resume skip.
- **Data as measured (first 2,000 kept conversations):** smol_magpie_ultra 41 %, OpenHermes 2.5 34 %, smol_summarize 12 %,
  systemchats_30k 6 %, smol_rewrite 4 %, tulu-3 personas IF 3 %, rest < 1 %; no think traces, no tool / function schemas, English only.
  74 % of SFT window tokens are targets -> ~148M target tokens over the run (0.8 x 250M x 0.74). Decontam / probe drop counts are
  logged in the cursor (#decon / #probe) and reported with the result.
- **Recipe (unchanged):** from final_step_749500 (49.119B; optimizer kept), 65,536 tokens / step, warm-up 200 steps to 2e-4,
  constant to 0.20B, then a separate linear decay to 0 over 0.05B (sci1_cont_chain's main + decay segments); saves every 0.05B.
  Note: 2e-4 is a quarter of the base's 8e-4, so the LR drop itself acts like part of a decay; the guards below are the protection.
- **Before the GO, on the 6000:** a 100-step smoke of the exact command (same env) to check peak VRAM, tok/s and the loss with masks;
  it is deleted, not a result.
- **Guards, chat judgment, verdict:** as in the draft above (vs the 49B CHECK; P_fu 8k <= 0.20 and scope_fu 8k <= 0.30; human eval).
- **Launch (one command, 6000, from a worktree at >= 3b9b2156 with the base at checkpoints_v13_sempty/sci1_long3r_91a87ed/
  final_step_749500.pt, sha b403ad63; the 6000's latest.pt in that dir IS that file, sha-verified 11-24):**
  `tmux new-session -d -s sci1sft1 "bash v13_sempty/tmp/sft1_launch.sh 2>&1 | tee -a logs/v13_sempty_sci1_sft1_launch_<hash>_<stamp>.log"`
  (sft1_launch.sh sources sci1_env.sh + sci1_long3r_env.sh + sft1_env.sh and runs sci1_cont_chain.sh; outputs
  checkpoints_v13_sempty/sci1_sft1_<hash> (constant) and sci1_decay_<hash> (the release candidate = its best_model.pt)).
- **Cost on the 6000:** ~2 h train (37k tok/s solo) + CHECK + chat eval ~1.5 h.
**reach32k RESULT (6000, 11:07-11:20 IST 11-24; release checkpoint sha b403ad63 verified; descriptive, no bar; logs/learning_probes/
v13_sempty_sci1_release_749500_91a87ed_reach32k_spread_s7000_t400.json, logs/memory_probes/v13_sempty_sci1_release_749500_91a87ed_t40_32k.json,
log logs/v13_sempty_sci1_reach32k_91a87ed_20261124_0517.log):**

| examples spread over | x training window | shift_wide acc (margin) | affine_map acc (margin) | s8 mean | free generation (shift / affine) |
|---|---|---|---|---|---|
| 8,192 (anchor) | 2x | 0.938 (+7.57) | 0.990 (+10.23) | **0.964** | 0.927 / 0.860 |
| 16,384 | 4x | 0.693 (+1.90) | 0.890 (+4.24) | **0.791** | 0.642 / 0.520 |
| 32,768 | 8x | 0.200 (-1.69) | 0.575 (+0.42) | **0.388** | 0.152 / 0.155 |

8-way candidates, chance 0.125; 400 trials per cell (binomial se <= 0.025). The 8k anchor agrees with the 47B CHECK (0.958).
**t40 needles at 32,768 (8x):** 0.781 overall (n 360, se 0.022); by needle position: start 0.575, middle 0.767, end 1.000; by
associations 1 / 4 / 8: 0.817 / 0.817 / 0.708.
Reading (descriptive): the induced rule holds at 4x the trained window (0.79, margins positive on both tasks) and fades by 8x
(affine still above chance with a small positive margin, shift near chance with a negative margin); exact recall of a fact holds
better than the rule at 8x (0.78; 0.58 even with the needle at the very start, 32k tokens back).

### SCI-1 base RELEASED (2026-11-24 11:38 IST, Lead)
- HF: https://huggingface.co/gowravvishwakarma/qllm-pam2-sci1-386m — final_step_749500 (49.12B tokens; ckpt sha256 b403ad63…),
  exported fp32 `sci1_386m_49b.pt` sha256 03254d70…; standalone `modeling_pam2.py` (parity max logit diff ≤ 2.4e-4, greedy identical).
  Card source: `hf_release_sci1/` (weights not in git). The v11 card points to it; it points back.
- Card numbers (release checkpoint): s8 spread 8k 0.94/0.99, 16k 0.69/0.89, 32k 0.20/0.58 (reach32k, c58aec8d);
  t40 8k 0.98, 16k 0.94, 32k 0.78; HS/ARC-e/PIQA 0.372/0.559/0.636 (acc_norm / acc / acc_norm, as Pythia; plain acc HS 0.320, PIQA 0.638); Library's lure / prose / scope limits (660734ee).
- Chat script fix before release: `run_chat.py` double-counted the pending `<|im_end|>\n` in its history / penalty context
  (the model's input was always right); SAMPLES regenerated, text unchanged.
- SFT (sft1) runs next on the 6000 as a separate release.

## Release CHECKs (Lead's lead_relcheck on the 4090, 2026-11-24, ../qllm_sci1check @ 59c3cf5 = the milestone CHECK code; outputs committed by Mind)

| save | tokens | s8 @8k (shift / affine) | skill @8k | t40 16k | clk2 8k / 32k-run @16k | HellaSwag acc / norm | ARC-e acc / norm | PIQA acc / norm |
|---|---|---|---|---|---|---|---|---|
| 48B (step 732,432) | 48.00B | 0.950 (0.925 / 0.975) | 0.680 | 0.922 | 0.92 / 0.58 | 0.3230 / 0.3698 | 0.5593 / 0.5059 | 0.6404 / 0.6268 |
| 49B (step 747,691) | 49.00B | 0.961 (0.955 / 0.968) | 0.635 | 0.744 | 0.89 / 0.46 | 0.3200 / 0.3669 | 0.5522 / 0.5000 | 0.6458 / 0.6469 |
| **release (step 749,500)** | **49.12B** | **0.964** (0.938 / 0.990) | 0.645 | **0.942** | 0.90 / 0.40 | 0.3200 / **0.3719** | **0.5593** / 0.5008 | 0.6376 / 0.6355 |

Harness: scripts/mc_eval.py (fp32; parity with lm-eval 0.4.9.1 to 4 decimals on Pythia-160M, 59e9197), full validation sets, zero-shot.
t40 16k moves 0.92 -> 0.74 -> 0.94 over three saves 0.07B apart: single-save noise on this cell is ~0.1 at 386M too.
**sft1 SMOKE + LAUNCH (6000; user GO via Lead; maya / nexus confirmed stopped, no compute apps, right before both):** smoke 11:36-12:55
IST (exact launch path, 100 constant + 10 decay steps, throwaway dirs deleted after): [continue] from final_step_749500 (49.119B),
lr 2e-4, decontam v2 a4b66d1b on, 624,211 probe 13-grams, kept-volume interleave; masked SFT loss 1.56 -> 1.15; val 2.978 / 2.977
(base 2.9946), wiki 3.541 / 3.537; decay resume skipped 7,696 kept conversations correctly; **peak 79.4 GB** of 98; both segments
exit 0. Each segment's stream warm-up is ~20-35 min (base streams re-open at exact cursors; known). **sft1 LAUNCHED 12:56 IST
2026-11-24** (tmux sci1sft1, worktree ~/Development/qllm_sft @ 2094021 (>= 3b9b2156), log
logs/v13_sempty_sci1_sft1_2094021_20261124_0726.log, dirs checkpoints_v13_sempty/sci1_sft1_2094021 then sci1_decay_2094021 = the
release candidate).
**sft1 RELAUNCH (13:56 IST 11-24, code 431eabec):** the first launch (12:56, 2094021) was DATA-BOUND: 12-13k tok/s after warm-up
(base 37k), GPU idle ~65 %; one data worker ran the HF smoltalk2 stream + v7 filters + two 13-gram hashes per conversation (~3 SFT
windows/s of ~7 needed), and its decay-segment resume would have re-filtered ~300k conversations. Stopped before any save (dirs
deleted, logs moved to logs/superseded on the 6000). Fix: one producer process per split (started in the trainer before the worker
forks), interleave per kept conversation by kept volume, cursor = draws + per-split raw rows (resume via skip_rows, no re-filtering),
one hash for both indexes, network errors retried from the current row. Verified on real smoltalk2 (with HF rate-limit and timeout
retries occurring): deterministic order, exact mid-stream resume. Recipe, data policy, filters and guards unchanged; only the order of
the SFT conversations differs from the first launch. New log logs/v13_sempty_sci1_sft1_431eabe_20261124_0826.log (6000).
