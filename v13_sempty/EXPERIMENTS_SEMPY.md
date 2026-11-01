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
