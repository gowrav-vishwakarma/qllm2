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

Mixed at 150M: best holdout (−0.031 vs echo, in-distribution DCLM-Edu) but
WikiText +2.6 % (out of distribution). Not a rule trigger (T1 is judged on
holdout, recall, PG-19); watch whether the WikiText gap closes.
