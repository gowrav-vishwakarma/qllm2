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
1. **R1 `--dt_spread 8`** (this commit): head h inits at −4 − 8·h/5 ⇒
   half-lives 38 / 190 / 940 / 4.6k / 23k / 113k tokens. Zero params, zero
   cost (98.9k tok/s = ref), decode intact. Launched `b55c49a`.
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
