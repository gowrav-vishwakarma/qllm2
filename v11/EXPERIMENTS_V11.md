# V11: Maturing PAM via new memory dynamics

Goal: close the gap to the transformer (7d **26.88** vs transformer B=18 **22.69** on
WikiText-103, ~100M) by changing **how the PAM state writes and forgets** — the one
frontier V9/V10 flagged as unexplored — while keeping the architecture identity
(complex, phase-first, matrix-state, conjugate retrieval) and **O(1)/token inference**.

NOT a transformer, NOT a vector-state SSM (Mamba/Samba). Every lever below is a
single-variable change vs the V7 7d baseline (`v11_baseline`).

## Architecture (unchanged core)

```
ComplexEmbed -> [ pre-norm CGU (ModSwish) + pre-norm PAM ] x16 -> tied complex LM head
PAM:  S_t = gamma_t * S_{t-1} + V_t (x) K_t* ;  Y_t = S_t * Q_t   (conjugate match, no softmax)
train = chunked dual form O(T*C) ; infer = recurrent O(1)/token, no KV cache
```

Only `V11PAMLayer` ([v11/model.py](model.py)) changes, dispatching on 3 flags
(defaults reproduce 7d exactly):

| Flag | baseline (7d) | new |
|------|---------------|-----|
| `decay_mode` | `head` (scalar/head) | **`per_channel`** (E1) |
| `write_mode` | `additive` | **`delta`** (E2) |
| `n_states` | `1` | **`K>1`** (E3) |

## Correctness

For all 4 modes the **parallel training form == O(1) recurrent form** (max abs Δ < 1e-7,
fp64). See [v11/selftest.py](selftest.py):

```
.venv/bin/python -m v11.selftest
```

## Experiments

### E1 — Per-channel data-dependent decay
`gamma` becomes per-(head, key-channel) `[B,H,T,d]` instead of per-head `[B,H,T]`, so
different associative slots forget at different rates (the matrix state's effective rank
saturates ~10/64 today — capacity is wasted). Chunked training folds a chunk-local
cumulative decay into Q (×α) and K (×1/α) → plain conjugate score (GLA-style, fp32,
clamped). Inference: `S = S·diag(γ_t) + V K*` = O(d²)/token. V9's explicitly
recommended next change.

### E2 — Delta-rule (error-correcting) write
`S_t = γ_t S_{t-1}(I − β_t k_t k_t*) + β_t v_t k_t*`: before writing key `k_t`, erase
the stale value already associated with it (targets entity drift / repetition). Training
uses a per-chunk UT transform — solve `(I+M)U=W` with `M` strictly-lower (complex,
block-real solve), then read. Inference: rank-1 update O(d²)/token. `delta_chunk=64`.

### E3 — Multi-state superposed PAM (phase-routed)
K=2 matrix states per head with distinct decay biases; retrieval interferes them by a
data-dependent phase: `Y_t = Σ_k e^{iφ_k(x_t)} S_k Q_t`. Constructive/destructive phase
interference selects which state answers — the uniquely-QLLM idea (V7 "Exp 5"). Nearly
param-matched (only `phase_proj` + K decay biases added). Inference O(K·d²)/token.

## Protocol

Control = `v11_baseline` (== 7d, B=18, chunk 256). Each lever: tiny + 3-epoch smoke,
then full 10-epoch WikiText-103 B=18. **Accept** if it beats control val PPL at matched
B/steps AND does not regress generation (`rep3`,`uniq`). **Kill** if epoch-3 val does not
track below control (same discipline as 7f-3).

Runs are serialized on the single GPU via [scripts/run_v11_queue.sh](../scripts/run_v11_queue.sh)
(waits for the GPU to free, then runs E1 → E3). Throughput note: E2's chunk solve is the
slow path; Flash-PAM ([v11/triton_kernels.py](triton_kernels.py)) targets the speedup.

## Live run status (2026-06-16)

Single 96GB GPU, serialized. Update as runs land.

- **RUNNING** — Phase 0 baseline `v7 7d` (ModSwish, B=18, chunk 256), tmux `v11_base`.
  10 epochs, ETA ~10h. Establishes the `v11_baseline`/control bar (target ~26.88).
- **QUEUED (armed)** — tmux `v11_queue` (`scripts/run_v11_queue.sh`), waits for ≥70GB
  free then runs in order:
  1. `v11_e1_perchannel` (E1, ~105M) — next up.
  2. `v11_e3_multistate` (E3, ~100.5M).
  3. `v11_e2_delta` (E2, ~100.4M) — last (slow path until Flash-PAM).
- **DONE (negative)** — Flash-PAM Track 1 (`v11/flash_pam.py`): correct but slower than
  the compiled baseline at every batch size → **not adopted** (details below).
- **AFTER** — Phase B (combine winning levers + LR/wd sweep) → Phase C (instruct/chat
  data: pretrain → SFT best 100M).

Smoke status: E1/E2/E3 build + train at full dims (B=2), no OOM/shape errors;
parallel==recurrent verified in `selftest.py`.

## Flash-PAM design constraint (from V7–V10 experience)

Custom-autograd **Triton kernels fight `torch.compile`** on this split-real complex
stack — recorded evidence:
- `v7/triton_kernels.py` guards every fused op with `not torch.compiler.is_compiling()`,
  i.e. under compiled training the Triton path is **bypassed** (PyTorch fallback runs).
- `v7/EXPERIMENTS_V7.md`: removing Triton (+ CGU) gave **+66% tok/s** — "no Triton
  compilation, simpler graph for `torch.compile`".
- `v10/base_to_start.md`: "Triton kernel compilation + complex compile graph" named as
  overhead; `v8/backbone/model.py`: opaque kernels cause **graph breaks** that block fusion.

**Therefore Flash-PAM is two-track:**
1. **Track 1 (primary, training):** pure-PyTorch **chunk-parallel** reformulation —
   batch independent intra-chunk GEMMs, keep only the cheap d×d state-carry sequential.
   No custom autograd → inductor fuses it into the compiled graph. Validated numerically
   (fwd+grad) vs `_forward_chunked_head`.
2. **Track 2 (secondary, eager + O(1) inference only):** tiled Triton kernel behind the
   same `is_compiling()` guard, so it never fragments the compiled training graph.

Benchmarks run in **spare VRAM** (small B/dims) alongside the live baseline to decide
whether the speedup justifies enabling it.

### Track 1 result — evaluated, NOT adopted (no speedup)

Built `v11/flash_pam.py` (`flash_pam_chunked_head`) + harness `v11/bench_flash_pam.py`.
Correctness: fp64 forward matches `_forward_chunked_head` to **2.5e-13**, grads ~1e-7
(fp32/triton diffs are pure float noise) — math is identical.

Speed (single PAM layer, fp32, T=2048, C=256, RTX PRO 6000), `torch.compile` fwd:

| B  | baseline | flash | verdict |
|----|---------:|------:|---------|
| 2  | 2.31 ms  | 3.21 ms | slower |
| 8  | 9.60 ms  | 13.02 ms | slower |
| 16 | 24.62 ms | 26.05 ms | slower (+ ~2× peak mem) |

**Why:** once inductor compiles the baseline, the n=8 chunk loop is already cheap and the
per-chunk GEMMs at `B*H≈288` batch already saturate the GPU. Folding chunks into the
batch (`B*H*n`) adds no utilisation but ~2× the D/score memory traffic → net slower.
Confirms the V7 lesson: the **compiled pure-PyTorch chunked dual form is already near
the efficient frontier**; the tok/s gap vs transformers is FLOP/bandwidth-fundamental
(PAM's matrix state), not an un-fused-kernel problem.

**Decision:** keep the existing `_forward_chunked_head`. Do NOT ship Track 1. Skip the
Track-2 Triton kernel for training (would be bypassed under `is_compiling()` anyway).
If **E2/delta** proves promising-but-slow, optimise its specific bottleneck (the per-chunk
complex triangular **solve**), not this generic reformulation.

## Results (fill from logs)

| Run | preset | val@10 | tok/s | params | verdict |
|-----|--------|-------:|------:|-------:|---------|
| baseline (anchor) | `v11_baseline` (7d) | **26.88** | ~48k | 100.4M | control |
| E1 per-channel decay | `v11_e1_perchannel` | **26.87** | ~58k | 105.1M | **tie** (no quality gain, ~20% faster) |
| E2 delta write | `v11_e2_delta` | — | — | 100.4M | **inconclusive** — hung after compile (linalg.solve × torch.compile) |
| E3 multistate K=2 | `v11_e3_multistate` | **26.01** | ~36k | 100.5M | **WIN** (−0.87 vs control, ~25% of gap to transformer; ~25% slower) |
| E1+E3 combo | `v11_e1e3_combo` | stopped @ep4 | ~47k | 105.1M | **per_channel does NOT stack** — behind E3 every epoch (see below) |

Anchors (same pipeline): transformer B=18 **22.69**, V7 7d B=18 ModSwish **26.88**.

### Read-out (2026-06-17)
- **E3 (multi-state superposition, K=2) is the best lever**: 26.88 → **26.01** at matched
  params/steps. The phase-routed superposition (the uniquely-QLLM idea) genuinely adds
  capacity. Cost: ~25% throughput (2× state work). Per-epoch it tracked below control from
  epoch ~3 — passes the accept rule.
- **E1 (per-channel decay) is a tie** (26.87 ≈ 26.88) but **faster** (~58k vs ~48k tok/s);
  it's essentially free, so it's a good *combine* candidate, not a standalone win.
- **E2 (delta) hung** under `torch.compile` (the per-chunk complex triangular solve), and
  the `--no-compile` fallback is impractical: probe OOM'd at B=18 (88 GB) and ran at only
  **~5.9k tok/s** → a 10-epoch run would take **~2–3 days**. **Killed and deferred.** To get
  E2's number it needs a compile-friendly back-substitution (replace `torch.linalg.solve`),
  only worth doing if delta later looks promising — we already have a winner (E3).

### Phase B finding — E1+E3 combo: per_channel does NOT stack (stopped @ epoch 4)
Per-epoch val PPL (combo tracked between E1 and E3, **behind E3 at every epoch**):

| epoch | E3 K=2 | E1 pc | combo |
|------:|-------:|------:|------:|
| 1 | 82.72 | 84.97 | 85.59 |
| 2 | 45.59 | 47.74 | 47.03 |
| 3 | 35.86 | 37.19 | 36.68 |
| 4 | 31.82 | 32.83 | 32.34 |

Gap to E3 narrowed (2.87→0.52) but combo was never ahead; best realistic outcome was a
*tie with E3 at higher cost* (105.1M params + per_channel overhead). Confirms E1's verdict:
per-channel decay is quality-neutral here. **Stopped to free the GPU for the K-sweep.**
Decision: **drop per_channel**; keep the plain head-decay E3 line as the winner to push.

### Next (Phase B candidates)
1. **E3 with K=3** (head decay, more states) — does the multistate gain scale past K=2? (running)
2. If K=3 helps: short LR/warmup sweep on the chosen config; else lock E3 K=2.
3. Scale the best config on better data (Phase C).

## Out of scope (already dead in V6–V9)
Hierarchy/grouped/cross-level, multi-scale loss, reverse-assoc, QK-norm-on, PIA
attention, V8 reasoning loops, V9 readout gates.
