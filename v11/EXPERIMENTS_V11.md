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

## Live run status (2026-06-17)

Single 96GB GPU. **Phase B complete** — all memory-dynamics ablations done.

- **DONE** — E3 K=3 (`v11_e3_k3`): val **25.77** @10e, ~29k tok/s, 12.5h wall.
  **New best PAM run.** Lock as `v11_best` scale candidate.
- **DONE** — E3 K=2 (**26.01**), E1 (**26.87** tie), E1+E3 combo (stopped — no stack),
  E2 (killed — compile/OOM).
- **NEXT** — Phase C: better pretrain data + instruct SFT on `v11_e3_k3` (or K=2 if
  cost/latency matters more than the 0.24 PPL gain).

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
| E3 multistate K=2 | `v11_e3_multistate` | **26.01** | ~36k | 100.5M | **WIN** (−0.87 vs control) |
| E3 multistate K=3 | `v11_e3_k3` | **25.77** | ~29k | 100.5M | **BEST** (−1.11 vs control, −0.24 vs K=2; ahead every epoch except ep2 tie) |
| E1+E3 combo | `v11_e1e3_combo` | stopped @ep4 | ~47k | 105.1M | **per_channel does NOT stack** — behind E3 every epoch (see below) |

Anchors (same pipeline): transformer B=18 **22.69**, V7 7d B=18 ModSwish **26.88**.

### Read-out (2026-06-17, Phase B complete)
- **E3 K=3 is the new best PAM config**: 26.88 → **25.77** (−1.11 vs 7d control).
  K=3 beat K=2 at **every epoch except ep2 (tie)**; final gain **−0.24 PPL** for ~20%
  throughput cost (29k vs 36k tok/s). Generation ep10: `rep3=0.030`, `uniq=0.670` —
  on par with K=2 and 7d. **Lock `v11_e3_k3` for Phase C scaling.**
- **E3 K=2 remains the efficiency pick**: 26.01 @ ~36k tok/s — use if inference/training
  cost matters more than the extra 0.24 PPL from K=3.
- **E1 (per-channel decay) is a tie** (26.87 ≈ 26.88) but **faster** (~58k vs ~48k tok/s);
  quality-neutral; does **not** combine with E3 (combo behind E3 every epoch).
- **E2 (delta) hung** under `torch.compile` (the per-chunk complex triangular solve), and
  the `--no-compile` fallback is impractical: probe OOM'd at B=18 (88 GB) and ran at only
  **~5.9k tok/s** → **Killed and deferred.** Needs compile-friendly back-substitution.

Per-epoch K-sweep (val PPL):

| epoch | 7d control | E3 K=2 | E3 K=3 | K=3 vs K=2 |
|------:|-----------:|-------:|-------:|------------:|
| 1 | — | 82.72 | 81.54 | −1.18 |
| 2 | — | 45.59 | 45.59 | tie |
| 3 | — | 35.86 | 35.75 | −0.11 |
| 4 | — | 31.82 | 31.47 | −0.35 |
| 5 | — | 29.33 | 29.07 | −0.26 |
| 6 | — | 27.85 | 27.65 | −0.20 |
| 7 | — | 26.83 | 26.57 | −0.26 |
| 8 | — | 26.29 | 26.05 | −0.24 |
| 9 | — | 26.05 | 25.80 | −0.25 |
| 10 | **26.88** | **26.01** | **25.77** | **−0.24** |

K=3 scaling verdict: **modest but monotonic** — worth adopting for quality-first runs;
diminishing returns vs K=2 suggest **K=4 is low priority** unless Phase C data unlocks it.

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

### Cross-version learnings (V6 → V7a → V7d → V11)

WikiText-103, ~100M, seq_len=2048, B=18 where noted. Transformer B=18 anchor: **22.69**.

| Milestone | val@10 | Δ vs prior | tok/s | What changed |
|-----------|-------:|-----------:|------:|--------------|
| V6 `medium-pam-v3` | **29.95** | — | ~96k* | Interleaved CGU+PAM, RoPE, fused QKV; proved PAM works at scale |
| V7 **7a** ModSwish (B=3) | **29.73** | −0.22 | ~21k | ModSwish CGU; V7 refactor parity with V6 |
| V7 **7d** chunked B=6 | **27.94** | −1.79 | ~32k | Chunked dual form O(T·C); big PPL + speed win |
| V7 **7d** chunked B=18 | **26.88** | −1.06 | ~22k | Larger batch; beats transformer B=3 (27.08) |
| V11 E3 K=2 | **26.01** | −0.87 | ~36k | Multi-state phase interference (K=2); first in-family memory win |
| **V11 E3 K=3** | **25.77** | −0.24 | ~29k | More superposed states; **best logged PAM** |
| Transformer B=18 | **22.69** | — | ~96k* | Ceiling; gap from best PAM: **+3.08 PPL** |

\*Transformer tok/s from V7 notes; PAM ~4× slower at matched geometry — FLOP-fundamental
(matrix-state d² work), not fixable by kernel fusion (Flash-PAM negative).

**What worked (cumulative stack):**
1. **Depth + narrow** (16×384) over wide-shallow — V7 3a proved depth dominates.
2. **ModSwish CGU** — phase-preserving activation; beats ModReLU (+0.58 at 7d B=18).
3. **Chunked dual form** — training speed + PPL; the compiled PyTorch path is near optimal.
4. **B=18 batch** — fewer steps/epoch but much lower val PPL vs B=3/B=6.
5. **E3 multi-state superposition** — the only architectural change that beat 7d on quality;
   uniquely PAM (phase-routed retrieval, not transformer/SSM).

**What did NOT work (dead ends, do not revisit without new evidence):**
- Hierarchy / grouped layers / cross-level drift (7f-3 halted ep1)
- Multi-scale auxiliary loss (7f-1: +0.82 PPL)
- Reverse association (7f-2: +2.46 PPL)
- Learned positional embeddings (7pos: ±0.2 PPL, neutral)
- V9 readout gates / conv gates (29.57 confounded → clean runs 30+ PPL)
- E1 per-channel decay (tie alone; hurts when stacked with E3)
- E2 delta-rule write (compile hang + OOM without compile)
- Flash-PAM kernel reformulation (correct, slower)
- Triton custom kernels under `torch.compile` (bypassed or graph-break)

**Architecture identity preserved throughout:** complex phase space, matrix-state PAM,
conjugate retrieval, O(1)/token recurrent inference, no softmax attention, no KV cache.

**Gap analysis:** V6→V11 closed **4.18 PPL** (29.95→25.77). Remaining gap to transformer
B=18 is **3.08 PPL** — likely needs **data + scale**, not more small ablations at 100M.
Phase C (richer pretrain → instruct SFT) is the sensible next move.

### Next (Phase C)
1. Wire instruct/chat data path (chat template, assistant-only loss, packing).
2. Pretrain `v11_e3_k3` on dclm-edu/fineweb-edu (or similar), then SFT (tulu-3/smoltalk2).
3. Optional: short LR/warmup sweep on K=3 before scaling (low priority — K=3 already stable).

## Out of scope (already dead in V6–V9)
Hierarchy/grouped/cross-level, multi-scale loss, reverse-assoc, QK-norm-on, PIA
attention, V8 reasoning loops, V9 readout gates.
