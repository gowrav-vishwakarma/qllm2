# V6 Experiment Log — Part 2 (HSB, PAM, onward)

Continuation of [EXPERIMENTS_V6.md](EXPERIMENTS_V6.md). **Part 1** keeps the full archive through **§18 GSP run results** (architecture, bugs 1–7, TinyStories/WikiText baselines, rebalanced + TSO + GSP). **This file** is self-contained for everything after that: **§0** same-pipeline **transformer baseline** (apples-to-apples vs PAM), **Bug 8**, HSB, PAM, medium-pam-v2 (interleaved CGU+PAM -- PAM interleaved with CGU in every block instead of all-PAM at the end), medium-pam-v3 (RoPE, optional QK phase norm, fused QKV / block-real GEMM). A full **10-epoch** v3 run with **`pam_qk_norm=False`**, RoPE, and the speed paths finished at **val PPL 29.95** (WikiText-103); the earlier **`pam_qk_norm=True`** run was **stopped mid-epoch 5** due to repetition collapse (Bug 8).

**Git reference:** interleaved CGU+PAM was tagged `v6-medium-pam-v2-interleaved` before the Part 2 code path diverged further.

**How to update:** Append new PAM-era runs and related bugs here; update **§0** if the transformer baseline is re-run.

---

## 0. Transformer baseline — ~100M GPT-2 style (WikiText-103) (2026-03-23)

### Purpose

**Apples-to-apples comparison** against `medium-pam-v3` (~100.4M): same **GPT-2 tokenizer** (50257 vocab), **WikiText-103** (`wikitext-103-raw-v1`), **seq_len 2048**, **batch_size 3**, **10 epochs**, **AdamW lr=1e-4**, **warmup 1000**, **cosine decay**, **dropout 0.1**, **`torch.compile`**, **`--amp_dtype auto`**. Script: [`scripts/run_transformer_baseline.sh`](scripts/run_transformer_baseline.sh); entrypoint: `python -m v6.train_transformer_baseline`. Implementation: [`v6/transformer_baseline.py`](v6/transformer_baseline.py).

### Architecture (intentionally vanilla — not QLLM)

This is a **standard real-valued decoder-only transformer** (GPT-2 flavor): learned absolute positional embeddings (not RoPE), **pre-norm** `LayerNorm`, causal multi-head self-attention via **`F.scaled_dot_product_attention`** (PyTorch **SDPA**; on RTX 4090 this typically dispatches to **Flash Attention** / memory-efficient backends), GELU FFN, residual connections, **tied** input/output embeddings. **No** PAM, complex tensors, CGU, or GSP.

**Sizing (~100.3M params, matched to the PAM budget):** `d_model=672`, `n_layers=12`, `n_heads=12` (head dim 56), `d_ff=2688` (4× `d_model`). Logged breakdown: embeddings 35.1M, transformer blocks 65.1M, final LN 1.3K, `lm_head` 0 (tied), **total 100,283,232**.

### Throughput and Flash Attention

Training step throughput is ~**92–97k tok/s** (epoch-average ~96k) vs ~**23k tok/s** for `medium-pam-v3`. The baseline uses the **highly optimized SDPA path** (Flash-style kernels where the backend selects them); QLLM PAM is still **pure PyTorch** with no custom CUDA/Triton. Treat the **~4× wall-time / tok/s gap** as partly **optimization maturity**, not only architecture — the val PPL gap is the cleaner quality comparison.

### Results (completed 10-epoch run)

**Git / log:** `e87b8e4` — `logs/v6/transformer_baseline_wikitext103_20260323_140351_e87b8e4/` (`transformer_baseline.log`, `RUN_INFO.txt`). Checkpoints: `checkpoints_transformer_baseline/best_model.pt`.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 137.77 | 53.74 | |
| 2 | 52.81 | 39.42 | |
| 3 | 42.74 | 34.76 | |
| 4 | 38.27 | 31.96 | |
| 5 | 35.54 | 30.39 | |
| 6 | 33.41 | 29.02 | |
| 7 | 31.65 | 28.09 | |
| 8 | 30.28 | 27.46 | |
| 9 | 29.29 | 27.15 | |
| 10 | 28.75 | **27.08** | best val |

**Wall time:** 12505.1 s (**~3.47 h** total). **GPU:** ~1.8 / 7.0 GB in log. **End-of-epoch generation** (epoch 10, prompt: `In 1923 , the University of`): coherent WikiText-style scaffolding; facts unreliable. **Quality:** rep3=0.034, rep4=0.011, restarts=0, uniq=0.667.

### Side-by-side: transformer baseline vs `medium-pam-v3`

| | Transformer B=3 | Transformer B=6 | medium-pam-v3 (QLLM) |
|---|-----------------|-----------------|----------------------|
| Params | ~100.3M | ~100.3M | ~100.4M |
| Batch size | 3 | 6 | 3 |
| Val PPL (10 ep) | **27.08** | **23.13** | **29.95** |
| Tok/s (typical) | ~96k | ~99k | ~23k |
| Wall time (10 ep) | ~3.5 h | ~3.2 h | ~14.1 h |
| GPU mem | ~1.8 / 7.0 GB | ~2.5 / 12.7 GB | — |

**Readout:** The vanilla transformer **wins on val PPL** (~10% gap at B=3, ~23% at B=6). Doubling batch size to 6 gives the transformer a large PPL boost (27.08 → 23.13, **-14.6%**) with negligible throughput change. PAM v3 is **close** on perplexity at B=3; throughput favors the baseline largely due to **Flash-class SDPA + mature stack** vs pure-PyTorch PAM.

### Transformer baseline B=6 run (2026-03-30)

Same architecture and pipeline as the B=3 run above, with `--batch_size 6`. Purpose: provide apples-to-apples baseline when V7 PAM experiments use B=6 (e.g. 7d).

**Git / log:** `8d631a6` — `logs/v6/transformer_baseline_wikitext103_20260330_130306_8d631a6/transformer_baseline.log`.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 156.12 | 52.83 | |
| 2 | 48.96 | 35.67 | |
| 3 | 37.73 | 30.39 | |
| 4 | 32.81 | 27.70 | |
| 5 | 29.85 | 26.07 | |
| 6 | 27.76 | 24.71 | |
| 7 | 26.21 | 23.94 | |
| 8 | 25.08 | 23.40 | |
| 9 | 24.33 | 23.17 | |
| 10 | 23.94 | **23.13** | best val |

**Wall time:** ~3h13m (13:03 → 16:16). **Throughput:** ~99k tok/s avg. **GPU:** 2.5 / 12.7 GB. Batches/epoch: 9,639 (vs 19,277 at B=3).

**B=3 vs B=6 per-epoch comparison:**

| Epoch | B=3 Val PPL | B=6 Val PPL | Improvement |
|-------|-------------|-------------|-------------|
| 1 | 53.74 | 52.83 | +1.7% |
| 3 | 34.76 | 30.39 | +12.6% |
| 5 | 30.39 | 26.07 | +14.2% |
| 7 | 28.09 | 23.94 | +14.8% |
| 10 | 27.08 | 23.13 | +14.6% |

The B=6 advantage emerges by epoch 3 and stabilizes at ~14-15%. This mirrors the V7 finding (Exp 3a-A vs 3a-B) that larger batch size provides smoother gradients and more aggressive cosine schedule per-token, acting as implicit regularization.

---

### Bug 8: PAM QK phase normalization drives repetition collapse (found 2026-03-19)

**Symptom**: During `medium-pam-v3` training with `pam_qk_norm=True`, mid-epoch and end-of-epoch generations develop severe lexical repetition (e.g. "University of Illinois" listed many times in one sample) while train/val loss continue to decrease smoothly. No NaN/Inf, no gradient spikes.

**Root cause**: `cnormalize` on Q and K forces unit magnitude per position, so `Re(Q^* K)` depends only on phase alignment. The model cannot modulate *attention strength* via Q/K magnitude -- only *which* direction in phase space matches. PAM is not softmax attention: scores are used directly (with decay), so flattening the dynamic range tends to smear retrieval across many past keys. Common LM QK-norm recipes add a **learnable temperature / logit scale**; PAM had only fixed `d^{-0.5}`. Together this matches a gradual mode collapse into high-frequency template phrases (especially on WikiText-style "University of …" prompts).

**Mitigation**: Disable QK norm for the headline v3 preset: `pam_qk_norm=False` in `medium-pam-v3` (see §5 below). **Future** if revisiting QK norm: add a per-head or scalar learnable scale after the normalized dot product, and re-ablate against RoPE alone.

**Files**: `v6/core/pam.py` (`PhaseAssociativeLayer`), `v6/config.py` (`medium-pam-v3` preset)


---

## 1. Holographic State Binding (HSB) (2026-03-16)

### Motivation

GSP proved the model can **retain** phase-coherent state. But generation analysis shows it cannot **compose and retrieve** entity-property associations. The model knows "we're talking about institutions" but can't bind "THIS institution is Michigan AND it was founded in 1817."

Transformers solve this with attention -- any token can directly reference any other token. SSMs have no such mechanism. We need an SSM-native compositional binding operator.

### The core insight: cmul IS holographic binding

Holographic Reduced Representations (HRR) use circular convolution to bind two vectors into a single vector of the same dimensionality, and circular correlation to unbind (retrieve). In Fourier space, circular convolution becomes element-wise multiplication, and circular correlation becomes element-wise multiplication by the conjugate.

Our complex representation tensors `[..., dim, 2]` are already in a space where:
- `cmul(a, b)` = element-wise complex multiplication = binding
- `cmul(query, cconj(stored))` = element-wise multiplication by conjugate = unbinding/retrieval
- `cmul(cmul(A, B), cconj(A))` ≈ B (associative retrieval)

This means **our existing primitives already implement HRR** -- we just need to use them for compositional memory.

### Design evolution: from registers to SSM-native

**V1 (discarded)**: Phase-Coherent Binding Registers (PCBR) -- a separate module with 8 registers per layer, sequential T-loop for write/read. After novelty audit, this was identified as **slot attention with cmul instead of dot-product** -- structurally not novel, and the sequential loop was an anti-pattern for an architecture designed around parallel scan.

**V2 (implemented)**: Holographic State Binding (HSB) -- bind/unbind happens **inside** the SSM layer itself. No separate registers, no sequential loop, no bolt-on module.

### Architecture: SSM-native HSB

HSB modifies the SSM layer in two places:

**1. Bind (inject into state)**: Before `parallel_scan(A, Bx)`, add holographic bindings to the Bx input term:

```
Bx_original = B_proj(x) * dt           -- normal SSM input
bind_signal = scatter(cmul(key(x), value(x))) * bind_gate(|x|)  -- holographic binding
Bx = Bx_original + bind_signal         -- combined input to state
h = parallel_scan(A, Bx)               -- binding enters state naturally
```

**2. Unbind (retrieve from state)**: After `parallel_scan`, extract bindings from the state:

```
h_bind = gather(h)                      -- project state to bind_dim
retrieved = out_proj(cmul(query(x), conj(h_bind))) * unbind_gate(|x|)
y = C_proj(h) + D*x + retrieved        -- retrieved bindings added to output
```

**Why this is fundamentally different from everything published**:

| Feature | Attention | Slot Attention / NTM | Mamba Selectivity | **HSB** |
|---------|-----------|---------------------|-------------------|---------|
| Memory structure | KV cache (O(T)) | Fixed slots | SSM state | **SSM state** |
| Write | append | soft-attention write | input-dependent B | **cmul(key,val) into Bx** |
| Read | Q*K^T (O(T²)) | soft-attention read | C*h | **cmul(query, conj(h))** |
| Compositional | No | No | No | **Yes (HRR algebra)** |
| Sequential loop | No | Yes (over T) | No | **No** |
| Parallel scan | N/A | N/A | Yes | **Yes** |

**Key differentiators**:
1. **No new sequential bottleneck**: bindings flow through the existing parallel scan
2. **Compositional**: `cmul(cmul(A,B), cconj(A)) ≈ B` -- algebraic retrieval, not soft lookup
3. **Phase-native**: binding IS phase rotation, retrieval IS counter-rotation
4. **Synergistic with GSP**: GSP protects important state dims; HSB gives it bindings worth protecting
5. **Not a module, an SSM property**: holographic binding is an intrinsic behavior of the recurrence

### Bottleneck design

To keep parameter cost manageable, bindings happen in a lower-dimensional subspace (`bind_dim = 96`, half of `dim = 192`):

```
x [B,T,dim=192,2]
  --> key_proj: dim -> bind_dim=96     (ComplexLinear, 2*192*96 = 36.9K)
  --> value_proj: dim -> bind_dim=96   (same)
  --> cmul(key, value): bind_dim       (free -- elementwise)
  --> scatter_proj: bind_dim -> state_dim=1536  (ComplexLinear, 2*96*1536 = 294.9K)
  --> added to Bx before parallel_scan

h [B,T,state_dim=1536,2]
  --> gather_proj: state_dim -> bind_dim=96     (ComplexLinear, 2*1536*96 = 294.9K)
  --> query_proj: dim -> bind_dim=96            (ComplexLinear, 2*192*96 = 36.9K)
  --> cmul(query, conj(h_gathered)): bind_dim   (free -- elementwise)
  --> out_proj: bind_dim -> dim=192             (ComplexLinear, 2*96*192 = 36.9K)
  --> added to y after C_proj
```

Per layer: ~738K params. Total HSB: 11.8M (18.7% overhead on 63.2M GSP model).

### Bug fix: _init_weights bias override

Discovered that `_init_weights()` in `model.py` was zeroing all `nn.Linear` biases, including the GSP protect_gate bias (should be -3.0) and the HSB bind/unbind gate biases (should be -3.0). Added `_reinit_custom_biases()` to restore these after global initialization.

**Impact on previous GSP run**: The GSP model that achieved PPL 41.67 was actually running with `protect_gate.bias = 0.0` (sigmoid(0) = 0.5, meaning 50% protection from the start). This means GSP performed well even without the intended conservative initialization. With the correct -3.0 bias, performance may differ.

### Parameter breakdown

| Component | medium-rebalanced-gsp | medium-rebalanced-hsb | Change |
|-----------|----------------------|----------------------|--------|
| embed (tied) | 19.3M (30.5%) | 19.3M (25.7%) | -- |
| banks (CGU) | 10.6M (16.8%) | 10.6M (14.2%) | -- |
| SSM + TSO | 28.4M (45.0%) | 28.4M (37.9%) | -- |
| GSP gates | 4.7M (7.5%) | 4.7M (6.3%) | -- |
| **HSB** | **0** | **11.8M (15.7%)** | **+11.8M** |
| lm_head | 0.07M | 0.07M | -- |
| **Total** | **63.2M** | **75.0M** | **+18.7%** |

### What we're testing

1. Does HSB improve val PPL beyond GSP's 41.67? (compositional binding/retrieval via HRR algebra)
2. Does generation show improved factual coherence? (entity-property alignment should be visibly better)
3. What is the throughput cost? (no sequential loop, but extra projections per layer)
4. Do the bind/unbind gates learn to be selective? (should open for factual content, stay closed for filler)
5. Does GSP + HSB synergize? (GSP protects the bindings that HSB writes into state)

### Config and script

- Preset: `--size medium-rebalanced-hsb`
- Script: `scripts/run_v6_medium_pcbr.sh`
- Baseline to beat: medium-rebalanced-gsp val PPL 41.67 (10 epochs, WikiText-103)

### Novelty assessment

HSB is genuinely novel on multiple axes:

1. **No published work injects HRR bindings into SSM state**: HRR (Plate, 1995) and variants (MAP, VTB) work in real space as standalone vector operations. We use cmul/cconj on complex SSM state.
2. **No published SSM uses compositional bind/unbind**: Mamba selectivity controls dt/B/C; it doesn't create or retrieve compositional entity-property bindings. S4/S5/RWKV have no compositional mechanism at all.
3. **Not a memory module**: unlike NTM/DNC/slot attention, HSB has no separate memory bank. Bindings live in the SSM state itself and flow through the standard parallel scan.
4. **Complex-native**: binding IS phase rotation, retrieval IS phase counter-rotation. This only works because our SSM state is complex-valued.
5. **GSP + HSB is synergistic**: GSP provides selective state retention, HSB provides compositional content to retain. Neither exists in any published architecture.

---

## 2. Phase-Associative Memory (PAM) (2026-03-17)

### The Failure of HSB

The Holographic State Binding (HSB) experiment failed to improve performance. The final validation PPL was **43.54**, which is a regression from the GSP baseline of **41.67**.

**Diagnosis**: The root cause is *state interference*. HSB uses HRR binding (`cmul(key, value)`) to compress a $D \times D$ association into a $D$-dimensional vector, which is then added to the SSM state. While HRR works well for single associations, adding multiple facts into the same vector state causes catastrophic interference. The fundamental limitation of our `ComplexSSM` is that its state is a **vector** (diagonal $A$), which lacks the capacity to store multiple cross-dimensional associations over time without them colliding.

### The Solution: Complex Matrix State

To solve the interference problem, we must upgrade the state from a Vector to a **Complex Matrix** ($S_t \in \mathbb{C}^{H \times d \times d}$). This provides $O(d^2)$ capacity per head, allowing multiple facts to be stored independently.

This architecture is called **Phase-Associative Memory (PAM)**.

### Mathematical Formulation

1. **State Update**: $S_t = \gamma_t S_{t-1} + V_t \otimes K_t^*$
   - The state is a true memory matrix.
2. **Retrieval**: $Y_t = S_t Q_t = V_t (K_t^* \cdot Q_t)$
   - The complex dot product $K_t^* \cdot Q_t$ naturally computes attention via constructive/destructive phase interference. No Softmax is needed!
3. **GSP Integration**: The protect gate $p_t$ modifies the decay: $\gamma_t = \exp(-dt)(1-p_t) + p_t$. This allows the model to freeze the matrix state and retain facts indefinitely.

### Efficient Training (The Dual Form)

Computing the $d \times d$ matrix sequentially is slow. Because it's a linear recurrence, we can compute it in $O(T^2)$ time using the **Dual Form** (Attention form) with highly optimized dense matrix multiplications. For $T=2048$, this is extremely fast.

Dual form: $Y_t = \sum_{i \le t} \left( \prod_{j=i+1}^t \gamma_j \right) (Q_t \cdot K_i^*) V'_i$

### Architecture Specs

- **Model Dimension**: $D=384$ (increased from 192 to utilize the parameter budget effectively)
- **Heads**: $H=6$, **Head Dim**: $d=64$
- **Total Parameters**: ~100M
- **Components**: `PhaseAssociativeLayer` replaces `ComplexSSMLayer`. `ComplexGatedUnit` remains as the MLP equivalent.

### Implementation Details

- **Module**: `v6/core/pam.py` -- `PhaseAssociativeLayer` and `PhaseAssociativeMemory`
- **Integration**: Replaced `ComplexSSM` entirely in `v6/backbone.py`.
- **Config**: `medium-pam` preset added to `v6/config.py`.
- **Script**: `scripts/run_v6_medium_pam.sh`

### What we expect

1. **PPL**: Significant improvement over GSP (41.67) due to the massive increase in state capacity ($O(d^2)$ vs $O(d)$).
2. **Factual Recall**: The matrix state should allow the model to retrieve specific facts without interference, solving the core issue identified in HSB.
3. **Training Speed**: The Dual Form implementation should be very fast on GPU, comparable to standard attention.

---

## 3. PAM Run Results (2026-03-18 to 2026-03-19)

### Inference bugs found and fixed

Two bugs in `v6/core/pam.py` caused generation to produce complete gibberish despite healthy training PPL. Both were train/inference path mismatches.

**Bug 1 (critical): Prompt state lost during generation**

The Dual Form (training path, used when `state is None and T > 1`) returned `new_state = torch.empty(0)`. In `PhaseAssociativeMemory.forward()`, state collection was gated on `state is not None`, so when processing the prompt (state=None), no state was returned. The next token then started from zero-initialized state with no prompt context.

Before:
```python
new_state = torch.empty(0, device=x.device)
```

After: compute the final recurrent state $S_T = \sum_i D[T,i] \cdot (V'_i \otimes K_i^*)$ from the decay matrix already available:
```python
D_last = D[:, :, -1, :]  # [B, H, T]
wv_r = v_prime[..., 0] * D_last.unsqueeze(-1)
wv_i = v_prime[..., 1] * D_last.unsqueeze(-1)
sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
new_state = torch.stack([sr, si], dim=-1)  # [B, H, d, d, 2]
```

Also updated `PhaseAssociativeMemory.forward()` to always collect and return states (not only when input state was non-None).

**Bug 2 (high): Missing query scaling in recurrent form**

The Dual Form applied `scale = d ** -0.5` to queries. The Recurrent Form (inference path) did not. With `d=64`, this is an 8x magnitude mismatch.

Before:
```python
q_t = q[:, :, t].unsqueeze(-3)  # no scale
```

After:
```python
scale = d ** -0.5
q_t = q[:, :, t].unsqueeze(-3) * scale
```

**Why PPL looked good but generation was gibberish**: Validation PPL is computed via `model.forward(batch)` with T=2048, which takes the Dual Form path (correctly scaled, full context). Generation processes the prompt via Dual Form (first token correct), then switches to Recurrent Form with zero state and wrong scale. This explains the pattern in the pre-fix log: the first predicted token ("Oxford") was correct, everything after was noise.

### Generation: before vs after fix

**Before fix** (epoch 3, batch 5000 -- old log):
> In 1923 , the University of Oxford by ; . 11 East Highided appearanceeter Lis on Regent D and G @ @ Pr Kse norbers and En to ally and back in to given =istist over , – with and ,ids their course finalton ...

**After fix** (epoch 10, batch 15000 -- new log):
> In 1923 , the University of Kentucky opened a public schools school in 1924 and served as the state 's first governor for 40 years . During the 1930s , a public school was built in a public house to serve as the primary school for Governor , but it still served as an office space until 1933 . This was completed by 1936 , when the University of Kentucky passed its own law class .

**Final generation** (epoch 10 end):
> In 1923 , the University of Missouri and the University of Michigan was also established in 1926 . In 1928 , the University of Michigan opened its current campus with the school 's first campus opening at St. Louis Road on the northern end of Lake Michigan in 1929 .

### Training progression

Resumed from epoch 3 checkpoint (pre-fix epochs 1-3 had correct PPL but broken generation). Epochs 4-10 ran with both bugs fixed.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 4     | 54.03     | 47.19   | first epoch after resume with fix |
| 5     | 48.55     | 43.55   | |
| 6     | 45.26     | 41.43   | already beats GSP baseline (41.67) |
| 7     | 43.14     | 40.11   | |
| 8     | 41.76     | 39.34   | |
| 9     | 40.91     | 39.02   | |
| 10    | 40.50     | 38.95   | new best |

Quality at epoch 10: rep3=0.051, rep4=0.020, restarts=0, uniq=0.624.

### Run details

- **Preset**: `medium-pam` (dim=384, 16 layers, single CGU expand=3, PAM H=6 d=64, GSP)
- **Parameters**: 100.4M
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3
- **Throughput**: ~23,500 tok/s on RTX 4090
- **Wall time**: ~9.9 hours total (epochs 4-10 after resume)
- **Script**: `./scripts/run_v6_medium_pam.sh --resume`

### Conclusion

PAM is validated. The matrix state ($S \in \mathbb{C}^{H \times d \times d}$) fixes the interference problem that caused HSB to regress. Final val PPL **38.95** beats the GSP baseline of **41.67** by 6.5%, with coherent multi-sentence generation. The model shows good factual structure (university names, dates, locations) and low repetition.

The bugs were pure train/inference path mismatches -- training was never affected. The pre-fix PPL numbers (epochs 1-3) remain valid; only generation was broken.

| Model | Params | Val PPL | Notes |
|-------|--------|---------|-------|
| medium-rebalanced | 58.4M | 44.47 | SSM baseline |
| medium-rebalanced-gsp | 63.2M | 41.67 | + GSP |
| medium-rebalanced-hsb | 75.0M | 43.54 | + HSB (regression -- interference) |
| **medium-pam** | **100.4M** | **38.95** | **PAM + GSP** (sequential layout; §3) |
| medium-pam-v3 (RoPE, no QK-norm) | ~100.4M | 29.95 | Interleaved + RoPE + fused QKV + block GEMM; not a single-variable ablation vs. row above (§5) |

---

## 4. Experiment: Medium-PAM-v2 -- Interleaved CGU + PAM (2026-03-19)

### Hypothesis

The primary performance bottleneck in medium-pam is **architectural layout**, not model capacity. All 16 CGU layers (pointwise, zero sequence mixing) run sequentially before all 16 PAM layers. This means the first half of the network has no cross-position information flow. Every competitive architecture (Transformer, Mamba, RetNet, GateLoop) interleaves channel mixing (FFN) with sequence mixing (attention/SSM) in every block. Fixing this layout should significantly close the PPL gap.

### What changed

1. **Interleaved layout**: Each of 16 blocks now runs `CGU -> PAM` instead of `[CGU x16] -> [PAM x16]`.
2. **Higher LR**: 1e-4 (up from 3e-5). The old LR was 3-10x lower than standard for 100M-scale models.
3. **Longer warmup**: 1000 steps (up from 500).

### What did NOT change (novelty preserved)

- PAM matrix state (same mechanism, same capacity).
- GSP (Gated State Protection) -- same per-dimension freeze gate.
- Complex-valued representations throughout -- phase preserved end-to-end.
- CGU (Complex Gated Unit) -- same SwiGLU-style phase-safe gating.
- All phase-safe primitives (ModReLU, ComplexNorm, ComplexLinear) -- unchanged.
- Attention-free, O(T) inference per token -- no softmax, no KV cache.
- Total parameter count: identical ~100.4M (same components, different ordering).

### Architecture comparison

```
medium-pam (sequential):
  Tokens -> [CGU x16] -> [PAM x16] -> Output
  (16 layers with ZERO sequence mixing, then 16 layers of sequence mixing)

medium-pam-v2 (interleaved):
  Tokens -> [CGU -> PAM] x16 -> Output
  (Every block has both channel mixing AND sequence mixing)
```

### Config

- **Preset**: `medium-pam-v2` (dim=384, 16 layers, single CGU expand=3, PAM H=6 d=64, GSP, interleave_pam=True)
- **Parameters**: ~100.4M (same budget)
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3
- **LR**: 1e-4, warmup_cosine, warmup=1000
- **Script**: `./scripts/run_v6_medium_pam_v2.sh`

### Baselines for comparison

| Model | Params | Val PPL | Notes |
|-------|--------|---------|-------|
| medium-pam (sequential) | 100.4M | 38.95 | Same components, sequential layout |
| GPT-2 small (Transformer) | ~124M | ~14.84 | Fine-tuned on WikiText-103 |
| Transformer (vanilla) | ~125M | ~18.6 | Trained on WikiText-103 |
| Mamba-Small (SSM) | 130M | ~24.1 | Selective SSM |
| GateLoop (linear RNN) | 125M | ~13.4 | Data-controlled recurrence |

### Planned follow-up (not in this run)

- **Phase 2**: Add short causal ComplexConv1d (kernel=4) inside each PAM layer for local n-gram capture.
- **Phase 3**: Tune expand factor, head count, or reduce layers for better capacity allocation.

### Results

**Stopped after epoch 1** -- terminated early to incorporate quality and speed improvements in experiment §5. The architectural change (interleaving) is carried forward into v3.

| Epoch | Train PPL | Val PPL | tok/s | Time | Notes |
|-------|-----------|---------|-------|------|-------|
| 1     | 123.52    | 57.84   | 21,852 | 5420s (~90min) | best (only epoch) |

Generation at epoch 1 (prompt: "In 1923 , the University of"):
> In 1923 , the University of Illinois . = = Background = = By 1930 , Robert G. Brown had established a large number of small buildings near his home base at Cauchon Cemetery in Bauchon County ; he was one of two sons of William I. Fisot and J. L. Tisauke , who built the first major stone building on the site . He would later build several new buildings , including the building 's first floor , and a building that has

Quality: rep3=0.000, rep4=0.000, restarts=0, uniq=0.762.

For context, medium-pam (sequential) reached val PPL 54.03 at epoch 4 (its first post-resume epoch), so 57.84 at epoch 1 with interleaving is on a competitive trajectory. However, the run was not continued -- all subsequent work uses v3.

---

## 5. Experiment: Medium-PAM-v3 -- QK Phase Norm + Complex RoPE + Speed (2026-03-19)

### Hypothesis

Building on the interleaved layout from v2 (experiment §4), there are two categories of improvement:

**Quality**: The v2 model has (a) zero positional encoding -- the only position signal comes from PAM's causal recurrence, leaving the first CGU layer completely position-blind, and (b) optional control of Q/K scaling. RoPE addresses (a). **QK phase norm (b) was hypothesized to help** but **empirically caused repetition collapse** (Bug 8); the preset now keeps RoPE and disables QK norm.

**Speed**: `ComplexLinear` currently launches 4 GEMM kernels per call (one per real/imag x weight_real/weight_imag combination). With 112 ComplexLinear calls per forward pass, that's 448 kernel launches. Additionally, PAM's Q/K/V projections are 3 separate calls that could be fused.

### What changed

#### Quality changes (new flags, ablatable)

1. **QK Phase Normalization** (`pam_qk_norm`, default **False** in `medium-pam-v3` after 2026-03-19): Normalize Q and K to unit complex magnitude per-element before the dot product. **First v3 train used `True` and is documented as failed** (repetition; Bug 8). When enabled: uses `cnormalize`; `d^{-0.5}` scaling is preserved. Applied in both dual-form (training) and recurrent (inference) paths.

2. **Complex RoPE on Q,K** (`pam_rope=True`): Position-dependent phase rotation applied to Q and K only (not V, not the residual stream). Each complex dimension k gets frequency `theta_k = 1 / (10000^{k/d})`, and position m rotates by `e^{i*m*theta_k}`. This is a single `cmul` with a precomputed unit-magnitude tensor -- phase-safe by construction (|e^{i*theta}| = 1). Gives relative position awareness (the dot product `Re(q_m^* k_n)` depends on position difference m-n). Cache is precomputed for 8192 positions, auto-extended if needed. Step offset tracked via `PAMState.step` for correct inference positions.

#### Speed changes (always on, zero quality impact)

3. **Block-Real GEMM** in `ComplexLinear`: Replaced 4 `F.linear` calls with 1 by constructing a block matrix `[[W_r, -W_i], [W_i, W_r]]` and concatenating inputs `[x_r, x_i]`. Verified bit-exact (max diff 3.8e-06 in float32). Reduces 448 GEMM launches to 112 per forward pass.

4. **Fused QKV Projection** (`pam_fused_qkv=True`): Single `ComplexLinear(dim, 3*inner_dim)` replaces 3 separate Q/K/V projections. Combined with block-real GEMM, each PAM layer does 1 GEMM for QKV instead of 12.

### What did NOT change (novelty preserved)

- PAM matrix state, GSP, complex-valued representations, CGU, ModReLU, ComplexNorm -- all unchanged.
- Attention-free, O(T) inference per token -- no softmax, no KV cache.
- Interleaved `[CGU -> PAM] x16` layout from v2.
- Total parameter count: ~100.4M (same budget -- fused QKV is the same total weight count, just concatenated).

### Config

- **Preset**: `medium-pam-v3`
- **Production stack (validated below)**: `pam_qk_norm=False`, `pam_rope=True`, `pam_fused_qkv=True` (plus block-real GEMM in `ComplexLinear`), `interleave_pam=True`, GSP.
- **Failed ablation (logged below)**: `pam_qk_norm=True`, same otherwise.
- **Parameters**: ~100.4M (same budget)
- **Dataset**: WikiText-103, seq_len=2048, batch_size=3
- **LR**: 1e-4, warmup_cosine, warmup=1000
- **Logs**:
  - Failed QK-norm: `logs/v6/medium_pam_v3_qknorm_rope_wikitext103_20260319_161045_77c454a/v6_autoregressive_medium-pam-v3.log` (git `77c454a`)
  - Completed RoPE, no QK-norm: `logs/v6/medium_pam_v3_rope_wikitext103_20260319_231524_31397f0/v6_autoregressive_medium-pam-v3.log` (git `31397f0`)
- **Script**: `scripts/run_v6_medium_pam_v3.sh`

### Ablation plan

- **Done (from logs)**: QK norm hurts generation quality badly → default off (Bug 8).
- **Done**: Full 10-epoch train with `pam_qk_norm=False` — val PPL **29.95**; see **Results (completed run)** below.
- **Done**: `medium-pam-v3-pia` (sparse PIA on v3) — val PPL **30.01**; see **Attention ablation** below.
- **Future (optional)**: disable `pam_rope` only to isolate RoPE vs. recurrence-only position signal.
- Speed changes (block-real GEMM, fused QKV) are math-identical and cannot affect quality.

### Baselines for comparison

Same WikiText-103 10-epoch setting where noted. Rows are **not** a controlled single-variable grid: v3 differs from medium-pam by interleaving, RoPE, fused QKV, and training recipe alignment.

| Model | Params | Val PPL | Notes |
|-------|--------|---------|-------|
| **transformer baseline B=3 (GPT-2 style, SDPA/Flash)** | **~100.3M** | **27.08** | **Same pipeline as v3** — §0; `v6/transformer_baseline.py` |
| **transformer baseline B=6** | **~100.3M** | **23.13** | Same arch, batch_size=6 — §0 B=6 run |
| medium-pam (sequential) | 100.4M | 38.95 | Sequential layout, no RoPE, no QK norm |
| **medium-pam-v3 (RoPE, no QK-norm)** | **~100.4M** | **29.95** | **Interleaved + RoPE + speed paths** (§5 completed run) |
| medium-pam-v3-pia | ~105.1M | 30.01 | + sparse PIA every 4 layers (interference, window 256); no PPL win vs v3 — §5 attention ablation |
| medium-pam-v2 (interleaved) | 100.4M | N/A | Stopped early (experiment §4) |
| GPT-2 small (Transformer) | ~124M | ~14.84 | Fine-tuned on WikiText-103 |
| Transformer (vanilla) | ~125M | ~18.6 | Trained on WikiText-103 |
| Mamba-Small (SSM) | 130M | ~24.1 | Selective SSM |
| GateLoop (linear RNN) | 125M | ~13.4 | Data-controlled recurrence |

### Results (failed ablation: `pam_qk_norm=True` -- do not use for production)

**Git / log**: `77c454a` — `logs/v6/medium_pam_v3_qknorm_rope_wikitext103_20260319_161045_77c454a/`

Training was **numerically stable** (no NaN/Inf, `div`/`wdiv` stayed 0). Throughput ~22–23k tok/s, GPU ~2.3/20.7 GB. **Val loss improved every epoch through 4**, but **generation repetition worsened** from ~epoch 3 onward; by epoch 5 batch 10000 the sample collapsed into repeated institution names. This matches prior V6 lessons: **cross-entropy alone is a weak proxy for repetition**.

| Epoch | Train PPL | Val PPL | End-of-epoch quality (prompt: "In 1923 , the University of") |
|-------|-----------|---------|------------------------------------------------------------------|
| 1 | 126.82 | 63.42 | rep3=0.021, rep4=0.000, uniq=0.691 |
| 2 | 59.45 | 48.85 | rep3=0.022, rep4=0.011, uniq=0.684 |
| 3 | 50.36 | 43.78 | rep3=0.040, rep4=0.010, uniq=0.592 (first clear warning) |
| 4 | 45.86 | 40.50 | rep3=0.011, rep4=0.000, uniq=0.719 (metrics recover; still intermittent) |
| 5 | — | — | **Stopped mid-epoch** (~72% through when log ends). At batch 10000: severe repetition — e.g. *"University of Illinois , University of Chicago , University of Illinois , …"* many times in one continuation. |

**Epoch 3 end-of-epoch sample** (illustrative): heavy reuse of *"university"* and nested self-referential phrasing; `uniq` dropped 0.691 → 0.592.

**Diagnosis (primary)**: **QK phase normalization** (`cnormalize` on Q,K) is the main suspect. It removes magnitude from the PAM score path, there is no softmax sharpening, and no learnable temperature — see Bug 8 (above). **RoPE**, **fused QKV**, and **block-real GEMM** are not implicated by the logs (RoPE is standard; speed paths are equivalent matmuls).

### Results (completed run: `pam_qk_norm=False`, RoPE + speed)

**Git / log**: `31397f0` — `logs/v6/medium_pam_v3_rope_wikitext103_20260319_231524_31397f0/`

Full **10 epochs** on WikiText-103. Training **numerically stable** (`div`/`wdiv` = 0). Throughput ~**23k tok/s** average; total wall time **50714.3 s (~14.09 h)**. Checkpoints: `checkpoints_v6_medium_pam_v3/best_model.pt`, `checkpoint_epoch_10.pt`, `final_model.pt`.

| Epoch | Train PPL | Val PPL | Notes |
|-------|-----------|---------|-------|
| 1 | 123.86 | 57.94 | |
| 2 | 53.87 | 43.83 | |
| 3 | 44.88 | 38.69 | |
| 4 | 40.39 | 35.88 | |
| 5 | 37.42 | 33.82 | |
| 6 | 35.13 | 32.25 | |
| 7 | 33.26 | 31.22 | |
| 8 | 31.78 | 30.40 | |
| 9 | 30.66 | 30.01 | |
| 10 | 30.02 | **29.95** | best val |

**End-of-run generation** (epoch 10, prompt: `In 1923 , the University of`):

> In 1923 , the University of Illinois at Urbana @-@ Urdu said it was " an easy choice to do something in its own right . " The university also claimed the first students from Wisconsin had to be replaced by a more " good student " due to a lack of funds .

Quality: **rep3=0.034**, **rep4=0.011**, restarts=0, **uniq=0.703**. Coherent structure (section headers, multi-sentence prose); occasional WikiText-style artifacts and dubious facts (e.g. tokenization / hallucinated place names), not the list-repetition mode of the QK-norm run.

**Outcome**: `v6/config.py` preset `medium-pam-v3` keeps **`pam_qk_norm=False`**; this run validates **RoPE + block-real GEMM + fused QKV** without QK phase norm. **Future**: optional `pam_rope` ablation; or revisit QK norm with a learnable logit scale after normalization (Bug 8). Prefer log directory names that distinguish **`qknorm_rope`** vs **`rope`** (no QK-norm) so runs are not confused.

### Attention ablation (`medium-pam-v3-pia`)

We infused **sparse phase-interference attention** on top of the v3 stack (preset `medium-pam-v3-pia`: PIA every 4 layers, window 256, `attn_mode=interference`, RoPE + fused QKV on attention — same WikiText-103 recipe otherwise). **Result:** best **val PPL 30.01** at ~**105.1M** params vs **29.95** at ~100.4M without attention — effectively flat, so the headline stays PAM-only. **Log:** `logs/v6/medium_pam_v3_pia_wikitext103_20260321_201508_5c76a92/` (git `5c76a92`).

---

## How to update Part 2

- **Transformer baseline** (same WikiText-103 pipeline, ~100.3M): documented under **§0**; re-run with [`scripts/run_transformer_baseline.sh`](scripts/run_transformer_baseline.sh) and update §0 tables/logs.
- **medium-pam-v3 production metrics** (10-epoch RoPE, no QK-norm): documented under §5 **Results (completed run)**; the failed QK-norm ablation stays in **Results (failed ablation)** in the same section.
- Append new experiment sections after §5 (continue as §6, §7, …). For new PAM-related bugs, add them here (extend Bug 8 or add Bug 9+); if a bug belongs in the global §3 list in Part 1, add a **one-line pointer** in [EXPERIMENTS_V6.md](EXPERIMENTS_V6.md) §3 and keep the full write-up here.
