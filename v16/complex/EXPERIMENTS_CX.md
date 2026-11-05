# v16/complex experiments (the complex PAM arm)

## cx_m2: complex PAM with the paper body's parts vs ctl's real PAM M2 (PRE-REGISTRATION, 08:12 IST, 2026-11-05; user GO via Lead 08:02 / 08:05)

**Question:** at matched params and the identical product recipe, does the complex-arithmetic PAM, carrying the same parts as the
paper's (real) body, differ from it? The parts were ported first (user: "the complex head carries the same parts, then a fair run").

**Arm cx_m2** (`v16/complex/model.py`, code 5c0b766 + this note): complex `baseline` geometry (dim 384 complex, 6 heads × 64, 16 layers),
**notebooks at 3 / 7 / 11 / 15** with **A3 delta** (realified fused kernel; parity 1.5e-5 vs a true-complex reference), **N1
chrono**, **N4 out_gate**, **M2 shared reads** in the other blocks (complex reader of the writer's WY write, writer's clock),
**RESET_AT_EOS**, **E1 echo** (complex succession memory at the head; ported 08:19 IST at the user's request because ctl's PAM M2 and
SCI-1 both run with echo, and the clk2 reach read dilates the echo memory), complex MLP width **1436** → **102,947,436 params**
(ctl PAM M2: 102,884,948; +0.06 %). **Every part of the paper's real body is ported**; only the arithmetic differs.

**Recipe = ctl's exactly:** the cont2 mix (`sci1_cont2_env.sh` sampler weights), SEQ 4096, BATCH 16 × 4096 = 65,536 tokens per
step (**16 micro-batches of 1**: the complex body peaks at 12.9 GB per 4096-token sequence, fwd + bwd, ~3x the real body, so 4 x 4 and 8 x 2 both OOM'd on the 4090; exact accumulation, same step), LR 1.2e-3, warmup 500, 0.9B wsd_stable + 0.1B decay = 1.0B tokens, seed 43, decontam index sha 08766210…,
answer weight 100, ANSWER_IN_WINDOW, per-source holdout val, venv qllm311 (as ctl). Code: ctl ran caa4da9, cx_m2 runs the launch hash (≥ 5c0b766);
the real-arm code is identical between them (v13 model restored; only v16/complex + the 2-line build_lm route + an unused data
source differ). Runner `v16/complex/cx_chain.sh` (the ctl scheduler: yields, resumes from latest.pt, VRAM gate).

**Bars (ctl's B1–B5, read as cx_m2 − ctl PAM M2):** B1 s8 D (|Δ| ≥ 0.10 = a difference); B2 far needles 8k / 16k (|Δ| ≥ 0.10);
B3 holdout (|Δ| ≥ 0.03 nats); B4 HS / ARC-e / PIQA (reported); B5 told rule (A26 G1 2k twin read; reported as "not run" if the
package path does not support the complex state). Reference (ctl PAM M2): s8 D 0.523, needles 8k / 16k 0.53 / 0.41, holdout 3.427,
HS / ARC-e / PIQA 0.282 / 0.455 / 0.588, B5 0.53. Also reported against ctl's T++ / Mamba-2 (same recipe).

**Prior record:** EXPERIMENTS_SEMPY:209 (Sept 2026, plain bodies, different recipes): real ≈ complex on WikiText-103, real better on
TinyStories. Prior work: complex-valued / unitary RNNs (Arjovsky 2016, Wisdom 2016), complex Transformers; our v11 complex PAM.
