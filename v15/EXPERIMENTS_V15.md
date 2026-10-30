# v15 lab notebook — Grove

## 0. Where it came from (the meditation, 2026-10-31 ~00:30 IST)

The user asked for a new species, not a variant: rethink the neuron, think
beyond what exists, do not reuse the words of the brief as the design. The
first attempt (prime-period "wheels" for time and numbers, exact rule
inference by rotation) was dropped within the hour: it was phases again and
it extended PAM. One lesson from it is kept: exactness and range-freedom come
from structure that is *built in*, not learned.

Then the step back, from far away and from inside:

- **From far away.** Every machine that works today is a filter: attention,
  Mamba, PAM are linear in their state and nonlinear only pointwise. Every
  v13/v14 win at micro was about *choosing* (echo, phase keys), and every
  failure was the body's inability to choose: linear reads that cannot pick
  one of eight, keys that collide, last write wins, circuits that form by
  seed luck, decay constants that never move the horizon (R1–R4, C1 all
  failed).
- **From inside nature.** Wherever biology computes, it does so by
  **competition for a conserved resource**: neurons for growth factors,
  synapses for space, antibodies for antigen, species for niches, neural
  populations for the global workspace. Selection is nature's nonlinearity.
  And in-context learning *is* selection: a posterior over latent rules
  (Xie et al. 2021 explain ICL as Bayesian inference over concepts). Our
  architectures have no conserved quantity. Nothing is at stake. Nothing has
  to earn its place.
- **What surfaced.** Give the model an economy. A unit is an organism with a
  share it must earn, a root it grows from what it wins, a fruit it yields.
  Forgetting is being outcompeted, not decaying. New things are born into the
  slots of the dead. Growth has seasons.

That is v15. The rest of this file makes it precise and testable.

**Isolation (user, 2026-10-31).** v15 thinks on its own. v13/v14 are consulted
only for plumbing (trainer, benches) and baseline numbers, never for ideas.

## 1. The unit and the organ

Model width d. A layer has G groves; a grove has K trees; roots and fruits
live in R^n. Shapes in comments are [B, T, G, K, n].

**Innate traits (learned by backprop, frozen at inference):** initial root
r_i^0, initial fruit u_i^0, fitness bias b_i, retention λ_g, growth rate α_g,
temperature τ_g, the soil projection W_x (what a tree is matched on), the sap
projection W_y (what it stores), the out projection W_o (zero-init, so the
block starts as identity), and a depthwise causal conv (k = 4) on the soil.

**Acquired state (per sequence, O(1)):** evidence ℓ_i, root r_i, fruit u_i,
last share m_i. Per grove K(2n + 2) numbers.

For token t with x_t = W_x h_t (after the conv) and y_t = W_y h_t:

```
fitness    f_i(t) = τ · cos(x_t, r_i) + b_i                       [B,G,K]
evidence   ℓ_i(t) = λ ℓ_i(t−1) + f_i(t)
share      m(t)   = softmax_i ℓ(t)             (Σ_i m_i = 1: a simplex)
read       o(t)   = Σ_i m_i(t) u_i             (a mixture, never a superposition)
growth     every season (C tokens), for each tree:
           g_i  = Σ_t m_i(t)                       mass it won this season
           x̄_i  = Σ_t m_i(t) x_t / g_i              mean of what it won
           r_i ← r_i + (1 − e^{−α g_i}) (x̄_i − r_i)
           u_i ← u_i + (1 − e^{−α g'_i}) (ȳ_i − u_i)   with weights m_i(t−1) and y_t
                                                  (what FOLLOWED what it won: succession)
```

With λ = 1 the share is the exact Bayesian posterior over K hypotheses whose
log-likelihoods are the fitnesses; with λ < 1 a forgetful posterior. λ is
initialised spread across the groves (0.1 → 0.9): some groves live in the
present (lookup), some accumulate (rule / topic selection). **Forgetting is
relative**: nothing on the state decays; a tree loses share only when another
gains. The growth step is in [0, 1) and never overshoots: a tree that wins one
token with α = 2 moves 86 % of the way to it.

**Seasons** make it a GPU model. Inside a season every read uses the roots and
fruits of the season's start, so a season is three batched ops (a C×C
retention matrix per grove for the evidence scan, one softmax, one matmul);
the growth is one more at the season's end. Decode keeps a C-token buffer and
grows every C tokens, so prefill == decode exactly (selftest: 1e-15). A fact
written in a season is readable from the next one; the conv covers the first
few tokens.

**Cost.** Per token per layer: G·K·n·3 MACs (G 8, K 32, n 64: 49k), far below
the MLP (d 768: ~3.6M). State per layer at those sizes: 33k numbers (132 KB);
16 layers ≈ 2 MB, between PAM (1.2–3.7 MB) and Mamba-2 (13.4 MB). Every
block can carry a grove.

## 2. What is new, honestly

| part | what Grove does | closest known thing |
|---|---|---|
| the recurrence is selection | state = accumulated evidence per unit; output weights = softmax of it; forgetting by competition, no state decay | Bayesian ICL theory (Xie 2021, a description, not an architecture); Hedge / replicator dynamics (NeuRD 2019, RL); softmax gating in MoE (per token, no accumulation) |
| roots grow in context | competitive learning in the forward pass, mass-weighted, seasonal | online k-means centroids in a sequence layer: OVQ-attention (Alonso, Figliolia, Millidge 2026) is the nearest neighbour; competitive learning (Rumelhart & Zipser 1985); prototype nets (few-shot) |
| fruits by succession | a unit stores what followed what it won | echo (v13) stores succession in a matrix; here it is per unit, selected not superposed |
| seasons | growth only at fixed boundaries, so parallel == decode | chunked scans (Mamba-2) are linear inside the chunk; here the chunk boundary is a nonlinear event |
| the unit as organism | share + root + fruit + innate traits; a layer is an ecology | HTM minicolumns (Hawkins) and LV / competitive networks in neuroscience; not used as an LM organ |

Not novel and not claimed: RMSNorm, SwiGLU MLP, tied head, depthwise conv.

## 3. What Grove is built to do (and what each bar tests)

1. **O(1).** By construction (G·K·(2n+2) per layer).
2. **LM quality.** Unknown. The economy is cheap, so every block gets one and
   the layout can scale K. The honest risk: a mixture over K prototypes may
   carry less of the context than a 13 MB linear filter. G2/G3 decide.
3. **Learning beyond recall.** Two routes, both native: (a) a grove with high
   λ is a posterior over K innate hypotheses (rule *selection* among learned
   functions, like PFC rule maintenance); (b) prototypes generalise: a new
   item of a seen category lands on the category's tree and reads its fruit
   (label_category is the prototype task). Held-out rule *parameters* (B2)
   are not promised by the mechanism; they are measured.
4. **Past the trained length.** Nothing is positional and nothing decays by
   time; roots, fruits and shares just keep evolving. Measured in G1/G2.

## 4. Pre-registered experiments

**Stages** (v14's ladder): selftest → micro (4090, minutes, seeds) → 150M
same-batch LM → 600M (hook into the v13 trainer) → ≥ 1.5B slope test (6000).
Control arms are the current references: echo (v13 body) and `tpp`
(Transformer++) in the v14 micro-bench; T++ 600M (3.4885) and the v13
reference c1m2 (3.4473) at 600M. Report params, tokens, state bytes, tok/s with
every number. Judge LM effects at ≥ 1.5B before belief (EUREKA 2026-10-30).

### G0 — selftest (done, 2026-10-31)
Parity season == decode (1e-15, conv on/off, continuation from state);
shares a simplex; gradients reach every innate trait; state O(1); toy
in-context lookup (8 redrawn key→value pairs per doc, query one): **0.98
held-out, chance 0.125, 400 steps, 20 s CPU**, 2 layers, G 4, K 16, n 24,
season 8. First version (dot-product fitness, step αg/(1+αg), λ 0.9) stayed
at chance: competition must be sharp (cosine × temperature) and growth
decisive (1 − e^{−αg}). Written down so it is not re-learned.

### G1 — micro: does the economy choose? (next; 4090; one small job)
Arm `grove` in `v14/microbench.py` (a V15 model class with the bench's
interface), size-matched to echo (27.6M), vs `echo` and `tpp`, on B1 run 4's
training set (rules + label_category + label_exact + two_hop), 3 seed pairs,
then the B2 set (shift_wide / step_wide / affine_map and `_new`).
Bars, written before the run:
- **L (lookup):** label_exact ≥ tpp (0.86–0.92 at 600M scale; micro: ≥ echo + 0.15 in ≥ 2 of 3 pairs).
- **P (prototype):** label_category formed (≥ 0.9) in ≥ 2 of 3 pairs (echo 1/5, N9a 5/5 on F1).
- **R (rule):** shift_wide_new mean d vs tpp ≥ +0.15 (echo's replicated edge is +0.23); not required to pass G1, recorded.
- **Guard:** train loss on the rule tasks within 0.1 nat of echo (the organ must not hurt the plain LM path).
Pass = L and P. Fail = neither in 3 pairs → the design goes back to §0 (not to tuning).

### G2 — 150M same-batch LM (after G1 passes; 4090)
GroveLM param-matched to M1 (102.96M: dim 768, 12 blocks, G 12, K 32, n 64,
season 32, MLP 1536; `v15/hook.py` DEFAULT_SPEC), v13 trainer data and
schedule (`--arch grove`, launcher `ARCH=grove`), holdout at 150M tokens
(step 4069 of the r600 protocol) vs M1 3.9225 and echo 3.9516. Bar: within
+0.05 of M1 (a new body at its first LM run) and tok/s ≥ 0.8× M1. Pass → G3.
Plumbing verified 2026-10-31 (CPU, no GPU): 4-step synthetic run with
checkpoints; greedy `generate` through `step` == full forward; step logits
vs season path 5e-16; checkpoint config round-trips. Launch (clean tree,
tmux, 4090 when a whole-card slot is free, or the 6000):
`TAG=v15g2 ARCH=grove CHRONO=0 OUT_GATE=0 DELTA=0 TARGET_TOKENS=600000000
GRAD_CKPT=1 bash v13_sempty/tmp_pretrain_mix.sh` (the M1 control ran on the
6000 at B 18 × T 2048, SEED 42; GEN_EVERY may stay on: Grove decodes).

### G3 — 600M on the 6000, then slope
r600 protocol, two stages, vs T++ 600M 3.4885 / c1m2 3.4473 / Mamba-2 3.4349;
recall 63-cell, learning probe, LCL past 2048. Bar: holdout ≤ T++ + 0.03 with
any of recall / learning / past-2048 ahead. Then 600M → 1.5B slope (EUREKA:
pick by slope, not by the 600M number).

### Later, only after G1–G2 (design, not pre-registered)
- **Births:** a dead tree's slot (share ≈ 0 for long) is re-seeded by the
  current input, so capacity is K *live* items rather than K prototypes.
- **Operator fruit:** a tree's fruit as a low-rank map (apply to the query)
  for rules; a delta-fruit (mean of y − x it won) for analogy.
- **Self-monitoring input:** the previous token's surprise as a scalar input
  to every grove (a neuromodulator); free at train and decode.
- **sempyt port and a fused season kernel** once the organ earns a 6000 run.

## 5. Roads not taken (so they are not retaken)
- Prime-period wheels / CRT phases for time and numbers (2026-10-31, 20
  minutes, code never committed): exact by construction and range-free, but
  it is phases and it extends PAM. Dropped at the user's direction. If a
  future design wants exact arithmetic, this is where to look (Z_p is a
  field; +k and ×k are rotations in two coordinates; grokking work shows nets
  learn such features on their own).
