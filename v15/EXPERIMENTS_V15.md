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

**Seasons** make it a GPU model. Inside a season the *selection* uses the
roots of the season's start (a tree's identity is fixed for a season), so
the evidence scan is a C×C retention matrix per grove and one softmax. The
*fruit ripens daily* (added 2026-10-31 after G1 pair 1, see §4): the read at
t uses the fruit as it would be if the season had ended at t−1,

```
ripening   u_i(t) = u_i + (1 − e^{−α g_i(<t)}) (ȳ_i(<t) − u_i)
           g_i(<t) = Σ_{s<t} m_i(s−1),  ȳ_i(<t) = Σ_{s<t} m_i(s−1) y_s / g_i(<t)
read       o(t)   = Σ_i m_i(t) u_i(t)
```

which is, without materialising the partial sums, one strict-causal C×C
mixing matrix per grove, A[t,s] = Σ_i m_i(t)·step_i(t)/g_i(<t)·m_i(s−1),
applied to the season's y. At t = C the ripened fruit *is* the next season's
fruit, so growth and ripening are one rule seen at two cadences. Reads see a
fact the token after it is written; the strict s < t matters: with s ≤ t the
tree read at t is the one just credited with the query's own sap (evidence
persists across tokens), and the toy fails (0.16). Decode keeps a C-token
buffer: read, then credit, then grow every C tokens; prefill == decode
exactly (selftest: 1e-14).

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

### G1 — micro: does the economy choose? (4090; one small job)

**Pair 1 as built (`84b41d8`, seed 21, 2026-10-31 02:34 IST): FAIL, and
the reason is the design, not the seed.** Grove sat at chance on every rule
task (label_exact 0.15/0.14/0.12 at 2/4/8-shot, label_category
0.14/0.21/0.25, chance 0.125); echo on the same seed: label_exact
0.50/0.26/0.12, label_category 0.50/0.49/0.25, letter_shift 1.00. Train
loss was LM-like (3.31 → 2.92), so the body learned text, not the context.
Log: `logs/v14_v15g1p1_84b41d8_20261031_0146.{log,json}`.
Cause, verified on CPU (`v15/tmp/toy_season.py`, selftest's toy lookup with
the season as the knob; 400 steps, chance 0.125): season 8 → 0.988, 16 →
1.000, 32 → **0.062**. With season-start reads a fact seen fewer than C
tokens ago is invisible, and the bench's shots are ~4 tokens apart: an
8-shot prompt lives inside one season of 32. The toy passed only because its
16 tokens of pairs straddled two seasons of 8.
Fix = **ripening** (§1, commit `abe4c89`), a design change, not a tuning:
the read sees the current season's growth so far. Same toy after it (400 /
1200 steps): season 8 0.750 / 0.992, 16 0.953 / 0.984, 32 0.957 / **1.000**.
Pairs 2–6 of `84b41d8` were not run (they could not change the verdict; the
clean-tree guard was used to stop the queue, no job was killed).

**G1r = G1 rerun at `abe4c89` with ripening, same six pairs, same bars
(unchanged, below), 02:37–04:44 IST 2026-10-31. Verdict: FAIL on L and P
after two of the three rules pairs; pairs 3–6 were not run (they could not
change the verdict).** Logs `logs/v14_v15g1rp{1,2}_abe4c89_*.{log,json}`.

| pair | label_exact 2/4/8 (grove vs echo) | label_category 2/4/8 | letter_shift 8 | number_step 8 | train loss | ms/step |
|---|---|---|---|---|---|---|
| p1 (seed 21) | 0.34/0.22/0.12 vs 0.50/0.26/0.12 | 0.33/0.42/0.25 vs 0.50/0.49/0.25 | 0.99 vs 1.00 | 0.86 vs 1.00 | 0.61 vs 0.38 | 319 vs 176 |
| p2 (seed 22) | 0.35/0.21/0.10 vs 0.48/0.25/0.12 | 0.33/0.47/0.27 vs 0.48/0.48/0.26 | 0.97 vs 1.00 | 0.69 vs 1.00 | 0.76 vs 0.46 | 310 vs 411* |

(*echo p2 shared the card with v14's whole-card job.) Reading: ripening
turned Grove from blind (chance everywhere) into a rule learner (reverse3
1.00, letter_shift 0.97–0.99), but it is behind echo everywhere by a margin
that looks like slower learning (train loss 0.6–0.76 vs 0.38–0.46), and the
2-shot lookup numbers (0.34 vs 0.50) say echo knows the label *set* (0.50 =
one of two seen labels) while Grove only partly does; nobody binds at
8-shot (0.12). The L and P bars as written are not met. CPU toy with a
large key vocabulary (`v15/tmp/toy_bigvocab.py`, 8 pairs, season 32, 1200
steps): 31 keys 0.998, 512 keys 0.879, 4096 keys 0.846: the soft hash
degrades with key vocabulary but is not the whole story. 
**G1d — what the trained organ does (05:40 IST; `logs/v14_v15diag1_abe4c89_*`,
probe `v15/tmp/probe_shares.py`).** The grove arm of pair 1 re-trained with
saved weights, probed on 40 held-out label_exact prompts per shot count:

| | bench Grove (27.7M, 6000 steps, lr 3e-4) | big-vocab toy that works (4096 keys, lr 3e-4) |
|---|---|---|
| τ after training (init 10) | 9.8–10.6, every layer and grove | 13.4–13.6 (fast groves) |
| λ, α | unchanged from init | λ unchanged |
| share entropy, fast groves (ln K 3.47 / 2.77) | 2.0–2.7 | 0.4–0.7 |
| cos(m_query, m_same_key) vs other keys, 8-shot | **0.78 vs 0.79** | **1.00 vs 0.13** |
| slow groves (λ ≥ 0.74) | collapsed: entropy 0.1–0.3, overlap 1.0 for every key | collapsed too (grove 3: overlap 1.0 / 1.0) |

So the organ never formed addresses: a repeated key is not re-selected any
more than a different key. In the single-task toy the same organ learns a
near one-hot hash by itself (and at the bench's learning rate), so the
difference is the regime: with twenty tasks and an MLP path the LM loss
never paid the groves for separating keys, and the innate traits sat where
they were initialised. Nature does not learn sharpness by gradient descent;
lateral inhibition gives a winner by construction. The collapsed slow groves
are the dead-unit problem of competitive learning, the same thing seen from
the other side: no mechanism re-seeds a tree nobody selects.
**Reading for the redesign (step-back below, §6):** selection must *decide*
(a winner per grove, structural), and empty niches must be *colonised*
(a token no tree fits founds a tree). Both exist in ecology (competitive
exclusion; colonisation of empty niches); neither is in Grove today.

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


## 6. Grove II — competitive exclusion and colonisation (design, 2026-10-31 06:00 IST)

**The void, after G1d.** Grove I made selection *soft* and hoped backprop
would sharpen it (τ, roots, soil). In a single-task toy it did; in a
twenty-task bench it never did, and half the groves collapsed to one tree.
Ecology does not sharpen by gradient. Two of its laws are missing here:
*competitive exclusion* (one occupant per niche: a token lands on exactly one
tree) and *colonisation* (a token that fits no occupant founds a tree in an
empty niche). A third follows: *starvation* (a tree nobody feeds loses its
niche). With these, addresses are sharp by construction, dead trees are
recycled by construction, and instances are remembered exactly while
prototypes still form (a tree fed by many similar tokens grows toward their
mean). Every part is in the organism frame; none is a tuning of Grove I.

**Rules (per grove, K niches, season C).** State as before plus `use_i`.
```
fitness     f_i(t) = τ cos(x_t, r_i) + b_i                   dawn trees (K)
            f_s(t) = τ cos(x_t, x_s) + b_born                 trees born this season at s < t
evidence    ℓ(t)   = λ ℓ(t−1) + f(t)                          over K + born
landing     w(t)   = argmax ℓ(t)                              ONE tree eats the token
            m(t)   = onehot(w) + softmax(ℓ) − stopgrad(softmax(ℓ))   (straight-through)
read        o(t)   = ripe_w(t)                                the winner's ripening fruit
novelty     nov(t) = max_i cos(x_t, r_i) < θ_g                 no dawn root fits
birth       born(t)= nov(t) ∧ ¬∃ s<t: nov(s) ∧ cos(x_t, x_s) ≥ θ_g
                                                               (first of its kind this season)
            a born tree's root is x_s; its fruit starts at fruit0_born and is
            credited y_{s+1} (founding) and then whatever follows its reigns
season end  dawn trees: grow roots/fruits toward what they ate (hard credits)
            use_i ← ρ use_i + wins_i ;  born: use = wins + 1
            placement: born trees, in birth order, take the niches with the
            smallest use (≤ K per season; the rest are not kept)
```
θ_g is an innate per-grove threshold spread across groves (eager groves are
caches of recent distinct tokens; strict groves keep prototypes), ρ an innate
decay of use. τ now shapes only the surrogate gradient. λ keeps the reign: a
tree that just won keeps winning for a few tokens and collects what follows,
which is how a key binds to the value after it (and groves with different λ
bind at different lags).

**Parallel form.** Inside a season everything is C×C per grove: the cosine
matrix of the season's soil gives births (a strict-causal "first of its
kind" mask) and born-tree fitness; the evidence scan is the same retention
matrix; the landing is one argmax over K + C columns; the reads are the
ripening matrices of §1 with hard credits; placement is one scatter at the
season's end. Decode mirrors it with a per-season buffer of born trees.
Prefill == decode is the first test, as before.

**What is new, honestly.** Hard assignment with a straight-through gradient
is standard (VQ); online centroids in a sequence layer exist (OVQ-attention,
2026). New here: births into *starved* niches from the token itself,
competition on *accumulated evidence* (a reign, not a per-token match), the
succession fruit, and seasons. The whole is the claim, as in §2.

**Pre-registered tests (Grove II).**
- GII-0 selftest: season == decode to round-off incl. births, placement and
  continuation; toy lookup ≥ 0.5; O(1) state.
- GII-1 toys (CPU): big-vocab 4096 keys, season 32, lr 3e-4, 1200 steps:
  ≥ 0.95 (Grove I: 0.955). Birth rate per grove logged.
- GII-2 = G1 again (same six pairs, same bars L, P, R, guard; `grove` arm
  at the new hash). Prediction, written before the run: label_exact 2-shot
  > 0.5 in ≥ 2 pairs (binding, not just the label set), label_category
  unknown (prototype formation now depends on the soil clustering tokens
  under θ), guard within 0.1 nat. If label_exact stays at echo's 0.50, the
  binding by reign (λ) is not happening and the conv/soil would have to
  carry the key: that is a finding, not a knob.

## 5. Roads not taken (so they are not retaken)
- Prime-period wheels / CRT phases for time and numbers (2026-10-31, 20
  minutes, code never committed): exact by construction and range-free, but
  it is phases and it extends PAM. Dropped at the user's direction. If a
  future design wants exact arithmetic, this is where to look (Z_p is a
  field; +k and ×k are rotations in two coordinates; grokking work shows nets
  learn such features on their own).
