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

### Reference rows and scorers (facts handed by Lead, Nov 2 22:44 IST; no mechanism)
Same data, ~102M params, 600M-token tier:

| body | recipe | holdout NLL | WikiText | recall 63-cell | source |
|---|---|---|---|---|---|
| Transformer++ tfc1b | aptitude | 3.4802 | 63.83 | 0.648 | `checkpoints_v13_sempty/tfc1b_main_b788731` |
| Mamba-2 mbs2b | two-stage r600 | 3.4393 | 65.04 | 0.762 (>2048: 0.46) | `checkpoints_v13_sempty/mbs2b_main_685de4e` |
| T++ b2t / Mamba-2 b2m | b2 mix (rule docs 3 %) | 3.5572 / 3.4754 | – | – | v13 lineage rows "Baselines", "B2-in-training pair" |

Rules at distance (s8, the learning probe `scripts/run_learning_probe.py
--suite b2 --seed 7000 --trials 400 --spread 0,2048,4096`, b2 `_new` tasks):
T++ 0.134, Mamba-2 0.10–0.116 (log `logs/v13_sempty_s8_driver_38e3c29_20261031_0100.log`).
Known-benchmark MC mean at 600M (kb1): T++ 0.388, Mamba-2 0.392
(`logs/kb1/v13_sempty_kb1_table_ccf187c.md`). Bench on a trainer
checkpoint: `.venv/bin/python v16/bench.py --ckpt <dir>/best_model.pt --arch
<arch> --task kv --needles 8 --keys synthetic --lengths 2048,8192 --docs 100
--seed 18000 --out logs/<name>.jsonl` (tasks kv kv1 kvupd hop2 para tool neg
instr natq natqneg syn synneg; `--device cpu` works); a new arch needs a
loader branch in `scripts/run_memory_behavioral.py::_load_v13_sempty` and a
Reader in `v16/bench.py` (both small; to do for Grove before G2 is scored).
Recipe lessons (facts): screen at ≥ 600M tokens (100M screens cannot pick a
learning rate for a recurrent body); answer-weighted synthetic docs ≤ 3–4 %
of the loss weight; packed unrelated docs teach forgetting unless the state
resets at separators; fine-tuning at 8k/32k did not grow reach; every
pre-registration names a bar and a closest prior work.

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
  continuation; toy lookup ≥ 0.5; O(1) state. **PASS** (`b599a56`): parity
  1e-15 with 39–41 births; toy 1.000 at 400 steps (Grove I 0.750).
- GII-1 toys (CPU): big-vocab 4096 keys, season 32, lr 3e-4, 1200 steps:
  ≥ 0.95 (Grove I: 0.955). **PASS: 1.000**; lr 3e-3 400 steps 1.000 (Grove
  I 0.062); 31 keys season 8 1.000. Fast strict grove: same-key overlap 1.00
  vs 0.00 (an exact instance cache); loose slow groves 0.5–0.9 vs 0.2–0.5.

**What the toys taught on the way (05:50–06:20 IST), all found by the
decode-parity test and the birth/landing trace, none by guessing:** (1) a
novel token must be eaten by its newborn only; landing on the best old tree
too corrupted it (toy 0.055). (2) Growth of 86 %/token (Grove I's α for
soft masses) destroyed instances; a root is now the running mean of what the
tree ate with `use` as the count, and α is gone. (3) The conv's bias made
every token look alike (cos 0.8–0.9), so nothing was novel and a repeat was
not recognised; the conv has no bias and starts as identity on the current
token. (4) λ and θ were both spread in grove order, so no grove was fast and
strict; the spreads are now paired the other way (fast = strict cache, slow
= loose prototype). In a slow grove (λ 0.9) the accumulated fitness noise of
other trees (a random walk) beats a perfect match, so slow groves are context
selectors, not lookups: a division of labour, not a bug.
- GII-2 = G1 again (same six pairs, same bars L, P, R, guard; `grove` arm
  at the new hash). **Running since 06:22 IST 2026-10-31 at `b599a56`.** Prediction, written before the run: label_exact 2-shot
  > 0.5 in ≥ 2 pairs (binding, not just the label set), label_category
  unknown (prototype formation now depends on the soil clustering tokens
  under θ), guard within 0.1 nat. If label_exact stays at echo's 0.50, the
  binding by reign (λ) is not happening and the conv/soil would have to
  carry the key: that is a finding, not a knob.

### 6.1 GII-2 pair 1 and what followed (08:00–09:30 IST 2026-10-31)

**GII-2 pair 1 (`b599a56`, seed 21): FAIL, and worse than Grove I on
rules.** label_exact 2/4/8-shot 0.39/0.22/0.12 (echo 0.50/0.26/0.12),
label_category 0.40/0.49/0.25 (0.50/0.49/0.25); letter_shift 8-shot 0.69
(Grove I 0.99, echo 1.00), number_step 0.59 (0.86), reverse3 0.12 (1.00);
train loss 1.53 (Grove I 0.61, echo 0.38); 933 ms/step. The chain was
stopped after pair 1 (the user asked v15 to leave the GPUs to v13/v14; two
more pairs could not change the verdict). Logs
`logs/v14_v15gii2p1_b599a56_*`. The probe of the saved grove showed the
mechanism: in layers 1–3 every token is born (contextual states are never
within cos 0.95 of an earlier self), so a query was "novel", eaten by its
newborn, and read that newborn's empty fruit. **Reading and eating must not
be the same winner**: a seed may found a tree, but what you see is the
nearest existing canopy.

**A rule toy for the CPU** (`v15/tmp/toy_rule.py`): infer a letter shift
from 8 in-context pairs and apply it to an unseen letter (chance 0.038;
2 layers, 4 groves × 16 trees, season 8, 4000 steps). It reproduces the
bench's rule result and separates the variants (one seed each unless noted;
Grove I reference extracted from `abe4c89` as `v15/tmp/grove_I.py`):

| variant | read | eat / credits / growth | births | rule toy | lookup toy (4096 keys, lr 3e-4) |
|---|---|---|---|---|---|
| Grove I (`abe4c89`) | soft | soft, 1−e^{−αg} | none | 0.839 (identity conv, seeds 2–4: **0.907, 0.661, 0.935**) | 0.955 |
| Grove II (`b599a56`) | hard (ST) | hard, running mean | yes | 0.255 | 1.000 |
| Grove III (`4b06ad2`) | soft | hard, running mean | yes / off | 0.575 / 0.090 | 1.000 |
| Grove IV-b (`8e6214b`, record) | soft | soft, 1−e^{−αg} | yes / off | 0.072, 0.081, 0.488 (seeds 2–4) / 0.655 | 1.000 |
| **Grove IV (`4a33187`)** | soft | soft, running mean | yes / off | **0.638, 0.093, 0.947** (seeds 2–4) / 0.481 (seed 2) | 1.000 |
| **Grove IV + sapling mask (`dcd5c86`)** | soft, fruit-bearing trees only | soft, running mean | yes | **0.846, 0.853, 0.626** (seeds 2–4; no collapse) | 1.000 |

Readings: (1) the soft read carries rules (II → III: 0.26 → 0.58); (2) soft
eating/growth carries more (III → IV: 0.58 → 0.64, and IV-b without births
0.66 vs III without births 0.09); (3) α-growth with births is unstable
(2 of 3 seeds collapse) where the running mean is not; (4) births help
rules in IV (0.64 vs 0.48) and lookups everywhere (1.000); (5) the toy's
seed noise is large (Grove I 0.66–0.94), so single-seed differences under
~0.15 mean nothing. (6) **Grove IV is bimodal**: seeds 2–4 give 0.64 / 0.09
/ 0.95, i.e. it either learns the rule as well as Grove I or sits on a dead
plateau (loss ≈ 2.55); Grove I never collapses. The loss trace of the
collapsed seed sits at 2.55–2.60 from step 250 to the end: a dead plateau,
not a divergence. Cause: at every novel token (every token, in the strict
groves) the read included the newborn's own empty fruit (fruit0_born), a
constant injected into the residual that training never escaped. **A
sapling bears no fruit**: the read now excludes newborns nobody has credited
yet (`dcd5c86`); seeds 2–4 give 0.846 / 0.853 / 0.626 with no collapse, lookup
1.000. This is the bench candidate (GIV-2 = the G1 pairs again, same bars).

### 6.2 GIV-2: Grove IV on the bench (pairs 1–2 ran Oct 31 13:00–18:29 IST; read Nov 2 22:45 IST)

Logs `logs/v14_v15giv2p{1,2}_dcd5c86_*.{log,json}` (the session that queued
them had exited before they landed). Grove IV (`dcd5c86`) vs echo, 27.67M vs
27.61M, 6000 steps × B16:

| pair | label_exact 2/4/8 (grove vs echo) | label_category 2/4/8 | letter_shift 8 | number_step 8 | reverse3 8 | train loss | ms/step |
|---|---|---|---|---|---|---|---|
| p1 (seed 21) | 0.28/0.20/0.13 vs 0.50/0.26/0.12 | **0.75/0.90/0.83 vs 0.50/0.49/0.25** | 0.70 vs 1.00 | 0.67 vs 1.00 | 0.95 vs 1.00 | **0.371 vs 0.377** | 1230 vs 653* |
| p2 (seed 22) | 0.30/0.23/0.12 vs 0.48/0.25/0.12 | 0.35/0.49/0.28 vs 0.48/0.48/0.26 | 0.71 vs 1.00 | 0.61 vs 1.00 | 0.94 vs 1.00 | 0.625 vs 0.462 | 1180 vs 169 |

(*shared card.) **Bars:** L (label_exact ≥ echo + 0.15 in ≥ 2 of 3) **FAIL**
after two misses. P (label_category ≥ 0.9 in ≥ 2 of 3): **split**, pair 1
met it (0.90 at 4-shot, +0.41; 8-shot 0.83 vs 0.25, +0.59: the first time
a Grove beat echo on anything, and it is prototype formation, the thing the
economy was built for), pair 2 did not (0.49). Guard: met in pair 1 (0.371
vs 0.377), missed in pair 2 (0.625 vs 0.462). reverse3 is back (0.94–0.95;
Grove II 0.12); letter_shift 0.70 is still below Grove I's 0.99.
**Why the pairs differ:** not a plateau this time. The training-loss
trajectories (grove / echo at steps 1000 / 3000 / 5000 / 6000): p1 1.55 /
1.12 / 0.36 / 0.37 vs 0.47 / 0.47 / – / 0.38; p2 1.89 / 1.36 / 0.70 / 0.63
vs 0.54 / 0.42 / – / 0.46. Grove IV learns slowly for ~3000 steps (the
economy warms up while the soil is untrained: births and placements churn)
and then drops fast; pair 1 reached echo's loss by step 5000, pair 2 was
still descending at 6000. The split on P and on the guard is the same
late convergence seen from two seeds. "A bit more training" is inside the
user's bar-2 tolerance; it is also a cost to state next to every number.
**Pair 3 (seed 23) runs now** (Nov 2 22:41 IST, 4090 handed to this lane)
to decide P as registered; a longer-training variant would be a new,
pre-registered row, not a re-roll. Alone on the card it trains at 33.9k
tok/s (Grove I alone 43.8k; the 12.5k of GII-2 was a shared card), so
Grove IV costs ~1.3× Grove I per step, not 3.5×.
**Probe of pair 1's saved grove (label_exact 8-shot, 40 prompts; Nov 2
22:50 IST):** this time the innate traits moved: τ rose in the strict
groves (8.2 → 13.9 across layers) and the newborn bias b_born grew to
0.34–0.41 in layers 1–2 (the body learned to trust newborns' evidence).
Births per token 0.35–0.67 in the strict groves, and the same-key overlap is
still 0.00 there: a repeated key in a new context is a new state, so exact
re-selection does not happen (bar L stays failed by mechanism, as in GII-2);
the loose groves (θ 0.35, 0.2) re-select at 0.64–1.00 for same and other
keys alike (prototype groves, no instance address). The label_category
probe needs its own prompt parser (not done).

## 5. Roads not taken (so they are not retaken)
- Prime-period wheels / CRT phases for time and numbers (2026-10-31, 20
  minutes, code never committed): exact by construction and range-free, but
  it is phases and it extends PAM. Dropped at the user's direction. If a
  future design wants exact arithmetic, this is where to look (Z_p is a
  field; +k and ×k are rotations in two coordinates; grokking work shows nets
  learn such features on their own).
