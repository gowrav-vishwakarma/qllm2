# EUREKA log: shared insights between the qllm2 sessions

Both sessions (v13_sempty "Qllm v13" = maturity, v14 "Qllm v14" = new design)
are building one novel O(1) architecture. Any insight that could change the
other's design goes here AND is sent at once by message, first line starting
`EUREKA:` (user rule, 2026-10-30). One entry per insight, newest first:
what we saw (with the run hash / log), what it really says, and what it
could mean for the other session. Read it before choosing a next step.

## 2026-10-30

- **v13 (after a quiet step back): write with hindsight, at the end of a
  meaning unit, not at every token.** Common thread of today's failures:
  PAM files every token the moment it arrives, keyed by whatever that token
  is ("is"), and never revisits. Minds store after the fact, once they know
  what the unit was ("truck → silver" is only a unit once "silver" lands).
  Idea: a tiny buffer of the last few tokens; at a boundary (comma, period,
  surprise drop) bind the finished unit and file it under what it is ABOUT
  (its content token), not under the last token. O(1); kills template-key
  collisions at the source; pands-like (read a unit, then bundle it). Also:
  all-or-nothing circuit formation looks like a phase transition, and those
  are crossed reliably by ANNEALING (start the read soft, sharpen it over
  training), a schedule-level twin of Q1. *For v14:* a write-side unit
  that complements N7/N9 (read side) and N8 (gate).
- **user + v13: "reason first, then know where to look"; two lookups
  and rules as phase continuation (vision, design ideas).** (1) *Analytic
  continuation* (the zeta-function idea): an analytic function known on a
  patch is fixed everywhere. For a mind: give ordered things phase
  coordinates (letters on a circle, numbers on several circles of
  different periods, multiplication via log-phase), so many rules become
  one rotation; inferring a rule = reading an angle from the examples;
  applying it to an unseen value = exact continuation, not recall (echo
  already infers held-out shifts, B2). (2) *Built-in RAG, pands-style*
  (point first, then speak; the user's pands v1 forms an essence
  direction before speaking): reason → form the query ("I need truck's
  label") → look in TWO memories, a knowledge store learned from all
  training data and the conversation memory (N7/N9) → speak toward the
  intent. Today PAM looks up with whatever key the current token gives
  ("is"), which is the label_category failure. O(1) if both are addressed
  by computed phases/hashes, not searched. Prior art for large learned
  stores: product-key memory, memory layers; the new part is reasoned
  addressing with content phases. (3) *Learns faster?* Holdout at 150M:
  PAM 3.905 vs T++ 3.981 (faster), the gap closes to −0.019 by 600M;
  Mamba-2 is ahead at every point (−0.02 to −0.035). *For v14:* (1) and
  (2) are new mechanisms; (2) is the third memory + planner of the
  two-memory design.
- **user + v13: engineer the start the lucky seeds had, instead of
  trusting the seed (Q1, `e89b7d2`).** In-context circuits form per seed,
  all or nothing (label_category: tied echo 4/4 vs echo 0/5; held-out
  shifts: echo-family 3 of 10, T++ 0 of 4). User: "it's what that seed
  did; can we initialise so it works for all seeds?" Clue: a q/k-aligned
  start forms the circuit; the permanent tie (E2) cost recall. Q1 = start
  tied (W_q = W_k at init), train free. Stage 1 = formation rate over 5
  seeds on v14's bench; stage 2 = 600M. *For v14:* the same lens fits
  every new unit: report the formation rate over seeds, not one seed;
  and an init that sets the intended circuit up (like v13's
  identity-at-init) may matter as much as the unit's design.
- **v14: the O(1) body can infer held-out rule parameters COMPLETELY, but it
  forms the circuit per seed, all or nothing** (B2 state sweep, `dd5b174`:
  held-out letter shifts 1.000 / 0.997 in 2 of 6 runs, 0.92 in a third;
  over all echo-family B2 runs 3 of 10 ≥ 0.9; T++ 0 of 4, ≤ 0.03). The
  state-size "bottleneck causes abstraction" hypothesis is NOT supported
  (the two seeds order the sizes oppositely). *For v13:* the open lever is
  what makes the circuit form reliably (data order, curriculum, a critical
  period, like your two-stage effect); single runs of rule probes need
  several seeds before any recipe or scale conclusion.

- **v13: past the trained context, PAM keeps the label SET, not the
  mapping; no model's learning holds past 2048** (LCL, `7707c8c`).
  label_category with the 8 examples spread to 4096 / 8192 tokens: PAM
  0.25 / 0.20 = in-context chance (it knows which labels appeared, not which
  item gets which); T++ 0.12 / 0.13 = candidate chance (lost the context).
  Pre-registered "PAM beats T++ by ≥ 0.10 at 4096 and 8192": NOT met (+0.13
  / +0.07). Rule tasks are at the floor for every model at every length.
  *For v14:* score the LCL bar against IN-CONTEXT chance (0.25 at 8
  shots), not candidate chance; a unit only "holds past 2048" if the
  mapping survives. The collision fix (N7 / N9) is what should lift it.
- **v13 + user: the nested version of the 2×O(1) memory: phase finds the
  region, SETTLING picks the best (design idea, refines the entry below).**
  User: the second O(1) should live inside the first ("first get the
  phase, then another memory learned in that phase picks the best"), and
  maybe be something altogether different. Two facts shape it: (1) more
  phase / more state = more address bits, but our collisions come from
  look-alike keys, not too few bits; (2) two plain linear memories read in
  sequence give the SAME score as one (forward Σ(e·v)(k·q) = backward
  Σ(q·k)(v·e)), so a second store cannot pick. Picking needs a nonlinear,
  competitive step. *Design:* step 1 = content-phase keys (each stored item
  its own valley); step 2 = a fixed 2–3 rounds of SETTLING: pull the query
  toward the nearest stored key (Hopfield-style relaxation on the key
  autocorrelation, with a sharpening nonlinearity) so close-but-wrong
  items lose instead of blending; then read the value. Constant cost; no
  attention cache, no state-in-state. Limit: a settling memory holds only
  a few dozen clean valleys per head, so writes need the N8 gate (store on
  surprise / maturity). Nature: systems relax into the nearest stable
  state; hippocampal theta-gamma nesting (slow phase = context, fast cycles
  = items). *For v14:* N9 = gate (N8) + content phase + settling; the
  two-bank vote is the fallback. Build in stages: phase keys alone (key
  cosine + label_exact), then add settling (5-pairs-back accuracy), then
  600M with the same bars.
- **v14: REPLICATED: the O(1) echo body infers unseen rule parameters where a
  same-size transformer only selects** (B2 run 2, `9e0a3fd`, 3 seed pairs:
  held-out letter shifts echo 0.14 / 0.26 / 0.34, rising with shots, vs
  T++ ≤ 0.03; d mean +0.230 ≥ 2·SD). First replicated bar-3 result in our
  favour (27M). *For v13:* worth a 600M check with B2 docs in the mix, inside
  and past the trained context (your LCL spread flag covers it).

- **v13 + user: a two-step O(1) memory, "align, then pick" (design idea,
  not yet tested).** From the key-collision diagnosis and the user's "2×O(1):
  first strong alignment, then pick the best one". *Step 1, align by
  phase:* rotate each memory key/query by phases computed from the last
  few tokens (learned angles per token, a few frequency bands; function
  words can learn ≈ 0). "truck is" and "red is" then point different
  ways (no template collision); a repeat of "truck is" aligns exactly;
  non-matches cancel by interference instead of adding blur; learned
  angles let car ≈ truck (label_category). Content-relative, not
  positional, so it cannot run out of range past the trained length.
  *Step 2, pick by agreement:* two such memories with independent phase
  banks; combine their votes as a product (AND): collisions are random per
  bank, the true value is common (count-min-sketch logic: error falls
  fast with each independent copy). Cost 2×O(1). *For v14:* fits the
  two-memory design as the learned store's alignment + the "body chooses"
  step (N7 exact store = the hard-hash limit of step 1). *Bars:*
  label_exact ≥ T++ (0.88), label_category ≥ 0.73, accuracy 5 pairs back ≥
  0.70, LCL past 2048, recall/holdout not worse. *For v13:* g1c (short
  conv = additive keys-see-previous-tokens) tests step 1's premise now.
- **v13: PAM's in-context lookup dies of key collisions, not distance or
  capacity** (`label_diag`, `bc7d57f`). Exact copy ("... truck is silver
  ... truck is" → silver): PAM 0.50–0.62, Mamba-2 0.62–0.72, T++ 0.78–0.88.
  3.3× the state does not help; 3B tokens do not help. Accuracy falls with
  the number of similar pairs in between (0.72 → 0.29 over 5 pairs, ~20
  tokens), while 1,000 tokens of unrelated filler barely hurt (0.44 →
  0.37). Echo keys of different "X is Y" pairs have cosine 0.67–0.72 (vs
  0.31 between random positions): every label is filed under the same
  "<noun> is" key, and the delta erase makes the last write win (45 % of
  wrong answers = the most recent label; T++ 23 %). *For v14:* keys must
  carry the content token, not the template: N7's hash of "truck is" is
  collision-free by construction; any learned memory needs a key that
  sees the previous tokens (Mamba-2's k = 4 conv likely explains its
  0.72), or centred keys. *For v13:* testing G1 short conv (`g1c`).
- **v14 (on v13's label_exact): format-free exact copy is N7's target.**
  PAM's copy skill is format-bound (recall probe 0.94, but label_exact
  0.54–0.73 vs T++ 0.86–0.92). An in-context n-gram store copies
  "truck is -> silver" in any format, never fades, O(1). *For v13:* the
  data lever (generic repeated-text / ICL data) and N7 are complementary;
  label_category (fuzzy match) stays a separate problem.

- **v13: label_category is not an attention-only skill; Mamba-2 (O(1))
  does it.** 600M, same two-stage recipe (`685de4e`): Mamba-2 label_category
  8-shot 0.55, learning mean 0.227; PAM M2 0.41 / 0.185; T++ 0.73. So a
  fixed-size state CAN learn this fuzzy in-context lookup; PAM's memory read
  does it worse. PAM's edge over Mamba-2 is recall past 2048 (0.72 vs
  0.46) at ~1/11 the state (1.2 vs 13.4 MB); Mamba-2 leads LM by 0.010.
  *For v14:* the gap is in how PAM reads (sharpness / key geometry), not in
  O(1) itself; Mamba-2's input-dependent decay + gating is the reference
  that works. label_category is a SEMANTIC match (new item of a seen
  category), so an exact n-gram store (N7) alone will not close it.
- **v13: the label_category lift depends on the body, not only the
  schedule.** The two-stage schedule lifts 8-shot label_category on the M1
  body (0.29 → 0.55, C1-ctl `7ae6f1f`) but not on M2 with shared reads
  (c2m2 0.41, c1m2 0.38). M2 is better at LM and recall, worse at this
  lookup. Single seeds, so treat it as a lead. *For v14:* a body that reads one
  writer's store from many blocks may dilute the in-context lookup
  circuit; in the two-memory design, give the exact store its own reader.
- **v13: reasoning data lifts learning, then it fades.** Right after the
  aptitude stage, models post their best learning-probe means (Transformer++
  tfc1a 0.229 with number_step 0.083 / reverse3 0.063; PAM c1m2a 0.195),
  and it fades during the web-heavy main stage (tfc1b 0.209, c1m2b 0.187).
  *For v13:* test rehearsal (a small reasoning share kept in stage B).
  *For v14:* judge a learning unit after the same data exposure; what the
  model saw last moves these probes as much as the unit does.
- **v13: Transformer++ gains WikiText from aptitude data, PAM does not**
  (T++ 66.2 → 63.8, PAM 65.6 → 65.7; tfc1 / c1m2 vs tfs2 / c2m2), while PAM
  wins the web holdout (−0.033). Encyclopedic text rewards exact recall of
  stored facts, which fits v14's lookup/abstraction split. *For v14:* an
  exact store (N7) may be where PAM's WikiText gap closes; check WikiText
  as well as holdout when N7 reaches 600M.
- **v14: lookup and abstraction pull apart.** Attention wins at exact
  lookup (600M label_category: T++ 0.73 vs PAM 0.41); the O(1) body wins at
  carrying a rule (B2 run 1, `bf59754`: held-out letter shifts, echo 0.33
  rising 0.20 -> 0.47 with shots, T++ 0.02 = one memorised table per shift).
  *For v13:* judge PAM on rule tasks and past-context learning, where it
  has an edge, not only on lookup probes. *For v14:* stop patching lookup
  into echo; give lookup to a separate exact store (N7).
- **v14: the bottleneck may cause the abstraction (hypothesis, under test).**
  Attention can keep every example, so a table per parameter is cheap; a
  fixed-size state cannot, so it is pushed to keep the rule. Test running:
  B2 state sweep (`dd5b174`, memory 12k / 25k / 49k at equal params).
  *For v13:* if it holds, a bigger state (3B, shared reads, wider heads)
  could trade abstraction for recall; worth a learning-probe check at 3B.
- **v14: symmetric addressing kills recall.** Tied echo (q = k) formed the
  similarity circuit in 4/4 small seeds but at 600M lost recall (0.825 ->
  0.602, E2). Recall is asymmetric ("what came AFTER X"); similarity is
  symmetric. A memory needs both, separately. *For v13:* keep q != k in
  any memory that serves recall.
- **v14: echo's numbers degrade with more shots** (B2 step_wide_new 0.98 /
  1.00 / 0.72 at 2 / 4 / 8 shots; T++ 0.97 at 8). Likely state saturation
  over long numeric runs. *For v13:* a capacity / precision clue for the
  notebooks at long context.
- **v14: micro-bench wins are leads, not evidence.** B0 broke even for
  attention (answer-only supervision); tied echo's clean micro win reversed
  at 600M. Micro-benches need dense LM-style supervision, and every win goes
  to seed pairs, then 600M, inside and past the trained length.
- **v13: the two-stage schedule helps only PAM** (tfs2: T++ gets nothing;
  c2m2 / C1-ctl gain recall). Reading: a critical period, where the
  notebooks need mature representations before they learn what to store.
  *Handed to v14 as N8:* writes gated by the model's own maturity.
- **v13: PAM's clearest edge is memory past the trained length** (two-stage
  M1 recall 0.60 vs T++ 0.15 past 2048). *For v14:* every new unit is
  judged inside and beyond the trained context (LCL probe, `cf2d7ec`).
