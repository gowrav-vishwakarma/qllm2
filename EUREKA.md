# EUREKA log: shared insights between the qllm2 sessions

Both sessions (v13_sempty "Qllm v13" = maturity, v14 "Qllm v14" = new design)
are building one novel O(1) architecture. Any insight that could change the
other's design goes here AND is sent at once by message, first line starting
`EUREKA:` (user rule, 2026-10-30). One entry per insight, newest first:
what we saw (with the run hash / log), what it really says, and what it
could mean for the other session. Read it before choosing a next step.

## 2026-10-30

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
