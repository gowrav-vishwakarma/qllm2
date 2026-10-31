# O(1) mind + growing library — the goal sheet (organiser, 2026-10-31)

User decision (2026-10-31, to v13 and v14): the O(1) body stays and is the MIND; the
demand that its state also be the LIBRARY is dropped. Knowledge lives in libraries:
a per-session library (the context, tool results) and a shared, growing library
(community-enriched; "skill = skill + lookup data"). Order: a pre-registered
benchmark first; the distributable model once it wins that table.

## Why (measured; EUREKA.md 2026-10-31)
- No fixed-size state holds facts at long range: PAM, Transformer++ and Mamba-2 all
  at the floor past ~8k tokens on the needle probe (`scripts/session_index_probe.py`).
- PAM's strength is rules, not facts: in-context rule learning 0.99/0.94 vs T++ 0.73
  (600M), held across 2k-8k of filler where T++ and Mamba-2 collapse; the label gap
  was data format (label pair PASS 0.934 vs 0.798).
- An exact organ outside the model is perfect at every length incl. two-hop; the
  model's own state at the salient word is a free address to ~128k, not beyond:
  the library key must be TRAINED, one row per event.

## Lanes
- **v13 = mind + benchmark + release.** The trained body (r3c/r3d), the exact organ
  in training (a4f), the mind half of the benchmark (rule-at-distance, instruction
  persistence, LM loss by token class), 3B runs, packaging. Owns the merged spec.
- **v14 = library.** Learned key at the salient token, one row per event, write
  gate (N12), planner/query former (N11), shared-library schema with provenance.
  Judged on the benchmark with v13's checkpoints.
- **Organiser = library-half harness + coordination.** `v16/bench.py` (was scripts/session_index_probe.py)
  grows into the library benchmark: needle, multi-needle (8, 64), latest-definition-
  wins, two-hop, at 32k / 128k / 400k / 1M; synthetic AND natural-text keys; >= 32
  docs per cell (target +-0.05); memory in MB per row (mind state + library index vs
  T++ KV cache at full context vs Mamba-2 state).
- Evol V1 (v15) is isolated: nothing from this plan is sent to it.

## Decision rules
- GPUs go only to runs on this path. "Make the notebook select" work (phase keys,
  salient phase) stops once the current runs report; a4f (organ) and N12 (write
  gate) continue because they are organs of the new design.
- r3c runs to its futility check (~15:00-16:00 IST) before anything on the 6000
  changes; its result is the last word on bar 2 either way.
- Every claim: params + tokens + memory MB next to it; inside AND beyond the trained
  length; replicated (seed pairs) before ntfy.

## Novelty rule (user, 2026-10-31 15:55 IST: "make sure we are making something novel")
- Every pre-registration names the CLOSEST PRIOR WORK and says in one line what is different.
  If nothing is different, the unit is plumbing: no claim, no GPU beyond what the benchmark
  needs. Known, to be cited not claimed: kNN-LM / RETRO (datastore keyed by the model's
  state; SL0 is kNN-LM on PAM), EM-LLM (surprise-based event segmentation of context into
  episodic memory; v14's burst rows), Titans (surprise-gated memory writes), MemGPT-class
  session memory, linear-attention / DeltaNet memories (the notebook), Hopfield settling.
- Ours so far: (1) the rule-carrying mind: an O(1) state that induces a rule from examples
  and keeps it across thousands of filler tokens past the trained length where attention
  and Mamba-2 collapse; (2) knowledge as a file, intelligence as the reader: one key space
  for session and shared libraries, text-first packs keyed by the reader at install time.
- Flagship claim to build: the SKILL LIBRARY. Packs of demonstrations the mind fetches once,
  induces the rule from, and then carries in O(1) state for the rest of a long session after
  the demonstrations are gone (a transformer's induced skill dies with its window; Mamba-2
  loses it at ~110 tokens). Cell: "skill-pack persistence": fetch at t0, queries at t0+32k,
  +128k, +1M, against T++ (re-fetch into window each time, KV cost) and Mamba-2.
- Candidate novelty to test before claiming: reading a fetched span by replaying it through
  the state (no window to paste into) integrates with what the state already holds.

## Milestones
1. Spec merged (v13 + organiser), commented by v14, frozen before any run.
2. Baselines on the suite: T++ (full context where it fits), Mamba-2, frozen PAM +
   organ + frozen address (today's numbers extended).
3. v14's trained key beats the frozen address (>0.75 top-1 at 128k, holds at 1M,
   one row per event); planner solves two-hop without a hand-coded chain.
4. Mind + library at 600M on the full suite vs T++/Mamba-2; then 3B.
5. Library format + provenance; release.
