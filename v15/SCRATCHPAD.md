# v15 handover — Grove

## State of play (2026-10-31 ~01:00 IST)
- Lane: v15 = a new species from the unit up (this session, "qllm2-59").
  v13 = PAM maturity, v14 = new mechanisms on PAM. EUREKA.md is read before
  every next step; insights go there and by message.
- Design written and the organ coded: `v15/grove.py` (GroveLayer season form
  + `step`, GroveLM). `python -m v15.selftest` PASS: parity 1e-15, toy
  in-context lookup 0.98 (chance 0.125) on CPU in 20 s.
- Isolation (user, 2026-10-31): v15's ideas stay unpolluted by v13/v14. Their
  work is used only as baselines, benches and trainer plumbing.
- Nothing has run on a GPU yet. 4090: v13's n9c was using ~15 GB at 00:30.

## Running
- **GIV-2 pair 3 RUNNING (Nov 2 22:41 IST)**: Grove IV (`dcd5c86`) vs echo,
  seed 23, SAVE=1, tmux `giv2p3_v15`, worktree `../qllm_v15_giv2`, ~8 GB,
  end ~01:00 IST Nov 3. Decides bar P (split 1/2 after pairs 1–2; L failed).
  Logs `../qllm_v15_giv2/logs/v14_v15giv2p3_dcd5c86_*` -> copy to `logs/`.
- GPU: the 4090 is this lane's since Nov 2 22:39 IST (Lead, from the user);
  Library runs a <= 10 GB eval job 1–2 h every ~7.5 h: keep ~12 GB free or
  checkpoint often. The 6000 needs Lead's OK. RAM: kill side jobs below
  ~12 GB free.
- Compile speed smoke (`v15/tmp/compile_speed.py`, needs PYTHONPATH=.) deferred
  until pair 3 ends (keep the card's headroom for Library/Mind evals).
  Alone, Grove IV trains at 33.9k tok/s (Grove I 43.8k): ~1.3x, not 3.5x.
- Probe of pair 1's saved grove done (EXPERIMENTS 6.2): traits moved (τ to
  13.9, b_born 0.4) but no exact re-selection in context (overlap 0.00).

## Next action
1. Read pair 3 (~01:00 IST): P decided as registered; record + lineage + tree.
2. Pre-register the next row before any run: either GIV-2x (same pairs,
   9000 steps: late convergence is the observed cause of the split; "a bit
   more training" is within bar-2 tolerance, stated as cost) or G2 (the
   150M-token LM holdout) if P passes. Compile smoke decides whether G2 is
   affordable (eager Grove IV is ~3x echo's step time).
3. Step back in writing before choosing (SCRATCHPAD notes).

## Speed note (06:40 IST 2026-10-31)
- Grove II on the bench: 12.5k tok/s vs Grove I 43.8k (3.5x slower; echo
  ~80k alone). CPU profile of one forward+backward (bench shape, 8 seasons):
  no hot op; ~360 slice-backwards, ~380 divs, ~770 copies, ~440 zero-fills
  per 8 seasons. On the GPU that is launch-bound: hundreds of tiny kernels
  per season x seasons x layers. Remedy = `torch.compile` of the season body
  or one fused season kernel. 09:35 IST: `torch.compile` works on the season
  form after the autocast refactor (`a8fe35f`): output identical, state 1e-15,
  2x on CPU at toy size; switch `V15_COMPILE=1` (bench adapter + trainer
  hook). Bench shape (T 1024, 32 seasons unrolled): compile 110 s, output
  identical, CPU fwd 0.08 -> 0.06 s (CPU is not launch-bound; the GPU gain
  is the open question, unmeasured: no slot). The queued GIV-2 runs the
  pinned `dcd5c86` (same maths, eager).

## Step-back notes
- 2026-11-02 22:50, back after the exit, pairs 1–2 of GIV-2 read. What
  surprised me: the economy's first real win is the thing it was built for
  (prototypes: label_category 0.90/0.83 vs 0.49/0.25) and it came in the
  pair whose training *finished*; the other pair was simply still
  descending. What it really says: Grove IV is a late learner. For ~3000
  steps the soil is untrained, births and placements churn, and the read is
  mostly noise the out-projection learns to ignore; once the soil makes
  similar words land together, prototypes form and the loss drops fast.
  That is a warm-up cost, and the user's bar-2 tolerance allows "a bit more
  training", but it must be stated as cost on every row, not hidden by
  longer runs. Bold different angle, not run: let the organ *start*
  useful instead of learning to be: innate roots initialised from the
  embedding's own structure (the soil of frequent tokens), so dawn trees are
  real words on day one and the economy has something to select among
  before any gradient. In the frame: a grove is planted, not sown from dust.
  Decide after pair 3 and the compile smoke; a longer-training row only as
  a pre-registered cost measurement, never as a re-roll.
- 2026-10-31 09:35, after the sapling fix. The bimodal rule toy (0.09 vs
  0.95 by seed) was not noise: a loss trace showed a dead plateau from step
  250, and the cause was one line of the read (a newborn's empty fruit read
  at every novel token). Two lessons. (1) When a run is bimodal across seeds,
  trace the bad seed before touching anything else; a plateau from the start
  is a constant being injected somewhere. (2) In the organism frame the fix
  read itself: a sapling bears no fruit. When the fix is a sentence in the
  frame's own language, it is probably right; when it needs a knob, it is
  probably not.
- 2026-10-31 06:25, after building Grove II. Four of the five faults found
  this morning were not in the idea but in defaults carried over from Grove I
  (soft-share growth rate, conv bias, coupled spreads) and in one rule I had
  not thought through (a novel token eating an old tree). The decode-parity
  test did not catch any of them: both forms agreed on the wrong thing. What
  caught them was a *trace of one document* (who was born, who ate what).
  Lesson for the lane: every organ gets a one-document trace tool on day one,
  next to the parity test. Parity says "consistent"; the trace says "right".
- 2026-10-31 03:40, after G1r pair 1 (rules yes, lookup no, guard no).
  What it says: ripening made Grove a rule learner (letter_shift 0.99,
  reverse3 1.00), so selection-as-recurrence does carry a rule. Lookup is
  weaker than echo because inside a season the address of a fact is a
  *soft hash*: the share vector softmax(τ·cos(x, dawn roots)) over 32 trees.
  Two random keys overlap on many trees, and every tree's fruit is a mix of
  everything that touched it; the read of a key is then a blur of its
  neighbours' values. The roots only sharpen at season ends, and growth
  merges (mean-seeking), it never separates. The higher train loss (0.61 vs
  0.38) is the same blur seen from the LM side.
  From far away: an economy allocates by *competition for territory*, and
  what I built competes only for share, not for territory. A tree that wins
  a token should push neighbours out of that region (the territory becomes
  its own), and a token nobody fits should found a new tree (a birth) rather
  than be split among strangers. Both are the "repulsion" that competitive
  learning in nature has and mean-seeking growth lacks.
  Not run, not designed yet: (a) births at season ends from the season's
  least-fit tokens, into trees with ~0 share for a whole season; (b) a
  territory term: growth moves the winner toward x and the runner-up away;
  (c) sharper innate address: τ per tree, not per grove. Decide after pairs
  2–6 land; if L/P fail in all pairs, the void again, not a knob.
- 2026-10-31 02:50, after G1 pair 1 (chance everywhere). What surprised me:
  the toy passed and the bench failed for a reason that was visible in the
  design text all along ("a fact written in a season is readable from the
  next one"). I had written the blind window down and not asked how wide the
  bench's facts were. Lesson: every bar needs a sentence on the *distance*
  the organ must bridge, in tokens, before it runs.
  What it really says: the organism had two clocks (token, season) and only
  the slow one wrote memory. Ripening gives the fruit the fast clock while
  the root keeps the slow one: identity changes slowly, contents at once.
  That split (who I am: seasonal; what I hold: immediate) feels right beyond
  this fix and is worth keeping as a principle for births later.
  Bold different angle, not run: let the *season length itself* be chosen
  by the grove (a season ends when the shares have settled, i.e. when the
  entropy of m stops falling), so seasons are events, not a clock. It would
  break the fixed C×C parallel form; a fused kernel could do it. Parked.
- 2026-10-31 00:50: the first Grove stayed at chance on the toy until the
  competition was made sharp and the growth decisive. The lesson generalises:
  an economy only works when winning is worth something. Expect the same at
  LM scale: the temperature and growth rate are the levers to watch, and the
  share entropy per grove is the diagnostic to log.
