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
- Nothing on a GPU (05:45 IST 2026-10-31). G1d done: the bench-trained
  organ never formed addresses (τ stayed at init; same-key overlap 0.78 vs
  other 0.79). EXPERIMENTS §4 G1d. Worktrees `../qllm_v15_g1`,
  `../qllm_v15_g1r`, `../qllm_v15_diag` can be removed (logs copied; the
  diag ckpt stays on d_drive).

## Next action
0. G2 plumbing is DONE and verified on CPU (`v15/hook.py`, `--arch grove`,
   launcher `ARCH=grove GROVE_SPEC=`); see EXPERIMENTS §4 G2 for the command.
1. Step back (the void) on G1d, then write §6 of EXPERIMENTS (design +
   pre-registration): selection that decides (competitive exclusion) and
   colonisation of empty niches (births); implement in `v15/grove.py`,
   selftest parity, toys (big-vocab 4096 keys at lr 3e-4 must stay >= 0.95;
   bench-like regime), then a new G1 chain (same bars).
2. G2 only after a G1 pass (plumbing is ready).

## Step-back notes
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
