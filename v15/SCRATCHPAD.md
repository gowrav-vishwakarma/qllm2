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
- **G1r RUNNING (started 02:35 IST 2026-10-31)**, tmux `v15g1r`, driver
  `v15/tmp/g1r_chain.sh` (untracked), pinned worktree `../qllm_v15_g1r` at
  `abe4c89` (ripening; .venv symlinked). Six pairs grove vs echo: B1 run 4's
  set (seeds 21/22/23: TAG v15g1rp{1,2,3}), then the B2 set (seeds 41/42/43:
  v15g1rb2p{1,2,3}); ~1 h per pair; chain end ~09:00 IST. Logs:
  `../qllm_v15_g1r/logs/v14_v15g1r*_abe4c89_*.{log,json}` (copy to `logs/`
  and commit with the verdict). Bars: EXPERIMENTS_V15 §4 G1 (unchanged).
  Arm: `V14_EXT_ARMS="grove=v15.bench:build_grove"` (27.67M vs echo 27.61M).
  4090 sharing: v14's whole-card n9r runs beside it from ~03:30 (agreed).
  Pair 1 (03:28 IST): rules learned (letter_shift 0.99, reverse3 1.00) but
  lookup below echo (label_exact 0.34 vs 0.50 at 2-shot; label_category 0.42
  vs 0.49 at 4-shot), guard fails (train loss 0.61 vs 0.38). L, P not met.
  Summary tool: `.venv/bin/python v15/tmp/g1_summary.py` (reads the jsons).
- G1 as built (`84b41d8`): pair 1 FAIL at chance, blind-window cause found
  and fixed (EXPERIMENTS §4 G1); its worktree `../qllm_v15_g1` is stopped by a
  docs-only dirty edit (the guard refuses launches) and can be removed.

## Next action
0. G2 plumbing is DONE and verified on CPU (`v15/hook.py`, `--arch grove`,
   launcher `ARCH=grove GROVE_SPEC=`); see EXPERIMENTS §4 G2 for the command.
1. Read G1r when the pairs land: L (label_exact >= echo + 0.15 in >= 2/3),
   P (label_category >= 0.9 in >= 2/3). Record in the index, EUREKA if met.
   Watch the guard too: train loss within 0.1 nat of echo.
2. If G1 passes L and P: G2 (150M same-batch LM via a trainer hook).
3. If G1 fails in all 3 pairs: back to the meditation, not to tuning.

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
