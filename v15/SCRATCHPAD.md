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
- **G1d QUEUED (04:44 IST 2026-10-31)**: grove arm of pair 1 with SAVE=1 for a
  CPU probe of the trained organ (tmux `v15diag`, worktree `../qllm_v15_diag`
  at `abe4c89`, ckpt -> `/media/gowrav/d_drive/qllm_ckpts/v15_diag_checkpoints_v14/`).
  Probe: `.venv/bin/python v15/tmp/probe_shares.py <ckpt>/grove@hd64L4.pt 4`
  (share entropy per grove, cos(m_query, m_same_key) vs other keys, learned
  τ/λ/α). Random init: entropy ~2.7 of 3.47 nats, overlap 0.86 vs 0.73.
- G1r DONE 04:44 IST: FAIL on L and P after 2 of 3 pairs (EXPERIMENTS §4 G1).
  Pairs 3–6 stopped by the guard (dirty docs edit in `../qllm_v15_g1r`).
  Worktrees `../qllm_v15_g1` and `../qllm_v15_g1r` can be removed (logs copied).

## Next action
0. G2 plumbing is DONE and verified on CPU (`v15/hook.py`, `--arch grove`,
   launcher `ARCH=grove GROVE_SPEC=`); see EXPERIMENTS §4 G2 for the command.
1. G1d: when the ckpt lands (~05:30 IST), run the probe; write what the
   trained organ does (sharp or blurred addresses? which groves live?).
2. Step back (the void), then redesign the addressing in the organism frame
   (territory / births, SCRATCHPAD 03:40) or something better; new G1 chain.
3. G2 only after a G1 pass (plumbing is ready).

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
