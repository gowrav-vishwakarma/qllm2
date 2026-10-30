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

## Next action
1. G1 (pre-registered in EXPERIMENTS_V15 §4): a `grove` arm for
   `v14/microbench.py` (size-matched to echo), 3 seed pairs on B1 run 4's set,
   then the B2 set. One small job on the 4090 by the shared rule.
2. If G1 passes L and P: G2 (150M same-batch LM via a trainer hook).
3. If G1 fails in all 3 pairs: back to the meditation, not to tuning.

## Step-back notes
- 2026-10-31 00:50: the first Grove stayed at chance on the toy until the
  competition was made sharp and the growth decisive. The lesson generalises:
  an economy only works when winning is worth something. Expect the same at
  LM scale: the temperature and growth rate are the levers to watch, and the
  share entropy per grove is the diagnostic to log.
