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
- **G1 RUNNING (started 01:46 IST 2026-10-31, n9c closed 01:45)**, tmux `v15g1`, driver `v15/tmp/g1_chain.sh`
  (untracked), pinned worktree `../qllm_v15_g1` at `84b41d8` (.venv symlinked).
  Waits for tmux `n9c` to end and >= 12 GB free on the 4090, then runs
  grove vs echo (B1 run 4's set, seeds 21/22/23: TAG v15g1p{1,2,3}), then the
  B2 set (seeds 41/42/43: v15g1b2p{1,2,3}). ~1 h per pair (grove ~0.3-1.3
  s/step: python season loop). Logs: `../qllm_v15_g1/logs/v14_v15g1*_84b41d8_*.{log,json}`
  (copy to `logs/` and commit with the verdict). Bars: EXPERIMENTS_V15 §4 G1.
  Arm: `V14_EXT_ARMS="grove=v15.bench:build_grove"` (27.67M vs echo 27.61M).

## Next action
1. Read G1 when the pairs land: L (label_exact >= echo + 0.15 in >= 2/3),
   P (label_category >= 0.9 in >= 2/3). Record in the index, EUREKA if met.
2. If G1 passes L and P: G2 (150M same-batch LM via a trainer hook).
3. If G1 fails in all 3 pairs: back to the meditation, not to tuning.

## Step-back notes
- 2026-10-31 00:50: the first Grove stayed at chance on the toy until the
  competition was made sharp and the growth decisive. The lesson generalises:
  an economy only works when winning is worth something. Expect the same at
  LM scale: the temperature and growth rate are the levers to watch, and the
  share entropy per grove is the diagnostic to log.
