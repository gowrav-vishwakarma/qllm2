# v14 handover

Read `README.md` (rules) and `EXPERIMENTS_V14.md` (do-not-repeat digest +
pre-registered plan) first.

## State (2026-10-29, IST)

* **N1 coded + selftest PASS** (`v14/units.py`, `python -m v14.selftest`).
  **B0 tool ready** (`v14/microbench.py`). N3 folded into N2 (analysis in
  EXPERIMENTS_V14). sempyt `rotate` op added + pushed (`e403dd8`); pull
  sempyt on the 6000 before running v14 there.
* **Queued on the 6000 (slot between M1 and M2, agreed with session
  qllm2-04):** tmux `v14b0` runs `~/v14_b0_chain.sh` (log
  `~/v14_b0_chain.log`): waits for "=== m1 DONE" (~18:15 IST), pulls
  qllm-private + sempyt, runs `v14/run_b0.sh` (arms delta, echo, n1sum,
  n1gate, n1phase; 6000 steps each; cap 165 min), then ALWAYS
  `touch ~/v14_slot.done`, which starts m2chain. Results:
  `logs/v14_b0_<hash>_<time>.{log,json}` on the 6000.
  Rerun: `tmux new-session -d -s v14b0 "bash ~/v14_b0_chain.sh 2>&1 | tee -a ~/v14_b0_chain.log"`.
* **Session coordination:** v14 (this) and v13_sempty (qllm2-04) ping each
  other on every GPU start/stop and on verdicts. The 4090 runs T1 until
  ~21:45 IST; the 6000 is qllm2-04's again after B0.
* **Machines are busy with v13_sempty work (other session):**
  - 6000: tmux `m1chain` (M1 sparse memory, from ~16:30 IST) → `m2chain`
    (M2 shared reads, done ~20:15 IST). Details: `v13_sempty/SCRATCHPAD.md`.
  - 4090: T1 thought-stream local check, log
    `logs/v13_sempty_t1loc_t1_ea002e6_20261029_1641.log`.
* **Control body = M1** (KEEP at 600M: holdout 3.4906 vs echo 3.4885,
  recall 0.615 vs 0.401, 1.8x faster). Flags: echo, pam_layers 3,7,11,15,
  mlp_hidden 2352 + chrono/out_gate/delta. N1 on M1 = +28k params.
* **B0 started on the 6000 at 18:02 IST** (delta arm ~22 min; 5 arms + evals,
  cap 165 min).
* **Git:** origin = `qllm-private-personal` on both boxes. Commit → push
  origin → pull on the 6000.

## Next action

When `v14b0` finishes: fill the B0 / N1 rows in EXPERIMENTS_V14 (headroom
cells H = echo < 0.90; N1 bars there). If N1 passes, add a v14 hook to the
v13_sempty trainer for the 150M LM check.
