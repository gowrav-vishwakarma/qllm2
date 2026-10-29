# v14 handover

Read `README.md` (rules) and `EXPERIMENTS_V14.md` (do-not-repeat digest +
pre-registered plan) first.

## State (2026-10-29, IST)

* **N1 coded + selftest PASS** (`v14/units.py`, `python -m v14.selftest`).
  **B0 tool ready** (`v14/microbench.py`). N3 folded into N2 (analysis in
  EXPERIMENTS_V14). sempyt `rotate` op added + pushed (`e403dd8`); pull
  sempyt on the 6000 before running v14 there.
* **Queued:** tmux `v14b0` on the 4090 runs `v14/run_b0.sh` (arms delta,
  echo, n1sum, n1gate, n1phase; 6000 steps each, ~20 min/arm). It waits for
  14 GB free GPU memory, then logs to `logs/v14_b0_<hash>_<time>.log` + `.json`.
  Restart: `tmux new-session -d -s v14b0 "bash v14/run_b0.sh"`.
* **Machines are busy with v13_sempty work (other session):**
  - 6000: tmux `m1chain` (M1 sparse memory, from ~16:30 IST) → `m2chain`
    (M2 shared reads, done ~20:15 IST). Details: `v13_sempty/SCRATCHPAD.md`.
  - 4090: T1 thought-stream local check, log
    `logs/v13_sempty_t1loc_t1_ea002e6_20261029_1641.log`.
* **Control body** = echo (`e1g_echo_a0fab65`) until M1/M2 report; switch to
  the winner before any v14 run.
* **Git:** origin = `qllm-private-personal` on both boxes. Commit → push
  origin → pull on the 6000.

## Next action

When `v14b0` finishes: fill the B0 / N1 rows in EXPERIMENTS_V14 (headroom
cells H = echo < 0.90; N1 bars there). If N1 passes, add a v14 hook to the
v13_sempty trainer for the 150M LM check.
