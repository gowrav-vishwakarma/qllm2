# v14 handover

Read `README.md` (rules) and `EXPERIMENTS_V14.md` (do-not-repeat digest +
pre-registered plan) first.

## State (2026-10-29, IST)

* **v14 is docs only.** Plan: B0 (find where delta + echo fails on the
  micro-bench) → N1 (v11 phase interference on the lean body), N3
  (formula-weight neuron), N2 (phase-modulated activation). N4, N5, S1 are
  design only.
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

Extend `v13_sempty/tmp/recall_microbench.py` with the B0 cases (a16/a32/a64,
bound attributes, rebinding) and run the control arm on the 4090.
