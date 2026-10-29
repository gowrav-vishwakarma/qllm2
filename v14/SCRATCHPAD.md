# v14 handover

Read `README.md` (rules) and `EXPERIMENTS_V14.md` (do-not-repeat digest +
pre-registered plan) first.

## State (2026-10-29, IST)

* **N1 coded + selftest PASS** (`v14/units.py`, `python -m v14.selftest`).
  **B0 tool ready** (`v14/microbench.py`). N3 folded into N2 (analysis in
  EXPERIMENTS_V14). sempyt `rotate` op added + pushed (`e403dd8`); pull
  sempyt on the 6000 before running v14 there.
* **B0 run 1 (6000) INVALID, slot returned to M2 at 18:29 IST:** no arm
  learned lookup (delta loss = ln 600; echo ~chance on facts/bind). Fix:
  sizes from 1 (easy rung). **B0 run 2** queued on the 4090: tmux `v14b0`
  runs `WAIT_TMUX=t1loc NEED_MB=12000 bash v14/run_b0.sh`, i.e. it starts
  only after the v13 session's T1 tmux ends (~21:45 IST). Do not gate on
  free memory alone: when maya stopped, B0 started next to T1 (18:30, killed
  after ~1 min). Validity: echo >= 0.90 on facts1/facts2.
* **4090 queue (both gated on the v13 session's tmux `t1loc` ending ~21:45 IST):**
  1. tmux `v14b1`: B1 run 1 (rules), arms echo, op, opdiag, n1phase, delta.
     First N6 test; bars frozen in EXPERIMENTS_V14 "N6".
  2. tmux `v14b0`: B0 run 2 (memory, sizes from 1), arms echo, delta,
     n1sum, n1gate, n1phase, op; waits for `v14b1` too.
  Logs `logs/v14_{b1,b0}_<hash>_<time>.{log,json}`.
* **N6 operator memory coded + selftest PASS** (mechanism test: the read
  turns a new input by the rotation inferred from examples, cos 1.0000).
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
* **Git:** origin = `qllm-private-personal` on both boxes. Commit → push
  origin → pull on the 6000.

## Next action

When `v14b0` finishes: fill the B0 / N1 rows in EXPERIMENTS_V14 (headroom
cells H = echo < 0.90; N1 bars there). If N1 passes, add a v14 hook to the
v13_sempty trainer for the 150M LM check.
