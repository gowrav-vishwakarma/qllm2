# v14 handover

Read `README.md` (rules), `EXPERIMENT_INDEX.md` (one row per idea) and
`EXPERIMENTS_V14.md` (do-not-repeat digest and pre-registered bars) first.
Keep all three current as things happen (user rule, 2026-10-29).

## State (2026-10-30 01:05 IST)

* **4090 (this session owns it until `v14b0` ends):**
  1. **B1 run 1 DONE** (00:17 IST, `5b1eb9d`). N6's frozen bar NOT MET:
     letter_shift / number_step / reverse3 hit the ceiling (1.00 for every
     arm, delta included). **label_category: op = opdiag = 1.00 vs echo
     0.76 / 0.77 / 0.52** (chance 0.50 / 0.50 / 0.25), so the diagonal
     memory helps, not the rotation. two_hop: every arm at chance. Full
     table and verdict: EXPERIMENTS_V14 "B1", "N6".
  2. **B0 run 2, attempt 4 queued** (tmux `v14b0`, behind `v14b1r2`; ETA
     ~04:00 IST). Attempt 3 was stopped a minute in so B1 run 2 could go
     first. Arms echo, delta, n1sum, n1gate, n1phase, op, opdiag.
     Validity: echo >= 0.90 on facts1/facts2 @1024. Attempt 1 (00:18):
     python3.9 segfault in a torch.compile recompile after 20 s (RAM peak
     3.5G, not the cap). Attempt 2 (00:20-00:39): CUDA OOM in the eval once
     echo had trained. B0's `evaluate()` ran with autograd on (no
     `no_grad`), which fit on the 96 GB 6000 but not on the 4090. Fixed.
  Both run under a 12G RAM cap (systemd-run scope) and close their tmux on
  exit.
  3. tmux `v14b1r2`: **B1 run 2 RUNNING** since 00:57 IST at `85ecf02`
     (pre-registered in EXPERIMENTS_V14 "B1"): arms echo, op, opdiag,
     opdiag_ownq, echo_tied; new seed; unseen rule parameters. Log
     `logs/v14_b1r2_85ecf02_20261030_0057.log`. ETA ~02:00 IST. The first
     start (00:39, `84f426b`) died at 00:55 in op's training: python3.9
     "invalid opcode". Its files were deleted. 85ecf02 includes the v13
     session's T1 removal (v14 selftest ALL PASS on it).
* **The 4090 box crashes python itself (2026-10-30):** a segfault (00:18,
  write into the interpreter's code) and an invalid opcode (00:55, in the
  interpreter's code), both in steady work, neither a Python error. The CPU
  is an **i9-14900KS** (microcode 0x133), the model most hit by Intel's
  13th/14th-gen instability, whose symptoms are exactly these. There is no
  ECC, so nothing else logs it. A hypothesis, not proven; it is the user's
  hardware call. Meanwhile `run_bench.sh` reruns a run that died by a
  signal with `--resume` (RETRIES=2), so a crash costs one arm, not the
  queue.
* **Launcher:** `v14/run_bench.sh` (replaces `run_b0.sh`; the running B0 run
  2 still executes its already-open copy). It also stops waiting when the
  other run's log prints its DONE line (`WAIT_DONE=file:pattern`): the v13
  T1 tmux stayed open two hours after its run ended on 2026-10-29.
* **6000:** the v13 session's chain until ~03:30 IST 2026-10-30 (tmux
  `c1ctl`, C1 control without aptitude, ETA ~01:25; then C1 on M2).
* **v13 results this evening (600M, r600 protocol):** M2 shared reads KEEP
  (holdout 3.4697 vs M1 3.4906; recall 0.522 vs 0.615; 1.18x slower). T1
  thought stream FAIL (3.4977 vs echo 3.4885). C1 aptitude curriculum
  primary FAIL (learning probe 0.199 vs 0.175), but recall jumped (63-cell
  0.907 vs 0.615) with fewer recall docs. Their guess: the multiple-choice
  "Answer: B) text" format acts as recall practice. Learning probe: every
  model is near chance (Mamba-2 3B 0.202). **Bar 3 is open everywhere.**
* **Control body for LM checks: M1** (EXPERIMENTS_V14 Part B: faster, reads
  not shared). Compare a passing unit with M2's 600M numbers before a 6000
  slot.
* **Data for the N6 LM check (if B1 passes):** the v13 session's real
  aptitude blend, `v7/aptitude.py` (`138cddb`). It uses the train splits of
  14 reasoning datasets (AQuA-RAT, GSM8K, LogiQA2, ReClor, RACE, ARC, CSQA,
  OBQA, SciQ, QASC, BoolQ, StrategyQA, WinoGrande, HellaSwag), ~48.8M tok,
  source `aptitude` in load_pretrain_mix (C1 stage-A weights
  dclm,fineweb,smoltalk2_mid,aptitude = 0.25,0.20,0.15,0.40,
  blend_warmup_tokens 30M). The test/validation splits are untouched, so
  they also work as a real-data reasoning eval.
* **Possible B0 family (later):** a multiple-choice recall format, if the C1
  control confirms the format is what helped.
* **Session coordination:** v14 (this session, "Qllm v14") and v13_sempty
  ("Qllm v13") ping each other on every GPU start/stop and on every verdict.
* **sempyt:** v14 needs ≥ `b99e038`. Pull it on the 6000 before running
  v14 there.
* **Git:** origin = `qllm-private-personal` on both boxes. Commit → push
  origin → pull on the 6000.

## Next action

1. When `v14b1r2` ends: score bars R, C, O (EXPERIMENTS_V14 "B1" run 2)
   into the notebook and the index first; commit with the logs; ping Qllm
   v13; tell the user. If R is met, pre-register the 150M LM check.
2. When `v14b0` ends: the same for B0 run 2 and N1 (headroom cells H = echo
   < 0.90; N1 bars in EXPERIMENTS_V14 "N1").
3. If N6 or N1 passes: build the v14 hook in the v13_sempty trainer for the
   150M LM check on M1. If both fail: record them, then choose N5 or N4 from
   what B0 run 2 shows.
