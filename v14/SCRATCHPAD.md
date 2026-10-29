# v14 handover

Read `README.md` (rules), `EXPERIMENT_INDEX.md` (one row per idea) and
`EXPERIMENTS_V14.md` (do-not-repeat digest and pre-registered bars) first.
Keep all three current as things happen (user rule, 2026-10-29).

## State (2026-10-30 00:35 IST)

* **4090 (this session owns it until `v14b0` ends):**
  1. **B1 run 1 DONE** (00:17 IST, `5b1eb9d`). N6's frozen bar NOT MET:
     letter_shift / number_step / reverse3 hit the ceiling (1.00 for every
     arm, delta included). **label_category: op = opdiag = 1.00 vs echo
     0.76 / 0.77 / 0.52** (chance 0.50 / 0.50 / 0.25), so the diagonal
     memory helps, not the rotation. two_hop: every arm at chance. Full
     table and verdict: EXPERIMENTS_V14 "B1", "N6".
  2. tmux `v14b0`: **B0 run 2 RUNNING** since 00:20 IST at `0671d65`
     (first start at 00:18 died after 20 s: python3.9 segfault during a
     torch.compile recompile, RAM peak 3.5G, so not the cap; relaunched
     as is). Log `logs/v14_b0_0671d65_20261030_0020.log`. ETA ~02:30 IST.
     Validity: echo >= 0.90 on facts1/facts2 @1024.
  Both run under a 12G RAM cap (systemd-run scope) and close their tmux on
  exit.
* **Why B1 was two hours late:** the v13 session's T1 tmux `t1loc` stayed
  open after "t1loc DONE" (21:31 IST): its launcher ended with `exec bash`.
  They removed it. **To do after `v14b0` ends** (editing the running
  launcher is unsafe): let `run_b0.sh` also start when the other run's
  driver log prints DONE, not only when its tmux closes.
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

1. Pre-register and build **B1 run 2** (next commit): rule parameters
   never seen in training (letter_shift k 4-5, number_step d 10-13), a new
   training seed (does the label_category gain replicate?), and an arm
   with its own query projection (shared projection vs diagonal state).
   Queue it on the 4090 behind `v14b0`.
2. When `v14b0` ends: the same for B0 run 2 and N1 (headroom cells H = echo
   < 0.90; N1 bars in EXPERIMENTS_V14 "N1").
3. If N6 or N1 passes: build the v14 hook in the v13_sempty trainer for the
   150M LM check on M1. If both fail: record them, then choose N5 or N4 from
   what B0 run 2 shows.
