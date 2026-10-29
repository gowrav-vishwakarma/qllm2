# v14 handover

Read `README.md` (rules), `EXPERIMENT_INDEX.md` (one row per idea) and
`EXPERIMENTS_V14.md` (do-not-repeat digest and pre-registered bars) first.
Keep all three current as things happen (user rule, 2026-10-29).

## State (2026-10-30 03:25 IST)

* **4090 (this session owns it until `v14b0` ends):**
  1. **B1 run 1 DONE** (00:17 IST, `5b1eb9d`) and **B1 run 2 DONE** (01:44,
     `85ecf02`). **N6 CLOSED, FAIL:** replication not met (label_category
     mean op 0.427, opdiag 0.587 vs echo 0.471), and every arm is below
     chance on unseen rule parameters (letter_shift_ood 0.00: the models
     pick a trained shift). N6 code removed; it is at `78e4354`.
     **Lead: echo_tied** (W_q = W_k) label_category 0.92 / 0.92 / 0.74 vs
     echo 0.61 / 0.54 / 0.27, one seed. Seed noise ≈ 0.2 (echo 0.68 ->
     0.47 between runs), so ≥ 3 seeds before any claim.
  2. tmux `v14b0`: **B0 run 2 RUNNING** since 01:45 IST at `bcfd88a`.
     Arms echo, delta, n1sum, n1gate, n1phase, echo_tied (op / opdiag
     dropped with N6). ETA ~03:45 IST. Validity: echo >= 0.90 on
     facts1/facts2 @1024. Earlier attempts lost: 00:18 python segfault,
     00:39 eval OOM (missing `no_grad`, fixed in 5eff3b9), 00:56 stopped
     so B1 run 2 could go first.
* **Launcher:** `v14/run_bench.sh` (replaces `run_b0.sh`; the running B0 run
  2 still executes its already-open copy). It also stops waiting when the
  other run's log prints its DONE line (`WAIT_DONE=file:pattern`): the v13
  T1 tmux stayed open two hours after its run ended on 2026-10-29.
* **6000:** the v13 session's chain. tfs1 (Transformer++ one stage, started
  03:15 IST at `b788731`) -> tfs2 -> c2m2 (ETA ~08:00 IST) -> tfc1
  (Transformer++ on c1m2's aptitude recipe, tmux `s3chain`, ETA ~09:30 IST).
  Their E2 tied-echo 600M arm waits on B1 run 3 bar P; its recipe is the
  reference picked after c2m2 (c1m2 leads). The 4090 frees at ~05:00 IST
  after run 3, offered to them for E2 if P passes.
* **v13 C1 on M2 (`fed73cf`): KEEP, best 600M LM so far.** Holdout 3.4473
  (M2 3.4697, C1-ctl 3.4743), WikiText 65.74, recall 0.870, learning probe
  0.187 (flat). **label_category 8-shot does NOT lift on M2:** c1m2 0.38,
  M2 one-stage 0.31, C1-ctl (M1, two stages) 0.55. So the two-stage lift
  depends on the body (shared reads hide it). Compare label_category with
  M1-body numbers, or within one body (an M2 arm against c1m2's 0.38).
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
* **v13 C1-ctl verdict (`237173a`):** most of C1's recall jump, and its
  label_category gain, come from the **two-stage schedule** (language first,
  recall docs later, LR re-warmed), not the aptitude data or its
  multiple-choice format. The ctl beats one-stage M1 on holdout (3.4743 vs
  3.4906), recall (0.825 vs 0.615) and label_category 8-shot (0.55 vs
  0.29). So the multiple-choice B0 family is dropped. For stage 3, both arms
  use the same schedule shape. Next on the 6000 (tmux `s2chain`, after C1
  on M2, ~03:30-08:00 IST): Transformer++ in one stage vs two, then two
  stages on M2.
* **Session coordination:** v14 (this session, "Qllm v14") and v13_sempty
  ("Qllm v13") ping each other on every GPU start/stop and on every verdict.
* **sempyt:** v14 needs ≥ `b99e038`. Pull it on the 6000 before running
  v14 there.
* **Git:** origin = `qllm-private-personal` on both boxes. Commit → push
  origin → pull on the 6000.

## Next action

1. **B1 run 3 queued** (tmux `v14b1r3`, behind `v14b0`; pre-registered in
   EXPERIMENTS_V14 "B1"): echo vs echo_tied in 3 seed pairs, trained with
   `label_marker` added; bars P (label_category) and A
   (label_marker_new, every item unseen) on the paired difference. ~70
   min, ETA ~05:00 IST. When it ends: notebook + index first, then ping
   Qllm v13 (their 600M tied-echo arm waits on P), then the user.
2. When `v14b0` ends: the same for B0 run 2 and N1 (headroom cells H = echo
   < 0.90; N1 bars in EXPERIMENTS_V14 "N1").
3. If N1 passes B0, or echo_tied replicates in run 3: build the v14 hook in
   the v13_sempty trainer for the 150M LM check on M1 (same schedule shape
   for both arms). If neither: record them, then choose N5 or N4. Bar 3 in
   its strong form (inferring an unseen rule parameter) is still unmet by
   every arm, which points at N4 (in-context learning in the unit).
