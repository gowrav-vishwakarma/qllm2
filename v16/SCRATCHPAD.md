# v16 handover + board (organiser leads v16 since 2026-10-31 15:20 IST; "v13"/"v14" are session names)

## Board (2026-10-31 15:25 IST)

| session | DONE today | DOING now | NEXT (pre-registered) | needs |
|---|---|---|---|---|
| v13 (mind, spec, release) | label-format pair PASS (0.934 vs T++ 0.798); s8 fresh-seed distance (PAM 0.65/0.69 vs 0.13/0.12); MLB-1 v0.2; meditation: PAM = low prior / high ceiling (skills come from the curriculum), the SHARED library is knowledge | SL0 smoke (60k-token shared library, kNN over the frozen 3B PAM's own state) before the 5M build; a4f (organ in training) to ~16:15; r3c to futility (~15:00-16:00); s9 = lbm after s8 | SL0 (568253a) vs the new-token gap to Mamba-2; curriculum: stated rule + demonstrations, fetched spans + question (+abstain); bar 3 on unseen rule FAMILIES | instr-on-b2p2 result (organiser, running); spec v1 freeze after the harness validation |
| v14 (library) | n9r FAIL; n9th seed 131 formed; n12m cancelled; W0 gate eval: met at tau 10 with the echo-included surprise (0.79 rows/16, caught 0.994/1.000); meditation: rows = event spans between surprise bursts, text-first shared library, planner v0 | L1 training pool (FineWeb shard, 6M tokens, never the held-out prose) | L1-fwd / L1-evt learned keys over burst spans on frozen c1ctlb (InfoNCE), exposed as `--key_fn`; bars A: >=0.90 @128k, >=0.75 @400k, >=0.60 @1M; negatives bank; L2 two-level index if A fails >= 400k | 4090 slot after a4f (~16:15) for vectors + head training |
| organiser (harness, coordination) | ORG-1 probe (EUREKA); v16/bench.py with all MLB-1 section-2 cells; v16 lane; PLAN sheet | 32k validation of all 16 cells (CPU); instr cell on b2p2 | burst rows (C5), distractor/planner cell (concat-m vs each-alone-margin), install-cost column; "validated" -> v13 freezes v1; baseline grid (CPU headline cells, GPU grid after 18:00) | nothing |

## Decisions (organiser)
- 15:35 IST: v14 may run the L1 data pass + head training on the 4090 NOW under a 3.5 GB cap beside a4f and Evol V1's pair (stop if a4f tok/s -30 % or < 1.5 GB free).
- 15:35 IST: N9s (v14's salient phase; two_hop bar met at 2 seeds, 27M): NO standalone 600M run; at most one paired arm inside v13's curriculum run if that mix carries the two_hop format; otherwise stays a 27M result. Its anchor idea seeds the planner (N11).
- GPU order after a4f ends (~16:15): 4090 -> v14 L1 (vectors + head; small) beside Evol V1's pair; v13's SL0 vector pass runs on the 6000 beside r3c (24 GB free there). The organiser's GPU grid takes whatever is free after 18:00.
- T++ / Mamba-2: existing checkpoints and logs only; no new baseline training; "+ library" arms are optional diagnostics.
- Every unit: pre-registered bars before numbers; params + tokens + MB on every row; nothing to Evol V1.

- 15:45 IST: 6000 disk: gscreen_onorm_3cc2b72/latest.pt (v13's abandoned G2 screen, resume-only) deleted on my authority; v13 prunes resume-only latest.pt of finished runs and moves superseded best_model.pt to d_drive (md5 first); target >= 15 GB free.
- Timing: a4f ends ~18:20 IST (stage B slower on the shared card); SL0 (shared library, 5M then 20M rows, /dev/shm, install cost printed) running on the 6000 in worktree qllm_sl0; v14's L1 data pass + head on the 4090 under the 3.5 GB cap.

## Open questions
- SL0's install cost (library tokens x one mind pass) must be printed next to MB (spec column).
- The instr cell on c1ctlb was invalid (no rule skill); b2p2 rerun decides whether words can write the rule state.


## State (2026-10-31 13:40 IST)
- Lane created. Nothing trained under v16 yet.
- GPU queue after the pivot (mutual call, 13:30 IST): keep a4f (organ in training, 4090), r3c to its
  futility check (~15:00-16:00) and s8 (6000); n12m cancelled (N12 survives as the write gate
  W0); n9th stops after seed 132 (~14:00); n9r FAIL (N9a recall +0.016/+0.014 vs bars +0.08/+0.15).
- Next: MLB-1 v1 freeze (v13 merges; organiser sends the row format; v14 comments), then baselines
  on the suite with existing T++ / Mamba-2 data, then W0 -> L1.
