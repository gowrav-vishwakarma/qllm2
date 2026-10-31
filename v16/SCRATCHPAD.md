# v16 handover + board (Lead = the organiser session; Mind = formerly "Qllm v13"; Library = formerly "Qllm v14"; Lead since 2026-10-31 13:20 IST)

## Board (2026-10-31 13:25 IST)

| session | DONE today | DOING now | NEXT (pre-registered) | needs |
|---|---|---|---|---|
| v13 (mind, spec, release) | label-format pair PASS (0.934 vs T++ 0.798); s8 fresh-seed distance (PAM 0.65/0.69 vs 0.13/0.12); MLB-1 v0.2; meditation: PAM = low prior / high ceiling (skills come from the curriculum), the SHARED library is knowledge | SL0 smoke (60k-token shared library, kNN over the frozen 3B PAM's own state) before the 5M build; a4f (organ in training) to ~16:50; r3c to futility (~15:20); s9 = lbm after s8 | SL0 (568253a) vs the new-token gap to Mamba-2; curriculum: stated rule + demonstrations, fetched spans + question (+abstain); bar 3 on unseen rule FAMILIES | instr-on-b2p2 result (organiser, running); spec v1 freeze after the harness validation |
| v14 (library) | n9r FAIL; n9th seed 131 formed; n12m cancelled; W0 gate eval: met at tau 10 with the echo-included surprise (0.79 rows/16, caught 0.994/1.000); meditation: rows = event spans between surprise bursts, text-first shared library, planner v0 | L1 training pool (FineWeb shard, 6M tokens, never the held-out prose) | L1-fwd / L1-evt learned keys over burst spans on frozen c1ctlb (InfoNCE), exposed as `--key_fn`; bars A: >=0.90 @128k, >=0.75 @400k, >=0.60 @1M; negatives bank; L2 two-level index if A fails >= 400k | 4090 slot after a4f (~16:50) for vectors + head training |
| organiser (harness, coordination) | ORG-1 probe (EUREKA); v16/bench.py with all MLB-1 section-2 cells; v16 lane; PLAN sheet | 32k validation of all 16 cells (CPU); instr cell on b2p2 | burst rows (C5), distractor/planner cell (concat-m vs each-alone-margin), install-cost column; "validated" -> v13 freezes v1; baseline grid (CPU headline cells, GPU grid after 18:00) | nothing |

## Decisions (organiser)
- 13:35 IST: v14 may run the L1 data pass + head training on the 4090 NOW under a 3.5 GB cap beside a4f and Evol V1's pair (stop if a4f tok/s -30 % or < 1.5 GB free).
- 13:35 IST: N9s (v14's salient phase; two_hop bar met at 2 seeds, 27M): NO standalone 600M run; at most one paired arm inside v13's curriculum run if that mix carries the two_hop format; otherwise stays a 27M result. Its anchor idea seeds the planner (N11).
- GPU order after a4f ends (~16:50): 4090 -> v14 L1 (vectors + head; small) beside Evol V1's pair; v13's SL0 vector pass runs on the 6000 beside r3c (24 GB free there). The organiser's GPU grid takes whatever is free after 18:00.
- T++ / Mamba-2: existing checkpoints and logs only; no new baseline training; "+ library" arms are optional diagnostics.
- Every unit: pre-registered bars before numbers; params + tokens + MB on every row; nothing to Evol V1.

- 13:45 IST: 6000 disk: gscreen_onorm_3cc2b72/latest.pt (v13's abandoned G2 screen, resume-only) deleted on my authority; v13 prunes resume-only latest.pt of finished runs and moves superseded best_model.pt to d_drive (md5 first); target >= 15 GB free.
- Timing: a4f ends ~16:50 IST (stage B on the shared card); SL0 (shared library, 5M then 20M rows, /dev/shm, install cost printed) running on the 6000 in worktree qllm_sl0; v14's L1 data pass + head on the 4090 under the 3.5 GB cap.

- 13:50 IST: 6000 disk 6.5 -> 21 GB free (v13: gscreen_onorm latest.pt deleted; 11 finished best_model.pt moved to d_drive with md5 checks). MLB-1 v0.3: related-work table (kNN-LM/RETRO, EM-LLM, Titans, State Soup, Memorizing Transformers); SL0 labelled kNN-LM on a PAM mind; skill-pack persistence + replay-integration cells in.
- PRE-REGISTERED PREDICTION (v13, before any number): today's checkpoints lose an induced skill between 8k and 32k of filler (echo filler half-lives 660-3100 tokens; spread affine 0.44-0.61 at 8k). Remedies as two lanes: v13 = long-gap rule docs at long training lengths; v14 = a structural SKILL REGISTER (written once at induction, no decay, protected) as a new mechanism; closest prior work State Soup (2406.08423).

- 13:55 IST: v14 order approved: L1 (running, data pass ~1 h beside a4f) -> SR-0 (eval-only transplant of the rule vector from a short demo prompt into a long-gap query on b2p2; bar: B@32k >= 0.9 x A@2k for some layer, wrong/random vectors <= 0.25 x B) -> SR (the register) only if SR-0 passes. Prior work named: Function Vectors (Todd 2023), Task Vectors (Hendel 2023), In-Context Vectors, State Soup, Titans. b2p2 is on d_drive (from_6000/...).

- 13:30 IST (clock corrected: notes had drifted ~2 h ahead; times above now follow `date`/commit times). Real ETAs: a4f ~16:50; r3c futility val (step 39060) ~15:20; s8 ~18:00; lbm ~20:00; SL0 20M ~13:50.
- SL0 at 5M rows (v13, unit landed): library 1.3 GB, install 106 s (one 3B-mind pass over 5M tokens), query 243 s; PAM 3B + library vs PAM 3B: -0.0014 PG-19 / -0.0040 web (bar -0.05) -> rule 1 FAILS; still +0.09 / +0.11 behind Mamba-2 -> rule 2 fails; the small gain sits on new tokens (-0.009 / -0.012). 20M point reported, running. Reading: a kNN-LM knowledge library over the mind's own state does not close the general-modelling gap at this size.

- 13:45 IST: instr cell on b2p2 (trained format): stated rule alone 0.000 at 2k/8k/32k (also 0 with the rule re-read in front); 2 demonstrations before the query 0.625/1.000/0.750 (oracle 1.0). Words do not write the rule register; demonstrations do. Curriculum item confirmed (Mind); skill packs must carry demonstrations (Library). EUREKA entry added.

- 13:50 IST: v16/bench.py VALIDATED at 32k (d319e8d): all 16 task x key cells plus the changed cells run without error. Readings (4 docs): surprise bursts at tau 10 = 0.60-0.63 rows/16 tokens with a burst starting inside the needle in 100 % (segmenter works); the burst-MEAN key retrieves 0/4 (a learned pooling key is L1's job); planner v0 on top-5 spans tracks the fetch (P_concat/P_margin 0.25-0.5, bounded by whether the right span is in the top-5); never-stored queries' max-sim 0.86 overlaps stored 0.79-0.88 at 32k (frozen address cannot abstain there). Spec v1 freeze requested from Mind. CPU baseline grid (32 docs, 32k, 7 tasks x 2 key modes) started.

- 13:55 IST: MLB-1 v1 FROZEN (Mind; harness d319e8d): thesis clauses per half; R6 (0-shot stated rule >= 0.75 at 32k, inside and past trained length); R7 flagship skill-pack persistence (t0+32k: PAM pack-read-once >= max(Mamba-2 read once, T++ without re-fetch) + 0.10 every seed; T++ with re-fetch reported with KV cost); abstain bar on the learned fetch only. Decision: Mind implements the skill-pack and replay-integration cells on run_learning_probe --spread (pack at t0 in v1), rows in bench.py's vocabulary, Lead reviews the code against the text before the first score; the fetch half lands in bench.py with L1.

- 14:10 IST: skill-pack / replay cells (scripts/run_skillpack_probe.py @ 53d0e52) REVIEWED against v1 R7 + A1 and APPROVED by Lead; captions: T++ O_reread = re-fetch into a fresh window (KV = pack + query), T++ without re-fetch limited by memory at 128k, Mamba-2 state = 13.4 MB constant. Timing: start now on the 6000 with --mem_frac 0.10 (b2p1, b2p2, b2t, b2m at L <= 32k), 128k/1M and the seed-43/44 arms after s8.

- 14:15 IST: SL0 VERDICT (Mind): kNN-LM over the frozen 3B PAM's state, 5M rows (1.3 GB, install 106 s) -0.0014/-0.0040 PG-19/web; 20M rows (5.2 GB, install 328 s, exact query 951 s per 320k tokens) -0.0059/-0.0119; gap to Mamba-2 stays +0.086/+0.100; -0.05 would need ~200M+ rows (~52 GB) + ANN. Rules 1 and 2 FAIL. Filed as plumbing / cost table (serves neither thesis clause); the general-modelling gap stays the mind's. /dev/shm cleared.

## Open questions
- SL0's install cost (library tokens x one mind pass) must be printed next to MB (spec column).
- The instr cell on c1ctlb was invalid (no rule skill); b2p2 rerun decides whether words can write the rule state.


## State (2026-10-31 13:40 IST)
- Lane created. Nothing trained under v16 yet.
- GPU queue after the pivot (mutual call, 13:30 IST): keep a4f (organ in training, 4090), r3c to its
  futility check (~15:20) and s8 (6000); n12m cancelled (N12 survives as the write gate
  W0); n9th stops after seed 132 (~14:00); n9r FAIL (N9a recall +0.016/+0.014 vs bars +0.08/+0.15).
- Next: MLB-1 v1 freeze (v13 merges; organiser sends the row format; v14 comments), then baselines
  on the suite with existing T++ / Mamba-2 data, then W0 -> L1.
