# Before arXiv: the one list (Lead; updated as items land)

Status: `[ ]` open · `[~]` running · `[x]` done and in the paper. Owner in brackets. Each experiment is pre-registered in its lane's notebook before it runs; the paper gets its section only after the verdict.

## A. Experiments that change the paper's claims
- [x] Equal-data control at the SCI-1 recipe (3 bodies, 1B): §6.1, Table 2, Fig 1-2. [Mind, Library]
- [x] Eval-time knockouts on the trained control: §3.5, Fig 8 (erase-at-key carries the rule). [Mind]
- [ ] **Mechanism ablation ladder** at the ctl recipe, 1B each, on the 6000 after SCI-1's decay: all-on, no-delta, no-chrono, no-echo, no-readers; s8 + told-twin cells (2k/8k, trained + held-out wording). Answers what training NEEDS (knockouts answer what the model USES). → new §6.x "training ablation" + table; replaces the `\todo{ladder}` in §3.5. [Mind; told cells Library]
- [ ] **Second seed** of the three control bodies (the paper's weakest line: one seed per body). Matters most for the told-rule held-out wording (+0.06). → Table 2 gains ± over seeds. [Mind]
- [~] **Complex twin** cx_m2 (every part of the real body, 102.95M, ctl recipe), training on the 4090 since 09:17 (c17775a, 16 x 1): 10.9k tok/s vs the real body's 63k at 4 x 4 (~5.8x slower), 15.9 GB peak; ETA done ~Nov 6 14:00 IST, readouts ~15:00. The ~3x memory per sequence and ~5.8x slower training are already a cost line. Either outcome is a section: complex wins on rules → the phase-coded key is the thing (and the naming note changes); tie → the real body stands. → §1 naming paragraph, footnote, new rows in Table 2. [Mind]
- [ ] **State-budget test**: PAM with its state matched to Mamba-2's 13.9 MB at equal params. Is the language gap capacity or mechanism? Library's prior: Mamba-2's 12x state holds the told rule WORSE (0.20 vs 0.53), so it is how the state is kept. After the ladder. → §9 limitations / §6.2. [Mind]
- [x] **State forks (A31)**: MET on SCI-1 386M at 9B (fork read 0.83 = twin read, answers identical 100/100, 3.674 MB per fork). In §8 as a package operation. [Library]
- [ ] **Told-twin cells on the knockouts** (what the trained model uses for TOLD rules): ~30 min once a28_twin is wired for the knockout patches; comes with the ladder. [Mind + Library]
- [ ] **A29 with fresh protocol wordings**: today's 0.93 / 0.91 uses an in-sample parser; not a product claim until held-out wordings pass. → §7.4 wording. [Library]
- [ ] **SCI-1 final milestones + decay** (137,331 / 152,590 / 160,218 + 1B decay): the last column of Fig 4; the protoc verdict (cont3) and the worked-solutions readout (cont4, indicative only). → §7, §7.5. [Mind]
- [x] **Told vs induced competition**: the 4-way read at 137,331 did NOT support it (both rose: told 0.54 → 0.83 at 2k, induced 0.589 → 0.770). §7.2 rewritten as "co-vary with rehearsal, lagged"; EUREKA 11-04 entry withdrawn as a claim; per-use-notebook idea loses its motivation. [Mind, Library]

## B. Findings from the step-backs that are IN the paper (check wording at review)
- [x] "Stored but not expressed in every body": a told rule is 0.00 under the record-contrast / plain / no-rule reads for all three matched bodies and visible only under the twin (§5, §7.3).
- [x] The rule skill is maintained by data, not acquired once; ~1B-token lag; judge mix changes after two milestones (§7.1).
- [x] Told and induced rules co-vary with rehearsal (§7.2); the seesaw hypothesis was tested and dropped.
- [x] Mamba-2 induces the adjacent rule (0.68) and loses it over 2k (0.18): the gap is retention under interference, not induction (§6.1, Fig 2).
- [x] The library closes the reach gap at query time (§7.4) with the honesty line (package path, not understanding).
- [x] Thinking traces: finish but reason wrongly at this size (§7.5).
- [x] Complex vs real on record (Sept 2026, recipe caveat) (§1 footnote).

## C. Paper housekeeping
- [ ] Co-authors and affiliations (title block). [user]
- [ ] Full name of the co-author of arXiv:2604.05030 in refs.bib (left as surname only, not guessed). [user]
- [ ] Public repository URL + Hugging Face release tag (§Reproducibility). [user]
- [ ] Mind's number check of data/numbers.json against the notebooks; Library's check of §5, §7.3-7.4, §8. [Mind, Library]
- [ ] Architecture figure as a drawing (today: a boxed summary, Fig 1 in §3). [Lead]
- [ ] Replace every remaining `\todo{}` in main.tex; re-run `make`; proofread for claims outside a tier. [Lead]

## D. Order on the GPUs (agreed by all three, 2026-11-05)
4090 now: cx_m2 → A31 forks → knockout told-twin cells. 6000 after SCI-1's decay (~Nov 6 05:30): ablation ladder (all-on rung FIRST, to reproduce ctl PAM on that box) → second seed (3 bodies) → cx_m2 if moved → state budget.
