# Before arXiv: the one list (Lead; updated as items land)

Status: `[ ]` open · `[~]` running · `[x]` done and in the paper. Owner in brackets. Each experiment is pre-registered in its lane's notebook before it runs; the paper gets its section only after the verdict.

## A. Experiments that change the paper's claims
- [x] Equal-data control at the SCI-1 recipe (3 bodies, 1B): §6.1, Table 2, Fig 1-2. [Mind, Library]
- [x] Eval-time knockouts on the trained control: §3.5, Fig 8 (erase-at-key carries the rule). [Mind]
- [ ] **Mechanism ablation ladder** at the ctl recipe, 1B each, on the 6000 after SCI-1's decay: all-on, no-delta, no-chrono, no-echo, no-readers; s8 + told-twin cells (2k/8k, trained + held-out wording). Answers what training NEEDS (knockouts answer what the model USES). → new §6.x "training ablation" + table; replaces the `\todo{ladder}` in §3.5. [Mind; told cells Library]
- [ ] **Warp-floor rung (clk3 finding, Mind 17089b6, from the outside review):** eval-time clk3 on M9 NOT met narrowly (slow heads + echo +0.12 D_far at NLL +0.013, bar +0.01; coupled clock + retention + echo +0.145 at +0.021; echo alone +0.105 at +0.005), but slowing the Chrono clock WITH retention gives 2.6x the retention-only gain, and 3 of the 6 longest-holding heads sit at the warp clamp floor exp(-3): the body asks for slower clocks than allowed. Ladder rung: warp floor widened to -6, at the ctl recipe, after cx_m2. → §3.5 / §7 analysis line (credit the outside review). [Mind]
- [ ] **Told rule held-out wording, split per A26's pre-registered G1 / G2 / G3 reporting rule (Library 4bfbb22, found by the outside review):** A28b protocol twin, 2k, right / unit: G2 (trained quantity, held-out wording, n 56) PAM 0.20 / 0.39 vs Mamba-2 0.07 / 0.30 vs T++ 0.09 / 0.16 (SE ~0.05 on right); G3 (unseen quantity, n 44) 0.02 / 0.11 vs 0.05 / 0.11 vs 0.00 / 0.00; Decay_M9 386M (no matched baselines, scale / data note only) G2 0.39 / 0.79, G3 0.32 / 0.55. Label: "split per A26's reporting rule; A28b's bar is the pooled one (not met); one seed, n 56 / 44". Line: at 103M the held-out-wording edge is on trained quantities; unseen quantities transfer only at 386M. → Table 2 / §6.1 columns + conv_ok beside right. [Lead writes; Library numbers]
- [x] **Far rule documents (Nov 12; §7 "Rules carry as far as they are rehearsed", Table far):** s8 @8k untrained 0.186 -> 0.979 / 0.974 / 0.964 (22.5B / 26B / 27B), told 8k 0.10 -> 0.45-0.53; decay findings in §7 "A learning-rate decay erases the far rule first". [Mind, Library; Lead wrote]
- [ ] **Far documents in the equal-data control** (103M, ctl recipe + far block, T++ / Mamba-2 / PAM): does Mamba-2 learn far carry from the same documents? Decides the "beats baselines beyond the trained length on rules" sentence. 4090 after cx_m2. [Mind]
- [ ] **Release recipe:** short decay from a far-block checkpoint, judged on s8 @8k (3-save mean + margin) + told 8k. [Mind]
- [ ] **Second seed** of the three control bodies (the paper's weakest line: one seed per body). Matters most for the told-rule held-out wording (+0.06). → Table 2 gains ± over seeds. [Mind]
- [x] **Complex twin** (cx_m2, Mind 3684032, Nov 16): complex vs real at the ctl recipe, 1.0B: holdout 3.775 vs 3.427 (+0.35 nats), needles 8k / 16k 0.29 / 0.24 vs 0.65 / 0.59, clk2 8k 0.27 vs 0.59, s8 0.536 vs 0.523 (equal), HS / ARC-e / PIQA 0.271 / 0.438 / 0.577 vs 0.274 / 0.455 / 0.606, 5.8x slower, ~3x memory. Real body stays; the naming paragraph + footnote get this matched result (replaces the Sept recipe-caveated one). [Mind; Lead writes]
- [ ] **State-budget test**: PAM with its state matched to Mamba-2's 13.9 MB at equal params. Is the language gap capacity or mechanism? Library's prior: Mamba-2's 12x state holds the told rule WORSE (0.20 vs 0.53), so it is how the state is kept. After the ladder. → §9 limitations / §6.2. [Mind]
- [x] **State forks (A31)**: MET on SCI-1 386M at 9B (fork read 0.83 = twin read, answers identical 100/100, 3.674 MB per fork). In §8 as a package operation. [Library]
- [ ] **Told-twin cells on the knockouts** (what the trained model uses for TOLD rules): ~30 min once a28_twin is wired for the knockout patches; comes with the ladder. [Mind + Library]
- [ ] **A29 with fresh protocol wordings**: today's 0.93 / 0.91 uses an in-sample parser; not a product claim until held-out wordings pass. → §7.4 wording. [Library]
- [ ] **SCI-1 final milestones + decay** (M9 137,331; M10 = the long run at 152,590 (sci1_long_3c2bf1f, cont3 mix; user 11-06); the release decay after s8reg): the last column of Fig 4; the protoc verdict (cont3); M10-worked (cont4) only as a failed-segment note. → §7, §7.5. [Mind]
- [x] **Told vs induced competition**: the 4-way read at 137,331 did NOT support it (both rose: told 0.54 → 0.83 at 2k, induced 0.589 → 0.770). §7.2 rewritten as "co-vary with rehearsal, lagged"; EUREKA 11-04 entry withdrawn as a claim; per-use-notebook idea loses its motivation. [Mind, Library]

- [ ] **Library arms for s8, needles, skill pack per item** (user, 14:35): pairs jsonl cells s8 / needle / skill from 152,590 on; old A26 section hidden where pairs exist (done). [Library]
- [~] **Package suite on every milestone** (row 1 at 9B DONE, in §7.6 + Fig 9; light guard every 1B, full suite every 5B / release per PLAN_LONGRUN) (user, Nov 5 13:40): R x K with the library fetch, A29 rewrite, recall through the organ at 32k / 128k with MB held, twin-fork read; one row per milestone, bare vs package side by side; backfill M4 / M7 if cheap; BABILong + package on the final SCI-1 model on the 6000 after the decay. → new §7.6 / Fig: "the same checkpoint, bare and with the library". [Library; Mind yields]

- [ ] **Chat continuation after SCI-1 (user, Nov 5 13:55):** the model answers simple prompts poorly ("answer only YES or NO. Is water wet?"); chat is ~6 % of the mix vs science ~25 %. After the final run + decay: continue from the final checkpoint (re-warm), KEEP science and the rest of the mix, RAMP the chat share gradually over the continuation (e.g. 6 % → ~20-25 % across milestones, each step a measured calibration row) with everyday chat / short direct answers / yes-no / instruction-following data (public, filtered, decontaminated); keep the rule-doc share fixed; pre-register a simple-chat probe (held-out wordings) that must rise while s8 / told / reach / benchmarks must not fall; judge after 2 milestones. Decide then: part of the release or a separate chat variant. → §7 / release notes. [Mind data + run; Library probe]

- [x] **ARC-LIB on the three ctl bodies (Library 8e2e685, Nov 5 16:31): NOT met.** Identical BM25 library, 103M / 1B each: gain on ARC-e + ARC-c pooled (n 3,548, ± 0.014) PAM +0.058, T++ +0.055, Mamba-2 +0.054 (bar PAM ≥ best + 0.02). Every body uses fetched passages, equally; with the answer passage given T++ 0.957 / 0.935 > Mamba-2 0.923 / 0.920 > PAM 0.868 / 0.856 (copying from adjacent text favours attention). The paper must NOT claim a library-use edge; PAM's edge stays rules across filler + reach past the trained length. → one honest row in §8 (the package helps every body; the O(1) mind's value is the state, not the reading). [Library; Lead writes the row]
- [ ] **Benchmarks with the package (user, Nov 5 15:00):** (a) BABILong qa1-qa5 + RULER subset to 32k on the final SCI-1, bare vs package, MB held per row; (b) a KNOWLEDGE-library arm for ARC-e / ARC-c, HellaSwag, PIQA (retrieval over filtered, decontaminated corpora; identical for every body) and a TOOL arm for GSM8K (model writes the calculation, sandboxed executor computes); on SCI-1 first, then the three ctl bodies with the identical library / tool: claim = "uses fetched knowledge better", not "retrieval helps"; per-item pairs on the review page (user, 14:40). [Library]
- [x] **Review page: model alone vs model + library per item** (user; live from checkpoint 9, automatic for later milestones): render Library's per-item package rows next to the bare answers; bare-only items marked why. [Lead, from Library's per-item jsonl]

- [ ] **Reference block vs public small models (user, Nov 5 14:30; after the final SCI-1 checkpoint):** Qwen ~0.5B and SmolLM2-360M through OUR harness on: (1) long context past the trained length (RULER-style + BABILong, 4k-32k, accuracy AND MB held per session), (2) in-context learning (s8 rule from examples, told rule twin read), (3) accuracy vs memory-per-session chart, (4) ARC with the shared science library; plus the standard suite (HellaSwag, ARC-e/c, PIQA, WinoGrande, OBQA) always with tokens trained. Framing: "~100x+ fewer tokens and smaller, and still competing HERE"; reference block, no verdict on knowledge benchmarks. [Lead runs on the 4090; Library's runners]

- [ ] **Outside review (Gemini plan, user Nov 5 17:00; reviewed by Lead / Mind / Library):** adopted, eval-only first: (1) two-regime read in v16/package.py (warm for evidence in view, rule twin for held rules; A9 removed) [Library, now]; (2) A32 self-compiled forks, gated by a 10-min pre-check (twin read at L = 0 on A28b held-out wording, 100 docs: >= 0.6 run, < 0.5 stop) [Library, M10 window]; (3) clk3: dilate only the heads with slow MEASURED retention (capture_decay on held-out text, not bias) + echo, told / s8 at 8k, prose NLL guard [Mind, after M10]; (4) two-bounce SharedRead for 2-hop: micro-bench, then a 103M screen, after the ladder + second seed [Mind]. Rejected: cgu_scale as the B3 cause (T++ has no scalar and is the worst LM; scale x ||down|| grows smoothly with depth; a reparametrisation; optional rung only if GPU time is left); ARC rerank (body-neutral plumbing after ARC-LIB). Fixed: contribution 4 wording (6b6c99c).

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
4090 now: cx_m2 → A31 forks → knockout told-twin cells. 6000 after SCI-1's decay: the open-ended long run (PLAN_LONGRUN). 4090 after cx_m2 (Mind, 11-05 17:05): clk3 (eval) → eviction log (eval) → ablation ladder (lad_all first) → second seed (3 bodies) → two-bounce micro-bench / screen → state budget.
- [ ] **s8 checkpoint jitter (Mind, 11-06):** 10 optimizer steps moved s8 D 0.423 -> 0.331 (C2 first save). Re-read the SCI-1 milestone series and the tier tables for ~±0.1 single-checkpoint jitter: report s8 as the mean over 3 adjacent saves + the mean target margin; the ctl gaps (0.523 / 0.149 / 0.074) are far outside it, the milestone swings are not. [Mind]
- [ ] **Decay vs far rule carry (Mind, 11-07):** state the four-way finding (plain decay 0.77 -> 0.23; lerp linear in l; rule-x2 decays from M10 0.42 -> 0.36 and from M9 0.77 -> 0.46, margin +1.86 -> -0.17; 0 / 600 b2s prompts reach 2k) in §7 as a limitation + cause; the paper cites M9 (rules) and Decay_M9 (benchmarks) side by side until a b2s_far release exists. [Mind]

## E. Findings of 2026-11-16 .. 11-18 to write into the paper (Lead, noted 11-18 07:13 IST; numbers + hashes are in the notebooks named)
Each item says where it goes. **Written into main.tex and data/numbers.json on 11-18 11:05 IST** (abstract, contribution 3, related work, new \S sec:wio with Table tab:wio, \S sec:axes, \S sec:excursion, the decay subsection, limitations, conclusion; 21 pages, builds clean). Still open from this list: figures for the five-arm table, the infrastructure appendix, and the number check by Mind / Library (item C).
- [ ] **"What is ours" at equal data (wio, 5 arms, ~103M, 1.0B tokens, far-block mix; EXPERIMENTS_SEMPY "wio", bcebc850, bd68e6b0).**
  New §: a four-way table (gdn / nochrono / pam / mamba2, tpp when it lands) on s8 @8k (3-save mean, per task by distance), t40,
  clk2, HS / ARC-e / PIQA, val NLL, params, solo tok/s. State plainly: the pre-registered claim (pam >= gdn + 0.10 on s8 @8k, and
  nochrono loses the gap) was NOT met: pam 0.343 < gdn 0.408 < nochrono 0.427; mamba2 0.259. Rewrite the abstract / contribution
  on rule carry accordingly: with far documents in everyone's data, example-rule carry is not an edge of the full model.
- [ ] **The clock trades rules for needles (one-flag ablation, same table).** pam vs nochrono differ only in Chrono: t40 16k 0.80 vs
  0.35 (gdn 0.33, mamba2 0.42), clk2 mixed, s8 lower at every distance on both tasks. This is the clean architectural result;
  move the paper's mechanism claim to "a content-modulated clock on a delta-rule memory buys verbatim recall past the trained length".
- [ ] **Gated DeltaNet as the closest prior work (related work + limitations).** Same recurrence; differences listed in
  EXPERIMENTS_SEMPY (tied vs untied erase, 12 vs 4 write sites, decay form, short conv vs RoPE / Chrono). GDN leads val NLL
  (3.385 vs 3.425) at ~1.8x our speed and fits affine maps better at distance 0 (0.97 vs 0.66-0.71). Mamba-2 leads NLL (3.381)
  and cannot do shift_wide even at distance 0.
- [ ] **Metric note (errata for §7):** the old "s8 D 0.52 vs 0.07 / 0.15" is the 2k / 4k mean on a mix WITHOUT far documents; on that
  metric wio pam is 0.538 while nochrono = gdn = 0.64. Label every s8 number with its distances and whether far docs were in training.
- [ ] **A told rule has four axes: presence (B), strength (C), scope, precedence (EXPERIMENTS_V14 "TOLD-MARGIN", "SCOPE"; probe code 57ad8435; results 139fc9fe for 41B / 42.75B / 43B, c2c104f7 for 44B).**
  Method §: twin gate = B + 5C exactly; the four-cell probe (one skeleton, held-out quantity direction; design due to the outside
  reviewer). Results: single-save told 8k tracks B (Spearman told 8k vs B 0.96, 1c38ed70; the reviewer's audit gate_win vs B 0.979; vs C 0.73), not the carry; at 41B / 44B scope_fu 2k is 0.73 / 0.55
  and precedence u1 0.49 / 0.42: the body knows a rule exists, not what it is about. Limitations § gets this in plain words.
- [ ] **Constant-LR excursions (EXPERIMENTS_V14 "3d series", "3f readout"; 88d742f0, 82bd02bd, fbc8cf33).** Doubling presence-only far
  protocol docs (3d) gave a ~1B-token excursion: C 0.45 (42B, n 300) -> 3.64 at 42.75B with P_fu 0.89-0.95 and scope_fu 1.00, back to
  baseline by 44B (8k series 0.45, 1.09, 0.59, 3.64, 2.45, 0.86, 0.58, 0.47; 2672840d). A second, single-save one at 45B under the scope-aware mix (3f); the next save (689,500, CPU guard read, method b539d107) is CLEAR (fbc8cf33). Lesson for the methods §: judge a mix change on a settled
  window (>= 1B after the switch) and on all four axes; a one-axis bar (dC) read "WORKS" on what was partly over-application.
- [ ] **Decay erases the carry itself (DECAY B / C split, 3f61ea54; C3 partial, c703578a).** Paired on 100 docs: decay_M9 - M9
  dC(2k) -0.37, LAWA -0.26, control +0.18 (ns); rule x2 inside the decay keeps about half (-0.175). Replaces the "read problem"
  hypothesis (recorded as not supported). Goes with the existing "a learning-rate decay erases the far rule first" subsection.
- [ ] **Package: negative results worth one paragraph each.** A32c (self-demos + split number read) met its bars (0.46 / 0.69 held-out
  2k, 0.83 / 0.61 at 8k; 41ba344) but, like the contrast gate v0.4 (0.76 at 8k), fails the scope guard (false units 0.86-0.97 on
  other-quantity protocols; 1a786a4e): no decode-time read fixes an unbound rule; v0.3.1 stays. Cite contrastive decoding (Li 2022),
  CFG for LMs (Sanchez 2023), context-aware decoding (Shi 2023), Paulo, Marshall & Belrose 2024 (arXiv:2404.05971).
- [ ] **SCI-1 status row:** 43B / 44B / 45B CHECKs (bc1cb217, 7ef901ef, bd68e6b0): s8 @8k 0.94-0.98, skill 0.71-0.85, t40 16k 0.81-0.98,
  clk2 headline read on 8k docs (dilation 8 + c4) 0.84 / 0.79 / 0.92 and its 16k read (32k run) 0.60 / 0.48 / 0.68 (label both; earlier
  notes called the first one "clk2 16k"), HS / ARC-e / PIQA ~0.32 / 0.55 / 0.64; chat sources cycle since 40.5B (CYCLE_SOURCES, a31ed81). No 386M baseline yet
  (b386 pre-registered d4f33c0 / b2300b0 / dcaaed5; 4-LR screen; ~102 h on the 4090).
- [ ] **Infrastructure lessons for the reproducibility appendix:** probe outputs written to .part and kept only on exit 0 + parse;
  never delete on an empty checksum (812463b9); launch guard fails closed (89aff04); free-space gate (3c7ccb9).
- [ ] After the tpp arm and the wio scope probe land: update data/numbers.json (force-add) and main.tex in one pass; then ask Mind /
  Library for the number check (item C).
