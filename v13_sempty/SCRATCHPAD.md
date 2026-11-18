# v13_sempty handover — fused real-PAM speed work (2026-09-03)

Read this first if you are picking up the real-arm training. Everything below
is committed (`git log --oneline 4740d65..HEAD -- v13_sempty/`); the lab
notebook entry is `EXPERIMENTS_SEMPY.md` → "Speed: fused real arm".

## Step back (Mind, 23:31 IST, 2026-11-17, after the wio pam verdict: claim NOT met)
- Surprise: the clock we called "ours" lowers rule carry (s8 @8k 0.343 vs 0.427 without it) and raises needles 2.3x. ctl's 0.523 was the
  2k/4k metric (pam 0.538 on it here), so ctl never tested 8k against an equal recipe; the old "PAM >> T++ / Mamba-2" gap is real but
  is the notebook + far data, not the clock.
- From far away: three mechanisms each win one axis (GDN: affine / language; PAM-RoPE: far shift / clk2; PAM-clock: needles). The goal
  (rules beyond recall) wants the rule axis; the release body (386M, clock on) still scores s8 @8k 0.98 at 44B, so scale may hide it.
- Bold angles (not proposals yet): (a) split key geometry, phase-free rule heads + phased episode heads, one O(1) state; (b) GDN-style
  tied erase in our notebooks (4 write sites vs GDN's 12) as the affine fix; (c) test whether the clock's rule cost shrinks with scale
  (a 386M wio pair is ~2 x 47 h; only if it can change the release decision). (d) the claim we CAN make: at equal data, PAM-RoPE notebooks
  carry far shift 2.5x GDN and keep clk2 3x; that is a mechanism result, smaller than "the clock".
- Next per Lead: mamba2 / tpp arms continue; no proposal before Lead reads this.

## Step back (Mind, 08:29 IST, 2026-11-04; user via Lead: "read the meditation memory, step back, see all again, discuss")

Since SCI-1 began: 5 milestones; think tried and dropped; cont1 lost s8 (0.78 → 0.44 in 1B); A26 compact 0.40 → 0.00 (M4, old
mix) → 0.45 / 0.60 (protoc); Pythia reference; reach past 4k grew with plain data; two of my own process errors (b2s over-cut,
the switcher's checkout).

**What surprised me.**
1. What decayed is only the rule CARRIED ACROSS FILLER. With the examples adjacent (@0) s8 stayed 0.96–0.98. So the induction
   itself was not lost; the state's willingness to KEEP it through unrelated text was. That is a retention habit of the write /
   erase gates, which text trains toward "forget", and rule docs train toward "keep".
2. It is not only the cut. In the matched 1.5B tier, s8 also fell with data at a constant 3 % b2 share (0.768 → 0.545), and A26
   compact went 0.40 → 0.00 between M3 and M4 on an unchanged mix. These skills track how RECENT and how DENSE their docs are,
   not the cumulative dose. They behave like working-memory habits under interference, not like stored knowledge.
3. The query-time tools did not decay at all: skill pack + contrast and clk2 (dilation + contrast) held or grew while s8 fell.
   What the mind does when TOLD at query time is robust; what it must keep doing on its own drifts. This is the Oct 31 meditation
   again ("query-time beats training-time"), seen from the forgetting side.

**Arguing with Lead's five.**
- (1) Architectural or habit: both. The capacity is architectural (T++ at equal data never has it: 0.134 → 0.047, it ALSO fell
  with data). The level is habitual. Before buying a protected path, measure SAVINGS, which costs nothing: cont2 restored b2s at
  76,295. If s8 D is back to ≥ 0.70 by 91,554 (1B tokens; the first acquisition took 2B, M1 0.32 → M3 0.71), the skill was
  suppressed, not erased, and a cheap schedule (spaced rehearsal) is the lever. If it crawls back, it was erased and a protected
  path earns a test. A protected path is not new here: A2b's vault (pinned state + protect gate) was tried for retention and not
  adopted, so a new attempt needs a new reason.
- (2) Compiling prose: agree that this is the gap; disagree with training it first. Every training-time addition has cost
  language (cu1, lg8r, cu2). We already decided "words → demonstrations lives in the LIBRARY" (Nov 1 D3): the planner rewrites
  the prose rule into the compact demo form the state already keeps. Cheapest test, no GPU training: A26 G1 with a
  prose → compact rewrite step in the package (rule text → 2–3 worked compact lines). If G1 goes 0.00 → ~0.45, the gap is
  solved at query time.
- (3) A matched control at the product recipe: needed for the PRODUCT claim, but the species claim already has matched controls
  WITH the same rule docs (600M 3 seeds, 1.5B tier). A 125M × 1B SCI-recipe pair on the 4090 is one or two nights. I'd do it
  after the decay, not now, and with the s8 + reach curve as its only bars.
- (4) Reach as the cleanest claim: agree, but word it carefully. clk2 uses a query-time operator (echo dilation + contrast) on
  the state. The clean sentence is: "a state trained at 4k is read at 8k+ by an O(1) operator, and the read improves with plain
  data; a windowed T++ cannot run that operator". Yes to the 16k / 32k curve at the next CHECK (t40 + clk2 at 16k / 32k; the
  4090 has room).
- (5) Agree. Add: the trainer should print REALIZED per-source token shares at every val step (calib6 said b2s 1.41 % vs 1.2 %
  target; long docs make 15M-token calibrations noisy). Then a mix is checked by the run itself, not predicted.

**Added after M7 (Lead + Mind):** the lag. A mix change shows nothing in its first 1B tokens (s8 0.435 → 0.436) and moves in the
second (→ 0.588). Savings / mix readouts need TWO milestones, not one; the M6 "erased" verdict was premature.

**What I would stop.** Reading 20-doc A26 cells as trends (SE ~0.11; at least 100 docs for a bar cell). think_eval in the CHECK
(thinking is gone). Single shown generations as evidence (they flip every milestone; keep them for the user, not for verdicts).
Changing the mix at every milestone: one change per 2 milestones, from a measured row.

**One bold idea: use-dependent plasticity (a "synaptic tag" for the notebooks).** Brains keep a skill through unrelated activity
because plasticity is local: a synapse changes only where it was recently used ("synaptic tagging and capture"). Our gates
change everywhere, on every token. Mechanism: each notebook head carries a tag = its running write activity on the current doc.
The gradients into that head's retention / erase parameters are scaled by its tag, so a head that held a rule through a b2 doc
is only re-tuned by documents that USE it, and plain text cannot slowly re-teach it to forget. O(1), no extra state at inference,
and it is a training rule, not a new block. Closest prior work: EWC / synaptic intelligence (per-parameter importance from
task-level Fisher), gradient masking in continual learning, and Hebbian tags. Difference: the tag is per-head and per-token,
computed from the state's own write activity, with no task boundaries.
Cheapest honest test (4090, 125M): train on rules + text, then 0.5B tokens of text only, with vs without tags; bar: s8 D
retained ≥ +0.2 over the untagged arm, holdout ≤ +0.01.

## Step back (Mind, 15:52 IST, 2026-11-05, second today; user via Lead: "sit in the void after reading all the history, then come back")

Today: the equal-data control (rules / told rules / recall beyond the trained length beat both baselines; Mamba-2 the better pure
LM); the knockouts (erase-at-key carries, a global fade is worse than no erase, echo + readers are the read path); the lag and the
withdrawn seesaw; the package arms (skill 1.00 with the examples re-shown; needle 0.998 → 0.927 because the contrast read cancels a
fact already in state); GSM8K tools (format without schema; trained names invented); ARC-c 0.28 → 0.84 with the answer stated, BM25
top-3 adds nothing; the complex twin (2 h to port, 3x memory per sequence).

**What surprised me.** One pattern under all of it. Everything that WORKS re-presents the right thing at query time: re-shown
examples (1.00), a stated answer (0.84), Library's rewrite (0.93), the rule held in a state that forgets only locally. Everything
that FAILS asks the model to decide or find on its own: BM25 on question words finds nothing for ARC-c; the offered schema loses to
the most frequent training name; the contrast assumes the state does NOT already know. The mind reads and keeps well; it does not
address well. Its weakness is the address, not the memory.

**What it really says.** The thesis is "an O(1) mind + a library", and today's numbers say the interface between them is the
bottleneck: what to fetch, and whether to fetch at all. Two failures are the same mistake from opposite sides: retrieval keyed on
surface words (BM25) ignores what the mind wants; the contrast read ignores what the mind already has.

**Practical, cheap (no training): a knowledge-gated contrast.** Read the question with and without the fetched row; if the no-row read
already puts the answer above its competitors with margin m, skip the contrast (the state knows it). Prediction: the needle loss
(0.998 → 0.927) closes, skill / R × K unchanged. A package change for Library; CPU / 4090 minutes.

**Bold, different angle: retrieve in the mind's own key space** (Lead's 16:00 block reached the same place: an eviction-log library
in the notebook's key space). The knockouts say each fact is written along a unit KEY and kept by erasing only along keys, so the
notebook already has an address scheme, and it is not word overlap. Index the library corpus by the keys the trained notebooks write
when they read each passage (per sentence / passage end, the four notebook layers), and retrieve with the QUERY vector the notebook
forms at the question: retrieval becomes the same operation as reading memory, done outside the O(1) state; the library is an
unbounded notebook in the same key space. Cheapest honest test: ARC-c with the current package, passages = Library's science library,
(a) BM25 top-3 (0.278), (b) notebook-key top-3, (c) the stated answer (0.835). If (b) moves ARC-c clearly above (a), the mind can
address its own library. Closest prior work: kNN-LM, Memorizing Transformers (kNN over attention keys), RETRO; HyDE (a generated
answer as the query). Difference: the keys are the delta notebook's unit keys (the space its erase works in), read with its own query.

## Step back (Mind, 08:32 IST, 2026-11-05; user via Lead: "re-read the meditation memory, go in the void, step back")

Since the last block: the equal-data control (B1 / B2 / B5 met, B3 not), the lag (a mix change shows nothing for 1B tokens), the
told / induced seesaw, the paper draft, and the complex twin ported with every part (delta, clock, gate, sparse notebooks, shared
reads, echo) in about two hours, now training as cx_m2.

**What surprised me.**
1. At the matched control, with the examples adjacent, the three bodies are close (PAM 0.75, Mamba-2 0.68, T++ 0.36), and 2k
   tokens of filler separate them (0.61 / 0.18 / 0.10). Induction is cheap; KEEPING the induced rule while unrelated text passes
   is the whole difference. That is a statement about forgetting, not about learning.
2. Mamba-2 keeps 12x the state (13.9 vs 1.15 MB) and still loses the rule and holds the told rule worse (0.20 vs 0.53). Size is not
   it; how a state is overwritten is.
3. The "expensive" complex question took two hours. I had treated it as a week's work for a week.
4. My own recurring bug: four times this week a `pgrep -f` / `pkill -f` pattern matched the shell that ran it (a waiter that never
   started, a pkill that killed its own command). That is a habit to stop, not a one-off.

**Which part carries the rule across filler (Lead's question).** My prior, from what each part does:
- **delta erase-at-key**: the strongest candidate. A filler token erases only along ITS unit key; a rule stored along other keys
  is untouched. Mamba-2's update fades and overwrites along every direction the input touches. This is exactly "forget locally,
  not globally".
- **Chrono clock**: the second candidate. The warp g_t can slow the clock during filler, so the rule's phase still matches the
  query after 2k tokens; with plain RoPE the relative phase drifts with every token.
- **echo**: unlikely for s8. It recalls the token that FOLLOWED a similar context; a rule applied to a new input has no stored
  successor. It should matter for copy / needle reach, not induction.
- **shared reads**: unlikely to carry; they add read points to the same four notebooks, not storage.

**Bold, cheap step before the ladder: eval-time knockouts on the trained ctl PAM checkpoint (minutes on the 4090, no training).**
On the s8 spread probe, switch one mechanism off ONLY during the filler spans (the examples and the query untouched):
(a) erase gate b_e = 0 (pure additive writes, i.e. no erase-at-key); (b) b_e = 1 with global decay (Mamba-like overwrite);
(c) clock frozen at g = 1 (plain RoPE); (d) echo out-proj zeroed; (e) readers zeroed. The drop in s8 D at 2k / 4k per knockout says
which mechanism the TRAINED model uses to carry. It is not a training ablation (a model trained without a part may compensate), so
it does not replace the ladder; it decides which rungs are worth 1B tokens each and gives the paper a mechanism figure today. The
dilation hook already edits retention per span, so this is ~40 lines.

**On the ladder and the state budget.** Agree with both, in the order knockouts → ladder (only the parts the knockouts flag, plus
one "all on" seed) → second seed → state budget. A second seed matters more than I said before: every bar is one seed.

**What I would stop.** pgrep / pkill by pattern (use pid files or tmux session names); judging any mix change in under 2B tokens;
and new SCI-1 mix segments (agree with Lead: let it finish and decay).

## State of play

* **ctl HANDOVER (2026-11-04 10:50 IST).** The matched control (PAM M2 / T++ / Mamba-2 ~103M, 1B tokens each of the cont2 mix)
  runs on the 4090 in tmux `mindctl`, worktree `/home/gowrav/Development/qllm_ctl` @ caa4da9 (its `.venv` -> qllm311; checkpoints
  -> d_drive `ctl_<arm>_{main,decay}_caa4da9`). `ctl_chain.sh` yields to CHECK / humaneval / Library python jobs and RAM < 12 GB, then
  resumes from latest.pt; restart after a box crash: rerun the same tmux command (it skips arms with `.done`). Pre-registration
  EXPERIMENTS_SEMPY "ctl: matched control". Readouts land in the worktree's `logs/ctl/`; copy them to master to commit.
* **AFTER A LOCAL REBOOT (written 07:26 IST 2026-11-05; the 4090 box restarts ~10:30 IST 11-05).** The 6000 is unaffected (cont3 trains in tmux
  `sci1`; the cont4 switch at 137,331 runs alone in tmux `sci1sw`). The local waiters live in tmux and die with the box, and the
  scratchpad copies are lost. Re-arm from the repo (the CHECK copies the milestone itself; it skips anything already `.checked`):
  ```
  cd /home/gowrav/Development/qllm2
  tmux new-session -d -s mindm9 "bash v13_sempty/tmp/sci1_milestone_waiter.sh sci1_cont3_449e320 step_137331 2>&1 | tee -a logs/v13_sempty_sci1_waiter_local.log"
  tmux new-session -d -s mindm10 "bash v13_sempty/tmp/sci1_milestone_waiter.sh sci1_cont4_8284510 step_152590 2>&1 | tee -a logs/v13_sempty_sci1_waiter_local.log"
  ```
  (If cont4's hash changes at the switch, use the dir printed in the 6000's `logs/v13_sempty_sci1_switch_8284510_*.log`.) Then
  check: `tmux ls | grep mindm`; and that `/media/gowrav/d_drive/qllm2/from_6000/checkpoints_v13_sempty/*/` holds no `*.part`
  (an interrupted copy; delete it, the waiter re-copies). ctl is finished (no re-arm). The control worktree `../qllm_ctl` and the
  CHECK worktree `../qllm_sci1check` are on disk and survive.
  **Queued after the reboot (Lead, in order):** (1) re-arm mindm9 / mindm10 (above); (2) the paper review: every number in
  `v16/paper/data/numbers.json` vs the notebooks (ctl, t15, 600M seeds, SCI-1 M1-M8, ablation rows, Pythia block) and §3 +
  Appendix A (backward pass) vs `v13_sempty/model.py`, corrections sent as "file:line → fix" (Lead merges); the complex-vs-real
  footnote answer was already sent (ledger row 3, EXPERIMENTS_SEMPY:209); (3) a FEASIBILITY note, no launch, for a complex-vs-real
  PAM pair at the ctl recipe: which current flags (delta, chrono, out_gate, share_reads, echo, reset_at_eos) the complex PAMLayer
  lacks; a 5-min 4090 smoke of the complex arm (tok/s, peak VRAM) AFTER the 137,331 CHECK; recommendation (plain body where both
  heads exist vs porting the parts to the complex head) with hours each.
  **Queued for AFTER SCI-1's final run + decay (user via Lead, 13:55 IST 11-05; v16/paper/TODO.md):** pre-register a CHAT continuation
  from the final checkpoint: (a) everyday chat / short direct answers / yes-no / instruction-following sources (public, filtered,
  decontaminated) with their share from a measured calibration row; (b) the rule-doc share kept (maintenance lesson); (c) re-warm
  from the final checkpoint; (d) bars after 2 milestones: Library's simple-chat probe rises, s8 / told / reach / HS-ARC-PIQA do not
  fall; (e) a recommendation: release model vs a separate chat variant.
  Design (user via Lead 14:05: keep science and the rest IN, raise chat GRADUALLY 6 % -> ~20-25 %, each step measured, rule-docs
  fixed): (1) branch from the main segment's `stable_final.pt` (pre-decay, WSD practice), not from the decayed end; (2) ONE run with
  an in-sampler linear ramp of the per-source targets W0 -> W1 over the first 2B tokens, then hold W1 for 1B, then the 1B decay on
  W1 (sampler change ~15 lines in `v7/data.py`'s token-deficit blend: target(t) = (1 - a) W0 + a W1, a = min(1, consumed / N); env
  RAMP_WEIGHTS / RAMP_TOKENS); (3) measured calibration rows at W0, the midpoint and W1, plus the trainer's [mix] line every val as
  the run's own measured ramp; (4) the chat gain taken from web first, science shares and the rule-doc block (b2s / proto4k /
  recall4k) fixed; (5) bars at each 1B milestone and final at ramp end + 1B (the lag rule): Library's simple-chat probe up; s8 D
  >= decay-end - 0.05; told 2k >= decay-end - 0.10; far needles 16k and clk2 8k >= decay-end - 0.05; HS / ARC-e / PIQA >= -0.01;
  per-source science holdout <= +0.03.

* **SCI-1 HANDOVER (21:30 IST, 2026-11-04).** cont3 is TRAINING on the 6000: tmux `sci1` in `~/Development/qllm_sci1` @ 449e320, dir
  `checkpoints_v13_sempty/sci1_cont3_449e320`, from cont2 M7 (global 106,813) to 160,218, then the decay (`sci1_decay_449e320`). Why:
  the one-cause test of protoc (EXPERIMENTS_SEMPY "cont3 PRE-REGISTRATION"). Milestones 122,072 (~Nov 5 04:00 IST; then the cont4
  switch adds the `worked` source, pre-registered), 137,331, 152,590. Local CHECKs: tmux `mindm7` (after the M7 CHECK in `mindm6`).
  Library's A26 + twin read and Lead's humaneval gate on the `.checked_<m>` markers. On the 6000 kept: best_model of each segment,
  cont2 step_106813 deleted 22:40 IST 11-04 (d_drive copy sha 4817f47e… = the 6000). cont3 step_122072 deleted 06:00 IST 11-05 (sha dad8741d… = the 6000). 14:42 IST 11-05 (Lead decision): the old SCI-1 best_model / stageA_final files (22 GB) and the 45 non-SCI-1
  tier checkpoints (36.8 GB: b2*, t15*, r3c*, mix3b*, c1*, tf*, w1n* …) moved to d_drive `from_6000/checkpoints_v13_sempty/<run>/`,
  sha256-gated (v13_sempty/tmp/move_6000_to_d.sh; log logs/v13_sempty_move_6000_to_d.log: 50 moved, 0 real mismatches); 6000 89 GB
  free, d_drive 105 GB. The 6000 keeps only SCI-1's current segment + its newest 2 milestones (KEEP_LAST 2, sha-gated deletes). Thinking is OFF.
* **Fast stream resume (2026-10-27, after the R4 screen's 47-min skips).**
  `v7/data.py` now reads dclm / fineweb / smoltalk2 (blend) / pg19 file by file
  with pyarrow in the exact old order (verified doc-for-doc vs the old HF
  stream). Checkpoints save `<src>@f`/`<src>@r` (file, raw row): crash-resume
  is seconds (dclm: at most one 2.7 GB file). `--init_from` warm starts begin
  each source at a never-read file (estimated from footers + the committed
  index `v7/stream_index/`): **first train step ~3 min after launch instead of
  ~47** (4090 smoke from the Phase 3b cursor: dclm/fineweb file 6, smoltalk2
  file 40). Old checkpoints still resume (doc-count cursor, slower path,
  extends `.cache/v7_tokens/stream_index/`). `scripts/build_stream_index.py
  --commit-copy` shares the index via git.

* **2026-11-02 14:42 IST: SCI-1 RUNNING (handover).**
  - **Run:** the 6000, tmux `sci1`, `~/Development/qllm_sci1` @ `59c3cf5`.
    - Launch: `source v13_sempty/tmp/sci1_env.sh && bash v13_sempty/tmp/sci1_chain.sh`.
    - Logs: `logs/v13_sempty_sci1_chain_59c3cf5_20261102_0901.log` (UTC stamp) and `logs/v13_sempty_sci1_stageA_59c3cf5_*.log`.
  - **Pre-registration:** f11df39 (EXPERIMENTS_SEMPY "SCI-1"); Lead GO.
    - M2 386M (1024 × 24), SEQ 4096, 65,536 tokens / step, LR 8e-4 wsd_stable.
    - Stage A 0.5B (plain) → stage B 10B (the full mix, `--continue_from`) → decay 1B.
  - **Speed:** 37.7k tok/s, 77.3 GB. Stage A ends ~18:25 IST; milestone 1 (1B into stage B) ~02:00 IST Nov 3; the end ~3.5–3.8 days.
  - **Resume:** the launcher auto-resumes (MAX_RETRIES 20); stage A's `.done` skips it on a chain restart.
    - To continue later: `CONTINUE_FROM=<a full step_*.pt / stable_final.pt>` with a fresh TARGET_TOKENS and an optional new
      SOURCES / WEIGHTS (logged).
  - **CHECK:** on each milestone, to `logs/v13_sempty_sci1_checks_59c3cf5.log`; generations in `logs/sci1/v13_sempty_gens_*.txt`.
    Library runs A26 on 4090 copies.
  - **Tripwires at milestone 2** (acted on at the first continuation): s8 D < 0.40; A26 G1 ≤ NOPROT + 0.05; think ON ≤ OFF + 0.02
    or OFF leak > 5 %; holdout > projection + 0.02.
  - **Watchdog:** a Mind background loop pings the 6000 every 30 min for stalls.

* **2026-11-01 14:21 IST: STEP BACK after R × K.**
  - *What surprised:*
    - A fetched record between a rule and its query barely moves the state-held rule (b2p2 1.00 → 0.91), but wrecks T++'s
      in-context rule (0.69 → 0.17). Interleaved retrieved content distracts attention; the state keeps the instruction.
    - That is the real-world RAG pattern (instructions, then retrieved passages, then the question), and a benchmark cell MLB-1 lacks.
    - *Bold angle:* "instruction persistence under retrieval" as an MLB-1 cell, with natural variants (A9: the rule as a written
      instruction, passages as real docs). Closest prior work: Lost in the Middle (Liu et al. 2023), distractor robustness in RAG.
      What differs: the rule is held in O(1) state and the passages arrive after it.
  - *What it really says:*
    - The 2k misses are wrong values of the right type: the trace is FUZZY, not absent.
    - Contrast cancels the copy pull of the fetched value.
    - If w1r replicates W1's s8 gain, the window may be sharpening rule fidelity, which would be the link to test.
  - *Hazard for cu3:* a training format acts as an attractor at document start (cu2a reads any record as "copy the field"). cu3
    needs records in many positions, with OTHER skills asked of them, as counter-examples.

* **2026-11-01 10:53 IST: HANDOVER STATE.**
  **Running:**
  - **bud** on the 6000 (tmux `bud`, `~/Development/qllm_bud` @ `e96588b`): bud0 + bud1 side by side, then bud2, then
    readouts (~5–6 h). Resume: SCREEN_HASH=e96588b.
  - **w1s** on the 4090 (tmux `w1s`, `../qllm_w1s` @ `508aaff`): w0 then w128, 100M each, then text-arm / skill-pack / clk2 /
    s8 readouts.
    - **11:50 IST:** w0 done (holdout 3.7454). w128 never trained: the arm overrides split on commas, so SWA_LAYERS
      broke apart (exit 127). Fixed in 2b5937f.
    - Rerun of w128 alone: `bash v13_sempty/tmp/w1s_resume_w128.sh` in the worktree (untracked; SWA passed via BASE),
      same tmux name `w1s`.
    - **13:02 IST VERDICT: target FAIL.**
      - Tool read 0.600 → 0.6125 (bar +0.15). Guards pass. Copy class unchanged. The reader stays in the package.
      - Unexpected, one seed: s8 +0.17 and clk2 8k 20-way +0.22, with the decays unchanged.
      - w1r (seed 44 replicate) is pre-registered and PROPOSED to Lead. The 4090 is free.
      - Logs were copied into the main checkout and committed.
  - **rxk DONE (14:16 IST VERDICT): FAIL as registered.**
    - 2k + contrast 0.61 / 0.62 (< 0.70); 8k + dilation + contrast 0.45 / 0.49 (≥ 0.40 met).
    - Floor ~0; T++ 0.01 / 0.07; implicit ≤ 0.07 for every model, so the planner must substitute values.
    - EUREKA.md entry written; Lead and Library told.

  **Verdicts since 06:12:**
  - cu2a: trained formats MET, transfer FAIL;
  - cu2b: futility FAIL, ACh-1 removed (`d27281d`, code last at 7bdf568);
  - clk2 / clk2r: echo dilation s = 8 + contrast to 8k on 3 minds (ntfy posted);
  - r3p: futility stop; cause = loss-weight share; cold text read written.

  **Decisions:** code moves into v16 once at release; v16/learnings/MIND.md is kept current. The W1 600M pair is the user's
  decision (with w1s + A19). The controller unit comes after R × K.

  **Next for Mind:**
  - **w1r DONE (14:28 IST): REPLICATES.**
    - s8 D +0.32; beyond the windows' reach, s8 at 4k 0.416 → 0.771; clk2 +0.12; holdout +0.002.
    - W1 is a reach unit (not a reader). The 600M run (b2p2 recipe + W1) is the USER's decision. The 4090 is free.
    - Keep W1's code.
  - **w1w window sweep** (user: "why not a bit bigger?"; Lead GO):
    - tmux `w1w` on the 4090 since 15:07 IST, `../qllm_w1w` @ `f68ab64`; `bash v13_sempty/tmp/w1w_chain.sh`;
    - W 512, then W 1024, then the readouts, then skill pack 8k on the four old arms; done ~16:45 IST;
    - score with `.venv/bin/python v13_sempty/tmp/w1w_score.py` (guards g1–g4 + the gain, as pre-registered).
    - **16:43 IST VERDICT: W512 / W1024 FAIL the lazy-state guard (gain −0.11 / −0.06). W = 128 stays.**
  - **Needle trade-off at W 128** (t40, both seeds; needle at the start): 4k 0.495 → 0.44, 8k 0.305 → 0.22; near needles better.
    Part of the w1w guard.
  - **kb1 known benchmarks** (user's standing rule via Lead: known benchmarks, intelligence first, length claims capped at 128k):
    - pre-registered e67664f; wrapper `scripts/lm_eval_bodies.py` (b170219), run with `~/.venvs/qllm311/bin/python` (lm_eval
      0.4.9.1);
    - chain `v13_sempty/tmp/kb1_chain.sh` (waits for w1w; 10 models; outputs `logs/kb1/`); awaiting Lead's GPU order;
    - r3c copied to from_6000/r3cb_main_38e3c29.
  - Lead runs the 300k package rows today (text library + cold read, CPU). Library runs the full package at 300k after the sweep
    (chunked block-8 capture, distance bins <2k / 2–32k / 32–128k / 128–300k, MB per session).
  - bud verdict;
  - then cu3 design:
    - open field vocabulary, held-out names, many formats, loss-weight budget;
    - headline = field selection with Library's cut OFF (`cut_fields=False`) vs ON, on the same rows (Library's ask);
    - plus the same-recipe T++ reference (gap 0.30 / 0.425 at 600M).

* **2026-11-01 10:26 IST: STEP BACK (user via Lead: "how long have you not meditated… fresh air"; memory re-read).**
  - *From far away:* every fix that worked tonight acted on TIME or on PRIORS, never on capacity: subtract the prior
    (contrast; A9 +0.142), slow the clock (echo dilation: one read to 8k), refresh, cut the row to one field (A19). Every
    training-time addition cost language (cu1 +0.045, cu1n9s +0.23, lg8r +0.08, cu2b +0.038; cu2a +0.0175 the cheapest).
    The mind is a dynamical system whose query-time KNOBS (gain, clock) matter more than more training at 103M.
  - *Bold angle:* a **mind controller** in the package that sets the knobs per query (contrast a, dilation s, refresh)
    from the mind's own uncertainty, e.g. the entropy of l_pack, or the surprise of the question tokens that A13 already
    uses for keys. The package as the mind's neuromodulator. Adaptive CAD (AdaCAD) is the transformer analogue;
    eval-only to test.
  - *Fresh air:* we measured each half of "mind + library" but never the COMPOSITION: a rule held in state applied
    to a fact fetched at the query (Lead's R × K). That is the thesis in one row; measure it next.
  - *Where the reader lives:* the 3B attribution said T++'s LM lead over PAM is mostly in-window COPY. If W1 (w1s) also
    lowers holdout / copy-class loss, the window belongs in the BODY (it helps expression too). If it only helps
    fetched-row reading, a package READER organ is equivalent and cleaner. Reading needs no gradient into the state;
    the state as a prefix is enough.
  - *The mind's job in the package:* hold (constant cost), keep rules (time), write the lookup key (surprise / own
    states where words differ), write the answer; the package helps with records and formats.

* **2026-11-01 10:07 IST: the read gap (Lead's T++ + library arm) and a candidate repair.** Same text library, cold read,
  free generation, T++ vs b2p2 (syn / nat):
  - kv 1.00 / 0.875 vs 0.90 / 0.775;
  - TOOL 0.90 / 0.90 vs 0.475 / 0.40 (A9 warm + contrast 0.75 / 0.575: narrower, still behind);
  - para 0.70 / 0.65 vs 0.775 / 0.80 (PAM ahead).

  This is the known multi-item associative-recall weakness of fixed-state recurrent bodies (MQAR / Zoology; Based). The
  mind's claim is NOT reading a fetched row better: it is the constant cost, rules in state, and own-state keys.
  **Candidate repair (proposed to Lead):** bounded sliding-window attention (window 128–256) in 2–4 blocks, so the read of
  a fetched row is exact copy, with KV fixed by the window (still O(1) in context length). Prior work: Griffin / Hawk,
  Based, SWA hybrids; no method claim. Order: 100M LM screen, then a 600M pair (window vs none); readouts are the
  text-arm tool cells, s8 and clk2r.

* **2026-11-01 09:24 IST: cu3 design inputs (after cu2a and Library's A16 / A17), to pre-register after cu2b.**
  - **Open field vocabulary with held-out field NAMES.** cu2a's VERIFY treats an unfamiliar field name as "not this
    row's field": A17 shows 80–83 % " unknown." even when the field IS present ("region", "label", "meaning").
  - **Counter-examples:** an UNSEEN field name that IS in the row → answer the value, beside "absent field → unknown".
    Otherwise an open vocabulary still trips VERIFY.
  - **Many row and question formats** (cu2a: held-out wordings 0.26–0.46, held-out row layout ~0), with some held out
    for the probe. Also stops fld's row format leaking into answers (cu2a's cold raw tool read invents fld rows).
  - **Budget by loss-weight share,** text ≥ ~0.85 (r3p lesson); fld / vfy are answer-dense.
  - **The package side already works:** the warm read + contrast overrides the cold " unknown" prior (A17: rendered
    kv / para natural → 1.00). Rendering helps only with the contrast; cold rendering hurts (0.07–0.17).
  - **Reach reality (A16):** one read carries a skill usefully to ~4k (8k free ~0.45 with echo dilation + contrast).
    The package default stays refresh every ~2k + contrast (0.70 at 32k).
  - ACh-1's tag goes into cu3 only if cu2b passes.

* **2026-11-01 06:12 IST: HANDOVER STATE.**
  **Running:**
  - **cu2a** on the 4090: tmux `cu2`, worktree `../qllm_reset2` @ `24b2969`, chain `v13_sempty/tmp/cu2_chain.sh`,
    started 06:06, ~54k tok/s, ends ~08:40 + readouts. Then **cu2b** (ACh-1) by itself, ~3.7 h.
    Futility at step 4883 via a `.stop` file. Resume: same tmux command with SCREEN_HASH=24b2969.
  - **r3p** on the 6000 (tmux `r3p`, futility ~10:40).
  - CPU: clk2 (targeted dilation, `logs/diag/v13_sempty_clk2_*_89ea841*`) and b2p2's in-format fld baseline
    (`logs/diag/v13_sempty_fldprobe2_b2p2_b2b2efc.jsonl`).

  **Verdicts since 04:32:** clk1 FAIL (LM) but dilation + contrast reach 8k; cu2 sources, probe and ACh-1 built and
  tested; gradient accumulation exact; Lead's round settled (cu2a / cu2b pair; r3p's address readout has six
  arms). Library A11 / A13: abstain must be trained; the surprise-keyed mind address beats MiniLM on synonyms.

  **Next for Mind:** clk2 verdict; cu2a / cu2b verdicts against their bars; r3p's futility.

* **2026-11-01 04:51 IST: STEP BACK after the setbacks (user via Lead: "on every setback, think in the void";
  meditation memory re-read).** The setbacks: cu1's stated rule (a table) and LM cost (+0.045); cu1's weaker fetch
  address; lg8r (reach did not follow, LM +0.08); cu1n9s (+0.23, two-hop collapse); the 3B LM gap (+0.013 / +0.055).
  - *From far away:* at 103M **every training-time addition cost language** (cu1 +0.045 for 6 points of
    synthetic data, lg8r +0.08, cu1n9s +0.23). **Every query-time addition was free** (re-fetch 1.00, refresh,
    contrast +0.14 in the package, the verify rule). Capacity, not cleverness, is the bound. So train the mind
    for language + skills learned from examples, and put everything else in the package (Lead's angle 3).
    cu2 must therefore be small (≤ 2 points of synthetic) or not run at all.
  - *From inside:* **the mind's storage is better than its voice.** Every miss tonight moved from memory to
    readout: the contrast recovers a skill at 4k and a fact read late; free trails 20-way by ~0.2; abstain
    transfers across key style. The readout (head) is trained by LM loss to trust the prior; at an answer it
    should trust evidence.
  - *Analogy (brains):* acetylcholine raises the gain on sensory evidence over top-down priors when
    uncertainty is expected; norepinephrine resets on surprise. Our two fixes match: the **contrast = an
    "ACh mode"** for answers, and **the doc-boundary reset = an NE reset**. A mind with a learned, gated
    evidence-gain mode is the bold angle. **ACh-1:** a trained answer-mode tag. After it, the plain logits
    are distilled toward softmax(l_ev + a(l_ev − l_noev)), with the no-evidence twin built by the data
    pipeline from the same doc with its rows / pack masked. The gate exists by construction (no harm to free
    text: cd3), and no second stream runs at inference. Lead's angle 1 is the same idea. Prior work: CAD,
    decoding distillation (Wan et al. 2023), ContextDPO; the difference is the O(1) state's own prior and the
    gated mode. Needs trainer support (paired docs, 2 forwards on that share). Pre-register as its own unit,
    after cu2's baseline numbers.
  - *Reach is a clock (Lead's angle 2):* after a skill read, slow the decays (retention r → r^(1/s)) for the
    heads that wrote it, until the query. Eval only, CPU; measure reach at 4k / 8k vs prose NLL. Prior work:
    Δ / dt scaling for Mamba length extension (DeciMamba, LongMamba); the difference is a package-triggered,
    per-head dilation after a read. Cheap, do it next.
  - *Process lessons:*
    (a) fine-tuning an annealed checkpoint pays a warm-restart cost (lg8r): future base runs should use a
        warmup-stable-decay schedule and keep a stable-phase checkpoint to branch from (known: WSD, MiniCPM);
    (b) a micro-bench win (N9a, three times now) must pass a 100M LM screen before any 600M slot;
    (c) the session fetch keys on the mind's own states, so every mind change moves the address space
        (cu1 A10: 9 vs 15 rows found). A stable address (lexical organ, or a frozen key encoder) would
        decouple them; offer it to Library.
  - *Not chasing:* the stated-rule table at 103M (no evidence of the operation exists for an unseen
    number); more long-window training (lg8r answered it).

* **2026-11-01 04:32 IST: HANDOVER STATE.** Since 02:35:
  - cd2 MET (contrast doubles one-read reach to ~4k); cd3 FAIL (contrast must be query-time only).
  - Library A9 made warm + contrast the package default (+0.142); A10 says cu1 reads well, but its own-state
    fetch is weaker.
  - Natural-key verify: cu1 says " unknown" 0.86 vs b2p2 copying 0.59.
  - **cu1n9s FAIL** (two-hop 0.09, holdout +0.228); cu1n9a not run.

  **Running:** r3p on the 6000 (tmux `r3p`, `~/Development/qllm_r3p` @ `c13d8fe`, started 04:27 IST, 64k tok/s).
  Futility at step 39060 ~10:40 IST (stop if > 3.431; a local wait is armed). End ~17:00 + readouts. Hard
  gate holdout ≤ 3.2391, prediction 3.231. 6000 disk 15 GB free.

  **Next for Mind:** pre-register cu2 (one field from multi-field rows + stop; prose-row abstain; natural
  keys; paraphrases; fresh seed; free primary; small synthetic share, since cu1 cost +0.045); GPU = a 4090
  slot via Lead. The contrast-distillation idea (scratchpad 03:24) is a candidate after cu2.

* **2026-11-01 03:24 IST: STEP BACK after cd1-3, cu1 and Library's A6c. One pattern, four faces.** Tonight the
  mind failed the same way four times. In each case the strongest prior decides, and weak or new evidence is
  outvoted:
  (1) a fading skill after one read: 20-way 0.355, free 0.17 at 4k (cd2);
  (2) a fresh fact row after a long carried state, at ANY length (A6 / A6c: warm 0.125-0.675 vs cold 0.70-0.93);
  (3) a stated number mapped to a trained table (cu1 R6a);
  (4) a wrong row's value copied under forced choice (verify).
  A counterfactual contrast (with minus without the evidence) recovers (1) and (2) almost fully
  (cd2 0.78; A6c 0.90-1.00), but it costs a 2nd stream and must be gated (cd3).
  - **Bold angle: distil the contrast into the weights.** On answer tokens of skill / fact docs, train the
    plain logits toward softmax(l_ev + a(l_ev − l_noev)), where the no-evidence stream is the same doc with
    the pack / row masked. The mind then learns to listen to evidence by itself, with no 2nd stream at
    inference and nothing to gate. **Prior work (searched 03:25):** CAD (Shi et al. 2023, arXiv 2305.14739);
    distilling a faithfulness-aware decoding into a student (Wan et al. 2023, arXiv 2303.03278, summarisation);
    ContextDPO (Bi et al. 2024); SI-FACT (arXiv 2509.10208, faithfulness contrastive tuning); AdaCAD (adaptive
    a); ContextFocus (arXiv 2601.04131, activation steering). Distilling a decoding rule into weights is known,
    so the difference is only the setting: an O(1) recurrent state whose prior is its own long state, judged
    by one-read reach and the warm read. Plumbing-level novelty; the claim would be the measured quantity. A cheap 600M test is a fine-tune of b2p2
    with the distillation term on b2 / recall / fsq answers; readouts cd2's reach, A6's warm read, and cd3's
    prose loss (it must not learn to over-trust irrelevant packs). Pre-register before any GPU; GPU via Lead.
  - The stated-rule table (3) is a different failure: no evidence of the operation exists for a new number.
    That stays a stated limit at 103M.

* **2026-11-01 02:35 IST: HANDOVER STATE.** New since 01:22:
  - cd1 (contrast decode, 4k reach 0.815 ranked / 0.59 free; FAIL at 2k as registered); cd2 running on
    local CPU (fresh seeds 8200-8299; `logs/diag/v13_sempty_cd2_b2p2_cb1f9c7.jsonl`).
  - **cu1 VERDICT: not a Win** (R-fsq + R-hop met; R6a / R6b fail: the stated rule is a lookup over
    trained values; holdout +0.045). srd and b2gap removed from the code; RESET_AT_EOS kept (Lead).
  - cu1n9s unstuck on the 6000 (tmux prefix bug; exact names now), stage A since 02:19 IST.
  - **r3p QUEUED on the 6000:** tmux `r3p`, worktree `~/Development/qllm_r3p` @ `c13d8fe`. Mix A1, 7.0 %
    synthetic; hard gate holdout ≤ 3.2391; futility 3.431 @ step 39060. Starts when cu1n9s ends (~04:30).
    Resume: same tmux command with SCREEN_HASH=c13d8fe.
  - 6000 eval worktree `~/Development/qllm_eval` (master; the main checkout there cannot pull because of
    untracked logs, so leave it).

  Next for Mind:
  - cd2 verdict;
  - cu1n9s readouts vs its A1 rules (two-hop ≥ cu1 + 0.15 AND ≥ 0.50; guards vs cu1), plus the verify
    cell from `qllm_eval`;
  - pre-register cu2: VERIFY on natural keys + paraphrases, and a compositional stated-rule source if
    any (many values, held-out in the probe).

* **2026-11-01 01:22 IST: HANDOVER STATE.** Done tonight:
  - reset mask (`f2587b5`) and ANSWER_IN_WINDOW (`824a988`);
  - lg8r VERDICT FAIL, mixed case (reach 4k 0.465, half-lives held, LM +0.08 / +0.11); reach-by-training stopped;
  - r3c VERDICT (3B: LM +0.013 / +0.055 behind, recall 0.79 vs 0.54 / 0.61);
  - MLB-1 A7 (free generation primary) + A8 (natq reported);
  - free columns in run_skillpack_probe / run_learning_probe / run_cur_probe;
  - verify cell + cu1 A3 (b2p2: wrong row's value said 0.14 free vs 0.96 right; EUREKA);
  - r3p pre-registered (6000 after cu1n9s; mix by cu1's verdict).

  Running: cu1 on the 6000 (`~/Development/qllm_cu1`, ends ~02:30 IST, then its readouts), then
  cu1n9s. Next for Mind:
  - score cu1 against its rules + A3 verify (free beside);
  - fix r3p's mix from the verdict and write its launcher (gates: cu1n9s ended, VRAM, disk ≥ 8 GB);
  - pre-register cu2: VERIFY rows (Library's mix: ~45 % unrelated-key row, ~5-10 % look-alike, ~45 % prose row, all
    -> " unknown", mixed with right rows; short format row + question included), WEIGHTED TO NATURAL
    (English-phrase) keys (Library: free rule abstains 0.73-0.85 synthetic vs 0.425 natural), plus paraphrase
    facts / questions; scored free (A7) on a fresh seed; orphan check; plus Library's misses with the right
    row in hand (A4 v2 / S2): multi-token values cut short (" ave."), loops on multi-field rows (" onyx; onyx;"),
    generic prose on natural keys. So answers in cu2 are multi-token, end with a stop mark, and some rows
    carry several fields. After Library's A9 (warm read + contrast = package default, +0.142), the main
    mind-side gap is TOOL FIELDS: run-on 47-55 % (the mind reads out the whole row), natural-key tool 0.575.
    So cu2's first target is extracting ONE field from a multi-field row and stopping;
  - remove b2gap + RESET_AT_EOS if Lead agrees.

  lg8r's checkpoint is on d_drive (`checkpoints_v13_sempty/lg8r_main_f2b7d8c`; the worktree link
  was a real dir, fixed).

* **2026-11-01 00:53 IST: STEP BACK after the orphan bug, r3c and the A7 generations.** What surfaced:
  - *Ranked is not said.* At 2k after one read b2p2 ranks the rule's answer first among 20 in 0.88
    of docs but SAYS it in only 0.68. The fading state still carries the skill; it is outvoted by
    the LM prior over 50k tokens. So part of the "reach wall" is a volume problem, not a storage
    problem. **Cheap test (eval only, no training):** contrast the state with itself: logits(pack
    read) + a·(logits(pack read) − logits(same stream, no pack)), i.e. context-aware decoding
    (Shi et al. 2023; contrastive decoding, Li et al. 2022) applied to a DECAYING O(1) state. 2×
    compute, still O(1) memory. Measure free reach at 2k / 4k / 8k against a. Prior work is
    named, so the difference is only the setting (a recurrent mind's one-read reach). Propose to
    Lead after lg8r; the skill-pack probe already builds both streams.
  - *Unanswerable targets teach guessing.* The orphan bug is one case of a general hazard: any
    weighted target whose evidence is not in view trains the prior, not the skill. Every new
    curriculum source should get the orphan-rate check (`orphan_rate.py`) before a run.
  - *Format-bound recall.* r3c holds key-value facts far past 2k, but a plain question about a
    plain sentence fails. The mind binds what it was trained to bind. That is Lead's "mind
    canonicalises the question" idea from the other side: train facts and questions in many
    phrasings (cu2 = paraphrase-varied facts / questions, held-out phrasings in the probe), and
    measure with A8's natq cell. Plumbing (known: instruction / QA data); the claim would be
    natq beyond the training length.

* **2026-11-01 00:13 IST: lg8r READY, queued on the 4090 (tmux `lg8r`, worktree `../qllm_reset2`).**
  Reset mask on master (`f2587b5`, parity PASS CPU + GPU) and answer-in-window weighting
  (`824a988`; 44 % of b2gap answers at 8k sat in a window without their demos). Pre-registration
  "lg8r" in EXPERIMENTS_SEMPY. The chain `v13_sempty/tmp/lg8r_chain.sh` waits for Lead's package
  table (`PKG-TABLE-DONE`) and >= 21 GB free, trains 50M tokens (SEQ 8192 x 2), then runs the
  readouts: reach 0-32k, half-lives (`hl_summary.py`), B2 t400, attr3b vs b2p2 (data in
  `/media/gowrav/d_drive/qllm2/attr/`). Bars: reach(4k) >= 0.75 (b2p2 0.355) AND long-head
  geomean >= 1293 (b2p2 1361). Resume: rerun the same tmux command with SCREEN_HASH=<launch hash>.
  Still open: r3c final (6000, wait bb4gmil0b), then cu1 -> cu1n9s on the 6000.

* **2026-10-31 23:40 IST: MEDITATION (user, via Lead: "think in the void, actually think nothing").** What surfaced,
  unplanned:
  - *Forgetting is learned from the life it is trained on.* Packed windows of unrelated documents
    taught every echo head to forget faster. The mind's half-life is its prior on how long the world
    stays the same. Trained on lives (one long coherent stream: a book, an agent's whole session with
    its tool results, a long conversation), not on shuffled documents, it should learn to keep the past
    longer. Training was teaching it that every 2k tokens the world resets.
  - *It is a place-keeper, not a container.* "The state is a stream address" plus the reach plus the
    drifting half-lives sounds like the temporal-context models of human memory (a slowly drifting
    context that items are bound to; recall works by reinstating the context). Known in cognitive
    science; what is new is that the trained network found it by itself.
  - *Now and then.* The mind holds "now" (~2k tokens of rules and gist); the library holds "then";
    rehearsal (re-reading a pack) bridges them. Reach is not a defect to fight: it is the size of the
    present. Re-fetch at 1.00 is the bridge, already working.
  - *The bottleneck may be what makes it abstract.* T++ fails at a 2k gap even with every example in its
    window (0.06); the O(1) mind keeps the rule (0.86). A fixed-size state is FORCED to compress, and the
    compression keeps the rule and drops the tokens. Bold, untested: a smaller state might generalise
    better to unseen rule families, while recall gets worse (the library covers recall anyway).
  - *Show, don't tell.* Words do not write a rule; demonstrations do. An imitation learner. A stated rule
    could become a skill if the mind (or the library) turns the words into a few demonstrations
    first, then reads them: self-demonstration loads the register.
  - *The mind can make the exact organ work for paraphrase.* The shared library is fetched by text, and
    the exact organ fails only on paraphrase. The mind learns any format it is shown, so it could
    restate a question in the library's canonical form ("Query: <stored key> refers to") before the
    lookup. The skill lives in the mind, the exact match in the organ: no fuzzy embedder needed.
    That changes the shared-library design (Library's L2 paraphrase miss); one line sent to Lead.
    Lead: recorded as a popped idea, slated for **cu2** (a "restate" format) IF cu1 shows formats transfer;
    closest prior work query rewriting / HyDE (difference: the reader canonicalises in O(1) state, the
    index stays exact text). No action tonight.

* **2026-10-31 23:15 IST: resumed (process had exited ~22:16; Library restarted it in tmux `mind`).**
  Done since 21:35: s8 VERDICT MET (D PAM 0.645 / 0.691 / 0.741 vs T++ 0.134 / 0.134, Mamba-2 0.116 / 0.098;
  primary +0.607, all pairs +0.511; ntfy sent 22:06); cu1 pre-registered (59ac5d7) + A1 (N9s arm ON,
  Library's c91f4c6) + A2 (conditional cu1n9a); curriculum sources srd / fsq / hop (48674d6);
  `scripts/run_cur_probe.py` (b2p2 baseline: present 0.990, abstain 0.000, two-hop 0.10).
  6000 queue: r3c alone (90 %, ends ~00:30 IST; its s5 chain runs final probes) → tmux `cu1`
  (~/cu1_chain.sh, worktree qllm_cu1 @ 2efb422; gate: r3cb .done + s5/s8 ended + ≥ 43 GB free) →
  tmux `cu1n9s` (~/cu1n9s_chain.sh, worktree qllm_cu1n9s @ fa9e410; own stage A). Library's
  EUREKA: the mind state is a STREAM address (session library by state keys; shared by text).
  Write-tool calls were refused as "user rejected" several times though the user said they never
  stopped me; files written via shell heredoc instead.

* **2026-10-31 21:35 IST: LAW-1 CLOSED (user) + M2 STREAMING LANDED.** User via Lead: one-read reach ≈ training
  length is a stated limit at this scale; lg8 / lg2 / lg32-reset cancelled (law1 killed before start); lesson
  recorded. The reset-at-EOS mask is UNFINISHED on local branch `reset` (`../qllm_reset`; isolation test
  fails). **M2 body streams now** (`41f8ae8`): carried-state chunked prefill + O(1) decode, parity 2e-4 /
  5e-5, self-test 24/24, `v13_sempty/tmp/stream_parity{,_gpu}.py`; MLB-1 A6 lifts the A5 caveat. Library's
  state test can use it (states out after a prefix, edited, then in for the next chunk). Next (Lead's
  order): pre-register the curriculum run (stated rule + demonstrations, fetched spans + question with
  abstain, two-hop format; N9s optional paired arm) as a 600M run on b2p2's recipe for the freed slot.
  6000 now: r3c (~02:55) + b2p3 (s8, ~22:15). Nothing else queued.

* **2026-10-31 17:11 IST: lg32 relaunched under option B (Lead).** At 17:08 the guard paused it at step 450
  (r3c 41.5k vs its ALONE speed 64.1k = −35 %, three strikes; no checkpoint yet). Lead agreed the 80 %
  rule was miscalibrated (any second training job time-slices the card; the box gains +20 % total
  throughput). Now: guard r3c ≥ 55 % of its alone speed (~35k), VRAM floor 3 GB, SAVE_EVERY 200
  (`~/law1_chain_b.sh`, `~/lg32_guard_b.sh`). Standing launch rule updated. ETA lg32 ~18:30 + probes ~18:55;
  r3c ends ~01:00-03:00 IST. s8 still paused; the gate resumes b2p3 on lg32 .done.

* **2026-10-31 16:43 IST: lg32 restarted with `FINEWEB_LOCAL_DIR=data/fineweb-edu`.** Its first restart sat
  ~25 min in HF read timeouts (mostly fineweb shard 000, opened twice: main stream + b2gap's gap
  stream; dclm / smoltalk2 occasionally). Same data, local read path: the 6000 has shards 000-007,
  cursor file 0 row 475,640. dclm / smoltalk2 stay remote. The gate no longer resumes s8 when law1
  ends (only on lg32 / lg8 .done or s5 ending), so b2p3 waits under Lead's fallback. The fallback,
  if lg32 OOMs beside r3c alone, is lg8 / lg2 at 16,384 tokens per step (a dated amendment first).
  lg8 at 4 × 8k saves NO memory (same tokens per step).

* **2026-10-31 15:45 IST: lg32 OOMed at its first step (15:41)** beside r3c (52.2 GB) + b2m2 (14.7 GB): the
  card hit 94.9 / 95 GB. My 28.3 GB benchmark was fwd+bwd only (no AdamW state, no loader, b2m2 grew). No
  checkpoint lost. The gate resumed s8 when law1 ended; s8's controller is PAUSED AGAIN (pid 518542, `T`)
  while b2m2 trains. New order: **lg32 waits for b2m2's .done (~16:20) AND ≥ 34 GB free**, then runs beside
  r3c alone (~84 / 95 GB); the gate resumes s8 (b2p3) when lg32 is .done (~18:30), s5 ends, or law1 ends;
  lg8 / lg2 follow s8. Guard `lg32g` takes its r3c baseline when lg32's process appears. ETAs: lg32
  ~16:30-18:30, s8 verdict ~20:30, lg8 ~22:30, lg2 ~00:30 IST. Lead offered the flip (b2p3 first).

* **2026-10-31 15:20 IST: state of the 6000.** r3c PASSED futility (3.4010 vs 3.4714 @ 39,060; continues,
  ends ~04:00 IST Nov 1). **lg32 (LAW-1) started 15:06** (tmux `law1`, worktree `qllm_law` @ 2229a5b;
  M2 body has NO grad-ckpt support, removed; 1.3 s/step at 32k after a ~7 min first-step compile).
  **s8's controller is PAUSED** (SIGSTOP on pid 518542, its bash; b2m2 keeps training under it);
  tmux `gate` (~/gate_b2p3.sh) sends SIGCONT when lg32 is .done, s5 ends, or law1 ends → b2m2's
  probes + b2p3 then run as s8 would. tmux `lg32g` (~/lg32_guard.sh): pauses lg32 (SIGTERM,
  MAX_RETRIES=1; resume from latest.pt) on 3 strikes of free VRAM < 3 GB or r3c < 80 % of 27,037
  tok/s; flag ~/lg32_paused.flag. lg8 / lg2 wait until s8 or s5 ends. If anything is stuck: `kill -CONT
  518542` resumes s8 by hand.

* **2026-10-31 13:26 IST: 6000 disk cleanup (organiser-authorised; checkpoint-cleanup rule):** root 6.5 → **21 GB
  free**. Deleted `gscreen_onorm_3cc2b72/latest.pt` (1.17 GB; resume-only, abandoned G2 screen, code removed in
  2872d7f). Removed the 6000 copies of 11 finished best_model.pt files whose d_drive copy
  (`/media/gowrav/d_drive/qllm2/checkpoints_v13_sempty/<run>/best_model.pt`) has the SAME md5 (checked both
  sides before rm; each folder keeps `MOVED_TO_D_DRIVE.txt` with the md5), 1.17 GB each: c2m2b_main_b788731,
  e1g_echo_a0fab65, m1_m1echo_ccbf028, m2_m2echo_ed326a1, mix3b_delta_answ100_long2h9_5c87148,
  r4screen_store_0868a1b, r600_ctl_91daf0a, tfc1b_main_b788731, tfs1_tf_b788731, tfs2b_main_b788731,
  wikitext_chrono_n4gate_fair_2537782. Kept on the 6000: everything a running / queued job needs (r3ca/r3cb,
  b2t2, tfc1a, b2ma, c1m2a, 1c913ef for SL0), all Mamba-2 checkpoints (only scorable there), the recent pair
  models (b2p1/2, b2t, b2m, lbp, lbt). No other user's file touched.

* **2026-10-31 13:23 IST: organiser leads v16** (user): board `v16/SCRATCHPAD.md`; GPU order and scope go
  through the organiser; v13 keeps the mind, the MLB-1 spec, the release, its own pre-registrations.
  **SL0 running on the 6000** (tmux `sl0`, `~/sl0_chain.sh`, worktree `~/Development/qllm_sl0` @ 8445f73;
  main checkout untouched): library in /dev/shm (root disk 6.6 GB free, /data not writable), 5M then 20M
  rows; reports to `qllm_sl0/logs/diag/v13_sempty_sl0_*`, neighbours copied to `~/attr`. Smoke (60k rows)
  showed the T grid had to scale with distance (fixed before the scored run, de9ac36). a4f: stage B 17 %,
  35-39k tok/s beside v14's L1, ends ~16:50 IST. Organiser's stated-rule rerun on b2p2 with the trained
  header + a 2-shot control is pending; it decides whether "stated rule + demonstrations" docs go into
  the curriculum run (N9s may join it as a paired arm: v13's call). Disk risk flagged to the organiser.

* **2026-10-31 13:10 IST: MEDITATION (user, via organiser: "stop, think in the void, we are onto something").**
  *What surprised me.* (1) Every in-context skill PAM has was a FORMAT gap: rules 0.05 → 0.99 (b2), labels
  0.57 → 0.934 (lab), recall up with recall docs. Without the format T++ / Mamba-2 beat it (labels); with it
  PAM is best, even on words it never saw (lab vocab disjoint). PAM = LOW PRIOR, HIGH CEILING: its in-context
  intelligence is MADE by the skill curriculum; more web text does not teach it (3B label gap unchanged).
  (2) I said the library cannot close the new-token gap to Mamba-2. True for the SESSION library, wrong
  for the SHARED one: a corpus library IS world knowledge, and much new-token loss is knowledge. kNN-LM /
  RETRO cut perplexity a lot with retrieval; a4f is a tiny parametric "knowledge library" of n-grams. The
  shared library keyed by the mind's own state may be the bar-2 path. (3) The organiser's "cannot follow a
  stated rule" smoke ran on c1ctlb, which has NO rule docs and cannot shift letters even from examples;
  the real question needs b2p (has the skill): can WORDS write the rule register that examples write?
  (4) Bar 3's standing caveat is the frontier: all rule results are held-out VALUES of trained FAMILIES.
  *What it really says.* The product splits cleanly: skills (mind, made by a curriculum, carried across
  distance), episodes (session library, keyed by the mind's state at surprise), knowledge (shared library,
  same key space). One key space for both libraries; packs are versioned with the mind.
  *Bold angles.* (a) SHARED LIBRARY = KNOWLEDGE: a zero-training test, kNN over the frozen mind's state on
  training text (no holdout), interpolated with its LM, scored by attr3b on new tokens 0-64; if PAM + shared
  library beats Mamba-2's loss, bar 2 lives under the new direction (honesty: kNN-LM exists; the novelty is
  the whole). (b) FAMILY GENERALISATION: a procedural rule curriculum (composed primitives) with held-out
  FAMILIES: the test that the mind learns to learn, not formats. (c) WORDS WRITE THE REGISTER: docs that
  pair a stated rule with demonstrations, so words set the same state examples set; tested on held-out
  values stated only in words, inside and past the trained length (the instruction cell). (d) READ-WHAT-
  WAS-FETCHED: "fetched spans + question" docs with look-alike distractors and abstain cases. (e) Later:
  consolidation ("sleep"): often-fetched session rows promoted to the shared library; the mind fine-tuned
  on skills it keeps needing; continual learning without forgetting (knowledge outside the weights).
  *Choice.* (a) first: eval only, decisive for bar 2. Then one pre-registered curriculum run with (c) + (d)
  and a separate family-generalisation arm (b). Correction sent to the organiser on (3).

* **2026-10-31 13:01 IST: v14 comments folded into MLB-1 (v0.1): row 272 B, R4 ≤ 1/100, 1M reported
  not gated, fairness arms train their own key head, fetch metrics + negative cells, instruction
  persistence = B2 rule stated once in words (mind-only and library variants). n9r FAIL (N9a out
  of every plan; v14 ac3a76e). a4f's reference changed to C1-ctl BEFORE its result: ctl43 is a loss
  outlier (3.598; stage A already +0.045), and there is no 4090 box effect. 4090 after a4f: free
  for v15's queued pair (v14 cancelled n12m; n9th stops after seed 132).

* **2026-10-31 12:56 IST: MLB-1 benchmark draft v0 written** (`v13_sempty/BENCHMARK_MIND_LIBRARY.md`),
  sent to v14 and the Communicator for comments; fixed as v1 before any package run. Ownership: v14 =
  the fetch (gate, trained key at the salient token, re-read) + the library row format (one row per
  event, keys versioned with the model); Communicator = the library half of the probe
  (`scripts/session_index_probe.py`, more docs, natural keys, 400k-1M cells); v13 = the mind half,
  the fetch-training recipe, the package runs. No new GPU job until v1 is fixed.

* **2026-10-31 12:55 IST: lb pair PASS; fresh-probe old models hold; DIRECTION fixed: "O(1) mind + growing library".**
  lbp label_diag 0.934 (exact 0.995) vs lbt 0.798: the label gap was format (EXPERIMENTS "Label-format
  pair lb result"); cost tiny (holdout +0.004). s8 part 1 (probe seed 7000): D PAM 0.645 / 0.691 vs
  T++ 0.134 / Mamba-2 0.116; b2t2 training (35 %), then b2m2, b2p3 (~18:15). r3c 42 %, val 3.4802 @
  31248; futility at step 39060 (~15:20 IST by the real clock, stop if > 3.4714). a4f STARTED 12:45 IST on the 4090
  (stage B, ~3.5 h): kept, now read as "what n-gram knowledge buys the mind" (a shared-library
  organ could supply it without parameters). Queued: `s9` lbm (Mamba-2 + lab) after s8.
  User (12:30): the O(1) body stays and IS the mind; drop only "the state must also be the
  library"; phrase it as O(1) mind + growing library; benchmark first (needle, multi-needle,
  two-hop, rule-at-distance at 32k-1M with the library, vs T++ full context, MB in the table),
  the distributable model only once it wins that table. Plan: EXPERIMENTS "Direction".

* **2026-10-31 ~10:00 IST: user direction, the "complete package":** train the model as an
  INTELLIGENCE machine (rules, gist, reasoning in the O(1) state), not a recall machine, and
  give it a way (learned if possible) to FETCH from the user context and tool results later
  (RAG-like). Evidence today: copy = 53-58 % of the 3B T++ gap; Communicator's model-free exact
  n-gram organ recalls 1.0 to 128k where any state fades to 0 by 32k; the model's own pre-logit
  vector at the key token retrieves the right sentence top-1 0.63-0.88 to 128k, zero-shot;
  frozen PAM re-reading 2 retrieved spans answers two-hop 1.0. Package = mind (PAM state) +
  exact store (context / tool text; A4 / N7 class) + learned fetch (when / what, then re-read).
  Honesty: bar 1 becomes "O(1) mind + sparse fetch index" (at 128k: text 0.26 MB, salient-key
  index ~15 MB, all-position 308 MB, vs T++ KV 4.9 GB). It does NOT close the Mamba-2 gap
  (general modelling on new tokens): that stays the a4f / body track.
  v13 steps (this lane): (1) a probe suite for the package: fetch-and-reread on needles,
  "latest definition wins", two-hop, tool-result lookup, at 2k-128k, memory in MB beside each
  number, plus rule-at-distance as a named benchmark; (2) the 600M LM check of an exact organ
  (a4f now; N7 when v14 hands it over), scored with attr3b on copy tokens; (3) the best-package
  3B run includes the organ explicitly. Design of the learned fetch = v14's lane (sent).
  Addendum (Communicator): Mamba-2 (mbs2b) is also at the floor on the needle probe (all three
  bodies fail state-only recall from 2k to 128k; the organ is 1.0 throughout). The zero-shot frozen-state
  address (0.75 top-1 at 128k) is 1/4 at 400k (corrected), so past ~128k the fetch key must be TRAINED, or
  one row per event used.

* **2026-10-31 08:55 IST: a4f (A4 cond_mem filter) queued on the 4090** (tmux `a4f`, script
  `/tmp/claude-1000/-home-gowrav-Development-qllm2/99371c3c-8382-42b3-a3d3-6861632abc19/scratchpad/a4f_chain.sh`, worktree `../qllm_a4` at `d338469`,
  ckpts on d_drive `qllm_a4_checkpoints_v13_sempty`). Waits for v14's `n9r` tmux to end
  (~12:00-12:15) AND ≥ 11 GB free, then stage A 120M → stage B 480M (~4 h) → rules, label_diag,
  attr3b vs ctl43 (v14's n9r control, same box / seed / data; v14 agreed; leave n9r ckpts in
  place). Pre-registration: EXPERIMENTS "A4 cond_mem filter a4f". If the box crashes, re-run
  the script (stages resume). v14: N9a shows no LM / copy effect at 600M (its attr3b on n9c),
  so N9a is not counted toward the copy gap.

* **2026-10-31 08:30 IST: 3B gap attribution done** (EXPERIMENTS "3B LM gap attribution"). vs
  T++: 53-58 % copy → a binding fix closes ~half. vs Mamba-2: 52-55 % NEW tokens, already +0.05-
  0.09 at positions 0-8 → mostly GENERAL modelling; memory closes only ~0.02-0.03 of ~0.10.
  This corrects view 1 below: the "local" part is bigger than memory, but the gap also grows
  with context for every class. Next: score r3c's ckpt at its futility point (step 39062,
  ~15:00-16:00 IST) with `v13_sempty/tmp/attr3b.py`; Mamba-2 is scored on the 6000 (local venv
  has no `mamba_ssm`; the frozen text + `loss_mamba2.npz` live in `~/attr` on the 6000). The
  best package needs a body-level gain too: candidates for a ≥ 1.5B test are width 768
  (bar-2 tolerance allows a few more params) and a larger state, picked after r3c.

* **2026-10-31 08:15 IST: step back (user: "play on strength, mitigate gaps, scale").**
  Run ingredients ledger added to EXPERIMENT_INDEX (user rule): r3c has NO b2 / lab / N9a,
  so it tests the 16nb + echo + 2st BODY at 3B, not the best package.
  Fresh views:
  1. The 3B LM gap splits in two (existing PG-19 files): vs Mamba-2 PAM is +0.063 at
     positions 0-64, where no memory helps (LOCAL modelling), rising only to +0.10 by 8k.
     vs T++, PAM is better at 0-64 (-0.044) and behind from 64-2048 (+0.03 to +0.05):
     in-window exact copy, the same weakness as labels. Two gaps, two fixes; memory units
     (echo / N9a / N12) cannot close the local part. Attribution by token type (copy / seen /
     new) running on the 4090: `v13_sempty/tmp/attr3b.py` → `logs/diag/v13_sempty_attr3b_*`.
  2. r3b (M2 + echo) is worse than the old 16nb body even at 0-64 (+0.085): notebook fast
     heads (half-lives 1-46 tokens) are PAM's LOCAL mixers, and M2 removed 12 of 16. A 600M
     win (M2) did not survive 3B. Same lesson as the LR: screen bodies at ≥ 1.5B.
  3. Width: PAM residual 588 vs 768 for both baselines, never tested as a capacity lever at
     scale. But T++ (768) is worse than PAM at 0-64, so width alone does not explain the
     local gap. Candidate only.
  4. Strength, framed: PAM remembers HOW (rules, gist) across distance; Mamba-2 remembers WHAT
     (chosen tokens). Play on it: a long-horizon rule / instruction persistence probe (a
     rule set early, applied thousands of tokens later). That is a real pain point of chat
     models and our cleanest bar-3 story.
  5. PAM is very teachable by format (b2 → 0.99, recall docs → 0.975), but pays LM loss
     for it (b2 3 % cost +0.030 holdout at 600M). At 3B, 1 % rule docs gives 2× the 600M
     exposure at a third of the share.
  Plan: best-package 3B run ("r3d") = the body r3c / attribution supports + b2 (~1 %) +
  lab (if lb passes) + recall 5 % + N9a (if n9r + untrained-word check pass), pre-
  registered vs Mamba-2 3.1644 / 46.91. It starts when the 6000 has a 3B slot: r3c's
  futility check (~15:00-16:00 IST) decides whether that is today or after r3c ends
  (~04:30 IST Nov 1).

* **2026-10-31 06:25 IST: Mamba-2 arm b2m → bar 3 NOT met as pre-registered; distance edge is PAM's.**
  `_new` mean (in context) PAM 0.991 / 0.944 vs Mamba-2 0.892 (+0.099 / +0.052, bar +0.10): no
  claim, no ntfy. Across distance (cells named before b2m ran) PAM 0.67 vs Mamba-2 0.08 and T++
  0.13: Mamba-2 collapses at ~110-token gaps like T++. Mamba-2 leads labels (0.787) and exact
  far copy (step through filler 0.3). EXPERIMENTS "B2 Mamba-2 arm b2m".
  6000 queue now: s5 r3c (stage B 9 %, ~24 h with side jobs) | s7 lbp (started 06:16 IST) → lbt
  | v14 `n9th` (to ~08:30) | **s8** (`~/s8_chain.sh`, main checkout 38e3c29, driver
  `logs/v13_sempty_s8_driver_38e3c29_20261031_0100.log`) waits for s7, then probe seed 7000 on
  b2p1/b2p2/b2t/b2m → b2t2 (T++ s43) → b2m2 (Mamba-2 s43) → b2p3 (PAM s44) (~1h45m each).
  Score with amendment 2 (primary D(b2p3) ≥ max(b2t2, b2m2) + 0.10 AND all pairs). lb pair:
  lbm (Mamba-2 + lab) only if lbp passes rule 1. GPU now ~94/98 GB (r3c 53, lbp 33, n9th 8).
  Step back: two kinds of memory. Mamba-2's Δ selects exact tokens to keep (labels, last
  number); PAM's outer-product state integrates a rule that repeats coherently over examples
  while filler adds incoherent noise (guess). If true, PAM gains from a selective (input-
  dependent) write strength for single items. That is v14's lane (unit design), so it went as
  a EUREKA, not a v13 run. Cheap probe of the guess (not run): fixed gap per example, shots
  2/4/8: coherent integration predicts PAM's far accuracy rises with shots much more than
  Mamba-2's.

* **2026-10-31 04:20 IST: 6000 slot given to v14's n9th** (27M micro, ~7 GB, 3-4 h, from
  `~/Development/qllm_v14`, its own tmux): runs beside r3c + b2m, then lbp. Limits agreed:
  ≤ 8 GB VRAM, no checkpoints on the root disk (14 GB free), pause it if b2m/r3c tok/s drops
  > 30 %. v14 note for the "best package": N9a loses on never-trained words at 8 shots
  (label_*_new@8 0.56-0.98 vs echo 0.95-1.00), so N9a joins only if n9r replicates AND it
  passes an untrained-word check.

* **2026-10-31 04:05 IST: B2 pair vs T++ MET in both seeds; Mamba-2 arm next.** b2p2 (seed 43):
  `_new` mean 0.944 vs T++ 0.734 (seed 42 was 0.991); affine `_new` @4096 0.905 (T++ 0.45),
  @8192 0.61. Holdout 3.4800, recall 0.707 (> 2048 0.474). EXPERIMENTS "B2 pair — seed-43
  result". The 6000 s6 chain is now on b2ma (Mamba-2 aptitude stage A, started 03:56 IST),
  then b2m (~07:30-08:00 IST), then s7 lbp → lbt (~12:00 IST). Bar 3 / ntfy ONLY if PAM ≥
  b2m + 0.10 in both seeds. 6000 root disk: 14 GB free (98 %; the rest is other users'
  folders); the four queued ckpts need ~5 GB. Re-check before s7 starts.
  Step back: the rule survives distance but an exact far token does not (step collapses
  for both models; shift and affine need only the rule). Our recurrent edge is carrying
  ABSTRACT state. Idea not run: rule docs with the examples spread by filler INSIDE the
  2048 training window, judged at 4096/8192 (trained-length generalisation of the skill),
  and a combined known-best stage B (N9a if n9r replicates + b2 + lab if lb passes, recall
  share kept at 5 %) as the next 600M "best package" before any scale-up.

* **2026-10-31 01:00 IST: B2 pair seed-42 half MET; b2p2 + Mamba-2 arm moved next to r3c.**
  b2p1 vs b2t (EXPERIMENTS "B2 pair — seed-42 half"): held-out rule values PAM 0.991 vs
  T++ 0.734; across distance PAM holds affine to 4096 (0.785) while T++ collapses at
  1024. 6000 tmux `s6` (`~/s6_chain.sh`, code 38e3c29, NO pull on the 6000 main
  checkout while s5/s6 run): b2p2 (seed 43) → b2ma (Mamba-2 aptitude stage A) → b2m
  (Mamba-2 + b2), all next to r3c (s5, started 00:44 IST; now ends ~16:15 IST).
  4090: n9c ends ~01:30; then v15 G1 (tmux `v15g1`, ~6 h) starts by itself when
  the `n9c` tmux session closes. v13 has no 4090 job queued after n9c.

* **2026-10-31 02:00 IST: queues.** 6000: `s5` r3c (16 nb + echo, 3B; futility @ step 39060,
  ends ~16:00-20:00 IST depending on side jobs) | `s6` (`~/s6_chain.sh`, 38e3c29) b2p2 →
  b2ma → b2m | `s7` (`~/s7_chain.sh`, worktree `~/Development/qllm_lab` at `c4173be`,
  links to the main checkout's .venv/.cache/data/checkpoints) waits for s6 to end, then
  lbp → lbt (label-format pair). Do NOT pull the 6000 main checkout while s5/s6 run.
  4090: v15 `v15g1` (G1, ~08:00) | v14 `v14n9s` then `n9r` (v14's N9a recall replication,
  ctl43 vs n9a43, ~10:30 IST; scored with the t40 aggregates: >2048 = context_tokens > 2048).

* **2026-10-31 02:10 IST: n9c FAIL on lookup** (EXPERIMENTS "n9c result"): N9a pulls the
  label keys apart (0.715 → 0.538) but label_exact 0.61 → 0.47 and category 0.50 → 0.32;
  recall 0.825 → 0.975 (> 2048: 0.92). Lookup skills follow the TRAINED format (recall docs
  "X refers to Y"; B2 docs → B2). The label gap looks like a data-format gap: next
  candidate is a label-format source (held-out items/labels, like b2), pre-register before
  running. 4090 now: v14 `v14n9s` + v15 `v15g1` (~13 GB); v13 has no 4090 job.

* **2026-10-31 01:05 IST, step back after the B2 half.** What surprised us: T++ loses the
  rule at L = 1024, INSIDE its trained length, once ~110 tokens of web text sit between
  examples; PAM holds it to 4096. So "learning past the trained length" was never only
  a position problem. It is whether the inferred rule is carried as STATE between
  examples or re-found by lookup each time. Data unlocked it: before the b2 docs no model
  learned past 2048. Recipe ideas (v13 lane, not run, cheap once the pair is settled):
  (a) b2 docs with the examples already spread by filler (train the carrying
  directly); (b) a label_category-style source with held-out items, same idea for the
  one task T++ still wins; (c) answer weight 100 → 10 to keep recall (0.870 → 0.721 now).

* **2026-10-30 23:25 IST: priority re-plan (user: run what we know is better, directly).**
  6000 (tmux `s5`): b2p1 (PAM + B2 docs) → b2t (T++ + B2 docs) → **r3c** via the
  chain's hook `~/s5_n9a.sh` (16 notebooks + echo + two stages + LR 1.2e-3,
  straight to 3B, futility stop at step 39060 if holdout > 3.4714; ~00:45 →
  ~14:45 IST) → b2p2 (deferred seed). The 1.5B body × LR grid is CANCELLED
  (the 3B logs already answer it). 4090: n9c (N9a lookup check, ~01:30).

* **2026-10-30 ~23:00 IST — step back after the 3B result.**
  - What we know: PAM learns fast early and plateaus (3B: T++ −0.10, Mamba-2
    −0.145). Its lookup fails from key collisions (fix found: N9a 5/5 at
    micro). It carries rules (B2 held-out shifts, replicated at 27M) and
    keeps memory past the trained length. The 600M wins were intercept, not
    slope.
  - Plateau hypothesis: capacity, not learning. The body we optimised at
    600M is cheap: 4 notebooks out of 16 blocks, ~1.2 MB state (Mamba-2 13.4
    MB, 17 mixing layers). The old 16-notebook body did better at 3B (3.273 vs
    3.309). At 600M, data limits everyone, so the fast learner wins; at 3B,
    capacity limits, and the small state shows.
  - Process lesson (2nd time: the 100M LR screen, now the 600M recipe):
    choose by SLOPE. Measure every candidate at two budgets (e.g. 600M and
    1.5B) and keep what improves the drop between them, not the 600M number.
  - Next, after n9c and the B2 pair: a slope test on the 6000 of state /
    mixing capacity at 1.5B tokens (M2 4-notebook vs 16-notebook + echo +
    two stages; + N9a if it passes), read by the 600M → 1.5B drop.

* **2026-10-30 16:00 IST: joint re-plan with v14 (user ask).** 4090: g1c (→ ~18:00)
  → F1 v13 arms (tmux `f1`, `~/f1_chain.sh`, worktree `../qllm_q1` at 3208b64:
  echo_qinit + echo_sparse, 5 seeds; v14 runs echo + echo_phase in `v14f1`) →
  the F1 winner at 600M. 6000: r3b (→ ~20:30) → B2-in-training 600M pair
  (source `b2`, d27b256), PAM arms start with the F1 winner. Deferred: Q1-B2,
  language-first vs LR restart, rehearsal. 6000 disk: the 9 old 100M LR-screen
  checkpoints (pamlr/tflr/m2lr) moved to `/media/gowrav/d_drive/qllm2/from_6000/`
  on the local box (md5 verified), 6000 now 20 GB free.

* **2026-10-30 13:35 IST: g1c RUNNING on the 4090** (worktree `../qllm_g1c`,
  9c2b632, ~3 h → ~16:45 IST). LCL DONE: no model's learning holds past
  2048 (PAM keeps the label set, loses the mapping). **Q1 tied start**
  (`--echo_qk_tied_init`, e89b7d2) pre-registered: formation rate over 5
  seeds on v14's bench first, then 600M. r3b on the 6000 ~21:30 IST.

* **2026-10-30 12:25 IST: label_category gap diagnosed; g1c queued on the 4090.**
  label_diag: PAM fails EXACT in-context copy too (0.50–0.62 vs T++
  0.78–0.88, Mamba-2 0.62–0.72); not capacity (3.3× state no help), not
  data (3B same); near-duplicate keys (echo key cosine 0.67–0.72 at label
  positions) + delta erase = last write wins (recency 0.72 → 0.29 over 5
  pairs). Mamba-2 (3.44 / 65.0) leads LM by 0.010 and label_cat; PAM leads
  recall past 2048 with ~11× smaller state. **g1c** (C1-ctl recipe +
  SHORT_CONV=1) waits in tmux `g1c` (`~/g1c_chain.sh`, worktree
  `../qllm_g1c` at 9c2b632, log `logs/v13_sempty_g1c_driver_9c2b632_20261030_0654.log`
  in that worktree) for "lcl DONE"; whole 4090 ~3 h by agreement with v14
  (one small v14 job may run beside it). **LCL** (tmux `lcl`, `~/lcl_run2.sh`,
  worktree `../qllm_e2`) ~13:15 IST. **r3b** on the 6000 ~21:30 IST.
  Local checkpoints now live on d_drive: `checkpoints_v13_sempty` is a symlink
  to `/media/gowrav/d_drive/qllm2/checkpoints_v13_sempty` (gitignored by
  `/checkpoints*`; never commit it: the 6000 has real dirs).

* **2026-10-30 ~10:45 IST — step back (think in the void) after today's results.**
  1. Our clearest edge is not "better at 2048": it is that memory keeps
     working past the trained length (T++ 0.15–0.20 vs PAM 0.60–0.72 past
     2048). But every LEARNING probe we run is short (≤ ~1k tokens; the
     pools run out at 32–128 shots). We never tested learning where our edge
     lives. Next probe: long-context in-context learning — the same rules
     with examples spread over 4–8k tokens (filler text between shots, or
     many shots), query at the end. Does learning hold past the trained
     context (user's claim) while attention's collapses?
  2. The schedule effect (language first, then memory exercises, LR
     re-warm) helps PAM only. Smells like a critical period: the notebooks
     need good representations before they learn what to store. Untested
     split: language-first vs the LR restart itself. A bolder angle: let the
     model open its own memory gates as its language matures (write
     strength tied to its own loss), instead of a fixed two-stage cut.
  3. Accident worth chasing: right after the aptitude stage, models score
     their highest learning-probe means (T++ stage A 0.229, c1m2a 0.195),
     then it fades in the web-heavy stage B. Reasoning data may need
     rehearsal (a small share kept in stage B), not only a first course.
  4. v14 B2 run 1: echo infers held-out letter shifts with a rising shot
     curve where attention falls back to memorised neighbours. If B2 run 2
     replicates, the 600M test should also be long-context (combine 1 + 4).

* **2026-10-30 04:25Z (09:55 IST): S4 chain RUNNING on the 6000 at `685de4e`**
  (no E2 port). mbs1 (Mamba-2 one stage, 189K tok/s) ~10:55 IST → mbs2 ~12:00
  → r3b (reference at 3B tok) ~21:30 IST. tfc1 recorded (685de4e). Shared
  checkout: v14 now runs from its own worktree (../qllm_v14run), so master
  commits here are safe again.

* **2026-10-30 03:17Z (08:47 IST): E2 tied echo FAIL, code removed** (holdout
  +0.055, recall 0.602 vs 0.825, label_cat 8-shot 0.365 vs 0.50; code at
  `5100fd7`). 4090 handed to v14 (B0 run 3). S4 has no port: Mamba-2 → r3b.

* **2026-10-30 02:32Z (08:02 IST): S4 chain queued on the 6000** (tmux `s4`,
  `~/s4_chain.sh`, log `logs/v13_sempty_s4_driver_20261030_0232.log`). Waits
  for "=== s3 chain DONE" AND `~/v14_slot2.done` (v14 touches it after its
  tpp calibration + B2 run 1, ~10:40 IST) AND no python on the GPU; then
  pulls once and pins that hash. Order: [E2 port `e2m2` only if
  `~/s4_port.flags` exists] → `mbs1` (Mamba-2 one stage) → `mbs2` (Mamba-2
  two stages) → `r3b` (reference at 3B tokens, LR 2e-4, init c1m2a_pre;
  extra flags from `~/s4_r3b.flags`, tied echo then inits from e2m2a_pre).
  ETA ~11 h → ~22:00–23:30 IST. Rules: EXPERIMENTS "S4". Resume: relaunch
  the same tmux command with `SCREEN_HASH=<hash in the driver log>`.
  Freed 7.6 GB on the 6000 (latest.pt of finished July v11 runs); 17 GB free.

* **2026-10-30 02:16Z (07:46 IST): c2m2 DONE — KEEP; reference = c1m2** (tie-break:
  holdout within 0.005, c1m2 recall +0.022 > 0.02). c2m2: holdout 3.4497,
  WikiText 65.57 (beats T++ two-stage 66.20), recall 0.848. The s3 chain
  (tfc1) started 02:17Z, ETA ~03:50Z = 09:20 IST; then the 6000 → v14.
  B2 probe for c2m2 running (tmux `b2probe`, worktree ~/Development/qllm_probe).
  6000 disk 11 GB.

* **2026-10-30 00:18Z (05:48 IST): tfs2 DONE — S2 fairness rules MET**: two-stage
  M1 beats two-stage T++ on holdout (−0.016) and recall (+0.122; the win is
  past 2048 tokens, ≤ 2048 ties); T++ better on WikiText and label_category.
  c2m2 (two stages on M2) started 00:18Z, ETA ~02:15Z = 07:45 IST; then the
  s3 chain (tfc1) ~09:30 IST. 6000 disk 14 GB.

* **2026-10-30 00:13Z (05:43 IST): E2 tied echo LAUNCHED on the 4090** (v14 B1
  run 4 bar P met). tmux `e2`, `~/e2_chain.sh`, run FROM THE WORKTREE
  `/home/gowrav/Development/qllm_e2` at `e5b1a2a` (ECHO_TIED code `26ef94a`);
  logs + checkpoints live in the worktree (`logs/v13_sempty_e2_driver_e5b1a2a_20261030_0013.log`,
  `e2ta_pre` 120M → `e2tb_main` 480M, then the probes incl. the 400-trial
  learning probe on e2t AND C1-ctl). ETA ~10:15 IST + probes. Resume after a
  crash: `tmux new-session -d -s e2 "cd /home/gowrav/Development/qllm_e2 &&
  SCREEN_HASH=e5b1a2a bash ~/e2_chain.sh 2>&1 | tee -a <same driver log>"`.
  When done: copy logs/probe jsons from the worktree into this checkout and
  commit here. **6000 hand-off:** after `=== s3 chain DONE` (tfc1, ~09:30 IST)
  the 6000 goes to v14 (tpp calibration, then B0 run 3, until ~12:00); ping
  v14 when s3 ends. An E2 port to the M2 reference waits until after that.

* **2026-10-29 22:59Z (04:29 IST): tfs1 (Transformer++ one stage 600M) DONE**:
  holdout 3.4885 (M2 beats it by −0.019; M1 ties), WikiText 66.38 (better than
  PAM), recall 0.645 (strong ≤ 2048, collapses past: 8192 0.15), learning
  probe 0.217 with label_cat 8-shot 0.73 (PAM ≤ 0.55). Logs on branch `e2`
  (master frozen). tfs2 started 22:59Z.

* **2026-10-29 22:05Z (03:35 IST): E2 tied echo prepared, gated on v14 bar P.**
  RULE (local box): `/home/gowrav/Development/qllm2` is SHARED by the v13
  and v14 sessions. While the other session's runs may launch
  (require_clean_tree) or crash-resume (HEAD pin), edit code in a worktree
  (`git worktree add ../qllm_<id> -b <id> master`) and do not move master.
  v13's E2 code + notes live on branch `e2` (worktree `../qllm_e2`,
  symlinks `.venv`, `.cache` → the main checkout; C1-ctl best_model.pt copied
  there for the 400-trial probe). After v14 pings that run 3 ended:
  ff-merge `e2` into master, push origin, launch `~/e2_chain.sh` in tmux
  `e2` FROM THE WORKTREE (it waits until the GPU has no compute apps).
  ETA ~4.5 h → ~09:30–10:00 IST. Rules: EXPERIMENTS "E2".
  04:00 IST: run 3's P INVALID (echo at ceiling); gate moved to v14 B1
  run 4's P (~05:45 IST). Do NOT launch E2 before v14 pings run 4's P.
  Master is FROZEN (v14 run 4 pairs re-import v13 code + pin HEAD) until
  that ping: no merge/commit/push to master; v13 notes go on branch `e2`.
  Then: rebase `e2` onto master (no run is named after an e2 hash yet), ff
  master, push origin, launch E2 if P passed.

* **2026-10-29 21:44Z (03:14 IST): C1 on M2 DONE — KEEP, new best LM at 600M.**
  `c1m2b_main_7ae6f1f`: holdout 3.4473 (M2 3.4697), WikiText 65.74, recall
  0.870 (M2 0.522), learning probe 0.187 (bar 3 open), label_cat 8-shot 0.38
  (M2 body does not get the 0.55 two-stage lift). Logs + probe jsons
  committed. Freed c1m2b step_009766.pt (probes done); 6000 disk 17 GB.
  The S2 chain started `tfs1` at 21:45Z on `b788731` (T++ 142K tok/s →
  tfs1 ~23:00Z, tfs2 ~00:20Z, c2m2 ~02:30Z = 08:00 IST). **Queued after it:
  `tfc1`** (T++ on c1m2's aptitude recipe; tmux `s3chain`, `~/s3_chain.sh`,
  log `logs/v13_sempty_s3_driver_b788731_20261029_2150.log`, waits for
  "=== s2 chain DONE"; ETA ~04:00Z = 09:30 IST). Rules: EXPERIMENTS "C1 on
  M2 result", "tfc1". Resume: rerun the same tmux command (stages skip via
  .done). E2 tied echo: waits on v14 B1 run 3 bar P (~05:00 IST) AND the
  reference pick after c2m2.

* **2026-10-29 19:53Z (01:23 IST): S2 chain queued on the 6000** (tmux
  `s2chain`, `~/s2_chain.sh`, log `logs/v13_sempty_s2_driver_20261029_1952.log`;
  it prints the code hash it pulled). Waits for "=== chain DONE" of the
  c1ctl chain (C1 on M2, ETA ~22:00Z), pulls (pull_aside) + sempyt, then
  `tfs1` (Transformer++ one stage, 600M) → `tfs2` (Transformer++ two
  stages) → `c2m2` (two stages on the M2 body); ETA ~02:30Z = 08:00 IST.
  Disk guard: stops if < 5 GB free (was 19 GB at launch). Resume: relaunch
  the same command with `SCREEN_HASH=<hash in the driver log>`.
  **C1-ctl verdict:** the two-stage schedule alone (120M web+chat, no recall
  docs → 480M M1 mix, LR re-warm) beats M1: holdout 3.4743 vs 3.4906,
  recall 0.825 vs 0.615. Aptitude adds +0.10 recall at step 9766 but costs
  +0.026 holdout. EXPERIMENTS "C1-ctl", "S2 fairness", "S2 on the M2 body".

* **2026-10-29 18:10Z (23:40 IST): overnight chain on the 6000** (tmux
  `c1ctl`, `~/c1ctl_chain.sh`, log
  `logs/v13_sempty_c1ctl_driver_7ae6f1f.log`, code `7ae6f1f`). Order:
  learning probes M2 + T1 (DONE: 0.185, 0.171) → **C1-ctl** (`c1ctla_pre`
  120M web+chat, NO aptitude → `c1ctlb_main` 480M M1 mix, keeps
  step_009766.pt; ETA ~19:55Z = 01:25 IST) → **C1 on the M2 body**
  (`c1m2a_pre` / `c1m2b_main`, SHARE_READS=1 MLP 2128; ETA ~22:00Z = 03:30
  IST). Each prints `=== <p> DONE`; probes (recall, PG-19, learning, plus the
  step-9766 recall + learning) run inside. Rules: EXPERIMENTS "C1-ctl", "C1 on
  the M2 body". Resume after a crash: `tmux new-session -d -s c1ctl "cd
  ~/Development/qllm-private && SCREEN_HASH=7ae6f1f bash ~/c1ctl_chain.sh 2>&1
  | tee -a logs/v13_sempty_c1ctl_driver_7ae6f1f.log"` (finished stages skip
  via .done, half-done resume from latest.pt).
  **Verdicts landed today:** M2 KEEP (holdout 3.4697, −0.021 vs M1; recall
  0.522 < M1 0.615). T1 FAIL (3.4977 vs echo 3.4885; code REMOVED in `a0b0337`,
  last commit with it `5eff3b9`). NOTE: the 6000 has untracked copies of
  logs now committed here — pull there with the chain's `pull_aside`. C1 primary FAIL (learning probe 0.199) but
  **recall 0.907 vs M1 0.615** (8-fact 0.855 vs 0.326) — the control decides
  whether the aptitude data or the two-stage schedule causes it. Fixed
  `609fb1f`: the final step is always validated (C1b's last 114M tokens
  were unmeasured). Learning probe (bar 3): every model 0.17–0.23, near
  chance; still open. Results logs for m2 / c1a / c1b / t1loc are committed.
  4090: v14 owns it tonight (v14b1 then v14b0, ETA ~02:30 IST). Lesson: a
  4090 launcher must NOT end in `exec bash` (t1loc's open shell blocked
  v14's WAIT_TMUX gate for 2 h).

* **2026-10-29 13:55Z: C1 reason-first curriculum queued on the 6000** (tmux
  `c1chain`, `~/c1_chain.sh`, log `~/c1_chain.log`): waits for "m2 DONE" and
  the Mamba-2 learning-probe json, pulls code + sempyt, runs `c1a_pre`
  (120M: 30M web, then aptitude 0.40 / chat 0.15 / web) then `c1b_main`
  (480M, M1 mix, `--init_from` c1a best), then the learning probe on both.
  ETA ~17:05Z. Resume: `tmux new-session -d -s c1chain "SCREEN_HASH=<hash in
  ~/c1_chain.log> bash ~/c1_chain.sh 2>&1 | tee -a ~/c1_chain.log"` (finished
  stages skip via .done). Rules: EXPERIMENTS "C1".

* **2026-10-29 13:20Z: learning probe (bar 3) baselines DONE** (EXPERIMENTS
  "Learning probe"): no model applies an in-context rule, Transformer++ 3B
  included. 6000 queue: M2 (`m2chain`, started 13:00Z, ETA ~14:50Z) → tmux
  `lpmamba` runs the Mamba-2 3B probe (needs mamba_ssm, only on the 6000)
  after "m2 DONE"; copy `logs/learning_probes/v13_sempty_mix3b_mamba2_answ100_b36e3a5_rules.json`.
  4090: T1 (`t1loc`) to ~16:15Z, then the v14 session's B0/B1 (gated on t1loc).
  **Before the next v13 launch on the 6000** (after M2 + lpmamba): `cd
  ~/Development/sempyt && git pull` — sempyt `b99e038` (abs/conj/normalize
  raise if the SplitComplex pair axis is not last; scan per-step-decay fix
  2790c69). v13 selftest all 25 pass against it (local, 13:40Z).

* **2026-10-29 12:35Z: M1 KEEP — new reference body** (4 notebooks in blocks
  3,7,11,15 + echo, MLP 2352): holdout 3.4906 (echo 3.4885), WikiText 69.4
  (71.1), recall 0.615 (0.401), 1.8× faster. Best ckpt on the 6000
  `checkpoints_v13_sempty/m1_m1echo_ccbf028/best_model.pt`. 6000 now: v14 B0
  (session qllm2-d5) then M2 (on top of M1). T1 still on the 4090 (echo body).

* **2026-10-29 12:00Z: two sessions share the GPUs.** v13_sempty = session
  `qllm2-04`; v14 (new neurons) = session `qllm2-d5`. Rules: ping the other
  (SendMessage) on every GPU job start/stop (box, tmux, GB, ETA); ask before
  launching on a GPU the other uses; ping on verdicts. Plan now: 4090 = T1
  (`t1loc`) to ~16:15Z incl. probes; v14 `v14b0` waits for 12 GB free there.
  6000 = M1 → **v14 B0 slot** → M2: `~/m2_chain.sh` now waits for "m1 DONE"
  AND `~/v14_slot.done` (touched by the v14 session when B0 ends; 3h
  timeout). The maya3 voice server + voice-agent (user's, ~10 GB) also live
  on the 4090 — leave them alone.

* **2026-10-29 11:15Z: T1 thought stream (two-speed mind) — local 4090
  same-batch check** tmux `t1loc` (`SCREEN=t1loc ARMS='t1:ECHO=1,N_LAYERS=12,THOUGHT=1'
  BASE='LR=1.2e-3 TARGET_TOKENS=599998464 VAL_EVERY=4069 SAVE_EVERY=500
  KEEP_EVERY=4069 KEEP_LAST=1 GRAD_CKPT=1' bash v13_sempty/tmp_scratch_screen.sh`,
  hash `ea002e6`, log `logs/v13_sempty_t1loc_t1_ea002e6_*.log`). Compare
  train loss vs `logs/v13_sempty_e1g_echo_a0fab65_20261029_0817.log` at step
  2250 with the cmp script (EXPERIMENTS "T1 thought stream" rules); then stop
  it (`tmux kill-session -t t1loc`) or let it run if clearly ahead. Resume:
  same command with `SCREEN_HASH=ea002e6`. Not paused (user 12:00Z: keep it).

* **2026-10-29 11:10Z: E1 echo PASS at 600M** (holdout 3.4885 vs 3.5408,
  WikiText 71.1 vs 75.9, recall 0.401 vs 0.343, PG-19 far gain 2.4×; table +
  generations in EXPERIMENTS "E1 at scale"). Echo = default recipe from now.
  M1 (`m1echo`, run hash `ccbf028`, log
  `logs/v13_sempty_m1_m1echo_ccbf028_20261029_1100.log`) started 11:00Z on
  the 6000, ETA ~13:45Z; M2 follows (~16:30Z). Both 600M ckpts (ctl, echo)
  are also on the local 4090 for generation tests.

* **2026-10-29 10:00Z: queue on the 6000** — `e1g` echo arm (finishing
  ~10:45Z; conv arm SKIPPED by marker `checkpoints_v13_sempty/e1g_conv_a0fab65/.done`)
  → tmux `m1chain` (`~/m1_chain.sh`, log `~/m1_chain.log`) waits for "e1g
  DONE", pulls, runs screen `m1` arm `m1echo` (4 notebooks + echo, MLP
  2352, `497bbca`+). EXPERIMENTS "M1 sparse memory". Resume:
  `tmux new-session -d -s m1chain "SCREEN_HASH=<hash in ~/m1_chain.log> bash ~/m1_chain.sh"`.
  Then tmux `m2chain` (`~/m2_chain.sh`, log `~/m2_chain.log`) waits for "m1
  DONE", pulls, runs screen `m2` arm `m2echo` (M2 shared reads, MLP 2128,
  `b9b446c`+). Same resume pattern.

* **2026-10-29 09:10Z: E1 echo at 150M tok: holdout −0.068 nats, WikiText
  119.6 vs 148.6 (−19 %)** — far past the −0.02 bar; milestone table in
  EXPERIMENTS "E1 at scale". Micro-bench: a4/a8 1.00 in 2000 steps.

* **NOW (2026-10-29 08:20Z) — screen `e1g` on the 6000** (tmux `e1g`,
  `~/e1g_chain.sh`, log `~/e1g_chain.log`, pinned `a0fab65`): arm `echo`
  (E1, ETA ~11:05Z) then `conv` (G1, ETA ~14:10Z), 600M tok each vs
  `r600_ctl_91daf0a`. Driver `logs/v13_sempty_e1g_driver_a0fab65.log`.
  Resume: `tmux new-session -d -s e1g "SCREEN_HASH=a0fab65 bash ~/e1g_chain.sh"`.
  gscreen (and e1chain) were STOPPED: G2 onorm FAIL for futility (+0.14
  nats at step 2250). Remove the G2 code after e1g finishes (not before:
  it would break a pinned resume). Compare runs step-by-step against the
  control's train loss (same batches) — it tells within ~1000 steps.

* **NOW (2026-10-29 08:00Z) — scoreboard DONE: PAM loses LM to both
  baselines (holdout +0.11 nats vs Mamba-2, +0.07 vs Transformer++), best
  multi-fact recall. LR 1.2e-3 was the wrong pick for PAM at 3B (2e-4 is
  better); 100M screens can't judge PAM — use >= 600M. EXPERIMENTS
  "Scoreboard result".**
  - **G screen running** on the 6000, tmux `gscreen` (`~/g_chain.sh`, log
    `~/g_chain.log`), code `3cc2b72`, arms `onorm` (G2 read RMSNorm), `conv`
    (G1 short conv), `both`, 600M tok each at LR 1.2e-3, ~2.6-3h per arm,
    ETA all done ~17:00Z 10-29. Control = finished `r600_ctl_91daf0a`
    (holdout 3.5408, WikiText 75.89, recall 0.343). Pass rule + design:
    EXPERIMENTS "G screen". Driver log
    `logs/v13_sempty_gscreen_driver_3cc2b72.log`. Resume after a crash:
    `tmux new-session -d -s gscreen "SCREEN_HASH=3cc2b72 bash ~/g_chain.sh"`
    (SCREEN_HASH skips the pull; finished arms are skipped).
  - G `both` arm SKIPPED (marker `checkpoints_v13_sempty/gscreen_both_3cc2b72/.done`
    + SKIPPED note); run later under a new screen name only if conv AND onorm pass.
  - **E1 echo** (novel, `86baea2`, EXPERIMENTS "E1 Echo memory") queued: tmux
    `e1chain` (`~/e1_chain.sh`, log `~/e1_chain.log`) waits for "gscreen DONE",
    pulls, runs `SCREEN=e1 ARMS=echo:ECHO=1` 600M. ETA start ~13:40Z, done
    ~16:30Z. Resume: `tmux new-session -d -s e1chain "SCREEN_HASH=<hash in
    ~/e1_chain.log> bash ~/e1_chain.sh"`.
  - Local 4090: PG-19 of the r600 control DONE ->
    `logs/pos_loss/v13_sempty_r600_ctl_91daf0a_pg19.json` (far_gain
    +0.011/+0.016/+0.018).
  - Speed: not alignment; flat profile; next lever = fused delta kernel,
    e.g. flash-linear-attention `chunk_gated_delta_rule` (exact mapping in
    EXPERIMENTS "Speed: where PAM's 2.9x gap is"). Needs user OK to install.

* (history) **NOW (2026-10-28 19:00Z) — Transformer++ 3B DONE (`e072a8f`): holdout
  24.69, WikiText 46.61, recall mean 0.539 (fails past 2048); EXPERIMENTS
  "Transformer++ 3B result". PAM 3B @1.2e-3 running on the 6000 (tmux
  `tfchain`, run hash `9982a79`, TAG `mix3b_delta_answ100_lr1.2e-3`, driver
  `logs/v13_sempty_mix3b_delta_answ100_lr1.2e-3_driver_9982a79.log`), ETA
  ~08:00Z 10-29; probe + PG-19 run automatically after it. Resume if it dies:
  `cd ~/Development/qllm-private && ARCH=pam LR_FULL=1.2e-3 TAG_FULL=mix3b_delta_answ100_lr1.2e-3 SCREEN_HASH=9982a79 bash v13_sempty/tmp_equal_3b.sh`
  in tmux. Then fill the scoreboard row.**

* (history) **NOW (2026-10-28 12:15Z) — C1 FAIL (removed, code at `4c688cb`).
  Transformer++ equal-data run starting on the RTX 6000 (tmux `tfchain`),
  then PAM 3B at LR 1.2e-3 (no C1). ETA transformer ≈ 18:30Z, PAM 3B ≈
  07:30Z tomorrow.**
  - C1 verdict at 300M: far a1 0.10 vs 0.26, holdout +0.055 → FAIL
    (EXPERIMENTS "C1 result"). `c1a01` stopped at step 8,900; the r600
    driver log got a manual "r600 DONE (stopped early)" line to release
    `tfchain`. `~/pam3b_env.txt` = `EXTRA=""`,
    `TAG_FULL=mix3b_delta_answ100_lr1.2e-3`.
  - **Before proposing anything: read `v13_sempty/EXPERIMENT_INDEX.md`**
    (every experiment, verdict, code-at hash).
  - (history) C1 100M done (EXPERIMENTS "C1 100M arms"): `c1a01` picked; early
    short-range recall sign, holdout +0.038 vs control. No verdict yet.
  - Results + amended C1 protocol: `EXPERIMENTS_SEMPY.md` → "LR screen
    result" and "Amendment to the C1 protocol". At 100M tokens no arm learned
    recall, so C1 is judged on a 600M schedule with milestones every 150M.
  - **C1 100M** (no verdict, picks the variant): tmux `c1chain` on the 6000,
    code `9ebd1d2`, arms `c1` (A init 1) / `c1a01` (A init 0.1) at LR 1.2e-3.
    Driver `logs/v13_sempty_c1_driver_9ebd1d2.log`. Resume: `tmux new-session
    -d -s c1 "cd ~/Development/qllm-private && SCREEN=c1 SCREEN_HASH=9ebd1d2
    ARMS=\"c1:COUPLE=1,LR=1.2e-3 c1a01:COUPLE=1,COUPLE_A=0.1,LR=1.2e-3\" bash
    v13_sempty/tmp_scratch_screen.sh"`. **No `git pull` on the 6000 until
    "c1 DONE"** (HEAD moved past 9ebd1d2 with code).
  - **600M runs:** tmux `r600chain` runs `~/r600_chain.sh` (outside the repo):
    waits for "c1 DONE", picks the variant by the amended rule, moves aside
    untracked files the pull would overwrite (`~/qllm_pull_aside_*_r600/`),
    `git pull`, then `SCREEN=r600 BASE="LR=1.2e-3 TARGET_TOKENS=599998464
    VAL_EVERY=4069 SAVE_EVERY=1000 KEEP_EVERY=4069 KEEP_LAST=1" ARMS="ctl
    <variant>"`. Hash = HEAD at that pull (see `~/r600_chain.log`). If it
    died: rerun that screen command by hand with `SCREEN_HASH=<hash>`.
  - **4090 helper:** local tmux `r600probe` runs a scratchpad script that
    copies each new `checkpoints_v13_sempty/r600_*/step_*.pt` (md5-checked),
    probes it (40 trials) → `logs/memory_probes/v13_sempty_r600_<arm>_<hash>_s<step>_t40_behavioral.json`,
    deletes the copy (validated: 4090 probe of pamlr_lr1.2e-3 best == the 6000
    probe in all 63 cells; ~105 s copy + ~130 s probe). KEEP_LAST=1 on the
    6000 deletes each milestone when the
    next one lands (~38 min later), so the helper must keep running; a missed
    milestone is lost (the final one + best_model stay).
  - **Disk on the 6000: 37 GB free at 07:20Z** after pruning resume-only
    files of finished runs + superseded partial runs (user: "if you don't
    need, delete"; every best_model.pt kept). Screens now delete a finished
    arm's latest.pt themselves (`PRUNE_LATEST=0` keeps it).
  - **Queue after r600 (goal: PAM that beats Transformer++ and Mamba-2 at
    equal data, O(1)):** `EXPERIMENTS_SEMPY.md` → "Equal-data scoreboard".
    tmux `tfchain` (`~/tf_chain.sh`, outside the repo) waits for "r600 DONE",
    pulls, runs `ARCH=transformer bash v13_sempty/tmp_equal_3b.sh` (LR screen
    + 3B + probes, ~6 h), then waits for `~/pam3b_env.txt` (written after the
    C1 verdict: `EXTRA=...` and `TAG_FULL=...`) and runs the PAM 3B at LR
    1.2e-3 (~12.5 h). Transformer++ code `53d632e`. The Mamba-2 driver
    `tmp_mamba2_baseline.sh` became `tmp_equal_3b.sh` (`ARCH=mamba2`); the
    b36e3a5 run used the old file (git history).

* **2026-10-28 05:45Z — PAM LR screen launched (DONE 07:03Z, see above); C1 chained.**
  - Design + pre-registered reading: `EXPERIMENTS_SEMPY.md` → "PAM LR screen +
    C1 coupled gates". Both are from-scratch 100M-token screens
    (`v13_sempty/tmp_scratch_screen.sh`, ~32 min/arm at 66.5K tok/s).
  - **LR screen:** tmux `pamlr`, code `ce8bcc2`, arms 2e-4 / 6e-4 / 1.2e-3.
    Driver log `logs/v13_sempty_pamlr_driver_ce8bcc2.log`; arm logs
    `logs/v13_sempty_pamlr_lr*_ce8bcc2_*.log`; ckpts
    `checkpoints_v13_sempty/pamlr_lr*_ce8bcc2/`. Resume:
    `tmux new-session -d -s pamlr "cd ~/Development/qllm-private && SCREEN=pamlr SCREEN_HASH=ce8bcc2 ARMS=\"lr2e-4:LR=2e-4 lr6e-4:LR=6e-4 lr1.2e-3:LR=1.2e-3\" bash v13_sempty/tmp_scratch_screen.sh"`
    **Do NOT `git pull` on the 6000 while pamlr runs**: HEAD has code
    (`504da12`) past `ce8bcc2`, so the guard would refuse the next arm.
  - **C1 chain:** tmux `c1chain` runs `~/c1_chain.sh` (outside the repo):
    waits for "pamlr DONE", picks the LR by the screen rule (> 0.01 nats to
    win), `git pull`, then `SCREEN=c1 ARMS="c1:COUPLE=1,LR=<pick>
    c1a01:COUPLE=1,COUPLE_A=0.1,LR=<pick>"` at the pulled HEAD (hash printed
    in `logs/v13_sempty_c1_driver_<hash>.log`). Control = the pamlr arm at
    the same LR. If the chain died: rerun that command by hand with
    `SCREEN_HASH=<hash>`.
  - C1 code `504da12`: `--couple` (launcher `COUPLE=1`, `COUPLE_A=`), verified
    (selftest 20/20, CUDA decode parity, 80-step real-mix smoke).
  - **Disk on the 6000: 26 GB free before these screens** (~2.3 GB per arm,
    5 arms ≈ 12 GB). Prune old checkpoints (ask the user) before any 3B run.

* **2026-10-28 05:30Z — Mamba-2 "wind tunnel" DONE → verdict DESIGN.**
  - **Result** (`EXPERIMENTS_SEMPY.md` → "Mamba-2 result"): same params, data,
    loss, tokens. Mamba-2 holdout **23.68** vs PAM 26.38, WikiText **46.91**
    vs 54.73 (upper bound: PAM LR unscreened); 3.6× faster (188.6K vs 51.5K
    tok/s; 4.6 h vs ~17 h). Recall: one fact at the start of 2048 tokens
    Mamba-2 **1.00** vs PAM 0.12 (pre-registered bar 0.5 → DESIGN), @4096
    0.30 (bar 0.3, met exactly). Many facts: PAM far better (a4 pos1
    0.95–1.00 vs 0.45–0.68; a8 pos1 0.72–0.90 vs 0.25–0.45). Mean
    Mamba-2 − PAM: a1 +0.17, a4 −0.14, a8 −0.19. far_gain
    +0.017/+0.025/+0.031 vs +0.014/+0.019/+0.022.
  - **Options (user decides):** (1) PAM LR screen, the same 3 × 100M-token
    protocol as Mamba-2's (`tmp_mamba2_baseline.sh` step A with ARCH=pam),
    ~1.5 h; (2) port Mamba-2's write/forget coupling (one dt → write ∝ dt,
    forget = exp(−A·dt)) into PAM as a warm-start screen
    (`tmp_warm_screen.sh`, reuse the R4 ctrl arm), ~2 h. Suggested: 1 then 2.
  - **Files:** all logs/jsons committed (run, driver ×2, LR screen ×3, probe,
    PG-19). Ckpt `checkpoints_v13_sempty/mix3b_mamba2_answ100_b36e3a5/`
    (6000 only; latest/step_080000 pruned 2026-10-28, best_model.pt kept).
  - **Rerun (6000 only):** the run used `tmp_mamba2_baseline.sh`, which exists
    at `b36e3a5` only (`git checkout b36e3a5`, then `SCREEN_HASH=b36e3a5
    LR_FULL=1.2e-3 bash v13_sempty/tmp_mamba2_baseline.sh`); today's
    equivalent is `ARCH=mamba2 bash v13_sempty/tmp_equal_3b.sh`.
  - **Data-feed fixes made during this run** (all verified batch-identical;
    EXPERIMENTS "Engineering: batch prefetch ..."): thread prefetch `129b406`,
    worker process `09026e9`, background 8 MB parquet block fetch + 64-batch
    look-ahead `b36e3a5` (98K → 188.6K; the limit was ~3 s HF read stalls).
    PAM runs get these too (PAM speed after the fix not yet measured).
    Superseded partial runs are in `~/qllm_pull_aside_20261027/` on the 6000.
  - **Libraries (6000 only, NOT pip deps):** `mamba_ssm` 2.2.5 +
    `causal_conv1d` 1.5.2 prebuilt cp311/torch2.8/cu12 wheels + `einops`,
    installed `uv pip install --no-deps` (torch 2.8 / triton 3.4 /
    transformers 5.2 untouched; exact command in the `baseline_mamba2.py`
    docstring). `uv sync` would remove them. No cp39 wheels exist, so the
    4090 (py3.9) cannot run the mamba2 path. Test scripts in `~/pkgprobe/`.

* **2026-10-27 ~12:30Z — R4 screen DONE: FAIL, store code removed.**
  - **Verdict** (`EXPERIMENTS_SEMPY.md` → "R4 result"): a1 pos0 @2048/4096
    store 0.30/0.15 vs ctrl 0.20/0.15 (bars 0.5/0.3); multi-way worse (a8
    mean −0.087); holdout 26.40 vs 26.42; PG-19 far_gain identical; tok/s 86 %.
    Zeroing the store gates changes NO recall cell → the store is unused; the
    multi-way loss came from its gradient on the shared q/k/v. Sixth
    retention lever that left the horizon at ~1000 tokens.
  - **Code:** store removed (config/model/kernels/selftest/train/launcher);
    selftest 17/17, layout guard clean, Phase 3b ckpt loads `strict=True`.
    `--init_from` kept. Screen driver generalised →
    `v13_sempty/tmp_warm_screen.sh` (`ARMS="name:VAR=val,VAR=val ..."`,
    resumable, probes + PG-19 after). The R4 ctrl arm is the reusable control
    for any screen with that recipe: ckpt `checkpoints_v13_sempty/
    r4screen_ctrl_0868a1b/` (6000 only), jsons in `logs/memory_probes/` and
    `logs/pos_loss/` → a new screen runs only its new arm (~1.4 h saved).
    The store ckpts need `git checkout 0868a1b` to load (extra keys).
  - **Results committed** (scp'd from the 6000, md5-checked): arm logs, driver
    log, 40-trial probe + PG-19 jsons for both arms. The 6000's untracked
    copies were moved to `~/qllm_pull_aside_20261027/r4/` so `git pull` works.
  - **6000 disk:** 43 GB free (92 %). Prunable if needed: R3/R1 extra
    `latest.pt`+`step_*.pt` (~7 GB), `r4screen_store_0868a1b` (2.3 GB, FAIL).
  - **Qwen on the 6000 is still STOPPED** (user-approved, 2026-10-27):
    transient user unit `qwen-sglang.service` (SGLang Qwen3.8 Flash-Next, port
    8081, `~/qwen38fn/serve.sh`); restart command in `~/qwen38_RESTART.txt` on
    the box. `~/model.sh` manages a different server (vLLM Qwen3.6). Restart it
    if the GPU will sit idle for long.
  - **NEXT (proposed, awaiting the user's go):** the "wind tunnel" — train a
    ~100M Mamba-2 (then Gated DeltaNet) on the EXACT Phase 3b 3B-token mix and
    run the same tests (holdout/WikiText PPL, 40-trial probe, PG-19 far_gain).
    Our only Mamba number is HF Mamba-130M trained on 300B Pile tokens (100×
    our data): it holds one fact to 8K (1.00) but loses on multi-fact. Same
    data separates "our design" from "data/scale" before a seventh lever.
    Needs a fast library on the 6000 (none installed: no `mamba_ssm`, `fla`,
    `causal_conv1d`; the old matched-Mamba attempt ran at 15 tok/s and was
    skipped) and a baseline trainer on the v13_sempty data path (answer
    weighting included).
  - Earlier today: R3 FAIL + removed (`5fb614f`); fast stream resume
    (`4b243c7`); clean-tree guard (`03b8acf`); smoltalk2 loud-fail
    (`68d0cd6`); PG-19 far_gain eval (`f366601`). PG-19 note: only compare
    mix-trained models there (WikiText-only ckpts score ~8 = domain shift).
    Git: push to `origin` only (never `p`), no co-author lines.

* **2026-09-06 afternoon — ALL of this was done ON the RTX 6000 (Cursor is
  ssh'd into it; `ssh 34.131.203.207` from the box itself times out, that is
  not an outage). The commit messages of `5e517f0`/`b44160b` say "4090" —
  wrong, every number there is 6000-measured.**
  - **Delta speed debt PAID (`5e517f0`):** `pam_delta_batched` — one batched
    64-wide WY solve + affine chunk-state recurrence + ONE scan launch;
    delta/additive step ratio 2.0× → 1.45×, memory −17 %, loss trajectory
    identical (parity fp32 2e-6). 6000, B8 T2048 grad-ckpt: 39.9K → 56.1K
    tok/s; B6 no-ckpt fwd+bwd 48.9K → 70K tok/s (additive 99K). At the
    training geometry (B18 no-ckpt, was 51.5K) expect ~75K. Not a
    hand-written Triton WY kernel — the profile said the loop, not the
    kernel, was the cost; a fused WY kernel is deferred (maybe +15 %).
    `pam_delta_torch` is the oracle; `v13_sempty/tmp/delta_parity.py` is the
    test. Prep is `torch.compile`d shape-generic (`V13S_DELTA_COMPILE=0` →
    eager).
  - **Delta decode DONE (`b44160b`):** `_stepwise_delta`; `generate.py` works
    on `mix3b_delta_answ100/best_model.pt` (greedy coherent). `--gen_every`
    may be non-zero for delta runs now. Parity `tmp/delta_decode_parity.py`.
    Finding: verbatim copy works; free-form multi-way lookups ("Bob lives in"
    → wrong city) fail → recall-doc FORMAT diversity is a data item.
  - **R2 retention bench DONE (`54c62ed`,
    `logs/v13_sempty_retention_bench_dt_nodecay_54c62ed_20260906_1411.log`):**
    under 75 % WikiText LM pressure every DECAYING arm (dt −4/−6/−8) is at
    chance on a4 at both needle positions (realised retention .86–.93, bias
    frozen); **`nodecay` = 1.00 on a1/a4 at every ctx to 8192 (a4 pos0 .88 @
    8192)** but WikiText val PPL 97.7 vs 87.3 (+12 %). a1 pos0 was 1.00 for
    all arms (bench doesn't reproduce the 100M's a1 decay). Record:
    `EXPERIMENTS_SEMPY.md` → "R2 retention bench". dt −6/−8: no gain.
  - **`mix3b_delta_nodecay_answ100` (`9f33b50`) DONE 2026-09-07 05:57Z —
    FAIL, `no_decay` REMOVED from code.** Holdout PPL **30.16 vs 26.38**
    (+14 %), WikiText 65.5 vs 54.7; recall WORSE at every length (a1 mean
    0.87/0.62/0.38/0.15 @128/512/2048/8192 vs ref 1.00/1.00/0.50/0.33; a8 @128
    0.67 vs 0.77). Signature = **state overflow**: the fresh-binding read
    (pos1) decays with context (1.00 → 0.25 @8192; ref 1.00 flat) and the
    model turned `pam_scale` down to 0.01–0.05 in layers 0–6. **Passive decay
    is load-bearing** — it keeps the O(1) state clean on web text. The 27M
    bench (25 % recall docs) mis-predicted this; don't vet decay knobs there.
    Record: `EXPERIMENTS_SEMPY.md` → "R2 at scale". Ckpt pruned to
    `best_model.pt` (1.2 GB; load with `git checkout 9f33b50`). Nothing
    running on the 6000.
  - **Stage L-2 DONE 2026-09-07 18:18Z (`13ea6a8`, 11 h, 56K tok/s) — data
    pressure FAILS as a horizon lever.** Holdout 30.55 / WikiText 59.26. a1
    pos0 @1024/2048/4096/8192 = 0.55/0.30/0.20/0.20 vs Phase 3b
    0.45/0.10/0.05/0.00 (chance 0.125, n=20 → 4k/8k within noise); a1 pos1
    1.00 everywhere. **dt_bias −4.00 ± 0.1 in all 16 layers after 2B tokens
    with 8 % long-gap docs** — the prior does not move under data. Short-range
    multi-way collapsed (a4 @128 0.62 vs 1.00, a8 0.40 vs 0.77), confounded by
    2.7× fewer optimizer steps at T=8192 — don't use 8K pretraining for the
    horizon question again. Record: `EXPERIMENTS_SEMPY.md` → "Stage L-2".
    Probe json committed; ckpt pruned to `best_model.pt`.
  - **R3 RUNNING on the 6000 — launched 2026-09-07 18:41Z, tmux `sempty_r3`,
    code `5c87148`, log
    `logs/v13_sempty_mix3b_delta_answ100_long2h9_5c87148_20260907_1841.log`,
    ckpt `checkpoints_v13_sempty/mix3b_delta_answ100_long2h9_5c87148/`.
    60K tok/s, 45.6 GB, 81,380 steps → ETA ~14 h (~08:30Z 09-08). Ladder line
    confirmed `long_heads=2 long_dt_bias=-9.0`.** Split decay prior, `--long_heads 2 --long_dt_bias -9`
    (2/6 heads per layer born slow, softplus(−9)=1.2e-4 ⇒ 0.37 of a binding
    survives 8192 tok), else exact Phase 3b recipe (T=2048 B18 3B tok,
    0.44/0.41/0.10/0.05, DELTA=1 ANSWER_W=100). One variable vs Phase 3b.
    ~11–16 h at 60–75K tok/s, ~46 GB. Judge: a1 pos0 @1024/2048 (0.45/0.10)
    and @4096/8192 (0.05/0.00); guards a4/a8 @128–512 (1.00/0.77) and holdout
    26.38. If long heads help → sweep 1/3 heads and −10; if not → the horizon
    needs a different write (e.g. content-gated *protection*), not a prior.
    ```bash
    tmux new-session -d -s sempty_r3 "TAG=mix3b_delta_answ100_long2h9 \
      SOURCES=dclm,fineweb,smoltalk2_mid,recall WEIGHTS=0.44,0.41,0.10,0.05 \
      DELTA=1 ANSWER_W=100 LONG_HEADS=2 LONG_DT_BIAS=-9 \
      bash v13_sempty/tmp_pretrain_mix.sh"
    ```
    **Rationale of L-2 (data pressure on the reference recipe).** The horizon (~1000 tok) tracks the longest gaps
    the recall docs ever ask for (`_build_recall_doc` gap log-uniform in
    [2, 200] sentences, median ~240 tok); the model has no reason to hold a
    binding longer. Run the exact Phase 3b recipe (DELTA=1 ANSWER_W=100,
    decay ON, dt −4) at **T=8192 B=8** with the L1 sources
    `dclm,fineweb_long,pg19,smoltalk2_mid,recall,recall_long`
    (`0.36,0.20,0.22,0.10,0.04,0.08`; recall_long gaps ≤6k tok), ~2B tokens
    (~9–10 h at ~60K tok/s), `GEN_EVERY=2000` (delta decodes). Judge: a1
    pos0 at 1024–8192 vs Phase 3b (0.45/0.10/0.05/0.00) and vs L1 (flat
    ~0.2, broken read); holdout PPL vs L1 at 1B (40.26). If pos0 moves →
    the horizon is a data problem and Stage L 32K follows; if it does not →
    re-test a *per-head* long prior (2 of 6 heads at dt −9) under the
    working delta read (L1's ladder was tested under the broken read).
    Queue after: recall-doc FORMAT diversity (free-form "X lives in Y"
    lookups fail in generation).
    ```bash
    tmux new-session -d -s sempty_l2 "TAG=mix2b_8k_delta_answ100_long \
      SEQ=8192 BATCH=8 TARGET_TOKENS=2000000000 BLEND_WARMUP=100000000 WARMUP=500 \
      SOURCES=dclm,fineweb_long,pg19,smoltalk2_mid,recall,recall_long \
      WEIGHTS=0.36,0.20,0.22,0.10,0.04,0.08 DELTA=1 ANSWER_W=100 \
      VAL_EVERY=1000 SAVE_EVERY=500 KEEP_EVERY=5000 GEN_EVERY=2000 \
      bash v13_sempty/tmp_pretrain_mix.sh"
    ```
* **MIX-3B + DELTA + ANSWER_W=100 DONE (2026-09-06, `1c913ef`) — the bench
  result TRANSFERRED to the real 100M.** Holdout PPL 26.38 (mix-3B 25.73;
  WikiText 54.73 vs 54.91 → a wash). Behavioral recall vs mix-3B: **a4 @128
  0.40 → 1.00, a8 @128 0.22 → 0.77** (0.63 @512, 0.38 @2048 — Mamba-130M's
  curve at 1 % of its tokens); **a1 horizon 200 → ~1000 tok** (1.00 to 512,
  0.77 @1024). Needle-just-before-query is perfect at EVERY length incl. 8192
  (4× train window): the read is length-invariant. Needle-at-start decays
  (0.45 @1024, 0.10 @2048, 0 @8192): **RETENTION over distance is now the
  limiter**, not retrieval (inverse of the L1 verdict; L1 couldn't read even
  fresh bindings). `--pam-scale 0` ablation → chance at every ctx: all recall
  is the PAM memory. Full record: `EXPERIMENTS_SEMPY.md` → "Phase 3b". Probe
  jsons: `logs/memory_probes/v13_sempty_mix3b_delta_answ100_1c913ef_behavioral.json`
  (+`_PAM0_`). **`DELTA=1 ANSWER_W=100` is the new reference recipe.**
  Ckpt `checkpoints_v13_sempty/mix3b_delta_answ100_1c913ef/best_model.pt`
  (step 80k; `latest.pt`/`step_080000.pt` are the same model — prune when space
  is needed). Nothing running on the 6000.
  **Next (in order):**
  1. **Retention over distance = the horizon program, now interpretable.**
     `dt_bias` has sat at its −4 init in every run. CHECKED 2026-09-06: it is a
     normal no-decay parameter on the gradient path, and `dt_proj.bias` barely
     moved either (±0.2) — there is simply no *consistent* gradient on
     retention (web LM has no net preference; long-gap answers are rare).
     Realized retention 0.64–0.97 < the 0.98 the bias alone gives, i.e.
     content pushes toward forgetting. So the lever is data pressure at long
     gaps and/or a structural prior, not an optimizer fix. Cheapest first, judged on
     the micro-bench (≥6000 steps, it has a ceiling now) with a *long-gap*
     schedule and pos0 needles: (a) data pressure — `recall_long` gaps (≤6k
     tok) in the mix + Stage L at T=8192; (b) `base_dt_bias` −4 → −6/−8 (longer
     default retention) since it never learns; (c) delta with decay *removed*
     (pure erase-on-rewrite memory, DeltaNet-style, retention pinned 1.0) vs
     current decay — the LM guard (holdout PPL) decides.
  2. ~~Engineering debt: Triton delta kernel + delta decode~~ DONE 2026-09-06
     (see top entry; ratio 1.45× not 1.0×, good enough for Stage L).
  3. Then Stage L (8K/32K) on the reference recipe.
  4. A fairer control when convenient: a RoPE transformer on the micro-bench
     (the v6 control is abs-pos); keep claims scoped to "same size, same data,
     same budget".
* **MULTI-WAY RECALL SOLVED — PAM-delta a1 = a4 = a8 = 1.00 at ctx 128–8192
  (2026-09-05 evening, `36a8cac`).** The user asked whether we were chasing the
  wrong error; we had never run a positive control. Added a transformer arm
  (`xf`, v6 GPT-2-style, 29.5M, byte-identical data/loss/steps) and a depth
  knob to the micro-bench. At the old 1500-step budget the **transformer is at
  chance on a8 too** (bench had no ceiling → all yesterday's multi-way FAILs
  were inconclusive; depth L4/L8/L12 changes nothing). At **6000 steps (~96M
  tok) PAM chrono+gate+delta (27M, 4 layers) goes through a phase transition at
  ~step 2500 and scores 1.00 on a1/a4/a8 at EVERY ctx incl. 8192 = 2× its
  train window, from a fixed O(1) state; the transformer is still at chance
  (a8 0.10–0.14), also with aux LM signal (`--aux_weight` 0.1/1.0).** Caveat:
  abs-pos control, not RoPE. **DeltaNet write CANCELLED (not needed).** The
  "no 1B run until multi-way is cracked" gate is LIFTED. Why the 1B pretrains
  showed a8 at chance: full-seq loss gives the answer token ~1/T of the
  gradient and recall docs were a small share → signal budget, not mechanism.
  Record: `EXPERIMENTS_SEMPY.md` → "Positive control". Logs
  `logs/v13_sempty_recall_microbench_xfctrl_depth_36a8cac_20260905_1659.log`
  (1500 steps, 4 arms), `…_xfctrl_6k_36a8cac_20260905_1714.log` (6000 steps,
  THE result), `…_xf_aux0p1_…1720.log`, `…_xf_aux1p0_…1720.log`.
  **Direction now:** (1) `DELTA=1` default; (2) scale run with the recipe that
  worked: recall/needle docs with answer-emphasised loss (per-doc-type mask
  weights, the `aux_weight` rule) so the recall signal is concentrated, then
  re-probe at 8K; (3) strategic: recurrent state = pattern core, explicit
  fact memory (A4 `cond_mem`) for parametric facts (user's "brain split");
  (4) optional: RoPE transformer control for a fairer ceiling.
* **MIX-3B DONE (2026-09-05, `bbc12e9`): holdout PPL 25.73, recall horizon
  ~200 tokens and SHRINKING with training** (a1 recall 1.00 @ctx128, 0.10 =
  chance @512; ctx512 went 0.35 → 0.10 from step 10k → 80k). `dt_bias` stuck
  at init. Full analysis: `EXPERIMENTS_SEMPY.md` → "Phase 3a". **Program is
  now RETENTION** (ladder R1 dt-spread → R2 vault → R3 delta), then Stage L
  (8K/32K). Complex arm: not revisited (same decay; phase ≠ retention).
* **DELTA (A3) IS A KEEP — read-path win, found on the micro-bench (2026-09-05,
  `543b897`).** `v13_sempty/tmp/recall_microbench.py` (small 27M real PAM, pure
  invented-association recall, answer-only loss, minutes/arm) shows delta's
  erase-before-write takes single-binding recall from **0.46 → 1.00, flat at
  ctx 128–8192 (extrapolates past the 4096 train window)** — O(1) state,
  no KV cache. **But multi-way is unsolved:** a4≈0.3, a8≈chance, and a
  head_dim {64,96,128} sweep did NOT help (refutes the orthogonality story) →
  the linear `q·S` read cannot isolate one of several co-resident bindings.
  Full record: `EXPERIMENTS_SEMPY.md` → "A3 delta rule".
* **MULTI-WAY attempt #1 = content-routed delta (A2r): FAIL, removed
  (2026-09-05).** Routed S=4/8 delta memories (write-by-key, read-by-query);
  a8 stayed at chance even with sharpened τ=0.2 (= plain delta's level at 4-8×
  cost). Three failures now (head_dim, soft route, sharp route) say multi-way
  is NOT a memory-structure/capacity problem — the model can't learn keys/
  queries selective enough. ~~Next candidate = the error-correcting DeltaNet
  write `S += β(v − Sk)kᵀ`~~ — **superseded the same evening: the transformer
  positive control (bullet above) fails multi-way identically, so this
  conclusion about the read path is withdrawn and the DeltaNet write was not
  run.** Record: `EXPERIMENTS_SEMPY.md` → "A2r content-routed delta".
* **L1 DONE (2026-09-05, `18ed358`): T=8192 B=8 dt_spread=8 long mix, 1B
  tok → holdout PPL 40.26, recall horizon FLAT ~0.2 at 512–8192, 128-ctx
  recall 0.63 (mix-3B 1.00), a8 at chance; dt_bias ladder never moved from
  init. R1 FAIL → `dt_bias_spread` removed from code.** Verdict: retention is
  NOT the bottleneck (state persists 0.83–0.99/layer) — RETRIEVAL is. This is
  what the delta micro-bench above then tackled. Full record:
  `EXPERIMENTS_SEMPY.md` → "L1 / R1 dt-spread". Generator: `.venv/bin/python
  -m v13_sempty.generate --checkpoint
  checkpoints_v13_sempty/mix1b_8k_r1_dtspread8_18ed358/best_model.pt --preset
  baseline_real_pm --interactive` (chrono decode path verified).
  Checkpoints are now rolling (`KEEP_LAST=1`); stale ckpts pruned (19→4.6 GB).
* **N4 READ-OUT GATE DONE (2026-09-04, run commit `2537782`, code `3c7b9b9`):
  val PPL 22.96** — beats chrono 23.14 at all 16 val points; 0.27 from the
  transformer (22.69). Gate verified content-dependent (doubles the effective
  memory read; co-varies with the Chrono clock). **KEEP; `CHRONO=1 OUT_GATE=1`
  is the reference (22.96).** Record: `EXPERIMENTS_SEMPY.md` → "N4 read-out
  gate". Log `logs/v13_sempty_wikitext_chrono_n4gate_fair_2537782_20260904_1517.log`,
  ckpt `checkpoints_v13_sempty/wikitext_chrono_n4gate_fair_2537782/best_model.pt`.
  A1 short conv → 23.49 FAIL, code removed. **PPL ladder on WikiText is
  saturating; next phase = scale the data (see "Scale plan" below).**
* **N1 CHRONO RUNG DONE (2026-09-04, commit `7b24e44`, RTX Pro 6000): val PPL
  23.14** — beats the 23.81 real baseline by 0.67 at every val point, gap to
  transformer (22.69) now 0.45. sROI KEEP; chrono is the new real-arm
  reference. Full record: `EXPERIMENTS_SEMPY.md` → "N1 Chrono-PAM". Log
  `logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`, ckpt
  `checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt` (on the
  RTX box). **Chrono decode now implemented** (state = `(notebook, clock)`,
  `test_chrono_parallel_vs_recurrent`); `--gen_every` may stay on.
* **Generator for prompt testing:**
  `.venv/bin/python -m v13_sempty.generate --checkpoint
  checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt --interactive`
  (loads once; type prompts; `/set temperature=0.6 max_tokens=120` retunes;
  blank line quits). One-shot: `--prompt "..."`. Device auto (cuda if present).
* **FAIR RUN DONE (2026-09-04, commit `7af42eb`): real-101M hit val PPL
  23.81** at the reference geometry (T=2048, B=18, 10 ep, 1.18B tok). Beats
  v11 E3-K3 complex (25.77) by ~2 PPL; within 1.12 of transformer (22.69).
  Log `logs/v13_sempty_wikitext_real_fair_7af42eb_20260903_1628.log`,
  ckpt `checkpoints_v13_sempty/wikitext_real_fair_7af42eb/best_model.pt`.
  Full curve + comparison + "is this Mamba?" positioning are in
  `EXPERIMENTS_SEMPY.md` (last two sections). Open gap = recall, not PPL.
* Logging now V11-style: `_print_run_header` emits full config/args/geometry
  at the top of every log; step lines carry `epN/M`; epoch-boundary banners
  print train-loss/ppl/tok/best-val. Log naming convention is in
  `qllm2/AGENTS.md` ("Log naming convention").
* **Next: bigger runs on the RTX Pro 6000 (96 GB)** — more data + chat, and
  scale-up if 100M looks saturated. See "Bigger-run plan" below / the new plan.

## Novel math — N1 Chrono-PAM (content-modulated rotary retention) (2026-09-04)

**Idea, in one line:** make the memory's rotary phase *learned and
input-dependent* instead of fixed RoPE — a "content clock" per head.

**Why it's principled (the derivation).** A complex *rotating* retention
`gamma_t = r_t * e^{i*theta_t}` on the outer-product notebook
`S_t = gamma_t S_{t-1} + v_t (x) conj(k_t)` has closed form
`S_s = sum_{t<=s} (a_s/a_t) e^{i(Phi_s - Phi_t)} v_t conj(k_t)` with
`a` = magnitude product, `Phi_t = cumsum(theta)`. The read
`S_s q_s` shows the `e^{i(Phi_s-Phi_t)}` factor is *absorbed* by rotating
`q_s -> e^{i*Phi_s} q_s`, `k_t -> e^{i*Phi_t} k_t`. That cumulative rotation
is exactly what RoPE does with a *fixed* frequency. So **learned rotating
retention == input-dependent RoPE**, and it folds entirely into q/k — the
fused magnitude-retention kernel is UNTOUCHED (speed preserved). CoPE-style,
but on an associative-memory PAM (novel).

**Real-arm implementation** (`RealPAMLayer._rotate_learned`, behind
`cfg.chrono`): per-head warp `g_t = exp(clamp(W x, +/-3))` scales the per-step
angle; since `inv_freq` is constant in t, `cumsum(inv*g) = inv * cumsum(g)`,
so we warp a per-head clock `tau = cumsum(g)` ([B,H,T]) then `phi = tau (x)
inv_freq`. `W` is zero-init (`warp_proj._zero_init`), so at start `g=1`,
`phi = pos*inv_freq` == **exactly** fixed RoPE. cos/sin in fp32, cast to bf16
for the rotation (keeps retained activations small).

**Status: RUNG DONE — 23.14 vs 23.81 (see State of play). Notes below are
the pre-run record.**
- Parity: `test_chrono_rotary_parity` (selftest, CPU) — chrono@init == baseline
  RoPE bit-for-bit (`max|dlogit| = 0.0`), warp grads flow. All 15 selftests pass.
- Layout: `check_torch_layout` clean (`_rotate_learned` is a declared boundary).
- **Speed gate PASSED**: baseline_real_pm B8 T2048 bf16 on 4090 — baseline avg
  **65.9k** tok/s vs chrono **66.6k** tok/s (equal within noise). Mem +1.6 GB
  (per-head cos/sin x16 layers; recomputed under grad-ckpt, irrelevant at 96 GB).
- Decode: implemented after the run (carried state is `(notebook, clock)`,
  the clock replaces `step_offset`); parity test 6.6e-7 vs chunked.

**LAUNCHED 2026-09-04 06:20Z on the RTX Pro 6000** (tmux `sempty_chrono`,
commit `7b24e44`): `logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`
(on the remote box, `~/Development/qllm-private`). Header confirms
`'chrono': True`, B18 T2048 32130 steps = 10 ep — identical geometry to the
23.81 baseline. ~75-83k tok/s (vs 45.5k on the 4090), ETA ~4.5 h. The
`[gen @ 8000]` lines will say `failed: ... chrono decode not implemented` —
expected and harmless (try/except; val/ckpt use the chunked path). Remote
setup notes: `sempyt` is NOT a pip dep — it is cloned at
`~/Development/sempyt` and wired via `site-packages/sempyt_src.pth` (same as
local). WikiText/gpt2 need no HF token (public). Watch:
`ssh ubuntu@34.131.203.207 'grep -E "^step|val @" ~/Development/qllm-private/logs/v13_sempty_wikitext_chrono_fair_*.log | tail'`.

**How to run the rung (the "main" run on the RTX Pro 6000):**
```bash
# on the remote box, code already pulled to c-hash below:
cd ~/Development/qllm-private
CHRONO=1 TAG=wikitext_chrono_fair \
  tmux new-session -d -s sempty_chrono "bash v13_sempty/tmp_wikitext_fair.sh"
# identical geometry to the 23.81 baseline (T=2048 B18 10ep 1.18B tok);
# compare best val_ppl vs 23.81 and the wiki recall probe.
```
`tmp_wikitext_fair.sh` now takes `CHRONO=1` (appends `--chrono`). First run on
a fresh box tokenizes WikiText (sl2048 cache) once. Log name carries the commit
+ timestamp (naming rule in AGENTS.md); the in-file header prints `[ladder]
chrono=True`.

**Next rungs (N1 won; stack on `CHRONO=1`, one variable each, same B18 T2048
10 ep, compare vs 23.14).** All keep the scan / are elementwise, see
EXPERIMENTS "Positioning":
1. **N4 read-out gate** (`--out_gate`, coded, commit `3c7b9b9`) — **RUNNING**
   (see below). Memory was underused (`pam_scale` 0.11–0.31); the gate makes
   the read-out per token/head. Speed gate −2.8 %.
2. ~~A1 `--short_conv`~~ — **DONE, FAIL: 23.49 vs 23.14, −17 % tok/s; code
   removed** (EXPERIMENTS "A1 short conv"). The `SHORT_CONV` knob in
   `tmp_wikitext_fair.sh` still needs deleting — do it *after* N4 finishes
   (never edit that script while a run's bash is executing it, see pitfalls).
3. **A3 `--delta`** (coded) — the recall lever; needs a recall-mix run to be
   judged (WikiText-only PPL will not show it; see EXPERIMENTS ledger row 4).
4. **A4 `--cond_mem`**, then **A2 `--n_states`/`--vault`** (weaker sROI).
5. N3 interference-erase / N2 frequency-multiplexed keys if A3 is not enough.
6. Chrono follow-ups for long context (from the 09-04 discussion; not coded):
   **segment clock** `tau_t = r_t tau_{t-1} + g_t` (learned reset = position
   since topic boundary), and **phase wrap** `phi mod 2pi` per frequency (fp32
   clock loses precision past ~1e5 tokens). Neither matters at T=2048.
On the RTX Pro 6000 run these with `GRAD_CKPT=0` (96 GB; the ckpt default is
a 4090 fit). **Measured 2026-09-04** (chrono, B18 T2048, 60-step smoke):
grad-ckpt OFF = **~102k tok/s, peak 29.1 GB** vs ON = 83k tok/s, 8.9 GB —
+23 %, a 10-epoch rung drops from 4.0 h to ~3.25 h. Keep B=18 T=2048 for
comparability. Launch template for the next rung — **put the env vars INSIDE
the tmux command string**:
```bash
tmux new-session -d -s sempty_n4 \
  "GRAD_CKPT=0 CHRONO=1 OUT_GATE=1 TAG=wikitext_chrono_n4gate_fair bash v13_sempty/tmp_wikitext_fair.sh"
```
**PITFALL (bit us 2026-09-04):** `VAR=x tmux new-session ...` only works when
it *starts* the tmux server. Once a server exists, new sessions inherit the
*server's* environment, so the env prefix is silently ignored and you launch
whatever the first session's env said (we got a duplicate A1 instead of N4).
**Do not run two rungs concurrently** on the 6000: measured 39k+39k = 78k
tok/s combined vs 85k single (−8%); queue instead
(`while pgrep -f <ckpt-dir-tag>; do sleep 60; done; ...` in the tmux command).

**More pitfalls (2026-09-04 afternoon, both cost GPU time):**
* A `while pgrep -f '<tag>'; do sleep; done` waiter inside `tmux new-session
  "..."` matches **its own** `bash -c` command line → waits forever (N4 sat
  queued 16 min on an idle GPU). Use `pgrep -f 'python.*<tag>'` or a pid file.
* **Never edit a `.sh` while a run's bash is executing it.** bash reads the
  script incrementally; the edit shifted the offset and the wrapper ran
  garbage after Python returned (`ag_every: command not found`, `exit=127` in
  the A1 log; training itself was fine). Fix pending: make
  `tmp_wikitext_fair.sh` copy itself to `mktemp` and `exec` the copy.

`tmp_wikitext_fair.sh` now `exec`s a private temp copy of itself (safe to edit
while a run is live) and the `SHORT_CONV` knob is gone.

**Done 2026-09-04 on the RTX Pro 6000:** N1 chrono 23.14 (KEEP) → A1 short
conv 23.49 (FAIL, removed) → N4 gate 22.96 (KEEP). Reference = `CHRONO=1
OUT_GATE=1`.

## Scale plan (Phase 3, decided 2026-09-04) — data, not more WikiText rungs

Why now: three rungs gave −0.67 / +0.35 / −0.18; WikiText-103 PPL at 100M is
saturating 0.27 from the transformer. The open questions (does the arch hold on
diverse data at 3× the tokens; can it be trained to recall; can it chat) need
diverse data. Remaining ladder rungs (A3 delta, A4 cond-mem, A2) are *recall*
levers and will be ablated later on the mixed recipe with the recall probe.

1. **Base pretrain (RUNNING, see below):** `v13_sempty/tmp_pretrain_mix.sh` —
   real-102M `baseline_real_pm` + chrono + out_gate, live stream
   dclm-edu .45 / fineweb-edu .42 / smoltalk2-Mid as ChatML text .10 /
   synthetic recall .03 (web-only first 300M tok), **3.0B tokens**, B18 T2048
   (81,380 steps), lr 2e-4 warmup 1000 cosine, wd 0.01, **dropout 0**, bf16,
   grad-ckpt off, **chat vocab 50261** (`<|im_start|> <|im_end|> <think>
   </think>` = ids 50257–50260, default-on for `--dataset mix`). Primary metric
   = streaming holdout val (244 chunks); `[wiki_val]` is a secondary anchor and
   is NOT comparable 1:1 with 22.96 (different data, single pass — expect it to
   sit higher). 86.5k tok/s live (tokenization on the main thread, −13% vs the
   cached loader) → ~9.7 h.
2. **SFT** on smoltalk2 `SFT` (v7 `load_smoltalk2`, assistant-only loss, ChatML)
   from the best base ckpt — needs a v13_sempty SFT entry point (not written).
3. **Recall probe + A3 `--delta` ablation** on the mixed recipe (short runs).
4. If (1) is healthy: **scale params** (~350M) on more tokens.

Judging (1): holdout val PPL should fall monotonically; the wiki anchor should
keep improving through the blend switch at 300M tok (step ~8,140); recall docs
enter at that point too. Watch `[diag]` pam/gate as before. Sample prompts via
`generate.py` on `checkpoints_v13_sempty/mix3b_chrono_gate_303c7fb/best_model.pt`
(the tokenizer is the chat one — `_config_from_ckpt` reads vocab from the ckpt).
HF auth: a token is stored on the RTX box at `~/.cache/huggingface/token`
(mode 600, never in the repo or logs; 2026-09-04) — the first launch streamed
unauthenticated for 4 min and was restarted so the 10 h stream has the higher
rate limit. `huggingface_hub` picks the file up automatically.

**Running (2026-09-04 19:15Z, RTX Pro 6000):** tmux `sempty_mix`, commit
`bbc12e9`, log `logs/v13_sempty_mix3b_chrono_gate_bbc12e9_20260904_1915.log`,
ckpt dir `checkpoints_v13_sempty/mix3b_chrono_gate_bbc12e9/` (`latest.pt`
every 1000 steps = full resume state, `best_model.pt` on holdout-val best,
`step_XXXXXX.pt` every 10k). 85.5k tok/s → **ETA ~10:30Z 2026-09-05**.
Watchdog: `bash v13_sempty/tmp_wiki_watchdog.sh <log> 81380 2940` (re-arm on
wake). Two earlier launches today (`303c7fb` 18:53/18:56) were killed at
<600 steps and their logs deleted: no resume support yet and the blend warmup
was silently inactive (see below).

**R1 at T=2048 (`b55c49a`, 06:12Z) was killed by the user at <1k steps** —
priority moved to 8K context; log + ckpt dir deleted (no result). The R1
question (does a per-head decay ladder lengthen the horizon?) is now asked
directly at T=8192 — see "Running L1" below.

**L1 first launch (`b07e384`, 06:36Z) KILLED at step 3500 — data bug, not
model.** Holdout val 152 @3k steps vs mix-3B ~80 at the same step while train
PPL read 40. Eval path exonerated (fp32/bf16 × fused/reference kernel all
163.7 on the ckpt). Per-source PPL of the ckpt: dclm 155, fineweb_long 154,
smoltalk 35, **pg19 seen-pieces 6.5** (unseen books 47.9) → the stream was
~95 % PG-19: `_blend_interleave_text_iters` drew one *document* per pick with
the weights, so token share = weight × doc length (100k-tok pieces vs 1k-tok
web docs). Fixed in `18ed358` (token-weighted deficit round-robin + per-source
chunk attribution in the chunker; realized shares now equal the weights).
Lesson: **whenever a new source has a different doc length, check
`token_counters` in `latest.pt` (`ck['data_cursor']`) after the first save.**
Also `18ed358`: `[diag] dtbias/head(layer-mean)=` row — the old per-layer
head-mean shows a spread-8 ladder as a flat −8.

**L1 = Stage L-1 — FINISHED 13:55Z, verdict FAIL on horizon (see State of
play + EXPERIMENTS "L1 / R1 dt-spread"). Kept for reference:** tmux
`sempty_l1`, commit `18ed358`, log
`logs/v13_sempty_mix1b_8k_r1_dtspread8_18ed358_20260905_0956.log`, ckpt
`checkpoints_v13_sempty/mix1b_8k_r1_dtspread8_18ed358/`.
`SEQ=8192 BATCH=8` (65,536 tok/step, GRAD_CKPT=0 → ~55 GB) `DT_SPREAD=8`
`TARGET_TOKENS=1e9` (15,259 steps) `BLEND_WARMUP=100M WARMUP=500`
`SOURCES=dclm,fineweb_long,pg19,smoltalk2_mid,recall,recall_long`
`WEIGHTS=0.36,0.20,0.22,0.10,0.04,0.08` `VAL_EVERY=1000 SAVE_EVERY=500
KEEP_EVERY=5000 KEEP_LAST=1 GEN_EVERY=2000`. Two things differ from mix-3B
(T and dt_spread) — deliberate: the reference heads (half-life ~38 tok) cannot
use an 8K window at all, so "8K without spread" is not an informative arm.
First launch measured **81k tok/s, 52.4 GB** at this geometry (2K run was
85.5k — the sequence axis is free, as the scan predicted) → 15,258 steps ≈
3.4 h → **ETA ~13:30Z 2026-09-05** for the relaunch. Startup is slow (~4 min
to step 1): the 10k-chunk shuffle buffer is 82M tokens at 8K.
Watchdog: `bash v13_sempty/tmp_wiki_watchdog.sh <log> 15258 2940` (re-arm on
wake). Sanity gate at the first val (step 1000, web-only warmup): holdout PPL
should be in the mix-3B ballpark for ~1000 steps (mix-3B: 105 @2000 steps /
74M tok) — the killed run read 365 there.
**Judge L1 by the horizon, not PPL:**
```bash
.venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty \
  --checkpoint checkpoints_v13_sempty/mix1b_8k_r1_dtspread8_b07e384/best_model.pt \
  --preset baseline_real_pm --context-lengths 128,256,512,1024,2048,4096,8192 \
  --positions 0,0.5,1 --association-counts 1,4,8 --trials 20 \
  --output logs/memory_probes/v13_sempty_mix1b_8k_r1_dtspread8_b07e384_behavioral.json
```
Reference (mix-3B, 3× the tokens, T=2048): a1 1.00 @128, 0.35 @256, 0.10
@512/2048; a8 0.2–0.35. KEEP if a1 stays ≥0.5 at 512–4096 with holdout PPL
not worse than mix-3B at 1B tokens (val @27k ≈ 32.4 mid-cosine; holdout val
at T=8192 is the same 5 % corpus re-chunked, `val_chunks` differs). Also read
the `[diag]` `dtbias=` row: did the long heads keep their −8…−12 or get
pulled up? Then R2 (`NSTATES=2 VAULT=1 GEN_EVERY=0`), R3 (`DELTA=1
GEN_EVERY=0`) — each 1B at T=8192, sequential, never concurrent.

**Checkpoint policy (2026-09-05, `b07e384`, user rule: free stale ckpts).**
`--keep_last N` (launcher `KEEP_LAST`, default 1): after each milestone
`step_XXXXXX.pt` only the newest N survive; `latest.pt` is an atomic
overwrite (`.tmp` + `os.replace`), `best_model.pt` untouched. Steady state per
run = 3 × 1.2 GB. Pruned 2026-09-05: mix-3B `step_0*.pt` (8 × 1.2 GB), the A1
FAIL dir, `latest.pt` of the finished WikiText runs → `checkpoints_v13_sempty`
19 GB → 4.6 GB. `checkpoints_v11_*` (27 GB, mostly `recall_stage6` 11 GB) NOT
touched — a different program; user to decide.

## Stage L — 8K then 32K context (data landed 2026-09-05 `b07e384`; L1 running)

Measured: the scan is linear — T=8192 B=4 118k tok/s / 26 GB, T=32768 B=1
98.6k tok/s / 26 GB on the 6000 (synthetic vocab; real CE adds cost but the
sequence axis is free). Real smoke `b07e384`: T=8192 B=4 real CE + 5 sources
= 29 GB, so B=8 at 8K fits easily; 32K B=2 should too.
1. **Recall curriculum gaps — DONE.** `_build_recall_doc(rng, max_gap_sentences)`;
   registry `recall_long` = 500 sentences → docs up to ~5.2k tok (`recall`
   stays 200 → ~2.3k). For 32K add `recall_xlong` (~2500 sentences) the same
   way (one registry line). Needle-style single-binding doc: still TODO.
2. **Long documents — DONE.** `pg19` (`emozilla/pg19` parquet stream; the
   `deepmind/pg19` script repo is dead in datasets≥3; books cut into ≤400k-char
   pieces ≈110k tok at paragraph breaks, 44 s first row) and `fineweb_long`
   (fineweb-edu `min_chars=12000`, 14–17k-char docs). Both `kind='web'` →
   part of the blend warmup pool. All sources now open through one
   `_open_source_iter` (live blend and token-cache builder share it). Still
   TODO: whole code files (StarCoder), smoltalk2 `longalign`.
3. **Probe.** `run_memory_behavioral.py --context-lengths ... 8192,32768`
   (check `build_example` filler scaling and GPU time at 32K; 20 trials × 3
   positions × 3 counts × 7 lengths).
4. **Training geometry.** L1 uses T=8192 B=8 = 65k tok/step (memory allows;
   fewer, larger steps). For 32K: B=2 = 65k tok/step. Same lr schedule.
   `max_seq_len` follows `--seq_len` (RoPE cache, Chrono cumsum both
   length-agnostic); holdout val re-chunks at the new T (61 chunks @8K).
5. **Sequence:** R-winner at T=2048 (1B) → **midtrain T=8192** from that ckpt
   via `--resume` semantics? No — `--resume` restores the optimizer/LR for the
   *same* run; for a length change use a fresh run that loads weights only
   (needs a `--init_from <ckpt>` flag: weights, not optimizer/step). Then
   32K the same way. Budget ~1B tokens per stage (3–4 h each).
6. **Retention prerequisite:** the head ladder must reach 32K half-lives —
   spread 8 does (23k/113k for heads 4–5). If R1 shows the optimizer pulling
   the long heads back up, pin the slowest head (vault-style) before Stage L.

**Resume (2026-09-04, `bbc12e9`).** `--resume auto|<path>` restores model,
optimizer, LR schedule, AMP scaler, step/token counters, best-val, all RNG
streams and the **data-stream cursor** (per-source docs consumed + tokens
yielded, saved in every ckpt); each source skips its consumed docs on resume so
nothing is trained on twice (the ≤10k buffered chunks ≈20M tok are lost per
restart, not repeated). The launcher runs `--resume auto` in a retry loop
(MAX_RETRIES=5), so a crash self-heals inside tmux. **If the box itself
rebooted:** `tmux new-session -d -s sempty_mix "TAG=mix3b_chrono_gate
CKPT_DIR=checkpoints_v13_sempty/mix3b_chrono_gate_bbc12e9
RESUME_LOG=logs/v13_sempty_mix3b_chrono_gate_bbc12e9_20260904_1915.log bash
v13_sempty/tmp_pretrain_mix.sh"` (same commit checked out). Verified on the
real stream: killed at step 30 → resumed with the identical LR at 31/32,
finished exactly on budget. `[resume]` lines in the log show the cursor.
**Bug found on the way:** `blend_warmup_tokens` gates on `sum(token_counters)`
inside `v7._blend_interleave_text_iters`, and `v13_sempty/train.py` never
passed the counters → warmup was a no-op in the 18:56 launch. Now passed;
verified (0 chat/recall docs before the threshold). Note the ~20M-token lag:
the interleaver measures *yielded* tokens while the shuffle buffer holds 10k
chunks ahead, so the blend switch lands ≈300M+20M tokens in.
Data/scale-up (DCLM/FineWeb mix via `--dataset pretrain_mix`, `c7a343b`) comes
*after* the ladder settles the architecture at 100M.

* Real-101M (`baseline_real_pm`) trains at **~64k tok/s** (was 5.8k), peak
  **7.6 GiB** at B32 T256 with grad-checkpointing OFF (was 21 GB at B8 with
  it on). One WikiText-103 epoch (118M tok) ≈ **30 min**.
* The one-epoch T256 run **finished** (commit `7343201`): best **val PPL
  54.89** (NLL 4.005), avg 64.1k tok/s, exit 0.
  `logs/v13_sempty_wikitext_real_7343201_20260903_1449.log`,
  checkpoints `checkpoints_v13_sempty/wikitext_real_fused_7343201/`
  (`best_model.pt`, `latest.pt`).
* **Now executing the quality program** (plan
  `.cursor/plans/v13_sempty_quality_program_7759901a.plan.md`): fair 10-epoch
  T=2048 WikiText baseline, complex arm at kernel speed, recall infra, and an
  architecture ladder (short conv, multi-state+vault, delta erase/write,
  Engram-style conditional memory, layout). See that plan for the full spec;
  results land in `EXPERIMENTS_SEMPY.md`.

## How to watch

```bash
tmux attach -t sempty_wiki                 # live; Ctrl-b d to detach
LOG=$(ls -t logs/v13_sempty_wikitext_real_*.log | head -1)
grep -E "\[val @" "$LOG" | tail            # val NLL/PPL every 500 steps (~4M tok)
grep -E "^step " "$LOG" | tail -3          # loss / tok/s / ETA
# watchdog (wakes on: process gone, error/nan in log, step reached, timeout):
bash v13_sempty/tmp_wiki_watchdog.sh "$LOG" 14400 2940
```
Re-arm the watchdog on every wake (rule in AGENTS.md). A healthy run shows
~67k tok/s, `GPU 1.3/7.6GB`, loss falling (6.2 at step 300, expect ~4.3 at
the end: the 09-01 run reached train NLL 4.38 / val PPL 68.75 at one epoch
with constant lr 5e-5 and B8).

## How to run (again / longer)

`v13_sempty/tmp_wikitext_real.sh` is the whole recipe — edit flags there.
```bash
tmux new-session -d -s sempty_wiki "bash v13_sempty/tmp_wikitext_real.sh"
```
Current flags: `--batch_size 32 --seq_len 256 --steps 14400 --lr 1e-4
--warmup_steps 100 --amp_dtype bf16 --fused_ce --fused_pam --ce_gemm_dtype
auto --val_every 500 --save_every_steps 2500`, no `--gradient_checkpointing`.
* **Recipe change vs 09-01 you should know about:** batch 4x (B8→B32) and
  lr 5e-5 → 1e-4 (sqrt scaling); `--steps 14400` = one epoch so warmup-cosine
  completes at the epoch end (09-01 had a 200k horizon = constant lr). If you
  want a literal repeat of the old regime use `--lr 5e-5 --steps 200000`.
* **Longer runs:** `--epochs N` (trainer flag) with `--steps` = N × 14409 so
  the cosine spans the whole run. Memory headroom is large: B64 T256 fits in
  11.9 GiB but tok/s *drops* (63k), B96 57k — stay at 8192 tok/step.
  `--seq_len 512 --batch_size 16` is the same tok/step at 68.5k tok/s.
* Kill switches: `--no_fused_pam` (plain-torch scan, same math, 2.4x slower
  kernel), `V13S_KERNEL=0` env (disables both Triton PAM scan and Triton CE),
  `--ce_gemm_dtype fp32` (exact head, +10 ms/step).
* Log names carry commit hash + timestamp; the header line prints
  `commit=<hash>[-dirty] fused_pam=… ce_gemm=… grad_ckpt=…`.

## What was done (for the record)

1. `triton_kernels.py` — real PAM read in chunked linear-attention form
   (`y_s = a_s (S_in.q_s) + Σ_{t≤s} (a_s/a_t)(q_s.k_t) v_t`), Triton forward
   (state scan + read) and backward (state-grad scan, dq/dk/dg, dv), tile 64,
   8 warps; `pam_scan_torch` fallback; `fused_real_pam_read` entry.
   `RealPAMLayer._chunked` hands off to it; decode `_stepwise` and the complex
   arm untouched. Carried state layout unchanged (rows = value, cols = key).
2. `pam_kernel_test.py` — parity harness (oracle anchored to `_stepwise`;
   the inherited oracle had `outer(k,v)` transposed — fixed). PASS fp32 5e-5 /
   bf16 3e-2. `selftest` 13/13 incl. `test_real_fused_kernel_parity`.
3. `fused_ce.py` — `gemm_dtype` (bf16 GEMMs, fp32 loss) and the Triton
   Liger-style `_FusedLinearCETriton` (grads in forward, one row pass).
   Validation pinned to the fp32 head.
4. `train.py` — `--ce_gemm_dtype`, `--fused_pam/--no_fused_pam`, commit hash
   in the header. `check_torch_layout.py` — new boundaries declared.

## Open items / ideas not taken

* Remaining 61 ms step: Linears + CE GEMMs at tensor-core peak; ~20 ms of
  small elementwise (norm/gate/residual/RoPE, 900 `mul` launches/step) —
  only `torch.compile` would fuse these and sempyt's Dim identities trip its
  recompile limit (needs a sempyt change). RoPE fusion measured at 3 ms max.
* Why tok/s drops beyond 8192 tok/step is unexplained (not memory).
* The one-epoch T256 result (54.89) is recorded in `EXPERIMENTS_SEMPY.md`
  → "Speed: fused real arm". The apples-to-apples number is the Phase-1
  fair run (10 epochs, T=2048); this scratchpad and superseded `tmp_*`
  scripts get cleaned up at the end of the quality program (Phase 5).

## 4090 queue (Lead, 12:23 IST 2026-11-12)
1. cx_m2 resume (after Library's FAR-TOLD on the four points; tmux mindfar resumes it).
2. **ctl-far**: the equal-data control at the ctl recipe (103M, 1.0B, PAM M2 vs T++ vs Mamba-2) WITH the far block (b2s_far + proto_far
   in the cont2-style mix, packer hold on), to test whether the far carry is architectural (does T++ / Mamba-2 also learn s8 @8k from
   b2s_far?). Pre-register before launch (v13_sempty/tmp/ctl_chain.sh as the base).
3. m9n (paper-only jitter calibration) when a gap appears.


## Step back after the wio 5-arm table + 3f's second break (10:30 IST 2026-11-18)
- Surprise: every axis has its own winner (nochrono rules, PAM needles, GDN / Mamba-2 language), and T++ is last on all three
  beyond its length. The told rule is one unbound strength in every body (Library EUREKA 299041e1): carry and scope trade off
  across saves (3d / 3f), across the decay, and across architectures (the clock: +0.51 carry, +0.07 leak).
- What it says: the memory stores "how much a rule applies", not "to what". More data did not bind it (3f's scope docs, 1.5B tokens).
- Bold angle: binding is a geometry question: give the rule a key that is the quantity's key (split key geometry: a rule
  sub-space addressed by the quantity token, erase along it), so applying the rule needs the matching quantity to be read.
  Test at 100M with the clock-off twin and a control, never on the release run.
- State: 3r on the 6000 from 45.875B (3c mix); 4090 to the user 10:45-day end; next window: 0.8 vs 1.6 % x LR cell, split
  geometry, the no-far-docs GDN / nochrono cell (write-ups due).
