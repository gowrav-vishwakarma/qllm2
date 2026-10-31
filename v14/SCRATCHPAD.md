# v14 handover

Read `README.md` (rules), `EXPERIMENT_INDEX.md` (one row per idea) and
`EXPERIMENTS_V14.md` (do-not-repeat digest and pre-registered bars) first.
Keep all three current as things happen (user rule, 2026-10-29).

## State (2026-11-01 01:10 IST) — read this first

Session: **Library**. Nothing of mine is on a GPU. The 4090 queue (Lead's):
S2 (strict, seed 23000, `P_free_sent` primary) → natq rows (b2p2 lenient, T++,
Mamba-2, seed 24000) → **A4 mind run** (`v16/tmp_pkg_a4norm.sh`, PACKAGE_STRICT=1
PACKAGE_NORM=1, seed 24000) with my **n11rf** (tmux `n11rf`, worktree
`../qllm_v14n11f`) allowed beside it.
* **Strict mode verdict:** NOT met as registered (kv natural free 0.85: the
  right row each time, the first word was " a" / " the" / " refers"); neg
  abstain 1.00 met; strict halves para / tool. S2 re-asks it with the sentence
  scorer on a fresh seed.
* **A3 (abstain from the mind's read gain):** fails in-sample (+0.062 < +0.10).
  The frozen rule still gets scored on S2's rows when Lead sends them.
* **A4 (normalised-word organ in the session, `PACKAGE_NORM=1`, fa95777,
  plumbing):** text check MET on bench's own docs (para / tool / kv right row
  1.00; neg hit 0.00 / 0.025). Mind run queued; bars `P_free_sent` para / tool
  ≥ 0.80, neg abstain ≥ 0.90. `v14/a4_check.py` rebuilds any bench cell's docs
  on CPU (verified 320 / 320 identical).
* **When rows land:** score S2 (+ A3 frozen rule on it), A4 mind run, n11rf.
* **Waking peers:** `scripts/wake_session.sh <Name> <id>` (memory
  wake-sessions); check ListAgents / tmux first.

## Step back (2026-11-01 01:10 IST, after strict mode, A3 and A4)

* **What surprised me.** (1) With the right row in front of it, the mind often
  answers in its own words (" a crimson ...") or re-reads the key (" refers"):
  "generation counts" cuts both ways, the one-token scorer undercounts it.
  (2) A row for a DIFFERENT key raises its value about as much as the right
  row does (gain median 3.2 vs 4.8 nats): the mind copies the nearest value
  and never checks that the row's key is the question's key. (3) On every
  templated cell, text alone decides "is it stored" (A4).
* **What it says.** The package splits along a clean line. TEXT decides
  whether a fact is stored when the words overlap; the MIND decides what the
  row says and says it. The only place neither can decide is a fact stored
  under DIFFERENT words (synonyms, rewording), and that is the only place the
  learned session key could earn its keep. No bench cell has it yet: natq
  rewords the frame but keeps the key words, so a rare-word text match can
  find it.
* **Bold angle 1 (for Mind, curriculum, not my lane):** teach the mind to
  VERIFY. Put a near-miss row (another key's fact) in context with an
  explicit "unknown" target. Abstain then comes out of the mind's own
  generation, the channel the user counts, and it would work under rewording
  because the mind reads meaning, not words. Prior work: SQuAD 2.0
  unanswerable questions (Rajpurkar 2018), R-Tuning (Zhang et al. 2023),
  Self-RAG's relevance token. Difference: the verifier is the O(1) mind
  reading a fetched row at question time, closing the abstain gap that text
  and similarity cannot.
* **Bold angle 2 (my lane):** a SYNONYM cell, where the fact is stored under
  one phrase and asked with a synonym phrase sharing no content word ("old
  ship" / "ancient vessel"). That is the real test of "a mind's key carries
  meaning": organs and BM25 must miss by construction; MiniLM sentence
  embeddings are the comparison row. Prior work: dense retrieval (DPR,
  Karpukhin 2020). Difference: the key is the O(1) mind's own reading state,
  with no separate encoder.

## Old state (2026-10-31 23:50 IST; superseded)

Session: **Library**. Nothing of mine is running. The 4090 belongs to Lead's
package table until ~00:50 IST; its 32k results ~04:00, 128k ~midday.
* **Package (handed to Lead 23:05):** session library = mind-state keys
  (L1 head, shipped mean, PACKAGE_T = −1); shared library = SL2 text fetch
  over suffix arrays (~11.4 MB per 1M tokens, 0.99-1.00 at 4000 entries on
  exact + surface variants); standing skills = N11-R refresh every ~2k tokens
  (32k: 0.96 / 0.80, +1.6 % tokens); explicit questions fetch at question time.
* **Verdicts today:** L1 Bar A not met (synthetic), natural 0.94-1.00 in-doc;
  L2 paraphrase not met; L3 not met; SR-0 not met; A2 abstain infeasible;
  SL1 not met → SL2 met; N11-R met. EUREKA: the mind's state is a stream
  address (a reader, not an embedder).
* **Open (board):** paraphrase by synonyms / rewording (no cell yet); "nothing
  stored" for a paraphrased question (similarity cannot; the simple gain idea
  failed Lead's smoke; null-row gain and top-5 outlier still to test OFFLINE on
  the table's P_cands); BM25 postings memory if the fallback stays.
* **Added 00:50 IST:** (1) N11-R's verdict is the 20-way pick; its FREE-generation
  rerun N11-R/F (pre-registered 00:49, seeds 8150-8199) is queued in tmux `n11rf`
  (worktree `../qllm_v14n11f`) behind Lead's strict + 128k fill jobs; the
  board says "20-way" until it reports. (2) Lead adds a natural-question cell
  `natq` to v16/bench.py (fact in prose, plain question, free generation to the
  first "."); MY next unit once it exists: the session key from a SALIENT token
  chosen in a free-form question, against the shared library's BM25 path as the
  first comparison row. (3) Strict mode (pre-registered 00:34, amended to free
  generation S/A1 00:39) runs in Lead's queue with PACKAGE_STRICT=1; Lead
  sends the cells, I write the verdict.
* **Next when Lead hands out work:** the offline gain tests on P_cands; N11-F
  is measured by the package table itself (no separate run).
* **Waking peers:** `scripts/wake_session.sh <Name> <id>` (memory
  wake-sessions); check ListAgents / tmux first.

## Old state (2026-10-31 23:10 IST; superseded)

Session: **Library**. Nothing of mine runs on the GPU; the CPU lanes are done.
* **Handed to Lead (23:05) for the package table (session library):**
  PACKAGE_HEAD = `qllm_ckpt/v14/l1p_fwd_L8c_99e6803.pt`, PACKAGE_MEAN =
  `...99e6803.mean.pt` (bench `--centre_ref ...mean_raw.pt`), PACKAGE_T = −1
  (abstain disabled when the organ misses: no threshold met both A2
  constraints), `--key_fn v14.package_fetch:key_fn`. Lead runs it on fresh
  seeds 18000+ against T++ / Mamba-2.
* **Verdicts today:** L1 Bar A not met (synthetic), natural 0.94-1.00 in-doc;
  L2 (two-organ) kv / neg met, paraphrase not met; L3 (cross-doc head) not
  met; SR-0 not met (residual vector cannot carry a rule); A2 abstain:
  infeasible. EUREKA: the mind's state is a STREAM ADDRESS, not an embedder.
* **Next, in order (Lead approved):** (1) SHARED library fetch by text: the
  exact organ with whole-key matching, a normalised lexical match for
  paraphrase, an off-the-shelf text embedder as a named baseline row
  (plumbing, cited, no claim); (2) N11 planner with two rules: fetch a fact,
  and REFRESH a skill pack every ~2k tokens (Mind's bar-3 result); its
  selector must not rely on the mind's confidence over natural spans;
  (3) SR-0b once, eval-only, as a product-option test (Mind's M2 streaming
  `41f8ae8`), only when the card is free and Lead says go.
* **Tools:** `v14/pack.py` (pack format + install, 28 MB per 1M tokens),
  `v14/pack_check.py` (cross-document check), `v14/package_fetch.py`
  (harness key_fn), `v14/l2_select.py`, `v14/l1p1_select.py`, `v14/sr0.py`.
* **Worktrees:** `../qllm_v14dev` `../qllm_v14eval` `../qllm_v14p1`
  `../qllm_v14ref` `../qllm_v14tune` `../qllm_v14dev2` (all idle). Push from a
  scratch worktree when a worktree has running jobs; in the shared checkout
  stage only my own paths.

## Old state (2026-10-31 21:25 IST; superseded)

Session name: **Library** (peers: **Mind** = v13, **Lead** = organiser and v16
lead). The local tmux server was wiped at ~21:00; nothing of mine is running.
The 4090 is free.

* **L1 verdicts (EXPERIMENTS_V14 "L1 VERDICTS", `5a29519`):** Bar A NOT met
  (synthetic 0.81 / 0.56 at 32k / 128k); natural keys 0.94 / 1.00; Bar P met on
  natural paraphrase (0.81 vs organ 0.31); P0, P1, concat-5, L1-256 not met.
  Best head: `qllm_ckpt/v14/l1p_fwd_L8c_99e6803.pt` (d_drive), used via
  `v16/bench.py --vec_layer 8 --centre --key_fn v14.l1_key:key_fn_bursts`.
* **SR-0 NOT met (`c8247b4`):** a residual rule vector cannot switch the skill
  on even at L = 0; the rule lives in the notebook state. SR-0b (hold the
  notebook state through the filler) needs an M2 decode / freeze hook in
  Mind's body code; proposed to Lead.
* **Proposed next (not pre-registered yet):** two-organ fetch (exact organ on an
  exact match, else the learned key; abstain on raw similarity), on a fresh
  seed; field-aware rows for the tool cell.
* **Worktrees:** `../qllm_v14dev` (paired-run code, `2cc01c5`), `../qllm_v14eval`
  (P0), `../qllm_v14p1` (P1), `../qllm_v14dev2` (sr0 / L1-256 / selector).
  Rebase a worktree only when nothing runs from it; push through a scratch
  worktree otherwise. Logs are all copied into `logs/`.
* **Lessons today:** check for self-matching (rows overlapping the query;
  `pkill -f` patterns that match the killer's own command line); zsh aborts a
  whole command on an unmatched glob (use `bash -c` with `nullglob`).

## Old state (2026-10-31 13:10 IST; superseded)

**Direction (user, 10-31): O(1) mind + growing library. v14 = the library
(learned key, one row per event, write gate, planner, shared-library
schema); v13 = mind + benchmark MLB-1 + 600M/3B; the Communicator = the
harness `v16/bench.py`; accepted units move into `v16/`. Do not send any of
this to v15 (Evol V1).** Plan sheet: `PLAN_MIND_LIBRARY.md`; benchmark:
`v13_sempty/BENCHMARK_MIND_LIBRARY.md` (v0.2).

* **Done today:** N9a recall replication n9r = FAIL (+0.016 / +0.014 vs bars
  +0.08 / +0.15; the control's seed spread is 5-10x the effect). n9th seed
  131: N9s two_hop 0.89 vs N9a 0.28, tpp 0.30 (second formed seed). `n12m`
  cancelled. W0 (write-gate eval) done, bar not met: the frozen surprise
  gate fires on the first word of an inserted sentence (topic break), not
  on the key; rows must be burst SPANS (amendment A3).
* **Running:** 4090: v13's a4f (A4 cond_mem 600M, ~12:45-16:15 IST) + my
  `n9th4090` seed 132 (micro, 4090, stops itself after seed 132 via tmux
  `n9thstop4090`, ~14:00 IST; the 6000 ran seed 131 only). v15's pair waits
  for the card. Nothing else of mine runs.
* **Decisions (organiser, v16 lead, 13:25 IST):** N9s gets NO standalone
  600M run; at most one paired arm inside v13's curriculum run that already
  carries two_hop-format data (proposed to v13). Its "anchor kept alive"
  idea goes into the planner (N11). L1 data pass approved on the 4090 now
  (hard cap 2.7 GB, guards: stop if a4f < 70 % of its start tok/s or free
  VRAM < 1.5 GB); tmux `l1data`, script `~/l1_data.sh`, vectors on d_drive
  `qllm_ckpt/v14/l1_cache/{train,val}`, log `logs/v14_l1_data_8445f73_*.log`
  in `../qllm_v14dev`.
* **Next unit L1** (pre-registered, EXPERIMENTS_V14 section "L" + A3): a
  learned key over BURST SPANS on the frozen C1-ctl (`c1ctlb_main_7ae6f1f`,
  /media/gowrav/d_drive/qllm2/checkpoints_v13_sempty/). Plan:
  1. Training pool: stream fineweb (skip 1M docs) into a token cache; never
     the held-out prose. 2. Build training docs with `v16/bench.py`'s
     builder (`build`), vectors from `Reader.vectors` (GPU), cache them.
  3. Head: K(p) = attentive pooling over vectors [p, p+24) -> W_k (128-d);
     InfoNCE, query = K at the question's key token, positives = the burst
     row of the answer sentence, negatives = all other burst rows.
  4. Expose as `v14.l1_key:key_fn` for `v16/bench.py --key_fn`, evaluate
     with `--gate_tau 11`, lengths 32k-1M, cells needle / multi / latest /
     two-hop / natural / paraphrase / instr / neg.
  W0 numbers (tau 11 = 0.6 rows per 16 tokens): `logs/v14_w0_c1ctl_232748e_*`.
  `v14/w0_gate.py` imports `session_index_probe`, which moved to
  `v16/bench.py` (d1e1a56): fix the import before rerunning.
* **Dev worktree:** edit in `../qllm_v14dev` (detached; `git fetch` then
  `checkout --detach origin/master`), push with `git push origin HEAD:master`
  after `git pull --rebase`; the shared checkout only fast-forwards.
* **Old state below is kept for history** (some runs there are done or
  cancelled).

## Old state (2026-10-31 04:15 IST; superseded by the section above)

* **4090:** v14's one job = **n9r** (tmux `n9r`, worktree `../qllm_v14n9r`
  at `37fa333`, checkpoints on d_drive `qllm_ckpt/v14_n9r`, driver
  `~/n9r_chain.sh`, chain log `logs/v14_n9r_chain.log` in the worktree):
  ctl43 then n9a43, stage A + B each, then probes; started 04:04 IST,
  ~47k tok/s, ends ~12:00 IST. Score with the recall scorer (v13's
  definition: t40 behavioral JSON, accuracy·n over `aggregates`, split at
  context_tokens 2048) against the bars in EXPERIMENTS_V14 "N9a recall
  replication". Beside it: v15's G1r (its own small job).
* **n9th split (06:30 IST):** the 6000 gives it only ~9k tok/s (saturated),
  so the 6000 runs seed 131 only (a tmux `n9thstop` watcher kills `n9th`
  after its DONE line, ~11:30 IST); seeds 132–135 run on the 4090 (tmux
  `n9th4090`, worktree `../qllm_v14n9th` at `161a3c9`, `SAVE=1` to d_drive,
  driver `~/n9th_4090.sh`, log `logs/v14_n9th4090_driver_161a3c9.log`;
  waits for tmux `n9r` to end). Score all 5 seeds together.
* **4090 since 08:51 IST:** v15 left (CPU only, asks before GPU jobs), so
  n9th seeds 132-135 (tmux `n9th4090`, ~4 h) run BESIDE n9r's last stage
  (n9a43 stage B, 08:45 -> ~11:45, then probes). Agreed with v13: its
  **a4f** (A4 cond_mem 600M filter, paired with OUR n9r ctl43; ~10 GB,
  tmux `a4f`) starts when tmux `n9r` ends (~12:00-12:15) and reads ctl43's
  checkpoint / probes read-only: keep the n9r checkpoints. v15's
  `giv2_v15` (2 micro pairs, ~8 GB) is gated on `n9r` ended + a4f training +
  `n12m` ended + 8.5 GB free (~16:00-16:30 start). Then **n12m** (N12 micro screen, tmux `n12m`, worktree
  `../qllm_v14n12m` at `7156302`, driver `~/n12m_4090.sh`, log
  `logs/v14_n12m_driver_7156302.log`; waits for `n9r` then `n9th4090`).
  Dev edits go in `../qllm_v14dev` (detached), never the shared checkout.
* **n9th on the 6000** (agreed with v13 04:10 IST): tmux `n9th`,
  worktree `~/Development/qllm_v14` at `161a3c9`, driver `~/n9th_chain.sh`,
  driver log `logs/v14_n9th_driver_161a3c9.log` (box clock UTC). Arms
  echo_phase / echo_phase_sal / tpp, seeds 131..135 in sequence (each pair
  ~45-60 min shared), bar H in EXPERIMENTS_V14 "N9s". v13's limits: one
  process, ≤ 8 GB (allocator capped, `V14_GPU_MEM_FRAC=0.076`), no
  checkpoints (`SAVE=0`), stop if free disk < 7 GB (driver watcher), pause
  if r3cb (30.2k tok/s at launch) or b2ma (86.7k) stays > 30 % down. Copy
  `logs/v14_n9thp*_161a3c9_*.{log,json}` back to this repo when done.
* **6000:** v13's runs (r3c flagship 3B, b2m Mamba-2 B2 arm, then the `lb`
  label-format pair). v14 asks before using it.
* **v13 news (2526ca7):** the B2 edge over T++ replicates at 600M in both
  seeds (`_new` 0.991 / 0.944 vs T++ 0.734); bar 3 waits on Mamba-2 (b2m).
  step_wide_new (copy the exact last far term through filler) collapses at
  1024 for both PAM and T++: an open cell for a v14 unit.

## Old state (2026-10-30 13:45 IST; superseded, kept for history)

* **4090 queue (15:40 IST):** v13 g1c (whole card) until ~18:00; v14's slot:
  **F1** (tmux `v14f1`, worktree `../qllm_v14f1` at `7c8c6e2`: echo +
  echo_phase × 5 seeds on B1) then **N10 formation run** (tmux `v14n10`,
  `../qllm_v14n10run` at `9bd36e6`, checkpoints symlinked to d_drive: echo /
  op / opcoord × 5 seeds on B2). After g1c, v13 runs echo_qinit (Q1) and
  echo_sparse (S1) on the F1 seeds. **Checkpoints lost** (worktree remove
  --force, 15:40): B2 run 2, state sweep, N7 pair 1 micro weights (numbers
  are in logs/); rerun from the commit if a formed checkpoint is needed for
  probing. F1's checkpoints are inside `../qllm_v14f1`: move them to d_drive
  before removing that worktree.
* **(old, 13:45) 4090 queue:** v13 g1c (whole card, agreed) since 13:35 ->
  ~16:45. Beside it, v14's one small job: **N7 micro run 1** (tmux `v14n7`,
  worktree `../qllm_v14n7run` at `73265e9`, driver `n7r1_driver.sh`; echo /
  echo_n7 / tpp, 3 pairs, bar X) since 13:35, then **N9a micro run 1**
  (tmux `v14n9`, `../qllm_v14n9run` at `b0e7fc6`; echo / echo_phase, 3
  pairs, bar Xa). After g1c, v13 runs **Q1** (tied START of echo q/k,
  formation rate over 5 seeds on B1 + B2; arm `echo_qinit`, `3cf2aa8`) on this
  bench as its small job, results ~21:30.
* **B2 so far:** run 2 R MET (echo infers held-out shifts, T++ only
  selects); state sweep: no reliable size effect, but the O(1) body solves
  held-out shifts FULLY in some seeds (3 of 10 echo-family runs ≥ 0.9; T++
  0 of 4). Formation rate is the open lever (Q1 and N8 target it).
 (6000, `bf59754`, one seed): L and I met (B2 is
  valid).** Held-out rule parameters: echo 0.739, n1sum 0.731, tpp 0.666.
  Numbers (held-out step / affine offset): every arm ~0.9–0.99 (exact
  inference). **Letters (held-out shift): echo 0.33, rising 0.20 → 0.47 with
  shots; tpp 0.02 (picks the neighbour = memorised tables).** First bar-3
  signal for the O(1) body. **B2 run 2** (echo vs tpp, 3 seed pairs
  41/42/43, bar R on shift_wide_new) is pre-registered and goes on the 4090.
  If R is met: a 600M check with B2 docs in training for both bodies
  (design with v13: they offered rules_ood docs in their mix).
* **Closed today:** N6 (B1 run 2), echo_tied (v13 E2 at 600M: label_category
  0.365 vs 0.500, holdout +0.055; the micro-bench win did not transfer), N1
  (B0 run 3 stop rule: n1phase 0.370 < n1sum 0.380), and B0 itself (bench
  broken: echo facts2 0.52, tpp facts1 0.17).
* **tpp calibration on B1:** attention does not form the label_category
  circuit at 27M either (0.42–0.43, 3 seeds), so micro-bench label_category
  does not predict the 600M gap (T++ 0.73).
* **Lesson (E2):** a micro-bench win is a lead, not evidence. Seed pairs
  first, then the LM check.
* **GPUs:** 6000 = v13 S4 chain (~11 h from ~10:00 IST; Mamba-2 600M ×2,
  then the reference at 3B); v14 runs there only from the worktree
  `~/Development/qllm_v14`, never by pulling `qllm-private`. 4090 = v14.
* **Shared checkout rule:** both sessions use
  `/home/gowrav/Development/qllm2`. v14 benches now run from a local
  worktree (`../qllm_v14run`, detached, `.venv` symlinked) so commits to
  master never block a crash-resume.
* **Launcher:** `v14/run_bench.sh` (replaced `run_b0.sh`). It also stops waiting when the
  other run's log prints its DONE line (`WAIT_DONE=file:pattern`): the v13
  T1 tmux stayed open two hours after its run ended on 2026-10-29.
* **6000:** the v13 session's chain. tfs1 (Transformer++ one stage, started
  03:15 IST at `b788731`) -> tfs2 -> c2m2 (ETA ~08:00 IST) -> tfc1
  (Transformer++ on c1m2's aptitude recipe, tmux `s3chain`, ETA ~09:30 IST).
  Their E2 (tied echo, 600M, M1 body, C1-ctl's exact recipe, paired against
  C1-ctl; pre-registered in EXPERIMENTS_SEMPY "E2" on branch `e2`) is gated on
  **B1 run 4's P** (run 3's P was invalid). If P passes, E2 runs on the 4090
  from ~05:45 IST.
* **v13 tfs1 (Transformer++ one stage, 600M, r600, `b788731`; recorded on
  branch `e2` 174c2b8):** holdout 3.4885 (M2 3.4697 beats it, M1 ties
  3.4906), WikiText 66.4 (M2 68.8), recall 0.645 (0.84 inside 2048, 0.15 at
  8192), learning probe 0.217. **label_category 2/4/8-shot: T++
  0.69/0.84/0.73** vs M1 0.65/0.67/0.29, M2 0.59/0.61/0.31, two-stage M1
  0.62/0.71/0.55. Attention does the in-context label lookup that the O(1)
  bodies do not: **this is the gap for v14 to close** (B1's `tpp` arm
  measures the same gap at 27M).
* **v13 reference body = c1m2** (M2 + aptitude two-stage; tie-break set
  before c2m2, `900dde2`). c2m2 (M2, two stages, no aptitude) KEEP: holdout
  3.4497, WikiText 65.57, recall 0.848, label_category 8-shot 0.41. vs T++ on
  the same recipe (tfs2): holdout −0.041, WikiText 65.6 vs 66.2, recall
  +0.145. **First PAM ahead of T++ on holdout AND WikiText at equal data.** T++
  still leads label_category (0.73 vs 0.41) and training speed (1.45× at
  T=2048).
* **v13 S2 fairness (`048f8c2`): MET.** T++ gets nothing from the two-stage
  schedule (tfs2 holdout 3.4907 vs 3.4885 one stage; recall 0.703;
  label_category 8-shot 0.73 unchanged); PAM does. Two-stage M1 vs tfs2:
  holdout −0.016, recall +0.122, but recall ties inside 2048 tokens (0.917
  vs 0.926); the win is past 2048, where T++ collapses (0.15 vs 0.60).
  WikiText and label_category favour T++.
* **B2 with no B2 training (v13 probe on 600M / 3B, `a214002`):** every
  model is below chance (0.04-0.09; chance 0.125), attention included:
  they pick the answer next to the right one. `_new` is 0.015-0.047 harder
  in itself (difficulty control). Numbers in EXPERIMENTS_V14 "B2".
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

## Step back (2026-10-30 ~10:45 IST, after B0 closed, N1/E2 failed, B2 run 1)

Habit (memory `meditate-before-next-step`): after each finding, look from far
and from inside before choosing the next run.

1. **Two kinds of in-context skill pull apart.** Attention wins at *lookup*
   (find this item and copy what followed: label_category at 600M, T++ 0.73
   vs PAM 0.41). The recurrent body wins at *abstraction* (carry one rule
   across examples: B2 letters, echo 0.33 rising with shots, T++ 0.02 =
   per-k tables). Today's failures all tried to buy lookup: the tie made
   q/k symmetric (similarity), which lost recall (asymmetric: "what came
   after X"), 0.825 -> 0.602 at 600M. Stop patching lookup into echo; lean
   into abstraction, where an O(1) state has an edge.
2. **(Tested 13:34: NOT supported; seeds order sizes oppositely. The O(1)
   body solves held-out shifts fully in some seeds: circuit formation is the
   open lever.) Hypothesis: the bottleneck causes the abstraction.** Attention can keep
   every example, so memorising a table per parameter is cheap. A fixed-size
   state cannot hold the examples, so it is pushed to hold the rule. If
   true, a smaller state should infer held-out rules *better* (and recall
   worse), and that is a new, testable claim. Cheap test: a head-dim sweep
   on B2 (the micro-bench already takes `echo@32`, `@64`, `@128`), with tpp
   as the reference.
3. **The accident to chase:** echo's held-out numbers get *worse* with more
   shots (step_wide_new 0.98 / 1.00 / 0.72 at 2 / 4 / 8 shots); tpp does
   not (0.97 at 8). Larger numbers (s + 8d) or state saturation over a long
   numeric run? It is a capacity / precision clue for the same story.
4. **Mechanism direction (N4 rewritten):** a *relation memory*: write the
   relation between paired items (δ = φ(y) − φ(x)) instead of the items, then
   read it once and apply it to the query. N6 tried the rotation version and
   was judged on B1, a bench that could not show inference (extrapolation
   from 3 values). B2 is the fair test for this family.
5. **Bench lesson:** B0 broke even for attention (answer-only supervision,
   redrawn words). Micro-benches need dense LM-style supervision, or they
   measure the bench. And a micro-bench win is a lead (E2): seed pairs, then
   600M, both inside and beyond the trained context.

## Meditation (2026-10-30 ~14:00 IST, asked by the user)

Everything today points at one picture: a brain-like O(1) mind, three parts.
1. **Innate coordinates (grid cells).** Whether a seed learns the rule is a
   lottery (3 of 10). When models do learn arithmetic-like rules they find
   clock / circle codes (grokking, number helices). Grid cells give brains
   multi-period phase codes from the start, and path integration (adding a
   velocity to a phase) IS rule continuation. So give tokens multi-period
   phase coordinates from the start (the user's circles idea, N10) and let
   the rule memory integrate the per-example phase step. Test: does
   formation go from 3/10 to ~10/10 on B2?
2. **Hippocampal memory for lookup: separate, then complete.** Our lookup dies
   because look-alike keys collide (template "is"). The dentate gyrus fixes
   exactly this with *pattern separation*: expand the key into a sparse,
   high-dimensional code, so similar inputs become nearly orthogonal. Sparse
   keys also raise an outer-product memory's capacity by orders of magnitude
   (Willshaw nets) and let the delta erase touch only its own entry, so
   "last write wins" stops. CA3 then does *pattern completion*, which is the
   settling step. So N9 = DG (sparse separating keys, phase as one way) +
   CA3 (settling); N7 is the extreme case (exact = maximally sparse).
3. **A prefrontal planner forms the query** (N11, pands): reason what is
   needed, then look in the knowledge store and the conversation memory.
Body = cortex; coordinates = grid cells; memory = hippocampus; planner =
prefrontal. Each part has a micro test on B2 / the N7 bench; N7 and N9a,
running now, are the first two data points.

## Step back (2026-10-30 ~22:20 IST, after v13's 3B result)

v13 r3b (`ef136f5`): the reference PAM recipe at 3B tokens ends at holdout
3.309 vs T++ 3.206 / Mamba-2 3.164 (WikiText 58.0 vs 46.6); PAM ties early
(1.2B) and plateaus, and Mamba-2 reaches PAM's 3B loss at ~1.6B tokens (~2×
data efficiency). Recall still beats T++ (0.620 vs 0.539). **So bar 2 is not
a lookup problem: the body's LM quality plateaus.** v14's wins so far
(N9a lookup formation 5/5; echo infers held-out rules where T++ selects) are
real but sit on a body that falls behind at scale. Lesson for every v14 unit:
LM effects need ≥ 1.5B tokens before belief; 600M checks only measure the
targeted skill (lookup, formation, inference) with holdout as a guard.
Open question for the design: what in the body plateaus (state precision /
decay, write rule, mixing), and does a unit change the slope, not just a
skill? A slope test needs ≥ 1.5B tokens (the 6000).

## Meditation 2 (2026-10-30 ~22:45 IST, asked by the user)

Looking at the whole day from far away:
1. **We have been patching skills onto a body that falls behind at scale.**
   Every v14 unit today (tied echo, N1, N6, N7, N9a, N10) adds a skill to
   PAM, but r3b shows the body's LM curve flattens (3B: 3.309 vs T++ 3.206,
   Mamba-2 3.164). Skills cannot fix a slope.
2. **The prime suspect for the slope is state size and how many layers mix.**
   Mamba-2 carries ~13.4 MB of state per sequence and mixes in every layer;
   PAM ~1.2 MB, with notebooks in only 4 of 16 blocks (M1/M2 were chosen at
   600M for speed and early loss). PAM's own older 16-notebook 3B run (more
   state, more mixing) ended better (3.273). Known O(1) models (Mamba-2,
   Gated DeltaNet, RWKV-7) do match T++ at this scale, so O(1) is not the
   wall; our layout may be. This is v13's lane, sent as a EUREKA.
3. **v14's real lever for the slope may be cheap capacity, not more skills.**
   N7's exact n-gram table is a large, sparse state at O(1) cost per token,
   and cache LMs (Grave 2017, kNN-LM) gave real perplexity gains on web text
   because web text repeats names and phrases. So N7 should be tested on LM
   loss (≥ 1.5B tokens), not on a micro copy task. Same family: sparse
   high-dimensional keys (Willshaw capacity) for the notebooks.
4. **Micro-benches answered "why" well, but transfer poorly.** Keep them for
   diagnosis (the collision story, formation), and send anything aimed at bar
   2 to a slope test early.
5. **The novel parts that hold up are ours to keep:** echo's rule inference
   (beats same-size T++), N9a's phase keys (lookup forms on every seed), long
   recall past 2048. The plan is a competitive base body plus these, not
   these instead of a base body.
**Checked by v13 the same night (EXPERIMENTS_SEMPY "Slope check"):** late
slope 2 -> 3B holdout: PAM 4 notebooks LR 2e-4 −0.045; 16 notebooks LR 2e-4
−0.060; **16 notebooks LR 1.2e-3 −0.086**; T++ −0.081; Mamba-2 −0.079. The
plateau is the layout AND the LR; with 16 notebooks at a high LR the body
keeps pace (it starts worse: 3.537 at 1.2B, ends 3.284). v13 runs body × LR
at 1.5B next. **So v14's LM tests use that body (16 notebooks + echo, high
LR), not M1/M2.**
Next for v14: after n9c and N10 land, pre-register N7 as an LM test (600M
first as a signal, then ≥ 1.5B on the 6000), and design the sparse-key
notebook (Willshaw) as the capacity unit.

## Bar 3 transferred to 600M (v13 B2 pair, seed 42 half, `ea0c24b`; 2026-10-31 ~01:00 IST)

With 3 % B2 base-value docs in training, PAM M2 (102.9M) infers held-out rule
values; T++ (101.9M) on the same data mostly does not. B2 `_new`: PAM 1.000 /
0.973 / 1.000 (shift / step / affine) vs T++ 0.257 / 1.000 / 0.944 (means
0.991 vs 0.734; T++ picks the ±1 decoy on shift). Spread past the trained
length (2048): affine_new L = 1024 PAM 0.965 vs T++ 0.075, L = 4096 0.785 vs
0.45; shift_new PAM 0.79 / 0.60 / 0.34 at 1024 / 2048 / 4096, T++ 0. Cost:
PAM recall 0.870 -> 0.721, holdout +0.030 (T++ +0.077). **The 27M B2 finding
(echo infers, T++ selects) holds at 600M**, one seed; b2p2 (PAM seed 43) and
b2m (Mamba-2) are running. Fixed rule: no bar-3 claim unless PAM beats BOTH
T++ and Mamba-2 by +0.10 in both seeds. If it holds, the v13 session posts the
ntfy EUREKA (its run); v14 does not double-post. For v14 units judged on B2:
also run with `--spread` (the distance edge is where the body differs).

## Step back (2026-10-31 ~04:15 IST, after the N9s micro run)

**What surprised us.** (1) The fixed 3-token window was not the limit on
label_gap: N9a formed it 3/3, because the key is W_k h_{t-1} and the body
state already carries the item across a few filler words. I had pictured
N9a's key as "the window", and it is not: the window gives a separating
angle, the body gives the content. (2) N9s formed two_hop once, and nothing
else ever has, attention included. (3) The phase arms lose on words never
seen in training when there are 8 entries (0.56–0.83 vs 0.95–1.00).

**What it really says.** In two_hop the query is "Ana's pet is called": the
word that decides everything sits outside the window, and the answer needs a
chain (Ana -> dog -> Rex). A running angle that holds "Ana" through "'s pet
is called" is a topic register: who we are talking about right now. Language
has this (the discourse topic, the entity in focus), and the brain has it
(a held item in working memory steering what gets retrieved). label_gap
could not show it because the body solved it anyway; two_hop shows it
because the chain needs a clean first address. One seed: replicate first.

**Bold angles (not built; only after n9th says yes).**
- **Pointer-following angle:** let the running angle be rewritten by what
  the memory READ at this layer, not only by the input token. Then layer 1
  reads "dog" with Ana's angle, layer 2 addresses with dog's angle and reads
  "Rex": hops = depth, still O(1). N9s writes the angle from embeddings
  only, so it cannot do this; a clean two_hop circuit would need it at more
  hops (three_hop as the bench).
- **Fixed random angles for unseen words:** the 8-shot `_new` drop says
  learned separation does not reach new words. A fixed random angle per
  token id (dentate-gyrus style: separation that needs no learning) plus
  the learned part would separate any word. Cheap ablation if N9a goes to
  scale and `_new` matters.
- **A held value, not a held angle (for v13's open cell):** step_wide_new
  through filler needs the exact last term copied across distance. A
  salience-gated slot that holds the last important token's content and is
  readable is the same idea as N9s one step further. Needs a micro
  `--spread` eval first.

**Next:** n9th (two_hop replication) when a slot is agreed; n9r readout
~12:00 IST.

## Step back (2026-10-31 ~06:45 IST, after v13's Mamba-2 EUREKA)

**What we saw (v13, `adbf1f3`).** Mamba-2 nearly matches PAM on rules in
context, collapses at distance (0.08 vs PAM 0.67), and beats PAM on exact
items (label_diag 0.787 vs 0.57–0.63, a far number through filler).

**What it really says.** (Corrected by v13 at ~06:45: with DELTA = 1 the
notebooks have a soft write gate, selective by rank (AUROC 0.84-0.95) but
~10x too weak in size; see EXPERIMENTS_V14 "N12" Correction.) It comes down
to two update rules. PAM's notebooks can write without forgetting, but
refuse filler only softly; Mamba-2
cannot write without forgetting (one Δ). So PAM integrates and Mamba-2
selects. Neither does both, and the brain uses two systems for this
(neocortex integrates, hippocampus stores the surprising single episode).
v14 has pointed at that pair since 10-30 (N7 was a first try at the
episodic part); now we know what the episodic part must do: select, and
not decay.

**The bold angle: let the body's own surprise choose what gets written.**
That is what the slow learner could not predict, which is what the
hippocampus stores. It needs no learned gate, so it cannot stay soft (v15's
G1d lesson), and it is sparse for free on natural text. It may also be the
right salience for N9s: the words you could not predict are the ones that
name the topic. Written as N12 (design only). Open: the cost of the body
logits, and capacity under web filler at 8k (threshold or bigger store).

**Order.** n9th first (seed 131 on the 6000 ~11:30 IST; seeds 132–135 on
the 4090 after n9r, ~12:00–16:00). N12 code + selftest while they run
(CPU / small); its micro run goes after n9th.

## Step back (2026-10-31 ~08:20 IST, after v13's 3B LM gap attribution, `0771dca`)

**What we saw.** At 3B, the gap to T++ (+0.05 / +0.06 nats) is mostly
in-window copy (53–58 %). The gap to Mamba-2 (+0.09 / +0.11) is mostly NEW
tokens (52–55 %), and Mamba-2 is already 0.05–0.09 better on new tokens at
positions 0–8, with almost no context. Memory units (N12, N9a, recall
formats) can close ~0.03 of the T++ gap but only ~0.02–0.03 of the Mamba-2
gap.

**Before designing for it, two cautions.**
1. **The 3B PAM was trained at LR 2e-4, one stage.** v13's slope check
   (10-30) showed its plateau tracks the LR, and at 1.2e-3 PAM keeps pace
   with both baselines on the late slope. Part of the "general modelling"
   gap may be under-training. r3c (LR 1.2e-3) must be scored the same way,
   at equal tokens, before a body or unit is built for it.
   **Checked by v13 (~08:30): the confound does NOT hold for the old body.**
   The same body at LR 1.2e-3 (`9982a79`, 3B) is worse overall (+0.043 PG-19
   / +0.013 web), with the same split vs Mamba-2 (new tokens 46-49 %, copy
   24 %). r3c (16nb + echo + two stages, 1.2e-3) is still the open case;
   v13 scores its ~1.56B checkpoint ~15:00-16:00 IST. v13 will pre-register
   A4 `cond_mem` as a paired 600M filter (new tokens at positions 0-64 +
   holdout) after r3c's futility point.
2. **Mamba-2's short conv is not the missing piece:** v13's G1 (conv on
   qkv) failed and the box control upheld it (+0.07 train loss).

**What it says for v14.** Bar 2 against Mamba-2 is a per-token processing
question (positions 0–64, new tokens), not a memory question. v14's memory
units are the bar-3 / copy half. If r3c still trails on new tokens, the
candidate levers are:
- **Parametric n-gram knowledge:** v13's A4 `cond_mem` Engram table is in
  code and has never been run. It targets exactly short-context new-token
  prediction, with extra params but O(1) lookups (bar-2 tolerance). The
  over-encoding literature shows big small-model gains from n-gram input
  embeddings; fairness caveat: it helps any architecture.
- **Surprise-routed compute:** the N12 signal reused, so hard tokens get
  extra processing. Speculative; not designed.

**Free data:** score n9c vs C1-ctl (600M, on d_drive) with
`v13_sempty/tmp/attr3b.py` when the 4090 has room (queued in the background
for when v15's job leaves the 4090, ~10:40 IST; v13's frozen text copied
from the 6000 `~/attr/data.npz`). That shows whether
N9a's recall jump moves copy tokens (part of the T++ gap).

## Direction (user via v13, 2026-10-31): intelligence mind + learned fetch

**Fixed (v13 `e27d88d`):**
- **The label gap was data format.** In the lb pair, PAM label_diag is
  0.934 vs T++ 0.798. v14 units no longer target short-range binding.
- **Roles:**
  - the O(1) body IS the mind (rules, gist, learning past the trained
    length);
  - verbatim knowledge goes to a per-session library and a shared growing
    library;
  - v14 owns the learned fetch (gate, trained key at the salient token,
    re-read) plus the library row format;
  - v13 writes the benchmark spec (32k-1M, MB in the table) and runs the
    600M checks of the organs handed over.
- **Order:** the benchmark first, the distributable model only once it wins.

The O(1) state is trained for rules / gist / reasoning, not recall. Recall
comes from a LEARNED FETCH over the user context and tool results
(RAG-like). This fetch is v14's mechanism to design, in three parts:
- **when to fetch:** a gate;
- **the query:** the state at the salient token, where N12 writes; the
  organiser probe got top-1 0.63-0.88 to 128k zero-shot with it;
- **re-reading:** the fetched span, which gave two-hop 1.0.

N7 and N12 are its first organs. Communicator's check on n9c's lookup
reversal: with 3 lags, the key includes the token before the item, which
varies in natural text, so one fact gets two keys. Try 2 lags or a
salient-gated variant. Design after the n9th / n12m readouts. Addendum (organiser): Mamba-2 is also at the floor on
the needle probe, while the exact organ stays at 1.0. The frozen-state
address is 0.75 top-1 at 128k but 1/4 at 400k. Build a LEARNED key at the
salient token, with one row per event (not per token).

## Step back (2026-10-31 ~13:00 IST, after the pivot, n9r and n9th seed 131)

**What surprised us.** (1) n9r killed the last recall claim: the control's
own seed moves recall by 0.08 / 0.14, five to ten times N9a's effect, and
ctl43 is a 0.14-worse outlier on loss. Every "notebook selects better"
result so far is smaller than one seed's spread, except the micro-bench
formation results. (2) The one robust notebook effect is N9s on two_hop
(2 of 2 seeds, attention and N9a at chance): an anchor that stays alive
across the words between the entity and the question. Two hops need an
anchor, not more memory.

**What it says for the library.** The same structure appears there. The
query "Ana's pet is called" works only if the state keeps "Ana" as the
current topic; the fetch's query former is that anchor. So N9s's finding
transfers: the planner (N11) can be the salient running angle used as the
retrieval query rather than as a notebook phase. I do not build that yet.

**Bold angles.**
- The library is the one place where the exact organ is perfect, so the
  learned parts should be judged only where the organ is not: paraphrase,
  an instruction with no shared words, and index size at 1M. If a learned
  key cannot beat the organ there it should not exist.
- "One row per event" is a claim about the write gate; the gate is the part
  that can fail silently (missing facts), so it is tested first, for free
  (W0), before any training.

**Plan.** W0 now (eval only, small), then L1 on the organiser's harness.
`n12m` cancelled; n9th stops after seed 132.

## Step back (2026-10-31 ~13:40 IST, user: "read the meditation memory, take a deep breath")

**What surprised us.** W0's gate "failed" in exactly the way the brain's
does: surprise marks the boundary of an EVENT, not the content. We were
anchoring the row on the firing token; the hippocampus binds the episode
that just ended.

**What it really says.** (1) A row is an event span between bursts, keyed by
a learned pooling (L1-evt), written with hindsight at the next burst. The
pooling is a running softmax sum, so the open event costs O(1) state.
(2) Keys are the mind's own vectors, so they belong to a model version; the
SHARED library should therefore be text + provenance and each model
re-derives keys by one pass (text first). (3) Selection after a fetch can use
the mind as the verifier, one span at a time. (4) The gate that matters is
trained by hindsight utility (the event later re-mentioned), not by an
absolute threshold, and event boundaries are relative to recent surprise.
(5) The key's scaling past 128k is the make-or-break; a memory bank of
negatives and a pre-registered two-level index (events -> chunks) are the
answers kept ready.

**Bold angle held back.** Train the mind itself to be retrievable: a small
auxiliary loss on natural text (state at a re-mention close to the state at
the first mention) would give the key head billions of free training pairs
and make the shared key space a property of the mind. It touches the mind's
training, so it is v13's lane; offered, not started.

## Design input from Mind's bar-3 result (2026-10-31 ~22:00 IST, `df0a44a`)

Bar 3 is met in its distance form, replicated: D = PAM 0.645 / 0.691 / 0.741
(3 seeds) vs T++ 0.134 / 0.134 and Mamba-2 0.116 / 0.098, with examples
~250-500 tokens apart and total context past the trained length. One-read reach
stays ~2k. **For the fetch (N11 / package):** a skill pack re-fetched
whenever ~1.5-2k tokens have passed since its last exposure keeps the skill
alive at any length, so the planner needs a REFRESH rule (tokens since the
skill was last read), not only a "when to fetch a fact" rule. Cost: the pack's
tokens once per ~2k of session. Mind's `replay_refresh` probe cell is the
measurement. To be pre-registered with N11.

## Shared-library fetch (approved by Lead, 22:50 IST)

Session library = mind-state keys (L1 head, in-document). Shared library =
text: (1) exact organ with WHOLE-KEY matching (fixes the 4-gram collisions
that drop it to 0.87 at 4000 entries); (2) a normalised lexical match for
paraphrase (case-folding, inflection; BM25 class); (3) an off-the-shelf text
embedder as a named baseline row, so the cost table shows what the mind's own
keys buy there (nothing, by the sweep). Plumbing, cited, no claim. To build
after the package table's handover.

## Meditation (2026-10-31 ~23:36 IST, asked by the user via Lead: think in the void, no planning)

**What surprised me.** The same state that finds a fact perfectly inside its
own stream finds nothing across texts, and training could not change it. The
organ, the dumbest tool, beat every learned key across documents. Surprise
marked where something new began, not what it was. "Stored but reworded"
and "never stored" looked identical to the key. Training to hold longer made
the mind forget faster.

**What it really says.** The mind's state is not a description of a thing;
it is the reader's position in its own experience. Retrieval inside a stream
is RECOGNITION: the query state already carries the trace of having read the
row. Across texts there is no shared experience, so nothing to recognise.
The mind is a reader: text should get things into its stream (cheap, exact),
and the mind does what only it can, reading and recognising within what it
has read. Forgetting after ~2k is not a defect to train away; a mind that
forgets fast and re-reads its notes is "intelligence separate from knowledge"
taken literally. **(Correction 2026-11-01 00:20: the "training to hold
longer made it forget faster" premise came from lg32, which Mind found was
confounded by an orphaned-answer packing bug; whether reach can be trained is
open again until lg8r reports. The re-read picture stands as a product choice,
not as a law.)**

**A bold different angle (not tested).** Raw confidence failed as a selector
because on natural text it follows the mind's prior. The GAIN a fetched row
gives (log p of the mind's answer with the row minus without it) cancels that
prior. That could be both the selector the planner lacks and the "nothing
stored" signal similarity could not give: abstain when no candidate row moves
the mind. Known idea (PMI scoring / contextual calibration: Holtzman et al.
2021, Zhao et al. 2021); what would be ours is using the O(1) mind's own
re-read as the verifier. **Lead's first smoke (23:40, `7f49c7b`, c1ctlb, 2k,
natural, 2 docs per cell) does NOT support the simple form:** right rows gain
0.7-2.5, wrong rows 1.9-5.9, never-stored rows 0.5-8.0. A wrong row raises the
probability of ITS OWN answer too, so a row's gain for its own answer rewards
any confident span; cancelling the prior needs the TRUE answer's gain, which is
unknown at question time. Kept as a popped idea; to test OFFLINE on the package
table's P_cands (no new runs): gain against a null row (a random span of the
same length), and whether the right row is an outlier among the top-5 gains.
A second, quieter thought: the session library could
store the mind's recognitions rather than keys, i.e. the places where the mind
was surprised, so that re-reading them is what keeps the mind current.

## Next action

1. **n9r** (4090, running, ends ~12:00 IST): score recall vs the bars
   (EXPERIMENTS_V14 "N9a recall replication"), report label_diag t200 and
   PG-19 far gain. If met: N9a gets its own arm on the r3c recipe, paired
   with r3c at a matched token budget (v13 runs it via `ECHO_PHASE_LAGS=3`).
2. **n12m** (N12 micro screen, echo vs echo_surp × 3 seeds, bars in
   EXPERIMENTS_V14 "N12"): queued on the 4090 after n9th (tmux `n12m`,
   worktree `../qllm_v14n12m`, driver `~/n12m_4090.sh`). Pass -> hand to v13
   for 600M on the B2 recipe paired with b2p2 (`ECHO_SURPRISE_TAU=3`).
3. **n9th** (N9s two_hop replication, bar H): running (seed 131 on the
   6000, seeds 132-135 on the 4090 after n9r; see State). H met -> mechanism
   check, then a 600M plan with v13. H not met -> N9s FAIL, record + remove.
4. **v13 flagship r3c** (`16f372e`, 6000, ~14 h from ~00:45 IST): 16
   notebooks + echo + two stages + LR 1.2e-3 straight to 3B vs T++ 3.206 /
   Mamba-2 3.164. v14 units are NOT in it.
5. **N7 as a cache LM**: its own arm on the r3c recipe (`NGRAM_ORDERS=2,3`,
   hook `a55e1f0`), paired with r3c, ≥ 1.5B tokens; pre-register after r3c's
   first readout.
6. Closed (see EXPERIMENTS_V14): N9s bar G (no claim), N7 micro (bench could
   not show copy), N10 formation (no verdict), Q1 / S1 (F1 0/5), N1, N6 on
   B1. Rule (user): run only what can be superior; run known-best directly.
