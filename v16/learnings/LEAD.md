# Lead learnings (v16: benchmark, package table, comparison arms, decisions), self-contained

Written 2026-11-01 so a new repo can re-implement every cell and re-derive every decision without the old code.
Conventions: "20-way" = the answer token ranks first among 20 candidate value nouns (chance 0.05); "free" = greedy
argmax over the whole vocabulary equals the answer's first token; "sentence" = greedy decode after the read until
the first "." / newline (cap 16 tokens), and the value (whole word, case-insensitive) occurs in the text: the
headline score from MLB-1 A7 on; "syn / nat" = synthetic nonsense keys / natural two-word phrases taken from the
filler text. Bodies: PAM b2p2 (M2 body, 102.9M, 600M tokens, 2k windows), PAM c1ctlb (M1, 103M), T++ tfc1b
(101.9M), Mamba-2 mbs2b (102.6M), all on the same data mix. Memory figures are per live session unless stated.

## 1. The benchmark (MLB-1) and its harness

- **Idea.** One frozen, pre-registered suite that scores the MIND alone, the PACKAGE (mind + outside library), and the
  baselines AS THEY ARE on the same documents, with memory and compute per answer next to every accuracy. The thesis it
  tests: intelligence (rules, gist, in O(1) state) separate from knowledge (rows outside the layers), holding beyond
  the training length.
- **Document builder.** Held-out prose filler (a 500k-token holdout cache, GPT-2 vocabulary) of length L; n facts
  inserted at random positions; the question appended at the end. Cells: `kv` (8 facts, "Definition: K refers to V.",
  ask one), `kv1` (one fact), `kvupd` (the asked key is re-defined later: latest wins), `hop2` (K1 maps to K2, K2 maps
  to V, far apart, two distractor chains), `para` (asked by case / plural / "the" variants), `tool` (a tool-result row
  "Result: job: K; f1: v1; f2: v2; ..." and one field asked), `neg` (facts stored, the asked key never stored: abstain
  is right), `instr` (a rule stated once in words), `natq` / `natqneg` (the fact stated in prose, asked as a plain
  question), `syn` / `synneg` (stored as "grand palace", asked as "magnificent castle": 82 hand-made pairs). Seeds
  are stable across processes (crc32 of the task name, not Python's hash): the same seed gives the same documents to
  every body, which is what makes rows comparable.
- **Read paths recorded per document.** A = state only (the body reads the whole stream, answers from its last
  position); B = a model-free exact organ (last 4 tokens -> what followed, latest write wins; the copy organ);
  C2 = the mind's own per-token vector at the question's key word matched against every stored position (a "stream
  address"); C5 = rows written only at surprise bursts (the mind's own surprisal over a threshold); P = the planner:
  re-read the top rows one at a time in front of the question, take the largest top1-top2 margin (P_margin), or all
  together (P_concat); each candidate row keeps sim / margin / gain / correct / row_has_answer for offline selectors.
- **Scorers (A7 / A9).** 20-way; free; sentence; the decoded text itself (one shown generation per row); for `hop2`
  the organ chain (K1 -> K2 -> V) and the body re-reading both spans. `--qstyle varied` samples the question wording
  per document from a pool of 8 (index 0 = the trained template; direct, "I forgot what K means", chat, "Remind me",
  Q/A, "stand for", "earlier we said"); `--syn_keys` asks by a synonym when the key has a pair. The fetch path that
  answered (token organ / word organ / mind keys) is written per row, so a template-bound win is visible.
- **Body-agnostic text-library path (`--text_package`).** Fetch by text only (token organ, then a normalised-word
  organ: case-folded, plural-stripped, articles dropped, word n-grams 3..6, latest match; strict abstain = both
  miss), then a COLD read of row + question through any body, scored 20-way / free / sentence, with the read-token
  count. `--skip_state` drops the full-document forward so this runs on CPU for T++ and PAM (~0.1-2 s per 32k doc).
- **Memory column.** Mind state 1.15 MB (fp32 at decode); library rows 272 B per event; raw text 2 B per token;
  organ ~11.4 MB per 1M tokens (suffix arrays); T++ KV cache for this width / depth = 2 x layers x 2 x dim x L x
  bytes (1.23 GB at 32k, 4.93 GB at 128k in fp32; 1.06 GB measured at 32k); Mamba-2 state 13.9 MB.
- **Rules (v1 + amendments).** R1 package recall >= 0.90 on needle / 8-needle / latest-wins / two-hop at every L <=
  128k, negatives abstain >= 0.90; R2 mind rule-at-distance >= baselines + 0.10 in every seed; R3 mind holdout within
  +0.02 of Mamba-2 at equal data; R4 package memory at 128k <= 1/100 of the T++ KV; R5 attribution (honesty line);
  R6 stated rule; R7 skill-pack persistence; A7 free generation primary + 3 shown generations per headline
  checkpoint; A9 the NATURAL variant (synonym key + varied wording) is every cell's headline, the template secondary.
- **Lesson.** Score what the user would ask, not the template; record the path that answered; keep the seed stable
  across processes; put MB and tokens-per-answer next to every accuracy; a 20-way pick over first tokens is lenient on
  multi-token values (14 of the 20 value nouns are multi-token), so the sentence scorer is the honest one.

## 2. The package table (MLB-1 Win at the package level; EUREKA 2026-11-01 01:10 IST)

- **Setup.** 32k (seed 18000, 40 docs per cell) and 128k (seed 18500, 24 docs), syn + nat, both PAM bodies, with
  Library's session fetch (exact organ + the mind's burst key at block 8, shipped key mean, abstain disabled) and one
  cold re-read; T++ and Mamba-2 as they are on the same documents.
- **Numbers (20-way, the registered v1 score).** Package: needle / 8-needle / latest-wins / two-hop 1.00 at 32k AND
  128k, both bodies; paraphrase 0.46-0.66; tool 0.29-0.50; negatives answered wrong 0.00-0.12 with abstain off.
  State only at 32k: T++ 0.04 / 0.04 / 0.06 / 0.04, Mamba-2 0.14 / 0.06 / 0.18 / 0.09, the PAM minds 0.08-0.19
  (chance 0.05). Memory at 128k: 1.15 MB state + 0.26 MB rows + 0.26 MB text + ~1.5 MB organ ~ 3.2 MB vs T++ KV
  4,932 MB (ratio ~1/1,500; R4 bar 1/100). R3 guard: the b2 pair at equal data and recipe, PAM 3.4775-3.4800 vs
  Mamba-2 3.4754-3.4768 (c1ctlb's older recipe is +0.035, so b2p2 is the headline body).
- **Free generation on the same cells (added after registration).** 0.975-1.00 on synthetic keys, 0.71-0.85 on
  natural keys: every miss holds the right row and the model writes " a ...", " the ..." or repeats " refers";
  the mind alone writes the value 0.00-0.04 of the time at 32k / 128k.
- **Honesty lines.** The baselines ran without a library (comparison rule); a T++ given the same fetch was not run in
  this table (it was run later: section 4); the exact-key 1.00 is the model-free organ's; the mind's own key shows on
  paraphrase (0.46-0.66 vs organ alone 0.26-0.35); the mind's own claims are R2 (rules at distance, s8) and R4
  (O(1) state). Closest prior work: kNN-LM / RETRO (retrieval into a transformer with its KV cache), EM-LLM, Titans;
  difference: the reader is an O(1) recurrent mind and the library sits outside the layers, keyed by the mind's own
  stream state (session) or by text (shared); the package's memory is flat in context length.
- **Verdict.** Win at the package level on the registered score; the beyond-context clause is demonstrated by
  re-fetch at 128k with a mind trained at 2k. Lesson: the recall win belongs to the library half; the mind's part must
  be stated as rules in state + constant cost + its own keys where words differ.

## 3. Strict mode, abstain and expression (S2, A4 v2, A9 of Library)

- **Abstain by rule.** "Answer only on an exact organ hit, else abstain" gives abstain 1.00 on negatives at 32k and 128k
  with exact-key recall unchanged; a similarity threshold found post hoc (0.97) does the same on the table's docs but
  is not a rule. Query-time relevance by key-swap (does the answer change when the question's key is swapped) is NOT
  enough on the base mind (AUROC 0.70, bar 0.85): a wrong row's answer barely depends on the key. A curriculum-trained
  verify skill (cu1 / cu2a) does it (wrong value 0.01-0.03, "unknown" 0.86-0.99), and it transfers to natural keys.
- **Expression.** With the right row in hand the base mind's cold read writes the value 0.70-0.875 (sentence); the
  misses are sub-word fragments on multi-token values, loops ("refers to refers to ...") and generic phrases. Reading
  the row WARM (the mind's session state carried) with the mind's own no-row stream subtracted on the answer tokens
  only (the twin = same weights, same state, continued without the row; <= 16 tokens, a state copy) lifts natural-key
  exact facts 0.775 -> 0.925, surface variants 0.675 -> 0.925, tool fields 0.40-0.525 -> 0.575-0.75; abstain unchanged.
  The contrast is never applied to free text (it wrecks it: prose NLL 3.82 -> 8.56 after an irrelevant pack).
- **Lesson.** The storage is better than the voice at 103M: the state holds the fact and the language prior outvotes it
  at the readout; subtract the prior at the answer, never retrain for a readout problem at this size.

## 4. The T++ + library arm (the honesty line as a measured row; 2026-11-01 10:06 IST)

- **Setup.** The SAME text-only library (section 1's organs, strict abstain) for T++, Mamba-2 and b2p2; cold read of
  row + question; same docs and seeds as the package table (32k seed 18000, 40 docs x syn / nat; 128k seed 18500).
- **Fetch (identical across bodies).** Organ hit 1.00 on kv / kv1 / kvupd / para / tool / hop2; negatives abstain
  0.925-1.00; syn 0.00 and natq ~0.1 written for every body: text matching cannot do "words differ" for anyone.
- **Expression (sentence, syn / nat at 32k).** T++: kv 1.00 / 0.875, kv1 0.95 / 0.725, kvupd 1.00 / 0.90, para 0.70 /
  0.65, tool 0.90 / 0.90, hop2 1.00. b2p2: 0.90 / 0.775, 0.90 / 0.825, 0.975 / 0.825, 0.775 / 0.80, tool 0.475 / 0.40,
  hop2 1.00. Mamba-2: kv-type 1.00 / 1.00, para 0.975 / 0.95, tool 0.425 / 0.475, hop2 1.00. 128k kv: T++ 1.00 / 0.75,
  b2p2 0.958 / 0.667. Memory in the cold regime: ~30-60 read tokens for every body (T++ KV ~1.3 MB ~ the mind's
  1.15 MB): equal.
- **Verdict.** A transformer reading a fetched row cold writes the answer as well as or better than the fixed-state
  bodies, and much better on multi-field tool rows; both fixed-state bodies (PAM, Mamba-2) share the tool weakness
  (binding (key, field, value) triples: the MQAR / Zoology weakness). A package-side cut (the library hands only the
  asked field's clause) lifts the PAM mind to 0.90 / 0.80 and levels the bodies, labelled "library picks the clause"
  (the uncut record stays the headline).
- **Lesson.** The O(1) mind's claim is NOT "reads a fetched row better". It is (1) constant cost when the session must
  be HELD (T++ holding 32k = 1.2 GB KV, a 2k sliding window = ~75 MB, the mind 1.15 MB; measured in Library's warm
  arms), (2) rules carried in state across the session (s8: 0.65-0.74 vs <= 0.13 at a 2-4k gap; clk2r: one read to
  8k on three minds), (3) its own-state keys where words differ (surprise-token keys: synonyms 0.575 top-1 vs a
  MiniLM encoder 0.35 and text 0.075; natural questions 0.775 vs BM25 0.525).

## 5. Realistic tests (A9): the first numbers

- With the question reworded (pool of 8), the word organ still finds the row (6 / 6 in the smoke) and the 20-way pick
  survives, but the cold free read collapses for both bodies (b2p2 0 / 6, T++ 2 / 6): the body continues the
  sentence instead of saying the value. Full natural-variant runs for every body are the headline from here;
  expectation written before the numbers: natural << template for every body, the package ~0.5 where the template
  shows ~0.9. Lesson: a score on one template measures the template.

## 6. Decisions and why (Lead)

- **Comparison rule.** Baselines as they are, no new baseline training; a "+ library" arm only as a measured row with
  its own honesty line (now done: section 4).
- **Novelty rule.** Every pre-registration names the closest prior work and one line of difference; else it is
  plumbing with no claim.
- **GPU order and safety.** VRAM: free >= peak + 3 GB before a launch; a neighbour training job keeps >= 55 % of its
  alone speed; launchers gate on log markers, never on tmux names (prefix matches bit us); detached launchers with a
  full PATH; `pkill -f` self-matches the caller's shell (kill by PID); read the clock before writing a timestamp.
- **Query-time beats training-time at 103M** (three sessions' step-back, 05:00 IST): every training-time addition
  cost language (cu1 +0.045, lg8r +0.08, cu1n9s +0.23) or did not transfer (N9); every query-time addition (re-fetch,
  refresh, word organ, warm + contrast, strict abstain, dilation) was free of LM cost. So the mind trains for language
  and example-based skills only; the rest lives in the package. One counterexample: verify (abstain by the mind's own
  words) must be trained (A11).
- **Twin = the answering mind.** A frozen older mind as the contrast twin would be expert-vs-amateur contrastive
  decoding, a different method; withdrawn.
- **Address choice** is decided empirically at the 3B run's end by a six-arm readout on the same docs (own state, own
  + head, frozen b2p2, text organ, MiniLM, surprise keys).
- **Body changes** (a bounded sliding-window attention to fix multi-field reads) are the USER's decision, presented
  with a 100M screen and the package-side cut side by side; the package repair comes first because it is free.
- **Code safety.** All worktrees and the remote box's checkouts audited at each stall check; local-only chains pushed as
  `save/` branches; package code moves to v16 once, at release; these learnings files are the record that survives.

## 7. Nov 2-5: SCI-1, the equal-data control, the paper (added 2026-11-05 16:38 IST)

Details live in the lanes' notebooks (v13_sempty/EXPERIMENTS_SEMPY.md for Mind, v14/EXPERIMENTS_V14.md for Library); the
day-by-day decisions are in v16/SCRATCHPAD.md "Decisions"; every paper number with its source in v16/paper/data/numbers.json.

- **SCI-1 (386M, SEQ 4096, the first product-size mind).** Segments base 59c3cf5 -> cont1 59d098d -> cont2 e61cd92 -> cont3
  449e320 -> cont4 8284510 (worked text). Milestone every 1B tokens. s8 D (rule from 8 examples across filler, chance
  0.125): M4 0.782, M5 0.435, M6 0.436, M7 0.588, M8 0.589, M9 (137,331, 9.0B) 0.770. Lesson: the rule skill is MAINTAINED by
  its documents and moves with a ~1B-token lag; judge a mix change only after 2 milestones. Told and induced rules
  co-vary with rehearsal (the "compete for one state" hypothesis was tested over five intervals and dropped at M9).
- **Equal-data control (ctl caa4da9, result f0a4695; EUREKA 11-05 05:30).** PAM / T++ / Mamba-2, ~103M each, 1.0B tokens
  of the IDENTICAL SCI-1 cont2 mix. s8 D 0.523 / 0.074 / 0.149; told rule (twin read, 98 docs) 0.53 / 0.12 / 0.20 (held-out
  wording only +0.06: one seed, caveat kept); needles past SEQ at 8k / 16k 0.53 / 0.41 vs <= 0.18; inside the window T++
  recalls better (0.92 vs 0.63); holdout Mamba-2 3.378 < PAM 3.427 < T++ 3.440 (B3 not met). Lesson: rule-carrying and
  beyond-length recall are architectural, not data; Mamba-2 induces adjacent rules (0.68) and LOSES them over 2k (0.18),
  so the difference is retention under interference, not induction, and not state size (Mamba-2 holds 12x more).
- **Knockouts (Mind, eval time on the trained ctl PAM, filler spans only).** all-on 0.523; no erase-at-key 0.197; an
  equal-energy global fade (Mamba-like) 0.136, worse than no erase at all. The delta rule's LOCAL erase carries the rule.
- **Package on SCI-1 M9 (Library).** A31 state forks met (fork read = twin read, 100/100 identical, 3.674 MB per fork);
  PKG-SUITE row 1 (R x K 0.01 -> 0.67, rewrite 0.92, recall 32k -> 1.00, ~4 MB per session); per item: s8 at 4k 0.63 -> 1.00 and
  skill at 8k 0.055 -> 1.00 (examples re-shown, not fetched); needle LOST 0.998 -> 0.927 (fetch right on all 2,520; the
  contrast read cancels a fact the state already holds); GSM8K calculator 0/200 both arms (invented tool names: format).
- **Knowledge library on benchmarks.** BENCH-LIB on M9 (14.87M decontaminated passages, BM25 top-3): ARC-e +0.063, ARC-c
  +0.001, HellaSwag +0.015, PIQA +0.012 n.s.; with the answer passage given ARC-c 0.28 -> 0.84 (the mind reads; search is
  the limit). ARC-LIB on the three ctl bodies NOT met: gain PAM +0.058 vs T++ +0.055 vs Mamba-2 +0.054 (+- 0.014); the
  answer-passage ceiling favours attention (T++ 0.96, PAM 0.87). Lesson: never claim "uses fetched knowledge better";
  every body reads a fetched passage; PAM's edge is the state (rules held, reach), not the reading.
- **GATE (Library 747e24f).** Evidence streamed through the mind's own delta write and read plainly ("warm") = 1.000 on
  all 5,720 needle / s8 / skill items vs the contrast read 0.967. Joint step-back (15:52): the contrast read IS a global
  fade of the prior, so the needle loss is the knockout law at read time. Open: told (plain 0.22 vs twin 0.83), recall-32k,
  R x K, rewrite decide the package's default read.
- **Complex twin (cx_m2, c17775a).** Every part of the real body ported to complex heads in v16/complex/ (102.95M); 10.9k
  tok/s vs the real body's 63k (~5.8x slower, ~3x memory per sequence); training on the 4090, paused only for short eval
  pieces. Lesson: we parked "complex vs real at equal data" for a week as expensive; the port took a morning. Ask "how many
  hours, really?" before parking a question.
- **Harness built by Lead.** The review page (v16/humaneval/: every prompt and answer per checkpoint, mobile, server
  sends only the tapped item, model alone vs with library on 12 cells); the paper (v16/paper/, 16 pp, every number from
  numbers.json, TODO.md the one pre-arXiv list); PLAN_LONGRUN.md (open-ended run, rules R1-R7, chat ramp); the
  architecture pseudocode (v16/SCI1_ARCHITECTURE_PSEUDOCODE.md).
- **Decisions.** Pause a long 4090 run only per GPU piece and batch pieces (each restart costs ~25 min of stream startup);
  the oracle column is a ceiling row, never a headline; a claim needs the identical library / tool on all three matched
  bodies; the ablation ladder + second seed are the paper's first 6000 use after SCI-1's decay.
