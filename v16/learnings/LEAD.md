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
