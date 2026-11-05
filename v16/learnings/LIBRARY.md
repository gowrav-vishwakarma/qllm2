# Library learnings (v14 lane), self-contained

The library is the knowledge side of "an O(1) mind + a growing library": what gets stored, how a question finds it, and
how the mind reads it. Each entry gives the idea (maths), the recipe, the numbers, the verdict and the lesson, so a new
repo can rebuild it without the old code. Conventions:
- **mind** = a frozen PAM body, b2p2 unless stated (102.9M params, 600M tokens, M2 body, trained at 2,048, state 1.15 MB);
  cu1 / cu2a are Mind's curriculum checkpoints (same size, a different stage A).
- **bench docs** = held-out prose of length L with 8 inserted facts and a question at the end (v16/bench.py `build()`).
  Cells: kv (" Definition: K refers to V." / " Query: K refers to"), para (the asked key in another case / plural / with
  "the"), tool (" Result: job: K; status: v; owner: v; region: v; label: v." / " Query: job: K; f:"), neg (the asked key
  was never stored), syn (stored under one phrase, asked by a synonym phrase sharing no word), natq (fact in one of 4
  prose templates, asked "what does K stand for? Answer:").
- **keys**: synthetic = one 2-syllable pseudo-word; natural = a 2-word phrase from other prose.
- **free** = greedy decoding over the whole vocabulary; **sentence scorer** = decode to "." / newline (cap 16), correct
  iff the value appears as a whole word. **20-way** = the value ranks first among the 20 candidate values.
- Every bar below was pre-registered before its numbers; seeds were fresh for every confirmation.

## 1. Where the answer comes from: three separable steps

Fetch (find the row), read (hand it to the mind), say (the mind writes the value). Every result below sits in one step.
The single most useful habit: measure each step on its own (an oracle row for "read", the same row for every body).

## 2. Writing rows

- **W0, surprise as the write gate.** u_t = −log p_mind(x_t | x_<t). A token "fires" if u_t > τ; firings ≤ 2 apart merge
  into a burst. Bar: at ≤ 1 row / 16 tokens, ≥ 95 % of facts contain a firing. Frozen C1-ctl, 8k docs: the lowest τ
  meeting the rate was 11 nats; catch 0.95 (synthetic) / 0.928 (natural, NOT met); plain sentences 0.887 vs 0.574 chance.
  The firing token is mostly the FIRST word of an inserted sentence (a topic break), not the key. With the echo read
  included in p (the real output), τ = 10 met. **Lesson:** surprise marks where an event BEGINS; a row must span from
  the burst to the end of its sentence (≤ 64 tokens), not sit on the firing token.
- **Row format (proposal):** key 128-d bf16 (256 B) + span pointer (6 B) + session / ordinal / version (10 B) = 272 B,
  raw text kept as u16 ids (2 B / token). Superseded for the session library by text organs (§3), which need no key.

## 3. Fetch

- **Token organ:** the latest exact match of the question's last 4 tokens in the stream; row = the sentence around it.
  1.00 on exact keys at every length. Cost: a scan, or a suffix array (4 B / token).
- **Normalised-word organ (A4, plumbing; IR case folding + suffix stripping):** words = letter / digit runs, lower-cased,
  "a / an / the" dropped, one trailing "s" stripped if ≥ 4 letters. Over the WHOLE question: (1) the longest suffix word
  n-gram, n = 6..3; (2) else the longest any-position n-gram; (3) else a bigram holding a word seen ≤ 2 times in the
  stream; latest occurrence wins. Bench 32k: right row 1.00 on kv / para / tool, never-stored false hit 0.00 / 0.025.
  Bug to avoid: read the question from its START, not from the key's last token (A4 v1 was invalid for this).
  Edge: in a very short document a frame bigram ("refers to") can be "rare" and fire.
- **Rare-in-English word (A20, MET):** reworded questions ("I forgot what K means. It means") put a one-word key
  between frame words never stored next to it, so no 2-3-word span matches (1 / 22). Step 4: a question word seen 1-2
  times in the stream AND ≤ 3 times in a shipped background count (6M FineWeb tokens → 101,252 words, 1.47 MB): any
  wording finds the key (0.225-0.275 → 1.00), never-stored unchanged; frame words are common English and never
  qualify. Order: token organ → word steps 1-2 → step 4 → (F2) → step 3 (the bigram step attracts frame words:
  "stand for", "that refers to").
- **Strict mode:** abstain iff both organs miss (a rule). Never-stored abstain 0.975-1.00 at 32k.
  `text_fetch(ids, doc_end) → (row span | None, {organ_hit, step, abstain})` is this whole chain, body-agnostic (no
  mind keys): every body (T++, Mamba-2, PAM) gets the same library in Lead's comparison.
- **A2, abstain by similarity: infeasible.** Stored-but-paraphrased and never-stored queries have overlapping top-1
  similarities: to reach neg wrong-fetch ≤ 0.10 the threshold (0.669) falsely abstains 0.44. **Lesson:** abstain is a
  rule (organ miss) or a learned skill (cu2 VERIFY), never a similarity threshold.
- **L1, a learned key over the mind's states.** key = normalise(Σ_i softmax(w · h_i) · W h_i) over a row span (block-8
  states, centred by a shipped mean, 588 → 128), InfoNCE with in-batch rows, trained on bench-format docs over held-out
  web text. c1ctlb, 32k / 128k: synthetic 0.81 / 0.56 (Bar A ≥ 0.90 / 0.75 NOT met), natural 0.94 / 1.00; natural
  paraphrase 0.81 vs token organ 0.31. Pooling loses a pseudo-word's identity (per-token state 1.00). 256-d no better.
- **L3 / EUREKA "stream address":** across DOCUMENTS the learned key fails (exact 0.29 / 0.66 vs organ 1.00): the
  mind's state is an address within its own stream, not an embedder. Shared libraries are fetched by TEXT.
- **SL2, the shared library's text fetch (MET):** normalised-word organ, then the token organ, then BM25. 400 / 4,000
  entries: 1.00 / 0.99-1.00 on exact and surface variants. Stored as depth-limited suffix arrays over word and token
  streams: ≈ 4.0 + 5.4 MB per 1M tokens + 2 MB raw text ≈ 11.4 MB / 1M (the dicts were 24-67 MB). MiniLM-L6 sentence
  vectors: 0.32-0.75 at 49.6 MB / 1M. SL1 (organ + BM25 only) NOT met on paraphrase (0.49 at 4,000).
- **A13, a question is many keys, chosen by the mind's surprise (B1, B2 MET).** Take the 2 question tokens with the
  highest surprisal (excluding the first 2); score each sentence row by Σ_keys max_{tokens in row} cos(h_key, h_tok) in
  block-8 states (centred). b2p2 32k: synonym facts top-1 0.575 / top-5 0.80 (MiniLM sentences 0.35, the L1 head 0.40 in
  A5, the key's last token 0.125); natq top-1 0.775 / top-5 0.975 (BM25 0.525). Block 2 is worse (0.40 / 0.55).
  Seed spread on synonyms ≈ ±0.1 (0.575 / 0.632 / 0.433 over three seeds at n = 19-40). **Lesson:** the frame of a
  question is predictable to the mind, the key is not; text cannot tell them apart (A12: BM25 top-1 0.175 / 0.525,
  rarest-span 0.45 / 0.725, abstain 0.125), the mind's surprise can.
- **A21, the question-time package on natural wording (32k, seed 18000):** text steps then F2, warm + contrast read.
  Natural keys, right row (full text chain → with F2): kv 0.725 → 0.775, kv1 0.60 → 0.75, kvupd 0.55 → 0.70, natq 0.70
  → 0.825; answers kv 0.725 (lenient) vs 0.20 strict; synthetic keys text-solved (1.00 fetch, 0.825-0.925 answer).
  F2 never abstains: lenient answers 30 / 57 % of never-stored questions with a value word (strict 0 / 3 %).
- **Memory of that address (A14, A15):** every token's 588-d state = 1,176 MB / 1M (fp16). int8 per vector (+ fp16
  scale) is lossless (A15: 0.433 = 0.433 on syn). A PCA-64 basis (fit on held-out web text, 47.9 % of variance)
  destroys the synonym signal (0.632 → 0.263): meaning lives in low-variance directions. Keeping only the top 1/16
  surprising stream tokens costs synonyms ~0.1 (int8: 0.333 vs 0.433, 37 MB / 1M; sign bits: 0.433 top-1 but top-5
  0.77 vs 0.93, 4.9 MB / 1M); natq unaffected. **Open:** an address under 50 MB / 1M that keeps synonyms (next: a token
  fraction sweep at sign bits / int8 with ≥ 80 docs, or a random projection, which keeps all directions equally).
- **A3 / A11, relevance from the mind's read: no.** The read gain log p(v | row) − log p(v | no row) is 3.2 nats for a
  WRONG key's row vs 4.8 for the right one; the best gain rule added +0.06 (bar +0.10): the mind copies the nearest value. A key-swap check
  (log p(v | row, q) − log p(v | row, q with a never-stored key)) gives AUROC 0.698 on b2p2 (bar 0.85); cu1 (trained
  to say " unknown") 0.815 and refuses 97 % of wrong rows in its own words. **Lesson:** checking a key is a SKILL the
  mind learns from examples (cu2 VERIFY), not something a query-time check can read off an untrained mind.

## 4. Read: how the row is handed to the mind

- **Cold** = row + question from a fresh state (≈ 40 tokens). **Warm** = the session's own state after the document,
  then row + question. **Contrast** = l_warm + 4 (l_warm − l_norow), l_norow = the same state + question without the
  row, on the ANSWER tokens only (Shi et al. 2023 CAD; Mind's cd1-cd3).
- **A6 / A6c:** warm alone is far worse than cold at every length, even 2k (kv natural 0.125-0.275 vs cold 0.775-0.875): the long
  state outvotes the row it was just shown. Warm + contrast beats cold at every length (0.90-1.00).
- **A9 (MET, the package read):** strict organ rows at 32k, cold → warm + contrast: kv 0.95 / 0.775 → 0.925 / 0.925,
  para 0.85 / 0.675 → 0.925 / 0.925, tool 0.525 / 0.40 → 0.75 / 0.575 (syn / nat). Cost: no extra prefill (the session
  state exists), two short O(1) continuations, a second 1.15 MB state during the answer only.
- **Rules:** contrast never on free text (on prose a = 4 takes NLL 3.90 → 8.56); stop at the value: 20-55 % of
  contrast answers run on ("marigold | status: ..."); the contrast on the FIRST answer token only cuts the run-on.
- **Reading several rows at once hurts** (A13 line-up of the top 5: 0.275 vs 0.55 one row): re-rank before the read.
- **A17, rendering a row into the mind's trained template (NOT met):** cu2a trained on " Record: K | f: v." reads a
  rendered row cold as " unknown." 80 % of the time when the field name was not in its training set (only colour /
  city / code / owner / room / tool were); warm + contrast overrides it. Skills are format- and field-bound at 103M; the
  fix is an open field vocabulary in training (cu3), not a library field map.
- **A19, field-cut tool rows (MET, labelled plumbing):** the library cuts a multi-field row to the asked field's clause
  (" Result: job: K; region: v."). Cold: b2p2 0.475 / 0.40 → 0.90 / 0.80; T++ 0.90 / 0.90 → 0.825 / 0.85. Mamba-2 fails
  the uncut row like PAM (0.425 / 0.475): binding several key-field-value triples is hard for fixed-state models even
  in a 35-token context. The uncut row stays the headline ("reads a record").

## 5. Skills: refresh instead of memory

- **N11-R:** re-insert a skill pack (header + 8 demonstrations) every R tokens of filler. 20-way at 32k: R 2048 0.96 /
  0.80 (+1.6 % tokens), R 1024 1.00 / 0.86 (+3.3 %). Free (N11-R/F, A8, A16): R 2048 0.59-0.66 plain, **0.70-0.79 with
  the query-token contrast**; R 4096 0.11-0.16 plain, 0.58-0.65 with contrast (± dilation); R 8192 0.42 even with echo
  dilation + contrast. No refresh: 0.00 from 8k. Echo dilation (retention r → r^(1/4) over the gap) costs +0.020 nats /
  token on the gap text: not used. **Package rule:** refresh every 2k + query-token contrast.
- **Skills are stored as demonstrations, never words only** (Mind: a rule taught in words is learned as a lookup table;
  one demonstration generalises).
- **SR-0, injecting a rule vector (mean residual at the demo arrows) into the state: NOT met** (0.10 / 0.00 at L = 0).
- **A24, rules as words vs demonstrations (40 items per cell, 20 affine rules y = a·x + b inside the trained family;
  rule, then 2k of held-out web text, then "\nx ->"; free = first token right):**
  - Words ("each output is a times the input plus b") reach neither mind: b2p2 0.05 / 0.025, cu2a 0.00 (contrast /
    plain).
  - Demos 2k back: 0.40-0.60. The same 8 demos 256 tokens back (the refresh): b2p2 0.65 / 0.775, cu2a 0.85 / 0.875.
  - The skill contrast is mind-dependent. DEMO8: b2p2 0.60 plain → 0.425 contrast; cu2a 0.475 → 0.60.
  - **Rule:** a register skill = ≥ 8 demonstrations in the trained format, refreshed near the query, contrast measured
    per mind. A rule written in words must be turned into demonstrations by the library before the mind sees it.
- **A23, rule × knowledge through the package (`apply`):** the library fetches the key's record row, takes the value
  out (its own row format), and writes the skill's trained query line "<v> ->". The mind answers with the skill
  contrast, whose twin is the whole session without the pack (a second O(1) state).
  - Mind's R × K harness is reproduced exactly (400 / 400 docs): b2p2 0.61 at 2k, 0.45 at 8k (echo dilation s 8);
    cu2a 0.62 / 0.49; T++ 0.01 / 0.07.
  - **Rule:** the planner substitutes. Asking the mind "K ->" with the row in view (compose key → row → rule) gives
    ≤ 0.07 for every model.
  - One story rule ("twice its weight plus one", 4 → 9) failed even as 8 demos (" 14"): a single item at a 0.61 rate
    proves nothing; A24 was the set.

## 6. Comparison with T++ and Mamba-2 (same text library, Lead's arm; seed 18000, 32k)

- **A18, holding the session (32k, same rows):** T++ holding the WHOLE session (1.06 GB KV, 16× its trained length)
  answers 0.00 on every cell; T++ holding the last 2k (67 MB) + contrast reads kv 0.95 / 0.95, para 0.95 / 0.975, tool
  0.75 / 0.825, natq 0.425; the PAM package holding the whole session (1.15 MB) reads kv 0.975 / 0.925, para 0.90 / 1.00,
  tool 0.75 / 0.675, natq 0.475. A 2k window WITHOUT the contrast falls on natural keys (0.475-0.575) like the mind's
  warm read: the context's prior outvotes the row in a transformer too. Limit: these cells need only the fetched row,
  so a window loses nothing; R x K (a rule held in state applied to a fetched fact) is the cell that separates them.
- Cold read of the same row, free (syn / nat): kv T++ 1.00 / 0.875, b2p2 0.90 / 0.775; para T++ 0.70 / 0.65, b2p2 0.775
  / 0.80; tool T++ 0.90 / 0.90, b2p2 0.475 / 0.40, Mamba-2 0.425 / 0.475; hop2 1.00 all. Memory to read ≈ equal (~35
  tokens). The difference is in holding the session: T++ 1.2 GB KV at 32k vs the mind's 1.15 MB (A18 measures T++
  warm with the whole session vs a 2k window: see EXPERIMENTS_V14 "A18").
- **Honesty line:** exact / surface recall is the library's (text organs, plumbing); what the mind adds is reading
  with a fixed state (warm + contrast), skills carried by demonstrations and refresh, and the surprise-keyed address.

- **A25, long sessions by needle distance (one streamed session per doc, 8 facts, the asked ones at fixed distances
  from the question; 10 docs per cell; package / mind alone / T++ 2k window / T++ window + row):**
  - 128k: kv 0.90-1.00 at ~1k, ~16k and ~100k back, both key types. Mind alone 0.00; T++ window 0.00-0.20; T++
    window + the package's row 0.90-1.00.
  - Varied wording on natural keys: 0.60-0.80 (strict 0.10-0.30). Never stored: strict abstain 0.90-1.00.
  - Pooled over distances: kv 0.967 / 0.933 (syn / nat); the one far miss was a written slip on the right row. No
    trend with distance. Verdict: B2-B4 met, B1 not met by one document of 10.
  - Beyond 128k: out of scope by the user's cap (the stopped run's 300k rows stay in its log, unquoted).
  - Bytes per 128k session: mind state 1.15 MB, text 0.26 MB, indexes ~1.2 MB; F2 address 78 MB at int8, 0.64 MB as
    sign bits (the two forms tie at 128k on 24 F2 rows: 0.71 vs 0.62).
  - **Lesson:** distance does not matter to text organs; the mind's job is reading the fetched row on a long state,
    and warm + contrast still works on a 128k state, 64× its trained 2k.
- **Story and BABILong (A22, Lead's entry):**
  - A four-chapter story (11 exam questions × 2 forms, keyword hits of 22 at 2k / 4k / 8k): b2p2 package 8 / 8 / 7 vs
    alone 1 / 3 / 2. The package answers single-sentence facts; order of events, rule arithmetic, negation and
    summaries stay at 0.
  - BABILong at 0k-2k, the mind composing (package column): qa1 0.74-0.85, qa2 0.05-0.13, qa3 0.01-0.07, qa4 / qa5
    0.25-0.54.
  - **Lesson:** fetch is solved for one fact. Combining 2-3 rows (object → holder → place) is the mind's open gap. A
    planner chain can do it in text, but it must be labelled TEXT.

## 7. The package v0: one entry point (`v16/package.py`, frozen at 7052e4a)

- `Package(ckpt).answer(session, question, mode)`: stream the session once (2k chunks, state carried), then fetch.
  The fetch chain is token organ → word steps 1-2 → rare-in-English step → (lenient) F2. Then warm + contrast on the
  answer tokens, with the field cut for records.
- Also: `fetch_rows` / `answer_rows` (k rows: name / latest content word / F2 + hops by name), `answer_chain` (TEXT
  planner), `apply` / `apply_value` (R × K).
- **The contract a new mind must keep** (nothing in the library is trained for a body):
  1. streaming `model(x, states, step_offset) → (logits, states, aux)` with chunked prefill = one pass and
     deep-copyable states;
  2. `model.blocks` with `(z, state)` outputs, the address hooked after block n_blocks // 2;
  3. a per-position LM head (surprisal);
  4. Mind's `_dilate` hook, if one-read skills are wanted;
  5. tokenizer-derived stops.
- State bytes are measured at read time: b2p2 1.15 MB, the 16nb body 3.92 MB.
- Checked unchanged on the 16nb body (r3cb). The address moves with the mind, so A13 / A21 are re-measured on a new
  body.
- The full spec (fetch order, read, refresh, planner, every cost in MB and compute, honesty lines, the benchmark map)
  is `v14/PACKAGE.md`.

## 8. Process lessons

- Pre-register with the decisive cell FIRST in the run order (a guard-stopped run still decides).
- Put the control in the same pre-registration as any cause you plan to name (A6's "beyond the trained length" was
  wrong; 2k failed too).
- A micro-bench win is a hypothesis until it holds at the target scale (N9a / N9s: 27M win, 600M FAIL, removed).
- Text checks must call the harness's own entry point with the harness's own inputs (A4 v1).
- Never pipe a check into grep / tail inside a commit chain; read the exit code first.
- Seeds on n = 30-40 docs move synonym numbers by ±0.1: decide on ≥ 80 docs or several seeds.

## 9. Nov 2-5: the read, the package suite, and the standard benchmarks (Library, written 2026-11-05)

Body for all rows unless named: SCI-1 386M, step 137,331 (9.0B tokens), state 3.67 MB. Details and per-cell tables:
`v14/EXPERIMENTS_V14.md`; one line per unit in `v14/EXPERIMENT_LINEAGE.tsv`.

- **A26 (protocol applied to a fetched value).** At DAY-0 no 600M / 1.5B body applied a stated convention to a fetched
  value (≤ 0.05).
- **A27 (decontamination audit, 37c291d).** An independent 13-gram audit: the filtered SCI-1 stream has 0 hits in 15
  sources (positive control 300 / 300).
- **A28 (be66f2d; replicated e2a87db).** "Prose rule held = 0.00" was the READ, not the state.
  - The row-vs-no-row contrast cancels anything both streams hold, including the rule.
  - Against a no-protocol twin the rule shows: 0.65 at the first milestone, 0.83 ± 0.04 at 137,331 (2k), 0.22 at 8k.
  - A28b (f56d4e6): on a held-out wording the edge over T++ / Mamba-2 is small (0.12 vs 0.05 / 0.06), not met.
- **A29 (a69cafa).** Rewrite the prose rule into 3 compact demos next to the question: 0.93 / 0.91 (2k / 8k).
  A30 (composed contrast, 531ad61): not met.
- **A31 (e2a87db).** Forking the O(1) state at the instruction (with / without) equals the twin read exactly (100 / 100
  answers) at 3.674 MB per fork.
- **PKG-SUITE (c93e3b7).** Per milestone, model alone vs model + library, at 137,331:
  - R×K 0.01 → 0.67 / 0.42 (2k / 8k);
  - rewrite 0.01 → 0.92 / 0.85;
  - told twin 0.13 → 0.83 (2k);
  - recall at 32k, 20-way, 0.10-0.45 → 1.00.
  - Per-item pairs feed the review page.
- **PAIRS-MORE (18f33af).** The page's own probe items:
  - s8 0.63 → 1.000 (4k); skill pack 0.055 → 1.000 (8k), examples re-shown.
  - **Needle 0.998 → 0.927, NOT met.** The fetch was right on 2,520 / 2,520, but the contrast read cancelled what the
    state already held.
- **GATE (20a770c).** The warm read (evidence streamed through the mind's own delta write, read plainly) is 1.000 on
  all 5,720 needle / s8 / skill items; the contrast 0.967. The agree gate meets its bar, but at ceiling it is plumbing.
  - **WRITE-IN** (running) decides on the cells that discriminate: told (plain 0.22 vs twin 0.83 already), R×K, rewrite
    and recall-32k.
- **BENCH-TOOL (be77c55).** GSM8K with a sandboxed calculator: 0 / 200 plain and tool.
  - The model calls a tool on 97 % of items, in the trained format, with invented glaive names: format, not schema.
  - Fed to Mind's chat ramp: a schema-grounded tool source.
- **BENCH-LIB (7a4e123).** One library for all bodies: the ARC Corpus + cosmo / fineweb, 14.87M passages, two
  independent 13-gram filters, BM25 top-3.
  - SCI-1: ARC-e 0.463 → 0.526, ARC-c +0.00, HellaSwag +0.015, PIQA n.s.
  - The answer-passage ceiling is ARC-c 0.28 → 0.84: the mind reads, retrieval is the limit.
- **ARC-LIB (20a770c), NOT met.** On the three ~103M ctl bodies with that identical library, the ARC gain is PAM
  +0.058, T++ +0.055, Mamba-2 +0.054 (± 0.014).
  - Every body uses fetched passages equally; with the answer passage in view T++ copies best (0.96 vs PAM 0.87).
  - The paper does NOT claim a library-use edge. PAM's edge stays at carrying rules across filler and reach past the
    trained length.
- **What these days taught (step back 15:50, 11-05).** Every failure sat at the seam between mind and library: who wins
  when the prior and the evidence disagree.
  - Mind's knockouts found the delta erase-at-key carries the rule, and that a global fade is worse than none.
  - The contrast read is such a global fade of the prior. Evidence streamed through the mind's own write (warm) was the
    better read wherever tested so far.

Process lessons added:
- Read the clock before writing a time (one estimate slipped in 11-05 and was fixed).
- Never rebase a worktree while a run is rewriting a tracked file there (the push waits for the run).
- A read that subtracts the mind's own prediction must be tested on cells where the mind is already right.
