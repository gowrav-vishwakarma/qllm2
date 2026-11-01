# The mind + library package (Library's spec, 2026-11-01 03:38 IST)

What a frozen O(1) mind (b2p2, 102.9M params, 600M tokens) plus the library does today, its rules, its cost,
and the measured numbers behind each line (EXPERIMENTS_V14 has the pre-registrations and verdicts).

## 1. Fetch (session library): text decides what is stored

1. **Token organ:** the latest exact match of the question's last 4 tokens in the stream; row = its sentence.
2. **Normalised-word organ** (`PACKAGE_NORM=1`, A4, plumbing): words lower-cased, "a / an / the" dropped, one
   plural "s" stripped; the question's longest suffix word n-gram (6..3), else any-position n-gram (6..3), else a
   bigram holding a word seen ≤ 2 times; read over the WHOLE question (238c62e).
2b. **Rare-in-English word (A20, `text_fetch(step4=True)`):** a question word seen 1-2 times in the stream and ≤ 3 times
   in shipped background counts (1.47 MB) is the key in any wording: reworded one-word keys 0.25 → 1.00, never-stored
   unchanged. Tried before the bigram step, which is where question-frame words cause false hits.
2c. **Tool rows (A19, labelled "field-cut rows"):** the library cuts a multi-field row to the asked field's clause;
   the package read then gives 1.00 / 1.00 (uncut 0.75 / 0.675; T++ cold uncut 0.90).
3. **Strict mode** (`PACKAGE_STRICT=1`): abstain iff both organs miss (a rule, not a similarity threshold:
   A2 showed similarity cannot separate "stored under other words" from "never stored").
4. Where words differ (lenient): the mind's 2 most-surprising question tokens, matched by max cosine against each
   sentence's tokens in its frozen block-8 states (A13: synonyms top-1 0.575, natural questions 0.775; a trained
   sentence encoder 0.35 / 0.55 on the same docs). The L1 head is superseded here.

Code: `v14/package_fetch.py` (`key_fn` for v16/bench.py), `v14/norm_organ.py`.

## 2. Read: warm + contrast (default since A9)

- The session's mind state after the document (it exists anyway) continues on **row + question** (stream A)
  and on **question only** (stream B); each answer token is the argmax of `l_A + 4 (l_A − l_B)`.
- **Rule (Mind's cd3):** the contrast is ON only for the answer tokens of a package-paired read, NEVER on free
  text (on ordinary prose a = 4 takes NLL 3.90 → 8.56 and collapses to loops).
- **Stop at the value:** first ".", ";" or newline (16 tokens cap); 25-55 % of contrast answers run on otherwise.
- Fallback / reference: the cold read (row + question from a fresh state), today's bench re-read.
- Reference implementation: `v14/a6_reread.py` (`Mind.feed`, `Mind.greedy_contrast`; chunked carried-state
  prefill, parity with the one-pass logits 2e-3).

## 3. Skills: refresh

- **The refresh interval depends on the body.** Mind's w1r (ac0559a): a bounded window (W1, 128 tokens, three
  blocks) lifts one-read skill reach at 4k 0.416 → 0.771 (seed 43: 0.435 → 0.617) at holdout parity, but does not help
  reading fetched rows (+0.03; the A19 field cut stays the reader). If W1 enters the body, re-measure A16 (R = 2k / 4k /
  8k) on the new mind before changing the refresh rule.
- Standing skills are stored as **demonstrations** (header + demo lines), never words only (Mind's 3c7a7fa: a
  rule taught in words is learned as a lookup table).
- Re-insert the skill pack every R = 2,048 tokens (N11-R: +1.6 % tokens) and read the question with the contrast
  on the query token: free 0.70 pooled at 32k (A16; plain 0.63). Every 4k + contrast ≈ every 2k plain; every 8k fails
  even with echo dilation (0.42), and dilation costs +0.020 nats on the gap: not used.
  R = 1,024 (+3.3 %) passes both rules plain.
- Mind's cd3: an irrelevant pack costs +0.08 nats on the following prose (+0.92 over the next 64 tokens).

## 3b. Rule x knowledge (R x K, Mind's 99d2951): the planner substitutes, the mind applies

- A rule held ONLY in the mind's state is applied to a FETCHED fact past the trained length when the planner puts
  the fetched VALUE into the rule's trained query line ("<v> ->"): free 0.61 / 0.62 at 2k (refresh + contrast), 0.45 /
  0.49 at 8k (one read + echo dilation s = 8 + contrast); T++ as it is 0.01 / 0.07 (0.30 with the pack re-inserted).
- **Planner rule:** never ask the mind to compose key -> row -> rule ("K ->" with the row in view: <= 0.07 for every
  model). The library fetches the key's row, takes the value out of it (it knows its own row formats; field_cut for
  records) and writes the rule's query line itself.
- Each standing skill in the register keeps its trained query format beside its demonstrations.
- Place a fetched record row AFTER some context, or render it in the rule's format: at the start of a document a
  record row can capture a held rule (cu2a 1.00 -> 0.02, b2p2 1.00 -> 0.91).
- Cost: the skill contrast needs the no-pack twin stream across the session (a second O(1) state, +1.15 MB, 2x read
  compute), unlike the fact read, whose twin runs on the answer tokens only.

## 3c. Known benchmarks: who does what (user's rule, 2026-11-01: known benchmarks, lengths capped at 128k)

TEXT = the library's organs / planner (plumbing, any body could use them); MIND = the O(1) mind's own work (state, address,
reading, writing). An entry must say which column produced each number.

| benchmark task | what it needs | fetch (who) | read / write (who) | our evidence |
|---|---|---|---|---|
| BABILong QA1 (single supporting fact: "Where is Mary?") | the latest fact about an entity | TEXT: name step (latest mention) | MIND: copy the location from the row | toy: right; A21 / A25 for scale |
| BABILong QA2 / QA3 (2-3 facts: object → holder → place) | a chain of facts | TEXT: latest-content-word step + name hops (fetch_rows) | MIND: combine rows (fails today: "The milk there is there.") | the open reasoning gap |
| BABILong QA4 / QA5 (two- / three-argument relations) | a relation lookup | TEXT: the two names / objects | MIND: read the relation | not measured |
| RULER S-NIAH (word / number / UUID needle) | exact key → value | TEXT: token organ (1.00) | MIND: write the value; multi-token values (numbers, UUIDs) are its weak spot (S2) | kv 0.925 at 32k (A9) |
| RULER MK-NIAH (look-alike distractor keys) | the exact key among similar ones | TEXT: exact organ (no key checking needed) | MIND: write | A11: the mind does NOT check keys; text does |
| RULER MV-NIAH (one key, several values) | all rows of one key | TEXT: top-k rows of the key | MIND: list several values (multi-row reading is weak: A13 line-up 0.275) | open |
| RULER MQ-NIAH (several keys in one question) | several fetches | TEXT: planner splits the question, one fetch per key | MIND: write each | not measured |
| RULER VT (variable tracking, multi-hop) | follow X1 = X2 = ... | TEXT: chained fetch (hop2: 1.00 in the package table) | MIND: write the names | the chain is the library's, not the mind's |
| RULER CWE / FWE (aggregation: frequent words) | counting over the whole context | TEXT: a counter tool (not built); the state cannot count | – | not covered: say so |
| RULER QA (SQuAD / HotpotQA in distractors) | natural questions, 1-2 paragraphs | TEXT + MIND: word organ / step 4, else the surprise-keyed address (F2) | MIND: read and write (natq 0.475-0.525 at 32k) | A13 / A21 |

**How comparisons are reported (user, 2026-11-01 16:30 IST):** by TOKENS trained on, not parameter count; benchmark runs
are diagnostics. Our rows (103M minds, 600M tokens, no benchmark data; as-is and package) form one block; deltas are
taken only against our own T++ / Mamba-2 trained on the SAME tokens. Leaderboard entries (fine-tuned 100-150M models,
untuned LLMs) go in a separate REFERENCE block with their training tokens and "tuned on the benchmark" stated.

Honesty line for any entry: retrieval-only tasks (S-NIAH, MK-NIAH, VT) are mostly the library's text organs; the mind's
share is writing the value and, for natural wording, its own address (F2). Multi-fact reasoning (QA2 / QA3, MV) and
aggregation are where the mind or a tool must still earn the number.

## 4. Shared library (text-first packs)

- `qllm-pack/v1` JSONL (header, then entries `{id, text, source, sha256, time, trust}`); `v14/pack.py`.
- Fetch (SL2): normalised-word organ, then the token organ, then BM25; depth-limited suffix arrays,
  **~11.4 MB per 1M tokens** (1.00 / 0.99 at 4,000 entries on exact + surface variants).

## 5. Cost (per session, b2p2)

| item | cost |
|---|---|
| mind state | 1.15 MB, O(1) in context length |
| raw text kept for rows | 2 bytes per token (2 MB per 1M tokens) |
| background word counts (A20) | 1.47 MB, fixed (not per session) |
| fetch | organ scans at question time (0.02-0.05 s per 32k doc in Python); indexed form = word suffix array ~5.4 MB per 1M tokens |
| read (default) | no extra prefill (the session state exists); two short O(1) continuations of row + question (~30-60 tokens) + ≤ 16 answer tokens; a second 1.15 MB state during the answer only |
| refresh | +1.6 % tokens at R = 2,048 |
| vs a Transformer++ KV cache | 4.9 GB at 128k (Lead's package table) |

## 6. Measured (32k, b2p2, free generation, sentence scorer; synthetic / natural keys)

| cell | number | source |
|---|---|---|
| exact facts | 0.925 / 0.925 | A9 (`logs/v14_a9_b2p2_32768_1b29b86_20261101_0327.jsonl`) |
| case / plural / "the" variants | 0.925 / 0.925 | A9 |
| tool-result fields | 0.75 / 0.575 | A9 |
| never stored: abstain | 1.00 / 0.925 | A9 (strict, rule-based) |
| exact-key recall, 20-way, vs T++ / Mamba-2 state | 1.00 vs 0.04-0.18 | Lead's package table |

## 7. Known limits (open)

- **Synonyms / natural questions** (words differ): fetch by surprise-chosen keys (A13), read warm + contrast:
  synonyms 0.55, natural questions 0.475 (ceiling 0.625), lenient. Abstain there is NOT solved by fetch scores
  (AUROC 0.66-0.77) or a key-swap check (A11: 0.70 on b2p2); it needs the trained VERIFY skill (cu1: wrong value
  on 3 % of wrong rows; cu2).
- **Writing on multi-field rows** (tool 0.575 natural) and **plain-question comprehension** (natq oracle
  20-way 0.45-0.55): the mind's side (Mind's cu2).
