# The mind + library package (Library's spec, 2026-11-01 03:38 IST)

What a frozen O(1) mind (b2p2, 102.9M params, 600M tokens) plus the library does today, its rules, its cost,
and the measured numbers behind each line (EXPERIMENTS_V14 has the pre-registrations and verdicts).

## 1. Fetch (session library): text decides what is stored

1. **Token organ:** the latest exact match of the question's last 4 tokens in the stream; row = its sentence.
2. **Normalised-word organ** (`PACKAGE_NORM=1`, A4, plumbing): words lower-cased, "a / an / the" dropped, one
   plural "s" stripped; the question's longest suffix word n-gram (6..3), else any-position n-gram (6..3), else a
   bigram holding a word seen ≤ 2 times; read over the WHOLE question (238c62e).
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

- Standing skills are stored as **demonstrations** (header + demo lines), never words only (Mind's 3c7a7fa: a
  rule taught in words is learned as a lookup table).
- Re-insert the skill pack every R = 2,048 tokens (N11-R: +1.6 % tokens) and read the question with the contrast
  on the query token: free 0.70 pooled at 32k (A16; plain 0.63). Every 4k + contrast ≈ every 2k plain; every 8k fails
  even with echo dilation (0.42), and dilation costs +0.020 nats on the gap: not used.
  R = 1,024 (+3.3 %) passes both rules plain.
- Mind's cd3: an irrelevant pack costs +0.08 nats on the following prose (+0.92 over the next 64 tokens).

## 4. Shared library (text-first packs)

- `qllm-pack/v1` JSONL (header, then entries `{id, text, source, sha256, time, trust}`); `v14/pack.py`.
- Fetch (SL2): normalised-word organ, then the token organ, then BM25; depth-limited suffix arrays,
  **~11.4 MB per 1M tokens** (1.00 / 0.99 at 4,000 entries on exact + surface variants).

## 5. Cost (per session, b2p2)

| item | cost |
|---|---|
| mind state | 1.15 MB, O(1) in context length |
| raw text kept for rows | 2 bytes per token (2 MB per 1M tokens) |
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
