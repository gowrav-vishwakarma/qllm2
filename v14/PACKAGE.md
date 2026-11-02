# The mind + library package, v0 (Library's spec; FROZEN 2026-11-01, code 7052e4a)

v0 is what a frozen O(1) mind plus the library does today, through ONE entry point (`v16/package.py`). This file gives
its rules, its cost in MB and compute, and the measured number behind each line. Pre-registrations and verdicts are in
`v14/EXPERIMENTS_V14.md`; the learnings for a new repo are in `v16/learnings/LIBRARY.md`. Changes after the freeze go in
a v1 section, so the v0 numbers keep their code hash.

Minds measured: **b2p2** (102.9M params, 600M tokens, M2 body, trained at 2,048) and **cu2a** (Mind's curriculum
checkpoint, same size and tokens). The contract (section 1) was also checked on the 16nb body (r3cb, the 1.5B tier's
body family): it runs unchanged.

## 0. How comparisons are reported (user, 2026-11-01)

By TOKENS trained on, not parameter count; benchmark runs are diagnostics. Our rows (the mind as is, and the package)
form one block. Deltas are taken only against our own T++ / Mamba-2 trained on the SAME tokens and data. Leaderboard
entries (fine-tuned 100-150M models, untuned LLMs) go in a separate REFERENCE block, with their training tokens and
"tuned on the benchmark" stated; never write "beats X" against that block. Every table gives params AND tokens.

## 1. The interface a mind must keep (the v0 contract)

The package needs nothing trained for it: no head and no index weights. It uses five things from the mind, which a new
body (the 1.5B tier) must keep:

1. **Streaming:** `model(input_ids, states=, step_offset=) → (logits, new_states, aux)`. A chunked prefill must equal
   one pass (parity). States are deep-copyable (tensors in lists / dicts). The contrast twin and every session read
   depend on this.
2. **Hookable blocks:** `model.blocks` is an indexable list whose outputs are `(z [batch, time, model_dim], state)`.
   The address (F2) is the residual after block `n_blocks // 2` (b2p2: 8 of 16). That is a rule, settable as
   `Package(address_layer=)`.
3. **A trained LM head on every position:** per-token surprisal picks the question's keys and the address's kept
   tokens.
4. **The `_dilate` hook** (Mind's echo dilation), only if one-read skills at 8k are wanted (`apply(dilate=)`).
5. **Tokenizer:** the sentence end, newline and `<|im_end|>` stops are read from the tokenizer. The background word
   counts are word-level, so any BPE works.

State bytes are MEASURED from the state at read time, never a per-body constant: b2p2 1,154,928 B, 16nb (r3cb)
3,921,168 B, W1 w1r_w128 2,947,152 B. That last figure is b2p2's 1.15 MB plus the window cache, 127 tokens × 3 blocks
× (k, v) × 588 × fp32 = 1.79 MB. The cache is O(W), not O(L).

**SCI-1 candidate sizes (random init, CPU, `v14/contract_check.py`, 2026-11-02 at 3886837):** the contract holds on
both bodies with no package change. A random-init echo reads out ~0, so `_dilate` is judged on the state.

| body (dim 980, 24 layers, 10 heads, mlp 3552) | params | address | state (measured) | address int8 / sign bits | parity: chunked / decode / deep copy |
|---|---|---|---|---|---|
| M2 (notebooks 5 / 11 / 17 / 23) | 348.3M | block 12 | **1.92 MB** | 984 / 7.9 B per token | 6e-7 / 1e-6 / 1e-6 |
| 16nb (notebooks in every block) | 397.2M | block 12 | **9.61 MB** | 984 / 7.9 B per token | 7e-7 / 1e-6 / 1e-6 |
| reference: b2p2 (trained) | 102.9M | block 8 | 1.15 MB | 592 / 4.9 | 6.1e-5 / 1.0e-5 / 1.4e-5 |

**Mind's SCI-1 candidates (8 heads × dim / 8, notebooks every 4th block for M2, mlp ≈ 3.62 × dim; random init,
2,048-token parity; 2026-11-02 ~11:00 IST at d5da73d):** every candidate passes, parity ≤ 1.2e-6.

| body | params | address | state | address int8 / sign bits (B per token) |
|---|---|---|---|---|
| M2 896 × 24 | 301.5M | block 12 | 2.81 MB | 900 / 7.2 |
| M2 1024 × 24 | 386.4M | block 12 | 3.67 MB | 1,028 / 8.2 |
| M2 1152 × 20 | 410.7M | block 10 | 3.99 MB | 1,156 / 9.2 |
| 16nb 1024 × 24 | 431.1M | block 12 | **13.11 MB** | 1,028 / 8.2 |

Rerun it on Mind's final geometry with `--like <ref ckpt> --set dim=… n_layers=… n_heads=… mlp_hidden=… pam_layers=…`.

State layout (Mind's 7d2c7f1): `[block states …] + [window caches: ONE extra entry, W1 models only] + [echo, LAST]`.
The package never indexes or counts the states (it streams, deep-copies and measures them), so `n_blocks + 1` and
`n_blocks + 2` entries both work unchanged. Checked 2026-11-01 17:14 IST on CPU at a7af87f with a 4,551-token toy:
- w1r_w128 (18 entries): exact ' amber.', field ' cedar.', F2 picks the right row, apply_value ' 9' (2·4 + 1).
- Chunked (1,024) vs one-pass read: last log-probs 2.7e-3 max abs, address states 1.2e-4 relative. b2p2 on the same
  toy: 3.8e-3 / 9.9e-5, so the same level. The address moves with the mind: re-measure A13 / A21 on a new body before quoting its fetch numbers.

## 2. API (`v16/package.py`)

```python
from v16.package import Package
pkg = Package('<mind best_model.pt>', device='cuda')            # alpha 4.0, max_new 40, address_layer n_blocks // 2
text, info = pkg.answer(session, question, mode='lenient')     # one fact; mode='strict' = text only, abstain on a miss
rows = pkg.fetch_rows(session, question, k=3)                   # several rows (name / latest / F2 + hops by name)
text, info = pkg.answer_rows(session, question, k=3)            # the mind reads the k rows together
text, info = pkg.answer_chain(session, q, prompt, k=3)          # PLANNER chain (TEXT): object -> holder -> sub-question
text, info = pkg.apply(session, key_question, skill)           # rule x knowledge: fetch the value, write "<v> ->", skill contrast
text, info = pkg.apply_value(session, skill, v)                 # the same for a given value
```

- `session` is text or token ids. It is read once and cached: `read()` streams it in 2,048-token chunks, carrying
  the state, so memory is O(1) in length.
- `info` holds the path (`text0 / text1 / text2 / text4 / F2 / None`), the row span, the row text, whether it
  abstained, whether the field was cut, and `bytes_held` (mind state measured, library text, index estimate, address
  int8 / sign-bit forms).
- The question text is appended verbatim. The caller chooses its form (exam form "\n\nQuestion: …\nAnswer:" or chat).

## 3. Fetch order

**One fact (`answer`):** the first step that hits gives the row (the sentence around the match, ≤ 64 tokens).

| step | rule | who | evidence |
|---|---|---|---|
| text0 token organ | the latest exact match of the question's last 4 tokens | TEXT | 1.00 on exact keys at every length |
| text1 word suffix | words lower-cased, "a / an / the" dropped, one plural "s" stripped; the longest suffix word n-gram of the WHOLE question, n = 6..3 | TEXT | A4: 1.00 on kv / para / tool |
| text2 word any-position | the longest any-position word n-gram, 6..3 | TEXT | A4 |
| text4 rare-in-English | a question word seen 1-2 times in the stream AND ≤ 3 times in the shipped background counts | TEXT | A20: reworded made-up keys 0.25 → 1.00 |
| F2 surprise address (lenient only) | the mind's 2 most surprising question tokens (after the first 2); each sentence row scored by Σ max cosine over its tokens' address states (block n//2, centred by the session mean) | MIND | A13: synonyms 0.575 vs a sentence encoder 0.35; natq 0.775 vs BM25 0.525 |
| strict | abstain if text0-text4 all miss | rule | never stored: abstain 0.925-1.00 at 32k (A9, A21), 0.90-1.00 at 128k (A25) |

- The bigram step (word step 3) is out of the package chain. Question-frame words ("stand for", "refers to") make it
  hit wrongly, and F2 behind it always returns a row.
- Abstain is a rule (text miss) or a trained skill (cu2 VERIFY). It is never a similarity threshold (A2), and never a
  query-time key check on an untrained mind (A11: AUROC 0.70).

**Several rows (`fetch_rows`, k = 3), for multi-fact questions:** the first row comes from the first of these that hits:
1. the text chain (text0-text4);
2. NAME: the latest sentence holding a mid-sentence proper noun of the question;
3. LATEST: the latest sentence holding the question's rarest content word (content = ≤ 3,000 background counts);
4. F2 (lenient only).

Then come hops by name, up to k rows: each capitalised name in a fetched row that is not in the question brings the
latest sentence mentioning it ("John took the milk." → "John went to the kitchen."). This is plumbing. The rows are
returned in session order.

## 4. Read: warm + contrast

- The session's own state continues on **row + question** (stream A) and on **question only** (stream B). Each answer
  token is the argmax of `l_A + 4 (l_A − l_B)` (Shi et al. 2023 CAD on the answer tokens; Mind's cd1-cd3). A9 MET:
  32k kv 0.925 / 0.925, para 0.925 / 0.925, tool 0.75 / 0.575 (syn / nat).
- **Contrast only on the answer tokens of a package read, never on free text.** On prose, a = 4 takes NLL 3.90 → 8.56.
- Decoding stops at the sentence end, a newline or the end of turn (`max_new` cap).
- **Field cut (A19, labelled "field-cut", plumbing):** a record row (" Result: job: K; f1: v1; …") is cut to the asked
  field's clause. Then b2p2 reads 0.90 / 0.80 cold and 1.00 / 1.00 warm (uncut 0.475 / 0.40).
- Several rows read together hurt (A13 line-up 0.275 vs 0.55 for one row). `answer_rows` is for multi-fact questions
  only.

## 5. Skills: refresh

- A standing skill is stored as **≥ 8 demonstrations in its trained format**, with its trained query line ("<v> ->").
  Words do not reach a 103M mind: A24 gives words 0.00-0.05; demos 2k back 0.40-0.60; demos 256 tokens back (the
  refresh) 0.65-0.875.
- **Re-insert the pack every R = 2,048 tokens** (+1.6 % tokens). Read the query with the contrast on the query token:
  free 0.70 at 32k (A16; plain 0.63). R = 4k with contrast ≈ R = 2k plain. R = 8k fails even with echo dilation
  (0.42), and dilation costs +0.020 nats on the gap text, so it is not used.
- The skill contrast is not a uniform gain. A24 DEMO8: b2p2 0.60 → 0.425, cu2a 0.475 → 0.60. Measure it on vs off per
  mind and skill before turning it on.
- Body note (Mind's w1r): a bounded window W1 (128) lifts one-read skill reach at 4k (0.416 → 0.771). If W1 enters the
  body, re-measure A16 before changing R.
- Cd3 cost: an irrelevant pack adds +0.08 nats on the following prose (+0.92 over the next 64 tokens).

## 6. Planner: substitution and chain

- **Substitution (R × K, Mind's 99d2951; `apply`):** the library fetches the key's row and takes the value out (it
  knows its own row formats; `field_cut` / `row_value` for records). The planner then writes the skill's trained query
  line "<v> ->", and the mind answers with the skill contrast (the twin = the session without the pack).
  - A23: the package reproduces Mind's harness exactly (400 / 400 docs). b2p2 scores 0.61 at 2k and 0.45 at 8k
    (dilation s 8); cu2a 0.62 / 0.49; T++ as is 0.01 / 0.07.
  - Never ask the mind to compose key → row → rule ("K ->" with the row in view: ≤ 0.07 for every model).
  - Place a record row after some context: at a document's start it can capture a held rule (cu2a 1.00 → 0.02).
- **Chain (`answer_chain`, labelled TEXT):** for "where is the object", if the first row was found by the object word
  (LATEST) and names a holder not in the question, the planner asks the sub-question with the holder in the object's
  place ("Where is John?"). The mind answers it from the holder's latest row; otherwise `answer_rows` runs. This is the library composing, not the mind. Its BABILong
  column (`pkg_chain`) sits beside the mind-composes column (`pkg`), which stays the headline. Measured so far only as
  a smoke (1 / 3).

## 7. Cost (per session)

| item | bytes | compute |
|---|---|---|
| mind state | measured: b2p2 1.15 MB, 16nb 3.92 MB; at ~350M (SCI-1 candidates) M2 1.92 MB, 16nb 9.61 MB; O(1) in length | one streaming pass over the session, carried in 2k chunks: 11.4 s at 128k (4090, shared with other jobs, with address capture and surprisal) |
| raw text for rows | 2 B / token (0.26 MB at 128k) | – |
| text indexes (word + token suffix arrays, estimate) | ~9.4 B / token (1.23 MB at 128k) | organ scans at question time: 0.02-0.05 s per 32k doc in Python |
| background word counts (A20) | 1.47 MB JSON, fixed, shared by all sessions | – |
| F2 address, every token int8 (lossless, A15) | D + 4 B / token: 592 for b2p2 (78 MB at 128k), 984 at dim 980 (129 MB at 128k) | one max-sim over the rows per question |
| F2 address, sign bits on the top 1/16 surprising tokens | (ceil(D / 8) + 4) / 16 B / token: 4.9 for b2p2 (0.64 MB at 128k), 7.9 at dim 980 (1.04 MB) | the same; costs synonyms ~0.1 at 32k (A15), at 128k equal (0.71 vs 0.62 on 24 F2 rows, A25) |
| fact read | a second copy of the state during the answer only | two O(1) continuations of row + question (~30-60 tokens) + ≤ 40 answer tokens: 0.38-0.42 s per question incl. fetch (4090) |
| skill contrast (`apply`) | a second full state for the no-pack twin | 2× the read compute |
| refresh | +1.6 % tokens at R = 2,048 | – |
| reference: T++ KV cache (tfc1b, 12 layers, d 768) | 36,864 B / token: 4.8 GB at 128k; the 2k window 67 MB measured (A18) | – |

## 8. Measured (b2p2 unless stated; free generation, sentence scorer: decode to the stop, the value as a whole word)

**32k, bench docs, synthetic / natural keys:**

| cell | package | strict | source |
|---|---|---|---|
| exact facts (template question) | 0.925 / 0.925 | same | A9 |
| case / plural / "the" variants | 0.925 / 0.925 | same | A9 |
| tool records, field-cut | 1.00 / 1.00 | same | A19 |
| varied wording (v14/styles.py), kv | 0.875 / **0.725** | 0.875 / 0.20 | A21 |
| varied wording, kv1 / kvupd | 0.925 / 0.675 · 0.825 / 0.70 | 0.925 / 0.275 · 0.825 / 0.25 | A21 |
| natural questions (natq) | 0.525 | 0.20 | A21 |
| never stored: answers with a value word | 0.30 / 0.57 (F2 never abstains) | **0.00 / 0.03** | A21 |
| T++ holding the whole 32k session (1.06 GB) | 0.00 on every cell | – | A18 |
| T++ 2k window + the same row + contrast (67 MB) | ≈ the package (kv 0.95 / 0.95, natq 0.425) | – | A18 |

**128k by needle distance (A25, 10 docs per cell; package / mind alone / T++ 2k window / T++ window + row):**

| 128k | ~1k back | ~16k | ~100k |
|---|---|---|---|
| kv synthetic | 1.00 / 0.00 / 0.20 / 1.00 | 1.00 / 0.00 / 0.00 / 1.00 | 0.90 / 0.00 / 0.00 / 0.90 |
| kv natural | 0.90 / 0.00 / 0.00 / 1.00 | 0.90 / 0.00 / 0.00 / 1.00 | 1.00 / 0.00 / 0.00 / 0.90 |
| varied wording, natural keys (strict) | 0.70 (0.10) | 0.60 (0.20) | 0.80 (0.30) |
| para · tool (both key types) | 0.80-1.00 · 1.00 | | |
| never stored, strict abstain | 1.00 / 0.90 (syn / nat) | | |

Pooled over the three distances (package / strict): kv 0.967 / 0.933 (syn / nat), varied wording 1.00 / 0.70 (strict
1.00 / 0.20), para 0.933 / 0.90, tool 1.00 / 1.00. Mind alone 0.00 everywhere; T++ 2k window 0.00-0.07; T++ window +
the package's row 0.73-1.00. No trend with distance: the one kv miss at ~100k was a written slip on the right row
(" (yx." for "onyx"). Lenient mode fetched a row for all 60 never-stored questions (57 by F2). A25 verdict: B2-B4
met, B1 not met by one document (EXPERIMENTS_V14). Lengths beyond 128k are out of scope by the user's 128k cap: the
300k rows the stopped run wrote stay in its log, unquoted.

**Other tests:**
- **Story (A22):** a four-chapter story, 11 exam questions × 2 forms, keyword hits of 22 at 2k / 4k / 8k.
  - b2p2 package 8 / 8 / 7 vs alone 1 / 3 / 2; cu2a 6 / 5 / 5 vs 1 / 1 / 1; T++ / Mamba-2 alone ~0.
  - Single-sentence facts are answered. Order of events, rule arithmetic, negation and summaries stay at 0.
- **R × K (A23):** see section 6.
- **BABILong (Lead's entry, `v16/babilong_eval.py`, package column = the mind composes, 100 docs per cell, at 0k / 1k / 2k):**
  - b2p2: qa1 0.85 / 0.74 / 0.77; qa2 0.05 / 0.11 / 0.13; qa3 0.01 / 0.07 / 0.06; qa4 0.45 / 0.37 / 0.25; qa5 0.53 / 0.46 / 0.54.
  - cu2a: qa1 0.99 / 0.69 / 0.69; qa2 0.00 / 0.09 / 0.09; qa3 0.00 / 0.00 / 0.08.
  - The mind-as-is column and longer lengths are on Lead's board.

## 9. Known benchmarks: who does what

TEXT = the library's organs / planner (plumbing; any body could use them). MIND = the O(1) mind's own work (state,
address, reading, writing). An entry must say which column produced each number.

| benchmark task | fetch (who) | read / write (who) | our evidence |
|---|---|---|---|
| BABILong QA1 (single supporting fact) | TEXT: name step (latest mention) | MIND: copy the location | b2p2 0.74-0.85 at 0-2k |
| BABILong QA2 / QA3 (object → holder → place) | TEXT: latest-content-word step + name hops | MIND: combine rows (fails) / TEXT chain (`pkg_chain`) | 0.01-0.13: the open reasoning gap |
| BABILong QA4 / QA5 (relations) | TEXT: the two names / objects | MIND: read the relation | 0.25-0.54 |
| RULER S-NIAH (word / number / UUID) | TEXT: token organ | MIND: write the value; multi-token values are its weak spot | kv 0.925 at 32k (A9), 0.90-1.00 at 128k (A25) |
| RULER MK-NIAH (look-alike keys) | TEXT: exact organ | MIND: write | A11: the mind does not check keys; text does |
| RULER MV-NIAH (one key, several values) | TEXT: top-k rows | MIND: list several values (multi-row reading is weak) | open |
| RULER MQ-NIAH (several keys) | TEXT: planner splits the question | MIND: write each | not measured |
| RULER VT (variable tracking) | TEXT: chained fetch | MIND: write the names | the chain is the library's |
| RULER CWE / FWE (frequent words) | a counter tool (not built); a state cannot count | – | not covered: say so |
| RULER QA (SQuAD / HotpotQA in distractors) | TEXT + MIND: word steps, else F2 | MIND: read and write | natq 0.525 at 32k (A21) |

## 10. Honesty lines (every write-up that quotes the package)

1. Recall of stored facts at any distance (32k, 128k) is the library's text organs: plumbing that any body
   could use. Alone, the mind and a 2k-window T++ answer ~0 beyond a few k (A25).
2. A T++ holding a 2k window, given the same row with the contrast, matches the package on single-row cells (A18,
   A25). The difference is cost: the session in 1.15 MB vs a 67 MB window, or 4.8 GB at 128k for the full KV. T++
   holding the whole session at 32k answers 0.00.
3. What the mind adds:
   - reading a fetched row on its own long state (warm + contrast);
   - the surprise-keyed address F2, its own (natural wording +0.05 to +0.15 over text);
   - skills carried by demonstrations plus refresh;
   - rule × knowledge through the planner (0.61 vs T++ 0.01).
4. Lenient mode answers never-stored questions with a value word 30-57 % of the time. Strict is safe (0-3 %) but
   misses reworded common-word keys (0.20 vs 0.725). Abstain where words differ needs the trained VERIFY skill
   (cu2).
5. Multi-fact reasoning (BABILong QA2 / QA3, MV-NIAH, the story's order of events) is NOT done by the mind (≤ 0.13).
   A TEXT planner chain is labelled as such.
6. Field cut and the planner's substitution are library plumbing and are labelled. The uncut record read stays the
   headline for "reads a record".
7. All numbers are 103M minds trained on 600M tokens, no benchmark data. The address and the read move with the mind:
   re-measure on a new body.

## 11. Known limits (open; for the mind's next version, not new library units)

- Composition over several rows (QA2 / QA3): the mind's side (the 1.5B tier, cu3).
- Abstain where words differ: trained VERIFY (cu2), not a library rule.
- An F2 address under 50 MB / 1M that keeps synonyms: sign bits on 1/16 of tokens hold at 128k (A25) but lose ~0.1 on
  synonyms at 32k (A15). PCA destroys it.
- Writing multi-token values (numbers, UUIDs) and multi-field records without the cut.
- Shared libraries (packs, `v14/pack.py`, SL2 text fetch 11.4 MB / 1M) cover surface variants only. Synonyms across
  documents are untested (L3: the mind's address is a stream address, not an embedder).

## v0.1 (after the freeze)

- **Decimal-aware sentence ends (d554d80; A26's prerequisite):** a "." ends a row, and a decode, only when the next
  token is not glued to it (it does not start with a letter or digit). GPT-2 splits "12.3" into " 12", ".", "3", so
  v0 fetched "Sample K7: mass 12.347." as "… mass 12." and cut answers at "12.". Every science number was truncated
  for every body.
  - Decoding holds a "." that follows a digit for one token: a glued token continues the number; any other token
    means the "." ended the sentence, and the look-ahead token is dropped.
  - Regression: the A20 text check on its registered documents is identical (640 questions). The contract toy is
    unchanged.
- **Address bytes from the address width** (3886837): `Package.address_bytes(n, form)`, no b2p2 constant.

## v1 design, NOT built: tool results as session rows + a sandboxed python executor (Lead / user 2026-11-02 11:10 IST)

Goal: the SCI-1 claim becomes "a protocol held in state, applied to a TOOL result". The mind writes a call; the
library runs it and writes the result into the session as a row. The text organs, F2 and the warm + contrast read
then treat that row like any other.

**Tool result as a session row: format FIXED (Mind a9838ca, `v7/sci_tools.py`; ONE format for training data and the
package; the package adapts to the data, never the reverse).**
- Training sources: glaive-function-calling-v2 + ToolACE (Apache-2.0), converted to this format, plus Mind's
  `library.fetch` docs (~11 % of tool docs). In those, the mind calls
  `{"name": "library.fetch", "arguments": {"query": "K field"}}` and gets
  `{"name": "library.fetch", "content": "Record: K | field: v."}`, then answers with v. The scientific-python data is
  code only, with no outputs.
- **Package tools:** `library.fetch` is the package's own fetch (text organs → F2), so the MIND can decide when to
  fetch, beside the planner's always-fetch. Its `content` is the fetched session row VERBATIM. Training shows rendered
  records, so raw sentences are a format shift: re-measure at milestones. `python` is the executor below.
- The format (Hermes / Qwen style inside ChatML; plain text in GPT-2 BPE, no new tokens):
  `<|im_start|>assistant\n<tool_call>\n{"name": "python", "arguments": {"code": "…"}}\n</tool_call><|im_end|>` then
  `<|im_start|>tool\n<tool_response>\n{"name": "python", "content": "mass_g = 0.0123"}\n</tool_response><|im_end|>`.
- Library rows: each `<tool_response>` block is ONE row (≤ 64 tokens; longer output is split by line, each piece
  tagged with the call id). The call block is also a row, paired with its response, so a question naming the call's
  subject hops call → result (as fetch_rows hops by name).
- The library registers these spans when it writes them. It does not rely on "." sentence ends, since tool output
  rarely has them.
- The executor's wrapper asks for labelled prints (`name = value`), so a result row carries its own key words for
  the text organs.

**Executor (`Package.run_tool`; plumbing, labelled TOOL in `info`, never counted as the mind's own computation).**
- **Loop:** the mind generates until `</tool_call>`. The planner parses the JSON (invalid → no call, labelled), runs
  it, appends the `<tool_response>` block to the session (the O(1) state reads it; the library keeps its span), and
  generation continues. At most 3 calls per answer.
- **Sandbox:** bubblewrap, present on both boxes, checked 11:13 IST: `--unshare-all` gives no network (checked
  locally: connect fails) and fresh pid / ipc / uts namespaces; also `--die-with-parent`, read-only /usr and the
  venv's site-packages (numpy / scipy / sympy), tmpfs /tmp as the only writable directory, no home, no repo.
- **Limits:** python `-I`; CPU 5 s, address space 1 GB, 32 files, 16 processes, 1 MB file size; wall clock 10 s;
  stdout + stderr capped at 4 KB (truncation is marked). Exceptions become an error row, so the mind sees the
  failure. Without bwrap the executor is DISABLED: it never runs unsandboxed.
- **Cost:** ~0.1-0.3 s per call (process start + run); the result row's tokens enter the state (O(1) bytes); 2 B per
  token of stored text.
- **Tests before first use:**
  - security: network blocked; writes outside /tmp fail; a fork bomb is capped; an infinite loop is killed at 5 s;
    a memory bomb is killed;
  - rows: the result is found by the text organs and by F2;
  - A26-T, pre-registered before its numbers: A26's protocols with the result arriving as a `<tool_response>` row
    (family split as in the addendum).
- **Gate (Lead):** build it when the first SCI-1 milestone shows the call format is learned (e.g. ≥ 0.8 of held-out
  prompts give a parseable call).

