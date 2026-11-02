# An O(1) mind with an outside library: write-up draft (Lead, started 2026-11-01 17:00 IST)

Rules of this document: claims only within a TOKENS tier (same data, same params); external models in a separate
reference block, never a verdict; every table carries params, training tokens and MB; the headline cell is the natural
variant (synonym key, varied wording) with the trained template secondary; "as is" means no library, no tools, no
tuning on the benchmark. Numbers are copied from the board / logs with their hashes; this file is updated as rows land.

## 1. Claims (what we say, in order of strength)

1. **Rules learned in context survive in a constant-size state across thousands of tokens of unrelated text**, where
   same-size transformers and Mamba-2 trained on the same data do not. (s8, 600M: PAM 0.645 / 0.691 / 0.741 vs T++
   0.134 / 0.134, Mamba-2 0.116 / 0.098; three PAM seeds.)
2. **Intelligence separate from knowledge: a rule held only in the state applied to a fact fetched at question time**,
   beyond the trained length. (R x K: PAM 0.55 / 0.48 plain, 0.61 / 0.62 with the contrast at 2k, 0.45 / 0.49 at 8k;
   T++ 0.01 / 0.07; Mamba-2 0.03 / 0.00; no rule anywhere 0.00; RAG + few-shot ceiling PAM 0.83-0.92, T++ 0.30,
   Mamba-2 0.44. 102.6-102.9M params, 600M tokens, same data. 2k bar 0.70 missed by 0.08-0.09; 8k bar met; the
   planner substitutes the fetched value; PAM's 8k uses its two query-time tools.)
3. **Holding beyond the training length at constant cost.** Package recall 1.00 (20-way) on needle / 8-needle /
   latest-wins / two-hop at 32k and 128k with ~3 MB held vs a 4.9 GB KV cache; one skill read carried to 8k on three
   minds with two query-time tools (free 0.43-0.58, 20-way 0.74-0.89; T++ 0.195, Mamba-2 0.015 at the same gap; prose
   cost <= +0.021 nats); the package recalls 0.90-1.00 at 1k / 16k / 100k back inside a 128k session (the cap); longer lengths
   are out of scope by the user's rule.

Honesty lines that travel with every claim: the recall win is the LIBRARY's (the exact-key 1.00 is a model-free text
organ, identical for every body); a transformer given the same library and a cold read reads single facts as well and
multi-field records better (tool 0.90 vs 0.45 for both fixed-state bodies); the mind's own parts are rules in state,
constant cost while the session is held, and its own-state keys where words differ (synonyms 0.575 vs MiniLM 0.35,
natural questions 0.775 vs BM25 0.525 on the template wording); the bounded 128-token window (if adopted) is a known
hybrid part (Griffin / Samba / Based), the whole is the claim; reworded questions collapse cold reads for every body
and the package answers them 0.68-0.73 on natural keys; multi-fact composition (BABILong QA2 / QA3, story order /
summary) is unsolved by any body at 103M.
Package v0.1 (d554d80): decimals were truncated by the row splitter for every body ("12.347" fetched and answered as
"12."); earlier quoted values contained no decimals.

## 2. Tables (to fill from the logs; hashes in the board entries)

### 2.1 Package table (MLB-1), 32k and 128k, seeds 18000 / 18500
| cell | package 20-way (b2p2 / c1ctlb) | package free | T++ as is | Mamba-2 as is | PAM as is |
|---|---|---|---|---|---|
| needle | 1.00 / 1.00 at 32k and 128k | 0.975-1.00 syn, 0.71-0.85 nat | 0.04 | 0.14 | 0.08-0.13 |
| 8-needle | 1.00 / 1.00 | " | 0.04 | 0.06 | 0.09 |
| latest wins | 1.00 / 1.00 | 1.00 syn / 0.71-0.83 nat | 0.06 | 0.18 | 0.14-0.19 |
| two-hop (chain + re-read) | 1.00 / 1.00 | 1.00 | 0.04 | 0.09 | 0.05-0.10 |
| paraphrase | 0.46-0.66 | | 0.05 | 0.01 | |
| tool field | 0.29-0.50 | | 0.01 | 0.05 | |
| memory at 128k | 1.15 MB state + 0.26 MB rows + 0.26 MB text + ~1.5 MB organ | | 4,932 MB KV | 13.9 MB state | |

### 2.2 The same text library for every body (cold read), 32k, seed 18000: free (sentence), syn / nat
| cell | T++ (tfc1b) | Mamba-2 (mbs2b) | PAM (b2p2) |
|---|---|---|---|
| kv | 1.00 / 0.875 | 1.00 / 1.00 | 0.90 / 0.775 |
| para | 0.70 / 0.65 | 0.975 / 0.95 | 0.775 / 0.80 |
| tool (uncut record) | 0.90 / 0.90 | 0.425 / 0.475 | 0.475 / 0.40 |
| hop2 | 1.00 | 1.00 | 1.00 |
| fetch (any body) | organ hit 1.00 on exact / surface / tool / hop2; neg abstain 0.925-1.00; syn 0.00; natq ~0.1 written |

### 2.3 Holding the session (A18), kv at 32k, same fetched rows, free (syn / nat)
| arm | memory per live session | kv | para | tool | natq |
|---|---|---|---|---|---|
| T++ cold (row only) | ~1.1 MB | 1.00 / 0.90 | 0.70 / 0.675 | 0.90 / 0.90 | 0.075 |
| T++ 2k window + contrast | 67 MB | 0.95 / 0.95 | 0.95 / 0.975 | 0.75 / 0.825 | 0.425 |
| T++ whole 32k session | 1,060 MB | 0.00 / 0.00 | 0.00 | 0.00 | 0.00 |
| PAM package (warm + contrast) | 1.15 MB | 0.975 / 0.925 | 0.90 / 1.00 | 0.75 / 0.675 | 0.475 |

### 2.3b Needles by distance back (A25, package v0, b2p2; 128k; strict abstain on)
| distance back | ~1k | ~16k | ~100k |
|---|---|---|---|
| package (template wording) | 0.90-1.00 | 0.90-1.00 | 0.90-1.00 |
| package, varied wording, natural keys | 0.60-0.80 (strict 0.10-0.30) | | |
| mind alone | 0.00 | 0.00 | 0.00 |
| T++ 2k window alone | 0.00-0.20 | 0.00 | 0.00 |
| T++ window + the package's row | 0.90-1.00 | 0.90-1.00 | 0.90-1.00 |
| strict abstain on never-stored | 0.95 (lenient fetches a row for every never-stored question: the mind's keys never abstain) | | |
A25 verdict at 128k (cd55706): recall flat from 1k to 100k back (pooled kv 0.967 / 0.933; the one far miss fetched the
right row and misspelled the value); varied wording on natural keys 0.80 at ~100k; 2.65 MB per 128k session excluding
the address (state 1.15 + text 0.26 + indexes 1.23; the address 78 MB int8 or 0.64 MB sign bits, tied on accuracy);
read 11.4 s per session, 0.38 s per answer on a 4090. The edge over a transformer with a 2k window plus the same row
is COST, not accuracy (that transformer reads the row 0.60-1.00); the edge over a transformer holding the session is
both (0.00 at 32k).
Cost (PACKAGE.md v0 §7): state per body (b2p2 1.15 MB, 16nb 3.92 MB) + text 2 B / token + indexes ~9.4 B / token + the
F2 address (int8 592 B / token, or sign bits ~4.9 B / token); session read 11.4 s at 128k, an answer ~0.4 s; a T++ KV
cache for the same session 4.8 GB at 128k.

### 2.4 Realistic wording (A9 / A21 / step 4), 32k, seed 18000: the headline from here
| | made-up keys (text fetch, step 4) | natural keys (text) | natural keys (package: F2 + warm + contrast) |
|---|---|---|---|
| fetch | 1.00 | 0.60-0.80 | 0.70-0.825 |
| written | 0.45-0.65 (cold, any body) | 0.15-0.30 (cold) | 0.675-0.725 (kv / kv1 / kvupd), natq 0.525 |

### 2.5 Story exam (11 questions, chat + plain, keyword hits of 22)
| | 2k | 4k | 8k |
|---|---|---|---|
| b2p2 alone / package | 1 / 8 | 3 / 8 | 2 / 7 |
| cu2a alone / package | 1 / 6 | 1 / 5 | 1 / 5 |
| T++ alone, Mamba-2 alone | 1, 0 | 0, 0 | 0, 0 |

### 2.6 BABILong (RMT-team/babilong, qa1-qa5, 100 docs per cell, 0k-32k; the 128k cap; 103M params, 600M tokens; no tuning on the benchmark; logs/v16_babilong_*_773d751_20261101_1548)
Package = stream the input into the state, fetch top-3 rows by name / latest / F2, warm + contrast read; chain = planner sub-questions (TEXT, library plumbing).
| task | T++ as is (tfc1b) | Mamba-2 as is (mbs2b) | PAM as is (b2p2 / cu2a) | PAM package b2p2: 0k / 4k / 32k | PAM package cu2a: 0k / 4k / 32k | chain b2p2 0k / 32k |
|---|---|---|---|---|---|---|
| qa1 single fact | 0.00 at every length | 0.00-0.03 beyond 4k | 0.00-0.06 | 0.85 / 0.71 / 0.59 | 0.99 / 0.58 / 0.53 | 0.85 / 0.59 |
| qa2 two facts | 0.00 | 0.00-0.01 | 0.00-0.01 | 0.05 / 0.18 / 0.19 | 0.00 / 0.14 / 0.12 | 0.23 / 0.16 |
| qa3 three facts | 0.00 | 0.00-0.01 | 0.00-0.02 | 0.01 / 0.06 / 0.07 | 0.00 / 0.04 / 0.07 | 0.08 / 0.04 |
| qa4 two-arg relation | 0.00 | 0.00-0.03 | 0.00-0.04 | 0.45 / 0.32 / 0.19 | 0.08 / 0.25 / 0.11 | 0.37 / 0.18 |
| qa5 three-arg relation | 0.00 | 0.00 | 0.40 at 0k, <= 0.04 beyond | 0.53 / 0.56 / 0.58 | 0.52 / 0.50 / 0.59 | 0.56 / 0.53 |
| mean qa1-5 | 0.00 | ~0.01 | ~0.01-0.09 | 0.38 / 0.37 / 0.32 | 0.32 / 0.30 / 0.28 | 0.42 / 0.31 |
Reading: at 103M / 600M tokens no body reads BABILong as it is beyond 0k (T++ 0.00 everywhere, Mamba-2 <= 0.03, PAM <= 0.06); the
package carries single-fact and relation questions (qa1 0.53-0.85, qa5 0.52-0.59) to 32k with a flat-to-slow decline (qa1 0.85 ->
0.59), while two- and three-fact composition (qa2, qa3) stays <= 0.20 for every column: a composition gap of the mind, not of
the fetch (the chain finds the rows and gains only at 0k on qa2, 0.05 -> 0.23). Package cost per doc on a 4090: 0.7 s at 0k, 13 s
at 32k, state 1.15 MB. Reference block (different data / tokens, tuned on the benchmark): Mamba-130M fine-tuned 97.7 % mean to
128k; RMT-137M 85 %; ARMT-137M 97.8 %; untuned GPT-4 Turbo 65.1, Llama-3.1-70B 67.8. Our column is "as is, 600M tokens".

### 2.7 Known-benchmark suite (kb1, Mind 41e7002; logs/kb1/v13_sempty_kb1_table_ccf187c.md): MC mean over HellaSwag /
ARC-e / ARC-c / PIQA / Winogrande / OpenBookQA, then LAMBADA acc / ppl; all bodies as is, bf16 (fp32 re-check: identical)
| tokens tier | PAM | T++ | Mamba-2 |
|---|---|---|---|
| 600M (103M params) | 0.391 (b2p2) / 0.388 (cu2a); LAMBADA 0.146 / 578 | 0.388; 0.148 / 1299 | 0.392; 0.132 / 945 |
| 3B | 0.403 (r3c, 16nb body, no b2 stage); 0.179 / 376 | 0.406; 0.196 / 404 | 0.415; 0.176 / 305 |
| slope 600M -> 3B | +0.012 | +0.018 | +0.023 |
| window arms, 220M | W0 0.377-0.383, W128 0.380-0.384, W512 / W1024 0.381: the window costs nothing | | |
Reference block (other data, far more tokens; no claim): SmolLM2-135M 2T 0.487 (LAMBADA 0.428); GPT-2 ~10B 0.397 (0.307);
Pythia-160M 300B 0.396 (0.354), in fp32 (Pythia degrades in bf16, our bodies do not).
Reading: on known general tasks the three bodies are LEVEL at equal data; PAM's edge is rules held in state (s8, R x K),
not general knowledge. PAM gains the least from 600M -> 3B on this suite, with the body-change caveat; the matched 1.5B
tier (t15) gives the clean slope. What to improve: plain LM quality and tokens.

### 2.8 Scaling with data at equal params (~102-103M), same mix: the matched 600M and 1.5B tiers (Mind 9616861, t15m2 verdict; 3B from the earlier runs without rule docs)
| body | holdout 600M -> 1.5B | rules at distance (s8 D) 600M -> 1.5B | one read used 8k later (clk2) 1.5B | rules at L = 0, 1.5B |
|---|---|---|---|---|
| PAM M2 (b2p2 recipe) | 3.4800 -> 3.3255 | 0.65-0.74 (3 seeds) -> 0.579 | 0.87 | 1.00 |
| PAM 16nb (plain) | 3.5662 -> 3.3784 | 0.768 -> 0.545 | 0.47 | 1.00 |
| Transformer++ | 3.5572 -> 3.3593 | 0.134 -> 0.047 | 0.29 | 0.53 |
| Mamba-2 | 3.4754 -> 3.3025 | ~0.10 -> 0.033 | 0.00 | 0.745 |
Reading: language improves alike with data (about -0.19 for every body; Mamba-2 keeps a 0.02-0.08 lead, PAM M2 passes T++
at 1.5B); rules held at distance FALL with data for every body, the baselines most (they also lose the rule placed right
before the question), while PAM keeps a +0.50 lead and its reach with the query-time tools GROWS (skill pack 4k + contrast
0.70 -> 0.775 on 16nb; baselines 0.105 / 0.005). Pre-registered bars at 1.5B: rules MET (both PAM bodies), language bar
3.3225 missed by 0.003 (M2) / 0.056 (16nb). Tool-record reading stays the package's job (T++ 0.875 / Mamba-2 0.69 / PAM
0.30). The window (W128) is a modest reach unit on M2 (+0.09 rules, shrinking with data) and hurts 16nb; it is not in the
release body. The 3B points (r3c 3.219 vs T++ 3.206 / Mamba-2 3.164) come from runs without rule docs, so they carry no
rules column. The product-size run (SCI-1: 386M M2, 11.5B tokens, 4k, science mix) is its own tokens tier and is
reported against public small models only as a reference block.

## 3. Limits stated
- Rules from words alone: a lookup at 103M (unseen value 0.00; one demonstration 0.97); rules reach the mind as
  demonstrations (0.78-0.88 refreshed near the question).
- One-read reach without tools ~2k; 4k with the contrast; 8k with echo dilation + contrast; nothing at 16k.
- The mind generalises values and keys, not formats (cu2a: trained formats 0.80-0.99, held-out wordings 0.26).
- Multi-fact composition and counting: not solved by any body at this size; the planner chain is library plumbing.
- Language at 3B: +0.013 nats behind T++, +0.055 behind Mamba-2 on one seed; T++ gains most from data.
- Stated conventions (A26 day-0, ccdf41f): applying one is a skill that no body has at 600M-1.5B tokens. The
  fetch delivers the value, and the convention is dropped: units, rounding and unit conversion score right
  0.00-0.05 (the value is copied 0.2-0.8), the same as without the protocol, even with the protocol ~600 tokens
  back. Reference tiers: b2p2 (102.9M, 600M tokens) through the package; T++ / Mamba-2 at 1.5B tokens, as is and
  with the library. SCI-1's scores are read against this.
