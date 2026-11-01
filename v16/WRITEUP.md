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
   cost <= +0.021 nats); the text library alone finds template / made-up facts 1.00 at 300k and the mind writes them
   0.65-0.90 cold (150x the trained length; ~4.5 MB vs ~11 GB KV).

Honesty lines that travel with every claim: the recall win is the LIBRARY's (the exact-key 1.00 is a model-free text
organ, identical for every body); a transformer given the same library and a cold read reads single facts as well and
multi-field records better (tool 0.90 vs 0.45 for both fixed-state bodies); the mind's own parts are rules in state,
constant cost while the session is held, and its own-state keys where words differ (synonyms 0.575 vs MiniLM 0.35,
natural questions 0.775 vs BM25 0.525 on the template wording); the bounded 128-token window (if adopted) is a known
hybrid part (Griffin / Samba / Based), the whole is the claim; reworded questions collapse cold reads for every body
and the package answers them 0.68-0.73 on natural keys; multi-fact composition (BABILong QA2 / QA3, story order /
summary) is unsolved by any body at 103M.

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

### 2.6 BABILong (as is + package; 100 docs per cell) — filling
| task | b2p2 package 0k / 1k / 2k | cu2a package | as-is bodies | planner chain |
|---|---|---|---|---|
| qa1 | 0.85 / 0.74 / 0.77 | 0.99 / 0.69 / 0.69 | pending | pending |
| qa2 | 0.05 / 0.11 / 0.13 | 0.00 / 0.09 / 0.09 | | |
| qa3 | 0.01 / 0.07 / 0.06 | 0.00 / 0.00 / 0.08 | | |
| qa4 | 0.45 / 0.37 / 0.25 | 0.08 / 0.30 / 0.22 | | |
| qa5 | 0.53 / 0.46 / 0.54 | 0.52 / 0.47 / 0.53 | | |
Reference block (different data / tokens, tuned on the benchmark): Mamba-130M fine-tuned 97.7 % mean to 128k; RMT-137M
85 %; ARMT-137M 97.8 %; untuned GPT-4 Turbo 65.1, Llama-3.1-70B 67.8.

### 2.7 Known-benchmark suite (kb1) — pending; blocks: 600M tier, 3B tier, reference (SmolLM2-135M 2T, Pythia-160M 300B, GPT-2)

### 2.8 Scaling watch — see the board; the matched 1.5B tier fills the rules / reach slope.

## 3. Limits stated
- Rules from words alone: a lookup at 103M (unseen value 0.00; one demonstration 0.97); rules reach the mind as
  demonstrations (0.78-0.88 refreshed near the question).
- One-read reach without tools ~2k; 4k with the contrast; 8k with echo dilation + contrast; nothing at 16k.
- The mind generalises values and keys, not formats (cu2a: trained formats 0.80-0.99, held-out wordings 0.26).
- Multi-fact composition and counting: not solved by any body at this size; the planner chain is library plumbing.
- Language at 3B: +0.013 nats behind T++, +0.055 behind Mamba-2 on one seed; T++ gains most from data.
