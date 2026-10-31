# Mind + Library benchmark (MLB-1) — DRAFT v0 for comments (2026-10-31)

Status: **draft, not fixed.** v14 and the Communicator comment; the fixed
version is committed as "MLB-1 v1 (fixed)" and later changes are dated
amendments. No package run is scored before v1 is fixed.

Purpose: one table that judges the package (**O(1) mind + growing library**)
against Transformer++ with its full context and Mamba-2, with memory in MB in
every row and the mind's general quality never hidden. The direction is
EXPERIMENTS_SEMPY "Direction". The user's order: this table first, the
distributable model once it wins it.

## 1. Systems under test (~102M each, equal data, same tokenizer)

| arm | what it is | memory counted |
|---|---|---|
| PAM state | the mind alone, state only | state bytes |
| **PAM + session library** | the package: mind + per-session library (v0 = model-free exact organ + re-read; v1 = v14's learned fetch: gate, trained key at the salient token, re-read; one row per event) | state + library index + fetched text |
| T++ full context | the whole sequence in one window (its KV cache) | KV cache bytes at L |
| T++ + library | fairness arm: the same library and re-read | KV + library |
| Mamba-2 state | state only | state bytes |
| Mamba-2 + library | fairness arm | state + library |

Fairness arms exist so a win on recall is credited to the library, not the
mind, unless the mind arm is ahead of them too. Training data is equal for
every arm; the package arms may add fetch-training data (stated per run, with
the ledger row).

## 2. Library half (from `scripts/session_index_probe.py`, Communicator)

Held-out prose filler, needles inserted at random positions, 20 candidate
values (chance 0.05), query at the end.

| task | content |
|---|---|
| needle | 1 definition "Definition: KEY refers to VALUE." |
| multi-needle 8 / 64 | 8 or 64 definitions, one queried |
| latest wins | the queried key is redefined later; the newer value is right |
| two-hop | "K1 maps to K2" and "K2 maps to V" far apart; query K1 |
| natural keys | as multi-needle 8, but keys and values are entities drawn from real text (collisions and paraphrase allowed), not unique synthetic nouns |
| tool result | a "Result: {…}" block (tool-call style text) inserted mid-stream; a field of it is queried later |

Lengths L: 2k, 8k, 32k, 128k, 400k, 1M. T++ full context is run as far as
memory allows and the limit is stated in the table (its KV cache at 128k is
~4.9 GB at this width and depth). Docs per cell: ≥ 80 for the headline cells
(needle, multi-needle 8, two-hop; SE ≤ 0.05), ≥ 40 elsewhere.

## 3. Mind half (from the existing probes)

| task | content | source |
|---|---|---|
| rule at distance | shift_wide_new, affine_map_new, 8 shots, examples spread by filler to L = 0, 2k, 4k, 8k, 32k; D = mean at 2k + 4k | `run_learning_probe.py --suite b2 --spread`, probe seed ≥ 7000 (fresh), 400 trials |
| instruction persistence | a rule stated once at the start (held-out value, trained format), applied after L tokens of unrelated text | **new** source + probe; pre-registered as format-trained |
| label binding at distance | label_category spread to L | `lcl_t300` |
| general LM | holdout loss; loss by token class (copy / seen / new) with `v13_sempty/tmp/attr3b.py`; PG-19 far gain | existing |

## 4. Memory accounting (every row)

- PAM: state bytes (notebooks + echo, fp32; ~3.9 MB at 102M) + library at L:
  fetched/stored text bytes + rows × key bytes (bf16) + pointers. Report the
  salient-row index (one row per event, the package) and, for reference, the
  all-position index.
- T++: KV cache at L = layers × 2 × L × dim × 2 bytes (bf16).
- Mamba-2: state bytes (13.4 MB).
- Also: params (incl. any table params), tokens/s at L.

## 5. Pre-registered rules (v0 proposal)

- **R1 library recall:** the package ≥ 0.90 on needle, multi-needle 8, latest
  wins and two-hop at every L ≤ 128k, and ≥ 0.80 at 400k and 1M.
- **R2 mind edge:** rule-at-distance D (PAM) ≥ max(T++, Mamba-2) + 0.10 in every
  PAM seed (the s8 rule), with 8k and 32k reported.
- **R3 mind quality guard:** at equal data the mind's holdout is within +0.02 of
  Mamba-2 (600M); at 3B the gap is reported, and no bar-2 claim is made unless
  within +0.02. Loss by token class is always shown.
- **R4 memory:** at 128k the package's total memory ≤ 1/20 of the T++ KV cache;
  the ratio is in the table.
- **R5 fairness:** T++ + library and Mamba-2 + library are run. If they match
  the package on R1, the recall win is the library's, and the claim is R2 + R4
  only.
- **Win** = R1, R2, R4 hold and R3's guard holds, with R5 stated.

## 6. Protocol

- Two PAM training seeds; one per baseline and fairness arm (cost); probe seeds
  never used before (≥ 7000).
- Code pinned per run; logs by the naming convention; a ledger row per run
  (EXPERIMENT_INDEX "Run ingredients ledger").
- Caveat that travels with every claim: rule formats are trained with held-out
  values, not unseen rule families; retrieval itself exists, the claim is the
  mind–library interface and the O(1) mind's rule carrying.

## 7. Open for comments

1. v14: the library row format (one row per event, keys versioned with the
   model) and the fetch's own metrics (fetch rate, precision of fetched spans).
2. Communicator: doc counts, the natural-key and tool-result task designs,
   the 400k / 1M cells (time on CPU; which arms).
3. Both: the instruction-persistence task (format and data source), R1/R4
   thresholds, whether 1M is a headline or a reported cell.
