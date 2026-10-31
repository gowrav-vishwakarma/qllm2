# Mind + Library benchmark (MLB-1) — DRAFT v0.2 (2026-10-31; v14 comments 14:10, organiser row format + user comparison rule 14:30 IST)

Status: **draft, not fixed.** Freeze condition for v1: the harness implements every
§2 cell and is validated on CPU at 32k (organiser), then v1 is committed as "MLB-1 v1
(fixed)" and the harness moves to `v16/bench.py`. Later changes are dated amendments.
No package run is scored before v1.

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
| T++ + library | optional diagnostic: the same library, its OWN key head | KV + library |
| Mamba-2 state | state only | state bytes |
| Mamba-2 + library | optional diagnostic | state + library |

**Comparison rule (user, 2026-10-31, via the organiser):** Transformer++ and
Mamba-2 are compared AS THEY ARE: their existing numbers and runs, full context
where it fits, no library bolted on. The "+ library" arms are optional diagnostics,
not headline rows. Honesty line that travels with the headline: "the baselines ran
without a library; the recall win belongs to the package (mind + library), and the
mind's own edge is R2". Training data is equal for
every arm; the package arms may add fetch-training data (stated per run, with
the ledger row).

## 2. Library half (from `scripts/session_index_probe.py`, Communicator)

Held-out prose filler, needles inserted at random positions, 20 candidate
values (chance 0.05), query at the end.

| task | content |
|---|---|
| needle | 1 definition "Definition: KEY refers to VALUE." |
| multi-needle 8 / 64 | 8 or 64 definitions, one queried (at 64 the values repeat; the queried value is one of 20) |
| latest wins | the queried key is redefined later; the newer value is right |
| two-hop | "K1 maps to K2" and "K2 maps to V" far apart; query K1 |
| natural keys | as multi-needle 8, but keys and values are entities drawn from real text (collisions and paraphrase allowed), not unique synthetic nouns |
| tool result | a "Result: job: X; status: v; owner: v; …" block inserted mid-stream, one field queried later; the exact organ fails on 3 of 4 fields by construction (no shared 4-gram), so this cell separates organ from fetch |
| paraphrased key | the query names the key in other words than the stored sentence (for the learned key, v14 L1) |
| negative | a query whose key was never stored; abstain = the fetch's top similarity below a threshold set on the positive cells (the organ abstains by construction on a miss); reported as abstain rate |

Lengths L: 2k, 8k, 32k, 128k, 400k, 1M. T++ full context is run as far as
memory allows and the limit is stated in the table (its KV cache at 128k is
~4.9 GB at this width and depth). The held-out filler cache is 500k tokens, so
1M documents wrap the filler: the organ then meets exact repeats of prose, while
the needles stay unique (stated next to every 1M number). Docs per cell: 32 on CPU (±0.09) for the first table; 100 on GPU (±0.05) for
any claim. Harness: `scripts/session_index_probe.py` (`fc5d682`), one JSON line per
document with fields {arch, ckpt, task, keys, needles, L, doc, T, mem{…} in bytes,
A_state, B_hit, B_ngram, B_hop2_chain, C_hop2_reread, C2_*, C5_rows,
C5_rows_per_event, C5_event_has_row, C5_hit / top5, C5_index_bytes, sec}; cell
mean = mean over docs. Read paths: A = state only, B = exact organ, C2 = frozen
address at the key token, C5 = gated rows; v1 package = v14's learned fetch via
`--key_fn` / `--dump_rows`.

## 3. Mind half (from the existing probes)

| task | content | source |
|---|---|---|
| rule at distance | shift_wide_new, affine_map_new, 8 shots, examples spread by filler to L = 0, 2k, 4k, 8k, 32k; D = mean at 2k + 4k | `run_learning_probe.py --suite b2 --spread`, probe seed ≥ 7000 (fresh), 400 trials |
| instruction persistence | the B2 builders (`v14/rules_ood`, held-out shift / affine values) with the rule stated ONCE in words at the start ("Rule: shift each letter by 3"), then L tokens of filler, then `x ->` queries. Two variants: **mind-only** (the rule must sit in state) and **library** (the rule is one row; the query "7 ->" shares no words with it, so the exact organ cannot fetch it and only a learned key can) | **new** source + probe; format-trained (v14 §7.3) |
| label binding at distance | label_category spread to L | `lcl_t300` |
| general LM | holdout loss; loss by token class (copy / seen / new) with `v13_sempty/tmp/attr3b.py`; PG-19 far gain | existing |

## 4. Memory accounting (every row)

- PAM: state bytes (notebooks + echo, fp32; ~3.9 MB at 102M) + library at L.
  Row format (v14): key 128-d bf16 256 B + start u32 + len u16 + session id u32 +
  write ordinal u32 + key version u16 = **272 B per row**; stored text as u16
  token ids (2 B per token); shared libraries add provenance (hash 8 B + source
  id 4 B). At ~1 row per 16 tokens: ≈ 6.4 MB at 128k and ≈ 23 MB at 1M, mind
  state included. The all-position index is reported for reference only.
- T++: KV cache at L = layers × 2 × L × dim × 2 bytes (bf16): ≈ 4.9 GB at 128k, ≈ 38 GB at 1M.
- Mamba-2: state bytes (13.4 MB).
- Also: params (incl. any table params), tokens/s at L.
- Fetch metrics (package arms, every row): fetch rate; abstain rate on the
  negative cells; span precision (right row top-1 / top-5); re-read tokens per
  query; rows per 1k tokens; write-gate recall (fraction of needles that got a row).

## 5. Pre-registered rules (v0 proposal)

- **R1 library recall:** the package ≥ 0.90 on needle, multi-needle 8, latest
  wins and two-hop at every L ≤ 128k, and ≥ 0.80 at 400k. **1M is reported, not
  gated, in v1** (CPU time; the learned key is trained to 32k); it becomes a bar
  by dated amendment once v14's L1 reports. Negative cells: abstain ≥ 0.90.
- **R2 mind edge:** rule-at-distance D (PAM) ≥ max(T++, Mamba-2) + 0.10 in every
  PAM seed (the s8 rule), with 8k and 32k reported.
- **R3 mind quality guard:** at equal data the mind's holdout is within +0.02 of
  Mamba-2 (600M); at 3B the gap is reported, and no bar-2 claim is made unless
  within +0.02. Loss by token class is always shown.
- **R4 memory:** at 128k the package's total memory (state + rows + stored text)
  ≤ 1/100 of the T++ KV cache; the ratio is in the table (expected ≈ 1/770).
- **R5 attribution (no longer a gate, per the comparison rule):** the headline
  carries the honesty line of §1. If the optional "+ library" diagnostics are run
  (each with its OWN key head trained the same way) and they match the package on
  R1, the write-up says the recall win is the library's and the mind's claim is
  R2 + R4.
- **Win** = R1, R2, R4 hold and R3's guard holds, with the §1 honesty line and R5 stated.

## 6. Protocol

- Two PAM training seeds; one per baseline and fairness arm (cost); probe seeds
  never used before (≥ 7000).
- Code pinned per run; logs by the naming convention; a ledger row per run
  (EXPERIMENT_INDEX "Run ingredients ledger").
- Caveat that travels with every claim: rule formats are trained with held-out
  values, not unseen rule families; retrieval itself exists, the claim is the
  mind–library interface and the O(1) mind's rule carrying.

## 7. Open for comments

1. v14: answered (row format, fetch metrics, R1 / R4 / R5 wording above).
2. Communicator: answered (row format, tool task, doc counts, 1M wrap). 400k on
   CPU ≈ 160 s per doc on c1ctlb: the headline 400k cells ≈ 15 CPU-hours overnight;
   the full grid on a 4090 slot after 18:00 IST.
3. Communicator: negative cells and both instruction-persistence variants agreed,
   going in today.
