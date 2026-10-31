# Mind + Library benchmark (MLB-1) — v1 (FIXED 2026-10-31 13:45 IST; harness `v16/bench.py` @ `d319e8d`)

Status: **v1, fixed.** Frozen after Lead validated every §2 cell of `v16/bench.py` at
32k on CPU (`d319e8d`; logs `logs/v16_bench_validate32k_{d1e1a56,b_d319e8d}_*.jsonl`).
Any later change is a dated amendment at the end of this file, written before the
numbers it affects. Drafts v0–v0.3 are in git history (§7).

**Thesis (user, 2026-10-31):** "having intelligence separate from knowledge is itself
a big novelty, specifically when it holds beyond the training context." Every cell and
every unit says which clause it serves: *holds beyond the training context* (the
rule-carrying mind, skill-pack persistence, the skill register) or *intelligence
separate from knowledge* (the exact organ, the learned fetch, text-first packs). A unit
that serves neither is a cost table, not a claim.

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

## 2. Library half (harness `v16/bench.py` @ `d319e8d`, Lead) — clause: *intelligence separate from knowledge*

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
any claim. Harness: `v16/bench.py` (`d319e8d`; was `scripts/session_index_probe.py`, `fc5d682`), one JSON line per
document with fields {arch, ckpt, task, keys, needles, L, doc, T, mem{…} in bytes,
A_state, B_hit, B_ngram, B_hop2_chain, C_hop2_reread, C2_*, C5_rows,
C5_rows_per_event, C5_event_has_row, C5_hit / top5, C5_index_bytes, O_reread,
C2_maxsim, C5_maxsim, C5_rows_per_16tok, C5_library_bytes_272,
mem.library_272B_per_event, A_letters / O_letters (instr), P_concat / P_margin
(planner v0 on the top-5 spans), C5_bursts, C5_bursts_per_16tok, C5_burst_in_needle,
C5_burst_hit, C5_library_bytes_272_bursts, mem.install_cost_*, sec}; cell
mean = mean over docs. Read paths: A = state only, B = exact organ, C2 = frozen
address at the key token, C5 = gated rows, O = oracle re-read (the fetch's upper bound); v1 package =
v14's learned fetch via `--key_fn` / `--dump_rows` / `--gate_tau`. Tasks in the
harness: kv, kv1, kvupd (latest wins), hop2, tool, instr, neg, para; synthetic and
natural keys (16 task × key cells).

Readings from the validation that bound what each path can claim (Lead, 32k): the
frozen address cannot abstain (never-stored max-sim 0.86 vs stored 0.79–0.88), so the
abstain bar applies to the learned fetch only; the burst gate at tau 10 segments
correctly (0.6 rows per 16 tokens, a burst inside the needle 100 %), but the burst-mean
key retrieves 0/4, so retrieval needs the learned key (Library's L1).

## 3. Mind half — clause: *holds beyond the training context* (general LM = quality guard)

| task | content | source |
|---|---|---|
| rule at distance | shift_wide_new, affine_map_new, 8 shots, examples spread by filler to L = 0, 2k, 4k, 8k, 32k; D = mean at 2k + 4k | `run_learning_probe.py --suite b2 --spread`, probe seed ≥ 7000 (fresh), 400 trials |
| instruction persistence | the B2 builders (`v14/rules_ood`, held-out shift / affine values) with the rule stated ONCE in words at the start ("Rule: shift each letter by 3"), then L tokens of filler, then `x ->` queries. Two variants: **mind-only** (the rule must sit in state) and **library** (the rule is one row; the query "7 ->" shares no words with it, so the exact organ cannot fetch it and only a learned key can) | harness `instr` (trained "Letter rule." format, `--shots` control); training docs "stated rule + demonstrations" are the curriculum's job. Baseline (b2p2): 0.000 at 2k / 8k / 32k even with the rule re-read in front; 2 demonstrations 0.625 / 1.0 / 0.75 |
| label binding at distance | label_category spread to L | `lcl_t300` |
| general LM | holdout loss; loss by token class (copy / seen / new) with `v13_sempty/tmp/attr3b.py`; PG-19 far gain | existing |
| **skill-pack persistence** (flagship) | a pack of demonstrations (B2 formats, held-out values; 8 shots, packed together) is fetched ONCE at t0; queries of the same rule follow at t0 + 2k / 8k / 32k / 128k / 1M of unrelated text, with the pack gone. PAM: the pack read once into the state, nothing re-fetched. T++: must re-fetch the pack into its window for each query (its KV cost counted per query) or hold the whole stream. Mamba-2: read once, like PAM | **defined in v1, implemented before its first score** (the code must match this text); probe seeds ≥ 7000. Packs carry demonstrations, not only a rule sentence (words do not write the rule state, see `instr`) |
| **replay integration** | a fetched span replayed through the state must combine with what the state already holds: two-hop where hop 1 sits in the state (read earlier) and hop 2 arrives as a replayed span; and a rule held in state applied to an item that arrives only in a replayed span | defined in v1, reported (no gate) in v1 |

**Prediction for skill-pack persistence (written before any number, from our own
measurements):** the trained mind's long memory is the echo memory, whose heads keep
filler with half-lives of 660–3100 tokens (gate diagnostic), and spread affine `_new`
already falls to 0.44–0.61 at 8k. So the CURRENT body is expected to lose the skill
somewhere between 8k and 32k. Winning the flagship cell will likely need (a) rule docs
with long gaps at long training lengths (cheap for an O(1) mind: per-token cost does not
grow with the sequence, a transformer's does) or (b) a structural skill register (a
slot written once at induction, no decay; a new mechanism, v14's lane). The cell is in
the spec so that this is measured, not assumed.

## 3b. Related work and what differs (novelty rule, user 2026-10-31)

| closest prior work | what it does | what differs here |
|---|---|---|
| kNN-LM (Khandelwal et al. 2020), RETRO (Borgeaud et al. 2022) | retrieval over a corpus datastore (hidden state → next token; chunks) | nothing in the mechanism for the knowledge library: SL0 is kNN-LM on a PAM mind, reported as the knowledge organ's cost table, never as an invention |
| EM-LLM (Fountas et al. 2024) | a transformer's KV split into events by Bayesian surprise, recalled by similarity + temporal contiguity, training-free | rows are written by a recurrent O(1) mind keyed by its own state, and read back as raw text through the state; no KV cache is kept |
| Titans (Behrouz et al. 2025) | a neural memory learned at test time (surprise = gradient, momentum, decay) beside attention | our state is a fixed delta/phase memory with no attention core; the growing library is non-parametric; the claim is rule carrying across distance |
| State Soup (Pióro et al. 2024) | recurrent states (Mamba-2) stored, retrieved and mixed as task vectors | skill packs are TEXT (portable across mind versions), induced once, then carried by the mind across long unrelated text after the demonstrations are gone; persistence across distance is measured, not assumed; Mamba-2 loses the rule at ~110-token gaps in our data |
| Memorizing Transformers (Wu et al. 2022), Landmark attention | kNN over past KV inside attention | no attention and no KV at any length |

What is ours and is protected: the rule-carrying mind (B2 spread, s8) and the skill
library built on it (skill-pack persistence); unseen rule FAMILIES are the frontier.

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

## 5. Pre-registered rules (v1, fixed)

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
- **R6 stated rule (mind):** 0-shot stated rule (instr, mind-only) ≥ 0.75 at 32k,
  inside and past the trained length; T++ and Mamba-2 reported as they are.
- **R7 skill-pack persistence (flagship, mind):** at t0 + 32k, PAM (pack read once) ≥
  max(Mamba-2 read once, T++ without re-fetch) + 0.10 in every PAM seed; T++ WITH
  re-fetch is reported with its KV cost per query; 128k and 1M reported. The written
  prediction (§3) is that today's checkpoints fail R7.
- **Win** = R1, R2, R4 hold and R3's guard holds, with the §1 honesty line and R5
  stated. R6 and R7 are claimed separately (they are the mind's beyond-context claims).

## 6. Protocol

- Two PAM training seeds; one per baseline and fairness arm (cost); probe seeds
  never used before (≥ 7000).
- Code pinned per run; logs by the naming convention; a ledger row per run
  (EXPERIMENT_INDEX "Run ingredients ledger").
- Caveat that travels with every claim: rule formats are trained with held-out
  values, not unseen rule families; retrieval itself exists, the claim is the
  mind–library interface and the O(1) mind's rule carrying.

## 7. Change log

- v0 (12:56 IST, `2e92399`): first draft (arms, both halves, MB accounting, R1–R5).
- v0.1 (13:01, `c15aed3`): Library's comments (272 B rows, R4 ≤ 1/100, 1M reported not
  gated, own key heads for fairness arms, fetch metrics, negative cells, instruction
  persistence variants).
- v0.2 (13:02, `aa7fa23`): Lead's row format and tool / paraphrase / negative cells, 1M
  filler wrap, docs 32 CPU / 100 GPU; the user's comparison rule (baselines as they are).
- v0.3 (13:27, `72c8e5e`): novelty rule, related work, skill-pack persistence and
  replay integration cells, the written prediction; thesis line (`b2c02a7`).
- **v1 (13:45, this commit): frozen** on Lead's 32k validation; harness `d319e8d`;
  R6, R7 added; validation readings recorded in §2.

## 8. Amendments (dated, each before the numbers it affects)

**A1 (2026-10-31 13:55 IST, before any skill-pack or replay number; Lead's decision).**
The skill-pack persistence and replay-integration cells are implemented by Mind in
`scripts/run_skillpack_probe.py` (rows in bench.py's vocabulary: arch, ckpt, task,
sub, L, doc, T, A_state, O_reread, mem{state_bytes, tpp_kv_bytes_full,
tpp_refetch_kv_bytes}, sec; Lead reviews the code against this text before the first
scored run). **In v1 the pack is PLACED at t0;** fetching a pack by key from a library
of packs is bench.py's and lands with Library's L1. Pack = the B2 header + 8
demonstration lines (shift_wide_new, affine_map_new; held-out values); probe seeds
8000–8099 (never used); 100 docs per cell; L = 0, 2k, 8k, 32k, 128k (1M reported
later). Arms: PAM read once (A_state), Mamba-2 read once (A_state), T++ without
re-fetch = the whole stream in its window (A_state, as far as memory allows), T++ with
re-fetch = the pack placed before the query (O_reread, KV cost
mem.tpp_refetch_kv_bytes). Replay integration (reported, no gate): replay_refresh
(R_alone vs R_state_plus vs A_state) and replay_hop2 (H_integrate vs H_oracle vs
H_hop2only; 4 owners).

**A3 (2026-10-31 15:00 IST, before any number uses it; Lead).** `v16/bench.py` gained two
additive options, `--vec_layer k` (take the address vector from the mind's residual
stream after block k instead of the pre-logit vector) and `--centre` (subtract the
running mean of that vector before matching). They come from Library's finding that the
mind's mid-layer centred residual is a far better zero-training address than the
pre-logit vector. Existing cells and their defaults are unchanged; every row records
`vec_layer` and `centre`, so numbers with and without the options are never mixed.

**A4 (2026-10-31 15:55 IST, before any number uses it; Lead). `v16/bench.py` @ `0eb2ead`.**
(a) Bug fixed: per-document seeding used Python's per-process randomised `hash(task)`, so
separate runs with the same `--seed` built different documents; it now uses `zlib.crc32`
(stable). All earlier bench.py numbers remain valid samples and within-run comparisons are
paired, but cross-run comparisons (separate processes, e.g. ORG-1's PAM vs T++ vs Mamba-2)
were UNPAIRED. Headline readings do not change (state-only floors at 0.0, organ at 1.0);
any PAIRED claim is rerun at `0eb2ead`. `scripts/run_skillpack_probe.py` does not use
`hash()` (checked by Mind: all seeds are stable tuples via `np.random.default_rng` and
`sum(map(ord, task))`), so the skill-pack and replay cells are unaffected and paired.
(b) `key_fn` receives the same key vectors as the built-in path (the layer-k residual under
`--vec_layer`); a 3-argument form serves learned burst keys. (c) New C5 span-rule fields
(answer inside the stored row span, 24 tokens) beside the old rule; rows record which rule.

**A5 (2026-10-31 21:20 IST; Lead): streaming caveat.** Every memory figure for an M2 body
(notebooks in blocks 3, 7, 11, 15 + shared reads; b2p1 / b2p2 / b2p3, state 1.15 MB) is
COMPUTED from the state size, not yet demonstrated by streaming: the M2 body has no
carried-state chunked prefill and no O(1) decode yet (a Mind release blocker). M1 bodies
(e.g. `c1ctlb`, no shared reads) stream today. Until streaming lands with parity tests,
M2 MB figures carry this line. Decision rule (Lead): if M2 streaming takes more than ~1 day,
quantify the M1-vs-M2 gap on the skill-pack and LM cells and pick the body that streams.

**A6 (2026-10-31 21:35 IST): the A5 streaming caveat is lifted.** The M2 body now streams:
carried-state chunked prefill and O(1) token-by-token decode (`41f8ae8`). Parity:
prefix-then-suffix vs one full pass 2.0e-4, decode vs full 5.0e-5 (b2p2, logit scale ~18;
CPU fp32 and the 4090 Triton path fp32; bf16 within its own chunking noise), self-test
24/24. M2 memory figures (1.15 MB state) are now demonstrated by streaming. Example
(b2p2, O(1) decode): "Map rule. 3 -> 10, 5 -> 14, 8 -> 20, 11 ->" continues " 26", then
generates correct lines of its own (27 -> 58, 6 -> 16, 28 -> 60, 25 -> 54, 15 -> 34).

**A7 (2026-11-01 00:55 IST; user via Lead): free generation is the primary number, and every
headline checkpoint ships shown generations.** User: "we need all these measures, but text
generation also counts: not a database, a model that is intelligent AND expresses its part well."
- **Primary = free generation.** R1, R6 and R7 headline cells report free generation first: the
  greedy next token over the WHOLE vocabulary equals the answer token (the answers are single
  tokens; a multi-token answer must match exactly under greedy decoding of its length). The
  20-way candidate score stays as the secondary column, for continuity with the v1 rows. The
  bars keep their values and apply to the free number from A7 on.
- **Harness:** `v16/bench.py` `P_free` / `A_free` (Lead; package re-read and state);
  `scripts/run_skillpack_probe.py` `*_free` columns (`a073d0d`; 20-way unchanged on 80 / 80
  cells vs the earlier run).
- **Rows from before A7** are 20-way only; they read "20-way, free pending" until re-run.
  Runs pre-registered before A7 (lg8r, cu1) keep their bars on the score they pre-registered
  and report free beside it.
- **Why it matters (first datum, b2p2 skill-pack, 20 docs):** at L = 0 and on re-read, 20-way
  and free agree (1.00 / 1.00). At L = 2k after one read, 20-way 0.88 but free **0.68**: the
  candidate list overstates a decaying skill. A fading state still ranks the right answer
  above 19 others more often than it can say it outright.
- **Shown generations:** every headline checkpoint ships 3 shown generations (prompt, decoded
  output, checkpoint path + hash; greedy unless stated), via `v13_sempty/generate.py` or the
  probe. A6's b2p2 "Map rule" demo is the model.
- **R3 is promoted to a first-class line ("expression")**, no longer a footnote: holdout, loss by
  token class (attr3b) and the PPL table go on the board with the gap stated. At 3B: r3c holdout
  3.2191 vs T++ 3.2064 (+0.013) / Mamba-2 3.1644 (+0.055); WikiText PPL 49.95 vs 46.61 / 46.91.
  Any curriculum change (cu1, cu2) keeps a generation-quality readout: holdout by token class
  plus 3 shown samples.
