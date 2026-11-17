# EUREKA log: shared insights between the qllm2 sessions

Both sessions (v13_sempty "Qllm v13" = maturity, v14 "Qllm v14" = new design)
are building one novel O(1) architecture. Any insight that could change the
other's design goes here AND is sent at once by message, first line starting
`EUREKA:` (user rule, 2026-10-30). One entry per insight, newest first:
what we saw (with the run hash / log), what it really says, and what it
could mean for the other session. Read it before choosing a next step.

## 2026-11-17 05:35 IST: single-save told 8k mostly measures the body's no-protocol '.' prior, not the told-rule carry
**What we saw:** TOLD-MARGIN jsonl at 10 points (M11 … 42B; code b4f1bc9 / 41ba344 / 41ba3448, logs/v14_told_margin_*), paired per doc at 8k.
- The protocol's own effect C = mean(held − noprot) M_gate is steady at 0.31–0.54 nats (344k 0.87), SE 0.02–0.03 per save.
- The no-protocol gate B (the body's pull to end a number with '.') swings from −1.1 to −5.9.
- Spearman(told 8k, B) = 0.96; Spearman(told 8k, C) = 0.73.
- 42B: told 8k 0.09 with C 0.42 (41B 0.49), and B fell to −4.79.
**What it says:** the ±0.3 save-to-save swings in told 8k come mostly from a format prior that moves with each save. The carry the far block trains moves little.
**For Mind / Lead:** judge 3d by C (Δ ≥ 0.10 nats is about 3 SE), with B reported beside it. A told read that does not have to out-vote the '.' prior would be steadier (A32c split, 0.61–0.83 at 8k). EXPERIMENTS_V14 "TOLD-MARGIN re-read".

## 2026-11-12 12:23 IST: rule carry trained at 1.8-3.2k generalises to 8k (beyond the window): s8 @8k 0.19 -> 0.97
**What we saw:** SCI-1 386M, segment 2 of the long run (`sci1_long2_4a97ca7`, code `4a97ca7`; far block = b2s_far 1.2 % + proto_far 0.8 %
ramped in from M11 over 2B, `8b2679d` / `b502422`). s8 @8k (pre-registered headline, untrained): M11 0.186 (margin −1.1) -> 22.5B 0.979
(+8.5) / 26B 0.974 / 27B 0.964 (+7.9); 2k / 4k 0.99 (trained distance); needles t40 8k 0.89 -> 0.92-0.93. Logs: logs/learning_probes/
v13_sempty_sci1_{best_model,step_396734,step_411993}_4a97ca7_*.json, EXPERIMENTS_SEMPY "far block RESULT, part 1" (`53778f7`).
**What it says:** the O(1) notebook carries an induced rule ~2x past the longest trained gap and past the training window (no state
crosses windows in training): distance supervision, not more task data, was the missing piece (the four-way decay finding).
**For Library / Lead / the paper:** a within-model gain; the comparative claim needs the far block in the equal-data control for PAM,
T++ and Mamba-2 (queued on the 4090 after cx_m2). Told 8k (Library's FAR-TOLD) is the keep rule's second half.

## 2026-11-07 01:29 IST: an LR decay erases far rule carry first; the training data never taught that distance
**What we saw:** four SCI-1 tests (plain 1B decay, M9 <-> decay interpolation, rule-x2 decays from M10 and from M9; logs/s8reg/, EXPERIMENTS_SEMPY
"FINDING: the far rule margin is the decay's first casualty"): s8 at 2k / 4k falls with the LR every time (M9 0.77 -> 0.23 plain;
-> 0.46 with the rule docs doubled), while benchmarks and NLL improve. b2s spread prompts never reach 2k tokens (0 / 600).
**What it says:** the far carry is an untrained extrapolation; decays sharpen it away. **For Library / Lead:** read any held-rule cell on
a decayed checkpoint as this effect first; units that rely on far rules need data at that distance (b2s_far) before a release decay.

## 2026-11-06 15:31 IST: the slow notebook heads want slower clocks than the Chrono warp clamp allows
**What we saw:** clk3 on SCI-1 M9 (`7a9eecd`, logs/diag/v13_sempty_clk3_m9_7a9eecd.json): 3 of the 6 heads with the longest realized
retention (half-lives 4,204 / 746 / 742 tokens) sit at the warp clamp floor g = exp(−3); slowing the clock WITH the retention at eval
raises far rule-carry 2.6x more than retention alone (+0.087 vs +0.033 D_far at 4k+8k).
**What it says:** long memory in a delta notebook needs both a slow decay and a slow rotary clock; the clamp, not the data, caps the
slowest heads. **For Library / v14 / the ladder:** a body with a wider warp floor (−6) is a cheap rung; any unit that reads from slow
heads should expect phase drift beyond ~4k unless the clock slows with it.

## 2026-11-06 12:50 IST: the induced-rule carry (s8) falls at every LR / segment change; recall and benchmarks do not
**What we saw:** SCI-1 s8 D over 11 checks: 0.32 / 0.54 / 0.71 / 0.78 / 0.44 / 0.44 / 0.59 / 0.59 / 0.77 / 0.42 / 0.23. The drops
are at the cont1 start (76k), the cont4 start (152k) and the 1B decay from M9 (`8290f7d`: s8 0.770 -> 0.231 while HS 0.333 -> 0.343,
ARC-e 0.519 -> 0.567, needles 16k 0.93 -> 0.91). Log: logs/learning_probes/v13_sempty_sci1_latest_8290f7d_spread_s7000_t400.json.
**What it says:** rule induction across filler is the most LR-fragile skill the body has; a mix verdict read across a segment switch is
confounded (the M10 `worked` "fail" is withdrawn to unresolved). **UPDATE same day (s8reg A/B): the re-warm alone costs only -0.10
(M9 + 1.2k steps 0.667), and a short 0.1B decay at 103M RAISES s8 (0.358 -> 0.523): it is the SCI-1 1B decay itself, not LR changes in
general; the M10 `worked` fail stands again.** **For Library / Lead:** read rule-cell drops on fresh segments and
decays as regime effects first; judge mix changes only inside one segment (in-sampler ramps, no restart), which the chat ramp does.

## 2026-11-05 05:30 IST: equal-data control at the product recipe: PAM beats BOTH baselines on rules and beyond-length recall (Lead; ctl caa4da9, result f0a4695, B5 logs/ctl/v14_a28_ctl_pam_vs_tpp_mamba2_caa4da9.jsonl)
**What we saw:** 3 bodies x ~103M params x 1.0B tokens of the IDENTICAL SCI-1 cont2 mix (same code / seed / order). Induced rules s8 D: PAM 0.523 vs Mamba-2 0.149 vs T++ 0.074. Told prose rule held in state (twin read, 98 docs): 0.53 +- 0.05 vs 0.20 vs 0.12. Far-needle recall past SEQ 8k / 16k: 0.53 / 0.41 vs 0.18 / 0.15 vs 0.17 / 0.13. Inside the window (2k) T++ recalls better (0.92 vs 0.63). Holdout: Mamba-2 3.378 < PAM 3.427 < T++ 3.440 (B3 NOT met). HS / ARC-e / PIQA level. PAM trains ~2x slower per token at this size.
**What it says:** with the same rule documents, the rule-carrying and beyond-length clauses are architectural, not data: both baselines see the docs and still fail to carry rules across filler (adjacent examples @0 are close: 0.75 / 0.68 / 0.36). Mamba-2 remains the better pure LM at this size.
**Caveat added 06:50 IST (Library A28b):** the told-rule edge holds on the TRAINED wording only; on a held-out wording at 2k PAM 0.12 vs Mamba-2 0.06 vs T++ 0.05 (+0.06, bar +0.20 NOT met), and 0.00 for all at 8k. One seed per body. The induced-rule (s8) and beyond-length recall edges are unaffected.
**What it means for the lanes:** the thesis claim now has an equal-data control at the product recipe (third tier after 600M and 1.5B). Language parity with Mamba-2 is the open gap (Mind); told vs induced competition and reach past 8k are the next questions (Mind, Library).

## 2026-11-05

- **Mind + Library, correction (13:23 IST, 2026-11-05): told and induced rules do NOT consistently compete.** At cont3 +2B both rose together (s8
  0.589 → 0.770; told twin G1 2k 0.54 → 0.83, 8k 0.04 → 0.22). Across five milestone intervals the seesaw shows in 2, a joint rise in
  1, one-sided moves in 2. The 11-04 entry "told and induced rules compete for the same O(1) state" is withdrawn as a claim; the
  simpler reading is that both follow the rule-doc rehearsal with a ~1B-token lag. For both lanes: keep reporting them side by side,
  but do not design around competition (the per-use notebook idea loses its main motivation; the synaptic tag stays as a forgetting
  idea).

## 2026-11-04

- **Mind, correction (21:39 IST, 2026-11-04): the rule skill recovered after a ~1B-token lag.** s8 D 0.436 (+1B of restored b2s) → 0.588 (+2B),
  and reach reached its best (skill pack 8k 0.35, clk2 16k 0.22, far needles at 16k 0.70). The "erased" reading at +1B was
  premature: the state relearns at about the first-acquisition rate. For both lanes: judge a mix change only after ≥ 2 milestones
  (2B tokens); a 1B readout can show nothing yet.

- **Mind + Library: TOLD rules and INDUCED rules compete for the same O(1) state (15:01 IST, 2026-11-04).** Since the SCI-1 cont1 switch, s8
  (a rule induced from spread examples) fell 0.78 → 0.44 and stayed there, while the prose protocol held in state (Library's A26 G1
  with the twin read) ROSE 0.26 → 0.65 → 0.81 at 2k, with reach starting at 8k / 32k (0.20). The mix had moved weight from induction
  docs (b2s) to instruction docs (proto4k + protoc). EXPERIMENTS_SEMPY "SCI-1 cont2 milestone 6"; EXPERIMENTS_V14 (A26 c2a).
  **What it says:** "rule carrying" is not one skill; the notebooks are allocated between uses by the data, so a gain in one can be
  paid for by the other. **For both lanes:** report told and induced rules side by side at every milestone; a product mix sets the
  balance on purpose; a mechanism that gives each use its own protected capacity (per-use notebooks, the synaptic-tag idea) is
  the architectural answer to test.

- **Library (v14): the package's own read was hiding the rule held in state; read against a no-RULE twin and the prose
  protocol shows up.** A28 (be66f2d; `logs/v14_a28_twin_{g1,a26w}_be66f2d_20261104_0830.*`), SCI-1 step 76,295 (386M, 5.0B
  tokens), A26 UNIT, the same fetched row, 20 docs per cell. G1 at 2k: the A9 read (row vs no-row contrast) 0.00; plain 0.00;
  contrast against the SAME session read WITHOUT the protocol (+ the same row) **0.65** (the unit right on 0.65); no-protocol
  control 0.00. A26 wordings at 2k: 0.20 (unit 0.45). 8k / 32k: 0.00-0.05 under every read.
  **What it says:** "prose held = 0.00 at every milestone" was largely the read. A row-vs-no-row contrast cancels anything the
  state holds in both streams (the rule) and amplifies only the row (the bare value). The prose rule IS in the state at 2k;
  it fades by 8k. **For Mind / Lead:** (1) the "compile prose into state form" gap is smaller than thought at 2k: the gap is
  REACH of a prose rule, not its storage; (2) every A26 held number since M1 is re-read with the rule twin (pre-registered
  rule); (3) the package must pick its contrast twin per question part (rule twin, fact twin): a no-training fix, now a
  pre-registered unit; (4) any probe that uses the no-row contrast for a rule-in-state question undercounts the state.

- **Mind (v13): the O(1) mind's rule-at-distance skill is MAINTAINED by data, not acquired once.** SCI-1 cont1 (59d098d, 386M):
  after the b2s rule-doc share was cut ~1.7 % → 0.5 % of tokens, s8 D fell 0.782 → 0.435 in 1.0B tokens (2k 0.91 → 0.56, 4k 0.65 →
  0.31), while adjacent examples (@0, 0.96–0.98), recall and one-read reach (skill pack flat; clk2 8k 0.62 → 0.67) did not fall.
  (EXPERIMENTS_SEMPY "SCI-1 cont1 milestone 5"; logs/learning_probes/v13_sempty_sci1_step_076295_59d098d_*.)
  **What it says:** carrying a rule across filler in the state is a practised skill that decays within ~1B tokens without its
  docs at this size; storage / reading is robust, rule-carrying is not. **For Library / Lead:** any product mix must keep the
  rule-doc share (≥ ~1.2 %) for the life of the run; the s8 numbers in the write-up hold for the mix that trained them; a
  continual-learning or replay angle (a fixed small rule-doc replay share) is the safe default; check A26 / G1 against the same cut.

## 2026-11-02

- **Mind (v13) + Lead: the rules-in-state edge holds at a second, larger equal-data tier.** Matched 1.5B tokens, same mix, same params (PAM 16nb plain 103.2M, T++ 101.9M, Mamba-2 102.6M; Mind 9616861): rules at distance (s8 D) PAM 0.545 vs T++ 0.047 / Mamba-2 0.033 (600M: 0.768 vs 0.134 / 0.10, three PAM seeds). More data made every body WORSE at rules at distance, the baselines most (they also lose the rule at L = 0: 0.53 / 0.745 vs PAM 1.00), while PAM's reach with its query-time tools grew (skill pack 4k + contrast 0.70 -> 0.775, 8k 0.39 -> 0.47; baselines <= 0.105 / 0.29). Honest line: language at 1.5B PAM 3.3784 vs T++ 3.3593 / Mamba-2 3.3025 (bar 3.3225 missed by 0.056; the 16nb body starts 0.09 behind M2 at 600M); tool-record reading stays the package's job (0.30 vs 0.875 / 0.69). What it says: bar 3 (learning beyond recall, held in O(1) state) is replicated across two tiers at equal data; bar 2 (language) is a body-choice question, not a mechanism question. What it means for the other sessions: the next run is the M2 body at 1.5B (Lead call, 09:04 IST); Library's package v0 stays the reader; the write-up gets its scaling table.

## 2026-11-01

- **Mind (v13) + Lead: a rule held ONLY in an O(1) state is applied to a fact fetched at query time; Transformer++ and Mamba-2 at
  equal params and tokens are near zero (posted to ntfy by Lead).** R × K (`41681fc` verdict, `5526489` Mamba-2 rows, `ed4c84f`
  Library's package.apply reproduction; `logs/diag/v13_sempty_rxk_*`). 600M tokens each, same data. Explicit question (the
  planner substitutes the fetched value), free:

  | model | 2k | 8k |
  |---|---|---|
  | PAM b2p2 / cu2a (102.9M), plain | 0.55 / 0.48 | – |
  | PAM, query-time tools (2k contrast; 8k echo dilation s = 8 + contrast) | 0.61 / 0.62 (20-way 0.94 / 0.90) | 0.45 / 0.49 |
  | T++ b2t2 (101.9M), as it is | 0.01 | 0.07 |
  | Mamba-2 b2m2 (102.6M), as it is | 0.03 | 0.00 |

  - RAG + few-shot ceiling (pack re-inserted at the query): PAM 0.83–0.92, T++ 0.30, Mamba-2 0.44. Floor (no pack) ~0 for all.
  - The registered 2k bar (0.70) was missed by 0.08–0.09; the 8k bar (0.40) was met.
  - Mamba-2 mbs2b (aptitude recipe, no B2 docs) is 0.00 even at the ceiling, so it is uninformative.
  - Implicit ("K ->", take v from the row): ≤ 0.07 for EVERY model, even with the pack in view.
  - A record row between rule and query: b2p2 1.00 → 0.91, T++ 0.69 → 0.17, cu2a 1.00 → 0.02 (format capture at document start).

  **For Library:** the planner must substitute fetched values into the mind's trained query line, and never ask the mind to compose
  key → row → rule. Always run the contrast's no-pack twin (it cancels the copy pull of v). For one read to 8k, use dilation
  s = 8 + contrast; the 2k refresh + contrast default stands. Put record rows after some context, or render them in the rule's
  format.

- **Mind (v13): a bounded local-attention window (128, three blocks, +4 %) makes the O(1) mind infer rules from SPREAD examples
  far better, beyond the window's own reach; it does NOT help read fetched records.** w1s + w1r (`508aaff`, 102.9M vs 107.0M,
  220M tokens, seeds 43 + 44):
  - s8 at 4k (last example 510 tokens back; three stacked windows reach 381): 0.435 → 0.617 / 0.416 → 0.771;
  - letter shift at 4k: 0.17 → 0.45 / 0.19 → 0.74;
  - holdout +0.006 / +0.002; the decays are unchanged;
  - fetched tool-row read +0.03 pooled.

  The gain is on rule INFERENCE, not on holding a contiguous pack (skill pack +0.04 to +0.13). Known hybrid (Samba / Griffin), so
  no mechanism claim. **For Library:** the reader stays in the package (A19). If W1 enters the body at 600M, the mind's own rule
  inference from session text improves, and the planner may need fewer refreshes. Re-measure A16 then.

- **Mind (v13): skills are FORMAT-bound at 103M; the package should speak the mind's trained formats.** cu2a (+2 % fld / vfy):
  - trained formats: one field + stop 0.80 with no run-on, verify 0.83–0.99, at +0.0175 LM;
  - held-out question wordings: 0.26 (fld) / 0.32–0.46 (verify); a held-out ROW layout: ~0.00;
  - in the bench's "Query: K refers to" format, cu2a copies a wrong row's value 0.78 (cu1, trained in that format, 0.01).

  Same boundary as cu1's stated-rule table and natq: the mind generalises values and keys, not surface formats.
  **For Library:** render fetched rows and the question in a trained template before the mind reads them (the package
  owns both). That is the cheap fix; generalising formats would need many formats in training.

- **Mind (v13): the skill's trace lives in the ECHO memory; slow only its clock and one read reaches 8k at
  almost no LM cost.** clk2 (b2p2, fresh seeds 8500–8549): echo retention r → r^(1/4) between a package read and its query,
  plus the query-time contrast:
  - free 0.46 / 20-way 0.64 at 8k (4× the training length);
  - **text cost +0.013 nats** (all-memory dilation: +0.18; notebooks only: +0.16 with little reach).

  Baselines at the same 8k gap, as they are: T++ 0.195, Mamba-2 0.015 (20-way). **What it says:** the reach wall is the
  echo memory's clock, and the package can slow it on purpose.

  **Replicated (clk2r, `4b51eef`; ntfy posted by Lead).** One read, 8k later, echo dilation + contrast, free (20-way):
  | checkpoint | s = 4 | s = 8 | prose cost s = 4 / s = 8 | 4k at s = 4 |
  |---|---|---|---|---|
  | b2p2 | 0.46 (0.64) | 0.49 (0.74) | +0.013 / +0.016 | 0.62 (0.88) |
  | cu2a | 0.47 (0.78) | 0.58 (0.89) | +0.016 / +0.021 | 0.79 (0.95) |
  | b2p1 | 0.34 (0.72), FAILS 0.40 | 0.43 (0.81) | +0.012 / +0.015 | 0.83 (0.97) |

  **s = 8 is the passing setting** (three minds, three seed sets); s = 4 failed on b2p1.
  **Deployment rule:** dilate the echo memory only, only between a package-paired read and its query, never on free
  text; the contrast only on the query's answer tokens (cost +1.15 MB and 2× compute there).
  **Package reality (Library A16):** a refresh every 8k with dilation (s = 4) gives 0.42 at 32k vs a 2k plain refresh 0.63;
  the package default stays 2k + contrast (0.70). This is a mind-level reach result: ~4× the trained length at about half
  the 2k accuracy.

- **Library (06:00 IST): the mind's own SURPRISE picks the keys of a question, and its own frozen states find
  the fact better than a trained sentence encoder, even for synonyms.** Seen: A13 (EXPERIMENTS_V14,
  pre-registered 05:12, `f917720`, `logs/v14_a13_manykey_b2p2_32768_b34636e_20261101_0514.jsonl`; b2p2, 32k,
  seed 33000, 40 docs per cell). Take the 2 question tokens the mind finds most surprising (" merchant",
  " cunning", " Texas"), match them by max cosine against each sentence's tokens in the block-8 states:
  synonym facts top-1 **0.575** / top-5 0.80 (MiniLM-L6 sentence encoder 0.35 / 0.65 on the same docs; my
  trained L1 head 0.40 in A5; the key's last token alone 0.125); natural questions top-1 **0.775** / top-5
  0.975 (BM25 0.525 on A12's seed). With the warm + contrast read the package answers synonyms 0.55 and
  natural questions 0.475 (ceiling with the true row 0.625). Says: text cannot tell a question's key from
  its frame (A12), the mind can (the frame is predictable to it). It reverses my 05:05 pick (a frozen
  sentence encoder for synonyms): on these docs the mind's own states with its own salience are the
  better address. Caveats: one seed; MiniLM pooled per sentence vs token-level late interaction. **For
  Mind:** surprise is a free, untrained salience signal; a curriculum that sharpens it is cheap to test.
  **For Lead / r3p's readout:** add F2 as an address arm. Open: abstain where words differ (score AUROC
  0.66-0.77, key-swap 0.41-0.57) needs the trained VERIFY.

- **Mind (v13): slow the clock after a read, and the trace survives to 8k.** clk1 (b2p2, fresh seeds,
  `4faa7ae`): every retention r → r^(1/4) between a skill read and the query, plus the query-time contrast
  (a = 4). One read then reaches:
  - **free 0.52 / 20-way 0.76 at 8k** (4× the training length); dilation alone 0.08, contrast alone 0.045;
  - 0.58 / 0.88 at 4k.

  **What it says:** the reach wall is set by the clock (decay over the text in between), not by capacity.
  Dilation keeps the trace and the contrast makes it audible. **Cost:** +0.18 nats on the text read during
  the dilation (all heads keep old context), so it FAILED its +0.05 LM bar as registered. Targeted dilation
  (clk2) is next. **For Library:** a refresh every ~8k with dilation + contrast could replace every ~2k plain,
  if clk2 cuts the text cost.

- **Mind (v13): cu1's abstain skill transfers to NATURAL keys.** cu1 was trained with synthetic-key
  fsq only. In the short re-read format (one row + the question), with English-phrase keys:
  - an unrelated key's row: cu1 says " unknown" outright 0.86 and the wrong value 0.01;
  - b2p2 says the wrong value 0.59 (Library's package rows: ~57 %);
  - right rows: cu1 0.91 free (8 % false abstain), b2p2 0.94.

  **What it says:** the near-miss copying Library found is fixed by a small amount of abstain training,
  and it generalises across key style. **For Library:** score your strict package rows on cu1
  (`checkpoints_v13_sempty/cu1_main_2efb422/best_model.pt`) before cu2 adds natural-key VERIFY; cu2 may
  only need the prose-row case (cu1 free 0.31) and the multi-token answers. Fresh seed needed for any claim.

- **Library (03:23 IST): a fetched fact read on the mind's own state WITH a no-row contrast is the best
  answer path we have (0.90-1.00 at 2k-32k), and the "outvoted" effect is NOT a length effect.** Seen:
  A6c (EXPERIMENTS_V14, pre-registered 01:53, `06f3109`, `logs/v14_a6c_b2p2_dec965d_20261101_0212.jsonl`;
  b2p2, seed 28000, 360 docs, fact row by construction, sentence scorer). Warm read (state after the
  doc + row + question) vs cold (row + question, fresh state) vs warm-contrast l_warm + 4(l_warm −
  l_norow) on the answer tokens only:
  kv syn / kv nat / para nat at 2k: cold 0.875 / 0.80 / 0.775, warm 0.675 / 0.275 / 0.125, contrast
  **0.95 / 0.975 / 1.00**; at 32k: cold 0.925 / 0.775 / 0.70, warm 0.45 / 0.125 / 0.125, contrast
  **0.90 / 0.95 / 0.975**. **Correction to my 01:52 entry (bd34118):** warm fails at the trained
  length 2k too, so the long state outvotes the newest sentence at ANY length, not only past 2,048.
  The contrast fixes the cold read's wording misses (" ave." → " indigo.") and closes most of S2's
  natural-key gap. Caveat: 20-38 % of contrast answers run on after the value; gate it to the answer
  tokens and stop at the value (Mind's cd3: on free text it collapses). **For Mind:** your cd1-cd3
  contrast is the same mechanism and works for single facts as well as skills; the body ignores fresh
  evidence at every length, and the contrast reads what the evidence ADDED. **For the package:**
  hand a fetched row warm with a no-row contrast stream (one prefill, two short O(1) continuations);
  confirmation on the package cells next (fresh seed).

- **Mind (v13): a rule taught in WORDS is learned as a lookup table; a rule shown by EXAMPLES
  generalises.** cu1 trained " Rule: shift each letter forward by k." docs (srd, 10 trained k):
  - stated rule, 0 shots: trained k 0.97 (b2p2 0.07), held-out k (5, 9) **0.00**; told "5" or "9", it
    applies a trained shift (10, 4, 1);
  - one demonstration: 0.97 on held-out k in both models; B2 `_new` from examples 1.000.

  (`3c7a7fa`, `v13_sempty/tmp/stated_rule_probe.py`.) **What it says:** at 103M the mind binds a
  stated number to a memorised behaviour; it does not compose "shift by <new number>". Rule
  inference beyond recall comes from examples, not from instructions with unseen values. **For
  Library:** a skill register written from a stated rule would store the table, not the rule; register
  from demonstrations (or one demo plus the words). cu1 also met abstain (1.00 on absent keys) and
  two-hop (0.57 ranked, 0.20 free) at a +0.045 LM cost; srd is removed from the code.

- **Library (01:52 IST): at 32k the mind ignores a fact sentence placed right before the question;
  the same 40 tokens read from a fresh state work.** Seen: A6 (EXPERIMENTS_V14, pre-registered 01:30,
  `92f4d85`, `logs/v14_a6_b2p2_32768_238c62e_20261101_0135.jsonl`; b2p2, seed 26000, 320 docs, parity
  2e-3). The fact row + question read COLD (fresh state) vs WARM (after the 32k document's carried
  state): sentence-level free kv 0.925 / 0.825 cold vs 0.525 / 0.15 warm (synthetic / natural), para
  0.725 / 0.55 vs 0.45 / 0.025; warm is worse on all 8 cells (mean −0.325). Warm answers on natural keys
  are nearly the state-only answers (" the place where the king was killed" with or without the row).
  Says: past its trained length (2,048; 32k = 16×) the long state OUTVOTES the newest sentence: the
  mind does not listen late. Same phenomenon as Mind's 3bc29c7 ("outvoted, not lost") seen from the
  fact side. **For Mind:** a body-level recency / working-memory gap beyond the trained length (a
  length sweep A6c is pre-registered next: does warm = cold at 2k?). **For the package / Lead:** read
  fetched rows in a fresh scratch state (the current cold re-read is right); a stream-inserted refresh
  (N11-R) is a warm read, so its free numbers may fall at 32k; contrast decoding may rescue warm.

- **Mind (v13): past the one-read reach the skill is outvoted, not lost.** cd1 (b2p2, skill-pack cell,
  eval only, `3bc29c7`): decoding with l_pack + 4·(l_pack − l_nopack), i.e. the same stream with
  minus without the skill read, lifts one-read reach at 4k:
  - free 0.17 → 0.59;
  - 20-way 0.355 → 0.815, above the 0.75 that 50M tokens of 8k training (lg8r) reached only to 0.465.

  At 2k, free stays ~0.70 (20-way 0.94), so the pre-registered bar FAILED there. **What it says:**
  the reach wall is partly a volume problem at decode time; the state still holds the rule.
  **For Library:** your refresh interval R (~2k, N11-R) may stretch if the package decodes with the
  contrast (2 streams, 2 × 1.15 MB, still O(1)); worth testing R with it. **For Lead:** a package
  decode option. Prior work: CAD (Shi et al. 2023), contrastive decoding (Li et al. 2022); no method
  claim. **Confirmed on fresh seeds (cd2, seeds 8200–8299, a fixed):** 4k 20-way 0.355 → 0.78 (a = 4) /
  0.855 (a = 8), free 0.165 → 0.575 / 0.665. By 8k the skill is really gone (≤ 0.18; 16k chance), so the
  contrast roughly doubles the one-read reach (~2k → ~4k), at the cost of a 2nd O(1) state and 2× compute.

- **Mind (v13): the mind's own free generation already tells a wrong row from the right one;
  the 21-way list hides it.** Short re-read format (one row + the question alone, Library's
  strict-run layout), b2p2, 200 docs (`scripts/run_cur_probe.py` verify cell, `b065405`):
  - right row: says the value 0.96 (free) / 1.00 (21-way);
  - unrelated key's row: says that row's value **0.14 free** but 0.70 when forced to pick among 21;
    its greedy token is mostly " a";
  - prose row: 0.00 / 0.065. It never says " unknown" (untrained).

  **What it says:** the "copies the nearest value" failure is largely the forced choice. Left free,
  the mind mostly does not commit to a wrong row's value. **For Library:** a "not stored" signal
  may be simply "the greedy token is not a candidate value" (right rows 96 % yes, wrong-key 14 %).
  That is the channel A7 counts, and cu2's VERIFY training would turn it into an explicit
  " unknown". Short format and bench vocabulary only.
  **Limit, measured by Library on the package's own rows (seeds 22000 / 23000):** the rule "answer
  iff the greedy token is a candidate value" abstains on never-stored questions 0.725 / 0.85 with
  synthetic keys but only 0.425 / 0.425 with NATURAL (English-phrase) keys. With a nonsense key the
  mind sees the row is not the question's; with an English key it copies the nearest value (~57 %).
  So VERIFY on natural keys is what cu2 must teach.

- **Library (01:10 IST): the mind copies the nearest value; it never checks that the row's key is the
  question's key.** Seen: Lead's strict run (`logs/v16_pkg_strict_b2p2_32768_d2aacdf_20261101_0035.jsonl`,
  b2p2, 32k, seed 22000, 320 rows), offline test A3 (EXPERIMENTS_V14, pre-registered 01:00, fit
  `17a12bd`). Reading a fetched row raises the log-prob of the row's value by a median of 3.2 nats
  when the row belongs to a DIFFERENT key (never-stored question) vs 4.8 when it is the right row;
  so no gain threshold separates them (best in-sample rule +0.06, bar +0.10). Says: abstain cannot
  come from the mind's read as trained; on templated cells text decides "is it stored" (A4 met,
  `fa95777`), and when the words differ nothing can decide yet. **For Mind (curriculum):** a VERIFY
  skill, i.e. near-miss rows (another key's fact) in context with an explicit "unknown" target,
  would make abstain come out of the mind's own generation and hold under rewording. Prior work:
  SQuAD 2.0 unanswerables, R-Tuning, Self-RAG relevance tokens; difference: the O(1) mind as the
  verifier of a fetched row at question time.

- **Lead: MLB-1 package table vs T++ and Mamba-2 as they are: the package meets R1 (recall), R2
  (rule edge, s8) and R4 (memory) with R3's guard, at 32k and 128k.** Same held-out prose, same
  docs and seeds for every body (v16/bench.py `7f49c7b`, A4 seeding; seeds 18000 at 32k with 40
  docs per cell, 18500 at 128k with 24 docs per cell, synthetic and natural keys; logs
  `logs/v16_pkg_table_{b2p2,c1ctlb}_{32768,131072}_7f49c7b_20261031_2337.jsonl` (commit `abf5b69`,
  1,600 rows, 0 errors) and `logs/v16_pkg_table_{mamba2,transformer}_32768_49b9332_20261031_2319.jsonl`
  (commit `0413e29`); PAM package = frozen mind (b2p2 `38e3c29` M2 body or c1ctlb `7ae6f1f` M1 body,
  ~103M params, 600M tokens) + Library's session fetch (`v14.package_fetch:key_fn`, L1 burst key at
  block 8, shipped mean, exact organ + per-token address, PACKAGE_T = -1) + one re-read of the fetched
  span; baselines = tfc1b `b788731` (T++, 102M) and mbs2b `685de4e` (Mamba-2, 103M) reading the same
  stream with their own state, no library (the comparison rule).
  - **R1 recall (package top-1, 20-way, chance 0.05):** needle 1.00 / 1.00, 8-needle 1.00 / 1.00
    at 32k / 128k for both bodies; latest-wins 1.00 and two-hop (chain + re-read) 1.00 at 32k
    and at 128k (latest-wins 1.00 / 1.00, two-hop 1.00 / 1.00 for b2p2 / c1ctlb; fill rows
    `logs/v16_pkg_table_{b2p2,c1ctlb}_131072_fill_d2aacdf_20261101_0036.jsonl`, 96 rows each, 0 errors). State only at 32k: T++ 0.04 / 0.04 / 0.06 / 0.04
    (needle / 8-needle / latest-wins / two-hop), Mamba-2 0.14 / 0.06 / 0.18 / 0.09; the PAM minds
    alone 0.08-0.19 (the same ~chance level: holding is not where any body wins at this size).
  - **Negative cells (abstain):** with the fetch threshold disabled (PACKAGE_T = -1, as run) the
    package answers 0.00-0.10 wrong. Offline, post-hoc (NOT pre-registered): every positive-cell
    fetch has sim 1.00 (exact organ hit) and the highest negative-cell sim is 0.962, so a threshold
    of 0.97 gives abstain 1.00 at 32k and 128k (256 negative docs) with needle / 8-needle /
    latest-wins unchanged at 1.00; the cost is paraphrase 0.46-0.66 -> 0.29 and tool 0.31-0.46 ->
    0.14, i.e. the abstain is "no exact hit", and the vector path does not yet separate a paraphrase
    from a miss. Reported, not claimed. Library's reading (agreed): since every exact hit has sim 1.00 by
    construction, the sweep is really the rule "abstain iff the organ misses"; Library pre-registered
    it as PACKAGE strict mode (EXPERIMENTS_V14, 00:34 IST, `64ea182`: fresh seed 22000, 32k, b2p2,
    40 docs, kv / para / tool / neg; bars neg abstain >= 0.90 with kv unchanged at 1.00; para / tool
    strict vs lenient = the trade-off), run by Lead when the 4090 frees (PACKAGE_T = 0.99).
  - **Below bar, reported:** paraphrase (package 0.46 / 0.54 b2p2, 0.66 / 0.65 c1ctlb at 32k / 128k;
    exact organ alone 0.26-0.35; T++ 0.05, Mamba-2 0.01 state only) and tool (package 0.34 / 0.46
    b2p2, 0.44 / 0.31 c1ctlb; T++ 0.01, Mamba-2 0.05). These are the open items of the session
    library (synonyms, rewording: Library's lane).
  - **R4 memory at 128k (same shape, 103M, 16 layers):** mind state 1.15 MB (O(1), fixed) + library
    rows 0.26 MB (272 B per event) + raw text 0.26 MB + organ (~1.5 MB at Library's 11.4 MB per 1M
    tokens) ~ 3.2 MB total vs the T++ KV cache 4,932 MB -> ratio ~ 1/1,500 (R4 bar 1/100; the
    stored 32k T++ KV cache measured 1,057 MB). Mamba-2's state 13.9 MB, T++ KV at 32k 1,233 MB.
    All numbers from the `mem` field of the rows.
  - **R2 (rule edge):** s8, already in this log (D 0.645 / 0.691 / 0.741 vs T++ 0.134, Mamba-2
    0.116 / 0.098; +0.51 worst pair, bar +0.10). **R3 guard:** the b2 pair at equal data and recipe
    (600M, `38e3c29`): PAM holdout 3.4775-3.4800 vs Mamba-2 3.4754-3.4768 (+0.005, bar +0.02);
    c1ctlb's older recipe is +0.035 vs mbs2b, so the headline body is b2p2. Loss by token class:
    attr3b split, Mind's table.
  - **Honesty line (§1 / R5):** T++ and Mamba-2 ran AS THEY ARE (no library, per the comparison
    rule); a T++ given the same fetch was NOT run, so 1.00 vs ~0.05 is a package-vs-body gap, not a
    body-vs-body win. Cost: PAM 102.9M params / 600M tokens vs T++ 101.9M and Mamba-2 102.6M at the
    same tokens. the recall win (R1) belongs to the PACKAGE, and inside it the exact-key 1.00 is the exact
    organ's (plumbing, model-free, identical for every body); the mind's own key earns its place on
    paraphrase in the session (0.46-0.66 vs organ alone 0.26-0.35) and its re-read does two-hop; the MIND's own claim
    is R2 (rules at distance) and R4 (O(1) state). Closest prior work: kNN-LM / RETRO (retrieval
    into a transformer with its KV cache), EM-LLM (episodic memory in a transformer), Titans (memory
    learned at test time inside the model); difference: the reader is an O(1) recurrent mind, the
    library is outside the layers, keyed by the mind's own stream state (session) or text (shared),
    and the package's memory is flat in context length.
  - **What it says / what it changes:** clause 2 of the thesis ("holds beyond the training
    context") is demonstrated at the package level with a frozen 103M mind trained at 2k: the
    package's recall at 128k is 1.00 where the same mind alone is at chance and T++ cannot read
    past its window. The open bars are the library's: abstain by a pre-set threshold, paraphrase,
    tool. *Caveats:* 20-way candidate scoring (not free generation); synthetic needles with unique
    keys favour the exact organ (natural keys are included and also 1.00 on exact cells); single
    package configuration; the two 128k cells for latest-wins and two-hop were run 25 minutes after the
    first table (same seed 18500, code d2aacdf = 7f49c7b + the free fields).

- **Mind (v13): at 3B the body alone leads both baselines on recall, in and beyond the training
  length, while LM quality stays just behind.** `r3cb_main_38e3c29` (16 notebooks + echo, LR 1.2e-3,
  two stages, 103.2M params, 3B tokens) vs T++ / Mamba-2 at the same data:
  - recall 0.792 vs 0.539 / 0.613; past 2,048 tokens 0.661 vs 0.153 / 0.403;
  - holdout 3.2191 vs 3.2064 / 3.1644 (+0.013 / +0.055); WikiText 49.95 vs 46.61 / 46.91;
  - late slope equal (−0.080 vs −0.081 / −0.079).

  **What it says:** the "PAM learns fast, then plateaus" story was the old body / LR. This body
  keeps pace at 3B and holds what is in context far better, with ~3.9 MB of state. **But**
  (A7) a plain-language recall question in a new format fails for both r3c and b2p2. The
  recall number is format-bound, so "expression" (free generation, holdout by class) is the
  open line. **For Library / Lead:** the 3B best-package run should start from this body +
  LR, add b2 / lab / curriculum with ANSWER_IN_WINDOW, and report free generation.

- **Mind (v13): the packer cuts answers away from their context.** Documents are packed and cut
  at window edges; an answer-weighted (×100) token whose document began in an earlier window
  is trained without the facts or demonstrations it needs, which only teaches guessing from
  the prior. Measured (`v13_sempty/tmp/orphan_rate.py`, real source docs; `824a988` adds
  `ANSWER_IN_WINDOW=1`, which drops those tokens to weight 1.0):
  - SEQ 2048: b2 3.3 %, lab 4.1 %, recall 15.9 %, srd 22.0 %, fsq 6.1 %, hop 7.5 %;
  - SEQ 8192: b2gap (G ≤ 7.5k) 44 %, recall 4.8 %, srd 5.0 %;
  - lg32 (32k, G ≤ 31k) was ~45 %.

  **What it says:** the long-gap recipe (LAW-1 lg32) trained nearly half of its rule answers
  toward guessing, so its "ceiling" reading is withdrawn pending lg8r (`lg8r_main_f2b7d8c`,
  running, both fixes on). b2p2's short B2 docs were barely hit, so its 2k reach stands.
  Recall at 2k loses one answer in six in every arm (T++, Mamba-2 too; same mix), so
  comparisons stay fair but every model learns from noisier recall targets. **For others:**
  turn the flag on in any new recipe with long answer-weighted docs (cu1's `srd` 22 % is the
  worst at 2k; cu1 is already running without it, so read its R6a with that in mind).

## 2026-10-31

- **Library (v14): the mind's state is a STREAM ADDRESS, not an embedder.** Inside one document
  the mind's own state finds a stored fact almost perfectly; across documents it does not, with or
  without training. In-document (v16 harness, c1ctlb, centred block-8 state, 16 docs per cell):
  per-token address C2 1.00 at 32k and 128k; the trained burst key (L1) 0.94 / 1.00 on natural keys
  at 32k / 128k. Cross-document (`v14/pack_check.py`, seed 19000, library and session are different
  texts), top-1 of the right entry at 400 entries (4.1k rows): L1 key exact 0.69 / 0.69, paraphrase
  0.33 / 0.30 (synthetic / natural); the same head trained on cross-document pairs (L3) 0.29 / 0.66
  and 0.20 / 0.37; the exact 4-gram organ 1.00 / 1.00 and 0.50 / 0.35. At 4,000 entries (43k rows)
  L1 falls to 0.47 / 0.42 exact, the organ stays 0.87 / 1.00. The state encodes WHERE something sits
  in this stream, not a context-free identity, and cross-document contrastive training did not
  change that. Closest prior work: dense retrievers (DPR, Contriever) learn a separate encoder with
  contrastive training on separate texts; our key is the reader's own state, which works only within
  its own stream. *What it changes (Lead approved):* the SESSION library (the conversation, tool
  results: the same stream as the question) is fetched by mind-state keys, which is what the v16
  harness measures (the package table is valid for it, L1 head); the SHARED library (separate texts,
  community packs) is fetched by TEXT: the exact organ with whole-key matching plus a normalised
  lexical match for paraphrase, plumbing, cited, no claim, with an off-the-shelf text embedder as a
  named baseline row. Commits `5a29519` (in-doc), `e1f37bd` (cross-doc sweep); EXPERIMENTS_V14 "L3
  VERDICT".


- **Mind: bar 3 met in its DISTANCE form, replicated (s8; ntfy sent 22:06 IST).** On a never-used
  probe sample (seed 7000), D (held-out rule values, examples spread across 2-4k tokens of unrelated
  text) = PAM 0.645 / 0.691 / 0.741 (3 seeds) vs T++ 0.134 / 0.134 and Mamba-2 0.116 / 0.098 (2 seeds
  each); primary +0.607, all pairs +0.511 (bar +0.10). ~102M params, 600M tokens each, code
  `38e3c29`; holdout at parity with Mamba-2. Wording: spread cell, gaps ~250-500 tokens; one-read
  reach is ~2k (the stated limit); trained formats with held-out values.

- **Mind: long-gap training at 32k SHRANK the one-read reach (LAW-1 first arm).** lg32 (b2p2 fine-tuned
  150M tokens at SEQ 32768, rule docs with gaps ~U(0, 31k)): reach at 2k / 8k / 32k = 0.31 / 0.06 / 0.04 vs
  b2p2's 0.86 / 0.10 / 0.04; margin at L = 0 +18 → +10; guards met. Hypothesis (untested): vanishing
  signal. Past the ~1-3k echo half-life the gradient from a far query to the pack is ~0, so training
  cannot teach longer retention, and the unsolvable up-weighted answers teach the trained-value prior.
  *For Library:* the strongest case yet for a non-decaying skill register written once at induction.
  lg8 (gaps ≤ 7.5k) and lg2 (control) decide the law.

- **Mind: replicated skill-pack result (ntfy sent 14:16 IST).** A rule read once is carried across
  2k tokens of unrelated text by the O(1) mind: reach 0.86 / 0.86 (b2p1, b2p2) vs T++ 0.03 and
  Mamba-2 0.06 (as they are); ~103M params, 600M tokens, state 1.15 MB vs Mamba-2 13.4 MB and
  T++ KV 67 MB at 2k; 100 docs per cell, probe `53d0e52`, seeds 8000-8099. All three fade by
  8k. LAW-1 (pre-registered) tests whether the reach tracks the training length.

- **Mind: after ONE read, PAM carries an induced rule ~2k tokens (≈ its training length), not
  8k; the spread cell overstated the reach.** Skill-pack cell (MLB-1 R7, b2p2 vs b2t, 100 docs):
  8 demos read once at t0, then L filler, then the query: PAM affine 0.98 / 0.14 / 0.07 at 2k /
  8k / 32k (shift 0.74 / 0.05 / 0.01); T++ 0.06 / 0.39 / 0.08 with the pack in its window.
  In the spread cell the last example is only L/8 from the query, so "holds to 8k" was
  carrying across ~1k gaps, re-exposed. The edge over T++ / Mamba-2 at ≤ 2k gaps is real;
  the single-exposure reach is ~2k, and Mind's own prediction (lost between 8k and 32k) was
  too optimistic. *For Library:* the structural skill register is now clearly needed
  for reach > 2k (unless long-gap training at 8k-32k teaches it). The register's bar is the
  skillpack cell at 8k / 32k against today's 0.14 / 0.07.

- **Lead: a stated rule does not switch a skill on; demonstrations do, even after 32k of
  filler.** `v16/bench.py --task instr` on b2p2 (`b2p2_main_38e3c29`, 103M, 600M tokens, HAS
  the letter-shift skill from rule docs; held-out shifts 5/9; trained "Letter rule." format;
  8 docs per cell; `logs/v16_bench_instr_b2p2_shots{0,2}_d319e8d_*.jsonl`). Rule stated once
  in words at the start ("Rule: shift each letter forward by 5."), L filler, then the query
  line: 0-shot = **0.000 at 2k / 8k / 32k**, and still 0.000 with the rule sentence re-read
  right in front of the query (O_reread). With 2 demonstration lines before the query:
  0.625 / 1.000 / 0.750 (letter-candidate scoring identical), 1.000 with the oracle re-read.
  *What it says:* the mind's rule register is written by examples, not by words; a library
  that fetches an instruction is useless to this body unless the instruction comes with (or
  is turned into) demonstrations. *For Mind:* the curriculum item "stated rule +
  demonstrations" docs is confirmed necessary; judge it on this cell (0-shot >= 0.75 at 32k).
  *For Library:* a fetched skill pack must contain demonstrations; and the planner's
  re-read puts them in front of the query, where they work even after 32k.

- **v14 (step back after W0 and the pivot): the surprise gate marks event BOUNDARIES, so a
  library row is the event between two bursts keyed by a learned pooling; and the library
  should be TEXT FIRST, with keys derived locally by the mind.** W0 (`232748e`, frozen
  C1-ctl, tau 11 = 0.6 rows per 16 tokens): needle caught 0.950 synthetic / 0.928 natural, but
  the firing token is the first word of the inserted sentence (" Definition", above 11 nats in
  84-86 %), the key in about 25 %. That is how event segmentation works (prediction error
  marks the boundary, the hippocampus binds the episode that just ended). *For v13 and the
  organiser:* (1) the harness's C5 gate rows should be judged as bursts (the span after or
  between firings), not as single tokens; (2) the shared library format can be plain text +
  provenance, and each model builds its own keys by one forward pass at install time, so the
  272 B/row index is a local cache and community libraries never go stale across model
  versions; MLB-1 should count install cost (one pass of the mind) next to the MB column;
  (3) selection after a fetch can be done by the mind itself: re-read each of the top-m spans
  alone and take the one with the highest answer margin (planner v0, no training); the
  "re-read with 4 distractors" cell tests it. Design: EXPERIMENTS_V14 "L" amendment A4.


- **v13: the label gap was a data-format gap. With label-format docs (different words) in
  training, PAM's label_exact is 0.995 and label_category 0.873 vs T++ 0.608 / 0.987 on the same
  data; cost +0.004 holdout.** `lbp` vs `lbt` (`c4173be`, 600M, ~102M). Short-range binding is
  fine when the format is trained; across distance both decay to chance by 2048. *For v14:*
  the binding gap your units targeted at short range is closed by data; judge N12 on the
  distance part and on fetch. **Direction (user): "O(1) mind + growing library."** The O(1)
  body stays and is the mind; what is dropped is the demand that the state also be the
  library (session library + shared library hold verbatim knowledge). Benchmark first:
  needle / multi-needle / two-hop / rule-at-distance at 32k-1M with the library, vs T++ with
  full context, MB in the same table. The rule-learning edge is PAM's and stays the headline.

- **v13 (user direction): the complete package = an intelligence mind + a learned fetch.**
  User: train the model as an intelligence machine, not a recall machine, and give it a
  learned way to fetch from the user context and tool results (RAG-like). It fits today's
  numbers: copy is over half the 3B T++ gap; a model-free exact organ recalls 1.0 to 128k;
  PAM's pre-logit vector at the key token already retrieves the right sentence top-1
  0.63-0.88 zero-shot; and re-reading the fetched spans gives two-hop 1.0 (Communicator,
  a1baefc). *For v14:* the learned fetch (when to fetch, the query = state at the salient
  token, re-read the span) is the mechanism to design. N7 / N12 are its first organs. The
  state can then spend its capacity on rules (its edge) instead of verbatim items. v13 builds
  the probe suite (fetch at 2k-128k with MB per number) and the 600M / 3B checks.

- **Organiser (CPU, frozen checkpoints, no training): the O(1) state cannot hold a
  needle past ~8k tokens, but its own per-token vector at the needle's key word is a
  usable address up to 128k, and a model-free n-gram organ gives exact recall and
  two-hop at every length.** `scripts/session_index_probe.py` (`f96c0e0` / `b554c67`),
  logs `logs/organiser_index_probe_*_{f96c0e0,b554c67}_*.jsonl`. Frozen C1-ctl
  (`c1ctlb_main_7ae6f1f`, 103M params, 600M tokens) and Transformer++
  (`tfc1b_main_b788731`, 102M) over held-out prose with "Definition: KEY refers to
  VALUE." needles at random positions, query at the end, 20-noun candidate scoring
  (chance 0.05), 8 docs per cell.
  - **State only (A):** T++ 0.75 / 0.25 / 0.0 at 2k / 8k / 32k (dies past its window).
    PAM 1 needle 0.625 / 0.25 / 0.0 / 0.0 at 2k / 8k / 32k / 128k; 8 needles 0.25 /
    0.125 / 0.0 / 0.0 (= picks among the needles, not the value). Holding is not where
    either body wins at this size.
  - **Exact n-gram organ (B, last 4 tokens -> what followed, latest write wins):** 1.0
    at every length and task, including "latest definition wins" and two-hop by two
    chained reads (1.0 at 2k / 8k / 32k); the frozen PAM re-reading the two retrieved
    spans + the query also answers two-hop 1.0, where its state alone cannot even
    produce hop 1 (0.0).
  - **Snapshot index keyed by the model's own vector (C2):** query = the pre-logit
    vector at the KEY token of the question, matched against every document position;
    the right sentence is top-1 in 0.75 / 0.875 / 0.625 / 0.75 (1 needle, 2k..128k)
    and 0.625 / 0.625 / 0.25 / 0.375 (8 needles); when found, the re-read answers.
    Roughly flat with length: the frozen state's representation of a nonsense key is
    a usable content address among 131k positions, without any training.
  - **Boundary snapshots fail:** the vector at the sentence's '.' (C3) and the mean
    over the sentence (C4) are ~0 top-1 at every length. A frozen state at a unit
    boundary does not carry the unit's key: "write with hindsight at the boundary"
    has to be trained, and the natural key is the state at the SALIENT token (what
    N12 surprise writes and N9s salient phase both reach for).
  - **Bytes at 128k:** raw text 262 KB; all-position snapshot index 308 MB (fp32,
    588-d); a T++ KV cache at this width/depth 4.9 GB. One row per sentence would be
    ~4.8k rows.
  *For v13:* the selector does not have to live inside the notebook; an exact organ
  (A4 / N7 class) plus a state-keyed index outside the layers recovers copy, lookup
  and two-hop with the thinking left O(1). *For v14:* the index key that works is the
  per-token state at the salient word; gate the write there (N12), not at boundaries;
  a learned version should beat 0.75 top-1 and shrink the index to one row per event.
  - **Mamba-2 (`mbs2b_main_685de4e`, 102.6M, 4090, 8 docs):** state only 1 needle
    0.0 / 0.375 / 0.0 / 0.0 and 8 needles 0.25 / 0.125 / 0.125 / 0.25 at 2k / 8k / 32k /
    128k: at the floor like the other two bodies on this format; organ 1.0 throughout.
  - **400k (PAM, 1 needle, 4 docs over two runs):** state 0/4, snapshot index 1/4
    (top-5 1/4), organ 4/4. The frozen-state address that is 0.75 top-1 at 128k
    mostly fails at 400k: the nearest-neighbour over 400k per-token positions is near
    the noise floor. A trained key, or one row per event instead of per token, is needed there.
  *Caveats:* 8 docs per cell (about +-0.15 noise), synthetic needles with unique
  nonsense keys favour the n-gram organ, one seed, CPU fp32 for PAM / T++ and the
  4090 for Mamba-2; the "Definition: KEY refers to VALUE." format is the recall-doc
  format the PAM and Mamba-2 recipes train on (5 % share), so T++'s in-window 0.75
  is the format-naive number.

- **v13: the 3B LM gap is copy vs T++ but GENERAL modelling vs Mamba-2.** Eval-only
  attribution on the 3B scoreboard models (PAM `1c913ef`, T++ `e072a8f`, Mamba-2 `b36e3a5`,
  r3b; PG-19 + web, 2048-token windows; EXPERIMENTS_SEMPY "3B LM gap attribution"). vs T++
  (+0.05 / +0.06): 53-58 % from copyable tokens (bigram seen earlier in the window). vs
  Mamba-2 (+0.09 / +0.11): copy is only 24 %, 52-55 % comes from NEW tokens, and Mamba-2
  is already 0.05-0.09 better on new tokens at positions 0-8, with almost no context.
  r3b's loss vs the old body is also on new tokens: M2 lost capacity, echo kept copy. *For
  v14:* a memory / binding unit can close about half the T++ gap (~0.03) but only ~0.02-
  0.03 of the ~0.10 Mamba-2 gap. Beating Mamba-2 on LM needs a better general token
  processor (local mixing / capacity), not only better memory. If a unit idea improves
  per-token modelling with no context (positions 0-64), that is the bigger lever.

- **v13: Mamba-2 on the same rule data nearly matches PAM in context but collapses across
  distance like T++. The distance edge is PAM's, and the two recurrent states keep different
  things.** `b2m_main_38e3c29` (Mamba-2 102.6M, 600M tokens): B2 `_new` mean 0.892 vs PAM
  0.991 / 0.944, so the pre-registered bar (+0.10 in both seeds) is NOT met: no bar-3 claim, no
  ntfy. Spread shift/affine `_new` at 2048 + 4096: PAM 0.67 (both seeds), Mamba-2 0.08, T++ 0.13.
  Mamba-2 collapses at ~110-token gaps. But Mamba-2 leads on exact items: label_diag 0.787
  (T++ 0.713, PAM 0.57-0.63), the last far number through filler (step 0.27-0.31 vs PAM ≤ 0.14),
  holdout 3.4754 (PAM 3.478-3.480). *For v14:* PAM's outer-product state seems to integrate a
  rule that repeats coherently across examples (filler is incoherent noise), but has no way
  to SELECT a single item to keep. Mamba-2's input-dependent Δ is that selector. A selective
  write strength for single items (keep the rule integration) is the gap a new unit could
  close; judge it on label_diag AND spread step_wide_new. v13 runs a fresh-seed distance
  confirmation (s8, pre-registered) for the distance form of bar 3.

- **v13: the B2 edge over Transformer++ replicates (seed 43); the rule survives distance,
  an exact far token does not.** `b2p2_main_38e3c29` (102.9M, 600M tokens): B2 `_new` mean
  0.944 (seed 42 0.991) vs T++ 0.734. LCL spread affine `_new` @2048 / 4096 / 8192 = 0.975 /
  0.905 / 0.61 (T++ 0.05 / 0.45 / 0.285). step_wide (needs an exact copy of the last far
  term) collapses at 1024 for BOTH models; shift and affine need only the rule's parameters
  and hold. Labels still T++'s (PAM 0.63 vs 0.71). *For v14:* the body's recurrent edge is
  in carrying an abstract quantity (a rule value), not in copying distant tokens through
  filler. A unit that adds exact far copying (step through filler) would be new ground;
  `--spread` with step_wide_new is the probe. Bar 3 still waits on Mamba-2 (b2m, ~08:00 IST).

- **v13: N9a at 600M de-collides the label keys but label lookup gets WORSE; recall
  jumps. In-context skills follow the trained format.** `n9cb_main_e5f0b73` vs C1-ctl
  (600M tokens, ~103M): key cosine at label positions 0.715 → 0.538, yet label_exact
  0.610 → 0.472 and label_category 8-shot 0.500 → 0.318; older pairs are lost faster
  (4–5 pairs back 0.16 / 0.12 vs 0.34 / 0.47). Recall 0.825 → 0.975, past 2048 0.597 →
  0.924 (T++ 0.20, Mamba-2 0.46); holdout equal. The recall docs train one template
  ("<key> refers to <value>"); no "<item> is <label>," text is ever trained. With the
  B2 result: PAM learns lookup/rule skills from the FORMATS it trains on, and the
  micro F1 win was formation on the trained format. *For v14:* collision was a
  symptom, not the cause; ask why exact lookup decays with pairs back even with
  distinct keys (write gates? the query window spans the previous label?). For N7,
  judge it on an UNTRAINED format too. v13 next: a label-format training source
  (held-out items), pre-registered first.

- **v13: with rule-format docs in training, PAM infers held-out rule values and keeps
  them across distance; Transformer++ on the same data does not (seed 42; replication
  + Mamba-2 arm running).** `b2p1` vs `b2t` (`38e3c29`, 600M tokens, ~102M each, 3 %
  `b2` docs with held-out values never trained): B2 `_new` mean PAM 0.991 vs T++ 0.734
  (T++ shift_wide_new 0.26 with negative margin = picks the ±1 decoy). LCL spread:
  affine_new at L = 1024 PAM 0.965 vs T++ 0.075 (inside the trained length), at 4096
  0.785 vs 0.45: the first in-context learning past 2048 for any of our models. The
  earlier "no model learns past 2048" was partly a data gap. Cost: PAM recall
  0.870 → 0.721 (answer weight 100 may be too heavy). *For v14:* a unit judged on B2
  should also be judged spread (`--spread`), where the body's recurrence shows its
  edge; the `b2` source (`v7/data.py`) is available for 600M arms. Not yet a bar-3
  claim: needs seed 43 and the Mamba-2 arm (EXPERIMENTS_SEMPY "B2 pair — seed-42
  half").

- **v15 (new lane, "qllm2-59"): a third species, built from the unit up, with
  no attention, no SSM, no PAM, no phases.** Grove: a unit is an organism
  with a *share* of a conserved budget (the K trees of a grove compete; shares
  are the softmax of accumulated evidence, i.e. a Bayesian posterior over K
  hypotheses), a *root* it grows toward the inputs it wins and a *fruit* of
  what followed them; growth happens in seasons (every C tokens), so training
  is parallel and decode == prefill (selftest 1e-15). Forgetting is relative
  (outcompeted), with no decay constant on the state. First signal: a 2-layer
  Grove learns in-context lookup on redrawn pairs (0.98, chance 0.125) only
  after the competition was made sharp (cosine × temperature) and the growth
  decisive (1 − e^{−α·mass}). *For v13/v14:* the read that "picks" is a
  softmax over a small fixed set of units, not a settling of a matrix; if
  N9's settling stalls, a K-slot competitive read is the cheap alternative.
  Design + bars: `v15/EXPERIMENTS_V15.md`.

## 2026-10-30

- **v14 meditation 2: the gap is the body's slope, and the prime suspect is state
  size + how many layers mix.** Mamba-2 ~13.4 MB state, mixing in every layer;
  PAM ~1.2 MB with notebooks in 4 of 16 blocks (M1/M2 chosen at 600M). PAM's
  older 16-notebook 3B run ended better than r3b (3.273 vs 3.309). *For v13:*
  a cheap check from existing logs is loss vs tokens for r3b, the 16-notebook
  3B run and Mamba-2: does the plateau point track state / mixing layers?
  *For v14:* cheap capacity at O(1) (N7 exact table as a cache LM; sparse
  Willshaw keys) is tested on LM loss at ≥ 1.5B tokens, not on micro copy.

- **v13: the 600M lead does NOT survive scale** (`r3b`, 3B tokens). The
  reference recipe (M2 + two stages + echo) ends at holdout 3.309 vs T++
  3.206 and Mamba-2 3.164, and even behind PAM's own old 16-notebook 3B run
  (3.273). PAM learns fast early (ties at 1.2B) and plateaus; Mamba-2 reaches
  PAM's 3B loss at ~1.6B tokens. Every 600M verdict we have is in the
  regime where fast early learning helps. *For v14:* judge LM effects of a
  unit at ≥ 1.5B tokens before believing them; formation/lookup effects
  (F1, label_diag) are less schedule-bound but still need a long run.
- **v14: content-phase keys (N9a) make the lookup circuit form on EVERY seed**
  (F1, `7c8c6e2`, 5 seeds on B1 run 4's set: label_category 1.000 in 5/5
  vs plain echo 1/5, 0.40–0.91; other tasks unchanged). The key-collision
  diagnosis is confirmed by intervention: separate "X is" keys by their
  content and the circuit is no longer a lottery. *For v13:* N9a is the
  first candidate for a 600M check (flag `echo_phase_lags=3`, ~0.1 % params);
  compare with Q1 / S1 when your F1 arms land.

- **v14 meditation: one brain-like picture for the O(1) mind.** (1) Grid cells:
  give tokens multi-period phase coordinates from the start, so rule
  inference = path integration and stops being a per-seed lottery (3/10
  today). (2) Hippocampus: pattern separation (sparse high-dimensional keys:
  look-alike "X is" keys become orthogonal, capacity rises, the delta erase
  stops wiping neighbours) + pattern completion (settling) = N9; N7 is the
  exact extreme. (3) Prefrontal planner forms the query (N11). *For v13:*
  sparse expansion keys are a direct, cheap test of the collision diagnosis
  on your body too.

- **v13 (after a quiet step back): write with hindsight, at the end of a
  meaning unit, not at every token.** Common thread of today's failures:
  PAM files every token the moment it arrives, keyed by whatever that token
  is ("is"), and never revisits. Minds store after the fact, once they know
  what the unit was ("truck → silver" is only a unit once "silver" lands).
  Idea: a tiny buffer of the last few tokens; at a boundary (comma, period,
  surprise drop) bind the finished unit and file it under what it is ABOUT
  (its content token), not under the last token. O(1); kills template-key
  collisions at the source; pands-like (read a unit, then bundle it). Also:
  all-or-nothing circuit formation looks like a phase transition, and those
  are crossed reliably by ANNEALING (start the read soft, sharpen it over
  training), a schedule-level twin of Q1. *For v14:* a write-side unit
  that complements N7/N9 (read side) and N8 (gate).
- **user + v13: "reason first, then know where to look"; two lookups
  and rules as phase continuation (vision, design ideas).** (1) *Analytic
  continuation* (the zeta-function idea): an analytic function known on a
  patch is fixed everywhere. For a mind: give ordered things phase
  coordinates (letters on a circle, numbers on several circles of
  different periods, multiplication via log-phase), so many rules become
  one rotation; inferring a rule = reading an angle from the examples;
  applying it to an unseen value = exact continuation, not recall (echo
  already infers held-out shifts, B2). (2) *Built-in RAG, pands-style*
  (point first, then speak; the user's pands v1 forms an essence
  direction before speaking): reason → form the query ("I need truck's
  label") → look in TWO memories, a knowledge store learned from all
  training data and the conversation memory (N7/N9) → speak toward the
  intent. Today PAM looks up with whatever key the current token gives
  ("is"), which is the label_category failure. O(1) if both are addressed
  by computed phases/hashes, not searched. Prior art for large learned
  stores: product-key memory, memory layers; the new part is reasoned
  addressing with content phases. (3) *Learns faster?* Holdout at 150M:
  PAM 3.905 vs T++ 3.981 (faster), the gap closes to −0.019 by 600M;
  Mamba-2 is ahead at every point (−0.02 to −0.035). *For v14:* (1) and
  (2) are new mechanisms; (2) is the third memory + planner of the
  two-memory design.
- **user + v13: engineer the start the lucky seeds had, instead of
  trusting the seed (Q1, `e89b7d2`).** In-context circuits form per seed,
  all or nothing (label_category: tied echo 4/4 vs echo 0/5; held-out
  shifts: echo-family 3 of 10, T++ 0 of 4). User: "it's what that seed
  did; can we initialise so it works for all seeds?" Clue: a q/k-aligned
  start forms the circuit; the permanent tie (E2) cost recall. Q1 = start
  tied (W_q = W_k at init), train free. Stage 1 = formation rate over 5
  seeds on v14's bench; stage 2 = 600M. *For v14:* the same lens fits
  every new unit: report the formation rate over seeds, not one seed;
  and an init that sets the intended circuit up (like v13's
  identity-at-init) may matter as much as the unit's design.
- **v14: the O(1) body can infer held-out rule parameters COMPLETELY, but it
  forms the circuit per seed, all or nothing** (B2 state sweep, `dd5b174`:
  held-out letter shifts 1.000 / 0.997 in 2 of 6 runs, 0.92 in a third;
  over all echo-family B2 runs 3 of 10 ≥ 0.9; T++ 0 of 4, ≤ 0.03). The
  state-size "bottleneck causes abstraction" hypothesis is NOT supported
  (the two seeds order the sizes oppositely). *For v13:* the open lever is
  what makes the circuit form reliably (data order, curriculum, a critical
  period, like your two-stage effect); single runs of rule probes need
  several seeds before any recipe or scale conclusion.

- **v13: past the trained context, PAM keeps the label SET, not the
  mapping; no model's learning holds past 2048** (LCL, `7707c8c`).
  label_category with the 8 examples spread to 4096 / 8192 tokens: PAM
  0.25 / 0.20 = in-context chance (it knows which labels appeared, not which
  item gets which); T++ 0.12 / 0.13 = candidate chance (lost the context).
  Pre-registered "PAM beats T++ by ≥ 0.10 at 4096 and 8192": NOT met (+0.13
  / +0.07). Rule tasks are at the floor for every model at every length.
  *For v14:* score the LCL bar against IN-CONTEXT chance (0.25 at 8
  shots), not candidate chance; a unit only "holds past 2048" if the
  mapping survives. The collision fix (N7 / N9) is what should lift it.
- **v13 + user: the nested version of the 2×O(1) memory: phase finds the
  region, SETTLING picks the best (design idea, refines the entry below).**
  User: the second O(1) should live inside the first ("first get the
  phase, then another memory learned in that phase picks the best"), and
  maybe be something altogether different. Two facts shape it: (1) more
  phase / more state = more address bits, but our collisions come from
  look-alike keys, not too few bits; (2) two plain linear memories read in
  sequence give the SAME score as one (forward Σ(e·v)(k·q) = backward
  Σ(q·k)(v·e)), so a second store cannot pick. Picking needs a nonlinear,
  competitive step. *Design:* step 1 = content-phase keys (each stored item
  its own valley); step 2 = a fixed 2–3 rounds of SETTLING: pull the query
  toward the nearest stored key (Hopfield-style relaxation on the key
  autocorrelation, with a sharpening nonlinearity) so close-but-wrong
  items lose instead of blending; then read the value. Constant cost; no
  attention cache, no state-in-state. Limit: a settling memory holds only
  a few dozen clean valleys per head, so writes need the N8 gate (store on
  surprise / maturity). Nature: systems relax into the nearest stable
  state; hippocampal theta-gamma nesting (slow phase = context, fast cycles
  = items). *For v14:* N9 = gate (N8) + content phase + settling; the
  two-bank vote is the fallback. Build in stages: phase keys alone (key
  cosine + label_exact), then add settling (5-pairs-back accuracy), then
  600M with the same bars.
- **v14: REPLICATED: the O(1) echo body infers unseen rule parameters where a
  same-size transformer only selects** (B2 run 2, `9e0a3fd`, 3 seed pairs:
  held-out letter shifts echo 0.14 / 0.26 / 0.34, rising with shots, vs
  T++ ≤ 0.03; d mean +0.230 ≥ 2·SD). First replicated bar-3 result in our
  favour (27M). *For v13:* worth a 600M check with B2 docs in the mix, inside
  and past the trained context (your LCL spread flag covers it).

- **v13 + user: a two-step O(1) memory, "align, then pick" (design idea,
  not yet tested).** From the key-collision diagnosis and the user's "2×O(1):
  first strong alignment, then pick the best one". *Step 1, align by
  phase:* rotate each memory key/query by phases computed from the last
  few tokens (learned angles per token, a few frequency bands; function
  words can learn ≈ 0). "truck is" and "red is" then point different
  ways (no template collision); a repeat of "truck is" aligns exactly;
  non-matches cancel by interference instead of adding blur; learned
  angles let car ≈ truck (label_category). Content-relative, not
  positional, so it cannot run out of range past the trained length.
  *Step 2, pick by agreement:* two such memories with independent phase
  banks; combine their votes as a product (AND): collisions are random per
  bank, the true value is common (count-min-sketch logic: error falls
  fast with each independent copy). Cost 2×O(1). *For v14:* fits the
  two-memory design as the learned store's alignment + the "body chooses"
  step (N7 exact store = the hard-hash limit of step 1). *Bars:*
  label_exact ≥ T++ (0.88), label_category ≥ 0.73, accuracy 5 pairs back ≥
  0.70, LCL past 2048, recall/holdout not worse. *For v13:* g1c (short
  conv = additive keys-see-previous-tokens) tests step 1's premise now.
- **v13: PAM's in-context lookup dies of key collisions, not distance or
  capacity** (`label_diag`, `bc7d57f`). Exact copy ("... truck is silver
  ... truck is" → silver): PAM 0.50–0.62, Mamba-2 0.62–0.72, T++ 0.78–0.88.
  3.3× the state does not help; 3B tokens do not help. Accuracy falls with
  the number of similar pairs in between (0.72 → 0.29 over 5 pairs, ~20
  tokens), while 1,000 tokens of unrelated filler barely hurt (0.44 →
  0.37). Echo keys of different "X is Y" pairs have cosine 0.67–0.72 (vs
  0.31 between random positions): every label is filed under the same
  "<noun> is" key, and the delta erase makes the last write win (45 % of
  wrong answers = the most recent label; T++ 23 %). *For v14:* keys must
  carry the content token, not the template: N7's hash of "truck is" is
  collision-free by construction; any learned memory needs a key that
  sees the previous tokens (Mamba-2's k = 4 conv likely explains its
  0.72), or centred keys. *For v13:* testing G1 short conv (`g1c`).
- **v14 (on v13's label_exact): format-free exact copy is N7's target.**
  PAM's copy skill is format-bound (recall probe 0.94, but label_exact
  0.54–0.73 vs T++ 0.86–0.92). An in-context n-gram store copies
  "truck is -> silver" in any format, never fades, O(1). *For v13:* the
  data lever (generic repeated-text / ICL data) and N7 are complementary;
  label_category (fuzzy match) stays a separate problem.

- **v13: label_category is not an attention-only skill; Mamba-2 (O(1))
  does it.** 600M, same two-stage recipe (`685de4e`): Mamba-2 label_category
  8-shot 0.55, learning mean 0.227; PAM M2 0.41 / 0.185; T++ 0.73. So a
  fixed-size state CAN learn this fuzzy in-context lookup; PAM's memory read
  does it worse. PAM's edge over Mamba-2 is recall past 2048 (0.72 vs
  0.46) at ~1/11 the state (1.2 vs 13.4 MB); Mamba-2 leads LM by 0.010.
  *For v14:* the gap is in how PAM reads (sharpness / key geometry), not in
  O(1) itself; Mamba-2's input-dependent decay + gating is the reference
  that works. label_category is a SEMANTIC match (new item of a seen
  category), so an exact n-gram store (N7) alone will not close it.
- **v13: the label_category lift depends on the body, not only the
  schedule.** The two-stage schedule lifts 8-shot label_category on the M1
  body (0.29 → 0.55, C1-ctl `7ae6f1f`) but not on M2 with shared reads
  (c2m2 0.41, c1m2 0.38). M2 is better at LM and recall, worse at this
  lookup. Single seeds, so treat it as a lead. *For v14:* a body that reads one
  writer's store from many blocks may dilute the in-context lookup
  circuit; in the two-memory design, give the exact store its own reader.
- **v13: reasoning data lifts learning, then it fades.** Right after the
  aptitude stage, models post their best learning-probe means (Transformer++
  tfc1a 0.229 with number_step 0.083 / reverse3 0.063; PAM c1m2a 0.195),
  and it fades during the web-heavy main stage (tfc1b 0.209, c1m2b 0.187).
  *For v13:* test rehearsal (a small reasoning share kept in stage B).
  *For v14:* judge a learning unit after the same data exposure; what the
  model saw last moves these probes as much as the unit does.
- **v13: Transformer++ gains WikiText from aptitude data, PAM does not**
  (T++ 66.2 → 63.8, PAM 65.6 → 65.7; tfc1 / c1m2 vs tfs2 / c2m2), while PAM
  wins the web holdout (−0.033). Encyclopedic text rewards exact recall of
  stored facts, which fits v14's lookup/abstraction split. *For v14:* an
  exact store (N7) may be where PAM's WikiText gap closes; check WikiText
  as well as holdout when N7 reaches 600M.
- **v14: lookup and abstraction pull apart.** Attention wins at exact
  lookup (600M label_category: T++ 0.73 vs PAM 0.41); the O(1) body wins at
  carrying a rule (B2 run 1, `bf59754`: held-out letter shifts, echo 0.33
  rising 0.20 -> 0.47 with shots, T++ 0.02 = one memorised table per shift).
  *For v13:* judge PAM on rule tasks and past-context learning, where it
  has an edge, not only on lookup probes. *For v14:* stop patching lookup
  into echo; give lookup to a separate exact store (N7).
- **v14: the bottleneck may cause the abstraction (hypothesis, under test).**
  Attention can keep every example, so a table per parameter is cheap; a
  fixed-size state cannot, so it is pushed to keep the rule. Test running:
  B2 state sweep (`dd5b174`, memory 12k / 25k / 49k at equal params).
  *For v13:* if it holds, a bigger state (3B, shared reads, wider heads)
  could trade abstraction for recall; worth a learning-probe check at 3B.
- **v14: symmetric addressing kills recall.** Tied echo (q = k) formed the
  similarity circuit in 4/4 small seeds but at 600M lost recall (0.825 ->
  0.602, E2). Recall is asymmetric ("what came AFTER X"); similarity is
  symmetric. A memory needs both, separately. *For v13:* keep q != k in
  any memory that serves recall.
- **v14: echo's numbers degrade with more shots** (B2 step_wide_new 0.98 /
  1.00 / 0.72 at 2 / 4 / 8 shots; T++ 0.97 at 8). Likely state saturation
  over long numeric runs. *For v13:* a capacity / precision clue for the
  notebooks at long context.
- **v14: micro-bench wins are leads, not evidence.** B0 broke even for
  attention (answer-only supervision); tied echo's clean micro win reversed
  at 600M. Micro-benches need dense LM-style supervision, and every win goes
  to seed pairs, then 600M, inside and past the trained length.
- **v13: the two-stage schedule helps only PAM** (tfs2: T++ gets nothing;
  c2m2 / C1-ctl gain recall). Reading: a critical period, where the
  notebooks need mature representations before they learn what to store.
  *Handed to v14 as N8:* writes gated by the model's own maturity.
- **v13: PAM's clearest edge is memory past the trained length** (two-stage
  M1 recall 0.60 vs T++ 0.15 past 2048). *For v14:* every new unit is
  judged inside and beyond the trained context (LCL probe, `cf2d7ec`).
