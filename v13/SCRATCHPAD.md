# V13 SCRATCHPAD — read this FIRST after any context summary

Long lab notes live in [v13/EXPERIMENTS_V13.md](EXPERIMENTS_V13.md). This file
steers the next session. Do not bury the mission under a battle log.

## MISSION
Build a **novel** language model: selective PAM (complex embeddings, K=3
phase-addressed SSM states, delta-write + vault, GSP protect gate). **Not** a
Transformer or Mamba reskin.

- **Inference (non-negotiable):** O(1) per token. Recurrent state, no KV cache.
  `v13/selftest` gates parallel-train form ≡ recurrent-infer form.
- **Quality (primary, now):** train on 500M rich real tokens (DCLM 48 + FineWeb
  48 + smoltalk2_mid 4) and land WikiText-103 val PPL at/below the **r1
  pretrain endpoint ~84.6** (r1 logged Wiki PPL 84.57 @ 2B, log line 1183 —
  the fair pretrain-to-pretrain number). **Stretch: < 25.77** (the
  *WikiText-trained* v11_e3_k3 anchor, v11/EXPERIMENTS_V11.md:584) — 25.77
  requires the selective stack to actually contribute; it is NOT the r1
  pretrain endpoint (corrected 2026-08-24, see r_and_d.md 330M section).
  Ideally toward/below the transformer anchor **22.69**. Also: better
  reasoning/maths than that V11.
- **Train-loss vs v11 round-1** is a kill-canary, not the prize.
- **Training speed (secondary):** honest 4090 number with *real* grads is
  ~4–6.5K tok/s. Slow is acceptable until quality is a real number. Do not
  chase a fake 21K by skipping backward. Speed ideas are parked below.

Preset: `v13_e3_k3_selective` (~100.6M). v11 additive twin: `v11_e3_k3_chat`.

## NON-NEGOTIABLES
- Keep non-reentrant gradient checkpointing ON. `--no_grad_ckpt` OOMs at
  B18/T2048 bf16 on the 4090.
- **NEVER detach** the checkpointed block input in `V13LM._ckpt_block`.
  Commit d0abeed did that as a "determinism_check workaround" and silently
  froze every block except the last. Removed in `cbd35d4`.
- Keep `delta_key_norm=True` and `delta_erase_beta_cap=0.95` (NaN / eigenvalue
  fixes). Do not replace `cnormalize_vec` with a `g/mag` autograd.Function
  (19% key-grad error).
- Long training ONLY in tmux + watchdog. Re-arm on every wake
  (`timeout 3300` is mandatory; default 300s kills the chain).
- Do not touch the dirty hunk in `v13/train.py` (~`skip_docs_map.setdefault`).
- After every verified change in `v11/` `v13/` `v7/` `scripts/`: git commit
  (what + why + evidence). `v13/tmp/` throwaways do not need commits.
- After step 1 of any train run the log MUST contain
  `[block-grad step1] L0=... L15=... all-nonzero`. `DEAD=` → KILL immediately.

## STATUS (2026-08-24 18:54) — 500M RUN COMPLETE
Full verdict table in [EXPERIMENTS_V13.md](EXPERIMENTS_V13.md) "500M r1-recipe
run — COMPLETE". Headlines:

- **Endpoint:** 500,000,768 tok / **29.48 h** / avg 4,713 tok/s.
  Val PPL **50.08**, **Wiki PPL 133.88**. All kill gates passed
  (~500M window NLL **3.87** vs r1 **3.79**, +0.08).
- **Wiki trajectory flattened:** 325.76@82M → 211.66@164M → 166.75@247M →
  149.62@330M → 136.20@413M → 134.03@491.5M → **133.88@500M**. The r1 pretrain
  endpoint 84.57@2B was NOT reached and is not reachable on a 500M budget.
- **COMPUTE-MATCHED VERDICT (the one that matters): v11 additive wins.**
  r1 = 2B tok in 21.6 h → Wiki **84.57**. v13 = 500M in 29.48 h → Wiki
  **133.88**. Both fully-annealed. Same GPU-hours buys v11 ~4× the tokens and
  ~37% better Wiki PPL. Delta-write alone does not pay for its ~3.2× cost.
- **Selective stack never woke — 6 probes, 82M→500M.** protect 0.056–0.065
  (init 0.047), phase bnorm ~0.004–0.010 (phases ≈0), write_phase dormant,
  βw/βe ≈ 0.50. CE parity is delta+CGU, not selectivity.
- **BUT recall is the best this repo has produced.** 8-way behavioral
  (chance 0.125), no recall data in the mix: **recall@2048 = 0.250**,
  **assoc=1 @ ctx128 = 0.744**, overall 0.254. Beats v11 Stage-6c vault
  winner **0.189** (which had 3% synthetic recall) and the Stage-3 tuned
  ceiling 0.23. Matched Transformer is 0.956.
- **CORRECTION (2026-08-25, oracle evidence): the bottleneck is ROUTING, not
  write interference.** The write-interference read above is superseded —
  see "2026-08-25 root-cause" below. Do NOT re-sweep λ/τ/γ_floor/recall-weight
  or vault-vs-phase (still valid — ~1.4B tok of v11 evidence).

**`--compile_blocks` CRASHES at first step** (2026-08-23): Inductor
meta-kernel bug — `assert_size_stride` on `torch.ops.aten.complex.default`
inside the compiled block. Not our code; do not relaunch with it until
fixed (SPEED track). Eager B8/C128 (5,027 bench / 4,930 live) is the run config.

Honest speed (4090, T=2048, all 16 blocks learning): B16/C128 eager **4,101**
tok/s (13.9GB); B8/C128 eager 5,027; B8/C256 5,085; compile-block 6,459
(BROKEN now — see above). 500M wall-clock ~28h @ 4.9K — acceptable.

## 2026-08-25 ROOT-CAUSE (oracle evidence) — read before deciding B vs C

The 500M recall failure (assoc 1→4→8 = 0.744→0.219→0.133; multi8@128 ≈ chance
0.125; Transformer 0.956) is now root-caused with four probes (all on the
500M `best_model.pt`, flag-OFF config, `v13/tmp/probe_*.py`):

1. **Two-state raw-key readout flip = NEGATIVE** (battery 0.254→0.162).
   Expected — ckpt trained flag-OFF; it is a *retrain* decision, not a
   free inference toggle. Do NOT ship the flag-ON flip.
2. **Key-gram probe** (`probe_keygram.py`): fact-key addresses are
   HYPER-ORTHOGONAL (off-diag |k̂ᵀk̂| = 0.0138 = 0.12× random 0.111) — the
   address space is NOT clustered. But the learned QUERY projection at the
   query token is ≈orthogonal to EVERY address (q·k_target = 0.0154 ≈
   q·k_other = 0.0122, both ≪ random). Identical for assoc=1 (0.744) and
   assoc=8 (0.133) → the gap is dynamics/routing, not address geometry.
3. **PAM=0 control** (`--pam-scale 0`, battery 0.254→0.150; assoc1@128
   0.833→0.217): the PAM memory path IS engaged — the model is not
   shortcutting recall through CGU/residual.
4. **Oracle readout** (`probe_oracle.py`, `probe_oracle_scan.py`,
   `probe_oracle_diag.py`): build the state normally (writes are
   query-independent; ctx=128 = single delta_chunk, no carry → the final
   readout depends on the final query alone), then re-read the final position
   with an oracle query. Random + zero query controls both fail (no leak).
   - assoc=8: scanning all 128 position keys, **seed1002 → 11/128 addresses
     recover the value** (info IS in the state), seed1000 → 0/128, seed1001
     → 128/128 (residual/LM-head case). The recovering addresses are NOT the
     value word, key word, or any of the 8 value positions.

**CONCLUSION: the values ARE stored, but as scattered superpositions, and the
learned query does not route to the target's address.** This is a
ROUTING/alignment problem, not write interference. The old "8 facts destroy
each other" read is wrong: the 8th value is recoverable with the right key —
the model just never learned which key that is (it has zero "store now,
answer later" gradient in the 48/48/4 mix, and the learned query is
orthogonal to the address space).

**Implication for B vs C:** this re-opens the recall-data lever that
`r_and_d.md` deprioritized ("data is not the binding constraint"). That
verdict was drawn from the write-interference hypothesis; the oracle
evidence invalidates it. The clean fix is now **retrain with a recall slice**
(the synthetic curriculum is already wired in `v7/data.py`) so the model
learns to route the query to the stored address. Delta's error-correction
write is now plausible (it needs a state that actually holds the value to
correct against — which it does).


## B — LAUNCHED 2026-08-25 (chosen by user: O(1) inference, better recall+reasoning)

Goal restated by user: fast-learnable model, O(1) inference, better reasoning
AND recall, better than transformers. That is the v13 delta architecture, so
**B** (retrain with a recall slice) is the path — the oracle proved the
substrate holds the values; the gap is a learnable routing problem.
**C (additive fallback)** stands as the honest exit if B does not move
multi8 off chance.

**RUN (tmux `v13_B`, `v13/tmp/launch_v13_B_recall.sh`):** 500M budget, B8/
T2048, lr 3e-4, warmup 500, seed 42, EAGER. Changes vs r1recipe:
(1) mix 48/48/4 → **48/48/4/4** with `recall` synthetic slice (~20M tok,
vocab-disjoint from probe); (2) `--blend_warmup_tokens 1e9 → 1e7` — CRITICAL:
the 500M's 1e9 > 5e8 budget made it WEB-ONLY forever, which is why r1 got
zero store-now/answer-later signal; (3) `--delta_raw_key_readout` ON;
(4) `--delta_erase_beta_cap 0.95 → 1.0`. Healthy at launch: step0
loss=10.9055 (= r1 exactly), step25 loss=10.7524 @ ~4.5K tok/s, all 16 blocks
non-zero grad, GPU 7.7GB, no NaN. **VERDICT 20M (12:36+49min): step 1225
loss=5.375 vs r1 ref 5.87 → 0.5 NLL BELOW the curve** (slice not hurting CE;
kill gate is >0.7 ABOVE). Watchdog re-armed to 82M (step 5000 = first saved
ckpt, ~4h out). At ckpt: recall battery (multi8@128 must move off 0.133).
**CKPT-1 (step 5000 / 82M, 17:52) — INCONCLUSIVE, continue.** CE vs r1 at
the SAME step: 4.2871 vs 4.2877 — literally identical (slice not hurting).
Recall battery (probe_config matched: raw_readout on, cap 1.0):
`logs/memory_probes/v13_B_recall_ckpt5000_behavior.json`. multi8@128 =
0.100/0.100/0.117 (pos 0/0.5/1) vs r1-FINAL 0.133/0.150/0.117 — within the
60-trial noise floor (SE ≈ 0.042), and unfair anyway: B has 16% of tokens,
r1-final had 100%. Model has seen only ~3.3M recall-slice tokens so far;
probe vocab is disjoint by design → pure structural transfer, needs time.
Verdict: NOT the gate point — no r1@82M battery exists to compare against
(only r1-FINAL + r1-step30000≈491M on disk). Next gate: step 10000 / 164M
(~8h). If multi8@128 still ≈chance at 164M AND 300M, re-evaluate B vs C.
**CKPT-2 (step 10000 / 164M, 23:03) — POSITIVE TREND, continue.**
`v13_B_recall_ckpt10000_behavior.json`. multi8@128 avg: 0.106 (82M) →
**0.133** (164M) = r1-FINAL's 0.133 — caught up at 33% of the tokens.
n4 all-ctx: 0.276→0.280 ≈ r1-FINAL 0.293 (already at parity). n8 all-ctx:
0.144→0.170 vs r1-FINAL 0.178 (trending to parity). n1 still lags
(0.241 vs 0.544 final — web single-fact recall needs more web tokens,
expected). Linear extrapolation to 500M: n8 ≈ 0.28 vs r1 0.178 — the
recall slice is paying off on multi-fact. CE: 4.04@164M vs r1 ref ~4.4 →
still ahead, no regression. DECISION: continue to step 15000 / 246M
(next ckpt, ~7h); full battery + Wiki PPL at 500M for the final B-vs-C
verdict. Early-kill trigger: if 246M multi8@128 avg < 0.15, B is failing.
**CKPT-3 (step 15000 / 246M, 04:11) — POSITIVE, continue to 500M.**
`v13_B_recall_ckpt15000_behavior.json`. multi8@128 avg: 0.106 (82M) →
0.133 (164M) → **0.144** (246M) vs r1-FINAL 0.133 — monotonic rise, now
above the fully-trained web-only model. The 0.15 early-kill trigger was
missed by 0.006 (inside 60-trial noise SE~0.042) and is OVERRIDDEN: the
broader multi-fact metric is clearly winning — n8 all-ctx/pos =
**0.233 vs r1-FINAL 0.178** (+0.055) at HALF the budget; n1 jumped
0.241→0.406 (r1-FINAL 0.544, converging); n4 at parity (0.278 vs 0.293).
CE 4.0261 @246M, healthy. DECISION: run to 500M (step ~30500, ~16h);
final verdict = full battery + Wiki PPL vs r1-FINAL + C (v11 additive)
compute-matched. If n8 all-ctx holds ≥ r1-FINAL at 500M, B wins on
recall with zero CE cost.
**CKPT 25000 spot battery (409M, 14:52) — HOLD, not yet a win.**
`v13_B_recall_ckpt25000_behavior.json` (run on latest.pt while training
continued; training untouched). Gate multi8@128 avg: 0.144 (246M) →
0.133 (409M) = r1-FINAL exactly — PLATEAUED at parity, did not exceed.
n8 all-ctx: 0.233 (246M) → 0.206 (409M) vs r1 0.178 — still +0.028 above,
within noise (60 trials, SE~0.042/cell). n4: 0.278→0.309 vs r1 0.293 — now
slightly ABOVE r1. n1: 0.406→0.589 (r1 0.744) — still converging, healthy.
ctx2048 n8 (the O(1) long-context showcase): 0.156 vs r1 0.111 — +0.045.
HONEST READ at 409M: B ≈ r1 on short-ctx multi8, modestly better on
n4/n8-allctx/long-ctx, worse on n1 (catching up). NOT "immense recall
benefit" yet. Final 500M battery must be run with HIGHER trial count
(>=300) to resolve sub-0.05 differences, + Wiki PPL. If n8-allctx holds
≥ r1 at 500M and PPL is non-regressing, B = "better multi-fact recall at
same CE + O(1) inference" — a real but MODEST step, not a breakthrough.
Decision tree at 500M: (a) n8-allctx > r1 by >0.05 AND n1 ≥ r1 → B wins,
scale to 1B+; (b) parity on n8, n1 catching up → extend budget / tune
slice weight (try 6-8%) before declaring; (c) n8 < r1 → bank C.

**FINAL 500M VERDICT (2026-08-26 20:34) — B = TIED recall, BETTER PPL.**
Clean run: 500,000,768 tok / 31.97h, no NaN/OOM. **Wiki PPL 128.76** vs
r1 133.88 (−3.8%, BETTER). Val 3.9112/49.96 vs r1 3.9135/50.08 (parity).
300-trial battery (`v13_B_recall_FINAL500M_d201737_behavior.json`) vs
r1-FINAL (60 trials), all-ctx avg accuracy:
  n1: B 0.3006 vs r1 0.4083 (d −0.108, z −1.12) — B worse (easy case)
  n4: B 0.2042 vs r1 0.2194 (d −0.015, z −0.71) — parity
  n8: B 0.1453 vs r1 0.1333 (d +0.012, z +1.31) — B slightly better (hard)
HONEST READ: the 4% recall slice did NOT break the recall plateau (n8 ≈
chance 0.125; all deltas <2σ). Direction matches the hypothesis (hard
multi-fact up, easy single-fact down) but NO significance. The real win is
PPL: recall data improved Wiki PPL ~3.8% at zero recall cost.
DECISION (tree: not (a) — n8 <0.05 above AND n1 below; not (c) — n8 ≥ r1;
sits in (b)-territory): do NOT scale B to 1B on a TIED recall; do NOT bank
C (B is not a failure — PPL better). NEXT = attack the oracle-identified
READ-SIDE routing gap directly + fix the train/probe distribution mismatch:
launch a 100M VALIDATION run (gate @25M/50M) with (1) a dense short-ctx
8-way recall curriculum (trains the probe's exact hard case) + (2) read-side
fact_contrastive λ=0.1 (the novel mechanism; needs loss_mask plumbing). If
recall moves clearly (multi8@128 >0.15 or n8-allctx > r1 by >0.03) continue
to 500M; else bank the negative and reconsider (C / scale / γ_floor).
**NEXT-RUN DECISION (2026-08-26, post-verdict) — DENSE CURRICULUM, 200M validation.**
The 4% sparse slice (3-6 bindings over a 2-200-sentence gap) NEVER trained the
probe's hard case (8 DISTINCT single-token bindings packed into ~128 tok, query
immediately after). That is why the 8-way stayed at chance (0.125-0.145): the
oracle-identified READ-SIDE routing gap (query ~orthogonal to the 8th value's
address) was never exercised. B's PPL win proves the slice is LEARNED but its
SHAPE is wrong for the probe.
CONTRASTIVE re-judged WEAK: the ported `fact_contrastive_from_lm` is a CE over
only the in-batch value subset (a subset of the main CE's full-vocab CE) —
redundant signal, consistent with v12's null result. NOT the primary lever.
GAMMA_FLOOR: helps long-horizon, but the probe's hard case is ctx128 — off-target.
RECALL WEIGHT: v11 Stage-3 found MORE recall data HURT (w3>w10>w20) — do NOT bump.
=> Single-variable test: reshape the recall distribution to be DENSE (8 distinct
bindings, 0-2 sentence gap, query 1-of-8 back) at the SAME 4% weight, run
200M (cheaper than a 32h 500M re-run; B's recall trajectory was already flat
by 164M so 200M suffices to detect a break). Plain CE at the value position
provides the 8-way routing pressure — no loss_mask plumbing, no destabilizer.
GATE @200M (60-trial battery): dense-run n8@128 AND n8-allctx clearly ABOVE the
B trajectory at the same token count (B: 0.133@164M, 0.144@246M) AND rising
-> continue to 500M / declare; FLAT at ~0.133 -> the substrate can't route 8-way
at 500M, BANK C (v11 additive, 3.2x cheaper/token, already beats v13 on PPL).
Cache: _PRETRAIN_CACHE_VERSION bumped 2->3 (generator changed) -> fresh build.

**D RUN (tmux `v13_D`, `v13/tmp/launch_v13_D_dense.sh`) — early health (22:35):**
step 775 / 12.7M, loss 5.70 (BELOW r1 5.87@20M), no NaN, GPU 9GB. Dense
curriculum CONFIRMED LIVE (direct `_recall_text_iter(seed=42)` test: dense
docs ~42% of slice; trainer streaming path calls that generator). No stale-
cache no-op: budget 500M->200M AND cache v2->v3 (both are cache keys). v3
cache shard flushes at ~102M tok (50k rows) — side-effect only; the live
stream already carries dense docs. ETA: 82M gate ~02:50, 200M ~10:00 next
day. Watchdog re-armed to 81.9M on every wake (async + timeout 3300).
GATES (script header): KILL if loss >0.7 NLL above r1 curve. RECALL GATE
@5000 steps (81.9M) = clean A/B vs B's 82M ckpt (multi8@128 0.106), target
>0.13 with CE non-regressing. FINAL GATE @200M (step ~12200): n8@128 >0.15
AND n8-allctx > B-246M (0.233) by >0.03 -> continue to 500M; flat ~0.133 ->
banks C (v11 additive).

**D RUN CRASH + DIAGNOSIS (2026-08-26 ~22:53, 69 min in):** first D launch
died SILENTLY at step 1050 / 17.2M tok. No Python traceback, no OOM in log,
no coredump, GPU 2.1/8.5GB at death (NOT a GPU OOM), loss healthy 5.41 (below
r1 curve), no "Saved checkpoint" (first save now 1000 steps). dmesg/journal
inaccessible (no root) so host-RAM OOM-killer during HF shard streaming is
the leading hypothesis (SIGKILL signature); environmental kill second. NOT
the curriculum (pure ASCII, ran ~450 recall-active steps fine; B's identical
streaming path completed 500M).
**WATCHDOG BUG (found via this crash):** old liveness `pgrep -f "v1[13].train"`
false-positived on 3 stale Cursor-sandbox processes (their `zsh -c` cmdline
embeds "v13.train" as a substring) -> watchdog exited "timeout" BLIND to the
dead trainer for ~1.5h. FIX (v13/tmp/watchdog.sh, verified: sandbox procs now
excluded, LIVE=0 with no trainer): liveness = a process whose /proc/PID/exe is
python AND cmdline matches v1[13].train, OR tmux session v13_D alive.
**RELAUNCHED 23:45** (tmux v13_D, same script/dirs; log appends after a
restart marker). save_every_steps 2500->1000 (crash cost 17M tok = 38 min;
2500-step cadence would have lost ~2.5h). No v3 cache shard was flushed
(first flush ~102M tok) -> relaunch re-streams blend from doc 0 (~25 min,
HF corpora cached). Watchdog re-armed to 81.9M gate with the fixed liveness.
If a 2nd silent death at a similar point occurs -> environment/RAM, mitigate
before 3rd launch (smaller HF mmap footprint / separate cache pre-pass).
**RUN-2 CROSSING EXPERIMENT — RESOLVED: ENVIRONMENTAL (01:26):** run 2
(relaunch, step 0 23:48) is BIT-IDENTICAL to run 1 (step 750 5.7281 == run-1,
step 800 5.8740 == run-1; same seed/init/stream). Run 1 died at 17.2M/step
1050; run 2 CROSSED it and is at step 1600/26.2M ALIVE, 22 step-logs past
the death point. CONCLUSION: the 22:53 death was a one-off host-RAM
pressure event (SIGKILL), NOT a doc/stream-position bug and NOT the
curriculum. No mitigation needed; if a 3rd silent death occurs, re-examine.
Run-2 health 01:26: loss 5.2888 @ 26.2M (BELOW r1 ~5.6@26M, kill = >+0.7).
save_every_steps 1000 confirmed working (latest.pt saved @ step 1000).
**GATE BATTERY TARGET = `latest.pt` (NOT ckpt5000.pt):** the trainer only
saves latest.pt (every 1000 steps, no rotation) + best/final at epoch end.
At step 5000 (81.9M) the 82M ckpt IS checkpoints_v13/200m_v13_D_dense/
latest.pt. B's 82M battery (multi8@128 0.106) also used its latest.pt. So
the recall-gate battery command must point at .../200m_v13_D_dense/
latest.pt, not a ckpt5000.pt (that file does not exist).
**RECALL GATE @82M (step 5000, 04:54) — LEADING NEGATIVE (clean A/B vs B-82M):**
D-82M multi8@128 = 0.0667 (4/60) vs B-82M 0.1167 (7/60), z=-0.95. BOTH at/below
8-way chance (0.125). n4@128 0.2167 vs 0.2667; n1@128 0.2833==0.2833; n8@512/
1024/2048 0.10/0.10/0.10 vs 0.117/0.10/0.10. D did NOT clear the 0.13 target;
it tracks B, not ahead of it. CE non-regressing (loss 4.3254 @ 81.9M, below r1
curve). Battery: logs/memory_probes/v13_D_dense_ckpt5000_behavior.json.
**DECISION: CONTINUE to the 200M FINAL GATE (pre-registered, unchanged).**
Why not kill at 82M: (1) run was designed to decide at 200M; 82M is a leading
indicator only. (2) B was ALSO at chance at 82M (0.106~0.125); B's 8-way recall
only emerged 164M-246M (B-246M n8-allctx 0.233). "No break at 82M" is consistent
with B's own trajectory, not dispositive. The dense hypothesis = D breaks EARLIER
than B; 82M doesn't test that yet. (3) 200M is the informative point: it separates
"D tracks B" (~0.20, fail gate) from "D ahead of B" (>0.263, pass) from "D at
chance while B was 0.233" (decisive neg -> bank C). 82M cannot. (4) z=-0.95 is
within noise, not a significant miss. 200M = +7.3h (the designed cost); fallback
bank C (v11 additive) is a good landing either way.
**200M FINAL GATE (unchanged, 300 trials): n8@128 > 0.15 AND n8-allctx > 0.263
(B-246M 0.233 + 0.03) -> continue to 500M; flat/at-chance/B-level -> BANK C.**
Battery at 200M: scripts/run_memory_behavioral.py --trials 300 on latest.pt
(=final_model.pt at budget), same preset/config as the 82M battery.
**NEXT-RUN LEVER RESEARCH (2026-08-26, for the post-500M call):**
The oracle said the gap is READ-SIDE routing (query→address). B adds recall
DATA (indirect pressure). Three levers target it more directly, in order of
novelty/effort:
(1) **fact_contrastive read-side loss — the missing half.** The trainer
already plumbs it (v7/train.py:532-541) and it is W12-validated, but v13
model.py LACKS `fact_contrastive_from_lm` (v12/model.py:1775-1807 has it;
v13 has identical `ce_from_lm`/`embed_real`/`embed_imag` so it's a ~35-line
port + 2 config fields + a CLI flag). At each value token it forces the
correct value to outrank the sibling answer tokens — EXACTLY the 8-way
discrimination the probe measures. This is the most direct novel fix for the
routing gap. (Caveat: v12 Phase-0 found it null on an EASY closed-set task;
the probe's 8-way dense ctx128 is HARD, so it may matter here — untested.)
(2) **gamma_floor memory horizon** (v13 cfg, default 0.0 = OFF; v11 used
0.98): keeps state ~50x longer. B's ctx2048 edge (+0.045) suggests longer
horizon could help long-ctx recall. Untested on v13 delta.
(3) **recall-loss weighting / denser curriculum.** The recall slice is
Sparse/long-range (3-6 bindings over 2-200 sentences) while the probe is
DENSE 8-binding @ctx128 — a distribution mismatch. Per-source loss weight
would need source-id threaded through mix→batch→loss (real change, not a
flag). Easier: add a dense-short-ctx recall variant to _build_recall_doc
(v7/data.py:1353) so training matches the probe's hard case.
NOTE: "increase weight of the recall loss" (user's question) ≈ lever (3) but
the higher-leverage moves are (1)+(3-dense) which change WHAT the model is
told to discriminate, not just how hard on the same sparse signal.
**CONTRASTIVE PORT DONE (9e73e7b).** `fact_contrastive_from_lm` ported to v13
(selftest ALL MODES PASS, smoke zero=0/pos=0.661/grad ok). REMAINING to make
it fire on a recall run: pretrain-mix batches carry NO loss_mask (cache is
{input_ids,labels} only), and the trainer branch is guarded by
`loss_mask is not None` (v7/train.py:534). Fix = thread per-token value
masks through _build_recall_doc (return value spans) -> blend interleave
(tuple payload) -> cache build (add value_mask column) -> load_pretrain_mix
-> StackedChunkDataset -> batch. Cost: ONE cache rebuild (~1.5-2h, blocks
launch). DEFERRED until the 500M verdict: if B is borderline (likely), this
is the highest-value next lever; if B is a clear win, scale instead.

- **GATE (re-arm watchdog on every wake).** Kill if loss > 0.7 NLL above r1
  (r1 curve: 7.52@5M, 6.66@10M, 5.87@20M, 4.81@50M, 4.36@100M, 3.97@200M).
  Recall gate at first ckpt (5000 steps): **multi8@128 off 0.133** (chance
  0.125) + CE non-regression; then full battery + Wiki PPL. Re-measure with:
  ```
  .venv/bin/python scripts/run_memory_behavioral.py --model-type v13 \
    --checkpoint <ckpt> --preset v13_e3_k3_selective \
    --context-lengths 128,512,1024,2048 --positions 0,0.5,1 \
    --association-counts 1,4,8 --trials 60 --candidate-count 8 \
    --output logs/memory_probes/<name>_behavior.json
  ```
  Baseline to beat: `logs/memory_probes/v13_500m_r1recipe_FINAL500M_d169584_behavior.json`.
- **Still do NOT:** re-sweep λ/τ/γ_floor/vault-vs-phase (v11 ~1.4B tok
  exhausted it). The old "do not add recall data" item is VOID — it rested on
  the write-interference hypothesis the oracle evidence overturned. (Caveat
  to re-check in B: v11 Stage-3 found *more* recall data hurt there, w3 >
  w10 > w20 — keep the slice small, ~3-6%, on the rich web base.)

Probe commands (CPU-safe alongside training):
`.venv/bin/python v13/tmp/dissect_ckpt.py <ckpt>` and
`.venv/bin/python -m v13.eval_checkpoints --checkpoints <ckpt> --labels wiki --batch_size 2`.

Kill if train loss is > ~0.7 NLL above r1 (e.g. >5.5 at 50M). SIGTERM is
safe (trainer writes `latest.pt`).

## SPEED (LATER)
Do not start this track unless 500M is healthy or the GPU is idle. Profile:
the step is **launch/elementwise bound** (`copy_`/`mul`/`fill_` dominate;
all matmuls+solves ~14% of CUDA). A faster triangular solve will not 5× us.
`[K,B,H,C,C]` mass/decay is why larger batch is *worse* per token.

Each idea gated on `v13/selftest` + ckpt-vs-no-ckpt grads:

1. `--compile_blocks`: **BROKEN 2026-08-23** — Inductor `assert_size_stride`
   on `aten.complex.default` at first step (meta/real layout mismatch).
   `--delta_decay_factored` (K-independent system; `[delta_factored]` PASS)
   is still unbenched.
2. Selective rematerialization: checkpoint CGU/norm, **save** the chunk-solve
   output. Today we recompute the whole block; backward is ~8× forward.
   Most likely honest 1.5–2× with no math change.
3. CUDA graphs / static chunk loop at fixed `(B,T,C)` — tens of thousands of
   elementwise launches are the profiler story.
4. Fused `mass+solve+project` so inductor (or one kernel) sees the whole
   chunk. Do **not** reopen Flash-PAM / Triton custom autograd first; that
   historically lost to `torch.compile` on this codebase.
5. Out-of-box, architecture-preserving: train with the **same O(1) recurrent
   step as inference**, parallelized by an associative scan over
   `S ← γS + (βw v − βe k@S)kᴴ`. No C×C, no UT solve, train ≡ infer by
   construction. Not a Mamba reskin (phase addressing + vault + GSP stay).
   New kernel + new selftest. Only after quality is a real Wiki PPL, or if
   4K tok/s makes 500M unusable.
6. Never: re-detach the block input; `g/mag` backward; claiming 21K;
   training without `[block-grad step1]`.

## ARCHIVE (settled — do not re-open)
- v13 additive path == v11 bit-identical (`test_v11_v13_forward_ab.py`,
  0.000e+00). Not a fork regression.
- `fused_ce` dropped `grad_weight += ...` (65546dc). Head got zero CE grad.
  Restored `79c7cc2`. Rel-L2 5.8e-6.
- 500M NaN: vault δ eigenvalue `γ(1 − βe‖k‖²)` flipped past −1. Fix:
  per-vector key-norm (`04dcebd`) + `delta_erase_beta_cap=0.95` (`03c3ede`)
  + gate-BCE nonfinite guard. Cap alone passed the real-trainer death at
  step 150.
- Ckpt crash: TorchScript `cnormalize_vec` operand-swap (`baaf5b3`). Not
  flaky. `g/mag` autograd.Function RETRACTED (19% key-grad error).
- 21K tok/s / "9×" RETRACTED — measured with `_ckpt_block` detach.
- Pre-`baaf5b3` 50M/100M train-loss verdicts VOID (frozen layers and/or
  dead head). Do not use them to kill or keep a run.
- Warmup-2000 runs and Jun-23 10B log are wrong references. Use r1 below.
- Do not chase HF config val_ppl 6.65 — user: "we never had val ppl 6 for PAM".

## REFERENCE
**v11 round-1** (Jul 1, new code, 75GB, `--no_grad_ckpt --compile`):
`--preset v11_e3_k3_chat --warmup 500 --lr 3e-4 --batch_size 18 --seq_len 2048
Loss: **10.31@2M, 7.52@5M, 6.66@10M, 5.87@20M, 4.81@50M, 4.36@100M, 3.97@200M,
3.96@300M, 3.83@400M, 3.82@500M** (last three verified 2026-08-23 from the
same log; the run goes to ~2B). Verdict gaps should use ±2M window means.
Log: `logs/v11/round1_pretrain_20260701_115022_cbb4dd2_dirty/v11_v11_e3_k3_chat_pretrain_pretrain_mix.log`

WikiText-103 val PPL: **r1 pretrain endpoint 84.57 @ 2B** (log line 1183 —
the fair pretrain-to-pretrain number to match); **stretch < 25.77** (the
WikiText-trained v11_e3_k3 anchor, requires selective stack to contribute);
**ideal ~22.69** (transformer). 500M probe decides match-vs-stretch.

## WAKE PROTOCOL (every wake, all of these)
1. Read this file fully.
2. `tmux ls`; `pgrep -af "v1[13].train"`; GPU; active log last steps + errors:
   `grep -oE "\[1\] [0-9]+ loss=[0-9.]+ .*gtok=[0-9]+" <log> | tail -8`
   `grep -icE "traceback|out of memory|nan" <log>`
3. Re-arm watchdog (timeout 3300 is mandatory):
   `bash v13/tmp/watchdog.sh <active_log> <verdict_gtok> 2940`
   as `async: true` + **timeout: 3300**. Exits early on process death / OOM /
   verdict gtok — that wake is the chain. Keep it alive until V13 is done.
4. Commit verified `v11/` `v13/` `v7/` `scripts/` changes immediately.
5. Act from STATUS + NEXT; yield only with the watchdog armed.

## MONITOR
- GPU: `nvidia-smi --query-gpu=memory.used --format=csv,noheader`
- tmux: `tmux ls | grep v13`
- Launch pattern: `set -euo pipefail; cd /home/gowrav/Development/qllm2;
  export HF_HUB_ETAG_TIMEOUT=120 HF_HUB_DOWNLOAD_TIMEOUT=300;
  exec .venv/bin/python -m v13.train ...` in tmux,
  `| tee -a logs/v13/<name>/tmux_console.log`.
- New CLI (2026-08-23): `--compile_blocks`, `--delta_decay_factored`,
  `--delta_key_norm` / `--no_delta_key_norm`, `--delta_erase_beta_cap`.
