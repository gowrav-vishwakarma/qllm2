# v13_sempty handover — fused real-PAM speed work (2026-09-03)

Read this first if you are picking up the real-arm training. Everything below
is committed (`git log --oneline 4740d65..HEAD -- v13_sempty/`); the lab
notebook entry is `EXPERIMENTS_SEMPY.md` → "Speed: fused real arm".

## State of play

* **MIX-3B DONE (2026-09-05, `bbc12e9`): holdout PPL 25.73, recall horizon
  ~200 tokens and SHRINKING with training** (a1 recall 1.00 @ctx128, 0.10 =
  chance @512; ctx512 went 0.35 → 0.10 from step 10k → 80k). `dt_bias` stuck
  at init. Full analysis: `EXPERIMENTS_SEMPY.md` → "Phase 3a". **Program is
  now RETENTION** (ladder R1 dt-spread → R2 vault → R3 delta), then Stage L
  (8K/32K). Complex arm: not revisited (same decay; phase ≠ retention).
* **RUNNING L1 (Stage L-1: T=8192 B=8, dt_spread=8, long mix incl. pg19 /
  fineweb_long / recall_long, 1B tok, tmux `sempty_l1`, commit `b07e384`)**
  — see "Running L1" below. R1 at T=2048 was killed by the user (8K first).
  Checkpoints are now rolling (`KEEP_LAST=1`); stale ckpts pruned (19→4.6 GB).
* **N4 READ-OUT GATE DONE (2026-09-04, run commit `2537782`, code `3c7b9b9`):
  val PPL 22.96** — beats chrono 23.14 at all 16 val points; 0.27 from the
  transformer (22.69). Gate verified content-dependent (doubles the effective
  memory read; co-varies with the Chrono clock). **KEEP; `CHRONO=1 OUT_GATE=1`
  is the reference (22.96).** Record: `EXPERIMENTS_SEMPY.md` → "N4 read-out
  gate". Log `logs/v13_sempty_wikitext_chrono_n4gate_fair_2537782_20260904_1517.log`,
  ckpt `checkpoints_v13_sempty/wikitext_chrono_n4gate_fair_2537782/best_model.pt`.
  A1 short conv → 23.49 FAIL, code removed. **PPL ladder on WikiText is
  saturating; next phase = scale the data (see "Scale plan" below).**
* **N1 CHRONO RUNG DONE (2026-09-04, commit `7b24e44`, RTX Pro 6000): val PPL
  23.14** — beats the 23.81 real baseline by 0.67 at every val point, gap to
  transformer (22.69) now 0.45. sROI KEEP; chrono is the new real-arm
  reference. Full record: `EXPERIMENTS_SEMPY.md` → "N1 Chrono-PAM". Log
  `logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`, ckpt
  `checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt` (on the
  RTX box). **Chrono decode now implemented** (state = `(notebook, clock)`,
  `test_chrono_parallel_vs_recurrent`); `--gen_every` may stay on.
* **Generator for prompt testing:**
  `.venv/bin/python -m v13_sempty.generate --checkpoint
  checkpoints_v13_sempty/wikitext_chrono_fair_7b24e44/best_model.pt --interactive`
  (loads once; type prompts; `/set temperature=0.6 max_tokens=120` retunes;
  blank line quits). One-shot: `--prompt "..."`. Device auto (cuda if present).
* **FAIR RUN DONE (2026-09-04, commit `7af42eb`): real-101M hit val PPL
  23.81** at the reference geometry (T=2048, B=18, 10 ep, 1.18B tok). Beats
  v11 E3-K3 complex (25.77) by ~2 PPL; within 1.12 of transformer (22.69).
  Log `logs/v13_sempty_wikitext_real_fair_7af42eb_20260903_1628.log`,
  ckpt `checkpoints_v13_sempty/wikitext_real_fair_7af42eb/best_model.pt`.
  Full curve + comparison + "is this Mamba?" positioning are in
  `EXPERIMENTS_SEMPY.md` (last two sections). Open gap = recall, not PPL.
* Logging now V11-style: `_print_run_header` emits full config/args/geometry
  at the top of every log; step lines carry `epN/M`; epoch-boundary banners
  print train-loss/ppl/tok/best-val. Log naming convention is in
  `qllm2/AGENTS.md` ("Log naming convention").
* **Next: bigger runs on the RTX Pro 6000 (96 GB)** — more data + chat, and
  scale-up if 100M looks saturated. See "Bigger-run plan" below / the new plan.

## Novel math — N1 Chrono-PAM (content-modulated rotary retention) (2026-09-04)

**Idea, in one line:** make the memory's rotary phase *learned and
input-dependent* instead of fixed RoPE — a "content clock" per head.

**Why it's principled (the derivation).** A complex *rotating* retention
`gamma_t = r_t * e^{i*theta_t}` on the outer-product notebook
`S_t = gamma_t S_{t-1} + v_t (x) conj(k_t)` has closed form
`S_s = sum_{t<=s} (a_s/a_t) e^{i(Phi_s - Phi_t)} v_t conj(k_t)` with
`a` = magnitude product, `Phi_t = cumsum(theta)`. The read
`S_s q_s` shows the `e^{i(Phi_s-Phi_t)}` factor is *absorbed* by rotating
`q_s -> e^{i*Phi_s} q_s`, `k_t -> e^{i*Phi_t} k_t`. That cumulative rotation
is exactly what RoPE does with a *fixed* frequency. So **learned rotating
retention == input-dependent RoPE**, and it folds entirely into q/k — the
fused magnitude-retention kernel is UNTOUCHED (speed preserved). CoPE-style,
but on an associative-memory PAM (novel).

**Real-arm implementation** (`RealPAMLayer._rotate_learned`, behind
`cfg.chrono`): per-head warp `g_t = exp(clamp(W x, +/-3))` scales the per-step
angle; since `inv_freq` is constant in t, `cumsum(inv*g) = inv * cumsum(g)`,
so we warp a per-head clock `tau = cumsum(g)` ([B,H,T]) then `phi = tau (x)
inv_freq`. `W` is zero-init (`warp_proj._zero_init`), so at start `g=1`,
`phi = pos*inv_freq` == **exactly** fixed RoPE. cos/sin in fp32, cast to bf16
for the rotation (keeps retained activations small).

**Status: RUNG DONE — 23.14 vs 23.81 (see State of play). Notes below are
the pre-run record.**
- Parity: `test_chrono_rotary_parity` (selftest, CPU) — chrono@init == baseline
  RoPE bit-for-bit (`max|dlogit| = 0.0`), warp grads flow. All 15 selftests pass.
- Layout: `check_torch_layout` clean (`_rotate_learned` is a declared boundary).
- **Speed gate PASSED**: baseline_real_pm B8 T2048 bf16 on 4090 — baseline avg
  **65.9k** tok/s vs chrono **66.6k** tok/s (equal within noise). Mem +1.6 GB
  (per-head cos/sin x16 layers; recomputed under grad-ckpt, irrelevant at 96 GB).
- Decode: implemented after the run (carried state is `(notebook, clock)`,
  the clock replaces `step_offset`); parity test 6.6e-7 vs chunked.

**LAUNCHED 2026-09-04 06:20Z on the RTX Pro 6000** (tmux `sempty_chrono`,
commit `7b24e44`): `logs/v13_sempty_wikitext_chrono_fair_7b24e44_20260904_0620.log`
(on the remote box, `~/Development/qllm-private`). Header confirms
`'chrono': True`, B18 T2048 32130 steps = 10 ep — identical geometry to the
23.81 baseline. ~75-83k tok/s (vs 45.5k on the 4090), ETA ~4.5 h. The
`[gen @ 8000]` lines will say `failed: ... chrono decode not implemented` —
expected and harmless (try/except; val/ckpt use the chunked path). Remote
setup notes: `sempyt` is NOT a pip dep — it is cloned at
`~/Development/sempyt` and wired via `site-packages/sempyt_src.pth` (same as
local). WikiText/gpt2 need no HF token (public). Watch:
`ssh ubuntu@34.131.203.207 'grep -E "^step|val @" ~/Development/qllm-private/logs/v13_sempty_wikitext_chrono_fair_*.log | tail'`.

**How to run the rung (the "main" run on the RTX Pro 6000):**
```bash
# on the remote box, code already pulled to c-hash below:
cd ~/Development/qllm-private
CHRONO=1 TAG=wikitext_chrono_fair \
  tmux new-session -d -s sempty_chrono "bash v13_sempty/tmp_wikitext_fair.sh"
# identical geometry to the 23.81 baseline (T=2048 B18 10ep 1.18B tok);
# compare best val_ppl vs 23.81 and the wiki recall probe.
```
`tmp_wikitext_fair.sh` now takes `CHRONO=1` (appends `--chrono`). First run on
a fresh box tokenizes WikiText (sl2048 cache) once. Log name carries the commit
+ timestamp (naming rule in AGENTS.md); the in-file header prints `[ladder]
chrono=True`.

**Next rungs (N1 won; stack on `CHRONO=1`, one variable each, same B18 T2048
10 ep, compare vs 23.14).** All keep the scan / are elementwise, see
EXPERIMENTS "Positioning":
1. **N4 read-out gate** (`--out_gate`, coded, commit `3c7b9b9`) — **RUNNING**
   (see below). Memory was underused (`pam_scale` 0.11–0.31); the gate makes
   the read-out per token/head. Speed gate −2.8 %.
2. ~~A1 `--short_conv`~~ — **DONE, FAIL: 23.49 vs 23.14, −17 % tok/s; code
   removed** (EXPERIMENTS "A1 short conv"). The `SHORT_CONV` knob in
   `tmp_wikitext_fair.sh` still needs deleting — do it *after* N4 finishes
   (never edit that script while a run's bash is executing it, see pitfalls).
3. **A3 `--delta`** (coded) — the recall lever; needs a recall-mix run to be
   judged (WikiText-only PPL will not show it; see EXPERIMENTS ledger row 4).
4. **A4 `--cond_mem`**, then **A2 `--n_states`/`--vault`** (weaker sROI).
5. N3 interference-erase / N2 frequency-multiplexed keys if A3 is not enough.
6. Chrono follow-ups for long context (from the 09-04 discussion; not coded):
   **segment clock** `tau_t = r_t tau_{t-1} + g_t` (learned reset = position
   since topic boundary), and **phase wrap** `phi mod 2pi` per frequency (fp32
   clock loses precision past ~1e5 tokens). Neither matters at T=2048.
On the RTX Pro 6000 run these with `GRAD_CKPT=0` (96 GB; the ckpt default is
a 4090 fit). **Measured 2026-09-04** (chrono, B18 T2048, 60-step smoke):
grad-ckpt OFF = **~102k tok/s, peak 29.1 GB** vs ON = 83k tok/s, 8.9 GB —
+23 %, a 10-epoch rung drops from 4.0 h to ~3.25 h. Keep B=18 T=2048 for
comparability. Launch template for the next rung — **put the env vars INSIDE
the tmux command string**:
```bash
tmux new-session -d -s sempty_n4 \
  "GRAD_CKPT=0 CHRONO=1 OUT_GATE=1 TAG=wikitext_chrono_n4gate_fair bash v13_sempty/tmp_wikitext_fair.sh"
```
**PITFALL (bit us 2026-09-04):** `VAR=x tmux new-session ...` only works when
it *starts* the tmux server. Once a server exists, new sessions inherit the
*server's* environment, so the env prefix is silently ignored and you launch
whatever the first session's env said (we got a duplicate A1 instead of N4).
**Do not run two rungs concurrently** on the 6000: measured 39k+39k = 78k
tok/s combined vs 85k single (−8%); queue instead
(`while pgrep -f <ckpt-dir-tag>; do sleep 60; done; ...` in the tmux command).

**More pitfalls (2026-09-04 afternoon, both cost GPU time):**
* A `while pgrep -f '<tag>'; do sleep; done` waiter inside `tmux new-session
  "..."` matches **its own** `bash -c` command line → waits forever (N4 sat
  queued 16 min on an idle GPU). Use `pgrep -f 'python.*<tag>'` or a pid file.
* **Never edit a `.sh` while a run's bash is executing it.** bash reads the
  script incrementally; the edit shifted the offset and the wrapper ran
  garbage after Python returned (`ag_every: command not found`, `exit=127` in
  the A1 log; training itself was fine). Fix pending: make
  `tmp_wikitext_fair.sh` copy itself to `mktemp` and `exec` the copy.

`tmp_wikitext_fair.sh` now `exec`s a private temp copy of itself (safe to edit
while a run is live) and the `SHORT_CONV` knob is gone.

**Done 2026-09-04 on the RTX Pro 6000:** N1 chrono 23.14 (KEEP) → A1 short
conv 23.49 (FAIL, removed) → N4 gate 22.96 (KEEP). Reference = `CHRONO=1
OUT_GATE=1`.

## Scale plan (Phase 3, decided 2026-09-04) — data, not more WikiText rungs

Why now: three rungs gave −0.67 / +0.35 / −0.18; WikiText-103 PPL at 100M is
saturating 0.27 from the transformer. The open questions (does the arch hold on
diverse data at 3× the tokens; can it be trained to recall; can it chat) need
diverse data. Remaining ladder rungs (A3 delta, A4 cond-mem, A2) are *recall*
levers and will be ablated later on the mixed recipe with the recall probe.

1. **Base pretrain (RUNNING, see below):** `v13_sempty/tmp_pretrain_mix.sh` —
   real-102M `baseline_real_pm` + chrono + out_gate, live stream
   dclm-edu .45 / fineweb-edu .42 / smoltalk2-Mid as ChatML text .10 /
   synthetic recall .03 (web-only first 300M tok), **3.0B tokens**, B18 T2048
   (81,380 steps), lr 2e-4 warmup 1000 cosine, wd 0.01, **dropout 0**, bf16,
   grad-ckpt off, **chat vocab 50261** (`<|im_start|> <|im_end|> <think>
   </think>` = ids 50257–50260, default-on for `--dataset mix`). Primary metric
   = streaming holdout val (244 chunks); `[wiki_val]` is a secondary anchor and
   is NOT comparable 1:1 with 22.96 (different data, single pass — expect it to
   sit higher). 86.5k tok/s live (tokenization on the main thread, −13% vs the
   cached loader) → ~9.7 h.
2. **SFT** on smoltalk2 `SFT` (v7 `load_smoltalk2`, assistant-only loss, ChatML)
   from the best base ckpt — needs a v13_sempty SFT entry point (not written).
3. **Recall probe + A3 `--delta` ablation** on the mixed recipe (short runs).
4. If (1) is healthy: **scale params** (~350M) on more tokens.

Judging (1): holdout val PPL should fall monotonically; the wiki anchor should
keep improving through the blend switch at 300M tok (step ~8,140); recall docs
enter at that point too. Watch `[diag]` pam/gate as before. Sample prompts via
`generate.py` on `checkpoints_v13_sempty/mix3b_chrono_gate_303c7fb/best_model.pt`
(the tokenizer is the chat one — `_config_from_ckpt` reads vocab from the ckpt).
HF auth: a token is stored on the RTX box at `~/.cache/huggingface/token`
(mode 600, never in the repo or logs; 2026-09-04) — the first launch streamed
unauthenticated for 4 min and was restarted so the 10 h stream has the higher
rate limit. `huggingface_hub` picks the file up automatically.

**Running (2026-09-04 19:15Z, RTX Pro 6000):** tmux `sempty_mix`, commit
`bbc12e9`, log `logs/v13_sempty_mix3b_chrono_gate_bbc12e9_20260904_1915.log`,
ckpt dir `checkpoints_v13_sempty/mix3b_chrono_gate_bbc12e9/` (`latest.pt`
every 1000 steps = full resume state, `best_model.pt` on holdout-val best,
`step_XXXXXX.pt` every 10k). 85.5k tok/s → **ETA ~10:30Z 2026-09-05**.
Watchdog: `bash v13_sempty/tmp_wiki_watchdog.sh <log> 81380 2940` (re-arm on
wake). Two earlier launches today (`303c7fb` 18:53/18:56) were killed at
<600 steps and their logs deleted: no resume support yet and the blend warmup
was silently inactive (see below).

**R1 at T=2048 (`b55c49a`, 06:12Z) was killed by the user at <1k steps** —
priority moved to 8K context; log + ckpt dir deleted (no result). The R1
question (does a per-head decay ladder lengthen the horizon?) is now asked
directly at T=8192 — see "Running L1" below.

**Running L1 = Stage L-1 (2026-09-05 06:36Z, RTX Pro 6000):** tmux
`sempty_l1`, commit `b07e384`, log
`logs/v13_sempty_mix1b_8k_r1_dtspread8_b07e384_20260905_0636.log`, ckpt
`checkpoints_v13_sempty/mix1b_8k_r1_dtspread8_b07e384/`.
`SEQ=8192 BATCH=8` (65,536 tok/step, GRAD_CKPT=0 → ~55 GB) `DT_SPREAD=8`
`TARGET_TOKENS=1e9` (15,259 steps) `BLEND_WARMUP=100M WARMUP=500`
`SOURCES=dclm,fineweb_long,pg19,smoltalk2_mid,recall,recall_long`
`WEIGHTS=0.36,0.20,0.22,0.10,0.04,0.08` `VAL_EVERY=1000 SAVE_EVERY=500
KEEP_EVERY=5000 KEEP_LAST=1 GEN_EVERY=2000`. Two things differ from mix-3B
(T and dt_spread) — deliberate: the reference heads (half-life ~38 tok) cannot
use an 8K window at all, so "8K without spread" is not an informative arm.
Measured at step 100: **81k tok/s, 52.4 GB** (2K run was 85.5k — the sequence
axis is free, as the scan predicted) → **ETA ~10:05Z 2026-09-05**. Startup is
slow (~4 min to step 1): the 10k-chunk shuffle buffer is 82M tokens at 8K.
Watchdog: `bash v13_sempty/tmp_wiki_watchdog.sh <log> 15258 2940` (armed
06:43Z; re-arm on wake).
**Judge L1 by the horizon, not PPL:**
```bash
.venv/bin/python scripts/run_memory_behavioral.py --model-type v13_sempty \
  --checkpoint checkpoints_v13_sempty/mix1b_8k_r1_dtspread8_b07e384/best_model.pt \
  --preset baseline_real_pm --context-lengths 128,256,512,1024,2048,4096,8192 \
  --positions 0,0.5,1 --association-counts 1,4,8 --trials 20 \
  --output logs/memory_probes/v13_sempty_mix1b_8k_r1_dtspread8_b07e384_behavioral.json
```
Reference (mix-3B, 3× the tokens, T=2048): a1 1.00 @128, 0.35 @256, 0.10
@512/2048; a8 0.2–0.35. KEEP if a1 stays ≥0.5 at 512–4096 with holdout PPL
not worse than mix-3B at 1B tokens (val @27k ≈ 32.4 mid-cosine; holdout val
at T=8192 is the same 5 % corpus re-chunked, `val_chunks` differs). Also read
the `[diag]` `dtbias=` row: did the long heads keep their −8…−12 or get
pulled up? Then R2 (`NSTATES=2 VAULT=1 GEN_EVERY=0`), R3 (`DELTA=1
GEN_EVERY=0`) — each 1B at T=8192, sequential, never concurrent.

**Checkpoint policy (2026-09-05, `b07e384`, user rule: free stale ckpts).**
`--keep_last N` (launcher `KEEP_LAST`, default 1): after each milestone
`step_XXXXXX.pt` only the newest N survive; `latest.pt` is an atomic
overwrite (`.tmp` + `os.replace`), `best_model.pt` untouched. Steady state per
run = 3 × 1.2 GB. Pruned 2026-09-05: mix-3B `step_0*.pt` (8 × 1.2 GB), the A1
FAIL dir, `latest.pt` of the finished WikiText runs → `checkpoints_v13_sempty`
19 GB → 4.6 GB. `checkpoints_v11_*` (27 GB, mostly `recall_stage6` 11 GB) NOT
touched — a different program; user to decide.

## Stage L — 8K then 32K context (data landed 2026-09-05 `b07e384`; L1 running)

Measured: the scan is linear — T=8192 B=4 118k tok/s / 26 GB, T=32768 B=1
98.6k tok/s / 26 GB on the 6000 (synthetic vocab; real CE adds cost but the
sequence axis is free). Real smoke `b07e384`: T=8192 B=4 real CE + 5 sources
= 29 GB, so B=8 at 8K fits easily; 32K B=2 should too.
1. **Recall curriculum gaps — DONE.** `_build_recall_doc(rng, max_gap_sentences)`;
   registry `recall_long` = 500 sentences → docs up to ~5.2k tok (`recall`
   stays 200 → ~2.3k). For 32K add `recall_xlong` (~2500 sentences) the same
   way (one registry line). Needle-style single-binding doc: still TODO.
2. **Long documents — DONE.** `pg19` (`emozilla/pg19` parquet stream; the
   `deepmind/pg19` script repo is dead in datasets≥3; books cut into ≤400k-char
   pieces ≈110k tok at paragraph breaks, 44 s first row) and `fineweb_long`
   (fineweb-edu `min_chars=12000`, 14–17k-char docs). Both `kind='web'` →
   part of the blend warmup pool. All sources now open through one
   `_open_source_iter` (live blend and token-cache builder share it). Still
   TODO: whole code files (StarCoder), smoltalk2 `longalign`.
3. **Probe.** `run_memory_behavioral.py --context-lengths ... 8192,32768`
   (check `build_example` filler scaling and GPU time at 32K; 20 trials × 3
   positions × 3 counts × 7 lengths).
4. **Training geometry.** L1 uses T=8192 B=8 = 65k tok/step (memory allows;
   fewer, larger steps). For 32K: B=2 = 65k tok/step. Same lr schedule.
   `max_seq_len` follows `--seq_len` (RoPE cache, Chrono cumsum both
   length-agnostic); holdout val re-chunks at the new T (61 chunks @8K).
5. **Sequence:** R-winner at T=2048 (1B) → **midtrain T=8192** from that ckpt
   via `--resume` semantics? No — `--resume` restores the optimizer/LR for the
   *same* run; for a length change use a fresh run that loads weights only
   (needs a `--init_from <ckpt>` flag: weights, not optimizer/step). Then
   32K the same way. Budget ~1B tokens per stage (3–4 h each).
6. **Retention prerequisite:** the head ladder must reach 32K half-lives —
   spread 8 does (23k/113k for heads 4–5). If R1 shows the optimizer pulling
   the long heads back up, pin the slowest head (vault-style) before Stage L.

**Resume (2026-09-04, `bbc12e9`).** `--resume auto|<path>` restores model,
optimizer, LR schedule, AMP scaler, step/token counters, best-val, all RNG
streams and the **data-stream cursor** (per-source docs consumed + tokens
yielded, saved in every ckpt); each source skips its consumed docs on resume so
nothing is trained on twice (the ≤10k buffered chunks ≈20M tok are lost per
restart, not repeated). The launcher runs `--resume auto` in a retry loop
(MAX_RETRIES=5), so a crash self-heals inside tmux. **If the box itself
rebooted:** `tmux new-session -d -s sempty_mix "TAG=mix3b_chrono_gate
CKPT_DIR=checkpoints_v13_sempty/mix3b_chrono_gate_bbc12e9
RESUME_LOG=logs/v13_sempty_mix3b_chrono_gate_bbc12e9_20260904_1915.log bash
v13_sempty/tmp_pretrain_mix.sh"` (same commit checked out). Verified on the
real stream: killed at step 30 → resumed with the identical LR at 31/32,
finished exactly on budget. `[resume]` lines in the log show the cursor.
**Bug found on the way:** `blend_warmup_tokens` gates on `sum(token_counters)`
inside `v7._blend_interleave_text_iters`, and `v13_sempty/train.py` never
passed the counters → warmup was a no-op in the 18:56 launch. Now passed;
verified (0 chat/recall docs before the threshold). Note the ~20M-token lag:
the interleaver measures *yielded* tokens while the shuffle buffer holds 10k
chunks ahead, so the blend switch lands ≈300M+20M tokens in.
Data/scale-up (DCLM/FineWeb mix via `--dataset pretrain_mix`, `c7a343b`) comes
*after* the ladder settles the architecture at 100M.

* Real-101M (`baseline_real_pm`) trains at **~64k tok/s** (was 5.8k), peak
  **7.6 GiB** at B32 T256 with grad-checkpointing OFF (was 21 GB at B8 with
  it on). One WikiText-103 epoch (118M tok) ≈ **30 min**.
* The one-epoch T256 run **finished** (commit `7343201`): best **val PPL
  54.89** (NLL 4.005), avg 64.1k tok/s, exit 0.
  `logs/v13_sempty_wikitext_real_7343201_20260903_1449.log`,
  checkpoints `checkpoints_v13_sempty/wikitext_real_fused_7343201/`
  (`best_model.pt`, `latest.pt`).
* **Now executing the quality program** (plan
  `.cursor/plans/v13_sempty_quality_program_7759901a.plan.md`): fair 10-epoch
  T=2048 WikiText baseline, complex arm at kernel speed, recall infra, and an
  architecture ladder (short conv, multi-state+vault, delta erase/write,
  Engram-style conditional memory, layout). See that plan for the full spec;
  results land in `EXPERIMENTS_SEMPY.md`.

## How to watch

```bash
tmux attach -t sempty_wiki                 # live; Ctrl-b d to detach
LOG=$(ls -t logs/v13_sempty_wikitext_real_*.log | head -1)
grep -E "\[val @" "$LOG" | tail            # val NLL/PPL every 500 steps (~4M tok)
grep -E "^step " "$LOG" | tail -3          # loss / tok/s / ETA
# watchdog (wakes on: process gone, error/nan in log, step reached, timeout):
bash v13_sempty/tmp_wiki_watchdog.sh "$LOG" 14400 2940
```
Re-arm the watchdog on every wake (rule in AGENTS.md). A healthy run shows
~67k tok/s, `GPU 1.3/7.6GB`, loss falling (6.2 at step 300, expect ~4.3 at
the end: the 09-01 run reached train NLL 4.38 / val PPL 68.75 at one epoch
with constant lr 5e-5 and B8).

## How to run (again / longer)

`v13_sempty/tmp_wikitext_real.sh` is the whole recipe — edit flags there.
```bash
tmux new-session -d -s sempty_wiki "bash v13_sempty/tmp_wikitext_real.sh"
```
Current flags: `--batch_size 32 --seq_len 256 --steps 14400 --lr 1e-4
--warmup_steps 100 --amp_dtype bf16 --fused_ce --fused_pam --ce_gemm_dtype
auto --val_every 500 --save_every_steps 2500`, no `--gradient_checkpointing`.
* **Recipe change vs 09-01 you should know about:** batch 4x (B8→B32) and
  lr 5e-5 → 1e-4 (sqrt scaling); `--steps 14400` = one epoch so warmup-cosine
  completes at the epoch end (09-01 had a 200k horizon = constant lr). If you
  want a literal repeat of the old regime use `--lr 5e-5 --steps 200000`.
* **Longer runs:** `--epochs N` (trainer flag) with `--steps` = N × 14409 so
  the cosine spans the whole run. Memory headroom is large: B64 T256 fits in
  11.9 GiB but tok/s *drops* (63k), B96 57k — stay at 8192 tok/step.
  `--seq_len 512 --batch_size 16` is the same tok/step at 68.5k tok/s.
* Kill switches: `--no_fused_pam` (plain-torch scan, same math, 2.4x slower
  kernel), `V13S_KERNEL=0` env (disables both Triton PAM scan and Triton CE),
  `--ce_gemm_dtype fp32` (exact head, +10 ms/step).
* Log names carry commit hash + timestamp; the header line prints
  `commit=<hash>[-dirty] fused_pam=… ce_gemm=… grad_ckpt=…`.

## What was done (for the record)

1. `triton_kernels.py` — real PAM read in chunked linear-attention form
   (`y_s = a_s (S_in.q_s) + Σ_{t≤s} (a_s/a_t)(q_s.k_t) v_t`), Triton forward
   (state scan + read) and backward (state-grad scan, dq/dk/dg, dv), tile 64,
   8 warps; `pam_scan_torch` fallback; `fused_real_pam_read` entry.
   `RealPAMLayer._chunked` hands off to it; decode `_stepwise` and the complex
   arm untouched. Carried state layout unchanged (rows = value, cols = key).
2. `pam_kernel_test.py` — parity harness (oracle anchored to `_stepwise`;
   the inherited oracle had `outer(k,v)` transposed — fixed). PASS fp32 5e-5 /
   bf16 3e-2. `selftest` 13/13 incl. `test_real_fused_kernel_parity`.
3. `fused_ce.py` — `gemm_dtype` (bf16 GEMMs, fp32 loss) and the Triton
   Liger-style `_FusedLinearCETriton` (grads in forward, one row pass).
   Validation pinned to the fp32 head.
4. `train.py` — `--ce_gemm_dtype`, `--fused_pam/--no_fused_pam`, commit hash
   in the header. `check_torch_layout.py` — new boundaries declared.

## Open items / ideas not taken

* Remaining 61 ms step: Linears + CE GEMMs at tensor-core peak; ~20 ms of
  small elementwise (norm/gate/residual/RoPE, 900 `mul` launches/step) —
  only `torch.compile` would fuse these and sempyt's Dim identities trip its
  recompile limit (needs a sempyt change). RoPE fusion measured at 3 ms max.
* Why tok/s drops beyond 8192 tok/step is unexplained (not memory).
* The one-epoch T256 result (54.89) is recorded in `EXPERIMENTS_SEMPY.md`
  → "Speed: fused real arm". The apples-to-apples number is the Phase-1
  fair run (10 epochs, T=2048); this scratchpad and superseded `tmp_*`
  scripts get cleaned up at the end of the quality program (Phase 5).
