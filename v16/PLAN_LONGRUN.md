# The open-ended run (Lead, 2026-11-05 14:30 IST; user: "we can go up to 60 days but can finish in 5; plan dynamic mixes")

Not a calendar. A stable run on the 6000 whose data mix evolves by RULES driven by the milestone readings; it can be stopped and released at any ~10B point. Rates: ~3.3B tokens/day on the 6000 (386M); 5 days ≈ 16B, 30 days ≈ 100B, 60 days ≈ 200B.

## The rules (apply at every 1B milestone; every mix change pre-registered with a MEASURED calibration row; judged two milestones later)
- **R1 Keep what carries the thesis:** the rule-doc block (b2s / proto4k / recall4k, ~1.4 % + fixed shares) never goes below its maintenance share; science shares are held unless a refresh raises them.
- **R2 Ramp, don't jump:** any new source enters by an in-sampler linear ramp over >= 2B tokens (RAMP_WEIGHTS / RAMP_TOKENS); at most one new ramp in flight at a time.
- **R3 Judge by bars, not dates:** a ramp is kept if its target probe rises and no guard bar falls (s8 D >= -0.05, told 2k and held-out wording >= -0.10, needles 16k / clk2 8k >= -0.05, HS/ARC/PIQA >= -0.01, per-source science holdout <= +0.03); if a guard falls at two consecutive milestones, the ramp is frozen at its current share and the next step waits.
- **R4 Release whenever:** a decay branch (0.5B / ~3.75 h for interim releases, 1B for named ones) can be taken at any milestone >= 10B after the last one, or when the target probe plateaus (< +0.02 over 3 milestones); base and chat variants are both tagged by token count on Hugging Face; the paper cites one named checkpoint.
- **R5 Decontaminate before, not after:** every new source gets the 13-gram audit (A27, CPU, ~1 h, Library) on its first 20k docs BEFORE its first step; the drop count goes in the calibration row.
- **R7 No source beyond 4 epochs** (Mind): the [mix] line prints epochs per source; a source near 4 epochs is replaced or enlarged (fineweb sample → the full fineweb-edu); memorised templates would also fake the chat probe.
- **R6 Stop rule:** if two consecutive refreshes move no probe and no benchmark by more than noise, the run is at its size's ceiling; stop (release) and decide on scale.

## The mix schedule (dynamic; first steps named, later ones decided by R3)
1. **Nov 6:** SCI-1 main ends, 1B decay → the paper's checkpoint and release v1 (base).
2. **Chat ramp (first):** branch from the pre-decay `stable_final.pt` at constant LR (re-warming a decayed model cost lg8r); chat ~6 % → ~22 % over 2B with a named INSTRUCTION-FOLLOWING share (strict formats: yes/no, one word, multi-step; public SFT sets, decontaminated); gain from web; science and rule docs fixed (R1). Target probe: Library's simple-chat probe with an instruction-following cell. Chat-variant release at ramp end + 1B (R4).
3. **Next ramps, chosen by the readings (candidates, in the likely order):** more instruction data; more science (new filtered sets); code for science; longer-context documents; a SEQ-8k stage (reach) as a pre-registered option.
4. **Optional stable-2 LR step (8e-4 → 4e-4)** if the holdout slope flattens; pre-registered, not now.
5. **Scale decision whenever R6 fires or the user asks:** keep feeding the 386M vs start a ~1B body.

Budget: the 6000 trains the 386M body at ~37.7k tok/s = ~3.3B tokens/day → ~100B tokens in 30 days, ~200B in 60. SCI-1 is at 9.0B (Nov 5).

## Machines
- **6000** = the run only, paused just for release decays.
- **4090** = the paper's controls at ~103M (ladder, second seed, state budget; ~4-5 h per 1B run) after cx_m2 (~Nov 6 14:00), plus the milestone checks in batched windows.

## (superseded phase text, kept for the record) (each segment is pre-registered by Mind with a MEASURED calibration row before its first step; judged two milestones later)
1. **Nov 6:** SCI-1 main ends, 1B decay → the paper's checkpoint and release v1 (base).
2. **Days 1-10, ~30B tokens: chat ramp (Mind's design, adopted).** Branch from the main segment's pre-decay `stable_final.pt` at constant LR (re-warming a decayed model cost lg8r); the current decay stays the base release. In-sampler linear ramp (RAMP_WEIGHTS / RAMP_TOKENS in v7/data.py): chat ~6 % → ~22 % over the first 2B tokens, then hold; the chat block carries a named INSTRUCTION-FOLLOWING share (strict formats: yes/no, one word, multi-step; public SFT sets, 13-gram decontaminated); the gain comes from web; science shares and the rule-doc block (b2s / proto4k / recall4k) fixed. Measured calibration rows at W0, midpoint, W1; the trainer's [mix] line per val is the run's own ramp record. Bars at each 1B milestone, final at ramp end + 1B: Library's simple-chat probe (with an instruction-following cell) rises; s8 D >= -0.05; told 2k >= -0.10; needles 16k and clk2 8k >= -0.05; HS/ARC/PIQA >= -0.01; per-source science holdout <= +0.03. The chat-variant release = a decay branch at ~ramp end + 1B.
3. **Days 10-30, ~65B tokens: steady run + data refreshes (user: more data).** New sources in measured steps (science, code, conversation, more instruction data), 13-gram decontaminated; one mix change per two milestones at most. Pre-registered option: a SEQ-8k stage (does reach past the trained length grow with it? lg8r at 103M said reset fixes forgetting, reach did not follow; untested at 386M).
4. **Day 30: scale decision.** From the curve: keep feeding the 386M, or start a ~1B body on the 6000 (fewer tokens/day). User's call then.

## Releases
- A cooldown branch every ~10B tokens (~every 3 days): base + chat variant, Hugging Face revision tags by token count. The paper cites one named checkpoint (the Nov 6 decay); later releases are "QLLM-2 at XB tokens".

## Milestone checks (Library's budget, adopted)
- Every 1B: Mind's bare CHECK; Library's LIGHT guard (told twin 2k@100 + R x K 2k@100, ~8 min); Lead's review page.
- Every 5B and at every release: the FULL package suite (R x K 2k/8k, rewrite, recall to 32k with MB held, fork read, told held-out wording) + BABILong with the package at 32k on the 4090.
- All of a day's milestones are checked in ONE batched window (checkpoints are kept), so the 4090's control runs restart once a day, not three times.
- Disk: Library keeps weights-only copies for every 5B + every release and deletes its other copies once their rows are committed; Mind sets the budget for full copies; the 6000 keeps the last 4 milestones with sha-verified d_drive copies.

## Housekeeping that the length forces
- Disk: 6000 at 30 GB free → keep_last 4 milestones + sha-verified d_drive copies, enforced; a d_drive size budget (set by Mind).
- Mix changes only from measured rows; judge after 2B tokens; never cut a skill source below its maintenance share.
- Watchdogs and resumable launchers as now; the local box's crashes never touch the 6000.
- Lineage rows per segment; the paper's numbers.json stays the one source for published numbers.

## Mind: disk budget, the 6000 keep routine, and three amendments (13:58 IST 2026-11-05)

**Full-copy disk budget (d_drive, set by Mind).** d_drive: 916 GB, 156 GB free (HF cache 216 GB, from_6000 75 GB). A FULL milestone
is 4.6 GB, weights-only 1.54 GB, so one full copy per 1B would be 460 GB per 100B: impossible. Rule:
- FULL local copies are kept only for: (i) the newest 2 milestones (the sha-verified backup that lets the 6000 delete them; also
  a release branch's source); (ii) every release's stable source + its decayed branch, for the newest 2 releases; (iii) the
  paper's checkpoint (Nov 6 decay) and its stable_final, forever. Everything older is reduced to weights-only (Library's every-5B set).
- **Cap: SCI-1 full copies ≤ 50 GB; SCI-1 total on d_drive (full + weights-only) ≤ 120 GB.** If d_drive free < 60 GB, prune in
  that order before any new run writes (and the HF datasets cache, 216 GB, is the next lever, by the owner's call).

**The 6000 keep routine.** "keep_last 4" does not fit: 4 × 4.6 + latest + best (9.2) + stable_final (4.6) + a decay branch's
latest + best (9.2) ≈ 41 GB vs 30 GB free. So: **KEEP_LAST = 2 on the 6000**, and after every local CHECK the waiter
sha-compares the newest milestone's d_drive copy and deletes the 6000's older milestones (never latest / best of the current
segment, never stable_final or a release branch until its copy is verified). Floor: free ≥ 30 GB (the chain refuses < 25, a decay
< 12). Old segments' best_model.pt files (stageA, main, cont1-3: 22 GB) move to d_drive (full, kept) and leave the 6000. The 72 GB of
non-SCI-1 tier checkpoints in qllm-private on the 6000 are not mine: Lead's call (most have from_6000 copies).

**Amendment A (new rule R7, data repetition).** At 100B+ tokens at fixed shares, small sources repeat many epochs; the plan has no
cap. Estimates to be measured: tools (~140k docs, ~0.1B tokens) at 4 % = 4B per 100B ≈ 36 epochs; worked (~0.09B) at 2 % ≈ 20 epochs;
fineweb-edu sample-10BT at ~20 % ≈ 3-4 passes per 100B. **R7: no source beyond 4 epochs**: the trainer's [mix] line also prints
per-source epochs; a source reaching its cap is replaced by a fresh source or its share is ramped down (R2); fineweb moves to the
full fineweb-edu before 20B more tokens. Memorized tool / instruction templates would also inflate the chat probe falsely.

**Amendment B (interim release cost).** A 1B decay is 7.5 h of the 6000 (~10 % of its time at one release per ~10B). Interim
releases: 0.5B decays (3.75 h); 1B decays for the named ones (paper, scale decision).

**Amendment C (LR, an option to pre-register, not a change now).** 8e-4 constant for 100B+ tokens: if the blend holdout's slope
over the last 5B falls below its noise, step the stable LR to 4e-4 (a WSD "stable-2" stage), judged by R3's bars.

## Lead decisions 2026-11-05 14:45 IST
- 6000 disk: old SCI-1 segments' best_model.pt + stageA_final (22 GB, moving now) and the non-SCI-1 tier checkpoints (b2p2 / t15 / r3c ..., ~37 GB, paper evidence) move to d_drive sha-verified, then are deleted on the 6000 (expected: 6000 ~89 GB free, d_drive ~97 GB free). NO pruning of the user's HF cache: if d_drive runs short, ask the user directly. KEEP_LAST 2 + sha-gated deletes + 30 GB floor on the 6000.
