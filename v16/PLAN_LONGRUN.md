# The open-ended run (Lead, 2026-11-05 14:30 IST; user: "we can go up to 60 days but can finish in 5; plan dynamic mixes")

Not a calendar. A stable run on the 6000 whose data mix evolves by RULES driven by the milestone readings; it can be stopped and released at any ~10B point. Rates: ~3.3B tokens/day on the 6000 (386M); 5 days ≈ 16B, 30 days ≈ 100B, 60 days ≈ 200B.

## The rules (apply at every 1B milestone; every mix change pre-registered with a MEASURED calibration row; judged two milestones later)
- **R1 Keep what carries the thesis:** the rule-doc block (b2s / proto4k / recall4k, ~1.4 % + fixed shares) never goes below its maintenance share; science shares are held unless a refresh raises them.
- **R2 Ramp, don't jump:** any new source enters by an in-sampler linear ramp over >= 2B tokens (RAMP_WEIGHTS / RAMP_TOKENS); at most one new ramp in flight at a time.
- **R3 Judge by bars, not dates:** a ramp is kept if its target probe rises and no guard bar falls (s8 D >= -0.05, told 2k and held-out wording >= -0.10, needles 16k / clk2 8k >= -0.05, HS/ARC/PIQA >= -0.01, per-source science holdout <= +0.03); if a guard falls at two consecutive milestones, the ramp is frozen at its current share and the next step waits.
- **R4 Release whenever:** a decay branch (1B, ~7.5 h) can be taken at any milestone >= 10B after the last one, or when the target probe plateaus (< +0.02 over 3 milestones); base and chat variants are both tagged by token count on Hugging Face; the paper cites one named checkpoint.
- **R5 Decontaminate before, not after:** every new source gets the 13-gram audit (A27, CPU, ~1 h, Library) on its first 20k docs BEFORE its first step; the drop count goes in the calibration row.
- **R6 Stop rule:** if two consecutive refreshes move no probe and no benchmark by more than noise, the run is at its size's ceiling; stop (release) and decide on scale.

## The mix schedule (dynamic; first steps named, later ones decided by R3)
1. **Nov 6:** SCI-1 main ends, 1B decay → the paper's checkpoint and release v1 (base).
2. **Chat ramp (first):** branch from the pre-decay `stable_final.pt` at constant LR (re-warming a decayed model cost lg8r); chat ~6 % → ~22 % over 2B with a named INSTRUCTION-FOLLOWING share (strict formats: yes/no, one word, multi-step; public SFT sets, decontaminated); gain from web; science and rule docs fixed (R1). Target probe: Library's simple-chat probe with an instruction-following cell. Chat-variant release at ramp end + 1B (R4).
3. **Next ramps, chosen by the readings (candidates, in the likely order):** more instruction data; more science (new filtered sets); code for science; longer-context documents; a SEQ-8k stage (reach) as a pre-registered option.
4. **Scale decision whenever R6 fires or the user asks:** keep feeding the 386M vs start a ~1B body.

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
