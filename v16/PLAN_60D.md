# The 30-60 day run (Lead, 2026-11-05 14:15 IST; user: "keep training this model for the next 30 or 60 days, plan accordingly")

Budget: the 6000 trains the 386M body at ~37.7k tok/s = ~3.3B tokens/day → ~100B tokens in 30 days, ~200B in 60. SCI-1 is at 9.0B (Nov 5).

## Machines
- **6000** = the long run only, paused just for release cooldowns (~7.5 h per 1B-token decay).
- **4090** = the paper's controls at ~103M (ablation ladder, second seed, state budget; ~4-5 h per 1B run), after cx_m2 (~Nov 6 14:00). The ladder no longer takes 6000 time.

## Phases (each segment is pre-registered by Mind with a MEASURED calibration row before its first step; judged two milestones later)
1. **Nov 6:** SCI-1 main ends, 1B decay → the paper's checkpoint and release v1 (base).
2. **Days 1-10, ~30B tokens: chat ramp (Mind's design, adopted).** Branch from the main segment's pre-decay `stable_final.pt` at constant LR (re-warming a decayed model cost lg8r); the current decay stays the base release. In-sampler linear ramp (RAMP_WEIGHTS / RAMP_TOKENS in v7/data.py): chat ~6 % → ~22 % over the first 2B tokens, then hold; the chat block carries a named INSTRUCTION-FOLLOWING share (strict formats: yes/no, one word, multi-step; public SFT sets, 13-gram decontaminated); the gain comes from web; science shares and the rule-doc block (b2s / proto4k / recall4k) fixed. Measured calibration rows at W0, midpoint, W1; the trainer's [mix] line per val is the run's own ramp record. Bars at each 1B milestone, final at ramp end + 1B: Library's simple-chat probe (with an instruction-following cell) rises; s8 D >= -0.05; told 2k >= -0.10; needles 16k and clk2 8k >= -0.05; HS/ARC/PIQA >= -0.01; per-source science holdout <= +0.03. The chat-variant release = a decay branch at ~ramp end + 1B.
3. **Days 10-30, ~65B tokens: steady run + data refreshes (user: more data).** New sources in measured steps (science, code, conversation, more instruction data), 13-gram decontaminated; one mix change per two milestones at most. Pre-registered option: a SEQ-8k stage (does reach past the trained length grow with it? lg8r at 103M said reset fixes forgetting, reach did not follow; untested at 386M).
4. **Day 30: scale decision.** From the curve: keep feeding the 386M, or start a ~1B body on the 6000 (fewer tokens/day). User's call then.

## Releases
- A cooldown branch every ~10B tokens (~every 3 days): base + chat variant, Hugging Face revision tags by token count. The paper cites one named checkpoint (the Nov 6 decay); later releases are "QLLM-2 at XB tokens".

## Per-milestone checks (every 1B, unchanged)
Mind's bare CHECK; Library's told-rule twin cells + package suite (R x K with fetch, rewrite, recall to 32k with MB held, fork read); Lead's review page; think_eval leak while worked docs are in.

## Housekeeping that the length forces
- Disk: 6000 at 30 GB free → keep_last 4 milestones + sha-verified d_drive copies, enforced; a d_drive size budget (set by Mind).
- Mix changes only from measured rows; judge after 2B tokens; never cut a skill source below its maintenance share.
- Watchdogs and resumable launchers as now; the local box's crashes never touch the 6000.
- Lineage rows per segment; the paper's numbers.json stays the one source for published numbers.
