# v14 — new neurons, not new layers

v4 → v13 changed *layers*: how the notebook writes, fades, erases and is read.
Every v13 retention lever failed (index: R1, R2, R3, R4, C1, A2r), and PAM
still trails Mamba-2 / Transformer++ on LM by 0.07–0.11 nats at 3B tokens.
v14 changes the **unit itself**. A neuron today is a weight and a bias, two
numbers frozen after training. v14 asks whether a unit with more internal
structure (phase, several weight bases, its own fast memory, its own rhythm)
learns better at the same size and speed.

## Files

| file | what |
|---|---|
| `EXPERIMENTS_V14.md` | lab notebook: the do-not-repeat digest of v4–v13, the never-tested leftovers, and the pre-registered v14 experiments (index table at the top) |
| `SCRATCHPAD.md` | handover: what is running, where, what is next |
| `NOTES.md` | the 2026-08-22 spatial-neuron design (track S1). Notes only, never built |
| `data/` | untracked scratch from August |

## Ground rules (from the v4–v13 record, see EXPERIMENTS_V14 part A)

- **Framework:** sempyt named axes (`~/Development/sempyt/src`), same contract
  as `v13_sempty` (`check_torch_layout.py` must stay clean). v14 imports the
  v13_sempty body (blocks, PAM, echo, trainer, data) and adds new units under
  `v14/`; it does not fork the trainer.
- **Micro-test first** (V12 finding 11): every idea gets a minutes-long
  micro-bench or a same-batch LM check on the 4090 before any 6000 budget.
- **Judge on holdout**, not train loss: the recall mix weights answer tokens
  ×100, so train loss mostly measures recall training (lesson from M1, 2026-10-29).
- **Control = the current best v13_sempty body** (today: echo, `e1g_echo_a0fab65`),
  same batches, same schedule, param-matched, speed reported.
- **One variable per arm.** Bars are written here **before** the run.
- **Failed paths are removed** and recorded with the last commit that has
  the code (AGENTS rule).
- **Logs:** `logs/v14_<tag>_<gitshorthash>_<YYYYMMDD_HHMM>.log` + the in-file
  run header.
- **Sync:** commit → `git push origin master` (origin = `qllm-private-personal`)
  → `git pull` on the 6000 → launch in tmux.

## Status (2026-10-29)

Docs only. No v14 code yet. First step is B0 (find the wall), then N1–N3.
