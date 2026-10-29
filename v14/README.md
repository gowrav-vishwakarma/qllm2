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
| `EXPERIMENT_INDEX.md` | one row per idea: status, verdict, run hash, rerun recipe. Read before proposing anything |
| `EXPERIMENTS_V14.md` | lab notebook: the do-not-repeat digest of v4–v13, the never-tested leftovers, and the pre-registered v14 experiments |
| `SCRATCHPAD.md` | handover: what is running, where, what is next |
| `config.py` | `V14Config` = v13_sempty `PAMConfig` + v14 switches (all off by default) |
| `units.py` | v14 units (N1 `PhaseStatesPAM`) and `V14LM` (the v13 LM with units swapped in) |
| `selftest.py` | CPU checks: off == v13, start parity, prefill == decode, gradients |
| `check_layout.py` | the sempyt layout guard for v14 (reuses v13_sempty's checker) |
| `microbench.py`, `run_b0.sh` | B0 memory micro-bench (facts / bind / rebind) and its launcher |
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

N1 and N6 coded and self-tested. B1 (rules, N6's first test) is running on
the 4090; B0 run 2 (memory, N1) follows. N3 is folded into N2 (see
EXPERIMENTS_V14 "N3"). Live state: `EXPERIMENT_INDEX.md` and `SCRATCHPAD.md`.
The notebooks are kept current as things happen (user rule, AGENTS.md).

    .venv/bin/python -m v14.selftest        # CPU, ~1 min
    .venv/bin/python -m v14.check_layout     # must print OK
