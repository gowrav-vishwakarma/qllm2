# v16 — O(1) mind + growing library (the product lane)

Started 2026-10-31 after the pivot (PLAN_MIND_LIBRARY.md). v16 is where the
integrated system lives: the mind (trained in `v13_sempty`), the session library
(organ + learned fetch from `v14`), the planner, and the benchmark harness that
judges the package (MLB-1, spec in `v13_sempty/BENCHMARK_MIND_LIBRARY.md`).

Lanes stay as they are: v13 trains the mind and owns the spec and the release;
v14 builds library units (W0 gate, L1 learned key, N11 planner) and moves the
accepted ones here; the organiser maintains the harness and this plan.

Files (day one): `EXPERIMENTS_V16.md` (lab notebook), `EXPERIMENT_INDEX.md` (one
row per unit: status, verdict, hash, rerun), `SCRATCHPAD.md` (handover).
The harness `scripts/session_index_probe.py` moves to `v16/bench.py` when MLB-1
v1 is fixed (until then it stays put so no path breaks).

Comparison rule (user, 2026-10-31): Transformer++ and Mamba-2 are compared AS
THEY ARE (their existing numbers and runs, full context where it fits); no
library is bolted onto them to make it apples to apples; fairness arms are
optional diagnostics, not the headline. Every row carries params, tokens and
memory in MB.
