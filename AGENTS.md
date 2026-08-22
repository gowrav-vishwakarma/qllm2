# Agent Rules (qllm2)

## Git: commit after every good code change (user rule, 2026-08-22)
- After ANY verified working change to model/training/inference code
  (`v11/`, `v13/`, `v7/`, `scripts/`), commit it immediately with a message that
  says WHAT changed and WHY, plus the verification evidence (test name + result).
- Never leave verified-good code in the working tree uncommitted. An uncommitted
  edit silently overwrote a committed-good line in `v13/fused_ce.py` (2026-08-22,
  commit 65546dc dropped `grad_weight += ...`), stalling every `--fused_ce` v13
  run 2-3.5 NLL above reference for hours. `git log` on the file would have
  shown the good line one commit back.
- Throwaway/debug scripts under `v13/tmp/` do NOT need commits.
- Never touch the dirty hunk in `v13/train.py` (~lines 445-449, synthetic-source
  resume cursors `skip_docs_map.setdefault`).

## Long training
- ONLY in tmux (`tmux new-session -d`), never hub/bash-managed.
- Keep the watchdog chain alive: `bash v13/tmp/watchdog.sh <log> <verdict_gtok> 2940`
  async + timeout 3300; re-arm on every wake (see v13/SCRATCHPAD.md).
