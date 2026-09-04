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
- Throwaway/debug scripts under `v*/tmp/` do NOT need commits.
- Never touch the dirty hunk in `v*/train.py` (~lines 445-449, synthetic-source
  resume cursors `skip_docs_map.setdefault`).
- When we do some ablation and it is proven to be not good, record in experiments, its maths logic but once it is proven not useful, remove its path from all code of that version, to keep code neat and clean. 
- logs files should be identifeable with name that what version or commit hash made this so we can find which log to see from hundreds of log files. (if we need, last run, we should get by its last modified time or by commit hash if at some specific time) 

## Log naming convention (traceability) (user rule, 2026-09-04)
Every training/benchmark log MUST be named so it can be located months later
without opening it. Canonical pattern (used by `v13_sempty/tmp_wikitext_fair.sh`):

```
logs/<version>_<tag>_<gitshorthash>_<YYYYMMDD_HHMM>.log
# e.g. logs/v13_sempty_wikitext_real_fair_7af42eb_20260903_1628.log
```

- `<version>`   : code family, e.g. `v13_sempty`, `v11`, `v13`.
- `<tag>`       : what the run is — dataset + arm + intent, e.g.
                  `wikitext_real_fair`, `wikitext_complex_fair`, `a1_conv_rung`.
- `<gitshorthash>`: `git rev-parse --short HEAD` **at launch** — the exact code.
- `<YYYYMMDD_HHMM>`: launch timestamp (disambiguates same-commit reruns).
- Commit-worthy result logs (a finished, recorded run) ARE committed with the
  code/docs that reference them, so the number and the code that produced it
  travel together. Smoke/errored/superseded logs are deleted, not committed.
- Every log MUST also carry the in-file header block (`_print_run_header` in
  `v13_sempty/train.py`): full config + args + geometry + params, so the file
  is self-describing even if renamed.

## Remote GPU box — RTX Pro 6000 (96 GB) (user rule, 2026-09-04)
- Big/long training runs go on the RTX Pro 6000 (96 GB VRAM), NOT the local
  4090 (24 GB). Local box is for dev, parity tests, and small smokes.
- SSH: `ssh ubuntu@34.131.203.207` (host `rtx6000pro`).
- Remote repo path: `/home/ubuntu/Development/qllm-private`.
- Git flow: this repo's `origin` is the private remote
  (`git@github-personal:gowrav-vishwakarma/qllm-private.git`). **Push here,
  then `git pull` on the remote box BEFORE launching** so the code matches.
  Typical: (local) commit → push origin → (remote) `cd
  ~/Development/qllm-private && git pull` → launch in tmux.
- Same rules as below apply on the remote: long runs ONLY in tmux, log to a
 file with the naming convention, keep the in-file header block.
- `v13_sempty` imports `sempyt`, which is NOT a pip dep of this repo. On any
 box it must be cloned (`https://github.com/gowrav-vishwakarma/sempyt`, e.g.
 `~/Development/sempyt`) and wired into the venv via a `.pth`:
 `echo ~/Development/sempyt/src > <site-packages>/sempyt_src.pth`. Done on
 the RTX box 2026-09-04. WikiText-103 + gpt2 tokenizer need no HF token.

## Long training
- ONLY in tmux (`tmux new-session -d`), never hub/bash-managed.
- Keep the watchdog chain alive: `bash v13/tmp/watchdog.sh <log> <verdict_gtok> 2940`
  async + timeout 3300; re-arm on every wake (see v13/SCRATCHPAD.md).

## Handover ready
- We run this code on server (rtx pro 6000) and local (rtx 4090) and code is synced using pull push. so always keep scracth pad as handover ready notebook so any time we switch the running context is clear to other agents on server to local or vice versa.