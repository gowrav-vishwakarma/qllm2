# v17 Experiment Index

| ID | Date | Arms & Seeds | Params / Tokens | Verdict | Code Commit | Log / Artifact |
|---|---|---|---|---|---|---|
| `V17-0` | 2026-11-19 | `POWLM` `sempyt` selftest (`pow_v17`, `pow_sync_only`, `pow_work_only`, `pam_v2`, `gdn_like`, `tpp`) | `31.08M` (`±0.10%`) | **PASS** (causality `0.00`, zero-spin `1.19e-7`, decode parity `3.3e-4`, all 90 grads `>0`) | current | `python -m v17.selftest` |
| `IMB17-R1` | 2026-11-19 | `pow_v17`, `pam_v2`, `gdn_like`, `tpp` × seeds `42, 43, 44` | `31.08M` / `~37.7M` tok per run (`max_train_L=1024`, eval to `4096`) | running | current | `logs/v17_imb17_*` |
