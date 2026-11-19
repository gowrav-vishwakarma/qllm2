# v17 Experiment Index

| ID | Date | Arms & Seeds | Params / Tokens | Verdict | Code Commit | Log / Artifact |
|---|---|---|---|---|---|---|
| `V17-0` | 2026-11-19 | `POWLM` `sempyt` selftest (`pow_v17`, `pow_sync_only`, `pow_work_only`, `pam_v2`, `gdn_like`, `tpp`) | `31.08M` (`±0.10%`) | **PASS** (causality `0.00`, zero-spin `1.19e-7`, decode parity `2.7e-4`, all 92 grads `>0`) | current | `python -m v17.selftest` |
| `IMB17-R1` | 2026-11-19 | `pow_v17`, `pam_v2`, `gdn_like`, `tpp` (Seed 42 canary) | `31.08M` / `38.2M` tok | **STOPPED AFTER S42 CANARY:** `unseen_rules@0` `pow_v17` `0.900 (+7.86 nats)` > `pam_v2` `0.892 (+4.75)` > `gdn_like` `0.708 (+3.03)` > `tpp` `0.575 (+0.43)`; exposed 25x B2 gradient hijack | `bedcf470` | `logs/v17_imb17_r1_bedcf470_20261119_1752.json` |
| `IMB17-R2` | 2026-11-19 | `pow_v17`, `pam_v2`, `gdn_like`, `tpp` (Seed 42) | `31.08M` / `50.9M` tok | **STOPPED AFTER S42:** `pow_v17` #1 (`0.310`), `unseen_rules@0 = 0.983 (+6.36)` & `unseen_rules@2k = 0.500`; `380k` open-web filler overfit `web_nll` on all arms incl. `tpp` (`0.11` on 2-hop) -> moved to frozen `v14.microbench` | `5677a3ed` | `logs/v17_imb17_r2_5677a3ed_20261119_1816.json` |
| `V17-B1H` | 2026-11-19 | `pow`, `echo`, `tpp` × 3 seeds (`131, 132, 133`) on frozen `v14.microbench` (`two_hop + label_gap + B2`) | `~28.8M` / `6000 x B16` | running | current | `logs/v14_v17b1hp*` |
