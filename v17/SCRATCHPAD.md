# v17 Handover & Scratchpad

## Current Focus (2026-11-19)
- Independent outsider Eureka track (`v17`: **Two-Organ O(1) Mind — Memory ≠ Intelligence**), written in `sempyt`.
- `v17.selftest` ALL PASS on CUDA (`causality = 0.00`, `S^1 zero-spin = 1.19e-7`, `prefill == chunked == 1-token O(1) decode`, all 90 parameter tensors non-zero grad, `< 0.11%` parameter parity across all 6 arms at `31.08M` params).
- Running `IMB17-R1`: 3-seed (`42, 43, 44`) equal-parameter, equal-data head-to-head (`pow_v17` vs `pam_v2` vs `gdn_like` vs `tpp`) on RTX 4090.

## Paused Previous 4090 Jobs (Stopped Cleanly at User Request 2026-11-19 17:05 IST)
- `nt_split_s44_main_0e76eac5`: stopped cleanly via `SIGTERM`; `checkpoints_v13_sempty/nt_split_s44_main_0e76eac5/latest.pt` verified intact on CPU load (`--resume auto` will continue seamlessly when resumed).
- Local tmux waiters stopped cleanly: `mindseeds`, `libnatrec2`, `mindnat2`, `mindm48`, `libuserguard2`, `lead_guardwatch`, `lead_dryalarm`.
