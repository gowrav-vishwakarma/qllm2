# v16 experiment index (one row per unit)

| id | what | status | verdict | hash | rerun |
|---|---|---|---|---|---|
| ORG-1 | frozen PAM/T++/Mamba-2 + non-learned library (organ, frozen address) 2k-400k | done | state-only at floor past ~8k (all three); organ 1.0; frozen address 0.6-0.9 to 128k, 1/4 at 400k; boundary ~0 | a1baefc..9f401d5 | `v16/bench.py` (was scripts/session_index_probe.py; EUREKA 2026-10-31) |
| W0 | surprise gate as library write gate (v14, eval only) | pre-registered | | | EXPERIMENTS_V14 "L" |
| L1 | contrastive key/query heads on frozen c1ctlb (v14) | pre-registered | | | EXPERIMENTS_V14 "L" |
| SL0 | shared knowledge library = kNN-LM over the frozen 3B PAM state (Mind) | done | FAIL rules 1+2: 20M rows -0.012 web vs bar -0.05, +0.10 behind Mamba-2; plumbing / cost table | see EXPERIMENTS_SEMPY "SL0 verdict" | logs/diag/v13_sempty_sl0_* |
| SKP-1 | skill-pack persistence, one read at t0 (Mind, scripts/run_skillpack_probe.py 53d0e52) | done | PAM carries a rule read once across 2k (0.86 vs T++ 0.03 / Mamba-2 0.06; 2 seeds, 100 docs); all fade by 8k; R7@32k FAIL | see EXPERIMENTS_SEMPY | logs committed by Mind |
| LAW-1 | one-read reach vs training length: lg2/lg8/lg32 fine-tunes with long-gap rule docs | pre-registered (7527ea3) | | | 6000 after s8 or in r3c's slot |
| L1 | burst-pooling key head on the mind's centred block-8 state (Library) | pre-registered (EXPERIMENTS_V14 "L1 proper") | bars on C5 burst rows: >= 0.9 @32k, >= 0.75 @128k, within 0.05 of the C2 per-token ceiling | | v16/bench.py --key_fn |
| A4F | hashed n-gram table (A4 cond_mem, +33.5M sparse params) on the 600M mind (Mind) | done | in between: holdout -0.026 (met), PG-19 new-token -0.0275 vs bar -0.03 (missed); Engram-class knowledge, cost table, not in the package | d338469 | EXPERIMENTS_SEMPY "A4 cond_mem filter a4f result" |

