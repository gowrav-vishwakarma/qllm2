# v16 learnings: index

Settled 2026-11-01 (the user asked Mind and Lead to discuss). The package code moves into `v16/` **once, at release**,
taking only what survived. Until then each session keeps a short learnings file: the key results and lessons, so the
new repo knows what was tried and why. This repo stays accessible (user, 2026-11-01), so the details and the code
are one commit hash away:

- the idea, with its maths in two lines;
- the exact recipe;
- the numbers, with params, tokens and MB;
- the verdict and the lesson.

Every entry names its commit / log hash, so the full write-up (EXPERIMENTS) and the code can be found here later.

| lane | file | covers |
|---|---|---|
| Mind (v13_sempty body) | [learnings/MIND.md](learnings/MIND.md) | body, recipes, scale, rules in context, reach, contrast, dilation, curricula, budgets, process |
| Library (v14) | learnings/LIBRARY.md | library units, fetch organs, PACKAGE.md decisions |
| Lead (v16) | learnings/LEAD.md | package table, T++ + library arm, harness cells and scorers, Lead decisions with reasons |

Release extraction: Lead's `v16/MANIFEST` and export test (the traced files, plus sempyt and the holdout token cache).
