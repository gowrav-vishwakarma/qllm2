# v16 learnings: index

Settled 2026-11-01 (the user asked Mind and Lead to discuss). The package code moves into `v16/` **once, at release**,
taking only what survived. Until then each session keeps a **self-contained** learnings file, written so every
experiment can be re-implemented in a new repo **without the old code**:

- the idea, with its maths in two lines;
- the exact recipe;
- the numbers, with params, tokens and MB;
- the verdict and the lesson.

The "code last at <hash>" pointers stay in each version's EXPERIMENT_INDEX while this repo exists; these files carry what
a new repo needs.

| lane | file | covers |
|---|---|---|
| Mind (v13_sempty body) | [learnings/MIND.md](learnings/MIND.md) | body, recipes, scale, rules in context, reach, contrast, dilation, curricula, budgets, process |
| Library (v14) | learnings/LIBRARY.md | library units, fetch organs, PACKAGE.md decisions |
| Lead (v16) | learnings/LEAD.md | package table, T++ + library arm, harness cells and scorers, Lead decisions with reasons |

Release extraction: Lead's `v16/MANIFEST` and export test (the traced files, plus sempyt and the holdout token cache).
