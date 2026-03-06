# V5 Paper (Working Preprint Package)

This folder contains the current LaTeX preprint draft for V5.
It is a working manuscript package, not yet a finalized arXiv submission.

## Files

- `main.tex` - full manuscript
- `references.bib` - bibliography

## Build locally

```bash
cd v5/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Before submission checklist

1. Replace placeholder metrics in:
   - any remaining draft-only placeholders
   - any ongoing-run numbers that should be finalized before submission
2. Update author block and affiliations.
3. Verify bibliography entries and citation formatting.
4. Add final seed-averaged numbers and variance.
5. Add matched real-valued / transformer baselines if they are intended as core claims.
6. Include final hardware/runtime details.
7. Add final figures/tables and appendix cleanup.
8. Recompile and verify no warnings that affect output.

## Suggested artifact additions

- training logs (`.csv` or `.json`)
- exact config files used per table row
- script for reproducing each experiment row
- explicit commit hashes for every reported run

