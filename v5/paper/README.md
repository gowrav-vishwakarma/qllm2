# V5 Paper (arXiv Package)

This folder contains an arXiv-ready LaTeX manuscript draft for V5.

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
   - Table 2 (`tab:main`)
   - Table 3 (`tab:ablation`)
2. Update author block and affiliations.
3. Verify bibliography entries and citation formatting.
4. Add final seed-averaged numbers and variance.
5. Include final hardware/runtime details.
6. Recompile and verify no warnings that affect output.

## Suggested artifact additions

- training logs (`.csv` or `.json`)
- exact config files used per table row
- script for reproducing each experiment row

