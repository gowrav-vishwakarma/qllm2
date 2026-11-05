# QLLM-2 paper (draft for arXiv)

- `main.tex`: the paper. Model / architecture names are macros at the top (`\model`, `\arch`); change them in one place.
- `data/numbers.json`: every number in the paper, with the notebook section / run hash / log it comes from. Edit here first.
- `make_figures.py`: draws `figures/*.pdf|png` from `numbers.json`.
- `refs.bib`: references.
- Build: `make` (needs `pdflatex`, `bibtex`, matplotlib). `main.pdf` (the current draft) is committed so it can be read on GitHub; rebuild with `make`.

Rules (same as the write-up): claims only inside an equal-data tier; public models in a reference block, never a verdict; every table carries params and tokens; failed bars are reported as failed; `\todo{}` marks what is still open (final SCI-1 milestones, cont4 readout, second seed, co-authors, public URL).
