# Option A: Tectonic (brew install tectonic) — self-contained, no MacTeX.
# Option B: latexmk + TeX Live / MacTeX.
.PHONY: pdf pdf-tectonic clean

pdf:
	latexmk -pdf -interaction=nonstopmode main.tex

pdf-tectonic:
	tectonic main.tex

clean:
	latexmk -C main.tex 2>/dev/null || true
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
