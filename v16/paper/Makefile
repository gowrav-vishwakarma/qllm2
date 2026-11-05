all: figures pdf
figures:
	python3 make_figures.py
pdf:
	pdflatex -interaction=nonstopmode main.tex >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main.tex >/dev/null; pdflatex -interaction=nonstopmode main.tex | grep -E "Output written|Error|Warning.*undefined" || true
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc
