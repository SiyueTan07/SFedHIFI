# Makefile for SpikDST-Fed ICLR 2026 submission
#
# Usage:
#   make          # build main.pdf
#   make clean    # remove aux/bbl/log artefacts
#   make full     # clean + build (forces full bibtex pass)

MAIN := main
LATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX := bibtex

.PHONY: all clean full

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex iclr2026_conference.bib
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

full: clean all

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk \
	      $(MAIN).synctex.gz
