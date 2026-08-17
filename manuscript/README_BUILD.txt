Research manuscript source package

Main manuscript:
  pe_robustness_nn_main.tex

Supplement:
  pe_robustness_nn_supplement.tex

Recommended repository build:
  python scripts/build_manuscript.py

Manual build order from manuscript/:
  pdflatex pe_robustness_nn_supplement.tex
  pdflatex pe_robustness_nn_supplement.tex
  pdflatex pe_robustness_nn_supplement.tex
  pdflatex pe_robustness_nn_main.tex
  pdflatex pe_robustness_nn_main.tex
  pdflatex pe_robustness_nn_main.tex

The supplement is built first because the main manuscript uses xr-hyper to
resolve Supplement table labels (including S1, S6, and S7) from
pe_robustness_nn_supplement.aux. The compiled bibliography file
pe_robustness_nn_main.bbl is included, so BibTeX is not required for the
release build.

LaTeX build intermediates (*.aux, *.log, *.out, *.spl, *.synctex.gz) are
intentionally excluded from Git and from release archives. PDFs and clean
source files are retained.
