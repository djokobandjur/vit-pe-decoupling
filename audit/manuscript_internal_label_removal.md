# Submission-facing internal-label removal audit

## Scope

The main manuscript, supplement, captions, table text, LaTeX labels, figure paths, and submission-source filenames were reviewed for internal working identifiers and version labels.

## Reviewer-facing replacements

| Internal identifier | Submission-facing term |
|---|---|
| D019 | all-restart Sinusoidal task-loss audit |
| D020 | direct-displacement audit / direct-displacement optimization |
| D029 | within-ViT-S/16 training-regime comparison |
| D031 | layerwise displacement-profile analysis |
| D059 | AMP-matched cross-architecture comparison |

The factorial-design table now uses descriptive analysis names. Section headings, running text, conclusions, limitations, data-availability text, and all captions were rewritten with the same scientific scope and claim qualifiers.

## Additional cleanup

- The visible reference to the “canonical v19 reproducibility package” was replaced by “the accompanying reproducibility package.”
- Submission-facing LaTeX labels and figure/table filenames were renamed descriptively.
- Internal identifiers remain only in the separate governance, reconciliation, and reproducibility evidence system; they are absent from the reviewer-facing manuscript and submission-source package.
- The final abstract retains the Learned–RoPE ordering, direct-displacement inversion, training-regime bridge, cohort-level ALiBi trainability boundary, and the “Together, these results show…” conclusion.

## Verification

- Internal D-codes in main/supplement source: 0
- Internal D-codes in rendered main/supplement PDFs: 0
- Internal D-codes or version labels in captions: 0
- Submission-facing version labels (V17–V20): 0
- Main manuscript: 45 pages
- Supplement: 5 pages
- Abstract: 243 words by texcount
- Conservative source-token heuristic: 256 tokens; informational only because LaTeX/math notation is split more aggressively than texcount. The Article-limit gate uses the texcount value.
- Final LaTeX warnings: 0
- Ghostscript PDF validation: PASS
- PDF preflight: openable, unencrypted, searchable, non-XFA
- Visual inspection: title/abstract, numerical-mode wording, convergence-threshold wording, coverage-extension disclosure, cross-architecture captions, limitations, and data-availability pages PASS
- The final submission-facing main manuscript uses explicit Supplement table identifiers S1/S6/S7 so the standalone rendered main PDF has fully resolved Supplement references without depending on external cross-document auxiliary state
- Page-count claims are machine-checked against the distributed PDFs by `scripts/verify_repository.py`

No numerical result, confidence interval, statistical test, support boundary, or scientific claim was weakened by this cleanup.
