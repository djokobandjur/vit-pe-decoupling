# N1–N5 audit closeout

**Status:** `PASS_ALL_FIVE_FINDINGS_CLOSED`  
**Date:** 2026-08-07

| Finding | Status | Closure |
|---|---:|---|
| N1 | PASS | Removed the misleading byte-identical ViT-S training file and limited public training-source claims to the historical ViT-B/16 AMP driver. |
| N2 | PASS | Propagated the qualified “random-noise extremes” wording and added a regression guard against the former full-ranking claim. |
| N3 | PASS | Added six original R32 raw JSONs, retained execution sources, all 48 checkpoint SHA-256 records, raw-to-processed reconstruction, and a 6/6 independent historical digest cross-check. |
| N4 | PASS | Documented finite-radius dependence and verified that the family order is preserved at the calibrated operating point. |
| N5 | PASS | Reconciled dropout context, gradient clipping, post-lock ViT-B support, ORCID metadata, and the abstract limit. |

No new GPU experiment was required. The complete 48-checkpoint manifest was
reconstructed post hoc from the retained original checkpoint cohort; six
ImageNet-100 Sinusoidal hashes independently preserved earlier match exactly.
Checkpoint binaries remain outside Git.
