# Appendix C final claim-language audit

**Status: PASS**

The final main manuscript and supplement were scanned case-insensitively for the locked prohibited formulations and close semantic variants.

## Results

| Pattern | Main | Supplement | Verdict |
|---|---:|---:|---|
| `ceiling` | 0 | 0 | PASS_ABSENT |
| `precision invariance` | 0 | 0 | PASS_ABSENT |
| `precision-invariant` | 0 | 0 | PASS_ABSENT |
| `localization/localisation` | 0 | 0 | PASS_ABSENT |
| `2-4x range` | 0 | 0 | PASS_ABSENT |
| `all points converged` | 0 | 0 | PASS_ABSENT |
| `all restarts converged` | 0 | 0 | PASS_ABSENT |
| `search was exhausted` | 1 | 0 | PASS_ABSENT, PASS_NEGATED_OR_LIMITING |
| `supremum` | 1 | 0 | PASS_ABSENT, PASS_NEGATED_OR_LIMITING |
| `structural upper bound` | 2 | 0 | PASS_ABSENT, PASS_NEGATED_OR_LIMITING |
| `robustness transfers unchanged` | 1 | 0 | PASS_ABSENT, PASS_NEGATED_OR_LIMITING |
| unsupported exact ViT-S AMP ALiBi failure count | 0 | 0 | PASS_ABSENT |
| all-restart audit described as a different objective | 0 | 0 | PASS_ABSENT |

## Context-sensitive retained wording

- `supremum`, `search was exhausted`, and `structural upper bound` occur only inside explicit negations that delimit the D020 claim.
- `robustness transfers unchanged` occurs only in the sentence that explicitly rejects a general AMP-to-FP32 invariance claim.
- No exact occurrence of `ceiling`, `precision invariance`, `localization`, `2-4x`, `all points converged`, or `all restarts converged` remains in the main manuscript or supplement.

- The ViT-S AMP--FP16 ALiBi boundary is stated only at cohort level; no exact per-seed failure count is asserted.
- The all-restart budget-0.020 points are identified as a post-lock same-objective grid extension, not as a different-objective audit.

## v6 evidence-boundary additions

- ViT-S/16 Sinusoidal coverage-extension attempts are described only as ledger-registered, non-canonical support diagnostics; the manuscript explicitly states that the original row-level SINEXT JSONs are not redistributed.
- Numerical-mode wording distinguishes float32 tensor/model dtype and disabled autocast/GradScaler from TF32-permitted matrix multiplication in the later audit lock.
- Historical canonical environment reconciliation is not described as complete; the later numerical-mode lock is not presented as contemporaneous evidence for every canonical run.
- The schedule-development audit is explicitly stated to rely on observed empirical separation rather than an a priori numerical convergence threshold.
- Main-text references to Supplement Tables S1, S6, and S7 use cross-document labels rather than hard-coded table numbers.
- The distinct ViT-B and re-extracted ViT-S ImageNet-100 validation paths are accompanied by identical sample-order and split SHA-256 verification.
