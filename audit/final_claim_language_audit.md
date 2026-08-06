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

## Context-sensitive retained wording

- `supremum`, `search was exhausted`, and `structural upper bound` occur only inside explicit negations that delimit the D020 claim.
- `robustness transfers unchanged` occurs only in the sentence that explicitly rejects a general AMP-to-FP32 invariance claim.
- No exact occurrence of `ceiling`, `precision invariance`, `localization`, `2-4x`, `all points converged`, or `all restarts converged` remains in the main manuscript or supplement.
