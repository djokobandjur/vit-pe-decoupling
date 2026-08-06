# ViT Positional-Encoding Decoupling v1.0.0 — submission reproducibility release

This release accompanies the Neural Networks submission
“Random and Adversarial Positional-Parameter Robustness Decouple Across
Positional-Encoding Families.”

## Included

- processed seed-level inputs for all reported analyses;
- scripts that reproduce every reported table and figure without a GPU;
- clean manuscript and supplementary source;
- canonical, cross-cohort, training-regime, cross-architecture, all-restart,
  direct-displacement, and layerwise analysis code and outputs;
- claim-to-evidence, numerical-integrity, and language audits;
- data-split, seed, cohort, numerical-mode, and checkpoint manifests.

## Verification performed before release

- repository integrity checks: PASS;
- Python tests: PASS;
- complete analysis-level reproduction: PASS;
- exact numerical/table verification: PASS;
- rendered comparison of all four manuscript figures: pixel-identical.

## Not included in Git

ImageNet images and model checkpoint binaries are not redistributed. The
processed outputs are sufficient to reproduce all tables and figures. Full
attack reruns require the original checkpoints, and complete retraining
requires a licensed ImageNet copy and substantial GPU resources.

The full raw all-restart and direct-displacement archives are distributed as
a separate release asset because they contain large saved perturbation data.
