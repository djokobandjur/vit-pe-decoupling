# ViT Positional-Encoding Decoupling v1.0.0 — submission reproducibility release

This release accompanies the Neural Networks submission
“Random and Adversarial Positional-Parameter Robustness Decouple Across
Positional-Encoding Families.”

Authors: Đoko Banđur (corresponding author) and Miloš Banđur.

## Reviewer-audit closure update (2026-08-07)

The release now explicitly separates publicly reproducible evidence from retained historical audit records. The ViT-B/16 harmonization runner and run ledger are distributed, but the original row-level N-to-2N comparison JSONs are not; the manuscript no longer claims otherwise or assigns the unrelated `0.005` evaluation-cross-entropy tolerance to selected attack loss. The retained hardware-audit summary and D033A provenance report are now distributed under `audit/historical/`.

The local geometry section is explicitly scoped to the 48 ViT-B/16 checkpoints. The checkpoint directory now includes the ViT-B/16 48-entry post-hoc digest manifest, a 48/48 canonical-attack path binding, the ViT-S/16 full-FP32 24-entry manifest, and the ViT-S/16 AMP 18-entry manifest. The ViT-S AMP ALiBi limitation is stated only at cohort level because raw per-seed training logs are not public.

## Included

- processed seed-level inputs for all reported analyses;
- scripts that reproduce every reported table and figure without a GPU;
- clean manuscript and supplementary source, including deterministic
  no-envelope and post-lock task-loss-grid sensitivity tables;
- canonical, cross-cohort, training-regime, cross-architecture, all-restart,
  direct-displacement, layerwise, clean-accuracy, and local-geometry analysis
  code and outputs;
- claim-to-evidence, numerical-integrity, and language audits;
- data-split, seed, cohort, numerical-mode, and checkpoint manifests;
- complete R32 geometry provenance: original raw JSONs, execution-source hashes,
  a 48-checkpoint digest manifest, and raw-to-processed verification.

## Verification performed before release

- repository integrity checks: PASS;
- Python tests: PASS;
- complete analysis-level reproduction: PASS;
- exact numerical/table verification, including all 14 generated tables: PASS;
- local geometry full provenance (6 raw JSONs / 1,584 directions / 48 checkpoint digests): PASS;
- post-lock grid sensitivity (all six seed deltas negative): PASS;
- rendered comparison of all four manuscript figures: pixel-identical.

## Not included in Git

ImageNet images and model checkpoint binaries are not redistributed directly
through Git. The processed outputs are sufficient to reproduce all tables and
figures without downloading the model binaries. For full attack reruns, the
exact checkpoints used in the study are available from the shared read-only
Google Drive checkpoint archive linked from the repository `README.md` and
`checkpoints/README.md`, together with checkpoint-level SHA-256 integrity
manifests. Complete retraining requires a licensed ImageNet copy and
substantial GPU resources.

The full raw all-restart and direct-displacement archives are distributed as
a separate release asset because they contain large saved perturbation data.
The compact six-file R32 geometry raw cohort is included directly in the
repository, together with all 48 checkpoint digests (but not checkpoint
binaries).

## Training-source boundary

Only the historical ViT-B/16 AMP training driver is redistributed. The
production ViT-S/16 full-FP32 and AMP launchers are not included, so this
release does not claim complete source-level retraining reproducibility for
those cohorts.

## v6 audit hardening (2026-08-07)

- disclosed non-canonical ViT-S Sinusoidal coverage-extension attempts and their evidence boundary;
- corrected FP32 wording to disclose TF32-permitted later audit mode and historical environment-reconciliation status;
- disclosed empirical, non-a-priori convergence-threshold semantics;
- removed Git-ignored LaTeX build artifacts from the release and added SHA-manifest guards;
- added automatic PDF page-count verification;
- converted main-text Supplement S1/S6/S7 references to cross-document labels;
- documented hash-verified equivalence of the original and re-extracted ImageNet-100 validation copies.

## v8 metadata-consistency hardening (2026-08-07)

The sole residual audit inconsistency was a stale abstract word-count field: the distributed abstract is 237 words by `texcount`, while the v7 public-release JSON incorrectly reported 242. The JSON is corrected to 237, and verifier/tests now guard agreement across audit records; when `texcount` is installed, the verifier also checks the distributed abstract directly. No manuscript text or numerical output changed.
