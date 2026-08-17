# Changelog

## v8 metadata-consistency hardening (2026-08-07)

- corrected the stale public-release abstract texcount claim from 242 to the actual `texcount` result of 237;
- added cross-record abstract-word-count consistency checks and optional live `texcount` verification;
- added claim gate V20-G30 so the abstract-count metadata regression is machine-guarded.

## v7 final audit hardening (2026-08-07)

- documented `configs/attack_eval_numerical_mode.json` as an annotated copy and linked it to the byte-exact locked numerical-mode artifact and SHA-256;
- made every claim-to-evidence matrix evidence token path-resolvable (with glob support) and added verifier/test enforcement;
- conditioned the MATH-SDPA claim on availability of the `torch.nn.attention` API, explicitly noting that the locked PyTorch 2.8 environment takes that branch;
- added a manual pre-submission repository-visibility check because public GitHub availability is external to offline package verification.

## 1.0.0 reviewer-audit closure update — 2026-08-07

- Corrected the ViT-B harmonization claim to match the publicly redistributed evidence: historical runner and run ledger are public, while original row-level N-to-2N outputs are not.
- Removed the unsupported `0.005 selected attack loss` harmonization threshold wording.
- Added the retained historical seed-42 Blackwell-to-H200 audit summary and named D033A provenance file, with an explicit raw-log availability boundary.
- Scoped the local directional probe explicitly to the 48 ViT-B/16 checkpoints and added the corresponding limitation.
- Replaced the unsupported exact ViT-S AMP ALiBi failure wording across repository Markdown and expanded the verifier guard.
- Published the 18-entry ViT-S/16 AMP checkpoint manifest and documented all three checkpoint-manifest cohorts.
- Removed stale `pending` statements from the locked ViT-S protocol.
- Added a 48/48 canonical ViT-B attack-path to post-hoc checkpoint-digest binding audit and disclosed the post-hoc timing in the manuscript.
- Quantified individual-restart non-convergence for the direct-displacement schedule and added missing Supplement table callouts.

## 1.0.0 — 2026-08-06

- Public, reviewer-facing reproducibility repository prepared for submission.
- Includes analysis-level inputs and scripts for the canonical cohorts, the
  within-ViT-S training-regime comparison, and the AMP-matched
  cross-architecture comparison.
- Includes compact evidence for the all-restart, direct-displacement, and
  layerwise structural audits.
- Includes clean manuscript and supplementary source.

- Added Miloš Banđur as coauthor and retained Đoko Banđur as corresponding author.
- Corrected the auxiliary ViT-S training-precision label to AMP-FP16.
- Added the deterministic no-envelope supplementary sensitivity analysis.
- Added explicit in-text callouts for all central cross-architecture and
  layerwise floats.
- Added public processed evidence and independent verification for the local directional-geometry section.
- Replaced the clean-accuracy table with values generated from the public locked evaluation subsets.
- Added the post-lock ViT-S Sinusoidal budget-0.020 sensitivity; the primary prelocked-grid estimate remains unchanged.
- Reframed the ALiBi trainability statement at cohort level because raw per-seed training logs are not public.
- Made all 14 main and supplementary tables reproducible through versioned table sources.
- Corrected architecture-specific perturbation degrees of freedom and qualified the middle random-noise ordering.
- Reconciled public audit paths and package page counts with the files actually distributed.
- Removed a misleading byte-identical ViT-S training-source duplicate and
  documented that only the historical ViT-B/16 AMP driver is redistributed.
- Propagated the qualified random-noise-extremes wording throughout the paper.
- Added the six retained original R32 geometry JSON files, exact execution
  sources, and a complete 48-checkpoint ViT-B/16 SHA-256 manifest; the full
  raw → checkpoint identity → processed-output chain now verifies.
- Added calibrated-radius sensitivity for the local functional-gain ratio.
- Clarified dropout metadata, gradient clipping, and the ViT-B post-lock
  support boundary.
- Added Đoko Banđur's ORCID to manuscript and release metadata.

## v6 audit hardening (2026-08-07)

- disclosed non-canonical ViT-S Sinusoidal coverage-extension attempts and their evidence boundary;
- corrected FP32 wording to disclose TF32-permitted later audit mode and historical environment-reconciliation status;
- disclosed empirical, non-a-priori convergence-threshold semantics;
- removed Git-ignored LaTeX build artifacts from the release and added SHA-manifest guards;
- added automatic PDF page-count verification;
- converted main-text Supplement S1/S6/S7 references to cross-document labels;
- documented hash-verified equivalence of the original and re-extracted ImageNet-100 validation copies.
