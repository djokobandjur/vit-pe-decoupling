# Final public-package critical-path reconciliation

**Status: PASS**  
**New GPU experiment required: NO**  
**Critical rows reconciled: 18/18**

This reconciliation describes the files actually present in the public
repository. It covers the canonical cohorts, the within-ViT-S training-regime
bridge, the AMP-matched architecture comparison, the all-restart and
direct-displacement audits, the layerwise analysis, the local directional
geometry probe, clean-accuracy provenance, and both reported sensitivity
analyses.

## Headline verification

- Canonical cohort: **72 checkpoints** (48 ViT-B + 24 ViT-S full-FP32).
- Canonical noise-minus-adversarial gap: **72/72 positive** rows.
- All-restart audit: **240/240 restarts**, **48/48 points**, PASS.
- Direct-displacement schedule: best-of-five `200x5` stability PASS.
- Training-regime bridge: rankings preserved; formal rho50 equivalence only
  for Sinusoidal.
- AMP-matched architecture interaction: exact Friedman `p=0.02893519`;
  Learned--RoPE ordering reverses.
- Local geometry: **six original raw JSON files**, **1,584 direction records**,
  and **48 checkpoint groups**; execution-source hashes, raw digests,
  raw-to-processed reconstruction, all 48 checkpoint SHA-256 records, and the
  6/6 independent historical Sinusoidal cross-check all PASS.
- Clean accuracy: all 12 table cells generated from public seed-level
  evaluation anchors.
- Post-lock grid sensitivity: ViT-S Sinusoidal adversarial nAUC
  `0.766366 -> 0.721777`; the headline conclusion is strengthened.
- Training-source scope: the historical ViT-B/16 AMP source is public; the
  production ViT-S/16 full-FP32 and AMP launchers are not redistributed, so
  Level 3 is explicitly partial provenance rather than source-complete
  retraining.

## Public evidence paths

The public package intentionally does not contain a `raw_inputs/v20/`
directory. The compact public evidence is stored under:

- `analysis/canonical_cohorts/reference_outputs/`;
- `analysis/cross_cohort/reference_bundle/` and `reference_outputs/`;
- `analysis/cross_architecture/reference_outputs/`;
- `analysis/precision_bridge/reference_outputs/`;
- `analysis/local_geometry/raw_sources/`, `execution_provenance/`, and `reference_outputs/`;
- `checkpoints/vitb_r32/`;
- `analysis/clean_accuracy/reference_outputs/`;
- `data/objective_specific/`;
- `audit/`.

The complete mapping is
`audit/final_claim_to_evidence_matrix.csv`. Historical internal archive names
are not required to navigate or verify the public package.

## Locked claim boundaries

- The ViT-S/16 AMP--FP16 ALiBi cohort is described as systematically
  divergent / not yielding valid checkpoints. No exact per-seed failure count
  is asserted without raw training logs.
- The primary task-loss AUC uses prespecified canonical native-budget grids.
  The later all-restart budget-0.020 points are reported as a separate
  grid-extension sensitivity.
- Direct-displacement results use a different objective and remain outside
  canonical task-loss AUC support.
- The direct-displacement PASS concerns a best-of-five schedule-level envelope,
  not convergence of every restart or exhaustion of the search.
- Local geometry is a directional diagnostic, not a complete Jacobian
  spectrum or causal localization result. Its execution sources, six original
  raw JSONs, 48 checkpoint identities, and processed outputs are hash-closed.
  The 48-checkpoint manifest was reconstructed post hoc from the retained
  original cohort and independently cross-checked for 6/6 historical
  ImageNet-Sinusoidal digests.
- The reported local functional-gain ratios are finite-radius quantities at
  `r0=1e-4`; their family ordering is preserved at the calibrated operating
  point, but their magnitudes are not scale invariant.
- Training-source coverage is partial: only the historical ViT-B/16 AMP source
  is redistributed.
- Layerwise evidence is descriptive and does not identify heads or tokens.
