# Local directional geometry

This directory supports the manuscript's local directional-geometry section.

`reference_outputs/geometry_direction_level.csv` contains 1,584 direction
records: one held-out task-gradient direction and 32 Gaussian directions for
each of 48 ViT-B/16 dataset/family/seed checkpoint groups. The table stores the
finite-radius functional gain, locked probe radius, calibration- and held-out-
subset displacement, cross-entropy change, and damage efficiency.

Run:

```bash
python analysis/local_geometry/verify_geometry_analysis.py
```

The verifier starts from the six retained original raw JSON files in
`raw_sources/`, checks their SHA-256 digests and the embedded hashes of the
redistributed execution sources, reconstructs all 1,584 direction rows and 48
seed summaries, joins every raw checkpoint path to the 48-entry ViT-B digest
manifest, and then verifies the aggregate tables and calibrated-radius
sensitivity.

## Complete provenance chain

- `raw_sources/`: six original paired-seed JSON files, six completion markers,
  original summaries, and their contemporaneous SHA-256 manifest;
- `execution_provenance/`: retained FMLE configuration, source modules, and
  launch/verification notebooks;
- `../../checkpoints/vitb_r32/`: SHA-256 manifest for all 48 retained original
  ViT-B/16 checkpoints (binaries are not redistributed);
- `reference_outputs/geometry_public_provenance.json`: machine-readable chain
  from execution-source bytes and raw JSON digests through checkpoint identities
  to the processed tables.

The complete checkpoint manifest was reconstructed post hoc from the retained
original checkpoint directories. Six ImageNet-100 Sinusoidal checkpoint hashes
that had been preserved independently match exactly (`6/6`), providing an
external integrity cross-check. This timing is disclosed explicitly; it does
not weaken the numerical raw-to-processed reconstruction.

The local functional-gain ratio remains a finite-radius diagnostic at
`r0=1e-4`. Its family ordering is preserved at the calibrated operating point,
while the magnitudes are not scale invariant.
