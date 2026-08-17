# Reproducing the canonical cross-cohort analysis

From the public repository root, run:

```bash
python scripts/reproduce_all.py
```

The generated cross-cohort outputs are written to
`artifacts/reproduced/cross_cohort/`.

The primary cross-architecture analysis uses the prespecified canonical
native-budget grids and integrates only over their measured common-support
intersection. The later all-restart ViT-S Sinusoidal budget-0.020 points use
the same task-loss objective but were collected after the canonical grid was
locked; they are reported separately through
`analysis/cross_cohort/build_postlock_grid_sensitivity.py`. Direct-displacement
points use a different objective and remain outside canonical task-loss nAUC.
