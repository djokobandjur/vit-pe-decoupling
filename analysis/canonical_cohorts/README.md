# Canonical cohort analysis

This pipeline reconstructs the seed-level random-noise and adversarial curves
from 75 raw JSON inputs, generates the primary and wider-range nAUC tables, and
regenerates the primary robustness figure.

The top-level `scripts/reproduce_all.py` stages the inputs in an isolated build
directory and runs this script. To run it directly, place an `inputs/` folder
next to the script and use:

```bash
python canonical_analysis.py --package-dir /path/to/package-root
```


The canonical seed-level output also stores adversarial nAUC before the
cumulative lower envelope. The deterministic supplementary sensitivity is
generated with:

```bash
python build_no_envelope_sensitivity.py \
  --seed-level reference_outputs/seed_level_nauc_v18.csv \
  --output-dir reference_outputs
```

This produces the aggregate CSV, machine-readable audit report, and
Supplementary Table S6 source. No additional model evaluation is required.
