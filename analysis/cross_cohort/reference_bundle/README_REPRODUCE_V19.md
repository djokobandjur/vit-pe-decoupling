# Reproducing the CANONICAL v19 analysis

From the package root:

```bash
python analysis/v19_support_aware/build_v19_full_analysis.py \
  --v18-outputs analysis/v18_canonical_pipeline/outputs \
  --vits-root raw_inputs/v19/vits_in100_full_robustness_v1 \
  --output-dir analysis/v19_support_aware/rebuilt_outputs

python analysis/v19_support_aware/generate_v19_figures.py \
  --output-dir analysis/v19_support_aware/rebuilt_outputs \
  --d031-point-csv analysis/v19_support_aware/inputs/D031_POINT_LEVEL_LAYER_METRICS_v1.csv
```

The canonical cross-architecture domain is inferred from the intersection of measured support. D019 and D020 are intentionally excluded from that support and are reported only as objective-specific structural audits.
