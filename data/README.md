# Data included in the repository

This repository contains **processed experiment outputs**, split manifests,
and analysis-level inputs. It does not contain ImageNet images or trained model
checkpoint binaries.

- `canonical_inputs/` contains the 75 JSON sources used by the canonical
  analysis pipeline.
- `vits_fp32_results.zip` contains the complete six-seed ViT-S/16 full-FP32
  robustness outputs.
- `vits_amp_results.zip` contains the complete six-seed ViT-S/16 AMP outputs
  for Learned, Sinusoidal, and RoPE.
- `objective_specific/` contains compact results for all-restart,
  direct-displacement, and layerwise analyses.

ImageNet-100 must be reconstructed from a licensed ImageNet copy. The images
are not redistributed. Split sizes and the split seed are recorded in
`configs/data_splits.json` and in the metadata of every raw result JSON.

## Historical provenance paths

Some immutable raw JSON records and locked audit manifests retain absolute
paths from the original compute environment (for example, paths beginning
with `/home/...`). These strings are inert provenance metadata: the public
analysis scripts do not dereference them. They are retained intentionally so
that the released inputs preserve their original hashes and evidence chain.
All executable paths used by the public reproduction workflow are resolved
relative to the repository root or supplied through command-line arguments.
