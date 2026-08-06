# Model checkpoints

Checkpoint binaries are not included in Git because of size. The included
analysis-level outputs reproduce all reported tables and figures without the
binaries.

Re-running attacks requires checkpoint directories using the structure
expected by the supplied experiment scripts. Before a public release of the
checkpoints, place them in a dedicated archival service such as Zenodo or
Hugging Face, publish SHA-256 hashes, and add the permanent download location
to this file.

The ViT-S/16 full-FP32 manifest is included as
`vits_fp32_checkpoint_manifest.json`. Equivalent manifests should be
published for ViT-B/16 AMP and ViT-S/16 AMP checkpoints before claiming full
attack-level reproducibility.
