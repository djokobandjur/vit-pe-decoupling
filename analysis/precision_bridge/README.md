# Within-ViT-S/16 training-regime comparison

This analysis compares the complete ViT-S/16 AMP and full-FP32 cohorts for
Learned, Sinusoidal, and RoPE. Equal seed numbers are aligned labels, not
strict training pairs. Formal rho50 equivalence is evaluated against the
prespecified absolute margin stored in `equivalence_margin_lock.json`.

The original internal output identifiers are retained only in generated file
names and reference files to preserve a byte-level audit trail. They are not
used in the manuscript narrative.
