# ViT-B/16 R32 checkpoint digest manifest

Checkpoint binaries are not stored in Git. This directory contains the
byte-level SHA-256 manifest reconstructed from the retained original 48-model
ViT-B/16 checkpoint cohort used by the R32 geometry probe:

`2 datasets × 4 PE families × 6 seeds = 48 checkpoints`.

All 48 files were present and uniquely identified. The six ImageNet-100
Sinusoidal digests independently preserved in the earlier
`SINUSOIDAL_PER_RESTART_AUDIT_INPUT_LOCK_v1.json` record matched exactly
(`6/6`). The complete 48-file manifest was generated post hoc from the retained
original checkpoint directories; this timing is recorded explicitly in the
JSON manifest and does not imply that all 48 hashes were captured before the
R32 run.
