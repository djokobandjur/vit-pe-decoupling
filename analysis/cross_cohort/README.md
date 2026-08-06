# Canonical cross-cohort comparison

This analysis combines AMP-trained ViT-B/16 outputs with the independent
full-FP32 ViT-S/16 replication. Architecture and training regime change
jointly, so the result is a system-level cross-cohort comparison rather than a
clean architecture effect. The separate AMP-matched analysis isolates the
backbone axis for the three shared trainable families.
