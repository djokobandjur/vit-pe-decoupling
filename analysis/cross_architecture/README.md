# AMP-matched cross-architecture comparison

This analysis holds AMP training fixed and compares ViT-B/16 with ViT-S/16
for Learned, Sinusoidal, and RoPE. It reproduces the support-matched nAUC
contrasts, exact sign-flip quantisation, Welch sensitivity, ordinal
interaction, and seed-dispersion analyses. ALiBi is not imputed because the
ViT-S/16 AMP training recipe did not yield a valid checkpoint cohort. The
public evidence supports this cohort-level trainability boundary; raw per-seed
ALiBi training logs are not redistributed, so no exact per-seed failure count
is claimed.
