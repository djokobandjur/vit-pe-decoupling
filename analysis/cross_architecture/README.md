# AMP-matched cross-architecture comparison

This analysis holds AMP training fixed and compares ViT-B/16 with ViT-S/16
for Learned, Sinusoidal, and RoPE. It reproduces the support-matched nAUC
contrasts, exact sign-flip quantisation, Welch sensitivity, ordinal
interaction, and seed-dispersion analyses. ALiBi is not imputed because every
attempted ViT-S/16 AMP training run diverged.
