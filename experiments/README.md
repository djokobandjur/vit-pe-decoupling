# Upstream GPU experiments

The repository's fully tested path is **analysis-level reproduction** from
immutable processed outputs. Research scripts for training and attacks are
included to document the upstream implementation, but they require:

- a licensed copy of ImageNet;
- the expected checkpoint tree or substantial GPU time for retraining;
- the locked CUDA/PyTorch numerical mode described in `configs/`;
- site-specific path configuration.

These scripts are not run by continuous integration.
