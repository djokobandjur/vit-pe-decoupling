# Upstream GPU experiments

The repository's fully tested path is **analysis-level reproduction** from
immutable processed outputs. Research attack scripts and one historical
ViT-B/16 AMP training driver are included to document upstream implementation
details, but they require:

- a licensed copy of ImageNet;
- the expected checkpoint tree or substantial GPU time;
- the locked CUDA/PyTorch numerical mode described in `configs/`;
- site-specific path configuration.

The public training source is intentionally partial.
`training/vitb_amp_training_source.py` is ViT-B/16 AMP-specific and retains
older default launch values. The production ViT-S/16 full-FP32 and AMP
training launchers are not redistributed. See `training/README.md`. The
repository therefore supports extension and implementation inspection, not a
claim that every reported training cohort can be recreated from the public
training scripts alone.

These GPU scripts are not run by continuous integration.
