# ViT Positional-Encoding Decoupling

Reproducibility repository for the manuscript **“Random and Adversarial
Positional-Parameter Robustness Decouple Across Positional-Encoding
Families.”**

Repository: https://github.com/djokobandjur/vit-pe-decoupling

The repository provides a tested, GPU-free path from immutable seed-level
outputs to the reported aggregate statistics, tables, and figures. It also
includes clean manuscript source, compact structural-audit evidence, upstream
training/attack scripts, split metadata, and provenance records.

## Main scientific scope

The study compares Learned, Sinusoidal, RoPE, and ALiBi positional mechanisms
under random parameter noise and task-loss adversarial perturbations, using
achieved relative RMS displacement of pre-softmax attention logits as a common
functional scale. The package covers:

- 72 canonical checkpoints: AMP-trained ViT-B/16 on CIFAR-100 and
  ImageNet-100, plus full-FP32 ViT-S/16 on ImageNet-100;
- an additional 18-checkpoint ViT-S/16 AMP cohort for Learned, Sinusoidal,
  and RoPE;
- the within-ViT-S training-regime comparison;
- the AMP-matched ViT-B/16 versus ViT-S/16 comparison; and
- all-restart, direct-displacement, and layerwise structural audits.

## Reproduction levels

### Level 1 — numerical analysis without a GPU

This is the fully packaged and tested path. It reconstructs the canonical
curves and nAUC values, the cross-cohort comparison, the training-regime
bridge, the AMP-matched architecture comparison, and the manuscript figures.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
python scripts/verify_repository.py
python scripts/reproduce_all.py
```

Outputs are written to `artifacts/reproduced/`. The final status file is
`artifacts/reproduced/REPRODUCTION_REPORT.json`.

The exact permutation calculations are CPU-intensive and may run for several
minutes even though this reproduction level does not require a GPU.

Equivalent Make targets:

```bash
make verify
make reproduce
```

### Level 2 — re-run perturbation experiments

This requires the original model checkpoints, a licensed ImageNet copy, and a
compatible CUDA/PyTorch environment. The repository includes checkpoint
manifests and the research attack implementation, but checkpoint binaries are
not stored in Git. See `checkpoints/README.md` and `experiments/README.md`.

### Level 3 — retrain models

Training source is included for provenance and extension. Full multi-seed
retraining is computationally expensive and requires site-specific dataset and
storage paths. The analysis-level results do not depend on retraining.

## Repository layout

```text
analysis/                 numerical pipelines and locked reference outputs
configs/                  cohorts, splits, seeds, and numerical mode
data/                     processed seed-level inputs; no ImageNet images
experiments/              upstream training and attack research scripts
manuscript/               clean Neural Networks submission source and PDFs
audit/                    final claim/evidence and language audits
provenance/               internal-to-public identifier map
scripts/                  one-command reproduction and verification tools
tests/                    lightweight repository integrity tests
checkpoints/              checkpoint manifest and publication instructions
```

## Expected headline checks

The verification path confirms, among other sentinels:

- the canonical CIFAR-100 ALiBi random-noise nAUC convention;
- the family-specific training-regime decisions: Sinusoidal equivalent,
  Learned and RoPE inconclusive under the prespecified rho50 margin;
- the 31/31 AMP-matched cross-architecture QA checks;
- completion of all 240 planned all-restart evaluations; and
- presence of the two direct-displacement central endpoints.

## Data and checkpoint policy

ImageNet images are not redistributed. The class/split construction must be
performed from a licensed ImageNet copy.

Model checkpoint binaries are not stored in Git because of their size. The
exact checkpoints used in the study are available from the shared read-only
Google Drive archive:

**Checkpoint archive:**  
https://drive.google.com/drive/folders/1UojvGk3oeoQui7jy8DSFM_0U6Xkwx-IB?usp=sharing

The archive contains 90 checkpoints in total:

- 48 ViT-B/16 AMP checkpoints: 24 ImageNet-100 and 24 CIFAR-100;
- 24 canonical ViT-S/16 full-FP32 ImageNet-100 checkpoints; and
- 18 auxiliary ViT-S/16 AMP-FP16 ImageNet-100 checkpoints.

The 72 canonical checkpoints consist of the 48 ViT-B/16 models and the
24 ViT-S/16 full-FP32 models. The additional 18 ViT-S/16 AMP-FP16
checkpoints form the auxiliary training-regime comparison cohort.

Checkpoint-level SHA-256 manifests are provided in `checkpoints/`, and all
90 checkpoint files were verified against their reference SHA-256 values
after transfer to the shared archive. See `checkpoints/README.md` for the
archive layout, checkpoint cohorts, integrity manifests, and expected
directory structure.

Processed result JSON/CSV files included in this repository are sufficient
to reproduce every reported table and figure without downloading the model
checkpoint binaries.

## Manuscript build

The submission source is in `manuscript/`. With a TeX installation that
provides `pdflatex` and the Elsevier class:

```bash
python scripts/build_manuscript.py
```

## Release and citation

`CITATION.cff` includes the public GitHub repository URL. After an archival
DOI is issued, add the DOI to that file. Instructions for the
first push, tag, and release assets are in `UPLOAD_TO_GITHUB.md`.

The root MIT licence applies to project code only; see `LICENSE_SCOPE.md`.
