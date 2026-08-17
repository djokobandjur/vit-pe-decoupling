# Model checkpoints

Checkpoint binaries are not stored in Git because of their size. The
analysis-level JSON/CSV outputs included in this repository are sufficient to
reproduce the reported tables, statistics, and figures without downloading the
model binaries.

For attack-level reproduction, the exact checkpoints used in the study are
available from the shared read-only Google Drive archive:

**Checkpoint archive:**  
https://drive.google.com/drive/folders/1UojvGk3oeoQui7jy8DSFM_0U6Xkwx-IB?usp=sharing

This Drive root is shared with related ViT projects. The folders used by this
manuscript are:

| Folder | Architecture / regime | Dataset | PE families | Checkpoints |
|---|---|---|---|---:|
| `ImageNet100/` | ViT-B/16 AMP | ImageNet-100 | Learned, Sinusoidal, RoPE, ALiBi | 24 |
| `CIFAR100/` | ViT-B/16 AMP | CIFAR-100 | Learned, Sinusoidal, RoPE, ALiBi | 24 |
| `ImageNet100_ViTS_FP32/` | ViT-S/16 full FP32 | ImageNet-100 | Learned, Sinusoidal, RoPE, ALiBi | 24 |
| `ImageNet100_ViTS_AMP_FP16/` | ViT-S/16 AMP-FP16 | ImageNet-100 | Learned, Sinusoidal, RoPE | 18 |
| **Total** |  |  |  | **90** |

The 72 canonical checkpoints consist of the 48 ViT-B/16 models plus the
24 ViT-S/16 full-FP32 models. The additional 18 ViT-S/16 AMP-FP16 checkpoints
form the auxiliary training-regime comparison cohort.

## SHA-256 integrity manifests

The repository includes checkpoint-level SHA-256 manifests:

- `NN_ViTB_48CHECKPOINTS_SHA256_CLEAN.txt`  
  SHA-256 hashes for the 48 ViT-B/16 `best_model.pth` files:
  24 ImageNet-100 + 24 CIFAR-100.

- `NN_ALL_90_CHECKPOINTS_SHA256.txt`  
  SHA-256 hashes for all 90 `best_model.pth` files used by the study.

- `vits_fp32_checkpoint_manifest.json`  
  Detailed manifest for the 24 canonical ViT-S/16 full-FP32 checkpoints.

All 90 checkpoint files were verified against their reference SHA-256 values
after transfer to the public checkpoint archive.

## Expected checkpoint structure

The relevant portion of the shared Drive archive is:

```text
ads-vit-forensics/
├── ImageNet100/
│   ├── learned_seed42/best_model.pth
│   ├── ...
│   └── alibi_seed1213/best_model.pth
│
├── CIFAR100/
│   ├── learned_seed42/best_model.pth
│   ├── ...
│   └── alibi_seed1213/best_model.pth
│
├── ImageNet100_ViTS_FP32/
│   ├── learned_seed42/best_model.pth
│   ├── sinusoidal_seed42/best_model.pth
│   ├── rope_seed42/best_model.pth
│   ├── alibi_seed42/best_model.pth
│   └── ...
│
└── ImageNet100_ViTS_AMP_FP16/
    ├── learned_seed42/best_model.pth
    ├── sinusoidal_seed42/best_model.pth
    ├── rope_seed42/best_model.pth
    └── ...
