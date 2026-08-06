# Public repository release audit — v1.0.0

**Status:** `PASS_PUBLIC_REPOSITORY_RELEASE_CANDIDATE`  
**Date:** 2026-08-06

## Reproducibility levels

- Analysis-level reproduction of every reported table and figure: **PASS**, no GPU required.
- Attack-level reruns: supported by code and manifests, but require checkpoint binaries to be published separately.
- Full retraining: supported by provenance source/configuration, but requires licensed datasets and substantial GPU resources.

## Final checks

| Check | Result |
|---|---:|
| Repository integrity | PASS |
| Python tests | 3/3 PASS |
| Complete analysis reproduction | PASS |
| Exact numerical/table checks | 13/13 PASS |
| AMP-matched cross-architecture QA | 31/31 PASS |
| Manuscript figure render comparisons | 4/4 pixel-identical |
| Portable ZIP extraction and verification | PASS |
| Internal workflow codes in reviewer-facing TEX | 0 |
| Obsolete package-version wording | 0 |

All four manuscript figures were rendered independently at 150 dpi and had
zero changed pages and zero changed pixels against the reproduced outputs.

## Repository tree

- Candidate public files: 263
- Candidate public size: 16.11 MiB
- Largest file: `data/canonical_inputs/cifar_base/session_1_seeds_42_123/noise_all_families.json` (1.38 MiB)
- No file approaches GitHub's normal single-file limit.

## Release boundaries

ImageNet images and checkpoint binaries are intentionally absent from Git.
Processed outputs are sufficient to reproduce every manuscript table and
figure. Full raw objective-specific audit archives are provided as a separate
GitHub Release asset. Historical absolute compute paths retained inside raw
records are inert provenance metadata and are not dereferenced by public
scripts.

The included MIT licence is a prepared default for project code and should be
confirmed by the author before the first public push. Repository and Zenodo
URLs/DOIs must be added to `CITATION.cff` after publication.
