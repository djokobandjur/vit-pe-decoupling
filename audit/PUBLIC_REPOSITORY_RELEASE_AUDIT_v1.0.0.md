# Public repository release audit - v1.0.0 / v8 final metadata hardening

**Status:** `PASS_PUBLIC_REPOSITORY_RELEASE_CANDIDATE_V8_S1_CLOSED`
**Date:** 2026-08-07

## Final checks

| Check | Result |
|---|---:|
| Original N1-N5 audit findings | 5/5 CLOSED |
| Prior V1-V7 audit findings | 7/7 CLOSED |
| New N1-N6 audit findings | 6/6 CLOSED |
| Final R1-R3 audit findings | 3/3 CLOSED |
| Final S1 metadata finding | 1/1 CLOSED |
| Repository verifier | PASS |
| Repository verifier checks | 175 PASS |
| Python tests | 26/26 PASS |
| Claim-to-evidence gates | 30/30 PASS |
| All-restart audit lock artifacts | 3/3 byte-exact PASS |
| Complete GPU-free analysis reproduction | PASS |
| Exact reproduction checks | 35/35 PASS |
| Manuscript builds | 49-page main; 5-page supplement |
| Figure render comparisons | 4/4 pixel-identical at 150 dpi |
| PDF preflight / Ghostscript | PASS / PASS |
| Local directional geometry | PASS; 6 raw JSONs / 1,584 directions / 48 ViT-B/16 checkpoints |
| ViT-B canonical attack checkpoint binding | 48/48 paths -> post-hoc SHA-256 manifest |
| Independent historical ViT-B digest cross-check | 6/6 PASS |
| ViT-S checkpoint manifests | 24 full-FP32 + 18 AMP records |
| ViT-S Sinusoidal coverage-extension registry | 3/3 target-not-reached; canonical maxima unchanged to ledger precision |
| Numerical-mode source boundary | float32 tensors/model state; no autocast/GradScaler; MATH SDPA explicitly selected in the locked PyTorch 2.8 environment; TF32 permitted in later audit lock |
| Supplement S1/S6/S7 references | label-based via xr-hyper |
| Git-ignored LaTeX intermediates in release | 0 |
| Fresh `git add .` coverage of SHA manifest | 372/372 listed files staged (+ manifest itself) |
| CSV byte-preservation under Git | PASS (`*.csv -text`) |
| Stale manuscript page-count metadata | 0 |
| Abstract texcount metadata | 237 in both audit records; live texcount guard when available |

## New N1-N6 closure

- **N1:** Three non-canonical ViT-S/16 Sinusoidal coverage-extension runs are disclosed as ledger-registered support diagnostics. Their registered combined maxima equal the corresponding canonical maxima to ledger precision and all failed the 0.09 target. Original row-level SINEXT JSONs are explicitly not claimed as public; the later public all-restart budget-0.020 audit is identified as a distinct experiment.
- **N2:** FP32 wording is source-accurate rather than absolute. Public evidence supports float32 inputs/model parameters, no autocast, no gradient scaling, and explicit MATH SDPA selection in the locked PyTorch 2.8 environment; the later H200 audit lock permits TF32 matrix multiplication. Historical canonical environment-level reconciliation remains explicitly pending.
- **N3:** The seed-42 schedule-development audit is described as using observed empirical separation rather than an a priori numerical convergence threshold. Observed selected-loss changes 0.00410 versus >=0.00579 are not presented as a prespecified 0.005 criterion.
- **N4:** LaTeX build intermediates are removed from the release tree and excluded by both `.gitignore` and release packing. The repository SHA manifest enumerates and hashes the complete release tree, verifier/tests reject ignored intermediates, and `.gitattributes` marks CSV evidence as `-text` to prevent Git line-ending normalization from changing byte-level SHA-256 values.
- **N5:** The page-count audit is synchronized to the actual PDFs (49/5) and is machine-checked with `pypdf`.
- **N6:** S1/S6/S7 main-text references use `xr-hyper` labels; the historical POINT_3_5 protocol hash is retained only as unbound source-record metadata; and the distinct ViT-B/ViT-S validation paths are documented with identical sample-order and split SHA-256 verification.


## Final R1-R3 closure

- **R1:** `configs/attack_eval_numerical_mode.json` is explicitly documented as an annotated reviewer-facing copy. Its `locked_artifact_reference` points to `audit/historical/sinusoidal_audit/ATTACK_EVAL_NUMERICAL_MODE_v1.json`, whose SHA-256 is `df0341e1953c9457bdf9dec409bbdd7f122246f85204f0b999ebc666a70748f7`; the verifier checks that this equals the digest recorded in `FINAL_AUDIT_REPORT_v1.json`.
- **R2:** Gate V20-G29 now names the two manuscript source files explicitly. The verifier and tests resolve every semicolon-delimited evidence token in all 30 claim gates, with glob-aware checking for wildcard entries.
- **R3:** The manuscript states that MATH SDPA is explicitly selected where the installed PyTorch exposes `torch.nn.attention`, and identifies the locked PyTorch 2.8 audit environment as taking that branch. Older-PyTorch fallback paths are explicitly outside the MATH-forcing claim.

## Final S1 closure

- **S1:** The distributed abstract is **237 words by `texcount`**. `audit/manuscript_internal_label_removal.md` already contained 237; the stale `242` value in `PUBLIC_REPOSITORY_RELEASE_AUDIT_v1.0.0.json` is corrected to 237. The verifier and tests require both audit records to agree, and the verifier additionally reruns `texcount` on the extracted abstract whenever the executable is installed. Gate V20-G30 records this invariant.

External GitHub visibility is deliberately treated as a manual pre-submission hosting check rather than an offline package PASS condition; see `audit/PRE_SUBMISSION_CHECKLIST.md`.

## Evidence boundaries retained deliberately

The original row-level ViT-S Sinusoidal coverage-extension JSONs are not redistributed. The public statement about those three runs is therefore limited to the historical run ledger plus the canonical source JSONs whose maxima match the ledger combined-max values. A later all-restart budget-0.020 audit is public at row level but is not represented as the original SINEXT execution.

The detailed `ATTACK_EVAL_NUMERICAL_MODE_v1` environment record was captured during the later provenance audit, not contemporaneously with every canonical run. Engine-source provenance verifies absence of autocast/GradScaler and conditional MATH SDPA selection, active in the locked PyTorch 2.8 environment; the later lock records TF32 matmul/cudnn enabled.

The historical ViT-B attack JSONs record checkpoint paths, not contemporaneous SHA-256 values. The complete 48-checkpoint manifest was reconstructed post hoc from the retained checkpoint cohort; all 48 paths bind exactly and six ImageNet-100 Sinusoidal digests have independent historical matches.

The earlier raw RTX Blackwell hardware log remains outside the public package. The Blackwell-to-H200 comparison is retained as an audit-summary result, not a publicly rerunnable raw paired-hardware experiment.

The protocol hash `3970b99b...` in the retained historical POINT_3_5 report is preserved as source-record metadata only; no independently distributed public protocol object is claimed to bind to it.
