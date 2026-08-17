# N1-N6 reviewer-audit closeout - v6

Status: **PASS - all six new findings closed without new experiments.**

## N1 - ViT-S Sinusoidal coverage-extension disclosure
**PASS (evidence-bounded).** Three ledger-registered non-canonical coverage-extension runs (seeds 123, 456, 1213) are disclosed. Their combined maxima match canonical maxima to ledger precision and all failed the 0.09 target. Original row-level SINEXT JSONs are not claimed as public. The later public all-restart budget-0.020 audit is explicitly identified as a distinct experiment.

## N2 - FP32 / numerical-mode wording
**PASS.** The manuscript now states the source-level facts: float32 inputs/model state, no autocast, no GradScaler, and MATH SDPA selection in the locked PyTorch 2.8 environment (with an older-PyTorch fallback explicitly outside that claim). It also discloses that TF32 matmul/cudnn are enabled in the later audit lock and that historical environment-level reconciliation is pending; the later lock is not presented as contemporaneous evidence for the canonical runs.

## N3 - Convergence-threshold semantics
**PASS.** The seed-42 schedule-development audit is explicitly described as using empirical separation rather than an a priori numerical threshold. The 0.00410/0.00579 selected-loss changes are retained as observed separation values, not a prespecified 0.005 criterion.

## N4 - Git-ignore / SHA-manifest release failure
**PASS.** LaTeX build artifacts (.aux/.log/.out/.spl/.synctex.gz) are removed from the release tree and excluded by the release packer. Verifier/tests reject their reappearance or inclusion in the repository SHA manifest. CSV evidence files are marked `*.csv -text` in `.gitattributes` so Git does not normalize their byte-level line endings; a fresh `git add .` check covers every manifest-listed file.

## N5 - Stale page count
**PASS.** The audit record is synchronized to the rendered PDFs and verifier/tests use pypdf to compare the claimed main/supplement page counts with the actual PDFs.

## N6 - Supplement references, orphan protocol hash, and validation-copy provenance
**PASS.** Main-text S1/S6/S7 references use cross-document LaTeX labels via xr-hyper; the build order compiles the supplement first. The historical protocol hash in POINT_3_5 remains source-record metadata only and is not presented as independently bound public evidence. The distinct ViT-B/ViT-S validation paths are disclosed together with identical sample-order and split SHA-256 verification.
