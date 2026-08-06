# D030a V20 final critical-path reconciliation

**Status: PASS**  
**New GPU experiment required: NO**  
**Critical rows reconciled: 13/13**

This is the final V20-wide reconciliation, extending the earlier V19 D030a gate through D029, D059 reanalysis v2, the final abstract, the supplement, and the final claim-language audit.

## Headline verification

- Canonical cohort: **72 checkpoints** (48 ViT-B + 24 ViT-S full-FP32).
- Canonical noise-minus-adversarial gap: **72/72 positive** seed-family-cohort rows.
- D019: **240/240 restarts**, **48/48 points**, PASS.
- D020: **PASS**, `LOCK_200x5_FOR_D020_VITS`; best-of-five envelope semantics retained.
- D029: rankings preserved; formal rho50 equivalence only for Sinusoidal; Learned and RoPE remain inconclusive.
- D059: exact Friedman `p=0.02893519`; Learned--RoPE ordering reverses; Welch sensitivity agrees.
- Appendix C claim-language audit: PASS.

## Evidence-chain repair made at this gate

The prior V20 source package contained D029 and D059 summaries, but not both portable replication ZIPs as physical source-package inputs. The final package now includes both under `raw_inputs/v20/`, plus the A7 methodology package and the historical D008 governance record. This closes the byte-level claim-to-evidence chain.

The abstract uses `systematically diverged` for ViT-S/16 AMP ALiBi. This is the strongest wording directly supported by the locked D008 governance artifact; it avoids turning a cohort-level governance result into an unevidenced per-seed raw-log count.

## Locked claim boundaries

- D019/D020 do not extend canonical support.
- Sinusoidal task-loss frontier is not a structural ceiling or supremum.
- D020 PASS is best-of-five envelope stability, not all-restart convergence or search exhaustion.
- D029 supports Sinusoidal equivalence only, not general precision invariance.
- D059 is seed-aligned, not strict paired, and is limited to two tested backbones and three shared families.
- D031 is layerwise descriptive evidence, not head/token localization or a causal mechanism.

## Matrix

See `D030a_V20_FINAL_CRITICAL_PATH_MATRIX_v1.csv` for the complete claim -> decision -> result -> artifact -> limit mapping.
