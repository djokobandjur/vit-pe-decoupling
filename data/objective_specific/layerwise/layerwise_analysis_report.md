# D031 per-layer rho decomposition — Sinusoidal audit

**Status:** PASS — completed for D019 Sinusoidal (ViT-B/16 and ViT-S/16 FP32, n=6) and the D020 ViT-S seed-42 branch.

## Locked mathematics

For each measured point, layer contributions are

```text
c_l = rho_abs,l^2 / sum_j rho_abs,j^2
N_eff = 1 / sum_l c_l^2
```

The identity `global_rho_abs^2 = mean_l(layer_rho_abs,l^2)` is satisfied with maximum relative error **3.924e-16**.

## D019 selection stability

- ViT-B: loss-selected vs rho-selected layer profiles have mean cosine **0.996458**, minimum **0.915144**; the selected restart differs at 19/24 points.
- ViT-S: mean cosine **0.999889**, minimum **0.997348**; the selected restart differs at 15/24 points.

Thus, changing the restart-selection criterion usually changes magnitude more than layer allocation, especially for ViT-S.

## D020 direct-rho versus D019 task-loss, same seed and budget

| Budget | Direct-rho N_eff | Task-loss N_eff | Direct max share | Task max share | Profile cosine | JSD (nats) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0200 | 4.448 | 10.978 | 0.407 | 0.121 | 0.613 | 0.126 |
| 0.0095 | 4.043 | 9.224 | 0.430 | 0.202 | 0.589 | 0.140 |

Direct-rho is substantially more layer-concentrated (dominant layer 2) than the task-loss comparator, while task-loss is more distributed and peaks at layer 4. This is additional direction-dependent structural evidence, not a causal mechanism claim.

## Matched-damage rule

Targets 0.8, 0.5 and 0.2 use the nearest actually measured loss-selected point within ±0.03. Profiles are never interpolated. Unavailable pairs are marked `NOT_MATCHED`.
