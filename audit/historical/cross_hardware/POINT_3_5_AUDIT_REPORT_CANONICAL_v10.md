# Point 3.5 audit closure and v10 numerical update

## Midpoint completion integrity

- Six seeds present: [42, 123, 456, 789, 1011, 1213].
- Thirty nonzero transition points present (five per seed).
- One protocol hash across all full runs: `3970b99be5c884b6930f91608eba24ab85337bf82407fb6d6f4fbca8f2b25a95`.
- Every nonzero point has five restarts; selected attack loss equals the maximum restart loss (max absolute discrepancy 0).
- Global-RMS constraint matches the requested native budget (max absolute discrepancy 2.09e-08).
- Clean evaluation accuracies match the original base files exactly.
- All full logs end with `COMPLETE` and `rc=0`.

## Seed-42 hardware recheck

The H200 rerun agrees with the earlier RTX Blackwell Server Edition values to the precision available in the earlier log. The largest absolute difference is 0.000491 in normalized accuracy and 2.55718e-05 in achieved rho. The H200 seed-42 values are used in the homogeneous six-seed transition grid; this substitution changes seed-42 wide-range nAUC only at the ~1e-4 level or less.

## Updated CIFAR-100 wide-range nAUC, rho in [0, 0.23]

| Family | Noise mean ± SD | Adversarial mean ± SD |
|---|---:|---:|
| Learned | 0.9976 ± 0.0025 | 0.7106 ± 0.0523 |
| Sinusoidal | 0.9922 ± 0.0008 | 0.7581 ± 0.0390 |
| Rope | 0.9948 ± 0.0009 | 0.7912 ± 0.1205 |
| Alibi | 0.9641 ± 0.0062 | 0.8475 ± 0.0146 |

The extremal reversal remains 6/6 in both directions: Learned exceeds ALiBi under noise, while ALiBi exceeds Learned under adversarial perturbations. RoPE exceeds Sinusoidal adversarially in 2/6 seeds, so the complete four-family ranking is not claimed as seed-stable.

## Convergence audit interpretation

The seed-42 step-doubling audit did not use a prespecified numerical convergence threshold. Instead, it showed a clear empirical separation: configurations retained for the final protocol changed evaluation accuracy by at most 0.226 percentage points and selected attack loss by at most 0.00410 under a further step doubling, whereas configurations that were extended changed accuracy by at least 0.331 percentage points and selected loss by at least 0.00579. This observed separation supports the locked schedules without presenting the values as an a priori threshold.

## Development-seed sensitivity

Seed 42 was used for protocol development. Excluding it leaves all 40 remaining primary noise-minus-adversarial gaps positive, preserves family-mean orderings, and gives Q=13.56 with exact p=1.38648e-4 on both datasets (1104/7,962,624 permutations).
