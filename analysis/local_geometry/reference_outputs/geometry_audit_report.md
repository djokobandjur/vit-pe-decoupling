# Independent R32 geometry audit

**Status:** PASS
- Raw JSON files: 6
- Checkpoints: 48 / 48
- Direction evaluations: 1584 / 1584
- Integrity errors: 0

## Recalculated aggregate table

| Dataset | PE | Gain ratio | Damage-efficiency gap | Positive gap seeds | Exact sign-flip p (gap) |
|---|---|---:|---:|---:|---:|
| ImageNet-100 | Learned | 11.4038 ± 3.4855× | 0.9137 ± 1.1032 | 6/6 | 0.03125 |
| ImageNet-100 | Sinusoidal | 1.5404 ± 0.0766× | 6.3897 ± 2.9092 | 6/6 | 0.03125 |
| ImageNet-100 | RoPE | 1.2456 ± 0.0327× | 0.0919 ± 0.3272 | 4/6 | 0.56250 |
| ImageNet-100 | ALiBi | 1.0606 ± 0.0323× | 2.2525 ± 3.4570 | 5/6 | 0.15625 |
| CIFAR-100 | Learned | 5.8078 ± 0.3278× | 0.1362 ± 0.3301 | 3/6 | 0.43750 |
| CIFAR-100 | Sinusoidal | 3.4205 ± 0.1442× | 0.4810 ± 0.1279 | 6/6 | 0.03125 |
| CIFAR-100 | RoPE | 1.6862 ± 0.4166× | 0.4735 ± 0.5924 | 5/6 | 0.12500 |
| CIFAR-100 | ALiBi | 1.0506 ± 0.0913× | 0.6338 ± 0.5001 | 5/6 | 0.06250 |

## Friedman tests

- `imagenet_gain_ratio`: chi-square(3)=18.000000, p=0.000439850
- `imagenet_damage_efficiency_gap`: chi-square(3)=9.800000, p=0.020344999
- `imagenet_task_functional_gain`: chi-square(3)=17.000000, p=0.000706742
- `cifar_gain_ratio`: chi-square(3)=18.000000, p=0.000439850
- `cifar_damage_efficiency_gap`: chi-square(3)=5.400000, p=0.144743579
- `cifar_task_functional_gain`: chi-square(3)=16.400000, p=0.000938742

## Agreement with supplied summaries

- Seed summary mismatches >1e-10: 0
- Group summary mismatches >1e-10: 0

## Seed-coherence diagnostics

| Dataset | PE | Metric | Most extreme seed | |z| | LOO mean range | Sign pattern |
|---|---|---|---:|---:|---:|---|
| ImageNet-100 | Learned | task_to_random_functional_gain_ratio | 456 | 1.786 | [10.6484, 12.6491] | + + + + + + |
| ImageNet-100 | Learned | damage_efficiency_gap_task_minus_random | 456 | 1.993 | [0.4739, 1.0611] | + + + + + + |
| ImageNet-100 | Sinusoidal | task_to_random_functional_gain_ratio | 1213 | 1.441 | [1.5234, 1.5625] | + + + + + + |
| ImageNet-100 | Sinusoidal | damage_efficiency_gap_task_minus_random | 456 | 1.749 | [5.6871, 7.4072] | + + + + + + |
| ImageNet-100 | RoPE | task_to_random_functional_gain_ratio | 42 | 1.274 | [1.2383, 1.2539] | + + + + + + |
| ImageNet-100 | RoPE | damage_efficiency_gap_task_minus_random | 789 | 1.479 | [-0.0049, 0.1852] | + - + + - + |
| ImageNet-100 | ALiBi | task_to_random_functional_gain_ratio | 456 | 1.348 | [1.0519, 1.0675] | + + + + + + |
| ImageNet-100 | ALiBi | damage_efficiency_gap_task_minus_random | 1011 | 1.586 | [1.5711, 3.3488] | + + + + - + |
| CIFAR-100 | Learned | task_to_random_functional_gain_ratio | 789 | 1.653 | [5.7514, 5.9162] | + + + + + + |
| CIFAR-100 | Learned | damage_efficiency_gap_task_minus_random | 123 | 1.903 | [0.0106, 0.1878] | - + - - + + |
| CIFAR-100 | Sinusoidal | task_to_random_functional_gain_ratio | 42 | 1.927 | [3.3962, 3.4761] | + + + + + + |
| CIFAR-100 | Sinusoidal | damage_efficiency_gap_task_minus_random | 42 | 1.437 | [0.4443, 0.5178] | + + + + + + |
| CIFAR-100 | RoPE | task_to_random_functional_gain_ratio | 1011 | 1.715 | [1.5433, 1.7445] | + + + + + + |
| CIFAR-100 | RoPE | damage_efficiency_gap_task_minus_random | 1213 | 1.813 | [0.3491, 0.6883] | + + + + + - |
| CIFAR-100 | ALiBi | task_to_random_functional_gain_ratio | 123 | 1.971 | [1.0146, 1.0625] | - + - + + - |
| CIFAR-100 | ALiBi | damage_efficiency_gap_task_minus_random | 42 | 1.760 | [0.5561, 0.8098] | - + + + + + |