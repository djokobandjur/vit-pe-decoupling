# Historical ViT-B/16 harmonization evidence

This directory documents the public evidence boundary for the ImageNet-100
ViT-B/16 confirmation-seed harmonization used to lock the family-specific PGD
schedules.

The redistributed historical `run_vitb_harmonization.py` performs the matched
N-to-2N runs on seed 123, selects three existing transition budgets nearest
normalized accuracies 0.9, 0.7, and 0.5, keeps five restart seeds matched, and
uses the fixed 256/1280/3464 calibration/attack/evaluation split. The retained
run ledger registers the Learned 400->800, Sinusoidal 400->800, RoPE 200->400,
ALiBi 100->200, and targeted ALiBi 200->400 executions.

The original row-level harmonization output JSONs and the original v17/v18
lock/comparison/closure files are not redistributed here. Consequently, the
public release records the historical schedule decision and its execution
provenance but does **not** claim public row-level reproduction of the original
N-to-2N comparison table. In particular, this public evidence does not support
relabeling an evaluation-cross-entropy tolerance as a `selected_attack_loss`
tolerance.
