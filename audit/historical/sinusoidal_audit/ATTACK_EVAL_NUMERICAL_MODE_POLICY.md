# Attack/eval numerical-mode policy

The audit engine explicitly disables autocast for attack forward, backward, evaluation and rho measurement. Inputs and loaded model parameters must be float32. No GradScaler is used during attack.

The preflight captures current TF32, cuDNN and float32-matmul settings into `ATTACK_EVAL_NUMERICAL_MODE_v1.json`. Both sessions reapply and verify those settings before any GPU work.

This lock establishes consistency of the new audit. It does not by itself prove historical cross-hardware numerical equivalence. The generated `D033A_PROVENANCE_REPORT.json` records which historical engine hashes match and which provenance remains pending reconciliation.
