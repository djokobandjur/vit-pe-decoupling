# ViT-S/16 AMP checkpoint manifest

`VITS_FP16_CHECKPOINT_MANIFEST.json` is extracted byte-for-byte from
`data/vits_amp_results.zip` (`vits_in100_fp16_robustness_v1/input_audit/`). It
contains 18 checkpoint records: Learned, Sinusoidal, and RoPE across six seeds.
There are no ALiBi checkpoint rows because the ViT-S/16 AMP training recipe did
not yield a valid ALiBi checkpoint cohort. The public evidence supports that
cohort-level boundary; raw per-seed ALiBi training logs are not redistributed,
so no exact per-seed failure count is claimed.
