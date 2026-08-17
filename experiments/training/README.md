# Training-source scope

The only training driver redistributed in this repository is
`vitb_amp_training_source.py`. It is an unmodified historical ViT-B/16 AMP
source file with ViT-B geometry hard-coded in its configuration. Its default
command-line values also reflect the earlier three-seed driver
(`42, 123, 456`) and batch size 256; they are not the locked six-seed,
batch-size-128 production launch configuration reported in the manuscript.

The ViT-S/16 full-FP32 and AMP production training launchers are **not**
redistributed. Consequently, this repository does not claim source-level
from-scratch retraining reproducibility for either ViT-S cohort. Their public
provenance is instead limited to protocol/configuration records, checkpoint
identity manifests where available, and the immutable processed outputs used
by the analysis-level reproduction.

The historical architecture-source hash recorded in the ViT-S checkpoint and
protocol audits matches the shared model-definition file from which the
architecture was derived. That equality must not be interpreted as evidence
that the public ViT-B AMP driver can train a ViT-S/16 FP32 model.
