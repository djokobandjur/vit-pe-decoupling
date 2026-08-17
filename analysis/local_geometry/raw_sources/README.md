# Original R32 geometry sources

This directory redistributes the retained original ViT-B/16 R32 geometry
outputs used by the manuscript: six paired-seed raw JSON files, six completion
markers, the original seed/group summaries, the global completion record, and
the contemporaneous SHA-256 list.

`SHA256_ALL_ORIGINAL_FILES.txt` verifies every original result file in this
directory. `RECOVERED_ARCHIVES_SHA256.txt` records the hashes of the recovered
source archives from which these files were restored. The raw JSON metadata
contains the SHA-256 digests of the two core execution sources; those exact
sources are redistributed under `../execution_provenance/code/`.
