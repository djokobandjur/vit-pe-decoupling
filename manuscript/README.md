# Manuscript source

The files in this directory are the clean submission sources. Internal working
identifiers were removed from the reviewer-facing text, captions, labels, and
filenames. Build instructions are in `README_BUILD.txt`. The supplement is compiled first so the main source can resolve cross-document Supplement table labels via `xr-hyper`.

The analysis scripts regenerate the numerical tables and figures into
`artifacts/reproduced/manuscript_assets/`. The checked-in manuscript assets
remain the locked submission versions.

The supplement includes the deterministic no-envelope sensitivity analysis
(Table S6) and the post-lock task-loss-grid sensitivity (Table S7).

All seven main tables and all seven supplementary tables are external
generated sources under `tables/`; regenerate them with
`python scripts/generate_manuscript_tables.py` from the repository root.
