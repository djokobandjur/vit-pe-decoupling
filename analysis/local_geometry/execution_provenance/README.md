# R32 execution provenance

This directory contains the retained FMLE execution package used for the
ViT-B/16 local directional-geometry probe: configuration, package provenance,
three source modules, and the four launch/verification notebooks. The six raw
JSON files under `../raw_sources/` embed the following core source hashes:

- `cross_family_rho_decoupling.py`: `d1bf3324353c1a0d377967d2088e32b64a9bda88edacba2383ea2d44a382a307`
- `full_scale_experiment.py`: `83fc337128dec7f896c9816842806789a634154dea8372bb0a43bae19188d3bf`

`verify_geometry_analysis.py` verifies those hashes against the redistributed
source bytes before reconstructing the processed geometry tables.
