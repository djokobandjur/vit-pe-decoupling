
# PE Geometry Probe R32

## Scientific change

The random-direction reference distribution is expanded from 8 to
32 deterministic Gaussian directions per model.

The nominal target rho_rel=0.03 is used only for direction calibration.

All reported damage-efficiency values are computed as:

    delta_CE / actual held-out rho_rel_geometry

The held-out rho is measured on the 64-image geometry subset, which is
disjoint from the 256-image task-gradient subset.

## Jobs

1. 01_geometry_R32_seeds_42_123.ipynb
2. 02_geometry_R32_seeds_456_789.ipynb
3. 03_geometry_R32_seeds_1011_1213.ipynb
4. 99_verify_and_summarize_geometry_R32.ipynb

## Output

/home/djoko.bandjur.ftnkm/Notebooks/results/pe_geometry_probe_n6_r32
