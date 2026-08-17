# UID/cache isolation fix v1.1

The original package inherited the Jupyter subprocess environment unchanged.
On FMLE this can expose UID 1545 without a resolvable passwd entry or reuse a
problematic shared `/tmp/torchinductor_uid1545` directory.

v1.1 fixes this by:
- explicitly setting USER/LOGNAME/USERNAME/HOME;
- using private per-stage cache and temporary directories under `$HOME/.cache`;
- passing the corrected environment to every notebook subprocess;
- applying the same isolation to shell wrappers;
- fixing the final display cell in `00_PREPARE_AND_LOCK.ipynb`.

No experimental method, checkpoint, split, budget, seed, step count, restart
count or output schema was changed.
