# Existing-repository update and release checklist

The public repository already exists at
`https://github.com/djokobandjur/vit-pe-decoupling`. The `main` branch and an
initial `v1.0.0` tag were pushed before the final evidence-completeness audit,
but no GitHub Release was published. Complete the corrected commit and CI run
before creating the first Release.

## 1. Final local review

```bash
python scripts/generate_repository_manifest.py
python scripts/verify_repository.py
python -m pytest -q
python scripts/reproduce_all.py
python scripts/verify_reproduction.py
```

## 2. Commit and push the corrected tree

```bash
git add .
git commit -m "Complete public evidence and manuscript QA"
git push origin main
```

The push automatically starts the `repository-checks` GitHub Actions workflow.
Do not create a Release until the new run is green.

## 3. Move the unpublished v1.0.0 tag to the corrected commit

Because no Release was created from the initial tag, replace it after the
corrected commit has passed CI:

```bash
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
git tag -a v1.0.0 -m "Submission reproducibility release"
git push origin v1.0.0
```

Verify on GitHub that `v1.0.0` points to the corrected commit.

## 4. Create release assets

```bash
python scripts/prepare_release.py
```

Create a GitHub Release from tag `v1.0.0` and attach:

- `VIT_PE_DECOUPLING_REPRODUCIBILITY_v1.0.0.zip`;
- `SHA256SUMS.txt`;
- `VIT_PE_DECOUPLING_OBJECTIVE_AUDITS_RAW_v1.0.0.zip` (separately supplied).

Use `RELEASE_NOTES_v1.0.0.md` as the release description.

## 5. Archive and cite

Enable the GitHub–Zenodo integration and publish the tagged release. Then add
the Zenodo DOI to `CITATION.cff`, the manuscript data-availability statement,
and the release notes.

## Release boundaries

- The root MIT licence applies to project code as described in
  `LICENSE_SCOPE.md`.
- ImageNet images must not be uploaded.
- Checkpoint binaries are not in Git. A permanent checkpoint archive, if
  published separately, should be linked with hashes from `checkpoints/`.

## Git/manifest consistency

LaTeX build intermediates are intentionally ignored and excluded from the public release (`*.aux`, `*.log`, `*.out`, `*.spl`, `*.synctex.gz`). Run `python scripts/generate_repository_manifest.py` after any final edit and before `git add .`; the verifier rejects either missing manifest entries or ignored build artifacts appearing in the manifest.
