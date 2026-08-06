# Upload checklist

The prepared repository is ready for a first public push after three manual
fields are decided: the GitHub account/repository name, confirmation of the
code licence, and whether checkpoint binaries will be published separately.

## 1. Final local review

```bash
python scripts/verify_repository.py
pytest -q
python scripts/reproduce_all.py
python scripts/verify_reproduction.py
```

## 2. Create an empty GitHub repository

Create a repository on GitHub without adding a README, licence, or `.gitignore`
(the prepared repository already contains them). The public repository is
`vit-pe-decoupling`: https://github.com/djokobandjur/vit-pe-decoupling.

## 3. Push the prepared tree

```bash
git init
git add .
git commit -m "Public reproducibility release v1.0.0"
git branch -M main
git remote add origin https://github.com/djokobandjur/vit-pe-decoupling.git
git push -u origin main
git tag -a v1.0.0 -m "Submission reproducibility release"
git push origin v1.0.0
```

## 4. Create the release assets

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
the GitHub URL and Zenodo DOI to `CITATION.cff`, the manuscript data/code
availability statement, and the final repository release notes.

## Decisions to confirm before upload

- The root MIT licence is a prepared default for project code. Replace it
  before the first push if a different licence is required.
- ImageNet images must not be uploaded.
- Checkpoints are not in Git. If they are published through Zenodo or Hugging
  Face, add the permanent link and hashes to `checkpoints/README.md`.
