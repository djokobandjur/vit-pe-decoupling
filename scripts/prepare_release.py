#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    archive = OUT / "VIT_PE_DECOUPLING_REPRODUCIBILITY_v1.0.0.zip"
    if archive.exists():
        archive.unlink()
    excluded = {".git", "dist", "artifacts", "build", ".pytest_cache", "__pycache__"}
    excluded_suffixes = {".aux", ".log", ".out", ".spl", ".synctex.gz"}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(ROOT.rglob("*")):
            if not path.is_file() or any(part in excluded for part in path.relative_to(ROOT).parts):
                continue
            if any(path.name.endswith(suffix) for suffix in excluded_suffixes):
                continue
            zf.write(path, Path("vit-pe-decoupling") / path.relative_to(ROOT))
    sums = OUT / "SHA256SUMS.txt"
    sums.write_text(f"{sha256(archive)}  {archive.name}\n")
    print(archive)
    print(sums)


if __name__ == "__main__":
    main()
