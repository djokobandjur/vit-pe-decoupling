#!/usr/bin/env python3
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript"


def require(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required executable not found: {name}")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=MAN, check=True)


def main() -> None:
    require("pdflatex")
    main_tex = "pe_robustness_nn_main.tex"
    supp_tex = "pe_robustness_nn_supplement.tex"
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main_tex])
    for _ in range(3):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", supp_tex])
    print("Manuscript build complete.")


if __name__ == "__main__":
    main()
