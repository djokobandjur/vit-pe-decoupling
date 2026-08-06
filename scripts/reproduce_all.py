#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run([str(x) for x in cmd], cwd=cwd or ROOT, check=True)


def extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)
    dirs = [p for p in destination.iterdir() if p.is_dir()]
    return dirs[0] if len(dirs) == 1 else destination


def stage_canonical(work: Path, output: Path) -> tuple[Path, Path]:
    package = work / "canonical_package"
    pipeline = package / "analysis" / "canonical_cohorts"
    pipeline.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "analysis/canonical_cohorts/canonical_analysis.py", pipeline / "canonical_analysis.py")
    shutil.copytree(ROOT / "data/canonical_inputs", pipeline / "inputs")
    (package / "figures").mkdir(parents=True, exist_ok=True)
    run([sys.executable, pipeline / "canonical_analysis.py", "--package-dir", package])
    generated = pipeline / "outputs"
    shutil.copytree(generated, output / "canonical", dirs_exist_ok=True)
    shutil.copytree(package / "figures", output / "figures", dirs_exist_ok=True)
    return generated, package / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "artifacts/reproduced")
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    work = output / "_work"
    work.mkdir()

    # 1. Canonical cohorts.
    canonical_out, canonical_figs = stage_canonical(work, output)

    # Shared ViT-S result archives.
    extracted = work / "extracted"
    fp32_root = extract(ROOT / "data/vits_fp32_results.zip", extracted / "fp32")
    amp_root = extract(ROOT / "data/vits_amp_results.zip", extracted / "amp")

    # 2. Canonical cross-cohort system comparison and layerwise figures.
    cross_cohort_out = output / "cross_cohort"
    cross_cohort_out.mkdir()
    run([
        sys.executable, ROOT / "analysis/cross_cohort/build_cross_cohort.py",
        "--v18-outputs", canonical_out,
        "--vits-root", fp32_root,
        "--output-dir", cross_cohort_out,
    ])
    run([
        sys.executable, ROOT / "analysis/cross_cohort/generate_figures.py",
        "--output-dir", cross_cohort_out,
        "--d031-point-csv", ROOT / "data/objective_specific/layerwise/layerwise_point_metrics.csv",
    ])

    # 3. Within-ViT-S training-regime comparison.
    precision_out = output / "precision_bridge"
    precision_out.mkdir()
    run([
        sys.executable, ROOT / "analysis/precision_bridge/run_precision_bridge.py",
        "--fp16-root", amp_root,
        "--fp32-root", fp32_root,
        "--out", precision_out,
        "--sesoi-lock", ROOT / "analysis/precision_bridge/equivalence_margin_lock.json",
    ])

    # 4. AMP-matched cross-architecture comparison.
    cross_arch_out = output / "cross_architecture"
    cross_arch_out.mkdir()
    run([
        sys.executable, ROOT / "analysis/cross_architecture/build_amp_matched_cross_architecture.py",
        "--vitb-adversarial-csv", canonical_out / "adversarial_native_budget_points_v18.csv",
        "--vitb-noise-csv", canonical_out / "noise_budget_paired_points_v18.csv",
        "--vitb-clean-csv", ROOT / "data/clean_accuracy_eval_subset_seed_level.csv",
        "--vitb-pipeline-report", canonical_out / "CANONICAL_PIPELINE_REPORT_v18.json",
        "--vits-amp-zip", ROOT / "data/vits_amp_results.zip",
        "--d033-provenance", ROOT / "data/provenance_report.json",
        "--attack-eval-mode", ROOT / "configs/attack_eval_numerical_mode.json",
        "--out", cross_arch_out,
    ])
    run([
        sys.executable, ROOT / "analysis/cross_architecture/augment_d059_v2.py",
        "--v1-out", cross_arch_out,
        "--out", cross_arch_out,
    ])
    run([
        sys.executable, ROOT / "analysis/cross_architecture/qa_d059_v2.py",
        "--out", cross_arch_out,
        "--report", cross_arch_out / "cross_architecture_QA.json",
    ])
    run([
        sys.executable, ROOT / "analysis/cross_architecture/plot_d059_figures.py",
        "--analysis", cross_arch_out,
        "--out", output / "figures",
    ])

    # Public manuscript-asset aliases.
    assets = output / "manuscript_assets"
    assets.mkdir()
    aliases = {
        output / "figures/fig_primary_robustness_curves.pdf": assets / "fig_primary_robustness_curves.pdf",
        cross_cohort_out / "fig_cross_architecture_robustness.pdf": assets / "fig_cross_architecture_robustness.pdf",
        output / "figures/fig_D059_attack_nauc_amp_matched.pdf": assets / "fig_amp_matched_attack_nauc.pdf",
        cross_cohort_out / "fig_layerwise_displacement_profiles.pdf": assets / "fig_layerwise_displacement_profiles.pdf",
        canonical_out / "table_primary_nauc_v18.tex": assets / "table_primary_nauc.tex",
        canonical_out / "table_cifar_wide_v18.tex": assets / "table_cifar_wide.tex",
    }
    for src, dst in aliases.items():
        shutil.copy2(src, dst)

    report = {
        "status": "PASS_REPRODUCTION_COMPLETE",
        "canonical": str(canonical_out.relative_to(output)),
        "cross_cohort": str(cross_cohort_out.relative_to(output)),
        "precision_bridge": str(precision_out.relative_to(output)),
        "cross_architecture": str(cross_arch_out.relative_to(output)),
        "manuscript_assets": str(assets.relative_to(output)),
        "new_gpu_experiment_required": False,
    }
    (output / "REPRODUCTION_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    run([sys.executable, ROOT / "scripts/verify_reproduction.py"])
    if not args.keep_work:
        shutil.rmtree(work)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
