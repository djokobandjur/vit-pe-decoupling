#!/usr/bin/env python3
from __future__ import annotations
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS: {message}")


def main() -> None:
    required = [
        "README.md", "LICENSE", "CITATION.cff", "environment.yml",
        "manuscript/pe_robustness_nn_main.tex",
        "manuscript/pe_robustness_nn_supplement.tex",
        "analysis/canonical_cohorts/canonical_analysis.py",
        "analysis/precision_bridge/run_precision_bridge.py",
        "analysis/cross_architecture/build_amp_matched_cross_architecture.py",
        "data/vits_fp32_results.zip", "data/vits_amp_results.zip",
    ]
    for rel in required:
        check((ROOT / rel).exists(), f"required file exists: {rel}")

    manuscript = (ROOT / "manuscript/pe_robustness_nn_main.tex").read_text(encoding="utf-8")
    check(re.search(r"\bD\d{3}[a-z]?\b", manuscript) is None,
          "reviewer-facing manuscript contains no internal D-codes")
    check("canonical v19 reproducibility package" not in manuscript.lower(),
          "reviewer-facing manuscript contains no obsolete package-version wording")

    canonical = list(csv.DictReader((ROOT / "analysis/canonical_cohorts/reference_outputs/primary_aggregate_nauc_v18.csv").open()))
    row = next(r for r in canonical if r["dataset"] == "cifar" and r["pe_family"] == "alibi")
    check(abs(float(row["noise_nauc_mean"]) - 0.993525541) < 1e-8,
          "canonical CIFAR-100 ALiBi nAUC sentinel")

    precision = json.loads((ROOT / "analysis/precision_bridge/reference_outputs/D029_ANALYSIS_SUMMARY_v1.json").read_text())
    check(precision["status"] == "PASS_ANALYSIS_COMPLETE", "precision bridge status")
    statuses = {r["family"]: r["equivalence_status"] for r in precision["family_rho50_statistics"]}
    check(statuses == {"learned": "INCONCLUSIVE", "sinusoidal": "EQUIVALENT", "rope": "INCONCLUSIVE"},
          "family-specific precision-bridge decisions")

    qa_ref = ROOT / "analysis/cross_architecture/reference_outputs/D059_V2_QA_REPORT.json"
    if qa_ref.exists():
        qa = json.loads(qa_ref.read_text())
        check(qa["status"] == "PASS" and qa["passed"] == qa["n_checks"],
              "AMP-matched cross-architecture QA")

    restart = json.loads((ROOT / "data/objective_specific/all_restart_audit/FINAL_AUDIT_REPORT_v1.json").read_text())
    check(restart.get("status") in {"PASS", "ok", "PASS_AUDIT_COMPLETE", "PASS_COMPLETE"},
          "all-restart audit status")

    direct = json.loads((ROOT / "data/objective_specific/direct_displacement/D020_VITS_FORMAL_CENTRAL_ENDPOINT_v1.json").read_text())
    text = json.dumps(direct)
    check("0.0095" in text and "0.02" in text, "direct-displacement central endpoints present")

    print("\nRepository verification completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
