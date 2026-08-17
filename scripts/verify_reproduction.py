#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    out = ROOT / "artifacts/reproduced"
    if not out.exists():
        raise SystemExit("Run scripts/reproduce_all.py first.")

    checks: list[dict] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})
        prefix = "PASS" if condition else "FAIL"
        print(prefix + ": " + name + (f" — {detail}" if detail else ""))

    exact_pairs = [
        (
            "canonical primary aggregate",
            ROOT / "analysis/canonical_cohorts/reference_outputs/primary_aggregate_nauc_v18.csv",
            out / "canonical/primary_aggregate_nauc_v18.csv",
        ),
        (
            "canonical wider-range aggregate",
            ROOT / "analysis/canonical_cohorts/reference_outputs/cifar_wide_aggregate_nauc_v18.csv",
            out / "canonical/cifar_wide_aggregate_nauc_v18.csv",
        ),
        (
            "canonical seed-level nAUC",
            ROOT / "analysis/canonical_cohorts/reference_outputs/seed_level_nauc_v18.csv",
            out / "canonical/seed_level_nauc_v18.csv",
        ),
        (
            "canonical statistics",
            ROOT / "analysis/canonical_cohorts/reference_outputs/canonical_statistics_v18.json",
            out / "canonical/canonical_statistics_v18.json",
        ),
        (
            "no-envelope aggregate",
            ROOT / "analysis/canonical_cohorts/reference_outputs/no_envelope_aggregate_nauc.csv",
            out / "canonical/no_envelope_aggregate_nauc.csv",
        ),
        (
            "no-envelope audit report",
            ROOT / "analysis/canonical_cohorts/reference_outputs/no_envelope_sensitivity_report.json",
            out / "canonical/no_envelope_sensitivity_report.json",
        ),
        (
            "precision bridge summary",
            ROOT / "analysis/precision_bridge/reference_outputs/D029_ANALYSIS_SUMMARY_v1.json",
            out / "precision_bridge/D029_ANALYSIS_SUMMARY_v1.json",
        ),
        (
            "precision bridge family endpoints",
            ROOT / "analysis/precision_bridge/reference_outputs/D029_FAMILY_ENDPOINT_STATISTICS_v1.csv",
            out / "precision_bridge/D029_FAMILY_ENDPOINT_STATISTICS_v1.csv",
        ),
        (
            "cross-architecture main table",
            ROOT / "analysis/cross_architecture/reference_outputs/D059_ATTACK_MAIN_TABLE_v2.csv",
            out / "cross_architecture/D059_ATTACK_MAIN_TABLE_v2.csv",
        ),
        (
            "cross-architecture summary",
            ROOT / "analysis/cross_architecture/reference_outputs/D059_ANALYSIS_SUMMARY_v2.json",
            out / "cross_architecture/D059_ANALYSIS_SUMMARY_v2.json",
        ),
        (
            "cross-architecture Welch sensitivity",
            ROOT / "analysis/cross_architecture/reference_outputs/D059_WELCH_SENSITIVITY_v2.csv",
            out / "cross_architecture/D059_WELCH_SENSITIVITY_v2.csv",
        ),
        (
            "cross-architecture ordinal reconstruction",
            ROOT / "analysis/cross_architecture/reference_outputs/D059_INTERACTION_ORDINAL_ROBUSTNESS_v2.csv",
            out / "cross_architecture/D059_INTERACTION_ORDINAL_ROBUSTNESS_v2.csv",
        ),
        (
            "clean-accuracy aggregate",
            ROOT / "analysis/clean_accuracy/reference_outputs/clean_accuracy_aggregate.csv",
            out / "clean_accuracy/clean_accuracy_aggregate.csv",
        ),
        (
            "clean-accuracy report",
            ROOT / "analysis/clean_accuracy/reference_outputs/clean_accuracy_report.json",
            out / "clean_accuracy/clean_accuracy_report.json",
        ),
        (
            "post-lock seed-level sensitivity",
            ROOT / "analysis/cross_cohort/reference_outputs/postlock_grid_sensitivity_seed_level.csv",
            out / "postlock_grid/postlock_grid_sensitivity_seed_level.csv",
        ),
        (
            "post-lock sensitivity report",
            ROOT / "analysis/cross_cohort/reference_outputs/postlock_grid_sensitivity_report.json",
            out / "postlock_grid/postlock_grid_sensitivity_report.json",
        ),
        (
            "local-geometry verification report",
            ROOT / "analysis/local_geometry/reference_outputs/geometry_public_verification.json",
            out / "local_geometry/geometry_public_verification.json",
        ),
        (
            "local-geometry radius sensitivity",
            ROOT / "analysis/local_geometry/reference_outputs/geometry_radius_sensitivity.csv",
            out / "local_geometry/geometry_radius_sensitivity.csv",
        ),
    ]

    table_names = [
        "table_primary_nauc.tex",
        "table_cifar_wide.tex",
        "table_clean_accuracy.tex",
        "table_factorial_comparison_design.tex",
        "table_cross_architecture_nauc.tex",
        "table_amp_matched_nauc.tex",
        "table_objective_comparison.tex",
        "table_s_welch.tex",
        "table_s_orders.tex",
        "table_s_did.tex",
        "table_s_dispersion.tex",
        "table_s_signs.tex",
        "table_no_envelope_sensitivity.tex",
        "table_postlock_grid_sensitivity.tex",
    ]
    for name in table_names:
        exact_pairs.append(
            (
                f"manuscript table {name}",
                ROOT / "manuscript/tables" / name,
                out / "manuscript_assets" / name,
            )
        )

    for name, reference, generated in exact_pairs:
        ok = (
            reference.exists()
            and generated.exists()
            and sha256(reference) == sha256(generated)
        )
        detail = sha256(generated) if generated.exists() else "missing"
        check(name, ok, detail)

    qa = json.loads(
        (out / "cross_architecture/cross_architecture_QA.json").read_text()
    )
    check(
        "cross-architecture 31/31 QA",
        qa["status"] == "PASS" and qa["passed"] == qa["n_checks"] == 31,
    )

    geometry = json.loads(
        (out / "local_geometry/geometry_public_verification.json").read_text()
    )
    check(
        "local geometry 1,584 directions / 48 checkpoints",
        geometry["status"] == "PASS"
        and geometry["direction_rows"] == 1584
        and geometry["checkpoint_groups"] == 48
        and geometry["probe_radius_verified"]
        and geometry["public_processed_artifact_hash_chain"] == "PASS"
        and geometry["full_raw_checkpoint_processed_hash_chain"] == "PASS"
        and geometry["raw_to_direction_level_reconstruction"] == "PASS_1584_OF_1584"
        and geometry["checkpoint_digest_status"] == "PASS_48_OF_48"
        and geometry["historical_checkpoint_crosscheck"] == "PASS_6_OF_6"
        and geometry["radius_sensitivity_family_order_preserved"],
    )

    postlock = json.loads(
        (out / "postlock_grid/postlock_grid_sensitivity_report.json").read_text()
    )
    check(
        "post-lock grid sensitivity strengthens separation",
        postlock["status"] == "PASS"
        and postlock["all_seed_deltas_negative"]
        and not postlock["qualitative_conclusion_changed"],
    )

    report = {
        "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
        "n_checks": len(checks),
        "passed": sum(item["pass"] for item in checks),
        "checks": checks,
        "note": (
            "PDF visual identity is checked in the repository release audit; "
            "PDF byte hashes can differ because of metadata."
        ),
    }
    (out / "REPRODUCTION_VERIFICATION_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
