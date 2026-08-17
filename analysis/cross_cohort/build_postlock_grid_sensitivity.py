#!/usr/bin/env python3
"""Quantify the post-lock ViT-S Sinusoidal budget-0.020 sensitivity.

The primary cross-cohort analysis uses the pre-specified canonical ViT-S
native-budget grid, ending at 0.010 for Sinusoidal. The later all-restart audit
also evaluated budget 0.020 with the same task-loss objective, checkpoint
cohort, split, and 200x5 schedule. This script adds only the loss-selected
0.020 points as a post-lock sensitivity analysis and recomputes adversarial
nAUC on the already locked cross-architecture support.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def near(a: float, b: float) -> bool:
    return abs(a - b) <= max(5e-7, 5e-6 * max(abs(a), abs(b), 1e-12))


def dedupe(points: pd.DataFrame) -> pd.DataFrame:
    points = points.sort_values(["rho", "budget", "source_index"]).reset_index(drop=True)
    groups: list[list[pd.Series]] = []
    for _, row in points.iterrows():
        if not groups or not near(float(groups[-1][0]["rho"]), float(row["rho"])):
            groups.append([row])
        else:
            groups[-1].append(row)
    selected = []
    for group in groups:
        # Same loss-selected canonical rule: at near-identical rho, retain the
        # lowest accuracy, then highest task loss.
        chosen = min(
            group,
            key=lambda row: (
                float(row["accuracy"]),
                -float(row["attack_loss"]),
                float(row["budget"]),
                int(row["source_index"]),
            ),
        )
        selected.append(dict(chosen))
    return pd.DataFrame(selected).sort_values("rho").reset_index(drop=True)


def curve(points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    points = points[
        np.isfinite(points["rho"])
        & np.isfinite(points["accuracy"])
        & (points["rho"] > 0)
    ].copy()
    points = dedupe(points)
    x = np.concatenate(([0.0], points["rho"].to_numpy(float)))
    y = np.concatenate(([1.0], points["accuracy"].to_numpy(float)))
    y = np.minimum.accumulate(y)
    if not np.all(np.diff(x) > 0):
        raise ValueError("rho coordinates are not strictly increasing")
    return x, y


def nauc(x: np.ndarray, y: np.ndarray, endpoint: float) -> float:
    if x[-1] < endpoint:
        raise ValueError(f"curve support {x[-1]} is below {endpoint}")
    inside = x < endpoint
    xc = np.concatenate((x[inside], [endpoint]))
    yc = np.concatenate((y[inside], [np.interp(endpoint, x, y)]))
    return float(np.trapezoid(yc, xc) / endpoint)


def load_vits_sinusoidal(zip_path: Path) -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith("attacks_sinusoidal.json"):
                continue
            data = json.loads(archive.read(info.filename).decode("utf-8"))
            family_data = data["results"]["sinusoidal"]
            for seed_string, result in family_data.items():
                seed = int(seed_string)
                for source_index, point in enumerate(result["attacks"]["pgd_pe"].values()):
                    budget = float(point["budget"])
                    if budget == 0:
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "budget": budget,
                            "rho": float(point["rho"]["rho_rel"]),
                            "accuracy": float(point["normalized_accuracy"]),
                            "attack_loss": float(point["selected_attack_loss"]),
                            "source_index": source_index,
                        }
                    )
    frame = pd.DataFrame(rows)
    if len(frame["seed"].unique()) != 6:
        raise ValueError("Expected six ViT-S Sinusoidal seeds")
    return frame


def write_table(frame: pd.DataFrame, report: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Post-lock sensitivity to adding the loss-selected ViT-S/16",
        r"Sinusoidal task-loss point at native budget 0.020. The primary column",
        r"uses the prespecified canonical grid ending at 0.010; the sensitivity",
        r"column adds the later all-restart audit point while keeping the locked",
        r"cross-architecture endpoint $\rho_{\mathrm{rel}}=0.0792384$.}",
        r"\label{tab:s-postlock-grid}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Seed & Audit $\rho_{\mathrm{rel}}$ & Audit normalized accuracy & "
        r"Primary adversarial nAUC & With 0.020 point & Difference \\",
        r"\midrule",
    ]
    for _, row in frame.sort_values("seed").iterrows():
        lines.append(
            f"{int(row.seed)} & {row.audit_rho_rel:.6f} & "
            f"{row.audit_normalized_accuracy:.6f} & "
            f"{row.primary_adversarial_nauc:.6f} & "
            f"{row.augmented_adversarial_nauc:.6f} & "
            f"{row.delta_nauc:+.6f} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{3}{r}{Across-seed mean $\pm$ sample SD} & "
            f"${report['primary_mean']:.6f}\\pm{report['primary_sd']:.6f}$ & "
            f"${report['augmented_mean']:.6f}\\pm{report['augmented_sd']:.6f}$ & "
            f"${report['delta_mean']:+.6f}$ \\\\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--vits-zip", type=Path, default=root / "data/vits_fp32_results.zip"
    )
    parser.add_argument(
        "--audit-points",
        type=Path,
        default=root
        / "data/objective_specific/all_restart_audit/POINT_SELECTION_SUMMARY_v1.csv",
    )
    parser.add_argument(
        "--primary-seed-level",
        type=Path,
        default=root
        / "analysis/cross_cohort/reference_bundle/v19_support_aware_seed_level_nauc.csv",
    )
    parser.add_argument(
        "--build-report",
        type=Path,
        default=root
        / "analysis/cross_cohort/reference_bundle/V19_SUPPORT_AWARE_BUILD_REPORT.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "analysis/cross_cohort/reference_outputs",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    endpoint = float(
        json.loads(args.build_report.read_text())["rho_common"]["cross_architecture"]
    )
    canonical_points = load_vits_sinusoidal(args.vits_zip)
    audit = pd.read_csv(args.audit_points)
    audit = audit.query("architecture == 'vits' and budget == 0.020").copy()
    if len(audit) != 6:
        raise ValueError(f"Expected six audit points, found {len(audit)}")

    locked = pd.read_csv(args.primary_seed_level)
    locked = locked.query(
        "scope == 'cross' and endpoint_multiplier == 1.0 "
        "and architecture == 'vits' and family == 'sinusoidal'"
    ).copy()
    if len(locked) != 6:
        raise ValueError("Expected six locked primary seed rows")

    rows = []
    for seed in sorted(locked["seed"].astype(int)):
        points = canonical_points.query("seed == @seed").copy()
        x, y = curve(points)
        primary_recomputed = nauc(x, y, endpoint)
        primary_locked = float(
            locked.loc[locked["seed"] == seed, "adversarial_nauc"].iloc[0]
        )
        if not np.isclose(primary_recomputed, primary_locked, atol=1e-12, rtol=1e-12):
            raise AssertionError(f"Primary nAUC mismatch for seed {seed}")

        audit_row = audit.loc[audit["seed"] == seed].iloc[0]
        added = {
            "seed": seed,
            "budget": 0.020,
            "rho": float(audit_row["loss_selected_rho_rel"]),
            "accuracy": float(audit_row["loss_selected_normalized_accuracy"]),
            "attack_loss": float(audit_row["loss_selected_attack_loss"]),
            "source_index": 10_000,
        }
        augmented_points = pd.concat(
            [points, pd.DataFrame([added])], ignore_index=True
        )
        xa, ya = curve(augmented_points)
        augmented = nauc(xa, ya, endpoint)
        rows.append(
            {
                "seed": seed,
                "audit_rho_rel": added["rho"],
                "audit_normalized_accuracy": added["accuracy"],
                "primary_adversarial_nauc": primary_locked,
                "augmented_adversarial_nauc": augmented,
                "delta_nauc": augmented - primary_locked,
            }
        )

    frame = pd.DataFrame(rows)
    primary = frame["primary_adversarial_nauc"].to_numpy(float)
    augmented = frame["augmented_adversarial_nauc"].to_numpy(float)
    report = {
        "status": "PASS",
        "interpretation": (
            "The 0.020 points were collected after the canonical grid was "
            "locked. Adding them strengthens the reported Sinusoidal "
            "random-versus-adversarial separation and leaves the qualitative "
            "family ordering unchanged."
        ),
        "locked_cross_endpoint": endpoint,
        "canonical_vits_sinusoidal_native_grid_max": 0.010,
        "postlock_audit_budget": 0.020,
        "same_task_loss_objective": True,
        "same_schedule": "200x5",
        "same_split_and_checkpoint_cohort": True,
        "primary_mean": float(primary.mean()),
        "primary_sd": float(primary.std(ddof=1)),
        "augmented_mean": float(augmented.mean()),
        "augmented_sd": float(augmented.std(ddof=1)),
        "delta_mean": float((augmented - primary).mean()),
        "all_seed_deltas_negative": bool(((augmented - primary) < 0).all()),
        "qualitative_conclusion_changed": False,
    }
    if not np.isclose(report["primary_mean"], 0.7663660668996223, atol=1e-12, rtol=0.0):
        raise AssertionError("Locked primary mean sentinel mismatch")
    if not np.isclose(report["augmented_mean"], 0.7217772442386754, atol=1e-12, rtol=0.0):
        raise AssertionError("Augmented mean sentinel mismatch")

    frame.to_csv(args.output_dir / "postlock_grid_sensitivity_seed_level.csv", index=False)
    (args.output_dir / "postlock_grid_sensitivity_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_table(
        frame,
        report,
        args.output_dir / "table_postlock_grid_sensitivity.tex",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
