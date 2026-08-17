#!/usr/bin/env python3
"""Build the deterministic no-envelope sensitivity summary.

The canonical seed-level output already contains both the primary adversarial
nAUC after the cumulative lower envelope and the raw selected-restart
adversarial nAUC before that envelope. This script aggregates those columns,
checks that seed-level gap signs and within-seed family rankings are preserved,
and writes the supplementary table and a machine-readable audit report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 123, 456, 789, 1011, 1213]
FAMILIES = ["learned", "sinusoidal", "rope", "alibi"]
FAMILY_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "rope": "RoPE",
    "alibi": "ALiBi",
}
DATASETS = ["imagenet", "cifar"]
DATASET_LABELS = {"imagenet": "ImageNet-100", "cifar": "CIFAR-100"}
ENDPOINTS = [0.09, 0.095]


def suffix(endpoint: float) -> str:
    return str(endpoint).replace(".", "p")


def rank_signature(values: np.ndarray) -> tuple[int, ...]:
    if len(np.unique(values)) != len(values):
        raise ValueError("Unexpected tie in family gaps")
    return tuple((np.argsort(np.argsort(values)) + 1).tolist())


def friedman_q(matrix: np.ndarray) -> float:
    ranks = np.vstack([rank_signature(row) for row in matrix])
    n_blocks, n_families = ranks.shape
    rank_sums = ranks.sum(axis=0)
    return float(
        12.0 / (n_blocks * n_families * (n_families + 1))
        * np.sum(rank_sums.astype(float) ** 2)
        - 3.0 * n_blocks * (n_families + 1)
    )


def build_summary(seed_level: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    required_rows = len(DATASETS) * len(FAMILIES) * len(SEEDS)
    if len(seed_level) != required_rows:
        raise ValueError(f"Expected {required_rows} rows, found {len(seed_level)}")

    rows: list[dict] = []
    rank_checks: dict[str, dict] = {}
    all_positive = True
    all_ranks_preserved = True

    for endpoint in ENDPOINTS:
        tag = suffix(endpoint)
        for dataset in DATASETS:
            env_pivot = (
                seed_level.query("dataset == @dataset")
                .pivot(index="seed", columns="pe_family", values=f"gap_{tag}")
                .reindex(index=SEEDS, columns=FAMILIES)
            )
            raw_pivot = (
                seed_level.query("dataset == @dataset")
                .pivot(index="seed", columns="pe_family", values=f"gap_raw_{tag}")
                .reindex(index=SEEDS, columns=FAMILIES)
            )
            if env_pivot.isna().any().any() or raw_pivot.isna().any().any():
                raise ValueError(f"Incomplete pivot for {dataset}/{endpoint}")

            env_signatures = [
                rank_signature(row) for row in env_pivot.to_numpy(dtype=float)
            ]
            raw_signatures = [
                rank_signature(row) for row in raw_pivot.to_numpy(dtype=float)
            ]
            preserved = env_signatures == raw_signatures
            all_ranks_preserved &= preserved

            rank_checks[f"{dataset}_{tag}"] = {
                "rank_configurations_preserved": preserved,
                "envelope_friedman_q": friedman_q(
                    env_pivot.to_numpy(dtype=float)
                ),
                "raw_friedman_q": friedman_q(raw_pivot.to_numpy(dtype=float)),
            }

            for family in FAMILIES:
                subset = seed_level.query(
                    "dataset == @dataset and pe_family == @family"
                ).sort_values("seed")
                env = subset[f"gap_{tag}"].to_numpy(dtype=float)
                raw = subset[f"gap_raw_{tag}"].to_numpy(dtype=float)
                positive = int(np.sum(raw > 0))
                all_positive &= positive == len(SEEDS)
                rows.append(
                    {
                        "endpoint": endpoint,
                        "dataset": dataset,
                        "pe_family": family,
                        "n_seeds": len(raw),
                        "envelope_gap_mean": float(env.mean()),
                        "raw_gap_mean": float(raw.mean()),
                        "raw_minus_envelope": float((raw - env).mean()),
                        "positive_raw_gap_seeds": positive,
                        "minimum_raw_gap": float(raw.min()),
                    }
                )

    summary = pd.DataFrame(rows)
    max_changes = {}
    for endpoint in ENDPOINTS:
        subset = summary.query("endpoint == @endpoint")
        row = subset.loc[subset["raw_minus_envelope"].abs().idxmax()]
        max_changes[suffix(endpoint)] = {
            "dataset": row["dataset"],
            "pe_family": row["pe_family"],
            "envelope_gap_mean": float(row["envelope_gap_mean"]),
            "raw_gap_mean": float(row["raw_gap_mean"]),
            "raw_minus_envelope": float(row["raw_minus_envelope"]),
        }

    report = {
        "status": (
            "PASS"
            if all_positive and all_ranks_preserved
            else "FAIL"
        ),
        "seed_level_rows": int(len(seed_level)),
        "all_48_raw_gaps_positive_at_0p09": bool(
            (
                seed_level["gap_raw_0p09"].to_numpy(dtype=float) > 0
            ).all()
        ),
        "all_48_raw_gaps_positive_at_0p095": bool(
            (
                seed_level["gap_raw_0p095"].to_numpy(dtype=float) > 0
            ).all()
        ),
        "all_within_seed_family_ranks_preserved": bool(all_ranks_preserved),
        "rank_checks": rank_checks,
        "largest_family_mean_changes": max_changes,
        "interpretation": (
            "The positive-gap direction and between-family rank structure "
            "do not depend on the adversarial cumulative lower envelope."
        ),
    }
    return summary, report


def write_table(summary: pd.DataFrame, path: Path) -> None:
    primary = summary.query("endpoint == 0.09").set_index(
        ["dataset", "pe_family"]
    )
    sensitivity = summary.query("endpoint == 0.095").set_index(
        ["dataset", "pe_family"]
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Deterministic sensitivity to omitting the adversarial cumulative",
        r"lower envelope. Values are family-mean noise-minus-adversarial nAUC",
        r"gaps across six seeds. ``Raw'' connects the same loss-selected",
        r"native-budget points without enforcing monotonicity; $\Delta$ is raw",
        r"minus envelope. Every raw seed-level gap is positive at both endpoints.}",
        r"\label{tab:s-no-envelope}",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrr rr r c}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{$\rho_{\max}=0.09$} & "
        r"\multicolumn{3}{c}{$\rho_{\max}=0.095$} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Dataset & PE family & Envelope gap & Raw gap & $\Delta$ & "
        r"Envelope gap & Raw gap & $\Delta$ & Raw gap $>0$ \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for family_index, family in enumerate(FAMILIES):
            p = primary.loc[(dataset, family)]
            s = sensitivity.loc[(dataset, family)]
            dataset_label = DATASET_LABELS[dataset] if family_index == 0 else ""
            lines.append(
                f"{dataset_label} & {FAMILY_LABELS[family]} "
                f"& {p.envelope_gap_mean:.6f} "
                f"& {p.raw_gap_mean:.6f} "
                f"& {p.raw_minus_envelope:+.6f} "
                f"& {s.envelope_gap_mean:.6f} "
                f"& {s.raw_gap_mean:.6f} "
                f"& {s.raw_minus_envelope:+.6f} "
                f"& {int(p.positive_raw_gap_seeds)}/6 \\\\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-level", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_level = pd.read_csv(args.seed_level)
    summary, report = build_summary(seed_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        args.output_dir / "no_envelope_aggregate_nauc.csv", index=False
    )
    write_table(
        summary,
        args.output_dir / "table_no_envelope_sensitivity.tex",
    )
    (args.output_dir / "no_envelope_sensitivity_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    if report["status"] != "PASS":
        raise SystemExit(1)
    print("No-envelope sensitivity: PASS")


if __name__ == "__main__":
    main()
