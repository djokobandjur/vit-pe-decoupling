#!/usr/bin/env python3
"""Canonical v18 curve, table, and figure generation pipeline.

This script is the single numerical path for:
  * the primary nAUC table over rho_rel in [0, 0.09],
  * the CIFAR-100 wider-range table over rho_rel in [0, 0.23], and
  * the 4x2 primary robustness figure displayed over [0, 0.095].

Random-noise aggregation is budget-paired: within each trained seed and
native budget, achieved rho and normalized accuracy are averaged over the ten
fixed Gaussian directions before curve construction. Adversarial aggregation
retains the highest-loss restart already stored at each native budget and then
applies the cumulative lower accuracy envelope after ordering by achieved rho.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

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
EXPECTED_SPLITS = {
    "imagenet": {
        "n_total_images": 5000,
        "n_calibration": 256,
        "n_attack": 1280,
        "n_eval": 3464,
    },
    "cifar": {
        "n_total_images": 10000,
        "n_calibration": 256,
        "n_attack": 1280,
        "n_eval": 8464,
    },
}
EXPECTED_STEPS = {
    "imagenet": {"learned": 400, "sinusoidal": 400, "rope": 200, "alibi": 200},
    "cifar": {"learned": 400, "sinusoidal": 200, "rope": 100, "alibi": 100},
}
PRIMARY_ENDPOINT = 0.09
SENSITIVITY_ENDPOINT = 0.095
WIDE_ENDPOINT = 0.23
FIGURE_DISPLAY_MAX = 0.095
FIGURE_GRID_POINTS = 401
BOOTSTRAP_REPS = 200_000
BOOTSTRAP_SEED = 20260718


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_source_class(path: Path) -> str:
    parts = set(path.parts)
    if "imagenet_base" in parts or "cifar_base" in parts:
        return "base"
    if "densification" in parts:
        return "densification"
    if "cifar_midpoint" in parts:
        return "transition_midpoint_h200"
    raise ValueError(f"Unrecognized input source class: {path}")


def selected_inputs(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON inputs found under {input_dir}")
    return paths


def validate_result_file(data: dict, path: Path) -> tuple[str, list[str]]:
    metadata = data["metadata"]
    config = metadata["config"]
    split = metadata["split"]
    dataset = metadata["dataset"]

    if metadata["protocol"] != (
        "parameter-space perturbation reparameterised "
        "by achieved attention-logit rho"
    ):
        raise ValueError(f"Unexpected protocol in {path}")
    if dataset not in EXPECTED_SPLITS:
        raise ValueError(f"Unexpected dataset in {path}: {dataset}")

    expected = EXPECTED_SPLITS[dataset]
    if metadata["n_total_images"] != expected["n_total_images"]:
        raise ValueError(f"Wrong total image count in {path}")
    if (
        split["n_calibration"],
        split["n_attack"],
        split["n_eval"],
    ) != (
        expected["n_calibration"],
        expected["n_attack"],
        expected["n_eval"],
    ):
        raise ValueError(f"Wrong split sizes in {path}")
    for field in (
        "calibration_attack_overlap",
        "calibration_eval_overlap",
        "attack_eval_overlap",
    ):
        if split[field] != 0:
            raise ValueError(f"Nonzero split overlap {field} in {path}")

    if config["parameter_norm"] != "global_rms":
        raise ValueError(f"Unexpected parameter norm in {path}")
    if config["rope_parameterization"] != "phase":
        raise ValueError(f"Unexpected RoPE parameterization in {path}")
    if config["inactive_rope_inv_freq_excluded"] is not True:
        raise ValueError(f"RoPE inv_freq exclusion not confirmed in {path}")

    stages = list(config.get("stages", []))
    if "attacks" in stages:
        for family in data.get("results", {}):
            if config["pgd_steps"] != EXPECTED_STEPS[dataset][family]:
                raise ValueError(
                    f"Wrong PGD steps in {path}: {config['pgd_steps']} "
                    f"for {dataset}/{family}"
                )
            if config["pgd_restarts"] != 5:
                raise ValueError(f"Wrong restart count in {path}")
            if not np.isclose(config["pgd_alpha_ratio"], 0.05):
                raise ValueError(f"Wrong PGD alpha ratio in {path}")
    return dataset, stages


def load_measurements(input_dir: Path, package_dir: Path):
    noise_records: list[dict] = []
    attack_records: list[dict] = []
    manifest_rows: list[dict] = []
    clean_accuracy: dict[tuple[str, str, int], float] = {}

    for path in selected_inputs(input_dir):
        source_class = input_source_class(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        dataset, stages = validate_result_file(data, path)

        manifest_rows.append(
            {
                "package_relative_path": str(path.relative_to(package_dir)),
                "source_class": source_class,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

        for family, family_results in data["results"].items():
            if family not in FAMILIES:
                raise ValueError(f"Unexpected family {family} in {path}")
            for seed_string, seed_result in family_results.items():
                seed = int(seed_string)
                if seed not in SEEDS or seed_result.get("status") != "ok":
                    raise ValueError(f"Unexpected seed/status in {path}: {seed_string}")

                key = (dataset, family, seed)
                clean = float(seed_result["clean_acc_eval"])
                if key in clean_accuracy and not np.isclose(
                    clean_accuracy[key], clean, atol=1e-12, rtol=0.0
                ):
                    raise ValueError(
                        f"Conflicting clean_acc_eval for {key}: "
                        f"{clean_accuracy[key]} vs {clean} in {path}"
                    )
                clean_accuracy.setdefault(key, clean)

                if "noise" in stages:
                    draws = seed_result["noise"]["draws"]
                    if len(draws) != 10:
                        raise ValueError(f"Expected 10 noise draws in {path}")
                    for draw in draws:
                        draw_seed = int(draw["seed"])
                        for point in draw["points"].values():
                            budget = float(point["budget"])
                            if budget == 0.0:
                                continue
                            noise_records.append(
                                {
                                    "dataset": dataset,
                                    "family": family,
                                    "seed": seed,
                                    "draw_seed": draw_seed,
                                    "budget": budget,
                                    "rho": float(point["rho"]["rho_rel"]),
                                    "accuracy": float(point["normalized_accuracy"]),
                                    "source": str(path.relative_to(package_dir)),
                                }
                            )

                if "attacks" in stages:
                    for point in seed_result["attacks"]["pgd_pe"].values():
                        budget = float(point["budget"])
                        if budget == 0.0:
                            continue
                        restart_records = point.get("restart_records", [])
                        if len(restart_records) != 5:
                            raise ValueError(f"Expected 5 restart records in {path}")
                        selected_loss = float(point["selected_attack_loss"])
                        max_restart_loss = max(
                            float(record["attack_loss"])
                            for record in restart_records
                        )
                        if not np.isclose(
                            selected_loss,
                            max_restart_loss,
                            atol=1e-12,
                            rtol=1e-12,
                        ):
                            raise ValueError(
                                f"Selected loss is not max restart loss in {path}"
                            )
                        attack_records.append(
                            {
                                "dataset": dataset,
                                "family": family,
                                "seed": seed,
                                "budget": budget,
                                "rho": float(point["rho"]["rho_rel"]),
                                "accuracy": float(point["normalized_accuracy"]),
                                "attack_loss": selected_loss,
                                "source_class": source_class,
                                "source": str(path.relative_to(package_dir)),
                            }
                        )

    noise_raw = pd.DataFrame(noise_records).drop_duplicates()
    duplicate_noise = noise_raw.duplicated(
        ["dataset", "family", "seed", "draw_seed", "budget"],
        keep=False,
    )
    if duplicate_noise.any():
        raise ValueError(
            "Duplicate/conflicting noise records:\n"
            + noise_raw[duplicate_noise].to_string(index=False)
        )

    # Budget-paired convention: average rho and normalized accuracy jointly
    # over the ten fixed draws at each native budget, within a trained seed.
    noise_budget_paired = (
        noise_raw.groupby(
            ["dataset", "family", "seed", "budget"],
            as_index=False,
        )
        .agg(
            rho=("rho", "mean"),
            accuracy=("accuracy", "mean"),
            n_draws=("draw_seed", "nunique"),
            rho_sd_across_draws=("rho", lambda x: float(np.std(x, ddof=1))),
            accuracy_sd_across_draws=(
                "accuracy", lambda x: float(np.std(x, ddof=1))
            ),
            sources=("source", lambda x: "|".join(sorted(set(x)))),
        )
        .sort_values(["dataset", "family", "seed", "budget"])
    )
    if not (noise_budget_paired["n_draws"] == 10).all():
        raise ValueError("Not every noise budget has ten unique draws")

    attack_raw = pd.DataFrame(attack_records)
    attack_points: list[dict] = []
    for key, group in attack_raw.groupby(
        ["dataset", "family", "seed", "budget"]
    ):
        if len(group) > 1 and not (
            np.allclose(group["rho"], group.iloc[0]["rho"], atol=1e-12, rtol=1e-9)
            and np.allclose(
                group["accuracy"],
                group.iloc[0]["accuracy"],
                atol=1e-12,
                rtol=1e-9,
            )
        ):
            raise ValueError(
                f"Conflicting attack records for {key}:\n"
                + group.to_string(index=False)
            )
        row = group.iloc[0].to_dict()
        row["sources"] = "|".join(sorted(set(group["source"])))
        row["source_classes"] = "|".join(
            sorted(set(group["source_class"]))
        )
        row.pop("source")
        row.pop("source_class")
        attack_points.append(row)

    attack_points_df = pd.DataFrame(attack_points).sort_values(
        ["dataset", "family", "seed", "budget"]
    )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["source_class", "package_relative_path"]
    )

    expected_curve_keys = {
        (dataset, family, seed)
        for dataset in DATASETS
        for family in FAMILIES
        for seed in SEEDS
    }
    observed_noise = set(
        map(
            tuple,
            noise_budget_paired[
                ["dataset", "family", "seed"]
            ].drop_duplicates().to_numpy(),
        )
    )
    observed_attack = set(
        map(
            tuple,
            attack_points_df[
                ["dataset", "family", "seed"]
            ].drop_duplicates().to_numpy(),
        )
    )
    if observed_noise != expected_curve_keys or observed_attack != expected_curve_keys:
        raise ValueError("Incomplete dataset/family/seed coverage")

    return noise_raw, noise_budget_paired, attack_raw, attack_points_df, manifest


def collapse_duplicate_rho(x: np.ndarray, y: np.ndarray, regime: str):
    frame = pd.DataFrame(
        {"rho_key": np.round(x, 13), "rho": x, "accuracy": y}
    )
    accuracy_rule = "min" if regime == "adversarial" else "mean"
    grouped = (
        frame.groupby("rho_key", as_index=False)
        .agg(rho=("rho", "mean"), accuracy=("accuracy", accuracy_rule))
        .sort_values("rho")
    )
    return (
        grouped["rho"].to_numpy(dtype=float),
        grouped["accuracy"].to_numpy(dtype=float),
    )


def construct_curve(points: pd.DataFrame, regime: str, envelope: bool = True):
    x = points["rho"].to_numpy(dtype=float)
    y = points["accuracy"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0.0)
    x, y = x[valid], y[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    x, y = collapse_duplicate_rho(x, y, regime)

    # Exact clean anchor is used for both regimes and is inserted before
    # the adversarial lower-envelope operation.
    x = np.concatenate(([0.0], x))
    y = np.concatenate(([1.0], y))
    if regime == "adversarial" and envelope:
        y = np.minimum.accumulate(y)
    if not np.all(np.diff(x) > 0):
        raise ValueError("Curve rho coordinates are not strictly increasing")
    return x, y


def integrate_nauc(x: np.ndarray, y: np.ndarray, endpoint: float) -> float:
    if x[-1] < endpoint:
        raise ValueError(
            f"Curve support {x[-1]:.9f} is below endpoint {endpoint:.9f}"
        )
    inside = x < endpoint
    x_clip = np.concatenate((x[inside], [endpoint]))
    y_clip = np.concatenate((y[inside], [np.interp(endpoint, x, y)]))
    return float(np.trapezoid(y_clip, x_clip) / endpoint)


def paired_bootstrap_ci(values: np.ndarray, random_seed: int):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPS, len(values))
    )
    bootstrap_means = values[indices].mean(axis=1)
    return tuple(
        map(float, np.quantile(bootstrap_means, [0.025, 0.975]))
    )


def exact_two_sided_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    signs = np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=len(values)))
    )
    null_statistics = np.abs((signs * values[None, :]).mean(axis=1))
    return float(np.mean(null_statistics >= observed - 1e-15))


def exact_friedman_permutation(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    n_blocks, n_families = matrix.shape
    if any(len(np.unique(row)) != n_families for row in matrix):
        raise ValueError("Exact Friedman implementation requires no ties")

    observed_ranks = np.vstack(
        [np.argsort(np.argsort(row)) + 1 for row in matrix]
    ).astype(int)
    observed_rank_sums = observed_ranks.sum(axis=0)

    def statistic(rank_sums):
        return float(
            12.0
            / (n_blocks * n_families * (n_families + 1))
            * np.sum(np.asarray(rank_sums, dtype=float) ** 2)
            - 3.0 * n_blocks * (n_families + 1)
        )

    observed_q = statistic(observed_rank_sums)
    permutations = list(itertools.permutations(range(1, n_families + 1)))
    states = {(0,) * n_families: 1}
    for _ in range(n_blocks):
        next_states = collections.defaultdict(int)
        for rank_sums, count in states.items():
            for permutation in permutations:
                updated = tuple(
                    rank_sums[index] + permutation[index]
                    for index in range(n_families)
                )
                next_states[updated] += count
        states = next_states

    tail_count = sum(
        count
        for rank_sums, count in states.items()
        if statistic(rank_sums) >= observed_q - 1e-12
    )
    total = len(permutations) ** n_blocks
    return {
        "statistic": observed_q,
        "exact_permutation_p_value": float(tail_count / total),
        "exact_tail_count": int(tail_count),
        "n_total_permutations": int(total),
    }


def build_curves_and_statistics(
    noise_points: pd.DataFrame,
    attack_points: pd.DataFrame,
):
    curve_rows: list[dict] = []
    seed_rows: list[dict] = []

    for dataset in DATASETS:
        for family in FAMILIES:
            for seed in SEEDS:
                noise_subset = noise_points.query(
                    "dataset == @dataset and family == @family and seed == @seed"
                )
                attack_subset = attack_points.query(
                    "dataset == @dataset and family == @family and seed == @seed"
                )
                x_noise, y_noise = construct_curve(
                    noise_subset, "noise", envelope=False
                )
                x_adv, y_adv = construct_curve(
                    attack_subset, "adversarial", envelope=True
                )
                x_raw, y_raw = construct_curve(
                    attack_subset, "adversarial", envelope=False
                )

                for regime, x, y, envelope in (
                    ("noise", x_noise, y_noise, False),
                    ("adversarial", x_adv, y_adv, True),
                    ("adversarial_raw", x_raw, y_raw, False),
                ):
                    for index, (rho, accuracy) in enumerate(zip(x, y)):
                        curve_rows.append(
                            {
                                "dataset": dataset,
                                "family": family,
                                "seed": seed,
                                "regime": regime,
                                "point_index": index,
                                "rho": float(rho),
                                "accuracy": float(accuracy),
                                "exact_clean_anchor": index == 0,
                                "adversarial_lower_envelope": envelope,
                            }
                        )

                row = {
                    "dataset": dataset,
                    "pe_family": family,
                    "seed": seed,
                    "noise_max_rho": float(x_noise[-1]),
                    "adversarial_max_rho": float(x_adv[-1]),
                }
                for endpoint in (PRIMARY_ENDPOINT, SENSITIVITY_ENDPOINT):
                    suffix = str(endpoint).replace(".", "p")
                    noise_nauc = integrate_nauc(x_noise, y_noise, endpoint)
                    adv_nauc = integrate_nauc(x_adv, y_adv, endpoint)
                    raw_nauc = integrate_nauc(x_raw, y_raw, endpoint)
                    row.update(
                        {
                            f"noise_nauc_{suffix}": noise_nauc,
                            f"adversarial_nauc_{suffix}": adv_nauc,
                            f"gap_{suffix}": noise_nauc - adv_nauc,
                            f"adversarial_raw_nauc_{suffix}": raw_nauc,
                            f"gap_raw_{suffix}": noise_nauc - raw_nauc,
                        }
                    )
                if dataset == "cifar":
                    noise_wide = integrate_nauc(
                        x_noise, y_noise, WIDE_ENDPOINT
                    )
                    adv_wide = integrate_nauc(x_adv, y_adv, WIDE_ENDPOINT)
                    row.update(
                        {
                            "noise_nauc_0p23": noise_wide,
                            "adversarial_nauc_0p23": adv_wide,
                            "gap_0p23": noise_wide - adv_wide,
                        }
                    )
                seed_rows.append(row)

    curves = pd.DataFrame(curve_rows)
    seed_level = pd.DataFrame(seed_rows)
    aggregate_rows: list[dict] = []
    statistics = {
        "version": "CANONICAL v18",
        "method": {
            "seed_unit_of_replication": True,
            "noise_aggregation": (
                "budget-paired mean of achieved rho and normalized accuracy "
                "over ten fixed draws within seed and native budget"
            ),
            "per_draw_nauc_not_used": True,
            "union_grid_across_noise_draws_not_used": True,
            "exact_clean_anchor": [0.0, 1.0],
            "noise_lower_envelope": False,
            "adversarial_restart_rule": (
                "highest attack loss among five restarts at each native budget"
            ),
            "adversarial_lower_envelope": True,
            "curve_coordinate": "achieved rho_rel",
            "interpolation": "piecewise linear",
            "integration": "numpy.trapezoid",
            "normalized_accuracy_clipping": False,
            "sample_sd_ddof": 1,
            "bootstrap_repetitions": BOOTSTRAP_REPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "friedman": {},
    }

    endpoints = (PRIMARY_ENDPOINT, SENSITIVITY_ENDPOINT)
    for endpoint_index, endpoint in enumerate(endpoints):
        suffix = str(endpoint).replace(".", "p")
        for dataset in DATASETS:
            for family in FAMILIES:
                subset = seed_level.query(
                    "dataset == @dataset and pe_family == @family"
                ).sort_values("seed")
                noise_values = subset[f"noise_nauc_{suffix}"].to_numpy()
                adversarial_values = subset[
                    f"adversarial_nauc_{suffix}"
                ].to_numpy()
                gap_values = noise_values - adversarial_values
                ci_low, ci_high = paired_bootstrap_ci(
                    gap_values,
                    BOOTSTRAP_SEED
                    + endpoint_index * 100
                    + FAMILIES.index(family)
                    + (0 if dataset == "imagenet" else 10),
                )
                aggregate_rows.append(
                    {
                        "endpoint": endpoint,
                        "dataset": dataset,
                        "pe_family": family,
                        "n_seeds": len(subset),
                        "noise_nauc_mean": float(noise_values.mean()),
                        "noise_nauc_sd": float(noise_values.std(ddof=1)),
                        "adversarial_nauc_mean": float(
                            adversarial_values.mean()
                        ),
                        "adversarial_nauc_sd": float(
                            adversarial_values.std(ddof=1)
                        ),
                        "gap_mean": float(gap_values.mean()),
                        "gap_sd": float(gap_values.std(ddof=1)),
                        "gap_ci95_low": ci_low,
                        "gap_ci95_high": ci_high,
                        "positive_gap_seeds": int(np.sum(gap_values > 0)),
                        "exact_sign_flip_p": exact_two_sided_sign_flip(
                            gap_values
                        ),
                    }
                )

            pivot = (
                seed_level.query("dataset == @dataset")
                .pivot(
                    index="seed",
                    columns="pe_family",
                    values=f"gap_{suffix}",
                )
                .reindex(index=SEEDS, columns=FAMILIES)
            )
            exact = exact_friedman_permutation(pivot.to_numpy(dtype=float))
            asymptotic = friedmanchisquare(
                *[pivot[family].to_numpy() for family in FAMILIES]
            )
            if not np.isclose(
                exact["statistic"],
                float(asymptotic.statistic),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ValueError("Exact/asymptotic Friedman Q mismatch")
            statistics["friedman"][f"{dataset}_{suffix}"] = {
                **exact,
                "asymptotic_p_value": float(asymptotic.pvalue),
                "family_order": FAMILIES,
                "seed_order": SEEDS,
            }

    wide_rows = []
    for family in FAMILIES:
        subset = seed_level.query(
            "dataset == 'cifar' and pe_family == @family"
        )
        for regime, column in (
            ("noise", "noise_nauc_0p23"),
            ("adversarial", "adversarial_nauc_0p23"),
        ):
            values = subset[column].to_numpy(dtype=float)
            wide_rows.append(
                {
                    "endpoint": WIDE_ENDPOINT,
                    "dataset": "cifar",
                    "pe_family": family,
                    "regime": regime,
                    "n_seeds": len(values),
                    "nauc_mean": float(values.mean()),
                    "nauc_sd": float(values.std(ddof=1)),
                    "nauc_min": float(values.min()),
                    "nauc_max": float(values.max()),
                }
            )

    return (
        curves,
        seed_level,
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(wide_rows),
        statistics,
    )


def format_value(mean: float, sd: float) -> str:
    return f"${mean:.4f} \\pm {sd:.4f}$"


def write_primary_table(aggregate: pd.DataFrame, path: Path) -> None:
    primary = aggregate[aggregate["endpoint"].eq(PRIMARY_ENDPOINT)]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Primary normalized area under the robustness curve over",
        r"$\rho_{\mathrm{rel}}\in[0,0.09]$. Values are mean $\pm$ sample SD across",
        r"six paired training seeds. Random-noise curves use the budget-paired",
        r"ten-draw aggregation defined in Section~\ref{subsec:statistics}.",
        r"Confidence intervals are paired seed-level bootstrap intervals for the",
        r"noise-minus-adversarial gap and are pointwise, not simultaneous. The",
        r"eight exact two-sided sign-flip tests each yielded the minimum attainable",
        r"unadjusted value $p=0.03125$ at $n=6$; they are secondary",
        r"directional-consistency checks and do not support multiplicity-adjusted",
        r"individual significance claims (see Section~\ref{subsec:statistics}).}",
        r"\label{tab:primary-nauc}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & PE family & Noise nAUC & Adversarial nAUC & Gap & 95\% CI & Seeds with gap $>0$ \\",
        r"\midrule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        for family_index, family in enumerate(FAMILIES):
            row = primary.query(
                "dataset == @dataset and pe_family == @family"
            ).iloc[0]
            dataset_text = DATASET_LABELS[dataset] if family_index == 0 else ""
            if family_index > 0:
                dataset_text = " " * 13
            lines.append(
                f"{dataset_text} & {FAMILY_LABELS[family]} "
                f"& {format_value(row.noise_nauc_mean, row.noise_nauc_sd)} "
                f"& {format_value(row.adversarial_nauc_mean, row.adversarial_nauc_sd)} "
                f"& {format_value(row.gap_mean, row.gap_sd)} "
                f"& $[{row.gap_ci95_low:.4f},{row.gap_ci95_high:.4f}]$ "
                f"& {int(row.positive_gap_seeds)}/6 \\\\" 
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_wide_table(wide: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Wider-range CIFAR-100 ordering-sensitivity analysis over",
        r"$\rho_{\mathrm{rel}}\in[0,0.23]$, the conservative common-support",
        r"endpoint defined in Section~\ref{subsec:statistics}. Values are",
        r"mean $\pm$ sample SD of nAUC across the six training seeds. Random-noise",
        r"curves use the same budget-paired ten-draw aggregation as the primary",
        r"analysis. Both regimes are generated by the locked canonical analysis pipeline",
        r"from the final merged measurement grid, including the completed six-seed",
        r"transition points, without extrapolation. Per-seed values and input",
        r"SHA-256 hashes are provided in the reproducibility package.}",
        r"\label{tab:cifar-wide-range}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"PE family & Random-noise nAUC & Adversarial nAUC \\",
        r"\midrule",
    ]
    for family in FAMILIES:
        noise = wide.query(
            "pe_family == @family and regime == 'noise'"
        ).iloc[0]
        adversarial = wide.query(
            "pe_family == @family and regime == 'adversarial'"
        ).iloc[0]
        lines.append(
            f"{FAMILY_LABELS[family]} "
            f"& {format_value(noise.nauc_mean, noise.nauc_sd)} "
            f"& {format_value(adversarial.nauc_mean, adversarial.nauc_sd)} \\\\" 
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_figure(
    curves: pd.DataFrame,
    figure_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    display_curves = curves[curves["regime"].isin(["noise", "adversarial"])]
    grid = np.linspace(0.0, FIGURE_DISPLAY_MAX, FIGURE_GRID_POINTS)
    plot_rows: list[dict] = []

    for dataset in DATASETS:
        for family in FAMILIES:
            for regime in ("noise", "adversarial"):
                interpolated = []
                for seed in SEEDS:
                    subset = display_curves.query(
                        "dataset == @dataset and family == @family "
                        "and seed == @seed and regime == @regime"
                    ).sort_values("rho")
                    interpolated.append(
                        np.interp(
                            grid,
                            subset["rho"].to_numpy(dtype=float),
                            subset["accuracy"].to_numpy(dtype=float),
                        )
                    )
                array = np.vstack(interpolated)
                mean = array.mean(axis=0)
                sd = array.std(axis=0, ddof=1)
                for index, rho in enumerate(grid):
                    plot_rows.append(
                        {
                            "dataset": dataset,
                            "family": family,
                            "regime": regime,
                            "rho": float(rho),
                            "mean_accuracy": float(mean[index]),
                            "sd_accuracy": float(sd[index]),
                            "low_accuracy": float(mean[index] - sd[index]),
                            "high_accuracy": float(mean[index] + sd[index]),
                        }
                    )

    plot_data = pd.DataFrame(plot_rows)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    figure, axes = plt.subplots(
        2, 4, figsize=(15.5, 7.4), sharex=True, sharey=True
    )

    for row_index, dataset in enumerate(DATASETS):
        for column_index, family in enumerate(FAMILIES):
            axis = axes[row_index, column_index]
            for regime, linestyle, label in (
                ("noise", "-", "Random noise"),
                ("adversarial", "--", "Adversarial PGD"),
            ):
                subset = plot_data.query(
                    "dataset == @dataset and family == @family "
                    "and regime == @regime"
                ).sort_values("rho")
                line, = axis.plot(
                    subset["rho"],
                    subset["mean_accuracy"],
                    linewidth=2.0,
                    linestyle=linestyle,
                    label=label,
                    zorder=3,
                )
                axis.fill_between(
                    subset["rho"],
                    subset["low_accuracy"],
                    subset["high_accuracy"],
                    color=line.get_color(),
                    alpha=0.18,
                    zorder=1,
                )
            axis.axvline(
                PRIMARY_ENDPOINT,
                color="0.3",
                linewidth=1.2,
                linestyle=":",
                alpha=0.9,
                zorder=2,
            )
            axis.set_title(FAMILY_LABELS[family])
            axis.set_xlim(0.0, FIGURE_DISPLAY_MAX)
            axis.set_ylim(0.0, 1.03)
            axis.grid(True, alpha=0.25, linewidth=0.6)
            if column_index == 0:
                axis.set_ylabel(
                    DATASET_LABELS[dataset] + "\nNormalized accuracy"
                )

    handles = [
        Line2D([0], [0], color="C0", linewidth=2.0, linestyle="-", label="Random noise"),
        Line2D([0], [0], color="C1", linewidth=2.0, linestyle="--", label="Adversarial PGD"),
        Line2D(
            [0], [0], color="0.3", linewidth=1.2, linestyle=":",
            label=r"Primary endpoint $\rho_{\max}=0.09$",
        ),
        Line2D(
            [0], [0], color="0.5", linewidth=6.0, alpha=0.18,
            label=r"Mean $\pm$ 1 sample SD, $n=6$",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.supxlabel(
        "Achieved relative attention-logit displacement",
        fontsize=13,
        y=0.015,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.105,
        top=0.895,
        wspace=0.08,
        hspace=0.18,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        figure_dir / "fig_primary_robustness_curves.pdf",
        bbox_inches="tight",
    )
    figure.savefig(
        figure_dir / "fig_primary_robustness_curves_v18.png",
        dpi=300,
        bbox_inches="tight",
    )
    figure.savefig(
        figure_dir / "fig_primary_robustness_curves_v18.svg",
        bbox_inches="tight",
    )
    plt.close(figure)
    plot_data.to_csv(
        output_dir / "figure_interpolated_mean_sd_v18.csv", index=False
    )
    return plot_data


def write_report(
    output_dir: Path,
    manifest: pd.DataFrame,
    noise_raw: pd.DataFrame,
    noise_points: pd.DataFrame,
    attack_points: pd.DataFrame,
    aggregate: pd.DataFrame,
    wide: pd.DataFrame,
    statistics: dict,
    figure_dir: Path,
):
    primary = aggregate[aggregate["endpoint"].eq(PRIMARY_ENDPOINT)]
    primary_alibi = primary.query(
        "dataset == 'cifar' and pe_family == 'alibi'"
    ).iloc[0]
    wide_alibi = wide.query(
        "pe_family == 'alibi' and regime == 'noise'"
    ).iloc[0]

    # Canonical value assertions prevent silent convention drift.
    assert np.isclose(primary_alibi.noise_nauc_mean, 0.993526, atol=5e-7)
    assert np.isclose(wide_alibi.nauc_mean, 0.964138, atol=5e-7)

    report = {
        "version": "CANONICAL v18",
        "source_files": int(len(manifest)),
        "source_manifest_sha256": sha256(output_dir / "source_manifest_v18.csv"),
        "noise_raw_records": int(len(noise_raw)),
        "noise_budget_paired_points": int(len(noise_points)),
        "noise_draws_per_point": sorted(
            map(int, noise_points["n_draws"].unique())
        ),
        "attack_points": int(len(attack_points)),
        "curve_keys_primary": 96,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "sensitivity_endpoint": SENSITIVITY_ENDPOINT,
        "cifar_wide_endpoint": WIDE_ENDPOINT,
        "canonical_checks": {
            "cifar_alibi_noise_primary_mean": float(
                primary_alibi.noise_nauc_mean
            ),
            "cifar_alibi_noise_wide_mean": float(wide_alibi.nauc_mean),
            "expected_primary_rounded": "0.9935",
            "expected_wide_rounded": "0.9641",
        },
        "figure_sha256": {
            "pdf": sha256(figure_dir / "fig_primary_robustness_curves.pdf"),
            "png": sha256(figure_dir / "fig_primary_robustness_curves_v18.png"),
            "svg": sha256(figure_dir / "fig_primary_robustness_curves_v18.svg"),
        },
        "friedman": statistics["friedman"],
    }
    (output_dir / "CANONICAL_PIPELINE_REPORT_v18.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    markdown = f"""# Canonical v18 analysis-pipeline closure

## Locked curve convention

- Statistical unit: trained seed/checkpoint (`n=6` per family).
- Random noise: for each seed and native budget, achieved `rho_rel` and
  normalized accuracy are averaged over the ten fixed Gaussian directions.
  The paired mean points define one noise curve per seed.
- The ten draw-specific curves are **not** integrated separately and no
  union grid across draw-specific achieved-rho coordinates is used.
- Adversarial: the highest-loss one of five restarts is retained at each
  native budget, points are ordered by achieved `rho_rel`, and the cumulative
  lower accuracy envelope is applied.
- Both regimes receive the exact clean anchor `(0, 1)`, use piecewise-linear
  interpolation, remain unclipped, and are integrated with `numpy.trapezoid`.

## Single-source outputs

The same seed-level curves generated by `canonical_analysis_v18.py` produce:

1. the primary Table 2 over `[0, 0.09]`;
2. the CIFAR-100 wider-range Table 3 over `[0, 0.23]`; and
3. the 4x2 primary figure displayed over `[0, 0.095]`.

The source manifest contains **{len(manifest)} JSON inputs** with SHA-256
hashes. The pipeline collected **{len(noise_raw)} raw noise measurements**,
collapsed them into **{len(noise_points)} budget-paired points** with exactly
10 unique draws each, and used **{len(attack_points)} unique adversarial
native-budget points**.

## Convention sentinel values

- CIFAR-100 ALiBi noise nAUC `[0, 0.09]`:
  `{primary_alibi.noise_nauc_mean:.9f}` -> `0.9935`.
- CIFAR-100 ALiBi noise nAUC `[0, 0.23]`:
  `{wide_alibi.nauc_mean:.9f}` -> `0.9641`.

These values are asserted in code so that a future switch to a per-draw or
union-grid convention fails rather than silently changing the manuscript.
"""
    (output_dir / "CANONICAL_PIPELINE_REPORT_v18.md").write_text(
        markdown, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    pipeline_dir = Path(__file__).resolve().parent
    input_dir = pipeline_dir / "inputs"
    output_dir = pipeline_dir / "outputs"
    figure_dir = package_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        noise_raw,
        noise_budget_paired,
        attack_raw,
        attack_points,
        manifest,
    ) = load_measurements(input_dir, package_dir)
    (
        curves,
        seed_level,
        aggregate,
        wide,
        statistics,
    ) = build_curves_and_statistics(noise_budget_paired, attack_points)

    manifest.to_csv(output_dir / "source_manifest_v18.csv", index=False)
    noise_raw.to_csv(output_dir / "noise_raw_draw_points_v18.csv", index=False)
    noise_budget_paired.to_csv(
        output_dir / "noise_budget_paired_points_v18.csv", index=False
    )
    attack_raw.to_csv(output_dir / "attack_raw_sources_v18.csv", index=False)
    attack_points.to_csv(
        output_dir / "adversarial_native_budget_points_v18.csv", index=False
    )
    curves.to_csv(output_dir / "canonical_curves_long_v18.csv", index=False)
    seed_level.to_csv(output_dir / "seed_level_nauc_v18.csv", index=False)
    aggregate.to_csv(output_dir / "primary_aggregate_nauc_v18.csv", index=False)
    wide.to_csv(output_dir / "cifar_wide_aggregate_nauc_v18.csv", index=False)
    (output_dir / "canonical_statistics_v18.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )

    write_primary_table(
        aggregate, output_dir / "table_primary_nauc_v18.tex"
    )
    write_wide_table(wide, output_dir / "table_cifar_wide_v18.tex")
    generate_figure(curves, figure_dir, output_dir)
    write_report(
        output_dir,
        manifest,
        noise_raw,
        noise_budget_paired,
        attack_points,
        aggregate,
        wide,
        statistics,
        figure_dir,
    )

    print((output_dir / "CANONICAL_PIPELINE_REPORT_v18.md").read_text())


if __name__ == "__main__":
    main()
