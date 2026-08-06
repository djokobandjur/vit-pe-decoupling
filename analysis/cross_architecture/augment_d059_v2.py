#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as student_t

SEEDS = [42, 123, 456, 789, 1011, 1213]
FAMILIES = ["learned", "sinusoidal", "rope"]
LABELS = {"learned": "Learned", "sinusoidal": "Sinusoidal", "rope": "RoPE"}


def safe_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(safe_json(obj), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def welch_interval(vitb: np.ndarray, vits: np.ndarray) -> dict[str, float]:
    vitb = np.asarray(vitb, float)
    vits = np.asarray(vits, float)
    n_b, n_s = len(vitb), len(vits)
    mean_delta = float(vits.mean() - vitb.mean())
    var_b = float(vitb.var(ddof=1))
    var_s = float(vits.var(ddof=1))
    se2 = var_b / n_b + var_s / n_s
    if se2 <= 0:
        return {
            "mean_delta_vits_minus_vitb": mean_delta,
            "welch_df": None,
            "welch_ci95_low": mean_delta,
            "welch_ci95_high": mean_delta,
        }
    df = se2**2 / ((var_b / n_b) ** 2 / (n_b - 1) + (var_s / n_s) ** 2 / (n_s - 1))
    q = float(student_t.ppf(0.975, df))
    se = math.sqrt(se2)
    return {
        "mean_delta_vits_minus_vitb": mean_delta,
        "welch_df": float(df),
        "welch_ci95_low": mean_delta - q * se,
        "welch_ci95_high": mean_delta + q * se,
    }


def dominant_sign(negative: int, positive: int) -> str:
    if negative > positive:
        return "negative"
    if positive > negative:
        return "positive"
    return "balanced"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed_level = pd.read_csv(args.v1_out / "D059_SEED_LEVEL_NAUC_v1.csv")
    contrasts = pd.read_csv(args.v1_out / "D059_SEED_LEVEL_ARCHITECTURE_CONTRASTS_v1.csv")
    family_contrasts = pd.read_csv(args.v1_out / "D059_FAMILY_ARCHITECTURE_CONTRASTS_v1.csv")
    did = pd.read_csv(args.v1_out / "D059_INTERACTION_PAIRWISE_DID_v1.csv")
    interaction = json.loads((args.v1_out / "D059_ARCHITECTURE_PE_INTERACTION_TEST_v1.json").read_text())
    summary_v1 = json.loads((args.v1_out / "D059_ANALYSIS_SUMMARY_v1.json").read_text())

    primary = seed_level.query("scope == 'cross' and support_multiplier == 1.0").copy()

    # Welch sensitivity intervals for architecture contrasts.
    welch_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for metric in ["attack_nauc", "noise_nauc", "noise_minus_attack_gap"]:
            vitb = primary.query("architecture == 'vitb_amp' and family == @family").sort_values("seed")[metric].to_numpy(float)
            vits = primary.query("architecture == 'vits_amp' and family == @family").sort_values("seed")[metric].to_numpy(float)
            paired = family_contrasts.query("family == @family and metric == @metric").iloc[0]
            row = {
                "family": family,
                "family_label": LABELS[family],
                "metric": metric,
                "vitb_mean": float(vitb.mean()),
                "vits_mean": float(vits.mean()),
                "paired_ci95_low": float(paired.ci95_t_low),
                "paired_ci95_high": float(paired.ci95_t_high),
                **welch_interval(vitb, vits),
            }
            row["paired_welch_same_direction"] = bool(
                (row["paired_ci95_high"] < 0 and row["welch_ci95_high"] < 0)
                or (row["paired_ci95_low"] > 0 and row["welch_ci95_low"] > 0)
                or (
                    row["paired_ci95_low"] <= 0 <= row["paired_ci95_high"]
                    and row["welch_ci95_low"] <= 0 <= row["welch_ci95_high"]
                )
            )
            welch_rows.append(row)
    welch_df = pd.DataFrame(welch_rows)

    # Per-seed ordinal reconstruction and rank sums.
    attack_delta = contrasts.query("metric == 'attack_nauc'").pivot(index="seed", columns="family", values="delta_vits_minus_vitb").reindex(index=SEEDS, columns=FAMILIES)
    ordinal_rows: list[dict[str, Any]] = []
    target_order = "learned < sinusoidal < rope"
    for seed, row in attack_delta.iterrows():
        values = row.to_numpy(float)
        ranks = rankdata(values, method="average")  # lower delta gets lower rank
        order = " < ".join(row.sort_values().index.tolist())
        ordinal_rows.append({
            "seed": int(seed),
            "learned_delta": float(row["learned"]),
            "sinusoidal_delta": float(row["sinusoidal"]),
            "rope_delta": float(row["rope"]),
            "learned_rank": float(ranks[FAMILIES.index("learned")]),
            "sinusoidal_rank": float(ranks[FAMILIES.index("sinusoidal")]),
            "rope_rank": float(ranks[FAMILIES.index("rope")]),
            "ascending_order": order,
            "matches_modal_order": order == target_order,
        })
    ordinal_df = pd.DataFrame(ordinal_rows)
    rank_sums = {family: float(ordinal_df[f"{family}_rank"].sum()) for family in FAMILIES}

    # Pairwise DiD sign consistency.
    did_v2 = did.copy()
    did_v2["dominant_sign"] = [dominant_sign(int(r.negative_deltas), int(r.positive_deltas)) for _, r in did_v2.iterrows()]
    did_v2["same_sign_seeds"] = did_v2[["negative_deltas", "positive_deltas"]].max(axis=1).astype(int)
    did_v2["same_sign_fraction"] = did_v2["same_sign_seeds"] / did_v2["n"]

    # Heteroskedasticity/dispersion audit.
    variance_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        arch_stats = {}
        for architecture in ["vitb_amp", "vits_amp"]:
            vals = primary.query("architecture == @architecture and family == @family").sort_values("seed")["attack_nauc"].to_numpy(float)
            arch_stats[architecture] = {
                "values_sorted": ";".join(f"{x:.9f}" for x in sorted(vals)),
                "minimum": float(vals.min()),
                "maximum": float(vals.max()),
                "range": float(vals.max() - vals.min()),
                "sd": float(vals.std(ddof=1)),
            }
        vitb_vals = primary.query("architecture == 'vitb_amp' and family == @family").set_index("seed")["attack_nauc"]
        vits_vals = primary.query("architecture == 'vits_amp' and family == @family").set_index("seed")["attack_nauc"]
        loo_ratios = []
        for seed in SEEDS:
            loo_ratios.append(float(vits_vals.drop(seed).std(ddof=1) / vitb_vals.drop(seed).std(ddof=1)))
        variance_rows.append({
            "family": family,
            "family_label": LABELS[family],
            "vitb_values_sorted": arch_stats["vitb_amp"]["values_sorted"],
            "vits_values_sorted": arch_stats["vits_amp"]["values_sorted"],
            "vitb_range": arch_stats["vitb_amp"]["range"],
            "vits_range": arch_stats["vits_amp"]["range"],
            "vitb_sd": arch_stats["vitb_amp"]["sd"],
            "vits_sd": arch_stats["vits_amp"]["sd"],
            "sd_ratio_vits_over_vitb": arch_stats["vits_amp"]["sd"] / arch_stats["vitb_amp"]["sd"],
            "loo_sd_ratio_min": min(loo_ratios),
            "loo_sd_ratio_max": max(loo_ratios),
            "single_seed_explanation_supported": False if min(loo_ratios) > 2 else None,
        })
    variance_df = pd.DataFrame(variance_rows)

    # Attack/noise sign balance.
    sign_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for metric in ["attack_nauc", "noise_nauc"]:
            vals = contrasts.query("family == @family and metric == @metric")["delta_vits_minus_vitb"].to_numpy(float)
            neg = int((vals < 0).sum())
            pos = int((vals > 0).sum())
            sign_rows.append({
                "family": family,
                "family_label": LABELS[family],
                "metric": metric,
                "negative_seeds": neg,
                "positive_seeds": pos,
                "zero_seeds": int((vals == 0).sum()),
                "dominant_sign": dominant_sign(neg, pos),
                "mean_delta_vits_minus_vitb": float(vals.mean()),
            })
    sign_df = pd.DataFrame(sign_rows)

    # Values above 1 are allowed for normalized accuracy/nAUC when perturbed accuracy marginally exceeds clean accuracy.
    above_one = primary.query("noise_nauc > 1.0")[["architecture", "family", "seed", "noise_nauc"]].copy()
    above_one["interpretation"] = "Allowed: perturbation-averaged normalized accuracy can marginally exceed 1 when noisy accuracy exceeds the clean reference on the fixed evaluation split."

    # Main-table rows with exact sign-flip quantisation metadata.
    attack_main = family_contrasts.query("metric == 'attack_nauc'").copy()
    attack_main["attainable_p_step"] = 2 / 64
    attack_main["attainable_p_floor"] = 2 / 64
    attack_main["p_rung_index"] = np.rint(attack_main["exact_sign_flip_p_two_sided"] / (2 / 64)).astype(int)

    # Summary v2.
    summary_v2 = {
        **summary_v1,
        "revision": "D059_REANALYSIS_v2",
        "sign_flip_quantisation": {
            "n": 6,
            "two_sided_assignments": 64,
            "step": 2 / 64,
            "floor": 2 / 64,
            "per_family_role": "secondary directional-consistency evidence, not primary inference",
        },
        "ordinal_interaction": {
            "modal_order": target_order,
            "modal_order_seed_count": int(ordinal_df.matches_modal_order.sum()),
            "exception_seeds": [int(x) for x in ordinal_df.loc[~ordinal_df.matches_modal_order, "seed"]],
            "rank_sums": rank_sums,
            "friedman_Q": interaction["Q"],
            "friedman_exact_p": interaction["exact_p"],
        },
        "welch_sensitivity": {
            "all_attack_intervals_same_direction_as_seed_aligned_intervals": bool(
                welch_df.query("metric == 'attack_nauc'")["paired_welch_same_direction"].all()
            ),
            "scope": "supplementary sensitivity because seed labels are aligned but not strict training pairs",
        },
        "dispersion_effect": {
            "sd_ratio_order": "Learned >> RoPE > Sinusoidal",
            "mean_effect_order": "Learned > Sinusoidal > RoPE",
            "interpretation": "Backbone reduction changes both mean adversarial robustness and seed dispersion in PE-dependent but differently ordered ways.",
        },
        "noise_sign_balance": {
            r.family: {
                "negative": int(r.negative_seeds),
                "positive": int(r.positive_seeds),
            }
            for _, r in sign_df.query("metric == 'noise_nauc'").iterrows()
        },
        "normalized_nauc_above_one": above_one.to_dict(orient="records"),
    }

    attack_main.to_csv(args.out / "D059_ATTACK_MAIN_TABLE_v2.csv", index=False, float_format="%.17g")
    welch_df.to_csv(args.out / "D059_WELCH_SENSITIVITY_v2.csv", index=False, float_format="%.17g")
    ordinal_df.to_csv(args.out / "D059_INTERACTION_ORDINAL_ROBUSTNESS_v2.csv", index=False, float_format="%.17g")
    did_v2.to_csv(args.out / "D059_INTERACTION_PAIRWISE_DID_v2.csv", index=False, float_format="%.17g")
    variance_df.to_csv(args.out / "D059_ARCHITECTURE_DISPERSION_AUDIT_v2.csv", index=False, float_format="%.17g")
    sign_df.to_csv(args.out / "D059_ATTACK_NOISE_SIGN_BALANCE_v2.csv", index=False, float_format="%.17g")
    above_one.to_csv(args.out / "D059_NORMALIZED_NAUC_ABOVE_ONE_AUDIT_v2.csv", index=False, float_format="%.17g")
    dump_json(args.out / "D059_ANALYSIS_SUMMARY_v2.json", summary_v2)

    print(json.dumps(safe_json(summary_v2), indent=2))


if __name__ == "__main__":
    main()
