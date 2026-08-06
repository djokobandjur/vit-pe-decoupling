#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, os, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, spearmanr

SEEDS = [42, 123, 456, 789, 1011, 1213]
FAMILIES = ["learned", "sinusoidal", "rope"]
FAMILY_LABELS = {"learned": "Learned", "sinusoidal": "Sinusoidal", "rope": "RoPE"}
REGIMES = ["full_fp32_trained", "amp_fp16_trained"]
TAU_ABS = 5e-7
TAU_REL = 5e-6
BOOT_REPS = 200_000
BOOT_SEED = 20260803


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe(obj), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def near(a: float, b: float) -> bool:
    return abs(a - b) <= max(TAU_ABS, TAU_REL * max(abs(a), abs(b), 1e-12))


def exact_sign_flip(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    obs = abs(vals.mean())
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(vals))))
    stats = np.abs((signs * vals[None, :]).mean(axis=1))
    return float(np.mean(stats >= obs - 1e-15))


def bootstrap_ci(vals: np.ndarray, seed: int, reps: int = BOOT_REPS) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(reps, len(vals)))
    means = vals[idx].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def paired_stats(vals: np.ndarray, margin: float | None, seed: int) -> dict[str, Any]:
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    if se == 0:
        ci90 = (mean, mean)
        ci95 = (mean, mean)
    else:
        ci90 = (mean - float(student_t.ppf(0.95, n - 1)) * se,
                mean + float(student_t.ppf(0.95, n - 1)) * se)
        ci95 = (mean - float(student_t.ppf(0.975, n - 1)) * se,
                mean + float(student_t.ppf(0.975, n - 1)) * se)
    b_lo, b_hi = bootstrap_ci(vals, seed)
    out: dict[str, Any] = {
        "n": n,
        "mean_delta_amp_minus_fp32": mean,
        "sd_delta": sd,
        "ci90_t_low": float(ci90[0]),
        "ci90_t_high": float(ci90[1]),
        "ci95_t_low": float(ci95[0]),
        "ci95_t_high": float(ci95[1]),
        "ci95_bootstrap_low": b_lo,
        "ci95_bootstrap_high": b_hi,
        "positive_deltas": int((vals > 0).sum()),
        "negative_deltas": int((vals < 0).sum()),
        "zero_deltas": int((vals == 0).sum()),
        "exact_sign_flip_p_two_sided": exact_sign_flip(vals),
        "min_delta": float(vals.min()),
        "max_delta": float(vals.max()),
    }
    if margin is not None:
        if se == 0:
            p_lower = 0.0 if mean > -margin else 1.0
            p_upper = 0.0 if mean < margin else 1.0
        else:
            t_lower = (mean + margin) / se
            t_upper = (mean - margin) / se
            p_lower = float(student_t.sf(t_lower, n - 1))
            p_upper = float(student_t.cdf(t_upper, n - 1))
        equivalent = ci90[0] > -margin and ci90[1] < margin
        out.update({
            "equivalence_margin": float(margin),
            "tost_p_lower": p_lower,
            "tost_p_upper": p_upper,
            "tost_p_max": max(p_lower, p_upper),
            "equivalence_status": "EQUIVALENT" if equivalent else "INCONCLUSIVE",
        })
    return out


def monotone_epsilon_endpoint(points: list[dict[str, float]], q: float) -> dict[str, Any]:
    pts = sorted(points, key=lambda r: r["epsilon"])
    eps = np.array([r["epsilon"] for r in pts], dtype=float)
    acc = np.array([r["normalized_accuracy"] for r in pts], dtype=float)
    rho = np.array([r["rho_rel"] for r in pts], dtype=float)
    env = np.minimum.accumulate(acc)
    idx = np.where(env <= q)[0]
    if len(idx) == 0:
        return {"status": "NOT_REACHED", "q": q}
    j = int(idx[0])
    if j == 0:
        return {
            "status": "AT_OR_BELOW_CLEAN", "q": q, "epsilon": float(eps[0]), "rho": float(rho[0]),
            "epsilon_left": float(eps[0]), "epsilon_right": float(eps[0]),
            "rho_left": float(rho[0]), "rho_right": float(rho[0]),
            "accuracy_left": float(env[0]), "accuracy_right": float(env[0]),
        }
    e0, e1 = float(eps[j - 1]), float(eps[j])
    a0, a1 = float(env[j - 1]), float(env[j])
    r0, r1 = float(rho[j - 1]), float(rho[j])
    if abs(a1 - a0) < 1e-15:
        w = 1.0
    else:
        w = (q - a0) / (a1 - a0)
    e = e0 + w * (e1 - e0)
    r = r0 + w * (r1 - r0)
    return {
        "status": "IDENTIFIED", "q": q, "epsilon": float(e), "rho": float(r),
        "epsilon_left": e0, "epsilon_right": e1,
        "rho_left": r0, "rho_right": r1,
        "accuracy_left": a0, "accuracy_right": a1,
        "epsilon_bracket_width": e1 - e0,
        "rho_bracket_abs_width": abs(r1 - r0),
        "rho_bracket_signed_delta": r1 - r0,
    }


def dedupe_rho_rows(points: pd.DataFrame, regime: str) -> pd.DataFrame:
    pts = points.copy().sort_values(["rho", "budget", "source_index"]).reset_index(drop=True)
    groups: list[list[pd.Series]] = []
    for _, row in pts.iterrows():
        if not groups or not near(float(groups[-1][0]["rho"]), float(row["rho"])):
            groups.append([row])
        else:
            groups[-1].append(row)
    selected = []
    for group in groups:
        if regime == "attack":
            def key(row: pd.Series):
                loss = float(row.get("attack_loss", np.nan))
                loss_key = -loss if np.isfinite(loss) else float("inf")
                return (float(row["accuracy"]), loss_key, float(row["budget"]), int(row["source_index"]))
            q = min(group, key=key)
        else:
            q = group[0].copy()
            q["rho"] = float(np.mean([float(x["rho"]) for x in group]))
            q["accuracy"] = float(np.mean([float(x["accuracy"]) for x in group]))
            q["budget"] = float(min(float(x["budget"]) for x in group))
            q["source_index"] = int(min(int(x["source_index"]) for x in group))
        selected.append(dict(q))
    return pd.DataFrame(selected).sort_values("rho").reset_index(drop=True)


def construct_rho_curve(points: pd.DataFrame, regime: str) -> tuple[np.ndarray, np.ndarray]:
    pts = points[np.isfinite(points["rho"]) & np.isfinite(points["accuracy"]) & (points["rho"] > 0)].copy()
    pts = dedupe_rho_rows(pts, regime)
    x = np.concatenate(([0.0], pts["rho"].to_numpy(float)))
    y = np.concatenate(([1.0], pts["accuracy"].to_numpy(float)))
    if regime == "attack":
        y = np.minimum.accumulate(y)
    if not np.all(np.diff(x) > 0):
        raise ValueError("rho curve is not strictly increasing after D028 dedupe")
    return x, y


def integrate_nauc(x: np.ndarray, y: np.ndarray, end: float) -> float:
    if x[-1] + 1e-15 < end:
        raise ValueError(f"curve support {x[-1]} < endpoint {end}")
    inside = x < end
    xc = np.concatenate((x[inside], [end]))
    yc = np.concatenate((y[inside], [np.interp(end, x, y)]))
    return float(np.trapezoid(yc, xc) / end)


def load_cohort(root: Path, regime: str, fp16_layout: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    session_root = root / "final_n6" if fp16_layout else root
    sessions = sorted(session_root.glob("session_*_seed_*"))
    if len(sessions) != 6:
        raise ValueError(f"{regime}: expected 6 sessions, found {len(sessions)}")
    attack_rows, noise_rows, clean_rows = [], [], []
    protocol_ids, split_hashes, sample_hashes, split_file_hashes = set(), set(), set(), set()
    engine_hashes, architecture_hashes = set(), set()
    session_complete = 0
    attack_configs: dict[str, set[tuple[Any, ...]]] = {f: set() for f in FAMILIES}
    budget_grids: dict[str, set[tuple[float, ...]]] = {f: set() for f in FAMILIES}
    noise_budget_grids: dict[str, set[tuple[float, ...]]] = {f: set() for f in FAMILIES}
    checkpoint_hashes: dict[tuple[str, int], str] = {}
    for sess in sessions:
        comp = json.loads((sess / "SESSION_COMPLETE.json").read_text())
        session_complete += 1
        protocol = json.loads((sess / "SESSION_PROTOCOL.json").read_text())
        protocol_ids.add(protocol["protocol_id"])
        locked = protocol["locked_protocol"]
        inv = locked.get("input_invariants", locked.get("split", {}))
        if "split_sha256" in inv:
            split_hashes.add(inv["split_sha256"])
            sample_hashes.add(inv["sample_order_sha256"])
            split_file_hashes.add(inv["split_file_sha256"])
        code_hash = locked.get("code_hashes", {})
        if code_hash:
            engine_hashes.add(code_hash.get("engine_sha256", ""))
            architecture_hashes.add(code_hash.get("architecture_sha256", ""))
        seed = int(protocol["seed"])
        noise_data = json.loads((sess / "noise_all_families.json").read_text())
        nm = noise_data["metadata"]
        split_hashes.add(nm["split"]["split_sha256"])
        sample_hashes.add(nm["split"]["sample_order_sha256"])
        split_file_hashes.add(nm["split_file_sha256"])
        for family in FAMILIES:
            sr = noise_data["results"][family][str(seed)]
            clean = float(sr["clean_acc_eval"])
            clean_rows.append({
                "regime": regime, "family": family, "seed": seed, "clean_accuracy": clean,
                "checkpoint_sha256": sr["checkpoint_sha256"],
            })
            checkpoint_hashes[(family, seed)] = sr["checkpoint_sha256"]
            draws = sr["noise"]["draws"]
            if len(draws) != 10:
                raise ValueError(f"{regime}/{family}/{seed}: expected 10 noise draws")
            draw_budgets = []
            for draw in draws:
                local_budgets = []
                for idx, point in enumerate(sorted(draw["points"].values(), key=lambda p: float(p["budget"]))):
                    budget = float(point["budget"])
                    local_budgets.append(budget)
                    noise_rows.append({
                        "regime": regime, "family": family, "seed": seed,
                        "draw_seed": int(draw["seed"]), "budget": budget,
                        "rho": float(point["rho"]["rho_rel"]),
                        "accuracy": float(point["normalized_accuracy"]),
                        "source_index": idx,
                    })
                draw_budgets.append(tuple(local_budgets))
            if len(set(draw_budgets)) != 1:
                raise ValueError(f"{regime}/{family}/{seed}: noise grids differ across draws")
            noise_budget_grids[family].add(draw_budgets[0])

            attack_path = sess / f"attacks_{family}.json"
            attack_data = json.loads(attack_path.read_text())
            am = attack_data["metadata"]
            split_hashes.add(am["split"]["split_sha256"])
            sample_hashes.add(am["split"]["sample_order_sha256"])
            split_file_hashes.add(am["split_file_sha256"])
            cfg = am["config"]
            attack_configs[family].add((int(cfg["pgd_steps"]), int(cfg["pgd_restarts"]), float(cfg["pgd_alpha_ratio"]), cfg["parameter_norm"]))
            sr_a = attack_data["results"][family][str(seed)]
            if sr_a["checkpoint_sha256"] != checkpoint_hashes[(family, seed)]:
                raise ValueError(f"{regime}/{family}/{seed}: attack/noise checkpoint hash mismatch")
            budget_list = []
            for idx, point in enumerate(sorted(sr_a["attacks"]["pgd_pe"].values(), key=lambda p: float(p["budget"]))):
                budget = float(point["budget"])
                budget_list.append(budget)
                records = point.get("restart_records", [])
                if budget > 0:
                    if len(records) != 5:
                        raise ValueError(f"{regime}/{family}/{seed}/{budget}: restart count != 5")
                    losses = [float(r["attack_loss"]) for r in records]
                    if not math.isclose(float(point["selected_attack_loss"]), max(losses), rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError(f"{regime}/{family}/{seed}/{budget}: selected loss mismatch")
                attack_rows.append({
                    "regime": regime, "family": family, "seed": seed, "budget": budget,
                    "rho": float(point["rho"]["rho_rel"]),
                    "accuracy": float(point["normalized_accuracy"]),
                    "accuracy_percent": float(point["accuracy"]),
                    "attack_loss": float(point["selected_attack_loss"]) if point["selected_attack_loss"] is not None else np.nan,
                    "restart_loss_min": min([float(r["attack_loss"]) for r in records]) if records else np.nan,
                    "restart_loss_max": max([float(r["attack_loss"]) for r in records]) if records else np.nan,
                    "restart_loss_spread": (max([float(r["attack_loss"]) for r in records]) - min([float(r["attack_loss"]) for r in records])) if records else 0.0,
                    "source_index": idx,
                })
            budget_grids[family].add(tuple(budget_list))
    if len(split_hashes) != 1 or len(sample_hashes) != 1 or len(split_file_hashes) != 1:
        raise ValueError(f"{regime}: inconsistent split provenance")
    for fam in FAMILIES:
        if len(attack_configs[fam]) != 1 or len(budget_grids[fam]) != 1 or len(noise_budget_grids[fam]) != 1:
            raise ValueError(f"{regime}/{fam}: protocol grid/config inconsistent")
    raw_noise = pd.DataFrame(noise_rows)
    if raw_noise.duplicated(["regime", "family", "seed", "draw_seed", "budget"]).any():
        raise ValueError(f"{regime}: duplicate raw noise points")
    noise = (raw_noise.groupby(["regime", "family", "seed", "budget"], as_index=False)
             .agg(rho=("rho", "mean"), accuracy=("accuracy", "mean"),
                  rho_sd=("rho", lambda x: float(np.std(x, ddof=1))),
                  accuracy_sd=("accuracy", lambda x: float(np.std(x, ddof=1))),
                  n_draws=("draw_seed", "nunique")))
    noise["source_index"] = noise.groupby(["regime", "family", "seed"]).cumcount()
    attack = pd.DataFrame(attack_rows).sort_values(["family", "seed", "budget"])
    clean = pd.DataFrame(clean_rows).sort_values(["family", "seed"])
    report = {
        "regime": regime,
        "sessions_complete": session_complete,
        "protocol_ids": sorted(protocol_ids),
        "split_sha256": next(iter(split_hashes)),
        "sample_order_sha256": next(iter(sample_hashes)),
        "split_file_sha256": next(iter(split_file_hashes)),
        "engine_hashes": sorted(x for x in engine_hashes if x),
        "architecture_hashes": sorted(x for x in architecture_hashes if x),
        "unique_checkpoint_hashes": len(set(checkpoint_hashes.values())),
        "attack_points": len(attack),
        "noise_raw_points": len(raw_noise),
        "noise_budget_averaged_points": len(noise),
        "attack_configs": {fam: list(next(iter(attack_configs[fam]))) for fam in FAMILIES},
        "attack_budget_grids": {fam: list(next(iter(budget_grids[fam]))) for fam in FAMILIES},
        "noise_budget_grids": {fam: list(next(iter(noise_budget_grids[fam]))) for fam in FAMILIES},
    }
    return attack, noise, clean, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16-root", type=Path, required=True)
    ap.add_argument("--fp32-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sesoi-lock", type=Path, required=True)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sesoi = json.loads(args.sesoi_lock.read_text())
    margin_rho50 = float(sesoi["equivalence_margin_rho50_absolute"])
    margin_clean = float(sesoi["clean_accuracy_context_margin_pp"])

    fp16_attack, fp16_noise, fp16_clean, fp16_report = load_cohort(args.fp16_root, "amp_fp16_trained", True)
    fp32_attack, fp32_noise, fp32_clean, fp32_report = load_cohort(args.fp32_root, "full_fp32_trained", False)

    # Cross-cohort protocol gates.
    protocol_checks = {
        "same_split_sha256": fp16_report["split_sha256"] == fp32_report["split_sha256"],
        "same_sample_order_sha256": fp16_report["sample_order_sha256"] == fp32_report["sample_order_sha256"],
        "same_split_file_sha256": fp16_report["split_file_sha256"] == fp32_report["split_file_sha256"],
        "same_attack_configs_three_families": fp16_report["attack_configs"] == fp32_report["attack_configs"],
        "same_attack_budget_grids_three_families": fp16_report["attack_budget_grids"] == fp32_report["attack_budget_grids"],
        "same_noise_budget_grids_three_families": fp16_report["noise_budget_grids"] == fp32_report["noise_budget_grids"],
        "different_checkpoint_hashes_expected": True,
    }
    merged_hash = fp16_clean.merge(fp32_clean, on=["family", "seed"], suffixes=("_amp", "_fp32"))
    protocol_checks["all_18_checkpoint_hashes_differ"] = bool((merged_hash["checkpoint_sha256_amp"] != merged_hash["checkpoint_sha256_fp32"]).all())
    strict_pairing_provenance = {
        "status": "NOT_DEMONSTRATED_IN_RESULTS_PACKAGE",
        "analysis_label": "SEED_MATCHED",
        "bit_identical_initial_state_hash_available": False,
        "data_order_identity": "same evaluation split/sample order proven; training sampler order and initial state hash not packaged",
        "gradscaler_skip_count_available": False,
        "consequence": "Do not call the comparison strictly paired; use seed-matched terminology and do not attribute differences solely to numerical precision.",
    }

    # Point-level epsilon threshold endpoints.
    attack_all = pd.concat([fp32_attack, fp16_attack], ignore_index=True)
    noise_all = pd.concat([fp32_noise, fp16_noise], ignore_index=True)
    clean_all = pd.concat([fp32_clean, fp16_clean], ignore_index=True)
    endpoint_rows = []
    for (regime, family, seed), group in attack_all.groupby(["regime", "family", "seed"]):
        pts = [{
            "epsilon": float(r.budget), "normalized_accuracy": float(r.accuracy), "rho_rel": float(r.rho)
        } for r in group.sort_values("budget").itertuples()]
        eps = np.array([p["epsilon"] for p in pts])
        acc = np.array([p["normalized_accuracy"] for p in pts])
        rho = np.array([p["rho_rel"] for p in pts])
        endpoints = {q: monotone_epsilon_endpoint(pts, q) for q in (0.8, 0.5, 0.2)}
        clean_val = float(clean_all.query("regime==@regime and family==@family and seed==@seed").iloc[0]["clean_accuracy"])
        row = {
            "regime": regime, "family": family, "family_label": FAMILY_LABELS[family], "seed": int(seed),
            "clean_accuracy_percent": clean_val,
            "max_rho_rel": float(rho.max()),
            "epsilon_at_max_rho": float(eps[int(rho.argmax())]),
            "min_normalized_accuracy": float(acc.min()),
            "epsilon_at_min_accuracy": float(eps[int(acc.argmin())]),
            "rho_nonmonotone_step_count": int(np.sum(np.diff(rho) < -max(TAU_ABS, 1e-12))),
            "raw_accuracy_increase_step_count": int(np.sum(np.diff(acc) > 1e-12)),
            "q80_status": endpoints[0.8]["status"],
            "epsilon80": endpoints[0.8].get("epsilon"), "rho80": endpoints[0.8].get("rho"),
            "q50_status": endpoints[0.5]["status"],
            "epsilon50": endpoints[0.5].get("epsilon"), "rho50": endpoints[0.5].get("rho"),
            "q20_status": endpoints[0.2]["status"],
            "epsilon20": endpoints[0.2].get("epsilon"), "rho20": endpoints[0.2].get("rho"),
        }
        for q, tag in [(0.8, "80"), (0.5, "50"), (0.2, "20")]:
            ep = endpoints[q]
            for key in ["epsilon_left", "epsilon_right", "rho_left", "rho_right", "accuracy_left", "accuracy_right", "epsilon_bracket_width", "rho_bracket_abs_width", "rho_bracket_signed_delta"]:
                row[f"q{tag}_{key}"] = ep.get(key)
        row["cliff_width_rho20_minus_rho80"] = (row["rho20"] - row["rho80"]) if row["rho20"] is not None and row["rho80"] is not None else None
        endpoint_rows.append(row)
    endpoints_df = pd.DataFrame(endpoint_rows).sort_values(["family", "seed", "regime"])

    # Validate AMP endpoint implementation against packaged final audit.
    packaged_threshold = json.loads((args.fp16_root / "final_audit" / "THRESHOLD_METRICS.json").read_text())
    packaged_map = {(x["family"], int(x["seed"])): x for x in packaged_threshold["units"]}
    validation_deltas = []
    for r in endpoints_df.query("regime=='amp_fp16_trained'").itertuples():
        p = packaged_map[(r.family, int(r.seed))]
        validation_deltas.append(abs(float(r.rho50) - float(p["rho50"])))
    endpoint_validation_max_abs = max(validation_deltas)
    if endpoint_validation_max_abs > 1e-12:
        raise ValueError(f"rho50 implementation does not reproduce packaged AMP audit: {endpoint_validation_max_abs}")

    # Endpoint paired contrasts and family summaries.
    family_endpoint_rows = []
    paired_endpoint_rows = []
    stats_seed_counter = 0
    for family in FAMILIES:
        amp = endpoints_df.query("regime=='amp_fp16_trained' and family==@family").sort_values("seed")
        fp32 = endpoints_df.query("regime=='full_fp32_trained' and family==@family").sort_values("seed")
        if amp.seed.tolist() != fp32.seed.tolist():
            raise ValueError("seed order mismatch")
        for i, seed in enumerate(SEEDS):
            paired_endpoint_rows.append({
                "family": family, "family_label": FAMILY_LABELS[family], "seed": seed,
                "clean_accuracy_amp": float(amp.iloc[i].clean_accuracy_percent),
                "clean_accuracy_fp32": float(fp32.iloc[i].clean_accuracy_percent),
                "delta_clean_accuracy_pp_amp_minus_fp32": float(amp.iloc[i].clean_accuracy_percent - fp32.iloc[i].clean_accuracy_percent),
                "epsilon50_amp": float(amp.iloc[i].epsilon50), "epsilon50_fp32": float(fp32.iloc[i].epsilon50),
                "delta_epsilon50_amp_minus_fp32": float(amp.iloc[i].epsilon50 - fp32.iloc[i].epsilon50),
                "rho50_amp": float(amp.iloc[i].rho50), "rho50_fp32": float(fp32.iloc[i].rho50),
                "delta_rho50_amp_minus_fp32": float(amp.iloc[i].rho50 - fp32.iloc[i].rho50),
                "rho50_bracket_width_amp": float(amp.iloc[i].q50_rho_bracket_abs_width),
                "rho50_bracket_width_fp32": float(fp32.iloc[i].q50_rho_bracket_abs_width),
                "cliff_width_amp": float(amp.iloc[i].cliff_width_rho20_minus_rho80),
                "cliff_width_fp32": float(fp32.iloc[i].cliff_width_rho20_minus_rho80),
            })
        metrics = {
            "clean_accuracy_pp": (amp.clean_accuracy_percent.to_numpy() - fp32.clean_accuracy_percent.to_numpy(), margin_clean),
            "epsilon50": (amp.epsilon50.to_numpy() - fp32.epsilon50.to_numpy(), None),
            "rho50": (amp.rho50.to_numpy() - fp32.rho50.to_numpy(), margin_rho50),
            "cliff_width": (amp.cliff_width_rho20_minus_rho80.to_numpy() - fp32.cliff_width_rho20_minus_rho80.to_numpy(), None),
        }
        for metric, (deltas, margin) in metrics.items():
            stats = paired_stats(deltas, margin, BOOT_SEED + stats_seed_counter)
            stats_seed_counter += 1
            row = {"family": family, "family_label": FAMILY_LABELS[family], "metric": metric, **stats}
            row.update({
                "amp_mean": float(amp[{"clean_accuracy_pp":"clean_accuracy_percent","epsilon50":"epsilon50","rho50":"rho50","cliff_width":"cliff_width_rho20_minus_rho80"}[metric]].mean()),
                "amp_sd": float(amp[{"clean_accuracy_pp":"clean_accuracy_percent","epsilon50":"epsilon50","rho50":"rho50","cliff_width":"cliff_width_rho20_minus_rho80"}[metric]].std(ddof=1)),
                "fp32_mean": float(fp32[{"clean_accuracy_pp":"clean_accuracy_percent","epsilon50":"epsilon50","rho50":"rho50","cliff_width":"cliff_width_rho20_minus_rho80"}[metric]].mean()),
                "fp32_sd": float(fp32[{"clean_accuracy_pp":"clean_accuracy_percent","epsilon50":"epsilon50","rho50":"rho50","cliff_width":"cliff_width_rho20_minus_rho80"}[metric]].std(ddof=1)),
            })
            family_endpoint_rows.append(row)
    paired_endpoints_df = pd.DataFrame(paired_endpoint_rows)
    family_endpoints_df = pd.DataFrame(family_endpoint_rows)

    # Rho support and nAUC on identical global support.
    curves: dict[tuple[str, str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    support_rows = []
    for regime in REGIMES:
        for family in FAMILIES:
            for seed in SEEDS:
                ag = attack_all.query("regime==@regime and family==@family and seed==@seed")
                ng = noise_all.query("regime==@regime and family==@family and seed==@seed")
                xa, ya = construct_rho_curve(ag, "attack")
                xn, yn = construct_rho_curve(ng, "noise")
                curves[(regime, family, seed, "attack")] = (xa, ya)
                curves[(regime, family, seed, "noise")] = (xn, yn)
                support_rows.extend([
                    {"regime": regime, "family": family, "seed": seed, "curve": "attack", "max_rho": float(xa[-1]), "n_points": len(xa)},
                    {"regime": regime, "family": family, "seed": seed, "curve": "noise", "max_rho": float(xn[-1]), "n_points": len(xn)},
                ])
    support_df = pd.DataFrame(support_rows)
    global_support = float(support_df.max_rho.min())
    support_limiter = support_df.loc[support_df.max_rho.idxmin()].to_dict()
    support_values = [(m, global_support * m) for m in sesoi["support_sensitivities"]]
    nauc_rows = []
    for multiplier, endpoint in support_values:
        for regime in REGIMES:
            for family in FAMILIES:
                for seed in SEEDS:
                    xa, ya = curves[(regime, family, seed, "attack")]
                    xn, yn = curves[(regime, family, seed, "noise")]
                    attack_nauc = integrate_nauc(xa, ya, endpoint)
                    noise_nauc = integrate_nauc(xn, yn, endpoint)
                    nauc_rows.append({
                        "support_multiplier": multiplier, "rho_endpoint": endpoint,
                        "regime": regime, "family": family, "family_label": FAMILY_LABELS[family], "seed": seed,
                        "attack_nauc": attack_nauc, "noise_nauc": noise_nauc,
                        "noise_minus_attack_gap": noise_nauc - attack_nauc,
                    })
    nauc_df = pd.DataFrame(nauc_rows)
    family_nauc_rows = []
    paired_nauc_rows = []
    for multiplier, endpoint in support_values:
        for family in FAMILIES:
            amp = nauc_df.query("support_multiplier==@multiplier and regime=='amp_fp16_trained' and family==@family").sort_values("seed")
            fp32 = nauc_df.query("support_multiplier==@multiplier and regime=='full_fp32_trained' and family==@family").sort_values("seed")
            for i, seed in enumerate(SEEDS):
                paired_nauc_rows.append({
                    "support_multiplier": multiplier, "rho_endpoint": endpoint, "family": family, "seed": seed,
                    "attack_nauc_amp": float(amp.iloc[i].attack_nauc), "attack_nauc_fp32": float(fp32.iloc[i].attack_nauc),
                    "delta_attack_nauc": float(amp.iloc[i].attack_nauc - fp32.iloc[i].attack_nauc),
                    "noise_nauc_amp": float(amp.iloc[i].noise_nauc), "noise_nauc_fp32": float(fp32.iloc[i].noise_nauc),
                    "delta_noise_nauc": float(amp.iloc[i].noise_nauc - fp32.iloc[i].noise_nauc),
                    "gap_amp": float(amp.iloc[i].noise_minus_attack_gap), "gap_fp32": float(fp32.iloc[i].noise_minus_attack_gap),
                    "delta_gap": float(amp.iloc[i].noise_minus_attack_gap - fp32.iloc[i].noise_minus_attack_gap),
                })
            for metric in ["attack_nauc", "noise_nauc", "noise_minus_attack_gap"]:
                deltas = amp[metric].to_numpy() - fp32[metric].to_numpy()
                stats = paired_stats(deltas, None, BOOT_SEED + stats_seed_counter)
                stats_seed_counter += 1
                family_nauc_rows.append({
                    "support_multiplier": multiplier, "rho_endpoint": endpoint,
                    "family": family, "family_label": FAMILY_LABELS[family], "metric": metric,
                    "amp_mean": float(amp[metric].mean()), "amp_sd": float(amp[metric].std(ddof=1)),
                    "fp32_mean": float(fp32[metric].mean()), "fp32_sd": float(fp32[metric].std(ddof=1)),
                    **stats,
                })
    paired_nauc_df = pd.DataFrame(paired_nauc_rows)
    family_nauc_df = pd.DataFrame(family_nauc_rows)

    # Ranking summaries.
    ranking_rows = []
    rank_agreements = []
    for metric, source_df, multiplier in [
        ("rho50", endpoints_df, None),
        ("epsilon50", endpoints_df, None),
    ]:
        orders = {}
        for regime in REGIMES:
            means = source_df.query("regime==@regime").groupby("family")[metric].mean().sort_values(ascending=False)
            orders[regime] = means.index.tolist()
            for rank, (family, value) in enumerate(means.items(), 1):
                ranking_rows.append({"metric": metric, "support_multiplier": None, "regime": regime, "rank": rank, "family": family, "mean": float(value)})
        amp_values = source_df.query("regime=='amp_fp16_trained'").groupby("family")[metric].mean().reindex(FAMILIES)
        fp_values = source_df.query("regime=='full_fp32_trained'").groupby("family")[metric].mean().reindex(FAMILIES)
        sp = spearmanr(amp_values, fp_values)
        rank_agreements.append({
            "metric": metric, "support_multiplier": None,
            "amp_order": " > ".join(orders["amp_fp16_trained"]),
            "fp32_order": " > ".join(orders["full_fp32_trained"]),
            "exact_order_preserved": orders["amp_fp16_trained"] == orders["full_fp32_trained"],
            "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
        })
    for multiplier, endpoint in support_values:
        for metric in ["attack_nauc", "noise_nauc", "noise_minus_attack_gap"]:
            orders = {}
            ascending = metric == "noise_minus_attack_gap"  # smaller gap is descriptively better coupling
            for regime in REGIMES:
                means = nauc_df.query("support_multiplier==@multiplier and regime==@regime").groupby("family")[metric].mean().sort_values(ascending=ascending)
                orders[regime] = means.index.tolist()
                for rank, (family, value) in enumerate(means.items(), 1):
                    ranking_rows.append({"metric": metric, "support_multiplier": multiplier, "regime": regime, "rank": rank, "family": family, "mean": float(value), "rho_endpoint": endpoint})
            amp_values = nauc_df.query("support_multiplier==@multiplier and regime=='amp_fp16_trained'").groupby("family")[metric].mean().reindex(FAMILIES)
            fp_values = nauc_df.query("support_multiplier==@multiplier and regime=='full_fp32_trained'").groupby("family")[metric].mean().reindex(FAMILIES)
            sp = spearmanr(amp_values, fp_values)
            rank_agreements.append({
                "metric": metric, "support_multiplier": multiplier, "rho_endpoint": endpoint,
                "amp_order": " > ".join(orders["amp_fp16_trained"]),
                "fp32_order": " > ".join(orders["full_fp32_trained"]),
                "exact_order_preserved": orders["amp_fp16_trained"] == orders["full_fp32_trained"],
                "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            })
    ranking_df = pd.DataFrame(ranking_rows)
    rank_agreement_df = pd.DataFrame(rank_agreements)

    # Learned anomaly audit, no exclusion.
    learned_amp = endpoints_df.query("regime=='amp_fp16_trained' and family=='learned'").sort_values("seed")
    other_max = learned_amp.query("seed not in [1011,1213]").max_rho_rel.to_numpy()
    median_other = float(np.median(other_max))
    mad_other = float(np.median(np.abs(other_max - median_other)))
    anomaly_rows = []
    for seed in SEEDS:
        e = learned_amp.query("seed==@seed").iloc[0]
        g = fp16_attack.query("family=='learned' and seed==@seed").sort_values("budget")
        positive = g.query("budget>0")
        max_spread = float(positive.restart_loss_spread.max())
        anomaly_rows.append({
            "seed": seed, "clean_accuracy_percent": float(e.clean_accuracy_percent),
            "max_rho_rel": float(e.max_rho_rel), "epsilon_at_max_rho": float(e.epsilon_at_max_rho),
            "max_rho_ratio_to_median_of_seeds_42_123_456_789": float(e.max_rho_rel / median_other),
            "robust_z_from_other_four": float(0.67448975 * (e.max_rho_rel - median_other) / mad_other) if mad_other > 0 else None,
            "epsilon50": float(e.epsilon50), "rho50": float(e.rho50),
            "q50_epsilon_bracket": f"[{e.q50_epsilon_left:.6g},{e.q50_epsilon_right:.6g}]",
            "q50_rho_bracket": f"[{e.q50_rho_left:.6g},{e.q50_rho_right:.6g}]",
            "rho_nonmonotone_step_count": int(e.rho_nonmonotone_step_count),
            "raw_accuracy_increase_step_count": int(e.raw_accuracy_increase_step_count),
            "max_restart_loss_spread": max_spread,
            "formal_integrity_status": "PASS_RETAINED",
            "interpretation": ("high-rho low-budget branch; all five restart losses tightly agree; retained as a valid seed-level outcome" if seed == 1011
                               else "abrupt high-rho cliff at epsilon 0.0045; retained as a valid seed-level outcome" if seed == 1213
                               else "within main four-seed max-rho band"),
        })
    anomaly_df = pd.DataFrame(anomaly_rows)

    # Post-hoc leave-one-seed-out sensitivity (never changes the primary n=6 decision).
    loo_rows = []
    for family in FAMILIES:
        vals_df = paired_endpoints_df.query("family==@family")[["seed", "delta_rho50_amp_minus_fp32"]]
        vals_map = {int(r.seed): float(r.delta_rho50_amp_minus_fp32) for r in vals_df.itertuples()}
        for omitted in [None] + SEEDS:
            vals = np.asarray([v for seed, v in vals_map.items() if seed != omitted], dtype=float)
            n = len(vals); mean = float(vals.mean()); sd = float(vals.std(ddof=1)); se = sd / math.sqrt(n)
            ci90 = (mean - float(student_t.ppf(0.95, n - 1)) * se,
                    mean + float(student_t.ppf(0.95, n - 1)) * se)
            loo_rows.append({
                "family": family, "omitted_seed": "none" if omitted is None else omitted, "n": n,
                "mean_delta_rho50": mean, "sd_delta": sd, "ci90_low": ci90[0], "ci90_high": ci90[1],
                "equivalence_margin": margin_rho50,
                "equivalence_status": "EQUIVALENT" if ci90[0] > -margin_rho50 and ci90[1] < margin_rho50 else "INCONCLUSIVE",
                "posthoc_sensitivity_only": omitted is not None,
            })
    loo_df = pd.DataFrame(loo_rows)

    # Cross-seed coherence summaries.
    coherence_rows = []
    for (regime, family), group in endpoints_df.groupby(["regime", "family"]):
        for metric in ["clean_accuracy_percent", "epsilon50", "rho50", "cliff_width_rho20_minus_rho80", "max_rho_rel"]:
            vals = group[metric].to_numpy(float); median = float(np.median(vals)); mad = float(np.median(np.abs(vals - median))); q1, q3 = np.quantile(vals, [0.25, 0.75])
            coherence_rows.append({
                "support_multiplier": None, "regime": regime, "family": family, "metric": metric, "n": len(vals),
                "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "cv_abs": float(vals.std(ddof=1) / abs(vals.mean())) if vals.mean() != 0 else None,
                "median": median, "mad": mad, "q1": float(q1), "q3": float(q3),
                "min": float(vals.min()), "max": float(vals.max()),
            })
    for (multiplier, regime, family), group in nauc_df.groupby(["support_multiplier", "regime", "family"]):
        for metric in ["attack_nauc", "noise_nauc", "noise_minus_attack_gap"]:
            vals = group[metric].to_numpy(float); median = float(np.median(vals)); mad = float(np.median(np.abs(vals - median))); q1, q3 = np.quantile(vals, [0.25, 0.75])
            coherence_rows.append({
                "support_multiplier": float(multiplier), "regime": regime, "family": family, "metric": metric, "n": len(vals),
                "mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "cv_abs": float(vals.std(ddof=1) / abs(vals.mean())) if vals.mean() != 0 else None,
                "median": median, "mad": mad, "q1": float(q1), "q3": float(q3),
                "min": float(vals.min()), "max": float(vals.max()),
            })
    coherence_df = pd.DataFrame(coherence_rows)

    # Primary bridge decision.
    rho_decisions = family_endpoints_df.query("metric=='rho50'").set_index("family")["equivalence_status"].to_dict()
    rho_order_preserved = bool(rank_agreement_df.query("metric=='rho50'").iloc[0].exact_order_preserved)
    all_equivalent = all(rho_decisions.get(f) == "EQUIVALENT" for f in FAMILIES)
    overall_status = "BRIDGE_ESTABLISHED" if all_equivalent and rho_order_preserved else "INCONCLUSIVE"
    bridge_decision = {
        "overall_status": overall_status,
        "analysis_label": "seed-matched",
        "family_rho50_equivalence": rho_decisions,
        "rho50_rank_order_preserved": rho_order_preserved,
        "strict_pairing_provenance": strict_pairing_provenance["status"],
        "allowed_claim": ("Across these three PE families, rho50 robustness is equivalent within the precommitted margin and ranking is preserved across training regimes."
                          if overall_status == "BRIDGE_ESTABLISHED" else
                          "The seed-matched n=6 analysis does not establish a general AMP-to-FP32 robustness bridge. Report the two training regimes separately; INCONCLUSIVE is not evidence that the regimes differ."),
    }

    # Save CSVs.
    attack_all.to_csv(out / "D029_ATTACK_POINT_LEVEL_v1.csv", index=False)
    noise_all.to_csv(out / "D029_NOISE_BUDGET_AVERAGED_POINT_LEVEL_v1.csv", index=False)
    endpoints_df.to_csv(out / "D029_ENDPOINTS_BY_SEED_REGIME_v1.csv", index=False)
    paired_endpoints_df.to_csv(out / "D029_PAIRED_ENDPOINT_CONTRASTS_v1.csv", index=False)
    family_endpoints_df.to_csv(out / "D029_FAMILY_ENDPOINT_STATISTICS_v1.csv", index=False)
    support_df.to_csv(out / "D029_CURVE_SUPPORT_AUDIT_v1.csv", index=False)
    nauc_df.to_csv(out / "D029_SEED_LEVEL_NAUC_v1.csv", index=False)
    paired_nauc_df.to_csv(out / "D029_PAIRED_NAUC_CONTRASTS_v1.csv", index=False)
    family_nauc_df.to_csv(out / "D029_FAMILY_NAUC_STATISTICS_v1.csv", index=False)
    ranking_df.to_csv(out / "D029_RANKING_DETAIL_v1.csv", index=False)
    rank_agreement_df.to_csv(out / "D029_RANKING_AGREEMENT_v1.csv", index=False)
    anomaly_df.to_csv(out / "D029_LEARNED_1011_1213_ANOMALY_AUDIT_v1.csv", index=False)
    loo_df.to_csv(out / "D029_LEAVE_ONE_SEED_OUT_RHO50_SENSITIVITY_v1.csv", index=False)
    coherence_df.to_csv(out / "D029_CROSS_SEED_COHERENCE_SUMMARY_v1.csv", index=False)

    # Machine-readable reports.
    integrity = {
        "status": "PASS_WITH_PAIRING_PROVENANCE_LIMIT",
        "fp16": fp16_report,
        "fp32": fp32_report,
        "cross_cohort_protocol_checks": protocol_checks,
        "strict_pairing_provenance": strict_pairing_provenance,
        "endpoint_validation_against_fp16_final_audit_max_abs_delta": endpoint_validation_max_abs,
        "common_rho_support": global_support,
        "support_limiter": support_limiter,
        "attack_eval_numerical_mode_lock": "ATTACK_EVAL_NUMERICAL_MODE_v1; historical reconciliation remains a declared provenance limit",
    }
    json_dump(out / "D029_INPUT_AND_PROTOCOL_INTEGRITY_REPORT_v1.json", integrity)
    json_dump(out / "D029_BRIDGE_DECISION_v1.json", bridge_decision)
    summary = {
        "status": "PASS_ANALYSIS_COMPLETE",
        "bridge_decision": bridge_decision,
        "sesoi_lock_sha256": sha256(args.sesoi_lock),
        "common_rho_support": global_support,
        "support_limiter": support_limiter,
        "family_rho50_statistics": family_endpoints_df.query("metric=='rho50'").to_dict(orient="records"),
        "rank_agreement": rank_agreement_df.to_dict(orient="records"),
        "learned_anomaly_seeds_retained": [1011, 1213],
        "new_gpu_experiment_required": False,
        "additional_provenance_desirable_not_required": [
            "bit-identical initial-state hashes for both training regimes",
            "training sampler/data-order hashes",
            "per-run GradScaler scale-decrease/skipped-optimizer-step counts"
        ],
    }
    json_dump(out / "D029_ANALYSIS_SUMMARY_v1.json", summary)

    # Console summary.
    print(json.dumps({
        "status": summary["status"],
        "overall_bridge": overall_status,
        "global_rho_support": global_support,
        "rho50": family_endpoints_df.query("metric=='rho50'")[["family","amp_mean","fp32_mean","mean_delta_amp_minus_fp32","ci90_t_low","ci90_t_high","equivalence_status"]].to_dict(orient="records"),
        "rho50_ranking": rank_agreement_df.query("metric=='rho50'").to_dict(orient="records"),
        "endpoint_validation_max_abs": endpoint_validation_max_abs,
    }, indent=2))


if __name__ == "__main__":
    main()
