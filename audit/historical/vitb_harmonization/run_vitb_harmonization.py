#!/usr/bin/env python3
"""Targeted ViT-B convergence harmonization on prespecified seed 123.

The wrapper reads the existing canonical N x 5 result for one PE family,
selects three distinct transition budgets closest to normalized accuracies
0.9, 0.7, and 0.5, and runs the corresponding 2N x 5 attack.

Family schedules:
    learned:     400 -> 800 steps
    sinusoidal:  400 -> 800 steps
    rope:        200 -> 400 steps
    alibi:       100 -> 200 steps

Typical use:
    python -u run_vitb_harmonization.py --family rope --dry-run
    python -u run_vitb_harmonization.py --family rope
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


USERNAME = "djoko.bandjur.ftnkm"
HOME_DIR = Path(f"/home/{USERNAME}")
PYTHON_BIN = Path("/usr/bin/python")

ENGINE = (
    HOME_DIR
    / "Notebooks"
    / "cross_family_final_n6_FMLE"
    / "code"
    / "cross_family_rho_decoupling.py"
)
CODE_DIR = ENGINE.parent

MODELS_DIR = HOME_DIR / "Notebooks" / "ImageNet100_checkpoints"
VAL_DIR = (
    HOME_DIR
    / "datasets"
    / "imagenet100"
    / "imagenet100_resized"
    / "val"
)

BASELINE_DIR = (
    HOME_DIR
    / "Notebooks"
    / "results"
    / "cross_family_rho_final_n6"
    / "imagenet"
    / "session_2_seeds_123"
)

OUTPUT_DIR = (
    HOME_DIR
    / "Notebooks"
    / "results"
    / "vitb_harmonization"
    / "imagenet"
    / "seed_123"
)

TARGET_NORMALIZED_ACCURACIES = (0.9, 0.7, 0.5)

FAMILY_PROTOCOLS: dict[str, dict[str, int]] = {
    "learned": {
        "baseline_steps": 400,
        "target_steps": 800,
    },
    "sinusoidal": {
        "baseline_steps": 400,
        "target_steps": 800,
    },
    "rope": {
        "baseline_steps": 200,
        "target_steps": 400,
    },
    "alibi": {
        "baseline_steps": 100,
        "target_steps": 200,
    },
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_for(
    document: dict[str, Any],
    family: str,
    seed: int,
) -> dict[str, Any]:
    return document["results"][family][str(seed)]


def attack_points_for(
    document: dict[str, Any],
    family: str,
    seed: int,
) -> list[dict[str, Any]]:
    attacks = result_for(document, family, seed)["attacks"]["pgd_pe"]
    points: list[dict[str, Any]] = []

    for point in attacks.values():
        budget = float(point["budget"])
        normalized_accuracy = point.get("normalized_accuracy")

        if budget <= 0 or normalized_accuracy is None:
            continue

        points.append(
            {
                "budget": budget,
                "normalized_accuracy": float(normalized_accuracy),
                "attack_loss": point.get("selected_attack_loss"),
                "rho_rel": float(point["rho"]["rho_rel"]),
                "restart_records": point.get("restart_records", []),
            }
        )

    return sorted(points, key=lambda item: item["budget"])


def select_transition_points(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select three distinct existing budgets nearest 0.9, 0.7, and 0.5."""
    if len(points) < len(TARGET_NORMALIZED_ACCURACIES):
        raise RuntimeError(
            "Baseline does not contain enough distinct non-zero attack budgets."
        )

    remaining = list(points)
    selected: list[dict[str, Any]] = []

    for target in TARGET_NORMALIZED_ACCURACIES:
        best = min(
            remaining,
            key=lambda point: (
                abs(point["normalized_accuracy"] - target),
                point["budget"],
            ),
        )
        chosen = dict(best)
        chosen["target_normalized_accuracy"] = target
        selected.append(chosen)
        remaining.remove(best)

    return sorted(selected, key=lambda item: item["budget"])


def find_budget_point(
    document: dict[str, Any],
    family: str,
    seed: int,
    budget: float,
) -> dict[str, Any]:
    attacks = result_for(document, family, seed)["attacks"]["pgd_pe"]

    for point in attacks.values():
        if abs(float(point["budget"]) - budget) < 1e-12:
            return point

    raise KeyError(
        f"Budget={budget} was not found for family={family}, seed={seed}."
    )


def validate_baseline(
    document: dict[str, Any],
    family: str,
    seed: int,
    expected_steps: int,
) -> None:
    metadata = document["metadata"]
    config = metadata["config"]

    if metadata.get("seeds") != [seed]:
        raise RuntimeError(
            f"Baseline seed mismatch: expected [{seed}], got {metadata.get('seeds')}."
        )

    if metadata.get("pe_types") != [family]:
        raise RuntimeError(
            f"Baseline PE mismatch: expected [{family}], "
            f"got {metadata.get('pe_types')}."
        )

    if int(config["pgd_steps"]) != expected_steps:
        raise RuntimeError(
            f"Baseline step mismatch for {family}: expected {expected_steps}, "
            f"got {config['pgd_steps']}."
        )

    if int(config["pgd_restarts"]) != 5:
        raise RuntimeError(
            f"Baseline must use five restarts; got {config['pgd_restarts']}."
        )

    if config["parameter_norm"] != "global_rms":
        raise RuntimeError(
            f"Baseline parameter norm is {config['parameter_norm']}, not global_rms."
        )

    split = metadata.get("split", {})
    expected_split = {
        "n_calibration": 256,
        "n_attack": 1280,
        "n_eval": 3464,
    }
    for key, expected in expected_split.items():
        if key in split and int(split[key]) != expected:
            raise RuntimeError(
                f"Baseline split mismatch for {key}: "
                f"expected {expected}, got {split[key]}."
            )


def compare_restart_seeds(
    baseline: dict[str, Any],
    new_result: dict[str, Any],
    family: str,
    seed: int,
    budgets: list[float],
    baseline_steps: int,
    target_steps: int,
) -> None:
    for budget in budgets:
        old_point = find_budget_point(
            baseline,
            family,
            seed,
            budget,
        )
        new_point = find_budget_point(
            new_result,
            family,
            seed,
            budget,
        )

        old_seeds = [
            int(record["seed"])
            for record in old_point.get("restart_records", [])
        ]
        new_seeds = [
            int(record["seed"])
            for record in new_point.get("restart_records", [])
        ]

        if len(old_seeds) != 5 or len(new_seeds) != 5:
            raise RuntimeError(
                f"Expected five restart records at budget={budget}; "
                f"got {len(old_seeds)} and {len(new_seeds)}."
            )

        if old_seeds != new_seeds:
            raise RuntimeError(
                "\nRestart seed mismatch.\n"
                f"family={family}, seed={seed}, budget={budget}\n"
                f"{baseline_steps} steps: {old_seeds}\n"
                f"{target_steps} steps: {new_seeds}"
            )

        print(
            f"RESTART_SEEDS_MATCH family={family} seed={seed} "
            f"budget={budget} {baseline_steps}->{target_steps}: {old_seeds}",
            flush=True,
        )


def validate_new_result(
    baseline: dict[str, Any],
    new_result: dict[str, Any],
    family: str,
    seed: int,
    target_steps: int,
) -> None:
    metadata = new_result["metadata"]
    config = metadata["config"]

    if metadata.get("seeds") != [seed]:
        raise RuntimeError("New result has the wrong seed.")
    if metadata.get("pe_types") != [family]:
        raise RuntimeError("New result has the wrong PE family.")
    if int(config["pgd_steps"]) != target_steps:
        raise RuntimeError("New result has the wrong PGD step count.")
    if int(config["pgd_restarts"]) != 5:
        raise RuntimeError("New result does not contain five restarts.")
    if config["parameter_norm"] != "global_rms":
        raise RuntimeError("New result does not use global_rms.")

    old_block = result_for(baseline, family, seed)
    new_block = result_for(new_result, family, seed)

    if old_block["checkpoint"] != new_block["checkpoint"]:
        raise RuntimeError(
            "Checkpoint path differs between baseline and harmonization result."
        )

    old_hash = old_block.get("checkpoint_sha256")
    new_hash = new_block.get("checkpoint_sha256")
    if old_hash is not None and new_hash is not None and old_hash != new_hash:
        raise RuntimeError(
            "Checkpoint SHA-256 differs between baseline and harmonization result."
        )

    clean_delta = abs(
        float(old_block["clean_acc_eval"])
        - float(new_block["clean_acc_eval"])
    )
    if clean_delta > 1e-12:
        raise RuntimeError(
            f"Clean accuracy differs between runs by {clean_delta}."
        )


def run_with_live_log(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> None:
    with log_path.open("x", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
            log_handle.flush()

        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ViT-B N->2N convergence harmonization audit on "
            "prespecified seed 123."
        )
    )
    parser.add_argument(
        "--family",
        required=True,
        choices=tuple(FAMILY_PROTOCOLS),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Prespecified confirmation seed; default: 123.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print selected budgets and command only.",
    )
    args = parser.parse_args()

    family = args.family
    seed = args.seed
    protocol = FAMILY_PROTOCOLS[family]
    baseline_steps = int(protocol["baseline_steps"])
    target_steps = int(protocol["target_steps"])

    baseline_path = BASELINE_DIR / f"attacks_{family}.json"

    required_paths = (
        PYTHON_BIN,
        ENGINE,
        MODELS_DIR,
        VAL_DIR,
        baseline_path,
    )
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required path: {required_path}")

    baseline = load_json(baseline_path)
    validate_baseline(
        baseline,
        family,
        seed,
        baseline_steps,
    )

    all_points = attack_points_for(
        baseline,
        family,
        seed,
    )
    selected = select_transition_points(all_points)
    budgets = [float(point["budget"]) for point in selected]

    print(
        f"Harmonization protocol: family={family}, seed={seed}, "
        f"{baseline_steps}x5 -> {target_steps}x5",
        flush=True,
    )
    print("\nSelected transition points from the existing baseline result:")
    print("-" * 88)
    for point in selected:
        print(
            f"target={point['target_normalized_accuracy']:.1f} | "
            f"budget={point['budget']:.8g} | "
            f"norm_acc={point['normalized_accuracy']:.6f} | "
            f"rho_rel={point['rho_rel']:.6f} | "
            f"loss={point['attack_loss']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUTPUT_DIR
        / f"vitb_{family}_seed{seed}_harmonization_{target_steps}steps.json"
    )
    log_path = output_path.with_suffix(".log")

    budget_flag = f"--budgets-{family}"
    command = [
        str(PYTHON_BIN),
        "-u",
        str(ENGINE),
        "--dataset",
        "imagenet",
        "--models-dir",
        str(MODELS_DIR),
        "--val-dir",
        str(VAL_DIR),
        "--scripts-dir",
        str(CODE_DIR),
        "--output-path",
        str(output_path),
        "--seeds",
        str(seed),
        "--calibration-images",
        "256",
        "--attack-images",
        "1280",
        "--split-seed",
        "0",
        "--rho-batch",
        "4",
        "--attack-batch",
        "8",
        "--eval-batch",
        "32",
        "--num-workers",
        "4",
        "--parameter-norm",
        "global_rms",
        "--pgd-restarts",
        "5",
        "--pgd-alpha-ratio",
        "0.05",
        "--pgd-seed-steps",
        "50",
        "--pgd-seed-alpha-ratio",
        "0.05",
        "--noise-draws",
        "10",
        "--noise-distribution",
        "gaussian",
        "--rope-parameterization",
        "phase",
        "--device",
        "cuda",
        "--pe-types",
        family,
        "--stages",
        "attacks",
        budget_flag,
        "0",
        *[str(budget) for budget in budgets],
        "--pgd-steps",
        str(target_steps),
    ]

    print("\nCommand:")
    print(shlex.join(command))
    print()

    # A dry run must remain usable even if an old output or orphan log exists.
    if args.dry_run:
        print("DRY_RUN_COMPLETE")
        return

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing JSON: {output_path}"
        )
    if log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing log: {log_path}"
        )

    environment = os.environ.copy()
    environment.update(
        {
            "USER": USERNAME,
            "LOGNAME": USERNAME,
            "USERNAME": USERNAME,
            "HOME": str(HOME_DIR),
            "PYTHONUNBUFFERED": "1",
            "TORCHINDUCTOR_CACHE_DIR": (
                f"/tmp/torchinductor_{os.getuid()}_"
                f"vitb_harmonization_{family}"
            ),
            "TRITON_CACHE_DIR": (
                f"/tmp/triton_{os.getuid()}_"
                f"vitb_harmonization_{family}"
            ),
        }
    )

    run_with_live_log(
        command,
        log_path,
        environment,
    )

    new_result = load_json(output_path)
    validate_new_result(
        baseline,
        new_result,
        family,
        seed,
        target_steps,
    )
    compare_restart_seeds(
        baseline,
        new_result,
        family,
        seed,
        budgets,
        baseline_steps,
        target_steps,
    )

    print("\nVITB_HARMONIZATION_RUN_COMPLETE")
    print(f"JSON: {output_path}")
    print(f"LOG:  {log_path}")


if __name__ == "__main__":
    main()
