#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local held-out geometry probe for PE robustness.

This is not a full Jacobian SVD and does not claim an exact condition number.
It measures task-direction versus random-direction behavior at matched achieved
attention-logit displacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cross_family_rho_decoupling as core


def atomic_json(payload, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manual_forward_with_z(
    model: nn.Module,
    images: torch.Tensor,
    retain_z_grad: bool = False,
):
    if model.training:
        raise RuntimeError("Model must be in eval mode")

    batch = images.shape[0]
    x = model.patch_embed(images)
    cls = model.cls_token.expand(batch, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = model.pos_encoding(x)

    layer_z = []

    for block in model.blocks:
        y = block.norm1(x)
        attn = block.attn
        bsz, n_tokens, channels = y.shape

        qkv = attn.qkv(y).reshape(
            bsz, n_tokens, 3, attn.num_heads, attn.head_dim
        ).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        if attn.pe_type == "rope":
            q, k = attn.rope(q, k, n_tokens)

        z = (q @ k.transpose(-2, -1)) * attn.scale

        if attn.pe_type == "alibi":
            z = z + attn.alibi.get_bias(n_tokens)

        if retain_z_grad:
            z.retain_grad()

        layer_z.append(z)

        weights = z.softmax(dim=-1)
        attn_out = weights @ v
        attn_out = attn_out.transpose(1, 2).reshape(
            bsz, n_tokens, channels
        )
        attn_out = attn.proj(attn_out)

        x = x + attn_out
        x = x + block.mlp(block.norm2(x))

    x = model.norm(x)
    output = model.head(x[:, 0])
    return output, layer_z


def dict_cosine(a: Mapping[str, torch.Tensor],
                b: Mapping[str, torch.Tensor]) -> float:
    dot = torch.zeros((), device=next(iter(a.values())).device)
    a2 = torch.zeros_like(dot)
    b2 = torch.zeros_like(dot)

    for name in a:
        af = a[name].float()
        bf = b[name].float()
        dot = dot + (af * bf).sum()
        a2 = a2 + af.pow(2).sum()
        b2 = b2 + bf.pow(2).sum()

    denom = torch.sqrt(a2 * b2).clamp_min(1e-30)
    return float((dot / denom).item())


def scaled_delta(direction, radius: float):
    return {
        name: tensor.detach() * float(radius)
        for name, tensor in direction.items()
    }


def compute_task_gradient_direction(
    model,
    adapter,
    loader,
    device,
):
    criterion = nn.CrossEntropyLoss(reduction="sum")

    adapter.restore()
    adapter.set_requires_grad(True)
    adapter.zero_grads()
    model.zero_grad(set_to_none=True)

    total_images = 0
    total_loss = 0.0
    t0 = time.time()

    try:
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with core.math_sdpa_context():
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()

            total_images += labels.numel()
            total_loss += float(loss.detach().item())

            if batch_index == 1 or batch_index % 8 == 0:
                print(
                    f"  gradient batch={batch_index:03d} "
                    f"images={total_images}",
                    flush=True,
                )

        if total_images == 0:
            raise RuntimeError("Gradient loader is empty")

        zero_delta = adapter.zeros()
        gradients = {
            name: value.detach().clone()
            for name, value in adapter.aggregate_grads(zero_delta).items()
        }

    finally:
        adapter.set_requires_grad(False)
        adapter.restore()

    direction = core.normalize_direction(gradients, "global_rms")

    return direction, {
        "n_images": int(total_images),
        "clean_ce": float(total_loss / total_images),
        "gradient_metrics": core.delta_metrics(gradients),
        "direction_metrics": core.delta_metrics(direction),
        "elapsed_sec": float(time.time() - t0),
    }


def measure_rho_for_radius(
    model,
    adapter,
    loader,
    device,
    direction,
    radius,
):
    delta = scaled_delta(direction, radius)

    try:
        result = core.measure_attention_logit_rho(
            model=model,
            calibration_loader=loader,
            device=device,
            adapter=adapter,
            deltas=delta,
        )
    finally:
        adapter.restore()

    return result, delta


def calibrate_to_target_rho(
    model,
    adapter,
    loader,
    device,
    direction,
    target_rho,
):
    probe_radius = 1e-4
    probe_result = None

    # Find a numerically stable local probe.
    for _ in range(6):
        probe_result, _ = measure_rho_for_radius(
            model, adapter, loader, device, direction, probe_radius
        )
        if probe_result["rho_rel"] >= 1e-6:
            break
        probe_radius *= 10.0

    if probe_result is None or probe_result["rho_rel"] <= 0:
        raise RuntimeError("Could not obtain a positive rho probe")

    local_gain = probe_result["rho_rel"] / probe_radius
    radius = target_rho / max(local_gain, 1e-20)
    radius = float(np.clip(radius, 1e-10, 2.0))

    history = [{
        "stage": "probe",
        "radius": float(probe_radius),
        "rho_rel": float(probe_result["rho_rel"]),
    }]

    final_result = None
    final_delta = None

    # Ratio correction; in the local regime this normally converges immediately.
    for correction in range(3):
        final_result, final_delta = measure_rho_for_radius(
            model, adapter, loader, device, direction, radius
        )

        achieved = float(final_result["rho_rel"])
        history.append({
            "stage": f"correction_{correction}",
            "radius": float(radius),
            "rho_rel": achieved,
        })

        relative_error = abs(achieved - target_rho) / target_rho
        if relative_error <= 0.08:
            break

        factor = target_rho / max(achieved, 1e-20)
        factor = float(np.clip(factor, 0.25, 4.0))
        radius = float(np.clip(radius * factor, 1e-10, 2.0))

    if final_result is None or final_delta is None:
        raise RuntimeError("Calibration failed")

    achieved = float(final_result["rho_rel"])

    return final_delta, {
        "target_rho_rel": float(target_rho),
        "parameter_radius_global_rms": float(radius),
        "achieved_rho_rel_calibration": achieved,
        "target_relative_error": float(
            abs(achieved - target_rho) / target_rho
        ),
        "probe_radius": float(probe_radius),
        "probe_rho_rel": float(probe_result["rho_rel"]),
        "local_functional_gain": float(local_gain),
        "final_delta_metrics": core.delta_metrics(final_delta),
        "history": history,
    }


def evaluate_matched_directions(
    model,
    adapter,
    loader,
    device,
    directions,
):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    n_layers = len(model.blocks)

    stats = {}
    for name in directions:
        stats[name] = {
            "loss_sum": 0.0,
            "correct": 0,
            "dot": 0.0,
            "dz_sq": 0.0,
            "layer_dot": np.zeros(n_layers, dtype=np.float64),
            "layer_dz_sq": np.zeros(n_layers, dtype=np.float64),
            "layer_diff_sq": np.zeros(n_layers, dtype=np.float64),
            "layer_elements": np.zeros(n_layers, dtype=np.int64),
        }

    clean_loss_sum = 0.0
    clean_correct = 0
    total_images = 0

    grad_sq = 0.0
    layer_grad_sq = np.zeros(n_layers, dtype=np.float64)
    layer_clean_sq = np.zeros(n_layers, dtype=np.float64)
    layer_clean_elements = np.zeros(n_layers, dtype=np.int64)

    t0 = time.time()

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        adapter.restore()
        model.zero_grad(set_to_none=True)

        # Images require gradients only to keep the clean computational graph
        # alive; the reported quantities are gradients with respect to z.
        clean_images = images.detach().requires_grad_(True)

        clean_output, clean_z = manual_forward_with_z(
            model, clean_images, retain_z_grad=True
        )
        clean_loss = criterion(clean_output, labels)
        clean_loss.backward()

        clean_loss_sum += float(clean_loss.detach().item())
        clean_correct += int(
            clean_output.detach().argmax(dim=1).eq(labels).sum().item()
        )
        total_images += labels.numel()

        clean_z_detached = [z.detach() for z in clean_z]
        clean_z_grad = [z.grad.detach() for z in clean_z]

        for layer_index, (z, gz) in enumerate(
            zip(clean_z_detached, clean_z_grad)
        ):
            g2 = float(gz.float().pow(2).sum().item())
            grad_sq += g2
            layer_grad_sq[layer_index] += g2
            layer_clean_sq[layer_index] += float(
                z.float().pow(2).sum().item()
            )
            layer_clean_elements[layer_index] += z.numel()

        for direction_name, delta in directions.items():
            adapter.apply(delta)

            with torch.no_grad():
                pert_output, pert_z = manual_forward_with_z(
                    model, images, retain_z_grad=False
                )
                pert_loss = criterion(pert_output, labels)

            current = stats[direction_name]
            current["loss_sum"] += float(pert_loss.item())
            current["correct"] += int(
                pert_output.argmax(dim=1).eq(labels).sum().item()
            )

            for layer_index, (z0, gz, z1) in enumerate(
                zip(clean_z_detached, clean_z_grad, pert_z)
            ):
                dz = z1.float() - z0.float()
                gz_f = gz.float()

                dot = float((dz * gz_f).sum().item())
                dz2 = float(dz.pow(2).sum().item())

                current["dot"] += dot
                current["dz_sq"] += dz2
                current["layer_dot"][layer_index] += dot
                current["layer_dz_sq"][layer_index] += dz2
                current["layer_diff_sq"][layer_index] += dz2
                current["layer_elements"][layer_index] += dz.numel()

            adapter.restore()
            del pert_output, pert_z

        if batch_index == 1 or batch_index % 4 == 0:
            print(
                f"  geometry batch={batch_index:03d} "
                f"images={total_images}",
                flush=True,
            )

        del clean_output, clean_z, clean_z_detached, clean_z_grad
        del clean_images

    if total_images == 0:
        raise RuntimeError("Geometry loader is empty")

    clean_ce = clean_loss_sum / total_images
    clean_accuracy = 100.0 * clean_correct / total_images

    layer_clean_mse = (
        layer_clean_sq / np.maximum(layer_clean_elements, 1)
    )
    clean_logit_rms = float(
        math.sqrt(float(np.mean(layer_clean_mse)))
    )

    output = {
        "clean": {
            "n_images": int(total_images),
            "ce": float(clean_ce),
            "accuracy": float(clean_accuracy),
            "clean_logit_rms": clean_logit_rms,
            "attention_gradient_l2": float(math.sqrt(max(grad_sq, 0.0))),
            "layer_attention_gradient_l2": [
                float(math.sqrt(max(v, 0.0)))
                for v in layer_grad_sq
            ],
        },
        "directions": {},
        "elapsed_sec": float(time.time() - t0),
    }

    for name, current in stats.items():
        layer_diff_mse = (
            current["layer_diff_sq"]
            / np.maximum(current["layer_elements"], 1)
        )
        rho_abs = float(
            math.sqrt(float(np.mean(layer_diff_mse)))
        )
        rho_rel = float(
            rho_abs / max(clean_logit_rms, 1e-12)
        )

        denom = math.sqrt(
            max(current["dz_sq"], 0.0) * max(grad_sq, 0.0)
        )
        alignment = (
            float(current["dot"] / denom) if denom > 0 else 0.0
        )

        layer_alignment = []
        for layer_index in range(n_layers):
            layer_denom = math.sqrt(
                max(current["layer_dz_sq"][layer_index], 0.0)
                * max(layer_grad_sq[layer_index], 0.0)
            )
            value = (
                current["layer_dot"][layer_index] / layer_denom
                if layer_denom > 0 else 0.0
            )
            layer_alignment.append(float(value))

        ce = current["loss_sum"] / total_images
        accuracy = 100.0 * current["correct"] / total_images
        delta_ce = ce - clean_ce

        output["directions"][name] = {
            "ce": float(ce),
            "delta_ce": float(delta_ce),
            "accuracy": float(accuracy),
            "drop_from_clean_pp": float(clean_accuracy - accuracy),
            "rho_abs_geometry": rho_abs,
            "rho_rel_geometry": rho_rel,
            "attention_task_alignment_cosine": alignment,
            "layer_attention_task_alignment_cosine": layer_alignment,
            "damage_efficiency_delta_ce_per_rho": float(
                delta_ce / max(rho_rel, 1e-12)
            ),
        }

    return output


def random_summary(direction_results):
    random_names = sorted(
        name for name in direction_results
        if name.startswith("random_")
    )

    def values(key):
        return np.asarray(
            [direction_results[name][key] for name in random_names],
            dtype=np.float64,
        )

    task = direction_results["task_gradient"]

    return {
        "n_random_directions": len(random_names),
        "task_alignment": float(
            task["attention_task_alignment_cosine"]
        ),
        "random_alignment_median": float(np.median(
            values("attention_task_alignment_cosine")
        )),
        "random_alignment_max": float(np.max(
            values("attention_task_alignment_cosine")
        )),
        "task_delta_ce": float(task["delta_ce"]),
        "random_delta_ce_median": float(np.median(values("delta_ce"))),
        "random_delta_ce_max": float(np.max(values("delta_ce"))),
        "task_damage_efficiency": float(
            task["damage_efficiency_delta_ce_per_rho"]
        ),
        "random_damage_efficiency_median": float(np.median(
            values("damage_efficiency_delta_ce_per_rho")
        )),
        "random_damage_efficiency_max": float(np.max(
            values("damage_efficiency_delta_ce_per_rho")
        )),
        "task_minus_random_median_alignment": float(
            task["attention_task_alignment_cosine"]
            - np.median(values("attention_task_alignment_cosine"))
        ),
        "task_minus_random_median_delta_ce": float(
            task["delta_ce"] - np.median(values("delta_ce"))
        ),
    }


def create_payload(args, n_total, split):
    return {
        "metadata": {
            "created_at": core.utc_now(),
            "experiment": "held-out local PE geometry probe",
            "interpretation_scope": (
                "task-direction versus random-direction diagnostics at "
                "matched achieved attention-logit displacement; "
                "not an exact full-Jacobian SVD"
            ),
            "dataset": args.dataset,
            "pe_types": list(args.pe_types),
            "seeds": list(args.seeds),
            "device": str(args.device_resolved),
            "n_total_images": int(n_total),
            "split": split,
            "config": {
                "target_rho_rel": float(args.target_rho),
                "random_directions": int(args.random_directions),
                "gradient_images": int(args.gradient_images),
                "geometry_images": int(args.geometry_images),
                "rho_images": int(args.rho_images),
                "parameter_norm": "global_rms",
                "random_distribution": "gaussian",
                "rope_parameterization": "phase",
            },
            "core_hashes": {
                "cross_family_rho_decoupling.py": file_sha256(
                    SCRIPT_DIR / "cross_family_rho_decoupling.py"
                ),
                "full_scale_experiment.py": file_sha256(
                    SCRIPT_DIR / "full_scale_experiment.py"
                ),
            },
        },
        "results": {},
    }


def run(args):
    device = torch.device(args.device_resolved)
    core.seed_everything(args.split_seed)

    dataset_args = SimpleNamespace(
        dataset=args.dataset,
        val_dir=args.val_dir,
    )
    dataset = core.build_dataset(dataset_args)
    n_total = len(dataset)

    cal_idx, attack_idx, eval_idx = core.make_split_indices(
        n_total=n_total,
        calibration_images=256,
        attack_images=1280,
        split_seed=args.split_seed,
    )

    geometry_idx = cal_idx[:args.geometry_images]
    rho_idx = geometry_idx[:args.rho_images]
    gradient_idx = attack_idx[:args.gradient_images]

    if set(geometry_idx) & set(gradient_idx):
        raise RuntimeError("Geometry and gradient subsets overlap")

    split = {
        "split_seed": int(args.split_seed),
        "base_calibration_images": 256,
        "base_attack_images": 1280,
        "gradient_subset": {
            "source": "attack split",
            "n": len(gradient_idx),
        },
        "geometry_subset": {
            "source": "calibration split",
            "n": len(geometry_idx),
        },
        "rho_calibration_subset": {
            "source": "first part of geometry subset",
            "n": len(rho_idx),
        },
        "gradient_geometry_overlap": 0,
    }

    gradient_loader = core.make_loader(
        dataset, gradient_idx, args.gradient_batch, args.num_workers
    )
    geometry_loader = core.make_loader(
        dataset, geometry_idx, args.geometry_batch, args.num_workers
    )
    rho_loader = core.make_loader(
        dataset, rho_idx, args.rho_batch, args.num_workers
    )

    output_path = Path(args.output_path)

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = create_payload(args, n_total, split)
        atomic_json(payload, output_path)

    cfg = core.DATASET_CONFIG[args.dataset]

    print("=" * 78)
    print("LOCAL HELD-OUT PE GEOMETRY PROBE")
    print("=" * 78)
    print("dataset:", args.dataset)
    print("device:", device)
    print("output:", output_path)
    print("target rho_rel:", args.target_rho)
    print("gradient images:", len(gradient_idx))
    print("geometry images:", len(geometry_idx))
    print("rho calibration images:", len(rho_idx))
    print("random directions:", args.random_directions)
    print()

    total_models = len(args.pe_types) * len(args.seeds)
    model_index = 0

    for pe_type in args.pe_types:
        family_results = payload["results"].setdefault(pe_type, {})

        for seed in args.seeds:
            model_index += 1
            existing = family_results.get(str(seed), {})

            if existing.get("status") == "ok":
                print(
                    f"[{model_index}/{total_models}] "
                    f"{pe_type} seed={seed}: already complete",
                    flush=True,
                )
                continue

            print()
            print("=" * 78)
            print(
                f"[{model_index}/{total_models}] "
                f"{pe_type.upper()} seed={seed}"
            )
            print("=" * 78)

            checkpoint = (
                Path(args.models_dir)
                / f"{pe_type}_seed{seed}"
                / "best_model.pth"
            )
            if not checkpoint.is_file():
                family_results[str(seed)] = {
                    "status": "missing_checkpoint",
                    "checkpoint": str(checkpoint),
                }
                atomic_json(payload, output_path)
                print("MISSING:", checkpoint)
                continue

            record = {
                "status": "running",
                "checkpoint": str(checkpoint),
                "started_at": core.utc_now(),
            }
            family_results[str(seed)] = record
            atomic_json(payload, output_path)

            model = core.load_model(
                checkpoint=checkpoint,
                pe_type=pe_type,
                dataset_cfg=cfg,
                device=device,
                scripts_dir=args.scripts_dir,
            )
            adapter = core.TargetAdapter(
                model=model,
                pe_type=pe_type,
                rope_parameterization="phase",
            )

            print("Computing task-gradient direction...", flush=True)
            task_direction, task_gradient_info = (
                compute_task_gradient_direction(
                    model, adapter, gradient_loader, device
                )
            )
            record["task_gradient"] = task_gradient_info
            record["target_adapter"] = adapter.metadata()
            atomic_json(payload, output_path)

            directions = {"task_gradient": task_direction}
            direction_meta = {
                "task_gradient": {
                    "kind": "held-out task-gradient ascent",
                    "draw": None,
                    "seed": None,
                    "parameter_space_cosine_to_task_gradient": 1.0,
                }
            }

            for draw in range(args.random_directions):
                random_seed = core.stable_int_seed(
                    "pe_geometry_random",
                    args.dataset,
                    seed,
                    draw,
                )
                random_dir = core.random_direction(
                    adapter.templates,
                    seed=random_seed,
                    norm="global_rms",
                    distribution="gaussian",
                )
                name = f"random_{draw:02d}"
                directions[name] = random_dir
                direction_meta[name] = {
                    "kind": "gaussian random",
                    "draw": int(draw),
                    "seed": int(random_seed),
                    "parameter_space_cosine_to_task_gradient":
                        dict_cosine(random_dir, task_direction),
                }

            print("Calibrating directions to matched rho_rel...", flush=True)
            matched_deltas = {}
            calibration = {}

            for direction_index, (name, direction) in enumerate(
                directions.items(), start=1
            ):
                t0 = time.time()
                delta, cal = calibrate_to_target_rho(
                    model=model,
                    adapter=adapter,
                    loader=rho_loader,
                    device=device,
                    direction=direction,
                    target_rho=args.target_rho,
                )
                cal.update(direction_meta[name])
                cal["elapsed_sec"] = float(time.time() - t0)

                matched_deltas[name] = delta
                calibration[name] = cal

                print(
                    f"  [{direction_index:02d}/{len(directions):02d}] "
                    f"{name:14s} "
                    f"radius={cal['parameter_radius_global_rms']:.6g} "
                    f"rho={cal['achieved_rho_rel_calibration']:.6f} "
                    f"err={100.0 * cal['target_relative_error']:.2f}%",
                    flush=True,
                )

            record["direction_calibration"] = calibration
            atomic_json(payload, output_path)

            print(
                "Evaluating matched directions on disjoint geometry subset...",
                flush=True,
            )
            evaluation = evaluate_matched_directions(
                model=model,
                adapter=adapter,
                loader=geometry_loader,
                device=device,
                directions=matched_deltas,
            )

            # Merge calibration and held-out measurements.
            merged = {}
            for name in directions:
                merged[name] = {
                    "calibration": calibration[name],
                    "heldout_geometry": evaluation["directions"][name],
                }

            record["clean_geometry"] = evaluation["clean"]
            record["directions"] = merged
            record["summary"] = random_summary(
                evaluation["directions"]
            )

            task_gain = calibration["task_gradient"][
                "local_functional_gain"
            ]
            random_gains = [
                calibration[name]["local_functional_gain"]
                for name in calibration
                if name.startswith("random_")
            ]
            record["summary"]["task_functional_gain"] = float(task_gain)
            record["summary"]["random_functional_gain_median"] = float(
                np.median(random_gains)
            )
            record["summary"][
                "task_to_random_median_functional_gain_ratio"
            ] = float(
                task_gain / max(float(np.median(random_gains)), 1e-20)
            )

            record["status"] = "ok"
            record["completed_at"] = core.utc_now()
            record["elapsed_geometry_sec"] = evaluation["elapsed_sec"]

            atomic_json(payload, output_path)

            summary = record["summary"]
            print()
            print(
                "SUMMARY "
                f"alignment task/random-med="
                f"{summary['task_alignment']:.4f}/"
                f"{summary['random_alignment_median']:.4f} | "
                f"deltaCE task/random-med="
                f"{summary['task_delta_ce']:.5f}/"
                f"{summary['random_delta_ce_median']:.5f}",
                flush=True,
            )

            adapter.restore()
            del adapter, model, matched_deltas, directions

            if device.type == "cuda":
                torch.cuda.empty_cache()

    payload["metadata"]["updated_at"] = core.utc_now()
    atomic_json(payload, output_path)

    missing = []
    for pe_type in args.pe_types:
        for seed in args.seeds:
            status = (
                payload["results"]
                .get(pe_type, {})
                .get(str(seed), {})
                .get("status")
            )
            if status != "ok":
                missing.append(f"{pe_type}:{seed}:{status}")

    if missing:
        raise RuntimeError("Incomplete models: " + ", ".join(missing))

    marker = output_path.with_suffix(".COMPLETE.json")
    atomic_json({
        "status": "complete",
        "dataset": args.dataset,
        "seeds": list(args.seeds),
        "pe_types": list(args.pe_types),
        "output": str(output_path),
        "completed_at": core.utc_now(),
    }, marker)

    print()
    print("DATASET JOB COMPLETE")
    print(output_path)
    print(marker)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("imagenet", "cifar"),
        required=True,
    )
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--scripts-dir", required=True)
    parser.add_argument("--output-path", required=True)

    parser.add_argument(
        "--pe-types",
        nargs="+",
        default=["learned", "sinusoidal", "rope", "alibi"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)

    parser.add_argument("--target-rho", type=float, default=0.03)
    parser.add_argument("--random-directions", type=int, default=8)
    parser.add_argument("--gradient-images", type=int, default=256)
    parser.add_argument("--geometry-images", type=int, default=64)
    parser.add_argument("--rho-images", type=int, default=32)

    parser.add_argument("--gradient-batch", type=int, required=True)
    parser.add_argument("--geometry-batch", type=int, required=True)
    parser.add_argument("--rho-batch", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    if args.dataset == "imagenet" and not args.val_dir:
        parser.error("--val-dir is required for ImageNet")
    if args.random_directions < 2:
        parser.error("Need at least two random directions")
    if args.rho_images > args.geometry_images:
        parser.error("rho-images cannot exceed geometry-images")
    if args.geometry_images > 256:
        parser.error("geometry-images cannot exceed calibration split")
    if args.gradient_images > 1280:
        parser.error("gradient-images cannot exceed attack split")

    args.device_resolved = (
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return args


if __name__ == "__main__":
    run(parse_args())
