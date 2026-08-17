#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-family functional robustness probe for positional encodings.

Purpose
-------
Evaluate stochastic (random coherent) and adversarial (PGD) perturbations of
Learned, Sinusoidal, RoPE, and ALiBi positional mechanisms on the *same old
n=6 checkpoints*, while re-parameterising every result by the achieved change
in pre-softmax attention logits:

    rho_abs = sqrt( mean_layer mean_{x,h,q,k} (Z_pert - Z_clean)^2 )
    rho_rel = rho_abs / sqrt( mean_layer mean_{x,h,q,k} Z_clean^2 )

This is the practical first-stage protocol: attacks/noise are constrained in
PE parameter space, then each point is measured on a common functional rho
axis. It is NOT yet a direct rho-constrained PGD attack.

Key protocol features
---------------------
* Original homogeneous n=6 checkpoints:
    42, 123, 456, 789, 1011, 1213
* Deterministic disjoint calibration / attack / evaluation splits
* Shared/coherent perturbation across all RoPE/ALiBi blocks
* Native phase-preserving RoPE perturbation by default; the phase delta is
  mapped to cos_cached/sin_cached while preserving cos^2+sin^2=1. A legacy
  cache-additive mode is retained only for sensitivity analysis. inv_freq is
  excluded because it is not read after caches have been constructed.
* Paired random seeds across PE families and training seeds
* Fixed random direction across the complete budget grid for each draw
* PGD with configurable global-RMS or L-infinity parameter-space projection
* Exact incremental JSON checkpointing and resume
* SDPA MATH guard for gradient correctness with differentiable ALiBi masks

Expected checkpoint layout
--------------------------
<models_dir>/learned_seed42/best_model.pth
<models_dir>/sinusoidal_seed42/best_model.pth
<models_dir>/rope_seed42/best_model.pth
<models_dir>/alibi_seed42/best_model.pth
...

Example CIFAR pilot
-------------------
python -u cross_family_rho_decoupling.py \
  --dataset cifar \
  --models-dir "/content/drive/MyDrive/Trained models_CIFAR100" \
  --val-dir "/content/cifar100_data" \
  --scripts-dir /content \
  --output-path "/content/drive/MyDrive/cross_family_rho/cifar_rho_pilot_seed42.json" \
  --pe-types learned sinusoidal rope alibi \
  --seeds 42 \
  --stages noise attacks \
  --budgets 0 0.001 0.005 0.01 0.05 0.1 0.2 0.5 1.0 \
  --noise-draws 3 \
  --calibration-images 64 \
  --attack-images 1280 \
  --pgd-steps 20 \
  --pgd-restarts 2 \
  --pgd-alpha-ratio 0.05

Final protocol should normally use calibration-images=256, pgd-steps=50,
pgd-restarts=5, and noise-draws=10 after the pilot confirms monotonic rho
coverage and a useful common overlap across PE families.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel
    from torch.nn.attention import SDPBackend as _SDPBackend
except Exception:  # pragma: no cover - older PyTorch fallback
    _sdpa_kernel = None
    _SDPBackend = None


PE_TYPES = ("learned", "sinusoidal", "rope", "alibi")
DEFAULT_SEEDS = (42, 123, 456, 789, 1011, 1213)
DEFAULT_BUDGETS = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
MODEL_KWARGS_BASE = dict(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.0)

DATASET_CONFIG = {
    "cifar": {
        "num_classes": 100,
        "img_size": 32,
        "patch_size": 4,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]),
    },
    "imagenet": {
        "num_classes": 100,
        "img_size": 224,
        "patch_size": 16,
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    },
}

TensorGroups = Dict[str, List[torch.Tensor]]
Snapshot = Dict[str, List[torch.Tensor]]
DeltaDict = Dict[str, torch.Tensor]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_int_seed(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def budget_key(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"


def save_json_atomic(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    os.replace(tmp, path)


@contextlib.contextmanager
def math_sdpa_context():
    if _sdpa_kernel is not None and _SDPBackend is not None:
        with _sdpa_kernel(backends=[_SDPBackend.MATH]):
            yield
    else:
        yield


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset and deterministic splits
# ---------------------------------------------------------------------------

def build_dataset(args: argparse.Namespace):
    cfg = DATASET_CONFIG[args.dataset]
    if args.dataset == "cifar":
        root = args.val_dir or "/content/cifar100_data"
        os.makedirs(root, exist_ok=True)
        return datasets.CIFAR100(root=root, train=False, download=False, transform=cfg["transform"])
    if not args.val_dir:
        raise ValueError("--val-dir is required for ImageNet-100")
    return datasets.ImageFolder(args.val_dir, transform=cfg["transform"])


def make_split_indices(n_total: int, calibration_images: int, attack_images: int, split_seed: int):
    if calibration_images < 1:
        raise ValueError("calibration_images must be >= 1")
    if attack_images < 1:
        raise ValueError("attack_images must be >= 1")
    if calibration_images + attack_images >= n_total:
        raise ValueError(
            f"Need a non-empty evaluation split: calibration={calibration_images}, "
            f"attack={attack_images}, total={n_total}"
        )
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_total)
    cal = perm[:calibration_images].tolist()
    attack = perm[calibration_images: calibration_images + attack_images].tolist()
    evaluation = perm[calibration_images + attack_images:].tolist()
    return cal, attack, evaluation


def make_loader(dataset, indices: Sequence[int], batch_size: int, num_workers: int) -> DataLoader:
    kwargs = dict(
        dataset=Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def import_model_class(scripts_dir: str):
    scripts_path = str(Path(scripts_dir).resolve())
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    module = importlib.import_module("full_scale_experiment")
    if not hasattr(module, "VisionTransformer"):
        raise AttributeError("full_scale_experiment.py has no VisionTransformer")
    return module.VisionTransformer


def load_model(
    checkpoint: Path,
    pe_type: str,
    dataset_cfg: Mapping,
    device: torch.device,
    scripts_dir: str,
) -> nn.Module:
    VisionTransformer = import_model_class(scripts_dir)
    model = VisionTransformer(
        **MODEL_KWARGS_BASE,
        img_size=dataset_cfg["img_size"],
        patch_size=dataset_cfg["patch_size"],
        num_classes=dataset_cfg["num_classes"],
        pe_type=pe_type,
    ).to(device)

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(state).__name__}")
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    # Freeze all ordinary weights. Target PE tensors are enabled only during PGD.
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Active positional tensors and shared-delta operations
# ---------------------------------------------------------------------------

class TargetAdapter:
    """Uniform interface for native PE perturbations.

    For Learned, Sinusoidal, and ALiBi, deltas are additive in the stored PE
    tensors. RoPE supports two modes:

    * phase (default): one shared phase perturbation is applied as
      cos(phi+delta), sin(phi+delta), preserving the unit-circle geometry.
      Gradients from cached cos/sin tensors are mapped back to phase by the
      exact chain rule.
    * cache_additive: additive deltas are applied directly to cos_cached and
      sin_cached. This reproduces the legacy active-cache attack and is kept
      only as a sensitivity analysis.
    """

    def __init__(self, model: nn.Module, pe_type: str, rope_parameterization: str):
        self.model = model
        self.pe_type = pe_type
        self.rope_parameterization = rope_parameterization
        self.actual_groups: TensorGroups
        self.originals: Snapshot
        self.templates: Dict[str, torch.Tensor]
        self.base_phases: Optional[List[torch.Tensor]] = None

        if pe_type == "learned":
            self.actual_groups = {"pos_embed": [model.pos_encoding.pos_embed]}
        elif pe_type == "sinusoidal":
            self.actual_groups = {"pe": [model.pos_encoding.pe]}
        elif pe_type == "alibi":
            self.actual_groups = {"slopes": [block.attn.alibi.slopes for block in model.blocks]}
        elif pe_type == "rope":
            cos_tensors = [block.attn.rope.cos_cached for block in model.blocks]
            sin_tensors = [block.attn.rope.sin_cached for block in model.blocks]
            self.actual_groups = {"cos_cached": cos_tensors, "sin_cached": sin_tensors}
        else:
            raise ValueError(f"Unsupported PE type: {pe_type}")

        self.originals = {
            name: [tensor.detach().clone() for tensor in tensors]
            for name, tensors in self.actual_groups.items()
        }

        if pe_type == "rope" and rope_parameterization == "phase":
            self.base_phases = [
                torch.atan2(sin, cos)
                for sin, cos in zip(self.originals["sin_cached"], self.originals["cos_cached"])
            ]
            self.templates = {"phase": self.base_phases[0].detach().clone()}
        elif pe_type == "rope" and rope_parameterization == "cache_additive":
            self.templates = {
                "cos_cached": self.originals["cos_cached"][0].detach().clone(),
                "sin_cached": self.originals["sin_cached"][0].detach().clone(),
            }
        else:
            self.templates = {
                name: tensors[0].detach().clone() for name, tensors in self.originals.items()
            }

    def restore(self) -> None:
        with torch.no_grad():
            for name, tensors in self.actual_groups.items():
                for tensor, original in zip(tensors, self.originals[name]):
                    tensor.copy_(original)

    def apply(self, deltas: DeltaDict) -> None:
        with torch.no_grad():
            if self.pe_type == "rope" and self.rope_parameterization == "phase":
                assert self.base_phases is not None
                delta = deltas["phase"]
                for cos_tensor, sin_tensor, base_phase in zip(
                    self.actual_groups["cos_cached"],
                    self.actual_groups["sin_cached"],
                    self.base_phases,
                ):
                    angle = base_phase + delta
                    cos_tensor.copy_(torch.cos(angle))
                    sin_tensor.copy_(torch.sin(angle))
                return

            for name, tensors in self.actual_groups.items():
                delta = deltas[name]
                for tensor, original in zip(tensors, self.originals[name]):
                    tensor.copy_(original + delta)

    def set_requires_grad(self, value: bool) -> None:
        for tensors in self.actual_groups.values():
            for tensor in tensors:
                tensor.requires_grad_(value)

    def zero_grads(self) -> None:
        for tensors in self.actual_groups.values():
            for tensor in tensors:
                tensor.grad = None

    def aggregate_grads(self, deltas: DeltaDict) -> DeltaDict:
        if self.pe_type == "rope" and self.rope_parameterization == "phase":
            assert self.base_phases is not None
            delta = deltas["phase"]
            aggregated = torch.zeros_like(delta)
            for cos_tensor, sin_tensor, base_phase in zip(
                self.actual_groups["cos_cached"],
                self.actual_groups["sin_cached"],
                self.base_phases,
            ):
                if cos_tensor.grad is None or sin_tensor.grad is None:
                    raise RuntimeError("Missing RoPE cache gradient for phase parameterization")
                angle = base_phase + delta
                aggregated = aggregated + (
                    cos_tensor.grad * (-torch.sin(angle))
                    + sin_tensor.grad * torch.cos(angle)
                )
            return {"phase": aggregated}

        out: DeltaDict = {}
        for name, tensors in self.actual_groups.items():
            grads = [tensor.grad for tensor in tensors if tensor.grad is not None]
            if not grads:
                raise RuntimeError(f"No gradient for target group '{name}'")
            out[name] = torch.stack(grads, dim=0).sum(dim=0)
        return out

    def zeros(self) -> DeltaDict:
        return {name: torch.zeros_like(template) for name, template in self.templates.items()}

    def metadata(self) -> Dict[str, object]:
        if self.pe_type == "rope" and self.rope_parameterization == "phase":
            return {
                "parameterization": "phase-preserving",
                "unique_delta_groups": {
                    "phase": {
                        "n_shared_instances": len(self.actual_groups["cos_cached"]),
                        "unique_delta_shape": list(self.templates["phase"].shape),
                        "unique_delta_numel": int(self.templates["phase"].numel()),
                    }
                },
                "actual_forward_tensors": ["cos_cached", "sin_cached"],
                "unit_circle_preserved": True,
            }
        return {
            "parameterization": (
                "cache-additive" if self.pe_type == "rope" else "native-additive"
            ),
            "unique_delta_groups": {
                name: {
                    "n_shared_instances": len(self.actual_groups[name]),
                    "unique_delta_shape": list(template.shape),
                    "unique_delta_numel": int(template.numel()),
                }
                for name, template in self.templates.items()
            },
            "unit_circle_preserved": False if self.pe_type == "rope" else None,
        }


def clone_delta(deltas: DeltaDict) -> DeltaDict:
    return {name: delta.detach().clone() for name, delta in deltas.items()}


def global_rms(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    sum_sq = None
    count = 0
    for tensor in tensors.values():
        value = tensor.float().pow(2).sum()
        sum_sq = value if sum_sq is None else sum_sq + value
        count += tensor.numel()
    if sum_sq is None or count == 0:
        raise ValueError("Empty tensor collection")
    return torch.sqrt(sum_sq / float(count))


def global_l2(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    sum_sq = None
    for tensor in tensors.values():
        value = tensor.float().pow(2).sum()
        sum_sq = value if sum_sq is None else sum_sq + value
    if sum_sq is None:
        raise ValueError("Empty tensor collection")
    return torch.sqrt(sum_sq)


def global_linf(tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = [tensor.detach().abs().max().float() for tensor in tensors.values()]
    return torch.stack(values).max()


def delta_metrics(deltas: DeltaDict) -> Dict[str, float]:
    return {
        "parameter_global_rms": float(global_rms(deltas).item()),
        "parameter_global_l2": float(global_l2(deltas).item()),
        "parameter_global_linf": float(global_linf(deltas).item()),
        "n_unique_delta_elements": int(sum(t.numel() for t in deltas.values())),
    }


def normalize_direction(direction: DeltaDict, norm: str) -> DeltaDict:
    if norm == "global_rms":
        scale = global_rms(direction)
    elif norm == "linf":
        scale = global_linf(direction)
    else:
        raise ValueError(norm)
    if not torch.isfinite(scale) or scale.item() <= 0:
        raise RuntimeError("Degenerate random direction")
    return {name: tensor / scale for name, tensor in direction.items()}


def project_delta(deltas: DeltaDict, radius: float, norm: str) -> DeltaDict:
    if radius < 0:
        raise ValueError("Budget radius must be non-negative")
    if norm == "linf":
        return {name: tensor.clamp(-radius, radius) for name, tensor in deltas.items()}
    if norm == "global_rms":
        current = global_rms(deltas)
        if current.item() <= radius or current.item() == 0:
            return deltas
        factor = radius / current
        return {name: tensor * factor for name, tensor in deltas.items()}
    raise ValueError(norm)


def random_direction(
    templates: Mapping[str, torch.Tensor],
    seed: int,
    norm: str,
    distribution: str,
) -> DeltaDict:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    direction: DeltaDict = {}
    for name, ref in templates.items():
        if distribution == "gaussian":
            value = torch.randn(ref.shape, generator=gen, dtype=torch.float32)
        elif distribution == "uniform":
            value = 2.0 * torch.rand(ref.shape, generator=gen, dtype=torch.float32) - 1.0
        elif distribution == "rademacher":
            value = torch.randint(0, 2, ref.shape, generator=gen, dtype=torch.int64).float()
            value = value.mul_(2.0).sub_(1.0)
        else:
            raise ValueError(distribution)
        direction[name] = value.to(device=ref.device, dtype=ref.dtype)
    return normalize_direction(direction, norm)


def random_start(adapter: TargetAdapter, radius: float, seed: int, norm: str) -> DeltaDict:
    if radius == 0:
        return adapter.zeros()
    direction = random_direction(adapter.templates, seed, norm, "gaussian")
    return {name: radius * tensor for name, tensor in direction.items()}


# ---------------------------------------------------------------------------
# Evaluation and manual pre-softmax attention-logit forward
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with math_sdpa_context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            correct += logits.argmax(dim=1).eq(labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / max(total, 1)


def average_attack_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad(), math_sdpa_context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item())
            total += labels.numel()
    return total_loss / max(total, 1)


def forward_with_attention_logits(model: nn.Module, images: torch.Tensor) -> List[torch.Tensor]:
    """Manual eval forward returning pre-softmax logits from every layer.

    This mirrors the project's VisionTransformer implementation and is used
    only for rho measurement. Dropout is inactive because the model is in eval.
    Returned tensors remain on the current device.
    """
    if model.training:
        raise RuntimeError("rho forward requires model.eval()")

    batch = images.shape[0]
    x = model.patch_embed(images)
    cls = model.cls_token.expand(batch, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = model.pos_encoding(x)

    layer_logits: List[torch.Tensor] = []
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
        layer_logits.append(z)

        weights = z.softmax(dim=-1)
        attn_out = weights @ v
        attn_out = attn_out.transpose(1, 2).reshape(bsz, n_tokens, channels)
        attn_out = attn.proj(attn_out)
        x = x + attn_out
        x = x + block.mlp(block.norm2(x))

    return layer_logits


@torch.no_grad()
def measure_attention_logit_rho(
    model: nn.Module,
    calibration_loader: DataLoader,
    device: torch.device,
    adapter: TargetAdapter,
    deltas: DeltaDict,
) -> Dict[str, object]:
    """Measure layer-balanced RMS change in pre-softmax attention logits."""
    model.eval()
    n_layers = len(model.blocks)
    diff_sums = np.zeros(n_layers, dtype=np.float64)
    clean_sums = np.zeros(n_layers, dtype=np.float64)
    n_images = 0

    for images, _ in calibration_loader:
        images = images.to(device, non_blocking=True)
        batch_size = images.shape[0]

        adapter.restore()
        clean_logits = forward_with_attention_logits(model, images)

        adapter.apply(deltas)
        pert_logits = forward_with_attention_logits(model, images)

        if len(clean_logits) != n_layers or len(pert_logits) != n_layers:
            raise RuntimeError("Unexpected number of attention layers")

        for layer_idx, (clean, pert) in enumerate(zip(clean_logits, pert_logits)):
            diff_mse = (pert.float() - clean.float()).pow(2).mean().item()
            clean_mse = clean.float().pow(2).mean().item()
            diff_sums[layer_idx] += batch_size * diff_mse
            clean_sums[layer_idx] += batch_size * clean_mse

        n_images += batch_size
        del clean_logits, pert_logits

    if n_images == 0:
        raise RuntimeError("Calibration loader is empty")

    layer_diff_mse = diff_sums / n_images
    layer_clean_mse = clean_sums / n_images
    rho_abs = float(math.sqrt(float(layer_diff_mse.mean())))
    clean_logit_rms = float(math.sqrt(float(layer_clean_mse.mean())))
    rho_rel = float(rho_abs / max(clean_logit_rms, 1e-12))

    adapter.apply(deltas)
    return {
        "rho_abs": rho_abs,
        "rho_rel": rho_rel,
        "clean_logit_rms": clean_logit_rms,
        "n_calibration_images": int(n_images),
        "n_layers": int(n_layers),
        "layer_rho_abs": [float(math.sqrt(max(v, 0.0))) for v in layer_diff_mse],
        "layer_clean_logit_rms": [float(math.sqrt(max(v, 0.0))) for v in layer_clean_mse],
    }


# ---------------------------------------------------------------------------
# PGD
# ---------------------------------------------------------------------------

def pgd_attack_shared(
    model: nn.Module,
    attack_loader: DataLoader,
    device: torch.device,
    adapter: TargetAdapter,
    radius: float,
    steps: int,
    restarts: int,
    alpha_ratio: float,
    norm: str,
    training_seed: int,
    seed_steps_tag: int,
    seed_alpha_tag: float,
) -> Tuple[DeltaDict, List[Dict[str, float]]]:
    if radius == 0:
        return adapter.zeros(), []
    if steps < 1 or restarts < 1:
        raise ValueError("PGD steps and restarts must be positive")

    criterion = nn.CrossEntropyLoss(reduction="mean")
    alpha = radius * alpha_ratio
    best_loss = -float("inf")
    best_delta: Optional[DeltaDict] = None
    restart_records: List[Dict[str, float]] = []

    adapter.set_requires_grad(True)
    try:
        for restart in range(restarts):
            rseed = stable_int_seed(
                "cross_family_pgd",
                training_seed,
                budget_key(radius),
                restart,
                seed_steps_tag,
                f"{seed_alpha_tag:.12g}",
            )
            deltas = random_start(adapter, radius, rseed, norm)

            for _step in range(steps):
                adapter.apply(deltas)
                model.zero_grad(set_to_none=True)
                adapter.zero_grads()

                n_batches = 0
                with math_sdpa_context():
                    for images, labels in attack_loader:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        logits = model(images)
                        loss = criterion(logits, labels)
                        loss.backward()
                        n_batches += 1
                if n_batches == 0:
                    raise RuntimeError("Attack loader is empty")

                grads = adapter.aggregate_grads(deltas)
                if norm == "linf":
                    deltas = {
                        name: delta.detach() + alpha * grads[name].detach().sign()
                        for name, delta in deltas.items()
                    }
                elif norm == "global_rms":
                    grad_scale = global_rms(grads)
                    if not torch.isfinite(grad_scale) or grad_scale.item() <= 0:
                        raise RuntimeError("Degenerate PGD gradient")
                    deltas = {
                        name: delta.detach() + alpha * grads[name].detach() / grad_scale
                        for name, delta in deltas.items()
                    }
                else:
                    raise ValueError(norm)
                deltas = project_delta(deltas, radius, norm)

            adapter.apply(deltas)
            adapter.set_requires_grad(False)
            final_loss = average_attack_loss(model, attack_loader, device)
            adapter.set_requires_grad(True)
            restart_records.append({
                "restart": int(restart),
                "seed": int(rseed),
                "attack_loss": float(final_loss),
            })
            if final_loss > best_loss:
                best_loss = final_loss
                best_delta = clone_delta(deltas)

            adapter.restore()

    finally:
        adapter.set_requires_grad(False)
        adapter.restore()

    if best_delta is None:
        raise RuntimeError("PGD did not produce a valid perturbation")
    return best_delta, restart_records


# ---------------------------------------------------------------------------
# Result structure and execution
# ---------------------------------------------------------------------------

def load_or_create_results(args: argparse.Namespace, n_total: int, split_info: Mapping) -> MutableMapping:
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        with output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload

    return {
        "metadata": {
            "created_at": utc_now(),
            "script": Path(__file__).name,
            "protocol": "parameter-space perturbation reparameterised by achieved attention-logit rho",
            "direct_rho_constrained_pgd": False,
            "dataset": args.dataset,
            "device": str(args.device_resolved),
            "n_total_images": int(n_total),
            "split": dict(split_info),
            "pe_types": list(args.pe_types),
            "seeds": list(args.seeds),
            "config": {
                "stages": list(args.stages),
                "budgets_default": [float(v) for v in args.budgets],
                "budgets_by_pe": {
                    pe: [float(v) for v in resolve_budgets(args, pe)] for pe in args.pe_types
                },
                "parameter_norm": args.parameter_norm,
                "noise_draws": int(args.noise_draws),
                "noise_distribution": args.noise_distribution,
                "noise_pattern": "shared/coherent across RoPE/ALiBi blocks",
                "fixed_noise_direction_across_budget_grid": True,
                "paired_noise_seed_across_pe_families": True,
                "pgd_steps": int(args.pgd_steps),
                "pgd_restarts": int(args.pgd_restarts),
                "pgd_alpha_ratio": float(args.pgd_alpha_ratio),
                "pgd_seed_steps": int(args.pgd_seed_steps),
                "pgd_seed_alpha_ratio": float(args.pgd_seed_alpha_ratio),
                "rope_parameterization": args.rope_parameterization,
                "active_rope_targets": (
                    ["phase -> cos_cached/sin_cached"]
                    if args.rope_parameterization == "phase"
                    else ["cos_cached", "sin_cached"]
                ),
                "inactive_rope_inv_freq_excluded": True,
            },
            "rho_definition": {
                "absolute": "sqrt(mean_layer mean_{image,head,query,key} (Zpert-Zclean)^2)",
                "relative": "rho_abs / sqrt(mean_layer mean_{image,head,query,key} Zclean^2)",
                "layer_weighting": "equal",
                "location": "pre-softmax attention logits after PE-specific transformation and ALiBi bias",
            },
        },
        "results": {},
    }


def resolve_budgets(args: argparse.Namespace, pe_type: str) -> List[float]:
    specific = getattr(args, f"budgets_{pe_type}")
    values = specific if specific is not None else args.budgets
    values = sorted(set(float(v) for v in values))
    if any(v < 0 for v in values):
        raise ValueError("Budgets must be non-negative")
    if 0.0 not in values:
        values = [0.0] + values
    return values


def ensure_run_record(results: MutableMapping, pe_type: str, seed: int, checkpoint: Path) -> MutableMapping:
    pe_results = results.setdefault("results", {}).setdefault(pe_type, {})
    record = pe_results.setdefault(str(seed), {})
    record.setdefault("status", "running")
    record.setdefault("checkpoint", str(checkpoint))
    record.setdefault("attacks", {}).setdefault("pgd_pe", {})
    record.setdefault("noise", {
        "pattern": "shared",
        "fixed_direction_across_budget_grid": True,
        "draws": [],
    })
    return record


def get_noise_draw_record(record: MutableMapping, draw: int, seed: int, budgets: Sequence[float]):
    draws = record["noise"]["draws"]
    while len(draws) <= draw:
        draws.append({})
    current = draws[draw]
    current.setdefault("draw", int(draw))
    current.setdefault("seed", int(seed))
    current.setdefault("budgets", [float(v) for v in budgets])
    current.setdefault("points", {})
    return current


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device_resolved)
    seed_everything(args.split_seed)

    dataset = build_dataset(args)
    n_total = len(dataset)
    cal_idx, attack_idx, eval_idx = make_split_indices(
        n_total, args.calibration_images, args.attack_images, args.split_seed
    )
    split_info = {
        "seed": int(args.split_seed),
        "n_calibration": len(cal_idx),
        "n_attack": len(attack_idx),
        "n_eval": len(eval_idx),
        "calibration_attack_overlap": 0,
        "calibration_eval_overlap": 0,
        "attack_eval_overlap": 0,
    }

    calibration_loader = make_loader(dataset, cal_idx, args.rho_batch, args.num_workers)
    attack_loader = make_loader(dataset, attack_idx, args.attack_batch, args.num_workers)
    eval_loader = make_loader(dataset, eval_idx, args.eval_batch, args.num_workers)

    results = load_or_create_results(args, n_total, split_info)
    output_path = Path(args.output_path)
    cfg = DATASET_CONFIG[args.dataset]

    print(f"dataset={args.dataset} | device={device} | images={n_total}")
    print(
        f"split: calibration={len(cal_idx)}, attack={len(attack_idx)}, "
        f"eval={len(eval_idx)}, overlap=0"
    )
    print(f"output: {output_path}")

    total_runs = len(args.pe_types) * len(args.seeds)
    run_index = 0

    for pe_type in args.pe_types:
        budgets = resolve_budgets(args, pe_type)
        for training_seed in args.seeds:
            run_index += 1
            print("\n" + "=" * 78)
            print(f"[{run_index}/{total_runs}] {pe_type} seed={training_seed}")
            print("=" * 78)

            checkpoint = Path(args.models_dir) / f"{pe_type}_seed{training_seed}" / "best_model.pth"
            record = ensure_run_record(results, pe_type, training_seed, checkpoint)
            if not checkpoint.exists():
                record["status"] = "missing_checkpoint"
                save_json_atomic(results, output_path)
                print(f"MISSING: {checkpoint}")
                continue

            model = load_model(checkpoint, pe_type, cfg, device, args.scripts_dir)
            adapter = TargetAdapter(model, pe_type, args.rope_parameterization)

            adapter.restore()
            clean_acc = evaluate_accuracy(model, eval_loader, device)
            record["clean_acc_eval"] = float(clean_acc)
            record["target_adapter"] = adapter.metadata()
            record["parameter_norm"] = args.parameter_norm
            print(f"clean eval accuracy: {clean_acc:.4f}%")

            if "noise" in args.stages:
                print("\n-- random coherent noise --")
                for draw in range(args.noise_draws):
                    nseed = stable_int_seed("cross_family_noise", training_seed, draw)
                    draw_record = get_noise_draw_record(record, draw, nseed, budgets)
                    direction = random_direction(
                        adapter.templates, nseed, args.parameter_norm, args.noise_distribution
                    )
                    for budget in budgets:
                        key = budget_key(budget)
                        if key in draw_record["points"] and not args.overwrite_points:
                            continue
                        t0 = time.time()
                        deltas = {name: budget * tensor for name, tensor in direction.items()}
                        deltas = project_delta(deltas, budget, args.parameter_norm)
                        adapter.apply(deltas)
                        accuracy = evaluate_accuracy(model, eval_loader, device)
                        rho = measure_attention_logit_rho(
                            model, calibration_loader, device, adapter, deltas
                        )
                        point = {
                            "budget": float(budget),
                            "accuracy": float(accuracy),
                            "normalized_accuracy": float(accuracy / max(clean_acc, 1e-12)),
                            "drop_from_clean_pp": float(clean_acc - accuracy),
                            "delta_metrics": delta_metrics(deltas),
                            "rho": rho,
                            "elapsed_sec": float(time.time() - t0),
                        }
                        draw_record["points"][key] = point
                        adapter.restore()
                        save_json_atomic(results, output_path)
                        print(
                            f"draw={draw:02d} budget={budget:g} "
                            f"acc={accuracy:6.2f}% rho={rho['rho_abs']:.6g} "
                            f"rho_rel={rho['rho_rel']:.6g}"
                        )

            if "attacks" in args.stages:
                print("\n-- PGD coherent attack --")
                attack_points = record["attacks"]["pgd_pe"]
                for budget in budgets:
                    key = budget_key(budget)
                    if key in attack_points and not args.overwrite_points:
                        continue
                    t0 = time.time()
                    if budget == 0:
                        deltas = adapter.zeros()
                        restart_records = []
                    else:
                        deltas, restart_records = pgd_attack_shared(
                            model=model,
                            attack_loader=attack_loader,
                            device=device,
                            adapter=adapter,
                            radius=budget,
                            steps=args.pgd_steps,
                            restarts=args.pgd_restarts,
                            alpha_ratio=args.pgd_alpha_ratio,
                            norm=args.parameter_norm,
                            training_seed=training_seed,
                            seed_steps_tag=args.pgd_seed_steps,
                            seed_alpha_tag=args.pgd_seed_alpha_ratio,
                        )
                    adapter.apply(deltas)
                    accuracy = evaluate_accuracy(model, eval_loader, device)
                    rho = measure_attention_logit_rho(
                        model, calibration_loader, device, adapter, deltas
                    )
                    selected_loss = (
                        max((r["attack_loss"] for r in restart_records), default=None)
                    )
                    point = {
                        "budget": float(budget),
                        "accuracy": float(accuracy),
                        "normalized_accuracy": float(accuracy / max(clean_acc, 1e-12)),
                        "drop_from_clean_pp": float(clean_acc - accuracy),
                        "selected_attack_loss": (
                            None if selected_loss is None else float(selected_loss)
                        ),
                        "restart_records": restart_records,
                        "delta_metrics": delta_metrics(deltas),
                        "rho": rho,
                        "elapsed_sec": float(time.time() - t0),
                    }
                    attack_points[key] = point
                    adapter.restore()
                    save_json_atomic(results, output_path)
                    print(
                        f"budget={budget:g} acc={accuracy:6.2f}% "
                        f"rho={rho['rho_abs']:.6g} rho_rel={rho['rho_rel']:.6g}"
                    )

            record["status"] = "ok"
            record["completed_at"] = utc_now()
            adapter.restore()
            save_json_atomic(results, output_path)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results["metadata"]["updated_at"] = utc_now()
    save_json_atomic(results, output_path)
    print("\nCOMPLETE")
    print(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Cross-family PE robustness on a common attention-logit rho axis.",
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--scripts-dir", default="/content")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--pe-types", nargs="+", choices=PE_TYPES, default=list(PE_TYPES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--stages", nargs="+", choices=("noise", "attacks"), default=["noise", "attacks"])

    parser.add_argument("--calibration-images", type=int, default=256)
    parser.add_argument("--attack-images", type=int, default=1280)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--rho-batch", type=int, default=16)
    parser.add_argument("--attack-batch", type=int, default=64)
    parser.add_argument("--eval-batch", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BUDGETS))
    for pe in PE_TYPES:
        parser.add_argument(f"--budgets-{pe}", dest=f"budgets_{pe}", nargs="+", type=float, default=None)

    parser.add_argument("--parameter-norm", choices=("global_rms", "linf"), default="global_rms")
    parser.add_argument(
        "--rope-parameterization",
        choices=("phase", "cache_additive"),
        default="phase",
        help=(
            "phase preserves cos^2+sin^2=1 and is the recommended native RoPE perturbation; "
            "cache_additive reproduces the legacy active-cache attack"
        ),
    )
    parser.add_argument("--noise-draws", type=int, default=10)
    parser.add_argument(
        "--noise-distribution", choices=("gaussian", "uniform", "rademacher"), default="gaussian"
    )

    parser.add_argument("--pgd-steps", type=int, default=50)
    parser.add_argument("--pgd-restarts", type=int, default=5)
    parser.add_argument("--pgd-alpha-ratio", type=float, default=0.05)
    parser.add_argument("--pgd-seed-steps", type=int, default=50)
    parser.add_argument("--pgd-seed-alpha-ratio", type=float, default=0.05)

    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace the entire output JSON")
    parser.add_argument(
        "--overwrite-points", action="store_true", help="Recompute points already present in a resumed JSON"
    )
    args = parser.parse_args()

    if args.dataset == "imagenet" and not args.val_dir:
        parser.error("--val-dir is required for ImageNet-100")
    if args.noise_draws < 1:
        parser.error("--noise-draws must be positive")
    if args.pgd_alpha_ratio <= 0:
        parser.error("--pgd-alpha-ratio must be positive")
    if args.rho_batch < 1 or args.attack_batch < 1 or args.eval_batch < 1:
        parser.error("Batch sizes must be positive")

    args.device_resolved = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
