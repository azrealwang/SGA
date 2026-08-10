"""Shared helpers for building reference pool XMLs with PGD."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from common.reference import load_reference_xml, write_reference_xml
from common.utils import chunk_tensor


def merge_reference_xml_paths(xml_paths: Sequence[str]) -> Tuple[List[str], Dict[str, str]]:
    """Union model ids from multiple reference XMLs; error on architecture conflict."""
    order: List[str] = []
    arch_by_id: Dict[str, str] = {}
    for xml_path in xml_paths:
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Reference XML not found: {xml_path}")
        for entry in load_reference_xml(xml_path):
            mid = str(entry["model_id"])
            arch = str(entry.get("architecture", mid))
            if mid in arch_by_id and arch_by_id[mid] != arch:
                raise ValueError(
                    f"Conflicting architecture for model {mid!r}: "
                    f"{arch_by_id[mid]!r} vs {arch!r} (from {xml_path})"
                )
            if mid not in arch_by_id:
                arch_by_id[mid] = arch
                order.append(mid)
    if not order:
        raise ValueError("No models found in --reference_xml inputs")
    return order, arch_by_id


def linf_from_int(value: int, name: str) -> float:
    if value <= 0 or value > 255:
        raise ValueError(f"{name} must be in 1..255 (will divide by 255), got {value}")
    return float(value) / 255.0


def resolve_pgd_params(
    threat_model: str,
    eps: float,
    step_size: float,
) -> Tuple[float, float]:
    norm = threat_model.strip().lower()
    if norm == "linf":
        return (
            linf_from_int(int(eps), "--eps"),
            linf_from_int(int(step_size), "--pgd_step_size"),
        )
    if norm == "l2":
        eps_f = float(eps)
        step_f = float(step_size)
        if eps_f <= 0:
            raise ValueError(f"--eps must be positive for L2, got {eps_f}")
        if step_f <= 0:
            raise ValueError(f"--pgd_step_size must be positive for L2, got {step_f}")
        return eps_f, step_f
    raise ValueError(f"Unsupported threat_model {threat_model!r} (use Linf or L2)")


def _l2_ball_project(delta: Tensor, eps: float) -> Tensor:
    flat = delta.view(delta.size(0), -1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    factor = torch.min(torch.ones_like(norms), eps / norms)
    return (flat * factor).view_as(delta)


def pgd_l2_attack(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool,
) -> Tensor:
    x_adv = x.detach().clone()
    if random_start:
        delta = torch.randn_like(x_adv)
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = delta_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
        scale = torch.empty(delta.size(0), 1, device=x_adv.device).uniform_(0.0, 1.0)
        delta = (delta_flat / delta_norm * scale * eps).view_as(delta)
        x_adv = (x + delta).clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()
        grad = x_adv.grad
        if grad is None:
            raise RuntimeError("PGD failed: input gradient is None")
        with torch.no_grad():
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_adv = x_adv + step_size * (grad_flat / grad_norm).view_as(grad)
            x_adv = (x + _l2_ball_project(x_adv - x, eps)).clamp(0.0, 1.0)
    return x_adv.detach()


def pgd_linf_attack(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    *,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool,
) -> Tensor:
    x_adv = x.detach().clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y, reduction="mean")
        loss.backward()
        grad = x_adv.grad
        if grad is None:
            raise RuntimeError("PGD failed: input gradient is None")
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = x_adv.clamp(0.0, 1.0)
    return x_adv.detach()


def per_image_clean_correct(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    flags: List[np.ndarray] = []
    model.eval()
    for xb, yb in zip(chunk_tensor(x, batch_size), chunk_tensor(y, batch_size)):
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)
        if yb.device != device:
            yb = yb.to(device, non_blocking=True).long()
        else:
            yb = yb.long()
        with torch.inference_mode():
            preds = model(xb).argmax(dim=1)
            flags.append((preds == yb).cpu().numpy())
    return np.concatenate(flags).astype(bool)


def per_image_robust_correct(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
    device: torch.device,
    batch_size: int,
    *,
    threat_model: str,
    eps: float,
    step_size: float,
    steps: int,
    random_start: bool,
) -> np.ndarray:
    norm = threat_model.strip().lower()
    attack_fn = pgd_l2_attack if norm == "l2" else pgd_linf_attack
    flags: List[np.ndarray] = []
    model.eval()
    for xb, yb in zip(chunk_tensor(x, batch_size), chunk_tensor(y, batch_size)):
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)
        if yb.device != device:
            yb = yb.to(device, non_blocking=True).long()
        else:
            yb = yb.long()
        x_adv = attack_fn(
            model,
            xb,
            yb,
            eps=eps,
            step_size=step_size,
            steps=steps,
            random_start=random_start,
        )
        with torch.inference_mode():
            preds = model(x_adv).argmax(dim=1)
            flags.append((preds == yb).cpu().numpy())
    return np.concatenate(flags).astype(bool)


def model_param_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def devices_match(a: torch.device, b: torch.device) -> bool:
    """Treat torch.device('cuda') and torch.device('cuda:0') as equivalent."""
    if a.type != b.type:
        return False
    if a.type == "cuda":
        return (a.index or 0) == (b.index or 0)
    return a == b


class ReferenceBuildLogger:
    def __init__(self) -> None:
        self._write(f"started_at={dt.datetime.now().isoformat(timespec='seconds')}")

    def _write(self, line: str) -> None:
        print(line, flush=True)

    def log_args(self, args: argparse.Namespace) -> None:
        self._write("args:")
        for key, value in sorted(vars(args).items()):
            self._write(f"  {key}={value!r}")

    def log_section(self, title: str) -> None:
        self._write("")
        self._write(title)

    def log_line(self, line: str) -> None:
        self._write(line)

    def close(self, *, started_at: float) -> None:
        elapsed = time.perf_counter() - started_at
        self._write("")
        self._write(f"finished_at={dt.datetime.now().isoformat(timespec='seconds')}")
        self._write(f"elapsed_sec={elapsed:.1f}")


def sort_entries_by_robust(
    entries: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    return sorted(
        entries,
        key=lambda e: (
            float(e["robust_acc"]),
            float(e["clean_acc"]),
            str(e["model_id"]),
        ),
        reverse=True,
    )


def write_results_xml(out_xml: str, entries: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_xml)) or ".", exist_ok=True)
    write_reference_xml(out_xml, sort_entries_by_robust(entries))
