"""Madry Lab L2-robust ImageNet models (Hugging Face: madrylab/robust-imagenet-models)."""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Iterator, Tuple

import dill
import torch
import torch.nn as nn
import torchvision.models as tvm

HF_REPO = "madrylab/robust-imagenet-models"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

MADRY_L2_EPS: Tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

# arch_tag -> (robustness arch name, human-readable architecture)
# Matches reference/dnn_l2.xml (3 arch x 4 eps = 12 Madry checkpoints).
MADRY_L2_ARCH_TAGS: Dict[str, Tuple[str, str]] = {
    "R18": ("resnet18", "ResNet-18"),
    "R50": ("resnet50", "ResNet-50"),
    "50_2": ("wide_resnet50_2", "WideResNet-50-2"),
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _eps_ckpt_tag(eps: float) -> str:
    if eps == 1.0:
        return "1"
    return f"{eps:g}"


def _eps_model_tag(eps: float) -> str:
    return "1.0" if eps == 1.0 else f"{eps:g}"


def madry_l2_model_id(arch_tag: str, eps: float) -> str:
    return f"Salman2020Do_{arch_tag}_L2_eps{_eps_model_tag(eps)}"


def madry_l2_ckpt_filename(arch_name: str, eps: float) -> str:
    return f"{arch_name}_l2_eps{_eps_ckpt_tag(eps)}.ckpt"


def iter_madry_l2_specs() -> Iterator[Tuple[str, str, str, float]]:
    """Yield (model_id, arch_name, architecture, eps)."""
    for arch_tag, (arch_name, architecture) in MADRY_L2_ARCH_TAGS.items():
        for eps in MADRY_L2_EPS:
            yield madry_l2_model_id(arch_tag, eps), arch_name, architecture, eps


MADRY_L2_MODEL_IDS: frozenset[str] = frozenset(
    model_id for model_id, *_ in iter_madry_l2_specs()
)

_MODEL_LOOKUP: Dict[str, Tuple[str, float]] = {
    model_id: (arch_name, eps) for model_id, arch_name, _architecture, eps in iter_madry_l2_specs()
}


def is_madry_l2_model(model_id: str) -> bool:
    return model_id in MADRY_L2_MODEL_IDS


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def madry_l2_weight_dir() -> str:
    return os.path.join(_repo_root(), "checkpoints", "madry_imagenet_l2")


def _build_arch(arch_name: str, num_classes: int = 1000) -> nn.Module:
    if arch_name == "resnet18":
        return tvm.resnet18(num_classes=num_classes)
    if arch_name == "resnet50":
        return tvm.resnet50(num_classes=num_classes)
    if arch_name == "wide_resnet50_2":
        return tvm.wide_resnet50_2(num_classes=num_classes)
    raise ValueError(f"Unknown Madry L2 arch {arch_name!r}")


def _extract_classifier_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "module.model.",
        "module.attacker.model.",
        "attacker.model.",
        "model.",
    )
    for prefix in prefixes:
        sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            return sub
    stripped = {}
    for key, value in state.items():
        if "normalizer" in key or "normalize" in key:
            continue
        nk = key
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        stripped[nk] = value
    if not stripped:
        raise ValueError("No classifier weights found in Madry checkpoint")
    return stripped


def _madry_ckpt_path(arch_name: str, eps: float) -> str:
    return os.path.join(madry_l2_weight_dir(), madry_l2_ckpt_filename(arch_name, eps))


def ensure_madry_l2_weights(arch_name: str, eps: float) -> str:
    path = _madry_ckpt_path(arch_name, eps)
    if os.path.isfile(path):
        return path
    script = os.path.join(_repo_root(), "scripts", "download_madry_l2_weights.sh")
    subprocess.run(["bash", script], check=True, cwd=_repo_root())
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Madry L2 checkpoint missing after download: {path}")
    return path


def load_madry_l2_model(model_id: str) -> nn.Module:
    from common.utils import NormalizeWrapper

    if model_id not in _MODEL_LOOKUP:
        raise ValueError(f"Unknown Madry L2 model id: {model_id!r}")
    arch_name, eps = _MODEL_LOOKUP[model_id]
    ckpt_path = ensure_madry_l2_weights(arch_name, eps)
    checkpoint = torch.load(ckpt_path, map_location="cpu", pickle_module=dill)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint type in {ckpt_path}: {type(checkpoint)}")
    raw_state = checkpoint.get("model", checkpoint.get("state_dict"))
    if raw_state is None:
        raise ValueError(f"No model/state_dict in {ckpt_path}")
    state = _extract_classifier_state(raw_state)
    model = _build_arch(arch_name)
    model.load_state_dict(state, strict=True)
    return NormalizeWrapper(model, _IMAGENET_MEAN, _IMAGENET_STD).eval()
