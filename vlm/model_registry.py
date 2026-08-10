"""
VLM encoder specs shared by ImageNet/CelebA zero-shot ranking (utils.load_one_model)
and LVLM VQA GT (build_gt_pgd_vqa).

Encoder weights: ``checkpoints/vlm_weights/`` (``scripts/download_vlm_weights.sh``).
LLaVA backend: ``checkpoints/llava-v1.5-7b/``.

VQA supports ViT-L/14 encoders only (stock LLaVA mm_projector). ViT-B/16 RCLIP models
remain in ``utils.load_one_model`` for ImageNet zero-shot ranking only.
No public LLaVA mm_projector exists for ConvNeXt-B in this repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root() -> str:
    return _REPO_ROOT


def vlm_weights_dir() -> str:
    return os.path.join(_REPO_ROOT, "checkpoints", "vlm_weights")


def llava_dir() -> str:
    return os.path.join(_REPO_ROOT, "checkpoints", "llava-v1.5-7b")


# mm_projector.bin paths relative to repo root (ViT-L/14 VQA only).
MM_PROJECTOR_BY_VISION_FAMILY: Dict[str, str] = {
    "ViT-L-14": "checkpoints/llava-v1.5-7b/mm_projector.bin",
}


@dataclass(frozen=True)
class VLMEncoderSpec:
    model_id: str
    weight_filename: Optional[str]
    openclip_arch: Optional[str]
    openclip_pretrained: Optional[str]
    encoder_type: Optional[str]  # fare | tecoa | simclip | leaf
    vision_arch: Optional[str]
    vision_family: Optional[str]  # key into MM_PROJECTOR_BY_VISION_FAMILY
    vqa_type: str  # llava | llava_encoder_replace


def _w(name: str) -> str:
    return os.path.join(vlm_weights_dir(), name)


VLM_ENCODER_SPECS: Dict[str, VLMEncoderSpec] = {
    "ViT-L-14_TeCoA4": VLMEncoderSpec(
        model_id="ViT-L-14_TeCoA4",
        weight_filename="tecoa_eps_4.pt",
        openclip_arch="ViT-L-14-quickgelu",
        openclip_pretrained=None,
        encoder_type="tecoa",
        vision_arch="ViT-L-14",
        vision_family="ViT-L-14",
        vqa_type="llava_encoder_replace",
    ),
    "ViT-L-14_SimCLIP4": VLMEncoderSpec(
        model_id="ViT-L-14_SimCLIP4",
        weight_filename="simclip4.pt",
        openclip_arch="ViT-L-14-quickgelu",
        openclip_pretrained="openai",
        encoder_type="simclip",
        vision_arch="ViT-L-14",
        vision_family="ViT-L-14",
        vqa_type="llava_encoder_replace",
    ),
    "ViT-L-14_FARE4": VLMEncoderSpec(
        model_id="ViT-L-14_FARE4",
        weight_filename="fare_eps_4.pt",
        openclip_arch="ViT-L-14-quickgelu",
        openclip_pretrained=None,
        encoder_type="fare",
        vision_arch="ViT-L-14",
        vision_family="ViT-L-14",
        vqa_type="llava_encoder_replace",
    ),
    "ConvNeXt-B_RCLIP-TeCoA": VLMEncoderSpec(
        model_id="ConvNeXt-B_RCLIP-TeCoA",
        weight_filename="convnext_base_w-tecoa-eps4.pt",
        openclip_arch="convnext_base_w",
        openclip_pretrained="laion2b_s13b_b82k_augreg",
        encoder_type="tecoa",
        vision_arch="convnext_base_w",
        vision_family=None,
        vqa_type="llava_encoder_replace",
    ),
    "ConvNeXt-B_RCLIP-FARE": VLMEncoderSpec(
        model_id="ConvNeXt-B_RCLIP-FARE",
        weight_filename="convnext_base_w-fare-eps4.pt",
        openclip_arch="convnext_base_w",
        openclip_pretrained="laion2b_s13b_b82k_augreg",
        encoder_type="fare",
        vision_arch="convnext_base_w",
        vision_family=None,
        vqa_type="llava_encoder_replace",
    ),
    "ViT-L-14_LEAF-CLIP": VLMEncoderSpec(
        model_id="ViT-L-14_LEAF-CLIP",
        weight_filename="leaf_vit_l.pt",
        openclip_arch="ViT-L-14-quickgelu",
        openclip_pretrained=None,
        encoder_type="leaf",
        vision_arch="ViT-L-14",
        vision_family="ViT-L-14",
        vqa_type="llava_encoder_replace",
    ),
    "ViT-L-14_CLIP": VLMEncoderSpec(
        model_id="ViT-L-14_CLIP",
        weight_filename=None,
        openclip_arch="ViT-L-14",
        openclip_pretrained="openai",
        encoder_type=None,
        vision_arch=None,
        vision_family="ViT-L-14",
        vqa_type="llava",
    ),
}


def get_encoder_spec(model_id: str) -> VLMEncoderSpec:
    if model_id not in VLM_ENCODER_SPECS:
        raise ValueError(
            f"Unknown VLM model id {model_id!r}. "
            f"Expected one of: {sorted(VLM_ENCODER_SPECS)}"
        )
    return VLM_ENCODER_SPECS[model_id]


def encoder_checkpoint_path(spec: VLMEncoderSpec) -> Optional[str]:
    if spec.weight_filename is None:
        return None
    return _w(spec.weight_filename)


def resolve_mm_projector_path(spec: VLMEncoderSpec) -> Optional[str]:
    if spec.vqa_type == "llava":
        return os.path.join(repo_root(), MM_PROJECTOR_BY_VISION_FAMILY["ViT-L-14"])
    if spec.vision_family is None:
        return None
    rel = MM_PROJECTOR_BY_VISION_FAMILY.get(spec.vision_family)
    if rel is None:
        return None
    return os.path.join(repo_root(), rel)


def vqa_load_error(spec: VLMEncoderSpec) -> Optional[str]:
    if spec.vision_family is None:
        return (
            f"No public LLaVA mm_projector for {spec.model_id} ({spec.vision_arch}). "
            "Use encoder-only ranking via load_one_model (ImageNet/CelebA zero-shot)."
        )
    proj = resolve_mm_projector_path(spec)
    if proj is None:
        return f"No projector mapping for vision family {spec.vision_family!r}."
    if not os.path.isfile(proj):
        return (
            f"Missing mm_projector: {proj}. "
            "Run: bash scripts/vlm/download_llava_v1_5_weights.sh"
        )
    if spec.vqa_type == "llava_encoder_replace":
        ckpt = encoder_checkpoint_path(spec)
        if ckpt is None or not os.path.isfile(ckpt):
            return f"Missing encoder weights: {ckpt}. Run: bash scripts/download_vlm_weights.sh"
    if not os.path.isdir(llava_dir()):
        return f"Missing LLaVA weights: {llava_dir()}. Run: bash scripts/vlm/download_llava_v1_5_weights.sh"
    return None


def is_vqa_loadable(model_id: str) -> bool:
    return vqa_load_error_for_id(model_id) is None


def vqa_load_error_for_id(model_id: str) -> Optional[str]:
    if model_id not in VLM_ENCODER_SPECS:
        return f"Unknown VLM model id {model_id!r} (not registered for VQA)."
    return vqa_load_error(get_encoder_spec(model_id))


def vqa_supported_model_ids() -> list[str]:
    return sorted(mid for mid in VLM_ENCODER_SPECS if is_vqa_loadable(mid))


def build_vqa_model_config(model_id: str, *, device: str = "cuda") -> Dict[str, Any]:
    spec = get_encoder_spec(model_id)
    err = vqa_load_error(spec)
    if err:
        raise ValueError(f"Cannot load {model_id!r} for LLaVA VQA: {err}")

    common = {
        "conv_mode": "llava_v1",
        "device": device,
        "attackable": True,
        "prompt_mode": "short_answer",
    }

    if spec.vqa_type == "llava":
        return {
            **common,
            "type": "llava",
            "robust": False,
            "model_path": llava_dir(),
            "model_base": None,
        }

    proj = resolve_mm_projector_path(spec)
    ckpt = encoder_checkpoint_path(spec)
    return {
        **common,
        "type": "llava_encoder_replace",
        "robust": True,
        "backend_model_path": llava_dir(),
        "model_base": None,
        "projector_path": proj,
        "encoder_type": spec.encoder_type,
        "encoder_checkpoint": ckpt,
        "vision_arch": spec.vision_arch,
        "openclip_arch": spec.openclip_arch,
        "openclip_pretrained": spec.openclip_pretrained,
    }
