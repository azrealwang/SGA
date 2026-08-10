"""
TeCoA robust encoder: loads an OpenCLIP-compatible ViT checkpoint.

If your TeCoA weights use a custom key layout from the TeCoA repository, convert them
or extend this loader. Unsupported checkpoints raise a clear error.
"""
from __future__ import annotations

from pathlib import Path

import torch.nn as nn

from vlm.models.fare_encoder import build_fare_visual as _build_openclip_tokens


def build_tecoa_visual(
    vision_arch: str,
    checkpoint: str | Path,
    device: str,
    dtype,
    clip_input_resolution: int | None = None,
    *,
    openclip_arch: str | None = None,
    openclip_pretrained: str | None = None,
) -> nn.Module:
    """
    TeCoA is distributed as robust CLIP-style ViT weights; we load into the same
    OpenCLIP trunk layout as FARE for a fixed VQA backend.
    """
    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"TeCoA encoder checkpoint not found: {ckpt}. "
            "Download TeCoA (ZSRobust4FoundationModel) ViT weights and set encoder_checkpoint."
        )
    try:
        return _build_openclip_tokens(
            vision_arch,
            ckpt,
            device,
            dtype=dtype,
            clip_input_resolution=clip_input_resolution,
            openclip_arch=openclip_arch,
            openclip_pretrained=openclip_pretrained,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to load TeCoA weights into the OpenCLIP visual trunk. "
            "Ensure vision_arch matches the TeCoA checkpoint (e.g. ViT-B-32) and that "
            "state_dict keys align with OpenCLIP after prefix stripping. Original error: "
            f"{e}"
        ) from e
