from __future__ import annotations

import torch.nn as nn

from vlm.models.fare_encoder import build_fare_visual
from vlm.models.simclip_encoder import build_simclip_visual
from vlm.models.tecoa_encoder import build_tecoa_visual


def build_robust_encoder(
    encoder_type: str,
    checkpoint: str,
    vision_arch: str,
    device: str,
    dtype=None,
    clip_input_resolution: int | None = None,
    *,
    openclip_arch: str | None = None,
    openclip_pretrained: str | None = None,
) -> nn.Module:
    """
    Factory for robust vision encoders used inside LLaVA-style pipelines.
    All encoders expect CLIP-normalized image tensors (same convention as this repo's CLIP tower).
    """
    enc = encoder_type.lower().strip()
    if dtype is None:
        dtype = torch.float16
    common = dict(
        vision_arch=vision_arch,
        checkpoint=checkpoint,
        device=device,
        dtype=dtype,
        clip_input_resolution=clip_input_resolution,
        openclip_arch=openclip_arch,
        openclip_pretrained=openclip_pretrained,
    )
    if enc == "fare":
        return build_fare_visual(**common)
    if enc == "tecoa":
        return build_tecoa_visual(**common)
    if enc == "simclip":
        return build_simclip_visual(**common)
    if enc == "leaf":
        return build_fare_visual(**common)
    raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Expected fare|tecoa|simclip|leaf.")
