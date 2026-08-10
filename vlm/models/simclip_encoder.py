"""
SimCLIP robust encoder: OpenCLIP ViT trunk + external checkpoint (SimCLIP official format).

If the checkpoint is a full open_clip model dict, visual weights are extracted.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from vlm.models.fare_encoder import (
    OpenClipTokenVisual,
    _extract_visual_state_dict,
    _interpolate_openclip_positional_embedding,
    _openclip_model_name_for_resolution,
    _prepare_visual_trunk,
    _resize_visual_positional_embedding,
    _visual_token_dim,
)


def build_simclip_visual(
    vision_arch: str,
    checkpoint: str | Path,
    device: str,
    dtype,
    clip_input_resolution: int | None = None,
    *,
    openclip_arch: str | None = None,
    openclip_pretrained: str | None = None,
) -> nn.Module:
    import open_clip

    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"SimCLIP encoder checkpoint not found: {ckpt}. "
            "Provide a SimCLIP ViT-L-14 (or matching arch) weight file."
        )
    model_name, pretrained = _openclip_model_name_for_resolution(vision_arch, clip_input_resolution)
    if openclip_arch is not None:
        model_name = openclip_arch
    if openclip_pretrained is not None:
        pretrained = openclip_pretrained
    kwargs = {}
    if pretrained == "openai":
        kwargs["force_quick_gelu"] = True
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, **kwargs)
    except TypeError:
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    visual = _prepare_visual_trunk(model.visual)
    token_dim = _visual_token_dim(visual)
    sd = torch.load(str(ckpt), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("visual.") for k in sd):
        sd = {k[len("visual.") :]: v for k, v in sd.items() if k.startswith("visual.")}
    elif isinstance(sd, dict):
        sd = _extract_visual_state_dict(sd)
    pe_key = "positional_embedding"
    if pe_key in sd and sd[pe_key].shape[0] != visual.positional_embedding.shape[0]:
        if clip_input_resolution is None:
            raise RuntimeError(
                f"SimCLIP checkpoint positional_embedding length {sd[pe_key].shape[0]} does not match OpenCLIP "
                f"visual ({visual.positional_embedding.shape[0]}). Set clip_input_resolution to LLaVA crop size."
            )
        ps = getattr(visual, "patch_size", 14)
        patch = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
        new_side = max(1, clip_input_resolution // patch)
        sd = {**sd, pe_key: _interpolate_openclip_positional_embedding(sd[pe_key], new_side)}
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    _resize_visual_positional_embedding(visual, clip_input_resolution)
    if len(missing) > 24:
        raise RuntimeError(
            f"SimCLIP checkpoint does not match {vision_arch}: too many missing keys "
            f"(first 8): {missing[:8]}. Verify checkpoint format from the SimCLIP repository."
        )
    enc = OpenClipTokenVisual(visual, token_dim).to(device=device, dtype=dtype)
    return enc
