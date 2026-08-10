"""
FARE-style robust CLIP visual encoder via OpenCLIP ViT.
Checkpoint must be compatible with the OpenCLIP visual trunk after removing the last transformer block.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_openclip_arch(vision_arch: str) -> tuple[str, str]:
    va = vision_arch.strip()
    mapping = {
        "ViT-L-14": ("ViT-L-14", "openai"),
        "ViT-L-14-336": ("ViT-L-14-336", "openai"),
        "ViT-B-32": ("ViT-B-32", "openai"),
        "ViT-B-16": ("ViT-B-16", "openai"),
    }
    if va not in mapping:
        raise ValueError(
            f"Unsupported vision_arch for FARE/OpenCLIP loader: {vision_arch!r}. "
            f"Supported: {list(mapping)}."
        )
    return mapping[va]


def _checkpoint_state_dict(checkpoint: str | Path) -> dict:
    sd = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        return sd["state_dict"]
    if isinstance(sd, dict):
        return sd
    raise RuntimeError(f"Unexpected checkpoint type in {checkpoint}: {type(sd)}")


def _is_full_openclip_state(state: dict) -> bool:
    keys = state.keys()
    has_vision = any(k.startswith("visual.") for k in keys)
    has_text = "token_embedding.weight" in keys or any(
        k.startswith("transformer.resblocks.") for k in keys
    )
    return has_vision and has_text


def _extract_visual_state_dict(state: dict) -> dict:
    """Keep only vision-tower weights (avoid loading text transformer resblocks)."""
    if any(k.startswith("visual.") for k in state):
        return {k[len("visual.") :]: v for k, v in state.items() if k.startswith("visual.")}
    stripped: dict = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model.", "visual.", "model.visual."):
            while nk.startswith(pref):
                nk = nk[len(pref) :]
        stripped[nk] = v
    return stripped


def _prepare_visual_trunk(visual: nn.Module) -> nn.Module:
    if hasattr(visual, "proj") and visual.proj is not None:
        visual.proj = None
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
        if len(blocks) < 2:
            raise RuntimeError("OpenCLIP ViT has too few blocks to drop the last one.")
        visual.transformer.resblocks = blocks[:-1]
    return visual


def _openclip_model_name_for_resolution(vision_arch: str, clip_input_resolution: int | None) -> tuple[str, str]:
    """Use ViT-L-14-336 when LLaVA feeds ~336px crops so patch grid matches positional embeddings."""
    model_name, pretrained = _parse_openclip_arch(vision_arch)
    if clip_input_resolution is None:
        return model_name, pretrained
    if vision_arch.strip() == "ViT-L-14" and clip_input_resolution >= 320:
        return "ViT-L-14-336", pretrained
    return model_name, pretrained


def _resolve_openclip_arch_name(
    vision_arch: str,
    openclip_arch: str | None,
) -> str:
    """Native OpenCLIP arch for checkpoint load (no ViT-L-14-336 upgrade; PE is resized after load)."""
    if openclip_arch is not None:
        return openclip_arch.strip()
    base, _ = _parse_openclip_arch(vision_arch)
    return base


def _load_visual_checkpoint(
    visual: nn.Module,
    state: dict,
    clip_input_resolution: int | None,
) -> None:
    sd = _extract_visual_state_dict(state)
    pe_key = "positional_embedding"
    if pe_key in sd and sd[pe_key].shape[0] != visual.positional_embedding.shape[0]:
        if clip_input_resolution is None:
            raise RuntimeError(
                f"Checkpoint positional_embedding length {sd[pe_key].shape[0]} does not match OpenCLIP visual "
                f"({visual.positional_embedding.shape[0]}). Pass clip_input_resolution (LLaVA crop size) when "
                "using encoder_replace with 336px LLaVA and 224px RobustVLM/FARE weights."
            )
        new_side = max(1, clip_input_resolution // _patch_size(visual))
        sd = {**sd, pe_key: _interpolate_openclip_positional_embedding(sd[pe_key], new_side)}
    visual.load_state_dict(sd, strict=False)


def _visual_pos_dim_from_state(state: dict) -> int | None:
    pos = state.get("visual.positional_embedding")
    if pos is not None:
        return int(pos.shape[-1])
    pos = state.get("positional_embedding")
    if pos is not None and int(pos.shape[0]) > 100:
        return int(pos.shape[-1])
    return None


def _validate_checkpoint_arch(state: dict, arch_name: str, ckpt: Path) -> None:
    dim = _visual_pos_dim_from_state(state)
    if dim is None:
        return
    if arch_name.startswith("ViT-B") and dim >= 1024:
        raise RuntimeError(
            f"{ckpt} looks like ViT-L/14 weights (hidden={dim}), not {arch_name}. "
            "Delete the file and re-run: bash scripts/download_vlm_weights.sh"
        )
    if arch_name.startswith("ViT-L") and dim <= 768:
        raise RuntimeError(
            f"{ckpt} looks like ViT-B/16 weights (hidden={dim}), not {arch_name}. "
            "Delete the file and re-run: bash scripts/download_vlm_weights.sh"
        )


def _patch_size(visual: nn.Module) -> int:
    ps = getattr(visual, "patch_size", 14)
    return int(ps[0] if isinstance(ps, (tuple, list)) else ps)


def _visual_token_dim(visual: nn.Module) -> int:
    """Transformer hidden size per token (for mm_projector), not CLIP proj output_dim."""
    width = getattr(visual, "width", None)
    if width is not None:
        return int(width)
    pe = getattr(visual, "positional_embedding", None)
    if pe is not None:
        return int(pe.shape[-1])
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
        if len(blocks) > 0 and hasattr(blocks[0], "ln_1"):
            return int(blocks[0].ln_1.weight.shape[0])
    embed_dim = getattr(visual, "embed_dim", None)
    if embed_dim is not None:
        return int(embed_dim)
    raise RuntimeError(
        f"Cannot infer visual token dim from {type(visual).__name__}. "
        "OpenCLIP visual trunk layout not recognized."
    )


def _resize_visual_positional_embedding(
    visual: nn.Module,
    clip_input_resolution: int | None,
) -> None:
    """LLaVA uses ~336px crops; robust ViT-L/14 checkpoints are often 224px."""
    if clip_input_resolution is None:
        return
    pe = visual.positional_embedding
    patch = _patch_size(visual)
    new_side = max(1, clip_input_resolution // patch)
    expected = new_side * new_side + 1
    if pe.shape[0] == expected:
        return
    new_pe = _interpolate_openclip_positional_embedding(pe.detach().clone(), new_side)
    visual.positional_embedding = nn.Parameter(new_pe.to(device=pe.device, dtype=pe.dtype))


def _interpolate_openclip_positional_embedding(pos_embed: torch.Tensor, new_side: int) -> torch.Tensor:
    """Resize CLS + spatial PE from a sqrt(N)-sided grid to new_side x new_side (OpenCLIP ViT layout)."""
    ntok, c = pos_embed.shape
    if ntok == new_side * new_side + 1:
        return pos_embed
    cls = pos_embed[0:1]
    patches = pos_embed[1:]
    old_side = int(round(math.sqrt(patches.shape[0])))
    if old_side * old_side != patches.shape[0]:
        raise ValueError(f"Cannot reshape positional_embedding length {patches.shape[0]} to a square grid.")
    x = patches.reshape(old_side, old_side, c).permute(2, 0, 1).unsqueeze(0)
    y = F.interpolate(x.float(), size=(new_side, new_side), mode="bicubic", align_corners=False).to(dtype=pos_embed.dtype)
    y = y.squeeze(0).permute(1, 2, 0).reshape(new_side * new_side, c)
    return torch.cat([cls, y], dim=0)


class OpenClipTokenVisual(nn.Module):
    """
    Wraps an OpenCLIP ViT `visual` with the last residual block removed (Robust-LLaVA convention)
    and returns patch(+cls) tokens [B, N, C].
    """

    def __init__(self, visual: nn.Module, token_dim: int):
        super().__init__()
        self.visual = visual
        self.token_dim = int(token_dim)

    def forward(self, images_normalized: torch.Tensor) -> torch.Tensor:
        if hasattr(self.visual, "output_tokens"):
            self.visual.output_tokens = True
        out = self.visual(images_normalized)
        if isinstance(out, tuple):
            a, b = out[0], out[1]
            flat = torch.cat([a.flatten(1), b.flatten(1)], dim=1)
        else:
            flat = out if out.dim() == 2 else out.flatten(1)
        bsz, fd = flat.shape
        if fd % self.token_dim != 0:
            raise RuntimeError(
                f"Encoder output feature dim {fd} is not divisible by token_dim={self.token_dim}. "
                "Checkpoint / architecture mismatch."
            )
        ntok = fd // self.token_dim
        return flat.view(bsz, ntok, self.token_dim)

    @property
    def hidden_size(self) -> int:
        return self.token_dim


def build_fare_visual(
    vision_arch: str,
    checkpoint: str | Path,
    device: str | torch.device,
    dtype: torch.dtype = torch.float16,
    clip_input_resolution: int | None = None,
    *,
    openclip_arch: str | None = None,
    openclip_pretrained: str | None = None,
) -> OpenClipTokenVisual:
    import open_clip

    ckpt = Path(checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"FARE encoder checkpoint not found: {ckpt}. "
            "Provide a valid encoder_checkpoint trained for the chosen OpenCLIP architecture."
        )
    state = _checkpoint_state_dict(ckpt)
    arch_name = _resolve_openclip_arch_name(vision_arch, openclip_arch)
    _validate_checkpoint_arch(state, arch_name, ckpt)

    if _is_full_openclip_state(state):
        try:
            model, _, _ = open_clip.create_model_and_transforms(
                arch_name,
                pretrained=str(ckpt.resolve()),
            )
            visual = _prepare_visual_trunk(model.visual)
            _resize_visual_positional_embedding(visual, clip_input_resolution)
            token_dim = _visual_token_dim(visual)
            enc = OpenClipTokenVisual(visual, token_dim)
            return enc.to(device=device, dtype=dtype)
        except Exception:
            pass

    pretrained = openclip_pretrained
    if pretrained is None:
        _, pretrained = _openclip_model_name_for_resolution(vision_arch, clip_input_resolution)
    kwargs = {}
    if pretrained == "openai":
        kwargs["force_quick_gelu"] = True
    try:
        model, _, _ = open_clip.create_model_and_transforms(arch_name, pretrained=pretrained, **kwargs)
    except TypeError:
        model, _, _ = open_clip.create_model_and_transforms(arch_name, pretrained=pretrained)

    visual = _prepare_visual_trunk(model.visual)
    _load_visual_checkpoint(visual, state, clip_input_resolution)
    _resize_visual_positional_embedding(visual, clip_input_resolution)
    token_dim = _visual_token_dim(visual)
    enc = OpenClipTokenVisual(visual, token_dim)
    enc = enc.to(device=device, dtype=dtype)
    return enc
