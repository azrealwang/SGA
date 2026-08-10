from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vlm.models.robust_encoder_loader import build_robust_encoder
from vlm.models.robust_llava import RobustLLaVAWrapper


def _strip_prefix_state_dict(sd: dict) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        nk = k
        for pref in ("model.", "base_model.model.", "module."):
            while nk.startswith(pref):
                nk = nk[len(pref) :]
        out[nk] = v
    return out


def _parse_mm_projector_state(raw_sd: dict) -> dict[str, torch.Tensor]:
    mm_sd: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if "mm_projector" in k:
            suffix = k.split("mm_projector.", 1)[-1]
            mm_sd[suffix] = v
        elif k.startswith(("0.", "1.", "2.", "3.")):
            mm_sd[k] = v
    return mm_sd


def _build_mm_projector_from_state(mm_sd: dict[str, torch.Tensor]) -> nn.Sequential:
    w0 = mm_sd.get("0.weight")
    w2 = mm_sd.get("2.weight")
    if w0 is None or w2 is None:
        raise RuntimeError(
            "mm_projector checkpoint must contain 0.weight and 2.weight "
            f"(LLaVA mlp2x_gelu). Got keys: {sorted(mm_sd)[:12]}."
        )
    in_features = int(w0.shape[1])
    hidden = int(w0.shape[0])
    out_features = int(w2.shape[0])
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_features),
    )


def load_mm_projector_weights(model: nn.Module, projector_path: str | Path) -> None:
    path = Path(projector_path)
    if path.is_file():
        ckpt_path = path
    elif (path / "mm_projector.bin").is_file():
        ckpt_path = path / "mm_projector.bin"
    else:
        raise FileNotFoundError(
            f"projector_path must be a weight file or a directory containing mm_projector.bin. Got: {path}"
        )
    raw = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected projector checkpoint format (expected dict): {ckpt_path}")
    sd = _strip_prefix_state_dict(raw)
    mm_sd = _parse_mm_projector_state(sd)
    if not mm_sd:
        raise RuntimeError(
            f"No mm_projector weights in {ckpt_path}. Provide a Robust-LLaVA-compatible mm_projector.bin."
        )

    inner = model.get_model()
    in_features = int(mm_sd["0.weight"].shape[1])
    current_in = None
    try:
        current_in = int(inner.mm_projector[0].in_features)
    except Exception:
        current_in = None
    if current_in != in_features:
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        inner.mm_projector = _build_mm_projector_from_state(mm_sd).to(device=device, dtype=dtype)

    tgt = inner.mm_projector
    missing, _ = tgt.load_state_dict(mm_sd, strict=False)
    tgt_keys = set(tgt.state_dict().keys())
    if missing and len(missing) == len(tgt_keys):
        raise RuntimeError(
            f"mm_projector keys did not match. Missing all projector params. "
            f"Checkpoint keys sample: {list(mm_sd.keys())[:8]}."
        )


class ReplacedCLIPVisionTower(nn.Module):
    """Replacement vision tower: CLIP-normalized images -> patch tokens for mm_projector."""

    def __init__(self, encoder: nn.Module, image_processor, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.image_processor = image_processor
        self.is_loaded = True
        self.vision_tower_name = "robust_encoder_replacement"
        self.load_encoder = None
        self._hidden_size = int(hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images.to(dtype=self.dtype))

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def dtype(self):
        return next(self.encoder.parameters()).dtype

    @property
    def device(self):
        return next(self.encoder.parameters()).device

    @property
    def config(self):
        class C:
            image_size = int(getattr(self.image_processor, "crop_size", {}).get("height", 224))
            patch_size = 14

        return C()

    @property
    def num_patches_per_side(self) -> int:
        return max(1, self.config.image_size // self.config.patch_size)


class LlavaEncoderReplaceWrapper(RobustLLaVAWrapper):
    def __init__(self, name: str, cfg: dict[str, Any]):
        if not cfg.get("robust", True):
            raise ValueError(f"Model {name} must have robust=true.")
        if not cfg.get("projector_path"):
            raise ValueError(
                f"Model {name} (llava_encoder_replace) requires projector_path (mm_projector.bin or file)."
            )
        cfg_load = {
            **cfg,
            "model_path": cfg["backend_model_path"],
            "model_base": cfg.get("model_base"),
            "conv_mode": cfg.get("conv_mode", "llava_v1"),
            "device": cfg.get("device", "cuda"),
            "robust": True,
            "attackable": cfg.get("attackable", True),
            "prompt_mode": cfg.get("prompt_mode", "short_answer"),
        }
        if cfg.get("load_encoder") is not None:
            cfg_load["load_encoder"] = cfg["load_encoder"]
        super().__init__(name, cfg_load)
        self.robust_encoder_name = str(cfg.get("encoder_type", "unknown"))
        self.backend_name = cfg.get("backend_name", "llava_llama")

        load_mm_projector_weights(self.model, cfg["projector_path"])

        dtype = next(self.model.parameters()).dtype
        device = cfg_load["device"]
        enc = build_robust_encoder(
            cfg["encoder_type"],
            cfg["encoder_checkpoint"],
            cfg["vision_arch"],
            device,
            dtype=dtype,
            clip_input_resolution=int(self._crop),
            openclip_arch=cfg.get("openclip_arch"),
            openclip_pretrained=cfg.get("openclip_pretrained"),
        )
        proj = self.model.get_model().mm_projector[0]
        expected_in = proj.in_features
        hidden = int(enc.hidden_size)
        if hidden != expected_in:
            raise RuntimeError(
                f"Encoder feature dimension {hidden} does not match mm_projector.in_features={expected_in}. "
                "Use a compatible projector checkpoint for this encoder."
            )
        tower = ReplacedCLIPVisionTower(enc, self.image_processor, hidden)
        self.model.get_model().vision_tower = tower

        with torch.no_grad():
            side = max(224, int(self._crop))
            dummy = torch.rand(1, 3, side, side, device=device, dtype=torch.float32).clamp(0, 1)
            norm = self.preprocess_for_model(dummy)
            feat = tower(norm)
            if feat.shape[-1] != expected_in:
                raise RuntimeError(
                    f"Forward check failed: encoder output dim {feat.shape[-1]} != projector {expected_in}."
                )
