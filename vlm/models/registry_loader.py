from __future__ import annotations

from typing import Any

from vlm.model_registry import build_vqa_model_config
from vlm.models.base import BaseRobustVQAModel
from vlm.models.llava_encoder_replace import LlavaEncoderReplaceWrapper
from vlm.models.robust_llava import RobustLLaVAWrapper


def load_vqa_model_by_id(model_id: str, *, device: str = "cuda") -> BaseRobustVQAModel:
    """Load LVLM VQA model using GT model id + ``checkpoints/vlm_weights`` encoders."""
    cfg = build_vqa_model_config(model_id, device=device)
    return _load_from_cfg(model_id, cfg)


def _load_from_cfg(model_key: str, mcfg: dict[str, Any]) -> BaseRobustVQAModel:
    typ = mcfg.get("type", "robust_llava")
    if typ in ("robust_llava", "llava"):
        return RobustLLaVAWrapper(model_key, mcfg)
    if typ == "llava_encoder_replace":
        return LlavaEncoderReplaceWrapper(model_key, mcfg)
    raise ValueError(f"Unknown model type: {typ!r}")
