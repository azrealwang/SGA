from __future__ import annotations

from types import MethodType
from typing import Any, Callable

import torch

from vlm.models.base import BaseRobustVQAModel
from vlm.models.llava_bridge import ensure_llava_on_path
from vlm.prompts import format_question
from vlm.utils.image import llava_style_preprocess


class RobustLLaVAWrapper(BaseRobustVQAModel):
    """
    LLaVA / Robust-LLaVA via `llava.model.builder.load_pretrained_model`.

    Use ``robust: true`` in YAML for Robust-LLaVA-style checkpoints; ``robust: false`` and
    ``type: llava`` for a standard merged LLaVA VQA model (same code path, same ``llava`` backend).
    """

    def __init__(self, name: str, cfg: dict[str, Any]):
        super().__init__()
        ensure_llava_on_path()
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IGNORE_INDEX,
        )
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        from llava.model.builder import load_pretrained_model

        self.name = name
        self.robust_encoder_name = cfg.get(
            "robust_encoder_name",
            "robust_llava_default" if cfg.get("robust", True) else "standard_llava",
        )
        self.backend_name = cfg.get("backend_name", "llava_llama")
        self.attackable = bool(cfg.get("attackable", True))

        self._IGNORE_INDEX = IGNORE_INDEX
        self._conv_mode = cfg.get("conv_mode", "llava_v1")
        self._prompt_mode = cfg.get("prompt_mode", "short_answer")
        self._device = cfg.get("device", "cuda")
        model_path = cfg["model_path"]
        model_base = cfg.get("model_base")
        model_name = get_model_name_from_path(model_path)
        load_kwargs: dict[str, Any] = {}
        load_enc = cfg.get("load_encoder")
        if load_enc is not None:
            # Robust-LLaVA forks only; standard haotian-liu/LLaVA must not pass this to from_pretrained.
            load_kwargs["load_encoder"] = load_enc
        self.tokenizer, self.model, self.image_processor, _ctx = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            device_map={"": self._device} if self._device != "cuda" else "auto",
            device=self._device,
            **load_kwargs,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self._image_aspect_ratio = getattr(self.model.config, "image_aspect_ratio", None)
        ip = self.image_processor
        self._shortest = ip.size.get("shortest_edge", 224)
        self._crop = ip.crop_size.get("height", ip.crop_size.get("width", 224))
        mean = torch.tensor(ip.image_mean, dtype=torch.float32)
        std = torch.tensor(ip.image_std, dtype=torch.float32)
        self.register_buffer("_clip_mean", mean, persistent=False)
        self.register_buffer("_clip_std", std, persistent=False)
        self._mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)

    def preprocess_for_model(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError("images must be [B,3,H,W] in [0,1]")
        dev = images.device
        mean, std = self._clip_mean.to(dev), self._clip_std.to(dev)
        return llava_style_preprocess(
            images,
            self._image_aspect_ratio,
            self._shortest,
            self._crop,
            mean,
            std,
            pad_fill=float(mean[0].item()),
        ).to(dtype=self.model.dtype)

    def _encode_images_for_loss(self, images: torch.Tensor) -> torch.Tensor:
        """
        Same as LLaVA ``encode_images``, but without CLIPVisionTower's ``@torch.no_grad``
        so input-gradient attacks / AdvSM can backprop to ``images``.
        """
        inner = self.model.get_model()
        tower = inner.get_vision_tower()
        if getattr(tower, "encoder", None) is not None:
            feats = tower(images)
        elif hasattr(tower, "vision_tower") and hasattr(tower, "feature_select"):
            px = images.to(device=tower.device, dtype=tower.dtype)
            outs = tower.vision_tower(px, output_hidden_states=True)
            feats = tower.feature_select(outs).to(images.dtype)
        else:
            feats = tower(images)
        return inner.mm_projector(feats)

    def _wrap_question(self, question: str) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

        q = format_question(question, self._prompt_mode)
        if self._mm_use_im_start_end:
            return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + q
        return DEFAULT_IMAGE_TOKEN + "\n" + q

    def _prompt_ids(self, question: str, with_answer: str | None) -> torch.Tensor:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        qs = self._wrap_question(question)
        conv = conv_templates[self._conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], with_answer)
        prompt = conv.get_prompt()
        return tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    def generate_answer(self, images: torch.Tensor, questions: list[str], max_new_tokens: int = 16) -> list[str]:
        self.model.eval()
        outs: list[str] = []
        pixel = self.preprocess_for_model(images.to(self._device)).detach()
        with torch.inference_mode():
            for i in range(len(questions)):
                input_ids = self._prompt_ids(questions[i], None).unsqueeze(0).to(self._device)
                img = pixel[i : i + 1].to(self._device)
                image_sizes = [(self._crop, self._crop)]
                gen = self.model.generate(
                    input_ids,
                    images=img,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0.0,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
                text = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
                outs.append(text)
        return outs

    def compute_nll_loss(
        self, images: torch.Tensor, questions: list[str], answers: list[str]
    ) -> torch.Tensor:
        self.model.eval()
        images = images.to(self._device)
        pixel = self.preprocess_for_model(images)
        B = len(questions)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        seqs: list[torch.Tensor] = []
        labs: list[torch.Tensor] = []
        for i in range(B):
            ans = answers[i].strip()
            ids_full = self._prompt_ids(questions[i], ans).to(self._device)
            ids_prefix = self._prompt_ids(questions[i], None).to(self._device)
            if ids_full.shape[0] >= ids_prefix.shape[0] and torch.equal(
                ids_full[: ids_prefix.shape[0]], ids_prefix
            ):
                prefix_len = int(ids_prefix.shape[0])
            else:
                prefix_len = 0
                for a, b in zip(ids_prefix.tolist(), ids_full.tolist()):
                    if a != b:
                        break
                    prefix_len += 1
            labels = torch.full_like(ids_full, self._IGNORE_INDEX)
            labels[prefix_len:] = ids_full[prefix_len:]
            seqs.append(ids_full)
            labs.append(labels)

        max_len = max(s.numel() for s in seqs)
        input_ids = torch.full((B, max_len), pad_id, device=self._device, dtype=torch.long)
        labels = torch.full((B, max_len), self._IGNORE_INDEX, device=self._device, dtype=torch.long)
        attn = torch.zeros(B, max_len, device=self._device, dtype=torch.bool)
        for i in range(B):
            L = seqs[i].numel()
            input_ids[i, :L] = seqs[i].to(self._device)
            labels[i, :L] = labs[i].to(self._device)
            attn[i, :L] = True

        pixel = pixel.to(self._device)
        need_image_grad = bool(images.requires_grad)
        orig_encode: Callable | None = None
        if need_image_grad:
            orig_encode = self.model.encode_images
            self.model.encode_images = MethodType(
                lambda _m, imgs: self._encode_images_for_loss(imgs),
                self.model,
            )

        try:
            with torch.set_grad_enabled(True):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    images=pixel,
                    image_sizes=[(self._crop, self._crop)] * B,
                    labels=labels,
                    return_dict=True,
                )
        finally:
            if orig_encode is not None:
                self.model.encode_images = orig_encode

        if out.loss is None:
            raise RuntimeError("Model returned no loss; check labels / multimodal inputs.")
        if need_image_grad and not out.loss.requires_grad:
            raise RuntimeError(
                "NLL loss is not connected to input images; vision tower may block gradients."
            )
        return out.loss
