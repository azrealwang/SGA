from __future__ import annotations

import abc
from typing import List

import torch
import torch.nn as nn


class BaseRobustVQAModel(nn.Module, abc.ABC):
    """Unified interface for robust VQA pipelines (Robust-LLaVA or encoder-replaced LLaVA)."""

    name: str
    robust_encoder_name: str
    backend_name: str
    attackable: bool

    @abc.abstractmethod
    def generate_answer(
        self, images: torch.Tensor, questions: list[str], max_new_tokens: int = 16
    ) -> list[str]:
        """images [B,3,H,W] in [0,1]."""

    @abc.abstractmethod
    def compute_nll_loss(
        self, images: torch.Tensor, questions: list[str], answers: list[str]
    ) -> torch.Tensor:
        """Scalar mean NLL over answer tokens (teacher forcing)."""

    @abc.abstractmethod
    def preprocess_for_model(self, images: torch.Tensor) -> torch.Tensor:
        """Differentiable preprocessing for vision stack (preserves grads)."""

    def assert_attackable(self) -> None:
        if not self.attackable:
            raise RuntimeError(
                f"Model {self.name} is not attackable (attackable=false). "
                "It cannot be used as a PGD source."
            )
