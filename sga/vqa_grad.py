"""Input gradients for the LVLM (LLaVA) branch of the SGA scoring pipeline."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor

from common.ranking_task import RankingSample


def _questions_answers(
    batch: Sequence[RankingSample],
) -> tuple[List[str], List[str], List[List[str]]]:
    qs = [s.question or "" for s in batch]
    ans = [s.answer or "" for s in batch]
    multi = [list(s.answers_multi or [s.answer or ""]) for s in batch]
    return qs, ans, multi


def resolve_vqa_attack_answers(
    model,
    images: Tensor,
    batch: Sequence[RankingSample],
    conditioning: str,
) -> tuple[List[str], List[str], List[List[str]]]:
    """Answers that condition the NLL objective: dataset answers or model predictions."""
    qs, gt_ans, multi = _questions_answers(batch)
    if conditioning == "ground_truth":
        return qs, gt_ans, multi
    with torch.inference_mode():
        preds = model.generate_answer(images, qs)
    return qs, list(preds), [[p] for p in preds]


def batch_raw_gradients_vqa(
    model,
    images: Tensor,
    questions: Sequence[str],
    answers: Sequence[str],
    *,
    amp: bool = False,
) -> Tensor:
    """Input gradient of the teacher-forced answer NLL, shape [B, C, H, W].

    ``compute_nll_loss`` averages over the answer tokens of the whole batch, so
    gradient magnitudes depend on the batch size. Keep ``--grad_batch_size`` fixed
    across every run that is compared.
    """
    x = images.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if amp and x.device.type == "cuda":
        with torch.cuda.amp.autocast():
            loss = model.compute_nll_loss(x, list(questions), list(answers))
    else:
        loss = model.compute_nll_loss(x, list(questions), list(answers))
    loss.backward()
    if x.grad is None:
        raise RuntimeError("VQA NLL gradient is None")
    return x.grad.detach()
