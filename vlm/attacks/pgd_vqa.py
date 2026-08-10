from __future__ import annotations

import torch

from vlm.utils.logging import get_logger

_log = get_logger()


def pgd_linf_vqa(
    model,
    images: torch.Tensor,
    questions: list[str],
    answers: list[str],
    eps: float = 4 / 255,
    alpha: float = 1 / 255,
    steps: int = 40,
    random_start: bool = True,
    clamp_min: float = 0.0,
    clamp_max: float = 1.0,
    *,
    log_each_step: bool = False,
    log_prefix: str = "",
) -> torch.Tensor:
    """
    Untargeted VQA PGD: maximize teacher-forcing NLL of the ground-truth answer.
    images: [B,3,H,W] in [0,1]. Returns detached adversarial images in [0,1].
    """
    model.assert_attackable()
    was_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    images = images.detach()
    adv = images.clone()
    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = torch.clamp(adv, clamp_min, clamp_max)

    for t in range(steps):
        adv = adv.detach().requires_grad_(True)
        loss = model.compute_nll_loss(adv, questions, answers)
        if log_each_step:
            _log.info("%spgd step %d/%d loss=%.6f", log_prefix, t + 1, steps, float(loss.detach().cpu()))
        loss.backward()
        with torch.no_grad():
            grad = adv.grad
            adv = adv + alpha * grad.sign()
            adv = torch.max(torch.min(adv, images + eps), images - eps)
            adv = torch.clamp(adv, clamp_min, clamp_max)
        model.zero_grad(set_to_none=True)

    if was_training:
        model.train()
    return adv.detach()
