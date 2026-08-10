"""
Build a reference pool XML with clean and PGD robust accuracy on local labeled images.

Modes:
  full          : phase_clean on all N, then phase_robust (PGD) on all N
  intersection  : phase_clean on all N, keep all-models-correct subset, phase_robust on subset
                  (XML clean_acc=1.0; full-set clean stats printed to stdout)
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from sga.structural import set_reproducible
from common.pgd import (
    ReferenceBuildLogger,
    merge_reference_xml_paths,
    model_param_device,
    devices_match,
    per_image_clean_correct,
    per_image_robust_correct,
    resolve_pgd_params,
    write_results_xml,
)
from common.ranking_task import classifier_backend_dataset
from common.utils import load_one_model, load_samples, maybe_resize_samples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a reference pool XML from clean and PGD robust accuracy.",
    )
    p.add_argument(
        "--threat_model",
        type=str,
        default="Linf",
        choices=["Linf", "L2"],
        help=(
            "PGD threat model / norm (Linf or L2). RobustBench weights are always "
            "loaded as Linf (ImageNet RobustBench only provides Linf checkpoints)."
        ),
    )
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=256)
    p.add_argument(
        "--reference_xml",
        type=str,
        nargs="+",
        required=True,
        help="One or more XML files; model ids are unioned (architecture conflict => error).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "intersection"],
        help="full: all N images; intersection: all models clean-correct subset only",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=8,
        help=(
            "Linf: integer in 1..255, divided by 255. "
            "L2: perturbation radius used directly (no /255)."
        ),
    )
    p.add_argument(
        "--pgd_step_size",
        type=float,
        default=1,
        help=(
            "Linf: integer in 1..255, divided by 255. "
            "L2: step size used directly (no /255)."
        ),
    )
    p.add_argument("--pgd_steps", type=int, default=50)
    p.add_argument(
        "--no_random_start",
        action="store_true",
        help="Disable PGD random initialization inside the perturbation ball.",
    )
    p.add_argument("--grad_batch_size", type=int, default=32)
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even when CUDA is available.",
    )
    p.add_argument("--out_xml", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def log_device(logger: ReferenceBuildLogger, device: torch.device, x: Tensor) -> None:
    logger.log_line(f"compute_device={device}")
    logger.log_line(f"images_tensor_device={x.device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        logger.log_line(f"cuda_device={torch.cuda.get_device_name(idx)}")
    else:
        logger.log_line("cuda_device=n/a (CPU mode)")


def load_and_eval_model(
    *,
    args: argparse.Namespace,
    model_id: str,
    device: torch.device,
    x_eval: Tensor,
    y_eval: Tensor,
    eps: float,
    step_size: float,
    random_start: bool,
    do_clean: bool,
    do_robust: bool,
    threat_model: str,
) -> Tuple[float | None, float | None, np.ndarray | None, np.ndarray | None, float, float]:
    t0 = time.perf_counter()
    model = load_one_model(
        classifier_backend_dataset("imagenet"), model_id, threat_model="Linf"
    ).to(device).eval()
    load_sec = time.perf_counter() - t0

    clean_flags = None
    robust_flags = None
    clean_acc = None
    robust_acc = None

    t_eval = time.perf_counter()
    if do_clean:
        clean_flags = per_image_clean_correct(model, x_eval, y_eval, device, args.grad_batch_size)
        clean_acc = float(clean_flags.mean()) if clean_flags.size else 0.0
    if do_robust:
        robust_flags = per_image_robust_correct(
            model,
            x_eval,
            y_eval,
            device,
            args.grad_batch_size,
            threat_model=threat_model,
            eps=eps,
            step_size=step_size,
            steps=args.pgd_steps,
            random_start=random_start,
        )
        robust_acc = float(robust_flags.mean()) if robust_flags.size else 0.0
    eval_sec = time.perf_counter() - t_eval

    param_dev = model_param_device(model)
    if not devices_match(param_dev, device):
        raise RuntimeError(
            f"Model {model_id} parameters on {param_dev}, expected {device}"
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return clean_acc, robust_acc, clean_flags, robust_flags, load_sec, eval_sec


def main() -> None:
    args = parse_args()
    if args.pgd_steps <= 0:
        raise ValueError("--pgd_steps must be positive")
    if args.grad_batch_size <= 0:
        raise ValueError("--grad_batch_size must be positive")

    set_reproducible(args.seed)
    t0 = time.perf_counter()
    logger = ReferenceBuildLogger()
    logger.log_args(args)

    eps, step_size = resolve_pgd_params(args.threat_model, args.eps, args.pgd_step_size)
    random_start = not args.no_random_start
    threat_model = args.threat_model

    device = resolve_device(args.cpu)
    model_names, arch_by_id = merge_reference_xml_paths(args.reference_xml)
    n_models = len(model_names)

    logger.log_section("setup")
    logger.log_line(f"n_models={n_models}")
    logger.log_line(f"pgd_threat_model={threat_model}")
    if threat_model.lower() == "linf":
        logger.log_line(f"eps={eps:.8f} ({int(args.eps)}/255)")
        logger.log_line(f"pgd_step_size={step_size:.8f} ({int(args.pgd_step_size)}/255)")
    else:
        logger.log_line(f"eps={eps:.8f}")
        logger.log_line(f"pgd_step_size={step_size:.8f}")
    logger.log_line(f"pgd_steps={args.pgd_steps} random_start={random_start}")

    t_load = time.perf_counter()
    x_np, y_np = load_samples(args.input, args.start_idx, args.end_idx)
    x_np = maybe_resize_samples(x_np, "imagenet")
    load_sec = time.perf_counter() - t_load
    n_total = int(x_np.shape[0])

    x = Tensor(x_np).to(device, non_blocking=device.type == "cuda")
    y = Tensor(y_np).long().to(device, non_blocking=device.type == "cuda")
    log_device(logger, device, x)
    logger.log_line(f"n_images={n_total} load_images_sec={load_sec:.1f}")

    clean_flags: Dict[str, np.ndarray] = {}
    full_clean_acc: Dict[str, float] = {}

    if args.mode == "full":
        logger.log_section("phase_clean (all models, full set)")
        for model_id in model_names:
            clean_acc, _, flags, _, load_sec_m, eval_sec_m = load_and_eval_model(
                args=args,
                model_id=model_id,
                device=device,
                x_eval=x,
                y_eval=y,
                eps=eps,
                step_size=step_size,
                random_start=random_start,
                do_clean=True,
                do_robust=False,
                threat_model=threat_model,
            )
            assert clean_acc is not None and flags is not None
            full_clean_acc[model_id] = clean_acc
            logger.log_line(
                f"  {model_id}: clean={clean_acc:.6f} ({int(flags.sum())}/{n_total}) "
                f"load_sec={load_sec_m:.1f} eval_sec={eval_sec_m:.1f}"
            )

        entries = []
        logger.log_section("phase_robust (PGD, full set)")
        for model_id in model_names:
            _, robust_acc, _, robust_flags, load_sec_m, eval_sec_m = load_and_eval_model(
                args=args,
                model_id=model_id,
                device=device,
                x_eval=x,
                y_eval=y,
                eps=eps,
                step_size=step_size,
                random_start=random_start,
                do_clean=False,
                do_robust=True,
                threat_model=threat_model,
            )
            assert robust_acc is not None and robust_flags is not None
            entries.append(
                {
                    "model_id": model_id,
                    "clean_acc": full_clean_acc[model_id],
                    "robust_acc": robust_acc,
                    "architecture": arch_by_id[model_id],
                }
            )
            logger.log_line(
                f"  {model_id}: robust={robust_acc:.6f} "
                f"({int(robust_flags.sum())}/{n_total}) "
                f"load_sec={load_sec_m:.1f} pgd_sec={eval_sec_m:.1f}"
            )
    else:
        logger.log_section("phase_clean (all models, find intersection)")
        for model_id in model_names:
            clean_acc, _, flags, _, load_sec_m, eval_sec_m = load_and_eval_model(
                args=args,
                model_id=model_id,
                device=device,
                x_eval=x,
                y_eval=y,
                eps=eps,
                step_size=step_size,
                random_start=random_start,
                do_clean=True,
                do_robust=False,
                threat_model=threat_model,
            )
            assert clean_acc is not None and flags is not None
            clean_flags[model_id] = flags
            full_clean_acc[model_id] = clean_acc
            logger.log_line(
                f"  {model_id}: full_clean={clean_acc:.6f} ({int(flags.sum())}/{n_total}) "
                f"load_sec={load_sec_m:.1f} eval_sec={eval_sec_m:.1f}"
            )

        stack = np.stack([clean_flags[mid] for mid in model_names], axis=0)
        subset_idx = np.where(np.all(stack, axis=0))[0]
        n_subset = int(subset_idx.size)

        logger.log_section("intersection_subset")
        logger.log_line(f"n_total={n_total}")
        logger.log_line(f"n_subset={n_subset}")
        logger.log_line(
            f"subset_frac={n_subset / n_total:.6f}" if n_total else "subset_frac=nan"
        )
        logger.log_line("full_set_clean_acc (log only):")
        for model_id in model_names:
            logger.log_line(f"  {model_id}: {full_clean_acc[model_id]:.6f}")

        if n_subset == 0:
            raise RuntimeError(
                "Intersection subset is empty: no image is clean-correct for all models."
            )

        x_eval = x[torch.as_tensor(subset_idx, device=x.device)]
        y_eval = y[torch.as_tensor(subset_idx, device=y.device)]

        entries = []
        logger.log_section("phase_robust (PGD on intersection subset)")
        for model_id in model_names:
            _, robust_acc, _, robust_flags, load_sec_m, eval_sec_m = load_and_eval_model(
                args=args,
                model_id=model_id,
                device=device,
                x_eval=x_eval,
                y_eval=y_eval,
                eps=eps,
                step_size=step_size,
                random_start=random_start,
                do_clean=False,
                do_robust=True,
                threat_model=threat_model,
            )
            assert robust_acc is not None and robust_flags is not None
            entries.append(
                {
                    "model_id": model_id,
                    "clean_acc": 1.0,
                    "robust_acc": robust_acc,
                    "architecture": arch_by_id[model_id],
                }
            )
            logger.log_line(
                f"  {model_id}: clean=1.000000 robust={robust_acc:.6f} "
                f"({int(robust_flags.sum())}/{n_subset}) "
                f"load_sec={load_sec_m:.1f} pgd_sec={eval_sec_m:.1f}"
            )

    write_results_xml(args.out_xml, entries)
    logger.log_section("output")
    logger.log_line(f"out_xml={args.out_xml}")
    logger.log_line("results (sorted by robust_acc desc in XML):")
    for e in sorted(entries, key=lambda x: float(x["robust_acc"]), reverse=True):
        logger.log_line(
            f"  {e['model_id']}: clean={float(e['clean_acc']):.6f} "
            f"robust={float(e['robust_acc']):.6f}"
        )

    logger.close(started_at=t0)


if __name__ == "__main__":
    main()
