"""
Build an LVLM reference pool XML with clean and PGD robust VQA accuracy on VQAv2.

Uses the same model ids as ``reference/lvlm_linf.xml``. ViT-L/14 models only; ConvNeXt-B
and ViT-B/16 RCLIP models are skipped (no supported LLaVA VQA stack).

Modes:
  full          : clean eval on all samples, then PGD robust eval on all
  intersection  : clean eval on [--start-idx, --end-idx), keep samples where every
                  model in --reference_xml is clean-correct, then PGD on that subset

Optional --subset-file pre-filters jsonl rows by question id before clean eval
(manual scope limit). For all-models-clean intersection, use --mode intersection
instead of a separate id-list script.
"""

from __future__ import annotations

import argparse
import gc
import time
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from sga.structural import set_reproducible
from common.reference import write_reference_xml
from common.pgd import ReferenceBuildLogger, merge_reference_xml_paths, resolve_pgd_params, sort_entries_by_robust
from vlm.attacks.pgd_vqa import pgd_linf_vqa
from vlm.eval.vqa_metrics import exact_match_correct
from vlm.model_registry import is_vqa_loadable, vqa_load_error_for_id
from common.vqa_data import (
    iter_vqa_batches,
    load_vqa_model,
    make_vqa_dataloader,
    make_vqa_dataloader_from_rows,
    resolve_vqa_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build an LVLM reference pool XML from clean and PGD robust VQA accuracy.")
    p.add_argument("--data-jsonl", type=str, required=True)
    p.add_argument("--input", type=str, required=True, help="COCO image directory (e.g. data/coco/val2014).")
    p.add_argument(
        "--subset-file",
        type=str,
        default=None,
        help="Optional: only load these question ids from jsonl (pre-filter). "
        "For all-models clean intersection use --mode intersection.",
    )
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None)
    p.add_argument(
        "--reference_xml",
        type=str,
        nargs="+",
        required=True,
        help="Model list XML (same ids as reference/lvlm_linf.xml).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "intersection"],
    )
    p.add_argument("--threat_model", type=str, default="Linf", choices=["Linf"])
    p.add_argument("--eps", type=float, default=8, help="Linf epsilon as integer 1..255, divided by 255.")
    p.add_argument("--pgd_step_size", type=float, default=1, help="Linf step as integer 1..255, divided by 255.")
    p.add_argument("--pgd_steps", type=int, default=50)
    p.add_argument("--grad_batch_size", type=int, default=1, help="VQA dataloader chunk (default: 1).")
    p.add_argument(
        "--no_random_start",
        action="store_true",
        help="Disable PGD random initialization inside the perturbation ball.",
    )
    p.add_argument("--out_xml", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _filter_vqa_models(model_ids: List[str], log: ReferenceBuildLogger) -> List[str]:
    kept: List[str] = []
    for mid in model_ids:
        if is_vqa_loadable(mid):
            kept.append(mid)
            continue
        reason = vqa_load_error_for_id(mid)
        log.log_line(f"skip {mid} (no LLaVA VQA stack): {reason}")
    if not kept:
        raise RuntimeError("No VQA-loadable models in --reference_xml.")
    return kept


def _eval_clean(model, dl, device, *, n_samples: int, desc: str = "clean") -> np.ndarray:
    flags: List[float] = []
    pbar = tqdm(total=n_samples, desc=desc, unit="sample", dynamic_ncols=True)
    for batch in iter_vqa_batches(dl, device):
        with torch.inference_mode():
            preds = model.generate_answer(batch["image"], batch["question"])
        for pred, ans in zip(preds, batch["answers"]):
            flags.append(float(exact_match_correct(pred, ans)))
        if flags:
            pbar.set_postfix(acc=f"{np.mean(flags):.3f}", refresh=False)
        pbar.update(len(batch["id"]))
    pbar.close()
    return np.asarray(flags, dtype=bool)


def _eval_robust(
    model,
    dl,
    device,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
    *,
    n_samples: int,
    desc: str = "PGD",
) -> np.ndarray:
    flags: List[float] = []
    pbar = tqdm(total=n_samples, desc=desc, unit="sample", dynamic_ncols=True)
    for batch in iter_vqa_batches(dl, device):
        adv = pgd_linf_vqa(
            model,
            batch["image"],
            batch["question"],
            batch["answer"],
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )
        with torch.inference_mode():
            preds = model.generate_answer(adv, batch["question"])
        for pred, ans in zip(preds, batch["answers"]):
            flags.append(float(exact_match_correct(pred, ans)))
        if flags:
            pbar.set_postfix(acc=f"{np.mean(flags):.3f}", refresh=False)
        pbar.update(len(batch["id"]))
    pbar.close()
    return np.asarray(flags, dtype=bool)


def main() -> None:
    args = parse_args()
    set_reproducible(args.seed)
    eps, alpha = resolve_pgd_params(args.threat_model, args.eps, args.pgd_step_size)
    random_start = not args.no_random_start
    all_ids, arch_by_id = merge_reference_xml_paths(args.reference_xml)
    log = ReferenceBuildLogger()
    t0 = time.perf_counter()
    log.log_args(args)
    model_ids = _filter_vqa_models(all_ids, log)

    dl, ds = make_vqa_dataloader(
        data_jsonl=args.data_jsonl,
        image_root=args.input,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        subset_file=args.subset_file,
        batch_size=args.grad_batch_size,
    )
    n_samples = len(ds)
    log.log_line(f"samples={n_samples} vqa_models={len(model_ids)}")

    clean_flags: Dict[str, np.ndarray] = {}
    for mid in model_ids:
        log.log_section(f"clean: {mid}")
        model = load_vqa_model(mid)
        device = resolve_vqa_device(model)
        model = model.to(device).eval()
        clean_flags[mid] = _eval_clean(model, dl, device, n_samples=n_samples, desc=f"clean {mid}")
        log.log_line(f"  clean_acc={clean_flags[mid].mean():.4f}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.mode == "intersection":
        mask = np.ones(n_samples, dtype=bool)
        for mid in model_ids:
            mask &= clean_flags[mid]
        keep = int(mask.sum())
        log.log_line(f"intersection clean-correct: {keep}/{n_samples}")
        if keep == 0:
            raise RuntimeError("No samples where all models are clean-correct.")
        for mid in model_ids:
            clean_flags[mid] = clean_flags[mid][mask]
        kept_rows = [ds.samples[i] for i, ok in enumerate(mask) if ok]
        dl, ds = make_vqa_dataloader_from_rows(
            kept_rows,
            image_root=args.input,
            batch_size=args.grad_batch_size,
        )
        n_samples = len(ds)

    entries: List[Dict[str, object]] = []
    log.log_line(
        f"pgd: eps={eps:.6f} alpha={alpha:.6f} steps={args.pgd_steps} "
        f"random_start={random_start}"
    )
    for mid in model_ids:
        log.log_section(f"robust PGD: {mid}")
        model = load_vqa_model(mid)
        device = resolve_vqa_device(model)
        model = model.to(device).eval()
        robust_flags = _eval_robust(
            model,
            dl,
            device,
            eps,
            alpha,
            args.pgd_steps,
            random_start,
            n_samples=n_samples,
            desc=f"PGD {mid}",
        )
        entries.append(
            {
                "model_id": mid,
                "clean_acc": float(clean_flags[mid].mean()),
                "robust_acc": float(robust_flags.mean()),
                "architecture": arch_by_id.get(mid, mid),
            }
        )
        log.log_line(f"  robust_acc={entries[-1]['robust_acc']:.4f}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_reference_xml(args.out_xml, sort_entries_by_robust(entries))
    log.log_line(f"Wrote {args.out_xml}")
    log.close(started_at=t0)


if __name__ == "__main__":
    main()
