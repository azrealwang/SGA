"""
Stage 1 of SGA: structural sensitivity alignment per model-probe pair.

For every model listed in --reference_xml, computes one input gradient per probe
in --input, turns gradient and probe into structural maps under each candidate
configuration c = (rho, psi, e, mu), and writes the resulting alignment scores as
one CSV per (configuration, similarity). Configuration selection and pairwise
agreement happen later, in the ranking scripts, which reuse these cached scores.

With --save_maps, each configuration directory also holds the intermediate maps:
  input_edges/img_*_gray.png    probe grayscale, before smoothing
  input_edges/img_*_smooth.png  probe grayscale after psi
  input_edges/img_*.png         probe structural map B^x
  grad_maps/<model>/img_*_grad.png          raw input gradient, RGB
  grad_maps/<model>/img_*_merged.png        ternary sensitivity representation Z
  grad_maps/<model>/img_*_merged_smooth.png Z after psi
  grad_maps/<model>/img_*_merged_edge.png   sensitivity structural map B^g
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor

from sga.structural import (
    PerConfigPerImage,
    chw01_to_gray_hw,
    compute_per_probe_alignment,
    precompute_score_matrices,
    set_reproducible,
)
from common.reference import load_reference_xml, models_from_reference_xml
from sga.rank_io import (
    add_configuration_args,
    args_meta_string,
    parse_configuration_args,
    setting_root_from_key,
    similarity_csv_path,
    write_export_summary_csv,
    write_similarity_csv,
)
from common.ranking_task import (
    RankingSample,
    _pad_images_chw01_to_batch,
    add_ranking_task_args,
    is_vqa_task,
    load_folder_ranking_samples,
    load_ranking_model,
    load_vqa_ranking_samples,
    normalize_ranking_args,
    resolve_grad_batch_size,
)
from common.utils import load_samples
from vlm.utils.image import _resize_shortest_edge, center_crop_square


def parse_resize_arg(value: str) -> int | None:
    """``none`` -> no resize; otherwise CLIP-style square side (shortest-edge + center crop)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none"}:
        return None
    size = int(text)
    if size < 1:
        raise ValueError(f"--resize must be 'none' or a positive integer, got {value!r}")
    return size


def resize_chw01_square(img: np.ndarray, size: int) -> np.ndarray:
    """Match CLIP: resize shortest edge to ``size``, then center-crop to ``size x size``.

    Preserves aspect ratio (unlike a direct stretch to square). ``img`` is CHW in [0, 1].
    """
    t = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).float()
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
        squeeze_gray = True
    else:
        squeeze_gray = False
    if int(t.shape[2]) == size and int(t.shape[3]) == size:
        return img
    t = _resize_shortest_edge(t, size)
    t = center_crop_square(t, size)
    out = t.squeeze(0)
    if squeeze_gray:
        out = out[:1]
    return np.ascontiguousarray(out.numpy())


def apply_resize_to_samples(
    samples: Sequence[RankingSample],
    resize: int | None,
) -> np.ndarray:
    images = [s.image for s in samples]
    if resize is None:
        x_np = _pad_images_chw01_to_batch(images)
        for i, sample in enumerate(samples):
            sample.image = x_np[i]
        return x_np

    aligned: List[np.ndarray] = []
    for i, sample in enumerate(samples):
        out = resize_chw01_square(images[i], resize)
        sample.image = out
        aligned.append(out)
    return np.stack(aligned, axis=0)


def load_native_ranking_samples(args: argparse.Namespace) -> List[RankingSample]:
    """Load samples at native resolution (no task-wide resize)."""
    if is_vqa_task(args.task):
        return load_vqa_ranking_samples(
            args.data_jsonl,
            args.input,
            args.start_idx,
            args.end_idx,
            args.subset_file,
            conditioning=args.conditioning,
        )
    if args.task == "cifar10":
        x_np, y_np = load_samples(args.input, args.start_idx, args.end_idx)
        return [
            RankingSample(image=x_np[i], class_id=int(y_np[i]))
            for i in range(x_np.shape[0])
        ]
    require_label = args.conditioning == "ground_truth"
    return load_folder_ranking_samples(
        args.input,
        args.start_idx,
        args.end_idx,
        require_label=require_label,
        spatial=None,
    )


def load_export_samples(
    args: argparse.Namespace,
) -> tuple[List[RankingSample], np.ndarray, np.ndarray, int | None]:
    normalize_ranking_args(args)
    resize = parse_resize_arg(args.resize)
    samples = load_native_ranking_samples(args)
    x_np = apply_resize_to_samples(samples, resize)
    if is_vqa_task(args.task):
        y_np = np.full(len(samples), -1, dtype=np.int64)
    else:
        y_np = np.array(
            [-1 if s.class_id is None else int(s.class_id) for s in samples],
            dtype=np.int64,
        )
    return samples, x_np, y_np, resize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and cache SGA alignment scores for every candidate configuration.",
    )
    add_ranking_task_args(p)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=256)
    p.add_argument("--reference_xml", type=str, required=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dilate", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--save_maps", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--resize",
        type=str,
        default="none",
        help=(
            "CLIP-style square side: shortest-edge bilinear resize then center crop "
            "(default: none = native resolution, batch-padded when sizes differ)."
        ),
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU scoring (default: GPU when CUDA is available).",
    )
    add_configuration_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    normalize_ranking_args(args)
    set_reproducible(args.seed)
    configurations, ranking_metrics, base_metrics, _sweep_keys = parse_configuration_args(
        args
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu_scoring = not args.cpu
    scoring_backend = "gpu" if use_gpu_scoring and device.type == "cuda" else "cpu"

    reference_entries = load_reference_xml(args.reference_xml)
    model_names, _labels, reference_by_id = models_from_reference_xml(reference_entries)

    t_load_input = time.perf_counter()
    samples, x_np, y_np, resize = load_export_samples(args)
    load_input_sec = time.perf_counter() - t_load_input
    n_samples = x_np.shape[0]
    sample_indices = list(range(args.start_idx, args.start_idx + n_samples))
    grad_bs = resolve_grad_batch_size(args, n_samples)

    x = Tensor(x_np)
    y = Tensor(y_np).long()
    vqa_samples = samples if is_vqa_task(args.task) else None
    input_grays = (
        [chw01_to_gray_hw(x_np[i]) for i in range(n_samples)]
        if scoring_backend == "cpu"
        else []
    )
    maps_settings_root = os.path.join(args.out_dir, "settings") if args.save_maps else None
    os.makedirs(args.out_dir, exist_ok=True)

    resize_note = "native" if resize is None else f"{resize}x{resize}"
    print(
        f"Export raw similarities | task={args.task} "
        f"models={len(model_names)} samples={n_samples} "
        f"settings={len(configurations)} metrics={list(ranking_metrics)} "
        f"device={device} scoring={scoring_backend} resize={resize_note}",
        flush=True,
    )

    per_config_all: PerConfigPerImage = {
        cfg: {mid: {m: [] for m in base_metrics} for mid in model_names}
        for cfg in configurations
    }
    model_timings: List[Dict[str, float]] = []
    total_load_model = total_grad = total_sim = total_save_maps = 0.0

    def load_scoring_model(name: str) -> torch.nn.Module:
        return load_ranking_model(args.task, name, device, threat_model="Linf")

    for model_id in model_names:
        t0 = time.perf_counter()
        model = load_scoring_model(model_id)
        load_model_sec = time.perf_counter() - t0
        total_load_model += load_model_sec
        print(
            f"  scoring {model_id} ({load_model_sec:.1f}s load)",
            flush=True,
        )
        t_score = time.perf_counter()
        per_config_scores = compute_per_probe_alignment(
            model_names=[model_id],
            model_loader=lambda _mid, _m=model: _m,
            x=x,
            y=y,
            input_grays=input_grays,
            configurations=configurations,
            metrics=base_metrics,
            device=device,
            conditioning=args.conditioning,
            amp=args.amp,
            dilate=args.dilate,
            grad_batch_size=grad_bs,
            start_idx=args.start_idx,
            save_maps=args.save_maps,
            maps_settings_root=maps_settings_root,
            show_progress=not args.quiet,
            use_gpu=use_gpu_scoring,
            vqa_samples=vqa_samples,
        )
        score_sec = time.perf_counter() - t_score
        total_grad += score_sec
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        for cfg in configurations:
            scores = per_config_scores[cfg][model_id]
            for m in base_metrics:
                per_config_all[cfg][model_id][m].extend(scores[m])

        model_timings.append(
            {
                "model_id": model_id,
                "load_model": load_model_sec,
                "grad": score_sec,
                "sim": score_sec,
                "save_maps": 0.0,
                "save_csv": 0.0,
            }
        )

    t_write = time.perf_counter()
    for configuration in configurations:
        setting_root = setting_root_from_key(args.out_dir, configuration)
        os.makedirs(setting_root, exist_ok=True)
        score_matrices = precompute_score_matrices(
            {configuration: per_config_all[configuration]},
            model_names,
            [configuration],
            ranking_metrics,
        )
        for metric in ranking_metrics:
            edge, sigma, ds, thr = configuration
            key = (edge, sigma, ds, thr, metric)
            csv_path = similarity_csv_path(setting_root, metric)
            write_similarity_csv(
                csv_path,
                model_ids=model_names,
                reference_by_id=reference_by_id,
                sample_indices=sample_indices,
                score_matrix=score_matrices[key],
                configuration=configuration,
                metric=metric,
            )
    save_csv_sec = time.perf_counter() - t_write
    total_save_maps = 0.0

    summary_path = os.path.join(args.out_dir, "export_summary.csv")
    write_export_summary_csv(
        summary_path,
        args_meta=args_meta_string(args),
        model_ids=model_names,
        reference_by_id=reference_by_id,
        sample_indices=sample_indices,
        n_settings=len(configurations) * len(ranking_metrics),
        timings={
            "load_input": load_input_sec,
            "load_model_total": total_load_model,
            "grad_sim_total": total_grad,
            "save_maps_total": total_save_maps,
            "save_csv": save_csv_sec,
            "total": load_input_sec + total_load_model + total_grad + save_csv_sec,
        },
        model_timings=model_timings,
    )
    print(f"Saved export summary: {summary_path}", flush=True)
    print(f"Saved {len(configurations)} settings x {len(ranking_metrics)} metrics", flush=True)


if __name__ == "__main__":
    main()
