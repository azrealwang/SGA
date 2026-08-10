"""GPU scoring path: gradient -> ternary representation -> structural maps -> similarity."""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sga.structural import (
    ConfigKey,
    PerConfigPerImage,
    SmoothKey,
    ThresholdSpec as Thr,
    batch_raw_gradients,
    iter_grad_batches,
    progress_bar,
    save_binary_png,
    save_gradient_rgb_png,
    save_grayscale_png,
    save_label_map_png,
    setting_dir_name,
    threshold_sort_key,
)

_LUMA = (0.2989, 0.5870, 0.1140)


def batch_raw_gradients_chw(
    model: nn.Module,
    xb: Tensor,
    yb: Tensor,
    device: torch.device,
    *,
    conditioning: str,
    amp: bool = False,
) -> Tensor:
    """[B, C, H, W] input gradients on device."""
    flat = batch_raw_gradients(model, xb, yb, device, conditioning=conditioning, amp=amp)
    b, d = flat.shape
    c, h, w = xb.shape[1], xb.shape[2], xb.shape[3]
    return flat.view(b, c, h, w)


def _gaussian_blur2d(x: Tensor, sigma: float) -> Tensor:
    """x: [B, 1, H, W]."""
    if sigma <= 0:
        return x
    size = int(max(3, 2 * math.ceil(3.0 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    ax = torch.arange(size, device=x.device, dtype=x.dtype) - (size - 1) / 2.0
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum()
    pad = size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x = F.conv2d(x, g1.view(1, 1, 1, size))
    x = F.conv2d(x, g1.view(1, 1, size, 1))
    return x


def _structural_smooth_gpu(gray: Tensor, factor: int) -> Tensor:
    """gray: [B, 1, H, W]."""
    if factor <= 1:
        return gray
    h, w = gray.shape[-2:]
    t = F.interpolate(
        gray,
        size=(max(1, h // factor), max(1, w // factor)),
        mode="bilinear",
        align_corners=False,
    )
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


def _smooth_gpu(
    gray: Tensor,
    *,
    smooth_sigma: float,
    smooth_scale: int,
) -> Tensor:
    if smooth_sigma > 0:
        return _gaussian_blur2d(gray, smooth_sigma)
    if smooth_scale > 1:
        return _structural_smooth_gpu(gray, smooth_scale)
    return gray


def edge_magnitude_gpu(
    gray: Tensor,
    *,
    smooth_sigma: float = 0.0,
    smooth_scale: int = 1,
) -> Tensor:
    """gray [B, H, W] -> magnitude [B, H, W]."""
    g = _smooth_gpu(
        gray.unsqueeze(1),
        smooth_sigma=smooth_sigma,
        smooth_scale=smooth_scale,
    ).squeeze(1)
    gx = torch.zeros_like(g)
    gy = torch.zeros_like(g)
    gx[:, :, 1:-1] = g[:, :, 2:] - g[:, :, :-2]
    gy[:, 1:-1, :] = g[:, 2:, :] - g[:, :-2, :]
    return torch.hypot(gx, gy)


def binary_edge_from_magnitude_gpu(mag: Tensor, percentile: float) -> Tensor:
    """mag [B, H, W] -> binary [B, H, W]."""
    out = torch.zeros_like(mag)
    q = float(percentile) / 100.0
    for i in range(mag.shape[0]):
        pos = mag[i][mag[i] > 0]
        if pos.numel() == 0:
            continue
        t = torch.quantile(pos, q)
        out[i] = (mag[i] >= t).to(mag.dtype)
    return out


def dilate_binary_gpu(mask: Tensor, radius: int) -> Tensor:
    """mask [B, H, W]."""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(
        mask.unsqueeze(1),
        kernel_size=k,
        stride=1,
        padding=radius,
    ).squeeze(1)


def labels_merged_gray01_gpu(lab_bchw: Tensor) -> Tensor:
    """lab [B, C, H, W] in {-1,0,1} -> gray [B, H, W] in [0,1]."""
    levels = torch.tensor([0.0, 128.0, 255.0], device=lab_bchw.device, dtype=lab_bchw.dtype) / 255.0
    idx = (lab_bchw + 1).long().clamp(0, 2)
    u8 = levels[idx]
    if u8.shape[1] == 3:
        w = torch.tensor(_LUMA, device=lab_bchw.device, dtype=lab_bchw.dtype).view(1, 3, 1, 1)
        return (u8 * w).sum(dim=1)
    return u8[:, 0]


def gradients_to_labels_abs_gpu(g: Tensor, threshold: float) -> Tensor:
    """g [B, C, H, W] -> labels same shape."""
    abs_g = g.abs()
    labels = torch.zeros_like(g)
    high = abs_g >= threshold
    labels[high & (g > 0)] = 1.0
    labels[high & (g < 0)] = -1.0
    return labels


def gradients_to_labels_top_pct_gpu(g: Tensor, top_pct: float) -> Tensor:
    """g [B, C, H, W]; per-sample top_pct threshold."""
    b, c, h, w = g.shape
    labels = torch.zeros_like(g)
    if top_pct <= 0:
        return labels
    if top_pct >= 100:
        labels[g > 0] = 1.0
        labels[g < 0] = -1.0
        return labels
    flat_n = c * h * w
    k = max(1, int(math.ceil(flat_n * top_pct / 100.0)))
    for i in range(b):
        abs_g = g[i].abs().reshape(-1)
        thr = abs_g.kthvalue(flat_n - k + 1).values
        high = g[i].abs() >= thr
        labels[i][high & (g[i] > 0)] = 1.0
        labels[i][high & (g[i] < 0)] = -1.0
    return labels


def gradients_to_labels_from_spec_gpu(g: Tensor, spec: Thr) -> Tensor:
    mode, val = spec
    if mode == "abs":
        return gradients_to_labels_abs_gpu(g, val)
    if mode == "top_pct":
        return gradients_to_labels_top_pct_gpu(g, val)
    raise ValueError(f"Unknown threshold mode {mode!r}")


def dice_similarity_gpu(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    """a, b [B, H, W] -> [B] dice."""
    af = (a > 0.5).reshape(a.shape[0], -1).float()
    bf = (b > 0.5).reshape(b.shape[0], -1).float()
    inter = (af * bf).sum(dim=1)
    return (2.0 * inter) / (af.sum(dim=1) + bf.sum(dim=1) + eps)


def _gaussian_window_2d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    w = torch.outer(g, g)
    return (w / w.sum()).view(1, 1, size, size)


def ssim_similarity_gpu(
    a: Tensor,
    b: Tensor,
    *,
    window_size: int = 11,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> Tensor:
    """a, b [B, H, W] -> [B] mean SSIM."""
    x = a.unsqueeze(1)
    y = b.unsqueeze(1)
    sigma = 1.5
    window = _gaussian_window_2d(window_size, sigma, a.device, a.dtype)
    pad = window_size // 2
    x = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.pad(y, (pad, pad, pad, pad), mode="reflect")
    mu_x = F.conv2d(x, window)
    mu_y = F.conv2d(y, window)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window) - mu_x2
    sigma_y2 = F.conv2d(y * y, window) - mu_y2
    sigma_xy = F.conv2d(x * y, window) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / (den + eps)
    return ssim_map.mean(dim=(1, 2, 3))


GPU_METRICS = {
    "dice": dice_similarity_gpu,
    "ssim": ssim_similarity_gpu,
}


def rgb2gray_batch(x: Tensor) -> Tensor:
    """[N, 3, H, W] in [0,1] -> [N, H, W]."""
    if x.shape[1] == 1:
        return x[:, 0]
    w = torch.tensor(_LUMA, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1)


def precompute_input_edge_refs_gpu(
    input_gray: Tensor,
    smooth_keys: Sequence[SmoothKey],
    edges_for_smooth: Dict[SmoothKey, Tuple[float, ...]],
    dilate: int,
) -> Dict[SmoothKey, Dict[float, Tensor]]:
    """input_gray [N, H, W] on device -> ref edges [N, H, W] per (smooth, edge)."""
    refs: Dict[SmoothKey, Dict[float, Tensor]] = {}
    for sigma, ds in smooth_keys:
        refs[(sigma, ds)] = {}
        for ep in edges_for_smooth[(sigma, ds)]:
            mag = edge_magnitude_gpu(
                input_gray,
                smooth_sigma=sigma,
                smooth_scale=ds,
            )
            edge = dilate_binary_gpu(binary_edge_from_magnitude_gpu(mag, ep), dilate)
            refs[(sigma, ds)][ep] = edge
    return refs


def compute_per_probe_alignment_gpu(
    *,
    model_names: List[str],
    x: Tensor,
    y: Tensor,
    configurations: Sequence[ConfigKey],
    metrics: Tuple[str, ...],
    device: torch.device,
    conditioning: str,
    amp: bool,
    dilate: int,
    grad_batch_size: int,
    model_loader,
    out: PerConfigPerImage,
    show_progress: bool = True,
    start_idx: int = 0,
    save_maps: bool = False,
    maps_settings_root: Optional[str] = None,
    vqa_samples: Optional[Sequence[object]] = None,
) -> Tuple[float, float]:
    """Score all models on GPU; fill ``out`` in place. Returns (grad_sec, scoring_sec).

    ``x``/``y`` may live on CPU; only each ram batch is moved to ``device``.
    Input edge refs are computed per batch (not for the full dataset).
    Similarity scores are accumulated on CPU; intermediate maps are discarded
    unless ``save_maps`` is True.
    """
    import time

    n_samples = x.shape[0]
    configs = list(configurations)
    config_set = set(configs)
    threshold_list = tuple(
        sorted({thr for _, _, _, thr in configs}, key=threshold_sort_key)
    )
    smooth_keys: List[SmoothKey] = sorted({(sigma, ds) for _, sigma, ds, _ in configs})
    edges_for_smooth: Dict[SmoothKey, Tuple[float, ...]] = {
        sk: tuple(sorted({ep for ep, s, d, _ in configs if (s, d) == sk}))
        for sk in smooth_keys
    }
    score_bufs: Dict[ConfigKey, Dict[str, Dict[str, Tensor]]] = {
        key: {
            mid: {
                m: torch.empty(n_samples, device="cpu", dtype=torch.float32)
                for m in metrics
            }
            for mid in model_names
        }
        for key in configs
    }
    grad_time_sec = 0.0
    scoring_time_sec = 0.0
    saved_input_edges: set[Tuple[float, float, int, float, int]] = set()
    saved_input_smooth: set[Tuple[float, float, int]] = set()
    saved_input_gray: set[int] = set()
    saved_raw_grad: set[Tuple[str, int]] = set()

    show_model_progress = show_progress and len(model_names) > 1
    single_model = len(model_names) == 1
    model_iter = progress_bar(
        range(len(model_names)),
        desc="GPU load+score",
        total=len(model_names),
        unit="model",
        disable=not show_model_progress,
    )
    for mi in model_iter:
        model_id = model_names[mi]
        safe = model_id.replace(os.sep, "_").replace(" ", "_")
        model = model_loader(model_id)
        img_offset = 0
        try:
            for xb, yb in iter_grad_batches(
                x,
                y,
                grad_batch_size,
                show_progress=show_progress,
                single_model=single_model,
            ):
                xb_gpu = xb.to(device, non_blocking=device.type == "cuda")
                input_gray_b = rgb2gray_batch(xb_gpu.detach().float())
                edge_refs_b = precompute_input_edge_refs_gpu(
                    input_gray_b,
                    smooth_keys,
                    edges_for_smooth,
                    dilate,
                )

                t0 = time.perf_counter()
                if vqa_samples is not None:
                    from sga.vqa_grad import (
                        batch_raw_gradients_vqa,
                        resolve_vqa_attack_answers,
                    )

                    sb = list(vqa_samples[img_offset : img_offset + xb.shape[0]])
                    qs, ans, _ = resolve_vqa_attack_answers(
                        model, xb_gpu, sb, conditioning
                    )
                    g = batch_raw_gradients_vqa(
                        model, xb_gpu, qs, ans, amp=amp
                    )
                else:
                    g = batch_raw_gradients_chw(
                        model, xb_gpu, yb, device, conditioning=conditioning, amp=amp
                    )
                grad_time_sec += time.perf_counter() - t0
                b = g.shape[0]
                if save_maps and maps_settings_root:
                    for i in range(b):
                        global_img_idx = start_idx + img_offset + i
                        raw_key = (model_id, global_img_idx)
                        if raw_key not in saved_raw_grad:
                            save_gradient_rgb_png(
                                g[i].detach().cpu().numpy(),
                                os.path.join(
                                    maps_settings_root,
                                    "grad_maps",
                                    safe,
                                    f"img_{global_img_idx:05d}_grad.png",
                                ),
                            )
                            saved_raw_grad.add(raw_key)
                t1 = time.perf_counter()

                labels_by_thr: Dict[Thr, Tensor] = {
                    thr: gradients_to_labels_from_spec_gpu(g, thr)
                    for thr in threshold_list
                }
                merged_by_thr = {
                    thr: labels_merged_gray01_gpu(lab) for thr, lab in labels_by_thr.items()
                }

                for thr in threshold_list:
                    merged = merged_by_thr[thr]
                    for sigma, ds in smooth_keys:
                        mag = edge_magnitude_gpu(
                            merged,
                            smooth_sigma=sigma,
                            smooth_scale=ds,
                        )
                        for ep in edges_for_smooth[(sigma, ds)]:
                            grad_edge = dilate_binary_gpu(
                                binary_edge_from_magnitude_gpu(mag, ep),
                                dilate,
                            )
                            ref_edge = edge_refs_b[(sigma, ds)][ep]
                            key = (ep, sigma, ds, thr)
                            if key not in config_set:
                                continue
                            if save_maps and maps_settings_root:
                                setting_root = os.path.join(
                                    maps_settings_root,
                                    setting_dir_name(ep, sigma, ds, thr),
                                )
                                for i in range(b):
                                    global_img_idx = start_idx + img_offset + i
                                    in_key = (sigma, ds, ep, global_img_idx)
                                    ref_np = ref_edge[i].detach().cpu().numpy()
                                    grad_np = grad_edge[i].detach().cpu().numpy()
                                    merged_np = merged[i].detach().cpu().numpy()
                                    glab = labels_by_thr[thr]
                                    if in_key not in saved_input_edges:
                                        input_root = os.path.join(setting_root, "input_edges")
                                        if global_img_idx not in saved_input_gray:
                                            save_grayscale_png(
                                                input_gray_b[i].detach().cpu().numpy(),
                                                os.path.join(
                                                    input_root,
                                                    f"img_{global_img_idx:05d}_gray.png",
                                                ),
                                            )
                                            saved_input_gray.add(global_img_idx)
                                        if (sigma, ds, global_img_idx) not in saved_input_smooth:
                                            smooth_in = _smooth_gpu(
                                                input_gray_b[i : i + 1].unsqueeze(1),
                                                smooth_sigma=sigma,
                                                smooth_scale=ds,
                                            ).squeeze().detach().cpu().numpy()
                                            save_grayscale_png(
                                                smooth_in,
                                                os.path.join(
                                                    input_root,
                                                    f"img_{global_img_idx:05d}_smooth.png",
                                                ),
                                            )
                                            saved_input_smooth.add(
                                                (sigma, ds, global_img_idx)
                                            )
                                        save_binary_png(
                                            ref_np,
                                            os.path.join(
                                                input_root, f"img_{global_img_idx:05d}.png"
                                            ),
                                        )
                                        saved_input_edges.add(in_key)
                                    maps_root = os.path.join(setting_root, "grad_maps", safe)
                                    save_grayscale_png(
                                        merged_np,
                                        os.path.join(
                                            maps_root, f"img_{global_img_idx:05d}_merged.png"
                                        ),
                                    )
                                    smooth_merged = _smooth_gpu(
                                        merged[i : i + 1].unsqueeze(1),
                                        smooth_sigma=sigma,
                                        smooth_scale=ds,
                                    ).squeeze().detach().cpu().numpy()
                                    save_grayscale_png(
                                        smooth_merged,
                                        os.path.join(
                                            maps_root,
                                            f"img_{global_img_idx:05d}_merged_smooth.png",
                                        ),
                                    )
                                    save_binary_png(
                                        grad_np,
                                        os.path.join(
                                            maps_root,
                                            f"img_{global_img_idx:05d}_merged_edge.png",
                                        ),
                                    )
                                    n_ch = int(glab.shape[1])
                                    for ch in range(n_ch):
                                        save_label_map_png(
                                            glab[i, ch].detach().cpu().numpy(),
                                            os.path.join(
                                                maps_root,
                                                f"img_{global_img_idx:05d}_ch{ch}.png",
                                            ),
                                        )
                            buf = score_bufs[key][model_id]
                            for metric_name in metrics:
                                buf[metric_name][img_offset : img_offset + b] = (
                                    GPU_METRICS[metric_name](grad_edge, ref_edge).cpu()
                                )

                img_offset += b
                scoring_time_sec += time.perf_counter() - t1
                del xb_gpu, input_gray_b, edge_refs_b, g
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if img_offset != n_samples:
            raise RuntimeError(
                f"image count mismatch for {model_id}: {img_offset} vs {n_samples}"
            )

    for key in configs:
        for mid in model_names:
            for metric_name in metrics:
                out[key][mid][metric_name] = score_bufs[key][mid][metric_name].tolist()

    return grad_time_sec, scoring_time_sec
