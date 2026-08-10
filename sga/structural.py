"""Structural post-processing: ternary sensitivity representation, smoothing,
central-difference structural maps, and the similarity measures."""

import math
import os
import random
import time
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
from torch import Tensor

from common.utils import chunk_tensor


def rgb2gray(x: np.ndarray) -> np.ndarray:
    """[B,3,H,W] in [0,1] -> [B,1,H,W]."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b


def chw01_to_gray_hw(x: np.ndarray) -> np.ndarray:
    x_b = x[np.newaxis, ...]
    if x_b.shape[1] == 1:
        x_b = np.repeat(x_b, 3, axis=1)
    return rgb2gray(x_b)[0, 0]


LABEL_TO_GRAY = {-1: 0, 0: 128, 1: 255}

# ("abs", value) absolute |g| cutoff; ("top_pct", x) keep top x% of |g| over all channels
ThresholdSpec = Tuple[str, float]


def gradients_to_labels(g: np.ndarray, threshold: float) -> np.ndarray:
    """g: [C, H, W] -> labels int8 in {-1, 0, 1}, same shape."""
    g = np.asarray(g, dtype=np.float64)
    abs_g = np.abs(g)
    labels = np.zeros(g.shape, dtype=np.int8)
    low = abs_g < threshold
    labels[low] = 0
    high = ~low
    labels[high & (g > 0)] = 1
    labels[high & (g < 0)] = -1
    return labels


def gradients_to_labels_top_pct(g: np.ndarray, top_pct: float) -> np.ndarray:
    """Label non-zero only for top top_pct%% of |g| pooled over all channels."""
    g = np.asarray(g, dtype=np.float64)
    if top_pct <= 0:
        return np.zeros(g.shape, dtype=np.int8)
    abs_g = np.abs(g)
    if top_pct >= 100:
        labels = np.zeros(g.shape, dtype=np.int8)
        labels[g > 0] = 1
        labels[g < 0] = -1
        return labels
    flat = abs_g.ravel()
    n = flat.size
    k = max(1, int(np.ceil(n * top_pct / 100.0)))
    thr = float(np.partition(flat, n - k)[n - k])
    high = abs_g >= thr
    labels = np.zeros(g.shape, dtype=np.int8)
    labels[high & (g > 0)] = 1
    labels[high & (g < 0)] = -1
    return labels


def gradients_to_labels_from_spec(g: np.ndarray, spec: ThresholdSpec) -> np.ndarray:
    mode, val = spec
    if mode == "abs":
        return gradients_to_labels(g, val)
    if mode == "top_pct":
        return gradients_to_labels_top_pct(g, val)
    raise ValueError(f"Unknown threshold mode {mode!r}")


DEFAULT_THRESHOLD_ABS = 0.0
DEFAULT_SMOOTH_SIGMA = 0.0
DEFAULT_SMOOTH_SCALE = 1

THRESHOLD_BASELINE: ThresholdSpec = ("abs", DEFAULT_THRESHOLD_ABS)
SMOOTH_BASELINE: Tuple[float, int] = (DEFAULT_SMOOTH_SIGMA, DEFAULT_SMOOTH_SCALE)


def _dedupe_threshold_specs(specs: List[ThresholdSpec]) -> Tuple[ThresholdSpec, ...]:
    seen: set[ThresholdSpec] = set()
    out: List[ThresholdSpec] = []
    for spec in specs:
        if spec not in seen:
            out.append(spec)
            seen.add(spec)
    return tuple(out)


def resolve_threshold_specs(
    abs_values: Sequence[float] = (),
    top_pcts: Sequence[float] = (),
) -> Tuple[ThresholdSpec, ...]:
    """
    Sweep only explicitly passed threshold values (including off: abs=0, top_pct=100).
    Empty on both sides -> baseline abs:0.
    """
    abs_list = [float(v) for v in abs_values]
    top_list = [float(v) for v in top_pcts]
    if not abs_list and not top_list:
        return (THRESHOLD_BASELINE,)

    specs: List[ThresholdSpec] = []
    for v in abs_list:
        specs.append(("abs", v))
    for v in top_list:
        if v <= 0 or v > 100:
            raise ValueError(f"top_pct must be in (0, 100], got {v}")
        specs.append(("top_pct", v))
    return _dedupe_threshold_specs(specs)


def threshold_sort_key(spec: ThresholdSpec) -> Tuple[int, float]:
    mode, val = spec
    return (0 if mode == "abs" else 1, val)


def threshold_spec_dir_tag(spec: ThresholdSpec) -> str:
    mode, val = spec
    if mode == "abs":
        return f"thr{val:g}"
    return f"top{val:g}pct"


def threshold_spec_label(spec: ThresholdSpec) -> str:
    mode, val = spec
    if mode == "abs":
        return f"abs:{val:g}"
    return f"top_pct:{val:g}"


def labels_to_grayscale_u8(lab: np.ndarray) -> np.ndarray:
    """[H, W] labels -> uint8 grayscale."""
    out = np.zeros(lab.shape, dtype=np.uint8)
    for label, gray in LABEL_TO_GRAY.items():
        out[lab == label] = gray
    return out


def save_label_map_png(lab_hw: np.ndarray, out_path: str) -> None:
    """Save exact uint8 gray levels (0/128/255); avoid matplotlib colormap re-quantization."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(labels_to_grayscale_u8(lab_hw), mode="L").save(out_path)


def gaussian_smooth(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian blur on [H,W] gray in [0,1]. sigma<=0: no-op."""
    if sigma <= 0:
        return gray.astype(np.float64)
    g = gray.astype(np.float64)
    size = int(max(3, 2 * math.ceil(3.0 * sigma) + 1))
    if size % 2 == 0:
        size += 1
    kernel = _gaussian_window(size, sigma=sigma)
    return _conv2_same(g, kernel)


SmoothMode = Literal["none", "gaussian", "structural"]


def resolve_smooth_mode(
    smooth_sigma: float,
    smooth_scale: int,
) -> SmoothMode:
    """Pick psi from the settings; the two non-identity smoothing operators are exclusive."""
    if smooth_scale < 1:
        raise ValueError(f"smooth_scale must be >= 1, got {smooth_scale}")
    use_gaussian = smooth_sigma > 0
    use_structural = smooth_scale > 1
    if use_gaussian and use_structural:
        raise ValueError(
            "Enable only one smoothing operator: --smooth_sigma > 0 OR "
            "--smooth_scale > 1, not both."
        )
    if use_structural:
        return "structural"
    if use_gaussian:
        return "gaussian"
    return "none"


def structural_smooth(gray: np.ndarray, downscale_factor: int) -> np.ndarray:
    """Bilinear downscale then upscale (structural low-pass). factor<=1: no-op."""
    if downscale_factor <= 1:
        return gray.astype(np.float64)
    t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    h, w = t.shape[2], t.shape[3]
    t_down = F.interpolate(
        t,
        size=(h // downscale_factor, w // downscale_factor),
        mode="bilinear",
        align_corners=False,
    )
    t_up = F.interpolate(
        t_down,
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    )
    return t_up.squeeze().numpy().astype(np.float64)


def apply_smooth(
    gray: np.ndarray,
    *,
    smooth_sigma: float = 0.0,
    smooth_scale: int = 1,
) -> np.ndarray:
    mode = resolve_smooth_mode(smooth_sigma, smooth_scale)
    if mode == "gaussian":
        return gaussian_smooth(gray, smooth_sigma)
    if mode == "structural":
        return structural_smooth(gray, smooth_scale)
    return gray.astype(np.float64)


def edge_magnitude(
    gray: np.ndarray,
    *,
    smooth_sigma: float = 0.0,
    smooth_scale: int = 1,
) -> np.ndarray:
    """[H,W] gray in [0,1] -> central-difference edge magnitude E(z), after psi."""
    g = apply_smooth(
        gray,
        smooth_sigma=smooth_sigma,
        smooth_scale=smooth_scale,
    )
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.hypot(gx, gy)


def binary_edge_from_magnitude(
    mag: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Edge magnitude -> binary structural map B_e(z) in {0,1}."""
    pos = mag[mag > 0]
    if pos.size == 0:
        return np.zeros_like(mag, dtype=np.float64)
    t = float(np.percentile(pos, percentile))
    return (mag >= t).astype(np.float64)


def edge_map(
    gray: np.ndarray,
    percentile: float = 85.0,
    *,
    smooth_sigma: float = 0.0,
    smooth_scale: int = 1,
) -> np.ndarray:
    """[H,W] gray in [0,1] -> binary edge map in {0,1}."""
    return binary_edge_from_magnitude(
        edge_magnitude(
            gray,
            smooth_sigma=smooth_sigma,
            smooth_scale=smooth_scale,
        ),
        percentile,
    )


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary [H,W] morphological dilation (square kernel)."""
    if radius <= 0:
        return mask.astype(np.float64)
    m = mask.astype(np.float64)
    k = 2 * radius + 1
    return maximum_filter(m, size=k, mode="constant", cval=0.0)


def dice_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    af = a.astype(bool).ravel()
    bf = b.astype(bool).ravel()
    inter = float(np.logical_and(af, bf).sum())
    return (2.0 * inter) / (float(af.sum()) + float(bf.sum()) + eps)


def iou_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    af = a.astype(bool).ravel()
    bf = b.astype(bool).ravel()
    inter = float(np.logical_and(af, bf).sum())
    union = float(np.logical_or(af, bf).sum())
    return inter / (union + eps)


def _gaussian_window(size: int, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    w = np.outer(g, g)
    return w / w.sum()


def _conv2_same(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    xp = np.pad(x, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    return convolve2d(xp, kernel, mode="valid")


T = TypeVar("T")


def progress_bar(
    iterable: Iterable[T],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "it",
    disable: bool = False,
) -> Iterator[T]:
    if disable:
        return iter(iterable)
    from tqdm import tqdm

    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False, dynamic_ncols=True)


def iter_grad_batches(
    x: Tensor,
    y: Tensor,
    grad_batch_size: int,
    *,
    show_progress: bool,
    single_model: bool,
) -> Iterator[Tuple[Tensor, Tensor]]:
    """Yield ``(xb, yb)`` chunks; show ram-batch tqdm when scoring one model."""
    n_samples = int(x.shape[0])
    bs = grad_batch_size or n_samples
    n_chunks = max(1, (n_samples + bs - 1) // bs)
    pairs = zip(chunk_tensor(x, bs), chunk_tensor(y, bs))
    show = show_progress and single_model and n_chunks > 1
    return progress_bar(
        pairs,
        desc="Score ram batches",
        total=n_chunks,
        unit="batch",
        disable=not show,
    )


def ssim_similarity(
    a: np.ndarray,
    b: np.ndarray,
    *,
    window_size: int = 11,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """SSIM between two [H,W] maps in [0,1]."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    window = _gaussian_window(window_size)
    mu_a = _conv2_same(a, window)
    mu_b = _conv2_same(b, window)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = _conv2_same(a * a, window) - mu_a2
    sigma_b2 = _conv2_same(b * b, window) - mu_b2
    sigma_ab = _conv2_same(a * b, window) - mu_ab
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    return float(np.mean(num / (den + eps)))


METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "dice": dice_similarity,
    "ssim": ssim_similarity,
    "iou": iou_similarity,
}

REPORT_METRICS = ("dice", "ssim")

# Ranking score = mean of listed base metrics (computed once per sample).
RANKING_COMBOS: Dict[str, Tuple[str, ...]] = {
    "dice": ("dice",),
    "ssim": ("ssim",),
    "dice_ssim": ("dice", "ssim"),
}
ALL_RANKING_METRICS = ("dice", "ssim", "dice_ssim")


def labels_merged_gray01(lab_chw: np.ndarray) -> np.ndarray:
    """Merge per-channel label maps (0/128/255) to single gray in [0,1] via luminance."""
    ch_u8 = [
        labels_to_grayscale_u8(lab_chw[c]).astype(np.float64) / 255.0
        for c in range(lab_chw.shape[0])
    ]
    return chw01_to_gray_hw(np.stack(ch_u8, axis=0))


def batch_raw_gradients(
    model: nn.Module,
    xb: torch.Tensor,
    yb: torch.Tensor,
    device: torch.device,
    *,
    conditioning: str,
    amp: bool = False,
) -> torch.Tensor:
    """Input gradients conditioned on the dataset label (ground_truth) or the model prediction (pseudo)."""
    y_gt = yb.to(device).long()
    x = xb.detach().clone().to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    if amp:
        with torch.cuda.amp.autocast():
            logits = model(x)
            y = logits.argmax(dim=1) if conditioning == "pseudo" else y_gt
            loss = F.cross_entropy(logits, y, reduction="sum")
    else:
        logits = model(x)
        y = logits.argmax(dim=1) if conditioning == "pseudo" else y_gt
        loss = F.cross_entropy(logits, y, reduction="sum")
    loss.backward()
    return x.grad.detach().view(x.shape[0], -1)


def save_binary_png(mask: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    u8 = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_path)


def save_grayscale_png(gray01: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr = np.squeeze(np.asarray(gray01, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"save_grayscale_png expects 2D grayscale, got shape {arr.shape}")
    u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(out_path)


def gradient_chw_to_rgb01(g: np.ndarray) -> np.ndarray:
    """Raw [C,H,W] gradient -> [H,W,3] RGB in [0,1] (no luminance merge).

    Uses the first three channels (repeat/pad when C<3). Per-image scale by max |g|;
    zero gradient -> 0.5 gray, signed components map to [0, 1].
    """
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 3:
        raise ValueError(f"gradient_chw_to_rgb01 expects [C,H,W], got shape {g.shape}")
    c = int(g.shape[0])
    if c == 1:
        chw = np.repeat(g, 3, axis=0)
    elif c == 2:
        chw = np.concatenate([g, np.zeros((1, g.shape[1], g.shape[2]), dtype=g.dtype)], axis=0)
    else:
        chw = g[:3]
    scale = float(np.max(np.abs(chw)))
    if scale <= 0.0:
        return np.full((int(chw.shape[1]), int(chw.shape[2]), 3), 0.5, dtype=np.float64)
    rgb = np.transpose(0.5 + 0.5 * (chw / scale), (1, 2, 0))
    return np.clip(rgb, 0.0, 1.0)


def save_gradient_rgb_png(g_chw: np.ndarray, out_path: str) -> None:
    """Save raw per-channel gradient as RGB PNG (values in [0,1])."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rgb = gradient_chw_to_rgb01(g_chw)
    u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(out_path)


def set_reproducible(seed: int) -> None:
    """Best-effort deterministic PyTorch/CUDA for repeatable gradients."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def base_metrics_for_grad(metric_arg: Optional[str]) -> Tuple[str, ...]:
    """Base similarities to compute during grad pass."""
    if metric_arg is None:
        return REPORT_METRICS
    if metric_arg in RANKING_COMBOS:
        parts = RANKING_COMBOS[metric_arg]
        return tuple(dict.fromkeys(parts))
    if metric_arg in METRICS:
        return (metric_arg,)
    raise ValueError(f"Unknown metric {metric_arg!r}")


ConfigKey = Tuple[float, float, int, ThresholdSpec]  # edge, sigma, downscale, threshold
ScoreMatrixKey = Tuple[float, float, int, ThresholdSpec, str]
SmoothKey = Tuple[float, float]  # smooth_sigma, smooth_scale
PerImageScores = Dict[str, Dict[str, List[float]]]
PerConfigPerImage = Dict[ConfigKey, PerImageScores]


def _dedupe_smooth_pairs(pairs: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
    seen: set[Tuple[float, int]] = set()
    out: List[Tuple[float, int]] = []
    for sigma, ds in pairs:
        key = (float(sigma), int(ds))
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def iter_valid_smooth_pairs(
    sigmas: Sequence[float] = (),
    downscales: Sequence[int] = (),
) -> List[Tuple[float, int]]:
    """
    Sweep only explicitly passed smooth values (including off: sigma=0, downscale=1).
    Empty on both sides -> baseline (0, 1).
    """
    sigma_list = [float(s) for s in sigmas]
    ds_list = [int(d) for d in downscales]
    if not sigma_list and not ds_list:
        return [SMOOTH_BASELINE]

    pairs: List[Tuple[float, int]] = []
    for sigma in sigma_list:
        pairs.append((sigma, DEFAULT_SMOOTH_SCALE))
    for ds in ds_list:
        pairs.append((DEFAULT_SMOOTH_SIGMA, ds))
    return _dedupe_smooth_pairs(pairs)


def iter_configurations(
    edges: Sequence[float],
    sigmas: Sequence[float],
    downscales: Sequence[int],
    threshold_specs: Sequence[ThresholdSpec],
) -> List[ConfigKey]:
    smooth_pairs = iter_valid_smooth_pairs(sigmas, downscales)
    return [
        (float(ep), sigma, ds, thr)
        for ep in edges
        for sigma, ds in smooth_pairs
        for thr in threshold_specs
    ]


def setting_dir_name(
    edge: float,
    sigma: float,
    downscale: int,
    threshold: ThresholdSpec,
) -> str:
    return (
        f"ep{edge:g}_sigma{sigma:g}_ds{downscale}_"
        f"{threshold_spec_dir_tag(threshold)}"
    )


def n_loaded_images(per_image_by_model: PerImageScores) -> int:
    if not per_image_by_model:
        return 0
    first = next(iter(per_image_by_model.values()))
    first_metric = next(iter(first.values()))
    return len(first_metric)


def build_metric_score_matrix(
    per_image_by_model: PerImageScores,
    model_ids: List[str],
    ranking_metric: str,
) -> np.ndarray:
    n_images = n_loaded_images(per_image_by_model)
    cols: List[np.ndarray] = []
    for mid in model_ids:
        per_img = per_image_by_model[mid]
        if ranking_metric in per_img:
            cols.append(np.asarray(per_img[ranking_metric], dtype=np.float64))
        elif ranking_metric in RANKING_COMBOS:
            parts = RANKING_COMBOS[ranking_metric]
            stacked = np.stack(
                [np.asarray(per_img[p], dtype=np.float64) for p in parts], axis=0
            )
            cols.append(stacked.mean(axis=0))
        else:
            raise KeyError(f"Unknown ranking metric {ranking_metric!r}")
    mat = np.stack(cols, axis=1)
    if mat.shape[0] != n_images:
        raise RuntimeError("per-image score length mismatch across models")
    return mat


def precompute_score_matrices(
    per_config_scores: PerConfigPerImage,
    model_ids: List[str],
    configs: Sequence[ConfigKey],
    metrics: Sequence[str],
) -> Dict[ScoreMatrixKey, np.ndarray]:
    return {
        (edge, sigma, ds, thr, metric): build_metric_score_matrix(
            per_config_scores[(edge, sigma, ds, thr)], model_ids, metric
        )
        for edge, sigma, ds, thr in configs
        for metric in metrics
    }


def similarities_from_score_matrix(
    score_matrix: np.ndarray,
    model_ids: List[str],
    image_indices: Sequence[int],
) -> Dict[str, float]:
    if not image_indices:
        return {mid: float("nan") for mid in model_ids}
    means = score_matrix[np.asarray(image_indices, dtype=np.intp), :].mean(axis=0)
    return {mid: float(means[j]) for j, mid in enumerate(model_ids)}


def _anchor_pairwise_reference_flags(
    anchor_ids: List[str],
    reference_by_id: Dict[str, Dict[str, object]],
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    from common.reference import reference_relation, reference_tied

    n = len(anchor_ids)
    pairs: List[Tuple[int, int]] = []
    want_first_better: List[bool] = []
    gt_robust_tied_flags: List[bool] = []
    for i in range(n):
        for j in range(i + 1, n):
            entry_i = reference_by_id[anchor_ids[i]]
            entry_j = reference_by_id[anchor_ids[j]]
            tied = reference_tied(entry_i, entry_j)
            pairs.append((i, j))
            gt_robust_tied_flags.append(tied)
            if tied:
                want_first_better.append(False)
            else:
                rel = reference_relation(entry_i, entry_j)
                assert rel is not None
                want_first_better.append(rel == "a_better")
    return (
        pairs,
        np.asarray(want_first_better, dtype=bool),
        np.asarray(gt_robust_tied_flags, dtype=bool),
    )


def compute_per_probe_alignment(
    *,
    model_names: List[str],
    x: Tensor,
    y: Tensor,
    input_grays: List[np.ndarray],
    configurations: Sequence[ConfigKey],
    metrics: Tuple[str, ...],
    device: torch.device,
    conditioning: str,
    amp: bool,
    dilate: int,
    grad_batch_size: Optional[int],
    models: Optional[List[nn.Module]] = None,
    model_loader: Optional[Callable[[str], nn.Module]] = None,
    start_idx: int = 0,
    save_maps: bool = False,
    maps_settings_root: Optional[str] = None,
    label_cache: Optional[Dict[Tuple[str, int, ThresholdSpec], np.ndarray]] = None,
    show_progress: bool = True,
    use_gpu: Optional[bool] = None,
    vqa_samples: Optional[Sequence[object]] = None,
) -> PerConfigPerImage:
    """One gradient pass per model; alignment scores for every configuration.

    Pass ``model_loader`` to load one model at a time (lower GPU memory), or
    ``models`` to use an already loaded pool.

    GPU path (default on CUDA): gradients and structural maps stay on device; only scores
    (and optional PNG maps with ``--save_maps``) are kept on CPU. ``x`` may stay
    on CPU; input edge refs are built per gradient batch. Disabled when ``label_cache``
    is set or ``use_gpu=False`` (``--cpu``).
    """
    if (models is None) == (model_loader is None):
        raise ValueError("Provide exactly one of models or model_loader")
    if models is not None and len(models) != len(model_names):
        raise ValueError("models length must match model_names")

    want_gpu = use_gpu if use_gpu is not None else device.type == "cuda"
    gpu_ok = (
        want_gpu
        and device.type == "cuda"
        and label_cache is None
        and model_loader is not None
    )
    if gpu_ok:
        from sga.gpu import compute_per_probe_alignment_gpu

        n_samples = x.shape[0]
        configs = list(configurations)
        smooth_keys: List[SmoothKey] = sorted({(sigma, ds) for _, sigma, ds, _ in configs})
        grad_bs = grad_batch_size or n_samples
        out: PerConfigPerImage = {
            key: {mid: {m: [] for m in metrics} for mid in model_names}
            for key in configs
        }
        if show_progress:
            print(
                f"Scoring: GPU path ({len(configs)} configurations, "
                f"{len(smooth_keys)} smooth groups, grad_batch={grad_bs})",
                flush=True,
            )
        grad_time_sec, scoring_time_sec = compute_per_probe_alignment_gpu(
            model_names=model_names,
            x=x,
            y=y,
            configurations=configs,
            metrics=metrics,
            device=device,
            conditioning=conditioning,
            amp=amp,
            dilate=dilate,
            grad_batch_size=grad_bs,
            model_loader=model_loader,
            out=out,
            show_progress=show_progress,
            start_idx=start_idx,
            save_maps=save_maps,
            maps_settings_root=maps_settings_root,
            vqa_samples=vqa_samples,
        )
        if show_progress:
            print(
                f"Scoring profile (GPU): grad={grad_time_sec:.1f}s "
                f"scoring={scoring_time_sec:.1f}s",
                flush=True,
            )
        return out

    if show_progress:
        reasons: List[str] = []
        if device.type != "cuda":
            reasons.append("no_cuda")
        elif use_gpu is False:
            reasons.append("--cpu")
        elif label_cache is not None:
            reasons.append("label_cache")
        elif model_loader is None:
            reasons.append("no_model_loader")
        note = ", ".join(reasons) if reasons else "disabled"
        print(f"Scoring: CPU path ({note})", flush=True)

    n_samples = x.shape[0]
    configs = list(configurations)
    threshold_list = tuple(
        sorted({thr for _, _, _, thr in configs}, key=threshold_sort_key)
    )
    smooth_keys: List[SmoothKey] = sorted({(sigma, ds) for _, sigma, ds, _ in configs})
    edges_for_smooth: Dict[SmoothKey, Tuple[float, ...]] = {
        sk: tuple(sorted({ep for ep, s, d, _ in configs if (s, d) == sk}))
        for sk in smooth_keys
    }
    n_configuration_keys = len(threshold_list) * len(smooth_keys) * sum(
        len(v) for v in edges_for_smooth.values()
    )

    out: PerConfigPerImage = {
        key: {mid: {m: [] for m in metrics} for mid in model_names}
        for key in configs
    }
    edge_refs: Dict[SmoothKey, Dict[float, List[np.ndarray]]] = {}
    smooth_iter = progress_bar(
        smooth_keys,
        desc="Input edges",
        total=len(smooth_keys),
        unit="smooth",
        disable=not show_progress,
    )
    for sigma, ds in smooth_iter:
        edge_refs[(sigma, ds)] = {}
        for ep in edges_for_smooth[(sigma, ds)]:
            edge_refs[(sigma, ds)][ep] = [
                dilate_binary(
                    edge_map(
                        gray,
                        ep,
                        smooth_sigma=sigma,
                        smooth_scale=ds,
                    ),
                    dilate,
                )
                for gray in input_grays
            ]
    grad_bs = grad_batch_size or n_samples
    saved_input_edges: set[Tuple[float, float, int, float, int]] = set()
    saved_input_smooth: set[Tuple[float, float, int]] = set()
    saved_input_gray: set[int] = set()
    saved_raw_grad: set[Tuple[str, int]] = set()
    grad_time_sec = 0.0
    cpu_time_sec = 0.0

    show_model_progress = show_progress and len(model_names) > 1
    single_model = len(model_names) == 1
    model_iter = progress_bar(
        range(len(model_names)),
        desc="Load+score models",
        total=len(model_names),
        unit="model",
        disable=not show_model_progress,
    )
    for mi in model_iter:
        model_id = model_names[mi]
        if model_loader is not None:
            model = model_loader(model_id)
        else:
            model = models[mi]  # type: ignore[index]
        safe = model_id.replace(os.sep, "_").replace(" ", "_")
        img_offset = 0
        try:
            for xb, yb in iter_grad_batches(
                x,
                y,
                grad_bs,
                show_progress=show_progress,
                single_model=single_model,
            ):
                t_grad = time.perf_counter()
                if vqa_samples is not None:
                    from sga.vqa_grad import (
                        batch_raw_gradients_vqa,
                        resolve_vqa_attack_answers,
                    )

                    sb = list(vqa_samples[img_offset : img_offset + xb.shape[0]])
                    qs, ans, _ = resolve_vqa_attack_answers(
                        model, xb.to(device), sb, conditioning
                    )
                    g = batch_raw_gradients_vqa(
                        model, xb.to(device), qs, ans, amp=amp
                    )
                    b, c, h, w = g.shape
                    g = g.detach().cpu().numpy()
                else:
                    g = batch_raw_gradients(
                        model, xb, yb, device, conditioning=conditioning, amp=amp
                    )
                    b, d = g.shape
                    c, h, w = xb.shape[1], xb.shape[2], xb.shape[3]
                    if c * h * w != d:
                        raise ValueError(f"Cannot reshape grad dim {d} to {c}x{h}x{w}")
                    g = g.view(b, c, h, w).detach().cpu().numpy()
                grad_time_sec += time.perf_counter() - t_grad

                t_cpu = time.perf_counter()
                for i in range(b):
                    global_img_idx = start_idx + img_offset
                    if save_maps and maps_settings_root:
                        raw_key = (model_id, global_img_idx)
                        if raw_key not in saved_raw_grad:
                            save_gradient_rgb_png(
                                g[i],
                                os.path.join(
                                    maps_settings_root,
                                    "grad_maps",
                                    safe,
                                    f"img_{global_img_idx:05d}_grad.png",
                                ),
                            )
                            saved_raw_grad.add(raw_key)
                    labels_by_thr = {
                        thr: gradients_to_labels_from_spec(g[i], thr)
                        for thr in threshold_list
                    }
                    if label_cache is not None:
                        for thr, glab in labels_by_thr.items():
                            label_cache[(model_id, global_img_idx, thr)] = glab

                    merged_by_thr = {
                        thr: labels_merged_gray01(lab) for thr, lab in labels_by_thr.items()
                    }
                    grad_edge_cache: Dict[ConfigKey, np.ndarray] = {}
                    for thr in threshold_list:
                        merged = merged_by_thr[thr]
                        for sigma, ds in smooth_keys:
                            mag = edge_magnitude(
                                merged,
                                smooth_sigma=sigma,
                                smooth_scale=ds,
                            )
                            for ep in edges_for_smooth[(sigma, ds)]:
                                key = (ep, sigma, ds, thr)
                                grad_edge_cache[key] = dilate_binary(
                                    binary_edge_from_magnitude(mag, ep),
                                    dilate,
                                )

                    for key in configs:
                        ep, sigma, ds, thr = key
                        grad_edge = grad_edge_cache[key]
                        ref_edge = edge_refs[(sigma, ds)][ep][img_offset]
                        scores = out[key][model_id]
                        for metric_name in metrics:
                            scores[metric_name].append(
                                float(METRICS[metric_name](grad_edge, ref_edge))
                            )

                        if save_maps and maps_settings_root:
                            setting_root = os.path.join(
                                maps_settings_root,
                                setting_dir_name(ep, sigma, ds, thr),
                            )
                            in_key = (sigma, ds, ep, global_img_idx)
                            if in_key not in saved_input_edges:
                                input_root = os.path.join(setting_root, "input_edges")
                                if global_img_idx not in saved_input_gray:
                                    save_grayscale_png(
                                        input_grays[global_img_idx],
                                        os.path.join(
                                            input_root,
                                            f"img_{global_img_idx:05d}_gray.png",
                                        ),
                                    )
                                    saved_input_gray.add(global_img_idx)
                                if (sigma, ds, global_img_idx) not in saved_input_smooth:
                                    save_grayscale_png(
                                        apply_smooth(
                                            input_grays[global_img_idx],
                                            smooth_sigma=sigma,
                                            smooth_scale=ds,
                                        ),
                                        os.path.join(
                                            input_root,
                                            f"img_{global_img_idx:05d}_smooth.png",
                                        ),
                                    )
                                    saved_input_smooth.add((sigma, ds, global_img_idx))
                                save_binary_png(
                                    ref_edge,
                                    os.path.join(
                                        input_root, f"img_{global_img_idx:05d}.png"
                                    ),
                                )
                                saved_input_edges.add(in_key)
                            glab = labels_by_thr[thr]
                            maps_root = os.path.join(setting_root, "grad_maps", safe)
                            save_grayscale_png(
                                merged_by_thr[thr],
                                os.path.join(
                                    maps_root, f"img_{global_img_idx:05d}_merged.png"
                                ),
                            )
                            save_grayscale_png(
                                apply_smooth(
                                    merged_by_thr[thr],
                                    smooth_sigma=sigma,
                                    smooth_scale=ds,
                                ),
                                os.path.join(
                                    maps_root,
                                    f"img_{global_img_idx:05d}_merged_smooth.png",
                                ),
                            )
                            save_binary_png(
                                grad_edge,
                                os.path.join(
                                    maps_root,
                                    f"img_{global_img_idx:05d}_merged_edge.png"
                                ),
                            )
                            for ch in range(glab.shape[0]):
                                save_label_map_png(
                                    glab[ch],
                                    os.path.join(
                                        maps_root,
                                        f"img_{global_img_idx:05d}_ch{ch}.png"
                                    ),
                                )

                    img_offset += 1
                cpu_time_sec += time.perf_counter() - t_cpu
        finally:
            if model_loader is not None:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if img_offset != n_samples:
            raise RuntimeError(
                f"image count mismatch for {model_id}: {img_offset} vs {n_samples}"
            )

    if show_progress:
        print(
            f"Scoring profile: grad={grad_time_sec:.1f}s "
            f"cpu_scoring={cpu_time_sec:.1f}s "
            f"(grad once per model; cpu ~{n_configuration_keys} edge x thr x smooth "
            f"x {len(metrics)} metrics per image)",
            flush=True,
        )

    return out


def anchor_checks_for_target(
    target_id: str,
    anchor_ids: Sequence[str],
    reference_by_id: Dict[str, Dict[str, object]],
) -> List[Tuple[str, Optional[bool], bool]]:
    from common.reference import reference_tied

    checks: List[Tuple[str, Optional[bool], bool]] = []
    target_entry = reference_by_id[target_id]
    for anchor_id in anchor_ids:
        if anchor_id == target_id:
            continue
        anchor_entry = reference_by_id[anchor_id]
        tied = reference_tied(target_entry, anchor_entry)
        want_target_better = None
        if not tied:
            want_target_better = float(target_entry["robust_acc"]) > float(
                anchor_entry["robust_acc"]
            )
        checks.append((anchor_id, want_target_better, tied))
    return checks


def pairwise_agreement_for_target(
    target_id: str,
    similarities: Dict[str, float],
    anchor_checks: List[Tuple[str, Optional[bool], bool]],
) -> Tuple[float, int, int]:
    if not anchor_checks:
        return float("nan"), 0, 0
    sim_t = similarities[target_id]
    correct = 0
    for anchor_id, want_target_better, reference_tie in anchor_checks:
        sim_a = similarities[anchor_id]
        if reference_tie:
            correct += 1
            continue
        if sim_t > sim_a:
            if want_target_better:
                correct += 1
        elif sim_t < sim_a:
            if want_target_better is False:
                correct += 1
    n_anchors = len(anchor_checks)
    return float(correct / n_anchors), correct, n_anchors


def iter_ranking_batches(
    n_samples: int,
    batch_size: int,
    start_idx: int,
) -> List[Tuple[int, int, int, int]]:
    batches: List[Tuple[int, int, int, int]] = []
    for bi, batch_start in enumerate(range(0, n_samples, batch_size)):
        batch_end = min(batch_start + batch_size, n_samples)
        batches.append(
            (
                bi,
                start_idx + batch_start,
                start_idx + batch_end - 1,
                batch_end - batch_start,
            )
        )
    return batches


def summarize_values(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr, ddof=0)),
    }

