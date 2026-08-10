"""Score-matrix I/O, configuration sweeps, and pairwise-agreement evaluation."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from common.reference import assign_reference_ranks, assign_score_ranks_with_ties
from sga.structural import (
    ALL_RANKING_METRICS,
    ConfigKey,
    METRICS,
    ScoreMatrixKey,
    ThresholdSpec,
    anchor_checks_for_target,
    base_metrics_for_grad,
    iter_configurations,
    resolve_threshold_specs,
    setting_dir_name,
    similarities_from_score_matrix,
    summarize_values,
    pairwise_agreement_for_target,
    threshold_spec_label,
)

ScoreMatrix = Union[np.ndarray, torch.Tensor]

DEFAULT_PROBE_BUDGETS: Tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500)


def parse_probe_budgets(raw: str) -> List[int]:
    sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not sizes:
        raise ValueError("probe_budgets must list at least one positive integer")
    if any(s <= 0 for s in sizes):
        raise ValueError(f"probe_budgets values must be positive, got {sizes}")
    return sorted(set(sizes))


def _tie_break_rng(
    seed: int,
    batch_id: int,
    target_id: str,
) -> np.random.Generator:
    """Deterministic RNG for residual ties (independent of target visit order)."""
    tid_key = 0
    for ch in target_id.encode("utf-8"):
        tid_key = (tid_key * 131 + int(ch)) % (2**32)
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(batch_id), int(tid_key)])
    )


def resolve_rank_eval_device(force_cpu: bool) -> Optional[torch.device]:
    """CUDA for batch aggregation when available; ``None`` means NumPy on CPU."""
    if force_cpu or not torch.cuda.is_available():
        return None
    return torch.device("cuda")


def stack_score_matrices(
    score_matrices: Dict[ScoreMatrixKey, np.ndarray],
    sweep_keys: Sequence[ScoreMatrixKey],
    device: torch.device,
) -> torch.Tensor:
    """Stack setting matrices to ``[K, n_samples, n_models]`` on ``device``."""
    return torch.stack(
        [
            torch.as_tensor(score_matrices[k], device=device, dtype=torch.float32)
            for k in sweep_keys
        ],
        dim=0,
    )


def _format_sweep_key_label(key: ScoreMatrixKey) -> str:
    edge, sigma, ds, thr, metric = key
    return f"{setting_dir_name(edge, sigma, ds, thr)}/{metric}"


def summarize_tie_break_stages(
    per_target_rows: Sequence[Dict[str, object]],
    *,
    n_settings: int,
    unique_by_batch: bool = False,
) -> Dict[str, float]:
    """Mean stage sizes over groups; ``n_settings`` = sweep size.

    Default groups = one per row (typically batch x target). With
    ``unique_by_batch=True``, keep the first row of each ``batch_id``
    (shared-config protocols such as calibrated).
    """
    rows: Sequence[Dict[str, object]] = per_target_rows
    if unique_by_batch and per_target_rows:
        seen: set[object] = set()
        deduped: List[Dict[str, object]] = []
        for r in per_target_rows:
            bid = r.get("batch_id")
            if bid in seen:
                continue
            seen.add(bid)
            deduped.append(r)
        rows = deduped

    n_groups = len(rows)
    if n_groups == 0:
        return {
            "n_settings": float(n_settings),
            "n_groups": 0.0,
            "mean_enter_astd": float("nan"),
            "mean_enter_med": float("nan"),
            "mean_enter_random": float("nan"),
            "frac_enter_astd": float("nan"),
            "frac_enter_med": float("nan"),
            "frac_enter_random": float("nan"),
            "n_groups_astd_active": 0.0,
            "n_groups_med_active": 0.0,
            "n_groups_random_active": 0.0,
        }

    def _col(key: str, fallback: str = "n_tied_at_best") -> List[float]:
        out: List[float] = []
        for r in rows:
            if key in r and r[key] is not None:
                out.append(float(r[key]))
            elif fallback in r:
                out.append(float(r[fallback]))
            else:
                out.append(1.0)
        return out

    enter_astd = _col("n_enter_astd")
    enter_med = _col("n_enter_med", "n_enter_astd")
    enter_random = _col("n_enter_random", "n_enter_med")
    denom = float(max(n_settings, 1))
    return {
        "n_settings": float(n_settings),
        "n_groups": float(n_groups),
        "mean_enter_astd": float(np.mean(enter_astd)),
        "mean_enter_med": float(np.mean(enter_med)),
        "mean_enter_random": float(np.mean(enter_random)),
        "frac_enter_astd": float(np.mean(enter_astd)) / denom,
        "frac_enter_med": float(np.mean(enter_med)) / denom,
        "frac_enter_random": float(np.mean(enter_random)) / denom,
        # active = stage entered with >1 survivors (tie still unresolved)
        "n_groups_astd_active": float(sum(1 for x in enter_astd if x > 1)),
        "n_groups_med_active": float(sum(1 for x in enter_med if x > 1)),
        "n_groups_random_active": float(sum(1 for x in enter_random if x > 1)),
    }


def format_tie_break_stage_log(
    stats: Dict[str, float],
    *,
    group_label: str = "batch x target groups",
    include_active: bool = True,
) -> str:
    """One-line human log for stage averages vs sweep size."""
    n_set = int(stats["n_settings"])
    n_g = int(stats["n_groups"])
    msg = (
        f"tie_break stages (mean over {n_g} {group_label}; "
        f"n_settings={n_set}): "
        f"enter_astd={stats['mean_enter_astd']:.2f} "
        f"({stats['mean_enter_astd']:.1f}/{n_set}), "
        f"enter_med={stats['mean_enter_med']:.2f} "
        f"({stats['mean_enter_med']:.1f}/{n_set}), "
        f"enter_random={stats['mean_enter_random']:.2f} "
        f"({stats['mean_enter_random']:.1f}/{n_set})"
    )
    if include_active:
        msg += (
            f"; astd_active={int(stats['n_groups_astd_active'])}/{n_g}, "
            f"med_active={int(stats['n_groups_med_active'])}/{n_g}, "
            f"random_active={int(stats['n_groups_random_active'])}/{n_g}"
        )
    return msg


def _is_full_index_range(candidates: Sequence[int], n: int) -> bool:
    return n > 0 and len(candidates) == n and candidates[0] == 0 and candidates[-1] == n - 1


def _indices_at_extreme_tensor(
    values: torch.Tensor,
    candidate_idx: Optional[torch.Tensor] = None,
    *,
    sense: str = "max",
    atol: float = 1e-9,
) -> torch.Tensor:
    """GPU indices whose ``values`` equal the min/max among ``candidate_idx`` (or all)."""
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    if candidate_idx is None:
        sub = values
        cand = torch.arange(values.numel(), device=values.device, dtype=torch.long)
    else:
        cand = candidate_idx
        sub = values.index_select(0, cand)
    best = sub.max() if sense == "max" else sub.min()
    keep = torch.isclose(sub, best, rtol=0.0, atol=atol)
    return cand[keep]


def _indices_at_max_tensor(
    values: torch.Tensor,
    candidate_idx: Optional[torch.Tensor] = None,
    *,
    atol: float = 1e-9,
) -> torch.Tensor:
    return _indices_at_extreme_tensor(
        values, candidate_idx, sense="max", atol=atol
    )


def _indices_at_extreme(
    values: Union[np.ndarray, torch.Tensor],
    candidates: Sequence[int],
    *,
    sense: str = "max",
    atol: float = 1e-9,
) -> List[int]:
    """Keep candidate indices at min/max ``values`` among candidates."""
    if not candidates:
        return []
    if sense not in ("min", "max"):
        raise ValueError(f"sense must be 'min' or 'max', got {sense!r}")
    if isinstance(values, torch.Tensor):
        n = int(values.numel())
        if _is_full_index_range(candidates, n):
            return (
                _indices_at_extreme_tensor(values, None, sense=sense, atol=atol)
                .detach()
                .cpu()
                .tolist()
            )
        cand_t = torch.as_tensor(list(candidates), device=values.device, dtype=torch.long)
        return (
            _indices_at_extreme_tensor(values, cand_t, sense=sense, atol=atol)
            .detach()
            .cpu()
            .tolist()
        )
    sub = [float(values[i]) for i in candidates]
    best = max(sub) if sense == "max" else min(sub)
    return [
        i
        for i, v in zip(candidates, sub)
        if abs(v - best) <= atol
    ]


def _indices_at_max(
    values: Union[np.ndarray, torch.Tensor],
    candidates: Sequence[int],
    *,
    atol: float = 1e-9,
) -> List[int]:
    return _indices_at_extreme(values, candidates, sense="max", atol=atol)


def sample_anchors_by_target(
    model_ids: Sequence[str],
    target_ids: Sequence[str],
    num_anchors: Optional[int],
    rng: np.random.Generator,
) -> Dict[str, List[str]]:
    """Per-target independent sample of ``num_anchors`` from non-target models.

    ``num_anchors is None`` keeps the full leave-one-out pool for every target.
    """
    out: Dict[str, List[str]] = {}
    for target_id in target_ids:
        pool = [m for m in model_ids if m != target_id]
        if num_anchors is None:
            out[str(target_id)] = pool
            continue
        if num_anchors < 2:
            raise ValueError(f"num_anchors must be >= 2, got {num_anchors}")
        if num_anchors > len(pool):
            raise ValueError(
                f"num_anchors={num_anchors} exceeds non-target pool size "
                f"{len(pool)} for target {target_id!r}"
            )
        pick = rng.choice(len(pool), size=int(num_anchors), replace=False)
        out[str(target_id)] = [pool[int(i)] for i in pick]
    return out


def _sims_dict_from_means(
    means: Union[np.ndarray, torch.Tensor],
    model_ids: List[str],
) -> Dict[str, float]:
    if isinstance(means, torch.Tensor):
        means = means.detach().cpu().numpy()
    return {mid: float(means[j]) for j, mid in enumerate(model_ids)}


def _batch_local_indices(
    local_sample_indices: Sequence[int],
    sample_start: int,
    sample_end: int,
) -> List[int]:
    return [
        li
        for li, gidx in enumerate(local_sample_indices)
        if sample_start <= int(gidx) <= sample_end
    ]


class LooEvalContext:
    """Precomputed leave-one-out anchor reference flags for vectorized GPU ranking."""

    __slots__ = ("target_ids", "model_ids", "target_meta")

    def __init__(
        self,
        *,
        target_ids: List[str],
        model_ids: List[str],
        target_meta: List[
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        self.target_ids = target_ids
        self.model_ids = model_ids
        self.target_meta = target_meta


def build_loo_eval_context(
    model_ids: List[str],
    reference_by_id: Dict[str, Dict[str, object]],
    device: Optional[torch.device],
) -> LooEvalContext:
    from common.reference import reference_tied

    target_meta: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    dev = device or torch.device("cpu")
    for ti, target_id in enumerate(model_ids):
        anchor_js: List[int] = []
        want_target: List[bool] = []
        reference_tie: List[bool] = []
        target_entry = reference_by_id[target_id]
        for aj, anchor_id in enumerate(model_ids):
            if anchor_id == target_id:
                continue
            anchor_entry = reference_by_id[anchor_id]
            tied = reference_tied(target_entry, anchor_entry)
            reference_tie.append(tied)
            want_target.append(
                float(target_entry["robust_acc"]) > float(anchor_entry["robust_acc"])
            )
            anchor_js.append(aj)
        target_meta.append(
            (
                ti,
                torch.tensor(anchor_js, device=dev, dtype=torch.long),
                torch.tensor(want_target, device=dev, dtype=torch.bool),
                torch.tensor(reference_tie, device=dev, dtype=torch.bool),
            )
        )
    return LooEvalContext(
        target_ids=list(model_ids),
        model_ids=list(model_ids),
        target_meta=target_meta,
    )


def upload_score_matrices(
    score_matrices: Dict[ScoreMatrixKey, np.ndarray],
    device: torch.device,
) -> Dict[ScoreMatrixKey, torch.Tensor]:
    return {
        key: torch.as_tensor(mat, device=device, dtype=torch.float32)
        for key, mat in score_matrices.items()
    }


def _batch_mean_sims_gpu(
    score_matrix: torch.Tensor,
    batch_specs: Sequence[Tuple[int, int, int, int]],
    local_sample_indices: Sequence[int],
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for _batch_id, sample_start, sample_end, batch_n in batch_specs:
        batch_local = _batch_local_indices(
            local_sample_indices, sample_start, sample_end
        )
        if len(batch_local) != batch_n:
            raise RuntimeError(
                f"Batch sample mismatch: expected {batch_n}, got {len(batch_local)}"
            )
        idx = torch.as_tensor(
            batch_local, device=score_matrix.device, dtype=torch.long
        )
        rows.append(score_matrix.index_select(0, idx).mean(dim=0))
    return torch.stack(rows, dim=0)


def _loo_target_acc_matrix_gpu(
    batch_sims: torch.Tensor,
    loo_ctx: LooEvalContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``acc [B,T]``, ``n_correct [B,T]``, ``n_anchors [T]``."""
    acc_rows: List[torch.Tensor] = []
    n_ok_rows: List[torch.Tensor] = []
    n_anchor_vals: List[int] = []
    for target_j, anchor_js, want_target, reference_tie in loo_ctx.target_meta:
        if anchor_js.numel() == 0:
            nan = torch.full(
                (batch_sims.shape[0],),
                float("nan"),
                device=batch_sims.device,
            )
            acc_rows.append(nan)
            n_ok_rows.append(torch.zeros_like(nan))
            n_anchor_vals.append(0)
            continue
        t_sim = batch_sims[:, target_j]
        a_sim = batch_sims.index_select(1, anchor_js)
        pred_target_better = t_sim.unsqueeze(1) > a_sim
        pred_anchor_better = t_sim.unsqueeze(1) < a_sim
        eq = t_sim.unsqueeze(1) == a_sim
        correct = reference_tie.unsqueeze(0) | (
            ~eq
            & (
                (pred_target_better & want_target.unsqueeze(0))
                | (pred_anchor_better & (~want_target).unsqueeze(0))
            )
        )
        n_ok_rows.append(correct.float().sum(dim=1))
        acc_rows.append(correct.float().mean(dim=1))
        n_anchor_vals.append(int(anchor_js.numel()))
    n_anchors = torch.tensor(
        n_anchor_vals, device=batch_sims.device, dtype=torch.float32
    )
    return torch.stack(acc_rows, dim=1), torch.stack(n_ok_rows, dim=1), n_anchors


def _rows_from_loo_eval(
    *,
    batch_specs: Sequence[Tuple[int, int, int, int]],
    batch_sims: torch.Tensor,
    acc_matrix: torch.Tensor,
    n_ok_matrix: torch.Tensor,
    n_anchors: torch.Tensor,
    loo_ctx: LooEvalContext,
    configuration: ConfigKey,
    ranking_metric: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    edge, sigma, ds, thr = configuration
    batch_sims_np = batch_sims.detach().cpu().numpy()
    acc_np = acc_matrix.detach().cpu().numpy()
    n_ok_np = n_ok_matrix.detach().cpu().numpy()
    n_anchors_np = n_anchors.detach().cpu().numpy()
    per_target_rows: List[Dict[str, object]] = []
    batch_means: List[Dict[str, object]] = []

    for bi, (batch_id, sample_start, sample_end, batch_n) in enumerate(batch_specs):
        sims = _sims_dict_from_means(batch_sims_np[bi], loo_ctx.model_ids)
        pred_ranks = assign_score_ranks_with_ties(loo_ctx.model_ids, sims)
        batch_accs = acc_np[bi]
        for ti, target_id in enumerate(loo_ctx.target_ids):
            acc = float(batch_accs[ti])
            n_anchors = int(n_anchors_np[ti])
            n_ok = int(n_ok_np[bi, ti])
            per_target_rows.append(
                {
                    "batch_id": batch_id,
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "n_samples": batch_n,
                    "target_id": target_id,
                    "pred_rank": pred_ranks[target_id],
                    "n_selected": batch_n,
                    "chosen_edge": edge,
                    "chosen_sigma": sigma,
                    "chosen_downscale": ds,
                    "chosen_threshold_mode": thr[0],
                    "chosen_threshold": thr[1],
                    "chosen_metric": ranking_metric,
                    "anchor_pairwise_acc": float("nan"),
                    "pa": acc,
                    "n_correct": n_ok,
                    "n_anchors": n_anchors,
                    "target_sim": sims[target_id],
                }
            )
        batch_means.append(
            {
                "batch_id": batch_id,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "n_samples": batch_n,
                "mean_pa": float(np.mean(batch_accs)),
            }
        )
    return per_target_rows, batch_means


def explicit_float_list(values: Optional[Sequence[float]]) -> Tuple[float, ...]:
    if values is None or len(values) == 0:
        return ()
    return tuple(float(v) for v in values)


def explicit_int_list(values: Optional[Sequence[int]]) -> Tuple[int, ...]:
    if values is None or len(values) == 0:
        return ()
    return tuple(int(v) for v in values)


def parse_multi_float(values: Optional[Sequence[float]]) -> Tuple[float, ...]:
    if not values:
        return (0.0,)
    return tuple(float(v) for v in values)


def parse_multi_str(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return ALL_RANKING_METRICS
    return tuple(str(v) for v in values)


def validate_ranking_metrics(metrics: Tuple[str, ...]) -> Tuple[str, ...]:
    allowed = set(ALL_RANKING_METRICS) | set(METRICS.keys())
    for m in metrics:
        if m not in allowed:
            raise ValueError(f"Unknown --metric {m!r}")
    return metrics


def base_metrics_for_ranking_metrics(ranking_metrics: Sequence[str]) -> Tuple[str, ...]:
    bases: List[str] = []
    for m in ranking_metrics:
        bases.extend(base_metrics_for_grad(m))
    return tuple(dict.fromkeys(bases))


def build_sweep_keys(
    configurations: Sequence[ConfigKey],
    ranking_metrics: Sequence[str],
) -> List[ScoreMatrixKey]:
    return [
        (edge, sigma, ds, thr, metric)
        for edge, sigma, ds, thr in configurations
        for metric in ranking_metrics
    ]


def add_configuration_args(p: argparse.ArgumentParser) -> None:
    """Candidate-space axes of the configuration c = (rho, psi, e, mu)."""
    p.add_argument(
        "--retain_abs",
        type=float,
        nargs="*",
        default=None,
        help="rho = abs(tau_g): keep |g| >= tau_g. Omit both retention flags for rho = all.",
    )
    p.add_argument(
        "--retain_top_pct",
        type=float,
        nargs="*",
        default=None,
        help="rho = top(p): keep the p%% largest |g| over pixels and channels.",
    )
    p.add_argument(
        "--edge_percentile",
        type=float,
        nargs="+",
        default=[90.0],
        help="e: percentile among nonzero edge magnitudes used as the binarization threshold.",
    )
    p.add_argument(
        "--smooth_sigma",
        type=float,
        nargs="*",
        default=None,
        help="psi = Gaussian smoothing with standard deviation sigma.",
    )
    p.add_argument(
        "--smooth_scale",
        type=int,
        nargs="*",
        default=None,
        help="psi = bilinear scale smoothing by integer factor d (d=1 is the identity).",
    )
    p.add_argument(
        "--similarity",
        type=str,
        nargs="+",
        default=list(ALL_RANKING_METRICS),
        help="mu: ssim, dice, or dice_ssim (the arithmetic mean, Dice+SSIM in the paper).",
    )


def parse_configuration_args(args: argparse.Namespace) -> Tuple[
    List[ConfigKey],
    Tuple[str, ...],
    Tuple[str, ...],
    List[ScoreMatrixKey],
]:
    edges = parse_multi_float(args.edge_percentile)
    abs_thresholds = explicit_float_list(args.retain_abs)
    top_pcts = explicit_float_list(args.retain_top_pct)
    sigmas = explicit_float_list(args.smooth_sigma)
    downscales = explicit_int_list(args.smooth_scale)
    threshold_specs = resolve_threshold_specs(abs_thresholds, top_pcts)
    ranking_metrics = validate_ranking_metrics(parse_multi_str(args.similarity))
    configurations = iter_configurations(edges, sigmas, downscales, threshold_specs)
    sweep_keys = build_sweep_keys(configurations, ranking_metrics)
    return configurations, ranking_metrics, base_metrics_for_ranking_metrics(ranking_metrics), sweep_keys


def similarity_csv_path(setting_root: str, metric: str) -> str:
    return os.path.join(setting_root, f"{metric}.csv")


def setting_root_from_key(raw_dir: str, configuration: ConfigKey) -> str:
    edge, sigma, ds, thr = configuration
    return os.path.join(raw_dir, "settings", setting_dir_name(edge, sigma, ds, thr))


def format_threshold_csv(val: float) -> str:
    if val == 0.0:
        return "0"
    if abs(val) < 1e-4 or abs(val) >= 1e4:
        return f"{val:.6e}"
    return f"{val:g}"


def write_similarity_csv(
    csv_path: str,
    *,
    model_ids: Sequence[str],
    reference_by_id: Dict[str, Dict[str, object]],
    sample_indices: Sequence[int],
    score_matrix: np.ndarray,
    configuration: ConfigKey,
    metric: str,
) -> None:
    """score_matrix shape [n_samples, n_models] aligned with model_ids."""
    edge, sigma, ds, thr = configuration
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    sample_cols = [str(int(i)) for i in sample_indices]
    with open(csv_path, "w", newline="") as f:
        f.write(
            f"# section=similarity,edge={edge:g},sigma={sigma:g},downscale={ds},"
            f"threshold_mode={thr[0]},threshold={format_threshold_csv(thr[1])},"
            f"threshold_label={threshold_spec_label(thr)},metric={metric},"
            f"n_models={len(model_ids)},n_samples={len(sample_indices)}\n"
        )
        writer = csv.writer(f)
        writer.writerow(["model_id", "clean_acc", "reference_robust_acc", *sample_cols])
        for j, mid in enumerate(model_ids):
            row = [
                mid,
                f"{float(reference_by_id[mid]['clean_acc']):.6f}",
                f"{float(reference_by_id[mid]['robust_acc']):.6f}",
            ]
            row.extend(f"{float(score_matrix[i, j]):.6f}" for i in range(len(sample_indices)))
            writer.writerow(row)


def read_similarity_csv(
    csv_path: str,
    *,
    wanted_sample_indices: Sequence[int],
    wanted_model_ids: Sequence[str],
) -> np.ndarray:
    """
    Return score_matrix [n_wanted_samples, n_wanted_models] in wanted order.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Similarity CSV not found: {csv_path}")

    with open(csv_path, newline="") as f:
        header_line = f.readline()
        if not header_line.startswith("#"):
            raise ValueError(f"Expected metadata line in {csv_path}")
        reader = csv.reader(f)
        header = next(reader)
        if len(header) < 4:
            raise ValueError(f"Invalid header in {csv_path}")
        file_model_ids = []
        col_by_sample: Dict[int, int] = {}
        for ci, col in enumerate(header[3:], start=3):
            col_by_sample[int(col)] = ci
        missing_samples = [s for s in wanted_sample_indices if int(s) not in col_by_sample]
        if missing_samples:
            raise ValueError(
                f"{csv_path} missing sample columns for indices: {missing_samples[:5]}"
                f"{'...' if len(missing_samples) > 5 else ''}"
            )
        rows_by_model: Dict[str, List[str]] = {}
        for row in reader:
            if not row:
                continue
            rows_by_model[str(row[0])] = row
        missing_models = [m for m in wanted_model_ids if m not in rows_by_model]
        if missing_models:
            raise ValueError(
                f"{csv_path} missing models: {missing_models[:5]}"
                f"{'...' if len(missing_models) > 5 else ''}"
            )
        n_s = len(wanted_sample_indices)
        n_m = len(wanted_model_ids)
        mat = np.zeros((n_s, n_m), dtype=np.float64)
        for j, mid in enumerate(wanted_model_ids):
            row = rows_by_model[mid]
            for i, sidx in enumerate(wanted_sample_indices):
                mat[i, j] = float(row[col_by_sample[int(sidx)]])
        return mat


def write_export_summary_csv(
    csv_path: str,
    *,
    args_meta: str,
    model_ids: Sequence[str],
    reference_by_id: Dict[str, Dict[str, object]],
    sample_indices: Sequence[int],
    n_settings: int,
    timings: Dict[str, float],
    model_timings: Optional[List[Dict[str, float]]] = None,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        f.write(f"# section=export_summary,{args_meta}\n")
        f.write("\n# section=timings_sec\n")
        f.write("phase,seconds\n")
        for phase, sec in timings.items():
            f.write(f"{phase},{sec:.6f}\n")
        f.write("\n# section=models\n")
        f.write("model_id,clean_acc,reference_robust_acc,architecture\n")
        for mid in model_ids:
            e = reference_by_id[mid]
            f.write(
                f"{mid},{float(e['clean_acc']):.6f},{float(e['robust_acc']):.6f},"
                f"{e.get('architecture', mid)}\n"
            )
        f.write("\n# section=run_info\n")
        f.write(f"n_models,{len(model_ids)}\n")
        f.write(f"n_samples,{len(sample_indices)}\n")
        f.write(f"n_settings,{n_settings}\n")
        f.write(f"sample_start,{min(sample_indices) if sample_indices else 0}\n")
        f.write(f"sample_end,{max(sample_indices) if sample_indices else 0}\n")
        if model_timings:
            f.write("\n# section=per_model_timings_sec\n")
            f.write("model_id,load_model,grad,sim,save_maps,save_csv\n")
            for row in model_timings:
                f.write(
                    f"{row['model_id']},{row.get('load_model', 0):.6f},"
                    f"{row.get('grad', 0):.6f},{row.get('sim', 0):.6f},"
                    f"{row.get('save_maps', 0):.6f},{row.get('save_csv', 0):.6f}\n"
                )


def write_loo_csv(
    *,
    csv_path: str,
    model_ids: List[str],
    reference_by_id: Dict[str, Dict[str, object]],
    probe_budget: int,
    run_label: str,
    edge: Optional[float],
    sigma: Optional[float],
    downscale: Optional[int],
    threshold: Optional[ThresholdSpec],
    metric: Optional[str],
    total_samples: int,
    per_target_rows: List[Dict[str, object]],
    batch_means: List[Dict[str, object]],
    extra_summary: str = "",
    timings: Optional[Dict[str, float]] = None,
    record_ties: Optional[str] = None,
    n_settings: Optional[int] = None,
    tie_unique_by_batch: bool = False,
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    reference_ranks = assign_reference_ranks([reference_by_id[m] for m in model_ids])
    # Overall mean/std over ranking-batch scores (batch_means), not over
    # every (target, batch) LOO row.
    if batch_means:
        batch_accs = [float(b["mean_pa"]) for b in batch_means]
        overall_acc_stats = summarize_values(batch_accs)
    else:
        target_batch_accs = [float(r["pa"]) for r in per_target_rows]
        overall_acc_stats = summarize_values(target_batch_accs) if target_batch_accs else {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
        }
    if n_settings is None:
        # Prefer candidates= from extra_summary (best-acc sweep size).
        n_settings = 0
        for part in extra_summary.split(","):
            part = part.strip()
            if part.startswith("candidates="):
                try:
                    n_settings = int(part.split("=", 1)[1])
                except ValueError:
                    n_settings = 0
                break
    stage_stats = summarize_tie_break_stages(
        per_target_rows,
        n_settings=int(n_settings),
        unique_by_batch=tie_unique_by_batch,
    )
    setting_note = (
        f"edge={edge:g},sigma={sigma:g},downscale={downscale},"
        f"threshold={threshold_spec_label(threshold)},metric={metric}"
        if edge is not None and threshold is not None and metric is not None
        else extra_summary
    )
    with open(csv_path, "w") as f:
        f.write(
            f"# section=summary,run_label={run_label},{setting_note},"
            f"probe_budget={probe_budget},"
            f"n_ranking_batches={len(batch_means)},n_models={len(model_ids)},"
            f"n_targets={len({r['target_id'] for r in per_target_rows})},"
            f"total_samples={total_samples},"
            f"n_settings={int(stage_stats['n_settings'])},"
            f"mean_enter_astd={stage_stats['mean_enter_astd']:.4f},"
            f"mean_enter_med={stage_stats['mean_enter_med']:.4f},"
            f"mean_enter_random={stage_stats['mean_enter_random']:.4f},"
            f"n_groups={int(stage_stats['n_groups'])},"
            f"n_groups_astd_active={int(stage_stats['n_groups_astd_active'])},"
            f"n_groups_med_active={int(stage_stats['n_groups_med_active'])},"
            f"n_groups_random_active={int(stage_stats['n_groups_random_active'])}\n"
        )
        f.write("\n# section=tie_break_stages\n")
        f.write(
            "n_settings,n_groups,mean_enter_astd,mean_enter_med,mean_enter_random,"
            "frac_enter_astd,frac_enter_med,frac_enter_random,"
            "n_groups_astd_active,n_groups_med_active,n_groups_random_active\n"
        )
        f.write(
            f"{int(stage_stats['n_settings'])},{int(stage_stats['n_groups'])},"
            f"{stage_stats['mean_enter_astd']:.6f},"
            f"{stage_stats['mean_enter_med']:.6f},"
            f"{stage_stats['mean_enter_random']:.6f},"
            f"{stage_stats['frac_enter_astd']:.8f},"
            f"{stage_stats['frac_enter_med']:.8f},"
            f"{stage_stats['frac_enter_random']:.8f},"
            f"{int(stage_stats['n_groups_astd_active'])},"
            f"{int(stage_stats['n_groups_med_active'])},"
            f"{int(stage_stats['n_groups_random_active'])}\n"
        )
        f.write("\n# section=target_batch_rank_acc_stats\n")
        f.write("stat,pa\n")
        for stat_name in ("mean", "median", "min", "max", "std"):
            f.write(f"{stat_name},{overall_acc_stats[stat_name]:.6f}\n")

        f.write("\n# section=batch_means\n")
        f.write("batch_id,sample_start,sample_end,n_samples,mean_pa\n")
        for b in batch_means:
            f.write(
                f"{b['batch_id']},{b['sample_start']},{b['sample_end']},"
                f"{b['n_samples']},{float(b['mean_pa']):.6f}\n"
            )

        f.write("\n# section=per_target_per_batch\n")
        hdr = (
            "batch_id,sample_start,sample_end,n_samples,target_id,gt_rank,pred_rank,"
            "chosen_edge,chosen_sigma,chosen_downscale,"
            "chosen_threshold_mode,chosen_threshold,chosen_metric,"
            "anchor_pairwise_acc,chosen_anchor_sim_std,chosen_target_med_dev,"
            "pa,n_correct,n_anchors,target_sim,"
            "n_enter_astd,n_enter_med,n_enter_random\n"
        )
        f.write(hdr)
        row_lines = []
        for row in per_target_rows:
            tid = str(row["target_id"])
            thr_mode = str(row["chosen_threshold_mode"])
            thr_val = float(row["chosen_threshold"])
            n_tied = int(row.get("n_tied_at_best", 1))
            n_enter_astd = int(row.get("n_enter_astd", n_tied))
            n_enter_med = int(row.get("n_enter_med", 1))
            n_enter_random = int(row.get("n_enter_random", 1))
            row_lines.append(
                f"{row['batch_id']},{row['sample_start']},{row['sample_end']},"
                f"{row['n_samples']},{tid},{reference_ranks[tid]},"
                f"{int(row['pred_rank'])},"
                f"{float(row['chosen_edge']):g},"
                f"{float(row['chosen_sigma']):g},{int(row['chosen_downscale'])},"
                f"{thr_mode},{thr_val:g},"
                f"{row['chosen_metric']},"
                f"{float(row.get('anchor_pairwise_acc', float('nan'))):.6f},"
                f"{float(row.get('chosen_anchor_sim_std', float('nan'))):.6f},"
                f"{float(row.get('chosen_target_med_dev', float('nan'))):.6f},"
                f"{float(row['pa']):.6f},"
                f"{row['n_correct']},{row['n_anchors']},"
                f"{float(row['target_sim']):.6f},"
                f"{n_enter_astd},{n_enter_med},{n_enter_random}\n"
            )
        f.write("".join(row_lines))

        tie_rows: List[Dict[str, object]] = []
        for row in per_target_rows:
            for detail in row.get("tied_details") or []:
                tie_rows.append(
                    {
                        "batch_id": row["batch_id"],
                        "target_id": row["target_id"],
                        **detail,
                    }
                )
        if tie_rows:
            f.write("\n# section=setting_ties\n")
            if record_ties == "full":
                f.write(
                    "# model_sims columns follow model_ids order: "
                    + ";".join(model_ids)
                    + "\n"
                )
                tie_hdr = (
                    "batch_id,target_id,tie_order,is_chosen,config_label,edge,sigma,"
                    "downscale,threshold_mode,threshold,metric,anchor_pairwise_acc,"
                    "anchor_sim_std,target_med_dev,"
                    "target_sim,model_sims,pred_rank,pa\n"
                )
            else:
                # slim: only metrics used by the fixed cascade + acc for offline checks
                tie_hdr = (
                    "batch_id,target_id,tie_order,is_chosen,config_label,edge,sigma,"
                    "downscale,threshold_mode,threshold,metric,anchor_pairwise_acc,"
                    "anchor_sim_std,target_med_dev,pa\n"
                )
            f.write(tie_hdr)
            for tr in tie_rows:
                thr_mode = str(tr["threshold_mode"])
                thr_val = float(tr["threshold"])
                base = (
                    f"{tr['batch_id']},{tr['target_id']},{tr['tie_order']},"
                    f"{int(bool(tr['is_chosen']))},{tr['config_label']},"
                    f"{float(tr['edge']):g},{float(tr['sigma']):g},"
                    f"{int(tr['downscale'])},{thr_mode},{thr_val:g},"
                    f"{tr['metric']},"
                    f"{float(tr['anchor_pairwise_acc']):.6f},"
                    f"{float(tr.get('anchor_sim_std', float('nan'))):.6f},"
                    f"{float(tr.get('target_med_dev', float('nan'))):.6f},"
                )
                if record_ties == "full":
                    f.write(
                        base
                        + f"{float(tr.get('target_sim', float('nan'))):.6f},"
                        f"{tr.get('model_sims', '')},"
                        f"{int(tr.get('pred_rank', 0))},"
                        f"{float(tr['pa']):.6f}\n"
                    )
                else:
                    f.write(base + f"{float(tr['pa']):.6f}\n")

        f.write("\n# section=per_target_over_batches\n")
        f.write("target_id,gt_rank,mean_pa,median,min,max,std,n_batches\n")
        by_target: Dict[str, List[float]] = {}
        for row in per_target_rows:
            by_target.setdefault(str(row["target_id"]), []).append(
                float(row["pa"])
            )
        for tid in sorted(by_target, key=lambda t: (reference_ranks[t], t)):
            stats = summarize_values(by_target[tid])
            f.write(
                f"{tid},{reference_ranks[tid]},{stats['mean']:.6f},{stats['median']:.6f},"
                f"{stats['min']:.6f},{stats['max']:.6f},{stats['std']:.6f},"
                f"{len(by_target[tid])}\n"
            )

        if timings:
            f.write("\n# section=timings_sec\n")
            f.write("phase,seconds\n")
            for phase, sec in timings.items():
                f.write(f"{phase},{sec:.6f}\n")


def write_py2_summary_csv(
    csv_path: str,
    *,
    probe_budget: int,
    rows: List[Dict[str, object]],
) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        f.write(f"# section=summary,probe_budget={probe_budget},n_settings={len(rows)}\n")
        writer = csv.writer(f)
        writer.writerow(
            [
                "edge",
                "sigma",
                "downscale",
                "threshold_mode",
                "threshold",
                "threshold_label",
                "metric",
                "setting_dir",
                "mean_acc",
                "median_acc",
                "min_acc",
                "max_acc",
                "std_acc",
                "n_batches",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["edge"],
                    row["sigma"],
                    row["downscale"],
                    row["threshold_mode"],
                    row["threshold"],
                    row["threshold_label"],
                    row["metric"],
                    row["setting_dir"],
                    f"{float(row['mean_acc']):.6f}",
                    f"{float(row['median_acc']):.6f}",
                    f"{float(row['min_acc']):.6f}",
                    f"{float(row['max_acc']):.6f}",
                    f"{float(row['std_acc']):.6f}",
                    row["n_batches"],
                ]
            )


def evaluate_fixed_setting(
    *,
    score_matrix: ScoreMatrix,
    configuration: ConfigKey,
    ranking_metric: str,
    target_ids: List[str],
    model_ids: List[str],
    reference_by_id: Dict[str, Dict[str, object]],
    batch_specs: List[Tuple[int, int, int, int]],
    local_sample_indices: Sequence[int],
    device: Optional[torch.device] = None,
    loo_ctx: Optional[LooEvalContext] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    edge, sigma, ds, thr = configuration

    if isinstance(score_matrix, torch.Tensor):
        if loo_ctx is None:
            loo_ctx = build_loo_eval_context(model_ids, reference_by_id, score_matrix.device)
        batch_sims = _batch_mean_sims_gpu(
            score_matrix, batch_specs, local_sample_indices
        )
        acc_matrix, n_ok_matrix, n_anchors = _loo_target_acc_matrix_gpu(
            batch_sims, loo_ctx
        )
        return _rows_from_loo_eval(
            batch_specs=batch_specs,
            batch_sims=batch_sims,
            acc_matrix=acc_matrix,
            n_ok_matrix=n_ok_matrix,
            n_anchors=n_anchors,
            loo_ctx=loo_ctx,
            configuration=configuration,
            ranking_metric=ranking_metric,
        )

    anchor_pool = list(model_ids)
    checks_by_target = {
        tid: anchor_checks_for_target(tid, anchor_pool, reference_by_id) for tid in target_ids
    }
    per_target_rows: List[Dict[str, object]] = []
    batch_means: List[Dict[str, object]] = []

    for batch_id, sample_start, sample_end, batch_n in batch_specs:
        batch_local = _batch_local_indices(local_sample_indices, sample_start, sample_end)
        if len(batch_local) != batch_n:
            raise RuntimeError(
                f"Batch sample mismatch batch_id={batch_id}: "
                f"expected {batch_n}, got {len(batch_local)}"
            )
        sims = similarities_from_score_matrix(score_matrix, model_ids, batch_local)
        pred_ranks = assign_score_ranks_with_ties(model_ids, sims)
        accs: List[float] = []
        for target_id in target_ids:
            acc, n_ok, n_anchors = pairwise_agreement_for_target(
                target_id, sims, checks_by_target[target_id]
            )
            per_target_rows.append(
                {
                    "batch_id": batch_id,
                    "sample_start": sample_start,
                    "sample_end": sample_end,
                    "n_samples": batch_n,
                    "target_id": target_id,
                    "pred_rank": pred_ranks[target_id],
                    "n_selected": batch_n,
                    "chosen_edge": edge,
                    "chosen_sigma": sigma,
                    "chosen_downscale": ds,
                    "chosen_threshold_mode": thr[0],
                    "chosen_threshold": thr[1],
                    "chosen_metric": ranking_metric,
                    "anchor_pairwise_acc": float("nan"),
                    "pa": acc,
                    "n_correct": n_ok,
                    "n_anchors": n_anchors,
                    "target_sim": sims[target_id],
                }
            )
            accs.append(acc)
        batch_means.append(
            {
                "batch_id": batch_id,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "n_samples": batch_n,
                "mean_pa": float(np.mean(accs)) if accs else float("nan"),
            }
        )
    return per_target_rows, batch_means


def read_export_summary_model_ids(raw_dir: str) -> List[str]:
    """Model ids listed in ``export_summary.csv`` from sga_export_raw.py."""
    path = os.path.join(raw_dir, "export_summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"export_summary.csv not found under {raw_dir}")
    ids: List[str] = []
    in_models = False
    with open(path, newline="") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("# section=models"):
                in_models = True
                continue
            if in_models and stripped.startswith("# section="):
                break
            if not in_models or not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("model_id,"):
                continue
            ids.append(stripped.split(",", 1)[0])
    if not ids:
        raise ValueError(f"No models listed in {path}")
    return ids


def resolve_model_raw_dirs(
    raw_dirs: Sequence[str],
    model_ids: Sequence[str],
) -> Dict[str, str]:
    """
    Map each model id to the first --raw_dir whose export_summary lists it.
    """
    if not raw_dirs:
        raise ValueError("At least one --raw_dir is required")
    dir_model_sets = [
        (raw_dir, set(read_export_summary_model_ids(raw_dir))) for raw_dir in raw_dirs
    ]
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for mid in model_ids:
        found = None
        for raw_dir, model_set in dir_model_sets:
            if mid in model_set:
                found = raw_dir
                break
        if found is None:
            missing.append(mid)
        else:
            mapping[mid] = found
    if missing:
        raise ValueError(
            "Models from --reference_xml not found in any --raw_dir export_summary: "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
        )
    return mapping


def validate_raw_dirs(raw_dirs: Sequence[str]) -> None:
    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            raise FileNotFoundError(f"--raw_dir not found: {raw_dir}")
        read_export_summary_model_ids(raw_dir)


def read_similarity_csv_merged(
    *,
    model_raw_dir: Dict[str, str],
    configuration: ConfigKey,
    metric: str,
    model_ids: Sequence[str],
    sample_indices: Sequence[int],
) -> np.ndarray:
    """Assemble [n_samples, n_models] from per-export similarity CSVs."""
    n_s = len(sample_indices)
    n_m = len(model_ids)
    mat = np.zeros((n_s, n_m), dtype=np.float64)
    by_dir: Dict[str, List[Tuple[int, str]]] = {}
    for j, mid in enumerate(model_ids):
        by_dir.setdefault(model_raw_dir[mid], []).append((j, mid))
    for raw_dir, items in by_dir.items():
        setting_root = setting_root_from_key(raw_dir, configuration)
        csv_path = similarity_csv_path(setting_root, metric)
        sub_ids = [mid for _, mid in items]
        sub_mat = read_similarity_csv(
            csv_path,
            wanted_sample_indices=sample_indices,
            wanted_model_ids=sub_ids,
        )
        for k, (j, _mid) in enumerate(items):
            mat[:, j] = sub_mat[:, k]
    return mat


def load_score_matrices_for_sweep(
    raw_dirs: Sequence[str],
    configurations: Sequence[ConfigKey],
    ranking_metrics: Sequence[str],
    model_ids: Sequence[str],
    sample_indices: Sequence[int],
) -> Dict[ScoreMatrixKey, np.ndarray]:
    raw_dir_list = list(raw_dirs)
    if len(raw_dir_list) == 1:
        model_raw_dir = {mid: raw_dir_list[0] for mid in model_ids}
    else:
        model_raw_dir = resolve_model_raw_dirs(raw_dir_list, model_ids)
    mats: Dict[ScoreMatrixKey, np.ndarray] = {}
    for configuration in configurations:
        for metric in ranking_metrics:
            mat = read_similarity_csv_merged(
                model_raw_dir=model_raw_dir,
                configuration=configuration,
                metric=metric,
                model_ids=model_ids,
                sample_indices=sample_indices,
            )
            edge, sigma, ds, thr = configuration
            mats[(edge, sigma, ds, thr, metric)] = mat
    return mats


def sample_indices_from_range(start_idx: int, end_idx: int) -> List[int]:
    if end_idx <= start_idx:
        raise ValueError(f"end_idx ({end_idx}) must be > start_idx ({start_idx})")
    return list(range(start_idx, end_idx))


def args_meta_string(args: argparse.Namespace, extra: Sequence[str] = ()) -> str:
    parts = [f"{k}={getattr(args, k)!r}" for k in sorted(vars(args))]
    parts.extend(extra)
    return ",".join(parts)
