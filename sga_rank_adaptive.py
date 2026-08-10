"""
Per-batch configuration selection on exported SGA alignment scores.

Selects one configuration c = (rho, psi, e, mu) for every probe batch, following
the cascade PA -> Std -> Gap -> seeded random. Each protocol uses the
order-preserving subsequence its evaluator knowledge permits
(``--protocol all`` runs all three, default)::

  uncalibrated: no reference information; selection on the complete pool:
      max Std(scores) -> max Gap -> seeded random
      Per batch: mean PA over leave-one-out targets.

  calibrated: ordered anchors, no target reference; ``--num_anchors`` = H
      (omit for H = M-1). Per probe batch, sample H anchors
      (seed = anchor_seed + batch_id, default 0 -> 0,1,2,...); the remainder
      are targets. Selection uses the anchors only:
        max anchor PA -> max Std(anchor scores) -> max Gap -> seeded random
      PA covers every pair containing at least one target; anchor-anchor pairs
      are excluded because they drove the selection.

  oracle: complete target pool and its reference order (not deployable,
      reported as an empirical upper bound):
      max PA -> max Std -> max Gap -> seeded random

Gap is the minimum adjacent gap of the sorted pool scores. Mean PA and its
standard deviation are reported over probe batches.

CSV compatibility: the Std and Gap criteria are stored in the historical
``chosen_anchor_sim_std`` / ``chosen_target_med_dev`` columns, and the cascade
stage sizes in the enter_astd / enter_med / enter_random slots.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from sga.structural import (
    ScoreMatrixKey,
    _anchor_pairwise_reference_flags,
    anchor_checks_for_target,
    iter_ranking_batches,
    set_reproducible,
    similarities_from_score_matrix,
    pairwise_agreement_for_target,
)
from common.reference import (
    assign_score_ranks_with_ties,
    reference_relation,
    reference_tied,
    load_reference_xml,
    models_from_reference_xml,
    score_pairwise_relation,
)
from sga.rank_io import (
    DEFAULT_PROBE_BUDGETS,
    parse_probe_budgets,
    _batch_local_indices,
    _format_sweep_key_label,
    _indices_at_extreme,
    _indices_at_extreme_tensor,
    _indices_at_max,
    _indices_at_max_tensor,
    _loo_target_acc_matrix_gpu,
    _sims_dict_from_means,
    _tie_break_rng,
    add_configuration_args,
    build_loo_eval_context,
    format_tie_break_stage_log,
    load_score_matrices_for_sweep,
    parse_configuration_args,
    resolve_model_raw_dirs,
    resolve_rank_eval_device,
    sample_anchors_by_target,
    sample_indices_from_range,
    stack_score_matrices,
    summarize_tie_break_stages,
    validate_raw_dirs,
    write_loo_csv,
)

PROTOCOLS = ("uncalibrated", "calibrated", "oracle")
# Seeded tie-breaking hashes this key, so it must stay byte-identical to the
# strings used for the published runs even though the protocols were renamed.
_TIE_BREAK_KEY = {
    "uncalibrated": "noanc_std_gap",
    "calibrated": "anc_std_gap",
    "oracle": "oracle_batch",
}
CASCADE = "max_std>max_gap>random"
ScoreArray = Union[np.ndarray, torch.Tensor]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Per-batch configuration selection: uncalibrated, calibrated, "
            "and oracle. Default --protocol all."
        ),
    )
    p.add_argument(
        "--raw_dir",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more sga_export_raw.py out dirs. With multiple dirs, each "
            "model is loaded from the first dir whose export_summary.csv lists it."
        ),
    )
    p.add_argument("--reference_xml", type=str, required=True)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=256)
    p.add_argument("--prefix", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument(
        "--probe_budgets",
        type=str,
        default=",".join(str(s) for s in DEFAULT_PROBE_BUDGETS),
        help=(
            "Comma-separated ranking batch sizes (default: "
            f"{','.join(str(s) for s in DEFAULT_PROBE_BUDGETS)})."
        ),
    )
    p.add_argument(
        "--protocol",
        type=str,
        nargs="+",
        default=["all"],
        metavar="NAME",
        help=(
            "One or more of: uncalibrated, calibrated, oracle. "
            "Pass 'all' (default) to run all three."
        ),
    )
    p.add_argument(
        "--num_anchors",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Anchor counts k for calibrated only (2 <= k <= M-1). "
            "Omit (or pass no values) for all = M-1 (one random target per "
            "batch). Example: --num_anchors 3 4. "
            "Ignored by uncalibrated / oracle."
        ),
    )
    p.add_argument(
        "--anchor_seed",
        type=int,
        default=0,
        help=(
            "Base RNG seed for calibrated anchor sampling "
            "(batch b uses seed=anchor_seed+b; default: 0 -> 0,1,2,...)."
        ),
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (NumPy) for batch aggregation (default: CUDA when available).",
    )
    p.add_argument("--seed", type=int, default=0)
    add_configuration_args(p)
    return p.parse_args()


def _resolve_protocols(raw: Sequence[str]) -> List[str]:
    names: List[str] = []
    seen: set[str] = set()
    for tok in raw:
        t = tok.strip().lower()
        if t == "all":
            for s in PROTOCOLS:
                if s not in seen:
                    seen.add(s)
                    names.append(s)
            continue
        if t not in PROTOCOLS:
            raise SystemExit(
                f"Unknown --protocol {tok!r}; choose from {PROTOCOLS} or 'all'"
            )
        if t not in seen:
            seen.add(t)
            names.append(t)
    if not names:
        raise SystemExit("--protocol requires at least one name or 'all'")
    return names


def _cascade_label(protocol: str) -> str:
    if protocol == "oracle":
        return f"max_loo_acc>{CASCADE}"
    if protocol == "calibrated":
        return f"max_acc>{CASCADE}"
    return CASCADE


def _mean_loo_acc_per_setting(
    *,
    sims_np: np.ndarray,
    sims_t: Optional[torch.Tensor],
    model_ids: List[str],
    reference_by_id: Dict[str, Dict[str, object]],
) -> ScoreArray:
    """Mean LOO ``pa`` over all targets, for each setting ``[K]``."""
    if sims_t is not None:
        loo_ctx = build_loo_eval_context(model_ids, reference_by_id, sims_t.device)
        acc_kt, _n_ok, _n_anc = _loo_target_acc_matrix_gpu(sims_t, loo_ctx)
        return acc_kt.mean(dim=1)

    k = int(sims_np.shape[0])
    anchors_by_target = sample_anchors_by_target(
        model_ids, model_ids, None, np.random.default_rng(0)
    )
    out = np.empty(k, dtype=np.float64)
    for ki in range(k):
        sims = {mid: float(sims_np[ki, j]) for j, mid in enumerate(model_ids)}
        accs: List[float] = []
        for tid in model_ids:
            checks = anchor_checks_for_target(
                tid, anchors_by_target[tid], reference_by_id
            )
            acc, _n_ok, _ = pairwise_agreement_for_target(tid, sims, checks)
            accs.append(acc)
        out[ki] = float(np.mean(accs)) if accs else float("nan")
    return out


def _parse_num_anchors_list(raw: Optional[Sequence[int]]) -> List[Optional[int]]:
    """``None`` / empty -> ``[None]`` meaning all (= M-1 at runtime)."""
    if raw is None or len(raw) == 0:
        return [None]
    out: List[Optional[int]] = []
    seen: set[int] = set()
    for k in raw:
        if k < 2:
            raise SystemExit(f"--num_anchors values must be >= 2, got {k}")
        if k in seen:
            continue
        seen.add(k)
        out.append(int(k))
    return out


def _validate_num_anchors_vs_pool(
    k_list: Sequence[Optional[int]],
    n_models: int,
) -> None:
    max_k = n_models - 1
    if max_k < 2:
        raise SystemExit(
            f"Need at least 3 models for --num_anchors, got n_models={n_models}"
        )
    for k in k_list:
        if k is None:
            continue
        if int(k) > max_k:
            raise SystemExit(
                f"--num_anchors={k} exceeds max {max_k} "
                f"(need >=1 target; n_models={n_models})"
            )


def _resolve_num_anchors(num_anchors: Optional[int], n_models: int) -> int:
    """Map omit/all -> M-1."""
    k = (n_models - 1) if num_anchors is None else int(num_anchors)
    if k < 2 or k > n_models - 1:
        raise ValueError(
            f"num_anchors must be in [2, {n_models - 1}], got {k}"
        )
    return k


def _sample_batch_anchors(
    model_ids: Sequence[str],
    num_anchors: int,
    *,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Sample ``num_anchors`` models; remainder are targets."""
    n = len(model_ids)
    if num_anchors < 2 or num_anchors > n - 1:
        raise ValueError(
            f"num_anchors must be in [2, {n - 1}], got {num_anchors}"
        )
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(n, size=int(num_anchors), replace=False)
    anchor_set = {model_ids[int(i)] for i in pick}
    anchors = [model_ids[int(i)] for i in pick]
    targets = [m for m in model_ids if m not in anchor_set]
    return anchors, targets


def _pairwise_acc_with_targets(
    reference_by_id: Dict[str, Dict[str, object]],
    pred_ranks: Dict[str, int],
    model_ids: Sequence[str],
    target_ids: Sequence[str],
) -> Tuple[float, int, int]:
    """Pairwise PA over pairs with >=1 target (skip pure anchor-anchor)."""
    target_set = set(target_ids)
    correct = 0
    total = 0
    ids = list(model_ids)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a not in target_set and b not in target_set:
                continue
            total += 1
            if reference_tied(reference_by_id[a], reference_by_id[b]):
                correct += 1
                continue
            score_rel = score_pairwise_relation(pred_ranks, a, b)
            if score_rel is None:
                continue
            reference_rel = reference_relation(reference_by_id[a], reference_by_id[b])
            assert reference_rel is not None
            if reference_rel == score_rel:
                correct += 1
    if total == 0:
        return 1.0, 0, 0
    return float(correct / total), correct, total


def _anchor_pairwise_acc_per_setting(
    *,
    sims_np: np.ndarray,
    sims_t: Optional[torch.Tensor],
    anchor_ids: List[str],
    mid_to_j: Dict[str, int],
    reference_by_id: Dict[str, Dict[str, object]],
) -> ScoreArray:
    """Per-setting pairwise ranking acc on ``anchor_ids`` only."""
    pairs, want_first, reference_tie = _anchor_pairwise_reference_flags(anchor_ids, reference_by_id)
    js = [mid_to_j[m] for m in anchor_ids]
    n_pairs = len(pairs)
    if sims_t is not None:
        a_sims = sims_t.index_select(
            1, torch.as_tensor(js, device=sims_t.device, dtype=torch.long)
        )
        if n_pairs == 0:
            return torch.ones(a_sims.shape[0], device=a_sims.device, dtype=torch.float32)
        ranks = (-a_sims).argsort(dim=1).argsort(dim=1)
        correct = torch.zeros(a_sims.shape[0], device=a_sims.device, dtype=torch.float32)
        for (ai, aj), want, tied in zip(pairs, want_first, reference_tie):
            if tied:
                correct = correct + 1.0
                continue
            pred_first = ranks[:, ai] < ranks[:, aj]
            pred_tie = ranks[:, ai] == ranks[:, aj]
            if want:
                pair_ok = torch.where(
                    pred_tie,
                    torch.zeros_like(correct),
                    pred_first.float(),
                )
            else:
                pair_ok = torch.where(
                    pred_tie,
                    torch.zeros_like(correct),
                    (~pred_first).float(),
                )
            correct = correct + pair_ok
        return correct / float(n_pairs)

    k = int(sims_np.shape[0])
    if n_pairs == 0:
        return np.ones(k, dtype=np.float64)
    a_sims = np.asarray(sims_np[:, js], dtype=np.float64)
    ranks = np.argsort(np.argsort(-a_sims, axis=1), axis=1)
    correct = np.zeros(k, dtype=np.float64)
    for (ai, aj), want, tied in zip(pairs, want_first, reference_tie):
        if tied:
            correct += 1.0
            continue
        pred_first = ranks[:, ai] < ranks[:, aj]
        pred_tie = ranks[:, ai] == ranks[:, aj]
        pair_ok = np.where(
            pred_tie,
            0.0,
            np.where(want, pred_first, ~pred_first),
        )
        correct += pair_ok
    return correct / float(n_pairs)


def _min_adj_gap_np(sims_km: np.ndarray) -> np.ndarray:
    if sims_km.ndim != 2:
        raise ValueError(f"expected [K,M], got {sims_km.shape}")
    k, m = sims_km.shape
    if m < 2:
        return np.zeros(k, dtype=np.float64)
    ordered = np.sort(sims_km, axis=1)
    return np.min(np.diff(ordered, axis=1), axis=1)


def _min_adj_gap_torch(sims_km: torch.Tensor) -> torch.Tensor:
    if sims_km.ndim != 2:
        raise ValueError(f"expected [K,M], got {tuple(sims_km.shape)}")
    if sims_km.shape[1] < 2:
        return torch.zeros(sims_km.shape[0], device=sims_km.device, dtype=sims_km.dtype)
    ordered, _ = torch.sort(sims_km, dim=1)
    return (ordered[:, 1:] - ordered[:, :-1]).min(dim=1).values


def _pool_std_and_gap(
    sims_row: ScoreArray,
) -> Tuple[ScoreArray, ScoreArray]:
    if isinstance(sims_row, torch.Tensor):
        return sims_row.std(dim=1, unbiased=False), _min_adj_gap_torch(sims_row)
    return np.std(sims_row, axis=1), _min_adj_gap_np(np.asarray(sims_row, dtype=np.float64))


def _cascade_std_then_gap(
    *,
    sim_std: ScoreArray,
    min_adj_gap: ScoreArray,
    sweep_keys: Sequence[ScoreMatrixKey],
    rng: np.random.Generator,
    batch_id: int,
    target_id: Optional[str] = None,
    start_pool: Optional[Union[List[int], torch.Tensor]] = None,
) -> Tuple[int, Dict[str, int], float, float]:
    """``max std -> max min_adj_gap -> seeded random``."""
    n = len(sweep_keys)
    if isinstance(sim_std, torch.Tensor):
        pool0 = (
            start_pool
            if start_pool is not None
            else torch.arange(n, device=sim_std.device, dtype=torch.long)
        )
        if not isinstance(pool0, torch.Tensor):
            pool0 = torch.as_tensor(pool0, device=sim_std.device, dtype=torch.long)
        enter_std = int(pool0.numel())
        pool1 = _indices_at_extreme_tensor(sim_std, pool0, sense="max")
        enter_gap = int(pool1.numel())
        pool2 = (
            _indices_at_extreme_tensor(min_adj_gap, pool1, sense="max")
            if enter_gap > 1
            else pool1
        )
        pool_list = [int(x) for x in pool2.detach().cpu().tolist()]
        v_std = float(sim_std[pool_list[0]].item())
        v_gap = float(min_adj_gap[pool_list[0]].item())
    else:
        pool0 = list(start_pool) if start_pool is not None else list(range(n))
        if isinstance(pool0, torch.Tensor):
            pool0 = [int(x) for x in pool0.detach().cpu().tolist()]
        enter_std = len(pool0)
        pool1 = _indices_at_extreme(sim_std, pool0, sense="max")
        enter_gap = len(pool1)
        pool2 = (
            _indices_at_extreme(min_adj_gap, pool1, sense="max")
            if enter_gap > 1
            else list(pool1)
        )
        pool_list = list(pool2)
        v_std = float(sim_std[pool_list[0]])
        v_gap = float(min_adj_gap[pool_list[0]])

    enter_rand = len(pool_list)
    if enter_rand > 1:
        labels = [_format_sweep_key_label(sweep_keys[i]) for i in pool_list]
        where = [f"batch={batch_id}"]
        if target_id is not None:
            where.append(f"target={target_id}")
        warnings.warn(
            f"adaptive tie_break fell back to seeded random ({', '.join(where)}): "
            f"{enter_rand} settings remain after max std > max min_adj_gap; "
            f"choosing among {labels}",
            stacklevel=2,
        )
        best_i = int(pool_list[int(rng.integers(0, len(pool_list)))])
        if isinstance(sim_std, torch.Tensor):
            v_std = float(sim_std[best_i].item())
            v_gap = float(min_adj_gap[best_i].item())
        else:
            v_std = float(sim_std[best_i])
            v_gap = float(min_adj_gap[best_i])
    else:
        best_i = int(pool_list[0])

    # CSV / summarize_tie_break_stages slots: astd <- std-stage, med <- gap-stage
    stage_n = {
        "enter_astd": enter_std,
        "enter_med": enter_gap,
        "enter_random": enter_rand,
    }
    return best_i, stage_n, v_std, v_gap


def _select_after_max_acc(
    *,
    acc: ScoreArray,
    sim_std: ScoreArray,
    min_adj_gap: ScoreArray,
    sweep_keys: Sequence[ScoreMatrixKey],
    rng: np.random.Generator,
    batch_id: int,
    target_id: str,
) -> Tuple[int, Dict[str, int], float, float, float]:
    if isinstance(acc, torch.Tensor):
        tied = _indices_at_max_tensor(acc, None)
        best_acc = float(acc[tied[0]].item())
        start = tied
        enter_acc = int(tied.numel())
    else:
        tied = _indices_at_max(acc, list(range(len(sweep_keys))))
        best_acc = float(acc[tied[0]])
        start = tied
        enter_acc = len(tied)
    best_i, stage_n, v_std, v_gap = _cascade_std_then_gap(
        sim_std=sim_std,
        min_adj_gap=min_adj_gap,
        sweep_keys=sweep_keys,
        rng=rng,
        batch_id=batch_id,
        target_id=target_id,
        start_pool=start,
    )
    # First slot = #tied on max acc (calibration-compatible semantics).
    stage_n["enter_astd"] = enter_acc
    return best_i, stage_n, best_acc, v_std, v_gap


def _batch_sims_km(
    *,
    score_matrices: Dict[ScoreMatrixKey, np.ndarray],
    sweep_keys: Sequence[ScoreMatrixKey],
    batch_local: Sequence[int],
    device: Optional[torch.device],
    stacked_scores: Optional[torch.Tensor],
) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
    keys = list(sweep_keys)
    if device is not None:
        if stacked_scores is None:
            stacked_scores = stack_score_matrices(score_matrices, keys, device)
        batch_idx = torch.as_tensor(batch_local, device=device, dtype=torch.long)
        sims_t = stacked_scores.index_select(1, batch_idx).mean(dim=1)
        return sims_t.detach().cpu().numpy(), sims_t
    rows = []
    idx = np.asarray(batch_local, dtype=np.int64)
    for key in keys:
        sm = np.asarray(score_matrices[key], dtype=np.float64)
        rows.append(sm[idx].mean(axis=0))
    return np.stack(rows, axis=0), None


def _sims_for_setting(
    *,
    best_i: int,
    keys: Sequence[ScoreMatrixKey],
    model_ids: List[str],
    batch_local: Sequence[int],
    score_matrices: Dict[ScoreMatrixKey, np.ndarray],
    sims_t: Optional[torch.Tensor],
) -> Dict[str, float]:
    if sims_t is not None:
        return _sims_dict_from_means(sims_t[best_i], model_ids)
    return similarities_from_score_matrix(
        score_matrices[keys[best_i]], model_ids, batch_local
    )


def _row_template(
    *,
    batch_spec: Tuple[int, int, int, int],
    target_id: str,
    pred_rank: int,
    key: ScoreMatrixKey,
    anchor_pairwise_acc: float,
    v_std: float,
    v_gap: float,
    acc: float,
    n_ok: int,
    n_anchors: int,
    target_sim: float,
    stage_n: Dict[str, int],
    n_tied_at_best: int,
) -> Dict[str, object]:
    batch_id, sample_start, sample_end, batch_n = batch_spec
    edge, sigma, ds, thr, metric = key
    return {
        "batch_id": batch_id,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "n_samples": batch_n,
        "target_id": target_id,
        "pred_rank": pred_rank,
        "chosen_edge": edge,
        "chosen_sigma": sigma,
        "chosen_downscale": ds,
        "chosen_threshold_mode": thr[0],
        "chosen_threshold": thr[1],
        "chosen_metric": metric,
        "anchor_pairwise_acc": float(anchor_pairwise_acc),
        "chosen_anchor_sim_std": float(v_std),
        "chosen_target_med_dev": float(v_gap),
        "pa": acc,
        "n_correct": n_ok,
        "n_anchors": n_anchors,
        "target_sim": target_sim,
        "n_tied_at_best": int(n_tied_at_best),
        "n_enter_astd": int(stage_n["enter_astd"]),
        "n_enter_med": int(stage_n["enter_med"]),
        "n_enter_random": int(stage_n["enter_random"]),
        "tied_details": [],
    }


def evaluate_adaptive_batch(
    *,
    protocol: str,
    score_matrices: Dict[ScoreMatrixKey, np.ndarray],
    sweep_keys: Sequence[ScoreMatrixKey],
    target_ids: List[str],
    model_ids: List[str],
    reference_by_id: Dict[str, Dict[str, object]],
    batch_spec: Tuple[int, int, int, int],
    local_sample_indices: Sequence[int],
    device: Optional[torch.device] = None,
    stacked_scores: Optional[torch.Tensor] = None,
    num_anchors: Optional[int] = None,
    anchor_seed: int = 0,
    tie_break_seed: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}")
    use_anchors = protocol == "calibrated"
    tie_key = _TIE_BREAK_KEY[protocol]

    batch_id, sample_start, sample_end, batch_n = batch_spec
    batch_local = _batch_local_indices(local_sample_indices, sample_start, sample_end)
    keys = list(sweep_keys)
    mid_to_j = {mid: j for j, mid in enumerate(model_ids)}

    sims_np, sims_t = _batch_sims_km(
        score_matrices=score_matrices,
        sweep_keys=keys,
        batch_local=batch_local,
        device=device,
        stacked_scores=stacked_scores,
    )

    per_target_rows: List[Dict[str, object]] = []
    accs: List[float] = []

    if protocol == "oracle":
        pool = sims_t if sims_t is not None else sims_np
        std_k, gap_k = _pool_std_and_gap(pool)
        loo_acc_k = _mean_loo_acc_per_setting(
            sims_np=sims_np,
            sims_t=sims_t,
            model_ids=model_ids,
            reference_by_id=reference_by_id,
        )
        best_i, stage_n, best_loo, v_std, v_gap = _select_after_max_acc(
            acc=loo_acc_k,
            sim_std=std_k,
            min_adj_gap=gap_k,
            sweep_keys=keys,
            rng=_tie_break_rng(tie_break_seed, batch_id, f"oracle:{tie_key}"),
            batch_id=batch_id,
            target_id="__oracle_batch__",
        )
        sims = _sims_for_setting(
            best_i=best_i,
            keys=keys,
            model_ids=model_ids,
            batch_local=batch_local,
            score_matrices=score_matrices,
            sims_t=sims_t,
        )
        pred_ranks = assign_score_ranks_with_ties(model_ids, sims)
        anchors_by_target = sample_anchors_by_target(
            model_ids, target_ids, None, np.random.default_rng(0)
        )
        for target_id in target_ids:
            checks = anchor_checks_for_target(
                target_id, anchors_by_target[target_id], reference_by_id
            )
            acc, n_ok, _ = pairwise_agreement_for_target(target_id, sims, checks)
            per_target_rows.append(
                _row_template(
                    batch_spec=batch_spec,
                    target_id=target_id,
                    pred_rank=pred_ranks[target_id],
                    key=keys[best_i],
                    anchor_pairwise_acc=best_loo,
                    v_std=v_std,
                    v_gap=v_gap,
                    acc=acc,
                    n_ok=n_ok,
                    n_anchors=len(checks),
                    target_sim=sims[target_id],
                    stage_n=stage_n,
                    n_tied_at_best=int(stage_n["enter_astd"]),
                )
            )
            accs.append(acc)
    elif not use_anchors:
        pool = sims_t if sims_t is not None else sims_np
        std_k, gap_k = _pool_std_and_gap(pool)
        best_i, stage_n, v_std, v_gap = _cascade_std_then_gap(
            sim_std=std_k,
            min_adj_gap=gap_k,
            sweep_keys=keys,
            rng=_tie_break_rng(tie_break_seed, batch_id, f"noanc:{tie_key}"),
            batch_id=batch_id,
            target_id=None,
        )
        sims = _sims_for_setting(
            best_i=best_i,
            keys=keys,
            model_ids=model_ids,
            batch_local=batch_local,
            score_matrices=score_matrices,
            sims_t=sims_t,
        )
        pred_ranks = assign_score_ranks_with_ties(model_ids, sims)
        anchors_by_target = sample_anchors_by_target(
            model_ids, target_ids, None, np.random.default_rng(0)
        )
        for target_id in target_ids:
            checks = anchor_checks_for_target(
                target_id, anchors_by_target[target_id], reference_by_id
            )
            acc, n_ok, _ = pairwise_agreement_for_target(target_id, sims, checks)
            per_target_rows.append(
                _row_template(
                    batch_spec=batch_spec,
                    target_id=target_id,
                    pred_rank=pred_ranks[target_id],
                    key=keys[best_i],
                    anchor_pairwise_acc=float("nan"),
                    v_std=v_std,
                    v_gap=v_gap,
                    acc=acc,
                    n_ok=n_ok,
                    n_anchors=len(checks),
                    target_sim=sims[target_id],
                    stage_n=stage_n,
                    n_tied_at_best=int(stage_n["enter_med"]),
                )
            )
            accs.append(acc)
    else:
        if num_anchors is None:
            raise ValueError("num_anchors required for calibrated")
        anchor_ids, eval_targets = _sample_batch_anchors(
            model_ids,
            int(num_anchors),
            seed=int(anchor_seed) + int(batch_id),
        )
        pool_js = [mid_to_j[m] for m in anchor_ids]
        if sims_t is not None:
            pool_t = sims_t.index_select(
                1,
                torch.as_tensor(pool_js, device=sims_t.device, dtype=torch.long),
            )
            std_k, gap_k = _pool_std_and_gap(pool_t)
        else:
            std_k, gap_k = _pool_std_and_gap(sims_np[:, pool_js])
        acc_k = _anchor_pairwise_acc_per_setting(
            sims_np=sims_np,
            sims_t=sims_t,
            anchor_ids=anchor_ids,
            mid_to_j=mid_to_j,
            reference_by_id=reference_by_id,
        )
        best_i, stage_n, best_acc, v_std, v_gap = _select_after_max_acc(
            acc=acc_k,
            sim_std=std_k,
            min_adj_gap=gap_k,
            sweep_keys=keys,
            rng=_tie_break_rng(
                tie_break_seed, batch_id, f"anc:{tie_key}:k{num_anchors}"
            ),
            batch_id=batch_id,
            target_id="__batch_anchors__",
        )
        sims = _sims_for_setting(
            best_i=best_i,
            keys=keys,
            model_ids=model_ids,
            batch_local=batch_local,
            score_matrices=score_matrices,
            sims_t=sims_t,
        )
        pred_ranks = assign_score_ranks_with_ties(model_ids, sims)
        batch_pa, n_ok, _n_pairs = _pairwise_acc_with_targets(
            reference_by_id, pred_ranks, model_ids, eval_targets
        )
        # One PA per batch; duplicate onto each model row for CSV compatibility.
        for mid in model_ids:
            per_target_rows.append(
                _row_template(
                    batch_spec=batch_spec,
                    target_id=mid,
                    pred_rank=pred_ranks[mid],
                    key=keys[best_i],
                    anchor_pairwise_acc=best_acc,
                    v_std=v_std,
                    v_gap=v_gap,
                    acc=batch_pa,
                    n_ok=n_ok,
                    n_anchors=len(anchor_ids),
                    target_sim=sims[mid],
                    stage_n=stage_n,
                    n_tied_at_best=int(stage_n["enter_astd"]),
                )
            )
        accs.append(batch_pa)

    batch_mean = {
        "batch_id": batch_id,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "n_samples": batch_n,
        "mean_pa": float(np.mean(accs)) if accs else float("nan"),
    }
    return per_target_rows, batch_mean


def _format_adaptive_tie_log(
    stats: Dict[str, float],
    *,
    use_acc_stage: bool,
    per_batch: bool = False,
    include_active: bool = True,
) -> str:
    """Like format_tie_break_stage_log but label stages as std/gap (and acc)."""
    base = format_tie_break_stage_log(
        stats,
        group_label="batches" if per_batch else "batch x target groups",
        include_active=include_active,
    )
    if use_acc_stage:
        return (
            base.replace("enter_astd=", "enter_acc(=astd)=")
            .replace("enter_med=", "enter_gap(=med)=")
            .replace("astd_active=", "acc_tied_active=")
            .replace("med_active=", "gap_active=")
        )
    return (
        base.replace("enter_astd=", "enter_std(=astd)=")
        .replace("enter_med=", "enter_gap(=med)=")
        .replace("astd_active=", "std_tied_active=")
        .replace("med_active=", "gap_active=")
    )


def _write_protocol_summary(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "protocol",
        "num_anchors",
        "probe_budget",
        "mean_acc",
        "std_acc",
        "n_batches",
        "n_target_batch",
        "n_settings",
        "run_label",
        "cascade",
        "mean_enter_astd",
        "mean_enter_med",
        "mean_enter_random",
        "n_groups_astd_active",
        "n_groups_med_active",
        "n_groups_random_active",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    args = parse_args()
    set_reproducible(args.seed)
    eval_device = resolve_rank_eval_device(args.cpu)
    eval_backend = "cpu" if eval_device is None else str(eval_device)
    probe_budgets = parse_probe_budgets(args.probe_budgets)
    protocols = _resolve_protocols(args.protocol)
    need_anchors = "calibrated" in protocols
    k_list: List[Optional[int]] = (
        _parse_num_anchors_list(args.num_anchors) if need_anchors else []
    )

    configurations, ranking_metrics, _base, sweep_keys = parse_configuration_args(args)
    reference_entries = load_reference_xml(args.reference_xml)
    model_names, _labels, reference_by_id = models_from_reference_xml(reference_entries)
    target_ids = list(model_names)
    if need_anchors:
        _validate_num_anchors_vs_pool(k_list, len(model_names))

    sample_indices = sample_indices_from_range(args.start_idx, args.end_idx)
    n_samples = len(sample_indices)

    validate_raw_dirs(args.raw_dir)
    if len(args.raw_dir) > 1:
        model_raw_dir = resolve_model_raw_dirs(args.raw_dir, model_names)
        for raw_root in args.raw_dir:
            n = sum(1 for d in model_raw_dir.values() if d == raw_root)
            print(f"  raw_dir {raw_root}: {n} models", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    anchor_log = "n/a"
    if need_anchors:
        parts = [
            "all(=M-1)" if k is None else str(k) for k in k_list
        ]
        anchor_log = ",".join(parts)
    print(
        f"Rank adaptive | models={len(model_names)} samples={n_samples} "
        f"sample_range=[{args.start_idx},{args.end_idx}) "
        f"raw_dirs={len(args.raw_dir)} candidates={len(sweep_keys)} "
        f"probe_budgets={probe_budgets} eval={eval_backend} "
        f"protocols={protocols} cascade={CASCADE} "
        f"anchors={anchor_log} "
        f"anchor_seed={args.anchor_seed if need_anchors else 'n/a'}",
        flush=True,
    )

    t_load = time.perf_counter()
    score_matrices = load_score_matrices_for_sweep(
        args.raw_dir,
        configurations,
        ranking_metrics,
        model_names,
        sample_indices,
    )
    load_sec = time.perf_counter() - t_load
    print(f"Loaded {len(score_matrices)} score matrices in {load_sec:.1f}s", flush=True)

    stacked_scores = None
    if eval_device is not None:
        t_up = time.perf_counter()
        stacked_scores = stack_score_matrices(score_matrices, sweep_keys, eval_device)
        print(
            f"Uploaded score tensors to {eval_device} in "
            f"{time.perf_counter() - t_up:.1f}s",
            flush=True,
        )

    n_models = len(model_names)
    n_settings = len(sweep_keys)
    summary_rows: List[Dict[str, object]] = []

    for protocol in protocols:
        use_anchors = protocol == "calibrated"
        cascade = _cascade_label(protocol)
        run_ks: List[Optional[int]] = k_list if use_anchors else [None]

        for num_anchors in run_ks:
            if use_anchors:
                resolved_k = _resolve_num_anchors(num_anchors, n_models)
                if num_anchors is None:
                    run_label = f"adaptive_{protocol}"
                    k_tag = ""
                    anchor_note = (
                        f"all(=M-1={resolved_k}),anchor_seed={args.anchor_seed}+batch_id,"
                        f"eval=target_pairs_excl_anchor_anchor"
                    )
                    num_anchors_label: object = "all"
                else:
                    run_label = f"adaptive_{protocol}_k{resolved_k}"
                    k_tag = f"_k{resolved_k}"
                    anchor_note = (
                        f"k={resolved_k},anchor_seed={args.anchor_seed}+batch_id,"
                        f"eval=target_pairs_excl_anchor_anchor"
                    )
                    num_anchors_label = str(resolved_k)
            else:
                run_label = f"adaptive_{protocol}"
                k_tag = ""
                anchor_note = "none" if protocol != "oracle" else "full_pool_reference"
                resolved_k = None
                num_anchors_label = ""

            for probe_budget in probe_budgets:
                batch_specs = iter_ranking_batches(
                    n_samples, probe_budget, args.start_idx
                )
                all_rows: List[Dict[str, object]] = []
                batch_means: List[Dict[str, object]] = []
                score_sec = 0.0
                for batch_spec in batch_specs:
                    t0 = time.perf_counter()
                    rows, batch_mean = evaluate_adaptive_batch(
                        protocol=protocol,
                        score_matrices=score_matrices,
                        sweep_keys=sweep_keys,
                        target_ids=target_ids,
                        model_ids=model_names,
                        reference_by_id=reference_by_id,
                        batch_spec=batch_spec,
                        local_sample_indices=sample_indices,
                        device=eval_device,
                        stacked_scores=stacked_scores,
                        num_anchors=resolved_k,
                        anchor_seed=int(args.anchor_seed),
                        tie_break_seed=args.seed,
                    )
                    score_sec += time.perf_counter() - t0
                    all_rows.extend(rows)
                    batch_means.append(batch_mean)

                csv_path = os.path.join(
                    args.out_dir,
                    f"{args.prefix}_{protocol}{k_tag}_bs{probe_budget}.csv",
                )
                extra = (
                    f"candidates={n_settings},protocol={protocol},"
                    f"anchors={anchor_note},cascade={cascade},"
                    f"tie_break=std_gap,"
                    f"note=chosen_target_med_dev_stores_min_adj_gap;"
                    f"enter_astd=std_or_acc_pool,enter_med=gap_pool"
                )
                timings = {
                    "score_total": score_sec,
                    "score_mean_per_model": score_sec / n_models if n_models else 0.0,
                    "n_models": float(n_models),
                    "n_ranking_batches": float(len(batch_specs)),
                }
                write_loo_csv(
                    csv_path=csv_path,
                    model_ids=model_names,
                    reference_by_id=reference_by_id,
                    probe_budget=probe_budget,
                    run_label=run_label,
                    edge=None,
                    sigma=None,
                    downscale=None,
                    threshold=None,
                    metric=None,
                    total_samples=n_samples,
                    per_target_rows=all_rows,
                    batch_means=batch_means,
                    extra_summary=extra,
                    timings=timings,
                    n_settings=n_settings,
                    tie_unique_by_batch=use_anchors,
                )
                # All adaptive protocols: mean/std over ranking-batch scores.
                pair_accs = [
                    float(b["mean_pa"]) for b in batch_means
                ]
                mean_acc = float(np.mean(pair_accs)) if pair_accs else float("nan")
                std_acc = (
                    float(np.std(pair_accs, ddof=0)) if pair_accs else float("nan")
                )
                print(
                    f"Saved {protocol}{k_tag} bs={probe_budget}: {csv_path}  "
                    f"mean_acc={mean_acc:.4f} std={std_acc:.4f} "
                    f"(over {len(pair_accs)} batches) cascade={cascade}",
                    flush=True,
                )
                # Tie-active stats only for calibrated (shared config per batch).
                if use_anchors:
                    stage_stats = summarize_tie_break_stages(
                        all_rows,
                        n_settings=n_settings,
                        unique_by_batch=True,
                    )
                    print(
                        f"  {_format_adaptive_tie_log(stage_stats, use_acc_stage=True, per_batch=True)}",
                        flush=True,
                    )
                    active_astd = int(stage_stats["n_groups_astd_active"])
                    active_med = int(stage_stats["n_groups_med_active"])
                    active_rand = int(stage_stats["n_groups_random_active"])
                    mean_astd = f"{stage_stats['mean_enter_astd']:.6f}"
                    mean_med = f"{stage_stats['mean_enter_med']:.6f}"
                    mean_rand = f"{stage_stats['mean_enter_random']:.6f}"
                else:
                    active_astd = ""
                    active_med = ""
                    active_rand = ""
                    mean_astd = ""
                    mean_med = ""
                    mean_rand = ""
                summary_rows.append(
                    {
                        "protocol": protocol,
                        "num_anchors": num_anchors_label,
                        "probe_budget": int(probe_budget),
                        "mean_acc": f"{mean_acc:.6f}",
                        "std_acc": f"{std_acc:.6f}",
                        "n_batches": len(batch_means),
                        "n_target_batch": len(pair_accs),
                        "n_settings": n_settings,
                        "run_label": run_label,
                        "cascade": cascade,
                        "mean_enter_astd": mean_astd,
                        "mean_enter_med": mean_med,
                        "mean_enter_random": mean_rand,
                        "n_groups_astd_active": active_astd,
                        "n_groups_med_active": active_med,
                        "n_groups_random_active": active_rand,
                    }
                )

    if summary_rows:
        summary_path = os.path.join(
            args.out_dir, f"{args.prefix}_adaptive_strategies_summary.csv"
        )
        _write_protocol_summary(summary_path, summary_rows)
        print(f"Saved protocol summary: {summary_path}", flush=True)

    print(
        f"Done. protocols={protocols} batch_sizes={probe_budgets} "
        f"n_settings={n_settings}",
        flush=True,
    )


if __name__ == "__main__":
    main()
