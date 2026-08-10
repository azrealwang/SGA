"""
Rank models under fixed configurations, from cached SGA alignment scores.

Reads the score matrices under --raw_dir (one or more export roots) and reports
pairwise agreement for each configuration and similarity measure. Two uses:
sweep the candidate space on a development pool to pick the Transfer preset, then
rerun with that single frozen configuration on the target pool.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common.reference import load_reference_xml, models_from_reference_xml
from sga.structural import (
    iter_ranking_batches,
    progress_bar,
    set_reproducible,
    setting_dir_name,
    threshold_spec_label,
)
from sga.rank_io import (
    DEFAULT_PROBE_BUDGETS,
    parse_probe_budgets,
    add_configuration_args,
    build_loo_eval_context,
    evaluate_fixed_setting,
    format_threshold_csv,
    load_score_matrices_for_sweep,
    parse_configuration_args,
    resolve_model_raw_dirs,
    resolve_rank_eval_device,
    sample_indices_from_range,
    upload_score_matrices,
    validate_raw_dirs,
    write_loo_csv,
    write_py2_summary_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pairwise agreement under fixed configurations (Transfer preset selection and application).",
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
    default_bs = ",".join(str(s) for s in DEFAULT_PROBE_BUDGETS)
    p.add_argument(
        "--probe_budgets",
        type=str,
        default=default_bs,
        help=f"Comma-separated ranking batch sizes (default: {default_bs}).",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (NumPy) for batch aggregation (default: CUDA when available).",
    )
    p.add_argument("--seed", type=int, default=0)
    add_configuration_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducible(args.seed)
    eval_device = resolve_rank_eval_device(args.cpu)
    eval_backend = "cpu" if eval_device is None else str(eval_device)

    configurations, ranking_metrics, _base_metrics, sweep_keys = parse_configuration_args(
        args
    )
    probe_budgets = parse_probe_budgets(args.probe_budgets)

    reference_entries = load_reference_xml(args.reference_xml)
    model_names, _labels, reference_by_id = models_from_reference_xml(reference_entries)
    target_ids = list(model_names)

    sample_indices = sample_indices_from_range(args.start_idx, args.end_idx)
    n_samples = len(sample_indices)

    validate_raw_dirs(args.raw_dir)
    if len(args.raw_dir) > 1:
        model_raw_dir = resolve_model_raw_dirs(args.raw_dir, model_names)
        for raw_root in args.raw_dir:
            n = sum(1 for d in model_raw_dir.values() if d == raw_root)
            print(f"  raw_dir {raw_root}: {n} models", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(
        f"Rank fixed settings | models={len(model_names)} samples={n_samples} "
        f"sample_range=[{args.start_idx},{args.end_idx}) "
        f"raw_dirs={len(args.raw_dir)} settings={len(sweep_keys)} "
        f"probe_budgets={probe_budgets} eval={eval_backend}",
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

    loo_ctx = None
    if eval_device is not None:
        t_up = time.perf_counter()
        score_matrices = upload_score_matrices(score_matrices, eval_device)
        loo_ctx = build_loo_eval_context(model_names, reference_by_id, eval_device)
        print(
            f"Uploaded matrices + LOO context to {eval_device} in "
            f"{time.perf_counter() - t_up:.1f}s",
            flush=True,
        )

    for bs in probe_budgets:
        batch_specs = iter_ranking_batches(n_samples, bs, args.start_idx)
        summary_rows = []
        eval_sec = 0.0
        write_sec = 0.0

        for key in progress_bar(
            sweep_keys,
            desc=f"Evaluate bs={bs}",
            total=len(sweep_keys),
            unit="setting",
        ):
            edge, sigma, ds, thr, metric = key
            configuration = (edge, sigma, ds, thr)
            t0 = time.perf_counter()
            rows, batch_means = evaluate_fixed_setting(
                score_matrix=score_matrices[key],
                configuration=configuration,
                ranking_metric=metric,
                target_ids=target_ids,
                model_ids=model_names,
                reference_by_id=reference_by_id,
                batch_specs=batch_specs,
                local_sample_indices=sample_indices,
                device=eval_device,
                loo_ctx=loo_ctx,
            )
            eval_sec += time.perf_counter() - t0
            accs = [float(b["mean_pa"]) for b in batch_means]
            acc_stats = {
                "mean_acc": float(np.mean(accs)),
                "median_acc": float(np.median(accs)),
                "min_acc": float(np.min(accs)),
                "max_acc": float(np.max(accs)),
                "std_acc": float(np.std(accs)),
            }
            setting_dir = setting_dir_name(edge, sigma, ds, thr)
            summary_rows.append(
                {
                    "edge": f"{edge:g}",
                    "sigma": f"{sigma:g}",
                    "downscale": ds,
                    "threshold_mode": thr[0],
                    "threshold": format_threshold_csv(thr[1]),
                    "threshold_label": threshold_spec_label(thr),
                    "metric": metric,
                    "setting_dir": setting_dir,
                    "n_batches": len(batch_means),
                    **acc_stats,
                }
            )
            setting_out = os.path.join(
                args.out_dir,
                "settings",
                setting_dir,
            )
            os.makedirs(setting_out, exist_ok=True)
            csv_path = os.path.join(setting_out, f"{args.prefix}_{metric}_bs{bs}.csv")
            t1 = time.perf_counter()
            write_loo_csv(
                csv_path=csv_path,
                model_ids=model_names,
                reference_by_id=reference_by_id,
                probe_budget=bs,
                run_label="fixed_setting_loo",
                edge=edge,
                sigma=sigma,
                downscale=ds,
                threshold=thr,
                metric=metric,
                total_samples=n_samples,
                per_target_rows=rows,
                batch_means=batch_means,
            )
            write_sec += time.perf_counter() - t1

        t_sum = time.perf_counter()
        summary_path = os.path.join(args.out_dir, f"{args.prefix}_summary_bs{bs}.csv")
        write_py2_summary_csv(summary_path, probe_budget=bs, rows=summary_rows)
        summary_sec = time.perf_counter() - t_sum
        print(
            f"Saved summary: {summary_path} ({len(summary_rows)} settings)  "
            f"eval={eval_sec:.1f}s write_csv={write_sec:.1f}s summary={summary_sec:.2f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
