#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute LM-SimBench-style metrics described in the paper's Benchmark Design section,
using only data available for the Calculon vs profiler qwen15_1.8b_8k comparison.

Outputs a long-form CSV under result/outputs/.

System-level: iteration MAE/sMAPE (+ optional R²), peak-memory MAE/sMAPE (+ optional R²),
  optional memory-feasibility agreement vs a device memory threshold (GB).

Strategy-level: Spearman rho, Top-k hit/regret on iteration time (all configs, and
  optionally only configs predicted feasible under --device-memory-gb).

Component-level tasks (operator category / matrix op / comm) are not computable without
the corresponding CSVs — emitted as N/A rows with notes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MERGE_CSV = OUTPUT_DIR / "calculon_vs_profiler_merge_qwen15_1.8b_8k.csv"
STEP_TIME_CSV = BASE_DIR / "step_time_summary.csv"
PEAK_MEMORY_CSV = BASE_DIR / "peak_memory.csv"
CALCULON_METRICS_CSV = BASE_DIR / "qwen15_1.8b_8k_metrics.csv"
# Memory feasibility: CP=1 configs from step_time_summary vs 64 GB device budget;
# Calculon CSV omits configs whose predicted peak exceeds the budget.
MEMORY_FEASIBILITY_BUDGET_GB = 64.0

# benchmark/src/ holds the lm_simbench package; this script lives at
# benchmark/docx/case_1/result/, so the package root is three levels up.
_LM_SIMBENCH_SRC = BASE_DIR.parents[2] / "src"
if _LM_SIMBENCH_SRC.is_dir():
    sys.path.insert(0, str(_LM_SIMBENCH_SRC))

from lm_simbench.metrics import mae, r2_score, smape, spearman_correlation  # noqa: E402
from lm_simbench.ranking import (  # noqa: E402
    strategy_ranking_summary,
    top_k_hit_rate,
    top_k_regret_ms,
)


def _rows() -> list[dict]:
    return []


def add_row(
    rows: list[dict],
    *,
    section: str,
    task: str,
    metric: str,
    value,
    unit: str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "task": task,
            "metric": metric,
            "value": value,
            "unit": unit,
            "notes": notes,
        }
    )


def subset_feasible_by_pred(df: pd.DataFrame, threshold_gb: float) -> pd.DataFrame:
    """Candidate configs where Calculon peak memory is within device budget."""
    return df.loc[df["pred_peak_gb"].astype(float) <= threshold_gb].copy()


def cp_from_parallel_config(config: str) -> int | None:
    for tok in str(config).split("-"):
        if tok.startswith("cp") and tok[2:].isdigit():
            return int(tok[2:])
    return None


def load_cp1_feasibility_frame(
    *,
    budget_gb: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    All optimization_config with CP=1 from step_time_summary, merged with measured
    peak (max rank, GB) and Calculon predicted peak when present.
    Configs absent from Calculon CSV are treated as predicted over budget (infeasible).
    """
    if not STEP_TIME_CSV.exists():
        raise FileNotFoundError(STEP_TIME_CSV)
    if not PEAK_MEMORY_CSV.exists():
        raise FileNotFoundError(PEAK_MEMORY_CSV)
    if not CALCULON_METRICS_CSV.exists():
        raise FileNotFoundError(CALCULON_METRICS_CSV)

    step = pd.read_csv(STEP_TIME_CSV)
    peak = pd.read_csv(PEAK_MEMORY_CSV)
    calc = pd.read_csv(CALCULON_METRICS_CSV)

    step = step.rename(columns={"optimization_config": "config"})
    step["cp"] = step["config"].map(cp_from_parallel_config)
    cp1 = step.loc[step["cp"] == 1, "config"].astype(str).tolist()

    peak_agg = (
        peak.groupby("optimization_config", as_index=False)["peak_memory_mb"]
        .max()
        .rename(columns={"optimization_config": "config", "peak_memory_mb": "peak_mb_max"})
    )
    peak_agg["actual_peak_gb"] = peak_agg["peak_mb_max"].astype(float) / 1024.0

    calc_pred = calc.rename(columns={"parallel_config": "config"})[
        ["config", "peak_memory"]
    ].copy()
    calc_pred = calc_pred.rename(columns={"peak_memory": "pred_peak_gb"})
    calc_pred["pred_peak_gb"] = calc_pred["pred_peak_gb"].astype(float)

    df = pd.DataFrame({"config": cp1})
    df = df.merge(peak_agg[["config", "actual_peak_gb"]], on="config", how="left")
    df = df.merge(calc_pred, on="config", how="left")

    missing_peak = df["actual_peak_gb"].isna()
    if missing_peak.any():
        bad = df.loc[missing_peak, "config"].tolist()
        raise ValueError(f"peak_memory.csv missing configs: {bad}")

    df["has_calculon_peak"] = df["pred_peak_gb"].notna()
    # Absent row ⇒ Calculon predicted peak above budget ⇒ predicted infeasible.
    df["predicted_feasible"] = np.where(
        df["has_calculon_peak"],
        df["pred_peak_gb"].astype(float) <= budget_gb,
        False,
    )
    df["measured_feasible"] = df["actual_peak_gb"].astype(float) <= budget_gb

    meta = {
        "n_cp1_configs": int(len(df)),
        "n_with_calculon_row": int(df["has_calculon_peak"].sum()),
        "n_calculon_excluded_over_budget": int((~df["has_calculon_peak"]).sum()),
    }
    return df, meta


def add_memory_feasibility_cp1_rows(
    rows: list[dict],
    *,
    budget_gb: float,
) -> None:
    feas_df, meta = load_cp1_feasibility_frame(budget_gb=budget_gb)
    agree = np.mean(
        feas_df["measured_feasible"].to_numpy()
        == feas_df["predicted_feasible"].to_numpy()
    )
    meas_feas = feas_df["measured_feasible"].to_numpy()
    pred_feas = feas_df["predicted_feasible"].to_numpy()

    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="device_budget_gb",
        value=budget_gb,
        unit="GB",
        notes="step_time_summary configs with CP=1 only",
    )
    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="n_configs_cp1_total",
        value=meta["n_cp1_configs"],
        unit="count",
        notes="from step_time_summary.csv",
    )
    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="n_configs_with_calculon_peak_in_csv",
        value=meta["n_with_calculon_row"],
        unit="count",
        notes="qwen15_1.8b_8k_metrics.csv (predicted peak <= budget row retained)",
    )
    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="n_configs_calculon_missing_predicted_over_budget",
        value=meta["n_calculon_excluded_over_budget"],
        unit="count",
        notes="no Calculon row ⇒ treat predicted_feasible=False",
    )
    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="feasibility_agreement_rate",
        value=round(float(agree), 6),
        unit="fraction",
        notes="mean(measured_feasible == predicted_feasible); predicted uses CSV peak or OOM",
    )
    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="n_measured_feasible",
        value=int(np.sum(meas_feas)),
        unit="count",
        notes="actual_peak_gb <= budget",
    )
    add_row(
        rows,
        section="system_level",
        task="memory_feasibility_cp1_64gb_device",
        metric="n_predicted_feasible",
        value=int(np.sum(pred_feas)),
        unit="count",
        notes="pred_peak in CSV and <= budget",
    )
    # Confusion-style counts (positive class = feasible under budget)
    tp = int(np.sum(meas_feas & pred_feas))
    tn = int(np.sum(~meas_feas & ~pred_feas))
    fp = int(np.sum(~meas_feas & pred_feas))
    fn = int(np.sum(meas_feas & ~pred_feas))
    for name, val in [
        ("confusion_tp_meas_feas_pred_feas", tp),
        ("confusion_tn_meas_inf_pred_inf", tn),
        ("confusion_fp_meas_inf_pred_feas", fp),
        ("confusion_fn_meas_feas_pred_inf", fn),
    ]:
        add_row(
            rows,
            section="system_level",
            task="memory_feasibility_cp1_64gb_device",
            metric=name,
            value=val,
            unit="count",
            notes="feasible ≡ peak<=budget; pred_feas uses Calculon CSV or missing→infeasible",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper benchmark metrics → CSV")
    parser.add_argument(
        "--merge-csv",
        type=Path,
        default=DEFAULT_MERGE_CSV,
        help="Merged Calculon vs profiler table",
    )
    parser.add_argument(
        "--device-memory-gb",
        type=float,
        default=MEMORY_FEASIBILITY_BUDGET_GB,
        help="Device budget for predicted-feasible ranking subset (default: same as CP1 feasibility)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "benchmark_metrics_calculon_qwen15_1p8b.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5",
        help="Comma-separated k for Top-k metrics",
    )
    args = parser.parse_args()

    if not args.merge_csv.exists():
        raise FileNotFoundError(args.merge_csv)

    df = pd.read_csv(args.merge_csv)
    need = {
        "config",
        "actual_iteration_s",
        "pred_iteration_s",
        "actual_peak_gb",
        "pred_peak_gb",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{args.merge_csv} missing columns: {sorted(missing)}")

    configs = df["config"].astype(str).tolist()
    y_true_s = df["actual_iteration_s"].astype(float).to_numpy()
    y_pred_s = df["pred_iteration_s"].astype(float).to_numpy()
    y_true_ms = y_true_s * 1000.0
    y_pred_ms = y_pred_s * 1000.0

    true_peak = df["actual_peak_gb"].astype(float).to_numpy()
    pred_peak = df["pred_peak_gb"].astype(float).to_numpy()

    rows: list[dict] = []

    # --- Component-level (data not provided) ---
    for task_name, files in [
        (
            "operator_category_latency",
            "operator_latency_by_category.csv",
        ),
        ("matrix_operator_latency", "matrix_operator_latency.csv"),
        ("communication_time", "comm_time.csv / comm_summary.csv"),
    ]:
        add_row(
            rows,
            section="component_level",
            task=task_name,
            metric="n/a",
            value="",
            unit="",
            notes=f"Not computable without {files}",
        )

    # --- System-level: iteration time ---
    add_row(
        rows,
        section="system_level",
        task="iteration_time_prediction",
        metric="mae",
        value=round(mae(y_true_ms, y_pred_ms), 6),
        unit="ms",
        notes="MAE on iteration_time_ms scale (merge actual_iteration_s×1000)",
    )
    add_row(
        rows,
        section="system_level",
        task="iteration_time_prediction",
        metric="mae",
        value=round(mae(y_true_s, y_pred_s), 6),
        unit="s",
        notes="same as ms row / 1000",
    )
    smape_it = smape(y_true_ms, y_pred_ms)
    add_row(
        rows,
        section="system_level",
        task="iteration_time_prediction",
        metric="smape",
        value=round(smape_it, 6),
        unit="percent_0_200",
        notes="LM-SimBench definition: mean(200*|y-ŷ|/(|y|+|ŷ|+ε))",
    )
    r2_it = r2_score(y_true_ms, y_pred_ms)
    add_row(
        rows,
        section="system_level",
        task="iteration_time_prediction",
        metric="r2",
        value=round(r2_it, 6) if np.isfinite(r2_it) else "",
        unit="dimensionless",
        notes="pointwise regression on n configs (same formula as component-level R²)",
    )

    # --- System-level: peak memory ---
    add_row(
        rows,
        section="system_level",
        task="peak_memory_prediction",
        metric="mae",
        value=round(mae(true_peak, pred_peak), 6),
        unit="GB",
        notes="config-level max-over-ranks actual vs Calculon peak (merge CSV)",
    )
    smape_mem = smape(true_peak, pred_peak)
    add_row(
        rows,
        section="system_level",
        task="peak_memory_prediction",
        metric="smape",
        value=round(smape_mem, 6),
        unit="percent_0_200",
        notes="same sMAPE formula on GB values",
    )
    r2_mem = r2_score(true_peak, pred_peak)
    add_row(
        rows,
        section="system_level",
        task="peak_memory_prediction",
        metric="r2",
        value=round(r2_mem, 6) if np.isfinite(r2_mem) else "",
        unit="dimensionless",
        notes="",
    )

    # --- Memory feasibility: CP=1 @ 64 GB (full 20 configs; Calculon omits OOM predictions) ---
    add_memory_feasibility_cp1_rows(rows, budget_gb=MEMORY_FEASIBILITY_BUDGET_GB)

    # --- Strategy ranking: iteration time ---
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    ks = [k for k in ks if k >= 1]
    n = len(configs)
    ks_eff = [k for k in ks if k <= n]
    summ = strategy_ranking_summary(configs, y_true_ms, y_pred_ms, ks=ks_eff)

    add_row(
        rows,
        section="strategy_ranking",
        task="iteration_time_ranking_all_configs",
        metric="spearman_rho",
        value=round(float(summ["spearman_rho"]), 6)
        if np.isfinite(summ["spearman_rho"])
        else "",
        unit="dimensionless",
        notes=f"n_configs={n}",
    )
    for k in ks_eff:
        hit = summ[f"top{k}_hit"]
        reg_ms = summ[f"top{k}_regret_ms"]
        rel = summ[f"top{k}_relative_regret"]
        add_row(
            rows,
            section="strategy_ranking",
            task="iteration_time_ranking_all_configs",
            metric=f"top{k}_hit",
            value=int(bool(hit)),
            unit="0_or_1",
            notes="true best iteration time in top-k by predicted time",
        )
        add_row(
            rows,
            section="strategy_ranking",
            task="iteration_time_ranking_all_configs",
            metric=f"top{k}_regret_ms",
            value=round(float(reg_ms), 6),
            unit="ms",
            notes="vs global best true iteration time",
        )
        add_row(
            rows,
            section="strategy_ranking",
            task="iteration_time_ranking_all_configs",
            metric=f"top{k}_relative_regret",
            value=round(float(rel), 6),
            unit="dimensionless",
            notes="regret / best true time",
        )

    # Ranking on predicted-feasible subset only (merge CSV; Calculon peak <= budget)
    sub = subset_feasible_by_pred(df, float(args.device_memory_gb))
    n_sub = len(sub)
    if n_sub >= 2:
        cs = sub["config"].astype(str).tolist()
        yt = sub["actual_iteration_s"].astype(float).to_numpy() * 1000.0
        yp = sub["pred_iteration_s"].astype(float).to_numpy() * 1000.0
        rho_sub = spearman_correlation(yt, yp)
        add_row(
            rows,
            section="strategy_ranking",
            task="iteration_time_ranking_pred_feasible_subset",
            metric="spearman_rho",
            value=round(float(rho_sub), 6) if np.isfinite(rho_sub) else "",
            unit="dimensionless",
            notes=f"budget_gb={args.device_memory_gb}; pred_peak_gb<=budget; n={n_sub}",
        )
        ks_sub = [k for k in ks if k <= n_sub]
        for k in ks_sub:
            hit = top_k_hit_rate(cs, yt, yp, k)
            reg_ms = top_k_regret_ms(cs, yt, yp, k)
            add_row(
                rows,
                section="strategy_ranking",
                task="iteration_time_ranking_pred_feasible_subset",
                metric=f"top{k}_hit",
                value=int(bool(hit)),
                unit="0_or_1",
                notes=f"budget_gb={args.device_memory_gb}",
            )
            add_row(
                rows,
                section="strategy_ranking",
                task="iteration_time_ranking_pred_feasible_subset",
                metric=f"top{k}_regret_ms",
                value=round(float(reg_ms), 6),
                unit="ms",
                notes="",
            )
    else:
        add_row(
            rows,
            section="strategy_ranking",
            task="iteration_time_ranking_pred_feasible_subset",
            metric="n/a",
            value="",
            unit="",
            notes=f"Bypass: fewer than 2 predicted-feasible configs at budget_gb={args.device_memory_gb}",
        )

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] wrote {args.output} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
