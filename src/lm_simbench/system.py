# -*- coding: utf-8 -*-
"""System-level tasks: iteration time and memory feasibility."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .metrics import mae, regression_metrics, smape


def iteration_time_metrics_ms(y_true_ms: Any, y_pred_ms: Any) -> dict[str, float]:
    """
    Iteration time prediction metrics (milliseconds).

    Returns keys: mae_ms, smape, r2 (R² is included for consistency; primary metrics per spec are MAE and sMAPE).
    """
    reg = regression_metrics(y_true_ms, y_pred_ms)
    return {
        "mae_ms": reg["mae"],
        "smape": reg["smape"],
        "r2": reg["r2"],
    }


def aggregate_config_peak_memory_mb(
    peak_memory_df: pd.DataFrame,
    *,
    config_col: str = "optimization_config",
    rank_col: str = "rank_id",
    mem_col: str = "peak_memory_mb",
) -> pd.Series:
    """Per-configuration peak memory = max peak over ranks (MB)."""
    required = {config_col, rank_col, mem_col}
    missing = required - set(peak_memory_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = peak_memory_df[[config_col, rank_col, mem_col]].copy()
    df[mem_col] = pd.to_numeric(df[mem_col], errors="coerce")
    return df.groupby(config_col, sort=False)[mem_col].max()


def memory_peak_mae_mb(y_true_mb: Any, y_pred_mb: Any) -> float:
    """Mean absolute error on configuration-level peak memory (MB)."""
    return mae(y_true_mb, y_pred_mb)


def memory_peak_metrics_mb(y_true_mb: Any, y_pred_mb: Any) -> dict[str, float]:
    """Peak memory regression: MAE and sMAPE on MB; R² included."""
    return {
        "mae_peak_memory_mb": mae(y_true_mb, y_pred_mb),
        "smape_peak_memory": smape(y_true_mb, y_pred_mb),
        "r2_peak_memory": regression_metrics(y_true_mb, y_pred_mb)["r2"],
    }


def feasibility_from_peak_mb(peak_mb: Any, device_memory_threshold_mb: float) -> np.ndarray:
    """True where peak_memory <= threshold (feasible on device)."""
    peak = np.asarray(peak_mb, dtype=float)
    return np.isfinite(peak) & (peak <= float(device_memory_threshold_mb))


def memory_feasibility_classification_metrics(
    true_peak_mb: Any,
    pred_peak_mb: Any,
    device_memory_threshold_mb: float,
) -> dict[str, float]:
    """
    Compare feasibility labels derived from true vs predicted peaks.

    Labels: feasible iff peak <= threshold. Returns accuracy and balanced error rate.
    """
    y_true = feasibility_from_peak_mb(true_peak_mb, device_memory_threshold_mb)
    y_pred = feasibility_from_peak_mb(pred_peak_mb, device_memory_threshold_mb)
    y_true = np.asarray(y_true, dtype=bool).ravel()
    y_pred = np.asarray(y_pred, dtype=bool).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("true_peak_mb and pred_peak_mb must align")
    mask = np.isfinite(np.asarray(true_peak_mb, dtype=float).ravel()) & np.isfinite(
        np.asarray(pred_peak_mb, dtype=float).ravel()
    )
    if not np.any(mask):
        return {"feasibility_accuracy": float("nan"), "feasible_prevalence_true": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    acc = float(np.mean(yt == yp))
    prev = float(np.mean(yt))
    return {
        "feasibility_accuracy": acc,
        "feasible_prevalence_true": prev,
    }


def align_configs_to_series(
    configs: list[str] | np.ndarray,
    values_by_config: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned (configs, values) arrays; skips configs missing from the mapping."""
    keys: list[str] = []
    vals: list[float] = []
    for c in configs:
        cs = str(c)
        if cs in values_by_config:
            keys.append(cs)
            vals.append(float(values_by_config[cs]))
    return np.array(keys, dtype=object), np.asarray(vals, dtype=float)
