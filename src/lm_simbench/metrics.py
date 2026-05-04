# -*- coding: utf-8 -*-
"""Core regression and ranking metrics (NumPy arrays in, scalar/array out)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D sequences; got shape {a.shape}")
    return a


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def smape(y_true: Any, y_pred: Any, epsilon: float = 1e-12) -> float:
    """
    Symmetric mean absolute percentage error (sMAPE), percentage scale 0–200.

    For each i: 200 * |y_i - ŷ_i| / (|y_i| + |ŷ_i| + epsilon), then mean.
    """
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = np.abs(yt) + np.abs(yp) + epsilon
    return float(np.mean(200.0 * np.abs(yt - yp) / denom))


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Coefficient of determination R². Returns NaN if the true variance is zero."""
    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("nan")
    yt = y_true[mask]
    yp = y_pred[mask]
    y_mean = float(np.mean(yt))
    ss_tot = float(np.sum((yt - y_mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    return float(1.0 - ss_res / ss_tot)


def spearman_correlation(y_true: Any, y_pred: Any) -> float:
    """
    Spearman rank correlation between two vectors (handles ties via average ranks).

    Requires SciPy. Returns NaN if fewer than two finite pairs or zero variance in ranks.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError as e:  # pragma: no cover - optional dependency path
        raise ImportError(
            "spearman_correlation requires scipy. Install with: pip install scipy"
        ) from e

    y_true = _as_float_array(y_true)
    y_pred = _as_float_array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(rho) if np.isfinite(rho) else float("nan")


def regression_metrics(y_true: Any, y_pred: Any, epsilon_smape: float = 1e-12) -> dict[str, float]:
    """Convenience bundle: MAE, sMAPE, R²."""
    return {
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred, epsilon=epsilon_smape),
        "r2": r2_score(y_true, y_pred),
    }
