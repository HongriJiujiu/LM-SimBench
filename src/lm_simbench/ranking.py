# -*- coding: utf-8 -*-
"""Parallel strategy ranking evaluation for a single model (one set of candidate configs)."""

from __future__ import annotations

import operator
from typing import Any, Iterable, Mapping

import numpy as np

from .metrics import spearman_correlation


def _as_config_order(configs: Iterable[Any]) -> list[str]:
    names = [str(c) for c in configs]
    if not names or len(set(names)) != len(names):
        raise ValueError("configs must be nonempty and unique")
    return names


def _as_time_vector(values: Any, name: str) -> np.ndarray:
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not times.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError(f"{name} must contain finite, nonnegative times")
    return times


def _aligned_times(configs: Iterable[Any], true: Any, predicted: Any):
    names = _as_config_order(configs)
    y_true = _as_time_vector(true, "iteration_time_true_ms")
    y_pred = _as_time_vector(predicted, "iteration_time_pred_ms")
    if len(names) != len(y_true) or y_true.shape != y_pred.shape:
        raise ValueError("configs, iteration_time_true_ms, iteration_time_pred_ms must align")
    return names, y_true, y_pred


def _validate_k(k: int) -> int:
    try:
        value = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be a positive integer") from exc
    if value < 1:
        raise ValueError("k must be >= 1")
    return value


def true_best_config(configs: Iterable[Any], iteration_time_ms: Any) -> str:
    """Config with minimum finite time; ties retain the input order."""
    cs = _as_config_order(configs)
    t = _as_time_vector(iteration_time_ms, "iteration_time_ms")
    if len(cs) != t.size:
        raise ValueError("configs and iteration_time_ms must have the same length")
    i = int(np.argmin(t))
    return cs[i]


def spearman_iteration_time_ranking(
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
) -> float:
    """
    Spearman correlation between true and predicted iteration times across configs.

    Same order of configs is assumed for both vectors.
    """
    true = _as_time_vector(iteration_time_true_ms, "iteration_time_true_ms")
    predicted = _as_time_vector(iteration_time_pred_ms, "iteration_time_pred_ms")
    return spearman_correlation(true, predicted)


def top_k_hit_rate(
    configs: Iterable[Any],
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
    k: int,
) -> bool:
    """
    Whether the true optimal configuration (lowest true iteration time) appears
    in the top-k configurations by predicted time (lowest predicted first).

    Any tied measured optimum counts as a hit. Predicted ties retain input order.
    """
    k = _validate_k(k)
    _, y_true, y_pred = _aligned_times(configs, iteration_time_true_ms, iteration_time_pred_ms)
    order = np.argsort(y_pred, kind="mergesort")
    topk = order[: min(k, len(order))]
    return bool(np.any(y_true[topk] == np.min(y_true)))


def top_k_regret_ms(
    configs: Iterable[Any],
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
    k: int,
) -> float:
    """
    Top-k regret: among the k configs with best (lowest) predicted times, take the one
    with minimum *true* time; regret = that true time minus global minimum true time (≥ 0).
    """
    k = _validate_k(k)
    _, y_true, y_pred = _aligned_times(configs, iteration_time_true_ms, iteration_time_pred_ms)
    true_min = float(np.min(y_true))
    order = np.argsort(y_pred, kind="mergesort")
    cand = order[: min(k, len(order))]
    best_true_in_topk = float(np.min(y_true[cand]))
    return max(0.0, best_true_in_topk - true_min)


def top_k_relative_regret(
    configs: Iterable[Any],
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
    k: int,
    epsilon_ms: float = 1e-9,
) -> float:
    """Regret divided by best true iteration time: (top_k_regret_ms) / true_min."""
    if not np.isfinite(epsilon_ms) or epsilon_ms <= 0:
        raise ValueError("epsilon_ms must be finite and positive")
    regret = top_k_regret_ms(configs, iteration_time_true_ms, iteration_time_pred_ms, k)
    y_true = np.asarray(iteration_time_true_ms, dtype=float)
    true_min = float(np.min(y_true))
    denom = max(true_min, epsilon_ms)
    return regret / denom


def strategy_ranking_summary(
    configs: Iterable[Any],
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """
    One-call summary: Spearman rho, per-k hit (bool), per-k regret_ms, per-k relative regret.

    Inputs must be aligned, finite, nonnegative time vectors with unique config
    names. Relative regret is a fraction; multiply by 100 for a percentage.
    """
    configs, iteration_time_true_ms, iteration_time_pred_ms = _aligned_times(
        configs, iteration_time_true_ms, iteration_time_pred_ms
    )
    ks_list = sorted({_validate_k(k) for k in ks})
    rho = spearman_iteration_time_ranking(iteration_time_true_ms, iteration_time_pred_ms)
    out: dict[str, Any] = {"spearman_rho": rho}
    for k in ks_list:
        out[f"top{k}_hit"] = top_k_hit_rate(configs, iteration_time_true_ms, iteration_time_pred_ms, k)
        out[f"top{k}_regret_ms"] = top_k_regret_ms(
            configs, iteration_time_true_ms, iteration_time_pred_ms, k
        )
        out[f"top{k}_relative_regret"] = top_k_relative_regret(
            configs, iteration_time_true_ms, iteration_time_pred_ms, k
        )
    return out


def macro_averaged_top_k_hit_rate(
    per_model_hits: Iterable[bool],
) -> float:
    """Mean of per-model boolean hit indicators."""
    h = np.asarray(list(per_model_hits), dtype=float)
    if h.size == 0:
        return float("nan")
    return float(np.mean(h))


def evaluations_from_config_dicts(
    true_by_config: Mapping[str, float],
    pred_by_config: Mapping[str, float],
    ks: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Convenience: build aligned arrays from config→value maps (intersection of keys)."""
    keys = sorted(set(true_by_config) & set(pred_by_config))
    if not keys:
        raise ValueError("No overlapping config keys between true and pred maps")
    y_true = np.asarray([float(true_by_config[c]) for c in keys], dtype=float)
    y_pred = np.asarray([float(pred_by_config[c]) for c in keys], dtype=float)
    return strategy_ranking_summary(keys, y_true, y_pred, ks=ks)
