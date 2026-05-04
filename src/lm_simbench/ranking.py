# -*- coding: utf-8 -*-
"""Parallel strategy ranking evaluation for a single model (one set of candidate configs)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .metrics import spearman_correlation


def _as_config_order(configs: Iterable[Any]) -> list[str]:
    return [str(c) for c in configs]


def true_best_config(configs: Iterable[Any], iteration_time_ms: Any) -> str:
    """Config name with minimum iteration time."""
    cs = _as_config_order(configs)
    t = np.asarray(iteration_time_ms, dtype=float)
    if len(cs) != t.size:
        raise ValueError("configs and iteration_time_ms must have the same length")
    i = int(np.nanargmin(np.where(np.isfinite(t), t, np.inf)))
    return cs[i]


def spearman_iteration_time_ranking(
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
) -> float:
    """
    Spearman correlation between true and predicted iteration times across configs.

    Same order of configs is assumed for both vectors.
    """
    return spearman_correlation(iteration_time_true_ms, iteration_time_pred_ms)


def top_k_hit_rate(
    configs: Iterable[Any],
    iteration_time_true_ms: Any,
    iteration_time_pred_ms: Any,
    k: int,
) -> bool:
    """
    Whether the true optimal configuration (lowest true iteration time) appears
    in the top-k configurations by predicted time (lowest predicted first).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    cs = _as_config_order(configs)
    y_true = np.asarray(iteration_time_true_ms, dtype=float)
    y_pred = np.asarray(iteration_time_pred_ms, dtype=float)
    if len(cs) != y_true.size or y_true.shape != y_pred.shape:
        raise ValueError("configs, iteration_time_true_ms, iteration_time_pred_ms must align")
    best_idx = int(np.nanargmin(np.where(np.isfinite(y_true), y_true, np.inf)))
    order = np.argsort(np.where(np.isfinite(y_pred), y_pred, np.inf), kind="mergesort")
    topk = set(order[: min(k, len(order))].tolist())
    return best_idx in topk


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
    if k < 1:
        raise ValueError("k must be >= 1")
    cs = _as_config_order(configs)
    y_true = np.asarray(iteration_time_true_ms, dtype=float)
    y_pred = np.asarray(iteration_time_pred_ms, dtype=float)
    if len(cs) != y_true.size or y_true.shape != y_pred.shape:
        raise ValueError("configs, iteration_time_true_ms, iteration_time_pred_ms must align")
    true_min = float(np.nanmin(y_true))
    order = np.argsort(np.where(np.isfinite(y_pred), y_pred, np.inf), kind="mergesort")
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
    regret = top_k_regret_ms(configs, iteration_time_true_ms, iteration_time_pred_ms, k)
    y_true = np.asarray(iteration_time_true_ms, dtype=float)
    true_min = float(np.nanmin(y_true))
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
    """
    ks_list = sorted({int(k) for k in ks if int(k) >= 1})
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
