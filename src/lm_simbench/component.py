# -*- coding: utf-8 -*-
"""Component-level latency evaluation (operators, matrix ops, communication)."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def component_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Standard component-level metrics: MAE, sMAPE, R² on aligned targets (any consistent unit).

    Use with e.g. total_time_us, duration_us, elapsed_time_ms depending on the task.
    """
    return regression_metrics(y_true, y_pred)


def merge_pred_true_arrays(
    df_true: pd.DataFrame,
    df_pred: pd.DataFrame,
    on: Sequence[str],
    y_col_true: str,
    y_col_pred: str,
    how: str = "inner",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Join on unique, non-null keys and return aligned targets as float arrays.

    Inner joins evaluate the key intersection. Outer/left/right joins retain
    missing targets as NaN. Aggregate duplicate keys before calling this helper.
    """
    on = list(on)
    if not on or len(set(on)) != len(on):
        raise ValueError("on must contain unique join columns")
    if y_col_true in on or y_col_pred in on:
        raise ValueError("Target columns must not be join keys")
    missing_t = (set(on) | {y_col_true}) - set(df_true.columns)
    missing_p = (set(on) | {y_col_pred}) - set(df_pred.columns)
    if missing_t:
        raise ValueError(f"df_true missing columns: {sorted(missing_t)}")
    if missing_p:
        raise ValueError(f"df_pred missing columns: {sorted(missing_p)}")
    t = df_true[list(on) + [y_col_true]].copy()
    p = df_pred[list(on) + [y_col_pred]].copy()
    t[y_col_true] = pd.to_numeric(t[y_col_true], errors="coerce")
    p[y_col_pred] = pd.to_numeric(p[y_col_pred], errors="coerce")
    if t[on].isna().any().any() or p[on].isna().any().any():
        raise ValueError("Join keys must not contain null values")
    true_col, pred_col = "__true_target", "__pred_target"
    while true_col in on:
        true_col += "_"
    while pred_col in on:
        pred_col += "_"
    t = t.rename(columns={y_col_true: true_col})
    p = p.rename(columns={y_col_pred: pred_col})
    merged = t.merge(p, on=on, how=how, validate="one_to_one")
    y_true = merged[true_col].to_numpy(dtype=float)
    y_pred = merged[pred_col].to_numpy(dtype=float)
    return y_true, y_pred


def message_size_bucket(
    size_mb: Any,
    *,
    edges_mb: Iterable[float] = (0.0, 0.125, 1.0, 8.0, 64.0, float("inf")),
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """
    Bucket message / transit sizes (MB) for grouped communication metrics.

    ``edges_mb`` must be non-decreasing; consecutive pairs define half-open intervals
    except the last interval which includes the final edge (e.g. ...64, inf).
    """
    edges = [float(x) for x in edges_mb]
    if len(edges) < 2:
        raise ValueError("edges_mb must contain at least two values")
    inner = np.asarray(edges[1:-1], dtype=float)
    if labels is None:
        labels = [f"[{edges[i]},{edges[i + 1]})" for i in range(len(edges) - 1)]
    if len(labels) != len(edges) - 1:
        raise ValueError("labels must have length len(edges_mb) - 1")
    s = np.asarray(size_mb, dtype=float)
    out = np.full(s.shape, "nan", dtype=object)
    valid = np.isfinite(s)
    if inner.size == 0:
        out[valid] = labels[0]
        return out
    idx = np.zeros(s.shape, dtype=int)
    idx[valid] = np.digitize(s[valid], inner, right=False)
    idx = np.clip(idx, 0, len(labels) - 1)
    for i, lab in enumerate(labels):
        out[(idx == i) & valid] = lab
    return out


def regression_metrics_by_group(
    df: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    y_col_true: str,
    y_col_pred: str,
) -> pd.DataFrame:
    """Compute metrics per group; n counts finite pairs and n_total counts rows."""
    missing = (set(group_cols) | {y_col_true, y_col_pred}) - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for keys, g in df.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = tuple([keys])
        y_true = pd.to_numeric(g[y_col_true], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(g[y_col_pred], errors="coerce").to_numpy(dtype=float)
        m = regression_metrics(y_true, y_pred)
        row = {col: k for col, k in zip(group_cols, keys)}
        row.update(m)
        row["n"] = int(np.count_nonzero(np.isfinite(y_true) & np.isfinite(y_pred)))
        row["n_total"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def comm_time_augmented_frame(
    comm_df: pd.DataFrame,
    *,
    size_col: str = "link_transit_size_mb",
    edges_mb: Iterable[float] = (0.0, 0.125, 1.0, 8.0, 64.0, float("inf")),
) -> pd.DataFrame:
    """Return a copy of comm_df with ``message_size_bucket`` column for grouping."""
    out = comm_df.copy()
    if size_col not in out.columns:
        raise ValueError(f"comm_df missing size column: {size_col}")
    out["message_size_bucket"] = message_size_bucket(
        pd.to_numeric(out[size_col], errors="coerce"), edges_mb=edges_mb
    )
    return out
