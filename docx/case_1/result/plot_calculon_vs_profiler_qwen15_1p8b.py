#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Calculon predictions (qwen15_1.8b_8k_metrics.csv) against profiler CSV exports:
step_time_summary.csv (iteration_time_ms) and peak_memory.csv (max peak_memory_mb per config).

Parallel configs: primary sort by suffix (-RC before -noRC); within each group,
order follows the Calculon metrics CSV row order.
Outputs PNG/PDF under result/outputs/ (same visual language as
benchmark/docx/matrix_operator_prediction/plot_prediction_effect.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
METRICS_CSV = BASE_DIR / "qwen15_1.8b_8k_metrics.csv"
STEP_TIME_CSV = BASE_DIR / "step_time_summary.csv"
PEAK_MEMORY_CSV = BASE_DIR / "peak_memory.csv"

MODEL_TAG = "qwen15_1.8b_8k"

ACTUAL_COLOR = "#4C78A8"
PRED_COLOR = "#F58518"
ERR_COLOR = "#D62728"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 1.15,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.15)
        spine.set_color("black")
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=True,
    )
    ax.grid(False)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def config_label_strip_cp1(config: str) -> str:
    """Remove the pipeline segment ``-cp1`` from display only (not ``cp10``, ``cp12``, …)."""
    return re.sub(r"-cp1(?=-|$)", "", str(config))


def load_merged_frame() -> pd.DataFrame:
    pred = pd.read_csv(METRICS_CSV)
    need = {"parallel_config", "calculon_iteration_time", "peak_memory"}
    missing_cols = need - set(pred.columns)
    if missing_cols:
        raise ValueError(f"{METRICS_CSV.name} missing columns: {sorted(missing_cols)}")

    step = pd.read_csv(STEP_TIME_CSV)
    peak = pd.read_csv(PEAK_MEMORY_CSV)

    if "optimization_config" not in step.columns or "iteration_time_ms" not in step.columns:
        raise ValueError(f"{STEP_TIME_CSV.name} expects optimization_config, iteration_time_ms")
    if "optimization_config" not in peak.columns or "peak_memory_mb" not in peak.columns:
        raise ValueError(f"{PEAK_MEMORY_CSV.name} expects optimization_config, peak_memory_mb")

    peak_agg = (
        peak.groupby("optimization_config", as_index=False)["peak_memory_mb"]
        .max()
        .rename(columns={"peak_memory_mb": "peak_memory_mb_max"})
    )

    merged = pred.rename(columns={"parallel_config": "config"}).merge(
        step.rename(columns={"optimization_config": "config"}),
        on="config",
        how="left",
    )
    merged = merged.merge(
        peak_agg.rename(columns={"optimization_config": "config"}),
        on="config",
        how="left",
    )

    merged["actual_iteration_s"] = merged["iteration_time_ms"].astype(float) / 1000.0
    merged["pred_iteration_s"] = merged["calculon_iteration_time"].astype(float)
    merged["actual_peak_gb"] = merged["peak_memory_mb_max"].astype(float) / 1024.0
    merged["pred_peak_gb"] = merged["peak_memory"].astype(float)

    row_missing = merged[["iteration_time_ms", "peak_memory_mb_max"]].isna().any(axis=1)
    if row_missing.any():
        bad = merged.loc[row_missing, "config"].tolist()
        raise ValueError(
            "Profiler CSVs missing rows for Calculon configs (check spelling): "
            + ", ".join(bad)
        )

    merged["abs_err_iteration_s"] = (merged["pred_iteration_s"] - merged["actual_iteration_s"]).abs()
    merged["abs_err_peak_gb"] = (merged["pred_peak_gb"] - merged["actual_peak_gb"]).abs()

    def suffix_bucket(cfg: str) -> int:
        """Sort key: RC before noRC."""
        s = str(cfg)
        if s.endswith("-noRC"):
            return 1
        if s.endswith("-RC"):
            return 0
        return 2

    order_map = {c: i for i, c in enumerate(pred["parallel_config"].tolist())}
    merged["_order"] = merged["config"].map(order_map)
    merged["_suffix_bucket"] = merged["config"].map(suffix_bucket)
    merged = merged.sort_values(["_suffix_bucket", "_order"]).reset_index(drop=True)
    return merged.drop(columns=["_order", "_suffix_bucket"])


def plot_paired_horizontal(
    df: pd.DataFrame,
    *,
    actual_col: str,
    pred_col: str,
    error_col: str,
    xlabel_main: str,
    xlabel_error: str,
    value_fmt_meas: str,
    value_fmt_pred: str,
    title_stub: str,
    file_prefix: str,
    meas_label: str,
    pred_label: str,
) -> tuple[Path, Path]:
    n = len(df)
    y = np.arange(n)
    bar_height = 0.34
    actual_x = df[actual_col].to_numpy(dtype=float)
    pred_x = df[pred_col].to_numpy(dtype=float)
    err_x = df[error_col].to_numpy(dtype=float)
    x_max = max(actual_x.max(), pred_x.max(), np.finfo(float).eps)

    fig_h = max(5.0, 0.42 * n + 2.0)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))

    ax.barh(
        y - bar_height / 2,
        actual_x,
        height=bar_height,
        color=ACTUAL_COLOR,
        edgecolor="black",
        linewidth=0.65,
        label=meas_label,
        zorder=3,
    )
    bars_pred = ax.barh(
        y + bar_height / 2,
        pred_x,
        height=bar_height,
        color=PRED_COLOR,
        edgecolor="black",
        linewidth=0.65,
        label=pred_label,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels([config_label_strip_cp1(c) for c in df["config"].tolist()])
    ax.set_xlabel(xlabel_main)
    ax.set_ylabel("Parallel configuration")
    ax.set_xlim(0, x_max * 1.22)
    ax.invert_yaxis()

    meas_bars = ax.containers[0]
    for rect_meas, rect_pred, row in zip(meas_bars, bars_pred, df.itertuples(index=False)):
        ax.text(
            rect_meas.get_width() + x_max * 0.015,
            rect_meas.get_y() + rect_meas.get_height() / 2,
            value_fmt_meas.format(getattr(row, actual_col)),
            va="center",
            ha="left",
            fontsize=7.8,
            color=ACTUAL_COLOR,
        )
        ax.text(
            rect_pred.get_width() + x_max * 0.015,
            rect_pred.get_y() + rect_pred.get_height() / 2,
            value_fmt_pred.format(getattr(row, pred_col)),
            va="center",
            ha="left",
            fontsize=7.8,
            color=PRED_COLOR,
        )

    style_axes(ax)

    ax_err = ax.twiny()
    ax_err.plot(
        err_x,
        y,
        color=ERR_COLOR,
        marker="D",
        markersize=4.2,
        linewidth=1.2,
        label=r"$|$Calculon − measured$|$",
        zorder=4,
    )
    ax_err.set_xlabel(xlabel_error, color=ERR_COLOR)
    emax = float(err_x.max())
    ax_err.set_xlim(0, max(emax * 1.25, 1e-6))
    ax_err.tick_params(
        axis="x",
        which="both",
        direction="in",
        colors=ERR_COLOR,
        top=True,
        bottom=False,
    )
    ax_err.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax_err.spines["top"].set_color(ERR_COLOR)
    ax_err.spines["top"].set_linewidth(1.15)
    for spine_name in ["bottom", "left", "right"]:
        ax_err.spines[spine_name].set_visible(False)
    ax_err.grid(False)

    ax.text(
        0.96,
        0.72,
        f"{MODEL_TAG}\n{title_stub}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            boxstyle="round,pad=0.25",
        ),
    )

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_err.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=8,
        borderaxespad=0.4,
    )

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(file_prefix)
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    for p in (METRICS_CSV, STEP_TIME_CSV, PEAK_MEMORY_CSV):
        if not p.exists():
            raise FileNotFoundError(p)

    df = load_merged_frame()

    summary_path = OUTPUT_DIR / (sanitize_filename(f"calculon_vs_profiler_merge_{MODEL_TAG}") + ".csv")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = df[
        [
            "config",
            "actual_iteration_s",
            "pred_iteration_s",
            "actual_peak_gb",
            "pred_peak_gb",
            "abs_err_iteration_s",
            "abs_err_peak_gb",
        ]
    ].copy()
    df_out.to_csv(summary_path, index=False, encoding="utf-8-sig")

    p1n, p1p = plot_paired_horizontal(
        df,
        actual_col="actual_iteration_s",
        pred_col="pred_iteration_s",
        error_col="abs_err_iteration_s",
        xlabel_main="Iteration time (s)",
        xlabel_error="Absolute error (s)",
        value_fmt_meas="meas {:.2f} s",
        value_fmt_pred="calc {:.2f} s",
        title_stub="Calculon vs profiler iteration time",
        file_prefix=f"calculon_iteration_effect_{MODEL_TAG}",
        meas_label="Measured iteration time",
        pred_label="Calculon iteration time",
    )

    p2n, p2p = plot_paired_horizontal(
        df,
        actual_col="actual_peak_gb",
        pred_col="pred_peak_gb",
        error_col="abs_err_peak_gb",
        xlabel_main="Peak memory (GB)",
        xlabel_error="Absolute error (GB)",
        value_fmt_meas="meas {:.2f} GB",
        value_fmt_pred="calc {:.2f} GB",
        title_stub="Calculon vs profiler peak memory\n(max over ranks)",
        file_prefix=f"calculon_peak_memory_effect_{MODEL_TAG}",
        meas_label="Measured peak memory",
        pred_label="Calculon peak memory",
    )

    print(f"[OK] merged table: {summary_path}")
    print(f"[OK] iteration PNG/PDF: {p1n} , {p1p}")
    print(f"[OK] peak memory PNG/PDF: {p2n} , {p2p}")


if __name__ == "__main__":
    main()
