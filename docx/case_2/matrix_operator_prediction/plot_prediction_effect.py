#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
# Raw per-call inputs: place `<parallel_config>_predicted.csv` here (columns at
# minimum: type, duration_us, pred_total_us, m, n, k; optional rank_id, rank_name).
INPUT_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_NAME = "qwen25_32b_32k"
TOP_K = 10
ACTUAL_COLOR = "#4C78A8"
PRED_COLOR = "#F58518"
RMSE_COLOR = "#D62728"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    "font.size": 10,
    "axes.labelsize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
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


def style_axes(ax):
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


def sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")


def get_config_name(csv_path):
    suffix = "_predicted"
    stem = csv_path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem


def load_prediction_csv(csv_path):
    required_cols = [
        "type",
        "duration_us",
        "pred_total_us",
        "m",
        "n",
        "k",
    ]
    optional_cols = ["rank_id", "rank_name"]

    header = pd.read_csv(csv_path, nrows=0)
    usecols = [col for col in required_cols + optional_cols if col in header.columns]

    missing = [col for col in required_cols if col not in usecols]
    if missing:
        raise ValueError(f"{csv_path.name} missing required columns: {missing}")

    df = pd.read_csv(csv_path, usecols=usecols)

    for col in ["duration_us", "pred_total_us", "m", "n", "k"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["duration_us", "pred_total_us", "m", "n", "k"])
    df = df[(df["duration_us"] > 0) & (df["pred_total_us"] > 0)].copy()
    if df.empty:
        raise ValueError(f"{csv_path.name} has no valid prediction rows")

    for col in ["m", "n", "k"]:
        df[col] = df[col].astype(int)

    if "rank_id" in df.columns:
        df["rank_id"] = pd.to_numeric(df["rank_id"], errors="coerce").astype("Int64")

    return df


def build_grouped_metrics(df):
    group_cols = ["type", "m", "n", "k"]
    rank_cols = []
    if "rank_id" in df.columns:
        rank_cols.append("rank_id")
    if "rank_name" in df.columns:
        rank_cols.append("rank_name")

    df = df.copy()
    df["actual_ms"] = df["duration_us"] / 1000.0
    df["pred_ms"] = df["pred_total_us"] / 1000.0
    df["squared_error_ms2"] = (df["pred_ms"] - df["actual_ms"]) ** 2
    df["abs_error_ms"] = (df["pred_ms"] - df["actual_ms"]).abs()
    df["abs_percent_error"] = df["abs_error_ms"] / df["actual_ms"] * 100.0

    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            actual_total_ms=("actual_ms", "sum"),
            pred_total_ms=("pred_ms", "sum"),
            call_count=("actual_ms", "size"),
            actual_avg_ms=("actual_ms", "mean"),
            pred_avg_ms=("pred_ms", "mean"),
            mse_ms2=("squared_error_ms2", "mean"),
            mae_ms=("abs_error_ms", "mean"),
            mape_percent=("abs_percent_error", "mean"),
        )
        .copy()
    )
    grouped["rmse_ms"] = grouped["mse_ms2"] ** 0.5

    if rank_cols:
        per_rank = (
            df.groupby(rank_cols + group_cols, as_index=False)
            .agg(
                actual_rank_cumulative_ms=("actual_ms", "sum"),
                pred_rank_cumulative_ms=("pred_ms", "sum"),
            )
        )
        rank_stats = (
            per_rank.groupby(group_cols, as_index=False)
            .agg(
                actual_errorbar_ms=("actual_rank_cumulative_ms", "std"),
                pred_errorbar_ms=("pred_rank_cumulative_ms", "std"),
                rank_count=("actual_rank_cumulative_ms", "size"),
            )
        )
        grouped = grouped.merge(rank_stats, on=group_cols, how="left")
    else:
        grouped["actual_errorbar_ms"] = 0.0
        grouped["pred_errorbar_ms"] = 0.0
        grouped["rank_count"] = 1

    grouped["actual_cumulative_ms"] = grouped["actual_total_ms"]
    grouped["pred_cumulative_ms"] = grouped["pred_total_ms"]

    grouped[["actual_errorbar_ms", "pred_errorbar_ms"]] = grouped[
        ["actual_errorbar_ms", "pred_errorbar_ms"]
    ].fillna(0.0)
    return grouped


def add_shape_labels(grouped):
    unique_m = sorted(grouped["m"].unique().tolist())
    if len(unique_m) == 1:
        fixed_m = unique_m[0]
        grouped["shape_label"] = (
            "n=" + grouped["n"].astype(str)
            + ", k=" + grouped["k"].astype(str)
            + " (" + grouped["type"].astype(str) + ")"
        )
        y_label = f"Matrix shape (m={fixed_m})"
    else:
        grouped["shape_label"] = (
            "m=" + grouped["m"].astype(str)
            + ", n=" + grouped["n"].astype(str)
            + ", k=" + grouped["k"].astype(str)
            + " (" + grouped["type"].astype(str) + ")"
        )
        y_label = "Matrix shape"

    return grouped, y_label


def plot_config_metrics(grouped, config_name):
    type_order = {"MatMulV3": 0, "MatMulV2": 1}
    grouped = grouped.copy()
    grouped["type_order"] = grouped["type"].map(type_order).fillna(99)
    grouped = (
        grouped.sort_values(
            ["type_order", "actual_cumulative_ms"],
            ascending=[True, False],
        )
        .head(TOP_K)
        .reset_index(drop=True)
    )
    grouped, y_label = add_shape_labels(grouped)

    y = list(range(len(grouped)))
    bar_height = 0.34
    actual_x = grouped["actual_cumulative_ms"].to_numpy() / 1000.0
    pred_x = grouped["pred_cumulative_ms"].to_numpy() / 1000.0
    rmse_x = grouped["rmse_ms"].to_numpy()
    x_max = max(actual_x.max(), pred_x.max())

    fig, ax = plt.subplots(figsize=(8.8, 5.2))


    actual_bars = ax.barh(
        [pos - bar_height / 2 for pos in y],
        actual_x,
        height=bar_height,
        color=ACTUAL_COLOR,
        edgecolor="black",
        linewidth=0.65,
        label="Actual total cumulative latency",
        zorder=3,
    )
    pred_bars = ax.barh(
        [pos + bar_height / 2 for pos in y],
        pred_x,
        height=bar_height,
        color=PRED_COLOR,
        edgecolor="black",
        linewidth=0.65,
        label="Predicted total cumulative latency",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(grouped["shape_label"].tolist())
    ax.set_xlabel("Total cumulative latency across all ranks (s)")
    ax.set_ylabel(y_label)
    ax.set_xlim(0, x_max * 1.20)
    ax.invert_yaxis()

    for actual_bar, pred_bar, row in zip(
        actual_bars,
        pred_bars,
        grouped.itertuples(index=False),
    ):
        ax.text(
            actual_bar.get_width() + x_max * 0.018,
            actual_bar.get_y() + actual_bar.get_height() / 2,
            f"actual {row.actual_cumulative_ms / 1000.0:.1f} s",
            va="center",
            ha="left",
            fontsize=7.8,
            color=ACTUAL_COLOR,
        )
        ax.text(
            pred_bar.get_width() + x_max * 0.018,
            pred_bar.get_y() + pred_bar.get_height() / 2,
            f"pred {row.pred_cumulative_ms / 1000.0:.1f} s",
            va="center",
            ha="left",
            fontsize=7.8,
            color=PRED_COLOR,
        )

    style_axes(ax)

    ax_rmse = ax.twiny()
    ax_rmse.plot(
        rmse_x,
        y,
        color=RMSE_COLOR,
        marker="D",
        markersize=4.2,
        linewidth=1.2,
        label="Per-call RMSE",
        zorder=4,
    )
    ax_rmse.set_xlabel("Per-call RMSE (ms)", color=RMSE_COLOR)
    ax_rmse.set_xlim(0, max(rmse_x.max() * 1.25, 0.1))
    ax_rmse.tick_params(
        axis="x",
        which="both",
        direction="in",
        colors=RMSE_COLOR,
        top=True,
        bottom=False,
    )
    ax_rmse.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    ax_rmse.spines["top"].set_color(RMSE_COLOR)
    ax_rmse.spines["top"].set_linewidth(1.15)
    for spine_name in ["bottom", "left", "right"]:
        ax_rmse.spines[spine_name].set_visible(False)
    ax_rmse.grid(False)

    ax.text(
        0.96,
        0.72,
        f"{MODEL_NAME}\n{config_name}, all ranks",
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

    handles, labels = ax.get_legend_handles_labels()
    rmse_handles, rmse_labels = ax_rmse.get_legend_handles_labels()
    ax.legend(
        handles + rmse_handles,
        labels + rmse_labels,
        loc="lower right",
        bbox_to_anchor=(0.96, 0.30),
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        fontsize=8,
        borderaxespad=0.4,
    )

    fig.tight_layout()

    file_prefix = sanitize_filename(f"prediction_effect_{config_name}")
    png_path = OUTPUT_DIR / f"{file_prefix}.png"
    pdf_path = OUTPUT_DIR / f"{file_prefix}.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    return png_path, pdf_path


def process_file(csv_path):
    config_name = get_config_name(csv_path)
    df = load_prediction_csv(csv_path)
    grouped = build_grouped_metrics(df)

    summary_prefix = sanitize_filename(f"prediction_metrics_{config_name}")
    summary_path = OUTPUT_DIR / f"{summary_prefix}.csv"
    grouped.sort_values("actual_total_ms", ascending=False).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    png_path, pdf_path = plot_config_metrics(grouped, config_name)
    return summary_path, png_path, pdf_path


def main():
    csv_files = sorted(INPUT_DIR.glob("*_predicted.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_predicted.csv files found in: {INPUT_DIR}\n"
            "Place predictor output CSVs under data/ (same directory as this script)."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        summary_path, png_path, pdf_path = process_file(csv_path)
        print(f"[OK] Saved summary: {summary_path}")
        print(f"[OK] Saved PNG: {png_path}")
        print(f"[OK] Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
