#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Paths (relative to this script)
#
# Reviewers should place their profiler exports under case_2/data/<model>/
# <parallel_config>/<rank>/. The expected file is `matrix_operator_latency.csv`
# with columns `type, duration_us, m, n, k`.
# ===============================
_CASE2_DIR = Path(__file__).resolve().parent

MODEL_NAME = "qwen25_32b_32k"
CONFIG_NAME = "dp1-tp8-pp4-cp1-RC"
RANK_NAME = "rank_0"

INPUT_DIR = _CASE2_DIR / "data" / MODEL_NAME / CONFIG_NAME / RANK_NAME
OUTPUT_DIR = _CASE2_DIR / "outputs"

CSV_PATH = INPUT_DIR / "matrix_operator_latency.csv"

OUTPUT_PNG_PATH = OUTPUT_DIR / "figure_x10_matrix_operator_mnk_latency_qwen25_32b_32k_rank0.png"
OUTPUT_PDF_PATH = OUTPUT_DIR / "figure_x10_matrix_operator_mnk_latency_qwen25_32b_32k_rank0.pdf"

TOP_K = 10


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
        top=True,
        right=True,
    )

    ax.grid(False)


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = ["type", "duration_us", "m", "n", "k"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].copy()

    for col in ["duration_us", "m", "n", "k"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["duration_us", "m", "n", "k"])
    df = df[df["duration_us"] > 0]

    df["m"] = df["m"].astype(int)
    df["n"] = df["n"].astype(int)
    df["k"] = df["k"].astype(int)

    df["latency_ms"] = df["duration_us"] / 1000.0

    grouped = (
        df.groupby(["type", "m", "n", "k"], as_index=False)
        .agg(
            cumulative_latency_ms=("latency_ms", "sum"),
            call_count=("latency_ms", "size"),
            avg_latency_ms=("latency_ms", "mean"),
        )
        .sort_values("cumulative_latency_ms", ascending=False)
        .head(TOP_K)
        .copy()
    )

    grouped = grouped.sort_values("cumulative_latency_ms", ascending=True).reset_index(drop=True)

    # 检查 m 是否唯一
    unique_m = sorted(grouped["m"].unique().tolist())
    if len(unique_m) == 1:
        fixed_m = unique_m[0]
        grouped["shape_label"] = (
            "n=" + grouped["n"].astype(str)
            + ", k=" + grouped["k"].astype(str)
        )
        y_label = f"Matrix shape (m={fixed_m})"
    else:
        fixed_m = None
        grouped["shape_label"] = (
            "m=" + grouped["m"].astype(str)
            + ", n=" + grouped["n"].astype(str)
            + ", k=" + grouped["k"].astype(str)
        )
        y_label = "Matrix shape"

    y = range(len(grouped))
    x = grouped["cumulative_latency_ms"].to_numpy()
    labels = grouped["shape_label"].tolist()

    colors = {
        "MatMulV3": "#4C78A8",
        "MatMulV2": "#F58518",
    }
    bar_colors = [colors.get(t, "#72B7B2") for t in grouped["type"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    bars = ax.barh(
        y,
        x,
        height=0.58,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.65,
        zorder=3,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)

    ax.set_xlabel("Cumulative latency (ms)")
    ax.set_ylabel(y_label)

    # 右侧预留更多空间，避免最长标签超边框
    x_max = x.max()
    ax.set_xlim(0, x_max * 1.52)

    for bar, latency, calls, avg in zip(
        bars,
        grouped["cumulative_latency_ms"],
        grouped["call_count"],
        grouped["avg_latency_ms"],
    ):
        ax.text(
            bar.get_width() + x_max * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{latency:.0f} ms / {int(calls)} calls / {avg:.2f} ms avg.",
            va="center",
            ha="left",
            fontsize=8.4,
        )

    style_axes(ax)

    # 配置信息框移到右上角
    ax.text(
        0.95,
        0.7,
        f"{MODEL_NAME}\n{CONFIG_NAME}, {RANK_NAME}",
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

    # operator type legend 放到右下角，避免与信息框重叠
    handles = []
    for op_type, color in colors.items():
        if op_type in set(grouped["type"]):
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="None",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=8,
                    label=op_type,
                )
            )

    if handles:
        ax.legend(
            handles=handles,
            title="Operator type",
            loc="lower right",
            bbox_to_anchor=(0.95, 0.05),
            frameon=True,
            fancybox=False,
            edgecolor="black",
            facecolor="white",
            fontsize=8,
            title_fontsize=8,
            borderaxespad=0.4,
        )

    fig.tight_layout()
    # 保存高清 PNG
    fig.savefig(
        OUTPUT_PNG_PATH,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    # 额外保存 PDF（矢量图）
    fig.savefig(
        OUTPUT_PDF_PATH,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    print(f"[OK] Saved PNG to: {OUTPUT_PNG_PATH}")
    print(f"[OK] Saved PDF to: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()