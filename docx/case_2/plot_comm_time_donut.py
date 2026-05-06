#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ===============================
# Paths (relative to this script)
#
# Reviewers should place their profiler exports under case_2/data/<model>/
# <parallel_config>/<rank>/. The expected file is `comm_time.csv` with a
# communication-type column (e.g. `communication_type`) and a time column
# (e.g. `communication_time_ms` or `*_us`).
# ===============================
_CASE2_DIR = Path(__file__).resolve().parent

MODEL_NAME = "qwen25_32b_32k"
CONFIG_NAME = "dp1-tp8-pp4-cp1-RC"
RANK_NAME = "rank_0"

INPUT_DIR = _CASE2_DIR / "data" / MODEL_NAME / CONFIG_NAME / RANK_NAME
OUTPUT_DIR = _CASE2_DIR / "outputs"

CSV_PATH = INPUT_DIR / "comm_time.csv"

OUTPUT_PNG_PATH = OUTPUT_DIR / "figure_x9_comm_time_bar_inset_qwen25_32b_32k_rank0.png"
OUTPUT_PDF_PATH = OUTPUT_DIR / "figure_x9_comm_time_bar_inset_qwen25_32b_32k_rank0.pdf"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    "font.size": 10,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,

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


def normalize_col(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def find_column(columns, candidates, fuzzy_keywords=None):
    norm_map = {normalize_col(c): c for c in columns}

    for cand in candidates:
        cand_norm = normalize_col(cand)
        if cand_norm in norm_map:
            return norm_map[cand_norm]

    if fuzzy_keywords:
        for col in columns:
            col_norm = normalize_col(col)
            if all(k in col_norm for k in fuzzy_keywords):
                return col

    return None


def clean_comm_name(name: str) -> str:
    name = str(name).strip()
    norm = name.lower().replace("-", "_").replace(" ", "_")
    compact = norm.replace("_", "")

    mapping = {
        "batchsendrecv": "batchSendRecv",
        "batch_send_recv": "batchSendRecv",
        "allgather": "allGather",
        "all_gather": "allGather",
        "reducescatter": "reduceScatter",
        "reduce_scatter": "reduceScatter",
        "allreduce": "allReduce",
        "all_reduce": "allReduce",
        "broadcast": "broadcast",
    }

    if norm in mapping:
        return mapping[norm]
    if compact in mapping:
        return mapping[compact]
    return name


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

    df = pd.read_csv(CSV_PATH, sep=None, engine="python")
    if df.empty:
        raise ValueError(f"Empty CSV file: {CSV_PATH}")

    type_col = find_column(
        df.columns,
        candidates=[
            "communication_type",
            "comm_type",
            "collective_type",
            "communication_name",
            "name",
            "type",
            "op_type",
        ],
        fuzzy_keywords=["type"],
    )

    time_col = find_column(
        df.columns,
        candidates=[
            "communication_time_ms",
            "total_time_ms",
            "duration_ms",
            "elapsed_time_ms",
            "time_ms",
            "communication_time_us",
            "total_time_us",
            "duration_us",
            "elapsed_time_us",
            "time_us",
        ],
        fuzzy_keywords=["time"],
    )

    if type_col is None:
        raise ValueError(f"Cannot find communication type column. Available columns: {list(df.columns)}")

    if time_col is None:
        raise ValueError(f"Cannot find communication time column. Available columns: {list(df.columns)}")

    data = df[[type_col, time_col]].copy()
    data[type_col] = data[type_col].astype(str).str.strip()
    data[time_col] = pd.to_numeric(data[time_col], errors="coerce")
    data = data.dropna(subset=[type_col, time_col])
    data = data[data[time_col] > 0]

    if data.empty:
        raise ValueError("No positive communication time values found.")

    time_col_norm = normalize_col(time_col)
    if time_col_norm.endswith("_us") or "us" in time_col_norm:
        data["time_ms"] = data[time_col] / 1000.0
    else:
        data["time_ms"] = data[time_col]

    grouped = (
        data.groupby(type_col, as_index=False)["time_ms"]
        .sum()
        .sort_values("time_ms", ascending=True)
        .reset_index(drop=True)
    )

    grouped["comm_type"] = grouped[type_col].apply(clean_comm_name)

    total_time = grouped["time_ms"].sum()
    grouped["share"] = grouped["time_ms"] / total_time * 100.0

    # 主图：全部通信类型
    labels = grouped["comm_type"].tolist()
    time_ms = grouped["time_ms"].to_numpy()
    share = grouped["share"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    colors = {
        "batchSendRecv": "#4C78A8",
        "allGather": "#F58518",
        "reduceScatter": "#54A24B",
        "allReduce": "#E45756",
        "broadcast": "#B279A2",
    }

    bar_colors = [colors.get(name, "#72B7B2") for name in labels]

    bars = ax.barh(
        np.arange(len(grouped)),
        time_ms,
        height=0.56,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.65,
        zorder=3,
    )

    ax.set_yticks(np.arange(len(grouped)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cumulative communication time (ms)")
    ax.set_ylabel("Communication type")

    x_max = time_ms.max()
    ax.set_xlim(0, x_max * 1.28)

    # 每个条形右侧标注百分比和时间
    for bar, pct, t in zip(bars, share, time_ms):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        if pct >= 1:
            label = f"{pct:.1f}%  ({t:.0f} ms)"
        else:
            label = f"{pct:.2f}%  ({t:.0f} ms)"

        ax.text(
            x + x_max * 0.018,
            y,
            label,
            va="center",
            ha="left",
            fontsize=9,
        )

    style_axes(ax)


    ax.text(
        0.98,
        0.05,
        f"{MODEL_NAME}\n{CONFIG_NAME}, {RANK_NAME}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            boxstyle="round,pad=0.25",
        ),
    )

    # inset：专门放大小类别
    small_df = grouped[grouped["share"] < 1.0].copy()

    if not small_df.empty:
        axins = inset_axes(
            ax,
            width="34%",
            height="38%",
            loc="center right",
            borderpad=2.0,
        )

        small_labels = small_df["comm_type"].tolist()
        small_time = small_df["time_ms"].to_numpy()
        small_share = small_df["share"].to_numpy()

        axins.barh(
            np.arange(len(small_df)),
            small_time,
            height=0.52,
            color=[colors.get(name, "#72B7B2") for name in small_labels],
            edgecolor="black",
            linewidth=0.55,
        )

        axins.set_yticks(np.arange(len(small_df)))
        axins.set_yticklabels(small_labels, fontsize=7.5)
        axins.set_xlabel("ms", fontsize=7.5)
        axins.tick_params(axis="x", labelsize=7.5)

        max_small = small_time.max()
        axins.set_xlim(0, max_small * 1.45)

        for i, (t, pct) in enumerate(zip(small_time, small_share)):
            axins.text(
                t + max_small * 0.04,
                i,
                f"{pct:.2f}%",
                va="center",
                ha="left",
                fontsize=7.3,
            )


        for spine in axins.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_color("black")

        axins.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            width=0.8,
            length=3,
        )

        axins.grid(False)

    
    fig.tight_layout()

    # 保存 PNG
    fig.savefig(
        OUTPUT_PNG_PATH,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    # 保存 PDF（矢量图）
    fig.savefig(
        OUTPUT_PDF_PATH,
        bbox_inches="tight",
        pad_inches=0.06,
    )

    plt.close(fig)

    print(f"[OK] Saved PNG to: {OUTPUT_PNG_PATH}")
    print(f"[OK] Saved PDF to: {OUTPUT_PDF_PATH}")
    print(grouped[["comm_type", "time_ms", "share"]].to_string(index=False))


if __name__ == "__main__":
    main()