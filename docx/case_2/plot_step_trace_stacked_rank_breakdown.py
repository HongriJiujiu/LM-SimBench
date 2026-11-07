#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Paths (relative to this script)
#
# Reviewers should place their profiler export under case_2/data/<model>/.
# The expected file is `step_trace_time_summary.csv` with columns
# `optimization_config, rank_id, computing, communication_not_overlapped,
# overlapped, free, stage` (column names are normalised to snake_case below).
# ===============================
_CASE2_DIR = Path(__file__).resolve().parent

MODEL_NAME = "qwen25_32b_32k"

CSV_PATH = _CASE2_DIR / "data" / MODEL_NAME / "step_trace_time_summary.csv"
OUTPUT_DIR = _CASE2_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def to_snake_case(name: str) -> str:
    name = str(name).strip()
    name = name.replace("(", "_").replace(")", "")
    name = name.replace("-", "_").replace("/", "_")
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def safe_filename(name: str) -> str:
    name = str(name)
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", name)
    return name.strip("_")


def parse_rank_id(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else np.nan


def parse_config(config: str):
    def get(pattern, default=-1):
        m = re.search(pattern, config)
        return int(m.group(1)) if m else default

    rc_order = 0 if config.endswith("noRC") else 1

    return (
        get(r"dp(\d+)"),
        get(r"tp(\d+)"),
        get(r"pp(\d+)"),
        get(r"cp(\d+)"),
        rc_order,
        config,
    )


if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {CSV_PATH}\n"
        "Place a profiler `step_trace_time_summary.csv` (columns: "
        "optimization_config, rank_id, computing, communication_not_overlapped, "
        "overlapped, free, stage) under this path."
    )

df = pd.read_csv(CSV_PATH)
df.columns = [to_snake_case(c) for c in df.columns]

required_cols = [
    "optimization_config",
    "rank_id",
    "computing",
    "communication_not_overlapped",
    "overlapped",
    "free",
    "stage",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["rank_id"] = df["rank_id"].apply(parse_rank_id)
df = df.dropna(subset=["rank_id"])
df["rank_id"] = df["rank_id"].astype(int)

time_cols = [
    "computing",
    "communication_not_overlapped",
    "overlapped",
    "free",
    "stage",
]

for c in time_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# us -> s
for c in time_cols:
    df[c] = df[c] / 1_000_000.0

df["comm_ratio"] = np.where(
    df["stage"] > 0,
    df["communication_not_overlapped"] / df["stage"] * 100.0,
    np.nan,
)


def style_axes(ax, use_grid: bool = False):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    if use_grid:
        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.55,
            color="#D6D6D6",
            alpha=0.65,
        )
    else:
        ax.grid(False)

    ax.set_axisbelow(True)


def plot_one_config(config_df: pd.DataFrame, config_name: str):
    config_df = config_df.sort_values("rank_id").copy()

    x = np.arange(len(config_df))
    rank_labels = config_df["rank_id"].astype(str).tolist()

    compute = config_df["computing"].to_numpy()
    comm = config_df["communication_not_overlapped"].to_numpy()
    free = config_df["free"].to_numpy()
    stage = config_df["stage"].to_numpy()
    overlap = config_df["overlapped"].to_numpy()
    comm_ratio = config_df["comm_ratio"].to_numpy()

    fig_width = max(10.5, min(18.0, 0.26 * len(config_df)))
    fig_height = 5.9

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={
            "height_ratios": [3.2, 1.0],
            "hspace": 0.08
        }
    )


    # Component colors derived from Tableau tab20.
    # ===============================
    COLOR_COMPUTE = "#1F77B4"   # blue
    COLOR_COMM = "#FF7F0E"      # orange
    COLOR_FREE = "#AEC7E8"      # light blue
    COLOR_STAGE = "#2CA02C"     # green
    COLOR_OVERLAP = "#D62728"   # red
    COLOR_RATIO = "#9467BD"     # purple

    bar_width = 0.82

    ax_top.bar(
        x, compute,
        width=bar_width,
        color=COLOR_COMPUTE,
        edgecolor="black",
        linewidth=0.22,
        label="Computing",
        zorder=3
    )

    ax_top.bar(
        x, comm,
        bottom=compute,
        width=bar_width,
        color=COLOR_COMM,
        edgecolor="black",
        linewidth=0.22,
        label="Non-overlapped comm.",
        zorder=3
    )

    ax_top.bar(
        x, free,
        bottom=compute + comm,
        width=bar_width,
        color=COLOR_FREE,
        edgecolor="black",
        linewidth=0.22,
        label="Free",
        zorder=3
    )

    ax_top.plot(
        x, stage,
        color=COLOR_STAGE,
        marker="o",
        markersize=3.7,
        markerfacecolor="white",
        markeredgecolor=COLOR_STAGE,
        markeredgewidth=1.1,
        linewidth=1.7,
        label="Stage",
        zorder=5
    )

    ax_top.plot(
        x, overlap,
        color=COLOR_OVERLAP,
        marker="s",
        markersize=3.1,
        markerfacecolor="white",
        markeredgecolor=COLOR_OVERLAP,
        markeredgewidth=1.0,
        linewidth=1.35,
        linestyle="--",
        label="Overlapped",
        zorder=5
    )

    y_max = max(
        np.nanmax(compute + comm + free),
        np.nanmax(stage),
        np.nanmax(overlap),
    )

    ax_top.set_ylim(0, y_max * 1.18)
    ax_top.set_ylabel("Time (s)")
    ax_top.set_title(config_name, fontsize=14, pad=10)

    # 上半图不使用网格
    style_axes(ax_top, use_grid=False)

    handles, labels = ax_top.get_legend_handles_labels()
    order = [3, 4, 0, 1, 2]

    ax_top.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        facecolor="white",
        handlelength=1.8,
        columnspacing=1.1,
        borderpad=0.45,
    )

    # ===============================
    # Bottom: communication ratio
    # ===============================
    ax_bottom.plot(
        x, comm_ratio,
        color=COLOR_RATIO,
        marker="o",
        markersize=3.1,
        markerfacecolor="white",
        markeredgecolor=COLOR_RATIO,
        markeredgewidth=1.0,
        linewidth=1.5,
        zorder=4
    )

    ax_bottom.fill_between(
        x,
        comm_ratio,
        0,
        color=COLOR_RATIO,
        alpha=0.07,
        zorder=2
    )

    mean_ratio = np.nanmean(comm_ratio)

    ax_bottom.axhline(
        mean_ratio,
        color="#404040",
        linestyle=":",
        linewidth=1.0,
        zorder=3
    )

    ax_bottom.text(
        0.985,
        0.87,
        f"Mean comm. ratio = {mean_ratio:.1f}%",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.35",
            linewidth=0.8
        )
    )

    ratio_max = np.nanmax(comm_ratio)
    ax_bottom.set_ylim(0, max(5, ratio_max * 1.28))

    ax_bottom.set_ylabel("Comm. ratio (%)")
    ax_bottom.set_xlabel("Rank ID")

    # 下半图保留极淡横向参考线
    style_axes(ax_bottom, use_grid=True)

    tick_interval = max(1, len(config_df) // 16)
    ax_bottom.set_xticks(x[::tick_interval])
    ax_bottom.set_xticklabels(rank_labels[::tick_interval])

    ax_top.tick_params(labelbottom=False)

    fig.subplots_adjust(
        top=0.86,
        bottom=0.11,
        left=0.075,
        right=0.985
    )

    out_name = safe_filename(config_name)

    pdf_path = OUTPUT_DIR / f"{out_name}_rank_step_breakdown_journal.pdf"
    png_path = OUTPUT_DIR / f"{out_name}_rank_step_breakdown_journal.png"

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)


configs = sorted(df["optimization_config"].dropna().unique(), key=parse_config)

print(f"[INFO] Found {len(configs)} optimization configs.")

for config in configs:
    sub = df[df["optimization_config"] == config].copy()
    plot_one_config(sub, config)
    print(f"[DONE] {config}")

print("[INFO] All figures saved.")