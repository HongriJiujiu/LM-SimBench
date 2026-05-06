#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Figure X.2: Rank-level peak memory heatmap for qwen15_1.8b_8k (DP=1 configs).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_CASE1_DIR = Path(__file__).resolve().parent

# ===============================
# 1. Paths (relative to this script)
# ===============================
MODEL_NAME = "qwen15_1.8b_8k"

# Profiler export bundled with the repository under case_1/result/.
# Reviewers can replace this CSV with their own profiler output of the same
# schema (`optimization_config`, `rank_id`, `peak_memory_mb`).
CSV_PATH = _CASE1_DIR / "result" / "peak_memory.csv"

OUTPUT_DIR = _CASE1_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# 2. Plot style
# ===============================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 11,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

sns.set_theme(
    style="white",
    font="Times New Roman",
    font_scale=1.0
)


def extract_parallel_value(config: str, key: str) -> int:
    match = re.search(rf"{key}(\d+)", config)
    return int(match.group(1)) if match else 0


def get_rc_type(config: str) -> str:
    if config.endswith("-noRC"):
        return "noRC"
    if config.endswith("-RC"):
        return "RC"
    return "UNKNOWN"


def simplify_config_label(config: str) -> str:
    label = config

    if label.startswith("dp1-"):
        label = label[len("dp1-"):]

    if label.endswith("-noRC"):
        label = label[:-5]
    elif label.endswith("-RC"):
        label = label[:-3]

    return label


def parse_base_config_label(label: str):
    tp = extract_parallel_value(label, "tp")
    pp = extract_parallel_value(label, "pp")
    cp = extract_parallel_value(label, "cp")
    ep = extract_parallel_value(label, "ep")
    return tp, pp, cp, ep, label


def load_peak_memory(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["optimization_config", "rank_id", "peak_memory_mb"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_cols].copy()

    df["rank_id"] = pd.to_numeric(df["rank_id"], errors="coerce")
    df["peak_memory_mb"] = pd.to_numeric(df["peak_memory_mb"], errors="coerce")

    df = df.dropna(subset=["optimization_config", "rank_id", "peak_memory_mb"])
    df["rank_id"] = df["rank_id"].astype(int)

    df["dp"] = df["optimization_config"].apply(lambda x: extract_parallel_value(x, "dp"))
    df["rc_type"] = df["optimization_config"].apply(get_rc_type)
    df["base_config"] = df["optimization_config"].apply(simplify_config_label)

    df = df[(df["dp"] == 1) & (df["rc_type"].isin(["RC", "noRC"]))].copy()

    if df.empty:
        raise ValueError("No data remains after filtering dp=1 and RC/noRC configurations.")

    return df


def build_pivot(df: pd.DataFrame, rc_type: str, base_order, rank_order) -> pd.DataFrame:
    sub_df = df[df["rc_type"] == rc_type].copy()

    pivot = sub_df.pivot_table(
        index="base_config",
        columns="rank_id",
        values="peak_memory_mb",
        aggfunc="max",
    )

    pivot = pivot.reindex(index=base_order, columns=rank_order)
    return pivot


def set_heatmap_border(ax, linewidth: float = 1.15) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(linewidth)
        spine.set_color("black")


def adjust_annotation_colors(ax, threshold: float) -> None:
    """
    深色格子用白字，浅色格子用黑字，提升可读性。
    """
    for text in ax.texts:
        try:
            value = float(text.get_text())
        except ValueError:
            continue
        text.set_color("white" if value >= threshold else "black")


def plot_heatmaps(no_rc_pivot: pd.DataFrame, rc_pivot: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    values = np.concatenate([
        no_rc_pivot.to_numpy(dtype=float).ravel(),
        rc_pivot.to_numpy(dtype=float).ravel(),
    ])
    values = values[~np.isnan(values)]

    if len(values) == 0:
        raise ValueError("No valid peak_memory values found for plotting.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    threshold = vmin + 0.58 * (vmax - vmin)

    fig = plt.figure(figsize=(11.6, 4.8))

    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.28,
    )

    ax_no_rc = fig.add_subplot(gs[0, 0])
    ax_rc = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    # 绝对值更适合连续色系，而不是 RdBu_r 这种差值色系
    cmap = sns.color_palette("Blues", as_cmap=True)

    heatmap_common_kwargs = dict(
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.55,
        linecolor="white",
        square=False,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 8.2},
    )

    sns.heatmap(
        no_rc_pivot,
        ax=ax_no_rc,
        cbar=False,
        mask=no_rc_pivot.isna(),
        **heatmap_common_kwargs,
    )

    sns.heatmap(
        rc_pivot,
        ax=ax_rc,
        cbar=True,
        cbar_ax=cbar_ax,
        cbar_kws={"label": "Peak memory (GB)"},
        mask=rc_pivot.isna(),
        **heatmap_common_kwargs,
    )

    adjust_annotation_colors(ax_no_rc, threshold)
    adjust_annotation_colors(ax_rc, threshold)

    # ===============================
    # Left panel
    # ===============================
    ax_no_rc.set_title("noRC", fontsize=13, pad=7)
    ax_no_rc.set_xlabel("Rank ID", fontsize=12)
    ax_no_rc.set_ylabel("Optimization configuration", fontsize=12)

    ax_no_rc.set_xticklabels(
        [str(x) for x in no_rc_pivot.columns],
        rotation=0,
        fontsize=10,
    )
    ax_no_rc.set_yticklabels(
        no_rc_pivot.index.tolist(),
        rotation=0,
        fontsize=10,
    )

    # ===============================
    # Right panel
    # ===============================
    ax_rc.set_title("RC", fontsize=13, pad=7)
    ax_rc.set_xlabel("Rank ID", fontsize=12)
    ax_rc.set_ylabel("")

    ax_rc.set_xticklabels(
        [str(x) for x in rc_pivot.columns],
        rotation=0,
        fontsize=10,
    )

    ax_rc.set_yticklabels([])
    ax_rc.tick_params(axis="y", length=0)

    # ===============================
    # Borders and ticks
    # ===============================
    for ax in [ax_no_rc, ax_rc]:
        set_heatmap_border(ax)
        ax.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=3.0,
            width=0.9,
            color="black",
        )

    cbar_ax.tick_params(labelsize=10, width=0.9, length=3)
    cbar_ax.set_ylabel("Peak memory (GB)", fontsize=12)
    set_heatmap_border(cbar_ax, linewidth=1.0)

    fig.subplots_adjust(
        top=0.92,
        bottom=0.18,
        left=0.15,
        right=0.93,
    )

    png_path = OUTPUT_DIR / "figure_x2_peak_memory_heatmap_qwen15_1.8b_8k_dp1_rc_split.png"
    pdf_path = OUTPUT_DIR / "figure_x2_peak_memory_heatmap_qwen15_1.8b_8k_dp1_rc_split.pdf"

    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[OK] PNG saved to: {png_path}")
    print(f"[OK] PDF saved to: {pdf_path}")


def main() -> None:
    df = load_peak_memory(CSV_PATH)

    base_order = sorted(df["base_config"].unique(), key=parse_base_config_label)
    rank_order = sorted(df["rank_id"].unique())

    no_rc_pivot = build_pivot(df, "noRC", base_order, rank_order)
    rc_pivot = build_pivot(df, "RC", base_order, rank_order)

    no_rc_pivot = no_rc_pivot / 1024
    rc_pivot = rc_pivot / 1024

    print(f"[INFO] Loaded rows after filtering dp=1: {len(df)}")
    print(f"[INFO] Number of base configurations: {len(base_order)}")
    print(f"[INFO] Number of ranks: {len(rank_order)}")
    print(f"[INFO] noRC matrix shape: {no_rc_pivot.shape}")
    print(f"[INFO] RC matrix shape: {rc_pivot.shape}")

    values = np.concatenate([
        no_rc_pivot.to_numpy(dtype=float).ravel(),
        rc_pivot.to_numpy(dtype=float).ravel(),
    ])
    values = values[~np.isnan(values)]

    print(f"[INFO] Peak memory range: {values.min():.2f} - {values.max():.2f} GB")

    plot_heatmaps(no_rc_pivot, rc_pivot)


if __name__ == "__main__":
    main()