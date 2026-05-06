#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Paths (relative to this script)
#
# Reviewers should place their profiler exports under case_2/data/<model>/.
# The expected file is `peak_memory.csv` with columns
# `optimization_config, rank_id, peak_memory_mb`.
# ===============================
_CASE2_DIR = Path(__file__).resolve().parent

MODEL_NAME = "qwen25_32b_32k"

CSV_PATH = _CASE2_DIR / "data" / MODEL_NAME / "peak_memory.csv"
OUTPUT_DIR = _CASE2_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# 1. Paper-style plotting setup
# ===============================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,

    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,

    "axes.linewidth": 1.2,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
})


def extract_int(pattern: str, text: str, default: int = -1) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default


def parse_config(config: str):
    rc_order = 0 if config.endswith("noRC") else 1
    return (
        extract_int(r"dp(\d+)", config),
        extract_int(r"tp(\d+)", config),
        extract_int(r"pp(\d+)", config),
        extract_int(r"cp(\d+)", config),
        rc_order,
        config,
    )


def short_label(config: str) -> str:
    """
    保持配置名完整，但略微压缩显示。
    """
    return config.replace("dp", "d").replace("tp", "t").replace("pp", "p").replace("cp", "c")


# ===============================
# 2. Load data
# ===============================
if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Input file not found: {CSV_PATH}\n"
        "Place a profiler `peak_memory.csv` (columns: optimization_config, "
        "rank_id, peak_memory_mb) under this path."
    )

df = pd.read_csv(CSV_PATH)

required_cols = ["optimization_config", "rank_id", "peak_memory_mb"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df = df[required_cols].copy()
df["rank_id"] = pd.to_numeric(df["rank_id"], errors="coerce")
df["peak_memory_mb"] = pd.to_numeric(df["peak_memory_mb"], errors="coerce")
df = df.dropna(subset=["optimization_config", "rank_id", "peak_memory_mb"])

df["rank_id"] = df["rank_id"].astype(int)
df["peak_memory_gb"] = df["peak_memory_mb"] / 1024.0

configs = sorted(df["optimization_config"].unique(), key=parse_config)
ranks = sorted(df["rank_id"].unique())

pivot = df.pivot_table(
    index="rank_id",
    columns="optimization_config",
    values="peak_memory_gb",
    aggfunc="max",
).reindex(index=ranks, columns=configs)


# ===============================
# 3. Plot
# ===============================
fig, ax = plt.subplots(figsize=(8.6, 5.0))

# 更接近论文风格的高对比色
colors = [
    "#D62728",  # red
    "#1F77B4",  # blue
    "#2CA02C",  # green
    "#FF7F0E",  # orange
]

markers = ["o", "s", "^", "D"]

for idx, config in enumerate(configs):
    ax.plot(
        pivot.index,
        pivot[config],
        color=colors[idx % len(colors)],
        marker=markers[idx % len(markers)],
        markersize=5.5,
        markerfacecolor="white",
        markeredgewidth=1.4,
        linewidth=2.2,
        label=short_label(config),
        zorder=3,
    )


# ===============================
# 4. Axes
# ===============================
ax.set_xlabel("Rank ID")
ax.set_ylabel("Peak memory (GB)")

ax.set_xticks(ranks)
ax.set_xticklabels([str(r) for r in ranks])

y_min = np.nanmin(pivot.to_numpy())
y_max = np.nanmax(pivot.to_numpy())
y_pad = (y_max - y_min) * 0.12

ax.set_ylim(max(0, y_min - y_pad), y_max + y_pad)

ax.grid(
    axis="both",
    linestyle="--",
    linewidth=0.8,
    color="#D9D9D9",
    alpha=0.85,
    zorder=0,
)

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
)

# 保留完整边框，更像你给的第一张学术折线图
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)


# ===============================
# 5. Legend
# ===============================
ax.legend(
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    loc="upper left",
    bbox_to_anchor=(0.02, 0.78),
    ncol=1,
    columnspacing=1.0,
    handlelength=2.2,
    handletextpad=0.5,
    borderpad=0.5,
)

# 不放大标题，论文中通常交给 caption
# ax.set_title(f"Rank-level Peak Memory across Configurations ({MODEL_NAME})")

plt.tight_layout()


# ===============================
# 6. Save
# ===============================
png_path = OUTPUT_DIR / "figure_x5_peak_memory_lines_qwen25_32b_32k.png"
pdf_path = OUTPUT_DIR / "figure_x5_peak_memory_lines_qwen25_32b_32k.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06)

plt.show()

print(f"[Saved] {png_path}")
print(f"[Saved] {pdf_path}")