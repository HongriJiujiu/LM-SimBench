#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_DOCX_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DOCX_ROOT))
from paths import dataset_root, figures_case1_dir


# ===============================
# 1. Paths
# ===============================
CSV_PATH = dataset_root() / "qwen3_0.6b_4k" / "step_trace_time_summary.csv"

OUTPUT_DIR = figures_case1_dir()


# ===============================
# 2. 绘图风格
# ===============================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
})


# ===============================
# 3. 时间单位换算
# ===============================
# 若结果明显偏大/偏小，改这里
TIME_SCALE = 10_000_000.0


# ===============================
# 4. 配置解析
# ===============================
def parse_parallel_config(config: str) -> dict:
    def extract_int(pattern: str, text: str, default: int = -1):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default

    return {
        "dp": extract_int(r"dp(\d+)", config),
        "tp": extract_int(r"tp(\d+)", config),
        "pp": extract_int(r"pp(\d+)", config),
        "cp": extract_int(r"cp(\d+)", config),
        "recompute": "noRC" if "noRC" in config else "RC",
    }


# ===============================
# 5. 读取数据
# ===============================
df = pd.read_csv(CSV_PATH)

breakdown_cols = [
    "computing",
    "communication_not_overlapped",
    "free",
    "bubble",
]

required_cols = ["optimization_config", "rank_id"] + breakdown_cols

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

parsed = df["optimization_config"].apply(parse_parallel_config).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

# 只保留 dp=1
df = df[df["dp"] == 1].copy()

# 同一配置多 rank 取平均
agg_df = (
    df.groupby("optimization_config", as_index=False)
    .agg({
        "computing": "mean",
        "communication_not_overlapped": "mean",
        "free": "mean",
        "bubble": "mean",
        "tp": "first",
        "pp": "first",
        "cp": "first",
        "recompute": "first",
    })
)

# 单位转秒
for col in breakdown_cols:
    agg_df[col] = agg_df[col] / TIME_SCALE

# 横轴标签：去掉 dp、RC
agg_df["config_label"] = (
    agg_df["optimization_config"]
    .str.replace(r"dp\d+-", "", regex=True)
    .str.replace("-noRC", "", regex=False)
    .str.replace("-RC", "", regex=False)
)

agg_df["recompute_order"] = agg_df["recompute"].map({"noRC": 0, "RC": 1})

agg_df = agg_df.sort_values(
    by=["tp", "pp", "cp", "recompute_order"]
).reset_index(drop=True)


# ===============================
# 6. 绘图
# ===============================
x = np.arange(len(agg_df))

fig_width = max(8, len(agg_df) * 0.55)
fig_height = 4.8

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# 色系优化
component_colors = {
    "computing": "#4575B4",
    "communication_not_overlapped": "#D73027",
    "free": "#FDAE61",
    "bubble": "#91BFDB",
}

component_labels = {
    "computing": "Computing",
    "communication_not_overlapped": "Communication (not overlapped)",
    "free": "Free",
    "bubble": "Bubble",
}

bottom = np.zeros(len(agg_df))

for col in breakdown_cols:
    ax.bar(
        x,
        agg_df[col],
        bottom=bottom,
        width=0.48,
        color=component_colors[col],
        edgecolor="black",
        linewidth=0.55,
        label=component_labels[col],
        zorder=3,
    )
    bottom += agg_df[col].to_numpy()

# 顶部标注总时间
for xi, total in zip(x, bottom):
    ax.text(
        xi,
        total,
        f"{total:.2f}s",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    agg_df["config_label"],
    rotation=35,
    ha="right",
)

ax.set_xlim(-0.55, len(agg_df) - 0.45)

ax.set_ylabel("Step trace time breakdown (s)")
ax.set_xlabel("Optimization configuration")

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.45,
)

ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=component_colors[col], edgecolor="black", label=component_labels[col])
    for col in breakdown_cols
]

ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    ncol=4,
    frameon=True,
    edgecolor="0.8",
)

ax.set_ylim(0, bottom.max() * 1.18)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.93])


# ===============================
# 7. 保存
# ===============================
png_path = OUTPUT_DIR / "figure_x3_step_trace_breakdown_qwen3_0.6b_4k_dp1.png"
pdf_path = OUTPUT_DIR / "figure_x3_step_trace_breakdown_qwen3_0.6b_4k_dp1.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)

plt.show()

print(f"[Saved] {png_path}")
print(f"[Saved] {pdf_path}")