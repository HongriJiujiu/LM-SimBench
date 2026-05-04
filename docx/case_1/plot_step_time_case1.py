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


MODEL_CASE1_QWEN3 = "qwen3_0.6b_4k"

CSV_PATH = dataset_root() / MODEL_CASE1_QWEN3 / "step_time_summary.csv"

OUTPUT_DIR = figures_case1_dir()


# ===============================
# 1. 论文级绘图风格
# ===============================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 7,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})


# ===============================
# 2. 解析并行配置
# ===============================
def parse_parallel_config(config: str) -> dict:
    def extract_int(pattern: str, text: str, default: int = -1) -> int:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default

    return {
        "dp": extract_int(r"dp(\d+)", config),
        "tp": extract_int(r"tp(\d+)", config),
        "pp": extract_int(r"pp(\d+)", config),
        "cp": extract_int(r"cp(\d+)", config),
        "recompute": "noRC" if "noRC" in config else "RC",
    }


def simplify_config_label(config: str) -> str:
    """
    dp1-tp2-pp4-cp1-RC / dp1-tp2-pp4-cp1-noRC
    -> tp2-pp4-cp1
    """
    label = config
    label = re.sub(r"^dp\d+-", "", label)
    label = label.replace("-noRC", "")
    label = label.replace("-RC", "")
    return label


# ===============================
# 3. 读取数据
# ===============================
df = pd.read_csv(CSV_PATH)

required_cols = [
    "optimization_config",
    "avg_elapsed_time_per_iteration_ms",
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

parsed = df["optimization_config"].apply(parse_parallel_config).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

df["config_label"] = df["optimization_config"].apply(simplify_config_label)

df["avg_elapsed_time_per_iteration_s"] = (
    df["avg_elapsed_time_per_iteration_ms"] / 1000.0
)

# 同一 dp/tp/pp/cp 下，将 noRC 和 RC 合并为一组
plot_df = (
    df.pivot_table(
        index=["dp", "tp", "pp", "cp", "config_label"],
        columns="recompute",
        values="avg_elapsed_time_per_iteration_s",
        aggfunc="mean",
    )
    .reset_index()
)

# 确保列存在，避免部分配置缺失时报错
for col in ["noRC", "RC"]:
    if col not in plot_df.columns:
        plot_df[col] = np.nan

plot_df = plot_df.sort_values(
    by=["dp", "tp", "pp", "cp"]
).reset_index(drop=True)


# ===============================
# 4. 绘图
# ===============================
color_map = {
    "noRC": "#1F77B4",
    "RC": "#D62728",
}

x = np.arange(len(plot_df))
bar_width = 0.34

y_max = np.nanmax(plot_df[["noRC", "RC"]].to_numpy())

fig_width = 14.5
fig_height = 4.8

fig, ax = plt.subplots(figsize=(fig_width, fig_height))


# ===============================
# 5. DP 分组边界与顶部标签
# ===============================
dp_groups = plot_df.groupby("dp", sort=False).indices

for dp_value, indices in dp_groups.items():
    start = min(indices) - 0.5
    end = max(indices) + 0.5

    if start > -0.5:
        ax.axvline(
            start,
            color="0.25",
            linewidth=0.9,
            linestyle="-",
            alpha=0.7,
            zorder=2,
        )

    ax.text(
        (start + end) / 2,
        y_max * 1.075,
        f"DP={dp_value}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="0.15",
    )


# ===============================
# 6. 并排柱状图
# ===============================
bars_no_rc = ax.bar(
    x - bar_width / 2,
    plot_df["noRC"],
    width=bar_width,
    color=color_map["noRC"],
    edgecolor="black",
    linewidth=0.55,
    label="noRC",
    zorder=3,
)

bars_rc = ax.bar(
    x + bar_width / 2,
    plot_df["RC"],
    width=bar_width,
    color=color_map["RC"],
    edgecolor="black",
    linewidth=0.55,
    label="RC",
    zorder=3,
)


# ===============================
# 7. 数值标注：横向显示
# ===============================
def add_bar_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_max * 0.008,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            rotation=0,
            zorder=4,
        )


add_bar_labels(bars_no_rc)
add_bar_labels(bars_rc)


# ===============================
# 8. 坐标轴与图例
# ===============================
ax.set_xticks(x)
ax.set_xticklabels(
    plot_df["config_label"],
    rotation=35,
    ha="right",
    fontsize=7,
)

ax.set_xlim(-0.6, len(plot_df) - 0.4)
ax.margins(x=0.001)

ax.set_ylabel("Training step time (s)")
ax.set_xlabel("Optimization configuration")

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    color="0.82",
    alpha=0.85,
    zorder=1,
)
ax.set_axisbelow(True)

legend_handles = [
    Patch(facecolor=color_map["noRC"], edgecolor="black", label="noRC"),
    Patch(facecolor=color_map["RC"], edgecolor="black", label="RC"),
]

ax.legend(
    handles=legend_handles,
    frameon=True,
    fancybox=False,
    edgecolor="black",
    facecolor="white",
    loc="upper left",
    bbox_to_anchor=(0.015, 0.985),
    ncol=1,
    borderpad=0.45,
    handlelength=1.5,
    handletextpad=0.5,
)

ax.set_ylim(0, y_max * 1.16)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.1)

ax.tick_params(
    axis="both",
    which="both",
    direction="in",
    top=True,
    right=True,
)

plt.tight_layout()


# ===============================
# 9. 保存图片
# ===============================
png_path = OUTPUT_DIR / "figure_x1_step_time_qwen3_0.6b_4k_grouped.png"
pdf_path = OUTPUT_DIR / "figure_x1_step_time_qwen3_0.6b_4k_grouped.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.08)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)

plt.show()

print(f"[Saved] {png_path}")
print(f"[Saved] {pdf_path}")