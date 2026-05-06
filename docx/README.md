# 论文图表脚本（`docx/`）

本目录下的脚本根据 LM-SimBench 导出的 CSV（或由预处理流水线生成的同 schema 数据）复现论文中的案例图。**所有路径均为相对于脚本文件的相对路径**，不依赖环境变量或 `paths.py` 一类集中配置。

更完整的一键命令与仓库总览见上一级：[`benchmark/README.md`](../README.md)。

## 依赖

在 **`benchmark/`** 目录下：

```bash
pip install -r docx/requirements-plot.txt
```

除 `pandas` / `numpy` 外，主要使用 `matplotlib`、`seaborn`。

## 路径约定

- 每个脚本用 `Path(__file__).resolve().parent` 定位所在目录，再拼出输入 / 输出子路径。
- 工作目录可以是任意位置，只要按文档中的方式调用脚本（例如 `python plot_xxx.py`）；**推荐**在对应子目录或 `benchmark/` 下执行，便于阅读报错中的相对路径说明。

## 目录一览

| 路径 | 说明 |
|------|------|
| `docx/case_1/` | 案例一（**qwen15_1.8b_8k**）：step 时间柱状图、峰值显存热力图；输入 CSV 放在同目录下的 `result/`（仓库已附带） |
| `docx/case_1/result/` | Calculon 与 profiler 对比图、论文 benchmark 指标计算；依赖本目录内 `*.csv` |
| `docx/case_2/` | 案例二（**qwen25_32b_32k**）：折线、堆叠 rank、通信、矩阵算子图；原始 profiler CSV 需放在 **`case_2/data/`** 下（见下表） |
| `docx/case_2/matrix_operator_prediction/` | 矩阵算子预测效果：读取 **`data/*_predicted.csv`**，输出到 **`outputs/`** |

### `case_1` 脚本（输入已打包时可直跑）

| 脚本 | 输入（相对 `case_1/`） | 输出 |
|------|------------------------|------|
| `plot_step_time_case1.py` | `result/step_time_summary.csv` | `outputs/` |
| `plot_peak_memory_heatmap_qwen15_1p8b_8k.py` | `result/peak_memory.csv` | `outputs/` |
| `result/plot_calculon_vs_profiler_qwen15_1p8b.py` | `result/*.csv`（见脚本内说明） | `result/outputs/` |
| `result/compute_paper_benchmark_metrics.py` | 默认使用 `result/outputs/` 下合并表等 | `result/outputs/` |

### `case_2` 脚本（需在 `data/` 中自备 CSV）

默认模型目录名为 **`qwen25_32b_32k`**。若仓库中尚未创建目录，请先创建  
`docx/case_2/data/qwen25_32b_32k/`，并按脚本顶部注释放入对应文件，例如：

| 脚本 | 期望输入路径（相对 `case_2/`） | 输出 |
|------|--------------------------------|------|
| `plot_peak_memory_lines_qwen25_32b_32k.py` | `data/qwen25_32b_32k/peak_memory.csv` | `outputs/` |
| `plot_step_trace_stacked_rank_breakdown.py` | `data/qwen25_32b_32k/step_trace_time_summary.csv` | `outputs/` |
| `plot_comm_time_donut.py` | `data/qwen25_32b_32k/dp1-tp8-pp4-cp1-RC/rank_0/comm_time.csv`（脚本内可改 `MODEL_NAME` / `CONFIG_NAME` / `RANK_NAME`） | `outputs/` |
| `plot_matrix_operator_bubble.py` | `data/qwen25_32b_32k/dp1-tp8-pp4-cp1-RC/rank_0/matrix_operator_latency.csv`（同上） | `outputs/` |

列名与格式以各 `.py` 文件内校验逻辑为准。

### `matrix_operator_prediction`

| 脚本 | 输入 | 输出 |
|------|------|------|
| `plot_prediction_effect.py` | `matrix_operator_prediction/data/*_predicted.csv`（列至少含 `type`, `duration_us`, `pred_total_us`, `m`, `n`, `k`） | `matrix_operator_prediction/outputs/`（`prediction_effect_*.{png,pdf}`, `prediction_metrics_*.csv`） |

公开仓库往往**不包含**原始 `*_predicted.csv`，需自行从预测器或实验导出后再运行。

## 输出目录小结

| 位置 | 内容 |
|------|------|
| `docx/case_1/outputs/` | 案例一：`figure_x1_*`, `figure_x2_*` 等 PNG/PDF |
| `docx/case_1/result/outputs/` | Calculon 对比图、合并表、`benchmark_metrics_*.csv` 等 |
| `docx/case_2/outputs/` | 案例二各脚本的 PNG/PDF |
| `docx/case_2/matrix_operator_prediction/outputs/` | 预测效果图与派生指标 CSV |

## 快速运行示例

在 **`benchmark/`** 下：

```bash
# 案例一（数据已在 result/ 中）
cd docx/case_1
python plot_step_time_case1.py
python plot_peak_memory_heatmap_qwen15_1p8b_8k.py

cd result
python plot_calculon_vs_profiler_qwen15_1p8b.py
python compute_paper_benchmark_metrics.py
```

案例二在放好 `data/qwen25_32b_32k/...` 下 CSV 后：

```bash
cd docx/case_2
python plot_peak_memory_lines_qwen25_32b_32k.py
# … 其余脚本同上
```

## `plot_step_time_case1.py` 说明

若 CSV 使用公开发行列名 **`iteration_time_ms`**（而非 **`avg_elapsed_time_per_iteration_ms`**），脚本会自动重命名列以兼容。

## 无头 / CI 环境

部分脚本调用 `matplotlib` 的 `show()`。无图形界面时可设置环境变量  
`MPLBACKEND=Agg` 再运行，避免交互后端告警。
