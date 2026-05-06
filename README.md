# LM-SimBench — Benchmark Code

可复用的评估指标、预处理工具，以及面向公开评测数据的示例启动脚本。公开 CSV 数据托管在 Hugging Face：[LM-SimBench](https://huggingface.co/datasets/jiujiudahaozi/LM-SimBench)、[LM-SimBench_raw](https://huggingface.co/datasets/jiujiudahaozi/LM-SimBench_raw)。

## 仓库结构（`benchmark/`）

```text
benchmark/
  README.md                 # 本文件
  pyproject.toml            # 可安装的 lm_simbench 包元数据
  requirements.txt          # 快速环境依赖
  src/
    lm_simbench/            # Python 包：指标与任务辅助函数
  scripts/
    preprocessing/          # Profiler → 数据集 CSV 流水线（01_ … 10_）
    launch/                 # 脱敏后的训练启动示例（*.sh）
  docx/                     # 论文级作图脚本（matplotlib / seaborn）
    requirements-plot.txt   # 作图额外依赖
    case_1/                 # 案例一：qwen15_1.8b_8k 等图表（含已打包 CSV）
    case_2/                 # 案例二：qwen25_32b_32k 等（输入需放入 case_2/data/）
    case_2/matrix_operator_prediction/  # 矩阵算子预测效果图（需自备 *_predicted.csv）
```

## 环境安装（Python 包）

在 **`benchmark/`** 目录下：

```bash
pip install -r requirements.txt
pip install -e .
```

若暂不安装，可将 `src` 加入 `PYTHONPATH`：

```bash
export PYTHONPATH=/path/to/benchmark/src:$PYTHONPATH
python -c "from lm_simbench import mae, strategy_ranking_summary"
```

## 预处理流水线

步骤顺序与 CLI 见 [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md)。各脚本支持 `--help` 查看输入输出路径与选项。

## 启动脚本

[`scripts/launch/`](scripts/launch/) 中为**脱敏**后的训练启动示例，供复现文档引用；机器路径、集群名与敏感信息已移除或替换。

## 指标 API（概要）

- **系统级**：迭代时间（MAE、sMAPE）、峰值显存聚合、与设备显存上界的可行性等。
- **组件级**：算子 / 矩阵 / 通信等延迟指标（MAE、sMAPE、R² 等）。
- **排序**：Spearman、Top-k hit、Top-k regret 等。

安装或配置 `PYTHONPATH` 后从 `lm_simbench` 导入；细节见 `src/lm_simbench/` 内注释。

## 维护计划

LM-SimBench 作为有版本记录的评测资源维护，我们将：

- 保留已发布版本以便复现；
- 在变更日志中记录重大更新；
- 当数据集字段或任务定义变化时更新 benchmark 脚本；
- 修复与数据加载、字段不一致或元数据缺失相关的问题；
- 在后续版本中在条件允许时扩展模型族、并行配置与 profiling 轨迹。

欢迎在 GitHub Issues 中报告问题或提出澄清需求。

---

## 论文图表脚本（`docx/`）

作图依赖：

```bash
pip install -r docx/requirements-plot.txt
```

更细的目录说明见 [`docx/README.md`](docx/README.md)。

以下命令均假设当前工作目录为 **`benchmark/`**（即本 README 所在目录）。所有脚本使用相对路径，可在任意机器上克隆后直接运行（在数据已就位的前提下）。

### 示例 1：`docx/case_1`（自带 `result/` 下 CSV，可直接复现）

仓库内含 **`docx/case_1/result/`** 中的 `step_time_summary.csv`、`peak_memory.csv` 以及 Calculon 对比所需文件，下列脚本**无需额外下载**即可生成主文案例一相关图与合并表。

```bash
# 图：各并行配置下的 step 时间与 rank 级峰值显存热力图
cd docx/case_1
python plot_step_time_case1.py
python plot_peak_memory_heatmap_qwen15_1p8b_8k.py
```

输出目录：`docx/case_1/outputs/`（PNG / PDF）。

```bash
# Calculon 与 profiler 对比图 + 合并 CSV；并据合并结果计算论文 benchmark 指标表
cd docx/case_1/result
python plot_calculon_vs_profiler_qwen15_1p8b.py
python compute_paper_benchmark_metrics.py
```

输出目录：`docx/case_1/result/outputs/`（图、CSV）。

### 示例 2：`docx/case_2`（输入放在 `case_2/data/`）

`case_2` 下的作图脚本从**相对路径**读取模型目录，例如：

- `docx/case_2/data/qwen25_32b_32k/peak_memory.csv` — 折线图
- `docx/case_2/data/qwen25_32b_32k/step_trace_time_summary.csv` — 按 rank 的 step trace 堆叠图
- `docx/case_2/data/qwen25_32b_32k/<parallel_config>/rank_0/comm_time.csv` — 通信时间图
- `docx/case_2/data/qwen25_32b_32k/<parallel_config>/rank_0/matrix_operator_latency.csv` — 矩阵算子条形图

具体文件名与列要求见各 `.py` 文件顶部注释。将 profiler 导出的 CSV **按上述相对路径放置**后，在 `benchmark/` 下执行：

```bash
cd docx/case_2
python plot_peak_memory_lines_qwen25_32b_32k.py
python plot_step_trace_stacked_rank_breakdown.py
python plot_comm_time_donut.py
python plot_matrix_operator_bubble.py
```

输出目录：`docx/case_2/outputs/`。

> **说明**：本仓库未必附带完整的 `qwen25_32b_32k` profiler 原始 CSV；若文件缺失，脚本会报错并提示期望路径。可从 Hugging Face 数据集或本地跑通预处理流水线后得到同 schema 的 CSV。

### 矩阵算子预测图（不作一键示例）

`docx/case_2/matrix_operator_prediction/plot_prediction_effect.py` 从 `matrix_operator_prediction/data/` 读取若干 `*_predicted.csv`（内含 `duration_us`、`pred_total_us` 等列），结果写入 `matrix_operator_prediction/outputs/`。**公开仓库中通常不包含该原始预测输入**，故不在此列出可复制粘贴的一键运行示例；自备输入文件后再执行：

```bash
cd docx/case_2/matrix_operator_prediction
python plot_prediction_effect.py
```

---

## 引用

使用本代码或数据集时，请引用 Hugging Face 数据集页说明及您的论文。[数据集页](https://huggingface.co/datasets/jiujiudahaozi/LM-SimBench)。

## 许可

- **本仓库中的源代码、预处理脚本与 benchmark 示例**：Apache-2.0（见仓库内许可证文件，若存在）。
- **LM-SimBench 数据集**（处理后的 profiling 表、原始轨迹、元数据与数据集文档）：以 Hugging Face 上的 **CC BY 4.0** 为准；详见各数据集页的说明。
