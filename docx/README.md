# Figure scripts (`docx/`)

Scripts in this directory reproduce case-study figures from LM-SimBench CSV exports (or equivalently structured files produced by the preprocessing pipeline). Paths are **relative to each script file**; there is no central path module or environment-variable indirection.

For install commands and a compact runbook, see [`benchmark/README.md`](../README.md).

## Dependencies

From **`benchmark/`**:

```bash
pip install -r docx/requirements-plot.txt
```

Uses `matplotlib`, `seaborn`, `pandas`, and `numpy`.

## Path conventions

- Each script uses `Path(__file__).resolve().parent` as the base directory for inputs and outputs.
- Run scripts as `python <script>.py` from any working directory; running from `benchmark/` or the script’s folder makes error messages easier to interpret.

## Directory map

| Path | Description |
|------|-------------|
| `docx/case_1/` | Case study 1 (**qwen15_1.8b_8k**): step-time bar chart, peak-memory heatmap; reads CSVs under `case_1/result/` (included in this tree) |
| `docx/case_1/result/` | Calculon vs. profiler comparison plots and benchmark-style metric tables |
| `docx/case_2/` | Case study 2 (**qwen25_32b_32k**): line plots, rank breakdown, communication, matrix-operator plots; reads profiler CSVs under **`case_2/data/`** |
| `docx/case_2/matrix_operator_prediction/` | Matrix prediction diagnostics: reads **`data/*_predicted.csv`**, writes to **`outputs/`** |

### Case 1 scripts

| Script | Input (relative to `case_1/`) | Output |
|--------|-------------------------------|--------|
| `plot_step_time_case1.py` | `result/step_time_summary.csv` | `outputs/` |
| `plot_peak_memory_heatmap_qwen15_1p8b_8k.py` | `result/peak_memory.csv` | `outputs/` |
| `result/plot_calculon_vs_profiler_qwen15_1p8b.py` | CSVs in `result/` (see script docstring) | `result/outputs/` |
| `result/compute_paper_benchmark_metrics.py` | Defaults under `result/outputs/` | `result/outputs/` |

### Case 2 scripts

Default model folder name: **`qwen25_32b_32k`**. Create  
`docx/case_2/data/qwen25_32b_32k/` and add the CSVs expected by each script (see script headers). Example layout:

| Script | Expected input (relative to `case_2/`) | Output |
|--------|----------------------------------------|--------|
| `plot_peak_memory_lines_qwen25_32b_32k.py` | `data/qwen25_32b_32k/peak_memory.csv` | `outputs/` |
| `plot_step_trace_stacked_rank_breakdown.py` | `data/qwen25_32b_32k/step_trace_time_summary.csv` | `outputs/` |
| `plot_comm_time_donut.py` | `data/qwen25_32b_32k/dp1-tp8-pp4-cp1-RC/rank_0/comm_time.csv` (`MODEL_NAME` / `CONFIG_NAME` / `RANK_NAME` are configurable in the script) | `outputs/` |
| `plot_matrix_operator_bubble.py` | `data/qwen25_32b_32k/dp1-tp8-pp4-cp1-RC/rank_0/matrix_operator_latency.csv` | `outputs/` |

Column names and dtypes are validated inside each Python file.

### Matrix operator prediction

| Script | Input | Output |
|--------|-------|--------|
| `plot_prediction_effect.py` | `matrix_operator_prediction/data/*_predicted.csv` (minimum columns: `type`, `duration_us`, `pred_total_us`, `m`, `n`, `k`; optional `rank_id`, `rank_name`) | `matrix_operator_prediction/outputs/` (`prediction_effect_*.{png,pdf}`, `prediction_metrics_*.csv`) |

## Output locations

| Location | Contents |
|----------|----------|
| `docx/case_1/outputs/` | Case 1: `figure_x1_*`, `figure_x2_*`, etc. |
| `docx/case_1/result/outputs/` | Calculon comparison figures, merged tables, `benchmark_metrics_*.csv` |
| `docx/case_2/outputs/` | Case 2 figure exports |
| `docx/case_2/matrix_operator_prediction/outputs/` | Prediction figures and derived metric CSVs |

## Example commands

From **`benchmark/`**:

```bash
cd docx/case_1
python plot_step_time_case1.py
python plot_peak_memory_heatmap_qwen15_1p8b_8k.py

cd result
python plot_calculon_vs_profiler_qwen15_1p8b.py
python compute_paper_benchmark_metrics.py
```

After Case 2 inputs are available under `data/qwen25_32b_32k/`:

```bash
cd docx/case_2
python plot_peak_memory_lines_qwen25_32b_32k.py
python plot_step_trace_stacked_rank_breakdown.py
python plot_comm_time_donut.py
python plot_matrix_operator_bubble.py
```

## Notes

**`plot_step_time_case1.py`**: if the CSV uses the public schema column `iteration_time_ms` instead of `avg_elapsed_time_per_iteration_ms`, the script renames it automatically.

**Headless environments**: some scripts call `plt.show()`. Set `MPLBACKEND=Agg` when no display is available.
