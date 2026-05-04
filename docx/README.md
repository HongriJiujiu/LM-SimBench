# Visualization examples (paper figures)

Scripts reproduce case-study figures from LM-SimBench CSV exports. **Paths are configurable** via environment variables so nothing machine-specific appears in source.

## Dependencies

```bash
pip install -r requirements-plot.txt
```

(Uses `matplotlib` and `seaborn` in addition to `pandas` / `numpy`.)

## Path configuration

Defaults assume this layout:

```text
<repo_root>/                 # LM_SIMBENCH_ROOT — parent of ``benchmark/``
  benchmark/docx/
  qwen3_0.6b_4k/             # unpacked model CSV tree
    step_time_summary.csv
    peak_memory.csv
    ...
  qwen25_32b_32k/
    dp1-tp8-pp4-cp1-RC/
      rank_0/
        ...
```

Optional overrides:

| Variable | Meaning |
|---------|---------|
| `LM_SIMBENCH_ROOT` | Repository root containing `benchmark/` and model dirs (otherwise inferred from `benchmark/docx/paths.py`). |
| `LM_SIMBENCH_DATA_ROOT` | Folder that contains `<model>/…` CSV trees (defaults to same as repo root when unset). |

## Layout

| Path | Purpose |
|------|---------|
| `paths.py` | `repo_root()`, `dataset_root()`, `figures_case1_dir()`, `figures_case2_dir()` |
| `case_1/` | `plot_step_time_case1.py`, `plot_peak_memory_heatmap_qwen3_0p6b_4k.py`, `plot_step_trace_breakdown_qwen3_0.6b_4k.py` (`qwen3_0.6b_4k`) |
| `case_2/` | `plot_peak_memory_lines_qwen25_32b_32k.py`, `plot_comm_time_donut.py`, `plot_matrix_operator_bubble.py`, `plot_step_trace_stacked_rank_breakdown.py` (`qwen25_32b_32k`) |
| `matrix_operator_prediction/` | `plot_prediction_effect.py` — predicted vs. actual matrix latency |

## Figure outputs

Case scripts resolve output directories via `paths.py` under the **repository root**:

| Directory | Contents |
|-----------|----------|
| `<repo_root>/case_1/outputs/` | Step time bars, peak-memory heatmaps, step-trace breakdown (PNG/PDF per script) |
| `<repo_root>/case_2/outputs/` | Peak memory curves, communication donut, matrix-operator bubble, stacked rank traces |

Matrix prediction figures are written next to the script:

| Directory | Contents |
|-----------|----------|
| `benchmark/docx/matrix_operator_prediction/figures/` | `prediction_effect_*`, `prediction_metrics_*` |

Run case scripts from `benchmark/docx` so `paths` resolves correctly:

```bash
cd benchmark/docx/case_1
python plot_step_time_case1.py
```

(Same pattern for `case_2/` scripts.)

## `plot_step_time_case1.py` note

If the CSV follows the public release schema (`iteration_time_ms` instead of `avg_elapsed_time_per_iteration_ms`), the script renames the column automatically.

## Matrix prediction subdirectory

See [`matrix_operator_prediction/README.md`](matrix_operator_prediction/README.md).
