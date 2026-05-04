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
| `paths.py` | `repo_root()`, `dataset_root()`, figure-output helpers |
| `case_1/` | `qwen3_0.6b_4k` step time, peak-memory heatmap, step-trace breakdown |
| `case_2/` | `qwen25_32b_32k` memory lines, communication breakdown, matrix-operator bubble, per-rank stacked traces |
| `matrix_operator_prediction/` | Predicted-vs-actual matrix latency plots |

## Figure outputs

Generated files are written under the **repository root** (unless you customize code):

| Directory | Contents |
|-----------|----------|
| `<repo_root>/case_1/figures/` | Step time bars, peak-memory heatmaps, step-trace breakdown (PNG/PDF per script) |
| `<repo_root>/case_2/figures/` | Peak memory curves, communication bar/inset, matrix-operator bubble |
| `<repo_root>/case_2/figures/rank_step_breakdown/` | Per-config stacked rank plots |
| `matrix_operator_prediction/figures/` | `prediction_effect_*`, `prediction_metrics_*` |

Run scripts from their folder, for example:

```bash
cd benchmark/docx/case_1
python plot_step_time_case1.py
```

## `plot_step_time_case1.py` note

If the CSV follows the public release schema (`iteration_time_ms` instead of `avg_elapsed_time_per_iteration_ms`), the script renames the column automatically.

## Matrix prediction subdirectory

See [`matrix_operator_prediction/README.md`](matrix_operator_prediction/README.md).
