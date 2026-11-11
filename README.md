# LM-SimBench — Benchmark Code

Reusable evaluation metrics, preprocessing utilities, and reference launch scripts for the LM-SimBench profiling benchmark. Public CSV releases are hosted on Hugging Face: [LM-SimBench](https://huggingface.co/datasets/anonymous-neurips26-ljasd/LM-SimBench), [LM-SimBench_raw](https://huggingface.co/datasets/anonymous-neurips26-ljasd/LM-SimBench_raw).

This repository contains the metrics package, NPU preprocessing and launch scripts,
and case-study plotting tools. It does not currently include the cross-platform
reproduction package, the vTrain adapter or its calibration/test splits.

## Repository layout (`benchmark/`)

```text
benchmark/
  README.md                 # This file
  pyproject.toml            # Installable `lm_simbench` package metadata
  requirements.txt          # Core Python dependencies
  src/
    lm_simbench/            # Metrics and task helpers
  scripts/
    preprocessing/          # Profiler → dataset CSV pipeline (01_ … 10_)
    launch/                 # Reference training launch scripts (*.sh), sanitized
  docx/                     # Figure-reproduction scripts (matplotlib / seaborn)
    requirements-plot.txt   # Extra dependencies for plotting
    case_1/                 # Case-study 1 (qwen15_1.8b_8k); bundled CSVs under case_1/result/
    case_2/                 # Case-study 2 (qwen25_32b_32k); place profiler CSVs under case_2/data/
    case_2/matrix_operator_prediction/  # Matrix prediction figures; input CSVs under data/
```

## Installation (Python package)

From the **`benchmark/`** directory:

```bash
pip install -r requirements.txt
pip install -e .
```

Without installing the package, add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/benchmark/src:$PYTHONPATH
python -c "from lm_simbench import mae, strategy_ranking_summary"
```

Core dependencies are listed in `requirements.txt` and `pyproject.toml` (`numpy`, `pandas`, `scipy`).

## Preprocessing pipeline

Step order and CLIs are documented in [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md). Run each script with `--help` for paths and options.

## Launch scripts

[`scripts/launch/`](scripts/launch/) contains **sanitized** shell scripts illustrating how profiling runs were configured. Host paths, cluster identifiers, and credentials are removed or replaced with placeholders.

## Metrics API (overview)

- **System-level**: iteration time (MAE, sMAPE), peak-memory aggregation, feasibility vs. device memory budget.
- **Component-level**: operator / matrix / communication latency metrics (MAE, sMAPE, R²), optional grouping.
- **Ranking**: Spearman correlation, top-k hit, top-k regret over parallel configurations.

Import from `lm_simbench` after installation or `PYTHONPATH` setup; see `src/lm_simbench/` for details.

Ranking functions require unique configuration names and aligned, finite,
nonnegative time vectors. Missing measurements or predictions raise `ValueError`;
they are not interpreted as successful recommendations. Relative regret is a
fraction; multiply by 100 to report a percentage. Predicted ties retain input
order, and any tied measured optimum counts as a Top-k hit.

Component joins require unique, non-null keys on each side. Aggregate repeated
records to the intended evaluation unit before joining. Grouped regression
tables report `n` finite pairs and `n_total` input rows; regression metrics omit
nonfinite pairs.

---

## Figure scripts (`docx/`)

Additional dependencies:

```bash
pip install -r docx/requirements-plot.txt
```

The following assumes the working directory is **`benchmark/`** (this README’s directory). Scripts resolve inputs and outputs with **relative paths** only.

### Case 1 — `docx/case_1`

Bundled CSVs under **`docx/case_1/result/`** (`step_time_summary.csv`, `peak_memory.csv`, and files used for Calculon comparison) support reproducing the primary case-study figures without extra downloads.

```bash
cd docx/case_1
python plot_step_time_case1.py
python plot_peak_memory_heatmap_qwen15_1p8b_8k.py
```

Outputs: `docx/case_1/outputs/` (PNG and PDF).

```bash
cd docx/case_1/result
python plot_calculon_vs_profiler_qwen15_1p8b.py
python compute_paper_benchmark_metrics.py
```

Outputs: `docx/case_1/result/outputs/` (figures and CSV summaries).

### Case 2 — `docx/case_2`

Plotting scripts read profiler exports under **`docx/case_2/data/`**, following the layout documented at the top of each script. Typical paths include:

- `docx/case_2/data/qwen25_32b_32k/peak_memory.csv`
- `docx/case_2/data/qwen25_32b_32k/step_trace_time_summary.csv`
- `docx/case_2/data/qwen25_32b_32k/<parallel_config>/rank_0/comm_time.csv`
- `docx/case_2/data/qwen25_32b_32k/<parallel_config>/rank_0/matrix_operator_latency.csv`

After placing CSVs that match the expected schema (see each `.py` file), run from **`benchmark/`**:

```bash
cd docx/case_2
python plot_peak_memory_lines_qwen25_32b_32k.py
python plot_step_trace_stacked_rank_breakdown.py
python plot_comm_time_donut.py
python plot_matrix_operator_bubble.py
```

Outputs: `docx/case_2/outputs/`.

Profiler CSVs with the same column layout can be obtained from the Hugging Face releases or by running the preprocessing pipeline on your exports.

### Matrix operator prediction — `docx/case_2/matrix_operator_prediction`

`plot_prediction_effect.py` reads per-configuration files named `*_predicted.csv` under **`matrix_operator_prediction/data/`** (required columns include `type`, `duration_us`, `pred_total_us`, `m`, `n`, `k`). It writes figures and per-group metric tables to **`matrix_operator_prediction/outputs/`**.

These files must contain both measured and predicted durations; downloading
profiler measurements alone does not supply `pred_total_us`. Existing outputs
for other configurations are retained. Missing input files raise an error
without deleting saved results.

```bash
cd docx/case_2/matrix_operator_prediction
python plot_prediction_effect.py
```

Further layout and I/O details: [`docx/README.md`](docx/README.md).

---

## Citation

If you use LM-SimBench in your research, please cite our [paper](https://arxiv.org/abs/2611.14285):

```bibtex
@misc{dong2026lmsimbench,
  title         = {{LM-SimBench}: A Unified Framework, Dataset, and Benchmark for Large Language Model Training Simulation},
  author        = {Yu Dong and Hong Li and Qiang Sun and Xu Zhou and Jinbiao Huo and Qixiu Cheng and Zhiyuan Liu},
  year          = {2026},
  eprint        = {2611.14285},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.14285}
}
```

For dataset details, see the [Hugging Face dataset card](https://huggingface.co/datasets/anonymous-neurips26-ljasd/LM-SimBench).

## License

Code is licensed under [Apache-2.0](LICENSE). Profiling data and derived data
tables use [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Checks

After installing the core and plotting dependencies, run:

```bash
python -m unittest discover -s tests -v
```
