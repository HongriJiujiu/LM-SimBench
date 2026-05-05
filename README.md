# LM-SimBench — Benchmark Code

Reusable evaluation metrics, preprocessing utilities, and **sanitized example launch scripts** for the LM-SimBench profiling dataset. Public CSV releases are hosted on Hugging Face ([LM-SimBench](https://huggingface.co/datasets/jiujiudahaozi/LM-SimBench), [LM-SimBench_raw](https://huggingface.co/datasets/jiujiudahaozi/LM-SimBench_raw)).

## Repository layout

```text
benchmark/
  README.md                 # This file
  pyproject.toml            # Installable `lm_simbench` package metadata
  requirements.txt          # Pinned-style dependencies for quick setup
  src/
    lm_simbench/            # Python package: metrics and task helpers
      __init__.py
      component.py          # Component-level latency metrics
      metrics.py            # Shared metric helpers
      ranking.py            # Configuration ranking / regret
      system.py             # System-level iteration time & memory
  scripts/
    preprocessing/          # Profiler → dataset CSV pipeline (numbered steps)
      01_filter_profiler_data.py
      02_extract_matrix_operators.py
      03_extract_operator_categories.py
      04_extract_step_trace_time.py
      05_extract_peak_memory.py
      06_extract_iteration_time.py
      07_extract_communication.py
      08_normalize_layout.py
      09_rename_rank_dirs.py
      10_normalize_csv_headers.py
    launch/                 # Sanitized MindSpeed-LLM training launch examples (*.sh)
      gemma2_9b_8k.sh
      gemma2_27b_8k.sh
      glm4_9b_8k.sh
      internlm25_1point8b_32k.sh
      internlm25_20b_32k.sh
      internlm25_7b_32k.sh
      internlm3_8b_8k.sh
      mistral_7b_32k.sh
      phi35_mini_4k.sh
      phi35_moe_4k.sh
      qwen15_1.8b_8k.sh
      qwen15_4b_8k.sh
      qwen25_0point5_32k.sh
      qwen25_14b_8k.sh
      qwen25_32b_32k.sh
      qwen25_3b_32k.sh
      qwen25_72b_8k.sh
      qwen25_7b_8k.sh
      qwen2_1.5b_4k.sh
      qwen3_0.6b_4k.sh
      qwen3_1.7b_4k.sh
      qwen3_30b_a3b_4k.sh
      qwen3_8b_4k.sh
  docx/                     # Paper-style figure scripts (matplotlib); see docx/README.md
    paths.py
    requirements-plot.txt
    case_1/
    case_2/
    matrix_operator_prediction/
```

## Installation (Python package)

From the `benchmark` directory:

```bash
pip install -e .
```

Or without installing, add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/repo/benchmark/src:$PYTHONPATH
python -c "from lm_simbench import mae, strategy_ranking_summary"
```

Dependencies: see `requirements.txt` or `pyproject.toml` (`numpy`, `pandas`, `scipy`).

## Preprocessing pipeline

See [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md) for step order (`01_` … `10_`) and CLI entry points. Run each script with `--help` for input/output paths and options.

## Launch scripts

[`scripts/launch/`](scripts/launch/) contains **sanitized** shell scripts illustrating how profiling runs were started. They are references for reproducibility documentation; paths, cluster names, and secrets are removed or replaced.

## Metrics API (high level)

- **System-level**: iteration time (MAE, sMAPE), peak memory aggregation, feasibility vs. device memory cap.
- **Component-level**: operator / matrix / communication latency metrics (MAE, sMAPE, R²), optional grouping (e.g. comm type × message size bucket).
- **Ranking**: Spearman correlation, top-k hit, top-k regret on parallel configurations.

Import symbols from `lm_simbench` after installation or `PYTHONPATH` setup; see package docstrings in `src/lm_simbench/`.

## Visualization (paper figures)

Case-study plotting scripts live under [`docx/`](docx/README.md). Install extra deps with `pip install -r docx/requirements-plot.txt`.

## Maintenance Plan

LM-SimBench will be maintained as a versioned benchmark resource. We will:

- preserve released versions for reproducibility;
- document major updates in a changelog;
- update the benchmark scripts when dataset fields or task definitions change;
- fix reported issues related to data loading, inconsistent fields, or missing metadata;
- add new model families, parallel configurations, and profiling traces in future releases when available.

Users can report issues or request clarifications through GitHub Issues.

## License

The source code, preprocessing scripts, and benchmark examples in this repository are released under the Apache-2.0 license.

The LM-SimBench datasets are released separately under the CC BY 4.0 license on Hugging Face. The dataset license applies to the processed profiling tables, raw profiling traces, metadata, and dataset documentation.
