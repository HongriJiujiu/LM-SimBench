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
  scripts/
    preprocessing/          # Profiler → dataset CSV pipeline (numbered steps)
    launch/                 # Sanitized MindSpeed-LLM training launch examples (*.sh)
  docx/                     # Paper-style figure scripts (matplotlib)
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

See [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md) for input/output directories, step order (`01_` … `10_`), and CLI options.

## Launch scripts

[`scripts/launch/`](scripts/launch/) contains **sanitized** shell scripts illustrating how profiling runs were started. They are references for reproducibility documentation; paths, cluster names, and secrets are removed or replaced.

## Metrics API (high level)

- **System-level**: iteration time (MAE, sMAPE), peak memory aggregation, feasibility vs. device memory cap.
- **Component-level**: operator / matrix / communication latency metrics (MAE, sMAPE, R²), optional grouping (e.g. comm type × message size bucket).
- **Ranking**: Spearman correlation, top-k hit, top-k regret on parallel configurations.

Import symbols from `lm_simbench` after installation or `PYTHONPATH` setup; see package docstrings in `src/lm_simbench/`.

## Visualization (paper figures)

Case-study plotting scripts live under [`docx/`](docx/README.md). Install extra deps with `pip install -r docx/requirements-plot.txt`.

## Citation

If you use this code or the dataset, cite the Hugging Face dataset card and your paper as appropriate. A BibTeX example is provided on the [dataset page](https://huggingface.co/datasets/jiujiudahaozi/LM-SimBench).

## License

Match the dataset license (e.g. CC BY-SA 4.0) and your repository’s `LICENSE` file.
