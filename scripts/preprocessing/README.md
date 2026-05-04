# Preprocessing pipeline (`profiler` → LM-SimBench CSV layout)

Scripts are intended to run **in numeric order** (`01_` … `10_`). Each accepts `--help` for defaults (input/output roots, glob patterns, and flags).

| Step | Script | Role |
|------|--------|------|
| 01 | `01_filter_profiler_data.py` | Copy raw profiler trees while excluding large or sensitive detail files. |
| 02 | `02_extract_matrix_operators.py` | Extract matrix-operator latency details per model folder. |
| 03 | `03_extract_operator_categories.py` | Aggregate operator latency by category per model folder. |
| 04 | `04_extract_step_trace_time.py` | Extract step-trace time distribution per model / config / rank. |
| 05 | `05_extract_peak_memory.py` | Extract peak memory by model, parallel config, and rank. |
| 06 | `06_extract_iteration_time.py` | Extract per-configuration iteration time summaries. |
| 07 | `07_extract_communication.py` | Extract communication metrics per model folder. |
| 08 | `08_normalize_layout.py` | Normalize CSV layout and optionally flatten profiler output directories. |
| 09 | `09_rename_rank_dirs.py` | Rename worker directories to `rank_<id>` using profiler metadata. |
| 10 | `10_normalize_csv_headers.py` | Normalize selected CSV headers to snake_case. |

Typical invocation (from a working directory where default relative paths apply):

```bash
python scripts/preprocessing/01_filter_profiler_data.py --help
```

Adjust `--output-root`, `--input-root`, or equivalent flags per your machine; defaults are defined inside each script.
