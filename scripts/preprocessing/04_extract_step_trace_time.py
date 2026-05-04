#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import re
import pandas as pd


DEFAULT_INPUT_ROOT = Path("raw_filtered")
DEFAULT_OUTPUT_ROOT = Path("LM-SimBench")

OUTPUT_CSV_NAME = "step_trace_time_summary.csv"
WARNING_LOG_NAME = "step_trace_time_warnings.log"

TIME_COLUMNS = [
    "Computing",
    "Communication(Not Overlapped)",
    "Overlapped",
    "Communication",
    "Free",
    "Stage",
    "Bubble",
    "Communication(Not Overlapped and Exclude Receive)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract time distribution by model, configuration, and rank."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"过滤后的 profiler 数据根目录，默认: {DEFAULT_INPUT_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"最终结果根目录，默认: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="可选：只处理指定模型；不传则遍历输入目录下的所有一级模型目录。",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="可选：遍历所有模型时需要排除的一级目录名。",
    )
    return parser.parse_args()


def get_model_dirs(input_root: Path, models: list[str] | None, exclude_models: set[str]) -> list[Path]:
    available_model_dirs = {
        p.name: p for p in input_root.iterdir()
        if p.is_dir()
    }

    if models is None:
        return sorted(
            p for name, p in available_model_dirs.items()
            if name not in exclude_models
        )

    missing_models = sorted(set(models) - set(available_model_dirs))
    if missing_models:
        print("[WARN] 以下模型在输入目录中不存在，将跳过:")
        for name in missing_models:
            print(f"  - {name}")

    return sorted(
        available_model_dirs[name]
        for name in set(models) & set(available_model_dirs)
        if name not in exclude_models
    )


def normalize_col_name(col: str) -> str:
    if col is None:
        return ""

    col = str(col).strip().lstrip("\ufeff").strip('"').strip("'").lower()
    return re.sub(r"[^a-z0-9]+", "", col)


def detect_delimiter_from_header(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            if line.strip():
                header = line
                break
        else:
            return ","

    candidates = [",", "\t", ";"]
    counts = {sep: header.count(sep) for sep in candidates}

    best_sep = max(counts, key=counts.get)

    if counts[best_sep] == 0:
        return ","

    return best_sep


def read_rank_id(profiler_output_dir: Path):
    """
    profiler_info_*.json 与 ASCEND_PROFILER_OUTPUT 文件夹同目录。

    目录结构：
      worker-xxx/
        profiler_info_0.json 或 profiler_info_1.json ...
        ASCEND_PROFILER_OUTPUT/
          step_trace_time.csv

    注意：
      profiler_info_*.json 文件名中的数字不是固定的，
      因此不能写死 profiler_info_0.json。
    """
    worker_dir = profiler_output_dir.parent

    info_files = sorted(worker_dir.glob("profiler_info*.json"))

    if not info_files:
        return None, f"[ERROR] Missing profiler_info*.json under: {worker_dir}"

    if len(info_files) > 1:
        # 一般一个 worker 目录下应该只有一个 profiler_info*.json。
        # 如果有多个，先取第一个，同时返回 warning。
        warning = (
            f"[WARN] Multiple profiler_info*.json files under {worker_dir}, "
            f"use {info_files[0]}"
        )
    else:
        warning = None

    info_file = info_files[0]

    try:
        with info_file.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception as e:
        return None, f"[ERROR] Failed to read {info_file}: {repr(e)}"

    rank_id = info.get("rank_id")

    if rank_id is None:
        return None, f"[ERROR] Missing rank_id in {info_file}"

    rank_id = str(rank_id).strip()

    if not rank_id:
        return None, f"[ERROR] Empty rank_id in {info_file}"

    if warning is not None:
        return rank_id, warning

    return rank_id, None

def read_step_trace_time(profiler_output_dir: Path):
    """
    读取单个 ASCEND_PROFILER_OUTPUT/step_trace_time.csv。

    device_type 从同级 profiler_info_0.json 的 rank_id 获取：
      device_type = NPU:<rank_id>
    """
    step_csv = profiler_output_dir / "step_trace_time.csv"

    if not step_csv.is_file():
        return None, f"[ERROR] Missing step_trace_time.csv: {step_csv}"

    rank_id, warning = read_rank_id(profiler_output_dir)

    if warning is not None:
        return None, warning

    device_type = f"NPU:{rank_id}"

    sep = detect_delimiter_from_header(step_csv)

    try:
        df = pd.read_csv(
            step_csv,
            sep=sep,
            encoding="utf-8-sig",
            engine="python",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as e:
        return None, f"[ERROR] Failed to read {step_csv}: {repr(e)}"

    if df.empty:
        return None, f"[WARN] Empty step_trace_time.csv: {step_csv}"

    col_map = {
        normalize_col_name(col): col
        for col in df.columns
    }

    real_cols = {}

    for col in TIME_COLUMNS:
        real_col = col_map.get(normalize_col_name(col))

        if real_col is None:
            return None, (
                f"[ERROR] Missing column '{col}' in {step_csv}. "
                f"Parsed columns: {list(df.columns)}"
            )

        real_cols[col] = real_col

    warnings = []

    if len(df) > 1:
        warnings.append(f"[WARN] Multiple rows in {step_csv}, use the last row.")

    # 一般只有一行 Step=6；如果有多行，取最后一行
    src_row = df.iloc[-1]

    out_row = {
        "device_type": device_type,
    }

    for out_col, real_col in real_cols.items():
        value = src_row[real_col]

        numeric_value = pd.to_numeric(value, errors="coerce")

        if pd.isna(numeric_value):
            out_row[out_col] = str(value).strip()
        else:
            out_row[out_col] = float(numeric_value)

    if warnings:
        return out_row, "\n".join(warnings)

    return out_row, None


def find_profiler_output_dirs(config_dir: Path):
    """
    查找一个并行配置目录下所有 ASCEND_PROFILER_OUTPUT 目录。

    标准结构：
      config_dir/
        worker-xxx/
          profiler_info_0.json
          ASCEND_PROFILER_OUTPUT/
            step_trace_time.csv
    """
    profiler_output_dirs = []

    for worker_dir in config_dir.iterdir():
        if not worker_dir.is_dir():
            continue

        profiler_output_dir = worker_dir / "ASCEND_PROFILER_OUTPUT"

        if profiler_output_dir.is_dir():
            profiler_output_dirs.append(profiler_output_dir)

    return sorted(profiler_output_dirs)


def collect_one_model(model_dir: Path):
    model_name = model_dir.name

    rows = []
    warnings = []

    config_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])

    for config_dir in config_dirs:
        optimization_config = config_dir.name

        profiler_output_dirs = find_profiler_output_dirs(config_dir)

        if not profiler_output_dirs:
            warnings.append(
                f"[MISS] {model_name}/{optimization_config}: no ASCEND_PROFILER_OUTPUT"
            )
            continue

        for profiler_output_dir in profiler_output_dirs:
            row, warning = read_step_trace_time(profiler_output_dir)

            if warning is not None:
                for line in str(warning).splitlines():
                    warnings.append(line)

            if row is None:
                continue

            final_row = {
                "optimization_config": optimization_config,
                "device_type": row["device_type"],
            }

            for col in TIME_COLUMNS:
                final_row[col] = row[col]

            final_row["source_dir"] = str(profiler_output_dir)

            rows.append(final_row)

    return rows, warnings


def write_model_result(output_root: Path, model_name: str, rows: list, warnings: list):
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    warning_log = output_dir / WARNING_LOG_NAME

    output_columns = [
        "optimization_config",
        "device_type",
        *TIME_COLUMNS,
    ]

    if rows:
        df = pd.DataFrame(rows)

        # 如果同一个 optimization_config + device_type 出现多次，
        # 保留最后一条。
        df = (
            df.sort_values(by=["optimization_config", "device_type", "source_dir"])
              .drop_duplicates(
                  subset=["optimization_config", "device_type"],
                  keep="last",
              )
        )

        result_df = df[output_columns].sort_values(
            by=["optimization_config", "device_type"]
        )

    else:
        result_df = pd.DataFrame(columns=output_columns)

    result_df.to_csv(output_csv, index=False, encoding="utf-8")

    if warnings:
        with warning_log.open("w", encoding="utf-8") as f:
            for line in warnings:
                f.write(line + "\n")

    return output_csv, len(result_df), warning_log if warnings else None


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    print(f"[INFO] Input root : {input_root}")
    print(f"[INFO] Output root: {output_root}")

    model_dirs = get_model_dirs(
        input_root=input_root,
        models=args.models,
        exclude_models=set(args.exclude_models),
    )

    total_models = 0
    total_rows = 0

    for model_dir in model_dirs:
        model_name = model_dir.name

        print(f"\n[INFO] Processing model: {model_name}")

        rows, warnings = collect_one_model(model_dir)

        output_csv, row_count, warning_log = write_model_result(
            output_root,
            model_name,
            rows,
            warnings,
        )

        total_models += 1
        total_rows += row_count

        print(f"[OK] Wrote: {output_csv}")
        print(f"[OK] Rows : {row_count}")

        if warning_log is not None:
            print(f"[WARN] See warning log: {warning_log}")

    print("\n[DONE]")
    print(f"[DONE] Models processed: {total_models}")
    print(f"[DONE] Total rows      : {total_rows}")


if __name__ == "__main__":
    main()