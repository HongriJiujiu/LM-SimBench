#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import json
import pandas as pd


DEFAULT_INPUT_ROOT = Path("raw_filtered")
DEFAULT_OUTPUT_ROOT = Path("LM-SimBench")

OUTPUT_CSV_NAME = "peak_memory.csv"
WARNING_LOG_NAME = "peak_memory_warnings.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract peak memory by model, configuration, and rank."
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


def normalize_col_name(col: str) -> str:
    if col is None:
        return ""

    col = str(col).strip().lstrip("\ufeff").strip('"').strip("'").lower()
    return re.sub(r"[^a-z0-9]+", "", col)


def extract_rank_id_from_obj(obj):
    """
    从 JSON 对象中递归查找 rank_id。

    兼容：
    1. rank_id
    2. rankId
    3. rank
    4. 嵌套字段中的 rank_id
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            norm_key = normalize_col_name(key)

            if norm_key in {"rankid", "rank"}:
                return value

        for value in obj.values():
            found = extract_rank_id_from_obj(value)
            if found is not None:
                return found

    elif isinstance(obj, list):
        for item in obj:
            found = extract_rank_id_from_obj(item)
            if found is not None:
                return found

    return None


def read_rank_id_from_profiler_info(worker_dir: Path):
    """
    从 worker 目录下的 profiler_info*.json 中读取 rank_id。

    不写死 profiler_info_0.json，因为文件名中的数字可能随卡号变化。
    """
    profiler_info_files = sorted(worker_dir.glob("profiler_info*.json"))

    if not profiler_info_files:
        return None, f"[MISS] no profiler_info*.json found: {worker_dir}"

    rank_ids = []

    for info_file in profiler_info_files:
        try:
            with info_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            return None, f"[ERROR] failed to read profiler_info json: {info_file}, reason: {repr(e)}"

        rank_id = extract_rank_id_from_obj(data)

        if rank_id is not None:
            rank_ids.append(rank_id)

    if not rank_ids:
        return None, f"[WARN] rank_id not found in profiler_info*.json under: {worker_dir}"

    unique_rank_ids = sorted(set(str(x).strip() for x in rank_ids if str(x).strip()))

    if len(unique_rank_ids) > 1:
        return None, (
            f"[WARN] multiple rank_id values found under {worker_dir}: "
            f"{unique_rank_ids}"
        )

    rank_id = unique_rank_ids[0]

    try:
        rank_id = int(rank_id)
    except Exception:
        pass

    return rank_id, None


def read_peak_memory_with_pandas(memory_csv: Path):
    """
    读取单个 memory_record.csv。

    当前逻辑：
    1. 只读取 Total Allocated(MB) 列；
    2. 取 Total Allocated(MB) 最大值作为该 rank 的峰值显存。
    """
    try:
        df = pd.read_csv(
            memory_csv,
            sep=",",
            encoding="utf-8-sig",
            usecols=["Total Allocated(MB)"],
        )
    except Exception as e:
        return None, f"[ERROR] Failed to read {memory_csv}: {repr(e)}"

    if df.empty:
        return None, f"[WARN] Empty csv: {memory_csv}"

    allocated_series = pd.to_numeric(
        df["Total Allocated(MB)"],
        errors="coerce"
    )

    if allocated_series.isna().all():
        return None, f"[WARN] All Total Allocated(MB) values are invalid: {memory_csv}"

    peak_memory_mb = float(allocated_series.max())

    return peak_memory_mb, None


def find_worker_memory_files(config_dir: Path):
    """
    查找一个并行配置目录下每个 worker 对应的 memory_record.csv。

    标准结构：
      config_dir / worker-xxx / profiler_info_*.json
      config_dir / worker-xxx / ASCEND_PROFILER_OUTPUT / memory_record.csv
    """
    results = []

    for worker_dir in config_dir.iterdir():
        if not worker_dir.is_dir():
            continue

        memory_csv = worker_dir / "ASCEND_PROFILER_OUTPUT" / "memory_record.csv"

        if memory_csv.is_file():
            results.append((worker_dir, memory_csv))

    return sorted(results, key=lambda x: str(x[0]))


def collect_one_model(model_dir: Path):
    model_name = model_dir.name

    rows = []
    warnings = []

    config_dirs = sorted([p for p in model_dir.iterdir() if p.is_dir()])

    for config_dir in config_dirs:
        optimization_config = config_dir.name

        worker_memory_files = find_worker_memory_files(config_dir)

        if not worker_memory_files:
            warnings.append(
                f"[MISS] {model_name}/{optimization_config}: no memory_record.csv"
            )
            continue

        for worker_dir, memory_csv in worker_memory_files:
            rank_id, rank_warning = read_rank_id_from_profiler_info(worker_dir)

            if rank_warning is not None:
                warnings.append(rank_warning)

            if rank_id is None:
                warnings.append(
                    f"[SKIP] skip memory file because rank_id is unavailable: {memory_csv}"
                )
                continue

            peak_memory_mb, memory_warning = read_peak_memory_with_pandas(memory_csv)

            if memory_warning is not None:
                warnings.append(memory_warning)

            if peak_memory_mb is None:
                continue

            rows.append({
                "optimization_config": optimization_config,
                "rank_id": rank_id,
                "peak_memory_mb": peak_memory_mb,
                "source_file": str(memory_csv),
            })

    return rows, warnings


def write_model_result(output_root: Path, model_name: str, rows: list, warnings: list):
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / OUTPUT_CSV_NAME
    warning_log = output_dir / WARNING_LOG_NAME

    if rows:
        df = pd.DataFrame(rows)

        result_df = (
            df.groupby(
                ["optimization_config", "rank_id"],
                as_index=False
            )["peak_memory_mb"]
            .max()
        )

        result_df = result_df.sort_values(
            by=["optimization_config", "rank_id"]
        )

    else:
        result_df = pd.DataFrame(
            columns=[
                "optimization_config",
                "rank_id",
                "peak_memory_mb",
            ]
        )

    result_df.to_csv(output_csv, index=False, encoding="utf-8")

    if warnings:
        with warning_log.open("w", encoding="utf-8") as f:
            for line in warnings:
                f.write(line + "\n")

    return output_csv, len(result_df), warning_log if warnings else None


def get_target_model_dirs(input_root: Path, target_model_names: list[str] | None, exclude_model_names: set[str]):
    """
    默认返回输入目录下所有模型目录；传入 target_model_names 时只返回指定模型目录。
    """
    available_model_dirs = {
        p.name: p for p in input_root.iterdir()
        if p.is_dir()
    }

    if target_model_names is None:
        model_dirs = [
            p for name, p in available_model_dirs.items()
            if name not in exclude_model_names
        ]
        return sorted(model_dirs), []

    target_model_set = set(target_model_names)
    missing_models = sorted(target_model_set - set(available_model_dirs))
    model_dirs = [
        available_model_dirs[name]
        for name in target_model_set & set(available_model_dirs)
        if name not in exclude_model_names
    ]

    return sorted(model_dirs), missing_models


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    print(f"[INFO] Input root : {input_root}")
    print(f"[INFO] Output root: {output_root}")

    model_dirs, missing_models = get_target_model_dirs(
        input_root=input_root,
        target_model_names=args.models,
        exclude_model_names=set(args.exclude_models),
    )

    if missing_models:
        print("\n[WARN] Missing target model dirs:")
        for model_name in missing_models:
            print(f"       - {model_name}")

    if not model_dirs:
        raise RuntimeError("No target model directories found.")

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