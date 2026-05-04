#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import pandas as pd


DEFAULT_BASE_ROOT = Path("LM-SimBench")


INPUT_FILE_NAME = "step_trace_time_summary.csv"
OUTPUT_FILE_NAME = "step_time_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-configuration iteration time summaries for all model folders."
    )
    parser.add_argument(
        "--base-root",
        type=Path,
        default=DEFAULT_BASE_ROOT,
        help=f"包含各模型最终结果目录的根目录，默认: {DEFAULT_BASE_ROOT}",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="可选：只处理指定模型；不传则遍历根目录下的所有一级模型目录。",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="可选：遍历所有模型时需要排除的一级目录名。",
    )
    return parser.parse_args()


def get_model_dirs(base_root: Path, models: list[str] | None, exclude_models: set[str]) -> list[Path]:
    available_model_dirs = {
        p.name: p for p in base_root.iterdir()
        if p.is_dir()
    }

    if models is None:
        return sorted(
            p for name, p in available_model_dirs.items()
            if name not in exclude_models
        )

    missing_models = sorted(set(models) - set(available_model_dirs))
    if missing_models:
        print("[WARN] 以下模型在根目录中不存在，将跳过:")
        for name in missing_models:
            print(f"  - {name}")

    return sorted(
        available_model_dirs[name]
        for name in set(models) & set(available_model_dirs)
        if name not in exclude_models
    )


def resolve_column(df: pd.DataFrame, column_name: str) -> str | None:
    normalized_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }
    return normalized_columns.get(column_name.lower())


def process_one_model(model_dir: Path) -> None:
    input_csv = model_dir / INPUT_FILE_NAME
    output_csv = model_dir / OUTPUT_FILE_NAME

    if not input_csv.exists():
        print(f"[SKIP] missing {INPUT_FILE_NAME}: {model_dir}")
        return

    df = pd.read_csv(input_csv)

    optimization_config_col = resolve_column(df, "optimization_config")
    stage_col = resolve_column(df, "stage")
    bubble_col = resolve_column(df, "bubble")

    required_cols = {
        "optimization_config": optimization_config_col,
        "stage": stage_col,
        "bubble": bubble_col,
    }
    missing_cols = [
        name for name, actual_col in required_cols.items()
        if actual_col is None
    ]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {input_csv}: {missing_cols}"
        )

    # 转成数值，避免字符串导致计算错误
    df["_stage_us"] = pd.to_numeric(df[stage_col], errors="coerce")
    df["_bubble_us"] = pd.to_numeric(df[bubble_col], errors="coerce")

    # 单个 rank 的 profile 级迭代时间，单位 us
    df["iteration_time_us"] = df["_stage_us"] + df["_bubble_us"]

    # 单个配置的总单次迭代时间：
    # 同步训练下通常取所有 rank 中的最大值
    summary = (
        df.groupby(optimization_config_col, as_index=False)["iteration_time_us"]
        .max()
        .rename(
            columns={
                optimization_config_col: "optimization_config",
                "iteration_time_us": "avg_elapsed_time_per_iteration_ms",
            }
        )
    )

    # us -> ms
    summary["avg_elapsed_time_per_iteration_ms"] = (
        summary["avg_elapsed_time_per_iteration_ms"] / 1000.0
    )

    # 保留一位小数，和你给的格式一致
    summary["avg_elapsed_time_per_iteration_ms"] = (
        summary["avg_elapsed_time_per_iteration_ms"].round(1)
    )

    summary = summary[
        [
            "optimization_config",
            "avg_elapsed_time_per_iteration_ms",
        ]
    ]

    summary.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(
        f"[OK] {model_dir.name}: "
        f"configs={len(summary)}, output={output_csv}"
    )


def main() -> None:
    args = parse_args()
    base_root = args.base_root.expanduser().resolve()

    if not base_root.exists():
        raise FileNotFoundError(f"BASE_ROOT not found: {base_root}")

    model_dirs = get_model_dirs(
        base_root=base_root,
        models=args.models,
        exclude_models=set(args.exclude_models),
    )

    print(f"[INFO] BASE_ROOT = {base_root}")
    print(f"[INFO] model dirs = {len(model_dirs)}")
    print("=" * 80)

    success = 0
    failed = 0
    skipped = 0

    for model_dir in model_dirs:
        input_csv = model_dir / INPUT_FILE_NAME

        if not input_csv.exists():
            print(f"[SKIP] {model_dir.name}: no {INPUT_FILE_NAME}")
            skipped += 1
            continue

        try:
            process_one_model(model_dir)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] {model_dir.name}: {e}")

    print("=" * 80)
    print("[DONE] all model folders processed")
    print(f"[SUMMARY] success: {success}")
    print(f"[SUMMARY] skipped: {skipped}")
    print(f"[SUMMARY] failed: {failed}")


if __name__ == "__main__":
    main()