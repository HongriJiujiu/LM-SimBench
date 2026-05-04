#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import csv
import re
import shutil
import tempfile


DEFAULT_OUTPUT_ROOT = Path("LM-SimBench")
NON_MODEL_DIR_NAMES = {"warnings"}

SUMMARY_FILES = [
    "comm_summary.csv",
    "peak_memory.csv",
    "step_trace_time_summary.csv",
]

COMM_TIME_DROP_COLUMNS = {
    "model_name",
    "parallel_config",
    "dp",
    "tp",
    "pp",
    "cp",
    "recompute",
    "worker_dir",
    "rank_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize LM-SimBench CSV files and optionally flatten "
            "ASCEND_PROFILER_OUTPUT directories."
        )
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
        help="可选：只处理指定模型；不传则遍历最终结果目录下的所有一级模型目录。",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="可选：遍历所有模型时需要排除的一级目录名。",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只处理模型级 summary CSV，等价于原来的重跑版逻辑。",
    )
    return parser.parse_args()


def get_model_dirs(
    output_root: Path,
    models: list[str] | None,
    exclude_models: set[str],
) -> list[Path]:
    available_model_dirs = {
        p.name: p for p in output_root.iterdir()
        if p.is_dir()
    }

    if models is None:
        return sorted(
            p for name, p in available_model_dirs.items()
            if name not in exclude_models and name not in NON_MODEL_DIR_NAMES
        )

    missing_models = sorted(set(models) - set(available_model_dirs))
    if missing_models:
        print("[WARN] 以下模型在最终结果目录中不存在，将跳过:")
        for name in missing_models:
            print(f"  - {name}")

    return sorted(
        available_model_dirs[name]
        for name in set(models) & set(available_model_dirs)
        if name not in exclude_models and name not in NON_MODEL_DIR_NAMES
    )


def detect_delimiter(csv_path: Path) -> str:
    """
    自动检测 CSV 分隔符。
    兼容逗号、制表符、分号。
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)

    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return ","


def extract_rank_id(value) -> str:
    """
    将 NPU:0 / NPU: 0 转换为 0。
    如果已经是 0，则保持不变。
    """
    if value is None:
        return ""

    value = str(value).strip()

    match = re.search(r"NPU\s*:\s*(\d+)", value, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return value


def rewrite_csv(csv_path: Path, new_fieldnames, rows, delimiter: str):
    """
    将修改后的 CSV 内容写回原文件。
    使用临时文件，写完后替换原文件。
    """
    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=csv_path.name + ".",
        dir=str(csv_path.parent),
        text=True,
    )

    with open(tmp_fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=new_fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    shutil.move(tmp_name, csv_path)


def convert_device_type_to_rank_id(csv_path: Path):
    """
    处理 comm_summary.csv / peak_memory.csv / step_trace_time_summary.csv。

    目标：
    device_type
    NPU:0

    改为：
    rank_id
    0
    """
    delimiter = detect_delimiter(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames

        if not fieldnames:
            print(f"[SKIP] empty csv: {csv_path}")
            return

        fieldnames = list(fieldnames)
        rows = list(reader)

    # 情况 1：已经处理过，只有 rank_id，没有 device_type
    if "device_type" not in fieldnames and "rank_id" in fieldnames:
        for row in rows:
            row["rank_id"] = extract_rank_id(row.get("rank_id", ""))
        rewrite_csv(csv_path, fieldnames, rows, delimiter)
        print(f"[OK] rank_id already exists, normalized: {csv_path}")
        return

    # 情况 2：没有 device_type，也没有 rank_id
    if "device_type" not in fieldnames:
        print(f"[SKIP] no device_type column: {csv_path}")
        return

    new_fieldnames = []

    for col in fieldnames:
        if col == "device_type":
            if "rank_id" not in new_fieldnames:
                new_fieldnames.append("rank_id")
        elif col == "rank_id":
            # 避免重复 rank_id 列
            continue
        else:
            new_fieldnames.append(col)

    new_rows = []

    for row in rows:
        new_row = {}

        for col in fieldnames:
            if col == "device_type":
                new_row["rank_id"] = extract_rank_id(row.get("device_type", ""))
            elif col == "rank_id":
                continue
            else:
                new_row[col] = row.get(col, "")

        new_rows.append(new_row)

    rewrite_csv(csv_path, new_fieldnames, new_rows, delimiter)
    print(f"[OK] device_type -> rank_id: {csv_path}")


def drop_comm_time_columns(csv_path: Path):
    """
    删除 comm_time.csv 中的以下列：

    model_name, parallel_config, dp, tp, pp, cp, recompute, worker_dir, rank_id
    """
    delimiter = detect_delimiter(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames

        if not fieldnames:
            print(f"[SKIP] empty csv: {csv_path}")
            return

        fieldnames = list(fieldnames)
        rows = list(reader)

    new_fieldnames = [
        col for col in fieldnames
        if col not in COMM_TIME_DROP_COLUMNS
    ]

    if new_fieldnames == fieldnames:
        print(f"[OK] no columns need dropping: {csv_path}")
        return

    new_rows = []

    for row in rows:
        new_row = {
            col: row.get(col, "")
            for col in new_fieldnames
        }
        new_rows.append(new_row)

    rewrite_csv(csv_path, new_fieldnames, new_rows, delimiter)
    print(f"[OK] dropped redundant columns: {csv_path}")


def move_ascend_output_up(ascend_dir: Path):
    """
    将：

    模型名称/并行配置/卡号/ASCEND_PROFILER_OUTPUT/文件

    上移为：

    模型名称/并行配置/卡号/文件

    然后删除 ASCEND_PROFILER_OUTPUT 文件夹。
    """
    parent_dir = ascend_dir.parent

    for item in sorted(ascend_dir.iterdir()):
        target = parent_dir / item.name

        if target.exists():
            raise FileExistsError(
                f"target already exists, stop to avoid overwrite: {target}"
            )

        shutil.move(str(item), str(target))
        print(f"[MOVE] {item} -> {target}")

    shutil.rmtree(ascend_dir)
    print(f"[OK] removed: {ascend_dir}")


def process_model_summary_files(model_dirs: list[Path]):
    """
    处理：

    LM-SimBench/模型名称/comm_summary.csv
    LM-SimBench/模型名称/peak_memory.csv
    LM-SimBench/模型名称/step_trace_time_summary.csv
    """
    for model_dir in model_dirs:
        for filename in SUMMARY_FILES:
            csv_path = model_dir / filename

            if csv_path.exists():
                convert_device_type_to_rank_id(csv_path)
            else:
                print(f"[SKIP] missing summary file: {csv_path}")


def process_comm_time_inside_ascend_dirs(model_dirs: list[Path]):
    """
    处理仍然位于 ASCEND_PROFILER_OUTPUT 下的 comm_time.csv，
    然后上移 ASCEND_PROFILER_OUTPUT 下全部文件。
    """
    ascend_dirs = []
    for model_dir in model_dirs:
        ascend_dirs.extend(
            p for p in model_dir.glob("*/*/ASCEND_PROFILER_OUTPUT")
            if p.is_dir()
        )
    ascend_dirs = sorted(ascend_dirs)

    if not ascend_dirs:
        print("[INFO] no ASCEND_PROFILER_OUTPUT dirs found under model/config/rank_id")
        return

    for ascend_dir in ascend_dirs:
        comm_time_path = ascend_dir / "comm_time.csv"

        if comm_time_path.exists():
            drop_comm_time_columns(comm_time_path)
        else:
            print(f"[SKIP] missing comm_time.csv: {comm_time_path}")

        move_ascend_output_up(ascend_dir)


def process_already_moved_comm_time_files(model_dirs: list[Path]):
    """
    兼容你已经运行过上一版代码的情况。

    如果 comm_time.csv 已经被上移到：

    LM-SimBench/模型名称/并行配置/卡号/comm_time.csv

    这里会再次检查并删除冗余列。
    """
    comm_time_files = []
    for model_dir in model_dirs:
        comm_time_files.extend(
            p for p in model_dir.glob("*/*/comm_time.csv")
            if p.is_file()
        )
    comm_time_files = sorted(comm_time_files)

    if not comm_time_files:
        print("[INFO] no already moved comm_time.csv files found")
        return

    for csv_path in comm_time_files:
        drop_comm_time_columns(csv_path)


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    if not output_root.exists():
        raise FileNotFoundError(f"not found: {output_root}")

    model_dirs = get_model_dirs(
        output_root=output_root,
        models=args.models,
        exclude_models=set(args.exclude_models),
    )

    print(f"[INFO] output root: {output_root}")
    print(f"[INFO] model dirs: {len(model_dirs)}")
    print(f"[INFO] summary only: {args.summary_only}")

    process_model_summary_files(model_dirs)

    if args.summary_only:
        print("[DONE] summary files processed.")
        return

    # 先处理还在 ASCEND_PROFILER_OUTPUT 里面的 comm_time.csv
    process_comm_time_inside_ascend_dirs(model_dirs)

    # 再处理已经被上移过的 comm_time.csv，兼容重复运行
    process_already_moved_comm_time_files(model_dirs)

    print("[DONE] all tasks finished.")


if __name__ == "__main__":
    main()