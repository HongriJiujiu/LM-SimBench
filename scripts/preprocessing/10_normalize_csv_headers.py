#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import csv
import shutil
import tempfile
import re


DEFAULT_OUTPUT_ROOT = Path("LM-SimBench")

TARGET_FILES = {
    "operator_latency_by_category.csv",
    "step_trace_time_summary.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize selected final CSV headers to snake_case."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"最终结果根目录，默认: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser.parse_args()


COLUMN_MAP = {
    # operator_latency_by_category.csv
    "optimization_config": "optimization_config",
    "rank_id": "rank_id",
    "op_category": "op_category",
    "OP Type": "op_type",
    "Core Type": "core_type",
    "Count": "count",
    "Total Time(us)": "total_time_us",
    "Min Time(us)": "min_time_us",
    "Avg Time(us)": "avg_time_us",
    "Max Time(us)": "max_time_us",
    "Ratio(%)": "ratio_percent",

    # step_trace_time_summary.csv
    "Computing": "computing",
    "Communication(Not Overlapped)": "communication_not_overlapped",
    "Overlapped": "overlapped",
    "Communication": "communication",
    "Free": "free",
    "Stage": "stage",
    "Bubble": "bubble",
    "Communication(Not Overlapped and Exclude Receive)": (
        "communication_not_overlapped_and_exclude_receive"
    ),
}


def detect_delimiter(csv_path: Path) -> str:
    """
    自动检测分隔符，兼容逗号、tab、分号。
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


def to_snake_case(col: str) -> str:
    """
    对未显式列出的字段做兜底 snake_case 转换。
    """
    col = str(col).strip()

    if col in COLUMN_MAP:
        return COLUMN_MAP[col]

    col = col.replace("%", "percent")
    col = col.replace("(us)", "_us")
    col = col.replace("(%)", "_percent")
    col = re.sub(r"[()]", "_", col)
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = col.lower()
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")

    return col


def normalize_csv_header(csv_path: Path):
    """
    只修改 CSV 表头，不改动数据内容。
    """
    delimiter = detect_delimiter(csv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        print(f"[SKIP] empty csv: {csv_path}")
        return

    old_header = rows[0]
    new_header = [to_snake_case(col) for col in old_header]

    if old_header == new_header:
        print(f"[SKIP] already normalized: {csv_path}")
        return

    rows[0] = new_header

    tmp_fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=csv_path.name + ".",
        dir=str(csv_path.parent),
        text=True,
    )

    with open(tmp_fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(rows)

    shutil.move(tmp_name, csv_path)

    print(f"[OK] normalized header: {csv_path}")
    print(f"     old: {old_header}")
    print(f"     new: {new_header}")


def main():
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()

    if not output_root.exists():
        raise FileNotFoundError(f"output root not found: {output_root}")

    csv_files = sorted(
        p for p in output_root.rglob("*.csv")
        if p.name in TARGET_FILES
    )

    if not csv_files:
        print("[WARN] no target csv files found")
        return

    print(f"[INFO] found {len(csv_files)} target csv files")

    for csv_path in csv_files:
        normalize_csv_header(csv_path)

    print("[DONE] all target csv headers normalized.")


if __name__ == "__main__":
    main()