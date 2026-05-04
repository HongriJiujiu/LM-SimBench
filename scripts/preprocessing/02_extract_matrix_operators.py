#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import re
from typing import Optional

import pandas as pd


DEFAULT_SRC_ROOT = Path("raw_filtered")
DEFAULT_DST_ROOT = Path("LM-SimBench")

MATRIX_TYPE_PATTERN = r"(?:matmul|batchmatmul|gemm|batchgemm|bmm)"


DROP_COLUMNS = [
    "Step Id",
    "Model ID",
    "OP State",
    "Context ID",
    "Input Formats",
    "Output Formats",
    "HF32 Eligible",
    "Mix Block Dim",
    "Start Time(us)",

    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_icache_miss_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract matrix operator latency details for all model folders."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help=f"过滤后的 profiler 数据根目录，默认: {DEFAULT_SRC_ROOT}",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help=f"最终结果根目录，默认: {DEFAULT_DST_ROOT}",
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


def get_model_dirs(src_root: Path, models: list[str] | None, exclude_models: set[str]) -> list[Path]:
    available_model_dirs = {
        p.name: p for p in src_root.iterdir()
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


def to_snake_case(col: str) -> str:
    col = col.strip()

    col = col.replace("(GB/s)", "_gbps")
    col = col.replace("(us)", "_us")
    col = col.replace("(ms)", "_ms")
    col = col.replace("(MB)", "_mb")
    col = col.replace("(KB)", "_kb")
    col = col.replace("(%)", "_percent")

    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_").lower()

    return col


def parse_shape_list(shape_str):
    if pd.isna(shape_str):
        return []

    s = str(shape_str).strip().strip('"').strip("'")
    if not s:
        return []

    shapes = []
    for part in s.split(";"):
        part = part.strip().strip('"').strip("'")
        if not part:
            continue

        dims = []
        for x in part.split(","):
            x = x.strip()
            if x == "":
                continue
            try:
                dims.append(int(x))
            except ValueError:
                return []

        if dims:
            shapes.append(tuple(dims))

    return shapes


def shape_to_str(shape) -> str:
    if not shape:
        return ""
    return "x".join(str(x) for x in shape)


def infer_matmul_dims(input_shapes, output_shapes):
    """
    从 A/B/C shape 推断 batch, m, n, k。

    这里只解析维度，不计算 FLOPs/TFLOPS。
    """
    result = {
        "a_shape": "",
        "b_shape": "",
        "c_shape": "",
        "batch": None,
        "m": None,
        "n": None,
        "k": None,
        "shape_signature": "",
    }

    if len(input_shapes) < 2 or len(output_shapes) < 1:
        return result

    a = input_shapes[0]
    b = input_shapes[1]
    c = output_shapes[0]

    result["a_shape"] = shape_to_str(a)
    result["b_shape"] = shape_to_str(b)
    result["c_shape"] = shape_to_str(c)
    result["shape_signature"] = (
        f"{shape_to_str(a)};{shape_to_str(b)}->{shape_to_str(c)}"
    )

    # 2D MatMul
    if len(a) == 2 and len(b) == 2 and len(c) == 2:
        c0, c1 = c[0], c[1]

        # A = M x K, B = N x K, C = M x N
        if a[0] == c0 and b[0] == c1 and a[1] == b[1]:
            result["batch"] = 1
            result["m"] = c0
            result["n"] = c1
            result["k"] = a[1]
            return result

        # A = M x K, B = K x N, C = M x N
        if a[0] == c0 and b[1] == c1 and a[1] == b[0]:
            result["batch"] = 1
            result["m"] = c0
            result["n"] = c1
            result["k"] = a[1]
            return result

        # A = M x N, B = M x K, C = N x K
        if a[0] == b[0] and a[1] == c[0] and b[1] == c[1]:
            result["batch"] = 1
            result["m"] = a[0]
            result["n"] = a[1]
            result["k"] = b[1]
            return result

        return result

    # 3D Batched MatMul
    if len(a) == 3 and len(b) == 3 and len(c) == 3:
        batch = c[0]
        c_m = c[1]
        c_n = c[2]

        # A = BATCH x M x K, B = BATCH x N x K, C = BATCH x M x N
        if (
            a[0] == batch
            and b[0] == batch
            and a[1] == c_m
            and b[1] == c_n
            and a[2] == b[2]
        ):
            result["batch"] = batch
            result["m"] = c_m
            result["n"] = c_n
            result["k"] = a[2]
            return result

        # A = BATCH x M x K, B = BATCH x K x N, C = BATCH x M x N
        if (
            a[0] == batch
            and b[0] == batch
            and a[1] == c_m
            and b[2] == c_n
            and a[2] == b[1]
        ):
            result["batch"] = batch
            result["m"] = c_m
            result["n"] = c_n
            result["k"] = a[2]
            return result

        return result

    return result


def read_rank_id(worker_dir: Path) -> Optional[str]:
    profiler_info_files = sorted(worker_dir.glob("profiler_info_*.json"))

    if not profiler_info_files:
        return None

    for json_file in profiler_info_files:
        try:
            with json_file.open("r", encoding="utf-8") as f:
                info = json.load(f)

            rank_id = info.get("rank_id")
            if rank_id is not None:
                return str(rank_id)

        except Exception as e:
            print(f"[WARN] failed to read rank_id from {json_file}: {e}")

    return None


def process_one_kernel_details(input_csv: Path, output_csv: Path) -> tuple[int, int]:
    """
    处理单个 kernel_details.csv。

    返回：
      total_rows, matrix_rows
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if "Type" not in df.columns:
        raise ValueError(f"'Type' column not found in {input_csv}")

    total_rows = len(df)

    # 1. 筛选矩阵算子
    df = df[
        df["Type"].astype(str).str.contains(
            MATRIX_TYPE_PATTERN,
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    matrix_rows = len(df)

    # 2. 删除低价值字段
    existing_drop_cols = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=existing_drop_cols)

    # 3. 字段名转 snake_case
    df = df.rename(columns={c: to_snake_case(c) for c in df.columns})

    # 4. 解析 shape
    if matrix_rows > 0:
        if "input_shapes" not in df.columns:
            raise ValueError(f"'input_shapes' column not found after renaming: {input_csv}")

        if "output_shapes" not in df.columns:
            raise ValueError(f"'output_shapes' column not found after renaming: {input_csv}")

        parsed_rows = []
        for _, row in df.iterrows():
            input_shapes = parse_shape_list(row["input_shapes"])
            output_shapes = parse_shape_list(row["output_shapes"])
            parsed_rows.append(infer_matmul_dims(input_shapes, output_shapes))

        parsed_df = pd.DataFrame(parsed_rows)
        df = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)
    else:
        for col in [
            "a_shape",
            "b_shape",
            "c_shape",
            "batch",
            "m",
            "n",
            "k",
            "shape_signature",
        ]:
            df[col] = pd.Series(dtype="object")

    # 5. 调整字段顺序
    preferred_cols = [
        "task_id",
        "stream_id",
        "name",
        "type",
        "accelerator_core",
        "start_time_us",
        "duration_us",
        "wait_time_us",
        "block_dim",

        "input_shapes",
        "input_data_types",
        "output_shapes",
        "output_data_types",

        "a_shape",
        "b_shape",
        "c_shape",
        "batch",
        "m",
        "n",
        "k",
        "shape_signature",

        "aicore_time_us",
        "aic_total_cycles",
        "aic_mac_time_us",
        "aic_mac_ratio",
        "aic_scalar_time_us",
        "aic_scalar_ratio",
        "aic_mte1_time_us",
        "aic_mte1_ratio",
        "aic_mte2_time_us",
        "aic_mte2_ratio",
        "aic_fixpipe_time_us",
        "aic_fixpipe_ratio",
        "aic_icache_miss_rate",
        "cube_utilization_percent",
    ]

    ordered_cols = [c for c in preferred_cols if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + remaining_cols]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    return total_rows, matrix_rows


def process_all(src_root: Path, dst_root: Path, models: list[str] | None, exclude_models: set[str]) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    success_files = 0
    failed_files = 0
    total_rows_all = 0
    matrix_rows_all = 0

    for model_dir in get_model_dirs(src_root, models, exclude_models):
        model_name = model_dir.name
        print(f"[MODEL] {model_name}")

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            for worker_dir in sorted(config_dir.iterdir()):
                if not worker_dir.is_dir():
                    continue

                ascend_dir = worker_dir / "ASCEND_PROFILER_OUTPUT"
                input_csv = ascend_dir / "kernel_details.csv"

                if not input_csv.exists():
                    continue

                # 仍然读取 rank_id 进行完整性检查，但不写入输出 CSV
                rank_id = read_rank_id(worker_dir)
                if rank_id is None:
                    print(f"[WARN] rank_id not found: {worker_dir}")

                rel_worker_path = worker_dir.relative_to(src_root)
                output_csv = (
                    dst_root
                    / rel_worker_path
                    / "ASCEND_PROFILER_OUTPUT"
                    / "matrix_operator_latency.csv"
                )

                total_files += 1

                try:
                    total_rows, matrix_rows = process_one_kernel_details(
                        input_csv=input_csv,
                        output_csv=output_csv,
                    )

                    total_rows_all += total_rows
                    matrix_rows_all += matrix_rows
                    success_files += 1

                    print(
                        f"[OK] {rel_worker_path} "
                        f"total_rows={total_rows}, matrix_rows={matrix_rows}"
                    )

                except Exception as e:
                    failed_files += 1
                    print(
                        f"[ERROR] failed: {input_csv}\n"
                        f"        reason: {e}"
                    )

    print("=" * 80)
    print("[DONE] all kernel_details.csv processed")
    print(f"[SUMMARY] total files found: {total_files}")
    print(f"[SUMMARY] success files: {success_files}")
    print(f"[SUMMARY] failed files: {failed_files}")
    print(f"[SUMMARY] total rows: {total_rows_all}")
    print(f"[SUMMARY] matrix rows: {matrix_rows_all}")
    print(f"[SUMMARY] output root: {dst_root}")


def main() -> None:
    args = parse_args()
    process_all(
        src_root=args.src_root.expanduser().resolve(),
        dst_root=args.dst_root.expanduser().resolve(),
        models=args.models,
        exclude_models=set(args.exclude_models),
    )


if __name__ == "__main__":
    main()