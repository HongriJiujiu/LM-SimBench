#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


DEFAULT_INPUT_ROOT = Path("raw_filtered")
DEFAULT_OUTPUT_ROOT = Path("LM-SimBench")

OUTPUT_FILENAME = "operator_latency_by_category.csv"

REQUIRED_COLUMNS = [
    "OP Type",
    "Core Type",
    "Count",
    "Total Time(us)",
    "Min Time(us)",
    "Avg Time(us)",
    "Max Time(us)",
    "Ratio(%)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate operator latency by category for all model folders."
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


def map_op_category(op_type: str) -> str:
    """
    将 Ascend / MindSpeed profiler 中的 OP Type 映射为研究友好的算子类别。

    该映射依据：
    1. 算子计算语义；
    2. Transformer 训练中的功能角色；
    3. operator-latency prediction 的建模需求。
    """
    if pd.isna(op_type):
        return "other"

    s = str(op_type).strip()
    lower = s.lower()

    # 1. Matrix multiplication
    if (
        "matmul" in lower
        or "batchmatmul" in lower
        or "gemm" in lower
        or "groupedmatmul" in lower
    ):
        return "matmul"

    # 2. Attention
    if (
        "flashattentionscore" in lower
        or lower in {"attention", "selfattention"}
        or "attention" in lower
    ):
        return "attention"

    # 3. Normalization
    if (
        "rmsnorm" in lower
        or "layernorm" in lower
        or "lpnorm" in lower
        or lower.endswith("norm")
        or "normalization" in lower
    ):
        return "normalization"

    # 4. Activation
    if (
        "swiglu" in lower
        or "gelu" in lower
        or "relu" in lower
        or "silu" in lower
        or "sigmoid" in lower
        or "tanh" in lower
    ):
        return "activation"

    # 5. Positional encoding
    if "rotarypositionembedding" in lower or "rope" in lower:
        return "positional_encoding"

    # 6. Embedding
    if "embeddingdensegrad" in lower or lower == "embedding" or lower.endswith("embedding"):
        return "embedding"

    # 7. Indexing / embedding lookup
    if lower in {"gatherv2", "gather"} or "gather" in lower:
        return "indexing_or_embedding"

    # 8. Elementwise
    elementwise_ops = {
        "add",
        "sub",
        "mul",
        "muls",
        "div",
        "realdiv",
        "pows",
        "pow",
        "greater",
        "less",
        "equal",
        "notequal",
        "cos",
        "sin",
        "exp",
        "log",
        "sqrt",
        "neg",
        "maximum",
        "minimum",
        "abs",
        "reciprocal",
    }
    if lower in elementwise_ops:
        return "elementwise"

    # 9. Shape transform
    shape_transform_ops = {
        "slice",
        "concatd",
        "concat",
        "cast",
        "reshape",
        "transpose",
        "permute",
        "view",
        "range",
        "sort",
        "nonzero",
        "squeeze",
        "unsqueeze",
        "expand",
        "broadcastto",
    }
    if lower in shape_transform_ops:
        return "shape_transform"

    if any(
        key in lower
        for key in [
            "slice",
            "concat",
            "cast",
            "reshape",
            "transpose",
            "permute",
            "nonzero",
            "broadcast",
        ]
    ):
        return "shape_transform"

    # 10. Memory movement
    if lower in {"tensormove", "memcopy", "memcpy"}:
        return "memory_movement"

    if "tensormove" in lower or "memcopy" in lower or "memcpy" in lower:
        return "memory_movement"

    # 11. Memory initialization
    if lower in {"memset", "zeroslike", "oneslike"}:
        return "memory_init"

    if "memset" in lower or "zeroslike" in lower or "oneslike" in lower:
        return "memory_init"

    # 12. Optimizer
    if (
        "applyadamw" in lower
        or lower in {"adam", "adamw", "fusedadam"}
        or "optimizer" in lower
        or "adam" in lower
    ):
        return "optimizer"

    # 13. Communication
    if (
        lower.startswith("hcom")
        or "allreduce" in lower
        or "reducescatter" in lower
        or "allgather" in lower
        or "broadcast" in lower
        or "send" in lower
        or "receive" in lower
        or "recv" in lower
    ):
        return "communication"

    return "other"


def read_rank_id(worker_dir: Path) -> Optional[str]:
    """
    从 worker 目录下的 profiler_info_*.json 读取 rank_id。

    例如：
    worker-xxx/
      profiler_info_0.json
      ASCEND_PROFILER_OUTPUT/
    """
    profiler_info_files = sorted(worker_dir.glob("profiler_info_*.json"))

    if not profiler_info_files:
        return None

    for info_file in profiler_info_files:
        try:
            with info_file.open("r", encoding="utf-8") as f:
                info = json.load(f)

            rank_id = info.get("rank_id")
            if rank_id is not None:
                return str(rank_id)

        except Exception as e:
            print(f"[WARN] failed to read profiler info: {info_file}, error={e}")

    return None


def process_op_statistic_file(
    op_file: Path,
    optimization_config: str,
    rank_id: str,
) -> Optional[pd.DataFrame]:
    """
    读取并处理单个 ASCEND_PROFILER_OUTPUT/op_statistic.csv。
    """
    try:
        df = pd.read_csv(op_file)
    except Exception as e:
        print(f"[WARN] failed to read op_statistic.csv: {op_file}, error={e}")
        return None

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"[WARN] missing columns in {op_file}: {missing_cols}")
        return None

    df = df[REQUIRED_COLUMNS].copy()

    df.insert(0, "op_category", df["OP Type"].apply(map_op_category))
    df.insert(0, "rank_id", rank_id)
    df.insert(0, "optimization_config", optimization_config)

    final_columns = [
        "optimization_config",
        "rank_id",
        "op_category",
        "OP Type",
        "Core Type",
        "Count",
        "Total Time(us)",
        "Min Time(us)",
        "Avg Time(us)",
        "Max Time(us)",
        "Ratio(%)",
    ]

    return df[final_columns]


def process_model(input_model_dir: Path, output_model_dir: Path) -> None:
    """
    处理一个模型目录。

    输入：
        raw_filtered/模型名称/

    输出：
        LM-SimBench/模型名称/operator_latency_by_category.csv
    """
    model_name = input_model_dir.name
    print(f"[MODEL] start: {model_name}")

    all_dfs = []

    for config_dir in sorted(input_model_dir.iterdir()):
        if not config_dir.is_dir():
            continue

        # 避免把 tables、metadata、其他后处理目录误当成并行配置
        optimization_config = config_dir.name
        if not optimization_config.startswith("dp"):
            print(f"[SKIP] not optimization config: {config_dir}")
            continue

        for worker_dir in sorted(config_dir.iterdir()):
            if not worker_dir.is_dir():
                continue

            op_file = worker_dir / "ASCEND_PROFILER_OUTPUT" / "op_statistic.csv"
            if not op_file.exists():
                print(f"[WARN] missing op_statistic.csv: {op_file}")
                continue

            rank_id = read_rank_id(worker_dir)
            if rank_id is None:
                print(f"[WARN] missing rank_id, skip worker: {worker_dir}")
                continue

            df = process_op_statistic_file(
                op_file=op_file,
                optimization_config=optimization_config,
                rank_id=rank_id,
            )

            if df is None or df.empty:
                continue

            all_dfs.append(df)

            print(
                f"[WORKER] {model_name}/{optimization_config}/{worker_dir.name} "
                f"rank_id={rank_id}, rows={len(df)}"
            )

    if not all_dfs:
        print(f"[WARN] no valid op_statistic data found for model: {model_name}")
        return

    result_df = pd.concat(all_dfs, ignore_index=True)

    # 排序：按配置、rank、类别、原始 OP Type 排列，便于阅读
    result_df["rank_id"] = pd.to_numeric(result_df["rank_id"], errors="coerce")
    result_df = result_df.sort_values(
        by=["optimization_config", "rank_id", "op_category", "OP Type"],
        ascending=[True, True, True, True],
    )

    # 输出目录：LM-SimBench/模型名称/
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_model_dir / OUTPUT_FILENAME

    result_df.to_csv(output_file, index=False)

    print(f"[MODEL] done: {model_name}")
    print(f"[OUTPUT] {output_file}")
    print(f"[ROWS] {len(result_df)}")
    print("-" * 80)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"raw_filtered directory not found: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    input_model_dirs = get_model_dirs(
        input_root=input_root,
        models=args.models,
        exclude_models=set(args.exclude_models),
    )

    print(f"[INFO] profiler input dir: {input_root}")
    print(f"[INFO] profiler output dir: {output_root}")
    print(f"[INFO] model count: {len(input_model_dirs)}")
    print("=" * 80)

    for input_model_dir in input_model_dirs:
        output_model_dir = output_root / input_model_dir.name

        process_model(
            input_model_dir=input_model_dir,
            output_model_dir=output_model_dir,
        )

    print("[DONE] all models processed")


if __name__ == "__main__":
    main()