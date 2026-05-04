#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
from typing import Set


DEFAULT_SRC_ROOT = Path("profiler_raw")
DEFAULT_DST_ROOT = Path("raw_filtered")


EXCLUDE_REL_PATHS = {
    Path("FRAMEWORK/torch.op_range"),
    Path("FRAMEWORK/torch.python_func_call"),
    Path("FRAMEWORK/torch.python_module_call"),
    Path("FRAMEWORK/torch.memory_usage"),
    Path("FRAMEWORK/torch.op_mark"),
    Path("ASCEND_PROFILER_OUTPUT/trace_view.json"),
    Path("ASCEND_PROFILER_OUTPUT/operator_details.csv"),
}


def should_exclude(file_path: Path, worker_dir: Path, exclude_rel_paths: Set[Path]) -> bool:
    rel_path = file_path.relative_to(worker_dir)
    return rel_path in exclude_rel_paths


def is_worker_dir(path: Path) -> bool:
    """
    worker 目录下面通常应包含 FRAMEWORK 或 ASCEND_PROFILER_OUTPUT。
    用这个判断可以避免把 ASCEND_PROFILER_OUTPUT 自己误当成 worker。
    """
    if not path.is_dir():
        return False

    return (
        (path / "FRAMEWORK").is_dir()
        or (path / "ASCEND_PROFILER_OUTPUT").is_dir()
    )


def copy_worker_dir(src_worker_dir: Path, dst_worker_dir: Path) -> tuple[int, int]:
    copied_files = 0
    skipped_files = 0

    for item in src_worker_dir.rglob("*"):
        if item.is_dir():
            continue

        if should_exclude(item, src_worker_dir, EXCLUDE_REL_PATHS):
            skipped_files += 1
            continue

        rel_path = item.relative_to(src_worker_dir)
        dst_file = dst_worker_dir / rel_path

        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dst_file)
        copied_files += 1

    return copied_files, skipped_files


def discover_model_names(src_root: Path, excluded_names: Set[str]) -> list[str]:
    return sorted(
        p.name for p in src_root.iterdir()
        if p.is_dir() and p.name not in excluded_names
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy raw profiler data while excluding large or sensitive detail files."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=DEFAULT_SRC_ROOT,
        help=f"原始 profiler 数据根目录，默认: {DEFAULT_SRC_ROOT}",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=DEFAULT_DST_ROOT,
        help=f"过滤后输出根目录，默认: {DEFAULT_DST_ROOT}",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="可选：只复制指定模型；不传则遍历源目录下的所有一级模型目录。",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=[],
        help="可选：遍历所有模型时需要排除的一级目录名。",
    )
    return parser.parse_args()


def copy_model_dir(src_root: Path, dst_root: Path, model_name: str) -> tuple[int, int, int]:
    src_model_dir = src_root / model_name
    dst_model_dir = dst_root / model_name

    if not src_model_dir.exists():
        print(f"[WARN] 模型目录不存在，跳过: {src_model_dir}")
        return 0, 0, 0

    copied_files_total = 0
    skipped_files_total = 0
    copied_workers_total = 0

    for config_dir in sorted(src_model_dir.iterdir()):
        if not config_dir.is_dir():
            # 模型目录下的普通文件，例如 directory_listing.txt，可按需复制
            dst_file = dst_model_dir / config_dir.name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_dir, dst_file)
            copied_files_total += 1
            continue

        # 严格认为 config_dir 是配置目录，其下一层才是 worker 目录
        for worker_dir in sorted(config_dir.iterdir()):
            if not is_worker_dir(worker_dir):
                print(f"[SKIP] 非 worker 目录，跳过: {worker_dir}")
                continue

            dst_worker_dir = dst_model_dir / config_dir.name / worker_dir.name

            copied_files, skipped_files = copy_worker_dir(worker_dir, dst_worker_dir)

            copied_files_total += copied_files
            skipped_files_total += skipped_files
            copied_workers_total += 1

            print(
                f"[WORKER] {model_name}/{config_dir.name}/{worker_dir.name} "
                f"copied={copied_files}, skipped={skipped_files}"
            )

    return copied_files_total, skipped_files_total, copied_workers_total


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"源目录不存在: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    excluded_model_names = set(args.exclude_models)
    if args.models is None:
        selected_model_names = discover_model_names(src_root, excluded_model_names)
        missing_model_names: list[str] = []
    else:
        requested_model_names = set(args.models)
        available_model_dirs = {
            p.name for p in src_root.iterdir()
            if p.is_dir()
        }
        selected_model_names = sorted(requested_model_names & available_model_dirs)
        missing_model_names = sorted(requested_model_names - available_model_dirs)

    print(f"[INFO] SRC_ROOT = {src_root}")
    print(f"[INFO] DST_ROOT = {dst_root}")
    print(f"[INFO] selected models = {len(selected_model_names)}")
    print(f"[INFO] missing models = {len(missing_model_names)}")
    print(f"[INFO] excluded first-level dirs = {len(excluded_model_names)}")
    print("=" * 80)

    if missing_model_names:
        print("[WARN] 以下模型在源目录中不存在，将跳过:")
        for name in missing_model_names:
            print(f"  - {name}")
        print("-" * 80)

    if excluded_model_names:
        print("[INFO] 以下第一层目录被手动排除:")
        for name in sorted(excluded_model_names):
            print(f"  - {name}")
        print("-" * 80)

    total_copied_files = 0
    total_skipped_files = 0
    total_workers = 0

    for model_name in selected_model_names:
        print(f"[MODEL] start copying: {model_name}")

        copied_files, skipped_files, copied_workers = copy_model_dir(
            src_root,
            dst_root,
            model_name,
        )

        total_copied_files += copied_files
        total_skipped_files += skipped_files
        total_workers += copied_workers

        print(
            f"[MODEL] done: {model_name}, "
            f"workers={copied_workers}, copied={copied_files}, skipped={skipped_files}"
        )
        print("-" * 80)

    print("[DONE] all selected models copied")
    print(f"[SUMMARY] workers copied: {total_workers}")
    print(f"[SUMMARY] files copied: {total_copied_files}")
    print(f"[SUMMARY] files skipped: {total_skipped_files}")


if __name__ == "__main__":
    main()