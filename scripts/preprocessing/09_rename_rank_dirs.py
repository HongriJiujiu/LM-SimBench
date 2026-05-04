#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
import re
from datetime import datetime
from collections import defaultdict


DEFAULT_OUTPUT_ROOT = Path("LM-SimBench")
DEFAULT_FILTERED_ROOT = Path("raw_filtered")
DEFAULT_WARNING_LOG_PATH = DEFAULT_OUTPUT_ROOT / "warnings" / "rank_dir_warnings.log"
NON_MODEL_DIR_NAMES = {"warnings"}

OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
FILTERED_ROOT = DEFAULT_FILTERED_ROOT
WARNING_LOG_PATH = DEFAULT_WARNING_LOG_PATH

RANK_DIR_TEMPLATE = "rank_{rank_id}"

# 开关 1：
# True  表示只检查重复 rank_id，不重命名。
# False 表示不只检查，可配合 DO_RENAME 执行重命名。
CHECK_DUPLICATE_ONLY = False

# 开关 2：
# True  表示允许执行 rename。
# False 表示不执行 rename。
DO_RENAME = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename final worker directories to rank_<id> using profiler_info files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"最终结果根目录，默认: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--filtered-root",
        type=Path,
        default=DEFAULT_FILTERED_ROOT,
        help=f"过滤后的 profiler 数据根目录，默认: {DEFAULT_FILTERED_ROOT}",
    )
    parser.add_argument(
        "--warning-log",
        type=Path,
        default=None,
        help=f"warning 日志路径，默认: <output-root>/warnings/rank_dir_warnings.log",
    )
    parser.add_argument(
        "--check-duplicates-only",
        action="store_true",
        help="只检查重复 rank_id，不执行重命名。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="打印将要执行的重命名操作，但不修改目录。",
    )
    return parser.parse_args()


def init_warning_log():
    WARNING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with WARNING_LOG_PATH.open("w", encoding="utf-8") as f:
        f.write(f"[INFO] warning log created at {datetime.now()}\n")
        f.write(f"[INFO] OUTPUT_ROOT = {OUTPUT_ROOT}\n")
        f.write(f"[INFO] FILTERED_ROOT = {FILTERED_ROOT}\n")
        f.write(f"[INFO] CHECK_DUPLICATE_ONLY = {CHECK_DUPLICATE_ONLY}\n")
        f.write(f"[INFO] DO_RENAME = {DO_RENAME}\n")
        f.write("\n")


def log_warning(message: str):
    print(message)

    with WARNING_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def find_rank_id_in_json(obj):
    """
    递归查找 JSON 中的 rank_id 字段。
    """
    if isinstance(obj, dict):
        if "rank_id" in obj:
            return str(obj["rank_id"])

        for value in obj.values():
            result = find_rank_id_in_json(value)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for item in obj:
            result = find_rank_id_in_json(item)
            if result is not None:
                return result

    return None


def read_rank_id_from_profiler_info(profiler_info_path: Path):
    """
    从 profiler_info_*.json 中读取 rank_id。
    读取失败返回 None，不中断整体流程。
    """
    try:
        with profiler_info_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        rank_id = find_rank_id_in_json(data)

        if rank_id is None:
            log_warning(
                "[WARN] rank_id not found in profiler_info:\n"
                f"       file: {profiler_info_path}"
            )
            return None

        rank_id = str(rank_id).strip()

        if not re.fullmatch(r"\d+", rank_id):
            log_warning(
                "[WARN] invalid rank_id in profiler_info:\n"
                f"       file: {profiler_info_path}\n"
                f"       rank_id: {rank_id}"
            )
            return None

        return rank_id

    except Exception as e:
        log_warning(
            "[WARN] failed to read profiler_info:\n"
            f"       file: {profiler_info_path}\n"
            f"       error: {e}"
        )
        return None


def find_profiler_info_file(model_name: str, config_name: str, final_rank_dir_name: str):
    """
    根据最终结果中的原始 worker 目录名，
    去 raw_filtered 中寻找对应的 profiler_info_*.json。
    """
    candidates = []

    standard_dir = (
        FILTERED_ROOT
        / model_name
        / config_name
        / final_rank_dir_name
    )
    candidates.extend(sorted(standard_dir.glob("profiler_info*.json")))

    fallback_dir = (
        FILTERED_ROOT
        / model_name
        / final_rank_dir_name
    )
    candidates.extend(sorted(fallback_dir.glob("profiler_info*.json")))

    candidates.extend(sorted(standard_dir.glob("**/profiler_info*.json")))
    candidates.extend(sorted(fallback_dir.glob("**/profiler_info*.json")))

    unique_candidates = []
    seen = set()

    for path in candidates:
        if path not in seen and path.is_file():
            unique_candidates.append(path)
            seen.add(path)

    if not unique_candidates:
        return None

    if len(unique_candidates) > 1:
        log_warning(
            "[WARN] multiple profiler_info files found, use first one:\n"
            + "\n".join(f"       {p}" for p in unique_candidates)
        )

    return unique_candidates[0]


def is_already_rank_dir(dirname: str) -> bool:
    """
    判断目录是否已经是 rank_0 / rank_1 这种格式。
    """
    return re.fullmatch(r"rank_\d+", dirname) is not None


def collect_rank_mapping_for_config(model_name: str, config_dir: Path):
    """
    收集一个 model/config 下，最终结果中所有未重命名 worker 目录对应的 rank_id。

    返回：
    rank_to_items:
        {
            "0": [
                {
                    "final_rank_dir": Path(...),
                    "profiler_info_path": Path(...)
                },
                ...
            ]
        }
    """
    config_name = config_dir.name
    rank_to_items = defaultdict(list)

    for final_rank_dir in sorted(config_dir.iterdir()):
        if not final_rank_dir.is_dir():
            continue

        if is_already_rank_dir(final_rank_dir.name):
            continue

        profiler_info_path = find_profiler_info_file(
            model_name=model_name,
            config_name=config_name,
            final_rank_dir_name=final_rank_dir.name,
        )

        if profiler_info_path is None:
            log_warning(
                "[WARN] profiler_info*.json not found, skip:\n"
                f"       final_dir: {final_rank_dir}\n"
                f"       expected_filtered_dir_1: "
                f"{FILTERED_ROOT / model_name / config_name / final_rank_dir.name}\n"
                f"       expected_filtered_dir_2: "
                f"{FILTERED_ROOT / model_name / final_rank_dir.name}"
            )
            continue

        rank_id = read_rank_id_from_profiler_info(profiler_info_path)

        if rank_id is None:
            continue

        rank_to_items[rank_id].append(
            {
                "final_rank_dir": final_rank_dir,
                "profiler_info_path": profiler_info_path,
            }
        )

    return rank_to_items


def check_duplicate_rank_ids_for_config(model_name: str, config_dir: Path, rank_to_items):
    """
    检查当前 model/config 下是否有重复 rank_id。
    如果有，写入 warning.log。
    """
    config_name = config_dir.name
    has_duplicate = False

    for rank_id, items in sorted(rank_to_items.items(), key=lambda x: int(x[0])):
        if len(items) <= 1:
            continue

        has_duplicate = True

        msg_lines = [
            "[WARN] duplicate rank_id detected in final output:",
            f"       model: {model_name}",
            f"       config: {config_name}",
            f"       rank_id: {rank_id}",
            f"       count: {len(items)}",
            "       dirs:",
        ]

        for item in items:
            msg_lines.append(f"         - final_dir: {item['final_rank_dir']}")
            msg_lines.append(f"           profiler_info: {item['profiler_info_path']}")

        log_warning("\n".join(msg_lines))

    return has_duplicate


def rename_rank_dir(final_rank_dir: Path, rank_id: str):
    """
    将最终结果中的 worker 目录重命名为 rank_x。
    如果目标已存在，只 warning，不覆盖、不删除。
    """
    target_name = RANK_DIR_TEMPLATE.format(rank_id=rank_id)
    target_dir = final_rank_dir.parent / target_name

    if final_rank_dir.name == target_name:
        print(f"[SKIP] already renamed: {final_rank_dir}")
        return

    if target_dir.exists():
        log_warning(
            "[WARN] target dir already exists, skip current dir to avoid overwrite:\n"
            f"       source: {final_rank_dir}\n"
            f"       target: {target_dir}"
        )
        return

    if not DO_RENAME:
        print(f"[DRY-RUN] rename: {final_rank_dir} -> {target_dir}")
        return

    final_rank_dir.rename(target_dir)
    print(f"[OK] rename: {final_rank_dir} -> {target_dir}")


def process_all_rank_dirs():
    """
    主流程：
    1. 收集每个 model/config 下的 rank_id 映射；
    2. 检查重复 rank_id；
    3. 如果 CHECK_DUPLICATE_ONLY=True，则不执行 rename；
    4. 如果 CHECK_DUPLICATE_ONLY=False，则根据 DO_RENAME 决定是否真正 rename。
    """
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"not found: {OUTPUT_ROOT}")

    if not FILTERED_ROOT.exists():
        raise FileNotFoundError(f"not found: {FILTERED_ROOT}")

    total_duplicate_configs = 0

    for model_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        if model_name in NON_MODEL_DIR_NAMES:
            continue

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            config_name = config_dir.name

            rank_to_items = collect_rank_mapping_for_config(
                model_name=model_name,
                config_dir=config_dir,
            )

            has_duplicate = check_duplicate_rank_ids_for_config(
                model_name=model_name,
                config_dir=config_dir,
                rank_to_items=rank_to_items,
            )

            if has_duplicate:
                total_duplicate_configs += 1

            if CHECK_DUPLICATE_ONLY:
                continue

            for rank_id, items in sorted(rank_to_items.items(), key=lambda x: int(x[0])):
                if len(items) > 1:
                    log_warning(
                        "[WARN] skip rename because duplicate rank_id exists:\n"
                        f"       model: {model_name}\n"
                        f"       config: {config_name}\n"
                        f"       rank_id: {rank_id}"
                    )
                    continue

                final_rank_dir = items[0]["final_rank_dir"]
                rename_rank_dir(final_rank_dir, rank_id)

    print(f"[INFO] duplicate config count: {total_duplicate_configs}")


def main():
    global OUTPUT_ROOT, FILTERED_ROOT, WARNING_LOG_PATH
    global CHECK_DUPLICATE_ONLY, DO_RENAME

    args = parse_args()
    OUTPUT_ROOT = args.output_root.expanduser().resolve()
    FILTERED_ROOT = args.filtered_root.expanduser().resolve()
    WARNING_LOG_PATH = (
        args.warning_log.expanduser().resolve()
        if args.warning_log is not None
        else OUTPUT_ROOT / "warnings" / "rank_dir_warnings.log"
    )
    CHECK_DUPLICATE_ONLY = args.check_duplicates_only
    DO_RENAME = not args.dry_run

    init_warning_log()

    print(f"[INFO] output root: {OUTPUT_ROOT}")
    print(f"[INFO] filtered root: {FILTERED_ROOT}")
    print(f"[INFO] warning log: {WARNING_LOG_PATH}")
    print(f"[INFO] CHECK_DUPLICATE_ONLY = {CHECK_DUPLICATE_ONLY}")
    print(f"[INFO] DO_RENAME = {DO_RENAME}")

    process_all_rank_dirs()

    print("[DONE] process finished.")
    print(f"[DONE] warning log saved to: {WARNING_LOG_PATH}")


if __name__ == "__main__":
    main()