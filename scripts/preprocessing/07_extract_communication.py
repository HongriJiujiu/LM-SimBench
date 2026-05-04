#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


DEFAULT_SRC_ROOT = Path("raw_filtered")
DEFAULT_DST_ROOT = Path("LM-SimBench")

COMM_SCOPES = ("collective", "p2p")
LINK_TYPES = ("HCCS", "RDMA", "PCIE", "SDMA", "SIO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract communication metrics for all model folders."
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


def get_model_dirs(src_root: Path, models: List[str] | None, exclude_models: set[str]) -> List[Path]:
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


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_parallel_config(config_name: str) -> Dict[str, str]:
    """
    解析并行配置目录名，例如：
    dp1-tp1-pp8-cp1-RC
    dp1-tp4-pp2-cp1-noRC
    """
    result = {
        "dp": "",
        "tp": "",
        "pp": "",
        "cp": "",
        "recompute": "",
    }

    for part in config_name.split("-"):
        if part.startswith("dp"):
            result["dp"] = part[2:]
        elif part.startswith("tp"):
            result["tp"] = part[2:]
        elif part.startswith("pp"):
            result["pp"] = part[2:]
        elif part.startswith("cp"):
            result["cp"] = part[2:]
        elif part in ("RC", "noRC"):
            result["recompute"] = part

    return result


def is_config_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    name = path.name
    return (
        name.startswith("dp")
        and "-tp" in name
        and "-pp" in name
        and "-cp" in name
        and (name.endswith("-RC") or name.endswith("-noRC"))
    )


def is_worker_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    return (path / "ASCEND_PROFILER_OUTPUT").is_dir()


def find_profiler_info(worker_dir: Path) -> Optional[Path]:
    """
    在 worker 目录下查找 profiler_info_*.json。
    例如：
    profiler_info_0.json
    profiler_info_1.json
    """
    candidates = sorted(worker_dir.glob("profiler_info_*.json"))
    if not candidates:
        return None
    return candidates[0]


def read_rank_id(worker_dir: Path) -> str:
    """
    从 profiler_info_*.json 中读取 rank_id。
    """
    profiler_info_path = find_profiler_info(worker_dir)

    if profiler_info_path is None:
        return ""

    try:
        payload = load_json(profiler_info_path)
        return str(payload.get("rank_id", ""))
    except Exception:
        return ""


def parse_op_key(op_key: str) -> Tuple[str, str, str]:
    if "@" in op_key:
        raw_op_name, comm_group = op_key.split("@", 1)
    else:
        raw_op_name, comm_group = op_key, ""

    op_name = raw_op_name

    if raw_op_name.startswith("hcom_"):
        comm_type = raw_op_name[len("hcom_"):].split("__", 1)[0]
    else:
        comm_type = raw_op_name.split("__", 1)[0]

    return comm_type, comm_group, op_name


def summarize_size_distribution(size_distribution: Dict[str, Any]) -> Tuple[str, int, float]:
    if not size_distribution:
        return "", 0, 0.0

    total_packet_count = 0
    total_packet_time_ms = 0.0
    parts = []

    for packet_size, value in size_distribution.items():
        if isinstance(value, list) and len(value) >= 2:
            packet_count = value[0] or 0
            packet_time_ms = value[1] or 0
        else:
            packet_count = 0
            packet_time_ms = 0

        total_packet_count += int(packet_count)
        total_packet_time_ms += float(packet_time_ms)
        parts.append(f"{packet_size}:{packet_count}:{packet_time_ms}")

    return ";".join(parts), total_packet_count, total_packet_time_ms


def build_matrix_op_summary(communication_matrix: Dict[str, Any]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Aggregate compact matrix entries by communication scope and op name.

    注意：
    step_id 只作为内部 key 使用，用来匹配 communication.json 和 communication_matrix.json。
    最终不会输出 step_id。
    """
    summary = defaultdict(
        lambda: {
            "matrix_pair_count": 0,
            "matrix_transit_size_mb": 0.0,
            "matrix_transit_time_ms": 0.0,
            "matrix_pairs": set(),
            "matrix_transport_types": set(),
        }
    )

    for step_id, step_payload in communication_matrix.items():
        if not isinstance(step_payload, dict):
            continue

        for comm_scope in COMM_SCOPES:
            scope_payload = step_payload.get(comm_scope, {})
            if not isinstance(scope_payload, dict):
                continue

            for matrix_entry_name, pair_payload in scope_payload.items():
                if "-total@" in matrix_entry_name:
                    continue

                if not isinstance(pair_payload, dict):
                    continue

                for rank_pair, pair_info in pair_payload.items():
                    if not isinstance(pair_info, dict):
                        continue

                    op_name = pair_info.get("Op Name") or ""
                    if not op_name:
                        continue

                    item = summary[(step_id, comm_scope, op_name)]

                    item["matrix_pair_count"] += 1
                    item["matrix_transit_size_mb"] += float(pair_info.get("Transit Size(MB)", 0) or 0)
                    item["matrix_transit_time_ms"] += float(pair_info.get("Transit Time(ms)", 0) or 0)
                    item["matrix_pairs"].add(str(rank_pair))

                    transport_type = pair_info.get("Transport Type") or ""
                    if transport_type:
                        item["matrix_transport_types"].add(str(transport_type))

    return summary


def extract_comm_rows(
    communication: Dict[str, Any],
    matrix_summary: Dict[Tuple[str, str, str], Dict[str, Any]],
    model_name: str,
    config_name: str,
    worker_name: str,
    rank_id: str,
) -> List[Dict[str, Any]]:
    rows = []
    config_info = parse_parallel_config(config_name)

    for step_id, step_payload in communication.items():
        if not isinstance(step_payload, dict):
            continue

        for comm_scope in COMM_SCOPES:
            scope_payload = step_payload.get(comm_scope, {})
            if not isinstance(scope_payload, dict):
                continue

            for op_key, op_payload in scope_payload.items():
                if op_key == "Total Op Info":
                    continue

                if not isinstance(op_payload, dict):
                    continue

                comm_type, comm_group, op_name = parse_op_key(op_key)

                time_info = op_payload.get("Communication Time Info", {})
                bandwidth_info = op_payload.get("Communication Bandwidth Info", {})

                if not isinstance(time_info, dict):
                    time_info = {}
                if not isinstance(bandwidth_info, dict):
                    bandwidth_info = {}

                matrix_info = matrix_summary.get((step_id, comm_scope, op_name), {})

                base_row = {
                    "model_name": model_name,
                    "parallel_config": config_name,
                    "dp": config_info["dp"],
                    "tp": config_info["tp"],
                    "pp": config_info["pp"],
                    "cp": config_info["cp"],
                    "recompute": config_info["recompute"],
                    "worker_dir": worker_name,
                    "rank_id": rank_id,

                    "comm_scope": comm_scope,
                    "comm_type": comm_type,
                    "comm_group": comm_group,
                    "op_name": op_name,

                    "start_timestamp_us": time_info.get("Start Timestamp(us)", ""),
                    "elapsed_time_ms": time_info.get("Elapse Time(ms)", 0),
                    "transit_time_ms": time_info.get("Transit Time(ms)", 0),
                    "wait_time_ms": time_info.get("Wait Time(ms)", 0),
                    "synchronization_time_ms": time_info.get("Synchronization Time(ms)", 0),
                    "idle_time_ms": time_info.get("Idle Time(ms)", 0),
                    "wait_time_ratio": time_info.get("Wait Time Ratio", 0),
                    "synchronization_time_ratio": time_info.get("Synchronization Time Ratio", 0),

                    "matrix_pair_count": matrix_info.get("matrix_pair_count", 0),
                    "matrix_transport_types": ";".join(
                        sorted(matrix_info.get("matrix_transport_types", []))
                    ),
                    "matrix_rank_pairs": ";".join(
                        sorted(matrix_info.get("matrix_pairs", []))
                    ),
                    "matrix_transit_size_mb": matrix_info.get("matrix_transit_size_mb", 0),
                    "matrix_transit_time_ms": matrix_info.get("matrix_transit_time_ms", 0),
                }

                nonzero_link_rows = []

                for link_type in LINK_TYPES:
                    link_info = bandwidth_info.get(link_type, {})
                    if not isinstance(link_info, dict):
                        link_info = {}

                    transit_size_mb = link_info.get("Transit Size(MB)", 0) or 0
                    transit_time_ms = link_info.get("Transit Time(ms)", 0) or 0
                    bandwidth_gbps = link_info.get("Bandwidth(GB/s)", 0) or 0
                    large_packet_ratio = link_info.get("Large Packet Ratio", 0) or 0

                    size_distribution, packet_count, packet_time_ms = summarize_size_distribution(
                        link_info.get("Size Distribution", {})
                    )

                    if not any(
                        (
                            transit_size_mb,
                            transit_time_ms,
                            bandwidth_gbps,
                            large_packet_ratio,
                            packet_count,
                        )
                    ):
                        continue

                    nonzero_link_rows.append(
                        {
                            **base_row,
                            "link_type": link_type,
                            "link_transit_size_mb": transit_size_mb,
                            "link_transit_time_ms": transit_time_ms,
                            "link_bandwidth_gbps": bandwidth_gbps,
                            "large_packet_ratio": large_packet_ratio,
                            "size_distribution": size_distribution,
                            "packet_count": packet_count,
                            "packet_time_ms": packet_time_ms,
                        }
                    )

                if nonzero_link_rows:
                    rows.extend(nonzero_link_rows)
                else:
                    rows.append(
                        {
                            **base_row,
                            "link_type": "",
                            "link_transit_size_mb": 0,
                            "link_transit_time_ms": 0,
                            "link_bandwidth_gbps": 0,
                            "large_packet_ratio": 0,
                            "size_distribution": "",
                            "packet_count": 0,
                            "packet_time_ms": 0,
                        }
                    )

    return rows


FIELDNAMES = [
    "model_name",
    "parallel_config",
    "dp",
    "tp",
    "pp",
    "cp",
    "recompute",
    "worker_dir",
    "rank_id",

    "comm_scope",
    "comm_type",
    "comm_group",
    "op_name",

    "start_timestamp_us",
    "elapsed_time_ms",
    "transit_time_ms",
    "wait_time_ms",
    "synchronization_time_ms",
    "idle_time_ms",
    "wait_time_ratio",
    "synchronization_time_ratio",

    "link_type",
    "link_transit_size_mb",
    "link_transit_time_ms",
    "link_bandwidth_gbps",
    "large_packet_ratio",
    "size_distribution",
    "packet_count",
    "packet_time_ms",

    "matrix_pair_count",
    "matrix_transport_types",
    "matrix_rank_pairs",
    "matrix_transit_size_mb",
    "matrix_transit_time_ms",
]


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def process_worker(
    dst_root: Path,
    model_dir: Path,
    config_dir: Path,
    worker_dir: Path,
) -> Tuple[bool, str]:
    model_name = model_dir.name
    config_name = config_dir.name
    worker_name = worker_dir.name

    src_ascend_dir = worker_dir / "ASCEND_PROFILER_OUTPUT"
    communication_path = src_ascend_dir / "communication.json"
    communication_matrix_path = src_ascend_dir / "communication_matrix.json"

    if not communication_path.exists():
        return False, f"missing communication.json: {communication_path}"

    if not communication_matrix_path.exists():
        return False, f"missing communication_matrix.json: {communication_matrix_path}"

    rank_id = read_rank_id(worker_dir)

    communication = load_json(communication_path)
    communication_matrix = load_json(communication_matrix_path)

    matrix_summary = build_matrix_op_summary(communication_matrix)

    rows = extract_comm_rows(
        communication=communication,
        matrix_summary=matrix_summary,
        model_name=model_name,
        config_name=config_name,
        worker_name=worker_name,
        rank_id=rank_id,
    )

    dst_ascend_dir = (
        dst_root
        / model_name
        / config_name
        / worker_name
        / "ASCEND_PROFILER_OUTPUT"
    )

    output_path = dst_ascend_dir / "comm_time.csv"
    write_csv(rows, output_path)

    return True, f"rank_id={rank_id}, rows={len(rows)}, output={output_path}"


def main() -> None:
    args = parse_args()
    src_root = args.src_root.expanduser().resolve()
    dst_root = args.dst_root.expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    total_models = 0
    total_configs = 0
    total_workers = 0
    success_workers = 0
    failed_workers = 0

    print(f"[INFO] SRC_ROOT = {src_root}")
    print(f"[INFO] DST_ROOT = {dst_root}")
    print("=" * 100)

    model_dirs = get_model_dirs(
        src_root=src_root,
        models=args.models,
        exclude_models=set(args.exclude_models),
    )

    for model_dir in model_dirs:
        total_models += 1
        print(f"[MODEL] {model_dir.name}")

        for config_dir in sorted(model_dir.iterdir()):
            if not is_config_dir(config_dir):
                continue

            total_configs += 1
            print(f"  [CONFIG] {config_dir.name}")

            for worker_dir in sorted(config_dir.iterdir()):
                if not is_worker_dir(worker_dir):
                    continue

                total_workers += 1

                try:
                    ok, msg = process_worker(dst_root, model_dir, config_dir, worker_dir)
                    if ok:
                        success_workers += 1
                        print(f"    [OK] {worker_dir.name} | {msg}")
                    else:
                        failed_workers += 1
                        print(f"    [SKIP] {worker_dir.name} | {msg}")

                except Exception as exc:
                    failed_workers += 1
                    print(f"    [ERROR] {worker_dir.name} | {exc}")
                    traceback.print_exc()

        print("-" * 100)

    print("[DONE]")
    print(f"[SUMMARY] models scanned    = {total_models}")
    print(f"[SUMMARY] configs scanned   = {total_configs}")
    print(f"[SUMMARY] workers scanned   = {total_workers}")
    print(f"[SUMMARY] workers succeeded = {success_workers}")
    print(f"[SUMMARY] workers failed    = {failed_workers}")


if __name__ == "__main__":
    main()