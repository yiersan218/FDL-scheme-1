"""Run and index the frozen A2 per-view prototype over missing-view rates."""

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import statistics
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

from config import PROJECT_ROOT, resolve_project_path
from main import run


DATASETS = (
    "ALOI_100", "flower17", "LandUse_21", "Mfeat",
    "NUSWIDE", "Scene-15", "animal",
)
RATES = (0.0, 0.1, 0.3, 0.5, 0.7)
DEFAULT_SEEDS = (42,)
PROTOCOL_VERSION = "a2-missing-matrix-v1"
FROZEN_PROTOCOL_VERSION = "a2-only-prototype-v1"
DEFAULT_RESULTS_DIR = "results-性能统计"
DEFAULT_REPORT = "性能统计.md"
DEFAULT_FROZEN = "results-优化2/protocol-v2/a2/frozen_selection.json"
SOURCE_FILES = (
    "system/config.py",
    "system/main.py",
    "system/flcore/trainmodel/multiview.py",
    "system/flcore/clients/clientcluster.py",
    "system/flcore/servers/servercluster.py",
    "system/flcore/compression.py",
    "system/run_a2_matrix.py",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--rates", nargs="+", type=float, default=list(RATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--frozen-selection", default=DEFAULT_FROZEN)
    parser.add_argument(
        "--index-only", action="store_true",
        help="rebuild indexes from complete existing A2 records without training",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_display_path(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def source_hash():
    records = [[name, sha256(PROJECT_ROOT / name)] for name in SOURCE_FILES]
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def runtime_environment(device):
    gpu_inventory = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            gpu_inventory.append({
                "index": index,
                "name": properties.name,
                "compute_capability": list(capability),
                "total_memory_bytes": int(properties.total_memory),
            })
    payload = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_inventory": gpu_inventory,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def rate_tag(rate):
    return f"rate-{float(rate):g}".replace(".", "p")


def source_scenario(rate):
    return "full" if float(rate) == 0.0 else "missing_0p3"


def frozen_config_path(frozen_path, entry):
    path = Path(entry["config_path"])
    if not path.is_absolute():
        path = Path(frozen_path).parent / path
    return path.resolve()


def load_frozen(frozen_path):
    frozen_path = Path(frozen_path).resolve()
    frozen = read_json(frozen_path)
    if frozen.get("protocol_version") != FROZEN_PROTOCOL_VERSION:
        raise RuntimeError(f"Unsupported A2 frozen protocol: {frozen_path}")
    rule = frozen.get("candidate", {})
    expected = {
        "variant": "A2",
        "prototype_mode": "per_view",
        "per_view_head_type": "student_t",
        "student_t_distance_scale": 1.0,
        "view_semantic_weight": 0.0,
    }
    for key, value in expected.items():
        if rule.get(key) != value:
            raise RuntimeError(
                f"Frozen A2 rule changed: expected {key}={value!r}, "
                f"found {rule.get(key)!r}"
            )
    for dataset in DATASETS:
        scenarios = frozen.get("datasets", {}).get(dataset, {}).get("scenarios", {})
        for scenario in ("full", "missing_0p3"):
            if scenario not in scenarios:
                raise RuntimeError(f"Missing frozen A2 config: {dataset}/{scenario}")
            entry = scenarios[scenario]
            config_path = frozen_config_path(frozen_path, entry)
            if not config_path.is_file():
                raise RuntimeError(f"Missing frozen A2 config file: {config_path}")
            if entry.get("config_sha256") != sha256(config_path):
                raise RuntimeError(f"Frozen A2 config hash mismatch: {config_path}")
            config = read_json(config_path)
            if config.get("dataset", {}).get("name") != dataset:
                raise RuntimeError(f"Frozen A2 dataset mismatch: {config_path}")
            model = config.get("model", {})
            if model.get("prototype_mode") != "per_view":
                raise RuntimeError(f"A2 config is not per_view: {config_path}")
            if model.get("per_view_head_type") != "student_t":
                raise RuntimeError(f"A2 config is not Student-t: {config_path}")
            if float(model.get("student_t_distance_scale", math.nan)) != 1.0:
                raise RuntimeError(f"A2 Student-t scale changed: {config_path}")
            if float(config.get("loss_weights", {}).get("view_semantic", math.nan)) != 0.0:
                raise RuntimeError(f"A2 semantic weight changed: {config_path}")
    return frozen


def build_config(
    frozen, frozen_path, dataset, rate, seed, device, results_root,
    source_sha256=None, environment=None,
):
    scenario = source_scenario(rate)
    entry = frozen["datasets"][dataset]["scenarios"][scenario]
    config = copy.deepcopy(read_json(frozen_config_path(frozen_path, entry)))
    config.pop("experiment", None)
    missing_enabled = float(rate) > 0.0
    for legacy_key in ("method", "heads", "anchor_size", "imputation_weight", "projector"):
        config["missing"].pop(legacy_key, None)
    config["missing"]["enabled"] = missing_enabled
    config["missing"]["rate"] = float(rate)
    config["compression"]["method"] = "stage" if missing_enabled else "none"
    config["compression"]["error_feedback"] = missing_enabled
    config["training"]["seed"] = int(seed)
    config["training"]["device"] = device
    config["model"]["prototype_mode"] = "per_view"
    config["model"]["cluster_head_type"] = "student_t"
    config["model"]["per_view_head_type"] = "student_t"
    config["model"]["student_t_distance_scale"] = 1.0
    config["loss_weights"]["view_semantic"] = 0.0
    output = results_root / "A2" / rate_tag(rate) / f"seed-{seed}"
    config["output"]["directory"] = str(output.resolve())
    config["output"]["save_model"] = False

    source_sha256 = source_sha256 or source_hash()
    environment = copy.deepcopy(environment or runtime_environment(device))
    data_path = resolve_project_path(config["dataset"]["file"]).resolve()
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "A2",
        "variant": "A2",
        "prototype_mode": "per_view",
        "dataset": dataset,
        "nominal_missing_rate": float(rate),
        "seed": int(seed),
        "source_scenario": scenario,
        "metric_source": "summary.metrics (clustering NMI-best checkpoint)",
        "source_sha256": source_sha256,
        "data_sha256": sha256(data_path),
        "environment": environment,
        "environment_sha256": environment["sha256"],
    }
    fingerprint_payload = copy.deepcopy(config)
    fingerprint_payload["experiment"] = metadata
    metadata["fingerprint"] = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config["experiment"] = metadata
    return config, output / dataset


def result_row(result_dir):
    summary_path = result_dir / "summary.json"
    history_path = result_dir / "history.json"
    summary = read_json(summary_path)
    history = read_json(history_path)
    meta = summary["config"]["experiment"]
    if meta.get("method") != "A2" or meta.get("prototype_mode") != "per_view":
        raise RuntimeError(f"Non-A2 result found: {summary_path}")
    metrics = summary["metrics"]
    keys = ("acc", "nmi", "ari")
    if not all(math.isfinite(float(metrics[key])) for key in keys):
        raise RuntimeError(f"Non-finite clustering metric: {summary_path}")
    rounds = int(summary["config"]["training"]["rounds"])
    best_round = int(summary["best_round"])
    if len(history) != rounds or not 1 <= best_round <= rounds:
        raise RuntimeError(f"Incomplete round history: {result_dir}")
    best_history = history[best_round - 1]
    if best_history.get("phase") != "clustering":
        raise RuntimeError(f"Best round is outside clustering: {result_dir}")
    for key in keys:
        if abs(float(best_history["clustering"][key]) - float(metrics[key])) > 1e-12:
            raise RuntimeError(f"Best metric/history mismatch for {key}: {result_dir}")
    return {
        "method": "A2",
        "dataset": summary["dataset"],
        "missing_rate": float(meta["nominal_missing_rate"]),
        "seed": int(meta["seed"]),
        "prototype_mode": "per_view",
        "acc": float(metrics["acc"]),
        "nmi": float(metrics["nmi"]),
        "ari": float(metrics["ari"]),
        "best_round": best_round,
        "rounds": rounds,
        "summary_path": project_display_path(summary_path),
        "history_path": project_display_path(history_path),
        "summary_sha256": sha256(summary_path),
        "history_sha256": sha256(history_path),
        "fingerprint": meta["fingerprint"],
        "record_source_sha256": meta["source_sha256"],
    }


def job_key(dataset, rate, seed):
    return dataset, float(rate), int(seed)


def _config_without_run_metadata(config):
    result = copy.deepcopy(config)
    result.pop("experiment", None)
    result.get("output", {}).pop("directory", None)
    for legacy_key in ("method", "heads", "anchor_size", "imputation_weight", "projector"):
        result.get("missing", {}).pop(legacy_key, None)
    return result


def validate_saved_config(saved, expected, result_dir):
    metadata = saved.get("experiment", {})
    if not isinstance(metadata.get("protocol_version"), str):
        raise RuntimeError(f"A2 record protocol is missing: {result_dir}")
    if metadata.get("method") != "A2" or metadata.get("prototype_mode") != "per_view":
        raise RuntimeError(f"Saved result is not A2: {result_dir}")
    claimed = metadata.get("fingerprint")
    payload = copy.deepcopy(saved)
    payload.get("experiment", {}).pop("fingerprint", None)
    payload.get("experiment", {}).pop("prototype_resolution", None)
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if claimed != actual:
        raise RuntimeError(f"Saved fingerprint is invalid: {result_dir}")
    if _config_without_run_metadata(saved) != _config_without_run_metadata(expected):
        raise RuntimeError(f"Saved A2 config differs from frozen plan: {result_dir}")


def scan_rows(root, expected_jobs):
    rows_by_key = {}
    expected_paths = {
        str(job["result_dir"].resolve()).lower(): key
        for key, job in expected_jobs.items()
    }
    for summary_path in root.glob("*/*/seed-*/*/summary.json"):
        if summary_path.relative_to(root).parts[0] != "A2":
            raise RuntimeError(f"Unexpected non-A2 matrix result: {summary_path}")
        summary = read_json(summary_path)
        meta = summary.get("config", {}).get("experiment", {})
        result_dir = summary_path.parent.resolve()
        path_key = str(result_dir).lower()
        if path_key not in expected_paths:
            raise RuntimeError(f"Unexpected A2 matrix result: {result_dir}")
        key = expected_paths[path_key]
        actual_key = job_key(
            summary.get("dataset"), meta.get("nominal_missing_rate"), meta.get("seed")
        )
        if actual_key != key:
            raise RuntimeError(
                f"Result metadata/path mismatch: expected {key}, found {actual_key}"
            )
        if key in rows_by_key:
            raise RuntimeError(f"Duplicate A2 matrix result: {key}")
        validate_saved_config(summary["config"], expected_jobs[key]["config"], result_dir)
        rows_by_key[key] = result_row(result_dir)
    order = {name: index for index, name in enumerate(DATASETS)}
    return sorted(
        rows_by_key.values(),
        key=lambda row: (order[row["dataset"]], row["missing_rate"], row["seed"]),
    )


def format_value(values):
    if len(values) == 1:
        return f"{values[0]:.6f}"
    return f"{statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}"


def expected_key_set(plan):
    return {
        job_key(dataset, rate, seed)
        for dataset in plan["datasets"]
        for rate in plan["rates"]
        for seed in plan["seeds"]
    }


def completed_key_set(rows):
    return {job_key(row["dataset"], row["missing_rate"], row["seed"]) for row in rows}


def write_report(report_path, results_root, rows, plan):
    expected = int(plan["expected_runs"])
    complete = completed_key_set(rows) == expected_key_set(plan)
    seeds = plan["seeds"]
    lines = [
        "# A2 多缺失率聚类性能统计",
        "",
        f"完成进度：**{len(rows)}/{expected}**。"
        + ("全部实验已完成。" if complete else "当前为断点续跑中的阶段性记录。"),
        "",
        "## 实验口径",
        "",
        f"- 数据集：`{'、'.join(plan['datasets'])}`。",
        f"- 名义缺失样本率：`{', '.join(f'{rate:g}' for rate in plan['rates'])}`；"
        "每个被选中的不完整样本随机遮蔽一个视图。",
        f"- 种子：`{', '.join(map(str, seeds))}`。",
        "- 方法仅保留 A2：每个视图维护独立 Student-t 原型，"
        "`model.prototype_mode=per_view`。",
        "- 缺失率 0 使用冻结完整视图配置；正缺失率统一从冻结 `missing_0p3` 配置派生，"
        "只改变缺失率，不按缺失率重新调参。",
        "- 缺失率 0 使用 `compression.method=none`；正缺失率使用 `stage`＋客户端误差反馈，"
        "预算为稠密普通参数载荷的 50%。因此跨越 0 与正缺失率时同时改变了缺失和压缩设置，"
        "不能当作严格单因素消融。",
        "- ACC、NMI、ARI 来自聚类阶段按 NMI 选出的最佳轮；ACC 与 ARI 取同一轮。"
        "标签只用于评估与最佳轮选择，不进入训练。",
        f"- 原始记录目录：`{project_display_path(results_root)}/A2/`。",
        "- 原始 A2 `summary.json` 保留首次配对矩阵运行时的协议版本和源码哈希，以维持"
        "指纹可审计；当前索引另列记录源码哈希与索引器源码哈希，不据此恢复已删除的其他方法。",
        "",
        "## 各数据集结果",
        "",
        "| 数据集 | 缺失率 | Seed | 最佳轮/总轮 | ACC | NMI | ARI |",
        "|---|---:|---|---|---:|---:|---:|",
    ]
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["missing_rate"]), []).append(row)
    for dataset in plan["datasets"]:
        for rate in plan["rates"]:
            group = sorted(
                grouped.get((dataset, float(rate)), []), key=lambda row: row["seed"]
            )
            if not group:
                continue
            round_text = ", ".join(
                f"{row['best_round']}/{row['rounds']}" for row in group
            )
            lines.append(
                f"| {dataset} | {rate:g} | "
                f"{', '.join(str(row['seed']) for row in group)} | "
                f"{round_text} | "
                f"{format_value([row['acc'] for row in group])} | "
                f"{format_value([row['nmi'] for row in group])} | "
                f"{format_value([row['ari'] for row in group])} |"
            )
    lines.extend([
        "",
        "## 七数据集宏平均",
        "",
        "| 缺失率 | ACC | NMI | ARI |",
        "|---:|---:|---:|---:|",
    ])
    for rate in plan["rates"]:
        selected = [row for row in rows if row["missing_rate"] == float(rate)]
        if len(selected) != len(plan["datasets"]) * len(seeds):
            continue
        lines.append(
            f"| {rate:g} | {statistics.mean(row['acc'] for row in selected):.6f} | "
            f"{statistics.mean(row['nmi'] for row in selected):.6f} | "
            f"{statistics.mean(row['ari'] for row in selected):.6f} |"
        )
    lines.extend([
        "",
        "每个单元的配置、逐轮历史、最佳轮、数据掩码信息和结果哈希均保存在原始目录；"
        f"汇总索引见 `{project_display_path(results_root / 'summary.json')}` 和 "
        f"`{project_display_path(results_root / 'manifest.json')}`。",
        "",
    ])
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")


def write_outputs(root, report, rows, plan):
    write_json(root / "matrix_plan.json", plan)
    write_json(root / "summary.json", rows)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "plan": plan,
        "completed_runs": len(rows),
        "complete": completed_key_set(rows) == expected_key_set(plan),
        "records": [
            {key: row[key] for key in (
                "method", "dataset", "missing_rate", "seed",
                "summary_path", "history_path", "summary_sha256",
                "history_sha256", "fingerprint", "record_source_sha256",
            )}
            for row in rows
        ],
    }
    write_json(root / "manifest.json", manifest)
    write_report(report, root, rows, plan)


def main():
    args = parse_args()
    datasets = list(dict.fromkeys(args.datasets))
    rates = list(dict.fromkeys(float(rate) for rate in args.rates))
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    if any(not math.isfinite(rate) or rate < 0 or rate >= 1 for rate in rates):
        raise ValueError("--rates must be in [0, 1)")
    if args.force and args.index_only:
        raise ValueError("--force and --index-only cannot be used together")
    if args.device == "cuda" and not torch.cuda.is_available() and not args.index_only:
        raise RuntimeError("CUDA was requested but is unavailable")

    root = resolve_project_path(args.results_dir).resolve()
    report = resolve_project_path(args.report).resolve()
    frozen_path = resolve_project_path(args.frozen_selection).resolve()
    frozen = load_frozen(frozen_path)
    current_source_hash = source_hash()
    environment = runtime_environment(args.device)
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "A2",
        "prototype_mode": "per_view",
        "datasets": datasets,
        "rates": rates,
        "seeds": seeds,
        "device": args.device,
        "metric_source": "summary.metrics (clustering NMI-best checkpoint)",
        "index_source_sha256": current_source_hash,
        "index_source_files": list(SOURCE_FILES),
        "environment": environment,
        "environment_sha256": environment["sha256"],
        "dataset_sha256": {
            dataset: sha256(resolve_project_path(
                read_json(frozen_config_path(
                    frozen_path,
                    frozen["datasets"][dataset]["scenarios"]["full"],
                ))["dataset"]["file"]
            ).resolve())
            for dataset in datasets
        },
        "frozen_selection": project_display_path(frozen_path),
        "frozen_selection_sha256": sha256(frozen_path),
        "expected_runs": len(datasets) * len(rates) * len(seeds),
    }
    expected_jobs = {}
    for dataset in datasets:
        for rate in rates:
            for seed in seeds:
                config, result_dir = build_config(
                    frozen, frozen_path, dataset, rate, seed, args.device, root,
                    current_source_hash, environment,
                )
                expected_jobs[job_key(dataset, rate, seed)] = {
                    "config": config, "result_dir": result_dir,
                }

    rows = scan_rows(root, expected_jobs)
    record_sources = sorted({row["record_source_sha256"] for row in rows})
    plan["record_source_sha256"] = record_sources
    plan["source_note"] = (
        "index_source_sha256 identifies the current A2 indexer; "
        "record_source_sha256 values come from immutable raw run metadata"
    )
    write_outputs(root, report, rows, plan)
    if args.index_only:
        if completed_key_set(rows) != expected_key_set(plan):
            missing = len(expected_key_set(plan) - completed_key_set(rows))
            raise RuntimeError(f"A2 index-only rebuild is missing {missing} runs")
        print(f"Indexed {len(rows)} existing A2 experiments. Report: {report}")
        return

    jobs = [
        (dataset, rate, seed)
        for dataset in datasets for rate in rates for seed in seeds
    ]
    for index, (dataset, rate, seed) in enumerate(jobs, 1):
        expected = expected_jobs[job_key(dataset, rate, seed)]
        config = copy.deepcopy(expected["config"])
        result_dir = expected["result_dir"]
        summary_path = result_dir / "summary.json"
        history_path = result_dir / "history.json"
        if summary_path.exists() and history_path.exists() and not args.force:
            saved = read_json(summary_path)
            validate_saved_config(saved["config"], config, result_dir)
            result_row(result_dir)
            print(
                f"[{index}/{len(jobs)}] REUSE A2 {dataset} rate={rate:g} seed={seed}",
                flush=True,
            )
        else:
            print(
                f"[{index}/{len(jobs)}] RUN A2 {dataset} rate={rate:g} seed={seed}",
                flush=True,
            )
            run(config)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = scan_rows(root, expected_jobs)
        plan["record_source_sha256"] = sorted({
            row["record_source_sha256"] for row in rows
        })
        write_outputs(root, report, rows, plan)
    print(f"Completed {len(jobs)} A2 experiments. Report: {report}", flush=True)


if __name__ == "__main__":
    main()
