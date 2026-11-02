"""核验矩阵原始记录，按随机种子汇总 ACC、NMI、ARI 的均值与方差。"""

import argparse
import copy
import itertools
import json
import math
import statistics
from pathlib import Path

from config import ALGORITHM_VERSION, ensure_writable_output
from run_incomplete import digest, file_hash, fingerprint, read_result


METRICS = ("acc", "nmi", "ari")


def _group_key(config):
    """用实验含义而非目录名称区分统计组。"""
    meta = config["experiment"]
    return (config["dataset"]["name"], float(meta["nominal_missing_rate"]),
            meta["variant"], meta["compression"])


def _repeat_signature(config):
    """忽略种子及输出位置，保证同组重复实验没有混入不同训练设置。"""
    value = copy.deepcopy(config)
    value["training"].pop("seed", None)
    value["output"].pop("directory", None)
    for name in ("seed", "fingerprint"):
        value["experiment"].pop(name, None)
    return digest(value)


def _validate_plan(plan, root):
    """检查计划的完整笛卡尔积、种子唯一性、配置指纹及组内设置。"""
    if plan.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("Unsupported matrix algorithm_version")
    dimensions = []
    for name in ("datasets", "rates", "seeds", "variants"):
        values = plan[name]
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Empty or duplicated plan dimension: {name}")
        dimensions.append(values)
    expected = set(itertools.product(*dimensions))
    if len(plan["jobs"]) != plan["expected_runs"] or len(expected) != plan["expected_runs"]:
        raise ValueError("Plan run count does not match its matrix dimensions")
    seen, directories, signatures = set(), set(), {}
    for job in plan["jobs"]:
        config = job["config"]
        meta = config["experiment"]
        dataset, rate, variant, compression = _group_key(config)
        seed = int(meta["seed"])
        identity = dataset, rate, seed, variant
        if identity in seen:
            raise ValueError(f"Duplicate planned seed/run: {identity}")
        seen.add(identity)
        if (not math.isfinite(rate) or not 0 <= rate < 1 or
                config.get("algorithm_version") != ALGORITHM_VERSION or
                meta.get("fingerprint") != fingerprint(config) or
                seed != config["training"]["seed"] or
                rate != config["missing"]["rate"] or
                compression != plan["compression"]):
            raise ValueError(f"Inconsistent planned config: {identity}")
        directory = Path(job["result_dir"]).resolve()
        if root not in directory.parents or directory in directories:
            raise ValueError(f"Invalid or repeated result directory: {directory}")
        directories.add(directory)
        key = dataset, rate, variant, compression
        signature = _repeat_signature(config)
        if signatures.setdefault(key, signature) != signature:
            raise ValueError(f"Different training settings within repeat group: {key}")
    if seen != expected:
        raise ValueError("Planned runs do not cover the declared matrix")
    for source in plan.get("source_files", []):
        relative = Path(source["path"])
        snapshot_root = (root / "source_snapshot").resolve()
        snapshot = (snapshot_root / relative).resolve()
        if snapshot_root not in snapshot.parents or not snapshot.is_file():
            raise ValueError(f"Missing or invalid source snapshot: {relative}")
        if file_hash(snapshot) != source["sha256"]:
            raise ValueError(f"Source snapshot hash mismatch: {relative}")


def collect_statistics(root, allow_partial=False):
    """读取计划及经过完整性核验的记录，返回统计和逐次结果。

    Args:
        root: 含 plan.json 的矩阵结果目录。
        allow_partial: 允许任务尚未完成；不完整组会明确标注样本数。

    Returns:
        tuple: 可序列化的统计报告及逐次实验结果列表。

    Raises:
        ValueError: 计划、统计分组或已完成记录不一致。
        FileNotFoundError: 严格模式下有实验尚未完成。
    """
    root = ensure_writable_output(root)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _validate_plan(plan, root)
    grouped = {_group_key(job["config"]): [] for job in plan["jobs"]}
    rows, pending = [], []
    for job in plan["jobs"]:
        directory = Path(job["result_dir"])
        if not all((directory / name).is_file() for name in ("summary.json", "history.json")):
            if not allow_partial:
                raise FileNotFoundError(f"Incomplete matrix run: {directory}")
            pending.append(str(directory))
            continue
        row = read_result(job)
        config = job["config"]
        key = _group_key(config)
        actual = row["dataset"], float(row["rate"]), row["variant"], row["compression"]
        if actual != key or row["seed"] != config["training"]["seed"]:
            raise ValueError(f"Result identity differs from plan: {directory}")
        rows.append(row)
        grouped[key].append(row)
    groups = []
    expected_seeds = sorted(plan["seeds"])
    for key in sorted(grouped):
        records = sorted(grouped[key], key=lambda row: row["seed"])
        seeds = [row["seed"] for row in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate result seeds: {key}")
        group = dict(zip(("dataset", "rate", "variant", "compression"), key))
        group.update(n=len(records), expected_n=len(expected_seeds), seeds=seeds,
                     complete=seeds == expected_seeds)
        for checkpoint in ("best", "last"):
            group[checkpoint] = {}
            for metric in METRICS:
                values = [float(row[checkpoint][metric]) for row in records]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Non-finite metric: {key}/{checkpoint}/{metric}")
                group[checkpoint][metric] = {
                    "mean": statistics.mean(values) if values else None,
                    "variance": statistics.variance(values) if len(values) >= 2 else None,
                    "population_variance": statistics.pvariance(values) if values else None,
                }
        groups.append(group)
    report = {
        "algorithm_version": ALGORITHM_VERSION,
        "plan_sha256": file_hash(plan_path),
        "compression": plan["compression"],
        "expected_seeds": expected_seeds,
        "variance_ddof": 1,
        "population_variance_ddof": 0,
        "metric_scale": "raw_ratio_not_percent",
        "best_definition": "同一次运行按配置 selection_metric 选定的同一最佳聚类轮；三个指标不分别取最大值",
        "last_definition": "配置规定的最后一个通信轮，取自原始 history.json",
        "expected_runs": plan["expected_runs"],
        "completed_runs": len(rows),
        "complete": not pending and all(group["complete"] for group in groups),
        "pending_result_directories": pending,
        "groups": groups,
    }
    return report, rows


def _format_number(value):
    """缺少足够重复实验时显示破折号，避免把未定义方差伪装为零。"""
    return "—" if value is None else f"{value:.10g}"


def _markdown(report):
    """生成最佳轮与末轮各自的六列统计，并解释方差口径。"""
    complete = report["complete"]
    lines = ["# 缺失多视图联邦聚类：五次重复实验统计", "",
             f"- 状态：{'全部完成' if complete else '进行中，以下为部分结果'}。",
             f"- 已完成：{report['completed_runs']}/{report['expected_runs']} 次运行。",
             f"- 随机种子：{', '.join(map(str, report['expected_seeds']))}。",
             f"- 通信压缩：`{report['compression']}`（none 表示不压缩）。",
             "- 均值与方差按每个数据集、缺失率、补全方案分别跨种子计算。",
             "- 表中方差为样本方差：Σ(x−均值)²/(n−1)；五次实验时分母为 4。",
             "- JSON 另保留总体方差（分母 n）。所有统计使用原始指标数值，没有乘以 100。",
             "- 单次结果的三个最佳指标来自同一个最佳轮，不分别挑选各自最大值。",
             "- 最佳轮属于按标签指标选择的评估结果；末轮表保留固定训练预算下的真实表现。",
             "- n 少于 2 时样本方差未定义，以 — 显示；未完成组不能作为五次实验结论。", ""]
    for checkpoint, title in (("best", "最佳轮统计"), ("last", "真实末轮统计")):
        lines.extend([f"## {title}", "",
                      "| 数据集 | 缺失率 | 方案 | 压缩 | n/计划 | ACC均值 | ACC方差 | NMI均值 | NMI方差 | ARI均值 | ARI方差 |",
                      "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for group in report["groups"]:
            cells = [group["dataset"], f"{group['rate']:g}", group["variant"],
                     group["compression"], f"{group['n']}/{group['expected_n']}"]
            for metric in METRICS:
                cells.extend(_format_number(group[checkpoint][metric][stat])
                             for stat in ("mean", "variance"))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["逐次结果及原始记录哈希见 [run_details.json](run_details.json)，完整统计见 [statistics.json](statistics.json)。", ""])
    return "\n".join(lines)


def write_statistics(root, allow_partial=False):
    """先完成全部核验，再写出统计；不修改单次训练结果或原有矩阵报告。"""
    root = ensure_writable_output(root)
    report, rows = collect_statistics(root, allow_partial)
    payloads = {
        "statistics.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "run_details.json": json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "statistics.md": _markdown(report),
    }
    for name, content in payloads.items():
        target = ensure_writable_output(root / name)
        temporary = ensure_writable_output(root / (name + ".tmp"))
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(target)
    return report


def main(argv=None):
    """解析统计目录与未完成实验的展示方式。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = write_statistics(args.results_dir, args.allow_partial)
    print(f"Statistics: {report['completed_runs']}/{report['expected_runs']} runs, complete={report['complete']}")
    return report


if __name__ == "__main__":
    main()
