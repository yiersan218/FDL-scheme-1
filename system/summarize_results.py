"""从已有训练历史提取最佳轮和真实末轮，生成跨数据集性能摘要。"""

import argparse
import json

from config import ensure_writable_output


METRIC_NAMES = ("acc", "nmi", "ari")


def _load_json(path):
    """读取 UTF-8 编码的结果或历史 JSON 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metric_record(history, round_number):
    """按通信轮编号查找聚类指标，缺失时拒绝用其他轮次替代。"""
    for record in history:
        if int(record["round"]) == int(round_number) and "clustering" in record:
            return record["clustering"]
    raise ValueError(f"No clustering metrics found for round {round_number}")


def collect_result(dataset_dir):
    """读取单个数据集的结果并对照最佳轮与配置要求的末轮。

    Args:
        dataset_dir: 同时包含 summary.json 和 history.json 的数据集目录。

    Returns:
        dict: 数据集、种子、轮次、最佳轮指标、末轮指标及末轮减最佳轮差值。

    Raises:
        FileNotFoundError: 缺少摘要或历史文件。
        ValueError: 历史为空，或找不到指定轮次的聚类指标。
    """
    summary_path = dataset_dir / "summary.json"
    history_path = dataset_dir / "history.json"
    if not summary_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"Missing summary.json or history.json in {dataset_dir}")

    summary = _load_json(summary_path)
    history = _load_json(history_path)
    if not history:
        raise ValueError(f"Empty history: {history_path}")

    best_round = int(summary["best_round"])
    configured_rounds = int(summary["config"]["training"]["rounds"])
    best_metrics = _metric_record(history, best_round)
    # 要求找到配置规定的最终通信轮，以免把中断训练的最后一条记录当作完成结果。
    last_metrics = _metric_record(history, configured_rounds)
    return {
        "dataset": summary["dataset"],
        "seed": int(summary["config"]["training"]["seed"]),
        "best_round": best_round,
        "configured_rounds": configured_rounds,
        "best": {name: float(best_metrics[name]) for name in METRIC_NAMES},
        "last": {name: float(last_metrics[name]) for name in METRIC_NAMES},
        "delta": {
            name: float(last_metrics[name]) - float(best_metrics[name])
            for name in METRIC_NAMES
        },
    }


def write_summary(root, expected_datasets=None, title="Federated Multi-View Clustering Results"):
    """汇总指定数据集并写入结果根目录的 summary.md。

    Args:
        root: 结果根目录；相对路径按项目根目录解析。
        expected_datasets: 明确指定的数据集名称集合，缺省时扫描直接子目录。
        title: Markdown 报告标题。

    Returns:
        tuple: 汇总文件路径和用于生成报告的结果行。
    """
    root = ensure_writable_output(root)
    dataset_names = (
        sorted(expected_datasets)
        if expected_datasets is not None
        else sorted(path.name for path in root.iterdir() if path.is_dir())
    )
    rows = [collect_result(root / name) for name in dataset_names]
    if not rows:
        raise ValueError(f"No dataset results found in {root}")

    lines = [
        f"# {title}",
        "",
        f"- Results directory: `{root}`",
        "- Selection metric: NMI",
        "- Metric changes use `last epoch - best epoch`; negative values indicate degradation.",
        "",
        "| Dataset | Seed | ACC | NMI | ARI | Best epoch | Configured epochs | ΔACC | ΔNMI | ΔARI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        best = row["best"]
        delta = row["delta"]
        lines.append(
            f"| {row['dataset']} | {row['seed']} | {best['acc']:.6f} | "
            f"{best['nmi']:.6f} | {best['ari']:.6f} | {row['best_round']} | "
            f"{row['configured_rounds']} | {delta['acc']:+.6f} | "
            f"{delta['nmi']:+.6f} | {delta['ari']:+.6f} |"
        )

    output_path = ensure_writable_output(root / "summary.md")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path, rows


def parse_args():
    """解析待汇总的结果目录、数据集范围和报告标题。"""
    parser = argparse.ArgumentParser(description="Summarize completed clustering runs")
    parser.add_argument("--results-dir", default="results-补全聚类")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--title", default="Federated Multi-View Clustering Results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path, _rows = write_summary(args.results_dir, args.datasets, args.title)
    print(path)
