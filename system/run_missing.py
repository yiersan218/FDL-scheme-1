"""Run the selected attention + Stage/EF missing-view protocol."""

import argparse
import copy
import json
import statistics
from pathlib import Path

from config import BACKUP_CONFIG_DIR, DEFAULT_CONFIG_DIR, apply_overrides, load_config, resolve_project_path
from main import run


DATASETS = {
    **{path.stem: path for path in DEFAULT_CONFIG_DIR.glob("*.json")},
    **{path.stem: path for path in BACKUP_CONFIG_DIR.glob("*.json")},
}
RATES = (0.0, 0.1, 0.3, 0.5, 0.7)
METHOD = "attention"
COMPRESSION = "stage_ef"


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention + Stage/EF missing-view clustering")
    parser.add_argument("--results-dir", default="results-缺失")
    parser.add_argument("--dataset", action="append", choices=sorted(DATASETS))
    parser.add_argument("--rate", type=float, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _tag(rate):
    return f"rate-{rate:g}".replace(".", "p")


def _expected_config(path, root, rate, seed, device):
    output = root / f"{METHOD}_{COMPRESSION}" / _tag(rate) / f"seed-{seed}"
    overrides = [
        "missing.enabled=true",
        f"missing.rate={rate}",
        f"missing.method={json.dumps(METHOD)}",
        f"training.seed={seed}",
        f"training.device={json.dumps(device)}",
        'compression.method="stage"',
        "compression.error_feedback=true",
        "compression.budget_ratio=0.5",
        "compression.candidates=4",
        f"output.directory={json.dumps(str(output), ensure_ascii=False)}",
    ]
    return apply_overrides(load_config(path), overrides), output


def _read(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compatible(saved, expected):
    """Reuse earlier attention runs after removing unused gated-only config keys."""
    saved = copy.deepcopy(saved)
    expected = copy.deepcopy(dict(expected))
    for config in (saved, expected):
        config.get("missing", {}).pop("imputation_weight", None)
        config.get("missing", {}).pop("projector", None)
    return saved == expected


def _row(result_dir, rate, seed):
    summary = _read(result_dir / "summary.json")
    history = _read(result_dir / "history.json")
    best = summary["metrics"]
    last = history[-1]["clustering"]
    return {
        "dataset": summary["dataset"],
        "method": METHOD,
        "compression": COMPRESSION,
        "rate": rate,
        "seed": seed,
        "acc": best["acc"],
        "nmi": best["nmi"],
        "ari": best["ari"],
        "best_round": summary["best_round"],
        "rounds": summary["config"]["training"]["rounds"],
        "last_nmi": last["nmi"],
        "stable": all(abs(best[key] - last[key]) <= 0.01 for key in ("acc", "nmi", "ari")),
        "missing_sample_rate": summary["missing"]["incomplete_sample_rate"],
        "missing_cell_rate": summary["missing"]["missing_cell_rate"],
        "uplink_bytes": summary["communication"]["uplink_bytes"],
        "dense_uplink_bytes": summary["communication"]["dense_uplink_bytes"],
        "compression_seconds": summary["communication"]["compression_seconds"],
        "elapsed_seconds": history[-1]["elapsed_seconds"],
        "path": str(result_dir),
    }


def _report(root, rows):
    rows = sorted(rows, key=lambda row: (row["dataset"], row["rate"], row["seed"]))
    (root / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    primary = {(row["dataset"], row["rate"]): row for row in rows if row["seed"] == 42}
    lines = [
        "# 缺失视图正式方案：Attention + Stage/EF",
        "",
        "每个不完整样本随机缺失恰好一个视图；客户端划分和掩码由种子确定，全程固定。"
        "注意力只使用客户端本地完整样本作锚点。Stage＋误差反馈的预算为当前模型稠密上行载荷的 50%。"
        "标签只用于 ACC/NMI/ARI 评估。本项目结果不等同于原论文 DDR-IMVC 的结果。",
        "",
        "## 七集缺失率曲线（seed 42，最佳轮 NMI）",
        "",
        "| 数据集 | 0（完整控制） | 0.1 | 0.3 | 0.5 | 0.7 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(DATASETS):
        values = [
            f"{primary[(dataset, rate)]['nmi']:.6f}"
            if (dataset, rate) in primary else "—"
            for rate in RATES
        ]
        lines.append(f"| {dataset} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## 完整记录",
        "",
        "指标来自聚类阶段 NMI 最佳轮恢复后的模型。稳定表示该轮至末轮的"
        " ACC、NMI、ARI 差值均不超过 0.01；上行 MB 是模拟器编码模型载荷，不是网络实测速率。",
        "",
        "| 数据集 | 缺失率 | 种子 | ACC | NMI | ARI | 最佳轮/总轮 | 实际缺失样本率 | 实际缺失单元格率 | 上行 MB | 稳定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['rate']:.1f} | {row['seed']} | "
            f"{row['acc']:.6f} | {row['nmi']:.6f} | {row['ari']:.6f} | "
            f"{row['best_round']}/{row['rounds']} | {row['missing_sample_rate']:.4f} | "
            f"{row['missing_cell_rate']:.4f} | {row['uplink_bytes'] / 1e6:.3f} | "
            f"{'是' if row['stable'] else '否'} |"
        )
    lines.extend([
        "",
        f"已完成 {len(rows)} 项 Attention＋Stage/EF 实验。每项完整配置、掩码哈希与逐轮记录"
        "分别保存在表内对应目录的 `summary.json` 和 `history.json`。",
        "",
    ])
    unstable = [row for row in rows if not row["stable"]]
    if unstable:
        lines.append("稳定性未达标：" + "、".join(
            f"{row['dataset']} / r={row['rate']:g} / seed={row['seed']}" for row in unstable
        ) + "。")
        lines.append("")
    repeated = {}
    for row in rows:
        repeated.setdefault((row["dataset"], row["rate"]), []).append(row)
    repeated = [(key, group) for key, group in sorted(repeated.items()) if len(group) >= 2]
    if repeated:
        lines.extend([
            "## 多种子描述统计",
            "",
            "下表是各 seed 最佳轮指标的均值±样本标准差；仅覆盖实际完成的条件，"
            "不能代替独立测试集或显著性检验。",
            "",
            "| 数据集 | 缺失率 | 种子数 | ACC | NMI | ARI |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for (dataset, rate), group in repeated:
            def mean_std(key):
                values = [row[key] for row in group]
                return f"{statistics.mean(values):.6f}±{statistics.stdev(values):.6f}"
            lines.append(
                f"| {dataset} | {rate:.1f} | {len(group)} | "
                f"{mean_std('acc')} | {mean_std('nmi')} | {mean_std('ari')} |"
            )
        lines.append("")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    datasets = args.dataset or sorted(DATASETS)
    rates = args.rate or list(RATES)
    seeds = args.seed or [42]
    root = resolve_project_path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    existing = root / "summary.json"
    rows = {}
    for row in (_read(existing) if existing.exists() else []):
        path = Path(row["path"])
        if (row.get("method") == METHOD and row.get("compression") == COMPRESSION
                and (path / "summary.json").exists() and (path / "history.json").exists()):
            rows[(row["dataset"], row["rate"], row["seed"])] = row
    jobs = [(dataset, rate, seed) for dataset in datasets for rate in rates for seed in seeds]
    for index, (dataset, rate, seed) in enumerate(jobs, 1):
        config, output = _expected_config(DATASETS[dataset], root, rate, seed, args.device)
        result_dir = output / dataset
        summary_path, history_path = result_dir / "summary.json", result_dir / "history.json"
        if summary_path.exists() and history_path.exists() and not args.force:
            if not _compatible(_read(summary_path)["config"], config):
                raise ValueError(f"Existing result has different configuration: {result_dir}")
            print(f"[{index}/{len(jobs)}] Reuse {result_dir}", flush=True)
        else:
            print(f"[{index}/{len(jobs)}] Run {dataset} attention stage_ef "
                  f"rate={rate} seed={seed}", flush=True)
            run(config)
        rows[(dataset, rate, seed)] = _row(result_dir, rate, seed)
        _report(root, list(rows.values()))
    print(f"Completed {len(jobs)} requested experiments; {len(rows)} retained. "
          f"Summary: {root / 'summary.md'}")


if __name__ == "__main__":
    main()
