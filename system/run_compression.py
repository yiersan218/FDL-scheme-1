"""Reproducible fixed-uplink-budget compression search and reporting."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from config import BACKUP_CONFIG_DIR, DEFAULT_CONFIG_DIR, apply_overrides, load_config, resolve_project_path
from main import run
from summarize_results import collect_result, write_summary


CANDIDATES = ("none", "topk", "topk_ef", "paper", "stage", "stage_ef", "stage_task")


def parse_args():
    parser = argparse.ArgumentParser(description="Tune model uplink compression at a fixed byte budget")
    parser.add_argument("--results-dir", default="results-压缩")
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--candidate", action="append", choices=CANDIDATES, dest="candidates")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--selection", choices=("stage-ef", "uniform", "per-dataset"), default="stage-ef")
    parser.add_argument("--force", action="store_true", help="Rerun existing matching candidate outputs")
    return parser.parse_args()


def _read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _settings(name):
    if name == "none":
        return {"method": "none"}
    if name == "topk_ef":
        return {"method": "topk", "error_feedback": True}
    if name == "stage_ef":
        return {"method": "stage", "candidates": 4, "view_weight": 0.1,
                "pair_weight": 0.1, "error_feedback": True}
    if name == "stage_task":
        return {"method": "stage", "candidates": 4, "view_weight": 0.0, "pair_weight": 0.0}
    if name == "stage":
        return {"method": "stage", "candidates": 4, "view_weight": 0.1, "pair_weight": 0.1}
    return {"method": name}


def _stable(summary, history):
    best = next(row["clustering"] for row in history if row["round"] == summary["best_round"])
    last = history[-1]["clustering"]
    return all(abs(last[key] - best[key]) <= 0.01 for key in ("acc", "nmi", "ari"))


def _publish_stage_ef(root, config_paths, budget_ratio, device, force):
    """Publish the user-selected single method and its matched Top-k+EF comparison."""
    results = {}
    for config_path in config_paths:
        dataset = config_path.stem
        for candidate in ("topk_ef", "stage_ef"):
            results[(dataset, candidate)] = _one_run(
                config_path, root, candidate, budget_ratio, device, force
            )
        summary, history, source_root = results[(dataset, "stage_ef")]
        destination = root / dataset
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / dataset / "history.json", destination / "history.json")
        formal = copy.deepcopy(summary)
        formal["config"]["output"]["directory"] = str(root)
        formal["compression_search"] = {
            "selected_candidate": "stage_ef",
            "selection_scope": "user_selected_uniform",
            "source_directory": str(source_root / dataset),
            "budget_ratio": budget_ratio,
            "stable": _stable(summary, history),
        }
        _write_json(destination / "summary.json", formal)

    datasets = [path.stem for path in config_paths]
    formal_rows = [collect_result(root / dataset) for dataset in datasets]
    summary_lines = [
        "# Stage＋误差反馈正式运行结果", "",
        f"- 结果目录：`{root}`",
        f"- 统一方法：Stage＋客户端误差反馈；实际编码上行预算为稠密上传的 {budget_ratio:.0%}。",
        "- 随机种子：42；模型选择指标：NMI。",
        "- 指标变化定义为“最后一轮减最佳轮”；负值表示末轮低于最佳轮。", "",
        "| 数据集 | ACC | NMI | ARI | 最佳轮/总轮数 | ΔACC | ΔNMI | ΔARI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in formal_rows:
        best, delta = row["best"], row["delta"]
        summary_lines.append(
            f"| {row['dataset']} | {best['acc']:.6f} | {best['nmi']:.6f} | "
            f"{best['ari']:.6f} | {row['best_round']}/{row['configured_rounds']} | "
            f"{delta['acc']:+.6f} | {delta['nmi']:+.6f} | {delta['ari']:+.6f} |"
        )
    stable_count = sum(_stable(*results[(dataset, "stage_ef")][:2]) for dataset in datasets)
    summary_lines.extend([
        "", f"{stable_count}/{len(datasets)} 个数据集满足最佳轮到末轮的 ACC/NMI/ARI "
        "绝对变化均不超过 0.01。逐轮结果与完整配置见各数据集的 `history.json`、`summary.json`；"
        "与 Top-k_ef 的同预算对比见 `tuning_summary.md`。", "",
    ])
    (root / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    comparison_lines = [
        "# Stage＋误差反馈与 Top-k_ef 同预算对比", "",
        f"- {len(datasets)} 个数据集统一方法，实际编码上行预算 {budget_ratio:.0%}，seed=42；两种方法均开启客户端误差反馈。",
        "- 指标取各自 NMI 最佳轮；Δ = Stage＋误差反馈 − Top-k_ef。",
        "- 当前正式方法按研究方案选择 Stage＋误差反馈，并非按本表平均 NMI 选优。", "",
        "| 数据集 | Stage NMI | Top-k_ef NMI | ΔNMI | ΔACC | ΔARI | Stage 上行 MB | Top-k_ef 上行 MB | 压缩秒数 Stage / Top-k_ef |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = {candidate: {"nmi": 0.0, "acc": 0.0, "ari": 0.0, "up": 0,
                          "dense_up": 0, "seconds": 0.0} for candidate in ("stage_ef", "topk_ef")}
    wins = 0
    for dataset in datasets:
        stage = results[(dataset, "stage_ef")][0]
        topk = results[(dataset, "topk_ef")][0]
        for candidate, item in (("stage_ef", stage), ("topk_ef", topk)):
            total = totals[candidate]
            for metric in ("nmi", "acc", "ari"):
                total[metric] += item["metrics"][metric]
            total["up"] += item["communication"]["uplink_bytes"]
            total["dense_up"] += item["communication"]["dense_uplink_bytes"]
            total["seconds"] += item["communication"]["compression_seconds"]
        wins += stage["metrics"]["nmi"] > topk["metrics"]["nmi"]
        comparison_lines.append(
            f"| {dataset} | {stage['metrics']['nmi']:.6f} | {topk['metrics']['nmi']:.6f} | "
            f"{stage['metrics']['nmi']-topk['metrics']['nmi']:+.6f} | "
            f"{stage['metrics']['acc']-topk['metrics']['acc']:+.6f} | "
            f"{stage['metrics']['ari']-topk['metrics']['ari']:+.6f} | "
            f"{stage['communication']['uplink_bytes']/1e6:.3f} | "
            f"{topk['communication']['uplink_bytes']/1e6:.3f} | "
            f"{stage['communication']['compression_seconds']:.2f} / "
            f"{topk['communication']['compression_seconds']:.2f} |"
        )
    stage, topk = totals["stage_ef"], totals["topk_ef"]
    n = len(datasets)
    comparison_lines.extend([
        "", "| 方法 | 平均 ACC | 平均 NMI | 平均 ARI | 累计上行 MB | 上行减少 | 累计压缩秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, total in (("Stage＋误差反馈", stage), ("Top-k_ef", topk)):
        comparison_lines.append(
            f"| {label} | {total['acc']/n:.6f} | {total['nmi']/n:.6f} | "
            f"{total['ari']/n:.6f} | {total['up']/1e6:.3f} | "
            f"{1-total['up']/total['dense_up']:.2%} | {total['seconds']:.2f} |"
        )
    comparison_lines.extend([
        "", f"Stage＋误差反馈在 {wins}/{n} 个数据集的 NMI 较高，"
        f"但七集等权平均 NMI 比 Top-k_ef 低 {topk['nmi']/n-stage['nmi']/n:.6f}，"
        "压缩计算耗时也更高。该结果支持把它作为待验证的创新方案，"
        "不支持声称目前整体性能优于 Top-k_ef。",
        "上行字节为单进程模拟中的编码模型载荷（含索引、数值、中心与计数），"
        "不是实测网络流量或延迟；下行未压缩。仅有单种子，同一评估集上的差异不能证明统计显著性。",
        "原始候选结果见 `tuning/stage_ef-b50/` 与 `tuning/topk_ef-b50/`。", "",
    ])
    (root / "tuning_summary.md").write_text("\n".join(comparison_lines), encoding="utf-8")
    print(f"Published Stage+EF results for {n} datasets in {root}", flush=True)


def _one_run(config_path, root, candidate, budget_ratio, device, force):
    settings = _settings(candidate)
    candidate_root = root / "tuning" / f"{candidate}-b{int(round(budget_ratio * 100))}"
    summary_path = candidate_root / config_path.stem / "summary.json"
    history_path = summary_path.with_name("history.json")
    if summary_path.exists() and history_path.exists() and not force:
        existing = _read_json(summary_path)
        compression = existing["config"].get("compression", {})
        if (compression.get("method") == settings["method"]
                and (candidate == "none" or abs(compression.get("budget_ratio", 0) - budget_ratio) < 1e-9)
                and all(compression.get(key) == value for key, value in settings.items())):
            print(f"Reuse {candidate} {config_path.stem}", flush=True)
            return existing, _read_json(history_path), candidate_root
    overrides = [
        f"compression.method={json.dumps(settings['method'])}",
        f"compression.budget_ratio={budget_ratio}",
        f"training.device={json.dumps(device)}",
        f"output.directory={json.dumps(str(candidate_root), ensure_ascii=False)}",
    ]
    for key, value in settings.items():
        if key != "method":
            overrides.append(f"compression.{key}={json.dumps(value)}")
    config = apply_overrides(load_config(config_path), overrides)
    print(f"Run {candidate} {config_path.stem} at budget {budget_ratio:.2f}", flush=True)
    run(config)
    return _read_json(summary_path), _read_json(history_path), candidate_root


def _comparison(root, chosen, results, budget_ratio, selection):
    lines = [
        "# 压缩通信与聚类性能对比",
        "",
        f"固定目标：模型上行编码字节不超过稠密上传的 {budget_ratio:.0%}；"
        + (f"{len(chosen)} 个数据集统一使用平均 NMI 更高的方法。" if selection == "uniform"
           else "每个数据集选择稳定候选中 NMI 最高者。"),
        "稠密基线与压缩候选均在当前代码、相同配置及 seed=42 下运行。指标为最佳聚类阶段检查点。",
        "",
        "| 数据集 | 选定方案 | 基线 NMI | 压缩 NMI | ΔNMI | 基线上行 MB | 压缩上行 MB | 上行减少 | 总模型传输减少 | 最佳轮/总轮 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    detailed = [
        "", "## 聚类质量与计算代价", "",
        "| 数据集 | ΔACC | ΔARI | 基线运行秒数 | 压缩运行秒数 | 其中压缩秒数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    total_dense_up = total_up = total_dense_down = total_down = 0
    for dataset in sorted(chosen):
        name = chosen[dataset]
        dense, dense_history, _ = results[(dataset, "none")]
        summary, history, _ = results[(dataset, name)]
        dense_up = dense["communication"]["uplink_bytes"]
        up = summary["communication"]["uplink_bytes"]
        dense_down = dense["communication"]["downlink_bytes"]
        down = summary["communication"]["downlink_bytes"]
        total_dense_up += dense_up
        total_up += up
        total_dense_down += dense_down
        total_down += down
        nmi_delta = summary["metrics"]["nmi"] - dense["metrics"]["nmi"]
        lines.append(
            f"| {dataset} | {name} | {dense['metrics']['nmi']:.6f} | {summary['metrics']['nmi']:.6f} | "
            f"{nmi_delta:+.6f} | {dense_up/1e6:.3f} | {up/1e6:.3f} | "
            f"{1-up/dense_up:.2%} | {1-(up+down)/(dense_up+down):.2%} | "
            f"{summary['best_round']}/{summary['config']['training']['rounds']} |"
        )
        detailed.append(
            f"| {dataset} | {summary['metrics']['acc']-dense['metrics']['acc']:+.6f} | "
            f"{summary['metrics']['ari']-dense['metrics']['ari']:+.6f} | "
            f"{dense_history[-1]['elapsed_seconds']:.2f} | {history[-1]['elapsed_seconds']:.2f} | "
            f"{summary['communication']['compression_seconds']:.2f} |"
        )
    lines.extend([
        "",
        f"{len(chosen)} 个数据集汇总：上行 {total_dense_up/1e6:.3f} MB → {total_up/1e6:.3f} MB "
        f"（减少 {1-total_up/total_dense_up:.2%}）；上下行合计减少 "
        f"{1-(total_up+total_down)/(total_dense_up+total_dense_down):.2%}。",
        *detailed,
        "",
        "上行包含编码后的普通参数、聚类中心、软簇计数和中心初始化摘要；下行按完整全局模型参数字节数估算，尚未压缩。训练诊断指标与传输协议外壳不计入，故结果是**模型载荷字节数**，不是实测网络流量或延迟。",
        "运行秒数由单进程模拟记录，包含训练、评估和压缩，受当前硬件与负载影响；不能据此声称网络传输加速。逐轮通信量见各数据集的 `history.json`；全部候选及稳定性见 `tuning_summary.md`。",
        "候选由这些数据集的评估 NMI 事后选择，适用于当前工程调参记录，不是无偏的泛化评估；正式论文需固定超参数后增加独立种子或验证集。",
        "若压缩方案的聚类指标低于稠密基线，应如实解释为通信与精度的权衡，不宣称双重提升。",
        "",
    ])
    (root / "通信与性能对比.md").write_text("\n".join(lines), encoding="utf-8")


def _uniform_comparison(root, results, datasets, budget_ratio):
    methods = ("topk_ef", "stage")
    statistics = {}
    for method in methods:
        summaries = [results[(dataset, method)][0] for dataset in datasets]
        histories = [results[(dataset, method)][1] for dataset in datasets]
        statistics[method] = {
            "mean_nmi": sum(item["metrics"]["nmi"] for item in summaries) / len(datasets),
            "mean_acc": sum(item["metrics"]["acc"] for item in summaries) / len(datasets),
            "mean_ari": sum(item["metrics"]["ari"] for item in summaries) / len(datasets),
            "compression_seconds": sum(item["communication"]["compression_seconds"] for item in summaries),
            "uplink_bytes": sum(item["communication"]["uplink_bytes"] for item in summaries),
            "dense_uplink_bytes": sum(item["communication"]["dense_uplink_bytes"] for item in summaries),
            "all_stable": all(_stable(item, history) for item, history in zip(summaries, histories)),
        }
    eligible = [method for method in methods if statistics[method]["all_stable"]]
    if not eligible:
        raise ValueError("Neither uniform candidate satisfies the stability criterion on all datasets")
    winner = max(eligible, key=lambda method: (
        statistics[method]["mean_nmi"], -statistics[method]["compression_seconds"]
    ))
    lines = [
        "# Top-k_ef 与 Stage 统一方案对比", "",
        f"比较范围：{len(datasets)} 个数据集，seed=42，实际编码上行预算为稠密上传的 {budget_ratio:.0%}。"
        "两种方法均对全部数据集使用同一算法，不逐数据集切换。",
        "统一选择规则：先要求所有数据集达到最佳轮至末轮的 ACC/NMI/ARI 变化均不超过 0.01，"
        f"再比较 {len(datasets)} 个数据集等权平均 NMI；平局时选择压缩计算时间更低者。", "",
        "| 数据集 | Top-k_ef NMI | Stage NMI | Top-k_ef − Stage | Top-k_ef ACC/ARI | Stage ACC/ARI | 上行 MB（两者相同） | 压缩耗时秒：EF / Stage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    wins = {method: 0 for method in methods}
    for dataset in datasets:
        ef = results[(dataset, "topk_ef")][0]
        stage = results[(dataset, "stage")][0]
        delta = ef["metrics"]["nmi"] - stage["metrics"]["nmi"]
        if delta > 0:
            wins["topk_ef"] += 1
        elif delta < 0:
            wins["stage"] += 1
        lines.append(
            f"| {dataset} | {ef['metrics']['nmi']:.6f} | {stage['metrics']['nmi']:.6f} | {delta:+.6f} | "
            f"{ef['metrics']['acc']:.6f}/{ef['metrics']['ari']:.6f} | "
            f"{stage['metrics']['acc']:.6f}/{stage['metrics']['ari']:.6f} | "
            f"{ef['communication']['uplink_bytes']/1e6:.3f} | "
            f"{ef['communication']['compression_seconds']:.2f} / "
            f"{stage['communication']['compression_seconds']:.2f} |"
        )
    lines.extend([
        "", "| 统一方法 | 平均 NMI | 平均 ACC | 平均 ARI | NMI 胜出数据集 | 累计压缩秒数 | 全部稳定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for method in methods:
        item = statistics[method]
        lines.append(
            f"| {method} | {item['mean_nmi']:.6f} | {item['mean_acc']:.6f} | "
            f"{item['mean_ari']:.6f} | {wins[method]}/{len(datasets)} | "
            f"{item['compression_seconds']:.2f} | {'是' if item['all_stable'] else '否'} |"
        )
    lines.extend([
        "", f"结论：正式 {len(datasets)} 个数据集统一使用 **{winner}**。Stage 在个别数据集可能更好，"
        "但当前单种子证据不足以抵消其整体 NMI 与计算开销劣势。",
        f"两者上行模型载荷相同，选定方案上行减少 "
        f"{1-statistics[winner]['uplink_bytes']/statistics[winner]['dense_uplink_bytes']:.2%}；"
        "下行未压缩，具体上下行合计减少比例见 `通信与性能对比.md`。"
        "字节数为单进程模拟的模型载荷，不是实测网络流量或延迟。",
        "这是基于同一评估集事后选择的工程结论，正式论文需额外随机种子或独立验证集确认。",
        "完整逐轮数据保留在 `tuning/topk_ef-b50/` 与 `tuning/stage-b50/`；"
        "稠密基线对照见 `通信与性能对比.md`。", "",
    ])
    (root / "统一方法对比.md").write_text("\n".join(lines), encoding="utf-8")
    return winner


def main():
    args = parse_args()
    root = resolve_project_path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_paths = sorted(DEFAULT_CONFIG_DIR.glob("*.json")) + sorted(BACKUP_CONFIG_DIR.glob("*.json"))
    if args.datasets:
        config_paths = [path for path in config_paths if path.stem in args.datasets]
    if not config_paths:
        raise ValueError("No matching dataset configs")
    if args.selection == "stage-ef":
        if args.candidates:
            raise ValueError("Fixed Stage+EF selection always compares stage_ef and topk_ef; omit --candidate")
        _publish_stage_ef(root, config_paths, args.budget_ratio, args.device, args.force)
        return
    candidates = args.candidates or list(CANDIDATES)
    if "none" not in candidates:
        candidates = ["none", *candidates]
    results = {}
    for config_path in config_paths:
        for candidate in candidates:
            results[(config_path.stem, candidate)] = _one_run(
                config_path, root, candidate, args.budget_ratio, args.device, args.force
            )

    datasets = [path.stem for path in config_paths]
    uniform_winner = None
    if args.selection == "uniform":
        if not {"topk_ef", "stage"}.issubset(candidates):
            raise ValueError("Uniform selection requires both topk_ef and stage candidates")
        uniform_winner = _uniform_comparison(root, results, datasets, args.budget_ratio)

    tuning_lines = [
        "# 模型压缩调参记录", "",
        f"- 搜索范围：{', '.join(candidates)}；固定预算比例 {args.budget_ratio:.2f}。",
        "- 使用 seed=42；稳定条件为最佳轮到末轮的 ACC、NMI、ARI 绝对变化都不超过 0.01。",
        (f"- 正式 {len(datasets)} 个数据集统一使用 {uniform_winner}；Top-k_ef 与 Stage 的整体选择依据见 `统一方法对比.md`。"
         if uniform_winner else
         "- 每个数据集在满足稳定条件的压缩候选中按 NMI 选优；若均不稳定，仍记录最高 NMI 并标注。"),
        "", "| 数据集 | 候选 | ACC | NMI | ARI | 上行减少 | 压缩秒数 | 稳定 | 最佳轮/总轮 |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    chosen = {}
    for config_path in config_paths:
        dataset = config_path.stem
        options = []
        for candidate in candidates:
            summary, history, _candidate_root = results[(dataset, candidate)]
            stable = _stable(summary, history)
            communication = summary["communication"]
            tuning_lines.append(
                f"| {dataset} | {candidate} | {summary['metrics']['acc']:.6f} | "
                f"{summary['metrics']['nmi']:.6f} | {summary['metrics']['ari']:.6f} | "
                f"{communication['uplink_reduction']:.2%} | {communication['compression_seconds']:.2f} | "
                f"{'是' if stable else '否'} | {summary['best_round']}/{summary['config']['training']['rounds']} |"
            )
            if candidate != "none":
                options.append((stable, summary["metrics"]["nmi"], -communication["uplink_bytes"], candidate))
        winner = uniform_winner or max(options)[-1]
        chosen[dataset] = winner
        summary, history, candidate_root = results[(dataset, winner)]
        destination = root / dataset
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_root / dataset / "history.json", destination / "history.json")
        formal = copy.deepcopy(summary)
        formal["config"]["output"]["directory"] = str(root)
        formal["compression_search"] = {
            "selected_candidate": winner,
            "selection_scope": args.selection,
            "source_directory": str(candidate_root / dataset),
            "budget_ratio": args.budget_ratio,
            "stable": _stable(summary, history),
        }
        _write_json(destination / "summary.json", formal)
        tuning_lines.append(f"| {dataset} | **选定：{winner}** | | | | | | | |")
    (root / "tuning_summary.md").write_text("\n".join(tuning_lines) + "\n", encoding="utf-8")
    write_summary(root, expected_datasets=chosen, title="固定上行预算模型压缩结果")
    _comparison(root, chosen, results, args.budget_ratio, args.selection)
    print(f"Selected: {chosen}", flush=True)
    print(f"Reports: {root / 'summary.md'}, {root / 'tuning_summary.md'}, {root / '通信与性能对比.md'}", flush=True)


if __name__ == "__main__":
    main()
