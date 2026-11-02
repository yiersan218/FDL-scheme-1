"""统一运行 completion-v1 缺失率矩阵和三组补全消融，隔离历史结果。"""

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from config import (ALGORITHM_VERSION, DEFAULT_CONFIG_DIR, PROJECT_ROOT,
                    apply_overrides, ensure_writable_output, load_config,
                    resolve_project_path)
from flcore.communication.codec import dense_uplink_bytes
from flcore.trainmodel.multiview import MultiViewClusteringModel
from main import run
from utils.mat_data import load_multiview_mat


VARIANTS = ("baseline", "observed_completion", "completed")
SOURCE_FILES = (
    "system/config.py", "system/main.py", "system/utils/mat_data.py",
    "system/flcore/trainmodel/multiview.py", "system/flcore/trainmodel/completion.py",
    "system/flcore/trainmodel/losses.py",
    "system/flcore/clients/clientcluster.py", "system/flcore/servers/servercluster.py",
    "system/flcore/communication/__init__.py", "system/flcore/communication/codec.py",
    "system/flcore/communication/compression.py", "system/run_incomplete.py",
)


def parse_args(argv=None):
    """解析同一配置、掩码规则和通信设置下的缺失率与消融组合。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(p.stem for p in DEFAULT_CONFIG_DIR.glob("*.json")),
                        default=sorted(p.stem for p in DEFAULT_CONFIG_DIR.glob("*.json")))
    parser.add_argument("--rates", nargs="+", type=float, default=[0.0, 0.1, 0.3, 0.5, 0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["completed"])
    parser.add_argument("--compression", choices=("none", "stage_ef", "topk_ef"), default="none")
    parser.add_argument("--budget-ratio", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--results-dir", default="results-补全聚类-矩阵")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true", help="只保存计划和源码快照，不训练")
    args = parser.parse_args(argv)
    if any(not math.isfinite(rate) or not 0 <= rate < 1 for rate in args.rates):
        parser.error("--rates must be in [0, 1)")
    if not 0 < args.budget_ratio <= 1:
        parser.error("--budget-ratio must be in (0, 1]")
    for field in ("datasets", "rates", "seeds", "variants"):
        setattr(args, field, list(dict.fromkeys(getattr(args, field))))
    return args


def digest(value):
    """计算规范 JSON 的摘要，作为计划和运行配置的稳定标识。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_hash(path):
    """分块计算文件摘要，避免一次性读取大型 MAT。"""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    """在经过保护检查的路径写出 UTF-8 JSON。"""
    path = ensure_writable_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fingerprint(config):
    """计算完整运行配置指纹，排除指纹自身。"""
    value = copy.deepcopy(config)
    value.get("experiment", {}).pop("fingerprint", None)
    return digest(value)


def baseline_budgets(config, ratio):
    """从关闭补全的共享模型推导所有候选相同的绝对 pre/post 上行预算。"""
    data_path = resolve_project_path(config["dataset"]["file"])
    data = load_multiview_mat(data_path, normalization="none")
    settings = config["model"]
    with torch.random.fork_rng(devices=[]):
        model = MultiViewClusteringModel(
            data.view_dims, config["dataset"]["num_clusters"], settings["hidden_dims"],
            settings["embedding_dim"], dropout=settings.get("dropout", 0.0),
            alpha=settings.get("student_t_alpha", 1.0),
            student_t_distance_scale=settings.get("student_t_distance_scale", 1.0),
            completion={"enabled": False},
        )
    return {"pre_center": int(dense_uplink_bytes(model, False) * ratio),
            "post_center": int(dense_uplink_bytes(model, True) * ratio)}


def build_config(base, dataset, rate, seed, variant, args, root, source_files, data_hash, budget_pair=None):
    """从同一根配置构造任务，仅按预注册三组开关区分补全作用。"""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config = copy.deepcopy(base)
    tag = format(float(rate), ".12g").replace(".", "p")
    output = ensure_writable_output(root / "runs" / ALGORITHM_VERSION / variant / f"rate-{tag}" / f"seed-{seed}")
    config["missing"].update(enabled=True, rate=float(rate))
    config["training"].update(seed=int(seed), device=args.device)
    config["output"]["directory"] = str(output)
    # 三组均使用观测初始化，避免把初始化变化混进预测融合的消融差值。
    config["completion"].update(
        enabled=variant != "baseline", use_completed_initialization=False,
        use_completed_assignments=variant == "completed",
    )
    config["compression"].update(
        method={"none": "none", "stage_ef": "stage", "topk_ef": "topk"}[args.compression],
        error_feedback=args.compression != "none", budget_ratio=args.budget_ratio,
        budget_bytes=None, budget_bytes_pair=copy.deepcopy(budget_pair),
    )
    config = apply_overrides(config, [])
    config["experiment"] = {
        "method": ALGORITHM_VERSION, "variant": variant, "nominal_missing_rate": float(rate),
        "seed": int(seed), "normalization_scope": "client_observed",
        "initialization": "observed_only", "teacher": "round_fixed_observed",
        "source_files": copy.deepcopy(source_files), "source_sha256": digest(source_files),
        "data_sha256": data_hash, "compression": args.compression,
        "budget_reference": "baseline_shared_model" if budget_pair else "dense_upload",
    }
    config["experiment"]["fingerprint"] = fingerprint(config)
    return config, output / dataset


def prepare_plan(args):
    """只在内存中构造完整计划；写入之前核对所有路径和覆盖项。"""
    root = ensure_writable_output(args.results_dir)
    forbidden = ("missing.", "training.seed", "training.device", "output.", "compression.",
                 "completion.enabled", "completion.use_completed_assignments",
                 "completion.use_completed_initialization", "algorithm_version")
    for expression in args.override:
        key = expression.split("=", 1)[0]
        if any(key == item.rstrip(".") or key.startswith(item) for item in forbidden):
            raise ValueError(f"Use matrix arguments or variant for paired setting: {key}")
    sources = [{"path": path, "sha256": file_hash(PROJECT_ROOT / path)} for path in SOURCE_FILES]
    jobs = []
    for dataset in args.datasets:
        base = apply_overrides(load_config(dataset), args.override)
        data_hash = file_hash(resolve_project_path(base["dataset"]["file"]))
        budget = baseline_budgets(base, args.budget_ratio) if args.compression != "none" else None
        for rate in args.rates:
            for seed in args.seeds:
                for variant in args.variants:
                    config, result_dir = build_config(base, dataset, rate, seed, variant, args, root,
                                                      sources, data_hash, budget)
                    jobs.append({"config": config, "result_dir": str(result_dir)})
    return root, {"algorithm_version": ALGORITHM_VERSION, "expected_runs": len(jobs),
                  "datasets": args.datasets, "rates": args.rates, "seeds": args.seeds,
                  "variants": args.variants, "compression": args.compression,
                  "source_files": sources, "jobs": jobs}


def read_result(job):
    """核验复用结果的算法、配置、历史完整性和最佳轮，返回摘要行。"""
    directory = Path(job["result_dir"])
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    saved = summary["config"]
    expected = job["config"]
    if (summary.get("algorithm_version") != ALGORITHM_VERSION or
            saved.get("algorithm_version") != ALGORITHM_VERSION or
            saved.get("experiment", {}).get("fingerprint") != fingerprint(saved) or
            fingerprint(saved) != fingerprint(expected)):
        raise RuntimeError(f"Existing result has a different algorithm/config/source; use a new directory: {directory}")
    rounds, best_round = int(saved["training"]["rounds"]), int(summary["best_round"])
    if (len(history) != rounds or not 1 <= best_round <= rounds or
            [row.get("round") for row in history] != list(range(1, rounds + 1)) or
            history[best_round - 1].get("phase") != "clustering"):
        raise RuntimeError(f"Incomplete or invalid round history: {directory}")
    best, last = summary["metrics"], history[-1]["clustering"]
    for metric in ("acc", "nmi", "ari"):
        if (not all(math.isfinite(float(value[metric])) for value in (best, last)) or
                abs(float(best[metric]) - float(history[best_round - 1]["clustering"][metric])) > 1e-10 or
                abs(float(last[metric]) - float(summary["last_metrics"][metric])) > 1e-10):
            raise RuntimeError(f"Invalid best/last metrics: {directory}")
    meta = saved["experiment"]
    return {"dataset": summary["dataset"], "variant": meta["variant"], "rate": meta["nominal_missing_rate"],
            "seed": meta["seed"], "best_round": best_round, "rounds": rounds, "best": best, "last": last,
            "communication": summary.get("communication", {}), "missing": summary.get("missing", {}),
            "compression": meta["compression"], "budget_bytes_pair": saved["compression"]["budget_bytes_pair"],
            "summary_path": str(directory / "summary.json"), "summary_sha256": file_hash(directory / "summary.json"),
            "history_sha256": file_hash(directory / "history.json")}


def write_report(root, rows, expected_runs):
    """写出新协议自己的完成进度、最佳轮和真实末轮，绝不覆盖旧报告。"""
    groups = {}
    for row in rows:
        key = row["dataset"], row["rate"], row["seed"]
        previous = groups.setdefault(key, row)
        if (not row["missing"].get("mask_sha256") or
                row["missing"].get("mask_sha256") != previous["missing"].get("mask_sha256") or
                row["budget_bytes_pair"] != previous["budget_bytes_pair"] or
                row["compression"] != previous["compression"]):
            raise RuntimeError(f"Paired runs differ in mask or compression budget: {key}")
    write_json(root / "summary.json", rows)
    lines = ["# completion-v1 缺失联邦聚类结果", "", f"已完成 {len(rows)}/{expected_runs} 次运行。", "",
             "各缺失率统一使用客户端观测归一化及所选通信方案；三组统一使用观测初始化。",
             "Best 为聚类阶段按配置指标选出的同一轮，Last 来自真实历史末轮。", "",
             "| 数据集 | 方法 | 缺失率 | Seed | Best轮/总轮 | Best ACC/NMI/ARI | Last ACC/NMI/ARI |",
             "|---|---|---:|---:|---|---|---|"]
    for row in rows:
        best = "/".join(f"{row['best'][name]:.6f}" for name in ("acc", "nmi", "ari"))
        last = "/".join(f"{row['last'][name]:.6f}" for name in ("acc", "nmi", "ari"))
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['rate']:g} | {row['seed']} | "
                     f"{row['best_round']}/{row['rounds']} | {best} | {last} |")
    ensure_writable_output(root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    """保存不可混用的计划和源码快照，再执行或严格复用当前版本结果。"""
    args = parse_args(argv)
    root, plan = prepare_plan(args)
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("Existing plan differs; choose a new --results-dir")
    write_json(plan_path, plan)
    for source in plan["source_files"]:
        origin = PROJECT_ROOT / source["path"]
        target = ensure_writable_output(root / "source_snapshot" / source["path"])
        if file_hash(origin) != source["sha256"]:
            raise RuntimeError("Source changed during planning; prepare a new plan")
        if target.exists() and file_hash(target) != source["sha256"]:
            raise RuntimeError("Existing source snapshot differs; choose a new directory")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
    if args.plan_only:
        print(f"Prepared {len(plan['jobs'])} completion-v1 jobs without training: {plan_path}")
        return plan
    rows = []
    for job in plan["jobs"]:
        if any(file_hash(PROJECT_ROOT / source["path"]) != source["sha256"] for source in plan["source_files"]):
            raise RuntimeError("Source changed after planning; use a new process and output directory")
        directory = ensure_writable_output(job["result_dir"])
        exists = [(directory / name).exists() for name in ("summary.json", "history.json")]
        if any(exists) and not all(exists):
            raise RuntimeError(f"Partial result exists; choose a new output directory: {directory}")
        if not any(exists):
            run(copy.deepcopy(job["config"]))
        rows.append(read_result(job))
        write_report(root, rows, plan["expected_runs"])
    return rows


if __name__ == "__main__":
    main()
