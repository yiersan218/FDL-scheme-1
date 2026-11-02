"""依次运行配置目录中的数据集，并生成最佳轮与真实末轮指标汇总。"""

import argparse
import json

from config import DEFAULT_CONFIG_DIR, apply_overrides, ensure_writable_output, load_config
from main import run
from summarize_results import write_summary


def parse_args():
    """解析批量运行的设备、结果目录及共享配置覆盖参数。"""
    parser = argparse.ArgumentParser(description="Run all federated clustering datasets")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--results-dir", default="results-补全聚类")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.results_dir = str(ensure_writable_output(args.results_dir))
    summaries = {}
    # 仅发现当前配置根目录中的 JSON，不递归收集旧实验配置或结果快照。
    config_paths = sorted(DEFAULT_CONFIG_DIR.glob("*.json"))
    for path in config_paths:
        overrides = list(args.override) + [
            f'training.device="{args.device}"',
            f'output.directory={json.dumps(args.results_dir, ensure_ascii=False)}',
        ]
        config = apply_overrides(load_config(path), overrides)
        print(f"\n{'=' * 20} {path.stem} {'=' * 20}")
        summaries[path.stem] = run(config)["metrics"]
    ensure_writable_output(ensure_writable_output(args.results_dir) / "summary.md")
    summary_path, _rows = write_summary(
        args.results_dir,
        expected_datasets=summaries,
        title="completion-v1 两阶段补全聚类结果（初始配置）",
    )
    print("\nFinal metrics")
    print(json.dumps(summaries, indent=2))
    print(f"Markdown summary: {summary_path}")
