#!/usr/bin/env python
"""提供训练后绘图和读取已有历史绘图两种命令行用法。"""

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = PROJECT_ROOT / "system"
VISUALIZATION_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIRECTORY = VISUALIZATION_DIR / "result"
DEFAULT_PLOT_DIRECTORY = VISUALIZATION_DIR / "output"
if str(PROJECT_ROOT) not in sys.path:
    # 支持从任意工作目录直接运行该文件，同时复用项目训练入口。
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SYSTEM_DIR) not in sys.path:
    sys.path.insert(0, str(SYSTEM_DIR))

from system.config import DEFAULT_CONFIG_DIR, apply_overrides, load_config, resolve_project_path
from system.main import run as run_training
from visualization.plot_history import visualize_history


def run(config, skip_training=False, results_directory=None, plot_directory=None, image_format="png", dpi=160):
    """按配置训练或读取已有记录，生成聚类指标图和损失图。

    Args:
        config: 已加载的实验配置；函数会更新其中的 output.directory。
        skip_training: 为 True 时只读取已有 history.json，不启动训练。
        results_directory: 结果根目录，其下按数据集名称定位运行记录。
        plot_directory: 图像根目录；未提供时使用 visualization/output。
        image_format: 输出图片格式，由绘图后端保存。
        dpi: 保存图像时使用的分辨率。

    Returns:
        包含 metrics 和 losses 两个文件路径的字典。

    Raises:
        FileNotFoundError: 指定数据集目录中没有可读取的训练历史。
    """
    dataset = config["dataset"]["name"]
    result_root = Path(results_directory) if results_directory else DEFAULT_RESULTS_DIRECTORY
    if not result_root.is_absolute():
        result_root = resolve_project_path(result_root)
    config["output"]["directory"] = str(result_root)

    if not skip_training:
        result = run_training(config)
        result_root = Path(result["config"]["output"]["directory"])

    run_directory = result_root / dataset
    history_path = run_directory / "history.json"
    summary_path = run_directory / "summary.json"
    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}. "
            "Run without --skip-training or select the correct --results-directory."
        )

    plot_root = Path(plot_directory) if plot_directory else DEFAULT_PLOT_DIRECTORY
    if not plot_root.is_absolute():
        plot_root = PROJECT_ROOT / plot_root
    paths = visualize_history(
        history_path,
        plot_root / dataset,
        summary_path=summary_path,
        image_format=image_format,
        dpi=dpi,
    )
    for label, path in paths.items():
        print(f"{label.capitalize()} visualization: {path}")
    return paths


def parse_args():
    """解析数据集配置、训练开关以及绘图格式等命令行参数。"""
    parser = argparse.ArgumentParser(description="Run and visualize federated multi-view clustering")
    parser.add_argument("--config", default="Scene-15", help="Config name or JSON path")
    parser.add_argument("--override", action="append", default=[], help="Override dotted key, e.g. training.rounds=5")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override training.device")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--skip-training", action="store_true", help="Visualize an existing history.json")
    parser.add_argument("--results-directory", help="Training result root; defaults to visualization/result")
    parser.add_argument("--plot-directory", help="Plot output root; defaults to visualization/output")
    parser.add_argument("--format", choices=["png", "pdf", "svg"], default="png", dest="image_format")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_configs:
        print("\n".join(path.stem for path in sorted(DEFAULT_CONFIG_DIR.glob("*.json"))))
        raise SystemExit(0)
    overrides = list(args.override)
    if args.device:
        overrides.append(f'training.device="{args.device}"')
    configuration = apply_overrides(load_config(args.config), overrides)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    run(
        configuration,
        skip_training=args.skip_training,
        results_directory=args.results_directory,
        plot_directory=args.plot_directory,
        image_format=args.image_format,
        dpi=args.dpi,
    )
