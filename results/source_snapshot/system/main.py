#!/usr/bin/env python
"""单数据集训练入口：设置随机种子、加载多视图数据并启动联邦服务端。"""

import argparse
import json
import os
import random

# 在服务端间接导入 sklearn 前限制线程数，稳定 KMeans 执行并避免 Windows/MKL 线程过量。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import torch

from config import (ALGORITHM_VERSION, DEFAULT_CONFIG_DIR, apply_overrides,
                    ensure_writable_output, load_config, resolve_project_path)
from flcore.servers.servercluster import FederatedMultiViewClusteringServer
from utils.mat_data import load_multiview_mat


def seed_everything(seed):
    """设置 Python、NumPy 和 PyTorch 的随机种子，包括可用的 CUDA 设备。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config):
    """按配置运行一次两阶段联邦聚类实验。

    Args:
        config: 已加载并校验的实验配置；输出目录会原地转换为绝对路径。

    Returns:
        dict: 服务端返回的实验摘要，包括聚类指标等运行结果。

    Raises:
        ValueError: MAT 数据中的类别数量与配置不一致。
    """
    # 先校验输出和协议，再加载数据或启动训练，防止新算法改写历史结果。
    config["output"]["directory"] = str(ensure_writable_output(config["output"]["directory"]))
    if config.get("experiment", {}).get("method") in {"A2", "missing2"}:
        raise ValueError("Archived A2/missing2 protocols are read-only; use run_incomplete.py")
    if config.get("algorithm_version", ALGORITHM_VERSION) != ALGORITHM_VERSION:
        raise ValueError("Current code executes completion-v1, not an archived algorithm")
    config["algorithm_version"] = ALGORITHM_VERSION
    training = config["training"]
    seed_everything(int(training["seed"]))
    requested_device = training.get("device", "cuda")
    device = torch.device(requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset_config = config["dataset"]
    dataset_path = resolve_project_path(dataset_config["file"])
    # 零率与正率统一交给客户端按真实观测行归一化，完整视图只是全真掩码。
    data = load_multiview_mat(
        dataset_path,
        name=dataset_config["name"],
        normalization="none",
    )
    if data.num_clusters != int(dataset_config["num_clusters"]):
        raise ValueError(
            f"Config expects {dataset_config['num_clusters']} clusters, MAT file contains {data.num_clusters}"
        )
    print(
        f"Dataset={data.name} samples={data.num_samples} views={data.view_dims} "
        f"clusters={data.num_clusters} clients={dataset_config['num_clients']} device={device}"
    )
    server = FederatedMultiViewClusteringServer(data, config, device)
    result = server.train()
    print(json.dumps(result["metrics"], indent=2))
    return result


def parse_args():
    """解析单数据集名称、配置覆盖、设备选择和配置列表查询参数。"""
    parser = argparse.ArgumentParser(description="Federated multi-view clustering")
    parser.add_argument("--config", default="Scene-15", help="Config name or JSON path")
    parser.add_argument("--override", action="append", default=[], help="Override dotted key, e.g. training.rounds=5")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override training.device")
    parser.add_argument("--list-configs", action="store_true")
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
    run(configuration)
