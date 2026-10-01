#!/usr/bin/env python
import argparse
import json
import os
import random

# Keep scikit-learn KMeans deterministic and avoid the known Windows/MKL
# over-subscription warning before importing sklearn through the server module.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import numpy as np
import torch

from config import DEFAULT_CONFIG_DIR, apply_overrides, load_config, resolve_project_path
from flcore.servers.servercluster import FederatedMultiViewClusteringServer
from utils.mat_data import load_multiview_mat


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(config):
    training = config["training"]
    seed_everything(int(training["seed"]))
    requested_device = training.get("device", "cuda")
    device = torch.device(requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu")
    dataset_config = config["dataset"]
    dataset_path = resolve_project_path(dataset_config["file"])
    data = load_multiview_mat(
        dataset_path,
        name=dataset_config["name"],
        normalization=(
            "none" if config.get("missing", {}).get("enabled", False)
            else dataset_config.get("normalization", "standard")
        ),
    )
    if data.num_clusters != int(dataset_config["num_clusters"]):
        raise ValueError(
            f"Config expects {dataset_config['num_clusters']} clusters, MAT file contains {data.num_clusters}"
        )
    config["output"]["directory"] = str(resolve_project_path(config["output"]["directory"]))
    print(
        f"Dataset={data.name} samples={data.num_samples} views={data.view_dims} "
        f"clusters={data.num_clusters} clients={dataset_config['num_clients']} device={device}"
    )
    server = FederatedMultiViewClusteringServer(data, config, device)
    result = server.train()
    print(json.dumps(result["metrics"], indent=2))
    return result


def parse_args():
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
