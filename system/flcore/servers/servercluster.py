import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from flcore.clients.clientcluster import FederatedClusteringClient
from flcore.compression import dense_uplink_bytes, reconstruct_state
from flcore.trainmodel.multiview import MultiViewClusteringModel
from utils.clustering_metrics import evaluate_clustering
from utils.mat_data import MultiViewSubset, balanced_client_indices


def progress_report_gap(total_rounds):
    """Report training progress after roughly every 10% of all rounds."""
    return max(1, int(np.ceil(int(total_rounds) / 10.0)))


def pretraining_clustering_scale(round_number, center_init_round, pretrain_rounds):
    """Return the gradual clustering-loss weight inside pretraining."""
    round_number = int(round_number)
    center_init_round = int(center_init_round)
    pretrain_rounds = int(pretrain_rounds)
    if round_number <= center_init_round:
        return 0.0
    if round_number > pretrain_rounds:
        return 1.0
    active_rounds = max(pretrain_rounds - center_init_round, 1)
    return min(1.0, (round_number - center_init_round) / active_rounds)


class FederatedMultiViewClusteringServer:
    def __init__(self, data, config, device):
        self.data = data
        self.config = config
        self.device = device
        self.training = config["training"]
        self.seed = int(self.training["seed"])
        self.rng = random.Random(self.seed)
        partitions = balanced_client_indices(
            data.num_samples,
            int(config["dataset"]["num_clients"]),
            self.seed,
        )
        if any(len(indices) < int(config["dataset"]["num_clusters"]) for indices in partitions):
            raise ValueError("Every client needs at least num_clusters samples for center initialization")
        self.clients = [
            FederatedClusteringClient(i, data, indices, config, device)
            for i, indices in enumerate(partitions)
        ]
        model = config["model"]
        self.global_model = MultiViewClusteringModel(
            view_dims=data.view_dims,
            num_clusters=int(config["dataset"]["num_clusters"]),
            hidden_dims=list(model["hidden_dims"]),
            embedding_dim=int(model["embedding_dim"]),
            dropout=float(model.get("dropout", 0.0)),
            alpha=float(model.get("student_t_alpha", 1.0)),
        ).to(device)
        self.history = []
        self.centers_initialized = False
        self.best_state = None
        self.best_round = None
        self.best_metrics = None
        self.initialization_bytes = 0

    def _selected_clients(self):
        count = max(1, int(np.ceil(len(self.clients) * float(self.training["join_ratio"]))))
        return self.rng.sample(self.clients, count) if count < len(self.clients) else list(self.clients)

    def _initialize_centers(self):
        summaries = [client.cluster_summary(self.global_model) for client in self.clients]
        self.initialization_bytes = sum(center.nbytes + count.nbytes for center, count in summaries)
        centers = np.concatenate([item[0] for item in summaries], axis=0)
        counts = np.concatenate([item[1] for item in summaries], axis=0)
        kmeans = KMeans(
            n_clusters=int(self.config["dataset"]["num_clusters"]),
            n_init=int(self.training.get("center_init_n_init", 10)),
            random_state=self.seed,
        ).fit(centers, sample_weight=np.maximum(counts, 1.0))
        with torch.no_grad():
            value = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(self.device)
            self.global_model.cluster_head.centers.copy_(value)
        self.centers_initialized = True

    @staticmethod
    def _align_centers(state_dict, reference_centers, cluster_counts=None):
        key = "cluster_head.centers"
        centers = state_dict[key].numpy()
        reference = reference_centers.numpy()
        distances = ((centers[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        rows, columns = linear_sum_assignment(distances)
        aligned = np.empty_like(centers)
        aligned[columns] = centers[rows]
        state_dict[key] = torch.from_numpy(aligned)
        if cluster_counts is None:
            return None
        counts = cluster_counts.numpy()
        aligned_counts = np.empty_like(counts)
        aligned_counts[columns] = counts[rows]
        return torch.from_numpy(aligned_counts)

    def _aggregate(self, updates):
        center_key = "cluster_head.centers"
        for update in updates:
            if "payload" in update:
                update["state_dict"] = reconstruct_state(
                    self.global_model, update["payload"], update.get("centers")
                )
        reference = self.global_model.state_dict()[center_key].detach().cpu()
        if self.centers_initialized:
            for update in updates:
                update["cluster_counts"] = self._align_centers(
                    update["state_dict"],
                    reference,
                    update.get("cluster_counts"),
                )
        total_samples = sum(update["num_samples"] for update in updates)
        result = {}
        for key, reference_value in self.global_model.state_dict().items():
            values = [update["state_dict"][key] for update in updates]
            if key == center_key and self.centers_initialized:
                numerator = torch.zeros_like(values[0])
                denominator = torch.zeros(values[0].shape[0], dtype=values[0].dtype)
                for update, value in zip(updates, values):
                    counts = update.get("cluster_counts")
                    if counts is None:
                        counts = torch.full_like(
                            denominator,
                            float(update["num_samples"]) / denominator.numel(),
                        )
                    counts = counts.to(dtype=values[0].dtype)
                    numerator.add_(value * counts.unsqueeze(1))
                    denominator.add_(counts)
                aggregated = numerator / denominator.clamp_min(1e-12).unsqueeze(1)
                empty_clusters = denominator <= 1e-12
                aggregated[empty_clusters] = reference[empty_clusters]
                momentum = float(self.training.get("center_momentum", 0.0))
                result[key] = momentum * reference + (1.0 - momentum) * aggregated
                continue
            if reference_value.is_floating_point():
                aggregated = torch.zeros_like(values[0])
                for update, value in zip(updates, values):
                    aggregated.add_(value, alpha=update["num_samples"] / total_samples)
                result[key] = aggregated
            else:
                result[key] = values[0]
        self.global_model.load_state_dict(result)
        self.global_model.to(self.device)

    @torch.no_grad()
    def evaluate(self):
        self.global_model.eval()
        dataset = MultiViewSubset(self.data, np.arange(self.data.num_samples))
        loader = DataLoader(
            dataset,
            batch_size=int(self.training.get("eval_batch_size", self.training["batch_size"])),
            shuffle=False,
            num_workers=int(self.training.get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
        )
        predictions, labels, confidences = [], [], []
        for views, batch_labels, _indices in loader:
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            assignments = self.global_model(views)["assignments"]
            predictions.append(assignments.argmax(dim=1).cpu().numpy())
            confidences.append(assignments.max(dim=1).values.cpu().numpy())
            labels.append(batch_labels.numpy())
        metrics = evaluate_clustering(np.concatenate(labels), np.concatenate(predictions))
        metrics["confidence"] = float(np.concatenate(confidences).mean())
        return metrics

    def train(self):
        rounds = int(self.training["rounds"])
        pretrain_rounds = int(self.training["pretrain_rounds"])
        center_init_round = int(self.training["center_init_round"])
        eval_gap = int(self.training.get("eval_gap", 1))
        report_gap = progress_report_gap(rounds)
        started = time.time()
        for round_index in range(rounds):
            center_initialization_bytes = 0
            if round_index == center_init_round and not self.centers_initialized:
                print("Initializing global cluster centers from client summaries ...")
                self._initialize_centers()
                center_initialization_bytes = self.initialization_bytes
            round_number = round_index + 1
            phase = "clustering" if round_index >= pretrain_rounds else "pretraining"
            clustering_enabled = self.centers_initialized
            clustering_weight_scale = pretraining_clustering_scale(
                round_number,
                center_init_round,
                pretrain_rounds,
            ) if clustering_enabled else 0.0
            selected = self._selected_clients()
            updates = [
                client.train(
                    self.global_model,
                    clustering_enabled,
                    round_index,
                    phase=phase,
                    clustering_weight_scale=clustering_weight_scale,
                )
                for client in selected
            ]
            dense_uplink = sum(
                dense_uplink_bytes(self.global_model, self.centers_initialized)
                for _client in selected
            )
            actual_uplink = sum(
                update["compression"]["bytes"] if "compression" in update
                else dense_uplink_bytes(self.global_model, self.centers_initialized)
                for update in updates
            )
            model_bytes = sum(
                value.numel() * value.element_size()
                for value in self.global_model.state_dict().values()
            )
            communication = {
                "uplink_bytes": int(actual_uplink + center_initialization_bytes),
                "dense_uplink_bytes": int(dense_uplink + center_initialization_bytes),
                "downlink_bytes": int(model_bytes * len(selected)),
                "center_initialization_bytes": int(center_initialization_bytes),
                "compression_seconds": sum(
                    update.get("compression", {}).get("compression_seconds", 0.0)
                    for update in updates
                ),
            }
            communication["uplink_reduction"] = (
                1.0 - communication["uplink_bytes"] / communication["dense_uplink_bytes"]
            )
            self._aggregate(updates)
            train_metrics = self._weighted_training_metrics(updates)
            record = {
                "round": round_index + 1,
                "phase": phase,
                "center_initialized": self.centers_initialized,
                "clients": [client.id for client in selected],
                "train": train_metrics,
                "communication": communication,
                "elapsed_seconds": time.time() - started,
            }
            if (round_index + 1) % eval_gap == 0 or round_index + 1 == rounds:
                record["clustering"] = self.evaluate()
                self._update_best(record["clustering"], round_index + 1, phase)
            self.history.append(record)
            if round_number % report_gap == 0 or round_number == rounds:
                clustering = record.get("clustering", {})
                print(
                    f"Round {round_number:03d}/{rounds} {record['phase']:<10} "
                    f"loss={train_metrics['loss']:.4f} "
                    f"ACC={clustering.get('acc', float('nan')):.4f} "
                    f"NMI={clustering.get('nmi', float('nan')):.4f} "
                    f"ARI={clustering.get('ari', float('nan')):.4f}"
                )
        if self.best_state is not None:
            self.global_model.load_state_dict(self.best_state)
            self.global_model.to(self.device)
        final_metrics = self.evaluate()
        return self._save(final_metrics)

    def _update_best(self, metrics, round_number, phase):
        if phase != "clustering":
            return
        selection_metric = self.training.get("selection_metric", "nmi")
        if selection_metric not in metrics:
            raise KeyError(f"Unknown selection metric: {selection_metric}")
        if self.best_metrics is None or metrics[selection_metric] > self.best_metrics[selection_metric]:
            self.best_metrics = copy.deepcopy(metrics)
            self.best_round = int(round_number)
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in self.global_model.state_dict().items()
            }

    @staticmethod
    def _weighted_training_metrics(updates):
        total = sum(update["num_samples"] for update in updates)
        keys = updates[0]["metrics"].keys()
        return {
            key: sum(update["metrics"][key] * update["num_samples"] for update in updates) / total
            for key in keys
        }

    def _save(self, final_metrics):
        output = self.config["output"]
        root = Path(output["directory"]) / self.data.name
        root.mkdir(parents=True, exist_ok=True)
        with (root / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2, ensure_ascii=False)
        summary = {
            "dataset": self.data.name,
            "samples": self.data.num_samples,
            "view_dims": self.data.view_dims,
            "num_clusters": self.data.num_clusters,
            "metrics": final_metrics,
            "best_round": self.best_round,
            "selection_metric": self.training.get("selection_metric", "nmi"),
            "communication": {
                "uplink_bytes": sum(item["communication"]["uplink_bytes"] for item in self.history),
                "dense_uplink_bytes": sum(item["communication"]["dense_uplink_bytes"] for item in self.history),
                "downlink_bytes": sum(item["communication"]["downlink_bytes"] for item in self.history),
                "compression_seconds": sum(item["communication"]["compression_seconds"] for item in self.history),
            },
            "config": self.config,
        }
        summary["communication"]["uplink_reduction"] = (
            1.0 - summary["communication"]["uplink_bytes"]
            / summary["communication"]["dense_uplink_bytes"]
        )
        with (root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        if output.get("save_model", True):
            # torch.save(self.global_model.state_dict(), root / "model.pt")
            pass
        return summary
