import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from flcore.clients.clientcluster import FederatedClusteringClient, spherical_kmeans
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
            cluster_head_type=model.get("cluster_head_type", "student_t"),
            student_t_distance_scale=float(model.get("student_t_distance_scale", 1.0)),
            cosine_temperature=float(model.get("cosine_temperature", 0.1)),
            prototype_mode="per_view",
            per_view_temperature=float(
                model.get("per_view_temperature", model.get("cosine_temperature", 0.1))
            ),
            per_view_head_type=model.get(
                "per_view_head_type", model.get("cluster_head_type", "student_t")
            ),
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
        self._initialize_per_view_centers(summaries)
        self.centers_initialized = True

    def _initialize_per_view_centers(self, summaries):
        """Initialize all view heads in one shared semantic slot order."""
        if not summaries or any(item.get("mode") != "per_view" for item in summaries):
            raise ValueError("Per-view initialization requires aggregate prototype summaries")
        num_views = len(self.global_model.prototype_heads)
        num_clusters = int(self.config["dataset"]["num_clusters"])
        expected = (num_views, num_clusters, self.global_model.embedding_dim)
        for item in summaries:
            if item["centers"].shape != expected or item["counts"].shape != expected[:2]:
                raise ValueError("Invalid per-view prototype summary shape")
            if item["coverage"].shape != (num_views,):
                raise ValueError("Invalid per-view coverage summary shape")
        self.initialization_bytes = sum(
            item["centers"].nbytes + item["counts"].nbytes + item["coverage"].nbytes
            for item in summaries
        )
        coverage = np.stack([item["coverage"] for item in summaries]).sum(axis=0)
        self.prototype_reference_view = int(np.argmax(coverage))
        reference_view = self.prototype_reference_view
        local_reference = np.concatenate(
            [item["centers"][reference_view] for item in summaries], axis=0
        )
        local_counts = np.concatenate(
            [item["counts"][reference_view] for item in summaries], axis=0
        )
        head_type = self.global_model.prototype_heads[0].head_type
        if head_type == "cosine":
            global_reference, _labels = spherical_kmeans(
                local_reference,
                num_clusters,
                n_init=int(self.training.get("center_init_n_init", 10)),
                max_iter=int(self.training.get("center_init_max_iter", 300)),
                random_state=self.seed,
                sample_weight=np.maximum(local_counts, 1e-6),
            )
        else:
            fitted = KMeans(
                n_clusters=num_clusters,
                n_init=int(self.training.get("center_init_n_init", 10)),
                random_state=self.seed,
            ).fit(local_reference, sample_weight=np.maximum(local_counts, 1e-6))
            global_reference = fitted.cluster_centers_.astype(np.float32)

        aligned_centers = []
        aligned_counts = []
        for item in summaries:
            local = item["centers"][reference_view]
            if head_type == "cosine":
                cost = 1.0 - local @ global_reference.T
            else:
                cost = ((local[:, None, :] - global_reference[None, :, :]) ** 2).sum(axis=2)
            rows, columns = linear_sum_assignment(cost)
            centers = np.empty_like(item["centers"])
            counts = np.empty_like(item["counts"])
            # One reference-view permutation is applied to every view. This is
            # the invariant that keeps row k semantically identical across heads.
            centers[:, columns, :] = item["centers"][:, rows, :]
            counts[:, columns] = item["counts"][:, rows]
            aligned_centers.append(centers)
            aligned_counts.append(counts)

        previous = [
            head.centers.detach().cpu().numpy().copy()
            for head in self.global_model.prototype_heads
        ]
        initialized = []
        for view_index in range(num_views):
            numerator = np.zeros_like(previous[view_index], dtype=np.float64)
            denominator = np.zeros(num_clusters, dtype=np.float64)
            for centers, counts in zip(aligned_centers, aligned_counts):
                weight = counts[view_index]
                numerator += centers[view_index] * weight[:, None]
                denominator += weight
            value = numerator / np.maximum(denominator[:, None], 1e-12)
            empty = denominator <= 1e-12
            value[empty] = previous[view_index][empty]
            initialized.append(value.astype(np.float32))
        with torch.no_grad():
            for head, value in zip(self.global_model.prototype_heads, initialized):
                head.centers.copy_(torch.from_numpy(value).to(self.device))
            self.global_model.project_cluster_centers_()

    def _aggregate(self, updates):
        self._aggregate_per_view(updates)

    def _align_per_view_prototypes(self, state_dict, references, cluster_counts=None):
        keys = self.global_model.prototype_center_keys()
        reference_view = int(getattr(self, "prototype_reference_view", 0))
        centers = state_dict[keys[reference_view]].numpy()
        reference = references[reference_view].numpy()
        if self.global_model.prototype_heads[reference_view].head_type == "cosine":
            cost = 1.0 - centers @ reference.T
        else:
            cost = ((centers[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        rows, columns = linear_sum_assignment(cost)
        for key in keys:
            local = state_dict[key].numpy()
            aligned = np.empty_like(local)
            aligned[columns] = local[rows]
            state_dict[key] = torch.from_numpy(aligned)
        if cluster_counts is None:
            return None
        counts = cluster_counts.numpy()
        if counts.shape != (len(keys), centers.shape[0]):
            raise ValueError("Per-view cluster counts have an invalid shape")
        aligned_counts = np.empty_like(counts)
        aligned_counts[:, columns] = counts[:, rows]
        return torch.from_numpy(aligned_counts)

    def _aggregate_per_view(self, updates):
        for update in updates:
            if "payload" in update:
                update["state_dict"] = reconstruct_state(
                    self.global_model, update["payload"], update.get("centers")
                )
        center_keys = self.global_model.prototype_center_keys()
        references = tuple(
            self.global_model.state_dict()[key].detach().cpu()
            for key in center_keys
        )
        if self.centers_initialized:
            for update in updates:
                update["cluster_counts"] = self._align_per_view_prototypes(
                    update["state_dict"], references, update.get("cluster_counts")
                )

        total_samples = sum(update["num_samples"] for update in updates)
        result = {}
        center_indices = {key: index for index, key in enumerate(center_keys)}
        for key, reference_value in self.global_model.state_dict().items():
            values = [update["state_dict"][key] for update in updates]
            if key in center_indices and self.centers_initialized:
                view_index = center_indices[key]
                numerator = torch.zeros_like(values[0])
                denominator = torch.zeros(values[0].shape[0], dtype=values[0].dtype)
                for update, value in zip(updates, values):
                    counts = update.get("cluster_counts")
                    if counts is None:
                        counts = torch.full_like(
                            denominator,
                            float(update["num_samples"]) / denominator.numel(),
                        )
                    else:
                        counts = counts[view_index].to(dtype=values[0].dtype)
                    numerator.add_(value * counts.unsqueeze(1))
                    denominator.add_(counts)
                aggregated = numerator / denominator.clamp_min(1e-12).unsqueeze(1)
                empty = denominator <= 1e-12
                aggregated[empty] = references[view_index][empty]
                momentum = float(self.training.get("center_momentum", 0.0))
                result[key] = (
                    momentum * references[view_index]
                    + (1.0 - momentum) * aggregated
                )
                continue
            if reference_value.is_floating_point():
                aggregated = torch.zeros_like(values[0])
                for update, value in zip(updates, values):
                    aggregated.add_(value, alpha=update["num_samples"] / total_samples)
                result[key] = aggregated
            else:
                result[key] = values[0]
        self.global_model.load_state_dict(result)
        self.global_model.project_cluster_centers_(
            tuple(reference.to(self.device) for reference in references)
        )
        self.global_model.to(self.device)

    @torch.no_grad()
    def evaluate(self):
        self.global_model.eval()
        predictions, labels, probabilities = [], [], []
        datasets = (
            [client.dataset for client in self.clients]
            if self.config.get("missing", {}).get("enabled", False)
            else [MultiViewSubset(self.data, np.arange(self.data.num_samples))]
        )
        for dataset in datasets:
            loader = DataLoader(
                dataset,
                batch_size=int(self.training.get("eval_batch_size", self.training["batch_size"])),
                shuffle=False,
                num_workers=int(self.training.get("num_workers", 0)),
                pin_memory=self.device.type == "cuda",
            )
            for batch in loader:
                views, batch_labels, _indices, mask = FederatedClusteringClient._unpack(batch)
                views = tuple(view.to(self.device, non_blocking=True) for view in views)
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)
                assignments = self.global_model(views, mask=mask)["assignments"]
                predictions.append(assignments.argmax(dim=1).cpu().numpy())
                probabilities.append(assignments.cpu().numpy())
                labels.append(batch_labels.numpy())
        probabilities = np.concatenate(probabilities)
        metrics = evaluate_clustering(np.concatenate(labels), np.concatenate(predictions))
        metrics["confidence"] = float(probabilities.max(axis=1).mean())
        entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1)
        top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
        soft_counts = probabilities.sum(axis=0)
        mean_assignment = soft_counts / max(float(soft_counts.sum()), 1e-12)
        self.last_diagnostics = {
            "normalized_entropy": float(entropy.mean() / np.log(probabilities.shape[1])),
            "top1_top2_margin": float((top_two[:, 1] - top_two[:, 0]).mean()),
            "effective_clusters": float(np.exp(
                -(mean_assignment * np.log(np.maximum(mean_assignment, 1e-12))).sum()
            )),
            "min_cluster_occupancy": float(mean_assignment.min()),
            "max_cluster_occupancy": float(mean_assignment.max()),
            "prototype_soft_counts": soft_counts.tolist(),
        }
        return metrics

    def train(self):
        rounds = int(self.training["rounds"])
        pretrain_rounds = int(self.training["pretrain_rounds"])
        center_init_round = int(self.training["center_init_round"])
        eval_gap = int(self.training.get("eval_gap", 1))
        report_gap = progress_report_gap(rounds)
        started = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
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
                record["diagnostics"] = copy.deepcopy(self.last_diagnostics)
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
            "selection_protocol": "oracle-best: labels used only for evaluation/checkpoint selection",
            "last_metrics": self.history[-1]["clustering"],
            "stability": {
                key: self.history[-1]["clustering"][key] - final_metrics[key]
                for key in ("acc", "nmi", "ari")
            },
            "runtime": {
                "device": str(self.device),
                "torch_version": str(torch.__version__),
                "parameters": sum(parameter.numel() for parameter in self.global_model.parameters()),
                "elapsed_seconds": self.history[-1]["elapsed_seconds"],
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda" else 0
                ),
            },
            "diagnostics": copy.deepcopy(self.last_diagnostics),
            "communication": {
                "uplink_bytes": sum(item["communication"]["uplink_bytes"] for item in self.history),
                "dense_uplink_bytes": sum(item["communication"]["dense_uplink_bytes"] for item in self.history),
                "downlink_bytes": sum(item["communication"]["downlink_bytes"] for item in self.history),
                "compression_seconds": sum(item["communication"]["compression_seconds"] for item in self.history),
            },
            "config": self.config,
        }
        if self.config.get("missing", {}).get("enabled", False):
            masks = [client.dataset.mask.numpy() for client in self.clients]
            joined = np.concatenate(masks, axis=0)
            summary["missing"] = {
                "nominal_rate": float(self.config["missing"]["rate"]),
                "incomplete_sample_rate": float((~joined.all(axis=1)).mean()),
                "missing_cell_rate": float((~joined).mean()),
                "observed_per_view": joined.sum(axis=0).astype(int).tolist(),
                "complete_per_client": [int(mask.all(axis=1).sum()) for mask in masks],
                "mask_sha256": hashlib.sha256(joined.tobytes()).hexdigest(),
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
