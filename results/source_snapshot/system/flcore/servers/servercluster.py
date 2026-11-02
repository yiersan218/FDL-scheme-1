"""两阶段联邦聚类服务端：中心对齐、共享聚合、评估与结果保存。"""

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

from config import ensure_writable_output
from flcore.clients.clientcluster import FederatedClusteringClient
from flcore.communication.codec import dense_uplink_bytes, reconstruct_state
from flcore.trainmodel.multiview import MultiViewClusteringModel
from utils.clustering_metrics import evaluate_clustering
from utils.mat_data import balanced_client_indices


def progress_report_gap(total_rounds):
    """按总轮数约每 10% 返回一次进度输出间隔，最小为一轮。"""
    return max(1, int(np.ceil(int(total_rounds) / 10.0)))


def pretraining_clustering_scale(round_number, center_init_round, pretrain_rounds):
    """计算中心初始化后预训练内部逐步增大的聚类损失系数。"""
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
    """协调统一补全聚类训练、真实原型聚合及客户端上下文评估。"""

    def __init__(self, data, config, device):
        """按无标签随机索引划分客户端，并建立全局模型与训练记录状态。"""
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
            prototype_mode="per_view",
            per_view_head_type=model.get(
                "per_view_head_type", model.get("cluster_head_type", "student_t")
            ),
            completion=config.get("completion"),
        ).to(device)
        self.history = []
        self.centers_initialized = False
        self.best_state = None
        self.best_private_states = None
        self.best_round = None
        self.best_metrics = None
        self.current_round = 0
        self.initialization_bytes = 0

    def _selected_clients(self):
        """按参与比例无放回抽取客户端，至少选择一个客户端。"""
        count = max(1, int(np.ceil(len(self.clients) * float(self.training["join_ratio"]))))
        return self.rng.sample(self.clients, count) if count < len(self.clients) else list(self.clients)

    def _initialize_centers(self):
        """收集所有客户端的聚类摘要并完成预训练内部的原型初始化。"""
        summaries = [client.cluster_summary(self.global_model) for client in self.clients]
        self._initialize_per_view_centers(summaries)
        self.centers_initialized = True

    def _initialize_per_view_centers(self, summaries):
        """以覆盖量最大的参考视图统一所有客户端的原型语义槽位。

        每个摘要包含 [V, K, d] 中心、[V, K] 硬计数与 [V] 覆盖量。
        参考视图求得的同一 Hungarian 排列应用于全部视图，再按各视图
        对应簇计数聚合；没有观测支持的槽位保留原中心。
        """
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
            cost = ((local[:, None, :] - global_reference[None, :, :]) ** 2).sum(axis=2)
            rows, columns = linear_sum_assignment(cost)
            centers = np.empty_like(item["centers"])
            counts = np.empty_like(item["counts"])
            # 参考视图求得的同一排列应用于所有视图，
            # 以维持不同原型头第 k 行的共同簇语义。
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

    def _aggregate(self, updates):
        """将客户端更新交给当前唯一的逐视图 A2 聚合路径。"""
        self._aggregate_per_view(updates)

    def _align_per_view_prototypes(self, state_dict, references, cluster_counts=None):
        """原地按参考视图排列客户端全部原型，并返回同序软簇计数。

        同一客户端的各视图共用一个排列，防止独立对齐破坏簇语义一致性。
        cluster_counts 为 None 时只排列中心，不生成虚构的观测计数。
        """
        keys = self.global_model.prototype_center_keys()
        reference_view = int(getattr(self, "prototype_reference_view", 0))
        centers = state_dict[keys[reference_view]].numpy()
        reference = references[reference_view].numpy()
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
        """解码上传包、对齐原型并聚合共享模型状态。

        共享普通参数按样本数 FedAvg；原型按对应视图软簇计数加权并使用
        中心动量。local 补全参数维持服务端公共初始化，不读取客户端私有值。
        """
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
        private_keys = set(getattr(self.global_model, "local_completion_state_keys", lambda: ())())
        for key, reference_value in self.global_model.state_dict().items():
            if key in private_keys:
                # 保留公共初始化，不读取、对齐或聚合任何客户端私有参数。
                result[key] = reference_value.detach().cpu().clone()
                continue
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
        self.global_model.to(self.device)

    @torch.no_grad()
    def evaluate(self):
        """用客户端真实掩码、锚点和配套私有状态评估受限补全分配。

        完整与缺失数据复用相同的客户端预处理。评估只计算预测，
        不计算自监督损失，不把标签送入补全或原型分配。
        """
        self.global_model.eval()
        predictions, labels, probabilities = [], [], []
        assignment_betas = []
        round_number = max(1, self.current_round)
        for client in self.clients:
            # 本地推理封装私有补全、掩码和锚点；本接口仅用于单进程仿真评估。
            local = client.predict(self.global_model, round_number)
            assignments = local["assignments"]
            assignment_betas.append(local["assignment_beta"].numpy())
            predictions.append(assignments.argmax(dim=1).numpy())
            probabilities.append(assignments.numpy())
            labels.append(local["labels"].numpy())
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
            "prediction_soft_counts": soft_counts.tolist(),
            "completion_assignment_beta": float(np.concatenate(assignment_betas).mean()),
        }
        return metrics

    def train(self):
        """执行预训练与正式聚类两阶段调度，并返回保存后的结果摘要。

        中心初始化发生在预训练内部；每轮记录实际编码载荷、训练指标和
        评估结果。结束后恢复最佳模型再评估，同时在 history 中保留真实末轮。
        """
        rounds = int(self.training["rounds"])
        ensure_writable_output(self.config["output"]["directory"])
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
            self.current_round = round_number
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
            private_keys = set(getattr(self.global_model, "local_completion_state_keys", lambda: ())())
            model_bytes = sum(
                value.numel() * value.element_size()
                for key, value in self.global_model.state_dict().items()
                if key not in private_keys
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
        # 末轮指标已写入 history；恢复最佳参数后的评估不能冒充真实末轮。
        self._restore_best()
        final_metrics = self.evaluate()
        return self._save(final_metrics)

    def _restore_best(self):
        """恢复同一最佳轮的全局模型、客户端私有参数和补全调度轮号。"""
        if self.best_state is not None:
            self.global_model.load_state_dict(self.best_state)
            self.global_model.to(self.device)
            self.current_round = self.best_round
            if self.best_private_states is not None:
                for client in self.clients:
                    client.private_completion_state = copy.deepcopy(self.best_private_states[client.id])

    def _update_best(self, metrics, round_number, phase):
        """仅在正式聚类阶段按指定指标保存最佳参数及对应轮号。"""
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
            # 私有补全与全局参数、调度轮次一起冻结，避免最佳轮和末轮状态混用。
            self.best_private_states = {
                client.id: copy.deepcopy(client.private_completion_state)
                for client in self.clients
            }

    @staticmethod
    def _weighted_training_metrics(updates):
        """按客户端样本数对本轮上传的训练指标求加权平均。"""
        total = sum(update["num_samples"] for update in updates)
        keys = updates[0]["metrics"].keys()
        return {
            key: sum(update["metrics"][key] * update["num_samples"] for update in updates) / total
            for key in keys
        }

    def _save(self, final_metrics):
        """保存逐轮历史与汇总，分别记录最佳检查点和真实末轮指标。

        汇总包含配置、编码通信字节、运行代价及掩码统计；不写入单样本
        表示、伪标签或锚点。本实现中的模型检查点写出分支保持停用。
        """
        output = self.config["output"]
        root = ensure_writable_output(Path(output["directory"]) / self.data.name)
        root.mkdir(parents=True, exist_ok=True)
        private_keys = set(getattr(self.global_model, "local_completion_state_keys", lambda: ())())
        with ensure_writable_output(root / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2, ensure_ascii=False)
        summary = {
            "algorithm_version": self.config.get("algorithm_version", "completion-v1"),
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
                "federated_parameters": sum(
                    parameter.numel() for key, parameter in self.global_model.named_parameters()
                    if key not in private_keys
                ),
                "local_completion_parameters_per_client": sum(
                    parameter.numel() for key, parameter in self.global_model.named_parameters()
                    if key in private_keys
                ),
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
        if getattr(self.global_model, "completion_enabled", False):
            structure = [
                (key, list(value.shape), str(value.dtype))
                for key, value in self.global_model.state_dict().items()
            ]
            summary["completion"] = {
                "enabled": True,
                "loss_type": self.global_model.completion_config["alignment_mode"],
                "consistency_type": "cross_view_completion",
                "prediction": (
                    "bounded_completed_assignments"
                    if self.global_model.completion_config.get("use_completed_assignments", True)
                    else "observed_per_view_assignments"
                ),
                "target_teacher": "observed_assignments_fixed_per_round",
                "prototype_statistics": "observed_only",
                "bank_scope": "client_local_ephemeral",
                "parameter_scope": getattr(self.global_model, "completion_parameter_scope", "federated"),
                "use_completed_initialization": bool(
                    self.global_model.completion_config.get("use_completed_initialization", True)
                ),
                "structure_sha256": hashlib.sha256(
                    json.dumps(structure, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            }
        if all(client.dataset.mask is not None for client in self.clients):
            masks = [client.dataset.mask.numpy() for client in self.clients]
            joined = np.concatenate(masks, axis=0)
            summary["missing"] = {
                "nominal_rate": (
                    float(self.config.get("missing", {}).get("rate", 0.0))
                    if self.config.get("missing", {}).get("enabled", True) else 0.0
                ),
                "normalization_scope": "client_observed",
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
        with ensure_writable_output(root / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary
