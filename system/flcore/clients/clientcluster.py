import copy
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from flcore.compression import compress_client_update
from flcore.trainmodel.multiview import clustering_objective, target_distribution
from utils.mat_data import MultiViewSubset


class FederatedClusteringClient:
    """One horizontal client owning an IID, non-overlapping sample subset."""

    def __init__(self, client_id, data, indices, config, device):
        self.id = int(client_id)
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = config
        self.device = device
        self.dataset = MultiViewSubset(data, self.indices)
        self.compression_residual = None

    @property
    def num_samples(self):
        return len(self.dataset)

    def _loader(self, shuffle, seed_offset=0):
        training = self.config["training"]
        generator = torch.Generator()
        generator.manual_seed(int(training["seed"]) + self.id * 1009 + seed_offset)
        return DataLoader(
            self.dataset,
            batch_size=min(int(training["batch_size"]), self.num_samples),
            shuffle=shuffle,
            drop_last=False,
            num_workers=int(training.get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
            generator=generator,
        )

    def train(
        self,
        global_model,
        clustering_enabled,
        round_index,
        phase,
        clustering_weight_scale=1.0,
    ):
        if phase not in {"pretraining", "clustering"}:
            raise ValueError(f"Unknown training phase: {phase}")
        model = copy.deepcopy(global_model).to(self.device)
        target_cache = self.local_target_cache(model) if clustering_enabled else None
        model.train()
        training = self.config["training"]
        learning_rate = float(training["learning_rate"])
        local_epochs = int(training["pretraining_local_epochs"])
        if phase == "clustering":
            learning_rate = float(training["clustering_learning_rate"])
            local_epochs = int(training["clustering_local_epochs"])
        elif clustering_enabled:
            # Center initialization and the LR milestone are internal events of
            # the single pretraining stage, not a separate train phase.
            learning_rate = float(training["pretraining_end_learning_rate"])
        head_learning_rate = learning_rate * float(
            training.get("cluster_head_learning_rate_multiplier", 1.0)
        )
        head_parameters = list(model.cluster_head.parameters())
        head_parameter_ids = {id(parameter) for parameter in head_parameters}
        representation_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in head_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": representation_parameters, "lr": learning_rate},
                {"params": head_parameters, "lr": head_learning_rate},
            ],
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        totals = defaultdict(float)
        steps = 0
        for local_epoch in range(local_epochs):
            loader = self._loader(True, round_index * 97 + local_epoch)
            for views, _labels, indices in loader:
                views = tuple(view.to(self.device, non_blocking=True) for view in views)
                targets = (
                    target_cache[indices].to(self.device, non_blocking=True)
                    if target_cache is not None
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = model(views)
                loss, metrics = clustering_objective(
                    outputs,
                    views,
                    self.config["loss_weights"],
                    clustering_enabled=clustering_enabled,
                    clustering_weight_scale=clustering_weight_scale,
                    target_assignments=targets,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Client {self.id} produced a non-finite loss")
                loss.backward()
                clip = float(training.get("gradient_clip", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                for key, value in metrics.items():
                    totals[key] += value
                steps += 1
        result = {
            "client_id": self.id,
            "num_samples": self.num_samples,
            "metrics": {key: value / max(steps, 1) for key, value in totals.items()},
        }
        if clustering_enabled:
            result["cluster_counts"] = self.soft_cluster_counts(model)
        compression = self.config.get("compression", {"method": "none"})
        if compression["method"] == "none":
            result["state_dict"] = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
        else:
            views, targets = None, None
            if compression["method"] in {"paper", "stage"}:
                rng = np.random.default_rng(
                    int(training["seed"]) + self.id * 1009 + round_index * 7919 + 17
                )
                positions = torch.from_numpy(rng.choice(
                    self.num_samples,
                    size=min(int(compression["calibration_size"]), self.num_samples),
                    replace=False,
                ))
                views = tuple(view[positions].to(self.device) for view in self.dataset.views)
                if target_cache is not None:
                    targets = target_cache[self.dataset.indices[positions]].to(self.device)
            packet = compress_client_update(
                model,
                global_model,
                {**compression, "loss_weights": self.config["loss_weights"]},
                views=views,
                targets=targets,
                centers_enabled=clustering_enabled,
                clustering_scale=clustering_weight_scale,
                residual=self.compression_residual if compression["error_feedback"] else None,
            )
            if compression["error_feedback"]:
                self.compression_residual = packet.pop("residual")
            else:
                packet.pop("residual")
            result["payload"] = packet.pop("payload")
            result["centers"] = packet.pop("centers")
            result["compression"] = packet
        return result

    @torch.no_grad()
    def local_target_cache(self, model):
        """Build one full-local-data DEC target and keep it fixed this round."""
        model.eval()
        assignments = []
        indices = []
        for views, _labels, batch_indices in self._loader(False):
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            _view_embeddings, fused = model.encode(views)
            assignments.append(model.cluster_head(fused))
            indices.append(batch_indices)
        local_assignments = torch.cat(assignments, dim=0)
        local_targets = target_distribution(local_assignments).cpu()
        global_indices = torch.cat(indices, dim=0)
        cache = torch.empty(
            (self.data.num_samples, local_targets.shape[1]),
            dtype=local_targets.dtype,
        )
        cache[global_indices] = local_targets
        return cache

    @torch.no_grad()
    def soft_cluster_counts(self, model):
        """Return per-cluster soft counts without exposing sample assignments."""
        model.eval()
        counts = None
        for views, _labels, _indices in self._loader(False):
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            _view_embeddings, fused = model.encode(views)
            batch_counts = model.cluster_head(fused).sum(dim=0)
            counts = batch_counts if counts is None else counts + batch_counts
        return counts.detach().cpu()

    @torch.no_grad()
    def cluster_summary(self, model):
        """Return local KMeans centers and counts; raw embeddings never leave the client."""
        model = copy.deepcopy(model).to(self.device)
        model.eval()
        embeddings = []
        for views, _labels, _indices in self._loader(False):
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            embeddings.append(model.encode(views)[1].cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        num_clusters = int(self.config["dataset"]["num_clusters"])
        kmeans = KMeans(
            n_clusters=num_clusters,
            n_init=int(self.config["training"].get("center_init_n_init", 10)),
            random_state=int(self.config["training"]["seed"]) + self.id,
        ).fit(embeddings)
        counts = np.bincount(kmeans.labels_, minlength=num_clusters).astype(np.float64)
        return kmeans.cluster_centers_.astype(np.float32), counts
