import copy
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from flcore.compression import compress_client_update
from flcore.trainmodel.multiview import clustering_objective, target_distribution
from utils.mat_data import MultiViewSubset, fixed_missing_mask


def _normalize_rows(values, eps=1e-12):
    """Normalize nonzero rows without producing NaNs for zero vectors."""
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    valid = norms[:, 0] > eps
    normalized[valid] = values[valid] / norms[valid]
    return normalized, valid


def spherical_kmeans(
    samples,
    n_clusters,
    *,
    n_init=10,
    max_iter=300,
    random_state=None,
    sample_weight=None,
    tol=1e-4,
):
    """Cluster directions with cosine assignment and unit-norm centroids.

    The public controls mirror the relevant sklearn KMeans controls. Each
    initialization uses weighted k-means++ on normalized samples, followed by
    spherical Lloyd iterations. Empty clusters and zero-mean clusters retain
    a finite unit-vector fallback instead of yielding NaNs.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError("samples must be a non-empty 2D array")
    if not np.isfinite(samples).all():
        raise ValueError("samples must contain only finite values")
    n_clusters = int(n_clusters)
    n_init = int(n_init)
    max_iter = int(max_iter)
    if not 1 <= n_clusters <= samples.shape[0]:
        raise ValueError("n_clusters must be between 1 and n_samples")
    if n_init <= 0 or max_iter <= 0:
        raise ValueError("n_init and max_iter must be positive")

    normalized, nonzero = _normalize_rows(samples)
    if sample_weight is None:
        weights = np.ones(samples.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if weights.shape != (samples.shape[0],):
            raise ValueError("sample_weight must contain one value per sample")
        if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("sample_weight must be finite, nonnegative, and sum to > 0")

    rng = check_random_state(random_state)
    best_centers = None
    best_labels = None
    best_objective = np.inf

    def repair_centers(centers):
        centers, valid = _normalize_rows(centers)
        if valid.all():
            return centers
        candidates = np.flatnonzero(nonzero & (weights > 0))
        if candidates.size:
            # Prefer high-weight observations and break ties by stable index.
            order = candidates[np.argsort(-weights[candidates], kind="stable")]
            for offset, row in enumerate(np.flatnonzero(~valid)):
                centers[row] = normalized[order[offset % len(order)]]
        else:
            # The data have no direction. Canonical fallbacks keep the cosine
            # head finite while preserving deterministic behavior.
            for row in np.flatnonzero(~valid):
                centers[row, row % centers.shape[1]] = 1.0
        return centers

    for _ in range(n_init):
        initial, _ = kmeans_plusplus(
            normalized,
            n_clusters=n_clusters,
            sample_weight=weights,
            random_state=rng,
        )
        centers = repair_centers(initial)
        labels = np.zeros(samples.shape[0], dtype=np.int64)

        for _iteration in range(max_iter):
            similarities = normalized @ centers.T
            labels = similarities.argmax(axis=1)
            updated = np.zeros_like(centers)
            for cluster in range(n_clusters):
                members = labels == cluster
                if members.any() and weights[members].sum() > 0:
                    updated[cluster] = np.average(
                        normalized[members], axis=0, weights=weights[members]
                    )
                else:
                    updated[cluster] = centers[cluster]
            updated = repair_centers(updated)
            shift = np.linalg.norm(updated - centers)
            centers = updated
            if shift <= float(tol):
                break

        similarities = normalized @ centers.T
        labels = similarities.argmax(axis=1)
        objective = np.sum(weights * (1.0 - similarities[np.arange(len(labels)), labels]))
        if objective < best_objective:
            best_objective = float(objective)
            best_centers = centers.copy()
            best_labels = labels.copy()

    return best_centers.astype(np.float32), best_labels


class FederatedClusteringClient:
    """One horizontal client owning an IID, non-overlapping sample subset."""

    def __init__(self, client_id, data, indices, config, device):
        self.id = int(client_id)
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = config
        self.device = device
        missing = config.get("missing", {})
        if missing.get("enabled", False):
            mask = fixed_missing_mask(
                len(self.indices), len(data.views), float(missing["rate"]),
                int(config["training"]["seed"]), self.id, data.name,
            )
            self.dataset = MultiViewSubset(
                data, self.indices, mask=mask,
                normalization=config["dataset"].get("normalization", "standard"),
            )
        else:
            self.dataset = MultiViewSubset(data, self.indices)
        self.compression_residual = None

    @staticmethod
    def _unpack(batch):
        if len(batch) == 3:
            views, labels, indices = batch
            return views, labels, indices, None
        views, labels, indices, mask = batch
        return views, labels, indices, mask

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
        head_parameters = [
            parameter
            for head in model.clustering_heads()
            for parameter in head.parameters()
        ]
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
            for batch in loader:
                views, _labels, indices, mask = self._unpack(batch)
                views = tuple(view.to(self.device, non_blocking=True) for view in views)
                if mask is not None:
                    mask = mask.to(self.device, non_blocking=True)
                targets = (
                    target_cache[indices].to(self.device, non_blocking=True)
                    if target_cache is not None
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                outputs = model(views, mask=mask)
                loss, metrics = clustering_objective(
                    outputs,
                    views,
                    self.config["loss_weights"],
                    clustering_enabled=clustering_enabled,
                    clustering_weight_scale=clustering_weight_scale,
                    target_assignments=targets,
                    mask=mask,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Client {self.id} produced a non-finite loss")
                loss.backward()
                clip = float(training.get("gradient_clip", 0.0))
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                model.project_cluster_centers_()
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
            views, targets, calibration_mask = None, None, None
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
                if self.dataset.mask is not None:
                    calibration_mask = self.dataset.mask[positions].to(self.device)
                if target_cache is not None:
                    targets = target_cache[self.dataset.indices[positions]].to(self.device)
            packet = compress_client_update(
                model,
                global_model,
                {**compression, "loss_weights": self.config["loss_weights"]},
                views=views,
                targets=targets,
                mask=calibration_mask,
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
        for batch in self._loader(False):
            views, _labels, batch_indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            outputs = model(views, mask=mask)
            assignments.append(outputs["assignments"])
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
        for batch in self._loader(False):
            views, _labels, _indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            outputs = model(views, mask=mask)
            # Missing rows are zeros in view_assignments and therefore add no
            # evidence to that view's prototype counts.
            batch_counts = torch.stack([
                assignment.sum(dim=0)
                for assignment in outputs["view_assignments"]
            ])
            counts = batch_counts if counts is None else counts + batch_counts
        return counts.detach().cpu()

    @torch.no_grad()
    def cluster_summary(self, model):
        """Return local KMeans centers and counts; raw embeddings never leave the client."""
        model = copy.deepcopy(model).to(self.device)
        model.eval()
        return self._per_view_cluster_summary(model)

    @torch.no_grad()
    def _per_view_cluster_summary(self, model):
        """Build aligned aggregate prototypes from local shared pseudo-labels.

        Only V x K centers, V x K counts, and V coverage totals leave the
        client. The shared labels are computed locally and are never uploaded.
        """
        fused_batches = []
        per_view_batches = [[] for _ in model.view_models]
        mask_batches = []
        for batch in self._loader(False):
            views, _labels, _indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is None:
                mask = torch.ones(
                    (views[0].shape[0], len(views)),
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                mask = mask.to(self.device, non_blocking=True)
            view_embeddings, fused = model.encode(views, mask=mask)
            fused_batches.append(fused.cpu().numpy())
            mask_batches.append(mask.cpu().numpy().astype(bool, copy=False))
            for view_index, embedding in enumerate(view_embeddings):
                per_view_batches[view_index].append(embedding.cpu().numpy())

        fused = np.concatenate(fused_batches, axis=0)
        masks = np.concatenate(mask_batches, axis=0)
        view_embeddings = [np.concatenate(items, axis=0) for items in per_view_batches]
        num_clusters = int(self.config["dataset"]["num_clusters"])
        if model.prototype_heads[0].head_type == "cosine":
            _fused_centers, shared_labels = spherical_kmeans(
                fused,
                num_clusters,
                n_init=int(self.config["training"].get("center_init_n_init", 10)),
                max_iter=int(self.config["training"].get("center_init_max_iter", 300)),
                random_state=int(self.config["training"]["seed"]) + self.id,
            )
        else:
            pseudo = KMeans(
                n_clusters=num_clusters,
                n_init=int(self.config["training"].get("center_init_n_init", 10)),
                random_state=int(self.config["training"]["seed"]) + self.id,
            ).fit(fused)
            shared_labels = pseudo.labels_

        centers = []
        counts = []
        for view_index, (embedding, head) in enumerate(
            zip(view_embeddings, model.prototype_heads)
        ):
            observed = masks[:, view_index]
            normalized, nonzero = _normalize_rows(embedding)
            view_counts = np.bincount(
                shared_labels[observed], minlength=num_clusters
            ).astype(np.float64)
            # An unsupported local semantic slot keeps a finite unit prototype
            # but has zero count, so it cannot affect server initialization.
            view_centers = head.centers.detach().cpu().numpy().astype(
                np.float64, copy=True
            )
            if head.head_type == "cosine":
                view_centers, _valid = _normalize_rows(view_centers)
            for cluster in range(num_clusters):
                members = observed & nonzero & (shared_labels == cluster)
                if members.any():
                    mean = normalized[members].mean(axis=0, keepdims=True)
                    if head.head_type == "cosine":
                        candidate, valid = _normalize_rows(mean)
                        if valid[0]:
                            view_centers[cluster] = candidate[0]
                    else:
                        view_centers[cluster] = mean[0]
            centers.append(view_centers.astype(np.float32))
            counts.append(view_counts)

        return {
            "mode": "per_view",
            "centers": np.stack(centers),
            "counts": np.stack(counts),
            "coverage": masks.sum(axis=0).astype(np.float64),
        }
