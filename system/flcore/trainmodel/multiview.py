import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dimensions, dropout=0.0, final_activation=False):
    layers = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        is_last = index == len(dimensions) - 2
        if not is_last or final_activation:
            # LayerNorm has no client-specific running statistics and is stable for
            # the small final batches common in federated partitions.
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ViewAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout):
        super().__init__()
        encoder_dims = [input_dim, *hidden_dims, embedding_dim]
        decoder_dims = [embedding_dim, *reversed(hidden_dims), input_dim]
        self.encoder = _mlp(encoder_dims, dropout=dropout)
        self.decoder = _mlp(decoder_dims, dropout=dropout)

    def encode(self, view):
        return self.encoder(view)

    def decode(self, embedding):
        return self.decoder(embedding)


class ClusteringHead(nn.Module):
    """Trainable prototypes with Student-t or spherical-cosine assignments."""
    def __init__(
        self,
        num_clusters,
        embedding_dim,
        alpha=1.0,
        head_type="student_t",
        distance_scale=1.0,
        cosine_temperature=0.1,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.head_type = str(head_type)
        self.distance_scale = float(distance_scale)
        self.cosine_temperature = float(cosine_temperature)
        if self.head_type not in {"student_t", "cosine"}:
            raise ValueError("head_type must be student_t or cosine")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not torch.isfinite(torch.tensor(self.distance_scale)) or self.distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive")
        if (
            not torch.isfinite(torch.tensor(self.cosine_temperature))
            or self.cosine_temperature <= 0
        ):
            raise ValueError("cosine_temperature must be finite and positive")
        self.centers = nn.Parameter(torch.empty(num_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embedding):
        if self.head_type == "cosine":
            embedding = F.normalize(embedding, dim=1)
            centers = F.normalize(self.centers, dim=1)
            logits = embedding @ centers.T / self.cosine_temperature
            return torch.softmax(logits, dim=1)
        distance = torch.sum((embedding.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        assignments = (
            1.0 + self.distance_scale * distance / self.alpha
        ).pow(-(self.alpha + 1.0) / 2.0)
        return assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)

    @torch.no_grad()
    def project_centers_(self, reference=None, eps=1e-12):
        """Keep cosine prototypes on the unit sphere, with a safe zero-row fallback."""
        if self.head_type != "cosine":
            return
        norms = self.centers.norm(dim=1, keepdim=True)
        invalid = norms.squeeze(1) <= eps
        if invalid.any():
            if reference is None:
                replacement = torch.zeros_like(self.centers[invalid])
                replacement[:, 0] = 1.0
            else:
                replacement = F.normalize(
                    reference.to(device=self.centers.device, dtype=self.centers.dtype)[invalid],
                    dim=1,
                )
                replacement_invalid = replacement.norm(dim=1) <= eps
                if replacement_invalid.any():
                    replacement[replacement_invalid] = 0.0
                    replacement[replacement_invalid, 0] = 1.0
            self.centers[invalid] = replacement
            norms = self.centers.norm(dim=1, keepdim=True)
        self.centers.div_(norms.clamp_min(eps))


class MultiViewClusteringModel(nn.Module):
    def __init__(self, view_dims, num_clusters, hidden_dims, embedding_dim, dropout=0.0,
                 alpha=1.0, cluster_head_type="student_t",
                 student_t_distance_scale=1.0, cosine_temperature=0.1,
                 prototype_mode="per_view", per_view_temperature=0.1,
                 per_view_head_type=None):
        super().__init__()
        self.prototype_mode = str(prototype_mode)
        if self.prototype_mode != "per_view":
            raise ValueError("prototype_mode only supports per_view (A2)")
        self.embedding_dim = int(embedding_dim)
        self.num_clusters = int(num_clusters)
        self.view_models = nn.ModuleList(
            ViewAutoencoder(dim, hidden_dims, embedding_dim, dropout) for dim in view_dims
        )
        self.view_logits = nn.Parameter(torch.zeros(len(view_dims)))
        self.per_view_head_type = str(per_view_head_type or cluster_head_type)
        if self.per_view_head_type not in {"student_t", "cosine"}:
            raise ValueError("per_view_head_type must be student_t or cosine")
        # A2 keeps a separate prototype coordinate system for every view.  The
        # common meaning of row k is enforced during initialization and upload
        # alignment rather than by sharing a prototype tensor.
        self.prototype_heads = nn.ModuleList([
            ClusteringHead(
                num_clusters,
                embedding_dim,
                alpha=alpha,
                head_type=self.per_view_head_type,
                distance_scale=student_t_distance_scale,
                cosine_temperature=per_view_temperature,
            )
            for _ in view_dims
        ])
    def observed_fusion(self, views, mask):
        if mask is None:
            mask = torch.ones((views[0].shape[0], len(views)), device=views[0].device,
                              dtype=torch.bool)
        else:
            mask = mask.to(device=views[0].device, dtype=torch.bool)
        if mask.shape != (views[0].shape[0], len(self.view_models)) or not mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one observed view")
        view_embeddings = []
        for view_index, (model, view) in enumerate(zip(self.view_models, views)):
            embedding = view.new_zeros((len(view), self.embedding_dim))
            observed = mask[:, view_index]
            if observed.any():
                embedding = embedding.index_copy(0, observed.nonzero(as_tuple=True)[0],
                                                 model.encode(view[observed]))
            view_embeddings.append(embedding)
        normalized = [F.normalize(embedding, dim=1) for embedding in view_embeddings]
        weights = torch.softmax(self.view_logits, dim=0)
        visible_weights = mask.to(weights.dtype) * weights.unsqueeze(0)
        fused = sum(
            visible_weights[:, index:index + 1] * embedding
            for index, embedding in enumerate(normalized)
        )
        fused = fused / visible_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return view_embeddings, F.normalize(fused, dim=1)

    def encode(self, views, mask=None):
        if mask is None:
            view_embeddings = [model.encode(view) for model, view in zip(self.view_models, views)]
            normalized = [F.normalize(embedding, dim=1) for embedding in view_embeddings]
            weights = torch.softmax(self.view_logits, dim=0)
            fused = sum(weight * embedding for weight, embedding in zip(weights, normalized))
            return view_embeddings, F.normalize(fused, dim=1)
        # A2 is an observation-mask model: hidden rows never enter its fused
        # evidence or per-view assignments.
        return self.observed_fusion(views, mask)

    def clustering_heads(self):
        """Return clustering modules without aliasing them in the state dict."""
        return tuple(self.prototype_heads)

    def prototype_center_keys(self):
        return tuple(
            f"prototype_heads.{index}.centers"
            for index in range(len(self.prototype_heads))
        )

    @torch.no_grad()
    def project_cluster_centers_(self, references=None):
        """Project every per-view cosine head onto the unit sphere."""
        heads = self.clustering_heads()
        if references is None:
            references = (None,) * len(heads)
        elif isinstance(references, torch.Tensor):
            references = (references,)
        if len(references) != len(heads):
            raise ValueError("One reference center tensor is required per clustering head")
        for head, reference in zip(heads, references):
            head.project_centers_(reference=reference)

    def _per_view_assignments(self, view_embeddings, mask):
        batch_size = view_embeddings[0].shape[0]
        if mask is None:
            mask = torch.ones(
                (batch_size, len(view_embeddings)),
                device=view_embeddings[0].device,
                dtype=torch.bool,
            )
        else:
            mask = mask.to(device=view_embeddings[0].device, dtype=torch.bool)
        if mask.shape != (batch_size, len(view_embeddings)) or not mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one observed view")

        assignments = []
        for view_index, (head, embedding) in enumerate(
            zip(self.prototype_heads, view_embeddings)
        ):
            observed = mask[:, view_index]
            # Missing rows are never passed through a prototype head. Besides
            # excluding their values, this makes their gradient contribution
            # exactly zero rather than relying on a later multiplication by 0.
            assignment = embedding.new_zeros((batch_size, self.num_clusters))
            if observed.any():
                indices = observed.nonzero(as_tuple=True)[0]
                assignment = assignment.index_copy(
                    0, indices, head(F.normalize(embedding[observed], dim=1))
                )
            assignments.append(assignment)
        stacked = torch.stack(assignments, dim=1)
        visible = mask.to(stacked.dtype).unsqueeze(2)
        fused = (stacked * visible).sum(dim=1)
        fused = fused / visible.sum(dim=1).clamp_min(1.0)
        return assignments, fused, mask

    def forward(self, views, mask=None):
        view_embeddings, fused = self.encode(views, mask=mask)
        reconstructions = [
            model.decode(embedding) for model, embedding in zip(self.view_models, view_embeddings)
        ]
        result = {
            "view_embeddings": view_embeddings,
            "embedding": fused,
            "reconstructions": reconstructions,
        }
        view_assignments, assignments, assignment_mask = self._per_view_assignments(
            view_embeddings, mask
        )
        result.update({
            "assignments": assignments,
            "view_assignments": view_assignments,
            "assignment_mask": assignment_mask,
        })
        return result


def target_distribution(assignments):
    frequency = assignments.sum(dim=0).clamp_min(1e-12)
    weight = assignments.pow(2) / frequency
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)


def clustering_objective(
    outputs,
    views,
    loss_weights,
    clustering_enabled,
    clustering_weight_scale=1.0,
    target_assignments=None,
    mask=None,
):
    if mask is None:
        reconstruction = torch.stack([
            F.mse_loss(reconstructed, original)
            for reconstructed, original in zip(outputs["reconstructions"], views)
        ]).mean()
    else:
        terms = [
            F.mse_loss(reconstructed[mask[:, index]], original[mask[:, index]])
            for index, (reconstructed, original) in enumerate(
                zip(outputs["reconstructions"], views)
            ) if mask[:, index].any()
        ]
        reconstruction = torch.stack(terms).mean()

    fused = outputs["embedding"]
    if mask is None:
        consistency = torch.stack([
            F.mse_loss(F.normalize(embedding, dim=1), fused)
            for embedding in outputs["view_embeddings"]
        ]).mean()
    else:
        terms = [
            F.mse_loss(F.normalize(embedding[mask[:, index]], dim=1),
                       fused[mask[:, index]])
            for index, embedding in enumerate(outputs["view_embeddings"])
            if mask[:, index].any()
        ]
        consistency = torch.stack(terms).mean()

    assignments = outputs["assignments"]
    if clustering_enabled:
        target = (
            target_distribution(assignments.detach())
            if target_assignments is None
            else target_assignments.detach()
        )
        if target.shape != assignments.shape:
            raise ValueError(
                f"Target shape {tuple(target.shape)} does not match "
                f"assignment shape {tuple(assignments.shape)}"
            )
        fused_clustering = F.kl_div(
            assignments.clamp_min(1e-12).log(), target, reduction="batchmean"
        )
        view_semantic = assignments.new_zeros(())
        if "view_assignments" in outputs:
            assignment_mask = outputs["assignment_mask"]
            semantic_terms = []
            for view_index, view_assignment in enumerate(outputs["view_assignments"]):
                observed = assignment_mask[:, view_index]
                if observed.any():
                    semantic_terms.append(F.kl_div(
                        view_assignment[observed].clamp_min(1e-12).log(),
                        target[observed],
                        reduction="batchmean",
                    ))
            if semantic_terms:
                # Equal weighting prevents a high-coverage view from silently
                # becoming a learned gate. This remains part of the existing
                # clustering objective, so the total still has four losses.
                view_semantic = torch.stack(semantic_terms).mean()
        semantic_weight = float(loss_weights.get("view_semantic", 0.0))
        clustering = fused_clustering + semantic_weight * view_semantic
        mean_assignment = assignments.mean(dim=0)
        balance = torch.sum(mean_assignment * torch.log(mean_assignment * assignments.shape[1] + 1e-12))
    else:
        fused_clustering = assignments.new_zeros(())
        view_semantic = assignments.new_zeros(())
        clustering = assignments.new_zeros(())
        balance = assignments.new_zeros(())

    clustering_weight_scale = float(clustering_weight_scale) if clustering_enabled else 0.0
    total = (
        loss_weights["reconstruction"] * reconstruction
        + loss_weights["consistency"] * consistency
        + clustering_weight_scale * loss_weights["clustering"] * clustering
        + clustering_weight_scale * loss_weights["balance"] * balance
    )
    metrics = {
        "loss": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "consistency": float(consistency.detach()),
        "clustering": float(clustering.detach()),
        "balance": float(balance.detach()),
        "weighted_reconstruction": float(
            (loss_weights["reconstruction"] * reconstruction).detach()
        ),
        "weighted_consistency": float(
            (loss_weights["consistency"] * consistency).detach()
        ),
        "weighted_clustering": float(
            (clustering_weight_scale * loss_weights["clustering"] * clustering).detach()
        ),
        "weighted_balance": float(
            (clustering_weight_scale * loss_weights["balance"] * balance).detach()
        ),
        "confidence": float(assignments.max(dim=1).values.mean().detach()),
        "clustering_weight_scale": clustering_weight_scale,
    }
    if "view_assignments" in outputs:
        metrics.update({
            "fused_clustering": float(fused_clustering.detach()),
            "view_semantic": float(view_semantic.detach()),
            "view_semantic_weight": float(loss_weights.get("view_semantic", 0.0)),
        })
    return total, metrics
