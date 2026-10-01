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
    """DEC-style trainable cluster centers returning Student-t assignments."""
    def __init__(self, num_clusters, embedding_dim, alpha=1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.centers = nn.Parameter(torch.empty(num_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embedding):
        distance = torch.sum((embedding.unsqueeze(1) - self.centers.unsqueeze(0)) ** 2, dim=2)
        assignments = (1.0 + distance / self.alpha).pow(-(self.alpha + 1.0) / 2.0)
        return assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)


class MultiViewClusteringModel(nn.Module):
    def __init__(self, view_dims, num_clusters, hidden_dims, embedding_dim, dropout=0.0,
                 alpha=1.0, completion=None):
        super().__init__()
        self.view_models = nn.ModuleList(
            ViewAutoencoder(dim, hidden_dims, embedding_dim, dropout) for dim in view_dims
        )
        self.view_logits = nn.Parameter(torch.zeros(len(view_dims)))
        self.cluster_head = ClusteringHead(num_clusters, embedding_dim, alpha=alpha)
        completion = completion or {}
        self.completion_mode = completion.get("method", "none")
        if self.completion_mode not in {"none", "attention"}:
            raise ValueError("Only attention missing-view completion is supported")
        if self.completion_mode == "attention":
            heads = int(completion.get("heads", 4))
            if embedding_dim % heads:
                raise ValueError("embedding_dim must be divisible by completion.heads")
            self.attention_heads = heads
            self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
            self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

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
            embedding = view.new_zeros((len(view), self.cluster_head.centers.shape[1]))
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

    def _correct(self, fused, mask, anchors):
        if anchors is None or anchors.shape[0] == 0 or mask.all():
            return fused
        missing = ~mask.all(dim=1)
        query = fused[missing]
        head_dim = query.shape[1] // self.attention_heads
        q = self.query(query).reshape(-1, self.attention_heads, head_dim).transpose(0, 1)
        k = self.key(anchors).reshape(-1, self.attention_heads, head_dim).transpose(0, 1)
        v = self.value(anchors).reshape(-1, self.attention_heads, head_dim).transpose(0, 1)
        weights = torch.softmax((q @ k.transpose(1, 2)) / head_dim ** 0.5, dim=-1)
        correction = (weights @ v).transpose(0, 1).reshape(len(query), -1)
        return fused.index_copy(
            0, missing.nonzero(as_tuple=True)[0],
            F.normalize(query + correction, dim=1),
        )

    def encode(self, views, mask=None, anchors=None):
        if mask is None:
            view_embeddings = [model.encode(view) for model, view in zip(self.view_models, views)]
            normalized = [F.normalize(embedding, dim=1) for embedding in view_embeddings]
            weights = torch.softmax(self.view_logits, dim=0)
            fused = sum(weight * embedding for weight, embedding in zip(weights, normalized))
            return view_embeddings, F.normalize(fused, dim=1)
        embeddings, observed = self.observed_fusion(views, mask)
        if self.completion_mode == "none" or anchors is None or len(anchors) == 0:
            return embeddings, observed
        corrected = self._correct(observed, mask, anchors)
        weights = torch.softmax(self.view_logits, dim=0)
        filled = []
        for index, embedding in enumerate(embeddings):
            filled.append(torch.where(
                mask[:, index:index + 1], F.normalize(embedding, dim=1), corrected
            ))
        fused = sum(weight * embedding for weight, embedding in zip(weights, filled))
        return embeddings, F.normalize(fused, dim=1)

    def forward(self, views, mask=None, anchors=None):
        view_embeddings, fused = self.encode(views, mask=mask, anchors=anchors)
        reconstructions = [
            model.decode(embedding) for model, embedding in zip(self.view_models, view_embeddings)
        ]
        assignments = self.cluster_head(fused)
        return {
            "view_embeddings": view_embeddings,
            "embedding": fused,
            "reconstructions": reconstructions,
            "assignments": assignments,
        }


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
        clustering = F.kl_div(assignments.clamp_min(1e-12).log(), target, reduction="batchmean")
        mean_assignment = assignments.mean(dim=0)
        balance = torch.sum(mean_assignment * torch.log(mean_assignment * assignments.shape[1] + 1e-12))
    else:
        clustering = assignments.new_zeros(())
        balance = assignments.new_zeros(())

    clustering_weight_scale = float(clustering_weight_scale) if clustering_enabled else 0.0
    total = (
        loss_weights["reconstruction"] * reconstruction
        + loss_weights["consistency"] * consistency
        + clustering_weight_scale * loss_weights["clustering"] * clustering
        + clustering_weight_scale * loss_weights["balance"] * balance
    )
    return total, {
        "loss": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "consistency": float(consistency.detach()),
        "clustering": float(clustering.detach()),
        "balance": float(balance.detach()),
        "confidence": float(assignments.max(dim=1).values.mean().detach()),
        "clustering_weight_scale": clustering_weight_scale,
    }
