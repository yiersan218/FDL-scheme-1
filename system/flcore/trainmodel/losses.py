"""统一聚类训练目标：观测重构、补全自监督、DEC 与簇均衡。"""

import torch
import torch.nn.functional as F


def target_distribution(assignments):
    """对 [N, K] 软分配作平方锐化与簇频率校正，生成 DEC 目标分布。"""
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
    """计算观测重构、补全自监督、DEC 聚类和簇均衡四项损失。

    Args:
        outputs: 多视图模型的前向输出字典。
        views: 用作重构监督的原始视图张量。
        loss_weights: 各顶层损失及可选视图语义消融的系数字典。
        clustering_enabled: 是否已完成原型初始化并启用聚类项。
        clustering_weight_scale: DEC 与均衡项在预训练内的渐增系数。
        target_assignments: 本轮固定的 [B, K] 目标；None 时由当前分配生成。
        mask: [B, V] 真实观测掩码，限制重构与一致性的有效行。

    Returns:
        可反向传播的总损失及已脱离计算图的标量指标字典。
        正式模型统一采用补全自监督；显式关闭补全的基线保留原融合一致性。
        consistency 指标及旧权重键仅为兼容别名，不额外添加一项损失。
    """
    if outputs.get("completion_configured", False) and not outputs.get("completion_enabled", False):
        raise ValueError(
            "Completion training requires completion_context; "
            "disable completion explicitly for the observed-only baseline"
        )
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
    # 完整与缺失数据共享同一补全目标，不根据缺失率切换损失类型。
    if outputs.get("completion_enabled", False):
        consistency = outputs["completion_loss"]
    elif mask is None:
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
            target_distribution(outputs.get("observed_assignments", assignments).detach())
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
                # 各有效视图等权，避免覆盖率较高的视图隐式占据更大权重。
                # 该消融项仍属于聚类目标；默认系数为零，顶层损失保持四项。
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
    completion_weight = float(
        loss_weights["completion"] if "completion" in loss_weights else loss_weights["consistency"]
    )
    total = (
        loss_weights["reconstruction"] * reconstruction
        + completion_weight * consistency
        + clustering_weight_scale * loss_weights["clustering"] * clustering
        + clustering_weight_scale * loss_weights["balance"] * balance
    )
    metrics = {
        "loss": float(total.detach()),
        "reconstruction": float(reconstruction.detach()),
        "consistency": float(consistency.detach()),
        "completion": float(consistency.detach()),
        "clustering": float(clustering.detach()),
        "balance": float(balance.detach()),
        "weighted_reconstruction": float(
            (loss_weights["reconstruction"] * reconstruction).detach()
        ),
        "weighted_consistency": float(
            (completion_weight * consistency).detach()
        ),
        "weighted_completion": float((completion_weight * consistency).detach()),
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
    if outputs.get("completion_enabled", False):
        metrics.update(outputs.get("completion_metrics", {}))
        metrics["robust_consistency"] = float(consistency.detach())
        metrics["assignment_beta"] = float(outputs["assignment_beta"].mean().detach())
    return total, metrics
