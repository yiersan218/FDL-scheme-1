"""客户端潜在补全：留一视图自监督、受限完成融合与真实观测边界。"""

import torch
from torch import nn
from torch.nn import functional as F


COMPLETION_DEFAULTS = {
    "enabled": True,
    "anchor_size": 256,
    "topk": 16,
    "heads": 4,
    "adapter_delta": 0.1,
    "attention_strength": 0.3,
    "fusion_weight": 0.25,
    "temperature": 0.2,
    "truncation": 9,
    "use_attention": True,
    "use_completed_initialization": True,
    "use_completed_assignments": True,
    "gate": True,
    "value_mode": "per_view",
    "alignment_mode": "mse",
    "parameter_scope": "federated",
    "seed": 0,
}


def completion_schedule(round_number, center_init_round):
    """从首轮的零强度逐步增加补全贡献，并将调度值限制在 [0, 1]。"""
    denominator = max(1, int(center_init_round) // 2)
    return min(1.0, max(0.0, (int(round_number) - 1) / denominator))


def truncated_robust_loss(probabilities, truncation):
    """用恰好 C 项有限级数近似负对数损失，并保持输入张量形状。"""
    remainder = 1.0 - probabilities.clamp(0.0, 1.0)
    return sum(remainder.pow(order) / order for order in range(1, int(truncation) + 1))


class LocalLatentCompletion(nn.Module):
    """利用本地完整锚点预测目标视图，并以真实观测目标训练补全器。"""

    def __init__(self, num_views, embedding_dim, config):
        """校验补全约束，创建逐视图残差适配器和共享多头注意力。"""
        super().__init__()
        self.config = dict(config)
        self.num_views = int(num_views)
        self.embedding_dim = int(embedding_dim)
        heads = int(config["heads"])
        if heads <= 0 or embedding_dim % heads:
            raise ValueError("completion.heads must divide embedding_dim")
        if not 0 <= float(config["adapter_delta"]) <= 0.1:
            raise ValueError("completion.adapter_delta must be in [0, 0.1]")
        if not 0 <= float(config["fusion_weight"]) <= 0.25:
            raise ValueError("completion.fusion_weight must be in [0, 0.25]")
        if not 0 <= float(config["attention_strength"]) <= 0.3:
            raise ValueError("completion.attention_strength must be in [0, 0.3]")
        if float(config["temperature"]) <= 0 or int(config["truncation"]) < 1:
            raise ValueError("completion temperature and truncation must be positive")
        if int(config["topk"]) < 1 or int(config["anchor_size"]) < 1:
            raise ValueError("completion topk and anchor_size must be positive")
        if config["value_mode"] not in {"per_view", "consensus"}:
            raise ValueError("completion.value_mode must be per_view or consensus")
        if config["alignment_mode"] not in {"mse", "contrastive", "paired"}:
            raise ValueError("completion.alignment_mode must be mse, contrastive, or paired")
        if config["parameter_scope"] not in {"federated", "local"}:
            raise ValueError("completion.parameter_scope must be federated or local")
        self.adapter_residual = nn.Parameter(torch.zeros(num_views, embedding_dim, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, heads, dropout=0.0, batch_first=True)

    def adapter_matrix(self, view_index):
        """构造单位矩阵加有界残差的 [d, d] 视图映射，限制适配偏移。"""
        identity = torch.eye(self.embedding_dim, device=self.adapter_residual.device,
                             dtype=self.adapter_residual.dtype)
        return identity + (float(self.config["adapter_delta"]) / self.embedding_dim) * torch.tanh(
            self.adapter_residual[view_index]
        )

    @staticmethod
    def _median(scores, allowed):
        """对每行允许的候选求中位数，排除样本自身及无效锚点。"""
        # 将有效候选排在前面，同时允许每个样本排除不同的自身 ID。
        sorted_scores = scores.masked_fill(~allowed, float("inf")).sort(dim=1).values
        counts = allowed.sum(dim=1)
        lower = ((counts - 1).clamp_min(0) // 2).unsqueeze(1)
        upper = (counts.clamp_min(1) // 2).unsqueeze(1)
        return (sorted_scores.gather(1, lower) + sorted_scores.gather(1, upper)).squeeze(1) / 2

    def _predict(self, normalized, mask, sample_ids, bank, weights, view_index, scale):
        """排除目标视图后，以源视图共识检索锚点并预测目标表示。

        Args:
            normalized: V 个 [B, d] 归一化观测编码。
            mask: [B, V] 真实观测掩码。
            sample_ids: 长度 B 的样本标识，用于排除同 ID 锚点。
            bank: 本地完整样本的停止梯度编码及标识。
            weights: 长度 V 的全局视图融合权重。
            view_index: 当前待预测的目标视图编号。
            scale: 本轮补全贡献的调度系数。

        Returns:
            [B, d] 预测、[B] 源侧匹配支持度和 [B] 是否存在源视图的标记。
            支持度仅描述邻居匹配，不代表目标预测的真实准确概率。
        """
        batch_size = len(sample_ids)
        prediction = normalized[0].new_zeros((batch_size, self.embedding_dim))
        confidence = normalized[0].new_zeros(batch_size)
        source_mask = mask.clone()
        # 即使目标视图真实可见，也必须从 Query/Key 的源信息中剔除。
        source_mask[:, view_index] = False
        rows = source_mask.any(dim=1).nonzero(as_tuple=True)[0]
        if not len(rows):
            return prediction, confidence, source_mask.any(dim=1)

        bank_ids = bank["sample_ids"].to(sample_ids.device)
        bank_values = tuple(F.normalize(value.detach().to(normalized[0]), dim=1)
                            for value in bank["view_embeddings"])
        if len(bank_values) != self.num_views or any(
            value.shape != (len(bank_ids), self.embedding_dim) for value in bank_values
        ):
            raise ValueError("Completion bank has invalid view embedding shapes")
        if self.config["value_mode"] == "consensus":
            value_embeddings = F.normalize(sum(weights[v] * bank_values[v]
                                               for v in range(self.num_views)), dim=1)
        else:
            value_embeddings = bank_values[view_index]

        # 相同观测模式共用源视图集合，避免为每个样本重复构造锚点 Key。
        patterns, inverse = torch.unique(source_mask[rows], dim=0, return_inverse=True)
        for pattern_index, pattern in enumerate(patterns):
            group_rows = rows[inverse == pattern_index]
            sources = pattern.nonzero(as_tuple=True)[0].tolist()
            denominator = weights[sources].sum().clamp_min(1e-12)
            query = F.normalize(sum(weights[v] * normalized[v][group_rows]
                                    for v in sources) / denominator, dim=1)
            adapted = F.linear(query, self.adapter_matrix(view_index))
            gate = query.new_zeros(len(group_rows))
            attention_output = torch.zeros_like(query)
            if len(bank_ids) and self.config["use_attention"]:
                keys = F.normalize(sum(weights[v] * bank_values[v] for v in sources)
                                   / denominator, dim=1)
                valid_keys = (keys.norm(dim=1) > 1e-12) & (value_embeddings.norm(dim=1) > 1e-12)
                allowed = ((sample_ids[group_rows, None] != bank_ids[None, :])
                           & valid_keys[None, :])
                allowed &= (query.norm(dim=1) > 1e-12)[:, None]
                scores = query @ keys.T
                support = allowed.sum(dim=1) >= 2
                if support.any():
                    supported_rows = support.nonzero(as_tuple=True)[0]
                    supported_scores = scores[support]
                    supported_allowed = allowed[support]
                    if self.config["gate"]:
                        median = self._median(supported_scores, supported_allowed)
                        maximum = supported_scores.masked_fill(~supported_allowed, -float("inf")).max(1).values
                        support_gate = ((maximum - median) / (1 - median + 1e-12)).clamp(0, 1).detach()
                    else:
                        support_gate = query.new_ones(len(supported_rows))
                    gate = gate.index_copy(0, supported_rows, support_gate)
                    active = support_gate > 0
                    if active.any() and scale > 0:
                        active_rows = supported_rows[active]
                        active_allowed = allowed[active_rows]
                        selected = scores[active_rows].masked_fill(~active_allowed, -float("inf")).topk(
                            min(int(self.config["topk"]), len(bank_ids)), dim=1, sorted=True
                        ).indices
                        selected_allowed = active_allowed.gather(1, selected)
                        attended, _ = self.attention(
                            query[active_rows].unsqueeze(1), keys[selected], value_embeddings[selected],
                            key_padding_mask=~selected_allowed, need_weights=False,
                        )
                        attention_output = attention_output.index_copy(
                            0, active_rows, F.normalize(attended.squeeze(1), dim=1)
                        )
            group_prediction = F.normalize(
                adapted + float(self.config["attention_strength"]) * scale * gate[:, None] * attention_output,
                dim=1,
            )
            prediction = prediction.index_copy(0, group_rows, group_prediction)
            confidence = confidence.index_copy(0, group_rows, gate)
        return prediction, confidence, source_mask.any(dim=1)

    def forward(self, outputs, mask, sample_ids, bank, view_logits, round_number,
                center_init_round, teacher_embeddings=None, compute_supervision=True):
        """生成独立补全表示、有限贡献融合和统一的补全自监督损失。

        Args:
            outputs: 原模型的真实观测编码与融合结果。
            mask: [B, V] 观测掩码；不会被补全预测覆盖。
            sample_ids: 长度 B 的样本标识。
            bank: 本地完整锚点库，不参与反向传播或通信。
            view_logits: 长度 V 的可训练视图权重 logits。
            round_number: 从 1 开始的通信轮号。
            center_init_round: 预训练内部的中心初始化轮数。
            teacher_embeddings: 可选的固定教师编码；缺省使用当前观测编码。
            compute_supervision: 是否计算观测目标的补全自监督损失。

        Returns:
            估计表示、完成融合、固定混合权重、自监督损失及诊断指标。
            教师停止梯度；真正隐藏的目标值从不用于补全监督。
        """
        normalized = tuple(F.normalize(value, dim=1) for value in outputs["view_embeddings"])
        batch_size = normalized[0].shape[0]
        mask = (torch.ones((batch_size, self.num_views), dtype=torch.bool, device=normalized[0].device)
                if mask is None else mask.to(device=normalized[0].device, dtype=torch.bool))
        sample_ids = torch.as_tensor(sample_ids, device=normalized[0].device, dtype=torch.long).reshape(-1)
        if mask.shape != (batch_size, self.num_views) or len(sample_ids) != batch_size or not mask.any(1).all():
            raise ValueError("Completion mask or sample IDs have invalid shapes")
        scale = completion_schedule(round_number, center_init_round)
        weights = torch.softmax(view_logits, dim=0)
        predictions, confidences, source_valid = [], [], []
        for view_index in range(self.num_views):
            predicted, confidence, has_source = self._predict(
                normalized, mask, sample_ids, bank, weights, view_index, scale
            )
            predictions.append(predicted)
            confidences.append(confidence)
            source_valid.append(has_source)
        confidence = torch.stack(confidences, dim=1)
        prediction_valid = torch.stack([value.norm(dim=1) > 1e-12 for value in predictions], dim=1)
        # 权重只表达本次检索支持，不能通过降低补全占比来逃避聚类监督。
        supported_weights = (
            weights[None, :] * (~mask).to(weights.dtype) * confidence * prediction_valid
        ).detach()
        missing_weights = (weights[None, :] * (~mask).to(weights.dtype)).sum(dim=1).detach()
        support_weights = supported_weights.sum(dim=1)
        imputed_sum = sum(supported_weights[:, v:v + 1] * predictions[v] for v in range(self.num_views))
        observed_fused = outputs["embedding"]
        valid_fusion = ((support_weights > 1e-12) & (imputed_sum.norm(dim=1) > 1e-12)
                        & (observed_fused.norm(dim=1) > 1e-12))
        beta = (float(self.config["fusion_weight"]) * scale * support_weights
                / missing_weights.clamp_min(1e-12)).clamp(0, 0.25)
        beta = torch.where(valid_fusion, beta, torch.zeros_like(beta)).detach()
        completed_fused = observed_fused.clone()
        if valid_fusion.any():
            rows = valid_fusion.nonzero(as_tuple=True)[0]
            fused = F.normalize((1 - beta[rows, None]) * observed_fused[rows]
                                + beta[rows, None] * F.normalize(imputed_sum[rows], dim=1), dim=1)
            completed_fused = completed_fused.index_copy(0, rows, fused)
        completed, imputed = [], []
        for view_index, predicted in enumerate(predictions):
            missing_rows = (~mask[:, view_index]).nonzero(as_tuple=True)[0]
            completed.append(normalized[view_index].index_copy(0, missing_rows, predicted[missing_rows]))
            imputed.append(torch.zeros_like(predicted).index_copy(0, missing_rows, predicted[missing_rows]))

        teachers = normalized if teacher_embeddings is None else teacher_embeddings
        if len(teachers) != self.num_views or any(value.shape != normalized[0].shape for value in teachers):
            raise ValueError("Completion teachers must have one [batch, embedding_dim] tensor per view")
        # 目标编码停止梯度，监督有效行由真实观测掩码决定，不使用隐藏真值。
        teachers = tuple(F.normalize(value.detach().to(normalized[0]), dim=1) for value in teachers)
        terms, supervised_pairs, positive_pair_cosines = [], 0, []
        if compute_supervision:
            for view_index, predicted in enumerate(predictions):
                targets = (mask[:, view_index] & (teachers[view_index].norm(dim=1) > 1e-12))
                supervision = (targets & source_valid[view_index] & prediction_valid[:, view_index])
                if not supervision.any():
                    continue
                query_rows = supervision.nonzero(as_tuple=True)[0]
                pair_cosine = (predicted[query_rows] * teachers[view_index][query_rows]).sum(dim=1).clamp(-1, 1)
                if self.config["alignment_mode"] == "mse":
                    # 按每个有效目标视图的监督行与表示维度取均值，再等权平均视图。
                    # 完整与缺失样本使用同一目标，真正隐藏的目标没有监督资格。
                    terms.append(F.mse_loss(predicted[query_rows], teachers[view_index][query_rows]))
                elif self.config["alignment_mode"] == "paired":
                    # paired 是不含实例负样本的独立候选，
                    # 除以表示维度以保持与原 MSE 的逐特征量纲相近。
                    probability = (1 + pair_cosine) / 2
                    terms.append(truncated_robust_loss(probability, self.config["truncation"]).mean()
                                 / self.embedding_dim)
                else:
                    if targets.sum() < 2:
                        continue
                    target_rows = targets.nonzero(as_tuple=True)[0]
                    logits = predicted[query_rows] @ teachers[view_index][target_rows].T / float(self.config["temperature"])
                    positive_mask = sample_ids[query_rows, None] == sample_ids[target_rows][None, :]
                    # 通常 ID 唯一；若出现重复 ID，则合并正例概率，避免制造假负例。
                    probability = (logits.log_softmax(dim=1).exp() * positive_mask).sum(dim=1)
                    terms.append(truncated_robust_loss(probability, self.config["truncation"]).mean())
                positive_pair_cosines.append(pair_cosine.detach())
                supervised_pairs += len(query_rows)
        completion_loss = (torch.stack(terms).mean() if terms else sum(value.sum() * 0 for value in normalized))
        missing = ~mask
        cross_view_cosines = []
        for first in range(self.num_views):
            for second in range(first + 1, self.num_views):
                paired = (mask[:, first] & mask[:, second]
                          & (normalized[first].norm(dim=1) > 1e-12)
                          & (normalized[second].norm(dim=1) > 1e-12))
                if paired.any():
                    cross_view_cosines.append((normalized[first][paired].detach()
                                               * normalized[second][paired].detach()).sum(dim=1))
        visible_weights = mask.to(weights.dtype) * weights[None, :]
        observed_before_norm = sum(visible_weights[:, v:v + 1] * normalized[v].detach()
                                   for v in range(self.num_views))
        observed_before_norm = observed_before_norm / visible_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        metrics = {
            "completion_supervised_pairs": float(supervised_pairs),
            "completion_robust_pairs": float(supervised_pairs),
            "completion_mse": float(completion_loss.detach()) if self.config["alignment_mode"] == "mse" else 0.0,
            "completion_anchor_count": float(len(bank["sample_ids"])),
            "completion_confidence": float(confidence[missing].mean().detach()) if missing.any() else 0.0,
            "completion_beta": float(beta.mean().detach()),
            "completion_beta_max": float(beta.max().detach()) if batch_size else 0.0,
            "completion_fallback_rate": float((~valid_fusion[missing.any(1)]).float().mean()) if missing.any() else 0.0,
            "completion_schedule": float(scale),
            "completion_cross_view_cosine": float(torch.cat(cross_view_cosines).mean()) if cross_view_cosines else 0.0,
            "completion_observed_cancellation_rate": float((observed_before_norm.norm(dim=1) <= 1e-6).float().mean()),
            "completion_positive_pair_cosine": float(torch.cat(positive_pair_cosines).mean()) if positive_pair_cosines else 0.0,
        }
        return {
            "completion_enabled": True,
            "imputed_embeddings": tuple(imputed),
            "completed_embeddings": tuple(completed),
            "completed_embedding": completed_fused,
            "completion_confidence": confidence,
            "completion_beta": beta,
            "completion_support_weights": supported_weights,
            "robust_predictions": tuple(predictions),
            "completion_loss": completion_loss,
            # 旧报告读取此字段；它是同一自监督项的别名，不额外加入总损失。
            "robust_loss": completion_loss,
            "completion_metrics": metrics,
        }
