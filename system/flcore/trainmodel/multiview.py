"""缺失联邦多视图模型：视图编码、独立原型与补全概率融合。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .completion import COMPLETION_DEFAULTS, LocalLatentCompletion


def _mlp(dimensions, dropout=0.0, final_activation=False):
    """按层宽构造 MLP，隐藏层使用 LayerNorm、ReLU 和可选 Dropout。"""
    layers = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
        layers.append(nn.Linear(input_dim, output_dim))
        is_last = index == len(dimensions) - 2
        if not is_last or final_activation:
            # LayerNorm 不保存客户端专属的滑动统计量，
            # 可稳定处理联邦划分后常见的小尾批次。
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ViewAutoencoder(nn.Module):
    """单个视图的独立编码器与对称解码器。"""

    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout):
        """按输入维度、隐藏层宽度和公共表示维度建立自编码器。"""
        super().__init__()
        encoder_dims = [input_dim, *hidden_dims, embedding_dim]
        decoder_dims = [embedding_dim, *reversed(hidden_dims), input_dim]
        self.encoder = _mlp(encoder_dims, dropout=dropout)
        self.decoder = _mlp(decoder_dims, dropout=dropout)

    def encode(self, view):
        """将形状为 [B, D_v] 的视图输入编码为 [B, d] 表示。"""
        return self.encoder(view)

    def decode(self, embedding):
        """将 [B, d] 潜在表示重构为该视图的 [B, D_v] 输入空间。"""
        return self.decoder(embedding)


class ClusteringHead(nn.Module):
    """使用 Student-t 分配的可训练单视图原型头。"""

    def __init__(
        self,
        num_clusters,
        embedding_dim,
        alpha=1.0,
        head_type="student_t",
        distance_scale=1.0,
    ):
        """校验分配参数并初始化形状为 [K, d] 的可训练中心。"""
        super().__init__()
        self.alpha = float(alpha)
        self.head_type = str(head_type)
        self.distance_scale = float(distance_scale)
        if self.head_type != "student_t":
            raise ValueError("head_type only supports student_t")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not torch.isfinite(torch.tensor(self.distance_scale)) or self.distance_scale <= 0:
            raise ValueError("distance_scale must be finite and positive")
        self.centers = nn.Parameter(torch.empty(num_clusters, embedding_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, embedding, detach_centers=False):
        """计算 Student-t 分配；预测支路可固定中心，仅向表示反传梯度。"""
        centers = self.centers.detach() if detach_centers else self.centers
        distance = torch.sum((embedding.unsqueeze(1) - centers.unsqueeze(0)) ** 2, dim=2)
        assignments = (
            1.0 + self.distance_scale * distance / self.alpha
        ).pow(-(self.alpha + 1.0) / 2.0)
        return assignments / assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)


class MultiViewClusteringModel(nn.Module):
    """组合各视图自编码器、A2 独立原型与可选客户端潜在补全模块。"""

    def __init__(self, view_dims, num_clusters, hidden_dims, embedding_dim, dropout=0.0,
                 alpha=1.0, cluster_head_type="student_t",
                 student_t_distance_scale=1.0,
                 prototype_mode="per_view",
                 per_view_head_type=None, completion=None):
        """创建逐视图模型并校验 A2 拓扑及补全参数作用域。

        每个视图保留独立中心坐标系；补全模块独立初始化随机流，
        以保持开启补全前的基础模型参数和后续随机采样不受影响。
        """
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
        if str(cluster_head_type) != "student_t":
            raise ValueError("cluster_head_type only supports student_t")
        self.per_view_head_type = str(per_view_head_type or cluster_head_type)
        if self.per_view_head_type != "student_t":
            raise ValueError("per_view_head_type only supports student_t")
        # A2 为每个视图保留独立原型坐标系；共同簇槽位由初始化和上传对齐保证，
        # 无需在不同视图之间共享同一个原型张量。
        self.prototype_heads = nn.ModuleList([
            ClusteringHead(
                num_clusters,
                embedding_dim,
                alpha=alpha,
                head_type=self.per_view_head_type,
                distance_scale=student_t_distance_scale,
            )
            for _ in view_dims
        ])
        self.completion_config = {**COMPLETION_DEFAULTS, **(completion or {})}
        self.completion_enabled = bool(self.completion_config["enabled"])
        self.completion_parameter_scope = self.completion_config["parameter_scope"]
        if self.completion_parameter_scope not in {"federated", "local"}:
            raise ValueError("completion.parameter_scope must be federated or local")
        if self.completion_enabled:
            # 基础模块已完成初始化；隔离新增模块的随机流，
            # 使基线参数和后续随机采样保持一致。
            with torch.random.fork_rng(devices=[]):
                # torch.manual_seed 还会重设 CUDA 随机流；新模块在 CPU 创建，
                # 这里只设置可由 fork_rng 恢复的 CPU 随机生成器。
                torch.random.default_generator.manual_seed(int(self.completion_config["seed"]))
                self.completion_module = LocalLatentCompletion(
                    len(view_dims), self.embedding_dim, self.completion_config
                )

    def observed_fusion(self, views, mask):
        """仅编码真实可见行，并在每个样本的观测集合内融合表示。

        Args:
            views: V 个 [B, D_v] 视图张量。
            mask: [B, V] 布尔观测掩码；None 表示全部视图可见。

        Returns:
            逐视图 [B, d] 编码列表及归一化的 [B, d] 融合表示。
            缺失行的逐视图编码为零，不进入编码器。
        """
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
        """编码并融合多视图输入；存在掩码时严格使用真实观测行。"""
        if mask is None:
            view_embeddings = [model.encode(view) for model, view in zip(self.view_models, views)]
            normalized = [F.normalize(embedding, dim=1) for embedding in view_embeddings]
            weights = torch.softmax(self.view_logits, dim=0)
            fused = sum(weight * embedding for weight, embedding in zip(weights, normalized))
            return view_embeddings, F.normalize(fused, dim=1)
        # A2 严格依据观测掩码，隐藏行不参与融合证据或逐视图分配。
        return self.observed_fusion(views, mask)

    def clustering_heads(self):
        """返回原型头元组，避免重复注册模块而产生 state_dict 别名。"""
        return tuple(self.prototype_heads)

    def prototype_center_keys(self):
        """返回各视图原型中心在 state_dict 中的键，供独立传输与聚合。"""
        return tuple(
            f"prototype_heads.{index}.centers"
            for index in range(len(self.prototype_heads))
        )

    def local_completion_state_keys(self):
        """返回需跨轮保存的客户端私有参数键；临时锚点库不属于模型状态。"""
        if not self.completion_enabled or self.completion_parameter_scope != "local":
            return ()
        return tuple(key for key in self.state_dict() if key.startswith("completion_module."))

    def _per_view_assignments(self, view_embeddings, mask):
        """计算各视图观测行的软分配，再对可见视图等权平均。

        Returns:
            V 个 [B, K] 分配张量、融合后的 [B, K] 分配和 [B, V] 掩码。
            缺失行不经过原型头，对原型计数和梯度的贡献均为零。
        """
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
            # 缺失行不经过原型头，其数值与梯度贡献都被直接隔离，
            # 而不是计算后再通过乘零进行屏蔽。
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

    @torch.no_grad()
    def build_completion_bank(self, views, mask, sample_ids, seed=0):
        """从本地完整样本建立停止梯度的外部锚点库。

        Args:
            views: 具有相同样本顺序的 V 个视图张量。
            mask: [N, V] 真实观测掩码；只选取整行全真的样本。
            sample_ids: 长度 N 的样本标识，用于检索时排除自身。
            seed: 超过容量上限时抽取锚点的固定种子。

        Returns:
            包含 sample_ids 和逐视图归一化编码的字典。
            锚点不注册为模型参数或缓冲区，也不随模型上传。
        """
        device = next(self.parameters()).device
        sample_ids = torch.as_tensor(sample_ids, device=device, dtype=torch.long).reshape(-1)
        if len(views) != len(self.view_models) or any(len(view) != len(sample_ids) for view in views):
            raise ValueError("Completion bank views and sample IDs have incompatible lengths")
        mask = (torch.ones((len(sample_ids), len(views)), dtype=torch.bool, device=device)
                if mask is None else mask.to(device=device, dtype=torch.bool))
        if mask.shape != (len(sample_ids), len(views)):
            raise ValueError("Completion bank mask has an invalid shape")
        complete_rows = mask.all(dim=1).nonzero(as_tuple=True)[0]
        cap = int(self.completion_config["anchor_size"])
        if len(complete_rows) > cap:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            selected = torch.randperm(len(complete_rows), generator=generator)[:cap].to(device)
            complete_rows = complete_rows[selected]
        previous_mode = self.training
        self.eval()
        try:
            embeddings = tuple(
                F.normalize(model.encode(view.to(device)[complete_rows]), dim=1).detach()
                if len(complete_rows) else next(model.parameters()).new_empty((0, self.embedding_dim))
                for model, view in zip(self.view_models, views)
            )
            return {"sample_ids": sample_ids[complete_rows].detach().clone(),
                    "view_embeddings": embeddings}
        finally:
            self.train(previous_mode)

    def completion_forward(self, outputs, mask, sample_ids, bank, round_number=1,
                           center_init_round=1, teacher_embeddings=None,
                           compute_supervision=True):
        """在补全开关启用时传递本地上下文；关闭时返回空扩展结果。"""
        if not self.completion_enabled:
            return {}
        return self.completion_module(
            outputs, mask, sample_ids, bank, self.view_logits, round_number,
            center_init_round, teacher_embeddings=teacher_embeddings,
            compute_supervision=compute_supervision,
        )

    def _completed_assignments(self, outputs):
        """将有支持的缺失视图概率有限混入观测概率，保持真实原型统计独立。

        预测支路使用停止梯度的中心，允许聚类梯度更新预测表示及源编码器，
        不允许生成的视图直接移动原型。混合系数和视图支持权重均停止梯度。
        """
        observed = outputs["observed_assignments"]
        mask = outputs["assignment_mask"]
        support = outputs["completion_support_weights"].detach()
        predicted_assignments = []
        for view_index, (head, predicted) in enumerate(
            zip(self.prototype_heads, outputs["imputed_embeddings"])
        ):
            active = (~mask[:, view_index]) & (support[:, view_index] > 0)
            assignment = observed.new_zeros(observed.shape)
            if active.any():
                rows = active.nonzero(as_tuple=True)[0]
                assignment = assignment.index_copy(
                    0, rows, head(F.normalize(predicted[rows], dim=1), detach_centers=True)
                )
            predicted_assignments.append(assignment)
        support_sum = support.sum(dim=1, keepdim=True)
        imputed = sum(
            support[:, index:index + 1] * assignment
            for index, assignment in enumerate(predicted_assignments)
        ) / support_sum.clamp_min(1e-12)
        # 无支持时用观测分布占位，并令混合系数严格为零。
        imputed = torch.where(support_sum > 1e-12, imputed, observed.detach())
        beta = outputs["completion_beta"].detach()
        beta = torch.where(support_sum[:, 0] > 1e-12, beta, torch.zeros_like(beta))
        if not self.completion_config["use_completed_assignments"]:
            beta = torch.zeros_like(beta)
        mixed = (1 - beta[:, None]) * observed + beta[:, None] * imputed
        return {
            "assignments": mixed,
            "imputed_assignments": imputed,
            "imputed_view_assignments": tuple(predicted_assignments),
            "assignment_beta": beta,
        }

    def forward(self, views, mask=None, completion_context=None):
        """计算观测路径，并在有本地上下文时执行补全与受限概率融合。

        Args:
            views: V 个 [B, D_v] 视图张量。
            mask: [B, V] 真实观测掩码；None 表示全部可见。
            completion_context: 本地锚点、样本标识及调度信息；None 不执行补全。

        Returns:
            含逐视图编码、重构、融合表示与软分配的字典。
            observed_assignments 与逐视图分配始终只来自真实观测；
            assignments 在正式补全路径中加入受限的缺失预测概率。
            无上下文调用保留纯观测路径，供固定 DEC 教师和真实计数使用。
        """
        view_embeddings, fused = self.encode(views, mask=mask)
        # 解码结果保留完整批次形状；缺失行占位输出由损失中的真实掩码排除。
        reconstructions = [
            model.decode(embedding) for model, embedding in zip(self.view_models, view_embeddings)
        ]
        result = {
            "view_embeddings": view_embeddings,
            "embedding": fused,
            "reconstructions": reconstructions,
            "completion_configured": self.completion_enabled,
        }
        view_assignments, assignments, assignment_mask = self._per_view_assignments(
            view_embeddings, mask
        )
        result.update({
            "assignments": assignments,
            "observed_assignments": assignments,
            "imputed_assignments": assignments.detach(),
            "assignment_beta": assignments.new_zeros(assignments.shape[0]),
            "view_assignments": view_assignments,
            "assignment_mask": assignment_mask,
        })
        if self.completion_enabled and completion_context is not None:
            result.update(self.completion_forward(
                result, assignment_mask, completion_context["sample_ids"],
                completion_context["bank"],
                round_number=completion_context.get("round_number", 1),
                center_init_round=completion_context.get("center_init_round", 1),
                teacher_embeddings=completion_context.get("teacher_embeddings"),
                compute_supervision=completion_context.get("compute_supervision", True),
            ))
            result.update(self._completed_assignments(result))
        return result
