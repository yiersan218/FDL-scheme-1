"""联邦聚类客户端：固定观测掩码、本地训练、私有补全参数与中心摘要。"""

import copy
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from flcore.communication.compression import compress_client_update
from flcore.trainmodel.losses import clustering_objective, target_distribution
from utils.mat_data import MultiViewSubset, fixed_missing_mask


def _normalize_rows(values, eps=1e-12):
    """归一化非零行并返回有效行标记，零向量保持为零以避免 NaN。"""
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.zeros_like(values)
    valid = norms[:, 0] > eps
    normalized[valid] = values[valid] / norms[valid]
    return normalized, valid


class FederatedClusteringClient:
    """持有随机、互不重叠样本子集的横向联邦聚类客户端。"""

    def __init__(self, client_id, data, indices, config, device):
        """建立客户端子集、固定缺失掩码以及仅在本地保存的跨轮状态。"""
        self.id = int(client_id)
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.config = config
        self.device = device
        missing = config.get("missing", {})
        # 完整数据只是全真掩码；两种场景都仅以客户端观测值做归一化。
        rate = float(missing.get("rate", 0.0)) if missing.get("enabled", True) else 0.0
        mask = (
            np.ones((len(self.indices), len(data.views)), dtype=np.bool_)
            if rate == 0.0
            else fixed_missing_mask(
                len(self.indices), len(data.views), rate,
                int(config["training"]["seed"]), self.id, data.name,
            )
        )
        self.dataset = MultiViewSubset(
            data, self.indices, mask=mask,
            normalization=config["dataset"].get("normalization", "standard"),
        )
        # 保存压缩中未发送的更新量，下一次参与训练时进行误差反馈。
        self.compression_residual = None
        # 该私有补全参数缓存仅由客户端持有，不进入返回更新或上传包
        self.private_completion_state = None

    @staticmethod
    def _unpack(batch):
        """统一解包完整视图与缺失视图批次，缺省掩码返回 None。"""
        if len(batch) == 3:
            views, labels, indices = batch
            return views, labels, indices, None
        views, labels, indices, mask = batch
        return views, labels, indices, mask

    @property
    def num_samples(self):
        """返回该客户端持有的样本数，供 FedAvg 和批次大小计算。"""
        return len(self.dataset)

    def _loader(self, shuffle, seed_offset=0):
        """使用客户端独立的固定随机流创建批次加载器，不丢弃尾批次。"""
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

    def _completion_seed(self, round_number):
        """生成与划分、掩码和批次采样分离的锚点抽样种子。"""
        # 该随机流与数据划分、缺失掩码及批次采样相互独立。
        # 同轮内保持种子不变，使各 epoch 的锚点 ID 一致而编码可刷新。
        return int(self.config["training"]["seed"]) + self.id * 1009 + int(round_number) * 7919 + 104729

    @torch.no_grad()
    def _build_completion_bank(self, model, round_number):
        """只编码本地真实完整行，按当前模型重建停止梯度的锚点库。"""
        if not getattr(model, "completion_enabled", False):
            return None
        local_mask = self.dataset.mask
        if local_mask is None:
            local_mask = torch.ones((self.num_samples, len(self.dataset.views)), dtype=torch.bool)
        complete = local_mask.all(dim=1)
        # 只有掩码全真且确实观测到的完整样本会送入锚点编码器。
        views = tuple(view[complete].to(self.device) for view in self.dataset.views)
        return model.build_completion_bank(
            views,
            local_mask[complete].to(self.device),
            self.dataset.indices[complete].to(self.device),
            seed=self._completion_seed(round_number),
        )

    def _completion_context(self, bank, sample_ids, round_number):
        """组装当前批次的本地锚点、样本标识与补全调度上下文。"""
        return {
            "bank": bank,
            "sample_ids": sample_ids.to(self.device),
            "round_number": int(round_number),
            "center_init_round": int(self.config["training"]["center_init_round"]),
        }

    @torch.no_grad()
    def _load_private_completion(self, model):
        """在最新全局模型副本中恢复本客户端的私有补全参数并返回其键。"""
        keys = tuple(getattr(model, "local_completion_state_keys", lambda: ())())
        if not keys:
            return keys
        if self.private_completion_state is None:
            self._save_private_completion(model)
        if set(self.private_completion_state) != set(keys):
            raise ValueError("Client private completion state has incompatible parameter keys")
        model.load_state_dict(self.private_completion_state, strict=False)
        return keys

    @torch.no_grad()
    def _save_private_completion(self, model):
        """将 local 补全参数保存为脱离计算图的 CPU 副本，不保存优化器状态。"""
        keys = tuple(getattr(model, "local_completion_state_keys", lambda: ())())
        if keys:
            state = model.state_dict()
            self.private_completion_state = {
                key: state[key].detach().cpu().clone() for key in keys
            }

    @torch.no_grad()
    def predict(self, global_model, round_number):
        """使用本地掩码与补全状态预测，供单进程仿真的服务端统一评估。

        Args:
            global_model: 待评估的全局模型；仅在深拷贝上恢复私有补全参数。
            round_number: 从 1 开始的通信轮号，保持补全调度与模型状态一致。

        Returns:
            包含 assignments、labels、assignment_beta 的 CPU 张量字典，
            按本地样本顺序排列。标签仅作为评估输出，不送入模型或损失。
            本接口不属于训练上传协议，锚点和私有模型仍由客户端管理。
        """
        model = copy.deepcopy(global_model).to(self.device)
        self._load_private_completion(model)
        model.eval()
        bank = self._build_completion_bank(model, round_number)
        training = self.config["training"]
        loader = DataLoader(
            self.dataset,
            batch_size=int(training.get("eval_batch_size", training["batch_size"])),
            shuffle=False,
            num_workers=int(training.get("num_workers", 0)),
            pin_memory=self.device.type == "cuda",
        )
        probabilities, labels, assignment_betas = [], [], []
        for batch in loader:
            views, batch_labels, indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            context = None
            if bank is not None:
                context = self._completion_context(bank, indices, round_number)
                context["compute_supervision"] = False
            outputs = model(views, mask=mask, completion_context=context)
            probabilities.append(outputs["assignments"].cpu())
            labels.append(batch_labels.cpu())
            assignment_betas.append(outputs["assignment_beta"].cpu())
        return {
            "assignments": torch.cat(probabilities),
            "labels": torch.cat(labels),
            "assignment_beta": torch.cat(assignment_betas),
        }

    def train(
        self,
        global_model,
        clustering_enabled,
        round_index,
        phase,
        clustering_weight_scale=1.0,
    ):
        """执行一轮客户端更新，返回共享参数或压缩包及聚合摘要。

        Args:
            global_model: 本轮开始时的全局模型，训练在其深拷贝上进行。
            clustering_enabled: 原型是否已初始化，控制 DEC 目标和软计数。
            round_index: 从 0 开始的通信轮索引。
            phase: pretraining 或 clustering，决定学习率与本地 epoch 数。
            clustering_weight_scale: 本轮 DEC 与均衡项的渐增权重。

        Returns:
            客户端标识、样本数、平均训练指标和共享模型上传内容。
            local 补全参数及 EF 残差留在客户端，原始样本与分配矩阵不上传。
        """
        if phase not in {"pretraining", "clustering"}:
            raise ValueError(f"Unknown training phase: {phase}")
        model = copy.deepcopy(global_model).to(self.device)
        private_keys = set(self._load_private_completion(model))
        # DEC 目标在本轮训练前固定，后续所有本地 epoch 和批次复用。
        target_cache = self.local_target_cache(model) if clustering_enabled else None
        model.train()
        training = self.config["training"]
        learning_rate = float(training["learning_rate"])
        local_epochs = int(training["pretraining_local_epochs"])
        if phase == "clustering":
            learning_rate = float(training["clustering_learning_rate"])
            local_epochs = int(training["clustering_local_epochs"])
        elif clustering_enabled:
            # 中心初始化与学习率切换均属于同一预训练阶段的内部事件，
            # 不单独增加训练阶段。
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
            bank = self._build_completion_bank(model, round_index + 1)
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
                if bank is None:
                    outputs = model(views, mask=mask)
                else:
                    outputs = model(
                        views, mask=mask,
                        completion_context=self._completion_context(bank, indices, round_index + 1),
                    )
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
                for key, value in metrics.items():
                    totals[key] += value
                steps += 1
        self._save_private_completion(model)
        result = {
            "client_id": self.id,
            "num_samples": self.num_samples,
            "metrics": {key: value / max(steps, 1) for key, value in totals.items()},
        }
        if clustering_enabled:
            result["cluster_counts"] = self.soft_cluster_counts(model)
        result.update(self._build_upload(
            model, global_model, private_keys, target_cache,
            clustering_enabled, round_index, clustering_weight_scale,
        ))
        return result

    def _compression_calibration(self, model, target_cache, round_index):
        """准备本地校准批次与上下文工厂，保持采样及 EF 参考教师的一致性。

        Top-k 不读取校准数据。激活感知和 Stage 仅使用本地观测行及本轮
        固定 DEC 目标；补全锚点和教师由包含 EF 残差的有效稠密参考生成。

        Args:
            model: 本地训练后的模型，用于判断补全模块是否启用。
            target_cache: 以全局样本 ID 索引的轮初 DEC 目标，未初始化时为 None。
            round_index: 从 0 开始的通信轮索引，用于固定校准采样。

        Returns:
            压缩器的 views、targets、mask 参数及可选 completion_context_builder。
            这些对象只用于本地校准，不进入上传包。
        """
        compression = self.config["compression"]
        calibration = {"views": None, "targets": None, "mask": None}
        if compression["method"] not in {"paper", "stage"}:
            return calibration
        training = self.config["training"]
        rng = np.random.default_rng(
            int(training["seed"]) + self.id * 1009 + round_index * 7919 + 17
        )
        positions = torch.from_numpy(rng.choice(
            self.num_samples,
            size=min(int(compression["calibration_size"]), self.num_samples),
            replace=False,
        ))
        views = tuple(view[positions].to(self.device) for view in self.dataset.views)
        calibration_mask = (
            self.dataset.mask[positions].to(self.device)
            if self.dataset.mask is not None else None
        )
        targets = (
            target_cache[self.dataset.indices[positions]].to(self.device)
            if target_cache is not None else None
        )
        calibration.update(views=views, targets=targets, mask=calibration_mask)
        if getattr(model, "completion_enabled", False):
            calibration_ids = self.dataset.indices[positions].to(self.device)

            @torch.no_grad()
            def completion_context_builder(full_model):
                """从含 EF 有效更新的稠密参考构造所有压缩候选共用的锚点和教师。"""
                bank = self._build_completion_bank(full_model, round_index + 1)
                context = self._completion_context(bank, calibration_ids, round_index + 1)
                was_training = full_model.training
                try:
                    full_model.eval()
                    teacher, _fused = full_model.encode(views, mask=calibration_mask)
                    context["teacher_embeddings"] = tuple(
                        torch.nn.functional.normalize(value, dim=1).detach()
                        for value in teacher
                    )
                finally:
                    full_model.train(was_training)
                return context

            calibration["completion_context_builder"] = completion_context_builder
        return calibration

    def _build_upload(self, model, global_model, private_keys, target_cache,
                      clustering_enabled, round_index, clustering_weight_scale):
        """包装共享状态或压缩载荷，将私有参数和误差反馈残差留在客户端。

        Args:
            model: 本地训练后的模型。
            global_model: 用于计算共享参数增量的轮初全局模型。
            private_keys: 不参与通信的客户端私有参数键。
            target_cache: 本轮固定的 DEC 目标缓存，或 None。
            clustering_enabled: 原型是否已初始化，决定是否独立传输中心。
            round_index: 从 0 开始的通信轮索引。
            clustering_weight_scale: 当前训练阶段的聚类与校准评分系数。

        Returns:
            稠密 state_dict，或稀疏 payload、独立 centers 与 compression 统计。
        """
        compression = self.config.get("compression", {"method": "none"})
        if compression["method"] == "none":
            state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
                if key not in private_keys
            }
            return {"state_dict": state}
        calibration = self._compression_calibration(model, target_cache, round_index)
        packet = compress_client_update(
            model,
            global_model,
            {**compression, "loss_weights": self.config["loss_weights"]},
            centers_enabled=clustering_enabled,
            clustering_scale=clustering_weight_scale,
            residual=self.compression_residual if compression["error_feedback"] else None,
            **calibration,
        )
        if compression["error_feedback"]:
            self.compression_residual = packet.pop("residual")
        else:
            packet.pop("residual")
        return {
            "payload": packet.pop("payload"),
            "centers": packet.pop("centers"),
            "compression": packet,
        }

    @torch.no_grad()
    def local_target_cache(self, model):
        """在完整本地样本上构造本轮固定的 DEC 目标缓存。

        缓存按全数据样本标识索引，仅本客户端所属行已赋值；本轮所有批次
        与本地 epoch 复用同一目标，不读取真实标签。
        """
        model.eval()
        assignments = []
        indices = []
        for batch in self._loader(False):
            views, _labels, batch_indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            outputs = model(views, mask=mask)
            # DEC 教师始终来自真实观测，不能将本轮补全预测再次锐化为监督。
            assignments.append(outputs["observed_assignments"])
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
        """累加各视图真实观测行的软分配，返回 [V, K] 计数而非单样本分配。"""
        model.eval()
        counts = None
        for batch in self._loader(False):
            views, _labels, _indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is not None:
                mask = mask.to(self.device, non_blocking=True)
            outputs = model(views, mask=mask)
            # 缺失行的 view_assignments 为零，不为该视图的原型计数增加证据。
            batch_counts = torch.stack([
                assignment.sum(dim=0)
                for assignment in outputs["view_assignments"]
            ])
            counts = batch_counts if counts is None else counts + batch_counts
        return counts.detach().cpu()

    @torch.no_grad()
    def cluster_summary(self, model):
        """在模型副本中恢复私有补全状态并生成中心摘要，原始编码保留本地。"""
        model = copy.deepcopy(model).to(self.device)
        self._load_private_completion(model)
        model.eval()
        return self._per_view_cluster_summary(model)

    @torch.no_grad()
    def _per_view_cluster_summary(self, model):
        """由本地共同伪标签构造语义槽位一致的逐视图聚类摘要。

        完成融合可用于共同伪标签，但每视图中心及计数只取真实观测编码。
        仅返回 [V, K, d] 中心、[V, K] 硬计数与 [V] 覆盖量；伪标签不上传。
        """
        fused_batches = []
        per_view_batches = [[] for _ in model.view_models]
        mask_batches = []
        use_completion = (
            getattr(model, "completion_enabled", False)
            and model.completion_config.get("use_completed_initialization", True)
        )
        round_number = (
            max(1, int(self.config["training"]["center_init_round"])) if use_completion else 1
        )
        bank = self._build_completion_bank(model, round_number) if use_completion else None
        for batch in self._loader(False):
            views, _labels, indices, mask = self._unpack(batch)
            views = tuple(view.to(self.device, non_blocking=True) for view in views)
            if mask is None:
                mask = torch.ones(
                    (views[0].shape[0], len(views)),
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                mask = mask.to(self.device, non_blocking=True)
            if use_completion:
                completion_context = self._completion_context(bank, indices, round_number)
                completion_context["compute_supervision"] = False
                outputs = model(
                    views, mask=mask,
                    completion_context=completion_context,
                )
                view_embeddings = outputs["view_embeddings"]
                fused = outputs["completed_embedding"]
            else:
                view_embeddings, fused = model.encode(views, mask=mask)
            fused_batches.append(fused.cpu().numpy())
            mask_batches.append(mask.cpu().numpy().astype(bool, copy=False))
            for view_index, embedding in enumerate(view_embeddings):
                per_view_batches[view_index].append(embedding.cpu().numpy())

        fused = np.concatenate(fused_batches, axis=0)
        masks = np.concatenate(mask_batches, axis=0)
        view_embeddings = [np.concatenate(items, axis=0) for items in per_view_batches]
        num_clusters = int(self.config["dataset"]["num_clusters"])
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
            # 无观测支持的本地语义槽位保留原型作为回退。
            # 其计数为零，因此不会影响服务端的加权初始化。
            view_centers = head.centers.detach().cpu().numpy().astype(
                np.float64, copy=True
            )
            for cluster in range(num_clusters):
                members = observed & nonzero & (shared_labels == cluster)
                if members.any():
                    mean = normalized[members].mean(axis=0, keepdims=True)
                    view_centers[cluster] = mean[0]
            centers.append(view_centers.astype(np.float32))
            counts.append(view_counts)

        return {
            "mode": "per_view",
            "centers": np.stack(centers),
            "counts": np.stack(counts),
            "coverage": masks.sum(axis=0).astype(np.float64),
        }
