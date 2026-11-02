"""固定字节预算下的更新评分、候选校准选择与客户端误差反馈。"""

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flcore.trainmodel.losses import clustering_objective
from .codec import (
    _HEADER,
    center_keys,
    dense_uplink_bytes,
    encode_sparse,
    flatten_update,
    load_communication_state,
    ordinary_keys,
    reconstruct_state,
    state_from_flat_update,
)


def _top_indices(scores, count):
    """返回分数最大的指定数量坐标，兼容零预算及保留全部坐标。"""
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if count >= scores.numel():
        return torch.arange(scores.numel())
    return torch.topk(scores, count, sorted=False).indices


def activation_scores(model, views, flat_update, keys, mask=None, completion_context=None):
    """用线性层输入能量修正更新平方，计算无标签激活感知分数。

    非线性层参数保留更新平方评分；偏置按校准样本数缩放。
    临时钩子在前向结束后移除，并恢复调用前的训练模式。
    """
    energies = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def capture(_module, inputs, layer_name=name):
                """累积当前线性层各输入通道的平方能量，不保留激活计算图。"""
                value = inputs[0].detach().float()
                energies[layer_name] = value.reshape(-1, value.shape[-1]).pow(2).sum(0).cpu()
            hooks.append(module.register_forward_pre_hook(capture))
    was_training = model.training
    try:
        with torch.no_grad():
            model.eval()
            if completion_context is None:
                model(views, mask=mask)
            else:
                model(views, mask=mask, completion_context=completion_context)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    scores = flat_update.square().clone()
    offset = 0
    for key in keys:
        shape = model.state_dict()[key].shape
        count = int(np.prod(shape))
        prefix, _, suffix = key.rpartition(".")
        if prefix in energies and suffix == "weight" and len(shape) == 2:
            scores[offset:offset + count] *= energies[prefix].repeat(shape[0])
        elif prefix in energies and suffix == "bias" and len(shape) == 1:
            scores[offset:offset + count] *= len(views[0])
        offset += count
    return scores


def _view_groups(model, keys):
    """按展平坐标区间划分视图专属参数与共享参数，供分组预算分配。"""
    groups = [[] for _ in model.view_models]
    shared = []
    offset = 0
    for key in keys:
        count = model.state_dict()[key].numel()
        if key.startswith("view_models."):
            view_id = int(key.split(".")[1])
            groups[view_id].append((offset, offset + count))
        else:
            shared.append((offset, offset + count))
        offset += count
    return groups, shared


def _allocated_mask(scores, count, groups, weights):
    """按组权重和容量分配坐标配额，组内选最高分并补齐剩余预算。"""
    group_indices = [torch.cat([torch.arange(start, end) for start, end in spans]) for spans in groups]
    sizes = np.asarray([len(indices) for indices in group_indices], dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not weights.any():
        weights = np.ones_like(weights)
    quotas = np.minimum(sizes, np.floor(count * weights / weights.sum()).astype(np.int64))
    remaining = count - int(quotas.sum())
    while remaining:
        available = np.flatnonzero(quotas < sizes)
        if not len(available):
            break
        shares = remaining * weights[available] / weights[available].sum()
        increments = np.minimum(sizes[available] - quotas[available], np.floor(shares).astype(np.int64))
        if not increments.any():
            increments[np.argmax(shares)] = 1
        quotas[available] += increments
        remaining -= int(increments.sum())
    selected = [_top_indices(scores[indices], int(quota)) for indices, quota in zip(group_indices, quotas)]
    return torch.cat([indices[chosen] for indices, chosen in zip(group_indices, selected)])


def candidate_masks(model, flat_update, paper_scores, count, limit):
    """构造幅值、激活分数及多种分组预算候选，按配置截取候选数量。"""
    magnitude = flat_update.abs()
    result = [_top_indices(magnitude, count), _top_indices(paper_scores, count)]
    groups, shared = _view_groups(model, ordinary_keys(model))
    shared_group = shared
    if shared_group:
        groups = groups + [shared_group]
    sizes = np.asarray([sum(end - start for start, end in group) for group in groups], dtype=float)
    view_weights = torch.softmax(model.view_logits.detach(), dim=0).cpu().numpy()
    if shared_group:
        view_weights = np.append(view_weights, max(sizes[-1] / max(sizes[:-1].sum(), 1), 1e-6))
    allocations = [np.ones(len(groups)), sizes, np.sqrt(sizes), view_weights]
    for weights in allocations:
        result.append(_allocated_mask(paper_scores, count, groups, weights))
    return result[:max(1, limit)]


def _relation(embedding):
    """计算归一化样本表示的余弦关系矩阵，结果形状为 [B, B]。"""
    normalized = F.normalize(embedding, dim=1)
    return normalized @ normalized.T


def _normalized_distance(first, second):
    """计算候选矩阵相对参考矩阵平方能量的归一化误差标量。"""
    return ((first - second).square().sum() / first.square().sum().clamp_min(1e-8)).item()


def _calibration_outputs(model, views, targets, loss_weights, centers_enabled, scale,
                         mask=None, completion_context=None):
    """在无梯度评估模式下生成任务损失、视图关系与软同簇关系。

    视图关系仅包含真实观测行；启用补全时融合关系可使用完成融合。
    软同簇关系使用正式受限补全分配，与训练目标保持一致；中心初始化前不计算。
    """
    model.eval()
    with torch.no_grad():
        outputs = (
            model(views, mask=mask) if completion_context is None
            else model(views, mask=mask, completion_context=completion_context)
        )
        loss, _ = clustering_objective(
            outputs, views, loss_weights, clustering_enabled=centers_enabled,
            clustering_weight_scale=scale, target_assignments=targets, mask=mask,
        )
        relations = [
            _relation(value if mask is None else value[mask[:, index]])
            if mask is None or mask[:, index].any() else None
            for index, value in enumerate(outputs["view_embeddings"])
        ]
        relations.append(_relation(outputs.get("completed_embedding", outputs["embedding"])))
        pair = outputs["assignments"] @ outputs["assignments"].T if centers_enabled else None
    return loss.detach(), relations, pair


def semantic_score(reference, candidate, view_weight, pair_weight, scale):
    """汇总任务损失退化、视图关系变化及阶段加权的软同簇关系变化。"""
    ref_loss, ref_relations, ref_pair = reference
    loss, relations, pair = candidate
    task = max(0.0, (loss - ref_loss).item()) / (abs(ref_loss.item()) + 1e-8)
    valid = [(a, b) for a, b in zip(ref_relations, relations) if a is not None]
    view = (
        sum(_normalized_distance(a, b) for a, b in valid) / len(valid)
        if valid else 0.0
    )
    cluster = _normalized_distance(ref_pair, pair) if ref_pair is not None else 0.0
    return task + view_weight * view + scale * pair_weight * cluster


def compress_client_update(local_model, global_model, config, views=None, targets=None,
                           centers_enabled=False, clustering_scale=0.0, residual=None,
                           mask=None, completion_context_builder=None):
    """在实际编码字节预算内压缩共享更新，并返回客户端残差。

    Args:
        local_model: 本地训练后的模型，保留本客户端私有补全状态。
        global_model: 本轮增量计算所用的全局参考模型。
        config: 压缩方法、预算、校准评分权重及损失权重。
        views: 本地校准视图；paper 和 stage 方法必须提供。
        targets: 本轮固定的校准样本 DEC 目标。
        centers_enabled: 是否独立上传已初始化的原型及计数。
        clustering_scale: 聚类项和软同簇关系的阶段系数。
        residual: 上次参与训练保留的 EF 残差，坐标顺序须与共享更新一致。
        mask: 校准样本的真实观测掩码。
        completion_context_builder: 从有效稠密参考构造只读补全上下文的回调。

    Returns:
        包含 payload、独立中心、实际字节数、评分和新残差的字典。
        调用方仅在本地保存 residual；字节统计不代表实测网络流量。
    """
    started = time.perf_counter()
    keys = ordinary_keys(global_model)
    delta = flatten_update(local_model, global_model, keys)
    if residual is not None:
        if residual.shape != delta.shape:
            raise ValueError("Error-feedback residual shape changed")
        delta = delta + residual
    keys_with_centers = center_keys(local_model)
    if centers_enabled and len(keys_with_centers) == 1:
        centers = local_model.state_dict()[keys_with_centers[0]].detach().cpu().clone()
    elif centers_enabled:
        centers = {
            key: local_model.state_dict()[key].detach().cpu().clone()
            for key in keys_with_centers
        }
    else:
        centers = None
    center_values = (
        [] if centers is None
        else list(centers.values()) if isinstance(centers, dict)
        else [centers]
    )
    center_bytes = sum(value.numel() * value.element_size() for value in center_values)
    count_bytes = sum(value.shape[0] * 4 for value in center_values)
    fixed_bytes = _HEADER.size + center_bytes
    if centers_enabled:
        fixed_bytes += count_bytes
    dense_bytes = dense_uplink_bytes(global_model, centers_enabled)
    explicit_budget = config.get("budget_bytes")
    budget_pair = config.get("budget_bytes_pair")
    if explicit_budget is not None:
        budget = int(explicit_budget)
    elif budget_pair is not None:
        budget = int(budget_pair["post_center" if centers_enabled else "pre_center"])
    else:
        budget = int(dense_bytes * float(config["budget_ratio"]))
    if budget <= 0:
        raise ValueError("Communication byte budget must be positive")
    # 每个坐标需要 4 字节索引和 4 字节数值，先扣除头、中心与计数开销。
    count = min(delta.numel(), max(0, (budget - fixed_bytes) // 8))
    if fixed_bytes > budget:
        raise ValueError("Communication budget is smaller than fixed metadata and center payload")
    method = config["method"]
    if method == "topk":
        selected = _top_indices(delta.abs(), count)
        score = None
    else:
        if views is None:
            raise ValueError("Calibration views are required for paper/stage compression")
        # 校准锚点与教师仅保留在本地，所有候选共用同一上下文。
        # 必须先恢复包含 EF 残差的有效更新，再构造该上下文。
        full_model = None
        completion_context = None
        if completion_context_builder is not None or method == "stage":
            full_model = copy.deepcopy(local_model)
            if residual is not None:
                load_communication_state(full_model, state_from_flat_update(global_model, delta, centers))
            if completion_context_builder is not None:
                completion_context = completion_context_builder(full_model)
        scoring_model = full_model if completion_context is not None else local_model
        paper_scores = activation_scores(
            scoring_model, views, delta, keys, mask=mask,
            completion_context=completion_context,
        )
        if method == "paper":
            selected = _top_indices(paper_scores, count)
            score = None
        elif method == "stage":
            reference = _calibration_outputs(
                full_model, views, targets, config["loss_weights"], centers_enabled, clustering_scale,
                mask=mask, completion_context=completion_context,
            )
            probe = copy.deepcopy(local_model)
            score = float("inf")
            selected = None
            for candidate_mask in candidate_masks(
                local_model, delta, paper_scores, count, config["candidates"]
            ):
                packet = encode_sparse(delta, candidate_mask)
                load_communication_state(probe, reconstruct_state(global_model, packet, centers))
                candidate = _calibration_outputs(
                    probe, views, targets, config["loss_weights"], centers_enabled, clustering_scale,
                    mask=mask, completion_context=completion_context,
                )
                value = semantic_score(
                    reference, candidate, config["view_weight"], config["pair_weight"], clustering_scale,
                )
                if np.isfinite(value) and value < score:
                    score, selected = value, candidate_mask
            if selected is None:
                selected = _top_indices(paper_scores, count)
                score = None
        else:
            raise ValueError(f"Unknown compression method: {method}")
    payload = encode_sparse(delta, selected)
    # 只保存未发送的有效更新坐标；该残差由客户端留存，不上传服务器。
    new_residual = delta.clone()
    new_residual[selected] = 0
    return {
        "payload": payload,
        "centers": centers,
        "bytes": len(payload) + center_bytes + count_bytes,
        "dense_bytes": dense_bytes,
        "budget_bytes": budget,
        "fixed_bytes": fixed_bytes,
        "kept": int(selected.numel()),
        "parameters": int(delta.numel()),
        "score": score,
        "compression_seconds": time.perf_counter() - started,
        "residual": new_residual,
    }
