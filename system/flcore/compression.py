"""Compact uplink updates and label-free calibration-based mask selection."""

import copy
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flcore.trainmodel.multiview import clustering_objective


_HEADER = struct.Struct("<4sII")
_MAGIC = b"FMC1"
_CENTER_KEY = "cluster_head.centers"


def ordinary_keys(model):
    return [key for key in model.state_dict() if key != _CENTER_KEY]


def flatten_update(local_model, global_model, keys):
    local = local_model.state_dict()
    global_state = global_model.state_dict()
    return torch.cat([
        (local[key].detach().cpu() - global_state[key].detach().cpu()).reshape(-1)
        for key in keys
    ]).to(dtype=torch.float32)


def dense_uplink_bytes(model, centers_enabled):
    state = model.state_dict()
    total = sum(value.numel() * value.element_size() for value in state.values())
    if centers_enabled:
        total += state[_CENTER_KEY].shape[0] * 4  # soft cluster counts
    return total


def encode_sparse(flat_update, indices):
    if flat_update.numel() >= 2**32:
        raise ValueError("Sparse protocol supports fewer than 2^32 coordinates")
    indices = indices.to(dtype=torch.long, device="cpu").sort().values
    if indices.numel() and (indices[0] < 0 or indices[-1] >= flat_update.numel()):
        raise ValueError("Sparse update contains an invalid index")
    if indices.numel() > 1 and torch.any(indices[1:] == indices[:-1]):
        raise ValueError("Sparse update contains duplicate indices")
    positions = indices.numpy().astype("<u4", copy=False)
    values = flat_update[indices].numpy().astype("<f4", copy=False)
    return _HEADER.pack(_MAGIC, flat_update.numel(), indices.numel()) + positions.tobytes() + values.tobytes()


def decode_sparse(payload, expected_numel):
    if len(payload) < _HEADER.size:
        raise ValueError("Truncated compressed update")
    magic, numel, count = _HEADER.unpack_from(payload)
    if magic != _MAGIC or numel != expected_numel or len(payload) != _HEADER.size + count * 8:
        raise ValueError("Invalid compressed update header or length")
    positions = np.frombuffer(payload, dtype="<u4", count=count, offset=_HEADER.size)
    values = np.frombuffer(payload, dtype="<f4", count=count, offset=_HEADER.size + count * 4)
    if count and (positions[-1] >= numel or np.any(positions[1:] <= positions[:-1])):
        raise ValueError("Invalid compressed update positions")
    result = torch.zeros(numel, dtype=torch.float32)
    if count:
        result[torch.from_numpy(positions.astype(np.int64))] = torch.from_numpy(values.copy())
    return result


def reconstruct_state(global_model, payload, centers=None):
    state = global_model.state_dict()
    keys = ordinary_keys(global_model)
    expected = sum(state[key].numel() for key in keys)
    flat = decode_sparse(payload, expected)
    return state_from_flat_update(global_model, flat, centers)


def state_from_flat_update(global_model, flat, centers=None):
    """Build model state from a dense flat update without a temporary wire packet."""
    state = global_model.state_dict()
    keys = ordinary_keys(global_model)
    expected = sum(state[key].numel() for key in keys)
    if flat.numel() != expected:
        raise ValueError("Dense update length does not match model parameters")
    result = {}
    offset = 0
    for key in keys:
        reference = state[key].detach().cpu()
        count = reference.numel()
        result[key] = reference + flat[offset:offset + count].reshape(reference.shape).to(reference.dtype)
        offset += count
    result[_CENTER_KEY] = (
        state[_CENTER_KEY].detach().cpu().clone()
        if centers is None else centers.detach().cpu().clone()
    )
    return result


def _top_indices(scores, count):
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if count >= scores.numel():
        return torch.arange(scores.numel())
    return torch.topk(scores, count, sorted=False).indices


def activation_scores(model, views, flat_update, keys):
    """ICLR-2026 linear-layer score; magnitude fallback for non-linear parameters."""
    energies = {}
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def capture(_module, inputs, layer_name=name):
                value = inputs[0].detach().float()
                energies[layer_name] = value.reshape(-1, value.shape[-1]).pow(2).sum(0).cpu()
            hooks.append(module.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            model.eval()
            model(views)
    finally:
        for hook in hooks:
            hook.remove()
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
    normalized = F.normalize(embedding, dim=1)
    return normalized @ normalized.T


def _normalized_distance(first, second):
    return ((first - second).square().sum() / first.square().sum().clamp_min(1e-8)).item()


def _calibration_outputs(model, views, targets, loss_weights, centers_enabled, scale):
    model.eval()
    with torch.no_grad():
        outputs = model(views)
        loss, _ = clustering_objective(
            outputs, views, loss_weights, clustering_enabled=centers_enabled,
            clustering_weight_scale=scale, target_assignments=targets,
        )
        relations = [_relation(value) for value in outputs["view_embeddings"]]
        relations.append(_relation(outputs["embedding"]))
        pair = outputs["assignments"] @ outputs["assignments"].T if centers_enabled else None
    return loss.detach(), relations, pair


def semantic_score(reference, candidate, view_weight, pair_weight, scale):
    ref_loss, ref_relations, ref_pair = reference
    loss, relations, pair = candidate
    task = max(0.0, (loss - ref_loss).item()) / (abs(ref_loss.item()) + 1e-8)
    view = sum(_normalized_distance(a, b) for a, b in zip(ref_relations, relations)) / len(relations)
    cluster = _normalized_distance(ref_pair, pair) if ref_pair is not None else 0.0
    return task + view_weight * view + scale * pair_weight * cluster


def compress_client_update(local_model, global_model, config, views=None, targets=None,
                           centers_enabled=False, clustering_scale=0.0, residual=None):
    """Return actual compact model-update bytes and local-only diagnostics."""
    started = time.perf_counter()
    keys = ordinary_keys(global_model)
    delta = flatten_update(local_model, global_model, keys)
    if residual is not None:
        if residual.shape != delta.shape:
            raise ValueError("Error-feedback residual shape changed")
        delta = delta + residual
    centers = local_model.cluster_head.centers.detach().cpu().clone() if centers_enabled else None
    fixed_bytes = _HEADER.size + (centers.numel() * centers.element_size() if centers is not None else 0)
    if centers_enabled:
        fixed_bytes += centers.shape[0] * 4
    dense_bytes = dense_uplink_bytes(global_model, centers_enabled)
    budget = int(dense_bytes * float(config["budget_ratio"]))
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
        paper_scores = activation_scores(local_model, views, delta, keys)
        if method == "paper":
            selected = _top_indices(paper_scores, count)
            score = None
        elif method == "stage":
            full_model = copy.deepcopy(local_model)
            if residual is not None:
                full_model.load_state_dict(state_from_flat_update(global_model, delta, centers))
            reference = _calibration_outputs(
                full_model, views, targets, config["loss_weights"], centers_enabled, clustering_scale,
            )
            probe = copy.deepcopy(local_model)
            score = float("inf")
            selected = None
            for mask in candidate_masks(local_model, delta, paper_scores, count, config["candidates"]):
                packet = encode_sparse(delta, mask)
                probe.load_state_dict(reconstruct_state(global_model, packet, centers))
                candidate = _calibration_outputs(
                    probe, views, targets, config["loss_weights"], centers_enabled, clustering_scale,
                )
                value = semantic_score(
                    reference, candidate, config["view_weight"], config["pair_weight"], clustering_scale,
                )
                if np.isfinite(value) and value < score:
                    score, selected = value, mask
            if selected is None:
                selected = _top_indices(paper_scores, count)
                score = None
        else:
            raise ValueError(f"Unknown compression method: {method}")
    payload = encode_sparse(delta, selected)
    new_residual = delta.clone()
    new_residual[selected] = 0
    return {
        "payload": payload,
        "centers": centers,
        "bytes": len(payload) + (centers.numel() * centers.element_size() if centers is not None else 0)
                 + (centers.shape[0] * 4 if centers is not None else 0),
        "dense_bytes": dense_bytes,
        "kept": int(selected.numel()),
        "parameters": int(delta.numel()),
        "score": score,
        "compression_seconds": time.perf_counter() - started,
        "residual": new_residual,
    }
