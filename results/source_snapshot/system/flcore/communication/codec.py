"""共享更新的稀疏协议、模型状态重建与通信字节计量。"""

import struct

import numpy as np
import torch


_HEADER = struct.Struct("<4sII")
_MAGIC = b"FMC1"


def center_keys(model):
    """返回独立于普通稀疏更新传输的 A2 各视图原型键。"""
    keys = tuple(model.prototype_center_keys())
    if not keys:
        raise ValueError("A2 requires at least one per-view prototype tensor")
    return keys


def ordinary_keys(model):
    """按模型状态顺序返回共享普通参数键，排除原型与私有补全参数。"""
    excluded = set(center_keys(model)) | set(local_completion_keys(model))
    return [key for key in model.state_dict() if key not in excluded]


def local_completion_keys(model):
    """获取客户端私有补全参数键；未提供相应接口时返回空元组。"""
    getter = getattr(model, "local_completion_state_keys", None)
    return tuple(getter()) if getter is not None else ()


def load_communication_state(model, state):
    """恢复共享通信状态并保留客户端私有补全参数，拒绝遗漏共享参数。"""
    private = set(local_completion_keys(model))
    if not private:
        model.load_state_dict(state)
        return
    transmitted = {key: value for key, value in state.items() if key not in private}
    loaded = model.load_state_dict(transmitted, strict=False)
    if set(loaded.missing_keys) != private or loaded.unexpected_keys:
        raise RuntimeError("Communication state is missing shared model parameters")


def flatten_update(local_model, global_model, keys):
    """按指定键展平本地模型相对全局模型的增量，返回 CPU float32 向量。"""
    local = local_model.state_dict()
    global_state = global_model.state_dict()
    return torch.cat([
        (local[key].detach().cpu() - global_state[key].detach().cpu()).reshape(-1)
        for key in keys
    ]).to(dtype=torch.float32)


def dense_uplink_bytes(model, centers_enabled):
    """计算稠密共享模型的上行基准字节数，中心启用后另计软簇计数。"""
    state = model.state_dict()
    private = set(local_completion_keys(model))
    total = sum(value.numel() * value.element_size() for key, value in state.items() if key not in private)
    if centers_enabled:
        total += sum(state[key].shape[0] * 4 for key in center_keys(model))
    return total


def encode_sparse(flat_update, indices):
    """将选中坐标编码为协议头、uint32 索引和 float32 数值。

    索引在 CPU 上排序并校验范围及唯一性；返回真正紧凑的 bytes，
    每个保留坐标占 8 字节，另加固定协议头。
    """
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
    """校验协议头与有序索引，将稀疏载荷恢复为未选坐标为零的增量向量。"""
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
    """解码普通共享更新并结合独立上传的原型，重建 CPU 模型状态。"""
    state = global_model.state_dict()
    keys = ordinary_keys(global_model)
    expected = sum(state[key].numel() for key in keys)
    flat = decode_sparse(payload, expected)
    return state_from_flat_update(global_model, flat, centers)


def state_from_flat_update(global_model, flat, centers=None):
    """从展平增量重建共享状态，无需先生成临时传输包。

    普通参数相对全局状态相加；原型使用独立提供的中心或全局参考。
    客户端私有补全参数不在返回状态中，由调用方保留。
    """
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
    keys_with_centers = center_keys(global_model)
    if centers is None:
        supplied = {}
    elif isinstance(centers, dict):
        supplied = centers
    elif len(keys_with_centers) == 1:
        supplied = {keys_with_centers[0]: centers}
    else:
        raise ValueError("Per-view compressed updates require keyed prototype centers")
    for key in keys_with_centers:
        value = supplied.get(key, state[key])
        result[key] = value.detach().cpu().clone()
    return result
