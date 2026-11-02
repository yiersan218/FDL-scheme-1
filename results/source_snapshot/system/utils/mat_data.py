"""加载多视图 MAT 数据，构造无标签客户端划分与固定观测掩码。"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass
class MultiViewData:
    """保存视图矩阵及评估标签；每个视图形状为 (样本数, 特征维度)。"""

    name: str
    views: list
    labels: np.ndarray

    @property
    def num_samples(self):
        """返回各视图共有的样本数。"""
        return len(self.labels)

    @property
    def view_dims(self):
        """按视图顺序返回各自的输入特征维度。"""
        return [view.shape[1] for view in self.views]

    @property
    def num_clusters(self):
        """从标签元数据取得数据集的簇数，不提供样本级训练目标。"""
        return len(np.unique(self.labels))


class MultiViewSubset(Dataset):
    """供单个客户端使用的数据子集，可按真实观测掩码隔离隐藏值。"""

    def __init__(self, data, indices, mask=None, normalization=None):
        """建立本地张量；有掩码时只用本地可见行计算归一化统计。

        Args:
            data: 包含各视图原始数组和评估标签的 MultiViewData。
            indices: 本客户端样本在完整数据集中的索引，顺序决定本地行号。
            mask: 可选的 (本地样本数, 视图数) 布尔矩阵，True 表示真实可见。
            normalization: 掩码路径必须显式指定的归一化方式。

        Raises:
            ValueError: 掩码形状不符、样本无可见视图、某视图完全不可见，
                或掩码路径未指定归一化方式。
        """
        indices = np.asarray(indices, dtype=np.int64)
        if mask is None:
            self.views = [torch.from_numpy(view[indices]) for view in data.views]
            self.mask = None
        else:
            mask = np.asarray(mask, dtype=np.bool_)
            expected = (len(indices), len(data.views))
            if mask.shape != expected or not mask.any(axis=1).all():
                raise ValueError(f"Invalid view mask shape or empty sample: {mask.shape}, expected {expected}")
            if normalization is None:
                raise ValueError("Observed-only normalization must be explicit for masked data")
            self.views = []
            for view_index, source in enumerate(data.views):
                observed = mask[:, view_index]
                if not observed.any():
                    raise ValueError(f"View {view_index} has no observed samples on this client")
                # 先筛选可见行再归一化，防止隐藏值通过统计量泄漏到模型。
                local = np.zeros((len(indices), source.shape[1]), dtype=np.float32)
                local[observed] = _normalize_view(source[indices[observed]], normalization)
                self.views.append(torch.from_numpy(local))
            self.mask = torch.from_numpy(mask.copy())
        self.labels = torch.from_numpy(data.labels[indices]).long()
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        """返回本客户端的样本数。"""
        return len(self.labels)

    def __getitem__(self, index):
        """返回视图、评估标签和全局样本 ID；掩码路径另附该行观测状态。"""
        base = (tuple(view[index] for view in self.views), self.labels[index], self.indices[index])
        return base if self.mask is None else (*base, self.mask[index])


def fixed_missing_mask(num_samples, num_views, rate, seed, client_id, dataset_name):
    """生成固定掩码，让选中的每个样本恰好缺失一个随机视图。

    Args:
        num_samples: 当前客户端的样本数。
        num_views: 视图数，至少为 2。
        rate: 不完整样本比例，范围为 [0, 1)，不是视图单元格缺失比例。
        seed: 实验随机种子。
        client_id: 客户端标识，用于区分各客户端的随机序列。
        dataset_name: 数据集名称，与种子、客户端和比例共同确定掩码。

    Returns:
        形状为 (num_samples, num_views) 的布尔数组；True 表示真实可见。
        缺失行数为 round(rate * num_samples)，每行至少保留一个视图。
    """
    if num_views < 2 or not 0 <= rate < 1:
        raise ValueError("Missing-view experiments need >=2 views and rate in [0, 1)")
    # 使用稳定摘要派生种子，不依赖进程间可能变化的 Python 内置 hash。
    digest = hashlib.sha256(
        f"{dataset_name}|{int(seed)}|{int(client_id)}|{float(rate):.8f}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    mask = np.ones((num_samples, num_views), dtype=np.bool_)
    missing_count = round(float(rate) * num_samples)
    positions = rng.choice(num_samples, size=missing_count, replace=False)
    missing_views = rng.integers(0, num_views, size=missing_count)
    mask[positions, missing_views] = False
    return mask


def load_multiview_mat(path, name=None, normalization="standard"):
    """读取 MAT 中的视图和标签，并统一样本轴、数据类型及标签编号。

    Args:
        path: 包含 X/Y、data/labels 或 X/gt 字段的 MAT 文件路径。
        name: 数据集名称；未提供时使用文件名去掉扩展名后的部分。
        normalization: 全数据归一化方式。缺失实验应传入 none，
            由 MultiViewSubset 在客户端按可见行完成归一化。

    Returns:
        MultiViewData，其中视图为连续 float32 数组，标签为连续整数编号。

    Raises:
        KeyError: 缺少支持的视图或标签字段。
        ValueError: 视图不是二维矩阵、样本数不匹配或含非有限数值。
    """
    path = Path(path)
    payload = loadmat(path)
    view_key = "X" if "X" in payload else "data" if "data" in payload else None
    label_key = next(
        (key for key in ("Y", "labels", "gt") if key in payload),
        None,
    )
    if view_key is None or label_key is None:
        raise KeyError(f"{path} must contain X/Y, data/labels, or X/gt")

    raw_views = np.asarray(payload[view_key], dtype=object).reshape(-1)
    labels = np.asarray(payload[label_key]).reshape(-1)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)

    views = []
    for raw_view in raw_views:
        view = np.asarray(raw_view)
        if view.ndim != 2:
            raise ValueError(f"Every view must be 2-D, got {view.shape} in {path}")
        # 兼容以特征为行的 MAT 文件；方阵保留原方向，避免任意转置。
        if view.shape[0] != len(labels) and view.shape[1] == len(labels):
            view = view.T
        if view.shape[0] != len(labels):
            raise ValueError(f"View sample count {view.shape[0]} != label count {len(labels)}")
        view = view.astype(np.float32, copy=False)
        if not np.isfinite(view).all():
            raise ValueError(f"View in {path} contains NaN or infinity")
        views.append(_normalize_view(view, normalization))

    return MultiViewData(name=name or path.stem, views=views, labels=labels)


def _normalize_view(view, method):
    """按列标准化或缩放，或按行做 L2 归一化，返回连续 float32 数组。

    极小分母替换为 1，避免常量特征或零向量导致除零。调用方负责传入
    允许参与统计的行；此函数自身不读取或推断观测掩码。
    """
    if method == "standard":
        mean = view.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        std = view.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
        std[std < 1e-6] = 1.0
        return np.ascontiguousarray((view - mean) / std, dtype=np.float32)
    if method == "minmax":
        minimum = view.min(axis=0, keepdims=True)
        scale = view.max(axis=0, keepdims=True) - minimum
        scale[scale < 1e-6] = 1.0
        return np.ascontiguousarray((view - minimum) / scale, dtype=np.float32)
    if method == "l2":
        norm = np.linalg.norm(view, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1.0
        return np.ascontiguousarray(view / norm, dtype=np.float32)
    if method in (None, "none"):
        return np.ascontiguousarray(view, dtype=np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def balanced_client_indices(num_samples, num_clients, seed):
    """随机打乱样本索引并均分，返回互不重叠、大小相差至多 1 的索引列表。

    划分只依赖样本数、客户端数和种子，不读取标签，也不按类别分层。
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(int(num_samples)).astype(np.int64, copy=False)
    return [np.ascontiguousarray(part) for part in np.array_split(indices, num_clients)]
