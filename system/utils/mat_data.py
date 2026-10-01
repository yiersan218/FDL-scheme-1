from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


@dataclass
class MultiViewData:
    name: str
    views: list
    labels: np.ndarray

    @property
    def num_samples(self):
        return len(self.labels)

    @property
    def view_dims(self):
        return [view.shape[1] for view in self.views]

    @property
    def num_clusters(self):
        return len(np.unique(self.labels))


class MultiViewSubset(Dataset):
    def __init__(self, data, indices, mask=None, normalization=None):
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
                # Never include a hidden value in preprocessing or the tensor seen by the model.
                local = np.zeros((len(indices), source.shape[1]), dtype=np.float32)
                local[observed] = _normalize_view(source[indices[observed]], normalization)
                self.views.append(torch.from_numpy(local))
            self.mask = torch.from_numpy(mask.copy())
        self.labels = torch.from_numpy(data.labels[indices]).long()
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        base = (tuple(view[index] for view in self.views), self.labels[index], self.indices[index])
        return base if self.mask is None else (*base, self.mask[index])


def fixed_missing_mask(num_samples, num_views, rate, seed, client_id, dataset_name):
    """Mask one random view on a fixed fraction of local samples, without labels."""
    if num_views < 2 or not 0 <= rate < 1:
        raise ValueError("Missing-view experiments need >=2 views and rate in [0, 1)")
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
    """Create balanced random partitions without consulting evaluation labels."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(int(num_samples)).astype(np.int64, copy=False)
    return [np.ascontiguousarray(part) for part in np.array_split(indices, num_clients)]
