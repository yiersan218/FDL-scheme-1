"""验证固定缺失掩码、观测值归一化以及默认 A2 的隐藏值隔离。"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from flcore.trainmodel.multiview import MultiViewClusteringModel  # noqa: E402
from flcore.trainmodel.losses import clustering_objective  # noqa: E402
from config import apply_overrides, load_config  # noqa: E402
from utils.mat_data import MultiViewData, MultiViewSubset, fixed_missing_mask  # noqa: E402


class MissingViewTests(unittest.TestCase):
    """覆盖缺失实验配置约束、掩码重现性和真实观测计算边界。"""

    def test_missing_runs_allow_independent_compression_choices(self):
        """验证缺失率与通信方法解耦，稠密、Top-k 和 Stage 均可显式选择。"""
        base = load_config("Scene-15")
        for method in ("none", "topk", "stage"):
            for feedback in (False, True):
                with self.subTest(method=method, feedback=feedback):
                    selected = apply_overrides(base, [
                        "missing.rate=0.3", f"compression.method={method}",
                        f"compression.error_feedback={str(feedback).lower()}",
                    ])
                    self.assertTrue(selected["missing"]["enabled"])
                    self.assertEqual(selected["compression"]["method"], method)
                    self.assertEqual(selected["completion"]["alignment_mode"], "mse")

    def test_mask_is_fixed_and_retains_an_observed_view(self):
        """验证同条件掩码可复现，选中样本各缺一个视图且仍保留观测。"""
        first = fixed_missing_mask(101, 3, 0.7, 42, 1, "toy")
        second = fixed_missing_mask(101, 3, 0.7, 42, 1, "toy")
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(first.any(axis=1).all())
        self.assertEqual((~first).sum(), round(101 * 0.7))
        self.assertTrue(np.all((~first).sum(axis=1) <= 1))

    def test_hidden_values_do_not_enter_normalization_or_dataset(self):
        """验证隐藏值的极端扰动不会改变观测归一化，隐藏张量始终置零。"""
        mask = np.array([[True, True], [False, True], [True, False], [True, True]])
        views = [
            np.arange(16, dtype=np.float32).reshape(4, 4),
            np.arange(12, dtype=np.float32).reshape(4, 3),
        ]
        changed = [view.copy() for view in views]
        # 只扰动隐藏行；若它们泄漏到归一化统计，观测张量也会随之变化。
        changed[0][1] = 1e9
        changed[1][2] = -1e9
        labels = np.array([0, 1, 0, 1])
        first = MultiViewSubset(MultiViewData("toy", views, labels), np.arange(4),
                                mask=mask, normalization="standard")
        second = MultiViewSubset(MultiViewData("toy", changed, labels), np.arange(4),
                                 mask=mask, normalization="standard")
        for left, right in zip(first.views, second.views):
            self.assertTrue(torch.equal(left, right))
        self.assertTrue(torch.equal(first.views[0][1], torch.zeros(4)))
        self.assertTrue(torch.equal(first.views[1][2], torch.zeros(3)))

    def test_a2_uses_only_observed_values(self):
        """验证全真掩码等价于完整视图路径，缺失前向不读取隐藏数据。"""
        torch.manual_seed(7)
        model = MultiViewClusteringModel(
            [5, 4], 3, [8], 4,
            completion={"enabled": False},
        ).eval()
        views = (torch.randn(8, 5), torch.randn(8, 4))
        full_mask = torch.ones((8, 2), dtype=torch.bool)
        full = model(views)
        complete = model(views, mask=full_mask)
        self.assertTrue(torch.allclose(full["embedding"], complete["embedding"], atol=1e-6))

        mask = full_mask.clone()
        mask[0, 1] = False
        mask[1, 0] = False
        observed = model.observed_fusion(views, mask)[1]
        fallback = model.encode(views, mask=mask)[1]
        self.assertTrue(torch.allclose(observed, fallback, atol=1e-6))
        first = model(views, mask=mask)
        self.assertTrue(torch.allclose(first["embedding"], observed, atol=1e-6))
        changed = (views[0].clone(), views[1].clone())
        changed[0][1] = 10000
        changed[1][0] = -10000
        second = model(changed, mask=mask)
        self.assertTrue(torch.allclose(first["embedding"], second["embedding"], atol=1e-6))
        loss, _metrics = clustering_objective(
            first, views,
            {"reconstruction": 1.0, "consistency": 0.2,
             "clustering": 0.5, "balance": 0.05},
            clustering_enabled=True, mask=mask,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
