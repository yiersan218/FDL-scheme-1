import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from flcore.trainmodel.multiview import MultiViewClusteringModel, clustering_objective  # noqa: E402
from config import apply_overrides, load_config  # noqa: E402
from utils.mat_data import MultiViewData, MultiViewSubset, fixed_missing_mask  # noqa: E402


class MissingViewTests(unittest.TestCase):
    def test_only_attention_with_stage_error_feedback_is_accepted(self):
        base = load_config("Scene-15")
        selected = apply_overrides(base, [
            "missing.enabled=true", 'compression.method="stage"',
            "compression.error_feedback=true",
        ])
        self.assertEqual(selected["missing"]["method"], "attention")
        for override in (
            'missing.method="gated"', 'missing.method="masked_fusion"',
            'compression.method="none"', "compression.error_feedback=false",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                apply_overrides(selected, [override])

    def test_mask_is_fixed_and_retains_an_observed_view(self):
        first = fixed_missing_mask(101, 3, 0.7, 42, 1, "toy")
        second = fixed_missing_mask(101, 3, 0.7, 42, 1, "toy")
        self.assertTrue(np.array_equal(first, second))
        self.assertTrue(first.any(axis=1).all())
        self.assertEqual((~first).sum(), round(101 * 0.7))
        self.assertTrue(np.all((~first).sum(axis=1) <= 1))

    def test_hidden_values_do_not_enter_normalization_or_dataset(self):
        mask = np.array([[True, True], [False, True], [True, False], [True, True]])
        views = [
            np.arange(16, dtype=np.float32).reshape(4, 4),
            np.arange(12, dtype=np.float32).reshape(4, 3),
        ]
        changed = [view.copy() for view in views]
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

    def test_completion_ignores_masked_values_and_preserves_complete_path(self):
        torch.manual_seed(7)
        model = MultiViewClusteringModel(
            [5, 4], 3, [8], 4,
            completion={"method": "attention", "heads": 2},
        ).eval()
        views = (torch.randn(8, 5), torch.randn(8, 4))
        full_mask = torch.ones((8, 2), dtype=torch.bool)
        full = model(views)
        complete = model(views, mask=full_mask)
        self.assertTrue(torch.allclose(full["embedding"], complete["embedding"], atol=1e-6))

        mask = full_mask.clone()
        mask[0, 1] = False
        mask[1, 0] = False
        anchors = model.observed_fusion(
            tuple(view[2:4] for view in views), full_mask[2:4]
        )[1].detach()
        observed = model.observed_fusion(views, mask)[1]
        fallback = model.encode(views, mask=mask, anchors=None)[1]
        self.assertTrue(torch.allclose(observed, fallback, atol=1e-6))
        first = model(views, mask=mask, anchors=anchors)
        self.assertFalse(torch.allclose(first["embedding"][:2], observed[:2], atol=1e-6))
        changed = (views[0].clone(), views[1].clone())
        changed[0][1] = 10000
        changed[1][0] = -10000
        second = model(changed, mask=mask, anchors=anchors)
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
