"""验证统一补全自监督、受限概率融合和预测支路的原型梯度隔离。"""

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from flcore.trainmodel.multiview import MultiViewClusteringModel  # noqa: E402
from flcore.trainmodel.losses import (  # noqa: E402
    clustering_objective,
    target_distribution,
)


class CompletedAssignmentTests(unittest.TestCase):
    """区分观测教师、可训练补全预测和不可由预测移动的真实原型。"""

    def setUp(self):
        """构造有完整锚点和单视图缺失行的小型确定性三视图数据。"""
        torch.manual_seed(2026)
        self.views = tuple(torch.randn(8, dimension) for dimension in (5, 4, 6))
        self.mask = torch.ones(8, 3, dtype=torch.bool)
        self.mask[6, 1] = False
        self.mask[7, 2] = False
        self.ids = torch.arange(8)

    def model(self, **options):
        """使用正式默认 MSE 目标建立模型，仅覆盖测试必要的设置。"""
        return MultiViewClusteringModel(
            [5, 4, 6], 3, [8], 4,
            completion={"gate": False, **options},
        )

    def context(self, model, mask=None, bank=None, **options):
        """准备可复用锚点和轮次信息，不把隐藏真值送入补全监督。"""
        mask = self.mask if mask is None else mask
        return {
            "bank": model.build_completion_bank(self.views, mask, self.ids) if bank is None else bank,
            "sample_ids": self.ids,
            "round_number": 6,
            "center_init_round": 4,
            **options,
        }

    def test_default_mse_is_same_for_complete_and_missing_inputs(self):
        """核对两种掩码下均按有效目标视图计算归一化 MSE，而非切换损失类型。"""
        model = self.model()
        self.assertTrue(model.completion_enabled)
        self.assertEqual(model.completion_config["alignment_mode"], "mse")
        self.assertTrue(model.completion_config["use_completed_assignments"])
        teachers = tuple(torch.randn(8, 4, requires_grad=True) for _ in range(3))
        for mask in (torch.ones_like(self.mask), self.mask):
            with self.subTest(missing=int((~mask).sum())):
                outputs = model(self.views, mask, self.context(model, mask, teacher_embeddings=teachers))
                terms = []
                for view_index, predicted in enumerate(outputs["robust_predictions"]):
                    other = mask.clone()
                    other[:, view_index] = False
                    valid = mask[:, view_index] & other.any(dim=1) & (predicted.norm(dim=1) > 1e-12)
                    terms.append(F.mse_loss(predicted[valid], F.normalize(teachers[view_index].detach(), dim=1)[valid]))
                expected = torch.stack(terms).mean()
                self.assertTrue(torch.allclose(outputs["completion_loss"], expected, atol=1e-8))
                self.assertGreater(outputs["completion_loss"].item(), 0)
                model.zero_grad(set_to_none=True)
                outputs["completion_loss"].backward()
                self.assertTrue(all(teacher.grad is None for teacher in teachers))
                self.assertTrue(all(head.centers.grad is None for head in model.prototype_heads))
                self.assertGreater(model.completion_module.adapter_residual.grad.abs().sum().item(), 0)

    def test_completed_probability_matches_explicit_convex_mixture(self):
        """补全仅改变有缺失和支持的行，融合概率保持归一化且系数受限。"""
        model = self.model().eval()
        outputs = model(self.views, self.mask, self.context(model))
        beta = outputs["assignment_beta"]
        expected = (1 - beta[:, None]) * outputs["observed_assignments"] + beta[:, None] * outputs["imputed_assignments"]
        self.assertTrue(torch.equal(outputs["assignments"], expected))
        self.assertTrue(torch.allclose(outputs["assignments"].sum(dim=1), torch.ones(8), atol=1e-6))
        self.assertTrue(torch.isfinite(outputs["assignments"]).all())
        self.assertTrue((beta[6:] > 0).all())
        self.assertTrue((beta <= model.completion_config["fusion_weight"]).all())
        self.assertFalse(beta.requires_grad)
        self.assertFalse(outputs["completion_support_weights"].requires_grad)
        self.assertTrue(torch.equal(outputs["assignments"][:6], outputs["observed_assignments"][:6]))
        self.assertGreater((outputs["assignments"][6:] - outputs["observed_assignments"][6:]).abs().sum().item(), 0)
        for view_index, observed in enumerate(outputs["view_assignments"]):
            self.assertTrue((observed[~self.mask[:, view_index]] == 0).all())

    def test_imputed_probability_backpropagates_without_moving_centers(self):
        """仅从补全概率反传时，源编码与补全器获得梯度，原型中心没有梯度。"""
        model = self.model()
        outputs = model(self.views, self.mask, self.context(model))
        loss = -outputs["imputed_assignments"][6, 0].log()
        loss.backward()
        self.assertTrue(all(head.centers.grad is None for head in model.prototype_heads))
        self.assertGreater(model.completion_module.adapter_residual.grad.abs().sum().item(), 0)
        self.assertGreater(model.completion_module.attention.in_proj_weight.grad.abs().sum().item(), 0)
        for index in (0, 2):
            self.assertGreater(model.view_models[index].encoder[0].weight.grad.abs().sum().item(), 0)

    def test_no_support_and_observed_ablation_keep_observed_probabilities(self):
        """空锚点或显式关闭完成分配时不虚构概率证据，MSE 自监督仍存在。"""
        empty = {"sample_ids": self.ids[:0], "view_embeddings": tuple(torch.empty(0, 4) for _ in range(3))}
        for options, bank in (({}, empty), ({"use_completed_assignments": False}, None)):
            model = self.model(**options)
            outputs = model(self.views, self.mask, self.context(model, bank=bank))
            self.assertTrue(torch.equal(outputs["assignments"], outputs["observed_assignments"]))
            self.assertEqual(outputs["assignment_beta"].count_nonzero().item(), 0)
            self.assertGreater(outputs["completion_loss"].item(), 0)

    def test_dec_fallback_uses_observed_teacher_and_canonical_four_weights(self):
        """没有显式缓存时也从观测分布构造停止梯度的 DEC 教师，避免补全自证。"""
        model = self.model()
        outputs = model(self.views, self.mask, self.context(model))
        weights = {"reconstruction": 1.0, "completion": 0.7, "clustering": 0.4, "balance": 0.01}
        total, metrics = clustering_objective(outputs, self.views, weights, True, mask=self.mask)
        target = target_distribution(outputs["observed_assignments"].detach())
        expected_kl = F.kl_div(outputs["assignments"].log(), target, reduction="batchmean")
        self.assertAlmostEqual(metrics["clustering"], expected_kl.item(), places=7)
        self.assertAlmostEqual(metrics["completion"], outputs["completion_loss"].item(), places=7)
        expected_total = sum(metrics[key] for key in (
            "weighted_reconstruction", "weighted_completion", "weighted_clustering", "weighted_balance"
        ))
        self.assertAlmostEqual(total.item(), expected_total, places=6)

    def test_no_supervision_does_not_fall_back_to_observation_consistency(self):
        """教师无效时补全自监督是图内零，不能静默启用另一种一致性目标。"""
        model = self.model()
        teachers = tuple(torch.zeros(8, 4) for _ in range(3))
        outputs = model(self.views, self.mask, self.context(model, teacher_embeddings=teachers))
        self.assertEqual(outputs["completion_loss"].item(), 0)
        weights = {"reconstruction": 0.0, "completion": 1.0, "clustering": 0.0, "balance": 0.0}
        total, metrics = clustering_objective(outputs, self.views, weights, False, mask=self.mask)
        self.assertEqual(metrics["completion"], 0)
        self.assertEqual(total.item(), 0)
        total.backward()

    def test_disabled_completion_is_an_explicit_observed_baseline(self):
        """显式基线不创建补全参数，仍可以使用旧融合一致性作为独立对照。"""
        model = self.model(enabled=False)
        outputs = model(self.views, self.mask)
        self.assertFalse(any(key.startswith("completion_module.") for key in model.state_dict()))
        self.assertTrue(torch.equal(outputs["assignments"], outputs["observed_assignments"]))
        weights = {"reconstruction": 0.0, "completion": 1.0, "clustering": 0.0, "balance": 0.0}
        total, metrics = clustering_objective(outputs, self.views, weights, False, mask=self.mask)
        self.assertGreater(total.item(), 0)
        self.assertEqual(metrics["completion"], metrics["consistency"])

    def test_enabled_completion_cannot_silently_train_without_context(self):
        """观测前向可用于教师和计数，但正式补全模型缺少上下文时不能计算旧损失。"""
        model = self.model()
        outputs = model(self.views, self.mask)
        self.assertTrue(torch.equal(outputs["assignments"], outputs["observed_assignments"]))
        weights = {"reconstruction": 1.0, "completion": 1.0, "clustering": 0.1, "balance": 0.01}
        with self.assertRaisesRegex(ValueError, "requires completion_context"):
            clustering_objective(outputs, self.views, weights, True, mask=self.mask)


if __name__ == "__main__":
    unittest.main()
