"""验证潜在补全的观测隔离、锚点排除、鲁棒损失、梯度及有界融合。"""

from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from flcore.trainmodel.completion import truncated_robust_loss  # noqa: E402
from flcore.trainmodel.multiview import MultiViewClusteringModel  # noqa: E402
from flcore.trainmodel.losses import clustering_objective  # noqa: E402


class CompletionTests(unittest.TestCase):
    """检查统一补全目标、显式消融模式及真实观测边界。"""

    def setUp(self):
        """构造可重复的三视图输入、固定缺失掩码和全局样本编号。"""
        torch.manual_seed(71)
        self.views = (torch.randn(12, 5), torch.randn(12, 4), torch.randn(12, 6))
        self.mask = torch.ones((12, 3), dtype=torch.bool)
        self.mask[8, 0] = False
        self.mask[9, 1] = False
        self.mask[10, 2] = False
        self.ids = torch.arange(100, 112)

    def model(self, **options):
        """创建启用补全的小型 A2 模型，并应用当前测试指定的参数。"""
        return MultiViewClusteringModel(
            [5, 4, 6], 3, [8], 4,
            completion={"enabled": True, **options},
        )

    def context(self, model, bank=None, **options):
        """组合本地锚点、样本编号和训练轮次，允许测试替换教师或锚点。"""
        return {
            "bank": model.build_completion_bank(self.views, self.mask, self.ids, seed=91)
            if bank is None else bank,
            "sample_ids": self.ids,
            "round_number": 6,
            "center_init_round": 4,
            **options,
        }

    def test_new_initialization_preserves_original_parameters_and_rng(self):
        """验证新增补全模块不改变原 A2 参数、CPU 随机状态和无上下文前向结果。"""
        torch.manual_seed(123)
        baseline = MultiViewClusteringModel([5, 4, 6], 3, [8], 4,
                                            completion={"enabled": False})
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        completed = self.model()
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, completed.state_dict()[key]), key)
        old_outputs = baseline(self.views, self.mask)
        new_outputs = completed(self.views, self.mask)
        for key in ("assignments", "embedding"):
            self.assertTrue(torch.equal(old_outputs[key], new_outputs[key]))

    def test_parameter_scope_validation_and_private_state_keys(self):
        """验证补全参数作用域有效性，以及私有键与原型中心键的隔离。"""
        federated = self.model()
        self.assertEqual(federated.completion_parameter_scope, "federated")
        self.assertEqual(federated.local_completion_state_keys(), ())
        local = self.model(parameter_scope="local")
        private_keys = local.local_completion_state_keys()
        self.assertTrue(private_keys)
        self.assertEqual(set(private_keys), {key for key in local.state_dict() if key.startswith("completion_module.")})
        self.assertFalse(set(private_keys).intersection(local.prototype_center_keys()))
        disabled = MultiViewClusteringModel([5, 4, 6], 3, [8], 4,
                                            completion={"enabled": False, "parameter_scope": "local"})
        self.assertEqual(disabled.local_completion_state_keys(), ())
        with self.assertRaisesRegex(ValueError, "parameter_scope"):
            self.model(parameter_scope="shared")
        with self.assertRaisesRegex(ValueError, "parameter_scope"):
            MultiViewClusteringModel([5, 4, 6], 3, [8], 4,
                                     completion={"enabled": False, "parameter_scope": "invalid"})

    def test_local_scope_preserves_initialization_rng_observations_and_four_losses(self):
        """验证仅切换私有作用域时初始化、前向结果及四项损失数值保持一致。"""
        for alignment_mode in ("mse", "contrastive", "paired"):
            torch.manual_seed(519)
            federated = self.model(parameter_scope="federated", alignment_mode=alignment_mode)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(519)
            local = self.model(parameter_scope="local", alignment_mode=alignment_mode)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for key, value in federated.state_dict().items():
                self.assertTrue(torch.equal(value, local.state_dict()[key]), key)
            context = self.context(federated)
            global_outputs = federated(self.views, self.mask, context)
            local_outputs = local(self.views, self.mask, context)
            for key in ("assignments", "embedding", "completed_embedding", "robust_loss"):
                self.assertTrue(torch.equal(global_outputs[key], local_outputs[key]), key)
            weights = {"reconstruction": 1.0, "consistency": 0.2, "clustering": 0.05, "balance": 0.05}
            global_total, _ = clustering_objective(global_outputs, self.views, weights, True, mask=self.mask)
            local_total, metrics = clustering_objective(local_outputs, self.views, weights, True, mask=self.mask)
            self.assertTrue(torch.equal(global_total, local_total))
            self.assertAlmostEqual(local_total.item(), sum(metrics[key] for key in
                ("weighted_reconstruction", "weighted_consistency", "weighted_clustering", "weighted_balance")), places=6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA RNG preservation requires an available CUDA device")
    def test_new_cpu_module_initialization_preserves_cuda_rng_states(self):
        """验证 CPU 上初始化补全模块不会重置已存在的 CUDA 随机状态。"""
        torch.cuda.manual_seed_all(9182)
        before = [value.clone() for value in torch.cuda.get_rng_state_all()]
        self.model(seed=7)
        after = torch.cuda.get_rng_state_all()
        self.assertEqual(len(before), len(after))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(before, after)))

    def test_bank_is_complete_detached_capped_and_external(self):
        """验证锚点仅来自完整行、满足容量上限、停止梯度且不进入模型状态。"""
        model = self.model(anchor_size=4)
        model.train()
        rng_state = torch.get_rng_state().clone()
        bank = model.build_completion_bank(self.views, self.mask, self.ids, seed=13)
        again = model.build_completion_bank(self.views, self.mask, self.ids, seed=13)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        self.assertEqual(len(bank["sample_ids"]), 4)
        self.assertTrue(torch.equal(bank["sample_ids"], again["sample_ids"]))
        self.assertTrue(set(bank["sample_ids"].tolist()).issubset(set(self.ids[:8].tolist())))
        self.assertTrue(all(not value.requires_grad for value in bank["view_embeddings"]))
        self.assertFalse(any("bank" in key or "sample_ids" in key for key in model.state_dict()))
        empty = model.build_completion_bank(tuple(view[:0] for view in self.views),
                                            self.mask[:0], self.ids[:0])
        self.assertEqual(empty["view_embeddings"][0].shape, (0, 4))
        model.eval()
        model.build_completion_bank(self.views, self.mask, self.ids)
        self.assertFalse(model.training)

    def test_hidden_values_do_not_change_completion_or_observation_statistics(self):
        """验证隐藏真值扰动不影响锚点、补全、观测分配及训练损失。"""
        model = self.model(gate=False).eval()
        changed = tuple(view.clone() for view in self.views)
        for view_index in range(3):
            changed[view_index][~self.mask[:, view_index]] = 1e8
        first_bank = model.build_completion_bank(self.views, self.mask, self.ids)
        second_bank = model.build_completion_bank(changed, self.mask, self.ids)
        for left, right in zip(first_bank["view_embeddings"], second_bank["view_embeddings"]):
            self.assertTrue(torch.equal(left, right))
        context = self.context(model, first_bank)
        original_mask = self.mask.clone()
        first = model(self.views, self.mask, context)
        second = model(changed, self.mask, context)
        for key in ("embedding", "assignments", "completed_embedding", "robust_loss"):
            self.assertTrue(torch.equal(first[key], second[key]), key)
        self.assertTrue(torch.equal(original_mask, self.mask))
        for index, assignment in enumerate(first["view_assignments"]):
            self.assertTrue((assignment[~self.mask[:, index]] == 0).all())
        weights = {"reconstruction": 1.0, "consistency": 0.2, "clustering": 0.05, "balance": 0.05}
        first_loss, _ = clustering_objective(first, self.views, weights, True, mask=self.mask)
        second_loss, _ = clustering_objective(second, changed, weights, True, mask=self.mask)
        self.assertTrue(torch.equal(first_loss, second_loss))

    def test_self_anchor_duplicates_and_target_are_excluded_from_prediction(self):
        """验证所有同 ID 锚点和当前目标视图都被排除，避免自监督泄漏。"""
        model = self.model(gate=False).eval()
        observed = model(self.views, self.mask)
        bank = model.build_completion_bank(self.views, self.mask, self.ids)
        duplicate_bank = {
            "sample_ids": torch.cat((bank["sample_ids"], bank["sample_ids"][:1])),
            "view_embeddings": tuple(torch.cat((value, value[:1])) for value in bank["view_embeddings"]),
        }
        first = model.completion_forward(observed, self.mask, self.ids, duplicate_bank, 6, 4)
        changed_bank = {"sample_ids": duplicate_bank["sample_ids"],
                        "view_embeddings": tuple(value.clone() for value in duplicate_bank["view_embeddings"])}
        own_rows = changed_bank["sample_ids"] == self.ids[0]
        # 同一 ID 可能在锚点库重复出现，必须排除所有副本而非只跳过一行。
        changed_bank["view_embeddings"][0][own_rows] = torch.tensor([1e7, -1e7, 1e7, -1e7])
        changed_observed = dict(observed)
        changed_observed["view_embeddings"] = list(observed["view_embeddings"])
        changed_observed["view_embeddings"][0] = observed["view_embeddings"][0].clone()
        changed_observed["view_embeddings"][0][0] *= -100
        second = model.completion_forward(changed_observed, self.mask, self.ids, changed_bank, 6, 4)
        self.assertTrue(torch.equal(first["robust_predictions"][0][0], second["robust_predictions"][0][0]))

    def test_robust_loss_has_exactly_c_terms_and_bounded_probability_derivative(self):
        """验证鲁棒级数恰有 C 项，且对输入概率的导数满足相应界限。"""
        probabilities = torch.tensor([0.0, 0.4, 1.0], requires_grad=True)
        values = truncated_robust_loss(probabilities, 9)
        self.assertAlmostEqual(values[0].item(), sum(1 / order for order in range(1, 10)), places=6)
        self.assertEqual(values[2].item(), 0.0)
        values.sum().backward()
        self.assertTrue((probabilities.grad.abs() <= 9.00001).all())
        self.assertAlmostEqual(probabilities.grad[0].item(), -9.0)

    def test_single_replacement_loss_trains_sources_and_new_parameters(self):
        """验证补全自监督训练源编码与补全参数，不更新教师和原型。"""
        model = self.model(gate=False)
        bank = model.build_completion_bank(self.views, self.mask, self.ids)
        # 教师显式允许求导，用梯度是否为空检查实现是否真正切断监督目标的梯度。
        teacher = tuple(F.normalize(torch.randn(12, 4), dim=1).requires_grad_() for _ in range(3))
        context = self.context(model, bank, teacher_embeddings=teacher)
        outputs = model(self.views, self.mask, context)
        outputs["robust_loss"].backward()
        self.assertIsNotNone(model.completion_module.adapter_residual.grad)
        self.assertGreater(model.completion_module.adapter_residual.grad.abs().sum().item(), 0)
        self.assertGreater(model.completion_module.attention.in_proj_weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.view_models[0].encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(value.grad is None for value in teacher))
        self.assertTrue(all(head.centers.grad is None for head in model.prototype_heads))
        weights = {"reconstruction": 1.0, "consistency": 0.2, "clustering": 0.05, "balance": 0.05}
        total, metrics = clustering_objective(outputs, self.views, weights, True, mask=self.mask)
        expected = sum(metrics[key] for key in ("weighted_reconstruction", "weighted_consistency",
                                               "weighted_clustering", "weighted_balance"))
        self.assertAlmostEqual(total.item(), expected, places=6)
        self.assertEqual(metrics["consistency"], outputs["robust_loss"].item())

    def test_adapter_and_completed_mixture_are_bounded_even_with_extreme_weights(self):
        """验证极端参数下适配器范数和补全占比仍受限，退化观测融合可安全回退。"""
        model = self.model(gate=False).eval()
        with torch.no_grad():
            # 使用远超正常训练范围的参数，避免约束只在小权重时成立。
            model.completion_module.adapter_residual.fill_(1000)
            model.view_logits.copy_(torch.tensor([100.0, -100.0, -100.0]))
        for index in range(3):
            residual = model.completion_module.adapter_matrix(index) - torch.eye(4)
            self.assertLessEqual(torch.linalg.matrix_norm(residual, ord=2).item(), 0.100001)
        outputs = model(self.views, self.mask, self.context(model))
        self.assertTrue((outputs["completion_beta"] <= 0.25).all())
        complete = self.mask.all(dim=1)
        self.assertTrue(torch.equal(outputs["completed_embedding"][complete], outputs["embedding"][complete]))
        cancelled = dict(model(self.views, self.mask))
        cancelled["embedding"] = cancelled["embedding"].clone()
        cancelled["embedding"][8] = 0
        completion = model.completion_forward(cancelled, self.mask, self.ids,
                                              self.context(model)["bank"], 6, 4)
        self.assertEqual(completion["completion_beta"][8].item(), 0)
        self.assertEqual(completion["completed_embedding"][8].norm().item(), 0)

    def test_empty_one_anchor_and_one_row_batches_are_finite(self):
        """验证空锚点、单锚点和单行 batch 下补全与一致性损失均有限。"""
        model = self.model().eval()
        empty = {"sample_ids": self.ids[:0], "view_embeddings": tuple(torch.empty(0, 4) for _ in range(3))}
        one = model.build_completion_bank(tuple(view[:1] for view in self.views), self.mask[:1], self.ids[:1])
        for bank in (empty, one):
            outputs = model(self.views, self.mask, self.context(model, bank))
            self.assertTrue(torch.isfinite(outputs["robust_loss"]))
            self.assertTrue(torch.isfinite(outputs["completed_embedding"]).all())
            self.assertTrue(torch.equal(outputs["completed_embedding"], outputs["embedding"]))
        context = self.context(model)
        context["sample_ids"] = self.ids[8:9]
        output = model(tuple(view[8:9] for view in self.views), self.mask[8:9], context)
        self.assertGreater(output["completion_loss"].item(), 0)
        self.assertEqual(output["completion_metrics"]["completion_supervised_pairs"], 2)
        self.assertTrue(torch.isfinite(output["completed_embedding"]).all())

    def test_paired_single_row_has_exact_finite_loss_per_dimension_and_stopped_targets(self):
        """验证单行 paired 损失按表示维度归一化，且只向预测源反传梯度。"""
        model = self.model(alignment_mode="paired", use_attention=False)
        observed_embeddings = [torch.eye(4)[v:v + 1].clone().requires_grad_() for v in range(3)]
        teacher = tuple(torch.tensor([[0.0, 0.0, 0.0, 1.0]], requires_grad=True) for _ in range(3))
        observed = {"view_embeddings": observed_embeddings,
                    "embedding": F.normalize(sum(observed_embeddings), dim=1)}
        bank = {"sample_ids": torch.empty(0, dtype=torch.long),
                "view_embeddings": tuple(torch.empty(0, 4) for _ in range(3))}
        completed = model.completion_forward(observed, torch.ones(1, 3, dtype=torch.bool),
                                               torch.tensor([42]), bank, 6, 4,
                                               teacher_embeddings=teacher)
        # 正交预测与教师使配对一致度恰为 0.5，便于直接核对有限级数和维度归一化。
        expected = sum((0.5 ** order) / order for order in range(1, 10)) / 4
        self.assertAlmostEqual(completed["robust_loss"].item(), expected, places=7)
        self.assertEqual(completed["completion_metrics"]["completion_robust_pairs"], 3)
        self.assertEqual(completed["completion_metrics"]["completion_positive_pair_cosine"], 0)
        completed["robust_loss"].backward()
        self.assertTrue(all(value.grad is not None and value.grad.norm() > 0 for value in observed_embeddings))
        self.assertTrue(all(value.grad is None for value in teacher))

    def test_paired_retains_mask_privacy_observed_assignments_and_four_losses(self):
        """验证 paired 模式保持隐藏值隔离、真实观测分配及四项损失结构。"""
        model = self.model(alignment_mode="paired", gate=False).eval()
        bank = model.build_completion_bank(self.views, self.mask, self.ids)
        context = self.context(model, bank)
        first = model(self.views, self.mask, context)
        changed = tuple(view.clone() for view in self.views)
        for view_index in range(3):
            changed[view_index][~self.mask[:, view_index]] = -1e7
        second = model(changed, self.mask, context)
        self.assertTrue(torch.equal(first["robust_loss"], second["robust_loss"]))
        self.assertTrue(torch.equal(first["completed_embedding"], second["completed_embedding"]))
        self.assertTrue(torch.equal(first["observed_assignments"], model(self.views, self.mask)["assignments"]))
        weights = {"reconstruction": 1.0, "consistency": 0.2, "clustering": 0.05, "balance": 0.05}
        total, metrics = clustering_objective(first, self.views, weights, True, mask=self.mask)
        expected = sum(metrics[key] for key in ("weighted_reconstruction", "weighted_consistency",
                                               "weighted_clustering", "weighted_balance"))
        self.assertAlmostEqual(total.item(), expected, places=6)
        self.assertEqual(metrics["consistency"], first["robust_loss"].item())


if __name__ == "__main__":
    unittest.main()
