"""验证补全参数作用域、Stage 校准上下文及固定字节预算之间的交互。"""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.communication.codec import (
    decode_sparse, dense_uplink_bytes,
    flatten_update, ordinary_keys, load_communication_state, reconstruct_state,
)
from flcore.communication.compression import compress_client_update
import flcore.communication.compression as compression
from flcore.trainmodel.multiview import MultiViewClusteringModel


class CompletionCompressionTests(unittest.TestCase):
    """检查补全模块在共享压缩、误差反馈和模型恢复过程中的边界。"""

    def test_local_imputer_survives_feedback_and_every_stage_probe_without_upload(self):
        """验证私有补全参数在 EF 参考与每个 Stage 候选中保留，且不会上传。"""
        torch.manual_seed(37)
        base = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={"enabled": False})
        torch.manual_seed(37)
        model = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={
            "enabled": True, "parameter_scope": "local", "heads": 2,
            "anchor_size": 8, "topk": 3,
        })
        local = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in local.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        private = {key: local.state_dict()[key].clone() for key in model.local_completion_state_keys()}
        self.assertEqual(ordinary_keys(base), ordinary_keys(model))
        self.assertEqual(dense_uplink_bytes(base, True), dense_uplink_bytes(model, True))
        keys = ordinary_keys(model)
        self.assertFalse(any(key.startswith("completion_module.") for key in keys))
        residual = torch.full_like(flatten_update(local, model, keys), 0.003)
        effective = flatten_update(local, model, keys) + residual
        views = (torch.randn(12, 8), torch.randn(12, 5))
        mask = torch.ones(12, 2, dtype=torch.bool)
        mask[::3, 1] = False
        ids = torch.arange(12)
        contexts, probes = [], []

        def assert_private(reference):
            """核对参考或候选模型中的补全参数仍等于该客户端私有状态。"""
            for key, expected in private.items():
                self.assertTrue(torch.equal(reference.state_dict()[key], expected), key)

        def builder(reference):
            """从包含误差反馈的有效参考模型建立一次可复用的补全上下文。"""
            assert_private(reference)
            self.assertTrue(torch.allclose(flatten_update(reference, model, keys), effective, atol=1e-7))
            bank = reference.build_completion_bank(views, mask, ids, seed=13)
            context = {"bank": bank, "sample_ids": ids, "round_number": 3, "center_init_round": 2}
            contexts.append(context)
            return context

        actual_calibration = compression._calibration_outputs

        def calibration(reference, *args, **kwargs):
            """拦截校准前向，检查私有模块和共享上下文在各候选间保持一致。"""
            assert_private(reference)
            self.assertIs(kwargs["completion_context"], contexts[0])
            probes.append(reference)
            return actual_calibration(reference, *args, **kwargs)

        settings = {"method": "stage", "budget_ratio": 0.5,
                    "budget_bytes": int(dense_uplink_bytes(base, True) * 0.5),
                    "candidates": 4, "view_weight": 0.1, "pair_weight": 0.1,
                    "loss_weights": {"reconstruction": 1., "consistency": .05,
                                     "clustering": .1, "balance": .05}}
        with patch.object(compression, "_calibration_outputs", side_effect=calibration):
            packet = compress_client_update(
                local, model, settings, views=views, mask=mask,
                targets=torch.full((12, 3), 1 / 3), centers_enabled=True,
                residual=residual, completion_context_builder=builder,
            )
        self.assertEqual(len(contexts), 1)
        self.assertGreater(len(probes), 1)
        sent = decode_sparse(packet["payload"], effective.numel())
        self.assertTrue(torch.allclose(sent + packet["residual"], effective))
        shared = reconstruct_state(model, packet["payload"], packet["centers"])
        self.assertFalse(set(private).intersection(shared))
        load_communication_state(local, shared)
        assert_private(local)
        self.assertLessEqual(packet["bytes"], settings["budget_bytes"])

    def test_local_state_restore_rejects_missing_shared_parameters(self):
        """验证允许省略私有参数的恢复过程仍拒绝缺少共享参数的载荷。"""
        model = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={
            "enabled": True, "parameter_scope": "local", "heads": 2,
        })
        state = {key: value for key, value in model.state_dict().items()
                 if key not in model.local_completion_state_keys()}
        state.pop(ordinary_keys(model)[0])
        with self.assertRaisesRegex(RuntimeError, "missing shared"):
            load_communication_state(model, state)

    def test_absolute_budget_and_stage_pair_override_relative_budget(self):
        """验证单一绝对预算优先于分阶段预算，分阶段预算优先于相对比例。"""
        model = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={"enabled": False})
        local = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in local.parameters():
                parameter.add_(0.01)
        config = {"method": "topk", "budget_ratio": 0.99,
                  "budget_bytes_pair": {"pre_center": 300, "post_center": 500}}
        before = compress_client_update(local, model, config, centers_enabled=False)
        after = compress_client_update(local, model, config, centers_enabled=True)
        self.assertEqual(before["budget_bytes"], 300)
        self.assertEqual(after["budget_bytes"], 500)
        self.assertLessEqual(before["bytes"], 300)
        self.assertLessEqual(after["bytes"], 500)
        explicit = compress_client_update(local, model, {**config, "budget_bytes": 400})
        self.assertEqual(explicit["budget_bytes"], 400)

    def test_effective_feedback_reference_builds_one_local_context(self):
        """验证共享补全模式用 EF 修正后的参考模型构建唯一校准上下文。"""
        torch.manual_seed(29)
        model = MultiViewClusteringModel(
            [8, 5], 3, [6], 4,
            completion={"enabled": True, "anchor_size": 8, "heads": 2,
                        "topk": 3, "temperature": 0.2, "truncation": 9},
        )
        local = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in local.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        keys = ordinary_keys(model)
        self.assertTrue(any(key.startswith("completion_module.") for key in keys))
        self.assertFalse(any("bank" in key or "sample_ids" in key for key in keys))
        residual = torch.full_like(flatten_update(local, model, keys), 0.003)
        effective = flatten_update(local, model, keys) + residual
        views = (torch.randn(12, 8), torch.randn(12, 5))
        mask = torch.ones((12, 2), dtype=torch.bool)
        mask[::3, 1] = False
        ids = torch.arange(12)
        seen = []

        def build_context(reference):
            """记录参考更新，并从该参考模型生成停止梯度的锚点和观测教师。"""
            seen.append(flatten_update(reference, model, keys))
            bank = reference.build_completion_bank(views, mask, ids, seed=11)
            embeddings, _ = reference.encode(views, mask)
            return {"bank": bank, "sample_ids": ids, "round_number": 3,
                    "center_init_round": 2,
                    "teacher_embeddings": tuple(F.normalize(z, dim=1).detach()
                                                for z in embeddings)}

        packet = compress_client_update(
            local, model,
            {"method": "stage", "budget_ratio": 0.5,
             "budget_bytes": int(dense_uplink_bytes(model, True) * 0.5),
             "candidates": 4, "view_weight": 0.1, "pair_weight": 0.1,
             "loss_weights": {"reconstruction": 1., "consistency": .1,
                              "clustering": .1, "balance": .05}},
            views=views, mask=mask, targets=torch.full((12, 3), 1 / 3),
            centers_enabled=True, residual=residual,
            completion_context_builder=build_context,
        )
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.allclose(seen[0], effective, atol=1e-7))
        sent = decode_sparse(packet["payload"], effective.numel())
        self.assertTrue(torch.allclose(sent + packet["residual"], effective))
        self.assertLessEqual(packet["bytes"], packet["budget_bytes"])
        self.assertTrue(torch.isfinite(torch.tensor(packet["score"])))


if __name__ == "__main__":
    unittest.main()
