"""验证完整与缺失场景共用补全聚类流程，以及评估状态的客户端边界。"""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from test_completion_integration import toy_config, toy_data, toy_model
from flcore.clients.clientcluster import FederatedClusteringClient
from flcore.servers.servercluster import FederatedMultiViewClusteringServer
from flcore.trainmodel.multiview import MultiViewClusteringModel
from flcore.trainmodel.losses import (
    clustering_objective,
    target_distribution,
)


class UnifiedClientTests(unittest.TestCase):
    """检查正式训练和评估真正使用统一路径，而非仅检查配置字段。"""

    @classmethod
    def setUpClass(cls):
        """限制小型 CPU 测试的线程开销，并保存进程原有设置。"""
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """恢复其他测试可能依赖的线程设置。"""
        torch.set_num_threads(cls.previous_threads)

    def client(self, data, config, indices=None):
        """创建本地测试客户端，允许采用非连续的全局样本标识。"""
        if indices is None:
            indices = np.arange(data.num_samples)
        return FederatedClusteringClient(0, data, indices, config, torch.device("cpu"))

    def test_complete_and_missing_use_only_local_observed_normalization(self):
        """完整数据也使用全真掩码和本地统计；零缺失率与关闭缺失生成等价。"""
        data = toy_data()
        indices = np.array([0, 2, 3, 5, 7, 8, 11, 13, 14, 17, 20, 23])
        complete_views = []
        for enabled, rate in ((False, 0.5), (True, 0.0), (True, 0.5)):
            with self.subTest(enabled=enabled, rate=rate):
                config = toy_config()
                config["dataset"]["normalization"] = "standard"
                config["missing"].update(enabled=enabled, rate=rate)
                client = self.client(data, config, indices)
                mask = client.dataset.mask.numpy()
                self.assertEqual(len(client.dataset[0]), 4)
                self.assertTrue(mask.any(axis=1).all())
                for view_index, actual in enumerate(client.dataset.views):
                    observed = mask[:, view_index]
                    raw = data.views[view_index][indices[observed]]
                    mean = raw.mean(axis=0, dtype=np.float64).astype(np.float32)
                    std = raw.std(axis=0, dtype=np.float64).astype(np.float32)
                    std[std < 1e-6] = 1.0
                    np.testing.assert_allclose(actual.numpy()[observed], (raw - mean) / std)
                    self.assertTrue(torch.equal(actual[~observed], torch.zeros_like(actual[~observed])))
                if not enabled or rate == 0:
                    self.assertTrue(mask.all())
                    complete_views.append(client.dataset.views)
                else:
                    self.assertTrue((~mask).any())

                # 他人的样本及本地隐藏原始值均不能改变该客户端的预处理结果。
                changed = copy.deepcopy(data)
                outsiders = np.setdiff1d(np.arange(data.num_samples), indices)
                for view_index, view in enumerate(changed.views):
                    view[outsiders] = 1e8
                    view[indices[~mask[:, view_index]]] = -1e8
                other = self.client(changed, config, indices)
                for first, second in zip(client.dataset.views, other.dataset.views):
                    self.assertTrue(torch.equal(first, second))
        for first, second in zip(*complete_views):
            self.assertTrue(torch.equal(first, second))

    def test_full_and_missing_training_use_context_and_completion_mse(self):
        """两种数据都在每个训练批次构造补全监督，不切换到旧融合一致性。"""
        original = MultiViewClusteringModel.forward
        for rate in (0.0, 0.5):
            with self.subTest(rate=rate):
                config, data = toy_config(), toy_data()
                config["missing"]["rate"] = rate
                config["training"]["clustering_local_epochs"] = 1
                client, model = self.client(data, config), toy_model(config)
                calls = []

                def capture_forward(instance, views, mask=None, completion_context=None):
                    """在正式训练调用处核对上下文与 MSE 模式。"""
                    if instance.training:
                        self.assertIsNotNone(completion_context)
                        self.assertEqual(completion_context["round_number"], 3)
                        self.assertTrue(completion_context.get("compute_supervision", True))
                        self.assertEqual(instance.completion_config["alignment_mode"], "mse")
                    return original(instance, views, mask=mask, completion_context=completion_context)

                def capture_loss(outputs, views, weights, **kwargs):
                    """检查反向传播采用补全损失及统一四项权重。"""
                    self.assertTrue(outputs["completion_enabled"])
                    self.assertIsNotNone(kwargs["mask"])
                    loss, metrics = clustering_objective(outputs, views, weights, **kwargs)
                    self.assertEqual(metrics["completion"], float(outputs["completion_loss"].detach()))
                    self.assertGreater(metrics["completion_supervised_pairs"], 0)
                    self.assertAlmostEqual(metrics["completion"], metrics["completion_mse"], places=7)
                    expected = sum(metrics[name] for name in (
                        "weighted_reconstruction", "weighted_completion", "weighted_clustering", "weighted_balance",
                    ))
                    self.assertAlmostEqual(float(loss.detach()), expected, places=5)
                    calls.append(metrics["completion"])
                    return loss, metrics

                with patch.object(MultiViewClusteringModel, "forward", new=capture_forward), patch(
                    "flcore.clients.clientcluster.clustering_objective", side_effect=capture_loss,
                ):
                    update = client.train(model, True, 2, "clustering")
                self.assertEqual(len(calls), 3)
                self.assertGreater(update["metrics"]["completion"], 0)

    def test_target_cache_uses_observed_probabilities_over_all_local_samples(self):
        """故意替换最终预测，确认 DEC 教师只锐化全本地观测分配。"""
        config, data = toy_config(), toy_data()
        config["training"]["batch_size"] = 5
        indices = np.array([23, 2, 18, 5, 0, 17, 7, 13, 3, 21, 9, 11])
        client, model = self.client(data, config, indices), toy_model(config)
        observed = torch.tensor([
            [0.95, 0.05], [0.7, 0.3], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1],
            [0.2, 0.8], [0.4, 0.6], [0.1, 0.9], [0.3, 0.7], [0.25, 0.75],
            [0.65, 0.35], [0.85, 0.15],
        ])
        offset = 0

        def forward_with_distinct_predictions(views, mask=None, completion_context=None):
            """提供不同的观测教师与补全预测，使错误取值可被断言识别。"""
            nonlocal offset
            self.assertIsNone(completion_context)
            batch = observed[offset:offset + len(views[0])]
            offset += len(batch)
            return {"observed_assignments": batch, "assignments": batch.flip(dims=[1])}

        with patch.object(model, "forward", side_effect=forward_with_distinct_predictions):
            cache = client.local_target_cache(model)
        expected = target_distribution(observed)
        self.assertEqual(offset, len(indices))
        torch.testing.assert_close(cache[indices], expected)
        self.assertFalse(torch.allclose(cache[indices], target_distribution(observed.flip(dims=[1]))))
        per_batch = torch.cat([target_distribution(part) for part in observed.split(5)])
        self.assertFalse(torch.allclose(expected, per_batch))
        self.assertFalse(cache.requires_grad)

    def test_training_reuses_one_fixed_target_cache_across_local_epochs(self):
        """本轮缓存只生成一次，按全局样本 ID 为全部本地 epoch 提供同一教师。"""
        config, data = toy_config(), toy_data()
        indices = np.array([22, 0, 3, 19, 8, 5, 17, 9, 2, 14, 11, 21])
        config["training"].update(batch_size=5, clustering_local_epochs=2)
        client, model = self.client(data, config, indices), toy_model(config)
        original_forward = MultiViewClusteringModel.forward
        original_cache = client.local_target_cache
        cache_snapshots, seen = [], []

        def capture_cache(instance):
            """在首个梯度更新前保存本轮教师快照。"""
            result = original_cache(instance)
            cache_snapshots.append(result.clone())
            return result

        def capture_forward(instance, views, mask=None, completion_context=None):
            """把当前批次的样本 ID 传给断言，不改变任何模型张量。"""
            result = original_forward(instance, views, mask=mask, completion_context=completion_context)
            if completion_context is not None:
                result["test_sample_ids"] = completion_context["sample_ids"].cpu()
            return result

        def capture_loss(outputs, views, weights, **kwargs):
            """比较每个批次收到的教师与本轮开始时的固定缓存。"""
            ids = outputs["test_sample_ids"]
            torch.testing.assert_close(kwargs["target_assignments"], cache_snapshots[0][ids])
            seen.extend(ids.tolist())
            return clustering_objective(outputs, views, weights, **kwargs)

        with patch.object(client, "local_target_cache", side_effect=capture_cache) as cache_call, patch.object(
            MultiViewClusteringModel, "forward", new=capture_forward,
        ), patch("flcore.clients.clientcluster.clustering_objective", side_effect=capture_loss):
            client.train(model, True, 2, "clustering")
        self.assertEqual(cache_call.call_count, 1)
        self.assertEqual(sorted(seen), sorted(indices.tolist() * 2))

    def local_server(self):
        """建立具有不同客户端私有参数的小型服务器，不运行写盘流程。"""
        config = toy_config()
        config["completion"].update(parameter_scope="local", anchor_size=3)
        config["training"]["center_init_round"] = 10
        torch.manual_seed(41)
        server = FederatedMultiViewClusteringServer(toy_data(), config, torch.device("cpu"))
        for client in server.clients:
            client._load_private_completion(server.global_model)
            for value in client.private_completion_state.values():
                value.add_(0.02 * (client.id + 1))
        return server

    def test_evaluation_uses_each_clients_private_state_bank_and_round(self):
        """评估按客户端恢复私有补全器，仅访问本地锚点并跳过自监督计算。"""
        server = self.local_server()
        server.current_round = 7
        original = MultiViewClusteringModel.forward
        seen = {client.id: [] for client in server.clients}
        predict_calls = []
        original_predict = FederatedClusteringClient.predict
        global_before = copy.deepcopy(server.global_model.state_dict())

        def inspect_predict(client, global_model, round_number):
            """核对公共预测接口的模型、轮号和脱离计算图的 CPU 返回值。"""
            self.assertIs(global_model, server.global_model)
            self.assertEqual(round_number, 7)
            result = original_predict(client, global_model, round_number)
            self.assertEqual(set(result), {"assignments", "labels", "assignment_beta"})
            for value in result.values():
                self.assertIsInstance(value, torch.Tensor)
                self.assertEqual(value.device.type, "cpu")
                self.assertFalse(value.requires_grad)
            self.assertEqual(result["assignments"].shape, (len(client.dataset), 2))
            self.assertEqual(result["assignment_beta"].numel(), len(client.dataset))
            self.assertTrue(torch.equal(result["labels"], client.dataset.labels))
            predict_calls.append(client.id)
            return result

        def inspect_forward(instance, views, mask=None, completion_context=None):
            """检查每个评估批次的实际模型参数和上下文归属。"""
            self.assertFalse(instance.training)
            self.assertIsNotNone(completion_context)
            self.assertFalse(completion_context["compute_supervision"])
            self.assertEqual(completion_context["round_number"], 7)
            batch_ids = set(completion_context["sample_ids"].tolist())
            clients = [client for client in server.clients if batch_ids <= set(client.indices)]
            self.assertEqual(len(clients), 1)
            client = clients[0]
            seen[client.id].extend(batch_ids)
            complete_ids = set(client.dataset.indices[client.dataset.mask.all(dim=1)].tolist())
            bank_ids = set(completion_context["bank"]["sample_ids"].tolist())
            self.assertTrue(bank_ids)
            self.assertTrue(bank_ids <= complete_ids)
            for key, value in client.private_completion_state.items():
                self.assertTrue(torch.equal(instance.state_dict()[key], value), key)
            output = original(instance, views, mask=mask, completion_context=completion_context)
            self.assertEqual(float(output["completion_loss"]), 0.0)
            self.assertEqual(output["completion_metrics"]["completion_supervised_pairs"], 0.0)
            return output

        with patch.object(MultiViewClusteringModel, "forward", new=inspect_forward), patch.object(
            FederatedClusteringClient, "predict", new=inspect_predict,
        ):
            metrics = server.evaluate()
        self.assertEqual(predict_calls, [client.id for client in server.clients])
        for client in server.clients:
            self.assertEqual(sorted(seen[client.id]), sorted(client.indices.tolist()))
        for key, value in global_before.items():
            self.assertTrue(torch.equal(server.global_model.state_dict()[key], value), key)
        self.assertTrue(np.isfinite(metrics["confidence"]))
        self.assertGreater(server.last_diagnostics["completion_assignment_beta"], 0.0)

    def test_server_evaluation_requires_only_public_client_predictions(self):
        """服务端可汇总仅提供 predict 的客户端，不依赖其数据集或私有补全方法。"""
        server = self.local_server()
        server.current_round = 7
        first = {"assignments": torch.tensor([[0.9, 0.1], [0.2, 0.8]]),
                 "labels": torch.tensor([0, 1]), "assignment_beta": torch.tensor([0.0, 0.2])}
        second = {"assignments": torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
                  "labels": torch.tensor([1, 0]), "assignment_beta": torch.tensor([0.1, 0.1])}
        clients = [SimpleNamespace(predict=Mock(return_value=prediction)) for prediction in (first, second)]
        server.clients = clients
        metrics = server.evaluate()
        for client in clients:
            client.predict.assert_called_once()
            args, kwargs = client.predict.call_args
            self.assertIs(args[0], server.global_model)
            round_number = kwargs["round_number"] if "round_number" in kwargs else args[1]
            self.assertEqual(round_number, 7)
        for name in ("acc", "nmi", "ari"):
            self.assertAlmostEqual(metrics[name], 1.0)
        self.assertAlmostEqual(metrics["confidence"], 0.85, places=6)
        self.assertAlmostEqual(server.last_diagnostics["completion_assignment_beta"], 0.1, places=6)

    def test_best_restore_recovers_global_private_round_and_exact_evaluation(self):
        """最佳检查点联合恢复共享参数、私有参数和调度轮号，重现同一预测指标。"""
        server = self.local_server()
        server.current_round = 3
        initial_state = copy.deepcopy(server.global_model.state_dict())
        server._restore_best()
        self.assertEqual(server.current_round, 3)
        for key, value in initial_state.items():
            self.assertTrue(torch.equal(server.global_model.state_dict()[key], value))
        best_metrics = server.evaluate()
        best_diagnostics = copy.deepcopy(server.last_diagnostics)
        server._update_best(best_metrics, 3, "clustering")
        private_snapshots = {client.id: copy.deepcopy(client.private_completion_state) for client in server.clients}
        with torch.no_grad():
            for parameter in server.global_model.parameters():
                parameter.add_(0.1)
        for client in server.clients:
            for value in client.private_completion_state.values():
                value.add_(0.2)
        server.current_round = 20
        server._restore_best()
        self.assertEqual(server.current_round, 3)
        for key, value in initial_state.items():
            self.assertTrue(torch.equal(server.global_model.state_dict()[key], value), key)
        for client in server.clients:
            for key, value in private_snapshots[client.id].items():
                self.assertTrue(torch.equal(client.private_completion_state[key], value), key)
                self.assertNotEqual(client.private_completion_state[key].data_ptr(), server.best_private_states[client.id][key].data_ptr())
        self.assertEqual(server.evaluate(), best_metrics)
        self.assertEqual(server.last_diagnostics, best_diagnostics)


if __name__ == "__main__":
    unittest.main()
