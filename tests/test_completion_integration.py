"""验证补全在客户端训练、两阶段服务端及私有参数通信中的完整接入。"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from config import load_config  # noqa: E402
from flcore.clients.clientcluster import FederatedClusteringClient  # noqa: E402
from flcore.communication.codec import decode_sparse, dense_uplink_bytes, ordinary_keys, reconstruct_state  # noqa: E402
from flcore.servers.servercluster import FederatedMultiViewClusteringServer  # noqa: E402
from flcore.trainmodel.multiview import MultiViewClusteringModel  # noqa: E402
from utils.mat_data import MultiViewData, fixed_missing_mask  # noqa: E402


def toy_config():
    """返回用于短程 CPU 集成测试的两客户端、三轮补全配置。"""
    config = load_config("Scene-15")
    config["dataset"].update(name="completion-toy", num_clusters=2, num_clients=2)
    config["model"].update(hidden_dims=[12], embedding_dim=8, dropout=0.0)
    config["training"].update(
        seed=11, batch_size=8, rounds=3, pretrain_rounds=2,
        center_init_round=1, pretraining_local_epochs=2, clustering_local_epochs=2,
        learning_rate=0.001, pretraining_end_learning_rate=0.001,
        clustering_learning_rate=0.001, center_init_n_init=1, num_workers=0,
    )
    config["missing"].update(enabled=True, rate=0.5)
    config["completion"].update(enabled=True, anchor_size=12, topk=4, gate=False)
    config["compression"].update(method="none", error_feedback=False)
    config["output"]["save_model"] = False
    return config


def toy_data():
    """生成固定随机种子的 24 个三视图样本，供客户端集成测试共用。"""
    rng = np.random.default_rng(17)
    labels = np.arange(24, dtype=np.int64) % 2
    views = [rng.normal(size=(24, dimension)).astype(np.float32) for dimension in (5, 4, 6)]
    return MultiViewData("completion-toy", views, labels)


def toy_model(config):
    """依据补全配置创建具有固定初始化的小型三视图 A2 模型。"""
    torch.manual_seed(23)
    return MultiViewClusteringModel(
        [5, 4, 6], 2, [12], 8, completion=config["completion"],
    )


class CompletionIntegrationTests(unittest.TestCase):
    """检查客户端缓存、真实统计、跨轮私有状态及服务端通信统计。"""

    @classmethod
    def setUpClass(cls):
        """暂存 PyTorch 线程数，并使用单线程降低小型测试的调度开销。"""
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """恢复测试前的 PyTorch 线程设置，避免影响其他测试组。"""
        torch.set_num_threads(cls.previous_threads)

    def client(self, data, config):
        """创建持有给定数据全部样本的 CPU 测试客户端。"""
        return FederatedClusteringClient(0, data, np.arange(data.num_samples), config, torch.device("cpu"))

    def test_epoch_banks_only_contain_real_complete_rows_and_refresh_embeddings(self):
        """验证锚点按 epoch 重建且仅含真实完整行，并保持为不上传的临时缓存。"""
        config, data = toy_config(), toy_data()
        client = self.client(data, config)
        model = toy_model(config)
        original = MultiViewClusteringModel.build_completion_bank
        banks, modes = [], []

        def capture(instance, views, mask, sample_ids, seed=0):
            """记录每次锚点构建的输入和结果，检查完整行筛选及训练模式恢复。"""
            modes.append(instance.training)
            self.assertTrue(mask.all())
            expected = client.dataset.indices[client.dataset.mask.all(dim=1)]
            self.assertTrue(torch.equal(sample_ids.cpu(), expected))
            bank = original(instance, views, mask, sample_ids, seed=seed)
            self.assertEqual(instance.training, modes[-1])
            banks.append(copy.deepcopy(bank))
            return bank

        with patch.object(MultiViewClusteringModel, "build_completion_bank", new=capture):
            update = client.train(model, False, 2, "pretraining")
        self.assertEqual(len(banks), 2)
        self.assertTrue(all(modes))
        self.assertTrue(torch.equal(banks[0]["sample_ids"], banks[1]["sample_ids"]))
        self.assertTrue(any(
            not torch.equal(left, right)
            for left, right in zip(banks[0]["view_embeddings"], banks[1]["view_embeddings"])
        ))
        self.assertTrue(all(not value.requires_grad for value in banks[0]["view_embeddings"]))
        self.assertFalse(any("bank" in key or "sample_ids" in key for key in update["state_dict"]))
        self.assertFalse(any(key.startswith("completion_completion_") for key in update["metrics"]))
        self.assertTrue(np.isfinite(update["metrics"]["loss"]))

    def test_hidden_raw_values_cannot_change_training_targets_or_true_counts(self):
        """验证隐藏原始值变化不影响本地训练，并保持固定目标缓存和真实簇计数。"""
        config, data = toy_config(), toy_data()
        changed = copy.deepcopy(data)
        mask = fixed_missing_mask(data.num_samples, 3, 0.5, 11, 0, data.name)
        for view_index, view in enumerate(changed.views):
            view[~mask[:, view_index]] = 1e8 * (view_index + 1)
        first, second = self.client(data, config), self.client(changed, config)
        model = toy_model(config)
        original_mask = first.dataset.mask.clone()
        with patch.object(first, "local_target_cache", wraps=first.local_target_cache) as target_cache:
            update = first.train(model, True, 2, "clustering")
        other = second.train(model, True, 2, "clustering")
        self.assertEqual(target_cache.call_count, 1)
        self.assertTrue(torch.equal(first.dataset.mask, original_mask))
        self.assertEqual(update["metrics"], other["metrics"])
        for key, value in update["state_dict"].items():
            self.assertTrue(torch.equal(value, other["state_dict"][key]), key)
        expected_coverage = original_mask.sum(dim=0)
        self.assertTrue(torch.allclose(update["cluster_counts"].sum(dim=1), expected_coverage.float()))
        self.assertTrue(torch.equal(update["cluster_counts"], other["cluster_counts"]))

    def test_completed_initialization_keeps_centers_and_counts_observed(self):
        """验证完成融合只提供共同初始化伪标签，中心及计数仍由真实观测编码产生。"""
        config, data = toy_config(), toy_data()
        client, model = self.client(data, config), toy_model(config)
        original = MultiViewClusteringModel.forward
        expected_fused = []
        pseudo_labels = np.arange(data.num_samples) % 2

        def completion_override(instance, views, mask=None, completion_context=None):
            """替换完成融合为已知样本分组，便于检查 KMeans 的实际输入来源。"""
            output = original(instance, views, mask=mask, completion_context=completion_context)
            ids = completion_context["sample_ids"]
            completed = output["embedding"].new_zeros((len(ids), 8))
            completed[torch.arange(len(ids)), ids % 2] = 1.0
            output["completed_embedding"] = completed
            expected_fused.append(completed.cpu().numpy())
            return output

        class FixedKMeans:
            """使用确定的共同伪标签，隔离聚类随机性对初始化边界检查的影响。"""

            def __init__(self, **kwargs):
                """接受 KMeans 构造参数，测试替身无需额外初始化状态。"""
                pass

            def fit(self, values):
                """核对输入来自完成融合，并返回预设的共同伪标签。"""
                np.testing.assert_array_equal(values, np.concatenate(expected_fused))
                self.labels_ = pseudo_labels
                return self

        with patch.object(MultiViewClusteringModel, "forward", new=completion_override), patch(
            "flcore.clients.clientcluster.KMeans", FixedKMeans
        ):
            summary = client.cluster_summary(model)
        mask = client.dataset.mask.numpy()
        model.eval()
        with torch.no_grad():
            encoded, _fused = model.encode(tuple(client.dataset.views), mask=client.dataset.mask)
        for view_index, embedding in enumerate(encoded):
            normalized = F.normalize(embedding, dim=1).numpy()
            expected_counts = np.bincount(pseudo_labels[mask[:, view_index]], minlength=2)
            np.testing.assert_array_equal(summary["counts"][view_index], expected_counts)
            for cluster in range(2):
                members = mask[:, view_index] & (pseudo_labels == cluster)
                np.testing.assert_allclose(
                    summary["centers"][view_index, cluster], normalized[members].mean(axis=0), atol=1e-6,
                )
        np.testing.assert_array_equal(summary["coverage"], mask.sum(axis=0))
        self.assertEqual(set(summary), {"mode", "centers", "counts", "coverage"})

    def test_stage_context_is_built_from_supplied_feedback_corrected_model(self):
        """验证客户端以压缩器提供的修正模型构建锚点和停止梯度教师。"""
        config, data = toy_config(), toy_data()
        config["compression"].update(method="stage", error_feedback=True, calibration_size=8)
        client, model = self.client(data, config), toy_model(config)
        inspected = []

        def fake_compress(local_model, global_model, compression, **kwargs):
            """模拟 EF 修正后的参考模型，并检查客户端回调实际使用该模型。"""
            full_model = copy.deepcopy(local_model)
            with torch.no_grad():
                full_model.view_models[0].encoder[-1].bias[0].add_(1.0)
            full_model.train()
            context = kwargs["completion_context_builder"](full_model)
            self.assertTrue(full_model.training)
            self.assertEqual(context["round_number"], 3)
            self.assertEqual(len(context["sample_ids"]), 8)
            expected_bank = client._build_completion_bank(full_model, 3)
            for actual, expected in zip(context["bank"]["view_embeddings"], expected_bank["view_embeddings"]):
                self.assertTrue(torch.equal(actual, expected))
            full_model.eval()
            with torch.no_grad():
                teacher, _ = full_model.encode(kwargs["views"], mask=kwargs["mask"])
            for actual, expected in zip(context["teacher_embeddings"], teacher):
                self.assertTrue(torch.allclose(actual, F.normalize(expected, dim=1)))
                self.assertFalse(actual.requires_grad)
            inspected.append(context)
            return {"residual": torch.zeros(1), "payload": b"", "centers": {}, "bytes": 0}

        with patch("flcore.clients.clientcluster.compress_client_update", side_effect=fake_compress):
            update = client.train(model, False, 2, "pretraining")
        self.assertEqual(len(inspected), 1)
        self.assertNotIn("completion_context", update)
        self.assertNotIn("bank", update)

    def test_small_server_crosses_both_stages_with_completed_predictions(self):
        """验证补全聚类跨越两个阶段，预测使用受限混合且统计仍来自观测。"""
        config, data = toy_config(), toy_data()
        with tempfile.TemporaryDirectory() as directory:
            config["output"]["directory"] = directory
            server = FederatedMultiViewClusteringServer(data, config, torch.device("cpu"))
            result = server.train()
        self.assertTrue(server.global_model.completion_enabled)
        self.assertEqual([record["phase"] for record in server.history], ["pretraining", "pretraining", "clustering"])
        self.assertEqual(result["completion"]["prediction"], "bounded_completed_assignments")
        self.assertEqual(result["completion"]["loss_type"], "mse")
        self.assertEqual(result["completion"]["prototype_statistics"], "observed_only")
        self.assertGreater(result["diagnostics"]["completion_assignment_beta"], 0.0)
        self.assertEqual(result["completion"]["bank_scope"], "client_local_ephemeral")
        self.assertTrue(np.isfinite(result["metrics"]["nmi"]))
        self.assertFalse(any("bank" in key or "sample_ids" in key for key in server.global_model.state_dict()))

    def test_local_completion_persists_across_rounds_and_clients_are_isolated(self):
        """验证私有补全参数跨轮恢复、客户端间隔离，并与最新共享编码器组合。"""
        config, data = toy_config(), toy_data()
        config["completion"].update(parameter_scope="local", alignment_mode="paired")
        model = toy_model(config)
        private_keys = set(model.local_completion_state_keys())
        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        clients = [
            FederatedClusteringClient(i, data, np.arange(i * 12, (i + 1) * 12), config, torch.device("cpu"))
            for i in range(2)
        ]
        # 各客户端从同一公共初始化复制私有状态，但缓存不能与公共参数或彼此共享存储。
        for client in clients:
            local = copy.deepcopy(model)
            client._load_private_completion(local)
            for key in private_keys:
                self.assertTrue(torch.equal(client.private_completion_state[key], initial[key]))
        for client in clients:
            update = client.train(model, False, 1, "pretraining")
            self.assertFalse(private_keys.intersection(update["state_dict"]))
            self.assertEqual(set(client.private_completion_state), private_keys)
            self.assertTrue(all(
                value.device.type == "cpu" and not value.requires_grad
                for value in client.private_completion_state.values()
            ))
        self.assertTrue(any(
            not torch.equal(clients[0].private_completion_state[key], clients[1].private_completion_state[key])
            for key in private_keys
        ))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[key]), key)

        resumed = copy.deepcopy(clients[0].private_completion_state)
        other = copy.deepcopy(clients[1].private_completion_state)
        encoder_key = next(key for key in initial if key.startswith("view_models."))
        # 跨轮只恢复私有补全参数，编码器必须使用最新全局值，不能恢复过期的完整本地模型。
        with torch.no_grad():
            model.state_dict()[encoder_key].add_(0.05)
        shared_value = model.state_dict()[encoder_key].clone()
        original = MultiViewClusteringModel.forward
        starts = []

        def capture(instance, views, mask=None, completion_context=None):
            """检查下一轮首次前向已恢复私有参数，且使用最新全局编码器。"""
            if not starts:
                starts.append(True)
                for key in private_keys:
                    self.assertTrue(torch.equal(instance.state_dict()[key].cpu(), resumed[key]))
                self.assertTrue(torch.equal(instance.state_dict()[encoder_key], shared_value))
            return original(instance, views, mask=mask, completion_context=completion_context)

        with patch.object(MultiViewClusteringModel, "forward", new=capture):
            clients[0].train(model, False, 2, "pretraining")
        self.assertTrue(starts)
        for key in private_keys:
            self.assertTrue(torch.equal(clients[1].private_completion_state[key], other[key]))
            self.assertTrue(torch.equal(model.state_dict()[key], initial[key]))

    def test_local_private_initialization_changes_only_common_pseudo_label_input(self):
        """验证初始化读取客户端私有模块，但每视图中心、计数和覆盖量仍只取观测值。"""
        config, data = toy_config(), toy_data()
        config["completion"].update(parameter_scope="local")
        client, model = self.client(data, config), toy_model(config)
        client._load_private_completion(copy.deepcopy(model))
        private_key = next(iter(model.local_completion_state_keys()))
        client.private_completion_state[private_key].add_(0.5)
        public_value = model.state_dict()[private_key].clone()
        expected_private = client.private_completion_state[private_key].clone()
        original = MultiViewClusteringModel.forward
        pseudo_labels = np.arange(data.num_samples) % 2
        completed_batches = []

        def capture(instance, views, mask=None, completion_context=None):
            """检查初始化前向加载私有参数，并保存用于共同伪标签的完成融合。"""
            self.assertTrue(torch.equal(instance.state_dict()[private_key].cpu(), expected_private))
            output = original(instance, views, mask=mask, completion_context=completion_context)
            completed_batches.append(output["completed_embedding"].cpu().numpy())
            return output

        class FixedKMeans:
            """使用确定的共同伪标签，隔离聚类随机性对初始化边界检查的影响。"""

            def __init__(self, **kwargs):
                """接受 KMeans 构造参数，测试替身无需额外初始化状态。"""
                pass

            def fit(self, values):
                """核对输入来自完成融合，并返回预设的共同伪标签。"""
                np.testing.assert_array_equal(values, np.concatenate(completed_batches))
                self.labels_ = pseudo_labels
                return self

        with patch.object(MultiViewClusteringModel, "forward", new=capture), patch(
            "flcore.clients.clientcluster.KMeans", FixedKMeans
        ):
            summary = client.cluster_summary(model)
        self.assertTrue(torch.equal(model.state_dict()[private_key], public_value))
        masks = client.dataset.mask.numpy()
        model.eval()
        with torch.no_grad():
            encoded, _ = model.encode(tuple(client.dataset.views), mask=client.dataset.mask)
        for view_index, embedding in enumerate(encoded):
            normalized = F.normalize(embedding, dim=1).numpy()
            np.testing.assert_array_equal(
                summary["counts"][view_index], np.bincount(pseudo_labels[masks[:, view_index]], minlength=2)
            )
            for cluster in range(2):
                members = masks[:, view_index] & (pseudo_labels == cluster)
                np.testing.assert_allclose(
                    summary["centers"][view_index, cluster], normalized[members].mean(axis=0), atol=1e-6
                )
        np.testing.assert_array_equal(summary["coverage"], masks.sum(axis=0))
        self.assertEqual(set(summary), {"mode", "centers", "counts", "coverage"})

    def test_local_dense_sparse_uploads_and_server_aggregation_exclude_private_parameters(self):
        """验证稠密、Top-k 和 Stage 上传、EF 坐标及聚合均排除私有补全参数。"""
        for method in ("none", "topk", "stage"):
            with self.subTest(method=method):
                config, data = toy_config(), toy_data()
                config["completion"].update(parameter_scope="local", alignment_mode="paired")
                config["compression"].update(
                    method=method, error_feedback=method != "none", calibration_size=8, candidates=2
                )
                server = FederatedMultiViewClusteringServer(data, config, torch.device("cpu"))
                model = server.global_model
                private_keys = set(model.local_completion_state_keys())
                public_values = {key: model.state_dict()[key].clone() for key in private_keys}
                updates = [client.train(model, False, 1, "pretraining") for client in server.clients]
                if method == "none":
                    for update in updates:
                        self.assertFalse(private_keys.intersection(update["state_dict"]))
                else:
                    self.assertFalse(private_keys.intersection(ordinary_keys(model)))
                    expected = sum(model.state_dict()[key].numel() for key in ordinary_keys(model))
                    for client, update in zip(server.clients, updates):
                        self.assertEqual(update["compression"]["parameters"], expected)
                        self.assertEqual(decode_sparse(update["payload"], expected).numel(), expected)
                        self.assertEqual(client.compression_residual.numel(), expected)
                        reconstructed = reconstruct_state(model, update["payload"], update["centers"])
                        self.assertFalse(private_keys.intersection(reconstructed))
                        self.assertNotIn("bank", update)
                        self.assertNotIn("sample_ids", update)
                server._aggregate(updates)
                for key, value in public_values.items():
                    self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_local_scope_server_counts_only_shared_downlink_parameters(self):
        """验证下行字节仅统计共享状态，并单独报告每客户端私有参数量。"""
        config, data = toy_config(), toy_data()
        config["completion"].update(parameter_scope="local", alignment_mode="paired")
        with tempfile.TemporaryDirectory() as directory:
            config["output"]["directory"] = directory
            server = FederatedMultiViewClusteringServer(data, config, torch.device("cpu"))
            model = server.global_model
            private_keys = set(model.local_completion_state_keys())
            public_values = {key: model.state_dict()[key].clone() for key in private_keys}
            state = model.state_dict()
            shared_bytes = sum(value.numel() * value.element_size() for key, value in state.items()
                               if key not in private_keys)
            local_parameters = sum(value.numel() for key, value in model.named_parameters() if key in private_keys)
            result = server.train()
        for record in server.history:
            self.assertEqual(record["communication"]["downlink_bytes"], shared_bytes * len(server.clients))
        self.assertEqual(result["runtime"]["local_completion_parameters_per_client"], local_parameters)
        self.assertEqual(result["runtime"]["federated_parameters"], result["runtime"]["parameters"] - local_parameters)
        self.assertEqual(result["completion"]["parameter_scope"], "local")
        for key, value in public_values.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)
        self.assertEqual(dense_uplink_bytes(model, False), shared_bytes)


if __name__ == "__main__":
    unittest.main()
