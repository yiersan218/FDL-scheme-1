"""验证 A2 配置、MAT 加载、无标签划分、两阶段损失及多视图原型聚合。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from scipy.io import savemat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from config import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    apply_overrides,
    load_config,
)
from flcore.clients.clientcluster import (  # noqa: E402
    FederatedClusteringClient,
)
from flcore.servers.servercluster import (  # noqa: E402
    FederatedMultiViewClusteringServer,
    pretraining_clustering_scale,
    progress_report_gap,
)
from flcore.trainmodel.multiview import (  # noqa: E402
    ClusteringHead,
    MultiViewClusteringModel,
)
from flcore.trainmodel.losses import clustering_objective  # noqa: E402
from utils.mat_data import balanced_client_indices, load_multiview_mat  # noqa: E402


class ClusteringProjectTests(unittest.TestCase):
    """覆盖当前 Student-t 聚类流程及配置、观测和联邦聚合约束。"""

    def test_progress_is_reported_at_ten_percent_intervals(self):
        """验证进度间隔约为总轮数的十分之一，且小规模训练至少每轮可报告。"""
        self.assertEqual(progress_report_gap(100), 10)
        self.assertEqual(progress_report_gap(16), 2)
        self.assertEqual(progress_report_gap(10), 1)
        self.assertEqual(progress_report_gap(2), 1)

    def test_all_dataset_configs_load(self):
        """验证七份正式配置均可加载，并满足 A2、两阶段和正损失权重约束。"""
        configs = [load_config(path) for path in DEFAULT_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 7)
        self.assertEqual(
            {config["dataset"]["name"] for config in configs},
            {"ALOI_100", "flower17", "LandUse_21", "Mfeat", "NUSWIDE", "Scene-15", "animal"},
        )
        for config in configs:
            self.assertEqual(config["model"]["cluster_head_type"], "student_t")
            self.assertEqual(config["model"]["student_t_distance_scale"], 1.0)
            self.assertEqual(config["model"]["prototype_mode"], "per_view")
            self.assertEqual(config["model"]["per_view_head_type"], "student_t")
            self.assertNotIn("cosine_temperature", config["model"])
            self.assertNotIn("per_view_temperature", config["model"])
            training = config["training"]
            self.assertGreaterEqual(training["center_init_round"], 0)
            self.assertLessEqual(
                training["center_init_round"],
                training["pretrain_rounds"],
            )
            self.assertGreater(training["pretraining_end_learning_rate"], 0)
            self.assertGreater(training["pretraining_local_epochs"], 0)
            self.assertGreater(training["clustering_local_epochs"], 0)
            self.assertGreater(training["cluster_head_learning_rate_multiplier"], 0)
            self.assertGreaterEqual(training["center_momentum"], 0)
            self.assertLess(training["center_momentum"], 1)
            for value in config["loss_weights"].values():
                self.assertGreater(float(value), 0)

    def test_animal_dataset_config_loads_by_name(self):
        """验证按名称加载 animal 时使用统一配置目录中的文件。"""
        config = load_config("animal")
        self.assertEqual(config["dataset"]["name"], "animal")
        self.assertEqual(Path(config["config_path"]), DEFAULT_CONFIG_DIR / "animal.json")

    def test_retired_prototype_modes_are_rejected(self):
        """验证配置覆盖和模型构造均拒绝已移除的共享或门控原型拓扑。"""
        config = load_config(DEFAULT_CONFIG_DIR / "Mfeat.json")
        for mode in ("shared", "support_gated"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                apply_overrides(config, [f'model.prototype_mode="{mode}"'])
            with self.subTest(model_mode=mode), self.assertRaises(ValueError):
                MultiViewClusteringModel([2], 2, [2], 2, prototype_mode=mode, completion={"enabled": False})

    def test_retired_cosine_heads_are_rejected(self):
        """验证配置和直接模型构造都拒绝余弦头，不会静默改为 Student-t。"""
        config = load_config("Mfeat")
        for key in ("cluster_head_type", "per_view_head_type"):
            with self.subTest(config_key=key), self.assertRaises(ValueError):
                apply_overrides(config, [f'model.{key}="cosine"'])
        for kwargs in (
            {"cluster_head_type": "cosine"},
            {"per_view_head_type": "cosine"},
            {"cluster_head_type": "cosine", "per_view_head_type": "student_t"},
        ):
            with self.subTest(model_kwargs=kwargs), self.assertRaises(ValueError):
                MultiViewClusteringModel([2], 2, [2], 2, **kwargs, completion={"enabled": False})
        with self.assertRaises(ValueError):
            ClusteringHead(2, 2, head_type="cosine")

    def test_server_constructs_a2_per_view_heads(self):
        """验证服务端依据正式配置创建尺度为 1 的每视图 Student-t 头。"""
        from types import SimpleNamespace

        config = load_config(DEFAULT_CONFIG_DIR / "Mfeat.json")
        data = SimpleNamespace(
            views=[np.zeros((2000, 2), dtype=np.float32)],
            view_dims=[2],
            num_samples=2000,
        )
        with patch("flcore.servers.servercluster.FederatedClusteringClient"):
            server = FederatedMultiViewClusteringServer(
                data, config, torch.device("cpu")
            )
        self.assertEqual(server.global_model.prototype_mode, "per_view")
        self.assertEqual(server.global_model.per_view_head_type, "student_t")
        self.assertTrue(all(
            head.distance_scale == 1.0 for head in server.global_model.prototype_heads
        ))

    def test_scene15_two_stage_schedule(self):
        """核对 Scene-15 当前冻结轮数、阶段学习率及中心动量。"""
        config = load_config(DEFAULT_CONFIG_DIR / "Scene-15.json")
        training = config["training"]
        self.assertEqual(training["rounds"], 30)
        self.assertEqual(training["pretrain_rounds"], 22)
        self.assertEqual(training["center_init_round"], 14)
        self.assertEqual(training["local_epochs"], 2)
        self.assertEqual(training["pretraining_local_epochs"], 2)
        self.assertEqual(training["clustering_local_epochs"], 3)
        self.assertEqual(training["pretraining_end_learning_rate"], 1e-4)
        self.assertEqual(training["clustering_learning_rate"], 1e-4)
        self.assertEqual(training["center_momentum"], 0.99875)
        self.assertEqual(config["loss_weights"]["clustering"], 0.05)
        self.assertEqual(config["loss_weights"]["balance"], 0.05)

    def test_mat_loader(self):
        """验证 LandUse_21 的实际 MAT 文件可正确解析样本数和视图维度。"""
        data = load_multiview_mat(PROJECT_ROOT / "dataset" / "LandUse_21.mat")
        self.assertEqual(data.num_samples, 2100)
        self.assertEqual(data.num_clusters, 21)
        self.assertEqual(data.view_dims, [20, 59, 40])

    def test_mat_loader_accepts_data_labels_and_transposed_views(self):
        """验证 data/labels 别名及特征维在前的视图矩阵可被正确识别。"""
        import tempfile

        views = np.empty((1, 2), dtype=object)
        views[0, 0] = np.arange(15, dtype=np.float64).reshape(3, 5)
        views[0, 1] = np.arange(10, dtype=np.float64).reshape(2, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.mat"
            savemat(path, {"data": views, "labels": np.array([[1, 1, 2, 2, 3]])})
            data = load_multiview_mat(path)
        self.assertEqual(data.num_samples, 5)
        self.assertEqual(data.num_clusters, 3)
        self.assertEqual(data.view_dims, [3, 2])

    def test_mat_loader_accepts_gt_label_alias(self):
        """验证 X/gt 格式与列向量标签可被统一加载。"""
        import tempfile

        views = np.empty((1, 1), dtype=object)
        views[0, 0] = np.arange(20, dtype=np.float64).reshape(4, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gt_alias.mat"
            savemat(path, {"X": views, "gt": np.array([[1], [1], [2], [2], [3]])})
            data = load_multiview_mat(path)
        self.assertEqual(data.num_samples, 5)
        self.assertEqual(data.num_clusters, 3)
        self.assertEqual(data.view_dims, [4])

    def test_balanced_partition_is_label_free(self):
        """验证仅凭样本索引生成互不重叠、数量差至多为 1 的客户端划分。"""
        parts = balanced_client_indices(103, 5, seed=42)
        merged = np.concatenate(parts)
        self.assertEqual(len(np.unique(merged)), 103)
        self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_model_and_unsupervised_loss(self):
        """验证多视图前向输出的分配维度正确，且无监督总损失有限。"""
        model = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={"enabled": False})
        views = (torch.randn(12, 8), torch.randn(12, 5))
        outputs = model(views)
        loss, metrics = clustering_objective(
            outputs,
            views,
            {"reconstruction": 1.0, "consistency": 0.2, "clustering": 0.5, "balance": 0.05},
            clustering_enabled=True,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(outputs["assignments"].shape, (12, 3))
        self.assertIn("clustering", metrics)

    def test_per_view_assignments_use_uniform_visible_view_average(self):
        """验证 Student-t 分配只等权平均真实可见视图，缺失视图保持零值。"""
        model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
            completion={"enabled": False},
        )
        with torch.no_grad():
            for head in model.prototype_heads:
                head.centers.copy_(torch.eye(2))
        embeddings = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1000.0, -1000.0], [1.0, 0.0]]),
        ]
        mask = torch.tensor([[True, False], [True, True]])
        view_assignments, fused, actual_mask = model._per_view_assignments(
            embeddings, mask
        )
        # 单位轴与两个中心的平方距离为 0、2；alpha=s=1 时概率为 3/4、1/4。
        self.assertTrue(torch.equal(actual_mask, mask))
        self.assertTrue(torch.equal(view_assignments[1][0], torch.zeros(2)))
        self.assertTrue(torch.allclose(fused[0], torch.tensor([0.75, 0.25])))
        self.assertTrue(torch.allclose(fused[1], torch.tensor([0.5, 0.5])))

    def test_per_view_forward_never_encodes_hidden_values(self):
        """验证缺失行不进入编码，并且隐藏值扰动不影响融合表示或分配。"""
        torch.manual_seed(29)
        model = MultiViewClusteringModel(
            [3, 3], 2, [4], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
            completion={"enabled": False},
        ).eval()
        first = torch.randn(4, 3)
        second = torch.randn(4, 3)
        changed = second.clone()
        # 缺失行填入极值，让错误读取隐藏数据造成可检测的前向差异。
        changed[0] = torch.tensor([1e6, -1e6, 1e6])
        mask = torch.tensor([
            [1, 0], [1, 1], [0, 1], [1, 1],
        ], dtype=torch.bool)
        original = model((first, second), mask=mask)
        perturbed = model((first, changed), mask=mask)
        self.assertTrue(torch.equal(
            original["view_embeddings"][1][0], torch.zeros(2)
        ))
        self.assertTrue(torch.allclose(
            original["assignments"], perturbed["assignments"]
        ))
        self.assertTrue(torch.allclose(original["embedding"], perturbed["embedding"]))

    def test_per_view_student_t_preserves_center_scale(self):
        """验证 Student-t 前向保留非单位中心，且输出概率逐行归一化。"""
        model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view",
            per_view_head_type="student_t",
            student_t_distance_scale=2.0,
            completion={"enabled": False},
        )
        original = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        with torch.no_grad():
            for head in model.prototype_heads:
                head.centers.copy_(original)
        outputs = model((torch.randn(5, 2), torch.randn(5, 2)))
        self.assertTrue(all(head.head_type == "student_t" for head in model.prototype_heads))
        self.assertTrue(all(torch.equal(head.centers, original) for head in model.prototype_heads))
        self.assertEqual(outputs["assignments"].shape, (5, 2))
        self.assertTrue(torch.allclose(
            outputs["assignments"].sum(dim=1), torch.ones(5), atol=1e-6
        ))

    def test_per_view_semantic_term_stays_inside_clustering_loss(self):
        """验证可选语义项按权重计入聚类损失，且保留融合聚类项的原值。"""
        torch.manual_seed(31)
        model = MultiViewClusteringModel(
            [3, 3], 2, [], 2,
            prototype_mode="per_view",
            per_view_head_type="student_t",
            completion={"enabled": False},
        )
        views = (torch.randn(6, 3), torch.randn(6, 3))
        mask = torch.tensor([
            [1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [0, 1],
        ], dtype=torch.bool)
        outputs = model(views, mask=mask)
        target = torch.tensor([[0.9, 0.1]]).repeat(6, 1)
        base_weights = {
            "reconstruction": 1.0, "consistency": 0.2,
            "clustering": 1.0, "balance": 0.05,
            "view_semantic": 0.0,
        }
        base_loss, base_metrics = clustering_objective(
            outputs, views, base_weights, True, target_assignments=target, mask=mask
        )
        semantic_weights = dict(base_weights, view_semantic=2.0)
        semantic_loss, semantic_metrics = clustering_objective(
            outputs, views, semantic_weights, True,
            target_assignments=target, mask=mask,
        )
        self.assertGreater(semantic_metrics["view_semantic"], 0.0)
        self.assertAlmostEqual(
            float(semantic_loss - base_loss),
            2.0 * semantic_metrics["view_semantic"],
            places=5,
        )
        self.assertAlmostEqual(
            semantic_metrics["fused_clustering"], base_metrics["clustering"], places=6
        )

    def test_student_t_scale_one_matches_original_formula(self):
        """验证距离尺度为 1 时软分配与 Student-t 定义完全一致。"""
        head = ClusteringHead(3, 4, alpha=1.5, distance_scale=1.0)
        embedding = torch.randn(7, 4)
        distance = torch.sum(
            (embedding.unsqueeze(1) - head.centers.unsqueeze(0)) ** 2,
            dim=2,
        )
        expected = (1.0 + distance / 1.5).pow(-1.25)
        expected = expected / expected.sum(dim=1, keepdim=True)
        self.assertTrue(torch.allclose(head(embedding), expected))

    def test_student_t_scale_controls_assignment_sharpness(self):
        """验证固定中心和样本时，增大距离尺度提高该样例的匹配概率。"""
        low = ClusteringHead(2, 2, distance_scale=1.0)
        high = ClusteringHead(2, 2, distance_scale=10.0)
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with torch.no_grad():
            low.centers.copy_(centers)
            high.centers.copy_(centers)
        sample = torch.tensor([[1.0, 0.0]])
        self.assertGreater(high(sample)[0, 0], low(sample)[0, 0])

    def test_per_view_client_initialization_uses_one_shared_pseudo_label_order(self):
        """验证客户端使用共同伪标签为各视图生成一致的簇编号与计数。"""
        client = object.__new__(FederatedClusteringClient)
        client.id = 0
        client.device = torch.device("cpu")
        client.config = {
            "dataset": {"num_clusters": 2},
            "training": {
                "center_init_n_init": 5, "seed": 13,
            },
        }
        points = torch.tensor([
            [1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1],
        ])
        client._loader = lambda _shuffle: [(
            (points, points.clone()),
            torch.tensor([0, 0, 1, 1]),
            torch.arange(4),
        )]
        model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
            completion={"enabled": False},
        )
        with torch.no_grad():
            for view_model in model.view_models:
                layer = view_model.encoder[0]
                layer.weight.copy_(torch.eye(2))
                layer.bias.zero_()
        summary = client.cluster_summary(model)
        self.assertEqual(summary["mode"], "per_view")
        self.assertEqual(summary["centers"].shape, (2, 2, 2))
        self.assertTrue(np.allclose(summary["centers"][0], summary["centers"][1]))
        self.assertTrue(np.array_equal(summary["counts"][0], summary["counts"][1]))
        self.assertTrue(np.array_equal(summary["coverage"], np.array([4.0, 4.0])))
        # 归一化输入的簇内均值不是单位向量，Student-t 初始化应保留其尺度。
        expected_axis = 1.0 / np.sqrt(1.01)
        self.assertTrue(np.allclose(
            np.sort(summary["centers"][0, :, 0]), [-expected_axis, expected_axis]
        ))
        self.assertTrue(np.allclose(summary["centers"][0, :, 1], 0.0))
        self.assertTrue(np.array_equal(summary["counts"], np.full((2, 2), 2.0)))

    def test_per_view_server_initialization_applies_one_mapping_to_every_view(self):
        """验证服务端初始化时对客户端全部视图应用同一簇排列。"""
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.config = {"dataset": {"num_clusters": 2}}
        server.training = {"center_init_n_init": 5}
        server.seed = 17
        server.device = torch.device("cpu")
        server.global_model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
            completion={"enabled": False},
        )
        first = np.array([[2.0, 0.0], [-2.0, 0.0]], dtype=np.float32)
        second = np.array([[0.0, 3.0], [0.0, -3.0]], dtype=np.float32)
        summaries = [
            {
                "mode": "per_view", "centers": np.stack([first, second]),
                "counts": np.full((2, 2), 3.0), "coverage": np.array([6.0, 6.0]),
            },
            {
                "mode": "per_view", "centers": np.stack([first[::-1], second[::-1]]),
                "counts": np.full((2, 2), 2.0), "coverage": np.array([4.0, 4.0]),
            },
        ]
        server._initialize_per_view_centers(summaries)
        view0 = server.global_model.prototype_heads[0].centers.detach().numpy()
        view1 = server.global_model.prototype_heads[1].centers.detach().numpy()
        self.assertTrue(np.array_equal(np.sign(view0[:, 0]), np.sign(view1[:, 1])))
        self.assertTrue(np.allclose(np.sort(view0[:, 0]), [-2.0, 2.0]))
        self.assertTrue(np.allclose(np.sort(view1[:, 1]), [-3.0, 3.0]))
        self.assertTrue(np.allclose(view0[:, 1], 0.0))
        self.assertTrue(np.allclose(view1[:, 0], 0.0))

    def test_per_view_upload_alignment_uses_reference_permutation_for_all_heads(self):
        """验证上传对齐依据参考视图统一重排所有中心及对应簇计数。"""
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
            completion={"enabled": False},
        )
        server.prototype_reference_view = 0
        reference0 = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        reference1 = torch.tensor([[4.0, 0.5], [-0.25, 5.0]])
        state = {
            key: value.detach().cpu().clone()
            for key, value in server.global_model.state_dict().items()
        }
        state["prototype_heads.0.centers"] = reference0.flip(0)
        state["prototype_heads.1.centers"] = reference1.flip(0)
        counts = torch.tensor([[3.0, 7.0], [4.0, 6.0]])
        aligned_counts = server._align_per_view_prototypes(
            state, (reference0, reference1), counts
        )
        self.assertTrue(torch.equal(state["prototype_heads.0.centers"], reference0))
        self.assertTrue(torch.equal(state["prototype_heads.1.centers"], reference1))
        self.assertTrue(torch.equal(
            aligned_counts, torch.tensor([[7.0, 3.0], [6.0, 4.0]])
        ))

    def test_server_center_initialization_uses_count_weighted_kmeans(self):
        """验证服务端以每簇计数作为普通 KMeans 的样本权重完成初始化。"""
        summaries = [
            {
                "mode": "per_view",
                "centers": np.array([[[1.0, 0.1], [-1.0, 0.1]]], dtype=np.float32),
                "counts": np.array([[6.0, 4.0]]),
                "coverage": np.array([10.0]),
            },
            {
                "mode": "per_view",
                "centers": np.array([[[1.0, -0.1], [-1.0, -0.1]]], dtype=np.float32),
                "counts": np.array([[5.0, 5.0]]),
                "coverage": np.array([10.0]),
            },
        ]
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.clients = []
        for summary in summaries:
            client = MagicMock()
            client.cluster_summary.return_value = summary
            server.clients.append(client)
        server.config = {"dataset": {"num_clusters": 2}}
        server.training = {"center_init_n_init": 4}
        server.seed = 23
        server.device = torch.device("cpu")
        server.global_model = MultiViewClusteringModel([2], 2, [2], 2, completion={"enabled": False})
        fitted = MagicMock()
        fitted.fit.return_value = fitted
        fitted.cluster_centers_ = np.array([[2.0, 0.0], [0.0, 3.0]])
        with patch("flcore.servers.servercluster.KMeans", return_value=fitted) as kmeans:
            server._initialize_centers()

        self.assertTrue(server.centers_initialized)
        kmeans.assert_called_once_with(n_clusters=2, n_init=4, random_state=23)
        fitted.fit.assert_called_once()
        fit_args, fit_kwargs = fitted.fit.call_args
        self.assertTrue(np.array_equal(
            fit_args[0], np.concatenate([s["centers"][0] for s in summaries])
        ))
        self.assertTrue(np.array_equal(fit_kwargs["sample_weight"], [6.0, 4.0, 5.0, 5.0]))

    def test_invalid_cluster_head_parameters_are_rejected(self):
        """验证未知头类型、无效 Student-t 自由度与距离尺度会被拒绝。"""
        for kwargs in (
            {"head_type": "unknown"},
            {"alpha": 0.0},
            {"distance_scale": 0.0},
            {"distance_scale": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ClusteringHead(2, 3, **kwargs)

    def test_pretraining_scales_clustering_terms(self):
        """验证聚类权重缩放为零时，总损失等价于关闭聚类项。"""
        model = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={"enabled": False})
        views = (torch.randn(12, 8), torch.randn(12, 5))
        outputs = model(views)
        weights = {
            "reconstruction": 1.0,
            "consistency": 0.2,
            "clustering": 0.5,
            "balance": 0.05,
        }
        disabled_loss, _ = clustering_objective(
            outputs,
            views,
            weights,
            clustering_enabled=False,
        )
        zero_scale_loss, metrics = clustering_objective(
            outputs,
            views,
            weights,
            clustering_enabled=True,
            clustering_weight_scale=0.0,
        )
        self.assertTrue(torch.allclose(disabled_loss, zero_scale_loss))
        self.assertEqual(metrics["clustering_weight_scale"], 0.0)
        self.assertGreaterEqual(metrics["clustering"], 0.0)

    def test_two_stage_pretraining_scale(self):
        """验证中心初始化前权重为零、预训练内线性增长、阶段二保持为一。"""
        self.assertEqual(pretraining_clustering_scale(1, 2, 4), 0.0)
        self.assertEqual(pretraining_clustering_scale(2, 2, 4), 0.0)
        self.assertEqual(pretraining_clustering_scale(3, 2, 4), 0.5)
        self.assertEqual(pretraining_clustering_scale(4, 2, 4), 1.0)
        self.assertEqual(pretraining_clustering_scale(5, 2, 4), 1.0)

    def test_base_overrides_refresh_inherited_two_stage_parameters(self):
        """验证覆盖基础参数时，仍处于继承状态的阶段学习率和 epoch 同步更新。"""
        import json
        import tempfile

        raw_config = json.loads(
            (DEFAULT_CONFIG_DIR / "ALOI_100.json").read_text(encoding="utf-8")
        )
        for key in (
            "pretraining_local_epochs",
            "clustering_local_epochs",
            "pretraining_end_learning_rate",
            "clustering_learning_rate",
        ):
            raw_config["training"].pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inherited.json"
            path.write_text(json.dumps(raw_config), encoding="utf-8")
            config = apply_overrides(
                load_config(path),
                ["training.local_epochs=3", "training.learning_rate=0.01"],
            )
        training = config["training"]
        self.assertEqual(training["pretraining_local_epochs"], 3)
        self.assertEqual(training["clustering_local_epochs"], 3)
        self.assertEqual(training["pretraining_end_learning_rate"], 0.001)
        self.assertEqual(training["clustering_learning_rate"], 0.01)

    def test_base_overrides_preserve_explicit_two_stage_parameters(self):
        """验证基础参数覆盖不会修改配置中显式指定的阶段参数。"""
        config = apply_overrides(
            load_config(DEFAULT_CONFIG_DIR / "Scene-15.json"),
            ["training.local_epochs=1", "training.learning_rate=0.01"],
        )
        training = config["training"]
        self.assertEqual(training["pretraining_local_epochs"], 2)
        self.assertEqual(training["clustering_local_epochs"], 3)
        self.assertEqual(training["pretraining_end_learning_rate"], 1e-4)
        self.assertEqual(training["clustering_learning_rate"], 1e-4)

    def test_explicit_two_stage_override_stops_inheriting(self):
        """验证阶段参数被单独覆盖后，不再随基础参数继续变化。"""
        import json
        import tempfile

        raw_config = json.loads(
            (DEFAULT_CONFIG_DIR / "ALOI_100.json").read_text(encoding="utf-8")
        )
        for key in ("pretraining_local_epochs", "clustering_local_epochs"):
            raw_config["training"].pop(key, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inherited.json"
            path.write_text(json.dumps(raw_config), encoding="utf-8")
            config = apply_overrides(
                load_config(path),
                ["training.local_epochs=1", "training.pretraining_local_epochs=3"],
            )
            config = apply_overrides(config, ["training.local_epochs=4"])
        training = config["training"]
        self.assertEqual(training["pretraining_local_epochs"], 3)
        self.assertEqual(training["clustering_local_epochs"], 4)

    def test_fixed_dec_target_is_accepted_and_shape_checked(self):
        """验证损失接受轮内固定 DEC 目标，并拒绝与分配维度不匹配的目标。"""
        model = MultiViewClusteringModel([8, 5], 3, [6], 4, completion={"enabled": False})
        views = (torch.randn(12, 8), torch.randn(12, 5))
        outputs = model(views)
        weights = {
            "reconstruction": 1.0,
            "consistency": 0.2,
            "clustering": 0.5,
            "balance": 0.05,
        }
        fixed_target = torch.full_like(outputs["assignments"], 1.0 / 3.0)
        loss, _metrics = clustering_objective(
            outputs,
            views,
            weights,
            clustering_enabled=True,
            target_assignments=fixed_target,
        )
        self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):
            clustering_objective(
                outputs,
                views,
                weights,
                clustering_enabled=True,
                target_assignments=fixed_target[:-1],
            )

    def test_cluster_centers_use_aligned_per_cluster_counts(self):
        """验证聚类中心按各簇计数加权，而非使用客户端总样本数加权。"""
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel([4], 2, [3], 2, completion={"enabled": False})
        server.device = torch.device("cpu")
        server.centers_initialized = True
        server.training = {"center_momentum": 0.0}
        with torch.no_grad():
            server.global_model.prototype_heads[0].centers.copy_(
                torch.tensor([[0.0, 0.0], [10.0, 10.0]])
            )

        states = []
        for centers in (
            torch.tensor([[2.0, 2.0], [8.0, 8.0]]),
            torch.tensor([[1.0, 1.0], [9.0, 9.0]]),
        ):
            state = {
                key: value.detach().cpu().clone()
                for key, value in server.global_model.state_dict().items()
            }
            state["prototype_heads.0.centers"] = centers
            states.append(state)
        updates = [
            {
                "num_samples": 10,
                "state_dict": states[0],
                "cluster_counts": torch.tensor([[9.0, 1.0]]),
            },
            {
                "num_samples": 10,
                "state_dict": states[1],
                "cluster_counts": torch.tensor([[1.0, 9.0]]),
            },
        ]

        server._aggregate(updates)

        # 两客户端总样本数相同，但各簇计数不同，不能退化为中心的简单平均。
        expected = torch.tensor([[1.9, 1.9], [8.9, 8.9]])
        self.assertTrue(torch.allclose(
            server.global_model.prototype_heads[0].centers, expected
        ))

    def test_student_t_centers_preserve_weighted_scale_after_aggregation(self):
        """验证 Student-t 中心按簇计数和动量融合，不额外归一化中心尺度。"""
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel([4], 2, [3], 2, completion={"enabled": False})
        server.device = torch.device("cpu")
        server.centers_initialized = True
        server.training = {"center_momentum": 0.5}
        with torch.no_grad():
            server.global_model.prototype_heads[0].centers.copy_(
                torch.tensor([[2.0, 0.0], [0.0, 4.0]])
            )

        updates = []
        for centers in (
            torch.tensor([[3.0, 0.0], [0.0, 5.0]]),
            torch.tensor([[5.0, 0.0], [0.0, 7.0]]),
        ):
            state = {
                key: value.detach().cpu().clone()
                for key, value in server.global_model.state_dict().items()
            }
            state["prototype_heads.0.centers"] = centers
            updates.append({
                "num_samples": 8,
                "state_dict": state,
                "cluster_counts": torch.tensor([[4.0, 4.0]]),
            })

        server._aggregate(updates)
        # 本轮加权中心为 (4, 0)、(0, 6)，与旧中心按 0.5 动量融合。
        expected = torch.tensor([[3.0, 0.0], [0.0, 5.0]])
        self.assertTrue(torch.allclose(server.global_model.prototype_heads[0].centers, expected))


if __name__ == "__main__":
    unittest.main()
