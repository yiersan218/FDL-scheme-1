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
    BACKUP_CONFIG_DIR,
    DEFAULT_CONFIG_DIR,
    INIT_CONFIG_DIR,
    apply_overrides,
    load_config,
)
from flcore.clients.clientcluster import (  # noqa: E402
    FederatedClusteringClient,
    spherical_kmeans,
)
from flcore.servers.servercluster import (  # noqa: E402
    FederatedMultiViewClusteringServer,
    pretraining_clustering_scale,
    progress_report_gap,
)
from flcore.trainmodel.multiview import (  # noqa: E402
    ClusteringHead,
    MultiViewClusteringModel,
    clustering_objective,
)
from utils.mat_data import balanced_client_indices, load_multiview_mat  # noqa: E402


class ClusteringProjectTests(unittest.TestCase):
    def test_progress_is_reported_at_ten_percent_intervals(self):
        self.assertEqual(progress_report_gap(100), 10)
        self.assertEqual(progress_report_gap(16), 2)
        self.assertEqual(progress_report_gap(10), 1)
        self.assertEqual(progress_report_gap(2), 1)

    def test_all_dataset_configs_load(self):
        configs = [load_config(path) for path in DEFAULT_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 6)
        self.assertEqual(
            {config["dataset"]["name"] for config in configs},
            {"ALOI_100", "flower17", "LandUse_21", "Mfeat", "NUSWIDE", "Scene-15"},
        )
        for config in configs:
            self.assertEqual(config["model"]["cluster_head_type"], "student_t")
            self.assertEqual(config["model"]["student_t_distance_scale"], 1.0)
            self.assertEqual(config["model"]["prototype_mode"], "per_view")
            self.assertEqual(config["model"]["per_view_head_type"], "student_t")
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

    def test_backup_dataset_config_loads(self):
        configs = [load_config(path) for path in BACKUP_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["dataset"]["name"], "animal")
        for value in configs[0]["loss_weights"].values():
            self.assertGreater(float(value), 0)

    def test_retired_prototype_modes_are_rejected(self):
        config = load_config(DEFAULT_CONFIG_DIR / "Mfeat.json")
        for mode in ("shared", "support_gated"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                apply_overrides(config, [f'model.prototype_mode="{mode}"'])
            with self.subTest(model_mode=mode), self.assertRaises(ValueError):
                MultiViewClusteringModel([2], 2, [2], 2, prototype_mode=mode)

    def test_server_constructs_a2_per_view_heads(self):
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

    def test_initial_configs_use_common_hyperparameters(self):
        configs = [load_config(path) for path in INIT_CONFIG_DIR.glob("*.json")]
        self.assertEqual(len(configs), 6)
        common_sections = []
        for config in configs:
            common_sections.append(
                {
                    "num_clients": config["dataset"]["num_clients"],
                    "normalization": config["dataset"]["normalization"],
                    "model": config["model"],
                    "training": config["training"],
                    "loss_weights": config["loss_weights"],
                    "output": config["output"],
                }
            )
        self.assertTrue(all(item == common_sections[0] for item in common_sections))

    def test_mat_loader(self):
        data = load_multiview_mat(PROJECT_ROOT / "dataset" / "LandUse_21.mat")
        self.assertEqual(data.num_samples, 2100)
        self.assertEqual(data.num_clusters, 21)
        self.assertEqual(data.view_dims, [20, 59, 40])

    def test_mat_loader_accepts_data_labels_and_transposed_views(self):
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
        parts = balanced_client_indices(103, 5, seed=42)
        merged = np.concatenate(parts)
        self.assertEqual(len(np.unique(merged)), 103)
        self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_model_and_unsupervised_loss(self):
        model = MultiViewClusteringModel([8, 5], 3, [6], 4)
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
        model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view",
            per_view_head_type="cosine",
            per_view_temperature=0.5,
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
        expected_first = model.prototype_heads[0](embeddings[0][0:1])[0]
        expected_second = (
            model.prototype_heads[0](embeddings[0][1:2])[0]
            + model.prototype_heads[1](embeddings[1][1:2])[0]
        ) / 2.0
        self.assertTrue(torch.equal(actual_mask, mask))
        self.assertTrue(torch.equal(view_assignments[1][0], torch.zeros(2)))
        self.assertTrue(torch.allclose(fused[0], expected_first))
        self.assertTrue(torch.allclose(fused[1], expected_second))

    def test_per_view_forward_never_encodes_hidden_values(self):
        torch.manual_seed(29)
        model = MultiViewClusteringModel(
            [3, 3], 2, [4], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
        ).eval()
        first = torch.randn(4, 3)
        second = torch.randn(4, 3)
        changed = second.clone()
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

    def test_per_view_student_t_is_supported_without_center_projection(self):
        model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view",
            per_view_head_type="student_t",
            student_t_distance_scale=2.0,
        )
        original = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        with torch.no_grad():
            for head in model.prototype_heads:
                head.centers.copy_(original)
        model.project_cluster_centers_()
        self.assertTrue(all(head.head_type == "student_t" for head in model.prototype_heads))
        self.assertTrue(torch.equal(model.prototype_heads[0].centers, original))
        outputs = model((torch.randn(5, 2), torch.randn(5, 2)))
        self.assertEqual(outputs["assignments"].shape, (5, 2))
        self.assertTrue(torch.allclose(
            outputs["assignments"].sum(dim=1), torch.ones(5), atol=1e-6
        ))

    def test_per_view_semantic_term_stays_inside_clustering_loss(self):
        torch.manual_seed(31)
        model = MultiViewClusteringModel(
            [3, 3], 2, [], 2,
            prototype_mode="per_view",
            per_view_head_type="cosine",
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
        low = ClusteringHead(2, 2, distance_scale=1.0)
        high = ClusteringHead(2, 2, distance_scale=10.0)
        centers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        with torch.no_grad():
            low.centers.copy_(centers)
            high.centers.copy_(centers)
        sample = torch.tensor([[1.0, 0.0]])
        self.assertGreater(high(sample)[0, 0], low(sample)[0, 0])

    def test_cosine_head_is_scale_invariant_and_projected(self):
        head = ClusteringHead(3, 4, head_type="cosine", cosine_temperature=0.1)
        embedding = torch.randn(6, 4, requires_grad=True)
        before = head(embedding)
        with torch.no_grad():
            head.centers.mul_(7.0)
        after = head(embedding * 3.0)
        self.assertTrue(torch.allclose(before, after, atol=1e-6))
        self.assertTrue(torch.allclose(after.sum(dim=1), torch.ones(6), atol=1e-6))
        after.square().mean().backward()
        self.assertTrue(torch.isfinite(embedding.grad).all())
        head.project_centers_()
        self.assertTrue(torch.allclose(head.centers.norm(dim=1), torch.ones(3), atol=1e-6))

    def test_spherical_kmeans_is_scale_invariant_and_reproducible(self):
        samples = np.array([
            [1.0, 0.05],
            [1.0, -0.03],
            [1.0, 0.01],
            [-1.0, 0.04],
            [-1.0, -0.02],
            [-1.0, 0.00],
        ])
        scales = np.array([0.1, 3.0, 100.0, 7.0, 0.5, 20.0])[:, None]
        centers, labels = spherical_kmeans(
            samples, 2, n_init=5, max_iter=50, random_state=19
        )
        scaled_centers, scaled_labels = spherical_kmeans(
            samples * scales, 2, n_init=5, max_iter=50, random_state=19
        )

        self.assertTrue(np.array_equal(labels, scaled_labels))
        self.assertTrue(np.allclose(centers, scaled_centers, atol=1e-6))
        self.assertTrue(np.allclose(np.linalg.norm(centers, axis=1), 1.0, atol=1e-6))
        self.assertEqual(len(np.unique(labels[:3])), 1)
        self.assertEqual(len(np.unique(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])

    def test_spherical_kmeans_uses_weights_and_handles_zero_vectors(self):
        centers, _labels = spherical_kmeans(
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            1,
            n_init=2,
            random_state=7,
            sample_weight=np.array([3.0, 1.0]),
        )
        expected = np.array([3.0, 1.0]) / np.sqrt(10.0)
        self.assertTrue(np.allclose(centers[0], expected, atol=1e-6))

        zero_centers, zero_labels = spherical_kmeans(
            np.zeros((4, 3)), 3, n_init=2, max_iter=5, random_state=7
        )
        self.assertTrue(np.isfinite(zero_centers).all())
        self.assertTrue(np.allclose(np.linalg.norm(zero_centers, axis=1), 1.0))
        self.assertTrue(np.logical_and(zero_labels >= 0, zero_labels < 3).all())

    def test_per_view_client_initialization_uses_one_shared_pseudo_label_order(self):
        client = object.__new__(FederatedClusteringClient)
        client.id = 0
        client.device = torch.device("cpu")
        client.config = {
            "dataset": {"num_clusters": 2},
            "training": {
                "center_init_n_init": 5, "center_init_max_iter": 50, "seed": 13,
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
            prototype_mode="per_view", per_view_head_type="cosine",
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

    def test_per_view_server_initialization_applies_one_mapping_to_every_view(self):
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.config = {"dataset": {"num_clusters": 2}}
        server.training = {"center_init_n_init": 5, "center_init_max_iter": 50}
        server.seed = 17
        server.device = torch.device("cpu")
        server.global_model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view", per_view_head_type="cosine",
        )
        first = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        second = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
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
        self.assertTrue(np.allclose(np.linalg.norm(view0, axis=1), 1.0))
        self.assertTrue(np.allclose(np.linalg.norm(view1, axis=1), 1.0))

    def test_per_view_upload_alignment_uses_reference_permutation_for_all_heads(self):
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel(
            [2, 2], 2, [], 2,
            prototype_mode="per_view", per_view_head_type="cosine",
        )
        server.prototype_reference_view = 0
        reference0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        reference1 = torch.tensor([[0.8, 0.6], [-0.6, 0.8]])
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

    def test_server_center_initialization_routes_only_cosine_to_spherical(self):
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
        server.training = {"center_init_n_init": 4, "center_init_max_iter": 25}
        server.seed = 23
        server.device = torch.device("cpu")
        server.global_model = MultiViewClusteringModel(
            [2], 2, [2], 2, per_view_head_type="cosine"
        )

        with patch(
            "flcore.servers.servercluster.KMeans",
            side_effect=AssertionError("cosine must not use Euclidean KMeans"),
        ):
            server._initialize_centers()

        norms = server.global_model.prototype_heads[0].centers.detach().norm(dim=1)
        self.assertTrue(server.centers_initialized)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

        server.global_model = MultiViewClusteringModel([2], 2, [2], 2)
        fitted = MagicMock()
        fitted.fit.return_value = fitted
        fitted.cluster_centers_ = np.array([[2.0, 0.0], [0.0, 3.0]])
        with patch("flcore.servers.servercluster.KMeans", return_value=fitted) as kmeans:
            with patch(
                "flcore.servers.servercluster.spherical_kmeans",
                side_effect=AssertionError("student_t must not use spherical KMeans"),
            ):
                server._initialize_centers()

        kmeans.assert_called_once_with(n_clusters=2, n_init=4, random_state=23)
        fitted.fit.assert_called_once()
        fit_args, fit_kwargs = fitted.fit.call_args
        self.assertTrue(np.array_equal(
            fit_args[0], np.concatenate([s["centers"][0] for s in summaries])
        ))
        self.assertTrue(np.array_equal(fit_kwargs["sample_weight"], [6.0, 4.0, 5.0, 5.0]))

    def test_invalid_cluster_head_parameters_are_rejected(self):
        for kwargs in (
            {"head_type": "unknown"},
            {"distance_scale": 0.0},
            {"distance_scale": float("nan")},
            {"cosine_temperature": 0.0},
        ):
            with self.assertRaises(ValueError):
                ClusteringHead(2, 3, **kwargs)

    def test_pretraining_scales_clustering_terms(self):
        model = MultiViewClusteringModel([8, 5], 3, [6], 4)
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
        self.assertEqual(pretraining_clustering_scale(1, 2, 4), 0.0)
        self.assertEqual(pretraining_clustering_scale(2, 2, 4), 0.0)
        self.assertEqual(pretraining_clustering_scale(3, 2, 4), 0.5)
        self.assertEqual(pretraining_clustering_scale(4, 2, 4), 1.0)
        self.assertEqual(pretraining_clustering_scale(5, 2, 4), 1.0)

    def test_base_overrides_refresh_inherited_two_stage_parameters(self):
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
        model = MultiViewClusteringModel([8, 5], 3, [6], 4)
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
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel([4], 2, [3], 2)
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

        expected = torch.tensor([[1.9, 1.9], [8.9, 8.9]])
        self.assertTrue(torch.allclose(
            server.global_model.prototype_heads[0].centers, expected
        ))

    def test_cosine_centers_remain_normalized_after_aggregation(self):
        server = object.__new__(FederatedMultiViewClusteringServer)
        server.global_model = MultiViewClusteringModel(
            [4], 2, [3], 2, per_view_head_type="cosine"
        )
        server.device = torch.device("cpu")
        server.centers_initialized = True
        server.training = {"center_momentum": 0.5}
        with torch.no_grad():
            server.global_model.prototype_heads[0].centers.copy_(torch.eye(2))

        updates = []
        for centers in (
            torch.tensor([[0.8, 0.6], [-0.6, 0.8]]),
            torch.tensor([[0.6, 0.8], [-0.8, 0.6]]),
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
        norms = server.global_model.prototype_heads[0].centers.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
