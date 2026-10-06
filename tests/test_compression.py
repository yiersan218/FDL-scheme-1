import copy
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from config import apply_overrides, load_config  # noqa: E402
from flcore.compression import (  # noqa: E402
    compress_client_update,
    decode_sparse,
    dense_uplink_bytes,
    encode_sparse,
    flatten_update,
    ordinary_keys,
    reconstruct_state,
    state_from_flat_update,
)
from flcore.trainmodel.multiview import MultiViewClusteringModel  # noqa: E402
from flcore.servers.servercluster import FederatedMultiViewClusteringServer  # noqa: E402


class CompressionTests(unittest.TestCase):
    def test_sparse_wire_roundtrip_and_integrity(self):
        update = torch.tensor([0.0, -2.5, 0.0, 4.0, 0.0])
        packet = encode_sparse(update, torch.tensor([3, 1]))
        self.assertEqual(len(packet), 12 + 2 * 8)
        self.assertTrue(torch.equal(decode_sparse(packet, 5), update))
        with self.assertRaises(ValueError):
            decode_sparse(packet[:-1], 5)
        with self.assertRaises(ValueError):
            decode_sparse(packet, 4)
        with self.assertRaises(ValueError):
            encode_sparse(update, torch.tensor([1, 1]))

    def test_stage_packet_budget_centers_and_residual(self):
        torch.manual_seed(2)
        global_model = MultiViewClusteringModel([8, 5], 3, [6], 4)
        local_model = copy.deepcopy(global_model)
        with torch.no_grad():
            for parameter in local_model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        views = (torch.randn(10, 8), torch.randn(10, 5))
        targets = torch.full((10, 3), 1 / 3)
        config = {
            "method": "stage", "budget_ratio": 0.5, "candidates": 6,
            "view_weight": 0.1, "pair_weight": 0.1,
            "loss_weights": {
                "reconstruction": 1.0, "consistency": 0.2,
                "clustering": 0.5, "balance": 0.05,
            },
        }
        packet = compress_client_update(
            local_model, global_model, config, views=views, targets=targets,
            centers_enabled=True, clustering_scale=0.5,
        )
        self.assertLessEqual(packet["bytes"], int(dense_uplink_bytes(global_model, True) * 0.5))
        self.assertGreater(packet["kept"], 0)
        self.assertLess(packet["kept"], packet["parameters"])
        recovered = reconstruct_state(global_model, packet["payload"], packet["centers"])
        for key in global_model.prototype_center_keys():
            self.assertTrue(torch.equal(recovered[key], local_model.state_dict()[key]))
        delta = flatten_update(local_model, global_model, ordinary_keys(global_model))
        restored_delta = decode_sparse(packet["payload"], delta.numel())
        self.assertTrue(torch.allclose(restored_delta + packet["residual"], delta))
        centers = {
            key: local_model.state_dict()[key]
            for key in global_model.prototype_center_keys()
        }
        dense_state = state_from_flat_update(global_model, delta, centers)
        for key in ordinary_keys(global_model):
            self.assertTrue(torch.allclose(dense_state[key], local_model.state_dict()[key]))

    def test_stage_error_feedback_preserves_effective_update(self):
        torch.manual_seed(3)
        global_model = MultiViewClusteringModel([8, 5], 3, [6], 4)
        local_model = copy.deepcopy(global_model)
        with torch.no_grad():
            for parameter in local_model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        residual = torch.full_like(flatten_update(local_model, global_model,
                                                 ordinary_keys(global_model)), 0.003)
        packet = compress_client_update(
            local_model, global_model,
            {"method": "stage", "budget_ratio": 0.5, "candidates": 4,
             "view_weight": 0.1, "pair_weight": 0.1,
             "loss_weights": {"reconstruction": 1.0, "consistency": 0.2,
                              "clustering": 0.5, "balance": 0.05}},
            views=(torch.randn(10, 8), torch.randn(10, 5)),
            centers_enabled=False, residual=residual,
        )
        effective = flatten_update(local_model, global_model, ordinary_keys(global_model)) + residual
        sent = decode_sparse(packet["payload"], effective.numel())
        self.assertTrue(torch.allclose(sent + packet["residual"], effective))

    def test_per_view_prototypes_use_fixed_center_payload_not_sparse_parameters(self):
        torch.manual_seed(5)
        global_model = MultiViewClusteringModel(
            [4, 3], 2, [3], 2,
            prototype_mode="per_view", per_view_head_type="student_t",
        )
        local_model = copy.deepcopy(global_model)
        with torch.no_grad():
            for parameter in local_model.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.01)
        center_names = set(global_model.prototype_center_keys())
        self.assertTrue(center_names.isdisjoint(ordinary_keys(global_model)))
        packet = compress_client_update(
            local_model, global_model,
            {"method": "topk", "budget_ratio": 0.8},
            centers_enabled=True,
        )
        self.assertIsInstance(packet["centers"], dict)
        self.assertEqual(set(packet["centers"]), center_names)
        recovered = reconstruct_state(
            global_model, packet["payload"], packet["centers"]
        )
        for key in center_names:
            self.assertTrue(torch.equal(recovered[key], local_model.state_dict()[key]))
        self.assertLessEqual(
            packet["bytes"], int(dense_uplink_bytes(global_model, True) * 0.8)
        )

    def test_compression_overrides_are_validated(self):
        config = load_config("Mfeat")
        tuned = apply_overrides(config, ["compression.method=stage", "compression.budget_ratio=0.4"])
        self.assertEqual(tuned["compression"]["method"], "stage")
        with self.assertRaises(ValueError):
            apply_overrides(config, ["compression.budget_ratio=0"])

    def test_server_decoding_preserves_dense_aggregation_and_center_alignment(self):
        torch.manual_seed(7)
        reference = MultiViewClusteringModel([4], 2, [3], 2)
        with torch.no_grad():
            reference.prototype_heads[0].centers.copy_(
                torch.tensor([[0.0, 0.0], [10.0, 10.0]])
            )
        local_models = []
        for amount, centers in (
            (0.01, torch.tensor([[2.0, 2.0], [8.0, 8.0]])),
            (-0.02, torch.tensor([[9.0, 9.0], [1.0, 1.0]])),
        ):
            local = copy.deepcopy(reference)
            with torch.no_grad():
                for name, parameter in local.named_parameters():
                    if name != "prototype_heads.0.centers":
                        parameter.add_(amount)
                local.prototype_heads[0].centers.copy_(centers)
            local_models.append(local)
        dense_updates, sparse_updates = [], []
        for index, local in enumerate(local_models):
            state = {key: value.detach().cpu().clone() for key, value in local.state_dict().items()}
            counts = torch.tensor([[9.0, 1.0]])
            dense_updates.append({"state_dict": state, "num_samples": 10, "cluster_counts": counts.clone()})
            delta = flatten_update(local, reference, ordinary_keys(reference))
            sparse_updates.append({
                "payload": encode_sparse(delta, torch.arange(delta.numel())),
                "centers": {"prototype_heads.0.centers": state["prototype_heads.0.centers"]},
                "num_samples": 10,
                "cluster_counts": counts.clone(),
            })
        servers = []
        for _ in range(2):
            server = object.__new__(FederatedMultiViewClusteringServer)
            server.global_model = copy.deepcopy(reference)
            server.device = torch.device("cpu")
            server.centers_initialized = True
            server.training = {"center_momentum": 0.0}
            servers.append(server)
        servers[0]._aggregate(dense_updates)
        servers[1]._aggregate(sparse_updates)
        for key in reference.state_dict():
            self.assertTrue(torch.allclose(
                servers[0].global_model.state_dict()[key],
                servers[1].global_model.state_dict()[key], atol=1e-6, rtol=1e-6,
            ), key)


if __name__ == "__main__":
    unittest.main()
