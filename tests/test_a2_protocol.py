import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

import run_a2_matrix as protocol  # noqa: E402


FROZEN_PATH = (
    PROJECT_ROOT / "results-优化2" / "protocol-v2" / "a2"
    / "frozen_selection.json"
)
RESULTS_ROOT = PROJECT_ROOT / "results-性能统计"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class A2ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = protocol.load_frozen(FROZEN_PATH)

    def test_frozen_selection_contains_only_per_view_a2_configs(self):
        self.assertEqual(self.frozen["protocol_version"], "a2-only-prototype-v1")
        self.assertEqual(self.frozen["candidate"]["variant"], "A2")
        self.assertEqual(self.frozen["candidate"]["prototype_mode"], "per_view")
        for dataset in protocol.DATASETS:
            scenarios = self.frozen["datasets"][dataset]["scenarios"]
            self.assertEqual(set(scenarios), {"full", "missing_0p3"})
            for entry in scenarios.values():
                path = protocol.frozen_config_path(FROZEN_PATH, entry)
                config = protocol.read_json(path)
                self.assertEqual(config["model"]["prototype_mode"], "per_view")
                self.assertEqual(config["model"]["per_view_head_type"], "student_t")
                self.assertEqual(entry["config_sha256"], file_hash(path))

    def test_build_config_never_constructs_a_shared_or_gated_route(self):
        environment = {"sha256": "test-environment"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, full_dir = protocol.build_config(
                self.frozen, FROZEN_PATH, "Mfeat", 0.0, 7, "cpu", root,
                "test-source", environment,
            )
            missing, missing_dir = protocol.build_config(
                self.frozen, FROZEN_PATH, "Mfeat", 0.7, 7, "cpu", root,
                "test-source", environment,
            )
        for config in (full, missing):
            self.assertEqual(config["model"]["prototype_mode"], "per_view")
            self.assertEqual(config["experiment"]["method"], "A2")
            self.assertEqual(config["experiment"]["variant"], "A2")
        self.assertFalse(full["missing"]["enabled"])
        self.assertEqual(full["compression"]["method"], "none")
        self.assertTrue(missing["missing"]["enabled"])
        self.assertEqual(missing["missing"]["rate"], 0.7)
        self.assertEqual(missing["compression"]["method"], "stage")
        self.assertTrue(missing["compression"]["error_feedback"])
        self.assertIn("A2", full_dir.parts)
        self.assertIn("A2", missing_dir.parts)

    def test_existing_matrix_has_35_verified_a2_records(self):
        summary = protocol.read_json(RESULTS_ROOT / "summary.json")
        manifest = protocol.read_json(RESULTS_ROOT / "manifest.json")
        plan = protocol.read_json(RESULTS_ROOT / "matrix_plan.json")
        self.assertEqual(plan["method"], "A2")
        self.assertEqual(plan["prototype_mode"], "per_view")
        self.assertEqual(plan["expected_runs"], 35)
        self.assertEqual(len(summary), 35)
        self.assertEqual(manifest["completed_runs"], 35)
        self.assertTrue(manifest["complete"])
        self.assertEqual({row["method"] for row in summary}, {"A2"})
        self.assertEqual({row["prototype_mode"] for row in summary}, {"per_view"})
        result_directories = {
            path.name for path in RESULTS_ROOT.iterdir() if path.is_dir()
        }
        self.assertEqual(result_directories, {"A2"})
        for record in manifest["records"]:
            summary_path = PROJECT_ROOT / record["summary_path"]
            history_path = PROJECT_ROOT / record["history_path"]
            self.assertEqual(record["summary_sha256"], file_hash(summary_path))
            self.assertEqual(record["history_sha256"], file_hash(history_path))


if __name__ == "__main__":
    unittest.main()
