"""验证多种子统计的方差口径、完整性和原始结果一致性。"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from config import ALGORITHM_VERSION  # noqa: E402
from run_incomplete import fingerprint  # noqa: E402
from summarize_matrix import collect_statistics, write_statistics  # noqa: E402


class MatrixStatisticsTests(unittest.TestCase):
    """用已知数值的原始记录检验汇总，不启动模型训练。"""

    def make_plan(self, root, completed=5):
        """构造五种子的完整计划及指定数量的真实格式 JSON 记录。"""
        seeds = [42, 2024, 2025, 2026, 2027]
        jobs = []
        for index, seed in enumerate(seeds):
            directory = root / "runs" / str(seed) / "Toy"
            config = {
                "algorithm_version": ALGORITHM_VERSION,
                "dataset": {"name": "Toy"},
                "training": {"seed": seed, "rounds": 3, "selection_metric": "nmi"},
                "missing": {"enabled": True, "rate": 0.3},
                "output": {"directory": str(directory.parent)},
                "compression": {"method": "none", "budget_bytes_pair": None},
                "experiment": {"seed": seed, "nominal_missing_rate": 0.3,
                               "variant": "completed", "compression": "none"},
            }
            config["experiment"]["fingerprint"] = fingerprint(config)
            jobs.append({"config": config, "result_dir": str(directory)})
            if index >= completed:
                continue
            value = (index + 1) / 10
            best = {"acc": value, "nmi": value, "ari": value - 0.4}
            # 末轮 ACC 更高但 NMI 更低，检验统计不会分别取各指标最大值。
            last = {"acc": 1 - value, "nmi": 0.01, "ari": -0.1}
            history = [
                {"round": 1, "phase": "pretraining", "clustering": best},
                {"round": 2, "phase": "clustering", "clustering": best},
                {"round": 3, "phase": "clustering", "clustering": last},
            ]
            summary = {"algorithm_version": ALGORITHM_VERSION, "dataset": "Toy",
                       "config": config, "best_round": 2, "metrics": best, "last_metrics": last}
            directory.mkdir(parents=True)
            for name, data in (("summary.json", summary), ("history.json", history)):
                (directory / name).write_text(json.dumps(data), encoding="utf-8")
        plan = {"algorithm_version": ALGORITHM_VERSION, "expected_runs": 5,
                "datasets": ["Toy"], "rates": [0.3], "seeds": seeds,
                "variants": ["completed"], "compression": "none", "source_files": [],
                "jobs": jobs}
        self.save_plan(root, plan)
        return plan

    @staticmethod
    def save_plan(root, plan):
        """保存测试计划。"""
        (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def test_five_seed_statistics_keep_same_best_round_and_sample_variance(self):
        """均值0.3，样本方差0.025、总体方差0.02，末轮单独统计。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_plan(root)
            report = write_statistics(root)
            self.assertTrue(report["complete"])
            self.assertEqual(report["completed_runs"], 5)
            group = report["groups"][0]
            self.assertAlmostEqual(group["best"]["acc"]["mean"], 0.3)
            self.assertAlmostEqual(group["best"]["acc"]["variance"], 0.025)
            self.assertAlmostEqual(group["best"]["acc"]["population_variance"], 0.02)
            self.assertAlmostEqual(group["best"]["ari"]["mean"], -0.1)
            self.assertAlmostEqual(group["last"]["acc"]["mean"], 0.7)
            self.assertAlmostEqual(group["last"]["nmi"]["variance"], 0.0)
            details = json.loads((root / "run_details.json").read_text(encoding="utf-8"))
            self.assertEqual([row["best_round"] for row in details], [2] * 5)
            self.assertIn("样本方差", (root / "statistics.md").read_text(encoding="utf-8"))

    def test_partial_results_do_not_claim_five_runs_or_zero_sample_variance(self):
        """严格模式拒绝缺项；零次、一次结果都不生成假零样本方差。"""
        for count in (0, 1):
            with self.subTest(completed=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.make_plan(root, completed=count)
                with self.assertRaises(FileNotFoundError):
                    write_statistics(root)
                self.assertFalse((root / "statistics.json").exists())
                report = write_statistics(root, allow_partial=True)
                group = report["groups"][0]
                self.assertFalse(report["complete"])
                self.assertEqual(group["n"], count)
                self.assertEqual(group["expected_n"], 5)
                self.assertIsNone(group["best"]["acc"]["variance"])
                self.assertEqual(len(report["pending_result_directories"]), 5 - count)

    def test_plan_rejects_duplicate_seed_and_missing_matrix_cell(self):
        """重复计划任务或缺失某种子不能制造五次重复实验。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.make_plan(root, completed=0)
            plan = copy.deepcopy(original)
            plan["jobs"][-1] = copy.deepcopy(plan["jobs"][0])
            self.save_plan(root, plan)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collect_statistics(root, allow_partial=True)
            plan = copy.deepcopy(original)
            plan["jobs"].pop()
            self.save_plan(root, plan)
            with self.assertRaisesRegex(ValueError, "count"):
                collect_statistics(root, allow_partial=True)

    def test_same_group_rejects_changed_training_configuration(self):
        """相同数据集缺失率下不同轮次不能合并为跨种子均值。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.make_plan(root, completed=0)
            config = plan["jobs"][-1]["config"]
            config["training"]["rounds"] = 4
            config["experiment"]["fingerprint"] = fingerprint(config)
            self.save_plan(root, plan)
            with self.assertRaisesRegex(ValueError, "Different training settings"):
                collect_statistics(root, allow_partial=True)

    def test_tampered_or_nonfinite_metrics_are_rejected_before_writing(self):
        """原始历史与摘要不一致、非有限指标均不能进入统计。"""
        for value in (0.99, float("nan")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan = self.make_plan(root)
                summary_path = Path(plan["jobs"][0]["result_dir"]) / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["metrics"]["nmi"] = value
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "metrics"):
                    write_statistics(root)
                self.assertFalse((root / "statistics.json").exists())


if __name__ == "__main__":
    unittest.main()
