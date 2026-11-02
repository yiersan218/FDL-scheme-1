"""验证统一缺失补全配置、同条件消融、历史输出保护和新计划隔离。"""

import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from config import (ALGORITHM_VERSION, DEFAULT_CONFIG_DIR, apply_overrides,
                    ensure_writable_output, load_config)  # noqa: E402
import config as config_module  # noqa: E402
import main as entry  # noqa: E402
import run_incomplete as protocol  # noqa: E402


class IncompleteConfigTests(unittest.TestCase):
    """覆盖默认补全、历史损失别名迁移及与压缩解耦的缺失配置。"""

    def test_declared_algorithm_version_cannot_be_silently_replaced(self):
        """未标版本的旧配置可迁移，但显式声明其他算法的配置必须拒绝。"""
        raw = json.loads((DEFAULT_CONFIG_DIR / "Mfeat.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versioned.json"
            raw["algorithm_version"] = "archived-v3"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different algorithm_version"):
                load_config(path)
            raw["algorithm_version"] = ALGORITHM_VERSION
            path.write_text(json.dumps(raw), encoding="utf-8")
            self.assertEqual(load_config(path)["algorithm_version"], ALGORITHM_VERSION)

    def test_seven_null_configs_enable_formal_completion_at_zero_rate(self):
        """七份占位配置保留 null，加载后采用未调优补全权重和统一零率语义。"""
        paths = list(DEFAULT_CONFIG_DIR.glob("*.json"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            with self.subTest(dataset=path.stem):
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.assertTrue(all(raw[key] is None for key in ("missing", "completion", "compression")))
                self.assertEqual(raw["loss_weights"]["completion"], 1.0)
                self.assertNotIn("consistency", raw["loss_weights"])
                config = load_config(path)
                self.assertEqual(config["algorithm_version"], ALGORITHM_VERSION)
                self.assertEqual(config["missing"], {"enabled": True, "rate": 0.0})
                self.assertTrue(config["completion"]["enabled"])
                self.assertTrue(config["completion"]["use_completed_assignments"])
                self.assertEqual(config["completion"]["alignment_mode"], "mse")
                self.assertEqual(config["completion"]["parameter_scope"], "federated")
                self.assertEqual(config["completion"]["anchor_size"], 256)
                self.assertEqual(config["compression"]["method"], "none")

    def test_legacy_weight_alias_migrates_and_conflicts_are_explicit(self):
        """验证旧 JSON 和覆盖命令可迁移，互相冲突的两种字段名不能静默取值。"""
        raw = json.loads((DEFAULT_CONFIG_DIR / "Mfeat.json").read_text(encoding="utf-8"))
        raw["loss_weights"]["consistency"] = raw["loss_weights"].pop("completion")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = load_config(path)
            self.assertNotIn("consistency", config["loss_weights"])
            raw["loss_weights"]["completion"] = 2.0
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_config(path)
        changed = apply_overrides(config, ["loss_weights.consistency=0.3"])
        self.assertEqual(changed["loss_weights"]["completion"], 0.3)
        self.assertNotIn("consistency", changed["loss_weights"])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            apply_overrides(config, ["loss_weights.completion=0.4", "loss_weights.consistency=0.3"])
        same = apply_overrides(config, ["loss_weights.completion=0.3", "loss_weights.consistency=0.3"])
        self.assertEqual(same["loss_weights"]["completion"], 0.3)

    def test_positive_missing_rate_does_not_require_compression_or_research(self):
        """零率和正率均可用稠密、Top-k 或 Stage，补全默认独立开启。"""
        base = load_config("Mfeat")
        for rate in (0.0, 0.7):
            for method in ("none", "topk", "stage"):
                config = apply_overrides(base, [f"missing.rate={rate}", f"compression.method={method}"])
                self.assertTrue(config["completion"]["enabled"])
                self.assertFalse(config["research"]["enabled"])
                self.assertEqual(config["compression"]["method"], method)
        legacy = apply_overrides(base, ["missing.enabled=false", "missing.rate=0.7"])
        self.assertFalse(legacy["missing"]["enabled"])
        for mode in ("mse", "contrastive", "paired"):
            self.assertEqual(apply_overrides(base, [f"completion.alignment_mode={mode}"])
                             ["completion"]["alignment_mode"], mode)

    def test_main_uses_raw_loading_for_complete_and_missing_inputs(self):
        """验证所有率和旧关闭写法都加载原始特征，由客户端按观测值归一化。"""
        data = mock.Mock(num_clusters=10, name="Mfeat", num_samples=20, view_dims=[4, 5])
        for enabled, rate in ((True, 0.0), (True, 0.7), (False, 0.7)):
            config = load_config("Mfeat")
            config["missing"].update(enabled=enabled, rate=rate)
            with mock.patch.object(entry, "load_multiview_mat", return_value=data) as loader, \
                    mock.patch.object(entry, "FederatedMultiViewClusteringServer") as server, \
                    contextlib.redirect_stdout(io.StringIO()):
                server.return_value.train.return_value = {"metrics": {}}
                entry.run(config)
                self.assertEqual(loader.call_args.kwargs["normalization"], "none")


class OutputProtectionTests(unittest.TestCase):
    """核验归档目录、嵌套文件和路径别名在加载数据前被拒绝。"""

    def test_protected_paths_and_main_fail_before_data_loading(self):
        """保护目录本身、目录内文件、根报告和包含 .. 的等价路径。"""
        paths = ("results-性能统计", "results-优化2/extra/summary.json", "results-缺失2",
                 "性能统计.md", "性能统计2.md", "new/../results-缺失2/frozen_selection.json")
        for path in paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                ensure_writable_output(path)
        config = load_config("Mfeat")
        config["output"]["directory"] = "results-性能统计"
        with mock.patch.object(entry, "load_multiview_mat") as loader, self.assertRaises(ValueError):
            entry.run(config)
        loader.assert_not_called()
        self.assertEqual(ensure_writable_output(PROJECT_ROOT / "test-output"), PROJECT_ROOT / "test-output")

    def test_hardlink_identity_cannot_modify_archived_files(self):
        """用临时受保护文件模拟硬链接 inode，不依赖真实历史记录。"""
        actual_stat = Path.stat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            originals = (root / "性能统计.md", root / "results-缺失2/frozen_selection.json")
            for index, original in enumerate(originals):
                original.parent.mkdir(parents=True, exist_ok=True)
                original.write_text("仅用于保护规则测试的记录", encoding="utf-8")
                alias = root / f"alias-{index}.json"
                alias.write_text("测试输出", encoding="utf-8")
                link_stat = list(original.stat())
                link_stat[3] = 2
                def simulated_stat(path, *args, **kwargs):
                    """仅替换测试文件 inode，其余文件读取真实元数据。"""
                    return os.stat_result(link_stat) if path == alias else actual_stat(path, *args, **kwargs)
                with mock.patch.object(config_module, "PROJECT_ROOT", root), \
                        mock.patch.object(Path, "stat", simulated_stat):
                    with self.assertRaises(ValueError):
                        ensure_writable_output(alias)


class IncompleteProtocolTests(unittest.TestCase):
    """验证三组消融同条件计划、源码隔离与完整结果核验。"""

    def build(self, root, variant="completed", rate=0.3, compression="none"):
        """构造轻量配置以检查协议，不启动模型训练。"""
        args = protocol.parse_args(["--datasets", "Mfeat", "--device", "cpu", "--compression", compression])
        config = apply_overrides(load_config("Mfeat"), ["training.rounds=3", "training.pretrain_rounds=1",
                                                        "training.center_init_round=1"])
        budget = {"pre_center": 10000, "post_center": 20000} if compression != "none" else None
        return protocol.build_config(config, "Mfeat", rate, 42, variant, args, root,
                                     [{"path": "test", "sha256": "source"}], "data", budget)

    def test_three_variants_share_preprocessing_initialization_and_absolute_budget(self):
        """三组只改变补全训练与分配开关，初始化和通信设置在所有率保持一致。"""
        for rate in (0.0, 0.7):
            with tempfile.TemporaryDirectory() as directory:
                configs = {variant: self.build(Path(directory), variant, rate, "stage_ef")[0]
                           for variant in protocol.VARIANTS}
            for variant, config in configs.items():
                self.assertEqual(config["missing"], {"enabled": True, "rate": rate})
                self.assertFalse(config["completion"]["use_completed_initialization"])
                self.assertEqual(config["completion"]["enabled"], variant != "baseline")
                self.assertEqual(config["completion"]["use_completed_assignments"], variant == "completed")
                self.assertEqual(config["compression"]["budget_bytes_pair"], {"pre_center": 10000, "post_center": 20000})
                self.assertEqual(config["experiment"]["normalization_scope"], "client_observed")
            for section in ("model", "training", "loss_weights", "compression", "dataset", "missing"):
                self.assertEqual(configs["baseline"][section], configs["completed"][section])

    def test_plan_only_saves_source_without_training_or_rewriting_plan(self):
        """生成小计划和源码快照，不启动训练；拒绝变更时保留既有计划原字节。"""
        with tempfile.TemporaryDirectory() as directory:
            args = ["--datasets", "Mfeat", "--rates", "0", "0.3", "--variants", *protocol.VARIANTS,
                    "--device", "cpu", "--plan-only", "--results-dir", directory]
            with mock.patch.object(protocol, "run") as training, contextlib.redirect_stdout(io.StringIO()):
                plan = protocol.main(args)
            training.assert_not_called()
            self.assertEqual(plan["expected_runs"], 6)
            root = Path(directory)
            self.assertFalse((root / "runs").exists())
            for source in plan["source_files"]:
                self.assertEqual(protocol.file_hash(root / "source_snapshot" / source["path"]), source["sha256"])
            before = protocol.file_hash(root / "plan.json")
            with self.assertRaisesRegex(RuntimeError, "plan differs"):
                protocol.main(args + ["--override", "loss_weights.completion=0.5"])
            self.assertEqual(before, protocol.file_hash(root / "plan.json"))

    def test_plan_rejects_paired_setting_overrides_and_protected_root(self):
        """共享覆盖不能绕开矩阵的掩码、预算、补全消融开关或版本隔离。"""
        for expression in ("missing.rate=0.5", "compression.method=stage", "completion.enabled=false",
                           "completion.use_completed_initialization=true", "algorithm_version=old"):
            args = protocol.parse_args(["--override", expression])
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                protocol.prepare_plan(args)
        with self.assertRaises(ValueError):
            protocol.prepare_plan(protocol.parse_args(["--results-dir", "results-缺失2"]))

    def test_result_validation_rejects_changed_config_and_fake_last(self):
        """复用结果须匹配完整配置和源码指纹，最佳轮及末轮都核对原始历史。"""
        with tempfile.TemporaryDirectory() as directory:
            config, result_dir = self.build(Path(directory))
            job = {"config": config, "result_dir": str(result_dir)}
            best = {"acc": 0.6, "nmi": 0.5, "ari": 0.4}
            last = {"acc": 0.59, "nmi": 0.49, "ari": 0.39}
            history = [{"round": 1, "phase": "pretraining", "clustering": best},
                       {"round": 2, "phase": "clustering", "clustering": best},
                       {"round": 3, "phase": "clustering", "clustering": last}]
            summary = {"algorithm_version": ALGORITHM_VERSION, "dataset": "Mfeat", "config": config,
                       "best_round": 2, "metrics": best, "last_metrics": last}
            protocol.write_json(result_dir / "summary.json", summary)
            protocol.write_json(result_dir / "history.json", history)
            self.assertEqual(protocol.read_result(job)["last"], last)
            wrong = copy.deepcopy(job)
            wrong["config"]["completion"]["topk"] += 1
            with self.assertRaises(RuntimeError):
                protocol.read_result(wrong)
            summary["last_metrics"] = best
            protocol.write_json(result_dir / "summary.json", summary)
            with self.assertRaisesRegex(RuntimeError, "metrics"):
                protocol.read_result(job)


if __name__ == "__main__":
    unittest.main()
