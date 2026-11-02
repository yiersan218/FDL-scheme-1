"""验证执行源码完整进入实验快照，通信编解码可独立于训练层导入。"""

import json
import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "system"))

from run_incomplete import SOURCE_FILES  # noqa: E402


class ModuleLayoutTests(unittest.TestCase):
    """防止拆分模块遗漏源码指纹或把模型训练依赖重新引入编解码层。"""

    def test_source_snapshot_covers_every_active_federated_module(self):
        """新增的联邦核心模块必须进入快照，已移除的压缩路径不能继续执行或记录。"""
        active = {
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in (PROJECT_ROOT / "system/flcore").rglob("*.py")
        }
        sources = set(SOURCE_FILES)
        self.assertTrue(active <= sources, f"Missing source snapshots: {sorted(active - sources)}")
        self.assertEqual(len(SOURCE_FILES), len(sources), "Duplicate source manifest entries")
        for source in sources:
            self.assertTrue((PROJECT_ROOT / source).is_file(), source)
        legacy = "system/flcore/compression.py"
        self.assertNotIn(legacy, sources)
        self.assertFalse((PROJECT_ROOT / legacy).exists())

    def test_codec_import_does_not_load_models_losses_or_compression_strategy(self):
        """在独立进程中导入 codec，检查包括包入口在内的传递依赖。"""
        # 独立进程避免其他测试已经导入模型，使本检查受到全局模块缓存影响。
        code = "\n".join([
            "import importlib, json, sys",
            "sys.path.insert(0, 'system')",
            "importlib.import_module('flcore.communication.codec')",
            "blocked = ('flcore.trainmodel', 'flcore.clients', 'flcore.servers', "
            "'flcore.communication.compression')",
            "loaded = [name for name in sys.modules "
            "if any(name == prefix or name.startswith(prefix + '.') for prefix in blocked)]",
            "print(json.dumps(sorted(loaded)))",
        ])
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
