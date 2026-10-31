"""Memory provider tests: store format, distillation, recall, and the real
Hermes ``memory.provider`` discovery seam.

Run: python -m unittest discover -s tests -v
Needs a Hermes checkout for the discovery tests (skipped otherwise).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLUGIN_DIR = os.path.join(ROOT, "plugins", "context_engine", "ctxfold")
HERMES_SRC = os.environ.get("HERMES_AGENT_SRC", "H:/hermes-agent")

sys.path.insert(0, os.path.join(PLUGIN_DIR))

HOME = tempfile.mkdtemp(prefix="hermes-ctxfold-mem-")
os.environ["HERMES_HOME"] = HOME
shutil.copytree(PLUGIN_DIR, os.path.join(HOME, "plugins", "ctxfold"), dirs_exist_ok=True)
import atexit  # noqa: E402

atexit.register(lambda: shutil.rmtree(HOME, ignore_errors=True))

from worker import Worker, WorkerError  # noqa: E402
from store import MemoryStore, parse_topic_file  # noqa: E402


def _load_memory_module():
    spec = importlib.util.spec_from_file_location(
        "ctxfold_memory_plugin",
        os.path.join(PLUGIN_DIR, "memory_provider.py"),
        submodule_search_locations=[PLUGIN_DIR],
    )
    # memory_provider.py uses relative imports; load the package first.
    package = importlib.util.spec_from_file_location(
        "ctxfold_plugin", os.path.join(PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[PLUGIN_DIR],
    )
    module = importlib.util.module_from_spec(package)
    sys.modules["ctxfold_plugin"] = module
    package.loader.exec_module(module)
    return importlib.import_module("ctxfold_plugin.memory_provider")


TRANSCRIPT = [
    {"role": "user", "content": "把 handoff.enabled 打开，阈值设为 25000，用 /bin/jq.exe 解析 H:\\pi-work\\config.json。"},
    {"role": "assistant", "content": "构建失败：Permission denied dist/babele.exe——根因是旧进程占用，教训是先 Stop-Process 再重建。"},
    {"role": "assistant", "content": "决定采用 Markdown 做权威层，改用确定性索引做派生层。"},
]


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ctxfold-store-")
        self.worker = Worker()
        self.store = MemoryStore(self.dir, self.worker).open()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_flush_reopen(self):
        result = self.store.add("handoff.enabled 阈值 25000", topic="pi", kind="fact",
                                anchors=["handoff.enabled", "25000"])
        self.assertFalse(result["merged"])
        self.store.flush()

        with open(os.path.join(self.dir, "topics", "pi.md"), encoding="utf-8") as handle:
            raw = handle.read()
        self.assertIn("<!-- ctxfold:entry", raw)
        self.assertIn('anchors="handoff.enabled;25000"', raw)

        reopened = MemoryStore(self.dir, self.worker).open()
        self.assertEqual(len(reopened.entries()), 1)
        self.assertTrue(reopened.verify()["ok"])

    def test_same_fact_is_a_hit_not_a_duplicate(self):
        first = self.store.add("决定采用 Markdown 权威层", topic="pi", kind="decision")
        second = self.store.add("决定采用  Markdown   权威层", topic="pi", kind="decision")
        self.assertEqual(first["id"], second["id"])
        self.assertTrue(second["merged"])
        self.assertEqual(len(self.store.entries()), 1)
        self.assertEqual(self.store.entries()[0].hits, 1)

    def test_parser_survives_a_truncated_entry(self):
        raw = "prefix notes\n<!-- ctxfold:entry id=abc topic=t kind=fact created=1 hits=0 anchors=\"\" -->\nbody\n"
        self.assertEqual(parse_topic_file(raw), [])

    def test_recall_ranks_anchor_matches(self):
        self.store.add("构建失败 Permission denied dist/babele.exe，教训是先杀进程",
                       topic="babele", kind="lesson",
                       anchors=["dist/babele.exe"])
        self.store.add("配色方案改为深色主题", topic="ui", kind="decision")
        self.store.flush()
        picked = self.store.recall("dist/babele.exe 构建失败怎么办", k=4)
        self.assertTrue(picked)
        self.assertIn("babele", picked[0].text)

    def test_curate_archives_old_zero_hit_entries(self):
        result = self.store.add("old unused fact", topic="tmp", kind="fact")
        self.store.flush()
        entries = self.store.entries()
        entries[0].created = 1  # ancient
        report = self.store.curate(retention_days=30)
        self.assertEqual(report["archived"], 1)
        self.assertEqual(len(self.store.entries()), 0)


class ProviderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_memory_module()
        cls.dir = tempfile.mkdtemp(prefix="ctxfold-prov-")
        cls.provider = cls.module.CtxfoldMemoryProvider()
        cls.provider.initialize("sess-1", hermes_home=cls.dir, agent_context="primary",
                                cwd="H:/proj")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_contract_surface(self):
        from agent.memory_provider import MemoryProvider
        self.assertIsInstance(self.provider, MemoryProvider)
        self.assertEqual(self.provider.name, "ctxfold")
        self.assertTrue(self.provider.get_config_schema())
        self.assertTrue(self.provider.system_prompt_block())

    def test_on_pre_compress_archives_the_transcript(self):
        """The core promise: what compaction drops is distilled, not forgotten."""
        before = len(self.provider._store.entries())
        self.provider.on_pre_compress(TRANSCRIPT)
        after = self.provider._store.entries()
        self.assertGreater(len(after), before)
        kinds = {entry.kind for entry in after}
        self.assertTrue({"fact", "lesson", "decision"} & kinds)
        self.assertTrue(any("babele.exe" in entry.text or "babele.exe" in str(entry.anchors)
                            for entry in after))

    def test_prefetch_returns_relevant_memory(self):
        recalled = self.provider.prefetch("babele.exe 构建失败 教训")
        self.assertIn("召回", recalled)
        self.assertIn("babele", recalled)
        status = self.provider.recall_status()
        self.assertIsNotNone(status)
        self.assertGreater(status.count, 0)

    def test_prefetch_is_cached(self):
        first = self.provider.prefetch("handoff.enabled 阈值")
        second = self.provider.prefetch("handoff.enabled 阈值")
        self.assertEqual(first, second)

    def test_trivial_prompt_is_skipped(self):
        self.assertEqual(self.provider.prefetch("thanks"), "")

    def test_tools(self):
        raw = self.provider.handle_tool_call("ctxfold_recall", {"query": "babele.exe"})
        data = json.loads(raw)
        self.assertGreaterEqual(data["count"], 1)
        stats = json.loads(self.provider.handle_tool_call("ctxfold_memory", {"action": "stats"}))
        self.assertIn("entries", stats)

    def test_sync_turn_writes_memory(self):
        before = len(self.provider._store.entries())
        self.provider.sync_turn(
            "部署脚本卡在 dist 体积 42MB",
            "决定改用 UPX 压缩，教训是先在本地验证再发版",
        )
        self.assertGreater(len(self.provider._store.entries()), before)


class DiscoveryTest(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(HERMES_SRC), "Hermes checkout not found")
    def test_provider_loads_by_name(self):
        sys.path.insert(0, HERMES_SRC)
        from plugins.memory import load_memory_provider
        provider = load_memory_provider("ctxfold")
        self.assertIsNotNone(provider, "Hermes could not load the ctxfold memory provider")
        self.assertEqual(provider.name, "ctxfold")
        from agent.memory_provider import MemoryProvider
        self.assertIsInstance(provider, MemoryProvider)


class FailOpenTest(unittest.TestCase):
    def test_missing_worker_never_breaks_recall(self):
        module = _load_memory_module()
        provider = module.CtxfoldMemoryProvider()
        directory = tempfile.mkdtemp(prefix="ctxfold-fail-")
        # The plugin package loads `worker` under its own namespace, so its
        # WorkerError is a different class than the top-level import above.
        worker_error = sys.modules["ctxfold_plugin.worker"].WorkerError
        try:
            provider.initialize("s", hermes_home=directory)
            provider._worker = None

            def broken():
                raise worker_error("simulated missing worker")

            provider._worker_for_call = broken
            self.assertEqual(provider.prefetch("anything"), "")
            provider.sync_turn("a", "b")  # must not raise
            provider.on_pre_compress(TRANSCRIPT)  # must not raise
        finally:
            shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
