"""Integration tests against the REAL Hermes context-engine seam.

Loads the plugin exactly the way Hermes does (``$HERMES_HOME/plugins/<name>/``
→ ``plugins.context_engine.load_context_engine``), then exercises the
``ContextEngine`` contract with a realistic transcript.

Needs a Hermes checkout (``HERMES_AGENT_SRC``, default ``H:/hermes-agent``);
tests skip when it is absent.
"""

from __future__ import annotations

import atexit
import copy
import importlib.util
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLUGIN_DIR = os.path.join(ROOT, "plugins", "context_engine", "ctxfold")
HERMES_SRC = os.environ.get("HERMES_AGENT_SRC", "H:/hermes-agent")

if not os.path.isdir(HERMES_SRC):
    raise unittest.SkipTest(f"Hermes checkout not found at {HERMES_SRC}")

# Hermes reads HERMES_HOME per call, but several modules snapshot profile state
# at import time, so the throwaway home is established BEFORE Hermes is imported.
HOME = tempfile.mkdtemp(prefix="hermes-ctxfold-")
os.environ["HERMES_HOME"] = HOME
shutil.copytree(PLUGIN_DIR, os.path.join(HOME, "plugins", "ctxfold"))
atexit.register(shutil.rmtree, HOME, True)
sys.path.insert(0, HERMES_SRC)


def _load_by_discovery():
    """What Hermes does when ``context.engine: ctxfold`` is configured."""
    from plugins.context_engine import load_context_engine
    return load_context_engine("ctxfold")


def _load_directly():
    """Fallback loader for contract tests that do not need discovery."""
    spec = importlib.util.spec_from_file_location(
        "ctxfold_plugin",
        os.path.join(PLUGIN_DIR, "__init__.py"),
        submodule_search_locations=[PLUGIN_DIR],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ctxfold_plugin"] = module
    spec.loader.exec_module(module)
    return module


def _transcript() -> list[dict]:
    messages = [
        {"role": "system", "content": "You are Hermes. Workspace H:/proj. Never touch C:/Windows."},
        {"role": "user", "content": "把 config.compaction.threshold 调到 0.35，再跑 gh pr list。"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "run", "arguments": "{\"cmd\":\"gh pr list\"}"},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "PR #12  fix(dist): shrink build\nPR #13  chore: bump deps"},
    ]
    for i in range(40):
        messages.append({"role": "assistant", "content": f"step {i}: patched src/app{i}.cpp, tests green."})
        messages.append({
            "role": "tool", "tool_call_id": f"pipe_{i}",
            "content": "\n".join(
                f"row {j}: /v1/items/{i}/{j}.json 200 elapsed={j}ms"
                for j in range(120)
            ),
        })
    messages.append({"role": "user", "content": "继续，然后把 dist/app.exe 从 42MB 压到 20MB。"})
    return messages


class Discovery(unittest.TestCase):
    def test_engine_loads_by_name(self):
        engine = _load_by_discovery()
        self.assertIsNotNone(engine, "Hermes could not load the ctxfold engine")
        self.assertEqual(engine.name, "ctxfold")
        from agent.context_engine import ContextEngine
        self.assertIsInstance(engine, ContextEngine)
        for attr in ("last_prompt_tokens", "threshold_tokens", "context_length",
                     "compression_count"):
            self.assertTrue(hasattr(engine, attr), attr)


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module = _load_directly()
        cls.engine = module.CtxfoldEngine(context_length=200_000)
        cls.engine.update_model("gpt-4o", 200_000)

    def test_threshold_and_usage_tracking(self):
        engine = copy.deepcopy(self.engine)
        engine.update_from_response({"prompt_tokens": 150_000, "completion_tokens": 500,
                                     "total_tokens": 150_500})
        self.assertEqual(engine.last_prompt_tokens, 150_000)
        threshold = engine.threshold_tokens
        self.assertTrue(0 < threshold <= 200_000, threshold)
        self.assertTrue(engine.should_compress(threshold))
        self.assertFalse(engine.should_compress(threshold - 1))

    def test_compress_returns_valid_openai_sequence(self):
        messages = _transcript()
        out = self.engine.compress(messages, current_tokens=150_000)
        self.assertIsInstance(out, list)
        self.assertTrue(all(isinstance(m, dict) and m.get("role") for m in out))
        self.assertEqual(len(out), len(messages) + 1)  # + the ANCHORS block
        self.assertEqual(len(self.engine._last_stats), 7)

    def test_input_is_never_mutated(self):
        messages = _transcript()
        before = copy.deepcopy(messages)
        self.engine.compress(messages, current_tokens=150_000)
        self.assertEqual(messages, before, "compress() must not mutate the live transcript")

    def test_system_prompt_and_tail_stay_verbatim(self):
        messages = _transcript()
        out = self.engine.compress(messages, current_tokens=150_000)
        self.assertEqual(out[0], messages[0])
        self.assertEqual(out[-1], messages[-1])

    def test_tool_pairs_stay_intact(self):
        messages = _transcript()
        out = self.engine.compress(messages, current_tokens=150_000)
        for i, message in enumerate(out):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                call_ids = {c["id"] for c in message["tool_calls"]}
                following = [m for m in out[i + 1:i + 1 + len(call_ids)]
                             if m.get("role") == "tool"]
                self.assertEqual(len(following), len(call_ids),
                                 "a tool_call lost its result after compaction")

    def test_anchor_block_is_installed_after_system_prompt(self):
        messages = _transcript()
        out = self.engine.compress(messages, current_tokens=150_000)
        block = out[1]
        self.assertEqual(block["role"], "system")
        self.assertIn("ANCHORS", block["content"])
        self.assertIn("config.compaction.threshold", block["content"])

    def test_deterministic(self):
        messages = _transcript()
        first = self.engine.compress(messages, current_tokens=150_000)
        second = self.engine.compress(messages, current_tokens=150_000)
        self.assertEqual(first, second)

    def test_human_turns_survive_as_text(self):
        messages = _transcript()
        out = self.engine.compress(messages, current_tokens=150_000)
        for message in out:
            if message.get("role") == "user":
                self.assertNotIn("pruned", (message.get("content") or "").lower())

    def test_prune_tool_results_only(self):
        messages = _transcript()
        out, pruned = self.engine.prune_tool_results_only(messages)
        self.assertGreater(pruned, 0)
        self.assertEqual(len(out), len(messages))
        self.assertLess(sum(len(m.get("content") or "") for m in out),
                        sum(len(m.get("content") or "") for m in messages))


class FailOpen(unittest.TestCase):
    def test_missing_worker_never_raises(self):
        module = _load_directly()
        engine = module.CtxfoldEngine(context_length=200_000)
        engine.update_model("gpt-4o", 200_000)
        error = sys.modules["ctxfold_plugin.worker"].WorkerError

        def missing_worker():
            raise error("simulated: worker binary is not installed")

        engine._worker_for_call = missing_worker
        messages = _transcript()
        out = engine.compress(messages, current_tokens=150_000)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), len(messages))
        self.assertTrue(all(m.get("role") for m in out))
        # The fallback still reclaims space: old tool payloads are truncated.
        self.assertLess(
            sum(len(m.get("content") or "") for m in out),
            sum(len(m.get("content") or "") for m in messages),
        )


if __name__ == "__main__":
    unittest.main()
