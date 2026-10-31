"""Contract tests for the ctxfold C++ worker and its Python client.

Run: python -m unittest discover -s tests -v   (from the repo root)

The parity tests compare the C++ worker against the JS reference engine on the
same inputs; they skip when node or a ctxfold checkout is unavailable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "plugins", "context_engine", "ctxfold"))

from worker import Worker, WorkerError  # noqa: E402

GOLDEN = os.path.join(HERE, "golden_input.txt")
JS_HOME = os.environ.get("CTXFOLD_HOME", "H:/ctxfold")


def golden_text() -> str:
    with open(GOLDEN, encoding="utf-8") as fh:
        return fh.read()


def synthetic_session(messages: int = 60, tool_chars: int = 6000):
    """A deterministic transcript: system prompt, tool-heavy middle, recent tail."""
    kinds = ["system"]
    texts = ["You are Hermes, a coding agent. Work in H:/proj, never touch C:/Windows."]
    for i in range(messages):
        if i % 3 == 0:
            kinds.append("tool")
            body = "\n".join(
                f"line {j}: result at /v1/items/{i}/{j}.json status=200 elapsed={j}ms"
                for j in range(tool_chars // 60)
            )
            texts.append(body)
        elif i % 3 == 1:
            kinds.append("chat")
            texts.append(f"第 {i} 轮：请把 config.compaction.threshold 调到 0.35，然后跑 gh pr list。")
        else:
            kinds.append("chat")
            texts.append(f"assistant {i}: done, tests pass; 剩余 risk 是 dist/app.exe 体积。")
    return kinds, texts


class WorkerBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = Worker()

    def test_ping(self):
        self.assertTrue(self.worker.ping())

    def test_entry_id_is_stable_and_short(self):
        first = self.worker.entry_id("ctxfold", "决定采用 Markdown 做权威层")
        second = self.worker.entry_id("ctxfold", "决定采用  Markdown   做权威层")
        self.assertEqual(first, second)  # whitespace-normalized
        self.assertRegex(first, r"^[0-9a-f]{12}$")

    def test_distill_matches_js_reference(self):
        text = golden_text()
        got = self.worker.distill(text)
        self.assertTrue(got["facts"])
        self.assertTrue(got["lessons"])
        self.assertTrue(got["decisions"])
        expected = js_reference(text)
        if expected is None:
            self.skipTest("node or the ctxfold JS checkout is unavailable")
        self.assertEqual(got["facts"], expected["facts"])
        self.assertEqual(got["lessons"], expected["lessons"])
        self.assertEqual(got["decisions"], expected["decisions"])
        self.assertEqual(
            self.worker.entry_id("pi-compaction", "25000"),
            expected["entryId"],
        )

    def test_bad_protocol_raises(self):
        with self.assertRaises(WorkerError):
            Worker()._call(b"garbage\n")


class Compression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = Worker()

    def test_message_count_is_preserved(self):
        kinds, texts = synthetic_session()
        plans, anchors, stats = self.worker.compress(kinds, texts, budget=20000)
        self.assertEqual(len(plans), len(texts))
        self.assertTrue(anchors)
        # Intent-bearing anchors (chat turns) win the budget over tool noise,
        # and the system prompt's own facts stay verbatim without being lifted.
        self.assertIn("config.compaction.threshold", anchors)

    def test_system_and_recent_tail_are_verbatim(self):
        kinds, texts = synthetic_session()
        plans, _, _ = self.worker.compress(kinds, texts, budget=20000, recent=20)
        self.assertEqual(plans[0], ("keep", None))  # system prompt untouched
        for i in range(len(texts) - 20, len(texts)):
            self.assertEqual(plans[i], ("keep", None), f"recent message {i} was rewritten")

    def test_no_message_is_deleted(self):
        kinds, texts = synthetic_session()
        plans, _, stats = self.worker.compress(kinds, texts, budget=20000)
        allowed = {"keep", "tomb", "rewrite"}
        self.assertTrue(all(p[0] in allowed for p in plans))
        self.assertEqual(stats["kept"] + stats["tomb"], len(texts))

    def test_output_fits_the_budget(self):
        # The protected tail (system + newest 20) is untouchable: with a tiny
        # budget the floor is that tail, not the requested number.
        kinds, texts = synthetic_session()
        for budget in (20000, 60000):
            plans, _, stats = self.worker.compress(kinds, texts, budget=budget)
            self.assertLessEqual(
                stats["out_tok"], budget * 1.10,
                f"budget {budget} overshot: {stats['out_tok']}",
            )

    def test_tight_budget_compacts_aggressively(self):
        kinds, texts = synthetic_session()
        _, _, loose = self.worker.compress(kinds, texts, budget=60000)
        plans, _, tight = self.worker.compress(kinds, texts, budget=5000, recent=10)
        self.assertGreater(tight["tomb"], 0)
        self.assertLess(tight["out_tok"], loose["out_tok"])
        self.assertEqual(
            sum(1 for p in plans if p[0] == "keep"), tight["kept"],
        )

    def test_long_tool_output_is_folded(self):
        kinds, texts = synthetic_session()
        plans, _, stats = self.worker.compress(kinds, texts, budget=20000, tool_fold=400)
        self.assertGreater(stats["folded"], 0)
        folded = [p for p in plans if p[0] == "rewrite" and "ctxfold: 折叠" in (p[1] or "")]
        self.assertTrue(folded, "expected at least one folded tool block")

    def test_deterministic(self):
        kinds, texts = synthetic_session()
        a = self.worker.compress(kinds, texts, budget=20000)
        b = self.worker.compress(kinds, texts, budget=20000)
        self.assertEqual(a, b)

    def test_tombstone_only_outside_protected_region(self):
        kinds, texts = synthetic_session()
        plans, _, _ = self.worker.compress(kinds, texts, budget=3000, recent=10)
        for i, plan in enumerate(plans):
            if plan[0] == "tomb":
                self.assertNotEqual(kinds[i], "system")

    def test_zero_budget_is_a_noop(self):
        kinds, texts = synthetic_session(9)
        plans, anchors, stats = self.worker.compress(kinds, texts, budget=0)
        self.assertTrue(all(p[0] == "keep" for p in plans))
        self.assertEqual(anchors, [])
        self.assertEqual(stats["tomb"], 0)


def js_reference(text: str):
    """Run the JS reference engine (ctxfold.js + memory.js) over the same text."""
    script = os.path.join(JS_HOME, "memory.js")
    if not os.path.isfile(script):
        return None
    try:
        proc = subprocess.run(
            ["node", "-e", (
                "const m=require(process.argv[1]);"
                "const c=require(process.argv[2]);"
                "const t=process.env.CTXFOLD_TEXT;"
                "console.log(JSON.stringify({facts:c.anchors(t).slice(0,40),"
                "...m.distillText(t,{maxFacts:40}),"
                "entryId:m.hashId('pi-compaction','25000')}));"
            ), script, os.path.join(JS_HOME, "ctxfold.js")],
            input=b"",
            capture_output=True,
            text=True,
            env={**os.environ, "CTXFOLD_TEXT": text},
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return data


if __name__ == "__main__":
    unittest.main()
