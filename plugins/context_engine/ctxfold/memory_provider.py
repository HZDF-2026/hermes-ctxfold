"""ctxfold memory provider for Hermes — durable recall across compaction.

Hermes's built-in memory is `MEMORY.md` / `USER.md` plus whatever the model
chooses to write. Compaction then throws transcript detail away. This provider
closes that loop with the ctxfold memory library:

* ``on_pre_compress``  — before the lossy rewrite, the transcript about to be
  folded is distilled into the store (facts / lessons / decisions). Compaction
  stops meaning "forget".
* ``prefetch``         — before each request, relevance-ranked entries are
  recalled (C++ scoring) and injected; results are cached so retries are free.
* ``sync_turn``        — every finished turn is distilled into the store.
* ``on_session_end``   — full transcript distilled at a real session boundary.

The authoritative layer stays plain Markdown (``MEMORY.md`` + ``topics/*.md``)
in the ctxfold entry format, so the same directory is readable by the dsh,
Pi and Torre di Babile ports. All deterministic work (distillation, ids,
ranking) runs in the compiled worker; failures degrade to "no memory this
turn", never to a broken conversation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

from agent.memory_provider import (  # noqa: F401 - MemoryProvider is the contract
    MemoryProvider, RecallStatus, is_trivial_prompt,
)

from .store import MemoryStore
from .worker import Worker, WorkerError

logger = logging.getLogger(__name__)

DEFAULT_RECALL_BUDGET = 900     # tokens of recalled context per request
DEFAULT_RECALL_K = 8
DEFAULT_MAX_SCAN = 200_000      # characters distilled from one transcript batch
SYSTEM_BLOCK = (
    "长期记忆由 ctxfold 维护（Markdown 权威层 + 确定性蒸馏）。"
    "需要回忆跨会话事实时调用 ctxfold_recall；写入用内置 memory 工具。"
)


class CtxfoldMemoryProvider(MemoryProvider):
    """External memory provider (``memory.provider: ctxfold``)."""

    def __init__(self) -> None:
        self._home = ""
        self._dir = ""
        self._store: Optional[MemoryStore] = None
        self._worker: Optional[Worker] = None
        self._session_id = ""
        self._primary = True
        self._topic = "hermes"
        self._cache: Dict[str, str] = {}
        self._cache_lock = threading.Lock()
        self._last_recall: Optional[RecallStatus] = None
        self._config: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    @property
    def name(self) -> str:
        return "ctxfold"

    def is_available(self) -> bool:
        return True  # pure local store; the worker is optional (fail-open)

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        self._session_id = session_id or ""
        self._home = str(kwargs.get("hermes_home") or "")
        self._primary = str(kwargs.get("agent_context") or "primary") in ("", "primary")
        cwd = str(kwargs.get("cwd") or "")
        if cwd:
            self._topic = _topic_from_cwd(cwd)
        self._config = _load_config(self._home)
        self._dir = (os.environ.get("CTXFOLD_MEMORY_DIR") or "").strip() \
            or str(self._config.get("memory_dir") or "").strip() \
            or os.path.join(self._home or ".", "ctxfold-memory")
        try:
            self._store = MemoryStore(self._dir, self._worker_for_call()).open()
        except Exception as exc:  # noqa: BLE001 - memory is optional, not critical
            logger.warning("ctxfold memory: store unavailable (%s)", exc)
            self._store = None

    def shutdown(self) -> None:
        if self._store is not None:
            try:
                self._store.flush()
            except Exception as exc:  # noqa: BLE001
                logger.debug("ctxfold memory: flush on shutdown failed (%s)", exc)

    def system_prompt_block(self) -> str:
        return SYSTEM_BLOCK

    # -- recall ------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return recalled context for this request ("" when nothing matches)."""
        self._last_recall = None
        if not query or is_trivial_prompt(query) or self._store is None:
            return ""
        with self._cache_lock:
            cached = self._cache.get(query)
        if cached is not None:
            self._last_recall = RecallStatus("ctxfold", _count_entries(cached))
            return cached
        try:
            rendered = self._recall_now(query)
        except (WorkerError, OSError, ValueError) as exc:
            logger.warning("ctxfold memory: recall skipped (%s)", exc)
            return ""
        self._last_recall = RecallStatus("ctxfold", _count_entries(rendered))
        return rendered

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the cache for `query` in the background (best-effort)."""
        if not query or is_trivial_prompt(query) or self._store is None:
            return
        with self._cache_lock:
            if query in self._cache:
                return
        try:
            from agent.memory_provider import spawn_context_thread
            spawn_context_thread(
                self._recall_now, name="ctxfold-recall", args=(query,),
            ).start()
        except Exception:  # noqa: BLE001 - background warmth is optional
            pass

    def recall_status(self) -> Optional[RecallStatus]:
        return self._last_recall

    def _recall_now(self, query: str) -> str:
        entries = self._store.recall(
            query, k=DEFAULT_RECALL_K, budget=DEFAULT_RECALL_BUDGET,
        )
        rendered = self._store.render_recall(entries, query)
        with self._cache_lock:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[query] = rendered
        return rendered

    # -- ingestion ---------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[list] = None,
                  turn_author: Optional[dict] = None) -> None:
        if not self._primary or self._store is None:
            return
        self._distill(f"{user_content or ''}\n{assistant_content or ''}")

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        """Archive the transcript BEFORE the lossy rewrite (the recall loop)."""
        if not self._primary or self._store is None:
            return ""
        self._distill(_transcript_text(messages))
        return ""

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        if not self._primary or self._store is None:
            return
        self._distill(_transcript_text(messages))

    def _distill(self, text: str) -> None:
        text = (text or "").strip()
        if len(text) < 40:
            return
        if len(text) > DEFAULT_MAX_SCAN:
            text = text[:DEFAULT_MAX_SCAN]
        try:
            distilled = self._worker_for_call().distill(text)
            for fact in distilled["facts"]:
                self._store.add(fact, topic=self._topic, kind="fact", anchors=[fact])
            for lesson in distilled["lessons"]:
                self._store.add(lesson, topic=self._topic, kind="lesson")
            for decision in distilled["decisions"]:
                self._store.add(decision, topic=self._topic, kind="decision")
            self._store.flush()
        except (WorkerError, OSError, ValueError) as exc:
            logger.warning("ctxfold memory: distill skipped (%s)", exc)

    # -- tools -------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "ctxfold_recall",
                "description": "Recall durable long-term memories (facts, lessons, "
                               "decisions) captured before compaction. Use before asking "
                               "the user to repeat context from an earlier session.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "what to remember about"},
                        "k": {"type": "integer", "description": "max entries (default 8)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ctxfold_memory",
                "description": "Inspect or maintain the ctxfold memory library.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string",
                                   "enum": ["stats", "verify", "curate"]},
                    },
                    "required": ["action"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        if self._store is None:
            return json.dumps({"error": "memory store unavailable"})
        try:
            if tool_name == "ctxfold_recall":
                query = str(args.get("query") or "")
                if not query:
                    return json.dumps({"error": "query is required"})
                entries = self._store.recall(
                    query, k=int(args.get("k") or DEFAULT_RECALL_K),
                    budget=DEFAULT_RECALL_BUDGET,
                )
                return json.dumps({"count": len(entries), "entries": [
                    {"id": e.id, "kind": e.kind, "topic": e.topic,
                     "text": e.text, "anchors": e.anchors, "hits": e.hits}
                    for e in entries
                ]}, ensure_ascii=False)
            if tool_name == "ctxfold_memory":
                action = str(args.get("action") or "stats")
                if action == "verify":
                    return json.dumps(self._store.verify(), ensure_ascii=False)
                if action == "curate":
                    return json.dumps(self._store.curate(), ensure_ascii=False)
                return json.dumps(self._store.stats(), ensure_ascii=False)
        except (WorkerError, OSError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # -- config ------------------------------------------------------------

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [{
            "key": "memory_dir",
            "description": "Memory library directory (default: <hermes_home>/ctxfold-memory)",
            "default": "",
            "env_var": "CTXFOLD_MEMORY_DIR",
        }]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = os.path.join(hermes_home, "ctxfold-memory.json")
        try:
            os.makedirs(hermes_home, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"memory_dir": str(values.get("memory_dir") or "")}, handle)
        except OSError as exc:
            logger.warning("ctxfold memory: save_config failed (%s)", exc)

    def backup_paths(self) -> list[str]:
        home = os.path.abspath(self._home) if self._home else ""
        target = os.path.abspath(self._dir) if self._dir else ""
        if not target or (home and target.startswith(home)):
            return []  # inside HERMES_HOME: `hermes backup` already covers it
        return [target]

    # -- internals ---------------------------------------------------------

    def _worker_for_call(self) -> Worker:
        if self._worker is None:
            self._worker = Worker(os.environ.get("CTXFOLD_WORKER", "").strip())
        return self._worker


def _topic_from_cwd(cwd: str) -> str:
    base = os.path.basename(os.path.normpath(cwd)) or "hermes"
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "-", base.lower()).strip("-")[:40]
    return cleaned or "hermes"


def _load_config(hermes_home: str) -> Dict[str, Any]:
    if not hermes_home:
        return {}
    path = os.path.join(hermes_home, "ctxfold-memory.json")
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _transcript_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages or []:
        role = message.get("role") or "?"
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(p.get("text") or "") for p in content if isinstance(p, dict)
            )
        else:
            text = ""
        if text:
            parts.append(f"[{role}] {text}")
    return "\n".join(parts)


def _count_entries(rendered: str) -> int:
    return sum(1 for line in rendered.splitlines() if line.startswith("- ["))
