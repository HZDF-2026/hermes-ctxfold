"""ctxfold context engine for Hermes — zero-LLM, deterministic compaction.

Hermes's built-in ``ContextCompressor`` pays for compaction with a second
model call: it ships the entire middle of the transcript to
``call_llm(task="compression")`` in one shot, so cost and latency scale with
how much context you just built up (and micro-compaction adds one such call
per turn). This engine replaces that pass with the compiled ctxfold worker:
token counting, anchor lifting, folding, gisting and budget allocation all run
in C++ over a length-framed stdout protocol.

Properties that matter:

* **Fast.** ~250 ms for a 2 M-character transcript on a laptop; no network, no
  second model, no summary prompt.
* **Deterministic.** Same transcript in, same transcript out — byte-identical
  across runs, machines and host platforms (the JS/C++ engines are kept in
  sync by golden tests).
* **Nothing is deleted.** Messages are kept, folded, gisted, or tombstoned.
  Tool-call/tool-result pairing stays intact, so the host never has to repair
  the list.
* **Anchors never drift.** Paths, URLs, hashes, commands, config keys and
  numbers are lifted verbatim into a system-level ANCHORS block (intent-bearing
  facts from human/assistant turns first, tool noise last).
* **Fail-open.** A missing/crashed worker degrades to a cheap truncation pass
  and the host keeps working — a broken plugin never blocks a turn.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

from .worker import Worker, WorkerError

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_PERCENT = 0.70   # later than the built-in 0.50: fewer boundaries
DEFAULT_TARGET_RATIO = 0.25        # post-compaction size, as a share of the trigger
DEFAULT_RECENT = 20                # newest messages kept verbatim
DEFAULT_TOOL_FOLD = 400            # fold a tool result longer than this (tokens)
DEFAULT_GIST_LEN = 120
DEFAULT_TIMEOUT = 15.0

ANCHOR_BLOCK_HEADER = (
    "ANCHORS(压缩中强制保留的事实，勿遗漏/勿改写；路径/URL/哈希/命令/配置键/数字均逐字保留):"
)
TOMB_BY_ROLE = {
    "tool": "[ctxfold: pruned]",
    "assistant": "[ctxfold: elided]",
    "user": "[ctxfold: elided]",
}
TOMB_DEFAULT = "[ctxfold: elided]"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _flatten(content: Any) -> str:
    """Best-effort text of a provider content field (str or multimodal parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


class CtxfoldEngine(ContextEngine):
    """Deterministic compaction engine (``context.engine: ctxfold``)."""

    def __init__(self, context_length: int = 0, **_: Any) -> None:
        self.context_length = int(context_length or 0)
        self.threshold_percent = _env_float("CTXFOLD_THRESHOLD", DEFAULT_THRESHOLD_PERCENT)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.protect_first_n = 3
        self._recent = max(2, _env_int("CTXFOLD_RECENT", DEFAULT_RECENT))
        self.protect_last_n = self._recent
        self._target_ratio = _env_float("CTXFOLD_TARGET_RATIO", DEFAULT_TARGET_RATIO)
        self._tool_fold = max(50, _env_int("CTXFOLD_TOOL_FOLD", DEFAULT_TOOL_FOLD))
        self._gist_len = max(40, _env_int("CTXFOLD_GIST_LEN", DEFAULT_GIST_LEN))
        self._timeout = _env_float("CTXFOLD_TIMEOUT", DEFAULT_TIMEOUT)
        self._worker_path = os.environ.get("CTXFOLD_WORKER", "").strip()
        self._worker: Optional[Worker] = None
        self._last_stats: Dict[str, Any] = {}

    # -- ContextEngine surface ---------------------------------------------

    @property
    def name(self) -> str:
        return "ctxfold"

    def update_from_response(self, usage: dict[str, Any]) -> None:
        if not isinstance(usage, dict):
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = self.last_prompt_tokens if prompt_tokens is None else int(prompt_tokens)
        if not self.threshold_tokens:
            return False
        return tokens >= self.threshold_tokens

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Compact the transcript. Never raises: a broken worker fails open."""
        try:
            result = self._compress(messages)
        except Exception as exc:  # noqa: BLE001 - fail-open is the contract
            self._warn(exc)
            return self._fallback_truncate(messages)
        self.compression_count += 1
        return result

    def prune_tool_results_only(
        self, messages: list[dict[str, Any]], current_tokens: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Cheap per-turn trim: truncate old tool payloads, no worker, no rewrite.

        Runs on the host's low, cost-oriented trigger so a long session stops
        re-sending megabytes of tool output long before a full compaction.
        """
        return self._truncate_old_tool_outputs(messages, keep_recent_tool=4)

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["engine"] = self.name
        status["last_plan"] = dict(self._last_stats)
        return status

    # -- internals ---------------------------------------------------------

    def _worker_for_call(self) -> Worker:
        if self._worker is None:
            self._worker = Worker(self._worker_path, timeout=self._timeout)
        return self._worker

    def _budget(self) -> int:
        base = self.threshold_tokens or int(
            (self.context_length or 128_000) * self.threshold_percent
        )
        return max(4_000, int(base * self._target_ratio))

    @staticmethod
    def _encode(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """Map provider messages onto the worker's (kind, text) protocol.

        Kinds: ``system`` (always kept, excluded from anchor lifting), ``tool``,
        ``user`` (never blanked, gisted at worst), ``chat`` (assistant/other).
        Any message whose ``content`` is not a plain string (multimodal, or
        missing) is sent as ``system``: the worker then leaves it untouched,
        because rewriting structured content would destroy it.
        """
        kinds: list[str] = []
        texts: list[str] = []
        for message in messages:
            role = message.get("role") or ""
            content = message.get("content")
            text = content if isinstance(content, str) else _flatten(content)
            if isinstance(content, str):
                if role == "system":
                    kind = "system"
                elif role == "tool":
                    kind = "tool"
                elif role == "user":
                    kind = "user"
                else:
                    kind = "chat"
            else:
                kind = "system"
            kinds.append(kind)
            texts.append(text)
        return kinds, texts

    def _compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not messages:
            return list(messages)
        kinds, texts = self._encode(messages)
        plans, anchors, stats = self._worker_for_call().compress(
            kinds,
            texts,
            budget=self._budget(),
            tool_fold=self._tool_fold,
            recent=self._recent,
            gist_len=self._gist_len,
        )
        self._last_stats = dict(stats)

        out: list[dict[str, Any]] = []
        for message, (action, new_text) in zip(messages, plans):
            if action == "keep":
                out.append(message)
            elif action == "tomb":
                out.append({**message, "content": TOMB_BY_ROLE.get(
                    message.get("role") or "", TOMB_DEFAULT)})
            else:
                out.append({**message, "content": new_text})

        if anchors:
            out.insert(self._anchor_index(out), {
                "role": "system",
                "content": ANCHOR_BLOCK_HEADER + "\n" + "\n".join(anchors),
            })
        return out

    @staticmethod
    def _anchor_index(messages: list[dict[str, Any]]) -> int:
        """Insert right after the leading system run (prompt-cache friendly)."""
        index = 0
        for i, message in enumerate(messages):
            if message.get("role") == "system":
                index = i + 1
            else:
                break
        return index

    def _truncate_old_tool_outputs(
        self, messages: list[dict[str, Any]], *, keep_recent_tool: int,
    ) -> tuple[list[dict[str, Any]], int]:
        tool_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "tool" and isinstance(m.get("content"), str)
        ]
        protected = set(tool_indices[-keep_recent_tool:]) if keep_recent_tool else set()
        out = list(messages)
        pruned = 0
        for i in tool_indices:
            if i in protected:
                continue
            content = messages[i]["content"]
            if len(content) <= 2000:
                continue
            out[i] = {**messages[i], "content": (
                content[:400] + f"\n[…ctxfold: 截断 {len(content) - 400} 字符，"
                f"原始工具输出已省略…]"
            )}
            pruned += 1
        return out, pruned

    def _fallback_truncate(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out, pruned = self._truncate_old_tool_outputs(messages, keep_recent_tool=4)
        if pruned:
            logger.warning(
                "ctxfold engine unavailable: fell back to truncating %d tool outputs", pruned
            )
        return out

    @staticmethod
    def _warn(exc: BaseException) -> None:
        if isinstance(exc, WorkerError):
            logger.warning("ctxfold worker unavailable (%s); transcript left intact", exc)
        else:
            logger.warning("ctxfold compression failed (%s); transcript left intact", exc)
