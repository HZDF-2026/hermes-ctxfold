"""Markdown-authoritative memory store (ctxfold memory format).

The authoritative layer is plain Markdown any agent can read or edit:

    <home>/MEMORY.md                 index page
    <home>/topics/<slug>.md          entries, marked with ctxfold:entry comments
    <home>/archive.md                curated-out entries

An entry is:

    <!-- ctxfold:entry id=.. topic=.. kind=.. created=.. hits=.. anchors="a;b" -->
    body markdown
    <!-- /ctxfold:entry -->

The file format is shared with the JS/C++ memory implementations, so one
memory directory can be read by any ctxfold-aware host. Everything
deterministic (entry ids, distillation, recall ranking) lives in the C++
worker; this module only does files and parsing.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

ENTRY_OPEN = re.compile(r"<!--\s*ctxfold:entry\s+([^>]*?)\s*-->")
ENTRY_CLOSE = re.compile(r"<!--\s*/ctxfold:entry\s*-->")
ATTR = re.compile(r'(\w+)=("[^"]*"|\S+)')
SLUG_STRIP = re.compile(r"[^a-z0-9\u4e00-\u9fa5]+")


def slug(topic: str) -> str:
    cleaned = SLUG_STRIP.sub("-", (topic or "general").lower()).strip("-")[:40].strip("-")
    return cleaned or "general"


def norm_text(text: str) -> str:
    return " ".join((text or "").split())


def _enc_anchors(anchors: Iterable[str]) -> str:
    return ";".join(str(a).replace(";", "；") for a in anchors or [])


def _dec_anchors(raw: str) -> List[str]:
    return [part.strip() for part in (raw or "").split(";") if part.strip()]


class Entry:
    __slots__ = ("id", "topic", "kind", "created", "hits", "anchors", "text")

    def __init__(self, id: str, topic: str, kind: str, created: int, hits: int,
                 anchors: List[str], text: str) -> None:
        self.id = id
        self.topic = topic
        self.kind = kind
        self.created = created
        self.hits = hits
        self.anchors = anchors
        self.text = text

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "topic": self.topic, "kind": self.kind,
            "created": self.created, "hits": self.hits,
            "anchors": self.anchors, "text": self.text,
        }

    def serialize(self) -> str:
        attrs = (f"id={self.id} topic={self.topic} kind={self.kind} "
                 f"created={self.created} hits={self.hits} "
                 f'anchors="{_enc_anchors(self.anchors)}"')
        return (f"<!-- ctxfold:entry {attrs} -->\n{self.text}\n"
                f"<!-- /ctxfold:entry -->\n")


class MemoryStore:
    """Entry store over one directory. Thread-safe enough for one agent."""

    def __init__(self, directory: str, worker) -> None:
        self.dir = directory
        self.topics_dir = os.path.join(directory, "topics")
        self.worker = worker
        self._entries: Dict[str, Entry] = {}
        self._lock = threading.RLock()

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "MemoryStore":
        os.makedirs(self.topics_dir, exist_ok=True)
        with self._lock:
            self._entries = {}
            for name in sorted(os.listdir(self.topics_dir)):
                if not name.endswith(".md"):
                    continue
                path = os.path.join(self.topics_dir, name)
                try:
                    with open(path, encoding="utf-8") as handle:
                        raw = handle.read()
                except OSError:
                    continue
                for entry in parse_topic_file(raw):
                    self._entries.setdefault(entry.id, entry)
        return self

    def close(self) -> None:
        """Nothing to release: the authoritative layer is files."""

    # -- writes ------------------------------------------------------------

    def add(self, text: str, *, topic: str = "general", kind: str = "fact",
            anchors: Optional[List[str]] = None) -> Dict[str, Any]:
        """Store one memory. Same fact again = hit (hits+1), not a duplicate."""
        body = norm_text(text)
        if not body:
            raise ValueError("memory.add: empty text")
        topic_slug = slug(topic)
        entry_id = self.worker.entry_id(topic_slug, body)
        with self._lock:
            existing = self._entries.get(entry_id)
            if existing is not None:
                existing.hits += 1
                return {"id": entry_id, "merged": True, "entry": existing}
            entry = Entry(
                id=entry_id, topic=topic_slug, kind=kind or "fact",
                created=int(time.time()), hits=0,
                anchors=list(anchors or []), text=body,
            )
            self._entries[entry_id] = entry
            return {"id": entry_id, "merged": False, "entry": entry}

    def forget(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id not in self._entries:
                return False
            del self._entries[entry_id]
        self.flush()
        return True

    def flush(self) -> None:
        """Rewrite the authoritative layer (deterministic order)."""
        with self._lock:
            by_topic: Dict[str, List[Entry]] = {}
            for entry in self._entries.values():
                by_topic.setdefault(slug(entry.topic), []).append(entry)
            os.makedirs(self.topics_dir, exist_ok=True)
            for topic, entries in by_topic.items():
                entries.sort(key=lambda e: (e.created, e.id))
                path = os.path.join(self.topics_dir, f"{topic}.md")
                _write(path, "".join(e.serialize() for e in entries))
            for name in os.listdir(self.topics_dir):
                if name.endswith(".md") and slug(name[:-3]) not in by_topic:
                    try:
                        os.remove(os.path.join(self.topics_dir, name))
                    except OSError:
                        pass
            _write(os.path.join(self.dir, "MEMORY.md"), self.render_index())

    def render_index(self) -> str:
        topics = sorted({e.topic for e in self._entries.values()})
        lines = [
            "# 记忆索引（MEMORY.md）",
            "",
            "> ctxfold memory 权威层索引页。正文在 topics/<主题>.md，条目以 "
            "ctxfold:entry 注释标记；可直接编辑，策展时会重写本页。",
            "",
        ]
        for topic in topics:
            entries = sorted(
                (e for e in self._entries.values() if e.topic == topic),
                key=lambda e: (-e.hits, e.created, e.id),
            )
            lines.append(f"## {topic}（{len(entries)}）")
            lines.append("")
            for entry in entries:
                preview = entry.text.split("\n")[0][:80]
                lines.append(f"- [{entry.kind}] {entry.id}（hits {entry.hits}）{preview}")
            lines.append("")
        return "\n".join(lines)

    # -- reads -------------------------------------------------------------

    def entries(self) -> List[Entry]:
        with self._lock:
            return list(self._entries.values())

    def recall(self, query: str, *, k: int = 12, budget: int = 900) -> List[Entry]:
        entries = self.entries()
        if not entries:
            return []
        picked, _used = self.worker.recall(
            query, [e.as_dict() for e in entries], k=k, budget=budget,
        )
        return [entries[index] for index, _cost in picked]

    def render_recall(self, picked: List[Entry], query: str = "") -> str:
        if not picked:
            return ""
        lines = [
            "## 召回的长期记忆（ctxfold memory，按相关度排序；锚点为逐字事实，勿改写）",
        ]
        for entry in picked:
            lines.append(f"- [{entry.kind}/{entry.topic}] {entry.text}")
            if entry.anchors:
                lines.append(f"  锚点: {' | '.join(entry.anchors)}")
        return "\n".join(lines)

    # -- maintenance -------------------------------------------------------

    def curate(self, *, retention_days: int = 180) -> Dict[str, int]:
        """Archive zero-hit entries older than the retention window."""
        cutoff = int(time.time()) - retention_days * 86400
        archived: List[Entry] = []
        with self._lock:
            for entry_id, entry in list(self._entries.items()):
                if entry.hits == 0 and 0 < entry.created < cutoff:
                    archived.append(entry)
                    del self._entries[entry_id]
            if archived:
                path = os.path.join(self.dir, "archive.md")
                previous = ""
                if os.path.exists(path):
                    with open(path, encoding="utf-8") as handle:
                        previous = handle.read()
                _write(path, previous + "".join(e.serialize() for e in archived))
        self.flush()
        return {"archived": len(archived), "kept": len(self._entries)}

    def verify(self) -> Dict[str, Any]:
        bad = []
        for entry in self.entries():
            expected = self.worker.entry_id(entry.topic, entry.text)
            if expected != entry.id:
                bad.append({"id": entry.id, "reason": "id 不匹配（正文被改过？）"})
        return {"ok": not bad, "bad": bad, "entries": len(self._entries)}

    def stats(self) -> Dict[str, Any]:
        by_kind: Dict[str, int] = {}
        by_topic: Dict[str, int] = {}
        for entry in self._entries.values():
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
            by_topic[entry.topic] = by_topic.get(entry.topic, 0) + 1
        return {
            "entries": len(self._entries), "topics": len(by_topic),
            "by_kind": by_kind, "by_topic": by_topic, "dir": self.dir,
        }


def parse_topic_file(raw: str) -> List[Entry]:
    entries: List[Entry] = []
    rest = raw
    while True:
        match = ENTRY_OPEN.search(rest)
        if not match:
            break
        after = rest[match.end():]
        close = ENTRY_CLOSE.search(after)
        if not close:
            break  # half-written entry (editor interrupted): keep the rest untouched
        attrs = {m.group(1): m.group(2).strip('"') for m in ATTR.finditer(match.group(1))}
        entries.append(Entry(
            id=attrs.get("id", ""), topic=attrs.get("topic", "general"),
            kind=attrs.get("kind", "fact"),
            created=int(attrs.get("created") or 0), hits=int(attrs.get("hits") or 0),
            anchors=_dec_anchors(attrs.get("anchors", "")),
            text=after[:close.start()].strip("\n"),
        ))
        rest = after[close.end():]
    return entries


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
