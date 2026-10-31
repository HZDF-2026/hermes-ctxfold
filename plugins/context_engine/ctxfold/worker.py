"""Client for the ctxfold C++ worker (length-framed stdin/stdout protocol).

This module is deliberately dumb: it moves bytes, checks the header, and
turns the reply into Python values. No algorithm lives here — the deterministic
core is the compiled worker, so the host pays one process start per call and
never runs tokenization, anchors, folding or budget math in Python.

Failure policy: every transport problem raises WorkerError with a short reason.
Callers must fail open (keep the transcript, let the host's own machinery run).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

PROTOCOL = "CTXFOLD/1"


class WorkerError(RuntimeError):
    """The worker was missing, crashed, timed out, or answered badly."""


def _no_window_flags() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return 0


def find_worker(explicit: str = "") -> str:
    """Resolve the worker binary: explicit path, $CTXFOLD_WORKER, then bin/
    next to this file, then PATH. Raises WorkerError when nothing is found."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("CTXFOLD_WORKER", "").strip()
    if env:
        candidates.append(env)
    here = os.path.dirname(os.path.abspath(__file__))
    name = "ctxfold.exe" if sys.platform == "win32" else "ctxfold"
    candidates.append(os.path.join(here, "bin", name))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("ctxfold")
    if found:
        return found
    raise WorkerError(
        "ctxfold worker not found — build it with build.ps1 (or set CTXFOLD_WORKER)"
    )


class Reply:
    """Cursor over the worker's reply bytes."""

    def __init__(self, raw: bytes, status: str) -> None:
        self.raw = raw
        self.pos = 0
        self.status = status

    def line(self) -> str:
        nl = self.raw.find(b"\n", self.pos)
        if nl < 0:
            text = self.raw[self.pos:].decode("utf-8", "replace")
            self.pos = len(self.raw)
            return text
        text = self.raw[self.pos:nl].decode("utf-8", "replace")
        self.pos = nl + 1
        return text

    def body(self, length: int) -> str:
        if self.pos + length > len(self.raw):
            raise WorkerError("truncated reply record")
        out = self.raw[self.pos:self.pos + length].decode("utf-8", "replace")
        self.pos += length
        return out


class Worker:
    """Stateless request/response client. One process per call."""

    def __init__(self, exe: str = "", timeout: float = 15.0) -> None:
        self.exe = find_worker(exe)
        self.timeout = timeout

    def _call(self, payload: bytes) -> Reply:
        try:
            proc = subprocess.run(
                [self.exe],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
                creationflags=_no_window_flags(),
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkerError(f"worker timed out after {self.timeout:g}s") from exc
        except OSError as exc:
            raise WorkerError(f"worker failed to start: {exc}") from exc
        if proc.returncode != 0:
            detail = proc.stdout.decode("utf-8", "replace").strip()[:200]
            if not detail:
                detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
            raise WorkerError(f"worker exit {proc.returncode}: {detail or 'no output'}")
        reply = Reply(proc.stdout, "")
        head = reply.line()
        if not head.startswith(PROTOCOL + " OK"):
            raise WorkerError(f"bad reply header: {head[:120]!r}")
        reply.status = head[len(PROTOCOL) + 4:].strip()
        return reply

    # -- ops ---------------------------------------------------------------

    def ping(self) -> bool:
        self._call(f"{PROTOCOL} ping\n".encode("ascii"))
        return True

    def compress(self, kinds, texts, *, budget, tool_fold=400, head=6, tail=4,
                 recent=20, gist_len=120):
        """Return (plans, anchors, stats).

        plans[i] is ("keep", None) | ("tomb", None) | ("rewrite", new_text).
        """
        n = len(texts)
        parts = [
            (f"{PROTOCOL} compress {int(budget)} {int(tool_fold)} {int(head)} "
             f"{int(tail)} {int(recent)} {int(gist_len)} {n}\n").encode("utf-8")
        ]
        for kind, text in zip(kinds, texts):
            raw = text.encode("utf-8")
            parts.append(f"{kind}\t{len(raw)}\n".encode("utf-8"))
            parts.append(raw)
        reply = self._call(b"".join(parts))

        fields = reply.status.split()
        if len(fields) != 7:
            raise WorkerError(f"compress: expected 7 stats, got {reply.status!r}")
        kept, tomb, gisted, folded, in_tok, out_tok, anchors_n = (int(x) for x in fields)

        plans = []
        for _ in range(n):
            line = reply.line()
            if line == "keep":
                plans.append(("keep", None))
            elif line == "tomb":
                plans.append(("tomb", None))
            elif line.startswith("rewrite\t"):
                plans.append(("rewrite", reply.body(int(line.split("\t", 1)[1]))))
            else:
                raise WorkerError(f"compress: bad action {line[:40]!r}")

        anchors = []
        for _ in range(anchors_n):
            line = reply.line()
            if not line.startswith("A\t"):
                raise WorkerError(f"compress: bad anchor record {line[:40]!r}")
            anchors.append(reply.body(int(line.split("\t", 1)[1])))

        stats = {
            "kept": kept, "tomb": tomb, "gisted": gisted, "folded": folded,
            "in_tok": in_tok, "out_tok": out_tok, "anchors": anchors_n,
        }
        return plans, anchors, stats

    def distill(self, text, *, max_facts=40, max_lessons=12, max_decisions=12):
        raw = text.encode("utf-8")
        payload = (
            f"{PROTOCOL} distill {int(max_facts)} {int(max_lessons)} "
            f"{int(max_decisions)} {len(raw)}\n".encode("utf-8") + raw
        )
        reply = self._call(payload)
        fields = reply.status.split()
        if len(fields) != 3:
            raise WorkerError(f"distill: expected 3 counts, got {reply.status!r}")
        n_facts, n_lessons, n_decisions = (int(x) for x in fields)
        facts, lessons, decisions = [], [], []
        for _ in range(n_facts + n_lessons + n_decisions):
            line = reply.line()
            tag, _, size = line.partition("\t")
            body = reply.body(int(size))
            {"F": facts, "L": lessons, "D": decisions}[tag].append(body)
        return {"facts": facts, "lessons": lessons, "decisions": decisions}

    def recall(self, query: str, entries, *, k: int = 12, budget: int = 800):
        """Rank stored memory entries against `query`.

        `entries` is a sequence of dicts with keys ``hits``, ``kind``, ``text``,
        ``topic``, ``anchors`` (a list of verbatim facts). Returns a list of
        ``(index, cost_tokens)`` in relevance order, within `budget`.
        """
        query_raw = query.encode("utf-8")
        parts = [
            (f"{PROTOCOL} recall {len(query_raw)} {int(k)} {int(budget)}\n").encode("utf-8"),
            query_raw,
            f"{len(entries)}\n".encode("utf-8"),
        ]
        for entry in entries:
            kind = str(entry.get("kind") or "fact").encode("utf-8")
            text = str(entry.get("text") or "").encode("utf-8")
            topic = str(entry.get("topic") or "general").encode("utf-8")
            anchors = ";".join(str(a) for a in (entry.get("anchors") or [])).encode("utf-8")
            parts.append(
                f"{int(entry.get('hits') or 0)}\t{len(kind)}\t{len(text)}\t"
                f"{len(topic)}\t{len(anchors)}\n".encode("utf-8")
            )
            parts.append(kind + text + topic + anchors)
        reply = self._call(b"".join(parts))

        fields = reply.status.split()
        if len(fields) != 2:
            raise WorkerError(f"recall: expected 2 stats, got {reply.status!r}")
        picked, used = (int(x) for x in fields)
        out = []
        for _ in range(picked):
            line = reply.line()
            index, _, cost = line.partition("\t")
            out.append((int(index), int(cost)))
        return out, used

    def entry_id(self, topic: str, text: str) -> str:
        topic_raw = topic.encode("utf-8")
        text_raw = text.encode("utf-8")
        payload = (
            f"{PROTOCOL} id {len(topic_raw)} {len(text_raw)}\n".encode("utf-8")
            + topic_raw + text_raw
        )
        reply = self._call(payload)
        return reply.status.strip()
