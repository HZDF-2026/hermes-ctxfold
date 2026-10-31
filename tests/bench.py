"""Rough latency check: how long does a real-size transcript take?

python tests/bench.py            (default: 2M characters, ~500K tokens)

Zero-LLM compaction is the point; this just keeps the claim honest.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "plugins", "context_engine", "ctxfold"))

from worker import Worker  # noqa: E402


def build(target_chars: int):
    kinds = ["system"]
    texts = ["You are Hermes. Workspace H:/proj. Never touch C:/Windows."]
    i = 0
    total = len(texts[0])
    while total < target_chars:
        if i % 4 == 0:
            chunk = "\n".join(
                f"row {j}: /v1/items/{i}/{j}.json 200 elapsed={j}ms token=abc{j}"
                for j in range(120)
            )
            kinds.append("tool")
        elif i % 4 == 1:
            chunk = f"第 {i} 轮：把 config.compaction.threshold 调到 0.35，再跑 gh pr list。"
            kinds.append("chat")
        else:
            chunk = f"turn {i}: patched src/app.cpp, tests green, remaining risk: dist/app.exe size."
            kinds.append("chat")
        texts.append(chunk)
        total += len(chunk)
        i += 1
    return kinds, texts


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    kinds, texts = build(target)
    worker = Worker()
    size = sum(len(t) for t in texts)
    print(f"messages={len(texts)} chars={size:,}")

    t0 = time.perf_counter()
    plans, anchors, stats = worker.compress(kinds, texts, budget=60000)
    dt = time.perf_counter() - t0
    print(f"cold  {dt * 1000:7.1f} ms  in={stats['in_tok']:,} tok  out={stats['out_tok']:,} tok"
          f"  anchors={stats['anchors']}  folded={stats['folded']}  gisted={stats['gisted']}")
    for _ in range(2):
        t0 = time.perf_counter()
        worker.compress(kinds, texts, budget=60000)
        print(f"warm  {(time.perf_counter() - t0) * 1000:7.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
