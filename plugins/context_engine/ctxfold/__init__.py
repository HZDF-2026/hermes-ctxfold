"""ctxfold for Hermes Agent — deterministic context engine + durable memory.

One package, two seams, each activated by its own config key:

* ``context.engine: ctxfold``   → :class:`CtxfoldEngine`, zero-LLM compaction
  (replaces Hermes's summary model call).
* ``memory.provider: ctxfold``  → :class:`CtxfoldMemoryProvider`, distill +
  recall so compaction never means "forget".

The deterministic core (tokens, anchors, folding, gisting, budget allocation,
distillation, entry ids, recall ranking) is the compiled worker in ``bin/``.
The Python files only move bytes and files.
"""

__all__ = ["CtxfoldEngine", "CtxfoldMemoryProvider"]


def register(ctx):
    """Hermes plugin entry point.

    The same package is loaded through two collectors: the context-engine
    collector exposes ``engine``, the memory collector exposes ``provider``.
    Register only for the seam that is asking, so neither route has side
    effects on the other.
    """
    if hasattr(ctx, "engine"):
        from .engine import CtxfoldEngine

        ctx.register_context_engine(CtxfoldEngine())
        return
    if hasattr(ctx, "provider"):
        from .memory_provider import CtxfoldMemoryProvider

        ctx.register_memory_provider(CtxfoldMemoryProvider())
        return
    # General plugin system: both providers are activated by config
    # (`context.engine` / `memory.provider`), so there is nothing to force here.
