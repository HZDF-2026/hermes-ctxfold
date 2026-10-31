"""ctxfold — deterministic context engine for Hermes Agent.

Hermes discovers this package either as a *context engine* (drop it into
``$HERMES_HOME/plugins/ctxfold/`` and set ``context.engine: ctxfold``) or as a
general plugin that calls ``ctx.register_context_engine(...)``. Both routes go
through the same ``CtxfoldEngine``.

The deterministic core (tokens, anchors, folding, gisting, budget allocation)
is the compiled ``ctxfold`` worker in ``bin/``; see ``engine.py`` for the
integration and ``worker.py`` for the protocol client.
"""

from .engine import CtxfoldEngine

__all__ = ["CtxfoldEngine"]


def register(ctx) -> None:
    """Hermes plugin entry point (context engine route)."""
    engine = CtxfoldEngine()
    ctx.register_context_engine(engine)
