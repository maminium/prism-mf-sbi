"""Experiment task registry (lazy registration on first `get_task` call)."""

from __future__ import annotations

def __getattr__(name: str):
    if name in ("get_task", "register_task"):
        from prism_vertex.experiments.task import get_task, register_task

        return get_task if name == "get_task" else register_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
