from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
import sys

from articulation.data.dataclasses import KeypointBatch, MatchBatch


MatchCallable = Callable[[KeypointBatch, KeypointBatch, str], MatchBatch | dict[str, Any]]


def _resolve_callable(dotted: str) -> MatchCallable:
    if ":" not in dotted:
        raise ValueError("backend_callable must be in format 'module.submodule:function'")
    mod_name, fn_name = dotted.split(":", 1)
    mod = import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"Loaded matcher backend target is not callable: {dotted}")
    return fn


@dataclass
class MatcherBackendAdapter:
    backend_callable: str
    repo_path: str | None = None

    def build(self) -> MatchCallable:
        if self.repo_path:
            p = str(Path(self.repo_path).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        return _resolve_callable(self.backend_callable)
