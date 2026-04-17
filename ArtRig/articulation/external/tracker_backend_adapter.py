from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from articulation.tracking.tracker_wrapper import TrackerWrapper


@dataclass
class TrackerBackendAdapter:
    cfg: dict
    repo_path: str | None = None

    def build(self) -> TrackerWrapper:
        if self.repo_path:
            p = str(Path(self.repo_path).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        return TrackerWrapper.from_config(self.cfg)
