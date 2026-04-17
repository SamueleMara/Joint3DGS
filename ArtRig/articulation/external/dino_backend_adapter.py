from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from articulation.features.dino_wrapper import DinoFeatureExtractor


@dataclass
class DinoBackendAdapter:
    model_name: str = "vit_small_patch14_dinov2"
    device: str = "cpu"
    repo_path: str | None = None

    def build(self) -> DinoFeatureExtractor:
        if self.repo_path:
            p = str(Path(self.repo_path).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        return DinoFeatureExtractor(model_name=self.model_name, device=self.device)
