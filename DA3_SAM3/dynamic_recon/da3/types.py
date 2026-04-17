from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class DA3FrameOutput:
    frame_index: int
    depth: torch.Tensor
    intrinsics: torch.Tensor
    extrinsics: torch.Tensor
    confidence: torch.Tensor | None = None
    rgb: torch.Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DA3SequenceOutput:
    frames: list[DA3FrameOutput]
    fps: float
    height: int
    width: int
    source_video: str
