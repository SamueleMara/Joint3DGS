from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class VideoMetadata:
    frame_paths: list[Path]
    timestamps: list[float]
    fps: float
    frame_size: tuple[int, int]


@dataclass(slots=True)
class DA3Result:
    depth: np.ndarray
    confidence: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SAM3Track:
    object_id: int
    label: str
    frame_indices: list[int]
    scores: list[float]


@dataclass(slots=True)
class SAM3Result:
    masks: np.ndarray
    object_ids: list[int]
    labels: dict[int, str]
    tracks: list[SAM3Track]


@dataclass(slots=True)
class FusionResult:
    dynamic_score: np.ndarray
    label_map: np.ndarray
    binary_dynamic: np.ndarray
    track_stats: dict[str, Any]
