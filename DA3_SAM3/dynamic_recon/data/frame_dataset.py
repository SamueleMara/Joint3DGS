from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FrameSample:
    frame_index: int
    rgb: np.ndarray
