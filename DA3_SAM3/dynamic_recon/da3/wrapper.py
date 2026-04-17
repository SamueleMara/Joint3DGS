from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from dynamic_recon.config.schema import DA3Config
from dynamic_recon.da3.adapters import build_adapter, load_video_for_da3
from dynamic_recon.da3.types import DA3SequenceOutput


def load_da3_model(cfg: DA3Config) -> Any:
    return build_adapter(cfg)


def run_da3_on_frames(frames: list[np.ndarray], cfg: DA3Config, source_video: str = "") -> DA3SequenceOutput:
    adapter = load_da3_model(cfg)
    return adapter.infer_sequence(frames, cfg, source_video=source_video)


def run_da3_on_video(video_path: str | Path, cfg: DA3Config) -> DA3SequenceOutput:
    frames, metadata = load_video_for_da3(video_path, cfg)
    output = run_da3_on_frames(frames, cfg, source_video=str(video_path))
    output.fps = float(metadata["fps"])
    return output
