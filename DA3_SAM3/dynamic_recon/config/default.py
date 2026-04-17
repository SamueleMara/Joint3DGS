from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from dynamic_recon.config.schema import PipelineConfig
from dynamic_recon.io_utils import load_yaml, merge_dicts


def load_config(path: str | Path | None = None) -> PipelineConfig:
    config = PipelineConfig()
    if path is None:
        return config
    raw = load_yaml(Path(path))
    merged = merge_dicts(asdict(config), raw)
    return PipelineConfig(
        device=merged["device"],
        dtype=merged["dtype"],
        seed=merged["seed"],
        video=config.video.__class__(**merged["video"]),
        da3=config.da3.__class__(**merged["da3"]),
        geometry=config.geometry.__class__(**merged["geometry"]),
        sam3=config.sam3.__class__(**merged["sam3"]),
        fusion=config.fusion.__class__(**merged["fusion"]),
        pose_init=config.pose_init.__class__(**merged["pose_init"]),
        pose_refine=config.pose_refine.__class__(**merged["pose_refine"]),
        pipeline=config.pipeline.__class__(**merged["pipeline"]),
    )
