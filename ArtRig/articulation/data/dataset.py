from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from articulation.data.dataclasses import RGBDSequence, TrackBatch
from articulation.data.io_rgbd import load_rgbd_sequence_from_folders, load_rgbd_sequence_npz
from articulation.data.io_tracks import load_tracks_npz


@dataclass
class SequenceSample:
    sequence: RGBDSequence
    tracks: Optional[TrackBatch]


class ArticulationDataset:
    """Simple dataset utility for single-sequence experiments.

    Supports either:
    - RGB/depth(/mask) folders + intrinsics
    - prepacked RGBD sequence `.npz`
    Optionally attaches tracks from a `.npz` file.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    def load(self) -> SequenceSample:
        sequence_cfg = self.cfg.get("sequence", {})
        tracks_cfg = self.cfg.get("tracks", {})

        if "npz" in sequence_cfg:
            sequence = load_rgbd_sequence_npz(sequence_cfg["npz"])
        else:
            required = ["rgb_dir", "depth_dir", "intrinsics"]
            missing = [k for k in required if k not in sequence_cfg]
            if missing:
                raise ValueError(f"Missing sequence config keys: {missing}")
            sequence = load_rgbd_sequence_from_folders(
                rgb_dir=sequence_cfg["rgb_dir"],
                depth_dir=sequence_cfg["depth_dir"],
                intrinsics=sequence_cfg["intrinsics"],
                mask_dir=sequence_cfg.get("mask_dir"),
                depth_scale=float(sequence_cfg.get("depth_scale", 1.0)),
                meta={"source": "folders"},
            )

        tracks: Optional[TrackBatch] = None
        tracks_npz = tracks_cfg.get("npz")
        if tracks_npz:
            tracks = load_tracks_npz(Path(tracks_npz), anchor_frame=int(tracks_cfg.get("anchor_frame", 0)))

        return SequenceSample(sequence=sequence, tracks=tracks)
