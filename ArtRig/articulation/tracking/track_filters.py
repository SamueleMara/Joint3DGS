from __future__ import annotations

from articulation.data.dataclasses import TrackBatch
from articulation.preprocess.filtering import filter_tracks



def apply_default_track_filters(tracks: TrackBatch, cfg: dict) -> TrackBatch:
    return filter_tracks(
        tracks,
        min_valid_ratio=float(cfg.get("min_valid_ratio", 0.7)),
        smoothness_zscore=float(cfg.get("smoothness_zscore", 3.0)),
        min_confidence=cfg.get("min_confidence"),
    )
