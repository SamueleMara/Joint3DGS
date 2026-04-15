from __future__ import annotations

from articulation.data.dataclasses import RGBDSequence, TrackBatch
from articulation.preprocess.filtering import filter_tracks
from articulation.preprocess.lifting import lift_tracks_to_3d
from articulation.tracking.tracker_wrapper import TrackerWrapper


def run_tracking_and_lift(
    sequence: RGBDSequence,
    tracker_cfg: dict,
    min_valid_ratio: float = 0.7,
    max_depth: float | None = None,
) -> TrackBatch:
    if tracker_cfg.get("backend") == "alltracker" and tracker_cfg.get("seqlen") is None:
        tracker_cfg = dict(tracker_cfg)
        tracker_cfg["seqlen"] = sequence.T
    tracker = TrackerWrapper.from_config(tracker_cfg)
    tracks = tracker.track(sequence)
    xyz, valid = lift_tracks_to_3d(tracks.xy, sequence.depth, sequence.K, max_depth=max_depth)
    tracks = TrackBatch(
        xy=tracks.xy,
        xyz=xyz,
        valid=valid,
        anchor_frame=tracks.anchor_frame,
        point_ids=tracks.point_ids,
        feature=tracks.feature,
        confidence=tracks.confidence,
    )
    tracks = filter_tracks(tracks, min_valid_ratio=min_valid_ratio)
    return tracks
