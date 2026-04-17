from __future__ import annotations

from dataclasses import dataclass

from articulation.data.dataclasses import RGBDSequence, TrackBatch
from articulation.data.io_tracks import load_tracks_npz
from articulation.tracking.alltracker_backend import AllTrackerBackend
from articulation.tracking.base import BaseTracker
from articulation.tracking.cotracker_backend import CoTrackerBackend
from articulation.tracking.tapnet_backend import TapnetBackend


@dataclass
class PrecomputedTracker:
    """Tracker backend that loads tracks from disk."""

    tracks_npz: str
    anchor_frame: int = 0

    def track(self, sequence: RGBDSequence) -> TrackBatch:
        tracks = load_tracks_npz(self.tracks_npz, anchor_frame=self.anchor_frame)
        if tracks.T != sequence.T:
            raise ValueError(
                f"Track length T={tracks.T} does not match sequence T={sequence.T}."
            )
        return tracks


class TrackerWrapper:
    """Stable facade to avoid hard-coding tracker dependencies in core logic."""

    def __init__(self, backend: BaseTracker):
        self.backend = backend

    @classmethod
    def from_config(cls, cfg: dict) -> "TrackerWrapper":
        backend = cfg.get("backend", "precomputed")

        if backend == "precomputed":
            tracks_npz = cfg.get("tracks_npz")
            if not tracks_npz:
                raise ValueError("'tracks_npz' is required for backend='precomputed'")
            anchor = int(cfg.get("anchor_frame", 0))
            impl: BaseTracker = PrecomputedTracker(tracks_npz=tracks_npz, anchor_frame=anchor)
            return cls(impl)

        if backend == "cotracker":
            impl = CoTrackerBackend(
                checkpoint=cfg.get("checkpoint"),
                grid_size=int(cfg.get("grid_size", 10)),
                grid_query_frame=int(cfg.get("grid_query_frame", 0)),
                backward_tracking=bool(cfg.get("backward_tracking", False)),
                offline=bool(cfg.get("offline", True)),
                v2=bool(cfg.get("v2", False)),
                window_len=int(cfg.get("window_len", 60)),
                device=str(cfg.get("device", "cuda")),
            )
            return cls(impl)

        if backend == "alltracker":
            repo_path = cfg.get("repo_path")
            checkpoint = cfg.get("checkpoint")
            if not repo_path or not checkpoint:
                raise ValueError("'repo_path' and 'checkpoint' required for backend='alltracker'")
            impl = AllTrackerBackend(
                repo_path=str(repo_path),
                checkpoint=str(checkpoint),
                seqlen=cfg.get("seqlen"),
                inference_iters=int(cfg.get("inference_iters", 4)),
                query_frame=int(cfg.get("query_frame", 0)),
                rate=int(cfg.get("rate", 4)),
                device=str(cfg.get("device", "cuda")),
            )
            return cls(impl)

        if backend == "tapnet":
            callable_name = cfg.get("backend_callable")
            if not callable_name:
                raise ValueError("'backend_callable' is required for backend='tapnet'")
            impl = TapnetBackend(
                backend_callable=str(callable_name),
                repo_path=cfg.get("repo_path"),
                device=str(cfg.get("device", "cuda")),
            )
            return cls(impl)

        raise ValueError(f"Unsupported tracker backend: {backend}")

    def track(self, sequence: RGBDSequence) -> TrackBatch:
        return self.backend.track(sequence)
