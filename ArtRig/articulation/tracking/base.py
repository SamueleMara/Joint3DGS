from __future__ import annotations

from typing import Protocol

from articulation.data.dataclasses import RGBDSequence, TrackBatch


class BaseTracker(Protocol):
    def track(self, sequence: RGBDSequence) -> TrackBatch:
        ...
