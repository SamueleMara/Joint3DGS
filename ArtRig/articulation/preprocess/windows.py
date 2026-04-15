from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSpec:
    start: int
    end: int
    anchor: int

    @property
    def length(self) -> int:
        return self.end - self.start



def build_sliding_windows(num_frames: int, size: int, stride: int) -> list[WindowSpec]:
    if num_frames <= 0:
        raise ValueError("num_frames must be > 0")
    if size <= 1:
        raise ValueError("size must be > 1")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    windows: list[WindowSpec] = []
    start = 0
    while start < num_frames:
        end = min(start + size, num_frames)
        if end - start >= 2:
            windows.append(WindowSpec(start=start, end=end, anchor=start))
        if end == num_frames:
            break
        start += stride

    return windows
