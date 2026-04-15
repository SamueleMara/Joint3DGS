from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable
import sys

import torch

from articulation.data.dataclasses import RGBDSequence, TrackBatch


@dataclass
class TapnetBackend:
    backend_callable: str
    repo_path: str | None = None
    device: str = "cuda"

    def _resolve_callable(self) -> Callable[..., Any]:
        if ":" not in self.backend_callable:
            raise ValueError("backend_callable must be 'module:callable'")
        mod_name, fn_name = self.backend_callable.split(":", 1)
        if self.repo_path:
            p = str(Path(self.repo_path).resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        mod = import_module(mod_name)
        fn = getattr(mod, fn_name)
        if not callable(fn):
            raise TypeError(f"TAPNet backend target is not callable: {self.backend_callable}")
        return fn

    def track(self, sequence: RGBDSequence) -> TrackBatch:
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        video = sequence.rgb.to(device).unsqueeze(0)
        fg_mask = sequence.fg_mask.to(device) if sequence.fg_mask is not None else None

        fn = self._resolve_callable()
        tracks, visibility = fn(video=video, fg_mask=fg_mask)

        xy = tracks[0].permute(1, 0, 2).contiguous().float()
        valid = visibility[0].permute(1, 0).contiguous().bool()
        p, t, _ = xy.shape

        return TrackBatch(
            xy=xy,
            xyz=torch.zeros((p, t, 3), dtype=xy.dtype, device=xy.device),
            valid=valid,
            anchor_frame=0,
            point_ids=torch.arange(p, device=xy.device),
            feature=torch.zeros((p, 1), dtype=xy.dtype, device=xy.device),
            confidence=None,
        )
