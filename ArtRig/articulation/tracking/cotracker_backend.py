from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from articulation.data.dataclasses import RGBDSequence, TrackBatch


@dataclass
class CoTrackerBackend:
    checkpoint: str | None = None
    grid_size: int = 10
    grid_query_frame: int = 0
    backward_tracking: bool = False
    offline: bool = True
    v2: bool = False
    window_len: int = 60
    device: str = "cuda"

    def _build_model(self):
        if self.checkpoint:
            if not Path(self.checkpoint).exists():
                raise FileNotFoundError(f"CoTracker checkpoint not found: {self.checkpoint}")
            from cotracker.predictor import CoTrackerPredictor

            return CoTrackerPredictor(
                checkpoint=self.checkpoint,
                offline=self.offline,
                v2=self.v2,
                window_len=self.window_len,
            )

        # fall back to torch.hub, requires internet/download if weights missing
        return torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

    def track(self, sequence: RGBDSequence) -> TrackBatch:
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = self._build_model().to(device)
        model.eval()

        video = sequence.rgb.to(device)
        video = video.unsqueeze(0)  # [1,T,3,H,W]
        segm_mask = sequence.fg_mask.to(device) if sequence.fg_mask is not None else None

        pred_tracks, pred_vis = model(
            video,
            grid_size=self.grid_size,
            grid_query_frame=self.grid_query_frame,
            backward_tracking=self.backward_tracking,
            segm_mask=segm_mask,
        )

        # pred_tracks: [1,T,N,2] -> [N,T,2]
        xy = pred_tracks[0].permute(1, 0, 2).contiguous().float()
        valid = pred_vis[0].permute(1, 0).contiguous().bool()
        p, t, _ = xy.shape

        return TrackBatch(
            xy=xy,
            xyz=torch.zeros((p, t, 3), dtype=xy.dtype, device=xy.device),
            valid=valid,
            anchor_frame=self.grid_query_frame,
            point_ids=torch.arange(p, device=xy.device),
            feature=torch.zeros((p, 1), dtype=xy.dtype, device=xy.device),
            confidence=None,
        )
