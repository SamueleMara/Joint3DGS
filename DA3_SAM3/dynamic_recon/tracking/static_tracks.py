from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dynamic_recon.geometry.projection import backproject, project


@dataclass(slots=True)
class StaticTrackResult:
    query_frame_idx: int
    points_xy: np.ndarray
    tracks_xy: np.ndarray
    valid: np.ndarray


def track_static_pixels(query_frame_idx: int, query_points_xy: np.ndarray, da3_seq: object, poses_refined: list[torch.Tensor]) -> StaticTrackResult:
    num_frames = len(da3_seq.frames)
    tracks = np.zeros((num_frames, len(query_points_xy), 2), dtype=np.float32)
    valid = np.zeros((num_frames, len(query_points_xy)), dtype=bool)
    frame = da3_seq.frames[query_frame_idx]
    world = backproject(frame.depth, frame.intrinsics, poses_refined[query_frame_idx])
    for t in range(num_frames):
        uv, z = project(world, da3_seq.frames[t].intrinsics, poses_refined[t])
        for idx, (x, y) in enumerate(query_points_xy.astype(int)):
            tracks[t, idx] = uv[y, x].detach().cpu().numpy()
            valid[t, idx] = bool(z[y, x] > 0)
    return StaticTrackResult(query_frame_idx=query_frame_idx, points_xy=query_points_xy, tracks_xy=tracks, valid=valid)
