from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from articulation.data.dataclasses import TrackBatch



def load_tracks_npz(path: str | Path, anchor_frame: int = 0) -> TrackBatch:
    data = np.load(path, allow_pickle=True)

    if "xy" not in data:
        raise KeyError("tracks npz must contain 'xy' with shape [P,T,2]")

    xy = torch.from_numpy(data["xy"]).float()
    p, t, _ = xy.shape

    xyz = torch.from_numpy(data["xyz"]).float() if "xyz" in data else torch.zeros((p, t, 3), dtype=torch.float32)
    valid = torch.from_numpy(data["valid"]).bool() if "valid" in data else torch.ones((p, t), dtype=torch.bool)

    if "point_ids" in data:
        point_ids = torch.from_numpy(data["point_ids"]).long()
    else:
        point_ids = torch.arange(p, dtype=torch.long)

    if "feature" in data:
        feature = torch.from_numpy(data["feature"]).float()
    else:
        feature = torch.zeros((p, 1), dtype=torch.float32)

    confidence: Optional[torch.Tensor]
    if "confidence" in data:
        confidence = torch.from_numpy(data["confidence"]).float()
    else:
        confidence = None

    if "anchor_frame" in data:
        anchor_frame = int(data["anchor_frame"])

    obs_count = torch.from_numpy(data["obs_count"]).float() if "obs_count" in data else None
    multiview_error = torch.from_numpy(data["multiview_error"]).float() if "multiview_error" in data else None
    meta = {}
    if "meta" in data:
        raw = data["meta"]
        if isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
            raw = raw.item()
        if isinstance(raw, dict):
            meta = raw

    return TrackBatch(
        xy=xy,
        xyz=xyz,
        valid=valid,
        anchor_frame=anchor_frame,
        point_ids=point_ids,
        feature=feature,
        confidence=confidence,
        obs_count=obs_count,
        multiview_error=multiview_error,
        meta=meta,
    )



def save_tracks_npz(track_batch: TrackBatch, path: str | Path) -> None:
    kwargs = {
        "xy": track_batch.xy.detach().cpu().numpy(),
        "xyz": track_batch.xyz.detach().cpu().numpy(),
        "valid": track_batch.valid.detach().cpu().numpy(),
        "anchor_frame": np.array(track_batch.anchor_frame),
        "point_ids": track_batch.point_ids.detach().cpu().numpy(),
        "feature": track_batch.feature.detach().cpu().numpy(),
    }
    if track_batch.confidence is not None:
        kwargs["confidence"] = track_batch.confidence.detach().cpu().numpy()
    if track_batch.obs_count is not None:
        kwargs["obs_count"] = track_batch.obs_count.detach().cpu().numpy()
    if track_batch.multiview_error is not None:
        kwargs["multiview_error"] = track_batch.multiview_error.detach().cpu().numpy()
    if track_batch.meta:
        kwargs["meta"] = np.array(track_batch.meta, dtype=object)
    np.savez_compressed(path, **kwargs)
