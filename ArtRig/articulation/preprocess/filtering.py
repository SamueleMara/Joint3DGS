from __future__ import annotations

from typing import Optional

import torch

from articulation.data.dataclasses import TrackBatch



def valid_ratio_filter(valid: torch.Tensor, min_valid_ratio: float) -> torch.Tensor:
    if valid.ndim != 2:
        raise ValueError("valid must be [P,T]")
    ratio = valid.float().mean(dim=1)
    return ratio >= float(min_valid_ratio)



def trajectory_smoothness_filter(
    xy: torch.Tensor,
    valid: torch.Tensor,
    zscore_thresh: float = 3.0,
) -> torch.Tensor:
    """Filter tracks with unusually large per-step acceleration."""
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy must be [P,T,2]")
    if valid.shape != xy.shape[:2]:
        raise ValueError("valid must match [P,T]")
    if xy.shape[1] < 3:
        return torch.ones(xy.shape[0], dtype=torch.bool, device=xy.device)

    vel = xy[:, 1:] - xy[:, :-1]
    acc = vel[:, 1:] - vel[:, :-1]
    valid_acc = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]

    acc_norm = torch.linalg.norm(acc, dim=-1)
    valid_f = valid_acc.float()
    count = valid_acc.sum(dim=1)
    count_f = count.clamp_min(1).float()

    mean = (acc_norm * valid_f).sum(dim=1) / count_f
    var = (((acc_norm - mean[:, None]) ** 2) * valid_f).sum(dim=1) / count_f
    std = torch.sqrt(var).clamp_min(1e-6)
    peak = torch.where(valid_acc, acc_norm, torch.zeros_like(acc_norm)).amax(dim=1)
    score = torch.where(count > 0, (peak - mean) / std, torch.zeros_like(mean))
    return score <= float(zscore_thresh)



def confidence_filter(confidence: Optional[torch.Tensor], min_confidence: float) -> torch.Tensor:
    if confidence is None:
        return torch.tensor([], dtype=torch.bool)
    if confidence.ndim != 1:
        raise ValueError("confidence must be [P]")
    return confidence >= float(min_confidence)



def filter_tracks(
    tracks: TrackBatch,
    min_valid_ratio: float = 0.7,
    smoothness_zscore: float = 3.0,
    min_confidence: Optional[float] = None,
) -> TrackBatch:
    p = tracks.xy.shape[0]
    keep = valid_ratio_filter(tracks.valid, min_valid_ratio)
    keep = keep & trajectory_smoothness_filter(tracks.xy, tracks.valid, zscore_thresh=smoothness_zscore)

    if min_confidence is not None and tracks.confidence is not None:
        keep = keep & confidence_filter(tracks.confidence, min_confidence)

    if keep.shape[0] != p:
        raise RuntimeError("Internal filtering mask shape error")

    confidence = tracks.confidence[keep] if tracks.confidence is not None else None

    return TrackBatch(
        xy=tracks.xy[keep],
        xyz=tracks.xyz[keep],
        valid=tracks.valid[keep],
        anchor_frame=tracks.anchor_frame,
        point_ids=tracks.point_ids[keep],
        feature=tracks.feature[keep],
        confidence=confidence,
    )
