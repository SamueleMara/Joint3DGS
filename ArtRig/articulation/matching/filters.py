from __future__ import annotations

import torch

from articulation.data.dataclasses import KeypointBatch, MatchBatch
from articulation.preprocess.lifting import lift_pixel_to_world


def _sample_mask(mask: torch.Tensor | None, xy: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones((xy.shape[0],), dtype=torch.bool, device=xy.device)

    if mask.ndim == 3:
        if mask.shape[0] != 1:
            raise ValueError("mask must be [1,H,W] or [H,W]")
        m = mask[0]
    elif mask.ndim == 2:
        m = mask
    else:
        raise ValueError("mask must be [1,H,W] or [H,W]")

    h, w = int(m.shape[0]), int(m.shape[1])
    u = xy[:, 0].round().long().clamp(0, w - 1)
    v = xy[:, 1].round().long().clamp(0, h - 1)
    inb = (xy[:, 0] >= 0) & (xy[:, 0] <= (w - 1)) & (xy[:, 1] >= 0) & (xy[:, 1] <= (h - 1))
    return inb & (m[v, u] > 0.5)


def filter_same_time_multiview_matches(
    match: MatchBatch,
    frame_a: KeypointBatch,
    frame_b: KeypointBatch,
    K_a: torch.Tensor,
    K_b: torch.Tensor,
    T_cw_a: torch.Tensor,
    T_cw_b: torch.Tensor,
    threshold_same_time: float,
    fg_mask_a: torch.Tensor | None = None,
    fg_mask_b: torch.Tensor | None = None,
) -> MatchBatch:
    m = match.idx_a.shape[0]
    if m == 0:
        return match

    idx_a = match.idx_a.long()
    idx_b = match.idx_b.long()

    xy_a = frame_a.xy[idx_a]
    xy_b = frame_b.xy[idx_b]
    z_a = frame_a.depth[idx_a]
    z_b = frame_b.depth[idx_b]

    valid_depth = frame_a.valid[idx_a] & frame_b.valid[idx_b]
    valid_fg = _sample_mask(fg_mask_a, xy_a) & _sample_mask(fg_mask_b, xy_b)

    X_a, ok_a = lift_pixel_to_world(xy_a, z_a, K_a, T_cw_a)
    X_b, ok_b = lift_pixel_to_world(xy_b, z_b, K_b, T_cw_b)
    dist = torch.linalg.norm(X_a - X_b, dim=1)

    keep = (
        valid_depth
        & valid_fg
        & ok_a
        & ok_b
        & torch.isfinite(dist)
        & (dist <= float(threshold_same_time))
    )

    out = MatchBatch(
        idx_a=idx_a[keep],
        idx_b=idx_b[keep],
        confidence=match.confidence[keep],
        pair_type=match.pair_type,
        meta={
            **dict(match.meta),
            "filtered": True,
            "num_in": int(m),
            "num_out": int(keep.sum().item()),
            "threshold_same_time": float(threshold_same_time),
            "mean_3d_error": float(dist[keep].mean().item()) if bool(keep.any()) else float("inf"),
        },
    )
    return out


def _cycle_consistency_keep(
    idx_a: torch.Tensor,
    idx_b: torch.Tensor,
    reverse_match: MatchBatch | None,
) -> torch.Tensor:
    if reverse_match is None or reverse_match.idx_a.numel() == 0:
        return torch.ones_like(idx_a, dtype=torch.bool)

    rev_map = {}
    for a, b in zip(reverse_match.idx_a.tolist(), reverse_match.idx_b.tolist()):
        if a not in rev_map:
            rev_map[a] = b

    out = torch.zeros((idx_a.shape[0],), dtype=torch.bool, device=idx_a.device)
    for i, (a, b) in enumerate(zip(idx_a.tolist(), idx_b.tolist())):
        out[i] = (b in rev_map) and (rev_map[b] == a)
    return out


def filter_cross_time_matches(
    match: MatchBatch,
    frame_a: KeypointBatch,
    frame_b: KeypointBatch,
    min_confidence: float,
    max_pixel_jump: float,
    max_depth_jump: float | None = None,
    require_cycle_consistency: bool = False,
    reverse_match: MatchBatch | None = None,
) -> MatchBatch:
    m = match.idx_a.shape[0]
    if m == 0:
        return match

    idx_a = match.idx_a.long()
    idx_b = match.idx_b.long()

    conf_keep = match.confidence >= float(min_confidence)

    xy_a = frame_a.xy[idx_a]
    xy_b = frame_b.xy[idx_b]
    jump = torch.linalg.norm(xy_b - xy_a, dim=1)
    jump_keep = torch.isfinite(jump) & (jump <= float(max_pixel_jump))

    valid_depth = frame_a.valid[idx_a] & frame_b.valid[idx_b]
    if max_depth_jump is not None:
        dz = torch.abs(frame_b.depth[idx_b] - frame_a.depth[idx_a])
        depth_keep = valid_depth & torch.isfinite(dz) & (dz <= float(max_depth_jump))
    else:
        depth_keep = valid_depth

    if require_cycle_consistency:
        cyc_keep = _cycle_consistency_keep(idx_a, idx_b, reverse_match=reverse_match)
    else:
        cyc_keep = torch.ones_like(conf_keep)

    keep = conf_keep & jump_keep & depth_keep & cyc_keep

    out = MatchBatch(
        idx_a=idx_a[keep],
        idx_b=idx_b[keep],
        confidence=match.confidence[keep],
        pair_type=match.pair_type,
        meta={
            **dict(match.meta),
            "filtered": True,
            "num_in": int(m),
            "num_out": int(keep.sum().item()),
            "min_confidence": float(min_confidence),
            "max_pixel_jump": float(max_pixel_jump),
            "max_depth_jump": None if max_depth_jump is None else float(max_depth_jump),
            "cycle_consistency": bool(require_cycle_consistency),
            "mean_pixel_jump": float(jump[keep].mean().item()) if bool(keep.any()) else float("inf"),
        },
    )
    return out
