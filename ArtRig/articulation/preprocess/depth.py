from __future__ import annotations

import torch
import torch.nn.functional as F



def sample_depth_at_xy(depth: torch.Tensor, xy: torch.Tensor, mode: str = "bilinear") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-point depth.

    Args:
        depth: [T,1,H,W]
        xy: [P,T,2] in pixel coordinates (u,v)
        mode: "bilinear" or "nearest"

    Returns:
        z: [P,T]
        valid: [P,T] in-bounds and positive depth
    """
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"depth must be [T,1,H,W], got {depth.shape}")
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be [P,T,2], got {xy.shape}")

    p, t, _ = xy.shape
    td, _, h, w = depth.shape
    if t != td:
        raise ValueError(f"xy T={t} must match depth T={td}")
    if h <= 1 or w <= 1:
        raise ValueError("depth H and W must be > 1")

    x = xy[..., 0]
    y = xy[..., 1]
    in_bounds = (x >= 0) & (x <= (w - 1)) & (y >= 0) & (y <= (h - 1))

    gx = (x / (w - 1)) * 2.0 - 1.0
    gy = (y / (h - 1)) * 2.0 - 1.0

    grid = torch.stack([gx.transpose(0, 1), gy.transpose(0, 1)], dim=-1).unsqueeze(2)  # [T,P,1,2]
    sampled = F.grid_sample(depth, grid, mode=mode, padding_mode="zeros", align_corners=True)
    z = sampled.squeeze(1).squeeze(-1).transpose(0, 1).contiguous()  # [P,T]

    valid = in_bounds & torch.isfinite(z) & (z > 0)
    return z, valid
