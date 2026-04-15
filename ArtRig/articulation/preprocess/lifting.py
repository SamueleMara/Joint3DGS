from __future__ import annotations

import torch

from articulation.preprocess.depth import sample_depth_at_xy



def _expand_intrinsics(K: torch.Tensor, t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if K.ndim == 2:
        if K.shape != (3, 3):
            raise ValueError(f"K must be [3,3] or [T,3,3], got {K.shape}")
        k = K.unsqueeze(0).expand(t, -1, -1)
    elif K.ndim == 3:
        if K.shape[0] != t or K.shape[1:] != (3, 3):
            raise ValueError(f"Per-frame K must be [T,3,3] with T={t}, got {K.shape}")
        k = K
    else:
        raise ValueError(f"K must be [3,3] or [T,3,3], got {K.shape}")
    return k.to(device=device, dtype=dtype)



def lift_tracks_to_3d(
    xy: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    max_depth: float | None = None,
    mode: str = "bilinear",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift 2D tracks into camera-frame 3D points.

    Args:
        xy: [P,T,2] pixel coordinates (u,v)
        depth: [T,1,H,W] depth in meters
        K: [3,3] or [T,3,3] camera intrinsics
        max_depth: optional clipping threshold
        mode: depth sampling mode

    Returns:
        xyz: [P,T,3]
        valid: [P,T]
    """
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError(f"xy must be [P,T,2], got {xy.shape}")

    p, t, _ = xy.shape
    z, valid = sample_depth_at_xy(depth, xy, mode=mode)
    k = _expand_intrinsics(K, t=t, device=xy.device, dtype=xy.dtype)

    fx = k[:, 0, 0].unsqueeze(0)
    fy = k[:, 1, 1].unsqueeze(0)
    cx = k[:, 0, 2].unsqueeze(0)
    cy = k[:, 1, 2].unsqueeze(0)

    u = xy[..., 0]
    v = xy[..., 1]

    x = ((u - cx) / fx) * z
    y = ((v - cy) / fy) * z

    xyz = torch.stack([x, y, z], dim=-1)

    if max_depth is not None:
        valid = valid & (z <= float(max_depth))

    xyz = torch.where(valid.unsqueeze(-1), xyz, torch.zeros_like(xyz))

    if xyz.shape != (p, t, 3):
        raise RuntimeError("Internal shape error while lifting tracks")

    return xyz, valid
