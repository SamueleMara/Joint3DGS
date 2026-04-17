from __future__ import annotations

import torch

from articulation.preprocess.depth import sample_depth_at_xy


def lift_pixel_to_world(
    xy: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    T_cw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift pixels with depth into world coordinates.

    Args:
        xy: [N,2] pixel coordinates (u,v)
        depth: [N] depth in meters
        K: [3,3] camera intrinsics
        T_cw: [4,4] world->camera transform
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be [N,2], got {xy.shape}")
    if depth.ndim != 1 or depth.shape[0] != xy.shape[0]:
        raise ValueError(f"depth must be [N] aligned with xy, got {depth.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be [3,3], got {K.shape}")
    if T_cw.shape != (4, 4):
        raise ValueError(f"T_cw must be [4,4], got {T_cw.shape}")

    xy = xy.float()
    z = depth.float()
    K = K.float().to(xy.device)
    T_cw = T_cw.float().to(xy.device)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = xy[:, 0]
    v = xy[:, 1]
    x_cam = ((u - cx) / fx) * z
    y_cam = ((v - cy) / fy) * z
    p_cam = torch.stack([x_cam, y_cam, z], dim=1)

    r_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    # Row-vector convention:
    # x_cam = X_world @ R_cw^T + t_cw  -> X_world = (x_cam - t_cw) @ R_cw
    X_world = (p_cam - t_cw.unsqueeze(0)) @ r_cw

    valid = torch.isfinite(X_world).all(dim=1) & torch.isfinite(z) & (z > 0)
    X_world = torch.where(valid.unsqueeze(1), X_world, torch.zeros_like(X_world))
    return X_world, valid


def project_world_to_pixel(
    X_world: torch.Tensor,
    K: torch.Tensor,
    T_cw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world points to image pixels.

    Args:
        X_world: [N,3]
        K: [3,3]
        T_cw: [4,4] world->camera
    """
    if X_world.ndim != 2 or X_world.shape[1] != 3:
        raise ValueError(f"X_world must be [N,3], got {X_world.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"K must be [3,3], got {K.shape}")
    if T_cw.shape != (4, 4):
        raise ValueError(f"T_cw must be [4,4], got {T_cw.shape}")

    X_world = X_world.float()
    K = K.float().to(X_world.device)
    T_cw = T_cw.float().to(X_world.device)

    r_cw = T_cw[:3, :3]
    t_cw = T_cw[:3, 3]
    x_cam = X_world @ r_cw.transpose(0, 1) + t_cw.unsqueeze(0)

    z = x_cam[:, 2]
    valid = torch.isfinite(z) & (z > 1e-8)
    z_safe = torch.where(valid, z, torch.ones_like(z))

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (x_cam[:, 0] / z_safe) + cx
    v = fy * (x_cam[:, 1] / z_safe) + cy
    uv = torch.stack([u, v], dim=1)
    valid = valid & torch.isfinite(uv).all(dim=1)
    uv = torch.where(valid.unsqueeze(1), uv, torch.zeros_like(uv))
    return uv, valid


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
