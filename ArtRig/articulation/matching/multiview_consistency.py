from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MultiViewConsistencyWeights:
    alpha_world: float = 1.0
    alpha_reproj: float = 0.5
    alpha_feat: float = 0.2


@dataclass
class MultiViewConsistencyResult:
    world_error: float
    reproj_error: float
    feature_error: float
    score: float
    support: int


def _project_world_to_pixel(X_world: torch.Tensor, K: torch.Tensor, T_cw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if X_world.ndim != 1 or X_world.shape[0] != 3:
        raise ValueError("X_world must be [3]")
    if K.shape != (3, 3):
        raise ValueError("K must be [3,3]")
    if T_cw.shape != (4, 4):
        raise ValueError("T_cw must be [4,4]")

    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    x_cam = R @ X_world + t
    z = x_cam[2]
    valid = torch.isfinite(z) & (z > 1e-8)
    if not bool(valid):
        return torch.zeros((2,), dtype=X_world.dtype, device=X_world.device), torch.tensor(False, device=X_world.device)

    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = fx * (x_cam[0] / z) + cx
    v = fy * (x_cam[1] / z) + cy
    uv = torch.stack([u, v], dim=0)
    ok = torch.isfinite(uv).all()
    return uv, ok


def world_agreement_error(world_points: torch.Tensor) -> torch.Tensor:
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError("world_points must be [N,3]")
    n = world_points.shape[0]
    if n <= 1:
        return world_points.new_tensor(0.0)

    d = torch.cdist(world_points, world_points, p=2.0)
    tri = torch.triu_indices(n, n, offset=1, device=world_points.device)
    vals = d[tri[0], tri[1]]
    if vals.numel() == 0:
        return world_points.new_tensor(0.0)
    return vals.mean()


def reprojection_error(
    fused_world: torch.Tensor,
    observed_xy: torch.Tensor,
    K_all: torch.Tensor,
    T_cw_all: torch.Tensor,
) -> torch.Tensor:
    if observed_xy.ndim != 2 or observed_xy.shape[1] != 2:
        raise ValueError("observed_xy must be [N,2]")
    n = observed_xy.shape[0]
    if K_all.shape != (n, 3, 3):
        raise ValueError("K_all must be [N,3,3]")
    if T_cw_all.shape != (n, 4, 4):
        raise ValueError("T_cw_all must be [N,4,4]")

    errs = []
    for i in range(n):
        uv_hat, ok = _project_world_to_pixel(fused_world, K_all[i], T_cw_all[i])
        if bool(ok):
            errs.append(torch.linalg.norm(uv_hat - observed_xy[i]))
    if not errs:
        return fused_world.new_tensor(float("inf"))
    return torch.stack(errs).mean()


def descriptor_coherence_error(desc: torch.Tensor) -> torch.Tensor:
    if desc.ndim != 2:
        raise ValueError("desc must be [N,C]")
    n = desc.shape[0]
    if n <= 1:
        return desc.new_tensor(0.0)

    d = F.normalize(desc.float(), dim=1)
    sim = d @ d.transpose(0, 1)
    tri = torch.triu_indices(n, n, offset=1, device=desc.device)
    vals = 1.0 - sim[tri[0], tri[1]]
    return vals.mean() if vals.numel() > 0 else desc.new_tensor(0.0)


def multiview_consistency_score(
    world_points: torch.Tensor,
    observed_xy: torch.Tensor,
    desc: torch.Tensor,
    K_all: torch.Tensor,
    T_cw_all: torch.Tensor,
    weights: MultiViewConsistencyWeights,
) -> MultiViewConsistencyResult:
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError("world_points must be [N,3]")

    support = int(world_points.shape[0])
    if support == 0:
        return MultiViewConsistencyResult(
            world_error=float("inf"),
            reproj_error=float("inf"),
            feature_error=float("inf"),
            score=float("inf"),
            support=0,
        )

    world_e = world_agreement_error(world_points)
    fused_world = torch.median(world_points, dim=0).values
    reproj_e = reprojection_error(fused_world, observed_xy, K_all, T_cw_all)
    feat_e = descriptor_coherence_error(desc)

    score = (
        float(weights.alpha_world) * world_e
        + float(weights.alpha_reproj) * reproj_e
        + float(weights.alpha_feat) * feat_e
    )

    return MultiViewConsistencyResult(
        world_error=float(world_e.detach().cpu()),
        reproj_error=float(reproj_e.detach().cpu()),
        feature_error=float(feat_e.detach().cpu()),
        score=float(score.detach().cpu()),
        support=support,
    )
