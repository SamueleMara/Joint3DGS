from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from articulation.data.dataclasses import FeatureGraph
from articulation.geometry.robust import huber
from articulation.geometry.se3 import transform_points



def motion_fit_loss(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    w: torch.Tensor,
    T_part0: torch.Tensor,
    T_part1: torch.Tensor,
    huber_delta: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft-assignment rigid motion fitting.

    Args:
        xyz: [P,Tw,3]
        valid: [P,Tw]
        w: [P]
        T_part0/T_part1: [Tw-1,4,4], transform from anchor (t=0) to frame t
    """
    p, tw, _ = xyz.shape
    if tw < 2:
        raise ValueError("Need at least two frames for motion fit")
    if valid.shape != (p, tw):
        raise ValueError("valid must match xyz[:, :, 0]")
    if w.shape != (p,):
        raise ValueError("w must be [P]")
    if T_part0.shape != (tw - 1, 4, 4) or T_part1.shape != (tw - 1, 4, 4):
        raise ValueError("Transforms must be [Tw-1,4,4]")

    x0 = xyz[:, 0, :]
    v0 = valid[:, 0]

    pred0 = transform_points(T_part0, x0).transpose(0, 1)  # [P,Tw-1,3]
    pred1 = transform_points(T_part1, x0).transpose(0, 1)  # [P,Tw-1,3]

    xt = xyz[:, 1:, :]
    vt = valid[:, 1:]
    v = vt & v0.unsqueeze(1)

    denom = torch.linalg.norm(x0.unsqueeze(1) - xt, dim=-1).clamp_min(eps)
    r0 = torch.linalg.norm(pred0 - xt, dim=-1) / denom
    r1 = torch.linalg.norm(pred1 - xt, dim=-1) / denom

    l0 = huber(r0, delta=huber_delta)
    l1 = huber(r1, delta=huber_delta)

    loss = (1.0 - w).unsqueeze(1) * l0 + w.unsqueeze(1) * l1
    loss = torch.where(v, loss, torch.zeros_like(loss))

    denom_valid = v.float().sum().clamp_min(1.0)
    return loss.sum() / denom_valid



def feature_smoothness_loss(w: torch.Tensor, graph: FeatureGraph, eps: float = 1e-8) -> torch.Tensor:
    if w.ndim != 1:
        raise ValueError("w must be [P]")
    idx = graph.nn_idx
    a = graph.nn_weight
    if idx.shape[0] != w.shape[0]:
        raise ValueError("Graph and w size mismatch")

    wn = w[idx]
    diff = torch.abs(w.unsqueeze(1) - wn)
    return (a * diff).sum() / (a.sum() + eps)



def entropy_loss(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = w.clamp(eps, 1.0 - eps)
    ent = -(w * torch.log(w) + (1.0 - w) * torch.log(1.0 - w))
    return ent.mean()


def _inverse_transform_points_per_frame(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Apply inverse rigid transform per frame.

    Args:
        T: [Tw-1,4,4] forward transforms (canonical t=0 -> frame t)
        X: [P,Tw-1,3] observed points in frame t
    Returns:
        [P,Tw-1,3] mapped back to canonical frame
    """
    if T.ndim != 3 or T.shape[-2:] != (4, 4):
        raise ValueError("T must be [Tw-1,4,4]")
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError("X must be [P,Tw-1,3]")
    if X.shape[1] != T.shape[0]:
        raise ValueError("Temporal mismatch between T and X")

    r = T[:, :3, :3]  # [Tw-1,3,3]
    t = T[:, :3, 3]   # [Tw-1,3]
    xt = X.transpose(0, 1)  # [Tw-1,P,3]

    # Forward uses row vectors: y = x @ R^T + t, so inverse is x = (y - t) @ R.
    x_can = (xt - t.unsqueeze(1)) @ r
    return x_can.transpose(0, 1)


def _masked_temporal_variance(x: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Masked temporal variance as mean squared distance from temporal mean."""
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("x must be [P,Tw,3]")
    if valid.shape != x.shape[:2]:
        raise ValueError("valid must match x[:2]")

    m = valid.float()
    n = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * m.unsqueeze(-1)).sum(dim=1) / n
    sqdist = torch.sum((x - mean.unsqueeze(1)) ** 2, dim=-1)
    return (sqdist * m).sum(dim=1) / n.squeeze(1).clamp_min(eps)


def rigidity_consistency_loss(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    w: torch.Tensor,
    T_part0: torch.Tensor,
    T_part1: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Geometry-driven assignment from canonical-frame stationarity."""
    p, tw, _ = xyz.shape
    if tw < 2:
        return xyz.new_tensor(0.0)
    if valid.shape != (p, tw):
        raise ValueError("valid must match xyz[:, :, 0]")
    if w.shape != (p,):
        raise ValueError("w must be [P]")
    if T_part0.shape != (tw - 1, 4, 4) or T_part1.shape != (tw - 1, 4, 4):
        raise ValueError("Transforms must be [Tw-1,4,4]")

    x0 = xyz[:, :1, :]
    xt = xyz[:, 1:, :]
    x_can0 = torch.cat([x0, _inverse_transform_points_per_frame(T_part0, xt)], dim=1)
    x_can1 = torch.cat([x0, _inverse_transform_points_per_frame(T_part1, xt)], dim=1)

    var0 = _masked_temporal_variance(x_can0, valid, eps=eps)
    var1 = _masked_temporal_variance(x_can1, valid, eps=eps)

    loss_per_point = (1.0 - w) * var0 + w * var1
    return loss_per_point.sum()



def _sample_pair_indices(num_points: int, num_pairs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    i = torch.randint(0, num_points, (num_pairs,), device=device)
    j = torch.randint(0, num_points, (num_pairs,), device=device)
    neq = i != j
    if not torch.all(neq):
        j = torch.where(neq, j, (j + 1) % num_points)
    return i, j



def pairwise_rigidity_loss(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    w: torch.Tensor,
    num_pairs: int = 4096,
    margin: float = 0.01,
    lambda_sep: float = 1.0,
    huber_delta: float = 1.0,
    min_overlap: int = 2,
) -> torch.Tensor:
    p, tw, _ = xyz.shape
    if tw < 2 or p < 2:
        return xyz.new_tensor(0.0)

    i, j = _sample_pair_indices(p, min(num_pairs, p * (p - 1)), xyz.device)

    xi = xyz[i]
    xj = xyz[j]
    vi = valid[i]
    vj = valid[j]
    vp = vi & vj

    overlap = vp.sum(dim=1)
    keep = overlap >= int(min_overlap)
    if keep.sum() == 0:
        return xyz.new_tensor(0.0)

    xi = xi[keep]
    xj = xj[keep]
    vp = vp[keep]
    wi = w[i][keep]
    wj = w[j][keep]

    d = torch.linalg.norm(xi - xj, dim=-1)
    d0 = d[:, :1]
    delta = torch.abs(d - d0)

    same = wi * wj + (1.0 - wi) * (1.0 - wj)
    diff = 1.0 - same

    rigid_term = huber(delta, delta=huber_delta)
    sep_term = F.relu(margin - delta)

    mask = vp.float()
    rigid_mean = (rigid_term * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    sep_mean = (sep_term * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    loss = same * rigid_mean + lambda_sep * diff * sep_mean
    return loss.mean()



def cog_rigidity_loss(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    w: torch.Tensor,
    eps: float = 1e-6,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    p, tw, _ = xyz.shape
    if tw < 2:
        return xyz.new_tensor(0.0)

    vw = valid.float()
    w1 = w.unsqueeze(1) * vw
    w0 = (1.0 - w).unsqueeze(1) * vw

    sum_w0 = w0.sum(dim=0).clamp_min(eps)
    sum_w1 = w1.sum(dim=0).clamp_min(eps)

    c0 = (w0.unsqueeze(-1) * xyz).sum(dim=0) / sum_w0.unsqueeze(-1)
    c1 = (w1.unsqueeze(-1) * xyz).sum(dim=0) / sum_w1.unsqueeze(-1)

    r0 = torch.linalg.norm(xyz - c0.unsqueeze(0), dim=-1)
    r1 = torch.linalg.norm(xyz - c1.unsqueeze(0), dim=-1)

    r0_ref = r0[:, :1]
    r1_ref = r1[:, :1]

    d0 = torch.abs(r0 - r0_ref) / (r0_ref + eps)
    d1 = torch.abs(r1 - r1_ref) / (r1_ref + eps)

    l0 = huber(d0, delta=huber_delta)
    l1 = huber(d1, delta=huber_delta)

    loss = ((1.0 - w).unsqueeze(1) * l0 + w.unsqueeze(1) * l1) * vw
    return loss.sum() / vw.sum().clamp_min(1.0)



def balance_loss(w: torch.Tensor) -> torch.Tensor:
    return (w.mean() - 0.5) ** 2


@dataclass
class SegmentationLossWeights:
    lambda_motion: float = 200.0
    lambda_smooth: float = 10.0
    lambda_rigid: float = 0.01
    lambda_pair: float = 2.0
    lambda_cog: float = 1.0
    lambda_balance: float = 0.0



def total_segmentation_loss(
    l_motion: torch.Tensor,
    l_smooth: torch.Tensor,
    l_rigid: torch.Tensor,
    l_pair: torch.Tensor,
    l_cog: torch.Tensor,
    l_balance: torch.Tensor,
    weights: SegmentationLossWeights,
    use_pair: bool,
    use_cog: bool,
    use_balance: bool,
) -> torch.Tensor:
    loss = (
        weights.lambda_motion * l_motion
        + weights.lambda_smooth * l_smooth
        + weights.lambda_rigid * l_rigid
    )
    if use_pair:
        loss = loss + weights.lambda_pair * l_pair
    if use_cog:
        loss = loss + weights.lambda_cog * l_cog
    if use_balance:
        loss = loss + weights.lambda_balance * l_balance
    return loss
