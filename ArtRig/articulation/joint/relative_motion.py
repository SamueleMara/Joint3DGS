from __future__ import annotations

import torch

from articulation.data.dataclasses import RelativeMotionResult, SegmentationResult, TrackBatch



def _kabsch_t_to_anchor(x_anchor: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
    """Estimate rigid transform mapping frame t points -> anchor points."""
    ca = x_anchor.mean(dim=0)
    ct = x_t.mean(dim=0)
    xa = x_anchor - ca
    xt = x_t - ct

    h = xt.transpose(0, 1) @ xa
    u, _, v = torch.linalg.svd(h)
    r = v.transpose(0, 1) @ u.transpose(0, 1)
    if torch.det(r) < 0:
        v[:, -1] *= -1
        r = v.transpose(0, 1) @ u.transpose(0, 1)

    t = ca - (r @ ct)

    T = torch.eye(4, dtype=x_anchor.dtype, device=x_anchor.device)
    T[:3, :3] = r
    T[:3, 3] = t
    return T



def choose_reference_part(tracks: TrackBatch, seg: SegmentationResult) -> tuple[int, int]:
    labels = seg.point_labels.long()
    xyz = tracks.xyz
    valid = tracks.valid
    p0 = labels == 0
    p1 = labels == 1

    def motion(mask: torch.Tensor) -> float:
        if mask.sum() == 0:
            return float("inf")
        x = xyz[mask]
        v = valid[mask]
        d = torch.linalg.norm(x[:, 1:] - x[:, :-1], dim=-1)
        d = torch.where(v[:, 1:] & v[:, :-1], d, torch.zeros_like(d))
        denom = (v[:, 1:] & v[:, :-1]).float().sum().clamp_min(1.0)
        return float(d.sum() / denom)

    m0 = motion(p0)
    m1 = motion(p1)
    return (0, 1) if m0 <= m1 else (1, 0)



def compute_relative_motion(
    xyz_part_ref: torch.Tensor,
    xyz_part_mov: torch.Tensor,
    valid_ref: torch.Tensor,
    valid_mov: torch.Tensor,
    weights: torch.Tensor | None = None,
    reference_part: int = 0,
    moving_part: int = 1,
) -> RelativeMotionResult:
    if xyz_part_ref.ndim != 3 or xyz_part_mov.ndim != 3:
        raise ValueError("xyz_part_ref and xyz_part_mov must be [P,T,3]")

    pr, t, _ = xyz_part_ref.shape
    pm = xyz_part_mov.shape[0]
    if valid_ref.shape != (pr, t) or valid_mov.shape != (pm, t):
        raise ValueError("Invalid valid mask shapes")

    out = torch.zeros_like(xyz_part_mov)
    Tinv = torch.eye(4, device=xyz_part_ref.device, dtype=xyz_part_ref.dtype).unsqueeze(0).repeat(t, 1, 1)

    x_ref0 = xyz_part_ref[:, 0, :]
    v_ref0 = valid_ref[:, 0]

    for ti in range(t):
        vr = v_ref0 & valid_ref[:, ti]
        if vr.sum() < 3:
            out[:, ti, :] = xyz_part_mov[:, ti, :]
            continue

        T_t = _kabsch_t_to_anchor(x_anchor=x_ref0[vr], x_t=xyz_part_ref[vr, ti, :])
        Tinv[ti] = T_t

        R = T_t[:3, :3]
        tt = T_t[:3, 3]
        out[:, ti, :] = xyz_part_mov[:, ti, :] @ R.transpose(0, 1) + tt

    canonical = out[:, 0, :]
    if weights is None:
        weights = torch.ones(pm, device=xyz_part_mov.device, dtype=xyz_part_mov.dtype)

    return RelativeMotionResult(
        reference_part=reference_part,
        moving_part=moving_part,
        canonical_points=canonical,
        moving_points_rel=out,
        valid=valid_mov,
        weights=weights,
        ref_transform_inv=Tinv,
        diagnostics={"num_ref_points": int(pr), "num_mov_points": int(pm)},
    )
