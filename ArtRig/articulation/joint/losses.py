from __future__ import annotations

from dataclasses import dataclass

import torch



def trajectory_reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    err = torch.linalg.norm(pred - target, dim=-1)
    mask = valid.float()
    if weights is not None:
        err = err * weights[:, None]
        mask = mask * weights[:, None]
    return (err * mask).sum() / mask.sum().clamp_min(1.0)



def temporal_smoothness_loss(state: torch.Tensor) -> torch.Tensor:
    if state.numel() < 3:
        return state.new_tensor(0.0)
    acc = state[2:] - 2.0 * state[1:-1] + state[:-2]
    return (acc ** 2).mean()



def axis_prior_loss(axis_dir: torch.Tensor, axis_prior: torch.Tensor) -> torch.Tensor:
    d = axis_dir / torch.linalg.norm(axis_dir).clamp_min(1e-8)
    p = axis_prior / torch.linalg.norm(axis_prior).clamp_min(1e-8)
    cos = torch.abs(torch.dot(d, p))
    return 1.0 - cos



def axis_point_reg(axis_point: torch.Tensor, axis_point_prior: torch.Tensor) -> torch.Tensor:
    return ((axis_point - axis_point_prior) ** 2).mean()



def pitch_reg(pitch: torch.Tensor) -> torch.Tensor:
    return pitch ** 2


@dataclass
class JointLossWeights:
    lambda_fit: float = 1.0
    lambda_temporal: float = 0.1
    lambda_axis: float = 0.05
    lambda_axis_point: float = 0.0
    lambda_pitch: float = 0.01



def total_joint_loss(
    l_fit: torch.Tensor,
    l_temporal: torch.Tensor,
    l_axis: torch.Tensor,
    l_axis_point: torch.Tensor,
    l_pitch: torch.Tensor,
    w: JointLossWeights,
) -> torch.Tensor:
    return (
        w.lambda_fit * l_fit
        + w.lambda_temporal * l_temporal
        + w.lambda_axis * l_axis
        + w.lambda_axis_point * l_axis_point
        + w.lambda_pitch * l_pitch
    )
