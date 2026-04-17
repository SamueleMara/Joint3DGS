from __future__ import annotations

import torch


def w2c_3x4_to_4x4(matrix: torch.Tensor) -> torch.Tensor:
    out = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
    out[:3, :4] = matrix
    return out


def invert_w2c(e_w2c: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(e_w2c)


def w2c_to_c2w(e_w2c: torch.Tensor) -> torch.Tensor:
    return invert_w2c(e_w2c)


def c2w_to_w2c(e_c2w: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(e_c2w)


def compose_extrinsics(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1)


def from_homogeneous(points_h: torch.Tensor) -> torch.Tensor:
    w = torch.clamp(points_h[..., -1:], min=1.0e-8)
    return points_h[..., :-1] / w


def normalize_intrinsics(k: torch.Tensor, width: int, height: int) -> torch.Tensor:
    out = k.clone()
    out[0] /= width
    out[1] /= height
    return out
