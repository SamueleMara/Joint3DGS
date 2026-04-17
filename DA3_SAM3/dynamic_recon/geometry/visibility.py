from __future__ import annotations

import torch


def inside_image_mask(uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return (uv[..., 0] >= 0) & (uv[..., 0] <= width - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= height - 1)


def positive_depth_mask(z_cam: torch.Tensor, min_depth: float = 1.0e-6) -> torch.Tensor:
    return z_cam > min_depth


def occlusion_mask(pred_depth: torch.Tensor, observed_depth: torch.Tensor, tol_abs: float, tol_rel: float) -> torch.Tensor:
    threshold = tol_abs + tol_rel * torch.abs(pred_depth)
    return observed_depth + threshold < pred_depth


def forward_backward_consistency(uv_fw: torch.Tensor, uv_bw: torch.Tensor, thresh_px: float = 1.0) -> torch.Tensor:
    cycle = torch.linalg.norm(uv_fw - uv_bw, dim=-1)
    return torch.exp(-cycle / max(thresh_px, 1.0e-6))
